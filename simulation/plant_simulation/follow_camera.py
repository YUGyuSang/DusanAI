"""
follow_camera: make the Gazebo GUI camera follow the robot (third-person view behind / above it, like
the reference video) - and keep the view the user sets with the mouse.

    ros2 run plant_simulation follow_camera
    ros2 run plant_simulation follow_camera --target spot --offset=-2.5,0,1.6 --pos-gain 0.02
    ros2 run plant_simulation follow_camera --no-mouse          # the old fixed follow (mouse pulled back)

Started by bringup.launch.py with demo:=true (or follow_camera:=true).

1. waits (gz CLI) until Gazebo runs, the robot exists (/model/<target>/odometry or
   /world/<world>/model/<target>/joint_state is advertised) and the GUI is up (CameraTracking
   advertises /gui/camera/pose); gives up quietly after --timeout s, e.g. with gui:=false;
2. publishes a gz.msgs.CameraTrack on /gui/track: track_mode FOLLOW, follow_target <target>,
   follow_offset (robot frame), follow_pgain = LOOK-AT gain, track_pgain = POSITION gain (gz-gui
   CameraTracking hands the two message gains over crossed: see follow_camera_logic) - understood by
   gz-gui 8 (Harmonic) ... 10 (Jetty);
3. confirms it on /gui/currently_tracked and re-sends if needed; if that never confirms and the GUI
   offers the older /gui/follow service (Harmonic), calls /gui/follow + /gui/follow/offset instead.

Then (needs rclpy, which the ROS launch provides) it stays up as the camera "director"
(follow_camera_logic.CameraDirector):

4. mouse: the GUI pulls a camera moved with the mouse back to the follow offset, so the director
   watches /gui/camera/pose against the robot pose (/odom): a camera move the follow law cannot
   explain is the mouse -> the camera position is held (still looking at the robot) while the user
   orbits / zooms / pans, and 0.4 s after the mouse stops the camera pose where the user left it,
   relative to the robot, becomes the view.  It is sent with the follow law's trail behind the
   walking robot (v / k, k learnt from the camera), re-sent when the robot's speed changes, so the
   camera stays at that view while the robot walks and while it stands;
5. fall guard: the GUI applies the offset with the robot's full attitude (a robot on its back would
   pull the camera under the floor), so while |roll| or |pitch| > --tilt-hold or the robot reports
   a fall the camera is held (position gain ~0, offset levelled), and --release-hold s after the robot
   is upright it follows again - with the user's view (a camera moved meanwhile becomes the view);
6. keys in the Gazebo window (KeyPublisher /keyboard/keypress, bridged to ROS): 'c' camera stays
   where it is and keeps looking at the robot / 'c' again follows from the camera pose of that moment;
   'z' back to the default view.  Esc (gz-gui) switches following off; 'c', 'z' or right-click ->
   Follow on the robot switch it on again;
7. walls: the default view (not a view the user chose) is pulled in front of the building walls of
   world_layout.yaml (corridor, hall, partition), so the robot is not hidden at the corridor gauges or
   at the dock.

Transport: gz-transport python bindings when installed (gz.transport15 / 14 / 13: /gui/track,
/gui/camera/pose, /gui/currently_tracked without CLI start-up), else `gz topic -p` for sends
(~0.3-0.7 s each), the bridged ROS /gui/camera/pose (geometry_msgs/Pose) for the camera and a
`gz topic -e` poll of /gui/currently_tracked every 2.5 s ('c' / Esc states) or 10 s (following).
Without rclpy, or with --once, it exits after step 3 as before.

The program never fails the launch: every error is logged and it exits 0.
"""
import argparse
import collections
import math
import re
import shutil
import subprocess
import sys
import threading
import time

from plant_simulation.follow_camera_logic import (  # noqa: F401  (re-exported for tools / tests)
    CAMERA_DEFAULTS, DEFAULT_OFFSET, HOLD_GAIN, LOOK_GAIN, POS_GAIN, CameraDirector, FallGuard, TrackRequest,
    WallMap, describe_view, layout_walls, level_offset, parse_tracked, robot_down, tracked_from_msg)

FOLLOW_DEFAULTS = {
    "target": "spot",
    "world": "plant",
    "offset": DEFAULT_OFFSET,       # m in the robot frame: behind (-x) and above (+z)
    "look_gain": LOOK_GAIN,         # look-at slerp per rendered frame (CameraTrack.follow_pgain)
    "pos_gain": POS_GAIN,           # position per rendered frame (CameraTrack.track_pgain): ~0.8 s at 60 fps
    "timeout": 240.0,               # s waiting for Gazebo, the robot and the GUI
    "attempts": 6,                  # /gui/track sends before falling back to /gui/follow
    "poll": 2.0,                    # s between checks
    "fall_guard": True,
    "mouse": True,                  # keep a view set with the mouse
    "walls": True,                  # pull the default view in front of the building walls
    "odom_topic": "/odom",          # robot pose in the world frame (odom_tf)
    "tracked_poll": 2.5,            # s between `gz topic -e` polls of /gui/currently_tracked (no bindings;
                                    # 4x that while simply following)
}
GUARD_KEYS = ("tilt_hold_deg", "tilt_release_deg", "release_hold", "hold_gain", "hold_update", "hold_min_change")
LOCOMOTION_STATE_TOPIC = "/locomotion/state"
KEY_TOPIC = "/keyboard/keypress"
TRACK_TOPIC = "/gui/track"
TRACKED_TOPIC = "/gui/currently_tracked"
GUI_TOPIC = "/gui/camera/pose"


def parse_offset(text):
    """'-2.5,0,1.6' or '-2.5 0 1.6' -> (x, y, z)."""
    parts = [p for p in re.split(r"[,\s]+", str(text).strip()) if p]
    if len(parts) != 3:
        raise ValueError(f"offset needs 3 numbers, got {text!r}")
    return tuple(float(p) for p in parts)


def follow_request(target, offset, look=LOOK_GAIN, pos=POS_GAIN):
    """The engagement message: FOLLOW <target> with the offset and both gains."""
    return TrackRequest("FOLLOW", str(target), tuple(float(v) for v in offset), float(look), float(pos))


def track_message(target, offset, look=LOOK_GAIN, pos=POS_GAIN, mode="FOLLOW"):
    """Protobuf text of a gz.msgs.CameraTrack request (see TrackRequest)."""
    return TrackRequest(mode, str(target), tuple(float(v) for v in offset), float(look), float(pos)).text()


def track_command(gz, req):
    return [gz, "topic", "-t", TRACK_TOPIC, "-m", "gz.msgs.CameraTrack", "-p", req.text()]


def follow_service_commands(gz, target, offset, timeout_ms=3000):
    """Harmonic-only fallback: /gui/follow (StringMsg) then /gui/follow/offset (Vector3d)."""
    x, y, z = offset
    base = ["--reptype", "gz.msgs.Boolean", "--timeout", str(int(timeout_ms))]
    num = lambda v: f"{float(v):.6g}"  # noqa: E731
    return [
        [gz, "service", "-s", "/gui/follow", "--reqtype", "gz.msgs.StringMsg"] + base + ["--req", f'data: "{target}"'],
        [gz, "service", "-s", "/gui/follow/offset", "--reqtype", "gz.msgs.Vector3d"] + base
        + ["--req", f"x: {num(x)}, y: {num(y)}, z: {num(z)}"],
    ]


def robot_topics(target, world):
    return (f"/model/{target}/odometry", f"/world/{world}/model/{target}/joint_state")


def is_following(echo_text, target):
    """True if a /gui/currently_tracked sample says FOLLOW* on <target>."""
    m = re.search(r"track_mode:\s*(\w+)", echo_text or "")
    if not m or not m.group(1).upper().startswith("FOLLOW"):
        return False
    t = re.search(r'follow_target\s*\{[^}]*name:\s*"([^"]*)"', echo_text, re.S)
    return bool(t and t.group(1) == target)


def default_runner(cmd, timeout):
    """(returncode, stdout+stderr); (-1, reason) when the command cannot run / times out."""
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                           universal_newlines=True)
        return r.returncode, r.stdout or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or b""
        return -1, out.decode(errors="replace") if isinstance(out, bytes) else str(out)
    except OSError as exc:
        return -1, str(exc)


# ---------------------------------------------------------------- transports
class GzGuiLink:
    """/gui/track publisher + /gui/camera/pose and /gui/currently_tracked subscriber through the
    gz-transport python bindings (ImportError when they are not installed).  Callbacks run in
    gz-transport threads and only append to a deque; drain() hands the events to the main loop."""

    VERSIONS = ((15, 12), (14, 11), (13, 10))          # (transport, msgs): Jetty, Ionic, Harmonic

    def __init__(self, modules=None, clock=time.monotonic, subscribe=True):
        import importlib
        load = modules or importlib.import_module
        self.clock = clock
        self.events = collections.deque(maxlen=2000)
        self.tracked_seen = False
        self.cam_seen = False
        last = None
        for vt, vm in self.VERSIONS:
            try:
                tr = load(f"gz.transport{vt}")
                ct = load(f"gz.msgs{vm}.cameratrack_pb2")
                pose = load(f"gz.msgs{vm}.pose_pb2") if subscribe else None
            except ImportError as exc:
                last = exc
                continue
            self.CameraTrack = ct.CameraTrack
            self.node = tr.Node()
            self.pub = self.node.advertise(TRACK_TOPIC, ct.CameraTrack)
            self.version = (vt, vm)
            self.subscribed = []
            if subscribe:
                for typ, topic, cb in ((pose.Pose, GUI_TOPIC, self._on_cam),
                                       (ct.CameraTrack, TRACKED_TOPIC, self._on_tracked)):
                    try:
                        if self.node.subscribe(typ, topic, cb):
                            self.subscribed.append(topic)
                    except Exception:  # an odd binding version: publishing still works
                        pass
            return
        raise ImportError(f"no gz-transport python bindings ({last})")

    def publish(self, req):
        return bool(self.pub.publish(req.fill(self.CameraTrack(), self.CameraTrack)))

    def _on_cam(self, m):
        self.cam_seen = True
        p = m.position
        self.events.append(("cam", self.clock(), ((float(p.x), float(p.y), float(p.z)), "gz")))

    def _on_tracked(self, m):
        self.tracked_seen = True
        self.events.append(("tracked", self.clock(), tracked_from_msg(m)))

    def drain(self):
        out = []
        while self.events:
            out.append(self.events.popleft())
        return out

    def close(self):
        for topic in getattr(self, "subscribed", []):
            try:
                self.node.unsubscribe(topic)
            except Exception:
                pass


class CliTrackedPoller:
    """/gui/currently_tracked through `gz topic -e -n 1` every `period` s (no bindings).  threaded=True
    polls in a daemon thread (a poll takes 0.3-1 s and must not stall the camera loop); else maybe_poll()
    runs it in the caller (tests).  Events carry the time the poll STARTED, so a status sampled before
    one of our sends is recognised as old.  period_fn() (optional) gives the period of the moment: every
    poll starts the gz CLI (Ruby, ~0.3-1 s CPU), so it polls often only while it matters."""

    def __init__(self, gz, runner, clock, period=2.5, threaded=True, period_fn=None):
        self.gz, self.run, self.clock, self.period = gz, runner, clock, float(period)
        self.period_fn = period_fn
        self.events = collections.deque(maxlen=50)
        self.threaded = bool(threaded)
        self._stop = threading.Event()
        self._next = -math.inf
        self._thread = None

    def poll_once(self):
        t = self.clock()
        rc, out = self.run([self.gz, "topic", "-e", "-n", "1", "-t", TRACKED_TOPIC], 4.0)
        if rc == 0 and (out or "").strip():
            self.events.append(("tracked", t, parse_tracked(out)))

    def current_period(self):
        try:
            return float(self.period_fn()) if self.period_fn is not None else self.period
        except Exception:
            return self.period

    def maybe_poll(self, now):
        if not self.threaded and now >= self._next:
            self._next = now + self.current_period()
            self.poll_once()

    def start(self):
        if not self.threaded:
            return self

        def loop():
            while not self._stop.is_set():
                t0 = self.clock()
                try:
                    self.poll_once()
                except Exception:
                    pass
                while not self._stop.is_set() and self.clock() - t0 < self.current_period():
                    self._stop.wait(0.25)
        self._thread = threading.Thread(target=loop, name="follow_camera_tracked", daemon=True)
        self._thread.start()
        return self

    def drain(self):
        out = []
        while self.events:
            out.append(self.events.popleft())
        return out

    def stop(self):
        self._stop.set()


class RosInputs:
    """rclpy subscriptions of the camera director, as timestamped events (arrival time, wall clock):
    ('loco', t, dict) /locomotion/state, ('robot', t, (p, q)) robot world pose (/odom),
    ('key', t, qt_code) /keyboard/keypress (Gazebo KeyPublisher, bridged), ('cam', t, (pos, 'ros'))
    /gui/camera/pose (bridged; used when the gz-transport bindings are missing)."""

    def __init__(self, ros_args=(), odom_topic="/odom", clock=time.monotonic):
        import json

        import rclpy
        from geometry_msgs.msg import Pose
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Int32, String
        self._rclpy, self._json, self.clock = rclpy, json, clock
        self._own = not rclpy.ok()
        if self._own:
            rclpy.init(args=["follow_camera"] + list(ros_args))
        self.node = Node("follow_camera")
        self.events = []
        self.count = 0
        be = lambda depth: QoSProfile(depth=depth, reliability=ReliabilityPolicy.BEST_EFFORT)  # noqa: E731
        # best effort matches a reliable and a best-effort publisher alike
        self.node.create_subscription(String, LOCOMOTION_STATE_TOPIC, self._on_state, be(5))
        self.node.create_subscription(Odometry, odom_topic, self._on_odom, be(20))
        self.node.create_subscription(Pose, GUI_TOPIC, self._on_cam, be(20))
        self.node.create_subscription(Int32, KEY_TOPIC, self._on_key,
                                      QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

    def _add(self, kind, data):
        self.events.append((kind, self.clock(), data))

    def _on_state(self, msg):
        self.count += 1
        try:
            d = self._json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if isinstance(d, dict):
            self._add("loco", d)

    def _on_odom(self, msg):
        self.count += 1
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        self._add("robot", ((float(p.x), float(p.y), float(p.z)), (float(o.w), float(o.x), float(o.y), float(o.z))))

    def _on_cam(self, msg):
        self.count += 1
        p = msg.position
        self._add("cam", ((float(p.x), float(p.y), float(p.z)), "ros"))

    def _on_key(self, msg):
        self.count += 1
        self._add("key", int(msg.data))

    def ok(self):
        return bool(self._rclpy.ok())

    def poll(self, timeout):
        """Wait up to timeout s for messages, then take whatever else is ready; return the events."""
        n0 = self.count
        self._rclpy.spin_once(self.node, timeout_sec=float(timeout))
        k = 0
        while self.count != n0 and k < 200:
            n0 = self.count
            self._rclpy.spin_once(self.node, timeout_sec=0.0)
            k += 1
        evs, self.events = self.events, []
        return evs

    def close(self):
        try:
            self.node.destroy_node()
        finally:
            if self._own and self._rclpy.ok():
                self._rclpy.shutdown()


def load_walls(layout_path=None, margin=0.3):
    """WallMap of the layout's building shells, or None (log the reason via the exception)."""
    from dusan_common.layout import Layout
    lay = Layout.load(layout_path or None)
    segs = layout_walls(lay)
    return WallMap(segs, margin=margin) if segs else None


# ---------------------------------------------------------------- the program
class FollowCamera:
    """Engagement (gz CLI) + the camera director loop, with injectable runner / clock / sleep / inputs."""

    def __init__(self, gz="gz", target="spot", world="plant", offset=DEFAULT_OFFSET, pos_gain=POS_GAIN,
                 timeout=240.0, attempts=6, poll=2.0, runner=default_runner, clock=time.monotonic,
                 sleep=time.sleep, log=print, inputs_factory=None, cam_cfg=None, link_factory=None, walls=None,
                 tracked_poll=2.5, threaded=True):
        self.gz, self.target, self.world = gz, target, world
        self.offset = tuple(float(v) for v in offset)
        self.cam_cfg = {k: v for k, v in dict(cam_cfg or {}).items() if k in CAMERA_DEFAULTS}
        self.cam_cfg["pos_gain"] = float(pos_gain)
        self.look_gain = float(self.cam_cfg.get("look_gain", LOOK_GAIN))
        self.pos_gain = float(pos_gain)
        self.timeout, self.attempts, self.poll = float(timeout), int(attempts), float(poll)
        self.run, self.clock, self.sleep, self.log = runner, clock, sleep, log
        # inputs_factory() -> object with poll(timeout) -> [(kind, t, data)], ok(), close() (RosInputs).
        # None: exit once the follow mode is engaged.
        self.inputs_factory = inputs_factory
        # link_factory() -> GzGuiLink-like (publish(req), drain(), tracked_seen); None / failing: gz CLI
        self.link_factory = link_factory
        self.walls = walls
        self.tracked_poll, self.threaded = float(tracked_poll), bool(threaded)
        self.link = None
        self.poller = None
        self.director = None
        self._last_gz_cam = -math.inf
        self._cli_fail_logged = False

    # ------------------------------------------------------------------ engagement (gz CLI)
    def topics(self):
        rc, out = self.run([self.gz, "topic", "-l"], 10.0)
        return set(out.split()) if rc == 0 else set()

    def services(self):
        rc, out = self.run([self.gz, "service", "-l"], 10.0)
        return set(out.split()) if rc == 0 else set()

    def wait_ready(self):
        """'ready' | 'no_gui' | 'no_robot' | 'no_gz' after waiting up to timeout."""
        t0 = self.clock()
        robot = robot_topics(self.target, self.world)
        seen_gz = False
        while True:
            tp = self.topics()
            seen_gz = seen_gz or bool(tp)
            have_robot = any(t in tp for t in robot)
            have_gui = GUI_TOPIC in tp or TRACKED_TOPIC in tp
            if have_robot and have_gui:
                return "ready"
            if self.clock() - t0 >= self.timeout:
                if not seen_gz:
                    return "no_gz"
                return "no_robot" if not have_robot else "no_gui"
            self.sleep(self.poll)

    def confirm(self):
        rc, out = self.run([self.gz, "topic", "-e", "-n", "1", "-t", TRACKED_TOPIC], 4.0)
        return is_following(out, self.target)

    def engage(self):
        """Send /gui/track until confirmed; fall back to /gui/follow.  Returns the method used or ''."""
        req = follow_request(self.target, self.offset, self.look_gain, self.pos_gain)
        for i in range(max(1, self.attempts)):
            rc, out = self.run(track_command(self.gz, req), 10.0)
            if rc != 0:
                self.log(f"[follow_camera] gz topic -p {TRACK_TOPIC} failed ({rc}): {out.strip()[:200]}")
            self.sleep(1.0)
            if self.confirm():
                return "track"
            self.sleep(self.poll)
        if "/gui/follow" in self.services():
            ok = True
            for cmd in follow_service_commands(self.gz, self.target, self.offset):
                rc, out = self.run(cmd, 10.0)
                ok = ok and rc == 0 and "false" not in out.lower()
                self.sleep(0.5)
            if ok:
                return "follow_service"
        return ""

    def main(self):
        status = self.wait_ready()
        if status == "no_gz":
            self.log(f"[follow_camera] no Gazebo topics after {self.timeout:.0f} s (is gz sim running in this "
                     "GZ_PARTITION?) - giving up")
            return 0
        if status == "no_robot":
            self.log(f"[follow_camera] model '{self.target}' not found after {self.timeout:.0f} s - giving up")
            return 0
        if status == "no_gui":
            self.log("[follow_camera] no Gazebo GUI (gui:=false?) - nothing to follow with")
            return 0
        method = self.engage()
        if method:
            self.log(f"[follow_camera] Gazebo GUI camera follows '{self.target}' ({describe_view(self.offset)}, "
                     f"position gain {self.pos_gain:g}, look-at gain {self.look_gain:g}, "
                     f"via {'/gui/track' if method == 'track' else '/gui/follow'})")
        else:
            self.log("[follow_camera] could not confirm the follow camera - right-click the robot in the Entity "
                     "tree and choose 'Follow' instead")
        if method == "track" and self.inputs_factory is not None:
            self.follow_loop()
        return 0

    # ------------------------------------------------------------------ sending
    def send(self, req):
        """-> 'gz' (bindings, fast) | 'cli' (blocking gz topic -p returned) | '' (failed)."""
        if self.link is not None:
            try:
                if self.link.publish(req):
                    return "gz"
            except Exception as exc:
                self.log(f"[follow_camera] gz-transport publish failed ({type(exc).__name__}: {exc}) - "
                         "using gz topic -p")
                self._drop_link()
        rc, out = self.run(track_command(self.gz, req), 10.0)
        if rc != 0:
            if not self._cli_fail_logged:
                self.log(f"[follow_camera] gz topic -p {TRACK_TOPIC} failed ({rc}): {out.strip()[:200]}")
                self._cli_fail_logged = True
            return ""
        self._cli_fail_logged = False
        return "cli"

    def _drop_link(self):
        if self.link is not None:
            try:
                self.link.close()
            except Exception:
                pass
        self.link = None
        if self.director is not None:
            self.director.fast = False
        if self.poller is None:
            self.poller = self._make_poller()

    def _poll_period(self):
        """Often while 'c' / Esc / right-click decide what the GUI camera does (TRIPOD, PASSIVE), seldom
        while following (Esc then only stops our messages from mattering: USE_LAST cannot switch it on)."""
        d = self.director
        if d is None or d.state in ("TRIPOD", "PASSIVE"):
            return self.tracked_poll
        return 4.0 * self.tracked_poll

    def _make_poller(self):
        return CliTrackedPoller(self.gz, self.run, self.clock, self.tracked_poll, threaded=self.threaded,
                                period_fn=self._poll_period).start()

    # ------------------------------------------------------------------ the director loop
    def _dispatch(self, d, ev):
        kind, t, data = ev
        if kind == "robot":
            d.on_robot(t, data[0], data[1])
        elif kind == "cam":
            pos, src = data
            if src == "gz":
                self._last_gz_cam = t
            elif t - self._last_gz_cam < 0.5:
                return                      # the bindings deliver the camera: skip the bridged copy
            d.on_camera(t, pos)
        elif kind == "tracked":
            d.on_tracked(t, data)
        elif kind == "loco":
            d.on_loco(t, data)
        elif kind == "key":
            d.on_key(t, data)

    def follow_loop(self, max_polls=None):
        """Camera director (module doc, steps 4-7) until ROS shuts down."""
        try:
            inputs = self.inputs_factory()
        except Exception as exc:  # no rclpy / no ROS environment
            self.log(f"[follow_camera] camera control off ({type(exc).__name__}: {exc}) - the camera follows with "
                     "the fixed offset, no mouse view / keys / fall guard")
            return
        via = "gz topic -p"
        if self.link_factory is not None:
            try:
                self.link = self.link_factory()
                via = "gz-transport python" + (" %d" % self.link.version[0] if hasattr(self.link, "version") else "")
            except Exception:
                self.link = None
        if self.link is None:
            self.poller = self._make_poller()
        d = CameraDirector(self.target, self.offset, self.send, log=self.log, clock=self.clock, cfg=self.cam_cfg,
                           walls=self.walls, t0=self.clock(), fast=self.link is not None)
        self.director = d
        c = d.cfg
        parts = []
        parts.append("mouse in the 3D view sets the view (kept)" if c["adjust"] else "mouse view off (--no-mouse)")
        parts.append("Gazebo window keys: c = camera stays / follows again, z = default view")
        parts.append(f"fall guard {'on' if c['fall_guard'] else 'off'}")
        parts.append(f"walls {'on' if d.walls else 'off'}")
        self.log(f"[follow_camera] camera control on: {'; '.join(parts)} (via {via})")
        polls = 0
        try:
            while inputs.ok() and (max_polls is None or polls < max_polls):
                polls += 1
                evs = list(inputs.poll(0.02) or [])
                if self.link is not None:
                    evs += self.link.drain()
                if self.poller is not None:
                    self.poller.maybe_poll(self.clock())
                    evs += self.poller.drain()
                evs.sort(key=lambda e: e[1])
                for ev in evs:
                    self._dispatch(d, ev)
                d.tick(self.clock())
                if self.link is not None and d.echo_misses >= 2 and not getattr(self.link, "tracked_seen", True):
                    self.log("[follow_camera] the gz-transport bindings get no answer from the GUI (other Gazebo "
                             "version?) - using the gz CLI")
                    self._drop_link()
        except KeyboardInterrupt:
            pass
        except Exception as exc:  # never fail the launch
            self.log(f"[follow_camera] camera control stopped: {type(exc).__name__}: {exc}")
        finally:
            if self.poller is not None:
                self.poller.stop()
            if self.link is not None:
                try:
                    self.link.close()
                except Exception:
                    pass
            try:
                inputs.close()
            except Exception:
                pass
            s = d.stats
            self.log(f"[follow_camera] camera control ended: {s['adopted']} view(s) kept, {s['reverted']} ignored "
                     f"camera nudge(s), {d.guard.holds} fall hold(s), {s['sent']} /gui/track message(s)")


def build_parser():
    d = FOLLOW_DEFAULTS
    c = CAMERA_DEFAULTS
    ap = argparse.ArgumentParser(prog="follow_camera", description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", default=d["target"], help="model to follow (default %(default)s)")
    ap.add_argument("--world", default=d["world"], help="world name (default %(default)s)")
    ap.add_argument("--offset", default=",".join(f"{v:g}" for v in d["offset"]),
                    help="default camera offset x,y,z in the robot frame, m (default %(default)s); "
                         "use --offset=-2,0,1")
    ap.add_argument("--pos-gain", "--pgain", dest="pos_gain", type=float, default=d["pos_gain"],
                    help="position gain per rendered frame (default %(default)s)")
    ap.add_argument("--look-gain", type=float, default=d["look_gain"],
                    help="look-at gain per rendered frame (default %(default)s)")
    ap.add_argument("--timeout", type=float, default=d["timeout"], help="s to wait for gz / robot / GUI")
    ap.add_argument("--attempts", type=int, default=d["attempts"])
    ap.add_argument("--gz", default="", help="gz executable ('' = from PATH)")
    ap.add_argument("--once", action="store_true",
                    help="exit once the camera follows (no mouse view, keys or fall guard)")
    ap.add_argument("--no-mouse", dest="mouse", action="store_false", default=d["mouse"],
                    help="do not keep a view set with the mouse (the GUI pulls the camera back)")
    ap.add_argument("--no-walls", dest="walls", action="store_false", default=d["walls"],
                    help="do not pull the default view in front of the building walls")
    ap.add_argument("--layout", default="", help="world_layout.yaml for the walls ('' = installed)")
    ap.add_argument("--odom-topic", default=d["odom_topic"], help="robot world pose (default %(default)s)")
    ap.add_argument("--no-fall-guard", dest="fall_guard", action="store_false", default=d["fall_guard"],
                    help="no camera hold while the robot is down")
    ap.add_argument("--tilt-hold", type=float, default=c["tilt_hold_deg"],
                    help="deg roll / pitch above which the camera is held (default %(default)s)")
    ap.add_argument("--tilt-release", type=float, default=c["tilt_release_deg"],
                    help="deg below which an upright robot is followed again (default %(default)s)")
    ap.add_argument("--release-hold", type=float, default=c["release_hold"],
                    help="s upright before following again (default %(default)s)")
    ap.add_argument("--hold-gain", "--hold-pgain", dest="hold_gain", type=float, default=c["hold_gain"],
                    help="position gain while the camera is held (default %(default)s, must be > 1e-5)")
    return ap


def main(argv=None, inputs_factory=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ros_args = []
    if "--ros-args" in argv:                       # launch_ros Node(...) appends ROS arguments
        ros_args = argv[argv.index("--ros-args"):]
        argv = argv[:argv.index("--ros-args")]
    try:
        args = build_parser().parse_args(argv)
        offset = parse_offset(args.offset)
    except (SystemExit, ValueError) as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            return 0
        print(f"[follow_camera] bad arguments ({exc}) - not following", flush=True)
        return 0
    gz = args.gz or shutil.which("gz")
    if not gz:
        print("[follow_camera] the gz command line tool is not on PATH - not following", flush=True)
        return 0

    def log(msg):
        print(msg, flush=True)

    if not args.once and inputs_factory is None:
        def inputs_factory():
            return RosInputs(ros_args, odom_topic=args.odom_topic)
    cam_cfg = {"look_gain": args.look_gain, "adjust": args.mouse, "fall_guard": args.fall_guard,
               "tilt_hold_deg": args.tilt_hold, "tilt_release_deg": args.tilt_release,
               "release_hold": args.release_hold, "hold_gain": max(args.hold_gain, 1.1e-5)}
    walls = None
    if args.walls and not args.once:
        try:
            walls = load_walls(args.layout or None)
        except Exception as exc:  # no dusan_common / bad layout: just no walls
            log(f"[follow_camera] building walls not loaded ({type(exc).__name__}: {exc}) - default view not "
                "pulled in front of walls")
    try:
        return FollowCamera(gz, args.target, args.world, offset, args.pos_gain, args.timeout, args.attempts,
                            log=log, inputs_factory=None if args.once else inputs_factory, cam_cfg=cam_cfg,
                            link_factory=GzGuiLink, walls=walls).main()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # never fail the launch
        log(f"[follow_camera] unexpected error {type(exc).__name__}: {exc} - not following")
        return 0


if __name__ == "__main__":
    sys.exit(main())
