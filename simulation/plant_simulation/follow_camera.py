"""
follow_camera: make the Gazebo GUI camera follow the robot (third-person view
behind / above it, like the reference video).

    ros2 run plant_simulation follow_camera
    ros2 run plant_simulation follow_camera --target spot --offset=-2.5,0,1.6 --pgain 0.02

Started by bringup.launch.py with demo:=true (or follow_camera:=true).  It only
uses the `gz` command line tool (no Gazebo Python bindings, no rclpy):

1. waits until Gazebo runs, the robot exists (its /model/<target>/odometry or
   /world/<world>/model/<target>/joint_state topic is advertised) and the GUI
   is up (CameraTracking advertises /gui/camera/pose); gives up quietly after
   --timeout s, e.g. with gui:=false;
2. publishes a gz.msgs.CameraTrack on the GUI topic /gui/track
   (track_mode FOLLOW, follow_target <target>, follow_offset in the robot frame,
   follow_pgain) - understood by gz-gui 8 (Harmonic) ... 10 (Jetty);
3. confirms it on /gui/currently_tracked and re-sends if needed; if that
   never confirms and the GUI offers the older /gui/follow service (Harmonic),
   calls /gui/follow + /gui/follow/offset instead.

config/gui.config also names the target in its CameraTracking plugin
(follow_target / follow_offset / follow_pgain); those values make the GUI wait
for the robot, but the follow mode itself is only switched on by step 2.

Falls (fall guard, on by default; needs rclpy, which the ROS launch provides):
the GUI applies follow_offset in the robot frame with the FULL attitude
(gz-rendering BaseCamera: target pose x offset), so a robot on its side or back
swings the camera sideways or under the floor (that follow law run on 80
recorded Gazebo falls: camera below the floor in 67 of them).  After the follow
mode is engaged the program therefore stays up, watches /locomotion/state
(roll, pitch, recovery_state, mode) and

4. as soon as |roll| or |pitch| exceeds --tilt-hold (or the robot reports a
   fall) re-sends /gui/track with a near-zero follow_pgain - the camera stays
   where it is and keeps looking at the robot - and an offset levelled for the
   current attitude (Rx(roll)^T Ry(pitch)^T offset: after the GUI applies the
   full attitude only the heading is left, i.e. 'behind and above'); the
   levelled offset is refreshed while the robot rolls over;
5. once the robot is upright (tilt below --tilt-release, recovery_state NORMAL)
   for --release-hold s, sends the normal offset and gain again - the camera
   glides back behind the robot.
The guard publishes through the gz-transport python bindings when they are
installed (gz.transport15 / 14 / 13: no CLI start-up per message), else with
`gz topic -p` (~0.3-0.7 s per message).  Same 80 recorded falls with the guard,
by the delay from the state message to the GUI: 0.1-0.2 s (bindings) - the
camera stays >= 0.95 m above the ground; 0.4 s - never below the floor, lowest
0.28 m; 0.7 s - 13 of 80 dip below it briefly (lowest -0.55 m: the camera keeps
moving while the hold is on its way).  Without the guard: 66 of 80, down to
-2.4 m.
Without rclpy, or with --no-fall-guard, it exits after step 3 as before.

The program never fails the launch: every error is logged and it exits 0.
"""
import argparse
import math
import re
import shutil
import subprocess
import sys
import time

FOLLOW_DEFAULTS = {
    "target": "spot",
    "world": "plant",
    "offset": (-2.5, 0.0, 1.6),     # m in the robot frame: behind (-x) and above (+z)
    "pgain": 0.02,                  # per rendered frame (~60 Hz): ~0.8 s time constant
    "timeout": 240.0,               # s waiting for Gazebo, the robot and the GUI
    "attempts": 6,                  # /gui/track sends before falling back to /gui/follow
    "poll": 2.0,                    # s between checks
    # fall guard (needs rclpy): hold the camera while the robot is falling / getting up
    "fall_guard": True,
    "tilt_hold_deg": 30.0,          # |roll| or |pitch| above (or a reported fall) -> hold the camera
    "tilt_release_deg": 20.0,       # upright below this (ramp 11.8 deg + gait sway stays below) ...
    "release_hold": 1.0,            # ... with recovery_state NORMAL for this long [s] -> follow again
    "hold_pgain": 0.0002,           # follow gain while holding (~frozen: 1 % of the way in 1.4 s at 60 fps)
    "hold_update": 1.0,             # s between levelled-offset refreshes while holding
    "hold_min_change": 0.25,        # m change of the levelled offset worth a refresh
}
GUARD_KEYS = ("tilt_hold_deg", "tilt_release_deg", "release_hold", "hold_pgain", "hold_update", "hold_min_change")
DOWN_MODES = ("FALLEN", "RECOVERING")
LOCOMOTION_STATE_TOPIC = "/locomotion/state"
TRACK_TOPIC = "/gui/track"
TRACKED_TOPIC = "/gui/currently_tracked"
GUI_TOPIC = "/gui/camera/pose"


def parse_offset(text):
    """'-2.5,0,1.6' or '-2.5 0 1.6' -> (x, y, z)."""
    parts = [p for p in re.split(r"[,\s]+", str(text).strip()) if p]
    if len(parts) != 3:
        raise ValueError(f"offset needs 3 numbers, got {text!r}")
    return tuple(float(p) for p in parts)


def _num(v):
    return f"{float(v):.4g}"


def track_message(target, offset, pgain, mode="FOLLOW"):
    """Protobuf text of the gz.msgs.CameraTrack request."""
    x, y, z = offset
    return (f'track_mode: {mode}, follow_target: {{name: "{target}"}}, '
            f"follow_offset: {{x: {_num(x)}, y: {_num(y)}, z: {_num(z)}}}, follow_pgain: {_num(pgain)}")


def track_command(gz, target, offset, pgain):
    return [gz, "topic", "-t", TRACK_TOPIC, "-m", "gz.msgs.CameraTrack", "-p", track_message(target, offset, pgain)]


def follow_service_commands(gz, target, offset, timeout_ms=3000):
    """Harmonic-only fallback: /gui/follow (StringMsg) then /gui/follow/offset (Vector3d)."""
    x, y, z = offset
    base = ["--reptype", "gz.msgs.Boolean", "--timeout", str(int(timeout_ms))]
    return [
        [gz, "service", "-s", "/gui/follow", "--reqtype", "gz.msgs.StringMsg"] + base + ["--req", f'data: "{target}"'],
        [gz, "service", "-s", "/gui/follow/offset", "--reqtype", "gz.msgs.Vector3d"] + base
        + ["--req", f"x: {_num(x)}, y: {_num(y)}, z: {_num(z)}"],
    ]


def robot_topics(target, world):
    return (f"/model/{target}/odometry", f"/world/{world}/model/{target}/joint_state")


def level_offset(offset, roll, pitch):
    """Robot-frame offset that the GUI's full-attitude follow (R(yaw) R(pitch) R(roll) x offset)
    turns into R(yaw) x offset: Rx(roll)^T Ry(pitch)^T offset - camera behind / above the
    robot's heading whatever its roll and pitch."""
    x, y, z = (float(v) for v in offset)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    x1, z1 = cp * x - sp * z, sp * x + cp * z
    return (x1, cr * y + sr * z1, -sr * y + cr * z1)


def robot_down(d):
    """/locomotion/state dict: the controller reports a fall / a recovery in progress."""
    rs = str(d.get("recovery_state") or "")
    return (rs not in ("", "NORMAL")) or str(d.get("mode") or "") in DOWN_MODES


class FallGuard:
    """When to hold / release the follow camera (ROS free).

    update(t, roll, pitch, down) -> None, ("hold", offset, pgain) when the robot starts to fall,
    ("update", offset, pgain) for a refreshed levelled offset while holding, ("follow", offset,
    pgain) once it is upright again.  Angles in rad, t in s of any monotonic clock."""

    def __init__(self, offset, pgain, tilt_hold_deg=30.0, tilt_release_deg=20.0, release_hold=1.0,
                 hold_pgain=0.0002, hold_update=1.0, hold_min_change=0.25):
        self.offset, self.pgain = tuple(float(v) for v in offset), float(pgain)
        self.hold_tilt = math.radians(float(tilt_hold_deg))
        self.release_tilt = math.radians(min(float(tilt_release_deg), float(tilt_hold_deg)))
        self.release_hold, self.hold_pgain = float(release_hold), float(hold_pgain)
        self.hold_update, self.min_change = float(hold_update), float(hold_min_change)
        self.holding = False
        self.holds = 0
        self._up_since = None
        self._last_t = -math.inf
        self._last_off = None

    def update(self, t, roll, pitch, down=False):
        try:
            roll, pitch = float(roll), float(pitch)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(roll) and math.isfinite(pitch)):
            return None
        tilt = max(abs(roll), abs(pitch))
        if not self.holding:
            if tilt > self.hold_tilt or down:
                self.holding, self._up_since, self._last_t = True, None, t
                self.holds += 1
                self._last_off = level_offset(self.offset, roll, pitch)
                return ("hold", self._last_off, self.hold_pgain)
            return None
        if tilt < self.release_tilt and not down:
            if self._up_since is None:
                self._up_since = t
            if t - self._up_since >= self.release_hold:
                self.holding, self._up_since = False, None
                return ("follow", self.offset, self.pgain)
        else:
            self._up_since = None
        if t - self._last_t >= self.hold_update:
            off = level_offset(self.offset, roll, pitch)
            if math.dist(off, self._last_off) > self.min_change:
                self._last_t, self._last_off = t, off
                return ("update", off, self.hold_pgain)
        return None


class GzTransportPublisher:
    """/gui/track publisher through the gz-transport python bindings: advertised once, so a
    hold reaches the GUI without the start-up of a `gz topic -p` process (ImportError when
    the bindings are not installed)."""

    VERSIONS = ((15, 12), (14, 11), (13, 10))          # (transport, msgs): Jetty, Ionic, Harmonic

    def __init__(self, modules=None):
        import importlib
        load = modules or importlib.import_module
        last = None
        for vt, vm in self.VERSIONS:
            try:
                tr = load(f"gz.transport{vt}")
                ct = load(f"gz.msgs{vm}.cameratrack_pb2")
            except ImportError as exc:
                last = exc
                continue
            self.CameraTrack = ct.CameraTrack
            self.node = tr.Node()
            self.pub = self.node.advertise(TRACK_TOPIC, ct.CameraTrack)
            self.version = (vt, vm)
            return
        raise ImportError(f"no gz-transport python bindings ({last})")

    def publish(self, target, offset, pgain):
        m = self.CameraTrack()
        m.track_mode = self.CameraTrack.FOLLOW
        m.follow_target.name = str(target)
        m.follow_offset.x, m.follow_offset.y, m.follow_offset.z = (float(v) for v in offset)
        m.follow_pgain = float(pgain)
        return bool(self.pub.publish(m))


class RosLocomotionMonitor:
    """Latest /locomotion/state (std_msgs/String JSON) through rclpy, for the fall guard."""

    def __init__(self, ros_args=()):
        import json

        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        self._rclpy, self._json = rclpy, json
        self._own = not rclpy.ok()
        if self._own:
            rclpy.init(args=["follow_camera"] + list(ros_args))
        self.node = Node("follow_camera")
        self.latest = None
        # best effort matches a reliable and a best-effort publisher alike
        self.node.create_subscription(String, LOCOMOTION_STATE_TOPIC, self._on_state,
                                      QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))

    def _on_state(self, msg):
        try:
            d = self._json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if isinstance(d, dict):
            self.latest = d

    def ok(self):
        return bool(self._rclpy.ok())

    def poll(self, timeout):
        self.latest = None
        self._rclpy.spin_once(self.node, timeout_sec=float(timeout))
        return self.latest

    def close(self):
        try:
            self.node.destroy_node()
        finally:
            if self._own and self._rclpy.ok():
                self._rclpy.shutdown()


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


class FollowCamera:
    """Step logic with injectable runner / clock / sleep (unit-tested)."""

    def __init__(self, gz="gz", target="spot", world="plant", offset=(-2.5, 0.0, 1.6), pgain=0.02,
                 timeout=240.0, attempts=6, poll=2.0, runner=default_runner, clock=time.monotonic,
                 sleep=time.sleep, log=print, monitor_factory=None, guard_cfg=None, publisher_factory=None):
        self.gz, self.target, self.world = gz, target, world
        self.offset, self.pgain = tuple(offset), float(pgain)
        self.timeout, self.attempts, self.poll = float(timeout), int(attempts), float(poll)
        self.run, self.clock, self.sleep, self.log = runner, clock, sleep, log
        # fall guard: monitor_factory() -> object with poll(timeout) -> newest /locomotion/state dict
        # or None, ok() -> keep running, close().  None: exit once the follow mode is engaged.
        self.monitor_factory = monitor_factory
        self.guard_cfg = {k: v for k, v in dict(guard_cfg or {}).items() if k in GUARD_KEYS}
        # publisher_factory() -> object with publish(target, offset, pgain) (gz-transport bindings);
        # None or failing: `gz topic -p` for every guard message
        self.publisher_factory = publisher_factory
        self.fast = None

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
        for i in range(max(1, self.attempts)):
            rc, out = self.run(track_command(self.gz, self.target, self.offset, self.pgain), 10.0)
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
        x, y, z = self.offset
        if method:
            self.log(f"[follow_camera] Gazebo GUI camera follows '{self.target}' (offset {x:g} {y:g} {z:g} m, "
                     f"pgain {self.pgain:g}, via {'/gui/track' if method == 'track' else '/gui/follow'})")
        else:
            self.log("[follow_camera] could not confirm the follow camera - right-click the robot in the Entity "
                     "tree and choose 'Follow' instead")
        if method == "track" and self.monitor_factory is not None:
            self.guard()
        return 0

    def send(self, offset, pgain):
        if self.fast is not None:
            try:
                if self.fast.publish(self.target, offset, pgain):
                    return True
            except Exception as exc:
                self.log(f"[follow_camera] gz-transport publish failed ({type(exc).__name__}: {exc}) - "
                         "using gz topic -p")
                self.fast = None
        rc, out = self.run(track_command(self.gz, self.target, offset, pgain), 10.0)
        if rc != 0:
            self.log(f"[follow_camera] gz topic -p {TRACK_TOPIC} failed ({rc}): {out.strip()[:200]}")
        return rc == 0

    def guard(self, max_polls=None):
        """Fall guard loop (module doc, steps 4-5): hold the camera while the robot is down."""
        try:
            mon = self.monitor_factory()
        except Exception as exc:  # no rclpy / no ROS environment
            self.log(f"[follow_camera] fall guard off ({type(exc).__name__}: {exc}) - the camera may swing "
                     "with a fallen robot")
            return
        g = FallGuard(self.offset, self.pgain, **self.guard_cfg)
        via = "gz topic -p"
        if self.publisher_factory is not None:
            try:
                self.fast = self.publisher_factory()
                via = "gz-transport python" + (" %d" % self.fast.version[0] if hasattr(self.fast, "version") else "")
            except Exception:
                self.fast = None
        self.log(f"[follow_camera] fall guard on: camera held above {math.degrees(g.hold_tilt):.0f} deg tilt or "
                 f"while the robot recovers, following again {g.release_hold:g} s after it is upright (via {via})")
        polls = 0
        try:
            while mon.ok() and (max_polls is None or polls < max_polls):
                polls += 1
                d = mon.poll(0.1)
                if not isinstance(d, dict):
                    continue
                act = g.update(self.clock(), d.get("roll"), d.get("pitch"), robot_down(d))
                if act is None:
                    continue
                kind, off, gain = act
                self.send(off, gain)
                if kind == "hold":
                    state = d.get("recovery_state") if robot_down(d) else ""
                    self.log("[follow_camera] robot falling" + (f" ({state})" if state else "")
                             + " - camera held, still looking at it")
                elif kind == "follow":
                    self.log("[follow_camera] robot upright again - following")
        except KeyboardInterrupt:
            pass
        except Exception as exc:  # never fail the launch
            self.log(f"[follow_camera] fall guard stopped: {type(exc).__name__}: {exc}")
        finally:
            try:
                mon.close()
            except Exception:
                pass


def build_parser():
    d = FOLLOW_DEFAULTS
    ap = argparse.ArgumentParser(prog="follow_camera", description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", default=d["target"], help="model to follow (default %(default)s)")
    ap.add_argument("--world", default=d["world"], help="world name (default %(default)s)")
    ap.add_argument("--offset", default=",".join(f"{v:g}" for v in d["offset"]),
                    help="camera offset x,y,z in the robot frame, m (default %(default)s); use --offset=-2,0,1")
    ap.add_argument("--pgain", type=float, default=d["pgain"], help="follow P gain per frame (default %(default)s)")
    ap.add_argument("--timeout", type=float, default=d["timeout"], help="s to wait for gz / robot / GUI")
    ap.add_argument("--attempts", type=int, default=d["attempts"])
    ap.add_argument("--gz", default="", help="gz executable ('' = from PATH)")
    ap.add_argument("--no-fall-guard", dest="fall_guard", action="store_false", default=d["fall_guard"],
                    help="exit once the camera follows (no camera hold while the robot is down)")
    ap.add_argument("--tilt-hold", type=float, default=d["tilt_hold_deg"],
                    help="deg roll / pitch above which the camera is held (default %(default)s)")
    ap.add_argument("--tilt-release", type=float, default=d["tilt_release_deg"],
                    help="deg below which an upright robot is followed again (default %(default)s)")
    ap.add_argument("--release-hold", type=float, default=d["release_hold"],
                    help="s upright before following again (default %(default)s)")
    ap.add_argument("--hold-pgain", type=float, default=d["hold_pgain"],
                    help="follow gain while holding (default %(default)s)")
    return ap


def main(argv=None, monitor_factory=None):
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

    if args.fall_guard and monitor_factory is None:
        def monitor_factory():
            return RosLocomotionMonitor(ros_args)
    guard_cfg = {"tilt_hold_deg": args.tilt_hold, "tilt_release_deg": args.tilt_release,
                 "release_hold": args.release_hold, "hold_pgain": args.hold_pgain}
    try:
        return FollowCamera(gz, args.target, args.world, offset, args.pgain, args.timeout, args.attempts,
                            log=log, monitor_factory=monitor_factory if args.fall_guard else None,
                            guard_cfg=guard_cfg, publisher_factory=GzTransportPublisher).main()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # never fail the launch
        log(f"[follow_camera] unexpected error {type(exc).__name__}: {exc} - not following")
        return 0


if __name__ == "__main__":
    sys.exit(main())
