"""Follow camera with a mouse-adjustable, kept view: follow_camera_logic.CameraDirector against a
simulated Gazebo GUI, plus the follow_camera program (gz CLI / bindings / rclpy inputs) with fakes.

FakeGui reproduces frame by frame what gz-gui CameraTracking (gz-gui 8 Harmonic / 9 / 10 Jetty, same
code) and gz-rendering BaseCamera do:
  * OnTrackSub: mode replaced unless USE_LAST, empty target keeps the target, offset replaced when
    present, a gain taken only when > 1e-5;
  * OnRender FOLLOW: message follow_pgain -> look-at gain, message track_pgain -> POSITION gain;
    FOLLOW_FREE_LOOK: position gain 1; TRACK: no position follow; NONE with no target: nothing;
  * per rendered frame: follow law (camera += (robot pose x offset - camera) * position gain) before
    the mouse (InteractiveViewControl) moves the camera; Esc applies with the next mouse event;
  * 50 Hz timer: /gui/camera/pose and /gui/currently_tracked.
"""
import heapq
import math
import random
import re
import types

import pytest

from plant_simulation import follow_camera as fc
from plant_simulation import follow_camera_logic as L

D = L.DEFAULT_OFFSET
ENUM = {"NONE": 0, "TRACK": 1, "FOLLOW": 2, "FOLLOW_FREE_LOOK": 3, "FOLLOW_LOOK_AT": 4, "USE_LAST": 5}


def quat(r, p, y):
    cr, sr, cp, sp = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


class Robot:
    """Pose trace (5 ms) of a trotting robot: segments (duration s, v m/s, w rad/s) with roll / pitch
    sway while moving; roll_fn(t) adds a roll (falls), z_fn(t) replaces the trunk height."""

    def __init__(self, segments, sway_deg=4.0, f=2.2, start=(0.0, 0.0, 0.0), roll_fn=None, z_fn=None, dt=0.005):
        self.dt = dt
        self.tr = []
        x, y, yaw = start
        t = 0.0
        for dur, v, w in segments:
            for _ in range(int(round(dur / dt))):
                a = math.radians(sway_deg) if (v or w) else 0.0
                r = a * math.sin(2 * math.pi * f * t) + (roll_fn(t) if roll_fn else 0.0)
                p = 0.7 * a * math.sin(2 * math.pi * f * t + 1.1)
                z = z_fn(t) if z_fn else 0.32 + (0.01 * math.sin(4 * math.pi * f * t) if (v or w) else 0.0)
                self.tr.append(((x, y, z), quat(r, p, yaw), r, p, yaw))
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
                yaw += w * dt
                t += dt

    def sample(self, t):
        return self.tr[min(max(int(t / self.dt), 0), len(self.tr) - 1)]

    def __call__(self, t):
        s = self.sample(t)
        return s[0], s[1]


def rel_heading(robot, t, cam):
    """Camera relative to the robot in its heading frame (what a viewer sees as 'behind / above')."""
    p, _, _, _, yaw = robot.sample(t)
    return L.q_inv_rot(L.yaw_quat(yaw), L.sub(cam, p))


class FakeTrackMsg:
    """Stand-in of the gz.msgs.CameraTrack protobuf class (enum values as class attributes, HasField)."""
    NONE, TRACK, FOLLOW, FOLLOW_FREE_LOOK, FOLLOW_LOOK_AT, USE_LAST = range(6)

    class _Vec:
        def __init__(self, owner, name):
            object.__setattr__(self, "_o", (owner, name))
            object.__setattr__(self, "x", 0.0)
            object.__setattr__(self, "y", 0.0)
            object.__setattr__(self, "z", 0.0)

        def __setattr__(self, k, v):
            object.__setattr__(self, k, v)
            self._o[0]._has.add(self._o[1])

    def __init__(self):
        self._has = set()
        self.track_mode = 0
        self.follow_target = types.SimpleNamespace(name="")
        self.track_target = types.SimpleNamespace(name="")
        self.follow_offset = FakeTrackMsg._Vec(self, "follow_offset")
        self.track_offset = FakeTrackMsg._Vec(self, "track_offset")
        self.follow_pgain = 0.0
        self.track_pgain = 0.0

    def HasField(self, name):
        return name in self._has


def msg_from_req(req, via="gz"):
    """The CameraTrack the GUI receives: through the bindings (TrackRequest.fill) or `gz topic -p` text."""
    if via == "gz":
        m = req.fill(FakeTrackMsg(), FakeTrackMsg)
        names = {v: k for k, v in ENUM.items()}
        return {"mode": names[m.track_mode], "follow_target": m.follow_target.name,
                "track_target": m.track_target.name,
                "follow_offset": (m.follow_offset.x, m.follow_offset.y, m.follow_offset.z)
                if m.HasField("follow_offset") else None,
                "follow_pgain": m.follow_pgain, "track_pgain": m.track_pgain}
    text = req.text()
    mode = re.search(r"track_mode:\s*(\w+)", text).group(1)
    ft = re.search(r'follow_target:\s*\{name:\s*"([^"]*)"\}', text)
    tt = re.search(r'track_target:\s*\{name:\s*"([^"]*)"\}', text)
    fo = re.search(r"follow_offset:\s*\{x:\s*([^,]+),\s*y:\s*([^,]+),\s*z:\s*([^}]+)\}", text)
    fg = re.search(r"follow_pgain:\s*([-+0-9.eE]+)", text)
    tg = re.search(r"track_pgain:\s*([-+0-9.eE]+)", text)
    return {"mode": mode, "follow_target": ft.group(1) if ft else "", "track_target": tt.group(1) if tt else "",
            "follow_offset": tuple(float(v) for v in fo.groups()) if fo else None,
            "follow_pgain": float(fg.group(1)) if fg else 0.0, "track_pgain": float(tg.group(1)) if tg else 0.0}


class FakeGui:
    def __init__(self, robot, fps=30.0, gui_lag=0.03, seed=0):
        self.robot, self.fps, self.gui_lag = robot, float(fps), float(gui_lag)
        self.rng = random.Random(seed)
        # CameraTracking, preset by gui.config (follow_target spot, follow_offset, follow_pgain)
        self.mode, self.sel_follow, self.sel_track = "NONE", "spot", ""
        self.offset, self.follow_pgain, self.track_pgain = D, L.LOOK_GAIN, 0.01
        self.new_track = True
        # camera (gz-rendering)
        p, q = robot(0.0)
        self.cam = L.add(p, (3.0, -4.0, 3.0))           # the GUI's own start pose, not behind the robot
        self.cam_follow, self.cam_track = False, False
        self.cam_off, self.cam_pos_gain, self.cam_look_gain = D, 1.0, 1.0
        self.esc_pending = False
        self.inputs = []                                  # (t0, kind, data) not started yet
        self.active = []
        self.received = []

    # --- CameraTracking::OnTrackSub
    def receive(self, m):
        self.received.append(m)
        if m["mode"] != "USE_LAST":
            self.mode = m["mode"]
        if m.get("follow_target"):
            self.sel_follow = m["follow_target"]
        if m.get("track_target"):
            self.sel_track = m["track_target"]
        if m.get("follow_offset") is not None:
            self.offset = tuple(m["follow_offset"])
        if (m.get("track_pgain") or 0.0) > 0.00001:
            self.track_pgain = m["track_pgain"]
        if (m.get("follow_pgain") or 0.0) > 0.00001:
            self.follow_pgain = m["follow_pgain"]
        self.new_track = True

    def context_follow(self):
        """Right-click on the robot -> Follow (gz-sim EntityContextMenu): mode + target only."""
        self.receive({"mode": "FOLLOW", "follow_target": "spot"})

    def press_esc(self):
        self.esc_pending = True

    def user(self, t0, kind, data=None):
        self.inputs.append((t0, kind, data))
        self.inputs.sort(key=lambda e: e[0])

    # --- CameraTracking::OnRender
    def _on_render(self):
        if self.sel_follow or self.sel_track:
            if self.mode in ("FOLLOW", "FOLLOW_FREE_LOOK", "FOLLOW_LOOK_AT"):
                if not self.cam_follow or self.new_track:
                    self.cam_follow, self.cam_off = True, self.offset
                    if self.mode == "FOLLOW":
                        self.cam_track = True
                        self.cam_look_gain, self.cam_pos_gain = self.follow_pgain, self.track_pgain   # crossed
                    elif self.mode == "FOLLOW_FREE_LOOK":
                        self.cam_track, self.cam_pos_gain = False, 1.0
                    self.new_track = False
            if self.mode == "TRACK":
                if not self.cam_track or self.cam_follow or self.new_track:
                    self.cam_follow, self.cam_track, self.cam_look_gain = False, True, self.track_pgain
                    self.new_track = False
        else:
            self.cam_follow = self.cam_track = False

    def frame(self, t):
        """One rendered frame at t; returns the time of the next frame."""
        p, q = self.robot(t - self.gui_lag)
        while self.inputs and self.inputs[0][0] <= t:
            self.active.append(self.inputs.pop(0))
        mouse = bool(self.active)
        if self.esc_pending and mouse:                    # key events reach the plugins with a mouse event
            self.mode, self.sel_follow, self.sel_track, self.esc_pending = "NONE", "", "", False
        if self.cam_follow:                               # BaseCamera::PreRender
            tgt = L.add(p, L.q_rot(q, self.cam_off))
            self.cam = L.add(self.cam, tuple(self.cam_pos_gain * v for v in L.sub(tgt, self.cam)))
        self._on_render()
        for ev in list(self.active):                      # InteractiveViewControl
            t0, kind, data = ev
            if kind == "zoom":                            # m towards the robot (negative: away)
                d = L.sub(p, self.cam)
                n = L.norm(d)
                self.cam = L.add(self.cam, tuple(data * v / n for v in d))
                self.active.remove(ev)
            elif kind == "move":
                self.cam = L.add(self.cam, data)
                self.active.remove(ev)
            elif kind == "click":
                self.active.remove(ev)
            elif kind == "orbit":                         # (angle rad, duration s) around the robot
                ang, dur = data
                if t > t0 + dur:
                    self.active.remove(ev)
                    continue
                step = ang / max(dur * self.fps, 1.0)
                rel = L.sub(self.cam, p)
                c, s = math.cos(step), math.sin(step)
                self.cam = L.add(p, (c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]))
        return t + (1.0 / self.fps) * self.rng.uniform(0.8, 1.25)

    def status(self):
        """/gui/currently_tracked as parse_tracked would give it."""
        if self.mode in ("FOLLOW", "FOLLOW_FREE_LOOK"):
            return {"mode": self.mode, "follow_target": self.sel_follow, "track_target": "",
                    "follow_offset": self.offset, "follow_pgain": self.follow_pgain}
        if self.mode == "TRACK":
            return {"mode": "TRACK", "follow_target": "", "track_target": self.sel_track, "follow_offset": None,
                    "follow_pgain": None}
        return {"mode": "NONE", "follow_target": "", "track_target": "", "follow_offset": None, "follow_pgain": None}


class Sim:
    """FakeGui <-> CameraDirector with message / sample delays.  transport 'gz': bindings (send_lag, echo
    on /gui/currently_tracked at 50 Hz); 'cli': `gz topic -p` blocks the director cli_lag s (events queue
    up and arrive together afterwards), status polled every 2.5 s."""

    def __init__(self, robot, fps=30.0, gui_lag=0.03, robot_lag=0.05, send_lag=0.005, transport="gz",
                 cli_lag=0.5, cfg=None, walls=None, loco=None, seed=0, offset=D):
        self.robot, self.gui = robot, FakeGui(robot, fps, gui_lag, seed)
        self.robot_lag, self.send_lag, self.transport, self.cli_lag = robot_lag, send_lag, transport, cli_lag
        self.loco = loco
        self.rng = random.Random(seed + 1)
        self.q, self.seq, self.t, self.busy_until = [], 0, 0.0, -1.0
        self.sent, self.logs, self.keys = [], [], []
        self.cams = []                                    # (t, camera) every frame
        self.d = L.CameraDirector("spot", offset, self._send, log=self.logs.append, clock=self._clock, cfg=cfg,
                                  walls=walls, t0=0.0, fast=transport == "gz")
        c = self.d.cfg
        self.gui.receive(msg_from_req(fc.follow_request("spot", offset, c["look_gain"], c["pos_gain"])))   # engage
        for kind, t in (("frame", 0.0), ("timer", 0.0), ("robot", 0.0), ("loco", 0.0), ("tick", 0.0),
                        ("poll", 0.5)):
            self._push(t, kind, None)

    def _clock(self):
        return max(self.t, self.busy_until)

    def _push(self, t, kind, data):
        self.seq += 1
        heapq.heappush(self.q, (t, self.seq, kind, data))

    def _send(self, req):
        self.sent.append((self.t, req))
        if self.transport == "gz":
            self._push(self.t + self.send_lag, "gui_rx", msg_from_req(req, "gz"))
            return "gz"
        self._push(self.t + 0.8 * self.cli_lag, "gui_rx", msg_from_req(req, "cli"))
        self.busy_until = self.t + self.cli_lag
        return "cli"

    def key(self, t, code):
        self._push(t, "dir", ("key", t, code))

    def at(self, t, fn):
        self._push(t, "call", fn)

    def run(self, until):
        while self.q and self.q[0][0] <= until:
            t, _, kind, data = heapq.heappop(self.q)
            self.t = t
            if kind == "frame":
                nxt = self.gui.frame(t)
                self.cams.append((t, self.gui.cam))
                self._push(nxt, "frame", None)
            elif kind == "timer":
                cam = self.gui.cam
                self._push(t + 0.002, "dir", ("cam", t + 0.002, cam))
                if self.transport == "gz":
                    self._push(t + 0.002, "dir", ("tracked", t + 0.002, self.gui.status()))
                self._push(t + 0.02 * self.rng.uniform(0.9, 1.6), "timer", None)
            elif kind == "poll":                          # CliTrackedPoller: stamp = poll start
                if self.transport == "cli":
                    self._push(t + 0.4, "dir", ("tracked", t, self.gui.status()))
                self._push(t + 2.5, "poll", None)
            elif kind == "robot":
                p, q = self.robot(t - self.robot_lag)
                self._push(t, "dir", ("robot", t, (p, q)))
                self._push(t + 0.02, "robot", None)
            elif kind == "loco":
                if self.loco is not None:
                    dct = self.loco(t)
                    if dct is not None:
                        self._push(t, "dir", ("loco", t, dct))
                self._push(t + 0.05, "loco", None)
            elif kind == "tick":
                if t >= self.busy_until:
                    self.d.tick(t)
                self._push(t + 0.05, "tick", None)
            elif kind == "gui_rx":
                self.gui.receive(data)
            elif kind == "call":
                data()
            elif kind == "dir":
                ek, ts, payload = data
                if t < self.busy_until:                   # director blocked in `gz topic -p`
                    self._push(self.busy_until, "dir", (ek, ts if ek == "tracked" else self.busy_until, payload))
                    continue
                if ek == "robot":
                    self.d.on_robot(ts, payload[0], payload[1])
                elif ek == "cam":
                    self.d.on_camera(ts, payload)
                elif ek == "tracked":
                    self.d.on_tracked(ts, payload)
                elif ek == "loco":
                    self.d.on_loco(ts, payload)
                elif ek == "key":
                    self.d.on_key(ts, payload)
        self.t = until

    def kinds(self, since=0.0):
        """Sent requests after `since` as (mode, gain class) tuples."""
        c = self.d.cfg
        out = []
        for t, r in self.sent:
            if t < since:
                continue
            g = "hold" if r.pos == c["hold_gain"] else ("pos" if r.pos == c["pos_gain"] else r.pos)
            out.append((r.mode, g))
        return out

    def cam_speed(self, t0, t1):
        pts = [(t, c) for t, c in self.cams if t0 <= t <= t1]
        return L.norm(L.sub(pts[-1][1], pts[0][1])) / max(pts[-1][0] - pts[0][0], 1e-9)


def walk_then_stand(walk=10.0, stand=60.0, v=0.5):
    return Robot([(walk, v, 0.0), (stand, 0.0, 0.0)])


def wander(T=120.0, seed=0, sway=4.0):
    rng = random.Random(seed)
    segs, tot = [], 0.0
    while tot < T + 1:
        kind = rng.choice(["walk", "arc", "spin", "stand"])
        dur = rng.uniform(3, 10)
        v, w = {"walk": (0.5, 0.0), "arc": (0.4, rng.choice([-0.35, 0.35])), "spin": (0.0, rng.choice([-0.5, 0.5])),
                "stand": (0.0, 0.0)}[kind]
        segs.append((dur, v, w))
        tot += dur
    return Robot(segs, sway_deg=sway)


# ============================================================================
# the fake GUI itself (the gz-gui facts the director relies on)
# ============================================================================
def test_fake_gui_follows_the_source_semantics():
    r = Robot([(30.0, 0.0, 0.0)])
    g = FakeGui(r, fps=60.0)
    g.receive(msg_from_req(fc.follow_request("spot", D, look=0.3, pos=0.02)))
    t = 0.0
    c0 = g.cam
    t = g.frame(t)                                  # OnRender switches the camera to follow ...
    t = g.frame(t)                                  # ... the next frame moves it by the POSITION gain
    tgt = L.add(r(0)[0], L.q_rot(r(0)[1], D))
    assert L.norm(L.sub(g.cam, c0)) == pytest.approx(0.02 * L.norm(L.sub(tgt, c0)), rel=1e-6)
    assert (g.cam_pos_gain, g.cam_look_gain) == (0.02, 0.3)              # follow_pgain = look, track_pgain = pos
    g.receive(msg_from_req(L.TrackRequest("USE_LAST", "spot", (-1, 0, 2), 0.000005, 0.000005)))
    g.frame(t)
    assert (g.cam_pos_gain, g.cam_look_gain) == (0.02, 0.3) and g.cam_off == (-1, 0, 2)   # <= 1e-5 ignored
    g.press_esc()
    g.user(0.0, "click")
    g.frame(t)
    assert g.mode == "NONE" and not g.cam_follow and g.status()["mode"] == "NONE"
    g.receive(msg_from_req(L.TrackRequest("USE_LAST", "spot", D, 0.05, 0.02)))  # cannot switch it back on
    g.frame(t)
    assert not g.cam_follow and g.mode == "NONE"
    g.context_follow()
    g.frame(t)
    assert g.cam_follow and g.cam_pos_gain == 0.02                     # stored gains are used again


def test_request_texts_and_protobuf_fill():
    f = fc.follow_request("spot", D)
    assert f.text() == ('track_mode: FOLLOW, follow_target: {name: "spot"}, follow_offset: {x: -2.5, y: 0, z: 1.6}, '
                        'follow_pgain: 0.05, track_pgain: 0.02')
    u = L.TrackRequest("USE_LAST", "spot", (-1.25, 0.5, 2.0), 0.05, L.HOLD_GAIN)
    assert u.text() == ("track_mode: USE_LAST, follow_offset: {x: -1.25, y: 0.5, z: 2}, follow_pgain: 0.05, "
                        "track_pgain: 2e-05")
    assert "follow_target" not in u.text()                                 # keeps the GUI's target (and Esc)
    tr = L.TrackRequest("TRACK", "spot", D, 0.05, 0.05)
    assert tr.text() == 'track_mode: TRACK, track_target: {name: "spot"}, track_pgain: 0.05'
    for req in (f, u, tr):                                                 # bindings and CLI say the same
        a, b = msg_from_req(req, "gz"), msg_from_req(req, "cli")
        assert {k: v for k, v in a.items() if k != "follow_offset"} == {k: v for k, v in b.items()
                                                                       if k != "follow_offset"}
        assert (a["follow_offset"] is None) == (b["follow_offset"] is None) or req.mode == "TRACK"
        if req.mode != "TRACK":
            assert a["follow_offset"] == pytest.approx(b["follow_offset"])
    m = msg_from_req(tr, "gz")
    assert m["mode"] == "TRACK" and m["track_target"] == "spot" and m["track_pgain"] == 0.05
    assert L.HOLD_GAIN > L.MIN_GUI_GAIN                                    # the GUI takes the hold gain
    cmd = fc.track_command("gz", u)
    assert cmd[:7] == ["gz", "topic", "-t", "/gui/track", "-m", "gz.msgs.CameraTrack", "-p"] and cmd[7] == u.text()


def test_parse_tracked_and_matches():
    echo = 'track_mode: FOLLOW\nfollow_target {\n  name: "spot"\n}\nfollow_offset {\n  x: -2.5\n  z: 1.6\n}\n' \
           "follow_pgain: 0.05\n"
    info = L.parse_tracked(echo)
    assert info == {"mode": "FOLLOW", "follow_target": "spot", "track_target": "", "follow_offset": (-2.5, 0.0, 1.6),
                    "follow_pgain": 0.05}
    assert L.tracked_matches(info, fc.follow_request("spot", D))
    assert not L.tracked_matches(info, fc.follow_request("spot", (-2.0, 0, 1.6)))
    assert L.parse_tracked("")["mode"] == "NONE"                            # NONE is omitted (proto3)
    t = L.parse_tracked('track_mode: TRACK\ntrack_target {\n  name: "spot"\n}\ntrack_pgain: 0.05\n')
    assert t["mode"] == "TRACK" and t["track_target"] == "spot"
    assert L.tracked_matches(t, L.TrackRequest("TRACK", "spot", D, 0.05, 0.05))
    m = FakeTrackMsg()
    m.track_mode = 2
    m.follow_target.name = "spot"
    m.follow_offset.x = -2.0
    m.follow_pgain = 0.05
    assert L.tracked_from_msg(m) == {"mode": "FOLLOW", "follow_target": "spot", "track_target": "",
                                     "follow_offset": (-2.0, 0.0, 0.0), "follow_pgain": 0.05}


# ============================================================================
# mouse -> kept view
# ============================================================================
@pytest.mark.parametrize("fps,gui_lag,robot_lag,sway,seed", [(20.0, 0.03, 0.08, 8.0, 1), (60.0, 0.15, 0.0, 8.0, 2),
                                                             (30.0, 0.0, 0.10, 4.0, 3), (60.0, 0.03, 0.03, 4.0, 4)])
def test_no_mouse_no_messages(fps, gui_lag, robot_lag, sway, seed):
    """Walking, arcing, spinning, standing, trunk sway, the GUI's robot pose behind / ahead of ours: the
    follow law alone never looks like the mouse."""
    s = Sim(wander(120.0, seed, sway), fps=fps, gui_lag=gui_lag, robot_lag=robot_lag, seed=seed)
    s.run(120.0)
    assert s.sent == [] and s.d.view == D and s.d.state == "FOLLOW"


@pytest.mark.parametrize("fps,transport,send_lag", [(20.0, "gz", 0.005), (60.0, "gz", 0.005), (30.0, "gz", 0.6),
                                                    (30.0, "cli", 0.0)])
def test_wheel_zoom_while_standing_is_kept(fps, transport, send_lag):
    r = walk_then_stand(walk=8.0, stand=60.0)
    s = Sim(r, fps=fps, send_lag=send_lag, transport=transport)
    s.gui.user(12.0, "zoom", 0.9)                  # two wheel notches in
    s.gui.user(12.15, "zoom", 0.6)
    s.run(16.0)
    assert s.kinds() == [("USE_LAST", "hold"), ("USE_LAST", "pos")] and s.d.stats["adopted"] == 1
    seen = rel_heading(r, 16.0, s.gui.cam)        # what the user looks at once the view is taken over
    assert L.norm(seen) < L.norm(D) - 0.8           # closer than the default view
    s.run(40.0)
    later = rel_heading(r, 40.0, s.gui.cam)
    assert L.norm(L.sub(later, seen)) < 0.05        # kept: no gliding back
    assert s.gui.offset == pytest.approx(s.d.view, abs=1e-4) and len(s.sent) == 2
    assert any("view kept" in x for x in s.logs)


@pytest.mark.parametrize("fps,send_lag", [(20.0, 0.005), (60.0, 0.6)])
def test_orbit_while_walking_is_kept(fps, send_lag):
    r = Robot([(40.0, 0.5, 0.0), (5.0, 0.0, 0.0), (40.0, 0.4, 0.2), (20.0, 0.0, 0.0)])
    s = Sim(r, fps=fps, send_lag=send_lag)
    s.gui.user(10.0, "orbit", (math.radians(60), 1.0))   # drag the camera 60 deg around the robot
    s.run(11.0)
    user_az = math.degrees(math.atan2(*reversed(rel_heading(r, 11.0, s.gui.cam)[:2])))
    s.run(30.0)
    assert s.d.stats["adopted"] == 1 and s.d.stats["reverted"] == 0 and s.kinds()[0] == ("USE_LAST", "hold")
    v = s.d.view
    view_az = math.degrees(math.atan2(v[1], v[0]))
    assert abs(view_az - user_az) < 12.0                 # the user's side of the robot
    assert abs(math.degrees(math.atan2(D[1], D[0])) - view_az) > 40.0
    s.run(105.0)                                         # long walk, an arc, standing: still that view
    assert s.gui.offset == pytest.approx(v, abs=1e-4) and s.d.stats["adopted"] == 1
    rel = rel_heading(r, 105.0, s.gui.cam)
    assert L.norm(L.sub(rel, v)) < 0.05                  # the gait sway of the moment was not frozen in


def test_small_camera_nudge_is_ignored():
    r = walk_then_stand(walk=5.0, stand=40.0)
    s = Sim(r, fps=30.0)
    s.gui.user(10.0, "move", (0.0, 0.07, 0.0))           # 7 cm: just above the trigger
    s.run(20.0)
    assert s.kinds() == [("USE_LAST", "hold"), ("USE_LAST", "pos")]
    assert s.d.stats["reverted"] == 1 and s.d.view == D and s.gui.offset == pytest.approx(D)


def test_no_mouse_option_keeps_the_old_follow():
    r = walk_then_stand(walk=5.0, stand=40.0)
    s = Sim(r, fps=30.0, cfg={"adjust": False})
    s.gui.user(10.0, "zoom", 1.0)
    s.run(30.0)
    assert s.sent == []
    assert L.norm(L.sub(rel_heading(r, 30.0, s.gui.cam), D)) < 0.05     # pulled back to the default view


# ============================================================================
# fall guard with the user's view
# ============================================================================
def fall_robot(t_fall=20.0, lie=6.0, x_slide=0.6):
    """Walk 10 s, stand, roll onto the back at t_fall within 0.8 s, lie, roll back within 1.2 s, stand."""
    def roll(t):
        if t < t_fall:
            return 0.0
        if t < t_fall + 0.8:
            return math.pi * (t - t_fall) / 0.8
        if t < t_fall + 0.8 + lie:
            return math.pi
        if t < t_fall + 2.0 + lie:
            return math.pi * (1 - (t - t_fall - 0.8 - lie) / 1.2)
        return 0.0

    def z(t):
        return 0.12 if t_fall <= t <= t_fall + 3.0 + lie else 0.32

    def loco(t):
        rr = roll(t)
        down = t_fall + 0.2 < t < t_fall + 3.5 + lie
        return {"roll": rr, "pitch": 0.0, "mode": "RECOVERING" if down else "STAND",
                "recovery_state": "TUCK" if down else "NORMAL"}
    return Robot([(10.0, 0.5, 0.0), (60.0, 0.0, 0.0)], roll_fn=roll, z_fn=z), loco


def test_fall_holds_the_camera_and_returns_to_the_users_view():
    r, loco = fall_robot()
    s = Sim(r, fps=30.0, loco=loco)
    s.gui.user(12.0, "zoom", 1.0)                    # user's view first
    s.run(19.0)
    view = s.d.view
    assert s.d.stats["adopted"] == 1 and L.norm(view) < L.norm(D) - 0.5
    n0 = len(s.sent)
    s.run(28.0)                                      # falling, on its back
    hold = s.sent[n0][1]
    assert s.d.state == "FALL" and hold.mode == "USE_LAST"
    assert hold.pos == s.d.cfg["hold_gain"] and hold.look == s.d.cfg["look_gain"]    # position frozen, still looking
    assert s.cam_speed(20.6, 28.0) < 0.01
    assert min(c[2] for t, c in s.cams if 20.0 <= t <= 28.0) > 0.5                  # never under the floor
    s.run(60.0)
    assert s.d.state == "FOLLOW" and s.sent[-1][1].pos == s.d.cfg["pos_gain"]
    assert s.sent[-1][1].offset == pytest.approx(view)                               # the user's view, not the default
    assert L.norm(L.sub(rel_heading(r, 60.0, s.gui.cam), view)) < 0.1
    assert any("robot falling" in x for x in s.logs) and any("upright again" in x for x in s.logs)


def test_fall_without_guard_would_put_the_camera_under_the_floor():
    r, loco = fall_robot()
    s = Sim(r, fps=30.0, loco=loco, cfg={"fall_guard": False})
    s.run(40.0)
    assert min(c[2] for t, c in s.cams if 20.0 <= t <= 30.0) < -0.5                # what the guard prevents


def test_camera_moved_during_the_fall_hold_becomes_the_view():
    r, loco = fall_robot()
    s = Sim(r, fps=30.0, loco=loco)
    s.gui.user(23.0, "orbit", (math.radians(-90), 1.0))   # while the robot lies on its back
    s.run(60.0)
    v = s.d.view
    assert s.d.view_auto is False and math.degrees(math.atan2(v[1], v[0])) == pytest.approx(90.0, abs=20.0)
    assert s.gui.offset == pytest.approx(v, abs=1e-4)
    assert any("from the view you set" in x for x in s.logs)


# ============================================================================
# keys: 'c' camera stays / follows from the new view, 'z' default view
# ============================================================================
def test_key_c_tripod_then_follow_from_the_new_view():
    r = Robot([(14.0, 0.4, 0.0), (16.0, 0.0, 0.0), (30.0, 0.4, 0.3), (40.0, 0.0, 0.0)])
    s = Sim(r, fps=30.0)
    s.key(10.0, L.QT_KEY_C)
    s.run(13.0)
    assert s.d.state == "TRIPOD" and s.sent[-1][1].mode == "TRACK" and s.gui.mode == "TRACK"
    assert s.cam_speed(11.0, 13.0) < 0.02                            # stays while the robot walks on
    s.gui.user(15.0, "move", (0.0, 3.0, 1.0))                         # put it somewhere else with the mouse
    s.run(18.0)
    assert s.d.state == "TRIPOD" and s.cam_speed(16.0, 18.0) < 0.02   # nothing pulls it back
    seen = rel_heading(r, 18.0, s.gui.cam)
    s.key(18.0, L.QT_KEY_C)
    s.run(19.0)
    assert s.d.state == "FOLLOW" and s.sent[-1][1].mode == "FOLLOW" and s.d.view_auto is False
    assert s.d.view == pytest.approx(seen, abs=0.02)
    s.run(98.0)                                                       # walked and turned, standing again
    assert L.norm(L.sub(rel_heading(r, 98.0, s.gui.cam), seen)) < 0.05
    assert s.gui.offset == pytest.approx(s.d.view, abs=1e-4)


def test_key_z_default_view_and_auto_repeat():
    r = walk_then_stand(walk=5.0, stand=60.0)
    s = Sim(r, fps=30.0)
    s.gui.user(8.0, "zoom", 1.2)
    s.run(12.0)
    assert s.d.view != D
    n = len(s.sent)
    for k in range(4):                                     # held key: OS auto-repeat
        s.key(12.0 + 0.03 * k, L.QT_KEY_Z)
    s.key(12.2, 0x41)                                      # other keys are not ours
    s.run(30.0)
    assert len(s.sent) == n + 1 and s.sent[-1][1].mode == "FOLLOW" and s.d.view == D and s.d.view_auto
    assert L.norm(L.sub(rel_heading(r, 30.0, s.gui.cam), D)) < 0.05


def test_esc_passive_then_right_click_follow_or_c():
    r, loco = fall_robot(t_fall=30.0)
    s = Sim(r, fps=30.0, loco=loco)
    s.gui.user(8.0, "zoom", 1.0)
    s.run(12.0)
    view = s.d.view
    s.at(14.0, s.gui.press_esc)
    s.gui.user(14.3, "click")                           # Esc applies with the next mouse event
    s.run(16.0)
    assert s.d.state == "PASSIVE" and any("Esc" in x for x in s.logs)
    n = len(s.sent)
    s.run(20.0)
    s.at(20.0, s.gui.context_follow)                    # right-click -> Follow: no gains / offset
    s.run(22.0)
    assert s.d.state == "FOLLOW" and len(s.sent) == n + 1
    assert s.sent[-1][1].mode == "USE_LAST" and s.sent[-1][1].offset == pytest.approx(view)
    assert s.gui.cam_pos_gain == s.d.cfg["pos_gain"]
    s.at(23.0, s.gui.press_esc)
    s.gui.user(23.2, "click")
    s.run(29.0)
    n = len(s.sent)
    s.run(36.0)                                         # robot falls while the GUI camera is free
    assert len(s.sent) == n and s.d.state == "PASSIVE"
    s.key(45.0, L.QT_KEY_C)                              # robot is up again: follow from where the camera is
    s.run(46.0)
    assert s.d.state == "FOLLOW" and s.sent[-1][1].mode == "FOLLOW" and s.gui.mode == "FOLLOW"


def test_key_c_while_the_robot_is_down_follows_with_the_hold():
    r, loco = fall_robot(t_fall=20.0)
    s = Sim(r, fps=30.0, loco=loco)
    s.key(10.0, L.QT_KEY_C)
    s.run(22.0)
    s.key(22.5, L.QT_KEY_C)                              # on its back: follow = hold until it is up
    s.run(23.0)
    assert s.d.state == "FALL" and s.sent[-1][1].mode == "FOLLOW" and s.sent[-1][1].pos == s.d.cfg["hold_gain"]
    s.run(50.0)
    assert s.d.state == "FOLLOW" and s.sent[-1][1].pos == s.d.cfg["pos_gain"]


# ============================================================================
# building walls (default view only)
# ============================================================================
def _stops():
    from dusan_common.layout import Layout
    lay = Layout.load()
    sx, sy, _, syaw = lay.spawn
    stops = [("dock", sx, sy, syaw)]
    for g in lay.gauges:
        px, py = g["position"][0], g["position"][1]
        nx, ny = g["normal"][0], g["normal"][1]
        stops.append((g["id"], px + 1.4 * nx, py + 1.4 * ny, math.atan2(-ny, -nx)))
    return lay, stops


def test_walls_pull_the_default_view_in_front_of_the_corridor_walls():
    from dusan_common.grid_map import OccupancyGridMap
    lay, stops = _stops()
    walls = L.WallMap(L.layout_walls(lay))
    assert len(walls.segments) >= 10
    m = OccupancyGridMap.load(lay.nav_map_yaml())
    blocked_before = []
    for name, x, y, yaw in stops:
        p = (x, y, 0.3)
        c = L.add(p, L.q_rot(L.yaw_quat(yaw), D))
        c2 = walls.pull_in(p, c)
        if not m.line_of_sight(c[0], c[1], x, y) or not m.is_free(c[0], c[1]):
            blocked_before.append(name)
        d = L.CameraDirector("spot", D, lambda req: "gz", log=lambda s: None, walls=walls, t0=-10.0)
        d.on_robot(0.0, p, L.yaw_quat(yaw))
        cam = L.add(p, L.q_rot(L.yaw_quat(yaw), d.eff))
        assert cam == pytest.approx(c2, abs=1e-6)
        if c2 != c:
            assert math.hypot(cam[0] - x, cam[1] - y) >= 0.6 - 1e-9 and cam[2] == pytest.approx(c[2])
            corr = lay["corridor"]["interior"]
            if corr["x_min"] < x < corr["x_max"]:                      # inside the corridor, off its walls
                assert corr["y_min"] + 0.29 < cam[1] < corr["y_max"] - 0.29 and cam[0] < corr["x_max"] - 0.29
            if name != "dock":    # dock: the 2-D map also blocks the 0.72 m backboard the camera looks over
                assert m.line_of_sight(cam[0], cam[1], x, y), name      # robot visible again
    # what the old fixed offset ran into (the report: dock, G07, G08), now pulled in
    assert {"dock", "G07", "G08"} <= set(blocked_before)
    pulled = {n for n, x, y, yaw in stops
              if walls.pull_in((x, y, 0.3), L.add((x, y, 0.3), L.q_rot(L.yaw_quat(yaw), D))) !=
              L.add((x, y, 0.3), L.q_rot(L.yaw_quat(yaw), D))}
    assert {"dock", "G07", "G08"} <= pulled
    assert "G01" not in pulled                                       # open hall: the default view as it is


def test_walls_never_touch_a_view_the_user_chose():
    lay, stops = _stops()
    walls = L.WallMap(L.layout_walls(lay))
    _, x, y, yaw = [s for s in stops if s[0] == "G07"][0]
    sent = []
    d = L.CameraDirector("spot", D, lambda req: sent.append(req) or "gz", log=lambda s: None, walls=walls, t0=-10.0)
    d.view, d.view_auto = (-2.5, 0.0, 1.6), False                    # e.g. adopted from the mouse
    d.on_robot(0.0, (x, y, 0.3), L.yaw_quat(yaw))
    assert sent == []
    d.on_key(1.0, L.QT_KEY_Z)                                        # 'z': default view, walls again
    assert d.view_auto and sent[-1].mode == "FOLLOW" and sent[-1].offset != pytest.approx(D)
    assert L.layout_walls({}) == [] and not L.WallMap([])


def test_walls_follow_the_robot_turning_towards_a_wall_in_the_sim():
    lay, stops = _stops()
    _, x, y, yaw = [s for s in stops if s[0] == "G07"][0]
    r = Robot([(10.0, 0.0, 0.0), (math.pi / 2 / 0.5, 0.0, 0.5), (20.0, 0.0, 0.0)], start=(x, y, yaw - math.pi / 2))
    s = Sim(r, fps=30.0, walls=L.WallMap(L.layout_walls(lay)))
    s.run(40.0)
    assert s.sent and all(k == ("USE_LAST", "pos") for k in s.kinds()) and s.d.stats["holds"] == 0
    cam = s.gui.cam
    assert -1.8 + 0.25 < cam[1] < 1.8 - 0.25                         # inside the corridor
    assert s.d.view == D and s.d.view_auto                           # the view itself is still the default


# ============================================================================
# the program: engagement + loop with fake inputs / transports
# ============================================================================
ECHO_FOLLOW = 'track_mode: FOLLOW\nfollow_target {\n  name: "spot"\n}\nfollow_offset {\n  x: -2.5\n  z: 1.6\n}\n'


class FakeGz:
    """Scripted gz CLI; /gui/currently_tracked shows the mode of the last `gz topic -p` (TRACK or FOLLOW)."""

    def __init__(self, echo=ECHO_FOLLOW):
        self.calls, self.echo = [], echo

    def run(self, cmd, timeout):
        self.calls.append(cmd)
        a = cmd[1:]
        if a[:2] == ["topic", "-l"]:
            return 0, "/clock\n/model/spot/odometry\n/gui/camera/pose\n/gui/track\n"
        if a[:2] == ["topic", "-e"]:
            if self.echo and self.sent() and self.sent()[-1].startswith("track_mode: TRACK"):
                return 0, 'track_mode: TRACK\ntrack_target {\n  name: "spot"\n}\n'
            return 0, self.echo
        if a[:2] == ["topic", "-t"]:
            return 0, ""
        return 1, "?"

    def sent(self):
        return [c[7] for c in self.calls if c[1:3] == ["topic", "-t"]]


class FakeInputs:
    def __init__(self, script, clock):
        self.script, self.clock, self.closed = list(script), clock, False

    def ok(self):
        return bool(self.script)

    def poll(self, timeout):
        self.clock["t"] += 0.1
        item = self.script.pop(0)
        return [(k, self.clock["t"], d) for k, d in item] if item else []

    def close(self):
        self.closed = True


def loco(rs="NORMAL", mode="WALK", roll=0.0, pitch=0.0):
    return {"recovery_state": rs, "mode": mode, "roll": roll, "pitch": pitch}


def _prog(g, script, link_factory=None, **kw):
    clock = {"t": 0.0}
    lines = []
    inp = FakeInputs(script, clock)
    f = fc.FollowCamera("gz", "spot", "plant", D, 0.02, runner=g.run, clock=lambda: clock["t"], sleep=lambda dt: None,
                        log=lines.append, inputs_factory=lambda: inp, link_factory=link_factory, threaded=False, **kw)
    return f, lines, inp


def test_program_engages_with_both_gains_and_runs_the_director_over_the_cli():
    g = FakeGz()
    script = [[("loco", loco())]] * 3 + [[("key", L.QT_KEY_C)]] + [[]] * 5 + [[("key", L.QT_KEY_Z)]] + [[]] * 3 \
        + [[("loco", loco("TUCK", "RECOVERING", roll=3.0))]] * 3 + [[("loco", loco("NORMAL", "STAND"))]] * 15
    f, lines, inp = _prog(g, script)
    assert f.main() == 0 and inp.closed
    sent = g.sent()
    assert sent[0] == fc.follow_request("spot", D).text()                       # engage: FOLLOW + both gains
    assert sent[1].startswith("track_mode: TRACK")                               # 'c'
    assert sent[2].startswith('track_mode: FOLLOW, follow_target: {name: "spot"}')  # 'z'
    assert "track_mode: USE_LAST" in sent[3] and "track_pgain: 2e-05" in sent[3]   # fall: hold, look-at kept
    assert "follow_pgain: 0.05" in sent[3]
    assert "track_mode: USE_LAST" in sent[-1] and "track_pgain: 0.02" in sent[-1]  # upright: follow again
    assert any("camera control on" in x and "via gz topic -p" in x for x in lines)
    assert [c for c in g.calls if c[1:3] == ["topic", "-e"]]                    # currently_tracked polled
    assert "camera control ended" in lines[-1]


class FakeNode:
    def __init__(self, ok=True):
        self.subs, self.published, self.ok = {}, [], ok

    def advertise(self, topic, typ):
        assert topic == fc.TRACK_TOPIC and typ is FakeTrackMsg
        node = self

        class Pub:
            def publish(self, m):
                node.published.append(m)
                return node.ok
        return Pub()

    def subscribe(self, typ, topic, cb):
        self.subs[topic] = (typ, cb)
        return True

    def unsubscribe(self, topic):
        self.subs.pop(topic, None)


def fake_gz_modules(node, versions=(14, 11)):
    vt, vm = versions
    pose_t = type("Pose", (), {})
    mods = {f"gz.transport{vt}": types.SimpleNamespace(Node=lambda: node),
            f"gz.msgs{vm}.cameratrack_pb2": types.SimpleNamespace(CameraTrack=FakeTrackMsg),
            f"gz.msgs{vm}.pose_pb2": types.SimpleNamespace(Pose=pose_t)}

    def load(name):
        if name in mods:
            return mods[name]
        raise ImportError(f"No module named '{name}'")
    return load


def test_gz_link_publishes_and_queues_camera_and_status():
    node = FakeNode()
    link = fc.GzGuiLink(modules=fake_gz_modules(node), clock=lambda: 5.0)
    assert link.version == (14, 11) and set(node.subs) == {fc.GUI_TOPIC, fc.TRACKED_TOPIC}
    assert link.publish(L.TrackRequest("USE_LAST", "spot", D, 0.05, L.HOLD_GAIN))
    m = node.published[-1]
    assert m.track_mode == FakeTrackMsg.USE_LAST and m.follow_target.name == "" and m.track_pgain == L.HOLD_GAIN
    pose = types.SimpleNamespace(position=types.SimpleNamespace(x=1.0, y=2.0, z=3.0))
    node.subs[fc.GUI_TOPIC][1](pose)
    st = FakeTrackMsg()
    st.track_mode = 1
    st.track_target.name = "spot"
    node.subs[fc.TRACKED_TOPIC][1](st)
    evs = link.drain()
    assert evs[0] == ("cam", 5.0, ((1.0, 2.0, 3.0), "gz")) and evs[1][2]["mode"] == "TRACK" and link.tracked_seen
    assert link.drain() == []
    link.close()
    assert node.subs == {}

    def nothing(name):
        raise ImportError(name)
    with pytest.raises(ImportError):
        fc.GzGuiLink(modules=nothing)


def test_program_uses_the_bindings_and_falls_back_to_the_cli_when_the_gui_never_answers():
    g = FakeGz()
    node = FakeNode()
    script = [[("loco", loco())]] * 2 + [[("key", L.QT_KEY_Z)]] + [[]] * 20 + [[("key", L.QT_KEY_C)]] + [[]] * 40 \
        + [[("key", L.QT_KEY_C)]] + [[]] * 3
    f, lines, _ = _prog(g, script, link_factory=lambda: fc.GzGuiLink(modules=fake_gz_modules(node), clock=lambda: 0.0))
    assert f.main() == 0
    assert node.published and node.published[0].track_mode == FakeTrackMsg.FOLLOW       # 'z' through the bindings
    assert any("via gz-transport python 14" in x for x in lines)
    assert any("get no answer from the GUI" in x for x in lines)                    # no echo twice -> CLI
    assert f.link is None and len(node.published) == 2                              # 'z', 'c'; then the CLI:
    # the GUI never got the TRACK, its status still says FOLLOW -> following (gains again), last 'c' -> TRACK
    assert g.sent()[1].startswith("track_mode: USE_LAST") and g.sent()[-1].startswith("track_mode: TRACK")
    assert any("follow switched on in the GUI" in x for x in lines)


def test_program_without_ros_or_with_errors_never_fails():
    g = FakeGz()
    lines = []

    def no_ros():
        raise ImportError("No module named 'rclpy'")
    f = fc.FollowCamera("gz", "spot", "plant", D, 0.02, runner=g.run, clock=lambda: 0.0, sleep=lambda dt: None,
                        log=lines.append, inputs_factory=no_ros, threaded=False)
    assert f.main() == 0 and "camera control off" in lines[-1]

    class Broken(FakeInputs):
        def poll(self, timeout):
            raise RuntimeError("context invalid")
    f2, lines2, _ = _prog(g, [[]])
    f2.inputs_factory = lambda: Broken([[]], {"t": 0.0})
    assert f2.main() == 0 and any("camera control stopped" in x for x in lines2)
    g3 = FakeGz(echo="")                                   # follow mode never confirmed: no loop
    called = []
    f3 = fc.FollowCamera("gz", "spot", "plant", D, 0.02, runner=g3.run, clock=lambda: 0.0, sleep=lambda dt: None,
                         log=lines.append, attempts=1, inputs_factory=lambda: called.append(1))
    assert f3.main() == 0 and not called


def test_main_arguments(monkeypatch, capsys):
    seen = {}

    class Rec(fc.FollowCamera):
        def main(self):
            seen.update(offset=self.offset, pos=self.pos_gain, cfg=dict(self.cam_cfg), inputs=self.inputs_factory,
                        walls=self.walls)
            return 0
    monkeypatch.setattr(fc, "FollowCamera", Rec)
    assert fc.main(["--gz", "gz", "--offset=-3,0.5,2", "--pgain", "0.03", "--tilt-hold", "35", "--ros-args",
                    "-r", "__node:=follow_camera"]) == 0
    assert seen["offset"] == (-3.0, 0.5, 2.0) and seen["pos"] == 0.03 and seen["cfg"]["tilt_hold_deg"] == 35.0
    assert seen["cfg"]["adjust"] is True and seen["cfg"]["fall_guard"] is True and seen["inputs"] is not None
    assert seen["walls"] is not None and len(seen["walls"].segments) >= 10        # installed layout
    assert fc.main(["--gz", "gz", "--no-mouse", "--no-walls", "--no-fall-guard"]) == 0
    assert seen["cfg"]["adjust"] is False and seen["cfg"]["fall_guard"] is False and seen["walls"] is None
    assert fc.main(["--gz", "gz", "--once"]) == 0 and seen["inputs"] is None
    assert fc.main(["--gz", "gz", "--layout", "/no/such/layout.yaml"]) == 0
    assert seen["walls"] is None and "walls not loaded" in capsys.readouterr().out
    assert fc.main(["--gz", "gz", "--hold-gain", "0"]) == 0 and seen["cfg"]["hold_gain"] > L.MIN_GUI_GAIN


def test_ros_inputs_against_the_stand_ins(monkeypatch):
    import json
    import sys

    import test_plant_nodes_stubbed as stub
    for name, m in stub._modules().items():
        monkeypatch.setitem(sys.modules, name, m)
    W = stub.World()
    monkeypatch.setattr(stub, "W", W)
    rclpy = sys.modules["rclpy"]
    queue = []

    def spin_once(node, timeout_sec=None):
        if queue:
            topic, m = queue.pop(0)
            W.deliver(topic, m)
    monkeypatch.setattr(rclpy, "spin_once", spin_once, raising=False)
    inp = fc.RosInputs(["--ros-args", "-r", "__node:=follow_camera"], clock=lambda: 7.0)
    topics = {s.topic: s for s in W.subs}
    assert set(topics) == {"/locomotion/state", "/odom", "/gui/camera/pose", "/keyboard/keypress"}
    assert topics["/odom"].qos.reliability == "be" and topics["/gui/camera/pose"].qos.reliability == "be"
    od = stub.odom_msg(1.0, 2.0, 0.5)
    pose = stub.Pose()
    pose.position.x = 3.0
    queue += [("/locomotion/state", stub.String(data=json.dumps({"mode": "WALK", "roll": 0.1}))),
              ("/locomotion/state", stub.String(data="not json")), ("/odom", od), ("/gui/camera/pose", pose),
              ("/keyboard/keypress", stub.Int32(data=67))]
    evs = inp.poll(0.1)
    assert [e[0] for e in evs] == ["loco", "robot", "cam", "key"] and all(e[1] == 7.0 for e in evs)
    assert evs[1][2][0] == (1.0, 2.0, 0.0) and evs[1][2][1][3] == pytest.approx(math.sin(0.25))
    assert evs[2][2] == ((3.0, 0.0, 0.0), "ros") and evs[3][2] == 67
    assert inp.ok() and inp.poll(0.1) == []
    inp.close()
    assert not rclpy.ok()
