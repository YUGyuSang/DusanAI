"""ROS-free LocomotionController with fake inputs (kinematic fake plant), and
the ROS node against strict rclpy / message stand-ins."""
import array
import math
import os
import struct
import sys
import threading
import time
import types
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dusan_common import robot_model as rm
from module1_locomotion import controller as C
from module1_locomotion import kinematics as kin
from module1_locomotion import qp as Q

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.01


@pytest.fixture(scope="module", autouse=True)
def _single_thread_blas():
    """Inline MPC solves: one BLAS thread is fastest and immune to thread
    oversubscription on a loaded machine (restored afterwards)."""
    before = Q.blas_threads()
    Q.limit_blas_threads(1)
    yield
    if before:
        Q.limit_blas_threads(before)


class FakeMap:
    """Minimal ElevationMap stand-in: height_at(x, y) of an analytic surface."""

    def __init__(self, fn):
        self.fn = fn

    def height_at(self, x, y):
        out = self.fn(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return float(out) if np.ndim(out) == 0 else out


class FakePlanner:
    def __init__(self):
        self.calls = 0

    def select(self, emap, nominal_xy, fallback_z):
        self.calls += 1
        z = emap.height_at(float(nominal_xy[0]), float(nominal_xy[1]))
        return np.array([nominal_xy[0], nominal_xy[1], z]), 0.0, True


class Sim:
    """Kinematic fake plant: joints track the targets perfectly, the body follows
    the commanded body pose while standing/walking, and FK otherwise.  An ideal
    (load-independent) servo does not deflect, so the torque feed-forward of
    the compliant gz PID servo is off unless a test enables it."""

    def __init__(self, params=None, floor=0.0, emap=None, yaw=0.0):
        # the body follows the commanded body pose: the MPC mode (the stable mode has no
        # body target and is tested with a foot-carried plant in test_stable_gait.py)
        p = {"use_mpc_thread": False, "body_lead_time": DT, "torque_feedforward": 0.0, "locomotion_mode": "mpc"}
        p.update(params or {})
        self.ctrl = C.LocomotionController(p)
        self.ctrl.planner = FakePlanner() if emap is not None else None
        self.ctrl._max_height_along = None
        self.floor = floor
        self.emap = emap
        self.st = C.RobotState()
        self.st.pos = np.array([0.0, 0.0, floor + 0.56])
        self.st.rpy = np.array([0.0, 0.0, yaw])
        self.st.q = np.zeros(12)
        self.t = 0.0
        self.dbg = None
        self.qs = []

    def run(self, duration, cmd=(0.0, 0.0, 0.0), terrain=None, override=None):
        n = int(round(duration / DT))
        for _ in range(n):
            q, dbg = self.ctrl.step(self.t, self.st, cmd, terrain, self.emap)
            assert np.all(np.isfinite(q))
            assert np.all(q >= kin.JOINT_LOWER - 1e-9) and np.all(q <= kin.JOINT_UPPER + 1e-9)
            self.qs.append(q.copy())
            self.dbg = dbg
            self.st.q = q.copy()
            if override is not None:
                override(self)
            elif self.ctrl.mode in ("STAND", "WALK", "SIT") and self.ctrl.body_cmd is not None:
                pos, rpy = self.ctrl.body_cmd
                self.st.vel_world = (np.asarray(pos) - self.st.pos) / DT
                drpy = (np.asarray(rpy) - self.st.rpy) / DT
                drpy[2] = math.remainder(float(rpy[2] - self.st.rpy[2]), 2 * math.pi) / DT
                self.st.omega_body = drpy                    # small roll/pitch: body rates ~ Euler rates
                self.st.pos = np.asarray(pos, dtype=float).copy()
                self.st.rpy = np.asarray(rpy, dtype=float).copy()
            else:
                fk = kin.fk_all(q)
                z = self.floor - fk[:, 2].min() + rm.FOOT_RADIUS
                self.st.vel_world = np.zeros(3)
                self.st.pos = np.array([self.st.pos[0], self.st.pos[1], z])
                self.st.rpy = np.array([0.0, 0.0, self.st.rpy[2]])
            self.t += DT
        return self


def _stand(sim):
    sim.run(2.6)
    assert sim.ctrl.mode == "STAND"
    return sim


def test_yaml_matches_parameter_defaults():
    from module1_locomotion import controller
    path = os.path.join(HERE, "..", "config", "locomotion.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params = data["locomotion_controller"]["ros__parameters"]
    # node-only parameters live in the node module (imports rclpy): parse its NODE_PARAMS block
    src = open(os.path.join(HERE, "..", "module1_locomotion", "locomotion_controller_node.py"),
               encoding="utf-8").read()
    block = src.split("NODE_PARAMS = {", 1)[1].split("\n}", 1)[0]
    node_keys = {line.split(":")[0].strip().strip('"') for line in block.splitlines() if ":" in line}
    expected = set(controller.DEFAULT_PARAMS) | node_keys
    assert set(params) == expected
    for k, v in controller.DEFAULT_PARAMS.items():
        assert type(params[k]) is type(v) or (isinstance(v, list) and all(isinstance(x, float) for x in params[k])), k
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            assert params[k] == pytest.approx(v), k


def test_startup_stands_up_from_straight_legs():
    sim = Sim()
    sim.run(1.0)
    assert sim.ctrl.mode == "STANDING_UP"
    sim.run(1.6)
    assert sim.ctrl.mode == "STAND"
    assert np.allclose(sim.qs[-1], kin.stand_joint_angles(), atol=1e-6)
    # the interpolation is smooth: no joint target jumps
    steps = np.abs(np.diff(np.array(sim.qs), axis=0)).max()
    assert steps < 12.0 * DT + 1e-9
    assert sim.dbg["body_height"] == pytest.approx(0.34, abs=0.01)


def test_walking_produces_finite_limited_targets_and_moves():
    sim = _stand(Sim())
    sim.run(4.0, cmd=(0.4, 0.0, 0.3))
    d = sim.dbg
    assert d["mode"] == "WALK" and d["gait"] == "trot"
    assert d["mpc_status"] in ("solved", "solved_inaccurate")
    assert d["mpc_horizon_s"] == pytest.approx(5.0)
    assert sim.st.pos[0] > 0.8
    assert sim.st.rpy[2] > 0.6
    assert abs(sim.st.pos[2] - 0.34) < 0.02
    assert d["ik_reachable"]
    # 4 contacts reported as list of bools, two in stance for trot (or 4 at a beat)
    assert len(d["contacts"]) == 4 and sum(d["contacts"]) in (2, 3, 4)
    # torque prediction summary present
    assert len(d["torque_horizon_max"]) == 12 and 0.0 < d["torque_ratio"] < 1.0
    # zero command -> back to STAND after the idle time
    sim.run(2.0)
    assert sim.ctrl.mode == "STAND"
    assert np.all(np.abs(np.diff(np.array(sim.qs), axis=0)) <= 12.0 * DT + 1e-9)


def test_velocity_and_acceleration_limits():
    sim = _stand(Sim())
    hist = []
    for _ in range(300):
        sim.run(DT, cmd=(3.0, -2.0, 5.0))
        hist.append(sim.ctrl.cmd_f.copy())
    hist = np.array(hist)
    lim = np.array([0.6, 0.3, 0.9])
    assert np.all(np.abs(hist) <= lim + 1e-9)
    acc = np.abs(np.diff(hist, axis=0)) / DT
    assert np.all(acc <= np.array([0.8, 0.5, 1.5]) + 1e-6)
    # at the limits, saturation-aware scaling may slow the robot down (never below 30 %)
    s = sim.ctrl.speed_scale
    assert 0.3 <= s <= 1.0
    assert np.all(np.abs(hist[-1]) >= 0.3 * lim - 1e-9)
    assert np.all(np.sign(hist[-1]) == np.array([1.0, -1.0, 1.0]))


def test_gait_auto_selection_and_forced_gait():
    sim = _stand(Sim({"gait_mode": "auto"}))
    crawl_info = {"recommended_gait": "crawl", "class": "stairs", "ahead_clear": True}
    sim.run(2.5, cmd=(0.5, 0.0, 0.0), terrain=crawl_info)
    assert sim.ctrl.gait.name == "crawl"
    assert sim.ctrl.cmd_f[0] <= 0.25 + 1e-9                      # crawl limit
    assert sim.dbg["terrain"] == "stairs" and sim.dbg["gait_auto"]
    trot_info = {"recommended_gait": "trot", "class": "flat", "ahead_clear": True}
    sim.run(0.5, cmd=(0.5, 0.0, 0.0), terrain=trot_info)
    assert sim.ctrl.gait.name == "crawl"                           # hysteresis
    sim.run(2.5, cmd=(0.5, 0.0, 0.0), terrain=trot_info)
    assert sim.ctrl.gait.name == "trot"
    sim.ctrl.handle_command("gait:crawl", sim.t, sim.st)
    sim.run(2.0, cmd=(0.5, 0.0, 0.0), terrain=trot_info)
    assert sim.ctrl.gait.name == "crawl" and not sim.dbg["gait_auto"]


def test_estop_freezes_and_release_stands_up():
    sim = _stand(Sim())
    sim.run(1.0, cmd=(0.3, 0.0, 0.0))
    sim.ctrl.handle_command("estop", sim.t, sim.st)
    sim.run(0.3, cmd=(0.3, 0.0, 0.0))
    frozen = sim.qs[-1].copy()
    sim.run(0.5, cmd=(0.5, 0.0, 0.0))
    assert sim.ctrl.mode == "ESTOP" and np.allclose(sim.qs[-1], frozen)
    sim.ctrl.handle_command("walk", sim.t, sim.st)                 # ignored in ESTOP
    assert sim.ctrl.mode == "ESTOP"
    sim.ctrl.handle_command("release", sim.t, sim.st)
    assert sim.ctrl.mode == "STANDING_UP"
    sim.run(2.6)
    assert sim.ctrl.mode == "STAND"


def test_sit_and_stand_commands():
    sim = _stand(Sim())
    sim.ctrl.handle_command("sit", sim.t, sim.st)
    sim.run(2.2)
    assert sim.ctrl.mode == "SIT"
    assert sim.st.pos[2] == pytest.approx(rm.SIT_BODY_HEIGHT, abs=0.01)
    sim.ctrl.handle_command("stand", sim.t, sim.st)
    sim.run(2.6)
    assert sim.ctrl.mode == "STAND" and sim.st.pos[2] == pytest.approx(0.34, abs=0.01)
    # sit while walking: stops first, then sits
    sim.run(1.0, cmd=(0.3, 0.0, 0.0))
    sim.ctrl.handle_command("sit", sim.t, sim.st)
    sim.run(3.0)
    assert sim.ctrl.mode == "SIT"


def test_fall_detection_switches_to_recovery():
    sim = _stand(Sim())

    def tipped(s):
        s.st.rpy = np.array([80 * math.pi / 180, 0.0, 0.0])
        s.st.pos = np.array([s.st.pos[0], s.st.pos[1], 0.15])
        s.st.omega_body = np.zeros(3)

    sim.run(0.5, override=tipped)
    assert sim.ctrl.mode in ("FALLEN", "RECOVERING")
    sim.run(1.5, override=tipped)
    assert sim.ctrl.mode == "RECOVERING"
    ev = sim.ctrl.pop_events()
    assert any(e["event"] == "fall" for e in ev)
    assert sim.dbg["fall_count"] == 1 and sim.dbg["recovery_attempts"] >= 1
    # commands other than estop / recover are ignored while recovering
    sim.ctrl.handle_command("walk", sim.t, sim.st)
    assert sim.ctrl.mode in ("FALLEN", "RECOVERING")
    # E-STOP + release during the recovery: fall detection must work again afterwards
    sim.ctrl.handle_command("estop", sim.t, sim.st)
    sim.run(0.2, override=tipped)
    sim.ctrl.handle_command("release", sim.t, sim.st)
    assert sim.ctrl.mode == "STANDING_UP" and sim.ctrl.recovery.state == "NORMAL"
    sim.run(0.5, override=tipped)
    assert sim.ctrl.mode in ("FALLEN", "RECOVERING") and sim.dbg["fall_count"] == 2


def test_failed_assisted_reset_with_robot_standing_continues():
    """Field report: fell mid-mission, got up, then stood still - the mission
    never continued.  A failed gz set_pose left the recovery FAILED (mode
    FALLEN) while the stand posture of ASSISTED_RESET held the robot up, and
    the map under the base was wrong after the tumble.  Now the robot standing
    on its legs (height above the measured feet) counts as recovered: the
    controller stands up and is ready again."""
    sim = _stand(Sim({"assisted_reset": True}))
    sim.ctrl.reset_fn = lambda *a: True
    rec = sim.ctrl.recovery
    assert rec.p.failed_retry_time == pytest.approx(sim.ctrl.p["recovery_retry_time"]) and rec.p.failed_retry_time > 0

    def tipped(s):
        s.st.rpy = np.array([80 * math.pi / 180, 0.0, 0.0])
        s.st.pos = np.array([s.st.pos[0], s.st.pos[1], 0.15])
        s.st.omega_body = np.zeros(3)

    sim.run(1.0, override=tipped)
    assert sim.ctrl.mode in ("FALLEN", "RECOVERING")
    rec.attempt = rec.p.max_attempts
    rec._attempt_failed(sim.t, sim.st.q, (0.0, 0.0, 1.4, 0.0), 0.0, 0.0, 0.15, 0.0)
    assert rec.waiting_for_reset
    sim.run(1.0)                                                 # stand posture held, body up on its legs
    sim.ctrl.on_pose_reset_done(sim.t, sim.st, False)             # the teleport failed
    sim.emap = FakeMap(lambda x, y: 0.2 + 0.0 * x)               # map artefact under the base (0.14 m "height")
    sim.run(0.1)
    assert rec.state == "FAILED" and sim.ctrl.mode == "FALLEN"
    for _ in range(200):
        sim.run(0.02)
        if sim.ctrl.mode == "STANDING_UP":
            break
    assert sim.ctrl.mode == "STANDING_UP" and rec.state == "NORMAL"
    sim.emap = None
    sim.run(3.0)
    assert sim.ctrl.mode == "STAND"
    assert "recovered" in [e["event"] for e in sim.ctrl.pop_events()]
    # and it walks again
    sim.run(2.0, cmd=(0.3, 0.0, 0.0))
    assert sim.ctrl.mode == "WALK"


def test_recovery_retry_only_with_auto_recover():
    ctrl = C.LocomotionController({"use_mpc_thread": False, "recovery_retry_time": 7.0})
    assert ctrl.recovery.p.failed_retry_time == pytest.approx(7.0)
    ctrl = C.LocomotionController({"use_mpc_thread": False, "auto_recover": False})
    assert ctrl.recovery.p.failed_retry_time == 0.0


def test_corridor_lower_level_floor():
    sim = _stand(Sim(floor=-0.88, yaw=math.pi))
    assert sim.ctrl.ground_z == pytest.approx(-0.88, abs=0.01)
    sim.run(2.0, cmd=(0.3, 0.0, 0.0))
    assert sim.dbg["body_height"] == pytest.approx(0.34, abs=0.02)
    assert sim.st.pos[2] == pytest.approx(-0.88 + 0.34, abs=0.02)
    assert sim.st.pos[0] < -0.3                                   # facing -x


def test_elevation_map_ramp_raises_footholds_and_body():
    ramp = FakeMap(lambda x, y: np.clip(0.25 * (x - 0.6), 0.0, 0.3) + 0.0 * y)
    sim = _stand(Sim(emap=ramp))
    pitch = []
    for _ in range(500):
        sim.run(DT, cmd=(0.4, 0.0, 0.0))
        pitch.append(sim.ctrl.terrain_pitch)
    assert sim.st.pos[0] > 1.5
    surf = ramp.height_at(sim.st.pos[0], sim.st.pos[1])
    assert sim.st.pos[2] == pytest.approx(surf + 0.34, abs=0.05)
    assert min(pitch) < -0.15                                     # nose up on the ramp (slope 0.245 rad)
    assert sim.ctrl.planner.calls > 0
    front = sim.ctrl.foot_w[:2]
    assert np.all(front[:, 2] > 0.05)


def test_missing_map_values_fall_back_to_feet():
    holes = FakeMap(lambda x, y: np.full(np.broadcast(x, y).shape, np.nan))
    sim = _stand(Sim(emap=holes))
    sim.run(2.0, cmd=(0.3, 0.0, 0.0))
    assert sim.ctrl.ground_z == pytest.approx(0.0, abs=0.01)
    assert sim.st.pos[2] == pytest.approx(0.34, abs=0.02)


def test_unknown_cells_under_body_do_not_tilt_the_body():
    """Regression (Gazebo plant world): right after spawn the depth camera has
    not seen the cells under the body; with a swing foot in the air the pitch
    target used to follow the foot heights and wobble by 5-15 deg."""
    for mode in ("mpc", "stable"):
        ctrl = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": mode})
        ctrl.ground_z = -0.88                          # corridor lower level
        feet = np.array(rm.nominal_foot_positions(), dtype=float)
        feet[:, 2] = -0.88 + rm.FOOT_RADIUS
        feet[0, 2] += 0.12                             # front-left foot mid-swing
        feet[3, 2] += 0.12                             # rear-right foot mid-swing
        unknown = np.full(4, np.nan)
        assert ctrl._terrain_pitch(unknown, feet) == pytest.approx(0.0, abs=1e-9)
        half = np.array([np.nan, -0.88, np.nan, -0.88])
        assert ctrl._terrain_pitch(half, feet) == pytest.approx(0.0, abs=1e-9)
        # a real slope still produces pitch (front higher -> nose up = negative pitch)
        ramp = np.array([-0.78, -0.78, -0.88, -0.88])
        assert ctrl._terrain_pitch(ramp, feet) < -0.1


def test_push_and_reset_actions():
    sim = _stand(Sim())
    a = sim.ctrl.handle_command("push:40,-25", sim.t, sim.st)
    assert a == {"action": "push", "force": (40.0, -25.0, 0.0), "duration": 0.2}
    a = sim.ctrl.handle_command("push", sim.t, sim.st)
    assert a["force"] == (0.0, 150.0, 0.0)
    assert sim.ctrl.handle_command("push:bad", sim.t, sim.st) is None
    a = sim.ctrl.handle_command("reset_pose", sim.t, sim.st)
    assert a == {"action": "reset_pose", "height_offset": 0.5}
    sim.ctrl.notify_pose_reset(sim.t, sim.st)
    assert sim.ctrl.mode == "STANDING_UP"
    sim.run(3.8)
    assert sim.ctrl.mode == "STAND"


def test_assisted_reset_callback_is_wired():
    calls = []
    ctrl = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "mpc"}, reset_fn=lambda *a: calls.append(a))
    ctrl._assisted_reset(1.0, 2.0, 0.8, 0.1)
    assert calls == [(1.0, 2.0, 0.8, 0.1)]


def test_pose_reset_result_starts_the_settle_window():
    """The settle window of a gz set_pose starts when the teleport has been
    executed (on_pose_reset_done), not at the request (~0.3-1 s earlier)."""
    sim = _stand(Sim({"assisted_reset": True}))
    a = sim.ctrl.handle_command("reset_pose", sim.t, sim.st)
    assert a["action"] == "reset_pose" and sim.ctrl.mode == "STAND"
    sim.ctrl.on_pose_reset_done(sim.t, sim.st, False)             # failed: robot not moved, carry on
    assert sim.ctrl.mode == "STAND"
    sim.ctrl.on_pose_reset_done(sim.t, sim.st, True)
    assert sim.ctrl.mode == "STANDING_UP"
    assert sim.ctrl.reset_settle_until == pytest.approx(sim.t + sim.ctrl.p["reset_settle_time"])
    sim.run(3.8)
    assert sim.ctrl.mode == "STAND"
    sim.ctrl.handle_command("estop", sim.t, sim.st)                # E-STOP engaged meanwhile: stays
    sim.ctrl.on_pose_reset_done(sim.t, sim.st, True)
    assert sim.ctrl.mode == "ESTOP"
    # assisted reset: the reset_fn result is passed on; True -> the recovery waits for the result
    ctrl = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "mpc", "assisted_reset": True},
                                  reset_fn=lambda *a: True)
    assert ctrl._assisted_reset(1.0, 2.0, 0.8, 0.1) is True
    rec = ctrl.recovery
    assert rec.p.reset_confirm_timeout == pytest.approx(ctrl.p["reset_confirm_timeout"])
    rec.attempt = rec.p.max_attempts
    rec._attempt_failed(10.0, np.zeros(12), (1.0, 2.0, 1.5, 0.0), 1.0, 2.0, 0.1, 0.5)
    assert rec.waiting_for_reset
    ctrl.on_pose_reset_done(10.7, None, True)
    assert not rec.waiting_for_reset and rec.state == "ASSISTED_RESET" and rec._t_state == 10.7
    rec.attempt = rec.p.max_attempts
    rec._attempt_failed(20.0, np.zeros(12), (1.0, 2.0, 1.5, 0.0), 1.0, 2.0, 0.1, 0.5)
    assert rec.waiting_for_reset
    ctrl.on_pose_reset_done(20.5, None, False)                    # teleport failed: FAILED
    assert rec.state == "FAILED"


def test_torque_saturation_speed_scaling():
    ctrl = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "mpc"})
    fake = SimpleNamespace(ok=True, torque_ratio=0.95, solve_time=0.01)
    for _ in range(5):
        ctrl._accept_mpc(fake)
    assert ctrl.speed_scale == pytest.approx(0.9 ** 5)
    fake.torque_ratio = 0.5
    for _ in range(100):
        ctrl._accept_mpc(fake)
    assert ctrl.speed_scale == pytest.approx(1.0)


def test_threaded_mpc_worker():
    sim = Sim(params={"use_mpc_thread": True})
    try:
        sim.run(2.6)
        for _ in range(150):
            sim.run(DT, cmd=(0.3, 0.0, 0.0))
            time.sleep(0.002)
        assert sim.ctrl.mpc_result is not None
        assert sim.dbg["mpc_status"] in ("solved", "solved_inaccurate")
        assert sim.ctrl.worker.error is None
    finally:
        sim.ctrl.shutdown()


def test_organised_cloud_parser():
    rows, cols = 3, 4
    pts = np.zeros((rows, cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            pts[r, c] = (0.1 * c, 0.2 * r, np.nan if (r, c) == (1, 2) else 0.01 * (r + c))
    step = 16                                     # x y z + 4 padding bytes
    row_step = cols * step + 8                    # padded rows
    data = bytearray()
    for r in range(rows):
        for c in range(cols):
            data += struct.pack("<fff", *pts[r, c]) + b"\0\0\0\0"
        data += b"\0" * 8
    fields = [SimpleNamespace(name=n, offset=o, datatype=7, count=1) for n, o in (("x", 0), ("y", 4), ("z", 8))]
    msg = SimpleNamespace(height=rows, width=cols, fields=fields, point_step=step, row_step=row_step,
                          data=bytes(data), is_bigendian=False)
    xyz = C.organised_cloud_to_xyz(msg)
    assert xyz.shape == (rows, cols, 3)
    assert np.allclose(xyz[..., :2], pts[..., :2])
    assert np.isnan(xyz[1, 2, 2]) and xyz[2, 3, 2] == pytest.approx(0.05)


def test_crawl_walks_at_commanded_speed_without_drift():
    """Regression for the crawl: speed tracking and no sideways runaway
    (kinematic fake plant, forced crawl, 7 s)."""
    sim = _stand(Sim())
    sim.ctrl.handle_command("gait:crawl", sim.t, sim.st)
    ts, xs = [], []
    for _ in range(700):
        sim.run(DT, cmd=(0.25, 0.0, 0.0))
        ts.append(sim.t)
        xs.append(sim.st.pos.copy())
    ts, xs = np.array(ts), np.array(xs)
    assert sim.ctrl.gait.name == "crawl" and sim.dbg["mpc_status"] in ("solved", "solved_inaccurate")
    late = ts > ts[0] + 2.0
    speed = np.polyfit(ts[late], xs[late, 0], 1)[0]
    assert speed == pytest.approx(0.25, abs=0.04)
    assert np.max(np.abs(xs[:, 1])) < 0.2
    assert abs(sim.st.rpy[2]) < 0.05
    assert sim.ctrl.speed_scale > 0.9 and sim.dbg["ik_reachable"]


def test_real_terrain_modules_two_steps_crawl():
    """Integration with the terrain agent's ElevationMap / FootholdPlanner /
    max_height_along (skipped if those modules are absent): crawl up two
    10 cm steps built as an organised /elevation_map/points cloud."""
    em = pytest.importorskip("module1_locomotion.elevation_map")
    fp = pytest.importorskip("module1_locomotion.foothold_planner")
    res = 0.04
    X, Y = np.meshgrid(np.arange(-1.5, 4.0, res), np.arange(-1.5, 1.5, res))
    Z = np.where(X > 1.0, 0.10, 0.0) + np.where(X > 1.6, 0.10, 0.0)
    emap = em.ElevationMap.from_cloud_xyz(np.stack([X, Y, Z], -1).astype(np.float32))
    sim = Sim({"gait_mode": "auto"}, emap=emap)
    sim.ctrl.planner = fp.FootholdPlanner(search_radius=0.10)
    sim.ctrl._max_height_along = fp.max_height_along
    _stand(sim)
    info = {"recommended_gait": "crawl", "class": "stairs", "ahead_clear": True}
    apex_seen = []
    for _ in range(1000):
        sim.run(DT, cmd=(0.25, 0.0, 0.0), terrain=info)
        apex_seen += [tr.apex_z for tr in sim.ctrl.swing if tr is not None]
        if sim.st.pos[0] > 2.1:
            break
    assert sim.ctrl.mode == "WALK" and sim.ctrl.gait.name == "crawl"
    assert sim.st.pos[0] > 2.1                                    # crossed both steps
    assert sim.st.pos[2] == pytest.approx(0.20 + 0.34, abs=0.05)  # body follows the terrain
    assert np.all(sim.ctrl.foot_w[:, 2] > 0.10)                   # all feet on the steps
    assert max(apex_seen) > 0.20 + rm.FOOT_RADIUS                 # swing cleared the second edge
    assert sim.dbg["ik_reachable"]


def _lagged(tau):
    """Sim override: body pose AND joints follow the commands with a
    first-order lag ``tau`` (joint position servo lag)."""
    state = {}

    def override(s):
        c = s.ctrl
        if "q" not in state:
            state["q"] = s.st.q.copy()
        a = min(1.0, DT / tau)
        state["q"] = state["q"] + a * (s.st.q - state["q"])      # s.st.q was set to the command
        s.st.q = state["q"].copy()
        if c.mode in ("STAND", "WALK", "SIT") and c.body_cmd is not None:
            pos, rpy = c.body_cmd
            newp = s.st.pos + a * (np.asarray(pos) - s.st.pos)
            dr = np.asarray(rpy) - s.st.rpy
            dr[2] = math.remainder(float(dr[2]), 2 * math.pi)
            newr = s.st.rpy + a * dr
            s.st.vel_world = (newp - s.st.pos) / DT
            s.st.omega_body = (newr - s.st.rpy) / DT
            s.st.pos, s.st.rpy = newp, newr

    return override


@pytest.mark.parametrize("mode, ok", [("integrated", True), ("prediction", False)])
def test_walking_speed_robust_to_servo_lag(mode, ok):
    """The integrated body target keeps walking at about the commanded speed
    with a 50 ms servo lag (in this kinematic model the lag-shifted touchdown
    latch adds a forward bias of ~+30 %); the re-anchored MPC pose
    (prediction mode) collapses to a fraction of the command (documents why
    'integrated' is the default)."""
    sim = _stand(Sim(params={"body_target_mode": mode, "body_lead_time": 0.02}))
    ov = _lagged(0.05)
    ts, xs = [], []
    for _ in range(500):
        sim.run(DT, cmd=(0.4, 0.0, 0.0), override=ov)
        ts.append(sim.t)
        xs.append(sim.st.pos[0])
    ts, xs = np.array(ts), np.array(xs)
    late = ts > ts[0] + 2.5
    speed = np.polyfit(ts[late], xs[late], 1)[0]
    if ok:
        assert 0.32 < speed < 0.56
        assert sim.ctrl.mode == "WALK" and sim.dbg["ik_reachable"]
    else:
        assert speed < 0.25


def _ik_pose_plant(tau, sag_rp=(0.03, -0.02), sag_z=-0.01):
    """Sim override: the body follows the pose actually used for the leg IK
    (``ctrl.body_ik`` = target + IMU PD correction + height integrator) with a
    first-order lag ``tau`` (stiff joint servo + body), plus a constant
    roll / pitch / height sag (compliance).  The measured body rates are those
    of the lagged response at the sample instant - this closes the attitude
    PD loop, which the plain kinematic Sim (it follows ``body_cmd``) never does."""
    state = {}
    sag = np.array([sag_rp[0], sag_rp[1], 0.0])

    def override(s):
        c = s.ctrl
        pos_c, rpy_c = c.body_ik
        if "r" not in state:
            state["r"] = np.asarray(s.st.rpy, dtype=float) - sag
            state["p"] = np.asarray(s.st.pos, dtype=float) - np.array([0.0, 0.0, sag_z])
        a = 1.0 - math.exp(-DT / tau)
        target = np.asarray(rpy_c, dtype=float).copy()
        target[2] = state["r"][2] + math.remainder(float(target[2] - state["r"][2]), 2 * math.pi)
        state["r"] = state["r"] + a * (target - state["r"])
        p_old = state["p"].copy()
        state["p"] = state["p"] + a * (np.asarray(pos_c, dtype=float) - state["p"])
        s.st.omega_body = (target - state["r"]) / tau
        s.st.vel_world = (state["p"] - p_old) / DT
        s.st.rpy = state["r"] + sag
        s.st.pos = state["p"] + np.array([0.0, 0.0, sag_z])

    return override


@pytest.mark.parametrize("kd, tau, stable", [(None, 0.003, True), (None, 0.006, True), (None, 0.01, True),
                                             (None, 0.03, True), (0.02, 0.006, False)])
def test_attitude_pd_loop_with_lagged_body(kd, tau, stable):
    """Closed attitude loop: with the default kd_attitude the IMU roll/pitch
    PD settles for any servo/body lag and reduces the sag; the former default
    kd 0.02 limit-cycles at the control rate for a 6 ms lag (documents the
    reason for the default)."""
    params = {} if kd is None else {"kd_attitude": kd}
    sim = _stand(Sim(params=params))
    ov = _ik_pose_plant(tau)
    rolls, pitches, zs = [], [], []
    for _ in range(300):
        sim.run(DT, override=ov)
        rolls.append(sim.st.rpy[0])
        pitches.append(sim.st.rpy[1])
        zs.append(sim.st.pos[2])
    r, pt = np.array(rolls[-100:]), np.array(pitches[-100:])
    if stable:
        assert np.ptp(r) < 2e-3 and np.ptp(pt) < 2e-3
        # proportional action: the residual tilt is the sag / (1 + kp), same sign
        kp = sim.ctrl.p["kp_attitude"]
        assert r.mean() == pytest.approx(0.03 / (1.0 + kp), abs=3e-3)
        assert pt.mean() == pytest.approx(-0.02 / (1.0 + kp), abs=3e-3)
        # the height integrator works against the 1 cm sag
        assert abs(zs[-1] - 0.34) < abs(zs[100] - 0.34) and sim.ctrl.z_int > 0.0
        assert sim.ctrl.mode == "STAND" and sim.dbg["ik_reachable"]
    else:
        assert np.ptp(r) > 0.05


def test_mpc_reset_waits_for_running_worker_solve():
    """ConvexMPC.solve writes the warm start at its end: a reset requested
    from the control thread while a solve runs must be applied after it."""
    gate = threading.Event()

    class SlowMPC:
        def __init__(self):
            self._prev = None
            self.resets = 0

        def solve(self, *job):
            gate.wait(5.0)
            self._prev = "warm start of the old motion"
            return SimpleNamespace(t=0.0)

        def reset(self):
            self._prev = None
            self.resets += 1

    mpc = SlowMPC()
    w = C._MPCWorker(mpc)
    try:
        w.request_reset()                       # idle: immediate
        assert mpc.resets == 1
        assert w.submit(("job",))
        deadline = time.time() + 5.0
        while not w._busy and time.time() < deadline:
            time.sleep(0.001)
        w.request_reset()                       # busy: deferred
        assert mpc.resets == 1
        gate.set()
        while w.busy and time.time() < deadline:
            time.sleep(0.001)
        assert mpc.resets == 2 and mpc._prev is None
        assert w.poll() is not None and w.error is None
    finally:
        gate.set()
        w.stop()


def test_mpc_worker_errors_are_logged_once():
    logs = []
    sim = Sim(params={"use_mpc_thread": True})
    sim.ctrl.log = logs.append
    try:
        _stand(sim)

        def broken(*a, **k):
            raise RuntimeError("solver exploded")

        sim.ctrl.mpc.solve = broken
        for _ in range(60):
            sim.run(DT, cmd=(0.3, 0.0, 0.0))
            time.sleep(0.001)
        msgs = [m for m in logs if "solver exploded" in m]
        assert len(msgs) == 1
        assert sim.ctrl.mode == "WALK"           # keeps walking on the command fallback
    finally:
        sim.ctrl.shutdown()


def test_body_x_offset_puts_body_ahead_of_feet():
    sim = _stand(Sim(params={"body_x_offset": 0.03}))
    sim.run(1.5)
    c = sim.ctrl
    assert sim.st.pos[0] - c.foot_w[:, 0].mean() == pytest.approx(0.03, abs=3e-3)
    assert np.allclose(c.nominal_feet_b[:, 0], kin.SIGN_X * rm.NOMINAL_FOOT_X - 0.03)
    assert c._mpc_input(sim.t, sim.st).hold_xy[0] == pytest.approx(c.foot_w[:, 0].mean() + 0.03)


# ---------------------------------------------------------------------------
# interruptions (fall / E-stop) and stale MPC results
# ---------------------------------------------------------------------------
def test_interruption_resets_shaped_command_and_requests():
    """After an E-stop / release (or a fall) the next walk starts from zero
    command with the acceleration ramp, and a 'walk' request made before the
    interruption is not executed by surprise afterwards."""
    sim = _stand(Sim())
    sim.ctrl.handle_command("walk", sim.t, sim.st)
    sim.run(3.0, cmd=(0.6, 0.0, 0.0))
    assert sim.ctrl.cmd_f[0] > 0.4
    sim.ctrl.handle_command("estop", sim.t, sim.st)
    sim.run(0.2)
    sim.ctrl.handle_command("release", sim.t, sim.st)
    sim.run(2.6)
    assert sim.ctrl.mode == "STAND" and np.allclose(sim.ctrl.cmd_f, 0.0)
    sim.run(0.5)
    assert sim.ctrl.mode == "STAND"                               # the old 'walk' request is gone
    sim.run(DT, cmd=(0.6, 0.0, 0.0))
    assert sim.ctrl.mode == "WALK"
    assert sim.ctrl.cmd_f[0] <= 0.8 * DT + 1e-9                   # ramp from zero

    # fall while walking fast: the command is cleared as well
    sim.run(2.0, cmd=(0.6, 0.0, 0.0))

    def tipped(s):
        s.st.rpy = np.array([80 * math.pi / 180, 0.0, s.st.rpy[2]])
        s.st.omega_body = np.zeros(3)

    sim.run(0.4, cmd=(0.6, 0.0, 0.0), override=tipped)
    assert sim.ctrl.mode in ("FALLEN", "RECOVERING")
    assert np.allclose(sim.ctrl.cmd_f, 0.0) and sim.ctrl.mpc_result is None


def test_stale_worker_result_is_dropped_and_retarget_ignores_old_plans():
    sim = Sim(params={"use_mpc_thread": True})
    try:
        _stand(sim)
        c = sim.ctrl
        # a solve is running when the motion is reset (e.g. a fall): its result must not be used
        while c.worker.busy:
            time.sleep(0.002)
        c.worker.poll()
        job = (c._mpc_input(sim.t, sim.st), c.gait.snapshot(), None, None)
        assert c.worker.submit(job)
        c._inflight_gen = c._mpc_gen
        c._invalidate_mpc(sim.t)
        deadline = time.time() + 5.0
        while c.worker.busy and time.time() < deadline:
            time.sleep(0.002)
        assert not c.worker.busy
        c._mpc_cycle(sim.t, sim.st, c.gait.current(), None)       # polls the stale result
        assert c.mpc_result is None
        assert c.worker.error is None
    finally:
        sim.ctrl.shutdown()

    # swing re-targeting only follows fresh MPC plans
    sim = _stand(Sim())
    sim.run(0.3, cmd=(0.3, 0.0, 0.0))
    c = sim.ctrl
    for _ in range(20):
        if any(tr is not None for tr in c.swing):
            break
        sim.run(DT, cmd=(0.3, 0.0, 0.0))
    leg = next(i for i in range(4) if c.swing[i] is not None)
    tr = c.swing[leg]
    p1 = tr.p1.copy()
    t_td = c.swing_t0[leg] + tr.duration
    jump = 2.0 * c.p["max_retarget_step"]
    old = SimpleNamespace(ok=True, t=sim.t - 1.0,
                          foothold_events=[(leg, t_td - (sim.t - 1.0), p1[0] + jump, p1[1], p1[2])])
    c.mpc_footholds = old                                         # footholds are taken from mpc_footholds
    c._retarget_stamp.pop(leg, None)
    c._retarget(leg, 0.1, sim.t, None)
    assert np.allclose(tr.p1, p1)                                 # 1 s old plan: ignored
    old.t = sim.t - 0.05
    old.foothold_events = [(leg, t_td - old.t, p1[0] + jump, p1[1], p1[2])]
    c._retarget(leg, 0.1, sim.t, None)
    assert tr.p1[0] == pytest.approx(p1[0] + c.p["max_retarget_step"])   # fresh plan: followed (rate limited)


def test_crawl_keeps_the_liftoff_foothold():
    """crawl_retarget_until 0: a fresh MPC foothold does not move a crawl swing
    (physics check: late re-targets tipped the crawl over); trot still follows."""
    sim = _stand(Sim({"gait_mode": "crawl"}))
    c = sim.ctrl
    for _ in range(200):
        sim.run(DT, cmd=(0.15, 0.0, 0.0))
        if c.gait.name == "crawl" and any(tr is not None for tr in c.swing):
            break
    assert c.gait.name == "crawl"
    leg = next(i for i in range(4) if c.swing[i] is not None)
    tr = c.swing[leg]
    p1 = tr.p1.copy()
    t_td = c.swing_t0[leg] + tr.duration
    fresh = SimpleNamespace(ok=True, t=sim.t - 0.02,
                            foothold_events=[(leg, t_td - (sim.t - 0.02), p1[0] + 0.05, p1[1], p1[2])])
    c.mpc_footholds = fresh
    c._retarget_stamp.pop(leg, None)
    c._retarget(leg, 0.05, sim.t, None)
    assert np.allclose(tr.p1, p1)
    # the same plan for a trot swing is followed
    c.p["crawl_retarget_until"] = 0.8
    c._retarget_stamp.pop(leg, None)
    c._retarget(leg, 0.05, sim.t, None)
    assert tr.p1[0] > p1[0] + 0.01


def test_torque_feedforward_through_the_pd_servo():
    """Stance targets get tau / servo_kp with tau = -J^T R^T f (static weight
    share without a plan, MPC forces with one); swing legs are untouched."""
    sim = _stand(Sim({"torque_feedforward": 1.0}))
    c = sim.ctrl
    q = kin.stand_joint_angles()
    gs = c.gait.current()
    c.mpc_result = None
    out = c._torque_feedforward(sim.t, q, gs, np.zeros(3))
    w = c.p["mpc_mass"] * 9.81 / 4.0
    f_body = np.tile([0.0, 0.0, w], (4, 1))
    tau = kin.leg_torques(q.reshape(4, 3), np.arange(4), f_body).reshape(12)
    np.testing.assert_allclose(out - q, tau / c.p["servo_kp"], atol=1e-12)
    # the knee resists flexion under load: the target moves towards extension (knee < 0)
    assert np.all((out - q)[2::3] > 0.0)
    # swing leg: no offset; the others share the weight
    c.swing[1] = SimpleNamespace()
    out = c._torque_feedforward(sim.t, q, gs, np.zeros(3))
    assert np.allclose(out[3:6], q[3:6])
    assert np.all((out - q)[[2, 8, 11]] > (tau / c.p["servo_kp"])[[2, 8, 11]])
    c.swing[1] = None
    # gain 0 / no servo stiffness: off (ff_gain = the effective gain, ramped towards
    # torque_feedforward, or slip_torque_feedforward in slip mode)
    assert c.ff_gain == 1.0
    c.ff_gain = 0.0
    assert np.array_equal(c._torque_feedforward(sim.t, q, gs, np.zeros(3)), q)
    c.ff_gain = 1.0
    c.p["servo_kp"] = 0.0
    assert np.array_equal(c._torque_feedforward(sim.t, q, gs, np.zeros(3)), q)
    c.p["servo_kp"] = 450.0
    # a fresh MPC plan: its (contact-normalised) forces are used
    fk = np.tile([5.0, 0.0, 30.0], 4)
    res = SimpleNamespace(ok=True, t=sim.t, times=np.array([0.0, 0.05, 0.1]), dts=np.array([0.05, 0.05]),
                          forces=np.stack([fk, fk]), contacts=np.full((2, 4), 0.5))
    c.mpc_result = res
    out = c._torque_feedforward(sim.t, q, gs, np.zeros(3))
    tau = kin.leg_torques(q.reshape(4, 3), np.arange(4), np.tile([10.0, 0.0, 60.0], (4, 1))).reshape(12)
    np.testing.assert_allclose(out - q, tau / c.p["servo_kp"], atol=1e-12)
    # stale plan -> static share again
    res.t = sim.t - 1.0
    out = c._torque_feedforward(sim.t, q, gs, np.zeros(3))
    tau = kin.leg_torques(q.reshape(4, 3), np.arange(4), f_body).reshape(12)
    np.testing.assert_allclose(out - q, tau / c.p["servo_kp"], atol=1e-12)
    # in the walking loop the stance legs carry the offset
    sim.run(0.5, cmd=(0.3, 0.0, 0.0))
    assert c.mode == "WALK"


# ---------------------------------------------------------------------------
# slippery ground: friction prior (/terrain/info) and online slip detection
# ---------------------------------------------------------------------------
FLAT_INFO = {"class": "flat", "recommended_gait": "trot", "ahead_clear": True, "slippery": False,
             "friction_mu": None}
SLIPPERY_INFO = {"class": "slippery", "recommended_gait": "trot", "ahead_clear": True, "slippery": True,
                 "friction_mu": 0.15, "friction_zones": ["F-OBS9"]}


def test_friction_prior_switches_the_slip_mode():
    sim = _stand(Sim())
    c = sim.ctrl
    sim.run(2.0, cmd=(0.6, 0.0, 0.0), terrain=FLAT_INFO)
    assert c.mode == "WALK" and c.gait.name == "trot" and not c.slip_mode
    assert c.cmd_f[0] == pytest.approx(0.6, abs=1e-6) and c.ff_gain == 0.0 and c.mu_mpc == pytest.approx(0.5)
    for k in ("slip", "slip_mode", "slip_source", "slip_speed", "slip_legs", "slip_events", "mu_est", "mpc_mu",
              "friction_prior", "torque_ff"):
        assert k in sim.dbg, k
    c.pop_events()
    # slippery zone ahead (/terrain/info): slip mode at once
    sim.run(0.01, cmd=(0.6, 0.0, 0.0), terrain=SLIPPERY_INFO)
    assert c.slip_mode and c.slip_source == "terrain" and c.friction_prior == pytest.approx(0.15)
    assert [e["event"] for e in c.pop_events()] == ["slip_mode_on"]
    assert c.mu_mpc == pytest.approx(0.15)
    st = sim.st
    assert c._mpc_input(sim.t, st).mu == pytest.approx(0.15)
    # speed limit 0.2 m/s reached with half the normal acceleration (0.4 m/s^2): 1 s from 0.6
    v = []
    for _ in range(12):
        sim.run(0.1, cmd=(0.6, 0.0, 0.0), terrain=SLIPPERY_INFO)
        v.append(float(c.cmd_f[0]))
    assert v[0] == pytest.approx(0.6 - 0.4 * 0.11, abs=0.005)
    assert np.all(np.diff(v) <= 1e-9) and v[-1] == pytest.approx(0.2, abs=1e-6)
    # torque feed-forward ramped in over slip_ff_ramp_time (0.5 s), inline MPC solved with mu 0.15
    assert c.ff_gain == pytest.approx(1.0)
    assert c.mpc_result is not None and c.mpc_result.mu == pytest.approx(0.15)
    s = sim.dbg
    assert s["slip_mode"] is True and s["slip"] is False and s["slip_source"] == "terrain"
    assert s["mpc_mu"] == pytest.approx(0.15) and s["torque_ff"] == pytest.approx(1.0)
    # trot in slip mode: the body-target leash is slip_max_body_lead
    xy = np.asarray(st.pos[:2], dtype=float)
    yaw = float(st.rpy[2])
    c._walk_target = (xy + np.array([0.5, 0.0]), yaw)
    pos, _ = c._integrated_body_target(sim.t, 0.0, st, None)     # no MPC result: command velocity
    v_cmd = np.array([math.cos(yaw) * c.cmd_f[0], math.sin(yaw) * c.cmd_f[0]])
    lead = np.hypot(*(pos[:2] - (xy + v_cmd * c.p["body_lead_time"])))
    assert lead == pytest.approx(c.p["slip_max_body_lead"], abs=1e-9)
    # zone left: slip mode off, limits / feed-forward / mu back
    sim.run(0.01, cmd=(0.6, 0.0, 0.0), terrain=FLAT_INFO)
    assert not c.slip_mode and c.mu_mpc == pytest.approx(0.5)
    assert [e["event"] for e in c.pop_events()] == ["slip_mode_off"]
    sim.run(1.2, cmd=(0.6, 0.0, 0.0), terrain=FLAT_INFO)
    assert c.ff_gain == 0.0 and c.cmd_f[0] == pytest.approx(0.6, abs=1e-6)
    # stale terrain info is ignored (terrain_info_timeout)
    sim.run(0.01, cmd=(0.6, 0.0, 0.0), terrain=SLIPPERY_INFO)
    assert c.slip_mode
    sim.run(2.5, cmd=(0.6, 0.0, 0.0))
    assert not c.slip_mode


def _sliding_plant(rate):
    """Kinematic plant whose base drifts by ``rate`` [m/s] (world x) away from the
    commanded pose: the measured stance feet slide in the world while the joints
    track their targets perfectly."""
    drift = {"d": np.zeros(3)}

    def override(s):
        pos, rpy = s.ctrl.body_cmd
        drift["d"] = drift["d"] + np.array([rate * DT, 0.0, 0.0])
        new = np.asarray(pos, dtype=float) + drift["d"]
        s.st.vel_world = (new - s.st.pos) / DT
        s.st.pos = new
        s.st.rpy = np.asarray(rpy, dtype=float).copy()
    return override, drift


def test_online_slip_detection_from_sliding_stance_feet():
    sim = _stand(Sim({"slip_release_time": 0.5, "slip_release_distance": 0.3}))
    c = sim.ctrl
    sim.run(1.0, cmd=(0.4, 0.0, 0.0), terrain=FLAT_INFO)
    assert not c.slip_mode and not c.slip.slip                  # ideal plant: stance feet fixed
    c.pop_events()
    override, _ = _sliding_plant(-0.3)
    seen = []
    measure = c._feet_world_measured
    c._feet_world_measured = lambda st: seen.append(measure(st)) or seen[-1]
    t0 = sim.t
    detected = None
    for _ in range(60):
        before = c.foot_w.copy()
        sim.run(0.01, cmd=(0.4, 0.0, 0.0), terrain=FLAT_INFO, override=override)
        if c.slip.slip and detected is None:
            detected = sim.t - t0
            # the latched stance-foot targets were re-synchronised with the measured (slid) feet
            stance = [i for i in range(4) if c.swing[i] is None]
            assert np.allclose(c.foot_w[stance, :2], seen[-1][stance, :2], atol=1e-9)
            assert np.max(np.abs(c.foot_w[stance, 0] - before[stance, 0])) > 0.01
    c._feet_world_measured = measure
    assert detected is not None and detected < 0.4
    assert c.slip_mode and c.slip_source == "detected" and c.slip_events == 1
    ev = [e for e in c.pop_events() if e["event"] == "slip_detected"]
    assert len(ev) == 1 and ev[0]["speed"] > c.p["slip_speed_threshold"]
    # ideal servo: no force estimate -> the coefficient drops by slip_mu_drop
    assert c.mu_mpc == pytest.approx(0.5 * c.p["slip_mu_drop"]) and sim.dbg["mu_est"] == pytest.approx(0.3)
    assert sim.dbg["slip"] is True and sim.dbg["slip_source"] == "detected"
    # sliding stops (ideal plant again): released after slip_release_time and slip_release_distance
    sim.run(3.0, cmd=(0.4, 0.0, 0.0), terrain=FLAT_INFO)
    assert not c.slip.slip and not c.slip_mode
    assert 0.3 < c.mu_mpc < 0.5                                  # recovers slowly (slip_recover_tau)
    assert "slip_mode_off" in [e["event"] for e in c.pop_events()]


def test_slip_mode_disabled_and_falling_robot():
    sim = _stand(Sim({"slip_detection": False, "use_terrain_friction": False}))
    c = sim.ctrl
    override, _ = _sliding_plant(-0.3)
    sim.run(1.5, cmd=(0.4, 0.0, 0.0), terrain=SLIPPERY_INFO, override=override)
    assert not c.slip_mode and c.mu_mpc == pytest.approx(0.5) and c.ff_gain == 0.0
    # tilted beyond slip_max_tilt_deg: the stance feet are not watched (a falling robot drags its feet)
    sim2 = _stand(Sim())
    c2 = sim2.ctrl
    st = sim2.st
    st.rpy = np.array([math.radians(30.0), 0.0, 0.0])
    for k in range(40):
        st.pos = st.pos + np.array([0.01, 0.0, 0.0])            # 1 m/s apparent foot motion
        c2._update_slip(sim2.t + 0.01 * k, 0.01, st, np.ones(4, dtype=bool))
    assert not c2.slip.slip and c2.slip.state.speed == 0.0


def test_slip_gait_option():
    rough_slippery = dict(SLIPPERY_INFO, **{"class": "rough", "recommended_gait": "crawl"})
    for opt, info, expected in (("keep", SLIPPERY_INFO, "trot"), ("keep", rough_slippery, "crawl"),
                                ("crawl", SLIPPERY_INFO, "crawl"), ("trot", rough_slippery, "trot")):
        sim = _stand(Sim({"slip_gait": opt, "gait_mode": "auto"}))
        sim.run(0.5, cmd=(0.3, 0.0, 0.0), terrain=info)
        assert sim.ctrl.slip_mode
        assert sim.ctrl._target_gait(sim.t) == expected, (opt, info["class"])


def test_slip_mode_stance_width_offset():
    sim = _stand(Sim({"slip_stance_width_offset": 0.03}))
    c = sim.ctrl
    y0 = np.abs(c.nominal_feet_b[:, 1]).copy()
    sim.run(0.3, cmd=(0.3, 0.0, 0.0), terrain=SLIPPERY_INFO)
    assert c.slip_mode and np.allclose(np.abs(c.nominal_feet_b[:, 1]), y0 + 0.03)
    assert np.allclose(c._mpc_input(sim.t, sim.st).nominal_feet_xy[:, 1], c.nominal_feet_b[:, 1])
    sim.run(0.3, cmd=(0.3, 0.0, 0.0), terrain=FLAT_INFO)
    assert not c.slip_mode and np.allclose(np.abs(c.nominal_feet_b[:, 1]), y0)


def test_servo_force_estimate_inverts_the_leg_jacobian():
    sim = _stand(Sim())
    c = sim.ctrl
    st = sim.st
    st.rpy = np.array([0.05, -0.08, 0.7])
    q = kin.stand_joint_angles().copy()
    st.q = q.copy()
    f_body = np.array([[3.0, -2.0, 40.0], [0.0, 1.0, 35.0], [-4.0, 0.0, 38.0], [1.0, 1.0, 36.0]])
    tau = kin.leg_torques(q.reshape(4, 3), np.arange(4), f_body).reshape(12)
    c.q_cmd = q + tau / c.p["servo_kp"]
    f_w = c._servo_ground_forces(st)
    np.testing.assert_allclose(f_w, f_body @ st.R.T, atol=1e-9)
    c.q_cmd = np.full(12, np.nan)
    assert c._servo_ground_forces(st) is None

# ---------------------------------------------------------------------------
# ROS node against strict rclpy / message stand-ins (no ROS on the dev PC)
# ---------------------------------------------------------------------------
_F, _I, _S, _B, _FA, _SA, _U8 = "float", "int", "str", "bool", "float[]", "str[]", "uint8[]"


def _field_ok(kind, v):
    """Type rules of the rosidl-generated Python setters."""
    if kind == _F:
        return isinstance(v, float)
    if kind == _I:
        return isinstance(v, int) and not isinstance(v, bool)
    if kind == _S:
        return isinstance(v, str)
    if kind == _B:
        return isinstance(v, bool)
    if kind == _FA:          # array.array('d') or a Sequence of Python floats (numpy arrays are rejected)
        if isinstance(v, array.array):
            return v.typecode == "d"
        return isinstance(v, Sequence) and not isinstance(v, str) and all(isinstance(x, float) for x in v)
    if kind == _SA:
        return isinstance(v, Sequence) and not isinstance(v, str) and all(isinstance(x, str) for x in v)
    if kind == _U8:
        return isinstance(v, (bytes, bytearray, array.array))
    if isinstance(kind, tuple):                                   # ("seq", MsgClass)
        return isinstance(v, list) and all(isinstance(x, kind[1]) for x in v)
    if isinstance(kind, str) and kind.startswith("f*"):           # fixed-size float array (numpy)
        return np.asarray(v).shape == (int(kind[2:]),)
    return isinstance(v, kind)


def _field_default(kind):
    defaults = {_F: 0.0, _I: 0, _S: "", _B: False}
    if isinstance(kind, str) and kind in defaults:
        return defaults[kind]
    if kind in (_FA, _SA) or isinstance(kind, tuple):
        return []
    if kind == _U8:
        return array.array("B")
    if isinstance(kind, str) and kind.startswith("f*"):
        return np.zeros(int(kind[2:]))
    return kind()


class _Msg:
    _spec: dict = {}

    def __init__(self, **kw):
        for k, kind in self._spec.items():
            object.__setattr__(self, k, _field_default(kind))
        for k, v in kw.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        kind = self._spec.get(k)
        if kind is None:
            raise AttributeError(f"{type(self).__name__} has no field '{k}'")
        if not _field_ok(kind, v):
            raise AssertionError(f"{type(self).__name__}.{k}: invalid value {v!r}")
        if kind == _U8 and not isinstance(v, array.array):
            v = array.array("B", v)
        object.__setattr__(self, k, v)

    def validate(self):
        for k, kind in self._spec.items():
            v = getattr(self, k)
            assert _field_ok(kind, v), f"{type(self).__name__}.{k}: invalid value {v!r}"
            if isinstance(v, _Msg):
                v.validate()
            elif isinstance(kind, tuple):
                for x in v:
                    x.validate()


def _msg(cls_name, consts=None, **spec):
    cls = type(cls_name, (_Msg,), {"_spec": spec})
    for k, v in (consts or {}).items():
        setattr(cls, k, v)
    return cls


class _Ros:
    """Message classes (real field names / types) + a fake rclpy world."""

    def __init__(self, overrides):
        self.Time = _msg("Time", sec=_I, nanosec=_I)
        self.Duration = _msg("Duration", sec=_I, nanosec=_I)
        self.Header = _msg("Header", stamp=self.Time, frame_id=_S)
        self.Vector3 = _msg("Vector3", x=_F, y=_F, z=_F)
        self.Point = _msg("Point", x=_F, y=_F, z=_F)
        self.Quaternion = _msg("Quaternion", x=_F, y=_F, z=_F, w=_F)
        self.Pose = _msg("Pose", position=self.Point, orientation=self.Quaternion)
        self.PoseStamped = _msg("PoseStamped", header=self.Header, pose=self.Pose)
        pwc = _msg("PoseWithCovariance", pose=self.Pose, covariance="f*36")
        self.Twist = _msg("Twist", linear=self.Vector3, angular=self.Vector3)
        twc = _msg("TwistWithCovariance", twist=self.Twist, covariance="f*36")
        self.Odometry = _msg("Odometry", header=self.Header, child_frame_id=_S, pose=pwc, twist=twc)
        self.Imu = _msg("Imu", header=self.Header, orientation=self.Quaternion, orientation_covariance="f*9",
                        angular_velocity=self.Vector3, angular_velocity_covariance="f*9",
                        linear_acceleration=self.Vector3, linear_acceleration_covariance="f*9")
        self.JointState = _msg("JointState", header=self.Header, name=_SA, position=_FA, velocity=_FA, effort=_FA)
        self.PointField = _msg("PointField", consts={"FLOAT32": 7, "FLOAT64": 8},
                               name=_S, offset=_I, datatype=_I, count=_I)
        self.PointCloud2 = _msg("PointCloud2", header=self.Header, height=_I, width=_I,
                                fields=("seq", self.PointField), is_bigendian=_B, point_step=_I, row_step=_I,
                                data=_U8, is_dense=_B)
        self.String = _msg("String", data=_S)
        self.Float64 = _msg("Float64", data=_F)
        self.Path = _msg("Path", header=self.Header, poses=("seq", self.PoseStamped))
        color = _msg("ColorRGBA", r=_F, g=_F, b=_F, a=_F)
        self.Marker = _msg("Marker", consts={"ARROW": 0, "CUBE": 1, "SPHERE": 2, "CYLINDER": 3, "LINE_STRIP": 4,
                                             "LINE_LIST": 5, "CUBE_LIST": 6, "SPHERE_LIST": 7, "POINTS": 8,
                                             "TEXT_VIEW_FACING": 9, "MESH_RESOURCE": 10, "TRIANGLE_LIST": 11,
                                             "ADD": 0, "MODIFY": 0, "DELETE": 2, "DELETEALL": 3},
                           header=self.Header, ns=_S, id=_I, type=_I, action=_I, pose=self.Pose,
                           scale=self.Vector3, color=color, lifetime=self.Duration, frame_locked=_B,
                           points=("seq", self.Point), colors=("seq", color), text=_S, mesh_resource=_S,
                           mesh_use_embedded_materials=_B)
        self.MarkerArray = _msg("MarkerArray", markers=("seq", self.Marker))
        self.t = 50.0
        self.overrides = dict(overrides)
        self.pubs, self.subs, self.timers, self.logs = {}, {}, [], []
        self.ok = False

    def published(self, topic):
        return self.pubs[topic].msgs if topic in self.pubs else []

    def deliver(self, topic, m):
        mtype, cb = self.subs[topic]
        assert isinstance(m, mtype)
        cb(m)

    def tick(self, dt=0.01):
        """Advance the fake ROS clock and run the timers that are due."""
        self.t += dt
        for tm in self.timers:
            if self.t + 1e-9 >= tm[2] + tm[0]:
                tm[2] = self.t
                tm[1]()

    def modules(self):
        ros = self

        def ptype(v):
            for t_, name in ((bool, "bool"), (int, "int"), (float, "double"), (str, "string")):
                if isinstance(v, t_):
                    return name
            if isinstance(v, (list, tuple)):
                return "double[]" if all(isinstance(x, float) for x in v) else "other[]"
            return type(v).__name__

        class InvalidParameterTypeException(Exception):
            pass

        class Param:
            def __init__(self, v):
                self.value = v

        class Clock:
            def now(self):
                ns = int(round(ros.t * 1e9))
                return SimpleNamespace(nanoseconds=ns,
                                       to_msg=lambda: ros.Time(sec=ns // 10 ** 9, nanosec=ns % 10 ** 9))

        class Logger:
            def __getattr__(self, level):
                return lambda m, **kw: ros.logs.append((level, str(m)))

        class Publisher:
            def __init__(self, mtype, topic):
                self.type, self.topic, self.msgs = mtype, topic, []

            def publish(self, m):
                assert isinstance(m, self.type), self.topic
                m.validate()
                self.msgs.append(m)

        class Node:
            def __init__(self, node_name, **kw):
                self._params = {"use_sim_time": Param(True)}

            def declare_parameter(self, name, value=None, descriptor=None, ignore_override=False):
                assert name not in self._params, f"parameter {name} declared twice"
                v = ros.overrides.get(name, value)
                # rclpy: an override must have the type of the default (no int -> double promotion)
                if not getattr(descriptor, "dynamic_typing", False) and ptype(v) != ptype(value):
                    raise InvalidParameterTypeException(name)
                self._params[name] = Param(v)
                return self._params[name]

            def get_parameter(self, name):
                return self._params[name]

            def create_subscription(self, mtype, topic, cb, qos, *, callback_group=None):
                assert topic not in ros.subs
                ros.subs[topic] = (mtype, cb)

            def create_publisher(self, mtype, topic, qos, *, callback_group=None):
                assert topic not in ros.pubs
                ros.pubs[topic] = Publisher(mtype, topic)
                return ros.pubs[topic]

            def create_timer(self, period, cb, callback_group=None, clock=None):
                ros.timers.append([float(period), cb, ros.t])

            def get_clock(self):
                return Clock()

            def get_logger(self):
                return Logger()

            def destroy_node(self):
                ros.logs.append(("info", "destroyed"))

        class Executor:
            def __init__(self, num_threads=None, context=None):
                self.nodes = []

            def add_node(self, n):
                self.nodes.append(n)

            def spin(self):
                raise KeyboardInterrupt

            def shutdown(self, timeout_sec=None):
                ros.logs.append(("info", "executor shutdown"))

        mods = {}

        def mod(name, **attrs):
            m = types.ModuleType(name)
            m.__dict__.update(attrs)
            mods[name] = m

        mod("rclpy", init=lambda args=None: setattr(ros, "ok", True), shutdown=lambda: setattr(ros, "ok", False),
            ok=lambda: ros.ok)
        mod("rclpy.node", Node=Node)
        mod("rclpy.executors", MultiThreadedExecutor=Executor,
            ExternalShutdownException=type("ExternalShutdownException", (Exception,), {}))
        mod("rclpy.callback_groups", MutuallyExclusiveCallbackGroup=SimpleNamespace,
            ReentrantCallbackGroup=SimpleNamespace)
        mod("rclpy.qos", qos_profile_sensor_data=SimpleNamespace(reliability="best_effort", depth=5))
        mod("rcl_interfaces")
        mod("rcl_interfaces.msg", ParameterDescriptor=SimpleNamespace)
        for pkg, names in (("std_msgs", ("String", "Float64")), ("geometry_msgs", ("Point", "PoseStamped", "Twist")),
                           ("nav_msgs", ("Odometry", "Path")),
                           ("sensor_msgs", ("Imu", "JointState", "PointCloud2", "PointField")),
                           ("visualization_msgs", ("Marker", "MarkerArray"))):
            mod(pkg)
            mod(pkg + ".msg", **{n: getattr(self, n) for n in names})
        return mods


@pytest.fixture()
def fake_ros(monkeypatch):
    path = os.path.join(HERE, "..", "config", "locomotion.yaml")
    with open(path, encoding="utf-8") as f:
        over = yaml.safe_load(f)["locomotion_controller"]["ros__parameters"]
    over["use_mpc_thread"] = False                              # deterministic inline solves
    over["mpc_rate"] = 10                                       # int for a double parameter (rclpy rejects it)
    ros = _Ros(over)
    for name, m in ros.modules().items():
        monkeypatch.setitem(sys.modules, name, m)
    sys.modules.pop("module1_locomotion.locomotion_controller_node", None)
    interval = sys.getswitchinterval()
    yield ros
    sys.setswitchinterval(interval)
    sys.modules.pop("module1_locomotion.locomotion_controller_node", None)


def test_ros_node_against_strict_message_stand_ins(fake_ros):
    """Construct the node from config/locomotion.yaml, feed it /odom, /imu,
    /joint_states, /cmd_vel, /elevation_map/points, /terrain/info and
    commands, tick its timers and check every published message (field names
    and rosidl types), the JSON contracts and the gz command lines."""
    import importlib
    import json
    from dusan_common.geometry import euler_to_quat, euler_to_rot
    ros = fake_ros
    ros.overrides["locomotion_mode"] = "mpc"                   # MPC topics + body-pose plant (stable: below)
    N = importlib.import_module("module1_locomotion.locomotion_controller_node")
    node = N.LocomotionControllerNode()
    try:
        # int YAML value for a double parameter: accepted through the dynamic-typing fallback
        assert node.cfg["mpc_rate"] == 10.0 and isinstance(node.cfg["mpc_rate"], float)
        assert set(ros.pubs) == {rm.joint_command_topic(j) for j in rm.JOINT_NAMES} | {
            "/locomotion/state", "/locomotion/fall_event", "/mpc/predicted_trajectory", "/mpc/footholds",
            "/mpc/grf", "/mpc/predicted_torques"}
        assert set(ros.subs) == {"/odom", "/imu", "/joint_states", "/cmd_vel", "/locomotion/command",
                                 "/terrain/info", "/elevation_map/points"}
        plant = {"pos": np.array([1.0, 2.0, 0.56]), "rpy": np.array([0.0, 0.0, 0.3]), "q": np.zeros(12),
                 "v": np.zeros(3), "w": np.zeros(3), "tipped": False}
        joint_pubs = [ros.pubs[rm.joint_command_topic(j)] for j in rm.JOINT_NAMES]

        def send():
            o = ros.Odometry()
            o.header.frame_id, o.child_frame_id = "odom", "base_link"
            p = o.pose.pose
            p.position.x, p.position.y, p.position.z = (float(v) for v in plant["pos"])
            q = euler_to_quat(*(float(v) for v in plant["rpy"]))
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in q)
            vb = euler_to_rot(*(float(v) for v in plant["rpy"])).T @ plant["v"]     # twist in the base frame
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = (float(v) for v in vb)
            ros.deliver("/odom", o)
            imu = ros.Imu()
            imu.orientation = p.orientation
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (float(v) for v in plant["w"])
            ros.deliver("/imu", imu)
            js = ros.JointState()
            order = list(reversed(rm.JOINT_NAMES))                    # name-mapped, any order
            js.name = order
            js.position = [float(plant["q"][rm.JOINT_NAMES.index(n)]) for n in order]
            js.velocity = [0.0] * 12
            ros.deliver("/joint_states", js)

        def step_plant():
            if all(pb.msgs for pb in joint_pubs):
                plant["q"] = np.array([pb.msgs[-1].data for pb in joint_pubs])
            c = node.ctrl
            if plant["tipped"]:
                plant["rpy"] = np.array([80 * math.pi / 180, 0.0, plant["rpy"][2]])
                plant["v"], plant["w"] = np.zeros(3), np.zeros(3)
            elif c.mode in ("STAND", "WALK", "SIT") and c.body_cmd is not None:
                pos, rpy = c.body_cmd
                plant["v"] = (np.asarray(pos) - plant["pos"]) / DT
                plant["pos"], plant["rpy"] = np.array(pos, dtype=float), np.array(rpy, dtype=float)
            else:
                fk = kin.fk_all(plant["q"])
                plant["pos"] = np.array([plant["pos"][0], plant["pos"][1], rm.FOOT_RADIUS - fk[:, 2].min()])
                plant["v"] = np.zeros(3)

        def run(sec, cmd=None):
            for _ in range(int(round(sec / DT))):
                if cmd is not None:
                    tw = ros.Twist()
                    tw.linear.x, tw.linear.y, tw.angular.z = (float(v) for v in cmd)
                    ros.deliver("/cmd_vel", tw)
                send()
                ros.tick(DT)
                step_plant()

        def state():
            return json.loads(ros.published("/locomotion/state")[-1].data)

        for _ in range(520):                                           # no data yet: no command, a warning
            ros.tick(DT)
        assert not joint_pubs[0].msgs
        assert any(lv in ("warn", "warning") and "/odom" in m and "/joint_states" in m for lv, m in ros.logs)
        run(3.3)                                                       # startup delay + 2.5 s stand-up
        s = state()
        assert s["mode"] == "STAND" and s["body_height"] == pytest.approx(0.34, abs=0.01)
        for k in ("mode", "gait", "gait_auto", "phase", "contacts", "body_height", "roll", "pitch", "fall_count",
                  "recovery_attempts", "recovery_success", "mpc_solve_ms", "mpc_status", "stability_margin",
                  "terrain", "cmd"):
            assert k in s, k
        assert len(s["contacts"]) == 4 and len(s["cmd"]) == 3

        # organised elevation-map cloud (x, y, z float32 + 4 padding bytes, NaN holes) and terrain info
        X, Y = np.meshgrid(np.arange(-1.0, 3.0, 0.04), np.arange(-1.0, 5.0, 0.04))
        buf = np.zeros(X.shape + (4,), np.float32)
        buf[..., 0], buf[..., 1] = X, Y
        buf[5:8, 5:8, 2] = np.nan
        pc = ros.PointCloud2(height=X.shape[0], width=X.shape[1], point_step=16, row_step=16 * X.shape[1],
                             is_bigendian=False, is_dense=False, data=buf.tobytes())
        pc.fields = [ros.PointField(name=n, offset=o, datatype=7, count=1) for n, o in (("x", 0), ("y", 4), ("z", 8))]
        ros.deliver("/elevation_map/points", pc)
        try:
            import module1_locomotion.elevation_map  # noqa: F401
            assert node._emap is not None
        except ImportError:
            assert node._emap is None
        ros.deliver("/terrain/info", ros.String(data=json.dumps({"class": "flat", "recommended_gait": "trot",
                                                                  "ahead_clear": True})))
        run(1.5, cmd=(0.4, 0.0, 0.2))
        s = state()
        assert s["mode"] == "WALK" and s["gait"] == "trot" and s["mpc_status"] in ("solved", "solved_inaccurate")
        assert s["terrain"] == "flat" and plant["pos"][0] > 1.2
        # slippery ground: /locomotion/state carries the slip state; the friction prior of
        # /terrain/info switches the slip mode on (lower MPC mu) and off again
        for k in ("slip", "slip_mode", "slip_source", "slip_speed", "slip_legs", "mu_est", "mpc_mu",
                  "friction_prior", "torque_ff"):
            assert k in s, k
        assert s["slip_mode"] is False and s["mpc_mu"] == pytest.approx(0.5)
        ros.deliver("/terrain/info", ros.String(data=json.dumps({
            "class": "slippery", "recommended_gait": "trot", "ahead_clear": True, "slippery": True,
            "friction_mu": 0.15, "friction_zones": ["F-OBS9"]})))
        run(0.3, cmd=(0.4, 0.0, 0.2))
        s = state()
        assert s["slip_mode"] is True and s["slip_source"] == "terrain" and s["slip"] is False
        assert s["mpc_mu"] == pytest.approx(0.15) and s["friction_prior"] == pytest.approx(0.15)
        assert s["terrain"] == "slippery" and s["cmd"][0] < 0.4
        ros.deliver("/terrain/info", ros.String(data=json.dumps({"class": "flat", "recommended_gait": "trot",
                                                                  "ahead_clear": True})))
        run(0.6, cmd=(0.4, 0.0, 0.2))
        assert state()["slip_mode"] is False and state()["torque_ff"] == 0.0
        node._publish_viz()
        path = ros.published("/mpc/predicted_trajectory")[-1]
        assert len(path.poses) == 29 and path.header.frame_id == "odom"
        assert len(ros.published("/mpc/footholds")[-1].markers) == 4
        assert len(ros.published("/mpc/grf")[-1].markers) == 4
        js = ros.published("/mpc/predicted_torques")[-1]
        assert list(js.name) == list(rm.JOINT_NAMES) and len(js.effort) == 12

        # push (base-frame force -> world frame) and reset_pose through a fake gz runner
        calls = []
        node.gz.runner = lambda cmd, timeout: (calls.append(list(cmd)) or (0, "data: true"))
        node.gz._sleep = lambda s: None
        # the push duration is measured on the node clock (sim time); the fake clock only
        # advances in ros.tick(), so time this push in wall time
        assert node.gz.clock is not None and node.gz.clock() == pytest.approx(ros.t)
        node.gz.clock = None
        yaw = float(node._odom[2][2])                                   # heading the node knows
        ros.deliver("/locomotion/command", ros.String(data="push:10,0"))
        node.gz.wait(5.0)
        applies = [c for c in calls if c[3] == "/world/plant/wrench/persistent"]
        clears = [c for c in calls if c[3] == "/world/plant/wrench/clear"]
        assert len(applies) == 1 and len(clears) == 2 and len(calls) == 3
        force = applies[0][-1].split("force: {")[1].split("}")[0]
        fxy = dict((kv.split(":")[0].strip(), float(kv.split(":")[1])) for kv in force.split(","))
        assert fxy["x"] == pytest.approx(10 * math.cos(yaw), abs=1e-3)
        assert fxy["y"] == pytest.approx(10 * math.sin(yaw), abs=1e-3)
        assert all(c[:4] == ["gz", "topic", "-t", "/world/plant/wrench/clear"] for c in clears)
        ros.deliver("/locomotion/command", ros.String(data="estop"))
        run(0.3)
        assert state()["mode"] == "ESTOP"
        ros.deliver("/locomotion/command", ros.String(data="release"))
        run(2.7)
        assert state()["mode"] == "STAND"
        # set_pose that fails (no UserCommands answer): retried once, the robot carries on
        calls.clear()
        node.gz.runner = lambda cmd, timeout: (calls.append(list(cmd)) or (0, "Service call timed out"))
        ros.deliver("/locomotion/command", ros.String(data="reset_pose"))
        node.gz.wait(5.0)
        assert len(calls) == 2 and calls[0][:4] == ["gz", "service", "-s", "/world/plant/set_pose"]
        run(0.05)
        assert node.ctrl.mode == "STAND"
        # set_pose that succeeds: the settle window starts when the result reaches the control
        # loop (teleport executed), not at the request
        calls.clear()
        node.gz.runner = lambda cmd, timeout: (calls.append(list(cmd)) or (0, "data: true"))
        ros.deliver("/locomotion/command", ros.String(data="reset_pose"))
        assert node.ctrl.mode == "STAND"
        node.gz.wait(5.0)
        assert len(calls) == 1 and calls[0][:4] == ["gz", "service", "-s", "/world/plant/set_pose"]
        run(0.02)
        assert node.ctrl.mode == "STANDING_UP"

        # fall -> /locomotion/fall_event
        plant["tipped"] = True
        run(1.0)
        ev = [json.loads(m.data) for m in ros.published("/locomotion/fall_event")]
        assert ev and ev[0]["event"] == "fall" and set(ev[0]) == {"event", "t", "x", "y", "roll", "pitch"}
        assert state()["mode"] in ("FALLEN", "RECOVERING") and state()["fall_count"] == 1
        assert node._errors == 0
        assert not [line for line in ros.logs if line[0] in ("error", "fatal")]
    finally:
        node.shutdown()

    # main(): construct, spin until Ctrl-C, orderly shutdown
    ros.subs.clear()
    ros.pubs.clear()
    ros.timers.clear()
    N.main()
    assert not ros.ok and ("info", "executor shutdown") in ros.logs and ("info", "destroyed") in ros.logs


def test_ros_node_resumes_joint_commands_after_a_sim_clock_jump_back(fake_ros):
    """Gazebo world reset: the sim time restarts near 0.  The node must re-arm
    its start-up gate (joint commands resume after startup_delay, not only
    once the clock has passed the old first-data time again) and must not
    apply the last /cmd_vel, stamped before the jump."""
    import importlib
    ros = fake_ros
    N = importlib.import_module("module1_locomotion.locomotion_controller_node")
    node = N.LocomotionControllerNode()
    try:
        q_stand = node.ctrl.q_stand

        def send():
            o = ros.Odometry()
            o.header.frame_id, o.child_frame_id = "odom", "base_link"
            o.pose.pose.position.z = 0.34
            o.pose.pose.orientation.w = 1.0
            ros.deliver("/odom", o)
            js = ros.JointState()
            js.name = list(rm.JOINT_NAMES)
            js.position = [float(v) for v in q_stand]
            js.velocity = [0.0] * 12
            ros.deliver("/joint_states", js)

        def run(sec):
            for _ in range(int(round(sec / DT))):
                send()
                ros.tick(DT)

        jp = ros.pubs[rm.joint_command_topic(rm.JOINT_NAMES[0])]
        run(3.5)
        assert node.ctrl.mode == "STAND" and jp.msgs
        tw = ros.Twist()
        tw.linear.x = 0.4
        ros.deliver("/cmd_vel", tw)                     # last command before the reset
        n0 = len(jp.msgs)
        ros.t = 2.0                                     # world reset: sim time restarts
        for tm in ros.timers:                           # rcl re-arms timers after a backward jump
            tm[2] = ros.t
        run(1.0)
        assert len(jp.msgs) - n0 >= 40                  # resumed after the 0.5 s gate (was: none until t > 50.5)
        assert node.ctrl.t_prev < 4.0 and node.ctrl.mode == "STANDING_UP"
        assert node._t_cmd < 0                          # stale /cmd_vel dropped
        assert any("clock jumped back" in m for _lv, m in ros.logs)
        assert node._errors == 0
    finally:
        node.shutdown()


def test_ros_node_default_stable_mode_and_mode_command(fake_ros):
    """config/locomotion.yaml selects the stable mode: no MPC solve, the
    state carries locomotion_mode, and /locomotion/command 'mode:mpc' /
    'mode:stable' switches at runtime (applied while standing)."""
    import importlib
    import json
    ros = fake_ros
    N = importlib.import_module("module1_locomotion.locomotion_controller_node")
    node = N.LocomotionControllerNode()
    try:
        assert node.cfg["locomotion_mode"] == "stable" and node.ctrl.loco_mode == "stable"
        q = {"q": np.zeros(12)}
        joint_pubs = [ros.pubs[rm.joint_command_topic(j)] for j in rm.JOINT_NAMES]

        def run(sec, cmd=None):
            for _ in range(int(round(sec / DT))):
                if cmd is not None:
                    tw = ros.Twist()
                    tw.linear.x = float(cmd)
                    ros.deliver("/cmd_vel", tw)
                o = ros.Odometry()
                o.header.frame_id, o.child_frame_id = "odom", "base_link"
                o.pose.pose.position.z = rm.FOOT_RADIUS - float(kin.fk_all(q["q"])[:, 2].min())
                o.pose.pose.orientation.w = 1.0
                ros.deliver("/odom", o)
                js = ros.JointState()
                js.name = list(rm.JOINT_NAMES)
                js.position = [float(v) for v in q["q"]]
                js.velocity = [0.0] * 12
                ros.deliver("/joint_states", js)
                ros.tick(DT)
                if all(pb.msgs for pb in joint_pubs):
                    q["q"] = np.array([pb.msgs[-1].data for pb in joint_pubs])

        def state():
            return json.loads(ros.published("/locomotion/state")[-1].data)

        run(3.3)
        s = state()
        assert s["mode"] == "STAND" and s["locomotion_mode"] == "stable" and s["mpc_status"] == "none"
        run(0.5, cmd=0.3)
        s = state()
        assert s["mode"] == "WALK" and s["gait"] == "trot" and s["mpc_status"] == "none"
        assert node.ctrl.mpc_stats == []                                  # the MPC never ran
        ros.deliver("/locomotion/command", ros.String(data="mode:mpc"))
        run(1.5)                                                          # stops at a boundary, switches
        assert state()["locomotion_mode"] == "mpc"
        run(2.5)                                                          # (the last /cmd_vel was still fresh)
        assert state()["mode"] == "STAND"
        ros.deliver("/locomotion/command", ros.String(data="mode:stable"))
        run(0.1)
        assert state()["locomotion_mode"] == "stable" and node._errors == 0
    finally:
        node.shutdown()


def test_ros_node_callbacks_never_raise(fake_ros):
    """An exception escaping a callback would propagate out of
    executor.spin() and stop the node: malformed /joint_states, a failing
    command handler and a broken MPC result must only be logged."""
    import importlib
    ros = fake_ros
    N = importlib.import_module("module1_locomotion.locomotion_controller_node")
    node = N.LocomotionControllerNode()
    try:
        js = ros.JointState()
        js.name = list(rm.JOINT_NAMES)
        js.position = [0.0] * 6                                      # shorter than name[]
        ros.deliver("/joint_states", js)
        assert node._joints is None
        js.position = [0.0] * 11 + [float("nan")]
        ros.deliver("/joint_states", js)
        assert node._joints is None

        def boom(*a, **k):
            raise RuntimeError("handler bug")

        node.ctrl.handle_command = boom
        ros.deliver("/locomotion/command", ros.String(data="stand"))
        assert any(lv == "error" and "handler bug" in m for lv, m in ros.logs)

        node.ctrl.mpc_result = SimpleNamespace(t=0.0, x_pred=None)    # broken result object
        node._publish_viz()
        assert any("visualisation failed" in m for _lv, m in ros.logs)
    finally:
        node.shutdown()


def test_non_finite_state_never_reaches_the_servos_and_does_not_latch():
    sim = _stand(Sim())
    sim.run(0.5)
    q_prev = sim.qs[-1].copy()
    good = sim.st.pos.copy()
    sim.st.pos = np.array([np.nan, 0.0, np.nan])
    q, _dbg = sim.ctrl.step(sim.t, sim.st, (0.0, 0.0, 0.0), None, None)
    assert np.all(np.isfinite(q)) and np.allclose(q, q_prev, atol=1e-6)
    assert math.isfinite(sim.ctrl.z_int)
    sim.st.pos = good
    sim.t += DT
    sim.run(1.0, cmd=(0.3, 0.0, 0.0))                              # recovers: walks on normally
    assert sim.ctrl.mode == "WALK" and np.all(np.isfinite(sim.qs[-1]))
    assert sim.dbg["mpc_status"] in ("solved", "solved_inaccurate")


def test_body_target_leash_is_tight_in_trot_and_wide_in_crawl():
    """The integrated body target's lead is a force demand on the stance feet
    (Gazebo check): trot limits max_body_lead / max_body_lead_lateral /
    max_yaw_lead, the crawl keeps crawl_* for its CoM sway."""
    p = C.DEFAULT_PARAMS
    assert p["max_body_lead"] <= 0.06 and p["max_body_lead_lateral"] <= 0.015 and p["max_yaw_lead"] <= 0.05
    assert p["crawl_max_body_lead"] > p["max_body_lead"]
    for gait, key, lat_key, yaw_key in (("trot", "max_body_lead", "max_body_lead_lateral", "max_yaw_lead"),
                                        ("crawl", "crawl_max_body_lead", "crawl_max_body_lead_lateral",
                                         "crawl_max_yaw_lead")):
        sim = _stand(Sim(params={"gait_mode": gait}))
        sim.run(1.0, cmd=(0.2, 0.0, 0.0))
        c, st = sim.ctrl, sim.st
        assert c.gait.name == gait
        xy, yaw = np.asarray(st.pos[:2], dtype=float), float(st.rpy[2])
        anchor = xy + np.array([math.cos(yaw), math.sin(yaw)]) * c.cmd_f[0] * c.p["body_lead_time"]
        # far ahead: clamped to the total lead
        c._walk_target = (xy + 0.5 * np.array([math.cos(yaw), math.sin(yaw)]), yaw)
        pos, _ = c._integrated_body_target(sim.t, 0.0, st, None)
        assert np.hypot(*(pos[:2] - anchor)) == pytest.approx(c.p[key], abs=1e-9)
        # far to the side: clamped to the lateral lead
        c._walk_target = (xy + 0.5 * np.array([-math.sin(yaw), math.cos(yaw)]), yaw)
        pos, _ = c._integrated_body_target(sim.t, 0.0, st, None)
        d = pos[:2] - anchor
        assert abs(-math.sin(yaw) * d[0] + math.cos(yaw) * d[1]) == pytest.approx(c.p[lat_key], abs=1e-6)
        # heading far ahead: clamped to the yaw lead
        c._walk_target = (xy, yaw + 1.0)
        _, rpy = c._integrated_body_target(sim.t, 0.0, st, None)
        assert rpy[2] - (yaw + c.cmd_f[2] * c.p["body_lead_time"]) == pytest.approx(c.p[yaw_key], abs=1e-9)


def _walking_stance(params=None):
    """Walking MPC-mode controller on the kinematic plant + its stance legs."""
    sim = _stand(Sim(params=params))
    sim.run(0.55, cmd=(0.2, 0.0, 0.0))
    c = sim.ctrl
    assert c.mode == "WALK"
    stance = np.array([c.swing[i] is None and bool(c.gait.current().contacts[i]) for i in range(4)])
    assert stance.any()
    return sim, c, stance, int(np.flatnonzero(stance)[0])


def test_mpc_stance_anchor_follows_a_slid_foot():
    """mpc_stance_max_error: the anchor of a stance foot that slid stays within
    e_max of the measured foot - the servo turns the anchor error into a
    horizontal ground force (~4.5 kN/m), a slid foot was dragged back with
    several hundred N in Gazebo (sliding -> spin -> fall)."""
    sim, c, stance, leg = _walking_stance()
    meas = c._feet_world_measured(sim.st)
    off = np.array([0.10, -0.05])
    c.foot_w[leg, :2] = meas[leg, :2] + off
    z0 = float(c.foot_w[leg, 2])
    c._mpc_stance_feet(DT, sim.st, stance, None)
    d = c.foot_w[leg, :2] - meas[leg, :2]
    assert float(np.hypot(*d)) == pytest.approx(c.p["mpc_stance_max_error"], abs=1e-9)
    assert np.allclose(d / np.hypot(*d), off / np.hypot(*off))      # pulled straight towards the foot
    assert c.foot_w[leg, 2] == z0
    # within the limit: untouched; swing legs: untouched
    c.foot_w[leg, :2] = meas[leg, :2] + np.array([0.01, 0.0])
    swing_legs = np.flatnonzero(~stance)
    before = c.foot_w.copy()
    c._mpc_stance_feet(DT, sim.st, stance, None)
    assert np.allclose(c.foot_w[leg, :2], meas[leg, :2] + np.array([0.01, 0.0]))
    assert np.allclose(c.foot_w[swing_legs], before[swing_legs])
    # off
    c.p["mpc_stance_max_error"] = 0.0
    c.foot_w[leg, :2] = meas[leg, :2] + off
    c._mpc_stance_feet(DT, sim.st, stance, None)
    assert np.allclose(c.foot_w[leg, :2], meas[leg, :2] + off)
    # the stable mode's own limit is a separate parameter
    assert C.DEFAULT_PARAMS["stable_stance_max_error"] == 0.015


def test_mpc_stance_foot_resting_on_a_mapped_obstacle_gets_the_measured_height():
    """A stance foot that came down on an obstacle (pipe) sits above its
    planned floor height: the leg would be compressed by the difference
    (Gazebo: 3-8 cm -> 150-350 N, pitch-over in 50 ms).  When the foot sits at
    the top of mapped terrain next to it, its target becomes the measured
    height; a foot above the mapped top (late swing, still in the air) or a
    foot lifted off flat ground by a tipping body (push) is not lifted."""
    sim, c, stance, leg = _walking_stance()
    meas = c._feet_world_measured(sim.st)
    xm, zf = float(meas[leg, 0]), float(meas[leg, 2])
    lift = c.p["mpc_contact_lift"]
    top = zf - rm.FOOT_RADIUS                              # mapped top at the measured foot
    pipe = FakeMap(lambda x, y: np.where(np.abs(x - xm) < 0.04, top, top - 0.08))
    c.foot_w[leg, 2] = zf - 0.08                           # target at the floor next to the pipe
    c._mpc_stance_feet(DT, sim.st, stance, pipe)
    assert c.foot_w[leg, 2] == pytest.approx(zf)
    # the foot 6 cm above the mapped top (still coming down): kept
    above = FakeMap(lambda x, y: np.where(np.abs(x - xm) < 0.04, top - 0.06, top - 0.14))
    c.foot_w[leg, 2] = zf - 0.14
    c._mpc_stance_feet(DT, sim.st, stance, above)
    assert c.foot_w[leg, 2] == pytest.approx(zf - 0.14)
    # flat ground under a lifted foot (tipping body), with or without a map: kept
    floor = FakeMap(lambda x, y: np.full_like(x, top - 0.06))
    c.foot_w[leg, 2] = zf - 0.06
    c._mpc_stance_feet(DT, sim.st, stance, None)
    assert c.foot_w[leg, 2] == pytest.approx(zf - 0.06)
    for _ in range(3):
        c._mpc_stance_feet(DT, sim.st, stance, floor)
    assert c.foot_w[leg, 2] <= zf - 0.06 + 1e-9
    # within mpc_contact_lift: kept (normal stance)
    c.foot_w[leg, 2] = zf - 0.5 * lift
    c._mpc_stance_feet(DT, sim.st, stance, pipe)
    assert c.foot_w[leg, 2] == pytest.approx(zf - 0.5 * lift)
    # off
    c.p["mpc_contact_lift"] = 0.0
    c.foot_w[leg, 2] = zf - 0.08
    c._mpc_stance_feet(DT, sim.st, stance, pipe)
    assert c.foot_w[leg, 2] == pytest.approx(zf - 0.08)

def test_mpc_stance_foot_height_is_lowered_to_the_map_never_raised():
    """With a map a stance anchor above the map height under it (planned onto
    a pipe, came down beside it: the foot would hang) settles down with
    mpc_land_tau; it is never raised to the map (the depth map shows wall edges
    as 5-15 cm terrain next to the feet)."""
    sim, c, stance, leg = _walking_stance()
    floor = FakeMap(lambda x, y: np.zeros_like(x))
    c.foot_w[leg, 2] = 0.08 + rm.FOOT_RADIUS
    for _ in range(40):
        c._mpc_stance_feet(DT, sim.st, stance, floor)
    assert c.foot_w[leg, 2] == pytest.approx(rm.FOOT_RADIUS, abs=0.002)
    high = FakeMap(lambda x, y: np.full_like(x, 0.10))
    for _ in range(40):
        c._mpc_stance_feet(DT, sim.st, stance, high)
    assert c.foot_w[leg, 2] == pytest.approx(rm.FOOT_RADIUS, abs=0.002)
    # implausible map heights (> 0.35 m from the ground estimate) are ignored
    c.foot_w[leg, 2] = 0.05
    wall = FakeMap(lambda x, y: np.full_like(x, -1.0))
    c._mpc_stance_feet(DT, sim.st, stance, wall)
    assert c.foot_w[leg, 2] == pytest.approx(0.05)


def test_mpc_heading_hold():
    """The MPC reference yaw is re-anchored on the measured yaw at every solve:
    while no turn is commanded the heading at the end of the last turn is held
    (k (yaw_ref - yaw), limited), as in the stable mode."""
    sim = _stand(Sim())
    sim.run(1.0, cmd=(0.2, 0.0, 0.0))
    c = sim.ctrl
    assert c.mode == "WALK" and c._yaw_ref is not None
    ref = float(c._yaw_ref)
    k, wmax = c.p["mpc_heading_hold_gain"], c.p["mpc_heading_hold_max"]
    st = C.RobotState(pos=sim.st.pos.copy(), rpy=sim.st.rpy.copy(), vel_world=sim.st.vel_world.copy(),
                      omega_body=sim.st.omega_body.copy(), q=sim.st.q.copy())
    st.rpy[2] = ref + 0.1
    assert c._mpc_input(sim.t, st).cmd[2] == pytest.approx(float(c.cmd_f[2]) - k * 0.1)
    st.rpy[2] = ref - 1.0
    assert c._mpc_input(sim.t, st).cmd[2] == pytest.approx(float(c.cmd_f[2]) + wmax)
    # turning: no hold, the reference is dropped (new heading after the turn)
    c.cmd_raw = np.array([0.2, 0.0, 0.4])
    assert c._mpc_input(sim.t, st).cmd[2] == pytest.approx(float(c.cmd_f[2]))
    assert c._yaw_ref is None
    c.cmd_raw = np.array([0.2, 0.0, 0.0])
    c._mpc_input(sim.t, st)
    assert c._yaw_ref == pytest.approx(ref - 1.0)
    # off
    c.p["mpc_heading_hold_gain"] = 0.0
    st.rpy[2] = ref + 0.3
    assert c._mpc_input(sim.t, st).cmd[2] == pytest.approx(float(c.cmd_f[2]))
    # STAND: zero command (the stand hold keeps the heading)
    sim2 = _stand(Sim())
    assert sim2.ctrl._mpc_input(sim2.t, sim2.st).cmd == (0.0, 0.0, 0.0)


def test_retarget_next_to_an_obstacle_jumps_whole_or_keeps_the_target():
    """A rate-limited re-target point between two planned footholds was never
    checked by the foothold planner: when the plan jumps across a pipe it lies
    on the pipe.  Next to higher terrain the jump is taken whole early in the
    swing (retarget_jump_until), later the current target is kept."""
    sim = _stand(Sim())
    sim.run(0.3, cmd=(0.3, 0.0, 0.0))
    c = sim.ctrl
    for _ in range(20):
        if any(tr is not None for tr in c.swing):
            break
        sim.run(DT, cmd=(0.3, 0.0, 0.0))
    leg = next(i for i in range(4) if c.swing[i] is not None)
    tr = c.swing[leg]
    p1 = tr.p1.copy()
    t_td = c.swing_t0[leg] + tr.duration
    step = c.p["max_retarget_step"]
    xp = p1[0] + step                                   # the rate-limited point lands on the pipe
    pipe = FakeMap(lambda x, y: np.where(np.abs(x - xp) < 0.04, 0.08, 0.0))
    floor = FakeMap(lambda x, y: np.zeros_like(x))
    new = p1 + np.array([0.20, 0.0, 0.0])               # beyond the pipe, on the floor

    def plan():
        res = SimpleNamespace(ok=True, t=sim.t - 0.02,
                              foothold_events=[(leg, t_td - (sim.t - 0.02), new[0], new[1], new[2])])
        c.mpc_footholds = res
        c._retarget_stamp.pop(leg, None)

    plan()
    c._retarget(leg, 0.1, sim.t, pipe)
    assert np.allclose(tr.p1, new)                      # early: whole jump
    tr.retarget(p1)
    plan()
    c._retarget(leg, 0.5, sim.t, pipe)
    assert np.allclose(tr.p1, p1)                       # late: the checked target is kept
    plan()
    c._retarget(leg, 0.5, sim.t, floor)
    assert tr.p1[0] == pytest.approx(p1[0] + step)      # flat: rate limited as before
    tr.retarget(p1)
    plan()
    c._retarget(leg, 0.5, sim.t, None)
    assert tr.p1[0] == pytest.approx(p1[0] + step)      # no map: rate limited


def test_swing_guard_keeps_a_lagging_foot_above_the_obstacle():
    """swing_guard_*: while the MEASURED foot is still over / next to terrain
    higher than its landing point in the descent half of the swing, the
    commanded height stays above that terrain (the servo lags the command by
    ~30 ms: Gazebo legs were driven into the pipe's flank with 300-450 N)."""
    sim, c, stance, _leg = _walking_stance()
    leg = next(i for i in range(4) if c.swing[i] is not None)
    tr = c.swing[leg]
    meas = c._feet_world_measured(sim.st)
    c._feet_meas_now = meas
    xm = float(meas[leg, 0])
    pipe = FakeMap(lambda x, y: np.where(np.abs(x - xm) < 0.04, 0.10, 0.0))
    floor = FakeMap(lambda x, y: np.zeros_like(x))
    tr.apex_z = 0.20
    z_low = float(tr.p1[2]) + 0.01                         # late in the descent
    top = 0.10 + rm.FOOT_RADIUS + c.p["swing_guard_clearance"]
    assert c._swing_guard_z(leg, 0.9, tr, z_low, pipe) == pytest.approx(top)
    assert c._swing_guard_z(leg, 0.3, tr, z_low, pipe) == pytest.approx(z_low)      # rise half: trajectory
    assert c._swing_guard_z(leg, 0.9, tr, z_low, floor) == pytest.approx(z_low)     # nothing under the foot
    assert c._swing_guard_z(leg, 0.9, tr, z_low, None) == pytest.approx(z_low)      # no map
    assert c._swing_guard_z(leg, 0.9, tr, 0.18, pipe) == pytest.approx(0.18)        # already above
    tr.apex_z = 0.12
    assert c._swing_guard_z(leg, 0.9, tr, z_low, pipe) == pytest.approx(0.12)       # never above the apex
    # landing on the obstacle itself: no guard
    tr.p1 = tr.p1.copy()
    tr.p1[2] = 0.10 + rm.FOOT_RADIUS
    assert c._swing_guard_z(leg, 0.9, tr, 0.11, pipe) == pytest.approx(0.11)
    c.p["swing_guard_clearance"] = 0.0
    tr.p1[2] = rm.FOOT_RADIUS
    assert c._swing_guard_z(leg, 0.9, tr, z_low, pipe) == pytest.approx(z_low)


def test_integrated_body_target_blends_the_sideways_velocity_towards_the_reference():
    """mpc_lateral_ref_blend: the integrated target moves with the MPC-predicted
    velocity forward, but sideways (reference heading frame) it is blended
    towards the reference - a prediction that keeps the body's outward drift
    after a turn no longer carries the target along with it (Gazebo: 0.05-0.11
    m/s sideways drift, feet against the wall of the 0.63 m gap)."""
    sim = _stand(Sim())
    sim.run(1.0, cmd=(0.2, 0.0, 0.0))
    c, st = sim.ctrl, sim.st
    yaw = float(st.rpy[2])
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    lat = np.array([-math.sin(yaw), math.cos(yaw)])
    x = np.zeros(13)
    x[2] = yaw
    x[9:11] = 0.2 * fwd + 0.10 * lat                     # predicted: drifting sideways at 0.1 m/s
    xr = np.zeros(13)
    xr[2] = yaw
    xr[5] = float(st.pos[2])
    xr[9:11] = 0.25 * fwd                                 # reference: straight ahead
    res = SimpleNamespace(t=sim.t, state_at=lambda _t: x, ref_at=lambda _t: xr)
    dt = 0.01

    def target_velocity(blend):
        c.p["mpc_lateral_ref_blend"] = blend
        xy0 = np.asarray(st.pos[:2], dtype=float).copy()
        c._walk_target = (xy0.copy(), yaw)
        pos, _ = c._integrated_body_target(sim.t, dt, st, res)
        return (pos[:2] - xy0) / dt

    v0 = target_velocity(0.0)
    assert float(v0 @ fwd) == pytest.approx(0.2, abs=1e-9) and float(v0 @ lat) == pytest.approx(0.10, abs=1e-9)
    v5 = target_velocity(0.5)
    assert float(v5 @ fwd) == pytest.approx(0.2, abs=1e-9)          # forward: the prediction
    assert float(v5 @ lat) == pytest.approx(0.05, abs=1e-9)         # sideways: half way to the reference
    v1 = target_velocity(1.0)
    assert float(v1 @ lat) == pytest.approx(0.0, abs=1e-9)
    assert C.DEFAULT_PARAMS["mpc_lateral_ref_blend"] == 0.5


def test_stance_deflection_limit_bounds_the_horizontal_servo_force():
    """mpc_stance_rel_error: the horizontal deflection of a stance leg (commanded
    foot in the body-target frame - measured foot - static sag of its weight
    share) is limited, whatever part comes from the anchor and whatever from the
    body target's lead (Gazebo: a wall-jammed knee shoved the rear, the opposite
    stance foot was asked for ~220 N sideways and slid -> spin into the wall)."""
    sim, c, stance, leg = _walking_stance({"mpc_stance_rel_error": 0.02})
    st = sim.st
    st.rpy = np.array([0.0, 0.0, float(st.rpy[2])])
    q = np.asarray(st.q, dtype=float).reshape(4, 3)
    kp = float(c.p["servo_kp"])
    n = int(stance.sum())
    f_static = np.array([0.0, 0.0, -float(c.p["mpc_mass"]) * c.mpc.cfg.gravity / n])

    def rel(lg, foot_w, pos_cmd, rpy_cmd):
        R = C.euler_to_rot(*rpy_cmd)
        J = kin.leg_jacobian(q[lg], lg)
        return R.T @ (np.asarray(foot_w) - pos_cmd) - kin.leg_fk(q[lg], lg) - J @ J.T @ f_static / kp

    feet = c._feet_world_measured(st)
    swing_legs = np.flatnonzero(~stance)
    # body target 5 cm ahead + 0.05 rad of yaw lead: the stance feet are asked for ~5 cm of deflection
    pos_cmd = np.asarray(st.pos, dtype=float) + np.array([0.05, 0.01, 0.003])
    rpy_cmd = np.asarray(st.rpy, dtype=float) + np.array([0.0, 0.0, 0.05])
    out = c._stance_deflection_limit(st, stance, pos_cmd, rpy_cmd, feet)
    for lg in np.flatnonzero(stance):
        before, after = rel(lg, feet[lg], pos_cmd, rpy_cmd), rel(lg, out[lg], pos_cmd, rpy_cmd)
        assert float(np.hypot(*before[:2])) > 0.03
        assert float(np.hypot(*after[:2])) == pytest.approx(0.02, abs=1e-9)
        assert np.allclose(after[:2] / np.hypot(*after[:2]), before[:2] / np.hypot(*before[:2]))
        assert after[2] == pytest.approx(before[2], abs=1e-12)          # vertical deflection untouched
    assert np.allclose(out[swing_legs], feet[swing_legs])
    # small deflections are left alone; 0 = off
    pos_small = np.asarray(st.pos, dtype=float) + np.array([0.005, 0.0, 0.0])
    feet_small = c._stance_deflection_limit(st, stance, pos_small, st.rpy, feet)
    lg0 = int(np.flatnonzero(stance)[0])
    if float(np.hypot(*rel(lg0, feet[lg0], pos_small, st.rpy)[:2])) < 0.02:
        assert np.allclose(feet_small[lg0], feet[lg0])
    c.p["mpc_stance_rel_error"] = 0.0
    assert np.allclose(c._stance_deflection_limit(st, stance, pos_cmd, rpy_cmd, feet), feet)


def test_mpc_turn_in_place_about_a_point_behind_the_base():
    """mpc_turn_pivot_x: the backward rear knees (x ~ -0.43 m) sweep a ~0.47 m
    circle when the robot turns in place about the base origin (Gazebo: the rear
    knee ran into the wall at the C-OBS1 gap exit); about x_p = -0.15 m the
    MPC command gets the sideways velocity w * 0.15 (towards the inside of the
    turn), faded out with the commanded translation, and the point x_p of the
    body stays put while the base circles it."""
    xp = C.DEFAULT_PARAMS["mpc_turn_pivot_x"]
    assert xp < 0.0
    sim = _stand(Sim())
    c = sim.ctrl
    assert c._mpc_input(sim.t, sim.st).cmd == (0.0, 0.0, 0.0)          # STAND: no pivot motion
    wz = 0.5
    pivot = []
    sim.run(0.8, cmd=(0.0, 0.0, wz))
    assert c.mode == "WALK"
    c.cmd_raw = np.array([0.0, 0.0, wz])
    c.cmd_f = np.array([0.0, 0.0, wz])
    assert c._turn_pivot_vy() == pytest.approx(wz * -xp)
    c.cmd_f = np.array([0.05, 0.0, wz])
    assert c._turn_pivot_vy() == pytest.approx(0.5 * wz * -xp)            # half faded
    c.cmd_f = np.array([0.2, 0.0, wz])
    assert c._turn_pivot_vy() == 0.0                                      # walking: plain turn
    c.cmd_f = np.array([0.0, 0.0, -wz])
    assert c._turn_pivot_vy() == pytest.approx(-wz * -xp)                 # right turn: to the right
    # a walk slowed down by the torque-saturation speed scaling is no in-place turn
    c.cmd_raw = np.array([0.25, 0.0, wz])
    c.cmd_f = np.array([0.02, 0.0, 0.3 * wz])
    assert c._turn_pivot_vy() == 0.0
    c.cmd_raw = np.array([0.0, 0.0, wz])
    base = []
    for _ in range(300):                                                  # 3 s ~ 1.5 rad
        sim.run(DT, cmd=(0.0, 0.0, wz))
        yaw = float(sim.st.rpy[2])
        base.append(sim.st.pos[:2].copy())
        pivot.append(sim.st.pos[:2] + xp * np.array([math.cos(yaw), math.sin(yaw)]))
    base, pivot = np.array(base), np.array(pivot)
    yaw_turned = float(sim.st.rpy[2])
    assert yaw_turned > 1.0
    span = lambda a: float(np.max(np.hypot(*(a - a.mean(axis=0)).T)))
    assert span(pivot) < 0.5 * span(base)                                 # the pivot stays, the base circles it
    assert span(base) > 0.05
    # 0 = turn about the base origin
    c.p["mpc_turn_pivot_x"] = 0.0
    c.cmd_f = np.array([0.0, 0.0, wz])
    assert c._turn_pivot_vy() == 0.0


def test_stable_foothold_keeps_an_unseen_nominal_cell():
    """Gazebo mission check (dock start): the depth camera looks ahead, the cells
    under / behind the body are unseen; the foothold planner then moved the hind
    touch-down points up to 0.16 m forward to the nearest SEEN cell - the hind
    feet ended under the body and the robot tipped over backwards.  An unseen
    nominal cell keeps the nominal foothold (``foothold_keep_unseen``)."""

    class Forward:                                      # a planner that always picks a cell 0.1 m ahead
        def __init__(self):
            self.calls = 0

        def select(self, emap, xy, z):
            self.calls += 1
            return np.array([xy[0] + 0.1, xy[1], 0.0]), 0.0, True

    seen_ahead = FakeMap(lambda x, y: np.where(np.asarray(x) > 0.0, 0.0, np.nan))
    st = C.RobotState()
    st.pos = np.array([0.0, 0.0, 0.34])
    st.rpy = np.zeros(3)
    for keep in (True, False):
        ctrl = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "stable",
                                       "foothold_keep_unseen": keep})
        ctrl.planner = Forward()
        ctrl._terrain_now = np.zeros(5)
        fh = ctrl._stable_terrain(0.01, st, seen_ahead, 0.0)[3]
        hind = fh(2, np.array([-0.22, 0.13]))
        front = fh(0, np.array([0.22, 0.13]))
        assert front == pytest.approx([0.32, 0.13])              # seen: the planner's choice
        if keep:
            assert hind == pytest.approx([-0.22, 0.13]) and ctrl.planner.calls == 1
        else:
            assert hind == pytest.approx([-0.12, 0.13]) and ctrl.planner.calls == 2
    assert C.DEFAULT_PARAMS["foothold_keep_unseen"] is True


def test_stable_trot_slows_down_at_a_bump_but_not_on_a_ramp():
    """Obstacle pace: the stable trot period becomes stable_obstacle_trot_period while
    a bump (lying pipe) is under / ahead of the legs and for stable_obstacle_hold s
    after; a ramp (a terrain line) keeps the normal pace; phase stays continuous."""
    ctrl = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "stable"})
    ctrl.mode = "WALK"
    ctrl.ground_z = 0.0
    st = C.RobotState()
    st.pos = np.array([0.0, 0.0, 0.34])
    st.rpy = np.zeros(3)
    pipe = FakeMap(lambda x, y: np.where(np.abs(np.asarray(x) - 0.3) < 0.04, 0.08, 0.0))
    ramp = FakeMap(lambda x, y: 0.2 * np.asarray(x, dtype=float))
    flat = FakeMap(lambda x, y: 0.0 * np.asarray(x, dtype=float))
    assert ctrl._obstacle_bump(st, pipe) > 0.06
    assert abs(ctrl._obstacle_bump(st, ramp)) < 1e-6 and abs(ctrl._obstacle_bump(st, flat)) < 1e-9
    assert math.isnan(ctrl._obstacle_bump(st, None))
    period = lambda: ctrl.gait_stable.gaits["trot"].period             # noqa: E731
    ctrl._update_stable_pace(0.0, st, ramp)
    assert period() == pytest.approx(0.4)
    ctrl._update_stable_pace(0.1, st, pipe)
    assert period() == pytest.approx(0.5) and ctrl._pace_period == pytest.approx(0.5)
    ctrl._update_stable_pace(0.6, st, flat)
    assert period() == pytest.approx(0.5)                                # held
    ctrl._update_stable_pace(1.2, st, flat)
    assert period() == pytest.approx(0.4)
    off = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "stable",
                                  "stable_obstacle_trot_period": 0.0})
    off.mode = "WALK"
    off._update_stable_pace(0.1, st, pipe)
    assert off.gait_stable.gaits["trot"].period == pytest.approx(0.4)


def test_stable_terrain_pitch_ignores_a_pipe_under_a_foothold():
    """stable_terrain_pitch_window: the support height of each nominal foothold is
    a low quantile of the map along the heading around it - a 0.08 m pipe under
    both front foothold points is not a 10 deg ramp (the body pitched nose-up over
    C-OBS5 and a front foot hung in the air), a real ramp still pitches the body."""
    st = C.RobotState()
    st.pos = np.array([0.0, 0.0, 0.34])
    st.rpy = np.zeros(3)
    on = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "stable",
                                 "stable_terrain_pitch_window": 0.12})
    off = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "stable",
                                  "stable_terrain_pitch_window": 0.0})
    xf = float(np.mean(on._nominal_world_xy(st)[:2, 0]))                 # front foothold x
    pipe = FakeMap(lambda x, y: np.where(np.abs(np.asarray(x, dtype=float) - xf) < 0.04, 0.08, 0.0))
    ramp = FakeMap(lambda x, y: -0.2 * np.asarray(x, dtype=float))        # downhill towards +x
    for ctrl in (on, off):
        ctrl.ground_z = 0.0
        ctrl._terrain_now = ctrl._query_terrain(pipe, st)
    assert off._support_heights(pipe, st, 0.0) is None
    low = on._support_heights(pipe, st, 0.12)
    assert np.allclose(low, 0.0, atol=1e-9)
    p_off = off._terrain_pitch(off._terrain_now[1:], off.foot_w)
    p_on = on._terrain_pitch(low, on.foot_w)
    assert p_off < -math.radians(8.0) and abs(p_on) < 1e-9
    # a ramp: front and rear shift alike, the pitch follows the slope as before
    for ctrl in (on, off):
        ctrl.ground_z = 0.0
    ramp_on = on._terrain_pitch(on._support_heights(ramp, st, 0.12), on.foot_w)
    ramp_off = off._terrain_pitch(off._query_terrain(ramp, st)[1:], off.foot_w)
    assert ramp_on == pytest.approx(ramp_off, abs=1e-6) and ramp_on > math.radians(10.0)
    # unknown map: NaN rows (filled with the ground estimate by _terrain_pitch)
    unknown = FakeMap(lambda x, y: np.full(np.shape(x), np.nan))
    assert np.all(np.isnan(on._support_heights(unknown, st, 0.12)))
    assert C.DEFAULT_PARAMS["stable_terrain_pitch_window"] >= 0.0
