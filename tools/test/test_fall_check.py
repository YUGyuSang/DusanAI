"""Tests of the headless Gazebo fall-recovery check (tools/fall_check.py).

* always (pure python): the cases (spawn attitude, knock direction / size,
  plant-world spots), the operator (walk before the fall, stop + knock, walk
  after the recovery) and the pass / fail logic;
* with the gz python bindings (``GZ_PYTHON`` = path of a python with
  gz.sim8 + gz.transport13, e.g. ``C:\\gzmamba\\envs\\gz\\python.exe``; skipped
  otherwise): a SHORT run of the side case - the robot spawned lying on its
  side gets up with its own legs (no teleport) in gz-sim (DART).
"""
import json
import math
import os
import subprocess
import threading

import numpy as np
import pytest

import fall_check as FC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CLI = os.path.join(ROOT, "tools", "fall_check.py")
DEG = math.pi / 180.0


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def test_cases_are_deterministic_and_cover_the_list():
    assert set(FC.DEFAULT_CASES) == {"side", "back", "trot", "roll", "push", "wall", "ramp_walk", "ramp_push",
                                     "mission", "pipe_back", "ramp_edge"}
    assert FC.canonical("trot-fall") == "trot" and FC.canonical("ramp-push") == "ramp_push"
    assert FC.canonical("pipe-back") == "pipe_back" and FC.canonical("ramp-edge") == "ramp_edge"
    with pytest.raises(KeyError):
        FC.canonical("cartwheel")
    for c in FC.CASES:
        assert FC.case_setup(c, 3) == FC.case_setup(c, 3)
        assert FC.case_setup(c, 3) != FC.case_setup(c, 4)
        assert FC.case_duration(c) >= 60.0


def test_lying_spawns_use_the_odometry_attitude():
    for seed in range(10):
        s = FC.case_setup("side", seed)
        roll, pitch = s["spawn_rp"]
        assert 75 * DEG <= abs(roll) <= 105 * DEG and abs(pitch) <= 0.2 + 1e-9
        assert math.copysign(1.0, roll) == (1.0 if seed % 2 == 0 else -1.0)     # both sides
        assert s["params"] == {"use_imu_orientation": False} and not s["pushes"]
        b = FC.case_setup("back", seed)
        assert 160 * DEG <= b["spawn_rp"][0] <= 200 * DEG
        assert b["params"] == {"use_imu_orientation": False}
    # the knock-down cases spawn upright and keep the IMU attitude
    for c in ("trot", "roll", "push", "wall", "ramp_walk", "ramp_push", "mission"):
        s = FC.case_setup(c, 0)
        assert s["spawn_rp"] == (0.0, 0.0) and "use_imu_orientation" not in s["params"]


def _heading(yaw):
    return np.array([math.cos(yaw), math.sin(yaw)])


def test_knock_directions_and_sizes():
    for seed in range(10):
        tr = FC.case_setup("trot", seed)
        (t0, t1, f, off), = tr["pushes"]
        yaw = tr["spawn"][3]
        assert t1 - t0 == pytest.approx(0.2) and off is None and tr["pre_cmd"][2] == (0.4, 0.0, 0.0)
        assert ("gait" in tr["commands"][0][1]) and 400 <= np.linalg.norm(f) <= 500
        assert abs(float(np.dot(f[:2], _heading(yaw)))) < 1e-6                    # lateral
        ro = FC.case_setup("roll", seed)
        (_, _, f, off), = ro["pushes"]
        assert off == (0.0, 0.0, 0.3) and abs(float(np.dot(f[:2], _heading(ro["spawn"][3])))) < 1e-6
        pu = FC.case_setup("push", seed)
        (_, _, f, off), = pu["pushes"]
        assert float(np.dot(f[:2], _heading(pu["spawn"][3]))) == pytest.approx(-np.linalg.norm(f))   # backward
        wa = FC.case_setup("wall", seed)
        (_, _, f, _), = wa["pushes"]
        assert f[1] > 0 and f[0] == 0.0 and wa["terrain"] == "plant"              # towards the north wall
        rp = FC.case_setup("ramp_push", seed)
        assert not rp["pushes"] and rp["stop_knock"]["stop_x"] == FC.RAMP_SPOT[0]
        assert 300 <= abs(rp["stop_knock"]["force"]) <= 400
        rw = FC.case_setup("ramp_walk", seed)
        assert rw["pre_cmd"][2][0] > 0 and abs(math.cos(rw["spawn"][3]) + 1.0) < 0.02          # facing west
    m = FC.case_setup("mission", 1)
    assert m["mission"] == "x_control_upper" and m["map_source"] == "depth"
    # the plant cases see the terrain through the depth camera like on Ubuntu, the flat ones need no map
    for c in ("wall", "ramp_walk", "ramp_push"):
        assert FC.case_setup(c, 0)["map_source"] == "depth"
    for c in ("side", "back", "trot", "roll", "push"):
        assert FC.case_setup(c, 0)["map_source"] == ""


def test_plant_spots_are_on_the_floor_the_wall_and_the_ramp():
    from gazebo_check import world as W
    hf = W.HeightField.plant()
    # wall case: lower corridor floor, the wall 0.4 .. 0.6 m to the north
    x, y = FC.WALL_SPOT
    assert hf.height(x, y) == pytest.approx(FC.Z_LOWER, abs=0.01)
    ys = np.arange(y, y + 1.0, 0.02)
    wall_y = ys[np.argmax(hf.height(np.full_like(ys, x), ys) > 0.5)]
    assert 0.4 <= wall_y - y <= 0.6
    # ramp cases: start on the lower floor east of the ramp, the stop point on the slope
    x, y = FC.RAMP_WALK_SPOT
    assert hf.height(x, y) == pytest.approx(FC.Z_LOWER, abs=0.01)
    x, y = FC.RAMP_SPOT
    z0, z1 = float(hf.height(x - 0.5, y)), float(hf.height(x + 0.5, y))
    assert math.degrees(math.atan2(z0 - z1, 1.0)) == pytest.approx(11.8, abs=1.0)


# ---------------------------------------------------------------------------
# operator
# ---------------------------------------------------------------------------
class _Rec:
    def __init__(self):
        self.fall_count = 0
        self.successes = 0


class _Harness:
    def __init__(self):
        self.ctrl = type("C", (), {})()
        self.ctrl.recovery = _Rec()
        self._lock = threading.Lock()
        self._odom = None
        self.wrenches = []
        self.commands = []


def test_operator_walks_before_the_fall_and_after_the_recovery():
    h = _Harness()
    op = FC.AfterRecoveryWalk((4.0, 30.0, (0.4, 0.0, 0.0)))
    op.bind(h)
    assert op(3.0) == (0.0, 0.0, 0.0) and op(5.0) == (0.4, 0.0, 0.0)
    h.ctrl.recovery.fall_count = 1                         # down: no command
    assert op(6.0) == (0.0, 0.0, 0.0)
    h.ctrl.recovery.successes = 1                          # recovered at t = 10
    assert op(10.0) == (0.0, 0.0, 0.0)
    assert op(10.0 + FC.WALK_DELAY + 0.1) == (FC.WALK_SPEED, 0.0, 0.0)
    assert op(10.0 + FC.WALK_DELAY + FC.WALK_TIME + 0.1) == (0.0, 0.0, 0.0)
    assert not h.commands
    # knocked over again by the walk: after the next stand-up it walks again
    h.ctrl.recovery.successes = 2                          # recovered again at t = 30
    assert op(30.0) == (0.0, 0.0, 0.0) and op.t_rec == 30.0
    assert op(30.0 + FC.WALK_DELAY + 0.1) == (FC.WALK_SPEED, 0.0, 0.0)
    # plant world: trot in place (walk / stand commands), no velocity
    h = _Harness()
    h.commands = []
    op = FC.AfterRecoveryWalk(None, in_place=True)
    op.bind(h)
    assert op(5.0) == (0.0, 0.0, 0.0)
    h.ctrl.recovery.successes = 1
    for t in (10.0, 12.5, 15.0, 20.0):
        assert op(t) == (0.0, 0.0, 0.0)
    assert h.commands == [(10.0 + FC.WALK_DELAY, "walk"), (10.0 + FC.WALK_DELAY + FC.WALK_TIME, "stand")]
    assert set(FC.IN_PLACE_CASES) == {"wall", "ramp_walk", "ramp_push", "pipe_back", "ramp_edge"}


def test_operator_stops_on_the_ramp_and_knocks():
    h = _Harness()
    knock = {"stop_x": 49.1, "delay": 2.5, "force": 350.0, "offset": (0.0, 0.0, 0.3)}
    op = FC.AfterRecoveryWalk((4.0, 60.0, (0.3, 0.0, 0.0)), knock)
    op.bind(h)
    h._odom = (0.0, np.array([50.0, 0.8, -0.6]), np.array([0.0, 0.0, math.pi]), None, None)
    assert op(20.0) == (0.3, 0.0, 0.0) and not h.wrenches
    h._odom = (0.0, np.array([49.05, 0.8, -0.4]), np.array([0.0, 0.0, math.pi]), None, None)
    assert op(30.0) == (0.0, 0.0, 0.0)
    (t0, t1, f, off), = h.wrenches
    assert t0 == pytest.approx(32.5) and t1 == pytest.approx(32.7) and off == (0.0, 0.0, 0.3)
    assert f[0] == pytest.approx(0.0, abs=1e-9) and f[1] == pytest.approx(-350.0)   # lateral (left of yaw pi)
    assert op(31.0) == (0.0, 0.0, 0.0) and len(h.wrenches) == 1


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _arrays(t_end=30.0, final_mode="STAND", h=0.34, roll=0.0, walk=True):
    t = np.arange(0.0, t_end, 0.01)
    n = len(t)
    pos = np.zeros((n, 3))
    pos[:, 2] = h
    pos[:, 0] = 0.3 * np.clip(t - 14.0, 0.0, 6.0)                  # walks 2 .. 8 s after the recovery at 12 s
    rpy = np.zeros((n, 3))
    rpy[:, 0] = roll
    mode = np.array(["STAND"] * (n - 1) + [final_mode], dtype=object)
    if walk:
        mode[(t >= 14.0) & (t < 20.0)] = "WALK"
    rs = np.array(["NORMAL"] * n, dtype=object)
    rs[(t > 5.0) & (t < 12.0)] = "DAMP"
    return {"t": t, "pos": pos, "rpy": rpy, "ground": np.zeros(n), "mode": mode, "rec_state": rs}


def _ev(name, t, roll=0.0):
    return {"event": name, "t": t, "x": 0.0, "y": 0.0, "roll": roll, "pitch": 0.0}


def test_evaluate_pass_and_failure_reasons():
    info = {"events": [_ev("fall", 5.0, 1.5), _ev("recovered", 12.0)], "fall_count": 1, "attempts_total": 1,
            "successes": 1, "assisted_resets": 0, "attempts_at_recovery": 1}
    m = FC.evaluate("side", _arrays(), info)
    assert m["ok"] and m["fell"] and m["first_try"] and m["ttr"] == pytest.approx(7.0)
    assert m["reason"] == "recovered physically" and m["walk_after_dist"] == pytest.approx(1.8, abs=0.05)
    assert m["seq"] == ["NORMAL", "DAMP", "NORMAL"]
    # a teleport is not a physical recovery
    m = FC.evaluate("side", _arrays(), dict(info, assisted_resets=1))
    assert not m["ok"] and "teleport" in m["reason"]
    # not standing at the end / not recovered / never fell
    assert not FC.evaluate("side", _arrays(final_mode="FALLEN"), info)["ok"]
    assert not FC.evaluate("side", _arrays(roll=0.5), info)["ok"]
    m = FC.evaluate("side", _arrays(), dict(info, events=[_ev("fall", 5.0)], successes=0))
    assert not m["ok"] and m["reason"] == "not recovered within the run"
    m = FC.evaluate("roll", _arrays(), dict(info, events=[], fall_count=0, successes=0))
    assert not m["ok"] and not m["fell"] and m["reason"].startswith("no fall")
    # got up but never walked again
    m = FC.evaluate("side", _arrays(walk=False), info)
    assert not m["ok"] and m["walk_after_frac"] == 0.0 and "walk" in m["reason"]
    # mission: the viewpoint must be reached too
    assert FC.evaluate("mission", _arrays(), info, {"reached": 1.0, "final_state": "DONE"})["ok"]
    m = FC.evaluate("mission", _arrays(), info, {"reached": 0.0, "final_state": "PATROL"})
    assert not m["ok"] and "mission" in m["reason"]


def test_summary_rate_counts_only_runs_that_fell():
    runs = {"roll": {"0": {"ok": True, "fell": True, "ttr": 8.0, "first_try": True, "assisted": 0},
                     "1": {"ok": False, "fell": True, "ttr": None, "assisted": 0, "retries": 2},
                     "2": {"ok": False, "fell": False}}}
    s = FC.summarize(runs)["roll"]
    assert (s["runs"], s["fell"], s["ok"], s["first_try"], s["teleports"], s["retries"]) == (3, 2, 1, 1, 0, 2)
    assert s["rate"] == pytest.approx(0.5) and s["ttr_mean"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# closed loop in gz-sim (needs the gz python bindings)
# ---------------------------------------------------------------------------
GZ_PYTHON = os.environ.get("GZ_PYTHON", "")


@pytest.mark.skipif(not (GZ_PYTHON and os.path.exists(GZ_PYTHON)),
                    reason="set GZ_PYTHON to a python with the gz-sim 8 bindings (gz.sim8, gz.transport13)")
def test_side_fall_recovers_physically_in_gazebo(tmp_path):
    env = dict(os.environ)
    if os.name == "nt":       # conda layout: DLLs next to python.exe / Library/bin
        base = os.path.dirname(GZ_PYTHON)
        extra = [base, os.path.join(base, "Library", "mingw-w64", "bin"), os.path.join(base, "Library", "usr", "bin"),
                 os.path.join(base, "Library", "bin"), os.path.join(base, "Scripts")]
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        env["CONDA_PREFIX"] = base
    out = tmp_path / "fall.json"
    cmd = [GZ_PYTHON, CLI, "--cases", "side", "--seeds", "0", "--jobs", "1", "--duration", "30", "--label", "pytest",
           "--out", str(out), "--work-dir", str(tmp_path)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    block = json.loads(out.read_text(encoding="utf-8"))["runs"]["pytest"]
    r = block["cases"]["side"]["0"]
    assert r["fell"] and r["ok"], r["reason"]
    assert r["assisted"] == 0 and r["recoveries"] >= 1 and r["final_mode"] in ("STAND", "WALK")
    assert block["summary"]["side"]["teleports"] == 0


def test_pipe_back_and_ramp_edge_spawns():
    """pipe_back: on the back across the C-OBS5 pipe (layout), heading across it west / east, the pipe
    under the rear of the trunk (tail up: pitch > 0); ramp_edge: upright, hanging over the unrailed
    south edge of the ramp, nose up (facing north) on even seeds, nose down (south) on odd seeds."""
    import yaml
    lay = yaml.safe_load(open(os.path.join(ROOT, "simulation", "config", "world_layout.yaml"), encoding="utf-8"))
    pipe = [o for o in lay["obstacles"] if o["id"] == "C-OBS5"][0]
    assert tuple(pipe["center"]) == FC.PIPE_SPOT and pipe["yaw"] == FC.PIPE_YAW and pipe["diameter"] == FC.PIPE_D
    n = np.array([-math.sin(FC.PIPE_YAW), math.cos(FC.PIPE_YAW)])
    u = np.array([math.cos(FC.PIPE_YAW), math.sin(FC.PIPE_YAW)])
    for seed in range(10):
        s = FC.case_setup("pipe_back", seed)
        x, y, z, yaw = s["spawn"]
        roll, pitch = s["spawn_rp"]
        assert s["terrain"] == "plant" and s["map_source"] == "depth" and s["params"] == {"use_imu_orientation": False}
        assert abs(math.cos(roll) + 1.0) < 0.02 and 0.13 <= pitch <= 0.25 and 0.15 <= z <= 0.25
        head = np.array([math.cos(yaw), math.sin(yaw)])
        assert abs(float(head @ n)) > math.cos(0.36)                              # across the pipe
        rel = np.array([x, y]) - np.array(FC.PIPE_SPOT)
        side = float(rel @ n)                                                     # base ahead of the pipe axis
        assert side * float(head @ n) > 0.0 and 0.03 <= abs(side) <= 0.15        # (pipe under the rear) ...
        assert abs(float(rel @ u)) <= 0.6                                         # ... inside the pipe span
        assert (float(head @ n) > 0) == (seed % 2 == 0)                           # west / east
        e = FC.case_setup("ramp_edge", seed)
        x, y, z, yaw = e["spawn"]
        roll, pitch = e["spawn_rp"]
        north = seed % 2 == 0
        assert abs(roll) <= 0.1 + 1e-9 and (pitch < -0.6 if north else pitch > 0.6)
        assert (math.sin(yaw) > 0.9) == north and 49.3 <= x <= 50.1 and abs(y - FC.RAMP_EDGE_Y) < 0.1
        assert e["terrain"] == "plant" and e["params"] == {"use_imu_orientation": False}
