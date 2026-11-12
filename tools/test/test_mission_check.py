"""Tests of the headless Gazebo whole-mission check (tools/mission_check.py).

* always (pure python): where a position lies on the route (area, nearest
  debris), the fall / stall analysis of the per-step logs, the completion
  verdict, the sub-plans of ``--leg`` / ``--legs`` and the route segments;
* with the gz python bindings (``GZ_PYTHON`` = path of a python with
  gz.sim8 + gz.transport13, e.g. ``C:\\gzmamba\\envs\\gz\\python.exe``; skipped
  otherwise): a SHORT route segment in the plant world (spawn next to the
  dock, one goal 1.5 m away) with the whole mission stack in the loop, and a
  corridor segment on which the robot is knocked over twice in a row
  (``--push``): it gets up by itself and the patrol carries on to the goals.
"""
import json
import math
import os
import subprocess

import numpy as np
import pytest

import mission_check as MC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CLI = os.path.join(ROOT, "tools", "mission_check.py")


@pytest.fixture(scope="module")
def layout():
    return MC._layout()


def test_route_areas_and_debris(layout):
    assert MC.area_of(61.9, 0.0, layout) == "dock"
    assert MC.area_of(57.0, 0.4, layout) == "lower corridor"
    assert MC.area_of(49.0, 0.8, layout) == "ramp"
    assert MC.area_of(48.0, -1.0, layout) == "stairs"
    assert MC.area_of(30.0, 0.0, layout) == "upper corridor"
    assert MC.area_of(15.3, 0.0, layout) == "corridor door"
    assert MC.area_of(12.0, 1.0, layout) == "back room"
    assert MC.area_of(9.8, 0.5, layout) == "partition door"
    assert MC.area_of(0.0, 0.0, layout) == "factory"
    assert MC.nearest_debris(22.0, 0.3, layout)[:2] == ("C-OBS5", "pipe")
    assert MC.nearest_debris(4.5, 0.0, layout)[:2] == ("F-OBS5", "rubble")
    assert MC.nearest_debris(55.0, 1.0, layout)[:2] == ("C-OBS2", "pallet")
    assert MC.nearest_debris(44.0, 0.0, layout) is None


def _arrays(n=400, dt=0.01, fall_at=None):
    t = np.arange(n) * dt
    pos = np.zeros((n, 3))
    pos[:, 0] = 23.0 - 0.25 * t
    pos[:, 1] = 0.3
    pos[:, 2] = 0.34
    rpy = np.zeros((n, 3))
    rpy[:, 2] = math.pi
    mode = np.array(["WALK"] * n, dtype=object)
    if fall_at is not None:
        k = int(fall_at / dt)
        rpy[k:, 0] = np.linspace(0.0, 2.5, n - k)
        mode[k + 30:] = "FALLEN"
    return {"t": t, "pos": pos, "rpy": rpy, "vel": np.tile([-0.25, 0.0, 0.0], (n, 1)),
            "cmd": np.tile([0.25, 0.0, 0.05], (n, 1)), "mode": mode,
            "gait": np.array(["trot"] * n, dtype=object), "gait_phase": np.linspace(0, 1, n) % 1.0,
            "terrain_class": np.array(["rough"] * n, dtype=object), "ground": np.zeros(n)}


def test_fall_analysis_locates_the_onset(layout):
    A = _arrays(fall_at=3.0)
    trace = [(round(k * 0.1, 2), 23.0 - 0.025 * k, 0.3, 0.34, math.pi, 0, 0, "PATROL", "navigate", "patrol",
              0.25, 0.0, 0.05, "WALK", "NORMAL") for k in range(40)]
    ev = [{"event": "fall", "t": 3.6, "x": 22.1, "y": 0.3, "roll": 1.0, "pitch": 0.0},
          {"event": "recovered", "t": 12.0, "x": 22.1, "y": 0.3, "roll": 0.0, "pitch": 0.0}]
    f = MC.analyse_falls(A, trace, ev, layout)
    assert len(f) == 1
    f = f[0]
    assert 3.0 <= f["t_onset"] <= 3.6 and f["t"] == 3.6
    assert f["area"] == "upper corridor" and f["debris"]["id"] == "C-OBS5"
    assert f["mission"] == "PATROL/navigate" and f["mux"] == "patrol" and f["mode_before"] == "WALK"
    assert f["speed"] == pytest.approx(0.25, abs=1e-6) and f["cmd"] == [0.25, 0.0, 0.05]
    assert f["time_to_recover"] == pytest.approx(8.4)
    assert MC.analyse_falls(A, trace, [], layout) == []


def test_stall_analysis(layout):
    rows = []
    for k in range(400):
        t = k * 0.1
        x = 30.0 - 0.02 * min(k, 100)                 # moves 2 m, then stands still while "navigating"
        rows.append((t, x, 0.0, 0.34, math.pi, 0, 0, "PATROL", "navigate", "patrol", 0.3, 0, 0, "WALK", "NORMAL"))
    st = MC.analyse_stalls(rows, layout)
    assert len(st) == 1 and st[0]["area"] == "upper corridor"
    assert st[0]["t0"] == pytest.approx(10.0, abs=1.5) and st[0]["duration"] >= 20.0
    # inspecting (not driving) or fallen: no stall
    rows2 = [r[:7] + ("INSPECTING", "dwell") + r[9:] for r in rows]
    assert MC.analyse_stalls(rows2, layout) == []
    rows3 = [r[:13] + ("RECOVERING", "SELF_RIGHT") for r in rows]
    assert MC.analyse_stalls(rows3, layout) == []


def test_completion_verdict():
    ok = {"final_state": "DOCKED", "completed": list("ABCDEFGH"), "missed": [], "seek_result": {"result": "FOUND"},
          "teleports": 0}
    assert MC.completion(ok, 8)["complete"]
    for k, v in (("final_state", "PATROL"), ("missed", ["H"]), ("seek_result", {"result": "FAILED"}),
                 ("teleports", 1), ("completed", list("ABC"))):
        r = dict(ok, **{k: v})
        assert not MC.completion(r, 8)["complete"], k
    s = MC.summarize([dict(ok, complete=True, falls=1, recoveries=1, stalls=[], mission_time=700.0),
                      dict(ok, complete=False, falls=3, recoveries=3, stalls=[{}], mission_time=None)])
    assert s["complete"] == 1 and s["falls_total"] == 4 and s["falls_mean"] == 2.0 and s["stalls"] == 1
    assert s["mission_time_mean"] == 700.0


def test_sub_plans_and_segments():
    pj = {"order": ["A", "B", "C"], "legs": [[1], [2], [3], [4]],
          "viewpoints": [{"gauge_id": g, "x": float(i), "y": 0.0, "yaw": 0.0} for i, g in enumerate("ABC")]}
    sub, start, cmd, stop = MC._sub_plan(pj, 1, 0)
    assert sub is pj and start is None and cmd == "start" and stop is None
    sub, start, cmd, stop = MC._sub_plan(pj, 2, 1)
    assert sub["order"] == ["B", "C"] and sub["legs"] == [[2], [3], [4]] and start == (0.0, 0.0, 0.0)
    assert cmd == "start" and stop == "B"
    sub, start, cmd, stop = MC._sub_plan(pj, 4, 0)                 # the way home
    assert cmd == "rth" and start == (2.0, 0.0, 0.0)
    seg = MC._segment_plan(["20.0,0.3", "-1.5,2,3.14"], (24.0, 0.3))
    assert seg["order"] == ["P1", "P2"] and seg["viewpoints"][0]["yaw"] == pytest.approx(math.pi)
    assert seg["viewpoints"][1]["x"] == -1.5 and seg["viewpoints"][1]["yaw"] == pytest.approx(3.14)


class _Rec:
    def __init__(self):
        self.state, self.fall_count, self.successes = "NORMAL", 0, 0


class _Ctrl:
    def __init__(self):
        self.mode, self.recovery = "WALK", _Rec()


class _Mission:
    def __init__(self, base="PATROL", phase="navigate"):
        self.base, self.phase = base, phase


def test_pusher_knocks_in_order_and_retries():
    with pytest.raises(ValueError):
        MC.parse_push("jump:1")
    assert MC.parse_push("after_stand:1.5") == {"trig": "after_stand", "delay": 1.5}
    p = MC.Pusher(["nav:2.0", "after_stand:1.0"], seed=1)
    c, m, w = _Ctrl(), _Mission(), []
    for k in range(19):                                    # 1.9 s of walking: not yet
        p.step(0.1 * k, 0.0, m, c, w)
    assert w == []
    p.step(2.0, 0.0, m, c, w)
    assert len(w) == 1 and w[0][2][1] < 0.0 and abs(w[0][2][0]) < 1e-9     # sideways (seed 1: -y)
    p.step(3.0, 0.0, m, c, w)                              # no fall within 2.5 s -> retried, stronger
    p.step(4.6, 0.0, m, c, w)
    assert len(w) == 2 and abs(w[1][2][1]) > abs(w[0][2][1]) and p.log[-1]["retry"] == 1
    c.recovery.fall_count, c.mode, c.recovery.state = 1, "FALLEN", "DAMP"
    p.step(4.8, 0.0, m, c, w)
    assert p.log[0]["fell"] is True and p.i == 1
    c.recovery.successes, c.mode, c.recovery.state = 1, "STAND", "NORMAL"
    p.step(20.0, 0.0, m, c, w)                             # stood up: the second knock 1 s later
    p.step(20.5, 0.0, m, c, w)
    assert len(w) == 2
    p.step(21.05, 0.0, m, c, w)
    assert len(w) == 3 and p.log[-1]["trig"] == "after_stand"
    c.recovery.fall_count = 2
    p.step(21.3, 0.0, m, c, w)
    assert p.done and [e["fell"] for e in p.log] == [True, True]


def test_resume_after_falls_from_the_trace():
    tr = [(0.1 * k, 0.0 if k < 30 else 0.05 * (k - 30), 0.0, 0, 0, 0, 0, "PATROL", "navigate", "patrol",
           0, 0, 0, "STAND", "NORMAL") for k in range(100)]
    ev = [{"event": "fall", "t": 1.0}, {"event": "recovered", "t": 2.0}, {"event": "recovered", "t": 9.5}]
    out = MC.resumed_after_falls(tr, ev)
    assert out[0]["resume_s"] == pytest.approx(1.0 + 1.0, abs=0.11) and out[0]["mission"] == "PATROL/navigate"
    assert out[1]["resume_s"] is None                      # the trace ends 0.4 m later


def test_standup_obstacles_from_the_trace():
    """Per stand-up: the mission's confirmed obstacle cells around it and the zero-command
    time after it (do the phantoms of the fall come back from terrain_mapper's map?)."""
    tr = []
    for k in range(400):
        t = round(0.1 * k, 2)
        mode, rec = ("FALLEN", "DAMP") if 50 <= k < 100 else ("RECOVERING", "SLIDE") if 100 <= k < 150 else (
            "STAND", "NORMAL")
        conf = 40 if 40 <= k < 152 else 5 if k < 40 else 60      # forgotten at the stand-up, back 1.5 s later
        phase, v = ("hold", 0.0) if 50 <= k < 150 else ("navigate", 0.0 if 165 <= k < 285 else 0.3)
        if k >= 300:
            phase, v = "dwell", 0.0                                # (an inspection is no standstill)
        tr.append((t, 0, 0, 0, 0, 0, 0, "PATROL" if k < 300 else "INSPECTING", phase, "patrol", v, 0.0, 0.0,
                   mode, rec, conf))
    out = MC.standup_obstacles(tr)
    assert len(out) == 1 and out[0]["t"] == pytest.approx(15.0)
    c = out[0]["confirmed"]
    assert c["-0.2"] == 40 and c["+0.1"] == 40 and c["+3"] == 60 and out[0]["max_after"] == 60
    assert out[0]["zero_cmd_s"] == pytest.approx(12.0, abs=0.11)
    assert MC.standup_obstacles([r[:15] for r in tr]) == []          # old traces without the column


def test_phantom_legs_only_while_down_or_tilted():
    import threading
    from types import SimpleNamespace as NS
    lay = NS(floor_z=lambda x, y: -0.88)
    fn = MC.phantom_legs(lay)
    h = NS(_lock=threading.Lock(), _odom=None, ctrl=NS(mode="WALK", recovery=NS(state="NORMAL")))
    assert fn(0.0, h) is None
    h._odom = (0.0, np.array([10.0, 2.0, -0.54]), np.array([0.0, 0.05, 0.3]))
    assert fn(0.1, h) is None                                         # upright, walking: nothing
    h._odom = (0.2, np.array([10.0, 2.0, -0.7]), np.array([1.4, 0.1, 0.3]))   # tipped over (not yet FALLEN)
    pc = fn(0.2, h)
    from module1_locomotion.pointcloud_utils import CONVENTION_LINK, PointFilter, camera_points_to_world
    pw, _rng, _o = camera_points_to_world(pc, (10.0, 2.0, -0.7, 1.4, 0.1, 0.3), CONVENTION_LINK,
                                          filt=PointFilter(min_range=0.0, max_range=10.0, body_box_min=(0, 0, 0),
                                                           body_box_max=(0, 0, 0), max_height_above_base=10.0))
    r = np.hypot(pw[:, 0] - 10.0, pw[:, 1] - 2.0)
    assert len(pw) == len(pc) and np.all((r > 0.43) & (r < 0.75)) and np.allclose(pw[:, 2], -0.58)
    h._odom = (0.3, np.array([10.0, 2.0, -0.54]), np.array([0.0, 0.0, 0.3]))
    h.ctrl.mode, h.ctrl.recovery.state = "STAND", "STAND_UP"         # upright, still in the recovery
    assert fn(0.3, h) is not None


GZ_PYTHON = os.environ.get("GZ_PYTHON", "")


@pytest.mark.skipif(not (GZ_PYTHON and os.path.exists(GZ_PYTHON)),
                    reason="set GZ_PYTHON to a python with the gz-sim 8 bindings (gz.sim8, gz.transport13)")
def test_route_segment_in_gazebo(tmp_path):
    env = dict(os.environ)
    if os.name == "nt":       # conda layout: DLLs next to python.exe / Library/bin
        base = os.path.dirname(GZ_PYTHON)
        extra = [base, os.path.join(base, "Library", "mingw-w64", "bin"), os.path.join(base, "Library", "usr", "bin"),
                 os.path.join(base, "Library", "bin"), os.path.join(base, "Scripts")]
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        env["CONDA_PREFIX"] = base
    out = tmp_path / "mission.json"
    cmd = [GZ_PYTHON, CLI, "--seeds", "0", "--jobs", "1", "--duration", "40", "--label", "pytest",
           "--spawn=46.3,0.75,3.14159", "--goal=44.6,0.75", "--out", str(out), "--work-dir", str(tmp_path)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    block = json.loads(out.read_text(encoding="utf-8"))["runs"]["pytest"]
    r = block["runs"]["0"]
    assert r["error"] is None and r["segment"]["ok"], (r["segment"], r["fall_info"])
    assert r["falls"] == 0 and r["teleports"] == 0 and r["stop_reason"] in ("finished", "leg end (P1)")
    assert r["timeline"][0][1] == "IDLE" and any(s[1] == "INSPECTING" for s in r["timeline"])
    assert r["sim_time"] < 40.0 and r["wall_time"] > 0


@pytest.mark.skipif(not (GZ_PYTHON and os.path.exists(GZ_PYTHON)),
                    reason="set GZ_PYTHON to a python with the gz-sim 8 bindings (gz.sim8, gz.transport13)")
def test_patrol_carries_on_after_two_falls_in_a_row_in_gazebo(tmp_path):
    """Field report: 'up again after falling (twice in a row) the robot did not patrol on'.
    Knocked over while walking the upper corridor, again 1 s after it stood up: it gets
    up by itself (no teleport) and the patrol walks on to both goals.  With
    --phantom-legs the depth frames of the tipped / lying robot show a shank-high ring
    around it (what Ubuntu's rendered camera sees of its own legs): terrain_mapper's fall
    guard keeps it out of the map, so the obstacle cells forgotten at the stand-up do not
    come back (review of fix/resume: they were re-confirmed 1.5 s later)."""
    env = dict(os.environ)
    if os.name == "nt":
        base = os.path.dirname(GZ_PYTHON)
        extra = [base, os.path.join(base, "Library", "mingw-w64", "bin"), os.path.join(base, "Library", "usr", "bin"),
                 os.path.join(base, "Library", "bin"), os.path.join(base, "Scripts")]
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        env["CONDA_PREFIX"] = base
    out = tmp_path / "mission.json"
    cmd = [GZ_PYTHON, CLI, "--seeds", "0", "--jobs", "1", "--duration", "180", "--label", "pytest",
           "--spawn=46.3,0.72,3.14159", "--goal=43.0,0.8", "--goal=40.8,0.9", "--push=nav:2.5",
           "--push=after_stand:1.0", "--phantom-legs", "--out", str(out), "--work-dir", str(tmp_path)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1500)
    assert p.returncode == 0, p.stdout + p.stderr
    r = json.loads(out.read_text(encoding="utf-8"))["runs"]["pytest"]["runs"]["0"]
    assert r["error"] is None, r["error"]
    assert [e["fell"] for e in r["pushes"]] == [True, True], r["pushes"]
    assert r["teleports"] == 0 and r["recoveries"] >= 2
    assert r["segment"]["ok"], (r["segment"], r["fall_info"], r["stalls"], r["resumed"])
    assert all(x["resume_s"] is not None and x["resume_s"] < 15.0 for x in r["resumed"]), r["resumed"]
    assert r["phantom_legs"] and r["map_fall_guard"]["enabled"] and r["map_fall_guard"]["restores"] >= 2
    last = r["standups"][-1]                         # (the first one is followed by the second knock)
    assert len(r["standups"]) >= 2 and last["max_after"] is not None, r["standups"]
    assert last["max_after"] <= max(5, last["confirmed"]["+0.1"] or 0), r["standups"]
    assert last["zero_cmd_s"] < 8.0, r["standups"]
