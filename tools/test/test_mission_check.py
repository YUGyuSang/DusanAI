"""Tests of the headless Gazebo whole-mission check (tools/mission_check.py).

* always (pure python): where a position lies on the route (area, nearest
  debris), the fall / stall analysis of the per-step logs, the completion
  verdict, the sub-plans of ``--leg`` / ``--legs`` and the route segments;
* with the gz python bindings (``GZ_PYTHON`` = path of a python with
  gz.sim8 + gz.transport13, e.g. ``C:\\gzmamba\\envs\\gz\\python.exe``; skipped
  otherwise): a SHORT route segment in the plant world (spawn next to the
  dock, one goal 1.5 m away) with the whole mission stack in the loop.
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
