"""tools/obstacle_check.py (Gazebo check of the novel-obstacle avoidance):
CLI plumbing, summary, seeds, never docs/results (no gz needed), plus a
short closed-loop run when GZ_PYTHON points to a python with the gz-sim 8
bindings."""
import json
import math
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CLI = os.path.join(ROOT, "tools", "obstacle_check.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import obstacle_check as OC  # noqa: E402
from gazebo_check import scenarios as S  # noqa: E402


def _run(sc, mode, seed, t_reach, path_len, ok=True, control=None, clr=None):
    return {"scenario": sc, "mode": mode, "seed": seed, "pass": ok, "reason": "" if ok else "fell at 3 s",
            "t_reach": t_reach, "path_len": path_len, "control": control, "min_box_clearance": clr,
            "detours": 1.0, "obstacle_replans": 0.0, "stuck_replans": 0.0, "falls": 0 if ok else 1, "missed": []}


def test_summary_compares_with_the_control_of_the_same_mode():
    runs = [_run("x_control_upper", "stable", 0, 20.0, 6.0), _run("x_control_upper", "stable", 1, 22.0, 6.2),
            _run("v_box_upper", "stable", 0, 25.2, 7.0, control="x_control_upper", clr=0.6),
            _run("v_box_upper", "stable", 1, 27.0, 7.4, control="x_control_upper", clr=0.5, ok=False),
            _run("v_box_upper", "mpc", 0, 30.0, 7.1, control="x_control_upper", clr=0.7)]
    s = OC.summarize(runs)
    v = s["v_box_upper|stable"]
    assert v["passed"] == 1 and v["runs"] == 2 and v["min_box_clearance"] == 0.5
    assert v["time_vs_control"] == pytest.approx(26.1 / 21.0, abs=1e-3)
    assert v["extra_path_m"] == pytest.approx(7.2 - 6.1, abs=1e-6)
    assert v["failures"] == ["seed 1: fell at 3 s"]
    assert s["v_box_upper|mpc"]["time_vs_control"] is None          # no mpc control run
    OC.print_summary(s)


def test_results_never_go_to_docs_results(tmp_path):
    with pytest.raises(SystemExit):
        OC.check_out_path(os.path.join(ROOT, "docs", "results", "gazebo_check.json"))
    with pytest.raises(SystemExit):
        OC.check_out_path(os.path.join(ROOT, "docs", "results", "x", "y.json"))
    assert OC.check_out_path(str(tmp_path / "r.json")) == str(tmp_path / "r.json")
    assert not os.path.normcase(os.path.abspath(OC.DEFAULT_OUT)).startswith(OC.PROTECTED)


def test_worker_command_carries_the_run():
    class A:
        params, mission_params, map, duration = '{"x": 1}', '{"obstacle_enabled": false}', "depth", 12.0
    cmd = OC.worker_cmd(A, "v_box_upper", "mpc", 2, "o.npz", "b")
    for a, b in (("--worker", "v_box_upper"), ("--mode", "mpc"), ("--seed", "2"), ("--npz", "o.npz"),
                 ("--map", "depth"), ("--duration", "12.0"), ("--mission-params", '{"obstacle_enabled": false}')):
        assert cmd[cmd.index(a) + 1] == b
    assert S.resolve_mission(list(OC.DEFAULT_SCENARIOS))


def test_seeds_change_the_spawn_and_the_tick_phase():
    sc = S.MISSION_SCENARIOS["v_box_upper"]
    assert sc.spawn_for(0) == sc.spawn
    s1, s2 = sc.spawn_for(1), sc.spawn_for(2)
    assert s1 != sc.spawn and s1 != s2 and s1 == sc.spawn_for(1)
    for s in (s1, s2):
        assert abs(s[0] - sc.spawn[0]) <= S.SPAWN_JITTER and abs(s[1] - sc.spawn[1]) <= S.SPAWN_JITTER
        assert abs(s[3] - sc.spawn[3]) <= S.SPAWN_JITTER and s[2] == sc.spawn[2]
    op = sc.operator(seed=1)
    assert op.t_start == pytest.approx(sc.t_start + S.CMD_SHIFT)
    op(0.5)                                                          # first call: the phase starts here
    assert op._next == pytest.approx(0.5 + 0.037)


def test_summary_only_reads_an_existing_file(tmp_path, capsys):
    out = tmp_path / "r.json"
    out.write_text(json.dumps({"summary": OC.summarize([_run("x_control_upper", "stable", 0, 20.0, 6.0)])}))
    assert OC.main(["--summary-only", "--out", str(out)]) == 0
    assert "x_control_upper|stable" in capsys.readouterr().out


GZ_PYTHON = os.environ.get("GZ_PYTHON", "")


@pytest.mark.skipif(not (GZ_PYTHON and os.path.exists(GZ_PYTHON)),
                    reason="set GZ_PYTHON to a python with the gz-sim 8 bindings (gz.sim8, gz.transport13)")
def test_cli_runs_a_short_mission_in_gazebo(tmp_path):
    env = dict(os.environ)
    if os.name == "nt":
        base = os.path.dirname(GZ_PYTHON)
        extra = [base, os.path.join(base, "Library", "mingw-w64", "bin"), os.path.join(base, "Library", "usr", "bin"),
                 os.path.join(base, "Library", "bin"), os.path.join(base, "Scripts")]
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        env["CONDA_PREFIX"] = base
    out = tmp_path / "r.json"
    p = subprocess.run([GZ_PYTHON, CLI, "--scenarios", "x", "--seeds", "0", "--duration", "14", "--jobs", "1",
                        "--out", str(out)], env=env, capture_output=True, text=True, timeout=1200)
    data = json.loads(out.read_text(encoding="utf-8"))
    r = data["runs"][0]
    assert r["scenario"] == "x_control_upper" and r["falls"] == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert r["detours"] == 0 and math.isfinite(r["path_len"]) and r["path_len"] > 1.0
