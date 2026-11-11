"""Tests of the MPC / gait parameter search (tools/tune_locomotion.py).

* search space: the defaults round-trip through the normalised space and map
  back onto controller.DEFAULT_PARAMS (Q list, keys, bounds);
* search machinery: Latin hypercube, CMA-ES, Pareto front, rank correlation,
  scores and the adoption rule;
* the whole multi-fidelity pipeline (successive halving -> CMA-ES -> Pareto ->
  confirmation -> decision) with a synthetic evaluator (no simulation);
* the recorded result docs/results/tuning.json agrees with the current code
  defaults (adopted parameters are in DEFAULT_PARAMS / locomotion.yaml);
* short real episodes: rigid-body push (eval_locomotion) and, with MuJoCo, a
  physics stand episode.
"""
import json
import math
import os

import numpy as np
import pytest
import yaml

import tune_locomotion as T
from module1_locomotion.controller import DEFAULT_PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


# ---------------------------------------------------------------------------
# search space
# ---------------------------------------------------------------------------
def test_defaults_round_trip_through_the_normalised_space():
    v = T.default_values()
    u = T.encode(v)
    assert np.all((u >= 0.0) & (u <= 1.0))
    assert T.decode(u) == v
    p = T.to_params(v)
    assert p["mpc_q_weights"] == [float(x) for x in DEFAULT_PARAMS["mpc_q_weights"]]
    for k, val in p.items():
        if k != "mpc_q_weights":
            assert val == pytest.approx(float(DEFAULT_PARAMS[k])), k


def test_space_maps_onto_controller_parameters():
    idx = sorted(i for ii in T.Q_INDEX.values() for i in ii)
    assert idx == list(range(12))                       # every Q entry tied to exactly one searched weight
    for p in T.SPACE:
        assert p.lo < p.hi
        if p.name not in T.Q_INDEX:
            assert p.name in DEFAULT_PARAMS, p.name
        if p.log:
            assert p.lo > 0.0
    assert len(set(T.NAMES)) == len(T.NAMES)
    # physical guards of the controller: kd_attitude <= 0.01 (loop-stable), crawl duty untouched
    assert dict((p.name, p) for p in T.SPACE)["kd_attitude"].hi <= 0.01
    assert "crawl_duty" not in T.NAMES


def test_decode_bounds_rounding_and_log_scale():
    lo = T.decode(np.zeros(len(T.SPACE)))
    hi = T.decode(np.ones(len(T.SPACE)))
    for p in T.SPACE:
        assert lo[p.name] == pytest.approx(p.lo) and hi[p.name] == pytest.approx(p.hi)
    q = dict((p.name, p) for p in T.SPACE)["q_rp"]
    assert q.decode(0.5) == pytest.approx(math.sqrt(q.lo * q.hi), rel=5e-3)     # geometric midpoint
    assert q.decode(1.7) == pytest.approx(q.hi)                                  # clipped
    assert T._round_sig(0.000212345) == 0.000212 and T._round_sig(37.449) == 37.4 and T._round_sig(0.0) == 0.0
    kd = dict((p.name, p) for p in T.SPACE)["kd_attitude"]
    assert kd.decode(1e-6) == 0.0 and kd.decode(0.5123) == pytest.approx(0.00512)      # range resolution


def test_latin_hypercube_is_stratified_and_seeded():
    u = T.latin_hypercube(20, 5, np.random.default_rng(3))
    assert u.shape == (20, 5)
    for j in range(5):
        assert sorted(np.floor(u[:, j] * 20).astype(int)) == list(range(20))
    assert np.array_equal(u, T.latin_hypercube(20, 5, np.random.default_rng(3)))


def test_candidates_are_identified_by_their_rounded_values():
    u = T.encode(T.default_values())
    a = T.make_candidate("lhs", u + 1e-6, 0)
    b = T.make_candidate("defaults", u, 0)
    assert a.cid == b.cid and a.values == b.values


# ---------------------------------------------------------------------------
# search machinery
# ---------------------------------------------------------------------------
def test_cmaes_minimises_a_shifted_sphere_inside_the_box():
    rng = np.random.default_rng(0)
    target = np.array([0.3, 0.7, 0.2, 0.5])
    es = T.CMAES(np.full(4, 0.8), 0.2, 10, rng)
    for _ in range(40):
        X = es.ask()
        assert np.all((X >= 0.0) & (X <= 1.0))
        es.tell(X, [float(np.sum((x - target) ** 2)) for x in X])
    assert np.linalg.norm(es.mean - target) < 0.03
    # non-finite fitness values are ranked last, not propagated
    X = es.ask()
    f = [math.nan] * len(X)
    f[0] = 0.0
    es.tell(X, f)
    assert np.all(np.isfinite(es.mean))


def test_pareto_front_and_rank_correlation():
    pts = [(1.0, 1.0), (0.8, 1.2), (1.2, 0.7), (1.1, 1.1), (0.8, 1.3), (math.nan, 0.1), (0.9, 0.9)]
    assert sorted(T.pareto_front(pts)) == [1, 2, 6]
    x = np.arange(10.0)
    assert T.spearman(x, x ** 3) == pytest.approx(1.0)
    assert T.spearman(x, -x) == pytest.approx(-1.0)
    assert math.isnan(T.spearman([1.0, 2.0], [3.0, 4.0]))


def _phys_ok(**kw):
    m = {"falls": 0, "complete": True, "diverged": False, "t_stand": 3.0, "height_mean": 0.34, "max_roll_deg": 1.0,
         "max_pitch_deg": 1.0, "drift_mm": 5.0, "vx_mean": 0.42, "track_rms_vx": 0.02, "track_rms_vy": 0.02,
         "stance_slip_p95": 0.1, "recovery_s": 1.5, "peak_vy": 0.3, "gait_frac": 1.0, "vy_mean": 0.0,
         "wz_mean": 0.5, "x_end": 2.5, "plate_slip_p95": 0.3}
    m.update(kw)
    return m


def test_physics_costs_rank_failures_last():
    ok = _phys_ok()
    assert T.stand_cost(ok) < 1.0 and T.stand_cost(_phys_ok(falls=1)) == T.FAIL
    assert T.stand_cost(_phys_ok(t_stand=float("nan"))) == T.FAIL
    assert T.trot_cost(_phys_ok(complete=False)) == T.FAIL
    assert T.trot_cost(_phys_ok(vx_mean=0.30)) > T.trot_cost(ok)
    worst_recovered = T.push_cost(_phys_ok(max_roll_deg=80.0, recovery_s=9.0, peak_vy=5.0))
    assert worst_recovered == pytest.approx(3.0)
    assert worst_recovered < T.push_cost(_phys_ok(falls=1)) == T.PUSH_FALL
    turn = _phys_ok(vx_mean=0.0)
    assert T.turn_cost(_phys_ok(vx_mean=0.0, wz_mean=0.4)) > T.turn_cost(turn)
    assert T.turn_cost(_phys_ok(vx_mean=0.0, falls=1)) == T.FAIL
    assert T.slip_cost(_phys_ok(x_end=1.5)) > T.slip_cost(ok) and T.slip_cost(_phys_ok(complete=False)) == T.FAIL
    f, d = T.phys_score({"stand": [ok], "trot": [ok, ok], "turn": [turn], "push": [ok, _phys_ok(falls=1)],
                         "slip": [ok], "crawl": [_phys_ok(vx_mean=0.2), _phys_ok(vx_mean=0.1)]})
    assert d["push_falls"] == 1 and d["push_runs"] == 2 and d["walk_failures"] == 0
    assert d["crawl_failures"] == 1 and d["turn_wz"] == 0.5 and d["slip_x_end"] == 2.5
    assert sum(T.W_PHYS.values()) == pytest.approx(1.0)
    assert f == pytest.approx(sum(T.W_PHYS[k] * d[k] for k in T.W_PHYS))
    # a failed slip-mode run counts as a walking failure (decide() rejects it)
    _, d = T.phys_score({"stand": [ok], "slip": [_phys_ok(falls=2)]})
    assert d["walk_failures"] == 1 and d["slip_failures"] == 1


def _rb_metrics(e=0.005, tilt=1.0, j=(24.0, 40.0), solve=12.0, fallen=False):
    tr = {"fallen": fallen, "rms_vx_ca": e, "rms_vy_ca": e, "rms_wz": e, "max_tilt_deg": tilt,
          "max_torque_ratio": 0.3}
    track = {"trot": dict(tr), "crawl": dict(tr), "solve_cpu_ms": solve}
    return track, {"mean_Ns": j[0]}, {"mean_Ns": j[1]}


def test_rigid_body_score():
    f0, d0 = T.rb_score(*_rb_metrics())
    assert math.isfinite(f0) and not d0["fallen"]
    assert T.rb_score(*_rb_metrics(e=0.01))[0] > f0
    assert T.rb_score(*_rb_metrics(j=(32.0, 48.0)))[0] < f0
    assert T.rb_score(*_rb_metrics(solve=24.0))[0] > f0
    assert T.rb_score(*_rb_metrics(fallen=True))[0] >= f0 + T.FAIL
    assert T.rb_score({"error": "x"}, {}, {})[0] == math.inf


def test_adoption_rule():
    base = {"rb": {"j_lat": 24.0, "j_fwd": 40.0, "fallen": False},
            "phys": {"push_falls": 2, "walk_failures": 0, "crawl_failures": 0}}
    good = {"F_rb": 0.85, "F_phys": 0.8, "rb": {"j_lat": 28.0, "j_fwd": 40.0, "fallen": False},
            "phys": {"push_falls": 1, "walk_failures": 0, "crawl_failures": 0}}
    assert T.decide(base, good) == (True, [])
    for change, words in (({"F_rb": 0.95}, "F_rb"), ({"F_phys": 0.91}, "F_phys"),
                          ({"phys": dict(good["phys"], push_falls=3)}, "push falls"),
                          ({"phys": dict(good["phys"], crawl_failures=1)}, "crawl"),
                          ({"phys": dict(good["phys"], walk_failures=1)}, "stand/trot/turn/slip"),
                          ({"rb": dict(good["rb"], j_fwd=36.0)}, "j_fwd")):
        ok, why = T.decide(base, {**good, **change})
        assert not ok and any(words in w for w in why), (change, why)
    assert T.decide(base, {**good, "F_rb": 0.95}, margin=0.04)[0]


# ---------------------------------------------------------------------------
# pipeline with a synthetic evaluator
# ---------------------------------------------------------------------------
def _quality(params):
    """Synthetic distance to an optimum at capture gain 0.8, q_rp 50 (defaults: 0.6, 30)."""
    return (params["mpc_capture_gain"] - 0.8) ** 2 + 0.1 * math.log(params["mpc_q_weights"][0] / 50.0) ** 2


GUARD_FAIL = {"enabled": False}


def fake_runner(t):
    cid, kind, seed, params, tag = t
    d = _quality(params)
    if kind.startswith("pc:"):
        # optionally: every candidate that is not the defaults fails the b_trot guard on seed 0
        bad = GUARD_FAIL["enabled"] and kind == "pc:b_trot" and seed == 0 and d < 0.05
        return {"key": T.task_key(t), "metrics": {"ok": not bad, "falls": int(bad)}, "wall": 0.01}
    if kind == "rb_track":
        tr = {"fallen": False, "rms_vx_ca": 0.004 + 0.05 * d, "rms_vy_ca": 0.003, "rms_wz": 0.002,
              "max_tilt_deg": 1.0 + 5 * d, "max_torque_ratio": 0.3}
        m = {"trot": tr, "crawl": dict(tr), "solve_cpu_ms": 12.0, "solve_wall_ms": 12.0, "qp_iters": 15.0}
    elif kind.startswith("rb_push"):
        j = (30.0 if kind.endswith("lat") else 50.0) - 40.0 * d
        m = {"mean_Ns": j, "min_Ns": j, "critical_Ns": [j, j]}
    elif kind == "push":
        m = _phys_ok(falls=int(d > 0.03 and seed % 3 == 0), max_roll_deg=5 + 50 * d, recovery_s=1 + 5 * d)
    elif kind == "crawl":
        m = _phys_ok(vx_mean=0.2)
    elif kind == "turn":
        m = _phys_ok(vx_mean=0.0, wz_mean=0.5 - 0.5 * d)
    elif kind == "slip":
        m = _phys_ok(x_end=2.5 - 2 * d)
    else:
        m = _phys_ok(vx_mean=0.4 + 0.3 * d, track_rms_vx=0.01 + 0.2 * d, stance_slip_p95=0.05 + d)
    return {"key": T.task_key(t), "metrics": m, "wall": 0.01}


def _small_settings(**kw):
    st = T.Settings(seed=1, n_explore=12, keep_rung1=6, keep_rung2=3, cma_gens=2, cma_pop=6, cma_sigma=0.15,
                    cma_promote=2, confirm_top=2, jobs=1, budget_min=60.0, confirm_perception="none",
                    guard_scenarios=("a_stand", "b_trot"), guard_seeds=(0, 1))
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def test_search_pipeline_with_a_synthetic_evaluator():
    msgs = []
    with T.Evaluator(1, runner=fake_runner, log=msgs.append) as ev:
        res = T.run_search(_small_settings(), ev, log=msgs.append)
    cands = {r["cid"]: r for r in res["candidates"]}
    base = res["baseline"]
    assert cands[base]["origin"] == "defaults" and cands[base]["rung"] == 2
    assert cands[base]["F_rb"] == pytest.approx(1.0) and cands[base]["J_rung2"] == pytest.approx(1.0)
    assert sum(r["origin"] == "lhs" for r in cands.values()) == 12
    assert sum(r["origin"].startswith("cma") for r in cands.values()) >= 6
    assert [h["gen"] for h in res["cma"]] == [0, 1]
    # successive halving: 6 (+ defaults) reach rung 1, 3 (+ defaults) + 2 CMA-ES children reach rung 2
    assert sum(r["rung"] >= 1 for r in cands.values() if r["origin"] != "cma0" and r["origin"] != "cma1") == 7
    assert sum(r["rung"] >= 2 for r in cands.values()) == 1 + 3 + 2
    # the confirmation starts with the defaults (F = 1) and the decision follows the adoption rule
    conf = res["confirmed"]
    assert conf[0]["cid"] == base and conf[0]["F_rb"] == pytest.approx(1.0) and conf[0]["F_phys"] == 1.0
    assert all(c["cid"] in res["pareto"] or c["cid"] in res["feasible_rung2"] for c in conf[1:])
    dec = res["decision"]
    best = next(c for c in conf if c["cid"] == dec["best_cid"])
    assert dec["adopted"] == best["adopt"]
    # the synthetic optimum is reachable: the best candidate beats the defaults clearly on both scores
    assert best["J"] < 0.9 and dec["adopted"]
    assert set(dec["changes"]) <= set(T.NAMES) and "mpc_capture_gain" in dec["changes"]
    assert dec["params"]["mpc_q_weights"][0] == dec["params"]["mpc_q_weights"][1]
    # the regression guard ran for the defaults and the adopted candidate
    g = dec["guard"]
    assert g["scenarios"] == ["a_stand", "b_trot"] and set(g["passes"]) == {base, best["cid"]}
    assert g["passes"][best["cid"]] == {"a_stand": 2, "b_trot": 2}
    assert set(res["sensitivity"]) == set(T.NAMES)
    assert res["evaluator"]["errors"] == 0


def test_guard_regression_blocks_the_adoption():
    GUARD_FAIL["enabled"] = True
    try:
        with T.Evaluator(1, runner=fake_runner, log=lambda *a: None) as ev:
            res = T.run_search(_small_settings(), ev, log=lambda *a: None)
    finally:
        GUARD_FAIL["enabled"] = False
    dec = res["decision"]
    assert not dec["adopted"]
    tried = [c for c in res["confirmed"][1:] if c["cid"] in dec["guard"]["passes"]]
    assert tried and all(not c["adopt"] for c in tried)
    assert all(any("guard: b_trot 1/2 < defaults 2/2" in w for w in c["reasons"]) for c in tried)
    assert dec["guard"]["passes"][res["baseline"]] == {"a_stand": 2, "b_trot": 2}


def test_budget_skips_the_cma_es_generations():
    msgs = []
    with T.Evaluator(1, runner=fake_runner, log=msgs.append) as ev:
        res = T.run_search(_small_settings(budget_min=0.0), ev, log=msgs.append)
    assert res["cma"] == [] and any("budget" in m for m in msgs)
    assert res["confirmed"][0]["cid"] == res["baseline"]


def test_figures_from_a_synthetic_search(tmp_path):
    with T.Evaluator(1, runner=fake_runner, log=lambda *a: None) as ev:
        res = T.run_search(_small_settings(cma_gens=1), ev, log=lambda *a: None)
    res["defaults"] = T.default_values()
    files = T.make_figures(res, str(tmp_path))
    assert [os.path.basename(f) for f in files] == ["tuning_pareto.png", "tuning_sensitivity.png",
                                                    "tuning_progress.png"]
    assert all(os.path.getsize(f) > 10000 for f in files)


# ---------------------------------------------------------------------------
# recorded result vs code defaults
# ---------------------------------------------------------------------------
RESULT = os.path.join(ROOT, "docs", "results", "tuning.json")


@pytest.mark.skipif(not os.path.isfile(RESULT), reason="docs/results/tuning.json not generated yet")
def test_recorded_tuning_result_matches_the_code_defaults():
    with open(RESULT, encoding="utf-8") as f:
        res = json.load(f)
    dec = res["decision"]
    expected = dict(res["defaults"])
    if dec["adopted"]:
        expected.update({k: new for k, (_old, new) in dec["changes"].items()})
    # defaults changed after the tuning run (e.g. kp_attitude 0.5 -> 0.0 by the Gazebo
    # MPC work) are documented in the record instead of rewriting its baseline: each
    # change must start from the value the record ends with
    later = res.get("later_changes", {}).get("changes", {})
    if later:   # an override must say when and why (the evidence), not only the new value
        assert res["later_changes"].get("date") and len(res["later_changes"].get("source", "")) > 40
    for k, (old, new) in later.items():
        assert k in expected, k
        assert expected[k] == pytest.approx(old), (k, expected[k], old)
        expected[k] = new
    assert T.default_values() == pytest.approx(expected)
    # ... and the same values are in config/locomotion.yaml
    with open(os.path.join(ROOT, "module1_locomotion", "config", "locomotion.yaml"), encoding="utf-8") as f:
        y = yaml.safe_load(f)["locomotion_controller"]["ros__parameters"]
    assert T.default_values(y) == pytest.approx(expected)
    assert [s["name"] for s in res["space"]] == list(T.NAMES)
    assert res["confirmed"][0]["cid"] == res["baseline"]
    for fig in res["figures"]:
        assert os.path.isfile(os.path.join(ROOT, fig)), fig


# ---------------------------------------------------------------------------
# real episodes (short)
# ---------------------------------------------------------------------------
@pytest.fixture()
def _single_thread_blas():
    from module1_locomotion import qp as Q
    before = Q.blas_threads()
    Q.limit_blas_threads(1)
    yield
    if before:
        Q.limit_blas_threads(before)


def test_rigid_body_push_episode(_single_thread_blas):
    """eval_locomotion closed loop: a 5 N s push is absorbed, 90 N s is a fall."""
    assert not T.rb_push_falls({}, "lat", 5.0, 2.5)
    assert T.rb_push_falls({}, "lat", 90.0, 2.5)


def test_physics_stand_episode(_single_thread_blas):
    pytest.importorskip("mujoco")
    m = T.phys_episode("stand", {}, 0, "none")
    assert m["complete"] and m["falls"] == 0 and m["t_stand"] < 3.5
    assert abs(m["height_mean"] - 0.34) < 0.03 and T.stand_cost(m) < 1.0
