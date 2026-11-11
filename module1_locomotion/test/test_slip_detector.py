"""Unit tests: online slip detection and friction estimation (slip_detector.py)."""
import math

import numpy as np
import pytest

from module1_locomotion.slip_detector import SlipDetector, SlipParams

DT = 0.01
FEET = np.array([[0.22, 0.13, 0.025], [0.22, -0.13, 0.025], [-0.22, 0.13, 0.025], [-0.22, -0.13, 0.025]])
ALL = np.ones(4, dtype=bool)


def run(det, t0, dur, feet_fn, stance=ALL, forces=None, body_fn=None):
    """Feed the detector for ``dur`` seconds; returns the list of states."""
    out = []
    for k in range(int(round(dur / DT))):
        t = t0 + k * DT
        out.append(det.update(t, feet_fn(t), stance, forces, body_xy=None if body_fn is None else body_fn(t)))
    return out


def sliding(legs, speed, t_start=0.0, direction=(1.0, 0.0)):
    """Feet ``legs`` slide at ``speed`` [m/s] from ``t_start`` on."""
    d = np.asarray(direction, dtype=float)

    def f(t):
        p = FEET.copy()
        for i in legs:
            p[i, :2] += d * speed * max(0.0, t - t_start)
        return p
    return f


def test_fixed_stance_feet_never_slip():
    det = SlipDetector()
    noise = np.random.default_rng(0)
    states = run(det, 0.0, 3.0, lambda t: FEET + noise.normal(0.0, 0.0005, FEET.shape))
    assert not any(s.slip for s in states) and det.mu_est == pytest.approx(0.5)
    assert max(s.speed for s in states) < 0.05


def test_sustained_sliding_raises_the_flag_once():
    det = SlipDetector()
    states = run(det, 0.0, 1.0, sliding([0, 3], 0.2))
    first = next(i for i, s in enumerate(states) if s.slip)
    # settle 0.03 s + 0.7 window (0.042 s) before a speed exists, then 0.1 s of evidence
    assert 0.15 <= first * DT <= 0.25
    assert sum(s.detected for s in states) == 1 and det.detections == 1
    s = states[-1]
    assert s.legs == [True, False, False, True] and s.speeds[0] == pytest.approx(0.2, rel=0.05)
    # no force estimates: the coefficient drops by mu_drop at the detection
    assert det.mu_est == pytest.approx(0.5 * 0.6)


def test_one_leg_short_jolt_and_touchdown_settle_do_not_trigger():
    det = SlipDetector()
    assert not any(s.slip for s in run(det, 0.0, 1.0, sliding([2], 0.3)))           # min_legs = 2
    det = SlipDetector()
    # 60 ms of sliding of two legs (shorter than the evidence needs), then still
    jolt = lambda t: sliding([0, 1], 0.3)(min(t, 0.11))
    assert not any(s.slip for s in run(det, 0.0, 1.0, jolt))
    # a foot sliding only during the settle time after its touch-down is ignored
    det = SlipDetector(SlipParams(settle_time=0.2))
    stance = np.ones(4, dtype=bool)
    states = run(det, 0.0, 0.18, sliding([0, 1], 0.5), stance=stance)
    assert not any(s.slip for s in states) and all(s.speed == 0.0 for s in states)


def test_force_ratio_gate_and_friction_estimate():
    # sliding while the tangential / normal force ratio is high (push on good ground): no vote
    f_push = np.tile([30.0, 0.0, 40.0], (4, 1))              # ratio 0.75
    det = SlipDetector()
    assert not any(s.slip for s in run(det, 0.0, 1.0, sliding([0, 3], 0.3), forces=f_push))
    # sliding under a small ratio = low friction: detected, mu estimate = observed ratio
    f_ice = np.tile([8.0, 6.0, 50.0], (4, 1))                # ratio 0.2
    det = SlipDetector()
    states = run(det, 0.0, 1.0, sliding([0, 3], 0.3), forces=f_ice)
    assert any(s.slip for s in states)
    assert det.mu_est == pytest.approx(0.2) and states[-1].mu_obs == pytest.approx(0.2)
    # the estimate never goes below mu_min and legs with too little normal force give no ratio
    det = SlipDetector()
    run(det, 0.0, 1.0, sliding([0, 3], 0.3), forces=np.tile([1.0, 0.0, 50.0], (4, 1)))
    assert det.mu_est == pytest.approx(0.15)
    assert SlipDetector.force_ratios(np.tile([5.0, 0.0, 10.0], (4, 1)), [0, 1], 20.0) == []
    assert SlipDetector.force_ratios(None, [0], 20.0) == []


def test_release_needs_time_and_distance_then_mu_recovers():
    p = SlipParams(release_time=2.0, release_distance=3.0, recover_tau=2.0)
    det = SlipDetector(p)
    states = run(det, 0.0, 0.5, sliding([0, 1], 0.3), body_fn=lambda t: (0.3 * t, 0.0))
    assert states[-1].slip
    hold = FEET.copy()
    # feet stop sliding, robot stands still: the flag stays (no distance travelled)
    states = run(det, 0.5, 4.0, lambda t: hold, body_fn=lambda t: (0.15, 0.0))
    assert all(s.slip for s in states)
    # walking on without slipping: released after 3 m (10 s at 0.3 m/s)
    states = run(det, 4.5, 12.0, lambda t: hold, body_fn=lambda t: (0.15 + 0.3 * (t - 4.5), 0.0))
    cleared = [k for k, s in enumerate(states) if s.cleared]
    assert len(cleared) == 1 and 9.5 <= cleared[0] * DT <= 10.5
    assert states[-1].slip is False
    mu_low = det.mu_est
    assert mu_low < 0.5
    run(det, 16.5, 10.0, lambda t: hold, body_fn=lambda t: (4.0, 0.0))
    assert mu_low < det.mu_est <= 0.5 and det.mu_est == pytest.approx(0.5, abs=0.01)
    # without body positions the release is time-based only
    det = SlipDetector(p)
    run(det, 0.0, 0.5, sliding([0, 1], 0.3))
    states = run(det, 0.5, 3.0, lambda t: hold)
    assert states[-1].slip is False and any(s.cleared for s in states)


def test_swing_legs_restart_their_history_and_clock_reset():
    det = SlipDetector()
    stance = np.array([True, False, False, True])
    # the swing legs (1, 2) move a lot - they are not watched
    swing = lambda t: FEET + np.array([[0, 0, 0], [5 * t, 0, 0.05], [5 * t, 0, 0.05], [0, 0, 0]])
    assert not any(s.slip for s in run(det, 0.0, 1.0, swing, stance=stance))
    run(det, 1.0, 1.0, sliding([0, 3], 0.3, t_start=1.0))
    assert det.slip
    det.update(0.1, FEET, ALL)                              # clock jumped back: fresh detector
    assert det.slip is False and det.mu_est == pytest.approx(0.5) and det.evidence == 0.0
    # non-finite feet are skipped
    s = det.update(0.2, np.full((4, 3), np.nan), ALL)
    assert s.speed == 0.0 and not s.slip


def test_params_from_controller_parameters():
    from module1_locomotion.controller import DEFAULT_PARAMS
    p = SlipParams.from_controller_params(DEFAULT_PARAMS)
    assert p.mu_nominal == pytest.approx(DEFAULT_PARAMS["mpc_mu"])
    assert p.speed_threshold == pytest.approx(DEFAULT_PARAMS["slip_speed_threshold"])
    assert p.min_legs == DEFAULT_PARAMS["slip_min_legs"] and isinstance(p.min_legs, int)
    assert p.release_distance == pytest.approx(DEFAULT_PARAMS["slip_release_distance"])
    assert p.max_force_ratio == pytest.approx(DEFAULT_PARAMS["slip_max_force_ratio"])
    # every SlipParams field is a controller parameter (slip_<name>), mu_nominal = mpc_mu
    from dataclasses import fields
    for f in fields(SlipParams):
        assert f.name == "mu_nominal" or ("slip_" + f.name) in DEFAULT_PARAMS, f.name
        if f.name != "mu_nominal":
            assert getattr(SlipParams(), f.name) == pytest.approx(DEFAULT_PARAMS["slip_" + f.name]), f.name
    assert math.isinf(SlipDetector().dist_since_slip)
