"""Gait scheduler, horizon prediction, transitions, swing trajectory, stability."""
import math

import numpy as np
import pytest

from module1_locomotion import gait as G
from module1_locomotion import kinematics as kin

FL, FR, RL, RR = 0, 1, 2, 3


def _run(sch, t0, t1, dt=0.0025):
    out = []
    for t in np.arange(t0, t1, dt):
        out.append(sch.update(float(t)))
    return out


def test_trot_diagonal_pairs_and_duty():
    sch = G.GaitScheduler()
    sch.reset(0.0, "trot")
    states = _run(sch, 0.0, 4.0)
    C = np.array([s.contacts for s in states])
    assert np.array_equal(C[:, FL], C[:, RR]) and np.array_equal(C[:, FR], C[:, RL])
    assert np.all(C[:, FL] != C[:, FR])                   # exactly one pair on the ground
    assert np.mean(C[:, FL]) == pytest.approx(0.5, abs=0.01)
    assert states[0].swing_duration == pytest.approx(0.2)


def test_crawl_one_leg_at_a_time_in_order():
    sch = G.GaitScheduler()
    sch.reset(0.0, "crawl")
    states = _run(sch, 0.0, 3.6)
    C = np.array([s.contacts for s in states])
    assert np.all((~C).sum(axis=1) <= 1)                  # at most one foot in the air
    assert np.mean(C, axis=0) == pytest.approx([0.75] * 4, abs=0.01)
    lift_order = [int(np.nonzero(s.liftoff)[0][0]) for s in states if s.liftoff.any()]
    seq = lift_order[:8]
    assert seq == [RR, FR, RL, FL, RR, FR, RL, FL] or seq[:4] in ([FL, RR, FR, RL], [RR, FR, RL, FL])
    # the 4-beat cycle is FL -> RR -> FR -> RL
    i = lift_order.index(FL)
    assert lift_order[i:i + 4] == [FL, RR, FR, RL]


def test_prediction_matches_sequential_update_with_pending_transition():
    sch = G.GaitScheduler()
    sch.reset(0.0, "trot")
    sch.update(0.13)
    sch.request("crawl")
    times = np.arange(0.0, 5.0, 0.01)
    pred = sch.contacts_at(times)
    sim = sch.snapshot()
    seq = np.array([sim.update(0.13 + t).contacts for t in times])
    assert np.array_equal(pred, seq)
    # the switch happens at the end of the current trot cycle (phase wrap)
    names = [sch.snapshot().pattern_at(t).name for t in (0.0, 0.26, 0.28, 2.0)]
    assert names == ["trot", "trot", "crawl", "crawl"]


def test_transitions_never_cut_a_swing():
    sch = G.GaitScheduler()
    sch.reset(0.0, "stand")
    dt = 0.0025
    t = 0.0
    plan = [(0.0, "trot"), (1.03, "crawl"), (3.31, "trot"), (4.77, "stand"), (5.5, "crawl"), (7.2, "stand")]
    prev = sch.update(0.0)
    swing_max = np.zeros(4)
    k = 0
    while t < 9.0:
        if k < len(plan) and t >= plan[k][0]:
            sch.request(plan[k][1])
            k += 1
        t += dt
        st = sch.update(t)
        for leg in range(4):
            if not prev.contacts[leg]:
                swing_max[leg] = max(swing_max[leg], prev.swing_progress[leg])
            if st.touchdown[leg]:
                # a foot only touches down after (nearly) completing its swing
                assert swing_max[leg] > 1.0 - 2.0 * dt / max(prev.swing_duration, 1e-3) - 1e-6, (t, leg)
                swing_max[leg] = 0.0
        prev = st
    assert sch.name == "stand" and np.all(prev.contacts)


def test_walk_to_stand_at_safe_boundary_and_stand_to_walk_immediate():
    sch = G.GaitScheduler()
    sch.reset(0.0, "stand")
    sch.request("trot")
    st = sch.update(0.0)
    assert st.name == "trot" and st.switched
    sch.update(0.05)                         # FL/RR mid-swing
    sch.request("stand")
    delay = sch._switch_delay()
    assert delay == pytest.approx(0.15, abs=1e-9)          # end of the FL/RR swing (phase 0.5)
    st = sch.update(0.05 + delay + 1e-4)
    assert st.name == "stand" and np.all(st.contacts)
    # crawl stops at the next quarter boundary
    sch.reset(0.0, "crawl")
    sch.update(0.4)
    sch.request("stand")
    assert sch._switch_delay() == pytest.approx(0.2, abs=1e-9)


def test_contact_schedule_on_mpc_grid():
    sch = G.GaitScheduler()
    sch.reset(0.0, "trot")
    knots = np.concatenate([[0.0], np.cumsum([0.05] * 10 + [0.25] * 18)])
    assert knots[-1] == pytest.approx(5.0)
    c = sch.contacts_at(0.5 * (knots[:-1] + knots[1:]))
    assert c.shape == (28, 4)
    assert np.all(c.sum(axis=1) == 2)


def test_swing_trajectory_endpoints_apex_and_continuity():
    p0 = np.array([0.0, 0.1, 0.025])
    p1 = np.array([0.15, 0.12, 0.045])
    tr = G.SwingTrajectory(p0, p1, apex_z=0.125, duration=0.2)
    pos0, vel0 = tr.evaluate(0.0)
    pos1, vel1 = tr.evaluate(1.0)
    assert np.allclose(pos0, p0) and np.allclose(pos1, p1)
    assert np.allclose(vel0, 0.0) and np.allclose(vel1, 0.0)
    s = np.linspace(0, 1, 2001)
    P = np.array([tr.position(v) for v in s])
    assert P[:, 2].max() == pytest.approx(0.125, abs=1e-9)
    assert np.max(np.abs(np.diff(P, axis=0))) < 2e-3          # continuous
    # analytic velocity matches finite differences
    for v in (0.1, 0.3, 0.5, 0.77, 0.9):
        h = 1e-6
        fd = (tr.position(v + h) - tr.position(v - h)) / (2 * h) / tr.duration
        assert np.allclose(tr.evaluate(v)[1], fd, atol=1e-4)


def test_swing_apex_over_obstacle_and_lift_first():
    p0 = np.array([0.0, 0.0, 0.025])
    p1 = np.array([0.30, 0.0, 0.195])                # step up onto a 0.17 m stair
    assert G.swing_apex(p0, p1, 0.08) == pytest.approx(0.195 + 0.08)
    apex = G.swing_apex(p0, p1, 0.08, obstacle_z=0.25, clearance=0.04)
    assert apex == pytest.approx(0.25 + 0.025 + 0.04)
    assert G.swing_apex(p0, p1, 0.08, obstacle_z=float("nan")) == pytest.approx(0.275)
    assert G.swing_apex(p0, p1, 0.08, obstacle_z=2.0, max_lift=0.25) == pytest.approx(0.195 + 0.25)
    tr = G.SwingTrajectory(p0, p1, apex, 0.3, lift_fraction=0.2)
    # the foot is (almost) at full height before it has moved 10 % forward
    s_up = 0.3
    pos = tr.position(s_up)
    assert pos[2] == pytest.approx(apex) and pos[0] < 0.1 * 0.30
    # ... and has arrived above the landing point before descending
    pos = tr.position(0.7)
    assert pos[2] == pytest.approx(apex) and pos[0] > 0.9 * 0.30


def test_support_margin():
    feet = np.array([[0.2, 0.15], [0.2, -0.15], [-0.2, 0.15], [-0.2, -0.15]])
    assert G.support_margin(feet, [1, 1, 1, 1], [0.0, 0.0]) == pytest.approx(0.15)
    assert G.support_margin(feet, [1, 1, 1, 1], [0.0, 0.25]) == pytest.approx(-0.10)
    # triangle FR, RL, RR with CoM at the centre: on the diagonal -> 0
    assert G.support_margin(feet, [0, 1, 1, 1], [0.0, 0.0]) == pytest.approx(0.0, abs=1e-12)
    # two feet: minus the distance to the segment
    assert G.support_margin(feet, [1, 0, 0, 1], [0.0, 0.1]) < 0.0
    assert G.support_margin(feet, [0, 0, 0, 0], [0.0, 0.0]) == -math.inf


def test_crawl_sway_moves_com_into_support_triangle():
    feet = kin.nominal_feet_body()[:, :2]
    sch = G.GaitScheduler()
    sch.reset(0.0, "crawl")
    ts = np.arange(0.0, 1.2, 0.005)
    sway = sch.sway_at(ts, feet, gain=1.0, smooth=0.2)
    C = sch.contacts_at(ts)
    # FL swings first (phase 0..0.25): CoM shifted back/right, away from FL
    mid_fl = np.argmin(np.abs(ts - 0.15))
    assert sway[mid_fl, 0] < -0.02 and sway[mid_fl, 1] < -0.01
    margins = np.array([G.support_margin(feet, C[i], sway[i]) for i in range(len(ts))])
    without = np.array([G.support_margin(feet, C[i], [0.0, 0.0]) for i in range(len(ts))])
    assert np.mean(margins > 0.02) > 0.9
    assert np.mean(margins) > np.mean(without) + 0.03
    # trot / stand: no sway
    sch.reset(0.0, "trot")
    assert np.allclose(sch.sway_at(ts, feet), 0.0, atol=1e-12)


def test_crawl_duty_validation():
    with pytest.raises(ValueError):
        G.make_gaits(crawl_duty=0.7)
    g = G.make_gaits(crawl_duty=0.8)["crawl"]
    bounds = g.stand_boundaries()
    assert len(bounds) >= 4
    assert not any(np.any(g.mid_swing(b)) for b in bounds)
