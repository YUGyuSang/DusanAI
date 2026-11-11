"""Convex 5-second MPC: model, standing / walking solutions, constraints, footholds, torques."""
import math
import time

import numpy as np
import pytest

from dusan_common import robot_model as rm
from module1_locomotion import gait as G
from module1_locomotion import kinematics as kin
from module1_locomotion import mpc as M
from module1_locomotion import qp as Q

MG = rm.TOTAL_MASS * 9.81


@pytest.fixture(scope="module", autouse=True)
def _single_thread_blas():
    """The MPC matrices are small: one BLAS thread is fastest and immune to
    thread oversubscription on a loaded machine (restored afterwards)."""
    before = Q.blas_threads()
    Q.limit_blas_threads(1)
    yield
    if before:
        Q.limit_blas_threads(before)


def _inp(t=0.0, pos=(0.0, 0.0, 0.34), vel=(0.0, 0.0, 0.0), cmd=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0),
         feet=None, **kw):
    pos = np.asarray(pos, dtype=float)
    if feet is None:
        feet = kin.body_to_world(kin.nominal_feet_body(), pos, rpy)
    return M.MPCInput(t=t, pos=pos, rpy=np.asarray(rpy, float), vel=np.asarray(vel, float),
                      omega=np.zeros(3), feet=feet, cmd=cmd, ground_z=float(pos[2]) - 0.34, **kw)


def _sched(name, t=0.0):
    s = G.GaitScheduler()
    s.reset(t, name)
    return s


def test_horizon_is_five_seconds_non_uniform():
    mpc = M.ConvexMPC()
    assert mpc.N == 28
    assert np.allclose(mpc.dts[:10], 0.05) and np.allclose(mpc.dts[10:], 0.25)
    assert mpc.T == pytest.approx(5.0)
    r = mpc.solve(_inp(), _sched("stand"))
    assert r.x_pred.shape == (29, 13) and r.forces.shape == (28, 12)
    assert r.footholds.shape == (28, 4, 3) and r.torques.shape == (28, 12)
    assert r.times[-1] == pytest.approx(5.0) and r.horizon == pytest.approx(5.0)


def test_exact_discretisation():
    """A_d, B_d (closed form, M^3 = 0) equal the matrix exponential of [[A, B], [0, 0]] dt."""
    mpc = M.ConvexMPC()
    r_feet = kin.nominal_feet_body() - np.array([0.0, 0.0, 0.025])
    affine = np.array([0.1, -0.2, 0.05, 0.3, -0.1, -9.81 + 0.5])
    for yaw, dt in ((0.0, 0.05), (1.2, 0.25), (-2.5, 0.25)):
        A, B = mpc._continuous(yaw, r_feet, affine)
        Mx = np.zeros((25, 25))
        Mx[:13, :13] = A
        Mx[:13, 13:] = B
        E = np.eye(25)
        term = np.eye(25)
        for k in range(1, 12):                      # series (exact after 3 terms)
            term = term @ (Mx * dt) / k
            E = E + term
        Ad, Bd = mpc.dynamics(yaw, r_feet, dt, affine)
        assert np.allclose(Ad, E[:13, :13], atol=1e-12)
        assert np.allclose(Bd, E[:13, 13:], atol=1e-12)
    Mx3 = np.linalg.matrix_power(Mx, 3)
    assert np.allclose(Mx3, 0.0)


def test_standing_forces_support_weight_symmetrically():
    mpc = M.ConvexMPC()
    r = mpc.solve(_inp(), _sched("stand"))
    assert r.ok
    f = r.forces.reshape(28, 4, 3)
    assert np.allclose(f[:, :, 2].sum(axis=1), MG, rtol=0.01)
    assert np.allclose(f[0, :, 2], MG / 4, rtol=0.02)
    assert np.allclose(f[:, 0, 2], f[:, 1, 2], rtol=0.01)          # left / right symmetric
    assert np.allclose(f[:, :, :2], 0.0, atol=0.5)
    assert np.max(np.abs(r.x_pred[:, 5] - 0.34)) < 0.005
    assert np.max(np.abs(r.x_pred[:, :2])) < 0.01
    # predicted torques: knees extend against the load, symmetric legs
    tau = r.torques[0].reshape(4, 3)
    assert np.all(tau[:, 2] > 3.0)
    assert np.allclose(np.abs(tau[0]), np.abs(tau[3]), atol=1e-3)


def test_standing_with_shifted_feet_loads_closer_legs():
    feet = kin.body_to_world(kin.nominal_feet_body(dx=-0.06), [0, 0, 0.34], [0, 0, 0])   # CoM ahead of feet centre
    mpc = M.ConvexMPC()
    r = mpc.solve(_inp(feet=feet, hold_xy=(0.0, 0.0)), _sched("stand"))
    f = r.forces[0].reshape(4, 3)
    assert f[0, 2] + f[1, 2] > f[2, 2] + f[3, 2] + 10.0
    assert f[:, 2].sum() == pytest.approx(MG, rel=0.02)


def _check_constraints(r, mu, fmax):
    f = r.forces.reshape(-1, 4, 3)
    c = r.contacts
    tol = 1e-6
    assert np.all(f[:, :, 2] >= -tol)
    assert np.all(f[:, :, 2] <= fmax * c + tol)
    assert np.all(np.abs(f[:, :, 0]) <= mu * f[:, :, 2] + tol)
    assert np.all(np.abs(f[:, :, 1]) <= mu * f[:, :, 2] + tol)
    swing = c < 0.05
    assert np.all(f[swing] == 0.0)


def test_trot_walking_solution_is_feasible_and_moves_forward():
    mpc = M.ConvexMPC()
    sch = _sched("trot")
    sch.update(0.0)
    r = mpc.solve(_inp(cmd=(0.4, 0.0, 0.0), vel=(0.3, 0.0, 0.0)), sch)
    assert r.ok, r.status
    _check_constraints(r, mpc.cfg.mu, mpc.cfg.f_max)
    # diagonal pairs carry the weight in the fine steps
    assert np.all((r.contacts[:10] > 0).sum(axis=1) == 2)
    # the body follows the commanded velocity over 5 s, height and attitude stay level
    assert r.x_pred[-1, 3] == pytest.approx(0.4 * 5.0, abs=0.25)
    assert np.max(np.abs(r.x_pred[:, 5] - 0.34)) < 0.03
    assert np.max(np.abs(r.x_pred[:, 0:2])) < 0.05
    # mean vertical force = weight (averaged over the horizon)
    fz = r.forces.reshape(28, 4, 3)[:, :, 2].sum(axis=1)
    assert np.average(fz, weights=r.dts) == pytest.approx(MG, rel=0.05)
    assert np.all(np.isfinite(r.torques)) and r.torque_ratio < 1.0


def test_footholds_over_horizon_follow_raibert():
    mpc = M.ConvexMPC()
    sch = _sched("trot")
    sch.update(0.0)
    v = 0.4
    r = mpc.solve(_inp(cmd=(v, 0.0, 0.0), vel=(v, 0.0, 0.0)), sch)
    ev = r.foothold_events
    per_leg = [sum(1 for e in ev if e[0] == leg) for leg in range(4)]
    assert all(11 <= n <= 14 for n in per_leg)          # ~12 touchdowns per leg in 5 s
    t_st = 0.2
    for (leg, ttd, x, y, z) in ev:
        # x = hip position at mid-stance (Raibert, symmetric stance) on the reference
        hip_x = v * (ttd + 0.5 * t_st) + kin.SIGN_X[leg] * rm.NOMINAL_FOOT_X
        assert x == pytest.approx(hip_x, abs=0.02)
        assert y == pytest.approx(kin.SIGN_Y[leg] * rm.NOMINAL_FOOT_Y, abs=0.02)
        assert z == pytest.approx(rm.FOOT_RADIUS, abs=1e-9)
    # a velocity error shifts the next footholds (capture point), not the late ones
    r2 = mpc.solve(_inp(cmd=(v, 0.0, 0.0), vel=(v, 0.3, 0.0)), sch)
    first = min((e for e in r2.foothold_events if e[0] == 0), key=lambda e: e[1])
    late = max((e for e in r2.foothold_events if e[0] == 0), key=lambda e: e[1])
    # FL lands at t = 0.2 s: the hip has drifted 0.2 exp(-0.2/tau) dv by then (critically damped
    # return to the path, tau = feedback_tau = 0.5 s), plus the capture-point offset
    # (T_st/2 + k_cp sqrt(h/g)) * dv * exp(-0.2/0.5)
    drift = 0.2 * math.exp(-0.2 / 0.5)
    expect = drift * 0.3 + (0.1 + 0.6 * math.sqrt(0.34 / 9.81)) * 0.3 * math.exp(-0.2 / 0.5)
    assert first[1] == pytest.approx(0.2, abs=0.01)
    assert first[3] == pytest.approx(rm.NOMINAL_FOOT_Y + expect, abs=0.003)
    assert late[3] == pytest.approx(rm.NOMINAL_FOOT_Y, abs=0.01)


def test_turning_prediction_and_crawl():
    mpc = M.ConvexMPC()
    sch = _sched("crawl")
    sch.update(0.0)
    r = mpc.solve(_inp(cmd=(0.15, 0.0, 0.3)), sch)
    assert r.ok
    _check_constraints(r, mpc.cfg.mu, mpc.cfg.f_max)
    assert r.x_pred[-1, 2] == pytest.approx(0.3 * 5.0, abs=0.1)
    # three legs in contact in the fine steps (crawl)
    assert np.all((r.contacts[:10] >= 0.05).sum(axis=1) >= 3)


def test_terrain_reference_and_foothold_callback():
    """Stairs ahead: the height reference and the predicted footholds rise."""
    def height(x, y):
        x = np.asarray(x, dtype=float)
        return np.where(x > 1.0, 0.17, 0.0) + 0.0 * np.asarray(y, dtype=float)

    calls = []

    def foothold(leg, nominal):
        calls.append(leg)
        p = np.array(nominal, dtype=float)
        p[0] += 0.01                                       # planner nudges the foot forward
        p[2] = float(height(p[0], p[1]))
        return p, True

    mpc = M.ConvexMPC()
    sch = _sched("trot")
    sch.update(0.0)
    r = mpc.solve(_inp(cmd=(0.4, 0.0, 0.0), vel=(0.4, 0.0, 0.0)), sch, height_fn=height, foothold_fn=foothold)
    assert r.ok and calls
    assert r.x_ref[-1, 5] == pytest.approx(0.17 + 0.34, abs=0.01)
    assert r.x_pred[-1, 5] > 0.45
    ev = [e for e in r.foothold_events if e[2] > 1.1]
    assert ev and all(e[4] == pytest.approx(0.17 + rm.FOOT_RADIUS) for e in ev)
    # terrain pitch: nose up while front feet are higher than rear feet
    assert r.x_ref[:, 1].min() < -0.1


def test_lower_level_floor_and_yaw_wrap():
    mpc = M.ConvexMPC()
    sch = _sched("trot")
    sch.update(0.0)
    pos = (61.9, 0.0, -0.88 + 0.34)
    r = mpc.solve(_inp(pos=pos, rpy=(0.0, 0.0, 3.1), cmd=(0.3, 0.0, 0.2)), sch)
    assert r.ok
    assert np.max(np.abs(r.x_pred[:, 5] - pos[2])) < 0.03
    assert r.x_pred[-1, 2] == pytest.approx(3.1 + 1.0, abs=0.1)          # unwrapped heading
    hold = mpc.solve(_inp(pos=pos, rpy=(0.0, 0.0, 3.1), hold_xy=pos[:2], hold_yaw=-3.1), _sched("stand"))
    assert abs(hold.x_ref[-1, 2] - (3.1 + (2 * math.pi - 6.2))) < 1e-6    # no 2*pi jump


def test_solve_time_and_warm_start():
    mpc = M.ConvexMPC()
    sch = _sched("trot")
    times, cpu, iters = [], [], []
    for k in range(12):
        t = 0.05 * k
        sch.update(t)
        r = mpc.solve(_inp(t=t, pos=(0.4 * t, 0.0, 0.34), cmd=(0.4, 0.0, 0.0), vel=(0.4, 0.0, 0.0)), sch)
        assert r.ok
        times.append(r.solve_time)
        cpu.append(r.solve_cpu_time)
        iters.append(r.qp_iterations)
    med = float(np.median(times[1:]))
    cpu_mean = float(np.mean(cpu[1:]))          # thread CPU time: tick-quantised on Windows, mean is fine
    print(f"\nMPC solve time median {med * 1e3:.1f} ms (CPU mean {cpu_mean * 1e3:.1f} ms), "
          f"max {max(times) * 1e3:.1f} ms, ADMM iterations {iters}")
    assert np.all(np.isfinite(cpu)) and min(cpu) >= 0.0
    # typical ~10-15 ms with one BLAS thread; the CPU-time criterion keeps the test
    # meaningful on a heavily loaded machine where wall-clock time is inflated
    assert med < 0.08 or cpu_mean < 0.05
    assert np.mean(iters[1:]) <= iters[0] + 5


def test_condensed_qp_solution_satisfies_kkt():
    """The condensed Hessian is very ill-conditioned (cond ~ 1e9: internal
    squeeze forces between stance feet are only weakly penalised); the
    polished solution must still satisfy the KKT conditions almost exactly."""
    mpc = M.ConvexMPC()
    cap = []
    orig = mpc.qp.solve

    def spy(*a, **k):
        r = orig(*a, **k)
        cap.append((a, r))
        return r

    mpc.qp.solve = spy
    for name, cmd in (("stand", (0.0, 0.0, 0.0)), ("trot", (0.4, 0.0, 0.3)), ("crawl", (0.15, 0.0, 0.2))):
        sch = _sched(name)
        sch.update(0.0)
        mpc.reset()
        mpc.solve(_inp(cmd=cmd, vel=(cmd[0], 0.0, 0.0)), sch)
        (H, q, A, lo, hi), r = cap[-1][0][:5], cap[-1][1]
        H = 0.5 * (H + H.T)
        assert r.ok and r.polished, (name, r.status)
        stationarity = np.max(np.abs(H @ r.x + q + A.T @ r.y)) / max(1.0, np.max(np.abs(q)))
        assert stationarity < 1e-9, (name, stationarity)
        z = A @ r.x
        assert np.all(z >= lo - 1e-7) and np.all(z <= hi + 1e-7)


def test_accel_bias_observer_learns_payload():
    obs = M.AccelBiasObserver(tau=0.2)
    m_true = 1.2 * rm.TOTAL_MASS
    feet = kin.nominal_feet_body() + np.array([0.0, 0.0, 0.34])
    f = np.tile([0.0, 0.0, MG / 4], (4, 1))                  # forces sized for the nominal mass
    v = np.zeros(3)
    for k in range(400):
        t = k * 0.01
        obs.update(t, v, np.zeros(3), 0.0, feet, np.array([0.0, 0.0, 0.34]), f)
        a = f.sum(axis=0) / m_true + np.array([0.0, 0.0, -9.81])
        v = v + a * 0.01
    expected = MG / m_true - 9.81 - (MG / rm.TOTAL_MASS - 9.81)
    assert obs.a_bias[2] == pytest.approx(expected, abs=0.05)


def test_crawl_anchor_removes_periodic_sway_but_not_lag():
    """The crawl path anchor averages the measured position over one gait
    period: a periodic sway cancels exactly, a body that lags the command gets
    a path that starts AHEAD of it (no neutral 'slow walking' mode)."""
    mpc = M.ConvexMPC()
    sch = _sched("crawl")
    W = sch.pattern.period
    v = 0.25
    ax = []
    for k in range(80):                                       # 4 s at 20 Hz
        t = 0.05 * k
        x = v * t + 0.05 * math.sin(2 * math.pi * t / W) + 0.02 * math.sin(4 * math.pi * t / W)
        y = 0.04 * math.sin(2 * math.pi * t / W + 0.7)
        a = mpc._crawl_anchor(_inp(t=t, pos=(x, y, 0.34), vel=(v, 0.0, 0.0)), sch, v, 0.0)
        ax.append((t, a[0], a[1]))
    ax = np.array(ax)
    late = ax[:, 0] > W + 0.1
    assert np.max(np.abs(ax[late, 1] - v * ax[late, 0])) < 2e-3     # on the un-swayed path
    assert np.max(np.abs(ax[late, 2])) < 2e-3
    # a body that only makes half the commanded speed: anchor ahead by ~dv * W / 2
    mpc.reset()
    for k in range(60):
        t = 0.05 * k
        a = mpc._crawl_anchor(_inp(t=t, pos=(0.5 * v * t, 0.0, 0.34), vel=(0.5 * v, 0.0, 0.0)), sch, v, 0.0)
    assert a[0] - 0.5 * v * t == pytest.approx(0.5 * v * (W - 0.05) / 2.0, abs=0.01)


def test_capture_point_uses_swayed_reference_velocity():
    """While crawling, the body moving with the intended sway velocity is not a
    velocity error: the footholds must not be pushed along the sway."""
    mpc = M.ConvexMPC()
    sch = _sched("crawl")
    sch.update(0.0)
    r0 = mpc.solve(_inp(cmd=(0.15, 0.0, 0.0), vel=(0.15, 0.0, 0.0)), sch)
    assert r0.ok
    vs = mpc._sway_velocity_now(0.05)
    assert np.linalg.norm(vs) > 0.01                                   # the crawl sways
    sch.update(0.05)
    inp = _inp(t=0.05, pos=r0.state_at(0.05)[3:6], cmd=(0.15, 0.0, 0.0),
               vel=(0.15 + vs[0], vs[1], 0.0), feet=r0.footholds[1])
    ref = mpc._reference(inp, sch, None, True)
    xy, _hip, _c, _s, _z = mpc._raibert_batch(inp, ref, [0], [0.3], [0.9])
    ref_no = dict(ref)
    keep = mpc._sway_state
    mpc._sway_state = None                                             # error w.r.t. the path only
    xy_no, _hip, _c, _s, _z = mpc._raibert_batch(inp, ref_no, [0], [0.3], [0.9])
    mpc._sway_state = keep
    tau = mpc.cfg.feedback_tau
    gain = (0.45 + mpc.cfg.capture_gain * math.sqrt(0.34 / 9.81)) * math.exp(-0.3 / tau)
    drift = 0.3 * math.exp(-0.3 / tau)                                 # predicted hip drift until touchdown
    shift = xy_no[0] - xy[0]
    assert np.allclose(shift, drift * vs + np.clip(gain * vs, -0.16, 0.16), atol=2e-3)


def test_crawl_closed_loop_tracks_speed_without_drift():
    """Regression: MPC-predicted body motion fed back as the measurement
    (ideal tracking) must crawl at the commanded speed without sideways drift."""
    mpc = M.ConvexMPC()
    sch = _sched("crawl")
    pos = np.array([0.0, 0.0, 0.34])
    feet = kin.body_to_world(kin.nominal_feet_body(), pos, (0.0, 0.0, 0.0))
    vel = np.zeros(3)
    xs = []
    r = None
    for k in range(100):                                      # 5 s at 20 Hz
        t = 0.05 * k
        gs = sch.update(t)
        if r is not None:
            st = r.state_at(0.05)
            pos, vel = st[3:6], st[9:12]
            # stance feet keep their position, touchdowns take the predicted foothold
            for leg in range(4):
                if gs.contacts[leg] and not r.contacts[0, leg] >= 0.05:
                    feet[leg] = r.footholds[1, leg]
                elif not gs.contacts[leg]:
                    feet[leg] = r.footholds[1, leg]
        r = mpc.solve(_inp(t=t, pos=pos, vel=vel, cmd=(0.2, 0.0, 0.0), feet=feet.copy()), sch)
        assert r.ok and np.all(np.isfinite(r.x_pred))
        xs.append((t, pos[0], pos[1]))
    xs = np.array(xs)
    late = xs[:, 0] >= 2.0
    speed = np.polyfit(xs[late, 0], xs[late, 1], 1)[0]
    assert speed == pytest.approx(0.2, abs=0.04)
    assert np.max(np.abs(xs[:, 2])) < 0.12


def test_per_solve_friction_coefficient():
    """MPCInput.mu (slip estimate / friction prior) replaces cfg.mu for one solve only."""
    mpc = M.ConvexMPC()
    sch = _sched("trot")
    sch.update(0.0)
    # a lateral velocity error needs tangential forces: the pyramid limits them
    inp = _inp(cmd=(0.2, 0.0, 0.0), vel=(0.0, 0.3, 0.0), mu=0.15)
    r = mpc.solve(inp, sch)
    assert r.ok, r.status
    assert r.mu == pytest.approx(0.15)
    _check_constraints(r, 0.15, mpc.cfg.f_max)
    f = r.forces.reshape(-1, 4, 3)
    assert np.max(np.abs(f[:, :, :2]) - 0.15 * f[:, :, 2:3]) <= 1e-6
    assert mpc.cfg.mu == pytest.approx(0.5)                     # config untouched
    r2 = mpc.solve(_inp(cmd=(0.2, 0.0, 0.0), vel=(0.0, 0.3, 0.0)), sch)
    assert r2.mu == pytest.approx(0.5)
    ratio = lambda res: np.max(np.abs(res.forces.reshape(-1, 4, 3)[:, :, :2]).sum(-1) /
                               np.maximum(res.forces.reshape(-1, 4, 3)[:, :, 2], 1e-6))
    assert ratio(r2) > ratio(r)                                 # more lateral force allowed at mu 0.5
    # never above the configured coefficient, never degenerate, NaN ignored
    assert mpc.effective_mu(_inp(mu=0.9)) == pytest.approx(0.5)
    assert mpc.effective_mu(_inp(mu=0.0)) == pytest.approx(0.01)
    assert mpc.effective_mu(_inp(mu=float("nan"))) == pytest.approx(0.5)


def test_robust_terrain_reference_ignores_a_pipe_but_follows_a_ramp():
    """Terrain height / pitch reference over the body footprint (median /
    Theil-Sen, MPCConfig.terrain_robust): a 0.10 m pipe across the path is not
    a ramp (the 4-footholds reference pitched the body by ~12 deg and lifted it
    by 5 cm while the feet passed it - Gazebo: fell at the plant's C-OBS1
    pipe); a ramp still gives its slope and height."""
    def pipe(x, y):
        x = np.asarray(x, dtype=float)
        return np.where(np.abs(x - 0.6) < 0.06, 0.10, 0.0) + 0.0 * np.asarray(y, dtype=float)

    slope = math.tan(0.2)

    def ramp(x, y):
        return slope * np.asarray(x, dtype=float) + 0.0 * np.asarray(y, dtype=float)

    sch = _sched("trot")
    sch.update(0.0)
    inp = _inp(cmd=(0.3, 0.0, 0.0), vel=(0.3, 0.0, 0.0))
    robust = M.ConvexMPC()
    ref = robust._reference(inp, sch, pipe)
    near = ref["t"] < 4.0                                   # the body passes the pipe at ~2 s
    assert np.max(np.abs(ref["pitch"][near])) < 0.03
    assert np.max(np.abs(ref["ground"][near])) < 0.02
    old = M.ConvexMPC(M.MPCConfig(terrain_robust=False))
    ref_old = old._reference(inp, sch, pipe)
    assert np.max(np.abs(ref_old["pitch"][near])) > 0.15   # documents the former reference
    # ramp under the whole footprint: slope and height of the plane
    ref = robust._reference(_inp(pos=(0.5, 0.0, 0.34 + slope * 0.5)), _sched("stand"), ramp)
    assert np.allclose(ref["pitch"], -0.2, atol=0.01)
    assert ref["ground"][0] == pytest.approx(slope * 0.5, abs=0.01)
    # nothing known (no depth data under the body yet): the controller's fallbacks
    ref = robust._reference(_inp(pitch_ref=0.05), _sched("stand"), lambda x, y: np.full(np.shape(x), np.nan))
    assert np.allclose(ref["pitch"], 0.05) and np.allclose(ref["ground"], 0.0)
