"""Source seeker: gradient estimation, state machine, closed-loop seeking."""
import math

import numpy as np
import pytest

from dusan_common.grid_map import OccupancyGridMap

from module3_gas_safety.sim_harness import SensorSuite, clone_model, run_seek_trial
from module3_gas_safety.source_seeking import (CAST, FAILED, FOUND, GRADIENT, IDLE, SEARCH, SURGE,
                                               SeekerConfig, SourceSeeker)


def _open_grid(w=200, h=200, res=0.1, x0=-10.0, y0=-10.0):
    occ = np.zeros((h, w), dtype=np.int8)
    occ[0, :] = occ[-1, :] = 100
    occ[:, 0] = occ[:, -1] = 100
    return OccupancyGridMap(occ, res, x0, y0)


def _array_readings(field, x, y, yaw, cfg):
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for ox, oy in cfg.array_offsets:
        out.append(field(x + c * ox - s * oy, y + s * ox + c * oy))
    return out


def test_array_gradient_recovers_exponential_field():
    """ln c is linear -> the 4-sensor LS plane fit returns the exact log gradient."""
    g_true = np.array([0.4, -0.25])
    field = lambda x, y: 50.0 * math.exp(g_true[0] * x + g_true[1] * y) - 1.0   # noqa: E731
    cfg = SeekerConfig(log_offset_ppm=1.0)
    sk = SourceSeeker(cfg)
    for yaw in (0.0, 1.0, -2.3):
        g, cov = sk._array_gradient(_array_readings(field, 0.5, 0.2, yaw, cfg), yaw)
        assert np.allclose(g, g_true, atol=1e-9)
        assert cov.shape == (2, 2) and np.all(np.linalg.eigvalsh(cov) > 0)


def test_window_fit_separates_drift_from_gradient():
    """Walking along x while the plume intensity grows in time must not be
    read as a gradient along x.  With the seeker's step-and-dwell motion the
    spatio-temporal plane fit separates drift from gradient; on a perfectly
    collinear walk the along-track gradient is reported as unreliable."""
    cfg = SeekerConfig(window_min_samples=5, window_radius=10.0, window_time=40.0, window_sigma=5.0,
                       window_tau=40.0)
    gy_true, drift = 0.3, 0.04

    def lv(t, x, y):
        return 2.0 + gy_true * y + drift * t          # no x gradient

    # step (2 s at 0.3 m/s) - dwell (1.5 s) motion with a lateral zig-zag
    sk = SourceSeeker(cfg)
    t, x, y = 0.0, 0.0, 0.0
    for cycle in range(8):
        for _ in range(20):
            t += 0.1
            x += 0.03
            y += 0.02 * (1 if cycle % 2 == 0 else -1)
            sk._win.append((t, x, y, lv(t, x, y)))
        for _ in range(15):
            t += 0.1
            sk._win.append((t, x, y, lv(t, x, y)))
    g, cov, n = sk._window_gradient(t, np.array([x, y]))
    assert n > 200
    assert abs(g[0]) < 0.03
    assert g[1] == pytest.approx(gy_true, abs=0.03)
    assert sk._drift == pytest.approx(drift, abs=0.005)

    # collinear walk (x = 0.5 t): unidentifiable along track -> large x variance
    sk2 = SourceSeeker(cfg)
    for k in range(120):
        t = 0.1 * k
        xx, yy = 0.05 * k, 0.3 * math.sin(k * 0.3)
        sk2._win.append((t, xx, yy, lv(t, xx, yy) + 0.01 * math.sin(7.0 * k)))
    g2, cov2, _ = sk2._window_gradient(t, np.array([0.05 * 119, 0.0]))
    assert cov2[0, 0] > 10.0 * cov2[1, 1]


def test_fusion_is_information_weighted():
    sk = SourceSeeker(SeekerConfig())
    ga, ca = np.array([1.0, 0.0]), np.eye(2) * 0.01
    gw, cw = np.array([0.0, 1.0]), np.eye(2) * 1.0
    est = sk._fuse(ga, ca, gw, cw)
    assert est.g[0] > 0.9 and est.g[1] < 0.1          # precise source dominates
    assert est.snr > 5.0


def test_states_search_surge_cast_and_timeout():
    cfg = SeekerConfig(lost_time=1.0, timeout=20.0)
    sk = SourceSeeker(cfg, _open_grid())
    assert sk.state == IDLE and sk.update(0.0, 0, 0, 0, 0.0) == (0.0, 0.0, 0.0)
    sk.enable(0.0)
    assert sk.state == SEARCH
    x = y = 0.0
    moved = False
    for k in range(1, 30):
        vx, vy, wz = sk.update(0.1 * k, x, y, 0.0, 0.0, [0.0] * 4, (0.4, 0.0))
        moved |= abs(vx) + abs(vy) + abs(wz) > 1e-3
    assert sk.state == SEARCH and moved
    for k in range(30, 40):   # plume appears
        sk.update(0.1 * k, x, y, 0.0, 10.0, [10.0] * 4, (0.4, 0.0))
    assert sk.state in (SURGE, GRADIENT)
    for k in range(40, 80):   # plume lost
        sk.update(0.1 * k, x, y, 0.0, 0.0, [0.0] * 4, (0.4, 0.0))
    assert sk.state == CAST
    for k in range(80, 260):
        sk.update(0.1 * k, x, y, 0.0, 0.0, [0.0] * 4, (0.4, 0.0))
    assert sk.state == FAILED
    assert sk.update(30.0, x, y, 0.0, 0.0) == (0.0, 0.0, 0.0)
    assert [s for _, s, _ in sk.transitions][:2] == [SEARCH, SURGE]


def test_free_length_respects_obstacles():
    g = _open_grid()
    g.occ[100:110, 120:125] = 100           # block at x = 2.0..2.5, y = 0..1
    g._dist = None
    g._inflated_cache = {}
    sk = SourceSeeker(SeekerConfig(robot_radius=0.35), g)
    p = np.array([0.0, 0.5])
    fl = sk.free_length(p, np.array([1.0, 0.0]), 5.0)
    assert 1.4 < fl <= 1.75                  # stops ~0.3-0.35 m before the block face at x=2.0
    assert sk.free_length(p, np.array([0.0, 1.0]), 3.0) == pytest.approx(3.0)
    d, fl2 = sk._free_direction(p, np.array([1.0, 0.0]), 4.0)   # needs >= 2 m: straight gives 1.7
    assert fl2 >= 2.0 and abs(d[1]) > 0.1   # rotated around the block


def test_estimate_weighted_centroid_of_top_bins():
    sk = SourceSeeker(SeekerConfig(top_k=3, estimate_bin=0.25))
    sk._samples = [(0.0, 0.0, 1.0), (1.0, 0.0, 100.0), (1.1, 0.1, 90.0), (5.0, 5.0, 2.0), (1.0, 1.0, 95.0)]
    est = sk.estimate()
    assert est is not None
    assert 0.9 < est[0] < 1.15 and -0.05 < est[1] < 0.8
    assert SourceSeeker(SeekerConfig()).estimate() is None


def test_seeker_finds_the_leak_fixed_seed(plant_model, layout_grid):
    """Closed loop on the plant: gusty plume, lagged noisy sensors, holonomic robot."""
    m = clone_model(plant_model, seed=4242)
    res = run_seek_trial(m, layout_grid, (8.0, 0.5, math.pi), SeekerConfig(), SensorSuite(seed=5),
                         warmup=20.0)
    assert res.found, f"state {res.state}, transitions {res.transitions}"
    assert res.error_m is not None and res.error_m < 1.0
    assert res.contacts == 0
    assert res.time_s < 300.0


def test_seeker_from_outside_the_plume(plant_model, layout_grid):
    """Start south of the turbine (no gas): SEARCH must find the plume first."""
    m = clone_model(plant_model, seed=77)
    res = run_seek_trial(m, layout_grid, (6.0, -8.0, 0.0), SeekerConfig(), SensorSuite(seed=9),
                         warmup=20.0)
    states = [s for _, s, _ in res.transitions]
    assert states[0] == SEARCH and SURGE in states
    assert res.found and res.error_m < 1.5


# ------------------------------------------------------------- obstacle detours
def _wall_with_gap_grid():
    """20 m x 12 m room, interior wall at x = 0 with a 1.6 m gap at y = 3.0..4.6."""
    g = _open_grid(w=200, h=120, res=0.1, x0=-10.0, y0=-6.0)
    g.occ[:, 99:101] = 100                      # wall x = -0.1 .. 0.1
    g.occ[90:106, 99:101] = 0                   # gap  y = 3.0 .. 4.6
    g._dist = None
    g._inflated_cache = {}
    return g


def test_detour_routes_through_the_gap_of_a_wall():
    from module3_gas_safety.sim_harness import RobotSim
    g = _wall_with_gap_grid()
    cfg = SeekerConfig(robot_radius=0.35)
    sk = SourceSeeker(cfg, g)
    p = np.array([1.5, -2.0])
    d = np.array([-1.0, 0.0])                   # desired step straight into the wall
    assert sk.free_length(p, d, 2.0) < 1.5
    goal = sk._beyond(p, d, 2.0)
    assert goal is not None and goal[0] < -0.4 and abs(goal[1] + 2.0) < 1e-9
    route = sk._detour(0.0, p, d, 2.0)
    assert route is not None and sk.detours == 1
    assert np.hypot(*(route[-1] - goal)) < 0.15
    # the route passes the wall through the gap, never through the wall
    pts = np.vstack([p, route])
    for a, b in zip(pts[:-1], pts[1:]):
        assert g.line_of_sight(a[0], a[1], b[0], b[1], radius=0.3)
    crossings = []
    for a, b in zip(pts[:-1], pts[1:]):
        if (a[0] > 0.0) != (b[0] > 0.0):              # segment crosses the wall line x = 0
            crossings.append(a[1] + (b[1] - a[1]) * (0.0 - a[0]) / (b[0] - a[0]))
    assert len(crossings) == 1 and 3.0 + 0.2 <= crossings[0] <= 4.6 - 0.2
    # closed loop: the tracker follows the waypoints to the goal without contact
    robot = RobotSim(p[0], p[1], math.pi, max_speed=cfg.max_speed)
    sk.state = GRADIENT
    sk._go(0.0, route[-1], math.pi, route=route)
    t = 0.0
    while t < 60.0 and not sk._arrived(t, np.array([robot.x, robot.y])):
        cmd = sk._track(t, np.array([robot.x, robot.y]), robot.yaw)
        robot.step(cmd, 0.1, g)
        sk._yaw = robot.yaw
        t += 0.1
    assert np.hypot(robot.x - goal[0], robot.y - goal[1]) < 0.3
    assert robot.contacts == 0 and t < 60.0


def test_detour_failure_is_cached():
    g = _open_grid(w=200, h=120, res=0.1, x0=-10.0, y0=-6.0)
    g.occ[:, 99:101] = 100                      # wall without a gap
    g._dist = None
    g._inflated_cache = {}
    sk = SourceSeeker(SeekerConfig(), g)
    p = np.array([1.5, 0.0])
    assert sk._detour(0.0, p, np.array([-1.0, 0.0]), 1.0) is None
    assert sk._detour_fail is not None
    t0 = sk._detour_fail[0]
    assert sk._detour(1.0, p + 0.2, np.array([-1.0, 0.0]), 1.0) is None
    assert sk._detour_fail[0] == t0             # no new A* attempt inside the retry window


def test_seeker_real_map_from_the_corridor_mouth(layout):
    """Regression on the generated plant map: the bay east of the interior
    wall (x ~ 9.9 m) is connected to the main hall by a single gap; the
    seeker must route through it instead of sliding along the wall."""
    import os
    from module3_gas_safety.gas_model import GasConfig, GasDispersion2D
    from module3_gas_safety.layout_map import load_nav_map
    if not os.path.exists(layout.nav_map_yaml()):
        pytest.skip("plant_map.yaml not generated yet")
    g, _ = load_nav_map("", layout)
    m = GasDispersion2D.from_nav_map(g, GasConfig())
    m.spin_up(400.0, 0.5)
    m = clone_model(m, seed=10_017 + 2026)
    res = run_seek_trial(m, g, (17.7, 0.27, 1.59), SeekerConfig(), SensorSuite(seed=2027), warmup=30.0)
    assert res.found and res.error_m < 1.0, f"{res.state} {res.error_m} {res.transitions[-3:]}"
    assert res.contacts == 0


def test_status_fields_and_elapsed_frozen_after_the_run():
    sk = SourceSeeker(SeekerConfig(timeout=2.0), _open_grid())
    assert sk.status(5.0)["state"] == IDLE and sk.status(5.0)["elapsed"] == 0.0
    sk.enable(10.0)
    for k in range(1, 40):
        sk.update(10.0 + 0.1 * k, 0.0, 0.0, 0.0, 0.0, [0.0] * 4, (0.4, 0.0))
    assert sk.state == FAILED
    st = sk.status(100.0)
    for key in ("state", "concentration", "gradient", "estimate", "error_m", "path_length", "elapsed",
                "detours"):
        assert key in st
    assert st["elapsed"] == pytest.approx(2.1, abs=0.11)      # frozen at the time-out
    assert st["estimate"] is None and st["error_m"] is None


# ----------------------------------------------------------------------------
# pause / resume (mission pause or teleop override toggles /gas/seek/enable)
# ----------------------------------------------------------------------------
def _run_into_plume(sk, t0=0.0):
    """SEARCH for 3 s without gas, then a plume -> SURGE / GRADIENT. Returns the last time."""
    sk.enable(t0)
    t = t0
    for k in range(1, 30):
        t = t0 + 0.1 * k
        sk.update(t, 0.0, 0.0, 0.0, 0.0, [0.0] * 4, (0.4, 0.0))
    for k in range(30, 45):
        t = t0 + 0.1 * k
        sk.update(t, 0.0, 0.0, 0.0, 10.0, [10.0] * 4, (0.4, 0.0))
    return t


def test_disable_enable_within_resume_window_continues_the_run():
    sk = SourceSeeker(SeekerConfig(timeout=30.0, resume_window_s=20.0), _open_grid())
    t = _run_into_plume(sk)
    st0 = sk.state
    assert st0 in (SURGE, GRADIENT)
    n_samples, n_visited, path0 = len(sk._samples), len(sk._visited), list(sk.path)
    elapsed0 = sk.status(t)["elapsed"]
    sk.disable(t)
    assert sk.state == IDLE and not sk.active
    assert sk.update(t + 5.0, 3.0, 3.0, 0.0, 10.0) == (0.0, 0.0, 0.0)   # nothing happens while paused
    assert sk.enable(t + 15.0) is True and sk.resumed
    assert sk.state == st0 and sk.transitions[-1][2] == "resumed"
    assert len(sk._samples) == n_samples and len(sk._visited) == n_visited and sk.path == path0
    assert sk.status(t + 15.0)["elapsed"] == pytest.approx(elapsed0, abs=1e-6)   # the pause does not count
    # the run goes on from the present pose and the time-out still refers to the active time:
    # 4.4 s before the pause + 20 s after it < 30 s
    moved = False
    for k in range(1, 200):
        vx, vy, wz = sk.update(t + 15.0 + 0.1 * k, 1.0, 0.5, 0.3, 10.0, [10.0] * 4, (0.4, 0.0))
        moved |= abs(vx) + abs(vy) + abs(wz) > 1e-3
    assert moved and sk.state != FAILED
    for k in range(200, 300):
        sk.update(t + 15.0 + 0.1 * k, 1.0, 0.5, 0.3, 10.0, [10.0] * 4, (0.4, 0.0))
    assert sk.state == FAILED and sk.status(t + 100.0)["elapsed"] == pytest.approx(30.0, abs=0.15)


def test_enable_starts_a_new_run_when_the_pause_was_long_or_the_run_ended():
    cfg = SeekerConfig(timeout=3.0, resume_window_s=10.0)
    # pause longer than the window
    sk = SourceSeeker(SeekerConfig(timeout=100.0, resume_window_s=10.0), _open_grid())
    t = _run_into_plume(sk)
    sk.disable(t)
    assert sk.enable(t + 10.5) is False and sk.state == SEARCH and sk._samples == []
    # disable() without a time (old callers) never resumes
    t = _run_into_plume(sk, t + 20.0)
    sk.disable()
    assert sk.enable(t + 1.0) is False and sk.state == SEARCH
    # clock went backwards (simulation reset) -> new run
    t = _run_into_plume(sk, t + 5.0)
    sk.disable(t)
    assert sk.enable(t - 50.0) is False and sk.state == SEARCH
    # resume_window_s = 0 -> literal restart
    sk0 = SourceSeeker(SeekerConfig(timeout=100.0, resume_window_s=0.0), _open_grid())
    t = _run_into_plume(sk0)
    sk0.disable(t)
    assert sk0.enable(t + 0.5) is False and sk0.state == SEARCH
    # a finished run (FAILED) is never continued
    sk2 = SourceSeeker(cfg, _open_grid())
    sk2.enable(0.0)
    for k in range(1, 50):
        sk2.update(0.1 * k, 0.0, 0.0, 0.0, 0.0, [0.0] * 4, (0.4, 0.0))
    assert sk2.state == FAILED
    sk2.disable(5.0)
    assert sk2.enable(6.0) is False and sk2.state == SEARCH


def test_resume_in_gradient_restarts_the_reference_dwell():
    sk = SourceSeeker(SeekerConfig(timeout=100.0), _open_grid())
    sk.enable(0.0)
    sk.update(0.1, 0.0, 0.0, 0.0, 30.0, [32.0, 31.0, 28.0, 27.0], (0.4, 0.0))
    sk._enter_gradient(0.2, np.array([0.0, 0.0]), 0.0)
    sk.phase = "step"
    sk._l_ref = 1.0
    sk._target = np.array([0.5, 0.0])
    sk.disable(0.3)
    assert sk.enable(5.0)
    assert sk.state == GRADIENT and sk.phase == "dwell_ref" and sk._l_ref is None
    assert sk._target is None and sk._dwell_until == pytest.approx(5.0 + sk.cfg.dwell_time)
    assert len(sk._poses) == 0                      # no lag interpolation across the pause
