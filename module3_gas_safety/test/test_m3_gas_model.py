"""Gas dispersion model: wind field consistency, conservation, plume physics."""
import copy
import math
import os
import time

import numpy as np
import pytest

from module3_gas_safety.gas_model import (GasConfig, GasDispersion2D, LaggedSensor, WindField,
                                          rects_from_flat, solid_mask_from_map)


def _box(nx=60, ny=30, obstacle=True):
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    if obstacle:
        solid[10:20, 25:32] = True
    return solid


# --------------------------------------------------------------- wind field
def test_projected_wind_is_divergence_free(plant_model):
    w = plant_model.wind
    for (u, v) in (w.W0, w.W90):
        div = w.divergence(u, v)
        assert np.max(np.abs(div)) < 5e-6
    # instantaneous gusty field (linear combination + eddies) stays consistent
    m = copy.deepcopy(plant_model)
    for _ in range(30):
        m.step(0.5)
    assert np.max(np.abs(m.wind.divergence(m.wind.u, m.wind.v))) < 1e-5


def test_no_flow_through_obstacles(plant_model):
    w = plant_model.wind
    m = copy.deepcopy(plant_model)
    for _ in range(10):
        m.step(0.5)
    closed_u = ~w.flow_u
    closed_v = ~w.flow_v
    assert np.all(m.wind.u[closed_u] == 0.0)
    assert np.all(m.wind.v[closed_v] == 0.0)
    # every wall face that carries flow is a configured opening
    fp = np.pad(w.fluid, 1, constant_values=False)
    wall_u = fp[1:-1, :-1] ^ fp[1:-1, 1:]
    assert np.all(~(wall_u & w.flow_u) | w.in_u | w.out_u)


def test_eddy_modes_are_solenoidal_and_wall_tangent(plant_model):
    w = plant_model.wind
    assert len(w.eddies) == plant_model.cfg.eddy_modes
    for eu, ev in w.eddies:
        assert np.max(np.abs(w.divergence(eu, ev))) < 1e-12
        assert np.all(eu[~w.open_u] == 0.0) and np.all(ev[~w.open_v] == 0.0)


def test_projection_mass_balance_of_openings(plant_model):
    """Net inflow through the vents equals the outflow through the outlets."""
    w = plant_model.wind
    u, v = w.W0
    h = w.h
    inflow = (np.sum(u[w.in_u] * np.where(np.pad(w.fluid, 1)[1:-1, 1:][w.in_u], 1, -1))
              + np.sum(v[w.in_v] * np.where(np.pad(w.fluid, 1)[1:, 1:-1][w.in_v], 1, -1))) * h
    outflow = (np.sum(u[w.out_u] * np.where(np.pad(w.fluid, 1)[1:-1, :-1][w.out_u], 1, -1))
               + np.sum(v[w.out_v] * np.where(np.pad(w.fluid, 1)[:-1, 1:-1][w.out_v], 1, -1))) * h
    assert inflow > 1.0
    assert abs(inflow - outflow) < 1e-3 * inflow


# --------------------------------------------------------------- transport
def test_mass_conservation_closed_box_without_decay():
    cfg = GasConfig(resolution=0.25, inflow_rects=[], outflow_rects=[], decay_rate=0.0,
                    emission_rate=0.0, eddy_rms=0.3, eddy_modes=4, diffusivity=0.05)
    m = GasDispersion2D(_box(), 0.25, (0.0, 0.0), cfg)
    X, Y = m.cell_centres()
    m.c = np.where(m.fluid, 100.0 * np.exp(-((X - 3.0) ** 2 + (Y - 3.5) ** 2)), 0.0)
    m0 = m.total_mass
    for _ in range(300):
        m.step(0.2)
    assert abs(m.total_mass - m0) / m0 < 1e-10
    assert m.outflow == 0.0 and m.decayed == 0.0
    assert np.all(m.c >= 0.0)
    # the gas actually moved (eddies + diffusion)
    assert np.max(m.c) < 50.0


def test_emission_accumulates_exactly_in_closed_box():
    cfg = GasConfig(resolution=0.25, inflow_rects=[], outflow_rects=[], decay_rate=0.0,
                    emission_rate=20.0, source_xy=(4.0, 2.0), eddy_rms=0.2)
    m = GasDispersion2D(_box(), 0.25, (0.0, 0.0), cfg)
    for _ in range(100):
        m.step(0.3)
    assert m.total_mass == pytest.approx(20.0 * 30.0, rel=1e-9)


def test_mass_budget_closes_in_open_plant(plant_model):
    m = plant_model
    total = m.total_mass + m.outflow + m.decayed
    assert total == pytest.approx(m.emitted, rel=1e-8)


def test_plume_follows_the_wind():
    """Open channel (vents west/south, outlets east/north): the uniform first guess is
    already mass consistent, so the plume must follow the wind direction exactly."""
    ny, nx = 24, 80
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    h = 0.25
    for wind, expect_dy in (((0.4, 0.0), 0.0), ((0.4, 0.08), 1.0)):
        cfg = GasConfig(resolution=h, wind_base=wind, factory_x_max=1e9, gusts_enabled=False,
                        inflow_rects=[(0.0, -1.0, 0.4, 7.0), (0.0, -0.2, 20.0, 0.4)],     # west + south
                        outflow_rects=[(19.6, -1.0, 20.2, 7.0), (0.0, 5.6, 20.0, 6.2)],   # east + north
                        source_xy=(5.0, 3.0), emission_rate=20.0, decay_rate=0.0, diffusivity=0.02,
                        eddy_modes=0)
        m = GasDispersion2D(solid, h, (0.0, 0.0), cfg)
        m.spin_up(150.0, 0.5)
        down = m.concentration_at(8.0, 3.0 + 0.2 * 3.0 * expect_dy)
        up = m.concentration_at(2.0, 3.0)
        assert down > 20.0 * max(up, 1e-3)
        X, Y = m.cell_centres()
        w = m.c * (X > 6.0)
        cy = float(np.sum(w * Y) / np.sum(w))
        if expect_dy == 0.0:
            assert abs(cy - 3.0) < 0.2
        else:   # positive v component tilts the plume north
            assert cy > 3.2


def test_plant_plume_reaches_doorway_and_calibration(plant_model):
    m = plant_model
    X, Y = m.cell_centres()
    near = (np.hypot(X + 3.2, Y - 2.0) < 0.75) & m.fluid
    assert 500.0 <= float(m.c[near].max()) <= 2000.0
    door = (X > 13.9) & (X < 14.9) & (np.abs(Y) < 2.8) & m.fluid
    assert 20.0 <= float(m.c[door].max()) <= 80.0
    # the plume direction: downwind (east) of the source is much higher than upwind (west wall side)
    assert m.concentration_at(0.0, 2.0) > 10.0 * m.concentration_at(-8.0, -4.0)
    # corridor concentration decays towards the dock
    assert m.concentration_at(20.0, 0.0) > m.concentration_at(45.0, 0.0) > m.concentration_at(60.0, 0.0)


def test_positivity_and_real_time_speed(plant_model):
    m = copy.deepcopy(plant_model)
    t0 = time.perf_counter()
    for _ in range(100):
        m.step(0.1)
    per_step = (time.perf_counter() - t0) / 100
    assert np.all(m.c >= 0.0)
    assert per_step < 0.05          # 10 Hz real time with a large margin (measured ~1 ms)


def test_gusts_change_the_wind_but_not_without_gusts(plant_model):
    m = copy.deepcopy(plant_model)
    u0 = m.wind.u.copy()
    m.step(5.0)
    assert np.max(np.abs(m.wind.u - u0)) > 1e-3
    m.cfg.gusts_enabled = False
    m.step(1.0)
    assert np.allclose(m.wind.u, m.wind.W0[0])


def test_source_inside_obstacle_falls_back_to_nearest_fluid_cell():
    cfg = GasConfig(resolution=0.25, inflow_rects=[], outflow_rects=[], source_xy=(7.0, 3.5))
    m = GasDispersion2D(_box(), 0.25, (0.0, 0.0), cfg)
    assert m.source_weights.sum() == pytest.approx(1.0)
    assert np.all(m.source_weights[m.solid] == 0.0)


def test_solid_mask_keeps_thin_walls_closed():
    from dusan_common.grid_map import OccupancyGridMap
    occ = np.zeros((40, 80), dtype=np.int8)
    occ[:, 41] = 100                       # 0.1 m wall
    occ[18:22, 41] = 0                     # 0.4 m door (narrower than a gas cell pair)
    g = OccupancyGridMap(occ, 0.1, 0.0, 0.0)
    solid, _ = solid_mask_from_map(g, 0.25, 0.15)
    assert solid[:, 16].sum() + solid[:, 17].sum() >= solid.shape[0] - 2


def test_sampling_and_sensor_models(plant_model):
    m = plant_model
    c = m.concentration_at(np.array([0.0, 5.0]), np.array([2.0, 0.0]))
    assert c.shape == (2,) and np.all(c >= 0.0)
    pts = m.body_points(1.0, 2.0, math.pi / 2, [(0.3, 0.15)])
    assert np.allclose(pts[0], [1.0 - 0.15, 2.0 + 0.3])
    s = LaggedSensor(1, tau=1.0, noise_abs=0.0, noise_rel=0.0)
    s.update([0.0], 0.1)
    y = 0.0
    for _ in range(10):
        y = float(s.update([100.0], 0.1)[0])
    assert y == pytest.approx(100.0 * (1.0 - math.exp(-1.0)), rel=1e-6)


def test_rects_from_flat():
    assert rects_from_flat([0, 1, 2, 3, 4, 5, 6, 7]) == [(0.0, 1.0, 2.0, 3.0), (4.0, 5.0, 6.0, 7.0)]
    with pytest.raises(ValueError):
        rects_from_flat([1, 2, 3])


def test_real_map_sanity(layout):
    """If the generated plant_map.yaml exists, the model must work on it too."""
    from module3_gas_safety.layout_map import load_nav_map
    path = layout.nav_map_yaml()
    if not os.path.exists(path):
        pytest.skip("plant_map.yaml not generated yet")
    g, _ = load_nav_map("", layout)
    m = GasDispersion2D.from_nav_map(g, GasConfig())
    assert np.max(np.abs(m.wind.divergence(*m.wind.W0))) < 5e-6
    m.spin_up(300.0, 0.5)
    X, Y = m.cell_centres()
    assert float(m.c[(np.hypot(X + 3.2, Y - 2.0) < 1.0) & m.fluid].max()) > 300.0
    assert float(m.c[(X > 13.9) & (X < 14.9) & (np.abs(Y) < 2.8) & m.fluid].max()) > 5.0
    assert WindField is not None
