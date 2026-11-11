"""Comm model: path loss, wall attenuation, shadow zones, batch ray casting, map."""
import numpy as np
import pytest

from dusan_common.grid_map import OccupancyGridMap

from module3_gas_safety.comm_model import (AccessPoint, CommConfig, CommModel, FadingProcess, RssiMap,
                                           ShadowZone, config_from_layout, rf_obstacle_grid)


def _empty(w=400, h=100, res=0.1):
    return OccupancyGridMap(np.zeros((h, w), np.int8), res, 0.0, 0.0)


def test_rssi_decreases_monotonically_with_distance():
    cfg = CommConfig(access_points=[AccessPoint("A", 1.0, 5.0, 0.6, -30.0)], path_loss_exponent=2.2)
    m = CommModel(_empty(), cfg)
    xs = np.linspace(2.0, 39.0, 40)
    r = [m.rssi_at(x, 5.0) for x in xs]
    assert all(a > b for a, b in zip(r[:-1], r[1:]))
    # 10 n log10 law: doubling the distance costs 10*2.2*log10(2) dB
    assert m.rssi_at(5.0, 5.0) - m.rssi_at(9.0, 5.0) == pytest.approx(22.0 * np.log10(2.0), abs=1e-6)


def test_wall_attenuation_per_crossing():
    g = _empty()
    # free-standing 0.2 x 10 m walls (2 m^2): literal per-crossing model (no clutter filter)
    cfg = CommConfig(access_points=[AccessPoint("A", 1.0, 5.0, 0.6, -30.0)], wall_attenuation_db=12.0,
                     rf_min_obstacle_area_m2=0.0)
    free = CommModel(_empty(), cfg).rssi_at(30.0, 5.0)
    g.occ[:, 100:102] = 100          # wall at x = 10
    one = CommModel(g, cfg).rssi_at(30.0, 5.0)
    g.occ[:, 200:202] = 100          # second wall at x = 20
    two = CommModel(g, cfg).rssi_at(30.0, 5.0)
    assert free - one == pytest.approx(12.0)
    assert free - two == pytest.approx(24.0)


def test_shadow_zone_and_strongest_ap():
    cfg = CommConfig(access_points=[AccessPoint("A", 1.0, 5.0, 0.6, -30.0), AccessPoint("B", 39.0, 5.0, 0.6, -30.0)],
                     shadow_zones=[ShadowZone("S", (18.0, 3.0, 22.0, 7.0), 40.0)])
    m = CommModel(_empty(), cfg)
    assert m.best_ap(5.0, 5.0) == "A" and m.best_ap(35.0, 5.0) == "B"
    assert m.rssi_at(20.0, 5.0) == pytest.approx(max(m.ap_rssi(20.0, 5.0)) - 40.0)
    assert m.rssi_at(17.0, 5.0) - m.rssi_at(19.0, 5.0) > 35.0


@pytest.mark.parametrize("min_area", [0.0, 2.5])
def test_batch_caster_matches_shared_raycast(layout, layout_grid, min_area):
    m = CommModel.from_layout(layout, layout_grid, rf_min_obstacle_area_m2=min_area)
    if min_area == 0.0:
        assert m.rf_grid is layout_grid
    rng = np.random.default_rng(1)
    xs = rng.uniform(-10.5, 64.0, 300)
    ys = rng.uniform(-9.5, 9.5, 300)
    for ap in m.cfg.access_points:
        batch = m.crossings_batch(ap, xs, ys)
        ref = [m.rf_grid.raycast_crossings(ap.x, ap.y, x, y)[0] for x, y in zip(xs, ys)]
        assert np.array_equal(batch, ref)


def test_small_clutter_is_rf_transparent_walls_are_not():
    g = _empty()
    g.occ[45:55, 150:155] = 100          # 1.0 x 0.5 m post (0.5 m^2) at x = 15
    g.occ[:, 250:252] = 100              # wall at x = 25 (full height of the map: 2 m^2)
    cfg = CommConfig(access_points=[AccessPoint("A", 1.0, 5.0, 0.6, -30.0)], wall_attenuation_db=12.0,
                     rf_min_obstacle_area_m2=1.0)
    free = CommModel(_empty(), cfg).rssi_at(35.0, 5.0)
    m = CommModel(g, cfg)
    assert m.rf_grid is not g and np.all(m.rf_grid.occ[45:55, 150:155] == 0)
    assert np.all(m.rf_grid.occ[:, 250:252] == 100)
    assert np.count_nonzero(g.occ == 100) == 50 + 200      # the nav map itself is untouched
    assert free - m.rssi_at(35.0, 5.0) == pytest.approx(12.0)   # only the wall attenuates
    cfg.rf_min_obstacle_area_m2 = 0.0                       # literal model: both count
    assert free - CommModel(g, cfg).rssi_at(35.0, 5.0) == pytest.approx(24.0)
    assert rf_obstacle_grid(None, 1.0) is None


def test_map_matches_point_model_and_encoding(layout, layout_grid):
    m = CommModel.from_layout(layout, layout_grid)
    rmap = m.compute_map(0.5)
    ny, nx = rmap.shape
    rng = np.random.default_rng(2)
    for _ in range(30):
        i, j = int(rng.integers(nx)), int(rng.integers(ny))
        x = rmap.origin_x + (i + 0.5) * 0.5
        y = rmap.origin_y + (j + 0.5) * 0.5
        assert rmap.at(x, y) == pytest.approx(m.rssi_at(x, y), abs=1e-6)
    vals = rmap.to_grid_values()
    assert vals.min() >= 0 and vals.max() <= 100
    back = RssiMap.from_grid_values(vals, 0.5, rmap.origin_x, rmap.origin_y)
    clipped = np.clip(rmap.field, -100.0, -30.0)
    assert np.max(np.abs(back.field - clipped)) <= 0.7 / 2 + 1e-9   # half a quantisation step


def test_layout_coverage_shadow_s1_and_good_dock(layout, layout_grid):
    m = CommModel.from_layout(layout, layout_grid)
    cfg = config_from_layout(layout)
    assert m.rssi_at(-8.0, -7.5) < cfg.rssi_min_dbm          # turbine pit (S1)
    assert m.rssi_at(61.9, 0.0) > -50.0                      # charging station next to AP1
    assert m.rssi_at(0.0, 0.0) > cfg.rssi_min_dbm            # hall centre is covered
    # gauge viewpoints (1.5 m in front of each dial) are all covered
    for g in layout.gauges:
        nx, ny = g.get("normal", [0, 0, 0])[:2]
        assert m.rssi_at(g["position"][0] + 1.5 * nx, g["position"][1] + 1.5 * ny) > cfg.rssi_min_dbm, g["id"]


def test_fading_process_statistics():
    f = FadingProcess(2.0, 1.0, seed=3)
    xs = np.array([f.step(0.5) for _ in range(20000)])
    assert abs(xs.mean()) < 0.2 and xs.std() == pytest.approx(2.0, rel=0.1)
    assert FadingProcess(0.0).step(1.0) == 0.0
