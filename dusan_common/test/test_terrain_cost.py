"""Terrain cost layer (dusan_common.terrain_cost) and the cost-aware planner /
smoother (dusan_common.planning)."""
import math
import os

import numpy as np
import pytest

from dusan_common import terrain_cost as tc
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout, default_layout_path
from dusan_common.planning import astar, densify, path_length, plan, smooth_path


def _corridor(length=12.0, half_width=1.8, res=0.1):
    """Open corridor x 0 .. length, walls at y = +-half_width."""
    w = int(round(length / res)) + 2
    h = int(round(2 * half_width / res)) + 2
    occ = np.zeros((h, w), np.int8)
    occ[0, :] = occ[-1, :] = 100
    occ[:, 0] = occ[:, -1] = 100
    return OccupancyGridMap(occ, res, -res, -half_width - res)


# =============================================================================
# rasteriser
# =============================================================================
def test_cost_inside_margin_and_outside():
    g = _corridor()
    rub = {"id": "R", "type": "rubble", "rect": [4.0, -1.0, 6.0, 0.0]}
    c = tc.terrain_cost_grid(g, [rub])
    prm = tc.TerrainCostParams()

    def at(x, y):
        ix, iy = g.world_to_grid(x, y)
        return c[iy, ix]

    assert at(5.0, -0.5) == pytest.approx(prm.w_rubble)                     # inside
    assert at(5.0, 0.05) == pytest.approx(prm.w_rubble)                     # within INSIDE_PAD
    assert at(5.0, 0.2) == pytest.approx(prm.margin_fraction * prm.w_rubble)  # margin
    assert at(5.0, 0.4) == 0.0 and at(2.0, 0.0) == 0.0                      # outside
    assert prm.margin_fraction * prm.w_rubble < prm.w_rubble                 # margin << inside
    # plates / platform small, pipes / rubble high
    assert prm.w_plate < prm.w_pipe and prm.w_platform < prm.w_rubble
    # drums / pallets are occupied in the nav map: no cost layer
    assert not np.any(tc.terrain_cost_grid(g, [{"id": "D", "type": "drum", "center": [5, 0], "radius": 0.3}]))


def test_margin_is_capped_and_params_from_dict():
    prm = tc.TerrainCostParams.from_params({"terrain_margin": 0.6, "terrain_w_pipe": 10, "terrain_cost_enabled": False,
                                            "terrain_w_rubble": "bogus", "unrelated": 1})
    assert prm.margin_m == tc.MAX_MARGIN == 0.25
    assert prm.w_pipe == 10.0 and isinstance(prm.w_pipe, float)
    assert prm.w_rubble == tc.TerrainCostParams().w_rubble               # bad value -> default
    assert not prm.enabled
    g = _corridor()
    assert tc.cost_from_params(g, [{"id": "P", "type": "pipe", "center": [5, 0], "yaw": 1.57,
                                    "length": 3, "diameter": 0.1}], {"terrain_cost_enabled": False}) is None
    assert set(tc.TERRAIN_COST_DEFAULTS) == {"terrain_cost_enabled", "terrain_w_pipe", "terrain_w_rubble",
                                             "terrain_w_plate", "terrain_w_platform", "terrain_margin",
                                             "terrain_margin_fraction"}
    assert tc.TerrainCostParams.from_params(tc.TERRAIN_COST_DEFAULTS) == tc.TerrainCostParams()


def test_thin_diagonal_pipe_is_a_closed_line():
    """An 8-connected A* must not slip diagonally through an 8 cm pipe."""
    g = _corridor()
    pipe = {"id": "P", "type": "pipe", "center": [6.0, 0.0], "yaw": math.radians(45.0), "length": 6.0,
            "diameter": 0.08}
    c = tc.terrain_cost_grid(g, [pipe])
    inside = c >= tc.TerrainCostParams().w_pipe
    h, w = inside.shape
    for iy in range(1, h - 1):
        for ix in range(1, w - 1):
            if not inside[iy, ix]:
                # a diagonal step between two non-inside cells must not cross the pipe axis
                for dx, dy in ((1, 1), (1, -1)):
                    jx, jy = ix + dx, iy + dy
                    if 0 < jx < w - 1 and 0 < jy < h - 1 and not inside[jy, jx]:
                        a, b = g.grid_to_world(ix, iy), g.grid_to_world(jx, jy)
                        d = tc.footprint_distance(pipe, np.array([a[0], b[0], 0.5 * (a[0] + b[0])]),
                                                  np.array([a[1], b[1], 0.5 * (a[1] + b[1])]))
                        assert d[2] > 0.0


def test_layout_stand_ins_without_obstacles():
    g = _corridor()

    class NoObstacles:
        pass

    assert not np.any(tc.terrain_cost_grid(g, NoObstacles()))
    assert not np.any(tc.terrain_cost_grid(g, None))
    assert not np.any(tc.terrain_cost_grid(g, {"obstacles": None}))


def test_cache_recomputes_only_on_grid_change():
    g = _corridor()
    obs = [{"id": "R", "type": "rubble", "rect": [4.0, -1.0, 6.0, 0.0]}]
    cache = tc.TerrainCostCache(obs)
    c1 = cache(g)
    assert cache(g) is c1
    g2 = _corridor(length=14.0)
    c2 = cache(g2)
    assert c2.shape == g2.occ.shape and c2 is not c1
    assert tc.TerrainCostCache(obs, tc.TerrainCostParams(enabled=False))(g) is None
    assert cache(None) is None


# =============================================================================
# planner + smoother
# =============================================================================
def test_detour_taken_when_available_and_crossing_when_not():
    g = _corridor()
    # pipe across the corridor leaving a 0.8 m gap at the north wall
    gap_pipe = {"id": "P1", "type": "pipe", "center": [6.0, -0.4], "yaw": math.pi / 2, "length": 2.8,
                "diameter": 0.10}
    cost = tc.terrain_cost_grid(g, [gap_pipe])
    p0 = plan(g, (1.0, 0.0), (11.0, 0.0), robot_radius=0.3)
    p1 = plan(g, (1.0, 0.0), (11.0, 0.0), robot_radius=0.3, extra_cost=cost)
    assert tc.debris_crossed(p0, [gap_pipe]) == ["P1"]
    assert tc.debris_crossed(p1, [gap_pipe]) == []
    assert tc.debris_crossed(p1, [gap_pipe], clearance=tc.INSIDE_PAD) == []
    assert path_length(p1) > path_length(p0)
    # the same pipe spanning the whole corridor: still a path, straight across
    full_pipe = dict(gap_pipe, id="P2", center=[6.0, 0.0], length=3.8)
    cost = tc.terrain_cost_grid(g, [full_pipe])
    p2 = plan(g, (1.0, 0.0), (11.0, 0.0), robot_radius=0.3, extra_cost=cost)
    assert p2 is not None and tc.debris_crossed(p2, [full_pipe]) == ["P2"]
    assert path_length(p2) == pytest.approx(10.0, abs=0.3)


def test_smoothing_does_not_cut_across_costly_cells():
    """A* goes around a rubble corner; the plain line-of-sight smoother would cut
    the corner, the cost-aware one keeps out of the rubble."""
    g = _corridor(length=12.0, half_width=3.0)
    rub = {"id": "R", "type": "rubble", "rect": [4.0, -3.0, 7.0, 1.0]}
    cost = tc.terrain_cost_grid(g, [rub])
    raw = astar(g, (2.0, -2.0), (9.0, -2.0), robot_radius=0.3, extra_cost=cost)
    assert tc.debris_crossed(raw, [rub]) == []
    plain = smooth_path(g, raw, 0.3)
    aware = smooth_path(g, raw, 0.3, extra_cost=cost)
    assert tc.debris_crossed(plain, [rub]) == ["R"]                  # the bug the cost check fixes
    assert tc.debris_crossed(aware, [rub]) == []
    assert len(aware) < len(densify(raw, 1.5)) + 5                    # still a smoothed path
    ia, ma = tc.path_cost(g, cost, aware)
    ir, mr = tc.path_cost(g, cost, raw)
    assert ma <= mr + 1e-9 and ia <= ir * 1.05 + 1e-6
    # without a cost layer the smoother is unchanged
    assert smooth_path(g, raw, 0.3, extra_cost=np.zeros_like(cost)) == plain


# =============================================================================
# the real plant (world_layout.yaml + the generated nav map)
# =============================================================================
@pytest.fixture(scope="module")
def plant():
    lay = Layout.load(default_layout_path())
    path = lay.nav_map_yaml()
    if not os.path.exists(path):
        pytest.skip("simulation/maps/plant_map.yaml not generated")
    grid = OccupancyGridMap.load(path)
    return lay, grid, tc.terrain_cost_grid(grid, lay)


def _min_dist(path, lay, oid):
    ob = next(o for o in lay.obstacles if o["id"] == oid)
    P = np.asarray(densify([tuple(p[:2]) for p in path], 0.05))
    return float(tc.footprint_distance(ob, P[:, 0], P[:, 1]).min())


@pytest.mark.parametrize("radius", [0.30, 0.35])
def test_plant_c_obs1_north_end_detour(plant, radius):
    lay, grid, cost = plant
    a, b = lay.dock_pose[:2], (52.5, 0.8)
    base = plan(grid, a, b, robot_radius=radius)
    p = plan(grid, a, b, robot_radius=radius, extra_cost=cost)
    assert "C-OBS1" in tc.debris_crossed(base, lay)
    assert tc.debris_crossed(p, lay) == []
    assert _min_dist(p, lay, "C-OBS1") > tc.INSIDE_PAD
    assert path_length(p) < path_length(base) + 1.5


def test_plant_c_obs3_north_passage(plant):
    lay, grid, cost = plant
    base = plan(grid, (44.0, 0.0), (34.0, 0.0), robot_radius=0.30)
    p = plan(grid, (44.0, 0.0), (34.0, 0.0), robot_radius=0.30, extra_cost=cost)
    assert tc.debris_crossed(base, lay) == ["C-OBS3"]
    assert tc.debris_crossed(p, lay) == []
    rub = next(o for o in lay.obstacles if o["id"] == "C-OBS3")
    on_field = [q for q in densify(p, 0.05) if rub["rect"][0] <= q[0] <= rub["rect"][2]]
    assert on_field and min(q[1] for q in on_field) >= rub["rect"][3] + 0.25 - 1e-6   # outside the margin
    assert path_length(p) < path_length(base) + 1.0


def test_plant_path_still_found_without_detour(plant):
    """C-OBS5 spans the corridor (0.09 m gap at the wall), C-OBS4 the plate: crossed."""
    lay, grid, cost = plant
    p = plan(grid, (25.0, 0.0), (19.0, 0.0), robot_radius=0.30, extra_cost=cost)
    assert p is not None and tc.debris_crossed(p, lay) == ["C-OBS5"]
    assert path_length(p) < 6.5
    p = plan(grid, (34.0, 0.0), (28.0, 0.0), robot_radius=0.30, extra_cost=cost)
    assert p is not None and tc.debris_crossed(p, lay) == ["C-OBS4"]
    # the whole corridor dock -> hall: only what cannot be avoided
    p = plan(grid, lay.dock_pose[:2], (-2.0, -2.0), robot_radius=0.35, extra_cost=cost)
    assert p is not None and set(tc.debris_crossed(p, lay)) <= {"C-OBS4", "C-OBS5"}
