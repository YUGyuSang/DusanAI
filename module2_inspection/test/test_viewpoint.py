"""Unit tests of module2_inspection.viewpoint_planner (ROS free).

The generated nav map (simulation/maps/plant_map.yaml) may not exist in a
fresh checkout, so the tests rasterise their own maps (from the layout or a
tiny synthetic room)."""
import json
import math
import os

import numpy as np
import pytest

from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout
from dusan_common.ros_helpers import to_json
from module2_inspection.viewpoint_planner import (
    ViewpointParams, ViewpointPlanner, brute_force_tour, held_karp, layout_occupancy_map, nearest_neighbour_2opt,
    tour_cost)

LAYOUT = Layout.load()


@pytest.fixture(scope="module")
def grid():
    return layout_occupancy_map(LAYOUT)


@pytest.fixture(scope="module")
def planner(grid):
    return ViewpointPlanner(LAYOUT, grid)


@pytest.fixture(scope="module")
def patrol(planner):
    return planner.plan()


def _room_layout(gauge_pos=(5.0, 2.0, 0.9), normal=(-1, 0, 0)):
    d = {
        "factory": {"interior": {"x_min": 0.0, "x_max": 8.0, "y_min": 0.0, "y_max": 4.0}, "height": 4.0, "floor_z": 0.0},
        "charging_station": {"dock_pose": [1.0, 2.0, 0.0]},
        "nav_map": {"bounds": [-0.5, -0.5, 8.5, 4.5]},
        "gauge_defaults": {"dial_diameter": 0.30},
        "gauges": [{"id": "T1", "position": list(gauge_pos), "normal": list(normal)}],
        "machines": [], "obstacles": [], "hotspots": [],
    }
    return Layout(d)


# --------------------------------------------------------------------------- map
def test_layout_map_rules(grid):
    occ = lambda x, y: grid.occ[grid.world_to_grid(x, y)[1], grid.world_to_grid(x, y)[0]]
    assert occ(-1.5, -4.5) == 100            # turbine footprint
    assert occ(-6.5, 3.8) == 100             # gas tank
    assert occ(30.0, 0.0) == 0               # corridor
    assert occ(61.9, 0.0) == 0               # dock pad
    assert occ(14.95, 0.0) == 0              # east opening
    assert occ(49.0, -1.0) == 0 and occ(49.0, 1.0) == 0      # stairs / ramp free
    cor = LAYOUT.get("corridor")
    st, rp = cor["stairs"], cor["ramp"]
    assert occ(48.0, 0.5 * (st["y_min"] + st["y_max"])) == 0 and occ(48.0, 0.5 * (rp["y_min"] + rp["y_max"])) == 0
    drop = cor.get("dropoff")                                 # ramp edge above the stairs
    if drop:
        assert occ(0.5 * (drop["x_min"] + drop["x_max"]), 0.5 * (drop["y_min"] + drop["y_max"])) == 100
    lo, hi = sorted([(st["y_min"], st["y_max"]), (rp["y_min"], rp["y_max"])])
    if hi[0] - lo[1] > 0.25:                                  # descent strip that is neither stairs nor ramp
        assert occ(49.0, 0.5 * (lo[1] + hi[0])) == 100
    assert occ(-1.0, 4.0) == 100             # drum
    assert occ(3.5, 6.4) == 100              # rack support
    assert occ(20.0, 5.0) == 100             # outside the building


# --------------------------------------------------------------------------- candidates
def test_candidate_geometry(planner):
    g = LAYOUT.gauge("G02")
    cands = planner.candidates(g)
    p = planner.p
    n_r = len(np.arange(p.ring_min, p.ring_max + 1e-9, p.ring_step))
    n_a = 2 * int(round(p.angle_max_deg / p.angle_step_deg)) + 1
    assert len(cands) == n_r * n_a
    gx, gy = g["position"][:2]
    for c in cands:
        # inspection camera (0.30 m ahead of the base) looks straight at the dial
        cam_x, cam_y = c.x + 0.30 * math.cos(c.yaw), c.y + 0.30 * math.sin(c.yaw)
        assert math.hypot(gx - cam_x, gy - cam_y) == pytest.approx(c.ring, abs=1e-6)
        bearing = math.atan2(gy - c.y, gx - c.x)
        assert abs(math.remainder(bearing - c.yaw, 2 * math.pi)) < 1e-6
        assert abs(c.angle_deg) <= p.angle_max_deg + 1e-9
        assert c.reason == "ok" or not c.valid


def test_score_terms_behave(planner):
    g = LAYOUT.gauge("G02")             # (-0.5, -2.55, 1.0) facing +y, open floor in front
    near = planner.evaluate(g, -0.5, -2.55 + 1.3, -math.pi / 2, 1.0, 0.0)
    far = planner.evaluate(g, -0.5, -2.55 + 2.5, -math.pi / 2, 2.2, 0.0)
    assert near.valid and far.valid
    assert near.pixel_diameter > far.pixel_diameter
    # pixel diameter ~ f * dial / distance (major axis is not foreshortened)
    f = planner.cam.fx
    assert far.pixel_diameter == pytest.approx(f * 0.30 / far.distance, rel=0.06)
    assert near.v_offset_px < 0 and near.elevation_deg > 0      # dial above the camera axis
    assert near.components["vertical"] < far.components["vertical"]
    # off-normal viewpoint -> larger incidence, lower incidence term
    a = math.radians(35.0)
    bx, by = -0.5 + 1.7 * math.sin(a), -2.55 + 1.7 * math.cos(a)
    obl = planner.evaluate(g, bx, by, math.atan2(-2.55 - by, -0.5 - bx), 1.4, 35.0)
    # independent check of the 3-D incidence (camera 0.30 m ahead, 0.54 m above the floor)
    cam = np.array([bx + 0.30 * math.cos(obl.yaw), by + 0.30 * math.sin(obl.yaw), 0.54])
    v = cam - np.asarray(g["position"])
    expected = math.degrees(math.acos(v[1] / np.linalg.norm(v)))
    assert obl.incidence_deg == pytest.approx(expected, abs=1e-6)
    assert obl.incidence_deg > near.incidence_deg + 10.0
    assert obl.components["incidence"] < near.components["incidence"]
    assert 0.0 <= obl.score <= 1.0


def test_hard_constraints():
    lay = _room_layout()
    occ = np.zeros((50, 90), np.int8)
    occ[:5, :] = occ[-5:, :] = 100
    occ[:, :5] = occ[:, -5:] = 100
    occ[10:40, 40:42] = 100                 # wall segment at x = 3.5 .. 3.7 between dock and gauge
    grid = OccupancyGridMap(occ, 0.1, -0.5, -0.5)
    pl = ViewpointPlanner(lay, grid)
    g = lay.gauge("T1")
    blocked = pl.evaluate(g, 2.7, 2.0, 0.0)            # wall between camera and dial
    assert not blocked.valid and "no_line_of_sight" in blocked.reason and blocked.score == 0.0
    clear = pl.evaluate(g, 3.9, 2.0, 0.0)
    assert clear.valid, clear.reason
    inwall = pl.evaluate(g, 3.6, 2.0, 0.0)
    assert not inwall.valid and "not_free" in inwall.reason
    behind = pl.evaluate(g, 6.0, 2.0, math.pi)         # behind the dial
    assert not behind.valid and "incidence" in behind.reason
    sideways = pl.evaluate(g, 3.9, 2.0, math.pi / 2)   # camera looks away
    assert "out_of_image" in sideways.reason
    best, cands = pl.best_viewpoint(g)
    assert best is not None and best.valid and best.x > 3.7
    assert best.score == max(c.score for c in cands)


def test_unreachable_region():
    lay = _room_layout()
    occ = np.zeros((50, 90), np.int8)
    occ[:5, :] = occ[-5:, :] = 100
    occ[:, :5] = occ[:, -5:] = 100
    occ[:, 40:43] = 100                     # full wall: gauge side cut off from the dock
    grid = OccupancyGridMap(occ, 0.1, -0.5, -0.5)
    pl = ViewpointPlanner(lay, grid)
    vp = pl.evaluate(lay.gauge("T1"), 4.2, 2.0, 0.0)      # free, but on the far side of the wall
    assert not vp.valid and "unreachable" in vp.reason
    plan = pl.plan()
    assert plan.unreachable == ["T1"] and plan.order == []


# --------------------------------------------------------------------------- TSP
@pytest.mark.parametrize("n", [2, 3, 5, 7, 8])
@pytest.mark.parametrize("closed", [True, False])
def test_held_karp_matches_brute_force(n, closed):
    rng = np.random.default_rng(n * 10 + closed)
    for sym in (True, False):
        pts = rng.uniform(0, 50, (n, 2))
        D = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
        if not sym:
            D = D * rng.uniform(1.0, 1.5, D.shape)
            np.fill_diagonal(D, 0.0)
        o_hk, c_hk = held_karp(D, closed)
        o_bf, c_bf = brute_force_tour(D, closed)
        assert c_hk == pytest.approx(c_bf, rel=1e-12)
        assert sorted(o_hk) == list(range(n)) and o_hk[0] == 0
        assert tour_cost(D, o_hk, closed) == pytest.approx(c_hk, rel=1e-12)
        o_nn, c_nn = nearest_neighbour_2opt(D, closed)
        assert c_nn >= c_hk - 1e-9 and sorted(o_nn) == list(range(n))


def test_held_karp_infinite_edges():
    D = np.array([[0, 1, np.inf], [1, 0, 2], [np.inf, 2, 0]], float)
    order, cost = held_karp(D, return_to_start=False)
    assert order == [0, 1, 2] and cost == pytest.approx(3.0)
    order, cost = held_karp(D, return_to_start=True)
    assert math.isinf(cost)


# --------------------------------------------------------------------------- full plan
def test_full_plan_on_layout_map(patrol, grid):
    plan = patrol
    ids = [g["id"] for g in LAYOUT.gauges]
    assert sorted(plan.order) == sorted(ids) and plan.unreachable == []
    assert plan.method == "held_karp"
    assert plan.total_length <= plan.naive_length + 1e-6
    assert len(plan.legs) == len(ids) + 1                       # return to dock
    dock = LAYOUT.dock_pose
    assert math.hypot(plan.legs[0][0][0] - dock[0], plan.legs[0][0][1] - dock[1]) < 1e-6
    assert math.hypot(plan.legs[-1][-1][0] - dock[0], plan.legs[-1][-1][1] - dock[1]) < 1e-6
    for i, vp in enumerate(plan.viewpoints):
        assert vp.valid and vp.score > 0.5
        assert math.hypot(plan.legs[i][-1][0] - vp.x, plan.legs[i][-1][1] - vp.y) < 1e-6
        assert math.hypot(plan.legs[i + 1][0][0] - vp.x, plan.legs[i + 1][0][1] - vp.y) < 1e-6
    assert sum(plan.leg_lengths) == pytest.approx(plan.total_length)
    for leg in plan.legs:
        for x, y in leg[1:-1]:
            assert grid.is_free(x, y, 0.2)
    # optimal cost equals the Held-Karp cost on the distance matrix
    D = plan.distance_matrix
    idx = [0] + [plan.node_ids.index(g) for g in plan.order]
    assert tour_cost(D, idx, True) == pytest.approx(plan.total_length, rel=1e-9)


def test_plan_json_contract(patrol):
    tags = {g["id"]: g["tag"] for g in LAYOUT.gauges}
    data = patrol.to_json(tags)
    for key in ("start", "viewpoints", "order", "legs", "total_length"):
        assert key in data
    assert len(data["start"]) == 2 and len(data["viewpoints"]) == len(data["order"])
    for v in data["viewpoints"]:
        for key in ("gauge_id", "x", "y", "yaw", "score", "distance", "incidence_deg", "pixel_diameter"):
            assert key in v and v[key] is not None
    assert [v["gauge_id"] for v in data["viewpoints"]] == data["order"]
    assert all(len(p) == 2 for leg in data["legs"] for p in leg)
    json.loads(to_json(data))
    pts = patrol.route_points()
    with_yaw = [p for p in pts if len(p) == 3]
    assert len(with_yaw) == len(patrol.viewpoints)
    assert with_yaw[0][2] == pytest.approx(patrol.viewpoints[0].yaw)


def test_open_route_not_longer(planner, patrol):
    open_plan = planner.plan(return_to_dock=False)
    assert len(open_plan.legs) == len(open_plan.order)
    assert open_plan.total_length <= patrol.total_length + 1e-6
    assert open_plan.total_length <= open_plan.naive_length + 1e-6


def test_params_from_dict():
    p = ViewpointParams.from_dict({"ring_max": 2.0, "avoid_zone_ids": ["Z-HOT", ""], "bogus": 1})
    assert p.ring_max == 2.0 and p.avoid_zone_ids == ("Z-HOT",)


_GEN_MAP = LAYOUT.nav_map_yaml()


@pytest.mark.skipif(not os.path.exists(_GEN_MAP), reason="simulation/maps/plant_map.yaml not generated yet")
def test_plan_on_generated_nav_map():
    """Contract check with the generated map: every gauge has a valid,
    reachable viewpoint and the optimal tour is not longer than the layout order."""
    grid = OccupancyGridMap.load(_GEN_MAP)
    plan = ViewpointPlanner(LAYOUT, grid).plan()
    assert plan.unreachable == []
    assert sorted(plan.order) == sorted(g["id"] for g in LAYOUT.gauges)
    assert all(vp.valid and vp.score > 0.5 for vp in plan.viewpoints)
    assert plan.total_length <= plan.naive_length + 1e-6
    for leg in plan.legs:
        for x, y in leg[1:-1]:
            assert grid.is_free(x, y, 0.2)


def test_patrol_yaml_matches_declared_parameters():
    """ROS 2 rejects YAML values whose type differs from the declared default."""
    import ast
    import yaml
    here = os.path.dirname(__file__)
    src = open(os.path.join(here, "..", "module2_inspection", "patrol_planner_node.py"), encoding="utf-8").read()
    declared = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ("NODE_PARAMS", "PLANNER_PARAMS")
                                                for t in node.targets):
            declared.update(eval(compile(ast.Expression(node.value), "node", "eval"),
                                 {"_VP": ViewpointParams(), "list": list}))
    assert set(ViewpointParams.__dataclass_fields__) - {"camera"} <= set(declared)
    cfg = yaml.safe_load(open(os.path.join(here, "..", "config", "patrol.yaml"), encoding="utf-8"))
    for key, val in cfg["patrol_planner"]["ros__parameters"].items():
        assert key in declared, key
        d = declared[key]
        if isinstance(d, list):
            assert isinstance(val, list) and all(type(v) is type(d[0]) for v in val), key
        else:
            assert type(val) is type(d), f"{key}: YAML {type(val).__name__} vs declared {type(d).__name__}"


def test_unknown_gauge_id_is_reported_not_fatal():
    lay = _room_layout()
    occ = np.zeros((50, 90), np.int8)
    occ[:5, :] = occ[-5:, :] = 100
    occ[:, :5] = occ[:, -5:] = 100
    pl = ViewpointPlanner(lay, OccupancyGridMap(occ, 0.1, -0.5, -0.5))
    plan = pl.plan(gauge_ids=["T1", "NOPE"])
    assert plan.order == ["T1"] and plan.unreachable == ["NOPE"]
    assert len(plan.legs) == 2 and plan.total_length > 0.0
    json.loads(to_json(plan.to_json()))


def test_route_legs_take_terrain_detours():
    """With the terrain cost (default) the legs dock -> G08 -> G07 -> dock pass the north end of
    C-OBS1 and the C-OBS3 north passage; only the corridor plate C-OBS4 (no detour) is crossed."""
    from dusan_common import terrain_cost as tc
    if not os.path.exists(LAYOUT.nav_map_yaml()):
        pytest.skip("simulation/maps/plant_map.yaml not generated")
    grid = OccupancyGridMap.load(LAYOUT.nav_map_yaml())
    with_cost = ViewpointPlanner(LAYOUT, grid).plan(gauge_ids=["G08", "G07"])
    without = ViewpointPlanner(LAYOUT, grid, ViewpointParams(terrain_cost_enabled=False)).plan(gauge_ids=["G08", "G07"])
    crossed = set().union(*(tc.debris_crossed(leg, LAYOUT) for leg in with_cost.legs))
    crossed0 = set().union(*(tc.debris_crossed(leg, LAYOUT) for leg in without.legs))
    assert crossed == {"C-OBS4"}
    assert {"C-OBS1", "C-OBS3", "C-OBS4"} <= crossed0
    assert without.total_length < with_cost.total_length < 1.05 * without.total_length
    p = ViewpointParams.from_dict({"terrain_w_rubble": 5.0, "terrain_cost_enabled": False})
    assert p.terrain_w_rubble == 5.0 and not p.terrain_cost_enabled
    assert ViewpointPlanner(LAYOUT, grid, p).terrain_cost is None
