"""Lying-pipe crossings (dusan_common.debris_crossing) and the seeker's debris-blocked map."""
import math
import os

import numpy as np
import pytest

from dusan_common.debris_crossing import (CROSSING_DEFAULTS, CrossingGovernor, CrossingParams, find_crossings,
                                          layout_pipes, straighten_crossings)
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.terrain_cost import debris_blocked_grid, debris_crossed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPE = {"id": "P", "type": "pipe", "center": [0.0, 0.0], "yaw": math.radians(110.0), "length": 2.8,
        "diameter": 0.08}                      # C-OBS5 like: 20 deg from square to an x-axis corridor


def _free(w=8.0, h=4.0, res=0.1):
    occ = np.zeros((int(h / res), int(w / res)), dtype=np.int8)
    return OccupancyGridMap(occ, res, -w / 2, -h / 2)


def test_find_crossings_inside_the_span_only_and_square_enough():
    pipes = layout_pipes([PIPE, {"id": "R", "type": "rubble", "rect": [0, 0, 1, 1]}])
    assert [p.id for p in pipes] == ["P"]
    cr = find_crossings([(2.0, 0.3), (-2.0, 0.3)], pipes)
    assert len(cr) == 1 and cr[0].pipe.id == "P" and cr[0].index == 0
    assert cr[0].progress(2.0, 0.3) < 0 < cr[0].progress(-2.0, 0.3)
    assert abs(math.degrees(cr[0].heading)) == pytest.approx(160.0, abs=1e-6)      # square to the pipe, westward
    # beyond the pipe end (half 1.4 + r + end margin): no crossing
    assert find_crossings([(2.0, 3.0), (-2.0, 3.0)], pipes) == []
    # a path along the pipe that meets its axis line near the end (rounding the end): not a crossing
    u = np.array([math.cos(PIPE["yaw"]), math.sin(PIPE["yaw"])])
    a, b = 1.3 * u + np.array([0.05, 0.0]), 1.6 * u - np.array([0.05, 0.0])
    assert find_crossings([tuple(a), tuple(b)], pipes) == []
    # path distance and order
    cr = find_crossings([(3.0, 0.0), (2.0, 0.0), (-2.0, 0.0)], pipes)
    assert len(cr) == 1 and cr[0].index == 1 and cr[0].dist == pytest.approx(3.0, abs=1e-6)


def test_straighten_makes_a_square_straight_crossing():
    pipes = layout_pipes([PIPE])
    # zig-zag over the pipe like the A* path at C-OBS5
    zig = [(3.0, 0.5), (0.4, 0.45), (0.2, 0.25), (-0.1, 0.05), (-3.0, 0.2)]
    out, n = straighten_crossings(zig, pipes, None, CrossingParams())
    assert n == 1 and out[0] == zig[0] and out[-1] == zig[-1]
    cr = find_crossings(out, pipes)
    assert len(cr) == 1
    c = cr[0]
    P = np.asarray(out)
    # entry 0.9 m before the pipe axis, exit 0.8 m after it, on one straight line square to the pipe
    # through the original crossing point
    k = c.index
    E, X = P[k], P[k + 1]
    assert c.progress(*E) == pytest.approx(-0.9, abs=1e-6) and c.progress(*X) == pytest.approx(0.8, abs=1e-6)
    assert np.allclose((X - E) / np.linalg.norm(X - E), c.direction)
    c0 = find_crossings(zig, pipes)[0]
    assert np.allclose(c.point, c0.point, atol=1e-9)
    # no vertex left within the slab
    assert all(not (-0.9 + 1e-6 < c.progress(*p) < 0.8 - 1e-6) for p in out)


def test_straighten_tilts_or_keeps_when_blocked():
    pipes = layout_pipes([PIPE])
    zig = [(3.0, 0.5), (0.4, 0.45), (0.2, 0.25), (-0.1, 0.05), (-3.0, 0.2)]
    out, n = straighten_crossings(zig, pipes, lambda a, b: False, CrossingParams())
    assert n == 0 and out == zig
    # square line blocked, a tilted one free: tilt within max_angle
    def ok(a, b):
        d = np.subtract(b, a)
        ang = abs(math.atan2(d[1], d[0]) - math.radians(-160.0))
        ang = min(ang, 2 * math.pi - ang)
        return ang > math.radians(5.0) or np.hypot(*d) < 1e-9 or abs(ang - math.pi) < 0.5
    out, n = straighten_crossings(zig, pipes, ok, CrossingParams())
    assert n == 1
    # the start inside the approach slab (re-plan next to the pipe): keep the path
    near = [(0.3, 0.3), (-3.0, 0.2)]
    assert straighten_crossings(near, pipes, None, CrossingParams()) == (near, 0)
    # disabled
    assert straighten_crossings(zig, pipes, None, CrossingParams(enabled=False)) == (zig, 0)


def test_governor_turns_before_the_pipe_then_crosses_steadily():
    g = CrossingGovernor([PIPE], CrossingParams(wz=0.12))
    path = np.array([(x, 0.0) for x in np.arange(3.0, -3.01, -0.1)])
    remaining = lambda x: path[path[:, 0] < x]                              # noqa: E731
    head = math.radians(-160.0)
    # far away: untouched
    assert g.govern(2.9, 0.0, math.pi, (0.4, 0.05, 0.1), remaining(2.9)) == (0.4, 0.05, 0.1) and g.state == ""
    # in the approach slab, misaligned: turn in place towards the crossing heading
    vx, vy, wz = g.govern(0.8, 0.0, math.pi, (0.4, 0.0, 0.0), remaining(0.8))
    assert g.state == "align" and vx == 0.0 and vy == 0.0 and wz > 0.0       # 180 -> 200 deg: left
    # aligned: approach at the crossing pace
    vx, vy, wz = g.govern(0.8, 0.0, head, (0.4, 0.1, 0.5), remaining(0.8))
    assert g.state == "approach" and vx == pytest.approx(0.25) and vy == 0.0 and abs(wz) <= g.p.wz + 1e-9
    # a foot can reach the pipe: steady pace, no strafing, no turning beyond corrections - even
    # when the follower asks for a stop or a large turn
    for x in (0.3, 0.0, -0.3, -0.55):
        vx, vy, wz = g.govern(x, 0.0, math.pi, (0.0, 0.1, 0.6), remaining(x))
        assert g.state == "cross" and vx == pytest.approx(0.25) and vy == 0.0 and wz == pytest.approx(g.p.wz)
    # hind feet clear: released
    assert g.govern(-0.8, 0.0, math.pi, (0.4, 0.0, 0.0), remaining(-0.8)) == (0.4, 0.0, 0.0)
    assert g.active is None and g.n_crossings == 1


def test_governor_near_a_pipe_end_keeps_the_curve():
    end = {"id": "E", "type": "pipe", "center": [0.0, 0.0], "yaw": 0.0, "length": 2.0, "diameter": 0.12}
    g = CrossingGovernor([end], CrossingParams())
    vx, vy, wz = g.govern(1.25, 0.3, math.pi / 2, (0.4, 0.1, 0.6), [(1.25, 1.0)])
    assert g.state == "near"
    f = vx / 0.4
    assert f < 1.0 and vy == pytest.approx(0.1 * f) and wz == pytest.approx(0.6 * f) and abs(wz) <= 0.35 + 1e-9
    g.p.enabled = False
    assert g.govern(1.25, 0.3, 0.0, (0.4, 0.1, 0.6), None) == (0.4, 0.1, 0.6)


def test_params_from_node_dict():
    p = CrossingParams.from_params(dict(CROSSING_DEFAULTS, crossing_speed=0.3, crossing_zone="bad"))
    assert p.speed == 0.3 and p.zone == CrossingParams().zone and p.enabled is True
    assert set(CROSSING_DEFAULTS) == {"crossing_" + k for k in CrossingParams().__dataclass_fields__}


def test_plant_c_obs5_route_is_straightened():
    from dusan_common.layout import Layout
    from dusan_common.planning import plan
    from dusan_common.terrain_cost import TerrainCostCache
    lay = Layout.load(os.path.join(ROOT, "simulation", "config", "world_layout.yaml"))
    grid = OccupancyGridMap.load(os.path.join(ROOT, "simulation", "maps", "plant_map.yaml"))
    raw = plan(grid, (24.2, 0.5), (19.8, 0.3), 0.35, extra_cost=TerrainCostCache(lay)(grid))
    pipes = layout_pipes(lay)
    out, n = straighten_crossings(raw, pipes, lambda a, b: grid.line_of_sight(a[0], a[1], b[0], b[1], 0.245))
    assert n == 1 and debris_crossed(out, lay, types=("pipe",)) == ["C-OBS5"]
    c = find_crossings(out, pipes)[0]
    # no turn within the slab -0.9 .. 0.8 m around the pipe axis
    P = np.asarray(out)
    k = c.index
    assert c.progress(*P[k]) <= -0.9 + 1e-6 and c.progress(*P[k + 1]) >= 0.8 - 1e-6
    assert np.allclose((P[k + 1] - P[k]) / np.linalg.norm(P[k + 1] - P[k]), c.direction)


def test_debris_blocked_grid_marks_only_the_requested_kinds():
    grid = _free()
    lay = [PIPE, {"id": "R", "type": "rubble", "rect": [2.0, -1.0, 3.0, 1.0]},
           {"id": "PL", "type": "plate", "rect": [-3.5, -1.0, -2.5, 1.0]}]
    g = debris_blocked_grid(grid, lay, kinds=("rubble",))
    assert g is not grid and g.occ[g.world_to_grid(2.5, 0.0)[1], g.world_to_grid(2.5, 0.0)[0]] == 100
    assert g.occ[g.world_to_grid(2.5 + 0.08, 1.05)[1], g.world_to_grid(2.5, 1.05)[0]] == 100     # margin
    assert not g.is_occupied(0.0, 0.0) if hasattr(g, "is_occupied") else True
    ix, iy = g.world_to_grid(0.0, 0.0)
    assert g.occ[iy, ix] == 0                                                  # the pipe stays free
    ix, iy = g.world_to_grid(-3.0, 0.0)
    assert g.occ[iy, ix] == 0                                                  # so does the plate
    assert np.all(grid.occ == 0)                                               # input untouched
    g2 = debris_blocked_grid(grid, lay, kinds=("pipe", "rubble"))
    ix, iy = g2.world_to_grid(0.0, 0.0)
    assert g2.occ[iy, ix] == 100
    assert debris_blocked_grid(grid, lay, kinds=()) is grid and debris_blocked_grid(None, lay) is None


def test_debris_blocked_grid_keeps_passages_open():
    """A pipe across a 1.6 m corridor (the only way through) stays free with
    keep_passable, a rubble patch beside the route is blocked."""
    res = 0.1
    occ = np.zeros((30, 80), dtype=np.int8)               # 8 x 3 m, walls top / bottom
    occ[0:7, :] = 100
    occ[23:, :] = 100                                     # free band y in [0.7, 2.3) -> 1.6 m wide
    grid = OccupancyGridMap(occ, res, 0.0, 0.0)
    pipe = {"id": "P", "type": "pipe", "center": [4.0, 1.5], "yaw": math.pi / 2, "length": 1.6, "diameter": 0.1}
    rub = {"id": "R", "type": "rubble", "rect": [1.0, 1.6, 2.0, 2.2]}
    g = debris_blocked_grid(grid, [pipe, rub], kinds=("pipe", "rubble"), keep_passable=0.35)
    def occ_at(x, y):
        ix, iy = g.world_to_grid(x, y)
        return int(g.occ[iy, ix])
    assert occ_at(4.0, 1.5) == 0 and occ_at(1.5, 1.9) == 100
    g0 = debris_blocked_grid(grid, [pipe, rub], kinds=("pipe", "rubble"))           # without the check
    ix, iy = g0.world_to_grid(4.0, 1.5)
    assert g0.occ[iy, ix] == 100


def test_governor_recheck_drops_a_crossing_the_goal_no_longer_needs():
    """A goal that changes on the way (gas seeker steps): before the feet reach
    the pipe the crossing is given up when the path ahead no longer crosses
    it; once over it the robot walks on until clear, whatever the goal."""
    head = math.radians(-160.0)
    for recheck, kept in ((False, True), (True, False)):
        g = CrossingGovernor([PIPE], CrossingParams())
        g.govern(0.8, 0.0, head, (0.3, 0.0, 0.0), [(-1.5, 0.0)], recheck=recheck)
        assert g.state == "approach" and g.active is not None
        out = g.govern(0.8, 0.0, head, (0.0, 0.0, 0.0), [(1.5, 0.0)], recheck=recheck)    # goal now behind
        assert (g.active is not None) is kept
        if not kept:
            assert out == (0.0, 0.0, 0.0) and g.state == ""
    g = CrossingGovernor([PIPE], CrossingParams())
    g.govern(0.8, 0.0, head, (0.3, 0.0, 0.0), [(-1.5, 0.0)], recheck=True)
    vx, vy, wz = g.govern(0.2, 0.0, head, (0.0, 0.0, 0.0), None, recheck=True)            # dwell on the pipe?
    assert g.state == "cross" and vx == pytest.approx(0.25) and vy == 0.0 and wz == 0.0
