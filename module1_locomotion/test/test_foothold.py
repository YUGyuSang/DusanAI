"""Unit tests: foothold planner + swing apex helper."""
import math

import numpy as np
import pytest

from module1_locomotion.elevation_map import ElevationMap
from module1_locomotion.foothold_planner import FootholdPlanner, max_height_along

RES = 0.04
EDGES = [0.6 + 0.3 * k for k in range(5)]          # riser positions of the test staircase


def grid_map(f, center=(0.0, 0.0)):
    m = ElevationMap(center=center)
    X, Y = m.grid_xy()
    return ElevationMap.from_elevation(f(X, Y), m.origin, m.resolution)


def stairs_fn(X, Y):
    """5 steps, 0.17 m rise, 0.30 m tread, first riser at x = 0.6 (cell boundary 0.58)."""
    return np.clip(np.floor((X - 0.6) / 0.3) + 1, 0, 5) * 0.17


def edge_distance(x):
    """Distance from x to the nearest riser (risers sit on the 0.58 + 0.30 k cell boundaries)."""
    return min(abs(x - (e - RES / 2)) for e in EDGES)


def test_flat_ground_returns_exact_nominal():
    m = grid_map(lambda X, Y: 0.0 * X + 0.05)
    fp = FootholdPlanner()
    for nom in [(0.123, -0.071), (1.0, 1.0), (-0.333, 0.25)]:
        p, score, valid = fp.select(m, nom, fallback_z=0.0)
        assert valid
        assert p[0] == nom[0] and p[1] == nom[1]       # no quantisation jitter
        assert p[2] == pytest.approx(0.05)
        assert score == pytest.approx(0.0, abs=1e-9)


def test_footholds_move_away_from_stair_edges():
    m = grid_map(stairs_fn)
    fp = FootholdPlanner(search_radius=0.10)
    moved = 0
    near = 0
    for nx in np.arange(0.40, 1.50, 0.01):
        p, score, valid = fp.select(m, (nx, 0.13), fallback_z=0.0)
        assert valid, nx
        assert abs(math.hypot(p[0] - nx, p[1] - 0.13)) <= 0.10 * fp.expand_factor + 1e-9
        assert edge_distance(p[0]) >= 0.06 - 1e-9, (nx, p)       # foot radius 0.025 + margin
        assert p[2] == pytest.approx(stairs_fn(np.array([p[0]]), None)[0])
        if edge_distance(nx) < 0.05:
            near += 1
            moved += abs(p[0] - nx) > 1e-9
    assert near > 10 and moved == near


def test_obstacle_cells_rejected_and_score_ordering():
    box = lambda X, Y: np.where((np.abs(X - 1.0) < 0.15) & (np.abs(Y) < 0.15), 0.5, 0.0)
    m = grid_map(box)
    fp = FootholdPlanner(search_radius=0.10)
    # nominal on top of the box: the flat top is locally perfect but unreachable
    # (relative height test) and its rim is an edge -> no valid foothold nearby
    p, s, valid = fp.select(m, (1.0, 0.0), 0.0)
    assert not valid and math.isinf(s)
    # with the reachability test disabled the box top would be accepted
    loose = FootholdPlanner(search_radius=0.10, max_relative_height=math.inf)
    assert loose.select(m, (1.0, 0.0), 0.0)[2]
    # a single stair riser (0.17 m) remains reachable
    st = grid_map(lambda X, Y: np.where(X >= 0.6, 0.17, 0.0))
    p4, _, v4 = fp.select(st, (0.8, 0.0), 0.0)
    assert v4 and p4[2] == pytest.approx(0.17)
    p2, s2, v2 = fp.select(m, (1.22, 0.0), 0.0)         # nominal next to the box
    assert v2 and p2[0] > 1.22 - 1e-9 and p2[2] == pytest.approx(0.0)
    p3, s3, v3 = fp.select(m, (2.0, 0.0), 0.0)          # free ground
    assert v3 and s3 == pytest.approx(0.0) and s2 > s3


def test_narrow_raised_features_and_slopes():
    # a lying pipe / beam 0.12 m wide, 0.10 m high: its flat-looking top is rejected
    bar = grid_map(lambda X, Y: np.where(np.abs(X - 1.0) <= 0.06, 0.10, 0.0))
    fp = FootholdPlanner(search_radius=0.10)
    for nx in (0.96, 1.0, 1.04):
        p, s, v = fp.select(bar, (nx, 0.0), 0.0)
        assert v and p[2] == pytest.approx(0.0) and abs(p[0] - 1.0) >= 0.10 - 1e-9, (nx, p)
    # but a wide plateau (platform 0.10 m) is a fine foothold away from its rim
    plat = grid_map(lambda X, Y: np.where(X >= 0.6, 0.10, 0.0))
    p, s, v = fp.select(plat, (0.9, 0.0), 0.0)
    assert v and p[0] == 0.9 and p[2] == pytest.approx(0.10)
    # a clean 16 deg ramp keeps the exact nominal foothold
    g = math.tan(math.radians(16.0))
    ramp = grid_map(lambda X, Y: g * X)
    p, s, v = fp.select(ramp, (0.33, -0.21), 0.0)
    assert v and (p[0], p[1]) == (0.33, -0.21) and p[2] == pytest.approx(g * 0.33, abs=1e-6)


def test_rough_terrain_prefers_flat_patch():
    rng = np.random.default_rng(4)
    H = np.zeros((150, 150))
    for _ in range(400):
        r, c = rng.integers(0, 148, 2)
        H[r:r + 2, c:c + 2] = rng.uniform(0.02, 0.07)
    m = ElevationMap.from_elevation(H, (-3.0, -3.0), RES)
    fp = FootholdPlanner(search_radius=0.10)
    costs_nom, costs_sel = [], []
    for k in range(40):
        nom = rng.uniform(-2.0, 2.0, 2)
        p, s, v = fp.select(m, nom, 0.0)
        if not v:
            continue
        r, c = m.world_to_index(p[0], p[1])
        costs_sel.append(m.layer("step")[r, c])
        rn, cn = m.world_to_index(nom[0], nom[1])
        costs_nom.append(m.layer("step")[rn, cn])
    assert len(costs_sel) > 30
    assert np.mean(costs_sel) < 0.5 * np.mean(costs_nom) + 1e-9


def test_unknown_terrain_and_no_map():
    m = ElevationMap(center=(0.0, 0.0))                 # completely unknown
    fp = FootholdPlanner()
    p, s, v = fp.select(m, (0.3, 0.1), fallback_z=-0.88)
    assert not v and math.isinf(s)
    assert np.allclose(p, [0.3, 0.1, -0.88])
    p, s, v = fp.select(None, (0.3, 0.1, 0.0), fallback_z=0.2)
    assert not v and np.allclose(p, [0.3, 0.1, 0.2])
    # allow_unknown: unknown cells are acceptable with a penalty, but still flagged
    fpu = FootholdPlanner(allow_unknown=True, unknown_penalty=2.0)
    p, s, v = fpu.select(m, (0.3, 0.1), fallback_z=-0.88)
    assert not v and s == pytest.approx(2.0) and p[2] == pytest.approx(-0.88)
    # outside the map
    p, s, v = fp.select(grid_map(lambda X, Y: 0 * X), (50.0, 0.0), 0.1)
    assert not v and p[2] == pytest.approx(0.1)


def test_partial_knowledge_prefers_known_cells():
    e = np.zeros((150, 150))
    e[:, 76:] = np.nan                                   # right half unknown
    m = ElevationMap.from_elevation(e, (-3.0, -3.0), RES, hole_fill_radius=0)
    fp = FootholdPlanner(search_radius=0.12)
    p, s, v = fp.select(m, (0.08, 0.0), 0.3)
    assert v and p[0] <= 0.0 + 1e-9 and p[2] == pytest.approx(0.0)


def test_weights_kwargs_and_aliases():
    fp = FootholdPlanner(0.15, w_roughness=5.0, w_displacement=2.0, w_unknown=3.0, foo=1)
    assert fp.search_radius == 0.15 and fp.w_rough == 5.0 and fp.w_dist == 2.0
    assert fp.unknown_penalty == 3.0 and fp.ignored == {"foo": 1}
    pts, scores, valid = fp.select_many(grid_map(lambda X, Y: 0 * X), [[0, 0], [0.5, 0.2]], 0.0)
    assert pts.shape == (2, 3) and valid.all()


def test_max_height_along():
    box = lambda X, Y: np.where((np.abs(X - 1.0) < 0.1) & (np.abs(Y) < 0.5), 0.25, 0.0)
    m = grid_map(box)
    assert max_height_along(m, (0.5, 0.0), (1.5, 0.0)) == pytest.approx(0.25)
    assert max_height_along(m, (0.2, 0.0), (0.6, 0.0)) == pytest.approx(0.0)
    # a segment passing just beside the box sees it through the lateral margin
    assert max_height_along(m, (0.5, 0.52), (1.5, 0.52)) == pytest.approx(0.25)
    assert max_height_along(m, (0.5, 0.52), (1.5, 0.52), margin=0.0) == pytest.approx(0.0)
    assert max_height_along(m, (1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.25)   # degenerate
    unknown = ElevationMap(center=(0.0, 0.0))
    assert math.isnan(max_height_along(unknown, (0.0, 0.0), (1.0, 0.0)))
    assert math.isnan(max_height_along(None, (0.0, 0.0), (1.0, 0.0)))
    assert math.isnan(max_height_along(m, (40.0, 0.0), (41.0, 0.0)))
