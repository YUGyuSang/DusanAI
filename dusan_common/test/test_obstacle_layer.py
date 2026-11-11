"""Unit tests of dusan_common.obstacle_layer / obstacle_sim (ROS free, synthetic)."""
import math
import os
import time

import numpy as np
import pytest
import yaml

from dusan_common.grid_map import OccupancyGridMap, submap
from dusan_common.obstacle_layer import (OBSTACLE_DEFAULTS, OBSTACLE_MEMORY_KEYS, OBSTACLE_NODE_KEYS, AvoidAction,
                                         DetourPlanner, EvidenceFrame, ObstacleAvoider, ObstacleMemory,
                                         ObstacleParams, SideCommitment, frame_accepted, nan_max_filter,
                                         nan_min_filter, organised_cloud_to_xyz, path_blockage, pursuit_path,
                                         speed_scale)
from dusan_common.obstacle_sim import Box, FakeElevationSensor, box_clearance, paint_boxes
from dusan_common.planning import astar, densify
from dusan_common.pure_pursuit import PurePursuit

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = 0.35


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def corridor(length=12.0, width=4.0, res=0.1, x0=0.0):
    """Free corridor (walls on the border cells), y from -width/2 .. width/2."""
    nx, ny = int(round(length / res)), int(round(width / res))
    occ = np.zeros((ny, nx), np.int8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 100
    return OccupancyGridMap(occ, res, x0, -width / 2.0)


def frame_from(pred, centre, half=3.0, res=0.04, known_pred=None, frame_id="odom"):
    """EvidenceFrame around ``centre`` with hit = pred(X, Y) (known everywhere
    unless ``known_pred``)."""
    n = int(round(2 * half / res))
    oi = int(math.floor(centre[0] / res + 0.5)) - n // 2
    oj = int(math.floor(centre[1] / res + 0.5)) - n // 2
    X, Y = np.meshgrid((oi + np.arange(n)) * res, (oj + np.arange(n)) * res)
    known = np.ones_like(X, dtype=bool) if known_pred is None else known_pred(X, Y)
    return EvidenceFrame(pred(X, Y) & known, known, oi * res, oj * res, res, frame_id)


def rect(cx, cy, hx, hy):
    return lambda X, Y: (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)


def nothing(X, Y):
    return np.zeros_like(X, dtype=bool)


def confirm(mem, pred, pose, n=2, t0=0.0, **kw):
    for k in range(n):
        mem.update(frame_from(pred, pose, **kw), pose, t0 + 0.2 * k)
    return t0 + 0.2 * n


def straight(x0, x1, y=0.0):
    return densify([(x0, y), (x1, y)], 0.1)


def cloud_bytes(xyz, big=False, pad=0):
    """Organised xyz -> PointCloud2-like object (12-byte float32 points + optional row padding)."""
    rows, cols = xyz.shape[:2]
    dt = ">f4" if big else "<f4"
    a = np.ascontiguousarray(xyz.astype(dt))
    rb = a.reshape(rows, cols * 12).view(np.uint8) if False else a.tobytes()
    if pad:
        body = np.frombuffer(rb, np.uint8).reshape(rows, cols * 12)
        rb = np.concatenate([body, np.zeros((rows, pad), np.uint8)], 1).tobytes()

    class F:
        def __init__(self, name, offset):
            self.name, self.offset, self.datatype, self.count = name, offset, 7, 1

    class M:
        pass
    m = M()
    m.height, m.width = rows, cols
    m.fields = [F("x", 0), F("y", 4), F("z", 8)]
    m.point_step, m.row_step = 12, cols * 12 + pad
    m.is_bigendian = big
    m.data = rb
    return m


# ----------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------
def test_defaults_keys_types_and_params_roundtrip():
    assert all(k.startswith("obstacle_") for k in OBSTACLE_DEFAULTS)
    assert set(OBSTACLE_NODE_KEYS) <= set(OBSTACLE_DEFAULTS) and set(OBSTACLE_MEMORY_KEYS) <= set(OBSTACLE_DEFAULTS)
    assert all(isinstance(OBSTACLE_DEFAULTS[k], str) for k in OBSTACLE_NODE_KEYS)
    assert all(not isinstance(v, str) for k, v in OBSTACLE_DEFAULTS.items() if k not in OBSTACLE_NODE_KEYS)
    assert OBSTACLE_DEFAULTS["obstacle_enabled"] is True and OBSTACLE_DEFAULTS["obstacle_source"] == "elevation"
    assert ObstacleParams.from_params(OBSTACLE_DEFAULTS) == ObstacleParams()
    for k, v in OBSTACLE_DEFAULTS.items():
        assert getattr(ObstacleParams(), k[len("obstacle_"):]) == v, k
        assert type(getattr(ObstacleParams(), k[len("obstacle_"):])) is type(v), k


def test_params_invalid_values_fall_back_and_clamps():
    p = ObstacleParams.from_params({"obstacle_margin": float("nan"), "obstacle_lookahead": "far",
                                    "obstacle_confirm_hits": 9, "obstacle_max_hits": 4,
                                    "obstacle_stop_dist": 2.0, "obstacle_slow_dist": 1.0,
                                    "obstacle_enabled": "false", "obstacle_source": 3, "obstacle_step_window": 4})
    assert p.margin == 0.10 and p.lookahead == 2.5
    assert p.max_hits == 4 and p.confirm_hits == 4                 # confirm <= max_hits
    assert p.stop_dist == p.slow_dist == 1.0                      # stop <= slow
    assert p.enabled is False and p.source == "elevation" and p.step_window == 5
    assert ObstacleParams.from_params({"obstacle_max_detours": 2.6}).max_detours == 3
    # static tolerance never as wide as a passage: < 2 r - res
    m = ObstacleMemory(ObstacleParams(static_tolerance=1.0), robot_radius=0.3)
    assert m.static_tolerance < 2 * 0.3 - 0.1


def test_step_parameters_match_the_terrain_mapper():
    with open(os.path.join(REPO, "module1_locomotion", "config", "terrain.yaml"), encoding="utf-8") as f:
        mp = yaml.safe_load(f)["terrain_mapper"]["ros__parameters"]["map"]
    assert OBSTACLE_DEFAULTS["obstacle_step_height"] == mp["step_crit"]
    assert OBSTACLE_DEFAULTS["obstacle_step_window"] == mp["step_window"]


def test_frame_accepted():
    csv = OBSTACLE_DEFAULTS["obstacle_frames"]
    for f in ("odom", "/odom", "map", "world"):
        assert frame_accepted(f, csv), f
    for f in ("base_link", "camera", "", None, "Odom"):
        assert not frame_accepted(f, csv), f
    assert frame_accepted("anything", "") and frame_accepted("", "*") and frame_accepted("x", " odom , x ")


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
def test_organised_cloud_roundtrip_endianness_padding_and_errors():
    rows, cols = 4, 5
    X, Y = np.meshgrid(np.arange(cols) * 0.04 + 1.0, np.arange(rows) * 0.04 - 2.0)
    Z = np.full((rows, cols), 0.1)
    Z[1, 2] = np.nan
    xyz = np.stack([X, Y, Z], -1)
    for big in (False, True):
        for pad in (0, 4):
            out = organised_cloud_to_xyz(cloud_bytes(xyz, big, pad))
            assert out.shape == (rows, cols, 3)
            assert np.allclose(out[..., :2], xyz[..., :2], atol=1e-6)
            assert np.isnan(out[1, 2, 2]) and np.allclose(np.delete(out[..., 2].ravel(), 7), 0.1, atol=1e-6)
    m = cloud_bytes(xyz)
    m.fields = m.fields[:2]
    with pytest.raises(ValueError):
        organised_cloud_to_xyz(m)
    m = cloud_bytes(xyz[:1])
    with pytest.raises(ValueError):
        organised_cloud_to_xyz(m)                                 # not organised
    m = cloud_bytes(xyz)
    m.data = m.data[:20]
    with pytest.raises(ValueError):
        organised_cloud_to_xyz(m)
    m = cloud_bytes(xyz)
    m.data = list(m.data)                                         # rclpy may hand over a list / array
    assert np.allclose(organised_cloud_to_xyz(m)[..., 0], X, atol=1e-6)


def test_nan_filters_ignore_unknown_cells():
    a = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan], [0.0, 2.0, np.nan]])
    mx, mn = nan_max_filter(a, 3), nan_min_filter(a, 3)
    assert mx[1, 1] == 3.0 and mn[1, 1] == 0.0 and mx[0, 0] == 1.0
    assert np.isnan(nan_max_filter(np.full((3, 3), np.nan), 3)).all()


def _terrain(kind):
    res = 0.04
    X, Y = np.meshgrid(np.arange(100) * res, np.arange(60) * res)
    Z = np.zeros_like(X)
    if kind == "box":
        Z[(np.abs(X - 2.0) <= 0.3) & (np.abs(Y - 1.2) <= 0.3)] = 0.5
    elif kind == "pipe":
        Z = np.where(np.abs(X - 2.0) < 0.05, 0.05 + np.sqrt(np.maximum(0.05 ** 2 - (X - 2.0) ** 2, 0)), 0.0)
    elif kind == "riser":
        Z[X >= 2.0] = 0.17
    elif kind == "ramp":
        Z = np.clip((X - 1.0) * math.tan(math.radians(11.8)), 0.0, None)
    return X, Y, Z


def test_evidence_from_elevation_step_test():
    X, Y, Z = _terrain("box")
    Z[0, 0] = np.nan
    f = EvidenceFrame.from_elevation(Z, 0.0, 0.0, 0.04, "odom", 0.22, 5)
    assert not f.known[0, 0] and f.known.sum() == Z.size - 1
    # a free cell clears evidence only where its whole 5 x 5 window is observed: not in the 3 x 3
    # corner block whose window holds the unknown cell
    assert not f.clear[2, 2] and f.clear[3, 3] and not (f.clear & f.hit).any()
    assert (f.known & ~f.hit & ~f.clear).sum() == 8
    hx, hy = X[f.hit], Y[f.hit]
    assert f.hit.any()
    # only the edge band (window 5 = +-2 cells) around the box
    d = np.maximum(np.abs(hx - 2.0) - 0.3, np.abs(hy - 1.2) - 0.3)
    assert d.max() <= 2 * 0.04 + 1e-9 and d.min() >= -2 * 0.04 - 1e-9
    assert not f.hit[(np.abs(X - 2.0) < 0.15) & (np.abs(Y - 1.2) < 0.15)].any()      # flat top: no step
    for kind in ("pipe", "riser", "ramp"):
        _, _, Z = _terrain(kind)
        assert not EvidenceFrame.from_elevation(Z, 0.0, 0.0, 0.04).hit.any(), kind


def test_evidence_from_cloud_recovers_geometry_and_traversability():
    X, Y, Z = _terrain("box")
    xyz = np.stack([X + 40.08, Y - 3.0, Z], -1).astype(np.float32)        # world offset, float32 rounding
    f = EvidenceFrame.from_elevation_cloud(xyz, "odom")
    assert f.res == pytest.approx(0.04) and f.x0 == pytest.approx(40.08, abs=1e-5)
    assert f.y0 == pytest.approx(-3.0, abs=1e-5)
    Xc, Yc = f.centres()
    assert np.allclose(Xc, X + 40.08, atol=1e-4)
    ref = EvidenceFrame.from_elevation(Z, 40.08, -3.0, 0.04)
    assert (f.hit == ref.hit).all()
    occ = np.full((10, 20), 20, np.int8)
    occ[3:5, 4:6] = 100
    occ[0, :] = -1
    g = OccupancyGridMap(occ, 0.1, 5.0, 1.0, "odom")
    ft = EvidenceFrame.from_traversability(g, 100)
    assert ft.hit.sum() == 4 and not ft.known[0].any() and ft.x0 == pytest.approx(5.05)
    assert EvidenceFrame.from_traversability(g, 15).hit.sum() == 180
    with pytest.raises(ValueError):
        EvidenceFrame.from_elevation_cloud(np.zeros((1, 5, 3)))


# ----------------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------------
def test_confirmation_needs_frames_and_revision_counts_output_changes():
    mem = ObstacleMemory(ObstacleParams(), R)
    pose = (0.0, 0.0, 0.0)
    box = rect(2.0, 0.0, 0.3, 0.3)
    rev0 = mem.revision
    assert mem.update(frame_from(box, pose), pose, 0.0) is False
    assert not mem.has_confirmed() and mem.raw.any() and mem.revision == rev0
    assert mem.update(frame_from(box, pose), pose, 0.2) is True
    assert mem.has_confirmed() and mem.revision == rev0 + 1
    c = mem.confirmed_xy()
    assert np.all(np.abs(c[:, 0] - 2.0) <= 0.35 + 0.1) and np.all(np.abs(c[:, 1]) <= 0.35 + 0.1)
    assert mem.update(frame_from(box, pose), pose, 0.4) is False and mem.revision == rev0 + 1
    assert mem.status()["confirmed"] == mem.confirmed_count > 30


def test_ghosts_are_never_confirmed():
    rng = np.random.default_rng(1)
    pose = (0.0, 0.0, 0.0)
    mem = ObstacleMemory(ObstacleParams(), R)
    for k in range(20):
        pts = rng.uniform(-2.5, 2.5, (50, 2))
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) > 0.7]

        def ghost(X, Y, pts=pts):
            m = np.zeros_like(X, dtype=bool)
            ix = np.clip(np.round((pts[:, 0] - X[0, 0]) / 0.04).astype(int), 0, X.shape[1] - 1)
            iy = np.clip(np.round((pts[:, 1] - Y[0, 0]) / 0.04).astype(int), 0, X.shape[0] - 1)
            m[iy, ix] = True
            return m
        mem.update(frame_from(ghost, pose), pose, 0.2 * k)
        mem.update(frame_from(nothing, pose), pose, 0.2 * k + 0.1)       # every ghost frame followed by a clean one
    assert not mem.has_confirmed()
    # alternating hit / free on the same cell never confirms
    mem2 = ObstacleMemory(ObstacleParams(), R)
    cell = rect(2.0, 0.0, 0.12, 0.12)
    for k in range(10):
        mem2.update(frame_from(cell if k % 2 == 0 else nothing, pose), pose, 0.2 * k)
    assert not mem2.has_confirmed()
    # an isolated confirmed cell is removed by the cluster filter
    mem3 = ObstacleMemory(ObstacleParams(), R)
    confirm(mem3, rect(2.04, 0.04, 0.015, 0.015), pose)
    assert mem3.confirmed.any() and not mem3.has_confirmed()
    mem4 = ObstacleMemory(ObstacleParams(min_cluster=1), R)
    confirm(mem4, rect(2.04, 0.04, 0.015, 0.015), pose)
    assert mem4.has_confirmed()


def test_clearing_unknown_and_forgetting():
    pose = (0.0, 0.0, 0.0)
    p = ObstacleParams()
    mem = ObstacleMemory(p, R)
    box = rect(2.0, 0.0, 0.3, 0.3)
    t = confirm(mem, box, pose, n=6)                          # saturated
    unknown = frame_from(nothing, pose, known_pred=lambda X, Y: np.abs(X - 2.0) > 1.0)
    for k in range(10):
        mem.update(unknown, pose, t + 0.2 * k)                # unknown cells never clear
    assert mem.has_confirmed()
    t += 2.0
    for k in range(p.max_hits):
        mem.update(frame_from(nothing, pose), pose, t + 0.2 * k)
    assert not mem.has_confirmed()
    # out of the sensor window: kept until forget_time, then forgotten
    mem = ObstacleMemory(ObstacleParams(forget_time=5.0), R)
    t = confirm(mem, box, pose)
    far = (8.0, 0.0, 0.0)
    for k in range(20):
        mem.update(frame_from(nothing, far), far, t + 0.2 * k)
    assert mem.has_confirmed()
    mem.update(frame_from(nothing, far), far, t + 6.0)
    assert not mem.has_confirmed()
    mem = ObstacleMemory(ObstacleParams(forget_time=0.0), R)
    t = confirm(mem, box, pose)
    mem.update(frame_from(nothing, far), far, t + 500.0)
    assert mem.has_confirmed()


def test_robot_footprint_is_ignored():
    pose = (0.0, 0.0, 0.0)
    mem = ObstacleMemory(ObstacleParams(), R)
    legs = rect(0.0, 0.0, 0.45, 0.3)
    confirm(mem, lambda X, Y: legs(X, Y) | rect(2.0, 0.0, 0.3, 0.3)(X, Y), pose)
    c = mem.confirmed_xy()
    assert len(c) and np.all(np.hypot(c[:, 0], c[:, 1]) >= 0.55 - 0.08)


def test_novelty_filter_against_the_static_map():
    g = corridor(12.0, 4.0)                                   # walls at y = +-1.95
    pose = (4.0, 0.0, 0.0)
    wall_band = lambda X, Y: (np.abs(Y) >= 1.9 - 0.08) & (np.abs(Y) < 2.0)   # step band 0.08 m into the floor
    mem = ObstacleMemory(ObstacleParams(), R)
    mem.set_static_map(g)
    confirm(mem, wall_band, pose, n=4)
    assert mem.confirmed.any() and not mem.has_confirmed()   # mapped structure: never output
    # a box 1 m from the wall is novel; a box touching the wall is novel except within the tolerance
    mem = ObstacleMemory(ObstacleParams(), R)
    mem.set_static_map(g)
    confirm(mem, lambda X, Y: wall_band(X, Y) | rect(6.0, 0.6, 0.3, 0.3)(X, Y), pose)
    c = mem.confirmed_xy()
    assert len(c) and np.all(np.abs(c[:, 1] - 0.6) <= 0.45)
    mem = ObstacleMemory(ObstacleParams(), R)
    mem.set_static_map(g)
    confirm(mem, rect(6.0, 1.6, 0.3, 0.35), pose)
    c = mem.confirmed_xy()
    assert len(c) and c[:, 1].max() <= 1.95 - 0.05 - mem.static_tolerance + 0.06
    # a row of boxes across the corridor (touching both walls) leaves no passable gap
    mem = ObstacleMemory(ObstacleParams(), R)
    mem.set_static_map(g)
    confirm(mem, rect(6.0, 0.0, 0.2, 2.0), pose)
    comb = mem.combined_grid(g)
    assert astar(comb, (2.0, 0.0), (10.0, 0.0), robot_radius=R) is None
    assert astar(g, (2.0, 0.0), (10.0, 0.0), robot_radius=R) is not None


def test_rolling_window_keeps_the_lattice_aligned():
    mem = ObstacleMemory(ObstacleParams(memory_size=10.0, forget_time=0.0), R)
    confirm(mem, rect(2.0, 0.0, 0.3, 0.3), (0.0, 0.0, 0.0))
    for k, x in enumerate(np.arange(0.0, 30.0, 0.5)):
        pose = (x, 0.3, 0.0)
        mem.update(frame_from(nothing, pose, known_pred=lambda X, Y: np.zeros_like(X, dtype=bool)), pose, 1.0 + k)
        if x < 6.0:
            assert mem.has_confirmed()
    assert not mem.has_confirmed()                             # dropped out of the window behind
    xs, ys = mem.lattice_centres()
    assert abs(30.0 - xs.mean()) < 4.0
    k = (xs - 0.05) / 0.1
    assert np.allclose(k, np.round(k), atol=1e-6)                 # centres on (i + 0.5) res


def test_clock_jump_clears_and_disabled_is_a_noop():
    pose = (0.0, 0.0, 0.0)
    mem = ObstacleMemory(ObstacleParams(), R)
    confirm(mem, rect(2.0, 0.0, 0.3, 0.3), pose, t0=100.0)
    assert mem.has_confirmed()
    mem.update(frame_from(nothing, pose, known_pred=lambda X, Y: np.zeros_like(X, dtype=bool)), pose, 5.0)
    assert not mem.has_confirmed()
    off = ObstacleMemory(ObstacleParams(enabled=False), R)
    assert confirm(off, rect(2.0, 0.0, 0.3, 0.3), pose) and not off.has_confirmed() and off.frames == 0
    av = ObstacleAvoider({"obstacle_enabled": False}, R)
    assert not av.update(frame_from(rect(2.0, 0.0, 0.3, 0.3), pose), pose, 0.0)
    assert av.step(pose, straight(0, 5), 0.0).kind == "none"


def test_combined_grid_geometry_caching_and_resolution():
    g = corridor(12.0, 4.0)
    pose = (4.0, 0.0, 0.0)
    mem = ObstacleMemory(ObstacleParams(), R)
    mem.set_static_map(g)
    confirm(mem, rect(6.0, 0.0, 0.3, 0.3), pose)
    out = mem.combined_grid(g, "output")
    assert out.occ.shape == g.occ.shape and out.origin_x == g.origin_x and out.resolution == g.resolution
    added = (out.occ == 100) & (g.occ != 100)
    c = mem.confirmed_xy()
    ix, iy = g.world_to_grid_array(c)
    assert added.sum() == len(c) and added[iy, ix].all()
    assert mem.combined_grid(g, "output") is out                    # cached
    plan = mem.combined_grid(g, "planning")
    assert ((plan.occ == 100) & ~(g.occ == 100)).sum() >= added.sum()
    confirm(mem, rect(9.0, 0.5, 0.3, 0.3), (7.0, 0.0, 0.0), t0=1.0)
    assert mem.combined_grid(g, "output") is not out
    # static map at 0.05 m: every lattice cell paints its 2 x 2 sub-cells
    g5 = OccupancyGridMap(np.zeros((80, 240), np.int8), 0.05, 0.0, -2.0)
    mem5 = ObstacleMemory(ObstacleParams(), R)
    mem5.set_static_map(g5)
    confirm(mem5, rect(6.0, 0.0, 0.3, 0.3), pose)
    assert (mem5.combined_grid(g5, "output").occ == 100).sum() == 4 * mem5.confirmed_count
    # map-free: a grid over the memory window
    free = ObstacleMemory(ObstacleParams(), R)
    confirm(free, rect(6.0, 0.0, 0.3, 0.3), pose)
    wg = free.combined_grid(None, "output")
    assert wg.width == wg.height == free.n and (wg.occ == 100).sum() == free.confirmed_count
    sub, (ix0, iy0) = submap(g, 1.0, -1.0, 3.0, 1.0)
    assert sub.width == 21 and sub.height == 21 and (ix0, iy0) == (10, 10)
    assert submap(g, 50.0, 50.0, 60.0, 60.0)[0] is None


def test_shadow_grows_only_into_unseen_cells():
    pose = (0.0, 0.0, 0.0)
    mem = ObstacleMemory(ObstacleParams(shadow_depth=0.3), R)
    face = rect(2.0, 0.0, 0.08, 0.3)
    behind = lambda X, Y: X <= 2.08                        # the camera sees up to the face only
    for k in range(3):
        mem.update(frame_from(face, pose, known_pred=behind), pose, 0.2 * k)
    out, pm = mem.output_mask(), mem.planning_mask()
    extra = pm & ~out
    assert extra.any()
    xs, ys = mem.lattice_centres()
    X, Y = np.meshgrid(xs, ys)
    assert X[extra].min() > 2.0 and X[extra].max() <= 2.1 + 0.3 + 1e-6      # behind the face, <= depth
    seen = ~np.isnan(mem.seen_free_t)
    assert not (extra & seen).any()


# ----------------------------------------------------------------------------
# path checks and detours
# ----------------------------------------------------------------------------
def boxed_memory(box_pred, pose=(3.0, 0.0, 0.0), grid=None, **kw):
    mem = ObstacleMemory(ObstacleParams(**kw), R)
    if grid is not None:
        mem.set_static_map(grid)
    confirm(mem, box_pred, pose)
    return mem


def test_path_blockage_free_travel():
    mem = boxed_memory(rect(6.0, 0.0, 0.3, 0.3))
    near = mem.confirmed_xy()[:, 0].min()
    b = path_blockage(mem, straight(3.0, 10.0), 0, R, 5.0)
    assert b is not None and b.s == pytest.approx(near - 3.0 - R, abs=0.12)
    assert path_blockage(mem, straight(3.0, 10.0, y=0.85), 0, R, 5.0) is None       # 0.5 m beside
    assert path_blockage(mem, straight(7.0, 12.0), 0, R, 5.0) is None                 # behind
    assert path_blockage(mem, straight(1.0, 10.0), 0, R, 2.0) is None                 # beyond the look-ahead


def detour_case(box, start=1.5, goal=11.0, grid=None, cost=None, commitment=None, t=0.0, seen_from=None, **kw):
    g = corridor(12.0, 4.0) if grid is None else grid
    mem = boxed_memory(box, pose=seen_from or (start + 2.0, 0.0, 0.0), grid=g, **kw)
    base = straight(start, goal)
    k = 0
    pl = DetourPlanner(mem.p, R)
    det, why = pl.plan(mem, base, k, (start + 0.0, 0.0), g, cost, commitment, t)
    return det, why, mem, g, base


def test_detour_in_a_corridor_keeps_clearance_and_rejoins():
    det, why, mem, g, base = detour_case(rect(4.5, 0.0, 0.3, 0.3))
    assert det is not None, why
    P = np.asarray(det.path)
    comb = mem.combined_grid(g, "planning")
    d = comb.distance_field()
    ix, iy = comb.world_to_grid_array(P)
    assert d[iy, ix].min() >= R + mem.p.margin - 0.07
    assert np.allclose(P[-1], base[-1]) and det.k_rejoin < len(base) - 1
    tail = np.asarray(base[det.k_rejoin:])
    assert np.allclose(P[-len(tail):], tail, atol=1e-6)
    box_far = mem.confirmed_xy()[:, 0].max()
    assert base[det.k_rejoin][0] >= box_far + mem.p.rejoin_gap - 0.05
    assert det.length <= mem.p.detour_max_ratio * det.replaced + 2.0
    assert np.all(np.diff(P[:, 0]) > -0.2)                         # no loop back


def test_side_choice_both_blocked_and_goal_blocked():
    det, why, *_ = detour_case(rect(4.5, -0.3, 0.3, 0.3))          # box right of the path -> left
    assert det is not None and det.side == "left", why
    det, why, *_ = detour_case(rect(4.5, 0.6, 0.3, 1.5))           # box against the left wall -> right
    assert det is not None and det.side == "right", why
    det, why, *_ = detour_case(rect(4.5, 0.0, 0.25, 2.0))          # wall across the corridor
    assert det is None and "blocked" in why
    det, why, *_ = detour_case(rect(10.8, 0.0, 0.3, 0.3), goal=10.8, start=8.0, seen_from=(9.0, 0.0, 0.0))
    assert det is None and why == "goal blocked"


def test_side_commitment_and_no_oscillation():
    g = corridor(12.0, 4.0)
    com = SideCommitment(15.0)
    # box slightly right: left is cheaper -> committed left
    det, _, mem, _, base = detour_case(rect(4.5, -0.1, 0.3, 0.3), grid=g, commitment=com, t=0.0)
    assert det.side == "left" and com.preferred(det.anchor, 1.0) == "left"
    # the same obstacle seen slightly left (right now shorter by < switch_ratio): stays left
    det2, *_ = detour_case(rect(4.5, 0.1, 0.3, 0.3), grid=g, commitment=com, t=1.0)
    assert det2.side == "left"
    # left becomes infeasible (more cells against the left wall): right
    det3, *_ = detour_case(rect(4.5, 0.6, 0.3, 1.5), grid=g, commitment=com, t=2.0)
    assert det3.side == "right"
    # after commit_time the cheaper side wins again
    com2 = SideCommitment(15.0)
    com2.commit("left", (4.2, 0.0), 0.0)
    det4, *_ = detour_case(rect(4.5, 0.25, 0.3, 0.3), grid=g, commitment=com2, t=20.0)
    assert det4.side == "right"
    # 20 jittered repetitions never flip the side
    rng = np.random.default_rng(3)
    com3 = SideCommitment(15.0)
    mem = boxed_memory(rect(4.5, 0.0, 0.3, 0.3), pose=(3.5, 0.0, 0.0), grid=g)
    pl = DetourPlanner(mem.p, R)
    sides = set()
    for k in range(20):
        start = (1.5 + rng.uniform(-0.1, 0.1), rng.uniform(-0.15, 0.15))
        det, _ = pl.plan(mem, straight(1.5, 11.0, y=start[1]), 0, start, g, None, com3, 0.3 * k)
        sides.add(det.side)
    assert len(sides) == 1


def test_detour_respects_the_static_terrain_cost():
    g = corridor(12.0, 4.0)
    cost = np.zeros(g.occ.shape)
    xs = g.origin_x + (np.arange(g.width) + 0.5) * 0.1
    ys = g.origin_y + (np.arange(g.height) + 0.5) * 0.1
    X, Y = np.meshgrid(xs, ys)
    cost[(Y > 0.3) & (np.abs(X - 4.5) < 0.8)] = 25.0               # a pipe field north of the box
    det, why, *_ = detour_case(rect(4.5, 0.0, 0.3, 0.3), grid=g, cost=cost)
    assert det is not None and det.side == "right", why
    det, why, *_ = detour_case(rect(4.5, 0.0, 0.3, 0.3), grid=g, cost=lambda: cost)     # lazy cost
    assert det is not None and det.side == "right", why


def test_speed_scale_governor():
    p = ObstacleParams()
    mem = boxed_memory(rect(6.0, 0.0, 0.3, 0.3))
    near = mem.confirmed_xy()[:, 0].min()
    for ft, want in ((0.6, (0.6 - 0.35) / 0.65), (0.30, 0.0), (1.5, 1.0)):
        x0 = near - R - ft
        sc, why, d = speed_scale(mem, straight(x0, 10.0), (x0, 0.0), R, p, 1.0)
        assert sc == pytest.approx(want, abs=0.12), (ft, sc, d)
    sc, *_ = speed_scale(mem, straight(1.0, 10.0, y=0.95), (1.0, 0.95), R, p, 1.0)
    assert sc == 1.0                                               # 0.6 m beside the path
    # unconfirmed (latest frame only) evidence: floor 0.5, stop when very close
    raw = ObstacleMemory(p, R)
    raw.update(frame_from(rect(6.0, 0.0, 0.3, 0.3), (3.0, 0.0, 0.0)), (3.0, 0.0, 0.0), 10.0)
    near_r = raw.raw_xy(10.0)[:, 0].min()
    sc, *_ = speed_scale(raw, straight(near_r - R - 0.5, 10.0), (near_r - R - 0.5, 0.0), R, p, 10.1)
    assert sc == pytest.approx(0.5, abs=0.01)                      # floor (ramp would give 0.23)
    sc, *_ = speed_scale(raw, straight(near_r - R - 0.8, 10.0), (near_r - R - 0.8, 0.0), R, p, 10.1)
    assert sc == pytest.approx((0.8 - 0.35) / 0.65, abs=0.1)       # above the floor: the ramp
    sc, *_ = speed_scale(raw, straight(near_r - R - 0.3, 10.0), (near_r - R - 0.3, 0.0), R, p, 10.1)
    assert sc == 0.0
    sc, *_ = speed_scale(raw, straight(near_r - R - 0.3, 10.0), (near_r - R - 0.3, 0.0), R, p, 11.5)
    assert sc == 1.0                                               # older than 1 s
    off = ObstacleParams(slowdown_enabled=False)
    assert speed_scale(mem, straight(near - R - 0.3, 10.0), (near - R - 0.3, 0.0), R, off, 1.0)[0] == 1.0


def test_avoider_step_transitions_counters_and_status():
    g = corridor(12.0, 4.0)
    av = ObstacleAvoider(OBSTACLE_DEFAULTS, R)
    av.set_static_map(g)
    route = straight(1.0, 11.0)
    av.new_goal()
    av.set_route(route)
    pose = (2.0, 0.0, 0.0)
    assert av.step(pose, route[10:], 0.0, g).kind == "none"
    t = confirm(av.memory, rect(4.5, 0.0, 0.3, 0.3), pose)
    act = av.step(pose, route[10:], t, g)
    assert act.kind == "detour" and act.side in ("left", "right") and av.detours == 1 and act.path
    assert av.status()["detour"] == act.side
    assert av.step(pose, act.path, t + 0.1, g).kind == "none"      # the detour itself is clear
    # a wall across the corridor: detour fails -> re-plan (until max_replans) -> hold
    av2 = ObstacleAvoider(OBSTACLE_DEFAULTS, R)
    av2.set_static_map(g)
    av2.set_route(route)
    t = confirm(av2.memory, rect(4.5, 0.0, 0.25, 2.0), pose)
    a = av2.step(pose, route[10:], t, g)
    assert a.kind == "replan" and av2.last.startswith("path blocked")
    # a failed detour is not retried before the memory / route changes
    calls = []
    orig = av2.planner.plan
    av2.planner.plan = lambda *a_, **k_: calls.append(1) or orig(*a_, **k_)
    for k in range(av2.p.max_replans):
        assert av2.step(pose, route[10:], t + 1 + k, g).kind == "replan"
        av2.note_replan()
    assert calls == [] and av2.replans_left == 0
    assert av2.step(pose, route[10:], t + 9, g).kind == "hold"
    av2.set_route(route)                                           # new route -> a new attempt
    av2.step(pose, route[10:], t + 10, g)
    assert len(calls) == len(av2._layers()) == 2                   # full shadow, then the near-shadow fallback
    av2.new_goal()
    assert av2.replans == 0 and av2.detours == 0 and av2.replans_left == av2.p.max_replans
    # min_interval between attempts / max_detours
    av3 = ObstacleAvoider({"obstacle_max_detours": 0}, R)
    av3.set_static_map(g)
    av3.set_route(route)
    t = confirm(av3.memory, rect(4.5, 0.0, 0.3, 0.3), pose)
    assert av3.step(pose, route[10:], t, g).kind == "replan"
    av4 = ObstacleAvoider({"obstacle_detour_enabled": False, "obstacle_replan_enabled": False}, R)
    av4.set_static_map(g)
    t = confirm(av4.memory, rect(4.5, 0.0, 0.3, 0.3), pose)
    assert av4.step(pose, route[10:], t, g).kind == "hold"
    # route_blocked / blocked_behind / planning_grid
    assert av.route_blocked(route, (1.0, 0.0)) and not av.route_blocked(straight(1.0, 11.0, y=1.2), (1.0, 1.2))
    assert av.blocked_behind((5.4, 0.0, 0.0), 0.6) and not av.blocked_behind((5.4, 0.0, math.pi), 0.6)
    assert av.planning_grid(g) is not None and av.planning_grid(None) is None
    assert ObstacleAvoider({}, R).planning_grid(g) is None          # nothing confirmed
    st = av.status()
    for k in ("enabled", "source", "frames", "rejected", "confirmed", "free_travel", "speed_scale", "detour",
              "detours", "replans", "holding", "last", "errors"):
        assert k in st
    import json
    json.dumps(st)


def test_pure_pursuit_remaining_path():
    pp = PurePursuit(lookahead=0.6)
    assert pp.remaining_path() is None
    pp.set_path(straight(0.0, 5.0))
    n0 = len(pp.remaining_path())
    pp.compute(2.0, 0.0, 0.0)
    rem = pp.remaining_path()
    assert len(rem) < n0 and rem[0][0] == pytest.approx(2.0, abs=0.1)
    pp.clear()
    assert pp.remaining_path() is None


def test_fake_sensor_detection_range_occlusion_and_removal():
    box = Box((5.0, 0.0), (0.6, 0.6))
    sen = FakeElevationSensor([box])
    f = sen.frame((2.5, 0.0, 0.0))                                # face 2.2 m away: seen, no step yet
    X, Y = f.centres()
    assert not f.hit.any() and f.known[np.abs(X - 4.6) < 0.05].any()
    f = sen.frame((3.6, 0.0, 0.0))                                # face 1.1 m away
    X, Y = f.centres()
    assert f.hit.any() and np.all(box.distance(X[f.hit], Y[f.hit]) <= 0.08 + 1e-9)
    assert not f.known[(np.abs(X - 5.25) < 0.04) & (np.abs(Y) < 0.1)].any()      # behind the face: occluded
    sen.boxes.clear()
    f = sen.frame((3.6, 0.0, 0.0))
    assert not f.hit.any()                                        # removed: seen free again
    assert box_clearance([box], 5.0, 1.0) == pytest.approx(0.7)
    g = paint_boxes(corridor(), [box])
    assert g.occ[g.world_to_grid(5.0, 0.0)[1], g.world_to_grid(5.0, 0.0)[0]] == 100


def test_performance_budget():
    mem = ObstacleMemory(ObstacleParams(), R)
    pose = (0.0, 0.0, 0.0)
    f = frame_from(rect(2.0, 0.0, 0.3, 0.3), pose)
    ts = []
    for k in range(20):
        t0 = time.perf_counter()
        mem.update(f, pose, 0.2 * k)
        ts.append(time.perf_counter() - t0)
    assert float(np.median(ts)) < 0.02
    t0 = time.perf_counter()
    det, why, *_ = detour_case(rect(4.5, 0.0, 0.3, 0.3))
    assert det is not None and time.perf_counter() - t0 < 0.5


# ----------------------------------------------------------------------------
# review regressions
# ----------------------------------------------------------------------------
def test_raw_layer_uses_the_speckle_filter_of_the_output():
    """F3: an isolated persistent evidence cell that the planners drop
    (min_cluster) must not stop the robot through the unconfirmed layer."""
    pose = (4.0, 0.0, 0.0)
    speck = rect(6.04, 0.04, 0.015, 0.015)                        # one 0.04 m cell = one lattice cell
    mem = boxed_memory(speck, pose=pose)
    for k in range(4):
        mem.update(frame_from(speck, pose), pose, 1.0 + 0.2 * k)
    assert mem.confirmed.any() and not mem.has_confirmed()       # cluster filter drops it ...
    assert len(mem.raw_xy(1.7)) == 0                             # ... and so does the governor's raw layer
    near = 6.04 - R - 0.2
    assert speed_scale(mem, straight(near, 10.0), (near, 0.0), R, mem.p, 1.7)[0] == 1.0
    one = ObstacleMemory(ObstacleParams(min_cluster=1), R)        # min_cluster 1: it counts everywhere
    one.update(frame_from(speck, pose), pose, 1.0)
    assert len(one.raw_xy(1.0)) == 1
    two = ObstacleMemory(ObstacleParams(), R)                     # a real (first-frame) obstacle is kept
    two.update(frame_from(rect(6.0, 0.0, 0.3, 0.3), pose), pose, 1.0)
    assert len(two.raw_xy(1.0)) > 10


def test_path_blockage_lets_a_robot_leave_the_clearance_zone():
    """F2 (b): a robot already closer than the check radius to a cell (a side
    confirmed while passing) may follow a path that does not get closer."""
    mem = boxed_memory(rect(5.0, 0.0, 0.8, 0.3), pose=(3.4, 1.2, 0.0))     # long box, north face y = 0.3
    y_face = mem.confirmed_xy()[:, 1].max()
    robot = (4.6, y_face + 0.36)
    away = densify([robot, (4.8, y_face + 0.9), (8.0, y_face + 0.9)], 0.1)
    along = densify([robot, (6.5, robot[1])], 0.1)
    closer = densify([robot, (5.0, y_face + 0.15)], 0.1)
    r_chk = R + 0.05
    assert path_blockage(mem, away, 0, r_chk, 2.5) is not None                # the old rule: blocked at 0 m
    assert path_blockage(mem, away, 0, r_chk, 2.5, robot_xy=robot) is None
    b = path_blockage(mem, along, 0, r_chk, 2.5, robot_xy=robot)   # staying in the zone: only LEAVE_LEN
    assert b is not None and 0.45 <= b.s <= 0.65
    b = path_blockage(mem, closer, 0, r_chk, 2.5, robot_xy=robot)
    assert b is not None and b.s < 0.2


def test_avoider_plans_one_detour_for_a_robot_inside_the_zone():
    """F2 (b): no new detour every min_interval while the robot is beside a late-confirmed side."""
    g = corridor(20.0, 4.0)
    box = Box((5.0, 0.0), (1.6, 0.6))
    for dy in (0.36, 0.40, 0.46):
        av = ObstacleAvoider({}, R)
        av.set_static_map(g)
        seen_from = (3.4, 1.2, 0.0)
        for k in range(3):
            av.update(frame_from(lambda X, Y: box.distance(X, Y) <= 0.08, seen_from), seen_from, 0.2 * k)
        base = densify([(0.0, 0.0), (19.0, 0.0)], 0.1)
        av.set_route(base)
        av._k = 46
        pose = (4.6, 0.3 + dy, 0.0)
        follow = np.asarray(base[46:])
        for k in range(30):                                      # 3 s standing still
            act = av.step(pose, follow, 1.0 + 0.1 * k, g)
            if act.kind == "detour":
                follow = act.path
        assert av.detours == 1, (dy, av.detours, av.last)


def test_detour_merges_an_obstacle_right_behind_the_first():
    """F2 (a): the rejoin point needs free route behind it, so a second box
    1.8 m behind the first is bypassed by the same detour (no rejoin between
    them that would leave the detour blocked and be re-planned forever)."""
    g = corridor(20.0, 4.0)
    A, C = Box((6.0, 0.0), (0.4, 0.6)), Box((8.6, 0.0), (0.4, 0.6))
    pred = lambda X, Y: (A.distance(X, Y) <= 0.08) | (C.distance(X, Y) <= 0.08)
    av = ObstacleAvoider({}, R)
    av.set_static_map(g)
    pose = (4.9, 0.0, 0.0)                                         # both faces within max_range
    t = confirm(av.memory, pred, pose, half=4.5)
    assert av.memory.confirmed_xy()[:, 0].max() > 8.0
    base = straight(1.0, 19.0)
    av.set_route(base)
    act = av.step(pose, base[39:], t, g)
    assert act.kind == "detour", av.last
    P = np.asarray(act.path)
    rejoin = av._rejoin_xy
    assert rejoin is not None and rejoin[0] > 8.8 + 0.4               # beyond C
    k_rej = int(np.argmin(np.hypot(P[:, 0] - rejoin[0], P[:, 1] - rejoin[1])))
    assert min(box_clearance([A, C], *q) for q in P[:k_rej + 1]) > R
    assert path_blockage(av.memory, act.path, 0, R + 0.05, av.p.lookahead, robot_xy=pose[:2]) is None
    for k in range(1, 20):                                         # the detour stays valid
        assert av.step(pose, act.path, t + 0.1 * k, g).kind == "none"
    assert av.detours == 1


def test_occlusion_shadow_along_the_line_of_sight():
    """F4: the unseen back of an obstacle is assumed obstacle_shadow_depth deep
    along the line of sight until a cell was seen free after the face was
    observed; stale free observations (from before the face was detected) do
    not count, the near layer keeps the short isotropic shadow."""
    pose = (0.0, 0.0, 0.0)
    p = ObstacleParams()
    assert p.shadow_depth == 1.0
    face = rect(2.0, 0.0, 0.04, 0.3)
    visible = lambda X, Y: X <= 2.04                               # behind the face: occluded
    mem = ObstacleMemory(p, R)
    early = (-1.0, 0.0, 0.0)                                       # from afar the face is not detected yet:
    mem.update(frame_from(nothing, early), early, 0.0)             # the whole area looked free
    for k in range(3):
        mem.update(frame_from(face, pose, known_pred=visible), pose, 1.0 + 0.2 * k)
    xs, ys = mem.lattice_centres()
    X, Y = np.meshgrid(xs, ys)
    out, near, plan_ = mem.output_mask(), mem.near_mask(), mem.planning_mask()
    los = plan_ & ~near
    assert los.any() and X[los].max() >= 2.0 + 0.9                 # ~1 m deep despite the stale 'free'
    assert np.abs(Y[los]).max() <= 0.3 + 0.1 + 0.05                # extruded, not fanned out sideways
    # seen free after the face evidence (looking round the corner): the shadow ends there
    mem2 = ObstacleMemory(p, R)
    for k in range(3):
        mem2.update(frame_from(face, pose, known_pred=visible), pose, 1.0 + 0.2 * k)
    side = (2.5, 1.5, -math.pi / 2)
    mem2.update(frame_from(nothing, side, known_pred=lambda X, Y: (X > 2.4) & (np.abs(Y) < 0.4)), side, 1.8)
    beh2 = mem2.planning_mask() & ~mem2.output_mask()
    assert X[beh2].max() < 2.4 + 0.05
    # shadow_depth <= 0.3: the near shadow only (legacy layer)
    short = ObstacleMemory(ObstacleParams(shadow_depth=0.3), R)
    for k in range(3):
        short.update(frame_from(face, pose, known_pred=visible), pose, 1.0 + 0.2 * k)
    assert (short.planning_mask() == short.near_mask()).all()
    assert X[short.near_mask() & ~short.output_mask()].max() <= 2.05 + 0.3 + 1e-6


def test_detour_falls_back_to_the_near_shadow_in_a_tight_passage():
    """A gap beside an obstacle that the full assumed depth closes is still
    used: the detour is planned on the near layer and validated on it."""
    g = corridor(20.0, 3.6)
    box = Box((7.0, 0.5), (0.6, 2.6))                              # 1.0 m gap at the south wall
    av = ObstacleAvoider({}, R)
    av.set_static_map(g)
    far = (4.5, -0.5, 0.0)                                         # from afar the face is not detected and
    av.update(frame_from(nothing, far, half=4.0), far, 0.0)        # the gap behind it was seen free
    pose = (5.6, 0.45, 0.0)
    known = lambda X, Y: X <= 6.7                                  # behind the face plane: occluded
    for k in range(3):
        av.update(frame_from(lambda X, Y: box.distance(X, Y) <= 0.08, pose, known_pred=known), pose,
                  0.2 + 0.2 * k)
    base = straight(1.0, 19.0)
    av.set_route(base)
    act = av.step(pose, base[46:], 1.0, g)
    assert act.kind == "detour" and act.side == "right" and av._head_layer == "near", av.last
    P = np.asarray(act.path)
    assert P[np.abs(P[:, 0] - 7.0) < 0.3, 1].max() < -1.0            # through the gap south of the box
    assert av.memory.distances(P[:60], "near").min() >= R + av.p.margin - 0.08
    # (its south face is unseen yet: the moving-sensor mission test checks the real clearance)


def test_side_choice_uses_the_terrain_cost():
    """Behaviour F2: the geometrically shorter side crosses costly debris, the
    clean side is chosen (sides compared with length + terrain integral)."""
    g = corridor(12.0, 4.0)
    xs = g.origin_x + (np.arange(g.width) + 0.5) * 0.1
    ys = g.origin_y + (np.arange(g.height) + 0.5) * 0.1
    X, Y = np.meshgrid(xs, ys)
    cost = np.zeros(g.occ.shape)
    cost[(Y > 0.35) & (np.abs(X - 4.5) < 0.6)] = 25.0               # rubble band north of the box, up to the wall
    box = rect(4.5, -0.45, 0.3, 0.3)                                # box south of the route: north is shorter
    det, why, *_ = detour_case(box)
    assert det is not None and det.side == "left", why             # without debris: the short side
    det, why, mem, g2, base = detour_case(box, cost=cost)
    assert det is not None and det.side == "right", why
    ix, iy = g.world_to_grid_array(np.asarray(det.path))
    assert cost[iy, ix].max() == 0.0


def test_frame_due_thins_frames_while_not_driving():
    """INT-2: one frame per idle_period while not driving, every frame while driving."""
    mem = ObstacleMemory(ObstacleParams(), R)
    acc = [mem.frame_due(10.0 + 0.2 * k, driving=False) for k in range(11)]
    assert acc == [True, False, False, False, False, True, False, False, False, False, True]
    assert all(mem.frame_due(20.0 + 0.2 * k, driving=True) for k in range(5))
    assert mem.frame_due(5.0, driving=False)                       # clock jumped back: accepted
    every = ObstacleMemory(ObstacleParams(idle_period=0.0), R)
    assert all(every.frame_due(1.0, driving=False) for _ in range(3))
    assert not ObstacleMemory(ObstacleParams(enabled=False), R).frame_due(1.0)


def test_astar_rejects_a_disconnected_goal_at_once():
    """INT-1: a failing search on a large map is answered by the connectivity
    check (labels cached per grid / radius), not by expanding all free cells."""
    g = OccupancyGridMap(np.zeros((200, 760), np.int8), 0.1, -2.0, -10.0)
    g.occ[0, :] = g.occ[-1, :] = g.occ[:, 0] = g.occ[:, -1] = 100
    g.occ[:, 400:404] = 100                                        # a wall across the whole map
    t0 = time.perf_counter()
    assert astar(g, (2.0, 0.0), (60.0, 0.0), robot_radius=R) is None
    assert time.perf_counter() - t0 < 0.2
    lab = g.free_labels(R)
    assert lab[100, 100] != lab[100, 700] and lab[100, 100] > 0 and lab[100, 402] == 0
    assert astar(g, (2.0, 0.0), (30.0, 5.0), robot_radius=R) is not None
    g.occ[:, 400:404] = 0                                          # opened: whoever edits occ resets the caches
    g._dist, g._inflated_cache = None, {}
    assert astar(g, (2.0, 0.0), (60.0, 0.0), robot_radius=R) is not None


def test_fake_sensor_field_of_view_starts_at_the_camera():
    """The depth camera is 0.34 m ahead of base_link (robot_model SENSORS):
    the ground beside the body is not observed (the side of an obstacle the
    robot walks past stays unseen); cam_offset=0 = the old, wider view."""
    pose = (5.0, 0.0, 0.0)
    real, wide = FakeElevationSensor([]), FakeElevationSensor([], cam_offset=0.0)
    assert real.cam_offset == pytest.approx(0.34)
    fr, fw = real.frame(pose), wide.frame(pose)
    X, Y = fr.centres()
    beside = (np.abs(X - 5.6) < 0.05) & (np.abs(Y - 0.55) < 0.05)
    assert fw.known[beside].any() and not fr.known[beside].any()
    ahead = (np.abs(X - 6.5) < 0.05) & (np.abs(Y) < 0.05)
    assert fr.known[ahead].all()


def test_pursuit_path_is_the_line_the_follower_drives():
    path = straight(0.0, 5.0)
    P = pursuit_path(path, (1.0, 0.3), 0.6)
    assert tuple(P[0]) == (1.0, 0.3)
    assert P[1] == pytest.approx((1.6, 0.0))           # first vertex >= 0.6 m away, searched from the closest one
    assert tuple(P[-1]) == pytest.approx((5.0, 0.0))
    P0 = pursuit_path(path, (1.0, 0.3), 0.0)
    assert P0[1] == pytest.approx((1.0, 0.0))          # lead 0: robot -> closest vertex -> rest
    assert len(pursuit_path(path, (4.8, 0.1), 0.6)) == 2   # near the end: robot -> goal
    assert len(pursuit_path([], (0.0, 0.0), 0.6)) == 0


def test_governor_measures_along_the_pursuit_chord_and_tightens_tracking():
    """Gazebo w: the follower drives robot -> look-ahead point, not the path
    from its first vertex.  At an L corner beside an obstacle the full chord
    (0.6 m) cuts towards it although the path keeps clear: the follower is
    asked for half the look-ahead instead of being stopped; straight at a
    wall no look-ahead helps -> stop."""
    p = ObstacleParams()
    av = ObstacleAvoider(p, R)
    confirm(av.memory, rect(4.4, 0.6, 0.15, 0.15), (3.0, 0.0, 0.0))
    path = densify([(0.0, 0.0), (5.0, 0.0), (5.0, 3.0)], 0.1)
    rest = path[46:]                                               # robot at (4.6, 0) on the path
    assert speed_scale(av.memory, rest, (4.6, 0.0), R, p, 1.0)[0] == 1.0
    assert speed_scale(av.memory, pursuit_path(rest, (4.6, 0.0), 0.6), (4.6, 0.0), R, p, 1.0)[0] == 0.0
    act = av.step((4.6, 0.0, 0.0), rest, 1.0, lead=0.6)
    assert act.kind == "none" and act.speed_scale == 1.0 and act.lead == pytest.approx(0.3)
    assert av.step((4.6, 0.0, 0.0), rest, 1.1).lead is None        # no lead: the path itself (legacy)
    wall = ObstacleAvoider(ObstacleParams(detour_enabled=False, replan_enabled=False), R)    # held in front
    confirm(wall.memory, rect(6.0, 0.0, 0.1, 1.5), (4.0, 0.0, 0.0))
    near = wall.memory.confirmed_xy()[:, 0].min()
    x = near - R - 0.25
    act = wall.step((x, 0.0, 0.0), straight(x, 10.0), 1.0, lead=0.6)
    assert act.kind == "hold" and act.speed_scale == 0.0 and act.lead == pytest.approx(0.6)


def test_near_shadow_route_only_when_much_cheaper():
    av = ObstacleAvoider(ObstacleParams(), R)
    assert av.near_is_cheaper(10.0, 6.0) and not av.near_is_cheaper(10.0, 7.0)      # switch_ratio 1.5
    assert not av.near_is_cheaper(10.0, math.inf)
    g = corridor()
    assert av.route_cost(g, None, straight(1.0, 4.0)) == pytest.approx(3.0)
    cost = np.zeros(g.occ.shape)
    cost[:, 25:30] = 10.0                                          # x 2.5 .. 3.0
    assert av.route_cost(g, cost, straight(1.0, 4.0)) == pytest.approx(3.0 + 10.0 * 0.5, abs=0.3)
    assert av.route_cost(g, cost, [(1.0, 0.0)]) == math.inf


def test_detour_takes_the_near_shadow_when_the_full_one_forces_debris():
    """The full occlusion shadow (1.0 m behind a face seen from afar) pushes
    the detour across a costly band that the detour past the near shadow
    avoids: the near one is taken (switch_ratio cheaper); without the
    comparison the full-shadow detour crosses the band."""
    g = corridor(12.0, 4.0)
    X, Y = np.meshgrid(g.origin_x + (np.arange(g.width) + 0.5) * 0.1, g.origin_y + (np.arange(g.height) + 0.5) * 0.1)
    cost = np.where((X >= 5.2) & (X <= 6.0) & (np.abs(Y) >= 0.5), 4.0, 0.0)
    base = straight(1.5, 11.0)
    pose = (2.6, 0.0, 0.0)

    def run(compare):
        av = ObstacleAvoider(ObstacleParams(), R)
        if not compare:
            av.near_is_cheaper = lambda a, b: False
        av.set_static_map(g)
        av.set_route(base)
        for k in range(3):
            av.update(frame_from(rect(4.5, 0.0, 0.3, 0.3), pose, known_pred=lambda X, Y: X < 4.3), pose, 0.2 * k)
        act = av.step(pose, base[11:], 1.0, g, cost, lead=0.6)
        P = np.asarray(act.path)
        return act, av, int(np.sum((P[:, 0] >= 5.2) & (P[:, 0] <= 6.0) & (np.abs(P[:, 1]) >= 0.5)))

    act, av, inband = run(True)
    assert act.kind == "detour" and av._head_layer == "near" and inband == 0
    act, av, inband = run(False)
    assert act.kind == "detour" and av._head_layer == "planning" and inband > 0


def test_follower_tracks_tightly_where_its_chord_would_cut_across_debris():
    """Gazebo s (mpc): the route turns sharply round a pipe end; the 0.6 m
    chord cut across the pipe (the robot fell on it).  Where the chord
    crosses clearly more terrain cost than the path it cuts, the follower
    gets half the look-ahead; away from debris the look-ahead is unchanged."""
    g = corridor()
    av = ObstacleAvoider(ObstacleParams(), R)
    confirm(av.memory, rect(10.5, 1.5, 0.1, 0.1), (8.0, 0.0, 0.0))  # a detection far away (the layer is active)
    cost = np.zeros(g.occ.shape)
    cost[18:21, 45:48] = 25.0                                      # 'pipe end' x 4.5..4.8, y -0.2..0.1
    path = densify([(0.0, 0.3), (4.95, 0.3), (4.95, 1.5)], 0.1)    # turns left just past it
    rest = path[45:]                                               # robot at (4.5, 0.3)
    assert ObstacleAvoider._cuts_debris(rest, (4.5, 0.3), 0.6, g, cost) is False   # chord stays north of it
    path2 = densify([(0.0, 0.3), (4.95, 0.3), (4.95, -1.5)], 0.1)  # turns right across the corner
    rest2 = path2[45:]
    assert ObstacleAvoider._cuts_debris(rest2, (4.5, 0.3), 0.6, g, cost)
    act = av.step((4.5, 0.3, 0.0), rest2, 1.0, g, cost, lead=0.6)
    assert act.kind == "none" and act.lead == pytest.approx(0.3)
    assert av.step((4.5, 0.3, 0.0), rest2, 1.1, g, None, lead=0.6).lead == pytest.approx(0.6)   # no cost layer
    assert av.step((4.5, 0.3, 0.0), rest, 1.2, g, cost, lead=0.6).lead == pytest.approx(0.6)


def test_unobserved_continuation_of_a_face_is_assumed():
    """Gazebo w: walking along a wall at 0.5 .. 0.7 m the camera never sees its
    continuation; the part seen ended at y = 0 and the detour rounded that
    'end' into the wall.  The planning and near layers extend a face along
    itself into cells not seen free (shadow_depth); a face whose end has the
    ground beside it seen free is not extended."""
    wall = rect(5.05, 0.5, 0.05, 0.5)                              # face x 5.0 .. 5.1, seen y 0 .. 1

    def layers(end_seen):
        mem = ObstacleMemory(ObstacleParams(), R)
        # known up to the face (x < 5.1) and, unless end_seen, only along the part seen (y > 0)
        known = (lambda X, Y: (X < 5.1) & (Y > 0.01)) if not end_seen else (lambda X, Y: X < 5.1)
        confirm(mem, wall, (3.8, 0.5, 0.0), known_pred=known)
        return mem.layer_cells("planning"), mem.layer_cells("near")

    for cells in layers(end_seen=False):
        on_line = cells[np.abs(cells[:, 0] - 5.05) < 0.11]
        assert on_line[:, 1].min() < -0.85                         # ~1 m past the end seen
    for cells in layers(end_seen=True):
        on_line = cells[np.abs(cells[:, 0] - 5.05) < 0.11]
        assert on_line[:, 1].min() > -0.35                         # the near shadow (0.3 m) only


def test_floor_in_front_of_an_unobserved_face_is_not_free():
    """Gazebo w: the camera saw the floor in front of a tall face but not the
    face (NaN); counted as free it cleared the confirmed face.  Floor next to
    unobserved cells is known (seen free for the shadow) but clears nothing;
    a confirmed face survives such frames and is still cleared by a fully
    observed free view."""
    n, res = 100, 0.04
    Z = np.zeros((n, n))
    Z[:, 60:70] = 0.6                                              # a wall x 2.4 .. 2.8
    seen_face = EvidenceFrame.from_elevation(Z, 0.0, 0.0, res)
    Zb = Z.copy()
    Zb[:, 60:] = np.nan                                            # later: only the floor in front observed
    floor_only = EvidenceFrame.from_elevation(Zb, 0.0, 0.0, res)
    assert seen_face.hit[:, 58].all() and not floor_only.hit.any()
    assert floor_only.known[:, 58].all() and not floor_only.clear[:, 58].any() and floor_only.clear[:, 50].all()
    mem = ObstacleMemory(ObstacleParams(), R)
    pose = (1.0, 2.0, 0.0)
    for k in range(3):
        mem.update(seen_face, pose, 0.2 * k)
    n0 = mem.confirmed_count
    assert n0 > 0
    for k in range(10):
        mem.update(floor_only, pose, 1.0 + 0.2 * k)
    assert mem.confirmed_count == n0                                # not cleared by the half-observed view
    for k in range(10):
        mem.update(EvidenceFrame.from_elevation(np.zeros((n, n)), 0.0, 0.0, res), pose, 3.0 + 0.2 * k)
    assert mem.confirmed_count == 0                                 # the wall removed and seen: cleared
