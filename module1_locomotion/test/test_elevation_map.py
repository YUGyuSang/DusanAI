"""Unit tests: elevation map (fusion, layers, transport) + point-cloud utilities."""
import math
import os
import struct

import numpy as np
import pytest
import yaml

from dusan_common.camera import camera_world_pose
from dusan_common.rclpy_logger_rules import CallSiteRules
from dusan_common.geometry import pose_matrix

from module1_locomotion.elevation_map import (ElevationMap, ElevationMapParams, binomial_weights,
                                              correlate_1d, fill_holes, max_filter, min_filter,
                                              nanmedian_rows, plane_fit, shifted)
from module1_locomotion.pointcloud_utils import (
    FLOAT32, FLOAT64, UINT8, ConventionDetector, DepthIntrinsics, PointFilter, PoseBuffer,
    camera_points_to_world, depth_image_to_points, depth_to_metres, detect_convention,
    pack_xyz_organised, pointcloud2_to_xyz, unpack_fields, world_to_camera_points)

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = 0.04


# ----------------------------------------------------------------- helpers
class FakeField:
    def __init__(self, name, offset, datatype, count=1):
        self.name, self.offset, self.datatype, self.count = name, offset, datatype, count


class FakeCloud:
    """Duck-typed sensor_msgs/PointCloud2."""

    def __init__(self, data, fields, width, height, point_step, row_step, is_bigendian=False):
        self.data, self.fields = data, fields
        self.width, self.height = width, height
        self.point_step, self.row_step = point_step, row_step
        self.is_bigendian = is_bigendian


def grid_map(f, center=(0.0, 0.0), **kw):
    """Map whose elevation is the analytic height function f(X, Y)."""
    m = ElevationMap(center=center, **kw)
    X, Y = m.grid_xy()
    return ElevationMap.from_elevation(f(X, Y), m.origin, m.resolution)


def simulate_depth(height_fn, robot_pose, intr, far=4.0, step=0.01):
    """Ray-march a height field from the depth camera; returns a depth image
    (H, W) in metres (NaN = no return).  The link-frame ray of pixel (u, v)
    is (1, a, b) so the ray parameter IS the depth."""
    T = camera_world_pose(robot_pose, "depth_camera")
    a, b = intr.ray_grid()
    dirs = np.stack([np.ones_like(a), a, b], -1).reshape(-1, 3) @ T[:3, :3].T
    o = T[:3, 3]
    n = dirs.shape[0]
    depth = np.full(n, np.nan)
    prev = np.full(n, 0.05)
    active = np.ones(n, dtype=bool)
    for t in np.arange(0.1, far, step):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        p = o + t * dirs[idx]
        hit = p[:, 2] <= height_fn(p[:, 0], p[:, 1])
        if hit.any():
            hi = idx[hit]
            lo_t, hi_t = prev[hi], np.full(hi.size, t)
            for _ in range(10):        # bisection refinement
                mid = 0.5 * (lo_t + hi_t)
                pm = o + mid[:, None] * dirs[hi]
                below = pm[:, 2] <= height_fn(pm[:, 0], pm[:, 1])
                hi_t = np.where(below, mid, hi_t)
                lo_t = np.where(below, lo_t, mid)
            depth[hi] = hi_t
            active[hi] = False
        prev[idx[~hit]] = t
    return depth.reshape(intr.height, intr.width)


def integrate_scene(emap, height_fn, poses, intr, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    for k, pose in enumerate(poses):
        d = simulate_depth(height_fn, pose, intr)
        if noise > 0:
            d = d + rng.normal(0.0, noise, d.shape) * (1 + d ** 2)
        pts = depth_image_to_points(d, intr, max_depth=3.5)
        pw, rr, origin = camera_points_to_world(pts, pose, "link")
        emap.recenter(pose[0], pose[1])
        emap.integrate(pw, origin, t=0.1 * k, ranges=rr)
    return emap


# ------------------------------------------------------------- map basics
def test_default_geometry_and_indexing():
    m = ElevationMap(center=(61.91, 0.0))
    assert m.shape == (150, 150)
    assert m.resolution == pytest.approx(0.04)
    x0, y0 = m.origin
    # cell centres on a global lattice (multiples of the resolution)
    assert abs(x0 / RES - round(x0 / RES)) < 1e-6 and abs(y0 / RES - round(y0 / RES)) < 1e-6
    r, c = m.world_to_index(61.91, 0.0)
    assert (int(r), int(c)) == (75, 75)
    cx, cy = m.center
    assert abs(cx - 61.91) <= RES / 2 + 1e-9 and abs(cy) <= RES / 2 + 1e-9
    assert np.isnan(m.height_at(61.91, 0.0))


def test_neighbourhood_helpers():
    a = np.arange(12, dtype=float).reshape(3, 4)
    b = shifted(a, 1, 0, -1.0)
    assert b[0, 0] == a[1, 0] and b[2, 0] == -1.0
    b = shifted(a, 0, -1, 0.0)
    assert b[0, 1] == a[0, 0] and b[0, 0] == 0.0
    z = np.zeros((7, 7))
    z[3, 3] = 1.0
    z[0, 0] = np.nan
    assert max_filter(z, 3)[2, 2] == 1.0 and max_filter(z, 3)[5, 5] == 0.0
    assert min_filter(np.where(z == 1, 5.0, 2.0), 3)[3, 3] == 2.0


def test_plane_fit_is_sobel_on_full_window():
    X, Y = np.meshgrid(np.arange(9.0), np.arange(9.0))
    z = 0.3 * X - 0.2 * Y + 1.0
    fit = plane_fit(z, 3, np.array([1.0, 2.0, 1.0]))
    assert np.allclose(fit["gx"][1:-1, 1:-1], 0.3)
    assert np.allclose(fit["gy"][1:-1, 1:-1], -0.2)
    assert np.allclose(fit["resid"][1:-1, 1:-1], 0.0, atol=1e-6)
    # NaN-aware: a missing neighbour does not bias the gradient of a plane
    z2 = z.copy()
    z2[4, 5] = np.nan
    fit2 = plane_fit(z2, 3, np.array([1.0, 2.0, 1.0]))
    assert fit2["gx"][4, 4] == pytest.approx(0.3, abs=1e-9)


def _plane_fit_bruteforce(z, window, w1, r, c):
    """Reference: explicit weighted least squares at one cell."""
    rad = window // 2
    rows, cols = z.shape
    A, b, w = [], [], []
    for dv in range(-rad, rad + 1):
        for du in range(-rad, rad + 1):
            rr, cc = r + dv, c + du
            if 0 <= rr < rows and 0 <= cc < cols and np.isfinite(z[rr, cc]):
                A.append([1.0, du, dv])
                b.append(z[rr, cc])
                w.append(w1[du + rad] * w1[dv + rad])
    A, b, sw = np.array(A), np.array(b), np.sqrt(np.array(w))
    sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    return sol


def test_separable_helpers_and_plane_fit_against_bruteforce():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 9))
    w = np.array([1.0, -2.0, 0.5])
    # correlate_1d(a, w, axis) == sum_d w[d+1] * a shifted by d (zero outside)
    for axis in (0, 1):
        ref = sum(w[k] * (shifted(a, k - 1, 0, 0.0) if axis == 0 else shifted(a, 0, k - 1, 0.0))
                  for k in range(3))
        assert np.allclose(correlate_1d(a, w, axis), ref, atol=1e-12)
    # NaN-aware plane fit == explicit weighted least squares (with holes, at borders)
    X, Y = np.meshgrid(np.arange(12.0), np.arange(10.0))
    z = 0.02 * X ** 1.5 - 0.03 * Y + rng.normal(0, 0.01, X.shape)
    z[rng.random(z.shape) < 0.25] = np.nan
    for window, w1 in ((3, binomial_weights(3)), (5, np.ones(5))):
        fit = plane_fit(z, window, w1, min_cells=5)
        checked = 0
        for r, c in ((0, 0), (5, 6), (9, 11), (3, 2), (7, 9), (4, 4)):
            if not fit["valid"][r, c]:
                continue
            a0, gx, gy = _plane_fit_bruteforce(z, window, w1, r, c)
            assert fit["a"][r, c] == pytest.approx(a0, abs=1e-9)
            assert fit["gx"][r, c] == pytest.approx(gx, abs=1e-9)
            assert fit["gy"][r, c] == pytest.approx(gy, abs=1e-9)
            checked += 1
        assert checked >= 3
    # row-wise NaN median == np.nanmedian
    m = rng.normal(size=(50, 25))
    m[rng.random(m.shape) < 0.4] = np.nan
    m[7] = np.nan                                     # an all-NaN row
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(m, axis=1)
    out = nanmedian_rows(m)
    assert np.isnan(out[7]) and np.allclose(out, ref, equal_nan=True)


# ------------------------------------------------------------------ fusion
def test_flat_plane_integration_converges():
    m = ElevationMap(center=(0.0, 0.0))
    rng = np.random.default_rng(1)
    var_prev = None
    for k in range(6):
        pts = np.column_stack([rng.uniform(0.5, 2.0, 20000), rng.uniform(-1.0, 1.0, 20000),
                               0.1 + rng.normal(0.0, 0.005, 20000)])
        m.integrate(pts, sensor_origin=(0.0, 0.0, 0.5), t=0.1 * k)
        v = np.nanmean(m.layer("variance_raw"))
        if var_prev is not None:
            assert v < var_prev
        var_prev = v
    e = m.layer("elevation_raw")
    sel = np.isfinite(e)
    assert sel.sum() > 1000
    assert np.nanmean(np.abs(e[sel] - 0.1)) < 0.003
    assert m.height_at(1.0, 0.0) == pytest.approx(0.1, abs=0.005)


def test_kalman_weighted_fusion_matches_theory():
    p = ElevationMapParams(mahalanobis_gate=100.0)
    m = ElevationMap(center=(0.0, 0.0), params=p)
    sig = lambda r: p.sensor_noise_base + p.sensor_noise_range_coeff * r * r
    v1 = sig(1.0) ** 2 + p.min_variance
    v2 = sig(3.0) ** 2 + p.min_variance
    m.integrate(np.array([[0.2, 0.2, 0.10]]), ranges=np.array([1.0]))
    m.integrate(np.array([[0.2, 0.2, 0.14]]), ranges=np.array([3.0]))
    r, c = m.world_to_index(0.2, 0.2)
    h = m.layer("elevation_raw")[r, c]
    v = m.layer("variance_raw")[r, c]
    assert h == pytest.approx((v2 * 0.10 + v1 * 0.14) / (v1 + v2), rel=1e-9)
    assert v == pytest.approx(v1 * v2 / (v1 + v2), rel=1e-9)
    assert v < min(v1, v2)


def test_range_dependent_variance():
    m = ElevationMap(center=(0.0, 0.0))
    m.integrate(np.array([[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]]), ranges=np.array([0.5, 3.0]))
    var = m.layer("variance_raw")
    rn, cn = m.world_to_index(0.2, 0.0)
    rf, cf = m.world_to_index(2.0, 0.0)
    assert var[rf, cf] > 10 * var[rn, cn]


def test_outlier_gating():
    m = ElevationMap(center=(0.0, 0.0), confirm_observations=False)
    pt = lambda z: np.array([[0.5, 0.5, z]])
    for k in range(10):
        m.integrate(pt(0.0), ranges=np.array([1.0]), t=0.05 * k)
    r, c = m.world_to_index(0.5, 0.5)
    # a single LOWER outlier is ignored ...
    m.integrate(pt(-0.3), ranges=np.array([1.0]))
    assert m.layer("elevation_raw")[r, c] == pytest.approx(0.0, abs=1e-3)
    # ... until it repeats outlier_reset_count times
    for _ in range(m.params.outlier_reset_count - 1):
        m.integrate(pt(-0.3), ranges=np.array([1.0]))
    assert m.layer("elevation_raw")[r, c] == pytest.approx(-0.3, abs=1e-6)
    # a single HIGHER outlier (mid-air spike) is rejected, a repeated one (new
    # obstacle) replaces the estimate after higher_outlier_count frames
    for _ in range(5):
        m.integrate(pt(-0.3), ranges=np.array([1.0]))
    m.integrate(pt(0.4), ranges=np.array([1.0]))
    assert m.layer("elevation_raw")[r, c] == pytest.approx(-0.3, abs=1e-3)
    m.integrate(pt(-0.3), ranges=np.array([1.0]))     # inlier breaks the run
    for _ in range(m.params.higher_outlier_count):
        m.integrate(pt(0.4), ranges=np.array([1.0]))
    assert m.layer("elevation_raw")[r, c] == pytest.approx(0.4, abs=1e-6)
    # inconsistent outliers never add up to a run (0.557 then 0.741 on a 0.5 m tread)
    mc = ElevationMap(center=(0.0, 0.0), confirm_observations=False)
    for k in range(5):
        mc.integrate(np.array([[0.5, 0.5, 0.508]]), ranges=np.array([1.0]))
    for zs in (0.557, 0.741, 0.60, 0.69):
        mc.integrate(np.array([[0.5, 0.5, zs]]), ranges=np.array([1.0]))
        assert mc.layer("elevation_raw")[r, c] == pytest.approx(0.508, abs=1e-3)
    mc.integrate(np.array([[0.5, 0.5, 0.70]]), ranges=np.array([1.0]))   # agrees with 0.69
    assert mc.layer("elevation_raw")[r, c] == pytest.approx(0.70, abs=1e-6)
    m1 = ElevationMap(center=(0.0, 0.0), higher_outlier_count=1, speckle_filter=False,
                      confirm_observations=False)                 # Fankhauser: keep the higher
    m1.integrate(pt(0.0), ranges=np.array([1.0]))
    m1.integrate(pt(0.4), ranges=np.array([1.0]))
    assert m1.height_at(0.5, 0.5) == pytest.approx(0.4)


def test_robust_cell_statistic_and_max_height_bias():
    # many ground points + a few spikes in one cell -> median ignores spikes
    z = np.r_[np.zeros(20), [0.5, 0.6]]
    pts = np.column_stack([np.full(22, 0.3), np.full(22, 0.3), z])
    m = ElevationMap(center=(0.0, 0.0), max_height_bias=False, speckle_filter=False)
    m.integrate(pts, ranges=np.ones(22))                  # 22 points -> confirmed at once
    assert m.height_at(0.3, 0.3) == pytest.approx(0.0, abs=1e-9)
    # a riser inside a cell (points spread 0 .. 0.17): biased map keeps the top
    zr = np.linspace(0.0, 0.17, 30)
    pts = np.column_stack([np.full(30, 0.3), np.full(30, 0.3), zr])
    mb = ElevationMap(center=(0.0, 0.0), max_height_bias=True, speckle_filter=False)
    mb.integrate(pts, ranges=np.ones(30))
    mu = ElevationMap(center=(0.0, 0.0), max_height_bias=False, speckle_filter=False)
    mu.integrate(pts, ranges=np.ones(30))
    assert mb.height_at(0.3, 0.3) > 0.14
    assert mu.height_at(0.3, 0.3) == pytest.approx(0.085, abs=0.01)


def test_recenter_keeps_world_referenced_data():
    m = ElevationMap(center=(0.0, 0.0), speckle_filter=False, confirm_observations=False)
    m.integrate(np.array([[1.0, 0.5, 0.2], [-2.9, 0.0, 0.3]]), ranges=np.ones(2))
    assert m.recenter(0.52, -0.31)
    assert m.height_at(1.0, 0.5) == pytest.approx(0.2)
    assert np.isnan(m.height_at(-2.9, 0.0))              # left the window
    cx, cy = m.center
    assert abs(cx - 0.52) <= RES / 2 + 1e-9 and abs(cy + 0.31) <= RES / 2 + 1e-9
    assert not m.recenter(cx, cy)
    # drift-free: many small moves return to the same lattice
    for k in range(100):
        m.recenter(0.52 + 0.013 * k, -0.31)
    x0 = m.origin[0]
    assert abs(x0 / RES - round(x0 / RES)) < 1e-6


def test_forgetting():
    m = ElevationMap(center=(0.0, 0.0), variance_growth_rate=1e-3, max_variance=0.01, speckle_filter=False,
                     confirm_observations=False)
    m.integrate(np.array([[0.4, 0.4, 0.1]]), ranges=np.ones(1), t=0.0)
    r, c = m.world_to_index(0.4, 0.4)
    v0 = m.layer("variance_raw")[r, c]
    m.predict(2.0)
    assert m.layer("variance_raw")[r, c] == pytest.approx(v0 + 2e-3)
    m.predict(20.0)
    assert np.isnan(m.height_at(0.4, 0.4))
    # a clock reset (simulation restart) does not corrupt the map
    m.integrate(np.array([[0.4, 0.4, 0.1]]), ranges=np.ones(1), t=0.5)
    assert m.height_at(0.4, 0.4) == pytest.approx(0.1)


def test_hole_filling_small_only():
    h = np.zeros((40, 40))
    h[:, 25:] = np.nan                     # large unexplored region
    h[10:12, 10:12] = np.nan               # small enclosed hole
    h[20, 5:8] = np.nan                    # thin gap (between scan lines)
    out, filled = fill_holes(h, 2)
    assert np.isfinite(out[10:12, 10:12]).all() and np.isfinite(out[20, 5:8]).all()
    assert np.isnan(out[:, 27:]).all()     # unexplored region stays unknown
    assert filled[10, 10] and not filled[5, 5]
    m = ElevationMap.from_elevation(h, (0.0, 0.0), RES)
    assert np.isfinite(m.height_at(10.5 * RES, 10.5 * RES))
    v = m.layer("variance")
    assert v[10, 10] > v[5, 5]             # in-painted cells are less certain


def test_occlusion_shadow_fill_uses_the_line_of_sight():
    """A hole behind a DESCENDING edge (seen from the upper side) is an
    occlusion shadow: it is filled from the far, lower side.  The same hole
    seen from the lower side (grazing front of an ascending tread) and holes
    on smooth ground keep the median."""
    h = np.where(np.arange(40)[None, :] < 20, 0.85, 0.68) * np.ones((40, 1))
    h[:, 20:22] = np.nan                                   # 2-cell shadow right behind the edge
    cols = np.mgrid[0:40, 0:40][1].astype(float)
    from_upper = (np.full((40, 40), 20.0), np.full((40, 40), 5.0))     # sensor at col 5 (upper side)
    from_lower = (np.full((40, 40), 20.0), np.full((40, 40), 35.0))    # sensor at col 35 (lower side)
    out_dn, filled = fill_holes(h, 2, 0.05, from_upper)
    assert filled[2:38, 20:22].all()                      # (the closing never fills the grid border)
    assert np.allclose(out_dn[5:35, 20:22], 0.68)        # mapped edge stays at the real edge
    out_med, _ = fill_holes(h, 2, 0.05, from_lower)       # near side lower -> not a shadow
    out_plain, _ = fill_holes(h, 2)
    assert np.allclose(out_med[5:35, 20:22], out_plain[5:35, 20:22])
    assert np.all(out_plain[5:35, 20] > 0.8)               # the median takes the upper tread
    assert np.allclose(fill_holes(h, 2, 0.0, from_upper)[0], out_plain, equal_nan=True)   # disabled
    # ramp (16 deg) viewed downhill: near/far difference below the threshold -> unbiased median
    ramp = -np.tan(math.radians(16)) * RES * cols
    ramp[18:20, 18:20] = np.nan
    out_r, _ = fill_holes(ramp, 2, 0.05, from_upper)
    truth = -np.tan(math.radians(16)) * RES * cols
    assert np.allclose(out_r[18:20, 18:20], truth[18:20, 18:20], atol=0.004)


def test_viewpoint_bookkeeping_and_descending_edge_map():
    """integrate() remembers the sensor position per cell (recentred,
    forgotten and copied with the map) and the shadow fill keeps a descending
    edge where it is."""
    m = ElevationMap(length_x=2.0, length_y=2.0)
    xs = np.arange(-0.6, 0.62, 0.01)
    X, Y = np.meshgrid(xs, np.arange(-0.6, 0.62, 0.01))
    Z = np.where(X < 0.0, 0.30, 0.10)
    shadow = (X >= 0.0) & (X < 0.10)                      # never observed (behind the edge)
    pts = np.stack([X[~shadow], Y[~shadow], Z[~shadow]], -1)
    m.integrate(pts, sensor_origin=(-0.8, 0.0, 0.6), t=0.0)
    r, c = m.world_to_index(-0.3, 0.0)
    assert m._vx[r, c] == pytest.approx(-0.8) and m._vy[r, c] == pytest.approx(0.0)
    e = m.elevation
    xr = m.index_to_world(0, np.arange(m.shape[1]))[0]
    row = m.world_to_index(0.0, 0.0)[0]
    shadow_cols = (xr > 0.01) & (xr < 0.09)
    assert np.allclose(e[row, shadow_cols], 0.10, atol=1e-6)          # filled from the far side
    assert m.height_at(-0.1, 0.0) == pytest.approx(0.30, abs=1e-6)
    # copy / recenter / forgetting carry the viewpoint along
    m2 = m.copy()
    assert np.array_equal(np.isfinite(m2._vx), np.isfinite(m._vx))
    m.recenter(0.2, 0.0)                                  # 5 cells
    r2, c2 = m.world_to_index(-0.3, 0.0)
    assert m._vx[r2, c2] == pytest.approx(-0.8)
    m.params = m.params.updated(max_variance=1e-4)
    m.predict(100.0)
    assert not np.isfinite(m._vx).any() and m.known_fraction() == 0.0


def test_speckle_filter_removes_unsupported_cells():
    h = np.zeros((30, 30))
    h[:, 15:] = np.nan
    h[10, 22] = 0.23                       # lone spurious return in an occlusion shadow
    h[5, 5] = 0.30                         # single-cell spike: no neighbour supports it
    h[20, 14] = 0.25                       # unsupported spike on the known border
    h[2:4, 8:10] = 0.12                    # 2 x 2 block: cells support each other -> kept
    m = ElevationMap.from_elevation(h, (0.0, 0.0), RES, hole_fill_radius=0)
    e = m.elevation
    assert np.isnan(e[10, 22]) and np.isnan(e[5, 5]) and np.isnan(e[20, 14])
    assert np.allclose(e[2:4, 8:10], 0.12) and e[0, 0] == 0.0
    assert np.isfinite(m.layer("elevation_raw")[10, 22])          # fused state untouched
    off = ElevationMap.from_elevation(h, (0.0, 0.0), RES, hole_fill_radius=0, speckle_filter=False)
    assert off.elevation[10, 22] == pytest.approx(0.23)
    assert np.isnan(m.layer("step")[:, 16:]).all()        # nothing known behind the border


def test_observation_confirmation():
    m = ElevationMap(center=(0.0, 0.0), speckle_filter=False)
    one = lambda z, x=0.5: m.integrate(np.array([[x, 0.5, z]]), ranges=np.array([2.0]))
    one(0.30)                                      # single spurious point in a new cell
    assert np.isnan(m.height_at(0.5, 0.5)) and m.known_fraction() == 0.0
    one(0.00)                                      # inconsistent -> re-initialised, still unconfirmed
    assert np.isnan(m.height_at(0.5, 0.5))
    one(0.003)                                     # consistent second look -> confirmed
    assert m.height_at(0.5, 0.5) == pytest.approx(0.0015, abs=0.002)
    # >= min_points_confirm points in one frame confirm immediately
    pts = np.column_stack([np.full(3, 1.0), np.full(3, 0.5), [0.10, 0.101, 0.099]])
    m.integrate(pts, ranges=np.full(3, 2.0))
    assert m.height_at(1.0, 0.5) == pytest.approx(0.10, abs=1e-3)
    assert m.layer("confirmed").sum() == 2
    # confirmed cells keep their protection: a single higher spike does not replace them
    one(0.40)
    assert m.height_at(0.5, 0.5) == pytest.approx(0.0015, abs=0.002)


# ------------------------------------------------------------------ layers
@pytest.mark.parametrize("azimuth", [0.0, 0.5, 2.0])
def test_slope_layer_on_16deg_ramp(azimuth):
    g = math.tan(math.radians(16.0))
    m = grid_map(lambda X, Y: g * (X * math.cos(azimuth) + Y * math.sin(azimuth)))
    s = np.degrees(m.layer("slope")[5:-5, 5:-5])
    assert np.allclose(s, 16.0, atol=0.05)
    assert np.nanmax(m.layer("roughness")[5:-5, 5:-5]) < 1e-6   # a plane is not rough
    t = m.layer("traversability")[5:-5, 5:-5]
    assert np.all((t > 0.6) & (t < 0.95))


def test_step_roughness_traversability_layers():
    stairs = grid_map(lambda X, Y: np.where(X >= 0.6, 0.17, 0.0))
    step = stairs.layer("step")
    assert stairs.cell_value("step", 0.58, 0.0) == pytest.approx(0.17)
    assert stairs.cell_value("step", 1.2, 0.0) == pytest.approx(0.0)
    assert stairs.cell_value("step", 0.0, 0.0) == pytest.approx(0.0)
    assert np.nanmax(step) == pytest.approx(0.17)
    assert stairs.cell_value("roughness", 0.58, 0.0) > 0.03
    t = stairs.layer("traversability")
    assert stairs.cell_value("traversability", 1.2, 0.0) == pytest.approx(1.0)
    assert 0.0 < stairs.cell_value("traversability", 0.58, 0.0) < 0.3   # edge: low but not blocked
    wall = grid_map(lambda X, Y: np.where(np.abs(X - 1.0) < 0.2, 0.6, 0.0))
    assert wall.cell_value("traversability", 0.8, 0.0) == 0.0             # step >= step_crit
    assert np.all(np.isfinite(t))
    occ = wall.occupancy_data()
    r, c = wall.world_to_index(0.8, 0.0)
    assert occ[r, c] == 100 and occ.dtype == np.int8
    r, c = wall.world_to_index(-1.0, 0.0)
    assert occ[r, c] == 0


def test_height_at_bilinear_and_unknown():
    f = lambda X, Y: 0.1 + 0.2 * X - 0.1 * Y
    m = grid_map(f)
    rng = np.random.default_rng(3)
    xs, ys = rng.uniform(-2.5, 2.5, 50), rng.uniform(-2.5, 2.5, 50)
    assert np.allclose(m.height_at(xs, ys), f(xs, ys), atol=1e-9)
    assert isinstance(m.height_at(0.3, 0.2), float)
    assert np.isnan(m.height_at(10.0, 0.0))
    e = m.elevation
    e[:, :75] = np.nan
    m2 = ElevationMap.from_elevation(e, m.origin, RES, hole_fill_radius=0)
    x75 = m.origin[0] + 75 * RES
    assert np.isnan(m2.height_at(x75 - 0.03, 0.0))       # nearest cell unknown
    assert m2.height_at(x75 - 0.01, 0.0) == pytest.approx(f(x75, 0.0), abs=1e-9)


def test_cloud_roundtrip_and_float32_precision():
    f = lambda X, Y: np.where(X > 62.3, 0.17, -0.88)
    m = grid_map(f, center=(61.9, 0.3))
    e = m.elevation
    e[:10, :] = np.nan
    m = ElevationMap.from_elevation(e, m.origin, RES)
    xyz = m.to_cloud_xyz()
    assert xyz.dtype == np.float32 and xyz.shape == (150, 150, 3)
    assert np.all(np.diff(xyz[0, :, 0]) > 0) and np.all(np.diff(xyz[:, 0, 1]) > 0)
    buf, fields, ps, rs = pack_xyz_organised(xyz)
    msg = FakeCloud(buf, [FakeField(*f[:3]) for f in fields], 150, 150, ps, rs)
    back = pointcloud2_to_xyz(msg, organised=True)
    m2 = ElevationMap.from_cloud_xyz(back)
    assert m2.resolution == pytest.approx(RES, abs=1e-12)
    assert m2.shape == m.shape
    assert np.allclose(m2.origin, m.origin, atol=1e-5)
    e1, e2 = m.elevation, m2.elevation
    assert np.array_equal(np.isnan(e1), np.isnan(e2))
    assert np.nanmax(np.abs(e1 - e2)) < 1e-6
    for name in ("slope", "roughness", "step", "traversability", "variance"):
        assert m2.layer(name).shape == m.shape
    assert m2.height_at(62.5, 0.3) == pytest.approx(0.17, abs=1e-6)
    with pytest.raises(ValueError):
        ElevationMap.from_cloud_xyz(xyz.reshape(-1, 3))
    # layers are computed eagerly by default (controller map callback pays the
    # cost, not the first query in the control loop); lazily on request
    assert m2._geo is not None
    lazy = ElevationMap.from_cloud_xyz(back, compute_layers=False)
    assert lazy._geo is None
    assert np.allclose(lazy.layer("slope"), m2.layer("slope"), equal_nan=True)


def test_yaml_map_section_matches_params():
    with open(os.path.join(PKG_DIR, "config", "terrain.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["terrain_mapper"]["ros__parameters"]
    defaults = ElevationMapParams()
    for k, v in cfg["map"].items():
        assert hasattr(defaults, k), k
        d = getattr(defaults, k)
        assert type(v) is type(d), f"{k}: yaml {type(v).__name__} vs default {type(d).__name__}"
        assert v == pytest.approx(d) if isinstance(d, float) else v == d
    fd = PointFilter()
    assert tuple(cfg["filter"]["body_box_min"]) == fd.body_box_min
    assert cfg["filter"]["max_range"] == fd.max_range


# --------------------------------------------------------- point clouds
def _pack_points(pts, layout, endian="<"):
    """Pack (N,3) points with an arbitrary per-point layout: list of
    (name, offset, fmt) entries, plus 'rgb' padding etc."""
    step = max(o + struct.calcsize(fm) for _n, o, fm in layout)
    step = (step + 3) // 4 * 4 + 4                       # extra padding
    out = bytearray(step * len(pts))
    for i, p in enumerate(pts):
        vals = {"x": p[0], "y": p[1], "z": p[2], "rgb": 0}
        for n, o, fm in layout:
            struct.pack_into(endian + fm, out, i * step + o, vals[n])
    return bytes(out), step


@pytest.mark.parametrize("endian,fmt,dt", [("<", "f", FLOAT32), (">", "f", FLOAT32), ("<", "d", FLOAT64)])
def test_pointcloud2_parsing_variants(endian, fmt, dt):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(24, 3))
    sz = struct.calcsize(fmt)
    layout = [("rgb", 0, "I"), ("x", 4, fmt), ("y", 4 + sz, fmt), ("z", 4 + 2 * sz, fmt)]
    buf, step = _pack_points(pts, layout, endian)
    fields = [FakeField("rgb", 0, 6), FakeField("x", 4, dt), FakeField("y", 4 + sz, dt),
              FakeField("z", 4 + 2 * sz, dt)]
    # organised 4 x 6 with 8 bytes of padding at the end of every row
    rows = [buf[r * 6 * step:(r + 1) * 6 * step] + b"\0" * 8 for r in range(4)]
    msg = FakeCloud(b"".join(rows), fields, 6, 4, step, 6 * step + 8, endian == ">")
    org = pointcloud2_to_xyz(msg, organised=True)
    tol = 1e-6 if dt == FLOAT32 else 1e-12
    assert org.shape == (4, 6, 3)
    assert np.allclose(org.reshape(-1, 3), pts, atol=tol)
    # unorganised + invalid points removed + array.array / list data
    import array
    pts2 = pts.copy()
    pts2[3] = np.nan
    buf2, _ = _pack_points(pts2, layout, endian)
    for data in (buf2, array.array("B", buf2), list(buf2)):
        m2 = FakeCloud(data, fields, 24, 1, step, 24 * step, endian == ">")
        flat = pointcloud2_to_xyz(m2)
        assert flat.shape == (23, 3)
        assert np.allclose(flat, np.delete(pts, 3, axis=0), atol=tol)


def test_pointcloud2_errors():
    with pytest.raises(KeyError):
        unpack_fields(b"\0" * 12, [("x", 0, FLOAT32), ("y", 4, FLOAT32)], 1, 1, 12)
    with pytest.raises(ValueError):
        unpack_fields(b"\0" * 8, [("x", 0, FLOAT32), ("y", 4, FLOAT32), ("z", 8, FLOAT32)], 1, 1, 12)
    out = unpack_fields(b"\x05", [("x", 0, UINT8), ("y", 0, UINT8), ("z", 0, UINT8)], 1, 1, 1)
    assert out.shape == (1, 1, 3) and out[0, 0, 0] == 5.0


def _ground_depth(pose, intr):
    """Analytic depth image of the z = 0 plane."""
    T = camera_world_pose(pose, "depth_camera")
    a, b = intr.ray_grid()
    dirs = np.stack([np.ones_like(a), a, b], -1) @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -T[2, 3] / dirs[..., 2]
    t[(dirs[..., 2] >= 0) | (t > 4.0)] = np.nan
    return t


def test_depth_image_to_world_points_flat_ground():
    intr = DepthIntrinsics.from_sensor("depth_camera")
    assert (intr.width, intr.height) == (212, 120)
    pose = (3.0, -1.0, 0.34, 0.02, -0.03, 0.7)
    d = _ground_depth(pose, intr)
    pts = depth_image_to_points(d, intr)
    assert pts.shape[0] > 0.8 * d.size
    pw, rng, origin = camera_points_to_world(pts, pose, "link", filt=PointFilter(max_range=10.0))
    assert np.max(np.abs(pw[:, 2])) < 1e-9
    assert np.allclose(np.linalg.norm(pw - origin, axis=1), rng)
    # 16UC1 millimetre images
    mm = np.nan_to_num(d * 1000.0, nan=0.0).astype(np.uint16)
    dm = depth_to_metres(mm, "16UC1")
    assert np.isnan(dm[mm == 0]).all()
    assert np.nanmax(np.abs(dm - d)) < 1e-3
    # organised output and an image of another resolution (scaled intrinsics)
    org = depth_image_to_points(d, intr, organised=True)
    assert org.shape == (120, 212, 3)
    half = depth_image_to_points(d[::2, ::2], intr)
    assert half.shape[0] > 0


def test_convention_detection():
    intr = DepthIntrinsics.from_sensor("depth_camera")
    pose = (0.0, 0.0, 0.34, 0.0, 0.0, 0.0)
    link = depth_image_to_points(_ground_depth(pose, intr), intr)
    from dusan_common.camera import PinholeCamera
    optical = PinholeCamera.link_to_optical(link)
    assert detect_convention(link)[0] == "link"
    assert detect_convention(optical)[0] == "optical"
    assert detect_convention(link[:5])[0] == "unknown"
    det = ConventionDetector("auto", lock_after=3)
    for _ in range(3):
        det.update(optical)
    assert det.locked == "optical"
    assert det.update(link) == "optical"             # locked
    assert ConventionDetector("link").update(optical) == "link"
    with pytest.raises(ValueError):
        ConventionDetector("sideways")
    # optical points transform to the same world points as link points
    pw1, _, _ = camera_points_to_world(link, pose, "link")
    pw2, _, _ = camera_points_to_world(optical, pose, "optical")
    assert np.allclose(pw1, pw2)


def test_point_filter_body_range_height():
    pose = (0.0, 0.0, 0.34, 0.0, 0.0, 0.0)
    T_wc = camera_world_pose(pose, "depth_camera")
    world = np.array([[0.3, 0.0, 0.30],     # on the trunk (body box)
                      [1.0, 0.0, 0.00],     # ground ahead -> kept
                      [5.0, 0.0, 0.00],     # beyond max range
                      [1.5, 0.0, 1.20],     # overhang above the robot
                      [0.35, 0.0, 0.33]])   # too close to the camera
    cam = world_to_camera_points(world, pose)
    assert np.allclose((cam @ T_wc[:3, :3].T) + T_wc[:3, 3], world)   # exact inverse
    pw, rng, _ = camera_points_to_world(cam, pose, "link")
    assert pw.shape == (1, 3) and np.allclose(pw[0], [1.0, 0.0, 0.0])


def test_pose_buffer_interpolation():
    pb = PoseBuffer(max_age=1.0, max_extrapolation=0.1)
    assert pb.at(0.0) is None
    pb.add(10.0, (0, 0, 0.3, 0, 0, 3.10))
    pb.add(10.1, (1, 2, 0.4, 0.1, 0, -3.10))      # yaw wraps through pi
    p = pb.at(10.05)
    assert p[0] == pytest.approx(0.5) and p[1] == pytest.approx(1.0) and p[2] == pytest.approx(0.35)
    assert abs(abs(p[5]) - math.pi) < 1e-6           # shortest arc, not through 0
    assert pb.at(10.15)[0] == 1 and pb.at(10.5) is None
    for k in range(30):
        pb.add(10.2 + 0.1 * k, (k, 0, 0, 0, 0, 0))
    assert len(pb) <= 12
    pb.add(1.0, (0, 0, 0, 0, 0, 0))                  # clock jumped back -> reset
    assert len(pb) == 1


# ------------------------------------------------------------- end to end
def test_end_to_end_depth_camera_step_scene():
    """Simulated depth frames of a 0.12 m platform -> fused map accuracy."""
    hf = lambda x, y: np.where((x > 1.2) & (x < 2.2) & (np.abs(y) < 0.8), 0.12, 0.0)
    intr = DepthIntrinsics.from_sensor("depth_camera")
    poses = [(x, 0.0, 0.34, 0.0, 0.0, 0.0) for x in (0.0, 0.15, 0.3)]
    m = integrate_scene(ElevationMap(center=(0.0, 0.0)), hf, poses, intr, noise=0.002)
    e = m.elevation
    X, Y = m.grid_xy()
    gt = hf(X, Y)
    near_edge = max_filter(gt, 5) - min_filter(gt, 5) > 0.01
    sel = np.isfinite(e) & ~near_edge & (np.abs(Y) < 1.0) & (X > 0.8) & (X < 2.0)
    assert sel.sum() > 200
    rmse = float(np.sqrt(np.mean((e[sel] - gt[sel]) ** 2)))
    assert rmse < 0.01
    assert m.height_at(1.6, 0.0) == pytest.approx(0.12, abs=0.01)
    assert m.cell_value("step", 1.18, 0.0) > 0.09


# ===================================================== node wrapper smoke tests
# ROS is not available on the development PC, so the thin ROS wrappers are
# executed against a FAKE rclpy / message layer: permissive message objects,
# a Node that records publishers / subscriptions / timers / parameters, and
# YAML-like parameter overrides (statically typed like rclpy).  This catches
# wrong attribute names, wrong message fields and broken callbacks before the
# code reaches the Ubuntu machine.  (Also used by test_teleop.py.)
class FakeMsg:
    """Permissive ROS message: kwargs become attributes, unknown attributes
    are created on first access as nested messages (header.stamp, ...)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        v = FakeMsg()
        object.__setattr__(self, name, v)
        return v


class FakeRosEnv:
    """Installs fake rclpy / *_msgs modules into sys.modules (via monkeypatch)
    and drops cached node modules so they are re-imported under the fakes."""

    MSGS = {
        "std_msgs.msg": ("String", "Bool", "Float32", "Int32", "Header"),
        "geometry_msgs.msg": ("Twist", "Vector3"),
        "nav_msgs.msg": ("Odometry", "OccupancyGrid", "Path"),
        "sensor_msgs.msg": ("PointCloud2", "PointField", "Image", "CameraInfo", "BatteryState", "Imu",
                            "JointState"),
        "rcl_interfaces.msg": ("ParameterDescriptor",),
        "visualization_msgs.msg": ("Marker", "MarkerArray"),
    }
    NODE_MODULES = ("terrain_mapping_node", "keyboard_teleop")

    def __init__(self, monkeypatch, overrides=None):
        import sys
        import types
        self.now = 10.0                              # node clock [s]
        self.overrides = dict(overrides or {})
        self.nodes = []
        env = self

        mods = {}
        for pkg, names in self.MSGS.items():
            root = pkg.split(".")[0]
            mods.setdefault(root, types.ModuleType(root))
            m = types.ModuleType(pkg)
            for n in names:
                setattr(m, n, type(n, (FakeMsg,), {}))
            mods[pkg] = m

        class Pub:
            def __init__(self, typ, topic, qos):
                self.type, self.topic, self.qos, self.msgs = typ, topic, qos, []

            def publish(self, msg):
                self.msgs.append(msg)

        class Logger:
            def __init__(self):
                self.lines = []
                self.rules = CallSiteRules(skip_files=(__file__,))   # rclpy: one severity per call site

            def _log(self, level, text, **kw):
                self.rules.check(level, **kw)
                self.lines.append((level, str(text)))

            def debug(self, t, **kw):
                self._log("debug", t, **kw)

            def info(self, t, **kw):
                self._log("info", t, **kw)

            def warning(self, t, **kw):
                self._log("warn", t, **kw)

            def warn(self, t, **kw):
                self._log("warn", t, **kw)

            def error(self, t, **kw):
                self._log("error", t, **kw)

        class Time:
            def __init__(self, t):
                self.nanoseconds = int(round(t * 1e9))

            def to_msg(self):
                s = int(self.nanoseconds // 1_000_000_000)
                return FakeMsg(sec=s, nanosec=int(self.nanoseconds - s * 1_000_000_000))

        class Clock:
            def now(self):
                return Time(env.now)

        class Node:
            def __init__(self, name, **kw):
                self.name = name
                self.params, self.pubs, self.subs, self.timers = {}, {}, {}, []
                self.logger = Logger()
                env.nodes.append(self)

            def declare_parameter(self, name, value=None, descriptor=None):
                if name in self.params:
                    raise RuntimeError(f"ParameterAlreadyDeclared {name}")
                v = env.overrides.get(name, value)
                dynamic = bool(getattr(descriptor, "dynamic_typing", False)) if descriptor else False
                if (not dynamic and v is not None and value is not None
                        and type(v) is not type(value)):     # rclpy: static typing
                    raise TypeError(f"InvalidParameterType {name}: {type(v).__name__} "
                                    f"!= {type(value).__name__}")
                self.params[name] = v
                return types.SimpleNamespace(value=v)

            def get_parameter(self, name):
                return types.SimpleNamespace(value=self.params[name])

            def create_publisher(self, typ, topic, qos, **kw):
                p = Pub(typ, topic, qos)
                self.pubs[topic] = p
                return p

            def create_subscription(self, typ, topic, cb, qos, **kw):
                self.subs[topic] = (typ, cb)
                return object()

            def create_timer(self, period, cb, **kw):
                self.timers.append((period, cb))
                return object()

            def get_logger(self):
                return self.logger

            def get_clock(self):
                return Clock()

            def destroy_node(self):
                pass

        class _Enum:
            def __getattr__(self, n):
                return n

        rclpy = types.ModuleType("rclpy")
        rclpy.init = lambda args=None: None
        rclpy.ok = lambda: True
        rclpy.shutdown = lambda: None
        rclpy.spin = lambda node: None
        node_mod = types.ModuleType("rclpy.node")
        node_mod.Node = Node
        qos = types.ModuleType("rclpy.qos")
        qos.QoSProfile = lambda **kw: dict(kw)
        qos.ReliabilityPolicy = qos.HistoryPolicy = qos.DurabilityPolicy = _Enum()
        qos.qos_profile_sensor_data = {"reliability": "BEST_EFFORT"}
        ex = types.ModuleType("rclpy.executors")
        ex.SingleThreadedExecutor = ex.MultiThreadedExecutor = object
        cg = types.ModuleType("rclpy.callback_groups")
        cg.MutuallyExclusiveCallbackGroup = cg.ReentrantCallbackGroup = object
        rclpy.node, rclpy.qos, rclpy.executors, rclpy.callback_groups = node_mod, qos, ex, cg
        mods.update({"rclpy": rclpy, "rclpy.node": node_mod, "rclpy.qos": qos,
                     "rclpy.executors": ex, "rclpy.callback_groups": cg})
        for name, mod in mods.items():
            monkeypatch.setitem(sys.modules, name, mod)
        # node modules are re-imported under the fakes; at teardown monkeypatch
        # restores the previous sys.modules entry / package attribute (or removes
        # the fake-backed module), so later tests never see it
        import module1_locomotion as pkg_mod
        for short in self.NODE_MODULES:
            full = "module1_locomotion." + short
            monkeypatch.setitem(sys.modules, full, None)
            del sys.modules[full]
            monkeypatch.setattr(pkg_mod, short, None, raising=False)
        self.msg = {pkg: mods[pkg] for pkg in self.MSGS}

    @staticmethod
    def stamp(t):
        s = int(t)
        return FakeMsg(sec=s, nanosec=int(round((t - s) * 1e9)))


def _odom_msg(env, pose, t):
    from dusan_common.geometry import euler_to_quat
    msg = env.msg["nav_msgs.msg"].Odometry()
    msg.header.stamp = env.stamp(t)
    msg.header.frame_id = "odom"
    p = msg.pose.pose.position
    p.x, p.y, p.z = pose[:3]
    q = msg.pose.pose.orientation
    q.x, q.y, q.z, q.w = euler_to_quat(*pose[3:6])
    return msg


def _gazebo_cloud_msg(env, pts_org, t, optical=False):
    """Organised x y z rgb float32 cloud like gz-sensors (point_step 16, inf = no return)."""
    from dusan_common.camera import PinholeCamera
    p = PinholeCamera.link_to_optical(pts_org) if optical else pts_org
    h, w = p.shape[:2]
    rec = np.zeros((h, w), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    p = np.where(np.isfinite(p), p, np.inf)
    rec["x"], rec["y"], rec["z"] = p[..., 0], p[..., 1], p[..., 2]
    PF = env.msg["sensor_msgs.msg"].PointField
    msg = env.msg["sensor_msgs.msg"].PointCloud2()
    msg.header.stamp = env.stamp(t)
    msg.header.frame_id = "spot/base_link/depth_camera"
    msg.height, msg.width = h, w
    msg.fields = [PF(name=n, offset=4 * k, datatype=FLOAT32, count=1) for k, n in enumerate(("x", "y", "z", "rgb"))]
    msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 16, 16 * w, False
    msg.data = rec.tobytes()
    return msg


def test_terrain_mapper_node_with_fake_ros(monkeypatch):
    import importlib
    import json
    env = FakeRosEnv(monkeypatch, overrides={"publish_rate": 5,          # int for a float parameter
                                             "map.length_x": 4.0, "map.length_y": 4.0,
                                             "classifier.trot_hold_time": 1.0})
    node_mod = importlib.import_module("module1_locomotion.terrain_mapping_node")
    node = node_mod.TerrainMapperNode()
    # contract: topics, message types, parameters (wrong YAML type coerced, not fatal)
    assert node.publish_rate == 5.0 and node.emap.shape == (100, 100)
    assert node.classifier.params.trot_hold_time == 1.0
    assert {t: p.type.__name__ for t, p in node.pubs.items()} == {
        "/elevation_map/points": "PointCloud2", "/elevation_map/traversability": "OccupancyGrid",
        "/terrain/info": "String"}
    assert {t: s[0].__name__ for t, s in node.subs.items()} == {
        "/odom": "Odometry", "/depth_camera/points": "PointCloud2", "/depth_camera/image": "Image",
        "/depth_camera/camera_info": "CameraInfo"}
    assert len(node.timers) == 1 and node.timers[0][0] == pytest.approx(0.2)
    timer = node.timers[0][1]
    timer()                                             # no odometry yet -> nothing published
    assert not node.pubs["/terrain/info"].msgs

    # flat floor at z = -0.88 (lower corridor), robot standing on it, heading 0.3 rad
    intr = DepthIntrinsics.from_sensor("depth_camera")
    pose = (58.0, 0.2, -0.88 + 0.34, 0.0, 0.0, 0.3)
    floor = lambda x, y: np.full(np.shape(x), -0.88)
    org = depth_image_to_points(simulate_depth(floor, pose, intr), intr, organised=True)
    for k in range(3):
        env.now = 10.0 + 0.1 * k
        node.subs["/odom"][1](_odom_msg(env, pose, env.now))
        node.subs["/depth_camera/points"][1](_gazebo_cloud_msg(env, org, env.now))
    assert node.n_clouds == 3 and node.last_source == "points"
    for k in range(6):
        env.now = 10.3 + 0.2 * k
        timer()
    # /elevation_map/points: organised float32 cloud in 'odom', readable by the controller API
    cmsg = node.pubs["/elevation_map/points"].msgs[-1]
    assert cmsg.header.frame_id == "odom" and (cmsg.height, cmsg.width) == (100, 100)
    assert [f.name for f in cmsg.fields] == ["x", "y", "z"] and cmsg.point_step == 12
    xyz = pointcloud2_to_xyz(cmsg, organised=True)
    m = ElevationMap.from_cloud_xyz(xyz)
    ahead = (pose[0] + 1.0 * math.cos(0.3), pose[1] + 1.0 * math.sin(0.3))
    assert m.height_at(*ahead) == pytest.approx(-0.88, abs=0.01)
    assert abs(m.center[0] - pose[0]) <= RES and abs(m.center[1] - pose[1]) <= RES
    # /elevation_map/traversability
    g = node.pubs["/elevation_map/traversability"].msgs[-1]
    data = np.asarray(g.data, dtype=int)
    assert data.size == g.info.width * g.info.height == 100 * 100
    assert data.min() >= -1 and data.max() <= 100 and np.any(data == 0)
    assert g.info.origin.position.x == pytest.approx(m.origin[0] - RES / 2, abs=1e-5)
    # /terrain/info JSON contract
    info = json.loads(node.pubs["/terrain/info"].msgs[-1].data)
    for key in ("roughness", "max_step", "slope_deg", "class", "recommended_gait", "ahead_clear"):
        assert key in info
    assert info["class"] == "flat" and info["recommended_gait"] == "trot" and info["ahead_clear"] is True
    assert info["convention"] == "link" and info["source"] == "points"
    # friction prior: the plates of the (source tree) world_layout.yaml, none near this pose
    assert {z.zone_id for z in node.friction_map.zones} == {"C-OBS4", "F-OBS9"}
    assert info["slippery"] is False and info["friction_mu"] is None and info["friction_zones"] == []

    # optical cloud (z forward) is detected and locked after convention_lock_frames
    node2 = node_mod.TerrainMapperNode()
    for k in range(6):
        env.now = 20.0 + 0.1 * k
        node2.subs["/odom"][1](_odom_msg(env, pose, env.now))
        node2.subs["/depth_camera/points"][1](_gazebo_cloud_msg(env, org, env.now, optical=True))
    assert node2.convention.locked == "optical"
    env.now += 0.2
    node2.timers[0][1]()
    m2 = ElevationMap.from_cloud_xyz(pointcloud2_to_xyz(node2.pubs["/elevation_map/points"].msgs[-1],
                                                        organised=True))
    assert np.nanmax(np.abs(m2.elevation + 0.88)) < 0.02

    # depth-image fallback (32FC1 metres) when no cloud has arrived
    node3 = node_mod.TerrainMapperNode()
    env.now = 30.0
    node3.subs["/odom"][1](_odom_msg(env, pose, env.now))
    d = simulate_depth(floor, pose, intr).astype(np.float32)
    img = env.msg["sensor_msgs.msg"].Image(height=120, width=212, encoding="32FC1", is_bigendian=0,
                                           step=212 * 4, data=np.where(np.isfinite(d), d, np.inf).tobytes())
    img.header.stamp = env.stamp(env.now)
    node3.subs["/depth_camera/image"][1](img)
    assert node3.n_images == 1 and node3.last_source == "image"
    assert node3.emap.known_fraction() > 0.1
    # ... ignored while clouds are flowing
    node3.subs["/depth_camera/points"][1](_gazebo_cloud_msg(env, org, env.now))
    node3.subs["/depth_camera/image"][1](img)
    assert node3.n_images == 1


def test_terrain_mapper_friction_prior_from_the_layout(monkeypatch, tmp_path):
    """layout_file: a plate 0.5 m ahead of the robot -> /terrain/info slippery, friction_mu;
    friction_prior false / a missing layout only disable the prior (warning)."""
    import importlib
    import json
    layout = tmp_path / "layout.yaml"
    layout.write_text(yaml.safe_dump({"obstacles": [
        {"id": "TEST-PLATE", "type": "plate", "rect": [58.5, -1.0, 60.5, 1.0], "floor_z": -0.88,
         "thickness": 0.01, "mu": 0.15}]}), encoding="utf-8")
    pose = (58.0, 0.0, -0.88 + 0.34, 0.0, 0.0, 0.0)
    for overrides, expect in (({"layout_file": str(layout)}, True),
                              ({"layout_file": str(layout), "friction_prior": False}, False),
                              ({"layout_file": str(tmp_path / "missing.yaml")}, False)):
        env = FakeRosEnv(monkeypatch, overrides=overrides)
        node_mod = importlib.import_module("module1_locomotion.terrain_mapping_node")
        node = node_mod.TerrainMapperNode()
        assert (node.friction_map is not None) == expect
        env.now = 10.0
        node.subs["/odom"][1](_odom_msg(env, pose, env.now))
        node.timers[0][1]()
        info = json.loads(node.pubs["/terrain/info"].msgs[-1].data)
        assert info["slippery"] is expect
        if expect:
            assert info["class"] == "slippery" and info["recommended_gait"] == "trot"
            assert info["friction_mu"] == pytest.approx(0.15) and info["friction_zones"] == ["TEST-PLATE"]
            assert any("friction prior: 1 low-friction zone" in m for _lv, m in node.logger.lines)
        elif "missing" in overrides["layout_file"]:
            assert any(lv == "warn" and "friction prior disabled" in m for lv, m in node.logger.lines)
