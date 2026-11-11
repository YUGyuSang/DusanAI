"""
Robot-centric 2.5-D elevation map with Kalman height fusion (numpy only).

Grid
----
* Axis-aligned with the world (``odom``) frame, ``rows ~ y``, ``cols ~ x``.
  Cell (r, c) has its CENTRE at ``(x0 + c * res, y0 + r * res)`` where
  ``origin = (x0, y0)`` is the centre of cell [0, 0].
* Robot-centric rolling window (default 6 m x 6 m at 0.04 m = 150 x 150):
  ``recenter(x, y)`` shifts the data by whole cells so the robot stays in the
  middle; cells leaving the window are dropped, entering cells are unknown.
  Cell centres stay on a fixed lattice (multiples of ``res``), so shifting
  never resamples (no blur).

Fusion (per cell, "ray-free" as in Fankhauser et al., 2014/2018)
-----------------------------------------------------------------
1. Points falling into one cell are reduced to ONE robust measurement: the
   median height (optionally an upper quantile when the points in the cell
   spread vertically - "max-height bias" so obstacle edges / risers are not
   averaged away).
2. Measurement variance ``sigma(r)^2 / n_eff + spread term`` with the depth
   noise model ``sigma(r) = sigma0 + k * r^2`` (stereo / ToF depth error grows
   with the square of the range).
3. 1-D Kalman update of (height, variance) with Mahalanobis gating:
   inlier -> fused; an outlier HIGHER than the estimate replaces it after
   ``higher_outlier_count`` consecutive, mutually CONSISTENT frames (new
   obstacle; the default 2 rejects single-frame mid-air spikes of mixed
   pixels); an outlier LOWER than the estimate is ignored until it repeats
   ``outlier_reset_count`` times (protects against occlusion / mixed
   pixels).  A run only grows while each outlier agrees with the previous
   one (|dz| <= outlier_consistency or 3 sigma), so random spikes never add up.
   Cells whose points spread vertically in a frame (wall faces, risers) only
   show the part of the face inside the field of view: such measurements are
   lower bounds, so a higher one raises the cell at once (max fusion,
   ``raise_on_vertical``) instead of being averaged with earlier low views.
4. Confirmation: a cell is used by the derived layers only once it is
   confirmed - by a measurement made of >= ``min_points_confirm`` points in
   one frame (already a robust median) or by a second, consistent
   observation.  An unconfirmed cell hit by an inconsistent measurement is
   simply re-initialised, so single spurious returns (mixed pixels grazing
   an edge, typically landing in occlusion shadows that are never observed
   again) never reach the map.
5. Forgetting: the variance of every known cell grows with time
   (``variance_growth_rate``); above ``max_variance`` the cell becomes unknown.

Derived layers (lazy, cached until the heights change)
------------------------------------------------------
``elevation``      fused heights, speckles removed (isolated cells and spikes
                   that differ > spike_threshold from every known neighbour -
                   e.g. a spurious return inside an occlusion shadow that is
                   never observed again), small holes in-painted (median of the
                   known neighbours; only holes enclosed by known cells,
                   detected by a morphological closing of the known mask).
                   A hole in the occlusion shadow BEHIND an edge (as seen from
                   the sensor position stored per cell) takes the far, lower
                   side instead - see ``fill_holes``.
``variance``       fused variance (in-painted cells: neighbour mean + penalty)
``slope``          [rad] inclination from a weighted least-squares plane fit
                   with Sobel (binomial) weights - identical to the Sobel
                   operator on complete neighbourhoods, NaN-aware otherwise
``roughness``      [m] residual standard deviation of a local plane fit
                   (slope invariant: a clean ramp has ~0 roughness)
``step``           [m] max - min height in a foot-sized window
``traversability`` 0..1, ``1 - weighted sum of slope/step/roughness``
                   normalised by their critical values; 0 when the step is
                   critical or a PLANAR surface is steeper than critical
``gradient_x``, ``gradient_y``, ``elevation_raw``, ``filled`` (extras)

The organised cloud of ``to_cloud_xyz`` / ``from_cloud_xyz`` is the transport
format of ``/elevation_map/points`` (DESIGN_SPEC section 4/5).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields as dc_fields, replace as dc_replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

LAYER_NAMES = ("elevation", "variance", "slope", "roughness", "step", "traversability")
EXTRA_LAYER_NAMES = ("elevation_raw", "variance_raw", "confirmed", "gradient_x", "gradient_y", "filled")


# =============================================================================
#  Parameters
# =============================================================================
@dataclass
class ElevationMapParams:
    """All tunables of the elevation map (ROS parameters ``map.<name>``)."""
    # geometry
    length_x: float = 6.0            # [m] map size along world x
    length_y: float = 6.0            # [m] map size along world y
    resolution: float = 0.04         # [m] cell size
    # depth sensor noise model  sigma(r) = sensor_noise_base + sensor_noise_range_coeff * r^2
    sensor_noise_base: float = 0.003
    sensor_noise_range_coeff: float = 0.0025
    max_points_per_cell: int = 4     # cap of independent samples when averaging inside a cell
    spread_variance_weight: float = 0.5   # adds (in-cell std^2 - noise^2) * weight to the variance
    min_variance: float = 1.0e-6     # [m^2] variance floor (keeps the filter responsive)
    # fusion / gating
    mahalanobis_gate: float = 3.0    # [sigma] inlier gate
    outlier_reset_count: int = 3     # consecutive LOWER outliers before the cell is re-initialised
    higher_outlier_count: int = 2    # consecutive HIGHER outliers before the cell is re-initialised
    outlier_consistency: float = 0.03  # [m] consecutive outliers must agree within this (or 3 sigma)
    max_height_bias: bool = True     # use an upper quantile in vertically spread cells
    bias_spread: float = 0.03        # [m] in-cell spread that triggers the bias
    bias_quantile: float = 0.9
    raise_on_vertical: bool = True   # vertically spread cells are LOWER BOUNDS: higher -> raise at once
    confirm_observations: bool = True
    min_points_confirm: int = 3      # points in one frame that confirm a cell immediately
    # forgetting
    variance_growth_rate: float = 1.0e-4   # [m^2/s] process noise
    max_variance: float = 0.02             # [m^2] above -> cell forgotten (unknown)
    # speckle removal (derived layers only; the fused state is kept)
    speckle_filter: bool = True
    spike_threshold: float = 0.08    # [m]
    # hole filling
    hole_fill_radius: int = 2        # [cells] closing radius; 0 disables in-painting
    fill_variance: float = 4.0e-4    # [m^2] variance penalty of in-painted cells
    shadow_fill_step: float = 0.05   # [m] near side higher than the far side by more -> hole is an
    #                                  occlusion shadow behind an edge, filled from the far side; 0 = off
    # derived layers
    slope_window: int = 3            # [cells, odd] Sobel-weighted plane fit
    roughness_window: int = 5        # [cells, odd] uniform plane fit
    step_window: int = 5             # [cells, odd] foot-sized max-min window
    min_fit_cells: int = 5           # minimum known cells for a plane fit
    # traversability
    slope_crit: float = 0.70         # [rad] ~40 deg
    step_crit: float = 0.22          # [m] steps above are untraversable (stairs are 0.17)
    roughness_crit: float = 0.05     # [m]
    planar_roughness: float = 0.02   # [m] residual below which a surface counts as planar
    w_slope: float = 1.0
    w_step: float = 1.0
    w_roughness: float = 1.0
    # transport
    default_variance: float = 1.0e-4  # [m^2] variance assigned by from_cloud_xyz

    @classmethod
    def from_dict(cls, d: Optional[dict] = None, **overrides) -> "ElevationMapParams":
        """Build from a dict (unknown keys ignored, values coerced to the
        field's type, e.g. ints from YAML for float fields)."""
        return cls().updated(**{**(d or {}), **overrides})

    def updated(self, **kw) -> "ElevationMapParams":
        valid = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        clean = {k: _coerce(v, valid[k]) for k, v in kw.items() if k in valid and v is not None}
        return dc_replace(self, **clean)


def _coerce(value, default):
    """Coerce ``value`` to the type of ``default`` (bool/int/float/str)."""
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        return int(round(float(value)))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, tuple):
        return tuple(float(v) for v in value)
    return value


# =============================================================================
#  Neighbourhood helpers (numpy only, NaN aware)
# =============================================================================
def shifted(a: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """b[i, j] = a[i + dr, j + dc]; cells shifted in from outside get ``fill``."""
    rows, cols = a.shape
    out = np.full(a.shape, fill, dtype=a.dtype)
    if abs(dr) >= rows or abs(dc) >= cols:
        return out
    src_r = slice(max(dr, 0), rows + min(dr, 0))
    dst_r = slice(max(-dr, 0), rows + min(-dr, 0))
    src_c = slice(max(dc, 0), cols + min(dc, 0))
    dst_c = slice(max(-dc, 0), cols + min(-dc, 0))
    out[dst_r, dst_c] = a[src_r, src_c]
    return out


def _separable(a: np.ndarray, radius: int, op, fill) -> np.ndarray:
    """Apply a separable (2r+1)x(2r+1) window reduction (np.maximum/minimum/
    logical_or/logical_and/add) with constant ``fill`` outside the grid."""
    if radius <= 0:
        return a.copy()
    out = a.copy()
    for d in range(1, radius + 1):
        out = op(out, op(shifted(a, d, 0, fill), shifted(a, -d, 0, fill)))
    tmp = out
    out = tmp.copy()
    for d in range(1, radius + 1):
        out = op(out, op(shifted(tmp, 0, d, fill), shifted(tmp, 0, -d, fill)))
    return out


def max_filter(a: np.ndarray, window: int) -> np.ndarray:
    """NaN-ignoring max filter (NaN where the whole window is unknown)."""
    v = np.where(np.isfinite(a), a, -np.inf)
    out = _separable(v, window // 2, np.maximum, -np.inf)
    return np.where(np.isfinite(out), out, np.nan)


def min_filter(a: np.ndarray, window: int) -> np.ndarray:
    """NaN-ignoring min filter (NaN where the whole window is unknown)."""
    v = np.where(np.isfinite(a), a, np.inf)
    out = _separable(v, window // 2, np.minimum, np.inf)
    return np.where(np.isfinite(out), out, np.nan)


def box_sum(a: np.ndarray, window: int) -> np.ndarray:
    """Sum over a (window x window) neighbourhood, zero outside the grid."""
    return _separable(np.asarray(a, dtype=np.float64), window // 2, np.add, 0.0)


def binomial_weights(window: int) -> np.ndarray:
    """1-D binomial weights ([1,2,1] for 3, [1,4,6,4,1] for 5): the Sobel
    smoothing kernel."""
    return np.array([math.comb(window - 1, i) for i in range(window)], dtype=np.float64)


def correlate_1d(a: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    """``out[i] = sum_d weights[d + r] * a[i + d]`` along ``axis`` (rows = 0,
    columns = 1), zero outside the grid; r = len(weights) // 2.

    Implemented with in-place slice accumulation (no shifted copies), so a
    separable 2-D window sum costs 2 * window array additions."""
    a = np.asarray(a, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    r = w.size // 2
    n = a.shape[axis]
    out = np.zeros(a.shape)
    for k in range(w.size):
        wk = float(w[k])
        d = k - r
        if wk == 0.0 or abs(d) >= n:
            continue
        dst_sl = slice(max(-d, 0), n - max(d, 0))
        src_sl = slice(max(d, 0), n - max(-d, 0))
        if axis == 0:
            dst, src = out[dst_sl, :], a[src_sl, :]
        else:
            dst, src = out[:, dst_sl], a[:, src_sl]
        if wk == 1.0:
            dst += src
        else:
            dst += wk * src
    return out


def plane_fit(z: np.ndarray, window: int, weights_1d: Optional[np.ndarray] = None,
              min_cells: int = 5) -> Dict[str, np.ndarray]:
    """Weighted least-squares plane  z = a + b*u + c*v  around EVERY cell.

    u (columns, x) and v (rows, y) are the cell offsets inside the
    (window x window) neighbourhood; only known (finite) cells participate.
    With weights w(u, v) = w1(u) * w1(v) = [1,2,1] x [1,2,1] and a complete
    neighbourhood the slope estimate b equals the Sobel x-derivative / 8 -
    the fit is the NaN-aware generalisation of the Sobel operator.

    The normal equations need the window moments
    ``sum w(u) w(v) u^a v^b f``  (f = known mask, height, height^2); the weight
    is a product, so every moment is a SEPARABLE correlation: a horizontal
    pass with ``w1(u) u^a`` followed by a vertical pass with ``w1(v) v^b``.

    Returns dict with 'gx', 'gy' (height change per CELL), 'a' (fitted height
    at the centre), 'resid' (weighted residual std, unbiased for 3
    parameters), 'n' (known cells) and 'valid' (enough cells, well posed)."""
    r = window // 2
    w1 = np.ones(window) if weights_1d is None else np.asarray(weights_1d, dtype=np.float64)
    off = np.arange(-r, r + 1, dtype=np.float64)
    w_u0, w_u1, w_u2 = w1, w1 * off, w1 * off * off
    known = np.isfinite(z)
    ref = float(np.nanmean(z)) if known.any() else 0.0   # conditioning
    zc = np.where(known, z - ref, 0.0)                   # 0 where unknown
    kf = known.astype(np.float64)
    ones = np.ones(window)
    # horizontal passes (along columns = u)
    hk0, hk1, hk2 = (correlate_1d(kf, w, 1) for w in (w_u0, w_u1, w_u2))
    hz0, hz1 = correlate_1d(zc, w_u0, 1), correlate_1d(zc, w_u1, 1)
    hzz = correlate_1d(zc * zc, w_u0, 1)
    # vertical passes (along rows = v)
    S, Sv, Svv = (correlate_1d(hk0, w, 0) for w in (w_u0, w_u1, w_u2))
    Su, Suv = correlate_1d(hk1, w_u0, 0), correlate_1d(hk1, w_u1, 0)
    Suu = correlate_1d(hk2, w_u0, 0)
    Sz, Svz = correlate_1d(hz0, w_u0, 0), correlate_1d(hz0, w_u1, 0)
    Suz = correlate_1d(hz1, w_u0, 0)
    Szz = correlate_1d(hzz, w_u0, 0)
    N = correlate_1d(correlate_1d(kf, ones, 1), ones, 0)   # unweighted count of known cells
    # closed-form symmetric 3x3 solve (Cramer / cofactors), fully vectorised
    a11, a12, a13, a22, a23, a33 = S, Su, Sv, Suu, Suv, Svv
    c11 = a22 * a33 - a23 * a23
    c12 = a13 * a23 - a12 * a33
    c13 = a12 * a23 - a13 * a22
    c22 = a11 * a33 - a13 * a13
    c23 = a12 * a13 - a11 * a23
    c33 = a11 * a22 - a12 * a12
    det = a11 * c11 + a12 * c12 + a13 * c13
    scale = np.maximum(S, 1e-12) ** 3
    valid = known & (N >= max(3, min_cells)) & (np.abs(det) > 1e-9 * scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), np.nan)
        t0 = (c11 * Sz + c12 * Suz + c13 * Svz) * inv
        t1 = (c12 * Sz + c22 * Suz + c23 * Svz) * inv
        t2 = (c13 * Sz + c23 * Suz + c33 * Svz) * inv
        sse = np.maximum(Szz - (t0 * Sz + t1 * Suz + t2 * Svz), 0.0)
        resid = np.sqrt(sse / np.maximum(S, 1e-12) * N / np.maximum(N - 3.0, 1.0))
    nan = np.nan
    return {"gx": np.where(valid, t1, nan), "gy": np.where(valid, t2, nan),
            "a": np.where(valid, t0 + ref, nan), "resid": np.where(valid, resid, nan),
            "n": N, "valid": valid}


def remove_speckles(h: np.ndarray, spike_threshold: float) -> np.ndarray:
    """Copy of ``h`` without unsupported cells: isolated cells (no known
    8-neighbour) and spikes / pits that differ by more than
    ``spike_threshold`` from EVERY known neighbour.  Genuine structures keep
    at least one supporting neighbour (wall faces along the wall, riser tops
    next to the upper tread, the cells of a pipe)."""
    known = np.isfinite(h)
    if not known.any():
        return h.copy()
    n_nb = np.zeros(h.shape)
    support = np.zeros(h.shape)
    with np.errstate(invalid="ignore"):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    nb = shifted(h, dr, dc, np.nan)
                    k = np.isfinite(nb)
                    n_nb += k
                    support += k & (np.abs(nb - h) <= spike_threshold)
    bad = known & ((n_nb == 0) | (support == 0))
    out = h.copy()
    out[bad] = np.nan
    return out


def nanmedian_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise median ignoring NaN (NaN for all-NaN rows).

    ``np.sort`` places NaN last, so with n finite values per row the median
    is the mean of the sorted elements (n-1)//2 and n//2.  Much faster than
    ``np.nanmedian`` for many short rows (which falls back to masked arrays)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0:
        return np.full(a.shape[0] if a.ndim else 0, np.nan)
    s = np.sort(a, axis=1)
    n = np.isfinite(a).sum(axis=1)
    out = np.full(a.shape[0], np.nan)
    ok = n > 0
    rows = np.flatnonzero(ok)
    lo = (n[ok] - 1) // 2
    hi = n[ok] // 2
    out[ok] = 0.5 * (s[rows, lo] + s[rows, hi])
    return out


def fill_holes(h: np.ndarray, radius: int, shadow_step: Optional[float] = None,
               view: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """In-paint small unknown regions that are ENCLOSED by known cells.

    Holes = closing(known, (2r+1)^2) & ~known (the closing never grows the
    known region outwards, so the unexplored part of the map stays unknown).
    Each hole cell takes the median of the known cells in its (2r+1)^2
    neighbourhood (edge preserving, unlike a mean).

    Occlusion shadows (``shadow_step`` > 0 and ``view`` given): ``view`` =
    (rows, cols) grids with the fractional grid position of the sensor that
    last observed each cell (NaN = unknown).  For a hole, the line of sight
    comes from the median sensor position of its known neighbours; the
    neighbours are split into a NEAR half (towards the sensor, offset along
    the line of sight < -0.5 cell) and a FAR half (> +0.5 cell).  If the near
    half is higher than the far half by more than ``shadow_step``, the hole
    lies in the shadow BEHIND an edge (descending stair edge, back of a box /
    pipe / block, drop-off): the line of sight passed above it, so the hidden
    surface belongs to the lower, far side and the cell takes the median of
    the FAR half.  With the plain median such a shadow is filled with the
    upper height, which moves a descending edge onto the lower tread and
    lets footholds land on the rim.  Holes in front of a rising edge (the
    near half is lower, e.g. the grazing front of an ascending stair tread)
    and smooth neighbourhoods keep the median.

    Returns (filled heights, mask of filled cells)."""
    known = np.isfinite(h)
    if radius <= 0 or not known.any() or known.all():
        return h.copy(), np.zeros(h.shape, dtype=bool)
    dil = _separable(known, radius, np.logical_or, False)
    closed = _separable(dil, radius, np.logical_and, False)
    holes = closed & ~known
    out = h.copy()
    if not holes.any():
        return out, holes
    ri, ci = np.nonzero(holes)
    pad = np.pad(h, radius, mode="constant", constant_values=np.nan)
    offs = np.arange(-radius, radius + 1)
    rr = ri[:, None, None] + radius + offs[None, :, None]
    cc = ci[:, None, None] + radius + offs[None, None, :]
    nb = pad[rr, cc].reshape(ri.size, -1)          # window index = i * W + j  (row offs[i], col offs[j])
    vals = nanmedian_rows(nb)
    if shadow_step is not None and shadow_step > 0.0 and view is not None:
        w = offs.size
        pv_r = np.pad(np.asarray(view[0], dtype=np.float64), radius, mode="constant",
                      constant_values=np.nan)[rr, cc].reshape(ri.size, -1)
        pv_c = np.pad(np.asarray(view[1], dtype=np.float64), radius, mode="constant",
                      constant_values=np.nan)[rr, cc].reshape(ri.size, -1)
        seen = np.isfinite(nb) & np.isfinite(pv_r) & np.isfinite(pv_c)
        v_r = nanmedian_rows(np.where(seen, pv_r, np.nan))      # sensor position seen by the neighbours
        v_c = nanmedian_rows(np.where(seen, pv_c, np.nan))
        d_r, d_c = ri - v_r, ci - v_c                            # line of sight sensor -> hole [cells]
        dist = np.hypot(d_r, d_c)
        has_dir = np.isfinite(dist) & (dist > 1.0)
        u_r = np.where(has_dir, d_r / np.where(has_dir, dist, 1.0), 0.0)
        u_c = np.where(has_dir, d_c / np.where(has_dir, dist, 1.0), 0.0)
        proj = u_r[:, None] * np.repeat(offs, w)[None, :] + u_c[:, None] * np.tile(offs, w)[None, :]
        near = nanmedian_rows(np.where(proj < -0.5, nb, np.nan))
        far = nanmedian_rows(np.where(proj > 0.5, nb, np.nan))
        with np.errstate(invalid="ignore"):
            shadow = has_dir & np.isfinite(near) & np.isfinite(far) & (near - far > shadow_step)
        vals = np.where(shadow, far, vals)
    ok = np.isfinite(vals)
    out[ri[ok], ci[ok]] = vals[ok]
    filled = np.zeros(h.shape, dtype=bool)
    filled[ri[ok], ci[ok]] = True
    return out, filled


# =============================================================================
#  Elevation map
# =============================================================================
class ElevationMap:
    """Robot-centric elevation grid.  See the module docstring.

    Public API used by the controller (DESIGN_SPEC section 5)::

        emap.resolution, emap.origin (x0, y0 of cell [0,0] centre), emap.shape
        emap.elevation                       # (rows, cols), NaN unknown
        ElevationMap.from_cloud_xyz(xyz)     # organised cloud -> map
        emap.height_at(x, y)                 # bilinear, NaN if unknown
        emap.layer(name)                     # see LAYER_NAMES
    """

    def __init__(self, length_x: Optional[float] = None, length_y: Optional[float] = None,
                 resolution: Optional[float] = None, center: Sequence[float] = (0.0, 0.0),
                 params=None, **overrides):
        if params is None:
            p = ElevationMapParams()
        elif isinstance(params, ElevationMapParams):
            p = params
        else:
            p = ElevationMapParams.from_dict(dict(params))
        p = p.updated(length_x=length_x, length_y=length_y, resolution=resolution, **overrides)
        if p.resolution <= 0:
            raise ValueError("resolution must be > 0")
        self.params = p
        self._res = float(p.resolution)
        cols = max(1, int(round(p.length_x / self._res)))
        rows = max(1, int(round(p.length_y / self._res)))
        self._shape = (rows, cols)
        # origin = base + offset * res  (offsets are integers -> drift-free shifts)
        self._base = (0.0, 0.0)
        self._off = [int(math.floor(center[0] / self._res + 0.5)) - cols // 2,
                     int(math.floor(center[1] / self._res + 0.5)) - rows // 2]
        self._h = np.full(self._shape, np.nan)
        self._var = np.full(self._shape, np.nan)
        self._outliers = np.zeros(self._shape, dtype=np.int16)
        self._conf = np.zeros(self._shape, dtype=bool)      # confirmed cells
        self._cand = np.full(self._shape, np.nan)           # height of the current outlier run
        self._vx = np.full(self._shape, np.nan)             # world x / y of the sensor that last
        self._vy = np.full(self._shape, np.nan)             # observed the cell (occlusion-shadow fill)
        self._t: Optional[float] = None
        self._geo: Optional[Dict[str, np.ndarray]] = None
        self._xy_cache = None

    # ------------------------------------------------------------------ basics
    @property
    def resolution(self) -> float:
        return self._res

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def origin(self) -> Tuple[float, float]:
        """World (x, y) of the CENTRE of cell [0, 0]."""
        return (self._base[0] + self._off[0] * self._res, self._base[1] + self._off[1] * self._res)

    @property
    def center(self) -> Tuple[float, float]:
        """World (x, y) of the map centre cell [rows // 2, cols // 2]."""
        x0, y0 = self.origin
        return (x0 + (self._shape[1] // 2) * self._res, y0 + (self._shape[0] // 2) * self._res)

    @property
    def length(self) -> Tuple[float, float]:
        return (self._shape[1] * self._res, self._shape[0] * self._res)

    @property
    def time(self) -> Optional[float]:
        return self._t

    @property
    def elevation(self) -> np.ndarray:
        """Fused + hole-filled heights (copy), NaN = unknown."""
        return self._geometry()["elevation"].copy()

    def __repr__(self) -> str:
        x0, y0 = self.origin
        return (f"ElevationMap(shape={self._shape}, res={self._res:.3f}, origin=({x0:.3f}, {y0:.3f}), "
                f"known={self.known_fraction():.2f})")

    def copy(self) -> "ElevationMap":
        m = ElevationMap.__new__(ElevationMap)
        m.params = self.params
        m._res, m._shape, m._base = self._res, self._shape, self._base
        m._off = list(self._off)
        m._h, m._var, m._outliers = self._h.copy(), self._var.copy(), self._outliers.copy()
        m._conf = self._conf.copy()
        m._cand = self._cand.copy()
        m._vx, m._vy = self._vx.copy(), self._vy.copy()
        m._t = self._t
        m._geo = None
        m._xy_cache = None
        return m

    def clear(self) -> None:
        self._h[:] = np.nan
        self._var[:] = np.nan
        self._outliers[:] = 0
        self._conf[:] = False
        self._cand[:] = np.nan
        self._vx[:] = np.nan
        self._vy[:] = np.nan
        self._geo = None

    def known_fraction(self) -> float:
        """Fraction of cells with a (confirmed) height."""
        return float(self._valid_raw().mean())

    def _valid_raw(self) -> np.ndarray:
        known = np.isfinite(self._h)
        return known & self._conf if self.params.confirm_observations else known

    # -------------------------------------------------------------- indexing
    def world_to_index(self, x, y):
        """World coordinates -> nearest (row, col) integer indices (may be out of range)."""
        x0, y0 = self.origin
        c = np.floor((np.asarray(x, dtype=float) - x0) / self._res + 0.5).astype(int)
        r = np.floor((np.asarray(y, dtype=float) - y0) / self._res + 0.5).astype(int)
        return r, c

    def index_to_world(self, r, c):
        x0, y0 = self.origin
        return x0 + np.asarray(c) * self._res, y0 + np.asarray(r) * self._res

    def in_bounds(self, r, c):
        r, c = np.asarray(r), np.asarray(c)
        return (r >= 0) & (r < self._shape[0]) & (c >= 0) & (c < self._shape[1])

    def grid_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """(X, Y) world coordinates of every cell centre, each (rows, cols)."""
        key = (self.origin, self._shape)
        if self._xy_cache is None or self._xy_cache[0] != key:
            x0, y0 = self.origin
            xs = x0 + np.arange(self._shape[1]) * self._res
            ys = y0 + np.arange(self._shape[0]) * self._res
            X, Y = np.meshgrid(xs, ys)
            self._xy_cache = (key, X, Y)
        return self._xy_cache[1], self._xy_cache[2]

    # ------------------------------------------------------------ recentring
    def recenter(self, x: float, y: float) -> bool:
        """Shift the window by whole cells so that (x, y) lies in the centre
        cell.  Returns True when the data moved."""
        rows, cols = self._shape
        x0, y0 = self.origin
        dc = int(math.floor((x - x0) / self._res + 0.5)) - cols // 2
        dr = int(math.floor((y - y0) / self._res + 0.5)) - rows // 2
        if dc == 0 and dr == 0:
            return False
        self._h = shifted(self._h, dr, dc, np.nan)
        self._var = shifted(self._var, dr, dc, np.nan)
        self._outliers = shifted(self._outliers, dr, dc, 0)
        self._conf = shifted(self._conf, dr, dc, False)
        self._cand = shifted(self._cand, dr, dc, np.nan)
        self._vx = shifted(self._vx, dr, dc, np.nan)
        self._vy = shifted(self._vy, dr, dc, np.nan)
        self._off[0] += dc
        self._off[1] += dr
        self._geo = None
        return True

    # ---------------------------------------------------------------- fusion
    def predict(self, t: float) -> int:
        """Time update: variance grows by ``variance_growth_rate * dt``; cells
        whose variance exceeds ``max_variance`` are forgotten.  Returns the
        number of forgotten cells."""
        t = float(t)
        if self._t is None:
            self._t = t
            return 0
        dt = t - self._t
        if dt < 0.0:
            if dt < -1.0:          # clock reset (simulation restarted)
                self._t = t
            return 0
        self._t = t
        p = self.params
        known = np.isfinite(self._h)
        if p.variance_growth_rate > 0.0 and dt > 0.0:
            self._var[known] += p.variance_growth_rate * dt
        forget = known & (self._var > p.max_variance)
        n = int(forget.sum())
        if n:
            self._h[forget] = np.nan
            self._var[forget] = np.nan
            self._outliers[forget] = 0
            self._conf[forget] = False
            self._cand[forget] = np.nan
            self._vx[forget] = np.nan
            self._vy[forget] = np.nan
            self._geo = None
        return n

    def integrate(self, points: np.ndarray, sensor_origin: Optional[Sequence[float]] = None,
                  t: Optional[float] = None, ranges: Optional[np.ndarray] = None) -> int:
        """Fuse world-frame points (N, 3).

        sensor_origin (or explicit per-point ``ranges``) drives the
        range-dependent measurement variance.  ``t`` (seconds) applies the
        time update first.  Returns the number of cells updated."""
        if t is not None:
            self.predict(t)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        fin = np.isfinite(pts).all(axis=1)
        if ranges is not None:
            rng = np.asarray(ranges, dtype=np.float64).reshape(-1)[fin]
        elif sensor_origin is not None:
            rng = np.linalg.norm(pts[fin] - np.asarray(sensor_origin, dtype=float)[None, :3], axis=1)
        else:
            rng = np.ones(int(fin.sum()))
        pts = pts[fin]
        if pts.shape[0] == 0:
            return 0
        rows, cols = self._shape
        r, c = self.world_to_index(pts[:, 0], pts[:, 1])
        inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        if not inside.any():
            return 0
        z = pts[inside, 2]
        rng = rng[inside]
        lin = r[inside] * cols + c[inside]
        cells, z_meas, var_meas, vertical, counts = self._cell_measurements(lin, z, rng)
        self._fuse(cells, z_meas, var_meas, vertical, counts)
        if sensor_origin is not None:
            # viewpoint of this observation (line of sight for the occlusion-shadow fill)
            ox, oy = float(sensor_origin[0]), float(sensor_origin[1])
            if math.isfinite(ox) and math.isfinite(oy):
                rows_i, cols_i = np.divmod(cells, cols)
                self._vx[rows_i, cols_i] = ox
                self._vy[rows_i, cols_i] = oy
        self._geo = None
        return int(cells.size)

    def _cell_measurements(self, lin: np.ndarray, z: np.ndarray, rng: np.ndarray):
        """Reduce all points of a cell to one robust measurement + variance."""
        p = self.params
        # sort by (cell, height) with ONE argsort of a combined key: integer cell
        # index + height normalised into [0, 0.5) as the fractional part (same
        # order as np.lexsort((z, lin)), ~3x faster; float64 keeps ~1e-11 m
        # height resolution for the 150 x 150 grid)
        z_min = float(z.min())
        span = max(float(z.max()) - z_min, 1e-9)
        order = np.argsort(lin.astype(np.float64) + (z - z_min) * (0.4999 / span))
        lin_s, z_s, r_s = lin[order], z[order], rng[order]
        cells, start, counts = np.unique(lin_s, return_index=True, return_counts=True)
        last = start + counts - 1
        med = 0.5 * (z_s[start + (counts - 1) // 2] + z_s[start + counts // 2])
        q25 = z_s[start + np.floor(0.25 * (counts - 1)).astype(int)]
        q75 = z_s[start + np.floor(0.75 * (counts - 1)).astype(int)]
        spread = z_s[last] - z_s[start]
        r_mean = np.add.reduceat(r_s, start) / counts
        sigma = p.sensor_noise_base + p.sensor_noise_range_coeff * r_mean ** 2
        var_pt = sigma ** 2
        z_meas = med
        # in-cell spread clearly larger than the sensor noise -> vertical structure
        # (wall face, riser, block edge); needs >= 3 points so a lone spike cannot trigger it
        vertical = (spread > np.maximum(p.bias_spread, 4.0 * sigma)) & (counts >= 3)
        if p.max_height_bias and vertical.any():
            qh = z_s[start + np.floor(p.bias_quantile * (counts - 1)).astype(int)]
            z_meas = np.where(vertical, qh, med)
        robust_std = (q75 - q25) / 1.349
        n_eff = np.minimum(counts, max(1, p.max_points_per_cell)).astype(np.float64)
        var_m = (var_pt / n_eff
                 + p.spread_variance_weight * np.maximum(robust_std ** 2 - var_pt, 0.0)
                 + p.min_variance)
        return cells, z_meas, var_m, vertical, counts

    def _fuse(self, cells: np.ndarray, z: np.ndarray, v: np.ndarray,
              vertical: Optional[np.ndarray] = None, counts: Optional[np.ndarray] = None) -> None:
        """Vectorised 1-D Kalman update with Mahalanobis gating.

        ``vertical`` marks cells whose points spread vertically in this frame.
        Such a measurement only sees PART of a vertical face (the camera's
        field of view / occlusion cuts it), i.e. it is a lower bound of the
        true cell height; a higher one raises the cell immediately instead of
        being averaged with earlier, lower views (max fusion).
        ``counts`` (points per cell) drives the confirmation logic."""
        p = self.params
        rows_i, cols_i = np.divmod(cells, self._shape[1])
        h = self._h[rows_i, cols_i]
        var = self._var[rows_i, cols_i]
        conf = self._conf[rows_i, cols_i]
        strong = (np.ones(z.shape, dtype=bool) if counts is None
                  else np.asarray(counts) >= max(1, p.min_points_confirm))
        if not p.confirm_observations:
            conf = np.ones(z.shape, dtype=bool)
        # signed run counter: +n = n consecutive LOWER outliers, -n = n consecutive HIGHER
        oc = self._outliers[rows_i, cols_i].astype(np.int32)
        new = ~np.isfinite(h) | ~np.isfinite(var)
        with np.errstate(invalid="ignore"):
            d2 = np.where(new, 0.0, (z - h) ** 2 / (var + v))
        inlier = ~new & (d2 <= p.mahalanobis_gate ** 2)
        higher = ~new & ~inlier & (z > h)
        lower = ~new & ~inlier & ~higher
        cand = self._cand[rows_i, cols_i]
        with np.errstate(invalid="ignore"):
            consistent = np.isfinite(cand) & (np.abs(z - cand) <= np.maximum(
                p.outlier_consistency, p.mahalanobis_gate * np.sqrt(2.0 * v)))
        oc_next = np.where(lower, np.where((oc > 0) & consistent, oc + 1, 1),
                           np.where(higher, np.where((oc < 0) & consistent, oc - 1, -1), 0))
        raise_now = np.zeros(z.shape, dtype=bool)
        if vertical is not None and p.raise_on_vertical:
            raise_now = ~new & np.asarray(vertical, dtype=bool) & (z > h)
        reset = (new | raise_now | (lower & (oc_next >= max(1, p.outlier_reset_count)))
                 | (higher & (-oc_next >= max(1, p.higher_outlier_count))))
        # unconfirmed cells: any inconsistent measurement simply re-initialises them
        unconf = ~new & ~conf
        reinit = unconf & ~inlier
        reset = reset | reinit
        inlier = inlier & ~raise_now
        h_out = h.copy()
        v_out = var.copy()
        # Kalman gain k = P / (P + R);  h+ = h + k (z - h);  P+ = (1 - k) P
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(inlier, var / (var + v), 0.0)
        h_out = np.where(inlier, h + k * (z - h), h_out)
        v_out = np.where(inlier, (1.0 - k) * var, v_out)
        h_out = np.where(reset, z, h_out)
        v_out = np.where(reset, v, v_out)
        v_out = np.maximum(v_out, p.min_variance)
        oc_out = np.where(reset | inlier, 0, oc_next)
        cand_out = np.where((lower | higher) & ~reset, z, np.nan)
        # confirmed after this update: it was confirmed (and not re-initialised as an
        # unconfirmed cell), a second consistent look, or a strong multi-point measurement
        conf_out = (conf & ~new & ~reinit) | (unconf & inlier) | strong
        self._conf[rows_i, cols_i] = conf_out
        self._cand[rows_i, cols_i] = cand_out
        self._h[rows_i, cols_i] = h_out
        self._var[rows_i, cols_i] = v_out
        self._outliers[rows_i, cols_i] = np.clip(oc_out, -32000, 32000).astype(np.int16)

    # ---------------------------------------------------------------- layers
    def update_layers(self) -> None:
        """Force computation of the derived layers (normally lazy)."""
        self._geometry()

    def _geometry(self) -> Dict[str, np.ndarray]:
        if self._geo is not None:
            return self._geo
        p = self.params
        res = self._res
        base = np.where(self._valid_raw(), self._h, np.nan)
        if p.speckle_filter:
            base = remove_speckles(base, p.spike_threshold)
        x0, y0 = self.origin
        view = ((self._vy - y0) / res, (self._vx - x0) / res)      # sensor positions in grid units
        elev, filled = fill_holes(base, p.hole_fill_radius, p.shadow_fill_step, view)
        known = np.isfinite(elev)
        # slope: Sobel-weighted plane fit
        sw = max(3, p.slope_window | 1)
        fs = plane_fit(elev, sw, binomial_weights(sw), min_cells=min(p.min_fit_cells, sw * sw))
        gx = fs["gx"] / res
        gy = fs["gy"] / res
        slope = np.arctan(np.hypot(gx, gy))
        # roughness: residual std of a uniform plane fit
        rw = max(3, p.roughness_window | 1)
        fr = plane_fit(elev, rw, None, min_cells=p.min_fit_cells)
        rough = fr["resid"]
        # step: max - min in a foot-sized window
        stw = max(1, p.step_window | 1)
        step = max_filter(elev, stw) - min_filter(elev, stw)
        step = np.where(known, step, np.nan)
        trav = self._traversability(slope, step, rough, known)
        self._geo = {"elevation": elev, "filled": filled, "slope": slope, "roughness": rough,
                     "step": step, "traversability": trav, "gradient_x": gx, "gradient_y": gy}
        return self._geo

    def _traversability(self, slope, step, rough, known) -> np.ndarray:
        p = self.params
        s_slope = np.clip(np.nan_to_num(slope, nan=0.0) / max(p.slope_crit, 1e-6), 0.0, 1.0)
        s_step = np.clip(np.nan_to_num(step, nan=0.0) / max(p.step_crit, 1e-6), 0.0, 1.0)
        s_rough = np.clip(np.nan_to_num(rough, nan=0.0) / max(p.roughness_crit, 1e-6), 0.0, 1.0)
        wsum = max(p.w_slope + p.w_step + p.w_roughness, 1e-9)
        trav = 1.0 - (p.w_slope * s_slope + p.w_step * s_step + p.w_roughness * s_rough) / wsum
        planar = np.nan_to_num(rough, nan=np.inf) < p.planar_roughness
        hard = (np.nan_to_num(step, nan=0.0) >= p.step_crit) | \
            (planar & (np.nan_to_num(slope, nan=0.0) >= p.slope_crit))
        trav = np.where(hard, 0.0, trav)
        return np.where(known, np.clip(trav, 0.0, 1.0), np.nan)

    def _variance_layer(self) -> np.ndarray:
        geo = self._geometry()
        used = np.isfinite(geo["elevation"]) & ~geo["filled"]      # cells that made it into the map
        var = np.where(used, self._var, np.nan)
        filled = geo["filled"]
        if filled.any():
            known = np.isfinite(var)
            w = self.params.hole_fill_radius * 2 + 1
            num = box_sum(np.where(known, self._var, 0.0), w)
            den = box_sum(known.astype(np.float64), w)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_var = num / den
            var[filled] = np.nan_to_num(mean_var[filled], nan=self.params.default_variance) \
                + self.params.fill_variance
        return var

    def layer(self, name: str) -> np.ndarray:
        """Copy of a layer: 'elevation', 'variance', 'slope' [rad],
        'roughness' [m], 'step' [m], 'traversability' [0..1] (NaN unknown)
        or the extras 'elevation_raw', 'variance_raw', 'gradient_x',
        'gradient_y', 'filled'."""
        if name == "variance":
            return self._variance_layer()
        if name == "elevation_raw":
            return self._h.copy()
        if name == "confirmed":
            return self._valid_raw().copy()
        if name == "variance_raw":
            return self._var.copy()
        geo = self._geometry()
        if name in geo:
            return geo[name].copy()
        raise KeyError(f"unknown elevation map layer '{name}' (have {LAYER_NAMES + EXTRA_LAYER_NAMES})")

    def layers(self) -> Dict[str, np.ndarray]:
        """All main layers (copies) as a dict."""
        return {n: self.layer(n) for n in LAYER_NAMES}

    # --------------------------------------------------------------- queries
    def height_at(self, x, y):
        """Bilinear height at world (x, y); scalar or array input.

        Unknown neighbours are dropped and the remaining weights renormalised
        as long as the NEAREST cell is known; otherwise NaN (also outside)."""
        return self.sample(self._geometry()["elevation"], x, y)

    def sample(self, grid: np.ndarray, x, y):
        """Bilinear NaN-aware sampling of any (rows, cols) grid of this map."""
        scalar = np.ndim(x) == 0 and np.ndim(y) == 0
        xa = np.atleast_1d(np.asarray(x, dtype=np.float64))
        ya = np.atleast_1d(np.asarray(y, dtype=np.float64))
        xa, ya = np.broadcast_arrays(xa, ya)
        rows, cols = self._shape
        x0, y0 = self.origin
        fc = (xa - x0) / self._res
        fr = (ya - y0) / self._res
        c0 = np.floor(fc).astype(int)
        r0 = np.floor(fr).astype(int)
        tx = fc - c0
        ty = fr - r0
        num = np.zeros(xa.shape)
        den = np.zeros(xa.shape)
        for dr, dc, w in ((0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)),
                          (1, 0, (1 - tx) * ty), (1, 1, tx * ty)):
            rr, cc = r0 + dr, c0 + dc
            ok = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
            val = np.full(xa.shape, np.nan)
            val[ok] = grid[rr[ok], cc[ok]]
            good = np.isfinite(val)
            num += np.where(good, w * np.nan_to_num(val), 0.0)
            den += np.where(good, w, 0.0)
        rn = np.floor(fr + 0.5).astype(int)
        cn = np.floor(fc + 0.5).astype(int)
        okn = (rn >= 0) & (rn < rows) & (cn >= 0) & (cn < cols)
        near_known = np.zeros(xa.shape, dtype=bool)
        near_known[okn] = np.isfinite(grid[rn[okn], cn[okn]])
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(near_known & (den > 1e-12), num / den, np.nan)
        return float(out[0]) if scalar else out

    def cell_value(self, name: str, x: float, y: float) -> float:
        """Nearest-cell value of a layer at world (x, y) (NaN outside)."""
        r, c = self.world_to_index(x, y)
        if not self.in_bounds(r, c):
            return float("nan")
        geo = self._geometry()
        if name in geo and isinstance(name, str):
            return float(geo[name][r, c])
        return float(self.layer(name)[r, c])     # variance / raw / confirmed layers (KeyError if unknown)

    # ------------------------------------------------------------- transport
    def to_cloud_xyz(self) -> np.ndarray:
        """Organised (rows, cols, 3) float32 cloud: x along cols, y along rows,
        z = elevation (NaN unknown), world frame - the /elevation_map/points
        payload."""
        X, Y = self.grid_xy()
        return np.stack([X, Y, self._geometry()["elevation"]], -1).astype(np.float32)

    def occupancy_data(self) -> np.ndarray:
        """Traversability as an OccupancyGrid payload (rows, cols) int8:
        0 = easy ... 100 = untraversable, -1 unknown (row 0 = lowest y)."""
        trav = self._geometry()["traversability"]
        cost = np.where(np.isfinite(trav), np.round((1.0 - np.nan_to_num(trav)) * 100.0), -1)
        return np.clip(cost, -1, 100).astype(np.int8)

    def occupancy_origin(self) -> Tuple[float, float]:
        """Lower-left CORNER of the grid (OccupancyGrid info.origin)."""
        x0, y0 = self.origin
        return x0 - 0.5 * self._res, y0 - 0.5 * self._res

    @classmethod
    def from_elevation(cls, elevation: np.ndarray, origin: Sequence[float], resolution: float,
                       variance: Optional[np.ndarray] = None, params=None, **overrides) -> "ElevationMap":
        """Wrap an existing height grid (rows ~ y, cols ~ x, NaN unknown)."""
        elev = np.array(elevation, dtype=np.float64)
        if elev.ndim != 2:
            raise ValueError("elevation must be 2-D (rows, cols)")
        rows, cols = elev.shape
        res = float(resolution)
        m = cls(length_x=cols * res, length_y=rows * res, resolution=res, params=params, **overrides)
        if m._shape != (rows, cols):   # rounding of length/res -> force exact shape
            m._shape = (rows, cols)
        m._base = (float(origin[0]), float(origin[1]))
        m._off = [0, 0]
        m._h = elev
        known = np.isfinite(elev)
        if variance is None:
            m._var = np.where(known, m.params.default_variance, np.nan)
        else:
            m._var = np.where(known, np.asarray(variance, dtype=np.float64), np.nan)
        m._outliers = np.zeros((rows, cols), dtype=np.int16)
        m._conf = known.copy()
        m._cand = np.full((rows, cols), np.nan)
        m._vx = np.full((rows, cols), np.nan)
        m._vy = np.full((rows, cols), np.nan)
        m._geo = None
        return m

    @classmethod
    def from_cloud_xyz(cls, xyz: np.ndarray, params=None, compute_layers: bool = True,
                       **overrides) -> "ElevationMap":
        """Rebuild a map (and all derived layers) from the organised
        (rows, cols, 3) cloud published on /elevation_map/points.

        Resolution and origin are recovered from the x / y coordinates (robust
        to float32 rounding and to NaN coordinates).  ``compute_layers``
        (default) evaluates the derived layers right away, so the cost is paid
        in the caller's thread (the controller's map callback) and not by the
        first ``height_at`` / foothold query inside a control loop.  Queries
        (``height_at``, ``layer``, the foothold planner) only read the map
        (plus idempotent caches), so it can be shared between threads."""
        a = np.asarray(xyz, dtype=np.float64)
        if a.ndim != 3 or a.shape[2] < 3:
            raise ValueError("from_cloud_xyz expects an ORGANISED (rows, cols, 3) cloud")
        rows, cols = a.shape[:2]
        X, Y, Z = a[..., 0], a[..., 1], a[..., 2]
        base = ElevationMapParams() if params is None else (
            params if isinstance(params, ElevationMapParams) else ElevationMapParams.from_dict(dict(params)))
        res = _estimate_resolution(X, Y, base.resolution)
        ci = np.arange(cols)[None, :]
        ri = np.arange(rows)[:, None]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x0 = float(np.nanmedian(X - ci * res))
            y0 = float(np.nanmedian(Y - ri * res))
        if not (math.isfinite(x0) and math.isfinite(y0)):
            raise ValueError("cloud has no valid x/y coordinates")
        m = cls.from_elevation(Z, (x0, y0), res, params=base, **overrides)
        if compute_layers:
            m.update_layers()
        return m


def _estimate_resolution(X: np.ndarray, Y: np.ndarray, default: float) -> float:
    """Cell size from the coordinate spans (float32-robust, rounded to 1 um)."""
    cands = []
    rows, cols = X.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if cols > 1:
            span = (X[:, -1] - X[:, 0]) / (cols - 1)
            span = span[np.isfinite(span)]
            if span.size:
                cands.append(float(np.median(span)))
            else:
                d = np.diff(X, axis=1)
                d = d[np.isfinite(d)]
                if d.size:
                    cands.append(float(np.median(d)))
        if rows > 1:
            span = (Y[-1, :] - Y[0, :]) / (rows - 1)
            span = span[np.isfinite(span)]
            if span.size:
                cands.append(float(np.median(span)))
            else:
                d = np.diff(Y, axis=0)
                d = d[np.isfinite(d)]
                if d.size:
                    cands.append(float(np.median(d)))
    cands = [c for c in cands if c > 0 and math.isfinite(c)]
    if not cands:
        return float(default)
    return round(float(np.mean(cands)), 6)
