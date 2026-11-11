"""
Terrain-aware foothold selection on the elevation map (numpy only).

For a nominal foothold (from the Raibert heuristic of the controller) every
map cell whose centre lies within ``search_radius`` - plus the exact nominal
point itself - is a candidate.  Cost of a candidate p (lower is better)::

    J(p) = w_slope * slope(p)            [rad]
         + w_rough * roughness(p)        [m]
         + w_step  * step(p)             [m]   max-min height in a foot-sized window
                                               (foot_window x foot_window cells, default
                                               3 x 3 = 0.12 m: the foot sphere + placement error)
         + w_edge  * edge_proximity(p)   [0..1] 1 at an edge, 0 beyond edge_margin
         + w_dist  * |p - nominal|       [m]
         + unknown_penalty               (only if unknown cells are allowed)

Candidates are REJECTED when the cell is untraversable
(traversability < min_traversability), the foot window contains a step
higher than ``max_step`` (= too close to an edge), the slope exceeds
``max_slope``, the roughness exceeds ``max_roughness`` or the cell lies more
than ``max_relative_height`` above the lowest known terrain in a square of
half-size ``relative_height_radius`` (the flat top of a tall box / drum is locally
perfect but unreachable for the leg; one 0.17 m stair riser passes).  If nothing within
the radius is acceptable, a second search with ``expand_factor * radius`` is
tried before giving up (valid = False, nominal returned).

``edge_proximity`` uses edge cells found directly in the local elevation patch
(4-neighbour height jump > ``edge_height``) and gives a smooth gradient away
from risers / pipe flanks beyond the foot-sized ``step`` window.

The nominal point is a candidate with zero displacement, so on flat ground the
nominal foothold is returned EXACTLY (no cell quantisation jitter).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .elevation_map import max_filter, min_filter

_ALIASES = {
    "w_roughness": "w_rough", "w_displacement": "w_dist", "w_distance": "w_dist",
    "w_unknown": "unknown_penalty", "w_edge_proximity": "w_edge", "radius": "search_radius",
}


class FootholdPlanner:
    """Foothold search (DESIGN_SPEC section 5 API)::

        planner = FootholdPlanner(search_radius=0.10, w_slope=1.0, ...)
        (x, y, z), score, valid = planner.select(emap, nominal_xy, fallback_z)

    ``score`` is the cost of the chosen foothold (>= 0, lower = better,
    ``inf`` when valid is False).  ``z`` is the TERRAIN SURFACE height (the
    foot sphere centre is z + foot radius).  When valid is False the nominal
    xy is returned with the map height there if known, else ``fallback_z``.
    Unknown keyword arguments are ignored (forward compatible)."""

    def __init__(self, search_radius: float = 0.10, w_slope: float = 1.0, w_rough: float = 20.0,
                 w_step: float = 8.0, w_edge: float = 0.6, w_dist: float = 4.0,
                 unknown_penalty: float = 1.0, min_traversability: float = 0.25,
                 max_step: float = 0.05, max_slope: float = 0.60, max_roughness: float = 0.04,
                 edge_height: float = 0.04, edge_margin: float = 0.10,
                 max_relative_height: float = 0.25, relative_height_radius: float = 0.25,
                 foot_window: int = 3, max_local_rise: float = 0.03, rise_window: int = 5,
                 allow_unknown: bool = False, expand_factor: float = 1.6, **weights):
        kw = {}
        for k, v in weights.items():
            kw[_ALIASES.get(k, k)] = v
        g = dict(search_radius=search_radius, w_slope=w_slope, w_rough=w_rough, w_step=w_step,
                 w_edge=w_edge, w_dist=w_dist, unknown_penalty=unknown_penalty,
                 min_traversability=min_traversability, max_step=max_step, max_slope=max_slope,
                 max_roughness=max_roughness, edge_height=edge_height, edge_margin=edge_margin,
                 max_relative_height=max_relative_height, foot_window=foot_window,
                 max_local_rise=max_local_rise, rise_window=rise_window,
                 relative_height_radius=relative_height_radius,
                 allow_unknown=allow_unknown, expand_factor=expand_factor)
        for k in list(kw):
            if k in g:
                g[k] = kw.pop(k)
        self.search_radius = max(0.0, float(g["search_radius"]))
        self.w_slope = float(g["w_slope"])
        self.w_rough = float(g["w_rough"])
        self.w_step = float(g["w_step"])
        self.w_edge = float(g["w_edge"])
        self.w_dist = float(g["w_dist"])
        self.unknown_penalty = float(g["unknown_penalty"])
        self.min_traversability = float(g["min_traversability"])
        self.max_step = float(g["max_step"])
        self.max_slope = float(g["max_slope"])
        self.max_roughness = float(g["max_roughness"])
        self.edge_height = float(g["edge_height"])
        self.edge_margin = max(1e-6, float(g["edge_margin"]))
        self.max_relative_height = float(g["max_relative_height"])
        self.foot_window = max(1, int(g["foot_window"]) | 1)
        self.max_local_rise = float(g["max_local_rise"])
        self.rise_window = max(1, int(g["rise_window"]) | 1)
        self.relative_height_radius = max(0.0, float(g["relative_height_radius"]))
        self.allow_unknown = bool(g["allow_unknown"])
        self.expand_factor = max(1.0, float(g["expand_factor"]))
        self.ignored = kw              # unknown options (kept for inspection)
        self.last_debug: dict = {}
        self._offset_cache: dict = {}

    # ------------------------------------------------------------------ API
    def select(self, emap, nominal_xy: Sequence[float], fallback_z: float):
        """Best foothold near ``nominal_xy`` -> (np.array([x, y, z]), score, valid)."""
        nx, ny = float(nominal_xy[0]), float(nominal_xy[1])
        fz = float(fallback_z) if fallback_z is not None else float("nan")
        if emap is None:
            self.last_debug = {"reason": "no_map"}
            return np.array([nx, ny, fz]), math.inf, False
        best = self._search(emap, nx, ny, fz, self.search_radius)
        if best is None and self.expand_factor > 1.0 and self.search_radius > 0.0:
            best = self._search(emap, nx, ny, fz, self.search_radius * self.expand_factor)
        if best is None:
            z = emap.height_at(nx, ny)
            z = z if math.isfinite(z) else fz
            return np.array([nx, ny, z]), math.inf, False
        (x, y, z), cost, valid = best
        return np.array([x, y, z]), float(cost), bool(valid)

    def select_many(self, emap, nominals: np.ndarray, fallback_z) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vector convenience: nominals (K, >=2), fallback_z scalar or (K,)."""
        nom = np.asarray(nominals, dtype=float).reshape(-1, np.shape(nominals)[-1])
        fz = np.broadcast_to(np.asarray(fallback_z, dtype=float), (nom.shape[0],))
        pts, scores, valid = [], [], []
        for k in range(nom.shape[0]):
            p, s, v = self.select(emap, nom[k, :2], fz[k])
            pts.append(p); scores.append(s); valid.append(v)
        return np.array(pts), np.array(scores), np.array(valid, dtype=bool)

    # ------------------------------------------------------------- internals
    def _context(self, emap) -> dict:
        """Per-map context layers, cached inside the map's layer cache (so they
        are recomputed exactly when the heights change): the edge mask (any
        4-neighbour height jump > edge_height) and the local minimum height in
        a square of half-size relative_height_radius (reachability test)."""
        geo = emap._geometry()
        key = ("_foothold_ctx", round(self.edge_height, 6), round(self.relative_height_radius, 6),
               self.foot_window, self.rise_window)
        ctx = geo.get(key)
        if ctx is None:
            E = geo["elevation"]
            edge = np.zeros(E.shape, dtype=bool)
            with np.errstate(invalid="ignore"):
                dv = np.abs(np.diff(E, axis=0)) > self.edge_height
                dh = np.abs(np.diff(E, axis=1)) > self.edge_height
            edge[:-1, :] |= dv
            edge[1:, :] |= dv
            edge[:, :-1] |= dh
            edge[:, 1:] |= dh
            rr = int(round(self.relative_height_radius / emap.resolution))
            lmin = min_filter(E, 2 * rr + 1) if rr > 0 else E
            foot_step = np.where(np.isfinite(E), max_filter(E, self.foot_window)
                                 - min_filter(E, self.foot_window), np.nan)
            # height above the lowest cell of the rise window, minus what a plane with
            # the local (Sobel) gradient would produce there -> slope invariant
            half = (self.rise_window // 2) * emap.resolution
            planar = (np.abs(np.nan_to_num(geo["gradient_x"])) + np.abs(np.nan_to_num(geo["gradient_y"]))) * half
            rise = E - min_filter(E, self.rise_window) - planar
            ctx = {"edge": edge, "local_min": lmin, "foot_step": foot_step, "rise": rise}
            geo[key] = ctx
        return ctx

    def _offsets(self, radius: float, res: float):
        """Integer cell offsets that can lie within ``radius`` of a point
        anywhere inside the centre cell (cached per radius / resolution)."""
        key = (round(radius, 6), round(res, 6))
        off = self._offset_cache.get(key)
        if off is None:
            n = int(math.ceil(radius / res)) + 1
            dr, dc = np.mgrid[-n:n + 1, -n:n + 1]
            keep = np.hypot(dr, dc) * res <= radius + res * 0.7072   # + half cell diagonal
            off = (dr[keep].astype(int), dc[keep].astype(int))
            self._offset_cache[key] = off
        return off

    def _search(self, emap, nx: float, ny: float, fz: float, radius: float):
        geo = emap._geometry()            # cached layers (read only, no copies)
        ctx = self._context(emap)
        res = emap.resolution
        rows, cols = emap.shape
        x0, y0 = emap.origin
        rn, cn = (int(v) for v in emap.world_to_index(nx, ny))
        dr, dc = self._offsets(radius, res)
        rr, cc = rn + dr, cn + dc
        inb = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        rr, cc = rr[inb], cc[inb]
        X = x0 + cc * res
        Y = y0 + rr * res
        d = np.hypot(X - nx, Y - ny)
        cand = d <= radius + 1e-9
        cx, cy, cd, cr, ccol = X[cand], Y[cand], d[cand], rr[cand], cc[cand]
        cz = geo["elevation"][cr, ccol]
        # the exact nominal point as an extra candidate (nearest-cell layers)
        if 0 <= rn < rows and 0 <= cn < cols:
            cx = np.append(cx, nx); cy = np.append(cy, ny); cd = np.append(cd, 0.0)
            cr = np.append(cr, rn); ccol = np.append(ccol, cn)
            cz = np.append(cz, emap.height_at(nx, ny))
        if cx.size == 0:
            self.last_debug = {"reason": "outside_map"}
            return None
        slope = geo["slope"][cr, ccol]
        rough = geo["roughness"][cr, ccol]
        step = ctx["foot_step"][cr, ccol]
        trav = geo["traversability"][cr, ccol]
        known = np.isfinite(cz) & np.isfinite(geo["elevation"][cr, ccol])

        # edge proximity from the edge cells around the nominal
        m = int(math.ceil((radius + self.edge_margin) / res)) + 1
        r0, r1 = max(0, rn - m), min(rows, rn + m + 1)
        c0, c1 = max(0, cn - m), min(cols, cn + m + 1)
        er, ec = np.nonzero(ctx["edge"][r0:r1, c0:c1]) if (r0 < r1 and c0 < c1) else ((), ())
        if len(er):
            ex = x0 + (np.asarray(ec) + c0) * res
            ey = y0 + (np.asarray(er) + r0) * res
            de = np.min(np.hypot(cx[:, None] - ex[None, :], cy[:, None] - ey[None, :]), axis=1)
            edge_prox = np.clip(1.0 - de / self.edge_margin, 0.0, 1.0)
        else:
            edge_prox = np.zeros(cx.shape)
        s0, r0_, st0 = (np.nan_to_num(v, nan=0.0) for v in (slope, rough, step))
        cost = (self.w_slope * s0 + self.w_rough * r0_ + self.w_step * st0
                + self.w_edge * edge_prox + self.w_dist * cd)
        # reachability: height above the lowest known terrain close by
        too_high = np.zeros(cx.shape, dtype=bool)
        if self.relative_height_radius > 0.0 and math.isfinite(self.max_relative_height):
            lmin = ctx["local_min"][cr, ccol]
            with np.errstate(invalid="ignore"):
                too_high = np.isfinite(lmin) & (cz - lmin > self.max_relative_height)
        # narrow raised features (pipe tops, small blocks, the outer rim of a tread):
        # higher than the lowest terrain within the rise window
        with np.errstate(invalid="ignore"):
            narrow = np.nan_to_num(ctx["rise"][cr, ccol], nan=0.0) > self.max_local_rise
        with np.errstate(invalid="ignore"):
            reject = known & ((np.nan_to_num(trav, nan=1.0) < self.min_traversability)
                              | (st0 > self.max_step) | (s0 > self.max_slope)
                              | (r0_ > self.max_roughness) | too_high | narrow)
        ok = known & ~reject
        if self.allow_unknown and math.isfinite(fz):
            unk = ~known
            cost = np.where(unk, self.unknown_penalty + self.w_dist * cd, cost)
            cz = np.where(unk, fz, cz)
            ok = ok | unk
        self.last_debug = {"radius": radius, "candidates": int(cx.size), "known": int(known.sum()),
                           "rejected": int(reject.sum()), "edges": int(len(er))}
        if not ok.any():
            return None
        cost = np.where(ok, cost, np.inf)
        k = int(np.argmin(cost))
        return (float(cx[k]), float(cy[k]), float(cz[k])), float(cost[k]), bool(known[k])


def max_height_along(emap, p0_xy: Sequence[float], p1_xy: Sequence[float],
                     margin: Optional[float] = None) -> float:
    """Highest known terrain point along the segment p0 -> p1 (swing apex
    helper), sampled every half cell and ``margin`` (default one cell) to
    either side of the segment.  NaN-safe: unknown cells are ignored; returns
    NaN when nothing along the path is known (or emap is None)."""
    if emap is None:
        return float("nan")
    elev = emap._geometry()["elevation"]
    res = emap.resolution
    p0 = np.asarray(p0_xy, dtype=float)[:2]
    p1 = np.asarray(p1_xy, dtype=float)[:2]
    seg = p1 - p0
    length = float(np.hypot(seg[0], seg[1]))
    n = max(2, int(math.ceil(length / (0.5 * res))) + 1)
    t = np.linspace(0.0, 1.0, n)
    pts = p0[None, :] + t[:, None] * seg[None, :]
    m = res if margin is None else max(0.0, float(margin))
    if length > 1e-9 and m > 0.0:
        normal = np.array([-seg[1], seg[0]]) / length
        lat = np.arange(-m, m + 1e-9, 0.5 * res)
        pts = (pts[:, None, :] + lat[None, :, None] * normal[None, None, :]).reshape(-1, 2)
    elif m > 0.0:   # degenerate segment: a small square around the point
        lat = np.arange(-m, m + 1e-9, 0.5 * res)
        gx, gy = np.meshgrid(lat, lat)
        pts = (p0[None, :] + np.stack([gx.ravel(), gy.ravel()], -1))
    r, c = emap.world_to_index(pts[:, 0], pts[:, 1])
    ok = emap.in_bounds(r, c)
    if not ok.any():
        return float("nan")
    vals = elev[r[ok], c[ok]]
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else float("nan")
