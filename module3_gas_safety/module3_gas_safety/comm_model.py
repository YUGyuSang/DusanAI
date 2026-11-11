"""
Wireless coverage model (Module 3, ROS free).

For every access point i at (x_i, y_i, z_i) with reference power p0_i [dBm]
at d0 = 1 m, the received power at a robot position (x, y) is

    RSSI_i(x, y) = p0_i - 10 n log10(max(d, d0) / d0) - A_wall * N_i(x, y)

* d   = 3-D distance AP -> robot antenna (antenna height ``antenna_height``),
* n   = path-loss exponent,
* N_i = number of free -> occupied transitions along the 2-D ray AP -> robot
        on the *RF obstacle map* (``OccupancyGridMap.raycast_crossings``), each
        costing ``A_wall`` dB (walls, machines).

The RF obstacle map is the nav map without occupied connected components
smaller than ``rf_min_obstacle_area_m2``: the 2-D nav map marks gauge
stands, pipe-rack posts, drums and similar clutter as occupied for path
planning, but such objects do not attenuate a Wi-Fi link like a wall does;
counting 12 dB for each of them made every ray along a cluttered wall a
"shadow".  ``rf_min_obstacle_area_m2 = 0`` reproduces the literal model.

The robot roams to the strongest AP and shielded areas subtract an extra
attenuation:

    RSSI(x, y) = max_i RSSI_i(x, y) - sum_{zones containing (x, y)} A_zone

The coverage map is pre-computed on a coarse grid with a vectorised batch
ray caster that reproduces ``raycast_crossings`` sample-for-sample (same
step, same ``linspace`` sampling, same cell lookup), so the published map and
the point model agree exactly at grid nodes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dusan_common.grid_map import OccupancyGridMap


@dataclass
class AccessPoint:
    id: str
    x: float
    y: float
    z: float = 2.0
    p0_dbm: float = -30.0


@dataclass
class ShadowZone:
    id: str
    rect: Tuple[float, float, float, float]   # x0, y0, x1, y1
    attenuation_db: float

    def contains(self, x, y):
        x0, y0, x1, y1 = self.rect
        return (x >= min(x0, x1)) & (x <= max(x0, x1)) & (y >= min(y0, y1)) & (y <= max(y0, y1))


@dataclass
class CommConfig:
    path_loss_exponent: float = 2.2
    wall_attenuation_db: float = 12.0
    d0: float = 1.0
    antenna_height: float = 0.6
    rssi_min_dbm: float = -80.0            # below = communication shadow
    map_resolution: float = 0.5
    vis_min_dbm: float = -100.0            # OccupancyGrid 0
    vis_max_dbm: float = -30.0             # OccupancyGrid 100
    rf_min_obstacle_area_m2: float = 2.5   # smaller occupied blobs are RF-transparent clutter
    access_points: List[AccessPoint] = field(default_factory=list)
    shadow_zones: List[ShadowZone] = field(default_factory=list)


def config_from_layout(layout, **overrides) -> CommConfig:
    """CommConfig from the ``comm`` section of world_layout.yaml."""
    c = layout.comm
    aps = [AccessPoint(str(a.get("id", f"AP{i + 1}")), float(a["position"][0]), float(a["position"][1]),
                       float(a["position"][2]) if len(a["position"]) > 2 else 2.0, float(a.get("p0_dbm", -30.0)))
           for i, a in enumerate(c.get("access_points", []))]
    zones = [ShadowZone(str(z.get("id", f"S{i + 1}")), tuple(float(v) for v in z["rect"]),
                        float(z.get("attenuation_db", 30.0)))
             for i, z in enumerate(c.get("shadow_zones", []))]
    cfg = CommConfig(path_loss_exponent=float(c.get("path_loss_exponent", 2.2)),
                     wall_attenuation_db=float(c.get("wall_attenuation_db", 12.0)),
                     rssi_min_dbm=float(c.get("rssi_min_dbm", -80.0)),
                     access_points=aps, shadow_zones=zones)
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


class RssiMap:
    """Gridded RSSI field [dBm]; ``field[j, i]`` at cell centre
    (x0 + (i+1/2) res, y0 + (j+1/2) res) (OccupancyGrid order)."""

    def __init__(self, field_dbm: np.ndarray, resolution: float, origin_x: float, origin_y: float,
                 vis_min: float = -100.0, vis_max: float = -30.0):
        self.field = np.asarray(field_dbm, dtype=float)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.vis_min = float(vis_min)
        self.vis_max = float(vis_max)

    @property
    def shape(self):
        return self.field.shape

    def at(self, x, y):
        """Bilinear RSSI at world points (clamped at the map border)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ny, nx = self.field.shape
        fx = np.clip((x - self.origin_x) / self.resolution - 0.5, 0.0, nx - 1.000001)
        fy = np.clip((y - self.origin_y) / self.resolution - 0.5, 0.0, ny - 1.000001)
        i0 = np.floor(fx).astype(int)
        j0 = np.floor(fy).astype(int)
        tx, ty = fx - i0, fy - j0
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        f = self.field
        out = ((1 - tx) * (1 - ty) * f[j0, i0] + tx * (1 - ty) * f[j0, i1]
               + (1 - tx) * ty * f[j1, i0] + tx * ty * f[j1, i1])
        return float(out) if out.ndim == 0 else out

    # ---------------------------------------------- OccupancyGrid encoding
    def to_grid_values(self) -> np.ndarray:
        """int8 0..100 <-> vis_min..vis_max dBm (linear)."""
        s = (self.field - self.vis_min) / max(self.vis_max - self.vis_min, 1e-9) * 100.0
        return np.clip(np.round(s), 0, 100).astype(np.int8)

    @classmethod
    def from_grid_values(cls, values, resolution, origin_x, origin_y, vis_min=-100.0, vis_max=-30.0):
        v = np.asarray(values, dtype=float)
        f = np.where(v < 0, np.nan, vis_min + v / 100.0 * (vis_max - vis_min))
        if np.isnan(f).any():   # unknown cells -> weakest signal (conservative)
            f = np.where(np.isnan(f), vis_min, f)
        return cls(f, resolution, origin_x, origin_y, vis_min, vis_max)


def rf_obstacle_grid(grid, min_area_m2: float):
    """Copy of the nav map in which occupied 8-connected components smaller
    than ``min_area_m2`` are free (RF-transparent clutter).  Large structures
    (walls, which are connected to the building shell, machines, pallet
    stacks) are kept.  Returns ``grid`` itself when nothing is filtered."""
    if grid is None or min_area_m2 <= 0.0:
        return grid
    occ = grid.occ >= 50
    try:
        import cv2
        n, labels, stats, _ = cv2.connectedComponentsWithStats(occ.astype(np.uint8), connectivity=8)
    except Exception:   # pragma: no cover - OpenCV is a runtime dependency (python3-opencv)
        return grid
    area = stats[:, cv2.CC_STAT_AREA].astype(float) * grid.resolution ** 2
    small = area < float(min_area_m2)
    small[0] = False                     # label 0 = free space
    clutter = small[labels] & occ
    if not clutter.any():
        return grid
    new = grid.occ.copy()
    new[clutter] = 0
    return OccupancyGridMap(new, grid.resolution, grid.origin_x, grid.origin_y, grid.frame_id)


class CommModel:
    def __init__(self, grid, cfg: CommConfig):
        self.grid = grid
        self.cfg = cfg
        #: obstacles that attenuate the radio link (nav map minus small clutter)
        self.rf_grid = rf_obstacle_grid(grid, cfg.rf_min_obstacle_area_m2)
        self._occ = None if self.rf_grid is None else (self.rf_grid.occ >= 50)

    @classmethod
    def from_layout(cls, layout, grid, **overrides) -> "CommModel":
        return cls(grid, config_from_layout(layout, **overrides))

    # ---------------------------------------------------------- point model
    def _path_loss(self, ap: AccessPoint, x, y):
        d = np.sqrt((np.asarray(x) - ap.x) ** 2 + (np.asarray(y) - ap.y) ** 2
                    + (ap.z - self.cfg.antenna_height) ** 2)
        return 10.0 * self.cfg.path_loss_exponent * np.log10(np.maximum(d, self.cfg.d0) / self.cfg.d0)

    def shadow_attenuation(self, x, y):
        att = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)
        for z in self.cfg.shadow_zones:
            att = att + np.where(z.contains(np.asarray(x), np.asarray(y)), z.attenuation_db, 0.0)
        return att

    def ap_rssi(self, x: float, y: float) -> List[float]:
        """Per-AP RSSI (no shadow zones) using the shared ray caster."""
        out = []
        for ap in self.cfg.access_points:
            n = 0
            if self.rf_grid is not None:
                n, _ = self.rf_grid.raycast_crossings(ap.x, ap.y, x, y)
            out.append(float(ap.p0_dbm - self._path_loss(ap, x, y) - self.cfg.wall_attenuation_db * n))
        return out

    def rssi_at(self, x: float, y: float) -> float:
        """RSSI [dBm] at a robot position (strongest AP minus shadow zones)."""
        per = self.ap_rssi(x, y)
        best = max(per) if per else self.cfg.vis_min_dbm
        return float(best - float(self.shadow_attenuation(x, y)))

    def best_ap(self, x: float, y: float) -> Optional[str]:
        per = self.ap_rssi(x, y)
        if not per:
            return None
        return self.cfg.access_points[int(np.argmax(per))].id

    # ------------------------------------------------------- batch caster
    def crossings_batch(self, ap: AccessPoint, xs: np.ndarray, ys: np.ndarray, chunk: int = 512) -> np.ndarray:
        """Vectorised equivalent of ``rf_grid.raycast_crossings(ap.x, ap.y, x, y)[0]``
        for many targets (identical sampling: step = res/2, n = max(2,
        int(dist/step)+1), numpy-linspace point placement)."""
        g = self.rf_grid
        xs = np.asarray(xs, dtype=float).ravel()
        ys = np.asarray(ys, dtype=float).ravel()
        out = np.zeros(xs.size, dtype=np.int32)
        if g is None:
            return out
        step = g.resolution * 0.5
        occ = self._occ
        for s0 in range(0, xs.size, chunk):
            x1 = xs[s0:s0 + chunk]
            y1 = ys[s0:s0 + chunk]
            dist = np.hypot(x1 - ap.x, y1 - ap.y)
            n = np.maximum(2, (dist / step).astype(int) + 1)
            nmax = int(n.max())
            k = np.arange(nmax)[None, :]
            div = (n - 1).astype(float)[:, None]
            stx = (x1 - ap.x)[:, None] / div
            sty = (y1 - ap.y)[:, None] / div
            px = k * stx + ap.x
            py = k * sty + ap.y
            last = k >= (n - 1)[:, None]          # endpoint exactly, then repeat it
            px = np.where(last, x1[:, None], px)
            py = np.where(last, y1[:, None], py)
            ix = np.floor((px - g.origin_x) / g.resolution).astype(int)
            iy = np.floor((py - g.origin_y) / g.resolution).astype(int)
            ok = (ix >= 0) & (ix < g.width) & (iy >= 0) & (iy < g.height)
            o = np.zeros(px.shape, dtype=bool)
            o[ok] = occ[iy[ok], ix[ok]]
            out[s0:s0 + chunk] = np.count_nonzero(o[:, 1:] & ~o[:, :-1], axis=1)
        return out

    def compute_map(self, resolution: Optional[float] = None) -> RssiMap:
        """Pre-compute the coverage map over the nav-map extent."""
        g = self.grid
        res = float(resolution or self.cfg.map_resolution)
        if g is None:
            raise ValueError("compute_map needs a nav map")
        nx = max(1, int(round(g.width * g.resolution / res)))
        ny = max(1, int(round(g.height * g.resolution / res)))
        xs = g.origin_x + (np.arange(nx) + 0.5) * res
        ys = g.origin_y + (np.arange(ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        best = np.full(X.shape, -np.inf)
        for ap in self.cfg.access_points:
            ncross = self.crossings_batch(ap, X, Y).reshape(X.shape)
            r = ap.p0_dbm - self._path_loss(ap, X, Y) - self.cfg.wall_attenuation_db * ncross
            best = np.maximum(best, r)
        if not np.isfinite(best).all():
            best = np.where(np.isfinite(best), best, self.cfg.vis_min_dbm)
        best = best - self.shadow_attenuation(X, Y)
        return RssiMap(best, res, g.origin_x, g.origin_y, self.cfg.vis_min_dbm, self.cfg.vis_max_dbm)


class FadingProcess:
    """Slow log-normal shadow fading (Ornstein-Uhlenbeck in dB)."""

    def __init__(self, sigma_db: float = 2.0, tau_s: float = 2.0, seed: Optional[int] = None):
        self.sigma = float(sigma_db)
        self.tau = float(tau_s)
        self.rng = np.random.default_rng(seed)
        self.x = 0.0

    def step(self, dt: float) -> float:
        if self.sigma <= 0.0:
            return 0.0
        f = math.exp(-max(dt, 0.0) / max(self.tau, 1e-6))
        self.x = self.x * f + self.sigma * math.sqrt(max(0.0, 1.0 - f * f)) * float(self.rng.normal())
        return self.x
