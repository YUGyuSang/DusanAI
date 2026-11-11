"""
Synthetic elevation-map evidence for tests and evaluations (ROS free).

:class:`FakeElevationSensor` emulates what ``terrain_mapper`` + the step test
of :mod:`dusan_common.obstacle_layer` report around the robot, with the
properties measured on the real depth pipeline (tools/gazebo_check DepthCamera
+ TerrainMapperSim over plant.sdf):

* robot-centred rolling window (6 x 6 m, 0.04 m cells, centres on multiples
  of the resolution) with persistence: cells keep their last observation
  while they are inside the window (the elevation map remembers);
* observed ground: a wedge of ``fov`` rad with its apex at the camera
  (``cam_offset`` m ahead of base_link, robot_model SENSORS depth_camera:
  0.34 m), ``ground_min`` .. ``ground_max`` m from base_link - the camera
  does not see the ground beside the body, so the side faces of an obstacle
  the robot walks past stay unobserved (``cam_offset=0`` = apex at base_link,
  a wider view than the real camera's);
* a tall box face reaches the step threshold only within ``detect_range``
  (1.2 m, camera pitched 0.6 rad down); evidence = a band ``band_out`` in front
  of the visible face plus ``band_in`` into the box; behind the face the
  ground is occluded (unknown); seen from farther, a face cell that was
  already flagged keeps its height (max-height bias), a removed box clears;
* optional static-map walls (``render_static``: the wall step band is flagged
  like any other edge) and single-frame ghost hits (``ghost_rate``).

Boxes are axis-aligned or rotated rectangles {center, size, yaw}; the list
``boxes`` may be changed between frames (obstacle appears / is removed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .grid_map import OccupancyGridMap
from .obstacle_layer import EvidenceFrame
from .robot_model import SENSORS

CAM_OFFSET = float(SENSORS["depth_camera"]["pose"][0])     # [m] depth camera ahead of base_link


@dataclass
class Box:
    center: Tuple[float, float]
    size: Tuple[float, float]
    yaw: float = 0.0

    def local(self, X, Y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = np.asarray(X, dtype=float) - self.center[0], np.asarray(Y, dtype=float) - self.center[1]
        return c * dx + s * dy, -s * dx + c * dy

    def distance(self, X, Y) -> np.ndarray:
        """Distance [m] from the points to the footprint (0 inside)."""
        u, v = self.local(X, Y)
        hx, hy = 0.5 * self.size[0], 0.5 * self.size[1]
        return np.hypot(np.maximum(np.abs(u) - hx, 0.0), np.maximum(np.abs(v) - hy, 0.0))

    def contains(self, X, Y, pad: float = 0.0) -> np.ndarray:
        u, v = self.local(X, Y)
        return (np.abs(u) <= 0.5 * self.size[0] + pad) & (np.abs(v) <= 0.5 * self.size[1] + pad)


def as_box(b) -> Box:
    if isinstance(b, Box):
        return b
    if isinstance(b, dict):
        return Box(tuple(float(v) for v in b["center"][:2]), tuple(float(v) for v in b["size"][:2]),
                   float(b.get("yaw", 0.0)))
    c, s = b[0], b[1]
    return Box((float(c[0]), float(c[1])), (float(s[0]), float(s[1])), float(b[2]) if len(b) > 2 else 0.0)


def box_clearance(boxes: Iterable, x: float, y: float) -> float:
    """Distance [m] from (x, y) to the nearest box footprint (inf without boxes)."""
    d = [float(as_box(b).distance(x, y)) for b in boxes]
    return min(d) if d else math.inf


def paint_boxes(grid: OccupancyGridMap, boxes: Iterable) -> OccupancyGridMap:
    """Copy of ``grid`` with the box footprints occupied (ground-truth grid)."""
    occ = grid.occ.copy()
    xs = grid.origin_x + (np.arange(grid.width) + 0.5) * grid.resolution
    ys = grid.origin_y + (np.arange(grid.height) + 0.5) * grid.resolution
    X, Y = np.meshgrid(xs, ys)
    for b in boxes:
        occ[as_box(b).contains(X, Y, pad=0.5 * grid.resolution)] = 100
    return OccupancyGridMap(occ, grid.resolution, grid.origin_x, grid.origin_y, grid.frame_id)


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class FakeElevationSensor:
    """Robot-centred synthetic EvidenceFrames (see the module docstring)."""

    def __init__(self, boxes: Sequence = (), res: float = 0.04, half: float = 3.0, fov: float = 1.5,
                 ground_min: float = 0.5, ground_max: float = 3.0, detect_range: float = 1.2,
                 band_in: float = 0.12, band_out: float = 0.08, static_grid: Optional[OccupancyGridMap] = None,
                 render_static: bool = False, ghost_rate: float = 0.0, ghost_cells: int = 2,
                 ghost_region: Optional[Tuple[float, float, float, float]] = None, seed: int = 0,
                 persist: bool = True, frame_id: str = "odom", cam_offset: float = CAM_OFFSET):
        self.boxes: List[Box] = [as_box(b) for b in boxes]
        self.res = float(res)
        self.n = int(round(2.0 * half / self.res))
        self.fov = float(fov)
        self.cam_offset = float(cam_offset)
        self.ground_min, self.ground_max = float(ground_min), float(ground_max)
        self.detect_range = float(detect_range)
        self.band_in, self.band_out = float(band_in), float(band_out)
        self.static_grid = static_grid
        self.render_static = bool(render_static)
        self.ghost_rate = float(ghost_rate)
        self.ghost_cells = int(ghost_cells)
        self.ghost_region = ghost_region
        self.rng = np.random.default_rng(seed)
        self.persist = bool(persist)
        self.frame_id = frame_id
        self.oi: Optional[int] = None
        self.oj: Optional[int] = None
        self.known = np.zeros((self.n, self.n), dtype=bool)
        self.hit = np.zeros((self.n, self.n), dtype=bool)
        self.frames = 0
        self.ghosts = 0

    def add_box(self, b) -> None:
        self.boxes.append(as_box(b))

    def _recenter(self, x: float, y: float) -> None:
        n = self.n
        oi = int(math.floor(x / self.res + 0.5)) - n // 2
        oj = int(math.floor(y / self.res + 0.5)) - n // 2
        if self.oi is None or not self.persist:
            self.known[:] = False
            self.hit[:] = False
        else:
            dc, dr = oi - self.oi, oj - self.oj
            if dc or dr:
                for name in ("known", "hit"):
                    a = getattr(self, name)
                    out = np.zeros_like(a)
                    if abs(dr) < n and abs(dc) < n:
                        out[max(-dr, 0):n + min(-dr, 0), max(-dc, 0):n + min(-dc, 0)] = \
                            a[max(dr, 0):n + min(dr, 0), max(dc, 0):n + min(dc, 0)]
                    setattr(self, name, out)
        self.oi, self.oj = oi, oj

    def frame(self, pose, t: Optional[float] = None) -> EvidenceFrame:
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        self._recenter(x, y)
        n, res = self.n, self.res
        X, Y = np.meshgrid((self.oi + np.arange(n)) * res, (self.oj + np.arange(n)) * res)
        r = np.hypot(X - x, Y - y)                                  # range from base_link
        ex, ey = x + self.cam_offset * math.cos(yaw), y + self.cam_offset * math.sin(yaw)   # camera
        dx, dy = X - ex, Y - ey
        rc = np.hypot(dx, dy)                                       # range from the camera
        vis = (np.abs(_wrap(np.arctan2(dy, dx) - yaw)) <= 0.5 * self.fov) & (r >= self.ground_min) & \
            (r <= self.ground_max)
        occluded = np.zeros((n, n), dtype=bool)
        band = np.zeros((n, n), dtype=bool)
        for b in self.boxes:
            if b.contains(x, y):
                continue
            # segment camera -> cell centre in box coordinates (slab test)
            ou, ov = b.local(ex, ey)
            cu, cv = b.local(X, Y)
            du, dv = cu - ou, cv - ov
            hx, hy = 0.5 * b.size[0], 0.5 * b.size[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                t0 = np.full((n, n), -np.inf)
                t1 = np.full((n, n), np.inf)
                for o, d, h in ((ou, du, hx), (ov, dv, hy)):
                    par = np.abs(d) < 1e-12
                    a1, a2 = (-h - o) / d, (h - o) / d
                    lo, hi = np.minimum(a1, a2), np.maximum(a1, a2)
                    lo = np.where(par, np.where(abs(o) <= h, -np.inf, np.inf), lo)
                    hi = np.where(par, np.where(abs(o) <= h, np.inf, -np.inf), hi)
                    t0, t1 = np.maximum(t0, lo), np.minimum(t1, hi)
            inter = (t0 <= t1) & (t1 >= 0.0) & (t0 <= 1.0)
            t_in = np.clip(np.nan_to_num(t0, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            depth = np.where(inter, (1.0 - t_in) * rc, 0.0)       # entry point -> cell along the ray
            occluded |= inter & (depth > self.band_in)
            band |= b.distance(X, Y) <= self.band_out
        static_band = np.zeros((n, n), dtype=bool)
        if self.render_static and self.static_grid is not None:
            g = self.static_grid
            ix, iy = g.world_to_grid_array(np.stack([X, Y], -1))
            ok = (ix >= 0) & (ix < g.width) & (iy >= 0) & (iy < g.height)
            sd = np.full((n, n), np.inf)
            sd[ok] = g.distance_field()[iy[ok], ix[ok]]
            static_band = sd <= self.band_out + 0.5 * g.resolution
        seen = vis & ~occluded
        near = r <= self.detect_range
        new_hit = (band | static_band) & near
        keep = (band | static_band) & ~near & self.hit             # max-height bias: far views keep the flag
        self.hit[seen] = (new_hit | keep)[seen]
        self.known[seen] = True
        hit, known = self.hit.copy(), self.known.copy()
        if self.ghost_rate > 0.0 and self.rng.random() < self.ghost_rate:
            cand = seen.copy()
            if self.ghost_region is not None:
                x0, y0, x1, y1 = self.ghost_region
                cand &= (X >= min(x0, x1)) & (X <= max(x0, x1)) & (Y >= min(y0, y1)) & (Y <= max(y0, y1))
            ii = np.flatnonzero(cand)
            if ii.size:
                k = ii[self.rng.integers(0, ii.size)]
                r0, c0 = divmod(int(k), n)
                for m in range(max(1, self.ghost_cells)):
                    rr, cc = min(r0 + m // 2, n - 1), min(c0 + m % 2, n - 1)
                    hit[rr, cc] = known[rr, cc] = True
                self.ghosts += 1
        self.frames += 1
        return EvidenceFrame(hit, known, self.oi * res, self.oj * res, res, self.frame_id)
