"""
Terrain cost layer of the global planners (ROS free, numpy only).

The nav map (simulation/maps/plant_map.yaml) is binary: the debris of
``world_layout.yaml`` that a quadruped can step over - lying pipes, rubble
fields, slippery steel plates, low platforms - is FREE there, so a plain A*
walks straight across it.  This module rasterises that debris onto the nav-map
grid as an *additional* cell cost for ``dusan_common.planning.astar`` /
``plan(extra_cost=...)``:

    cell centre within INSIDE_PAD (0.12 m) of the footprint : w_<type>                    (inside)
    cell centre within ``margin`` m of the footprint        : margin_fraction * w_<type>  (margin)
    elsewhere                                               : 0

* Pipes and rubble get a HIGH inside cost, plates and the low platform a small
  one; the margin cost is much smaller than the inside cost so a smoothed
  path never trades a few centimetres of debris for margin cells.
* The margin is capped at :data:`MAX_MARGIN` (0.25 m): the free strip at the
  north end of pipe C-OBS1 is only ~0.3 m wide (robot centre band), a wider
  margin would cover it and the route would cross the pipe again.
* Every cost is finite: where no detour exists (C-OBS5 spans the corridor up
  to a 0.09 m gap) the planner still finds the path across the debris; where
  a free detour exists (north end of C-OBS1, the passage along the north wall
  at the C-OBS3 rubble) the planner takes it.
* "inside" is padded by INSIDE_PAD: a robot centre that close puts its feet on
  the item (squeezing 0.1 m past the end of C-OBS5 at the wall is crossing it,
  not a detour), and even a 0.08 m pipe forms a closed line of costly cells
  that an 8-connected A* cannot slip through diagonally (pad > half a cell
  diagonal).

The same parameter names are used by every node that plans (mission_manager,
return_to_home, patrol_planner; see :data:`TERRAIN_COST_DEFAULTS`):

    terrain_cost_enabled     bool   use the layer
    terrain_w_pipe           float  inside cost of a lying pipe
    terrain_w_rubble         float  inside cost of a rubble field
    terrain_w_plate          float  inside cost of a steel plate (slippery, crossable)
    terrain_w_platform       float  inside cost of the low step platform
    terrain_margin           float  [m] margin around each item (<= 0.25 m)
    terrain_margin_fraction  float  margin cost = fraction x inside cost
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAX_MARGIN = 0.25           # [m] see the module docstring (C-OBS1 north gap)
INSIDE_PAD = 0.12           # [m] centre closer than this to the footprint counts as inside
COST_TYPES = ("pipe", "rubble", "plate", "platform")

TERRAIN_COST_DEFAULTS: Dict[str, object] = {
    "terrain_cost_enabled": True,
    "terrain_w_pipe": 25.0,
    "terrain_w_rubble": 25.0,
    "terrain_w_plate": 1.0,
    "terrain_w_platform": 2.0,
    "terrain_margin": 0.25,
    "terrain_margin_fraction": 0.15,
}


@dataclass
class TerrainCostParams:
    enabled: bool = True
    w_pipe: float = 25.0
    w_rubble: float = 25.0
    w_plate: float = 1.0
    w_platform: float = 2.0
    margin: float = 0.25
    margin_fraction: float = 0.15

    @classmethod
    def from_params(cls, p: Optional[dict], prefix: str = "terrain_") -> "TerrainCostParams":
        """From a node parameter dict (``terrain_cost_enabled``, ``terrain_w_pipe`` ...);
        missing keys keep their defaults."""
        p = p or {}
        base = cls()

        def num(key, default):
            v = p.get(prefix + key, default)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return float(default)
            return v if math.isfinite(v) else float(default)

        return cls(enabled=bool(p.get(prefix + "cost_enabled", base.enabled)),
                   w_pipe=num("w_pipe", base.w_pipe), w_rubble=num("w_rubble", base.w_rubble),
                   w_plate=num("w_plate", base.w_plate), w_platform=num("w_platform", base.w_platform),
                   margin=num("margin", base.margin), margin_fraction=num("margin_fraction", base.margin_fraction))

    def weight(self, kind: str) -> float:
        return max(0.0, float(getattr(self, "w_" + kind, 0.0)))

    @property
    def margin_m(self) -> float:
        return min(max(float(self.margin), 0.0), MAX_MARGIN)

    def key(self) -> tuple:
        return (bool(self.enabled), self.weight("pipe"), self.weight("rubble"), self.weight("plate"),
                self.weight("platform"), round(self.margin_m, 4), round(max(0.0, float(self.margin_fraction)), 4))


# =============================================================================
# geometry
# =============================================================================
def _obstacles(source) -> List[dict]:
    """Layout (``.obstacles``), a layout dict (``['obstacles']``) or a list of obstacle dicts."""
    if source is None:
        return []
    if isinstance(source, dict):
        obs = source.get("obstacles", [])
    elif isinstance(source, (list, tuple)):
        obs = source
    else:
        try:
            obs = getattr(source, "obstacles", [])
        except Exception:          # a layout stand-in without obstacles
            obs = []
    return [o for o in (obs or []) if isinstance(o, dict)]


def footprint_bounds(ob: dict) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned bounds (x0, y0, x1, y1) of the 2-D footprint of a cost item, or None."""
    t = ob.get("type")
    try:
        if t == "pipe":
            cx, cy = float(ob["center"][0]), float(ob["center"][1])
            yaw, half, r = float(ob.get("yaw", 0.0)), 0.5 * float(ob["length"]), 0.5 * float(ob["diameter"])
            dx, dy = half * abs(math.cos(yaw)), half * abs(math.sin(yaw))
            return cx - dx - r, cy - dy - r, cx + dx + r, cy + dy + r
        if t in ("rubble", "plate", "platform"):
            x0, y0, x1, y1 = (float(v) for v in ob["rect"])
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return None


def footprint_distance(ob: dict, X, Y):
    """Euclidean distance [m] from the points (X, Y) to the footprint of ``ob``
    (0 inside).  Pipe = capsule around its axis (radius d / 2), rubble / plate /
    platform = axis-aligned rect.  None for other types."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    t = ob.get("type")
    if t == "pipe":
        cx, cy = float(ob["center"][0]), float(ob["center"][1])
        yaw, half, r = float(ob.get("yaw", 0.0)), 0.5 * float(ob["length"]), 0.5 * float(ob["diameter"])
        ux, uy = math.cos(yaw), math.sin(yaw)
        rx, ry = X - cx, Y - cy
        s = np.clip(rx * ux + ry * uy, -half, half)
        return np.maximum(np.hypot(rx - s * ux, ry - s * uy) - r, 0.0)
    if t in ("rubble", "plate", "platform"):
        x0, y0, x1, y1 = footprint_bounds(ob)
        dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
        return np.hypot(dx, dy)
    return None


# =============================================================================
# rasteriser
# =============================================================================
def terrain_cost_grid(grid, source, params: Optional[TerrainCostParams] = None) -> np.ndarray:
    """Extra A* cell cost (float array, shape ``grid.occ.shape``, >= 0) of the
    debris in ``source`` (Layout / layout dict / obstacle list).  Overlapping
    items take the maximum.  Disabled params -> all zeros."""
    prm = params or TerrainCostParams()
    h, w = grid.occ.shape
    cost = np.zeros((h, w), dtype=float)
    if not prm.enabled:
        return cost
    res = float(grid.resolution)
    margin = prm.margin_m
    frac = max(0.0, float(prm.margin_fraction))
    pad = margin + res
    for ob in _obstacles(source):
        kind = ob.get("type")
        if kind not in COST_TYPES:
            continue
        wt = prm.weight(kind)
        b = footprint_bounds(ob)
        if wt <= 0.0 or b is None:
            continue
        # window of cells around the item
        i0 = max(0, int(math.floor((b[0] - pad - grid.origin_x) / res)))
        i1 = min(w, int(math.ceil((b[2] + pad - grid.origin_x) / res)) + 1)
        j0 = max(0, int(math.floor((b[1] - pad - grid.origin_y) / res)))
        j1 = min(h, int(math.ceil((b[3] + pad - grid.origin_y) / res)) + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        xs = grid.origin_x + (np.arange(i0, i1) + 0.5) * res
        ys = grid.origin_y + (np.arange(j0, j1) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        d = footprint_distance(ob, X, Y)
        inside = d <= max(INSIDE_PAD, 0.5 * math.sqrt(2.0) * res) + 1e-9
        val = np.where(inside, wt, np.where(d <= margin + 1e-9, frac * wt, 0.0))
        win = cost[j0:j1, i0:i1]
        np.maximum(win, val, out=win)
    return cost


def cost_from_params(grid, source, p: Optional[dict]) -> Optional[np.ndarray]:
    """``terrain_cost_grid`` from a node parameter dict, or None when
    ``terrain_cost_enabled`` is false (planners then run without extra cost)."""
    prm = TerrainCostParams.from_params(p)
    if not prm.enabled or source is None:
        return None
    return terrain_cost_grid(grid, source, prm)


def _grid_signature(grid) -> tuple:
    occ = np.ascontiguousarray(grid.occ)
    return (occ.shape, round(float(grid.resolution), 6), round(float(grid.origin_x), 4),
            round(float(grid.origin_y), 4), zlib.crc32(occ.tobytes()))


class TerrainCostCache:
    """Cost layer of one layout, recomputed only when the grid geometry or the
    parameters change (planners call it before every A* run)."""

    def __init__(self, source, params: Optional[TerrainCostParams] = None):
        self.source = source
        self.params = params or TerrainCostParams()
        self._key = None
        self._cost = None

    @classmethod
    def from_params(cls, source, p: Optional[dict]) -> "TerrainCostCache":
        return cls(source, TerrainCostParams.from_params(p))

    @property
    def enabled(self) -> bool:
        return bool(self.params.enabled) and self.source is not None

    def __call__(self, grid) -> Optional[np.ndarray]:
        if grid is None or not self.enabled:
            return None
        key = (_grid_signature(grid), self.params.key())
        if key != self._key:
            self._cost = terrain_cost_grid(grid, self.source, self.params)
            self._key = key
        return self._cost


# =============================================================================
# evaluation helpers (tests, tools/eval_patrol.py)
# =============================================================================
def sample_cost(grid, cost, a, b) -> Tuple[float, float]:
    """(integral [cost x m], max) of ``cost`` along the segment a -> b (midpoint
    samples every half cell, outside the map 0) - the measure the smoother uses."""
    from .planning import _segment_cost
    return _segment_cost(grid, np.asarray(cost, dtype=float), a, b)


def path_cost(grid, cost, path: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """(integral, max) of the cost layer along a polyline."""
    tot, mx = 0.0, 0.0
    if not path or len(path) < 2:
        return tot, mx
    for a, b in zip(path[:-1], path[1:]):
        s, m = sample_cost(grid, cost, a, b)
        tot += s
        mx = max(mx, m)
    return tot, mx


def debris_crossed(path: Sequence[Sequence[float]], source, clearance: float = 0.0,
                   types: Iterable[str] = COST_TYPES, spacing: float = 0.05) -> List[str]:
    """Ids of the debris items whose footprint the robot CENTRE passes within
    ``clearance`` m of (0 = drives over it), checked every ``spacing`` m."""
    if not path or len(path) < 1:
        return []
    pts = [tuple(map(float, path[0][:2]))]
    for a, b in zip(path[:-1], path[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(d / spacing)))
        for k in range(1, n + 1):
            pts.append((a[0] + k / n * (b[0] - a[0]), a[1] + k / n * (b[1] - a[1])))
    P = np.asarray(pts, dtype=float)
    out = []
    types = set(types)
    for ob in _obstacles(source):
        if ob.get("type") not in types:
            continue
        d = footprint_distance(ob, P[:, 0], P[:, 1])
        if d is not None and np.any(d <= clearance + 1e-9):
            out.append(str(ob.get("id", ob.get("type"))))
    return out


def _big_components(grid, radius: float, min_area: float = 1.0) -> int:
    """Number of free components (at ``radius``) larger than ``min_area`` m^2."""
    lab = grid.free_labels(radius)
    n = np.bincount(lab.ravel())
    if len(n) <= 1:
        return 0
    return int(np.count_nonzero(n[1:] * float(grid.resolution) ** 2 >= min_area))


def debris_blocked_grid(grid, source, kinds: Iterable[str] = ("pipe", "rubble"), margin: float = 0.10,
                        keep_passable: float = 0.0):
    """Copy of the nav map with the debris of ``kinds`` in ``source`` (Layout /
    layout dict / obstacle list) marked occupied, grown by ``margin`` m.

    For planners without a cost layer that steer the robot themselves, e.g.
    the gas source seeker: its gradient steps and local A* see only the
    binary nav map (debris FREE) and drove the robot straight over the
    F-OBS5 rubble field / F-OBS4 pipe, where it fell (Gazebo mission check).
    The cost-layer planners (patrol, RTH) keep the debris crossable.

    ``keep_passable`` > 0 (the planner's robot radius): an item whose blocking
    would cut the free space apart at that radius (more free regions of
    >= 1 m^2, e.g. the F-OBS3 pipe across the partition doorway) stays free."""
    if grid is None or source is None:
        return grid
    from .grid_map import OccupancyGridMap
    occ = np.array(grid.occ, dtype=np.int8, copy=True)
    h, w = occ.shape
    res = float(grid.resolution)
    kinds = set(kinds)
    changed = False
    frame = getattr(grid, "frame_id", "map")
    n_ref = _big_components(grid, keep_passable) if keep_passable > 0.0 else 0
    for ob in _obstacles(source):
        if ob.get("type") not in kinds:
            continue
        b = footprint_bounds(ob)
        if b is None:
            continue
        pad = float(margin) + res
        i0 = max(0, int(math.floor((b[0] - pad - grid.origin_x) / res)))
        i1 = min(w, int(math.ceil((b[2] + pad - grid.origin_x) / res)) + 1)
        j0 = max(0, int(math.floor((b[1] - pad - grid.origin_y) / res)))
        j1 = min(h, int(math.ceil((b[3] + pad - grid.origin_y) / res)) + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        xs = grid.origin_x + (np.arange(i0, i1) + 0.5) * res
        ys = grid.origin_y + (np.arange(j0, j1) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        d = footprint_distance(ob, X, Y)
        if d is None:
            continue
        win = occ[j0:j1, i0:i1]
        before = win.copy()
        win[np.asarray(d) <= float(margin)] = 100
        if keep_passable > 0.0:
            trial = OccupancyGridMap(occ, res, grid.origin_x, grid.origin_y, frame_id=frame)
            if _big_components(trial, keep_passable) > n_ref:
                win[...] = before                  # blocking it would close a passage
                continue
        changed = True
    if not changed:
        return grid
    return OccupancyGridMap(occ, res, grid.origin_x, grid.origin_y, frame_id=frame)
