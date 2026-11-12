"""
Dynamic obstacle layer: avoid NEW obstacles that are not in the nav map
(ROS free, numpy only; map agnostic).

The global planners of the project (mission patrol / self-RTH, return_to_home,
the gas seeker's local planner) only know the static nav map.  This module
turns the robot's own perception (the organised elevation cloud of
``terrain_mapper`` on /elevation_map/points, or any OccupancyGrid cost grid)
into an obstacle memory and decides, once per control tick, how to react:

    (a) evidence on the followed path        -> slow down; within the stop
                                                distance -> stop forward motion
    (b) a CONFIRMED obstacle blocks the path -> local detour left or right
        within the look-ahead                   (shorter / free side, committed
                                                per obstacle) that rejoins the
                                                base route beyond the obstacle
    (c) no acceptable detour                 -> global re-plan on static + dynamic
    (d) no path on static + dynamic          -> keep the static route; the
                                                caller's stuck / back-up / skip
                                                logic decides

Building blocks
---------------
EvidenceFrame    perception snapshot -> per-cell obstacle evidence (hit) and
                 observation (known) on the sensor's own grid.  Default source:
                 STEP test on the elevation cloud (NaN-aware max - min over a
                 ``step_window`` window >= ``step_height``; the terrain
                 mapper's own step_crit / step_window).  The cost-100 cells of
                 /elevation_map/traversability also contain planar steep
                 slopes (pipe flanks, ramp edges: measured false positives),
                 so that source is optional (``source: traversability``).
ObstacleMemory   world-fixed rolling lattice around the robot: multi-frame
                 confirmation (hysteresis), clearing when seen free again,
                 forgetting, robot-footprint exclusion, novelty filter against
                 the static map (mapped walls are not "new"), speckle filter
                 (also for the unconfirmed layer of the governor), a fall
                 guard (``set_upright``: no fusion while the robot is down,
                 the evidence first seen while it fell is forgotten when it
                 stands again) and the
                 occlusion shadow (planning only).  Layers: ``output``
                 (confirmed), ``raw`` (latest frame), ``planning`` (output +
                 the unseen back of each obstacle assumed ``shadow_depth``
                 deep along the line of sight, until a cell was seen free
                 after the evidence: the depth camera never sees the sides or
                 the back of a box it walks past) and ``near`` (output + the
                 short isotropic shadow; the fallback where the full shadow
                 closes a real gap).  ``frame_due`` thins the frames while the
                 owner does not drive.
path_blockage    first blockage along a path (free travel); a robot already
                 inside the clearance zone may leave it without getting closer.
DetourPlanner    local A* in a window with a hard left / right "cut" so each
                 side is planned separately, rejoining the base route beyond
                 the obstacle where the route stays free (a second obstacle
                 right behind is merged); sides compared by length + terrain
                 cost (A*'s own metric); side commitment against oscillation.
speed_scale      slow-down / stop governor; ``pursuit_path`` = the line the
                 pure-pursuit follower actually drives (robot -> look-ahead
                 vertex -> path), along which the governor measures.
ObstacleAvoider  facade: one ``step`` per control tick ->
                 AvoidAction(none | detour | replan | hold, speed_scale).
                 Detours / global re-plans try the planning layer, then the
                 near layer (also taken when switch_ratio cheaper in length +
                 terrain cost: ``near_is_cheaper``); a followed detour is
                 validated on the layer it was planned on up to its rejoin
                 point (a face confirmed later refines it), the route on
                 confirmed evidence.

Every parameter has the prefix ``obstacle_`` in the node parameter files
(:data:`OBSTACLE_DEFAULTS`); ``obstacle_enabled: false`` gives exactly the
behaviour without this module.  Frames must be expressed in the pose frame
(``obstacle_frames``; odom = map = world in this project), no TF lookups.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields as dc_fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .grid_map import OccupancyGridMap, _label_components, submap
from .planning import _segment_cost, astar, densify, path_length, smooth_path

# =============================================================================
# parameters
# =============================================================================
OBSTACLE_DEFAULTS: Dict[str, object] = {
    # switches
    "obstacle_enabled": True,                 # master switch (false = legacy behaviour, no subscription)
    "obstacle_detour_enabled": True,          # local left / right detours
    "obstacle_replan_enabled": True,          # global re-plans on static + dynamic, patrol-leg validation
    "obstacle_slowdown_enabled": True,        # speed governor
    "obstacle_static_fallback": True,         # no path around: keep the map route (stuck logic decides)
    # input
    "obstacle_source": "elevation",           # elevation (cloud step test) | traversability (cost grid)
    "obstacle_elevation_topic": "/elevation_map/points",
    "obstacle_traversability_topic": "/elevation_map/traversability",
    "obstacle_frames": "odom,map,world",      # frame ids that coincide with the pose frame
    "obstacle_step_height": 0.22,             # [m] = terrain.yaml map.step_crit
    "obstacle_step_window": 5,                # [cells] = terrain.yaml map.step_window
    "obstacle_cost_threshold": 100,           # traversability source: cell >= this is evidence
    "obstacle_max_range": 3.5,                # [m] evidence farther from base_link is ignored
    # memory
    "obstacle_resolution": 0.10,              # [m] lattice anchored at the world origin
    "obstacle_memory_size": 20.0,             # [m] edge of the robot-centred rolling memory
    "obstacle_confirm_hits": 2,               # evidence frames before a cell is confirmed
    "obstacle_max_hits": 5,                   # saturation = free frames needed to clear a cell
    "obstacle_min_known_fraction": 0.5,       # known share of a lattice cell for a 'seen free'
    "obstacle_ignore_radius": 0.55,           # [m] own body / legs around base_link
    "obstacle_forget_time": 60.0,             # [s] without new evidence -> forgotten (0 = never)
    "obstacle_min_cluster": 2,                # confirmed cells in the 3x3 block (speckle filter)
    "obstacle_static_tolerance": 0.30,        # [m] evidence this close to the static map = mapped structure
    "obstacle_shadow_depth": 1.0,             # [m] planning: assumed depth of the unseen back of an obstacle
    #                                           (along the line of sight, until a cell was seen free)
    "obstacle_idle_period": 1.0,              # [s] frame period while not driving (0 = every frame)
    # detour / re-plan
    "obstacle_margin": 0.10,                  # [m] detour clearance beyond robot_radius (hysteresis)
    "obstacle_lookahead": 2.5,                # [m] of followed path checked for blockage
    "obstacle_detour_window": 2.5,            # [m] padding of the local planning window (<= 16 m)
    "obstacle_rejoin_gap": 0.8,               # [m] beyond the blocked section where the detour rejoins
    "obstacle_detour_max_ratio": 3.0,         # detour <= ratio x replaced route + 2 m
    "obstacle_switch_ratio": 1.5,             # committed side kept unless the other is this much shorter
    "obstacle_commit_time": 15.0,             # [s] side choice remembered per obstacle
    "obstacle_min_interval": 0.5,             # [s] between detour computations
    "obstacle_max_detours": 8,                # per goal, then re-plan
    "obstacle_max_replans": 3,                # obstacle-triggered global re-plans per goal
    # speed governor (distances = free travel of the robot_radius footprint along the path)
    "obstacle_slow_dist": 1.0,                # [m] slow down below (ramp to 0 at stop_dist)
    "obstacle_stop_dist": 0.35,               # [m] stop forward motion below
    "obstacle_unconfirmed_scale": 0.5,        # speed floor for not yet confirmed evidence
}

# string parameters (node level: input selection), declared by every node that subscribes
OBSTACLE_NODE_KEYS: Tuple[str, ...] = ("obstacle_source", "obstacle_elevation_topic",
                                       "obstacle_traversability_topic", "obstacle_frames")
# memory / input subset (the gas seeker uses only the memory, no detour planner / governor)
OBSTACLE_MEMORY_KEYS: Tuple[str, ...] = (
    "obstacle_enabled", "obstacle_source", "obstacle_elevation_topic", "obstacle_traversability_topic",
    "obstacle_frames", "obstacle_step_height", "obstacle_step_window", "obstacle_cost_threshold",
    "obstacle_max_range", "obstacle_resolution", "obstacle_memory_size", "obstacle_confirm_hits",
    "obstacle_max_hits", "obstacle_min_known_fraction", "obstacle_ignore_radius", "obstacle_forget_time",
    "obstacle_min_cluster", "obstacle_static_tolerance", "obstacle_shadow_depth", "obstacle_idle_period")

RAW_MAX_AGE = 1.0          # [s] unconfirmed (latest-frame) evidence older than this is ignored
SHADOW_FRESH_S = 10.0      # [s] a cell counts as 'seen free' for the shadow only this long
SHADOW_NEAR = 0.30         # [m] isotropic part of the shadow (cells without a recorded viewpoint, ray gaps)
MAX_SHADOW = 5.0           # [m] cap of obstacle_shadow_depth
MAX_WINDOW = 16.0          # [m] edge cap of the detour planning window
LEAVE_LEN = 0.5            # [m] a robot inside the clearance zone must leave it within this path length
MIN_LEAD = 0.15            # [m] the governor never asks the pure-pursuit follower for a shorter look-ahead
DEBRIS_CUT = 1.0           # [cost x m] a chord this much costlier than the path it cuts = cutting across debris
LEFT, RIGHT = "left", "right"
# fall guard (ObstacleMemory.set_upright): a camera that tips over with the body sees the floor, the
# robot's own legs and the sky at odd angles - phantom evidence around the robot (headless Gazebo, a
# fall on the ramp: 7-10 cells 0.46-0.67 m from the robot, true step 0.03-0.18 m, kept the path
# blocked after the stand-up; inside the ignore radius / outside the field of view they are never seen
# free again and lasted until forget_time = 60 s)
FALL_LOOKBACK = 2.0        # [s] evidence first seen this long before the fall was reported is forgotten too
FALL_SETTLE = 1.0          # [s] after standing up again: no fusion yet (body still settling)
DOWN_MODES = ("FALLEN", "RECOVERING")


def robot_down(mode, recovery_state=None) -> bool:
    """/locomotion/state ``mode`` / ``recovery_state`` -> True while the robot is
    down: fallen, self-righting, or any recovery state other than NORMAL (an
    E-STOP engaged in the middle of a recovery keeps it, FAILED = lying)."""
    m = str(mode or "").strip().upper()
    rs = str(recovery_state or "").strip().upper()
    return m in DOWN_MODES or rs not in ("", "NORMAL")


def _coerce(value, default):
    """``value`` with the type of ``default``; invalid / non-finite -> default."""
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(value, (float, int, np.number)) and not math.isfinite(float(value)):
                return default
            return bool(value)
        if isinstance(default, int):
            if isinstance(value, bool):
                return default
            v = float(value)
            return int(round(v)) if math.isfinite(v) else default
        if isinstance(default, float):
            if isinstance(value, bool):
                return default
            v = float(value)
            return v if math.isfinite(v) else default
        if isinstance(default, str):
            return value if isinstance(value, str) else default
    except (TypeError, ValueError):
        return default
    return value


@dataclass
class ObstacleParams:
    """:data:`OBSTACLE_DEFAULTS` without the ``obstacle_`` prefix."""
    enabled: bool = True
    detour_enabled: bool = True
    replan_enabled: bool = True
    slowdown_enabled: bool = True
    static_fallback: bool = True
    source: str = "elevation"
    elevation_topic: str = "/elevation_map/points"
    traversability_topic: str = "/elevation_map/traversability"
    frames: str = "odom,map,world"
    step_height: float = 0.22
    step_window: int = 5
    cost_threshold: int = 100
    max_range: float = 3.5
    resolution: float = 0.10
    memory_size: float = 20.0
    confirm_hits: int = 2
    max_hits: int = 5
    min_known_fraction: float = 0.5
    ignore_radius: float = 0.55
    forget_time: float = 60.0
    min_cluster: int = 2
    static_tolerance: float = 0.30
    shadow_depth: float = 1.0
    idle_period: float = 1.0
    margin: float = 0.10
    lookahead: float = 2.5
    detour_window: float = 2.5
    rejoin_gap: float = 0.8
    detour_max_ratio: float = 3.0
    switch_ratio: float = 1.5
    commit_time: float = 15.0
    min_interval: float = 0.5
    max_detours: int = 8
    max_replans: int = 3
    slow_dist: float = 1.0
    stop_dist: float = 0.35
    unconfirmed_scale: float = 0.5

    def __post_init__(self):
        self.max_hits = int(min(max(int(self.max_hits), 1), 127))           # int8 counter
        self.confirm_hits = int(min(max(int(self.confirm_hits), 1), self.max_hits))
        self.min_cluster = int(min(max(int(self.min_cluster), 1), 9))
        self.step_window = max(1, int(self.step_window) | 1)
        self.resolution = max(float(self.resolution), 0.01)
        self.memory_size = max(float(self.memory_size), 8.0 * self.resolution)
        self.min_known_fraction = min(max(float(self.min_known_fraction), 0.0), 1.0)
        self.unconfirmed_scale = min(max(float(self.unconfirmed_scale), 0.0), 1.0)
        self.slow_dist = max(float(self.slow_dist), 0.0)
        self.stop_dist = min(max(float(self.stop_dist), 0.0), self.slow_dist)
        for k in ("ignore_radius", "forget_time", "static_tolerance", "shadow_depth", "idle_period", "margin",
                  "lookahead", "detour_window", "rejoin_gap", "commit_time", "min_interval", "max_range"):
            setattr(self, k, max(float(getattr(self, k)), 0.0))
        self.shadow_depth = min(self.shadow_depth, MAX_SHADOW)
        self.detour_max_ratio = max(float(self.detour_max_ratio), 1.0)
        self.switch_ratio = max(float(self.switch_ratio), 1.0)
        self.max_detours = max(int(self.max_detours), 0)
        self.max_replans = max(int(self.max_replans), 0)

    @classmethod
    def from_params(cls, p: Optional[dict] = None, prefix: str = "obstacle_") -> "ObstacleParams":
        """From a node parameter dict (``obstacle_enabled`` ...).  Missing or
        invalid values keep their defaults, values are coerced to the
        default's type (like ``TerrainCostParams.from_params``)."""
        if isinstance(p, ObstacleParams):
            return p
        p = p or {}
        base = cls()
        kw = {}
        for f in dc_fields(cls):
            key = prefix + f.name
            if key in p:
                kw[f.name] = _coerce(p[key], getattr(base, f.name))
        return cls(**kw)


def frame_accepted(frame_id, frames_csv) -> bool:
    """Is ``frame_id`` one of the comma separated ``frames_csv`` (leading '/'
    stripped, case sensitive)?  An empty list or '*' accepts every frame;
    an empty frame id is accepted only then."""
    names = [s.strip().lstrip("/") for s in str(frames_csv or "").split(",")]
    names = [s for s in names if s]
    if not names or "*" in names:
        return True
    fid = str(frame_id or "").strip().lstrip("/")
    return bool(fid) and fid in names


# =============================================================================
# inputs
# =============================================================================
_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def organised_cloud_to_xyz(msg) -> np.ndarray:
    """Duck-typed ``sensor_msgs/PointCloud2`` (height, width, fields with
    name / offset / datatype, point_step, row_step, is_bigendian, data) ->
    (rows, cols, 3) float array.  Raises ValueError for a missing x / y / z
    field, an unsupported datatype, a short buffer or a non-organised cloud
    (no sensor_msgs import: ROS free)."""
    rows, cols = int(msg.height), int(msg.width)
    if rows < 2 or cols < 2:
        raise ValueError(f"organised cloud expected, got {rows} x {cols}")
    fields = {str(f.name): f for f in (msg.fields or [])}
    for n in ("x", "y", "z"):
        if n not in fields:
            raise ValueError(f"point cloud has no '{n}' field (fields {sorted(fields)})")
    end = ">" if bool(msg.is_bigendian) else "<"
    names, formats, offsets = [], [], []
    for n in ("x", "y", "z"):
        f = fields[n]
        code = _DTYPES.get(int(f.datatype))
        if code is None:
            raise ValueError(f"unsupported datatype {f.datatype} of field '{n}'")
        names.append(n)
        formats.append(end + code)
        offsets.append(int(f.offset))
    point_step, row_step = int(msg.point_step), int(msg.row_step)
    data = msg.data
    if isinstance(data, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(data, dtype=np.uint8)
    else:
        try:
            buf = np.frombuffer(memoryview(data), dtype=np.uint8)       # array.array('B')
        except TypeError:
            buf = np.asarray(data, dtype=np.uint8)
    need = (rows - 1) * row_step + cols * point_step
    if buf.size < need or point_step <= 0 or row_step < cols * point_step:
        raise ValueError(f"point cloud buffer too short / inconsistent ({buf.size} < {need} bytes)")
    if row_step != cols * point_step:
        idx = (np.arange(rows, dtype=np.int64)[:, None] * row_step
               + np.arange(cols * point_step, dtype=np.int64)[None, :])
        body = buf[idx.reshape(-1)]
    else:
        body = buf[: rows * row_step]
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step})
    rec = np.frombuffer(np.ascontiguousarray(body), dtype=dt, count=rows * cols)
    out = np.empty((rows * cols, 3), dtype=np.float64)
    for k, n in enumerate(names):
        out[:, k] = rec[n]
    return out.reshape(rows, cols, 3)


def _shifted(a: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """b[i, j] = a[i + dr, j + dc]; cells shifted in from outside get ``fill``."""
    rows, cols = a.shape
    out = np.full(a.shape, fill, dtype=a.dtype)
    if abs(dr) >= rows or abs(dc) >= cols:
        return out
    out[max(-dr, 0):rows + min(-dr, 0), max(-dc, 0):cols + min(-dc, 0)] = \
        a[max(dr, 0):rows + min(dr, 0), max(dc, 0):cols + min(dc, 0)]
    return out


def _separable(a: np.ndarray, radius: int, op, fill) -> np.ndarray:
    if radius <= 0:
        return a.copy()
    out = a.copy()
    for d in range(1, radius + 1):
        out = op(out, op(_shifted(a, d, 0, fill), _shifted(a, -d, 0, fill)))
    tmp = out
    out = tmp.copy()
    for d in range(1, radius + 1):
        out = op(out, op(_shifted(tmp, 0, d, fill), _shifted(tmp, 0, -d, fill)))
    return out


def nan_max_filter(a: np.ndarray, window: int) -> np.ndarray:
    """NaN-ignoring max filter (NaN where the whole window is unknown); same
    semantics as module1 elevation_map.max_filter (duplicated: dusan_common
    must not import module1)."""
    v = np.where(np.isfinite(a), a, -np.inf)
    out = _separable(v, int(window) // 2, np.maximum, -np.inf)
    return np.where(np.isfinite(out), out, np.nan)


def nan_min_filter(a: np.ndarray, window: int) -> np.ndarray:
    """NaN-ignoring min filter (NaN where the whole window is unknown)."""
    v = np.where(np.isfinite(a), a, np.inf)
    out = _separable(v, int(window) // 2, np.minimum, np.inf)
    return np.where(np.isfinite(out), out, np.nan)


def _snap(v: float, res: float) -> float:
    """``v`` snapped to a multiple of ``res`` when it is one up to float noise."""
    k = round(v / res)
    return k * res if abs(v / res - k) < 1e-3 else v


@dataclass
class EvidenceFrame:
    """Obstacle evidence on the sensor's own grid: ``hit`` / ``known`` (rows,
    cols) bool, (x0, y0) = world centre of cell [0, 0], row = y, col = x.
    ``clear`` (optional, default = known and not hit): the free cells whose
    observation may clear earlier evidence (the step test: only where the
    whole window was observed); every known free cell still counts as seen
    free (occlusion shadow)."""
    hit: np.ndarray
    known: np.ndarray
    x0: float
    y0: float
    res: float
    frame_id: str = "odom"
    clear: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _xy: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.known = np.asarray(self.known, dtype=bool)
        self.hit = np.asarray(self.hit, dtype=bool) & self.known
        if self.hit.shape != self.known.shape or self.hit.ndim != 2:
            raise ValueError("hit / known must be 2-D arrays of the same shape")
        if self.clear is not None:
            self.clear = np.asarray(self.clear, dtype=bool)
            if self.clear.shape != self.known.shape:
                raise ValueError("clear must have the shape of known")
            self.clear = self.clear & self.known & ~self.hit
        self.res = float(self.res)
        if not (self.res > 0.0 and math.isfinite(self.res) and math.isfinite(self.x0) and math.isfinite(self.y0)):
            raise ValueError("invalid frame geometry")

    @property
    def shape(self):
        return self.hit.shape

    def centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """(X, Y) world coordinates of every cell centre (cached)."""
        if self._xy is None:
            rows, cols = self.hit.shape
            X, Y = np.meshgrid(self.x0 + self.res * np.arange(cols), self.y0 + self.res * np.arange(rows))
            self._xy = (X, Y)
        return self._xy

    def upsampled(self, k: int) -> "EvidenceFrame":
        """Each cell split into k x k sub-cells (a coarse source on a finer lattice)."""
        k = int(k)
        if k <= 1:
            return self
        res = self.res / k
        off = -0.5 * self.res + 0.5 * res
        return EvidenceFrame(np.repeat(np.repeat(self.hit, k, 0), k, 1),
                             np.repeat(np.repeat(self.known, k, 0), k, 1),
                             self.x0 + off, self.y0 + off, res, self.frame_id,
                             None if self.clear is None else np.repeat(np.repeat(self.clear, k, 0), k, 1))

    @classmethod
    def from_elevation(cls, Z, x0: float, y0: float, res: float, frame_id: str = "odom",
                       step_height: float = 0.22, window: int = 5) -> "EvidenceFrame":
        """Height grid (NaN unknown) -> evidence = step test: max - min over
        a ``window`` x ``window`` neighbourhood >= ``step_height``.

        A free cell may clear earlier evidence (``clear``) only when its whole
        window is observed: the step test cannot tell 'no step' next to
        unobserved cells.  Walking along a tall face the depth camera keeps
        seeing the floor in front of it but not the face itself (its window
        half unknown); that floor cleared the confirmed face cell by cell and
        the robot walked into it (Gazebo w).  Such a cell still counts as seen
        free for the occlusion shadow (stricter there, the shadows closed real
        gaps: the C-OBS1 north gap in Gazebo s)."""
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2:
            raise ValueError("elevation grid must be 2-D")
        known = np.isfinite(Z)
        w = max(1, int(window) | 1)
        step = nan_max_filter(Z, w) - nan_min_filter(Z, w)
        hit = known & (np.nan_to_num(step, nan=0.0) >= float(step_height))
        gap = nan_max_filter(np.where(known, 0.0, 1.0), w) > 0.0       # an unobserved cell in the window
        return cls(hit, known, float(x0), float(y0), float(res), str(frame_id or ""), known & ~gap & ~hit)

    @classmethod
    def from_elevation_cloud(cls, xyz, frame_id: str = "odom", step_height: float = 0.22,
                             window: int = 5) -> "EvidenceFrame":
        """Organised (rows, cols, 3) /elevation_map/points payload -> evidence.
        Resolution and origin are recovered from the x / y coordinates (robust
        to float32 rounding and NaN coordinates)."""
        a = np.asarray(xyz, dtype=np.float64)
        if a.ndim != 3 or a.shape[2] < 3 or a.shape[0] < 2 or a.shape[1] < 2:
            raise ValueError("organised (rows, cols, 3) cloud expected")
        rows, cols = a.shape[:2]
        X, Y, Z = a[..., 0], a[..., 1], a[..., 2]
        with np.errstate(invalid="ignore"):
            cand = np.concatenate([(X[:, -1] - X[:, 0]) / (cols - 1), (Y[-1, :] - Y[0, :]) / (rows - 1)])
        cand = cand[np.isfinite(cand) & (cand > 0)]
        if cand.size == 0:
            raise ValueError("cloud has no valid x / y coordinates")
        res = round(float(np.median(cand)), 6)
        if not res > 0.0:
            raise ValueError("cloud resolution could not be recovered")
        with np.errstate(invalid="ignore"):
            dx = X - np.arange(cols)[None, :] * res
            dy = Y - np.arange(rows)[:, None] * res
        dx, dy = dx[np.isfinite(dx)], dy[np.isfinite(dy)]
        if dx.size == 0 or dy.size == 0:
            raise ValueError("cloud has no valid x / y coordinates")
        x0, y0 = _snap(float(np.median(dx)), res), _snap(float(np.median(dy)), res)
        return cls.from_elevation(Z, x0, y0, res, frame_id, step_height, window)

    @classmethod
    def from_traversability(cls, grid: OccupancyGridMap, threshold: int = 100) -> "EvidenceFrame":
        """OccupancyGrid cost grid (0 easy .. 100 untraversable, -1 unknown)
        -> evidence = cost >= ``threshold``."""
        occ = np.asarray(grid.occ)
        known = occ >= 0
        hit = known & (occ >= int(threshold))
        r = float(grid.resolution)
        return cls(hit, known, grid.origin_x + 0.5 * r, grid.origin_y + 0.5 * r, r, str(grid.frame_id or ""))


# =============================================================================
# memory
# =============================================================================
def _box3(a: np.ndarray) -> np.ndarray:
    """3x3 neighbourhood sum (zero outside)."""
    a = a.astype(np.int16)
    s = a.copy()
    for d in (1, -1):
        s += _shifted(a, d, 0, 0)
    t = s.copy()
    for d in (1, -1):
        t += _shifted(s, 0, d, 0)
    return t


def _disk_offsets(k: int, square: bool = False) -> List[Tuple[int, int]]:
    rr = (k + 0.5) ** 2 if square else k * k + 1e-9          # square: also the diagonal neighbours of k = 1
    return [(i, j) for i in range(-k, k + 1) for j in range(-k, k + 1) if i * i + j * j <= rr]


def _dilate(mask: np.ndarray, k: int, square: bool = False) -> np.ndarray:
    """Disk dilation by ``k`` cells (only around the bounding box of the mask)."""
    if k <= 0 or not mask.any():
        return mask.copy()
    rr, cc = np.nonzero(mask)
    r0, r1 = max(rr.min() - k, 0), min(rr.max() + k + 1, mask.shape[0])
    c0, c1 = max(cc.min() - k, 0), min(cc.max() + k + 1, mask.shape[1])
    sub = mask[r0:r1, c0:c1]
    out = sub.copy()
    for i, j in _disk_offsets(k, square):
        if i or j:
            out |= _shifted(sub, i, j, False)
    full = np.zeros_like(mask)
    full[r0:r1, c0:c1] = out
    return full


class ObstacleMemory:
    """World-fixed rolling obstacle lattice around the robot (see the module
    docstring).  ``update`` once per perception frame; the queries are cached
    per ``revision`` (incremented whenever the output layer changes)."""

    def __init__(self, params=None, robot_radius: float = 0.35):
        self.p = ObstacleParams.from_params(params) if not isinstance(params, ObstacleParams) else params
        self.robot_radius = float(robot_radius)
        self.res = float(self.p.resolution)
        self.n = max(8, int(round(self.p.memory_size / self.res)))
        # evidence this close to the static map is mapped structure; never as wide as a passage
        self.static_tolerance = min(float(self.p.static_tolerance),
                                    max(0.0, 2.0 * self.robot_radius - self.res - 1e-3))
        self._static = None
        self.revision = 0
        self.frames = 0
        self.rejected = 0
        self.clear()

    # ------------------------------------------------------------------ state
    def clear(self) -> None:
        n = self.n
        self.oi: Optional[int] = None          # world cell index of lattice column 0
        self.oj: Optional[int] = None          # world cell index of lattice row 0
        self.hits = np.zeros((n, n), dtype=np.int8)
        self.confirmed = np.zeros((n, n), dtype=bool)
        self.last_hit_t = np.full((n, n), -np.inf)
        self.first_hit_t = np.full((n, n), np.nan)      # first evidence since the cell was clear
        self.seen_free_t = np.full((n, n), np.nan)
        # robot position of the first evidence of each cell (line of sight of the occlusion shadow)
        self.view_x = np.full((n, n), np.nan, dtype=np.float32)
        self.view_y = np.full((n, n), np.nan, dtype=np.float32)
        self.raw = np.zeros((n, n), dtype=bool)
        self.raw_t = -np.inf
        self._output = np.zeros((n, n), dtype=bool)
        self.t_last: Optional[float] = None
        self._t_frame = -math.inf               # last accepted frame (frame_due)
        self.down_since: Optional[float] = None    # set_upright(False): robot down, fusion suspended
        self._settle = (math.inf, -math.inf)       # set_upright(True): no fusion in [t_up, t_up + settle)
        self.fall_forgotten = 0                    # cells forgotten after falls (status / tests)
        self._novel = None
        self._novel_key = None
        self.revision += 1
        self._cache: Dict = {}

    def set_static_map(self, grid) -> None:
        """Static nav map for the novelty filter / combined grids (None = map-free)."""
        if grid is self._static:
            return
        self._static = grid
        self._novel = None
        self._novel_key = None
        self._invalidate()

    @property
    def static_map(self):
        return self._static

    def _invalidate(self) -> None:
        self._cache = {}

    def note_rejected(self) -> None:
        self.rejected += 1

    # ------------------------------------------------------------ lattice
    def _recenter(self, x: float, y: float) -> None:
        n, res = self.n, self.res
        ci, cj = int(math.floor(x / res)), int(math.floor(y / res))
        if self.oi is None:
            self.oi, self.oj = ci - n // 2, cj - n // 2
            self._novel_key = None
            return
        dc, dr = ci - (self.oi + n // 2), cj - (self.oj + n // 2)
        if max(abs(dc), abs(dr)) <= n // 4:
            return
        self.hits = _shifted(self.hits, dr, dc, 0)
        self.confirmed = _shifted(self.confirmed, dr, dc, False)
        self.last_hit_t = _shifted(self.last_hit_t, dr, dc, -np.inf)
        self.first_hit_t = _shifted(self.first_hit_t, dr, dc, np.nan)
        self.seen_free_t = _shifted(self.seen_free_t, dr, dc, np.nan)
        self.view_x = _shifted(self.view_x, dr, dc, np.nan)
        self.view_y = _shifted(self.view_y, dr, dc, np.nan)
        self.raw = _shifted(self.raw, dr, dc, False)
        self._output = _shifted(self._output, dr, dc, False)
        self.oi += dc
        self.oj += dr
        self._novel_key = None
        self.revision += 1
        self._invalidate()

    def lattice_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """(xs, ys) world centres of the lattice columns / rows."""
        n, res = self.n, self.res
        oi = self.oi if self.oi is not None else 0
        oj = self.oj if self.oj is not None else 0
        return (oi + np.arange(n) + 0.5) * res, (oj + np.arange(n) + 0.5) * res

    def _novelty(self) -> np.ndarray:
        key = (self.oi, self.oj, id(self._static))
        if self._novel is not None and self._novel_key == key:
            return self._novel
        n = self.n
        g = self._static
        if g is None or self.oi is None:
            nov = np.ones((n, n), dtype=bool)
        else:
            xs, ys = self.lattice_centres()
            X, Y = np.meshgrid(xs, ys)
            ix, iy = g.world_to_grid_array(np.stack([X, Y], -1))
            ok = (ix >= 0) & (ix < g.width) & (iy >= 0) & (iy < g.height)
            nov = np.ones((n, n), dtype=bool)                 # outside the static map: novel
            nov[ok] = g.distance_field()[iy[ok], ix[ok]] >= self.static_tolerance
        self._novel, self._novel_key = nov, key
        return nov

    # ---------------------------------------------------------- fall guard
    @property
    def suspended(self) -> bool:
        """True while the robot is down (``set_upright(False)``): frames are not fused."""
        return self.down_since is not None

    def set_upright(self, upright: bool, t: float, lookback: float = FALL_LOOKBACK,
                    settle: float = FALL_SETTLE) -> int:
        """Locomotion state of the robot carrying the camera.  Down (fallen,
        self-righting): the frames are not fused.  Up again: the evidence
        first seen since ``lookback`` s before the robot was reported down is
        forgotten (the camera tipped over with the body before the fall was
        detected) and fusion resumes ``settle`` s later.  Returns the number
        of forgotten cells (0 when nothing changed)."""
        t = float(t)
        if not upright:
            if self.down_since is None:
                self.down_since = t
            return 0
        if self.down_since is None:
            return 0
        t_down, self.down_since = self.down_since, None
        self._settle = (t, t + max(0.0, float(settle)))
        return self.forget_since(t_down - max(0.0, float(lookback)))

    def _paused(self, t: float) -> bool:
        """Robot down or still settling after the stand-up (a clock that jumped
        back before the stand-up ends the settle window)."""
        return self.down_since is not None or self._settle[0] <= t < self._settle[1]

    def forget_since(self, t0: float) -> int:
        """Forget every cell whose first evidence (since it was last clear) is
        at or after ``t0``.  Returns the number of cells forgotten."""
        new = (self.hits > 0) & (self.first_hit_t >= float(t0))
        k = int(np.count_nonzero(new))
        if not k:
            return 0
        self.hits[new] = 0
        self.confirmed[new] = False
        self.first_hit_t[new] = np.nan
        self.view_x[new] = np.nan
        self.view_y[new] = np.nan
        self.raw[:] = False
        self.raw_t = -np.inf
        self.fall_forgotten += k
        out = self._compute_output(self._novelty())
        if not np.array_equal(out, self._output):
            self._output = out
            self.revision += 1
        self._invalidate()
        return k

    # ------------------------------------------------------------- update
    def frame_due(self, t: float, driving: bool = True) -> bool:
        """Should a frame arriving at time ``t`` be converted and fused?  Every
        frame while the owner drives (or ``idle_period`` is 0), otherwise one
        per ``idle_period`` s: the memory stays reasonably current while the
        robot stands / inspects / another node drives, at a fraction of the
        CPU (three nodes fuse the same 5 Hz cloud).  Call before converting
        the message; a True answer counts as the accepted frame."""
        if not self.p.enabled:
            return False
        t = float(t)
        if t < self._t_frame - 1.0:                  # clock jumped back (sim reset)
            self._t_frame = -math.inf
        if self._paused(t):
            return False                             # robot down / settling: the frame is not fused
        if driving or self.p.idle_period <= 0.0 or t - self._t_frame >= self.p.idle_period - 1e-6:
            self._t_frame = t
            return True
        return False

    def update(self, frame: EvidenceFrame, pose, t: float) -> bool:
        """Fuse one evidence frame seen from ``pose`` (x, y[, yaw]) at time t.
        Returns True when the output layer changed."""
        p = self.p
        if not p.enabled or frame is None:
            return False
        t = float(t)
        if self._paused(t):
            return False                             # robot down / settling (set_upright)
        x, y = float(pose[0]), float(pose[1])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(t)):
            return False
        had = bool(self._output.any())
        if self.t_last is not None and t < self.t_last - 1.0:      # clock jumped back (sim reset)
            self.clear()
        self._recenter(x, y)
        n, res = self.n, self.res
        if frame.res > 1.01 * res:
            frame = frame.upsampled(int(math.ceil(frame.res / res - 1e-6)))
        X, Y = frame.centres()
        r2 = (X - x) ** 2 + (Y - y) ** 2
        ring = (r2 >= p.ignore_radius ** 2) & (r2 <= p.max_range ** 2)
        li = np.floor(X / res + 1e-6).astype(np.int64) - self.oi
        lj = np.floor(Y / res + 1e-6).astype(np.int64) - self.oj
        inside = ring & (li >= 0) & (li < n) & (lj >= 0) & (lj < n)
        idx = (lj * n + li)[inside]
        known = frame.known[inside]
        hit = frame.hit[inside]
        nn = n * n
        n_tot = np.bincount(idx, minlength=nn)
        n_known = np.bincount(idx[known], minlength=nn)
        n_hit = np.bincount(idx[hit], minlength=nn)
        HIT = (n_hit >= 1).reshape(n, n)
        need = np.maximum(1, np.ceil(p.min_known_fraction * n_tot))
        FREE = ((n_hit == 0) & (n_tot > 0) & (n_known >= need)).reshape(n, n)
        if frame.clear is not None:                          # free observations that may clear evidence
            n_clear = np.bincount(idx[frame.clear[inside]], minlength=nn)
            CLEAR = ((n_hit == 0) & (n_tot > 0) & (n_clear >= need)).reshape(n, n)
        else:
            CLEAR = FREE
        h = self.hits.astype(np.int16)
        first = HIT & ((h == 0) | np.isnan(self.view_x))     # first evidence since the cell was clear
        h[HIT] = np.minimum(h[HIT] + 1, p.max_hits)
        h[CLEAR] = np.maximum(h[CLEAR] - 1, 0)
        self.hits = h.astype(np.int8)
        self.last_hit_t[HIT] = t
        # line of sight of the occlusion shadow: the viewpoint of the first evidence (stable while the
        # robot swings out round the obstacle - the latest viewpoint would rotate the assumed back
        # towards the detour and refine it again and again)
        self.view_x[first] = x
        self.view_y[first] = y
        self.first_hit_t[first] = t
        self.seen_free_t[FREE] = t
        self.confirmed |= self.hits >= p.confirm_hits
        self.confirmed &= self.hits > 0
        if p.forget_time > 0.0:
            stale = (self.hits > 0) & (t - self.last_hit_t > p.forget_time)
            if stale.any():
                self.hits[stale] = 0
                self.confirmed[stale] = False
                self.first_hit_t[stale] = np.nan
        nov = self._novelty()
        out = self._compute_output(nov)
        raw = HIT & nov
        if p.min_cluster > 1 and raw.any():
            # same speckle rule as the output layer: an isolated evidence cell that the planners
            # ignore must not stop the robot either (the governor would hold it in front of
            # nothing until the stuck logic skips the goal)
            raw &= _box3(raw | out) >= p.min_cluster
        self.raw = raw
        self.raw_t = t
        self._cache.pop("raw_cells", None)
        changed = had != bool(out.any()) or not np.array_equal(out, self._output)
        if changed:
            self._output = out
            self.revision += 1
            self._invalidate()
        self.frames += 1
        self.t_last = t
        return changed

    def _compute_output(self, nov: np.ndarray) -> np.ndarray:
        c = self.confirmed & nov
        if self.p.min_cluster <= 1 or not c.any():
            return c
        return c & (_box3(c) >= self.p.min_cluster)

    # ------------------------------------------------------------ layers
    def output_mask(self) -> np.ndarray:
        """Confirmed novel obstacle cells (read only)."""
        return self._output

    def _seen(self, k: int = 0) -> np.ndarray:
        """Cells seen free recently (and not output) - where the occlusion
        shadow ends.  A free observation counts only when it is not older than
        the latest evidence of the output cells within ``k`` cells: from
        farther than ~1.2 m the camera does not reach the step of a tall face,
        so the face and the cells right behind it look free until the face is
        detected - such stale observations say nothing about its back."""
        t = self.t_last if self.t_last is not None else 0.0
        out = self._output
        with np.errstate(invalid="ignore"):
            seen = (self.seen_free_t >= t - SHADOW_FRESH_S) & ~out
            if k > 0 and out.any():
                t_obs = _separable(np.where(out, self.last_hit_t, -np.inf), int(k), np.maximum, -np.inf)
                seen &= self.seen_free_t >= t_obs - 1e-6
        return seen

    def near_mask(self) -> np.ndarray:
        """Output grown by min(shadow_depth, SHADOW_NEAR) into cells not seen
        free recently - the short occlusion shadow - plus the unobserved
        continuation of each face (:meth:`_face_extension`, shadow_depth);
        the fallback planning layer where the full shadow leaves no way (read
        only, cached).  Not directional, so any free observation counts
        (cells beside the robot, inside the own-body radius, cannot be
        refreshed)."""
        nm = self._cache.get("near")
        if nm is not None:
            return nm
        out = self._output
        k = int(round(min(float(self.p.shadow_depth), SHADOW_NEAR) / self.res))
        nm = out | (_dilate(out, k) & ~self._seen()) if (k > 0 and out.any()) else out
        if float(self.p.shadow_depth) > SHADOW_NEAR + 1e-9 and out.any() and self.oi is not None:
            nm = nm | self._face_extension(out, float(self.p.shadow_depth))
        self._cache["near"] = nm
        return nm

    def planning_mask(self) -> np.ndarray:
        """Output plus its occlusion shadow (the unseen back of an obstacle),
        for planning only (read only):

        * line of sight: every 8-connected cluster of output cells is
          extruded away from the mean robot position of its first evidence, up to
          ``shadow_depth``; an extrusion ray ends at the first cell seen free
          since the evidence of its source cell.  The depth camera sees the
          front face of a box but neither its sides (outside the field of
          view / inside the own-body radius while passing) nor its back, so a
          detour must not rejoin the route through space never seen free (one
          direction per cluster: the hidden part of a box lies behind its
          visible face, a per-cell perspective fan would close real gaps
          beside it);
        * the near shadow and face continuation (:meth:`near_mask`) and one
          cell of gap filling.

        Cached per revision (recomputed when the output layer changes)."""
        pm = self._cache.get("planning")
        if pm is not None:
            return pm
        out = self._output
        depth = float(self.p.shadow_depth)
        near = self.near_mask()
        if depth <= SHADOW_NEAR + 1e-9 or not out.any() or self.oi is None:
            pm = near
        else:
            kd = int(math.ceil(depth / self.res - 1e-6))
            shadow = self._los_shadow(out, depth)
            if shadow.any():
                shadow |= _dilate(shadow, 1) & ~self._seen(kd) & _dilate(out, kd)
            pm = near | shadow
        self._cache["planning"] = pm
        return pm

    def _cluster_views(self, out: np.ndarray):
        """Per output cell of the clusters with a known viewpoint: (cell centre
        x, y in lattice units, unit direction viewpoint -> cluster centroid,
        cluster label, evidence time) or None."""
        res = self.res
        lab = _label_components(out)
        jj, ii = np.nonzero(out)
        lb = lab[jj, ii]
        nl = int(lab.max()) + 1
        vx = self.view_x[jj, ii].astype(float) / res - self.oi        # viewpoints in lattice units
        vy = self.view_y[jj, ii].astype(float) / res - self.oj
        ok = np.isfinite(vx) & np.isfinite(vy)
        if not ok.any():
            return None
        cx, cy = ii + 0.5, jj + 0.5
        n_all = np.maximum(np.bincount(lb, minlength=nl), 1)
        n_v = np.bincount(lb[ok], minlength=nl)
        mvx = np.bincount(lb[ok], weights=vx[ok], minlength=nl) / np.maximum(n_v, 1)
        mvy = np.bincount(lb[ok], weights=vy[ok], minlength=nl) / np.maximum(n_v, 1)
        dx = np.bincount(lb, weights=cx, minlength=nl) / n_all - mvx            # viewpoint -> centroid
        dy = np.bincount(lb, weights=cy, minlength=nl) / n_all - mvy
        L = np.hypot(dx, dy)
        valid = (n_v > 0) & (L > 1e-6)
        alive = valid[lb]
        if not alive.any():
            return None
        ux = np.where(valid, dx / np.where(valid, L, 1.0), 0.0)[lb][alive]
        uy = np.where(valid, dy / np.where(valid, L, 1.0), 0.0)[lb][alive]
        t_src = self.last_hit_t[jj, ii][alive] - 1e-6               # free observations must be this recent
        return cx[alive], cy[alive], ux, uy, lb[alive], t_src

    def _march(self, out: np.ndarray, cx, cy, ux, uy, t_src, depth: float) -> np.ndarray:
        """Rays from cell centres (cx, cy) along (ux, uy), half-cell steps up to
        ``depth`` m, each ending at the first cell seen free since t_src."""
        n, res = self.n, self.res
        shadow = np.zeros((n, n), dtype=bool)
        if len(cx) == 0:
            return shadow
        t = self.t_last if self.t_last is not None else 0.0
        with np.errstate(invalid="ignore"):
            fresh = (self.seen_free_t >= t - SHADOW_FRESH_S) & ~out
        sft = np.where(fresh, self.seen_free_t, -np.inf)
        alive = np.ones(len(cx), dtype=bool)
        for s in np.arange(0.5, depth / res + 1e-9, 0.5):           # half-cell steps
            pi = np.floor(cx + s * ux).astype(np.int64)
            pj = np.floor(cy + s * uy).astype(np.int64)
            alive &= (pi >= 0) & (pi < n) & (pj >= 0) & (pj < n)
            if not alive.any():
                break
            a = np.nonzero(alive)[0]
            stop = sft[pj[a], pi[a]] >= t_src[a]
            alive[a[stop]] = False
            a = a[~stop]
            shadow[pj[a], pi[a]] = True
        return shadow & ~out

    def _los_shadow(self, out: np.ndarray, depth: float) -> np.ndarray:
        """Cells behind the output clusters along their line of sight (see planning_mask)."""
        cv = self._cluster_views(out)
        if cv is None:
            return np.zeros((self.n, self.n), dtype=bool)
        cx, cy, ux, uy, _, t_src = cv
        return self._march(out, cx, cy, ux, uy, t_src, depth)

    def _face_extension(self, out: np.ndarray, depth: float) -> np.ndarray:
        """The unobserved continuation of a detected face: from the end cells of
        every cluster (extremes along the face = perpendicular to its line of
        sight) along the face, up to ``depth``, until a cell seen free since
        the evidence.  The depth camera sees a face only inside its field of
        view and its 1.0 .. 1.45 m band: walking along a wall at 0.5 .. 0.7 m
        its continuation is never seen, and a detour rounding the 'end' of the
        part seen walks into it (Gazebo w).  A box seen whole has the ground
        beside its ends seen free: no extension."""
        cv = self._cluster_views(out)
        if cv is None:
            return np.zeros((self.n, self.n), dtype=bool)
        cx, cy, ux, uy, lb, t_src = cv
        tx, ty = -uy, ux                                             # along the face
        sp = cx * tx + cy * ty
        nl = int(lb.max()) + 1
        smin = np.full(nl, np.inf)
        smax = np.full(nl, -np.inf)
        np.minimum.at(smin, lb, sp)
        np.maximum.at(smax, lb, sp)
        lo, hi = sp <= smin[lb] + 1.0, sp >= smax[lb] - 1.0          # the end cells (one cell deep)
        ext = self._march(out, cx[lo], cy[lo], -tx[lo], -ty[lo], t_src[lo], depth)
        ext |= self._march(out, cx[hi], cy[hi], tx[hi], ty[hi], t_src[hi], depth)
        return ext

    def raw_mask(self, t: Optional[float] = None) -> np.ndarray:
        """Novel evidence of the latest frame (empty when older than RAW_MAX_AGE)."""
        if t is not None and float(t) - self.raw_t > RAW_MAX_AGE:
            return np.zeros_like(self.raw)
        return self.raw

    def has_confirmed(self) -> bool:
        return bool(self._output.any())

    @property
    def confirmed_count(self) -> int:
        return int(np.count_nonzero(self._output))

    def _cells(self, layer: str, t: Optional[float] = None) -> np.ndarray:
        """(M, 2) world centres of the cells of a layer."""
        if layer == "raw":
            if t is not None and float(t) - self.raw_t > RAW_MAX_AGE:
                return np.zeros((0, 2))
            key = "raw_cells"
            mask = self.raw
        elif layer == "planning":
            key, mask = "planning_cells", self.planning_mask()
        elif layer == "near":
            key, mask = "near_cells", self.near_mask()
        elif layer == "output":
            key, mask = "output_cells", self._output
        else:
            raise ValueError(f"unknown layer {layer!r}")
        c = self._cache.get(key)
        if c is None:
            if self.oi is None or not mask.any():
                c = np.zeros((0, 2))
            else:
                jj, ii = np.nonzero(mask)
                c = np.stack([(self.oi + ii + 0.5) * self.res, (self.oj + jj + 0.5) * self.res], -1)
            self._cache[key] = c
        return c

    def layer_cells(self, layer: str = "output", t: Optional[float] = None) -> np.ndarray:
        return self._cells(layer, t)

    def confirmed_xy(self) -> np.ndarray:
        return self._cells("output")

    def raw_xy(self, t: Optional[float] = None) -> np.ndarray:
        return self._cells("raw", t)

    def distances(self, xy, layer: str = "output", max_dist: float = math.inf,
                  t: Optional[float] = None) -> np.ndarray:
        """Distance [m] from each point (N, 2) to the nearest cell centre of
        ``layer`` (inf when none within ``max_dist``)."""
        P = np.asarray(xy, dtype=float).reshape(-1, 2)
        out = np.full(len(P), np.inf)
        C = self._cells(layer, t)
        if len(P) == 0 or len(C) == 0:
            return out
        if math.isfinite(max_dist):
            lo, hi = P.min(0) - max_dist, P.max(0) + max_dist
            sel = (C[:, 0] >= lo[0]) & (C[:, 0] <= hi[0]) & (C[:, 1] >= lo[1]) & (C[:, 1] <= hi[1])
            C = C[sel]
            if len(C) == 0:
                return out
        step = max(1, 200000 // max(len(C), 1))
        for a in range(0, len(P), step):
            Q = P[a:a + step]
            d = np.hypot(Q[:, None, 0] - C[None, :, 0], Q[:, None, 1] - C[None, :, 1]).min(1)
            out[a:a + step] = d
        if math.isfinite(max_dist):
            out[out > max_dist] = np.inf
        return out

    def blocked(self, xy, radius: float, layer: str = "output", t: Optional[float] = None) -> np.ndarray:
        """Per point: a ``layer`` cell centre closer than ``radius``."""
        return self.distances(xy, layer, max_dist=float(radius), t=t) < float(radius)

    def nearest(self, x: float, y: float, max_dist: float = 1.0) -> Tuple[float, np.ndarray]:
        """(distance to the nearest confirmed cell, unit vector pointing away
        from it), (inf, 0) when none within ``max_dist``."""
        C = self._cells("output")
        if len(C) == 0 or not (math.isfinite(x) and math.isfinite(y)):
            return math.inf, np.zeros(2)
        d = np.hypot(C[:, 0] - x, C[:, 1] - y)
        k = int(np.argmin(d))
        if d[k] > max_dist:
            return math.inf, np.zeros(2)
        v = np.array([x - C[k, 0], y - C[k, 1]])
        nv = float(np.hypot(*v))
        return float(d[k]), (v / nv if nv > 1e-9 else np.zeros(2))

    # ------------------------------------------------------------- grids
    def _mask(self, layer: str) -> np.ndarray:
        if layer == "planning":
            return self.planning_mask()
        if layer == "near":
            return self.near_mask()
        return self._output

    def combined_grid(self, static_grid=None, layer: str = "planning", grow: float = 0.0) -> OccupancyGridMap:
        """Static map with the ``layer`` cells painted occupied (new grid,
        cached until the memory changes).  ``grow`` [m] dilates only the
        detected cells: a planner inflating by robot_radius then keeps
        robot_radius + grow from detected obstacles but only robot_radius
        from mapped walls (tight mapped passages stay passable).  Without a
        static map: a grid over the memory window (unobserved space free)."""
        static_grid = self._static if static_grid is None else static_grid
        kg = int(round(max(float(grow), 0.0) / self.res))
        key = ("combined", layer, self.revision, id(static_grid), kg)
        c = self._cache.get(key)
        if c is not None and c[0] is static_grid:
            return c[1]
        mask = self._mask(layer)
        if kg > 0:
            mask = _dilate(mask, kg, square=True)
        res = self.res
        if static_grid is None:
            occ = np.zeros((self.n, self.n), dtype=np.int8)
            occ[mask] = 100
            oi = self.oi if self.oi is not None else -(self.n // 2)
            oj = self.oj if self.oj is not None else -(self.n // 2)
            g = OccupancyGridMap(occ, res, oi * res, oj * res, "odom")
        else:
            occ = static_grid.occ.copy()
            if mask.any():
                jj, ii = np.nonzero(mask)
                cx, cy = (self.oi + ii + 0.5) * res, (self.oj + jj + 0.5) * res
                s = float(static_grid.resolution)
                k = max(1, int(math.ceil(res / s - 1e-6)))
                offs = ((np.arange(k) + 0.5) / k - 0.5) * res
                for ox in offs:
                    for oy in offs:
                        ix, iy = static_grid.world_to_grid_array(np.stack([cx + ox, cy + oy], -1))
                        ok = (ix >= 0) & (ix < static_grid.width) & (iy >= 0) & (iy < static_grid.height)
                        occ[iy[ok], ix[ok]] = 100
            g = OccupancyGridMap(occ, static_grid.resolution, static_grid.origin_x, static_grid.origin_y,
                                 static_grid.frame_id)
        self._cache[key] = (static_grid, g)
        return g

    def window_grid(self, static_grid, x0: float, y0: float, x1: float, y1: float, static_cost=None,
                    layer: str = "planning", grow: float = 0.0):
        """(combined grid cropped to the rectangle, matching crop of the static
        cost layer or None); (None, None) when the rectangle misses the grid."""
        static_grid = self._static if static_grid is None else static_grid
        full = self.combined_grid(static_grid, layer, grow)
        sub, (ix0, iy0) = submap(full, x0, y0, x1, y1)
        if sub is None:
            return None, None
        cost = None
        if static_cost is not None and static_grid is not None:
            sc = np.asarray(static_cost, dtype=float)
            if sc.shape == full.occ.shape:
                cost = sc[iy0:iy0 + sub.height, ix0:ix0 + sub.width].copy()
        return sub, cost

    def status(self) -> dict:
        return {"frames": int(self.frames), "rejected": int(self.rejected), "confirmed": self.confirmed_count,
                "revision": int(self.revision)}


# =============================================================================
# path checks
# =============================================================================
@dataclass
class Blockage:
    k: int                     # path vertex index at / before the first blocked sample
    s: float                   # [m] free travel along the path to it
    xy: Tuple[float, float]    # the blocked sample


def _xy(path) -> np.ndarray:
    """Path (list of (x, y[, ...]) or array) -> (N, 2) float array."""
    if isinstance(path, np.ndarray) and path.ndim == 2 and path.shape[1] >= 2:
        return path[:, :2].astype(float, copy=False)
    if path is None or len(path) == 0:
        return np.zeros((0, 2))
    return np.asarray([(float(q[0]), float(q[1])) for q in path], dtype=float)


def _samples(path, k0: int, length: float, step: float = 0.1):
    """Samples of path[k0:] every <= step m up to ``length`` m of arclength:
    (points (N, 2), arclength (N,), source vertex index (N,))."""
    P = _xy(path)[max(int(k0), 0):]
    if len(P) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=int)
    pts, ss, ks = [P[0]], [0.0], [0]
    s = 0.0
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        L = float(np.hypot(*(b - a)))
        if L < 1e-9:
            continue
        m = max(1, int(math.ceil(L / step - 1e-9)))
        for j in range(1, m + 1):
            sj = s + L * j / m
            if sj > length + 1e-9:
                return np.asarray(pts), np.asarray(ss), np.asarray(ks) + int(k0)
            pts.append(a + (b - a) * (j / m))
            ss.append(sj)
            ks.append(i if j < m else i + 1)
        s += L
    return np.asarray(pts), np.asarray(ss), np.asarray(ks) + int(k0)


def path_blockage(memory: ObstacleMemory, path, k0: int, robot_radius: float, lookahead: float,
                  layer: str = "output", t: Optional[float] = None, robot_xy=None) -> Optional[Blockage]:
    """First sample of path[k0:] (every <= 0.1 m, up to ``lookahead`` m) whose
    ``robot_radius`` circle touches a ``layer`` cell, or None.

    ``robot_xy``: a robot already closer than ``robot_radius`` to a cell (a
    side of the obstacle confirmed only while passing it) may leave that
    zone: along the initial part of the path, until it first gets out of the
    zone (at most LEAVE_LEN m), a sample is blocked only when it is closer
    than the robot is now (minus 5 cm).  Without it, every path from such a
    robot - including a fresh detour - would be 'blocked' at 0 m and
    re-planned forever; a path that stays in the zone longer is blocked."""
    if path is None or len(path) == 0 or (layer != "raw" and not memory.has_confirmed()):
        return None
    P, S, K = _samples(path, k0, lookahead)
    if len(P) == 0:
        return None
    r = float(robot_radius)
    d = memory.distances(P, layer, r + 0.5, t)
    b = d < r
    if b.any() and robot_xy is not None:
        d_now = float(memory.distances([robot_xy], layer, r + 0.5, t)[0])
        if d_now < r:
            out = np.nonzero((d >= r) | (S > LEAVE_LEN))[0]
            n_init = int(out[0]) if len(out) else len(d)
            b[:n_init] = d[:n_init] < d_now - 0.05
    if not b.any():
        return None
    k = int(np.argmax(b))
    return Blockage(int(K[k]), float(S[k]), (float(P[k, 0]), float(P[k, 1])))


def pursuit_path(path, robot_xy, lead: float = 0.0) -> np.ndarray:
    """The line a pure-pursuit follower actually drives, for the governor:
    robot -> the first vertex at >= ``lead`` m from the robot (searched from
    the closest vertex, as dusan_common.pure_pursuit does) -> rest of the
    path; ``lead`` <= 0: robot -> closest vertex -> rest.

    The followed path alone starts where the robot was when it was planned:
    after a drift, an in-place turn or a re-plan that starts beside the robot,
    the free travel along it is not the robot's (Gazebo w: the follower cut
    towards a wall 0.2 m further than the path said and stopped 0.43 m from
    it instead of 0.6 m)."""
    P = _xy(path)
    if len(P) == 0 or robot_xy is None:
        return P
    r = np.asarray([float(robot_xy[0]), float(robot_xy[1])], dtype=float)
    d = np.hypot(P[:, 0] - r[0], P[:, 1] - r[1])
    k = int(np.argmin(d[:min(len(d), 60)]))
    if lead > 0.0:
        far = np.nonzero(d[k:] >= float(lead))[0]
        k = k + int(far[0]) if len(far) else len(P) - 1
    return np.vstack([r[None, :], P[k:]])


def _free_travel(memory: ObstacleMemory, P, S, robot_xy, radius: float, layer: str, t) -> float:
    """Arclength to the first sample closer than the effective radius to a
    ``layer`` cell.  A robot already inside the clearance zone may move as
    long as it does not get closer than it is now."""
    C = memory.layer_cells(layer, t)
    if len(C) == 0 or len(P) == 0:
        return math.inf
    d_now = float(memory.distances([robot_xy], layer, radius + 0.5, t)[0])
    r_eff = min(radius, d_now - 0.05) if math.isfinite(d_now) else radius
    d = memory.distances(P, layer, radius + 0.5, t)
    blk = d < r_eff
    if not blk.any():
        return math.inf
    return float(S[int(np.argmax(blk))])


def speed_scale(memory: ObstacleMemory, path_ahead, robot_xy, robot_radius: float, params: ObstacleParams,
                t: Optional[float] = None) -> Tuple[float, str, float]:
    """Slow-down / stop governor -> (scale 0..1, reason, confirmed free travel).

    d_c / d_r = free travel along the followed path to the first sample whose
    ``robot_radius`` circle touches a confirmed / latest-frame (unconfirmed)
    cell; ramp(d) = clip((d - stop) / (slow - stop), 0, 1):

        scale = min(ramp(d_c),  0 if d_r < stop else max(unconfirmed_scale, ramp(d_r)) if d_r < slow else 1)
    """
    p = params
    if not p.enabled or not p.slowdown_enabled or path_ahead is None or len(path_ahead) == 0:
        return 1.0, "", math.inf
    P, S, _ = _samples(path_ahead, 0, p.slow_dist + 0.1, step=0.05)
    d_c = _free_travel(memory, P, S, robot_xy, robot_radius, "output", t)
    d_r = _free_travel(memory, P, S, robot_xy, robot_radius, "raw", t)
    span = max(p.slow_dist - p.stop_dist, 1e-6)

    def ramp(d):
        return float(min(max((d - p.stop_dist) / span, 0.0), 1.0))

    sc_c = ramp(d_c) if math.isfinite(d_c) else 1.0
    if d_r < p.stop_dist:
        sc_r = 0.0
    elif d_r < p.slow_dist:
        sc_r = max(p.unconfirmed_scale, ramp(d_r))
    else:
        sc_r = 1.0
    scale = min(sc_c, sc_r)
    reason = ""
    if scale < 1.0:
        reason = ("obstacle ahead" if sc_c <= sc_r else "unconfirmed obstacle ahead") + \
                 (" - stop" if scale <= 0.0 else " - slowing")
    return scale, reason, d_c


# =============================================================================
# detours
# =============================================================================
@dataclass
class SideCommitment:
    """Remembers the side chosen around an obstacle (anchor within ``radius``)
    for ``commit_time`` s."""
    commit_time: float = 15.0
    radius: float = 1.5
    entries: List[Tuple[str, Tuple[float, float], float]] = field(default_factory=list)

    def _prune(self, t: float) -> None:
        self.entries = [e for e in self.entries if 0.0 <= t - e[2] <= self.commit_time]

    def preferred(self, anchor, t: float) -> Optional[str]:
        self._prune(float(t))
        best, best_t = None, -math.inf
        for side, a, te in self.entries:
            if math.hypot(a[0] - anchor[0], a[1] - anchor[1]) <= self.radius and te > best_t:
                best, best_t = side, te
        return best

    def commit(self, side: str, anchor, t: float) -> None:
        t = float(t)
        self.entries = [e for e in self.entries
                        if math.hypot(e[1][0] - anchor[0], e[1][1] - anchor[1]) > self.radius]
        self.entries.append((side, (float(anchor[0]), float(anchor[1])), t))
        self._prune(t)

    def clear(self) -> None:
        self.entries = []


@dataclass
class Detour:
    path: List[Tuple[float, float]]      # robot -> around the obstacle -> base[k_rejoin:] (0.1 m spacing)
    side: str                            # 'left' | 'right' (w.r.t. the base route direction)
    k_rejoin: int                        # base index where the detour rejoins
    length: float                        # [m] robot -> rejoin point along the detour
    replaced: float                      # [m] base route length it replaces
    anchor: Tuple[float, float]          # obstacle cell nearest to the blocked section
    free_travel: float = math.inf        # [m] to the first blocked base sample
    rejoin: Optional[Tuple[float, float]] = None     # base[k_rejoin] (end of the planned head)
    layer: str = "planning"                          # memory layer the head was planned on
    cost: float = math.nan                           # length + terrain-cost integral of the head (A*'s metric)


def _cumlen(P: np.ndarray) -> np.ndarray:
    if len(P) < 2:
        return np.zeros(len(P))
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))])


SMOOTH_EXTRA = 0.10        # [m] detour shortcuts keep this much more clearance (pure-pursuit corner cutting)
REJOIN_SLACK = 0.15        # [m] the rejoin point keeps this much more from the planning layer: the
#                            assumed back grows a little as more of the face is confirmed, a rejoin
#                            exactly at robot_radius + margin would then be 'blocked' and re-planned


def _terrain_integral(grid, cost, path) -> float:
    """Integral [cost x m] of the terrain cost layer (shape of ``grid``) along a
    polyline; A* adds cost x step per cell, so length + this integral is the
    cost A* minimises (clearance term aside).  0 without a cost layer."""
    if cost is None or path is None or len(path) < 2:
        return 0.0
    c = np.asarray(cost, dtype=float)
    if c.shape != grid.occ.shape or not np.any(c > 0.0):
        return 0.0
    return float(sum(_segment_cost(grid, c, a, b)[0] for a, b in zip(path[:-1], path[1:])))


def check_radius(robot_radius: float, p: ObstacleParams) -> float:
    """Blockage check radius: robot_radius + margin / 2.  Detours and dynamic
    re-plans keep robot_radius + margin, so a fresh plan is never 'blocked'
    (hysteresis), while cells detected later (the rest of a partly seen
    obstacle) that come closer than this trigger a refined detour."""
    return float(robot_radius) + 0.5 * float(p.margin)


class DetourPlanner:
    """Local detour around a confirmed obstacle blocking the base route.

    The window grid holds the static map plus the planning layer grown by
    ``margin``; A* inflates by robot_radius, so the detour keeps
    robot_radius + margin from detected obstacles (hysteresis against the
    blockage check at robot_radius) but only robot_radius from mapped walls
    (tight mapped passages stay usable).  Shortcuts of the smoother keep
    SMOOTH_EXTRA more where possible (pure pursuit cuts corners)."""

    def __init__(self, params=None, robot_radius: float = 0.35):
        self.p = ObstacleParams.from_params(params) if not isinstance(params, ObstacleParams) else params
        self.robot_radius = float(robot_radius)

    def plan(self, memory: ObstacleMemory, base, k_robot: int, robot_xy, static_grid=None, static_cost=None,
             commitment: Optional[SideCommitment] = None, t: float = 0.0, layer: str = "planning"):
        """-> (Detour | None, reason).  ``layer``: 'planning' (full occlusion
        shadow) or 'near' (short shadow, the fallback where the full one
        leaves no way)."""
        p = self.p
        r = self.robot_radius
        r_chk = check_radius(r, p)
        r_plan = r + p.margin
        B = np.asarray(base, dtype=float).reshape(len(base), -1)[:, :2] if base is not None and len(base) else None
        if B is None or len(B) < 2:
            return None, "no route"
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        k_robot = int(min(max(int(k_robot), 0), len(B) - 1))
        S = _cumlen(B)
        s0 = S[k_robot]
        k_end = int(min(np.searchsorted(S, s0 + p.lookahead + 6.0, side="right"), len(B)))
        idx = np.arange(k_robot, k_end)
        blk = memory.blocked(B[idx], r_chk, "output")
        if not blk.any():
            # a detour is also refined when only the occlusion shadow of a face confirmed after it was
            # planned reaches it (the robot is then beside / past the face on the route)
            blk = memory.blocked(B[idx], r_chk, layer)
            if not blk.any():
                return None, "not blocked"
        j0 = int(np.argmax(blk))
        k_in = int(idx[j0])
        free_travel = float(S[k_in] - s0)
        if free_travel > p.lookahead:
            return None, "beyond the look-ahead"
        # blocked section (runs separated by less than rejoin_gap are merged)
        k_out = k_in
        bi = np.nonzero(blk)[0]
        for j in bi[bi > j0]:
            kk = int(idx[j])
            if S[kk] - S[k_out] <= p.rejoin_gap + 1e-9:
                k_out = kk
            else:
                break
        # rejoin point: rejoin_gap beyond the section, clear of the planning layer (incl. the occlusion
        # shadow), and followed by free route (r_chk) for rejoin_gap and at least to the end of the
        # look-ahead: a rejoin right in front of the next obstacle would leave the detour 'blocked',
        # and it would be re-planned around the first obstacle again and again (merged instead)
        k_rej = None
        cand = np.nonzero(S >= S[k_out] + p.rejoin_gap - 1e-9)[0]
        cand = cand[cand > k_out]
        if len(cand):
            lim = cand[S[cand] <= S[cand[0]] + 12.0]
            bad = memory.blocked(B[lim], r_plan + REJOIN_SLACK, layer)
            k_hi = int(min(np.searchsorted(S, max(S[lim[-1]] + p.rejoin_gap, s0 + p.lookahead) + 0.05,
                                           side="right"), len(B)))
            rng = np.arange(int(lim[0]), k_hi)
            b_rng = memory.blocked(B[rng], r_chk, "output")
            nxt = np.minimum.accumulate(np.where(b_rng, S[rng], np.inf)[::-1])[::-1]   # next blocked arclength
            nxt_lim = nxt[lim - int(lim[0])]
            need = np.maximum(S[lim] + p.rejoin_gap, s0 + p.lookahead)
            ok = np.nonzero(~bad & (nxt_lim > need))[0]
            if len(ok):
                k_rej = int(lim[ok[0]])
        if k_rej is None:
            if k_out < len(B) - 1 and not memory.blocked(B[-1:], r_plan, layer)[0]:
                k_rej = len(B) - 1
            else:
                return None, "goal blocked"
        goal = (float(B[k_rej, 0]), float(B[k_rej, 1]))
        # obstacle anchor and the base direction across the blocked section
        sec = B[k_in:k_out + 1]
        cells = memory.layer_cells("output")
        if len(cells):
            lo, hi = sec.min(0) - 1.0, sec.max(0) + 1.0
            near = cells[(cells[:, 0] >= lo[0]) & (cells[:, 0] <= hi[0]) & (cells[:, 1] >= lo[1]) &
                         (cells[:, 1] <= hi[1])]
        else:
            near = np.zeros((0, 2))
        if len(near):
            dd = np.hypot(near[:, None, 0] - sec[None, :, 0], near[:, None, 1] - sec[None, :, 1]).min(1)
            anchor = near[int(np.argmin(dd))]
        else:
            anchor = B[k_in]
        tv = B[k_out] - B[k_in]
        if float(np.hypot(*tv)) < 0.05:
            tv = B[min(k_in + 5, len(B) - 1)] - B[max(k_in - 5, 0)]
        nt = float(np.hypot(*tv))
        if nt < 1e-9:
            tv = np.array([goal[0] - rx, goal[1] - ry])
            nt = float(np.hypot(*tv))
            if nt < 1e-9:
                return None, "degenerate route"
        tv = tv / nt
        nrm = np.array([-tv[1], tv[0]])                  # left normal
        # planning window
        pcells = memory.layer_cells(layer)
        pts = [np.array([[rx, ry]]), B[k_in:k_rej + 1]]
        if len(pcells):
            lo, hi = sec.min(0) - 1.0, sec.max(0) + 1.0
            pts.append(pcells[(pcells[:, 0] >= lo[0]) & (pcells[:, 0] <= hi[0]) & (pcells[:, 1] >= lo[1]) &
                              (pcells[:, 1] <= hi[1])])
        Q = np.vstack(pts)
        x0, y0 = Q.min(0) - p.detour_window
        x1, y1 = Q.max(0) + p.detour_window
        if x1 - x0 > MAX_WINDOW or y1 - y0 > MAX_WINDOW:
            ext = np.vstack([[rx, ry], goal, anchor])
            need_lo, need_hi = ext.min(0) - 1.0, ext.max(0) + 1.0
            if np.any(need_hi - need_lo > MAX_WINDOW):
                return None, "window too large"
            c = 0.5 * (need_lo + need_hi)
            x0, x1 = max(x0, c[0] - 0.5 * MAX_WINDOW), min(x1, c[0] + 0.5 * MAX_WINDOW)
            y0, y1 = max(y0, c[1] - 0.5 * MAX_WINDOW), min(y1, c[1] + 0.5 * MAX_WINDOW)
        cost_full = static_cost() if callable(static_cost) else static_cost
        win, cost = memory.window_grid(static_grid, x0, y0, x1, y1, cost_full, layer, p.margin)
        if win is None or win.width < 3 or win.height < 3:
            return None, "no planning window"
        res_w = win.resolution
        diag = math.hypot(win.width, win.height) * res_w
        feasible: Dict[str, Tuple[list, float]] = {}
        for side, sgn in ((LEFT, 1.0), (RIGHT, -1.0)):
            occ = win.occ.copy()
            d = -sgn * nrm
            us = np.arange(0.0, diag + res_w, 0.5 * res_w)
            ray = anchor[None, :] + us[:, None] * d[None, :]
            keep = (np.hypot(ray[:, 0] - rx, ray[:, 1] - ry) > r_plan + 2 * res_w) & \
                   (np.hypot(ray[:, 0] - goal[0], ray[:, 1] - goal[1]) > r_plan + 2 * res_w)
            ix, iy = win.world_to_grid_array(ray[keep])
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    jx, jy = ix + ddx, iy + ddy
                    ok = (jx >= 0) & (jx < win.width) & (jy >= 0) & (jy < win.height)
                    occ[jy[ok], jx[ok]] = 100
            g2 = OccupancyGridMap(occ, res_w, win.origin_x, win.origin_y, win.frame_id)
            raw = astar(g2, (rx, ry), goal, robot_radius=r, extra_cost=cost, snap_radius=0.6,
                        max_expansions=g2.width * g2.height)
            if raw is None or len(raw) < 2:
                continue
            sm = smooth_path(g2, raw, r + SMOOTH_EXTRA, max_segment=0.5, extra_cost=cost)
            A = np.asarray(sm, dtype=float)
            lat = (A - anchor[None, :]) @ nrm * sgn
            if float(lat.max()) <= 0.0:
                continue                                      # went round the other way
            head = [(rx, ry)] + [tuple(q) for q in sm[1:-1]] + [goal]
            L_side = path_length(head)
            feasible[side] = (sm, L_side, L_side + _terrain_integral(win, cost, head))
        if not feasible:
            return None, "both sides blocked"
        # sides compared with the cost A* minimises inside each side: length + terrain cost
        # (pipes / rubble are crossed only when no clean side exists), same for the commitment
        side = min(feasible, key=lambda s: feasible[s][2])
        if commitment is not None:
            pref = commitment.preferred(anchor, t)
            if pref in feasible:
                other = RIGHT if pref == LEFT else LEFT
                if other not in feasible or feasible[pref][2] <= p.switch_ratio * feasible[other][2]:
                    side = pref
        sm, L, C = feasible[side]
        replaced = float(S[k_rej] - s0) + math.hypot(B[k_robot, 0] - rx, B[k_robot, 1] - ry)
        c_replaced = replaced + _terrain_integral(win, cost, [(rx, ry)] + [tuple(q) for q in B[k_robot:k_rej + 1]])
        if C > p.detour_max_ratio * c_replaced + 2.0:
            return None, (f"detour too long ({L:.1f} m for {replaced:.1f} m)" if C - L < 1e-6 else
                          f"detour too costly ({L:.1f} m, terrain {C - L:.1f}, for {replaced:.1f} m)")
        if commitment is not None:
            commitment.commit(side, anchor, t)
        head = densify([(rx, ry)] + [(float(q[0]), float(q[1])) for q in sm[1:-1]] + [goal], 0.10)
        path = head + [(float(q[0]), float(q[1])) for q in B[k_rej + 1:]]      # base tail kept as is
        return Detour(path, side, k_rej, float(L), replaced,
                      (float(anchor[0]), float(anchor[1])), free_travel, goal, layer, float(C)), "ok"


# =============================================================================
# facade
# =============================================================================
@dataclass
class AvoidAction:
    kind: str = "none"                    # none | detour | replan | hold
    path: Optional[List[Tuple[float, float]]] = None   # new followed path (detour)
    speed_scale: float = 1.0
    message: str = ""
    side: Optional[str] = None
    free_travel: float = math.inf
    lead: Optional[float] = None          # pure-pursuit look-ahead for this tick (step(lead=...) only)


class ObstacleAvoider:
    """Memory + detour planner + governor behind one call per control tick
    (shared by the mission patrol / self-RTH and return_to_home).

    The caller feeds frames with :meth:`update`, announces every new goal
    (:meth:`new_goal`) and every (re)planned base route (:meth:`set_route`),
    and calls :meth:`step` with the path it is following.  Exceptions are the
    caller's to catch (count them in ``errors``)."""

    def __init__(self, params=None, robot_radius: float = 0.35):
        self.p = params if isinstance(params, ObstacleParams) else ObstacleParams.from_params(params)
        self.robot_radius = float(robot_radius)
        self.memory = ObstacleMemory(self.p, self.robot_radius)
        self.planner = DetourPlanner(self.p, self.robot_radius)
        self.commitment = SideCommitment(self.p.commit_time)
        self.errors = 0
        self.detours = 0
        self.replans = 0
        self.last = ""
        self.holding = False
        self.free_travel = math.inf
        self.scale = 1.0
        self.detour_side: Optional[str] = None
        self._rejoin_xy: Optional[Tuple[float, float]] = None   # rejoin point of the followed detour
        self._head_layer = "planning"                           # ... and the layer it was planned on
        self._route: Optional[np.ndarray] = None
        self._route_version = 0
        self._k = 0
        self._last_attempt = -math.inf
        self._failed_key = None

    @classmethod
    def from_params(cls, p, robot_radius: float) -> "ObstacleAvoider":
        return cls(p, robot_radius)

    @property
    def enabled(self) -> bool:
        return bool(self.p.enabled)

    # ------------------------------------------------------------- inputs
    def set_static_map(self, grid) -> None:
        self.memory.set_static_map(grid)

    def update(self, frame: EvidenceFrame, pose, t: float) -> bool:
        if not self.enabled:
            return False
        return self.memory.update(frame, pose, t)

    def frame_due(self, t: float, driving: bool = True) -> bool:
        """See :meth:`ObstacleMemory.frame_due` (thinned fusion while not driving)."""
        return self.memory.frame_due(t, driving)

    def set_upright(self, upright: bool, t: float) -> int:
        """Fall guard, see :meth:`ObstacleMemory.set_upright`.  Standing up
        again also drops the detour / blockage state of the route (the robot
        is somewhere else now, the caller re-plans).  Returns the number of
        forgotten cells."""
        was_down = self.memory.suspended
        k = self.memory.set_upright(upright, t)
        if upright and was_down:
            self.detour_side = None
            self._rejoin_xy = None
            self._head_layer = "planning"
            self.holding = False
            self._last_attempt = -math.inf
            self._failed_key = None
            if k:
                self.last = f"fall: {k} obstacle cells seen while falling forgotten"
        return k

    def note_rejected(self) -> None:
        self.memory.note_rejected()

    def new_goal(self) -> None:
        """New goal: per-goal counters, detour state and route reset (memory
        and side commitments kept)."""
        self.detours = 0
        self.replans = 0
        self.detour_side = None
        self._rejoin_xy = None
        self.holding = False
        self._route = None
        self._k = 0
        self._last_attempt = -math.inf
        self._failed_key = None

    def set_route(self, path) -> None:
        """The base route after every (re)plan."""
        if path is None or len(path) == 0:
            self._route = None
        else:
            P = np.asarray([tuple(q[:2]) for q in path], dtype=float)
            self._route = np.asarray(densify([tuple(q) for q in P], 0.10), dtype=float) if len(P) > 1 else P
        self._route_version += 1
        self._k = 0
        self.detour_side = None
        self._rejoin_xy = None

    @property
    def replans_left(self) -> int:
        return max(0, int(self.p.max_replans) - self.replans)

    def note_replan(self) -> None:
        self.replans += 1

    # ------------------------------------------------------------ queries
    def route_blocked(self, path, from_xy=None) -> bool:
        """Does a confirmed obstacle block ``path`` (from the projection of
        ``from_xy``, whole length) at robot_radius?"""
        if not self.enabled or path is None or len(path) < 1 or not self.memory.has_confirmed():
            return False
        P = np.asarray([tuple(q[:2]) for q in path], dtype=float)
        k0 = 0
        if from_xy is not None and len(P) > 1:
            k0 = int(np.argmin(np.hypot(P[:, 0] - from_xy[0], P[:, 1] - from_xy[1])))
        return path_blockage(self.memory, P, k0, check_radius(self.robot_radius, self.p), math.inf) is not None

    def planning_grid(self, static_grid, extra: float = 0.0, layer: str = "planning"):
        """Static + dynamic grid for a global re-plan with robot_radius (detected
        cells grown by margin + ``extra``: a planner using a smaller radius
        r' passes ``extra = robot_radius - r'`` to keep the same clearance from
        detected obstacles), or None (disabled, re-plan disabled, nothing
        confirmed or no static map).  ``layer`` 'near' = the fallback with the
        short occlusion shadow (None when it equals the full one)."""
        if not self.enabled or not self.p.replan_enabled or static_grid is None or not self.memory.has_confirmed():
            return None
        if layer == "near" and len(self._layers()) < 2:
            return None
        return self.memory.combined_grid(static_grid, layer, self.p.margin + max(float(extra), 0.0))

    def planning_grids(self, static_grid, robot_radius: float, reduced: float = 0.7):
        """Global re-plan attempts on static + detected obstacles, best first:
        [(grid, radius, layer)] for the full then the near occlusion shadow,
        each with robot_radius and with ``reduced`` x robot_radius from mapped
        walls (the same robot_radius + margin from detected obstacles); []
        when no dynamic grid applies (legacy behaviour)."""
        r = float(robot_radius)
        out = []
        for layer in self._layers():
            for radius in (r, reduced * r):
                g = self.planning_grid(static_grid, r - radius, layer)
                if g is not None:
                    out.append((g, radius, layer))
        return out

    @staticmethod
    def route_cost(grid, cost, path) -> float:
        """A*'s own metric of a route: length + terrain-cost integral [m]."""
        if path is None or len(path) < 2:
            return math.inf
        P = [(float(q[0]), float(q[1])) for q in path]
        return float(path_length(P) + (_terrain_integral(grid, cost, P) if grid is not None else 0.0))

    def near_is_cheaper(self, cost_full: float, cost_near: float) -> bool:
        """Take the route planned with the NEAR occlusion shadow over the one
        planned with the full shadow?  Only when it is ``switch_ratio`` x
        cheaper (length + terrain cost): the assumed unseen back of an
        obstacle seen from afar must not push a route across debris or a
        long way round when its near side leaves a normal way (Gazebo t: the
        1.0 m shadow of a box seen from the dock closed the C-OBS1 north gap
        and the route went across the pipe); the followed route is checked
        against every face confirmed on the way (detour / re-plan)."""
        return math.isfinite(cost_near) and cost_near * float(self.p.switch_ratio) < float(cost_full)

    def blocked_behind(self, pose, dist: float) -> bool:
        """Confirmed cells within robot_radius of the segment ``dist`` m behind the robot."""
        if not self.enabled or not self.memory.has_confirmed() or pose is None:
            return False
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        n = max(2, int(math.ceil(dist / 0.1)) + 1)
        s = np.linspace(0.1, max(dist, 0.1), n)
        P = np.stack([x - s * math.cos(yaw), y - s * math.sin(yaw)], -1)
        return bool(self.memory.blocked(P, self.robot_radius, "output").any())

    # --------------------------------------------------------------- step
    def _head_length(self, path) -> float:
        """Arclength along ``path`` to the rejoin point of the active detour
        (0 when there is none or the robot has passed it)."""
        if self._rejoin_xy is None:
            return 0.0
        P = _xy(path)
        if len(P) == 0:
            self._rejoin_xy = None
            return 0.0
        d = np.hypot(P[:, 0] - self._rejoin_xy[0], P[:, 1] - self._rejoin_xy[1])
        k = int(np.argmin(d))
        if d[k] > 0.05:                         # the rejoin vertex left the remaining path: back on the route
            self._rejoin_xy = None
            return 0.0
        return float(_cumlen(P[:k + 1])[-1])

    def _layers(self) -> Tuple[str, ...]:
        """Planning layers tried for a detour / global re-plan: the full occlusion
        shadow, then the near shadow (tight passages beside an obstacle whose
        assumed back would close them)."""
        return ("planning", "near") if self.p.shadow_depth > SHADOW_NEAR + 1e-9 else ("planning",)

    def _blockage(self, path, rxy, r_chk: float, s_head: float, layer: str = "planning") -> Optional[Blockage]:
        """First blockage of a followed path: confirmed cells over the look-ahead,
        and up to the rejoin point of a detour (``s_head`` m) also the planning
        ``layer`` the detour was planned on (occlusion shadow included) - a
        face confirmed after the detour was planned refines it before the
        robot cuts the unseen back corner."""
        p = self.p
        blk = path_blockage(self.memory, path, 0, r_chk, p.lookahead, robot_xy=rxy)
        if s_head > 0.0:
            bp = path_blockage(self.memory, path, 0, r_chk, min(p.lookahead, s_head), layer, robot_xy=rxy)
            if bp is not None and (blk is None or bp.s < blk.s):
                blk = bp
        return blk

    def _project(self, xy) -> int:
        R = self._route
        if R is None or len(R) == 0:
            return 0
        lo = max(0, self._k - 10)
        hi = min(len(R), self._k + 61)
        d = np.hypot(R[lo:hi, 0] - xy[0], R[lo:hi, 1] - xy[1])
        self._k = lo + int(np.argmin(d))
        return self._k

    @staticmethod
    def _cuts_debris(follow, rxy, lead: float, grid, static_cost) -> bool:
        """Would the pure-pursuit chord robot -> look-ahead vertex cross clearly
        more terrain cost (pipes, rubble: the static cost layer) than the path
        it cuts?"""
        if grid is None or static_cost is None:
            return False
        cost = static_cost() if callable(static_cost) else static_cost
        if cost is None:
            return False
        P = _xy(follow)
        if len(P) < 2:
            return False
        d = np.hypot(P[:, 0] - rxy[0], P[:, 1] - rxy[1])
        k = int(np.argmin(d[:min(len(d), 60)]))
        far = np.nonzero(d[k:] >= float(lead))[0]
        if not len(far):
            return False
        kt = k + int(far[0])
        chord = _terrain_integral(grid, cost, [tuple(rxy), (float(P[kt, 0]), float(P[kt, 1]))])
        if chord <= DEBRIS_CUT:
            return False
        along = _terrain_integral(grid, cost, [tuple(rxy)] + [(float(q[0]), float(q[1])) for q in P[k:kt + 1]])
        return chord > along + DEBRIS_CUT

    def _plan_detour(self, base, kb: int, rxy, r_chk: float, static_grid, static_cost, t: float,
                     layer: str) -> Tuple[Optional[Detour], str]:
        """One detour attempt on ``layer`` -> (Detour | None, reason)."""
        det, reason = self.planner.plan(self.memory, base, kb, rxy, static_grid, static_cost, self.commitment, t,
                                        layer)
        if det is not None and self._blockage(det.path, rxy, r_chk, det.length, layer) is not None:
            # the detour does not clear the look-ahead: applying it would only repeat it every
            # min_interval (each one counted) - a global re-plan instead
            return None, "detour still blocked"
        return det, reason

    def step(self, pose, followed_path, t: float, static_grid=None, static_cost=None,
             lead: Optional[float] = None) -> AvoidAction:
        """One control tick -> AvoidAction (see the module docstring).

        ``lead``: look-ahead distance of the caller's pure-pursuit follower;
        when given, the governor measures the free travel along the line the
        follower drives (:func:`pursuit_path`), else along the path itself,
        and ``AvoidAction.lead`` is the look-ahead the follower uses for this
        tick (halved where the full one would cut a corner towards an
        obstacle the path keeps clear of, or across mapped debris -
        ``static_cost`` - that the path goes round)."""
        p = self.p
        if not self.enabled or pose is None or followed_path is None or len(followed_path) == 0:
            self.scale, self.free_travel, self.holding = 1.0, math.inf, False
            return AvoidAction()
        t = float(t)
        rxy = (float(pose[0]), float(pose[1]))
        k_robot = self._project(rxy)
        act = AvoidAction()
        follow = followed_path
        r_chk = check_radius(self.robot_radius, p)
        blk = self._blockage(followed_path, rxy, r_chk, self._head_length(followed_path), self._head_layer)
        if blk is not None:
            act.kind = "hold"
            act.free_travel = blk.s
            pending = False                     # a detour attempt is due shortly (min_interval)
            if p.detour_enabled and self.detours < p.max_detours:
                key = (self.memory.revision, self._route_version)
                if key != self._failed_key:     # never re-try a failed detour before the memory / route changed
                    if t - self._last_attempt < p.min_interval:
                        pending = True
                    else:
                        self._last_attempt = t
                        base = self._route if self._route is not None else _xy(followed_path)
                        kb = k_robot if self._route is not None else 0
                        det, reason = None, "not planned"
                        for layer in self._layers():
                            kept = list(self.commitment.entries)
                            cand, why = self._plan_detour(base, kb, rxy, r_chk, static_grid, static_cost, t, layer)
                            if det is None:
                                det, reason = cand, why
                                if det is not None and layer == "planning" and len(self._layers()) > 1:
                                    continue        # also the near shadow: taken when much cheaper (below)
                            elif cand is not None and self.near_is_cheaper(det.cost, cand.cost):
                                det = cand          # the assumed unseen back forced debris / a long way round
                            else:
                                self.commitment.entries = kept      # the side of the detour kept stays committed
                            if det is not None:
                                break
                        if det is not None:
                            self.detours += 1
                            self.detour_side = det.side
                            self._rejoin_xy, self._head_layer = det.rejoin, det.layer
                            act.kind, act.path, act.side = "detour", det.path, det.side
                            act.message = (f"obstacle at {blk.s:.1f} m - detour {det.side} "
                                           f"(+{max(det.length - det.replaced, 0.0):.1f} m)")
                            follow = det.path
                            self.last = act.message
                        else:
                            self._failed_key = key
                            self.last = f"no detour: {reason}"
            if act.kind == "hold" and not pending and p.replan_enabled and self.replans_left > 0:
                act.kind = "replan"
                act.message = f"path blocked by a detected obstacle at {blk.s:.1f} m - re-planning"
                self.last = act.message
        self.holding = act.kind == "hold"
        if lead is None:
            sc, reason, d_c = speed_scale(self.memory, follow, rxy, self.robot_radius, p, t)
        else:
            # the governor measures along the line the follower drives; when that chord cuts a corner
            # towards an obstacle the path itself keeps clear of, the follower tracks the path more
            # tightly (half the look-ahead) instead of being slowed / stopped by its own corner cutting
            act.lead = float(lead)
            sc, reason, d_c = speed_scale(self.memory, pursuit_path(follow, rxy, act.lead), rxy,
                                          self.robot_radius, p, t)
            if sc < 1.0 and act.lead > 2.0 * MIN_LEAD:
                tight = speed_scale(self.memory, pursuit_path(follow, rxy, 0.5 * act.lead), rxy,
                                    self.robot_radius, p, t)
                if tight[0] > sc:
                    act.lead = 0.5 * act.lead
                    sc, reason, d_c = tight
            if act.lead > 2.0 * MIN_LEAD + 1e-9 and self._cuts_debris(follow, rxy, act.lead, static_grid, static_cost):
                # a sharp turn past mapped debris (the near-side route squeezing round the C-OBS1 pipe end):
                # the full chord would cut across it, the planned path does not (Gazebo s, mpc: fell on it)
                act.lead = 0.5 * act.lead
                sc, reason, d_c = speed_scale(self.memory, pursuit_path(follow, rxy, act.lead), rxy,
                                              self.robot_radius, p, t)
        act.speed_scale = sc
        if act.kind in ("none", "hold") and reason and act.message == "":
            act.message = reason
        self.scale, self.free_travel = sc, d_c
        return act

    def status(self) -> dict:
        m = self.memory
        ft = self.free_travel
        return {"enabled": self.enabled, "source": self.p.source, "frames": int(m.frames),
                "rejected": int(m.rejected), "confirmed": m.confirmed_count,
                "free_travel": round(float(ft), 2) if math.isfinite(ft) else None,
                "speed_scale": round(float(self.scale), 2), "detour": self.detour_side,
                "detours": int(self.detours), "replans": int(self.replans), "holding": bool(self.holding),
                "last": self.last, "errors": int(self.errors), "suspended": bool(m.suspended),
                "fall_forgotten": int(m.fall_forgotten)}
