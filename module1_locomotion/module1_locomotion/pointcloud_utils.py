"""
Point-cloud / depth-image utilities for the terrain mapper (ROS free).

* ``sensor_msgs/PointCloud2`` <-> numpy WITHOUT ``sensor_msgs_py``: the field
  table (name / offset / datatype / count), ``point_step``, ``row_step``,
  ``is_bigendian`` and organised (height > 1) or unorganised clouds are all
  honoured.  Messages are duck-typed, so the parser is unit-testable with
  plain Python objects.
* Depth image (32FC1 metres or 16UC1 millimetres) -> 3-D points in the
  camera LINK frame using the pinhole model of
  ``dusan_common.camera.PinholeCamera`` (or a received CameraInfo).
* Auto-detection of the cloud convention: Gazebo depth clouds are normally
  expressed in the camera link frame (x forward, y left, z up) but some
  bridges / drivers publish the optical frame (z forward, x right, y down).
* Camera -> world transform with robot-body, range and height filtering.
* ``PoseBuffer``: time-interpolated robot poses, so that a 10 Hz cloud is
  registered with the 50 Hz odometry pose at the cloud's own time stamp.

Frames: robot pose = (x, y, z, roll, pitch, yaw) of ``base_link`` in the world
(= ``odom``) frame, as produced by ``dusan_common.geometry.odom_to_pose``.
"""
from __future__ import annotations

import array
import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from dusan_common.camera import PinholeCamera, sensor_extrinsic
from dusan_common.geometry import pose_matrix, wrap_angle

# sensor_msgs/PointField datatype constants (identical numeric values)
INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64 = range(1, 9)
_PF_CODES = {INT8: "i1", UINT8: "u1", INT16: "i2", UINT16: "u2",
             INT32: "i4", UINT32: "u4", FLOAT32: "f4", FLOAT64: "f8"}

CONVENTION_LINK = "link"        # x forward, y left, z up (Gazebo camera link)
CONVENTION_OPTICAL = "optical"  # z forward, x right, y down (ROS optical frame)


# =============================================================================
#  PointCloud2 parsing / building
# =============================================================================
def _as_uint8(data) -> np.ndarray:
    """Message ``data`` (bytes / array.array('B') / memoryview / list / ndarray)
    -> flat uint8 array without copying where possible."""
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data).view(np.uint8).reshape(-1)
    try:
        return np.frombuffer(data, dtype=np.uint8)
    except TypeError:  # plain Python list of ints
        return np.asarray(data, dtype=np.uint8)


def _field_tuple(f) -> Tuple[str, int, int, int]:
    """PointField-like object / dict / tuple -> (name, offset, datatype, count)."""
    if isinstance(f, dict):
        return str(f["name"]), int(f["offset"]), int(f["datatype"]), int(f.get("count", 1) or 1)
    if isinstance(f, (tuple, list)):
        name, offset, datatype = f[0], f[1], f[2]
        count = f[3] if len(f) > 3 else 1
        return str(name), int(offset), int(datatype), int(count or 1)
    return str(f.name), int(f.offset), int(f.datatype), int(getattr(f, "count", 1) or 1)


def unpack_fields(data, fields: Iterable, width: int, height: int, point_step: int,
                  row_step: int = 0, is_bigendian: bool = False,
                  names: Sequence[str] = ("x", "y", "z")) -> np.ndarray:
    """Decode the requested scalar fields of a packed point buffer.

    Returns an array of shape (height, width, len(names)) in float64 (organised
    layout; an unorganised cloud simply has height == 1).  Every PointField
    datatype is supported; padding inside a point (point_step larger than the
    fields) and at the end of a row (row_step > width * point_step) is skipped.
    Raises ``KeyError`` when a requested field is missing and ``ValueError``
    when the buffer is too short.
    """
    width, height, point_step = int(width), int(height), int(point_step)
    row_step = int(row_step) if row_step else width * point_step
    if width <= 0 or height <= 0 or point_step <= 0:
        return np.zeros((max(height, 0), max(width, 0), len(names)))
    table = {}
    for f in fields:
        name, off, dt, _cnt = _field_tuple(f)
        table[name] = (off, dt)
    endian = ">" if is_bigendian else "<"
    formats, offsets = [], []
    for n in names:
        if n not in table:
            raise KeyError(f"point cloud has no field '{n}' (fields: {sorted(table)})")
        off, dt = table[n]
        if dt not in _PF_CODES:
            raise ValueError(f"unsupported PointField datatype {dt} for field '{n}'")
        formats.append(endian + _PF_CODES[dt])
        offsets.append(off)
    buf = _as_uint8(data)
    need = (height - 1) * row_step + width * point_step
    if buf.size < need:
        raise ValueError(f"point cloud buffer too short: {buf.size} < {need} bytes")
    if row_step == width * point_step:
        body = buf[: height * row_step]
    else:  # strip the padding at the end of every row
        idx = (np.arange(height, dtype=np.int64)[:, None] * row_step
               + np.arange(width * point_step, dtype=np.int64)[None, :])
        body = buf[idx.reshape(-1)]
    body = np.ascontiguousarray(body)
    dt = np.dtype({"names": list(names), "formats": formats, "offsets": offsets, "itemsize": point_step})
    rec = np.frombuffer(body, dtype=dt, count=width * height)
    out = np.empty((width * height, len(names)), dtype=np.float64)
    for k, n in enumerate(names):
        out[:, k] = rec[n]
    return out.reshape(height, width, len(names))


def pointcloud2_to_xyz(msg, organised: bool = False, remove_invalid: bool = True) -> np.ndarray:
    """``sensor_msgs/PointCloud2`` (or any object with the same attributes)
    -> xyz points.

    organised=False: (N, 3) array, invalid (NaN/inf) points removed when
    ``remove_invalid``.  organised=True: (height, width, 3) array as stored.
    """
    xyz = unpack_fields(msg.data, msg.fields, msg.width, msg.height, msg.point_step,
                        msg.row_step, bool(msg.is_bigendian))
    if organised:
        return xyz
    pts = xyz.reshape(-1, 3)
    if remove_invalid:
        pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def pack_xyz_organised(xyz: np.ndarray):
    """(rows, cols, 3) array -> (bytes, fields, point_step, row_step).

    The layout matches the /elevation_map/points contract: float32 x, y, z at
    offsets 0, 4, 8, point_step 12, little endian, row-major."""
    a = np.asarray(xyz, dtype="<f4")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("pack_xyz_organised expects a (rows, cols, 3) array")
    a = np.ascontiguousarray(a)
    fields = [("x", 0, FLOAT32, 1), ("y", 4, FLOAT32, 1), ("z", 8, FLOAT32, 1)]
    return a.tobytes(), fields, 12, 12 * a.shape[1]


def _assign_uint8_data(msg, buf: bytes) -> None:
    """Fast assignment of a uint8[] message field (array.array path of rosidl)."""
    try:
        msg.data = array.array("B", buf)
    except (AssertionError, TypeError, ValueError):  # pragma: no cover - other rosidl versions
        msg.data = buf


def xyz_to_pointcloud2(xyz: np.ndarray, frame_id: str = "odom", stamp=None):
    """Organised (rows, cols, 3) xyz array -> ``sensor_msgs/PointCloud2``.

    Imports sensor_msgs lazily (only usable inside ROS)."""
    from sensor_msgs.msg import PointCloud2, PointField
    buf, fields, point_step, row_step = pack_xyz_organised(xyz)
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = int(xyz.shape[0])
    msg.width = int(xyz.shape[1])
    msg.fields = [PointField(name=n, offset=o, datatype=d, count=c) for n, o, d, c in fields]
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = row_step
    msg.is_dense = bool(np.isfinite(np.asarray(xyz)).all())
    _assign_uint8_data(msg, buf)
    return msg


# =============================================================================
#  Depth image -> points
# =============================================================================
@dataclass
class DepthIntrinsics:
    """Pinhole intrinsics of the depth camera (pixel u right, v down).

    Points are produced in the camera LINK frame (x = depth along the optical
    axis, y left, z up), consistent with ``PinholeCamera.project_link``."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_pinhole(cls, cam: PinholeCamera) -> "DepthIntrinsics":
        return cls(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)

    @classmethod
    def from_sensor(cls, name: str = "depth_camera") -> "DepthIntrinsics":
        return cls.from_pinhole(PinholeCamera.from_sensor(name))

    @classmethod
    def from_camera_info(cls, msg) -> Optional["DepthIntrinsics"]:
        """sensor_msgs/CameraInfo (field ``k`` or ``K``) -> intrinsics, or None
        when the matrix is empty / degenerate."""
        k = getattr(msg, "k", None)
        if k is None:
            k = getattr(msg, "K", None)
        if k is None:
            return None
        k = [float(v) for v in k]
        if len(k) != 9 or k[0] <= 0.0 or k[4] <= 0.0 or msg.width <= 0 or msg.height <= 0:
            return None
        return cls(int(msg.width), int(msg.height), k[0], k[4], k[2], k[5])

    def scaled(self, width: int, height: int) -> "DepthIntrinsics":
        """Intrinsics for the same lens at another image resolution."""
        if width == self.width and height == self.height:
            return self
        sx, sy = width / float(self.width), height / float(self.height)
        return DepthIntrinsics(int(width), int(height), self.fx * sx, self.fy * sy,
                               self.cx * sx, self.cy * sy)

    def ray_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-pixel (H, W) factors a, b so that a link-frame point of depth d is
        (d, a * d, b * d)."""
        u = np.arange(self.width, dtype=np.float64)
        v = np.arange(self.height, dtype=np.float64)
        a = -(u - self.cx) / self.fx
        b = -(v - self.cy) / self.fy
        return np.broadcast_to(a[None, :], (self.height, self.width)), \
            np.broadcast_to(b[:, None], (self.height, self.width))


def depth_to_metres(arr: np.ndarray, encoding: str) -> np.ndarray:
    """Raw depth image -> float64 metres; invalid pixels become NaN.

    32FC1 / 64FC1: metres (Gazebo uses +/-inf or NaN for no return).
    16UC1 / mono16: millimetres, 0 = no return."""
    enc = (encoding or "").lower()
    if enc in ("16uc1", "mono16"):
        d = arr.astype(np.float64) * 1e-3
        d[arr == 0] = np.nan
        return d
    d = arr.astype(np.float64)
    d[~np.isfinite(d)] = np.nan
    return d


def depth_image_to_points(depth_m: np.ndarray, intr: DepthIntrinsics, stride: int = 1,
                          min_depth: float = 0.05, max_depth: float = math.inf,
                          organised: bool = False) -> np.ndarray:
    """Depth image (H, W) in metres -> link-frame points.

    organised=False: (N, 3) valid points only.  organised=True: (H', W', 3)
    with NaN for invalid pixels (H' = ceil(H / stride)).
    The image depth is the distance along the optical axis (= link x)."""
    d = np.asarray(depth_m, dtype=np.float64)
    if d.shape != (intr.height, intr.width):
        intr = intr.scaled(d.shape[1], d.shape[0])
    a, b = intr.ray_grid()
    s = max(1, int(stride))
    d, a, b = d[::s, ::s], a[::s, ::s], b[::s, ::s]
    valid = np.isfinite(d) & (d >= min_depth) & (d <= max_depth)
    if organised:
        out = np.stack([d, a * d, b * d], -1)
        out[~valid] = np.nan
        return out
    dv = d[valid]
    return np.stack([dv, a[valid] * dv, b[valid] * dv], -1)


# =============================================================================
#  Convention detection
# =============================================================================
def detect_convention(points: np.ndarray, eps: float = 0.02,
                      max_samples: int = 6000) -> Tuple[str, float]:
    """Guess whether camera points are in the LINK (x forward) or OPTICAL
    (z forward) convention from their sign statistics.

    A depth camera only sees points in front of it, so the forward coordinate
    is positive for (almost) every point, whereas the two lateral coordinates
    take both signs.  The forward axis is therefore the one with the larger
    fraction of positive values:  score = P(x > eps) - P(z > eps)
    (+1 -> link, -1 -> optical).  Returns (convention, |score|) or
    ('unknown', 0.0) when there are too few points."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if p.shape[0] < 30:
        return "unknown", 0.0
    if p.shape[0] > max_samples:  # deterministic sub-sample
        p = p[:: int(math.ceil(p.shape[0] / max_samples))]
    fx = float(np.mean(p[:, 0] > eps))
    fz = float(np.mean(p[:, 2] > eps))
    score = fx - fz
    if abs(score) < 0.05:
        # Degenerate (e.g. every point in the upper-right image quadrant).
        # Secondary cue: the forward axis has the largest mean magnitude.
        mx, mz = float(np.mean(np.abs(p[:, 0]))), float(np.mean(np.abs(p[:, 2])))
        return (CONVENTION_LINK if mx >= mz else CONVENTION_OPTICAL), abs(score)
    return (CONVENTION_LINK if score > 0 else CONVENTION_OPTICAL), abs(score)


class ConventionDetector:
    """Stateful wrapper: votes over several clouds and then locks the result.

    mode: 'auto' | 'link' | 'optical' (the latter two bypass detection)."""

    def __init__(self, mode: str = "auto", lock_after: int = 5, min_margin: float = 0.3):
        mode = (mode or "auto").strip().lower()
        if mode not in ("auto", CONVENTION_LINK, CONVENTION_OPTICAL):
            raise ValueError(f"cloud convention must be auto|link|optical, got '{mode}'")
        self.mode = mode
        self.lock_after = max(1, int(lock_after))
        self.min_margin = float(min_margin)
        self.votes = {CONVENTION_LINK: 0, CONVENTION_OPTICAL: 0}
        self.locked: Optional[str] = None
        self.last_margin = 0.0

    @property
    def current(self) -> str:
        if self.mode != "auto":
            return self.mode
        if self.locked:
            return self.locked
        if self.votes[CONVENTION_OPTICAL] > self.votes[CONVENTION_LINK]:
            return CONVENTION_OPTICAL
        return CONVENTION_LINK   # Gazebo default

    def update(self, points: np.ndarray) -> str:
        if self.mode != "auto" or self.locked:
            return self.current
        conv, margin = detect_convention(points)
        self.last_margin = margin
        if conv in self.votes and margin >= self.min_margin:
            self.votes[conv] += 1
            if self.votes[conv] >= self.lock_after:
                self.locked = conv
        return self.current


# =============================================================================
#  Camera -> world with filtering
# =============================================================================
@dataclass
class PointFilter:
    """Which camera points are allowed into the elevation map.

    body_box_*: axis-aligned box in ``base_link`` (trunk + upper legs); points
    inside are the robot itself.  max_height_above_base discards overhangs
    (pipe racks, ceilings) the robot can walk under (2.5-D map limitation)."""
    min_range: float = 0.15
    max_range: float = 3.5
    body_box_min: Tuple[float, float, float] = (-0.50, -0.28, -0.29)
    body_box_max: Tuple[float, float, float] = (0.50, 0.28, 0.25)
    max_height_above_base: float = 0.55
    stride: int = 1


def camera_points_to_world(points_cam: np.ndarray, robot_pose: Sequence[float],
                           convention: str = CONVENTION_LINK, sensor: str = "depth_camera",
                           filt: Optional[PointFilter] = None):
    """Transform camera points into the world frame and filter them.

    points_cam: (N, 3) in the camera link (or optical) frame.
    robot_pose: (x, y, z, roll, pitch, yaw) of base_link in the world.
    Returns (points_world (M, 3), ranges (M,), sensor_origin_world (3,)).
    """
    filt = filt or PointFilter()
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    if filt.stride > 1:
        p = p[:: int(filt.stride)]
    p = p[np.isfinite(p).all(axis=1)]
    if convention == CONVENTION_OPTICAL:
        p = PinholeCamera.optical_to_link(p)
    T_bc = sensor_extrinsic(sensor)
    T_wb = pose_matrix(*[float(v) for v in robot_pose[:6]])
    origin = (T_wb @ T_bc)[:3, 3]
    if p.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0), origin
    rng = np.linalg.norm(p, axis=1)
    keep = (rng >= filt.min_range) & (rng <= filt.max_range)
    pb = p @ T_bc[:3, :3].T + T_bc[:3, 3]                      # base_link frame
    bmin = np.asarray(filt.body_box_min, dtype=float)
    bmax = np.asarray(filt.body_box_max, dtype=float)
    on_body = np.all(pb >= bmin, axis=1) & np.all(pb <= bmax, axis=1)
    keep &= ~on_body
    pw = pb @ T_wb[:3, :3].T + T_wb[:3, 3]                     # world frame
    keep &= (pw[:, 2] - float(robot_pose[2])) <= filt.max_height_above_base
    return pw[keep], rng[keep], origin


def world_to_camera_points(points_world: np.ndarray, robot_pose: Sequence[float],
                           sensor: str = "depth_camera") -> np.ndarray:
    """Inverse of the camera -> world transform (link convention).  Used when a
    driver already publishes the cloud in the world / odom frame, so the same
    range and body filters can be applied."""
    T_wc = pose_matrix(*[float(v) for v in robot_pose[:6]]) @ sensor_extrinsic(sensor)
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    p = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    return (p - t) @ R          # R^T (p - t) for row vectors


# =============================================================================
#  Pose buffer
# =============================================================================
class PoseBuffer:
    """Ring buffer of time-stamped robot poses with linear interpolation.

    Poses are (x, y, z, roll, pitch, yaw); angles are interpolated on the
    shortest arc.  ``at(t)`` returns None when ``t`` lies further than
    ``max_extrapolation`` outside the buffered interval."""

    def __init__(self, max_age: float = 2.0, max_extrapolation: float = 0.3):
        self.max_age = float(max_age)
        self.max_extrapolation = float(max_extrapolation)
        self._t: deque = deque()
        self._p: deque = deque()

    def __len__(self) -> int:
        return len(self._t)

    def clear(self) -> None:
        self._t.clear()
        self._p.clear()

    def add(self, t: float, pose: Sequence[float]) -> None:
        t = float(t)
        if self._t and t < self._t[-1] - 1.0:   # clock jumped back (sim reset)
            self.clear()
        if self._t and t <= self._t[-1]:
            if t == self._t[-1]:
                self._p[-1] = tuple(float(v) for v in pose[:6])
            return
        self._t.append(t)
        self._p.append(tuple(float(v) for v in pose[:6]))
        while len(self._t) > 2 and self._t[-1] - self._t[0] > self.max_age:
            self._t.popleft()
            self._p.popleft()

    def latest(self):
        if not self._t:
            return None
        return self._t[-1], self._p[-1]

    def at(self, t: float):
        if not self._t:
            return None
        t = float(t)
        ts = self._t
        if t <= ts[0]:
            return self._p[0] if ts[0] - t <= self.max_extrapolation else None
        if t >= ts[-1]:
            return self._p[-1] if t - ts[-1] <= self.max_extrapolation else None
        i = bisect.bisect_right(list(ts), t)
        t0, t1 = ts[i - 1], ts[i]
        p0, p1 = self._p[i - 1], self._p[i]
        s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        out = [p0[k] + s * (p1[k] - p0[k]) for k in range(3)]
        out += [float(wrap_angle(p0[k] + s * wrap_angle(p1[k] - p0[k]))) for k in range(3, 6)]
        return tuple(out)
