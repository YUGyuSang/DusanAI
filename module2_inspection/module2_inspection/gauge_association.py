"""
Gauge identity association (ROS-free, numpy + OpenCV).

Given the robot pose (x, y, z, roll, pitch, yaw of ``base_link`` in the world, i.e. /odom)
the inspection-camera pose follows from ``dusan_common.camera`` + ``robot_model.SENSORS``.
Every gauge of the layout registry (``world_layout.yaml``) is projected into the image:

* predicted centre / face ellipse / radius of the dial,
* depth, distance and incidence angle (angle between dial normal and line of sight),
* visibility = in front of the camera, facing it, incidence below a limit, whole face in
  the image, large enough to be read, optional line of sight on the 2-D occupancy map,
* the plane-induced homography dial frame -> image and the dial normal in the optical frame
  (used by the reader to disambiguate the two circle-pose solutions).

``hints`` turns the visible predictions into ``GaugeHint`` objects (ROI + registry range)
for the reader, ``assign`` gives detections their gauge id (gated nearest neighbour with a
pose-uncertainty gate; without a pose the detection closest to the image centre / largest
becomes the ``/inspection/target`` gauge), and ``reconcile_range`` cross-checks a range read
by OCR with the registry (the registry wins on disagreement, both are reported).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dusan_common import gauge_design as gd
from dusan_common.camera import PinholeCamera, camera_world_pose

from module2_inspection.gauge_reader import GaugeHint

# link frame (x fwd, y left, z up) -> optical frame (z fwd, x right, y down)
R_OPTICAL_FROM_LINK = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def dial_axes(normal, up_ref=(0.0, 0.0, 1.0)):
    """(right, up, n) unit vectors of a dial facing ``normal`` whose texture 'up' is world up."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    up_ref = np.asarray(up_ref, float)
    if abs(float(np.dot(up_ref, n))) > 0.95:
        up_ref = np.array([1.0, 0.0, 0.0])
    right = np.cross(up_ref, n)
    right /= np.linalg.norm(right)
    return right, np.cross(n, right), n


@dataclass
class GaugePrediction:
    gauge_id: str
    tag: str
    center: Tuple[float, float]                 # projected dial centre [px]
    radius: float                               # semi-major axis of the projected face [px]
    ellipse: Tuple[float, float, float, float, float]   # cx, cy, a, b, angle_deg
    depth: float                                # along the optical axis [m]
    distance: float                             # camera centre <-> dial centre [m]
    incidence_deg: float
    in_front: bool
    facing: bool
    in_image: bool
    line_of_sight: bool
    visible: bool
    H_dial2img: Optional[np.ndarray]            # dial frame (texture upright) -> image
    normal_cam: np.ndarray                      # dial normal in the optical frame
    vrange: Tuple[float, float]
    unit: str

    @property
    def roi(self) -> Tuple[float, float, float, float]:
        cx, cy = self.center
        r = self.radius
        return (cx - r, cy - r, cx + r, cy + r)


@dataclass
class RangeCheck:
    range_used: Tuple[float, float]
    source: str                         # 'registry' | 'ocr' | 'registry(ocr_disagrees)' | 'registry+ocr'
    agree: Optional[bool]
    layout_range: Optional[Tuple[float, float]]
    ocr_range: Optional[Tuple[float, float]]


class GaugeAssociator:
    """Projects registry gauges into the inspection camera and assigns detections to them."""

    def __init__(self, gauges: Sequence[dict], camera: Optional[PinholeCamera] = None,
                 sensor: str = "inspection_camera", max_incidence_deg: float = 70.0,
                 min_radius_px: float = 12.0, max_distance: float = 6.0, image_margin_px: float = 2.0,
                 gate_px: float = 70.0, gate_radius_factor: float = 1.2, occupancy_map=None):
        self.gauges = [dict(g) for g in gauges]
        self.camera = camera or PinholeCamera.from_sensor(sensor)
        self.sensor = sensor
        self.max_incidence_deg = float(max_incidence_deg)
        self.min_radius_px = float(min_radius_px)
        self.max_distance = float(max_distance)
        self.image_margin_px = float(image_margin_px)
        self.gate_px = float(gate_px)
        self.gate_radius_factor = float(gate_radius_factor)
        self.map = occupancy_map
        self._by_id = {g["id"]: g for g in self.gauges}

    @classmethod
    def from_layout(cls, layout, **kw) -> "GaugeAssociator":
        return cls(layout.gauges, **kw)

    def gauge(self, gauge_id: Optional[str]) -> Optional[dict]:
        return self._by_id.get(gauge_id) if gauge_id else None

    def set_intrinsics(self, width: int, height: int, K: Optional[np.ndarray] = None, hfov: Optional[float] = None):
        """Update the camera model (e.g. from sensor_msgs/CameraInfo)."""
        if K is not None and K[0, 0] > 0:
            cam = PinholeCamera(width, height, 2.0 * math.atan(width / (2.0 * float(K[0, 0]))))
            cam.fx, cam.fy = float(K[0, 0]), float(K[1, 1])
            cam.cx, cam.cy = float(K[0, 2]), float(K[1, 2])
            self.camera = cam
        elif hfov is not None:
            self.camera = PinholeCamera(width, height, hfov)

    # ------------------------------------------------------------ projection
    def camera_pose(self, robot_pose) -> np.ndarray:
        """4x4 world <- camera link (Gazebo convention, x forward)."""
        return camera_world_pose(tuple(float(v) for v in robot_pose), self.sensor)

    def predict(self, robot_pose) -> List[GaugePrediction]:
        """Predictions for every registry gauge (``visible`` tells whether it can be read)."""
        T = self.camera_pose(robot_pose)
        R_wl, C = T[:3, :3], T[:3, 3]
        R_ow = R_OPTICAL_FROM_LINK @ R_wl.T                      # world -> optical
        cam = self.camera
        K = np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]])
        out = []
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        for g in self.gauges:
            c = np.asarray(g["position"], float)
            # texture side: generate_world maps the texture side onto dial_diameter (the dial
            # disc of radius dial_diameter / 2 shows the inscribed circle of the texture)
            D = float(g.get("dial_diameter", 0.30))
            right, up, n = dial_axes(g["normal"])
            c_opt = R_ow @ (c - C)
            to_cam = C - c
            dist = float(np.linalg.norm(to_cam))
            cos_inc = float(np.dot(n, to_cam) / max(dist, 1e-9))
            inc = math.degrees(math.acos(np.clip(cos_inc, -1.0, 1.0)))
            facing = cos_inc > 0.0
            n_cam = R_ow @ n
            in_front = bool(c_opt[2] > 0.05)
            H = None
            ell = (float("nan"),) * 5
            center = (float("nan"), float("nan"))
            radius = 0.0
            in_img = False
            if in_front:
                H = K @ np.stack([R_ow @ (D * right), R_ow @ (D * up), c_opt], axis=1)
                H = H / H[2, 2]
                ring = c[None, :] + gd.R_FACE * D * (np.cos(t)[:, None] * right + np.sin(t)[:, None] * up)
                q = (ring - C) @ R_ow.T
                if np.all(q[:, 2] > 0.02):
                    uv = q[:, :2] / q[:, 2:3] * [cam.fx, cam.fy] + [cam.cx, cam.cy]
                    (ex, ey), (w2, h2), angd = cv2.fitEllipse(uv.astype(np.float32))
                    a, b = max(w2, h2) / 2.0, min(w2, h2) / 2.0
                    angd = angd if w2 >= h2 else angd + 90.0
                    cc = H @ np.array([0.0, 0.0, 1.0])
                    center = (float(cc[0] / cc[2]), float(cc[1] / cc[2]))
                    ell = (float(ex), float(ey), float(a), float(b), float(angd % 180.0))
                    radius = float(a)
                    m = self.image_margin_px
                    in_img = bool(uv[:, 0].min() >= m and uv[:, 1].min() >= m and
                                  uv[:, 0].max() <= cam.width - 1 - m and uv[:, 1].max() <= cam.height - 1 - m)
            los = True
            if self.map is not None and in_front and facing:
                # stop short of the gauge (its stand / the machine is an obstacle on the 2-D map)
                p_end = c + 0.4 * n
                try:
                    los = bool(self.map.line_of_sight(C[0], C[1], p_end[0], p_end[1]))
                except Exception:
                    los = True
            visible = bool(in_front and facing and in_img and inc <= self.max_incidence_deg
                           and radius >= self.min_radius_px and dist <= self.max_distance and los)
            out.append(GaugePrediction(
                gauge_id=str(g["id"]), tag=str(g.get("tag", "")), center=center, radius=radius, ellipse=ell,
                depth=float(c_opt[2]), distance=dist, incidence_deg=inc, in_front=in_front, facing=facing,
                in_image=in_img, line_of_sight=los, visible=visible, H_dial2img=H, normal_cam=n_cam,
                vrange=(float(g["min"]), float(g["max"])), unit=str(g.get("unit", ""))))
        return out

    def hints(self, robot_pose, target_id: Optional[str] = None, do_ocr: bool = True,
              pose_sigma_px: float = 30.0) -> List[GaugeHint]:
        """Reader hints for the visible gauges (target first).  The ROI half size grows with
        the expected pose / timing error ``pose_sigma_px`` (robot shake)."""
        preds = [p for p in self.predict(robot_pose) if p.visible]
        preds.sort(key=lambda p: (p.gauge_id != target_id, p.distance))
        hints = []
        for p in preds:
            # visible => facing the camera => normal_cam already points towards the camera
            hints.append(GaugeHint(gauge_id=p.gauge_id, center=p.center, radius=p.radius, vrange=p.vrange,
                                   unit=p.unit, normal_cam=p.normal_cam,
                                   search_radius=2.0 * p.radius + 3.0 * pose_sigma_px + 20.0, do_ocr=do_ocr))
        return hints

    # ------------------------------------------------------------ assignment
    def assign(self, centers: Sequence[Tuple[float, float]], radii: Sequence[float],
               predictions: Optional[Sequence[GaugePrediction]] = None,
               target_id: Optional[str] = None,
               image_size: Optional[Tuple[int, int]] = None) -> List[Optional[str]]:
        """Gauge id for each detection (centre, radius) - None when unidentified.

        With predictions: greedy assignment by normalised distance inside the gate
        |d| < gate_px + gate_radius_factor * r_pred and radius ratio in [0.6, 1.6].  If the
        ``target_id`` gauge (the one the mission says is in view) is still unassigned after
        that (pose error larger than the gate), the remaining detection of similar size
        closest to the target's prediction gets the target id.
        Without predictions (no pose): the best detection - closest to the image centre,
        larger first - becomes ``target_id`` (if given), the others stay unidentified."""
        n = len(centers)
        ids: List[Optional[str]] = [None] * n
        if n == 0:
            return ids
        if predictions:
            pairs = []
            for i in range(n):
                for p in predictions:
                    if not p.in_front or not np.isfinite(p.center[0]):
                        continue
                    d = math.hypot(centers[i][0] - p.center[0], centers[i][1] - p.center[1])
                    gate = self.gate_px + self.gate_radius_factor * p.radius
                    ratio = radii[i] / max(p.radius, 1e-6)
                    if d <= gate and 0.6 <= ratio <= 1.6:
                        pairs.append((d / gate + abs(math.log(ratio)), i, p.gauge_id))
            pairs.sort()
            used_d, used_g = set(), set()
            for _cost, i, gid in pairs:
                if i in used_d or gid in used_g:
                    continue
                ids[i] = gid
                used_d.add(i)
                used_g.add(gid)
            tp = next((p for p in predictions if p.gauge_id == target_id), None) if target_id else None
            rest = [i for i in range(n) if ids[i] is None]
            if tp is not None and target_id not in used_g and rest and tp.in_front and np.isfinite(tp.center[0]):
                ok = [i for i in rest if 0.5 <= radii[i] / max(tp.radius, 1e-6) <= 2.0]
                if ok:
                    ids[min(ok, key=lambda i: math.hypot(centers[i][0] - tp.center[0],
                                                         centers[i][1] - tp.center[1]))] = target_id
            return ids
        if target_id:
            W, H = image_size if image_size else (self.camera.width, self.camera.height)
            def cost(i):
                return math.hypot(centers[i][0] - W / 2.0, centers[i][1] - H / 2.0) / max(W, H) - 0.002 * radii[i]
            ids[min(range(n), key=cost)] = target_id
        return ids

    # ------------------------------------------------------------ range cross-check
    def reconcile_range(self, gauge_id: Optional[str], ocr_range: Optional[Tuple[float, float]],
                        rel_tol: float = 0.01) -> Optional[RangeCheck]:
        """Registry range vs OCR range: the registry wins on disagreement, both are reported."""
        g = self.gauge(gauge_id)
        lay = (float(g["min"]), float(g["max"])) if g is not None else None
        if lay is None and ocr_range is None:
            return None
        if lay is None:
            return RangeCheck(tuple(ocr_range), "ocr", None, None, tuple(ocr_range))
        if ocr_range is None:
            return RangeCheck(lay, "registry", None, lay, None)
        fs = max(abs(lay[1] - lay[0]), 1e-9)
        agree = abs(ocr_range[0] - lay[0]) <= rel_tol * fs and abs(ocr_range[1] - lay[1]) <= rel_tol * fs
        return RangeCheck(lay, "registry+ocr" if agree else "registry(ocr_disagrees)", bool(agree), lay,
                          (float(ocr_range[0]), float(ocr_range[1])))
