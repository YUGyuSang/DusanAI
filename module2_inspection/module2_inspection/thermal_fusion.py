"""
Thermal / RGB fusion for Module 2 (ROS free: numpy + OpenCV only).

Pipeline (``ThermalFusion.process``)
-----------------------------------
1. **Radiometry** - Gazebo's thermal camera publishes an L16 image whose
   counts are ``Kelvin / resolution`` (resolution 0.01 K).  ``raw_to_kelvin``
   accepts every encoding the bridge may produce (``mono16``, ``16UC1``,
   ``L16``; 8-bit and float images are supported as well).
2. **Hot-spot segmentation** in the *thermal* image (where the radiometric
   values live): threshold ``>= warning_k``, optional morphological opening,
   8-connected components, minimum area, per-blob max / mean temperature,
   temperature-weighted centroid, peak pixel and bounding box.
3. **Localisation** - a ray is cast from the thermal camera through the blob
   centroid (robot pose from /odom + extrinsics from ``robot_model``).  The
   layout hot spot with the smallest point-to-ray distance inside a range
   dependent tolerance gives id / tag / world position; otherwise the blob is
   placed where the ray meets the registration plane.
4. **Registration** thermal -> RGB with the plane-induced homography

       H = K_rgb (R - t n^T / d) K_th^-1

   (R, t) = pose of the thermal optical frame expressed in the RGB optical
   frame (from the two sensor extrinsics), n = unit plane normal in the
   thermal optical frame pointing *towards* the camera, d > 0 = distance of the
   plane (plane: n.X + d = 0).  The plane distance comes from (in order of
   preference) the depth of matched layout hot spots, the median depth of the
   layout thermal objects in view, or a fixed parameter.  Optionally an ECC
   alignment (OpenCV ``findTransformECC``) between *edge* images of the two
   modalities refines the residual parallax, accepted only inside physical
   sanity limits (the baseline is horizontal, so the correction must be
   mostly horizontal and bounded).
5. **Visualisation** - INFERNO colour map blended on the RGB image only where
   T > ambient + delta, isotherm contours for warning / alarm, labelled boxes
   and a temperature scale bar.
6. **Temporal confirmation** - a hot spot is reported only after it persisted
   for ``confirm_frames`` consecutive thermal frames (``TemporalConfirmer``).

Conventions: camera *link* frame = Gazebo (x forward, y left, z up); camera
*optical* frame = (z forward, x right, y down); see ``dusan_common.camera``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dusan_common.camera import PinholeCamera, sensor_extrinsic
from dusan_common.geometry import pose_matrix
from dusan_common.robot_model import SENSORS, THERMAL_RESOLUTION

# p_link = R_LINK_OPT @ p_opt   (columns = optical axes expressed in the link frame)
R_LINK_OPT = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])

STATUS_COLORS = {  # BGR
    "normal": (60, 200, 60),
    "warning": (0, 200, 255),
    "alarm": (40, 40, 255),
}

_ENC_16 = ("mono16", "16uc1", "l16")
_ENC_8 = ("mono8", "8uc1", "l8")
_ENC_F = ("32fc1", "64fc1")


# =============================================================================
# Radiometry
# =============================================================================
def raw_to_kelvin(raw: np.ndarray, encoding: str = "mono16", resolution: float = THERMAL_RESOLUTION,
                  l8_range: Tuple[float, float] = (SENSORS["thermal_camera"]["min_temp"],
                                                    SENSORS["thermal_camera"]["max_temp"])) -> np.ndarray:
    """Convert a raw thermal image to Kelvin (float32).

    * 16-bit (``mono16`` / ``16UC1`` / ``L16``): ``K = count * resolution``.
    * 8-bit (``mono8`` / ``L8``): linear over ``l8_range`` (Gazebo L8 mode).
    * float (``32FC1``): already Kelvin.

    Zero counts (no return / invalid pixel) become NaN.
    """
    enc = (encoding or "mono16").lower()
    a = np.asarray(raw)
    if enc in _ENC_16 or (enc not in _ENC_8 + _ENC_F and a.dtype == np.uint16):
        k = a.astype(np.float32) * np.float32(resolution)
        k[a == 0] = np.nan
        return k
    if enc in _ENC_8 or a.dtype == np.uint8:
        lo, hi = float(l8_range[0]), float(l8_range[1])
        k = (lo + a.astype(np.float32) * np.float32((hi - lo) / 255.0)).astype(np.float32)
        return k
    if enc in _ENC_F or np.issubdtype(a.dtype, np.floating):
        k = a.astype(np.float32).copy()
        k[~np.isfinite(k) | (k <= 0.0)] = np.nan
        return k
    raise ValueError(f"unsupported thermal encoding {encoding!r}")


def kelvin_to_l16(kelvin: np.ndarray, resolution: float = THERMAL_RESOLUTION) -> np.ndarray:
    """Kelvin -> L16 counts (uint16, saturating).  NaN -> 0 (invalid)."""
    k = np.asarray(kelvin, dtype=np.float64)
    counts = np.where(np.isfinite(k), np.round(k / resolution), 0.0)
    return np.clip(counts, 0, 65535).astype(np.uint16)


def k_to_c(k: float) -> float:
    return float(k) - 273.15


# =============================================================================
# Camera rig + plane induced homography
# =============================================================================
@dataclass
class CameraRig:
    """Thermal + RGB cameras with their extrinsics in ``base_link``."""

    thermal: PinholeCamera
    rgb: PinholeCamera
    T_base_th: np.ndarray      # 4x4  base_link <- thermal link
    T_base_rgb: np.ndarray     # 4x4  base_link <- rgb (inspection) link

    @classmethod
    def from_robot_model(cls, thermal_size: Optional[Tuple[int, int]] = None,
                         rgb_size: Optional[Tuple[int, int]] = None) -> "CameraRig":
        th = SENSORS["thermal_camera"]
        rg = SENSORS["inspection_camera"]
        tw, thh = thermal_size or (th["width"], th["height"])
        rw, rh = rgb_size or (rg["width"], rg["height"])
        return cls(PinholeCamera(tw, thh, th["hfov"]), PinholeCamera(rw, rh, rg["hfov"]),
                   sensor_extrinsic("thermal_camera"), sensor_extrinsic("inspection_camera"))

    def resized(self, thermal_size: Tuple[int, int], rgb_size: Tuple[int, int]) -> "CameraRig":
        """Same rig with intrinsics for other image sizes (hfov is kept)."""
        if (thermal_size == (self.thermal.width, self.thermal.height)
                and rgb_size == (self.rgb.width, self.rgb.height)):
            return self
        return replace(self, thermal=PinholeCamera(thermal_size[0], thermal_size[1], self.thermal.hfov),
                       rgb=PinholeCamera(rgb_size[0], rgb_size[1], self.rgb.hfov))

    # ---------------------------------------------------------------- poses
    def thermal_world(self, robot_pose: Sequence[float]) -> np.ndarray:
        """4x4 world <- thermal link for robot_pose = (x, y, z, roll, pitch, yaw)."""
        return pose_matrix(*robot_pose) @ self.T_base_th

    def rgb_world(self, robot_pose: Sequence[float]) -> np.ndarray:
        return pose_matrix(*robot_pose) @ self.T_base_rgb

    def relative_optical(self) -> Tuple[np.ndarray, np.ndarray]:
        """(R, t) with X_rgb_opt = R X_th_opt + t."""
        T_rgb_th = np.linalg.inv(self.T_base_rgb) @ self.T_base_th      # rgb link <- thermal link
        R = R_LINK_OPT.T @ T_rgb_th[:3, :3] @ R_LINK_OPT
        t = R_LINK_OPT.T @ T_rgb_th[:3, 3]
        return R, t

    def epipolar_direction(self) -> Optional[np.ndarray]:
        """Image direction of residual parallax in the RGB image when the
        epipole is (near) infinity, i.e. the baseline is parallel to the image
        plane (true for this rig: thermal 5 cm to the right).  None otherwise."""
        _, t = self.relative_optical()
        if np.linalg.norm(t[:2]) < 1e-9 or abs(t[2]) > 0.2 * np.linalg.norm(t[:2]):
            return None
        return t[:2] / np.linalg.norm(t[:2])

    # ---------------------------------------------------------- homography
    def homography(self, n_opt: np.ndarray, d: float) -> np.ndarray:
        """Plane-induced homography thermal pixel -> RGB pixel.

        Plane in the thermal optical frame: n.X + d = 0 with unit normal n
        pointing towards the camera and d > 0 its distance.  For X on the
        plane, -n.X / d = 1, hence

            X_rgb = R X + t = R X + t (-n.X / d) = (R - t n^T / d) X

        and with x ~ K X on both sides: H = K_rgb (R - t n^T / d) K_th^-1.
        """
        R, t = self.relative_optical()
        n = np.asarray(n_opt, dtype=float).reshape(3)
        n = n / max(np.linalg.norm(n), 1e-12)
        d = max(float(d), 1e-3)
        H = self.rgb.K @ (R - np.outer(t, n) / d) @ np.linalg.inv(self.thermal.K)
        return H / H[2, 2]

    def plane_normal_opt(self, mode: str, robot_pose: Optional[Sequence[float]] = None) -> np.ndarray:
        """Unit plane normal in the thermal optical frame.

        ``fronto``  : plane perpendicular to the optical axis, n = (0, 0, -1).
        ``vertical``: vertical plane facing the camera (walls, pipe racks,
                      machine faces) - accounts for the camera pitch.
        """
        if mode != "vertical":
            return np.array([0.0, 0.0, -1.0])
        pose = robot_pose if robot_pose is not None else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        R_w_link = self.thermal_world(pose)[:3, :3]
        fwd = R_w_link[:, 0]
        h = np.array([fwd[0], fwd[1], 0.0])
        if np.linalg.norm(h) < 1e-6:          # looking straight up/down: fall back
            return np.array([0.0, 0.0, -1.0])
        n_world = -h / np.linalg.norm(h)
        n_opt = R_LINK_OPT.T @ (R_w_link.T @ n_world)
        return n_opt / np.linalg.norm(n_opt)

    def plane_homography(self, depth: float, mode: str = "fronto",
                         robot_pose: Optional[Sequence[float]] = None,
                         point_opt: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Homography for a plane through the optical-axis point at ``depth``
        (or through ``point_opt``).  Returns (H, n_opt, d)."""
        n = self.plane_normal_opt(mode, robot_pose)
        P = np.asarray(point_opt, dtype=float) if point_opt is not None else np.array([0.0, 0.0, float(depth)])
        d = float(-n @ P)
        if d <= 1e-3:   # degenerate (plane through / behind the camera) -> fronto at depth
            n = np.array([0.0, 0.0, -1.0])
            d = max(float(P[2]), 0.1)
        return self.homography(n, d), n, d


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N,2) pixels -> (N,2) pixels."""
    p = np.atleast_2d(np.asarray(pts, dtype=float))
    ph = np.c_[p, np.ones(len(p))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def transform_bbox(H: np.ndarray, bbox: Sequence[float]) -> List[int]:
    """Axis-aligned bbox [u, v, w, h] mapped through H (bbox of the 4 corners)."""
    u, v, w, h = bbox
    c = np.array([[u, v], [u + w, v], [u + w, v + h], [u, v + h]], dtype=float)
    q = apply_homography(H, c)
    u0, v0 = np.floor(q.min(0))
    u1, v1 = np.ceil(q.max(0))
    return [int(u0), int(v0), int(u1 - u0), int(v1 - v0)]


# =============================================================================
# ECC refinement on edge images
# =============================================================================
def _edge_image(img: np.ndarray, blur_sigma: float = 1.2) -> Tuple[np.ndarray, float]:
    """Normalised gradient magnitude (float32 in [0, 1]) and the raw p99 gradient."""
    a = img.astype(np.float32)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    p99 = float(np.percentile(mag, 99.0)) if mag.size else 0.0
    e = np.clip(mag / max(p99, 1e-6), 0.0, 1.0)
    if blur_sigma > 0:
        e = cv2.GaussianBlur(e, (0, 0), blur_sigma)
    return e.astype(np.float32), p99


@dataclass
class EccResult:
    ok: bool
    H: np.ndarray
    correction: np.ndarray          # 3x3 in full-res RGB pixels: H = correction @ H0
    cc: float = float("nan")
    shift_px: Tuple[float, float] = (0.0, 0.0)
    reason: str = ""


def refine_homography_ecc(kelvin: np.ndarray, rgb_bgr: np.ndarray, H0: np.ndarray,
                          scales: Sequence[float] = (0.25, 0.5), motion: str = "translation",
                          max_iter: int = 60, eps: float = 1e-4, max_shift_px: float = 30.0,
                          max_vertical_px: float = 8.0, max_linear_dev: float = 0.06,
                          min_cc: float = 0.20, min_thermal_grad_k: float = 0.4,
                          min_valid_fraction: float = 0.15,
                          epipolar_dir: Optional[Sequence[float]] = None) -> EccResult:
    """Refine H0 (thermal -> RGB) with ECC between edge images.

    The thermal gradient image is warped into the (down-scaled) RGB frame with
    H0 and aligned to the RGB gradient image.  ``findTransformECC`` returns W
    with  inp(W x) ~ tmpl(x)  (x in RGB coordinates), so a thermal pixel p
    lands at x = W^-1 S H0 p and the full-resolution result is
    H = S^-1 W^-1 S H0.  Coarse-to-fine over ``scales``.

    Epipolar constraint: a depth error of the registration plane displaces a
    point along its epipolar line.  For this rig the baseline is (almost)
    parallel to the image rows, so the epipole is at infinity and every
    residual parallax is a shift along ``epipolar_dir`` (unit 2-vector in the
    RGB image).  With ``motion='translation'`` the ECC translation is
    projected onto that direction; the perpendicular component must stay
    below ``max_vertical_px`` (otherwise ECC locked onto a wrong structure).
    """
    ident = np.eye(3)
    if rgb_bgr is None or kelvin is None:
        return EccResult(False, H0, ident, reason="missing image")
    T = np.asarray(kelvin, dtype=np.float32)
    if not np.any(np.isfinite(T)):
        return EccResult(False, H0, ident, reason="no valid thermal pixels")
    T = np.where(np.isfinite(T), T, np.nanmedian(T)).astype(np.float32)
    th_edges, th_p99 = _edge_image(T, blur_sigma=0.0)
    if th_p99 < min_thermal_grad_k:
        return EccResult(False, H0, ident, reason="thermal image has no structure")
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY) if rgb_bgr.ndim == 3 else rgb_bgr
    gray = gray.astype(np.float32) / 255.0
    H_rgb, W_rgb = gray.shape[:2]

    mtype = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
             "affine": cv2.MOTION_AFFINE}.get(motion, cv2.MOTION_TRANSLATION)
    W = np.eye(2, 3, dtype=np.float32)          # in the current scale's pixels
    prev_s = None
    cc = float("nan")
    for s in scales:
        s = float(s)
        if prev_s is not None:
            W[:, 2] *= s / prev_s
        prev_s = s
        sw, sh = max(16, int(round(W_rgb * s))), max(16, int(round(H_rgb * s)))
        S = np.diag([sw / W_rgb, sh / H_rgb, 1.0])
        tmpl, _ = _edge_image(cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA), blur_sigma=1.2)
        Hs = S @ H0
        inp = cv2.warpPerspective(th_edges, Hs, (sw, sh), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inp = cv2.GaussianBlur(inp, (0, 0), 1.2)
        valid = cv2.warpPerspective(np.ones(T.shape, np.uint8), Hs, (sw, sh), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
        if valid.mean() < min_valid_fraction:
            return EccResult(False, H0, ident, reason="thermal/RGB overlap too small")
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iter), float(eps))
        try:
            cc, W = cv2.findTransformECC(tmpl, inp, W, mtype, criteria, valid, 5)
        except cv2.error as exc:   # did not converge / singular
            return EccResult(False, H0, ident, reason=f"ecc failed: {str(exc).splitlines()[-1][:80]}")
    s = float(prev_s)
    sw, sh = max(16, int(round(W_rgb * s))), max(16, int(round(H_rgb * s)))
    S = np.diag([sw / W_rgb, sh / H_rgb, 1.0])
    W3 = np.vstack([W.astype(float), [0.0, 0.0, 1.0]])
    C = np.linalg.inv(S) @ np.linalg.inv(W3) @ S        # full-res correction
    shift = np.array([C[0, 2], C[1, 2]], dtype=float)
    if epipolar_dir is not None:
        e = np.asarray(epipolar_dir, dtype=float)
        e = e / max(np.linalg.norm(e), 1e-12)
        perp = float(abs(shift[0] * -e[1] + shift[1] * e[0]))
    else:
        e, perp = None, float(abs(shift[1]))
    lin_dev = float(np.max(np.abs(C[:2, :2] - np.eye(2))))
    if not np.isfinite(cc) or cc < min_cc:
        return EccResult(False, H0, ident, cc, (float(shift[0]), float(shift[1])), "correlation too low")
    if np.linalg.norm(shift) > max_shift_px or perp > max_vertical_px:
        return EccResult(False, H0, ident, cc, (float(shift[0]), float(shift[1])), "shift outside sanity limits")
    if e is not None and mtype == cv2.MOTION_TRANSLATION:
        shift = float(shift @ e) * e                    # keep the epipolar component only
        C = np.eye(3)
        C[0, 2], C[1, 2] = shift
    shift = (float(shift[0]), float(shift[1]))
    if lin_dev > max_linear_dev:
        return EccResult(False, H0, ident, cc, shift, "scale/rotation outside sanity limits")
    H = C @ H0
    return EccResult(True, H / H[2, 2], C, float(cc), shift, "ok")


# =============================================================================
# Hot-spot segmentation
# =============================================================================
@dataclass
class Blob:
    area_px: int
    max_temp_k: float
    mean_temp_k: float
    centroid_uv: Tuple[float, float]      # temperature-weighted
    peak_uv: Tuple[int, int]
    bbox: Tuple[int, int, int, int]       # u, v, w, h (thermal pixels)


def segment_hotspots(kelvin: np.ndarray, threshold_k: float, min_area_px: int = 6,
                     open_px: int = 0, connectivity: int = 8) -> List[Blob]:
    """Connected components of ``kelvin >= threshold_k`` (NaN never hot).

    Returns blobs sorted by decreasing max temperature.  The centroid is
    weighted with (T - threshold) so that it sits on the hottest part of the
    object (better ray for localisation than the geometric centroid).
    """
    T = np.asarray(kelvin, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        mask = (T >= threshold_k).astype(np.uint8)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if not mask.any():
        return []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    out: List[Blob] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x, y = int(stats[lab, cv2.CC_STAT_LEFT]), int(stats[lab, cv2.CC_STAT_TOP])
        w, h = int(stats[lab, cv2.CC_STAT_WIDTH]), int(stats[lab, cv2.CC_STAT_HEIGHT])
        sub_lab = labels[y:y + h, x:x + w] == lab
        sub_T = T[y:y + h, x:x + w]
        vals = sub_T[sub_lab]
        vs, us = np.nonzero(sub_lab)
        wts = np.maximum(vals - threshold_k, 0.0) + 1e-3
        cu = float(np.sum(wts * (us + x)) / np.sum(wts))
        cv_ = float(np.sum(wts * (vs + y)) / np.sum(wts))
        k = int(np.argmax(vals))
        out.append(Blob(area, float(vals.max()), float(vals.mean()), (cu, cv_),
                        (int(us[k] + x), int(vs[k] + y)), (x, y, w, h)))
    out.sort(key=lambda b: -b.max_temp_k)
    return out


# =============================================================================
# Localisation
# =============================================================================
@dataclass
class HotspotRef:
    id: str
    tag: str
    position: np.ndarray
    radius: float
    length: float = 0.0
    kind: str = ""

    @property
    def extent(self) -> float:
        """Half size of the primitive (radius of its bounding sphere)."""
        return float(math.hypot(self.radius, 0.5 * self.length))


def hotspot_refs(layout) -> List[HotspotRef]:
    refs = []
    for h in layout.hotspots:
        refs.append(HotspotRef(str(h["id"]), str(h.get("tag", h["id"])),
                               np.asarray(h["position"], dtype=float), float(h.get("radius", 0.1)),
                               float(h.get("length", 0.0)), str(h.get("kind", ""))))
    return refs


def pixel_ray_world(T_world_cam: np.ndarray, cam: PinholeCamera, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
    """(origin, unit direction) in the world for pixel (u, v)."""
    d_link = cam.pixel_ray_link(np.array([u]), np.array([v]))[0]
    d = T_world_cam[:3, :3] @ d_link
    return T_world_cam[:3, 3].copy(), d / np.linalg.norm(d)


def point_ray_distance(p: np.ndarray, o: np.ndarray, d: np.ndarray) -> Tuple[float, float]:
    """(perpendicular distance, ray parameter t) of point p to the ray o + t d (d unit).
    Points behind the origin get distance = |p - o| and t < 0."""
    w = np.asarray(p, dtype=float) - o
    t = float(w @ d)
    if t < 0.0:
        return float(np.linalg.norm(w)), t
    return float(np.linalg.norm(w - t * d)), t


def match_ray_to_hotspots(o: np.ndarray, d: np.ndarray, refs: Sequence[HotspotRef], tol_m: float = 0.30,
                          tol_rel: float = 0.04, max_range: float = 30.0
                          ) -> Tuple[Optional[HotspotRef], float, float]:
    """Nearest layout hot spot to the ray within  tol_m + tol_rel * range + extent.
    Returns (ref | None, distance, range)."""
    best, best_q, best_d, best_t = None, math.inf, math.nan, math.nan
    for r in refs:
        dist, t = point_ray_distance(r.position, o, d)
        if t <= 0.05 or t > max_range:
            continue
        tol = tol_m + tol_rel * t + r.extent
        q = dist / tol
        if q < 1.0 and q < best_q:
            best, best_q, best_d, best_t = r, q, dist, t
    return best, best_d, best_t


# =============================================================================
# Temporal confirmation
# =============================================================================
@dataclass
class _Track:
    key: str
    hits: int
    misses: int
    last_t: float
    pos: Optional[np.ndarray]
    uv: Tuple[float, float]
    max_temp_k: float


class TemporalConfirmer:
    """A detection is *confirmed* once it was seen in ``confirm_frames``
    consecutive frames (at most ``max_misses`` frames may be skipped and the
    gap must stay below ``max_gap_s``).  Layout-matched detections are keyed by
    their id; anonymous blobs are associated by world distance (or pixel
    distance when no position is known)."""

    def __init__(self, confirm_frames: int = 3, max_misses: int = 1, max_gap_s: float = 2.0,
                 radius_m: float = 0.6, radius_px: float = 30.0):
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_misses = max(0, int(max_misses))
        self.max_gap_s = float(max_gap_s)
        self.radius_m = float(radius_m)
        self.radius_px = float(radius_px)
        self.tracks: Dict[str, _Track] = {}
        self._next_anon = 0

    def reset(self):
        self.tracks.clear()

    def _associate(self, det: dict, used: set) -> Optional[str]:
        if det.get("id"):
            return str(det["id"])
        pos = det.get("position")
        best, best_d = None, math.inf
        for k, tr in self.tracks.items():
            if not k.startswith("anon") or k in used:
                continue
            if pos is not None and tr.pos is not None:
                dd = float(np.linalg.norm(np.asarray(pos) - tr.pos)) / self.radius_m
            else:
                dd = math.hypot(det["centroid_px"][0] - tr.uv[0], det["centroid_px"][1] - tr.uv[1]) / self.radius_px
            if dd < 1.0 and dd < best_d:
                best, best_d = k, dd
        return best

    def update(self, detections: List[dict], t: float) -> List[dict]:
        """Feed one frame of detections; returns the confirmed ones (each with
        ``confirmed_frames`` set)."""
        used: set = set()
        confirmed: List[dict] = []
        for det in detections:
            key = self._associate(det, used)
            pos = None if det.get("position") is None else np.asarray(det["position"], dtype=float)
            uv = tuple(det.get("centroid_px", (0.0, 0.0)))
            tr = self.tracks.get(key) if key else None
            if tr is not None and (t - tr.last_t) <= self.max_gap_s and key not in used:
                tr.hits += 1
                tr.misses = 0
                tr.last_t, tr.pos, tr.uv = t, pos, uv
                tr.max_temp_k = max(tr.max_temp_k, det.get("max_temp_k", -math.inf))
            else:
                if key is None or key in used:
                    key = f"anon{self._next_anon}"
                    self._next_anon += 1
                tr = _Track(key, 1, 0, t, pos, uv, det.get("max_temp_k", math.nan))
                self.tracks[key] = tr
            used.add(key)
            det["track"] = key
            det["confirmed_frames"] = tr.hits
            if tr.hits >= self.confirm_frames:
                confirmed.append(det)
        # age tracks that were not seen in this frame
        for key in list(self.tracks):
            if key in used:
                continue
            tr = self.tracks[key]
            tr.misses += 1
            if tr.misses > self.max_misses or (t - tr.last_t) > self.max_gap_s:
                del self.tracks[key]
        return confirmed


# =============================================================================
# Visualisation helpers
# =============================================================================
def colorize_temperature(kelvin: np.ndarray, t_min: float, t_max: float,
                         colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Kelvin -> BGR uint8 with a fixed temperature scale (NaN -> t_min)."""
    T = np.nan_to_num(np.asarray(kelvin, dtype=np.float32), nan=t_min)
    g = np.clip((T - t_min) / max(t_max - t_min, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, colormap)


def _nice_step(span: float, target: int = 6) -> float:
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(max(raw, 1e-9)))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag + 1e-12:
            return m * mag
    return 10 * mag


def draw_legend(img: np.ndarray, t_min: float, t_max: float, warning_k: float, alarm_k: float,
                colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Vertical temperature scale bar (deg C) on the right edge, in place."""
    H, W = img.shape[:2]
    bar_h = int(H * 0.62)
    bar_w = max(10, W // 50)
    x1 = W - 58
    x0 = x1 - bar_w
    y0 = (H - bar_h) // 2
    y1 = y0 + bar_h
    # translucent panel
    px0, py0, px1, py1 = max(0, x0 - 22), max(0, y0 - 30), W - 2, min(H - 1, y1 + 14)
    roi = img[py0:py1, px0:px1]
    roi[:] = (roi.astype(np.float32) * 0.45).astype(np.uint8)
    ramp = np.linspace(255, 0, bar_h).astype(np.uint8)[:, None]
    bar = cv2.applyColorMap(np.repeat(ramp, bar_w, axis=1), colormap)
    img[y0:y1, x0:x1] = bar
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (230, 230, 230), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.33, H / 1500.0)

    def y_of(tk: float) -> int:
        return int(round(y1 - 1 - (tk - t_min) / max(t_max - t_min, 1e-6) * (bar_h - 1)))

    c_min, c_max = k_to_c(t_min), k_to_c(t_max)
    step = _nice_step(c_max - c_min, 6)
    c = math.ceil(c_min / step) * step
    while c <= c_max + 1e-9:
        yy = y_of(c + 273.15)
        cv2.line(img, (x1, yy), (x1 + 4, yy), (230, 230, 230), 1)
        cv2.putText(img, f"{c:g}", (x1 + 6, yy + 4), font, fs, (235, 235, 235), 1, cv2.LINE_AA)
        c += step
    for tk, col, name in ((warning_k, STATUS_COLORS["warning"], "W"), (alarm_k, STATUS_COLORS["alarm"], "A")):
        if t_min <= tk <= t_max:
            yy = y_of(tk)
            cv2.line(img, (x0 - 6, yy), (x1 + 2, yy), col, 2)
            cv2.putText(img, name, (x0 - 19, yy + 4), font, fs, col, 1, cv2.LINE_AA)
    cv2.putText(img, "degC", (x0 - 4, y0 - 12), font, fs * 1.05, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def draw_isotherms(img: np.ndarray, kelvin_rgb: np.ndarray, levels: Sequence[Tuple[float, Tuple[int, int, int]]],
                   thickness: int = 2) -> np.ndarray:
    """Contours of kelvin_rgb >= level (in place)."""
    T = np.nan_to_num(kelvin_rgb, nan=0.0)
    for level, color in levels:
        mask = (T >= level).astype(np.uint8)
        if not mask.any():
            continue
        res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[-2]          # OpenCV 3 returns 3 values, 4.x/5.x return 2
        cv2.drawContours(img, contours, -1, color, thickness, cv2.LINE_AA)
    return img


# =============================================================================
# Fusion pipeline
# =============================================================================
@dataclass
class FusionParams:
    # thresholds (<= 0 -> take them from the layout `thermal` block)
    ambient_k: float = -1.0
    warning_k: float = -1.0
    alarm_k: float = -1.0
    segmentation_k: float = -1.0       # <= 0 -> warning_k
    resolution: float = THERMAL_RESOLUTION
    # segmentation
    min_area_px: int = 6
    open_px: int = 0
    # registration
    plane_mode: str = "vertical"       # fronto | vertical
    depth_mode: str = "layout"         # layout | fixed
    plane_distance: float = 2.0        # fixed / fallback plane distance (m)
    depth_min: float = 0.4
    depth_max: float = 30.0
    ecc_enabled: bool = True
    ecc_every_n: int = 2
    ecc_scales: Tuple[float, ...] = (0.25, 0.5)
    ecc_motion: str = "translation"
    ecc_max_iter: int = 60
    ecc_eps: float = 1e-4
    ecc_max_shift_px: float = 30.0
    ecc_max_vertical_px: float = 8.0
    ecc_min_cc: float = 0.20
    ecc_smoothing: float = 0.5
    ecc_epipolar: bool = True
    ecc_skip_hotspot_plane: bool = True
    ecc_reset_depth_change: float = 0.25
    # localisation
    match_tolerance_m: float = 0.30
    match_tolerance_rel: float = 0.04
    max_match_range_m: float = 30.0
    report_normal_in_view: bool = True
    roi_min_excess_k: float = 3.0
    roi_max_range_m: float = 8.0       # beyond: too few pixels for a meaningful ROI measurement
    # temporal confirmation
    confirm_frames: int = 3
    max_misses: int = 1
    max_gap_s: float = 2.0
    track_radius_m: float = 0.6
    track_radius_px: float = 30.0
    # visualisation
    blend_delta_k: float = 4.0
    blend_ramp_k: float = 6.0
    alpha_max: float = 0.70
    display_min_k: float = -1.0        # <= 0 -> ambient - 5
    display_max_k: float = -1.0        # <= 0 -> alarm + 40
    draw_isotherms: bool = True
    draw_legend: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "FusionParams":
        names = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in names}
        if "ecc_scales" in kw:
            kw["ecc_scales"] = tuple(float(s) for s in kw["ecc_scales"])
        return cls(**kw)


@dataclass
class FusionResult:
    detections: List[dict]               # this frame (single-frame, incl. roi measurements)
    confirmed: List[dict]                # temporally confirmed (what is published)
    H: np.ndarray                        # thermal -> RGB homography used
    H0: np.ndarray                       # plane homography before ECC
    plane_depth: float
    depth_source: str
    ecc: Optional[EccResult]
    fusion: Optional[np.ndarray]         # BGR image
    max_temp_k: float


class ThermalFusion:
    """Stateful thermal/RGB fusion (one instance per node)."""

    def __init__(self, layout, params: Optional[FusionParams] = None, rig: Optional[CameraRig] = None):
        self.layout = layout
        self.p = params or FusionParams()
        th = layout.thermal if layout is not None else {}
        self.ambient_k = self.p.ambient_k if self.p.ambient_k > 0 else float(th.get("ambient_k", 298.15))
        self.warning_k = self.p.warning_k if self.p.warning_k > 0 else float(th.get("warning_k", 333.15))
        self.alarm_k = self.p.alarm_k if self.p.alarm_k > 0 else float(th.get("alarm_k", 353.15))
        self._thresholds_from_layout = self.p.warning_k <= 0 and self.p.alarm_k <= 0 and layout is not None
        self.seg_k = self.p.segmentation_k if self.p.segmentation_k > 0 else self.warning_k
        self.t_min = self.p.display_min_k if self.p.display_min_k > 0 else self.ambient_k - 5.0
        self.t_max = self.p.display_max_k if self.p.display_max_k > 0 else self.alarm_k + 40.0
        self.rig = rig or CameraRig.from_robot_model()
        self.refs = hotspot_refs(layout) if layout is not None else []
        self.depth_points = layout_thermal_points(layout) if layout is not None else np.zeros((0, 4))
        self.confirmer = TemporalConfirmer(self.p.confirm_frames, self.p.max_misses, self.p.max_gap_s,
                                           self.p.track_radius_m, self.p.track_radius_px)
        self._frame = 0
        self._correction = np.eye(3)
        self._correction_depth: Optional[float] = None
        self._last_ecc: Optional[EccResult] = None
        self._plane_opt: Tuple[np.ndarray, float] = (np.array([0.0, 0.0, -1.0]), float(self.p.plane_distance))

    # ------------------------------------------------------------ helpers
    def status(self, temp_k: float) -> str:
        if self._thresholds_from_layout:
            return self.layout.temperature_status(temp_k)
        if temp_k >= self.alarm_k:
            return "alarm"
        if temp_k >= self.warning_k:
            return "warning"
        return "normal"

    def rig_for(self, thermal_shape: Tuple[int, int], rgb_shape: Optional[Tuple[int, int]]) -> CameraRig:
        rgb_size = (rgb_shape[1], rgb_shape[0]) if rgb_shape is not None else (self.rig.rgb.width, self.rig.rgb.height)
        return self.rig.resized((thermal_shape[1], thermal_shape[0]), rgb_size)

    # ---------------------------------------------------------- detection
    def detect(self, kelvin: np.ndarray, robot_pose: Optional[Sequence[float]] = None,
               rig: Optional[CameraRig] = None) -> List[dict]:
        """Single-frame hot-spot detections with localisation (no temporal logic)."""
        rig = rig or self.rig_for(kelvin.shape, None)
        cam = rig.thermal
        blobs = segment_hotspots(kelvin, self.seg_k, self.p.min_area_px, self.p.open_px)
        T_wc = rig.thermal_world(robot_pose) if robot_pose is not None else None
        dets: List[dict] = []
        for b in blobs:
            det = {
                "id": None, "tag": None, "position": None, "source": "blob", "localization": None,
                "max_temp_k": b.max_temp_k, "mean_temp_k": b.mean_temp_k, "area_px": b.area_px,
                "centroid_px": [b.centroid_uv[0], b.centroid_uv[1]], "peak_px": [b.peak_uv[0], b.peak_uv[1]],
                "bbox": list(b.bbox), "match_error_m": None, "range_m": None,
            }
            if T_wc is not None:
                o, d = pixel_ray_world(T_wc, cam, *b.centroid_uv)
                ref, dist, rng = match_ray_to_hotspots(o, d, self.refs, self.p.match_tolerance_m,
                                                       self.p.match_tolerance_rel, self.p.max_match_range_m)
                det["_ray"] = (o, d)
                if ref is not None:
                    det.update(id=ref.id, tag=ref.tag, position=[float(v) for v in ref.position],
                               localization="layout", match_error_m=dist, range_m=rng)
            dets.append(det)
        dets = self._merge_same_id(dets)
        if self.p.report_normal_in_view and T_wc is not None:
            dets += self._roi_measurements(kelvin, T_wc, cam, {d["id"] for d in dets if d["id"]})
        for d in dets:
            d["status"] = self.status(d["max_temp_k"])
        return dets

    @staticmethod
    def _merge_same_id(dets: List[dict]) -> List[dict]:
        """Several blobs on the same layout hot spot (e.g. split by an occluder)
        are merged: hottest blob kept, bbox = union, area = sum."""
        out: List[dict] = []
        by_id: Dict[str, dict] = {}
        for d in dets:
            if d["id"] is None:
                out.append(d)
                continue
            if d["id"] not in by_id:
                by_id[d["id"]] = d
                out.append(d)
                continue
            m = by_id[d["id"]]
            u0, v0 = min(m["bbox"][0], d["bbox"][0]), min(m["bbox"][1], d["bbox"][1])
            u1 = max(m["bbox"][0] + m["bbox"][2], d["bbox"][0] + d["bbox"][2])
            v1 = max(m["bbox"][1] + m["bbox"][3], d["bbox"][1] + d["bbox"][3])
            tot = m["area_px"] + d["area_px"]
            m["mean_temp_k"] = (m["mean_temp_k"] * m["area_px"] + d["mean_temp_k"] * d["area_px"]) / tot
            m["area_px"] = tot
            m["bbox"] = [u0, v0, u1 - u0, v1 - v0]
            if d["max_temp_k"] > m["max_temp_k"]:
                m["max_temp_k"], m["peak_px"] = d["max_temp_k"], d["peak_px"]
        return out

    def _roi_measurements(self, kelvin: np.ndarray, T_wc: np.ndarray, cam: PinholeCamera, seen: set) -> List[dict]:
        """Measure layout hot spots that are in view but produced no blob
        (e.g. below the warning threshold -> status 'normal')."""
        out = []
        T_cw = np.linalg.inv(T_wc)
        H, W = kelvin.shape
        for r in self.refs:
            if r.id in seen:
                continue
            pl = T_cw[:3, :3] @ r.position + T_cw[:3, 3]
            if pl[0] < 0.3 or pl[0] > self.p.roi_max_range_m:
                continue
            uv, _ = cam.project_link(pl[None, :])
            u, v = uv[0]
            rad = max(2.0, cam.fx * r.extent / pl[0])
            if not (rad <= u < W - rad and rad <= v < H - rad):
                continue
            u0, u1 = int(max(0, u - rad)), int(min(W, u + rad + 1))
            v0, v1 = int(max(0, v - rad)), int(min(H, v + rad + 1))
            win = kelvin[v0:v1, u0:u1]
            if win.size == 0 or not np.any(np.isfinite(win)):
                continue
            tmax = float(np.nanmax(win))
            if tmax < self.ambient_k + self.p.roi_min_excess_k:
                continue                       # nothing warm there: occluded / not visible
            hot = win[win >= 0.5 * (tmax + self.ambient_k)]
            k = int(np.nanargmax(win))
            pv, pu = divmod(k, win.shape[1])
            out.append({
                "id": r.id, "tag": r.tag, "position": [float(c) for c in r.position], "source": "roi",
                "localization": "layout", "max_temp_k": tmax, "mean_temp_k": float(np.nanmean(hot)),
                "area_px": int(hot.size), "centroid_px": [float(u), float(v)], "peak_px": [u0 + pu, v0 + pv],
                "bbox": [u0, v0, u1 - u0, v1 - v0], "match_error_m": 0.0, "range_m": float(pl[0]),
            })
        return out

    # ------------------------------------------------------- registration
    def estimate_plane(self, robot_pose: Optional[Sequence[float]], detections: List[dict],
                       rig: CameraRig) -> Tuple[float, str, Optional[np.ndarray]]:
        """Registration plane: (depth along the optical axis, source, point_opt|None)."""
        p = self.p
        if robot_pose is None or p.depth_mode == "fixed":
            return p.plane_distance, "fixed", None
        T_cw = np.linalg.inv(rig.thermal_world(robot_pose))
        # 1) matched hot spots (blob detections), hottest first -> plane through it
        for d in detections:
            if d.get("localization") == "layout" and d.get("source") == "blob":
                pl = T_cw[:3, :3] @ np.asarray(d["position"]) + T_cw[:3, 3]
                if p.depth_min <= pl[0] <= p.depth_max:
                    return float(pl[0]), f"hotspot:{d['id']}", R_LINK_OPT.T @ pl
        # 2) median depth of layout thermal objects inside the image
        if len(self.depth_points):
            pts = self.depth_points[:, :3] @ T_cw[:3, :3].T + T_cw[:3, 3]
            uv, z = rig.thermal.project_link(pts)
            ok = (z > p.depth_min) & (z < p.depth_max) & rig.thermal.in_image(uv)
            if np.count_nonzero(ok) >= 3:
                w = self.depth_points[ok, 3]
                zs = z[ok]
                order = np.argsort(zs)
                cw = np.cumsum(w[order])
                med = float(zs[order][np.searchsorted(cw, 0.5 * cw[-1])])
                return med, "layout_median", None
        return p.plane_distance, "fixed", None

    def registration(self, kelvin: np.ndarray, rgb_bgr: Optional[np.ndarray],
                     robot_pose: Optional[Sequence[float]], detections: List[dict],
                     rig: CameraRig) -> Tuple[np.ndarray, np.ndarray, float, str, Optional[EccResult]]:
        depth, src, point_opt = self.estimate_plane(robot_pose, detections, rig)
        H0, n_opt, d_plane = rig.plane_homography(depth, self.p.plane_mode, robot_pose, point_opt)
        self._plane_opt = (n_opt, d_plane)          # also used to place unmatched blobs
        ecc = None
        if self.p.ecc_skip_hotspot_plane and src.startswith("hotspot:"):
            # the plane passes through a matched layout hot spot: the homography
            # is already exact where it matters (eval: ECC only adds noise there)
            return H0, H0, depth, src, None
        if rgb_bgr is not None and self.p.ecc_enabled:
            if (self._correction_depth is not None and
                    abs(depth - self._correction_depth) > self.p.ecc_reset_depth_change * self._correction_depth):
                self._correction = np.eye(3)           # plane changed: stale correction
                self._correction_depth = None
            if self._frame % max(1, self.p.ecc_every_n) == 0:
                ecc = refine_homography_ecc(kelvin, rgb_bgr, H0, self.p.ecc_scales, self.p.ecc_motion,
                                            self.p.ecc_max_iter, self.p.ecc_eps, self.p.ecc_max_shift_px,
                                            self.p.ecc_max_vertical_px, min_cc=self.p.ecc_min_cc,
                                            epipolar_dir=rig.epipolar_direction() if self.p.ecc_epipolar else None)
                if ecc.ok:
                    b = float(np.clip(self.p.ecc_smoothing, 0.0, 1.0))
                    self._correction = (1.0 - b) * self._correction + b * ecc.correction
                    self._correction_depth = depth
                self._last_ecc = ecc
            else:
                ecc = self._last_ecc
        H = self._correction @ H0
        return H / H[2, 2], H0, depth, src, ecc

    # ------------------------------------------------------------ process
    def process(self, kelvin: np.ndarray, rgb_bgr: Optional[np.ndarray] = None,
                robot_pose: Optional[Sequence[float]] = None, stamp: float = 0.0,
                render: bool = True) -> FusionResult:
        T = np.asarray(kelvin, dtype=np.float32)
        rig = self.rig_for(T.shape, rgb_bgr.shape if rgb_bgr is not None else None)
        dets = self.detect(T, robot_pose, rig)
        H, H0, depth, src, ecc = self.registration(T, rgb_bgr, robot_pose, dets, rig)
        T_cw = np.linalg.inv(rig.thermal_world(robot_pose)) if robot_pose is not None else None
        for d in dets:
            # per-detection registration: a layout-matched hot spot gets the
            # homography of the plane through its own position, so that its box
            # is exact even when other objects at other depths share the frame
            d["_H"] = H
            if T_cw is not None and d.get("localization") == "layout":
                pl = T_cw[:3, :3] @ np.asarray(d["position"], dtype=float) + T_cw[:3, 3]
                if self.p.depth_min <= pl[0] <= self.p.depth_max:
                    d["_H"] = rig.plane_homography(float(pl[0]), self.p.plane_mode, robot_pose, R_LINK_OPT.T @ pl)[0]
            d["bbox_rgb"] = transform_bbox(d["_H"], d["bbox"])
            if d["position"] is None and "_ray" in d:      # ray / registration-plane intersection
                o, dr = d["_ray"]
                n_opt, d_plane = self._plane_opt
                # world ray direction -> thermal optical frame; plane n.X + d = 0 there
                dr_opt = R_LINK_OPT.T @ (rig.thermal_world(robot_pose)[:3, :3].T @ dr)
                denom = float(n_opt @ dr_opt)
                if denom < -1e-6:
                    tt = -d_plane / denom                  # distance along the unit ray
                    if tt <= self.p.depth_max:
                        d["position"] = [float(v) for v in (o + tt * dr)]
                        d["localization"] = "ray_depth"
                        d["range_m"] = tt
            d.pop("_ray", None)
        confirmed = self.confirmer.update(dets, stamp)
        self._frame += 1
        tmax = float(np.nanmax(T)) if np.any(np.isfinite(T)) else math.nan
        fusion = self.render_fusion(T, rgb_bgr, H, confirmed, rig, depth, src, ecc) if render else None
        return FusionResult(dets, confirmed, H, H0, depth, src, ecc, fusion, tmax)

    # ------------------------------------------------------------- output
    @staticmethod
    def hotspot_json(d: dict) -> dict:
        """One entry of /inspection/thermal_alerts (DESIGN_SPEC section 4 + extras)."""
        return {
            "id": d.get("id"), "tag": d.get("tag"),
            "max_temp_c": round(k_to_c(d["max_temp_k"]), 2),
            "mean_temp_c": round(k_to_c(d["mean_temp_k"]), 2),
            "position": None if d.get("position") is None else [round(float(v), 3) for v in d["position"]],
            "status": d.get("status", "normal"),
            "bbox": [int(v) for v in d["bbox"]],
            # extras (not required by the contract)
            "bbox_rgb": [int(v) for v in d.get("bbox_rgb", d["bbox"])],
            "area_px": int(d.get("area_px", 0)),
            "source": d.get("source"),
            "localization": d.get("localization"),
            "range_m": None if d.get("range_m") is None else round(float(d["range_m"]), 3),
            "match_error_m": None if d.get("match_error_m") is None else round(float(d["match_error_m"]), 3),
            "confirmed_frames": int(d.get("confirmed_frames", 0)),
        }

    def alerts_dict(self, result: FusionResult, stamp: float) -> dict:
        """Payload of /inspection/thermal_alerts."""
        ecc = result.ecc
        return {
            "stamp": float(stamp),
            "hotspots": [self.hotspot_json(d) for d in result.confirmed],
            "n_candidates": len(result.detections),
            "frame_max_temp_c": None if not math.isfinite(result.max_temp_k) else round(k_to_c(result.max_temp_k), 2),
            "thresholds_c": {"warning": round(k_to_c(self.warning_k), 2), "alarm": round(k_to_c(self.alarm_k), 2)},
            "registration": {
                "plane_depth_m": round(float(result.plane_depth), 3),
                "depth_source": result.depth_source,
                "plane_mode": self.p.plane_mode,
                "ecc_ok": bool(ecc.ok) if ecc is not None else None,
                "ecc_cc": None if ecc is None or not math.isfinite(ecc.cc) else round(float(ecc.cc), 3),
                "ecc_shift_px": None if ecc is None else [round(float(v), 2) for v in ecc.shift_px],
            },
        }

    # ---------------------------------------------------------- rendering
    def render_fusion(self, kelvin: np.ndarray, rgb_bgr: Optional[np.ndarray], H: np.ndarray,
                      hotspots: List[dict], rig: CameraRig, depth: float = math.nan, depth_src: str = "",
                      ecc: Optional[EccResult] = None) -> np.ndarray:
        """Colour-mapped thermal blended on the RGB image (or thermal only)."""
        p = self.p
        W, Hh = rig.rgb.width, rig.rgb.height
        T = np.asarray(kelvin, dtype=np.float32)
        if rgb_bgr is not None:
            base = rgb_bgr.copy()
            H_disp = H
        else:   # no RGB: thermal-only view scaled to the RGB canvas size
            base = None
            H_disp = np.diag([W / T.shape[1], Hh / T.shape[0], 1.0])
        T_fill = np.where(np.isfinite(T), T, self.ambient_k - 50.0).astype(np.float32)
        Tw = cv2.warpPerspective(T_fill, H_disp, (W, Hh), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=float(self.ambient_k - 50.0))
        valid = cv2.warpPerspective(np.ones(T.shape, np.uint8), H_disp, (W, Hh), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        color = colorize_temperature(Tw, self.t_min, self.t_max)
        if base is None:
            out = color.copy()
            out[~valid] = 0
        else:
            a = p.alpha_max * np.clip((Tw - (self.ambient_k + p.blend_delta_k)) / max(p.blend_ramp_k, 1e-6), 0.0, 1.0)
            a[~valid] = 0.0
            a = a[..., None].astype(np.float32)
            out = (base.astype(np.float32) * (1.0 - a) + color.astype(np.float32) * a).astype(np.uint8)
            # thin outline of the thermal field of view
            fov = apply_homography(H, np.array([[0, 0], [T.shape[1], 0], [T.shape[1], T.shape[0]], [0, T.shape[0]]]))
            cv2.polylines(out, [np.round(fov).astype(np.int32)], True, (200, 200, 200), 1, cv2.LINE_AA)
        if p.draw_isotherms:
            draw_isotherms(out, np.where(valid, Tw, 0.0), [(self.warning_k, STATUS_COLORS["warning"]),
                                                           (self.alarm_k, STATUS_COLORS["alarm"])], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.4, Hh / 1200.0)
        for d in hotspots:
            col = STATUS_COLORS.get(d.get("status", "normal"), (255, 255, 255))
            Hd = d.get("_H", H_disp) if rgb_bgr is not None else H_disp
            u, v, w, h = transform_bbox(Hd, d["bbox"])
            cv2.rectangle(out, (u, v), (u + w, v + h), col, 2)
            pk = apply_homography(Hd, np.array([d["peak_px"]], dtype=float))[0]
            cv2.drawMarker(out, (int(pk[0]), int(pk[1])), col, cv2.MARKER_CROSS, 14, 2)
            name = d.get("id") or "?"
            label = f"{name} {k_to_c(d['max_temp_k']):.1f}C {d.get('status', '').upper()}"
            (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
            ty = v - 6 if v - th - 8 > 0 else v + h + th + 6
            cv2.rectangle(out, (u, ty - th - 4), (u + tw + 6, ty + 4), (0, 0, 0), -1)
            cv2.putText(out, label, (u + 3, ty), font, fs, col, 1, cv2.LINE_AA)
        if p.draw_legend:
            draw_legend(out, self.t_min, self.t_max, self.warning_k, self.alarm_k)
        # header
        tmax = float(np.nanmax(T)) if np.any(np.isfinite(T)) else math.nan
        hdr = f"THERMAL {'FUSION' if rgb_bgr is not None else 'ONLY (no RGB)'}  max {k_to_c(tmax):.1f}C"
        if rgb_bgr is not None and math.isfinite(depth):
            hdr += f"  plane {depth:.2f}m ({depth_src})"
            if ecc is not None and ecc.ok:
                hdr += f"  ECC {ecc.shift_px[0]:+.1f},{ecc.shift_px[1]:+.1f}px"
        (tw, th), _ = cv2.getTextSize(hdr, font, fs, 1)
        cv2.rectangle(out, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
        cv2.putText(out, hdr, (6, th + 6), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        return out


# =============================================================================
# Time synchronisation of the three input streams
# =============================================================================
class StampedBuffer:
    """Small time-ordered ring buffer (stamp, item) with nearest-stamp lookup
    and 6-DOF pose interpolation.  Used to pair a thermal frame with the RGB
    frame and the robot pose of (almost) the same time: at 0.8 rad/s yaw rate
    a 0.1 s offset would misalign the overlay by ~66 RGB pixels."""

    def __init__(self, maxlen: int = 20):
        self.maxlen = int(maxlen)
        self.t: List[float] = []
        self.items: list = []

    def __len__(self) -> int:
        return len(self.t)

    def add(self, stamp: float, item) -> None:
        stamp = float(stamp)
        if self.t and stamp < self.t[-1] - 1.0:      # clock jumped back (sim reset): start over
            self.t.clear()
            self.items.clear()
        k = len(self.t)
        while k > 0 and self.t[k - 1] > stamp:       # keep sorted (rare out-of-order arrivals)
            k -= 1
        self.t.insert(k, stamp)
        self.items.insert(k, item)
        if len(self.t) > self.maxlen:
            del self.t[0]
            del self.items[0]

    def latest(self):
        return (self.t[-1], self.items[-1]) if self.t else (None, None)

    def nearest(self, stamp: float, tolerance: float):
        """Item whose stamp is closest to ``stamp`` (None if |dt| > tolerance)."""
        if not self.t:
            return None
        ts = np.asarray(self.t)
        k = int(np.argmin(np.abs(ts - stamp)))
        return self.items[k] if abs(ts[k] - stamp) <= tolerance else None

    def interpolate_pose(self, stamp: float, tolerance: float) -> Optional[Tuple[float, ...]]:
        """Pose (x, y, z, roll, pitch, yaw) at ``stamp``: linear interpolation
        between the bracketing samples (angles along the shortest arc); the
        nearest sample within ``tolerance`` otherwise (no extrapolation)."""
        if not self.t:
            return None
        ts = self.t
        k = int(np.searchsorted(np.asarray(ts), stamp))
        if 0 < k < len(ts) and ts[k] - ts[k - 1] <= 2.0 * tolerance + 1e-9:
            t0, t1 = ts[k - 1], ts[k]
            a = 0.0 if t1 <= t0 else (stamp - t0) / (t1 - t0)
            p0, p1 = np.asarray(self.items[k - 1], float), np.asarray(self.items[k], float)
            out = p0 + a * (p1 - p0)
            d_ang = (p1[3:] - p0[3:] + math.pi) % (2.0 * math.pi) - math.pi
            out[3:] = p0[3:] + a * d_ang
            out[3:] = (out[3:] + math.pi) % (2.0 * math.pi) - math.pi
            return tuple(float(v) for v in out)
        near = self.nearest(stamp, tolerance)
        return None if near is None else tuple(float(v) for v in near)


# =============================================================================
# Layout thermal objects (depth estimation for the registration plane)
# =============================================================================
def layout_thermal_points(layout, step: float = 0.25) -> np.ndarray:
    """(N, 4) array [x, y, z, weight] of points on the layout's thermal objects:
    hot spots (weight 3), pipe-rack pipe and the corridor pipe stub (weight 1)."""
    pts: List[Tuple[float, float, float, float]] = []
    for h in layout.hotspots:
        x, y, z = h["position"]
        pts.append((float(x), float(y), float(z), 3.0))
    pr = layout.get("pipe_rack") or {}
    pipe = pr.get("pipe")
    if pipe:
        a, b = np.asarray(pipe["start"], float), np.asarray(pipe["end"], float)
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        for s in np.linspace(0.0, 1.0, n):
            q = a + s * (b - a)
            pts.append((q[0], q[1], q[2], 1.0))
    stub = layout.get("corridor_pipe_stub")
    if stub:
        x, y = stub["position"]
        z0 = layout.floor_z(x, y)
        for z in np.arange(z0, z0 + float(stub.get("height", 2.0)), step):
            pts.append((float(x), float(y), float(z), 1.0))
    return np.asarray(pts, dtype=float).reshape(-1, 4)
