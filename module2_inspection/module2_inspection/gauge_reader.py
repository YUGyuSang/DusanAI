"""
Analog gauge reader (ROS-free, numpy + OpenCV) - Module 2 of the plant-inspection robot.

Pipeline for one camera frame (DESIGN_SPEC section 5)::

    enhance  -> white-point stretch, adaptive gamma for dark frames, CLAHE on the L channel
                (LAB), noise-adaptive denoising (Immerkaer noise estimate)
    detect   -> multi-scale Hough circles (image pyramid) + bright-blob ellipses (optionally
                inside the ROI predicted from the robot pose); every candidate is snapped to
                the white-face edge (median radial step), duplicates are merged keeping the
                best-centred one, the most promising get a centre/shape correction (per-ray
                edges + robust ellipse fit); validated by face brightness, dark rim ring,
                closure, tick-ring energy and needle presence
    fit      -> ray-wise matched step filter on the face/rim edge, robust (Tukey IRLS)
                direct least-squares ellipse fit, 2 refinement rounds; edge spread gives
                the blur metric (frames/dials that are too blurred are rejected)
    rectify  -> metric rectification of the dial plane from the fitted ellipse and the
                camera intrinsics (circle-pose-from-conic: two-fold ambiguity resolved by
                the predicted dial normal or by the dark needle hub); affine fallback
    calibrate-> polar unwrap; angular darkness profiles of the tick ring (outer + major-only
                bands) and a redness profile of the danger band are correlated (FFT, 0.5 deg
                bins, parabolic sub-bin peak) with templates rendered by
                dusan_common.gauge_design -> in-plane rotation of the printed scale,
                mirror flag (the red band marks the max end) and number of major divisions
    needle   -> coarse angle = peak of the angular darkness profile in an annulus that
                excludes the hub, the needle tail, the labels and the 90 deg gap; fine angle
                = robust line fit through per-radius sub-degree darkness centroids
    ocr      -> scale labels read at every major tick (gauge_ocr): characters + RANSAC
                range fit, or analysis-by-synthesis verification of standard ranges
    value    -> dusan_common.gauge_design.angle_to_value, confidence score
    fusion   -> GaugeFusion: per-gauge median / MAD outlier rejection over the last N
                samples within T seconds

All lengths on the dial are "dial units" = fractions of the dial texture side, so the white
face radius is ``gauge_design.R_FACE`` (0.44).  Dial angles follow gauge_design: degrees,
counter-clockwise from 3 o'clock, min at 225 deg, max at -45 deg.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dusan_common import gauge_design as gd
from module2_inspection.gauge_ocr import (DigitOCR, LABEL_HEIGHT_DIAL, RangeEstimate, hypothesis_confidence,
                                          infer_range, major_label_angles, n_major_divisions, score_ranges)

# ---------------------------------------------------------------------------------- dial layout
R_FACE = gd.R_FACE                       # radius of the fitted face/rim edge
TICK_OUTER_BAND = (0.368, 0.404)         # all ticks + danger band
TICK_INNER_BAND = (0.338, 0.362)         # major ticks only (minor ticks start at 0.37)
RED_BAND = (0.370, 0.410)                # danger band (0.365 .. 0.412)
NEEDLE_COARSE_BAND = (0.09, 0.22)        # excludes hub (0.035), tail (0.08) and labels (>0.235)
NEEDLE_FINE_RADII = tuple(np.round(np.r_[np.arange(0.075, 0.2251, 0.006),
                                         np.arange(0.305, 0.3351, 0.006)], 4))
FACE_REF_BANDS = ((0.414, 0.433), (0.302, 0.330))
POLAR_R = np.round(np.arange(0.07, 0.4351, 0.005), 4)
N_THETA = 720                            # 0.5 deg polar bins
THETA_DEG = np.arange(N_THETA) * (360.0 / N_THETA)
LABEL_PATCH_W, LABEL_PATCH_H = 0.12, 0.05     # label patch size (dial units)
LABEL_PPU = 22.0 / LABEL_HEIGHT_DIAL          # patch pixels per dial unit (label ~22 px tall)

# representative ranges giving 4..8 major divisions (unknown-range calibration templates)
GENERIC_RANGES = {4: (0.0, 100.0), 5: (0.0, 10.0), 6: (0.0, 6.0), 7: (0.0, 7.0), 8: (0.0, 16.0)}


# ---------------------------------------------------------------------------------- config
@dataclass
class GaugeReaderConfig:
    """All tunables of the reader (mirrored 1:1 by the ROS parameters of gauge_reader_node)."""

    # enhancement
    white_percentile: float = 99.5     # white point for the exposure stretch
    max_gain: float = 12.0             # max exposure gain for dark frames
    gamma_target: float = 0.42         # target mean (0..1) of the adaptive gamma
    gamma_min: float = 0.4             # strongest brightening gamma
    clahe_clip: float = 2.0
    clahe_tiles: int = 8
    denoise_sigma_thresh: float = 2.5  # noise std [DN] above which smoothing is applied
    # detection
    min_radius_px: float = 14.0        # face radius limits
    max_radius_px: float = 320.0
    hough_canny: float = 90.0
    hough_acc: float = 16.0
    max_candidates: int = 24
    max_refine: int = 10               # candidates per search that get the centre correction
    min_detection_score: float = 0.25
    max_dials: int = 4                 # dials read per frame without hints
    roi_scale: float = 2.6             # hinted ROI half size = roi_scale * r_pred + roi_margin_px
    roi_margin_px: float = 70.0
    # ellipse fit
    n_rays: int = 180
    # blur gating
    max_blur_ratio: float = 0.35       # reject if blur extent / face radius exceeds this
    mid_level_blur_px: float = 3.0     # above this blur the face edge is re-fitted on 50 % crossings
    min_frame_sharpness: float = 0.0   # optional global gate (normalised Laplacian variance)
    # calibration
    allow_mirror: bool = True
    inner_band_weight: float = 0.5
    red_weight: float = 0.6
    min_calibration_ncc: float = 0.5   # real dials >= 0.64 in the benchmark, round distractors <= 0.4
    # needle
    needle_model: str = "auto"         # analysis-by-synthesis refinement: auto | always | off
    needle_model_blur_px: float = 2.5  # 'auto' uses the model above this blur extent
    # ocr
    ocr_enabled: bool = True
    ocr_min_label_px: float = 7.0      # char OCR if the label is at least this tall in the image
    ocr_min_label_px_hyp: float = 4.0  # standard-range verification down to this label size
    ocr_min_confidence: float = 0.35
    ocr_div_ncc_margin: float = 0.08    # OCR also tries division counts within this NCC of the best
    # output
    min_face_contrast: float = 0.25    # (face - ink) / face of a real dial is ~0.9 when sharp
    min_confidence: float = 0.30
    clamp_margin_fs: float = 0.04      # values are clamped to [min - m*FS, max + m*FS]

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GaugeReaderConfig":
        cfg = cls()
        if d:
            names = {f.name: f for f in fields(cls)}
            for k, v in d.items():
                if k in names:
                    setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


# ---------------------------------------------------------------------------------- data types
@dataclass
class GaugeHint:
    """Prior knowledge about one gauge expected in the frame (from the association step)."""

    gauge_id: Optional[str] = None
    center: Optional[Tuple[float, float]] = None     # predicted face centre [px]
    radius: Optional[float] = None                   # predicted face radius (semi-major) [px]
    vrange: Optional[Tuple[float, float]] = None     # (min, max) from the gauge registry
    unit: str = ""
    normal_cam: Optional[np.ndarray] = None          # predicted face normal, optical frame
    search_radius: Optional[float] = None            # override of the ROI half size [px]
    do_ocr: bool = True


@dataclass
class DialCandidate:
    cx: float
    cy: float
    a: float          # semi-axes of the face edge [px]
    b: float
    ang: float        # rotation of the a-axis from +u towards +v [rad]
    score: float = 0.0
    source: str = ""
    parts: Dict[str, float] = field(default_factory=dict)


@dataclass
class Calibration:
    rotation_deg: float      # canonical angle of dial angle 0
    mirrored: bool
    n_div: int
    vrange: Optional[Tuple[float, float]]
    ncc: float
    margin: float
    ncc_by_div: Dict[int, float] = field(default_factory=dict)   # best NCC per major-division count


@dataclass
class GaugeReading:
    value: Optional[float]
    vmin: Optional[float]
    vmax: Optional[float]
    fraction: float                    # needle position along the 270 deg scale (0 = min, 1 = max)
    angle_deg: float                   # needle angle in the dial frame (gauge_design convention)
    confidence: float
    center: Tuple[float, float]        # image of the dial centre (hub) [px]
    ellipse: Tuple[float, float, float, float, float]   # cx, cy, a, b, angle_deg (face edge)
    H_dial2img: np.ndarray             # canonical dial frame -> image
    rotation_deg: float
    mirrored: bool
    n_div: int
    range_source: str                  # 'registry' | 'ocr' | 'none'
    range_ocr: Optional[Tuple[float, float]] = None
    ocr_confidence: float = 0.0
    ocr_method: str = ""
    ocr_labels: List[Tuple[int, str, float]] = field(default_factory=list)
    blur_px: float = 0.0
    gauge_id: Optional[str] = None
    accepted: bool = True
    reject_reason: str = ""
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def radius_px(self) -> float:
        return float(self.ellipse[2])

    def dial_to_image(self, pts_dial) -> np.ndarray:
        """Map points given in the *printed* dial frame (gauge_design) to image pixels."""
        return apply_h(self.H_dial2img @ dial_to_canonical(self.rotation_deg, self.mirrored), pts_dial)

    def needle_tip_px(self, r: float = gd.R_NEEDLE) -> Tuple[float, float]:
        a = math.radians(self.angle_deg)
        u, v = self.dial_to_image([[r * math.cos(a), r * math.sin(a)]])[0]
        return float(u), float(v)


# ---------------------------------------------------------------------------------- geometry helpers
def apply_h(H, pts) -> np.ndarray:
    p = np.atleast_2d(np.asarray(pts, float))
    q = p @ np.asarray(H, float)[:, :2].T + np.asarray(H, float)[:, 2]
    return q[:, :2] / q[:, 2:3]


def dial_to_canonical(rotation_deg: float, mirrored: bool) -> np.ndarray:
    """3x3 map printed-dial frame -> canonical (rectified) frame: phi_c = rho + s * phi_d."""
    r = math.radians(rotation_deg)
    c, s = math.cos(r), math.sin(r)
    M = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if mirrored:
        M = M @ np.diag([1.0, -1.0, 1.0])
    return M


def ellipse_conic(cx, cy, a, b, ang) -> np.ndarray:
    """3x3 conic matrix C (x^T C x = 0) of an ellipse."""
    c, s = math.cos(ang), math.sin(ang)
    A_aff = np.array([[a * c, -b * s, cx], [a * s, b * c, cy], [0.0, 0.0, 1.0]])
    Ai = np.linalg.inv(A_aff)
    return Ai.T @ np.diag([1.0, 1.0, -1.0]) @ Ai


def fit_ellipse_direct(x, y, w=None):
    """Weighted direct least-squares ellipse fit (Fitzgibbon; Halir-Flusser numerics).

    Returns conic coefficients (A, B, C, D, E, F) of A x^2 + B xy + C y^2 + D x + E y + F = 0
    in the original coordinates, or None."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 6:
        return None
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    sw = np.sqrt(np.clip(w, 0, None))
    mx, my = np.average(x, weights=w + 1e-12), np.average(y, weights=w + 1e-12)
    sc = math.sqrt(2.0) / max(1e-9, float(np.average(np.hypot(x - mx, y - my), weights=w + 1e-12)))
    xn, yn = (x - mx) * sc, (y - my) * sc
    D1 = np.stack([xn * xn, xn * yn, yn * yn], 1) * sw[:, None]
    D2 = np.stack([xn, yn, np.ones_like(xn)], 1) * sw[:, None]
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
    try:
        evals, evecs = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    evecs = np.real(evecs)
    cond = 4 * evecs[0] * evecs[2] - evecs[1] ** 2
    idx = np.where(cond > 0)[0]
    if len(idx) == 0:
        return None
    a1 = evecs[:, idx[0]]
    a2 = T @ a1
    A, B, C = a1
    D, E, F = a2
    # undo the normalisation x_n = sc (x - mx)
    A2, B2, C2 = A * sc * sc, B * sc * sc, C * sc * sc
    D2_ = D * sc - 2 * A2 * mx - B2 * my
    E2_ = E * sc - 2 * C2 * my - B2 * mx
    F2_ = (A2 * mx * mx + B2 * mx * my + C2 * my * my - D * sc * mx - E * sc * my + F)
    return np.array([A2, B2, C2, D2_, E2_, F2_])


def conic_to_params(coef):
    """Conic coefficients -> (cx, cy, a, b, ang) with a >= b, or None if not an ellipse."""
    A, B, C, D, E, F = coef
    M2 = np.array([[A, B / 2.0], [B / 2.0, C]])
    det = np.linalg.det(M2)
    if det <= 0:
        return None
    try:
        cx, cy = np.linalg.solve(2 * M2, [-D, -E])
    except np.linalg.LinAlgError:
        return None
    Fc = A * cx * cx + B * cx * cy + C * cy * cy + D * cx + E * cy + F
    mu, V = np.linalg.eigh(M2)
    if Fc == 0 or np.any(-Fc / mu <= 0):
        return None
    ax = np.sqrt(-Fc / mu)          # semi-axis along each eigenvector
    i_major = int(np.argmax(ax))
    a, b = float(ax[i_major]), float(ax[1 - i_major])
    vx, vy = V[:, i_major]
    return float(cx), float(cy), a, b, float(math.atan2(vy, vx))


def sampson_distance(coef, x, y) -> np.ndarray:
    A, B, C, D, E, F = coef
    f = A * x * x + B * x * y + C * y * y + D * x + E * y + F
    gx = 2 * A * x + B * y + D
    gy = B * x + 2 * C * y + E
    return f / np.maximum(np.hypot(gx, gy), 1e-12)


def robust_ellipse_fit(x, y, w0=None, iters: int = 5):
    """Tukey-IRLS direct ellipse fit. Returns (params, inlier mask, rms residual) or None."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(x) if w0 is None else np.asarray(w0, float).copy()
    base = w.copy()
    coef = None
    for _ in range(iters):
        coef = fit_ellipse_direct(x, y, w)
        if coef is None:
            return None
        d = sampson_distance(coef, x, y)
        s = 1.4826 * float(np.median(np.abs(d))) + 0.05
        u = d / (4.685 * s)
        w = base * np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        if np.count_nonzero(w) < 6:
            return None
    params = conic_to_params(coef)
    if params is None:
        return None
    d = sampson_distance(coef, x, y)
    inl = w > 1e-3
    rms = float(np.sqrt(np.average(d[inl] ** 2))) if np.any(inl) else float("inf")
    return params, inl, rms


def circle_poses_from_conic(conic: np.ndarray, K: np.ndarray):
    """Supporting planes of a circle whose image is ``conic`` (pixel coordinates).

    The viewing cone Q = K^T C K has eigenvalues l1 >= l2 > 0 > l3 (after a sign flip).
    Subtracting l2 I makes Q - l2 I a pair of planes, so every plane parallel to
        n ~ (sqrt(l1 - l2), 0, -+ sqrt(l2 - l3))      (eigen-basis)
    cuts the cone in a circle (the section is the intersection with a sphere).  Returns up to
    two solutions [(n_cam (unit, towards the camera), centre_cam, radius)], with the scale
    fixed so that the plane is at unit distance along the eigen normal (only ratios matter).
    """
    Q = K.T @ conic @ K
    Q = 0.5 * (Q + Q.T)
    lam, V = np.linalg.eigh(Q)
    if np.sum(lam > 0) == 1:
        lam = -lam
    if np.sum(lam > 0) != 2:
        return []
    order = np.argsort(-lam)
    lam, V = lam[order], V[:, order]
    l1, l2, l3 = lam
    if l2 <= 0 or l3 >= 0:
        return []
    alpha = math.sqrt(max(l1 - l2, 0.0))
    beta = math.sqrt(max(l2 - l3, 0.0))
    sols = []
    for sgn in (1.0, -1.0):
        n_e = np.array([alpha, 0.0, -sgn * beta])
        m_e = np.array([alpha, 0.0, sgn * beta])
        nn = float(n_e @ n_e)
        S0 = -(1.0 / (2.0 * l2)) * m_e                   # sphere centre (plane n.X = 1)
        t = (float(n_e @ S0) - 1.0) / nn
        Xc = S0 - t * n_e                                # circle centre
        rs2 = (1.0 / (2.0 * l2)) ** 2 * float(m_e @ m_e)
        dist2 = (float(n_e @ S0) - 1.0) ** 2 / nn
        rc = math.sqrt(max(rs2 - dist2, 0.0))
        Xc_cam = V @ Xc
        n_cam = V @ (n_e / math.sqrt(nn))
        if Xc_cam[2] < 0:
            Xc_cam = -Xc_cam
        if float(n_cam @ Xc_cam) > 0:
            n_cam = -n_cam
        if rc > 0:
            sols.append((n_cam, Xc_cam, rc))
        if alpha < 1e-12:
            break                                        # fronto-parallel: one solution
    return sols


def homography_from_circle_pose(K, n_cam, Xc_cam, rc) -> np.ndarray:
    """Canonical dial frame (x right, y up as seen in the image, face radius R_FACE) -> image."""
    n = n_cam / np.linalg.norm(n_cam)
    up = np.array([0.0, -1.0, 0.0])                      # image 'up' in the optical frame
    e_v = up - (up @ n) * n
    if np.linalg.norm(e_v) < 1e-6:
        e_v = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e_v /= np.linalg.norm(e_v)
    e_u = np.cross(e_v, n)                               # e_u x e_v = n (towards the camera)
    k = rc / R_FACE
    H = K @ np.stack([k * e_u, k * e_v, Xc_cam], 1)
    return H / H[2, 2]


def homography_affine(cx, cy, a, b, ang) -> np.ndarray:
    """Affine fallback: canonical dial frame -> image (y flipped: dial y is up)."""
    c, s = math.cos(ang), math.sin(ang)
    A = np.array([[a * c, -b * s, cx], [a * s, b * c, cy], [0.0, 0.0, 1.0]])
    return A @ np.diag([1.0 / R_FACE, -1.0 / R_FACE, 1.0])


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise std estimate (Laplacian-difference mask)."""
    g = gray.astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(g, -1, M)[1:-1, 1:-1]
    return float(np.mean(np.abs(conv)) * math.sqrt(math.pi / 2.0) / 6.0)


def frame_sharpness(gray: np.ndarray) -> float:
    """Global blur metric: variance of the Laplacian normalised by the squared mean."""
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(lap.var() / max(1.0, float(g.mean()) ** 2) * 1000.0)


def _circ_ncc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised circular cross-correlation c[k] = sum_j a[j] b[j-k] (a, b length N)."""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.zeros_like(a)
    return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a)) / (na * nb)


def _masked_circ_ncc(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Masked normalised circular cross-correlation (Padfield 2012), weights w on ``a``.

    c[k] = NCC over {j : w[j] > 0} of a[j] and b[j - k]; all sums via FFT."""
    n = len(a)
    F = np.fft.rfft
    iF = lambda x: np.fft.irfft(x, n=n)
    Fw, Fb, Fb2 = F(w), F(b), F(b * b)
    aw = a * w
    sw = float(w.sum())
    if sw < 10:
        return np.zeros(n)
    s_ab = iF(F(aw) * np.conj(Fb))            # sum_j w a[j] b[j-k]
    s_b = iF(Fw * np.conj(Fb))                # sum_j w b[j-k]
    s_b2 = iF(Fw * np.conj(Fb2))              # sum_j w b[j-k]^2
    s_a = float(aw.sum())
    s_a2 = float((aw * a).sum())
    num = s_ab - s_a * s_b / sw
    den = np.sqrt(np.maximum(s_a2 - s_a * s_a / sw, 1e-12) * np.maximum(s_b2 - s_b * s_b / sw, 1e-12))
    return num / den


def _gauss_circ(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Circular Gaussian smoothing of a 1-D profile."""
    if sigma_bins < 0.3:
        return x
    n = len(x)
    k = np.fft.rfftfreq(n) * 2 * np.pi
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-0.5 * (k * sigma_bins) ** 2), n=n)


def _parabolic(ym, y0, yp) -> float:
    den = ym - 2 * y0 + yp
    return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5))


def needle_half_width(r: float) -> float:
    """Half width of the gauge_design needle polygon at radius r (dial units)."""
    return 0.022 + (0.006 - 0.022) * (r + 0.08) / (gd.R_NEEDLE + 0.08)


def _motion_kernel(length: float, angle_rad: float) -> np.ndarray:
    """Normalised linear motion-blur kernel; angle in image coordinates (atan2(dv, du))."""
    length = float(max(length, 0.0))
    k = int(math.ceil(length)) + 3
    k += (k + 1) % 2
    ker = np.zeros((k, k), np.float32)
    c = (k - 1) / 2.0
    d = (math.cos(angle_rad), math.sin(angle_rad))
    for t in np.linspace(-length / 2, length / 2, max(8, int(length * 6))):
        x, y = c + t * d[0], c + t * d[1]
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        ker[y0, x0] += (1 - fx) * (1 - fy)
        ker[y0, x0 + 1] += fx * (1 - fy)
        ker[y0 + 1, x0] += (1 - fx) * fy
        ker[y0 + 1, x0 + 1] += fx * fy
    return ker / max(ker.sum(), 1e-9)


# ---------------------------------------------------------------------------------- templates
class _TemplateBank:
    """Angular tick-ring / danger-band profiles rendered from gauge_design (cached)."""

    def __init__(self, size: int = 1024):
        self.size = int(size)
        self._cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def profiles(self, vmin: float, vmax: float):
        key = (round(float(vmin), 9), round(float(vmax), 9))
        if key in self._cache:
            return self._cache[key]
        S = self.size
        img = gd.render_gauge_face(vmin, vmin, vmax, unit="", tag="", size=S, draw_needle=False).astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = float(np.mean(gd.COLORS["face"]))
        ink = float(np.mean(gd.COLORS["ink"]))
        dark = (face - gray) / (face - ink)
        red = np.clip(img[..., 2] - 0.5 * (img[..., 0] + img[..., 1]), 0, None) / face

        def band(arr, r0, r1, n=10):
            rr = np.linspace(r0, r1, n)
            th = np.radians(THETA_DEG)
            x = S / 2.0 + np.outer(rr, np.cos(th)) * S
            y = S / 2.0 - np.outer(rr, np.sin(th)) * S
            v = cv2.remap(arr, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
            return v.mean(axis=0).astype(np.float64)

        out = (band(dark, *TICK_OUTER_BAND), band(dark, *TICK_INNER_BAND), band(red, *RED_BAND))
        self._cache[key] = out
        return out


# ---------------------------------------------------------------------------------- reader
class GaugeReader:
    """Reads analog gauges in BGR frames.  Construct once, call :meth:`read` per frame."""

    def __init__(self, config: Optional[GaugeReaderConfig] = None, camera_matrix: Optional[np.ndarray] = None):
        self.cfg = config or GaugeReaderConfig()
        self.K = None if camera_matrix is None else np.asarray(camera_matrix, float)
        self.templates = _TemplateBank()
        self._ocr: Optional[DigitOCR] = None
        self._clahe = cv2.createCLAHE(clipLimit=float(self.cfg.clahe_clip),
                                      tileGridSize=(int(self.cfg.clahe_tiles), int(self.cfg.clahe_tiles)))
        cth = np.radians(THETA_DEG)
        self._cos, self._sin = np.cos(cth), np.sin(cth)
        self.last_debug: Dict[str, object] = {}

    # ------------------------------------------------------------ setup
    def set_camera_matrix(self, K: Optional[np.ndarray]):
        self.K = None if K is None else np.asarray(K, float)

    @property
    def ocr(self) -> DigitOCR:
        if self._ocr is None:
            self._ocr = DigitOCR()
        return self._ocr

    # ------------------------------------------------------------ 1. enhancement
    def enhance(self, bgr: np.ndarray):
        """Returns dict with 'meas' (float32 gray, linear stretch), 'det' (uint8 gray, gamma +
        CLAHE + denoise), 'bgr' (float32 colour, linear stretch), 'vis' (uint8 enhanced
        colour for overlays) and the applied gain / gamma / noise."""
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        f = bgr.astype(np.float32)
        gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        white = float(np.percentile(gray0, self.cfg.white_percentile))
        gain = float(np.clip(235.0 / max(white, 1.0), 1.0, self.cfg.max_gain))
        lin = np.clip(f * gain, 0, 255)
        meas = cv2.cvtColor(lin, cv2.COLOR_BGR2GRAY)
        mean = float(meas.mean()) / 255.0
        gamma = 1.0
        if 1e-3 < mean < self.cfg.gamma_target:
            gamma = float(np.clip(math.log(self.cfg.gamma_target) / math.log(mean), self.cfg.gamma_min, 1.0))
        lut = np.clip(255.0 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)
        g8 = cv2.LUT(lin.astype(np.uint8), lut)
        lab = cv2.cvtColor(g8, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = self._clahe.apply(L)
        vis = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
        det = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        noise = estimate_noise_sigma(meas)
        if noise > self.cfg.denoise_sigma_thresh:
            s = float(np.clip(0.25 * noise, 0.8, 2.5))
            det = cv2.GaussianBlur(det, (0, 0), s)
            meas = cv2.GaussianBlur(meas, (0, 0), min(1.2, 0.5 * s))
            lin = cv2.GaussianBlur(lin, (0, 0), min(1.2, 0.5 * s))
        return {"meas": meas.astype(np.float32), "det": det, "bgr": lin.astype(np.float32), "vis": vis,
                "gain": gain, "gamma": gamma, "noise": noise}

    # ------------------------------------------------------------ 2. candidates
    def _sample_ellipse(self, img, cx, cy, a, b, ang, rho, t):
        c, s = math.cos(ang), math.sin(ang)
        ex, ey = a * np.cos(t), b * np.sin(t)
        dx, dy = c * ex - s * ey, s * ex + c * ey
        X = (cx + np.outer(dx, rho)).astype(np.float32)
        Y = (cy + np.outer(dy, rho)).astype(np.float32)
        prof = cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        H, W = img.shape[:2]
        inside = (X >= 0) & (X <= W - 1) & (Y >= 0) & (Y <= H - 1)
        return prof.astype(np.float32), X, Y, dx, dy, inside

    def _snap_radius(self, gray: np.ndarray, cand: DialCandidate, rho_lo: float, rho_hi: float):
        """Scale a candidate onto the face/rim edge: strongest bright->dark step of the median
        radial profile (median over 48 rays rejects ticks, needle and partial occluders).
        The step height is stored in ``parts['snap']``: it is largest when the candidate is
        well centred (the edge is at the same rho on every ray)."""
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        rho = np.linspace(0.0, 1.6, 97)
        P, _, _, _, _, inside = self._sample_ellipse(gray, cand.cx, cand.cy, cand.a, cand.b, cand.ang, rho, t)
        if inside.mean() < 0.6:
            return None
        med = np.median(P, axis=0).astype(np.float64)
        dr = rho[1] - rho[0]
        n_in, n_out = 4, 5          # ~0.07 / 0.08 rho windows
        cs = np.concatenate([[0.0], np.cumsum(med)])
        js = np.arange(n_in, len(rho) - n_out)
        resp = (cs[js] - cs[js - n_in]) / n_in - (cs[js + n_out] - cs[js]) / n_out
        resp[(rho[js] < rho_lo) | (rho[js] > rho_hi)] = -np.inf
        if not np.isfinite(resp.max()):
            return None
        i_best = int(np.argmax(resp))
        k = rho[int(js[i_best])] - 0.5 * dr
        return DialCandidate(cand.cx, cand.cy, cand.a * k, cand.b * k, cand.ang, 0.0, cand.source,
                             {"snap": float(resp[i_best])})

    def _recentre(self, gray: np.ndarray, cand: DialCandidate) -> Optional[DialCandidate]:
        """Coarse centre/shape correction of a candidate: per-ray strongest bright->dark step
        in rho = 0.6 .. 1.45 (wide enough for a centre error of ~30 % of the radius, e.g.
        a Hough circle locked onto the tick ring or a pyramid-level quantisation error)
        followed by a robust ellipse fit.  None if the edge points do not form a sane ellipse."""
        n = 64
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        n_rho = int(np.clip(1.0 * cand.a, 40, 160))
        rho = np.linspace(0.5, 1.6, n_rho)
        P, _, _, dx, dy, inside = self._sample_ellipse(gray, cand.cx, cand.cy, cand.a, cand.b, cand.ang, rho, t)
        dr = rho[1] - rho[0]
        n_in = max(2, int(round(0.06 / dr)))
        n_out = max(2, int(round(0.07 / dr)))
        cs = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(P, axis=1)], axis=1)
        js = np.arange(n_in, n_rho - n_out)
        resp = (cs[:, js] - cs[:, js - n_in]) / n_in - (cs[:, js + n_out] - cs[:, js]) / n_out
        resp[:, (rho[js] < 0.6) | (rho[js] > 1.45)] = -1e9
        jj = np.argmax(resp, axis=1)
        strength = resp[np.arange(n), jj]
        r_e = rho[js[jj]]
        ok = inside[np.arange(n), np.clip(js[jj], 0, n_rho - 1)] & (strength > 12.0)
        if np.count_nonzero(ok) < 16:
            return None
        res = robust_ellipse_fit(cand.cx + r_e[ok] * dx[ok], cand.cy + r_e[ok] * dy[ok], strength[ok], iters=4)
        if res is None:
            return None
        (cx, cy, a, b, ang), inl, _rms = res
        if (np.mean(inl) < 0.5 or b < 0.35 * a or not (0.6 * cand.a <= a <= 1.5 * cand.a)
                or math.hypot(cx - cand.cx, cy - cand.cy) > 0.45 * cand.a):
            return None
        return DialCandidate(cx, cy, a, b, ang, 0.0, cand.source + "+rc")

    def _improve(self, gray: np.ndarray, cand: DialCandidate) -> DialCandidate:
        """Up to two rounds of centre/shape correction (``_recentre``) + radius snap for a
        scored candidate that does not close well; returns the best scoring version."""
        best = cur = cand
        for _ in range(2):
            if cur.parts.get("closure", 0.0) > 0.9:
                break
            rc = self._recentre(gray, cur)
            if rc is None:
                break
            rc2 = self._snap_radius(gray, rc, 0.85, 1.15)
            if rc2 is None:
                break
            cur = self._score(gray, rc2)
            if cur.score > best.score:
                best = cur
        return best

    @staticmethod
    def _promise(c: DialCandidate) -> float:
        """Ranking of scored candidates for the (more expensive) centre correction: a bright
        face with a dark surround matters most, closure is what the correction improves."""
        p = c.parts
        if "contrast" not in p:
            return 0.0
        return (float(np.clip((p["contrast"] - 0.15) / 0.45, 0, 1)) * (0.25 + p["closure"])
                * (0.3 + 0.35 * p["tick"] + 0.35 * p["needle"]))

    def _snap_and_score(self, gray: np.ndarray, cand: DialCandidate, rho_lo=0.72, rho_hi=1.38) -> DialCandidate:
        """Snap a candidate to the face/rim edge, score it and correct its centre if needed."""
        snapped = self._snap_radius(gray, cand, rho_lo, rho_hi)
        if snapped is None:
            cand.score = 0.0
            return cand
        return self._improve(gray, self._score(gray, snapped))

    def _score(self, gray, cand: DialCandidate) -> DialCandidate:
        t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        rho = np.linspace(0.0, 1.2, 61)
        P, _, _, _, _, inside = self._sample_ellipse(gray, cand.cx, cand.cy, cand.a, cand.b, cand.ang, rho, t)
        if inside.mean() < 0.6 or cand.b < 0.35 * cand.a:
            cand.score = 0.0
            cand.parts = {}
            return cand

        def sel(r0, r1):
            return (rho >= r0) & (rho <= r1)

        F = float(np.median(P[:, sel(0.5, 0.95)]))
        Rm = float(np.median(P[:, sel(1.02, 1.08)]))
        span = max(F - Rm, 1e-3)
        contrast = float(np.clip(span / max(F, 1.0), 0, 1))
        mid = 0.5 * (F + Rm)
        white_ring = np.median(P[:, sel(0.92, 0.98)], axis=1) > mid
        dark_ring = np.min(P[:, sel(1.0, 1.1)], axis=1) < mid
        closure = float(np.mean(white_ring & dark_ring))
        # tick ring: dark fraction of angles in the tick band (ticks + red band ~ 15-40 %)
        D = (F - P) / span
        tick = np.mean(D[:, sel(0.84, 0.91)], axis=1)
        tick_frac = float(np.mean(tick > 0.25))
        tick_term = float(np.clip(tick_frac / 0.12, 0, 1)) * float(np.clip((0.75 - tick_frac) / 0.25, 0, 1))
        # needle: strongest dark ray in the inner annulus
        needle = np.mean(D[:, sel(0.25, 0.5)], axis=1)
        needle_term = float(np.clip((needle.max() - np.median(needle)) / 0.45, 0, 1))
        score = (np.clip((contrast - 0.15) / 0.45, 0, 1) * closure ** 1.5
                 * (0.2 + 0.4 * tick_term + 0.4 * needle_term))
        cand.score = float(score)
        cand.parts = {"contrast": contrast, "closure": closure, "tick": tick_term, "needle": needle_term,
                      "face": F, "rim": Rm}
        return cand

    def detect_candidates(self, det: np.ndarray, roi=None, r_range=None) -> List[DialCandidate]:
        """Hough (multi-scale) + bright-blob candidates, snapped and scored, best first."""
        H, W = det.shape
        x0, y0, x1, y1 = (0, 0, W, H) if roi is None else (max(0, int(roi[0])), max(0, int(roi[1])),
                                                            min(W, int(roi[2])), min(H, int(roi[3])))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []
        sub = det[y0:y1, x0:x1]
        rmin, rmax = r_range if r_range else (self.cfg.min_radius_px, self.cfg.max_radius_px)
        rmax = min(rmax, 0.6 * max(x1 - x0, y1 - y0))
        rmin = max(4.0, min(rmin, rmax * 0.9))
        cands: List[DialCandidate] = []
        # --- Hough circles on an image pyramid: each level searches radii of ~10..42 px, so the
        #     levels 1, 1/2, 1/4, 1/8 cover face radii of ~10..330 px at a bounded cost
        for f in (1.0, 0.5, 0.25, 0.125):
            lo, hi = max(rmin * f, 9.5), min(rmax * f, 44.0)
            if hi <= lo or (f < 1.0 and rmax * f < 12.0) or rmin * f > 44.0:
                continue
            img = sub if f == 1.0 else cv2.resize(sub, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            if min(img.shape) < 2 * lo:
                continue
            img = cv2.GaussianBlur(img, (0, 0), 1.2)
            try:
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(6.0, lo),
                                           param1=float(self.cfg.hough_canny), param2=float(self.cfg.hough_acc),
                                           minRadius=int(lo), maxRadius=int(math.ceil(hi)))
            except cv2.error:
                circles = None
            if circles is None:
                continue
            for cx, cy, r in np.asarray(circles).reshape(-1, 3)[: self.cfg.max_candidates]:
                cands.append(DialCandidate(x0 + cx / f, y0 + cy / f, r / f, r / f, 0.0, 0.0, f"hough{f}"))
        # --- bright blobs (white faces) -> ellipses (handle strongly oblique dials)
        for blk in (0, 1):
            if blk == 0:
                _, bw = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                bs = int(max(15, 2 * int(rmax) + 1)) | 1
                bw = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, -5)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
            for c in contours:
                if len(c) < 12:
                    continue
                area = cv2.contourArea(c)
                if area < math.pi * (0.6 * rmin) ** 2 or area > math.pi * (1.3 * rmax) ** 2:
                    continue
                hull = cv2.convexHull(c)
                ha = cv2.contourArea(hull)
                if ha <= 0 or area / ha < 0.75:
                    continue
                (ex, ey), (w2, h2), angd = cv2.fitEllipse(c)
                a, b = max(w2, h2) / 2.0, min(w2, h2) / 2.0
                if b < 0.4 * a or a < rmin * 0.7 or a > rmax * 1.3:
                    continue
                if area / (math.pi * a * b) < 0.7:
                    continue
                ang = math.radians(angd) + (0.0 if w2 >= h2 else math.pi / 2)
                cands.append(DialCandidate(x0 + ex, y0 + ey, a, b, ang, 0.0, "blob"))
        # --- snap every raw candidate onto the strongest face/rim edge; concentric Hough circles
        #     (tick ring, face edge, housing) collapse onto the same edge -> drop duplicates
        raw = [self._snap_radius(det, c, 0.72, 1.38) for c in cands]
        raw = [s for s in raw if s is not None and rmin * 0.6 <= s.a <= rmax * 1.4]
        raw.sort(key=lambda s: -s.parts["snap"])          # best centred version of a dial first
        snapped: List[DialCandidate] = []
        for s in raw:
            if all(math.hypot(s.cx - u.cx, s.cy - u.cy) > 0.12 * max(u.a, s.a) or
                   abs(math.log(max(s.a, 1e-3) / max(u.a, 1e-3))) > 0.10 for u in snapped):
                snapped.append(s)
        scored = [self._score(det, s) for s in snapped]
        # --- centre/shape correction for the most promising candidates that do not close well
        #     (Hough centre error, oblique dial detected as a circle)
        order = sorted(range(len(scored)), key=lambda i: -self._promise(scored[i]))
        for i in order[: int(self.cfg.max_refine)]:
            if self._promise(scored[i]) <= 0.0:
                break
            scored[i] = self._improve(det, scored[i])
        scored = [c for c in scored if c.score > 0 and rmin * 0.6 <= c.a <= rmax * 1.4]
        scored.sort(key=lambda c: -c.score)
        keep: List[DialCandidate] = []
        for c in scored:
            if all(math.hypot(c.cx - k.cx, c.cy - k.cy) > 0.5 * max(k.a, c.a) for k in keep):
                keep.append(c)
            if len(keep) >= self.cfg.max_candidates:
                break
        return keep

    # ------------------------------------------------------------ 3. ellipse refinement
    def refine_ellipse(self, gray: np.ndarray, cand: DialCandidate):
        """Ray-wise matched step filter on the face edge + robust ellipse fit (2 rounds).

        Returns (ellipse (cx, cy, a, b, ang), info dict) or (None, info)."""
        ell = (cand.cx, cand.cy, cand.a, cand.b, cand.ang)
        info: Dict[str, float] = {}
        n = int(self.cfg.n_rays)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for it, (lo, hi) in enumerate(((0.80, 1.22), (0.90, 1.10), (0.94, 1.06))):
            cx, cy, a, b, ang = ell
            n_rho = int(np.clip((hi - lo + 0.16) * a * 2.5, 40, 400))
            rho = np.linspace(lo - 0.07, hi + 0.09, n_rho)
            P, X, Y, dx, dy, inside = self._sample_ellipse(gray, cx, cy, a, b, ang, rho, t)
            dr = rho[1] - rho[0]
            n_in = max(2, int(round(0.06 / dr)))
            n_out = max(2, int(round(0.08 / dr)))
            cs = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(P, axis=1)], axis=1)
            js = np.arange(n_in, n_rho - n_out)
            inner = (cs[:, js] - cs[:, js - n_in]) / n_in
            outer = (cs[:, js + n_out] - cs[:, js]) / n_out
            resp = inner - outer
            valid_j = (rho[js] >= lo) & (rho[js] <= hi)
            resp[:, ~valid_j] = -1e9
            jj = np.argmax(resp, axis=1)
            strength = resp[np.arange(n), jj]
            # sub-sample peak (step location = boundary between the windows)
            off = np.array([_parabolic(resp[i, j - 1], resp[i, j], resp[i, j + 1])
                            if 0 < j < resp.shape[1] - 1 else 0.0 for i, j in enumerate(jj)])
            rho_e = rho[js[jj]] + (off - 0.5) * dr
            ex = cx + rho_e * dx
            ey = cy + rho_e * dy
            ok_in = inside[np.arange(n), np.clip(js[jj], 0, n_rho - 1)]
            s_med = float(np.median(strength[ok_in])) if np.any(ok_in) else 0.0
            good = ok_in & (strength > max(6.0, 0.35 * s_med))
            if np.count_nonzero(good) < max(12, n // 4):
                info["reason"] = "few_edges"
                return None, info
            res = robust_ellipse_fit(ex[good], ey[good], strength[good])
            if res is None:
                info["reason"] = "fit_failed"
                return None, info
            params, inl, rms = res
            ell_new = params
            # sanity: the refined ellipse must stay close to the candidate
            if (abs(ell_new[2] / max(a, 1e-3) - 1) > 0.35 or
                    math.hypot(ell_new[0] - cx, ell_new[1] - cy) > 0.4 * a or ell_new[3] < 0.3 * ell_new[2]):
                info["reason"] = "fit_diverged"
                return None, info
            ell = ell_new
            info.update({"rms": rms, "edge_strength": s_med,
                         "edge_frac": float(np.count_nonzero(good)) / n,
                         "inlier_frac": float(np.mean(inl))})
            if it == 2:
                blur, L, beta, s0 = self._edge_blur(P, js[jj], good, dr, dx, dy, n_in, n_out)
                info.update({"blur_px": blur, "psf_len": L, "psf_dir": beta, "psf_sigma": s0})
        if info.get("blur_px", 0.0) > self.cfg.mid_level_blur_px:
            # strongly blurred: the steepest point of the edge ramp is biased by the structure
            # around the rim (ticks, housing, background); the 50 % level crossing of a
            # symmetric blur is not -> refit on mid-level crossings
            ell2, rms2, frac2 = self._refine_mid_level(gray, ell, info["blur_px"])
            if ell2 is not None:
                ell = ell2
                info.update({"rms": rms2, "edge_frac": frac2, "edge_mode": 1.0})
        return ell, info

    def _refine_mid_level(self, gray: np.ndarray, ell, blur_px: float):
        """Face-edge fit on the outermost crossing of the level halfway between the face
        plateau and the dark rim / housing, per ray (unbiased under symmetric blur)."""
        cx, cy, a, b, ang = ell
        n = int(self.cfg.n_rays)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        bl = blur_px / max(b, 1.0)                                  # half blur ~ 0.5 * bl (rho units)
        r_out = 1.0 + 0.5 * bl + 0.12
        rho = np.linspace(0.45, r_out, int(np.clip((r_out - 0.45) * a * 2.0, 40, 500)))
        P, X, Y, dx, dy, inside = self._sample_ellipse(gray, cx, cy, a, b, ang, rho, t)
        face_m = (rho >= 0.5) & (rho <= max(0.6, 0.86 - 0.5 * bl))
        out_m = (rho >= 1.0) & (rho <= r_out)
        hi = np.percentile(P[:, face_m], 75, axis=1)
        lo = np.percentile(P[:, out_m], 25, axis=1)
        mid = 0.5 * (hi + lo)
        ex, ey, wt = [], [], []
        j_start = int(np.searchsorted(rho, 1.0 + 0.5 * bl + 0.04))
        j_stop = int(np.searchsorted(rho, 0.82))
        for i in range(n):
            if hi[i] - lo[i] < 15.0 or not inside[i, min(j_start, len(rho) - 1)]:
                continue
            pr = P[i]
            j = min(j_start, len(rho) - 1)
            while j > j_stop and pr[j] < mid[i]:
                j -= 1
            if j <= j_stop or j + 1 >= len(rho):
                continue
            fr = (pr[j] - mid[i]) / max(pr[j] - pr[j + 1], 1e-6)
            r_e = rho[j] + fr * (rho[j + 1] - rho[j])
            ex.append(cx + r_e * dx[i])
            ey.append(cy + r_e * dy[i])
            wt.append(hi[i] - lo[i])
        if len(ex) < max(12, n // 3):
            return None, 0.0, 0.0
        res = robust_ellipse_fit(np.array(ex), np.array(ey), np.array(wt))
        if res is None:
            return None, 0.0, 0.0
        params, inl, rms = res
        if (abs(params[2] / a - 1) > 0.15 or math.hypot(params[0] - cx, params[1] - cy) > 0.15 * a
                or params[3] < 0.3 * params[2]):
            return None, 0.0, 0.0
        return params, rms, len(ex) / float(n)

    @staticmethod
    def _edge_blur(P, j_edge, good, dr, dx, dy, n_in, n_out):
        """Blur estimate from the 20-80 % width of the face/rim edge along the rays.

        A box (motion) blur of length L widens a step edge to a 20-80 % width of 0.6 L along
        the blur direction; the anti-aliasing / optics floor adds in quadrature.  Fitting
            w(psi)^2 = w0^2 + (0.6 L cos(psi - beta))^2 = c0 + c1 cos 2psi + c2 sin 2psi
        over the ray directions psi gives the motion-blur length L and direction beta.
        Returns (blur_px, L, beta [rad], sigma0 [px])."""
        widths, psis = [], []
        for i in np.where(good)[0]:
            j0 = int(j_edge[i])
            prof = P[i]
            ppr = float(math.hypot(dx[i], dy[i]))
            n_px = int(math.ceil(10.0 / max(dr * ppr, 1e-6)))   # >= 10 px each side
            a0, b0 = max(0, j0 - max(n_in + 3, n_px)), min(len(prof), j0 + max(n_out + 3, n_px))
            hi = float(prof[a0:j0].max()) if j0 > a0 else float(prof[j0])
            lo = float(prof[j0:b0].min()) if b0 > j0 else float(prof[j0])
            if hi - lo < 8.0:
                continue
            l80, l20 = lo + 0.8 * (hi - lo), lo + 0.2 * (hi - lo)
            k = j0
            while k > a0 and prof[k] < l80:
                k -= 1
            p80 = k + (prof[k] - l80) / max(prof[k] - prof[k + 1], 1e-6) if k + 1 < len(prof) else k
            k = j0 - 1
            while k < b0 - 1 and prof[k] > l20:
                k += 1
            p20 = k - 1 + (prof[k - 1] - l20) / max(prof[k - 1] - prof[k], 1e-6) if k > 0 else k
            widths.append(max(0.0, p20 - p80) * dr * ppr)
            psis.append(math.atan2(dy[i], dx[i]))
        if len(widths) < 8:
            return 0.0, 0.0, 0.0, 0.7
        w = np.asarray(widths)
        psi = np.asarray(psis)
        w90 = float(np.percentile(w, 90))
        blur = float(math.sqrt(max(w90 ** 2 - 1.1 ** 2, 0.0)) / 0.6)
        # anisotropy fit (robust: clip extreme widths)
        w2 = np.minimum(w, np.percentile(w, 95)) ** 2
        A = np.stack([np.ones_like(psi), np.cos(2 * psi), np.sin(2 * psi)], 1)
        try:
            c0, c1, c2 = np.linalg.lstsq(A, w2, rcond=None)[0]
        except np.linalg.LinAlgError:
            return blur, 0.0, 0.0, 0.7
        amp = math.hypot(c1, c2)
        L = math.sqrt(max(amp / 0.18, 0.0))
        beta = 0.5 * math.atan2(c2, c1)
        w0sq = max(c0 - amp, 0.0)
        sigma0 = float(np.clip(math.sqrt(w0sq) / 1.68, 0.5, 4.0))
        return blur, float(L), float(beta), sigma0

    # ------------------------------------------------------------ 4. rectification
    def rectify(self, ell, meas: np.ndarray, hint: Optional[GaugeHint] = None):
        """Homography canonical dial frame -> image from the fitted face ellipse."""
        cx, cy, a, b, ang = ell
        Hs = []
        if self.K is not None:
            try:
                sols = circle_poses_from_conic(ellipse_conic(cx, cy, a, b, ang), self.K)
            except np.linalg.LinAlgError:
                sols = []
            for n_cam, Xc, rc in sols:
                Hs.append((homography_from_circle_pose(self.K, n_cam, Xc, rc), n_cam))
        if not Hs:
            return homography_affine(cx, cy, a, b, ang), "affine"
        if len(Hs) == 1:
            return Hs[0][0], "conic"
        (H1, n1), (H2, n2) = Hs
        sep = math.degrees(math.acos(float(np.clip(n1 @ n2, -1, 1))))
        if hint is not None and hint.normal_cam is not None and sep > 4.0:
            nh = np.asarray(hint.normal_cam, float)
            nh = nh / max(1e-9, np.linalg.norm(nh))
            return (H1, "conic_prior") if n1 @ nh >= n2 @ nh else (H2, "conic_prior")
        # hub test: the hub disc (r < 0.035, drawn over the needle) is uniform and centred on
        # the true dial centre; the wrong solution's centre falls off the hub or onto the needle
        s1, s2 = self._hub_score(meas, H1), self._hub_score(meas, H2)
        return (H1, "conic_hub") if s1 >= s2 else (H2, "conic_hub")

    @staticmethod
    def _ring_samples(img, H, r0, r1, n_theta=36, n_r=3) -> np.ndarray:
        th = np.radians(np.arange(n_theta) * (360.0 / n_theta))
        rr = np.linspace(r0, r1, n_r)
        pts = np.stack([np.outer(rr, np.cos(th)).ravel(), np.outer(rr, np.sin(th)).ravel()], 1)
        uv = apply_h(H, pts).astype(np.float32)
        return cv2.remap(img, np.ascontiguousarray(uv[:, :1]), np.ascontiguousarray(uv[:, 1:]),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()

    def _hub_score(self, meas, H) -> float:
        """Contrast of a uniform dark hub ring against the face just outside it."""
        hub = self._ring_samples(meas, H, 0.013, 0.028)
        out = self._ring_samples(meas, H, 0.05, 0.065)
        face = float(np.median(out))
        return (face - float(np.median(hub))) / (float(np.std(hub)) + 0.05 * abs(face) + 2.0)

    # ------------------------------------------------------------ polar sampling
    def _polar(self, img: np.ndarray, H: np.ndarray, radii: Sequence[float]) -> np.ndarray:
        rr = np.asarray(radii, float)
        pts = np.stack([np.outer(rr, self._cos).ravel(), np.outer(rr, self._sin).ravel()], 1)
        uv = apply_h(H, pts)
        X = uv[:, 0].reshape(len(rr), N_THETA).astype(np.float32)
        Y = uv[:, 1].reshape(len(rr), N_THETA).astype(np.float32)
        return cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------ 5. calibration
    def calibrate(self, D: np.ndarray, red: np.ndarray, vrange: Optional[Tuple[float, float]],
                  res_deg: float, prior_rot: Optional[float],
                  valid: Optional[np.ndarray] = None) -> Optional[Calibration]:
        """Rotation / mirror / divisions from the tick-ring and danger-band profiles."""
        rr = POLAR_R

        def band(arr, r0, r1):
            m = (rr >= r0 - 1e-9) & (rr <= r1 + 1e-9)
            return arr[m].mean(axis=0).astype(np.float64)

        M_out = band(D, *TICK_OUTER_BAND)
        M_in = band(D, *TICK_INNER_BAND)
        M_red = band(red, *RED_BAND)
        red_ok = float(np.max(_gauss_circ(M_red, 4.0)) - np.median(M_red)) > 0.04
        w_in = self.cfg.inner_band_weight
        w_red = self.cfg.red_weight if red_ok else 0.0
        sig = max(0.0, res_deg / 0.5 * 0.6)      # template blur in bins ~ measured resolution
        if vrange is not None:
            tmpl = [(n_major_divisions(*vrange), tuple(vrange))]
        else:
            tmpl = [(n, GENERIC_RANGES[n]) for n in sorted(GENERIC_RANGES)]
        idx = np.arange(N_THETA)
        results = []
        for n_div, rng_t in tmpl:
            T_out, T_in, T_red = self.templates.profiles(*rng_t)
            T_out, T_in, T_red = (_gauss_circ(T_out, sig), _gauss_circ(T_in, sig), _gauss_circ(T_red, sig + 1.0))
            for mirrored in ((False, True) if self.cfg.allow_mirror else (False,)):
                if mirrored:
                    to, ti, tr = T_out[(-idx) % N_THETA], T_in[(-idx) % N_THETA], T_red[(-idx) % N_THETA]
                else:
                    to, ti, tr = T_out, T_in, T_red
                if valid is not None and not np.all(valid):
                    wv = valid.astype(np.float64)
                    c = (_masked_circ_ncc(M_out, to, wv) + w_in * _masked_circ_ncc(M_in, ti, wv)
                         + w_red * _masked_circ_ncc(M_red, tr, wv)) / (1.0 + w_in + w_red)
                else:
                    c = (_circ_ncc(M_out, to) + w_in * _circ_ncc(M_in, ti) + w_red * _circ_ncc(M_red, tr)) / (1.0 + w_in + w_red)
                if prior_rot is not None:
                    c = c + 0.02 * np.cos(np.radians(THETA_DEG - prior_rot))
                results.append((c, n_div, rng_t, mirrored))
        best = max(results, key=lambda r: float(r[0].max()))
        c, n_div, rng_t, mirrored = best
        k = int(np.argmax(c))
        peak = float(c[k])
        off = _parabolic(c[(k - 1) % N_THETA], c[k], c[(k + 1) % N_THETA])
        rot = (k + off) * (360.0 / N_THETA)
        # margin: best alternative (other mirror / other division count, or a rotation > 12 deg away)
        alt = []
        for c2, n2, _r2, m2 in results:
            if m2 != mirrored or n2 != n_div:
                alt.append(float(c2.max()))
        dist = np.abs(((THETA_DEG - rot) + 180.0) % 360.0 - 180.0)
        far = c[dist > 12.0]
        if far.size:
            alt.append(float(far.max()))
        margin = peak - (max(alt) if alt else 0.0)
        if vrange is None:
            # the division count only matters for OCR: keep the mirror/rotation margin
            alt_m = [float(c2.max()) for c2, n2, _r2, m2 in results if m2 != mirrored]
            if far.size:
                alt_m.append(float(far.max()))
            margin = peak - (max(alt_m) if alt_m else 0.0)
        by_div: Dict[int, float] = {}
        for c2, n2, _r2, _m2 in results:
            by_div[n2] = max(by_div.get(n2, -1.0), float(c2.max()))
        return Calibration(float(rot % 360.0), bool(mirrored), int(n_div), tuple(vrange) if vrange else None,
                           peak, float(margin), by_div)

    # ------------------------------------------------------------ 6. needle
    def _needle(self, meas: np.ndarray, H: np.ndarray, D: np.ndarray, cal: Calibration,
                face_level: np.ndarray, ink_level: float, px_per_unit: float, blur_px: float = 0.0,
                valid: Optional[np.ndarray] = None):
        rr = POLAR_R
        # blur smears the hub/tail outwards and the labels inwards: shrink the annulus
        blur_u = 0.5 * blur_px / max(px_per_unit, 1e-6)          # half blur length, dial units
        r_lo = max(NEEDLE_COARSE_BAND[0], 0.08 + blur_u + 0.01)
        r_hi = max(r_lo + 0.05, NEEDLE_COARSE_BAND[1] - blur_u)
        m = (rr >= r_lo) & (rr <= r_hi)
        prof = D[m].mean(axis=0).astype(np.float64)
        prof_s = _gauss_circ(prof, 2.0)
        s = -1.0 if cal.mirrored else 1.0
        phi_d = (s * (THETA_DEG - cal.rotation_deg) + 360.0) % 360.0          # 0..360
        phi_d = np.where(phi_d > 270.0, phi_d - 360.0, phi_d)                  # (-90, 270]
        on_arc = (phi_d >= gd.PHI_MAX_DEG - 6.0) & (phi_d <= gd.PHI_MIN_DEG + 6.0)
        if valid is not None and np.count_nonzero(on_arc & valid) > 20:
            on_arc = on_arc & valid
        cand = np.where(on_arc, prof_s, -np.inf)
        k0 = int(np.argmax(cand))
        theta0 = THETA_DEG[k0]
        peak = float(prof_s[k0])
        base = float(np.median(prof_s[on_arc]))
        contrast = peak - base
        # second strongest peak away from the first (ambiguity, e.g. occlusion / shadow)
        dist = np.abs(((THETA_DEG - theta0) + 180.0) % 360.0 - 180.0)
        other = np.where(on_arc & (dist > 15.0), prof_s, -np.inf)
        second = float(other.max()) - base if np.isfinite(other.max()) else 0.0

        # ---- fine: per-radius darkness centroids -> robust line fit
        pts, wts = [], []
        f_ref = float(np.interp(theta0, THETA_DEG, face_level, period=360.0))
        span = max(f_ref - ink_level, 1e-3)
        px_deg = 180.0 / math.pi / max(px_per_unit * 0.1, 1e-3)               # deg per px at r=0.1
        r_min_fine = max(NEEDLE_FINE_RADII[0], 0.035 + blur_u + 0.012)
        for r in NEEDLE_FINE_RADII:
            if r < r_min_fine or (blur_u > 0.01 and 0.225 - blur_u < r < 0.305):
                continue
            hw_deg = math.degrees(needle_half_width(r) / r)
            res_deg = px_deg * 0.1 / r
            win = hw_deg + max(3.0, 2.5 * res_deg) + math.degrees(1.2 * blur_u / r)
            step = max(0.1, min(0.5, win / 25.0, res_deg / 3.0))
            th = np.arange(theta0 - win, theta0 + win + 1e-9, step)
            ang = np.radians(th)
            uv = apply_h(H, np.stack([r * np.cos(ang), r * np.sin(ang)], 1)).astype(np.float32)
            v = cv2.remap(meas, uv[:, 0:1], uv[:, 1:2], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
            d = (f_ref - v) / span
            nb = max(2, len(d) // 6)
            base_r = float(np.median(np.r_[d[:nb], d[-nb:]]))
            d = d - base_r
            j = int(np.argmax(d))
            dmax = float(d[j])
            if dmax < 0.15:
                continue
            lo = j
            while lo > 0 and d[lo - 1] > 0.25 * dmax:
                lo -= 1
            hi = j
            while hi < len(d) - 1 and d[hi + 1] > 0.25 * dmax:
                hi += 1
            if lo == 0 or hi == len(d) - 1:
                # peak region touches the window border: something else is dark (label, occluder)
                continue
            w = np.clip(d[lo:hi + 1], 0, None)
            if w.sum() <= 0:
                continue
            th_c = float(np.sum(w * th[lo:hi + 1]) / w.sum())
            a = math.radians(th_c)
            pts.append((r * math.cos(a), r * math.sin(a)))
            wts.append(min(1.0, dmax) * (0.5 + r / 0.33))
        info = {"needle_contrast": contrast, "needle_second": second, "n_points": len(pts)}
        if len(pts) < 5:
            return float(theta0), info, False
        P = np.asarray(pts)
        w = np.asarray(wts)
        # soft constraint: the needle pivots at the dial centre
        P = np.vstack([P, [0.0, 0.0]])
        w = np.r_[w, 2.0 * np.median(w)]
        u0 = np.array([math.cos(math.radians(theta0)), math.sin(math.radians(theta0))])
        base_w = w.copy()
        direction = u0
        for _ in range(4):
            mu = np.average(P, axis=0, weights=w)
            Cm = np.cov((P - mu).T, aweights=w, bias=True)
            evals, evecs = np.linalg.eigh(Cm)
            direction = evecs[:, -1]
            if direction @ u0 < 0:
                direction = -direction
            nrm = np.array([-direction[1], direction[0]])
            res = (P - mu) @ nrm
            sres = 1.4826 * float(np.median(np.abs(res))) + 0.002
            uu = res / (4.685 * sres)
            w = base_w * np.where(np.abs(uu) < 1, (1 - uu * uu) ** 2, 0.0)
            if np.count_nonzero(w[:-1]) < 4:
                break
        theta_fine = math.degrees(math.atan2(direction[1], direction[0]))
        # guard: the fine angle must agree with the coarse peak
        dd = abs(((theta_fine - theta0) + 180.0) % 360.0 - 180.0)
        info["fine_minus_coarse"] = dd
        info["line_rms"] = float(np.sqrt(np.average(res ** 2, weights=np.maximum(w, 1e-9))))
        if dd > 6.0:
            return float(theta0), info, False
        return float(theta_fine % 360.0), info, True

    def _needle_model_fit(self, meas: np.ndarray, H: np.ndarray, cal: Calibration, theta0: float,
                          finfo: dict, px_per_unit: float, valid: Optional[np.ndarray],
                          search_deg: float = 12.0) -> Optional[dict]:
        """Analysis-by-synthesis needle angle for blurred dials.

        The dial without needle (ticks, danger band, hub) is rendered from gauge_design with the
        calibrated rotation/mirror, warped into the image and blurred with the estimated PSF
        (motion blur of length L along beta + Gaussian); for each needle angle the blurred
        needle mask B is rendered the same way and the observation is fitted linearly,
            O ~ a + b * S - c * B(theta),
        over the face pixels excluding the labels, the unit/tag gap and occluded sectors.  The
        angle minimising the residual (coarse 1 deg grid, then parabolic refinement on 0.5 and
        0.25 deg) is returned with the relative residual reduction as needle evidence."""
        Hc = H @ dial_to_canonical(cal.rotation_deg, cal.mirrored)        # printed dial -> image
        # working scale: keep the dial radius <= ~85 px
        a_px = R_FACE * px_per_unit
        f = min(1.0, 85.0 / max(a_px, 1.0))
        Sm = np.diag([f, f, 1.0])
        Hw = Sm @ Hc
        ring = [[0.45 * math.cos(t), 0.45 * math.sin(t)] for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
        corners = apply_h(Hw, ring)
        L = finfo.get("psf_len", 0.0) * f
        s0 = max(0.5, finfo.get("psf_sigma", 0.7) * f)
        pad = int(math.ceil(L / 2 + 3 * s0 + 2))
        x0, y0 = (np.floor(corners.min(axis=0)).astype(int) - pad).tolist()
        x1, y1 = (np.ceil(corners.max(axis=0)).astype(int) + pad).tolist()
        W, Hh = int(x1 - x0), int(y1 - y0)
        if W < 16 or Hh < 16 or W > 700 or Hh > 700:
            return None
        T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
        Hroi = T @ Hw                                                     # printed dial -> ROI px
        Hobs = np.linalg.inv(T @ Sm)                                      # ROI px -> image px
        obs = cv2.warpPerspective(meas, Hobs, (W, Hh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
        # PSF = motion blur (estimated length / direction) * Gaussian (optics + AA)
        ker = _motion_kernel(L, finfo.get("psf_dir", 0.0)) if L > 1.5 else np.ones((1, 1), np.float32)
        g = max(1, int(math.ceil(3 * s0)))
        gk = cv2.getGaussianKernel(2 * g + 1, s0)
        ker = cv2.filter2D(np.pad(ker, g), -1, (gk @ gk.T).astype(np.float32), borderType=cv2.BORDER_CONSTANT)
        ker = (ker / ker.sum()).astype(np.float32)

        def blur(img):
            return cv2.filter2D(img.astype(np.float32), -1, ker, borderType=cv2.BORDER_REPLICATE).astype(np.float64)

        # static dial (no needle) + hub, rendered at ~2x the working resolution
        vr = cal.vrange if cal.vrange is not None else GENERIC_RANGES[cal.n_div]
        Ts = int(min(1024, 2 ** math.ceil(math.log2(max(64.0, 2.0 * f * px_per_unit)))))
        stat = self._static_face(vr, Ts)
        A_t = np.array([[1.0 / Ts, 0.0, -0.5], [0.0, -1.0 / Ts, 0.5], [0.0, 0.0, 1.0]])
        S = blur(cv2.warpPerspective(stat, Hroi @ A_t, (W, Hh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=float(np.mean(gd.COLORS["rim"]))))
        # dial coordinates of every ROI pixel -> fit weights
        yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float64)
        dpts = apply_h(np.linalg.inv(Hroi), np.stack([xx.ravel(), yy.ravel()], 1))
        r = np.hypot(dpts[:, 0], dpts[:, 1]).reshape(Hh, W)
        ph = np.degrees(np.arctan2(dpts[:, 1], dpts[:, 0])).reshape(Hh, W)
        ph = np.where(ph < -90.0, ph + 360.0, ph)                         # (-90, 270]
        in_gap = (ph > gd.PHI_MIN_DEG + 6.0) & (ph < 360.0 + gd.PHI_MAX_DEG - 6.0)
        Wm = (r < 0.425) & ~((r > 0.228) & (r < 0.312)) & ~(in_gap & (r > 0.09))
        if valid is not None and not np.all(valid):
            s_ = -1.0 if cal.mirrored else 1.0
            th_c = (cal.rotation_deg + s_ * ph) % 360.0
            Wm &= valid[np.clip((th_c / (360.0 / N_THETA)).astype(int), 0, N_THETA - 1)]
        idx = np.flatnonzero(Wm.ravel())
        if idx.size < 200:
            return None
        o = obs.ravel()[idx]
        sv = S.ravel()[idx]
        hub = apply_h(Hroi, [[gd.R_HUB * math.cos(t), gd.R_HUB * math.sin(t)]
                             for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
        hub_i = np.round(hub * 16).astype(np.int32)
        sgn = -1.0 if cal.mirrored else 1.0

        def needle_mask(theta_c_deg: float) -> np.ndarray:
            phi = math.radians(sgn * (theta_c_deg - cal.rotation_deg))
            d = np.array([math.cos(phi), math.sin(phi)])
            n = np.array([-d[1], d[0]])
            tip, tail = d * gd.R_NEEDLE, -d * 0.08
            poly = np.array([tail + n * 0.022, tip + n * 0.006, tip - n * 0.006, tail - n * 0.022])
            q = apply_h(Hroi, poly)
            m = np.zeros((Hh, W), np.uint8)
            cv2.fillPoly(m, [np.round(q * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            cv2.fillPoly(m, [hub_i], 0, cv2.LINE_AA, shift=4)            # hub is drawn over the needle
            return blur(m.astype(np.float32) / 255.0)

        X0 = np.stack([np.ones_like(sv), sv], 1)
        beta0 = np.linalg.lstsq(X0, o, rcond=None)[0]
        rss0 = float(np.sum((o - X0 @ beta0) ** 2))
        cache: Dict[float, Tuple[float, float]] = {}

        def rss(theta: float) -> float:
            key = round(theta, 4)
            if key not in cache:
                bv = needle_mask(theta).ravel()[idx]
                X = np.stack([np.ones_like(sv), sv, -bv], 1)
                coef = np.linalg.lstsq(X, o, rcond=None)[0]
                cache[key] = (float(np.sum((o - X @ coef) ** 2)), float(coef[2]))
            return cache[key][0]

        grid = np.arange(theta0 - search_deg, theta0 + search_deg + 1e-9, 1.0)
        vals = np.array([rss(t) for t in grid])
        k = int(np.argmin(vals))
        t_best = float(grid[k])
        for step in (0.5, 0.25):
            vm, v0, vp = rss(t_best - step), rss(t_best), rss(t_best + step)
            t_best += _parabolic(vm, v0, vp) * step
        v_best = rss(t_best)
        return {"theta": t_best % 360.0, "evidence": float((rss0 - v_best) / max(rss0, 1e-9)),
                "needle_coef": cache[round(t_best, 4)][1], "at_edge": bool(k in (0, len(grid) - 1)),
                "scale": f}

    def _static_face(self, vrange, size: int) -> np.ndarray:
        """Gray dial face without needle but with the hub (cached)."""
        key = (round(float(vrange[0]), 9), round(float(vrange[1]), 9), int(size))
        cache = self.__dict__.setdefault("_static_cache", {})
        if key not in cache:
            img = gd.render_gauge_face(vrange[0], vrange[0], vrange[1], unit="", tag="", size=size, draw_needle=False)
            c = (size // 2, size // 2)
            cv2.circle(img, c, int(gd.R_HUB * size), gd.COLORS["hub"], -1, cv2.LINE_AA)
            cv2.circle(img, c, max(1, int(0.01 * size)), (160, 160, 160), -1, cv2.LINE_AA)
            if len(cache) > 32:
                cache.clear()
            cache[key] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cache[key]

    # ------------------------------------------------------------ 7. OCR
    def label_patch(self, meas: np.ndarray, H: np.ndarray, cal: Calibration, phi_deg: float) -> np.ndarray:
        """Upright patch around the scale label at dial angle ``phi_deg``."""
        W = int(round(LABEL_PATCH_W * LABEL_PPU))
        Hh = int(round(LABEL_PATCH_H * LABEL_PPU))
        a = math.radians(phi_deg)
        px, py = gd.R_LABEL * math.cos(a), gd.R_LABEL * math.sin(a)
        A = np.array([[1.0 / LABEL_PPU, 0.0, px - W / (2.0 * LABEL_PPU)],
                      [0.0, -1.0 / LABEL_PPU, py + Hh / (2.0 * LABEL_PPU)],
                      [0.0, 0.0, 1.0]])
        Hp = H @ dial_to_canonical(cal.rotation_deg, cal.mirrored) @ A
        return cv2.warpPerspective(meas, Hp, (W, Hh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE)

    def read_range(self, meas: np.ndarray, H: np.ndarray, cal: Calibration, label_px: float,
                   n_div_candidates: Sequence[int], blur_sigma: float = 0.0,
                   blur_px: float = 0.0) -> Optional[RangeEstimate]:
        """OCR of the scale labels at every major tick -> (vmin, vmax).

        For every candidate division count the label patches are (1) read character by
        character and fitted with ``infer_range`` and (2) scored against the standard ranges
        by analysis-by-synthesis.  The hypothesis margin is taken over *all* division counts.
        Agreement of both paths raises the confidence, disagreement halves it."""
        if blur_px > max(3.0, 0.8 * label_px):
            return None                            # labels smeared beyond recognition
        exp_h = LABEL_HEIGHT_DIAL * LABEL_PPU
        char_best: Optional[RangeEstimate] = None
        hyps: List[Tuple[float, float, float, int]] = []
        for n_div in n_div_candidates:
            angles = major_label_angles(n_div)
            patches = {k: self.label_patch(meas, H, cal, phi) for k, phi in enumerate(angles)}
            if label_px >= self.cfg.ocr_min_label_px:
                reads = [(k, self.ocr.read(p, exp_h, blur_sigma)) for k, p in patches.items()]
                est = infer_range(reads, n_div)
                if est is not None and (char_best is None or est.confidence > char_best.confidence):
                    char_best = est
            if label_px >= self.cfg.ocr_min_label_px_hyp:
                hyps += [(sc, v0, v1, n_div) for sc, v0, v1 in score_ranges(self.ocr, patches, n_div, exp_h,
                                                                           blur_sigma=blur_sigma)]
        hyp_best: Optional[RangeEstimate] = None
        if hyps:
            hyps.sort(key=lambda t: -t[0])
            s1, v0, v1, nd = hyps[0]
            s2 = next((h[0] for h in hyps[1:] if (h[1], h[2]) != (v0, v1)), -1.0)
            hyp_best = RangeEstimate(v0, v1, hypothesis_confidence(s1, s2), "hypothesis", nd, [])
        if char_best is None:
            return hyp_best
        if hyp_best is None:
            return char_best
        same = abs(char_best.vmin - hyp_best.vmin) < 1e-6 and abs(char_best.vmax - hyp_best.vmax) < 1e-6
        if same:
            char_best.confidence = float(min(1.0, 0.5 + 0.5 * max(char_best.confidence, hyp_best.confidence)))
            char_best.method = "chars+hypothesis"
            return char_best
        best = char_best if char_best.confidence >= hyp_best.confidence else hyp_best
        best.confidence *= 0.5
        return best

    # ------------------------------------------------------------ main entry points
    def read(self, bgr: np.ndarray, hints: Optional[Sequence[GaugeHint]] = None,
             detect_unhinted: bool = True, enhanced: Optional[dict] = None) -> List[GaugeReading]:
        """Read every gauge in the frame.

        hints: gauges expected in view (predicted centre/radius/range/normal).  Each hint is
        searched in its ROI; with ``detect_unhinted`` (or without hints) the whole frame is
        searched for further dials (id = None)."""
        enh = enhanced or self.enhance(bgr)
        self.last_debug = {"enhance": {k: enh[k] for k in ("gain", "gamma", "noise")}, "rejected": []}
        det = enh["det"]
        if self.cfg.min_frame_sharpness > 0:
            sh = frame_sharpness(enh["meas"])
            self.last_debug["sharpness"] = sh
            if sh < self.cfg.min_frame_sharpness:
                self.last_debug["frame_rejected"] = "blurred"
                return []
        readings: List[GaugeReading] = []
        used: List[Tuple[float, float, float]] = []
        for hint in hints or []:
            if hint.center is None or hint.radius is None:
                continue
            r_p = float(hint.radius)
            half = hint.search_radius or (self.cfg.roi_scale * r_p + self.cfg.roi_margin_px)
            cx, cy = hint.center
            roi = (cx - half, cy - half, cx + half, cy + half)
            cands = self.detect_candidates(det, roi, (max(self.cfg.min_radius_px * 0.8, 0.5 * r_p), 1.9 * r_p))

            def prior(c: DialCandidate) -> float:
                dpos = math.hypot(c.cx - cx, c.cy - cy) / (r_p + 0.5 * self.cfg.roi_margin_px)
                drad = math.log(max(c.a, 1e-3) / r_p) / 0.4
                return c.score * math.exp(-0.5 * dpos * dpos) * math.exp(-0.5 * drad * drad)

            cands = [c for c in cands if c.score >= self.cfg.min_detection_score]
            cands.sort(key=lambda c: -prior(c))
            for c in cands[:3]:
                rd = self._measure(enh, c, hint)
                if rd is not None and rd.accepted:
                    rd.gauge_id = hint.gauge_id
                    readings.append(rd)
                    used.append((rd.ellipse[0], rd.ellipse[1], rd.ellipse[2]))
                    break
                if rd is not None:
                    self.last_debug["rejected"].append((hint.gauge_id, rd.reject_reason))
        if detect_unhinted or not hints:
            cands = self.detect_candidates(det)
            n_new = 0
            for c in cands:
                if c.score < self.cfg.min_detection_score or n_new >= self.cfg.max_dials:
                    continue
                if any(math.hypot(c.cx - u, c.cy - v) < 0.7 * max(r, c.a) for u, v, r in used):
                    continue
                rd = self._measure(enh, c, None)
                if rd is None or not rd.accepted:
                    if rd is not None:
                        self.last_debug["rejected"].append((None, rd.reject_reason))
                    continue
                readings.append(rd)
                used.append((rd.ellipse[0], rd.ellipse[1], rd.ellipse[2]))
                n_new += 1
        return readings

    def read_one(self, bgr: np.ndarray, vrange: Optional[Tuple[float, float]] = None,
                 hint: Optional[GaugeHint] = None) -> Optional[GaugeReading]:
        """Convenience: best single reading in the frame (optionally with a known range)."""
        if hint is None:
            hint = GaugeHint(vrange=vrange)
        elif vrange is not None:
            hint.vrange = vrange
        enh = self.enhance(bgr)
        if hint.center is not None and hint.radius is not None:
            res = self.read(bgr, [hint], detect_unhinted=False, enhanced=enh)
            if res:
                return res[0]
        cands = [c for c in self.detect_candidates(enh["det"]) if c.score >= self.cfg.min_detection_score]
        best = None
        self.last_debug = {"enhance": {k: enh[k] for k in ("gain", "gamma", "noise")}, "rejected": []}
        for c in cands[:4]:
            rd = self._measure(enh, c, hint)
            if rd is not None and not rd.accepted:
                self.last_debug["rejected"].append((rd.reject_reason, round(rd.confidence, 3),
                                                    {k: round(v, 3) for k, v in rd.scores.items()}))
            if rd is not None and rd.accepted and (best is None or rd.confidence > best.confidence):
                best = rd
                if rd.confidence > 0.6:
                    break
        return best

    # ------------------------------------------------------------ measurement of one dial
    def _measure(self, enh: dict, cand: DialCandidate, hint: Optional[GaugeHint]) -> Optional[GaugeReading]:
        meas, det, bgr = enh["meas"], enh["det"], enh["bgr"]
        ell, finfo = self.refine_ellipse(det, cand)
        if ell is None:
            return None
        cx, cy, a, b, ang = ell
        blur_px = float(finfo.get("blur_px", 0.0))
        H, rect_mode = self.rectify(ell, meas, hint)
        px_per_unit = math.sqrt(a * b) / R_FACE           # image px per dial unit (geometric mean)
        # effective angular resolution at the tick ring: pixel footprint (+AA) and blur spread
        sigma_px = math.sqrt(0.5 ** 2 + (blur_px / math.sqrt(12.0)) ** 2)
        res_deg = min(8.0, 2.0 * sigma_px * math.degrees(1.0 / max(px_per_unit * 0.385, 1e-3)))

        # ---- polar unwrap + photometric normalisation
        G = self._polar(meas, H, POLAR_R)                                    # (n_r, 720)
        Cb = self._polar(bgr, H, POLAR_R)                                    # (n_r, 720, 3)
        ref_rows = np.zeros(len(POLAR_R), bool)
        for r0, r1 in FACE_REF_BANDS:
            ref_rows |= (POLAR_R >= r0) & (POLAR_R <= r1)
        face_raw = np.percentile(G[ref_rows], 75, axis=0)
        # occluded sectors: the white reference annuli are not white there
        face_med = float(np.median(face_raw))
        valid = face_raw > 0.6 * face_med
        valid = np.convolve(np.r_[valid[-6:], valid, valid[:6]].astype(float), np.ones(13) / 13.0, "valid") > 0.99
        # remove the needle / occluders from the face level: circular median over +-10 deg
        k = 20
        fr = np.where(valid, face_raw, face_med)
        ext = np.r_[fr[-k:], fr, fr[:k]]
        win = np.lib.stride_tricks.sliding_window_view(ext, 2 * k + 1)
        face = _gauss_circ(np.median(win, axis=1)[:N_THETA], 6.0)
        tick_rows = (POLAR_R >= 0.33) & (POLAR_R <= 0.405)
        ink = float(np.percentile(G[tick_rows], 1.0))
        span = np.maximum(face - ink, 1.0)
        D = (face[None, :] - G) / span[None, :]
        red = np.clip(Cb[..., 2] - 0.5 * (Cb[..., 0] + Cb[..., 1]), 0, None) / np.maximum(face, 1.0)[None, :]
        face_contrast = float(np.median(span) / max(float(np.median(face)), 1.0))

        prior_rot = 0.0 if rect_mode.startswith("conic") else None
        vrange = tuple(hint.vrange) if (hint is not None and hint.vrange is not None) else None
        cal = self.calibrate(D, red, vrange, res_deg, prior_rot, valid)

        def solve_needle(cal_: Calibration):
            th, ni, ok = self._needle(meas, H, D, cal_, face, ink, px_per_unit, blur_px, valid)
            mode = self.cfg.needle_model
            use_model = mode == "always" or (mode == "auto" and (blur_px > self.cfg.needle_model_blur_px or not ok))
            ni["method"] = "centroid" if ok else "coarse"
            if use_model:
                mf = self._needle_model_fit(meas, H, cal_, th, finfo, px_per_unit, valid,
                                            search_deg=12.0 if ok else 20.0)
                if mf is not None and mf["needle_coef"] > 0 and not mf["at_edge"]:
                    ni.update({"model_evidence": mf["evidence"], "model_shift": abs(((mf["theta"] - th) + 180) % 360 - 180)})
                    th, ok = mf["theta"], True
                    ni["method"] = "model"
            sgn = -1.0 if cal_.mirrored else 1.0
            ph = (sgn * (th - cal_.rotation_deg) + 360.0) % 360.0
            if ph > 270.0:
                ph -= 360.0
            return th, ph, (gd.PHI_MIN_DEG - ph) / gd.SWEEP_DEG, ni, ok

        theta, phi_d, fraction, ninfo, fine_ok = solve_needle(cal)

        # ---- range: registry hint, OCR
        range_ocr = None
        ocr_conf = 0.0
        ocr_method = ""
        ocr_labels: List[Tuple[int, str, float]] = []
        label_px = LABEL_HEIGHT_DIAL * px_per_unit
        want_ocr = self.cfg.ocr_enabled and (hint is None or hint.do_ocr) and label_px >= self.cfg.ocr_min_label_px_hyp
        if want_ocr:
            n_cands = [cal.n_div] if vrange is not None else self._division_candidates(cal)
            # blur of the up-sampled label patch: source PSF (~0.6 px) + measured blur
            sig_patch = (LABEL_PPU / max(px_per_unit, 1e-6)) * math.sqrt(0.6 ** 2 + (blur_px / 3.46) ** 2)
            est = self.read_range(meas, H, cal, label_px, n_cands, sig_patch, blur_px)
            if est is not None and est.confidence >= self.cfg.ocr_min_confidence:
                range_ocr = (est.vmin, est.vmax)
                ocr_conf = est.confidence
                ocr_method = est.method
                ocr_labels = est.labels
        if vrange is not None:
            vmin, vmax = vrange
            range_source = "registry"
        elif range_ocr is not None:
            vmin, vmax = range_ocr
            range_source = "ocr"
            if n_major_divisions(vmin, vmax) != cal.n_div:
                # recalibrate with the exact template of the OCR range
                cal = self.calibrate(D, red, range_ocr, res_deg, prior_rot, valid)
                theta, phi_d, fraction, ninfo, fine_ok = solve_needle(cal)
        else:
            vmin = vmax = None
            range_source = "none"
        value = None
        if vmin is not None:
            fr = float(np.clip(fraction, -self.cfg.clamp_margin_fs, 1.0 + self.cfg.clamp_margin_fs))
            value = gd.angle_to_value(gd.PHI_MIN_DEG - fr * gd.SWEEP_DEG, vmin, vmax)

        # ---- confidence
        c_det = float(np.clip(cand.score / 0.6, 0, 1))
        rms_rel = finfo.get("rms", 1.0) / max(a, 1.0)
        c_fit = float(math.exp(-(rms_rel / 0.02) ** 2)) * float(np.clip(finfo.get("edge_frac", 0) / 0.6, 0, 1))
        blur_ratio = blur_px / max(a, 1.0)
        c_blur = float(np.clip(1.0 - (blur_ratio - 0.06) / 0.30, 0.05, 1.0))
        c_cal = float(np.clip((cal.ncc - 0.4) / 0.4, 0, 1) * np.clip(cal.margin / 0.08, 0, 1))
        if ninfo.get("method") == "model":
            # relative residual reduction by the needle (real needles: >= 0.24 in the benchmark,
            # a needle hidden by an occluder ~0.03)
            c_needle = float(np.clip((ninfo.get("model_evidence", 0.0) - 0.05) / 0.2, 0, 1))
        else:
            nc = ninfo.get("needle_contrast", 0.0)
            c_needle = float(np.clip(nc / 0.5, 0, 1) * np.clip((nc - ninfo.get("needle_second", 0.0)) / 0.3, 0, 1))
            c_needle *= 1.0 if fine_ok else 0.6
        c_res = float(np.clip((px_per_unit * R_FACE - 12.0) / 25.0, 0.2, 1.0))
        conf = (c_det ** 0.5 * c_fit ** 0.5 * c_blur * c_cal ** 0.7 * c_needle ** 0.7 * c_res ** 0.5) ** (1.0 / 1.6)
        if range_source == "registry" and range_ocr is not None:
            agree = abs(range_ocr[0] - vmin) < 1e-6 and abs(range_ocr[1] - vmax) < 1e-6
            conf *= 1.0 if agree else 0.9
        conf = float(np.clip(conf, 0.0, 1.0))

        centre = apply_h(H, [[0.0, 0.0]])[0]
        reading = GaugeReading(
            value=value, vmin=vmin, vmax=vmax, fraction=float(fraction), angle_deg=float(phi_d),
            confidence=conf, center=(float(centre[0]), float(centre[1])),
            ellipse=(float(cx), float(cy), float(a), float(b), float(math.degrees(ang) % 180.0)),
            H_dial2img=H, rotation_deg=cal.rotation_deg, mirrored=cal.mirrored, n_div=cal.n_div,
            range_source=range_source, range_ocr=range_ocr, ocr_confidence=ocr_conf,
            ocr_method=ocr_method, ocr_labels=ocr_labels, blur_px=blur_px,
            gauge_id=hint.gauge_id if hint is not None else None,
            scores={"detect": c_det, "fit": c_fit, "blur": c_blur, "calibration": c_cal, "needle": c_needle,
                    "resolution": c_res, "cal_ncc": cal.ncc, "cal_margin": cal.margin,
                    "face_contrast": face_contrast, "rect_" + rect_mode: 1.0,
                    "needle_points": float(ninfo.get("n_points", 0)), "gain": float(enh["gain"]),
                    "needle_model_evidence": float(ninfo.get("model_evidence", -1.0)),
                    "needle_" + str(ninfo.get("method", "")): 1.0})
        # ---- acceptance
        if blur_ratio > self.cfg.max_blur_ratio:
            reading.accepted, reading.reject_reason = False, "blurred"
        elif cal.ncc < self.cfg.min_calibration_ncc:
            reading.accepted, reading.reject_reason = False, "no_scale"
        elif face_contrast < self.cfg.min_face_contrast:
            reading.accepted, reading.reject_reason = False, "no_face"
        elif conf < self.cfg.min_confidence:
            reading.accepted, reading.reject_reason = False, "low_confidence"
        return reading

    def _division_candidates(self, cal: Calibration) -> List[int]:
        """Division counts to try for OCR when the range is unknown: the calibration's best
        first, then the counts whose tick-ring template fits almost as well (the minor/major
        tick spacing tells 5 from 8 divisions much more reliably than tiny labels do)."""
        best = cal.ncc_by_div.get(cal.n_div, cal.ncc)
        others = sorted((n for n in GENERIC_RANGES if n != cal.n_div
                         and cal.ncc_by_div.get(n, best) >= best - self.cfg.ocr_div_ncc_margin),
                        key=lambda n: -cal.ncc_by_div.get(n, best))
        return [cal.n_div] + others


# ---------------------------------------------------------------------------------- temporal fusion
@dataclass
class FusedReading:
    value: float
    confidence: float
    n_samples: int        # inliers used
    n_total: int          # samples in the window
    mad: float
    outliers: int


class GaugeFusion:
    """Per-gauge temporal fusion: weighted median + MAD outlier rejection.

    Keeps the last ``window_n`` samples not older than ``window_s`` seconds.  Inliers satisfy
    |v - median| <= mad_k * max(1.4826 * MAD, min_sigma_fs * FS) (weighted median, MAD = median
    absolute deviation; with fewer than 3 samples the MAD is meaningless and only the floor
    min_sigma_fs * FS is used).  The fused value is the confidence-weighted mean of the inliers.
    Without any inlier (e.g. two disagreeing samples) the most confident, then most recent,
    sample is reported until a consistent majority forms."""

    def __init__(self, window_n: int = 15, window_s: float = 10.0, mad_k: float = 3.0,
                 min_sigma_fs: float = 0.004):
        self.window_n = int(window_n)
        self.window_s = float(window_s)
        self.mad_k = float(mad_k)
        self.min_sigma_fs = float(min_sigma_fs)
        self._buf: Dict[str, Deque[Tuple[float, float, float, float]]] = {}

    def reset(self, gauge_id: Optional[str] = None):
        if gauge_id is None:
            self._buf.clear()
        else:
            self._buf.pop(gauge_id, None)

    def add(self, gauge_id: str, t: float, value: float, confidence: float, full_scale: float) -> FusedReading:
        q = self._buf.setdefault(gauge_id, deque(maxlen=self.window_n))
        if q and t < q[-1][0] - 1.0:
            q.clear()                       # clock jumped backwards (sim reset)
        q.append((float(t), float(value), float(max(confidence, 1e-3)), float(abs(full_scale) or 1.0)))
        return self.fused(gauge_id, t)

    def fused(self, gauge_id: str, now: Optional[float] = None) -> Optional[FusedReading]:
        q = self._buf.get(gauge_id)
        if not q:
            return None
        if now is not None:
            while q and now - q[0][0] > self.window_s:
                q.popleft()
            if not q:
                return None
        arr = np.array(q, float)
        v, w, fs = arr[:, 1], arr[:, 2], arr[-1, 3]
        med = weighted_median(v, w)
        mad = float(np.median(np.abs(v - med)))
        sigma = self.min_sigma_fs * fs if len(v) < 3 else max(1.4826 * mad, self.min_sigma_fs * fs)
        inl = np.abs(v - med) <= self.mad_k * sigma
        if not np.any(inl):
            # no consistent majority: most confident sample, ties -> the most recent one
            best = max(range(len(v)), key=lambda i: (round(float(w[i]), 6), i))
            inl = np.zeros(len(v), bool)
            inl[best] = True
        val = float(np.sum(v[inl] * w[inl]) / np.sum(w[inl]))
        n_in = int(np.count_nonzero(inl))
        conf = float(np.mean(w[inl])) * min(1.0, 0.55 + 0.15 * n_in) * (n_in / len(v)) ** 0.5
        return FusedReading(val, float(np.clip(conf, 0, 1)), n_in, len(v), mad, len(v) - n_in)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    half = 0.5 * cw[-1]
    i = int(np.searchsorted(cw, half))
    if i < len(v) - 1 and abs(cw[i] - half) < 1e-12:
        return float(0.5 * (v[i] + v[i + 1]))
    return float(v[min(i, len(v) - 1)])


# ---------------------------------------------------------------------------------- overlay
STATUS_COLORS = {"normal": (60, 200, 60), "warning": (0, 165, 255), "alarm": (0, 0, 255), "unknown": (200, 200, 200)}


def draw_reading(img: np.ndarray, rd: GaugeReading, lines: Sequence[str], status: str = "unknown",
                 thickness: int = 2) -> np.ndarray:
    """Draw one reading in place: fitted face ellipse, needle, scale end points and text."""
    col = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
    cx, cy, a, b, angd = rd.ellipse
    cv2.ellipse(img, (int(round(cx * 4)), int(round(cy * 4))), (int(round(a * 4)), int(round(b * 4))), angd,
                0, 360, col, thickness, cv2.LINE_AA, shift=2)
    c = rd.center
    tip = rd.needle_tip_px(gd.R_NEEDLE)
    cv2.line(img, (int(round(c[0] * 4)), int(round(c[1] * 4))), (int(round(tip[0] * 4)), int(round(tip[1] * 4))),
             (255, 0, 255), thickness, cv2.LINE_AA, shift=2)
    ends = rd.dial_to_image([[0.40 * math.cos(math.radians(p)), 0.40 * math.sin(math.radians(p))]
                             for p in (gd.PHI_MIN_DEG, gd.PHI_MAX_DEG)])
    for (u, v), cc in zip(ends, ((255, 255, 0), (0, 0, 255))):
        cv2.circle(img, (int(round(u)), int(round(v))), max(2, thickness + 1), cc, -1, cv2.LINE_AA)
    scale = float(np.clip(a / 120.0, 0.45, 0.8))
    y = int(cy - b - 10 - (len(lines) - 1) * 22 * scale)
    y = max(y, int(18 * scale) + 2)
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = int(np.clip(cx - tw / 2, 2, img.shape[1] - tw - 2))
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + 4), (0, 0, 0), -1)
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1 if scale < 0.6 else 2, cv2.LINE_AA)
        y += int(22 * scale) + 4
    return img


def draw_header(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """Banner with status text at the top of an overlay image (in place)."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img
