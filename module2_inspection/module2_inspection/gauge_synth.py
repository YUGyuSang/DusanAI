"""
Synthetic camera frames of analog pressure gauges (ROS-free, numpy + OpenCV).

Used by the unit tests and by ``tools/eval_gauge.py`` to benchmark the gauge
reader under controlled, *known* conditions.  The dial texture is exactly the
one Gazebo renders (``dusan_common.gauge_design.render_gauge_face``); it is
projected as a planar 3-D disc into a pinhole camera that matches the robot's
inspection camera, composited over a cluttered industrial background and then
degraded the way a real (or simulated) camera on a shaking legged robot
degrades it:

    scene  : background clutter + gauge case (cylinder side) + stem pipe +
             circular distractors + optional occluding bar
    light  : global illumination scale (0.1 .. 1), spatial gradient/vignette,
             colour tint, optional specular glare on the glass, gamma
    optics : defocus (disc PSF), linear motion blur (random direction)
    sensor : shot (signal dependent) + read noise, 8-bit quantisation,
             optional JPEG compression artefacts

Geometry conventions
--------------------
Dial frame ("dial units"): origin at the dial centre, x to the right and y up
when looking at the face, lengths in units of the texture side ``S`` (so the
white face has radius ``gauge_design.R_FACE = 0.44``).  A texture pixel
(u_t, v_t) maps to dial units (u_t / S - 0.5, 0.5 - v_t / S).

Camera: OpenCV optical frame (z forward, x right, y down), intrinsics of
``dusan_common.camera.PinholeCamera``.  ``R_ow`` rotates world vectors into
the optical frame, ``C`` is the camera centre in world coordinates.

Every render returns the image *and* a ground-truth dictionary (value, needle
angle, projected centre, face ellipse, homographies, view parameters).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from dusan_common import gauge_design as gd
from dusan_common.camera import PinholeCamera

# link frame (x fwd, y left, z up) -> optical frame (z fwd, x right, y down)
R_OPTICAL_FROM_LINK = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

# tools/generate_world.py: the textured dial disc (radius dial_diameter / 2) carries the
# inscribed circle of the texture (DIAL_UV_RADIUS = 0.5), i.e. the texture side spans exactly
# dial_diameter metres; a dark housing cylinder (radius + 12 mm, 40 mm deep) sits behind it
# and a light square mounting plate (side dial_diameter + 0.10 m) behind that.
DIAL_UV_RADIUS = 0.5            # texture radius visible on the Gazebo dial disc (fraction of side)
HOUSING_EXTRA_M = 0.012         # housing cylinder radius - dial disc radius [m]
HOUSING_DEPTH_M = 0.040


# --------------------------------------------------------------------------- dataclasses
@dataclass
class ViewSpec:
    """Where the camera is relative to the dial (dial-centric parameterisation)."""

    distance: float = 1.5          # camera centre <-> dial centre [m]
    azimuth_deg: float = 0.0       # horizontal incidence (rotation about the dial 'up' axis)
    elevation_deg: float = -15.0   # vertical incidence; negative = camera below the dial
    roll_deg: float = 0.0          # camera roll about its optical axis (robot shake)
    offset_px: Tuple[float, float] = (0.0, 0.0)   # dial centre position w.r.t. the principal point

    @property
    def incidence_deg(self) -> float:
        a, e = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(a) * math.cos(e)))))


@dataclass
class SynthConditions:
    """Photometric / optical / sensor degradations applied to a rendered frame."""

    illumination: float = 1.0       # global light scale (0.1 = very dim plant room)
    gamma: float = 1.0              # scene/display gamma (>1 darkens mid-tones)
    illum_gradient: float = 0.15    # strength of the spatial illumination gradient + vignette
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # B, G, R gains (lamp colour)
    glare: float = 0.0              # peak of a specular highlight on the dial glass (0..1)
    noise_sigma: float = 1.5        # read noise std [DN] (8-bit scale)
    shot_noise: float = 0.0         # signal dependent noise: std = shot_noise * sqrt(DN)
    motion_blur_px: float = 0.0     # linear motion blur length [px]
    motion_blur_angle_deg: Optional[float] = None   # None -> random direction
    defocus_px: float = 0.0         # radius of the disc PSF [px]
    jpeg_quality: int = 0           # 0 = no compression, else 1..100
    occlusion: float = 0.0          # 0 = none, else bar thickness as a fraction of the face radius
    clutter: int = 14               # number of background clutter primitives
    distractors: int = 1            # number of round gauge-like distractors (white discs)
    case_depth: float = HOUSING_DEPTH_M   # depth of the gauge case cylinder [m]
    stem: bool = True               # draw the pipe stem below the gauge
    plate: bool = False             # light square mounting plate behind the case (Gazebo stand)
    plate_margin: float = 0.05      # plate half side = dial radius + margin [m] (Gazebo: 0.05)


@dataclass
class GaugeSpec:
    """The gauge being rendered (subset of a world_layout.yaml gauge entry)."""

    value: float
    vmin: float = 0.0
    vmax: float = 10.0
    unit: str = "bar"
    tag: str = ""
    diameter: float = 0.30          # physical side length of the dial texture [m]
    gauge_id: str = ""

    @classmethod
    def from_layout(cls, g: dict, value: Optional[float] = None) -> "GaugeSpec":
        return cls(value=float(g["value"] if value is None else value), vmin=float(g["min"]),
                   vmax=float(g["max"]), unit=str(g.get("unit", "")), tag=str(g.get("tag", "")),
                   diameter=texture_side(float(g.get("dial_diameter", 0.30))), gauge_id=str(g.get("id", "")))


def texture_side(dial_diameter: float) -> float:
    """Physical side length of the dial texture for a layout ``dial_diameter``.

    tools/generate_world.py maps the texture side onto ``dial_diameter`` (the disc of radius
    dial_diameter / 2 shows the inscribed circle of the texture), so the white face has radius
    ``gauge_design.R_FACE * dial_diameter`` in the world."""
    return float(dial_diameter)


# --------------------------------------------------------------------------- geometry
def default_camera() -> PinholeCamera:
    """The robot's inspection camera (800 x 600, hfov 0.9 rad)."""
    return PinholeCamera.from_sensor("inspection_camera")


def dial_axes(normal, up_ref=(0.0, 0.0, 1.0)):
    """Right / up unit vectors of a dial facing ``normal`` (texture 'up' = world up).

    right = up_ref x n  (to the right of a viewer looking at the face), up = n x right.
    """
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    up_ref = np.asarray(up_ref, float)
    if abs(float(np.dot(up_ref, n))) > 0.95:          # dial facing up/down: use world x as reference
        up_ref = np.array([1.0, 0.0, 0.0])
    right = np.cross(up_ref, n)
    right /= np.linalg.norm(right)
    up = np.cross(n, right)
    return right, up, n


def tex_to_dial(tex_size: int) -> np.ndarray:
    """3x3 map texture pixel (u_t, v_t, 1) -> dial units (x, y, 1)."""
    S = float(tex_size)
    return np.array([[1.0 / S, 0.0, -0.5], [0.0, -1.0 / S, 0.5], [0.0, 0.0, 1.0]])


def dial_homography(K, R_ow, C, centre, right, up, diameter) -> np.ndarray:
    """Homography dial units (x, y, 1) -> image pixels for a planar dial.

    X_world = centre + diameter * (x * right + y * up);   X_opt = R_ow (X_world - C).
    """
    R_ow = np.asarray(R_ow, float)
    cols = np.stack([R_ow @ (diameter * np.asarray(right, float)),
                     R_ow @ (diameter * np.asarray(up, float)),
                     R_ow @ (np.asarray(centre, float) - np.asarray(C, float))], axis=1)
    H = np.asarray(K, float) @ cols
    return H / H[2, 2]


def apply_h(H, pts) -> np.ndarray:
    """Apply a homography to (N, 2) points."""
    p = np.atleast_2d(np.asarray(pts, float))
    q = p @ H[:, :2].T + H[:, 2]
    return q[:, :2] / q[:, 2:3]


def ellipse_of_circle(H, radius, n=72):
    """Fit (cx, cy, a, b, angle_deg) to the image of the dial circle of ``radius`` (dial units)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = apply_h(H, np.stack([radius * np.cos(t), radius * np.sin(t)], 1)).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.fitEllipse(pts)
    if w < h:
        w, h, ang = h, w, ang + 90.0
    return float(cx), float(cy), float(w / 2), float(h / 2), float(ang % 180.0)


def look_at_rotation(C, target, up_world) -> np.ndarray:
    """R_ow for an optical camera at C looking at target with image-up ~ up_world."""
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    y_down = -np.asarray(up_world, float)
    x = np.cross(y_down, z)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 0)


def _rot_between(a, b) -> np.ndarray:
    """Minimal rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def view_pose(view: ViewSpec, camera: PinholeCamera):
    """Camera pose for a dial at the origin facing +z (right = +x, up = +y).

    Returns (R_ow, C) in that dial-centric world.
    """
    a, e = math.radians(view.azimuth_deg), math.radians(view.elevation_deg)
    w = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
    C = view.distance * w
    R0 = look_at_rotation(C, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    cr, sr = math.cos(math.radians(view.roll_deg)), math.sin(math.radians(view.roll_deg))
    R_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    du, dv = view.offset_px
    d_target = np.array([du / camera.fx, dv / camera.fy, 1.0])
    R_off = _rot_between(np.array([0.0, 0.0, 1.0]), d_target)
    return R_off @ R_roll @ R0, C


# --------------------------------------------------------------------------- image helpers
def motion_blur_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Normalised linear motion-blur kernel (sub-pixel length, any direction)."""
    length = float(max(length, 0.0))
    if length < 0.5:
        return np.ones((1, 1), np.float32)
    k = int(math.ceil(length)) + 3
    k += (k + 1) % 2
    ker = np.zeros((k, k), np.float32)
    c = (k - 1) / 2.0
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), -math.sin(a)])
    for s in np.linspace(-length / 2, length / 2, max(8, int(length * 6))):
        x, y = c + s * d[0], c + s * d[1]
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        ker[y0, x0] += (1 - fx) * (1 - fy)
        ker[y0, x0 + 1] += fx * (1 - fy)
        ker[y0 + 1, x0] += (1 - fx) * fy
        ker[y0 + 1, x0 + 1] += fx * fy
    return ker / ker.sum()


def defocus_kernel(radius: float) -> np.ndarray:
    """Normalised disc (pill-box) PSF with anti-aliased border."""
    radius = float(max(radius, 0.0))
    if radius < 0.3:
        return np.ones((1, 1), np.float32)
    ss = 8
    k = 2 * int(math.ceil(radius)) + 3
    big = np.zeros((k * ss, k * ss), np.uint8)
    cv2.circle(big, (k * ss // 2, k * ss // 2), int(round(radius * ss)), 255, -1, cv2.LINE_AA)
    ker = cv2.resize(big.astype(np.float32), (k, k), interpolation=cv2.INTER_AREA)
    return ker / max(ker.sum(), 1e-9)


def _rand_color(rng, lo=40, hi=200, sat=0.25):
    base = rng.uniform(lo, hi)
    return tuple(float(np.clip(base * (1 + rng.uniform(-sat, sat)), 0, 255)) for _ in range(3))


def make_background(width: int, height: int, rng: np.random.Generator, clutter: int = 14,
                    distractors: int = 1) -> np.ndarray:
    """Cluttered industrial background (uint8 BGR)."""
    W, H = int(width), int(height)
    base = np.array(_rand_color(rng, 60, 170, 0.2), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = rng.uniform(-0.35, 0.35, 2)
    shade = 1.0 + g[0] * (xx / W - 0.5) + g[1] * (yy / H - 0.5)
    img = base[None, None, :] * shade[..., None]
    # low-frequency texture (dirty walls, concrete)
    lf = cv2.resize(rng.normal(0, 1, (H // 24 + 2, W // 24 + 2)).astype(np.float32), (W, H),
                    interpolation=cv2.INTER_CUBIC)
    img += (rng.uniform(4, 14) * lf)[..., None]
    # OpenCV only anti-aliases (and putText only accepts) 8-bit images -> draw on uint8
    img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
    for _ in range(int(clutter)):
        kind = rng.choice(["rect", "pipe", "line", "bolt", "text"], p=[0.3, 0.25, 0.2, 0.15, 0.1])
        col = _rand_color(rng, 20, 230, 0.3)
        if kind == "rect":
            x0, y0 = rng.uniform(-50, W), rng.uniform(-50, H)
            w, h = rng.uniform(30, 300), rng.uniform(30, 300)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), col, -1, cv2.LINE_AA)
            if rng.random() < 0.5:
                cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)),
                              _rand_color(rng, 10, 80), int(rng.integers(1, 4)), cv2.LINE_AA)
        elif kind == "pipe":
            th = int(rng.uniform(12, 60))
            p0 = (int(rng.uniform(-100, W + 100)), int(rng.uniform(-100, H + 100)))
            p1 = (int(rng.uniform(-100, W + 100)), int(rng.uniform(-100, H + 100)))
            cv2.line(img, p0, p1, col, th, cv2.LINE_AA)
            hl = tuple(min(255.0, c * 1.35 + 15) for c in col)        # cylinder highlight stripe
            cv2.line(img, p0, p1, hl, max(2, th // 4), cv2.LINE_AA)
        elif kind == "line":
            p0 = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
            p1 = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
            cv2.line(img, p0, p1, col, int(rng.integers(1, 5)), cv2.LINE_AA)
        elif kind == "bolt":
            c = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
            r = int(rng.uniform(4, 30))
            cv2.circle(img, c, r, col, -1, cv2.LINE_AA)
            cv2.circle(img, c, r, _rand_color(rng, 10, 60), max(1, r // 5), cv2.LINE_AA)
        else:
            txt = "".join(rng.choice(list("ABCDEFGHKLMNPRSTUVW0123456789-")) for _ in range(int(rng.integers(3, 9))))
            cv2.putText(img, txt, (int(rng.uniform(0, W - 80)), int(rng.uniform(20, H))),
                        cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.4, 1.2)), col,
                        int(rng.integers(1, 3)), cv2.LINE_AA)
    for _ in range(int(distractors)):
        # round, bright, dark-rimmed objects (signs, lamps, other instruments) to stress
        # the dial validation (they lack the tick ring / needle)
        c = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        r = int(rng.uniform(15, 110))
        cv2.circle(img, c, int(r * 1.08), _rand_color(rng, 10, 50), -1, cv2.LINE_AA)
        cv2.circle(img, c, r, _rand_color(rng, 190, 250, 0.05), -1, cv2.LINE_AA)
        if rng.random() < 0.6:
            cv2.putText(img, str(int(rng.integers(0, 99))), (c[0] - r // 3, c[1] + r // 4),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.3, r / 50.0), (30, 30, 30), 2, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- renderer
class GaugeSynth:
    """Render realistic synthetic inspection-camera frames of an analog gauge."""

    def __init__(self, camera: Optional[PinholeCamera] = None, tex_size: int = 1024,
                 seed: Optional[int] = None):
        self.camera = camera or default_camera()
        self.tex_size = int(tex_size)
        self.rng = np.random.default_rng(seed)
        self._tex_cache: dict = {}

    # ------------------------------------------------------------ texture (mip-mapped)
    def _texture(self, g: GaugeSpec, level_size: int) -> np.ndarray:
        key = (round(g.value, 6), g.vmin, g.vmax, g.unit, g.tag)
        full = self._tex_cache.get(key)
        if full is None:
            full = gd.render_gauge_face(g.value, g.vmin, g.vmax, g.unit, g.tag, size=self.tex_size)
            if len(self._tex_cache) > 64:
                self._tex_cache.clear()
            self._tex_cache[key] = full
        if level_size >= full.shape[0]:
            return full
        # emulate GPU mip-mapping: box-filter the texture down to ~2x the projected size
        return cv2.resize(full, (level_size, level_size), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------ public API
    def render(self, gauge: GaugeSpec, view: ViewSpec, cond: Optional[SynthConditions] = None,
               rng: Optional[np.random.Generator] = None):
        """Render a dial seen from a dial-centric ``view``. Returns (bgr uint8, ground truth)."""
        R_ow, C = view_pose(view, self.camera)
        right, up, normal = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
        img, gt = self._render_geometry(gauge, R_ow, C, np.zeros(3), right, up, normal,
                                        cond or SynthConditions(), rng or self.rng)
        gt["view"] = asdict(view)
        gt["incidence_deg"] = view.incidence_deg
        return img, gt

    def render_world(self, gauge_layout: dict, T_world_camlink: np.ndarray,
                     cond: Optional[SynthConditions] = None, value: Optional[float] = None,
                     rng: Optional[np.random.Generator] = None):
        """Render a world_layout gauge seen by a camera whose *link* pose is T_world_camlink
        (Gazebo convention, x forward).  Returns (bgr uint8, ground truth)."""
        g = GaugeSpec.from_layout(gauge_layout, value)
        T = np.asarray(T_world_camlink, float)
        R_ow = R_OPTICAL_FROM_LINK @ T[:3, :3].T
        C = T[:3, 3]
        right, up, normal = dial_axes(gauge_layout["normal"])
        centre = np.asarray(gauge_layout["position"], float)
        img, gt = self._render_geometry(g, R_ow, C, centre, right, up, normal,
                                        cond or SynthConditions(), rng or self.rng)
        to_cam = C - centre
        gt["incidence_deg"] = math.degrees(math.acos(np.clip(np.dot(normal, to_cam) / np.linalg.norm(to_cam), -1, 1)))
        return img, gt

    def render_empty(self, cond: Optional[SynthConditions] = None,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """A frame without any gauge (clutter + round distractors) for false-positive tests."""
        cond = cond or SynthConditions()
        rng = rng or self.rng
        img = make_background(self.camera.width, self.camera.height, rng, cond.clutter,
                              max(cond.distractors, 2)).astype(np.float32)
        return self._degrade(img, cond, rng)[0]

    def _degrade(self, img: np.ndarray, cond: SynthConditions, rng: np.random.Generator):
        """Lighting, optics and sensor model applied to a float BGR frame -> (uint8, blur angle)."""
        Hh, W = img.shape[:2]
        yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float32)
        gdir = rng.uniform(-1, 1, 2)
        field_ = 1.0 + cond.illum_gradient * (gdir[0] * (xx / W - 0.5) + gdir[1] * (yy / Hh - 0.5))
        r2 = ((xx - W / 2) / W) ** 2 + ((yy - Hh / 2) / Hh) ** 2
        field_ *= 1.0 - cond.illum_gradient * 0.8 * r2           # vignetting
        img = img * (cond.illumination * field_)[..., None] * np.asarray(cond.tint, np.float32)
        if abs(cond.gamma - 1.0) > 1e-6:
            img = 255.0 * np.power(np.clip(img / 255.0, 0, 1), cond.gamma)
        if cond.defocus_px > 0.3:
            img = cv2.filter2D(img, -1, defocus_kernel(cond.defocus_px), borderType=cv2.BORDER_REFLECT)
        blur_angle = cond.motion_blur_angle_deg
        if cond.motion_blur_px > 0.5:
            if blur_angle is None:
                blur_angle = float(rng.uniform(0, 180))
            img = cv2.filter2D(img, -1, motion_blur_kernel(cond.motion_blur_px, blur_angle),
                               borderType=cv2.BORDER_REFLECT)
        sig = np.sqrt(cond.noise_sigma ** 2 + (cond.shot_noise ** 2) * np.clip(img, 0, None))
        img = img + rng.normal(0.0, 1.0, img.shape).astype(np.float32) * sig
        out = np.clip(np.round(img), 0, 255).astype(np.uint8)
        if cond.jpeg_quality and 0 < cond.jpeg_quality < 100:
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(cond.jpeg_quality)])
            if ok:
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return out, blur_angle

    # ------------------------------------------------------------ core
    def _render_geometry(self, g: GaugeSpec, R_ow, C, centre, right, up, normal,
                         cond: SynthConditions, rng: np.random.Generator):
        cam = self.camera
        W, Hh = cam.width, cam.height
        K = cam.K
        H_d2i = dial_homography(K, R_ow, C, centre, right, up, g.diameter)
        depth = float((R_ow @ (np.asarray(centre) - C))[2])
        face = ellipse_of_circle(H_d2i, gd.R_FACE)

        img = make_background(W, Hh, rng, cond.clutter, cond.distractors)
        D = g.diameter

        def project(pts_world):
            q = (np.asarray(pts_world) - C) @ R_ow.T
            q = q[q[:, 2] > 1e-3]
            uv = q[:, :2] / q[:, 2:3]
            return uv * [cam.fx, cam.fy] + [cam.cx, cam.cy]

        t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        r_case = DIAL_UV_RADIUS * D + HOUSING_EXTRA_M                  # housing cylinder radius [m]
        ring = (np.cos(t)[:, None] * right + np.sin(t)[:, None] * up) * r_case
        # mounting plate (as in the generated Gazebo gauge models: light grey square behind
        # the housing) - a bright surround right outside the dark rim
        if cond.plate:
            hs = DIAL_UV_RADIUS * D + cond.plate_margin
            back_c = centre - normal * (cond.case_depth + 0.008)
            corners = [back_c + hs * (sx * right + sy * up) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            uv = project(np.array(corners))
            if len(uv) == 4:
                cv2.fillPoly(img, [np.round(uv * 4).astype(np.int32)], _rand_color(rng, 170, 215, 0.04),
                             cv2.LINE_AA, shift=2)
        # stem pipe below the case (towards the mounting point)
        if cond.stem:
            w = 0.018
            p = [centre + up * (-r_case * 0.8) - normal * cond.case_depth * 0.5 + right * s * w
                 for s in (-1, 1)]
            p += [q - up * 0.9 for q in p[::-1]]
            uv = project(np.array(p))
            if len(uv) == 4:
                col = _rand_color(rng, 60, 140, 0.1)
                cv2.fillPoly(img, [np.round(uv * 4).astype(np.int32)], col, cv2.LINE_AA, shift=2)
        # gauge case: convex hull of the front and back rims (visible side at oblique views)
        back = centre + ring - normal * cond.case_depth
        front = centre + ring
        uv = project(np.concatenate([back, front], 0))
        if len(uv) > 3:
            hull = cv2.convexHull(np.round(uv * 4).astype(np.int32))
            cv2.fillPoly(img, [hull], _rand_color(rng, 45, 95, 0.05), cv2.LINE_AA, shift=2)

        # dial texture, mip level ~ 2x projected diameter
        proj_d = 2.0 * face[2] / (2 * gd.R_FACE)       # projected texture side [px]
        level = int(2 ** math.ceil(math.log2(max(64.0, 2.0 * proj_d))))
        level = min(level, self.tex_size)
        tex = self._texture(g, level).astype(np.float32)
        # pixel-centre exact mapping of the down-sampled level onto the full-size texture
        sc = self.tex_size / float(tex.shape[0])
        A_lvl = np.array([[sc, 0.0, 0.5 * sc - 0.5], [0.0, sc, 0.5 * sc - 0.5], [0.0, 0.0, 1.0]])
        H_t2i = H_d2i @ tex_to_dial(self.tex_size) @ A_lvl
        warped = cv2.warpPerspective(tex, H_t2i, (W, Hh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)
        level = tex.shape[0]
        mask_t = np.zeros((level, level), np.uint8)
        # the Gazebo disc shows the inscribed circle of the texture (radius 0.5 of the side)
        cv2.circle(mask_t, (level * 2 - 2, level * 2 - 2), int(DIAL_UV_RADIUS * level * 4), 255, -1,
                   cv2.LINE_AA, shift=2)
        mask = cv2.warpPerspective(mask_t.astype(np.float32) / 255.0, H_t2i, (W, Hh),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
        img = img.astype(np.float32) * (1 - mask) + warped * mask

        # specular glare on the glass
        if cond.glare > 0:
            gu, gv = apply_h(H_d2i, [[rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)]])[0]
            sig = face[2] * rng.uniform(0.15, 0.4)
            yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float32)
            blob = np.exp(-((xx - gu) ** 2 + (yy - gv) ** 2) / (2 * sig * sig)) * mask[..., 0]
            img = img + (255.0 * cond.glare * blob)[..., None]

        # occluding bar in front of the dial (pipe, cable, railing)
        if cond.occlusion > 0:
            a = rng.uniform(0, np.pi)
            off = rng.uniform(0.35, 0.8) * face[2] * rng.choice([-1, 1])
            d = np.array([math.cos(a), math.sin(a)])
            nrm = np.array([-d[1], d[0]])
            c0 = np.array(face[:2]) + nrm * off
            hw = 0.5 * cond.occlusion * face[2]
            L = 4 * face[2]
            poly = np.array([c0 - d * L + nrm * hw, c0 + d * L + nrm * hw,
                             c0 + d * L - nrm * hw, c0 - d * L - nrm * hw])
            occ = np.zeros((Hh, W), np.uint8)
            cv2.fillPoly(occ, [np.round(poly * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
            a_occ = (occ.astype(np.float32) / 255.0)[..., None]
            img = img * (1 - a_occ) + np.asarray(_rand_color(rng, 30, 160, 0.2), np.float32) * a_occ

        out, blur_angle = self._degrade(img, cond, rng)

        centre_px = apply_h(H_d2i, [[0.0, 0.0]])[0]
        gt = {
            "value": float(g.value), "vmin": float(g.vmin), "vmax": float(g.vmax),
            "unit": g.unit, "tag": g.tag, "gauge_id": g.gauge_id,
            "angle_deg": float(gd.value_to_angle(g.value, g.vmin, g.vmax)),
            "centre_px": (float(centre_px[0]), float(centre_px[1])),
            "face_ellipse": face, "radius_px": face[2],
            "H_dial2img": H_d2i, "H_tex2img": H_d2i @ tex_to_dial(self.tex_size),
            "depth": depth, "distance": float(np.linalg.norm(np.asarray(centre) - C)),
            "normal_opt": R_ow @ np.asarray(normal, float),
            "motion_blur_angle_deg": blur_angle,
            "in_image": bool(0 <= centre_px[0] < W and 0 <= centre_px[1] < Hh),
            "conditions": asdict(cond),
        }
        return out, gt


# --------------------------------------------------------------------------- random scenarios
CONDITION_KINDS = ("normal", "low_light", "motion_blur", "oblique", "noise", "combined", "occluded")


def random_view(rng: np.random.Generator, kind: str = "normal", camera: Optional[PinholeCamera] = None,
                distance_range=(0.9, 2.5)) -> ViewSpec:
    """Random but realistic robot-camera view of a gauge for an evaluation condition.

    The incidence angle (angle between the dial normal and the line of sight) is drawn
    from 0-20 deg for normal views and 25-42 deg for the oblique / combined conditions;
    the camera is mostly *below* the dial (robot camera at ~0.55 m, dials at 0.9-1.3 m).
    """
    cam = camera or default_camera()
    d = float(rng.uniform(*distance_range))
    if kind in ("oblique", "combined"):
        inc = math.radians(float(rng.uniform(25, 42)))
    else:
        inc = math.radians(float(rng.uniform(0, 20)))
    # direction of the camera around the normal: mostly below (sin psi < 0), sometimes level/above
    psi = math.radians(float(rng.uniform(180, 360) if rng.random() < 0.85 else rng.uniform(0, 180)))
    w = np.array([math.sin(inc) * math.cos(psi), math.sin(inc) * math.sin(psi), math.cos(inc)])
    az = math.degrees(math.atan2(w[0], w[2]))
    el = math.degrees(math.asin(np.clip(w[1], -1, 1)))
    roll = float(rng.normal(0, 3.0 if kind not in ("motion_blur", "combined") else 5.0))
    # keep the whole dial in the image (radius ~ f * 0.15 / d)
    r = cam.fx * 0.15 / d
    mu = max(0.0, cam.width / 2 - r - 20)
    mv = max(0.0, cam.height / 2 - r - 20)
    off = (float(rng.uniform(-0.8, 0.8) * mu), float(rng.uniform(-0.8, 0.8) * mv))
    return ViewSpec(distance=d, azimuth_deg=az, elevation_deg=el, roll_deg=roll, offset_px=off)


def random_conditions(rng: np.random.Generator, kind: str = "normal") -> SynthConditions:
    """Degradation preset for one of ``CONDITION_KINDS`` (random within the preset)."""
    c = SynthConditions(illumination=float(rng.uniform(0.7, 1.0)), gamma=float(rng.uniform(0.9, 1.15)),
                        illum_gradient=float(rng.uniform(0.0, 0.3)), noise_sigma=float(rng.uniform(0.5, 2.0)),
                        tint=tuple(float(x) for x in rng.uniform(0.92, 1.08, 3)),
                        clutter=int(rng.integers(8, 22)), distractors=int(rng.integers(0, 3)),
                        jpeg_quality=int(rng.choice([0, 0, 90])), plate=bool(rng.random() < 0.5),
                        glare=float(rng.uniform(0.03, 0.12)) if rng.random() < 0.3 else 0.0)
    if kind == "low_light":
        c.illumination = float(rng.uniform(0.1, 0.3))
        c.gamma = float(rng.uniform(1.0, 1.6))
        c.noise_sigma = float(rng.uniform(1.0, 3.0))
        c.shot_noise = float(rng.uniform(0.2, 0.6))
        c.illum_gradient = float(rng.uniform(0.2, 0.6))
    elif kind == "motion_blur":
        c.motion_blur_px = float(rng.uniform(4, 15))
        c.defocus_px = float(rng.uniform(0, 1.0))
    elif kind == "noise":
        c.noise_sigma = float(rng.uniform(6, 14))
        c.shot_noise = float(rng.uniform(0.3, 1.0))
        c.jpeg_quality = int(rng.choice([0, 40, 60]))
    elif kind == "oblique":
        c.defocus_px = float(rng.uniform(0, 0.8))
    elif kind == "combined":
        c.illumination = float(rng.uniform(0.15, 0.4))
        c.gamma = float(rng.uniform(1.0, 1.4))
        c.noise_sigma = float(rng.uniform(2, 6))
        c.shot_noise = float(rng.uniform(0.2, 0.6))
        c.motion_blur_px = float(rng.uniform(3, 10))
        c.defocus_px = float(rng.uniform(0, 1.2))
        c.illum_gradient = float(rng.uniform(0.2, 0.5))
        c.jpeg_quality = int(rng.choice([0, 60, 80]))
        c.glare = float(rng.choice([0.0, rng.uniform(0.05, 0.2)]))
    elif kind == "occluded":
        c.occlusion = float(rng.uniform(0.12, 0.3))
    return c
