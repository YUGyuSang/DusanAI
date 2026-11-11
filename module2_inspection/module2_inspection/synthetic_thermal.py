"""
Synthetic thermal camera (ROS free: numpy + OpenCV).

Renders a radiometric thermal frame (Kelvin, or L16 counts = K / 0.01) of the
virtual plant from the robot pose.  It is the fallback for the Gazebo thermal
sensor (``thermal_source: synthetic``) and the image source of the offline
thermal benchmark (``tools/eval_thermal.py``).

Scene (everything from ``world_layout.yaml``)
---------------------------------------------
* **Analytic, per-pixel ray casting** (initialises the z-buffer, exact and
  hole free for large planar geometry): floor (piecewise linear: factory,
  corridor upper level, descent, lower level), walls and ceilings of the
  turbine hall and the corridor (with the east opening as a portal),
  machines (boxes from the layout footprints, 303 K), pipe-rack supports,
  pallets and the low platform (oriented boxes) and drums (vertical
  cylinders).
* **Z-buffered splatting of sampled 3-D surface points** for the thermally
  relevant curved geometry: the pipe-rack pipe (``pipe_body_k``), the hot-spot
  primitives (flanges / bearing, ``temp_k``), the corridor pipe stub, lying
  debris pipes and the gauge dials.  Each cylinder is cut into short chunks;
  every chunk is frustum culled and sampled on a (s, theta) grid whose spacing
  follows the chunk depth (~0.9 px per sample), back faces are culled, and
  every sample is splatted as a k x k pixel square (k from its projected
  spacing).  The nearest sample per pixel wins if it is in front of the
  analytic depth.

Temperatures
------------
* hot-spot primitive: ``temp_k`` in the middle, dropping by 10 % of the
  excess towards its rims (``T = T_h - 0.1 (T_h - T_base) rho^2``);
* heat halo around every hot spot (radial falloff with the distance to the
  primitive surface): conductive on connected bodies (pipe, stub, machine
  housing: 50 %, lambda 0.10 m), radiative elsewhere (8 %, lambda 0.30 m);
* pipe body ``pipe_body_k``, machines 303 K with a vertical gradient, floor /
  walls / ceiling at ambient with a slight stratification and texture;
* directional emissivity (surfaces seen at grazing angles look closer to the
  reflected ambient temperature);
* sensor model: Gaussian PSF, NETD white noise, column fixed-pattern noise,
  row noise, 0.01 K quantisation, clamp to the sensor range.
"""
from __future__ import annotations

import math
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dusan_common.camera import PinholeCamera, camera_world_pose
from dusan_common.geometry import rot_z
from dusan_common.robot_model import NOMINAL_BODY_HEIGHT, SENSORS, THERMAL_RESOLUTION

# ---------------------------------------------------------------- labels
LBL_NONE, LBL_FLOOR, LBL_WALL, LBL_CEIL = 0, 1, 2, 3
LBL_MACHINE = 10          # + machine index
LBL_HOT = 100             # + hot-spot index (layout order)
LBL_PIPE, LBL_SUPPORT, LBL_STUB, LBL_PLATFORM = 200, 201, 202, 203
LBL_OBSTACLE = 300        # + obstacle index
LBL_GAUGE = 400           # + gauge index

DEFAULT_MACHINE_HEIGHTS = {"turbine_machine": 3.1, "gas_tank": 6.6}

# visible-proxy albedo (BGR, 0..1) per label class
_ALBEDO = {
    "floor": (0.55, 0.56, 0.57), "wall": (0.70, 0.76, 0.80), "ceiling": (0.40, 0.42, 0.45),
    "machine": (0.35, 0.55, 0.30), "tank": (0.62, 0.62, 0.58), "pipe": (0.72, 0.72, 0.70),
    "flange": (0.25, 0.25, 0.28), "bearing": (0.20, 0.22, 0.25), "support": (0.10, 0.55, 0.85),
    "pallet": (0.35, 0.55, 0.72), "drum": (0.15, 0.20, 0.65), "gauge": (0.93, 0.94, 0.95),
    "platform": (0.45, 0.45, 0.45), "debris": (0.40, 0.42, 0.45),
}


def full_pose(layout, x: float, y: float, yaw: float, body_height: float = NOMINAL_BODY_HEIGHT,
              roll: float = 0.0, pitch: float = 0.0) -> Tuple[float, float, float, float, float, float]:
    """(x, y, yaw) -> (x, y, z, roll, pitch, yaw) with the base at floor + body height."""
    return (float(x), float(y), float(layout.floor_z(x, y) + body_height), float(roll), float(pitch), float(yaw))


def _crc32(data) -> int:
    """CRC32 of a message payload (bytes, array.array, memoryview, numpy array or int list)."""
    try:
        return zlib.crc32(data)
    except TypeError:  # not a contiguous buffer (e.g. a plain list of ints)
        return zlib.crc32(bytes(bytearray(data)))


class OwnFrameFilter:
    """Tells frames this node published from frames of another publisher on the same topic.

    A publisher count cannot do this: ros_gz_bridge keeps a publisher on /thermal_camera/image
    even when the Gazebo sensor is disabled.  The node records the CRC32 of every payload it
    publishes (``add_own``) and checks each message received on its own output topic
    (``is_foreign``); the synthetic frames carry fresh noise, so a frame of a second renderer
    never matches one of the last ``history`` own frames."""

    def __init__(self, history: int = 50):
        self._own = deque(maxlen=max(1, int(history)))

    def add_own(self, data) -> None:
        self._own.append(_crc32(data))

    def is_foreign(self, data) -> bool:
        return _crc32(data) not in self._own


def _orthonormal(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return u, v


# =============================================================================
# Scene primitives
# =============================================================================
@dataclass
class HotSource:
    """Hot-spot primitive (cylinder) used for the heat halo."""
    center: np.ndarray
    axis: np.ndarray
    radius: float
    half_len: float
    temp_k: float
    base_k: float

    def surface_distance(self, P: np.ndarray) -> np.ndarray:
        """Euclidean distance of points (N,3) to the finite cylinder (0 inside)."""
        w = P - self.center
        s = w @ self.axis
        rho = np.linalg.norm(w - s[:, None] * self.axis, axis=1)
        dx = np.maximum(np.abs(s) - self.half_len, 0.0)
        dr = np.maximum(rho - self.radius, 0.0)
        return np.hypot(dx, dr)


@dataclass
class SplatCylinder:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    length: float
    label: int
    kind: str                 # pipe | flange | bearing | stub | debris | gauge
    base_k: float
    hot: int = -1             # index into ThermalScene.hot (hot-spot primitive) or -1
    caps: bool = True
    connected: bool = True    # receives the conductive halo
    u: np.ndarray = field(default=None)
    v: np.ndarray = field(default=None)

    def __post_init__(self):
        self.axis = np.asarray(self.axis, dtype=float)
        self.axis /= np.linalg.norm(self.axis)
        self.center = np.asarray(self.center, dtype=float)
        self.u, self.v = _orthonormal(self.axis)


@dataclass
class AnalyticBox:
    center: np.ndarray
    half: np.ndarray
    yaw: float
    label: int
    kind: str
    base_k: float
    connected: bool = False
    R: np.ndarray = field(default=None)

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.half = np.asarray(self.half, dtype=float)
        self.R = rot_z(self.yaw)


@dataclass
class AnalyticVCylinder:
    x: float
    y: float
    z0: float
    z1: float
    radius: float
    label: int
    kind: str
    base_k: float


@dataclass
class RenderBuffers:
    depth: np.ndarray        # (H, W) depth along the optical axis (camera link x), m
    label: np.ndarray        # (H, W) int32 surface label
    temp: np.ndarray         # (H, W) apparent surface temperature (K) before the sensor model
    normal: np.ndarray       # (H, W, 3) world surface normal
    point: np.ndarray        # (H, W, 3) world surface point
    n_splats: int = 0


# =============================================================================
# Scene
# =============================================================================
class ThermalScene:
    """Thermal scene description built from the layout."""

    def __init__(self, layout, machine_k: float = 303.15, machine_heights: Optional[dict] = None,
                 halo_conductive: Tuple[float, float] = (0.50, 0.10),
                 halo_radiative: Tuple[float, float] = (0.08, 0.30),
                 edge_drop: float = 0.10, emissivity_power: float = 5.0,
                 density_px: float = 0.9, min_spacing: float = 0.002, chunk_len: float = 0.25,
                 max_splat: int = 5):
        self.layout = layout
        th = layout.thermal
        self.ambient_k = float(th.get("ambient_k", 298.15))
        self.pipe_k = float(th.get("pipe_body_k", self.ambient_k + 10.0))
        self.machine_k = float(machine_k)
        self.halo_conductive = halo_conductive
        self.halo_radiative = halo_radiative
        self.edge_drop = float(edge_drop)
        self.emissivity_power = float(emissivity_power)
        self.density_px = float(density_px)
        self.min_spacing = float(min_spacing)
        self.chunk_len = float(chunk_len)
        self.max_splat = int(max_splat)
        heights = dict(DEFAULT_MACHINE_HEIGHTS)
        heights.update(machine_heights or {})

        self.hot: List[HotSource] = []
        self.splats: List[SplatCylinder] = []
        self.boxes: List[AnalyticBox] = []
        self.vcyls: List[AnalyticVCylinder] = []
        self.label_kind = {LBL_FLOOR: "floor", LBL_WALL: "wall", LBL_CEIL: "ceiling"}
        self._ray_cache: dict = {}    # per-pixel unit rays in the camera link frame, per intrinsics

        # ---- rooms (axis aligned interiors) + floor profile
        f = layout.get("factory", {})
        fi = f.get("interior", {"x_min": -10.0, "x_max": 15.0, "y_min": -9.0, "y_max": 9.0})
        self.rooms = [dict(lo=np.array([fi["x_min"], fi["y_min"], -np.inf]),
                           hi=np.array([fi["x_max"], fi["y_max"], f.get("floor_z", 0.0) + f.get("height", 7.3)]),
                           name="factory")]
        c = layout.get("corridor")
        self.portal = None
        floor0 = float(f.get("floor_z", 0.0))
        self.floor_segments = [(-np.inf, np.inf, floor0, 0.0)]
        if c:
            ci = c["interior"]
            self.rooms.append(dict(lo=np.array([fi["x_max"], ci["y_min"], -np.inf]),
                                   hi=np.array([ci["x_max"], ci["y_max"], c["upper_floor"]["z"] + c.get("height", 4.0)]),
                                   name="corridor"))
            # the factory east opening connects both rooms (portal on x = factory x_max)
            self.portal = dict(x=fi["x_max"], y_min=ci["y_min"], y_max=ci["y_max"],
                               z_max=c["upper_floor"]["z"] + c.get("height", 4.0))
            de, up, lo = c["descent"], c["upper_floor"], c["lower_floor"]
            slope = (lo["z"] - up["z"]) / (de["x_max"] - de["x_min"])
            self.floor_segments = [(-np.inf, de["x_min"], up["z"], 0.0),
                                   (de["x_min"], de["x_max"], up["z"] - slope * de["x_min"], slope),
                                   (de["x_max"], np.inf, lo["z"], 0.0)]
            if abs(up["z"] - floor0) > 1e-6:   # factory floor differs from the corridor upper floor
                self.floor_segments.insert(0, (-np.inf, fi["x_max"], floor0, 0.0))
                self.floor_segments[1] = (fi["x_max"], de["x_min"], up["z"], 0.0)

        # ---- machines (boxes from the footprints)
        for i, m in enumerate(layout.get("machines", []) or []):
            fp = m["footprint"]
            hgt = float(m.get("height", heights.get(m.get("model", ""), 3.0)))
            z0 = layout.floor_z(0.5 * (fp["x_min"] + fp["x_max"]), 0.5 * (fp["y_min"] + fp["y_max"]))
            ctr = [0.5 * (fp["x_min"] + fp["x_max"]), 0.5 * (fp["y_min"] + fp["y_max"]), z0 + 0.5 * hgt]
            half = [0.5 * (fp["x_max"] - fp["x_min"]), 0.5 * (fp["y_max"] - fp["y_min"]), 0.5 * hgt]
            kind = "tank" if "tank" in str(m.get("model", "")) else "machine"
            self.boxes.append(AnalyticBox(ctr, half, 0.0, LBL_MACHINE + i, kind, self.machine_k, connected=True))
            self.label_kind[LBL_MACHINE + i] = kind

        # ---- pipe rack
        pr = layout.get("pipe_rack") or {}
        if pr.get("pipe"):
            a = np.asarray(pr["pipe"]["start"], float)
            b = np.asarray(pr["pipe"]["end"], float)
            self.splats.append(SplatCylinder(0.5 * (a + b), b - a, float(pr["pipe"]["radius"]),
                                             float(np.linalg.norm(b - a)), LBL_PIPE, "pipe", self.pipe_k))
            self.label_kind[LBL_PIPE] = "pipe"
            sx, sy, sz = pr.get("support_size", [0.12, 0.5, 1.5])
            yc = float(a[1])
            for x in pr.get("supports_x", []):
                z0 = layout.floor_z(x, yc)
                self.boxes.append(AnalyticBox([x, yc, z0 + 0.5 * sz], [0.5 * sx, 0.5 * sy, 0.5 * sz], 0.0,
                                              LBL_SUPPORT, "support", self.ambient_k + 0.8, connected=True))
            self.label_kind[LBL_SUPPORT] = "support"

        # ---- corridor pipe stub
        stub = layout.get("corridor_pipe_stub")
        if stub:
            x, y = stub["position"]
            z0 = layout.floor_z(x, y)
            hgt = float(stub.get("height", 2.5))
            self.splats.append(SplatCylinder([x, y, z0 + 0.5 * hgt], [0, 0, 1], float(stub.get("radius", 0.06)),
                                             hgt, LBL_STUB, "stub", self.pipe_k))
            self.label_kind[LBL_STUB] = "pipe"

        # ---- hot spots
        for i, h in enumerate(layout.hotspots):
            kind = str(h.get("kind", "flange"))
            base = self.machine_k if kind == "bearing" else self.pipe_k
            axis = np.asarray(h.get("axis", [1, 0, 0]), float)
            axis /= np.linalg.norm(axis)
            r, L = float(h.get("radius", 0.1)), float(h.get("length", 0.1))
            self.hot.append(HotSource(np.asarray(h["position"], float), axis, r, 0.5 * L, float(h["temp_k"]), base))
            self.splats.append(SplatCylinder(h["position"], axis, r, L, LBL_HOT + i,
                                             "bearing" if kind == "bearing" else "flange", base, hot=i))
            self.label_kind[LBL_HOT + i] = "bearing" if kind == "bearing" else "flange"

        # ---- obstacles
        for i, o in enumerate(layout.obstacles):
            t = o.get("type")
            lbl = LBL_OBSTACLE + i
            fz = float(o.get("floor_z", 0.0))
            if t == "drum":
                cx, cy = o["center"]
                self.vcyls.append(AnalyticVCylinder(cx, cy, fz, fz + float(o.get("height", 0.9)),
                                                    float(o.get("radius", 0.3)), lbl, "drum", self.ambient_k + 0.3))
                self.label_kind[lbl] = "drum"
            elif t == "pallet":
                cx, cy = o["center"]
                hgt = 0.18 * int(o.get("stack", 1))
                self.boxes.append(AnalyticBox([cx, cy, fz + 0.5 * hgt], [0.61, 0.70, 0.5 * hgt], float(o.get("yaw", 0.0)),
                                              lbl, "pallet", self.ambient_k - 0.3))
                self.label_kind[lbl] = "pallet"
            elif t == "platform":
                x0, y0, x1, y1 = o["rect"]
                hgt = float(o.get("height", 0.1))
                self.boxes.append(AnalyticBox([0.5 * (x0 + x1), 0.5 * (y0 + y1), fz + 0.5 * hgt],
                                              [0.5 * abs(x1 - x0), 0.5 * abs(y1 - y0), 0.5 * hgt], 0.0,
                                              lbl, "platform", self.ambient_k - 0.6))
                self.label_kind[lbl] = "platform"
            elif t == "pipe":
                cx, cy = o["center"]
                d = float(o.get("diameter", 0.1))
                yaw = float(o.get("yaw", 0.0))
                self.splats.append(SplatCylinder([cx, cy, fz + 0.5 * d], [math.cos(yaw), math.sin(yaw), 0.0], 0.5 * d,
                                                 float(o.get("length", 2.0)), lbl, "debris", self.ambient_k + 0.3,
                                                 connected=False))
                self.label_kind[lbl] = "debris"

        # ---- gauge dials (thin discs, slightly below ambient: glass + reflections)
        dial_t = 0.06
        for i, g in enumerate(layout.gauges):
            n = np.asarray(g.get("normal", [1, 0, 0]), float)
            n /= np.linalg.norm(n)
            ctr = np.asarray(g["position"], float) - 0.5 * dial_t * n
            self.splats.append(SplatCylinder(ctr, n, 0.5 * float(g.get("dial_diameter", 0.3)), dial_t,
                                             LBL_GAUGE + i, "gauge", self.ambient_k - 0.8, connected=False))
            self.label_kind[LBL_GAUGE + i] = "gauge"

    # ------------------------------------------------------------- rooms
    def _room_index(self, p: np.ndarray) -> int:
        best, best_d = 0, math.inf
        for i, r in enumerate(self.rooms):
            dx = max(r["lo"][0] - p[0], 0.0, p[0] - r["hi"][0])
            dy = max(r["lo"][1] - p[1], 0.0, p[1] - r["hi"][1])
            d = math.hypot(dx, dy)
            if d < best_d - 1e-9:
                best, best_d = i, d
        return best

    @staticmethod
    def _slab_far(o: np.ndarray, D: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Exit distance of rays from an axis aligned box and the exit axis."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hi = (hi[None, :] - o[None, :]) / D
            t_lo = (lo[None, :] - o[None, :]) / D
        t_far = np.where(D > 0, t_hi, np.where(D < 0, t_lo, np.inf))
        t_far = np.where(np.isfinite(t_far), t_far, np.inf)
        ax = np.argmin(t_far, axis=1)
        return t_far[np.arange(len(D)), ax], ax

    # -------------------------------------------------------- background
    def _background(self, o: np.ndarray, D: np.ndarray, subset=None):
        """Analytic ray cast of all rays D (N,3) from o.  Returns t, label,
        normal (N,3).  ``subset(corners_world)`` may return the indices of the
        rays that can hit an object with these bounding corners (screen-space
        culling; None = all rays)."""
        N = len(D)
        t_best = np.full(N, np.inf)
        label = np.zeros(N, np.int32)
        normal = np.zeros((N, 3))
        # ---- floor (piecewise linear in x)
        for x0, x1, a, b in self.floor_segments:
            denom = D[:, 2] - b * D[:, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (a + b * o[0] - o[2]) / denom
            x = o[0] + t * D[:, 0]
            ok = (np.abs(denom) > 1e-9) & (t > 1e-4) & (x >= x0 - 1e-6) & (x <= x1 + 1e-6) & (t < t_best)
            t_best[ok] = t[ok]
            label[ok] = LBL_FLOOR
            nrm = np.array([-b, 0.0, 1.0]) / math.hypot(b, 1.0)
            normal[ok] = nrm
        # ---- walls / ceiling of the room containing the camera (+ portal)
        ri = self._room_index(o)
        room = self.rooms[ri]
        t_w, ax = self._slab_far(o, D, room["lo"], room["hi"])
        if self.portal is not None and len(self.rooms) > 1:
            hit = o[None, :] + t_w[:, None] * D
            pr = self.portal
            through = ((ax == 0) & (np.abs(hit[:, 0] - pr["x"]) < 1e-3)
                       & (hit[:, 1] > pr["y_min"]) & (hit[:, 1] < pr["y_max"]) & (hit[:, 2] < pr["z_max"]))
            if np.any(through):
                other = self.rooms[1 - ri] if len(self.rooms) == 2 else self.rooms[0]
                t2, ax2 = self._slab_far(o, D[through], other["lo"], other["hi"])
                t_w[through] = t2
                ax[through] = ax2
        ok = t_w < t_best
        t_best[ok] = t_w[ok]
        label[ok] = np.where(ax[ok] == 2, LBL_CEIL, LBL_WALL)
        sgn = np.sign(D[np.arange(N), ax])
        wn = np.zeros((N, 3))
        wn[np.arange(N), ax] = -sgn
        normal[ok] = wn[ok]
        # ---- boxes (oriented about z); only rays inside the box's screen rectangle
        for bx in self.boxes:
            corners = bx.center + (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                                   * bx.half) @ bx.R.T
            sel = subset(corners) if subset is not None else None
            if sel is not None and len(sel) == 0:
                continue
            Ds = D if sel is None else D[sel]
            Rt = bx.R.T
            ol = Rt @ (o - bx.center)
            Dl = Ds @ Rt.T
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-bx.half[None, :] - ol[None, :]) / Dl
                t2 = (bx.half[None, :] - ol[None, :]) / Dl
            tmin = np.minimum(t1, t2)
            tmax = np.maximum(t1, t2)
            # rays parallel to a slab: inside -> (-inf, inf), outside -> empty
            par = np.abs(Dl) < 1e-12
            if np.any(par):
                inside = np.abs(ol)[None, :] <= bx.half[None, :]
                tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin)
                tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
            tn = tmin.max(axis=1)
            tf = tmax.min(axis=1)
            tb = t_best if sel is None else t_best[sel]
            hit = (tn <= tf) & (tn > 1e-4) & (tn < tb)
            if not np.any(hit):
                continue
            axn = np.argmax(tmin[hit], axis=1)
            nl = np.zeros((int(hit.sum()), 3))
            nl[np.arange(len(axn)), axn] = -np.sign(Dl[hit][np.arange(len(axn)), axn])
            gi = np.nonzero(hit)[0] if sel is None else sel[hit]
            t_best[gi] = tn[hit]
            label[gi] = bx.label
            normal[gi] = nl @ bx.R.T
        # ---- vertical cylinders (drums)
        for cy in self.vcyls:
            corners = np.array([[cy.x + sx * cy.radius, cy.y + sy * cy.radius, z]
                                for sx in (-1, 1) for sy in (-1, 1) for z in (cy.z0, cy.z1)])
            sel = subset(corners) if subset is not None else None
            if sel is not None and len(sel) == 0:
                continue
            gi_all = np.arange(N) if sel is None else sel
            Ds = D[gi_all]
            tb = t_best[gi_all]
            ox, oy = o[0] - cy.x, o[1] - cy.y
            a = Ds[:, 0] ** 2 + Ds[:, 1] ** 2
            b = 2.0 * (ox * Ds[:, 0] + oy * Ds[:, 1])
            c = ox * ox + oy * oy - cy.radius ** 2
            disc = b * b - 4.0 * a * c
            with np.errstate(divide="ignore", invalid="ignore"):
                ts = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
            z = o[2] + ts * Ds[:, 2]
            side = (disc > 0) & (a > 1e-12) & (ts > 1e-4) & (z >= cy.z0) & (z <= cy.z1) & (ts < tb)
            if np.any(side):
                gi = gi_all[side]
                t_best[gi] = ts[side]
                tb[side] = ts[side]
                label[gi] = cy.label
                px = o[0] + ts[side] * Ds[side, 0] - cy.x
                py = o[1] + ts[side] * Ds[side, 1] - cy.y
                nn = np.stack([px, py, np.zeros_like(px)], -1)
                normal[gi] = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
            # top cap
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = (cy.z1 - o[2]) / Ds[:, 2]
            xx = o[0] + tt * Ds[:, 0] - cy.x
            yy = o[1] + tt * Ds[:, 1] - cy.y
            top = (Ds[:, 2] < 0) & (tt > 1e-4) & (xx * xx + yy * yy <= cy.radius ** 2) & (tt < tb)
            if np.any(top):
                gi = gi_all[top]
                t_best[gi] = tt[top]
                label[gi] = cy.label
                normal[gi] = np.array([0.0, 0.0, 1.0])
        return t_best, label, normal

    def _halo(self, P: np.ndarray, base: np.ndarray, connected: np.ndarray, skip_hot: int = -1) -> np.ndarray:
        """Add the heat halo of every hot spot (radial exponential falloff with
        the distance to the hot primitive's surface)."""
        out = base.astype(np.float64).copy()
        if len(P) == 0:
            return out
        kc, lc = self.halo_conductive
        kr, lr = self.halo_radiative
        k = np.where(connected, kc, kr)
        lam = np.where(connected, lc, lr)
        reach = 6.0 * max(lc, lr)
        for i, h in enumerate(self.hot):
            if i == skip_hot:
                continue
            # cheap bounding-sphere prefilter before the exact surface distance
            r_b = math.hypot(h.radius, h.half_len) + reach
            cand = np.nonzero(np.sum((P - h.center) ** 2, axis=1) < r_b * r_b)[0]
            if len(cand) == 0:
                continue
            dist = h.surface_distance(P[cand])
            near = dist < 6.0 * lam[cand]
            if not np.any(near):
                continue
            idx = cand[near]
            excess = np.maximum(h.temp_k - base[idx], 0.0)
            out[idx] += excess * k[idx] * np.exp(-dist[near] / lam[idx])
        if self.hot:
            out = np.minimum(out, max(h.temp_k for h in self.hot))
        return out

    def _background_temperature(self, P: np.ndarray, label: np.ndarray) -> np.ndarray:
        amb = self.ambient_k
        T = np.full(len(P), amb, dtype=np.float64)
        connected = np.zeros(len(P), bool)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        fl = label == LBL_FLOOR
        T[fl] = amb - 1.0 + 0.012 * x[fl] + 0.25 * np.sin(0.7 * x[fl]) * np.sin(0.9 * y[fl])
        wl = label == LBL_WALL
        T[wl] = amb - 0.6 + 0.30 * (z[wl] - self.layout.floor_z(0.0, 0.0)) + 0.15 * np.sin(1.3 * (x[wl] + y[wl]))
        ce = label == LBL_CEIL
        T[ce] = amb + 1.8
        for bx in self.boxes:
            m = label == bx.label
            if not np.any(m):
                continue
            if bx.kind in ("machine", "tank"):
                zr = (z[m] - (bx.center[2] - bx.half[2])) / max(2.0 * bx.half[2], 1e-6)
                T[m] = bx.base_k + 1.2 * (zr - 0.5) + 0.35 * np.sin(2.1 * x[m]) * np.cos(1.7 * y[m])
            else:
                T[m] = bx.base_k
            connected[m] = bx.connected
        for cy in self.vcyls:
            m = label == cy.label
            T[m] = cy.base_k + 0.3 * (z[m] - cy.z0)
        return self._halo(P, T, connected)

    # ----------------------------------------------------------- splats
    def _splat_samples(self, cam_pos: np.ndarray, T_cw: np.ndarray, cam: PinholeCamera,
                       near: float, far: float):
        """Sample visible-side surface points of all splat cylinders.
        Returns P (N,3), normals (N,3), spacing (N,), label (N,), temperature (N,)."""
        tan_h = math.tan(0.5 * cam.hfov) * 1.05
        tan_v = math.tan(0.5 * cam.vfov) * 1.05
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
        Ps, Ns, Ss, Ls, Ts = [], [], [], [], []

        def visible(c_world: np.ndarray, rb: float) -> Tuple[bool, float]:
            pl = R_cw @ c_world + t_cw
            if pl[0] + rb < near or pl[0] - rb > far:
                return False, 0.0
            xr = max(pl[0] + rb, 1e-6)
            if abs(pl[1]) - rb > xr * tan_h or abs(pl[2]) - rb > xr * tan_v:
                return False, 0.0
            return True, max(near, pl[0] - rb)

        for sc in self.splats:
            pts_list, nrm_list, sp_list, rho_list = [], [], [], []
            n_chunks = max(1, int(math.ceil(sc.length / self.chunk_len)))
            cl = sc.length / n_chunks
            rb = math.hypot(sc.radius, 0.5 * cl)
            for k in range(n_chunks):
                s0 = -0.5 * sc.length + (k + 0.5) * cl
                ck = sc.center + s0 * sc.axis
                vis, dmin = visible(ck, rb)
                if not vis:
                    continue
                sp = max(self.min_spacing, self.density_px * dmin / cam.fx)
                ns = int(min(400, max(1, math.ceil(cl / sp))))
                nt = int(min(720, max(12, math.ceil(2.0 * math.pi * sc.radius / sp))))
                s = s0 + (np.arange(ns) + 0.5) / ns * cl - 0.5 * cl
                th = (np.arange(nt) + 0.5) / nt * 2.0 * math.pi
                radial = np.cos(th)[:, None] * sc.u + np.sin(th)[:, None] * sc.v            # (nt,3)
                P = (sc.center[None, None, :] + s[:, None, None] * sc.axis[None, None, :]
                     + sc.radius * radial[None, :, :]).reshape(-1, 3)
                Nr = np.broadcast_to(radial[None, :, :], (ns, nt, 3)).reshape(-1, 3)
                front = np.einsum("ij,ij->i", Nr, cam_pos[None, :] - P) > 0.0
                pts_list.append(P[front])
                nrm_list.append(Nr[front])
                sp_list.append(np.full(int(front.sum()), max(cl / ns, 2.0 * math.pi * sc.radius / nt)))
                rho_list.append(np.abs(s[:, None].repeat(nt, 1).reshape(-1)[front]) / max(0.5 * sc.length, 1e-9))
            if sc.caps:
                for sgn in (-1.0, 1.0):
                    e = sc.center + sgn * 0.5 * sc.length * sc.axis
                    nrm = sgn * sc.axis
                    if nrm @ (cam_pos - e) <= 0.0:
                        continue
                    vis, dmin = visible(e, sc.radius)
                    if not vis:
                        continue
                    sp = max(self.min_spacing, self.density_px * dmin / cam.fx)
                    nr = int(min(200, max(1, math.ceil(sc.radius / sp))))
                    rad = (np.arange(nr) + 0.5) / nr * sc.radius
                    pts, rh = [], []
                    for rr in rad:
                        nt = int(min(720, max(6, math.ceil(2.0 * math.pi * rr / sp))))
                        th = (np.arange(nt) + 0.5) / nt * 2.0 * math.pi
                        pts.append(e[None, :] + rr * (np.cos(th)[:, None] * sc.u + np.sin(th)[:, None] * sc.v))
                        rh.append(np.full(nt, rr / sc.radius))
                    P = np.concatenate(pts)
                    pts_list.append(P)
                    nrm_list.append(np.broadcast_to(nrm, P.shape).copy())
                    sp_list.append(np.full(len(P), sc.radius / nr))
                    rho_list.append(np.concatenate(rh))
            if not pts_list:
                continue
            P = np.concatenate(pts_list)
            if len(P) == 0:
                continue
            Nn = np.concatenate(nrm_list)
            rho = np.concatenate(rho_list)
            if sc.hot >= 0:
                h = self.hot[sc.hot]
                T = h.temp_k - self.edge_drop * (h.temp_k - h.base_k) * np.clip(rho, 0.0, 1.0) ** 2
                T = self._halo(P, T, np.ones(len(P), bool), skip_hot=sc.hot)
            else:
                T = self._halo(P, np.full(len(P), sc.base_k), np.full(len(P), sc.connected))
                if sc.kind in ("pipe", "stub"):   # slight lengthwise variation of the pipe body
                    T = T + 0.4 * np.sin(1.1 * (P @ sc.axis))
            Ps.append(P)
            Ns.append(Nn)
            Ss.append(np.concatenate(sp_list))
            Ls.append(np.full(len(P), sc.label, np.int32))
            Ts.append(T)
        if not Ps:
            z = np.zeros((0, 3))
            return z, z, np.zeros(0), np.zeros(0, np.int32), np.zeros(0)
        return (np.concatenate(Ps), np.concatenate(Ns), np.concatenate(Ss),
                np.concatenate(Ls), np.concatenate(Ts))

    # ------------------------------------------------------------ render
    def render(self, T_world_cam: np.ndarray, cam: PinholeCamera, near: float = 0.10,
               far: float = 60.0) -> RenderBuffers:
        """Render depth / label / apparent temperature for a camera pose
        (4x4 world <- camera link, Gazebo camera convention)."""
        Hh, W = cam.height, cam.width
        R_wc, o = T_world_cam[:3, :3], T_world_cam[:3, 3].astype(float)
        T_cw = np.linalg.inv(T_world_cam)
        # the pixel rays in the camera link frame only depend on the intrinsics:
        # computed once per camera (saves ~15 % of the frame time at 5 Hz)
        key = (W, Hh, cam.fx, cam.fy, cam.cx, cam.cy)
        d_link = self._ray_cache.get(key)
        if d_link is None:
            us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(Hh) + 0.5)
            d_link = cam.pixel_ray_link(us, vs).reshape(-1, 3)
            d_link.setflags(write=False)
            self._ray_cache[key] = d_link
        D = d_link @ R_wc.T
        def subset(corners: np.ndarray):
            """Indices of the pixel rays inside the screen rectangle of the
            projected corners (None if a corner is behind the camera)."""
            pl = (corners - o[None, :]) @ R_wc
            if np.any(pl[:, 0] < near):
                return None if np.any(pl[:, 0] > 0.0) else np.zeros(0, np.int64)
            u = cam.cx - cam.fx * pl[:, 1] / pl[:, 0]
            v = cam.cy - cam.fy * pl[:, 2] / pl[:, 0]
            u0, u1 = max(0, int(math.floor(u.min()))), min(W - 1, int(math.ceil(u.max())))
            v0, v1 = max(0, int(math.floor(v.min()))), min(Hh - 1, int(math.ceil(v.max())))
            if u0 > u1 or v0 > v1:
                return np.zeros(0, np.int64)
            return (np.arange(v0, v1 + 1)[:, None] * W + np.arange(u0, u1 + 1)[None, :]).ravel()

        t, label, normal = self._background(o, D, subset)
        t = np.where(np.isfinite(t), t, far)
        P = o[None, :] + t[:, None] * D
        depth = t * d_link[:, 0]
        temp = self._background_temperature(P, label)

        # ---- splatting with z-buffer
        Ps, Ns, Ss, Ls, Ts = self._splat_samples(o, T_cw, cam, near, far)
        n_spl = len(Ps)
        if n_spl:
            pl = (Ps - o[None, :]) @ R_wc          # world -> camera link: R_wc^T (p - o)
            uv, dz = cam.project_link(pl)
            ok = (dz > near) & (dz < far) & np.isfinite(uv[:, 0])
            idx_pts = np.nonzero(ok)[0]
            uv, dz = uv[ok], dz[ok]
            k = np.clip(np.ceil(Ss[ok] * cam.fx / dz + 0.5), 1, self.max_splat).astype(np.int32)
            all_pix, all_dep, all_src = [], [], []
            for kk in np.unique(k):
                sel = np.nonzero(k == kk)[0]
                bu = np.floor(uv[sel, 0] - 0.5 * (kk - 1)).astype(np.int64)
                bv = np.floor(uv[sel, 1] - 0.5 * (kk - 1)).astype(np.int64)
                for du in range(kk):
                    for dv in range(kk):
                        pu, pv = bu + du, bv + dv
                        inb = (pu >= 0) & (pu < W) & (pv >= 0) & (pv < Hh)
                        all_pix.append(pv[inb] * W + pu[inb])
                        all_dep.append(dz[sel][inb])
                        all_src.append(idx_pts[sel][inb])
            if all_pix:
                pix = np.concatenate(all_pix)
                dep = np.concatenate(all_dep)
                src = np.concatenate(all_src)
                order = np.lexsort((dep, pix))
                pix, dep, src = pix[order], dep[order], src[order]
                first = np.ones(len(pix), bool)
                first[1:] = pix[1:] != pix[:-1]
                pix, dep, src = pix[first], dep[first], src[first]
                win = dep < depth[pix] - 1e-4
                pix, dep, src = pix[win], dep[win], src[win]
                depth[pix] = dep
                label[pix] = Ls[src]
                normal[pix] = Ns[src]
                P[pix] = Ps[src]
                temp[pix] = Ts[src]

        # ---- directional emissivity: grazing surfaces reflect the ambient
        cosang = np.abs(np.einsum("ij,ij->i", normal, D))
        eps_rel = 1.0 - (1.0 - np.clip(cosang, 0.0, 1.0)) ** self.emissivity_power
        temp = self.ambient_k + eps_rel * (temp - self.ambient_k)
        return RenderBuffers(depth.reshape(Hh, W), label.reshape(Hh, W), temp.reshape(Hh, W).astype(np.float32),
                             normal.reshape(Hh, W, 3), P.reshape(Hh, W, 3), n_spl)


# =============================================================================
# Sensor model
# =============================================================================
class SyntheticThermalCamera:
    """Thermal camera model on top of ``ThermalScene`` (defaults from
    ``robot_model.SENSORS['thermal_camera']``)."""

    def __init__(self, layout, width: Optional[int] = None, height: Optional[int] = None,
                 hfov: Optional[float] = None, sensor_name: str = "thermal_camera",
                 netd_k: float = 0.05, column_fpn_k: float = 0.03, row_noise_k: float = 0.02,
                 blur_sigma_px: float = 0.6, resolution: float = THERMAL_RESOLUTION,
                 seed: int = 0, scene: Optional[ThermalScene] = None, **scene_kwargs):
        s = SENSORS[sensor_name]
        self.layout = layout
        self.sensor_name = sensor_name
        self.cam = PinholeCamera(width or s["width"], height or s["height"], hfov or s["hfov"])
        self.near = float(s.get("near", 0.1))
        self.far = float(s.get("far", 50.0))
        self.t_lo = float(s.get("min_temp", 253.15))
        self.t_hi = float(s.get("max_temp", 655.0))
        self.netd_k = float(netd_k)
        self.row_noise_k = float(row_noise_k)
        self.blur_sigma_px = float(blur_sigma_px)
        self.resolution = float(resolution)
        self.scene = scene or ThermalScene(layout, **scene_kwargs)
        self.rng = np.random.default_rng(seed)
        self.column_fpn = self.rng.normal(0.0, column_fpn_k, self.cam.width).astype(np.float32)
        self.last_buffers: Optional[RenderBuffers] = None

    def camera_pose(self, robot_pose: Sequence[float]) -> np.ndarray:
        if len(robot_pose) == 3:
            robot_pose = full_pose(self.layout, *robot_pose)
        return camera_world_pose(tuple(float(v) for v in robot_pose), self.sensor_name)

    def render_kelvin(self, robot_pose: Sequence[float], noise: bool = True) -> np.ndarray:
        """Radiometric frame in Kelvin (float32, H x W).  ``robot_pose`` is
        (x, y, z, roll, pitch, yaw) of base_link or (x, y, yaw)."""
        buf = self.scene.render(self.camera_pose(robot_pose), self.cam, self.near, self.far)
        self.last_buffers = buf
        T = buf.temp.astype(np.float32)
        if self.blur_sigma_px > 0:
            T = cv2.GaussianBlur(T, (0, 0), self.blur_sigma_px)
        if noise:
            T = (T + self.rng.normal(0.0, self.netd_k, T.shape).astype(np.float32)
                 + self.column_fpn[None, :]
                 + self.rng.normal(0.0, self.row_noise_k, (T.shape[0], 1)).astype(np.float32))
        return np.clip(T, self.t_lo, self.t_hi).astype(np.float32)

    def render_l16(self, robot_pose: Sequence[float], noise: bool = True) -> np.ndarray:
        """L16 frame (uint16 counts, Kelvin = count * resolution)."""
        T = self.render_kelvin(robot_pose, noise)
        return np.clip(np.round(T / self.resolution), 1, 65535).astype(np.uint16)


# =============================================================================
# Visible-light proxy (benchmark only: gives the ECC registration a target)
# =============================================================================
def render_visible_proxy(scene: ThermalScene, robot_pose: Sequence[float], sensor_name: str = "inspection_camera",
                         cam: Optional[PinholeCamera] = None, rng: Optional[np.random.Generator] = None,
                         noise_dn: float = 2.0, light_dir: Sequence[float] = (0.35, 0.25, 0.90)
                         ) -> Tuple[np.ndarray, RenderBuffers]:
    """Simple shaded RGB rendering of the same scene from the inspection
    camera (Lambert shading, albedo per surface class, floor grid lines,
    panel seams, sensor noise).  Returns (bgr uint8, buffers)."""
    cam = cam or PinholeCamera.from_sensor(sensor_name)
    if len(robot_pose) == 3:
        robot_pose = full_pose(scene.layout, *robot_pose)
    T_wc = camera_world_pose(tuple(float(v) for v in robot_pose), sensor_name)
    buf = scene.render(T_wc, cam, 0.05, 60.0)
    lab = buf.label.ravel()
    alb = np.zeros((lab.size, 3))
    kinds = np.array([scene.label_kind.get(int(l), "wall") for l in np.unique(lab)])
    for l, kind in zip(np.unique(lab), kinds):
        alb[lab == l] = _ALBEDO.get(str(kind), (0.6, 0.6, 0.6))
    L = np.asarray(light_dir, float)
    L /= np.linalg.norm(L)
    n = buf.normal.reshape(-1, 3)
    shade = 0.42 + 0.58 * np.abs(n @ L)
    P = buf.point.reshape(-1, 3)
    # floor tiles (1 m grid lines), wall panel seams, machine panel seams
    fl = lab == LBL_FLOOR
    fx, fy = np.abs(P[fl, 0] - np.round(P[fl, 0])), np.abs(P[fl, 1] - np.round(P[fl, 1]))
    shade[fl] *= np.where((fx < 0.02) | (fy < 0.02), 0.55, 1.0)
    wl = lab == LBL_WALL
    shade[wl] *= np.where(np.abs(P[wl, 2] - np.round(P[wl, 2] / 1.5) * 1.5) < 0.03, 0.6, 1.0)
    mach = (lab >= LBL_MACHINE) & (lab < LBL_HOT)
    seam = ((np.abs(P[mach, 0] * 2 - np.round(P[mach, 0] * 2)) < 0.03)
            | (np.abs(P[mach, 2] * 2 - np.round(P[mach, 2] * 2)) < 0.03))
    shade[mach] *= np.where(seam, 0.6, 1.0)
    img = (alb * shade[:, None] * 255.0).reshape(cam.height, cam.width, 3)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.7)
    if noise_dn > 0:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0.0, noise_dn, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), buf
