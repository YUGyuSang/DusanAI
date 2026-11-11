"""
Viewpoint-based patrol route planning for gauge inspection (ROS free).

1. **Candidate viewpoints** per gauge: robot base positions on rings around
   the dial (horizontal *camera*-to-dial distance ``ring_min .. ring_max``,
   default 0.9 - 2.2 m) within +-``angle_max_deg`` (40 deg) of the dial normal;
   the yaw makes the inspection camera look straight at the dial.
2. **Readability score** of a candidate, from the camera model of
   ``dusan_common.camera`` at the camera height of that floor level
   (``Layout.floor_z`` + stand height + camera mount):

       hard constraints : free (robot radius), reachable from the dock,
                          line of sight on the nav map, whole dial inside the
                          image (margin), incidence <= incidence_max
       soft terms (0..1): f_px   pixel diameter of the projected dial rim
                          f_inc  (cos(theta) - cos(theta_max)) / (1 - cos(theta_max))
                          f_vert 1 - (v_offset / (image half height))^2
                                 (camera pitch -0.30 rad -> the dial must be
                                 centred vertically, not only "in view")
                          f_clr  clearance ramp (robot radius .. clearance_good)
       score = sum_i w_i f_i / sum_i w_i  - zone penalty

3. **Best viewpoint per gauge** = argmax score (with a relaxed second pass
   for gauges without any valid candidate).
4. **Visiting order**: exact Held-Karp dynamic programme (O(2^n n^2), up to
   ``max_exact_nodes`` nodes incl. the dock) over the A* path lengths of
   ``dusan_common.planning.plan`` (robot radius 0.30 m) between all pairs of
   nodes, starting at the dock pose, optionally returning to it.  The naive
   layout order is evaluated with the same distance matrix for comparison.
5. **Terrain cost** (``terrain_cost_enabled``, default on): the crossable debris of
   the layout (pipes, rubble high cost; plates, the low platform small cost;
   margin <= 0.25 m, ``dusan_common.terrain_cost``) is an extra A* cell cost of
   every route leg, so legs go around pipes / rubble where the map leaves a
   free detour and cross them only where none exists.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dusan_common.camera import PinholeCamera, camera_world_pose, world_to_camera_link
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.planning import path_length, plan as astar_plan
from dusan_common.terrain_cost import TERRAIN_COST_DEFAULTS, TerrainCostParams, terrain_cost_grid
from dusan_common.robot_model import NOMINAL_BODY_HEIGHT, SENSORS


# =============================================================================
# Parameters / results
# =============================================================================
@dataclass
class ViewpointParams:
    ring_min: float = 0.9              # horizontal camera -> dial distance (m)
    ring_max: float = 2.2
    ring_step: float = 0.1
    angle_max_deg: float = 40.0        # around the dial normal
    angle_step_deg: float = 5.0
    robot_radius: float = 0.30
    body_height: float = NOMINAL_BODY_HEIGHT
    camera: str = "inspection_camera"
    image_margin_px: float = 12.0
    px_min: float = 60.0               # dial diameter (px) with zero pixel score
    px_good: float = 170.0             # dial diameter (px) with full pixel score
    incidence_max_deg: float = 60.0
    clearance_good: float = 0.8
    los_ignore_m: float = 0.35         # last part of the sight line (gauge stand) not checked
    w_pixel: float = 0.35
    w_incidence: float = 0.25
    w_vertical: float = 0.25
    w_clearance: float = 0.15
    avoid_zone_ids: Tuple[str, ...] = ("Z-HOT",)
    zone_penalty: float = 0.10
    fallback_ring_max: float = 3.0     # relaxed second pass
    fallback_angle_max_deg: float = 60.0
    return_to_dock: bool = True
    max_exact_nodes: int = 13          # Held-Karp limit (dock + 12 gauges)
    smooth_paths: bool = True
    # terrain cost of the route legs (dusan_common.terrain_cost, same names in every node)
    terrain_cost_enabled: bool = bool(TERRAIN_COST_DEFAULTS["terrain_cost_enabled"])
    terrain_w_pipe: float = float(TERRAIN_COST_DEFAULTS["terrain_w_pipe"])
    terrain_w_rubble: float = float(TERRAIN_COST_DEFAULTS["terrain_w_rubble"])
    terrain_w_plate: float = float(TERRAIN_COST_DEFAULTS["terrain_w_plate"])
    terrain_w_platform: float = float(TERRAIN_COST_DEFAULTS["terrain_w_platform"])
    terrain_margin: float = float(TERRAIN_COST_DEFAULTS["terrain_margin"])
    terrain_margin_fraction: float = float(TERRAIN_COST_DEFAULTS["terrain_margin_fraction"])

    @classmethod
    def from_dict(cls, d: dict) -> "ViewpointParams":
        names = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in names}
        if "avoid_zone_ids" in kw:
            kw["avoid_zone_ids"] = tuple(str(z) for z in (kw["avoid_zone_ids"] or []) if str(z))
        return cls(**kw)


@dataclass
class Viewpoint:
    gauge_id: str
    x: float
    y: float
    yaw: float
    score: float = 0.0
    valid: bool = False
    reason: str = ""
    ring: float = 0.0                  # horizontal camera -> dial distance
    angle_deg: float = 0.0             # offset from the dial normal
    distance: float = math.nan         # 3-D camera -> dial centre
    incidence_deg: float = math.nan
    pixel_diameter: float = math.nan   # major axis of the projected dial (px)
    pixel_minor: float = math.nan
    v_offset_px: float = math.nan      # dial centre below (+) / above (-) the principal point
    elevation_deg: float = math.nan    # dial elevation relative to the optical axis
    clearance: float = math.nan
    components: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        def r(v, n=3):
            return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), n)
        return {
            "gauge_id": self.gauge_id, "x": r(self.x), "y": r(self.y), "yaw": r(self.yaw, 4),
            "score": r(self.score, 4), "distance": r(self.distance), "incidence_deg": r(self.incidence_deg, 2),
            "pixel_diameter": r(self.pixel_diameter, 1), "pixel_minor": r(self.pixel_minor, 1),
            "elevation_deg": r(self.elevation_deg, 2), "clearance": r(self.clearance),
            "components": {k: r(v, 3) for k, v in self.components.items()},
        }


@dataclass
class PatrolPlan:
    start: Tuple[float, float, float]
    viewpoints: List[Viewpoint]                   # in visiting order
    order: List[str]
    legs: List[List[Tuple[float, float]]]
    leg_lengths: List[float]
    total_length: float
    return_to_dock: bool
    naive_order: List[str]
    naive_length: float
    unreachable: List[str]
    distance_matrix: np.ndarray
    node_ids: List[str]                           # 'start' + gauge ids (matrix order)
    method: str = "held_karp"
    compute_s: float = 0.0
    candidates: Dict[str, List[Viewpoint]] = field(default_factory=dict)

    def to_json(self, tags: Optional[Dict[str, str]] = None) -> dict:
        """Payload of /patrol/plan_json (DESIGN_SPEC section 4 + extras)."""
        vps = []
        for i, vp in enumerate(self.viewpoints):
            d = vp.to_json()
            d["order"] = i + 1
            d["leg_length"] = round(float(self.leg_lengths[i]), 3) if i < len(self.leg_lengths) else None
            if tags:
                d["tag"] = tags.get(vp.gauge_id)
            vps.append(d)
        imp = (100.0 * (self.naive_length - self.total_length) / self.naive_length
               if math.isfinite(self.naive_length) and self.naive_length > 0 else None)
        return {
            "start": [round(self.start[0], 3), round(self.start[1], 3)],
            "start_yaw": round(self.start[2], 4),
            "viewpoints": vps,
            "order": list(self.order),
            "legs": [[[round(float(x), 3), round(float(y), 3)] for x, y in leg] for leg in self.legs],
            "leg_lengths": [round(float(v), 3) for v in self.leg_lengths],
            "total_length": round(float(self.total_length), 3),
            "return_to_dock": bool(self.return_to_dock),
            "naive_order": list(self.naive_order),
            "naive_length": None if not math.isfinite(self.naive_length) else round(float(self.naive_length), 3),
            "improvement_pct": None if imp is None else round(imp, 2),
            "unreachable": list(self.unreachable),
            "method": self.method,
            "compute_s": round(float(self.compute_s), 3),
        }

    def route_points(self) -> List[Tuple[float, ...]]:
        """Concatenated legs; every viewpoint appears as (x, y, yaw) so that
        a Path built from it carries the viewing yaw."""
        out: List[Tuple[float, ...]] = []
        for i, leg in enumerate(self.legs):
            pts = list(leg)
            if out and pts and math.hypot(pts[0][0] - out[-1][0], pts[0][1] - out[-1][1]) < 1e-6:
                pts = pts[1:]
            out.extend((float(p[0]), float(p[1])) for p in pts)
            if i < len(self.viewpoints) and out:
                vp = self.viewpoints[i]
                out[-1] = (vp.x, vp.y, vp.yaw)
        return out


# =============================================================================
# Small maths helpers
# =============================================================================
def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _dial_frame(normal: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, u, w): unit normal, horizontal in-plane axis, second in-plane axis."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    u = np.cross([0.0, 0.0, 1.0], n)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    return n, u, w


def held_karp(D: np.ndarray, return_to_start: bool = True) -> Tuple[List[int], float]:
    """Exact shortest Hamiltonian path/cycle starting at node 0.

    dp[S][j] = length of the shortest path that starts at node 0, visits the
    node set S (bitmask over nodes 1..n-1) and ends at j in S:

        dp[{j}][j]   = D[0, j]
        dp[S + k][k] = min_j dp[S][j] + D[j, k]          (k not in S)
        answer       = min_j dp[ALL][j] (+ D[j, 0] for a closed tour)

    O(2^(n-1) (n-1)^2) time; the inner minimisation over k is vectorised.
    Returns (order of node indices starting with 0, cost); cost = inf when no
    finite tour exists.
    """
    D = np.asarray(D, dtype=float)
    n = len(D)
    if n <= 1:
        return [0], 0.0
    m = n - 1
    full = 1 << m
    dp = np.full((full, m), np.inf)
    par = np.full((full, m), -1, dtype=np.int64)
    for j in range(m):
        dp[1 << j, j] = D[0, j + 1]
    bits = 1 << np.arange(m)
    Dm = D[1:, 1:]
    for mask in range(1, full):
        row = dp[mask]
        if not np.any(np.isfinite(row)):
            continue
        free = (mask & bits) == 0
        if not np.any(free):
            continue
        ks = np.nonzero(free)[0]
        for j in np.nonzero(((mask & bits) != 0) & np.isfinite(row))[0]:
            vals = row[j] + Dm[j, ks]
            nm = mask | bits[ks]
            better = vals < dp[nm, ks]
            if np.any(better):
                dp[nm[better], ks[better]] = vals[better]
                par[nm[better], ks[better]] = j
    last = dp[full - 1] + (D[1:, 0] if return_to_start else 0.0)
    j = int(np.argmin(last))
    cost = float(last[j])
    if not math.isfinite(cost):
        return [0], math.inf
    order = []
    mask = full - 1
    while j >= 0:
        order.append(j + 1)
        pj = int(par[mask, j])
        mask &= ~(1 << j)
        j = pj
    order.append(0)
    order.reverse()
    return order, cost


def tour_cost(D: np.ndarray, order: Sequence[int], return_to_start: bool = True) -> float:
    c = sum(float(D[a, b]) for a, b in zip(order[:-1], order[1:]))
    if return_to_start and len(order) > 1:
        c += float(D[order[-1], order[0]])
    return c


def brute_force_tour(D: np.ndarray, return_to_start: bool = True) -> Tuple[List[int], float]:
    """Reference solver (tests): all permutations of nodes 1..n-1."""
    n = len(D)
    best, best_c = [0], math.inf
    for perm in itertools.permutations(range(1, n)):
        o = [0, *perm]
        c = tour_cost(D, o, return_to_start)
        if c < best_c:
            best, best_c = o, c
    return best, best_c


def nearest_neighbour_2opt(D: np.ndarray, return_to_start: bool = True) -> Tuple[List[int], float]:
    """Heuristic fallback for instances larger than the Held-Karp limit."""
    n = len(D)
    order, left = [0], set(range(1, n))
    while left:
        k = min(left, key=lambda j: D[order[-1], j])
        order.append(k)
        left.remove(k)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                cand = order[:i] + order[i:k + 1][::-1] + order[k + 1:]
                if tour_cost(D, cand, return_to_start) < tour_cost(D, order, return_to_start) - 1e-9:
                    order, improved = cand, True
    return order, tour_cost(D, order, return_to_start)


# =============================================================================
# Map from the layout (fallback when the generated nav map is missing)
# =============================================================================
def layout_occupancy_map(layout, resolution: float = 0.10) -> OccupancyGridMap:
    """2-D occupancy grid rasterised from ``world_layout.yaml`` following the
    DESIGN_SPEC rules (walls / machines / drums / pallets / rack supports /
    gauge stands / drop-offs occupied; stairs, ramp and low debris free).
    Used by tests and as a fallback of the evaluation when
    ``simulation/maps/plant_map.yaml`` has not been generated yet."""
    nm = layout.get("nav_map", {}) or {}
    x0, y0, x1, y1 = nm.get("bounds", [-10.6, -9.6, 64.2, 9.6])
    res = float(resolution)
    w = int(math.ceil((x1 - x0) / res))
    h = int(math.ceil((y1 - y0) / res))
    occ = np.full((h, w), 100, dtype=np.int8)
    xs = x0 + (np.arange(w) + 0.5) * res
    ys = y0 + (np.arange(h) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)

    def rect(xa, ya, xb, yb, val):
        m = (X >= min(xa, xb)) & (X <= max(xa, xb)) & (Y >= min(ya, yb)) & (Y <= max(ya, yb))
        occ[m] = val

    def disc(cx, cy, r, val):
        occ[(X - cx) ** 2 + (Y - cy) ** 2 <= r * r] = val

    def obox(cx, cy, hx, hy, yaw, val):
        c, s = math.cos(yaw), math.sin(yaw)
        lx = (X - cx) * c + (Y - cy) * s
        ly = -(X - cx) * s + (Y - cy) * c
        occ[(np.abs(lx) <= hx) & (np.abs(ly) <= hy)] = val

    fi = layout["factory"]["interior"]
    rect(fi["x_min"], fi["y_min"], fi["x_max"], fi["y_max"], 0)
    c = layout.get("corridor")
    if c:
        ci = c["interior"]
        rect(ci["x_min"], ci["y_min"], ci["x_max"], ci["y_max"], 0)
        rect(fi["x_max"] - 0.05, ci["y_min"], ci["x_min"] + 0.05, ci["y_max"], 0)   # east opening
        de = c["descent"]
        free_bands = [(c["stairs"]["y_min"], c["stairs"]["y_max"]), (c["ramp"]["y_min"], c["ramp"]["y_max"])]
        band = (X > de["x_min"]) & (X < de["x_max"]) & (Y >= ci["y_min"]) & (Y <= ci["y_max"])
        ok = np.zeros_like(band)
        for ya, yb in free_bands:
            ok |= (Y >= ya) & (Y <= yb)
        occ[band & ~ok] = 100                                                        # drop-offs
        drop = c.get("dropoff")              # explicit edge (ramp south edge above the stairs)
        if drop:
            rect(drop["x_min"], drop["y_min"], drop["x_max"], drop["y_max"], 100)
    for m in layout.get("machines", []) or []:
        fp = m["footprint"]
        rect(fp["x_min"], fp["y_min"], fp["x_max"], fp["y_max"], 100)
    pr = layout.get("pipe_rack") or {}
    if pr.get("pipe"):
        sx, sy, _ = pr.get("support_size", [0.12, 0.5, 1.5])
        yc = pr["pipe"]["start"][1]
        for x in pr.get("supports_x", []):
            rect(x - 0.5 * sx, yc - 0.5 * sy, x + 0.5 * sx, yc + 0.5 * sy, 100)
    stub = layout.get("corridor_pipe_stub")
    if stub:
        disc(stub["position"][0], stub["position"][1], max(stub.get("radius", 0.06), 0.08), 100)
    for o in layout.obstacles:
        t = o.get("type")
        if t == "drum":
            disc(o["center"][0], o["center"][1], float(o.get("radius", 0.3)), 100)
        elif t == "pallet":
            obox(o["center"][0], o["center"][1], 0.61, 0.70, float(o.get("yaw", 0.0)), 100)
    for g in layout.gauges:                     # gauge stand: small post behind the dial
        n = np.asarray(g.get("normal", [1, 0, 0]), float)
        p = np.asarray(g["position"], float) - 0.08 * n
        rect(p[0] - 0.1, p[1] - 0.1, p[0] + 0.1, p[1] + 0.1, 100)
    gl = layout.get("gas_leak") or {}
    if gl.get("valve_prop"):
        vx, vy = gl["valve_prop"][:2]
        rect(vx - 0.12, vy - 0.12, vx + 0.12, vy + 0.12, 100)
    return OccupancyGridMap(occ, res, x0, y0)


# =============================================================================
# Planner
# =============================================================================
class ViewpointPlanner:
    """Candidate generation, readability scoring, TSP ordering."""

    def __init__(self, layout, grid: OccupancyGridMap, params: Optional[ViewpointParams] = None):
        self.layout = layout
        self.grid = grid
        self.p = params or ViewpointParams()
        s = SENSORS[self.p.camera]
        self.cam = PinholeCamera(s["width"], s["height"], s["hfov"])
        self.cam_offset = np.asarray(s["pose"][:3], dtype=float)       # camera position in base_link
        self.gauges = {g["id"]: g for g in layout.gauges}
        self._reach_labels: Optional[np.ndarray] = None
        self._reach_label = -1
        self._dock_xy = tuple(layout.dock_pose[:2])
        self._path_cache: Dict[Tuple[Tuple[float, float], Tuple[float, float]], Optional[List[Tuple[float, float]]]] = {}
        self._zones = [z for z in (layout.get("zones") or []) if z.get("id") in set(self.p.avoid_zone_ids)]
        tprm = TerrainCostParams.from_params(vars(self.p))
        self.terrain_cost = terrain_cost_grid(grid, layout, tprm) if tprm.enabled else None

    # ------------------------------------------------------- reachability
    def reachable_mask_label(self, start_xy: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, int]:
        """Connected components of the free space inflated by the robot
        radius, and the label of the component of start_xy (default dock)."""
        import cv2
        start_xy = start_xy or self._dock_xy
        if self._reach_labels is None or start_xy != getattr(self, "_reach_start", None):
            free = (~self.grid.inflated_obstacles(self.p.robot_radius)).astype(np.uint8)
            _, labels = cv2.connectedComponents(free, connectivity=8)
            s = self.grid.nearest_free(start_xy[0], start_xy[1], self.p.robot_radius, 2.0)
            lab = -1
            if s is not None:
                ix, iy = self.grid.world_to_grid(*s)
                lab = int(labels[iy, ix])
            self._reach_labels, self._reach_label, self._reach_start = labels, lab, start_xy
        return self._reach_labels, self._reach_label

    def is_reachable(self, x: float, y: float) -> bool:
        labels, lab = self.reachable_mask_label()
        ix, iy = self.grid.world_to_grid(x, y)
        return lab > 0 and self.grid.in_bounds(ix, iy) and int(labels[iy, ix]) == lab

    def line_of_sight(self, x0: float, y0: float, gx: float, gy: float) -> bool:
        """Sight line camera -> dial on the nav map, ignoring the last
        ``los_ignore_m`` (the gauge's own stand is an occupied cell)."""
        d = math.hypot(gx - x0, gy - y0)
        L = d - self.p.los_ignore_m
        if L <= 0.0:
            return True
        n = max(2, int(L / (0.5 * self.grid.resolution)) + 1)
        ts = np.linspace(0.0, L / d, n)
        xs, ys = x0 + ts * (gx - x0), y0 + ts * (gy - y0)
        ix, iy = self.grid.world_to_grid_array(np.stack([xs, ys], -1))
        inb = (ix >= 0) & (ix < self.grid.width) & (iy >= 0) & (iy < self.grid.height)
        if not np.all(inb):
            return False
        return not bool(np.any(self.grid.occ[iy, ix] >= 50))

    def in_avoid_zone(self, x: float, y: float) -> bool:
        for z in self._zones:
            xa, ya, xb, yb = z["rect"]
            if min(xa, xb) <= x <= max(xa, xb) and min(ya, yb) <= y <= max(ya, yb):
                return True
        return False

    # -------------------------------------------------------- evaluation
    def evaluate(self, gauge: dict, x: float, y: float, yaw: float, ring: float = math.nan,
                 angle_deg: float = math.nan) -> Viewpoint:
        """Score the robot base pose (x, y, yaw) for reading ``gauge``."""
        p = self.p
        vp = Viewpoint(gauge["id"], float(x), float(y), float(yaw), ring=ring, angle_deg=angle_deg)
        g = np.asarray(gauge["position"], dtype=float)
        n, u, w = _dial_frame(gauge.get("normal", [1, 0, 0]))
        R = 0.5 * float(gauge.get("dial_diameter", 0.30))
        z_base = self.layout.floor_z(x, y) + p.body_height
        T_wc = camera_world_pose((x, y, z_base, 0.0, 0.0, yaw), p.camera)
        c_pos = T_wc[:3, 3]
        vp.distance = float(np.linalg.norm(g - c_pos))
        to_cam = (c_pos - g) / max(vp.distance, 1e-9)
        cos_inc = float(np.clip(n @ to_cam, -1.0, 1.0))
        vp.incidence_deg = math.degrees(math.acos(cos_inc))
        vp.clearance = self.grid.clearance(x, y)
        # projected dial rim
        th = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        rim = g[None, :] + R * (np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * w[None, :])
        pl = world_to_camera_link(T_wc, np.vstack([g, rim]))
        uv, depth = self.cam.project_link(pl)
        reasons = []
        if np.all(depth > 0.05):
            diam = np.linalg.norm(uv[1:13] - uv[13:25], axis=1)
            vp.pixel_diameter, vp.pixel_minor = float(diam.max()), float(diam.min())
            vp.v_offset_px = float(uv[0, 1] - self.cam.cy)
            vp.elevation_deg = math.degrees(math.atan2(pl[0, 2], pl[0, 0]))
            in_img = bool(np.all(self.cam.in_image(uv[1:], p.image_margin_px)))
        else:
            in_img = False
        # ---- hard constraints
        if not self.grid.is_free(x, y, p.robot_radius):
            reasons.append("not_free")
        elif not self.is_reachable(x, y):
            reasons.append("unreachable")
        if cos_inc <= 0.0 or vp.incidence_deg > p.incidence_max_deg:
            reasons.append("incidence")
        if not in_img:
            reasons.append("out_of_image")
        if not self.line_of_sight(c_pos[0], c_pos[1], g[0], g[1]):
            reasons.append("no_line_of_sight")
        # ---- soft terms
        f_px = smoothstep((vp.pixel_diameter - p.px_min) / max(p.px_good - p.px_min, 1e-6)) \
            if math.isfinite(vp.pixel_diameter) else 0.0
        cmax = math.cos(math.radians(p.incidence_max_deg))
        f_inc = float(np.clip((cos_inc - cmax) / max(1.0 - cmax, 1e-6), 0.0, 1.0))
        f_vert = float(np.clip(1.0 - (vp.v_offset_px / self.cam.cy) ** 2, 0.0, 1.0)) \
            if math.isfinite(vp.v_offset_px) else 0.0
        f_clr = float(np.clip((vp.clearance - p.robot_radius) / max(p.clearance_good - p.robot_radius, 1e-6), 0.0, 1.0))
        wsum = p.w_pixel + p.w_incidence + p.w_vertical + p.w_clearance
        score = (p.w_pixel * f_px + p.w_incidence * f_inc + p.w_vertical * f_vert + p.w_clearance * f_clr) / max(wsum, 1e-9)
        zone = self.in_avoid_zone(x, y)
        if zone:
            score -= p.zone_penalty
        vp.components = {"pixel": f_px, "incidence": f_inc, "vertical": f_vert, "clearance": f_clr,
                         "zone_penalty": p.zone_penalty if zone else 0.0}
        vp.valid = not reasons
        vp.reason = ",".join(reasons) if reasons else "ok"
        vp.score = float(max(score, 0.0)) if vp.valid else 0.0
        return vp

    def candidates(self, gauge: dict, ring_max: Optional[float] = None,
                   angle_max_deg: Optional[float] = None) -> List[Viewpoint]:
        """All candidate viewpoints of a gauge (valid and invalid)."""
        p = self.p
        ring_max = p.ring_max if ring_max is None else ring_max
        angle_max = p.angle_max_deg if angle_max_deg is None else angle_max_deg
        n = np.asarray(gauge.get("normal", [1, 0, 0]), float)
        nh = n[:2] if np.linalg.norm(n[:2]) > 1e-6 else np.array([1.0, 0.0])
        alpha = math.atan2(nh[1], nh[0])
        g = np.asarray(gauge["position"], float)
        cx, cy = float(self.cam_offset[0]), float(self.cam_offset[1])
        rings = np.arange(p.ring_min, ring_max + 1e-9, p.ring_step)
        n_ang = int(round(angle_max / p.angle_step_deg))
        angles = np.arange(-n_ang, n_ang + 1) * p.angle_step_deg
        out = []
        for r in rings:
            for a in angles:
                phi = alpha + math.radians(a)
                d_base = r + cx                                   # base -> dial (camera in front)
                bx, by = g[0] + d_base * math.cos(phi), g[1] + d_base * math.sin(phi)
                yaw0 = math.atan2(g[1] - by, g[0] - bx)
                yaw = yaw0 - math.asin(max(-1.0, min(1.0, cy / max(d_base, 1e-6))))   # lateral camera offset
                out.append(self.evaluate(gauge, bx, by, yaw, float(r), float(a)))
        return out

    def best_viewpoint(self, gauge: dict) -> Tuple[Optional[Viewpoint], List[Viewpoint]]:
        cands = self.candidates(gauge)
        valid = [c for c in cands if c.valid]
        if not valid:     # relaxed second pass
            extra = self.candidates(gauge, self.p.fallback_ring_max, self.p.fallback_angle_max_deg)
            cands = cands + [c for c in extra if c.ring > self.p.ring_max + 1e-9
                             or abs(c.angle_deg) > self.p.angle_max_deg + 1e-9]
            valid = [c for c in cands if c.valid]
        if not valid:
            return None, cands
        # tie-break: higher score, then shorter distance to the dial
        best = max(valid, key=lambda c: (round(c.score, 6), -c.ring))
        return best, cands

    # ---------------------------------------------------------- planning
    def path(self, a: Tuple[float, float], b: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """A* path (dusan_common.planning, with the terrain cost layer) from a to b,
        cached and exact at the ends."""
        key = (tuple(np.round(a, 3)), tuple(np.round(b, 3)))
        rkey = (key[1], key[0])
        if key in self._path_cache:
            return self._path_cache[key]
        if rkey in self._path_cache:
            p = self._path_cache[rkey]
            return None if p is None else p[::-1]
        p = astar_plan(self.grid, a, b, robot_radius=self.p.robot_radius, smooth=self.p.smooth_paths,
                       extra_cost=self.terrain_cost)
        if p is not None:
            p = [tuple(map(float, q)) for q in p]
            if math.hypot(p[0][0] - a[0], p[0][1] - a[1]) > 1e-3:
                p.insert(0, (float(a[0]), float(a[1])))
            if math.hypot(p[-1][0] - b[0], p[-1][1] - b[1]) > 1e-3:
                p.append((float(b[0]), float(b[1])))
        self._path_cache[key] = p
        return p

    def plan(self, start: Optional[Sequence[float]] = None, return_to_dock: Optional[bool] = None,
             gauge_ids: Optional[Sequence[str]] = None) -> PatrolPlan:
        """Full pipeline: best viewpoint per gauge -> distance matrix -> exact TSP."""
        t0 = time.time()
        p = self.p
        ret = p.return_to_dock if return_to_dock is None else bool(return_to_dock)
        start = tuple(start) if start is not None else tuple(self.layout.dock_pose)
        if len(start) == 2:
            start = (start[0], start[1], 0.0)
        ids = list(gauge_ids) if gauge_ids is not None else [g["id"] for g in self.layout.gauges]
        best: Dict[str, Viewpoint] = {}
        cands: Dict[str, List[Viewpoint]] = {}
        unreachable: List[str] = []
        for gid in ids:
            if gid not in self.gauges:          # unknown id (e.g. typo in the gauge_ids parameter)
                unreachable.append(gid)
                continue
            vp, cs = self.best_viewpoint(self.gauges[gid])
            cands[gid] = cs
            if vp is None:
                unreachable.append(gid)
            else:
                best[gid] = vp
        node_ids = ["start"] + [g for g in ids if g in best]
        pts = [(start[0], start[1])] + [(best[g].x, best[g].y) for g in node_ids[1:]]
        n = len(pts)
        D = np.full((n, n), np.inf)
        np.fill_diagonal(D, 0.0)
        for i in range(n):
            for j in range(i + 1, n):
                path = self.path(pts[i], pts[j])
                if path is not None:
                    D[i, j] = D[j, i] = path_length(path)
        # gauges whose viewpoint cannot be reached from the start are dropped
        keep = [0] + [i for i in range(1, n) if math.isfinite(D[0, i])]
        for i in range(1, n):
            if i not in keep:
                unreachable.append(node_ids[i])
        Dk = D[np.ix_(keep, keep)]
        # Every kept node is reachable from the start, so a pair without a direct
        # A* path (e.g. expansion limit, start snapped differently) can always be
        # travelled via the start: use that detour instead of an infinite edge,
        # otherwise Held-Karp would return no tour at all.
        via = np.zeros_like(Dk, dtype=bool)
        if np.any(~np.isfinite(Dk)):
            via = ~np.isfinite(Dk)
            Dk = np.where(via, Dk[:, :1] + Dk[:1, :], Dk)
        if len(keep) <= p.max_exact_nodes:
            order_k, cost = held_karp(Dk, ret)
            method = "held_karp"
        else:
            order_k, cost = nearest_neighbour_2opt(Dk, ret)
            method = "nearest_neighbour_2opt"
        order_idx = [keep[i] for i in order_k]
        visit = order_idx[1:]
        order_ids = [node_ids[i] for i in visit]
        legs, leg_len = [], []
        seq = order_idx + ([0] if ret and visit else [])      # nothing to visit -> no legs
        pos_k = {node: k for k, node in enumerate(keep)}
        for a, b in zip(seq[:-1], seq[1:]):
            path = self.path(pts[a], pts[b])
            if path is None and via[pos_k[a], pos_k[b]]:      # detour via the start (see above)
                p1, p2 = self.path(pts[a], pts[0]), self.path(pts[0], pts[b])
                path = (p1 + p2[1:]) if p1 is not None and p2 is not None else None
            path = path or [pts[a], pts[b]]
            legs.append(path)
            leg_len.append(path_length(path))
        # naive comparison: layout order with the same (detour-completed) distance matrix
        naive_idx = [0] + [i for i in keep[1:]]
        naive_len = tour_cost(Dk, list(range(len(keep))), ret)
        return PatrolPlan(start=(float(start[0]), float(start[1]), float(start[2])),
                          viewpoints=[best[g] for g in order_ids], order=order_ids, legs=legs,
                          leg_lengths=leg_len, total_length=float(sum(leg_len)), return_to_dock=ret,
                          naive_order=[node_ids[i] for i in naive_idx[1:]], naive_length=float(naive_len),
                          unreachable=unreachable, distance_matrix=D, node_ids=node_ids, method=method,
                          compute_s=time.time() - t0, candidates=cands)


def plan_patrol(layout, grid: OccupancyGridMap, params: Optional[ViewpointParams] = None,
                start: Optional[Sequence[float]] = None) -> PatrolPlan:
    """Convenience wrapper."""
    return ViewpointPlanner(layout, grid, params).plan(start)
