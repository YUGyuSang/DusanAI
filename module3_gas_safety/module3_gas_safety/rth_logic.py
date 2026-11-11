"""
Return-to-home (RTH) safety supervisor (Module 3, ROS free).

State machine
-------------
MONITOR --trigger--> RTH_PLANNING --A* ok--> RTH_NAVIGATING --near dock--> DOCKING
   ^                       ^   |                    |  (stuck 10 s -> re-plan)
   |                       +---+ (retry)            v
   +--- battery >= 95 % / 'resume' --- CHARGING <-- DOCKED ('sit')

Operator commands (/mission/command)
------------------------------------
'rth' / 'dock'   MONITOR: manual return home (trigger 'manual').
'pause'          RTH_PLANNING / RTH_NAVIGATING / DOCKING: hold the robot
                 (zero /cmd_vel/rth) until 'resume' (or 'rth' / 'dock'),
                 then plan again from the present pose.
'resume'         paused: continue; DOCKED / CHARGING: leave the charger.
'abort'          RTH_PLANNING / RTH_NAVIGATING / DOCKING: cancel the return
                 (back to MONITOR, rth_active false) unless an automatic
                 trigger condition holds right now (battery low, energy
                 reserve, comm shadow at the robot, comm lost) - the return
                 would re-trigger at once, so it continues (use 'pause' or
                 teleop to stop the robot).  Ignored on the dock.

Triggers (evaluated in MONITOR, highest priority first)
-------------------------------------------------------
manual                 /mission/command 'rth' (or 'dock')
battery_low            battery % < rth_battery_pct (debounced)
energy_reserve         E_home% * safety_factor + margin% > battery %, where
                       E_home is the energy to walk the geodesic distance
                       home at the nominal speed (BatteryModel power model)
comm_shadow_predicted  predicted RSSI < rssi_min along the next
                       ``shadow_lookahead_m`` of the followed path
                       (/nav/plan) or along the velocity extrapolation over
                       ``shadow_horizon_s`` (RSSI map lookup)
comm_lost (optional)   measured RSSI < rssi_min for > comm_lost_time_s

Navigation: A* (``dusan_common.planning.astar``) on the nav map with the
terrain cost layer of the layout debris (``dusan_common.terrain_cost``; pipes
and rubble are crossed only where no free detour exists; ``terrain_*``
parameters), smoothed with a clearance margin (cost-aware), pure pursuit (``dusan_common.pure_pursuit``) with a
velocity safety filter (the component towards the closest obstacle is
removed and a repulsion added when the clearance gets small) to the dock
area, then a holonomic P controller aligns position and yaw with the dock
pose.  No progress for ``stuck_time_s`` -> back off from the closest obstacle
and re-plan.  Novel (unmapped) obstacles seen in the elevation map
(``RthInputs.obstacle_frame``, dusan_common.obstacle_layer): the home drive
slows down / stops for evidence ahead, detours left or right around a
confirmed obstacle, re-plans on the static + detected map (full, then near
occlusion shadow; then on the map alone), and the safety filter / back-off
also respect detected obstacles (``obstacle_*`` parameters, ``obstacle`` in
/safety/status).  DOCKING starts only when the straight line to the dock is
free of detected obstacles; while docking, the governor slows / stops for
detected obstacles on that line, the safety filter keeps clear of them, and
no progress for ``stuck_time_s`` re-plans (a box in front of / on the dock
never makes the robot push into it).  On the dock the robot sits ('sit' on /locomotion/command) and
waits until the battery is recharged, then stands up ('stand') and returns
to MONITOR.
"""
from __future__ import annotations

import heapq
import inspect
import math
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from dusan_common.obstacle_layer import (OBSTACLE_DEFAULTS, EvidenceFrame, ObstacleAvoider, check_radius,
                                         path_blockage, speed_scale)
from dusan_common.planning import astar, densify, path_length, smooth_path
from dusan_common.pure_pursuit import PurePursuit
from dusan_common.terrain_cost import TERRAIN_COST_DEFAULTS, TerrainCostCache, TerrainCostParams

_OD = OBSTACLE_DEFAULTS

from .battery_model import BatteryModel

MONITOR, RTH_PLANNING, RTH_NAVIGATING, DOCKING, DOCKED, CHARGING = (
    "MONITOR", "RTH_PLANNING", "RTH_NAVIGATING", "DOCKING", "DOCKED", "CHARGING")
RTH_STATES = (MONITOR, RTH_PLANNING, RTH_NAVIGATING, DOCKING, DOCKED, CHARGING)
MOVING_STATES = (RTH_PLANNING, RTH_NAVIGATING, DOCKING)     # 'pause' / 'abort' act here
MANUAL_COMMANDS = ("rth", "dock", "return", "home")
REASONS = ("manual", "battery_low", "energy_reserve", "comm_shadow_predicted", "comm_lost")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class RthConfig:
    rth_battery_pct: float = 20.0
    resume_battery_pct: float = 95.0
    battery_debounce_s: float = 1.0
    energy_enabled: bool = True
    energy_safety_factor: float = 1.3
    energy_margin_pct: float = 3.0
    energy_debounce_s: float = 2.0
    nominal_speed: float = 0.30            # speed used to predict the energy home [m/s]
    docking_overhead_s: float = 30.0       # standing time budget for docking
    comm_shadow_enabled: bool = True
    rssi_min_dbm: float = -80.0
    shadow_lookahead_m: float = 5.0
    shadow_horizon_s: float = 4.0
    shadow_sample_m: float = 0.25
    plan_max_offset_m: float = 3.0         # ignore /nav/plan if the robot is farther away
    comm_lost_enabled: bool = True
    comm_lost_time_s: float = 2.0
    robot_radius: float = 0.30
    smooth_margin: float = 0.10            # path smoothing keeps robot_radius + margin from walls
    safety_dist: float = 0.50              # velocity safety filter acts below this clearance [m]
    safety_repulse: float = 0.10           # repulsion speed at half the robot radius [m/s]
    max_speed: float = 0.40
    max_wz: float = 0.8
    lookahead: float = 0.6
    goal_tolerance: float = 0.25
    docking_radius: float = 1.0
    dock_speed: float = 0.15
    dock_k_pos: float = 0.8
    dock_k_yaw: float = 1.2
    dock_pos_tolerance: float = 0.12
    dock_yaw_tolerance: float = 0.12
    dock_settle_s: float = 1.0
    sit_delay_s: float = 2.0
    stuck_time_s: float = 10.0
    stuck_progress_m: float = 0.3
    backoff_s: float = 2.0                 # back off from the closest obstacle before re-planning [s]
    backoff_speed: float = 0.15            # [m/s]
    replan_retry_s: float = 2.0
    max_plan_failures: int = 5
    geodesic_downsample: int = 2
    # terrain cost of the path home (dusan_common.terrain_cost, same names in every node)
    terrain_cost_enabled: bool = bool(TERRAIN_COST_DEFAULTS["terrain_cost_enabled"])
    terrain_w_pipe: float = float(TERRAIN_COST_DEFAULTS["terrain_w_pipe"])
    terrain_w_rubble: float = float(TERRAIN_COST_DEFAULTS["terrain_w_rubble"])
    terrain_w_plate: float = float(TERRAIN_COST_DEFAULTS["terrain_w_plate"])
    terrain_w_platform: float = float(TERRAIN_COST_DEFAULTS["terrain_w_platform"])
    terrain_margin: float = float(TERRAIN_COST_DEFAULTS["terrain_margin"])
    terrain_margin_fraction: float = float(TERRAIN_COST_DEFAULTS["terrain_margin_fraction"])
    # novel (unmapped) obstacles from the elevation map (dusan_common.obstacle_layer, same names in
    # every node; the str input keys obstacle_source / _topics / _frames are node parameters only)
    obstacle_enabled: bool = bool(_OD["obstacle_enabled"])
    obstacle_detour_enabled: bool = bool(_OD["obstacle_detour_enabled"])
    obstacle_replan_enabled: bool = bool(_OD["obstacle_replan_enabled"])
    obstacle_slowdown_enabled: bool = bool(_OD["obstacle_slowdown_enabled"])
    obstacle_static_fallback: bool = bool(_OD["obstacle_static_fallback"])
    obstacle_step_height: float = float(_OD["obstacle_step_height"])
    obstacle_step_window: int = int(_OD["obstacle_step_window"])
    obstacle_cost_threshold: int = int(_OD["obstacle_cost_threshold"])
    obstacle_max_range: float = float(_OD["obstacle_max_range"])
    obstacle_resolution: float = float(_OD["obstacle_resolution"])
    obstacle_memory_size: float = float(_OD["obstacle_memory_size"])
    obstacle_confirm_hits: int = int(_OD["obstacle_confirm_hits"])
    obstacle_max_hits: int = int(_OD["obstacle_max_hits"])
    obstacle_min_known_fraction: float = float(_OD["obstacle_min_known_fraction"])
    obstacle_ignore_radius: float = float(_OD["obstacle_ignore_radius"])
    obstacle_forget_time: float = float(_OD["obstacle_forget_time"])
    obstacle_min_cluster: int = int(_OD["obstacle_min_cluster"])
    obstacle_static_tolerance: float = float(_OD["obstacle_static_tolerance"])
    obstacle_shadow_depth: float = float(_OD["obstacle_shadow_depth"])
    obstacle_idle_period: float = float(_OD["obstacle_idle_period"])
    obstacle_margin: float = float(_OD["obstacle_margin"])
    obstacle_lookahead: float = float(_OD["obstacle_lookahead"])
    obstacle_detour_window: float = float(_OD["obstacle_detour_window"])
    obstacle_rejoin_gap: float = float(_OD["obstacle_rejoin_gap"])
    obstacle_detour_max_ratio: float = float(_OD["obstacle_detour_max_ratio"])
    obstacle_switch_ratio: float = float(_OD["obstacle_switch_ratio"])
    obstacle_commit_time: float = float(_OD["obstacle_commit_time"])
    obstacle_min_interval: float = float(_OD["obstacle_min_interval"])
    obstacle_max_detours: int = int(_OD["obstacle_max_detours"])
    obstacle_max_replans: int = int(_OD["obstacle_max_replans"])
    obstacle_slow_dist: float = float(_OD["obstacle_slow_dist"])
    obstacle_stop_dist: float = float(_OD["obstacle_stop_dist"])
    obstacle_unconfirmed_scale: float = float(_OD["obstacle_unconfirmed_scale"])


@dataclass
class RthInputs:
    t: float
    x: float
    y: float
    yaw: float
    vx: float = 0.0                        # world-frame velocity [m/s]
    vy: float = 0.0
    battery_pct: Optional[float] = None
    charging: Optional[bool] = None
    rssi_dbm: Optional[float] = None
    plan: Optional[Sequence[Tuple[float, float]]] = None
    command: Optional[str] = None
    obstacle_frame: Optional[EvidenceFrame] = None     # latest elevation-map evidence (pose frame)


@dataclass
class RthOutput:
    cmd: Optional[Tuple[float, float, float]] = None   # body (vx, vy, wz); None = do not publish
    locomotion: Optional[str] = None                   # one-shot /locomotion/command
    plan: Optional[List[Tuple[float, float]]] = None   # path to publish on /nav/plan ([] = clear, None = nothing)
    rth_active: bool = False
    status: dict = field(default_factory=dict)


# =============================================================================
# geodesic distance to the dock
# =============================================================================
class GeodesicField:
    """Dijkstra distance [m] from the goal over the free (inflated) nav map,
    computed once on a ``downsample``-times coarser grid (8-connected)."""

    def __init__(self, grid, goal_xy, robot_radius: float = 0.30, downsample: int = 2):
        self.grid = grid
        self.goal = (float(goal_xy[0]), float(goal_xy[1]))
        k = max(1, int(downsample))
        self.k = k
        blocked = grid.inflated_obstacles(robot_radius * 0.9)
        h, w = blocked.shape
        H, W = int(math.ceil(h / k)), int(math.ceil(w / k))
        pad = np.ones((H * k, W * k), dtype=bool)
        pad[:h, :w] = blocked
        # a coarse cell is free if any of its fine cells is free (keeps doorways open)
        self.free = ~pad.reshape(H, k, W, k).all(axis=(1, 3))
        self.res = grid.resolution * k
        self.ox, self.oy = grid.origin_x, grid.origin_y
        self.dist = np.full((H, W), np.inf)
        gi, gj = self._cell(*self.goal)
        if not (0 <= gi < W and 0 <= gj < H) or not self.free[gj, gi]:
            near = grid.nearest_free(self.goal[0], self.goal[1], robot_radius * 0.9, 3.0)
            if near is not None:
                gi, gj = self._cell(*near)
        if 0 <= gi < W and 0 <= gj < H:
            self._dijkstra(gj, gi)

    def _cell(self, x, y):
        return int(math.floor((x - self.ox) / self.res)), int(math.floor((y - self.oy) / self.res))

    def _dijkstra(self, gj: int, gi: int) -> None:
        H, W = self.free.shape
        free = self.free.ravel()
        dist = self.dist.ravel()
        r = self.res
        nb = [(-1, 0, r), (1, 0, r), (0, -1, r), (0, 1, r),
              (-1, -1, r * math.sqrt(2)), (-1, 1, r * math.sqrt(2)),
              (1, -1, r * math.sqrt(2)), (1, 1, r * math.sqrt(2))]
        s = gj * W + gi
        dist[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, c = heapq.heappop(pq)
            if d > dist[c]:
                continue
            cj, ci = divmod(c, W)
            for dj, di, cost in nb:
                nj, ni = cj + dj, ci + di
                if 0 <= nj < H and 0 <= ni < W:
                    n = nj * W + ni
                    if free[n] and d + cost < dist[n]:
                        dist[n] = d + cost
                        heapq.heappush(pq, (d + cost, n))
        self.dist = dist.reshape(H, W)

    def distance(self, x: float, y: float) -> float:
        """Geodesic distance home; inside obstacles/inflation: nearest finite
        neighbour within 1 m plus the offset, else 1.4 x Euclidean."""
        H, W = self.dist.shape
        i, j = self._cell(x, y)
        if 0 <= i < W and 0 <= j < H and math.isfinite(self.dist[j, i]):
            return float(self.dist[j, i])
        rr = int(math.ceil(1.0 / self.res))
        best = math.inf
        for dj in range(-rr, rr + 1):
            for di in range(-rr, rr + 1):
                jj, ii = j + dj, i + di
                if 0 <= jj < H and 0 <= ii < W and math.isfinite(self.dist[jj, ii]):
                    best = min(best, self.dist[jj, ii] + math.hypot(di, dj) * self.res)
        if math.isfinite(best):
            return float(best)
        return 1.4 * math.hypot(x - self.goal[0], y - self.goal[1])


# =============================================================================
# planners
# =============================================================================
def plan_home(grid, start, goal, robot_radius: float = 0.30, smooth_radius: Optional[float] = None,
              extra_cost=None):
    """A* + line-of-sight smoothing.  Smoothing with ``smooth_radius`` >
    ``robot_radius`` keeps the shortcut segments away from wall corners (the
    pure-pursuit chord would otherwise clip the corner); in passages narrower
    than that the smoother falls back to the raw A* cells.  ``extra_cost``
    (terrain cost layer) is used by the search and the smoother."""
    path = astar(grid, start, goal, robot_radius=robot_radius, extra_cost=extra_cost)
    if path is None:
        return None
    r = robot_radius if smooth_radius is None else max(smooth_radius, robot_radius * 0.9)
    return smooth_path(grid, path, r, extra_cost=extra_cost)


def terrain_cost_for(cfg: "RthConfig", layout) -> Optional[TerrainCostCache]:
    """Terrain cost layer (callable grid -> cost) of the layout debris, None when
    disabled or without a layout."""
    prm = TerrainCostParams.from_params(vars(cfg))
    if not prm.enabled or layout is None:
        return None
    return TerrainCostCache(layout, prm)


class SyncPlanner:
    """Synchronous A* (tests / offline evaluation).  ``cost_fn(grid)`` returns
    the extra A* cost layer (``terrain_cost_for``) or None."""

    def __init__(self, grid, robot_radius: float = 0.30, smooth_radius: Optional[float] = None,
                 cost_fn: Optional[Callable] = None):
        self.grid = grid
        self.robot_radius = robot_radius
        self.smooth_radius = robot_radius + 0.1 if smooth_radius is None else smooth_radius
        self.cost_fn = cost_fn
        self._result = None
        self._pending = False

    def _extra_cost(self, grid):
        return self.cost_fn(grid) if self.cost_fn is not None and grid is not None else None

    def submit(self, start, goal, grid=None) -> None:
        """Plan on ``grid`` (static + detected obstacles) or the nav map.  The
        terrain cost of the nav map is reused for a grid of the same geometry."""
        g = self.grid if grid is None else grid
        self._result = plan_home(g, start, goal, self.robot_radius, self.smooth_radius, self._cost_for(g))
        self._pending = True

    def _cost_for(self, g):
        base = self.grid
        if (g is not base and base is not None and g is not None and g.occ.shape == base.occ.shape
                and g.resolution == base.resolution and g.origin_x == base.origin_x and g.origin_y == base.origin_y):
            return self._extra_cost(base)
        return self._extra_cost(g)

    def poll(self):
        """None while pending, else (ok, path)."""
        if not self._pending:
            return None
        self._pending = False
        return (self._result is not None, self._result)

    def set_grid(self, grid) -> None:
        self.grid = grid

    def cancel(self) -> None:
        """Forget a pending request (its result will not be returned)."""
        self._pending = False
        self._result = None


class PlannerWorker(SyncPlanner):
    """A* in a background thread so the ROS executor keeps spinning."""

    def __init__(self, grid, robot_radius: float = 0.30, smooth_radius: Optional[float] = None,
                 cost_fn: Optional[Callable] = None):
        super().__init__(grid, robot_radius, smooth_radius, cost_fn)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._done = False
        self._gen = 0                      # request generation (cancel() discards older results)

    def submit(self, start, goal, grid=None) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._done = False
        grid = self.grid if grid is None else grid      # immutable snapshot for the worker thread

        def work():
            try:
                res = plan_home(grid, start, goal, self.robot_radius, self.smooth_radius, self._cost_for(grid))
            except Exception:   # pragma: no cover - planner failure = no path
                res = None
            with self._lock:
                if gen != self._gen:
                    return                 # cancelled meanwhile
                self._result = res
                self._done = True

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self._pending = True

    def poll(self):
        if not self._pending:
            return None
        with self._lock:
            if not self._done:
                return None
            self._pending = False
            return (self._result is not None, self._result)

    def cancel(self) -> None:
        with self._lock:
            self._gen += 1
            self._pending = False
            self._done = False
            self._result = None


# =============================================================================
# shadow prediction helpers
# =============================================================================
def path_ahead(path: Sequence[Tuple[float, float]], x: float, y: float, lookahead: float,
               spacing: float = 0.25) -> Tuple[np.ndarray, float]:
    """Points every ``spacing`` metres along ``path`` starting at the projection
    of (x, y) on the polyline, up to ``lookahead`` metres ahead.  Returns
    (points (N,2), distance from the robot to the path)."""
    P = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(P) == 0:
        return np.zeros((0, 2)), math.inf
    if len(P) == 1:
        return P.copy(), float(math.hypot(P[0, 0] - x, P[0, 1] - y))
    A, B = P[:-1], P[1:]
    AB = B - A
    L2 = np.maximum(np.sum(AB * AB, axis=1), 1e-12)
    t = np.clip(((x - A[:, 0]) * AB[:, 0] + (y - A[:, 1]) * AB[:, 1]) / L2, 0.0, 1.0)
    proj = A + AB * t[:, None]
    d = np.hypot(proj[:, 0] - x, proj[:, 1] - y)
    k = int(np.argmin(d))
    pts = [proj[k]]
    remaining = lookahead
    cur = proj[k]
    seg = k
    acc = 0.0
    while remaining > 1e-9 and seg < len(AB):
        nxt = B[seg]
        L = float(np.hypot(*(nxt - cur)))
        if L < 1e-9:
            seg += 1
            continue
        take = min(L, remaining)
        # emit samples along this piece
        s = spacing - acc
        while s <= take + 1e-9:
            pts.append(cur + (nxt - cur) * (s / L))
            s += spacing
        acc = (acc + take) % spacing
        remaining -= take
        cur = cur + (nxt - cur) * (take / L)
        if take >= L - 1e-9:
            seg += 1
    return np.asarray(pts), float(d[k])


# =============================================================================
# supervisor
# =============================================================================
class RthLogic:
    def __init__(self, cfg: Optional[RthConfig], grid, dock_pose: Tuple[float, float, float],
                 battery: Optional[BatteryModel] = None,
                 rssi_lookup: Optional[Callable] = None,
                 home_distance: Optional[Callable[[float, float], float]] = None,
                 planner=None, terrain_cost: Optional[Callable] = None, obstacle_source: Optional[str] = None):
        """``terrain_cost``: callable grid -> extra A* cost (``terrain_cost_for(cfg,
        layout)``) for the default planner; a given ``planner`` brings its own.
        ``obstacle_source``: the node's obstacle input ('elevation' /
        'traversability', a str node parameter that RthConfig does not hold),
        reported in /safety/status ``obstacle.source``."""
        self.cfg = cfg or RthConfig()
        self.grid = grid
        self.dock = (float(dock_pose[0]), float(dock_pose[1]), float(dock_pose[2]))
        self.battery = battery or BatteryModel()
        self.rssi_lookup = rssi_lookup
        self.home_distance = home_distance
        self.planner = planner or SyncPlanner(grid, self.cfg.robot_radius,
                                              self.cfg.robot_radius + self.cfg.smooth_margin, terrain_cost)
        self.pp = PurePursuit(lookahead=self.cfg.lookahead, max_vx=self.cfg.max_speed,
                              max_wz=self.cfg.max_wz, goal_tolerance=self.cfg.goal_tolerance)
        # novel obstacles seen by the robot (elevation map) on top of the nav map
        self.obstacles = ObstacleAvoider.from_params(vars(self.cfg), self.cfg.robot_radius)
        if isinstance(obstacle_source, str) and obstacle_source.strip():
            self.obstacles.p.source = obstacle_source.strip().lower()
        self.obstacles.set_static_map(grid)
        # planning tier: 0 static + detected obstacles with the full occlusion shadow, 1 with the near
        # shadow, 2 the map alone (the previous tier found no path)
        self._obst_tier = 0
        self._submitted_tier: Optional[int] = None
        self._full_route = None            # (path, cost) of a full-shadow route while its near alternative is planned
        self._tick_lead: Optional[float] = None   # pure-pursuit look-ahead the obstacle governor asked for
        self.state = MONITOR
        self.reason: Optional[str] = None
        self.message = ""
        self.paused = False                # operator 'pause' while returning
        self.path: Optional[List[Tuple[float, float]]] = None
        self.trigger_t: Optional[float] = None
        self.replans = 0
        self.plan_failures = 0
        self.transitions: List[Tuple[float, str, Optional[str]]] = []
        self._t_state = 0.0
        self._last_plan_req = -1e9
        self._progress: Optional[Tuple[float, float]] = None
        self._dock_ref: Optional[Tuple[float, float, float]] = None   # (t, distance, |yaw error|) docking progress
        self._force_plan = False           # docking made no progress: plan a path even inside the dock area
        self._backoff_until: Optional[float] = None
        self._clr_cache = None
        self._obst_cache = None
        self._settle_since: Optional[float] = None
        self._below_batt: Optional[float] = None
        self._below_energy: Optional[float] = None
        self._below_rssi: Optional[float] = None
        self._external_plan: Optional[List[Tuple[float, float]]] = None
        self.predicted_min_rssi: Optional[float] = None
        self.energy_to_home_pct: Optional[float] = None
        self.distance_home: Optional[float] = None

    # ------------------------------------------------------------- helpers
    def _set(self, s: str, t: float, msg: str = "") -> None:
        if s != self.state:
            self.transitions.append((t, s, self.reason))
        self.state = s
        self._t_state = t
        self.message = msg

    @property
    def active(self) -> bool:
        return self.state != MONITOR

    def obstacle_frame_due(self, t: float) -> bool:
        """Should the node convert / fuse the obstacle frame arriving at t?
        Every frame while returning home, one per ``obstacle_idle_period`` s
        while monitoring / docked (the memory stays reasonably current for a
        return at a fraction of the CPU)."""
        return self.obstacles.enabled and self.obstacles.frame_due(float(t), self.state in MOVING_STATES)

    def set_grid(self, grid) -> None:
        self.grid = grid
        self._clr_cache = None
        self._obst_cache = None
        self.obstacles.set_static_map(grid)
        if hasattr(self.planner, "set_grid"):
            self.planner.set_grid(grid)

    # ----------------------------------------------------- obstacle safety
    def _static_clearance(self, x: float, y: float) -> Tuple[float, np.ndarray]:
        g = self.grid
        if g is None:
            return math.inf, np.zeros(2)
        if self._clr_cache is None or self._clr_cache[0] is not g:
            dist = g.distance_field()
            gy, gx = np.gradient(dist)
            self._clr_cache = (g, dist, gx, gy)
        _, dist, gx, gy = self._clr_cache
        ix, iy = g.world_to_grid(x, y)
        if not g.in_bounds(ix, iy):
            return 0.0, np.zeros(2)
        n = np.array([gx[iy, ix], gy[iy, ix]])
        nn = float(np.hypot(n[0], n[1]))
        return float(dist[iy, ix]), (n / nn if nn > 1e-9 else np.zeros(2))

    def _clearance(self, x: float, y: float, detected_only: bool = False) -> Tuple[float, np.ndarray]:
        """(clearance [m], unit vector pointing away from the closest obstacle):
        the nav map and the detected (novel) obstacles, whichever is closer -
        the velocity safety filter and the stuck back-off respect both
        (``detected_only``: the detected obstacles alone, for docking - the
        dock pose may lie close to a mapped wall)."""
        clr, n = (math.inf, np.zeros(2)) if detected_only else self._static_clearance(x, y)
        if self.obstacles.enabled and self.obstacles.memory.has_confirmed():
            d, nd = self.obstacles.memory.nearest(x, y, max(self.cfg.safety_dist, self.cfg.robot_radius + 0.25) + 0.2)
            if d < clr and nd.any():
                return d, nd
        return clr, n

    def _safe_cmd(self, x: float, y: float, yaw: float, cmd: Tuple[float, float, float],
                  detected_only: bool = False):
        """Velocity safety filter.  Below ``safety_dist`` the world-frame
        velocity component towards the closest obstacle is removed
        progressively (completely at half the robot radius) and a small
        repulsion is added:

            k = clip((d - r/2) / (d_safe - r/2), 0, 1)
            v <- v - (1 - k) min(v.n, 0) n + v_rep (1 - k) n
        """
        cfg = self.cfg
        clr, n = self._clearance(x, y, detected_only)
        if clr >= cfg.safety_dist or not n.any():
            return cmd
        vx, vy, wz = cmd
        c, s = math.cos(yaw), math.sin(yaw)
        v = np.array([c * vx - s * vy, s * vx + c * vy])
        r0 = 0.5 * cfg.robot_radius
        k = min(1.0, max(0.0, (clr - r0) / max(cfg.safety_dist - r0, 1e-6)))
        vn = float(v @ n)
        if vn < 0.0:
            v = v - (1.0 - k) * vn * n
        v = v + cfg.safety_repulse * (1.0 - k) * n
        sp = float(np.hypot(v[0], v[1]))
        if sp > cfg.max_speed:
            v *= cfg.max_speed / sp
        return (float(c * v[0] + s * v[1]), float(-s * v[0] + c * v[1]), wz)

    def dist_to_dock(self, x: float, y: float) -> float:
        return math.hypot(x - self.dock[0], y - self.dock[1])

    def _home_distance(self, x: float, y: float) -> float:
        if self.home_distance is not None:
            try:
                return float(self.home_distance(x, y))
            except Exception:
                pass
        return 1.4 * self.dist_to_dock(x, y)

    def energy_pct_home(self, x: float, y: float) -> float:
        d = self._home_distance(x, y)
        e = self.battery.energy_for_distance(d, self.cfg.nominal_speed, self.cfg.docking_overhead_s)
        return self.battery.pct_for_energy(e)

    def _occupied(self, pts: np.ndarray) -> np.ndarray:
        """Boolean per point: occupied / unknown / outside the nav map."""
        g = self.grid
        if g is None or len(pts) == 0:
            return np.zeros(len(pts), dtype=bool)
        if self._obst_cache is None or self._obst_cache[0] is not g:
            self._obst_cache = (g, g.obstacle_mask())
        mask = self._obst_cache[1]
        ix, iy = g.world_to_grid_array(pts)
        ok = (ix >= 0) & (ix < g.width) & (iy >= 0) & (iy < g.height)
        occ = np.ones(len(pts), dtype=bool)
        occ[ok] = mask[iy[ok], ix[ok]]
        return occ

    def predict_min_rssi(self, x: float, y: float, vx: float, vy: float,
                         plan: Optional[Sequence[Tuple[float, float]]]) -> Optional[float]:
        """Minimum predicted RSSI along the plan look-ahead and the velocity
        extrapolation (None if no RSSI map).

        Only positions the robot can physically reach are predicted: plan
        samples inside obstacles are dropped and the straight velocity
        extrapolation ends at the first occupied cell (otherwise walking
        towards an outer wall would 'predict' the attenuated RSSI *inside* or
        behind the wall and trigger a false comm-shadow return)."""
        if self.rssi_lookup is None:
            return None
        cfg = self.cfg
        pts = [np.array([[x, y]])]
        if plan is not None and len(plan) >= 1:
            ahead, off = path_ahead(plan, x, y, cfg.shadow_lookahead_m, cfg.shadow_sample_m)
            if off <= cfg.plan_max_offset_m and len(ahead):
                pts.append(ahead[~self._occupied(ahead)])
        sp = math.hypot(vx, vy)
        if sp > 0.05 and cfg.shadow_horizon_s > 0.0:
            ts = np.arange(cfg.shadow_sample_m / sp, cfg.shadow_horizon_s + 1e-9,
                           max(cfg.shadow_sample_m / sp, 0.05))
            ext = np.stack([x + vx * ts, y + vy * ts], -1)
            occ = self._occupied(ext)
            if occ.any():
                ext = ext[:int(np.argmax(occ))]
            pts.append(ext)
        P = np.concatenate(pts, axis=0)
        r = np.asarray(self.rssi_lookup(P[:, 0], P[:, 1]), dtype=float)
        return float(np.min(r)) if r.size else None

    def _trigger(self, inp: RthInputs) -> Optional[str]:
        cfg = self.cfg
        t = inp.t
        cmd = (inp.command or "").strip().lower()
        if cmd in MANUAL_COMMANDS:
            return "manual"
        # battery low (debounced)
        if inp.battery_pct is not None and inp.battery_pct < cfg.rth_battery_pct:
            if self._below_batt is None:
                self._below_batt = t
            if t - self._below_batt >= cfg.battery_debounce_s:
                return "battery_low"
        else:
            self._below_batt = None
        # energy reserve
        if cfg.energy_enabled and inp.battery_pct is not None and self.energy_to_home_pct is not None:
            need = self.energy_to_home_pct * cfg.energy_safety_factor + cfg.energy_margin_pct
            if need > inp.battery_pct and self.dist_to_dock(inp.x, inp.y) > cfg.docking_radius:
                if self._below_energy is None:
                    self._below_energy = t
                if t - self._below_energy >= cfg.energy_debounce_s:
                    return "energy_reserve"
            else:
                self._below_energy = None
        # predicted comm shadow
        if cfg.comm_shadow_enabled and self.predicted_min_rssi is not None \
                and self.predicted_min_rssi < cfg.rssi_min_dbm:
            return "comm_shadow_predicted"
        # measured comm loss
        if cfg.comm_lost_enabled and inp.rssi_dbm is not None and inp.rssi_dbm < cfg.rssi_min_dbm:
            if self._below_rssi is None:
                self._below_rssi = t
            if t - self._below_rssi > cfg.comm_lost_time_s:
                return "comm_lost"
        else:
            self._below_rssi = None
        return None

    def trigger_conditions(self, inp: RthInputs) -> List[str]:
        """Automatic trigger conditions that hold right now (no debounce):
        an operator 'abort' is refused while any of them holds, because the
        return would re-trigger at once.  Outside MONITOR the comm-shadow
        prediction covers the robot position only."""
        cfg = self.cfg
        out = []
        if inp.battery_pct is not None and inp.battery_pct < cfg.rth_battery_pct:
            out.append("battery_low")
        if (cfg.energy_enabled and inp.battery_pct is not None and self.energy_to_home_pct is not None
                and self.energy_to_home_pct * cfg.energy_safety_factor + cfg.energy_margin_pct > inp.battery_pct
                and self.dist_to_dock(inp.x, inp.y) > cfg.docking_radius):
            out.append("energy_reserve")
        if (cfg.comm_shadow_enabled and self.predicted_min_rssi is not None
                and self.predicted_min_rssi < cfg.rssi_min_dbm):
            out.append("comm_shadow_predicted")
        if cfg.comm_lost_enabled and inp.rssi_dbm is not None and inp.rssi_dbm < cfg.rssi_min_dbm:
            out.append("comm_lost")
        return out

    def _operator_command(self, cmd: str, inp: RthInputs, out: RthOutput) -> None:
        """'pause' / 'resume' / 'abort' (and 'rth' to continue) while returning."""
        t = inp.t
        if cmd == "abort":
            block = self.trigger_conditions(inp)
            if block:
                self.message = (f"abort refused: {', '.join(block)} - the return home continues "
                                "('pause' or teleop stops the robot)")
                return
            self._finish(t, out, "aborted by operator", stand=False)
        elif cmd == "pause":
            if not self.paused:
                self.paused = True
                self.message = "paused by operator ('resume' continues)"
        elif self.paused and (cmd == "resume" or cmd in MANUAL_COMMANDS):
            self.paused = False
            self._settle_since = None
            self._backoff_until = None
            self._dock_ref = None              # the pause does not count as docking without progress
            if self.state in (RTH_PLANNING, RTH_NAVIGATING):
                # the robot may have been moved (teleop) while paused: plan again from here
                if hasattr(self.planner, "cancel"):
                    self.planner.cancel()
                self.pp.clear()
                self._last_plan_req = -1e9
                self._set(RTH_PLANNING, t, "resumed by operator: re-planning")
            else:
                self._t_state = t
                self.message = "resumed by operator"

    # -------------------------------------------------------------- update
    def update(self, inp: RthInputs) -> RthOutput:
        cfg = self.cfg
        t = inp.t
        out = RthOutput()
        cmd = (inp.command or "").strip().lower()
        # ---- bookkeeping that runs in every state
        if inp.plan is not None and self.state == MONITOR:
            self._external_plan = [tuple(p[:2]) for p in inp.plan] if len(inp.plan) else None
        self.distance_home = self._home_distance(inp.x, inp.y)
        self.energy_to_home_pct = self.energy_pct_home(inp.x, inp.y)
        if self.state == MONITOR:
            self.predicted_min_rssi = self.predict_min_rssi(inp.x, inp.y, inp.vx, inp.vy, self._external_plan)
        else:
            self.predicted_min_rssi = self.predict_min_rssi(inp.x, inp.y, 0.0, 0.0, None)

        if inp.obstacle_frame is not None and self.obstacles.enabled:
            self.obstacles.update(inp.obstacle_frame, (inp.x, inp.y, inp.yaw), t)   # in every state

        st = self.state
        if st == MONITOR:
            reason = self._trigger(inp)
            if reason is not None:
                self.reason = reason
                self.trigger_t = t
                self.replans = 0
                self.plan_failures = 0
                self._external_plan = None
                self.obstacles.new_goal()
                self._obst_tier = 0
                self._full_route = None
                self._set(RTH_PLANNING, t, f"triggered: {reason}")
                st = RTH_PLANNING
        elif st in MOVING_STATES and cmd:
            self._operator_command(cmd, inp, out)
            st = self.state
        elif st in (DOCKED, CHARGING) and cmd == "resume":
            self._finish(t, out, "resumed by operator")
            st = MONITOR

        if st in MOVING_STATES and self.paused:
            out.cmd = (0.0, 0.0, 0.0)          # hold position until 'resume'
        elif st == RTH_PLANNING:
            self._do_planning(inp, out)
        elif st == RTH_NAVIGATING:
            self._do_navigating(inp, out)
        elif st == DOCKING:
            self._do_docking(inp, out)
        elif st == DOCKED:
            out.cmd = (0.0, 0.0, 0.0)
            if t - self._t_state >= cfg.sit_delay_s:
                self._set(CHARGING, t, "charging")
        elif st == CHARGING:
            if inp.battery_pct is not None and inp.battery_pct >= cfg.resume_battery_pct:
                self._finish(t, out, "battery recharged")
            elif inp.charging is False and t - self._t_state > 30.0:
                self.message = "charger not detected (still waiting)"

        out.rth_active = self.state != MONITOR
        out.status = self.status(inp)
        return out

    def _finish(self, t: float, out: RthOutput, msg: str, stand: bool = True) -> None:
        """Back to MONITOR.  ``stand``: get up from the dock ('stand' on
        /locomotion/command); an aborted return leaves the gait mode alone."""
        self.reason = None
        self.path = None
        self.paused = False
        self.pp.clear()
        if hasattr(self.planner, "cancel"):
            self.planner.cancel()          # a late A* result must not leak into the next return
        self._external_plan = None
        self._backoff_until = None
        self._settle_since = None
        self._progress = None
        self._dock_ref = None
        self._force_plan = False
        self._last_plan_req = -1e9
        self._below_batt = self._below_energy = self._below_rssi = None
        self.obstacles.new_goal()
        self._obst_tier = 0
        self._full_route = None
        self._set(MONITOR, t, msg)
        out.plan = []                      # clear the latched home path on /nav/plan
        if stand:
            out.locomotion = "stand"

    def _do_planning(self, inp: RthInputs, out: RthOutput) -> None:
        cfg = self.cfg
        t = inp.t
        out.cmd = (0.0, 0.0, 0.0)
        if (self.dist_to_dock(inp.x, inp.y) < cfg.docking_radius and not self._force_plan
                and self._dock_line_clear(inp.x, inp.y)):
            self._backoff_until = None
            self._enter_docking(t, "at the dock area")
            return
        if self._backoff_until is not None:
            # stuck: step away from the closest obstacle before re-planning
            clr, n = self._clearance(inp.x, inp.y)
            if t < self._backoff_until and n.any() and clr < cfg.robot_radius + 0.25:
                c, s = math.cos(inp.yaw), math.sin(inp.yaw)
                vw = cfg.backoff_speed * n
                out.cmd = (float(c * vw[0] + s * vw[1]), float(-s * vw[0] + c * vw[1]), 0.0)
                self.message = "backing off before re-planning"
                return
            self._backoff_until = None
        res = self.planner.poll()
        if res is None:
            if not getattr(self.planner, "_pending", False) and t - self._last_plan_req >= cfg.replan_retry_s:
                # static map + detected obstacles (full, then near occlusion shadow), then the map only
                dyn, tier = self._dyn_grid()
                self._full_route = None
                self._submit((inp.x, inp.y), (self.dock[0], self.dock[1]), dyn, tier)
                self._last_plan_req = t
                res = self.planner.poll()      # synchronous planners answer at once
            if res is None:
                return
        ok, path = res
        tier, self._submitted_tier = self._submitted_tier, None
        full, self._full_route = self._full_route, None
        note = ""
        if full is not None:
            # answer of the near-shadow alternative: taken only when much cheaper than the full-shadow route
            av, cost = self.obstacles, self._static_cost()
            c_near = av.route_cost(self.grid, cost, path) if (ok and path) else math.inf
            if av.near_is_cheaper(full[1], c_near):
                note = f" past the near side of the detected obstacle(s) ({c_near:.1f} vs {full[1]:.1f})"
            else:
                ok, path = True, full[0]
            tier = None
        elif ok and path and tier == 0:
            near = self.obstacles.planning_grid(self.grid, layer="near")
            c_full = self.obstacles.route_cost(self.grid, self._static_cost(), path) if near is not None else 0.0
            if near is not None and self.obstacles.near_is_cheaper(c_full, self.dist_to_dock(inp.x, inp.y)):
                # the full occlusion shadow may have pushed the route across debris / a long way round
                # (no route could be switch_ratio cheaper otherwise): plan the near-shadow alternative
                # before committing (ObstacleAvoider.near_is_cheaper)
                self._full_route = ([tuple(p) for p in path], c_full)
                self._submit((inp.x, inp.y), (self.dock[0], self.dock[1]), near, 1)
                self._last_plan_req = t
                return
        if (not ok or not path) and tier is not None:
            # no path around the detected obstacle(s): the next tier at once - the near occlusion
            # shadow, then the map alone (not a failure; pure pursuit + the speed governor hold the
            # robot in front, the stuck logic decides)
            self._obst_tier = tier + 1
            nxt, _ = self._dyn_grid()
            if nxt is not None or self.obstacles.p.static_fallback:
                self._last_plan_req = -1e9
                self.message = ("no path around the detected obstacle(s) - planning with their near side only"
                                if nxt is not None else
                                "no path around the detected obstacle(s) - planning on the map")
                return
            self._obst_tier = 0                # no map fallback: a planning failure, retried later
        if not ok or not path:
            self.plan_failures += 1
            self.message = f"planning failed ({self.plan_failures})"
            if self.plan_failures >= cfg.max_plan_failures:
                # last resort: straight line (pure pursuit + obstacle-free docking area)
                path = [(inp.x, inp.y), (self.dock[0], self.dock[1])]
            else:
                return
        self.path = [tuple(p) for p in path]
        # dusan_common.PurePursuit searches the look-ahead point from the
        # closest *vertex*; on sparse paths (smooth_path emits 1.5 m segments)
        # that vertex can lie behind the robot and be picked as the target.
        # A dense path (spacing << look-ahead) avoids it.
        self.pp.set_path(densify(self.path, min(0.1, 0.25 * cfg.lookahead)))
        self._obst_tier = 0
        self._force_plan = False
        self.obstacles.set_route(self.pp.path)
        self._progress = (t, self.pp.remaining_distance(inp.x, inp.y))
        out.plan = list(self.path)
        self._set(RTH_NAVIGATING, t, f"path {path_length(self.path):.1f} m{note}")

    def _dyn_grid(self):
        """(static + detected grid of the current planning tier or the next one
        that exists, its tier) or (None, None) for the map alone."""
        layers = ("planning", "near")
        for tier in range(self._obst_tier, len(layers)):
            g = self.obstacles.planning_grid(self.grid, layer=layers[tier])
            if g is not None:
                return g, tier
        return None, None

    def _submit(self, start, goal, grid, tier: Optional[int] = None) -> None:
        """planner.submit with the static + dynamic grid when the planner takes one."""
        self._submitted_tier = None
        if grid is not None:
            try:
                takes_grid = "grid" in inspect.signature(self.planner.submit).parameters
            except (TypeError, ValueError):
                takes_grid = False
            if takes_grid:
                self._submitted_tier = 0 if tier is None else int(tier)
                self.planner.submit(start, goal, grid=grid)
                return
        self.planner.submit(start, goal)

    def _static_cost(self):
        """Terrain cost layer of the nav map (the planner's cost_fn), or None."""
        fn = getattr(self.planner, "_extra_cost", None)
        if fn is None or self.grid is None:
            return None
        try:
            return fn(self.grid)
        except Exception:
            return None

    def _avoid(self, inp: RthInputs, out: RthOutput) -> Optional[float]:
        """Novel-obstacle step of the home drive: speed scale, or None when the
        tick switched to RTH_PLANNING (global re-plan on static + dynamic)."""
        av = self.obstacles
        if not av.enabled or not self.pp.active:
            return 1.0
        t = inp.t
        try:
            act = av.step((inp.x, inp.y, inp.yaw), self.pp.remaining_path(), t, self.grid, self._static_cost,
                          lead=float(self.cfg.lookahead))
        except Exception:  # noqa: BLE001 - an avoidance bug never stops the return home
            av.errors += 1
            return 1.0
        self._tick_lead = act.lead             # pure-pursuit look-ahead of this tick (tight near obstacles)
        if act.kind == "detour":
            self.path = list(act.path)
            self.pp.set_path(act.path)
            out.plan = list(act.path)
            self.message = act.message
            # the detour is longer: re-base the progress reference, keep the stuck timer running
            t0 = self._progress[0] if self._progress else t
            self._progress = (t0, self.pp.remaining_distance(inp.x, inp.y))
        elif act.kind == "replan":
            av.note_replan()
            self.pp.clear()
            self._last_plan_req = -1e9
            self._set(RTH_PLANNING, t, act.message)
            out.cmd = (0.0, 0.0, 0.0)
            return None
        return act.speed_scale

    def _dock_line_clear(self, x: float, y: float) -> bool:
        """Straight docking line robot -> dock free of detected (novel)
        obstacles (robot_radius + margin / 2; a robot already closer may move
        as long as it does not get closer)?  The docking P controller drives
        that line: entering DOCKING beside a box near the dock would push into
        its corner."""
        av = self.obstacles
        if not av.enabled or not av.memory.has_confirmed():
            return True
        line = [(float(x), float(y)), (self.dock[0], self.dock[1])]
        return path_blockage(av.memory, line, 0, check_radius(self.cfg.robot_radius, av.p), math.inf,
                             robot_xy=(float(x), float(y))) is None

    def _enter_docking(self, t: float, msg: str) -> None:
        self.pp.clear()
        self._dock_ref = None
        self._set(DOCKING, t, msg)

    def _do_navigating(self, inp: RthInputs, out: RthOutput) -> None:
        cfg = self.cfg
        t = inp.t
        if self.dist_to_dock(inp.x, inp.y) < cfg.docking_radius and self._dock_line_clear(inp.x, inp.y):
            self._enter_docking(t, "docking")
            self._do_docking(inp, out)
            return
        scale = self._avoid(inp, out)
        if scale is None:
            return
        lead, self._tick_lead = self._tick_lead, None
        if lead is not None:
            self.pp.lookahead = lead
        try:
            vx, vy, wz, done = self.pp.compute(inp.x, inp.y, inp.yaw, speed_scale=scale)
        finally:
            self.pp.lookahead = float(self.cfg.lookahead)
        if scale <= 0.0:
            vx = vy = wz = 0.0           # stop: no trotting in place in front of the obstacle
        else:
            vy *= min(1.0, scale)
        if done:
            self._enter_docking(t, "path end reached")
            self._do_docking(inp, out)
            return
        out.cmd = self._safe_cmd(inp.x, inp.y, inp.yaw, (vx, vy, wz))
        rem = self.pp.remaining_distance(inp.x, inp.y)
        t0, r0 = self._progress or (t, rem)
        if rem < r0 - cfg.stuck_progress_m:
            self._progress = (t, rem)
        elif t - t0 > cfg.stuck_time_s:
            self.replans += 1
            self._last_plan_req = -1e9
            self._backoff_until = t + cfg.backoff_s
            self._set(RTH_PLANNING, t, f"no progress for {cfg.stuck_time_s:.0f} s: re-planning")
            out.cmd = (0.0, 0.0, 0.0)

    def _do_docking(self, inp: RthInputs, out: RthOutput) -> None:
        cfg = self.cfg
        t = inp.t
        ex, ey = self.dock[0] - inp.x, self.dock[1] - inp.y
        d = math.hypot(ex, ey)
        eyaw = _wrap(self.dock[2] - inp.yaw)
        if d > 2.5 * cfg.docking_radius:      # pushed away: plan again
            self._last_plan_req = -1e9
            self._set(RTH_PLANNING, t, "left the dock area: re-planning")
            out.cmd = (0.0, 0.0, 0.0)
            return
        if d < cfg.dock_pos_tolerance and abs(eyaw) < cfg.dock_yaw_tolerance:
            out.cmd = (0.0, 0.0, 0.0)
            if self._settle_since is None:
                self._settle_since = t
            if t - self._settle_since >= cfg.dock_settle_s:
                self._settle_since = None
                self._set(DOCKED, t, "docked")
                out.locomotion = "sit"
            return
        self._settle_since = None
        # progress watchdog: a detected obstacle at / beside the dock (the governor holds the robot,
        # or the dock is unreachable) must not keep the return in DOCKING forever
        ref = self._dock_ref
        if ref is None or d < ref[1] - 0.05 or abs(eyaw) < ref[2] - 0.1:
            self._dock_ref = (t, d, abs(eyaw))
        elif t - ref[0] > cfg.stuck_time_s:
            self.replans += 1
            self._force_plan = True
            self._last_plan_req = -1e9
            self._backoff_until = t + cfg.backoff_s
            self._set(RTH_PLANNING, t, f"no docking progress for {cfg.stuck_time_s:.0f} s: re-planning")
            out.cmd = (0.0, 0.0, 0.0)
            return
        vxw, vyw = cfg.dock_k_pos * ex, cfg.dock_k_pos * ey
        sp = math.hypot(vxw, vyw)
        if sp > cfg.dock_speed:
            vxw *= cfg.dock_speed / sp
            vyw *= cfg.dock_speed / sp
        if d < cfg.dock_pos_tolerance:
            vxw = vyw = 0.0
        c, s = math.cos(inp.yaw), math.sin(inp.yaw)
        wz = float(np.clip(cfg.dock_k_yaw * eyaw, -cfg.max_wz, cfg.max_wz))
        av = self.obstacles
        if av.enabled and av.memory.has_confirmed():
            # novel obstacles on the docking line: slow down / stop (true zero command), and the
            # velocity safety filter w.r.t. the detected obstacles (mapped walls near the dock pose
            # are the docking geometry, they stay out of it)
            scale, why, _ = speed_scale(av.memory, [(inp.x, inp.y), (self.dock[0], self.dock[1])], (inp.x, inp.y),
                                        cfg.robot_radius, av.p, t)
            if scale <= 0.0:
                out.cmd = (0.0, 0.0, 0.0)
                self.message = f"docking: {why}"
                return
            vxw, vyw = vxw * scale, vyw * scale
            out.cmd = self._safe_cmd(inp.x, inp.y, inp.yaw, (c * vxw + s * vyw, -s * vxw + c * vyw, wz),
                                     detected_only=True)
            return
        out.cmd = (c * vxw + s * vyw, -s * vxw + c * vyw, wz)

    # -------------------------------------------------------------- status
    def status(self, inp: Optional[RthInputs] = None) -> dict:
        docked = self.state in (DOCKED, CHARGING)
        if inp is not None and not docked and self.dist_to_dock(inp.x, inp.y) < 0.35 and inp.charging:
            docked = True
        rem = None
        if self.state == RTH_NAVIGATING and inp is not None:
            rem = self.pp.remaining_distance(inp.x, inp.y)
        return {
            "state": self.state,
            "rth_active": self.state != MONITOR,
            "reason": self.reason,
            "battery_pct": None if inp is None else inp.battery_pct,
            "rssi_dbm": None if inp is None else inp.rssi_dbm,
            "predicted_min_rssi": self.predicted_min_rssi,
            "distance_home": self.distance_home,
            "energy_to_home_pct": self.energy_to_home_pct,
            "docked": bool(docked),
            "trigger_time": self.trigger_t,
            "plan_length": None if not self.path else path_length(self.path),
            "remaining_m": rem,
            "replans": self.replans,
            "paused": bool(self.paused),
            "message": self.message,
            "obstacle": self.obstacles.status(),
        }
