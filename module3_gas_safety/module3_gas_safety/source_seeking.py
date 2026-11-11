"""
Olfactory gas-source seeking for the quadruped (Module 3, ROS free).

The seeker is a hybrid of bio-inspired plume tracking (surge / cast, as in
moth flight) and *gradient descent on -c* using a fused gradient estimate.

States
------
IDLE      seeker disabled (no commands).
SEARCH    no detection yet: coverage search; wall-to-wall legs scored by
          novelty (unvisited cells), cross-wind and upwind components.
SURGE     plume detected but gradient unreliable: walk upwind, steering
          towards the plume centre with the cross-wind gradient component.
CAST      plume lost: cross-wind zig-zag of growing amplitude around the
          last detection point, drifting downwind.
GRADIENT  gradient descent on f = -ln(c + c0) with normalised steps,
          momentum, adaptive step length and Armijo-like backtracking.
          When c > found_ppm a local-maximum confirmation pattern (centre +
          4 probes) is executed.
FOUND     concentration above found_ppm and local maximum confirmed.
FAILED    time-out.

Gradient estimation (log space, l = ln(c + c0))
-----------------------------------------------
1. Sensor array: least-squares plane through the 4 sensors at body offsets
   o_i:  l_i = a + g_b . o_i  ->  g_b = [(A^T A)^-1 A^T l]_{1:2},
   Cov_a = s_l^2 [(A^T A)^-1]_{1:2,1:2},  rotated to the world frame.
2. Sliding window: weighted least squares plane fit over the recent nose
   samples (delay-compensated positions) with weights
   w_i = exp(-age_i / tau_w) exp(-|p_i - p0|^2 / (2 sigma_w^2)),
   Cov_w = s^2 [(A^T W A)^-1]_{1:2,1:2}.
3. Fusion (information form, independent estimates + weak zero-mean prior):
   g = (I_a + I_w + I_p)^-1 (I_a g_a + I_w g_w),   I = Cov^-1.
   Reliability: SNR = |g| / sqrt(g_hat^T Cov g_hat).

Sensor lag: the gas sensors have a first-order lag tau; a reading at time t
is associated with the pose at t - tau (delay compensation).

Step control (GRADIENT)
-----------------------
d_k   = unit(g_hat + beta_up * u_up)          (upwind bias keeps the robot on
                                               the source side of the plume)
m_k   = mu m_{k-1} + (1 - mu) d_k,  step direction = unit(m_k)
x_k+1 = x_k + alpha_k unit(m_k)               (nose target)
Armijo test after the step (dwell-filtered readings):
      l(x_k+1) >= l(x_k) + c1 alpha_k (g . d) - eps_noise
accept -> alpha *= grow (capped) ; reject -> alpha *= shrink and go back
along the same direction (backtracking line search), momentum reset when
alpha < alpha_min.

Obstacles: a step whose straight segment would violate the robot clearance on
the nav map is *projected onto free space*: the first point along the step
ray that lies beyond the obstacle with enough clearance becomes the goal and
a local A* (dusan_common.planning on a window of the nav map) routes around
the obstacle (through doorways / gaps in interior walls).  When no detour
exists the step is shortened / rotated instead.  The velocity command is
projected onto the free half-space of the distance-field gradient with a
repulsive term near walls.

Lying pipes (``set_pipes``, the layout obstacles of type pipe): a step over
a pipe is walked like a patrol crossing - the velocity command goes through
dusan_common.debris_crossing.CrossingGovernor (turn in place square to the
pipe before the feet reach it, then straight over it at a steady 0.25 m/s
without turning or strafing until the hind feet are clear; slower next to a
pipe that is passed), and a step target never lies on a pipe: one within the
governor's ``zone`` of a pipe axis moves out to that distance on its side
(Gazebo mission check: the seeker turned at 0.8 rad/s over the F-OBS3 pipe in
the partition doorway and fell there four times in one run).

Source estimate: weighted centroid of the top-k samples
      x_hat = sum w_i p_i / sum w_i,   w_i = c_i - c_(k+1)  (excess over the
      (k+1)-th largest sample).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from dusan_common.debris_crossing import CrossingGovernor, CrossingParams
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.obstacle_layer import OBSTACLE_DEFAULTS, ObstacleMemory, ObstacleParams
from dusan_common.planning import astar, path_length, smooth_path

_OD = OBSTACLE_DEFAULTS

IDLE, SEARCH, SURGE, CAST, GRADIENT, FOUND, FAILED = (
    "IDLE", "SEARCH", "SURGE", "CAST", "GRADIENT", "FOUND", "FAILED")
STATES = (IDLE, SEARCH, SURGE, CAST, GRADIENT, FOUND, FAILED)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else np.zeros(2)


def _rot(v, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclass
class SeekerConfig:
    """Tunables (mirrors config/seeker.yaml)."""
    # --- sensing / thresholds [ppm]
    detect_ppm: float = 2.5
    lost_ppm: float = 1.2
    lost_time: float = 5.0
    gradient_min_ppm: float = 4.0
    found_ppm: float = 600.0
    sensor_lag: float = 1.0                 # first-order lag of the gas sensors [s]
    filter_tau: float = 0.4                 # EMA of the nose reading [s]
    nose_offset: Tuple[float, float] = (0.42, 0.0)
    array_offsets: Tuple[Tuple[float, float], ...] = (
        (0.30, 0.15), (0.30, -0.15), (-0.30, 0.15), (-0.30, -0.15))
    sensor_noise_ppm: float = 0.3
    sensor_noise_rel: float = 0.03
    model_noise_log: float = 0.05           # unmodelled turbulence in log units
    log_offset_ppm: float = 1.0             # c0 in l = ln(c + c0)
    # --- gradient estimation
    window_time: float = 20.0
    window_radius: float = 2.5
    window_sigma: float = 1.2
    window_tau: float = 10.0
    window_min_samples: int = 10
    gradient_prior_std: float = 3.0         # [1/m] weak zero-mean prior
    drift_prior_std: float = 0.05           # [1/s] prior on the temporal drift of ln c
    snr_min: float = 2.0
    grad_found_tol: float = 0.3             # flat-top test |grad ln c| [1/m]
    # --- motion
    max_speed: float = 0.35
    max_vy: float = 0.25
    max_wz: float = 0.8
    k_pos: float = 1.2
    k_yaw: float = 1.5
    goal_tolerance: float = 0.12
    yaw_tolerance: float = 0.25
    robot_radius: float = 0.35              # clearance kept from obstacles [m]
    repulse_dist: float = 0.6
    repulse_gain: float = 0.15
    # --- obstacle detours (local A* around walls / machines)
    detour_enabled: bool = True
    detour_reach: float = 5.0               # search this far along a blocked step for free space [m]
    detour_window: float = 8.0              # half size of the local planning window [m]
    detour_max_ratio: float = 6.0           # accept routes <= ratio x straight distance + 2 m
    detour_max_length: float = 20.0         # [m]
    detour_retry_s: float = 8.0             # no new attempt near a failed one for this long [s]
    waypoint_tolerance: float = 0.35        # route waypoint reached [m]
    # --- gradient steps
    step_init: float = 0.8
    step_min: float = 0.25
    step_max: float = 1.6
    step_grow: float = 1.3
    step_shrink: float = 0.5
    momentum: float = 0.5
    upwind_bias: float = 0.5
    grad_ref: float = 0.5                   # |grad ln c| [1/m] at which the gradient gets full weight
    armijo_c1: float = 0.1
    armijo_noise: float = 0.10              # minimum tolerance in log units
    armijo_noise_gain: float = 1.5          # x residual std of the window fit (temporal fluctuation)
    dwell_time: float = 0.8
    max_backtracks: int = 3
    # --- local maximum confirmation
    probe_radius: float = 0.6               # initial confirmation step [m]
    probe_min_radius: float = 0.15          # converged when the step drops below [m]
    probe_max_radius: float = 1.0
    confirm_dwell: float = 2.0              # dwell per confirmation point [s]
    confirm_snr: float = 3.0                # averaged array gradient significance
    confirm_tol: float = 0.15               # reading drop that abandons a point
    max_confirm_moves: int = 16
    confirm_weight: float = 0.5             # weight of the confirmation points in the final estimate
    # --- search / surge / cast
    wind_min: float = 0.05
    wind_tau: float = 3.0
    search_leg_max: float = 20.0            # legs run until a wall/obstacle or this length
    search_leg_min: float = 1.0
    search_cell: float = 1.0                # coverage memory resolution [m]
    search_mark_radius: float = 1.5         # cells within this radius of the path are 'visited'
    search_sweep: float = 6.0               # leg length that earns full novelty [m]
    search_w_novelty: float = 1.0
    search_w_cross: float = 0.6
    search_w_upwind: float = 0.2
    search_w_length: float = 0.1
    search_w_reverse: float = 0.3
    surge_cross_gain: float = 0.8
    surge_lookahead: float = 1.0
    cast_amp0: float = 1.0
    cast_amp_growth: float = 1.0
    cast_amp_max: float = 6.0
    cast_drift: float = 0.3
    stuck_time: float = 5.0
    # --- termination / estimate
    timeout: float = 300.0
    top_k: int = 8
    estimate_bin: float = 0.25              # spatial binning of samples for the estimate [m]
    # --- pause / resume: a run disabled while still seeking and re-enabled
    # within this time continues (plume memory, path, estimate kept, the pause
    # does not count against the time-out) instead of starting over
    # (mission pause / teleop override toggle /gas/seek/enable). 0 = never.
    resume_window_s: float = 45.0
    # --- novel (unmapped) obstacles from the elevation map (dusan_common.obstacle_layer memory):
    # merged into the nav map used by free_length / detours / the velocity projection
    obstacle_enabled: bool = bool(_OD["obstacle_enabled"])
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
    obstacle_idle_period: float = float(_OD["obstacle_idle_period"])   # [s] frame period while not seeking
    obstacle_rebuild_period: float = 1.0    # [s] min. time between distance-field rebuilds


@dataclass
class GradientEstimate:
    g: np.ndarray = field(default_factory=lambda: np.zeros(2))       # grad ln c (world) [1/m]
    cov: np.ndarray = field(default_factory=lambda: np.eye(2) * 9.0)
    snr: float = 0.0
    g_array: Optional[np.ndarray] = None
    g_window: Optional[np.ndarray] = None
    n_window: int = 0


class SourceSeeker:
    """Stateful seeker; call :meth:`update` at a fixed rate (e.g. 10 Hz)."""

    def __init__(self, cfg: Optional[SeekerConfig] = None, grid=None):
        self.cfg = cfg or SeekerConfig()
        self.grid = None
        self._dist = None
        self._dgx = None
        self._dgy = None
        # novel obstacles (elevation map): merged into the grid every obstacle_rebuild_period
        self.obstacles = ObstacleMemory(ObstacleParams.from_params(vars(self.cfg)), self.cfg.robot_radius)
        self._static = None
        self._dyn_dirty = False
        self._t_rebuild = -math.inf
        self.reroutes = 0
        self.crossing: Optional[CrossingGovernor] = None   # lying pipes (set_pipes)
        self.set_map(grid)
        offs = np.asarray(self.cfg.array_offsets, dtype=float).reshape(-1, 2)
        self._arr_offsets = offs
        A = np.column_stack([np.ones(len(offs)), offs])
        self._arr_pinv = np.linalg.pinv(A)
        self._arr_cov_unit = np.linalg.inv(A.T @ A)[1:, 1:]
        self.state = IDLE
        self._suspended: Optional[Tuple[str, Optional[float]]] = None   # (state, t) of a paused run
        self.resumed = False                                              # last enable() continued a run
        self._reset_run(0.0)

    # ================================================================ setup
    def set_pipes(self, source, params: Optional[CrossingParams] = None) -> None:
        """Lying pipes of ``source`` (Layout / layout dict / obstacle list; None:
        none) for the crossing governor and the off-pipe step targets (module
        docstring); ``params``: CrossingParams (default: the patrol's)."""
        gov = CrossingGovernor(source, params) if source is not None else None
        self.crossing = gov if gov is not None and gov.enabled else None

    def set_map(self, grid) -> None:
        """Nav OccupancyGridMap used for obstacle-aware steps (None = free space).
        Detected (novel) obstacles already in memory are merged right away."""
        self._static = grid
        self.obstacles.set_static_map(grid)
        self._apply_grid(self._obstacle_grid())
        self._dyn_dirty = False

    def _obstacle_grid(self):
        """Static map + detected obstacles (near occlusion shadow: the seeker
        projects / measures free space on it, the full assumed depth would
        close gaps beside an obstacle), or the static map."""
        g = self._static
        if g is not None and self.obstacles.p.enabled and self.obstacles.has_confirmed():
            return self.obstacles.combined_grid(g, "near")
        return g

    def _apply_grid(self, grid) -> None:
        self.grid = grid
        if grid is None:
            self._dist = None
            return
        self._dist = grid.distance_field()
        gy, gx = np.gradient(self._dist)
        self._dgx, self._dgy = gx, gy

    def update_obstacles(self, frame, pose, t: float) -> bool:
        """Fuse an elevation-map evidence frame (dusan_common.obstacle_layer).
        The nav grid behind free_length / _beyond / detours / the velocity
        projection is rebuilt from static + detected obstacles at most every
        ``obstacle_rebuild_period`` s while seeking (when idle only the memory
        is kept current; :meth:`enable` rebuilds); an active route or step
        that now crosses a detected obstacle is re-planned.  Returns True when
        the detected obstacles changed.  No-op when disabled or without a map."""
        if not self.obstacles.p.enabled or self._static is None or frame is None or pose is None:
            return False
        t = float(t)
        changed = self.obstacles.update(frame, pose, t)
        if changed:
            self._dyn_dirty = True
        if self._dyn_dirty and self.active and t - self._t_rebuild >= float(self.cfg.obstacle_rebuild_period):
            self._rebuild_grid(t)
            self._reroute(t, np.array([float(pose[0]), float(pose[1])]))
        return changed

    def _rebuild_grid(self, t: float) -> None:
        self._t_rebuild = float(t)
        self._dyn_dirty = False
        self._apply_grid(self._obstacle_grid())

    def obstacle_frame_due(self, t: float) -> bool:
        """Convert / fuse the frame arriving at ``t``?  Every frame while
        seeking, one per ``obstacle_idle_period`` s otherwise."""
        return self.obstacles.frame_due(t, self.active)

    def _reroute(self, t: float, p: np.ndarray) -> None:
        """Re-plan the active route / straight step when it now crosses an obstacle."""
        if not self.active or self._target is None or self.grid is None:
            return
        pts = [p] + ([q for q in self._route[self._route_i:]] if self._route is not None else [self._target])
        blocked = False
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.hypot(*(b - a)))
            if L > 1e-6 and self.free_length(a, (b - a) / L, L) < L - 0.05:
                blocked = True
                break
        if not blocked:
            return
        path = self._local_plan(p, self._target)
        if path is None or len(path) < 2:
            return                      # keep the target: the stuck timer ends the leg if it stays blocked
        route = np.asarray(path, dtype=float)
        if float(np.hypot(*(route[0] - p))) < 0.15:
            route = route[1:]
        if len(route):
            self._route, self._route_i = route, 0
            self._target = route[-1].copy()
            self._progress_ref = (t, float("inf"))
            self.reroutes += 1

    def _reset_run(self, t: float) -> None:
        self._t_start = t
        self._t_end: Optional[float] = None          # time of FOUND / FAILED
        self._t_last: Optional[float] = None
        self._poses: Deque[Tuple[float, float, float, float]] = deque()
        self._win: Deque[Tuple[float, float, float, float]] = deque()   # (t, x, y, l)
        self._samples: List[Tuple[float, float, float]] = []            # (x, y, c) delay-compensated
        self._c_f: Optional[float] = None
        self._c_arr_f: Optional[float] = None
        self._wind_f: Optional[np.ndarray] = None
        self._grad = GradientEstimate()
        self.path: List[Tuple[float, float]] = []
        self.path_length = 0.0
        self._last_detect_xy: Optional[np.ndarray] = None
        self._below_since: Optional[float] = None
        self._target: Optional[np.ndarray] = None
        self._route: Optional[np.ndarray] = None    # waypoints of an obstacle detour (last = _target)
        self._route_i = 0
        self._detour_fail: Optional[Tuple[float, float, float]] = None   # (t, x, y) of the last failure
        self.detours = 0
        self._yaw_des: Optional[float] = None
        self._yaw: Optional[float] = None
        self._dwell_until: Optional[float] = None
        self._progress_ref: Optional[Tuple[float, float]] = None   # (t, dist)
        self._found_xy: Optional[np.ndarray] = None
        self.phase = ""
        self.message = ""
        # search (coverage memory is reset below)
        # cast
        self._cast_center: Optional[np.ndarray] = None
        self._cast_amp = self.cfg.cast_amp0
        self._cast_sign = 1.0
        # gradient
        self._alpha = self.cfg.step_init
        self._mom = np.zeros(2)
        self._dir = np.zeros(2)
        self._l_ref: Optional[float] = None
        self._p_ref: Optional[np.ndarray] = None
        self._backtracks = 0
        self._weak_count = 0
        self._step_detour = False
        self._win_resid = 0.0
        self._drift = 0.0
        self._probe_r = self.cfg.probe_radius
        self._confirm_moves = 0
        self._came_from: Optional[np.ndarray] = None
        self._confirm_pts: List[Tuple[float, np.ndarray]] = []
        self._arr_acc: List[np.ndarray] = []
        self.confirm_snr = 0.0
        self.transitions: List[Tuple[float, str, str]] = []
        self._visited: set = set()
        self._last_leg_dir: Optional[np.ndarray] = None

    def enable(self, t: float) -> bool:
        """Start a seek run at time t.

        A run that was disabled while still seeking (SEARCH / SURGE / CAST /
        GRADIENT, not FOUND / FAILED) and is re-enabled within
        ``resume_window_s`` is *continued* (pause / manual override of the
        mission); otherwise a new run starts.  Returns True when continued."""
        if self._dyn_dirty:
            self._rebuild_grid(t)              # obstacles fused while idle
        sus, self._suspended = self._suspended, None
        win = float(self.cfg.resume_window_s)
        if (sus is not None and sus[1] is not None and win > 0.0
                and 0.0 <= t - sus[1] <= win and sus[0] in (SEARCH, SURGE, CAST, GRADIENT)):
            self._resume_run(t, sus[0], t - sus[1])
            self.resumed = True
            return True
        self._reset_run(t)
        self._set_state(SEARCH, t, "enabled")
        self.resumed = False
        return False

    def disable(self, t: Optional[float] = None) -> None:
        """Stop commanding.  With the time ``t`` an unfinished run is kept
        suspended so that :meth:`enable` within ``resume_window_s`` continues it."""
        self._suspended = (self.state, None if t is None else float(t)) if self.active else None
        self.state = IDLE
        self._target = None
        self._route = None

    def _resume_run(self, t: float, state: str, paused_s: float) -> None:
        """Continue a suspended run: the plume memory (samples, sliding
        window, gradient, coverage, cast centre, path, estimate) is kept, the
        clocks are shifted by the pause, and the current manoeuvre is
        re-planned from the present pose (the robot may have been moved by
        teleop meanwhile)."""
        cfg = self.cfg
        self.state = state
        self.transitions.append((t, state, "resumed"))
        self.message = f"resumed after {paused_s:.1f} s pause"
        self._t_start += paused_s                      # the pause does not count against the time-out
        if self._below_since is not None:
            self._below_since += paused_s
        if self._detour_fail is not None:
            tf, xf, yf = self._detour_fail
            self._detour_fail = (tf + paused_s, xf, yf)
        # _t_last is kept (a None would restart the run's time base); the first
        # update after the pause then sees a long dt, which resets the reading
        # filters to the present values.  The pose history for the sensor-lag
        # compensation must not interpolate across the gap.
        self._poses.clear()
        self._target = None
        self._route = None
        self._progress_ref = None
        self._dwell_until = None
        if state == GRADIENT:
            if self.phase.startswith("confirm"):
                # repeat the dwell of the current confirmation point here
                self._arr_acc = []
                self._stop_and_dwell(t, cfg.confirm_dwell)
            else:
                # new reference reading here, then the next step (alpha / momentum kept)
                self._l_ref = None
                self._step_detour = False
                self.phase = "dwell_ref"
                self._stop_and_dwell(t, cfg.dwell_time)
        elif state == SURGE and self.phase == "detour":
            self.phase = ""

    @property
    def active(self) -> bool:
        return self.state in (SEARCH, SURGE, CAST, GRADIENT)

    def _set_state(self, s: str, t: float, why: str = "") -> None:
        if s != self.state:
            self.transitions.append((t, s, why))
        self.state = s
        self.message = why
        if s in (FOUND, FAILED) and self._t_end is None:
            self._t_end = t
        self._target = None
        self._route = None
        self._dwell_until = None
        self._progress_ref = None
        self.phase = ""

    # ============================================================ map utils
    def clearance(self, x: float, y: float) -> float:
        if self.grid is None:
            return float("inf")
        ix, iy = self.grid.world_to_grid(x, y)
        if not self.grid.in_bounds(ix, iy):
            return 0.0
        return float(self._dist[iy, ix])

    def _clearance_dir(self, x: float, y: float) -> np.ndarray:
        """Unit vector pointing away from the closest obstacle."""
        if self.grid is None:
            return np.zeros(2)
        ix, iy = self.grid.world_to_grid(x, y)
        if not self.grid.in_bounds(ix, iy):
            return np.zeros(2)
        return _unit((self._dgx[iy, ix], self._dgy[iy, ix]))

    def free_length(self, p0, d, length: float) -> float:
        """Distance the robot centre can travel from p0 along unit d (<= length)
        without entering the clearance zone (or, if already inside it, without
        getting closer to obstacles than it currently is)."""
        if self.grid is None or length <= 0.0:
            return max(length, 0.0)
        step = 0.05
        n = max(1, int(math.ceil(length / step)))
        s = np.linspace(step, length, n) if n > 1 else np.array([length])
        xs = p0[0] + d[0] * s
        ys = p0[1] + d[1] * s
        ix, iy = self.grid.world_to_grid_array(np.stack([xs, ys], -1))
        ok = (ix >= 0) & (ix < self.grid.width) & (iy >= 0) & (iy < self.grid.height)
        clr = np.zeros(n)
        clr[ok] = self._dist[iy[ok], ix[ok]]
        d0 = self.clearance(p0[0], p0[1])
        allowed = clr >= min(self.cfg.robot_radius, d0 - 0.05)
        if allowed.all():
            return float(length)
        k = int(np.argmin(allowed))
        return float(s[k - 1]) if k > 0 else 0.0

    def _free_direction(self, p0, d, length: float, min_frac: float = 0.5,
                        angles_deg=(0, 25, -25, 50, -50, 75, -75, 100, -100)) -> Tuple[np.ndarray, float]:
        """Closest direction to d (rotating in both senses) with enough free length."""
        best = (np.zeros(2), 0.0)
        for a in angles_deg:
            dd = _rot(d, math.radians(a))
            fl = self.free_length(p0, dd, length)
            if fl >= min_frac * length:
                return dd, fl
            if fl > best[1]:
                best = (dd, fl)
        return best

    # -------------------------------------------------------- detours
    def _clearance_many(self, pts: np.ndarray) -> np.ndarray:
        """Clearance [m] at world points (N, 2); outside the map = 0."""
        g = self.grid
        ix, iy = g.world_to_grid_array(pts)
        ok = (ix >= 0) & (ix < g.width) & (iy >= 0) & (iy < g.height)
        out = np.zeros(len(pts))
        out[ok] = self._dist[iy[ok], ix[ok]]
        return out

    def _beyond(self, p: np.ndarray, d: np.ndarray, length: float) -> Optional[np.ndarray]:
        """Projection of a blocked step onto free space: the first point of the
        ray p + s d (s <= detour_reach) that lies *behind* the blocking
        obstacle with clearance >= robot_radius + 0.1 m (and not before the
        intended step end when that end is already free)."""
        cfg = self.cfg
        reach = max(length, cfg.detour_reach)
        s = np.arange(0.1, reach + 1e-9, 0.1)
        pts = p[None, :] + s[:, None] * d[None, :]
        clr = self._clearance_many(pts)
        blocked = clr < cfg.robot_radius
        if not blocked.any():
            return p + d * length
        k_block = int(np.argmax(blocked))
        ok = (np.arange(len(s)) > k_block) & (clr >= cfg.robot_radius + 0.1)
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            return None
        late = idx[s[idx] >= length - 1e-9]
        return pts[late[0]] if late.size and s[late[0]] - s[idx[0]] < 1.0 else pts[idx[0]]

    def _local_plan(self, p: np.ndarray, goal: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """A* (dusan_common.planning) restricted to a window of the nav map
        around start and goal (bounded cost: at most a few 10^4 expansions)."""
        g = self.grid
        cfg = self.cfg
        half = max(cfg.detour_window, 0.5 * float(np.hypot(*(goal - p))) + 2.0)
        cx, cy = 0.5 * (p[0] + goal[0]), 0.5 * (p[1] + goal[1])
        ix0, iy0 = g.world_to_grid(cx - half, cy - half)
        ix1, iy1 = g.world_to_grid(cx + half, cy + half)
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        ix1, iy1 = min(ix1, g.width - 1), min(iy1, g.height - 1)
        if ix1 - ix0 < 3 or iy1 - iy0 < 3:
            return None
        sub = OccupancyGridMap(g.occ[iy0:iy1 + 1, ix0:ix1 + 1].copy(), g.resolution,
                               g.origin_x + ix0 * g.resolution, g.origin_y + iy0 * g.resolution)
        n_cells = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        path = astar(sub, (float(p[0]), float(p[1])), (float(goal[0]), float(goal[1])),
                     robot_radius=cfg.robot_radius, snap_radius=1.0, max_expansions=n_cells)
        if path is None:
            return None
        return smooth_path(sub, path, cfg.robot_radius, max_segment=1.0)

    def _detour(self, t: float, p: np.ndarray, d: np.ndarray, length: float) -> Optional[np.ndarray]:
        """Waypoints routing a blocked step (direction d, length) around the
        obstacle, or None (no free space behind it / route too long)."""
        cfg = self.cfg
        if not cfg.detour_enabled or self.grid is None:
            return None
        if self._detour_fail is not None:
            tf, xf, yf = self._detour_fail
            if t - tf < cfg.detour_retry_s and math.hypot(p[0] - xf, p[1] - yf) < 1.0:
                return None
        goal = self._beyond(p, d, length)
        route = None
        if goal is not None:
            path = self._local_plan(p, goal)
            if path is not None and len(path) >= 2:
                straight = float(np.hypot(*(goal - p)))
                if path_length(path) <= min(cfg.detour_max_length, cfg.detour_max_ratio * straight + 2.0):
                    route = np.asarray(path, dtype=float)
                    if float(np.hypot(*(route[0] - p))) < 0.15:
                        route = route[1:]
        if route is None or len(route) == 0:
            self._detour_fail = (t, float(p[0]), float(p[1]))
            return None
        self.detours += 1
        return route

    # ======================================================== measurements
    def _pose_at(self, t: float) -> Tuple[float, float, float]:
        """Interpolated pose from the history (clamped at the ends)."""
        ps = self._poses
        if not ps:
            return 0.0, 0.0, 0.0
        if t <= ps[0][0]:
            return ps[0][1], ps[0][2], ps[0][3]
        if t >= ps[-1][0]:
            return ps[-1][1], ps[-1][2], ps[-1][3]
        # history is short (lag + margin) -> linear scan from the end
        for k in range(len(ps) - 1, 0, -1):
            t0, x0, y0, a0 = ps[k - 1]
            t1, x1, y1, a1 = ps[k]
            if t0 <= t <= t1:
                f = (t - t0) / max(t1 - t0, 1e-9)
                return x0 + f * (x1 - x0), y0 + f * (y1 - y0), a0 + f * _wrap(a1 - a0)
        return ps[-1][1], ps[-1][2], ps[-1][3]

    def _body_to_world(self, x, y, yaw, off) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([x + c * off[0] - s * off[1], y + s * off[0] + c * off[1]])

    def _log(self, c) -> np.ndarray:
        return np.log(np.maximum(np.asarray(c, dtype=float), 0.0) + self.cfg.log_offset_ppm)

    def _array_gradient(self, c_array, yaw_lag: float):
        cfg = self.cfg
        c = np.maximum(np.asarray(c_array, dtype=float).reshape(-1), 0.0)
        if c.size != len(self._arr_offsets):
            return None, None
        lv = self._log(c)
        coef = self._arr_pinv @ lv
        gb = coef[1:]
        s2 = float(np.mean((cfg.sensor_noise_ppm / (c + cfg.log_offset_ppm)) ** 2)) \
            + cfg.sensor_noise_rel ** 2 + cfg.model_noise_log ** 2
        cov_b = s2 * self._arr_cov_unit
        R = np.array([[math.cos(yaw_lag), -math.sin(yaw_lag)], [math.sin(yaw_lag), math.cos(yaw_lag)]])
        return R @ gb, R @ cov_b @ R.T

    def _window_gradient(self, t: float, p0: np.ndarray):
        """Spatio-temporal WLS plane fit  l = a + g . (p - p0) + b (t_i - t).

        The temporal drift ``b`` absorbs plume intensity changes (gusts);
        without it, a concentration change while the robot walks along a
        line is mistaken for a gradient along the track.  When position and
        time are collinear (straight walk) the along-track component becomes
        poorly determined and its covariance grows accordingly, so the fusion
        falls back on the sensor array for that direction.
        """
        cfg = self.cfg
        if len(self._win) < cfg.window_min_samples:
            return None, None, 0
        W = np.asarray(self._win)
        age = t - W[:, 0]
        dx = W[:, 1] - p0[0]
        dy = W[:, 2] - p0[1]
        d2 = dx * dx + dy * dy
        sel = (age <= cfg.window_time) & (d2 <= cfg.window_radius ** 2)
        n = int(np.count_nonzero(sel))
        if n < cfg.window_min_samples:
            return None, None, n
        w = np.exp(-age[sel] / cfg.window_tau) * np.exp(-d2[sel] / (2 * cfg.window_sigma ** 2))
        A = np.column_stack([np.ones(n), dx[sel], dy[sel], -age[sel]])
        lv = W[sel, 3]
        s2_meas = cfg.sensor_noise_rel ** 2 + cfg.model_noise_log ** 2
        # priors (information form, scaled to the measurement variance)
        prior = np.diag([1e-9, 1.0 / cfg.gradient_prior_std ** 2,
                         1.0 / cfg.gradient_prior_std ** 2, 1.0 / cfg.drift_prior_std ** 2]) * s2_meas
        N = A.T @ (A * w[:, None]) + prior
        try:
            Ninv = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            return None, None, n
        theta = Ninv @ (A.T @ (w * lv))
        r = lv - A @ theta
        sw = float(np.sum(w))
        s2 = max(s2_meas, float(np.sum(w * r * r)) / max(sw - 4.0, 1.0))
        cov = s2 * Ninv[1:3, 1:3]
        self._win_resid = math.sqrt(s2)
        self._drift = float(theta[3])
        return theta[1:3], cov, n

    def _fuse(self, ga, ca, gw, cw) -> GradientEstimate:
        I_p = np.eye(2) / self.cfg.gradient_prior_std ** 2
        info = I_p.copy()
        vec = np.zeros(2)
        for g, c in ((ga, ca), (gw, cw)):
            if g is None:
                continue
            try:
                Ii = np.linalg.inv(c + np.eye(2) * 1e-9)
            except np.linalg.LinAlgError:
                continue
            info += Ii
            vec += Ii @ g
        cov = np.linalg.inv(info)
        g = cov @ vec
        gn = float(np.hypot(g[0], g[1]))
        if gn > 1e-12:
            u = g / gn
            snr = gn / math.sqrt(max(float(u @ cov @ u), 1e-12))
        else:
            snr = 0.0
        return GradientEstimate(g=g, cov=cov, snr=snr, g_array=ga, g_window=gw)

    # ============================================================== update
    def update(self, t: float, x: float, y: float, yaw: float, c_nose: float,
               c_array: Optional[Sequence[float]] = None,
               wind: Optional[Sequence[float]] = None) -> Tuple[float, float, float]:
        """Process one measurement set and return the body velocity (vx, vy, wz)."""
        cfg = self.cfg
        if self.state == IDLE:
            return 0.0, 0.0, 0.0
        if self._t_last is None or t < self._t_last - 1e-6:
            # The run's time base starts with the first measurement: enable()
            # may have been called before /clock was running (node time 0 with
            # use_sim_time), and a simulation reset makes time jump backwards.
            # Either would otherwise corrupt the time-out and the histories.
            self._t_start = t
            self._t_last = None
            self._poses.clear()
            self._win.clear()
        dt = 0.0 if self._t_last is None else max(0.0, t - self._t_last)
        self._t_last = t
        # ---- pose history / path
        self._poses.append((t, x, y, yaw))
        while len(self._poses) > 2 and self._poses[1][0] < t - cfg.sensor_lag - 1.0:
            self._poses.popleft()
        if not self.path or math.hypot(x - self.path[-1][0], y - self.path[-1][1]) >= 0.05:
            if self.path:
                self.path_length += math.hypot(x - self.path[-1][0], y - self.path[-1][1])
            self.path.append((x, y))
        if self.state in (FOUND, FAILED):
            return 0.0, 0.0, 0.0
        self._mark_visited(np.array([x, y]))
        if t - self._t_start > cfg.timeout:
            self._set_state(FAILED, t, "timeout")
            return 0.0, 0.0, 0.0
        # ---- filters
        c_nose = max(0.0, float(c_nose)) if c_nose is not None and math.isfinite(c_nose) else 0.0
        if self._c_f is None:
            self._c_f = c_nose
        else:
            a = 1.0 - math.exp(-dt / max(cfg.filter_tau, 1e-3)) if dt > 0 else 0.0
            self._c_f += a * (c_nose - self._c_f)
        if wind is not None and all(math.isfinite(float(v)) for v in wind[:2]):
            wv = np.array([float(wind[0]), float(wind[1])])
            if self._wind_f is None:
                self._wind_f = wv
            else:
                a = 1.0 - math.exp(-dt / max(cfg.wind_tau, 1e-3)) if dt > 0 else 0.0
                self._wind_f = self._wind_f + a * (wv - self._wind_f)
        # ---- delay-compensated sample
        xl, yl, yawl = self._pose_at(t - cfg.sensor_lag)
        p_nose_lag = self._body_to_world(xl, yl, yawl, cfg.nose_offset)
        lv = float(self._log(c_nose))
        self._win.append((t, p_nose_lag[0], p_nose_lag[1], lv))
        while self._win and self._win[0][0] < t - cfg.window_time:
            self._win.popleft()
        self._samples.append((float(p_nose_lag[0]), float(p_nose_lag[1]), c_nose))
        if c_array is not None:
            ca_mean = float(np.mean(np.maximum(np.asarray(c_array, dtype=float), 0.0)))
            # the array mean is a smoothed reading at the robot centre
            self._samples.append((float(xl), float(yl), ca_mean))
            if self._c_arr_f is None:
                self._c_arr_f = ca_mean
            else:
                a = 1.0 - math.exp(-dt / max(cfg.filter_tau, 1e-3)) if dt > 0 else 0.0
                self._c_arr_f += a * (ca_mean - self._c_arr_f)
        # ---- gradient
        ga, ca = (self._array_gradient(c_array, yawl) if c_array is not None else (None, None))
        self._accumulate_array(t, ga)
        gw, cw, nw = self._window_gradient(t, p_nose_lag)
        self._grad = self._fuse(ga, ca, gw, cw)
        self._grad.n_window = nw
        # ---- detection bookkeeping
        c = self._c_f
        p = np.array([x, y])
        self._yaw = yaw
        if c >= cfg.detect_ppm:
            self._last_detect_xy = p.copy()
            self._below_since = None
        elif c < cfg.lost_ppm:
            if self._below_since is None:
                self._below_since = t
        else:
            self._below_since = None
        lost = self._below_since is not None and t - self._below_since >= cfg.lost_time
        # ---- state machine
        st = self.state
        if st == SEARCH:
            self._do_search(t, p, yaw, c)
        elif st == SURGE:
            self._do_surge(t, p, yaw, c, lost)
        elif st == CAST:
            self._do_cast(t, p, yaw, c)
        elif st == GRADIENT:
            self._do_gradient(t, p, yaw, c, lost)
        if self.state in (FOUND, FAILED):
            return 0.0, 0.0, 0.0
        return self._govern(x, y, yaw, self._track(t, p, yaw))

    # ============================================================ helpers
    def _upwind(self) -> Optional[np.ndarray]:
        if self._wind_f is None:
            return None
        n = float(np.hypot(*self._wind_f))
        if n < self.cfg.wind_min:
            return None
        return -self._wind_f / n

    def _nose_to_centre(self, nose_target: np.ndarray, yaw: float) -> np.ndarray:
        o = self.cfg.nose_offset
        c, s = math.cos(yaw), math.sin(yaw)
        return nose_target - np.array([c * o[0] - s * o[1], s * o[0] + c * o[1]])

    def _nose(self, p: np.ndarray, yaw: float) -> np.ndarray:
        return self._body_to_world(p[0], p[1], yaw, self.cfg.nose_offset)

    def _go(self, t: float, target: np.ndarray, yaw_des: Optional[float],
            route: Optional[np.ndarray] = None) -> None:
        """Drive to ``target`` (final heading ``yaw_des``), optionally along
        the detour waypoints ``route`` (route[-1] == target)."""
        self._route = None if route is None else np.asarray(route, dtype=float).reshape(-1, 2)
        self._route_i = 0
        self._target = self._off_pipe(np.asarray(target if route is None else self._route[-1], dtype=float))
        if self._route is not None:
            self._route[-1] = self._target
        self._yaw_des = yaw_des
        self._dwell_until = None
        self._progress_ref = (t, float("inf"))

    def _remaining(self, p: np.ndarray) -> float:
        """Distance still to go (along the detour route when one is active)."""
        if self._target is None:
            return 0.0
        if self._route is None:
            return float(np.hypot(*(self._target - p)))
        r = self._route[self._route_i:]
        d = float(np.hypot(*(r[0] - p)))
        if len(r) > 1:
            d += float(np.sum(np.hypot(np.diff(r[:, 0]), np.diff(r[:, 1]))))
        return d

    def _arrived(self, t: float, p: np.ndarray) -> bool:
        """Target reached, or no progress for ``stuck_time`` (blocked)."""
        if self._target is None:
            return True
        d_final = float(np.hypot(*(self._target - p)))
        if d_final < self.cfg.goal_tolerance:
            if self._yaw_des is None or self._yaw is None:
                return True
            if abs(_wrap(self._yaw_des - self._yaw)) < self.cfg.yaw_tolerance:
                return True
        d = self._remaining(p)
        t0, d0 = self._progress_ref or (t, d)
        if d < d0 - 0.1:
            self._progress_ref = (t, d)
        elif t - t0 > self.cfg.stuck_time:
            return True
        return False

    def _stop_and_dwell(self, t: float, dur: float) -> None:
        self._target = None
        self._route = None
        self._dwell_until = t + dur

    # ============================================================== SEARCH
    def _mark_visited(self, p: np.ndarray) -> None:
        cell = self.cfg.search_cell
        r = self.cfg.search_mark_radius
        ix0, iy0 = int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell))
        k = int(math.ceil(r / cell))
        for dx in range(-k, k + 1):
            for dy in range(-k, k + 1):
                cx, cy = (ix0 + dx + 0.5) * cell, (iy0 + dy + 0.5) * cell
                if (cx - p[0]) ** 2 + (cy - p[1]) ** 2 <= r * r:
                    self._visited.add((ix0 + dx, iy0 + dy))

    def _novelty(self, p: np.ndarray, d: np.ndarray, length: float) -> float:
        """Unvisited fraction of a leg, scaled by min(1, length / search_sweep)
        so that a long leg is not preferred over a cross-wind sweep merely
        because it is long."""
        cell = self.cfg.search_cell
        n = max(1, int(length / (0.5 * cell)))
        seen = set()
        new = 0
        for k in range(1, n + 1):
            q = p + d * (k * length / n)
            key = (int(math.floor(q[0] / cell)), int(math.floor(q[1] / cell)))
            if key in seen:
                continue
            seen.add(key)
            if key not in self._visited:
                new += 1
        frac = new / max(len(seen), 1)
        return frac * min(1.0, length / max(self.cfg.search_sweep, 1e-6))

    def _do_search(self, t, p, yaw, c) -> None:
        """Coverage search: score 16 candidate legs by novelty (unvisited
        cells), cross-wind and upwind components, free length and a penalty
        for reversing; legs run until blocked (wall to wall sweeps), so a
        cross-wind leg intersects the plume band from anywhere in the hall."""
        cfg = self.cfg
        if c >= cfg.detect_ppm:
            self._set_state(SURGE, t, "plume detected")
            return
        if self._target is not None and not self._arrived(t, p):
            return
        up = self._upwind()
        cross = None if up is None else _rot(up, math.pi / 2)
        best = None
        for k in range(16):
            d = np.array([math.cos(k * math.pi / 8), math.sin(k * math.pi / 8)])
            fl = self.free_length(p, d, cfg.search_leg_max)
            if fl < cfg.search_leg_min:
                continue
            score = cfg.search_w_novelty * self._novelty(p, d, fl) + cfg.search_w_length * fl / cfg.search_leg_max
            if up is not None:
                score += cfg.search_w_cross * abs(float(d @ cross)) + cfg.search_w_upwind * float(d @ up)
            if self._last_leg_dir is not None:
                score -= cfg.search_w_reverse * max(0.0, -float(d @ self._last_leg_dir))
            if best is None or score > best[0]:
                best = (score, d, fl)
        if best is None:   # boxed in: back off
            back = -np.array([math.cos(yaw), math.sin(yaw)])
            self._go(t, p + 0.5 * back, None)
            return
        _, d, fl = best
        self._last_leg_dir = d
        self._go(t, p + d * fl, math.atan2(d[1], d[0]))
        self.phase = "leg"

    # =============================================================== SURGE
    def _do_surge(self, t, p, yaw, c, lost) -> None:
        cfg = self.cfg
        if lost:
            self._enter_cast(t)
            return
        g = self._grad
        if g.snr >= cfg.snr_min and c >= cfg.gradient_min_ppm:
            self._enter_gradient(t, p, yaw)
            return
        if self.phase == "detour" and self._target is not None and not self._arrived(t, p):
            return                                   # keep following the detour route
        up = self._upwind()
        gh = _unit(g.g)
        if up is None:
            d = gh if g.snr > 0.5 else np.array([math.cos(yaw), math.sin(yaw)])
        else:
            cross = gh - (gh @ up) * up
            w = cfg.surge_cross_gain * min(1.0, g.snr / max(cfg.snr_min, 1e-6))
            d = _unit(up + w * cross)
        if self.free_length(p, d, cfg.surge_lookahead) < 0.3 * cfg.surge_lookahead:
            # upwind is blocked (wall / machine): route around it instead of
            # sliding along the obstacle surface
            route = self._detour(t, p, d, cfg.surge_lookahead)
            if route is not None:
                self._go(t, route[-1], None, route=route)
                self.phase = "detour"
                return
        d, fl = self._free_direction(p, d, cfg.surge_lookahead, min_frac=0.3)
        if fl < 0.1:
            self._enter_cast(t)
            return
        self._route = None
        self._target = p + d * fl
        self._yaw_des = math.atan2(d[1], d[0])
        self._dwell_until = None
        self.phase = "upwind"

    # ================================================================ CAST
    def _enter_cast(self, t: float) -> None:
        center = self._last_detect_xy
        self._set_state(CAST, t, "plume lost")
        self._cast_center = None if center is None else center.copy()
        self._cast_amp = self.cfg.cast_amp0
        self._cast_sign = 1.0

    def _do_cast(self, t, p, yaw, c) -> None:
        cfg = self.cfg
        if c >= cfg.detect_ppm:
            self._set_state(SURGE, t, "plume re-acquired")
            return
        if self._cast_center is None:
            self._cast_center = p.copy()
        if self._target is not None and not self._arrived(t, p):
            return
        if self._cast_amp > cfg.cast_amp_max:
            self._set_state(SEARCH, t, "cast exhausted")
            return
        up = self._upwind()
        if up is None:
            up = np.array([math.cos(yaw), math.sin(yaw)])
        cross = _rot(up, math.pi / 2)
        # move the cast line downwind a little each leg (plume drifts downwind)
        self._cast_center = self._cast_center - cfg.cast_drift * up
        goal = self._cast_center + self._cast_sign * self._cast_amp * cross
        d = goal - p
        L = float(np.hypot(*d))
        if L > 1e-6:
            dd = _unit(d)
            fl = self.free_length(p, dd, L)
            goal = p + dd * fl
            yaw_des = math.atan2(dd[1], dd[0])
        else:
            yaw_des = None
        self._go(t, goal, yaw_des)
        self.phase = f"leg amp={self._cast_amp:.1f}"
        self._cast_sign = -self._cast_sign
        self._cast_amp += cfg.cast_amp_growth * 0.5   # grows every second leg on average

    # ============================================================ GRADIENT
    def _enter_gradient(self, t, p, yaw) -> None:
        self._set_state(GRADIENT, t, "gradient reliable")
        self._alpha = self.cfg.step_init
        self._mom = np.zeros(2)
        self._backtracks = 0
        self._weak_count = 0
        self._l_ref = None
        self.phase = "dwell_ref"
        self._stop_and_dwell(t, self.cfg.dwell_time)

    def _plan_step(self, t, p, yaw) -> bool:
        """Plan the next gradient step from the current position. Returns False if blocked."""
        cfg = self.cfg
        g = self._grad
        up = self._upwind()
        gh = _unit(g.g)
        # gradient weight: small / unreliable gradients (broad, fluctuating
        # plume core) defer to the upwind direction, strong ones dominate
        gn = float(np.hypot(*g.g))
        w_g = min(1.0, gn / max(cfg.grad_ref, 1e-6)) * min(1.0, g.snr / max(2.0 * cfg.snr_min, 1e-6))
        d = gh.copy()
        if up is not None:
            d = _unit(w_g * gh + cfg.upwind_bias * up)
        if not d.any():
            d = np.array([math.cos(yaw), math.sin(yaw)])
        self._mom = cfg.momentum * self._mom + (1.0 - cfg.momentum) * d
        dirn = _unit(self._mom) if self._mom.any() else d
        nose = self._nose(p, yaw)
        yaw_des = math.atan2(dirn[1], dirn[0])
        centre_goal = self._nose_to_centre(nose + self._alpha * dirn, yaw_des)
        step_vec = centre_goal - p
        L = float(np.hypot(*step_vec))
        if L < 0.15:
            # mostly a turn on the spot (the nose swings onto the step)
            self._step_detour = False
            self._dir = dirn
            self._go(t, centre_goal, yaw_des)
            self.phase = "step"
            return True
        sd = _unit(step_vec)
        if self.free_length(p, sd, L) < 0.5 * L:
            # the step runs into an obstacle: project it onto the free space
            # behind the obstacle and route there (doorways, wall gaps)
            route = self._detour(t, p, sd, L)
            if route is not None:
                self._dir = dirn
                self._go(t, route[-1], yaw_des, route=route)
                self.phase = "step"
                self._step_detour = True
                return True
        self._step_detour = False
        sd2, fl = self._free_direction(p, sd, L, min_frac=0.5)
        if fl < 0.15:
            return False
        if not np.allclose(sd2, sd):
            yaw_des = math.atan2(sd2[1], sd2[0])
            dirn = sd2
        self._dir = dirn
        self._go(t, p + sd2 * fl, yaw_des)
        self.phase = "step"
        return True

    def _do_gradient(self, t, p, yaw, c, lost) -> None:
        cfg = self.cfg
        if lost:
            self._enter_cast(t)
            return
        # ------------------------------------------------------ dwelling
        if self._dwell_until is not None:
            if t < self._dwell_until:
                return
            self._dwell_until = None
            self._after_dwell(t, p, yaw, c)
            return
        # ------------------------------------------------------ moving
        if self._target is not None and not self._arrived(t, p):
            return
        self._stop_and_dwell(t, cfg.confirm_dwell if self.phase.startswith("confirm") else cfg.dwell_time)

    def _after_dwell(self, t, p, yaw, c) -> None:
        cfg = self.cfg
        l_now = float(self._log(c))
        nose = self._nose(p, yaw)
        ph = self.phase
        if ph.startswith("confirm"):
            self._confirm_step(t, p, yaw, c, nose)
            return
        if ph == "dwell_ref" or self._l_ref is None:
            self._l_ref, self._p_ref = l_now, nose.copy()
            if self._near_peak(c):
                self._start_confirm(t, p, yaw, c, nose)
                return
            if not self._plan_step(t, p, yaw):
                self._blocked(t, p, yaw, c, nose)
            return
        # ---- Armijo-like test of the step just taken
        g = self._grad
        slope = max(0.0, float(g.g @ self._dir))
        expected = cfg.armijo_c1 * self._alpha * slope
        tol = max(cfg.armijo_noise, cfg.armijo_noise_gain * self._win_resid)
        # a detour around an obstacle is a relocation, not a line-search step:
        # backtracking would lead straight back into the obstacle -> accept
        detoured, self._step_detour = self._step_detour, False
        if detoured or l_now >= self._l_ref + expected - tol:
            if not detoured and l_now > self._l_ref + tol:
                self._alpha = min(cfg.step_max, self._alpha * cfg.step_grow)
            self._l_ref, self._p_ref = l_now, nose.copy()
            self._backtracks = 0
            self._weak_count = self._weak_count + 1 if g.snr < 0.5 * cfg.snr_min else 0
            if self._near_peak(c):
                self._start_confirm(t, p, yaw, c, nose)
                return
            if self._weak_count >= 3:
                self._set_state(SURGE, t, "gradient too weak")
                return
            if not self._plan_step(t, p, yaw):
                self._blocked(t, p, yaw, c, nose)
            return
        # ---- rejected: backtracking line search along the same direction
        self._backtracks += 1
        self._alpha *= cfg.step_shrink
        if self._backtracks <= cfg.max_backtracks and self._alpha >= cfg.step_min and self._p_ref is not None:
            back_nose = self._p_ref + self._alpha * self._dir
            goal = self._nose_to_centre(back_nose, yaw)
            d = goal - p
            L = float(np.hypot(*d))
            if L > 0.05:
                dd = _unit(d)
                fl = self.free_length(p, dd, L)
                self._go(t, p + dd * fl, yaw)
                self.phase = "backtrack"
                return
        # direction failed: restart from here with fresh direction / momentum
        self._alpha = cfg.step_init
        self._mom = np.zeros(2)
        self._l_ref, self._p_ref = l_now, nose.copy()
        if self._backtracks > 2 * cfg.max_backtracks:
            self._set_state(SURGE, t, "backtracking failed")
            return
        if not self._plan_step(t, p, yaw):
            self._blocked(t, p, yaw, c, nose)

    def _confirmed_estimate(self) -> Optional[np.ndarray]:
        """Weighted centroid of the confirmation dwell points on the flat top
        (readings within confirm_tol of the best; drift-free, centre based),
        blended with the binned top-k estimate of all samples."""
        est_all = self.estimate()
        if not self._confirm_pts:
            return est_all
        vals = np.array([v for v, _ in self._confirm_pts])
        pts = np.array([q for _, q in self._confirm_pts])
        floor = (1.0 - self.cfg.confirm_tol) * float(vals.max())
        sel = vals >= floor
        w = vals[sel] - floor + 1e-6
        est_c = (pts[sel] * w[:, None]).sum(axis=0) / w.sum()
        if est_all is None:
            return est_c
        a = self.cfg.confirm_weight
        return a * est_c + (1.0 - a) * est_all

    def _near_peak(self, c: float) -> bool:
        """Start the local-maximum confirmation: concentration above
        found_ppm, or a flat top (small reliable gradient) at >= 70 % of it."""
        cfg = self.cfg
        if c >= cfg.found_ppm:
            return True
        g = self._grad
        flat = g.n_window >= cfg.window_min_samples and float(np.hypot(*g.g)) < cfg.grad_found_tol
        return flat and c >= 0.7 * cfg.found_ppm

    def _blocked(self, t, p, yaw, c, nose) -> None:
        if c >= self.cfg.found_ppm:
            self._start_confirm(t, p, yaw, c, nose)
        else:
            self._set_state(SURGE, t, "ascent blocked by obstacle")

    # ----------------------------------------------- local max confirmation
    # Near the peak the plume is flat and slowly fluctuating, so comparing
    # readings taken at different times is unreliable.  The confirmation
    # therefore uses *simultaneous* spatial differences: at every dwell point
    # the per-reading array gradients are averaged (white sensor noise drops
    # with sqrt(n)) and the robot moves by r along the averaged gradient while
    # it is significant (SNR >= confirm_snr).  r is halved whenever the
    # direction reverses (Rprop-like), and a dwell point that reads lower than
    # the best one by more than confirm_tol is abandoned (step back, halve r).
    # The local maximum is confirmed when the averaged gradient is no longer
    # significant ("gradient magnitude small") at the best point visited
    # ("no higher samples around").
    def _start_confirm(self, t, p, yaw, c, nose) -> None:
        best = self.best_sample()
        self._probe_r = self.cfg.probe_radius
        self._confirm_moves = 0
        self._came_from = None
        self._confirm_pts: List[Tuple[float, np.ndarray]] = []   # (reading, robot centre)
        self.phase = "confirm"
        if best is not None and best[2] > 1.3 * c and math.hypot(best[0] - nose[0], best[1] - nose[1]) < 2.0:
            self._goto_nose(t, p, yaw, np.array(best[:2]))      # start at the best sample seen
        else:
            self._stop_and_dwell(t, self.cfg.confirm_dwell)
        self._arr_acc = []

    def _goto_nose(self, t, p, yaw, nose_target) -> None:
        goal = self._nose_to_centre(nose_target, yaw)
        self._goto_centre(t, p, yaw, goal)

    def _goto_centre(self, t, p, yaw, goal) -> None:
        d = goal - p
        L = float(np.hypot(*d))
        if L < 0.05:
            self._stop_and_dwell(t, self.cfg.confirm_dwell)
            return
        dd = _unit(d)
        fl = self.free_length(p, dd, L)
        self._go(t, p + dd * fl, yaw)   # translate, keep heading (sensor geometry exact)

    def _accumulate_array(self, t: float, g_world: Optional[np.ndarray]) -> None:
        """Collect array gradients during the second half of a confirm dwell."""
        if (g_world is None or self._dwell_until is None or not self.phase.startswith("confirm")):
            return
        if t >= self._dwell_until - 0.5 * self.cfg.confirm_dwell:
            self._arr_acc.append(np.asarray(g_world, dtype=float))

    def _confirm_step(self, t, p, yaw, c, nose) -> None:
        cfg = self.cfg
        acc = np.asarray(self._arr_acc) if self._arr_acc else np.zeros((0, 2))
        self._arr_acc = []
        # the array is centred on the robot: compare array-mean readings
        c = self._c_arr_f if self._c_arr_f is not None else c
        self._confirm_pts.append((c, p.copy()))
        best_c, best_p = max(self._confirm_pts, key=lambda e: e[0])
        # ---- reading dropped clearly below the best point: go back, smaller r
        if c < best_c * (1.0 - cfg.confirm_tol) and self._confirm_moves < cfg.max_confirm_moves:
            self._confirm_moves += 1
            self._probe_r *= 0.5
            self._came_from = None
            self.phase = "confirm"
            self._goto_centre(t, p, yaw, best_p)
            return
        # ---- averaged simultaneous gradient
        snr = 0.0
        g_avg = np.zeros(2)
        if len(acc) >= 3:
            g_avg = acc.mean(axis=0)
            cov = np.cov(acc.T) / len(acc) + np.eye(2) * 1e-6
            gn = float(np.hypot(*g_avg))
            if gn > 1e-9:
                u = g_avg / gn
                snr = gn / math.sqrt(float(u @ cov @ u))
        self.confirm_snr = snr
        if (snr >= cfg.confirm_snr and self._confirm_moves < cfg.max_confirm_moves
                and self._probe_r >= cfg.probe_min_radius):
            d = _unit(g_avg)
            if self._came_from is not None:
                cosang = float(d @ self._came_from)
                if cosang < 0.0:            # overshoot: direction reversed
                    self._probe_r *= 0.5
                elif cosang > 0.7:          # consistent direction: lengthen
                    self._probe_r = min(cfg.probe_max_radius, 1.25 * self._probe_r)
            if self._probe_r >= cfg.probe_min_radius:
                fl = self.free_length(p, d, self._probe_r)
                if fl < 0.1:
                    # gradient points into an obstacle: slide along its surface
                    n = self._clearance_dir(p[0], p[1])
                    tang = d - float(d @ n) * n
                    if float(np.hypot(*tang)) > 0.3:
                        d = _unit(tang)
                        fl = self.free_length(p, d, self._probe_r)
                if fl >= 0.1:
                    self._confirm_moves += 1
                    self._came_from = d
                    self.phase = "confirm"
                    self._go(t, p + d * fl, yaw)
                    return
        # ---- converged: flat (insignificant gradient) at the best point visited
        if max(best_c, c) >= cfg.found_ppm:
            self._found_xy = self._confirmed_estimate()
            self._set_state(FOUND, t, "local maximum confirmed" if snr < cfg.confirm_snr
                            else "confirmation budget used")
            return
        # local maximum below found_ppm (plume meander / plateau): keep climbing
        self._l_ref, self._p_ref = float(self._log(c)), nose.copy()
        self._alpha = cfg.step_init
        self._mom = np.zeros(2)
        if not self._plan_step(t, p, yaw):
            self._set_state(SURGE, t, "weak local maximum")

    def _off_pipe(self, target: np.ndarray) -> np.ndarray:
        """``target`` moved out of the crossing zone of a lying pipe: to the
        zone's edge along the pipe normal, on the target's side (on the axis:
        on the robot's side), when that spot is free; else unchanged."""
        gov = self.crossing
        if gov is None:
            return target
        q = np.asarray(target, dtype=float).copy()
        zone = float(gov.p.zone)
        here = self.path[-1] if self.path else None
        for pp in gov.pipes:
            s = pp.side(q[0], q[1])
            if abs(s) >= zone or abs(pp.along(q[0], q[1])) > pp.half + pp.r + gov.p.end_margin:
                continue
            ref = s if (abs(s) > 0.05 or here is None) else pp.side(here[0], here[1])
            cand = q + ((zone if ref >= 0.0 else -zone) - s) * pp.n
            if self.clearance(cand[0], cand[1]) >= self.cfg.robot_radius:
                q = cand
        return q

    def _govern(self, x: float, y: float, yaw: float, cmd) -> Tuple[float, float, float]:
        """The tracking command through the pipe crossing governor (module docstring)."""
        gov = self.crossing
        if gov is None:
            return cmd
        if self._target is None:
            ahead = None
        elif self._route is not None:
            ahead = self._route[self._route_i:]
        else:
            ahead = self._target.reshape(1, 2)
        vx, vy, wz = gov.govern(x, y, yaw, cmd, ahead, recheck=True)
        return float(vx), float(vy), float(wz)

    # ============================================================= control
    def _track(self, t: float, p: np.ndarray, yaw: float) -> Tuple[float, float, float]:
        """Velocity command towards the current target with obstacle projection."""
        cfg = self.cfg
        if self._target is None:
            return 0.0, 0.0, 0.0
        yaw_des = self._yaw_des
        if self._route is not None:
            # detour: advance through the waypoints, full speed towards the
            # intermediate ones while facing the direction of travel
            r = self._route
            while self._route_i < len(r) - 1 and float(np.hypot(*(r[self._route_i] - p))) < cfg.waypoint_tolerance:
                self._route_i += 1
            aim = r[self._route_i]
            e = aim - p
            if self._route_i < len(r) - 1:
                v = _unit(e) * cfg.max_speed
                yaw_des = math.atan2(e[1], e[0])
            else:
                v = cfg.k_pos * e
        else:
            e = self._target - p
            v = cfg.k_pos * e
        sp = float(np.hypot(*v))
        if sp > cfg.max_speed:
            v *= cfg.max_speed / sp
        # obstacle handling: remove the inward component near walls + repulsion
        clr = self.clearance(p[0], p[1])
        if clr < cfg.repulse_dist:
            n = self._clearance_dir(p[0], p[1])
            if n.any():
                s = min(1.0, max(0.0, (clr - 0.5 * cfg.robot_radius) / max(cfg.repulse_dist - 0.5 * cfg.robot_radius, 1e-6)))
                vn = float(v @ n)
                if vn < 0.0:
                    v = v - (1.0 - s) * vn * n
                v = v + cfg.repulse_gain * (1.0 - s) * n
                sp = float(np.hypot(*v))
                if sp > cfg.max_speed:
                    v *= cfg.max_speed / sp
        # heading
        err = _wrap(yaw_des - yaw) if yaw_des is not None else 0.0
        wz = float(np.clip(cfg.k_yaw * err, -cfg.max_wz, cfg.max_wz))
        # turn before walking fast when the heading error is large (legged robot)
        if abs(err) > math.radians(60):
            v *= 0.3
        c, s_ = math.cos(yaw), math.sin(yaw)
        vx = c * v[0] + s_ * v[1]
        vy = -s_ * v[0] + c * v[1]
        if abs(vy) > cfg.max_vy:
            k = cfg.max_vy / abs(vy)
            vx *= k
            vy *= k
        return float(vx), float(vy), wz

    # ============================================================ outputs
    def best_sample(self) -> Optional[Tuple[float, float, float]]:
        if not self._samples:
            return None
        S = np.asarray(self._samples)
        k = int(np.argmax(S[:, 2]))
        return float(S[k, 0]), float(S[k, 1]), float(S[k, 2])

    def estimate(self) -> Optional[np.ndarray]:
        """Source estimate = weighted centroid of the top-k spatial bins.

        Samples are first averaged in ``estimate_bin`` cells (a dwell produces
        many readings at one spot; binning keeps one vote per location), then
        x_hat = sum w_i p_i / sum w_i over the k highest bins with
        w_i = c_i - c_(k+1).  None before any detection.
        """
        if self._found_xy is not None:
            return self._found_xy.copy()
        if not self._samples:
            return None
        S = np.asarray(self._samples)
        if float(S[:, 2].max()) < self.cfg.detect_ppm:
            return None
        b = max(self.cfg.estimate_bin, 1e-3)
        keys = np.floor(S[:, :2] / b).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        cnt = np.bincount(inv).astype(float)
        cb = np.bincount(inv, weights=S[:, 2]) / cnt
        xb = np.bincount(inv, weights=S[:, 0]) / cnt
        yb = np.bincount(inv, weights=S[:, 1]) / cnt
        order = np.argsort(-cb)
        k = min(self.cfg.top_k, len(order))
        top = order[:k]
        ref = cb[order[k]] if len(order) > k else 0.9 * cb[top[-1]]
        w = np.maximum(cb[top] - ref, 1e-6)
        return np.array([float(np.sum(w * xb[top]) / w.sum()), float(np.sum(w * yb[top]) / w.sum())])

    @property
    def concentration(self) -> float:
        return float(self._c_f) if self._c_f is not None else 0.0

    @property
    def gradient(self) -> GradientEstimate:
        return self._grad

    def status(self, t: Optional[float] = None) -> dict:
        g = self._grad
        c = self.concentration
        est = self.estimate()
        best = self.best_sample()
        if t is None or self.state == IDLE:
            elapsed = 0.0
        else:   # frozen once the run has ended (FOUND / FAILED)
            elapsed = max(0.0, (self._t_end if self._t_end is not None else t) - self._t_start)
        return {
            "state": self.state,
            "concentration": c,
            "gradient": [float(g.g[0] * (c + self.cfg.log_offset_ppm)),
                         float(g.g[1] * (c + self.cfg.log_offset_ppm))],
            "gradient_log": [float(g.g[0]), float(g.g[1])],
            "gradient_snr": float(g.snr),
            "estimate": None if est is None else [float(est[0]), float(est[1])],
            "error_m": None,
            "path_length": float(self.path_length),
            "elapsed": float(elapsed),
            "phase": self.phase,
            "step": float(self._alpha),
            "max_ppm": None if best is None else best[2],
            "n_samples": len(self._samples),
            "detours": int(self.detours),
            "message": self.message,
            "obstacles": self.obstacles.confirmed_count,
            "crossing": self.crossing.state if self.crossing is not None else "",
        }
