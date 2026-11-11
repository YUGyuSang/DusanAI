"""
Online slip detection and friction estimation (ROS free, numpy only).

A stance foot should not move in the world.  Every control step the measured
foot centres in the world (leg FK of the measured joints + odometry pose /
IMU attitude, see ``LocomotionController._feet_world_measured``) are pushed
into a per-leg history that restarts at every touch-down.  After a settle
time (touch-down impact, the foot sphere seating) the horizontal foot speed
over a sliding window ``window`` is

    v_i = |p_i(t) - p_i(t - window)| / window

A leg *slips* while v_i > ``speed_threshold``.  A slip *vote* needs at least
``min_legs`` slipping stance legs whose tangential / normal ground-force
ratio (below) is under ``max_force_ratio``: feet also slide on good ground
when a large tangential force exceeds its friction (a 15 N s push on the
mu 0.9 floor: ratios 0.7-1.5 in the MuJoCo check), which says nothing about
the floor, whereas sliding under a small force ratio (0.15-0.45 on the
mu 0.15 plate) is evidence of low friction.  Without force estimates the
speed alone decides.  Slip evidence is integrated (leaky: + dt per vote,
- ``decay`` * dt otherwise) and the slip flag is raised when the evidence reaches
``confirm_time``: a single touch-down jolt or one noisy odometry sample does
not trigger, sustained sliding does.  The flag is released after
``release_time`` seconds AND ``release_distance`` metres of body travel
without a slip vote (hysteresis): the slow, feed-forward slip mode makes the
feet stop sliding on the plate, so "no slip for a while" alone would hand
back the fast gait in the middle of the plate; a slippery area is assumed
to extend ``release_distance`` beyond the last slip.

Friction estimate ``mu_est``: while a foot is sliding, Coulomb friction makes
its tangential / normal ground-force ratio equal to the (kinetic) friction
coefficient.  The ground forces are estimated from the joint servo law of the
gz JointPositionController (PID force mode, see simulation/urdf):
``tau = kp (q_target - q)`` and ``f = -J^-T tau`` (quasi-static leg), so

    mu_obs = median_i |f_t,i| / f_n,i      over sliding legs with f_n > min_normal_force

and ``mu_est`` drops to ``max(mu_min, margin * mu_obs)`` (never rises while
slipping).  Without usable force estimates ``mu_est`` is multiplied by
``mu_drop`` at every new slip detection.  After the flag is released the
estimate recovers towards the nominal coefficient with time constant
``recover_tau`` (slowly: a plate may continue after a short grippy patch).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, fields as dc_fields
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SlipParams:
    speed_threshold: float = 0.08        # [m/s] horizontal stance-foot speed counted as slipping
    window: float = 0.06                 # [s] finite-difference window of the foot speed
    settle_time: float = 0.03            # [s] ignored after each touch-down
    min_legs: int = 2                    # slipping stance legs needed for a slip vote
    confirm_time: float = 0.10           # [s] integrated slip evidence that raises the flag
    decay: float = 0.5                   # evidence decay rate (x dt) without a slip vote
    release_time: float = 2.0            # [s] without a slip vote before the flag is released ...
    release_distance: float = 3.0        # [m] ... and this much body travel since the last slip vote
    mu_nominal: float = 0.5              # coefficient without slip (= mpc_mu)
    mu_min: float = 0.15                 # lower bound of the estimate
    mu_margin: float = 1.0               # mu_est = margin * observed force ratio
    mu_drop: float = 0.6                 # factor per detection without force estimates
    recover_tau: float = 8.0             # [s] recovery time constant after the release
    min_normal_force: float = 20.0       # [N] legs with less normal force give no ratio
    max_force_ratio: float = 0.4         # sliding legs with a higher |f_t| / f_n do not vote (push, not friction)

    @classmethod
    def from_controller_params(cls, p: Dict) -> "SlipParams":
        """Map the flat ``slip_*`` controller parameters onto SlipParams."""
        kw = {}
        for f in dc_fields(cls):
            key = "slip_" + f.name
            if key in p:
                kw[f.name] = type(f.default)(p[key])
        if "mpc_mu" in p:
            kw["mu_nominal"] = float(p["mpc_mu"])
        return cls(**kw)


@dataclass
class SlipState:
    """Result of :meth:`SlipDetector.update`."""
    slip: bool = False                   # hysteretic slip flag
    detected: bool = False               # flag raised in this update
    cleared: bool = False                # flag released in this update
    legs: List[bool] = field(default_factory=lambda: [False] * 4)   # legs slipping now
    speeds: List[float] = field(default_factory=lambda: [0.0] * 4)  # [m/s] stance-foot speeds (0 = n/a)
    speed: float = 0.0                   # max stance-foot speed
    evidence: float = 0.0                # [s]
    mu_est: float = 0.5
    mu_obs: float = float("nan")         # force-ratio observation of this update (NaN = none)


class SlipDetector:
    """See the module docstring."""

    def __init__(self, params: Optional[SlipParams] = None):
        self.p = params or SlipParams()
        self.reset()

    def reset(self) -> None:
        self.slip = False
        self.evidence = 0.0
        self.mu_est = float(self.p.mu_nominal)
        self.t_last_slip = -math.inf
        self.dist_since_slip = math.inf          # body path length since the last slip vote [m]
        self._xy_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None
        self.detections = 0
        self._hist: List[deque] = [deque() for _ in range(4)]
        self._stance_t0 = [None] * 4
        self.state = SlipState(mu_est=self.mu_est)

    # ------------------------------------------------------------ helpers
    def _leg_speed(self, leg: int, t: float, xy: np.ndarray) -> float:
        """Speed over the window (0 when the stance is too young / too short)."""
        p = self.p
        h = self._hist[leg]
        h.append((t, float(xy[0]), float(xy[1])))
        while len(h) > 2 and t - h[1][0] >= p.window - 1e-9:
            h.popleft()
        t0, x0, y0 = h[0]
        span = t - t0
        if span < 0.7 * p.window or span <= 1e-6:
            return 0.0
        return math.hypot(xy[0] - x0, xy[1] - y0) / span

    @staticmethod
    def force_ratios(forces_w: Optional[np.ndarray], legs, min_fn: float) -> List[float]:
        """|f_t| / f_n of the given legs (world ground forces ON the feet)."""
        if forces_w is None:
            return []
        f = np.asarray(forces_w, dtype=float).reshape(4, 3)
        out = []
        for i in legs:
            fn = float(f[i, 2])
            if math.isfinite(fn) and fn >= min_fn and np.all(np.isfinite(f[i])):
                out.append(float(math.hypot(f[i, 0], f[i, 1]) / fn))
        return out

    # ------------------------------------------------------------ update
    def update(self, t: float, feet_w, stance, forces_w: Optional[np.ndarray] = None,
               body_xy=None) -> SlipState:
        """One control step.

        feet_w   (4, 3) measured world foot centres
        stance   (4,) bool, legs in stance (gait schedule, not swinging)
        forces_w (4, 3) estimated world ground forces on the feet (optional)
        body_xy  (2,) body position for the release distance (None: time only)"""
        p = self.p
        t = float(t)
        if self.t_prev is not None and t < self.t_prev - 0.5:
            self.reset()                         # clock jumped back (simulation reset)
        dt = 0.0 if self.t_prev is None else max(0.0, min(t - self.t_prev, 0.05))
        self.t_prev = t
        if body_xy is not None:
            xy = np.asarray(body_xy, dtype=float).reshape(2)
            if np.all(np.isfinite(xy)):
                if self._xy_prev is not None and math.isfinite(self.dist_since_slip):
                    self.dist_since_slip += float(np.hypot(*(xy - self._xy_prev)))
                self._xy_prev = xy
        feet = np.asarray(feet_w, dtype=float).reshape(4, 3)
        stance = np.asarray(stance, dtype=bool).reshape(4)
        speeds = [0.0] * 4
        legs = [False] * 4
        for i in range(4):
            if not stance[i] or not np.all(np.isfinite(feet[i])):
                self._stance_t0[i] = None
                self._hist[i].clear()
                continue
            if self._stance_t0[i] is None:
                self._stance_t0[i] = t
                self._hist[i].clear()
            if t - self._stance_t0[i] < p.settle_time - 1e-9:
                self._hist[i].clear()            # the history starts after the settle time
                continue
            v = self._leg_speed(i, t, feet[i, :2])
            speeds[i] = v
            legs[i] = v > p.speed_threshold
        n_slip = sum(legs)
        ratios = self.force_ratios(forces_w, [i for i in range(4) if legs[i]], p.min_normal_force)
        vote = n_slip >= max(1, int(p.min_legs))
        if vote and ratios and float(np.median(ratios)) >= p.max_force_ratio:
            vote = False                         # large tangential force: overload, not low friction
        if vote:
            self.evidence = min(p.confirm_time, self.evidence + dt)
            self.t_last_slip = t
            self.dist_since_slip = 0.0 if body_xy is not None else math.inf
        else:
            self.evidence = max(0.0, self.evidence - p.decay * dt)
        detected = cleared = False
        if not self.slip and vote and self.evidence >= p.confirm_time - 1e-9:
            self.slip = True
            detected = True
            self.detections += 1
        elif (self.slip and t - self.t_last_slip >= p.release_time
              and self.dist_since_slip >= p.release_distance):
            self.slip = False
            cleared = True
        # ---------------- friction estimate
        mu_obs = float("nan")
        if self.slip and vote:
            if ratios:
                mu_obs = float(np.median(ratios))
                target = float(np.clip(p.mu_margin * mu_obs, p.mu_min, p.mu_nominal))
                self.mu_est = min(self.mu_est, target)
            elif detected:
                self.mu_est = max(p.mu_min, self.mu_est * p.mu_drop)
        elif not self.slip and dt > 0.0:
            a = min(1.0, dt / max(p.recover_tau, 1e-3))
            self.mu_est += a * (p.mu_nominal - self.mu_est)
        self.mu_est = float(np.clip(self.mu_est, p.mu_min, p.mu_nominal))
        self.state = SlipState(slip=self.slip, detected=detected, cleared=cleared, legs=legs,
                               speeds=[float(v) for v in speeds], speed=float(max(speeds)),
                               evidence=float(self.evidence), mu_est=self.mu_est, mu_obs=mu_obs)
        return self.state
