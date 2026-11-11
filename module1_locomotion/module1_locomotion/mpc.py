"""
Convex centroidal MPC with a 5-second non-uniform horizon (ROS free).

Model (Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3 Through Convex
Model-Predictive Control", IROS 2018) — single rigid body, state

    x = [roll pitch yaw | px py pz | wx wy wz | vx vy vz | g]       (13)

(orientation as ZYX Euler angles, angular velocity w and linear velocity v in
the world frame, the constant gravity state g = -9.81 m/s^2).  With the GRFs
f_i (world frame, force of the ground on foot i) and r_i the vector from the
CoM to the contact point:

    d/dt [Theta]  = Rz(psi)^T w              (small roll / pitch)
    d/dt p        = v
    d/dt w        = I_w^-1 sum_i r_i x f_i + alpha_bias
    d/dt v        = sum_i f_i / m + [0 0 g] + a_bias
    I_w           = Rz(psi) I_body Rz(psi)^T

linearised about the *reference yaw* psi_k (and the reference CoM position for
the moment arms) of every step.  The affine terms (gravity, plus the optional
disturbance-observer biases ``a_bias`` / ``alpha_bias``) are carried by the
13th state through the column A[:, 12].

Discretisation: for this model the matrix M = [[A, B], [0, 0]] is nilpotent
with M^3 = 0 (A^3 = 0 and A^2 B = 0), therefore the zero-order-hold
discretisation is *exact* and cheap:

    A_d = I + A dt + A^2 dt^2 / 2 ,      B_d = B dt + A B dt^2 / 2 .

Horizon: 10 x 0.05 s + 18 x 0.25 s = 5.0 s (28 steps).  Because the coarse steps
are longer than a trot stance, each step uses the *contact fraction* c_ik of
leg i during step k (from the predicted gait schedule): the GRF of a coarse
step is the time-averaged force, bounded by  fmin c <= f_z <= fmax c, and the
lever arm is the contact-time-weighted average of the footholds used in that
step.  Legs with c < contact_threshold are swing legs and their forces are
removed from the problem (f = 0 exactly).

Condensed QP over the active GRFs U (x_k = Phi_k x_0 + Gamma_k U):

    min  sum_k w_k (x_k - x_ref,k)' Q (x_k - x_ref,k) + sum_k w_k alpha ||u_k||^2
    s.t. |f_x| <= mu f_z ,  |f_y| <= mu f_z ,  fmin c <= f_z <= fmax c

with the step weights w_k = dt_k / dt_0 (time integral of the cost) and an
optional exponential discount of the far horizon.  Solved by the dense ADMM
solver :mod:`qp` with warm start from the previous solution shifted in time.

Footholds over the horizon are predicted for every future touchdown from the
reference trajectory with the Raibert heuristic plus a capture-point feedback
term,

    p_foot = p_hip(t_td + T_st / 2) + (T_st / 2 + k_cp sqrt(h / g)) (v - v_ref) e^(-t_td / tau)
             + (h / 2g) (v_ref x w_ref)

clamped to the leg workspace and then adjusted by an optional terrain callback
(foothold planner, terrain height).  Predicted joint torques for every step are
tau = -J(q)^T R^T f_peak with q from the IK of the predicted footholds relative
to the predicted body pose (f_peak = f / c is the force during contact).

Crawl (three-leg support): the CoM reference is swayed towards the centroid of
the planned support triangle (``sway_gain`` = fraction of the way, Hann
smoothing ``sway_smooth``, faded in over ``sway_ramp`` after the crawl
starts), the path along which the footholds are planned is
anchored at the gait-cycle average of the measured position (the sway cancels,
:meth:`ConvexMPC._crawl_anchor`) and the capture-point term uses the velocity
error with respect to the swayed reference.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from dusan_common import robot_model as rm

from . import kinematics as kin
from .qp import OK_STATUSES, QPSolver

NX = 13
NU = 12
HORIZON_DTS = tuple([0.05] * 10 + [0.25] * 18)

# default state weights  [roll pitch yaw  x y z  wx wy wz  vx vy vz]
DEFAULT_Q = (30.0, 30.0, 10.0, 4.0, 4.0, 60.0, 0.5, 0.5, 0.5, 2.0, 2.0, 1.0)


def _rz(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def euler_to_rot_batch(roll, pitch, yaw) -> np.ndarray:
    """Vectorised R = Rz(yaw) Ry(pitch) Rx(roll), shape (…, 3, 3)."""
    roll, pitch, yaw = np.broadcast_arrays(np.asarray(roll, float), np.asarray(pitch, float),
                                           np.asarray(yaw, float))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.empty(roll.shape + (3, 3))
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def _smoothstep_arr(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def terrain_heights(height_fn: Optional[Callable], xs, ys) -> np.ndarray:
    """Evaluate a terrain height function on arrays (vectorised if supported,
    point-wise otherwise).  Unknown heights / errors -> NaN."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    out = np.full(xs.shape, np.nan)
    if height_fn is None or xs.size == 0:
        return out
    try:
        h = np.asarray(height_fn(xs, ys), dtype=float)
        if h.shape == xs.shape:
            return h
    except Exception:
        pass
    flat_x, flat_y, flat_o = xs.ravel(), ys.ravel(), out.ravel()
    for i in range(flat_x.size):
        try:
            flat_o[i] = float(height_fn(float(flat_x[i]), float(flat_y[i])))
        except Exception:
            flat_o[i] = np.nan
    return flat_o.reshape(xs.shape)


# ---------------------------------------------------------------------------
# Configuration / IO
# ---------------------------------------------------------------------------
@dataclass
class MPCConfig:
    mass: float = rm.TOTAL_MASS
    inertia: Sequence[float] = rm.MPC_BODY_INERTIA
    gravity: float = 9.81
    mu: float = 0.5
    f_min: float = 1.0                   # > 0 keeps the friction-pyramid apex (degenerate) inactive
    f_max: float = 250.0
    dts: Sequence[float] = HORIZON_DTS
    q_weights: Sequence[float] = DEFAULT_Q
    r_weight: float = 2e-4
    time_weighting: bool = True
    tail_decay: float = 0.0              # [s] exponential discount of stage costs (0 = off)
    contact_threshold: float = 0.05
    sample_dt: float = 0.0025
    # footholds
    capture_gain: float = 0.6            # k_cp
    feedback_tau: float = 0.5            # [s] decay of the velocity-error terms over the horizon
    max_foot_dx: float = 0.16
    max_foot_dy: float = 0.10
    terrain_horizon: float = 1.2         # [s] footholds adjusted by the foothold planner
    max_pitch: float = 0.30
    terrain_pitch_gain: float = 1.0
    ground_smooth: float = 0.2           # [s] smoothing window of the terrain height reference
    terrain_robust: bool = True          # terrain height / pitch reference: robust fit over the body footprint
                                         # (see ConvexMPC._robust_terrain) instead of the 4 nominal footholds
    terrain_columns: int = 9             # footprint samples along the body (x 3 lines)
    sway_gain: float = 0.5               # crawl sway: fraction of the way to the support centroid
    sway_smooth: float = 0.5             # [s] Hann window of the sway reference
    linearisation_tau: float = 0.5       # [s] blend from predicted to reference CoM path (moment arms)
    anchor_window: float = 0.0           # [s] crawl path anchor averaging window (0 = one gait period)
    sway_ramp: float = 0.6               # [s] fade-in of the crawl sway after the crawl starts (0 = off)
    # torques
    torque_check_horizon: float = 1.0    # [s] window used for the saturation ratio
    # QP
    qp_eps_abs: float = 1e-3
    qp_eps_rel: float = 1e-3
    qp_max_iter: int = 250
    qp_rho: float = 0.1
    qp_alpha: float = 1.6
    qp_sigma: float = 1e-6
    qp_scaling: int = 4
    qp_polish: bool = True
    qp_time_limit: float = 0.0           # [s] 0 = unlimited

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "MPCConfig":
        cfg = cls()
        for k, v in (d or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass
class MPCInput:
    """Measured state + command handed to :meth:`ConvexMPC.solve`."""
    t: float                               # absolute time of the measurement [s]
    pos: Sequence[float]                   # world CoM / base position
    rpy: Sequence[float]
    vel: Sequence[float]                   # world linear velocity
    omega: Sequence[float]                 # world angular velocity
    feet: Sequence[Sequence[float]]        # (4,3) current world foot centres
    cmd: Sequence[float] = (0.0, 0.0, 0.0)  # body-frame (vx, vy, wz)
    body_height: float = rm.NOMINAL_BODY_HEIGHT
    ground_z: float = 0.0                  # fallback terrain height under the robot
    pitch_ref: float = 0.0                 # fallback terrain pitch
    hold_xy: Optional[Sequence[float]] = None   # stand: hold this xy instead of integrating cmd
    hold_yaw: Optional[float] = None
    nominal_feet_xy: Optional[Sequence[Sequence[float]]] = None   # (4,2) body-frame default footholds
    accel_bias: Sequence[float] = (0.0, 0.0, 0.0)
    ang_accel_bias: Sequence[float] = (0.0, 0.0, 0.0)
    # friction coefficient of this solve (None = MPCConfig.mu); the controller
    # lowers it on slippery ground (slip estimate / friction prior).  Passed
    # with the job, so changing it never races a solve in the worker thread.
    mu: Optional[float] = None


@dataclass
class MPCResult:
    t: float                         # absolute time of the solve input
    times: np.ndarray                # (N+1,) relative knot times 0 .. 5.0 s
    dts: np.ndarray                  # (N,)
    x_pred: np.ndarray               # (N+1, 13) predicted state
    x_ref: np.ndarray                # (N+1, 13) reference
    forces: np.ndarray               # (N, 12) world GRFs (time-averaged per step)
    contacts: np.ndarray             # (N, 4) contact fractions
    footholds: np.ndarray            # (N, 4, 3) foot centre per step (stance: contact, swing: next foothold)
    torques: np.ndarray              # (N, 12) predicted joint torques during contact
    joint_angles: np.ndarray         # (N, 12) IK of the predicted footholds
    foothold_events: list            # [(leg, t_touchdown_rel, x, y, z)]
    status: str
    solve_time: float                # wall-clock time of solve() [s]
    qp_iterations: int = 0
    solve_cpu_time: float = float("nan")   # CPU time of the solving thread [s] (load independent)
    cost: float = float("nan")
    torque_ratio: float = 0.0        # max |tau| / effort over torque_check_horizon
    torque_horizon_max: np.ndarray = field(default_factory=lambda: np.zeros(12))
    reachable: bool = True
    mu: float = float("nan")         # friction coefficient of the pyramid used in this solve

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES

    @property
    def horizon(self) -> float:
        return float(self.times[-1])

    def state_at(self, t_rel: float) -> np.ndarray:
        """Linear interpolation of the predicted state at relative time t_rel."""
        t_rel = float(np.clip(t_rel, self.times[0], self.times[-1]))
        return np.array([np.interp(t_rel, self.times, self.x_pred[:, j]) for j in range(NX)])

    def ref_at(self, t_rel: float) -> np.ndarray:
        """Linear interpolation of the reference state at relative time t_rel."""
        t_rel = float(np.clip(t_rel, self.times[0], self.times[-1]))
        return np.array([np.interp(t_rel, self.times, self.x_ref[:, j]) for j in range(NX)])

    def forces_at(self, t_rel: float) -> np.ndarray:
        """(4,3) GRFs of the step containing t_rel."""
        k = int(np.clip(np.searchsorted(self.times, t_rel, side="right") - 1, 0, len(self.dts) - 1))
        return self.forces[k].reshape(4, 3)

    def next_foothold(self, leg: int, t_rel: float = 0.0):
        """First predicted touchdown (t_td, xyz) of ``leg`` after t_rel, or None."""
        best = None
        for (lg, ttd, x, y, z) in self.foothold_events:
            if lg == leg and ttd >= t_rel - 1e-9 and (best is None or ttd < best[0]):
                best = (ttd, np.array([x, y, z]))
        return best


# ---------------------------------------------------------------------------
# Disturbance observer (payload / CoM offset adaptation)
# ---------------------------------------------------------------------------
class AccelBiasObserver:
    """First-order estimate of constant world-frame linear / angular
    acceleration biases (unmodelled payload, CoM offset, slope drag) from the
    measured velocity derivatives and the applied GRFs.  The biases enter the
    MPC model through the affine column (MPCInput.accel_bias/ang_accel_bias)."""

    def __init__(self, mass: float = rm.TOTAL_MASS, inertia=rm.MPC_BODY_INERTIA, tau: float = 0.5,
                 max_lin: float = 4.0, max_ang: float = 20.0, gravity: float = 9.81):
        self.m = float(mass)
        self.Ib = np.diag(np.asarray(inertia, dtype=float))
        self.tau = float(tau)
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.g = float(gravity)
        self.reset()

    def reset(self) -> None:
        self.a_bias = np.zeros(3)
        self.alpha_bias = np.zeros(3)
        self._prev = None

    def update(self, t: float, vel_world, omega_world, yaw: float, feet_world, com_world,
               forces_world) -> None:
        v = np.asarray(vel_world, dtype=float)
        w = np.asarray(omega_world, dtype=float)
        if self._prev is not None:
            t0, v0, w0, f_tot, tau_tot, Iw_inv = self._prev
            dt = t - t0
            if 1e-4 < dt < 0.2:
                a_meas = (v - v0) / dt
                a_model = f_tot / self.m + np.array([0.0, 0.0, -self.g])
                k = min(1.0, dt / self.tau)
                self.a_bias += k * ((a_meas - a_model) - self.a_bias)
                al_meas = (w - w0) / dt
                al_model = Iw_inv @ tau_tot
                self.alpha_bias += k * ((al_meas - al_model) - self.alpha_bias)
                n = np.linalg.norm(self.a_bias)
                if n > self.max_lin:
                    self.a_bias *= self.max_lin / n
                n = np.linalg.norm(self.alpha_bias)
                if n > self.max_ang:
                    self.alpha_bias *= self.max_ang / n
        f = np.asarray(forces_world, dtype=float).reshape(4, 3)
        r = np.asarray(feet_world, dtype=float).reshape(4, 3) - np.asarray(com_world, dtype=float)
        Rz = _rz(yaw)
        Iw_inv = Rz @ np.linalg.inv(self.Ib) @ Rz.T
        self._prev = (t, v.copy(), w.copy(), f.sum(axis=0), np.cross(r, f).sum(axis=0), Iw_inv)


# ---------------------------------------------------------------------------
# MPC
# ---------------------------------------------------------------------------
class ConvexMPC:
    """5-second-horizon convex centroidal MPC (see module docstring)."""

    def __init__(self, config: Optional[MPCConfig] = None, **overrides):
        self.cfg = config if config is not None else MPCConfig()
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise AttributeError(f"unknown MPC option '{k}'")
            setattr(self.cfg, k, v)
        self._setup()
        self._prev = None
        self._anchor_hist: list = []
        self._crawl_t0: Optional[float] = None
        self._sway_state = None
        self.last_result: Optional[MPCResult] = None

    def _setup(self) -> None:
        c = self.cfg
        self.dts = np.asarray(c.dts, dtype=float)
        self.N = len(self.dts)
        self.knots = np.concatenate([[0.0], np.cumsum(self.dts)])
        self.T = float(self.knots[-1])
        sdt = float(c.sample_dt)
        # sample grid aligned with the knots (every dt must be a multiple of sample_dt)
        self.n_per_step = np.maximum(1, np.round(self.dts / sdt).astype(int))
        self.sample_dt = sdt
        self.S = int(self.n_per_step.sum())
        self.step_start = np.concatenate([[0], np.cumsum(self.n_per_step)[:-1]])
        self.ts = (np.arange(self.S) + 0.5) * sdt
        self.sample_step = np.repeat(np.arange(self.N), self.n_per_step)
        self.Ib = np.diag(np.asarray(c.inertia, dtype=float))
        self.Ib_inv = np.linalg.inv(self.Ib)
        qd = np.zeros(NX)
        qd[:12] = np.asarray(c.q_weights, dtype=float)[:12]
        self.qdiag = qd
        w = self.dts / self.dts[0] if c.time_weighting else np.ones(self.N)
        if c.tail_decay and c.tail_decay > 0:
            w = w * np.exp(-self.knots[1:] / c.tail_decay)
        self.step_w = w
        tl = c.qp_time_limit if c.qp_time_limit and c.qp_time_limit > 0 else None
        self.qp = QPSolver(rho=c.qp_rho, sigma=c.qp_sigma, alpha=c.qp_alpha, eps_abs=c.qp_eps_abs,
                           eps_rel=c.qp_eps_rel, max_iter=c.qp_max_iter, scaling=c.qp_scaling,
                           time_limit=tl, polish=c.qp_polish)
        self.blk = self._friction_block(float(c.mu))

    @staticmethod
    def _friction_block(mu: float) -> np.ndarray:
        """5 constraint rows per active leg: friction pyramid + normal force."""
        return np.array([[1.0, 0.0, -mu], [-1.0, 0.0, -mu], [0.0, 1.0, -mu],
                         [0.0, -1.0, -mu], [0.0, 0.0, 1.0]])

    def effective_mu(self, inp: "MPCInput") -> float:
        """Friction coefficient used for ``inp``: ``inp.mu`` if given (clamped
        to (0.01, cfg.mu]), else ``cfg.mu``."""
        mu_cfg = float(self.cfg.mu)
        mu = getattr(inp, "mu", None)
        if mu is None or not math.isfinite(float(mu)):
            return mu_cfg
        return float(np.clip(float(mu), 0.01, mu_cfg))

    def reset(self) -> None:
        """Forget the warm start and the path anchor."""
        self._prev = None
        self._anchor_hist = []
        self._crawl_t0 = None
        self._sway_state = None
        self.last_result = None

    # ------------------------------------------------------------------ model
    def dynamics(self, yaw: float, r_feet, dt: float, affine=None):
        """Exact discrete (A_d, B_d) of the linearised model for one step.

        r_feet : (4,3) contact point minus CoM (world).
        affine : (6,) [alpha_bias(3), gvec(3)] — angular and linear acceleration
                 of the affine column (default: gravity only)."""
        A, B = self._continuous(yaw, r_feet, affine)
        Ad = np.eye(NX) + A * dt + (A @ A) * (0.5 * dt * dt)
        Bd = B * dt + (A @ B) * (0.5 * dt * dt)
        return Ad, Bd

    def _continuous(self, yaw, r_feet, affine=None):
        c = self.cfg
        g_state = -c.gravity
        if affine is None:
            affine = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -c.gravity])
        Rz = _rz(yaw)
        Iw_inv = Rz @ self.Ib_inv @ Rz.T
        A = np.zeros((NX, NX))
        A[0:3, 6:9] = Rz.T
        A[3:6, 9:12] = np.eye(3)
        A[6:9, 12] = affine[0:3] / g_state
        A[9:12, 12] = affine[3:6] / g_state
        B = np.zeros((NX, NU))
        r = np.asarray(r_feet, dtype=float).reshape(4, 3)
        for i in range(4):
            B[6:9, 3 * i:3 * i + 3] = Iw_inv @ _skew(r[i])
            B[9:12, 3 * i:3 * i + 3] = np.eye(3) / c.mass
        return A, B

    # ------------------------------------------------------------------ reference
    def _reference(self, inp: MPCInput, gait, height_fn, sway_active: bool = False):
        """Reference trajectory on a 10 ms grid covering the horizon + 1 s
        (un-swayed path; the crawl sway is added by :meth:`_add_sway` once the
        footholds are known)."""
        c = self.cfg
        dt = 0.01
        tg = np.arange(0.0, self.T + 1.0 + 1e-9, dt)
        vx, vy, wz = (float(v) for v in inp.cmd)
        yaw_meas = float(inp.rpy[2])
        if inp.hold_yaw is not None:
            # unwrap the held heading next to the measured one (no 2*pi jumps)
            dpsi = (float(inp.hold_yaw) - yaw_meas + math.pi) % (2.0 * math.pi) - math.pi
            yaw0 = yaw_meas + dpsi
        else:
            yaw0 = yaw_meas
        if inp.hold_xy is not None:
            wz = 0.0 if inp.hold_yaw is not None else wz
        yaw = yaw0 + wz * tg
        cy, sy = np.cos(yaw), np.sin(yaw)
        if inp.hold_xy is not None:
            vwx = np.zeros_like(tg)
            vwy = np.zeros_like(tg)
            px = np.full_like(tg, float(inp.hold_xy[0]))
            py = np.full_like(tg, float(inp.hold_xy[1]))
        else:
            vwx = cy * vx - sy * vy
            vwy = sy * vx + cy * vy
            px = float(inp.pos[0]) + np.concatenate([[0.0], np.cumsum(0.5 * (vwx[1:] + vwx[:-1]) * dt)])
            py = float(inp.pos[1]) + np.concatenate([[0.0], np.cumsum(0.5 * (vwy[1:] + vwy[:-1]) * dt)])
        nom = self._nominal_xy(inp)
        if inp.hold_xy is None and sway_active:
            # Un-swayed path anchor for the crawl (see _crawl_anchor)
            ax, ay = self._crawl_anchor(inp, gait, float(vwx[0]), float(vwy[0]))
            px = px - float(inp.pos[0]) + ax
            py = py - float(inp.pos[1]) + ay
        else:
            self._anchor_hist = []
        path_x, path_y = px.copy(), py.copy()

        # terrain height and pitch under the four nominal footholds (50 ms grid)
        tc = tg[::5]
        ycs, ysn = np.cos(yaw[::5]), np.sin(yaw[::5])
        fx = path_x[::5, None] + ycs[:, None] * nom[None, :, 0] - ysn[:, None] * nom[None, :, 1]
        fy = path_y[::5, None] + ysn[:, None] * nom[None, :, 0] + ycs[:, None] * nom[None, :, 1]
        if c.terrain_robust:
            ground, pitch = self._robust_terrain(height_fn, path_x[::5], path_y[::5], yaw[::5], nom,
                                                 float(inp.ground_z), float(inp.pitch_ref))
        else:
            hz = terrain_heights(height_fn, fx, fy)                     # (T5, 4)
            known = np.isfinite(hz)
            hz0 = np.where(known, hz, 0.0)
            cnt = known.sum(axis=1)
            ground = np.where(cnt > 0, hz0.sum(axis=1) / np.maximum(cnt, 1), float(inp.ground_z))
            n_front = known[:, :2].sum(axis=1)
            n_rear = known[:, 2:].sum(axis=1)
            zf = hz0[:, :2].sum(axis=1) / np.maximum(n_front, 1)
            zr = hz0[:, 2:].sum(axis=1) / np.maximum(n_rear, 1)
            span = max(float(np.mean(nom[:2, 0]) - np.mean(nom[2:, 0])), 0.1)
            # ZYX pitch > 0 is nose-down: terrain rising ahead -> negative pitch
            pitch_t = -np.arctan2(zf - zr, span) * c.terrain_pitch_gain
            pitch = np.where((n_front > 0) & (n_rear > 0), pitch_t, float(inp.pitch_ref))
        pitch = np.clip(pitch, -c.max_pitch, c.max_pitch)
        if c.ground_smooth > 0 and tc.size > 3:
            k = max(1, int(round(c.ground_smooth / 0.05)))
            if k > 1:
                w = np.ones(k) / k
                ground = np.convolve(np.pad(ground, (k // 2, k - 1 - k // 2), mode="edge"), w, "valid")
                pitch = np.convolve(np.pad(pitch, (k // 2, k - 1 - k // 2), mode="edge"), w, "valid")
        ground_g = np.interp(tg, tc, ground)
        pitch_g = np.interp(tg, tc, pitch)
        pz = ground_g + float(inp.body_height)
        vz = np.gradient(pz, dt)
        return {
            "t": tg, "yaw": yaw, "px": px, "py": py, "pz": pz,
            "vx": vwx.copy(), "vy": vwy.copy(), "vz": vz, "wz": wz,
            "pitch": pitch_g, "ground": ground_g, "path_x": path_x, "path_y": path_y,
            "vpx": vwx, "vpy": vwy,
        }

    def _robust_terrain(self, height_fn, px, py, yaw, nom, ground_fb: float, pitch_fb: float):
        """Terrain height under the body and terrain pitch along the path (one
        value per path sample), robust against narrow obstacles.

        The map is sampled on the body footprint: ``terrain_columns`` points
        from 5 cm behind the rear to 5 cm ahead of the front nominal footholds,
        on the left, centre and right foot lines.  Height = median of the known
        samples; pitch = Theil-Sen slope (median of all pairwise slopes) of the
        column medians.  On a ramp both equal the plane; a pipe or the depth
        map's wall-edge cells cover up to ~2/9 of the columns and are ignored
        - with the 4 nominal footholds alone a 0.10 m pipe under the front feet
        was a 12 deg 'ramp': the body pitched up and rose by 5 cm, then dropped
        back within 0.15 s when the feet had passed it, on two diagonal feet
        (Gazebo plant world: fell at the C-OBS1 pipe).  Fallbacks (too few known
        samples: no forward-looking depth data under the body yet) are the
        controller's ground / pitch estimates."""
        c = self.cfg
        nx = max(3, int(c.terrain_columns))
        xf = float(np.mean(nom[:2, 0]))
        xr = float(np.mean(nom[2:, 0]))
        y_left = float(np.max(nom[:, 1]))
        y_right = float(np.min(nom[:, 1]))
        bx = np.linspace(xr - 0.05, xf + 0.05, nx)
        by = np.array([y_right, 0.5 * (y_left + y_right), y_left])
        BX, BY = np.meshgrid(bx, by)                                  # (3, nx)
        bxf, byf = BX.ravel(), BY.ravel()
        cy, sy = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        wx = px[:, None] + cy * bxf[None, :] - sy * byf[None, :]
        wy = py[:, None] + sy * bxf[None, :] + cy * byf[None, :]
        h = terrain_heights(height_fn, wx, wy).reshape(len(px), 3, nx)
        known = np.isfinite(h)
        n_known = known.sum(axis=(1, 2))
        T = len(px)
        ground = np.full(T, float(ground_fb))
        pitch = np.full(T, float(pitch_fb))
        ok = n_known >= 3
        if not np.any(ok):
            return ground, pitch
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)       # all-NaN slices
            ground[ok] = np.nanmedian(h[ok].reshape(int(ok.sum()), -1), axis=1)
            prof = np.nanmedian(h, axis=1)                            # (T, nx) column medians
            col_ok = np.isfinite(prof)
            ii, jj = np.triu_indices(nx, 1)
            slopes = (prof[:, jj] - prof[:, ii]) / (bx[jj] - bx[ii])[None, :]
            slope = np.nanmedian(slopes, axis=1)
            # pitch only from columns spanning most of the support (else the fallback)
            xk = np.where(col_ok, bx[None, :], np.nan)
            extent = np.nanmax(xk, axis=1) - np.nanmin(xk, axis=1)
        good = ok & np.isfinite(slope) & (col_ok.sum(axis=1) >= 3) & (np.nan_to_num(extent) >= 0.6 * (xf - xr))
        pitch[good] = -np.arctan(slope[good]) * c.terrain_pitch_gain
        return ground, pitch

    def _crawl_anchor(self, inp: MPCInput, gait, vcx: float, vcy: float):
        """Start point (x, y) of the un-swayed crawl path.

        The measured position contains the intended CoM sway, so it cannot be
        used directly as the start of the path along which the footholds are
        planned.  The sway is periodic with the gait cycle, therefore the
        measured positions of the last cycle, each propagated to the present
        with the commanded displacement D(t) = integral of the commanded world
        velocity, are averaged over exactly one gait period W:

            a = mean_i ( p_i - D(t_i) ) + D(t)

        This removes the sway (every harmonic of 1/W) but keeps the feedback
        on the body's progress: a body that is slower than commanded by dv
        gets a path that starts dv * W / 2 ahead of it, so the MPC pushes it
        forward instead of the path drifting back with the body.  Samples are
        weighted by their time spacing (irregular solve intervals)."""
        t = float(inp.t)
        x, y = float(inp.pos[0]), float(inp.pos[1])
        window = float(self.cfg.anchor_window)
        if window <= 0.0:
            pat = gait.pattern_at(0.0)
            window = pat.period if not pat.is_stand else 1.0
        hist = self._anchor_hist
        if hist and 0.0 < t - hist[-1][0] < 0.5:
            t_l, _xl, _yl, dx_l, dy_l, vx_l, vy_l = hist[-1]
            h = t - t_l
            dxc = dx_l + 0.5 * (vx_l + vcx) * h          # trapezoidal commanded displacement
            dyc = dy_l + 0.5 * (vy_l + vcy) * h
        else:
            # (re)start of the crawl: seed one window of past samples, the body
            # moving with its measured velocity, so that the first sway
            # excursions enter the average gradually instead of dragging the
            # path sideways (a lasting lateral offset otherwise)
            hist.clear()
            self._crawl_t0 = t
            dxc = dyc = 0.0
            vm = np.asarray(inp.vel, dtype=float)
            for back in np.arange(window, 1e-9, -0.05):
                hist.append((t - back, x - vm[0] * back, y - vm[1] * back, -vcx * back, -vcy * back, vcx, vcy))
        hist.append((t, x, y, dxc, dyc, vcx, vcy))
        while len(hist) > 1 and t - hist[0][0] > window - 1e-6:
            hist.pop(0)
        if len(hist) == 1:
            return x, y
        a = np.asarray(hist, dtype=float)
        ts = a[:, 0]
        # time-spacing weights (all equal for a regular solve rate)
        gaps = np.diff(ts)
        w = np.empty(len(ts))
        w[0] = gaps[0]
        w[-1] = gaps[-1]
        w[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
        w /= w.sum()
        ax = float(w @ (a[:, 1] - a[:, 3])) + dxc
        ay = float(w @ (a[:, 2] - a[:, 4])) + dyc
        return ax, ay

    def _add_sway(self, inp: MPCInput, ref, C, F) -> None:
        """Crawl CoM sway towards the support polygon (static stability).

        On the sample grid the centroid of the *actual / predicted* stance
        footholds is computed, smoothed with a Hann window of ``sway_smooth``
        seconds and the reference becomes

            p_ref(t) = path(t) + sway_gain * (centroid_smoothed(t) - path(t))

        i.e. with gain 1 the CoM is steered to the support-triangle centroid of
        the planned footholds (positive static margin during three-leg support,
        the transfer happens while the support polygons overlap)."""
        c = self.cfg
        ts = self.ts
        cs = C.astype(float)
        n = cs.sum(axis=1)
        cen = (cs[:, :, None] * F[:, :, :2]).sum(axis=1) / np.maximum(n, 1.0)[:, None]
        path = np.stack([np.interp(ts, ref["t"], ref["path_x"]), np.interp(ts, ref["t"], ref["path_y"])], axis=1)
        off = cen - path
        L = max(3, int(round(max(c.sway_smooth, 2 * self.sample_dt) / self.sample_dt)) | 1)
        w = np.hanning(L)
        w /= w.sum()
        pad = L // 2
        off_s = np.stack([np.convolve(np.pad(off[:, j], pad, mode="edge"), w, mode="valid")
                          for j in range(2)], axis=1) * float(c.sway_gain)
        if c.sway_ramp > 0 and self._crawl_t0 is not None:
            # fade the sway in over the first gait cycle after the crawl started
            ramp = np.clip((float(inp.t) + ts - self._crawl_t0) / float(c.sway_ramp), 0.0, 1.0)
            off_s *= _smoothstep_arr(ramp)[:, None]
        tg = ref["t"]
        sx = np.interp(tg, ts, off_s[:, 0])
        sy = np.interp(tg, ts, off_s[:, 1])
        dt = tg[1] - tg[0]
        ref["px"] = ref["path_x"] + sx
        ref["py"] = ref["path_y"] + sy
        ref["vx"] = ref["vpx"] + np.gradient(sx, dt)
        ref["vy"] = ref["vpy"] + np.gradient(sy, dt)
        # (t, sample times, sway x / y, sway velocity x / y) for the next solve
        self._sway_state = (float(inp.t), ts.copy(), off_s[:, 0].copy(), off_s[:, 1].copy(),
                            np.gradient(off_s[:, 0], ts), np.gradient(off_s[:, 1], ts))

    def _sway_velocity_now(self, t: float) -> np.ndarray:
        """World-frame sway velocity (2,) the body is expected to have at time
        ``t`` according to the previous solution (zero without crawl sway).

        The capture-point term of the foothold heuristic acts on the velocity
        error with respect to this *swayed* reference: the crawl sway moves the
        CoM at up to ~0.4 m/s between support triangles, and treating that
        intended motion as a velocity error would push every foothold in the
        direction of the sway, which moves the support centroid and with it
        the next sway further out (a positive feedback that makes the crawl
        drift sideways)."""
        sw = self._sway_state
        if sw is None or len(sw) < 6:
            return np.zeros(2)
        tr = float(t) - sw[0]
        if not 0.0 <= tr < 0.5:
            return np.zeros(2)
        return np.array([np.interp(tr, sw[1], sw[4]), np.interp(tr, sw[1], sw[5])])

    def _nominal_xy(self, inp: MPCInput) -> np.ndarray:
        if inp.nominal_feet_xy is not None:
            return np.asarray(inp.nominal_feet_xy, dtype=float).reshape(4, 2)
        return np.stack([kin.SIGN_X * rm.NOMINAL_FOOT_X, kin.SIGN_Y * rm.NOMINAL_FOOT_Y], axis=1)

    @staticmethod
    def _interp(ref, key, t):
        return np.interp(t, ref["t"], ref[key])

    # ------------------------------------------------------------------ footholds
    def _raibert_batch(self, inp: MPCInput, ref, legs, t_td, t_stance):
        """Vectorised Raibert + capture-point foothold heuristic (module docstring)
        for touchdowns of ``legs`` at relative times ``t_td`` with stance
        durations ``t_stance`` (arrays of length k).

        Returns xy (k, 2) clamped footholds, hip (k, 2) predicted hip
        positions at mid-stance, cos / sin (k,) of the heading at mid-stance
        and the fallback ground height z_fb (k,) at touchdown.

        The hip is predicted on the reference path PLUS the displacement the
        current velocity error causes until touchdown.  The MPC brings the
        body back onto the commanded path (position and velocity cost); with a
        critically damped return of time constant ``feedback_tau`` (the decay
        already assumed by the capture-point term) the displacement is
        ``v_err * t_td * exp(-t_td / tau)``: the full error at first, back on
        the path for the late touchdowns.  Without it, a body that has been
        pushed sideways at 0.7 m/s gets footholds clamped around a hip that
        stays on the commanded path - ~10 cm on the wrong side of the real
        hip, i.e. the capture step is placed inwards (MuJoCo physics check:
        every 15 N s lateral push while trotting ended in a fall)."""
        c = self.cfg
        nom = self._nominal_xy(inp)
        legs = np.asarray(legs, dtype=int)
        t_td = np.asarray(t_td, dtype=float)
        t_st = np.asarray(t_stance, dtype=float)
        tg = ref["t"]
        t_mid = t_td + 0.5 * t_st
        yaw_m = np.interp(t_mid, tg, ref["yaw"])
        cy, sy = np.cos(yaw_m), np.sin(yaw_m)
        h = max(float(inp.body_height), 0.1)
        # capture-point feedback on the velocity error, fading over the horizon
        # (error w.r.t. the path velocity plus the intended crawl sway velocity)
        v_ref_now = np.array([ref["vpx"][0], ref["vpy"][0]]) + self._sway_velocity_now(inp.t)
        v_err = np.asarray(inp.vel, dtype=float)[:2] - v_ref_now
        tau = max(c.feedback_tau, 1e-3)
        t_pos = np.maximum(t_td, 0.0)
        drift = t_pos * np.exp(-t_pos / tau)
        hx = np.interp(t_mid, tg, ref["path_x"]) + drift * v_err[0]
        hy = np.interp(t_mid, tg, ref["path_y"]) + drift * v_err[1]
        hip = np.stack([hx + cy * nom[legs, 0] - sy * nom[legs, 1],
                        hy + sy * nom[legs, 0] + cy * nom[legs, 1]], axis=1)
        decay = np.exp(-np.maximum(t_td, 0.0) / tau)
        gain = (0.5 * t_st + c.capture_gain * math.sqrt(h / c.gravity)) * decay
        off = gain[:, None] * v_err[None, :]
        # centrifugal term (h / 2g) (v_ref x w_ref) for turning
        wz = float(ref["wz"])
        k_c = 0.5 * h / c.gravity
        off[:, 0] += k_c * np.interp(t_td, tg, ref["vpy"]) * wz
        off[:, 1] -= k_c * np.interp(t_td, tg, ref["vpx"]) * wz
        # clamp the offset in the heading frame (leg workspace)
        ox = np.clip(cy * off[:, 0] + sy * off[:, 1], -c.max_foot_dx, c.max_foot_dx)
        oy = np.clip(-sy * off[:, 0] + cy * off[:, 1], -c.max_foot_dy, c.max_foot_dy)
        xy = hip + np.stack([cy * ox - sy * oy, sy * ox + cy * oy], axis=1)
        z_fb = np.interp(t_td, tg, ref["ground"])
        return xy, hip, cy, sy, z_fb

    def _planner_adjust(self, foothold_fn, leg: int, xy, hip, cy: float, sy: float, z_fb: float):
        """Refine one foothold with the terrain callback (foothold planner).
        The planner may move the foot, but never more than 5 cm beyond the
        workspace clamp around the hip.  Returns (xy, z) with z = NaN if the
        planner gave no height."""
        c = self.cfg
        z = float("nan")
        try:
            res = foothold_fn(leg, np.array([xy[0], xy[1], z_fb]))
            if res is not None:
                p, valid = res
                p = np.asarray(p, dtype=float)
                if valid and np.all(np.isfinite(p[:2])):
                    d = p[:2] - hip
                    dx = float(np.clip(cy * d[0] + sy * d[1], -c.max_foot_dx - 0.05, c.max_foot_dx + 0.05))
                    dy = float(np.clip(-sy * d[0] + cy * d[1], -c.max_foot_dy - 0.05, c.max_foot_dy + 0.05))
                    xy = hip + np.array([cy * dx - sy * dy, sy * dx + cy * dy])
                    if np.isfinite(p[2]):
                        z = float(p[2])
        except Exception:
            pass
        return np.asarray(xy, dtype=float), z

    def raibert_foothold(self, inp: MPCInput, ref, leg: int, t_td: float, t_stance: float,
                         height_fn=None, foothold_fn=None, lookup_height: bool = True):
        """Predicted foot centre (world) for one touchdown at relative time t_td.

        With ``lookup_height=False`` the z of a foothold not set by the foothold
        planner is returned as NaN (the caller batches the terrain queries)."""
        xy, hip, cy, sy, z_fb = self._raibert_batch(inp, ref, [leg], [t_td], [t_stance])
        xy, hip, cy, sy, z_fb = xy[0], hip[0], float(cy[0]), float(sy[0]), float(z_fb[0])
        z = float("nan")
        if foothold_fn is not None and t_td <= self.cfg.terrain_horizon:
            xy, z = self._planner_adjust(foothold_fn, leg, xy, hip, cy, sy, z_fb)
        if not lookup_height and not np.isfinite(z):
            return np.array([xy[0], xy[1], np.nan])
        if not np.isfinite(z):
            z = float(terrain_heights(height_fn, np.array([xy[0]]), np.array([xy[1]]))[0])
        if not np.isfinite(z):
            z = z_fb
        return np.array([xy[0], xy[1], z + rm.FOOT_RADIUS])

    def _footholds(self, inp: MPCInput, gait, ref, C, height_fn, foothold_fn):
        """Per-sample foothold arrays.

        Returns F (S,4,3) foot centre while in contact, G (S,4,3) planned
        location (current stance foothold or next touchdown), events list.
        All touchdowns of the horizon are predicted in one vectorised Raibert
        evaluation; the foothold planner refines those within
        ``terrain_horizon``; the terrain heights of the others are looked up in
        one vectorised call."""
        c = self.cfg
        S, sdt = self.S, self.sample_dt
        F = np.zeros((S, 4, 3))
        G = np.zeros((S, 4, 3))
        feet_now = np.asarray(inp.feet, dtype=float).reshape(4, 3)
        reqs = []                    # touchdown requests (leg, t_td, t_stance, planner allowed)
        per_leg = []                 # (intervals, request index per interval (-1 = current foot), next index)
        for leg in range(4):
            cl = C[:, leg]
            d = np.diff(cl.astype(np.int8))
            starts = list(np.nonzero(d == 1)[0] + 1)
            ends = list(np.nonzero(d == -1)[0] + 1)
            if cl[0]:
                starts = [0] + starts
            if cl[-1]:
                ends = ends + [S]
            intervals = list(zip(starts, ends))
            hold_idx = []
            for (sa, sb) in intervals:
                if sa == 0 and cl[0]:
                    hold_idx.append(-1)
                    continue
                t_td = sa * sdt
                pat = gait.pattern_at(t_td)
                t_st = pat.stance_duration if not pat.is_stand else max((sb - sa) * sdt, 0.2)
                hold_idx.append(len(reqs))
                reqs.append((leg, t_td, t_st, True))
            nxt = None
            if intervals:
                last_end = intervals[-1][1]
                if last_end < S:
                    # horizon ends in swing: predict the touchdown after the horizon
                    pat = gait.pattern_at(self.T)
                    t_td = min((last_end * sdt) + max(pat.swing_duration, 0.0), self.T + 0.5)
                    t_st = pat.stance_duration if not pat.is_stand else 0.5
                    nxt = len(reqs)
                    reqs.append((leg, t_td, t_st, False))
            else:
                pat = gait.pattern_at(0.0)
                nxt = len(reqs)
                reqs.append((leg, max(pat.swing_duration, 0.1), max(pat.stance_duration, 0.2), True))
            per_leg.append((intervals, hold_idx, nxt))

        P = np.zeros((len(reqs), 3))
        if reqs:
            legs = np.array([r[0] for r in reqs], dtype=int)
            ttd = np.array([r[1] for r in reqs], dtype=float)
            tst = np.array([r[2] for r in reqs], dtype=float)
            xy, hip, cy, sy, z_fb = self._raibert_batch(inp, ref, legs, ttd, tst)
            z = np.full(len(reqs), np.nan)
            if foothold_fn is not None:
                for i, (leg, t_td, _t_st, allow) in enumerate(reqs):
                    if allow and t_td <= c.terrain_horizon:
                        xy[i], z[i] = self._planner_adjust(foothold_fn, leg, xy[i], hip[i],
                                                           float(cy[i]), float(sy[i]), float(z_fb[i]))
            miss = ~np.isfinite(z)
            if np.any(miss):
                hz = terrain_heights(height_fn, xy[miss, 0], xy[miss, 1])
                z[miss] = np.where(np.isfinite(hz), hz, z_fb[miss])
            P[:, :2] = xy
            P[:, 2] = z + rm.FOOT_RADIUS

        events = []
        for leg, (intervals, hold_idx, nxt) in enumerate(per_leg):
            holds = [feet_now[leg].copy() if j < 0 else P[j] for j in hold_idx]
            nxt_p = P[nxt] if nxt is not None else None
            for (sa, sb), p, j in zip(intervals, holds, hold_idx):
                F[sa:sb, leg] = p
                if j >= 0:
                    events.append((leg, float(reqs[j][1]), float(p[0]), float(p[1]), float(p[2])))
            # walk backwards so every swing sample gets the *next* foothold
            cur = nxt_p
            idx = S
            for (sa, sb), p in zip(intervals[::-1], holds[::-1]):
                if sb < idx:
                    G[sb:idx, leg] = cur if cur is not None else p
                G[sa:sb, leg] = p
                cur = p
                idx = sa
            if idx > 0:
                G[0:idx, leg] = cur if cur is not None else feet_now[leg]
        return F, G, events

    # ------------------------------------------------------------------ solve
    def solve(self, inp: MPCInput, gait, height_fn: Optional[Callable] = None,
              foothold_fn: Optional[Callable] = None) -> MPCResult:
        """Solve the MPC for the measured state ``inp`` and the gait schedule
        ``gait`` (a GaitScheduler updated to time ``inp.t``; it is only read).

        height_fn(x, y) -> terrain height (NaN unknown), arrays allowed.
        foothold_fn(leg, nominal_xyz) -> (xyz_surface, valid)."""
        t_start = time.perf_counter()
        cpu_start = time.thread_time()
        c = self.cfg
        N = self.N

        # ---------------- contact schedule on the sample grid
        C = np.asarray(gait.contacts_at(self.ts), dtype=bool)               # (S,4)
        counts = self.n_per_step.astype(float)
        cfrac = np.add.reduceat(C.astype(float), self.step_start, axis=0) / counts[:, None]   # (N,4)

        # ---------------- reference + footholds (+ crawl sway from the footholds)
        ncont = C.sum(axis=1)
        sway_active = bool(c.sway_gain > 0 and inp.hold_xy is None and ncont.min() >= 3 and ncont.min() < 4)
        ref = self._reference(inp, gait, height_fn, sway_active)
        F, G, events = self._footholds(inp, gait, ref, C, height_fn, foothold_fn)
        if sway_active:
            self._add_sway(inp, ref, C, F)
        else:
            self._sway_state = None
        Cf = C.astype(float)[:, :, None]
        Fsum = np.add.reduceat(Cf * F, self.step_start, axis=0)                 # (N,4,3)
        csum = np.add.reduceat(Cf, self.step_start, axis=0)                     # (N,4,1)
        mid_idx = self.step_start + self.n_per_step // 2
        foot_pos = np.where(csum > 0, Fsum / np.maximum(csum, 1e-12), G[mid_idx])
        active = cfrac >= c.contact_threshold                                   # (N,4)

        # ---------------- reference states at the knots
        tk = self.knots
        tm = 0.5 * (tk[:-1] + tk[1:])
        x_ref = np.zeros((N + 1, NX))
        x_ref[:, 1] = self._interp(ref, "pitch", tk)
        x_ref[:, 2] = self._interp(ref, "yaw", tk)
        x_ref[:, 3] = self._interp(ref, "px", tk)
        x_ref[:, 4] = self._interp(ref, "py", tk)
        x_ref[:, 5] = self._interp(ref, "pz", tk)
        x_ref[:, 8] = ref["wz"]
        x_ref[:, 9] = self._interp(ref, "vx", tk)
        x_ref[:, 10] = self._interp(ref, "vy", tk)
        x_ref[:, 11] = self._interp(ref, "vz", tk)
        x_ref[:, 12] = -c.gravity

        x0 = np.zeros(NX)
        x0[0:3] = np.asarray(inp.rpy, dtype=float)
        x0[3:6] = np.asarray(inp.pos, dtype=float)
        x0[6:9] = np.asarray(inp.omega, dtype=float)
        x0[9:12] = np.asarray(inp.vel, dtype=float)
        x0[12] = -c.gravity
        # x_ref[0] stays the reference at t = 0 (the controller reads the terrain-aware
        # height / attitude reference from it); x0 is the measured initial state

        # ---------------- discrete dynamics per step, active variables
        affine = np.concatenate([np.asarray(inp.ang_accel_bias, dtype=float),
                                 np.asarray(inp.accel_bias, dtype=float) + np.array([0.0, 0.0, -c.gravity])])
        yaw_lin = self._interp(ref, "yaw", tm)
        com_lin = self._linearisation_path(inp, ref, tm)
        contact_pts = foot_pos - np.array([0.0, 0.0, rm.FOOT_RADIUS])
        var_k, var_leg = np.nonzero(active)                     # active (step, leg) pairs in order
        na = var_k.size
        n = 3 * na
        col_of = -np.ones((N, 4), dtype=int)
        col_of[var_k, var_leg] = np.arange(na) * 3

        Gam = np.zeros((N, NX, n))
        xfree = np.zeros((N, NX))
        Gk = np.zeros((NX, n))
        xk = x0.copy()
        for k in range(N):
            r = contact_pts[k] - com_lin[k]
            Ad, Bd = self.dynamics(yaw_lin[k], r, self.dts[k], affine)
            Gk = Ad @ Gk
            for leg in range(4):
                j = col_of[k, leg]
                if j >= 0:
                    Gk[:, j:j + 3] += Bd[:, 3 * leg:3 * leg + 3]
            Gam[k] = Gk
            xk = Ad @ xk
            xfree[k] = xk

        forces = np.zeros((N, NU))
        status = "no_contacts"
        iters = 0
        cost = float("nan")
        x_pred = np.vstack([x0, xfree])
        if n > 0:
            Gm = Gam.reshape(N * NX, n)
            # H = sum_k Gam_k' W_k Gam_k.  The variables are ordered by step, and
            # Gam_k has nonzero columns only for the forces of steps <= k (block
            # lower-triangular condensing), so each term touches only the leading
            # ncol_k x ncol_k block (~1/3 of the flops of the dense product).
            wq_k = self.step_w[:, None] * self.qdiag[None, :]                  # (N, NX)
            err_k = xfree - x_ref[1:]                                          # (N, NX)
            ncol = 3 * np.cumsum(active.sum(axis=1))
            H = np.zeros((n, n))
            fvec = np.zeros(n)
            for k in range(N):
                mk = int(ncol[k])
                if mk == 0:
                    continue
                Gk = Gam[k, :, :mk]
                WG = wq_k[k][:, None] * Gk
                H[:mk, :mk] += Gk.T @ WG
                fvec[:mk] += WG.T @ err_k[k]
            # force regularisation towards the gravity-compensating distribution
            # u_nom (fz = m g c_i / sum_j c_j), i.e. alpha ||u - u_nom||^2: it keeps
            # internal (squeezing) forces small without biasing the height
            r_diag = c.r_weight * np.repeat(self.step_w[var_k], 3)
            H[np.diag_indices(n)] += r_diag
            u_nom = self._nominal_forces(var_k, var_leg, cfrac)
            fvec -= r_diag * u_nom
            # constraints: friction pyramid + normal force bounds, 5 rows per active leg
            mu = self.effective_mu(inp)
            blk = self.blk if mu == float(c.mu) else self._friction_block(mu)
            Acon = np.kron(np.eye(na), blk)
            cf = cfrac[var_k, var_leg]
            lo = np.tile(np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.0]), na)
            hi = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0]), na)
            lo[4::5] = c.f_min * cf
            hi[4::5] = c.f_max * cf
            x_ws, y_ws = self._warm_start(inp.t, var_k, var_leg, cfrac, tm)
            # solve for the deviation from the warm start, U = U_ws + dU: the linear
            # term becomes the cost gradient at U_ws, so the relative termination
            # criteria are not swamped by the huge free-fall term of the condensed QP
            q_dev = fvec + H @ x_ws
            a_ws = Acon @ x_ws
            res = self.qp.solve(H, q_dev, Acon, lo - a_ws, hi - a_ws, x0=None, y0=y_ws)
            status = res.status
            iters = res.iterations
            U = x_ws + res.x
            if res.ok:
                # keep the adapted ADMM step size for the next (similar) problem
                self.qp.rho = float(np.clip(res.rho, 1e-3, 10.0))
            if not np.all(np.isfinite(U)):
                U = x_ws if x_ws is not None else np.zeros(n)
                status = "nan"
            # enforce the constraints exactly on the returned forces (ADMM is approximate)
            U3 = U.reshape(na, 3).copy()
            U3[:, 2] = np.clip(U3[:, 2], c.f_min * cf, c.f_max * cf)
            lim = mu * U3[:, 2]
            U3[:, 0] = np.clip(U3[:, 0], -lim, lim)
            U3[:, 1] = np.clip(U3[:, 1], -lim, lim)
            U = U3.reshape(-1)
            for j in range(na):
                forces[var_k[j], 3 * var_leg[j]:3 * var_leg[j] + 3] = U3[j]
            x_pred = np.vstack([x0, (Gm @ U + xfree.reshape(-1)).reshape(N, NX)])
            cost = float(0.5 * U @ H @ U + fvec @ U)
            if np.all(np.isfinite(x_pred)) and np.all(np.isfinite(res.y)):
                self._prev = {"t": float(inp.t), "knots": tk.copy(), "forces": forces.copy(),
                              "cfrac": cfrac.copy(), "y": self._dual_table(res.y, var_k, var_leg, N),
                              "x_pred": x_pred.copy()}
            else:
                # a non-finite prediction must not become the next warm start /
                # linearisation path (it would poison every following solve)
                self._prev = None
        else:
            self._prev = None

        # ---------------- torques from the predicted pose / footholds
        xm = 0.5 * (x_pred[:-1] + x_pred[1:])
        Rk = euler_to_rot_batch(xm[:, 0], xm[:, 1], xm[:, 2])                   # (N,3,3)
        rel = foot_pos - xm[:, None, 3:6]                                       # (N,4,3)
        feet_b = np.einsum("kji,klj->kli", Rk, rel)                             # R^T rel
        legs = np.broadcast_to(np.arange(4), (N, 4))
        q, reach = kin.leg_ik(feet_b, legs)
        f4 = forces.reshape(N, 4, 3)
        fb = np.einsum("kji,klj->kli", Rk, f4)
        peak = np.where(active[:, :, None], fb / np.maximum(cfrac[:, :, None], c.contact_threshold), 0.0)
        tau = kin.leg_torques(q, legs, peak).reshape(N, NU)
        win = tm <= max(c.torque_check_horizon, self.dts[0])
        eff = kin.EFFORT_LIMITS
        tau_max = np.max(np.abs(tau[win]), axis=0) if np.any(win) else np.zeros(NU)
        ratio = float(np.max(tau_max / eff))
        reachable = bool(np.all(reach[active]))

        result = MPCResult(
            t=float(inp.t), times=tk.copy(), dts=self.dts.copy(), x_pred=x_pred, x_ref=x_ref,
            forces=forces, contacts=cfrac, footholds=foot_pos, torques=tau,
            joint_angles=q.reshape(N, NU), foothold_events=events, status=status,
            solve_time=time.perf_counter() - t_start, qp_iterations=iters, cost=cost,
            torque_ratio=ratio, torque_horizon_max=tau_max, reachable=reachable,
            solve_cpu_time=time.thread_time() - cpu_start, mu=self.effective_mu(inp))
        self.last_result = result
        return result

    def _linearisation_path(self, inp: MPCInput, ref, tm) -> np.ndarray:
        """CoM positions (N, 3) used for the moment arms r_i = p_contact - p_com.

        Near the start of the horizon the moment arms are linearised about the
        *predicted* CoM path of the previous solution (shifted in time and
        translated to start at the measured position): a few centimetres
        between reference and actual CoM (sway tracking, pushes) would
        otherwise produce a torque error of m g * offset that the small roll
        inertia amplifies.  The deviation from the reference path is blended
        out with exp(-t / tau_lin) because the far part of an old prediction is
        stale when the command changes, while the predicted footholds always
        follow the current reference.  Without a previous solution the
        reference path translated to the measurement is used."""
        tau = max(float(self.cfg.linearisation_tau), 1e-3)
        p0 = np.asarray(inp.pos, dtype=float)
        ref_path = np.stack([self._interp(ref, "px", tm), self._interp(ref, "py", tm),
                             self._interp(ref, "pz", tm)], axis=1)
        ref0 = np.array([ref["px"][0], ref["py"][0], ref["pz"][0]])
        dev = np.broadcast_to(p0 - ref0, ref_path.shape).copy()
        prev = self._prev
        if prev is not None and "x_pred" in prev:
            dt = float(inp.t) - prev["t"]
            if 0.0 <= dt < 0.5:
                kt = prev["knots"]
                xp = prev["x_pred"]
                ts = np.clip(tm + dt, kt[0], kt[-1])
                pp = np.stack([np.interp(ts, kt, xp[:, 3 + j]) for j in range(3)], axis=1)
                start = np.array([np.interp(dt, kt, xp[:, 3 + j]) for j in range(3)])
                dev = pp + (p0 - start)[None, :] - ref_path
        # a sane moment arm never exceeds the leg workspace: cap the deviation
        dev = np.clip(dev, -0.3, 0.3)
        return ref_path + dev * np.exp(-tm / tau)[:, None]

    # ------------------------------------------------------------------ warm start
    def _nominal_forces(self, var_k, var_leg, cfrac) -> np.ndarray:
        """Gravity-compensating force distribution (3*na,) over the active legs."""
        c = self.cfg
        csum_step = np.zeros(self.N)
        np.add.at(csum_step, var_k, cfrac[var_k, var_leg])
        u = np.zeros((var_k.size, 3))
        u[:, 2] = c.mass * c.gravity * cfrac[var_k, var_leg] / np.maximum(csum_step[var_k], 1e-6)
        return u.reshape(-1)

    @staticmethod
    def _dual_table(y, var_k, var_leg, N):
        tab = np.zeros((N, 4, 5))
        if y is not None and y.size == 5 * var_k.size:
            tab[var_k, var_leg] = y.reshape(-1, 5)
        return tab

    def _warm_start(self, t_abs, var_k, var_leg, cfrac, tm):
        """Primal/dual initial guess: previous solution shifted in time, gravity
        compensation where no previous data exists."""
        c = self.cfg
        na = var_k.size
        cf = cfrac[var_k, var_leg]
        csum_step = np.zeros(self.N)
        np.add.at(csum_step, var_k, cf)
        x = np.zeros((na, 3))
        y = np.zeros((na, 5))
        x[:, 2] = c.mass * c.gravity * cf / np.maximum(csum_step[var_k], 1e-6)
        prev = self._prev
        if prev is not None and na:
            knots = prev["knots"]
            nk = len(knots) - 1
            ta = t_abs + tm[var_k] - prev["t"]                  # same instant in the previous horizon
            kp = np.searchsorted(knots, ta, side="right") - 1
            kpc = np.clip(kp, 0, nk - 1)
            cp = prev["cfrac"][kpc, var_leg]
            use = (kp >= 0) & (kp < nk) & (cp >= c.contact_threshold)
            if np.any(use):
                f_prev = prev["forces"].reshape(nk, 4, 3)
                x[use] = f_prev[kpc[use], var_leg[use]] * (cf[use] / cp[use])[:, None]
                y[use] = prev["y"][kpc[use], var_leg[use]]
        return x.reshape(-1), y.reshape(-1)
