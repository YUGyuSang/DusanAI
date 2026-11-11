"""
"stable" locomotion mode: CHAMP-style kinematic gait (ROS free, numpy only).

The approach of the CHAMP quadruped controller (github.com/chvmp/champ),
re-implemented here without any ROS dependency and adapted to the compliant
Gazebo joint position servos (tuned in gz-sim 8 + DART, tools/gazebo_check.py):

* **gait phase generator** - the controller's :class:`gait.GaitScheduler`
  with its own patterns (``stable_trot_*``: diagonal pairs, duty 0.6;
  ``stable_walk_*``: the 4-beat walk FL -> RR -> FR -> RL with a sideways
  CoM sway);
* **foot trajectories in the body frame** - nothing is anchored in the world:
  the stance feet move backwards under the body at the commanded velocity
  (``v + w x r``), i.e. they stay fixed in a frame that moves with the
  *command*; the body velocity emerges from the feet sweeping;
* **stance force limit** (``stance_max_error``) - with the stiff servo
  (~4.5 kN/m at the foot) the horizontal error between a stance foot target
  and the measured foot IS the horizontal ground force.  Pure open-loop
  sweeping lets it grow by (v_cmd - v_body) T_stance whenever the body lags
  or leads the command (start, stop, turns, impacts): several cm, > 150 N per
  foot, beyond friction - the stance feet slid and the robot fell in 3/3
  Gazebo teleop runs.  The stance targets are therefore kept within 1.5 cm
  (~70 N) of the measured feet - the same lesson as the MPC mode's body-target
  leash (controller._integrated_body_target), applied per foot;
* **Raibert footstep** - touch-down point = nominal foot (hip) position +
  (v + w x r) T_stance / 2 + ``raibert_gain`` (v_meas - v_cmd) (capture
  feedback, needed for pushes), re-targeted during the first 80 % of the swing
  so command changes act within one step;
* **Bezier swing / stance push** - Bernstein (Bezier) profiles: horizontal
  with control values (0 0 0 0 1 1 1 1) (zero velocity / acceleration at
  lift-off and touch-down in the ground frame = ground-speed matched),
  vertical bump to ``step_height``; during stance the foot is pressed
  ``stance_depth`` * sin(pi s) below the ground line (CHAMP's stance curve);
* **body pose** - fixed nominal stance height, roll / pitch levelled by a
  small PI loop on the IMU attitude, stance-leg height trim from the leg
  kinematics (``ki_height``) and a static gravity feed-forward
  (``gravity_ff``: tau = -J^T R^T (0, 0, m g w_i), added to the joint target
  as tau / servo_kp) that removes the load sag of the PD servos - without it
  the swing feet, positioned for the nominal height, land 1-2 cm "below" the
  sagged body; the load shares w_i ramp over ``load_ramp`` at the start / end
  of each stance (smooth hand-over in the 4-leg overlap, unloaded lift-off);
* **terrain** (optional, when the controller has an elevation map): touch-down
  heights and the stance feet follow the map, body height = map ground +
  body_height, body pitch = terrain pitch, swing apex raised over obstacles
  along the swing (pipes);
* **leg IK** - :func:`kinematics.ik_all` (analytic, knee-backwards).

Frames: ``L`` = level frame at the base origin, x along the heading, z up
(gravity); the foot positions of this module live in L.  The joint targets
use ``p_base = R(roll_cmd, pitch_cmd)^T p_L``.

Measured inputs: IMU roll / pitch (levelling), joint angles (height trim,
stance force limit) and the odometry velocity (Raibert feedback).  No world
pose is used by the gait itself (the controller's optional heading hold and
the map lookups use the odometry pose).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields as dc_fields
from math import comb
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from dusan_common import robot_model as rm
from dusan_common.geometry import euler_to_rot

from . import kinematics as kin

GRAVITY = 9.80665

# ---------------------------------------------------------------------------
# Bezier profiles
# ---------------------------------------------------------------------------
# horizontal progress 0 -> 1: zero velocity and acceleration at both ends
SWING_X_CTRL = (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
# vertical bump 0 -> 1 -> 0 (scaled so that the peak is exactly 1): zero
# vertical velocity at lift-off and touch-down (soft landing)
_Z_RAW = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)


def bezier(ctrl: Sequence[float], s) -> Tuple[np.ndarray, np.ndarray]:
    """Value and derivative d/ds of the 1-D Bezier curve with control values
    ``ctrl`` at s in [0, 1] (scalar or array)."""
    c = np.asarray(ctrl, dtype=float)
    n = len(c) - 1
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    val = np.zeros_like(s)
    der = np.zeros_like(s)
    for k in range(n + 1):
        b = comb(n, k) * s ** k * (1.0 - s) ** (n - k)
        val = val + c[k] * b
    for k in range(n):
        b = comb(n - 1, k) * s ** k * (1.0 - s) ** (n - 1 - k)
        der = der + n * (c[k + 1] - c[k]) * b
    return val, der


_Z_PEAK = float(bezier(_Z_RAW, 0.5)[0])
SWING_Z_CTRL = tuple(v / _Z_PEAK for v in _Z_RAW)


def swing_point(p0, p1, s: float, height: float, lift: float = 0.0, early: bool = False) -> np.ndarray:
    """Swing foot position at progress ``s`` from ``p0`` to ``p1`` (ground
    frame) with a ``height`` bump above the straight line.  ``lift`` (0..0.3)
    delays the horizontal motion to [lift, 1 - lift] of the swing and holds
    the apex over [0.5 - lift, 0.5 + lift] (step over an obstacle: up first,
    then forward at the apex, then down).  ``early`` (servo lag over an
    obstacle): the horizontal motion over [lift / 4, 1 - 1.5 lift] and the
    apex held until 1 - lift - the lagging real foot is past the obstacle
    before it comes down (Gazebo: with the symmetric profile a 0.3 m step over
    the C-OBS5 pipe set the foot down on the pipe)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    lam = float(np.clip(lift, 0.0, 0.3))
    if lam > 0.0 and early:
        x0, x1, dn = 0.25 * lam, 1.0 - 1.5 * lam, 1.0 - lam
    else:
        x0, x1, dn = lam, 1.0 - lam, 0.5 + lam
    u = (float(s) - x0) / max(x1 - x0, 1e-6)
    bx = float(bezier(SWING_X_CTRL, u)[0])
    s = float(np.clip(s, 0.0, 1.0))
    if lam <= 0.0:
        bz = float(bezier(SWING_Z_CTRL, s)[0])
    else:
        # obstacle step: rise over [0, 0.5 - lam], hold the apex while the foot moves
        # over the obstacle, descend over [dn, 1] (cubic Bezier 0 0 1 1 ramps)
        up = 0.5 - lam
        if s <= up:
            bz = float(bezier((0.0, 0.0, 1.0, 1.0), s / up)[0])
        elif s < dn:
            bz = 1.0
        else:
            bz = float(bezier((1.0, 1.0, 0.0, 0.0), (s - dn) / (1.0 - dn))[0])
    out = p0 + (p1 - p0) * bx
    out[2] += float(height) * bz
    return out


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class StableParams:
    """Tunables of the stable gait (controller parameters ``stable_*``)."""
    body_height: float = rm.NOMINAL_BODY_HEIGHT
    step_height: float = 0.07            # [m] trot swing apex above the ground line
    walk_step_height: float = 0.08       # [m] walk (4-beat) swing apex
    stance_depth: float = 0.005          # [m] stance push below the nominal ground (sin profile)
    raibert_gain: float = 0.1            # [s] touch-down offset per m/s velocity error (0 = CHAMP: none)
    max_feedback: float = 0.05           # [m] limit of that offset
    max_step_x: float = 0.12             # [m] |touch-down - nominal| limits (leg workspace)
    max_step_y: float = 0.08
    retarget_until: float = 0.8          # swing progress up to which the touch-down point follows the command
    land_tau: float = 0.05               # [s] stance foot height after touch-down -> nominal ground
    stance_max_error: float = 0.015      # [m] horizontal stance target - measured foot limit (force limit, 0 = off)
    swing_attitude_blend: float = 0.0    # 0: swing feet in the commanded attitude (like the stance) | 1: in the
                                         # MEASURED attitude at touch-down (lands on level ground whatever the
                                         # body tilt; Gazebo: worse push recovery with the stance force limit)
    kp_level: float = 0.3                # roll / pitch levelling: P on the IMU attitude
    ki_level: float = 1.0                # [1/s] I
    max_level: float = 0.15              # [rad] attitude correction limit
    ki_height: float = 1.0               # [1/s] stance height trim from the leg kinematics
    max_height_corr: float = 0.03        # [m]
    gravity_ff: float = 1.0              # static weight feed-forward through the PD servo (0 = off)
    load_ramp: float = 0.03              # [s] load hand-over at the start / end of each stance
    load_tau: float = 0.1                # [s] load filter while standing / at the walk <-> stand switch
    servo_kp: float = 450.0              # [N m/rad] JointPositionController p_gain
    mass: float = rm.TOTAL_MASS
    walk_sway: float = 0.4               # walk: CoM shift towards the support-triangle centroid (fraction)
    walk_sway_smooth: float = 0.5        # [s] Hann smoothing of the sway reference
    stand_center_tau: float = 0.5        # [s] STAND: body re-centred over the feet / nominal height
    # with an elevation map (controller: ground_fn / obstacle_fn): swing apex over obstacles
    clearance: float = 0.04              # [m] foot clearance above the highest map point along the swing
    max_lift: float = 0.25               # [m] apex limit above the higher swing end
    obstacle_lift: float = 0.2           # lift fraction (up first, then forward) when the apex is raised
    obstacle_early_x: bool = True        # obstacle swing: forward early, down late (servo lag, swing_point)
    max_retarget_step: float = 0.015     # [m] per control step: touch-down point changes after lift-off
                                         # (only with a foothold planner, whose choice can jump)
    retarget_jump: float = 0.05          # [m] the planner's displacement of the touch-down point changing by
                                         # more than this = a different foothold (the other side of a pipe) ...
    retarget_jump_until: float = 0.25    # ... taken whole up to this swing progress, later the committed
                                         # touch-down point is kept (slewing would set the foot on the pipe)

    # controller parameter name -> field
    _MAP = {
        "body_height": "body_height", "stable_step_height": "step_height",
        "stable_walk_step_height": "walk_step_height", "stable_stance_depth": "stance_depth",
        "stable_raibert_gain": "raibert_gain", "stable_max_feedback": "max_feedback",
        "stable_max_step_x": "max_step_x", "stable_max_step_y": "max_step_y",
        "stable_retarget_until": "retarget_until", "stable_kp_level": "kp_level",
        "stable_ki_level": "ki_level", "stable_max_level": "max_level", "stable_ki_height": "ki_height",
        "stable_max_height_corr": "max_height_corr", "stable_gravity_ff": "gravity_ff",
        "stable_load_ramp": "load_ramp", "stable_load_tau": "load_tau", "servo_kp": "servo_kp", "mpc_mass": "mass",
        "stable_walk_sway": "walk_sway", "stable_walk_sway_smooth": "walk_sway_smooth",
        "stable_stand_center_tau": "stand_center_tau",
        "stable_swing_attitude_blend": "swing_attitude_blend",
        "swing_clearance": "clearance", "swing_max_lift": "max_lift", "swing_lift_fraction": "obstacle_lift",
        "stable_max_retarget_step": "max_retarget_step", "stable_retarget_jump": "retarget_jump",
        "stable_retarget_jump_until": "retarget_jump_until",
        "stable_land_tau": "land_tau", "stable_stance_max_error": "stance_max_error",
        "stable_obstacle_early_x": "obstacle_early_x",
    }

    @classmethod
    def from_controller_params(cls, p: Dict[str, Any]) -> "StableParams":
        names = {f.name for f in dc_fields(cls)}
        kw = {}
        for key, name in cls._MAP.items():
            if key in p and name in names:
                kw[name] = float(p[key])
        return cls(**kw)


# ---------------------------------------------------------------------------
# Gait
# ---------------------------------------------------------------------------
def _smoothstep(u: float) -> float:
    u = min(max(float(u), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


class StableGait:
    """CHAMP-style kinematic gait (see module docstring).

    ``step`` is called at the control rate with the gait state of the
    scheduler and the shaped command and returns the (12,) joint targets."""

    def __init__(self, params: Optional[StableParams] = None, nominal_feet_b=None):
        self.p = params or StableParams()
        if nominal_feet_b is None:
            nominal_feet_b = kin.nominal_feet_body(self.p.body_height)
        self.set_nominal(nominal_feet_b)
        self.reset()

    # ------------------------------------------------------------------ state
    def set_nominal(self, nominal_feet_b) -> None:
        self.nominal = np.asarray(nominal_feet_b, dtype=float)[:, :2].copy()

    @property
    def ground_z(self) -> float:
        """Nominal foot-centre height in L (ground at the nominal body height)."""
        return -(float(self.p.body_height) - rm.FOOT_RADIUS)

    def reset(self) -> None:
        self.initialized = False
        self.feet = np.zeros((4, 3))             # L frame, without push / trim / sway
        self.p0 = np.zeros((4, 3))
        self.p1 = np.zeros((4, 3))
        self.swinging = np.zeros(4, dtype=bool)
        self.level_i = np.zeros(2)
        self.rp_cmd = np.zeros(2)
        self.z_corr = 0.0
        self.load = np.zeros(4)                  # load share per leg 0..1 (gravity feed-forward, height trim)
        self.sway = np.zeros(2)
        self.out_L = np.zeros((4, 3))
        self.out_b = np.zeros((4, 3))
        self.reachable = True
        self.height_meas = float("nan")
        self.R_meas = np.eye(3)
        self.height = np.zeros(4)                # swing apex above the line p0 -> p1 per leg
        self.lift = np.zeros(4)                  # swing lift fraction per leg
        self.fresh = np.zeros(4, dtype=bool)     # swing just started: first touch-down target not yet set
        self.fh_off = np.zeros((4, 2))           # foothold planner displacement of the current touch-down point

    def init_from_measured(self, q, rpy) -> None:
        """Start from the measured posture (feet on the ground)."""
        R = euler_to_rot(float(rpy[0]), float(rpy[1]), 0.0)
        self.reset()
        self.feet = kin.fk_all(np.asarray(q, dtype=float)) @ R.T
        self.load = np.ones(4)
        self.out_L = self.feet.copy()
        self.initialized = True

    # ------------------------------------------------------------------ helpers
    def _advance(self, dt: float, vx: float, vy: float, wz: float) -> None:
        """Move the frame with the commanded body motion: ground-fixed points
        (stance feet, swing end points) move backwards under the body."""
        if dt <= 0.0:
            return
        d = np.array([vx, vy]) * dt
        Rm = _rot2(-wz * dt)
        for arr in (self.feet, self.p0, self.p1):
            arr[:, :2] = (arr[:, :2] - d) @ Rm.T

    def touchdown_target(self, leg: int, cmd, t_stance: float, t_remaining: float,
                         v_err=(0.0, 0.0)) -> np.ndarray:
        """Touch-down point of ``leg`` in the CURRENT L frame for a swing that
        ends in ``t_remaining`` s (Raibert: nominal + (v + w x r) T_st / 2)."""
        p = self.p
        vx, vy, wz = (float(v) for v in cmd)
        r = self.nominal[leg]
        vf = np.array([vx - wz * r[1], vy + wz * r[0]])
        fb = np.clip(float(p.raibert_gain) * np.asarray(v_err, dtype=float), -p.max_feedback, p.max_feedback)
        off = vf * (0.5 * max(t_stance, 0.0)) + fb
        off = np.array([np.clip(off[0], -p.max_step_x, p.max_step_x), np.clip(off[1], -p.max_step_y, p.max_step_y)])
        td = r + off
        # the frame moves on until the touch-down: a ground point that is at td
        # then is here now
        vtd = np.array([vx - wz * td[1], vy + wz * td[0]])
        now = td + vtd * max(t_remaining, 0.0)
        return np.array([now[0], now[1], self.ground_z])

    # ------------------------------------------------------------------ step
    def _ground(self, ground_fn, xy) -> np.ndarray:
        """Foot-centre ground height(s) in L at L positions ``xy`` (n, 2)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        if ground_fn is None:
            return np.full(len(xy), self.ground_z)
        z = np.asarray(ground_fn(xy), dtype=float).reshape(-1)
        return np.where(np.isfinite(z), z, self.ground_z)

    def _apex(self, leg: int, base: float, obstacle_fn, s: float = 0.0) -> None:
        """Swing apex (above the line p0 -> p1) and lift fraction of ``leg``.
        The lift fraction (delayed horizontal motion) only changes at the very
        start of the swing, where it does not move the foot target."""
        p = self.p
        early = s <= 0.5 * float(p.obstacle_lift)
        self.height[leg] = base
        if early:
            self.lift[leg] = 0.0
        if obstacle_fn is None:
            return
        top = obstacle_fn(self.p0[leg], self.p1[leg])        # highest ground point along the swing (L z)
        if top is None or not math.isfinite(float(top)):
            return
        hi = max(float(self.p0[leg, 2]), float(self.p1[leg, 2]))
        need = float(top) + rm.FOOT_RADIUS + float(p.clearance) - hi
        if need > base + 0.01:
            self.height[leg] = min(need, float(p.max_lift))
            if early:
                self.lift[leg] = float(p.obstacle_lift)

    def step(self, dt: float, gs, cmd, rpy_meas, q_meas, walking: bool = True,
             v_meas_heading=None, sway=None, ground_fn=None, obstacle_fn=None,
             rp_des=None, foothold_fn=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """One control cycle.

        gs: gait.GaitState of the stable scheduler; cmd: shaped (vx, vy, wz)
        (heading frame); rpy_meas: IMU attitude; q_meas: measured joints;
        walking: False in STAND (feet are re-centred, no stepping);
        v_meas_heading: measured horizontal velocity in the heading frame
        (Raibert feedback, optional); sway: (2,) body offset of the walk.
        Terrain (optional, from the elevation map): ground_fn((n, 2) L xy) ->
        (n,) foot-centre ground height in L for the body at its reference
        height (None: flat ground at the nominal height); obstacle_fn(p0, p1)
        -> highest ground point (L z) along a swing (NaN: unknown); rp_des:
        (roll, pitch) reference of the levelling (terrain pitch);
        foothold_fn(leg, (2,) L xy) -> (2,) L xy: terrain-aware touch-down
        point (foothold planner: off pipe tops / edges), its changes after
        lift-off are rate limited (``max_retarget_step``).
        Returns ((12,) joint targets, info)."""
        p = self.p
        if not self.initialized:
            self.init_from_measured(q_meas, rpy_meas)
        cmd = np.asarray(cmd, dtype=float).reshape(3)
        if not walking:
            cmd = np.zeros(3)
        vx, vy, wz = (float(v) for v in cmd)
        self._advance(dt, vx, vy, wz)

        rpy_meas = np.asarray(rpy_meas, dtype=float)
        if not np.all(np.isfinite(rpy_meas[:2])):
            rpy_meas = np.array([self.rp_cmd[0], self.rp_cmd[1], 0.0])
        R_meas = euler_to_rot(float(rpy_meas[0]), float(rpy_meas[1]), 0.0)
        self.R_meas = R_meas
        feet_meas_b = kin.fk_all(np.asarray(q_meas, dtype=float))
        feet_meas_L = feet_meas_b @ R_meas.T          # measured feet in the true level frame

        # ---------------- roll / pitch levelling (PI on the IMU attitude)
        rp_ref = np.zeros(2) if rp_des is None else np.nan_to_num(np.asarray(rp_des, dtype=float)[:2])
        e_rp = rp_ref - rpy_meas[:2]
        if dt > 0.0:
            self.level_i = np.clip(self.level_i + float(p.ki_level) * e_rp * dt, -p.max_level, p.max_level)
        self.rp_cmd = rp_ref + np.clip(float(p.kp_level) * e_rp + self.level_i, -p.max_level, p.max_level)
        R_cmd = euler_to_rot(float(self.rp_cmd[0]), float(self.rp_cmd[1]), 0.0)

        # ---------------- walk: CoM sway (body offset = all feet move the other way)
        if sway is not None and np.all(np.isfinite(sway)):
            self.sway = np.asarray(sway, dtype=float)[:2].copy()
        else:
            self.sway = np.zeros(2)
        sw3 = np.array([self.sway[0], self.sway[1], 0.0])

        # ---------------- lift-off / touch-down
        contacts = np.asarray(gs.contacts, dtype=bool)
        for leg in range(4):
            if contacts[leg] and self.swinging[leg]:
                # touch-down: the stance target starts at the swing end point, expressed
                # in the commanded-attitude frame (continuous joint targets)
                self.feet[leg] = (self.p1[leg] - sw3) @ R_meas @ R_cmd.T + sw3
                self.swinging[leg] = False
            elif not contacts[leg] and not self.swinging[leg]:
                self.p0[leg] = self.feet[leg].copy()     # stance push / trim are zero at lift-off
                self.p1[leg] = self.feet[leg].copy()
                self.swinging[leg] = True
                self.height[leg] = 0.0
                self.lift[leg] = 0.0
                self.fresh[leg] = True
        stance = ~self.swinging

        # ---------------- swing targets (Raibert, re-targeted while s <= retarget_until)
        v_err = np.zeros(2)
        if v_meas_heading is not None and p.raibert_gain != 0.0:
            vm = np.asarray(v_meas_heading, dtype=float)[:2]
            if np.all(np.isfinite(vm)):
                v_err = vm - cmd[:2]
        crawl = gs.name == "crawl"
        height = float(p.walk_step_height if crawl else p.step_height)
        for leg in range(4):
            if not self.swinging[leg]:
                continue
            s = float(gs.swing_progress[leg])
            if s <= float(p.retarget_until):
                t_rem = (1.0 - s) * float(gs.swing_duration)
                tgt = self.touchdown_target(leg, cmd, float(gs.stance_duration), t_rem, v_err)
                if foothold_fn is not None:
                    raw = tgt[:2].copy()
                    adj = foothold_fn(leg, tgt[:2])
                    if adj is not None and np.all(np.isfinite(adj)):
                        tgt[:2] = np.asarray(adj, dtype=float)[:2]
                    off = tgt[:2] - raw
                    if not self.fresh[leg]:
                        jump = float(np.hypot(*(off - self.fh_off[leg])))
                        if float(p.retarget_jump) > 0.0 and jump > float(p.retarget_jump):
                            if s > float(p.retarget_jump_until):
                                # the planner switched to another foothold late in the swing (the
                                # nominal point drifted over a pipe: near side <-> far side): keep the
                                # committed touch-down point - slewing over to the new one at the rate
                                # limit set the foot down ON the pipe (Gazebo C-OBS5)
                                tgt[:2] = self.p1[leg, :2]
                                off = self.fh_off[leg].copy()
                            # else: early in the swing the new foothold is taken whole
                        else:
                            d = tgt[:2] - self.p1[leg, :2]
                            n = float(np.hypot(d[0], d[1]))
                            if n > float(p.max_retarget_step):
                                tgt[:2] = self.p1[leg, :2] + d * (float(p.max_retarget_step) / n)
                    self.fh_off[leg] = off
                self.fresh[leg] = False
                tgt[2] = float(self._ground(ground_fn, tgt[:2])[0])
                self.p1[leg] = tgt
                if s <= 0.5 or self.height[leg] <= 0.0:
                    self._apex(leg, height, obstacle_fn, s)
            self.feet[leg] = swing_point(self.p0[leg], self.p1[leg], s, self.height[leg], self.lift[leg],
                                         early=bool(p.obstacle_early_x))

        # ---------------- stance feet settle on the ground (nominal height / map) after touch-down
        z_tgt = self._ground(ground_fn, self.feet[:, :2])
        if walking and dt > 0.0 and float(p.land_tau) > 0.0:
            a = min(1.0, dt / float(p.land_tau))
            for leg in range(4):
                if not self.swinging[leg]:
                    self.feet[leg, 2] += a * (z_tgt[leg] - self.feet[leg, 2])

        # ---------------- stance force limit: a stance foot target never leads the
        # measured foot by more than ``stance_max_error`` horizontally.  With the
        # stiff servo (~4.5 kN/m at the foot) the horizontal target error IS the
        # horizontal ground force; open-loop sweeping alone lets it grow by
        # (v_cmd - v_body) T_stance (several cm -> > 150 N per foot, beyond friction)
        # whenever the body lags or leads the command (start, stop, turns, impacts)
        emax = float(p.stance_max_error)
        if emax > 0.0:
            meas_cmdL = feet_meas_b @ R_cmd.T + sw3
            for leg in range(4):
                if self.swinging[leg]:
                    continue
                d = self.feet[leg, :2] - meas_cmdL[leg, :2]
                n = float(np.hypot(d[0], d[1]))
                if n > emax:
                    self.feet[leg, :2] = meas_cmdL[leg, :2] + d * (emax / n)

        # ---------------- STAND: re-centre the body over the feet, nominal height
        if not walking and gs.name == "stand":
            a = min(1.0, dt / max(float(p.stand_center_tau), 1e-3)) if dt > 0.0 else 0.0
            c = self.feet[:, :2].mean(axis=0) - self.nominal.mean(axis=0)
            self.feet[:, :2] -= a * c
            if ground_fn is None:
                # flat: nominal mean height, the per-foot differences (uneven ground) are kept
                dz = float(self.feet[:, 2].mean()) - self.ground_z
                self.feet[:, 2] -= a * dz
            else:
                self.feet[:, 2] += a * (z_tgt - self.feet[:, 2])

        # ---------------- per-leg load share: ramps over the first / last ``load_ramp`` s
        # of the stance (the 4-leg overlap of the trot), so the weight hand-over between
        # the pairs is smooth and a leg is unloaded when it lifts off
        if walking and gs.name != "stand":
            ramp = float(np.clip(float(p.load_ramp) / max(float(gs.stance_duration), 1e-3), 1e-3, 0.3))
            for leg in range(4):
                if stance[leg]:
                    sp = float(gs.stance_progress[leg])
                    self.load[leg] = max(_smoothstep(sp / ramp) * _smoothstep((1.0 - sp) / ramp), 1e-3)
                else:
                    self.load[leg] = 0.0
        else:
            a = min(1.0, dt / max(float(p.load_tau), 1e-3)) if dt > 0.0 else 0.0
            self.load += a * (stance.astype(float) - self.load)

        # ---------------- height trim from the leg kinematics (stance legs)
        if stance.any():
            self.height_meas = -float(np.mean(feet_meas_L[stance, 2])) + rm.FOOT_RADIUS
            # measured stance feet above their ground targets = body too low
            e = float(np.mean(feet_meas_L[stance, 2])) - float(np.mean(z_tgt[stance]))
            if math.isfinite(e) and dt > 0.0:
                self.z_corr = float(np.clip(self.z_corr + float(p.ki_height) * e * dt,
                                            -p.max_height_corr, p.max_height_corr))

        # ---------------- body-frame foot targets
        out_L = self.feet - sw3
        feet_b = np.zeros((4, 3))
        for leg in range(4):
            if stance[leg]:
                sp = float(gs.stance_progress[leg]) if walking and gs.name != "stand" else 0.0
                if sp > 0.0:
                    out_L[leg, 2] -= float(p.stance_depth) * math.sin(math.pi * sp)     # stance push
                out_L[leg, 2] -= self.z_corr * self.load[leg]
                feet_b[leg] = out_L[leg] @ R_cmd                  # = R_cmd^T p
            else:
                # swing: commanded attitude (default); swing_attitude_blend > 0 moves the
                # touch-down towards the MEASURED attitude (lands on level ground whatever
                # the body tilt - worse push recovery in Gazebo, hence off)
                s = float(gs.swing_progress[leg])
                w = float(p.swing_attitude_blend) * _smoothstep((s - 0.1) / 0.3)
                rp = self.rp_cmd + w * (rpy_meas[:2] - self.rp_cmd)
                R_sw = euler_to_rot(float(rp[0]), float(rp[1]), 0.0)
                feet_b[leg] = out_L[leg] @ R_sw
        q, ok = kin.ik_all(feet_b, reach_margin=0.002)
        self.reachable = bool(np.all(ok))
        self.out_L = out_L
        self.out_b = feet_b

        # ---------------- static weight feed-forward through the PD servo
        if float(p.gravity_ff) > 0.0 and float(p.servo_kp) > 0.0 and self.load.sum() > 1e-6:
            w = self.load / self.load.sum()
            f_w = np.zeros((4, 3))
            f_w[:, 2] = float(p.mass) * GRAVITY * w
            f_b = f_w @ R_meas                    # world -> base (roll / pitch only)
            tau = kin.leg_torques(q.reshape(4, 3), np.arange(4), f_b).reshape(12)
            tau = np.clip(tau, -kin.EFFORT_LIMITS, kin.EFFORT_LIMITS)
            q = q + float(p.gravity_ff) * tau / float(p.servo_kp)

        info = {"stance": stance.copy(), "feet_b": feet_b.copy(), "feet_L": out_L.copy(),
                "rp_cmd": self.rp_cmd.copy(), "z_corr": float(self.z_corr), "sway": self.sway.copy(),
                "height_meas": float(self.height_meas), "swing_height": self.height.copy(),
                "reachable": self.reachable}
        return q, info

    def lowered_targets(self) -> np.ndarray:
        """Joint targets with every swing foot put down at its current xy on
        the ground estimate (E-STOP while walking)."""
        out = self.out_b.copy()
        for leg in range(4):
            if self.swinging[leg]:
                pt = self.R_meas @ out[leg]
                pt[2] = self.ground_z
                out[leg] = self.R_meas.T @ pt
        q, _ok = kin.ik_all(out, reach_margin=0.002)
        return q
