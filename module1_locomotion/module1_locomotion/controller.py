"""
ROS-free locomotion controller: mode logic, gait, 5 s MPC, swing / stance leg
control, terrain adaptation, fall detection and recovery.

    ctrl = LocomotionController(params)
    ctrl.start(t, state)                     # -> STANDING_UP from the measured joints
    q_targets, debug = ctrl.step(t, state, cmd_vel, terrain_info, emap)   # 100 Hz

Modes: STAND, WALK, SIT, STANDING_UP, FALLEN, RECOVERING, ESTOP.

Locomotion modes (``locomotion_mode``, runtime command ``mode:stable`` |
``mode:mpc`` | ``mode:toggle``, applied while not walking):

* ``stable`` (default): CHAMP-style kinematic gait of :mod:`stable_gait`
  (body-frame foot trajectories, Raibert footholds, Bezier swing, IMU
  levelling, gravity feed-forward, stance force limit); STAND / WALK go
  through :meth:`LocomotionController._step_stable` and the MPC is not run.
  Chosen as the default because it was the more robust mode in the Gazebo
  check (gz-sim 8 + DART, tools/gazebo_check.py, module1 README section 4.6);
* ``mpc``: the 5 s MPC described below.

Stand-up, sit, fall detection / recovery, E-STOP, the slip speed limits and
the gait selection are shared by both modes.

Torque realisation is the Gazebo joint *position* servo (DESIGN_SPEC §5): the
MPC (re-solved at ``mpc_rate`` in a worker thread or inline) predicts the body
trajectory, GRFs, footholds and joint torques over 5 s; the controller turns
the prediction into joint position targets:

* stance feet stay fixed in the world; the body pose target at t + lead comes
  from the MPC (default ``integrated``: horizontal position / heading
  integrated with the MPC-predicted velocity and yaw rate, height and
  roll / pitch from the MPC's terrain reference; ``prediction``: the predicted
  pose itself) plus an IMU roll/pitch PD correction and a small height
  integrator; stance foot targets in the base frame
  p_b = R_cmd^T (p_foot_w - p_cmd)  go through the analytic IK;
* swing feet follow the swing trajectory (gait.SwingTrajectory) from the
  lift-off point to the MPC / foothold-planner foothold, apex raised over
  obstacles along the swing (max_height_along on the elevation map);
* body height = terrain height under the feet (elevation map) + nominal height,
  body pitch follows the terrain slope; without a map the stance feet give the
  ground estimate (works on the corridor lower level, floor z = -0.88);
* gait auto-selection from /terrain/info, velocity limits per gait,
  acceleration limits, and saturation-aware speed scaling when the predicted
  torque exceeds 85 % of the effort limits;
* slippery ground (see :meth:`LocomotionController._update_slip`): online slip
  detection from the stance-foot world velocity (slip_detector.py) and the
  friction prior of /terrain/info (``slippery`` / ``friction_mu``) switch on
  the *slip mode* - lower speed and acceleration limits, the slip gait in
  auto mode, the MPC torque feed-forward and a lower MPC friction
  coefficient (slip estimate / prior, recovering slowly).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from dusan_common import robot_model as rm
from dusan_common.geometry import euler_to_rot, wrap_angle

from . import kinematics as kin
from .fall_recovery import FallParams, FallRecovery
from .gait import GaitScheduler, SwingTrajectory, make_gaits, support_margin, swing_apex
from .mpc import ConvexMPC, MPCConfig, MPCInput, MPCResult, terrain_heights
from .slip_detector import SlipDetector, SlipParams
from .stable_gait import StableGait, StableParams

MODES = ("STAND", "WALK", "SIT", "STANDING_UP", "FALLEN", "RECOVERING", "ESTOP")
LOCOMOTION_MODES = ("stable", "mpc")

# ---------------------------------------------------------------------------
# Parameters (every one is a ROS parameter of locomotion_controller)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Any] = {
    # locomotion mode (runtime: /locomotion/command "mode:stable" | "mode:mpc" | "mode:toggle")
    "locomotion_mode": "stable",         # stable: CHAMP-style kinematic gait (stable_gait.py) | mpc: 5 s MPC
    # loop
    "control_rate": 100.0,
    "mpc_rate": 20.0,
    "use_mpc_thread": True,
    "mpc_max_age": 0.30,
    # posture
    "body_height": rm.NOMINAL_BODY_HEIGHT,
    "sit_height": rm.SIT_BODY_HEIGHT,
    "stand_up_duration": 2.5,
    "sit_duration": 2.0,
    "stance_width_offset": 0.0,
    "body_x_offset": 0.0,
    # gait
    "gait_mode": "trot",                 # trot (default: verified in Gazebo plant routes) | auto | crawl
    "default_gait": "trot",
    "trot_period": 0.40,
    "trot_duty": 0.50,
    "crawl_period": 1.20,                # stride 0.24 m at 0.2 m/s (1.6 s: 0.32 m = the foothold workspace limit)
    "crawl_duty": 0.80,                  # 4-leg phases between the beats (0.75: none -> tipped at the leg switches)
    "auto_walk": True,
    "idle_to_stand_time": 1.0,
    "crawl_to_trot_delay": 1.0,
    "terrain_info_timeout": 2.0,
    "blocked_speed_scale": 0.7,
    # command limits
    "trot_max_vx": 0.6, "trot_max_vy": 0.3, "trot_max_wz": 0.9,
    "crawl_max_vx": 0.25, "crawl_max_vy": 0.12, "crawl_max_wz": 0.5,
    "max_accel_x": 0.8, "max_accel_y": 0.5, "max_accel_wz": 1.5,
    "cmd_deadband": 0.02,
    # swing
    "step_height": 0.08,
    "crawl_step_height": 0.10,
    "swing_clearance": 0.04,
    "swing_max_lift": 0.25,
    "swing_lift_fraction": 0.2,
    "swing_retarget_until": 0.8,
    "crawl_retarget_until": 0.0,         # crawl: keep the lift-off foothold (see _retarget)
    "max_retarget_step": 0.06,
    "retarget_check_radius": 0.05,       # [m] a rate-limited re-target next to higher terrain ... (_retarget)
    "retarget_jump_until": 0.3,          # ... jumps whole up to this swing progress, later the target is kept
    "touchdown_measured_xy": True,
    # stance / body
    "body_lead_time": 0.02,
    "body_target_mode": "integrated",    # integrated | prediction (see _integrated_body_target)
    # leash of the integrated body target = horizontal force demand on the stance feet
    # (see _integrated_body_target; Gazebo check: 0.10 m / 0.03 m / 0.20 rad slid the feet in turns)
    "max_body_lead": 0.06,
    "max_body_lead_lateral": 0.015,      # sideways lead limit (heading frame), see _integrated_body_target
    "max_yaw_lead": 0.05,
    "mpc_lateral_ref_blend": 0.5,        # sideways velocity of the integrated target: prediction -> reference (0..1)
    "crawl_max_body_lead": 0.10,         # crawl: the planned CoM sway needs the lead
    "crawl_max_body_lead_lateral": 0.03,
    "crawl_max_yaw_lead": 0.20,
    "kp_attitude": 0.0,                  # IMU P on the IK attitude (Gazebo: 0.5 tipped the robot at pipes, see
                                         # _step_locomotion)
    "kd_attitude": 0.0,                  # D on the gyro limit-cycles with a stiff joint servo (see _step_locomotion)
    # stance force limit / heading hold / stance height of the MPC mode (see _mpc_stance_feet, _mpc_heading_hold)
    "mpc_stance_max_error": 0.02,        # [m] stance anchor - measured foot (horizontal), 0 = off
    "mpc_stance_rel_error": 0.02,        # [m] stance leg deflection beyond its static sag (horizontal) = force
                                         # limit incl. the body target lead, 0 = off (_stance_deflection_limit)
    "mpc_heading_hold_gain": 1.0,        # [1/s] heading hold while no turn is commanded (0 = off)
    "mpc_heading_hold_max": 0.2,         # [rad/s]
    "mpc_land_tau": 0.05,                # [s] stance foot height lowered to the map height under the foot (0 = off)
    "mpc_contact_lift": 0.02,            # [m] stance foot this far above its target (on a pipe) ... (0 = off)
    "mpc_contact_map_tol": 0.04,         # [m] ... and sits within this of the mapped terrain top next to it (0 = off)
    "swing_guard_clearance": 0.02,       # [m] descending swing foot kept this far above an obstacle under it (0 = off)
    "swing_guard_radius": 0.06,          # [m] "under it": map window around the measured foot
    "mpc_turn_pivot_x": -0.15,           # [m] body x of the in-place turning centre (MPC mode, 0 = base origin),
                                         # see _turn_pivot_vy
    "mpc_turn_pivot_fade": 0.10,         # [m/s] ... faded out with the commanded translation speed
    "max_attitude_correction": 0.15,
    "ki_height": 0.5,
    "max_height_integral": 0.03,
    "max_body_pitch": 0.30,
    "terrain_pitch_gain": 1.0,
    "ground_filter_tau": 0.2,
    "stand_filter_tau": 0.3,
    "max_joint_speed": 12.0,
    # joint servo (simulation/urdf: gz JointPositionController, PID force mode)
    "torque_feedforward": 0.0,           # fraction of the MPC stance torques fed forward (0 = off, see
                                         # _torque_feedforward: better tracking, worse push recovery)
    "servo_kp": 450.0,                   # [N m/rad] the servos' p_gain (spot.urdf.xacro servo_p_gain)
    # terrain
    "use_foothold_planner": True,
    "foothold_search_radius": 0.10,
    # torque saturation
    "torque_scale_threshold": 0.85,
    "torque_recover_threshold": 0.70,
    "speed_scale_min": 0.3,
    "speed_scale_down": 0.9,
    "speed_scale_up": 0.02,
    # fall recovery
    "auto_recover": True,
    "fall_angle_deg": 55.0,
    "fall_height": 0.12,
    "fall_detect_time": 0.3,
    "fall_lean_deg": 30.0,               # |roll| or |pitch| beyond for fall_lean_time -> fall (leaning on a wall)
    "fall_lean_time": 2.0,               # [s] 0 = off
    "max_recovery_attempts": 3,
    "assisted_reset": False,             # True: gz set_pose (teleport) after the attempts; False: physical only
    "reset_height_offset": 0.5,
    "reset_settle_time": 1.0,
    "reset_confirm_timeout": 8.0,        # longest wait for the gz set_pose answer (2 x gz_timeout + margin)
    "recovery_retry_time": 5.0,          # recovery FAILED: new series of attempts after this long (0 = stay FAILED)
    # push test (gz)
    "push_force_x": 0.0,
    "push_force_y": 150.0,
    "push_duration": 0.2,
    # MPC (see mpc.MPCConfig)
    "mpc_mu": 0.5,
    "mpc_f_min": 0.0,
    "mpc_f_max": 250.0,
    "mpc_q_weights": [30.0, 30.0, 10.0, 4.0, 4.0, 60.0, 0.5, 0.5, 0.5, 2.0, 2.0, 1.0],
    "mpc_r_weight": 2e-4,
    "mpc_time_weighting": True,
    "mpc_tail_decay": 0.0,
    "mpc_capture_gain": 0.6,
    "mpc_feedback_tau": 0.5,
    "mpc_max_foot_dx": 0.16,
    "mpc_max_foot_dy": 0.14,
    "mpc_terrain_horizon": 1.2,
    "mpc_sway_gain": 0.5,
    "mpc_sway_smooth": 0.5,
    "mpc_sway_ramp": 0.6,
    "mpc_anchor_window": 0.0,
    "mpc_terrain_robust": True,          # terrain reference: robust footprint fit (mpc.ConvexMPC._robust_terrain)
    "mpc_qp_max_iter": 250,
    "mpc_qp_eps": 1e-3,
    "mpc_qp_time_limit": 0.04,
    "mpc_mass": rm.TOTAL_MASS,
    # slippery ground: online slip detection (slip_detector.SlipParams) ...
    "slip_detection": True,
    "slip_speed_threshold": 0.08,        # [m/s] stance-foot world speed counted as slipping
    "slip_window": 0.06,                 # [s] foot speed window
    "slip_settle_time": 0.03,            # [s] ignored after each touch-down
    "slip_min_legs": 2,
    "slip_confirm_time": 0.10,           # [s] integrated slip evidence -> slip flag
    "slip_decay": 0.5,
    "slip_release_time": 2.0,            # [s] without a slip vote ...
    "slip_release_distance": 3.0,        # [m] ... and this much travel since -> flag released
    "slip_mu_min": 0.15,                 # lower bound of the MPC friction coefficient
    "slip_mu_margin": 1.0,               # MPC mu = margin * (observed force ratio | prior mu)
    "slip_mu_drop": 0.6,                 # mu factor per detection without force estimates
    "slip_recover_tau": 8.0,             # [s] recovery of the mu estimate after the release
    "slip_min_normal_force": 20.0,       # [N]
    "slip_max_force_ratio": 0.4,         # sliding legs with a higher |f_t| / f_n do not vote (push, not friction)
    "slip_max_tilt_deg": 15.0,           # no slip votes while |roll| or |pitch| is larger (the robot is falling)
    # ... and the friction prior of /terrain/info -> slip mode
    "use_terrain_friction": True,
    "slip_gait": "keep",                 # gait in auto mode while slippery: keep (terrain choice) | trot | crawl
    "slip_max_vx": 0.2, "slip_max_vy": 0.1, "slip_max_wz": 0.4,
    "slip_accel_scale": 0.5,             # acceleration limits x this factor in slip mode
    "slip_torque_feedforward": 1.0,      # torque feed-forward gain in slip mode (see _torque_feedforward)
    "slip_ff_ramp_time": 0.5,            # [s] ramp of the feed-forward gain
    "slip_max_body_lead": 0.015,         # [m] max_body_lead in slip mode (propulsive force demand, see _update_slip)
    "slip_resync_feet": True,            # at a slip detection: stance foot targets <- measured feet
    "slip_stance_width_offset": 0.0,     # [m] extra lateral stance offset in slip mode
    # "stable" mode (stable_gait.StableParams; shares stand-up / sit / fall recovery / slip limits)
    "stable_trot_period": 0.40,
    "stable_trot_duty": 0.60,            # 4-leg support for 0.04 s at each pair change
    "stable_walk_period": 1.20,          # 4-beat walk (gait name "crawl", or "walk")
    "stable_walk_duty": 0.80,
    "stable_trot_max_vx": 0.5, "stable_trot_max_vy": 0.25, "stable_trot_max_wz": 0.8,
    "stable_walk_max_vx": 0.2, "stable_walk_max_vy": 0.1, "stable_walk_max_wz": 0.4,
    "stable_step_height": 0.07,
    "stable_walk_step_height": 0.08,
    "stable_stance_depth": 0.005,
    "stable_raibert_gain": 0.1,
    "stable_max_feedback": 0.05,
    "stable_max_step_x": 0.12,
    "stable_max_step_y": 0.08,
    "stable_retarget_until": 0.8,
    "stable_max_retarget_step": 0.015,
    "stable_land_tau": 0.05,
    "stable_stance_max_error": 0.015,
    "stable_swing_attitude_blend": 0.0,
    "stable_kp_level": 0.3,
    "stable_ki_level": 1.0,
    "stable_max_level": 0.15,
    "stable_ki_height": 1.0,
    "stable_max_height_corr": 0.03,
    "stable_gravity_ff": 1.0,
    "stable_load_ramp": 0.03,
    "stable_load_tau": 0.1,
    "stable_walk_sway": 0.4,
    "stable_walk_sway_smooth": 0.5,
    "stable_walk_sway_x": 0.0,           # fore-aft part of the sway (the sideways part gives most of the margin)
    "stable_stand_center_tau": 0.5,
    "stable_heading_hold_gain": 1.0,     # [1/s] heading hold while no turn is commanded (0 = off)
    "stable_heading_hold_max": 0.2,      # [rad/s] limit of the heading correction
}


def mpc_config_from_params(p: Dict[str, Any]) -> MPCConfig:
    """Map the flat controller parameters onto MPCConfig."""
    cfg = MPCConfig()
    cfg.mass = float(p["mpc_mass"])
    cfg.mu = float(p["mpc_mu"])
    cfg.f_min = float(p["mpc_f_min"])
    cfg.f_max = float(p["mpc_f_max"])
    q = list(p["mpc_q_weights"])
    if len(q) != 12:
        raise ValueError("mpc_q_weights needs 12 entries [roll pitch yaw x y z wx wy wz vx vy vz]")
    cfg.q_weights = tuple(float(v) for v in q)
    cfg.r_weight = float(p["mpc_r_weight"])
    cfg.time_weighting = bool(p["mpc_time_weighting"])
    cfg.tail_decay = float(p["mpc_tail_decay"])
    cfg.capture_gain = float(p["mpc_capture_gain"])
    cfg.feedback_tau = float(p["mpc_feedback_tau"])
    cfg.max_foot_dx = float(p["mpc_max_foot_dx"])
    cfg.max_foot_dy = float(p["mpc_max_foot_dy"])
    cfg.terrain_horizon = float(p["mpc_terrain_horizon"])
    cfg.sway_gain = float(p["mpc_sway_gain"])
    cfg.sway_smooth = float(p["mpc_sway_smooth"])
    cfg.sway_ramp = float(p["mpc_sway_ramp"])
    cfg.anchor_window = float(p["mpc_anchor_window"])
    cfg.terrain_robust = bool(p["mpc_terrain_robust"])
    cfg.max_pitch = float(p["max_body_pitch"])
    cfg.terrain_pitch_gain = float(p["terrain_pitch_gain"])
    cfg.qp_max_iter = int(p["mpc_qp_max_iter"])
    cfg.qp_eps_abs = float(p["mpc_qp_eps"])
    cfg.qp_eps_rel = float(p["mpc_qp_eps"])
    cfg.qp_time_limit = float(p["mpc_qp_time_limit"])
    return cfg


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class RobotState:
    """Measured robot state (world frame = odom = map)."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q: np.ndarray = field(default_factory=lambda: np.zeros(12))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(12))

    @property
    def R(self) -> np.ndarray:
        return euler_to_rot(float(self.rpy[0]), float(self.rpy[1]), float(self.rpy[2]))

    @property
    def omega_world(self) -> np.ndarray:
        return self.R @ np.asarray(self.omega_body, dtype=float)


class OdomFilter:
    """Conditions the Gazebo odometry for the 100 Hz controller (ROS free).

    gz-sim's OdometryPublisher (simulation/urdf: 50 Hz) sends the ground-truth
    pose every 20 ms and a body-frame twist that is the rolling mean of
    ``velocity_rolling_window_size`` (default 10) finite differences, i.e. a
    0.2 s average that lags the true velocity by ~0.1 s.  Both errors bias the
    walk (MuJoCo physics check, trot 0.4 m/s command, compliant servo: 0.51 m/s
    with the raw stream; 0.42 m/s with a fresh pose and velocity):

    * the controller anchors stance feet (touch-down latch) and the body target
      on a pose that is up to 20 ms old - at 0.5 m/s a 1 cm offset per step;
    * the MPC starts every plan from a velocity 0.1 s in the past.

    This filter therefore (``velocity_source = "differentiate"``) differentiates
    consecutive odometry poses (world frame) and smooths them with a
    first-order filter of time constant ``velocity_tau`` (one 20 ms sample
    already averages over a trot half-step; 0.08 s keeps the footholds calm),
    and extrapolates the pose to the query time with that velocity
    (``extrapolate``, at most ``max_extrapolation`` s).  ``"twist"`` keeps the
    plugin twist (rotated into the world frame) - the former behaviour."""

    def __init__(self, velocity_source: str = "differentiate", velocity_tau: float = 0.08,
                 extrapolate: bool = True, max_extrapolation: float = 0.05):
        if velocity_source not in ("differentiate", "twist"):
            raise ValueError(f"odom velocity source must be differentiate|twist, got {velocity_source!r}")
        self.source = velocity_source
        self.tau = max(float(velocity_tau), 1e-4)
        self.extrapolate = bool(extrapolate)
        self.max_extrapolation = max(0.0, float(max_extrapolation))
        self.reset()

    def reset(self) -> None:
        self.stamp: Optional[float] = None
        self.pos = np.zeros(3)
        self.rpy = np.zeros(3)
        self.v_twist = np.zeros(3)
        self.v_diff: Optional[np.ndarray] = None

    def update(self, stamp: float, pos, rpy, v_base) -> None:
        """One odometry sample: world pose (pos, rpy) and the body-frame twist."""
        pos = np.asarray(pos, dtype=float).reshape(3)
        rpy = np.asarray(rpy, dtype=float).reshape(3)
        R = euler_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        v_twist = R @ np.asarray(v_base, dtype=float).reshape(3)
        stamp = float(stamp)
        if self.stamp is not None and 1e-4 < stamp - self.stamp <= 0.2:
            v_fd = (pos - self.pos) / (stamp - self.stamp)
            if self.v_diff is None:
                self.v_diff = v_fd
            else:
                a = min(1.0, (stamp - self.stamp) / self.tau)
                self.v_diff = self.v_diff + a * (v_fd - self.v_diff)
        elif self.stamp is None or abs(stamp - self.stamp) > 1e-4:
            self.v_diff = None                     # first sample, gap or clock jump: restart from the twist
        self.stamp, self.pos, self.rpy, self.v_twist = stamp, pos, rpy, v_twist

    @property
    def velocity(self) -> np.ndarray:
        if self.source == "differentiate" and self.v_diff is not None:
            return self.v_diff.copy()
        return self.v_twist.copy()

    def state(self, now: float):
        """(pos, rpy, vel_world) at time ``now`` (None before the first sample)."""
        if self.stamp is None:
            return None
        v = self.velocity
        pos = self.pos.copy()
        if self.extrapolate:
            age = float(np.clip(float(now) - self.stamp, 0.0, self.max_extrapolation))
            pos = pos + v * age
        return pos, self.rpy.copy(), v


def _terrain_tools():
    """Lazy import of the terrain agent's foothold planner (optional)."""
    planner_cls = None
    mha = None
    try:
        from .foothold_planner import FootholdPlanner as planner_cls  # noqa: N813
    except Exception:
        planner_cls = None
    try:
        from .foothold_planner import max_height_along as mha
    except Exception:
        try:
            from .elevation_map import max_height_along as mha
        except Exception:
            mha = None
    return planner_cls, mha


# PointField datatypes (sensor_msgs/PointField)
_PF_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def organised_cloud_to_xyz(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 (organised, fields x/y/z) -> (rows, cols, 3) float array.

    Pure numpy (no sensor_msgs_py): uses height/width/point_step/row_step,
    the field offsets/datatypes and the endianness flag.  Works on any object
    with those attributes (unit-testable without ROS)."""
    h, w = int(msg.height), int(msg.width)
    if h == 0 or w == 0:
        return np.zeros((0, 0, 3))
    fields = {f.name: (int(f.offset), int(f.datatype)) for f in msg.fields}
    for k in ("x", "y", "z"):
        if k not in fields:
            raise ValueError(f"PointCloud2 has no '{k}' field")
    step, row = int(msg.point_step), int(msg.row_step)
    endian = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if row < w * step:
        row = w * step
    if buf.size < h * row:
        raise ValueError("PointCloud2 data shorter than height * row_step")
    arr = buf[:h * row].reshape(h, row)[:, :w * step].reshape(h, w, step)
    out = np.empty((h, w, 3), dtype=float)
    for j, k in enumerate(("x", "y", "z")):
        off, dt = fields[k]
        code = _PF_DTYPES.get(dt, "f4")
        size = int(code[1])
        raw = np.ascontiguousarray(arr[:, :, off:off + size])
        out[:, :, j] = raw.view(endian + code).reshape(h, w).astype(float)
    return out


def _max_height_along_fallback(emap, p0, p1, n: int = 12) -> float:
    """NaN-safe highest terrain point along a segment via height_at samples."""
    if emap is None:
        return float("nan")
    t = np.linspace(0.0, 1.0, n)
    xs = p0[0] + (p1[0] - p0[0]) * t
    ys = p0[1] + (p1[1] - p0[1]) * t
    vals = []
    for x, y in zip(xs, ys):
        try:
            h = float(emap.height_at(float(x), float(y)))
        except Exception:
            h = float("nan")
        if math.isfinite(h):
            vals.append(h)
    return max(vals) if vals else float("nan")


def _height_of(emap, x: float, y: float) -> float:
    if emap is None:
        return float("nan")
    try:
        h = float(emap.height_at(float(x), float(y)))
    except Exception:
        return float("nan")
    return h if math.isfinite(h) else float("nan")


def _max_height_near(emap, x: float, y: float, r: float) -> float:
    """Highest known map height on a 3 x 3 grid of spacing r around (x, y)
    (the foot sphere + placement error); NaN without a map / known cells."""
    if emap is None or r < 0.0:
        return float("nan")
    vals = []
    for dx in (-r, 0.0, r):
        for dy in (-r, 0.0, r):
            h = _height_of(emap, x + dx, y + dy)
            if math.isfinite(h):
                vals.append(h)
    return max(vals) if vals else float("nan")


def _footholds_finite(res) -> bool:
    ev = getattr(res, "foothold_events", None)
    if not ev:
        return False
    try:
        return bool(np.all(np.isfinite(np.asarray([e[1:] for e in ev], dtype=float))))
    except (TypeError, ValueError):
        return False


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


# ---------------------------------------------------------------------------
# MPC worker
# ---------------------------------------------------------------------------
class _MPCWorker:
    """Runs ConvexMPC.solve in a background thread (one job at a time)."""

    def __init__(self, mpc: ConvexMPC):
        self.mpc = mpc
        self._cv = threading.Condition()
        self._job = None
        self._result: Optional[MPCResult] = None
        self._busy = False
        self._stop = False
        self._reset_pending = False
        self.error: Optional[str] = None
        self._th = threading.Thread(target=self._loop, name="mpc_worker", daemon=True)
        self._th.start()

    def request_reset(self) -> None:
        """Reset the MPC's internal state (warm start, crawl path anchor,
        sway) from the control thread.  ``ConvexMPC.solve`` writes that state
        at its end, so a reset while a solve is running is deferred until the
        solve has finished (and always happens before the next job starts)."""
        with self._cv:
            if self._busy:
                self._reset_pending = True
            else:
                self.mpc.reset()
                self._reset_pending = False

    def submit(self, job) -> bool:
        with self._cv:
            if self._busy or self._job is not None:
                return False
            self._job = job
            self._cv.notify()
            return True

    def poll(self) -> Optional[MPCResult]:
        with self._cv:
            r, self._result = self._result, None
            return r

    @property
    def busy(self) -> bool:
        with self._cv:
            return self._busy or self._job is not None

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._th.join(timeout=1.0)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job, self._job = self._job, None
                self._busy = True
            try:
                res = self.mpc.solve(*job)
                err = None
            except Exception as e:      # keep the worker alive
                res, err = None, f"{type(e).__name__}: {e}"
            with self._cv:
                self._busy = False
                if self._reset_pending:
                    self.mpc.reset()
                    self._reset_pending = False
                if res is not None:
                    self._result = res
                self.error = err


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class LocomotionController:
    """See module docstring."""

    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 reset_fn: Optional[Callable[[float, float, float, float], Any]] = None,
                 threaded_mpc: Optional[bool] = None, logger: Optional[Callable[[str], None]] = None):
        """reset_fn(x, y, z, yaw): non-blocking gz set_pose for the assisted
        reset; return True only if its completion will be reported through
        :meth:`on_pose_reset_done`."""
        p = dict(DEFAULT_PARAMS)
        for k, v in (params or {}).items():
            if k in p:
                p[k] = v
        self.p = p
        self.log = logger or (lambda msg: None)
        self.reset_fn = reset_fn
        mode = str(p["locomotion_mode"]).strip().lower()
        if mode not in LOCOMOTION_MODES:
            raise ValueError(f"locomotion_mode must be one of {LOCOMOTION_MODES}, got {p['locomotion_mode']!r}")
        self.loco_mode = mode
        self.gaits = make_gaits(float(p["trot_period"]), float(p["trot_duty"]),
                                float(p["crawl_period"]), float(p["crawl_duty"]))
        self.gaits_stable = make_gaits(float(p["stable_trot_period"]), float(p["stable_trot_duty"]),
                                       float(p["stable_walk_period"]), float(p["stable_walk_duty"]))
        self.gait_mpc = GaitScheduler(self.gaits, initial="stand")
        self.gait_stable = GaitScheduler(self.gaits_stable, initial="stand")
        self.gait = self.gait_stable if mode == "stable" else self.gait_mpc
        self.mpc = ConvexMPC(mpc_config_from_params(p))
        threaded = bool(p["use_mpc_thread"]) if threaded_mpc is None else bool(threaded_mpc)
        self.worker: Optional[_MPCWorker] = _MPCWorker(self.mpc) if threaded else None
        fp = FallParams.from_dict({
            "fall_angle_deg": p["fall_angle_deg"], "fall_angle_exit_deg": float(p["fall_angle_deg"]) - 5.0,
            "fall_height": p["fall_height"], "fall_height_exit": float(p["fall_height"]) + 0.02,
            "detect_time": p["fall_detect_time"], "max_attempts": int(p["max_recovery_attempts"]),
            "lean_deg": p["fall_lean_deg"], "lean_time": p["fall_lean_time"],
            "assisted_reset": bool(p["assisted_reset"]), "reset_height_offset": p["reset_height_offset"],
            "reset_settle_time": p["reset_settle_time"], "reset_confirm_timeout": p["reset_confirm_timeout"],
            "failed_retry_time": float(p["recovery_retry_time"]) if bool(p["auto_recover"]) else 0.0,
            "stand_up_time": p["stand_up_duration"], "body_height": p["body_height"]})
        self.recovery = FallRecovery(fp, reset_fn=self._assisted_reset)
        self.slip = SlipDetector(SlipParams.from_controller_params(p))
        planner_cls, mha = _terrain_tools()
        self.planner = None
        if planner_cls is not None and bool(p["use_foothold_planner"]):
            try:
                self.planner = planner_cls(search_radius=float(p["foothold_search_radius"]))
            except Exception:
                self.planner = None
        self._max_height_along = mha
        self.q_stand = kin.stand_joint_angles(float(p["body_height"]))
        # body_x_offset > 0 puts the body ahead of the feet centre: the nominal
        # footholds (walking) move back by the same amount, and the STAND /
        # MPC hold position is the feet centre plus the offset
        self.nominal_feet_b = kin.nominal_feet_body(float(p["body_height"]), dx=-float(p["body_x_offset"]),
                                                    dy_out=float(p["stance_width_offset"]))
        self.stable = StableGait(StableParams.from_controller_params(p), self.nominal_feet_b)
        self._loco_mode_pending: Optional[str] = None
        self._last_worker_error: Optional[str] = None
        self._reset_internal()

    # ------------------------------------------------------------------ state
    def _reset_internal(self) -> None:
        self.mode = "STANDING_UP"
        self.started = False
        self.t_prev: Optional[float] = None
        self.t_mode = 0.0
        self.q_cmd = self.q_stand.copy()
        self.q_from = self.q_stand.copy()
        self.cmd_f = np.zeros(3)                 # filtered/limited (vx, vy, wz)
        self.cmd_raw = np.zeros(3)
        self.speed_scale = 1.0
        self.gait_mode = str(self.p["gait_mode"])
        self.gait_auto_choice = str(self.p["default_gait"])
        self._crawl_since_trot_rec: Optional[float] = None
        self.walk_in_place = False
        self.idle_time = 0.0
        self.pending_sit = False
        self.foot_w = np.zeros((4, 3))          # commanded world foot centres
        self.swing: List[Optional[SwingTrajectory]] = [None] * 4
        self.swing_t0 = np.zeros(4)
        self.ground_z = 0.0
        self.ground_valid = False
        self.terrain_pitch = 0.0
        self.body_cmd = None                     # (pos(3), rpy(3)) last body pose target (before PD / integrator)
        self.body_ik = None                      # (pos(3), rpy(3)) pose actually used for the leg IK
        self._walk_target = None                 # (xy(2), yaw) integrated WALK body target
        self._yaw_ref: Optional[float] = None    # stable mode heading hold
        self._stable_fh: Dict[int, Any] = {}     # stable mode foothold planner cache
        self.z_int = 0.0
        self.yaw_hold = 0.0
        self.sit_from_h = float(self.p["body_height"])
        self.mpc_result: Optional[MPCResult] = None
        self.mpc_footholds: Optional[MPCResult] = None   # newest result with valid footholds (see _accept_mpc)
        self.t_mpc_submit = -1e9
        self._mpc_gen = getattr(self, "_mpc_gen", 0) + 1   # bumped on every reset of the motion
        self._inflight_gen = -1                  # generation of the job running in the worker
        self.mpc_stats: List[float] = []
        self.mpc_rejected = 0
        self.terrain_info: Optional[dict] = None
        self.t_terrain = -1e9
        self.events: List[dict] = []
        self.last_debug: Dict[str, Any] = {}
        self.estop_q: Optional[np.ndarray] = None
        self.reset_settle_until = -1e9
        self.reachable = True
        self._retarget_stamp: Dict[int, float] = {}
        self._feet_meas_now: Optional[np.ndarray] = None    # measured feet of this WALK step (MPC mode)
        self._feet_cmd_w = np.zeros((4, 3))
        self._terrain_now = np.full(5, np.nan)     # map heights under base + nominal feet (this step)
        # slippery ground (see _update_slip)
        self.slip.reset()
        self.slip_mode = False
        self.slip_source = "none"
        self.slip_events = 0
        self.ff_gain = float(self.p["torque_feedforward"])
        self.mu_mpc = float(self.p["mpc_mu"])
        self.friction_prior: Optional[float] = None
        self._set_stance_width(0.0)
        self.recovery.reset()
        self.gait.reset(0.0, "stand")
        self.stable.reset()
        self._reset_mpc_state()

    def _reset_mpc_state(self) -> None:
        """Reset the MPC warm start / crawl anchor without racing a solve
        that is running in the worker thread."""
        if self.worker is not None:
            self.worker.request_reset()
        else:
            self.mpc.reset()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def pop_events(self) -> List[dict]:
        ev = self.events + self.recovery.pop_events()
        self.events = []
        return ev

    def _set_mode(self, mode: str, t: float) -> None:
        if mode != self.mode:
            self.log(f"mode {self.mode} -> {mode}")
        self.mode = mode
        self.t_mode = t

    # ------------------------------------------------------------------ helpers
    def _feet_world_measured(self, st: RobotState) -> np.ndarray:
        return kin.body_to_world(kin.fk_all(st.q), st.pos, st.R)

    def _ground_from_feet(self, feet_w: np.ndarray, mask=None) -> float:
        f = feet_w if mask is None else feet_w[np.asarray(mask, dtype=bool)]
        if len(f) == 0:
            return self.ground_z
        return float(np.mean(f[:, 2])) - rm.FOOT_RADIUS

    def _update_ground(self, dt: float, z_new: float) -> None:
        if not math.isfinite(z_new):
            return
        if not self.ground_valid:
            self.ground_z = z_new
            self.ground_valid = True
            return
        a = min(1.0, dt / max(float(self.p["ground_filter_tau"]), 1e-3))
        self.ground_z += a * (z_new - self.ground_z)

    def _query_terrain(self, emap, st: RobotState) -> np.ndarray:
        """Terrain heights (5,) under the base and the four nominal footholds
        in ONE (vectorised) elevation-map query per control step; NaN where
        unknown or without a map."""
        xy = np.vstack([np.asarray(st.pos[:2], dtype=float), self._nominal_world_xy(st)])
        if emap is None:
            return np.full(5, np.nan)
        return terrain_heights(emap.height_at, xy[:, 0], xy[:, 1])

    def _body_height_kinematic(self, st: RobotState) -> float:
        """Base height above the measured feet (leg kinematics + orientation).
        Needs no map: equals the body height whenever the feet stand on the
        ground, and is small for a robot on its belly or side."""
        return float(st.pos[2] - self._ground_from_feet(self._feet_world_measured(st)))

    def _body_height_measured(self, st: RobotState, h_terrain: float) -> float:
        """Base height above the terrain (map height under the base if it is
        plausible, else the ground estimate from the stance feet)."""
        h = float(h_terrain)
        g = h if math.isfinite(h) and abs(h - self.ground_z) < 0.6 else self.ground_z
        return float(st.pos[2] - g)

    def _limits(self, gait_name: str) -> np.ndarray:
        p = self.p
        if self.loco_mode == "stable":
            g = "walk" if gait_name == "crawl" else "trot"
            return np.array([p[f"stable_{g}_max_vx"], p[f"stable_{g}_max_vy"], p[f"stable_{g}_max_wz"]],
                            dtype=float)
        g = "crawl" if gait_name == "crawl" else "trot"
        return np.array([p[f"{g}_max_vx"], p[f"{g}_max_vy"], p[f"{g}_max_wz"]], dtype=float)

    def _assisted_reset(self, x: float, y: float, z: float, yaw: float):
        """FallRecovery's reset_fn.  Passes on the return value of
        ``reset_fn``: True promises a later :meth:`on_pose_reset_done` call
        (see FallRecovery), anything else starts the settle window now."""
        self.events.append({"event": "reset_request", "t": self.t_prev or 0.0})
        if self.reset_fn is not None:
            return self.reset_fn(x, y, z, yaw)
        return None

    # ------------------------------------------------------------------ commands
    def start(self, t: float, st: RobotState) -> None:
        """Begin control: STANDING_UP from the measured joint configuration."""
        self._reset_internal()
        self.started = True
        self.t_prev = t
        self.gait.reset(t, "stand")
        feet = self._feet_world_measured(st)
        self.ground_z = float(np.min(feet[:, 2])) - rm.FOOT_RADIUS
        self.ground_valid = True
        self._begin_standing_up(t, st)

    def _begin_standing_up(self, t: float, st: RobotState, delay: float = 0.0) -> None:
        self.q_from = kin.clamp_joints(np.asarray(st.q, dtype=float))
        self.q_cmd = self.q_from.copy()
        self.reset_settle_until = t + delay
        self.gait.reset(t, "stand")
        self.swing = [None] * 4
        self._reset_motion_state()
        self._invalidate_mpc(t)
        self._set_mode("STANDING_UP", t + delay)

    def _reset_motion_state(self) -> None:
        """Forget the shaped command and the pending motion requests.

        Called whenever locomotion is interrupted (fall, stand-up, E-stop
        release): ``cmd_f`` is only updated while standing / walking, so a
        value left over from before the interruption would otherwise skip the
        acceleration ramp at the next walk start (e.g. resume at 0.6 m/s right
        after a recovery), and a ``walk`` / ``sit`` request made before a fall
        must not be executed by surprise afterwards."""
        self.cmd_f = np.zeros(3)
        self.walk_in_place = False
        self.pending_sit = False
        self.idle_time = 0.0

    def _invalidate_mpc(self, t: float) -> None:
        """Drop the current prediction and any solve still running in the
        worker (its result belongs to the previous motion; see
        :meth:`_mpc_cycle`).  A generation counter rather than a time stamp,
        so that it also works after the clock jumped back (simulation reset)."""
        self._reset_mpc_state()
        self.mpc_result = None
        self.mpc_footholds = None
        self._mpc_gen += 1
        self.t_mpc_submit = -1e9

    def notify_pose_reset(self, t: float, st: RobotState) -> None:
        """Called after a gz set_pose: wait for the drop, then stand up."""
        self.recovery.reset()
        self._begin_standing_up(t, st, delay=float(self.p["reset_settle_time"]))

    def on_pose_reset_done(self, t: float, st: RobotState, ok: bool) -> None:
        """Result of a gz set_pose request (``reset_pose`` command or assisted
        reset), reported when the service call has returned - i.e. when the
        robot has actually been teleported, ~0.3-1 s after the request.

        * an assisted reset waiting for the teleport: its settle window starts
          now (failure: recovery FAILED, see FallRecovery.confirm_reset);
        * otherwise (``reset_pose`` command) ok: settle, then stand up
          (:meth:`notify_pose_reset`); E-STOP stays engaged; failure: logged,
          the robot carries on in its current mode."""
        if self.recovery.waiting_for_reset:
            self.recovery.confirm_reset(t, ok)
            if not ok:
                self.log("assisted reset: gz set_pose failed - recovery FAILED "
                         "(send 'recover' or 'reset_pose' to retry)")
            return
        if not ok:
            self.log("reset_pose: gz set_pose failed - robot not moved")
            return
        if self.mode == "ESTOP":
            return
        self.notify_pose_reset(t, st)

    def handle_command(self, text: str, t: Optional[float] = None,
                       st: Optional[RobotState] = None) -> Optional[dict]:
        """Process a /locomotion/command string.  Returns an action dict for the
        node when the command needs the gz CLI (push / reset_pose)."""
        cmd = (text or "").strip().lower()
        t = self.t_prev if t is None else t
        t = 0.0 if t is None else t
        if cmd.startswith("gait:"):
            g = cmd.split(":", 1)[1].strip()
            g = "crawl" if g == "walk" else g
            if g in ("trot", "crawl", "auto"):
                self.gait_mode = g
            return None
        if cmd.startswith("mode:") or cmd.startswith("locomotion_mode:"):
            self.request_locomotion_mode(cmd.split(":", 1)[1].strip())
            return None
        if cmd == "estop":
            if self.mode != "ESTOP":
                self.estop_q = self.q_cmd.copy()
                self._estop_lower_swing(st)
                self._set_mode("ESTOP", t)
            return None
        if cmd == "release":
            if self.mode == "ESTOP":
                # E-STOP engaged during a fall / recovery: the recovery state machine
                # must not stay active (fall detection only runs while it is NORMAL)
                self.recovery.reset()
            if self.mode == "ESTOP" and st is not None:
                self._begin_standing_up(t, st)
            elif self.mode == "ESTOP":
                self._reset_motion_state()
                self._set_mode("STANDING_UP", t)
                self.q_from = self.q_cmd.copy()
            return None
        if self.mode == "ESTOP":
            return None                          # everything else is ignored until release
        if cmd == "push" or cmd.startswith("push:"):
            fx, fy = float(self.p["push_force_x"]), float(self.p["push_force_y"])
            if ":" in cmd:
                try:
                    parts = [float(v) for v in cmd.split(":", 1)[1].split(",")]
                    fx = parts[0]
                    fy = parts[1] if len(parts) > 1 else 0.0
                except ValueError:
                    return None
            return {"action": "push", "force": (fx, fy, 0.0), "duration": float(self.p["push_duration"])}
        if cmd == "reset_pose":
            return {"action": "reset_pose", "height_offset": float(self.p["reset_height_offset"])}
        if cmd == "recover":
            if self.mode in ("FALLEN", "RECOVERING"):
                if st is not None:
                    self.recovery.begin(t, st.q, reset_attempts=True)
                    self._set_mode("RECOVERING", t)
            elif self.mode in ("STAND", "SIT", "STANDING_UP") and st is not None:
                # not fallen: re-run the stand-up from the measured posture
                self._begin_standing_up(t, st)
            return None
        if self.mode in ("FALLEN", "RECOVERING"):
            return None
        if cmd == "stand":
            self.walk_in_place = False
            self.pending_sit = False
            if self.mode == "WALK":
                self.gait.request("stand")
            elif self.mode == "SIT" and st is not None:
                self._begin_standing_up(t, st)
            return None
        if cmd == "sit":
            self.walk_in_place = False
            if self.mode == "WALK":
                self.pending_sit = True
                self.gait.request("stand")
            elif self.mode == "STAND":
                self._enter_sit(t)
            return None
        if cmd == "walk":
            self.walk_in_place = True
            self.pending_sit = False
            if self.mode == "STAND":
                self._enter_walk(t)
            return None
        return None

    def request_locomotion_mode(self, mode: str) -> bool:
        """Switch between the ``stable`` and the ``mpc`` locomotion mode
        (``toggle``: the other one).  The switch is applied at the next control
        step in which the robot is not walking; while walking the gait is
        first brought to STAND at a safe boundary (and resumes in the new mode
        if the velocity command is still active).  False for an unknown mode."""
        m = (mode or "").strip().lower()
        if m == "toggle":
            cur = self._loco_mode_pending or self.loco_mode
            m = "mpc" if cur == "stable" else "stable"
        if m not in LOCOMOTION_MODES:
            self.log(f"unknown locomotion mode {mode!r} (stable | mpc | toggle)")
            return False
        self._loco_mode_pending = None if m == self.loco_mode else m
        return True

    def _apply_locomotion_mode(self, t: float, st: Optional[RobotState]) -> None:
        m, self._loco_mode_pending = self._loco_mode_pending, None
        if m is None or m == self.loco_mode:
            return
        self.log(f"locomotion mode {self.loco_mode} -> {m}")
        self.events.append({"event": "locomotion_mode", "t": float(t), "mode": m, "from": self.loco_mode})
        self.loco_mode = m
        self.gait = self.gait_stable if m == "stable" else self.gait_mpc
        self.gait.reset(t, "stand")
        self.swing = [None] * 4
        self.stable.reset()
        self.speed_scale = 1.0
        self._walk_target = None
        self._invalidate_mpc(t)
        if self.mode == "STAND" and st is not None:
            self._enter_stand(t, st)             # re-anchor the new mode on the measured posture

    def set_terrain_info(self, info: Optional[dict], t: float) -> None:
        if isinstance(info, dict):
            self.terrain_info = info
            self.t_terrain = t

    # ------------------------------------------------------------------ step
    def step(self, t: float, st: RobotState, cmd_vel=(0.0, 0.0, 0.0), terrain_info: Optional[dict] = None,
             emap=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """One control cycle -> ((12,) joint position targets, debug dict)."""
        if not self.started:
            self.start(t, st)
        if self.t_prev is not None and t < self.t_prev - 0.5:
            # clock jumped back (simulation reset): restart cleanly
            self.start(t, st)
        dt = 0.0 if self.t_prev is None else float(np.clip(t - self.t_prev, 0.0, 0.05))
        self.t_prev = t
        if terrain_info is not None:
            self.set_terrain_info(terrain_info, t)
        self.cmd_raw = np.asarray(cmd_vel, dtype=float).reshape(3)

        self._terrain_now = self._query_terrain(emap, st)
        body_h = self._body_height_measured(st, self._terrain_now[0])
        roll, pitch, yaw = (float(v) for v in st.rpy)

        # ---------------- fall detection
        if self.mode in ("STAND", "WALK", "SIT", "STANDING_UP"):
            if self.recovery.check(t, roll, pitch, body_h, st.pos[0], st.pos[1], st.q):
                self.swing = [None] * 4
                self._reset_motion_state()
                self._invalidate_mpc(t)
                self._set_mode("FALLEN", t)
                if not bool(self.p["auto_recover"]):
                    self.recovery.state = "FAILED"

        # ---------------- locomotion mode switch (never mid-stride)
        if self._loco_mode_pending is not None and self.mode != "WALK":
            self._apply_locomotion_mode(t, st)

        # ---------------- mode dispatch
        if self.mode in ("FALLEN", "RECOVERING"):
            q = self._step_recovery(t, st, body_h)
        elif self.mode == "ESTOP":
            q = self.estop_q.copy() if self.estop_q is not None else self.q_cmd.copy()
        elif self.mode == "STANDING_UP":
            q = self._step_standing_up(t, st)
        elif self.mode == "SIT":
            q = self._step_sit(t, dt, st, emap)
        else:
            q = self._step_locomotion(t, dt, st, emap)
        if self.mode not in ("STAND", "WALK"):
            # no stance feet to watch (fallen, standing up, sitting, E-STOP): the
            # slip histories restart, the flag can time out, the feed-forward fades
            self._update_slip(t, dt, st, np.zeros(4, dtype=bool))

        # ---------------- joint limits + rate limit
        if not np.all(np.isfinite(q)):
            # e.g. a non-finite pose from a diverged simulation: never send NaN to the servos
            self.log("non-finite joint targets - holding the previous targets")
            q = self.q_cmd.copy()
        q = kin.clamp_joints(q)
        if dt > 0.0:
            dq_max = float(self.p["max_joint_speed"]) * dt
            q = self.q_cmd + np.clip(q - self.q_cmd, -dq_max, dq_max)
        self.q_cmd = q
        dbg = self._debug(t, st, body_h)
        self.last_debug = dbg
        return q.copy(), dbg

    # ------------------------------------------------------------------ STANDING_UP
    def _step_standing_up(self, t: float, st: RobotState) -> np.ndarray:
        if t < self.reset_settle_until:
            self.q_from = kin.clamp_joints(np.asarray(st.q, dtype=float))
            return self.q_from.copy()
        u = (t - self.t_mode) / max(float(self.p["stand_up_duration"]), 1e-3)
        q = self.q_from + (self.q_stand - self.q_from) * _smoothstep(u)
        if u >= 1.0:
            self._enter_stand(t, st)
        return q

    def _enter_stand(self, t: float, st: RobotState) -> None:
        self.stable.reset()                      # stable mode: re-initialised from the measured posture
        self.foot_w = self._feet_world_measured(st)
        self._update_ground(1.0, self._ground_from_feet(self.foot_w))
        self.yaw_hold = float(st.rpy[2])
        self.body_cmd = (np.asarray(st.pos, dtype=float).copy(), np.array([0.0, 0.0, float(st.rpy[2])]))
        self.z_int = 0.0
        self._walk_target = None
        self.swing = [None] * 4
        self.gait.reset(t, "stand")
        self.idle_time = 0.0
        self._set_mode("STAND", t)
        if self.pending_sit:
            self.pending_sit = False
            self._enter_sit(t)

    def _enter_walk(self, t: float) -> None:
        self.idle_time = 0.0
        self._walk_target = None                 # re-initialised from the last body command
        self._feet_meas_now = None
        self.gait.request(self._target_gait(t))
        self.t_mpc_submit = -1e9                 # re-plan immediately
        self._set_mode("WALK", t)

    def _enter_sit(self, t: float) -> None:
        cur = self.body_cmd[0][2] - self.ground_z if self.body_cmd is not None else float(self.p["body_height"])
        self.sit_from_h = float(cur)
        self._set_mode("SIT", t)

    # ------------------------------------------------------------------ SIT
    def _step_sit(self, t: float, dt: float, st: RobotState, emap) -> np.ndarray:
        u = (t - self.t_mode) / max(float(self.p["sit_duration"]), 1e-3)
        h = self.sit_from_h + (float(self.p["sit_height"]) - self.sit_from_h) * _smoothstep(u)
        centre = self.foot_w[:, :2].mean(axis=0)
        yaw = self.yaw_hold
        pos = np.array([centre[0], centre[1], self.ground_z + h])
        rpy = np.array([0.0, 0.0, yaw])
        self.body_cmd = (pos, rpy)
        self.body_ik = (pos.copy(), rpy.copy())
        return self._ik_from_body(pos, rpy, self.foot_w)

    # ------------------------------------------------------------------ recovery
    def _step_recovery(self, t: float, st: RobotState, body_h: float) -> np.ndarray:
        rec = self.recovery
        # upright check of the recovery: the map under the base can be wrong right
        # after a fall (the depth camera tumbled with the body) - the height above
        # the measured feet confirms a robot standing on its legs
        h_kin = self._body_height_kinematic(st)
        if math.isfinite(h_kin):
            body_h = max(body_h, h_kin) if math.isfinite(body_h) else h_kin
        q = rec.update(t, st.q, float(st.rpy[0]), float(st.rpy[1]), body_h, st.omega_body,
                       x=float(st.pos[0]), y=float(st.pos[1]), yaw=float(st.rpy[2]), z=float(st.pos[2]))
        if rec.take_reset_done():
            self._begin_standing_up(t, st)
            return self.q_cmd.copy()
        if rec.state == "NORMAL":
            # recovered: continue as STAND from the measured posture
            self._enter_stand(t, st)
            return q if q is not None else self.q_stand.copy()
        if rec.fallen and self.mode != "FALLEN":
            self._set_mode("FALLEN", t)
        elif rec.recovering and self.mode != "RECOVERING":
            self._set_mode("RECOVERING", t)
        return q if q is not None else np.asarray(st.q, dtype=float).copy()

    def _estop_lower_swing(self, st: Optional[RobotState]) -> None:
        """On E-STOP put swinging feet down at their current xy, then hold."""
        if self.mode != "WALK" or self.body_cmd is None or self.estop_q is None:
            return
        if self.loco_mode == "stable":
            if self.stable.initialized:
                self.estop_q = self.stable.lowered_targets()
            return
        pos, rpy = self.body_cmd
        feet = self.foot_w.copy()
        for leg in range(4):
            if self.swing[leg] is not None:
                feet[leg, 2] = self.ground_z + rm.FOOT_RADIUS
        self.estop_q = self._ik_from_body(pos, rpy, feet)
        self.swing = [None] * 4

    # ------------------------------------------------------------------ gait choice
    def _target_gait(self, t: float) -> str:
        if self.gait_mode in ("trot", "crawl"):
            return self.gait_mode
        info = self.terrain_info
        fresh = info is not None and (t - self.t_terrain) <= float(self.p["terrain_info_timeout"])
        rec = str(info.get("recommended_gait", "")).lower() if fresh else ""
        if rec == "crawl":
            self.gait_auto_choice = "crawl"
            self._crawl_since_trot_rec = None
        elif rec == "trot":
            if self.gait_auto_choice == "crawl":
                # hysteresis: return to trot only after a sustained recommendation
                if self._crawl_since_trot_rec is None:
                    self._crawl_since_trot_rec = t
                if t - self._crawl_since_trot_rec >= float(self.p["crawl_to_trot_delay"]):
                    self.gait_auto_choice = "trot"
                    self._crawl_since_trot_rec = None
            else:
                self.gait_auto_choice = "trot"
        if self.gait_auto_choice not in ("trot", "crawl"):
            self.gait_auto_choice = str(self.p["default_gait"])
        slip_gait = str(self.p["slip_gait"]).lower()
        if self.slip_mode and slip_gait in ("trot", "crawl"):
            return slip_gait                     # forced slip gait overrides the terrain choice
        return self.gait_auto_choice

    def _shape_command(self, t: float, dt: float, gait_name: str) -> None:
        lim = self._limits(gait_name)
        lim_target = lim
        acc = np.array([self.p["max_accel_x"], self.p["max_accel_y"], self.p["max_accel_wz"]], dtype=float)
        if self.slip_mode:
            # slippery: lower limits, reached with lower (friction-compatible)
            # accelerations - the target is limited, not cmd_f, so the slow-down
            # itself is rate limited (a sudden stop would slide the feet as well)
            slip_lim = np.array([self.p["slip_max_vx"], self.p["slip_max_vy"], self.p["slip_max_wz"]], dtype=float)
            lim_target = np.minimum(lim, slip_lim)
            acc = acc * float(self.p["slip_accel_scale"])
        v = np.clip(self.cmd_raw, -lim_target, lim_target) * self.speed_scale
        info = self.terrain_info
        if info is not None and (t - self.t_terrain) <= float(self.p["terrain_info_timeout"]):
            if info.get("ahead_clear") is False and v[0] > 0:
                v[0] *= float(self.p["blocked_speed_scale"])
        # the speed scale acts on the target, so a saturation slow-down is rate limited too
        self.cmd_f = self.cmd_f + np.clip(v - self.cmd_f, -acc * dt, acc * dt)
        # never exceed the limits of the gait actually running (e.g. after trot -> crawl)
        self.cmd_f = np.clip(self.cmd_f, -lim, lim)

    # ------------------------------------------------------------------ STAND / WALK
    def _step_locomotion(self, t: float, dt: float, st: RobotState, emap) -> np.ndarray:
        p = self.p
        target_gait = self._target_gait(t)
        running = self.gait.name if self.gait.name != "stand" else target_gait
        # while the gait is switching, respect the slower of both limits
        lim_gait = "crawl" if "crawl" in (running, target_gait) else "trot"
        self._shape_command(t, dt, lim_gait)
        db = float(p["cmd_deadband"])
        moving_cmd = bool(np.any(np.abs(self.cmd_raw) > db))
        moving = moving_cmd or bool(np.any(np.abs(self.cmd_f) > db))

        if self.mode == "STAND":
            if self.pending_sit:
                self.pending_sit = False
                self._enter_sit(t)
                return self._step_sit(t, dt, st, emap)
            if (moving_cmd and bool(p["auto_walk"])) or self.walk_in_place:
                self._enter_walk(t)
        elif self.mode == "WALK":
            switching = self._loco_mode_pending is not None
            if switching:
                # locomotion mode switch requested: stop at the next safe boundary first
                if self.gait.name != "stand":
                    self.gait.request("stand")
            elif self.gait.name != "stand" and self.gait.pending is None and target_gait != self.gait.name:
                self.gait.request(target_gait)
            if not moving and not self.walk_in_place:
                self.idle_time += dt
                if self.idle_time >= float(p["idle_to_stand_time"]):
                    self.gait.request("stand")
            else:
                self.idle_time = 0.0
                if self.gait.pending is not None and self.gait.pending.name == "stand" and not self.pending_sit \
                        and not switching:
                    self.gait.request(target_gait if self.gait.name == "stand" else self.gait.name)

        gs = self.gait.update(t)
        if self.mode == "WALK" and gs.name == "stand" and self.gait.pending is None:
            self._enter_stand_from_walk(t, st)
            if self.mode == "SIT":
                return self._step_sit(t, dt, st, emap)

        if self.loco_mode == "stable":
            return self._step_stable(t, dt, st, gs, emap)

        # ---------------- ground / terrain estimate
        stance_mask = gs.contacts & np.array([self.swing[i] is None for i in range(4)])
        z_feet = self._ground_from_feet(self.foot_w, stance_mask) if stance_mask.any() else self.ground_z
        terr = self._terrain_now[1:]
        known = np.isfinite(terr)
        if known.sum() >= 2:
            # median: a pipe / a wall-edge cell under one nominal foot does not lift the body
            z_map = float(np.median(terr[known])) if bool(p["mpc_terrain_robust"]) else float(np.mean(terr[known]))
            # trust the map only if it agrees with the feet within a stair height
            self._update_ground(dt, z_map if abs(z_map - z_feet) < 0.35 else z_feet)
        else:
            self._update_ground(dt, z_feet)
        self.terrain_pitch = self._terrain_pitch(terr, self.foot_w)

        # ---------------- touchdown / lift-off events
        for leg in range(4):
            if gs.touchdown[leg] or (gs.contacts[leg] and self.swing[leg] is not None):
                self._touchdown(leg, st)
            if not gs.contacts[leg] and self.swing[leg] is None:
                self._liftoff(leg, t, gs, st, emap)

        # ---------------- stance feet: force limit + height on the map
        stance_now = np.array([bool(gs.contacts[i]) and self.swing[i] is None for i in range(4)])
        if self.mode == "WALK":
            self._mpc_stance_feet(dt, st, stance_now, emap)

        # ---------------- slip detection / slip mode (before the MPC: its friction coefficient)
        self._update_slip(t, dt, st, stance_now)

        # ---------------- MPC
        self._mpc_cycle(t, st, gs, emap)

        # ---------------- body target
        pos_des, rpy_des = self._body_target(t, dt, st)
        # IMU roll/pitch PD correction + height integrator.  With a stiff joint
        # servo the body follows the IK attitude within a few ms, so the loop
        # corr_k = kp (des - meas_k) - kd w_k sees the plant as a short lag tau:
        # the D term then acts like a gain kd / tau on the previous correction
        # and kd = 0.02 limit-cycles (about +-0.1 rad at 50 Hz) for tau ~ 5-10 ms;
        # kd <= 0.01 keeps the loop stable for any tau (test_controller).
        # kp_attitude defaults to 0: in Gazebo (gz-sim 8 + DART, ~30 ms MPC delay)
        # the P correction reached 8.6 deg of IK roll in the failed pipe / push runs;
        # with the stance force limit and the heading hold, kp 0 crossed the 0.08 m
        # pipe on 5/6 seeds (0.5: 2/6) and survived the pushes 6/6 (0.5: 4/6).
        kp, kd = float(p["kp_attitude"]), float(p["kd_attitude"])
        cmax = float(p["max_attitude_correction"])
        w = np.asarray(st.omega_body, dtype=float)
        corr_r = float(np.clip(kp * (rpy_des[0] - st.rpy[0]) - kd * w[0], -cmax, cmax))
        corr_p = float(np.clip(kp * (rpy_des[1] - st.rpy[1]) - kd * w[1], -cmax, cmax))
        rpy_cmd = np.array([rpy_des[0] + corr_r, rpy_des[1] + corr_p, rpy_des[2]])
        zi_max = float(p["max_height_integral"])
        z_err = float(pos_des[2] - st.pos[2])
        if math.isfinite(z_err):                 # a NaN must not latch in the integrator
            self.z_int = float(np.clip(self.z_int + float(p["ki_height"]) * z_err * dt, -zi_max, zi_max))
        pos_cmd = pos_des + np.array([0.0, 0.0, self.z_int])
        self.body_cmd = (pos_des, rpy_des)
        self.body_ik = (pos_cmd.copy(), rpy_cmd.copy())

        # ---------------- feet
        feet = self.foot_w.copy()
        for leg in range(4):
            tr = self.swing[leg]
            if tr is not None:
                s = gs.swing_progress[leg] if not gs.contacts[leg] else 1.0
                self._retarget(leg, s, t, emap)
                feet[leg] = tr.position(s)
                feet[leg, 2] = self._swing_guard_z(leg, s, tr, float(feet[leg, 2]), emap)
        if self.mode == "WALK":
            feet = self._stance_deflection_limit(st, stance_now, pos_cmd, rpy_cmd, feet)
        self._feet_cmd_w = feet
        q = self._ik_from_body(pos_cmd, rpy_cmd, feet)
        return self._torque_feedforward(t, q, gs, rpy_cmd)

    def _torque_feedforward(self, t: float, q: np.ndarray, gs, rpy_cmd) -> np.ndarray:
        """Realise the MPC ground reaction forces through the PD joint servo.

        The gz JointPositionController (force mode) produces
        ``tau = p (q_target - q) - d de/dt``: a stance leg only pushes with the
        force the MPC planned after it has been deflected by ``tau / p``.
        Under two legs (trot) that deflection sagged the body ~1-2 cm and the
        swing legs, positioned relative to the commanded body, were then
        driven 1-3 cm into the ground: the next pair touched down pre-loaded
        and launched the body (MuJoCo physics check: airborne phases, hopping
        after pushes).  Adding the planned torque ``tau = -J^T R^T f`` of the
        stance legs to the target as ``tau / servo_kp`` removes the deflection
        (``torque_feedforward`` = 1: the full MPC force, 0: off).  Legs in
        stance for which the plan has no force (timing mismatch, stale or no
        plan) get the static weight share ``m g / n_stance``.

        Physics check (3 seeds each): trot 0.4 m/s stance slip p95 0.015 vs
        0.19 m/s, stand / payload height exactly 0.340 m, turn 0.497 vs
        0.49 rad/s - but the 15 N s lateral push while trotting fell on 6/6
        seeds (0/6 also with the vertical component only) against 3/6
        without it.  Off by default until the push case is understood."""
        p = self.p
        gain = float(self.ff_gain)               # torque_feedforward, slip_torque_feedforward in slip mode
        kp = float(p["servo_kp"])
        if gain <= 0.0 or kp <= 0.0:
            return q
        stance = np.array([bool(gs.contacts[i]) and self.swing[i] is None for i in range(4)])
        if not stance.any():
            return q
        f = np.zeros((4, 3))
        res = self.mpc_result
        if res is not None and res.ok and (t - res.t) <= float(p["mpc_max_age"]):
            tr = t - res.t
            k = int(np.clip(np.searchsorted(res.times, tr, side="right") - 1, 0, len(res.dts) - 1))
            cf = np.asarray(res.contacts[k], dtype=float)
            fk = np.asarray(res.forces[k], dtype=float).reshape(4, 3)
            f = np.where(cf[:, None] > 0.05, fk / np.maximum(cf[:, None], 0.05), 0.0)
        f[~stance] = 0.0
        missing = stance & (np.linalg.norm(f, axis=1) < 1e-6)
        if missing.any():
            f[missing] = np.array([0.0, 0.0, float(p["mpc_mass"]) * self.mpc.cfg.gravity / stance.sum()])
        if not np.all(np.isfinite(f)):
            return q
        R = euler_to_rot(float(rpy_cmd[0]), float(rpy_cmd[1]), float(rpy_cmd[2]))
        tau = kin.leg_torques(np.asarray(q, dtype=float).reshape(4, 3), np.arange(4), f @ R).reshape(12)
        tau = np.clip(tau, -kin.EFFORT_LIMITS, kin.EFFORT_LIMITS)
        out = np.asarray(q, dtype=float).copy()
        mask = np.repeat(stance, 3)
        out[mask] += gain * tau[mask] / kp
        return out

    def _stable_terrain(self, dt: float, st: RobotState, emap, stance_feet_z: float):
        """Terrain inputs of the stable gait from the elevation map:
        (ground_fn, obstacle_fn, rp_des, foothold_fn) or all None without a
        usable map (flat ground at the nominal height).  foothold_fn = the
        foothold planner (as in the MPC mode) on the Raibert touch-down point.  The reference ground under
        the body is the filtered mean map height under the nominal footholds
        (trusted only within 0.35 m of the stance feet, like the MPC mode);
        heights are expressed in the level frame of the body at its reference
        height (ground + body_height)."""
        terr = self._terrain_now[1:]
        known = np.isfinite(terr)
        if emap is None or known.sum() < 2:
            return None, None, None, None
        z_map = float(np.mean(terr[known]))
        if abs(z_map - stance_feet_z) >= 0.35:
            return None, None, None, None
        self._update_ground(dt, z_map)
        self.terrain_pitch = self._terrain_pitch(terr, self.foot_w)
        z_ref = self.ground_z
        h = float(self.p["body_height"])
        yaw = float(st.rpy[2])
        c, s_ = math.cos(yaw), math.sin(yaw)
        bx, by = float(st.pos[0]), float(st.pos[1])

        def to_world(xy):
            xy = np.atleast_2d(np.asarray(xy, dtype=float))
            return bx + c * xy[:, 0] - s_ * xy[:, 1], by + s_ * xy[:, 0] + c * xy[:, 1]

        def ground_fn(xy, _emap=emap):
            wx, wy = to_world(xy)
            hz = terrain_heights(_emap.height_at, wx, wy)
            hz = np.where(np.isfinite(hz) & (np.abs(hz - z_ref) < 0.35), hz, z_ref)
            return hz - z_ref - h + rm.FOOT_RADIUS

        mha = self._max_height_along

        def obstacle_fn(p0, p1, _emap=emap):
            w0 = np.array(to_world(p0[:2])).reshape(2)
            w1 = np.array(to_world(p1[:2])).reshape(2)
            try:
                top = float(mha(_emap, w0, w1)) if mha is not None else \
                    _max_height_along_fallback(_emap, w0, w1)
            except Exception:
                top = _max_height_along_fallback(_emap, w0, w1)
            if not math.isfinite(top) or abs(top - z_ref) > 0.35:
                return float("nan")              # unknown / implausible (wall, map artefact)
            return top - z_ref - h

        foothold_fn = None
        planner = self.planner
        if planner is not None:
            cache = self._stable_fh                  # leg -> (raw world xy, chosen world xy)

            def foothold_fn(leg, xy, _emap=emap):
                wx, wy = to_world(xy)
                w = np.array([float(wx[0]), float(wy[0])])
                hit = cache.get(leg)
                if hit is not None and float(np.hypot(*(w - hit[0]))) < 0.01:
                    wa = hit[1]                      # ground-fixed target barely moved: same choice
                else:
                    try:
                        pt, _score, valid = planner.select(_emap, w, z_ref)
                        wa = np.asarray(pt[:2], dtype=float) if valid else w
                    except Exception:
                        wa = w
                    cache[leg] = (w, wa)
                d = wa - np.array([bx, by])
                return np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]])

        return ground_fn, obstacle_fn, np.array([0.0, float(self.terrain_pitch)]), foothold_fn

    def _stable_heading_hold(self, st: RobotState, walking: bool) -> np.ndarray:
        """Shaped command + heading hold: while no turn is commanded the yaw
        measured when the last turn ended is held with a small P correction
        (``stable_heading_hold_gain`` / ``_max``).  The kinematic gait itself
        has no heading feedback; without it a straight walk drifts (Gazebo:
        ~20 deg over 6 m on the 11.8 deg ramp)."""
        cmd = np.asarray(self.cmd_f, dtype=float).copy()
        cmd[2] += self._heading_hold(st, walking, float(self.p["stable_heading_hold_gain"]),
                                     float(self.p["stable_heading_hold_max"]))
        return cmd

    def _heading_hold(self, st: RobotState, walking: bool, k: float, wmax: float) -> float:
        """Yaw-rate correction of the heading hold (both locomotion modes):
        while walking without a turn command (raw and shaped yaw rate within
        the deadband) the yaw measured when the last turn ended is held with
        ``clip(k (yaw_ref - yaw), +-wmax)``; 0 otherwise."""
        db = float(self.p["cmd_deadband"])
        if not walking or k <= 0.0 or abs(float(self.cmd_raw[2])) > db or abs(float(self.cmd_f[2])) > db:
            self._yaw_ref = None
            return 0.0
        yaw = float(st.rpy[2])
        if self._yaw_ref is None or not math.isfinite(yaw):
            self._yaw_ref = yaw
            return 0.0
        return float(np.clip(k * wrap_angle(self._yaw_ref - yaw), -wmax, wmax))

    def _step_stable(self, t: float, dt: float, st: RobotState, gs, emap=None) -> np.ndarray:
        """STAND / WALK of the ``stable`` mode (stable_gait.StableGait):
        kinematic body-frame gait, no MPC, no world-frame body target.  With an
        elevation map the touch-down heights, the body height / pitch and the
        swing apex over obstacles follow the map (_stable_terrain)."""
        sg = self.stable
        p_ = self.p
        walking = self.mode == "WALK"
        yaw = float(st.rpy[2])
        c, s_ = math.cos(yaw), math.sin(yaw)
        v = np.asarray(st.vel_world, dtype=float)
        v_head = np.array([c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1]])
        sway = None
        if gs.name == "crawl" or (self.gait.pending is not None and self.gait.pending.name == "crawl"):
            sway = self.gait.sway_at([0.0], sg.nominal, gain=float(sg.p.walk_sway),
                                     smooth=float(sg.p.walk_sway_smooth))[0]
            sway[0] *= float(p_["stable_walk_sway_x"])
        feet_meas = self._feet_world_measured(st)
        stance0 = ~sg.swinging if sg.initialized else np.ones(4, dtype=bool)
        z_feet = self._ground_from_feet(feet_meas, stance0) if stance0.any() else self.ground_z
        ground_fn, obstacle_fn, rp_des, foothold_fn = self._stable_terrain(dt, st, emap, z_feet)
        if ground_fn is None:
            self.terrain_pitch = 0.0
            self._update_ground(dt, z_feet)
        cmd = self._stable_heading_hold(st, walking)
        q, info = sg.step(dt, gs, cmd, st.rpy, st.q, walking=walking, v_meas_heading=v_head, sway=sway,
                          ground_fn=ground_fn, obstacle_fn=obstacle_fn, rp_des=rp_des, foothold_fn=foothold_fn)
        stance = np.asarray(info["stance"], dtype=bool)
        # world-frame bookkeeping shared with SIT / E-STOP / ground estimate / debug
        self.foot_w = feet_meas
        self._update_slip(t, dt, st, stance)
        rp = info["rp_cmd"]
        self.body_cmd = (np.asarray(st.pos, dtype=float).copy(), np.array([float(rp[0]), float(rp[1]), yaw]))
        self.body_ik = (self.body_cmd[0].copy(), self.body_cmd[1].copy())
        self._feet_cmd_w = kin.body_to_world(info["feet_b"], st.pos, st.R)
        self.reachable = bool(info["reachable"])
        if self.mode == "STAND":
            self.yaw_hold = yaw
        return q

    def _enter_stand_from_walk(self, t: float, st: RobotState) -> None:
        for leg in range(4):
            if self.swing[leg] is not None:
                self._touchdown(leg, st)
        self.yaw_hold = float(st.rpy[2])
        self.idle_time = 0.0
        self.t_mpc_submit = -1e9
        self._walk_target = None
        self._set_mode("STAND", t)
        if self.pending_sit:
            self.pending_sit = False
            self._enter_sit(t)

    # ------------------------------------------------------------------ slippery ground
    def _set_stance_width(self, extra: float) -> None:
        """Nominal footholds with ``stance_width_offset + extra`` lateral offset."""
        self._stance_extra = float(extra)
        self.nominal_feet_b = kin.nominal_feet_body(
            float(self.p["body_height"]), dx=-float(self.p["body_x_offset"]),
            dy_out=float(self.p["stance_width_offset"]) + self._stance_extra)
        if getattr(self, "stable", None) is not None:
            self.stable.set_nominal(self.nominal_feet_b)

    def _servo_ground_forces(self, st: RobotState) -> Optional[np.ndarray]:
        """Ground forces on the feet (4, 3, world) estimated from the joint
        servo law: the gz JointPositionController (PID force mode) applies
        ``tau = servo_kp (q_target - q)`` (the small D term is neglected) and a
        quasi-static leg transmits ``f = -J^-T tau`` (kinematics.leg_torques).
        None if the gain is unknown or the measurement is broken."""
        kp = float(self.p["servo_kp"])
        q = np.asarray(st.q, dtype=float)
        if kp <= 0.0 or q.shape != (12,) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(self.q_cmd)):
            return None
        tau = (kp * (self.q_cmd - q)).reshape(4, 3)
        R = st.R
        out = np.full((4, 3), np.nan)
        for leg in range(4):
            J = kin.leg_jacobian(q[3 * leg:3 * leg + 3], leg)
            try:
                if abs(float(np.linalg.det(J))) < 1e-5:
                    continue                     # (nearly) straight leg: force not observable
                f_b = np.linalg.solve(J.T, -tau[leg])
            except np.linalg.LinAlgError:
                continue
            out[leg] = R @ f_b
        return out

    def _terrain_friction(self, t: float) -> Tuple[bool, Optional[float]]:
        """(slippery, friction_mu) of a fresh /terrain/info (friction prior)."""
        info = self.terrain_info
        if (not bool(self.p["use_terrain_friction"]) or info is None
                or (t - self.t_terrain) > float(self.p["terrain_info_timeout"])):
            return False, None
        mu = info.get("friction_mu")
        try:
            mu = float(mu) if mu is not None else None
        except (TypeError, ValueError):
            mu = None
        if mu is not None and not (math.isfinite(mu) and mu > 0.0):
            mu = None
        return bool(info.get("slippery", False)), mu

    def _update_slip(self, t: float, dt: float, st: RobotState, stance: np.ndarray) -> None:
        """Slip detection + friction prior -> slip mode, MPC friction
        coefficient and torque feed-forward gain (every control step).

        Slip mode is on while the slip detector's flag is raised (sustained
        stance-foot sliding under a low force ratio, released after
        ``slip_release_time`` and ``slip_release_distance`` without a slip
        vote) or /terrain/info reports ``slippery`` (layout friction zone
        under / ahead of the robot).  In slip mode (MuJoCo physics check on the
        mu 0.15 plate, tools/physics_check.py):

        * speed limits ``slip_max_*`` reached with ``slip_accel_scale`` x the
          normal accelerations (trot 0.4 m/s: falls on the plate; 0.2-0.25 m/s:
          crosses);
        * the MPC torque feed-forward ``slip_torque_feedforward`` (ramped over
          ``slip_ff_ramp_time``): without it the load-dependent sag of the PD
          servos moves the stance feet horizontally whenever the load shifts
          between legs, which the mu 0.15 plate cannot resist (slow trot without
          feed-forward fell, with it 3/3 seeds crossed);
        * a short body-target leash ``slip_max_body_lead`` (trot): the lead of
          the integrated body target over the measured base is what the stiff
          stance legs push against, a propulsive force demand far above
          mu m g = 22 N for the default 0.10 m; after an online detection the
          latched stance-foot targets are also re-synchronised with the
          measured (slid) feet (``slip_resync_feet``) - without both a trot
          that had started sliding at 0.4 m/s kept sliding and fell on 6/6
          seeds, with both 4/4 crossed (2/3 with the detector's force-ratio
          gate, which delays the detection slightly);
        * the MPC friction pyramid ``mu_mpc`` = min(slip estimate, prior) x
          ``slip_mu_margin``, at least ``slip_mu_min``, so the planned (and fed
          forward) tangential forces stay inside the friction cone;
        * ``slip_gait`` (auto mode): ``keep`` = the terrain's choice (default:
          on a flat plate the classifier recommends trot; the crawl, whose CoM
          sway needs lateral accelerations of 2-3.6 m/s^2 > mu g = 1.5 m/s^2,
          fell on every mu 0.15 seed, with or without the slip mode), ``trot``
          / ``crawl`` force that gait while slippery."""
        p = self.p
        detect = bool(p["slip_detection"])
        tilt = math.radians(float(p["slip_max_tilt_deg"]))
        if max(abs(float(st.rpy[0])), abs(float(st.rpy[1]))) > tilt:
            # a tipping robot drags its feet: that says nothing about the ground
            # (MuJoCo check: obstacle-course falls were "detected" as slips)
            stance = np.zeros(4, dtype=bool)
        if detect:
            feet = self._feet_world_measured(st) if np.any(stance) else np.zeros((4, 3))
            forces = self._servo_ground_forces(st) if np.any(stance) else None
            sst = self.slip.update(t, feet, stance, forces, body_xy=st.pos[:2])
        else:
            sst = self.slip.state
        detected = detect and sst.slip
        terrain_slip, mu_prior = self._terrain_friction(t)
        self.friction_prior = mu_prior if terrain_slip else None
        mode = bool(detected or terrain_slip)
        source = "both" if detected and terrain_slip else ("detected" if detected else
                                                            ("terrain" if terrain_slip else "none"))
        x, y = float(st.pos[0]), float(st.pos[1])
        if detect and sst.detected:
            if bool(p["slip_resync_feet"]) and self.mode == "WALK":
                # the stance feet have slid away from their latched targets: the
                # IK would drag them back (more sliding); restart from where they are
                meas = self._feet_world_measured(st)
                for i in range(4):
                    if stance[i]:
                        self.foot_w[i, :2] = meas[i, :2]
                self._walk_target = None         # body target re-anchored (leash below)
            self.slip_events += 1
            self.events.append({"event": "slip_detected", "t": float(t), "x": x, "y": y,
                                "speed": float(sst.speed), "mu_est": float(sst.mu_est)})
            self.log(f"slip detected at ({x:.2f}, {y:.2f}): foot speed {sst.speed:.2f} m/s, "
                     f"mu estimate {sst.mu_est:.2f}")
        if mode != self.slip_mode:
            self.events.append({"event": "slip_mode_on" if mode else "slip_mode_off", "t": float(t),
                                "x": x, "y": y, "source": source if mode else self.slip_source})
            self.log(f"slip mode {'on' if mode else 'off'} ({source if mode else self.slip_source})")
            extra = float(p["slip_stance_width_offset"]) if mode else 0.0
            if extra != self._stance_extra:
                self._set_stance_width(extra)
        self.slip_mode = mode
        self.slip_source = source
        # MPC friction coefficient: slip estimate (recovers slowly after the release) and prior
        mu_nom = float(p["mpc_mu"])
        mu = float(sst.mu_est) if detect else mu_nom
        if terrain_slip and mu_prior is not None:
            mu = min(mu, float(p["slip_mu_margin"]) * mu_prior)
        self.mu_mpc = float(np.clip(mu, min(float(p["slip_mu_min"]), mu_nom), mu_nom))
        # torque feed-forward gain, ramped (a step of tau / kp would jolt the stance legs)
        target = float(p["slip_torque_feedforward"]) if mode else float(p["torque_feedforward"])
        rate = dt / max(float(p["slip_ff_ramp_time"]), 1e-3)
        self.ff_gain += float(np.clip(target - self.ff_gain, -rate, rate))

    def _nominal_world_xy(self, st: RobotState) -> np.ndarray:
        c, s = math.cos(st.rpy[2]), math.sin(st.rpy[2])
        nb = self.nominal_feet_b[:, :2]
        return np.stack([st.pos[0] + c * nb[:, 0] - s * nb[:, 1], st.pos[1] + s * nb[:, 0] + c * nb[:, 1]], axis=1)

    def _terrain_pitch(self, terr: np.ndarray, feet_w: np.ndarray) -> float:
        """Body pitch that follows the terrain (front vs rear support height).

        Unknown map cells (NaN) are filled with the filtered ground estimate,
        never with the measured foot heights: right after spawn the forward
        looking depth camera has not seen the cells under the body yet, and a
        swing foot in the air then made the pitch target wobble by 5-15 deg
        until the robot fell (found in the Gazebo plant-world check)."""
        g = float(self.ground_z)
        zf = np.where(np.isfinite(terr[:2]), terr[:2], g)
        zr = np.where(np.isfinite(terr[2:]), terr[2:], g)
        span = 2.0 * rm.HIP_X_OFFSET
        pitch = -math.atan2(float(np.mean(zf) - np.mean(zr)), span) * float(self.p["terrain_pitch_gain"])
        m = float(self.p["max_body_pitch"])
        return float(np.clip(pitch, -m, m))

    # ------------------------------------------------------------------ swing legs
    def _swing_target(self, leg: int, t: float, t_td_abs: float, st: RobotState, emap) -> np.ndarray:
        """Landing point for a swing that touches down at t_td_abs."""
        res = self.mpc_footholds
        if res is not None and (t - res.t) <= float(self.p["mpc_max_age"]):
            best = None
            for (lg, ttd, x, y, z) in res.foothold_events:
                if lg != leg:
                    continue
                err = abs(res.t + ttd - t_td_abs)
                if err < 0.12 and (best is None or err < best[0]):
                    best = (err, np.array([x, y, z]))
            if best is not None:
                return best[1]
        return self._raibert_local(leg, t_td_abs - t, st, emap)

    def _raibert_local(self, leg: int, t_to_td: float, st: RobotState, emap) -> np.ndarray:
        """Fallback foothold (no MPC result): Raibert + capture point from the state."""
        cfg = self.mpc.cfg
        pat = self.gait.pattern_at(max(t_to_td, 0.0))
        t_st = pat.stance_duration if not pat.is_stand else 0.3
        vx, vy, wz = self.cmd_f
        yaw = float(st.rpy[2]) + wz * (t_to_td + 0.5 * t_st)
        c, s = math.cos(yaw), math.sin(yaw)
        v_cmd_w = np.array([c * vx - s * vy, s * vx + c * vy])
        t_mid = max(t_to_td, 0.0) + 0.5 * t_st
        nb = self.nominal_feet_b[leg, :2]
        h = float(self.p["body_height"])
        v_err = np.asarray(st.vel_world[:2], dtype=float) - v_cmd_w
        # hip predicted with the returning velocity error (same model as ConvexMPC._raibert_batch)
        tau = max(cfg.feedback_tau, 1e-3)
        drift = max(t_to_td, 0.0) * math.exp(-max(t_to_td, 0.0) / tau)
        hip = st.pos[:2] + v_cmd_w * t_mid + drift * v_err + np.array([c * nb[0] - s * nb[1], s * nb[0] + c * nb[1]])
        off = (0.5 * t_st + cfg.capture_gain * math.sqrt(h / cfg.gravity)) * v_err
        ox = float(np.clip(c * off[0] + s * off[1], -cfg.max_foot_dx, cfg.max_foot_dx))
        oy = float(np.clip(-s * off[0] + c * off[1], -cfg.max_foot_dy, cfg.max_foot_dy))
        xy = hip + np.array([c * ox - s * oy, s * ox + c * oy])
        z = self.ground_z
        if emap is not None:
            if self.planner is not None:
                try:
                    pt, _score, valid = self.planner.select(emap, xy, self.ground_z)
                    if valid:
                        xy = np.asarray(pt[:2], dtype=float)
                    if math.isfinite(float(pt[2])):
                        z = float(pt[2])
                except Exception:
                    pass
            else:
                hz = _height_of(emap, xy[0], xy[1])
                if math.isfinite(hz):
                    z = hz
        if abs(z - self.ground_z) > 0.35:
            z = self.ground_z
        return np.array([xy[0], xy[1], z + rm.FOOT_RADIUS])

    def _apex(self, p0: np.ndarray, p1: np.ndarray, emap) -> Tuple[float, float]:
        """Swing apex height + lift fraction (obstacle-aware)."""
        p = self.p
        step_h = float(p["crawl_step_height"] if self.gait.name == "crawl" else p["step_height"])
        obst = float("nan")
        if emap is not None:
            try:
                if self._max_height_along is not None:
                    obst = float(self._max_height_along(emap, p0[:2], p1[:2]))
                else:
                    obst = _max_height_along_fallback(emap, p0, p1)
            except Exception:
                obst = _max_height_along_fallback(emap, p0, p1)
        if math.isfinite(obst) and obst > max(p0[2], p1[2]) + 0.35:
            obst = float("nan")                  # implausible (wall / map artefact)
        apex = swing_apex(p0, p1, step_h, obst, float(p["swing_clearance"]), float(p["swing_max_lift"]))
        nominal = max(p0[2], p1[2]) + step_h
        lift = float(p["swing_lift_fraction"]) if apex > nominal + 0.01 else 0.0
        return apex, lift

    def _liftoff(self, leg: int, t: float, gs, st: RobotState, emap) -> None:
        p0 = self.foot_w[leg].copy()
        s_now = float(gs.swing_progress[leg])
        t_td = t + (1.0 - s_now) * gs.swing_duration
        p1 = self._swing_target(leg, t, t_td, st, emap)
        apex, lift = self._apex(p0, p1, emap)
        self.swing[leg] = SwingTrajectory(p0, p1, apex, gs.swing_duration, lift)
        self.swing_t0[leg] = t - s_now * gs.swing_duration

    def _retarget(self, leg: int, s: float, t: float, emap) -> None:
        """Follow updated MPC footholds during the first part of the swing.

        Trot (``swing_retarget_until``): the capture-point footholds must follow
        a disturbance (push) within the 0.2 s swing.  Crawl
        (``crawl_retarget_until``, default 0 = never): the gait is quasi-static
        and its long stance (0.96 s) makes the Raibert / capture feedback
        ``(T_st / 2 + k sqrt(h/g)) v_err`` ~0.6 s x v_err, so the stride-
        frequency velocity ripple of a compliant joint servo moved the landing
        point by several cm per MPC update late in the 0.24 s swing; the foot
        could not follow and touched down up to 13 cm from the commanded point
        (MuJoCo physics check: every crawl run tipped over; without
        re-targeting 3/3 seeds walk at 0.19 m/s for a 0.2 m/s command)."""
        tr = self.swing[leg]
        key = "crawl_retarget_until" if self.gait.name == "crawl" else "swing_retarget_until"
        if tr is None or s > float(self.p[key]):
            return
        res = self.mpc_footholds
        if res is None or self._retarget_stamp.get(leg) == res.t:
            return
        if (t - res.t) > float(self.p["mpc_max_age"]):
            return                               # stale plan (worker stalled): keep the current target
        t_td = self.swing_t0[leg] + tr.duration
        best = None
        for (lg, ttd, x, y, z) in res.foothold_events:
            if lg == leg:
                err = abs(res.t + ttd - t_td)
                if err < 0.12 and (best is None or err < best[0]):
                    best = (err, np.array([x, y, z]))
        self._retarget_stamp[leg] = res.t
        if best is None:
            return
        new = best[1]
        d = new - tr.p1
        n = float(np.linalg.norm(d[:2]))
        step = float(self.p["max_retarget_step"])
        if n > step:
            d_lim = d.copy()
            d_lim[:2] *= step / n
            # The rate-limited point between two planned footholds was never checked
            # by the foothold planner: when the plan jumps across an obstacle (pipe
            # front <-> back) it lies on the obstacle, at the floor height of the new
            # target (Gazebo: feet landed on the pipe flank, compressed by 3-8 cm).
            # Next to higher terrain the jump is taken whole early in the swing (the
            # commanded foot moves by <= h(u) x jump, a few cm at s 0.3); later the
            # current, planner-checked target is kept (Gazebo: a whole 18 cm jump at
            # s 0.5 left the foot behind its command, on the pipe).
            mid = tr.p1 + d_lim
            top = _max_height_near(emap, mid[0], mid[1], float(self.p["retarget_check_radius"]))
            if not (math.isfinite(top) and top + rm.FOOT_RADIUS > max(new[2], tr.p1[2]) + 0.02):
                d = d_lim
            elif s > float(self.p["retarget_jump_until"]):
                return
        p1 = tr.p1 + d
        apex, lift = self._apex(tr.p0, p1, emap)
        tr.retarget(p1, apex)
        tr.lam = max(tr.lam, lift)

    def _mpc_stance_feet(self, dt: float, st: RobotState, stance: np.ndarray, emap) -> None:
        """World anchors of the MPC mode's stance feet (every WALK step).

        * force limit (``mpc_stance_max_error``, stable mode: stance_max_error):
          the anchor never sits more than e_max horizontally from the measured
          foot.  The servo (~4.5 kN/m at the foot) turns the anchor error into
          a horizontal ground force; a trot stance leg carries ~75 N, i.e. a
          friction limit of ~65 N (mu 0.9) or ~2 cm.  A foot that slid (pipe
          contact, push, turn) used to keep its old anchor, so the leg kept
          dragging it back with several hundred N: the sliding went on, the
          diagonal pair slid unevenly and the body spun up (Gazebo: anchor
          error p95 > 7 cm in every failed MPC run, <= 3.3 cm in the passing
          ones; yaw rate errors of 2-10 rad/s, then the fall).  The slip
          detector's re-synchronisation does not help there: it ignores legs
          with a high force ratio and tilted bodies on purpose.
        * height: a foot that came down on a pipe or its flank (planner choice
          changed during the swing, swing landing early / late) kept the
          planned floor height: the leg was compressed by 3-8 cm (150-350 N),
          the stance torques saturated and the body pitched over within 50 ms
          (Gazebo).  With a map, a stance foot more than ``mpc_contact_lift``
          above its target that sits at the top of mapped terrain next to it
          (within ``mpc_contact_map_tol``, window ``swing_guard_radius``) rests
          on that obstacle - the servo pushes it down with >= 90 N - and its
          target becomes the measured height.  A foot still coming down (late
          swing) is above the mapped top and is not lifted.  A foot-speed test
          alone ("resting" = barely moving) also lifted the targets of stance
          feet that a tipping body had lifted off flat ground (push: up to
          0.38 m, fell), and the odometry height lags while a foot on a pipe
          shoves the body up.  The anchor height is also lowered to the map
          height under the foot (``mpc_land_tau``): a foot planned onto an
          obstacle that came down beside it would otherwise hang above the floor.  Raising the
          target to the map height instead (as the stable mode's
          stable_land_tau does) lifted feet next to the wall, where the depth
          map shows the wall's edge as 5-15 cm high terrain."""
        meas = self._feet_world_measured(st)
        if not np.all(np.isfinite(meas)):
            meas = None
        try:
            if np.any(stance):
                self._stance_feet_update(dt, stance, emap, meas)
        finally:
            self._feet_meas_now = meas

    def _stance_feet_update(self, dt: float, stance: np.ndarray, emap, meas) -> None:
        p = self.p
        emax = float(p["mpc_stance_max_error"])
        if emax > 0.0 and meas is not None:
            for leg in np.flatnonzero(stance):
                d = self.foot_w[leg, :2] - meas[leg, :2]
                n = float(np.hypot(d[0], d[1]))
                if n > emax:
                    self.foot_w[leg, :2] = meas[leg, :2] + d * (emax / n)
        tau = float(p["mpc_land_tau"])
        if tau > 0.0 and emap is not None and dt > 0.0:
            a = min(1.0, dt / tau)
            for leg in np.flatnonzero(stance):
                z = _height_of(emap, self.foot_w[leg, 0], self.foot_w[leg, 1])
                if math.isfinite(z) and abs(z - self.ground_z) < 0.35:
                    z += rm.FOOT_RADIUS
                    if z < self.foot_w[leg, 2]:
                        self.foot_w[leg, 2] += a * (z - self.foot_w[leg, 2])
        lift = float(p["mpc_contact_lift"])
        tol = float(p["mpc_contact_map_tol"])
        if lift > 0.0 and tol > 0.0 and emap is not None and meas is not None:
            for leg in np.flatnonzero(stance):
                zm = float(meas[leg, 2])
                if not lift < zm - float(self.foot_w[leg, 2]) < 0.35:
                    continue
                # resting on mapped terrain: the foot sits at the top of terrain next to
                # it that is higher than its target.  A foot speed test is not enough: a
                # stance foot lifted by a tipping body (push) pauses at the top as well,
                # and its target ratcheted up 0.38 m in Gazebo (fell)
                top = _max_height_near(emap, meas[leg, 0], meas[leg, 1], float(p["swing_guard_radius"]))
                if (math.isfinite(top) and abs(zm - top - rm.FOOT_RADIUS) < tol
                        and top + rm.FOOT_RADIUS > float(self.foot_w[leg, 2]) + lift):
                    self.foot_w[leg, 2] = zm

    def _swing_guard_z(self, leg: int, s: float, tr: SwingTrajectory, z_cmd: float, emap) -> float:
        """Swing foot height command with an obstacle guard (descent half).

        The servo lags the swing command by ~30 ms; a step over a pipe moves
        the foot ~0.3 m in ~0.12 s (lift-then-move), so the measured foot was
        still above the pipe when the commanded descent began, and the leg was
        driven into the pipe's flank with 300-450 N (Gazebo plant world: the
        body reared up / was shoved sideways and fell).  While the MEASURED
        foot is over or next to terrain higher than the landing point
        (``swing_guard_radius``), the commanded height stays
        ``swing_guard_clearance`` above that terrain (never above the apex)."""
        p = self.p
        clr = float(p["swing_guard_clearance"])
        meas = self._feet_meas_now
        if clr <= 0.0 or emap is None or s < 0.5 or meas is None:
            return z_cmd
        top = _max_height_near(emap, meas[leg, 0], meas[leg, 1], float(p["swing_guard_radius"]))
        if not math.isfinite(top) or abs(top - self.ground_z) > 0.35:
            return z_cmd
        z_min = top + rm.FOOT_RADIUS + clr
        if top + rm.FOOT_RADIUS <= float(tr.p1[2]) + 0.02 or z_min <= z_cmd:
            return z_cmd                         # nothing higher than the landing point / already above
        return float(min(z_min, max(float(tr.apex_z), z_cmd)))

    def _stance_deflection_limit(self, st: RobotState, stance: np.ndarray, pos_cmd, rpy_cmd,
                                 feet: np.ndarray) -> np.ndarray:
        """Bound the horizontal servo deflection of every stance leg (``mpc_stance_rel_error``, 0 = off).

        The horizontal ground force of a position-servoed stance leg is its
        stiffness (~3 kN/m forward, ~4.5 kN/m sideways) times the deflection
        commanded foot (in the body-target frame) - measured foot (leg FK),
        beyond the static sag under its weight share.  That deflection is the
        anchor error (``mpc_stance_max_error``, 2 cm) PLUS the body target's
        lead over the measured body (1.5 cm sideways, 0.05 rad ~ 1.3 cm of
        yaw, up to 6 cm forward): up to ~5 cm, ~220 N - beyond friction.
        The stable mode bounds the total per foot (stance_max_error 1.5 cm,
        gravity feed-forward removes its sag).  Here the deflection minus the
        static sag ``C (0, 0, -m g / n)`` (C = J J^T / kp, the servo's foot
        compliance) is limited to e_max horizontally; the vertical part is
        left untouched."""
        e = float(self.p["mpc_stance_rel_error"])
        kp = float(self.p["servo_kp"])
        if e <= 0.0 or kp <= 0.0 or not np.any(stance):
            return feet
        q = np.asarray(st.q, dtype=float).reshape(4, 3)
        rpy = np.asarray(st.rpy, dtype=float)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(rpy[:2]))):
            return feet
        R_cmd = euler_to_rot(float(rpy_cmd[0]), float(rpy_cmd[1]), float(rpy_cmd[2]))
        R_rp = euler_to_rot(float(rpy[0]), float(rpy[1]), 0.0)
        f_static = R_rp.T @ np.array([0.0, 0.0, -float(self.p["mpc_mass"]) * self.mpc.cfg.gravity / stance.sum()])
        pos_cmd = np.asarray(pos_cmd, dtype=float)
        out = feet.copy()
        for leg in np.flatnonzero(stance):
            p_cmd = R_cmd.T @ (out[leg] - pos_cmd)
            p_meas = kin.leg_fk(q[leg], leg)
            J = kin.leg_jacobian(q[leg], leg)
            p_ref = p_meas + (J @ J.T / kp) @ f_static      # measured foot + static sag of its weight share
            d = p_cmd - p_ref
            n = float(np.hypot(d[0], d[1]))
            if n <= e or not math.isfinite(n):
                continue
            d[:2] *= e / n
            out[leg] = pos_cmd + R_cmd @ (p_ref + d)
        return out

    def _touchdown(self, leg: int, st: RobotState) -> None:
        tr = self.swing[leg]
        if tr is None:
            return
        land = tr.p1.copy()
        if bool(self.p["touchdown_measured_xy"]):
            meas = self._feet_world_measured(st)[leg]
            if np.linalg.norm(meas[:2] - land[:2]) < 0.08:
                land[:2] = meas[:2]
        self.foot_w[leg] = land
        self.swing[leg] = None

    # ------------------------------------------------------------------ MPC
    def _mpc_input(self, t: float, st: RobotState) -> MPCInput:
        feet = self.foot_w.copy()
        stand = self.mode == "STAND"
        hold_xy = None
        hold_yaw = None
        if stand:
            hold_xy = feet[:, :2].mean(axis=0) + self._body_offset_world(st)
            hold_yaw = self.yaw_hold
        cmd = np.zeros(3) if stand else np.asarray(self.cmd_f, dtype=float).copy()
        if not stand:
            cmd[1] += self._turn_pivot_vy()
        # heading hold (as the stable mode): the MPC reference yaw is re-anchored on
        # the measured yaw at every solve, so without it every yaw impulse (pipe
        # contact, push, a slipping foot) is kept for good (Gazebo: 14-41 deg drift
        # over 4 m on the pipe course with no turn commanded, 3 deg with the hold)
        cmd[2] += self._mpc_heading_hold(st)
        return MPCInput(
            t=t, pos=np.asarray(st.pos, dtype=float).copy(), rpy=np.asarray(st.rpy, dtype=float).copy(),
            vel=np.asarray(st.vel_world, dtype=float).copy(), omega=st.omega_world.copy(), feet=feet,
            cmd=tuple(float(v) for v in cmd),
            body_height=float(self.p["body_height"]), ground_z=float(self.ground_z),
            pitch_ref=float(self.terrain_pitch), hold_xy=hold_xy, hold_yaw=hold_yaw,
            nominal_feet_xy=self.nominal_feet_b[:, :2].copy(), mu=float(self.mu_mpc))

    def _turn_pivot_vy(self) -> float:
        """Body-frame sideways velocity that moves the in-place turning centre to
        ``mpc_turn_pivot_x`` (MPC mode; faded out over ``mpc_turn_pivot_fade`` of
        commanded translation speed).

        The knees bend backwards: the rear knees sit ~0.43 m behind the base
        origin (outer thigh-box corner ~(-0.44, +-0.16) m), the trunk front only
        0.30 m ahead.  Turning in place about the base origin sweeps the rear
        knees through a ~0.47 m circle; about a point 0.15 m behind it the rear
        knees sweep ~0.33 m, the trunk front corners ~0.46 m and the front feet
        ~0.42 m - the whole footprint stays inside the origin turn's circle,
        while the rear swings ~14 cm less.  Rotating about x_p < 0 moves the base origin
        sideways at w (-x_p) (towards the inside of the turn).  At the C-OBS1
        gap exit (0.31 m from the wall, left turn of ~30 deg in place) the rear
        knee otherwise ran ~5-7 cm into the wall and the MPC mode jammed or
        tipped over there (Gazebo scratch gap-exit runs: 15/27 passed about
        the origin, 22/27 about x_p = -0.15, 19/27 about -0.095; stable 8/9).
        Only a COMMANDED in-place turn moves the centre: the fade uses the
        larger of the raw and the shaped translation command, so a walk that
        the torque-saturation speed scaling slowed down (stuck on a pipe,
        speed scale 0.3) keeps the plain turn (Gazebo: the pivot motion there
        coincided with 2 extra falls on the C-OBS1 pipe in 9 runs of (j)).
        Broad Gazebo check (tools/gazebo_check.py --mode mpc, --mpc-latency
        0.03, depth map; same seeds with mpc_turn_pivot_x 0 as the reference):
        (o) 6/6 vs 2/6, (l) 6/6 vs 0/3 (stable 3/3, 3/3); (j) 3/6 vs 3/6 (its
        falls are on the C-OBS1 pipe and the ramp, the pivot is off there);
        a d e i f k q r s 3/3 in both.  Final check with the measured MPC
        delay (gazebo_check.json 'mpc_final2'): no fall at the gap exit; the
        (o) / (l) / (j) falls left are on the 11.8 deg ramp (MPC-delay
        sensitive) and the pipes, where the pivot is off.
        The stable mode keeps its own (base-origin) turn."""
        xp = float(self.p["mpc_turn_pivot_x"])
        if xp == 0.0:
            return 0.0
        vx, vy, wz = (float(v) for v in self.cmd_f)
        v0 = float(self.p["mpc_turn_pivot_fade"])
        v = max(math.hypot(vx, vy), math.hypot(float(self.cmd_raw[0]), float(self.cmd_raw[1])))
        s = max(0.0, 1.0 - v / v0) if v0 > 0.0 else float(v == 0.0)
        return s * wz * (-xp)

    def _mpc_heading_hold(self, st: RobotState) -> float:
        """Yaw-rate correction of the MPC mode's heading hold (``mpc_heading_hold_gain``
        / ``_max``, see :meth:`_heading_hold`)."""
        return self._heading_hold(st, self.mode == "WALK", float(self.p["mpc_heading_hold_gain"]),
                                  float(self.p["mpc_heading_hold_max"]))

    def _body_offset_world(self, st: RobotState) -> np.ndarray:
        dx = float(self.p["body_x_offset"])
        c, s = math.cos(st.rpy[2]), math.sin(st.rpy[2])
        return np.array([c * dx, s * dx])

    def _mpc_cycle(self, t: float, st: RobotState, gs, emap) -> None:
        if self.worker is not None:
            res = self.worker.poll()
            # a solve that was running when the controller was reset (fall,
            # stand-up, clock jump) belongs to the previous motion: drop it
            if res is not None and self._inflight_gen == self._mpc_gen:
                self._accept_mpc(res)
            err = self.worker.error
            if err and err != self._last_worker_error:
                # an exception inside ConvexMPC.solve: the controller keeps
                # walking on the command-based fallback, but say why
                self.log(f"MPC failed: {err}")
            self._last_worker_error = err
        period = 1.0 / max(float(self.p["mpc_rate"]), 1e-3)
        if t - self.t_mpc_submit < period - 1e-6:
            return
        height_fn = None
        foothold_fn = None
        if emap is not None:
            height_fn = emap.height_at
            if self.planner is not None:
                planner = self.planner

                def foothold_fn(leg, nominal, _emap=emap, _pl=planner):
                    pt, _score, valid = _pl.select(_emap, nominal[:2], nominal[2])
                    return pt, valid
        if not all(np.all(np.isfinite(v)) for v in (st.pos, st.rpy, st.vel_world, st.omega_body)):
            return                               # never plan from (and warm-start with) a broken state
        job = (self._mpc_input(t, st), self.gait.snapshot(), height_fn, foothold_fn)
        if self.worker is not None:
            if self.worker.submit(job):
                self.t_mpc_submit = t
                self._inflight_gen = self._mpc_gen
        else:
            self.t_mpc_submit = t
            try:
                self._accept_mpc(self.mpc.solve(*job))
            except Exception as e:
                self.log(f"MPC failed: {type(e).__name__}: {e}")

    @staticmethod
    def plausible(res: MPCResult, window: float = 0.5) -> bool:
        """Reject numerically broken predictions (e.g. an unconverged QP) before
        they reach the body pose target: within the first ``window`` seconds
        the prediction must be finite, |roll|, |pitch| < 0.6 rad and the height
        within 0.15 m of the reference (stair transients stay well inside)."""
        if res is None or res.x_pred is None or not np.all(np.isfinite(res.x_pred)):
            return False
        k = max(2, int(np.searchsorted(res.times, window)) + 1)
        x = res.x_pred[1:k]                      # index 0 is the measured state
        return bool(np.max(np.abs(x[:, 0:2])) < 0.6 and np.max(np.abs(x[:, 5] - res.x_ref[1:k, 5])) < 0.15)

    def _accept_mpc(self, res: MPCResult) -> None:
        self.mpc_stats.append(res.solve_time)
        if len(self.mpc_stats) > 200:
            self.mpc_stats = self.mpc_stats[-200:]
        # The footholds come from the Raibert / capture-point heuristic and the
        # foothold planner, not from the QP: they stay valid when the QP did not
        # converge.  Keeping the previous result for them as well made every
        # swing after a 15 N s push land at the pre-push footholds (the QP
        # returned solved_inaccurate for ~0.5 s while the body was moving
        # sideways at 0.7 m/s - MuJoCo physics check).
        if _footholds_finite(res):
            self.mpc_footholds = res
        if getattr(res, "x_pred", None) is not None and not self.plausible(res):
            # keep the previous body prediction (it ages out after mpc_max_age)
            self.mpc_rejected += 1
            self.log(f"MPC prediction rejected (status {res.status})")
            return
        self.mpc_result = res
        p = self.p
        if res.ok:
            if res.torque_ratio > float(p["torque_scale_threshold"]):
                self.speed_scale = max(float(p["speed_scale_min"]), self.speed_scale * float(p["speed_scale_down"]))
            elif res.torque_ratio < float(p["torque_recover_threshold"]):
                self.speed_scale = min(1.0, self.speed_scale + float(p["speed_scale_up"]))

    def _body_target(self, t: float, dt: float, st: RobotState):
        """Desired body pose (pos, rpy) at t + lead."""
        p = self.p
        lead = float(p["body_lead_time"])
        h = float(p["body_height"])
        if self.mode == "STAND":
            centre = self.foot_w[:, :2].mean(axis=0) + self._body_offset_world(st)
            z = float(np.mean(self.foot_w[:, 2])) - rm.FOOT_RADIUS + h
            target = np.array([centre[0], centre[1], z])
            rpy_t = np.array([0.0, self.terrain_pitch, self.yaw_hold])
            if self.body_cmd is None:
                self.body_cmd = (np.asarray(st.pos, dtype=float).copy(), np.array([0.0, 0.0, st.rpy[2]]))
            pos0, rpy0 = self.body_cmd
            a = min(1.0, dt / max(float(p["stand_filter_tau"]), 1e-3))
            pos = pos0 + a * (target - pos0)
            dyaw = wrap_angle(rpy_t[2] - rpy0[2])
            rpy = rpy0 + a * np.array([rpy_t[0] - rpy0[0], rpy_t[1] - rpy0[1], dyaw])
            return pos, rpy
        res = self.mpc_result
        fresh = res is not None and res.ok and (t - res.t) <= float(p["mpc_max_age"])
        if fresh and str(p["body_target_mode"]) == "prediction":
            # pose of the MPC prediction itself (re-anchored at every solve on the
            # measured state; only suitable for a stiff, lag-free joint servo)
            x = res.state_at(t - res.t + lead)
            pos = x[3:6].copy()
            rpy = np.array([x[0], x[1], x[2]])
            # MPC yaw is unwrapped relative to the measurement at solve time
            rpy[2] = float(st.rpy[2]) + wrap_angle(rpy[2] - float(st.rpy[2]))
            return pos, rpy
        return self._integrated_body_target(t, dt, st, res if fresh else None)

    def _integrated_body_target(self, t: float, dt: float, st: RobotState, res: Optional[MPCResult]):
        """Body pose target for a position-servoed robot (``integrated`` mode).

        The horizontal position and the heading of the target are internal
        states *integrated* with the MPC-predicted velocity / yaw rate at
        t + lead (the shaped command velocity without a fresh MPC result).
        The target therefore keeps moving at the planned speed when the joint
        servos lag behind it; the lag only shows up as a constant offset
        between target and measured pose.  Using the predicted *pose* instead
        (``prediction`` mode) re-anchors the target on the measured pose at
        every solve, which throws away the lag-induced offset each time: in a
        kinematic test with a first-order servo lag (joints and body), a
        0.5 m/s trot command gave 0.50 / 0.55 / 0.53 / 0.50 m/s at 0 / 30 / 60 /
        100 ms lag in ``integrated`` mode but 0.50 / 0.31 / 0.08 m/s in
        ``prediction`` mode (crawl 0.2 m/s at 60 ms lag: 0.22 vs 0.01 m/s).
        The lead over the measured pose is clamped (``max_body_lead`` /
        ``max_body_lead_lateral`` / ``max_yaw_lead``; ``crawl_*`` while the
        crawl runs) so that a blocked robot does not wind the target up.

        The lead is also a force demand: the stiff stance legs (servo_kp 450
        N m/rad, ~4.5 kN/m at the foot) push the body towards the target, so
        a lead of L metres asks the two trot stance feet for roughly
        2 x 4.5 kN/m x L of horizontal force against a friction limit of
        mu m g ~ 130 N.  Gazebo check (gz-sim 8 + DART, tools/gazebo_check.py):
        with the former trot limits 0.10 m / 0.20 rad a yaw lag of ~12 deg
        during turns demanded ~250 N per foot, the stance feet slid (p95
        stance-foot speed 0.5 m/s in random teleop), the body yawed at up to
        4 rad/s and the robot reared up and fell in 2 of 3 teleop runs (all
        teleop runs with 20 ms bridge latency, 100 ms MPC latency or 30 %
        dropped control ticks).  The heading and sideways leads (0.05 rad,
        0.015 m) matter most; the forward lead 0.06 m keeps enough propulsion
        for the compliant servo (0.03 m: Gazebo fine, but the MuJoCo trot
        slowed to 0.30 m/s and fell on one seed): 3/3 teleop runs, also with
        20 ms latency or 50 ms MPC latency.  The crawl keeps the larger
        limits: its planned CoM sway needs the lead (MuJoCo physics check:
        crawl 0/3 with the trot limits).

        Height and roll / pitch are absolute: the MPC reference at t + lead
        (terrain height + body_height, terrain pitch), held against servo
        compliance by the IMU PD and the height integrator."""
        p = self.p
        lead = float(p["body_lead_time"])
        yaw_m = float(st.rpy[2])
        if res is not None:
            tr = t - res.t + lead
            x = res.state_at(tr)
            xr = res.ref_at(tr)
            v_xy = x[9:11].copy()
            # sideways (reference heading frame) part of the integrated velocity:
            # blended from the prediction towards the reference.  The prediction
            # starts at the measured velocity and corrects it slowly, so the target
            # followed the outward drift of the body in turns (Gazebo plant world:
            # 0.05-0.11 m/s sideways after turns, +-0.02 m/s in the stable mode; the
            # robot entered the 0.63 m wall gap 0.12 m off its line and its feet hit
            # the wall).  Blend 0.5: +-0.03 m/s, gap entered on the line.  The
            # forward part keeps the prediction (a reference-velocity target is a
            # propulsive force demand beyond friction, see max_body_lead).
            a_lat = float(p["mpc_lateral_ref_blend"])
            if a_lat > 0.0:
                psi = float(xr[2])
                n_lat = np.array([-math.sin(psi), math.cos(psi)])
                v_xy = v_xy + a_lat * float(n_lat @ (np.asarray(xr[9:11], dtype=float) - v_xy)) * n_lat
            wz = float(x[8])
            z_t = float(xr[5])
            roll_t, pitch_t = float(xr[0]), float(xr[1])
        else:
            vx, vy, wz = (float(v) for v in self.cmd_f)
            c, s = math.cos(yaw_m), math.sin(yaw_m)
            v_xy = np.array([c * vx - s * vy, s * vx + c * vy])
            z_t = self.ground_z + float(p["body_height"])
            roll_t, pitch_t = 0.0, self.terrain_pitch
        if self._walk_target is None:
            if self.body_cmd is not None:
                xy0 = np.asarray(self.body_cmd[0][:2], dtype=float).copy()
                yaw0 = yaw_m + wrap_angle(float(self.body_cmd[1][2]) - yaw_m)
            else:
                xy0, yaw0 = np.asarray(st.pos[:2], dtype=float).copy(), yaw_m
            self._walk_target = (xy0, yaw0)
        xy, yaw = self._walk_target
        xy = xy + v_xy * dt
        yaw = yaw + wz * dt
        # leash: bounded lead over the measured pose (+ the planned lead)
        anchor = np.asarray(st.pos[:2], dtype=float) + v_xy * lead
        d = xy - anchor
        # sideways (heading frame) the target may only lead by max_body_lead_lateral:
        # with two diagonal feet on the ground the body can fall sideways about the
        # support line, which no joint target can prevent; a target left behind by
        # such a motion (a push) is enforced as soon as the next pair touches down -
        # an impulse that throws the body back (MuJoCo physics check: 15 N s push,
        # target 8 cm beside the body -> 0.55 m/s sideways slam -> fall)
        ch, sh = math.cos(yaw), math.sin(yaw)
        d_lat = -sh * d[0] + ch * d[1]
        crawl = self.gait.name == "crawl"
        lat_max = float(p["crawl_max_body_lead_lateral"] if crawl else p["max_body_lead_lateral"])
        if abs(d_lat) > lat_max:
            corr = d_lat - math.copysign(lat_max, d_lat)
            d = d - corr * np.array([-sh, ch])
        n = float(np.linalg.norm(d))
        # slip mode: the lead is the propulsive force demand on the stance feet
        # (servo stiffness x lead), keep it small on low friction (see _update_slip);
        # not in the crawl, whose CoM sway needs the lead (physics check: the clipped
        # sway tipped the crawl over already on the normal floor)
        tight = self.slip_mode and not crawl
        if tight:
            max_lead = min(float(p["slip_max_body_lead"]), float(p["max_body_lead"]))
        else:
            max_lead = float(p["crawl_max_body_lead"] if crawl else p["max_body_lead"])
        if n > max_lead:
            d = d * (max_lead / n)
        xy = anchor + d
        yaw_anchor = yaw_m + wz * lead
        yaw_max = float(p["crawl_max_yaw_lead"] if crawl else p["max_yaw_lead"])
        dyaw = float(np.clip(wrap_angle(yaw - yaw_anchor), -yaw_max, yaw_max))
        yaw = yaw_anchor + dyaw
        self._walk_target = (xy.copy(), yaw)
        return np.array([xy[0], xy[1], z_t]), np.array([roll_t, pitch_t, yaw])

    # ------------------------------------------------------------------ IK
    def _ik_from_body(self, pos_cmd, rpy_cmd, feet_w) -> np.ndarray:
        feet_b = kin.world_to_body(feet_w, pos_cmd, rpy_cmd)
        q, ok = kin.ik_all(feet_b, reach_margin=0.002)
        self.reachable = bool(np.all(ok))
        return q

    # ------------------------------------------------------------------ debug / state
    def _debug(self, t: float, st: RobotState, body_h: float) -> Dict[str, Any]:
        gs = self.gait.current()
        contacts = [bool(c) and self.swing[i] is None for i, c in enumerate(gs.contacts)]
        if self.mode not in ("STAND", "WALK", "SIT"):
            contacts = [False] * 4 if self.mode in ("FALLEN", "RECOVERING") else [True] * 4
        res = self.mpc_result
        feet = self._feet_cmd_w if self.mode == "WALK" else self.foot_w
        margin = support_margin(feet[:, :2], contacts, st.pos[:2]) if any(contacts) else float("nan")
        info = self.terrain_info if (self.terrain_info is not None and
                                     (t - self.t_terrain) <= float(self.p["terrain_info_timeout"])) else None
        torque_max = res.torque_horizon_max if res is not None else np.zeros(12)
        return {
            "mode": self.mode,
            "locomotion_mode": self.loco_mode,
            "locomotion_mode_pending": self._loco_mode_pending,
            "gait": gs.name,
            "gait_auto": self.gait_mode == "auto",
            "gait_mode": self.gait_mode,
            "gait_target": self.gait_auto_choice if self.gait_mode == "auto" else self.gait_mode,
            "phase": float(gs.phase),
            "contacts": contacts,
            "body_height": float(body_h),
            "roll": float(st.rpy[0]),
            "pitch": float(st.rpy[1]),
            "yaw": float(st.rpy[2]),
            "fall_count": int(self.recovery.fall_count),
            "recovery_attempts": int(self.recovery.attempts_total),
            "recovery_success": int(self.recovery.successes),
            "assisted_resets": int(self.recovery.assisted_resets),
            "recovery_state": self.recovery.state,
            "mpc_solve_ms": float(res.solve_time * 1e3) if res is not None else float("nan"),
            "mpc_solve_ms_mean": float(np.mean(self.mpc_stats) * 1e3) if self.mpc_stats else float("nan"),
            "mpc_status": res.status if res is not None else "none",
            "mpc_rejected": int(self.mpc_rejected),
            "mpc_age": float(t - res.t) if res is not None else float("nan"),
            "mpc_horizon_s": float(res.horizon) if res is not None else 0.0,
            "stability_margin": float(margin),
            "terrain": info.get("class", "unknown") if info else "unknown",
            "cmd": [float(v) for v in self.cmd_f],
            "cmd_raw": [float(v) for v in self.cmd_raw],
            "speed_scale": float(self.speed_scale),
            "torque_ratio": float(res.torque_ratio) if res is not None else 0.0,
            "torque_horizon_max": [float(v) for v in torque_max],
            "ground_z": float(self.ground_z),
            "terrain_pitch": float(self.terrain_pitch),
            "ik_reachable": bool(self.reachable),
            # slippery ground
            "slip": bool(self.slip.slip),
            "slip_mode": bool(self.slip_mode),
            "slip_source": self.slip_source,
            "slip_speed": float(self.slip.state.speed),
            "slip_legs": [bool(v) for v in self.slip.state.legs],
            "slip_events": int(self.slip_events),
            "mu_est": float(self.slip.mu_est),
            "mpc_mu": float(self.mu_mpc),
            "friction_prior": self.friction_prior,
            "torque_ff": float(self.ff_gain),
        }
