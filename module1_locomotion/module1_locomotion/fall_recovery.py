"""
Fall detection and self-righting state machine (ROS free).

States (DESIGN_SPEC §5)::

    NORMAL --(|roll| or |pitch| > 55 deg, or body height < 0.12 m, for 0.3 s)--> FALL_DETECTED
    FALL_DETECTED -> DAMP -> TUCK -> [ROCK] -> SELF_RIGHT -> PREPARE -> STAND_UP -> NORMAL
                                    (any failure)  -> next attempt (DAMP) ... 3 attempts
                                    -> FAILED  (or ASSISTED_RESET via gz set_pose, counted separately)
    FAILED --(upright for 0.3 s, e.g. held up by the stand posture of a failed assisted reset)--> NORMAL
           --(failed_retry_time > 0: after that long)--> DAMP (new series of attempts)

Joint-space keyframes are designed for this leg geometry (hip_x +-0.70 rad,
hip_y -1.6..2.6 rad, knee -2.8..-0.2 rad, thigh 0.226 m, shank 0.300 m):

* DAMP: the servos follow the measured joints (no fighting the impact) until
  the body has come to rest.
* TUCK: all legs folded (hip_y 1.25, knee -2.70: foot 0.11 m below and 0.08 m
  in front of the hip) so the body can roll freely.
* ROCK (only when lying on the back, |roll| > 120 deg): the legs swing
  sideways in phase (hip_x +-0.7) with growing amplitude to tip the body onto
  a side.
* SELF_RIGHT: lying on a side, the legs on the *ground side* are swung into
  the leg plane towards the robot's top side (front leg forward-up
  hip_y -1.5 / knee -1.5, rear leg backward-up hip_y 2.4 / knee -0.6; the leg
  plane is horizontal when the robot lies on its side) and then abducted
  (hip_x 0.6 towards the ground).  The feet press on the ground on the *top*
  side of the CoM, so the ground reaction produces a moment that rolls the
  body back onto its belly (pressing on the belly side would roll it further
  over).  The high-side legs stay tucked.
* PREPARE: on the belly, feet under the hips with deeply bent knees.
* STAND_UP: smooth (smoothstep) joint interpolation to the stand posture.
* Verification: upright (|roll|, |pitch| < 25 deg, height > 0.22 m) for 0.3 s
  -> NORMAL ("recovered" event); otherwise the attempt failed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from dusan_common import robot_model as rm

from . import kinematics as kin

NORMAL = "NORMAL"
FALL_DETECTED = "FALL_DETECTED"
DAMP = "DAMP"
TUCK = "TUCK"
ROCK = "ROCK"
SELF_RIGHT = "SELF_RIGHT"
PREPARE = "PREPARE"
STAND_UP = "STAND_UP"
VERIFY = "VERIFY"
FAILED = "FAILED"
ASSISTED_RESET = "ASSISTED_RESET"

FALLEN_STATES = (FALL_DETECTED, DAMP, FAILED)
RECOVERING_STATES = (TUCK, ROCK, SELF_RIGHT, PREPARE, STAND_UP, VERIFY, ASSISTED_RESET)

DEG = math.pi / 180.0


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _legs(*per_leg) -> np.ndarray:
    """Assemble a (12,) joint vector from four (hip_x, hip_y, knee) tuples."""
    return np.concatenate([np.asarray(p, dtype=float) for p in per_leg])


@dataclass
class FallParams:
    fall_angle_deg: float = 55.0          # |roll| or |pitch| beyond -> fall candidate
    fall_angle_exit_deg: float = 50.0     # hysteresis: candidate persists until below this
    fall_height: float = 0.12             # body height above ground [m]
    fall_height_exit: float = 0.14
    detect_time: float = 0.3              # condition must persist [s]
    damp_min_time: float = 0.5
    damp_max_time: float = 2.0
    damp_rest_rate: float = 1.0           # |omega| [rad/s] considered at rest
    tuck_time: float = 0.8
    rock_time: float = 3.0
    rock_period: float = 0.8
    self_right_reach_time: float = 0.7
    self_right_push_time: float = 0.6
    self_right_timeout: float = 3.0
    self_right_done_deg: float = 35.0     # |roll| below -> on the belly
    self_right_skip_deg: float = 35.0     # already on the belly -> skip SELF_RIGHT
    upside_down_deg: float = 120.0
    prepare_time: float = 1.0
    stand_up_time: float = 2.5
    verify_time: float = 0.3
    verify_timeout: float = 1.5
    upright_deg: float = 25.0
    upright_height: float = 0.22
    max_attempts: int = 3
    assisted_reset: bool = True
    reset_settle_time: float = 2.0
    reset_height_offset: float = 0.5
    reset_confirm_timeout: float = 8.0    # longest wait for confirm_reset() [s] (2 x gz_timeout + margin)
    failed_retry_time: float = 0.0        # FAILED: start a new series of attempts after this long [s] (0 = stay)
    body_height: float = rm.NOMINAL_BODY_HEIGHT

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "FallParams":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))
        return p


class FallRecovery:
    """Fall detector + recovery sequencer.  ``update`` returns joint targets.

    reset_fn(x, y, z, yaw) is called for the assisted reset (e.g.
    ``GzTools.reset_pose``); it must not block.  If it returns ``True`` the
    caller promises to report the completion of the teleport with
    :meth:`confirm_reset`: ASSISTED_RESET then holds the stand posture until
    the confirmation (at most ``reset_confirm_timeout``) and only then starts
    the ``reset_settle_time`` window - the gz CLI needs ~0.3-1 s before the
    teleport happens, so a window started at the request could end before the
    robot was even moved.  Any other return value (None / False): the window
    starts at the request.
    """

    # keyframes (hip_x, hip_y, knee)
    TUCK_LEG = (0.0, 1.25, -2.70)
    PREPARE_LEG = (0.0, 1.20, -2.55)
    REACH_FRONT = (-1.50, -1.50)          # (hip_y, knee): foot forward and towards the top side
    REACH_REAR = (2.40, -0.60)            # foot backward and towards the top side
    PUSH_HIP_X = 0.60

    def __init__(self, params: Optional[FallParams] = None,
                 reset_fn: Optional[Callable[[float, float, float, float], None]] = None):
        self.p = params if params is not None else FallParams()
        self.reset_fn = reset_fn
        self.state = NORMAL
        self.events: List[dict] = []
        self.fall_count = 0
        self.attempts_total = 0
        self.successes = 0
        self.assisted_resets = 0
        self.attempt = 0
        self._cand_since: Optional[float] = None
        self._t_state = 0.0
        self._q_from = np.zeros(12)
        self._q_last = np.zeros(12)
        self._roll_dir = 1.0
        self._verify_since: Optional[float] = None
        self._reset_done = False
        self._reset_wait = False              # ASSISTED_RESET waiting for confirm_reset()
        self.q_stand = kin.stand_joint_angles(self.p.body_height)

    # ------------------------------------------------------------------ helpers
    @property
    def active(self) -> bool:
        return self.state != NORMAL

    @property
    def fallen(self) -> bool:
        return self.state in FALLEN_STATES

    @property
    def recovering(self) -> bool:
        return self.state in RECOVERING_STATES

    @property
    def waiting_for_reset(self) -> bool:
        """True while ASSISTED_RESET waits for the teleport (confirm_reset)."""
        return self.state == ASSISTED_RESET and self._reset_wait

    def pop_events(self) -> List[dict]:
        ev, self.events = self.events, []
        return ev

    def _event(self, name: str, t: float, pose) -> None:
        x, y, roll, pitch = pose
        self.events.append({"event": name, "t": float(t), "x": float(x), "y": float(y),
                            "roll": float(roll), "pitch": float(pitch)})

    def _enter(self, state: str, t: float, q_meas) -> None:
        self.state = state
        self._t_state = t
        self._verify_since = None
        self._q_from = np.asarray(q_meas if q_meas is not None else self._q_last, dtype=float).copy()

    def is_upright(self, roll: float, pitch: float, body_height: float) -> bool:
        """Standing posture of the verification (|roll|, |pitch| < upright_deg,
        body height > upright_height; an unknown height does not veto)."""
        p = self.p
        return (abs(roll) < p.upright_deg * DEG and abs(pitch) < p.upright_deg * DEG
                and (not math.isfinite(body_height) or body_height > p.upright_height))

    def is_fall_condition(self, roll: float, pitch: float, body_height: float, latched: bool) -> bool:
        """Instantaneous fall condition with hysteresis (enter / exit thresholds)."""
        a_th = (self.p.fall_angle_exit_deg if latched else self.p.fall_angle_deg) * DEG
        h_th = self.p.fall_height_exit if latched else self.p.fall_height
        tilted = abs(roll) > a_th or abs(pitch) > a_th
        low = math.isfinite(body_height) and body_height < h_th
        return tilted or low

    # ------------------------------------------------------------------ detection
    def check(self, t: float, roll: float, pitch: float, body_height: float, x: float = 0.0,
              y: float = 0.0, q_meas=None) -> bool:
        """Run the detector (call every control step while not recovering).
        Returns True once a fall is confirmed (state -> FALL_DETECTED)."""
        if self.state != NORMAL:
            return False
        cond = self.is_fall_condition(roll, pitch, body_height, self._cand_since is not None)
        if not cond:
            self._cand_since = None
            return False
        if self._cand_since is None:
            self._cand_since = t
        if t - self._cand_since >= self.p.detect_time - 1e-9:
            self._cand_since = None
            self.fall_count += 1
            self.attempt = 0
            self._event("fall", t, (x, y, roll, pitch))
            self._enter(FALL_DETECTED, t, q_meas)
            return True
        return False

    def begin(self, t: float, q_meas, reset_attempts: bool = False) -> None:
        """Manually (re)start a recovery (``recover`` command)."""
        if reset_attempts:
            self.attempt = 0
        self._enter(DAMP, t, q_meas)

    def reset(self) -> None:
        """Back to NORMAL without events (e.g. controller restart)."""
        self.state = NORMAL
        self._cand_since = None
        self._verify_since = None
        self._reset_wait = False

    def confirm_reset(self, t: float, ok: bool) -> bool:
        """Completion of the assisted-reset teleport requested through a
        ``reset_fn`` that returned True.  ok: the settle window starts now;
        failed (the robot was not moved): FAILED - reported through the state,
        a ``recover`` command restarts the recovery.  Returns False (and does
        nothing) when no assisted reset is waiting."""
        if not self.waiting_for_reset:
            return False
        self._reset_wait = False
        if ok:
            self._t_state = float(t)
        else:
            self._enter(FAILED, t, None)
        return True

    # ------------------------------------------------------------------ keyframes
    def tuck_pose(self) -> np.ndarray:
        return _legs(*([self.TUCK_LEG] * 4))

    def prepare_pose(self) -> np.ndarray:
        return _legs(*([self.PREPARE_LEG] * 4))

    def self_right_poses(self, roll_dir: float):
        """(reach, push) keyframes for lying on the right side (roll_dir=+1,
        roll > 0, left side up) or on the left side (roll_dir=-1).

        For a leg pointing to the top side (z' > 0 in the leg plane) a hip_x
        rotation q0 moves the foot laterally by -z' sin(q0): q0 > 0 pushes the
        right legs (ground side for roll > 0) towards -y, i.e. into the ground."""
        tuck = self.TUCK_LEG
        if roll_dir >= 0:
            ground = (1, 3)           # FR, RR are underneath
            sgn = 1.0
        else:
            ground = (0, 2)           # FL, RL
            sgn = -1.0
        reach = [list(tuck) for _ in range(4)]
        push = [list(tuck) for _ in range(4)]
        for leg in ground:
            hy, kn = self.REACH_FRONT if leg in (0, 1) else self.REACH_REAR
            reach[leg] = [-0.25 * sgn, hy, kn]          # slightly lifted off the ground while sweeping
            push[leg] = [self.PUSH_HIP_X * sgn, hy, kn]
        return _legs(*reach), _legs(*push)

    def rock_pose(self, t_rel: float) -> np.ndarray:
        """Sideways leg swing (all hip_x in phase) with growing amplitude."""
        amp = min(1.0, 0.4 + 0.6 * t_rel / max(self.p.rock_time, 1e-3)) * kin.JOINT_UPPER_3[0]
        hx = amp * math.sin(2.0 * math.pi * t_rel / self.p.rock_period)
        leg = (hx, 0.3, -0.6)        # legs fairly extended to maximise the swinging mass
        return _legs(leg, leg, leg, leg)

    # ------------------------------------------------------------------ update
    def update(self, t: float, q_meas, roll: float, pitch: float, body_height: float,
               omega=(0.0, 0.0, 0.0), x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
               z: float = 0.0) -> Optional[np.ndarray]:
        """Advance the recovery sequence; returns (12,) joint targets while
        active, None in NORMAL."""
        p = self.p
        if self.state == NORMAL:
            return None
        q_meas = np.asarray(q_meas, dtype=float)
        dt_s = t - self._t_state
        pose = (x, y, roll, pitch)
        q = self._q_last.copy()

        if self.state == FALL_DETECTED:
            self._enter(DAMP, t, q_meas)
            dt_s = 0.0

        if self.state == DAMP:
            q = q_meas.copy()
            rate = float(np.linalg.norm(np.asarray(omega, dtype=float)))
            if (dt_s >= p.damp_min_time and rate < p.damp_rest_rate) or dt_s >= p.damp_max_time:
                self.attempt += 1
                self.attempts_total += 1
                self._enter(TUCK, t, q_meas)
                dt_s = 0.0

        if self.state == TUCK:
            q = self._interp(self._q_from, self.tuck_pose(), dt_s / p.tuck_time)
            if dt_s >= p.tuck_time:
                if abs(roll) > p.upside_down_deg * DEG:
                    self._enter(ROCK, t, q)
                elif abs(roll) < p.self_right_skip_deg * DEG:
                    self._enter(PREPARE, t, q)
                else:
                    self._roll_dir = 1.0 if roll > 0 else -1.0
                    self._enter(SELF_RIGHT, t, q)
                dt_s = 0.0

        if self.state == ROCK:
            q = self._interp(self._q_from, self.rock_pose(dt_s), dt_s / 0.3)
            if abs(roll) < p.upside_down_deg * DEG - 10 * DEG:
                self._roll_dir = 1.0 if roll > 0 else -1.0
                self._enter(SELF_RIGHT, t, q)
                dt_s = 0.0
            elif dt_s >= p.rock_time:
                return self._attempt_failed(t, q_meas, pose, x, y, z, yaw)

        if self.state == SELF_RIGHT:
            reach, push = self.self_right_poses(self._roll_dir)
            t1 = p.self_right_reach_time
            if dt_s < t1:
                q = self._interp(self._q_from, reach, dt_s / t1)
            else:
                q = self._interp(reach, push, (dt_s - t1) / p.self_right_push_time)
            if abs(roll) < p.self_right_done_deg * DEG and dt_s > 0.2:
                self._enter(PREPARE, t, q)
                dt_s = 0.0
            elif dt_s >= p.self_right_timeout:
                return self._attempt_failed(t, q_meas, pose, x, y, z, yaw)

        if self.state == PREPARE:
            q = self._interp(self._q_from, self.prepare_pose(), dt_s / p.prepare_time)
            if dt_s >= p.prepare_time:
                if abs(roll) > p.self_right_done_deg * DEG + 15 * DEG:
                    # rolled back onto a side while folding: retry
                    return self._attempt_failed(t, q_meas, pose, x, y, z, yaw)
                self._enter(STAND_UP, t, q)
                dt_s = 0.0

        if self.state == STAND_UP:
            q = self._interp(self._q_from, self.q_stand, dt_s / p.stand_up_time)
            if dt_s >= p.stand_up_time:
                self._enter(VERIFY, t, q)
                self._verify_since = None
                dt_s = 0.0

        if self.state == VERIFY:
            q = self.q_stand.copy()
            if self.is_upright(roll, pitch, body_height):
                if self._verify_since is None:
                    self._verify_since = t
                if t - self._verify_since >= p.verify_time:
                    self.successes += 1
                    self._event("recovered", t, pose)
                    self.state = NORMAL
                    self._q_last = q
                    return q
            else:
                self._verify_since = None
            if dt_s >= p.verify_timeout:
                return self._attempt_failed(t, q_meas, pose, x, y, z, yaw)

        if self.state == ASSISTED_RESET:
            q = self.q_stand.copy()
            if self._reset_wait:
                if dt_s >= p.reset_confirm_timeout:
                    # no confirmation (lost callback): carry on as if the teleport happened now
                    self._reset_wait = False
                    self._t_state = t
            elif dt_s >= p.reset_settle_time:
                self._reset_done = True
                self.state = NORMAL
                self._q_last = q
                return q

        if self.state == FAILED:
            q = q_meas.copy()
            if self.is_upright(roll, pitch, body_height):
                # standing after all (e.g. the stand posture held during a failed
                # assisted reset pushed the body up): the controller stands up
                # from the measured posture - a mission must not wait forever
                if self._verify_since is None:
                    self._verify_since = t
                if t - self._verify_since >= p.verify_time:
                    self.successes += 1
                    self._event("recovered", t, pose)
                    self._reset_done = True
                    self.state = NORMAL
                    self._q_last = q
                    return q
            else:
                self._verify_since = None
                if p.failed_retry_time > 0.0 and dt_s >= p.failed_retry_time:
                    self._event("recovery_retry", t, pose)
                    self.attempt = 0
                    self._enter(DAMP, t, q_meas)

        q = kin.clamp_joints(q)
        self._q_last = q
        return q

    def take_reset_done(self) -> bool:
        """True once when the recovery hands over to the controller's stand-up
        from the measured posture: after an assisted reset, or when the robot
        was found upright in FAILED."""
        r, self._reset_done = self._reset_done, False
        return r

    def _interp(self, q0, q1, u) -> np.ndarray:
        return np.asarray(q0, dtype=float) + (np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float)) \
            * _smoothstep(u)

    def _attempt_failed(self, t, q_meas, pose, x, y, z, yaw):
        p = self.p
        if self.attempt < p.max_attempts:
            self._enter(DAMP, t, q_meas)
            return kin.clamp_joints(np.asarray(q_meas, dtype=float))
        self._event("recovery_failed", t, pose)
        if p.assisted_reset and self.reset_fn is not None:
            self.assisted_resets += 1
            try:
                sent = self.reset_fn(float(x), float(y), float(z) + p.reset_height_offset, float(yaw))
            except Exception:
                sent = False
            self._event("assisted_reset", t, pose)
            self.attempt = 0
            self._enter(ASSISTED_RESET, t, q_meas)
            # True: the completion will be reported through confirm_reset() (see class doc)
            self._reset_wait = sent is True
            return self.q_stand.copy()
        self._enter(FAILED, t, q_meas)
        return kin.clamp_joints(np.asarray(q_meas, dtype=float))
