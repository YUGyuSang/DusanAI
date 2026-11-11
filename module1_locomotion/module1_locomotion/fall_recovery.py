"""
Fall detection and self-righting state machine (ROS free).

States (DESIGN_SPEC §5)::

    NORMAL --(|roll| or |pitch| > 55 deg, or body height < 0.12 m, for 0.3 s;
              or |roll| or |pitch| > 30 deg for 2 s: leaning, e.g. against a wall)--> FALL_DETECTED
    FALL_DETECTED -> DAMP -> TUCK -> [ROCK] -> SELF_RIGHT -> PREPARE -> STAND_UP -> NORMAL
                                    (any failure)  -> next attempt (DAMP) ... 3 attempts
                                    -> FAILED  (or ASSISTED_RESET via gz set_pose when
                                                assisted_reset is enabled, counted separately)
    SELF_RIGHT --(rolled over onto the back)--> ROCK (counted as the next attempt)
    FAILED --(upright for 0.3 s, e.g. held up by the stand posture of a failed assisted reset)--> NORMAL
           --(failed_retry_time > 0: after that long)--> DAMP (new series of attempts)

The recovery is physical by default (``assisted_reset`` False): the robot
gets up with its own legs, and a failed series of attempts is followed by a
pause (``failed_retry_time``, set by the controller) and a new series - it
never gives up for good while the controller runs with ``auto_recover``.

Joint-space keyframes are designed for this leg geometry (hip_x +-0.70 rad,
hip_y -1.6..2.6 rad, knee -2.8..-0.2 rad, thigh 0.226 m, shank 0.300 m):

* DAMP: the servos follow the measured joints (no fighting the impact) until
  the body has come to rest.
* TUCK: all legs folded (hip_y 1.25, knee -2.70: foot 0.11 m below and 0.08 m
  in front of the hip) so the body can roll freely.
* ROCK (back roll, only when lying on the back, |roll| > 120 deg): "open" -
  the legs of the side that stays low are stretched straight and tilted out
  over the low edge of the trunk (hip_x 0.7), which moves the CoM towards
  that edge and halves the roll needed to tip over (~58 -> ~29 deg); the legs
  of the side that has to rise tilt their leg plane outwards.  Then cycles
  of "push" (those legs swing over the dorsal side, their feet meet the
  ground beside the trunk and lift the high side) - "retract" (the pushing
  legs leave the ground at the peak of the roll, the CoM falls back towards
  the low edge and the body tips over onto its side) - "hold".  Ends at
  |roll| < 135 deg (on a side -> SELF_RIGHT) or after ``rock_time``; the
  next attempt then rolls the other way (a wall or the uphill side of a
  slope can block one direction).
* SELF_RIGHT: lying on a side, the legs on the *ground side* are swung into
  the leg plane towards the robot's top side (front leg forward-up
  hip_y -1.5 / knee -1.5, rear leg backward-up hip_y 2.4 / knee -0.6; the leg
  plane is horizontal when the robot lies on its side) and then abducted
  (hip_x 0.6 towards the ground).  The feet press on the ground on the *top*
  side of the CoM, so the ground reaction produces a moment that rolls the
  body back onto its belly (pressing on the belly side would roll it further
  over).  The high-side legs stay tucked.  If the body rolls over onto its
  back instead, the back roll starts right away (no new DAMP / TUCK).
* PREPARE: on the belly, in three phases - fold every knee and centre hip_x
  without moving hip_y, swing hip_y to the tuck angle with the knee folded
  (the hip lifts the body over a foot 0.14 m away with ~5-10 N m instead of
  jamming an extended 0.5 m leg against the ground at the 45 N m limit),
  then open to the feet-under-the-hips pose with deeply bent knees.
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
    lean_deg: float = 30.0                # |roll| or |pitch| beyond for lean_time -> fall (leaning on a wall,
    lean_time: float = 2.0                #   propped up on an obstacle: not standing, never 55 deg either)
    damp_min_time: float = 0.5
    damp_max_time: float = 2.0
    damp_rest_rate: float = 1.0           # |omega| [rad/s] considered at rest
    tuck_time: float = 0.8
    rock_time: float = 6.0                # back roll: longest time per attempt [s]
    rock_open_time: float = 0.6           # back roll phases [s]: open (once), then push / retract / hold cycles
    rock_push_time: float = 0.4
    rock_retract_time: float = 0.35
    rock_hold_time: float = 0.5
    rock_exit_deg: float = 135.0          # back roll done: |roll| below -> on a side -> SELF_RIGHT
    self_right_reach_time: float = 0.7
    self_right_push_time: float = 0.6
    self_right_timeout: float = 3.0
    self_right_done_deg: float = 35.0     # |roll| below -> on the belly
    self_right_skip_deg: float = 35.0     # already on the belly -> skip SELF_RIGHT
    self_right_over_deg: float = 150.0    # SELF_RIGHT rolled the body onto its back -> back roll
    upside_down_deg: float = 120.0
    prepare_fold_time: float = 0.35       # PREPARE phases [s]: fold knees / swing hip_y / open
    prepare_swing_time: float = 0.6
    prepare_set_time: float = 0.35
    stand_up_time: float = 2.5
    verify_time: float = 0.3
    verify_timeout: float = 1.5
    upright_deg: float = 25.0
    upright_height: float = 0.22
    max_attempts: int = 3
    assisted_reset: bool = False          # gz set_pose (teleport) after the attempts - off: physical only
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
    BACK_FRONT = (-1.60, -1.50)           # back roll (hip_y, knee), front leg: foot forward, far onto the dorsal side
    BACK_REAR = (2.60, -0.20)             # rear leg: straight, backward over the dorsal side
    BACK_LOW = (0.00, -0.25)              # low-side legs: straight, pointing up (away from the back)
    BACK_HIP_X = 0.70                     # |hip_x| of the back roll: leg planes tilted outwards

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
        self._lean_since: Optional[float] = None
        self._t_state = 0.0
        self._q_from = np.zeros(12)
        self._q_last = np.zeros(12)
        self._roll_dir = 1.0
        self._rock_failed_dir: Optional[float] = None   # direction of the last back roll that timed out
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
        p = self.p
        lean = p.lean_time > 0.0 and (abs(roll) > p.lean_deg * DEG or abs(pitch) > p.lean_deg * DEG)
        if not lean:
            self._lean_since = None
        elif self._lean_since is None:
            self._lean_since = t
        leaning = lean and t - self._lean_since >= p.lean_time - 1e-9
        cond = self.is_fall_condition(roll, pitch, body_height, self._cand_since is not None)
        if not cond:
            self._cand_since = None
        elif self._cand_since is None:
            self._cand_since = t
        if leaning or (cond and t - self._cand_since >= p.detect_time - 1e-9):
            self._cand_since = None
            self._lean_since = None
            self.fall_count += 1
            self.attempt = 0
            self._rock_failed_dir = None
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
        self._lean_since = None
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

    def back_roll_poses(self, roll_dir: float):
        """(open, push) keyframes of the back roll.  roll_dir=+1 (roll ~ +180
        deg): the LEFT side has to rise (roll -> +90 deg, then lying on the
        right side); roll_dir=-1 mirrored.

        * low-side legs (right for roll_dir +1): straight, pointing up and
          tilted outwards over the low edge of the trunk (hip_x -0.7 for a right
          leg) - moves the CoM towards the pivot edge, which halves the roll
          needed to tip over (~58 -> ~29 deg);
        * high-side legs: leg plane tilted outwards (hip_x -0.7 for a left leg
          lying on its back), tucked in the open pose, swung over the dorsal
          side in the push pose: the feet meet the ground beside the trunk and
          lift the high side."""
        tuck = self.TUCK_LEG
        if roll_dir >= 0:
            push_legs, low_legs, sgn = (0, 2), (1, 3), -1.0
        else:
            push_legs, low_legs, sgn = (1, 3), (0, 2), 1.0
        hx = sgn * self.BACK_HIP_X
        opn = [list(tuck) for _ in range(4)]
        push = [list(tuck) for _ in range(4)]
        for leg in push_legs:
            hy, kn = self.BACK_FRONT if leg in (0, 1) else self.BACK_REAR
            opn[leg] = [hx, tuck[1], tuck[2]]
            push[leg] = [hx, hy, kn]
        for leg in low_legs:
            opn[leg] = [hx, *self.BACK_LOW]
            push[leg] = [hx, *self.BACK_LOW]
        return _legs(*opn), _legs(*push)

    def rock_pose(self, t_rel: float) -> np.ndarray:
        """Back roll (ROCK) targets ``t_rel`` s after entering: open (from the
        entry posture, ``rock_open_time``) -> cycles of push (high-side legs
        over the dorsal side) -> retract to open (the pushing legs leave the
        ground at the peak of the roll, the CoM moves back towards the low
        edge and the body tips over) -> hold open."""
        p = self.p
        opn, push = self.back_roll_poses(self._roll_dir)
        if t_rel < p.rock_open_time:
            return kin.clamp_joints(self._interp(self._q_from, opn, t_rel / p.rock_open_time))
        u = (t_rel - p.rock_open_time) % (p.rock_push_time + p.rock_retract_time + p.rock_hold_time)
        if u < p.rock_push_time:
            return self._interp(opn, push, u / p.rock_push_time)
        if u < p.rock_push_time + p.rock_retract_time:
            return self._interp(push, opn, (u - p.rock_push_time) / p.rock_retract_time)
        return opn

    def prepare_time(self) -> float:
        """Duration of PREPARE [s]."""
        return self.p.prepare_fold_time + self.p.prepare_swing_time + self.p.prepare_set_time

    def prepare_traj(self, t_rel: float) -> np.ndarray:
        """PREPARE targets ``t_rel`` s after entering (on the belly): (1) fold
        every knee and centre hip_x without moving hip_y (an extended leg left
        over from SELF_RIGHT stays in the air or only short-levers the body),
        (2) swing hip_y to the tuck angle with the knee folded, (3) open to the
        PREPARE pose."""
        p = self.p
        q0 = np.asarray(self._q_from, dtype=float)
        fold = q0.copy()
        fold[0::3] = 0.0
        fold[2::3] = self.TUCK_LEG[2]
        swing = fold.copy()
        swing[1::3] = self.TUCK_LEG[1]
        t1 = p.prepare_fold_time
        t2 = t1 + p.prepare_swing_time
        if t_rel < t1:
            q = self._interp(q0, fold, t_rel / t1)
        elif t_rel < t2:
            q = self._interp(fold, swing, (t_rel - t1) / p.prepare_swing_time)
        else:
            q = self._interp(swing, self.prepare_pose(), (t_rel - t2) / p.prepare_set_time)
        return kin.clamp_joints(q)

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
                    self._roll_dir = self._back_roll_dir(roll)
                    self._enter(ROCK, t, q)
                elif abs(roll) < p.self_right_skip_deg * DEG:
                    self._enter(PREPARE, t, q)
                else:
                    self._roll_dir = 1.0 if roll > 0 else -1.0
                    self._enter(SELF_RIGHT, t, q)
                dt_s = 0.0

        if self.state == ROCK:
            q = self.rock_pose(dt_s)
            if abs(roll) < p.rock_exit_deg * DEG and dt_s >= p.rock_open_time:
                self._roll_dir = 1.0 if roll > 0 else -1.0
                self._rock_failed_dir = None
                self._enter(SELF_RIGHT, t, q)
                dt_s = 0.0
            elif dt_s >= p.rock_time:
                self._rock_failed_dir = self._roll_dir
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
            elif abs(roll) > p.self_right_over_deg * DEG and dt_s > 0.3:
                # rolled over onto the back instead of the belly: back roll right away
                # (saves the DAMP + TUCK of a new attempt, ~2-3 s), counted as an attempt
                if self.attempt >= p.max_attempts:
                    return self._attempt_failed(t, q_meas, pose, x, y, z, yaw)
                self.attempt += 1
                self.attempts_total += 1
                self._roll_dir = 1.0 if roll > 0 else -1.0
                self._enter(ROCK, t, q)
                dt_s = 0.0
            elif dt_s >= p.self_right_timeout:
                return self._attempt_failed(t, q_meas, pose, x, y, z, yaw)

        if self.state == PREPARE:
            q = self.prepare_traj(dt_s)
            if dt_s >= self.prepare_time():
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

    def _back_roll_dir(self, roll: float) -> float:
        """Direction of a back roll: towards the side the body already leans
        to (sign of roll), but the other way when the last back roll in that
        direction timed out - a wall or the uphill side of a slope can block
        one direction, and on the back the sign of roll (~ +-180 deg) hardly
        tells which side is free."""
        d = 1.0 if roll > 0 else -1.0
        if self._rock_failed_dir is not None and d == self._rock_failed_dir:
            d = -d
        return d

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
