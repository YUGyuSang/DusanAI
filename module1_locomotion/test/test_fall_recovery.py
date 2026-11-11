"""Fall detection (with hysteresis), recovery sequence, attempts, assisted reset, gz tools."""
import math

import numpy as np
import pytest

from dusan_common import robot_model as rm
from module1_locomotion import fall_recovery as F
from module1_locomotion import gz_tools as GZ
from module1_locomotion import kinematics as kin

DEG = math.pi / 180.0


def _detect(fr, roll_fn, h=0.3, t_end=2.0, dt=0.01):
    t = 0.0
    while t < t_end:
        if fr.check(t, roll_fn(t), 0.0, h):
            return t
        t += dt
    return None


def test_short_tilt_does_not_trigger_long_tilt_does():
    fr = F.FallRecovery()
    # 0.2 s above the threshold: no fall
    assert _detect(fr, lambda t: 70 * DEG if 0.5 <= t < 0.7 else 0.0) is None
    assert fr.state == F.NORMAL and fr.fall_count == 0
    # sustained: detected after 0.3 s
    t = _detect(fr, lambda t: 70 * DEG if t >= 0.5 else 0.0)
    assert t == pytest.approx(0.8, abs=0.011)
    assert fr.state == F.FALL_DETECTED and fr.fall_count == 1
    ev = fr.pop_events()
    assert ev[0]["event"] == "fall" and set(ev[0]) == {"event", "t", "x", "y", "roll", "pitch"}


def test_hysteresis_keeps_candidate_between_thresholds():
    fr = F.FallRecovery()
    # crosses 55 deg once, then oscillates between 51 and 56 deg: still a fall (exit threshold 50)
    t = _detect(fr, lambda t: (56 if int(t * 100) % 2 == 0 else 51) * DEG if t >= 0.2 else 0.0)
    assert t is not None and t == pytest.approx(0.5, abs=0.011)
    fr2 = F.FallRecovery()
    # oscillating 45 <-> 56 deg resets the timer every time it drops below 50
    assert _detect(fr2, lambda t: (56 if int(t * 10) % 2 == 0 else 45) * DEG) is None


def test_low_body_height_triggers():
    fr = F.FallRecovery()
    t = 0.0
    hit = None
    while t < 1.0:
        if fr.check(t, 0.0, 0.0, 0.08):
            hit = t
            break
        t += 0.01
    assert hit == pytest.approx(0.3, abs=0.011)


def test_leaning_for_a_while_is_a_fall():
    """A robot pushed against a wall can stay leaning at 30-55 deg on its
    legs: never a 55 deg fall, but not standing either.  Beyond lean_deg for
    lean_time it counts as fallen (and gets up from there)."""
    fr = F.FallRecovery()
    assert fr.p.lean_deg == pytest.approx(30.0) and fr.p.lean_time == pytest.approx(2.0)
    # 41 deg for 1.9 s, 20 deg for 0.1 s, 41 deg again for 1.85 s: the timer restarts, no fall
    assert _detect(fr, lambda t: (20 if 1.9 <= t < 2.0 else 41) * DEG, t_end=3.85) is None
    fr = F.FallRecovery()
    t = _detect(fr, lambda t: 41 * DEG if t >= 0.5 else 0.0, t_end=4.0)
    assert t == pytest.approx(2.5, abs=0.011) and fr.state == F.FALL_DETECTED and fr.fall_count == 1
    ev = fr.pop_events()
    assert ev[0]["event"] == "fall" and ev[0]["roll"] == pytest.approx(41 * DEG)
    # pitch counts too; below lean_deg never; lean_time 0 switches it off
    fr = F.FallRecovery()
    assert _detect(fr, lambda t: 0.0, t_end=4.0) is None
    t = 0.0
    while t < 4.0 and not fr.check(t, 0.0, -35 * DEG, 0.3):
        t += 0.01
    assert t == pytest.approx(2.0, abs=0.011)
    fr = F.FallRecovery()
    assert _detect(fr, lambda t: 29 * DEG, t_end=5.0) is None
    fr = F.FallRecovery(F.FallParams(lean_time=0.0))
    assert _detect(fr, lambda t: 50 * DEG, t_end=5.0) is None
    # the controller passes its parameters on
    from module1_locomotion.controller import LocomotionController
    ctrl = LocomotionController({"use_mpc_thread": False, "fall_lean_deg": 33.0, "fall_lean_time": 1.5})
    assert ctrl.recovery.p.lean_deg == pytest.approx(33.0) and ctrl.recovery.p.lean_time == pytest.approx(1.5)


def test_keyframes_respect_joint_limits_and_push_direction():
    fr = F.FallRecovery()
    poses = [fr.tuck_pose(), fr.prepare_pose(), fr.q_stand] + list(fr.self_right_poses(1.0)) + \
        list(fr.self_right_poses(-1.0)) + [fr.rock_pose(t) for t in np.linspace(0, 3, 31)]
    for q in poses:
        assert np.all(q >= kin.JOINT_LOWER - 1e-9) and np.all(q <= kin.JOINT_UPPER + 1e-9)
    # lying on the right side (roll > 0): FR/RR reach towards the top side (+z body) and the
    # push moves their feet towards -y (into the ground), the high-side legs stay tucked
    reach, push = fr.self_right_poses(1.0)
    fk_r, fk_p = kin.fk_all(reach), kin.fk_all(push)
    for leg in (1, 3):
        assert fk_r[leg, 2] > 0.1 and fk_p[leg, 2] > 0.1
        assert fk_p[leg, 1] < fk_r[leg, 1] - 0.15
        assert fk_p[leg, 1] < -0.2
    assert np.allclose(push[[0, 1, 2, 6, 7, 8]], fr.tuck_pose()[[0, 1, 2, 6, 7, 8]])
    # mirror for roll < 0
    reach_l, push_l = fr.self_right_poses(-1.0)
    fk_pl = kin.fk_all(push_l)
    for leg in (0, 2):
        assert fk_pl[leg, 1] > 0.2 and fk_pl[leg, 2] > 0.1
    # tucked feet are close to the body
    tuck = kin.fk_all(fr.tuck_pose())
    assert np.all(np.linalg.norm(tuck - kin.HIP_POS - [0, 0, 0], axis=1) < 0.2)


def _simulate(fr, roll0, recover=True, t_end=30.0, dt=0.01):
    """Toy body: roll follows the SELF_RIGHT push, height follows STAND_UP."""
    roll = roll0
    t = 0.0
    q = np.zeros(12)
    states = []
    fr.check(0.0, roll, 0.0, 0.1)
    fr.check(0.31, roll, 0.0, 0.1)
    assert fr.state == F.FALL_DETECTED
    t = 0.31
    while t < t_end:
        if fr.state == F.SELF_RIGHT and recover and t - fr._t_state > 0.9:
            roll -= math.copysign(min(abs(roll), 120 * DEG * dt), roll)     # pushed back onto the belly
        if fr.state in (F.PREPARE, F.STAND_UP, F.VERIFY) and recover:
            roll *= 0.9                                                    # settles flat on the belly
        h = 0.08
        if fr.state in (F.STAND_UP, F.VERIFY, F.NORMAL) and recover:
            h = 0.30
        out = fr.update(t, q, roll, 0.0, h, omega=(0.0, 0.0, 0.0), x=1.0, y=2.0, yaw=0.5, z=0.1)
        if out is not None:
            q = out
        states.append(fr.state)
        if fr.state == F.NORMAL:
            break
        t += dt
    return states, t


def test_recovery_sequence_progresses_to_normal():
    fr = F.FallRecovery()
    states, t = _simulate(fr, 90 * DEG)
    order = []
    for s in states:
        if not order or order[-1] != s:
            order.append(s)
    assert order == [F.DAMP, F.TUCK, F.SELF_RIGHT, F.PREPARE, F.STAND_UP, F.VERIFY, F.NORMAL]
    assert fr.successes == 1 and fr.attempts_total == 1
    names = [e["event"] for e in fr.pop_events()]
    assert names == ["fall", "recovered"]
    assert t < 12.0


def test_on_belly_skips_self_right():
    fr = F.FallRecovery()
    states, _ = _simulate(fr, 10 * DEG)
    assert F.SELF_RIGHT not in states and F.PREPARE in states and states[-1] == F.NORMAL


def test_failed_attempts_then_assisted_reset():
    calls = []
    fr = F.FallRecovery(F.FallParams(assisted_reset=True), reset_fn=lambda *a: calls.append(a))
    states, _ = _simulate(fr, 90 * DEG, recover=False, t_end=60.0)
    assert fr.attempts_total == 3
    assert len(calls) == 1
    x, y, z, yaw = calls[0]
    assert (x, y, yaw) == (1.0, 2.0, 0.5) and z == pytest.approx(0.1 + 0.5)
    names = [e["event"] for e in fr.pop_events()]
    assert names == ["fall", "recovery_failed", "assisted_reset"]
    assert fr.state == F.NORMAL and fr.take_reset_done() and not fr.take_reset_done()
    assert fr.assisted_resets == 1


def _run_until(fr, target, roll=90 * DEG, t_end=60.0, dt=0.01):
    """Fallen on the side and never rolling back: all attempts fail."""
    fr.check(0.0, roll, 0.0, 0.1)
    fr.check(0.31, roll, 0.0, 0.1)
    t = 0.31
    while t < t_end and fr.state != target:
        fr.update(t, np.zeros(12), roll, 0.0, 0.08, x=1.0, y=2.0, yaw=0.5, z=0.1)
        t += dt
    assert fr.state == target
    return t


def test_assisted_reset_settle_starts_at_the_teleport_confirmation():
    """reset_fn returning True promises confirm_reset(): the gz CLI needs
    ~0.3-1 s before the teleport happens, so the settle window must start at
    the confirmation, not at the request."""
    calls = []

    def reset_fn(*a):
        calls.append(a)
        return True

    fr = F.FallRecovery(F.FallParams(assisted_reset=True), reset_fn=reset_fn)
    t = _run_until(fr, F.ASSISTED_RESET)
    assert fr.waiting_for_reset and len(calls) == 1 and fr.attempts_total == 3
    t_req = t
    q = None
    while t < t_req + 5.0:                       # far beyond reset_settle_time (2 s): still waiting
        q = fr.update(t, np.zeros(12), 90 * DEG, 0.0, 0.08)
        t += 0.01
    assert fr.state == F.ASSISTED_RESET and np.allclose(q, fr.q_stand) and not fr.take_reset_done()
    assert fr.confirm_reset(t, True)
    t_conf = t
    while fr.state != F.NORMAL and t < t_conf + 10.0:
        fr.update(t, np.zeros(12), 0.0, 0.0, 0.3)
        t += 0.01
    assert fr.state == F.NORMAL and fr.take_reset_done()
    assert t - t_conf == pytest.approx(fr.p.reset_settle_time, abs=0.02)
    assert not fr.confirm_reset(t, True)         # nothing waiting any more
    names = [e["event"] for e in fr.pop_events()]
    assert names == ["fall", "recovery_failed", "assisted_reset"]


def test_assisted_reset_failure_and_lost_confirmation():
    # the teleport failed: FAILED (reported through the state), 'recover' restarts
    fr = F.FallRecovery(F.FallParams(assisted_reset=True), reset_fn=lambda *a: True)
    t = _run_until(fr, F.ASSISTED_RESET)
    assert fr.confirm_reset(t + 0.5, False)
    assert fr.state == F.FAILED and fr.fallen and not fr.waiting_for_reset
    q_meas = np.full(12, 0.1)
    assert np.allclose(fr.update(t + 0.6, q_meas, 90 * DEG, 0.0, 0.08), kin.clamp_joints(q_meas))
    assert fr.state == F.FAILED
    fr.begin(t + 1.0, q_meas, reset_attempts=True)
    assert fr.state == F.DAMP
    # no confirmation at all: carry on after reset_confirm_timeout (+ the settle time)
    fr2 = F.FallRecovery(F.FallParams(assisted_reset=True, reset_confirm_timeout=3.0), reset_fn=lambda *a: True)
    t = _run_until(fr2, F.ASSISTED_RESET)
    t_req = t
    while fr2.state != F.NORMAL and t < t_req + 20.0:
        fr2.update(t, np.zeros(12), 0.0, 0.0, 0.3)
        t += 0.01
    assert fr2.state == F.NORMAL and fr2.take_reset_done()
    assert t - t_req == pytest.approx(3.0 + fr2.p.reset_settle_time, abs=0.05)
    # a reset_fn that raises / reports "not sent": the settle window starts at the request
    for fn_ret in (False, None):
        fr3 = F.FallRecovery(F.FallParams(assisted_reset=True), reset_fn=lambda *a, r=fn_ret: r)
        _run_until(fr3, F.ASSISTED_RESET)
        assert not fr3.waiting_for_reset


def test_failed_without_reset_stays_failed_until_begin():
    p = F.FallParams(assisted_reset=False, max_attempts=2)
    fr = F.FallRecovery(p)
    states, _ = _simulate(fr, 90 * DEG, recover=False, t_end=40.0)
    assert fr.state == F.FAILED and fr.attempts_total == 2
    fr.begin(50.0, np.zeros(12), reset_attempts=True)
    assert fr.state == F.DAMP and fr.attempt == 0


def test_failed_retries_after_failed_retry_time():
    p = F.FallParams(assisted_reset=False, max_attempts=2, failed_retry_time=4.0)
    fr = F.FallRecovery(p)
    t = _run_until(fr, F.FAILED)
    assert fr.attempts_total == 2
    t0 = fr._t_state
    t = t0
    while fr.state == F.FAILED and t < t0 + 10.0:
        t += 0.01
        fr.update(t, np.zeros(12), 90 * DEG, 0.0, 0.08)
    assert fr.state == F.DAMP and fr.attempt == 0
    assert t - t0 == pytest.approx(4.0, abs=0.02)
    assert "recovery_retry" in [e["event"] for e in fr.pop_events()]
    # the next series of attempts runs as usual
    while fr.state not in (F.FAILED, F.NORMAL) and t < t0 + 60.0:
        t += 0.01
        fr.update(t, np.zeros(12), 90 * DEG, 0.0, 0.08)
    assert fr.state == F.FAILED and fr.attempts_total == 4


def test_failed_but_standing_hands_over_to_the_stand_up():
    """FAILED while the robot stands on its legs (e.g. the stand posture of an
    assisted reset whose teleport failed): recovered after verify_time."""
    fr = F.FallRecovery(F.FallParams(assisted_reset=True), reset_fn=lambda *a: True)
    t = _run_until(fr, F.ASSISTED_RESET)
    assert fr.confirm_reset(t, False) and fr.state == F.FAILED
    fr.pop_events()
    # a short upright moment is not enough
    fr.update(t + 0.1, np.zeros(12), 0.0, 0.0, 0.30)
    fr.update(t + 0.2, np.zeros(12), 0.0, 0.0, 0.08)
    fr.update(t + 0.45, np.zeros(12), 0.0, 0.0, 0.30)
    assert fr.state == F.FAILED
    tt = t + 0.45
    while fr.state == F.FAILED and tt < t + 3.0:
        tt += 0.01
        fr.update(tt, np.zeros(12), 0.0, 0.0, 0.30)
    assert fr.state == F.NORMAL and fr.take_reset_done() and not fr.take_reset_done()
    assert tt - (t + 0.45) == pytest.approx(fr.p.verify_time, abs=0.02)
    assert [e["event"] for e in fr.pop_events()] == ["recovered"]
    # the default (module) behaviour without failed_retry_time: lying robot stays FAILED
    fr2 = F.FallRecovery(F.FallParams(assisted_reset=True), reset_fn=lambda *a: True)
    t = _run_until(fr2, F.ASSISTED_RESET)
    fr2.confirm_reset(t, False)
    for k in range(3000):
        fr2.update(t + 0.01 * k, np.zeros(12), 90 * DEG, 0.0, 0.08)
    assert fr2.state == F.FAILED


def test_upside_down_rocks_first():
    fr = F.FallRecovery()
    fr.check(0.0, 175 * DEG, 0.0, 0.1)
    fr.check(0.4, 175 * DEG, 0.0, 0.1)
    t = 0.4
    seen = set()
    while t < 4.0:
        fr.update(t, np.zeros(12), 175 * DEG, 0.0, 0.1)
        seen.add(fr.state)
        t += 0.01
    assert F.ROCK in seen


def test_default_recovery_is_physical():
    """No teleport by default: FallParams, the controller defaults and
    config/locomotion.yaml all have assisted_reset off; exhausted attempts end
    in FAILED (the controller retries from there), never in ASSISTED_RESET."""
    import os
    import yaml
    from module1_locomotion.controller import DEFAULT_PARAMS
    assert F.FallParams().assisted_reset is False
    assert DEFAULT_PARAMS["assisted_reset"] is False
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "locomotion.yaml")
    with open(path, encoding="utf-8") as f:
        assert yaml.safe_load(f)["locomotion_controller"]["ros__parameters"]["assisted_reset"] is False
    calls = []
    fr = F.FallRecovery(reset_fn=lambda *a: calls.append(a))
    states, _ = _simulate(fr, 90 * DEG, recover=False, t_end=40.0)
    assert fr.state == F.FAILED and fr.attempts_total == 3
    assert calls == [] and fr.assisted_resets == 0 and F.ASSISTED_RESET not in states
    assert [e["event"] for e in fr.pop_events()] == ["fall", "recovery_failed"]


def test_physical_recovery_never_gives_up():
    """assisted_reset off + failed_retry_time (what the controller sets): a
    body that stays on its side gets series after series of attempts with a
    pause in between; once the body rolls back, the robot stands up
    ("recovered") - no teleport at any time."""
    p = F.FallParams(failed_retry_time=5.0)
    fr = F.FallRecovery(p, reset_fn=lambda *a: pytest.fail("teleport requested"))
    roll = 90 * DEG
    fr.check(0.0, roll, 0.0, 0.1)
    fr.check(0.31, roll, 0.0, 0.1)
    t, dt, t_free = 0.31, 0.01, 100.0            # the body cannot roll back before t_free
    q = np.zeros(12)
    while t < 160.0 and fr.state != F.NORMAL:
        if t > t_free:
            if fr.state == F.SELF_RIGHT and t - fr._t_state > 0.9:
                roll -= math.copysign(min(abs(roll), 120 * DEG * dt), roll)
            if fr.state in (F.PREPARE, F.STAND_UP, F.VERIFY):
                roll *= 0.9
        h = 0.30 if t > t_free and fr.state in (F.STAND_UP, F.VERIFY) else 0.08
        out = fr.update(t, q, roll, 0.0, h)
        if out is not None:
            q = out
        t += dt
    names = [e["event"] for e in fr.pop_events()]
    assert fr.state == F.NORMAL and names[-1] == "recovered"
    assert names.count("recovery_retry") >= 3 and names.count("recovery_failed") == names.count("recovery_retry")
    assert fr.assisted_resets == 0 and fr.attempts_total >= 12


def test_back_roll_keyframes():
    """Back roll (roll ~ +180 deg, roll_dir +1): the right (low-side) legs are
    straight and tilted out over the right edge, the left legs swing over the
    dorsal side (+z body = towards the ground on the back) beside the trunk;
    roll_dir -1 is the mirror image.  Joint limits are respected."""
    fr = F.FallRecovery()
    o1, p1 = fr.back_roll_poses(1.0)
    o2, p2 = fr.back_roll_poses(-1.0)
    for q in (o1, p1, o2, p2):
        assert np.all(q >= kin.JOINT_LOWER - 1e-9) and np.all(q <= kin.JOINT_UPPER + 1e-9)
    fo, fp = kin.fk_all(o1), kin.fk_all(p1)
    for leg in (1, 3):                           # low side: straight, out over the edge, away from the back
        assert fo[leg, 1] < -0.3 and fo[leg, 2] < -0.2
        assert np.allclose(fo[leg], fp[leg])
    for leg in (0, 2):                           # high side: tucked when open, over the dorsal side when pushing
        assert np.linalg.norm(fo[leg] - kin.HIP_POS[leg]) < 0.2
        assert fp[leg, 2] > 0.1 and fp[leg, 1] > 0.25
    mirror = [1, 0, 3, 2]
    for a, b in ((o1, o2), (p1, p2)):
        A, B = a.reshape(4, 3), b.reshape(4, 3)
        for leg in range(4):
            m = A[mirror[leg]]
            assert np.allclose(B[leg], [-m[0], m[1], m[2]])


def test_back_roll_cycles_and_hands_over_to_self_right():
    fr = F.FallRecovery()
    p = fr.p
    roll = 175 * DEG
    fr.check(0.0, roll, 0.0, 0.1)
    fr.check(0.31, roll, 0.0, 0.1)
    t = 0.31
    while fr.state != F.ROCK and t < 5.0:
        fr.update(t, np.zeros(12), roll, 0.0, 0.1)
        t += 0.01
    assert fr.state == F.ROCK and fr._roll_dir == 1.0
    t_rock = fr._t_state
    opn, push = fr.back_roll_poses(1.0)
    # open, then push / retract / hold cycles
    cyc = p.rock_push_time + p.rock_retract_time + p.rock_hold_time
    assert np.allclose(fr.rock_pose(p.rock_open_time), opn)
    assert np.allclose(fr.rock_pose(p.rock_open_time + p.rock_push_time), push)
    assert np.allclose(fr.rock_pose(p.rock_open_time + p.rock_push_time + p.rock_retract_time + 0.1), opn)
    assert np.allclose(fr.rock_pose(p.rock_open_time + cyc + p.rock_push_time), push)
    # tipping onto a side before the legs are open does not end the back roll
    fr.update(t_rock + 0.3, np.zeros(12), 120 * DEG, 0.0, 0.1)
    assert fr.state == F.ROCK
    fr.update(t_rock + p.rock_open_time + 0.5, np.zeros(12), 150 * DEG, 0.0, 0.1)
    assert fr.state == F.ROCK
    fr.update(t_rock + p.rock_open_time + 0.6, np.zeros(12), 120 * DEG, 0.0, 0.1)
    assert fr.state == F.SELF_RIGHT and fr._roll_dir == 1.0 and fr.attempts_total == 1
    # stuck on the back: the attempt ends after rock_time
    fr2 = F.FallRecovery()
    fr2.check(0.0, -178 * DEG, 0.0, 0.1)
    fr2.check(0.31, -178 * DEG, 0.0, 0.1)
    t = 0.31
    while fr2.state != F.ROCK:
        fr2.update(t, np.zeros(12), -178 * DEG, 0.0, 0.1)
        t += 0.01
    assert fr2._roll_dir == -1.0
    t0 = fr2._t_state
    while fr2.state == F.ROCK and t < t0 + 10.0:
        fr2.update(t, np.zeros(12), -178 * DEG, 0.0, 0.1)
        t += 0.01
    assert fr2.state == F.DAMP and fr2.attempt == 1
    assert t - t0 == pytest.approx(p.rock_time, abs=0.02)


def test_back_roll_alternates_after_a_blocked_roll():
    """A back roll that times out (e.g. a wall or the uphill side of a slope on
    that side) is followed by one in the other direction; a new fall starts
    from the side the body leans to again."""
    fr = F.FallRecovery()
    roll = 178 * DEG
    fr.check(0.0, roll, 0.0, 0.1)
    fr.check(0.31, roll, 0.0, 0.1)
    t, dirs, prev = 0.31, [], None
    while t < 40.0 and fr.state != F.FAILED:
        fr.update(t, np.zeros(12), roll, 0.0, 0.1)
        if fr.state == F.ROCK and prev != F.ROCK:
            dirs.append(fr._roll_dir)
        prev = fr.state
        t += 0.01
    assert dirs == [1.0, -1.0, 1.0] and fr.state == F.FAILED
    # a successful back roll clears the memory; so does a new fall
    fr.reset()
    fr.check(50.0, roll, 0.0, 0.1)
    fr.check(50.31, roll, 0.0, 0.1)
    t = 50.31
    while fr.state != F.ROCK:
        fr.update(t, np.zeros(12), roll, 0.0, 0.1)
        t += 0.01
    assert fr._roll_dir == 1.0


def test_self_right_rolling_onto_the_back_starts_the_back_roll():
    """SELF_RIGHT that rolls the body over onto its back: back roll at once
    (counted as the next attempt, no DAMP / TUCK); on the last attempt the
    series ends instead."""
    fr = F.FallRecovery()
    fr.check(0.0, 90 * DEG, 0.0, 0.1)
    fr.check(0.31, 90 * DEG, 0.0, 0.1)
    t = 0.31
    while fr.state != F.SELF_RIGHT:
        fr.update(t, np.zeros(12), 90 * DEG, 0.0, 0.1)
        t += 0.01
    assert fr.attempts_total == 1
    t_sr = fr._t_state
    fr.update(t_sr + 0.2, np.zeros(12), -165 * DEG, 0.0, 0.1)     # too early to judge
    assert fr.state == F.SELF_RIGHT
    fr.update(t_sr + 0.4, np.zeros(12), -165 * DEG, 0.0, 0.1)
    assert fr.state == F.ROCK and fr._roll_dir == -1.0 and fr.attempts_total == 2 and fr.attempt == 2
    # last attempt: no further back roll, the series fails (FAILED without assisted reset)
    fr.attempt = fr.p.max_attempts
    fr.state = F.SELF_RIGHT
    fr._t_state = 20.0
    fr.update(20.5, np.zeros(12), 170 * DEG, 0.0, 0.1)
    assert fr.state == F.FAILED and fr.attempts_total == 2


def test_prepare_folds_the_legs_before_swinging_them():
    """PREPARE from an extended SELF_RIGHT posture: knees folded and hip_x
    centred first (hip_y unchanged), then hip_y swung to the tuck angle, then
    the PREPARE pose."""
    fr = F.FallRecovery()
    p = fr.p
    _reach, push = fr.self_right_poses(1.0)
    fr._q_from = push.copy()
    t1 = p.prepare_fold_time
    t2 = t1 + p.prepare_swing_time
    assert fr.prepare_time() == pytest.approx(t2 + p.prepare_set_time)
    q = fr.prepare_traj(t1)
    assert np.allclose(q[0::3], 0.0) and np.allclose(q[2::3], fr.TUCK_LEG[2]) and np.allclose(q[1::3], push[1::3])
    q = fr.prepare_traj(t2)
    assert np.allclose(q[1::3], fr.TUCK_LEG[1]) and np.allclose(q[2::3], fr.TUCK_LEG[2])
    assert np.allclose(fr.prepare_traj(fr.prepare_time()), fr.prepare_pose())
    for tt in np.linspace(0.0, fr.prepare_time(), 27):
        q = fr.prepare_traj(tt)
        assert np.all(q >= kin.JOINT_LOWER - 1e-9) and np.all(q <= kin.JOINT_UPPER + 1e-9)


# ---------------------------------------------------------------------------- gz tools
def test_gz_command_lines():
    cmd = GZ.build_wrench_cmd("plant", "spot::base_link", (0.0, 150.0, 0.0))
    assert cmd[:6] == ["gz", "topic", "-t", "/world/plant/wrench/persistent", "-m", "gz.msgs.EntityWrench"]
    assert cmd[6] == "-p"
    assert 'entity: {name: "spot::base_link", type: LINK}' in cmd[7]
    assert "force: {x: 0, y: 150, z: 0}" in cmd[7]
    clr = GZ.build_clear_cmd("plant", "spot::base_link")
    assert clr[3] == "/world/plant/wrench/clear" and clr[5] == "gz.msgs.Entity"
    assert clr[7] == 'name: "spot::base_link", type: LINK'
    sp = GZ.build_set_pose_cmd("plant", "spot", 61.9, 0.0, 0.18, math.pi)
    assert sp[:4] == ["gz", "service", "-s", "/world/plant/set_pose"]
    assert sp[sp.index("--reqtype") + 1] == "gz.msgs.Pose"
    assert sp[sp.index("--reptype") + 1] == "gz.msgs.Boolean"
    assert sp[sp.index("--timeout") + 1] == "2000"
    req = sp[sp.index("--req") + 1]
    assert req.startswith('name: "spot", position: {x: 61.9, y: 0, z: 0.18}')
    qx, qy, qz, qw = GZ.yaw_to_quat(math.pi)
    assert qz == pytest.approx(1.0) and abs(qw) < 1e-12


def test_gz_push_runs_async_and_clears():
    log = []

    def runner(cmd, timeout):
        log.append(list(cmd))
        return 0, "data: true"

    sleeps = []
    gz = GZ.GzTools(runner=runner, sleep=lambda s: sleeps.append(s))
    assert gz.push(10.0, -20.0, 0.0, duration=0.25)
    gz.wait()
    topics = [c[3] for c in log]
    # apply + clear (started concurrently, order not fixed) + the unconditional safety clear
    assert sorted(topics) == ["/world/plant/wrench/clear"] * 2 + ["/world/plant/wrench/persistent"]
    assert topics[-1] == "/world/plant/wrench/clear"
    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(0.25, abs=0.05)
    assert gz.history[-1]["ok"]
    done = []
    assert gz.reset_pose(1.0, 2.0, 0.6, 0.3, on_done=done.append)
    gz.wait()
    assert log[-1][1] == "service" and gz.history[-1]["job"] == "reset_pose"
    assert gz.history[-1]["ok"] and done == [True]
    # a set_pose service that does not answer (UserCommands missing) or fails is reported as
    # failed even when the gz CLI exits with code 0; retried once; on_done always called
    for reply in ("Service call timed out", "data: false", "Service call failed"):
        calls = []
        gz.runner = lambda cmd, timeout, reply=reply: (calls.append(cmd) or (0, reply))
        done.clear()
        assert gz.reset_pose(1.0, 2.0, 0.6, 0.3, on_done=done.append)
        gz.wait()
        assert not gz.history[-1]["ok"], reply
        assert done == [False] and len(calls) == 2, reply


def test_gz_push_clear_is_timed_from_the_apply_start():
    """Every gz CLI call publishes only after its start-up and discovery: the
    clear must be *started* ``duration`` after the apply was started - timed
    from the end of the apply process the force stayed on for duration + the
    whole start-up time of the clear process (~1 s instead of 0.2 s)."""
    import threading
    import time
    starts = {}
    lk = threading.Lock()
    lat = 0.3

    def slow_cli(cmd, timeout):
        with lk:
            starts.setdefault(cmd[3], []).append(time.monotonic())
        time.sleep(lat)
        return 0, ""

    gz = GZ.GzTools(runner=slow_cli)
    assert gz.push(0.0, 150.0, duration=0.05)
    gz.wait()
    t_apply = starts["/world/plant/wrench/persistent"][0]
    t_clear = starts["/world/plant/wrench/clear"]
    assert len(t_clear) == 2
    assert 0.04 <= t_clear[0] - t_apply < 0.2       # sequential timing: >= lat + duration = 0.35 s
    assert t_clear[1] - t_clear[0] >= lat - 0.02     # safety clear after both processes ended
    h = gz.history[-1]
    assert h["ok"] and h["on_time"] == pytest.approx(0.05, abs=0.05)
    assert h["latency"][0] == pytest.approx(lat, abs=0.2)


def test_gz_push_duration_is_measured_on_the_sim_clock():
    sim = [100.0]
    sleeps = []

    def rtf_half_sleep(s):          # real-time factor 0.5
        sleeps.append(s)
        sim[0] += 0.5 * s

    gz = GZ.GzTools(runner=lambda cmd, timeout: (0, ""), sleep=rtf_half_sleep, clock=lambda: sim[0])
    assert gz.push(0.0, 150.0, duration=0.2)
    gz.wait()
    h = gz.history[-1]
    assert h["ok"] and h["on_time"] == pytest.approx(0.2, abs=0.01)
    assert sum(sleeps) == pytest.approx(0.4, abs=0.02)          # 0.2 s sim = 0.4 s wall
    # a clock that does not advance (paused / no /clock) cannot keep the force on: wall-time cap
    gz2 = GZ.GzTools(runner=lambda cmd, timeout: (0, ""), sleep=lambda s: None, clock=lambda: 5.0)
    wall = [0.0]

    def fake_monotonic():
        wall[0] += 0.01
        return wall[0]

    gz2._monotonic = fake_monotonic
    assert gz2.push(0.0, 150.0, duration=0.2)
    gz2.wait()
    assert gz2.history[-1]["ok"] and gz2.history[-1]["on_time"] == 0.0
    assert len(gz2.history[-1]["cmds"]) == 3


def test_gz_reset_on_done_is_always_called():
    def boom(cmd, timeout):
        raise RuntimeError("runner bug")

    done = []
    gz = GZ.GzTools(runner=boom)
    assert gz.reset_pose(0.0, 0.0, 0.5, 0.0, on_done=done.append)
    gz.wait()
    assert done == [False]
    assert GZ.set_pose_reply_ok("data: true\n\n")
    for bad in ("Service call timed out", "data: false", "Service call failed", "timeout after 3.0s"):
        assert not GZ.set_pose_reply_ok(bad), bad


def test_gz_busy_rejection_and_missing_executable():
    import threading
    gate = threading.Event()

    def slow(cmd, timeout):
        gate.wait(2.0)
        return 0, ""

    gz = GZ.GzTools(runner=slow, sleep=lambda s: None)
    assert gz.push(1, 2)
    assert not gz.push(3, 4)                  # second push rejected while busy
    assert gz.reset_pose(0, 0, 0.5, 0)        # a reset is still accepted
    gate.set()
    gz.wait()
    missing = GZ.GzTools(gz_exe="definitely_not_a_gz_binary_123")
    assert not missing.available()
    assert not missing.push(1, 1)
    # the default runner reports instead of raising
    rc, out = GZ._default_runner(["definitely_not_a_gz_binary_123"], 1.0)
    assert rc != 0 and out
