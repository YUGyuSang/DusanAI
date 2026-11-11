"""Leg kinematics: FK conventions, analytic IK round trips, Jacobian, statics."""
import math

import numpy as np
import pytest

from dusan_common import robot_model as rm
from module1_locomotion import kinematics as kin

LEGS = np.arange(4)


def _random_q(rng, n):
    """Random joint configurations inside the limits whose foot lies below the
    hip_y axis in the leg plane (the branch the analytic IK returns)."""
    out = []
    while len(out) < n:
        q = rng.uniform(kin.JOINT_LOWER_3, kin.JOINT_UPPER_3)
        q1, q2 = q[1], q[2]
        zp = -kin.L1 * math.cos(q1) - kin.L2 * math.cos(q1 + q2)
        if zp < -0.05:
            out.append(q)
    return np.array(out)


def test_zero_pose_is_straight_down():
    for leg in range(4):
        p = kin.leg_fk(np.zeros(3), leg)
        hx, hy, _ = rm.hip_x_position(leg)
        expect = [hx, hy + rm.LEG_SIGN_Y[leg] * rm.HIP_TO_THIGH_Y, -(rm.UPPER_LEG_LENGTH + rm.LOWER_LEG_LENGTH)]
        assert np.allclose(p, expect, atol=1e-12)


def test_sign_conventions():
    # hip_y > 0 swings the foot backwards (all legs)
    for leg in range(4):
        assert kin.leg_fk([0.0, 0.3, -0.6], leg)[0] < kin.leg_fk([0.0, 0.0, -0.6], leg)[0] - 0.05
    # stand posture: foot under the hip, knee behind the hip-foot line (knee points backwards)
    q = kin.stand_joint_angles()
    for leg in range(4):
        q1 = q[3 * leg + 1]
        assert q[3 * leg + 2] < 0.0
        knee_x = rm.hip_x_position(leg)[0] - kin.L1 * math.sin(q1)
        assert knee_x < rm.hip_x_position(leg)[0]
    # positive hip_x rotates the leg about +X: the foot of a straight-down leg moves to +y
    for leg in range(4):
        assert kin.leg_fk([0.3, 0.8, -1.6], leg)[1] > kin.leg_fk([0.0, 0.8, -1.6], leg)[1]


def test_stand_pose_matches_nominal_feet():
    q = kin.stand_joint_angles(rm.NOMINAL_BODY_HEIGHT)
    feet = kin.fk_all(q)
    assert np.allclose(feet, np.asarray(rm.nominal_foot_positions()), atol=1e-9)
    assert np.allclose(q.reshape(4, 3)[:, 0], 0.0, atol=1e-12)
    assert np.all(q >= kin.JOINT_LOWER) and np.all(q <= kin.JOINT_UPPER)


def test_fk_ik_roundtrip_random():
    rng = np.random.default_rng(1)
    for leg in range(4):
        qs = _random_q(rng, 300)
        p = kin.leg_fk(qs, leg)
        q2, ok = kin.leg_ik(p, leg)
        assert np.all(ok)
        assert np.allclose(q2, qs, atol=1e-8)


def test_ik_reaches_cartesian_targets_vectorised():
    rng = np.random.default_rng(2)
    nominal = kin.nominal_feet_body()
    targets = nominal[None, :, :] + rng.uniform([-0.12, -0.06, -0.08], [0.12, 0.06, 0.06], size=(200, 4, 3))
    legs = np.broadcast_to(LEGS, (200, 4))
    q, ok = kin.leg_ik(targets, legs)
    assert q.shape == (200, 4, 3) and ok.shape == (200, 4)
    assert np.all(ok)
    assert np.allclose(kin.leg_fk(q, legs), targets, atol=1e-9)


def test_workspace_clamping():
    # far below the hip: unreachable, projected onto the reach boundary, still within limits
    for leg in range(4):
        hip = np.asarray(rm.hip_x_position(leg))
        target = hip + np.array([0.0, rm.LEG_SIGN_Y[leg] * 0.09, -1.0])
        q, ok = kin.leg_ik(target, leg)
        assert not ok
        assert np.all(np.isfinite(q))
        assert np.all(q >= kin.JOINT_LOWER_3 - 1e-12) and np.all(q <= kin.JOINT_UPPER_3 + 1e-12)
        p = kin.leg_fk(q, leg)
        reach = np.linalg.norm(p - (hip + np.array([0.0, rm.LEG_SIGN_Y[leg] * 0.09, 0.0])))
        assert reach == pytest.approx(kin.R_MAX, abs=1e-6)
        # too close to the hip: knee limit reached
        q2, ok2 = kin.leg_ik(hip + np.array([0.0, rm.LEG_SIGN_Y[leg] * 0.09, -0.05]), leg)
        assert not ok2 and q2[2] == pytest.approx(kin.JOINT_LOWER_3[2])
    # clamp_to_workspace returns a reachable point
    p, ok = kin.clamp_to_workspace(np.array([0.9, 0.2, -0.2]), 0)
    q, ok2 = kin.leg_ik(p, 0)
    assert not ok and np.allclose(kin.leg_fk(q, 0), p, atol=1e-6)


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(3)
    eps = 1e-6
    for leg in range(4):
        for q in _random_q(rng, 50):
            J = kin.leg_jacobian(q, leg)
            Jn = np.zeros((3, 3))
            for j in range(3):
                dq = np.zeros(3)
                dq[j] = eps
                Jn[:, j] = (kin.leg_fk(q + dq, leg) - kin.leg_fk(q - dq, leg)) / (2 * eps)
            assert np.allclose(J, Jn, atol=1e-8)
    # vectorised shape
    Js = kin.jacobians_all(kin.stand_joint_angles())
    assert Js.shape == (4, 3, 3)


def test_static_torques_virtual_work_and_signs():
    q = kin.stand_joint_angles()
    f = np.tile([0.0, 0.0, rm.TOTAL_MASS * 9.81 / 4.0], (4, 1))     # ground pushes up on every foot
    tau = kin.joint_torques(q, f, np.eye(3)).reshape(4, 3)
    # knee extensors carry the weight: positive knee torque, hip_y ~ 0 (foot under the hip)
    assert np.all(tau[:, 2] > 5.0)
    assert np.all(np.abs(tau[:, 1]) < 0.5)
    # virtual work: tau . dq = -f . dp  (joint power balances the external force)
    rng = np.random.default_rng(4)
    for leg in range(4):
        dq = rng.normal(size=3) * 1e-6
        dp = kin.leg_fk(q[3 * leg:3 * leg + 3] + dq, leg) - kin.leg_fk(q[3 * leg:3 * leg + 3], leg)
        assert np.dot(tau[leg], dq) == pytest.approx(-np.dot(f[leg], dp), rel=1e-4, abs=1e-12)
    # rotated body: the body-frame force is R^T f
    R = kin._rot([0.0, 0.0, 1.0])
    tau_r = kin.joint_torques(q, f, R)
    assert np.allclose(tau_r.reshape(4, 3)[:, 1:], tau[:, 1:], atol=1e-9)


def test_body_world_roundtrip():
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(4, 3))
    pos, rpy = np.array([61.9, 0.3, -0.54]), np.array([0.1, -0.2, 3.0])
    w = kin.body_to_world(pts, pos, rpy)
    assert np.allclose(kin.world_to_body(w, pos, rpy), pts, atol=1e-12)


def test_rest_pose_from_rig_is_consistent_with_robot_model():
    """The FBX rig bones (lead's analysis) imply the link lengths and rest angles;
    FK with those angles must reproduce the hand/foot bone positions."""
    def to_robot(g):
        return np.array([g[1] - 0.0145, -g[0], g[2] - 0.365])

    rig = {  # leg: (hip_y joint bone, knee bone, foot bone) in the G frame
        "fr": ((0.134, 0.237, 0.364), (0.135, 0.110, 0.178), (0.134, 0.354, 0.004)),
        "fl": ((-0.134, 0.248, 0.369), (-0.143, 0.124, 0.181), (-0.124, 0.365, 0.004)),
        "rr": ((0.133, -0.209, 0.365), (0.130, -0.405, 0.252), (0.134, -0.236, 0.004)),
        "rl": ((-0.134, -0.209, 0.365), (-0.131, -0.405, 0.252), (-0.134, -0.236, 0.004)),
    }
    for name, (hip_g, knee_g, foot_g) in rig.items():
        leg = rm.LEG_NAMES.index(name)
        h, k, f = to_robot(hip_g), to_robot(knee_g), to_robot(foot_g)
        assert np.linalg.norm(k - h) == pytest.approx(rm.UPPER_LEG_LENGTH, abs=0.006)
        assert np.linalg.norm(f - k) == pytest.approx(rm.LOWER_LEG_LENGTH, abs=0.006)
        theta = math.atan2(-(k - h)[0], -(k - h)[2])
        phi = math.atan2(-(f - k)[0], -(f - k)[2])
        q = np.array([0.0, theta, phi - theta])
        p = kin.leg_fk(q, leg)
        # foot bone sits at the sole (z ~ 0.004 above the ground plane of the rig)
        assert abs(p[0] - f[0]) < 0.012 and abs(p[2] - f[2]) < 0.012
        if name in ("fr", "fl"):
            assert math.degrees(theta) == pytest.approx(34.3, abs=1.5)
            assert math.degrees(phi - theta) == pytest.approx(-88.8, abs=2.0)
        else:
            assert math.degrees(theta) == pytest.approx(60.0, abs=1.5)
            assert math.degrees(phi - theta) == pytest.approx(-94.3, abs=2.0)


def test_manipulability_vanishes_when_straight():
    straight = kin.manipulability(np.array([0.0, 0.0, -1e-6]), 0)
    bent = kin.manipulability(kin.stand_joint_angles()[:3], 0)
    assert straight < 1e-4 < bent
