"""odom_tf helpers: spawn-relative odometry detection and twist estimation."""
import math

import numpy as np
import pytest

from dusan_common.geometry import euler_to_quat, quat_to_euler
from dusan_common.layout import Layout

from plant_simulation.odom_logic import SpawnOffsetCorrector, TwistEstimator, quat_multiply, yaw_quat

SPAWN = Layout.load().spawn          # (61.9, 0.0, -0.32, pi) - far from the origin


def test_world_frame_odometry_is_untouched():
    c = SpawnOffsetCorrector(SPAWN)
    p, q = c.apply((61.9, 0.02, -0.30), euler_to_quat(0.0, 0.0, math.pi))
    assert c.decision == "world"
    assert p == pytest.approx((61.9, 0.02, -0.30))
    assert quat_to_euler(*q)[2] == pytest.approx(math.pi, abs=1e-9) or \
        quat_to_euler(*q)[2] == pytest.approx(-math.pi, abs=1e-9)


def test_spawn_relative_odometry_is_composed_with_the_spawn_pose():
    c = SpawnOffsetCorrector(SPAWN)
    p, q = c.apply((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert c.decision == "spawn_relative"
    assert "spawn" in c.reason
    assert p == pytest.approx(SPAWN[:3])
    assert abs(math.remainder(quat_to_euler(*q)[2] - SPAWN[3], 2 * math.pi)) < 1e-9
    # the robot walks 1 m "forward" in its odometry frame: spawn yaw = pi -> world -x
    p, q = c.apply((1.0, 0.5, 0.02), euler_to_quat(0.0, 0.1, 0.2))
    assert p == pytest.approx((SPAWN[0] - 1.0, SPAWN[1] - 0.5, SPAWN[2] + 0.02), abs=1e-6)
    r, pi_, y = quat_to_euler(*q)
    assert pi_ == pytest.approx(0.1, abs=1e-9)
    assert abs(math.remainder(y - (SPAWN[3] + 0.2), 2 * math.pi)) < 1e-9


def test_decision_is_made_once():
    c = SpawnOffsetCorrector(SPAWN)
    c.apply((61.0, 0.0, 0.0), (0, 0, 0, 1))
    assert c.decision == "world"
    p, _ = c.apply((0.1, 0.1, 0.0), (0, 0, 0, 1))      # later samples never flip the decision
    assert p == pytest.approx((0.1, 0.1, 0.0))


def test_spawn_near_origin_cannot_be_spawn_relative():
    c = SpawnOffsetCorrector((1.0, 0.5, 0.2, 0.0))
    c.apply((0.0, 0.0, 0.0), (0, 0, 0, 1))
    assert c.decision == "world"


def test_forced_modes_and_validation():
    c = SpawnOffsetCorrector(SPAWN, mode="spawn_relative")
    p, _ = c.apply((10.0, 0.0, 0.0), (0, 0, 0, 1))
    assert p == pytest.approx((SPAWN[0] - 10.0, SPAWN[1], SPAWN[2]), abs=1e-6)
    c = SpawnOffsetCorrector(SPAWN, mode="world")
    assert c.apply((0.0, 0.0, 0.0), (0, 0, 0, 1))[0] == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        SpawnOffsetCorrector(SPAWN, mode="bogus")


def test_quaternion_helpers():
    q = quat_multiply(yaw_quat(0.3), yaw_quat(0.4))
    assert quat_to_euler(*q)[2] == pytest.approx(0.7)
    # degenerate quaternion is normalised to identity by apply()
    c = SpawnOffsetCorrector(SPAWN, mode="world")
    _, q = c.apply((0, 0, 0), (0, 0, 0, 0))
    assert q == (0.0, 0.0, 0.0, 1.0)


def test_twist_estimator_body_frame_velocity():
    est = TwistEstimator(alpha=1.0)
    yaw = 0.5
    v_body = np.array([0.4, 0.1, 0.0])
    wz = 0.3
    x = np.zeros(3)
    dt = 0.02
    for k in range(50):
        t = k * dt
        yk = yaw + wz * t
        q = euler_to_quat(0.0, 0.0, yk)
        v, w = est.update(t, x, q)
        c, s = math.cos(yk), math.sin(yk)
        x = x + dt * np.array([c * v_body[0] - s * v_body[1], s * v_body[0] + c * v_body[1], 0.0])
    # finite differences of a rotating frame: allow a small discretisation error
    assert v == pytest.approx(v_body, abs=0.01)
    assert w[2] == pytest.approx(wz, abs=1e-6)


def test_twist_estimator_resets_on_gaps():
    est = TwistEstimator(alpha=1.0, max_dt=0.5)
    est.update(0.0, (0, 0, 0), (0, 0, 0, 1))
    v, _ = est.update(0.1, (0.05, 0, 0), (0, 0, 0, 1))
    assert v[0] == pytest.approx(0.5)
    v, _ = est.update(5.0, (10.0, 0, 0), (0, 0, 0, 1))   # 4.9 s gap: no 2 m/s spike
    assert v[0] == pytest.approx(0.5)
