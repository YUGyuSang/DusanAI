"""
Leg kinematics of the Spot-like quadruped (ROS free, numpy only).

Kinematic tree of one leg (zero joint angles = leg straight down), identical to
``dusan_common.robot_model`` and the URDF::

    base_link --hip_x (revolute, axis +X) @ (sx*0.2245, sy*0.044, 0)--> <leg>_hip
    <leg>_hip --hip_y (axis +Y) @ (0, sy*0.090, 0)-------------------> <leg>_upper_leg
    <leg>_upper_leg --knee (axis +Y) @ (0, 0, -L1)-------------------> <leg>_lower_leg
    <leg>_lower_leg --foot_fixed @ (0, 0, -L2)-----------------------> <leg>_foot

with ``sx = +1`` front / ``-1`` rear and ``sy = +1`` left / ``-1`` right.  The foot
centre (centre of the r = 0.025 m foot sphere) in the base frame is

    p = h + Rx(q0) * ( [0, sy*l0, 0] + Ry(q1) * ( [0, 0, -L1] + Ry(q2) * [0, 0, -L2] ) )

Writing the sagittal (leg plane) coordinates

    x' = -L1 sin(q1) - L2 sin(q1 + q2)
    z' = -L1 cos(q1) - L2 cos(q1 + q2)

the foot is ``p = h + [x',  a cos q0 - z' sin q0,  a sin q0 + z' cos q0]`` with
``a = sy*l0``.  Sign convention: ``hip_y > 0`` swings the foot backwards and the
knee is always negative ("knee points backwards", Spot style, all four legs).

Every function is vectorised: joint/foot arrays may have any leading shape
``(..., 3)`` and the ``leg`` argument is an int or an int array broadcastable to
the leading shape.  Leg order is FL, FR, RL, RR (0..3).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from dusan_common import robot_model as rm
from dusan_common.geometry import euler_to_rot

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
L0 = rm.HIP_TO_THIGH_Y          # hip_x joint -> hip_y joint lateral offset
L1 = rm.UPPER_LEG_LENGTH        # hip_y -> knee
L2 = rm.LOWER_LEG_LENGTH        # knee -> foot centre
N_LEGS = 4

SIGN_X = np.asarray(rm.LEG_SIGN_X, dtype=float)
SIGN_Y = np.asarray(rm.LEG_SIGN_Y, dtype=float)
HIP_POS = np.asarray([rm.hip_x_position(i) for i in range(N_LEGS)], dtype=float)   # (4, 3)

_LIM = rm.JOINT_LIMITS
JOINT_LOWER_3 = np.array([_LIM["hip_x"]["lower"], _LIM["hip_y"]["lower"], _LIM["knee"]["lower"]])
JOINT_UPPER_3 = np.array([_LIM["hip_x"]["upper"], _LIM["hip_y"]["upper"], _LIM["knee"]["upper"]])
EFFORT_LIMIT_3 = np.array([_LIM["hip_x"]["effort"], _LIM["hip_y"]["effort"], _LIM["knee"]["effort"]])
VELOCITY_LIMIT_3 = np.array([_LIM["hip_x"]["velocity"], _LIM["hip_y"]["velocity"], _LIM["knee"]["velocity"]])
JOINT_LOWER = np.tile(JOINT_LOWER_3, N_LEGS)       # (12,) in JOINT_NAMES order
JOINT_UPPER = np.tile(JOINT_UPPER_3, N_LEGS)
EFFORT_LIMITS = np.tile(EFFORT_LIMIT_3, N_LEGS)
VELOCITY_LIMITS = np.tile(VELOCITY_LIMIT_3, N_LEGS)

# Reach of the planar 2-link chain (hip_y -> foot) allowed by the knee limits:
# |p'|^2 = L1^2 + L2^2 + 2 L1 L2 cos(q2)
R_MAX = math.sqrt(L1 * L1 + L2 * L2 + 2.0 * L1 * L2 * math.cos(JOINT_UPPER_3[2]))   # knee at -0.20
R_MIN = math.sqrt(L1 * L1 + L2 * L2 + 2.0 * L1 * L2 * math.cos(JOINT_LOWER_3[2]))   # knee at -2.80


def _leg_arrays(leg, shape) -> Tuple[np.ndarray, np.ndarray]:
    """Broadcast hip position (…,3) and sy*l0 (…) for a leg index (array)."""
    leg = np.broadcast_to(np.asarray(leg, dtype=int), shape)
    return HIP_POS[leg], SIGN_Y[leg] * L0


# ---------------------------------------------------------------------------
# Forward kinematics / Jacobian
# ---------------------------------------------------------------------------
def leg_fk(q, leg) -> np.ndarray:
    """Foot centre position(s) in the base frame for joint angles ``q`` (…,3)."""
    q = np.asarray(q, dtype=float)
    hip, a = _leg_arrays(leg, q.shape[:-1])
    q0, q1, q2 = q[..., 0], q[..., 1], q[..., 2]
    q12 = q1 + q2
    xp = -L1 * np.sin(q1) - L2 * np.sin(q12)
    zp = -L1 * np.cos(q1) - L2 * np.cos(q12)
    c0, s0 = np.cos(q0), np.sin(q0)
    out = np.empty(q.shape, dtype=float)
    out[..., 0] = hip[..., 0] + xp
    out[..., 1] = hip[..., 1] + a * c0 - zp * s0
    out[..., 2] = hip[..., 2] + a * s0 + zp * c0
    return out


def leg_jacobian(q, leg) -> np.ndarray:
    """Analytic foot Jacobian dp/dq (…,3,3) in the base frame (column j = joint j)."""
    q = np.asarray(q, dtype=float)
    _hip, a = _leg_arrays(leg, q.shape[:-1])
    q0, q1, q2 = q[..., 0], q[..., 1], q[..., 2]
    q12 = q1 + q2
    zp = -L1 * np.cos(q1) - L2 * np.cos(q12)
    c0, s0 = np.cos(q0), np.sin(q0)
    # derivatives of the sagittal coordinates
    dxp1 = -L1 * np.cos(q1) - L2 * np.cos(q12)
    dzp1 = L1 * np.sin(q1) + L2 * np.sin(q12)
    dxp2 = -L2 * np.cos(q12)
    dzp2 = L2 * np.sin(q12)
    J = np.zeros(q.shape[:-1] + (3, 3), dtype=float)
    # column 0: rotation of the leg plane about +X
    J[..., 1, 0] = -a * s0 - zp * c0
    J[..., 2, 0] = a * c0 - zp * s0
    # column 1: hip_y
    J[..., 0, 1] = dxp1
    J[..., 1, 1] = -dzp1 * s0
    J[..., 2, 1] = dzp1 * c0
    # column 2: knee
    J[..., 0, 2] = dxp2
    J[..., 1, 2] = -dzp2 * s0
    J[..., 2, 2] = dzp2 * c0
    return J


def fk_all(q12) -> np.ndarray:
    """(12,) joint vector in JOINT_NAMES order -> (4, 3) foot centres (base frame)."""
    q = np.asarray(q12, dtype=float).reshape(N_LEGS, 3)
    return leg_fk(q, np.arange(N_LEGS))


def jacobians_all(q12) -> np.ndarray:
    """(12,) joint vector -> (4, 3, 3) foot Jacobians."""
    q = np.asarray(q12, dtype=float).reshape(N_LEGS, 3)
    return leg_jacobian(q, np.arange(N_LEGS))


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------
def leg_ik(p, leg, clamp: bool = True, reach_margin: float = 0.0):
    """Analytic inverse kinematics (knee-backwards branch, knee < 0).

    Parameters
    ----------
    p : (…, 3) desired foot centre position(s) in the base frame.
    leg : leg index or index array.
    clamp : clamp the result to the joint limits (always recommended).
    reach_margin : extra margin [m] kept from the workspace boundary.

    Returns
    -------
    q : (…, 3) joint angles (hip_x, hip_y, knee).
    reachable : (…) bool, False if the target had to be projected onto the
        workspace or the joint limits.  ``leg_fk(q)`` is then the closest
        achievable foot position along the projection used.
    """
    p = np.asarray(p, dtype=float)
    hip, a = _leg_arrays(leg, p.shape[:-1])
    d = p - hip
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    reachable = np.ones(p.shape[:-1], dtype=bool)

    # --- hip_x: the (y, z) projection of the foot is the vector (a, z') rotated by q0
    ryz2 = dy * dy + dz * dz
    zp2 = ryz2 - a * a
    min_zp = 1e-4
    bad = zp2 < min_zp * min_zp
    reachable &= ~bad
    zp = -np.sqrt(np.maximum(zp2, min_zp * min_zp))          # foot below the hip_y axis
    q0 = np.arctan2(dz, dy) - np.arctan2(zp, a)
    q0 = (q0 + math.pi) % (2.0 * math.pi) - math.pi

    # --- planar two-link chain in the leg plane
    xp = dx
    r = np.sqrt(xp * xp + zp * zp)
    r_hi = R_MAX - reach_margin
    r_lo = R_MIN + reach_margin
    out = (r > r_hi) | (r < r_lo)
    reachable &= ~out
    r_c = np.clip(r, r_lo, r_hi)
    scale = np.where(r > 1e-9, r_c / np.maximum(r, 1e-9), 1.0)
    xp = xp * scale
    zp = zp * scale
    cos_q2 = (r_c * r_c - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    q2 = -np.arccos(np.clip(cos_q2, -1.0, 1.0))                 # knee-backwards branch
    q1 = np.arctan2(-xp, -zp) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))

    q = np.stack([q0, q1, q2], axis=-1)
    if clamp:
        qc = np.clip(q, JOINT_LOWER_3, JOINT_UPPER_3)
        reachable &= np.all(np.abs(qc - q) < 1e-9, axis=-1)
        q = qc
    return q, reachable


def ik_all(feet_body, clamp: bool = True, reach_margin: float = 0.0):
    """(4, 3) foot centres in the base frame -> ((12,) joints, (4,) reachable)."""
    feet = np.asarray(feet_body, dtype=float).reshape(N_LEGS, 3)
    q, ok = leg_ik(feet, np.arange(N_LEGS), clamp=clamp, reach_margin=reach_margin)
    return q.reshape(-1), ok


def clamp_joints(q12) -> np.ndarray:
    """Clamp a (12,) (or (…,12)) joint vector to the URDF joint limits."""
    return np.clip(np.asarray(q12, dtype=float), JOINT_LOWER, JOINT_UPPER)


def clamp_to_workspace(p, leg, reach_margin: float = 0.005):
    """Project foot target(s) onto the reachable workspace: returns (p_reachable, was_reachable)."""
    q, ok = leg_ik(p, leg, clamp=True, reach_margin=reach_margin)
    return leg_fk(q, leg), ok


# ---------------------------------------------------------------------------
# Statics
# ---------------------------------------------------------------------------
def leg_torques(q, leg, f_body) -> np.ndarray:
    """Joint torques that realise the ground reaction force ``f_body`` (force of
    the ground ON the foot, base frame):  tau = -J(q)^T f.  Shapes (…,3)."""
    J = leg_jacobian(q, leg)
    f = np.asarray(f_body, dtype=float)
    return -np.einsum("...ji,...j->...i", J, f)


def joint_torques(q12, forces_world, R_body) -> np.ndarray:
    """(12,) joints, (4,3) world-frame GRFs, body rotation (3,3) -> (12,) torques.

    tau_i = -J_i(q_i)^T R^T f_i  (DESIGN_SPEC §5)."""
    q = np.asarray(q12, dtype=float).reshape(N_LEGS, 3)
    f_body = np.asarray(forces_world, dtype=float).reshape(N_LEGS, 3) @ np.asarray(R_body)  # = (R^T f)^T
    return leg_torques(q, np.arange(N_LEGS), f_body).reshape(-1)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def _rot(rpy_or_R) -> np.ndarray:
    a = np.asarray(rpy_or_R, dtype=float)
    if a.shape == (3, 3):
        return a
    return euler_to_rot(float(a[0]), float(a[1]), float(a[2]))


def body_to_world(points_body, pos, rpy_or_R) -> np.ndarray:
    """Base-frame point(s) (…,3) -> world frame, base pose (pos, rpy | R)."""
    R = _rot(rpy_or_R)
    return np.asarray(points_body, dtype=float) @ R.T + np.asarray(pos, dtype=float)


def world_to_body(points_world, pos, rpy_or_R) -> np.ndarray:
    """World point(s) (…,3) -> base frame, base pose (pos, rpy | R)."""
    R = _rot(rpy_or_R)
    return (np.asarray(points_world, dtype=float) - np.asarray(pos, dtype=float)) @ R


# ---------------------------------------------------------------------------
# Postures
# ---------------------------------------------------------------------------
def nominal_feet_body(body_height: float = rm.NOMINAL_BODY_HEIGHT, dx: float = 0.0,
                      dy_out: float = 0.0) -> np.ndarray:
    """(4,3) nominal foot centres in the base frame for a stand height above the
    ground (foot sphere touching the ground).  ``dx`` shifts all feet forward,
    ``dy_out`` widens the stance laterally."""
    feet = np.asarray(rm.nominal_foot_positions(body_height), dtype=float)
    feet[:, 0] += dx
    feet[:, 1] += SIGN_Y * dy_out
    return feet


def stand_joint_angles(body_height: float = rm.NOMINAL_BODY_HEIGHT) -> np.ndarray:
    """(12,) joint angles of the symmetric stand posture at ``body_height``."""
    q, _ok = ik_all(nominal_feet_body(body_height))
    return q


def manipulability(q, leg) -> np.ndarray:
    """Yoshikawa manipulability sqrt(det(J J^T)) — near 0 close to singularities."""
    J = leg_jacobian(q, leg)
    return np.sqrt(np.abs(np.linalg.det(J @ np.swapaxes(J, -1, -2))))
