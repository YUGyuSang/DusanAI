"""Small rotation / pose helpers (numpy only, no tf_transformations dependency)."""
import math
import numpy as np


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_euler(x, y, z, w):
    """Quaternion (x, y, z, w) -> (roll, pitch, yaw), ZYX convention."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """(roll, pitch, yaw) -> quaternion (x, y, z, w), ZYX convention."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def yaw_from_quat(q):
    """Yaw from a geometry_msgs Quaternion-like object or (x, y, z, w) tuple."""
    if hasattr(q, "w"):
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_rot(roll, pitch, yaw):
    """Rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll) (body -> world)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rot_to_euler(R):
    """Inverse of euler_to_rot."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def quat_to_rot(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def pose_matrix(x, y, z, roll, pitch, yaw):
    """4x4 homogeneous transform from translation + ZYX euler."""
    T = np.eye(4)
    T[:3, :3] = euler_to_rot(roll, pitch, yaw)
    T[:3, 3] = (x, y, z)
    return T


def transform_points(T, pts):
    """Apply 4x4 transform to (N,3) points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def odom_to_pose(msg):
    """nav_msgs/Odometry -> (x, y, z, roll, pitch, yaw)."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    r, pi, yw = quat_to_euler(q.x, q.y, q.z, q.w)
    return p.x, p.y, p.z, r, pi, yw
