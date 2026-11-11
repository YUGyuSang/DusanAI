"""
Pinhole camera model matching Gazebo cameras.

Gazebo convention: a camera looks along +X of its link frame (y left, z up).
Pixel (u, v): u to the right, v down, principal point at the image centre,
fx = fy = (width / 2) / tan(hfov / 2).
"""
import math
import numpy as np

from .geometry import pose_matrix
from .robot_model import SENSORS


class PinholeCamera:
    def __init__(self, width, height, hfov):
        self.width = int(width)
        self.height = int(height)
        self.hfov = float(hfov)
        self.fx = (self.width / 2.0) / math.tan(self.hfov / 2.0)
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.vfov = 2.0 * math.atan(self.cy / self.fy)

    @classmethod
    def from_sensor(cls, name):
        s = SENSORS[name]
        return cls(s["width"], s["height"], s["hfov"])

    @property
    def K(self):
        """Intrinsic matrix for the OPTICAL frame (z forward, x right, y down)."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    # link frame (x fwd, y left, z up)  <->  optical frame (z fwd, x right, y down)
    @staticmethod
    def link_to_optical(p):
        p = np.asarray(p, dtype=float)
        return np.stack([-p[..., 1], -p[..., 2], p[..., 0]], -1)

    @staticmethod
    def optical_to_link(p):
        p = np.asarray(p, dtype=float)
        return np.stack([p[..., 2], -p[..., 0], -p[..., 1]], -1)

    def project_link(self, pts_link):
        """(N,3) points in camera link frame -> (N,2) pixels, (N,) depth (x).
        Points behind the camera get NaN pixels."""
        p = np.atleast_2d(np.asarray(pts_link, dtype=float))
        x = p[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx - self.fx * p[:, 1] / x
            v = self.cy - self.fy * p[:, 2] / x
        bad = x <= 1e-6
        u[bad] = np.nan
        v[bad] = np.nan
        return np.stack([u, v], -1), x

    def pixel_ray_link(self, u, v):
        """Unit ray direction(s) in the link frame for pixel(s)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d = np.stack([np.ones_like(u), -(u - self.cx) / self.fx, -(v - self.cy) / self.fy], -1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def in_image(self, uv, margin=0):
        uv = np.atleast_2d(uv)
        return ((uv[:, 0] >= margin) & (uv[:, 0] < self.width - margin)
                & (uv[:, 1] >= margin) & (uv[:, 1] < self.height - margin))


def sensor_extrinsic(name):
    """4x4 transform base_link <- sensor link."""
    x, y, z, r, p, yw = SENSORS[name]["pose"]
    return pose_matrix(x, y, z, r, p, yw)


def camera_world_pose(robot_pose, sensor_name):
    """robot_pose = (x, y, z, roll, pitch, yaw) of base_link in world.
    Returns 4x4 transform world <- camera link."""
    return pose_matrix(*robot_pose) @ sensor_extrinsic(sensor_name)


def world_to_camera_link(T_world_cam, pts_world):
    T_inv = np.linalg.inv(T_world_cam)
    pts = np.atleast_2d(np.asarray(pts_world, dtype=float))
    return pts @ T_inv[:3, :3].T + T_inv[:3, 3]
