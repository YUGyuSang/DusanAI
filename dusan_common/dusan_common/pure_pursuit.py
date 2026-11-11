"""
Pure-pursuit path follower producing body velocity commands (vx, vy, wz)
for the legged robot.  ROS free: feed it the pose, get a command back.
"""
import math
import numpy as np

from .geometry import wrap_angle


def _densify(p, spacing):
    """(N, 2) polyline -> same polyline with no segment longer than `spacing`
    (a 1e-6 relative slack leaves already dense paths untouched)."""
    if len(p) < 2:
        return p.copy()
    seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    n = np.maximum(1, np.ceil(seg / spacing - 1e-6)).astype(int)
    if np.all(n == 1):
        return p.copy()
    out = [p[:1]]
    for a, b, k in zip(p[:-1], p[1:], n):
        t = (np.arange(1, k + 1, dtype=float) / k)[:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


class PurePursuit:
    def __init__(self, lookahead=0.6, max_vx=0.45, max_wz=0.8, goal_tolerance=0.25,
                 yaw_tolerance=0.25, slow_down_dist=0.8, rotate_in_place_angle=0.9, k_heading=1.6):
        self.lookahead = lookahead
        self.max_vx = max_vx
        self.max_wz = max_wz
        self.goal_tolerance = goal_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.slow_down_dist = slow_down_dist
        self.rotate_in_place_angle = rotate_in_place_angle
        self.k_heading = k_heading
        self.path = None
        self.goal_yaw = None
        self._idx = 0

    def set_path(self, path, goal_yaw=None):
        """path: list of (x, y).  goal_yaw: optional final heading.

        The look-ahead target is chosen among path vertices, so the path is
        resampled to at most ``min(0.1, 0.25 * lookahead)`` between vertices.
        On a sparse path (planning.plan() emits up to 1.5 m segments) the
        nearest vertex can otherwise lie behind the robot and be picked as the
        target (rotation in place).  Already dense paths are kept as given."""
        if path is None or len(path) == 0:
            self.path = None
        else:
            p = np.asarray(path, dtype=float).reshape(len(path), -1)[:, :2]
            self.path = _densify(p, max(0.02, min(0.1, 0.25 * self.lookahead)))
        self.goal_yaw = goal_yaw
        self._idx = 0

    def clear(self):
        self.path = None
        self.goal_yaw = None
        self._idx = 0

    @property
    def active(self):
        return self.path is not None

    def remaining_path(self):
        """The part of the (resampled) path from the current closest vertex
        on, as an (N, 2) array (a view), or None when no path is active."""
        if self.path is None:
            return None
        return self.path[self._idx:]

    def remaining_distance(self, x, y):
        if self.path is None:
            return 0.0
        p = self.path[self._idx:]
        if len(p) == 0:
            return 0.0
        d = math.hypot(p[0, 0] - x, p[0, 1] - y)
        if len(p) > 1:
            d += float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))))
        return d

    def compute(self, x, y, yaw, speed_scale=1.0):
        """Return (vx, vy, wz, done)."""
        if self.path is None:
            return 0.0, 0.0, 0.0, True
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
            return 0.0, 0.0, 0.0, False          # bad pose sample: stand still, keep the path
        p = self.path
        # advance the closest index monotonically
        d = np.hypot(p[self._idx:, 0] - x, p[self._idx:, 1] - y)
        self._idx += int(np.argmin(d[: max(1, min(len(d), 60))]))
        goal = p[-1]
        dist_goal = math.hypot(goal[0] - x, goal[1] - y)
        if dist_goal < self.goal_tolerance:
            if self.goal_yaw is None:
                self.clear()
                return 0.0, 0.0, 0.0, True
            err = wrap_angle(self.goal_yaw - yaw)
            if abs(err) < self.yaw_tolerance:
                self.clear()
                return 0.0, 0.0, 0.0, True
            wz = float(np.clip(self.k_heading * err, -self.max_wz, self.max_wz))
            return 0.0, 0.0, wz, False
        # lookahead point
        target = goal
        for k in range(self._idx, len(p)):
            if math.hypot(p[k, 0] - x, p[k, 1] - y) >= self.lookahead:
                target = p[k]
                break
        heading = math.atan2(target[1] - y, target[0] - x)
        err = wrap_angle(heading - yaw)
        wz = float(np.clip(self.k_heading * err, -self.max_wz, self.max_wz))
        if abs(err) > self.rotate_in_place_angle:
            return 0.0, 0.0, wz, False
        vx = self.max_vx * speed_scale * max(0.0, math.cos(err))
        vx *= min(1.0, max(0.25, dist_goal / max(self.slow_down_dist, 1e-6)))
        # small lateral correction (legged robots can strafe)
        lat = math.sin(err) * min(dist_goal, self.lookahead)
        vy = float(np.clip(0.3 * lat, -0.15, 0.15))
        return float(vx), vy, wz, False
