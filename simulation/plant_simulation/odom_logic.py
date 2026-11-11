"""
ROS-free helpers for the odom_tf node.

* :class:`SpawnOffsetCorrector` - Gazebo's OdometryPublisher reports the world
  pose of the model in gz-sim 8+, but some versions / configurations report
  the pose relative to where the model was spawned.  The corrector inspects
  the FIRST odometry sample: if it lies within ``near_origin_m`` of the origin
  while the layout spawn pose is farther than ``spawn_far_m`` away, the
  odometry is spawn-relative and every sample is composed with the spawn pose:

        p_world = p_spawn + Rz(yaw_spawn) * p_odom
        q_world = q_z(yaw_spawn) (x) q_odom

  The body-frame twist is invariant under this composition.
* :class:`TwistEstimator` - optional body-frame twist from finite differences
  of the ground-truth pose (first-order low-pass), for when the plugin's twist
  is unusable.
"""
import math

import numpy as np

from dusan_common.geometry import quat_to_rot

ODOM_DEFAULTS = {
    "input_topic": "/model/spot/odometry",
    "output_topic": "/odom",
    "odom_frame": "odom",
    "base_frame": "base_link",
    "map_frame": "map",
    "publish_tf": True,
    "publish_map_to_odom": True,
    "spawn_mode": "auto",          # auto | world | spawn_relative
    "near_origin_m": 1.0,
    "spawn_far_m": 2.0,
    "twist_mode": "passthrough",   # passthrough | differentiate
    "twist_filter_alpha": 0.3,     # EMA weight of the newest finite-difference sample
    "layout_file": "",
    "stale_warn_s": 3.0,           # warn when no odometry for this long (wall clock)
}


def quat_multiply(q1, q2):
    """Hamilton product q1 (x) q2 for (x, y, z, w) tuples."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def quat_normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12 or not math.isfinite(n):
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(c / n for c in q)


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class SpawnOffsetCorrector:
    """Decides once (on the first sample) whether odometry is spawn-relative."""

    def __init__(self, spawn, mode="auto", near_origin_m=1.0, spawn_far_m=2.0):
        sx, sy, sz, syaw = (float(v) for v in spawn)
        self.spawn = (sx, sy, sz, syaw)
        self.mode = str(mode).lower()
        if self.mode not in ("auto", "world", "spawn_relative"):
            raise ValueError(f"spawn_mode must be auto|world|spawn_relative, got {mode!r}")
        self.near_origin_m = float(near_origin_m)
        self.spawn_far_m = float(spawn_far_m)
        self.decision = None   # 'world' | 'spawn_relative' once decided
        self.reason = ""

    @property
    def decided(self):
        return self.decision is not None

    def decide(self, x, y):
        """Inspect the first odometry position.  Returns the decision string."""
        if self.decision is not None:
            return self.decision
        sx, sy = self.spawn[0], self.spawn[1]
        d_origin = math.hypot(x, y)
        d_spawn_origin = math.hypot(sx, sy)
        if self.mode == "world":
            self.decision, self.reason = "world", "forced by spawn_mode=world"
        elif self.mode == "spawn_relative":
            self.decision, self.reason = "spawn_relative", "forced by spawn_mode=spawn_relative"
        elif d_origin < self.near_origin_m and d_spawn_origin > self.spawn_far_m:
            self.decision = "spawn_relative"
            self.reason = (f"first odometry pose ({x:.2f}, {y:.2f}) is {d_origin:.2f} m from the origin "
                           f"but the layout spawn is {d_spawn_origin:.2f} m away")
        else:
            self.decision = "world"
            self.reason = (f"first odometry pose ({x:.2f}, {y:.2f}) is consistent with world-frame odometry "
                           f"(spawn at ({sx:.2f}, {sy:.2f}))")
        return self.decision

    def apply(self, pos, quat):
        """Return (pos_world, quat_world).  Decides on the first call."""
        x, y, z = (float(v) for v in pos)
        q = quat_normalize(tuple(float(v) for v in quat))
        if self.decide(x, y) == "world":
            return (x, y, z), q
        sx, sy, sz, syaw = self.spawn
        c, s = math.cos(syaw), math.sin(syaw)
        pw = (sx + c * x - s * y, sy + s * x + c * y, sz + z)
        qw = quat_normalize(quat_multiply(yaw_quat(syaw), q))
        return pw, qw


class TwistEstimator:
    """Body-frame linear / angular velocity from successive world poses."""

    def __init__(self, alpha=0.3, max_dt=0.5):
        self.alpha = float(alpha)
        self.max_dt = float(max_dt)
        self._prev = None
        self.v = np.zeros(3)
        self.w = np.zeros(3)

    def reset(self):
        self._prev = None
        self.v[:] = 0.0
        self.w[:] = 0.0

    def update(self, t, pos, quat):
        """Feed a pose; returns (v_body(3), w_body(3)) filtered."""
        p = np.asarray(pos, dtype=float)
        q = quat_normalize(tuple(quat))
        if self._prev is None:
            self._prev = (t, p, q)
            return self.v.copy(), self.w.copy()
        t0, p0, q0 = self._prev
        dt = t - t0
        if dt <= 1e-6:
            return self.v.copy(), self.w.copy()
        if dt > self.max_dt:          # gap (pause / reset): restart the estimate
            self._prev = (t, p, q)
            return self.v.copy(), self.w.copy()
        R = quat_to_rot(*q)
        v_body = R.T @ ((p - p0) / dt)
        # relative rotation expressed in the body frame: dq = q0^-1 (x) q
        q0_inv = (-q0[0], -q0[1], -q0[2], q0[3])
        dq = quat_normalize(quat_multiply(q0_inv, q))
        if dq[3] < 0.0:               # shortest path
            dq = tuple(-c for c in dq)
        vec = np.array(dq[:3])
        s = float(np.linalg.norm(vec))
        angle = 2.0 * math.atan2(s, dq[3])
        w_body = (vec / s) * angle / dt if s > 1e-12 else np.zeros(3)
        a = self.alpha
        self.v = (1.0 - a) * self.v + a * v_body
        self.w = (1.0 - a) * self.w + a * w_body
        self._prev = (t, p, q)
        return self.v.copy(), self.w.copy()
