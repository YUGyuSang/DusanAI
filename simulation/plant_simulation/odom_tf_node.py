"""
odom_tf node: ground-truth odometry relay + TF.

  /model/spot/odometry (gz OdometryPublisher via ros_gz_bridge)
      -> /odom (nav_msgs/Odometry, frame 'odom', child 'base_link',
                pose in the world frame, twist in the base frame)
      -> TF odom -> base_link
  static TF map -> odom (identity; Gazebo world = map = odom)

A spawn-relative odometry source is auto-detected on the first sample and
composed with the layout spawn pose (see plant_simulation.odom_logic).
"""
import time

import rclpy  # noqa: F401
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from dusan_common.layout import Layout
from dusan_common.ros_helpers import declare_params

from plant_simulation.node_utils import best_effort_qos, reliable_qos, run_node
from plant_simulation.odom_logic import ODOM_DEFAULTS, SpawnOffsetCorrector, TwistEstimator


def _wall_timer(node, period, callback):
    """Timer on the steady clock (keeps firing when /clock is absent)."""
    try:
        from rclpy.clock import Clock, ClockType
        return node.create_timer(period, callback, clock=Clock(clock_type=ClockType.STEADY_TIME))
    except TypeError:  # very old rclpy without the clock argument
        return node.create_timer(period, callback)


class OdomTfNode(Node):
    def __init__(self):
        super().__init__("odom_tf")
        self.p = declare_params(self, ODOM_DEFAULTS)
        layout = Layout.load(self.p["layout_file"] or None)
        spawn = layout.spawn
        self.corrector = SpawnOffsetCorrector(spawn, self.p["spawn_mode"], self.p["near_origin_m"],
                                              self.p["spawn_far_m"])
        self.twist_est = TwistEstimator(alpha=float(self.p["twist_filter_alpha"]))
        self.differentiate = str(self.p["twist_mode"]).lower() == "differentiate"

        self.pub = self.create_publisher(Odometry, self.p["output_topic"], reliable_qos(10))
        self.tf_br = TransformBroadcaster(self) if self.p["publish_tf"] else None
        self.static_br = None
        if self.p["publish_map_to_odom"]:
            self.static_br = StaticTransformBroadcaster(self)
            self._send_static()
        self.create_subscription(Odometry, self.p["input_topic"], self._on_odom, best_effort_qos(10))

        self._last_rx = None
        self._n_rx = 0
        self._wall_t0 = time.monotonic()
        _wall_timer(self, 1.0, self._watchdog)
        self.get_logger().info(
            f"odom_tf: {self.p['input_topic']} -> {self.p['output_topic']} + TF "
            f"{self.p['odom_frame']}->{self.p['base_frame']}; spawn {spawn} (mode {self.p['spawn_mode']}), "
            f"twist {self.p['twist_mode']}")

    def _send_static(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.p["map_frame"]
        t.child_frame_id = self.p["odom_frame"]
        t.transform.rotation.w = 1.0
        self.static_br.sendTransform(t)

    # ------------------------------------------------------------------ odom
    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        first = not self.corrector.decided
        (x, y, z), (qx, qy, qz, qw) = self.corrector.apply((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        if first:
            if self.corrector.decision == "spawn_relative":
                self.get_logger().warning(
                    f"spawn-relative odometry detected: {self.corrector.reason}; composing with the layout "
                    f"spawn pose {self.corrector.spawn}")
            else:
                self.get_logger().info(f"world-frame odometry: {self.corrector.reason}")

        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()

        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = self.p["odom_frame"]
        out.child_frame_id = self.p["base_frame"]
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = x, y, z
        o = out.pose.pose.orientation
        o.x, o.y, o.z, o.w = qx, qy, qz, qw
        out.pose.covariance = msg.pose.covariance
        if self.differentiate:
            v, w = self.twist_est.update(stamp.sec + stamp.nanosec * 1e-9, (x, y, z), (qx, qy, qz, qw))
            out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = (float(c) for c in v)
            out.twist.twist.angular.x, out.twist.twist.angular.y, out.twist.twist.angular.z = (float(c) for c in w)
        else:
            out.twist.twist = msg.twist.twist   # gz plugin: already expressed in the body (heading) frame
        out.twist.covariance = msg.twist.covariance
        self.pub.publish(out)

        if self.tf_br is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.p["odom_frame"]
            tf.child_frame_id = self.p["base_frame"]
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x, y, z
            r = tf.transform.rotation
            r.x, r.y, r.z, r.w = qx, qy, qz, qw
            self.tf_br.sendTransform(tf)
        self._last_rx = time.monotonic()
        self._n_rx += 1

    def _watchdog(self):
        now = time.monotonic()
        stale = float(self.p["stale_warn_s"])
        if self._last_rx is None:
            if now - self._wall_t0 > max(stale, 10.0):
                self.get_logger().warning(
                    f"no odometry received on {self.p['input_topic']} yet (is the robot spawned and the "
                    "ros_gz_bridge running?)", throttle_duration_sec=15.0)
        elif now - self._last_rx > stale:
            self.get_logger().warning(f"odometry stale for {now - self._last_rx:.1f} s",
                                      throttle_duration_sec=10.0)


def main(args=None):
    run_node(OdomTfNode, args)


if __name__ == "__main__":
    main()
