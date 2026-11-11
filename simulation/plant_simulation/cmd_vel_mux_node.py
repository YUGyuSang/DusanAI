"""
cmd_vel_mux node: priority multiplexer for the body velocity command.

Subscribes  /cmd_vel/teleop /cmd_vel/rth /cmd_vel/seek /cmd_vel/patrol (Twist)
            /teleop/active /safety/rth_active (Bool)
Publishes   /cmd_vel (Twist, fixed rate, zero when nothing is fresh)
            /cmd_vel_mux/active (String: teleop|rth|seek|patrol|none)

The selection logic lives in :mod:`plant_simulation.mux_logic` (ROS free).
"""
import rclpy  # noqa: F401  (imported here so a missing ROS install fails loudly)
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

from dusan_common.ros_helpers import declare_params, latched_qos

from plant_simulation.mux_logic import MUX_DEFAULTS, CmdVelMux, SOURCE_PRIORITY
from plant_simulation.node_utils import best_effort_qos, now_sec, reliable_qos, run_node, subscribe_any

INPUT_TOPICS = {
    "teleop": "/cmd_vel/teleop",
    "rth": "/cmd_vel/rth",
    "seek": "/cmd_vel/seek",
    "patrol": "/cmd_vel/patrol",
}


class CmdVelMuxNode(Node):
    def __init__(self):
        super().__init__("cmd_vel_mux")
        self.p = declare_params(self, MUX_DEFAULTS)
        self.mux = CmdVelMux.from_params(self.p)

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", reliable_qos(10))
        self.pub_active = self.create_publisher(String, "/cmd_vel_mux/active", latched_qos())

        for source in SOURCE_PRIORITY:
            # best effort matches reliable and best-effort publishers alike
            self.create_subscription(Twist, INPUT_TOPICS[source],
                                     lambda m, s=source: self._on_cmd(s, m), best_effort_qos(5))
        # flags: idempotent -> subscribe with reliable AND best-effort QoS so
        # that any publisher QoS matches (volatile durability on our side also
        # matches latched publishers)
        subscribe_any(self, Bool, "/teleop/active", lambda m: self._on_flag("teleop_active", m))
        subscribe_any(self, Bool, "/safety/rth_active", lambda m: self._on_flag("rth_active", m))

        self._active = None
        self._t_active_pub = -1e9
        rate = max(1.0, float(self.p["rate"]))
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"cmd_vel mux up: priorities {' > '.join(SOURCE_PRIORITY)}, timeout {self.p['timeout']:.2f} s, "
            f"{rate:.0f} Hz on /cmd_vel")

    # ------------------------------------------------------------- callbacks
    def _on_cmd(self, source, msg):
        self.mux.update(source, (msg.linear.x, msg.linear.y, msg.angular.z), now_sec(self))

    def _on_flag(self, name, msg):
        self.mux.set_flag(name, bool(msg.data), now_sec(self))

    def _tick(self):
        t = now_sec(self)
        source, (vx, vy, wz) = self.mux.select(t)
        out = Twist()
        out.linear.x, out.linear.y, out.angular.z = float(vx), float(vy), float(wz)
        self.pub_cmd.publish(out)
        if source != self._active or (t - self._t_active_pub) >= float(self.p["active_publish_period"]) \
                or t < self._t_active_pub:
            if source != self._active:
                self.get_logger().info(f"cmd_vel source: {self._active} -> {source}")
            self._active = source
            self._t_active_pub = t
            self.pub_active.publish(String(data=source))


def main(args=None):
    run_node(CmdVelMuxNode, args)


if __name__ == "__main__":
    main()
