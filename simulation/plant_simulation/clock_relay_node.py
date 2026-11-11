"""
clock_relay node: decimated simulation clock for the rclpy nodes.

Subscribes  /clock_gz (rosgraph_msgs/Clock) - the Gazebo clock as bridged by
            ros_gz_bridge (sim.launch.py remaps the bridge's /clock to it);
            1000 msg/s with the 1 ms physics step
Publishes   /clock (rosgraph_msgs/Clock) at `rate` Hz of sim time (200 Hz:
            times aligned to 5 ms, see plant_simulation.clock_relay)

Must run with use_sim_time false (it is the clock source).  It is a single
subscription + publisher without parameter services, so this one process
is the only Python process that handles the full 1 kHz stream.
"""
import time

import rclpy  # noqa: F401  (imported here so a missing ROS install fails loudly)
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

from dusan_common.ros_helpers import declare_params

from plant_simulation.clock_relay import CLOCK_RELAY_DEFAULTS, ClockDecimator
from plant_simulation.node_utils import best_effort_qos, reliable_qos, run_node


class ClockRelayNode(Node):
    def __init__(self):
        # no parameter services: they would add six services to the wait set
        # this process rebuilds 1000 times per second (overrides still apply)
        super().__init__("clock_relay", start_parameter_services=False)
        self.p = declare_params(self, CLOCK_RELAY_DEFAULTS)
        self.dec = ClockDecimator(self.p["rate"], self.p["keepalive"])
        # reliable publisher: matches best-effort (rclpy/rclcpp TimeSource) and reliable subscribers
        self.pub = self.create_publisher(Clock, "/clock", reliable_qos(1))
        # best effort, depth 1: always the newest sample, matches any bridge QoS
        self.create_subscription(Clock, "/clock_gz", self._on_clock, best_effort_qos(1))
        self._log_period = float(self.p["log_period"])
        self._next_log = time.monotonic() + self._log_period
        self._first = True
        rate = float(self.p["rate"])
        self.get_logger().info(
            "clock relay: /clock_gz -> /clock " + (f"at {rate:.0f} Hz of sim time" if rate > 0.0 else "(every sample)"))

    def _on_clock(self, msg):
        wall = time.monotonic()
        ns = int(msg.clock.sec) * 1000000000 + int(msg.clock.nanosec)
        if self.dec.accept(ns, wall):
            self.pub.publish(msg)
            if self._first:
                self._first = False
                self.get_logger().info(f"first clock sample relayed: t = {ns * 1e-9:.3f} s")
        if self._log_period > 0.0 and wall >= self._next_log:
            self._next_log = wall + self._log_period
            self.get_logger().info(f"clock relay: {self.dec.n_in} samples in, {self.dec.n_out} out")


def main(args=None):
    run_node(ClockRelayNode, args)


if __name__ == "__main__":
    main()
