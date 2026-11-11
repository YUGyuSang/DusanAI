"""
metrics_logger node: records a run to ~/dusan_metrics/<YYYYmmdd_HHMMSS>/
(events.csv, gauge_readings.csv, thermal.csv, gas.csv, battery.csv, pose.csv,
summary.json, gauges/*.jpg, inspection_report.json / .html) using the ROS-free
plant_simulation.metrics.MetricsRecorder.

summary.json is rewritten every `summary_period` s (wall clock, so it keeps
working when the simulation is paused) and on shutdown.  The evidence photos of
/inspection/gauge_snapshot go to gauges/; the inspection report is rewritten
whenever a gauge result or photo arrives and at mission end.

/metrics/session (std_msgs/String JSON, latched) tells the dashboard where the
session lives and which photo / report files exist
(MetricsRecorder.session_info(): {session_dir, session, report, report_json,
report_final, report_written, gauges: {id: {image, image_full, stamp, value,
unit, status, wall}}}); republished after every photo / report update.
"""
import os

import rclpy  # noqa: F401
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, String

from dusan_common.geometry import odom_to_pose
from dusan_common.layout import Layout
from dusan_common.ros_helpers import declare_params, json_msg, latched_qos, parse_json_msg, stamp_to_sec

from plant_simulation.metrics import METRICS_DEFAULTS, MetricsRecorder, make_session_dir
from plant_simulation.node_utils import best_effort_qos, guarded, now_sec, reliable_qos, run_node, subscribe_any


def _wall_timer(node, period, callback):
    try:
        from rclpy.clock import Clock, ClockType
        return node.create_timer(period, callback, clock=Clock(clock_type=ClockType.STEADY_TIME))
    except TypeError:
        return node.create_timer(period, callback)


class MetricsLoggerNode(Node):
    def __init__(self):
        super().__init__("metrics_logger")
        self.p = declare_params(self, METRICS_DEFAULTS)
        layout = Layout.load(self.p["layout_file"] or None)
        out = make_session_dir(self.p["output_dir"], self.p["session_name"])
        self.rec = MetricsRecorder(out, layout, self.p)
        R = self.rec
        t = lambda: now_sec(self)  # noqa: E731

        self.create_subscription(Odometry, "/odom", self._on_odom, best_effort_qos(10))
        subscribe_any(self, String, "/locomotion/state", lambda m: R.on_locomotion(t(), parse_json_msg(m)))
        self.create_subscription(String, "/locomotion/fall_event", lambda m: R.on_fall_event(t(), parse_json_msg(m)),
                                 reliable_qos(20))
        subscribe_any(self, String, "/mission/state", lambda m: R.on_mission(t(), parse_json_msg(m)))
        subscribe_any(self, String, "/inspection/target", lambda m: R.on_target(t(), m.data))
        # readings / alerts are rows -> single reliable subscription (no duplicates)
        self.create_subscription(String, "/inspection/gauge_readings",
                                 lambda m: R.on_gauge_readings(t(), parse_json_msg(m)), reliable_qos(50))
        # one evidence photo per inspection visit (base64 JPEG, ~100 KiB): reliable, no duplicates
        self.create_subscription(String, "/inspection/gauge_snapshot",
                                 guarded(self, "/inspection/gauge_snapshot",
                                         lambda m: R.on_gauge_snapshot(t(), parse_json_msg(m))), reliable_qos(10))
        self.create_subscription(String, "/inspection/thermal_alerts",
                                 lambda m: R.on_thermal(t(), parse_json_msg(m)), reliable_qos(20))
        self.create_subscription(Float32, "/gas/sensor", lambda m: R.on_gas(t(), m.data), best_effort_qos(10))
        subscribe_any(self, String, "/gas/seek_status", lambda m: R.on_seek_status(t(), parse_json_msg(m)))
        self.create_subscription(PointStamped, "/gas/source_estimate",
                                 lambda m: R.on_source_estimate(t(), m.point.x, m.point.y), best_effort_qos(5))
        self.create_subscription(BatteryState, "/battery/state", self._on_battery, best_effort_qos(5))
        self.create_subscription(Float32, "/comm/rssi", lambda m: R.on_rssi(t(), m.data), best_effort_qos(5))
        subscribe_any(self, String, "/safety/status", lambda m: R.on_safety(t(), parse_json_msg(m)))
        subscribe_any(self, Bool, "/safety/rth_active", lambda m: R.on_rth_active(t(), m.data))

        self.pub_session = self.create_publisher(String, "/metrics/session", latched_qos())
        R.on_change = self._publish_session
        self._publish_session()

        _wall_timer(self, max(0.5, float(self.p["flush_period"])), R.flush)
        _wall_timer(self, max(1.0, float(self.p["summary_period"])), self._write_summary)
        R.event(None, "metrics", "start", f"session {out}")
        self.get_logger().info(f"recording metrics to {out}")

    def _publish_session(self):
        self.pub_session.publish(json_msg(self.rec.session_info()))

    def _on_odom(self, msg):
        x, y, z, roll, pitch, yaw = odom_to_pose(msg)
        tw = msg.twist.twist
        ts = stamp_to_sec(msg.header.stamp) or now_sec(self)
        self.rec.on_pose(ts, x, y, z, roll, pitch, yaw, tw.linear.x, tw.linear.y, tw.angular.z)

    def _on_battery(self, msg):
        p = float(msg.percentage)
        pct = p * 100.0 if p <= 1.0 else p
        charging = int(msg.power_supply_status) == int(BatteryState.POWER_SUPPLY_STATUS_CHARGING)
        self.rec.on_battery(now_sec(self), pct, msg.voltage, charging)

    def _write_summary(self):
        try:
            self.rec.write_summary()
        except OSError as exc:
            self.get_logger().error(f"cannot write summary.json: {exc}", throttle_duration_sec=30.0)

    def on_shutdown(self):
        try:
            t = now_sec(self)
        except Exception:  # context already shut down
            t = self.rec.t_last
        self.rec.event(t, "metrics", "stop", "")
        self.rec.on_change = None             # the context may already be shut down: no publishing
        self.rec.close()
        print(f"[metrics_logger] metrics written to {self.rec.dir}", flush=True)
        if self.rec.report_written:
            print(f"[metrics_logger] inspection report: {os.path.join(self.rec.dir, 'inspection_report.html')}",
                  flush=True)


def main(args=None):
    run_node(MetricsLoggerNode, args)


if __name__ == "__main__":
    main()
