"""
demo_hud node: robot camera + status overlay for the demo (reference video look).

Subscribes
  /inspection_camera/image   sensor_msgs/Image        robot camera (background)
  /gas/sensor                std_msgs/Float32         "현재 가스 농도: X ppm" + log bar
  /battery/state             sensor_msgs/BatteryState battery gauge (percentage 0..1 or 0..100)
  /locomotion/state          std_msgs/String JSON     mode / locomotion_mode / gait, recovery_state +
                                                      fall / recovery counters (robot-down panel)
  /mission/state             std_msgs/String JSON     state, target, progress, gas_alarm, rth_reason,
                                                      completed / missed / results, seek (banners,
                                                      result panel at the dock)
  /metrics/session           std_msgs/String JSON     (latched) session directory = where the
                                                      results were saved (result panel)
  /inspection/gauge_snapshot std_msgs/String JSON     inspection photo of an accepted gauge (corner
                                                      of the 'gauge read' banner)
Publishes (rate Hz)
  <hud_topic>     (default /demo/hud,     bgr8) for ROS tools (rqt_image_view, ros2 bag)
  <gz_hud_topic>  (default /demo/hud_rgb, rgb8) only while somebody subscribes: ros_gz_bridge
                  forwards it (config/ros_gz_bridge.yaml, ROS_TO_GZ) to the Gazebo topic
                  /demo/hud that the GUI ImageDisplay panel "Robot HUD" shows (gz-gui's
                  ImageDisplay does not decode BGR images, hence the rgb8 copy)

The frame itself is drawn by the ROS-free plant_simulation.demo_hud.compose_hud(); the
transient items (banners, robot-down panel, result panel) come from demo_hud.HudStory.
"""
import threading
import time

import numpy as np
import rclpy  # noqa: F401
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Image
from std_msgs.msg import Float32, String

from dusan_common.image_utils import image_msg_to_bgr, numpy_to_image_msg
from dusan_common.ros_helpers import declare_params, parse_json_msg

from plant_simulation.demo_hud import HUD_DEFAULTS, HudCompositor, HudStory
from plant_simulation.node_utils import (best_effort_qos, finite, reliable_qos, run_node, subscribe_any,
                                         subscribe_latched)


class DemoHudNode(Node):
    def __init__(self):
        super().__init__("demo_hud")
        self.p = declare_params(self, HUD_DEFAULTS)
        self.hud = HudCompositor(self.p)
        r = self.hud.renderer
        if r.korean:
            self.get_logger().info(f"HUD text: Korean font {r.font_file}")
        else:
            self.get_logger().warning(f"HUD text in English ASCII ({r.reason or 'language en'}) - for Korean "
                                      "install python3-pil and fonts-noto-cjk (or set font_path)")
        self._lock = threading.Lock()
        self._cam = None                       # (wall time, Image msg)
        self._vals = {}                        # key -> (wall time, value)
        self._warned = set()
        layout = None
        try:                                   # ground truth for the mean reading error of the result panel
            from dusan_common.layout import Layout
            layout = Layout.load(str(self.p["layout_file"]) or None)
        except Exception as exc:
            self.get_logger().warning(f"no layout ({type(exc).__name__}: {exc}) - result panel without reading "
                                      "errors")
        self.story = HudStory(layout, float(self.p["gauge_hold"]), float(self.p["event_hold"]),
                              float(self.p["recovered_hold"]))

        self.pub = self.create_publisher(Image, str(self.p["hud_topic"]), reliable_qos(2))
        gz_topic = str(self.p["gz_hud_topic"]).strip()
        self.pub_gz = self.create_publisher(Image, gz_topic, reliable_qos(2)) if gz_topic else None

        self.create_subscription(Image, "/inspection_camera/image", self._on_camera, best_effort_qos(2))
        self.create_subscription(Float32, "/gas/sensor", lambda m: self._set("gas_ppm", finite(m.data)),
                                 best_effort_qos(10))
        self.create_subscription(BatteryState, "/battery/state", self._on_battery, best_effort_qos(5))
        subscribe_any(self, String, "/locomotion/state", self._on_locomotion)
        subscribe_any(self, String, "/mission/state", self._on_mission)
        subscribe_latched(self, String, "/metrics/session", self._on_session)
        self.create_subscription(String, "/inspection/gauge_snapshot", self._on_snapshot, reliable_qos(5))
        self.create_timer(1.0 / max(0.2, float(self.p["rate"])), self._tick)
        self.get_logger().info(f"HUD {self.p['width']}x{self.p['height']} at {self.p['rate']} Hz on "
                               f"{self.p['hud_topic']}" + (f" (+ rgb8 {gz_topic} for Gazebo)" if gz_topic else ""))

    # ------------------------------------------------------------ callbacks
    def _set(self, key, value):
        with self._lock:
            self._vals[key] = (time.monotonic(), value)

    def _on_camera(self, msg):
        with self._lock:
            self._cam = (time.monotonic(), msg)

    def _on_battery(self, msg):
        p = finite(msg.percentage)
        if p is not None and p <= 1.0:
            p *= 100.0                               # sensor_msgs convention 0..1; tolerate 0..100
        charging = int(msg.power_supply_status) == int(BatteryState.POWER_SUPPLY_STATUS_CHARGING)
        self._set("battery", (p, charging))

    def _on_locomotion(self, msg):
        d = parse_json_msg(msg)
        if isinstance(d, dict):
            self._set("loco", (str(d.get("mode") or ""), str(d.get("locomotion_mode") or ""),
                               str(d.get("gait") or "")))
            with self._lock:
                self.story.on_locomotion(d, time.monotonic())

    def _on_mission(self, msg):
        d = parse_json_msg(msg)
        if isinstance(d, dict):
            self._set("mission", d)
            with self._lock:
                self.story.on_mission(d, time.monotonic())

    def _on_session(self, msg):
        d = parse_json_msg(msg)
        with self._lock:
            self.story.on_session(d)

    def _on_snapshot(self, msg):
        d = parse_json_msg(msg)
        with self._lock:
            ok = self.story.on_snapshot(d)
        if not ok and isinstance(d, dict) and "snapshot" not in self._warned:
            self._warned.add("snapshot")
            self.get_logger().warning(f"gauge snapshot {d.get('id')} without a decodable photo - banner without it")

    # ----------------------------------------------------------------- frame
    def _fresh(self, key, now):
        with self._lock:
            item = self._vals.get(key)
        if item is None or now - item[0] > float(self.p["status_timeout"]):
            return None
        return item[1]

    def status(self, now=None):
        """EMPTY_STATUS-style dict from the latest (fresh) inputs."""
        now = time.monotonic() if now is None else now
        st = {"gas_ppm": self._fresh("gas_ppm", now)}
        batt = self._fresh("battery", now)
        if batt is not None:
            st["battery_pct"], st["charging"] = batt
        loco = self._fresh("loco", now)
        if loco is not None:
            st["loco_mode"], st["locomotion_mode"], st["gait"] = loco
        m = self._fresh("mission", now)
        if m is not None:
            st["mission_state"] = str(m.get("state") or "")
            st["mission_target"] = str(m.get("target") or "")
            st["mission_progress"] = m.get("progress")
            st["gas_alarm"] = bool(m.get("gas_alarm"))
        with self._lock:
            st.update(self.story.status(now, mission=m is not None, loco=loco is not None))
        st["sim_time"] = self.get_clock().now().nanoseconds * 1e-9
        return st

    def _camera_bgr(self, now):
        with self._lock:
            cam = self._cam
        if cam is None or now - cam[0] > float(self.p["camera_timeout"]):
            return None
        try:
            return image_msg_to_bgr(cam[1])
        except Exception as exc:  # unsupported encoding / truncated data
            if "camera" not in self._warned:
                self._warned.add("camera")
                self.get_logger().warning(f"cannot decode the camera image ({cam[1].encoding}): {exc}")
            return None

    def _tick(self):
        now = time.monotonic()
        try:
            frame = self.hud.compose(self._camera_bgr(now), self.status(now), t=now)
        except Exception as exc:  # never let a drawing problem kill the node
            self.get_logger().error(f"HUD compose failed: {type(exc).__name__}: {exc}",
                                    throttle_duration_sec=10.0)
            return
        stamp = self.get_clock().now().to_msg()
        self.pub.publish(numpy_to_image_msg(frame, "bgr8", "demo_hud", stamp))
        if self.pub_gz is not None and self._has_subscribers(self.pub_gz):
            rgb = np.ascontiguousarray(frame[..., ::-1])
            self.pub_gz.publish(numpy_to_image_msg(rgb, "rgb8", "demo_hud", stamp))

    @staticmethod
    def _has_subscribers(pub):
        fn = getattr(pub, "get_subscription_count", None)
        if fn is None:
            return True
        try:
            return int(fn()) > 0
        except Exception:
            return True


def main(args=None):
    run_node(DemoHudNode, args)


if __name__ == "__main__":
    main()
