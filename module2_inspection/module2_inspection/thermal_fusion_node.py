"""
ROS 2 node ``thermal_fusion`` - thin wrapper around ``thermal_fusion.ThermalFusion``.

Subscribes
    /thermal_camera/image     sensor_msgs/Image  mono16 | 16UC1 | L16 (Kelvin x 100), L8, 32FC1
    /inspection_camera/image  sensor_msgs/Image  rgb8 / bgr8 (frame nearest to the thermal stamp)
    /odom                     nav_msgs/Odometry  world-frame ground truth pose of base_link
Publishes
    /inspection/thermal_alerts  std_msgs/String (JSON, DESIGN_SPEC section 4)
    /inspection/thermal_fusion  sensor_msgs/Image bgr8 (colour-mapped thermal on RGB)

Processing is driven by the thermal frames (5 Hz).  The RGB frame with the
nearest header stamp and the odometry pose interpolated at the thermal stamp
are used (``StampedBuffer``; receive time when a stamp is zero).  Without RGB
the node still publishes alerts and a thermal-only visualisation; without
odometry hot spots are reported without id / position.
"""
import array
import math
import time

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String

from dusan_common.geometry import odom_to_pose
from dusan_common.image_utils import image_msg_to_bgr, image_msg_to_numpy
from dusan_common.layout import Layout
from dusan_common.ros_helpers import json_msg, sensor_qos, stamp_to_sec
from module2_inspection.thermal_fusion import FusionParams, StampedBuffer, ThermalFusion, raw_to_kelvin

try:  # Jazzy and later: spin() raises this when the context is shut down (e.g. SIGINT from launch)
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover - older rclpy
    class ExternalShutdownException(Exception):
        """Placeholder for rclpy versions without ExternalShutdownException."""

# ROS parameter defaults (all tunables; config/thermal.yaml overrides them)
NODE_PARAMS = {
    "layout_file": "",
    "thermal_topic": "/thermal_camera/image",
    "rgb_topic": "/inspection_camera/image",
    "odom_topic": "/odom",
    "alerts_topic": "/inspection/thermal_alerts",
    "fusion_topic": "/inspection/thermal_fusion",
    "publish_fusion": True,
    "fusion_scale": 1.0,
    "process_every_n": 1,
    "rgb_sync_tolerance_s": 0.15,   # max |t_rgb - t_thermal| (RGB at 10 Hz)
    "odom_sync_tolerance_s": 0.25,  # max gap for pose interpolation / nearest pose
    "sync_fallback_age_s": 0.5,     # stamps inconsistent -> latest message if received within this age
    "thermal_resolution": 0.01,
    "l8_min_k": 253.15,
    "l8_max_k": 655.0,
    "no_image_warn_s": 8.0,
}
_FP = FusionParams()
FUSION_PARAMS = {
    "ambient_k": _FP.ambient_k, "warning_k": _FP.warning_k, "alarm_k": _FP.alarm_k,
    "segmentation_k": _FP.segmentation_k, "min_area_px": _FP.min_area_px, "open_px": _FP.open_px,
    "plane_mode": _FP.plane_mode, "depth_mode": _FP.depth_mode, "plane_distance": _FP.plane_distance,
    "depth_min": _FP.depth_min, "depth_max": _FP.depth_max,
    "ecc_enabled": _FP.ecc_enabled, "ecc_every_n": _FP.ecc_every_n, "ecc_scales": list(_FP.ecc_scales),
    "ecc_motion": _FP.ecc_motion, "ecc_max_iter": _FP.ecc_max_iter, "ecc_eps": _FP.ecc_eps,
    "ecc_max_shift_px": _FP.ecc_max_shift_px, "ecc_max_vertical_px": _FP.ecc_max_vertical_px,
    "ecc_min_cc": _FP.ecc_min_cc, "ecc_smoothing": _FP.ecc_smoothing, "ecc_epipolar": _FP.ecc_epipolar,
    "ecc_skip_hotspot_plane": _FP.ecc_skip_hotspot_plane,
    "ecc_reset_depth_change": _FP.ecc_reset_depth_change,
    "match_tolerance_m": _FP.match_tolerance_m, "match_tolerance_rel": _FP.match_tolerance_rel,
    "max_match_range_m": _FP.max_match_range_m, "report_normal_in_view": _FP.report_normal_in_view,
    "roi_min_excess_k": _FP.roi_min_excess_k, "roi_max_range_m": _FP.roi_max_range_m,
    "confirm_frames": _FP.confirm_frames, "max_misses": _FP.max_misses, "max_gap_s": _FP.max_gap_s,
    "track_radius_m": _FP.track_radius_m, "track_radius_px": _FP.track_radius_px,
    "blend_delta_k": _FP.blend_delta_k, "blend_ramp_k": _FP.blend_ramp_k, "alpha_max": _FP.alpha_max,
    "display_min_k": _FP.display_min_k, "display_max_k": _FP.display_max_k,
    "draw_isotherms": _FP.draw_isotherms, "draw_legend": _FP.draw_legend,
}


def bgr8_image_msg(bgr: np.ndarray, frame_id: str, stamp) -> Image:
    """BGR uint8 array -> sensor_msgs/Image (bgr8).

    ``data`` is assigned as ``array.array('B')``: the generated setter of a
    ``uint8[]`` field accepts it directly, whereas assigning ``bytes`` makes
    the setter validate every element in Python (~0.2 s for an 800 x 600 x 3
    image, i.e. a whole CPU core at 5 Hz)."""
    img = np.ascontiguousarray(bgr, dtype=np.uint8)
    msg = Image()
    msg.header.frame_id = str(frame_id or "")
    msg.header.stamp = stamp
    msg.height = int(img.shape[0])
    msg.width = int(img.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(img.shape[1] * 3)
    msg.data = array.array("B", img.tobytes())
    return msg


class ThermalFusionNode(Node):
    def __init__(self):
        super().__init__("thermal_fusion")
        self.cfg = {}
        for name, default in {**NODE_PARAMS, **FUSION_PARAMS}.items():
            self.declare_parameter(name, default)
            self.cfg[name] = self.get_parameter(name).value
        cfg = self.cfg

        layout = Layout.load(cfg["layout_file"] or None)
        fp = FusionParams.from_dict({k: cfg[k] for k in FUSION_PARAMS})
        fp.resolution = float(cfg["thermal_resolution"])
        self.fusion = ThermalFusion(layout, fp)

        self.rgb_buf = StampedBuffer(maxlen=8)      # (bgr, frame_id) by stamp
        self.pose_buf = StampedBuffer(maxlen=100)   # (x, y, z, roll, pitch, yaw) by stamp
        self.rgb_latest = (None, None)               # (receive time, item) for the stamp fallback
        self.pose_latest = (None, None)
        self.n_thermal = 0
        self.last_thermal_rx = None
        self.alarmed = set()
        # wall-clock start: the watchdog must also fire while /clock is not running yet
        self.t_start = time.monotonic()

        qos = sensor_qos()
        self.create_subscription(Image, cfg["thermal_topic"], self.on_thermal, qos)
        self.create_subscription(Image, cfg["rgb_topic"], self.on_rgb, qos)
        self.create_subscription(Odometry, cfg["odom_topic"], self.on_odom, qos)
        self.pub_alerts = self.create_publisher(String, cfg["alerts_topic"], 10)
        self.pub_fusion = self.create_publisher(Image, cfg["fusion_topic"], 2)
        # steady-clock timer: with use_sim_time a ROS-time timer would never fire
        # while /clock is missing - exactly the situation the watchdog reports
        self.create_timer(2.0, self.on_watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"thermal_fusion ready: warning {self.fusion.warning_k - 273.15:.1f} C, alarm "
            f"{self.fusion.alarm_k - 273.15:.1f} C, plane {fp.plane_mode}/{fp.depth_mode}, ECC "
            f"{'on' if fp.ecc_enabled else 'off'}, {len(self.fusion.refs)} layout hot spots")

    # ------------------------------------------------------------ inputs
    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp(self, header) -> float:
        """Header stamp in seconds (receive time if the publisher left it zero)."""
        t = stamp_to_sec(header.stamp)
        return t if t > 0.0 else self._now_s()

    def on_rgb(self, msg: Image):
        try:
            item = (image_msg_to_bgr(msg), msg.header.frame_id)
            self.rgb_buf.add(self._stamp(msg.header), item)
            self.rgb_latest = (self._now_s(), item)
        except Exception as exc:  # unsupported encoding etc.
            self.get_logger().warning(f"inspection image ignored: {exc}", throttle_duration_sec=10.0)

    def on_odom(self, msg: Odometry):
        pose = odom_to_pose(msg)
        self.pose_buf.add(self._stamp(msg.header), pose)
        self.pose_latest = (self._now_s(), pose)

    def on_thermal(self, msg: Image):
        self.n_thermal += 1
        self.last_thermal_rx = self._now_s()
        if (self.n_thermal - 1) % max(1, int(self.cfg["process_every_n"])) != 0:
            return
        try:
            raw = image_msg_to_numpy(msg)
            kelvin = raw_to_kelvin(raw, msg.encoding, float(self.cfg["thermal_resolution"]),
                                   (float(self.cfg["l8_min_k"]), float(self.cfg["l8_max_k"])))
        except Exception as exc:
            self.get_logger().warning(f"thermal image ignored: {exc}", throttle_duration_sec=10.0)
            return
        stamp = self._stamp(msg.header)
        rgb_item = self.rgb_buf.nearest(stamp, float(self.cfg["rgb_sync_tolerance_s"]))
        pose = self.pose_buf.interpolate_pose(stamp, float(self.cfg["odom_sync_tolerance_s"]))
        # fallback when the publishers use inconsistent time bases (e.g. zero / wall-clock stamps)
        max_age = float(self.cfg["sync_fallback_age_s"])
        rx = self.last_thermal_rx
        if rgb_item is None and self.rgb_latest[0] is not None and rx - self.rgb_latest[0] <= max_age:
            rgb_item = self.rgb_latest[1]
            self._warn_sync("inspection image", self.rgb_buf.latest()[0], stamp)
        if pose is None and self.pose_latest[0] is not None and rx - self.pose_latest[0] <= max_age:
            pose = self.pose_latest[1]
            self._warn_sync("odometry", self.pose_buf.latest()[0], stamp)
        rgb, rgb_frame = rgb_item if rgb_item is not None else (None, None)
        try:
            res = self.fusion.process(kelvin, rgb, pose, stamp, render=bool(self.cfg["publish_fusion"]))
        except Exception as exc:  # never let one bad frame kill the node
            self.get_logger().error(f"thermal fusion failed: {exc}", throttle_duration_sec=5.0)
            return
        alerts = self.fusion.alerts_dict(res, stamp)
        alerts["robot_pose"] = None if pose is None else [round(pose[0], 3), round(pose[1], 3), round(pose[5], 4)]
        alerts["rgb_available"] = rgb is not None
        self.pub_alerts.publish(json_msg(alerts))
        for h in alerts["hotspots"]:
            key = h["id"] or "unknown"
            if h["status"] == "alarm" and key not in self.alarmed:
                self.alarmed.add(key)
                self.get_logger().warning(f"THERMAL ALARM {key} {h.get('tag')}: {h['max_temp_c']:.1f} C at {h['position']}")
        if res.fusion is not None and self.cfg["publish_fusion"]:
            img = res.fusion
            sc = float(self.cfg["fusion_scale"])
            if 0.05 < sc < 0.999:
                import cv2
                img = cv2.resize(img, (max(1, int(img.shape[1] * sc)), max(1, int(img.shape[0] * sc))),
                                 interpolation=cv2.INTER_AREA)
            frame = rgb_frame if rgb is not None else msg.header.frame_id
            if img.ndim == 2:      # defensive: the renderer always returns BGR
                img = np.repeat(img[..., None], 3, axis=2)
            self.pub_fusion.publish(bgr8_image_msg(img, frame, msg.header.stamp))

    def _warn_sync(self, what: str, t_other, t_thermal: float):
        dt = math.nan if t_other is None else t_other - t_thermal
        self.get_logger().warning(
            f"{what} stamps differ from the thermal stamps by {dt:.2f} s - using the latest message "
            "(check use_sim_time / header stamps)", throttle_duration_sec=30.0)

    def on_watchdog(self):
        waited = time.monotonic() - self.t_start
        if self.n_thermal == 0 and waited > float(self.cfg["no_image_warn_s"]):
            self.get_logger().warning(
                f"no thermal image on {self.cfg['thermal_topic']} yet - is the Gazebo thermal camera bridged? "
                "(launch with thermal_source:=synthetic to use the synthetic thermal camera)",
                throttle_duration_sec=30.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ThermalFusionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
