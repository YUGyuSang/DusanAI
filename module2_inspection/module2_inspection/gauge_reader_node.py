"""
ROS 2 node ``gauge_reader`` - analog gauge reading for the plant-inspection robot.

Thin wrapper around the ROS-free modules
    module2_inspection.gauge_reader       (detection, rectification, needle, OCR, fusion)
    module2_inspection.gauge_association  (camera projection of the layout gauges, ids)

Subscriptions
    /inspection_camera/image        sensor_msgs/Image (rgb8 / bgr8 / mono8 ...)
    /inspection_camera/camera_info  sensor_msgs/CameraInfo (optional, intrinsics)
    /odom                           nav_msgs/Odometry  (robot pose -> gauge ROI + id)
    /imu                            sensor_msgs/Imu    (frames are skipped while |omega| is large)
    /inspection/target              std_msgs/String    (gauge id expected in view, or "")
Publications
    /inspection/gauge_readings      std_msgs/String JSON
        {stamp, readings: [{id, tag, value, unit, confidence, angle_deg, range: [min, max],
                            status, n_samples, robot_pose: [x, y, yaw], ...extras}]}
        (latest fused reading of every gauge seen in this frame).  Registry gauges carry the
        layout range (an OCR range that disagrees is reported in ``range_ocr`` and logged);
        an unidentified dial (no pose match, no target) is published with ``id: null``, its
        range / value from OCR only (null if the labels cannot be read).
    /inspection/gauge_overlay       sensor_msgs/Image bgr8 (ellipse, needle, value, status, id)

Per frame (timer, <= ``process_rate_hz``, newest image only; <= ``idle_process_rate_hz`` while
/inspection/target is empty): skip while the IMU says the body rotates faster than
``max_angular_velocity`` (motion blur from shake); odometry pose at the image stamp ->
GaugeAssociator hints (ROI, registry range, dial normal) -> GaugeReader in the ROIs ->
whole-frame search (expensive, CPU shared with Gazebo) when there is no pose or the
/inspection/target gauge was not found, otherwise every ``full_search_period_s`` (predicted gauge
missed, unregistered dials), never more often than ``full_search_min_period_s`` -> id assignment
(gated nearest neighbour, or the /inspection/target gauge without a pose) -> GaugeFusion
(median / MAD over N samples / T seconds) -> JSON + overlay.  Frames without any search (no
gauge predicted in view, no full search due) skip the enhancement; their overlay (and that of
motion-skipped frames) is published at most every ``overlay_idle_period_s``.  OpenCV is limited to
``opencv_threads`` worker threads so it does not oversubscribe the cores Gazebo needs.

QoS: images / camera info / IMU / odometry are subscribed best effort (matches reliable and
best-effort publishers); /inspection/target is subscribed twice, transient-local (the mission
manager latches it) and volatile (manual ``ros2 topic pub``); readings and overlay are
published reliable / volatile.
"""
import math
import time
from collections import deque
from dataclasses import fields

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import String

from dusan_common import gauge_design as gd
from dusan_common.geometry import odom_to_pose
from dusan_common.image_utils import bgr_to_image_msg, image_msg_to_bgr
from dusan_common.layout import Layout
from dusan_common.ros_helpers import json_msg, latched_qos, sensor_qos, stamp_to_sec

from module2_inspection.gauge_association import GaugeAssociator
from module2_inspection.gauge_reader import (GaugeFusion, GaugeHint, GaugeReader, GaugeReaderConfig,
                                             draw_header, draw_reading)

try:  # Jazzy and later: spin() raises this when the context is shut down (e.g. SIGINT from launch)
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover - older rclpy
    class ExternalShutdownException(Exception):
        """Placeholder for rclpy versions without ExternalShutdownException."""

NODE_DEFAULTS = {
    "layout_file": "",
    "image_topic": "/inspection_camera/image",
    "camera_info_topic": "/inspection_camera/camera_info",
    "odom_topic": "/odom",
    "imu_topic": "/imu",
    "target_topic": "/inspection/target",
    "readings_topic": "/inspection/gauge_readings",
    "overlay_topic": "/inspection/gauge_overlay",
    "process_rate_hz": 5.0,
    "idle_process_rate_hz": 2.0,     # ... and at most this while /inspection/target is empty (<= 0: off)
    "opencv_threads": 2,             # OpenCV worker threads (<= 0: OpenCV default = all cores)
    "camera_hfov": 0.9,
    "use_camera_info": True,
    "max_angular_velocity": 0.6,     # rad/s - skip frames while the body rotates faster
    "imu_window_s": 0.15,
    "odom_max_age_s": 0.5,
    "pose_sigma_px": 30.0,
    "max_incidence_deg": 70.0,
    "min_radius_px": 12.0,
    "max_distance": 6.0,
    "use_map_occlusion": False,
    "detect_unhinted": True,         # also search the whole frame (unknown / mis-predicted dials)
    "full_search_period_s": 2.0,     # ... at most this often while every hinted gauge is found
    "full_search_min_period_s": 0.5,  # ... never more often, even without pose / target not found
    "publish_unknown": True,         # unidentified dials are published with id null
    "publish_empty": False,
    "fusion_window_n": 15,
    "fusion_window_s": 10.0,
    "fusion_mad_k": 3.0,
    "ocr_period_s": 2.0,             # OCR re-check period per gauge until the range is confirmed
    "ocr_confirmed_period_s": 30.0,  # ... and after it was confirmed twice
    "overlay_enabled": True,
    "overlay_enhanced": True,        # draw on the enhanced (brightened) frame
    "overlay_scale": 1.0,
    "overlay_idle_period_s": 0.5,    # overlay period of frames without any search / motion-skipped
    "stats_period_s": 30.0,          # processing statistics in the log (0 = off)
}


def value_from_fraction(fraction, vrange, margin=0.04):
    """Reading on a known scale from the needle position along the 270 deg arc."""
    f = float(np.clip(fraction, -margin, 1.0 + margin))
    return gd.angle_to_value(gd.PHI_MIN_DEG - f * gd.SWEEP_DEG, vrange[0], vrange[1])


class GaugeReaderNode(Node):
    def __init__(self):
        super().__init__("gauge_reader")
        self.p = {}
        for name, default in NODE_DEFAULTS.items():
            self.declare_parameter(name, default)
            self.p[name] = self.get_parameter(name).value
        # reader tunables live under 'reader.' (nested 'reader:' block in the YAML)
        rcfg = {}
        for f in fields(GaugeReaderConfig):
            default = getattr(GaugeReaderConfig(), f.name)
            self.declare_parameter("reader." + f.name, default)
            rcfg[f.name] = self.get_parameter("reader." + f.name).value
        if int(self.p["opencv_threads"]) > 0:
            # OpenCV's default pool spans every core; Gazebo and the controllers need them too
            cv2.setNumThreads(int(self.p["opencv_threads"]))
        self.reader = GaugeReader(GaugeReaderConfig.from_dict(rcfg))

        self.layout = Layout.load(self.p["layout_file"] or None)
        occ = None
        if self.p["use_map_occlusion"]:
            try:
                from dusan_common.grid_map import OccupancyGridMap
                occ = OccupancyGridMap.load(self.layout.nav_map_yaml())
            except Exception as e:  # map is optional
                self.get_logger().warning(f"nav map not available for line-of-sight checks: {e}")
        self.assoc = GaugeAssociator.from_layout(
            self.layout, max_incidence_deg=float(self.p["max_incidence_deg"]),
            min_radius_px=float(self.p["min_radius_px"]), max_distance=float(self.p["max_distance"]),
            occupancy_map=occ)
        self.fusion = GaugeFusion(int(self.p["fusion_window_n"]), float(self.p["fusion_window_s"]),
                                  float(self.p["fusion_mad_k"]))

        self._img_msg = None
        self._img_new = False
        self._cam_info = None
        self._intr_key = None
        self._odom = deque(maxlen=200)       # (t, pose6)
        self._imu = deque(maxlen=400)        # (t, |omega|)
        self._target = ""
        self._ocr_state = {}                 # gid -> {"t": last attempt, "agree": n}
        self._stats = {"processed": 0, "skipped_motion": 0, "errors": 0, "full_searches": 0,
                       "readings": 0, "ms_sum": 0.0, "ms_max": 0.0}
        self._last_full_search = -1e9
        self._last_processed = -1e9          # image stamp of the last processed frame
        self._last_overlay = -1e9            # image stamp of the last published overlay
        self._last_status = {}

        qos = QoSProfile(depth=10)
        self.create_subscription(Image, self.p["image_topic"], self._on_image, sensor_qos())
        if self.p["use_camera_info"]:
            self.create_subscription(CameraInfo, self.p["camera_info_topic"], self._on_info, sensor_qos())
        self.create_subscription(Odometry, self.p["odom_topic"], self._on_odom, sensor_qos())
        self.create_subscription(Imu, self.p["imu_topic"], self._on_imu, sensor_qos())
        # the mission manager latches /inspection/target (transient local); a volatile twin
        # subscription also accepts plain publishers (both callbacks are idempotent)
        self.create_subscription(String, self.p["target_topic"], self._on_target, latched_qos())
        self.create_subscription(String, self.p["target_topic"], self._on_target, qos)
        self.pub_readings = self.create_publisher(String, self.p["readings_topic"], qos)
        self.pub_overlay = self.create_publisher(Image, self.p["overlay_topic"], qos)
        rate = max(0.2, float(self.p["process_rate_hz"]))
        self.create_timer(1.0 / rate, self._process)
        if float(self.p["stats_period_s"]) > 0.0:
            self.create_timer(float(self.p["stats_period_s"]), self._log_stats)
        self.get_logger().info(
            f"gauge_reader: {len(self.layout.gauges)} registry gauges, <= {rate:.1f} Hz, "
            f"image {self.p['image_topic']}, IMU gate {self.p['max_angular_velocity']} rad/s")

    # ------------------------------------------------------------ callbacks
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _stamp(msg_stamp, fallback: float) -> float:
        t = stamp_to_sec(msg_stamp)
        return t if t > 0.0 else fallback

    def _on_image(self, msg: Image):
        self._img_msg = msg
        self._img_new = True

    def _on_info(self, msg: CameraInfo):
        self._cam_info = msg

    def _on_odom(self, msg: Odometry):
        self._odom.append((self._stamp(msg.header.stamp, self._now()), odom_to_pose(msg)))

    def _on_imu(self, msg: Imu):
        w = msg.angular_velocity
        self._imu.append((self._stamp(msg.header.stamp, self._now()), math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)))

    def _on_target(self, msg: String):
        tgt = msg.data.strip()
        if tgt and self.assoc.gauge(tgt) is None:
            self.get_logger().warning(f"/inspection/target '{tgt}' is not a registry gauge id",
                                      throttle_duration_sec=10.0)
        self._target = tgt

    def _log_stats(self):
        s = self._stats
        n = max(1, s["processed"])
        self.get_logger().info(
            f"stats: processed {s['processed']} frames ({s['full_searches']} full-frame searches), "
            f"skipped (motion) {s['skipped_motion']}, errors {s['errors']}, readings {s['readings']}, "
            f"processing mean {s['ms_sum'] / n:.0f} ms / max {s['ms_max']:.0f} ms")

    # ------------------------------------------------------------ helpers
    def _pose_at(self, t: float):
        if not self._odom:
            return None
        best = min(self._odom, key=lambda s: abs(s[0] - t))
        if abs(best[0] - t) > float(self.p["odom_max_age_s"]):
            return None
        return best[1]

    def _rotating(self, t: float) -> float:
        """Max |omega| around the exposure time (0 when no IMU data)."""
        win = float(self.p["imu_window_s"])
        vals = [w for (ti, w) in self._imu if t - win <= ti <= t + 0.03]
        if not vals and self._imu:
            ti, w = self._imu[-1]
            if abs(ti - t) < 0.5:
                vals = [w]
        return max(vals) if vals else 0.0

    def _update_intrinsics(self, width: int, height: int):
        info = self._cam_info
        if info is not None and info.width == width and info.height == height and info.k[0] > 0:
            key = ("info", width, height, tuple(info.k))
            if key != self._intr_key:
                K = np.array(info.k, float).reshape(3, 3)
                self.reader.set_camera_matrix(K)
                self.assoc.set_intrinsics(width, height, K=K)
                self._intr_key = key
            return
        key = ("hfov", width, height, float(self.p["camera_hfov"]))
        if key != self._intr_key:
            hfov = float(self.p["camera_hfov"])
            f = (width / 2.0) / math.tan(hfov / 2.0)
            self.reader.set_camera_matrix(np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]]))
            self.assoc.set_intrinsics(width, height, hfov=hfov)
            self._intr_key = key

    def _want_ocr(self, gid: str, t: float) -> bool:
        if not self.reader.cfg.ocr_enabled:
            return False
        st = self._ocr_state.get(gid)
        if st is None:
            return True
        period = float(self.p["ocr_confirmed_period_s"] if st["agree"] >= 2 else self.p["ocr_period_s"])
        return (t - st["t"]) >= period

    @staticmethod
    def _since(t: float, t_last: float) -> float:
        """Seconds since ``t_last``; infinite when the clock jumped back (simulation reset)."""
        dt = t - t_last
        return dt if dt >= 0.0 else float("inf")

    def _overlay_due(self, t: float, searched: bool) -> bool:
        """Overlay on every frame that was searched, otherwise at most every overlay_idle_period_s."""
        if not self.p["overlay_enabled"]:
            return False
        return searched or self._since(t, self._last_overlay) >= float(self.p["overlay_idle_period_s"])

    # ------------------------------------------------------------ main loop
    def _process(self):
        if not self._img_new or self._img_msg is None:
            return
        msg = self._img_msg
        t_img = self._stamp(msg.header.stamp, self._now())
        idle_rate = float(self.p["idle_process_rate_hz"])
        if not self._target and idle_rate > 0.0 and self._since(t_img, self._last_processed) < 1.0 / idle_rate:
            return      # no inspection target: reduced rate (the newest frame is taken at a later tick)
        self._img_new = False
        self._last_processed = t_img
        t_wall = time.perf_counter()
        try:
            bgr = image_msg_to_bgr(msg)
        except Exception as e:
            self.get_logger().warning(f"cannot convert image ({msg.encoding}): {e}", throttle_duration_sec=5.0)
            return
        omega = self._rotating(t_img)
        if omega > float(self.p["max_angular_velocity"]):
            self._stats["skipped_motion"] += 1
            if self._overlay_due(t_img, False):
                vis = bgr.copy()
                draw_header(vis, f"gauge_reader: frame skipped, body rotating {omega:.2f} rad/s", (0, 165, 255))
                self._publish_overlay(vis, msg, t_img)
            return
        try:
            self._read_frame(bgr, msg, t_img, omega, t_wall)
        except Exception as e:  # never let one bad frame kill the node
            self._stats["errors"] += 1
            self.get_logger().error(f"gauge reading failed: {e!r}", throttle_duration_sec=5.0)

    def _read_frame(self, bgr, msg, t_img, omega, t_wall):
        H, W = bgr.shape[:2]
        self._update_intrinsics(W, H)
        pose = self._pose_at(t_img)
        target = self._target or None
        preds = self.assoc.predict(pose) if pose is not None else []
        hints = []
        if pose is not None:
            hints = self.assoc.hints(pose, target_id=target, pose_sigma_px=float(self.p["pose_sigma_px"]))
            for h in hints:
                h.do_ocr = self._want_ocr(h.gauge_id, t_img)
                if h.do_ocr:
                    self._ocr_state.setdefault(h.gauge_id, {"t": -1e9, "agree": 0})["t"] = t_img
        # the enhancement (~40 ms per 800 x 600 frame) is computed only when something is searched
        enh = None
        # 1) pose-predicted ROIs (cheap); 2) whole-frame search when there is no usable pose, a
        #    predicted gauge was not found there, or periodically for unregistered dials
        readings = []
        if hints:
            enh = self.reader.enhance(bgr)
            readings = self.reader.read(bgr, hints, detect_unhinted=False, enhanced=enh)
        found = {r.gauge_id for r in readings}
        since_full = self._since(t_img, self._last_full_search)
        target_missing = bool(target) and target not in found
        # the whole-frame search costs several ROI searches: it is the sole source of detections
        # without a pose and the fallback when the mission expects a gauge that the pose prediction
        # did not deliver - then on every frame, but at most every full_search_min_period_s (CPU is
        # shared with Gazebo and the controllers); otherwise (missed prediction, unregistered dials,
        # nothing predicted in view) every full_search_period_s
        urgent = pose is None or target_missing
        periodic = bool(self.p["detect_unhinted"]) and since_full >= float(self.p["full_search_period_s"])
        if (urgent or periodic) and since_full >= float(self.p["full_search_min_period_s"]):
            self._last_full_search = t_img
            self._stats["full_searches"] += 1
            if enh is None:
                enh = self.reader.enhance(bgr)
            for r in self.reader.read(bgr, None, detect_unhinted=True, enhanced=enh):
                if all(math.hypot(r.center[0] - q.center[0], r.center[1] - q.center[1])
                       > 0.7 * max(r.radius_px, q.radius_px) for q in readings):
                    readings.append(r)

        # ---- ids for detections found outside the predicted ROIs
        pred_by_id = {p.gauge_id: p for p in preds}
        taken = {r.gauge_id for r in readings if r.gauge_id}
        free = [i for i, r in enumerate(readings) if not r.gauge_id]
        if free:
            ids = self.assoc.assign([readings[i].center for i in free], [readings[i].radius_px for i in free],
                                    predictions=[p for p in preds if p.gauge_id not in taken] if pose is not None else None,
                                    target_id=target if (target and target not in taken) else None,
                                    image_size=(W, H))
            for i, gid in zip(free, ids):
                if gid is None:
                    continue
                readings[i].gauge_id = gid
                # the free read was calibrated with a generic tick template (range unknown):
                # re-measure in a tight ROI with the registry range of the assigned gauge
                g = self.assoc.gauge(gid)
                r0 = readings[i]
                pr = pred_by_id.get(gid)
                hint = GaugeHint(gauge_id=gid, center=r0.center, radius=r0.radius_px,
                                 vrange=(float(g["min"]), float(g["max"])), unit=str(g.get("unit", "")),
                                 normal_cam=pr.normal_cam if pr is not None else None,
                                 search_radius=1.2 * r0.radius_px + 25.0, do_ocr=False)
                rr = self.reader.read(bgr, [hint], detect_unhinted=False, enhanced=enh)
                if rr:
                    rr[0].range_ocr, rr[0].ocr_confidence = r0.range_ocr, r0.ocr_confidence
                    readings[i] = rr[0]

        out = []
        overlay_items = []
        robot_pose = [float(pose[0]), float(pose[1]), float(pose[5])] if pose is not None else None
        for r in readings:
            gid = r.gauge_id
            g = self.assoc.gauge(gid)
            if g is None:
                if not self.p["publish_unknown"]:
                    continue
                rng = r.range_ocr
                val = value_from_fraction(r.fraction, rng) if rng else None
                # unidentified dial (no pose match, no target): id null, like the thermal alerts
                item = {"id": None, "tag": None, "value": val, "unit": "", "confidence": r.confidence * 0.5,
                        "angle_deg": r.angle_deg, "range": list(rng) if rng else None, "status": "unknown",
                        "n_samples": 1, "robot_pose": robot_pose, "fraction": r.fraction,
                        "range_source": "ocr" if rng else "none", "pixel": [r.center[0], r.center[1]],
                        "radius_px": r.radius_px}
                out.append(item)
                overlay_items.append((r, ["?  " + (f"{val:.2f}" if val is not None else f"{100 * r.fraction:.0f}% scale")],
                                      "unknown"))
                continue
            rc = self.assoc.reconcile_range(gid, r.range_ocr)
            vr = rc.range_used
            raw = value_from_fraction(r.fraction, vr)
            fs = abs(vr[1] - vr[0])
            fused = self.fusion.add(gid, t_img, raw, r.confidence, fs)
            status = self.layout.gauge_status(gid, fused.value)
            if r.range_ocr is not None:
                st = self._ocr_state.setdefault(gid, {"t": t_img, "agree": 0})
                st["agree"] = st["agree"] + 1 if rc.agree else 0
                if rc.agree is False:
                    self.get_logger().warning(
                        f"{gid}: OCR range {r.range_ocr} disagrees with registry {vr} - using registry",
                        throttle_duration_sec=10.0)
            pr = pred_by_id.get(gid)
            item = {
                "id": gid, "tag": g.get("tag", ""), "value": fused.value, "unit": g.get("unit", ""),
                "confidence": fused.confidence, "angle_deg": r.angle_deg, "range": [vr[0], vr[1]],
                "status": status, "n_samples": fused.n_samples, "robot_pose": robot_pose,
                # extras
                "raw_value": raw, "frame_confidence": r.confidence, "mad": fused.mad, "outliers": fused.outliers,
                "range_source": rc.source, "range_ocr": list(r.range_ocr) if r.range_ocr else None,
                "ocr_confidence": r.ocr_confidence, "pixel": [r.center[0], r.center[1]], "radius_px": r.radius_px,
                "blur_px": r.blur_px, "mirrored": r.mirrored,
                "distance": pr.distance if pr is not None else None,
                "incidence_deg": pr.incidence_deg if pr is not None else None,
            }
            out.append(item)
            overlay_items.append((r, [f"{gid} {fused.value:.2f} {g.get('unit', '')}",
                                      f"{status.upper()}  c={fused.confidence:.2f} n={fused.n_samples}"], status))
            if self._last_status.get(gid) != status:
                self.get_logger().info(f"{gid} ({g.get('tag', '')}): {fused.value:.3f} {g.get('unit', '')} -> {status}")
                self._last_status[gid] = status

        self._stats["processed"] += 1
        self._stats["readings"] += len(out)
        ms = (time.perf_counter() - t_wall) * 1000.0
        self._stats["ms_sum"] += ms
        self._stats["ms_max"] = max(self._stats["ms_max"], ms)
        if out or self.p["publish_empty"]:
            self.pub_readings.publish(json_msg({"stamp": t_img, "readings": out}))
        if self._overlay_due(t_img, enh is not None):
            if self.p["overlay_enhanced"]:
                if enh is None:
                    enh = self.reader.enhance(bgr)
                vis = enh["vis"].copy()
            else:
                vis = bgr.copy()
            for p in preds:
                if p.visible:
                    cv2.circle(vis, (int(p.center[0]), int(p.center[1])), int(p.radius), (160, 160, 160), 1, cv2.LINE_AA)
            for r, lines, st in overlay_items:
                draw_reading(vis, r, lines, st)
            dt = (time.perf_counter() - t_wall) * 1000.0
            draw_header(vis, f"gauge_reader  {len(out)} dial(s)  target={target or '-'}  "
                             f"pose={'ok' if pose is not None else 'none'}  |w|={omega:.2f}  {dt:.0f} ms")
            self._publish_overlay(vis, msg, t_img)

    def _publish_overlay(self, vis, src_msg, t_img: float):
        self._last_overlay = t_img
        s = float(self.p["overlay_scale"])
        if 0.05 < s < 0.999:
            vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        # dusan_common.image_utils assigns the pixel data as array.array('B') (fast rosidl path)
        self.pub_overlay.publish(bgr_to_image_msg(vis, src_msg.header.frame_id, src_msg.header.stamp))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GaugeReaderNode()
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
