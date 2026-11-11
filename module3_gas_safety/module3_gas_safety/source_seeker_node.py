"""
source_seeker - olfactory gas-source seeking (gradient descent on -c).

Subscribes
  /gas/seek/enable   std_msgs/Bool       start (true) / stop (false) a seek run (latched or volatile);
                     true again within resume_window_s of a false continues an unfinished run
  /odom              nav_msgs/Odometry
  /gas/sensor        std_msgs/Float32    nose concentration [ppm]
  /gas/sensor_array  std_msgs/Float32MultiArray [FL, FR, RL, RR] ppm
  /gas/wind          geometry_msgs/Vector3Stamped (world frame)
  /map               nav_msgs/OccupancyGrid (latched; fallback map_yaml / layout)
  /safety/rth_active std_msgs/Bool       commands are suppressed while true
  obstacle_elevation_topic (/elevation_map/points, sensor_msgs/PointCloud2) or, with obstacle_source
                     traversability, obstacle_traversability_topic (OccupancyGrid): novel obstacles
                     merged into the nav map of the local planner (obstacle_enabled)
Publishes
  /cmd_vel/seek      geometry_msgs/Twist (only while enabled, seeking (not FOUND/FAILED) and no RTH;
                     a single zero Twist when commanding stops)
  /gas/seek_status   std_msgs/String JSON {state, concentration, gradient, estimate, error_m, path_length, elapsed, ...}
  /gas/source_estimate geometry_msgs/PointStamped
  /gas/seek_path     nav_msgs/Path (trajectory while seeking)
"""
import math

from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from dusan_common.grid_map import OccupancyGridMap
from dusan_common.obstacle_layer import EvidenceFrame, frame_accepted, organised_cloud_to_xyz
from dusan_common.planning import path_to_msg
from dusan_common.ros_helpers import json_msg

from .layout_map import map_fingerprint
from .node_common import (best_effort_qos, declare, latched_qos, load_layout_and_map, now_sec,
                          odom_pose_twist_finite, reliable_qos, run_node, stamp, subscribe_latched, twist_msg,
                          warn_throttled)
from .params import SEEKER_PARAMS, seeker_config
from .source_seeking import IDLE, SourceSeeker


class SourceSeekerNode(Node):
    def __init__(self):
        super().__init__("source_seeker")
        self.p = p = declare(self, SEEKER_PARAMS)
        self.layout, grid = load_layout_and_map(self, p["layout_file"], p["map_yaml"])
        self._map_fp = map_fingerprint(grid)
        self.seeker = SourceSeeker(seeker_config(p), grid)
        if p["ground_truth_override"]:
            self.truth = (float(p["ground_truth_x"]), float(p["ground_truth_y"]))
        else:
            src = self.layout.gas_leak.get("source")
            self.truth = (float(src[0]), float(src[1])) if src else None

        self.enabled = False
        self.rth_active = False
        self.pose = None
        self.c_nose = None
        self.t_nose = -1e9
        self.c_array = None
        self.t_array = -1e9
        self.wind = None
        self._was_commanding = False
        self._last_state = IDLE

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel/seek", reliable_qos(10))
        self.pub_status = self.create_publisher(String, "/gas/seek_status", reliable_qos(10))
        self.pub_est = self.create_publisher(PointStamped, "/gas/source_estimate", reliable_qos(10))
        self.pub_path = self.create_publisher(Path, "/gas/seek_path", latched_qos(1))

        # mission_manager latches /gas/seek/enable: a late-joining seeker must
        # still receive the current request (callback is idempotent)
        subscribe_latched(self, Bool, "/gas/seek/enable", self._on_enable)
        self.create_subscription(Odometry, p["odom_topic"], self._on_odom, best_effort_qos(5))
        self.create_subscription(Float32, "/gas/sensor", self._on_sensor, best_effort_qos(10))
        self.create_subscription(Float32MultiArray, "/gas/sensor_array", self._on_array, best_effort_qos(10))
        self.create_subscription(Vector3Stamped, "/gas/wind", self._on_wind, best_effort_qos(10))
        self.create_subscription(Bool, "/safety/rth_active", self._on_rth, best_effort_qos(10))
        if p["use_map_topic"]:
            subscribe_latched(self, OccupancyGrid, "/map", self._on_map)
        # novel obstacles from terrain_mapper (best effort depth 1 = latest frame of a reliable publisher)
        self.obstacle_topic = None
        if p["obstacle_enabled"]:
            if str(p["obstacle_source"]).strip().lower() == "traversability":
                self.obstacle_topic = p["obstacle_traversability_topic"]
                self.create_subscription(OccupancyGrid, self.obstacle_topic, self._on_obstacle_grid,
                                         best_effort_qos(1))
            else:
                self.obstacle_topic = p["obstacle_elevation_topic"]
                self.create_subscription(PointCloud2, self.obstacle_topic, self._on_obstacle_cloud,
                                         best_effort_qos(1))

        self.create_timer(1.0 / max(p["control_rate"], 0.5), self._on_control)
        self.create_timer(1.0 / max(p["status_rate"], 0.1), self._on_status)
        self.create_timer(1.0 / max(p["path_rate"], 0.1), self._on_path)

    # ------------------------------------------------------------ callbacks
    def _on_enable(self, msg: Bool) -> None:
        if msg.data and not self.enabled:
            self.enabled = True
            if self.seeker.enable(now_sec(self)):
                self.get_logger().info(f"gas source seeking RESUMED ({self.seeker.state}, run continued)")
            else:
                self.get_logger().info("gas source seeking ENABLED")
            self._on_status()
        elif not msg.data and self.enabled:
            self.enabled = False
            # an unfinished run stays suspended: re-enabling within
            # resume_window_s (mission pause / teleop override) continues it
            self.seeker.disable(now_sec(self))
            self._stop()
            self.get_logger().info("gas source seeking disabled")
            self._on_status()

    def _on_odom(self, msg: Odometry) -> None:
        v = odom_pose_twist_finite(msg)
        if v is None:
            warn_throttled(self, "odom", "non-finite /odom sample ignored")
            return
        x, y, yaw = v[:3]
        self.pose = (x, y, yaw)

    def _on_sensor(self, msg: Float32) -> None:
        self.c_nose = float(msg.data)
        self.t_nose = now_sec(self)

    def _on_array(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 4:
            self.c_array = [float(v) for v in msg.data[:4]]
            self.t_array = now_sec(self)

    def _on_wind(self, msg: Vector3Stamped) -> None:
        self.wind = (float(msg.vector.x), float(msg.vector.y))

    def _on_rth(self, msg: Bool) -> None:
        if msg.data and not self.rth_active and self._was_commanding:
            self._was_commanding = False   # RTH owns the robot now (no zero burst into the mux)
        self.rth_active = bool(msg.data)

    def _obstacle_frame(self, msg, convert) -> None:
        fid = msg.header.frame_id
        if not frame_accepted(fid, self.p["obstacle_frames"]):
            self.seeker.obstacles.note_rejected()
            warn_throttled(self, "obstacle_frame",
                           f"obstacle frame '{fid}' on {self.obstacle_topic} rejected (obstacle_frames "
                           f"'{self.p['obstacle_frames']}')", 30.0)
            return
        if self.pose is None:
            return
        t = now_sec(self)
        if not self.seeker.obstacle_frame_due(t):      # not seeking: one frame per obstacle_idle_period
            return
        try:
            self.seeker.update_obstacles(convert(msg, fid), self.pose, t)
        except Exception as exc:  # noqa: BLE001 - one bad message must not kill the node
            warn_throttled(self, "obstacle_msg", f"obstacle message on {self.obstacle_topic} ignored: {exc}", 30.0)

    def _on_obstacle_cloud(self, msg) -> None:
        self._obstacle_frame(msg, lambda m, fid: EvidenceFrame.from_elevation_cloud(
            organised_cloud_to_xyz(m), fid, float(self.p["obstacle_step_height"]),
            int(self.p["obstacle_step_window"])))

    def _on_obstacle_grid(self, msg) -> None:
        self._obstacle_frame(msg, lambda m, fid: EvidenceFrame.from_traversability(
            OccupancyGridMap.from_msg(m), int(self.p["obstacle_cost_threshold"])))

    def _on_map(self, msg: OccupancyGrid) -> None:
        try:
            grid = OccupancyGridMap.from_msg(msg)
        except Exception as exc:  # malformed map
            self.get_logger().warning(f"/map ignored: {exc}")
            return
        fp = map_fingerprint(grid)
        if fp != self._map_fp:
            self._map_fp = fp
            self.seeker.set_map(grid)
            self.get_logger().info(f"nav map updated from /map ({grid.width}x{grid.height})")

    # -------------------------------------------------------------- control
    def _stop(self) -> None:
        if self._was_commanding:
            self.pub_cmd.publish(Twist())
        self._was_commanding = False

    def _on_control(self) -> None:
        if not self.enabled or self.rth_active:
            self._stop()
            return
        t = now_sec(self)
        if self.pose is None or self.c_nose is None or t - self.t_nose > self.p["sensor_timeout"]:
            self._stop()
            return
        arr = self.c_array if (self.c_array is not None and t - self.t_array <= self.p["sensor_timeout"]) else None
        x, y, yaw = self.pose
        vx, vy, wz = self.seeker.update(t, x, y, yaw, self.c_nose, arr, self.wind)
        if self.seeker.active:
            self.pub_cmd.publish(twist_msg(vx, vy, wz))
            self._was_commanding = True
        else:
            # FOUND / FAILED: one zero Twist, then silence so that the
            # cmd_vel mux falls back to the next source (patrol) even if the
            # mission manager keeps /gas/seek/enable true
            self._stop()
        if self.seeker.state != self._last_state:
            self._last_state = self.seeker.state
            self.get_logger().info(f"seek state -> {self.seeker.state} ({self.seeker.message})")
            self._on_status()

    def _on_status(self) -> None:
        t = now_sec(self)
        st = self.seeker.status(t)
        est = st.get("estimate")
        if est is not None and self.truth is not None:
            st["error_m"] = math.hypot(est[0] - self.truth[0], est[1] - self.truth[1])
        st["enabled"] = self.enabled
        st["rth_active"] = self.rth_active
        self.pub_status.publish(json_msg(st))
        if est is not None:
            pm = PointStamped()
            pm.header.stamp = stamp(self)
            pm.header.frame_id = self.p["frame_id"]
            pm.point.x, pm.point.y = float(est[0]), float(est[1])
            pm.point.z = 0.4
            self.pub_est.publish(pm)

    def _on_path(self) -> None:
        if self.seeker.path:
            self.pub_path.publish(path_to_msg(self.seeker.path, self.p["frame_id"], stamp(self)))


def main(args=None):
    run_node(SourceSeekerNode, args)


if __name__ == "__main__":
    main()
