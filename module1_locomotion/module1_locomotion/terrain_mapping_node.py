"""
ROS 2 node ``terrain_mapper`` - robot-centric elevation mapping (thin wrapper).

Subscribes
  /depth_camera/points       sensor_msgs/PointCloud2   (primary input)
  /depth_camera/image        sensor_msgs/Image         (fallback, 32FC1 m / 16UC1 mm,
                                                        used when no cloud for 1 s)
  /depth_camera/camera_info  sensor_msgs/CameraInfo    (optional intrinsics override)
  /odom                      nav_msgs/Odometry         (world-frame robot pose)
  /locomotion/state          std_msgs/String (JSON)    fall guard: no fusion while the robot is
                                                        down (mode FALLEN / RECOVERING or a
                                                        recovery_state other than NORMAL); up
                                                        again -> the map goes back to its state
                                                        fall_lookback s before the fall
                                                        (elevation_map.MapFallGuard)
Publishes (at ``publish_rate``, default 5 Hz)
  /elevation_map/points          sensor_msgs/PointCloud2  organised x,y,z float32, frame 'odom'
  /elevation_map/traversability  nav_msgs/OccupancyGrid   0 easy .. 100 untraversable, -1 unknown
  /terrain/info                  std_msgs/String (JSON)   see terrain_classifier.py

Friction prior: the low-friction zones of world_layout.yaml (``layout_file``,
default = the installed plant_simulation config; plates + ``friction_zones``,
see friction_map.py) are handed to the classifier, which reports
``slippery`` / ``friction_mu`` in /terrain/info.  A missing layout only
disables the prior (warning); the online slip detection of the controller
works without it.

All algorithms live in elevation_map.py / terrain_classifier.py /
pointcloud_utils.py / friction_map.py; parameters are documented in
config/terrain.yaml.
"""
import array
import math
import time
from collections import deque
from dataclasses import fields as dc_fields

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import String

from dusan_common.geometry import odom_to_pose
from dusan_common.image_utils import image_msg_to_numpy
from dusan_common.obstacle_layer import robot_down
from dusan_common.ros_helpers import json_msg, parse_json_msg, sensor_qos, stamp_to_sec

from .elevation_map import (FALL_LOOKBACK, FALL_SETTLE, FALL_SNAPSHOT_PERIOD, ElevationMap, ElevationMapParams,
                            MapFallGuard)
from .friction_map import FrictionMap
from .terrain_classifier import TerrainClassifier, ClassifierParams
from .pointcloud_utils import (ConventionDetector, DepthIntrinsics, PointFilter, PoseBuffer,
                               camera_points_to_world, depth_image_to_points, depth_to_metres,
                               pointcloud2_to_xyz, world_to_camera_points, xyz_to_pointcloud2)


def _reliable_qos(depth=1):
    return QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)


def _coerce_like(value, default):
    """Coerce a dynamically typed parameter value to the type of ``default``."""
    if value is None:
        return default
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(round(float(value)))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, (list, tuple)):
        items = [value] if isinstance(value, str) or not hasattr(value, "__iter__") else list(value)
        if default and isinstance(default[0], float):
            return [float(v) for v in items]
        return items
    return str(value) if isinstance(default, str) else value


class TerrainMapperNode(Node):
    def __init__(self):
        super().__init__("terrain_mapper")
        P = self._param
        # ------------------------------------------------------------ topics
        self.cloud_topic = P("cloud_topic", "/depth_camera/points")
        self.image_topic = P("image_topic", "/depth_camera/image")
        self.info_topic_cam = P("camera_info_topic", "/depth_camera/camera_info")
        self.odom_topic = P("odom_topic", "/odom")
        self.loco_topic = P("locomotion_state_topic", "/locomotion/state")
        map_cloud_topic = P("map_cloud_topic", "/elevation_map/points")
        trav_topic = P("traversability_topic", "/elevation_map/traversability")
        info_topic = P("info_topic", "/terrain/info")
        self.frame_id = P("frame_id", "odom")
        self.sensor_name = P("sensor_name", "depth_camera")
        self.publish_rate = max(0.5, float(P("publish_rate", 5.0)))
        self.fallback_timeout = float(P("image_fallback_timeout", 1.0))
        self.use_camera_info = bool(P("use_camera_info", True))
        self.use_image_fallback = bool(P("use_image_fallback", True))
        self.publish_traversability = bool(P("publish_traversability", True))
        self.log_period = float(P("log_period", 10.0))
        conv_mode = str(P("cloud_convention", "auto"))
        lock_frames = int(P("convention_lock_frames", 5))
        pose_buf_len = float(P("pose_buffer_length", 2.0))
        max_extrap = float(P("max_pose_extrapolation", 0.3))
        self.image_stride = max(1, int(P("image_stride", 1)))
        # clouds whose header.frame_id is one of these are already in the world frame
        self.world_frames = {str(f).strip("/") for f in P("world_frame_ids", ["odom", "map", "world"])}
        # ------------------------------------------------------- algorithms
        dflt = ElevationMapParams()
        map_params = ElevationMapParams.from_dict(
            {f.name: P("map." + f.name, getattr(dflt, f.name)) for f in dc_fields(dflt)})
        cdflt = ClassifierParams()
        cls_params = ClassifierParams.from_dict(
            {f.name: P("classifier." + f.name, getattr(cdflt, f.name)) for f in dc_fields(cdflt)})
        fd = PointFilter()
        self.filter = PointFilter(
            min_range=float(P("filter.min_range", fd.min_range)),
            max_range=float(P("filter.max_range", fd.max_range)),
            body_box_min=tuple(float(v) for v in P("filter.body_box_min", list(fd.body_box_min))),
            body_box_max=tuple(float(v) for v in P("filter.body_box_max", list(fd.body_box_max))),
            max_height_above_base=float(P("filter.max_height_above_base", fd.max_height_above_base)),
            stride=max(1, int(P("filter.point_stride", fd.stride))))

        self.emap = ElevationMap(params=map_params)
        self.fall_guard = MapFallGuard(self.emap, lookback=float(P("fall_lookback", FALL_LOOKBACK)),
                                       settle=float(P("fall_settle", FALL_SETTLE)),
                                       period=float(P("fall_snapshot_period", FALL_SNAPSHOT_PERIOD)),
                                       enabled=bool(P("fall_guard", True)))
        self.classifier = TerrainClassifier(cls_params)
        self.friction_map = self._load_friction_map(str(P("layout_file", "")), bool(P("friction_prior", True)),
                                                    float(P("friction_zone_max_mu", 0.6)))
        self.classifier.set_friction_map(self.friction_map)
        self.poses = PoseBuffer(max_age=pose_buf_len, max_extrapolation=max_extrap)
        try:
            self.convention = ConventionDetector(conv_mode, lock_after=lock_frames)
        except ValueError as e:
            self._warn(f"{e}; using 'auto'")
            self.convention = ConventionDetector("auto", lock_after=lock_frames)
        self.intr_model = DepthIntrinsics.from_sensor(self.sensor_name)
        self.intr_info = None

        # ------------------------------------------------------------ state
        self.latest_pose = None
        self.last_cloud_time = None       # node-clock seconds of the last usable cloud
        self.n_clouds = 0
        self.n_images = 0
        self.n_points = 0
        self.proc_ms = deque(maxlen=50)
        self.last_info = {}
        self.last_source = "none"         # 'points' | 'image' | 'none'
        self._last_log = time.monotonic()

        # ------------------------------------------------------------ ROS io
        qos_in = sensor_qos()
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos_in)
        if self.fall_guard.enabled:
            self.create_subscription(String, self.loco_topic, self._on_loco_state, _reliable_qos(10))
        self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, qos_in)
        if self.use_image_fallback:
            self.create_subscription(Image, self.image_topic, self._on_image, qos_in)
        if self.use_camera_info:
            self.create_subscription(CameraInfo, self.info_topic_cam, self._on_camera_info, qos_in)
        self.pub_cloud = self.create_publisher(PointCloud2, map_cloud_topic, _reliable_qos(1))
        self.pub_trav = self.create_publisher(OccupancyGrid, trav_topic, _reliable_qos(1))
        self.pub_info = self.create_publisher(String, info_topic, _reliable_qos(5))
        self.create_timer(1.0 / self.publish_rate, self._on_timer)
        rows, cols = self.emap.shape
        self.get_logger().info(
            f"terrain_mapper: {cols}x{rows} cells @ {self.emap.resolution:.3f} m, cloud '{self.cloud_topic}' "
            f"(convention {conv_mode}), image fallback '{self.image_topic}', odom '{self.odom_topic}', "
            f"publishing at {self.publish_rate:.1f} Hz in frame '{self.frame_id}'")

    # ----------------------------------------------------------------- utils
    def _load_friction_map(self, layout_file, enabled, max_mu):
        """FrictionMap of the layout (None when disabled or unavailable)."""
        if not enabled:
            self.get_logger().info("friction prior disabled (friction_prior: false)")
            return None
        path = layout_file
        if not path:
            try:
                from dusan_common.layout import default_layout_path
                path = default_layout_path()
            except Exception as e:  # pragma: no cover - dusan_common is a dependency
                self._warn(f"no layout path ({e}) - friction prior disabled")
                return None
        try:
            fmap = FrictionMap.load(path, max_mu=max_mu)
        except Exception as e:
            self._warn(f"cannot load the friction zones from '{path}': {e} - friction prior disabled")
            return None
        zones = ", ".join(f"{z.zone_id} (mu {z.mu:g})" for z in fmap.zones) or "none"
        self.get_logger().info(f"friction prior: {len(fmap)} low-friction zone(s) from {path}: {zones}")
        return fmap

    def _param(self, name, default):
        """Declare a parameter and return its value.

        A type mismatch in the YAML (e.g. ``6`` for a float parameter) makes
        rclpy reject the statically typed declaration; the parameter is then
        re-declared with ``dynamic_typing`` and the value coerced to the type
        of the default.  Only if that fails too the default is used (with a
        warning) - the node never crashes on a config typo."""
        try:
            self.declare_parameter(name, default)
            v = self.get_parameter(name).value
            return default if v is None else v
        except Exception as e:  # InvalidParameterTypeException & co.
            first_error = e
        try:
            from rcl_interfaces.msg import ParameterDescriptor
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
            v = _coerce_like(self.get_parameter(name).value, default)
            self._warn(f"parameter '{name}': wrong type in the parameter file, coerced to {v!r}")
            return v
        except Exception:
            self._warn(f"parameter '{name}': {first_error} -> using default {default!r}")
            return default

    def _warn(self, text, throttle=None):
        log = self.get_logger()
        fn = getattr(log, "warning", None) or log.warn
        if throttle:
            fn(text, throttle_duration_sec=float(throttle))
        else:
            fn(text)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp(self, header) -> float:
        t = stamp_to_sec(header.stamp)
        return t if t > 0.0 else self._now()

    # ------------------------------------------------------------- callbacks
    def _on_odom(self, msg: Odometry):
        pose = odom_to_pose(msg)
        if not all(math.isfinite(v) for v in pose):
            return
        self.latest_pose = pose
        self.poses.add(self._stamp(msg.header), pose)

    def _on_loco_state(self, msg: String):
        d = parse_json_msg(msg)
        if not isinstance(d, dict) or not d.get("mode"):
            return
        g = self.fall_guard
        was_down = g.down
        n = g.set_upright(not robot_down(d.get("mode"), d.get("recovery_state")), self._now())
        if g.down and not was_down:
            self.get_logger().info(f"robot down ({d.get('mode')}/{d.get('recovery_state')}) - depth frames are "
                                   "not fused until it stands again")
        elif n is not None:
            r = g.last_restore or {}
            back = (f"to its state {r['snapshot_age']:.1f} s ago" if r.get("snapshot_age") is not None
                    else "cleared (no older state)")
            self.get_logger().info(f"robot up again after {r.get('down_s', 0.0):.1f} s - elevation map {back}, "
                                   f"{n} cells changed; fusion resumes in {g.settle:.1f} s")

    def _on_camera_info(self, msg: CameraInfo):
        intr = DepthIntrinsics.from_camera_info(msg)
        if intr is not None:
            self.intr_info = intr

    def _on_cloud(self, msg: PointCloud2):
        try:
            pts = pointcloud2_to_xyz(msg)
        except (KeyError, ValueError) as e:
            self._warn(f"cannot parse {self.cloud_topic}: {e}", throttle=5.0)
            return
        if pts.shape[0] == 0:
            return
        self.last_cloud_time = self._now()
        self.n_clouds += 1
        t = self._stamp(msg.header)
        if msg.header.frame_id.strip("/") in self.world_frames:
            pose = self._pose_at(t)
            if pose is None:
                return
            self._integrate(world_to_camera_points(pts, pose, self.sensor_name), t, "link", "points")
            return
        conv = self.convention.update(pts)
        self._integrate(pts, t, conv, "points")

    def _on_image(self, msg: Image):
        if self.last_cloud_time is not None and 0.0 <= self._now() - self.last_cloud_time < self.fallback_timeout:
            return                      # point clouds are flowing - image not needed
        try:
            raw = image_msg_to_numpy(msg)
        except ValueError as e:
            self._warn(f"cannot decode {self.image_topic}: {e}", throttle=5.0)
            return
        if raw.ndim != 2:
            self._warn(f"depth image must be single channel, got {msg.encoding}", throttle=5.0)
            return
        depth = depth_to_metres(raw, msg.encoding)
        intr = self.intr_info or self.intr_model
        intr = intr.scaled(depth.shape[1], depth.shape[0])
        pts = depth_image_to_points(depth, intr, stride=self.image_stride,
                                    min_depth=self.filter.min_range, max_depth=self.filter.max_range)
        if pts.shape[0] == 0:
            return
        self.n_images += 1
        self._integrate(pts, self._stamp(msg.header), "link", "image")   # computed in the link frame

    def _pose_at(self, t):
        """Odometry pose interpolated at time t (latest pose if t is outside
        the buffer), or None before the first odometry message."""
        pose = self.poses.at(t)
        if pose is None:
            latest = self.poses.latest()
            if latest is None:
                self._warn(f"no odometry on '{self.odom_topic}' yet - dropping depth data", throttle=5.0)
                return None
            pose = latest[1]
        return pose

    def _integrate(self, pts_cam, t, convention, source):
        """Register camera points with the odometry pose at the sensor stamp
        ``t`` and fuse them.  The map's forgetting clock always uses the node
        clock (same base as the publish timer), so a sensor stamped in another
        time base (use_sim_time mismatch) cannot wipe the map."""
        pose = self._pose_at(t)
        if pose is None:
            return
        if not self.fall_guard.accept(self._now()):
            return                      # robot down / settling: the tipped camera's frame is not fused
        self.last_source = source
        t0 = time.perf_counter()
        pw, rng, origin = camera_points_to_world(pts_cam, pose, convention, self.sensor_name, self.filter)
        self.emap.recenter(pose[0], pose[1])
        self.emap.integrate(pw, origin, t=self._now(), ranges=rng)
        self.n_points += int(pw.shape[0])
        self.proc_ms.append((time.perf_counter() - t0) * 1e3)

    # ----------------------------------------------------------------- output
    def _on_timer(self):
        pose = self.latest_pose
        if pose is None:
            return
        t = self._now()
        self.emap.recenter(pose[0], pose[1])
        self.emap.predict(t)
        stamp = self.get_clock().now().to_msg()
        t0 = time.perf_counter()
        self.pub_cloud.publish(xyz_to_pointcloud2(self.emap.to_cloud_xyz(), self.frame_id, stamp))
        if self.publish_traversability:
            self.pub_trav.publish(self._occupancy_msg(stamp))
        info = self.classifier.update(self.emap, pose, t)
        info["stamp"] = t
        info["frame_id"] = self.frame_id
        info["source"] = self.last_source
        info["convention"] = self.convention.current
        self.last_info = info
        self.pub_info.publish(json_msg(info))
        layer_ms = (time.perf_counter() - t0) * 1e3
        now_w = time.monotonic()
        if now_w - self._last_log >= self.log_period:
            dt = now_w - self._last_log
            self._last_log = now_w
            mean_ms = sum(self.proc_ms) / len(self.proc_ms) if self.proc_ms else 0.0
            self.get_logger().info(
                f"clouds {self.n_clouds} images {self.n_images} ({self.n_points / dt:.0f} pts/s), "
                f"convention {self.convention.current}, integrate {mean_ms:.1f} ms, publish {layer_ms:.1f} ms, "
                f"known {self.emap.known_fraction():.2f}, class {info.get('class')} "
                f"-> {info.get('recommended_gait')}"
                + (f", robot down: {self.fall_guard.dropped_frames} frames dropped" if self.fall_guard.down else "")
                + (f", slippery (mu {info.get('friction_mu')})" if info.get("slippery") else ""))
            self.n_points = 0

    def _occupancy_msg(self, stamp):
        occ = self.emap.occupancy_data()
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = stamp
        msg.info.map_load_time = stamp
        msg.info.resolution = float(self.emap.resolution)
        msg.info.height, msg.info.width = (int(v) for v in occ.shape)
        ox, oy = self.emap.occupancy_origin()
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.orientation.w = 1.0
        try:
            msg.data = array.array("b", occ.astype("int8").tobytes())
        except (AssertionError, TypeError, ValueError):  # pragma: no cover - other rosidl versions
            msg.data = occ.flatten().astype(int).tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = TerrainMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # ExternalShutdownException on newer rclpy
        if rclpy.ok():
            raise
        del e
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
