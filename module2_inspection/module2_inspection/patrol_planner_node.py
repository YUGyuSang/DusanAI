"""
ROS 2 node ``patrol_planner`` - viewpoint-based inspection route
(``viewpoint_planner.ViewpointPlanner``).

Map source (first available): parameter ``map_yaml`` -> the layout's nav map
(``Layout.nav_map_yaml()``) -> the latched ``/map`` topic -> (optional) a map
rasterised from the layout.  Plans once at startup, again on
``/patrol/replan`` (std_msgs/Empty) and, if enabled, when ``/map`` changes.

Publishes (all latched / transient local):
    /patrol/route      nav_msgs/Path               full A* route through the viewpoints
    /patrol/plan_json  std_msgs/String (JSON)      DESIGN_SPEC section 4
    /patrol/markers    visualization_msgs/MarkerArray

Planning runs in a worker thread so the executor stays responsive; results
are published from a timer callback.
"""
import math
import os
import threading
import time
import zlib

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Empty, String
from visualization_msgs.msg import Marker, MarkerArray

from dusan_common.geometry import odom_to_pose
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout
from dusan_common.planning import path_to_msg
from dusan_common.robot_model import SENSORS
from dusan_common.ros_helpers import json_msg, latched_qos, make_marker, sensor_qos
from module2_inspection.viewpoint_planner import ViewpointParams, ViewpointPlanner, layout_occupancy_map

try:  # Jazzy and later: spin() raises this when the context is shut down (e.g. SIGINT from launch)
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover - older rclpy
    class ExternalShutdownException(Exception):
        """Placeholder for rclpy versions without ExternalShutdownException."""

_VP = ViewpointParams()
NODE_PARAMS = {
    "layout_file": "",
    "map_yaml": "",
    "map_topic": "/map",
    "odom_topic": "/odom",
    "frame_id": "map",
    "use_map_topic": True,
    "replan_on_map_update": True,  # replans only when the map content actually changes
    "fallback_layout_map": True,
    "map_wait_s": 10.0,
    "start_mode": "dock",          # dock | robot (replans start at the current robot pose)
    "gauge_ids": "",               # comma separated subset, '' = all gauges of the layout
    "marker_text_size": 0.35,
}
PLANNER_PARAMS = {
    "ring_min": _VP.ring_min, "ring_max": _VP.ring_max, "ring_step": _VP.ring_step,
    "angle_max_deg": _VP.angle_max_deg, "angle_step_deg": _VP.angle_step_deg,
    "robot_radius": _VP.robot_radius, "body_height": _VP.body_height,
    "image_margin_px": _VP.image_margin_px, "px_min": _VP.px_min, "px_good": _VP.px_good,
    "incidence_max_deg": _VP.incidence_max_deg, "clearance_good": _VP.clearance_good,
    "los_ignore_m": _VP.los_ignore_m,
    "w_pixel": _VP.w_pixel, "w_incidence": _VP.w_incidence, "w_vertical": _VP.w_vertical,
    "w_clearance": _VP.w_clearance,
    "avoid_zone_ids": list(_VP.avoid_zone_ids), "zone_penalty": _VP.zone_penalty,
    "fallback_ring_max": _VP.fallback_ring_max, "fallback_angle_max_deg": _VP.fallback_angle_max_deg,
    "return_to_dock": _VP.return_to_dock, "max_exact_nodes": _VP.max_exact_nodes,
    "smooth_paths": _VP.smooth_paths,
    "terrain_cost_enabled": _VP.terrain_cost_enabled, "terrain_w_pipe": _VP.terrain_w_pipe,
    "terrain_w_rubble": _VP.terrain_w_rubble, "terrain_w_plate": _VP.terrain_w_plate,
    "terrain_w_platform": _VP.terrain_w_platform, "terrain_margin": _VP.terrain_margin,
    "terrain_margin_fraction": _VP.terrain_margin_fraction,
}


def _map_signature(grid: OccupancyGridMap):
    return (grid.occ.shape, round(grid.resolution, 6), round(grid.origin_x, 4), round(grid.origin_y, 4),
            zlib.crc32(grid.occ.tobytes()))


def _quat_z(yaw: float):
    return math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def _score_color(score: float):
    s = min(1.0, max(0.0, score))
    return (1.0 - s, 0.25 + 0.75 * s, 0.1, 0.95)


class PatrolPlannerNode(Node):
    def __init__(self):
        super().__init__("patrol_planner")
        self.cfg = {}
        for name, default in {**NODE_PARAMS, **PLANNER_PARAMS}.items():
            self.declare_parameter(name, default)
            self.cfg[name] = self.get_parameter(name).value
        c = self.cfg
        self.layout = Layout.load(c["layout_file"] or None)
        self.vparams = ViewpointParams.from_dict({k: c[k] for k in PLANNER_PARAMS})
        self.gauge_ids = [g.strip() for g in str(c["gauge_ids"]).split(",") if g.strip()] or None
        unknown = [g for g in (self.gauge_ids or []) if self.layout.gauge(g) is None]
        if unknown:
            self.get_logger().warning(f"gauge_ids not in the layout (ignored, reported as unreachable): {unknown}")
        self.grid = None
        self.grid_sig = None
        self.grid_source = ""
        self.robot_pose = None
        self._lock = threading.Lock()
        self._worker = None
        self._request = None        # reason string of a pending planning request
        self._result = None         # (plan, reason) ready to publish
        # wall clock: the map fallback must also work while the simulation clock is not running
        self._t0 = time.monotonic()

        q = latched_qos()
        self.pub_route = self.create_publisher(Path, "/patrol/route", q)
        self.pub_json = self.create_publisher(String, "/patrol/plan_json", q)
        self.pub_markers = self.create_publisher(MarkerArray, "/patrol/markers", q)
        self.create_subscription(Empty, "/patrol/replan", self.on_replan, 10)
        self.create_subscription(Odometry, c["odom_topic"], self.on_odom, sensor_qos())
        if c["use_map_topic"]:
            self.create_subscription(OccupancyGrid, c["map_topic"], self.on_map, latched_qos())

        self._load_map_file()
        if self.grid is not None:
            self._request = "startup"
        # steady-clock timer: planning does not depend on the simulation time, and a
        # ROS-time timer (use_sim_time) would not fire while Gazebo is paused or
        # /clock is not published yet - the latched plan must still appear
        self.create_timer(0.25, self.on_timer, clock=Clock(clock_type=ClockType.STEADY_TIME))

    # ------------------------------------------------------------ inputs
    def _load_map_file(self):
        path = self.cfg["map_yaml"] or self.layout.nav_map_yaml()
        if path and os.path.exists(path):
            try:
                self._set_grid(OccupancyGridMap.load(path), f"file {path}")
            except Exception as exc:
                self.get_logger().error(f"cannot load map {path}: {exc}")
        else:
            self.get_logger().info(f"map file '{path}' not found - waiting for {self.cfg['map_topic']}")

    def _set_grid(self, grid: OccupancyGridMap, source: str) -> bool:
        sig = _map_signature(grid)
        if sig == self.grid_sig:
            return False
        with self._lock:
            self.grid, self.grid_sig, self.grid_source = grid, sig, source
        self.get_logger().info(f"nav map {grid.width}x{grid.height} @ {grid.resolution:.2f} m from {source}")
        return True

    def on_map(self, msg: OccupancyGrid):
        try:
            grid = OccupancyGridMap.from_msg(msg)
        except Exception as exc:
            self.get_logger().warning(f"bad /map message: {exc}")
            return
        first = self.grid is None
        if first or self.cfg["replan_on_map_update"]:
            if self._set_grid(grid, f"topic {self.cfg['map_topic']}"):
                self._request = "map" if not first else "startup"

    def on_odom(self, msg: Odometry):
        self.robot_pose = odom_to_pose(msg)

    def on_replan(self, _msg: Empty):
        self.get_logger().info("replan requested")
        self._request = "request"

    # ---------------------------------------------------------- planning
    def _start_pose(self, reason: str):
        if self.cfg["start_mode"] == "robot" and reason != "startup" and self.robot_pose is not None:
            x, y, _z, _r, _p, yaw = self.robot_pose
            return (x, y, yaw)
        return tuple(self.layout.dock_pose)

    def _work(self, grid: OccupancyGridMap, start, reason: str):
        try:
            planner = ViewpointPlanner(self.layout, grid, self.vparams)
            plan = planner.plan(start=start, gauge_ids=self.gauge_ids)
            with self._lock:
                self._result = (plan, reason)
        except Exception as exc:
            self.get_logger().error(f"patrol planning failed: {exc}")

    def on_timer(self):
        # publish a finished plan
        with self._lock:
            res, self._result = self._result, None
        if res is not None:
            self.publish_plan(*res)
        # fallback map when nothing arrived in time
        if self.grid is None and self.cfg["fallback_layout_map"]:
            waited = time.monotonic() - self._t0
            if waited > float(self.cfg["map_wait_s"]):
                self.get_logger().warning("no nav map available - rasterising a map from world_layout.yaml")
                self._set_grid(layout_occupancy_map(self.layout), "layout (fallback)")
                self._request = "startup"
        # start a worker for a pending request
        if self._request is not None and self.grid is not None and (self._worker is None or not self._worker.is_alive()):
            reason, self._request = self._request, None
            start = self._start_pose(reason)
            self.get_logger().info(f"planning patrol ({reason}) from ({start[0]:.2f}, {start[1]:.2f}) ...")
            self._worker = threading.Thread(target=self._work, args=(self.grid, start, reason), daemon=True)
            self._worker.start()

    # ------------------------------------------------------------ output
    def publish_plan(self, plan, reason: str):
        stamp = self.get_clock().now().to_msg()
        frame = self.cfg["frame_id"]
        tags = {g["id"]: g.get("tag", "") for g in self.layout.gauges}
        data = plan.to_json(tags)
        data["stamp"] = stamp.sec + stamp.nanosec * 1e-9
        data["reason"] = reason
        data["map_source"] = self.grid_source
        self.pub_json.publish(json_msg(data))
        route = path_to_msg(plan.route_points(), frame, stamp)
        for ps in route.poses:      # follow the floor level (corridor lower level at -0.88 m)
            ps.pose.position.z = float(self.layout.floor_z(ps.pose.position.x, ps.pose.position.y))
        self.pub_route.publish(route)
        self.pub_markers.publish(self.make_markers(plan, frame, stamp))
        self.get_logger().info(
            f"patrol plan: {' -> '.join(['dock'] + plan.order + (['dock'] if plan.return_to_dock else []))}  "
            f"length {plan.total_length:.1f} m (layout order {plan.naive_length:.1f} m), "
            f"{plan.method}, {plan.compute_s:.1f} s"
            + (f", unreachable: {plan.unreachable}" if plan.unreachable else ""))

    def make_markers(self, plan, frame: str, stamp) -> MarkerArray:
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        ts = float(self.cfg["marker_text_size"])
        cam = SENSORS[self.vparams.camera]["pose"]
        gauges = {g["id"]: g for g in self.layout.gauges}
        # route
        route = make_marker("patrol_route", 0, Marker.LINE_STRIP, frame, stamp, scale=(0.06, 0.0, 0.0),
                            color=(0.0, 0.8, 1.0, 0.9))
        for p in plan.route_points():
            route.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(self.layout.floor_z(p[0], p[1]) + 0.05)))
        if len(route.points) >= 2:          # RViz rejects line strips with fewer than 2 points
            arr.markers.append(route)
        links = make_marker("patrol_gauge_links", 0, Marker.LINE_LIST, frame, stamp, scale=(0.02, 0.0, 0.0),
                            color=(1.0, 1.0, 0.2, 0.9))
        for i, vp in enumerate(plan.viewpoints):
            fz = self.layout.floor_z(vp.x, vp.y)
            arrow = make_marker("patrol_viewpoints", i, Marker.ARROW, frame, stamp, pose=(vp.x, vp.y, fz + 0.3),
                                scale=(0.6, 0.1, 0.1), color=_score_color(vp.score))
            arrow.pose.orientation.z, arrow.pose.orientation.w = _quat_z(vp.yaw)
            arr.markers.append(arrow)
            text = make_marker("patrol_order", i, Marker.TEXT_VIEW_FACING, frame, stamp, pose=(vp.x, vp.y, fz + 0.9),
                               scale=(0.0, 0.0, ts), color=(1.0, 1.0, 1.0, 1.0),
                               text=f"{i + 1}: {vp.gauge_id} ({vp.score:.2f})")
            arr.markers.append(text)
            g = gauges.get(vp.gauge_id)
            if g is not None:
                cx = vp.x + cam[0] * math.cos(vp.yaw) - cam[1] * math.sin(vp.yaw)
                cy = vp.y + cam[0] * math.sin(vp.yaw) + cam[1] * math.cos(vp.yaw)
                cz = fz + self.vparams.body_height + cam[2]
                links.points.append(Point(x=float(cx), y=float(cy), z=float(cz)))
                links.points.append(Point(x=float(g["position"][0]), y=float(g["position"][1]), z=float(g["position"][2])))
        if links.points:
            arr.markers.append(links)
        for j, g in enumerate(self.layout.gauges):
            col = (1.0, 0.1, 0.1, 1.0) if g["id"] in plan.unreachable else (0.9, 0.9, 0.9, 1.0)
            arr.markers.append(make_marker("patrol_gauges", j, Marker.SPHERE, frame, stamp, pose=g["position"],
                                           scale=(0.18, 0.18, 0.18), color=col))
        dx, dy, _ = self.layout.dock_pose
        arr.markers.append(make_marker("patrol_start", 0, Marker.TEXT_VIEW_FACING, frame, stamp,
                                       pose=(plan.start[0], plan.start[1], self.layout.floor_z(dx, dy) + 0.9),
                                       scale=(0.0, 0.0, ts), color=(0.4, 1.0, 0.4, 1.0),
                                       text=f"START  route {plan.total_length:.1f} m"))
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PatrolPlannerNode()
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
