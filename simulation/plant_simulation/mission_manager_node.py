"""
mission_manager node: thin ROS wrapper around plant_simulation.mission_logic.

Subscribes  /patrol/plan_json (latched), /map (latched), /odom, /gas/sensor,
            /gas/seek_status, /inspection/gauge_readings, /safety/status,
            /safety/rth_active, /teleop/active, /locomotion/state,
            /mission/command, and (obstacle_enabled) the elevation map of
            terrain_mapper: obstacle_elevation_topic (/elevation_map/points,
            PointCloud2) or, with obstacle_source traversability,
            obstacle_traversability_topic (OccupancyGrid) - novel obstacles
Publishes   /cmd_vel/patrol (Twist), /nav/plan (Path, latched),
            /inspection/target (String, latched), /gas/seek/enable (Bool,
            latched), /locomotion/command, /mission/command ('rth' at the end
            of a full mission), /mission/state (JSON, latched)
"""
import os
import time
import traceback

import rclpy  # noqa: F401
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, String

from dusan_common.geometry import yaw_from_quat
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout
from dusan_common.obstacle_layer import EvidenceFrame, frame_accepted, organised_cloud_to_xyz
from dusan_common.planning import path_to_msg
from dusan_common.ros_helpers import declare_params, json_msg, latched_qos, parse_json_msg

from plant_simulation.mission_logic import MISSION_DEFAULTS, MissionLogic
from plant_simulation.node_utils import (best_effort_qos, guarded, map_fingerprint, now_sec, reliable_qos,
                                         run_node, subscribe_any, subscribe_latched)

MAP_FILE_FALLBACK_AFTER = 10.0   # s (wall) without /map before loading the map file directly
STEP_FAILURES_BEFORE_ABORT = 3   # consecutive MissionLogic.step exceptions -> abort, 't' starts afresh
FRAME_WARN_PERIOD = 30.0         # s (wall) between warnings about rejected obstacle frames


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__("mission_manager")
        self.p = declare_params(self, MISSION_DEFAULTS)
        self.layout = Layout.load(self.p["layout_file"] or None)
        self.logic = MissionLogic(self.p, self.layout)
        self._map_key = None          # fingerprint of the last /map used

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel/patrol", reliable_qos(10))
        self.pub_path = self.create_publisher(Path, "/nav/plan", latched_qos())
        self.pub_target = self.create_publisher(String, "/inspection/target", latched_qos())
        self.pub_seek = self.create_publisher(Bool, "/gas/seek/enable", latched_qos())
        self.pub_loco = self.create_publisher(String, "/locomotion/command", reliable_qos(10))
        self.pub_mcmd = self.create_publisher(String, "/mission/command", reliable_qos(10))
        self.pub_state = self.create_publisher(String, "/mission/state", latched_qos())

        # every callback is guarded: an exception is logged and the node keeps running
        # (run_node re-raises - one bad message must not stop the mission for good)
        def g(name, fn=None):
            return guarded(self, name, fn or getattr(self, name))
        subscribe_latched(self, String, "/patrol/plan_json", g("_on_plan"))
        self.create_subscription(OccupancyGrid, "/map", g("_on_map"), latched_qos())
        self.create_subscription(Odometry, "/odom", g("_on_odom"), best_effort_qos(10))
        self.create_subscription(Float32, "/gas/sensor", g("_on_gas"), best_effort_qos(10))
        subscribe_any(self, String, "/gas/seek_status", g("_on_seek_status"))
        # gauge readings are counted -> one reliable subscription (the logic also de-duplicates by stamp)
        self.create_subscription(String, "/inspection/gauge_readings", g("_on_readings"), reliable_qos(20))
        subscribe_any(self, String, "/safety/status", g("_on_safety"))
        subscribe_any(self, Bool, "/safety/rth_active",
                      g("rth_active", lambda m: self.logic.on_rth_active(m.data, now_sec(self))))
        subscribe_any(self, Bool, "/teleop/active",
                      g("teleop_active", lambda m: self.logic.on_teleop_active(m.data, now_sec(self))))
        subscribe_any(self, String, "/locomotion/state", g("_on_loco"))
        self.create_subscription(String, "/mission/command", g("_on_command"), reliable_qos(10))
        # novel obstacles from terrain_mapper's elevation map (RELIABLE depth-1 publisher: a
        # BEST_EFFORT depth-1 subscription matches and always hands over the latest frame)
        self._frame_warn_t = -1e9
        self.obstacle_topic = None
        if self.p["obstacle_enabled"]:
            if str(self.p["obstacle_source"]).strip().lower() == "traversability":
                self.obstacle_topic = self.p["obstacle_traversability_topic"]
                self.create_subscription(OccupancyGrid, self.obstacle_topic, g("_on_obstacle_grid"),
                                         best_effort_qos(1))
            else:
                self.obstacle_topic = self.p["obstacle_elevation_topic"]
                self.create_subscription(PointCloud2, self.obstacle_topic, g("_on_obstacle_cloud"),
                                         best_effort_qos(1))

        self._t_state_pub = -1e9
        self._map_wall_t0 = time.monotonic()
        self._map_from_file_tried = False
        self._step_failures = 0
        self._outputs = guarded(self, "publishing the mission outputs", self._publish_outputs)
        self._state_pub = guarded(self, "publishing /mission/state", self._publish_state)
        rate = max(1.0, float(self.p["control_rate"]))
        self.create_timer(1.0 / rate, g("_tick"))
        # initial latched outputs so that late joiners see a consistent picture
        self.pub_target.publish(String(data=""))
        self.pub_seek.publish(Bool(data=False))
        self.pub_state.publish(json_msg(self.logic.state_dict(now_sec(self))))
        self.get_logger().info(
            f"mission manager up (autostart={self.p['autostart']}, delay {self.p['autostart_delay']:.0f} s); "
            "commands on /mission/command: start patrol seek rth pause resume abort dock")

    # ------------------------------------------------------------- callbacks
    def _on_plan(self, msg):
        d = parse_json_msg(msg)
        if d is not None:
            self.logic.set_plan(d)

    def _on_map(self, msg):
        # map_server_lite re-publishes the same map every few seconds (and the
        # latched + volatile subscriptions may deliver it twice): rebuilding
        # the grid would throw away its cached distance field / inflation
        # masks, so identical maps are ignored.
        key = map_fingerprint(msg)
        if key == self._map_key:
            return
        try:
            grid = OccupancyGridMap.from_msg(msg)
        except Exception as exc:
            self.get_logger().error(f"bad /map message: {exc}")
            return
        first = self.logic.grid is None
        self._map_key = key
        self.logic.set_map(grid)
        self.get_logger().info(f"/map {'received' if first else 'updated'}: {grid.width}x{grid.height} "
                               f"@ {grid.resolution:.2f} m")

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self.logic.set_pose(p.x, p.y, yaw_from_quat(msg.pose.pose.orientation), now_sec(self))

    def _on_gas(self, msg):
        self.logic.on_gas(msg.data, now_sec(self))

    def _on_seek_status(self, msg):
        self.logic.on_seek_status(parse_json_msg(msg), now_sec(self))

    def _on_readings(self, msg):
        self.logic.on_gauge_readings(parse_json_msg(msg), now_sec(self))

    def _on_safety(self, msg):
        self.logic.on_safety_status(parse_json_msg(msg), now_sec(self))

    def _on_loco(self, msg):
        self.logic.on_locomotion_state(parse_json_msg(msg), now_sec(self))

    def _on_command(self, msg):
        cmd = (msg.data or "").strip()
        if cmd:
            self.get_logger().info(f"mission command: '{cmd}'")
            self.logic.command(cmd, now_sec(self))

    def _obstacle_frame_ok(self, frame_id):
        if frame_accepted(frame_id, self.p["obstacle_frames"]):
            return True
        self.logic.obstacles.note_rejected()
        now = time.monotonic()
        if now - self._frame_warn_t >= FRAME_WARN_PERIOD:
            self._frame_warn_t = now
            self.get_logger().warning(
                f"obstacle frame '{frame_id}' on {self.obstacle_topic} rejected (not in obstacle_frames "
                f"'{self.p['obstacle_frames']}'; {self.logic.obstacles.memory.rejected} rejected so far)")
        return False

    def _on_obstacle_cloud(self, msg):
        fid = msg.header.frame_id
        if not self._obstacle_frame_ok(fid):
            return
        t = now_sec(self)
        if not self.logic.obstacle_frame_due(t):          # not driving: obstacle_idle_period
            return
        frame = EvidenceFrame.from_elevation_cloud(organised_cloud_to_xyz(msg), fid,
                                                   float(self.p["obstacle_step_height"]),
                                                   int(self.p["obstacle_step_window"]))
        self.logic.on_obstacle_frame(frame, t)

    def _on_obstacle_grid(self, msg):
        fid = msg.header.frame_id
        if not self._obstacle_frame_ok(fid):
            return
        t = now_sec(self)
        if not self.logic.obstacle_frame_due(t):
            return
        frame = EvidenceFrame.from_traversability(OccupancyGridMap.from_msg(msg),
                                                  int(self.p["obstacle_cost_threshold"]))
        self.logic.on_obstacle_frame(frame, t)

    # ------------------------------------------------------------------ tick
    def _maybe_load_map_file(self):
        if self.logic.grid is not None or self._map_from_file_tried:
            return
        if time.monotonic() - self._map_wall_t0 < MAP_FILE_FALLBACK_AFTER:
            return
        self._map_from_file_tried = True
        path = self.p["map_yaml"] or self.layout.nav_map_yaml()
        if os.path.exists(path):
            try:
                self.logic.set_map(OccupancyGridMap.load(path))
                self.get_logger().warning(f"no /map received - loaded {path} directly")
            except Exception as exc:
                self.get_logger().error(f"could not load {path}: {exc}")
        else:
            self.get_logger().warning("no /map received and no map file - navigation will drive straight lines")

    def _tick(self):
        t = now_sec(self)
        self._maybe_load_map_file()
        try:
            out = self.logic.step(t)
            self._step_failures = 0
        except Exception as exc:  # noqa: BLE001 - keep the node (and /mission/state) alive
            out = None
            self._step_failures += 1
            self.get_logger().error(f"mission step failed ({self._step_failures}x): {type(exc).__name__}: {exc}"
                                    + chr(10) + traceback.format_exc())
            if self._step_failures >= STEP_FAILURES_BEFORE_ABORT:
                # a state the logic cannot leave: stop cleanly so that 't' starts afresh
                self._step_failures = 0
                self.logic.command("abort", t)
                self.logic.message = "mission aborted after an internal error - press 't' to start again"
                self.get_logger().error(self.logic.message)
                self.pub_cmd.publish(Twist())
        if out is not None:
            self._outputs(out)
        self._state_pub(t, force=out is not None and out.state_changed)

    def _publish_outputs(self, out):
        stamp = self.get_clock().now().to_msg()
        for c in out.locomotion:
            self.pub_loco.publish(String(data=c))
        if out.target is not None:
            self.pub_target.publish(String(data=out.target))
        if out.seek_enable is not None:
            self.pub_seek.publish(Bool(data=bool(out.seek_enable)))
        if out.path is not None:
            self.pub_path.publish(path_to_msg(out.path, frame_id="map", stamp=stamp))
        if out.cmd_vel is not None:
            m = Twist()
            m.linear.x, m.linear.y, m.angular.z = (float(v) for v in out.cmd_vel)
            self.pub_cmd.publish(m)
        for c in out.mission:
            self.pub_mcmd.publish(String(data=c))
        # events last: logging must never keep the commands of this tick from going out
        for level, text in out.events:
            self._log_event(level, text)

    def _log_event(self, level, text):
        """One call site per severity: rclpy keys a logger call site by file and
        line and raises 'Logger severity cannot be changed between calls' when the
        same line logs with another severity (field report on Lyrical: the first
        warning after info events killed the node on entering the factory)."""
        log = self.get_logger()
        if level in ("warn", "alarm"):
            log.warning(text)
        elif level == "error":
            log.error(text)
        else:
            log.info(text)

    def _publish_state(self, t, force=False):
        if force or (t - self._t_state_pub) >= float(self.p["state_publish_period"]) or t < self._t_state_pub:
            self._t_state_pub = t
            self.pub_state.publish(json_msg(self.logic.state_dict(t)))


def main(args=None):
    run_node(MissionManagerNode, args)


if __name__ == "__main__":
    main()
