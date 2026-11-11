"""
return_to_home - safety supervisor: battery / energy / comm-shadow triggered RTH.

Subscribes
  /odom               nav_msgs/Odometry
  /battery/state      sensor_msgs/BatteryState
  /comm/rssi          std_msgs/Float32 [dBm]
  /comm/rssi_map      nav_msgs/OccupancyGrid (latched; replaces the local RSSI model)
  /nav/plan           nav_msgs/Path (path being followed by the mission; ignored while RTH runs)
  /mission/command    std_msgs/String  'rth' / 'dock' -> manual RTH, 'resume' -> leave the charger,
                      'pause' / 'resume' -> hold / continue a return, 'abort' -> cancel a return
                      (refused while battery low / energy reserve / comm shadow / comm lost holds)
  /locomotion/state   std_msgs/String JSON (stand up first if the robot sits when RTH starts)
  /map                nav_msgs/OccupancyGrid (latched)
  obstacle_elevation_topic (/elevation_map/points, sensor_msgs/PointCloud2) or, with
                      obstacle_source traversability, obstacle_traversability_topic (OccupancyGrid):
                      novel obstacles on the way home (obstacle_enabled)
Publishes
  /cmd_vel/rth        geometry_msgs/Twist (while RTH drives)
  /nav/plan           nav_msgs/Path (home path, latched; an empty Path when the return ends)
  /locomotion/command std_msgs/String 'sit' on the dock, 'stand' when charged / resumed
  /safety/status      std_msgs/String JSON (DESIGN_SPEC section 4)
  /safety/rth_active  std_msgs/Bool
"""
import threading
from collections import deque

import numpy as np

from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import BatteryState, PointCloud2
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist

from dusan_common.grid_map import OccupancyGridMap
from dusan_common.obstacle_layer import EvidenceFrame, frame_accepted, organised_cloud_to_xyz
from dusan_common.planning import path_to_msg
from dusan_common.ros_helpers import json_msg

from .battery_model import BatteryConfig, BatteryModel
from .comm_model import CommModel, RssiMap, config_from_layout
from .layout_map import map_fingerprint
from .node_common import (best_effort_qos, declare, finite_or_none, guarded, json_dict, latched_qos,
                          load_layout_and_map, now_sec, odom_pose_twist_finite, reliable_qos, run_node, stamp,
                          subscribe_latched, twist_msg, warn_throttled)
from .params import RTH_PARAMS, rth_config
from .rth_logic import MONITOR, GeodesicField, PlannerWorker, RthInputs, RthLogic, terrain_cost_for


class ReturnToHomeNode(Node):
    def __init__(self):
        super().__init__("return_to_home")
        self.p = p = declare(self, RTH_PARAMS)
        self.layout, self.grid = load_layout_and_map(self, p["layout_file"], p["map_yaml"])
        self._map_fp = map_fingerprint(self.grid)
        cfg = rth_config(p, self.layout)
        self.dock = self.layout.dock_pose
        bat = BatteryModel(BatteryConfig(capacity_wh=p["capacity_wh"], p_compute_w=p["p_compute_w"],
                                         p_sensors_w=p["p_sensors_w"], p_stand_w=p["p_stand_w"],
                                         p_sit_w=p["p_sit_w"], p_per_mps_w=p["p_per_mps_w"],
                                         p_per_radps_w=p["p_per_radps_w"]))
        self._rssi_map = None
        self._rssi_source = "none"
        if p["local_comm_model"]:
            try:
                ccfg = config_from_layout(self.layout, rf_min_obstacle_area_m2=float(p["rf_min_obstacle_area_m2"]))
                self._rssi_map = CommModel(self.grid, ccfg).compute_map(p["comm_map_resolution"])
                self._rssi_source = "local model"
            except Exception as exc:
                self.get_logger().warning(f"local RSSI model failed: {exc}")
        self._geo = None
        # layout debris (pipes, rubble, plates, platform) as extra A* cost of the path home
        self.planner = PlannerWorker(self.grid, cfg.robot_radius, cfg.robot_radius + cfg.smooth_margin,
                                     cost_fn=terrain_cost_for(cfg, self.layout))
        self.logic = RthLogic(cfg, self.grid, self.dock, bat, rssi_lookup=self._rssi_lookup,
                              home_distance=self._home_distance, planner=self.planner,
                              obstacle_source=str(p["obstacle_source"]))
        self._start_geodesic(self.grid)

        self.pose = None
        self.vel = (0.0, 0.0)
        self.battery_pct = None
        self.charging = None
        self.rssi = None
        self.plan_in = None
        self._own_plan_sig = None
        self.commands = deque(maxlen=8)   # /mission/command strings, one consumed per tick
        self.loco_mode = ""
        self._last_state = MONITOR
        self._last_active = None
        self.obst_in = None               # latest EvidenceFrame (handed to the logic on the next tick)

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel/rth", reliable_qos(10))
        self.pub_plan = self.create_publisher(Path, "/nav/plan", latched_qos(1))
        self.pub_loco = self.create_publisher(String, "/locomotion/command", reliable_qos(10))
        self.pub_status = self.create_publisher(String, "/safety/status", reliable_qos(10))
        self.pub_active = self.create_publisher(Bool, "/safety/rth_active", reliable_qos(10))

        # every callback is guarded: an exception is logged and the node keeps running (a dead
        # return_to_home would silently end the battery / comm-shadow supervision of the run)
        g = lambda fn: guarded(self, fn.__name__, fn)  # noqa: E731
        self.create_subscription(Odometry, p["odom_topic"], g(self._on_odom), best_effort_qos(5))
        self.create_subscription(BatteryState, "/battery/state", g(self._on_battery), best_effort_qos(5))
        self.create_subscription(Float32, "/comm/rssi", g(self._on_rssi), best_effort_qos(5))
        subscribe_latched(self, OccupancyGrid, "/comm/rssi_map", g(self._on_rssi_map))
        subscribe_latched(self, Path, "/nav/plan", g(self._on_plan))
        self.create_subscription(String, "/mission/command", g(self._on_command), reliable_qos(10))
        self.create_subscription(String, "/locomotion/state", g(self._on_loco), best_effort_qos(5))
        if p["use_map_topic"]:
            subscribe_latched(self, OccupancyGrid, "/map", g(self._on_map))
        # novel obstacles from terrain_mapper (RELIABLE depth-1 publisher; best effort depth 1 = latest frame)
        self.obstacle_topic = None
        if p["obstacle_enabled"]:
            if str(p["obstacle_source"]).strip().lower() == "traversability":
                self.obstacle_topic = p["obstacle_traversability_topic"]
                self.create_subscription(OccupancyGrid, self.obstacle_topic, g(self._on_obstacle_grid),
                                         best_effort_qos(1))
            else:
                self.obstacle_topic = p["obstacle_elevation_topic"]
                self.create_subscription(PointCloud2, self.obstacle_topic, g(self._on_obstacle_cloud),
                                         best_effort_qos(1))

        self.create_timer(1.0 / max(p["rate"], 1.0), g(self._on_timer))
        self.create_timer(1.0 / max(p["status_rate"], 0.1), g(self._publish_status))
        self.create_timer(1.0 / max(p["active_rate"], 0.1), g(self._publish_active))
        self._last_out = None
        self.get_logger().info(
            f"RTH supervisor: battery < {cfg.rth_battery_pct:.0f} %, resume at {cfg.resume_battery_pct:.0f} %, "
            f"energy x{cfg.energy_safety_factor} + {cfg.energy_margin_pct} %, shadow < {cfg.rssi_min_dbm:.0f} dBm "
            f"({cfg.shadow_lookahead_m} m / {cfg.shadow_horizon_s} s look-ahead), dock {self.dock}")

    # ---------------------------------------------------------- helpers
    def _start_geodesic(self, grid) -> None:
        def work():
            try:
                self._geo = GeodesicField(grid, self.dock[:2], self.logic.cfg.robot_radius,
                                          self.logic.cfg.geodesic_downsample)
            except Exception as exc:  # pragma: no cover
                self.get_logger().warning(f"geodesic distance field failed: {exc}")
        threading.Thread(target=work, daemon=True).start()

    def _home_distance(self, x, y):
        geo = self._geo
        if geo is None:
            return 1.4 * self.logic.dist_to_dock(x, y)
        return geo.distance(x, y)

    def _rssi_lookup(self, xs, ys):
        m = self._rssi_map
        if m is None:
            return np.full(np.shape(xs), 0.0)    # unknown coverage: never predicts a shadow
        return m.at(xs, ys)

    # -------------------------------------------------------- callbacks
    def _on_odom(self, msg: Odometry) -> None:
        v = odom_pose_twist_finite(msg)
        if v is None:   # NaN / inf would crash the map lookups in the timer
            warn_throttled(self, "odom", "non-finite /odom sample ignored")
            return
        x, y, yaw, vx, vy, _wz, _sp = v
        self.pose = (x, y, yaw)
        self.vel = (vx, vy)

    def _on_battery(self, msg: BatteryState) -> None:
        pct = finite_or_none(msg.percentage)
        if pct is not None:   # not NaN / inf
            self.battery_pct = 100.0 * pct if pct <= 1.0 + 1e-6 else pct
        self.charging = msg.power_supply_status in (BatteryState.POWER_SUPPLY_STATUS_CHARGING,
                                                    BatteryState.POWER_SUPPLY_STATUS_FULL)

    def _on_rssi(self, msg: Float32) -> None:
        rssi = finite_or_none(msg.data)
        if rssi is not None:
            self.rssi = rssi

    def _on_rssi_map(self, msg: OccupancyGrid) -> None:
        if msg.info.width <= 0 or msg.info.width * msg.info.height != len(msg.data):
            warn_throttled(self, "rssi_map", f"/comm/rssi_map ignored: {len(msg.data)} cells for "
                                             f"{msg.info.width}x{msg.info.height}", 30.0)
            return
        self._rssi_map = RssiMap.from_grid_values(
            np.array(msg.data, dtype=float).reshape(msg.info.height, msg.info.width),
            msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y)
        if self._rssi_source != "/comm/rssi_map":
            self._rssi_source = "/comm/rssi_map"
            self.get_logger().info("RSSI map received from comm_simulator")

    def _on_plan(self, msg: Path) -> None:
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        sig = (len(pts), tuple(np.round(pts[0], 2)) if pts else None, tuple(np.round(pts[-1], 2)) if pts else None)
        if sig == self._own_plan_sig:
            return            # our own home path echoed back
        self.plan_in = pts

    def _on_command(self, msg: String) -> None:
        cmd = (msg.data or "").strip().lower()
        if cmd:
            self.commands.append(cmd)

    def _on_loco(self, msg: String) -> None:
        mode = json_dict(msg).get("mode")
        self.loco_mode = mode.upper() if isinstance(mode, str) else ""   # malformed JSON -> ''

    def _obstacle_frame(self, msg, convert) -> None:
        """Frame check + conversion; a malformed message is logged, never raised."""
        fid = msg.header.frame_id
        if not frame_accepted(fid, self.p["obstacle_frames"]):
            self.logic.obstacles.note_rejected()
            warn_throttled(self, "obstacle_frame",
                           f"obstacle frame '{fid}' on {self.obstacle_topic} rejected (obstacle_frames "
                           f"'{self.p['obstacle_frames']}', {self.logic.obstacles.memory.rejected} so far)", 30.0)
            return
        if self.pose is None or not self.logic.obstacle_frame_due(now_sec(self)):
            return                       # not returning: one frame per obstacle_idle_period
        try:
            self.obst_in = convert(msg, fid)
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
        except Exception as exc:
            self.get_logger().warning(f"/map ignored: {exc}")
            return
        fp = map_fingerprint(grid)
        if fp == self._map_fp:
            return
        self._map_fp = fp
        self.grid = grid
        self.logic.set_grid(grid)
        self._start_geodesic(grid)
        self.get_logger().info(f"nav map updated from /map ({grid.width}x{grid.height})")

    # ------------------------------------------------------------ loop
    def _on_timer(self) -> None:
        if self.pose is None:
            return
        t = now_sec(self)
        x, y, yaw = self.pose
        plan, self.plan_in = self.plan_in, None
        cmd = self.commands.popleft() if self.commands else None
        frame, self.obst_in = self.obst_in, None
        inp = RthInputs(t=t, x=x, y=y, yaw=yaw, vx=self.vel[0], vy=self.vel[1],
                        battery_pct=self.battery_pct, charging=self.charging, rssi_dbm=self.rssi,
                        plan=plan, command=cmd, obstacle_frame=frame)
        was = self.logic.state
        out = self.logic.update(inp)
        self._last_out = out
        if was == MONITOR and self.logic.state != MONITOR:
            self.get_logger().warning(f"RETURN TO HOME triggered: {self.logic.reason} "
                                      f"(battery {self.battery_pct}, predicted RSSI {self.logic.predicted_min_rssi})")
            if self.loco_mode == "SIT":
                self.pub_loco.publish(String(data="stand"))
        if out.plan is not None:
            if out.plan:
                self._own_plan_sig = (len(out.plan), tuple(np.round(out.plan[0], 2)),
                                      tuple(np.round(out.plan[-1], 2)))
            else:
                # return ended: replace the latched home path by an empty Path
                # (late joiners must not get a stale home path); no echo
                # filter for it, so the mission's own empty plans still clear
                # the comm-shadow look-ahead
                self._own_plan_sig = None
            self.pub_plan.publish(path_to_msg(out.plan, self.p["frame_id"], stamp(self)))
        if out.locomotion:
            self.pub_loco.publish(String(data=out.locomotion))
        if out.cmd is not None and out.rth_active:
            self.pub_cmd.publish(twist_msg(*out.cmd))
        if self.logic.state != self._last_state:
            self.get_logger().info(f"safety state -> {self.logic.state} ({self.logic.message})")
            self._last_state = self.logic.state
            self._publish_status()
            self._publish_active()

    def _publish_status(self) -> None:
        if self._last_out is None:
            st = self.logic.status(None)
            st["battery_pct"] = self.battery_pct
            st["rssi_dbm"] = self.rssi
        else:
            st = dict(self._last_out.status)
        st["rssi_map_source"] = self._rssi_source
        st["geodesic_ready"] = self._geo is not None
        self.pub_status.publish(json_msg(st))

    def _publish_active(self) -> None:
        self.pub_active.publish(Bool(data=bool(self.logic.active)))


def main(args=None):
    run_node(ReturnToHomeNode, args)


if __name__ == "__main__":
    main()
