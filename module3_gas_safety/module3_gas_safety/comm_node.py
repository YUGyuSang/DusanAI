"""
comm_simulator - Wi-Fi coverage (RSSI) simulation.

Publishes
  /comm/rssi_map   nav_msgs/OccupancyGrid (latched): 0..100 <-> vis_min..vis_max dBm
                   (default -100..-30 dBm, linear), pre-computed on a coarse grid
  /comm/rssi       std_msgs/Float32 RSSI [dBm] at the robot, 5 Hz (exact point model
                   + slow log-normal fading)
Subscribes
  /odom            nav_msgs/Odometry

Model: log-distance path loss per access point, wall attenuation per
free->occupied crossing on the nav map, extra attenuation in shielded zones
(see module3_gas_safety.comm_model).
"""
import time

from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Float32

from .comm_model import CommModel, FadingProcess, config_from_layout
from .node_common import (best_effort_qos, declare, latched_qos, load_layout_and_map, now_sec,
                          odom_pose_twist_finite, reliable_qos, run_node, stamp, warn_throttled)
from .params import COMM_PARAMS


class CommNode(Node):
    def __init__(self):
        super().__init__("comm_simulator")
        self.p = p = declare(self, COMM_PARAMS)
        self.layout, grid = load_layout_and_map(self, p["layout_file"], p["map_yaml"])
        cfg = config_from_layout(self.layout)
        cfg.antenna_height = float(p["antenna_height"])
        cfg.map_resolution = float(p["map_resolution"])
        cfg.vis_min_dbm = float(p["vis_min_dbm"])
        cfg.vis_max_dbm = float(p["vis_max_dbm"])
        if p["path_loss_exponent"] > 0.0:
            cfg.path_loss_exponent = float(p["path_loss_exponent"])
        if p["wall_attenuation_db"] >= 0.0:
            cfg.wall_attenuation_db = float(p["wall_attenuation_db"])
        cfg.rf_min_obstacle_area_m2 = float(p["rf_min_obstacle_area_m2"])
        self.model = CommModel(grid, cfg)
        t0 = time.monotonic()
        self.rssi_map = self.model.compute_map(cfg.map_resolution)
        self.get_logger().info(
            f"RSSI map {self.rssi_map.shape[1]}x{self.rssi_map.shape[0]} @ {cfg.map_resolution} m for "
            f"{len(cfg.access_points)} APs in {time.monotonic() - t0:.2f} s "
            f"(min {self.rssi_map.field.min():.1f} dBm, shadow < {cfg.rssi_min_dbm:.0f} dBm)")
        self.fading = FadingProcess(p["fading_sigma_db"], p["fading_tau_s"], int(p["seed"]))
        self.pose = None
        self._t_last = None

        self.pub_map = self.create_publisher(OccupancyGrid, "/comm/rssi_map", latched_qos(1))
        self.pub_rssi = self.create_publisher(Float32, "/comm/rssi", reliable_qos(10))
        self.create_subscription(Odometry, p["odom_topic"], self._on_odom, best_effort_qos(5))
        self.create_timer(1.0 / max(p["publish_rate"], 0.1), self._on_timer)
        if p["map_republish_period"] > 0.0:
            self.create_timer(float(p["map_republish_period"]), self._publish_map)
        self._publish_map()

    def _publish_map(self) -> None:
        m = self.rssi_map
        msg = OccupancyGrid()
        msg.header.frame_id = self.p["frame_id"]
        msg.header.stamp = stamp(self)
        msg.info.resolution = float(m.resolution)
        msg.info.width = int(m.shape[1])
        msg.info.height = int(m.shape[0])
        msg.info.origin.position.x = float(m.origin_x)
        msg.info.origin.position.y = float(m.origin_y)
        msg.info.origin.orientation.w = 1.0
        msg.data = m.to_grid_values().flatten().astype(int).tolist()
        self.pub_map.publish(msg)

    def _on_odom(self, msg: Odometry) -> None:
        v = odom_pose_twist_finite(msg)
        if v is None:   # NaN / inf would crash the RSSI map lookup in the timer
            warn_throttled(self, "odom", "non-finite /odom sample ignored")
            return
        self.pose = (v[0], v[1])

    def _on_timer(self) -> None:
        t = now_sec(self)
        dt = 0.0 if self._t_last is None else max(0.0, t - self._t_last)
        self._t_last = t
        if self.pose is None:
            return
        rssi = self.model.rssi_at(*self.pose) + self.fading.step(dt)
        self.pub_rssi.publish(Float32(data=float(rssi)))


def main(args=None):
    run_node(CommNode, args)


if __name__ == "__main__":
    main()
