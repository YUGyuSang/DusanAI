"""
gas_dispersion_sim - real-time (sim-time) gas leak + airflow simulation.

Publishes
  /gas/concentration_grid  nav_msgs/OccupancyGrid  (2 Hz, latched, 0..100; log scale
                           v = 100 log10(1+c)/log10(1+vis_max_ppm) by default,
                           obstacles = -1)
  /gas/field_info          std_msgs/String JSON {max_ppm, vis_max_ppm, scale, source, wind, t, ...}
  /gas/sensor              std_msgs/Float32 ppm at the robot nose (lag + noise), 10 Hz
  /gas/sensor_array        std_msgs/Float32MultiArray [FL, FR, RL, RR] ppm, 10 Hz
  /gas/wind                geometry_msgs/Vector3Stamped local wind at the robot (world frame)
Subscribes
  /odom                    nav_msgs/Odometry (robot pose)

The model (module3_gas_safety.gas_model) is spun up to steady state in the
constructor so the plume exists as soon as the node is up.
"""
import math
import time

import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension, String

from dusan_common.grid_map import scalar_field_to_occupancy_msg
from dusan_common.ros_helpers import json_msg

from .gas_model import GasDispersion2D, LaggedSensor
from .node_common import (best_effort_qos, declare, guarded, latched_qos, load_layout_and_map, now_sec,
                          odom_pose_twist_finite, reliable_qos, run_node, stamp, warn_throttled)
from .params import GAS_PARAMS, gas_config


class GasDispersionNode(Node):
    def __init__(self):
        super().__init__("gas_dispersion_sim")
        self.p = p = declare(self, GAS_PARAMS)
        self.layout, grid = load_layout_and_map(self, p["layout_file"], p["map_yaml"])
        cfg = gas_config(p, self.layout)

        t0 = time.monotonic()
        self.model = GasDispersion2D.from_nav_map(grid, cfg)
        t_build = time.monotonic() - t0
        t0 = time.monotonic()
        self.model.spin_up(float(p["spinup_seconds"]), float(p["spinup_dt"]), gusts=False)
        if p["spinup_gust_seconds"] > 0.0 and cfg.gusts_enabled:
            self.model.spin_up(float(p["spinup_gust_seconds"]), float(p["spinup_dt"]), gusts=True)
        t_spin = time.monotonic() - t0
        self.model.t = 0.0
        info = self.model.info()
        self.get_logger().info(
            f"gas model {self.model.shape[1]}x{self.model.shape[0]} @ {cfg.resolution} m: wind projection "
            f"{self.model.wind.last_projection_iters} it (|div| {self.model.wind.last_projection_residual:.1e} 1/s) "
            f"built in {t_build:.2f} s, spin-up {t_spin:.2f} s -> max {info['max_ppm']:.0f} ppm at "
            f"({info['max_xy'][0]:.2f}, {info['max_xy'][1]:.2f}); source {cfg.source_xy}, wind {cfg.wind_base}")

        rng = np.random.default_rng(int(p["sensor_seed"]))
        self.nose_offset = [tuple(p["nose_offset"][:2])]
        arr = [float(v) for v in p["array_offsets"]]
        self.array_offsets = [(arr[i], arr[i + 1]) for i in range(0, len(arr) - 1, 2)]
        self.nose = LaggedSensor(1, p["sensor_tau"], p["sensor_noise_ppm"], p["sensor_noise_rel"],
                                 p["sensor_quantum"], rng)
        self.array = LaggedSensor(len(self.array_offsets), p["sensor_tau"], p["sensor_noise_ppm"],
                                  p["sensor_noise_rel"], p["sensor_quantum"], rng)
        self.anemo = LaggedSensor(2, p["wind_sensor_tau"], p["wind_sensor_noise"], 0.0, 0.0, rng, floor=None)

        self.pose = None
        self._t_last = None
        self._step_ms = 0.0

        self.pub_grid = self.create_publisher(OccupancyGrid, "/gas/concentration_grid", latched_qos(1))
        self.pub_info = self.create_publisher(String, "/gas/field_info", reliable_qos(10))
        self.pub_sensor = self.create_publisher(Float32, "/gas/sensor", reliable_qos(10))
        self.pub_array = self.create_publisher(Float32MultiArray, "/gas/sensor_array", reliable_qos(10))
        self.pub_wind = self.create_publisher(Vector3Stamped, "/gas/wind", reliable_qos(10))
        g = lambda fn: guarded(self, fn.__name__, fn)  # noqa: E731  (exceptions logged, node keeps running)
        self.create_subscription(Odometry, p["odom_topic"], g(self._on_odom), best_effort_qos(5))
        self.create_timer(1.0 / max(p["update_rate"], 0.1), g(self._on_update))
        self.create_timer(1.0 / max(p["grid_rate"], 0.05), g(self._on_grid))
        self._publish_grid()   # latched snapshot of the spun-up plume right away

    # ------------------------------------------------------------ callbacks
    def _on_odom(self, msg: Odometry) -> None:
        v = odom_pose_twist_finite(msg)
        if v is None:   # NaN / inf would crash the grid lookups in the timer
            warn_throttled(self, "odom", "non-finite /odom sample ignored")
            return
        self.pose = (v[0], v[1], v[2])

    def _on_update(self) -> None:
        t = now_sec(self)
        if self._t_last is None or t < self._t_last:   # first call / sim reset
            self._t_last = t
            return
        dt = t - self._t_last
        if dt <= 0.0:
            return
        self._t_last = t
        dt = min(dt, float(self.p["max_dt"]))
        t0 = time.perf_counter()
        self.model.step(dt)
        self._step_ms = 0.9 * self._step_ms + 0.1 * (time.perf_counter() - t0) * 1e3
        if self.pose is None:
            return
        x, y, yaw = self.pose
        pts = self.model.body_points(x, y, yaw, self.nose_offset + self.array_offsets)
        c = self.model.concentration_at(pts[:, 0], pts[:, 1])
        nose = float(self.nose.update(c[:1], dt)[0])
        arr = self.array.update(c[1:], dt)
        u, v = self.model.wind_at(x, y)
        w = self.anemo.update([u, v], dt)

        self.pub_sensor.publish(Float32(data=nose))
        ma = Float32MultiArray()
        ma.layout.dim = [MultiArrayDimension(label="front_left,front_right,rear_left,rear_right",
                                             size=len(arr), stride=len(arr))]
        ma.data = [float(a) for a in arr]
        self.pub_array.publish(ma)
        wm = Vector3Stamped()
        wm.header.stamp = stamp(self)
        wm.header.frame_id = self.p["frame_id"]
        wm.vector.x, wm.vector.y = float(w[0]), float(w[1])
        self.pub_wind.publish(wm)

    def _on_grid(self) -> None:
        self._publish_grid()

    def _publish_grid(self) -> None:
        p = self.p
        m = self.model
        c = m.c
        vmax = max(float(p["vis_max_ppm"]), 1e-6)
        if p["vis_log_scale"]:
            f = np.log10(1.0 + c) / math.log10(1.0 + vmax)
            f[c < float(p["vis_min_ppm"])] = 0.0
            lo, hi, scale = 0.0, 1.0, "log"
        else:
            f, lo, hi, scale = c.copy(), 0.0, vmax, "linear"
        f = np.where(m.fluid, f, np.nan)
        msg = scalar_field_to_occupancy_msg(f, m.h, m.x0, m.y0, lo, hi, frame_id=p["frame_id"], stamp=stamp(self))
        self.pub_grid.publish(msg)
        info = m.info()
        wx, wy = m.mean_wind()
        self.pub_info.publish(json_msg({
            "max_ppm": info["max_ppm"],
            "max_xy": info["max_xy"],
            "vis_max_ppm": vmax,
            "scale": scale,
            "scale_formula": "ppm = (1+vis_max_ppm)**(v/100) - 1" if scale == "log" else "ppm = v/100*vis_max_ppm",
            "source": list(m.cfg.source_xy),
            "emission_rate": m.cfg.emission_rate,
            "wind": [wx, wy],
            "wind_base": list(m.cfg.wind_base),
            "gust_speed_factor": info["gust_speed_factor"],
            "gust_angle_deg": info["gust_angle_deg"],
            "t": m.t,
            "step_ms": self._step_ms,
            "substeps": info["substeps"],
            "resolution": m.h,
            "mass": info["mass"],
        }))


def main(args=None):
    run_node(GasDispersionNode, args)


if __name__ == "__main__":
    main()
