"""
battery_simulator - state-of-charge simulation of the robot battery.

Subscribes
  /odom                nav_msgs/Odometry   (speed / yaw rate -> power, pose -> docking)
  /locomotion/state    std_msgs/String JSON (mode SIT/ESTOP/... -> lower posture power)
  /battery/set_level   std_msgs/Float32    test override, percent 0..100
Publishes
  /battery/state       sensor_msgs/BatteryState (percentage 0..1, voltage, current
                       (< 0 discharging), charge/capacity [Ah], power_supply_status)

Docking: within ``dock_tolerance`` of the layout dock pose and low-pass filtered
speed (time constant ``speed_filter_tau``; the body of a standing legged robot
sways, so the raw odometry twist is noisy) below ``dock_speed_max`` for
``dock_settle_s`` -> charging (CC/CV model).
"""
import math

from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32, String

from .battery_model import (STATUS_CHARGING, STATUS_DISCHARGING, STATUS_FULL, STATUS_NAMES,
                            BatteryModel, DockDetector)
from .layout_map import resolve_layout
from .node_common import (best_effort_qos, declare, finite_or_none, guarded, json_dict, now_sec,
                          odom_pose_twist_finite, reliable_qos, run_node, stamp, warn_throttled)
from .params import BATTERY_PARAMS, battery_config


class BatteryNode(Node):
    def __init__(self):
        super().__init__("battery_simulator")
        self.p = p = declare(self, BATTERY_PARAMS)
        layout = resolve_layout(p["layout_file"])
        self.model = BatteryModel(battery_config(p))
        dock = layout.dock_pose
        tol = p["dock_tolerance"] if p["dock_tolerance"] > 0.0 else \
            float(layout["charging_station"].get("dock_tolerance", 0.35))
        self.dock = DockDetector(dock[:2], tol, p["dock_speed_max"], p["dock_settle_s"])
        self.speed = 0.0
        self.speed_f = 0.0          # low-pass filtered speed (dock detection)
        self.wz = 0.0
        self.pose = None
        self.mode = "stand"
        self._t_last = None
        self._was_docked = False
        self._nominal_v = self.model.ocv(0.5)

        self.pub = self.create_publisher(BatteryState, "/battery/state", reliable_qos(10))
        # every callback is guarded: an exception is logged and the node keeps running
        g = lambda fn: guarded(self, fn.__name__, fn)  # noqa: E731
        self.create_subscription(Odometry, p["odom_topic"], g(self._on_odom), best_effort_qos(5))
        self.create_subscription(String, "/locomotion/state", g(self._on_loco), best_effort_qos(5))
        self.create_subscription(Float32, "/battery/set_level", g(self._on_set), reliable_qos(10))
        self.create_timer(1.0 / max(p["update_rate"], 0.5), g(self._on_update))
        self.create_timer(1.0 / max(p["publish_rate"], 0.1), g(self._publish))
        self.get_logger().info(
            f"battery {self.model.cfg.capacity_wh:.0f} Wh, start {self.model.percentage:.0f} %, dock at "
            f"({dock[0]:.2f}, {dock[1]:.2f}) tol {tol:.2f} m, time_scale {self.model.cfg.time_scale}")

    def _on_odom(self, msg: Odometry) -> None:
        v = odom_pose_twist_finite(msg)
        if v is None:   # NaN speed would poison the state of charge for good
            warn_throttled(self, "odom", "non-finite /odom sample ignored")
            return
        x, y, yaw, _vx, _vy, wz, speed = v
        self.pose = (x, y)
        self.speed = speed
        self.wz = wz

    def _on_loco(self, msg: String) -> None:
        mode = json_dict(msg).get("mode")
        mode = mode.lower() if isinstance(mode, str) else ""   # malformed / non-object JSON -> ignored
        if mode:
            self.mode = mode

    def _on_set(self, msg: Float32) -> None:
        pct = finite_or_none(msg.data)
        if pct is None:                       # NaN would poison the state of charge for good
            warn_throttled(self, "set_level", f"/battery/set_level {msg.data!r} ignored (not a finite number)")
            return
        self.model.set_level(pct)
        self.get_logger().info(f"battery level set to {self.model.percentage:.1f} % (test override)")
        self._publish()

    def _on_update(self) -> None:
        t = now_sec(self)
        if self._t_last is None or t < self._t_last:
            self._t_last = t
            return
        dt = min(t - self._t_last, 1.0)
        self._t_last = t
        tau = max(float(self.p["speed_filter_tau"]), 1e-3)
        self.speed_f += (1.0 - math.exp(-dt / tau)) * (self.speed - self.speed_f)
        docked = False
        if self.pose is not None and self.p["charge_on_dock"]:
            docked = self.dock.update(t, self.pose[0], self.pose[1], self.speed_f)
        if docked != self._was_docked:
            self._was_docked = docked
            self.get_logger().info("docked: charging" if docked else "undocked")
        mode = "walk" if self.speed > 0.02 else self.mode
        self.model.step(dt, self.speed, self.wz, mode, docked)

    def _publish(self) -> None:
        m = self.model
        msg = BatteryState()
        msg.header.stamp = stamp(self)
        msg.header.frame_id = "base_link"
        msg.voltage = float(m.voltage_v)
        msg.temperature = float("nan")
        msg.current = float(m.current_a)
        cap_ah = m.cfg.capacity_wh / self._nominal_v
        msg.charge = float(m.soc * cap_ah)
        msg.capacity = float(cap_ah)
        msg.design_capacity = float(cap_ah)
        msg.percentage = float(m.soc)
        st = m.status
        msg.power_supply_status = {STATUS_CHARGING: BatteryState.POWER_SUPPLY_STATUS_CHARGING,
                                   STATUS_DISCHARGING: BatteryState.POWER_SUPPLY_STATUS_DISCHARGING,
                                   STATUS_FULL: BatteryState.POWER_SUPPLY_STATUS_FULL}.get(
            st, BatteryState.POWER_SUPPLY_STATUS_UNKNOWN)
        msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
        msg.present = True
        msg.location = str(self.p["location"])
        msg.serial_number = str(self.p["serial_number"])
        self.pub.publish(msg)
        if not math.isfinite(msg.percentage):   # pragma: no cover - defensive
            self.get_logger().error(f"invalid battery state ({STATUS_NAMES.get(st)})")


def main(args=None):
    run_node(BatteryNode, args)


if __name__ == "__main__":
    main()
