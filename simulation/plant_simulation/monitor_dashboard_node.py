"""
monitor_dashboard node: digital-twin control room at http://<host>:<port>.

* subscribes to every module's status topics and feeds them into the ROS-free
  aggregator (plant_simulation.dashboard_state.DashboardState),
* serves the web UI (simulation/web) + Server-Sent Events + rendered layers
  through plant_simulation.web_server.DashboardServer (stdlib http.server in a
  daemon thread),
* forwards operator commands (POST /api/command) to /mission/command,
  /locomotion/command or /battery/set_level,
* publishes /twin/markers (MarkerArray) for RViz: gauges coloured by status
  with values, hot spots, leak truth vs estimate, dock, access points.
"""
import os
import threading
import time

import rclpy  # noqa: F401
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import BatteryState, Image
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np

from dusan_common.geometry import odom_to_pose
from dusan_common.image_utils import image_msg_to_bgr
from dusan_common.layout import Layout
from dusan_common.ros_helpers import declare_params, latched_qos, make_marker, parse_json_msg, stamp_to_sec

from plant_simulation.dashboard_render import (encode_jpeg, placeholder_jpeg, render_gas_png, render_map_png,
                                               render_rssi_png)
from plant_simulation.dashboard_state import DASHBOARD_DEFAULTS, DashboardState
from plant_simulation.node_utils import (best_effort_qos, reliable_qos, run_node, subscribe_any,
                                         subscribe_latched)
from plant_simulation.web_server import DashboardServer

# status colours (RGBA 0..1) - same tokens as the web UI
STATUS_RGBA = {
    "normal": (0.047, 0.639, 0.047, 1.0),     # good     #0ca30c
    "warning": (0.980, 0.698, 0.098, 1.0),    # warning  #fab219
    "alarm": (0.816, 0.231, 0.231, 1.0),      # critical #d03b3b
}
MUTED_RGBA = (0.537, 0.529, 0.506, 1.0)       # #898781
AQUA_RGBA = (0.098, 0.620, 0.439, 1.0)        # #199e70
INK_RGBA = (1.0, 1.0, 1.0, 1.0)


def _default_web_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        d = os.path.join(get_package_share_directory("plant_simulation"), "web")
        if os.path.isdir(d):
            return d
    except Exception:
        pass
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web"))


def _grid_array(msg):
    return np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)


def _xyz(pos, fallback=None):
    """First of (pos, fallback) that is a finite [x, y, z]; a finite [x, y]
    takes z from the fallback (or 0).  None if neither is usable."""
    def finite3(p):
        try:
            v = [float(c) for c in p[:3]]
        except (TypeError, ValueError, IndexError):
            return None
        return v if len(v) == 3 and all(np.isfinite(v)) else None

    p = finite3(pos) if isinstance(pos, (list, tuple)) else None
    if p is not None:
        return p
    fb = finite3(fallback) if isinstance(fallback, (list, tuple)) else None
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            x, y = float(pos[0]), float(pos[1])
            if np.isfinite(x) and np.isfinite(y):
                return [x, y, fb[2] if fb is not None else 0.0]
        except (TypeError, ValueError):
            pass
    return fb


class MonitorDashboardNode(Node):
    def __init__(self):
        super().__init__("monitor_dashboard")
        self.p = declare_params(self, DASHBOARD_DEFAULTS)
        self.layout = Layout.load(self.p["layout_file"] or None)
        self.state = DashboardState(self.layout, history_s=self.p["history_s"], max_events=self.p["max_events"],
                                    path_max_points=self.p["path_max_points"], gas_alarm_ppm=self.p["gas_alarm_ppm"])
        self._pub_lock = threading.Lock()
        self._img_lock = threading.Lock()
        self._img_msgs = {"gauge": None, "thermal": None}
        self._img_cache = {}
        self._layer_cache = {}

        # ------------------------------------------------------- publishers
        self.pub_mission = self.create_publisher(String, "/mission/command", reliable_qos(10))
        self.pub_loco = self.create_publisher(String, "/locomotion/command", reliable_qos(10))
        self.pub_batt = self.create_publisher(Float32, "/battery/set_level", reliable_qos(10))
        self.pub_markers = self.create_publisher(MarkerArray, "/twin/markers", latched_qos())

        # ------------------------------------------------------ subscriptions
        S = self.state
        be = best_effort_qos
        self.create_subscription(Odometry, "/odom", self._on_odom, be(5))
        subscribe_any(self, String, "/locomotion/state", lambda m: S.set_locomotion(parse_json_msg(m)))
        self.create_subscription(String, "/locomotion/fall_event", lambda m: S.set_fall_event(parse_json_msg(m)),
                                 reliable_qos(10))
        subscribe_any(self, String, "/terrain/info", lambda m: S.set_terrain(parse_json_msg(m)))
        subscribe_any(self, String, "/mission/state", lambda m: S.set_mission(parse_json_msg(m)))
        subscribe_any(self, String, "/safety/status", lambda m: S.set_safety(parse_json_msg(m)))
        subscribe_any(self, Bool, "/safety/rth_active", lambda m: S.set_flag("rth_active", m.data))
        subscribe_any(self, Bool, "/teleop/active", lambda m: S.set_flag("teleop_active", m.data))
        self.create_subscription(BatteryState, "/battery/state", self._on_battery, be(5))
        self.create_subscription(Float32, "/comm/rssi", lambda m: S.set_rssi(m.data), be(5))
        subscribe_latched(self, OccupancyGrid, "/comm/rssi_map", lambda m: self._on_grid("rssi", m))
        self.create_subscription(Float32, "/gas/sensor", lambda m: S.set_gas_ppm(m.data), be(10))
        self.create_subscription(OccupancyGrid, "/gas/concentration_grid", lambda m: self._on_grid("gas", m), be(2))
        subscribe_any(self, String, "/gas/field_info", lambda m: S.set_gas_field(parse_json_msg(m)))
        subscribe_any(self, String, "/gas/seek_status", lambda m: S.set_seek_status(parse_json_msg(m)))
        self.create_subscription(PointStamped, "/gas/source_estimate",
                                 lambda m: S.set_source_estimate(m.point.x, m.point.y), be(5))
        self.create_subscription(Path, "/gas/seek_path", lambda m: self._on_path("seek", m), be(2))
        # gauge readings / thermal alerts update per-id tables idempotently
        self.create_subscription(String, "/inspection/gauge_readings",
                                 lambda m: S.set_gauge_readings(parse_json_msg(m)), reliable_qos(20))
        self.create_subscription(String, "/inspection/thermal_alerts",
                                 lambda m: S.set_thermal_alerts(parse_json_msg(m)), reliable_qos(10))
        self.create_subscription(Image, "/inspection/gauge_overlay", lambda m: self._on_image("gauge", m), be(2))
        self.create_subscription(Image, "/inspection/thermal_fusion", lambda m: self._on_image("thermal", m), be(2))
        subscribe_latched(self, String, "/patrol/plan_json", lambda m: S.set_plan(parse_json_msg(m)))
        subscribe_latched(self, Path, "/patrol/route", lambda m: self._on_path("route", m))
        subscribe_latched(self, Path, "/nav/plan", lambda m: self._on_path("nav", m))
        self.create_subscription(Path, "/mpc/predicted_trajectory", lambda m: self._on_path("mpc", m), be(2))
        self.create_subscription(OccupancyGrid, "/map", lambda m: self._on_grid("map", m), latched_qos())
        self.create_subscription(Twist, "/cmd_vel", lambda m: S.set_cmd_vel(m.linear.x, m.linear.y, m.angular.z),
                                 be(5))
        subscribe_any(self, String, "/cmd_vel_mux/active", lambda m: S.set_mux(m.data))

        # ------------------------------------------------------------ server
        web_dir = self.p["web_dir"] or _default_web_dir()
        self.server = DashboardServer(self.state.snapshot, lambda: self.state.static, web_dir,
                                      port=int(self.p["port"]), host=str(self.p["host"]),
                                      sse_rate=float(self.p["sse_rate"]), command_fn=self._command,
                                      blob_fn=self._blob, logger=self._log,
                                      port_fallbacks=int(self.p["port_fallbacks"]))
        try:
            port = self.server.start()
            host = self.p["host"] if self.p["host"] not in ("0.0.0.0", "") else "localhost"
            self.get_logger().info(f"digital twin dashboard: http://{host}:{port}  (web dir {web_dir})")
        except OSError as exc:
            self.get_logger().error(f"dashboard HTTP server not started: {exc}")
        self.create_timer(1.0 / max(0.1, float(self.p["markers_rate"])), self._publish_markers)

    # ------------------------------------------------------------- logging
    def _log(self, level, msg):
        if level == "debug":
            self.get_logger().debug(msg)
        elif level == "warn":
            self.get_logger().warning(msg)
        elif level == "error":
            self.get_logger().error(msg)
        else:
            self.get_logger().info(msg)

    # ----------------------------------------------------------- callbacks
    def _on_odom(self, msg):
        x, y, z, roll, pitch, yaw = odom_to_pose(msg)
        tw = msg.twist.twist
        self.state.set_robot(x, y, z, roll, pitch, yaw, tw.linear.x, tw.linear.y, tw.angular.z,
                             t_sim=stamp_to_sec(msg.header.stamp))

    def _on_battery(self, msg):
        p = float(msg.percentage)
        pct = p * 100.0 if p <= 1.0 else p          # tolerate publishers that send 0..100
        charging = int(msg.power_supply_status) == int(BatteryState.POWER_SUPPLY_STATUS_CHARGING)
        self.state.set_battery(pct, msg.voltage, msg.current, charging)

    def _on_grid(self, name, msg):
        try:
            arr = _grid_array(msg)
        except ValueError as exc:
            self.get_logger().warning(f"bad grid on {name}: {exc}", throttle_duration_sec=10.0)
            return
        o = msg.info.origin.position
        self.state.set_layer(name, arr, o.x, o.y, msg.info.resolution)

    def _on_path(self, name, msg):
        self.state.set_path(name, [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses])

    def _on_image(self, name, msg):
        with self._img_lock:
            self._img_msgs[name] = msg
        self.state.touch_image(name, {"gauge": "/inspection/gauge_overlay",
                                      "thermal": "/inspection/thermal_fusion"}[name])

    # ------------------------------------------------------- HTTP helpers
    def _command(self, target, command):
        with self._pub_lock:
            if target == "mission":
                self.pub_mission.publish(String(data=command))
            elif target == "locomotion":
                self.pub_loco.publish(String(data=command))
            elif target == "battery":
                self.pub_batt.publish(Float32(data=float(command)))
            else:
                return False, f"unknown target {target}"
        self.state.add_event("info", "operator", f"{target} command '{command}'")
        self.get_logger().info(f"operator command: {target} '{command}'")
        return True, f"{target} '{command}' sent"

    def _blob(self, name):
        if name.startswith("image/"):
            return self._image_blob(name.split("/", 1)[1].split(".")[0])
        layer = name.split(".")[0]
        return self._layer_blob(layer)

    def _image_blob(self, key):
        now = time.monotonic()
        min_dt = 1.0 / max(0.1, float(self.p["image_max_rate"]))
        with self._img_lock:
            cached = self._img_cache.get(key)
            if cached is not None and now - cached[0] < min_dt:
                return cached[1], "image/jpeg"
            msg = self._img_msgs.get(key)
        if msg is None:
            data = placeholder_jpeg("no signal: " + {"gauge": "/inspection/gauge_overlay",
                                                     "thermal": "/inspection/thermal_fusion"}.get(key, key))
        else:
            try:
                data = encode_jpeg(image_msg_to_bgr(msg), int(self.p["jpeg_quality"]), int(self.p["image_max_width"]))
            except Exception as exc:
                self.get_logger().warning(f"cannot convert {key} image: {exc}", throttle_duration_sec=10.0)
                data = placeholder_jpeg(f"unsupported encoding {getattr(msg, 'encoding', '?')}")
        with self._img_lock:
            self._img_cache[key] = (now, data)
        return data, "image/jpeg"

    def _layer_blob(self, layer):
        if layer not in ("map", "gas", "rssi"):
            return None
        arr, meta = self.state.get_layer(layer)
        if arr is None:
            return None
        now = time.monotonic()
        cached = self._layer_cache.get(layer)
        if cached is not None:
            ver, t, data = cached
            min_dt = 1.0 / max(0.1, float(self.p["gas_png_max_rate"])) if layer == "gas" else 0.0
            if ver == meta["version"] or now - t < min_dt:
                return data, "image/png"
        if layer == "map":
            data = render_map_png(arr)
        elif layer == "gas":
            data = render_gas_png(arr)
        else:
            data = render_rssi_png(arr, float(self.layout.comm.get("rssi_min_dbm", -80.0)))
        self._layer_cache[layer] = (meta["version"], now, data)
        return data, "image/png"

    # -------------------------------------------------------------- markers
    def _publish_markers(self):
        # an exception escaping a timer callback would stop rclpy.spin (and
        # with it the whole dashboard): a malformed module message must only
        # cost one marker update
        try:
            arr = self._build_markers()
        except Exception as exc:
            self.get_logger().warning(f"/twin/markers not published: {exc!r}", throttle_duration_sec=10.0)
            return
        self.pub_markers.publish(arr)

    def _build_markers(self):
        snap = self.state.snapshot(max_events=0)
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        # the set of markers changes over time (leak estimate appears, robot
        # label ...): clear everything first so that RViz never keeps stale ids
        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        mid = 0

        def add(m):
            nonlocal mid
            m.id = mid
            mid += 1
            arr.markers.append(m)

        for g in snap["gauges"]:
            gdef = self.layout.gauge(g["id"]) or {}
            pos = gdef.get("position")
            if not pos:
                continue
            col = STATUS_RGBA.get(g.get("status"), MUTED_RGBA)
            add(make_marker("twin_gauges", 0, Marker.SPHERE, stamp=stamp, pose=pos, scale=(0.22, 0.22, 0.22),
                            color=col))
            val = g.get("value")
            txt = f"{g['id']} " + (f"{val:.2f} {g.get('unit', '')}" if val is not None else "unread")
            if g.get("error_pct_fs") is not None:
                txt += f" ({g['error_pct_fs']:.1f}%FS)"
            add(make_marker("twin_gauge_labels", 0, Marker.TEXT_VIEW_FACING, stamp=stamp,
                            pose=(pos[0], pos[1], pos[2] + 0.35), scale=(0.0, 0.0, 0.18), color=INK_RGBA, text=txt))
        for h in snap["thermal"]:
            hdef = next((x for x in self.layout.hotspots if x["id"] == h["id"]), None)
            pos = _xyz(h.get("position"), (hdef or {}).get("position"))
            if pos is None:
                continue
            col = STATUS_RGBA.get(h.get("status"), MUTED_RGBA)
            add(make_marker("twin_hotspots", 0, Marker.CUBE, stamp=stamp, pose=pos, scale=(0.2, 0.2, 0.2), color=col))
            t = h.get("max_temp_c")
            add(make_marker("twin_hotspot_labels", 0, Marker.TEXT_VIEW_FACING, stamp=stamp,
                            pose=(pos[0], pos[1], pos[2] + 0.35), scale=(0.0, 0.0, 0.16), color=INK_RGBA,
                            text=f"{h['id']} " + (f"{t:.1f} C" if t is not None else "unseen")))
        leak = self.state.static.get("gas_leak") or {}
        if leak.get("x") is not None:
            lz = (self.layout.gas_leak.get("source") or [0, 0, 0.4])[2]
            add(make_marker("twin_leak", 0, Marker.SPHERE, stamp=stamp, pose=(leak["x"], leak["y"], lz),
                            scale=(0.35, 0.35, 0.35), color=(0.816, 0.231, 0.231, 0.5)))
            add(make_marker("twin_leak", 0, Marker.TEXT_VIEW_FACING, stamp=stamp, pose=(leak["x"], leak["y"], lz + 0.5),
                            scale=(0.0, 0.0, 0.18), color=INK_RGBA, text="LEAK (truth)"))
            est = snap["gas"].get("estimate")
            if est:
                add(make_marker("twin_leak_estimate", 0, Marker.SPHERE, stamp=stamp, pose=(est[0], est[1], lz),
                                scale=(0.3, 0.3, 0.3), color=AQUA_RGBA))
                line = make_marker("twin_leak_estimate", 0, Marker.LINE_LIST, stamp=stamp, scale=(0.03, 0.0, 0.0),
                                   color=AQUA_RGBA)
                line.points = [Point(x=float(est[0]), y=float(est[1]), z=float(lz)),
                               Point(x=float(leak["x"]), y=float(leak["y"]), z=float(lz))]
                add(line)
                err = snap["gas"].get("estimate_error_m")
                add(make_marker("twin_leak_estimate", 0, Marker.TEXT_VIEW_FACING, stamp=stamp,
                                pose=(est[0], est[1], lz + 0.8), scale=(0.0, 0.0, 0.18), color=INK_RGBA,
                                text="estimate" + (f" (error {err:.2f} m)" if err is not None else "")))
        dock = self.state.static.get("dock") or {}
        if dock:
            dz = (self.layout.get("charging_station", {}).get("pad") or [0, 0, 0])[2]
            add(make_marker("twin_dock", 0, Marker.CUBE, stamp=stamp, pose=(dock["x"], dock["y"], dz + 0.01),
                            scale=(dock["size"][0], dock["size"][1], 0.02), color=(0.235, 0.529, 0.898, 0.6)))
        for ap in self.state.static.get("aps") or []:
            apos = next((a["position"] for a in self.layout.comm.get("access_points", []) if a["id"] == ap["id"]),
                        [ap["x"], ap["y"], 2.0])
            add(make_marker("twin_aps", 0, Marker.CYLINDER, stamp=stamp, pose=apos, scale=(0.15, 0.15, 0.3),
                            color=MUTED_RGBA))
            add(make_marker("twin_aps", 0, Marker.TEXT_VIEW_FACING, stamp=stamp,
                            pose=(apos[0], apos[1], apos[2] + 0.35), scale=(0.0, 0.0, 0.18), color=INK_RGBA,
                            text=ap["id"]))
        robot = snap.get("robot")
        if robot and robot.get("x") is not None:
            ms = snap.get("mission") or {}
            label = f"{ms.get('state', '?')}"
            if ms.get("target"):
                label += f" -> {ms['target']}"
            add(make_marker("twin_robot", 0, Marker.TEXT_VIEW_FACING, stamp=stamp,
                            pose=(robot["x"], robot["y"], (robot.get("z") or 0.0) + 0.7), scale=(0.0, 0.0, 0.22),
                            color=INK_RGBA, text=label))
        return arr

    def on_shutdown(self):
        self.server.stop()


def main(args=None):
    run_node(MonitorDashboardNode, args)


if __name__ == "__main__":
    main()
