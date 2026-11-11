"""
locomotion_controller node (DESIGN_SPEC §4/§5) — thin ROS 2 wrapper around
:class:`module1_locomotion.controller.LocomotionController`.

Subscribes : /odom (nav_msgs/Odometry, twist in base frame), /imu, /joint_states,
             /cmd_vel, /locomotion/command, /terrain/info, /elevation_map/points
Publishes  : /spot/joint/<joint>/cmd_pos (12 x std_msgs/Float64, 100 Hz),
             /locomotion/state (JSON, incl. the slip state: slip, slip_mode, mu_est, mpc_mu ...),
             /locomotion/fall_event (JSON),
             /mpc/predicted_trajectory (Path), /mpc/footholds, /mpc/grf (MarkerArray),
             /mpc/predicted_torques (JointState) - the /mpc/* topics only in locomotion_mode mpc

locomotion_mode (parameter, default stable; runtime /locomotion/command
mode:stable | mode:mpc | mode:toggle): stable = CHAMP-style kinematic gait
(stable_gait.py, no MPC), mpc = 5 s MPC (see controller.py).

Threading: MultiThreadedExecutor; the 100 Hz control timer has its own
mutually-exclusive callback group, the elevation-map parsing another one, and
the MPC runs in the controller's worker thread, so a slow MPC solve or map
update never stalls the joint loop.  The node waits for /joint_states and
/odom, then stands up from the measured joint configuration.
"""
import os

# Keep BLAS single-threaded: the MPC matrices are ~300 x 300, where a thread
# pool brings little, and an oversubscribed OpenBLAS pool (Gazebo + rviz on the
# same CPU) is 10-100x slower.  The environment variables only act if they are
# set before numpy is imported; the ``blas_threads`` parameter re-applies the
# limit at run time through qp.limit_blas_threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from geometry_msgs.msg import Point, PoseStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry, Path  # noqa: E402
from sensor_msgs.msg import Imu, JointState, PointCloud2  # noqa: E402
from std_msgs.msg import Float64, String  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from dusan_common import robot_model as rm  # noqa: E402
from dusan_common.geometry import euler_to_quat, euler_to_rot, quat_to_euler  # noqa: E402
from dusan_common.ros_helpers import json_msg, make_marker, parse_json_msg, sensor_qos  # noqa: E402

from module1_locomotion.controller import (DEFAULT_PARAMS, LocomotionController,  # noqa: E402
                                           OdomFilter, RobotState, organised_cloud_to_xyz)
from module1_locomotion.gz_tools import GzTools  # noqa: E402
from module1_locomotion.qp import blas_threads, limit_blas_threads  # noqa: E402

try:  # newer rclpy raises this on SIGINT
    from rclpy.executors import ExternalShutdownException  # noqa: E402
except ImportError:  # pragma: no cover
    class ExternalShutdownException(Exception):
        pass

LEG_COLORS = ((0.95, 0.30, 0.25, 0.9), (0.25, 0.55, 0.95, 0.9), (0.95, 0.75, 0.20, 0.9), (0.35, 0.85, 0.40, 0.9))

NODE_PARAMS = {
    "world_frame": "odom",
    "state_rate": 20.0,
    "viz_rate": 5.0,
    "cmd_timeout": 0.5,
    "use_imu_orientation": True,
    "use_imu_gyro": True,
    "odom_velocity_source": "differentiate",
    "odom_velocity_tau": 0.08,
    "odom_extrapolate": True,
    "odom_max_extrapolation": 0.05,
    "sensor_timeout": 0.5,
    "startup_delay": 0.5,
    "enable_gz_tools": True,
    "gz_timeout": 3.0,
    "grf_arrow_scale": 0.004,
    "blas_threads": 1,
    "odom_topic": "/odom",
    "imu_topic": "/imu",
    "joint_states_topic": "/joint_states",
    "cmd_vel_topic": "/cmd_vel",
    "command_topic": "/locomotion/command",
    "terrain_info_topic": "/terrain/info",
    "elevation_topic": "/elevation_map/points",
}


class LocomotionControllerNode(Node):
    def __init__(self):
        super().__init__("locomotion_controller")
        self.cfg = {}
        for name, default in list(DEFAULT_PARAMS.items()) + list(NODE_PARAMS.items()):
            self.cfg[name] = self._declare(name, default)
        c = self.cfg
        if int(c["blas_threads"]) > 0:
            limit_blas_threads(int(c["blas_threads"]))

        self._lock = threading.Lock()          # sensor data
        self._ctrl_lock = threading.Lock()     # controller (step / commands)
        self._reset_results = []               # gz set_pose results (bool), handed to the control loop
        # push durations are measured on the node clock (simulation time with use_sim_time)
        self.gz = GzTools(timeout=float(c["gz_timeout"]), logger=lambda m: self.get_logger().info(m),
                          clock=self._now) if bool(c["enable_gz_tools"]) else None
        self.ctrl = LocomotionController(
            {k: c[k] for k in DEFAULT_PARAMS}, reset_fn=self._gz_reset,
            logger=lambda m: self.get_logger().info(m))

        # latest measurements
        self._odom = None          # (t, pos, rpy, v_base, w_base)
        self._imu = None           # (t, rpy, gyro)
        self._joints = None        # (t, q, dq)
        self._joint_index = None
        self._cmd = (0.0, 0.0, 0.0)
        self._t_cmd = -1e9
        self._emap = None
        # 50 Hz pose + 0.2 s rolling-mean twist of the gz odometry -> fresh pose / velocity (see OdomFilter)
        self._odom_filter = OdomFilter(str(c["odom_velocity_source"]), float(c["odom_velocity_tau"]),
                                       bool(c["odom_extrapolate"]), float(c["odom_max_extrapolation"]))
        self._first_data_t = None
        self._wait_t0 = None
        self._t_last_control = None
        self._step_count = 0
        self._errors = 0
        self._warned = set()

        grp_ctrl = MutuallyExclusiveCallbackGroup()
        grp_map = MutuallyExclusiveCallbackGroup()
        grp_sub = ReentrantCallbackGroup()
        grp_viz = MutuallyExclusiveCallbackGroup()
        sq = sensor_qos()

        self.create_subscription(Odometry, c["odom_topic"], self._on_odom, sq, callback_group=grp_sub)
        self.create_subscription(Imu, c["imu_topic"], self._on_imu, sq, callback_group=grp_sub)
        self.create_subscription(JointState, c["joint_states_topic"], self._on_joints, sq, callback_group=grp_sub)
        self.create_subscription(Twist, c["cmd_vel_topic"], self._on_cmd_vel, 10, callback_group=grp_sub)
        self.create_subscription(String, c["command_topic"], self._on_command, 10, callback_group=grp_sub)
        self.create_subscription(String, c["terrain_info_topic"], self._on_terrain_info, sq, callback_group=grp_sub)
        self.create_subscription(PointCloud2, c["elevation_topic"], self._on_elevation, sq, callback_group=grp_map)

        self.joint_pubs = [self.create_publisher(Float64, rm.joint_command_topic(j), 10) for j in rm.JOINT_NAMES]
        self.state_pub = self.create_publisher(String, "/locomotion/state", 10)
        self.event_pub = self.create_publisher(String, "/locomotion/fall_event", 10)
        self.path_pub = self.create_publisher(Path, "/mpc/predicted_trajectory", 10)
        self.foot_pub = self.create_publisher(MarkerArray, "/mpc/footholds", 10)
        self.grf_pub = self.create_publisher(MarkerArray, "/mpc/grf", 10)
        self.torque_pub = self.create_publisher(JointState, "/mpc/predicted_torques", 10)

        rate = max(float(c["control_rate"]), 1.0)
        self.create_timer(1.0 / rate, self._control, callback_group=grp_ctrl)
        self.create_timer(1.0 / max(float(c["viz_rate"]), 0.1), self._publish_viz, callback_group=grp_viz)
        self._state_every = max(1, int(round(rate / max(float(c["state_rate"]), 0.1))))
        self.get_logger().info(
            f"locomotion_controller ready: {rate:.0f} Hz loop, MPC {float(c['mpc_rate']):.0f} Hz "
            f"(horizon {self.ctrl.mpc.T:.1f} s, {self.ctrl.mpc.N} steps, BLAS threads {blas_threads()}), "
            f"gz tools {'on' if self.gz is not None and self.gz.available() else 'off'}; waiting for "
            f"{c['joint_states_topic']} and {c['odom_topic']}")

    # ------------------------------------------------------------------ params
    def _declare(self, name, default):
        try:
            self.declare_parameter(name, default)
        except Exception:
            # type mismatch between YAML and default (e.g. int vs double): accept any type
            from rcl_interfaces.msg import ParameterDescriptor
            try:
                self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
            except Exception:
                return default
        value = self.get_parameter(name).value
        if value is None:
            return default
        try:
            if isinstance(default, bool):
                return bool(value)
            if isinstance(default, float):
                return float(value)
            if isinstance(default, int):
                return int(value)
            if isinstance(default, list):
                return [float(v) for v in value]
            return str(value)
        except (TypeError, ValueError):
            return default

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warning(msg)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ callbacks
    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        rpy = quat_to_euler(q.x, q.y, q.z, q.w)
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        if not all(math.isfinite(float(a)) for a in (p.x, p.y, p.z, v.x, v.y, v.z, w.x, w.y, w.z) + tuple(rpy)):
            self._warn_once("odom_nan", "/odom contains non-finite values - ignored")
            return
        now = self._now()
        st = msg.header.stamp
        stamp = float(st.sec) + float(st.nanosec) * 1e-9
        if not (stamp > 0.0) or abs(stamp - now) > 1.0:
            stamp = now                            # unstamped / other time base: time of arrival
        with self._lock:
            self._odom = (now, np.array([p.x, p.y, p.z]), np.array(rpy),
                          np.array([v.x, v.y, v.z]), np.array([w.x, w.y, w.z]))
            self._odom_filter.update(stamp, (p.x, p.y, p.z), rpy, (v.x, v.y, v.z))

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        rpy = quat_to_euler(q.x, q.y, q.z, q.w) if (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z) > 0.5 \
            else (float("nan"),) * 3
        g = msg.angular_velocity
        with self._lock:
            self._imu = (self._now(), np.array(rpy), np.array([g.x, g.y, g.z]))

    def _on_joints(self, msg: JointState):
        names = list(msg.name)
        if self._joint_index is None or len(self._joint_index) != len(rm.JOINT_NAMES) or \
                any(i >= len(names) or names[i] != j for i, j in zip(self._joint_index, rm.JOINT_NAMES)):
            try:
                self._joint_index = [names.index(j) for j in rm.JOINT_NAMES]
            except ValueError:
                self._warn_once("joints", f"/joint_states lacks some of {rm.JOINT_NAMES}")
                return
        idx = self._joint_index
        if len(msg.position) != len(names):
            # malformed message: an exception here would propagate out of
            # executor.spin() and stop the node
            self._warn_once("joints_len", "/joint_states: position and name arrays differ in length - ignored")
            return
        pos = np.array([msg.position[i] for i in idx], dtype=float)
        vel = np.array([msg.velocity[i] for i in idx], dtype=float) if len(msg.velocity) == len(names) \
            else np.zeros(12)
        if not np.all(np.isfinite(pos)):
            self._warn_once("joints_nan", "/joint_states contains non-finite positions - ignored")
            return
        with self._lock:
            self._joints = (self._now(), pos, vel)

    def _on_cmd_vel(self, msg: Twist):
        with self._lock:
            self._cmd = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            self._t_cmd = self._now()

    def _on_terrain_info(self, msg: String):
        info = parse_json_msg(msg, None)
        if isinstance(info, dict):
            with self._ctrl_lock:
                self.ctrl.set_terrain_info(info, self._now())

    def _on_elevation(self, msg: PointCloud2):
        try:
            xyz = organised_cloud_to_xyz(msg)
        except Exception as e:
            self._warn_once("cloud", f"cannot parse {self.cfg['elevation_topic']}: {e}")
            return
        try:
            from module1_locomotion.elevation_map import ElevationMap
        except Exception as e:
            self._warn_once("emap", f"elevation_map module unavailable ({e}); flat-ground fallback")
            return
        try:
            emap = ElevationMap.from_cloud_xyz(xyz)
        except Exception as e:
            self._warn_once("emap2", f"ElevationMap.from_cloud_xyz failed: {e}")
            return
        with self._lock:
            self._emap = emap

    def _on_command(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        # never let a command (or a gz problem) raise out of the callback: an
        # exception would propagate out of executor.spin() and stop the node
        try:
            st = self._robot_state()
            t = self._now()
            with self._ctrl_lock:
                action = self.ctrl.handle_command(text, t, st)
            self.get_logger().info(f"command '{text}' -> mode {self.ctrl.mode}")
            if action is None:
                return
            if action["action"] == "push":
                self._gz_push(action, st)
            elif action["action"] == "reset_pose" and st is not None:
                # the controller is told when the teleport has happened (_on_gz_reset_done
                # -> _control -> ctrl.on_pose_reset_done), not now: the gz CLI needs
                # ~0.3-1 s, a settle window started here could end before the teleport
                self._gz_reset(float(st.pos[0]), float(st.pos[1]),
                               float(st.pos[2]) + float(action["height_offset"]), float(st.rpy[2]))
        except Exception as e:
            self.get_logger().error(f"command '{text}' failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ gz
    def _gz_push(self, action, st):
        if self.gz is None:
            self._warn_once("gz", "gz tools disabled; push ignored")
            return
        fx, fy, fz = action["force"]
        yaw = float(st.rpy[2]) if st is not None else 0.0
        # the command gives the force in the base frame (x forward, y left); gz expects world
        c, s = math.cos(yaw), math.sin(yaw)
        if not self.gz.push(c * fx - s * fy, s * fx + c * fy, fz, float(action["duration"])):
            self.get_logger().warning("push not executed (gz busy or not available)")

    def _gz_reset(self, x, y, z, yaw) -> bool:
        """gz set_pose (reset_pose command and the controller's assisted reset).
        True: request started, its result reaches ctrl.on_pose_reset_done."""
        if self.gz is None:
            self._warn_once("gz", "gz tools disabled; reset_pose ignored")
            return False
        ok = self.gz.reset_pose(x, y, z, yaw, on_done=self._on_gz_reset_done)
        if not ok:
            self.get_logger().warning("reset_pose not executed (gz busy or not available)")
        return ok

    def _on_gz_reset_done(self, ok: bool) -> None:
        """gz worker thread: hand the set_pose result to the control loop."""
        with self._lock:
            self._reset_results.append(bool(ok))
        if not ok:
            self.get_logger().warning("gz set_pose failed (is the UserCommands system loaded?)")

    # ------------------------------------------------------------------ state
    def _robot_state(self):
        now = self._now()
        with self._lock:
            odom, imu, joints = self._odom, self._imu, self._joints
            filt = self._odom_filter.state(now)
        if odom is None or joints is None or filt is None:
            return None
        _t, _pos, _rpy, _v_base, w_base = odom
        pos, rpy, vel_world = filt
        rpy = np.array(rpy, dtype=float)
        gyro = w_base
        if imu is not None and (self._now() - imu[0]) < float(self.cfg["sensor_timeout"]):
            if bool(self.cfg["use_imu_gyro"]) and np.all(np.isfinite(imu[2])):
                gyro = imu[2]
            if bool(self.cfg["use_imu_orientation"]) and np.all(np.isfinite(imu[1])):
                rpy[0], rpy[1] = imu[1][0], imu[1][1]
        st = RobotState()
        st.pos = np.array(pos, dtype=float)
        st.rpy = rpy
        st.vel_world = np.asarray(vel_world, dtype=float)       # world frame (OdomFilter)
        st.omega_body = np.asarray(gyro, dtype=float)
        st.q = joints[1].copy()
        st.dq = joints[2].copy()
        return st

    # ------------------------------------------------------------------ control loop
    def _control(self):
        t = self._now()
        if self._t_last_control is not None and t < self._t_last_control - 0.5:
            # the clock jumped back (Gazebo world reset: sim time restarts near 0).
            # Time stamps taken before the jump lie in the "future": re-arm the
            # start-up gate (else no joint command would be sent until the sim
            # time passed the old first-data time again, i.e. for several
            # seconds) and drop the last /cmd_vel.  The controller itself
            # restarts in step() (clock-jump check there).
            self.get_logger().warning(f"clock jumped back ({self._t_last_control:.2f} s -> {t:.2f} s): "
                                      "simulation reset? restarting")
            self._first_data_t = None
            self._wait_t0 = None
            with self._lock:
                self._t_cmd = -1e9
                self._odom_filter.reset()
        self._t_last_control = t
        st = self._robot_state()
        if st is None:
            # tell the user what is missing instead of standing still silently
            if self._wait_t0 is None:
                self._wait_t0 = t
            elif t - self._wait_t0 > 5.0:
                with self._lock:
                    missing = [name for name, v in ((self.cfg["odom_topic"], self._odom),
                                                    (self.cfg["joint_states_topic"], self._joints)) if v is None]
                self._warn_once("waiting", f"no data on {', '.join(missing)} after 5 s - is the simulation "
                                           f"(bridge, odom_tf) running? (joint names expected: {rm.JOINT_NAMES})")
            return
        if self._first_data_t is None:
            self._first_data_t = t
            self.get_logger().info("joint states and odometry received")
        if t - self._first_data_t < float(self.cfg["startup_delay"]):
            return
        with self._lock:
            odom_age = t - self._odom[0]
            cmd = self._cmd if (t - self._t_cmd) <= float(self.cfg["cmd_timeout"]) else (0.0, 0.0, 0.0)
            emap = self._emap
            reset_results, self._reset_results = self._reset_results, []
        if odom_age > float(self.cfg["sensor_timeout"]):
            self._warn_once("odom_stale", f"odometry is {odom_age:.2f}s old")
        with self._ctrl_lock:
            try:
                if not self.ctrl.started:
                    self.ctrl.start(t, st)
                for ok in reset_results:
                    self.ctrl.on_pose_reset_done(t, st, ok)
                q, dbg = self.ctrl.step(t, st, cmd, None, emap)
            except Exception as e:
                self._errors += 1
                if self._errors == 1 or self._errors % 100 == 0:
                    self.get_logger().error(f"controller step failed ({self._errors}x): {type(e).__name__}: {e}")
                return
            events = self.ctrl.pop_events()
        for j, pub in enumerate(self.joint_pubs):
            m = Float64()
            m.data = float(q[j])
            pub.publish(m)
        for ev in events:
            if ev.get("event") in ("fall", "recovered", "recovery_failed", "recovery_retry", "assisted_reset"):
                self.event_pub.publish(json_msg(ev))
                self.get_logger().warning(f"fall event: {ev}")
            elif str(ev.get("event", "")).startswith("slip"):
                # slip_detected / slip_mode_on / slip_mode_off (state: /locomotion/state slip_*)
                self.get_logger().info(f"slip event: {ev}")
        self._step_count += 1
        if self._step_count % self._state_every == 0:
            self.state_pub.publish(json_msg(dbg))

    # ------------------------------------------------------------------ visualisation
    def _publish_viz(self):
        try:
            self._publish_viz_impl()
        except Exception as e:      # visualisation must never stop the node
            self._errors += 1
            self._warn_once("viz", f"MPC visualisation failed: {type(e).__name__}: {e}")

    def _publish_viz_impl(self):
        res = self.ctrl.mpc_result
        if res is None:
            return
        stamp = self.get_clock().now().to_msg()
        frame = str(self.cfg["world_frame"])
        # horizon step that contains t + dt (now + one control period): the
        # plan is up to 1 / mpc_rate (+ solve time) old when it is published
        t_rel = self._now() - float(res.t) + 1.0 / max(float(self.cfg["control_rate"]), 1.0)
        k = int(np.clip(np.searchsorted(res.times, t_rel, side="right") - 1, 0, len(res.dts) - 1))
        # predicted CoM trajectory (5 s)
        path = Path()
        path.header.frame_id = frame
        path.header.stamp = stamp
        for x in res.x_pred:
            ps = PoseStamped()
            ps.header.frame_id = frame
            ps.header.stamp = stamp
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(x[3]), float(x[4]), float(x[5])
            qx, qy, qz, qw = euler_to_quat(float(x[0]), float(x[1]), float(x[2]))
            ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
            ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
            path.poses.append(ps)
        self.path_pub.publish(path)

        # footholds over the horizon: one sphere list per leg (+ current stance feet)
        ma = MarkerArray()
        for leg in range(4):
            m = make_marker("footholds", leg, Marker.SPHERE_LIST, frame, stamp, scale=(0.04, 0.04, 0.04),
                            color=LEG_COLORS[leg], lifetime=1.0)
            pts = [res.footholds[k, leg]] + [np.array([x, y, z]) for (lg, ttd, x, y, z) in res.foothold_events
                                             if lg == leg and ttd >= t_rel]
            for p in pts:
                m.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2]) - rm.FOOT_RADIUS))
            ma.markers.append(m)
        self.foot_pub.publish(ma)

        # ground reaction forces of the horizon step at t + dt
        ga = MarkerArray()
        f0 = res.forces[k].reshape(4, 3)
        scale = float(self.cfg["grf_arrow_scale"])
        for leg in range(4):
            m = make_marker("grf", leg, Marker.ARROW, frame, stamp, scale=(0.015, 0.03, 0.04),
                            color=LEG_COLORS[leg], lifetime=1.0)
            if res.contacts[k, leg] <= 0.0 or not np.all(np.isfinite(f0[leg])):
                m.action = Marker.DELETE
            else:
                base = res.footholds[k, leg] - np.array([0.0, 0.0, rm.FOOT_RADIUS])
                tip = base + f0[leg] * scale
                m.points.append(Point(x=float(base[0]), y=float(base[1]), z=float(base[2])))
                m.points.append(Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2])))
            ga.markers.append(m)
        self.grf_pub.publish(ga)

        # predicted joint torques at t + dt
        js = JointState()
        js.header.stamp = stamp
        js.name = list(rm.JOINT_NAMES)
        js.position = [float(v) for v in res.joint_angles[k]]
        js.velocity = []
        js.effort = [float(v) for v in res.torques[k]]
        self.torque_pub.publish(js)

    def shutdown(self):
        self.ctrl.shutdown()


def main(args=None):
    # The MPC worker thread and the 100 Hz control timer share the GIL; the
    # Python-level parts of an MPC solve would otherwise delay a control step
    # by up to the default 5 ms switch interval.
    sys.setswitchinterval(0.001)
    rclpy.init(args=args)
    node = LocomotionControllerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
