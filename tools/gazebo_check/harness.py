"""gz-sim (DART) + gz-transport harness around the REAL ROS-free controller.

Runs in ONE process with the gz python bindings (gz.sim8 TestFixture, gz.transport13 =
Gazebo Harmonic, verified; :class:`GzBindings` also tries the Ionic / Jetty module
versions, e.g. ``GZ_CHECK_RELEASE=jetty`` - untested):

* the gz server steps the test world of :mod:`gazebo_check.world` (plant.sdf
  physics: DART, 1 ms) with the robot exactly as spawned on Ubuntu, i.e. its
  own plugins do the work: JointStatePublisher (``/world/plant/model/spot/
  joint_state``), OdometryPublisher (``/model/spot/odometry``, 50 Hz), the IMU
  sensor (``/imu``, 200 Hz) and the 12 JointPositionControllers
  (``/spot/joint/<j>/cmd_pos``, PID force mode);
* the harness subscribes to those gz topics and publishes the joint targets
  on the cmd_pos topics with gz-transport (what ros_gz_bridge does), and runs
  :class:`module1_locomotion.controller.LocomotionController` at
  ``control_rate`` (100 Hz) of SIM time from the TestFixture pre-update
  callback (the post-update callback crashes on Windows);
* the controller's RobotState is built like ``locomotion_controller_node.
  _robot_state`` (OdomFilter on the 50 Hz odometry, roll / pitch and gyro
  from the IMU), after the node's start-up gate (first joint state + odometry
  + ``startup_delay``);
* ``latency`` (sim seconds) delays every joint command between the control
  tick and its publication (ROS bridge + node processing; 0.01 = one control
  period).  The MPC runs synchronously but its result is released
  ``rtf x wall solve time`` (or a fixed latency) of SIM time later, like the
  real worker thread on a machine running at real-time factor ``rtf``;
  ``drop_ticks`` skips control ticks at random (overloaded executor);
* ``map_source``: "" blind (no elevation map, no /terrain/info), "truth" a
  ground-truth height grid of the world (:class:`world.HeightField`), "depth"
  the emulated depth camera + terrain_mapper pipeline of :mod:`gazebo_check.
  mapper` (elevation map from ray-cast depth images through the module's own
  ElevationMap / PointFilter / TerrainClassifier code, /terrain/info fed to the
  controller like the node does);
* ``perception_depth``: the depth camera + mapper run (and ``elevation_cloud`` /
  ``traversability`` hold the latest /elevation_map/points and
  /elevation_map/traversability payloads at publish_rate) even when the
  controller's own map is blind / truth - the mission's obstacle layer reads
  them; ``extra_objects``: novel static boxes added to the world (physics,
  depth camera, truth height) but not to the nav map;
* ``cmd_fn.bind(harness)`` is called once before the first command when the
  operator has a ``bind`` method (mission in the loop reads odometry / mapper);
* ``controller_start``: the locomotion node starts only at this sim time
  (bringup.launch.py starts the modules ``module_delay`` after the sim, the
  robot stands on its JointPositionController ``initial_position`` meanwhile);
* terrain ``plant``: the REAL simulation/worlds/plant.sdf (world.plant_world_sdf).

Ground truth (base pose / velocity from the ECM) is only logged and given to
an operator model (the person watching Gazebo) and to the depth camera
renderer, never to the controller.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dusan_common import robot_model as rm
from dusan_common.geometry import euler_to_rot, quat_to_euler

from module1_locomotion import kinematics as kin
from module1_locomotion.controller import LocomotionController, OdomFilter, RobotState

from . import world as W

JOINTS = rm.JOINT_NAMES
GZ_DT = 0.001

# locomotion_controller_node.NODE_PARAMS entries that shape the RobotState (same defaults)
NODE_STATE_DEFAULTS = {
    "use_imu_orientation": True,
    "use_imu_gyro": True,
    "sensor_timeout": 0.5,
    "startup_delay": 0.5,
    "odom_velocity_source": "differentiate",
    "odom_velocity_tau": 0.08,
    "odom_extrapolate": True,
    "odom_max_extrapolation": 0.05,
}


# gz python binding versions per Gazebo release: (sim, transport, msgs, math, common)
GZ_RELEASES = (("harmonic", 8, 13, 10, 7, 5), ("ionic", 9, 14, 11, 8, 6), ("jetty", 10, 15, 12, 9, 7))


class GzBindings:
    """The gz python modules of the first installed Gazebo release (Harmonic
    is the one verified on the development PC; Ionic / Jetty are tried with
    their own version numbers, untested)."""

    def __init__(self, release: str = ""):
        import importlib
        errors = []
        for name, vs, vt, vm, vmath, vc in GZ_RELEASES:
            if release and release != name:
                continue
            try:
                importlib.import_module(f"gz.math{vmath}")      # Pose3d conversion of the sim bindings
                self.sim = importlib.import_module(f"gz.sim{vs}")
                self.transport = importlib.import_module(f"gz.transport{vt}")
                self.common = importlib.import_module(f"gz.common{vc}")
                self.math = importlib.import_module(f"gz.math{vmath}")
                self.msgs = {k: importlib.import_module(f"gz.msgs{vm}.{k}_pb2")
                             for k in ("double", "imu", "model", "odometry")}
                self.release = name
                return
            except ImportError as e:
                errors.append(f"{name}: {e}")
        raise ImportError("no gz python bindings found (gz.sim8 / gz.sim9 / gz.sim10 with matching transport, "
                          "msgs, math, common): " + "; ".join(errors))


def _stamp(msg) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nsec) * 1e-9


class SimTimeMPCWorker:
    """Drop-in replacement of ``controller._MPCWorker`` on simulated time:
    the solve runs synchronously, its result is released ``latency_fn(wall)``
    sim seconds after the request."""

    def __init__(self, mpc, latency_fn: Callable[[float], float]):
        self.mpc = mpc
        self.latency_fn = latency_fn
        self.now = 0.0
        self._pending = None
        self._ready_at = -1.0
        self._result = None
        self.error: Optional[str] = None
        self.solve_times: List[float] = []

    def request_reset(self) -> None:
        self.mpc.reset()

    def submit(self, job) -> bool:
        if self._pending is not None and self.now < self._ready_at:
            return False
        if self._pending is not None:
            self._result, self._pending = self._pending, None
        t0 = time.perf_counter()
        try:
            res = self.mpc.solve(*job)
            self.error = None
        except Exception as e:  # keep going like the real worker
            res = None
            self.error = f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t0
        self.solve_times.append(wall)
        if res is not None:
            self._pending = res
            self._ready_at = self.now + float(self.latency_fn(wall))
        return True

    def poll(self):
        if self._pending is not None and self.now >= self._ready_at - 1e-9:
            self._result, self._pending = self._pending, None
        r, self._result = self._result, None
        return r

    @property
    def busy(self) -> bool:
        return self._pending is not None and self.now < self._ready_at

    def stop(self) -> None:
        pass


@dataclass
class HarnessConfig:
    terrain: str = "flat"
    spawn: Tuple[float, float, float, float] = (0.0, 0.0, 0.56, 0.0)
    spawn_rp: Tuple[float, float] = (0.0, 0.0)        # spawn roll / pitch [rad] (lying robot); the gz IMU
    #                                                   orientation is relative to the spawn attitude
    params: Dict = field(default_factory=dict)        # controller / node parameter overrides
    latency: float = 0.01                             # [s] command delay tick -> cmd_pos publication
    mpc_latency: object = "measured"                  # "measured" (rtf x wall solve time) | float [s]
    rtf: float = 1.0                                  # real-time factor of the emulated machine
    drop_ticks: float = 0.0                           # probability of a skipped control tick
    seed: int = 0
    servo: Optional[Dict[str, float]] = None          # URDF what-if overrides (world.robot_tree)
    foot_mu: Optional[float] = None
    ground_truth_map: bool = False                    # legacy flag = map_source "truth"
    map_source: str = ""                              # "" blind | "truth" ground-truth grid | "depth" camera + mapper
    map_rate: float = 5.0                             # [Hz] truth map updates (6 x 6 m, 0.04 m cells, around the robot)
    controller_start: float = 0.0                     # [s] sim time at which the locomotion node starts
    extra_objects: Tuple = ()                         # novel static boxes in the world, not in the nav map
    perception_depth: bool = False                    # depth camera + mapper also when the controller map is
    #                                                   blind / truth (the obstacle layer always gets depth)
    lite_log: bool = False                            # skip the per-joint logs (q, dq, targets, torque, body
    #                                                   command): long missions
    out_dir: str = ""
    verbose: bool = False


class GzHarness:
    """One gz server + controller.  ``run(duration)`` advances the simulation."""

    def __init__(self, cfg: HarnessConfig):
        self.gz = gz = GzBindings(os.environ.get("GZ_CHECK_RELEASE", ""))
        TestFixture, Node = gz.sim.TestFixture, gz.transport.Node
        Double, IMU = gz.msgs["double"].Double, gz.msgs["imu"].IMU
        ModelMsg, Odometry = gz.msgs["model"].Model, gz.msgs["odometry"].Odometry

        gz.common.set_verbosity(1 if not cfg.verbose else 3)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        out = cfg.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build")
        tag = f"_{os.getpid()}_{id(self) & 0xffff:x}"
        self.world_path, self.urdf_path = W.build(out, cfg.terrain, cfg.spawn, cfg.servo, cfg.foot_mu, tag=tag,
                                                  extra=cfg.extra_objects, spawn_rp=cfg.spawn_rp)
        import xml.etree.ElementTree as ET
        self.servo = W.servo_params(ET.parse(self.urdf_path).getroot())
        # ground truth terrain (map stand-in, fall metric) and the optional depth camera / mapper
        self.height_fn, self.hf = W.terrain_height_fn(cfg.terrain, cfg.extra_objects)
        self.map_source = cfg.map_source or ("truth" if cfg.ground_truth_map else "")
        if self.map_source not in ("", "truth", "depth"):
            raise ValueError(f"map_source must be '', 'truth' or 'depth', got {self.map_source!r}")
        self.depth_cam = self.mapper = None
        # latest terrain_mapper outputs (what /elevation_map/points and /elevation_map/traversability carry)
        self.elevation_cloud = None
        self.elevation_cloud_t: Optional[float] = None
        self.traversability = None
        self.traversability_t: Optional[float] = None
        if self.map_source == "depth" or cfg.perception_depth:
            from .mapper import DepthCamera, TerrainMapperSim
            self.depth_cam = DepthCamera(W.collision_shapes(ET.parse(self.world_path).getroot()))
            # friction prior of the plant layout only in the plant world (its zones are plant coordinates)
            self.mapper = TerrainMapperSim(layout=W.LAYOUT if cfg.terrain == W.PLANT else None)
        self._t_depth = -1e9
        self._t_depth_pub = -1e9
        self.terrain_info: Optional[Dict] = None
        self.pre_log: List[Tuple[float, float, float, float]] = []    # (t, height above ground, roll, pitch) before the node starts
        self._Double = Double
        # ------------------------------------------------ gz transport (= ros_gz_bridge)
        self.node = Node()
        self._lock = threading.Lock()
        self._js = None          # (stamp, q, dq)
        self._odom = None        # (stamp, pos, rpy, v_base, w_base)
        self._imu = None         # (stamp, rpy, gyro)
        self._stamps = {"js": [], "odom": [], "imu": []}
        self._jidx = None
        self.node.subscribe(ModelMsg, rm.GZ_JOINT_STATE_TOPIC, self._on_js)
        self.node.subscribe(Odometry, rm.GZ_ODOM_TOPIC, self._on_odom)
        self.node.subscribe(IMU, "/imu", self._on_imu)
        self.pubs = [self.node.advertise(rm.joint_command_topic(j), Double) for j in JOINTS]
        # ------------------------------------------------ node emulation
        p = dict(cfg.params)
        self.node_p = {k: p.get(k, v) for k, v in NODE_STATE_DEFAULTS.items()}
        self.odom_filter = OdomFilter(str(self.node_p["odom_velocity_source"]),
                                      float(self.node_p["odom_velocity_tau"]),
                                      bool(self.node_p["odom_extrapolate"]),
                                      float(self.node_p["odom_max_extrapolation"]))
        self.messages: List[Tuple[float, str]] = []
        self.ctrl = LocomotionController(p, threaded_mpc=False, logger=self._log)
        self.worker = SimTimeMPCWorker(self.ctrl.mpc, self._mpc_latency)
        self.ctrl.worker = self.worker
        rate = float(self.ctrl.p["control_rate"])
        self.ctrl_every = max(1, int(round(1.0 / (rate * GZ_DT))))
        self.cmd_fn: Callable[[float], Tuple[float, float, float]] = lambda t: (0.0, 0.0, 0.0)
        self.commands: List[Tuple[float, str]] = []
        # (t0, t1, world force[, application point in the base link frame, default its origin])
        self.wrenches: List[Tuple] = []
        self._cmd_done = set()
        self._queue: deque = deque()             # (due time, q)
        self._first_data_t = None
        self.q_sent = None
        self.error: Optional[str] = None
        self.wait_timeouts = 0
        self.ticks = 0
        self.dropped = 0
        self.log: Dict[str, list] = {k: [] for k in (
            "t", "pos", "rpy", "vel", "omega", "mode", "gait", "q", "dq", "q_target", "tau", "contacts",
            "cmd", "cmd_f", "feet", "mpc_ms", "rec_state", "fall_metric", "body_cmd", "ik_ok", "ground",
            "terrain_class", "gait_phase")}
        # ------------------------------------------------ gz server
        self.fixture = TestFixture(self.world_path)
        self.fixture.on_pre_update(self._pre)
        self.fixture.finalize()
        self.server = self.fixture.server()
        self.t = 0.0
        self._link = None
        self._wrench_pub = None
        self.emap = None
        self._t_map = -1e9

    # ------------------------------------------------------------------ transport callbacks
    def _on_js(self, msg) -> None:
        if self._jidx is None:
            names = [j.name for j in msg.joint]
            try:
                self._jidx = [names.index(j) for j in JOINTS]
            except ValueError:
                return
        js = msg.joint
        q = np.array([js[i].axis1.position for i in self._jidx])
        dq = np.array([js[i].axis1.velocity for i in self._jidx])
        with self._lock:
            self._js = (_stamp(msg), q, dq)

    def _on_odom(self, msg) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        rpy = quat_to_euler(o.x, o.y, o.z, o.w)
        v, w = msg.twist.linear, msg.twist.angular
        st = _stamp(msg)
        with self._lock:
            self._odom = (st, np.array([p.x, p.y, p.z]), np.array(rpy), np.array([v.x, v.y, v.z]),
                          np.array([w.x, w.y, w.z]))
            self.odom_filter.update(st, (p.x, p.y, p.z), rpy, (v.x, v.y, v.z))

    def _on_imu(self, msg) -> None:
        o, g = msg.orientation, msg.angular_velocity
        rpy = quat_to_euler(o.x, o.y, o.z, o.w)
        with self._lock:
            self._imu = (_stamp(msg), np.array(rpy), np.array([g.x, g.y, g.z]))

    def _log(self, msg: str) -> None:
        self.messages.append((self.t, msg))
        if self.cfg.verbose:
            print(f"[{self.t:7.3f}] {msg}")

    def _mpc_latency(self, wall: float) -> float:
        if isinstance(self.cfg.mpc_latency, (int, float)):
            return float(self.cfg.mpc_latency)
        return max(GZ_DT, wall * float(self.cfg.rtf))

    # ------------------------------------------------------------------ data sync
    def _wait_for_data(self, t: float) -> None:
        """Make the lock-step deterministic: in the pre-update of the step that
        starts at t, the plugins' messages of the previous step (stamp t - dt)
        must have been delivered (gz-transport hands them over on its own
        thread).  Waits until the newest stamp of each stream is not older
        than one publish period."""
        want = t - GZ_DT
        t0 = time.perf_counter()
        periods = {"js": GZ_DT, "imu": 0.005, "odom": 0.02}
        while True:
            with self._lock:
                st = {"js": self._js[0] if self._js else None, "imu": self._imu[0] if self._imu else None,
                      "odom": self._odom[0] if self._odom else None}
            ok = all(s is not None and s > want - periods[k] + 1e-6 for k, s in st.items()) \
                if st["js"] is not None else False
            if ok or t < 0.15:
                return
            if time.perf_counter() - t0 > 0.2:
                self.wait_timeouts += 1
                return
            time.sleep(0)

    # ------------------------------------------------------------------ robot state (node emulation)
    def robot_state(self, now: float) -> Optional[RobotState]:
        with self._lock:
            odom, imu, js = self._odom, self._imu, self._js
            filt = self.odom_filter.state(now)
        if odom is None or js is None or filt is None:
            return None
        pos, rpy, vel_world = filt
        rpy = np.array(rpy, dtype=float)
        gyro = odom[4]
        if imu is not None and (now - imu[0]) < float(self.node_p["sensor_timeout"]):
            if bool(self.node_p["use_imu_gyro"]) and np.all(np.isfinite(imu[2])):
                gyro = imu[2]
            if bool(self.node_p["use_imu_orientation"]) and np.all(np.isfinite(imu[1])):
                rpy[0], rpy[1] = imu[1][0], imu[1][1]
        st = RobotState()
        st.pos = np.array(pos, dtype=float)
        st.rpy = rpy
        st.vel_world = np.asarray(vel_world, dtype=float)
        st.omega_body = np.asarray(gyro, dtype=float)
        st.q = js[1].copy()
        st.dq = js[2].copy()
        return st

    # ------------------------------------------------------------------ ground truth
    def _ground_truth(self, ecm):
        Link, Model, World, world_entity = (self.gz.sim.Link, self.gz.sim.Model, self.gz.sim.World,
                                            self.gz.sim.world_entity)
        if self._link is None:
            w = World(world_entity(ecm))
            m = Model(w.model_by_name(ecm, "spot"))
            self._link = Link(m.link_by_name(ecm, "base_link"))
            self._link.enable_velocity_checks(ecm, True)
        p = self._link.world_pose(ecm)
        if p is None:
            return None
        q = p.rot()
        pos = np.array([p.pos().x(), p.pos().y(), p.pos().z()])
        rpy = np.array(quat_to_euler(q.x(), q.y(), q.z(), q.w()))
        v = self._link.world_linear_velocity(ecm)
        w_ = self._link.world_angular_velocity(ecm)
        vel = np.array([v.x(), v.y(), v.z()]) if v is not None else np.zeros(3)
        om = np.array([w_.x(), w_.y(), w_.z()]) if w_ is not None else np.zeros(3)
        return pos, rpy, vel, om

    def ground_at(self, x: float, y: float) -> float:
        """Ground height under (x, y) (median over the base footprint in the plant world)."""
        if self.hf is not None:
            return self.hf.ground(float(x), float(y))
        return float(self.height_fn(float(x), float(y)))

    def _apply_wrench(self, ecm, t: float) -> None:
        V = self.gz.math.Vector3d
        for w in self.wrenches:
            t0, t1, f = w[0], w[1], w[2]
            if t0 <= t < t1 and self._link is not None:
                if len(w) > 3 and w[3] is not None:
                    self._link.add_world_force(ecm, V(*f), V(*w[3]))
                else:
                    self._link.add_world_force(ecm, V(*f))

    # ------------------------------------------------------------------ pre-update (server thread)
    def _pre(self, info, ecm) -> None:
        if self.error is not None:
            return
        try:
            self._pre_impl(info, ecm)
        except Exception as e:  # never raise into the gz server
            import traceback
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    def _publish(self, q) -> None:
        m = self._Double()
        for j, pub in enumerate(self.pubs):
            m.data = float(q[j])
            pub.publish(m)
        self.q_sent = np.asarray(q, dtype=float).copy()

    def _pre_impl(self, info, ecm) -> None:
        if info.paused:
            return
        it = int(info.iterations)
        t = it * GZ_DT
        self.t = t
        self.worker.now = t
        if self.wrenches:
            if self._link is None:
                self._ground_truth(ecm)
            self._apply_wrench(ecm, t)
        # delayed joint commands (bridge latency)
        while self._queue and self._queue[0][0] <= t + 1e-9:
            _due, q = self._queue.popleft()
            self._publish(q)
        if it % self.ctrl_every != 0:
            return
        self._wait_for_data(t)
        if t < float(self.cfg.controller_start) - 1e-9:
            # locomotion node not running yet: the servos hold initial_position
            if it % 100 == 0:
                gt = self._ground_truth(ecm)
                if gt is not None:
                    self.pre_log.append((t, float(gt[0][2] - self.ground_at(gt[0][0], gt[0][1])),
                                         float(gt[1][0]), float(gt[1][1])))
            return
        for (tc, text) in self.commands:
            if tc <= t and (tc, text) not in self._cmd_done:
                self._cmd_done.add((tc, text))
                self.ctrl.handle_command(text, t, self.robot_state(t))
        st = self.robot_state(t)
        if st is None:
            return
        if self._first_data_t is None:
            self._first_data_t = t
        if t - self._first_data_t < float(self.node_p["startup_delay"]):
            return
        if self.cfg.drop_ticks > 0.0 and self.rng.random() < self.cfg.drop_ticks:
            self.dropped += 1
            return
        if not self.ctrl.started:
            self.ctrl.start(t, st)
        gt = None
        if getattr(self.cmd_fn, "needs_pose", False) or self.depth_cam is not None:
            gt = self._ground_truth(ecm)
        if hasattr(self.cmd_fn, "bind") and not getattr(self.cmd_fn, "bound", False):
            self.cmd_fn.bind(self)                   # operators that read the harness (odometry, mapper)
        if getattr(self.cmd_fn, "needs_pose", False):
            cmd = tuple(float(v) for v in self.cmd_fn(t, gt))
        else:
            cmd = tuple(float(v) for v in self.cmd_fn(t))
        if self.map_source == "truth" and t - self._t_map >= 1.0 / max(self.cfg.map_rate, 0.1) - 1e-9:
            self._t_map = t
            self.emap = self._build_map(st.pos)
        if self.depth_cam is not None:
            self._update_depth_map(t, gt, feed_controller=self.map_source == "depth")
        q, dbg = self.ctrl.step(t, st, cmd, None, self.emap)
        self.ticks += 1
        if self.cfg.latency <= 1e-9:
            self._publish(q)
        else:
            self._queue.append((t + float(self.cfg.latency), q.copy()))
        self._record(t, ecm, st, q, dbg, cmd)

    def _update_depth_map(self, t: float, gt, feed_controller: bool = True) -> None:
        """Depth camera frame (10 Hz, rendered at the ground-truth camera pose)
        fused at the odometry pose; map cloud + traversability grid (+ /terrain/info
        to the controller with ``feed_controller``) at publish_rate."""
        with self._lock:
            odom = self._odom
        if odom is None:
            return
        pose = [float(v) for v in odom[1]] + [float(v) for v in odom[2]]
        if gt is not None and t - self._t_depth >= 1.0 / self.depth_cam.rate - 1e-9:
            self._t_depth = t
            self.mapper.integrate_depth(self.depth_cam.render(gt[0], gt[1]), pose, t)
        if t - self._t_depth_pub >= 1.0 / self.mapper.publish_rate - 1e-9 and self.mapper.n_frames > 0:
            from dusan_common.grid_map import OccupancyGridMap
            self._t_depth_pub = t
            cloud, info = self.mapper.publish(pose, t)
            self.elevation_cloud, self.elevation_cloud_t = cloud, t
            ox, oy = self.mapper.emap.occupancy_origin()
            self.traversability = OccupancyGridMap(self.mapper.emap.occupancy_data(), self.mapper.emap.resolution,
                                                   ox, oy, frame_id="odom")
            self.traversability_t = t
            if feed_controller:
                from module1_locomotion.elevation_map import ElevationMap
                self.emap = ElevationMap.from_cloud_xyz(cloud)     # what the locomotion node rebuilds
                self.terrain_info = info
                self.ctrl.set_terrain_info(info, t)

    def _build_map(self, pos):
        """Ground-truth elevation map around ``pos`` (what terrain_mapping_node
        publishes on /elevation_map/points, without sensor noise / holes)."""
        from module1_locomotion.elevation_map import ElevationMap
        res, half = 0.04, 3.0
        x0 = math.floor((float(pos[0]) - half) / res) * res
        y0 = math.floor((float(pos[1]) - half) / res) * res
        n = int(round(2 * half / res))
        xs = x0 + res * np.arange(n)
        ys = y0 + res * np.arange(n)
        X, Y = np.meshgrid(xs, ys)
        Z = self.height_fn(X, Y)
        emap = ElevationMap.from_elevation(Z, (x0, y0), res)
        emap.update_layers()
        return emap

    def _record(self, t, ecm, st, q, dbg, cmd) -> None:
        gt = self._ground_truth(ecm)
        if gt is None:
            return
        pos, rpy, vel, om = gt
        L = self.log
        L["t"].append(t)
        L["pos"].append(pos)
        L["rpy"].append(rpy)
        L["vel"].append(vel)
        L["omega"].append(om)
        L["mode"].append(dbg["mode"])
        L["gait"].append(dbg["gait"])
        L["gait_phase"].append(float(dbg.get("phase", math.nan)))
        if not self.cfg.lite_log:
            L["q"].append(st.q.copy())
            L["dq"].append(st.dq.copy())
            L["q_target"].append(np.asarray(q, dtype=float).copy())
            # servo torque estimate: the PID force of the plugin at the last applied target
            tgt = self.q_sent if self.q_sent is not None else q
            kp = np.array([self.servo[j]["p_gain"] for j in JOINTS])
            kd = np.array([self.servo[j]["d_gain"] for j in JOINTS])
            cmax = np.array([self.servo[j]["cmd_max"] for j in JOINTS])
            tau = np.clip(kp * (tgt - st.q) - kd * st.dq, -cmax, cmax)
            L["tau"].append(tau)
        L["contacts"].append(np.array(dbg["contacts"], dtype=bool))
        L["cmd"].append(np.array(cmd))
        L["cmd_f"].append(np.array(dbg["cmd"]))
        R = euler_to_rot(*rpy)
        L["feet"].append(kin.body_to_world(kin.fk_all(st.q), pos, R))
        L["mpc_ms"].append(dbg["mpc_solve_ms"])
        L["rec_state"].append(dbg["recovery_state"])
        ground = self.ground_at(pos[0], pos[1])
        L["ground"].append(ground)
        L["terrain_class"].append(str(self.terrain_info.get("class", "")) if self.terrain_info else "")
        L["fall_metric"].append(bool(max(abs(rpy[0]), abs(rpy[1])) > math.radians(55.0) or pos[2] - ground < 0.12))
        if not self.cfg.lite_log:
            bc = self.ctrl.body_ik
            L["body_cmd"].append(np.r_[bc[0], bc[1]] if bc is not None else np.full(6, np.nan))
        L["ik_ok"].append(bool(dbg["ik_reachable"]))

    # ------------------------------------------------------------------ run
    def run(self, duration: float, chunk: float = 1.0) -> None:
        end = self.t + duration
        while self.t < end - 1e-9 and self.error is None:
            n = int(round(min(chunk, end - self.t) / GZ_DT))
            self.server.run(True, max(n, 1), False)
        if self.error:
            raise RuntimeError(self.error)

    def arrays(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in self.log.items():
            out[k] = np.array(v) if k not in ("mode", "gait", "rec_state", "terrain_class") \
                else np.array(v, dtype=object)
        return out

    def close(self) -> None:
        try:
            self.ctrl.shutdown()
        except Exception:
            pass
        for p in (self.world_path, self.urdf_path):
            try:
                os.remove(p)
            except OSError:
                pass
