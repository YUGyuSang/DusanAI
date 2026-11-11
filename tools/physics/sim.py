"""
gz-like MuJoCo plant + harness around the REAL ROS-free controller.

Plant (mirrors simulation/urdf/spot.urdf + simulation/worlds/plant.sdf)
-----------------------------------------------------------------------
* gz step 1 ms (plant.sdf ``max_step_size``), MuJoCo integrates it in
  ``mjcf.SUBSTEPS`` substeps (see mjcf.py: servo / contact stability);
* :class:`GzServo` - gz-sim JointPositionController configured exactly like
  the plugin blocks of spot.urdf (PID force mode: ``-(p e + d de/dt)`` every
  gz step, clamped to +-effort; the velocity mode is available as a what-if).
  Until the first command the targets are the plugins' ``initial_position``;
* sensors at the Gazebo rates: joint states 200 Hz (JointStatePublisher
  ``update_rate``), IMU gyro 200 Hz (gaussian noise 2e-4 rad/s), odometry
  50 Hz (OdometryPublisher: ground-truth pose, body-frame twist from finite
  differences of the pose over one publish period averaged over a rolling
  window of ``odom_window`` samples - gz default
  ``velocity_rolling_window_size`` = 10, i.e. a 0.2 s mean);
* joint commands reach the servos ``cmd_delay`` after the control tick.

Harness (mirrors the ROS nodes)
-------------------------------
* ``locomotion_controller``: LocomotionController at 100 Hz on sim time, after
  the node's start-up gate (first joint state + odometry + ``startup_delay``);
  RobotState built exactly as ``locomotion_controller_node._robot_state``
  (odom pose / rpy, ``vel_world = R @ v_base``, gyro from the IMU);
* MPC worker: :class:`SimTimeMPCWorker` replaces the controller's thread -
  the solve runs synchronously but its result is released ``latency`` of
  SIMULATED time later (the real worker delivers a result one solve time
  after the request), so the controller sees the same plan age as on the
  target regardless of how fast this PC simulates;
* ``terrain_mapper``: either simulated depth (ray casting the depth camera of
  robot_model.SENSORS, 212 x 120 at 10 Hz) -> ``camera_points_to_world`` ->
  ``ElevationMap.integrate``, published at 5 Hz through ``to_cloud_xyz`` /
  ``ElevationMap.from_cloud_xyz`` + ``TerrainClassifier`` (exactly the node
  pipeline), or the ground-truth height raster (``perception="gt"``).
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import mujoco

from dusan_common import robot_model as rm
from dusan_common.camera import sensor_extrinsic
from dusan_common.geometry import euler_to_rot, quat_to_euler, rot_to_euler

from module1_locomotion import kinematics as kin
from module1_locomotion.controller import LocomotionController, OdomFilter, RobotState
from module1_locomotion.elevation_map import ElevationMap, ElevationMapParams
from module1_locomotion.pointcloud_utils import (DepthIntrinsics, PointFilter, PoseBuffer,
                                                 camera_points_to_world)
from module1_locomotion.terrain_classifier import TerrainClassifier

from . import mjcf
from .terrain import Terrain

JOINTS = rm.JOINT_NAMES
CTRL_DT = 0.01

# node-level parameters of locomotion_controller_node (NODE_PARAMS) that shape
# the RobotState; same defaults (tools/test/test_physics_check.py checks them)
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


# ---------------------------------------------------------------------------
# plant pieces
# ---------------------------------------------------------------------------
class GzServo:
    """gz-sim JointPositionController, one per joint, stepped every gz step.

    Per joint, from the URDF plugin block (:class:`physics.mjcf.ServoSpec`):

    * force mode (``use_velocity_commands`` false, the URDF contract): the
      gz::math::PID of the plugin on ``e = q - q_target`` with the gz step as
      dt: ``cmd = cmd_offset - p e - i_err - d (e - e_last) / dt`` (the
      derivative acts on the error, so a new target gives a one-step kick,
      ``e_last`` starts at 0 like in gz), ``i_err`` integrated with
      ``i_gain dt e`` and clamped to ``i_min .. i_max``, ``cmd`` clamped to
      ``cmd_min .. cmd_max``; the result is the joint force (the MuJoCo motor
      additionally clamps it to the URDF effort, like DART);
    * velocity mode (``use_velocity_commands`` true): the joint velocity
      command ``clamp((q_target - q) / dt, cmd_min, cmd_max)`` realised by the
      force-limited velocity actuators of :mod:`physics.mjcf`.

    ``modes`` / ``p_gain`` / ``d_gain`` override the URDF (what-if studies)."""

    def __init__(self, specs, dt: float = mjcf.GZ_DT, modes=None, p_gain=None, d_gain=None):
        specs = list(specs)
        n = len(specs)
        self.dt = float(dt)
        self.modes = list(modes) if modes is not None else [s.mode for s in specs]
        self.target = np.array([s.initial_position for s in specs], dtype=float)
        self.cmd_max = np.array([s.cmd_max for s in specs], dtype=float)
        self.cmd_min = np.array([s.cmd_min for s in specs], dtype=float)
        self.p = np.array([s.p_gain for s in specs], dtype=float)
        self.i = np.array([s.i_gain for s in specs], dtype=float)
        self.d = np.array([s.d_gain for s in specs], dtype=float)
        self.i_max = np.array([s.i_max for s in specs], dtype=float)
        self.i_min = np.array([s.i_min for s in specs], dtype=float)
        self.offset = np.array([s.cmd_offset for s in specs], dtype=float)
        if p_gain is not None:
            self.p = np.broadcast_to(np.asarray(p_gain, dtype=float), (n,)).copy()
        if d_gain is not None:
            self.d = np.broadcast_to(np.asarray(d_gain, dtype=float), (n,)).copy()
        self.force = np.array([m == "force" for m in self.modes])
        self._e_last = np.zeros(n)
        self._i_err = np.zeros(n)

    def command(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        e = q - self.target
        out = np.empty_like(e)
        f = self.force
        if np.any(f):
            self._i_err = self._i_err + self.i * self.dt * e
            lim = self.i_max >= self.i_min
            self._i_err = np.where(lim, np.clip(self._i_err, self.i_min, self.i_max), self._i_err)
            de = (e - self._e_last) / self.dt
            cmd = self.offset - self.p * e - self._i_err - self.d * de
            ok = self.cmd_max >= self.cmd_min
            cmd = np.where(ok, np.clip(cmd, self.cmd_min, self.cmd_max), cmd)
            out[f] = cmd[f]
            self._e_last = e.copy()
        v = np.clip(-e / self.dt, self.cmd_min, self.cmd_max)
        out[~f] = v[~f]
        return out


class RollingMean:
    def __init__(self, n: int):
        self.buf = deque(maxlen=max(1, int(n)))

    def push(self, v) -> np.ndarray:
        self.buf.append(np.asarray(v, dtype=float).copy())
        return np.mean(np.stack(self.buf), axis=0)

    def clear(self):
        self.buf.clear()


@dataclass
class SensorConfig:
    joint_rate: float = 200.0
    imu_rate: float = 200.0
    odom_rate: float = 50.0
    odom_window: int = 10            # gz OdometryPublisher velocity_rolling_window_size (default 10)
    gyro_noise: float = 2e-4         # URDF imu angular_velocity stddev
    cmd_delay_steps: int = 1         # physics steps between the control tick and the servo update


class SimTimeMPCWorker:
    """Drop-in replacement of ``controller._MPCWorker`` on simulated time."""

    def __init__(self, mpc, latency_fn: Callable[[float], float]):
        self.mpc = mpc
        self.latency_fn = latency_fn
        self.now = 0.0
        self._pending = None
        self._ready_at = -1.0
        self._result = None
        self.error: Optional[str] = None
        self.solve_times: List[float] = []
        self.solve_cpu: List[float] = []

    def request_reset(self) -> None:
        # the synchronous solve has already finished: resetting now is what the
        # real worker does after its running solve
        self.mpc.reset()

    def submit(self, job) -> bool:
        if self._pending is not None and self.now < self._ready_at:
            return False
        if self._pending is not None:            # finished but not yet polled -> deliver first
            self._result, self._pending = self._pending, None
        t0 = time.perf_counter()
        try:
            res = self.mpc.solve(*job)
            self.error = None
        except Exception as e:                   # keep going like the real worker
            res = None
            self.error = f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t0
        self.solve_times.append(wall)
        if res is not None:
            self.solve_cpu.append(float(getattr(res, "solve_cpu_time", float("nan"))))
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


# ---------------------------------------------------------------------------
# perception
# ---------------------------------------------------------------------------
class Perception:
    """terrain_mapper emulation: depth ray casting or ground-truth raster."""

    def __init__(self, model, data, terrain: Terrain, mode: str = "depth", res: float = 0.04,
                 depth_rate: float = 10.0, publish_rate: float = 5.0, stride: int = 1):
        self.m, self.d = model, data
        self.terrain = terrain
        self.mode = mode
        self.res = res
        self.depth_period = 1.0 / depth_rate
        self.publish_period = 1.0 / publish_rate
        self.t_depth = -1e9
        self.t_pub = -1e9
        self.ctrl_map: Optional[ElevationMap] = None
        self.info: Optional[dict] = None
        self.gt = self._gt_raster()
        self.emap = ElevationMap(params=ElevationMapParams())
        self.classifier = TerrainClassifier()
        self.filter = PointFilter()
        self.poses = PoseBuffer(max_age=2.0, max_extrapolation=0.3)
        intr = DepthIntrinsics.from_sensor("depth_camera")
        a, b = intr.ray_grid()
        s = max(1, int(stride))
        self._a, self._b = np.ascontiguousarray(a[::s, ::s]).ravel(), np.ascontiguousarray(b[::s, ::s]).ravel()
        dirs = np.stack([np.ones_like(self._a), self._a, self._b], axis=1)
        self._norm = np.linalg.norm(dirs, axis=1)
        self._dirs_link = dirs / self._norm[:, None]
        self._T_bc = sensor_extrinsic("depth_camera")
        self._group = np.zeros(6, dtype=np.uint8)
        self._group[mjcf.TERRAIN_GROUP] = 1
        n = len(self._a)
        self._geomid = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n, dtype=np.float64)
        self.timing: Dict[str, List[float]] = {"depth": [], "publish": []}

    # ------------------------------------------------------------ ground truth
    def _gt_raster(self):
        x0, y0, x1, y1 = self.terrain.extent
        xs = np.arange(x0, x1 + 1e-9, self.res)
        ys = np.arange(y0, y1 + 1e-9, self.res)
        H = np.zeros((len(ys), len(xs)))
        geomid = np.zeros(1, dtype=np.int32)
        vec = np.array([0.0, 0.0, -1.0])
        top = 3.0
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                dist = mujoco.mj_ray(self.m, self.d, np.array([x, y, top]), vec, self._group_static(), 1, -1,
                                     geomid)
                H[r, c] = top - dist if dist >= 0 else np.nan
        return {"x0": float(xs[0]), "y0": float(ys[0]), "res": self.res, "H": H}

    def _group_static(self):
        g = np.zeros(6, dtype=np.uint8)
        g[mjcf.TERRAIN_GROUP] = 1
        return g

    def height(self, x: float, y: float) -> float:
        """Ground-truth terrain height (bilinear on the raster; floor 0 outside)."""
        g = self.gt
        fx = (x - g["x0"]) / g["res"]
        fy = (y - g["y0"]) / g["res"]
        H = g["H"]
        c0, r0 = int(math.floor(fx)), int(math.floor(fy))
        if c0 < 0 or r0 < 0 or c0 + 1 >= H.shape[1] or r0 + 1 >= H.shape[0]:
            return 0.0
        u, v = fx - c0, fy - r0
        h = (H[r0, c0] * (1 - u) * (1 - v) + H[r0, c0 + 1] * u * (1 - v) + H[r0 + 1, c0] * (1 - u) * v
             + H[r0 + 1, c0 + 1] * u * v)
        return float(h) if math.isfinite(h) else 0.0

    def gt_map(self, x: float, y: float, size: float = 6.0) -> ElevationMap:
        g = self.gt
        n = int(round(size / g["res"]))
        c0 = int(round((x - g["x0"]) / g["res"])) - n // 2
        r0 = int(round((y - g["y0"]) / g["res"])) - n // 2
        H = np.full((n, n), np.nan)
        rr0, cc0 = max(r0, 0), max(c0, 0)
        rr1, cc1 = min(r0 + n, g["H"].shape[0]), min(c0 + n, g["H"].shape[1])
        if rr1 > rr0 and cc1 > cc0:
            H[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = g["H"][rr0:rr1, cc0:cc1]
        origin = (g["x0"] + c0 * g["res"], g["y0"] + r0 * g["res"])
        m = ElevationMap.from_elevation(H, origin, g["res"])
        m.update_layers()
        return m

    # ------------------------------------------------------------ depth pipeline
    def depth_points(self, pose_true) -> np.ndarray:
        """Ray-cast the depth camera at the TRUE pose -> link-frame points (N, 3)."""
        T_wb = np.eye(4)
        T_wb[:3, :3] = euler_to_rot(*pose_true[3:6])
        T_wb[:3, 3] = pose_true[:3]
        T_wc = T_wb @ self._T_bc
        origin = T_wc[:3, 3].copy()
        vec = (self._dirs_link @ T_wc[:3, :3].T).ravel().copy()
        n = len(self._a)
        mujoco.mj_multiRay(self.m, self.d, origin, vec, self._group, 1, -1, self._geomid, self._dist, None, n,
                           4.0)
        ok = self._dist >= 0
        depth = self._dist[ok] / self._norm[ok]
        return np.stack([depth, self._a[ok] * depth, self._b[ok] * depth], axis=1)

    def update(self, t: float, pose_true, pose_odom_stamped: Optional[Tuple[float, tuple]]):
        """Call every physics step (cheap when nothing is due).  Returns True
        when a new map / terrain info was published."""
        if pose_odom_stamped is not None:
            self.poses.add(*pose_odom_stamped)
        published = False
        if self.mode == "depth" and t - self.t_depth >= self.depth_period - 1e-9:
            self.t_depth = t
            t0 = time.perf_counter()
            pts = self.depth_points(pose_true)
            pose = self.poses.at(t)
            if pose is None:
                lat = self.poses.latest()
                pose = lat[1] if lat is not None else None
            if pose is not None and pts.shape[0] > 0:
                pw, rng, origin = camera_points_to_world(pts, pose, "link", "depth_camera", self.filter)
                self.emap.recenter(pose[0], pose[1])
                self.emap.integrate(pw, origin, t=t, ranges=rng)
            self.timing["depth"].append(time.perf_counter() - t0)
        if t - self.t_pub >= self.publish_period - 1e-9:
            lat = self.poses.latest()
            if lat is None:
                return False
            self.t_pub = t
            pose = lat[1]
            t0 = time.perf_counter()
            if self.mode == "depth":
                self.emap.recenter(pose[0], pose[1])
                self.emap.predict(t)
                self.ctrl_map = ElevationMap.from_cloud_xyz(self.emap.to_cloud_xyz())
                self.info = self.classifier.update(self.emap, pose, t)
            elif self.mode == "gt":
                self.ctrl_map = self.gt_map(pose[0], pose[1])
                self.info = self.classifier.update(self.ctrl_map, pose, t)
            else:
                self.ctrl_map, self.info = None, None
            self.timing["publish"].append(time.perf_counter() - t0)
            published = True
        return published


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
@dataclass
class Push:
    t: float
    force_body: Tuple[float, float, float]      # [N] in the base heading frame (yaw only)
    duration: float


@dataclass
class SimConfig:
    terrain: Terrain
    spawn: Tuple[float, float, float, float] = (0.0, 0.0, 0.56, 0.0)    # x y z yaw (z = floor + 0.56)
    spawn_rpy: Tuple[float, float] = (0.0, 0.0)                           # roll, pitch at spawn
    spawn_q: Optional[Sequence[float]] = None                              # None: straight legs (knee at limit)
    params: Dict = field(default_factory=dict)
    seed: int = 0
    perception: str = "depth"                    # depth | gt | none
    mpc_latency: object = "jitter"               # float [s] | "measured" | "jitter" (10/20/30 ms, seeded)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    payload: Optional[dict] = None
    kv: float = mjcf.SERVO_KV
    servo: str = "urdf"                          # urdf (the plugin blocks) | velocity | force (what-if overrides)
    servo_p: Optional[float] = None              # what-if p_gain override [N m/rad] (None: URDF)
    servo_d: Optional[float] = None              # what-if d_gain override [N m s/rad] (None: URDF)
    substeps: int = mjcf.SUBSTEPS
    startup_delay: float = 0.5
    yaw_jitter: float = 0.0
    xy_jitter: float = 0.0
    log_rate: float = 100.0
    snapshot_rate: float = 10.0


class PhysicsSim:
    """MuJoCo plant + controller harness.  ``run`` advances the simulation."""

    def __init__(self, cfg: SimConfig, desc: Optional[mjcf.RobotDescription] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.desc = desc or mjcf.parse_urdf()
        sx, sy, sz, syaw = cfg.spawn
        syaw += self.rng.uniform(-1, 1) * cfg.yaw_jitter
        sx += self.rng.uniform(-1, 1) * cfg.xy_jitter
        sy += self.rng.uniform(-1, 1) * cfg.xy_jitter
        xml = mjcf.build_mjcf(self.desc, cfg.terrain.mjcf(), kv=cfg.kv, spawn_pos=(sx, sy, sz), dt=mjcf.GZ_DT / max(1, int(cfg.substeps)), servo=cfg.servo,
                              spawn_yaw=syaw, payload=cfg.payload)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        self.substeps = int(round(mjcf.GZ_DT / m.opt.timestep))
        self.dt = mjcf.GZ_DT                      # gz step: sensors, servo command, pushes
        rate = float(cfg.params.get("control_rate", 1.0 / CTRL_DT))
        self.ctrl_every = max(1, int(round(1.0 / (rate * self.dt))))   # node timer (control_rate)
        # joint / actuator indices in JOINT_NAMES order
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.vadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j + "_servo") for j in JOINTS])
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.foot_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{l}_foot") for l in rm.LEG_NAMES]
        self.foot_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_foot_collision") for l in
                           rm.LEG_NAMES]
        self.leg_bodies = {l: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{l}_{p}") for p in
                               ("hip", "upper_leg", "lower_leg", "foot")] for l in rm.LEG_NAMES}
        self.effort = np.array([self.desc.joints[j].effort for j in JOINTS])
        servos = [self.desc.servos[j] for j in JOINTS]
        modes = [mjcf.servo_mode(sp, cfg.servo) for sp in servos]
        specs = []
        for sp, mode, eff in zip(servos, modes, self.effort):
            if mode != sp.mode:
                # what-if override of the URDF mode: the URDF cmd limits belong to
                # the other mode (rad/s vs N m) -> effort (force) / unlimited (velocity)
                lim = float(eff) if mode == "force" else math.inf
                sp = replace(sp, use_velocity_commands=(mode == "velocity"), cmd_max=lim, cmd_min=-lim)
            specs.append(sp)
        self.servo = GzServo(specs, self.dt, modes, cfg.servo_p, cfg.servo_d)
        # initial joint configuration
        roll0, pitch0 = cfg.spawn_rpy
        q0 = np.array([s.initial_position for s in servos]) if cfg.spawn_q is None else np.asarray(cfg.spawn_q)
        q0 = kin.clamp_joints(q0)
        d.qpos[self.qadr] = q0
        if roll0 or pitch0:
            from .mjcf import rpy_to_quat
            d.qpos[3:7] = rpy_to_quat(roll0, pitch0, syaw)
        self.servo.target = q0.copy()
        mujoco.mj_forward(m, d)
        # sensors
        self.sc = cfg.sensors
        self._joint_msg = None
        self._imu_msg = None
        self._odom_msg = None
        self._odom_prev = None
        self._odom_lin = RollingMean(self.sc.odom_window)
        self._odom_ang = RollingMean(self.sc.odom_window)
        self._first_data_t = None
        self._pending_cmd: deque = deque()
        self.node = {k: cfg.params.get(k, v) for k, v in NODE_STATE_DEFAULTS.items()}
        self.odom_filter = OdomFilter(str(self.node["odom_velocity_source"]), float(self.node["odom_velocity_tau"]),
                                      bool(self.node["odom_extrapolate"]), float(self.node["odom_max_extrapolation"]))
        # controller
        self.ctrl = LocomotionController(dict(cfg.params), reset_fn=self._reset_fn, threaded_mpc=False,
                                         logger=self._log)
        self.worker = SimTimeMPCWorker(self.ctrl.mpc, self._latency)
        self.ctrl.worker = self.worker
        self.messages: List[Tuple[float, str]] = []
        self.events: List[dict] = []
        self.perception = Perception(m, d, cfg.terrain, cfg.perception) if cfg.perception != "none" else None
        self._gt = self.perception if self.perception is not None else Perception(m, d, cfg.terrain, "none")
        self.cmd_fn: Callable[[float], Tuple[float, float, float]] = lambda t: (0.0, 0.0, 0.0)
        self.pushes: List[Push] = []
        self.commands: List[Tuple[float, str]] = []
        self._cmd_done = set()
        self._reset_request = None
        self.step_count = 0
        self.log: Dict[str, list] = {k: [] for k in (
            "t", "pos", "rpy", "vel_body", "omega", "mode", "gait", "q", "q_target", "tau", "contacts",
            "body_h", "ground_h", "cmd", "cmd_f", "foot_pos", "foot_slip", "mpc_ms", "rec_state",
            "terrain_class", "rec_gait", "fall_metric")}
        self.snapshots: List[dict] = []
        self._jacp = np.zeros((3, self.model.nv))
        self._log_every = max(1, int(round(1.0 / (cfg.log_rate * self.dt))))
        self._snap_every = max(1, int(round(1.0 / (cfg.snapshot_rate * self.dt))))
        self.started = False
        self.wall = {"physics": 0.0, "controller": 0.0, "perception": 0.0}

    # ------------------------------------------------------------ helpers
    def _log(self, msg: str) -> None:
        self.messages.append((float(self.data.time), msg))

    def _latency(self, wall: float) -> float:
        lat = self.cfg.mpc_latency
        if lat == "measured":
            return wall
        if lat == "jitter":
            return float(self.rng.choice([0.01, 0.02, 0.03], p=[0.25, 0.5, 0.25]))
        return float(lat)

    def _reset_fn(self, x, y, z, yaw):
        """Assisted reset (gz set_pose) emulation: teleport after 0.5 s."""
        self._reset_request = (float(self.data.time) + 0.5, (x, y, z, yaw))
        return True

    @property
    def t(self) -> float:
        return float(self.data.time)

    def true_pose(self):
        d = self.data
        w, x, y, z = d.qpos[3:7]
        rpy = quat_to_euler(x, y, z, w)
        return np.array([d.qpos[0], d.qpos[1], d.qpos[2], *rpy])

    def true_rot(self) -> np.ndarray:
        return self.data.xmat[self.base].reshape(3, 3).copy()

    def terrain_height(self, x: float, y: float) -> float:
        return self._gt.height(x, y)

    def joint_q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    # ------------------------------------------------------------ sensors
    def _update_sensors(self, t: float, k: int) -> Optional[Tuple[float, tuple]]:
        d = self.data
        sc = self.sc
        rate_steps = lambda r: max(1, int(round(1.0 / (r * self.dt))))
        if k % rate_steps(sc.joint_rate) == 0:
            self._joint_msg = (t, d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy())
        if k % rate_steps(sc.imu_rate) == 0:
            gyro = d.qvel[3:6].copy() + self.rng.normal(0.0, sc.gyro_noise, 3)
            self._imu_msg = (t, self.true_pose()[3:6].copy(), gyro)     # gz IMU: orientation (world) + gyro
        odom_stamped = None
        if k % rate_steps(sc.odom_rate) == 0:
            pose = self.true_pose()
            R = self.true_rot()
            if self._odom_prev is not None:
                tp, pp, rp = self._odom_prev
                dtp = max(t - tp, 1e-6)
                v_world = (pose[:3] - pp[:3]) / dtp
                v_body = R.T @ v_world
                dr = np.array([(pose[3 + i] - pp[3 + i] + math.pi) % (2 * math.pi) - math.pi for i in range(3)])
                v_lin = self._odom_lin.push(v_body)
                v_ang = self._odom_ang.push(dr / dtp)
            else:
                v_lin = np.zeros(3)
                v_ang = np.zeros(3)
            if sc.odom_window <= 0:              # ideal odometry (validation option)
                v_lin = R.T @ d.qvel[0:3]
                v_ang = d.qvel[3:6].copy()
            self._odom_prev = (t, pose.copy(), R)
            self._odom_msg = (t, pose[:3].copy(), pose[3:6].copy(), v_lin.copy(), v_ang.copy())
            self.odom_filter.update(t, pose[:3], pose[3:6], v_lin)
            odom_stamped = (t, tuple(float(v) for v in pose))
        return odom_stamped

    def robot_state(self) -> Optional[RobotState]:
        """Exactly locomotion_controller_node._robot_state (OdomFilter, IMU gyro / orientation)."""
        filt = self.odom_filter.state(self.t)
        if self._odom_msg is None or self._joint_msg is None or filt is None:
            return None
        w_base = self._odom_msg[4]
        pos, rpy, vel_world = filt
        rpy = np.array(rpy, dtype=float)
        gyro = w_base
        if self._imu_msg is not None and (self.t - self._imu_msg[0]) < float(self.node["sensor_timeout"]):
            if bool(self.node["use_imu_gyro"]):
                gyro = self._imu_msg[2]
            if bool(self.node["use_imu_orientation"]):
                rpy[0], rpy[1] = self._imu_msg[1][0], self._imu_msg[1][1]
        st = RobotState()
        st.pos = np.array(pos, dtype=float)
        st.rpy = rpy
        st.vel_world = np.asarray(vel_world, dtype=float)
        st.omega_body = np.asarray(gyro, dtype=float)
        st.q = self._joint_msg[1].copy()
        st.dq = self._joint_msg[2].copy()
        return st

    # ------------------------------------------------------------ main loop
    def run(self, duration: float, cmd_fn: Optional[Callable[[float], Tuple[float, float, float]]] = None,
            stop_fn: Optional[Callable[["PhysicsSim"], bool]] = None) -> None:
        if cmd_fn is not None:
            self.cmd_fn = cmd_fn
        m, d = self.model, self.data
        n = int(round(duration / self.dt))
        for _ in range(n):
            k = self.step_count
            t = k * self.dt
            tw = time.perf_counter()
            odom = self._update_sensors(t, k)
            if self.perception is not None:
                tp = time.perf_counter()
                self.perception.update(t, self.true_pose(), odom)
                self.wall["perception"] += time.perf_counter() - tp
            if k % self.ctrl_every == 0:
                tc = time.perf_counter()
                self._control_tick(t)
                self.wall["controller"] += time.perf_counter() - tc
            # commands reaching the servos
            while self._pending_cmd and self._pending_cmd[0][0] <= k:
                self.servo.target = self._pending_cmd.popleft()[1]
            q = d.qpos[self.qadr]
            d.ctrl[self.act] = self.servo.command(q)
            # external pushes (base frame yaw-rotated into the world, like the node)
            d.xfrc_applied[self.base, :] = 0.0
            for p in self.pushes:
                if p.t <= t < p.t + p.duration:
                    yaw = self.true_pose()[5]
                    c, s = math.cos(yaw), math.sin(yaw)
                    fx, fy, fz = p.force_body
                    d.xfrc_applied[self.base, :3] += (c * fx - s * fy, s * fx + c * fy, fz)
            self._assisted_reset_tick(t)
            tph = time.perf_counter()
            mujoco.mj_step(m, d, self.substeps)
            self.wall["physics"] += time.perf_counter() - tph
            self.step_count += 1
            if k % self._log_every == 0:
                self._record(t)
            if k % self._snap_every == 0:
                self._snapshot(t)
            if not np.all(np.isfinite(d.qpos)):
                self._log("simulation diverged")
                break
            if stop_fn is not None and k % self.ctrl_every == 0 and stop_fn(self):
                break

    def _control_tick(self, t: float) -> None:
        st = self.robot_state()
        if st is None:
            return
        if self._first_data_t is None:
            self._first_data_t = t
        if t - self._first_data_t < float(self.node["startup_delay"]) - 1e-9:
            return
        for (tc, text) in self.commands:
            if tc <= t + 1e-9 and (tc, text) not in self._cmd_done:
                self._cmd_done.add((tc, text))
                self.ctrl.handle_command(text, t, st)
        if self.perception is not None and self.perception.info is not None:
            self.ctrl.set_terrain_info(self.perception.info, t)
        emap = self.perception.ctrl_map if self.perception is not None else None
        self.worker.now = t
        if not self.ctrl.started:
            self.ctrl.start(t, st)
            self.started = True
        q, dbg = self.ctrl.step(t, st, self.cmd_fn(t), None, emap)
        for ev in self.ctrl.pop_events():
            self.events.append(ev)
        self._pending_cmd.append((self.step_count + self.sc.cmd_delay_steps, np.asarray(q, dtype=float).copy()))

    def _assisted_reset_tick(self, t: float) -> None:
        if self._reset_request is None or t < self._reset_request[0]:
            return
        _, (x, y, z, yaw) = self._reset_request
        self._reset_request = None
        d = self.data
        from .mjcf import rpy_to_quat
        d.qpos[0:3] = (x, y, z)
        d.qpos[3:7] = rpy_to_quat(0.0, 0.0, yaw)
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)
        self.events.append({"event": "sim_teleport", "t": t})
        st = self.robot_state()
        if st is not None:
            self.ctrl.on_pose_reset_done(t, st, True)

    # ------------------------------------------------------------ logging
    def foot_contacts(self) -> np.ndarray:
        d = self.data
        c = np.zeros(4, dtype=bool)
        for i in range(d.ncon):
            con = d.contact[i]
            for leg, g in enumerate(self.foot_geoms):
                if con.geom1 == g or con.geom2 == g:
                    c[leg] = True
        return c

    def _record(self, t: float) -> None:
        d = self.data
        pose = self.true_pose()
        R = self.true_rot()
        L = self.log
        L["t"].append(t)
        L["pos"].append(pose[:3])
        L["rpy"].append(pose[3:6])
        L["vel_body"].append(R.T @ d.qvel[0:3])
        L["omega"].append(d.qvel[3:6].copy())
        L["mode"].append(self.ctrl.mode if self.ctrl.started else "WAIT")
        L["gait"].append(self.ctrl.gait.name)
        L["q"].append(d.qpos[self.qadr].copy())
        L["q_target"].append(self.servo.target.copy())
        L["tau"].append(d.actuator_force[self.act].copy())
        contacts = self.foot_contacts()
        L["contacts"].append(contacts)
        gh = self.terrain_height(pose[0], pose[1])
        L["ground_h"].append(gh)
        L["body_h"].append(pose[2] - gh)
        L["cmd"].append(np.asarray(self.cmd_fn(t), dtype=float))
        L["cmd_f"].append(self.ctrl.cmd_f.copy())
        feet = np.array([d.xpos[b] for b in self.foot_bodies])
        L["foot_pos"].append(feet)
        # slip: horizontal speed of the contact point (bottom of the foot sphere,
        # J(point) qvel) of feet in contact - excludes the rolling of the sphere
        slip = np.zeros(4)
        for leg, b in enumerate(self.foot_bodies):
            if contacts[leg]:
                pt = feet[leg] - np.array([0.0, 0.0, rm.FOOT_RADIUS])
                mujoco.mj_jac(self.model, d, self._jacp, None, pt, b)
                v = self._jacp @ d.qvel
                slip[leg] = float(np.hypot(v[0], v[1]))
        L["foot_slip"].append(slip)
        res = self.ctrl.mpc_result
        L["mpc_ms"].append(float(res.solve_time * 1e3) if res is not None else float("nan"))
        L["rec_state"].append(self.ctrl.recovery.state)
        info = self.perception.info if self.perception is not None else None
        L["terrain_class"].append(info.get("class", "") if info else "")
        L["rec_gait"].append(info.get("recommended_gait", "") if info else "")
        fallen = abs(pose[3]) > math.radians(55) or abs(pose[4]) > math.radians(55) or (pose[2] - gh) < 0.12
        L["fall_metric"].append(bool(fallen))

    def _snapshot(self, t: float) -> None:
        d = self.data
        pts = {"base": d.xpos[self.base].copy(), "R": self.true_rot()}
        for l, ids in self.leg_bodies.items():
            pts[l] = np.array([d.xpos[i] for i in ids])
        pts["t"] = t
        pts["mode"] = self.ctrl.mode if self.ctrl.started else "WAIT"
        pts["rec_state"] = self.ctrl.recovery.state
        self.snapshots.append(pts)

    def arrays(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in self.log.items():
            if k in ("mode", "gait", "rec_state", "terrain_class", "rec_gait"):
                out[k] = np.array(v, dtype=object)
            else:
                out[k] = np.asarray(v)
        return out

    def close(self) -> None:
        self.ctrl.shutdown()
