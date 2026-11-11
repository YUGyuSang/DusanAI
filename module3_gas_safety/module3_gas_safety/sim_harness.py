"""
Offline closed-loop harness (ROS free): gas model + sensors + kinematic robot
+ source seeker.  Used by the unit tests and ``tools/eval_gas.py`` to measure
seeking performance without ROS / Gazebo.

The robot is simulated as a holonomic base with velocity / acceleration /
yaw-rate limits; a motion that would bring its centre closer than
``body_radius`` to an occupied nav-map cell is rejected (counted as a
contact), mimicking a legged robot bumping into an obstacle.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .gas_model import GasDispersion2D, GustProcess, LaggedSensor
from .source_seeking import FOUND, SeekerConfig, SourceSeeker


@dataclass
class RobotSim:
    """Holonomic kinematic robot (world pose x, y, yaw)."""
    x: float
    y: float
    yaw: float
    max_speed: float = 0.35
    max_wz: float = 0.8
    max_accel: float = 0.8
    max_alpha: float = 2.0
    body_radius: float = 0.25
    vx_w: float = 0.0
    vy_w: float = 0.0
    wz: float = 0.0
    contacts: int = 0

    def step(self, cmd: Sequence[float], dt: float, grid=None) -> None:
        vx_b, vy_b, wz = (float(v) for v in cmd)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        vxw = c * vx_b - s * vy_b
        vyw = s * vx_b + c * vy_b
        sp = math.hypot(vxw, vyw)
        if sp > self.max_speed:
            vxw *= self.max_speed / sp
            vyw *= self.max_speed / sp
        # acceleration limits
        dvx, dvy = vxw - self.vx_w, vyw - self.vy_w
        dv = math.hypot(dvx, dvy)
        lim = self.max_accel * dt
        if dv > lim:
            dvx *= lim / dv
            dvy *= lim / dv
        self.vx_w += dvx
        self.vy_w += dvy
        wz = max(-self.max_wz, min(self.max_wz, wz))
        dw = max(-self.max_alpha * dt, min(self.max_alpha * dt, wz - self.wz))
        self.wz += dw
        nx = self.x + self.vx_w * dt
        ny = self.y + self.vy_w * dt
        if grid is not None:
            ok_new = grid.clearance(nx, ny) >= self.body_radius
            ok_old = grid.clearance(self.x, self.y) >= self.body_radius
            if not ok_new and (ok_old or grid.clearance(nx, ny) < grid.clearance(self.x, self.y)):
                self.contacts += 1
                self.vx_w = self.vy_w = 0.0
                nx, ny = self.x, self.y
        self.x, self.y = nx, ny
        self.yaw = math.atan2(math.sin(self.yaw + self.wz * dt), math.cos(self.yaw + self.wz * dt))


@dataclass
class SensorSuite:
    """Nose sensor + 4-sensor array + anemometer with the node's noise models."""
    nose_offset: Tuple[float, float] = (0.42, 0.0)
    array_offsets: Tuple[Tuple[float, float], ...] = (
        (0.30, 0.15), (0.30, -0.15), (-0.30, 0.15), (-0.30, -0.15))
    tau: float = 1.0
    noise_abs: float = 0.3
    noise_rel: float = 0.03
    wind_noise: float = 0.03
    wind_tau: float = 0.5
    seed: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.nose = LaggedSensor(1, self.tau, self.noise_abs, self.noise_rel, rng=self.rng)
        self.array = LaggedSensor(len(self.array_offsets), self.tau, self.noise_abs, self.noise_rel, rng=self.rng)
        self.anemo = LaggedSensor(2, self.wind_tau, self.wind_noise, 0.0, rng=self.rng, floor=None)

    def read(self, model: GasDispersion2D, x: float, y: float, yaw: float, dt: float):
        pts = model.body_points(x, y, yaw, [self.nose_offset] + list(self.array_offsets))
        c = model.concentration_at(pts[:, 0], pts[:, 1])
        nose = float(self.nose.update([c[0]], dt)[0])
        arr = self.array.update(c[1:], dt)
        w = model.wind_at(x, y)
        wind = self.anemo.update([w[0], w[1]], dt)
        return nose, arr, wind, float(c[0])


def clone_model(base: GasDispersion2D, seed: int) -> GasDispersion2D:
    """Independent copy of a spun-up model with a new gust realisation."""
    m = copy.deepcopy(base)
    m.rng = np.random.default_rng(seed)
    m.gust = GustProcess(m.cfg, m.rng)
    m.wind.update(m.gust)
    return m


@dataclass
class TrialResult:
    found: bool
    state: str
    estimate: Optional[Tuple[float, float]]
    error_m: Optional[float]
    time_s: float
    path_length: float
    contacts: int
    max_ppm: float
    start: Tuple[float, float, float]
    trajectory: List[Tuple[float, float]]
    transitions: List[Tuple[float, str, str]]
    log: dict
    detours: int = 0


def run_seek_trial(model: GasDispersion2D, grid, start: Tuple[float, float, float],
                   seeker_cfg: Optional[SeekerConfig] = None, sensors: Optional[SensorSuite] = None,
                   true_source: Optional[Tuple[float, float]] = None, dt: float = 0.1,
                   t_max: Optional[float] = None, warmup: float = 0.0, keep_log: bool = False,
                   robot_kwargs: Optional[dict] = None) -> TrialResult:
    """Closed-loop seek: model and seeker both stepped at ``dt``."""
    cfg = seeker_cfg or SeekerConfig()
    sensors = sensors or SensorSuite()
    seeker = SourceSeeker(cfg, grid)
    robot = RobotSim(start[0], start[1], start[2], max_speed=cfg.max_speed, max_wz=cfg.max_wz,
                     **(robot_kwargs or {}))
    true_source = true_source or model.cfg.source_xy
    t_max = t_max if t_max is not None else cfg.timeout + 1.0
    # gust warm-up before the robot starts (decorrelates trials)
    tw = 0.0
    while tw < warmup:
        model.step(min(1.0, warmup - tw))
        tw += 1.0
    t = 0.0
    log = {"t": [], "c": [], "c_true": [], "state": [], "x": [], "y": []} if keep_log else {}
    # prime sensor filters at the start pose
    sensors.read(model, robot.x, robot.y, robot.yaw, dt)
    seeker.enable(t)
    while t < t_max:
        model.step(dt)
        t += dt
        nose, arr, wind, c_true = sensors.read(model, robot.x, robot.y, robot.yaw, dt)
        cmd = seeker.update(t, robot.x, robot.y, robot.yaw, nose, arr, wind)
        robot.step(cmd, dt, grid)
        if keep_log:
            log["t"].append(t)
            log["c"].append(nose)
            log["c_true"].append(c_true)
            log["state"].append(seeker.state)
            log["x"].append(robot.x)
            log["y"].append(robot.y)
        if not seeker.active:
            break
    est = seeker.estimate()
    err = None if est is None else float(math.hypot(est[0] - true_source[0], est[1] - true_source[1]))
    best = seeker.best_sample()
    return TrialResult(
        found=seeker.state == FOUND, state=seeker.state,
        estimate=None if est is None else (float(est[0]), float(est[1])),
        error_m=err, time_s=t, path_length=seeker.path_length, contacts=robot.contacts,
        max_ppm=0.0 if best is None else best[2], start=tuple(start),
        trajectory=list(seeker.path), transitions=list(seeker.transitions), log=log,
        detours=int(seeker.detours))
