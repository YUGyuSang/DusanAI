"""Whole demo mission stack without ROS, wired like simulation/launch/bringup.launch.py.

REAL code of every node that takes part in the demo mission:

* ``mission_manager``      plant_simulation.mission_logic.MissionLogic
* ``patrol_planner``       module2_inspection.viewpoint_planner.ViewpointPlanner (patrol.yaml)
* ``gas_dispersion_sim``   module3_gas_safety.gas_model GasDispersion2D + LaggedSensor
* ``source_seeker``        module3_gas_safety.source_seeking.SourceSeeker (on the nav map with the
                           ``debris_blocked_kinds`` debris as obstacles, as the node does)
* ``battery_simulator``    module3_gas_safety.battery_model BatteryModel + DockDetector
* ``comm_simulator``       module3_gas_safety.comm_model CommModel + FadingProcess
* ``return_to_home``       module3_gas_safety.rth_logic RthLogic + GeodesicField + SyncPlanner
* ``cmd_vel_mux``          plant_simulation.mux_logic.CmdVelMux
* obstacle layer           dusan_common.obstacle_layer.EvidenceFrame from the terrain_mapper cloud

Stubbed from the ground truth (no camera headless): the gauge reader (visibility
of the target gauge by ViewpointPlanner.evaluate at the true pose, 5 Hz, motion
gate |omega| <= 0.6 rad/s, reading = true value + 1 %FS noise).  The thermal
pipeline does not steer the mission and is not run.

:class:`MissionStack` ticks all of them at 10 Hz (:meth:`MissionStack.tick`);
:class:`MissionStackOperator` is the harness operator (``GzHarness.cmd_fn``)
that feeds it the odometry, the locomotion mode and the elevation cloud and
returns the cmd_vel_mux output to the controller.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from . import world as W

ROOT = W.ROOT

# gauge reader stub (module2 gauge_reader_node gates)
GAUGE_MAX_INCIDENCE_DEG = 70.0
GAUGE_MIN_PIXELS = 24.0
GAUGE_MAX_DIST = 6.0
GAUGE_MAX_OMEGA = 0.6          # [rad/s] motion blur gate
GAUGE_NOISE_FS = 0.01          # reading noise, fraction of full scale


def node_params(rel: str, node: str) -> Dict:
    """``ros__parameters`` of ``node`` in the YAML file ``rel`` (repo relative)."""
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return dict((d.get(node) or {}).get("ros__parameters") or {})


class MissionStack:
    """Every mission node of the demo, ROS free, ticked together at 10 Hz."""

    def __init__(self, seed: int = 0, mission_over: Optional[Dict] = None, rth_over: Optional[Dict] = None,
                 seeker_over: Optional[Dict] = None, patrol_over: Optional[Dict] = None,
                 battery_over: Optional[Dict] = None):
        from dusan_common.grid_map import OccupancyGridMap
        from dusan_common.layout import Layout
        from module2_inspection.viewpoint_planner import ViewpointParams, ViewpointPlanner
        from module3_gas_safety.battery_model import BatteryConfig, BatteryModel, DockDetector
        from module3_gas_safety.comm_model import CommModel, FadingProcess, RssiMap, config_from_layout
        from module3_gas_safety.gas_model import GasDispersion2D, LaggedSensor
        from module3_gas_safety.params import (BATTERY_PARAMS, COMM_PARAMS, GAS_PARAMS, RTH_PARAMS, SEEKER_PARAMS,
                                               battery_config, gas_config, rth_config, seeker_config)
        from module3_gas_safety.rth_logic import GeodesicField, RthLogic, SyncPlanner, terrain_cost_for
        from module3_gas_safety.source_seeking import SourceSeeker
        from plant_simulation.mission_logic import MissionLogic
        from plant_simulation.mux_logic import CmdVelMux

        self.seed = int(seed)
        self.layout = L = Layout.load(os.path.join(ROOT, "simulation", "config", "world_layout.yaml"))
        self.grid = grid = OccupancyGridMap.load(os.path.join(ROOT, "simulation", "maps", "plant_map.yaml"))
        # ---------------- patrol_planner (module2)
        pp = node_params("module2_inspection/config/patrol.yaml", "patrol_planner")
        pp.update(patrol_over or {})
        self.vplanner = ViewpointPlanner(L, grid, ViewpointParams.from_dict(pp))
        plan = self.vplanner.plan(start=tuple(L.dock_pose))
        tags = {g["id"]: g.get("tag", "") for g in L.gauges}
        self.plan_json = plan.to_json(tags)
        self.plan_obj = plan
        self.gauges = {g["id"]: g for g in L.gauges}
        # ---------------- mission_manager
        mp = node_params("simulation/config/mission.yaml", "mission_manager")
        mp.update(mission_over or {})
        self.m = MissionLogic(mp, L)
        self.m.set_map(grid)
        self.m.set_plan(self.plan_json)
        # ---------------- gas_dispersion_sim
        gp = dict(GAS_PARAMS)
        gp.update(node_params("module3_gas_safety/config/gas_dispersion.yaml", "gas_dispersion_sim"))
        self.gas = GasDispersion2D.from_nav_map(grid, gas_config(gp, L))
        self.gas.spin_up(float(gp["spinup_seconds"]), float(gp["spinup_dt"]), gusts=False)
        self.gas.spin_up(float(gp["spinup_gust_seconds"]), float(gp["spinup_dt"]), gusts=True)
        self.gas.t = 0.0
        rng = np.random.default_rng(int(gp["sensor_seed"]) + 100 * self.seed)
        self.nose_off = [tuple(gp["nose_offset"][:2])]
        a = [float(v) for v in gp["array_offsets"]]
        self.arr_off = [(a[i], a[i + 1]) for i in range(0, len(a) - 1, 2)]
        self.nose = LaggedSensor(1, gp["sensor_tau"], gp["sensor_noise_ppm"], gp["sensor_noise_rel"],
                                 gp["sensor_quantum"], rng)
        self.array = LaggedSensor(len(self.arr_off), gp["sensor_tau"], gp["sensor_noise_ppm"],
                                  gp["sensor_noise_rel"], gp["sensor_quantum"], rng)
        self.anemo = LaggedSensor(2, gp["wind_sensor_tau"], gp["wind_sensor_noise"], 0.0, 0.0, rng, floor=None)
        self.c_nose: Optional[float] = None
        self.c_arr: List[float] = []
        self.wind = (0.0, 0.0)
        # ---------------- source_seeker
        sp = dict(SEEKER_PARAMS)
        sp.update(node_params("module3_gas_safety/config/seeker.yaml", "source_seeker"))
        sp.update(seeker_over or {})
        kinds = tuple(k.strip() for k in str(sp.get("debris_blocked_kinds", "")).split(",") if k.strip())
        sgrid = grid
        if kinds:                              # = source_seeker_node._seek_grid
            from dusan_common.terrain_cost import debris_blocked_grid
            sgrid = debris_blocked_grid(grid, L, kinds, keep_passable=float(sp["robot_radius"]))
        self.seeker = SourceSeeker(seeker_config(sp), sgrid)
        from dusan_common.debris_crossing import CrossingParams
        self.seeker.set_pipes(L, CrossingParams.from_params(sp))      # = source_seeker_node
        src = L.gas_leak.get("source")
        self.leak = (float(src[0]), float(src[1]))
        self.seek_enabled = False
        self.seek_was_cmd = False
        self._seek_last_state = None
        # ---------------- battery_simulator
        bp = dict(BATTERY_PARAMS)
        bp.update(node_params("module3_gas_safety/config/safety.yaml", "battery_simulator"))
        bp.update(battery_over or {})
        self.bat = BatteryModel(battery_config(bp))
        tol = bp["dock_tolerance"] if bp["dock_tolerance"] > 0 else float(L["charging_station"].get("dock_tolerance",
                                                                                                     0.35))
        self.dockdet = DockDetector(L.dock_pose[:2], tol, bp["dock_speed_max"], bp["dock_settle_s"])
        self.bp = bp
        self.speed_f = 0.0
        self.bat_pub: Tuple[Optional[float], Optional[bool]] = (None, None)
        # ---------------- comm_simulator
        cp = dict(COMM_PARAMS)
        cp.update(node_params("module3_gas_safety/config/safety.yaml", "comm_simulator"))
        ccfg = config_from_layout(L)
        ccfg.antenna_height = float(cp["antenna_height"])
        ccfg.map_resolution = float(cp["map_resolution"])
        ccfg.rf_min_obstacle_area_m2 = float(cp["rf_min_obstacle_area_m2"])
        self.comm = CommModel(grid, ccfg)
        cmap = self.comm.compute_map(ccfg.map_resolution)
        self.rssi_map = RssiMap.from_grid_values(cmap.to_grid_values(), cmap.resolution, cmap.origin_x,
                                                 cmap.origin_y)
        self.fading = FadingProcess(cp["fading_sigma_db"], cp["fading_tau_s"], int(cp["seed"]) + self.seed)
        self.rssi: Optional[float] = None
        # ---------------- return_to_home
        rp = dict(RTH_PARAMS)
        rp.update(node_params("module3_gas_safety/config/safety.yaml", "return_to_home"))
        rp.update(rth_over or {})
        rcfg = rth_config(rp, L)
        rbat = BatteryModel(BatteryConfig(capacity_wh=rp["capacity_wh"], p_compute_w=rp["p_compute_w"],
                                          p_sensors_w=rp["p_sensors_w"], p_stand_w=rp["p_stand_w"],
                                          p_sit_w=rp["p_sit_w"], p_per_mps_w=rp["p_per_mps_w"],
                                          p_per_radps_w=rp["p_per_radps_w"]))
        self.geo = GeodesicField(grid, L.dock_pose[:2], rcfg.robot_radius, rcfg.geodesic_downsample)
        planner = SyncPlanner(grid, rcfg.robot_radius, rcfg.robot_radius + rcfg.smooth_margin,
                              terrain_cost_for(rcfg, L))
        self.rth = RthLogic(rcfg, grid, L.dock_pose, rbat, rssi_lookup=self.rssi_map.at,
                            home_distance=self.geo.distance, planner=planner,
                            obstacle_source=str(rp["obstacle_source"]))
        self.rth_cmds: List[str] = []
        self.rth_plan_in = None
        self._rth_own_sig = None
        self.rth_status = None
        # ---------------- cmd_vel_mux
        self.mux = CmdVelMux.from_params(node_params("simulation/config/mux.yaml", "cmd_vel_mux"))
        self.mux_src = "none"
        # ---------------- bookkeeping
        self.t_last: Optional[float] = None
        self.t_status = self.t_active = self.t_bat = self.t_comm = self.t_seek_status = self.t_gauge = -1e9
        self.target = ""
        self.events: List[Tuple[float, str, str]] = []
        self.loco_cmds: List[Tuple[float, str]] = []
        self.max_ppm = 0.0
        self.gauge_log: Dict[str, Dict] = {}
        self.seek_log: List[Tuple] = []
        self.state_log: List[Tuple] = []
        self._last_state = None

    # ------------------------------------------------------------------ io
    def ev(self, t: float, lvl: str, text: str) -> None:
        self.events.append((round(float(t), 2), lvl, text))

    def command(self, cmd: str, t: float) -> None:
        """/mission/command (the teleop 't' key = 'start'): mission_manager and return_to_home get it."""
        self.m.command(cmd, t)
        self.rth_cmds.append(cmd)

    def _drain_mission(self, t: float, out) -> None:
        for c in out.locomotion:
            self.loco_cmds.append((t, c))
        if out.cmd_vel is not None:
            self.mux.update("patrol", out.cmd_vel, t)
        if out.path is not None:
            self.rth_plan_in = [tuple(p[:2]) for p in out.path]
        if out.target is not None:
            self.target = out.target
        if out.seek_enable is not None:
            self._on_seek_enable(bool(out.seek_enable), t)
        for c in out.mission:
            self.rth_cmds.append(c)
            self.m.command(c, t)              # the mission manager hears its own /mission/command
        for lvl, text in out.events:
            self.ev(t, lvl, text)

    def _on_seek_enable(self, flag: bool, t: float) -> None:
        if flag and not self.seek_enabled:
            self.seek_enabled = True
            self.seeker.enable(t)
            self.ev(t, "seeker", "enabled")
            self._seek_status(t)
        elif not flag and self.seek_enabled:
            self.seek_enabled = False
            self.seeker.disable(t)
            if self.seek_was_cmd:
                self.mux.update("seek", (0.0, 0.0, 0.0), t)
            self.seek_was_cmd = False
            self.ev(t, "seeker", "disabled")
            self._seek_status(t)

    def _seek_status(self, t: float) -> None:
        st = self.seeker.status(t)
        est = st.get("estimate")
        if est is not None:
            st["error_m"] = math.hypot(est[0] - self.leak[0], est[1] - self.leak[1])
        st["enabled"] = self.seek_enabled
        st["rth_active"] = self.rth.active
        self.m.on_seek_status(json.loads(json.dumps(st, default=float)), t)

    # ------------------------------------------------------------------ 10 Hz round
    def tick(self, t: float, pose, vel_w, wz: float, loco_mode: str, rec_state: str, frame=None, gt_pose=None,
             omega_norm: float = 0.0) -> None:
        """One 10 Hz round of every node.  ``pose`` = odometry (x, y, yaw), ``vel_w`` = world (vx, vy)."""
        from module3_gas_safety.battery_model import STATUS_CHARGING, STATUS_FULL
        from module3_gas_safety.rth_logic import MONITOR, RthInputs
        t = float(t)
        dt = 0.0 if self.t_last is None else max(0.0, min(0.5, t - self.t_last))
        self.t_last = t
        x, y, yaw = pose
        speed = math.hypot(vel_w[0], vel_w[1])
        m = self.m
        m.set_pose(x, y, yaw, t)
        lm = str(loco_mode).upper()
        m.on_locomotion_state({"mode": lm, "recovery_state": rec_state}, t)
        # return_to_home / source_seeker nodes: the same obstacle-memory fall guard (/locomotion/state)
        from dusan_common.obstacle_layer import robot_down
        down = robot_down(lm, rec_state)
        self.rth.obstacles.set_upright(not down, t)
        self.seeker.obstacles.set_upright(not down, t)
        # ---- gas_dispersion_sim (10 Hz)
        if dt > 0:
            self.gas.step(dt)
            pts = self.gas.body_points(x, y, yaw, self.nose_off + self.arr_off)
            c = self.gas.concentration_at(pts[:, 0], pts[:, 1])
            self.c_nose = float(self.nose.update(c[:1], dt)[0])
            self.c_arr = [float(v) for v in self.array.update(c[1:], dt)]
            u, v = self.gas.wind_at(x, y)
            self.wind = tuple(float(q) for q in self.anemo.update([u, v], dt))
            self.max_ppm = max(self.max_ppm, self.c_nose)
            m.on_gas(self.c_nose, t)
        # ---- battery_simulator (10 Hz model, 2 Hz publish)
        if dt > 0:
            tau = max(float(self.bp["speed_filter_tau"]), 1e-3)
            self.speed_f += (1.0 - math.exp(-dt / tau)) * (speed - self.speed_f)
            docked = self.dockdet.update(t, x, y, self.speed_f) if self.bp["charge_on_dock"] else False
            mode = "walk" if speed > 0.02 else lm.lower()
            self.bat.step(dt, speed, wz, mode, docked)
        if t - self.t_bat >= 0.5:
            self.t_bat = t
            self.bat_pub = (100.0 * float(self.bat.soc), self.bat.status in (STATUS_CHARGING, STATUS_FULL))
        # ---- comm_simulator (5 Hz)
        if t - self.t_comm >= 0.2:
            dtc = 0.0 if self.t_comm < 0 else t - self.t_comm
            self.t_comm = t
            self.rssi = float(self.comm.rssi_at(x, y) + self.fading.step(dtc))
        # ---- obstacle frame (terrain_mapper cloud) to its three consumers
        rframe = None
        if frame is not None:
            if m.obstacle_frame_due(t):
                m.on_obstacle_frame(frame, t)
            if self.seeker.obstacle_frame_due(t):
                try:
                    self.seeker.update_obstacles(frame, (x, y, yaw), t)
                except Exception as e:  # noqa: BLE001 - the node logs and goes on
                    self.ev(t, "error", f"seeker obstacle: {e}")
            rframe = frame if self.rth.obstacle_frame_due(t) else None
        # ---- return_to_home (10 Hz)
        plan = self.rth_plan_in
        self.rth_plan_in = None
        if plan is not None:
            sig = (len(plan), tuple(np.round(plan[0], 2)) if plan else None,
                   tuple(np.round(plan[-1], 2)) if plan else None)
            if sig == self._rth_own_sig:
                plan = None
        cmd = self.rth_cmds.pop(0) if self.rth_cmds else None
        was = self.rth.state
        inp = RthInputs(t=t, x=x, y=y, yaw=yaw, vx=vel_w[0], vy=vel_w[1], battery_pct=self.bat_pub[0],
                        charging=self.bat_pub[1], rssi_dbm=self.rssi, plan=plan, command=cmd,
                        obstacle_frame=rframe)
        rout = self.rth.update(inp)
        if was == MONITOR and self.rth.state != MONITOR:
            self.ev(t, "rth", f"RTH triggered: {self.rth.reason} (bat {self.bat_pub[0]}, rssi {self.rssi})")
            if lm == "SIT":
                self.loco_cmds.append((t, "stand"))
        if rout.plan is not None:
            self._rth_own_sig = ((len(rout.plan), tuple(np.round(rout.plan[0], 2)),
                                  tuple(np.round(rout.plan[-1], 2))) if rout.plan else None)
        if rout.locomotion:
            self.loco_cmds.append((t, rout.locomotion))
        if rout.cmd is not None and rout.rth_active:
            self.mux.update("rth", rout.cmd, t)
        self.rth_status = dict(rout.status)
        changed = self.rth.state != was
        if changed:
            self.ev(t, "rth", f"safety state -> {self.rth.state} ({self.rth.message})")
        if changed or t - self.t_status >= 0.5:
            self.t_status = t
            m.on_safety_status(json.loads(json.dumps(self.rth_status, default=lambda o: None)), t)
        if changed or t - self.t_active >= 0.2:
            self.t_active = t
            m.on_rth_active(bool(self.rth.active), t)
            self.mux.set_flag("rth_active", bool(self.rth.active), t)
        # ---- source_seeker (10 Hz control, 2 Hz status)
        if not self.seek_enabled or self.rth.active or self.c_nose is None:
            if self.seek_was_cmd:
                self.mux.update("seek", (0.0, 0.0, 0.0), t)
            self.seek_was_cmd = False
        else:
            vx, vy, w = self.seeker.update(t, x, y, yaw, self.c_nose, self.c_arr, self.wind)
            if self.seeker.active:
                self.mux.update("seek", (vx, vy, w), t)
                self.seek_was_cmd = True
                self.seek_log.append((round(t, 2), round(x, 2), round(y, 2), self.seeker.state))
            else:
                if self.seek_was_cmd:
                    self.mux.update("seek", (0.0, 0.0, 0.0), t)
                self.seek_was_cmd = False
            if self.seeker.state != self._seek_last_state:
                self._seek_last_state = self.seeker.state
                self.ev(t, "seeker", f"seek state -> {self.seeker.state} ({self.seeker.message})")
                self._seek_status(t)
        if t - self.t_seek_status >= 0.5:
            self.t_seek_status = t
            self._seek_status(t)
        # ---- gauge_reader stub (5 Hz while a target is set)
        if self.target and t - self.t_gauge >= 0.2 - 1e-6:
            self.t_gauge = t
            self._read_gauge(t, gt_pose if gt_pose is not None else pose, omega_norm)
        # ---- mission_manager (10 Hz)
        out = m.step(t)
        self._drain_mission(t, out)
        key = (m.base, m.phase)
        if key != self._last_state:
            self._last_state = key
            self.state_log.append((round(t, 2), m.base, m.phase, round(x, 2), round(y, 2)))

    def _read_gauge(self, t: float, pose, omega_norm: float) -> None:
        g = self.gauges.get(self.target)
        if g is None:
            return
        gx, gy, gyaw = pose
        vpe = self.vplanner.evaluate(g, gx, gy, gyaw)
        bad = [r for r in vpe.reason.split(",") if r in ("out_of_image", "no_line_of_sight")]
        inc_ok = math.isfinite(vpe.incidence_deg) and vpe.incidence_deg <= GAUGE_MAX_INCIDENCE_DEG
        px_ok = math.isfinite(vpe.pixel_diameter) and vpe.pixel_diameter >= GAUGE_MIN_PIXELS
        moving = omega_norm > GAUGE_MAX_OMEGA
        ok = not bad and inc_ok and px_ok and vpe.distance <= GAUGE_MAX_DIST and not moving
        rec = self.gauge_log.setdefault(self.target, {"attempts": 0, "ok": 0, "moving": 0, "first": None,
                                                      "reasons": {}})
        rec["attempts"] += 1
        rec["moving"] += int(moving)
        if rec["first"] is None:
            rec["first"] = {"t": round(t, 2), "pose": [round(gx, 3), round(gy, 3), round(gyaw, 3)],
                            "dist": round(vpe.distance, 2), "inc": round(vpe.incidence_deg, 1),
                            "px": round(vpe.pixel_diameter, 1)}
        if not ok:
            key = ",".join(bad) or ("moving" if moving else "px/inc/dist")
            rec["reasons"][key] = rec["reasons"].get(key, 0) + 1
            return
        rec["ok"] += 1
        fs = float(g["max"]) - float(g["min"])
        val = float(g["value"]) + GAUGE_NOISE_FS * fs * float(np.random.default_rng(int(t * 10)).normal())
        self.m.on_gauge_readings({"stamp": round(t, 3), "readings": [
            {"id": self.target, "value": val, "unit": g.get("unit"), "confidence": 0.9, "status": "normal"}]}, t)

    def select(self, t: float):
        src, cmd = self.mux.select(float(t))
        self.mux_src = src
        return cmd

    def pop_loco(self) -> List[Tuple[float, str]]:
        c, self.loco_cmds = self.loco_cmds, []
        return c

    def finished(self) -> bool:
        m = self.m
        return (m.base == "DOCKED" and m.mission_t_end is not None) or (m.base == "IDLE" and m.mission_t0 is not None)


class MissionStackOperator:
    """Harness operator (``GzHarness.cmd_fn``): every 1 / rate s of sim time the
    odometry pose (/odom), the locomotion mode / recovery state and the latest
    terrain_mapper cloud go into :meth:`MissionStack.tick`; /locomotion/command
    strings go into the harness commands; returns the cmd_vel_mux output.
    Ground truth only feeds the gauge stub, the battery speed and the logs."""

    needs_pose = True

    def __init__(self, stack: MissionStack, t_start: float = 3.5, start_cmd: str = "start", rate: float = 10.0,
                 phase: float = 0.0):
        self.S = stack
        self.t_start = float(t_start)
        self.start_cmd = start_cmd
        self.period = 1.0 / float(rate)
        self._next = 0.0
        self._phase: Optional[float] = phase
        self.h = None
        self.bound = False
        self.started = False
        self._cloud_t = None
        # 10 Hz trace: t, x, y, z, yaw, roll, pitch, base, phase, mux source, cmd (3), mode, rec_state
        self.trace: List[Tuple] = []
        self.path_len = 0.0
        self._last_xy = None
        self.frames = 0

    def bind(self, h) -> None:
        self.h = h
        self.bound = True

    def __call__(self, t: float, gt=None):
        if self._phase is not None:
            self._next, self._phase = t + self._phase, None
        if t >= self._next - 1e-9:
            self._next = t + self.period
            self._tick(t, gt)
        return self.S.select(t)

    def _tick(self, t: float, gt) -> None:
        from dusan_common.obstacle_layer import EvidenceFrame
        h, S = self.h, self.S
        with h._lock:
            odom = h._odom
            imu = h._imu
        if odom is not None:
            pose = (float(odom[1][0]), float(odom[1][1]), float(odom[2][2]))
        elif gt is not None:
            pose = (float(gt[0][0]), float(gt[0][1]), float(gt[1][2]))
        else:
            return
        if gt is not None:
            vel_w = (float(gt[2][0]), float(gt[2][1]))
            wz = float(gt[3][2])
            gtp = (float(gt[0][0]), float(gt[0][1]), float(gt[1][2]))
            if self._last_xy is not None:
                self.path_len += math.hypot(gtp[0] - self._last_xy[0], gtp[1] - self._last_xy[1])
            self._last_xy = gtp[:2]
        else:
            vel_w, wz, gtp = (0.0, 0.0), 0.0, pose
        omega = float(np.linalg.norm(imu[2])) if imu is not None else 0.0
        modes = h.log.get("mode") or []
        mode = str(modes[-1]) if modes else "STANDING_UP"
        recs = h.log.get("rec_state") or []
        rec = str(recs[-1]) if recs else "NORMAL"
        frame = None
        ct = getattr(h, "elevation_cloud_t", None)
        if ct is not None and ct != self._cloud_t and h.elevation_cloud is not None:
            self._cloud_t = ct
            m = S.m
            if m.obstacle_frame_due(t) or S.seeker.obstacle_frame_due(t) or S.rth.obstacle_frame_due(t):
                frame = EvidenceFrame.from_elevation_cloud(h.elevation_cloud, "odom",
                                                           float(m.p["obstacle_step_height"]),
                                                           int(m.p["obstacle_step_window"]))
                self.frames += 1
        if not self.started and t >= self.t_start - 1e-9:
            self.started = True
            S.command(self.start_cmd, t)
        S.tick(t, pose, vel_w, wz, mode, rec, frame=frame, gt_pose=gtp, omega_norm=omega)
        for _t, c in S.pop_loco():
            h.commands.append((t + 1e-3, c))
            S.ev(t, "loco_cmd", c)
        if gt is not None:
            c = S.select(t)
            self.trace.append((round(t, 2), round(gtp[0], 3), round(gtp[1], 3), round(float(gt[0][2]), 3),
                               round(gtp[2], 3), round(float(gt[1][0]), 3), round(float(gt[1][1]), 3), S.m.base,
                               S.m.phase, S.mux_src, round(c[0], 3), round(c[1], 3), round(c[2], 3), mode, rec))
