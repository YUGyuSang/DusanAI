#!/usr/bin/env python3
"""
Offline evaluation of the module-1 locomotion core (no ROS, runs on Windows).

Plant: a single rigid body (mass, full 3x3 inertia, optional CoM offset /
payload) with ideal point-foot contacts: stance feet are fixed in the world at
the footholds chosen by the MPC (Raibert + capture point), every stance foot
applies exactly the MPC ground-reaction force of the current step (peak force
f / c of the contact-averaged step force), swing feet apply nothing.  The
schedule comes from the same GaitScheduler the controller uses.  A leg
kinematics check (analytic IK reachability of every stance foot from the
simulated body pose) flags plans that a real leg could not follow, and the
joint torques tau = -J^T R^T f of the stance legs are recorded.

Experiments
  (a) velocity tracking with trot and crawl (accel-limited command profiles),
  (b) disturbance rejection: lateral and forward push impulses while walking,
      payload (+20 % mass) and CoM offset, with / without the acceleration-bias
      observer; peak roll/pitch, recovery time, falls
      (|roll| or |pitch| > 55 deg, height < 0.12 m, or unreachable stance leg),
  (c) MPC solve-time histogram (all solves of all runs),
  (d) static stability margin of the crawl (CoM projection vs support polygon)
      for duty 0.75 (DESIGN_SPEC) and 0.80, with the CoM sway.
  (+) an example 5-second prediction (CoM path, footholds, GRFs, torques).

Outputs: docs/images/loco_*.png and docs/results/locomotion_eval.json.

    python tools/eval_locomotion.py            # full evaluation (~2-4 min)
    python tools/eval_locomotion.py --quick    # shorter runs
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")       # see locomotion_controller_node.py
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in ("dusan_common", "module1_locomotion"):
    _d = os.path.join(ROOT, _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dusan_common import robot_model as rm  # noqa: E402
from dusan_common.geometry import rot_to_euler  # noqa: E402
from module1_locomotion import kinematics as kin  # noqa: E402
from module1_locomotion.controller import DEFAULT_PARAMS, mpc_config_from_params  # noqa: E402
from module1_locomotion.gait import GaitScheduler, make_gaits, support_margin  # noqa: E402
from module1_locomotion.mpc import AccelBiasObserver, ConvexMPC, MPCInput  # noqa: E402

G = 9.81
DEG = math.pi / 180.0
IMG_DIR = os.path.join(ROOT, "docs", "images")
RES_DIR = os.path.join(ROOT, "docs", "results")


def _expm_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues: exp([w]x)."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


class RigidBodyPlant:
    """Single rigid body with point-foot forces (CoM state, body-frame rates)."""

    def __init__(self, mass_scale=1.0, com_offset=(0.0, 0.0, 0.0), inertia_scale=1.0, height=0.34):
        self.m = rm.TOTAL_MASS * mass_scale
        I0 = np.diag(rm.MPC_BODY_INERTIA) * inertia_scale
        self.com_off = np.asarray(com_offset, dtype=float)
        # payload modelled as an extra point mass at the CoM offset (parallel axis)
        extra = self.m - rm.TOTAL_MASS
        d = self.com_off
        self.I = I0 + extra * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        self.I_inv = np.linalg.inv(self.I)
        self.R = np.eye(3)
        self.c = np.array([0.0, 0.0, height]) + self.com_off    # CoM (world)
        self.v = np.zeros(3)
        self.w = np.zeros(3)                                      # body-frame angular velocity

    @property
    def base_pos(self) -> np.ndarray:
        return self.c - self.R @ self.com_off

    def rpy(self):
        return np.array(rot_to_euler(self.R))

    def step(self, dt, contact_pts, forces, f_ext=np.zeros(3)):
        F = forces.sum(axis=0) + np.array([0.0, 0.0, -self.m * G]) + f_ext
        tau_w = np.cross(contact_pts - self.c, forces).sum(axis=0)
        tau_b = self.R.T @ tau_w
        wd = self.I_inv @ (tau_b - np.cross(self.w, self.I @ self.w))
        # semi-implicit Euler
        self.v = self.v + F / self.m * dt
        self.c = self.c + self.v * dt
        self.w = self.w + wd * dt
        self.R = self.R @ _expm_so3(self.w * dt)


class EvalRun:
    """Closed loop: GaitScheduler + ConvexMPC + RigidBodyPlant."""

    def __init__(self, gait="trot", mass_scale=1.0, com_offset=(0.0, 0.0, 0.0), observer=False,
                 mpc_rate=20.0, crawl_duty=0.75, params=None, dt=0.001):
        p = dict(DEFAULT_PARAMS)
        p.update(params or {})
        self.p = p
        self.cfg = mpc_config_from_params(p)
        self.mpc = ConvexMPC(self.cfg)
        self.gaits = make_gaits(float(p["trot_period"]), float(p["trot_duty"]), float(p["crawl_period"]),
                                crawl_duty)
        self.sched = GaitScheduler(self.gaits, "stand")
        self.sched.reset(0.0, "stand")
        self.gait_name = gait
        self.plant = RigidBodyPlant(mass_scale, com_offset)
        self.feet = kin.nominal_feet_body() + np.array([0.0, 0.0, 0.34])
        self.feet[:, 2] = rm.FOOT_RADIUS
        self.contact = np.ones(4, dtype=bool)
        self.observer = AccelBiasObserver(tau=0.3) if observer else None
        self.dt = dt
        self.mpc_period = 1.0 / mpc_rate
        self.res = None
        self.t_res = -1e9
        self.cmd = np.zeros(3)
        self.solve_times = []
        self.cpu_times = []
        self.qp_iters = []
        self.statuses = {}
        self.planned = [None] * 4            # landing point of the current swing

    # ---------------------------------------------------------------- helpers
    def _limits(self):
        g = self.gait_name
        return np.array([self.p[f"{g}_max_vx"], self.p[f"{g}_max_vy"], self.p[f"{g}_max_wz"]])

    def _landing(self, leg, t_td):
        res = self.res
        best = None
        if res is not None:
            for (lg, ttd, x, y, z) in res.foothold_events:
                if lg == leg:
                    err = abs(res.t + ttd - t_td)
                    if err < 0.12 and (best is None or err < best[0]):
                        best = (err, np.array([x, y, z]))
        if best is not None:
            return best[1]
        # fallback: under the hip
        R = self.plant.R
        nb = kin.nominal_feet_body()[leg]
        p = self.plant.base_pos + R @ np.array([nb[0], nb[1], 0.0])
        return np.array([p[0], p[1], rm.FOOT_RADIUS])

    def _solve(self, t):
        pl = self.plant
        rpy = pl.rpy()
        inp = MPCInput(t=t, pos=pl.base_pos.copy(), rpy=rpy, vel=pl.v.copy(), omega=pl.R @ pl.w,
                       feet=self.feet.copy(), cmd=tuple(self.cmd), body_height=0.34, ground_z=0.0,
                       accel_bias=self.observer.a_bias.copy() if self.observer else (0.0, 0.0, 0.0),
                       ang_accel_bias=self.observer.alpha_bias.copy() if self.observer else (0.0, 0.0, 0.0))
        res = self.mpc.solve(inp, self.sched)
        self.res = res
        self.t_res = t
        self.solve_times.append(res.solve_time)
        self.cpu_times.append(res.solve_cpu_time)
        self.qp_iters.append(res.qp_iterations)
        self.statuses[res.status] = self.statuses.get(res.status, 0) + 1

    # ---------------------------------------------------------------- run
    def run(self, T, cmd_fn=None, push_fn=None, start_walking=0.3):
        n = int(round(T / self.dt))
        ctrl_every = int(round(0.01 / self.dt))
        log = {k: [] for k in ("t", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "wz",
                               "cmd_vx", "cmd_vy", "cmd_wz", "fz_sum", "margin", "reach", "tau_ratio",
                               "phase", "f_ext")}
        acc = np.array([self.p["max_accel_x"], self.p["max_accel_y"], self.p["max_accel_wz"]], dtype=float)
        fallen = False
        fall_t = None
        unreach_time = 0.0
        tau_ratio = 0.0
        reach_ok = True
        for i in range(n):
            t = i * self.dt
            pl = self.plant
            if i % ctrl_every == 0:
                if t >= start_walking and self.sched.name == "stand" and self.sched.pending is None:
                    self.sched.request(self.gait_name)
                raw = np.asarray(cmd_fn(t) if cmd_fn else (0.0, 0.0, 0.0), dtype=float)
                raw = np.clip(raw, -self._limits(), self._limits())
                self.cmd = self.cmd + np.clip(raw - self.cmd, -acc * 0.01, acc * 0.01)
                gs = self.sched.update(t)
                for leg in range(4):
                    if gs.liftoff[leg]:
                        self.planned[leg] = None
                    if gs.touchdown[leg]:
                        self.feet[leg] = self._landing(leg, t)
                self.contact = gs.contacts.copy()
                if t - self.t_res >= self.mpc_period - 1e-9:
                    self._solve(t)
                # kinematic feasibility and torques of the stance legs
                rpy = pl.rpy()
                fb = kin.world_to_body(self.feet, pl.base_pos, pl.R)
                q, ok = kin.leg_ik(fb, np.arange(4))
                reach_ok = bool(np.all(ok[self.contact]))
                unreach_time = 0.0 if reach_ok else unreach_time + 0.01
                fw = self._forces(t)
                tau = kin.leg_torques(q, np.arange(4), fw @ pl.R)
                tau_ratio = float(np.max(np.abs(tau).reshape(-1) / kin.EFFORT_LIMITS))
                if self.observer is not None:
                    self.observer.update(t, pl.v, pl.R @ pl.w, rpy[2], self.feet - [0, 0, rm.FOOT_RADIUS],
                                         pl.base_pos, fw)
                log["t"].append(t)
                bp = pl.base_pos
                log["x"].append(bp[0]); log["y"].append(bp[1]); log["z"].append(bp[2])
                log["roll"].append(rpy[0]); log["pitch"].append(rpy[1]); log["yaw"].append(rpy[2])
                vb = pl.R.T @ pl.v
                log["vx"].append(vb[0]); log["vy"].append(vb[1]); log["vz"].append(pl.v[2])
                log["wz"].append((pl.R @ pl.w)[2])
                log["cmd_vx"].append(self.cmd[0]); log["cmd_vy"].append(self.cmd[1]); log["cmd_wz"].append(self.cmd[2])
                log["fz_sum"].append(float(fw[:, 2].sum()))
                log["margin"].append(support_margin(self.feet[:, :2], self.contact, pl.c[:2]))
                log["reach"].append(reach_ok)
                log["tau_ratio"].append(tau_ratio)
                log["phase"].append(gs.phase)
                fe = push_fn(t) if push_fn else np.zeros(3)
                log["f_ext"].append(float(np.linalg.norm(fe)))
                rpy_now = rpy
                if not fallen and (abs(rpy_now[0]) > 55 * DEG or abs(rpy_now[1]) > 55 * DEG
                                   or bp[2] < 0.12 or unreach_time > 0.1):
                    fallen = True
                    fall_t = t
                if fallen and t - fall_t > 0.3:
                    break
            if i % ctrl_every != 0:
                # contacts follow the schedule at the plant rate (1 ms), so the
                # contact-averaged MPC forces are applied for the right time
                gs_fine = self.sched.update(t)
                for leg in range(4):
                    if gs_fine.touchdown[leg]:
                        self.feet[leg] = self._landing(leg, t)
                self.contact = gs_fine.contacts.copy()
            fw = self._forces(t)
            f_ext = push_fn(t) if push_fn else np.zeros(3)
            pl.step(self.dt, self.feet - np.array([0.0, 0.0, rm.FOOT_RADIUS]), fw, f_ext)
        out = {k: np.asarray(v) for k, v in log.items()}
        out["fallen"] = fallen
        out["fall_t"] = fall_t
        return out

    def _forces(self, t):
        """Stance-foot forces of the current MPC step (peak force during contact)."""
        fw = np.zeros((4, 3))
        res = self.res
        if res is None:
            fw[:, 2] = self.plant.m * G / 4.0
            return fw
        k = int(np.clip(np.searchsorted(res.times, t - res.t, side="right") - 1, 0, len(res.dts) - 1))
        f = res.forces[k].reshape(4, 3)
        c = res.contacts[k]
        for leg in range(4):
            if self.contact[leg] and c[leg] >= self.cfg.contact_threshold:
                fw[leg] = f[leg] / c[leg]
        return fw


# ---------------------------------------------------------------------------- metrics
def recovery_time(log, t_end_push, v_tol=0.15, ang_tol=3 * DEG, hold=0.5):
    t = log["t"]
    ok = (np.hypot(log["vx"] - log["cmd_vx"], log["vy"] - log["cmd_vy"]) < v_tol) & \
         (np.abs(log["roll"]) < ang_tol) & (np.abs(log["pitch"]) < ang_tol)
    dt = t[1] - t[0]
    need = int(round(hold / dt))
    idx = np.nonzero(t >= t_end_push)[0]
    for i in idx:
        if i + need <= len(ok) and np.all(ok[i:i + need]):
            return float(t[i] - t_end_push)
    return None


def window_rms(err, t, windows):
    m = np.zeros_like(t, dtype=bool)
    for a, b in windows:
        m |= (t >= a) & (t < b)
    return float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")


def cycle_mean(sig, t, period):
    """Moving average over one gait period (centred): removes the periodic
    intra-cycle motion (trot bounce, crawl CoM sway) from a velocity signal."""
    dt = float(t[1] - t[0])
    n = max(1, int(round(period / dt)))
    k = np.ones(n) / n
    return np.convolve(np.pad(sig, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


# ---------------------------------------------------------------------------- experiments
def exp_tracking(quick):
    out = {}
    logs = {}
    # trot: forward, lateral, turning, stop
    T = 10.0 if quick else 14.0

    def trot_cmd(t):
        if t < 1.0:
            return (0.0, 0.0, 0.0)
        if t < 5.0:
            return (0.5, 0.0, 0.0)
        if t < 7.5:
            return (0.0, 0.25, 0.0)
        if t < 11.0:
            return (0.3, 0.0, 0.6)
        return (0.0, 0.0, 0.0)

    run = EvalRun("trot")
    log = run.run(T, trot_cmd)
    logs["trot"] = (log, run)
    win = [(2.5, 5.0), (6.0, 7.5), (8.5, 11.0)]
    P = float(DEFAULT_PARAMS["trot_period"])
    out["trot"] = {
        "fallen": log["fallen"],
        "rms_vx_error": window_rms(log["vx"] - log["cmd_vx"], log["t"], win),
        "rms_vy_error": window_rms(log["vy"] - log["cmd_vy"], log["t"], win),
        "rms_wz_error": window_rms(log["wz"] - log["cmd_wz"], log["t"], win),
        "rms_vx_error_cycle_avg": window_rms(cycle_mean(log["vx"] - log["cmd_vx"], log["t"], P), log["t"], win),
        "rms_vy_error_cycle_avg": window_rms(cycle_mean(log["vy"] - log["cmd_vy"], log["t"], P), log["t"], win),
        "rms_height_error": float(np.sqrt(np.mean((log["z"] - 0.34) ** 2))),
        "max_abs_roll_deg": float(np.max(np.abs(log["roll"])) / DEG),
        "max_abs_pitch_deg": float(np.max(np.abs(log["pitch"])) / DEG),
        "max_torque_ratio": float(np.max(log["tau_ratio"])),
        "stance_legs_reachable": float(np.mean(log["reach"])),
        "distance_m": float(np.hypot(log["x"][-1] - log["x"][0], log["y"][-1] - log["y"][0])),
    }

    def crawl_cmd(t):
        if t < 1.0:
            return (0.0, 0.0, 0.0)
        if t < 6.0:
            return (0.2, 0.0, 0.0)
        if t < (8.0 if quick else 9.0):
            return (0.1, 0.0, 0.3)
        return (0.0, 0.0, 0.0)

    run = EvalRun("crawl")
    log = run.run(9.0 if quick else 11.0, crawl_cmd)
    logs["crawl"] = (log, run)
    win = [(3.0, 6.0)]
    P = float(DEFAULT_PARAMS["crawl_period"])
    sel = (log["t"] >= 3.0) & (log["t"] < 6.0)
    out["crawl"] = {
        "fallen": log["fallen"],
        "note": "instantaneous velocity errors include the intended CoM sway; the cycle-averaged "
                "errors (moving average over one 1.2 s gait cycle) measure the tracking",
        "rms_vx_error": window_rms(log["vx"] - log["cmd_vx"], log["t"], win),
        "rms_vy_error": window_rms(log["vy"] - log["cmd_vy"], log["t"], win),
        "rms_vx_error_cycle_avg": window_rms(cycle_mean(log["vx"] - log["cmd_vx"], log["t"], P), log["t"], win),
        "rms_vy_error_cycle_avg": window_rms(cycle_mean(log["vy"] - log["cmd_vy"], log["t"], P), log["t"], win),
        "mean_speed_3_6s": float(np.polyfit(log["t"][sel], log["x"][sel], 1)[0]) if sel.sum() > 2 else None,
        "commanded_speed_3_6s": 0.2,
        "lateral_offset_at_6s_m": float(np.interp(6.0, log["t"], log["y"])),   # straight segment, vy cmd = 0
        "sway_amplitude_y_m": float(0.5 * np.ptp(log["y"][sel] - cycle_mean(log["y"], log["t"], P)[sel]))
        if sel.any() else None,
        "rms_wz_error": window_rms(log["wz"] - log["cmd_wz"], log["t"], [(7.0, 8.0)]),
        "rms_height_error": float(np.sqrt(np.mean((log["z"] - 0.34) ** 2))),
        "max_abs_roll_deg": float(np.max(np.abs(log["roll"])) / DEG),
        "max_abs_pitch_deg": float(np.max(np.abs(log["pitch"])) / DEG),
        "max_torque_ratio": float(np.max(log["tau_ratio"])),
        "stance_legs_reachable": float(np.mean(log["reach"])),
    }
    return out, logs


def exp_push(quick):
    results = []
    logs = {}
    impulses = [10.0, 20.0, 30.0, 40.0] if quick else [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    t_push, dur = 2.5, 0.1
    for direction in ("lateral", "forward"):
        for J in impulses:
            F = J / dur
            vec = np.array([0.0, F, 0.0]) if direction == "lateral" else np.array([F, 0.0, 0.0])

            def push(t, vec=vec):
                return vec if t_push <= t < t_push + dur else np.zeros(3)

            run = EvalRun("trot")
            log = run.run(6.0, lambda t: (0.3, 0.0, 0.0) if t >= 1.0 else (0.0, 0.0, 0.0), push)
            sel = log["t"] >= t_push
            rec = None if log["fallen"] else recovery_time(log, t_push + dur)
            results.append({
                "direction": direction, "impulse_Ns": J, "force_N": F, "duration_s": dur,
                "fallen": bool(log["fallen"]),
                "peak_roll_deg": float(np.max(np.abs(log["roll"][sel])) / DEG) if sel.any() else None,
                "peak_pitch_deg": float(np.max(np.abs(log["pitch"][sel])) / DEG) if sel.any() else None,
                "peak_lateral_velocity": float(np.max(np.abs(log["vy"][sel]))) if sel.any() else None,
                "recovery_time_s": rec,
                "max_torque_ratio": float(np.max(log["tau_ratio"][sel])) if sel.any() else None,
            })
            logs[(direction, J)] = log
    return results, logs


def exp_payload(quick):
    cases = [
        ("nominal", 1.0, (0.0, 0.0, 0.0), False),
        ("mass +20 %", 1.2, (0.0, 0.0, 0.0), False),
        ("mass +20 % + observer", 1.2, (0.0, 0.0, 0.0), True),
        ("CoM +4 cm fwd, +20 %", 1.2, (0.04, 0.0, 0.0), False),
        ("CoM +4 cm fwd, +20 % + observer", 1.2, (0.04, 0.0, 0.0), True),
    ]
    out = []
    logs = {}
    for name, ms, off, obs in cases:
        run = EvalRun("trot", mass_scale=ms, com_offset=off, observer=obs)
        log = run.run(6.0, lambda t: (0.3, 0.0, 0.0) if t >= 1.0 else (0.0, 0.0, 0.0))
        ss = log["t"] >= 3.5
        out.append({
            "case": name, "mass_kg": run.plant.m, "com_offset_m": list(off), "observer": obs,
            "fallen": bool(log["fallen"]),
            "steady_height_error_m": float(np.mean(log["z"][ss] - 0.34)) if ss.any() else None,
            "steady_pitch_deg": float(np.mean(log["pitch"][ss]) / DEG) if ss.any() else None,
            "rms_vx_error": float(np.sqrt(np.mean((log["vx"][ss] - log["cmd_vx"][ss]) ** 2))) if ss.any() else None,
            "max_torque_ratio": float(np.max(log["tau_ratio"])),
            "observer_bias": run.observer.a_bias.tolist() if obs else None,
        })
        logs[name] = log
    return out, logs


def exp_crawl_margin(quick):
    out = {}
    logs = {}
    for duty in (0.75, 0.80):
        run = EvalRun("crawl", crawl_duty=duty)
        log = run.run(8.0 if quick else 10.0, lambda t: (0.15, 0.0, 0.0) if t >= 1.0 else (0.0, 0.0, 0.0))
        sel = (log["t"] >= 3.0) & np.isfinite(log["margin"])
        m = log["margin"][sel]
        # nominal (schedule) margin with the sway reference, CoM tracking it
        # exactly: second gait cycle (the first one contains the start transient)
        sch = GaitScheduler(make_gaits(crawl_duty=duty), "crawl")
        sch.reset(0.0, "crawl")
        period = float(DEFAULT_PARAMS["crawl_period"])
        ts_all = np.arange(0.0, 2.0 * period, 0.005)
        feet = kin.nominal_feet_body()[:, :2]
        sw = sch.sway_at(ts_all, feet, gain=float(DEFAULT_PARAMS["mpc_sway_gain"]),
                         smooth=float(DEFAULT_PARAMS["mpc_sway_smooth"]), dt=0.0025)
        cc = sch.contacts_at(ts_all)
        cyc = ts_all >= period
        ts = ts_all[cyc] - period
        nominal = np.array([support_margin(feet, cc[i], sw[i]) for i in np.nonzero(cyc)[0]])
        no_sway = np.array([support_margin(feet, cc[i], [0.0, 0.0]) for i in np.nonzero(cyc)[0]])
        sway_v = np.gradient(sw[cyc], ts, axis=0)
        sway_a = np.gradient(sway_v, ts, axis=0)
        out[f"duty_{duty:.2f}"] = {
            "fallen": bool(log["fallen"]),
            "fall_time_s": log["fall_t"],
            "closed_loop_min_margin_m": float(np.min(m)) if m.size else None,
            "closed_loop_mean_margin_m": float(np.mean(m)) if m.size else None,
            "closed_loop_fraction_above_2cm": float(np.mean(m > 0.02)) if m.size else None,
            "reference_peak_sway_velocity_mps": float(np.max(np.linalg.norm(sway_v, axis=1))),
            "reference_peak_sway_accel_mps2": float(np.max(np.linalg.norm(sway_a, axis=1))),
            "reference_min_margin_m": float(np.min(nominal)),
            "reference_mean_margin_m": float(np.mean(nominal)),
            "reference_fraction_above_2cm": float(np.mean(nominal > 0.02)),
            "no_sway_mean_margin_m": float(np.mean(no_sway)),
        }
        logs[duty] = (log, ts, nominal, no_sway)
    return out, logs


def example_prediction():
    sch = GaitScheduler(make_gaits(), "trot")
    sch.reset(0.0, "trot")
    sch.update(0.0)
    mpc = ConvexMPC(mpc_config_from_params(DEFAULT_PARAMS))
    feet = kin.nominal_feet_body() + np.array([0.0, 0.0, 0.34])
    inp = MPCInput(t=0.0, pos=np.array([0.0, 0.0, 0.34]), rpy=np.zeros(3), vel=np.array([0.4, 0.0, 0.0]),
                   omega=np.zeros(3), feet=feet, cmd=(0.4, 0.0, 0.35), ground_z=0.0)
    mpc.solve(inp, sch)                      # warm up
    return mpc.solve(inp, sch)


# ---------------------------------------------------------------------------- plots
def make_plots(track_logs, push_logs, payload_logs, margin_logs, solve_times, pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(IMG_DIR, exist_ok=True)
    c_ref, c_act, c2, c3 = "#888888", "#1f6fb4", "#d6604d", "#4daf4a"

    # (a) tracking
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex="col")
    for col, key in enumerate(("trot", "crawl")):
        log, _run = track_logs[key]
        t = log["t"]
        axs[0, col].plot(t, log["cmd_vx"], "--", color=c_ref, label="vx cmd")
        axs[0, col].plot(t, log["vx"], color=c_act, label="vx")
        axs[0, col].plot(t, log["cmd_vy"], "--", color="#bbbbbb", label="vy cmd")
        axs[0, col].plot(t, log["vy"], color=c2, label="vy")
        axs[0, col].set_ylabel("velocity [m/s]")
        axs[0, col].legend(loc="upper right", fontsize=8, ncol=2)
        axs[0, col].set_title(f"{key}: velocity tracking (rigid-body plant, MPC forces)")
        axs[1, col].plot(t, log["cmd_wz"], "--", color=c_ref, label="wz cmd")
        axs[1, col].plot(t, log["wz"], color=c_act, label="wz")
        axs[1, col].set_ylabel("yaw rate [rad/s]")
        axs[1, col].legend(loc="upper right", fontsize=8)
        axs[2, col].plot(t, np.degrees(log["roll"]), color=c_act, label="roll")
        axs[2, col].plot(t, np.degrees(log["pitch"]), color=c2, label="pitch")
        axs[2, col].plot(t, (log["z"] - 0.34) * 100, color=c3, label="height err [cm]")
        axs[2, col].set_ylabel("deg / cm")
        axs[2, col].set_xlabel("time [s]")
        axs[2, col].legend(loc="upper right", fontsize=8)
        for a in axs[:, col]:
            a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "loco_tracking.png"), dpi=110)
    plt.close(fig)

    # (b) pushes
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for (direction, J), log in sorted(push_logs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ax = axs[0] if direction == "lateral" else axs[1]
        sig = log["roll"] if direction == "lateral" else log["pitch"]
        lbl = f"{J:.0f} Ns" + (" (fall)" if log["fallen"] else "")
        ax.plot(log["t"], np.degrees(sig), label=lbl)
    axs[0].set_title("lateral push (0.1 s at t = 2.5 s): roll")
    axs[1].set_title("forward push (0.1 s at t = 2.5 s): pitch")
    for a in axs:
        a.set_xlabel("time [s]")
        a.set_ylabel("deg")
        a.set_xlim(2.0, 6.0)
        a.set_ylim(-60, 60)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "loco_push.png"), dpi=110)
    plt.close(fig)

    # payload
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for name, log in payload_logs.items():
        axs[0].plot(log["t"], (log["z"] - 0.34) * 100, label=name)
        axs[1].plot(log["t"], np.degrees(log["pitch"]), label=name)
    axs[0].set_ylabel("height error [cm]")
    axs[1].set_ylabel("pitch [deg]")
    axs[0].set_title("payload / CoM offset (MPC model: nominal 15 kg)")
    axs[1].set_title("pitch with CoM offset")
    for a in axs:
        a.set_xlabel("time [s]")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "loco_payload.png"), dpi=110)
    plt.close(fig)

    # (c) solve times
    st = np.asarray(solve_times) * 1e3
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(st, bins=60, color=c_act, alpha=0.85)
    for q, ls in ((50, "-"), (95, "--")):
        v = np.percentile(st, q)
        ax.axvline(v, color=c2, ls=ls, label=f"p{q} = {v:.1f} ms")
    ax.axvline(30.0, color="k", ls=":", label="30 ms target")
    ax.set_xlabel("MPC solve time [ms] (build + ADMM + polish + torques, numpy, 1 BLAS thread)")
    ax.set_ylabel("count")
    ax.set_title(f"5 s horizon MPC solve time, {len(st)} solves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "loco_solve_time.png"), dpi=110)
    plt.close(fig)

    # (d) crawl margin
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for duty, (log, ts, nominal, no_sway) in margin_logs.items():
        axs[0].plot(ts / 1.2, nominal * 100, label=f"duty {duty:.2f} reference sway")
        axs[0].plot(ts / 1.2, no_sway * 100, ":", label=f"duty {duty:.2f} no sway")
        sel = log["t"] >= 3.0
        axs[1].plot(log["t"][sel], log["margin"][sel] * 100, label=f"duty {duty:.2f} closed loop")
    axs[0].set_xlabel("gait phase")
    axs[0].set_title("crawl static stability margin (schedule + sway reference)")
    axs[1].set_xlabel("time [s]")
    axs[1].set_title("crawl static margin of the simulated CoM (0.15 m/s)")
    for a in axs:
        a.set_ylabel("margin [cm]")
        a.axhline(0.0, color="k", lw=0.8)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "loco_crawl_margin.png"), dpi=110)
    plt.close(fig)

    # (+) example horizon
    fig = plt.figure(figsize=(13, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(pred.x_ref[:, 3], pred.x_ref[:, 4], "--", color=c_ref, label="reference")
    ax1.plot(pred.x_pred[:, 3], pred.x_pred[:, 4], "o-", color=c_act, ms=3, label="predicted CoM (5 s)")
    cols = ("#e41a1c", "#377eb8", "#ff7f00", "#4daf4a")
    for leg in range(4):
        ev = np.array([[e[2], e[3]] for e in pred.foothold_events if e[0] == leg])
        if len(ev):
            ax1.plot(ev[:, 0], ev[:, 1], "x", color=cols[leg], label=f"{rm.LEG_NAMES[leg]} footholds")
    ax1.set_aspect("equal")
    ax1.set_title("trot 0.4 m/s, 0.35 rad/s: predicted CoM and footholds")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(2, 2, 2)
    f = pred.forces.reshape(-1, 4, 3)
    for leg in range(4):
        ax2.step(pred.times[:-1], f[:, leg, 2], where="post", color=cols[leg], label=f"{rm.LEG_NAMES[leg]} fz")
    ax2.set_title("predicted vertical GRF (contact-averaged per step)")
    ax2.set_xlabel("horizon time [s]")
    ax2.set_ylabel("N")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pred.times, pred.x_pred[:, 5], color=c_act, label="z")
    ax3.plot(pred.times, np.degrees(pred.x_pred[:, 0]) / 100 + 0.34, color=c2, label="roll/100 [deg] + 0.34")
    ax3.plot(pred.times, pred.x_pred[:, 2] / 10 + 0.34, color=c3, label="yaw/10 [rad] + 0.34")
    ax3.set_xlabel("horizon time [s]")
    ax3.set_title("predicted body states")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)
    ax4 = fig.add_subplot(2, 2, 4)
    tq = pred.torques.reshape(-1, 4, 3)
    for leg in range(4):
        ax4.step(pred.times[:-1], tq[:, leg, 2], where="post", color=cols[leg], label=f"{rm.LEG_NAMES[leg]} knee")
        ax4.step(pred.times[:-1], tq[:, leg, 1], where="post", color=cols[leg], ls=":", lw=0.8)
    ax4.axhline(0.85 * 60.0, color="k", ls="--", lw=0.8, label="85 % knee limit")
    ax4.set_title("predicted joint torques (knee solid, hip_y dotted)")
    ax4.set_xlabel("horizon time [s]")
    ax4.set_ylabel("Nm")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "loco_horizon.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true", help="shorter runs")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    all_solve = []
    all_cpu = []

    print("(a) velocity tracking ...", flush=True)
    track, track_logs = exp_tracking(args.quick)
    for _k, (_log, run) in track_logs.items():
        all_solve += run.solve_times
        all_cpu += run.cpu_times
    print(json.dumps(track, indent=1))

    print("(b) push recovery ...", flush=True)
    push, push_logs = exp_push(args.quick)
    for r in push:
        print(f"  {r['direction']:8s} {r['impulse_Ns']:5.0f} Ns  fall={r['fallen']!s:5s} "
              f"roll={r['peak_roll_deg']:.1f} pitch={r['peak_pitch_deg']:.1f} rec={r['recovery_time_s']}")
    print("(b) payload / CoM offset ...", flush=True)
    payload, payload_logs = exp_payload(args.quick)
    for r in payload:
        print(f"  {r['case']:34s} fall={r['fallen']!s:5s} dz={r['steady_height_error_m']:+.4f} m "
              f"pitch={r['steady_pitch_deg']:+.2f} deg")

    print("(d) crawl stability margin ...", flush=True)
    margin, margin_logs = exp_crawl_margin(args.quick)
    print(json.dumps(margin, indent=1))

    # (c) solve-time statistics: an extra closed-loop run collects the solver telemetry
    run = EvalRun("trot")
    run.run(4.0 if args.quick else 8.0, lambda t: (0.4, 0.0, 0.3) if t > 1.0 else (0.0, 0.0, 0.0))
    all_solve += run.solve_times
    all_cpu += run.cpu_times
    st = np.asarray(all_solve)
    ct = np.asarray([c for c in all_cpu if np.isfinite(c)])
    solve = {
        "n": int(st.size), "mean_ms": float(st.mean() * 1e3), "median_ms": float(np.median(st) * 1e3),
        "p95_ms": float(np.percentile(st, 95) * 1e3), "max_ms": float(st.max() * 1e3),
        "fraction_below_30ms": float(np.mean(st < 0.030)),
        # CPU time of the solving thread: independent of other processes on the
        # machine (wall-clock numbers above grow with the machine load); only the
        # mean is meaningful on Windows (thread CPU time has scheduler-tick resolution)
        "cpu_mean_ms": float(ct.mean() * 1e3) if ct.size else None,
        "qp_iterations_mean": float(np.mean(run.qp_iters)), "statuses_last_run": run.statuses,
        "note": "wall clock, single-threaded OpenBLAS on the development PC (shared with other jobs); "
                "build + ADMM + polish + footholds + torques",
    }
    print("(c) solve time:", json.dumps(solve))

    pred = example_prediction()
    falls_push = [r for r in push if r["fallen"]]
    max_ok = {d: max([r["impulse_Ns"] for r in push if r["direction"] == d and not r["fallen"]] or [0.0])
              for d in ("lateral", "forward")}
    result = {
        "description": "Offline evaluation of the 5 s convex MPC on a single-rigid-body plant with ideal "
                       "stance contacts (tools/eval_locomotion.py)",
        "horizon": {"steps": int(pred.dts.size), "seconds": float(pred.horizon),
                    "dts": [float(v) for v in pred.dts]},
        "tracking": track,
        "push": push,
        "push_summary": {"max_impulse_without_fall_Ns": max_ok, "falls": len(falls_push), "runs": len(push)},
        "payload": payload,
        "crawl_static_margin": margin,
        "solve_time": solve,
        "example_prediction": {"status": pred.status, "solve_ms": pred.solve_time * 1e3,
                               "torque_ratio": pred.torque_ratio,
                               "footholds_per_leg": [sum(1 for e in pred.foothold_events if e[0] == leg)
                                                     for leg in range(4)]},
        "runtime_s": time.perf_counter() - t0,
    }
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "locomotion_eval.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        f.write("\n")
    if not args.no_plots:
        make_plots(track_logs, push_logs, payload_logs, margin_logs, all_solve, pred)
    print(f"done in {result['runtime_s']:.0f} s -> {os.path.join(RES_DIR, 'locomotion_eval.json')}")
    return result


if __name__ == "__main__":
    main()
