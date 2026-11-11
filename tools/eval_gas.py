#!/usr/bin/env python3
"""
Offline evaluation of Module 3 (gas dispersion, source seeking, return-to-home).
Runs on Windows without ROS; writes figures to docs/images/ and numbers to
docs/results/gas_eval.json.

    python tools/eval_gas.py                    # full evaluation (40 seek trials + ablation)
    python tools/eval_gas.py --trials 10 --quick --no-ablation
    python tools/eval_gas.py --fallback-map     # use the map rasterised from world_layout.yaml

(a) steady-state plume + mass-consistent wind field   -> docs/images/gas_field.png
(b) Monte-Carlo source seeking (gusty plume, lagged noisy sensors, holonomic
    robot 0.35 m/s, obstacle contacts counted)        -> docs/images/gas_seek_paths.png,
                                                         docs/images/gas_seek_timeline.png
    + ablation: the same trials without obstacle detours (local A*)
(c) RTH trigger scenarios (battery drop, comm shadow ahead on a planned path
    through zone S1, energy reserve)                  -> docs/images/rth_timeline.png,
                                                         docs/images/rth_scenarios.png
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in ("dusan_common", "module3_gas_safety"):
    if os.path.join(ROOT, _p) not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, _p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from dusan_common.layout import Layout  # noqa: E402
from dusan_common.planning import densify, path_length, plan as plan_path  # noqa: E402
from dusan_common.pure_pursuit import PurePursuit  # noqa: E402

from module3_gas_safety.battery_model import BatteryConfig, BatteryModel, DockDetector  # noqa: E402
from module3_gas_safety.comm_model import CommModel  # noqa: E402
from module3_gas_safety.gas_model import GasConfig, GasDispersion2D  # noqa: E402
from module3_gas_safety.layout_map import build_layout_map, load_nav_map  # noqa: E402
from module3_gas_safety.rth_logic import (CHARGING, DOCKED, DOCKING, MONITOR, RTH_NAVIGATING,  # noqa: E402
                                          RTH_PLANNING, GeodesicField, RthConfig, RthInputs, RthLogic)
from module3_gas_safety.sim_harness import RobotSim, SensorSuite, clone_model, run_seek_trial  # noqa: E402
from module3_gas_safety.source_seeking import (CAST, FAILED, FOUND, GRADIENT, SEARCH, SURGE,  # noqa: E402
                                               SeekerConfig)

IMG = os.path.join(ROOT, "docs", "images")
RES = os.path.join(ROOT, "docs", "results")
STATE_COLORS = {SEARCH: "#9e9e9e", SURGE: "#42a5f5", CAST: "#ffb300", GRADIENT: "#43a047", FOUND: "#d81b60",
                FAILED: "#000000", MONITOR: "#e0e0e0", RTH_PLANNING: "#ffcc80", RTH_NAVIGATING: "#ef6c00",
                DOCKING: "#8e24aa", DOCKED: "#5e35b1", CHARGING: "#2e7d32"}


def _extent(g):
    return (g.origin_x, g.origin_x + g.width * g.resolution, g.origin_y, g.origin_y + g.height * g.resolution)


def _draw_map(ax, g, alpha=1.0):
    occ = np.where(g.occ >= 50, 1.0, np.nan)
    ax.imshow(occ, origin="lower", extent=_extent(g), cmap="Greys", vmin=0.0, vmax=1.3, alpha=alpha,
              interpolation="nearest")


# =============================================================================
# (a) plume
# =============================================================================
def eval_plume(grid, layout, cfg: GasConfig):
    t0 = time.perf_counter()
    model = GasDispersion2D.from_nav_map(grid, cfg)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.spin_up(600.0, 0.5)
    t_spin = time.perf_counter() - t0
    X, Y = model.cell_centres()
    fl = model.fluid
    c = model.c
    src = cfg.source_xy
    near = (np.hypot(X - src[0], Y - src[1]) < 0.75) & fl
    door = (X > 13.9) & (X < 14.9) & (np.abs(Y) < 2.8) & fl
    corr = {}
    for xc in (16.0, 20.0, 30.0, 45.0, 60.0):
        sel = (np.abs(X - xc) < 0.13) & (np.abs(Y) < 1.8) & fl
        corr[f"x{int(xc)}"] = float(c[sel].max()) if sel.any() else None
    uc, vc = model.wind.cell_centred()
    sp = np.hypot(uc, vc)[fl]
    # real-time cost of one 10 Hz update with gusts
    m2 = copy.deepcopy(model)
    t0 = time.perf_counter()
    subs = []
    for _ in range(300):
        subs.append(m2.step(0.1))
    step_ms = (time.perf_counter() - t0) / 300 * 1e3
    res = {
        "grid": [int(model.shape[1]), int(model.shape[0])], "resolution_m": model.h,
        "build_s": t_build, "spinup_600s_wall_s": t_spin,
        "projection_iterations": int(model.wind.last_projection_iters),
        "max_divergence_W0": float(np.max(np.abs(model.wind.divergence(*model.wind.W0)))),
        "near_source_max_ppm": float(c[near].max()),
        "doorway_max_ppm": float(c[door].max()), "doorway_mean_ppm": float(c[door].mean()),
        "corridor_max_ppm": corr,
        "wind_speed_mean": float(sp.mean()), "wind_speed_max": float(sp.max()),
        "step_10hz_ms": step_ms, "substeps_mean": float(np.mean(subs)),
        "mass_budget_rel_error": float(abs(model.total_mass + model.outflow + model.decayed - model.emitted)
                                       / max(model.emitted, 1e-9)),
    }
    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10.5), gridspec_kw={"height_ratios": [1.35, 1.0]})
    cm = np.where(fl & (c > 0.3), c, np.nan)
    for ax, xlim, stride, title in ((axes[0], (-10.8, 16.2), 3, "turbine hall"),
                                    (axes[1], (-10.8, 64.4), 5, "hall + corridor")):
        im = ax.imshow(cm, origin="lower", extent=model.extent, cmap="inferno", norm=LogNorm(0.5, 2000),
                       interpolation="bilinear")
        _draw_map(ax, grid, 0.85)
        s = stride
        q_u = np.where(fl, uc, np.nan)[::s, ::s]
        q_v = np.where(fl, vc, np.nan)[::s, ::s]
        ax.quiver(X[::s, ::s], Y[::s, ::s], q_u, q_v, color="#4dd0e1", scale=9.0, width=0.0016)
        ax.plot(*src, marker="*", color="#00e676", ms=16, mec="k", label="leak (layout gas_leak.source)")
        dk = layout.dock_pose
        ax.plot(dk[0], dk[1], "s", color="#2979ff", ms=9, mec="k", label="dock")
        ax.add_patch(Rectangle((13.9, -2.8), 1.0, 5.6, fill=False, ec="#00e676", ls="--", lw=1.2))
        ax.set_xlim(*xlim)
        ax.set_ylim(-9.8, 9.8)
        ax.set_aspect("equal")
        ax.set_title(f"Steady-state plume ({title}): Q = {cfg.emission_rate:.0f} ppm m$^2$/s, "
                     f"k = {cfg.decay_rate} 1/s, K = {cfg.diffusivity} m$^2$/s, wind_base = {cfg.wind_base}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.01)
    cb.set_label("concentration [ppm] (log)")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].text(-10.3, 8.0, f"near leak {res['near_source_max_ppm']:.0f} ppm\n"
                             f"doorway max {res['doorway_max_ppm']:.0f} / mean {res['doorway_mean_ppm']:.0f} ppm\n"
                             f"10 Hz update {step_ms:.2f} ms", color="w", fontsize=9, va="top",
                 bbox=dict(fc="k", alpha=0.5))
    fig.savefig(os.path.join(IMG, "gas_field.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return model, res


# =============================================================================
# (b) Monte-Carlo source seeking
# =============================================================================
def sample_starts(grid, n, rng):
    starts = []
    while len(starts) < n:
        if rng.uniform() < 0.85:
            x, y = rng.uniform(2.5, 14.5), rng.uniform(-8.5, 8.5)      # east half of the hall
        else:
            x, y = rng.uniform(14.6, 18.0), rng.uniform(-1.0, 1.0)     # doorway / corridor mouth
        if grid.clearance(x, y) >= 0.6:
            starts.append((float(x), float(y), float(rng.uniform(-math.pi, math.pi))))
    return starts


def start_region(x, y):
    """Coarse start region of a trial (for the per-region statistics)."""
    if x > 14.6:
        return "doorway"
    if x > 9.9:
        return "east_bay"           # behind the interior wall / columns at x ~ 9.9 m
    return "main_hall"


def eval_seeking(base_model, grid, layout, n_trials, seed, timeout, success_radius=1.0, figures=True,
                 label="", **cfg_overrides):
    rng = np.random.default_rng(seed)
    starts = sample_starts(grid, n_trials, rng)
    truth = tuple(layout.gas_leak["source"][:2])
    cfg = SeekerConfig(timeout=timeout, **cfg_overrides)
    trials = []
    t_wall = time.perf_counter()
    for k, st in enumerate(starts):
        m = clone_model(base_model, seed=10_000 + 17 * k + seed)
        warm = float(rng.uniform(20.0, 60.0))
        r = run_seek_trial(m, grid, st, cfg, SensorSuite(seed=seed + k), true_source=truth, warmup=warm,
                           keep_log=figures)
        ok = bool(r.found and r.error_m is not None and r.error_m < success_radius)
        trials.append({"start": [round(v, 3) for v in st], "state": r.state, "found": r.found, "success": ok,
                       "error_m": r.error_m, "time_s": r.time_s, "path_length_m": r.path_length,
                       "contacts": r.contacts, "max_ppm": r.max_ppm, "warmup_s": warm,
                       "estimate": r.estimate, "detours": r.detours, "region": start_region(st[0], st[1]),
                       "states": [s for _, s, _ in r.transitions],
                       "transitions": [[round(tt, 1), s] for tt, s, _ in r.transitions],
                       "_traj": r.trajectory, "_log": r.log})
        print(f"  {label}trial {k:2d} start ({st[0]:6.2f},{st[1]:6.2f}) {r.state:7s} err "
              f"{'-' if r.error_m is None else f'{r.error_m:.2f}'} m  t {r.time_s:5.1f} s  L {r.path_length:5.1f} m"
              f"  detours {r.detours}", flush=True)
    wall = time.perf_counter() - t_wall
    found = [t for t in trials if t["found"]]
    succ = [t for t in trials if t["success"]]
    errs = np.array([t["error_m"] for t in found]) if found else np.array([np.nan])
    stats = {
        "trials": n_trials, "success_radius_m": success_radius, "timeout_s": timeout,
        "success_rate": len(succ) / n_trials, "found_rate": len(found) / n_trials,
        "error_mean_m": float(np.mean(errs)), "error_median_m": float(np.median(errs)),
        "error_p90_m": float(np.percentile(errs, 90)), "error_max_m": float(np.max(errs)),
        "time_mean_s": float(np.mean([t["time_s"] for t in found])) if found else None,
        "time_median_s": float(np.median([t["time_s"] for t in found])) if found else None,
        "path_mean_m": float(np.mean([t["path_length_m"] for t in found])) if found else None,
        "contacts_total": int(sum(t["contacts"] for t in trials)),
        "cast_used_in": int(sum(CAST in t["states"] for t in trials)),
        # trials that started outside the plume (SEARCH for more than 1 s before the first detection)
        "started_outside_plume": int(sum(any(s == SURGE and tt > 1.0 for tt, s in t["transitions"][1:2])
                                         for t in trials)),
        "detection_time_mean_s": float(np.mean([t["transitions"][1][0] for t in trials
                                                if len(t["transitions"]) > 1])) if trials else None,
        "time_p90_s": float(np.percentile([t["time_s"] for t in found], 90)) if found else None,
        "detours_mean": float(np.mean([t["detours"] for t in trials])),
        "wall_time_s": wall,
        "per_region": {},
        "targets": {"success_rate": 0.9, "error_mean_m": 1.0},
        "config": {k: v for k, v in cfg_overrides.items()},
    }
    for reg in ("main_hall", "east_bay", "doorway"):
        sel = [t for t in trials if t["region"] == reg]
        if sel:
            fr = [t for t in sel if t["found"]]
            stats["per_region"][reg] = {
                "trials": len(sel), "success_rate": sum(t["success"] for t in sel) / len(sel),
                "time_mean_s": float(np.mean([t["time_s"] for t in fr])) if fr else None,
                "error_mean_m": float(np.mean([t["error_m"] for t in fr])) if fr else None}
    stats["targets_met"] = bool(stats["success_rate"] >= 0.9 and stats["error_mean_m"] < 1.0)
    if not figures:
        for t in trials:
            t.pop("_traj", None)
            t.pop("_log", None)
        return stats, trials
    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(13, 9.5))
    c = np.where(base_model.fluid & (base_model.c > 0.3), base_model.c, np.nan)
    ax.imshow(c, origin="lower", extent=base_model.extent, cmap="Oranges", norm=LogNorm(0.5, 3000), alpha=0.55)
    _draw_map(ax, grid)
    show = list(range(min(8, n_trials)))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, k in enumerate(show):
        t = trials[k]
        tr = np.asarray(t["_traj"])
        ax.plot(tr[:, 0], tr[:, 1], "-", color=cols[i % 10], lw=1.4,
                label=f"#{k}: {t['state']}, err {t['error_m']:.2f} m, {t['time_s']:.0f} s" if t["error_m"] is not None
                else f"#{k}: {t['state']}")
        ax.plot(*t["start"][:2], "o", color=cols[i % 10], ms=7, mec="k")
        if t["estimate"] is not None:
            ax.plot(*t["estimate"], "x", color=cols[i % 10], ms=10, mew=2.5)
    for t in trials[len(show):]:
        if t["estimate"] is not None:
            ax.plot(*t["estimate"], "x", color="0.3", ms=6, mew=1.2)
    ax.plot(*truth, "*", color="#00c853", ms=20, mec="k", label="true leak")
    ax.add_patch(plt.Circle(truth, success_radius, fill=False, ec="#00c853", ls="--"))
    ax.set_xlim(-10.8, 18.5)
    ax.set_ylim(-9.8, 9.8)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_title(f"Gas source seeking, {n_trials} Monte-Carlo trials (gusts on, sensor lag 1 s + noise): "
                 f"success {100 * stats['success_rate']:.0f} % (error < {success_radius} m), "
                 f"mean error {stats['error_mean_m']:.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.savefig(os.path.join(IMG, "gas_seek_paths.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # timeline of the most eventful successful run (most distinct states, then fastest)
    k_show = max(range(len(trials)), key=lambda i: (trials[i]["success"], len(set(trials[i]["states"])),
                                                    -trials[i]["time_s"]))
    t0 = trials[k_show]
    lg = t0["_log"]
    if lg:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        tt = np.asarray(lg["t"])
        ax[0].semilogy(tt, np.maximum(lg["c_true"], 0.05), color="0.5", lw=1.0, label="true concentration at nose")
        ax[0].semilogy(tt, np.maximum(lg["c"], 0.05), color="#d84315", lw=0.9, label="sensor (lag 1 s + noise)")
        ax[0].axhline(cfg.found_ppm, color="#d81b60", ls="--", lw=1, label="found_ppm")
        ax[0].axhline(cfg.detect_ppm, color="#1e88e5", ls=":", lw=1, label="detect_ppm")
        ax[0].set_ylabel("ppm")
        ax[0].legend(fontsize=8, loc="lower right")
        ax[0].set_title(f"Seek trial #{k_show} from {tuple(round(v, 1) for v in t0['start'][:2])}: "
                        f"state timeline ({t0['state']}, error {t0['error_m']:.2f} m)")
        _state_bands(ax[1], tt, lg["state"])
        ax[1].set_xlabel("t [s]")
        fig.savefig(os.path.join(IMG, "gas_seek_timeline.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
    for t in trials:
        t.pop("_traj", None)
        t.pop("_log", None)
    return stats, trials


def _state_bands(ax, tt, states, order=None):
    order = order or [s for s in STATE_COLORS if s in set(states)]
    tt = np.asarray(tt)
    k0 = 0
    for k in range(1, len(states) + 1):
        if k == len(states) or states[k] != states[k0]:
            s = states[k0]
            t_end = tt[k] if k < len(tt) else tt[-1]
            ax.axvspan(tt[k0], t_end, color=STATE_COLORS.get(s, "w"), alpha=0.9, lw=0)
            k0 = k
    handles = [Rectangle((0, 0), 1, 1, color=STATE_COLORS[s]) for s in order]
    ax.legend(handles, order, ncol=len(order), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    ax.set_yticks([])


# =============================================================================
# (c) RTH scenarios
# =============================================================================
def _rth_scenario(name, grid, layout, rmap, comm, geo, mission_path, battery_cfg, battery_pct, t_max=900.0,
                  speed=0.35):
    """Robot follows ``mission_path`` (pure pursuit) until the RTH supervisor takes over."""
    bat = BatteryModel(battery_cfg)
    bat.set_level(battery_pct)
    lg = RthLogic(RthConfig(), grid, layout.dock_pose, bat, rssi_lookup=rmap.at, home_distance=geo.distance)
    pp = PurePursuit(lookahead=0.6, max_vx=speed, max_wz=0.8)
    pp.set_path(densify(mission_path, 0.1))
    x0, y0 = mission_path[0]
    yaw0 = math.atan2(mission_path[1][1] - y0, mission_path[1][0] - x0)
    robot = RobotSim(x0, y0, yaw0, max_speed=0.45)
    dock = DockDetector(layout.dock_pose[:2], float(layout["charging_station"]["dock_tolerance"]))
    log = {k: [] for k in ("t", "x", "y", "battery", "rssi", "pred", "state", "dist_home", "energy")}
    t, dt = 0.0, 0.1
    events = []
    plan_sent = False
    while t < t_max:
        t += dt
        sp = math.hypot(robot.vx_w, robot.vy_w)
        docked = dock.update(t, robot.x, robot.y, sp)
        bat.step(dt, sp, robot.wz, "sit" if lg.state in (DOCKED, CHARGING) else "walk", docked)
        rssi = comm.rssi_at(robot.x, robot.y)
        out = lg.update(RthInputs(t, robot.x, robot.y, robot.yaw, robot.vx_w, robot.vy_w, bat.percentage,
                                  bat.charging, rssi, plan=None if plan_sent else mission_path))
        plan_sent = True
        if out.locomotion:
            events.append((t, out.locomotion))
        if lg.state == MONITOR and not lg.transitions:
            vx, vy, wz, done = pp.compute(robot.x, robot.y, robot.yaw)
            cmd = (0.0, 0.0, 0.0) if done else (vx, vy, wz)
        else:
            cmd = out.cmd or (0.0, 0.0, 0.0)
        robot.step(cmd, dt, grid)
        for k, v in (("t", t), ("x", robot.x), ("y", robot.y), ("battery", bat.percentage), ("rssi", rssi),
                     ("pred", lg.predicted_min_rssi), ("state", lg.state), ("dist_home", lg.distance_home),
                     ("energy", lg.energy_to_home_pct)):
            log[k].append(v)
        if lg.transitions and lg.state == MONITOR:
            break
    tr = lg.transitions
    trig = tr[0] if tr else None
    res = {"name": name, "reason": trig[2] if trig else None, "trigger_t": trig[0] if trig else None,
           "transitions": [[round(a, 1), b, c] for a, b, c in tr],
           "locomotion_commands": [[round(a, 1), b] for a, b in events],
           "battery_at_trigger_pct": None, "battery_min_pct": float(min(log["battery"])),
           "final_state": lg.state, "replans": lg.replans, "contacts": robot.contacts,
           "min_rssi_dbm": float(min(log["rssi"])),
           "dock_error_m": float(math.hypot(robot.x - layout.dock_pose[0], robot.y - layout.dock_pose[1])),
           "completed": bool(len(tr) >= 6 and tr[-1][1] == MONITOR)}
    if trig:
        k = int(np.searchsorted(np.asarray(log["t"]), trig[0]))
        k = min(k, len(log["t"]) - 1)
        res["battery_at_trigger_pct"] = float(log["battery"][k])
        res["position_at_trigger"] = [float(log["x"][k]), float(log["y"][k])]
        res["energy_to_home_at_trigger_pct"] = float(log["energy"][k])
        res["distance_home_at_trigger_m"] = float(log["dist_home"][k])
        nav = [i for i, s in enumerate(log["state"]) if s in (RTH_NAVIGATING, DOCKING)]
        if nav:
            res["rth_path_walked_m"] = float(np.sum(np.hypot(np.diff(np.asarray(log["x"])[nav]),
                                                             np.diff(np.asarray(log["y"])[nav]))))
    return res, log


def eval_rth(grid, layout):
    comm = CommModel.from_layout(layout, grid)
    rmap = comm.compute_map(0.5)
    geo = GeodesicField(grid, layout.dock_pose[:2], 0.3)
    s1 = [z for z in layout.comm.get("shadow_zones", []) if z.get("id") == "S1"]
    s1_rect = s1[0]["rect"] if s1 else [-9.8, -8.8, -6.0, -6.3]
    results, logs = [], []

    # 1) battery drop while patrolling the hall
    patrol = plan_path(grid, (6.0, 2.5), (-1.5, 6.5), robot_radius=0.3)
    patrol = patrol + list(reversed(plan_path(grid, (6.0, -1.0), (-1.5, 6.5), robot_radius=0.3) or []))
    r, lg = _rth_scenario("battery_drop", grid, layout, rmap, comm, geo, patrol, BatteryConfig(), 21.2)
    results.append(r)
    logs.append(lg)

    # 2) mission path from the hall centre into the shielded turbine pit (zone S1)
    s1_target = (0.5 * (s1_rect[0] + s1_rect[2]), 0.5 * (s1_rect[1] + s1_rect[3]))
    mpath = plan_path(grid, (6.0, -1.0), s1_target, robot_radius=0.3)
    r, lg = _rth_scenario("comm_shadow_ahead", grid, layout, rmap, comm, geo, mpath, BatteryConfig(), 75.0)
    # where along the mission path the coverage drops below the threshold
    dense = np.asarray(densify(mpath, 0.1))
    s_path = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(dense[:, 0]), np.diff(dense[:, 1])))])
    below = np.asarray(rmap.at(dense[:, 0], dense[:, 1])) < RthConfig().rssi_min_dbm
    r["path_first_shadow_m"] = float(s_path[int(np.argmax(below))]) if below.any() else None
    xs, ys = np.asarray(lg["x"]), np.asarray(lg["y"])
    inside = ((xs >= min(s1_rect[0], s1_rect[2])) & (xs <= max(s1_rect[0], s1_rect[2]))
              & (ys >= min(s1_rect[1], s1_rect[3])) & (ys <= max(s1_rect[1], s1_rect[3])))
    r["entered_shadow_zone"] = bool(inside.any())
    if r.get("position_at_trigger"):
        px, py = r["position_at_trigger"]
        dx = max(min(s1_rect[0], s1_rect[2]) - px, 0.0, px - max(s1_rect[0], s1_rect[2]))
        dy = max(min(s1_rect[1], s1_rect[3]) - py, 0.0, py - max(s1_rect[1], s1_rect[3]))
        r["distance_to_zone_at_trigger_m"] = float(math.hypot(dx, dy))
        r["rssi_at_trigger_dbm"] = float(comm.rssi_at(px, py))
        k_robot = int(np.argmin(np.hypot(dense[:, 0] - px, dense[:, 1] - py)))
        if r["path_first_shadow_m"] is not None:
            r["path_distance_to_shadow_at_trigger_m"] = float(r["path_first_shadow_m"] - s_path[k_robot])
    results.append(r)
    logs.append(lg)

    # 3) energy reserve: aged pack (50 Wh) far from the dock, patrol in the west hall
    west = plan_path(grid, (-1.0, -1.8), (-6.5, -1.8), robot_radius=0.3) or [(-1.0, -1.8), (-6.5, -1.8)]
    loop = west + list(reversed(west))
    loop = loop * 6
    r, lg = _rth_scenario("energy_reserve", grid, layout, rmap, comm, geo, loop, BatteryConfig(capacity_wh=50.0),
                          34.0, t_max=1500.0)
    r["arrival_battery_pct"] = None
    results.append(r)
    logs.append(lg)
    dock_k = [i for i, s in enumerate(lg["state"]) if s in (DOCKED, CHARGING)]
    if dock_k:
        r["arrival_battery_pct"] = float(lg["battery"][dock_k[0]])

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    for ax, r, lg in zip(axes, results, logs):
        tt = np.asarray(lg["t"])
        st = lg["state"]
        k0 = 0
        for k in range(1, len(st) + 1):
            if k == len(st) or st[k] != st[k0]:
                ax.axvspan(tt[k0], tt[min(k, len(tt) - 1)], color=STATE_COLORS[st[k0]], alpha=0.35, lw=0)
                k0 = k
        ax.plot(tt, lg["battery"], color="#2e7d32", lw=1.8, label="battery [%]")
        ax.plot(tt, lg["energy"], color="#6d4c41", lw=1.2, ls="--", label="energy to home [%]")
        ax.axhline(20.0, color="#2e7d32", ls=":", lw=1)
        ax.set_ylabel("%")
        ax2 = ax.twinx()
        ax2.plot(tt, lg["rssi"], color="#1565c0", lw=1.0, label="RSSI at robot [dBm]")
        pred = [np.nan if p is None else p for p in lg["pred"]]
        ax2.plot(tt, pred, color="#00acc1", lw=1.0, ls="--", label="predicted min RSSI [dBm]")
        ax2.axhline(-80.0, color="#1565c0", ls=":", lw=1)
        ax2.set_ylabel("dBm")
        ax2.set_ylim(-115, -25)
        if r["trigger_t"] is not None:
            ax.axvline(r["trigger_t"], color="r", lw=1.5)
            ax.text(r["trigger_t"], ax.get_ylim()[1] * 0.92, f"  trigger: {r['reason']}", color="r", fontsize=9)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        used = [s for s in STATE_COLORS if s in set(st)]
        hs = [Rectangle((0, 0), 1, 1, color=STATE_COLORS[s], alpha=0.35) for s in used]
        ax.legend(h1 + h2 + hs, l1 + l2 + used, fontsize=7, ncol=5, loc="lower center",
                  bbox_to_anchor=(0.5, -0.42))
        ax.set_title(f"RTH scenario '{r['name']}': reason {r['reason']}, completed {r['completed']}")
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "rth_timeline.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5.2))
    im = ax.imshow(rmap.field, origin="lower", cmap="viridis", vmin=-100, vmax=-30,
                   extent=(rmap.origin_x, rmap.origin_x + rmap.shape[1] * rmap.resolution,
                           rmap.origin_y, rmap.origin_y + rmap.shape[0] * rmap.resolution))
    ax.contour(np.linspace(rmap.origin_x + 0.25, rmap.origin_x + (rmap.shape[1] - 0.5) * 0.5, rmap.shape[1]),
               np.linspace(rmap.origin_y + 0.25, rmap.origin_y + (rmap.shape[0] - 0.5) * 0.5, rmap.shape[0]),
               rmap.field, levels=[-80.0], colors="r", linewidths=1.2)
    _draw_map(ax, grid, 0.9)
    for ap in comm.cfg.access_points:
        ax.plot(ap.x, ap.y, "^", color="w", ms=10, mec="k")
        ax.text(ap.x + 0.4, ap.y + 0.4, ap.id, color="w", fontsize=8)
    for r, lg, col in zip(results, logs, ("#ff7043", "#e040fb", "#ffee58")):
        ax.plot(lg["x"], lg["y"], color=col, lw=1.6, label=f"{r['name']} ({r['reason']})")
        if r.get("position_at_trigger"):
            ax.plot(*r["position_at_trigger"], "o", color=col, ms=8, mec="k")
    ax.add_patch(Rectangle((s1_rect[0], s1_rect[1]), s1_rect[2] - s1_rect[0], s1_rect[3] - s1_rect[1],
                           fill=False, ec="r", ls="--"))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01).set_label("RSSI [dBm] (red contour: -80 dBm shadow)")
    ax.set_title("Wi-Fi coverage (log-distance + wall crossings + shielded zone S1) and RTH trajectories "
                 "(dot = trigger)")
    fig.savefig(os.path.join(IMG, "rth_scenarios.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return results, {"rssi_map_min_dbm": float(rmap.field.min()),
                     "shadow_fraction_of_map": float(np.mean(rmap.field < -80.0))}


# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--fallback-map", action="store_true", help="use the map rasterised from world_layout.yaml")
    ap.add_argument("--quick", action="store_true", help="skip the RTH scenarios")
    ap.add_argument("--no-ablation", action="store_true", help="skip the seeking run without obstacle detours")
    args = ap.parse_args(argv)
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    layout = Layout.load()
    if args.fallback_map:
        grid, map_desc = build_layout_map(layout), "layout-rasterised"
    else:
        grid, map_desc = load_nav_map("", layout)
    print(f"nav map: {map_desc}")
    cfg = GasConfig(source_xy=tuple(layout.gas_leak["source"][:2]), wind_base=tuple(layout.gas_leak["wind_base"]))
    print("(a) plume ...")
    model, plume = eval_plume(grid, layout, cfg)
    print(json.dumps(plume, indent=1))
    print(f"(b) seeking: {args.trials} trials ...")
    seek, trials = eval_seeking(model, grid, layout, args.trials, args.seed, args.timeout)
    print(json.dumps(seek, indent=1))
    ablation = None
    if not args.no_ablation:
        print(f"(b') ablation: {args.trials} trials without obstacle detours ...")
        abl, _ = eval_seeking(model, grid, layout, args.trials, args.seed, args.timeout, figures=False,
                              label="no-detour ", detour_enabled=False)
        ablation = {k: abl[k] for k in ("success_rate", "found_rate", "error_mean_m", "time_mean_s",
                                        "time_median_s", "time_p90_s", "path_mean_m", "contacts_total",
                                        "per_region", "config")}
        print(json.dumps(ablation, indent=1))
    rth, comm = ([], {}) if args.quick else eval_rth(grid, layout)
    for r in rth:
        print(f"(c) {r['name']}: reason {r['reason']} at t={r['trigger_t']}, completed {r['completed']}, "
              f"battery min {r['battery_min_pct']:.1f} %, min RSSI {r['min_rssi_dbm']:.1f} dBm")
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nav_map": os.path.relpath(map_desc, ROOT).replace(os.sep, "/") if os.path.isabs(map_desc) else map_desc,
        "gas_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()},
        "plume": plume,
        "seeking": seek,
        "seeking_ablation_no_detour": ablation,
        "seeking_trials": trials,
        "rth_scenarios": rth,
        "comm": comm,
    }
    with open(os.path.join(RES, "gas_eval.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wrote {os.path.join(RES, 'gas_eval.json')}")
    return 0 if (seek["targets_met"]) else 1


if __name__ == "__main__":
    sys.exit(main())
