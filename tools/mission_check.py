#!/usr/bin/env python3
"""
Whole demo mission in headless Gazebo (development tool).

The robot walks the complete demo mission of simulation/launch/bringup.launch.py
in the REAL plant world (simulation/worlds/plant.sdf, gz-sim + DART, the robot
with its own gz plugins, gz-transport lock-step, tools/gazebo_check/): spawn on
the charging dock, 't' (= ``start``) at ~3.5 s, patrol over the patrol planner's
viewpoints with the gas source search when the alarm goes off on the way, return
home and docking.

In the loop (all REAL code, ROS free, :mod:`gazebo_check.mission_stack`):
LocomotionController (stable gait, elevation map from the emulated depth camera
-> terrain_mapper pipeline), mission_manager (MissionLogic: path planning with
the terrain cost layer, path following, obstacle layer), patrol_planner,
gas dispersion + source seeker, battery, comm, return_to_home, cmd_vel_mux.
The gauge reader is stubbed from the ground truth (visibility at the true pose,
motion gate), the thermal pipeline is not run.  Falls are recovered by the
controller itself (``assisted_reset`` false: no teleport).

A run is COMPLETE when the mission ends DOCKED with every gauge read, the gas
source FOUND and no teleport.  Per run the JSON holds completion, the gauge
results, the gas result, every fall (time, position, area / nearest debris,
mission state, gait / gait phase, speed, command, heading change in the 2 s
before, terrain class, time to get up), recoveries, teleports, stalls (stuck
events of the mission and windows without progress while driving), sim / wall
time and the mission timeline.

Usage (needs the gz python bindings, e.g. conda-forge gz-sim8 / Harmonic)::

    python tools/mission_check.py --seeds 0 1 2 3 4 --jobs 5 --out <file>.json
    python tools/mission_check.py --seeds 0 --leg 9 --out <file>.json    # leg 9 of the plan only (from viewpoint 8)
    python tools/mission_check.py --seeds 0 1 --leg 2 --legs 3 --out <file>.json
    python tools/mission_check.py --seeds 0 1 2 3 4 --save               # docs/results/mission_check.json
    python tools/mission_check.py --seeds 0 1 --spawn 23.6,0.45,3.14 --goal 20.0,0.25 --out <file>.json
    python tools/mission_check.py --summary <file>.json                  # print a saved result

``--leg K`` starts at the viewpoint of leg K - 1 (leg 1 = dock -> first viewpoint;
the leg after the last viewpoint = the way home, started with ``rth``), ``--legs N``
stops after N legs.  ``--spawn x,y,yaw --goal x,y[,yaw] ...`` runs a route segment
instead: spawn there, a patrol (``patrol``) over the goal points, done when the last
one is reached (segment runs pass with every goal reached and no fall).  ``--save-npz`` keeps the per-step logs (100 Hz robot, 10 Hz
mission trace) of every run in ``--work-dir``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in ("dusan_common", "module1_locomotion", "module2_inspection", "module3_gas_safety", "simulation", "tools"):
    _q = os.path.join(ROOT, _p)
    if _q not in sys.path:
        sys.path.insert(0, _q)

import numpy as np  # noqa: E402

RESULTS = os.path.join(ROOT, "docs", "results", "mission_check.json")
DEFAULT_DURATION = 1500.0     # [s] sim time cap of a run
T_START = 3.5                 # [s] 't' (start) after the spawn (+ CMD_SHIFT per seed)
STALL_WINDOW = 15.0           # [s] driving without STALL_DIST progress = stall
STALL_DIST = 0.25             # [m]
DRIVING = {("PATROL", "navigate"), ("PATROL", "backup"), ("RTH", "self"), ("RTH", "yield"), ("SEEK", "")}
NEAR_DEBRIS = 1.0             # [m] nearest debris reported within this distance
TILT_ONSET_DEG = 12.0         # fall onset: last time |roll|, |pitch| below this before the fall
DOCKED_STOP = 30.0            # [s] a run ends this long after docking even when the mission is not finished


# ---------------------------------------------------------------------------
# where (pure python)
# ---------------------------------------------------------------------------
def _layout():
    from dusan_common.layout import Layout
    return Layout.load(os.path.join(ROOT, "simulation", "config", "world_layout.yaml"))


def area_of(x: float, y: float, layout) -> str:
    """Route area of a plant world position (world_layout.yaml geometry)."""
    dock = layout.dock_pose
    if math.hypot(x - dock[0], y - dock[1]) <= 1.2:
        return "dock"
    cor = layout["corridor"]
    fac = layout["factory"]
    door = cor.get("door_frame", {})
    part = fac.get("partition", {})
    if float(door.get("x_min", 15.15)) - 0.4 <= x <= float(door.get("x_max", 15.5)) + 0.4 and abs(y) <= 2.0:
        return "corridor door"
    if part and float(part["x_min"]) - 0.5 <= x <= float(part["x_max"]) + 0.5 and abs(y) <= 2.0:
        return "partition door"
    if x >= float(cor["interior"]["x_min"]):
        desc = cor.get("descent", {})
        if float(desc.get("x_min", 46.97)) <= x <= float(desc.get("x_max", 51.22)):
            return "ramp" if y > -0.165 else "stairs"
        lower = cor.get("lower_floor", {})
        return "lower corridor" if x > float(lower.get("x_min", 51.22)) else "upper corridor"
    if part and x > float(part["x_max"]):
        return "back room"
    return "factory"


def nearest_debris(x: float, y: float, layout, within: float = NEAR_DEBRIS):
    """(id, type, distance) of the nearest layout obstacle footprint within ``within`` m, else None."""
    from dusan_common.terrain_cost import footprint_distance
    best = None
    for ob in layout.obstacles:
        d = footprint_distance(ob, x, y)
        if d is None:
            if ob.get("type") in ("pallet", "drum") and "center" in ob:
                r = 0.75 if ob.get("type") == "pallet" else float(ob.get("radius", 0.3))
                d = max(0.0, math.hypot(x - float(ob["center"][0]), y - float(ob["center"][1])) - r)
            else:
                continue
        d = float(d)
        if d <= within and (best is None or d < best[2]):
            best = (str(ob.get("id")), str(ob.get("type")), round(d, 2))
    return best


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# analysis of one run (pure python)
# ---------------------------------------------------------------------------
def trace_at(trace, t: float):
    """Latest 10 Hz mission trace row at or before t (None before the first)."""
    row = None
    for r in trace:
        if r[0] > t + 1e-9:
            break
        row = r
    return row


def analyse_falls(A: dict, trace, rec_events, layout) -> list:
    """One entry per controller-detected fall: where / when / what the robot did."""
    t = np.asarray(A["t"], dtype=float)
    if not len(t):
        return []
    pos = np.asarray(A["pos"], dtype=float)
    rpy = np.asarray(A["rpy"], dtype=float)
    vel = np.asarray(A["vel"], dtype=float)
    cmd = np.asarray(A["cmd"], dtype=float)
    modes = [str(m) for m in A["mode"]]
    gait = [str(g) for g in A["gait"]]
    gph = np.asarray(A.get("gait_phase", np.full(len(t), np.nan)), dtype=float)
    tcls = [str(c) for c in A.get("terrain_class", [""] * len(t))]
    ground = np.asarray(A["ground"], dtype=float)
    falls = [e for e in rec_events if e["event"] == "fall"]
    recs = [e for e in rec_events if e["event"] == "recovered"]
    out = []
    for e in falls:
        tf = float(e["t"])
        i = min(int(np.searchsorted(t, tf)), len(t) - 1)
        # onset: last sample before the fall with a level body
        tilt = np.degrees(np.max(np.abs(rpy[:i + 1, :2]), axis=1))
        lvl = np.flatnonzero(tilt < TILT_ONSET_DEG)
        k = int(lvl[-1]) if len(lvl) else i
        k0 = max(0, int(np.searchsorted(t, t[k] - 2.0)))
        k1 = max(0, int(np.searchsorted(t, t[k] - 1.0)))
        x, y = float(pos[k, 0]), float(pos[k, 1])
        deb = nearest_debris(x, y, layout)
        row = trace_at(trace, float(t[k]))
        rec = next((r for r in recs if r["t"] > tf), None)
        sp = np.hypot(vel[k1:k + 1, 0], vel[k1:k + 1, 1])
        out.append({
            "t": round(tf, 2), "t_onset": round(float(t[k]), 2),
            "x": round(x, 2), "y": round(y, 2), "yaw_deg": round(math.degrees(float(rpy[k, 2])), 1),
            "fall_x": round(float(pos[i, 0]), 2), "fall_y": round(float(pos[i, 1]), 2),
            "rpy_deg": [round(math.degrees(float(v)), 1) for v in rpy[i]],
            "area": area_of(x, y, layout),
            "debris": None if deb is None else {"id": deb[0], "type": deb[1], "dist": deb[2]},
            "mission": None if row is None else f"{row[7]}/{row[8]}",
            "mux": None if row is None else row[9],
            "mode_before": modes[k1], "gait": gait[k], "gait_phase": round(float(gph[k]), 2)
            if math.isfinite(float(gph[k])) else None,
            "speed": round(float(np.mean(sp)) if len(sp) else 0.0, 3),
            "cmd": [round(float(v), 2) for v in cmd[k1]],
            "wz_cmd_max": round(float(np.max(np.abs(cmd[k0:k + 1, 2]))) if k >= k0 else 0.0, 2),
            "heading_change_deg": round(math.degrees(_wrap(float(rpy[k, 2] - rpy[k0, 2]))), 1),
            "terrain_class": tcls[k1],
            "height": round(float(pos[k, 2] - ground[k]), 3),
            "t_recovered": None if rec is None else round(float(rec["t"]), 2),
            "time_to_recover": None if rec is None else round(float(rec["t"]) - tf, 2),
        })
    return out


def analyse_stalls(trace, layout, window: float = STALL_WINDOW, dist: float = STALL_DIST) -> list:
    """Windows of >= ``window`` s driving (mission navigating, robot standing / walking)
    with less than ``dist`` m progress, merged; one entry per stall."""
    rows = [r for r in trace if (r[7], r[8]) in DRIVING or r[7] == "SEEK"]
    if not rows:
        return []
    T = np.array([r[0] for r in rows])
    P = np.array([[r[1], r[2]] for r in rows])
    ok = np.array([str(r[13]) in ("WALK", "STAND") for r in rows])
    stalls, cur = [], None
    j = 0
    for i in range(len(T)):
        while j < i and T[i] - T[j] > window:
            j += 1
        span = T[i] - T[j]
        stuck = (span >= window - 0.2 and ok[j:i + 1].all() and
                 np.max(np.hypot(*(P[j:i + 1] - P[i]).T)) < dist and
                 np.all(np.diff(T[j:i + 1]) < 0.5))
        if stuck:
            if cur is not None and T[j] <= cur["t1"] + 0.5:
                cur["t1"] = float(T[i])
            else:
                cur = {"t0": float(T[j]), "t1": float(T[i]), "x": float(P[i, 0]), "y": float(P[i, 1]),
                       "mission": f"{rows[i][7]}/{rows[i][8]}"}
                stalls.append(cur)
    for s in stalls:
        s["duration"] = round(s["t1"] - s["t0"], 1)
        s["t0"], s["t1"] = round(s["t0"], 1), round(s["t1"], 1)
        s["x"], s["y"] = round(s["x"], 2), round(s["y"], 2)
        s["area"] = area_of(s["x"], s["y"], layout)
        deb = nearest_debris(s["x"], s["y"], layout)
        s["debris"] = None if deb is None else deb[0]
    return stalls


def completion(res: dict, n_gauges: int) -> dict:
    """COMPLETE = DOCKED, every gauge read, gas source FOUND, no teleport."""
    seek = (res.get("seek_result") or {}).get("result")
    checks = {"docked": res.get("final_state") == "DOCKED",
              "gauges": len(res.get("completed") or []) == n_gauges and not res.get("missed"),
              "gas_found": seek == "FOUND", "no_teleport": int(res.get("teleports") or 0) == 0}
    return {"complete": all(checks.values()), "checks": checks}


def summarize(runs: list) -> dict:
    n = len(runs)
    done = [r for r in runs if r.get("complete")]
    falls = [int(r.get("falls") or 0) for r in runs]
    return {"runs": n, "complete": len(done),
            "falls_total": int(sum(falls)), "falls_mean": (float(np.mean(falls)) if falls else None),
            "teleports": int(sum(int(r.get("teleports") or 0) for r in runs)),
            "stalls": int(sum(len(r.get("stalls") or []) for r in runs)),
            "recovered": int(sum(int(r.get("recoveries") or 0) for r in runs)),
            "gauges_read": [len(r.get("completed") or []) for r in runs],
            "gas": [(r.get("seek_result") or {}).get("result") for r in runs],
            "mission_time_mean": (float(np.mean([r["mission_time"] for r in done if r.get("mission_time")]))
                                  if any(r.get("mission_time") for r in done) else None),
            "errors": int(sum(1 for r in runs if r.get("error")))}


# ---------------------------------------------------------------------------
# worker: one seed in this process
# ---------------------------------------------------------------------------
def _sub_plan(pj: dict, k: int, n: int):
    """Plan from leg ``k`` (1-based; leg 1 = dock -> first viewpoint), ``n`` legs (0 = all).
    Returns (plan, spawn (x, y, yaw) or None = dock, start command, stop after viewpoint id or None)."""
    order, vps = list(pj["order"]), pj["viewpoints"]
    if k <= 1:
        start_xy, sub = None, pj
        rest = order
    elif k <= len(order):
        prev = next(v for v in vps if v["gauge_id"] == order[k - 2])
        start_xy = (float(prev["x"]), float(prev["y"]), float(prev["yaw"]))
        rest = order[k - 1:]
        sub = dict(pj)
        sub["order"] = rest
        sub["viewpoints"] = [v for v in vps if v["gauge_id"] in rest]
        sub["legs"] = list(pj.get("legs") or [])[k - 1:]
    else:                                     # the way home
        last = next(v for v in vps if v["gauge_id"] == order[-1])
        return pj, (float(last["x"]), float(last["y"]), float(last["yaw"])), "rth", None
    stop = None
    if n and n <= len(rest):
        stop = rest[n - 1]
    return sub, start_xy, "start", stop


def _segment_plan(goals, start=(0.0, 0.0)) -> dict:
    """Patrol plan of a route segment: goal points P1, P2 ... (yaw: given, else the
    heading from the previous point)."""
    vps = []
    px, py = float(start[0]), float(start[1])
    for i, g in enumerate(goals):
        v = [float(q) for q in str(g).split(",")]
        yaw = v[2] if len(v) > 2 else math.atan2(v[1] - py, v[0] - px)
        vps.append({"gauge_id": f"P{i + 1}", "x": v[0], "y": v[1], "yaw": yaw})
        px, py = v[0], v[1]
    return {"viewpoints": vps, "order": [v["gauge_id"] for v in vps], "legs": []}


def run_one(a) -> dict:
    from gazebo_check import scenarios as SC
    from gazebo_check.harness import GzHarness, HarnessConfig
    from gazebo_check.mission_stack import MissionStack, MissionStackOperator
    t0w = time.time()
    seed = int(a.seed)
    S = MissionStack(seed=seed, mission_over=json.loads(a.mission_params or "{}"),
                     rth_over=json.loads(a.rth_params or "{}"), seeker_over=json.loads(a.seeker_params or "{}"),
                     patrol_over=json.loads(a.patrol_params or "{}"))
    L = S.layout
    full_order = list(S.plan_json["order"])
    sx, sy, sz, syaw = L.spawn
    if a.spawn:
        sp = [float(q) for q in a.spawn.split(",")]
        sub = _segment_plan(a.goal or [], sp[:2])
        start_xy, start_cmd, stop_vp = (sp[0], sp[1], sp[2] if len(sp) > 2 else 0.0), "patrol", sub["order"][-1]
        full_order = list(sub["order"])
    else:
        sub, start_xy, start_cmd, stop_vp = _sub_plan(S.plan_json, int(a.leg or 1), int(a.legs or 0))
    if sub is not S.plan_json:
        S.m.set_plan(sub)
        S.plan_json = sub
    if start_xy is not None:
        sx, sy, syaw = start_xy
        sz = float(L.floor_z(sx, sy)) + SC.SPAWN_DZ
    if seed:
        rng = np.random.default_rng(1000 + seed)
        dx, dy, dyaw = rng.uniform(-1.0, 1.0, 3) * SC.SPAWN_JITTER
        sx, sy, syaw = sx + dx, sy + dy, syaw + dyaw
    params = json.loads(a.params or "{}")
    params.setdefault("locomotion_mode", "stable")
    cfg = HarnessConfig(terrain="plant", spawn=(sx, sy, sz, syaw), params=params, map_source=a.map,
                        perception_depth=True, seed=seed, out_dir=a.build_dir, verbose=False, lite_log=True)
    h = GzHarness(cfg)
    op = MissionStackOperator(S, t_start=T_START + SC.CMD_SHIFT * seed, start_cmd=start_cmd,
                              phase=(0.037 * seed) % 0.1)
    h.cmd_fn = op
    err = None
    t_setup = time.time() - t0w
    stop_reason = "duration"
    t_docked = None
    try:
        while h.t < a.duration - 1e-6:
            h.run(min(10.0, a.duration - h.t))
            m = S.m
            t_docked = (t_docked if t_docked is not None else h.t) if m.base == "DOCKED" else None
            if a.progress:
                tr = op.trace[-1] if op.trace else None
                print(f"PROGRESS t={h.t:.0f} wall={time.time() - t0w:.0f} {m.base}/{m.phase} "
                      f"pos={tr[1:3] if tr else None} done={len(m.completed)} falls={h.ctrl.recovery.fall_count} "
                      f"mode={tr[13] if tr else None}", file=sys.stderr, flush=True)
            if (S.finished() or m.base == "DONE") and h.t > 10.0:
                h.run(3.0)
                stop_reason = "finished"
                break
            if t_docked is not None and h.t - t_docked >= DOCKED_STOP:
                stop_reason = "docked (mission not finished)"    # e.g. sent home by the battery / comm RTH
                break
            if stop_vp is not None and (stop_vp in m.completed or stop_vp in m.missed):
                stop_reason = f"leg end ({stop_vp})"
                break
            if a.stop_down_s and _down_for(h, a.stop_down_s):
                err = f"stopped: not standing for > {a.stop_down_s:.0f} s"
                stop_reason = "down"
                break
    except RuntimeError as e:
        err = str(e)[:2000]
        stop_reason = "error"
    A = h.arrays()
    rec = h.ctrl.recovery
    rec_events = list(rec.events)
    m = S.m
    falls = analyse_falls(A, op.trace, rec_events, L)
    stalls = analyse_stalls(op.trace, L)
    res = {
        "seed": seed, "leg": int(a.leg or 1), "legs": int(a.legs or 0), "plan_order": full_order,
        "spawn": [round(float(v), 3) for v in (sx, sy, sz, syaw)],
        "params": params, "mission_params": json.loads(a.mission_params or "{}"),
        "final_state": m.base, "final_phase": m.phase, "stop_reason": stop_reason, "error": err,
        "sim_time": round(h.t, 2), "wall_time": round(time.time() - t0w, 1), "setup_time": round(t_setup, 1),
        "mission_t0": m.mission_t0, "mission_t_end": m.mission_t_end,
        "mission_time": (round(m.mission_t_end - m.mission_t0, 1)
                         if m.mission_t_end is not None and m.mission_t0 is not None else None),
        "completed": list(m.completed), "missed": list(m.missed), "results": m.results,
        "seek_result": m.seek_result,
        "falls": len(falls), "fall_info": falls,
        "recoveries": int(rec.successes), "recovery_attempts": int(rec.attempts_total),
        "teleports": int(rec.assisted_resets),
        "recovery_events": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}
                            for e in rec_events[:80]],
        "stalls": stalls, "stuck_events": [e for e in S.events if "stuck" in e[2]][:40],
        "path_len": round(op.path_len, 1), "battery_end": S.bat_pub[0], "max_ppm": round(S.max_ppm, 1),
        "timeline": S.state_log, "rth_transitions": S.rth.transitions,
        "events": [e for e in S.events if e[1] != "loco_cmd"][:300],
        "gauge_log": S.gauge_log, "seek_track": S.seek_log[::10], "frames": op.frames,
        "wait_timeouts": h.wait_timeouts,
    }
    if a.spawn:
        arrived = [g for g in full_order if g in m.completed or
                   (m.results.get(g) or {}).get("reason") == "no reading"]
        res["segment"] = {"goals": list(a.goal or []), "reached": arrived,
                          "ok": len(arrived) == len(full_order) and not falls}
        res["complete"] = None
        res["checks"] = None
    elif not (int(a.leg or 1) <= 1 and not a.legs):
        res["complete"] = None                       # partial runs: no mission verdict
        res["checks"] = None
    else:
        res.update(completion(res, len(full_order)))
    if a.npz:
        keep = ("t", "pos", "rpy", "vel", "mode", "gait", "gait_phase", "cmd", "cmd_f", "contacts", "feet",
                "rec_state", "terrain_class", "ground")
        arrs = {k: (A[k].astype(str) if A[k].dtype == object else A[k]) for k in keep if k in A}
        arrs["trace"] = np.array([tuple(str(v) for v in r) for r in op.trace], dtype=str)
        np.savez_compressed(a.npz, **arrs)
    h.close()
    return res


def _down_for(h, dur: float) -> bool:
    modes = h.log.get("mode") or []
    n = int(dur / 0.01)
    return len(modes) > n and not any(m in ("STAND", "WALK") for m in modes[-n:])


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 4) if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


def _tag(a, seed: int) -> str:
    leg = "full" if (int(a.leg or 1) <= 1 and not a.legs) else f"leg{a.leg}" + (f"x{a.legs}" if a.legs else "")
    if a.spawn:
        leg = "seg"
    return f"{a.label}_{leg}_s{seed}"


def run_all(a) -> dict:
    work = os.path.abspath(a.work_dir or os.path.join(tempfile.gettempdir(), "mission_check"))
    os.makedirs(work, exist_ok=True)
    jobs = [int(s) for s in a.seeds]
    running, runs, t0 = [], {}, time.time()
    while jobs or running:
        while jobs and len(running) < max(1, a.jobs):
            sd = jobs.pop(0)
            tag = _tag(a, sd)
            res_path = os.path.join(work, tag + ".json")
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--seed", str(sd), "--duration",
                   str(a.duration), "--build-dir", work, "--params", a.params, "--mission-params", a.mission_params,
                   "--rth-params", a.rth_params, "--seeker-params", a.seeker_params, "--patrol-params",
                   a.patrol_params, "--map", a.map, "--stop-down-s", str(a.stop_down_s), "--result", res_path,
                   "--leg", str(a.leg or 1), "--legs", str(a.legs or 0), "--progress"]
            if a.spawn:
                cmd += [f"--spawn={a.spawn}"] + [f"--goal={g}" for g in (a.goal or [])]
            if a.save_npz:
                cmd += ["--npz", os.path.join(work, tag + ".npz")]
            env = dict(os.environ, GZ_PARTITION=f"mission_{uuid.uuid4().hex[:10]}")
            log = open(os.path.join(work, tag + ".log"), "w", encoding="utf-8")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((p, sd, tag, log, res_path))
        time.sleep(2.0)
        for item in list(running):
            p, sd, tag, log, res_path = item
            if p.poll() is None:
                continue
            running.remove(item)
            log.close()
            if os.path.exists(res_path):
                with open(res_path, "r", encoding="utf-8") as f:
                    r = json.load(f)
            else:
                r = {"seed": sd, "error": f"worker failed (exit {p.returncode})", "complete": False, "falls": 0}
            runs[str(sd)] = r
            print(f"[{time.time() - t0:6.0f}s] {tag}: " + one_line(r), flush=True)
    rs = [runs[k] for k in sorted(runs, key=int)]
    return {"config": {"seeds": [int(s) for s in a.seeds], "leg": int(a.leg or 1), "legs": int(a.legs or 0),
                       "spawn": a.spawn or None, "goals": list(a.goal or []) or None,
                       "params": json.loads(a.params or "{}"), "mission_params": json.loads(a.mission_params or "{}"),
                       "map": a.map, "duration": a.duration},
            "summary": summarize(rs), "runs": runs, "created": time.strftime("%Y-%m-%d %H:%M:%S")}


def one_line(r: dict) -> str:
    seg = r.get("segment")
    verdict = ("COMPLETE" if r.get("complete") else "incomplete" if r.get("complete") is False else
               ("segment OK" if seg and seg.get("ok") else "segment FAIL" if seg else "partial"))
    fl = "; ".join(f"t={f['t']:.0f} ({f['x']:.1f},{f['y']:.1f}) {f['area']}"
                   + (f" {f['debris']['id']}@{f['debris']['dist']}" if f.get("debris") else "")
                   + f" {f.get('mission')}" for f in (r.get("fall_info") or []))
    st = "; ".join(f"{s['t0']:.0f}-{s['t1']:.0f}s ({s['x']:.1f},{s['y']:.1f}) {s['area']}"
                   for s in (r.get("stalls") or []))
    return (f"{verdict} "
            f"state={r.get('final_state')} sim={r.get('sim_time')} wall={r.get('wall_time')} "
            f"gauges={len(r.get('completed') or [])} missed={r.get('missed')} "
            f"gas={(r.get('seek_result') or {}).get('result')} falls={r.get('falls')} rec={r.get('recoveries')} "
            f"teleports={r.get('teleports')} stalls={len(r.get('stalls') or [])}"
            + (f" | falls: {fl}" if fl else "") + (f" | stalls: {st}" if st else "")
            + (f" | error: {str(r.get('error'))[:120]}" if r.get("error") else ""))


def print_summary(block: dict) -> None:
    s = block.get("summary", {})
    print(f"complete {s.get('complete')}/{s.get('runs')}  falls {s.get('falls_total')} "
          f"(mean {s.get('falls_mean')})  teleports {s.get('teleports')}  stalls {s.get('stalls')}  "
          f"gauges {s.get('gauges_read')}  gas {s.get('gas')}  mission time {s.get('mission_time_mean')}")
    for k in sorted(block.get("runs", {}), key=int):
        print(f"  seed {k}: " + one_line(block["runs"][k]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    ap.add_argument("--jobs", type=int, default=max(1, min(5, (os.cpu_count() or 2) // 3)))
    ap.add_argument("--label", default="latest", help="result block in the JSON")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="sim time cap of a run [s]")
    ap.add_argument("--leg", type=int, default=1, help="start at leg K of the patrol plan (1 = from the dock)")
    ap.add_argument("--legs", type=int, default=0, help="stop after N legs (0 = the whole mission)")
    ap.add_argument("--spawn", default="", help="route segment: spawn x,y,yaw (with --goal)")
    ap.add_argument("--goal", action="append", default=None, help="route segment goal x,y[,yaw] (repeatable)")
    ap.add_argument("--params", default="{}", help="JSON locomotion controller parameter overrides")
    ap.add_argument("--mission-params", default="{}", help="JSON mission_manager parameter overrides")
    ap.add_argument("--rth-params", default="{}")
    ap.add_argument("--seeker-params", default="{}")
    ap.add_argument("--patrol-params", default="{}")
    ap.add_argument("--map", default="depth", choices=("", "truth", "depth"), help="controller elevation map")
    ap.add_argument("--stop-down-s", type=float, default=240.0,
                    help="stop a run whose robot has not stood for this long [s] (0 = never)")
    ap.add_argument("--out", default="", help="result JSON (default: <work-dir>/mission_check.json)")
    ap.add_argument("--save", action="store_true", help=f"write {os.path.relpath(RESULTS, ROOT)}")
    ap.add_argument("--work-dir", default="", help="per-run logs / results (default: <tmp>/mission_check)")
    ap.add_argument("--save-npz", action="store_true", help="keep the per-step logs of every run in --work-dir")
    ap.add_argument("--summary", default="", help="print the summary of a saved result JSON and exit")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--build-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--npz", default="", help=argparse.SUPPRESS)
    ap.add_argument("--result", default="", help=argparse.SUPPRESS)
    ap.add_argument("--progress", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.worker:
        res = run_one(a)
        with open(a.result, "w", encoding="utf-8", newline="\n") as f:
            json.dump(_clean(res), f, indent=1, default=str)
        return 0
    if a.summary:
        with open(a.summary, "r", encoding="utf-8") as f:
            data = json.load(f)
        for label, block in (data.get("runs") or {}).items():
            print(f"{label} ({block.get('created')}):")
            print_summary(block)
        return 0
    out = RESULTS if a.save else (a.out or os.path.join(os.path.abspath(a.work_dir or os.path.join(
        tempfile.gettempdir(), "mission_check")), "mission_check.json"))
    block = run_all(a)
    data = {}
    if os.path.exists(out):
        try:
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
    data.setdefault("runs", {})[a.label] = _clean(block)
    data["tool"] = "tools/mission_check.py"
    data["engine"] = "gz-sim 8 (Harmonic) + DART, headless, gz-transport lock-step"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print(f"wrote {out}")
    print_summary(data["runs"][a.label])
    return 0


if __name__ == "__main__":
    sys.exit(main())
