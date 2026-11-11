#!/usr/bin/env python3
"""
Headless Gazebo check of the novel-obstacle avoidance (development tool).

Mission in the loop in the REAL plant world: simulation/worlds/plant.sdf
(gz-sim + DART, every collision unchanged) PLUS extra collision + visual
boxes that are NOT in simulation/maps/plant_map.yaml, the REAL ROS-free
LocomotionController (stable | mpc) at 100 Hz of sim time, the REAL
MissionLogic (plant_simulation.mission_logic) with the obstacle layer
(dusan_common.obstacle_layer), the odometry (ground truth fallback) as /odom
and the depth camera ray cast against the world collisions + the module's
own terrain_mapper pipeline producing /elevation_map/points, converted into
obstacle evidence exactly as mission_manager does
(EvidenceFrame.from_elevation_cloud).  Ground truth is used only for the
metrics (box clearance, walked path, falls).

Scenarios (tools/gazebo_check/scenarios.py MISSION_SCENARIOS):

    t  dock -> G08 (first patrol leg), 0.6 x 0.6 x 0.5 m box on the A* leg (seen during the stand-up)
    r  dock -> G08 without a box (time reference of t, no false detours)
    u  G08 -> V2 south of the pallet, box discovered at ~1.2 m              (control: z)
    v  upper corridor (3.6 m wide) -> V1, 0.6 x 0.6 x 0.5 m box on the route  (control: x)
    w  as (v) but a 0.4 x 3.6 m wall closes the corridor -> re-plans, stuck logic, skip, DONE
    x  (v) without the box, z  (u) without the box: no phantom detours, time reference
    y  dock -> G08 -> G07 (45 m: pipe gap, pallet, ramp, corridor), no box: false positives

Each run in its own process (own GZ_PARTITION).  Results (per scenario /
mode / seed: pass, reached, time, time vs. the control, walked path vs. the
control = detour length, detours, re-plans, min clearance to the box, falls)
go to ``--out`` (default <tmp>/obstacle_check/results.json) - never to
docs/results.

Usage (needs the gz python bindings, e.g. conda-forge gz-sim8 / Harmonic)::

    python tools/obstacle_check.py                                   # stable, r t u v w x z, seeds 0 1 2
    python tools/obstacle_check.py --modes stable mpc --seeds 0 1 --jobs 6
    python tools/obstacle_check.py --scenarios v x --seeds 0 --map depth   # controller on the depth map too
    python tools/obstacle_check.py --scenarios u --mission-params '{"obstacle_enabled": false}'
    python tools/obstacle_check.py --summary-only --out <results.json>
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
for _p in ("dusan_common", "module1_locomotion", "simulation", "tools"):
    _q = os.path.join(ROOT, _p)
    if _q not in sys.path:
        sys.path.insert(0, _q)

DEFAULT_SCENARIOS = ("r", "t", "u", "v", "w", "x", "z")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "obstacle_check", "results.json")
PROTECTED = os.path.normcase(os.path.abspath(os.path.join(ROOT, "docs", "results")))
SUMMARY_KEYS = ("scenario", "mode", "seed", "pass", "reason", "reached", "t_reach", "min_box_clearance", "detours",
                "side", "obstacle_replans", "stuck_replans", "path_len", "falls", "slow_frac", "missed", "wall_s")


def _clean(v):
    import numpy as np
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


# ---------------------------------------------------------------------------
# worker: one scenario / mode / seed in this process
# ---------------------------------------------------------------------------
def run_one(args) -> dict:
    from dataclasses import replace

    import numpy as np
    from gazebo_check import scenarios as S
    from gazebo_check.harness import GzHarness, HarnessConfig

    sc = S.MISSION_SCENARIOS[S.resolve_mission([args.worker])[0]]
    dur = float(args.duration or sc.duration)
    params = json.loads(args.params or "{}")
    params["locomotion_mode"] = args.mode
    cfg = HarnessConfig(terrain="plant", spawn=sc.spawn_for(int(args.seed)), params=params, map_source=args.map,
                        perception_depth=True, extra_objects=tuple(sc.extra), seed=int(args.seed),
                        out_dir=args.build_dir, verbose=args.verbose)
    t0 = time.time()
    h = GzHarness(cfg)
    op = sc.operator(json.loads(args.mission_params or "{}"), seed=int(args.seed))
    h.cmd_fn = op
    err = None
    try:
        h.run(dur)
    except RuntimeError as e:                    # harness errors end the run, the metrics so far are kept
        err = str(e)[:2000]
    A = h.arrays()
    res = op.result()
    res["confirmed_cells"] = [[round(float(a), 2), round(float(b), 2)]
                              for a, b in op.m.obstacles.memory.confirmed_xy()]
    solve = getattr(h.worker, "solve_times", None) if getattr(h, "worker", None) is not None else None
    extra = dict(res, seed=int(args.seed), mode=args.mode, map=args.map, wall_s=round(time.time() - t0, 1),
                 error=err, scenario_key=sc.key, control=sc.control, duration_run=dur,
                 mpc_solve_wall_ms_mean=(float(np.mean(solve) * 1e3) if solve else None),
                 ctrl_messages=[(round(t, 2), m) for t, m in h.messages[:40]])
    out = S.evaluate(replace(sc.scenario(), duration=dur, window=(0.0, dur)), A, extra)
    if args.npz:
        np.savez_compressed(args.npz, t=A["t"], pos=A["pos"], rpy=A["rpy"], mode=A["mode"].astype(str),
                            traj=np.asarray(op.traj), trace=np.asarray(op.trace))
    h.close()
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def worker_cmd(args, scen: str, mode: str, seed: int, npz: str, build: str) -> list:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", scen, "--mode", mode, "--seed", str(seed),
           "--build-dir", build, "--params", args.params or "{}", "--mission-params", args.mission_params or "{}"]
    if npz:
        cmd += ["--npz", npz]
    if args.map:
        cmd += ["--map", args.map]
    if args.duration:
        cmd += ["--duration", str(args.duration)]
    return cmd


def run_all(args) -> dict:
    from gazebo_check import scenarios as S
    scens = S.resolve_mission(args.scenarios)
    seeds = [int(s) for s in args.seeds]
    work = os.path.abspath(args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "work"))
    os.makedirs(work, exist_ok=True)
    jobs = [(sc, md, sd) for md in args.modes for sc in scens for sd in seeds]
    running, runs = [], []
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < max(1, args.jobs):
            sc, md, sd = jobs.pop(0)
            tag = f"{sc}_{md}_s{sd}"
            npz = os.path.join(work, tag + ".npz") if args.save_npz else ""
            env = dict(os.environ, GZ_PARTITION=f"obst_{uuid.uuid4().hex[:10]}")
            log = open(os.path.join(work, tag + ".log"), "w", encoding="utf-8")
            # stdout to a file, not a pipe read after the exit: a large RESULT line (events, plans)
            # fills the pipe buffer and the worker blocks forever on its own print
            res_path = os.path.join(work, tag + ".out")
            out_f = open(res_path, "w", encoding="utf-8")
            p = subprocess.Popen(worker_cmd(args, sc, md, sd, npz, work), stdout=out_f, stderr=log, env=env,
                                 text=True)
            running.append((p, sc, md, sd, log, out_f, res_path))
        time.sleep(0.5)
        for item in list(running):
            p, sc, md, sd, log, out_f, res_path = item
            if p.poll() is None:
                continue
            running.remove(item)
            log.close()
            out_f.close()
            res = None
            with open(res_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("RESULT "):
                        res = json.loads(line[7:])
            if res is None:
                res = {"scenario": sc, "mode": md, "seed": sd, "pass": False,
                       "reason": f"worker failed (exit {p.returncode}, see {os.path.join(work, sc)}_*.log)"}
            runs.append(res)
            print(f"[{time.time() - t0:6.0f}s] {sc:16s} {md:6s} seed {sd}: {'PASS' if res.get('pass') else 'FAIL'}"
                  f"  t_reach={res.get('t_reach')} clr={res.get('min_box_clearance')} detours={res.get('detours')}"
                  f" falls={res.get('falls')} {res.get('reason', '')}", flush=True)
    runs.sort(key=lambda r: (str(r.get("mode")), str(r.get("scenario")), int(r.get("seed", 0))))
    return {"config": {"modes": list(args.modes), "seeds": seeds, "map": args.map or "",
                       "params": json.loads(args.params or "{}"),
                       "mission_params": json.loads(args.mission_params or "{}"),
                       "duration": float(args.duration or 0.0)},
            "runs": runs, "summary": summarize(runs), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool": "tools/obstacle_check.py",
            "engine": "gz-sim 8 (Harmonic) + DART, headless, gz-transport lock-step"}


def _mean(v):
    v = [float(x) for x in v if x is not None]
    return sum(v) / len(v) if v else None


def summarize(runs) -> dict:
    """Per scenario / mode: passes, mean time, time and walked path vs. the
    control scenario of the same mode (no box) = slow-down and detour length."""
    groups = {}
    for r in runs:
        groups.setdefault((str(r.get("scenario")), str(r.get("mode"))), []).append(r)
    ref = {k: (_mean([r.get("t_reach") for r in v]), _mean([r.get("path_len") for r in v]))
           for k, v in groups.items()}
    out = {}
    for (sc, md), v in sorted(groups.items()):
        ctrl = next((r.get("control") for r in v if r.get("control")), None)
        t_ref, l_ref = ref.get((ctrl, md), (None, None)) if ctrl else (None, None)
        t = _mean([r.get("t_reach") for r in v])
        L = _mean([r.get("path_len") for r in v])
        clr = [r.get("min_box_clearance") for r in v if r.get("min_box_clearance") is not None]
        out[f"{sc}|{md}"] = {
            "passed": sum(1 for r in v if r.get("pass")), "runs": len(v),
            "t_reach_mean": None if t is None else round(t, 2),
            "time_vs_control": None if (t is None or not t_ref) else round(t / t_ref, 3),
            "path_len_mean": None if L is None else round(L, 2),
            "extra_path_m": None if (L is None or l_ref is None) else round(L - l_ref, 2),
            "min_box_clearance": round(min(clr), 3) if clr else None,
            "detours": [r.get("detours") for r in v], "obstacle_replans": [r.get("obstacle_replans") for r in v],
            "stuck_replans": [r.get("stuck_replans") for r in v], "falls": [r.get("falls") for r in v],
            "missed": sorted({m for r in v for m in (r.get("missed") or [])}),
            "failures": [f"seed {r.get('seed')}: {r.get('reason')}" for r in v if not r.get("pass")]}
    return out


def print_summary(summary: dict) -> None:
    hdr = ("scenario|mode", "pass", "t_reach", "x_ctrl", "path", "+path", "min_clr", "detours", "obs_repl", "stuck",
           "falls", "missed")
    print(" | ".join(hdr))
    for k, s in summary.items():
        print(" | ".join(str(x) for x in (k, f"{s['passed']}/{s['runs']}", s["t_reach_mean"], s["time_vs_control"],
                                          s["path_len_mean"], s["extra_path_m"], s["min_box_clearance"],
                                          s["detours"], s["obstacle_replans"], s["stuck_replans"], s["falls"],
                                          s["missed"])))
        for f in s["failures"]:
            print("     FAIL " + f)


def check_out_path(path: str) -> str:
    """The results never go to docs/results (the published gazebo_check.json lives there)."""
    p = os.path.normcase(os.path.abspath(path))
    if p == PROTECTED or p.startswith(PROTECTED + os.sep):
        raise SystemExit(f"refusing to write {path}: docs/results is not a scratch directory")
    return os.path.abspath(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    ap.add_argument("--modes", nargs="+", default=["stable"], choices=("stable", "mpc"),
                    help="locomotion_mode of the controller")
    ap.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--map", default="", choices=("", "truth", "depth"),
                    help="controller elevation map ('' blind = the default runtime, depth = the full stack); the "
                         "obstacle layer always gets the depth camera + terrain_mapper cloud")
    ap.add_argument("--params", default="{}", help="JSON controller parameter overrides")
    ap.add_argument("--mission-params", default="{}", help="JSON MissionLogic parameter overrides (obstacle_*)")
    ap.add_argument("--duration", type=float, default=0.0, help="cap the scenario duration [s]")
    ap.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--out", default=DEFAULT_OUT, help="results JSON (never docs/results)")
    ap.add_argument("--work-dir", default="", help="log / npz / world directory (default: next to --out)")
    ap.add_argument("--save-npz", action="store_true", help="also save each run's time series (.npz)")
    ap.add_argument("--summary-only", action="store_true", help="print the table of an existing --out")
    ap.add_argument("--verbose", action="store_true")
    # worker mode (internal)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--mode", default="stable", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--npz", default="", help=argparse.SUPPRESS)
    ap.add_argument("--build-dir", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        print("RESULT " + json.dumps(_clean(run_one(args))), flush=True)
        return 0
    out = check_out_path(args.out)
    if args.summary_only:
        with open(out, "r", encoding="utf-8") as f:
            print_summary(json.load(f)["summary"])
        return 0
    block = run_all(args)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_clean(block), f, indent=1)
        f.write("\n")
    print_summary(block["summary"])
    print(f"wrote {out}")
    return 0 if all(s["passed"] == s["runs"] for s in block["summary"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
