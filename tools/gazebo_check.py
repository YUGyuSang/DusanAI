#!/usr/bin/env python3
"""
Headless Gazebo (gz-sim + DART) check of the locomotion stack (development tool).

Mirrors the Ubuntu runtime: world physics / systems of simulation/worlds/
plant.sdf, the robot of simulation/urdf/spot.urdf(.xacro) with its own gz
plugins (JointPositionController x 12, JointStatePublisher,
OdometryPublisher, IMU), gz-transport in place of ros_gz_bridge, and the REAL
ROS-free LocomotionController at 100 Hz of sim time (tools/gazebo_check/).
Runs the scenarios (a) stand, (b) trot, (c) crawl, (d) turn, (e) teleop,
(f) 0.08 m pipe, (g) / (h) 11.8 deg ramp up / down, (i) pushes, and in the
REAL plant world (simulation/worlds/plant.sdf, spawn = world_layout.yaml
robot_spawn): (j) dock -> over the 0.10 m pipe -> pallet -> ramp with a
keyboard-teleop operator, (o) the same around the pipe, (k) locomotion node
started 15 s after the spawn, (l) the whole corridor to the hall door, (m) =
(j) at full teleop speed, and single obstacles approached straight: (n) the
0.10 m pipe, (p) / (r) the rubble field (through / along the wall), (q) the
0.08 m pipe, (s) the mu 0.2 steel plate; over several seeds, each in its own
process (own GZ_PARTITION), and writes

    docs/results/gazebo_check.json     per scenario / seed metrics + pass/fail,
                                       one block per --label (e.g. stable / mpc)
    docs/images/gazebo_*.png           timelines, stick figures, summary (--figures)

Usage (needs the gz python bindings, e.g. conda-forge gz-sim8 / Harmonic)::

    python tools/gazebo_check.py --mode stable --label stable --scenarios a b c d e i
    python tools/gazebo_check.py --mode mpc --label mpc --jobs 6   # fewer jobs: the MPC solve time is its delay
    python tools/gazebo_check.py --mode stable --label stable_map --map --scenarios f g h
    python tools/gazebo_check.py --mode stable --label plant_depth --map depth --scenarios j k m o
    python tools/gazebo_check.py --scenarios a b --seeds 0 --no-save
    python tools/gazebo_check.py --latency 0.02 --drop-ticks 0.3 --label slow_machine
    python tools/gazebo_check.py --params '{"stable_step_height": 0.09}' --label what_if
    python tools/gazebo_check.py --figures-only           # re-plot from the saved runs

Options: --mode (stable | mpc = locomotion_mode), --latency (s, command
delay tick -> cmd_pos, default 0.01 = one control period), --mpc-latency
(measured | seconds), --rtf (real-time factor of the emulated machine: MPC
result delay = rtf x wall solve time), --drop-ticks (probability of a
skipped control tick), --map (truth: ground-truth elevation map; depth: the
depth camera ray-cast against the world collisions + the module's own
terrain_mapper pipeline and /terrain/info; no --map: blind), --push-scale,
--jobs (parallel runs).
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
for _p in ("dusan_common", "module1_locomotion", "tools"):
    _q = os.path.join(ROOT, _p)
    if _q not in sys.path:
        sys.path.insert(0, _q)

import numpy as np  # noqa: E402

RESULTS = os.path.join(ROOT, "docs", "results", "gazebo_check.json")
IMAGES = os.path.join(ROOT, "docs", "images")
DEFAULT_SEEDS = (0, 1, 2)


def _rel(path: str) -> str:
    """Path relative to the repository when possible (another drive on Windows: absolute)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 5) if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


# ---------------------------------------------------------------------------
# worker: one scenario / seed in this process
# ---------------------------------------------------------------------------
def run_one(args) -> dict:
    from gazebo_check import scenarios as S
    from gazebo_check.harness import GzHarness, HarnessConfig

    sc = S.SCENARIOS[S.resolve([args.worker])[0]]
    if args.duration:
        sc = S.with_duration(sc, float(args.duration))
    ml = args.mpc_latency if args.mpc_latency == "measured" else float(args.mpc_latency)
    from gazebo_check import world as W
    terrain = sc.terrain or args.terrain
    if sc.spawn:
        spawn = sc.spawn
    elif terrain == W.PLANT:
        spawn = W.layout_spawn()                       # what sim.launch.py spawns at
    else:
        spawn = (0.0, 0.0, 0.56, float(args.yaw))
    cfg = HarnessConfig(terrain=terrain, spawn=spawn,
                        params=json.loads(args.params or "{}"), latency=float(args.latency), mpc_latency=ml,
                        rtf=float(args.rtf), drop_ticks=float(args.drop_ticks), seed=int(args.seed),
                        servo=json.loads(args.servo) if args.servo else None, foot_mu=args.foot_mu,
                        map_source=args.map or "", controller_start=float(sc.controller_start),
                        out_dir=args.build_dir, verbose=args.verbose)
    t0 = time.time()
    h = GzHarness(cfg)
    h.cmd_fn = S.command_fn(sc, int(args.seed))
    h.commands = S.gait_commands(sc)
    # pushes: shifted by 0.13 s per seed, so the seeds hit different gait phases
    shift = 0.13 * int(args.seed)
    k = float(args.push_scale)
    h.wrenches = [(float(t0) + shift, float(t0) + shift + float(d), tuple(k * float(v) for v in f))
                  for (t0, d, f) in sc.pushes]
    err = None
    try:
        h.run(sc.duration)
    except RuntimeError as e:
        err = str(e)
    A = h.arrays()
    extra = {"seed": int(args.seed), "wall_s": time.time() - t0, "ticks": h.ticks, "dropped_ticks": h.dropped,
             "sync_timeouts": h.wait_timeouts, "error": err, "terrain": terrain, "map_source": args.map or "",
             "mpc_solve_wall_ms_mean": float(np.mean(h.worker.solve_times) * 1e3) if h.worker.solve_times else None,
             "messages": [(round(t, 3), m) for t, m in h.messages[:60]]}
    extra.update(_plant_extra(h, A))
    res = S.evaluate(sc, A, extra)
    if args.npz:
        np.savez_compressed(args.npz, **{k: (v if v.dtype != object else v.astype(str)) for k, v in A.items()})
    h.close()
    return res


def _plant_extra(h, A) -> dict:
    """Route / start-up / terrain metrics of the plant scenarios."""
    out = {}
    if hasattr(h.cmd_fn, "result"):
        out.update(h.cmd_fn.result())
    if h.pre_log:
        pre = np.array(h.pre_log)
        out["pre_min_height"] = float(pre[:, 1].min())
        out["pre_max_tilt_deg"] = float(np.degrees(np.abs(pre[:, 2:4]).max()))
    if len(A["t"]):
        out["final_pos"] = [float(v) for v in A["pos"][-1]]
        out["final_ground"] = float(A["ground"][-1])
        gaits = [str(g) for g in A["gait"]]
        out["gait_switches"] = int(sum(1 for a, b in zip(gaits, gaits[1:]) if a != b))
        out["gaits_used"] = sorted(set(gaits))
        tc = [str(c) for c in A["terrain_class"] if str(c)]
        if tc:
            out["terrain_classes"] = {c: tc.count(c) for c in sorted(set(tc))}
        # stance-foot slip (horizontal world speed of feet in contact)
        feet, con, t = A["feet"], A["contacts"], A["t"]
        if len(t) > 2:
            v = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) / np.maximum(np.diff(t)[:, None], 1e-6)
            both = con[1:] & con[:-1]
            if both.any():
                out["stance_slip_p95"] = float(np.percentile(v[both], 95))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _worker_cmd(args, scen: str, seed: int, npz: str, build: str) -> list:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", scen, "--seed", str(seed), "--npz", npz,
           "--build-dir", build, "--latency", str(args.latency), "--mpc-latency", str(args.mpc_latency),
           "--rtf", str(args.rtf), "--drop-ticks", str(args.drop_ticks), "--terrain", args.terrain,
           "--yaw", str(args.yaw), "--params", args.params or "{}"]
    if args.duration:
        cmd += ["--duration", str(args.duration)]
    if args.servo:
        cmd += ["--servo", args.servo]
    if args.foot_mu is not None:
        cmd += ["--foot-mu", str(args.foot_mu)]
    if args.push_scale != 1.0:
        cmd += ["--push-scale", str(args.push_scale)]
    if args.map:
        cmd += ["--map", args.map]
    return cmd


def run_all(args) -> dict:
    from gazebo_check import scenarios as S
    scens = S.resolve(args.scenarios)
    seeds = [int(s) for s in args.seeds]
    work = os.path.abspath(args.work_dir or os.path.join(tempfile.gettempdir(), "gazebo_check"))
    os.makedirs(work, exist_ok=True)
    jobs = [(sc, sd) for sc in scens for sd in seeds]
    running, results = [], {}
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < max(1, args.jobs):
            sc, sd = jobs.pop(0)
            npz = os.path.join(work, f"{args.label}_{sc}_s{sd}.npz")
            env = dict(os.environ, GZ_PARTITION=f"gzc_{uuid.uuid4().hex[:10]}")
            log = open(os.path.join(work, f"{args.label}_{sc}_s{sd}.log"), "w", encoding="utf-8")
            p = subprocess.Popen(_worker_cmd(args, sc, sd, npz, work), stdout=subprocess.PIPE, stderr=log,
                                 env=env, text=True)
            running.append((p, sc, sd, log, npz))
        time.sleep(0.5)
        for item in list(running):
            p, sc, sd, log, npz = item
            if p.poll() is None:
                continue
            running.remove(item)
            out = p.stdout.read()
            log.close()
            res = None
            for line in out.splitlines():
                if line.startswith("RESULT "):
                    res = json.loads(line[7:])
            if res is None:
                res = {"scenario": sc, "seed": sd, "pass": False, "reason": f"worker failed (exit {p.returncode})"}
            res["npz"] = npz
            results.setdefault(sc, {})[str(sd)] = res
            tag = "PASS" if res.get("pass") else "FAIL"
            print(f"[{time.time() - t0:6.0f}s] {sc:9s} seed {sd}: {tag}  {res.get('reason', '')}  "
                  f"vx={res.get('vx')} wz={res.get('wz')} falls={res.get('falls')}", flush=True)
    summary = {}
    for sc in scens:
        rs = results.get(sc, {})
        summary[sc] = {"passed": sum(1 for r in rs.values() if r.get("pass")), "runs": len(rs),
                       "title": S.SCENARIOS[sc].title}
    return {"config": {"latency": args.latency, "mpc_latency": args.mpc_latency, "rtf": args.rtf,
                       "drop_ticks": args.drop_ticks, "terrain": args.terrain, "yaw": args.yaw,
                       "push_scale": args.push_scale, "map": args.map or False,
                       "params": json.loads(args.params or "{}"),
                       "servo": json.loads(args.servo) if args.servo else None, "foot_mu": args.foot_mu,
                       "seeds": seeds},
            "summary": summary, "scenarios": results,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["a", "b", "c", "d", "e"])
    ap.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--label", default="latest", help="result block in the JSON (e.g. stable / mpc)")
    ap.add_argument("--latency", type=float, default=0.01, help="[s] command delay (0.01 = one control period)")
    ap.add_argument("--mpc-latency", default="measured", help="measured | seconds")
    ap.add_argument("--rtf", type=float, default=1.0, help="real-time factor of the emulated machine")
    ap.add_argument("--drop-ticks", type=float, default=0.0)
    ap.add_argument("--terrain", default="flat", choices=("flat", "obstacles", "pipe", "ramp", "plant"),
                    help="terrain of the scenarios without their own (f/g/h set pipe / ramp, j..s the plant)")
    ap.add_argument("--yaw", type=float, default=0.0, help="spawn yaw [rad]")
    ap.add_argument("--params", default="{}", help="JSON controller parameter overrides")
    ap.add_argument("--mode", default=None, choices=("stable", "mpc"),
                    help="locomotion_mode (shortcut for --params '{\"locomotion_mode\": ...}')")
    ap.add_argument("--servo", default=None, help='JSON servo overrides, e.g. {"p_gain": 300, "d_gain": 2}')
    ap.add_argument("--foot-mu", type=float, default=None)
    ap.add_argument("--push-scale", type=float, default=1.0, help="scale of the push forces of (i)")
    ap.add_argument("--map", nargs="?", const="truth", default="", choices=("", "truth", "depth"),
                    help="elevation map: truth (--map alone: ground-truth grid) | depth (depth camera ray cast + "
                         "terrain_mapper pipeline + /terrain/info); omitted: blind")
    ap.add_argument("--duration", type=float, default=0.0, help="cap the scenario duration [s] (quick checks)")
    ap.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--figures", action="store_true", help="write docs/images/gazebo_*.png")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--work-dir", default="", help="npz / log directory (default: <tmp>/gazebo_check)")
    ap.add_argument("--verbose", action="store_true")
    # worker mode (internal)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--npz", default="", help=argparse.SUPPRESS)
    ap.add_argument("--build-dir", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.mode:
        prm = json.loads(args.params or "{}")
        prm["locomotion_mode"] = args.mode
        args.params = json.dumps(prm)
    if args.worker:
        res = run_one(args)
        print("RESULT " + json.dumps(_clean(res)), flush=True)
        return 0

    data = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
    if not args.figures_only:
        block = run_all(args)
        data.setdefault("runs", {})[args.label] = _clean(block)
        data["tool"] = "tools/gazebo_check.py"
        data["engine"] = "gz-sim 8 (Harmonic) + DART, headless, gz-transport lock-step"
        if not args.no_save:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, indent=1)
                f.write("\n")
            print(f"wrote {_rel(args.out)}")
        s = block["summary"]
        print("summary: " + ", ".join(f"{k} {v['passed']}/{v['runs']}" for k, v in s.items()))
    if args.figures or args.figures_only:
        from gazebo_check import plots
        for p in plots.make_all(data, IMAGES):
            print(f"figure {_rel(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
