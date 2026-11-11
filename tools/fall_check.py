#!/usr/bin/env python3
"""
Headless Gazebo check of the physical fall recovery (development tool).

Same harness as tools/gazebo_check.py (gz-sim + DART, the robot with its own
gz plugins, gz-transport lock-step, the REAL ROS-free LocomotionController at
100 Hz of sim time, tools/gazebo_check/).  Every case knocks the robot over
(or spawns it lying) and checks that it gets up with its own legs: fall
detected -> self-righting -> STAND, without any gz set_pose teleport (the
controller defaults, ``assisted_reset`` false), and walks again 2 s after
the stand-up: 6 s of trot at 0.3 m/s on the flat floor, 6 s of trot in place
(``walk`` command) in the plant world - its obstacles (pallet, pipe, ramp
edge) are close, and where to go is the mission's job (case ``mission``).

Cases (flat floor / the REAL plant world simulation/worlds/plant.sdf)::

    side       spawn lying on a side (roll +-(90 +- 15) deg, pitch +-11 deg, random yaw)
    back       spawn on the back (roll 180 +- 20 deg)
    trot       trotting 0.4 m/s, lateral push 400..500 N x 0.2 s at the base (trot fall)
    roll       standing, lateral push 400..500 N x 0.2 s 0.3 m above the base (topples over the feet)
    push       standing, backward push 400..500 N x 0.2 s 0.3 m above the base
    wall       plant: standing 0.5 m from the lower-corridor wall, pushed towards it (300..400 N)
    ramp_walk  plant: trotting 0.3 m/s up the 11.8 deg corridor ramp, lateral knock (250..350 N) on the ramp
    ramp_push  plant: trots up onto the ramp, stops in the middle of it, stands 2.5 s, lateral
               knock (300..400 N)
    mission    plant: mission in the loop (the REAL MissionLogic, upper corridor -> V1, depth-camera
               map), lateral knock (450..550 N) at ~14 s while walking: the robot gets up and the
               mission carries on to the viewpoint (DONE)
    pipe_back  plant: spawn on the back ACROSS the C-OBS5 pipe (d 0.08 m) like a fall while crossing it
               (heading across the pipe west / east, the pipe 0.04..0.14 m behind the base under the rear
               of the trunk: tail propped up, nose on the floor, pitch ~ +11 deg)
    ramp_edge  plant: spawn upright but hanging over the unrailed south edge of the corridor ramp (fallen
               off it sideways): one pair of feet on the ramp, the other in the 0.2..0.4 m lower channel
               beside it, pitch ~ 45 deg (even seeds facing north = nose up, odd seeds facing south)

The flat-floor cases run without an elevation map (flat ground), the plant
cases with the emulated depth camera -> terrain_mapper map (``--map depth``
of tools/gazebo_check.py), like the robot on Ubuntu.

Lying spawns: the gz IMU orientation is relative to the spawn attitude (the
URDF IMU has no orientation reference frame), so the side / back cases take
roll / pitch from the odometry (``use_imu_orientation`` false); on Ubuntu the
robot spawns upright and the IMU reads the true attitude.

A run passes when the controller detected the fall, got up with no teleport
(``assisted_resets`` 0), walked again (WALK mode for at least half of the
walk window) and ends standing (mode STAND / WALK, |roll|, |pitch| < 15 deg
over the last 2 s, base > 0.25 m above the ground); the mission case needs
the viewpoint reached instead of the walk.  Runs in which the knock did
not topple the robot count as ``no_fall`` (not in the recovery rate).  A
robot knocked over again by its walk walks again after the next stand-up
(counted as ``falls_after_rec``).  A run ends once the robot stands 10 s
after its latest recovery (mission: at the mission's end), at the latest
after ``--duration``.

Results::

    docs/results/fall_check.json   per case / seed metrics + summary, one block per --label

Usage (needs the gz python bindings, e.g. conda-forge gz-sim8 / Harmonic)::

    python tools/fall_check.py                                  # all cases, seeds 0..9
    python tools/fall_check.py --cases side back --seeds 0 1 2 --jobs 3
    python tools/fall_check.py --cases ramp_push --seeds 0 --no-save
    python tools/fall_check.py --params '{"assisted_reset": true}' --label assisted --out <file>.json
    python tools/fall_check.py --summary-only                   # print the saved summary
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

import numpy as np  # noqa: E402

RESULTS = os.path.join(ROOT, "docs", "results", "fall_check.json")
CASES = ("side", "back", "trot", "roll", "push", "wall", "ramp_walk", "ramp_push", "mission", "pipe_back",
         "ramp_edge")
DEFAULT_CASES = CASES
DEFAULT_SEEDS = tuple(range(10))
ALIASES = {"trot_fall": "trot", "trot-fall": "trot", "ramp-walk": "ramp_walk", "ramp-push": "ramp_push",
           "pipe-back": "pipe_back", "ramp-edge": "ramp_edge"}

# acceptance of a run
FINAL_TILT_DEG = 15.0         # |roll|, |pitch| over the last FINAL_WINDOW s
FINAL_HEIGHT = 0.25           # [m] base above the ground over the last FINAL_WINDOW s
FINAL_WINDOW = 2.0
WALK_DELAY, WALK_TIME, WALK_SPEED = 2.0, 6.0, 0.3     # walk after a recovery
END_AFTER_RECOVERY = WALK_DELAY + WALK_TIME + 2.0     # [s] the run ends this long after the latest recovery
IN_PLACE_CASES = ("wall", "ramp_walk", "ramp_push", "pipe_back", "ramp_edge")   # walk in place after the recovery
                                                                                  # (plant world)

# plant world spots (simulation/config/world_layout.yaml geometry)
Z_LOWER = -0.88               # lower corridor floor (dock, pallet; foot of the ramp)
SPAWN_DZ = 0.557              # spawn height of the base above the floor (gazebo_check scenarios)
WALL_SPOT = (57.0, 1.3)       # lower corridor, 0.5 m from the north wall (y = 1.8)
RAMP_WALK_SPOT = (52.6, 0.8)  # lower corridor floor east of the ramp (ramp x 46.97 .. 51.22, uphill to -x)
RAMP_SPOT = (49.1, 0.8)       # middle of the ramp (11.8 deg; north wall y = 1.8, drop-off south of y = -0.17)
RAMP_EDGE_Y = -0.165          # unrailed south edge of the ramp (lower channel y -1.7 .. -0.165, z -0.88)
PIPE_SPOT = (22.0, 0.0)       # C-OBS5 pipe across the upper corridor: centre, axis yaw, diameter
PIPE_YAW = 1.92
PIPE_D = 0.08


def canonical(name: str) -> str:
    n = ALIASES.get(name, name)
    if n not in CASES:
        raise KeyError(f"unknown case {name!r} (known: {', '.join(CASES)})")
    return n


def _rel(path: str) -> str:
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
        return round(f, 4) if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


# ---------------------------------------------------------------------------
# case set-up (pure python)
# ---------------------------------------------------------------------------
def case_setup(case: str, seed: int) -> dict:
    """Spawn, pushes and commands of one case / seed (deterministic).

    Returns a dict: terrain, spawn (x, y, z, yaw), spawn_rp (roll, pitch),
    params (controller / node overrides), commands [(t, text)], pushes
    [(t0, t1, world force, application point in the base frame | None)],
    pre_cmd (t0, t1, (vx, vy, wz)) before the fall | None, stop_knock
    {stop_x, delay, force, offset} | None (walk until the odometry x <= stop_x,
    then a lateral knock of ``force`` N in the heading frame ``delay`` s after
    the stop), mission (scenario key) | None, map_source."""
    case = canonical(case)
    seed = int(seed)
    rng = np.random.default_rng(9000 + 31 * seed + sum(map(ord, case)))
    sign = 1.0 if seed % 2 == 0 else -1.0
    yaw = float(rng.uniform(-math.pi, math.pi))
    out = {"case": case, "seed": seed, "terrain": "flat", "spawn": (0.0, 0.0, 0.56, yaw), "spawn_rp": (0.0, 0.0),
           "params": {}, "commands": [], "pushes": [], "pre_cmd": None, "stop_knock": None, "mission": None,
           "map_source": ""}
    c, s = math.cos(yaw), math.sin(yaw)
    if case == "pipe_back":
        # lying on the back across the pipe after a fall while crossing it (mission_check: the robot
        # crosses at right angles; the pipe ends up under the rear of the trunk, the nose on the floor)
        from gazebo_check import world as W
        heading = PIPE_YAW + (math.pi / 2 if seed % 2 == 0 else -math.pi / 2)    # west / east across
        yaw = math.atan2(math.sin(heading), math.cos(heading)) + float(rng.uniform(-0.35, 0.35))
        ax = (math.cos(PIPE_YAW), math.sin(PIPE_YAW))
        u = float(rng.uniform(-0.4, 0.4))                     # along the pipe
        d = float(rng.uniform(0.04, 0.14))                    # pipe behind the base
        x = PIPE_SPOT[0] + u * ax[0] + d * math.cos(yaw)
        y = PIPE_SPOT[1] + u * ax[1] + d * math.sin(yaw)
        out["terrain"] = W.PLANT
        out["map_source"] = "depth"
        out["spawn"] = (x, y, 0.19, yaw)
        out["spawn_rp"] = (math.pi + float(rng.uniform(-0.17, 0.17)), 0.19 + float(rng.uniform(-0.05, 0.05)))
        out["params"]["use_imu_orientation"] = False          # see the module docstring
    elif case == "ramp_edge":
        from gazebo_check import world as W
        north = seed % 2 == 0
        yaw = (math.pi / 2 if north else -math.pi / 2) + float(rng.uniform(-0.3, 0.3))
        out["terrain"] = W.PLANT
        out["map_source"] = "depth"
        out["spawn"] = (float(rng.uniform(49.3, 50.1)), RAMP_EDGE_Y - 0.04 + float(rng.uniform(-0.04, 0.04)),
                        -0.36, yaw)
        out["spawn_rp"] = (float(rng.uniform(-0.1, 0.1)), (-0.75 if north else 0.75) + float(rng.uniform(-0.1, 0.1)))
        out["params"]["use_imu_orientation"] = False          # see the module docstring
    elif case in ("side", "back"):
        if case == "side":
            roll = sign * (math.pi / 2 + float(rng.uniform(-0.26, 0.26)))
        else:
            roll = math.pi + float(rng.uniform(-0.35, 0.35))
        out["spawn"] = (0.0, 0.0, 0.30, yaw)
        out["spawn_rp"] = (roll, float(rng.uniform(-0.2, 0.2)))
        out["params"]["use_imu_orientation"] = False          # see the module docstring
    elif case == "trot":
        out["commands"] = [(3.5, "gait:trot")]
        out["pre_cmd"] = (4.0, 30.0, (0.4, 0.0, 0.0))
        tp = 7.0 + float(rng.uniform(0.0, 0.4))
        F = sign * float(rng.uniform(400.0, 500.0))
        out["pushes"] = [(tp, tp + 0.2, (-s * F, c * F, 0.0), None)]               # lateral, at the base
    elif case == "roll":
        F = sign * float(rng.uniform(400.0, 500.0))
        out["pushes"] = [(5.0, 5.2, (-s * F, c * F, 0.0), (0.0, 0.0, 0.3))]
    elif case == "push":
        F = float(rng.uniform(400.0, 500.0))
        out["pushes"] = [(5.0, 5.2, (-c * F, -s * F, 0.0), (0.0, 0.0, 0.3))]      # backward
    elif case in ("wall", "ramp_walk", "ramp_push"):
        from gazebo_check import world as W
        out["terrain"] = W.PLANT
        out["map_source"] = "depth"                    # depth camera -> terrain_mapper map, as on Ubuntu
        if case == "wall":
            yaw = math.pi + float(rng.uniform(-0.3, 0.3))
            out["spawn"] = (WALL_SPOT[0], WALL_SPOT[1], Z_LOWER + SPAWN_DZ, yaw)
            F = float(rng.uniform(300.0, 400.0))
            out["pushes"] = [(6.0, 6.2, (0.0, F, 0.0), (0.0, 0.0, 0.3))]           # towards the wall (+y)
        elif case == "ramp_walk":
            yaw = math.pi + float(rng.uniform(-0.15, 0.15))
            c, s = math.cos(yaw), math.sin(yaw)
            out["spawn"] = (RAMP_WALK_SPOT[0], RAMP_WALK_SPOT[1], Z_LOWER + SPAWN_DZ, yaw)
            out["commands"] = [(3.5, "gait:trot")]
            out["pre_cmd"] = (4.0, 60.0, (0.3, 0.0, 0.0))                           # west = uphill
            tp = 13.0 + float(rng.uniform(0.0, 0.5))
            F = sign * float(rng.uniform(250.0, 350.0))
            out["pushes"] = [(tp, tp + 0.2, (-s * F, c * F, 0.0), (0.0, 0.0, 0.3))]
        else:
            # walked onto the ramp like on a patrol (a robot spawned on the slope with level
            # legs lands on the uphill feet first and tips over backwards before it stands)
            yaw = math.pi + float(rng.uniform(-0.15, 0.15))
            out["spawn"] = (RAMP_WALK_SPOT[0], RAMP_WALK_SPOT[1], Z_LOWER + SPAWN_DZ, yaw)
            out["commands"] = [(3.5, "gait:trot")]
            out["pre_cmd"] = (4.0, 60.0, (0.3, 0.0, 0.0))
            out["stop_knock"] = {"stop_x": RAMP_SPOT[0], "delay": 2.5 + float(rng.uniform(0.0, 0.3)),
                                 "force": sign * float(rng.uniform(300.0, 400.0)), "offset": (0.0, 0.0, 0.3)}
    elif case == "mission":
        from gazebo_check import scenarios as S
        sc = S.MISSION_SCENARIOS["x_control_upper"]
        out["terrain"] = "plant"
        out["mission"] = sc.key
        out["spawn"] = sc.spawn_for(seed)
        out["map_source"] = "depth"
        tp = 14.0 + float(rng.uniform(0.0, 0.4))
        F = sign * float(rng.uniform(450.0, 550.0))
        out["pushes"] = [(tp, tp + 0.2, (0.0, F, 0.0), (0.0, 0.0, 0.3))]           # walking west: lateral = y
    return out


def case_duration(case: str) -> float:
    return {"mission": 120.0, "ramp_walk": 100.0, "ramp_push": 100.0, "pipe_back": 90.0,
            "ramp_edge": 90.0}.get(canonical(case), 60.0)


class AfterRecoveryWalk:
    """Velocity command of the knock-down cases: ``pre_cmd`` until the first
    fall, zero while down, then WALK_SPEED forward for WALK_TIME s starting
    WALK_DELAY s after a recovery (the latest one: knocked over again by the
    walk, it walks again after the next stand-up), zero afterwards
    (``in_place``: the ``walk`` command instead - trot in place - and
    ``stand`` after WALK_TIME).
    ``stop_knock`` (see :func:`case_setup`): ``pre_cmd`` ends when the
    odometry x reaches ``stop_x``, the knock is added to the harness wrenches
    ``delay`` s later."""

    def __init__(self, pre_cmd=None, stop_knock=None, in_place: bool = False):
        self.pre_cmd = pre_cmd
        self.stop_knock = stop_knock
        self.in_place = bool(in_place)
        self.t_stop = None
        self.h = None
        self.bound = False
        self.t_rec = None                     # latest recovery
        self.n_rec = 0

    def bind(self, h) -> None:
        self.h = h
        self.bound = True

    def __call__(self, t: float):
        rec = self.h.ctrl.recovery if self.h is not None else None
        if rec is not None and rec.successes > self.n_rec:
            self.n_rec = rec.successes
            self.t_rec = float(t)
            if self.in_place and self.h is not None:
                self.h.commands += [(self.t_rec + WALK_DELAY, "walk"), (self.t_rec + WALK_DELAY + WALK_TIME, "stand")]
        if self.t_rec is not None:
            if not self.in_place and self.t_rec + WALK_DELAY <= t < self.t_rec + WALK_DELAY + WALK_TIME:
                return (WALK_SPEED, 0.0, 0.0)
            return (0.0, 0.0, 0.0)
        if self.stop_knock is not None and self.t_stop is None and self.h is not None:
            with self.h._lock:
                odom = self.h._odom
            if odom is not None and float(odom[1][0]) <= float(self.stop_knock["stop_x"]):
                self.t_stop = float(t)
                yaw = float(odom[2][2])
                F = float(self.stop_knock["force"])
                t0 = self.t_stop + float(self.stop_knock["delay"])
                self.h.wrenches.append((t0, t0 + 0.2, (-math.sin(yaw) * F, math.cos(yaw) * F, 0.0),
                                        tuple(self.stop_knock["offset"])))
        if self.pre_cmd is not None and self.t_stop is None and (rec is None or rec.fall_count == 0):
            t0, t1, cmd = self.pre_cmd
            if t0 <= t < t1:
                return cmd
        return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# evaluation (pure python)
# ---------------------------------------------------------------------------
def evaluate(case: str, A: dict, rec_info: dict, mission: dict = None) -> dict:
    """Metrics + pass / fail of one run.  ``A``: harness arrays (t, pos, rpy,
    ground, mode, rec_state); ``rec_info``: controller recovery counters and
    events (FallRecovery.events, t_first_recovery, attempts_at_recovery)."""
    ev = rec_info.get("events", [])
    falls = int(rec_info.get("fall_count", 0))
    t_fall = next((e["t"] for e in ev if e["event"] == "fall"), None)
    t_rec = next((e["t"] for e in ev if e["event"] == "recovered"), None)
    m = {"falls": falls, "attempts": int(rec_info.get("attempts_total", 0)),
         "recoveries": int(rec_info.get("successes", 0)), "assisted": int(rec_info.get("assisted_resets", 0)),
         "retries": sum(1 for e in ev if e["event"] == "recovery_retry"),
         "series_failed": sum(1 for e in ev if e["event"] == "recovery_failed"),
         "t_fall": t_fall, "t_rec": t_rec,
         "ttr": (t_rec - t_fall) if (t_rec is not None and t_fall is not None) else None,
         "first_try": bool(t_rec is not None and rec_info.get("attempts_at_recovery") == 1),
         "events": [(e["event"], round(float(e["t"]), 2), round(math.degrees(float(e["roll"])), 1),
                     round(math.degrees(float(e["pitch"])), 1)) for e in ev[:30]]}
    t = np.asarray(A.get("t", []), dtype=float)
    seq = []
    for st in A.get("rec_state", []):
        if not seq or seq[-1] != str(st):
            seq.append(str(st))
    m["seq"] = seq[:50]
    if len(t) == 0:
        m.update({"ok": False, "fell": falls > 0, "reason": "no data"})
        return m
    modes = [str(v) for v in A["mode"]]
    w = t >= t[-1] - FINAL_WINDOW
    h = np.asarray(A["pos"])[w, 2] - np.asarray(A["ground"])[w]
    rpy = np.asarray(A["rpy"])
    m["final_mode"] = modes[-1]
    m["final_h"] = float(np.mean(h))
    m["final_roll"] = float(np.degrees(np.abs(rpy[w, 0]).max()))
    m["final_pitch"] = float(np.degrees(np.abs(rpy[w, 1]).max()))
    m["sim_t"] = float(t[-1])
    if t_fall is not None:
        i = min(int(np.searchsorted(t, t_fall)), len(t) - 1)
        m["fall_rpy"] = [round(float(np.degrees(v)), 1) for v in rpy[i]]
    walked = False
    if t_rec is not None:
        t_last = [e["t"] for e in ev if e["event"] == "recovered"][-1]     # the walk follows the latest one
        ww = (t >= t_last + WALK_DELAY) & (t <= t_last + WALK_DELAY + WALK_TIME)
        if ww.sum() > 2:
            p = np.asarray(A["pos"])[ww]
            m["walk_after_dist"] = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
            m["walk_after_frac"] = float(np.mean([modes[i] == "WALK" for i in np.flatnonzero(ww)]))
            walked = m["walk_after_frac"] >= 0.5
        m["falls_after_rec"] = sum(1 for e in ev if e["event"] == "fall" and e["t"] > t_rec)
    standing = (m["final_mode"] in ("STAND", "WALK") and m["final_roll"] < FINAL_TILT_DEG
                and m["final_pitch"] < FINAL_TILT_DEG and m["final_h"] > FINAL_HEIGHT)
    m["fell"] = falls > 0
    ok = m["fell"] and t_rec is not None and m["assisted"] == 0 and standing
    if mission is not None:
        m["mission"] = mission
        ok = ok and float(mission.get("reached") or 0.0) >= 1.0
    else:
        ok = ok and walked
    m["ok"] = bool(ok)
    if not m["fell"]:
        m["reason"] = "no fall (the knock did not topple the robot)"
    elif t_rec is None:
        m["reason"] = "not recovered within the run"
    elif m["assisted"]:
        m["reason"] = "teleport (assisted reset)"
    elif not standing:
        m["reason"] = (f"not standing at the end (mode {m['final_mode']}, h {m['final_h']:.2f} m, "
                       f"roll {m['final_roll']:.0f}, pitch {m['final_pitch']:.0f} deg)")
    elif mission is None and not walked:
        m["reason"] = "did not walk again after the stand-up"
    elif mission is not None and not ok:
        m["reason"] = f"mission did not reach the viewpoint (state {mission.get('final_state')})"
    else:
        m["reason"] = "recovered physically" + (", mission continued" if mission is not None else "")
    return m


def summarize(runs: dict) -> dict:
    """Per case: runs, fell, ok (physical recoveries with the final checks),
    rate = ok / fell, first-try recoveries, time to recover, teleports."""
    out = {}
    for case, rs in runs.items():
        rs = list(rs.values())
        fell = [r for r in rs if r.get("fell")]
        ttr = [r["ttr"] for r in fell if r.get("ok") and r.get("ttr") is not None]
        out[case] = {"runs": len(rs), "fell": len(fell), "ok": sum(1 for r in fell if r.get("ok")),
                     "rate": (sum(1 for r in fell if r.get("ok")) / len(fell)) if fell else None,
                     "first_try": sum(1 for r in fell if r.get("ok") and r.get("first_try")),
                     "ttr_mean": float(np.mean(ttr)) if ttr else None, "ttr_max": float(max(ttr)) if ttr else None,
                     "teleports": sum(int(r.get("assisted") or 0) for r in rs),
                     "retries": sum(int(r.get("retries") or 0) for r in rs),
                     "falls_after_rec": sum(int(r.get("falls_after_rec") or 0) for r in rs),
                     "errors": sum(1 for r in rs if r.get("error"))}
    return out


# ---------------------------------------------------------------------------
# worker: one case / seed in this process
# ---------------------------------------------------------------------------
def run_one(args) -> dict:
    from gazebo_check import scenarios as S
    from gazebo_check.harness import GzHarness, HarnessConfig

    case, seed = canonical(args.worker), int(args.seed)
    su = case_setup(case, seed)
    params = dict(su["params"])
    params.update(json.loads(args.params or "{}"))
    duration = float(args.duration) if args.duration else case_duration(case)
    cfg = HarnessConfig(terrain=su["terrain"], spawn=su["spawn"], spawn_rp=su["spawn_rp"], params=params,
                        latency=float(args.latency), mpc_latency="measured", seed=seed,
                        map_source=su["map_source"] or args.map, perception_depth=su["mission"] is not None,
                        out_dir=args.build_dir, verbose=args.verbose)
    t_wall = time.time()
    h = GzHarness(cfg)
    op = None
    if su["mission"]:
        sc = S.MISSION_SCENARIOS[su["mission"]]
        op = sc.operator({}, seed=seed)
        h.cmd_fn = op
    else:
        h.cmd_fn = AfterRecoveryWalk(su["pre_cmd"], su["stop_knock"], in_place=case in IN_PLACE_CASES)
    h.commands = list(su["commands"])
    h.wrenches = [tuple(w) for w in su["pushes"]]
    rec = h.ctrl.recovery
    err = None
    t_first_rec = None                       # first recovery (attempts used up to it: first_try)
    att_at_rec = None
    t_end_mission = None
    try:
        while h.t < duration - 1e-9:
            h.run(min(1.0, duration - h.t))
            if t_first_rec is None and rec.successes > 0:
                t_first_rec, att_at_rec = h.t, rec.attempts_total
            if op is not None:
                if t_end_mission is None and op.m.base in ("DONE", "RTH", "DOCKED") and rec.successes > 0:
                    t_end_mission = h.t
                if t_end_mission is not None and h.t >= t_end_mission + FINAL_WINDOW + 0.5:
                    break
            elif (h.cmd_fn.t_rec is not None and h.t >= h.cmd_fn.t_rec + END_AFTER_RECOVERY
                  and h.ctrl.mode in ("STAND", "WALK")):
                break                         # (knocked down again by the walk: wait for that recovery)
    except RuntimeError as e:
        err = str(e)[:500]
    A = h.arrays()
    info = {"events": list(rec.events), "fall_count": rec.fall_count, "attempts_total": rec.attempts_total,
            "successes": rec.successes, "assisted_resets": rec.assisted_resets,
            "attempts_at_recovery": att_at_rec}
    mission = None
    if op is not None:
        r = op.result()
        mission = {"reached": r.get("reached"), "t_reach": r.get("t_reach"), "final_state": r.get("final_state"),
                   "events": [e for e in r.get("events", []) if any(k in e[2] for k in (
                       "locomotion", "recover", "reached", "DONE", "skip"))][:20]}
    res = evaluate(case, A, info, mission)
    if args.npz:
        np.savez_compressed(args.npz, **{k: (v if v.dtype != object else v.astype(str)) for k, v in A.items()})
    res.update({"case": case, "seed": seed, "error": err, "wall_s": round(time.time() - t_wall, 1),
                "spawn": [round(float(v), 3) for v in su["spawn"]],
                "spawn_rp_deg": [round(math.degrees(v), 1) for v in su["spawn_rp"]],
                "pushes": [[round(w[0], 2), [round(float(f), 1) for f in w[2]],
                            None if w[3] is None else [float(v) for v in w[3]]] for w in h.wrenches]})
    if err:
        res["ok"] = False
        res["reason"] = "harness error: " + err.splitlines()[0]
    h.close()
    return res


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _worker_cmd(args, case: str, seed: int, build: str) -> list:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", case, "--seed", str(seed), "--build-dir", build,
           "--latency", str(args.latency), "--params", args.params or "{}"]
    if args.save_npz:
        cmd += ["--npz", os.path.join(build, f"{args.label}_{case}_s{seed}.npz")]
    if args.duration:
        cmd += ["--duration", str(args.duration)]
    if args.map:
        cmd += ["--map", args.map]
    return cmd


def run_all(args) -> dict:
    cases = [canonical(c) for c in args.cases]
    seeds = [int(s) for s in args.seeds]
    work = os.path.abspath(args.work_dir or os.path.join(tempfile.gettempdir(), "fall_check"))
    os.makedirs(work, exist_ok=True)
    jobs = [(c, s) for c in cases for s in seeds]
    running, runs = [], {}
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < max(1, args.jobs):
            c, s = jobs.pop(0)
            env = dict(os.environ, GZ_PARTITION=f"fall_{uuid.uuid4().hex[:10]}")
            log = open(os.path.join(work, f"{args.label}_{c}_s{s}.log"), "w", encoding="utf-8")
            p = subprocess.Popen(_worker_cmd(args, c, s, work), stdout=subprocess.PIPE, stderr=log, env=env,
                                 text=True)
            running.append((p, c, s, log))
        time.sleep(0.5)
        for item in list(running):
            p, c, s, log = item
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
                res = {"case": c, "seed": s, "ok": False, "fell": False, "error": f"worker exit {p.returncode}",
                       "reason": f"worker failed (exit {p.returncode})"}
            runs.setdefault(c, {})[str(s)] = res
            print(f"[{time.time() - t0:6.0f}s] {c:9s} seed {s}: {'PASS' if res.get('ok') else 'FAIL'}  "
                  f"{res.get('reason', '')}  ttr={res.get('ttr')} attempts={res.get('attempts')} "
                  f"seq={'>'.join((res.get('seq') or [])[:14])}", flush=True)
    runs = {c: runs.get(c, {}) for c in cases}
    return {"config": {"cases": cases, "seeds": seeds, "params": json.loads(args.params or "{}"),
                       "latency": args.latency, "map": args.map or "", "duration": args.duration or None},
            "summary": summarize(runs), "cases": runs, "created": time.strftime("%Y-%m-%d %H:%M:%S")}


def print_summary(summary: dict) -> None:
    for c, v in summary.items():
        rate = "-" if v["rate"] is None else f"{100 * v['rate']:.0f} %"
        ttr = "-" if v["ttr_mean"] is None else f"{v['ttr_mean']:.1f} s (max {v['ttr_max']:.1f})"
        print(f"  {c:9s} {v['ok']:2d}/{v['fell']:2d} fell ({v['runs']} runs)  rate {rate:5s}  first try "
              f"{v['first_try']}  time to recover {ttr}  teleports {v['teleports']}  retries {v['retries']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help=f"of {', '.join(CASES)}")
    ap.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--label", default="latest", help="result block in the JSON")
    ap.add_argument("--params", default="{}", help="JSON controller parameter overrides")
    ap.add_argument("--latency", type=float, default=0.01, help="[s] command delay (0.01 = one control period)")
    ap.add_argument("--map", default="", choices=("", "truth", "depth"),
                    help="elevation map of the flat-floor cases (the plant cases always use depth)")
    ap.add_argument("--duration", type=float, default=0.0, help="cap of the run duration [s] (default per case)")
    ap.add_argument("--jobs", type=int, default=max(1, min(5, (os.cpu_count() or 2) // 3)))
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="print the summary of --out and exit")
    ap.add_argument("--work-dir", default="", help="log directory (default: <tmp>/fall_check)")
    ap.add_argument("--save-npz", action="store_true", help="keep the logged arrays of every run in --work-dir")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--build-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--npz", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

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
    if args.summary_only:
        for label, block in (data.get("runs") or {}).items():
            print(f"{label} ({block.get('created')}):")
            print_summary(block.get("summary", {}))
        return 0
    block = run_all(args)
    data.setdefault("runs", {})[args.label] = _clean(block)
    data["tool"] = "tools/fall_check.py"
    data["engine"] = "gz-sim 8 (Harmonic) + DART, headless, gz-transport lock-step"
    if not args.no_save:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=1)
            f.write("\n")
        print(f"wrote {_rel(args.out)}")
    print("summary:")
    print_summary(block["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
