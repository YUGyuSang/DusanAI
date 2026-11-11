#!/usr/bin/env python3
"""
Reproducible MPC / gait parameter search for module1_locomotion (development tool).

Searches the key MPC and gait parameters of ``controller.DEFAULT_PARAMS``
(MPC state weights Q, force weight R, capture-point gain and its decay, lateral
foothold limit, trot period / duty, swing height, IMU attitude kp / kd) with a
multi-objective score from two evaluations of the REAL ROS-free controller code:

* rigid-body plant (tools/eval_locomotion.py ``EvalRun``: GaitScheduler +
  ConvexMPC + single rigid body with ideal stance contacts, deterministic):
  velocity tracking (trot profile + crawl profile), push robustness (critical
  lateral / forward impulse found by bisection at two push phases) and MPC
  solve time (CPU time of the solving thread);
* MuJoCo physics (tools/physics_check.py scenarios through tools/physics:
  gz-like PID joint servos, sensor rates, MPC latency): short versions of
  (a) stand, (b) trot 0.4 m/s, (d) turn 0.5 rad/s, (f) the 15 N s lateral push
  while trotting and (k) the slip-mode trot onto the mu 0.15 plate (friction
  prior -> speed limit, torque feed-forward, MPC mu).

Method (multi-fidelity, all random draws from ``--seed``)
---------------------------------------------------------
1. exploration: the current defaults + ``--n-explore`` Latin-hypercube samples
   of the normalised search space (log scale for the weights);
2. successive halving over three fidelity rungs
   rung 0  rigid-body suite (all candidates)                       -> keep the best F_rb
   rung 1  physics, 1 seed stand / trot / turn / slip, 2 pushes      -> keep the best J
   rung 2  physics, 3 seeds stand / trot / turn / slip, 6 pushes
3. local refinement: a small CMA-ES (numpy) started at the best rung-2
   candidate, children scored at rungs 0 + 1; the best children go to rung 2;
4. Pareto front of (F_rb, F_phys) over the rung-2 candidates without walking
   failures; the front (by J) is CONFIRMED on held-out data - new physics
   seeds (stand / trot / turn / slip / crawl 10-12, push 10-17) with the
   depth-camera perception of the physics check and held-out push phases on
   the rigid body - together with the defaults;
5. adoption rule (``decide``): F_rb <= 1 - margin AND F_phys <= 1 - margin
   (clearly better on BOTH evaluations), no stand / trot / turn / slip
   failure, not more push falls than the defaults, rigid-body critical
   impulses not lower, crawl guard (crawl 0.2 m/s within 20 %) on every seed;
6. regression guard: the candidates passing (5), best J first, run the FULL
   required scenarios of tools/physics_check.py (seeds 0-2, as the physics
   check) next to the defaults; a candidate is adopted only if no scenario
   passes on fewer seeds than with the defaults.  (Added after the first
   search: its best candidate passed (5) but broke the slippery-plate and
   turn scenarios, which stand / trot / push alone did not cover.)

Scores (lower is better; F = 1 is the current defaults on the same data)
------------------------------------------------------------------------
f_rb   = 0.35 track + 0.10 attitude + 0.40 push + 0.15 solve        (+ penalties)
f_phys = 0.10 stand + 0.25 trot + 0.15 turn + 0.35 push + 0.15 slip  (seed means)
F_rb = f_rb / f_rb(defaults), F_phys = f_phys / f_phys(defaults),
J = (F_rb + F_phys) / 2.  The terms are tolerance-normalised metrics, see
``rb_score`` / ``stand_cost`` / ``trot_cost`` / ``turn_cost`` / ``push_cost``
/ ``slip_cost``.

Outputs: docs/results/tuning.json, docs/images/tuning_pareto.png,
tuning_sensitivity.png, tuning_progress.png.

Usage::

    python tools/tune_locomotion.py                    # full search (24 min with 12 workers on the 16-core dev PC)
    python tools/tune_locomotion.py --budget-min 20    # CMA-ES generations / guard skipped to fit
    python tools/tune_locomotion.py --replot docs/results/tuning.json   # redraw the figures only
    python tools/tune_locomotion.py --seed 1 --n-explore 64 --cma-gens 3
    python tools/tune_locomotion.py --quick            # smoke run (~2.5 min, results to <tmp>/tune_locomotion_quick)

Requires ``pip install mujoco matplotlib`` (dev / validation only; nothing in
module1_locomotion imports mujoco).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")       # small MPC matrices: one BLAS thread per worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in ("dusan_common", "module1_locomotion", "tools"):
    _q = os.path.join(ROOT, _p)
    if _q not in sys.path:
        sys.path.insert(0, _q)

from module1_locomotion.controller import DEFAULT_PARAMS  # noqa: E402

RESULTS = os.path.join(ROOT, "docs", "results", "tuning.json")
IMAGES = os.path.join(ROOT, "docs", "images")
DEG = math.pi / 180.0


# ---------------------------------------------------------------------------
# search space
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Param:
    """One searched dimension, normalised to u in [0, 1]."""
    name: str
    lo: float
    hi: float
    log: bool = False
    group: str = "mpc"
    phys_only: bool = False          # invisible to the rigid-body plant (ideal contacts)
    doc: str = ""

    def decode(self, u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        if self.log:
            return _round_sig(math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo))))
        # linear: 1e-3 of the range as the absolute resolution (no 1e-8 values near a zero bound), then 3 digits
        dec = 3 - int(math.floor(math.log10(self.hi - self.lo)))
        return _round_sig(round(self.lo + u * (self.hi - self.lo), dec))

    def encode(self, v: float) -> float:
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)


def _round_sig(v: float, sig: int = 3) -> float:
    """3 significant digits (clean YAML values; the defaults round-trip exactly)."""
    if v == 0.0 or not math.isfinite(v):
        return float(v)
    return float(round(v, sig - 1 - int(math.floor(math.log10(abs(v))))))


SPACE: Tuple[Param, ...] = (
    Param("q_rp", 10.0, 100.0, True, doc="MPC weight roll = pitch"),
    Param("q_yaw", 3.0, 30.0, True, doc="MPC weight yaw"),
    Param("q_xy", 1.0, 16.0, True, doc="MPC weight x = y position"),
    Param("q_z", 20.0, 200.0, True, doc="MPC weight height"),
    Param("q_w", 0.1, 2.5, True, doc="MPC weight angular rate wx = wy = wz"),
    Param("q_vxy", 0.5, 8.0, True, doc="MPC weight vx = vy"),
    Param("q_vz", 0.3, 3.0, True, doc="MPC weight vz"),
    Param("mpc_r_weight", 5e-5, 1e-3, True, doc="MPC force regularisation R"),
    Param("mpc_capture_gain", 0.3, 1.0, doc="capture-point gain of the Raibert foothold heuristic"),
    Param("mpc_feedback_tau", 0.25, 1.0, doc="[s] decay of the velocity-error feedback over the horizon"),
    Param("mpc_max_foot_dy", 0.10, 0.18, doc="[m] lateral foothold offset limit around the hip"),
    Param("trot_period", 0.32, 0.48, group="gait", doc="[s] trot cycle"),
    Param("trot_duty", 0.50, 0.60, group="gait", doc="trot stance fraction"),
    Param("step_height", 0.05, 0.11, group="swing", phys_only=True, doc="[m] trot swing apex"),
    Param("kp_attitude", 0.0, 1.0, group="attitude", phys_only=True, doc="IMU roll/pitch P correction"),
    Param("kd_attitude", 0.0, 0.01, group="attitude", phys_only=True,
          doc="gyro D term (controller: kd <= 0.01 is loop-stable)"),
)
# searched name -> indices in mpc_q_weights [roll pitch yaw x y z wx wy wz vx vy vz]
Q_INDEX = {"q_rp": (0, 1), "q_yaw": (2,), "q_xy": (3, 4), "q_z": (5,), "q_w": (6, 7, 8), "q_vxy": (9, 10),
           "q_vz": (11,)}
NAMES = tuple(p.name for p in SPACE)


def default_values(base: Optional[Dict] = None) -> Dict[str, float]:
    """Searched values of a controller parameter dict (default: DEFAULT_PARAMS)."""
    base = DEFAULT_PARAMS if base is None else base
    q = list(base["mpc_q_weights"])
    out = {}
    for p in SPACE:
        if p.name in Q_INDEX:
            idx = Q_INDEX[p.name]
            vals = {float(q[i]) for i in idx}
            if len(vals) != 1:
                raise ValueError(f"{p.name}: tied Q entries differ {[q[i] for i in idx]}")
            out[p.name] = vals.pop()
        else:
            out[p.name] = float(base[p.name])
    return out


def encode(values: Dict[str, float]) -> np.ndarray:
    return np.array([p.encode(values[p.name]) for p in SPACE], dtype=float)


def decode(u: Sequence[float]) -> Dict[str, float]:
    return {p.name: p.decode(ui) for p, ui in zip(SPACE, u)}


def to_params(values: Dict[str, float], base: Optional[Dict] = None) -> Dict:
    """Controller parameter overrides of a candidate (searched keys + the full Q list)."""
    base = DEFAULT_PARAMS if base is None else base
    q = [float(v) for v in base["mpc_q_weights"]]
    out: Dict = {}
    for name, v in values.items():
        if name in Q_INDEX:
            for i in Q_INDEX[name]:
                q[i] = float(v)
        else:
            out[name] = float(v)
    out["mpc_q_weights"] = q
    return out


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n stratified samples in [0, 1]^d: every dimension has one sample per 1/n stratum."""
    u = np.empty((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
    return u


# ---------------------------------------------------------------------------
# CMA-ES (numpy, Hansen's tutorial equations; box [0, 1] by repair)
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, x0: Sequence[float], sigma0: float, popsize: int, rng: np.random.Generator):
        n = len(x0)
        self.n, self.rng = n, rng
        self.mean = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma0)
        self.lam = int(popsize)
        self.mu = self.lam // 2
        w = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / float(np.sum(self.w ** 2))
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.C = np.eye(n)
        self.gen = 0

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        x = self.mean + self.sigma * (z * self.D) @ self.B.T
        return np.clip(x, 0.0, 1.0)              # repair: the clipped point is evaluated AND used in tell

    def tell(self, X: np.ndarray, f: Sequence[float]) -> None:
        X = np.asarray(X, dtype=float)
        f = np.asarray(f, dtype=float)
        f = np.where(np.isfinite(f), f, np.inf)
        idx = np.argsort(f, kind="stable")[: self.mu]
        y = (X[idx] - self.mean) / self.sigma
        yw = self.w @ y
        self.mean = self.mean + self.sigma * yw
        c_isqrt = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (c_isqrt @ yw)
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chi_n
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (y.T * self.w) @ y
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.C = 0.5 * (self.C + self.C.T)
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))
        self.gen += 1


# ---------------------------------------------------------------------------
# evaluation plans: kind -> seeds (rigid body: push-phase sets)
# ---------------------------------------------------------------------------
RB_PHASES_SEARCH = (0.0, 0.1)        # [s] push time offsets within the trot cycle (search)
RB_PHASES_CONFIRM = (0.05, 0.15)     # held-out push phases (confirmation)
PLANS = {
    "rung0": {"rb_track": ["s"], "rb_push_lat": ["s"], "rb_push_fwd": ["s"]},
    "rung1": {"stand": [0], "trot": [0], "turn": [0], "push": [0, 1], "slip": [0]},
    "rung2": {"stand": [0, 1, 2], "trot": [0, 1, 2], "turn": [0, 1, 2], "push": [0, 1, 2, 3, 4, 5],
              "slip": [0, 1, 2]},
    "confirm_rb": {"rb_track": ["s"], "rb_push_lat": ["c"], "rb_push_fwd": ["c"]},
    "confirm": {"stand": [10, 11, 12], "trot": [10, 11, 12], "turn": [10, 11, 12], "push": list(range(10, 18)),
                "slip": [10, 11, 12], "crawl": [10, 11, 12]},
}
SEARCH_PERCEPTION = "none"           # physics screening: no depth ray casting (2x faster)
CONFIRM_PERCEPTION = "depth"         # confirmation: the physics-check pipeline (elevation map from depth)
GUARD_SEEDS = (0, 1, 2)              # regression guard: the physics_check required scenarios on these seeds


def plan_tasks(cid: str, params: Dict, plan: Dict[str, list], perception: str) -> List[tuple]:
    out = []
    for kind, seeds in plan.items():
        for sd in seeds:
            out.append((cid, kind, sd, params, perception if not kind.startswith("rb_") else "rb"))
    return out


def task_key(t: tuple) -> tuple:
    cid, kind, seed, _params, tag = t
    return (cid, kind, seed, tag)


# ---------------------------------------------------------------------------
# rigid-body episodes (tools/eval_locomotion.py)
# ---------------------------------------------------------------------------
RB_TROT_T = 10.0
RB_TROT_WINDOWS = [(2.5, 5.0), (6.0, 7.5), (8.5, 10.0)]
RB_CRAWL_T = 9.0
RB_PUSH_RANGE = {"lat": (0.0, 64.0), "fwd": (0.0, 96.0)}     # [N s] bisection brackets
RB_BISECT_STEPS = 4


def _trot_cmd(t):
    if t < 1.0:
        return (0.0, 0.0, 0.0)
    if t < 5.0:
        return (0.5, 0.0, 0.0)
    if t < 7.5:
        return (0.0, 0.25, 0.0)
    if t < 11.0:
        return (0.3, 0.0, 0.6)
    return (0.0, 0.0, 0.0)


def _crawl_cmd(t):
    if t < 1.0:
        return (0.0, 0.0, 0.0)
    if t < 6.0:
        return (0.2, 0.0, 0.0)
    if t < 8.0:
        return (0.1, 0.0, 0.3)
    return (0.0, 0.0, 0.0)


def rb_track(params: Dict) -> Dict:
    """Trot (forward / lateral / turning) and crawl velocity tracking on the rigid body."""
    import eval_locomotion as E
    p = {**DEFAULT_PARAMS, **params}
    out = {}
    cpu, wall, iters = [], [], []
    for gait in ("trot", "crawl"):
        run = E.EvalRun(gait, params=params, crawl_duty=float(p["crawl_duty"]))
        T, cmd = (RB_TROT_T, _trot_cmd) if gait == "trot" else (RB_CRAWL_T, _crawl_cmd)
        log = run.run(T, cmd)
        per = float(p["trot_period"] if gait == "trot" else p["crawl_period"])
        win = RB_TROT_WINDOWS if gait == "trot" else [(3.0, 6.0)]
        t = log["t"]
        complete = len(t) > 0 and t[-1] >= T - 0.02
        ev = {"fallen": bool(log["fallen"]) or not complete}
        if len(t) > 10:
            ev["rms_vx_ca"] = E.window_rms(E.cycle_mean(log["vx"] - log["cmd_vx"], t, per), t, win)
            ev["rms_vy_ca"] = E.window_rms(E.cycle_mean(log["vy"] - log["cmd_vy"], t, per), t, win)
            wwin = RB_TROT_WINDOWS[2:] if gait == "trot" else [(7.0, 8.0)]
            ev["rms_wz"] = E.window_rms(log["wz"] - log["cmd_wz"], t, wwin)
            ev["max_tilt_deg"] = float(max(np.max(np.abs(log["roll"])), np.max(np.abs(log["pitch"]))) / DEG)
            ev["rms_height"] = float(np.sqrt(np.mean((log["z"] - 0.34) ** 2)))
            ev["max_torque_ratio"] = float(np.max(log["tau_ratio"]))
            ev["reachable"] = float(np.mean(log["reach"]))
        out[gait] = ev
        cpu += [c for c in run.cpu_times if np.isfinite(c)]
        wall += list(run.solve_times)
        iters += list(run.qp_iters)
    out["solve_cpu_ms"] = float(np.mean(cpu) * 1e3) if cpu else float("nan")
    out["solve_wall_ms"] = float(np.mean(wall) * 1e3) if wall else float("nan")
    out["qp_iters"] = float(np.mean(iters)) if iters else float("nan")
    out["solves"] = len(wall)
    return out


def rb_push_falls(params: Dict, direction: str, impulse: float, t_push: float) -> bool:
    """Trot 0.3 m/s, push ``impulse`` for 0.1 s at ``t_push``: fall (eval_locomotion criterion)?"""
    import eval_locomotion as E
    p = {**DEFAULT_PARAMS, **params}
    dur = 0.1
    F = impulse / dur
    vec = np.array([0.0, F, 0.0]) if direction == "lat" else np.array([F, 0.0, 0.0])
    zero = np.zeros(3)
    run = E.EvalRun("trot", params=params, crawl_duty=float(p["crawl_duty"]))
    log = run.run(t_push + 2.0, lambda t: (0.3, 0.0, 0.0) if t >= 1.0 else (0.0, 0.0, 0.0),
                  lambda t: vec if t_push <= t < t_push + dur else zero)
    return bool(log["fallen"])


def rb_push(params: Dict, direction: str, phases: Sequence[float]) -> Dict:
    """Critical impulse (largest without a fall) per push phase by bisection."""
    lo0, hi0 = RB_PUSH_RANGE[direction]
    crit = []
    for ph in phases:
        a, b = lo0, hi0
        for _ in range(RB_BISECT_STEPS):
            m = 0.5 * (a + b)
            if rb_push_falls(params, direction, m, 2.5 + float(ph)):
                b = m
            else:
                a = m
        crit.append(a)
    return {"critical_Ns": crit, "phases": list(phases), "mean_Ns": float(np.mean(crit)),
            "min_Ns": float(np.min(crit)), "resolution_Ns": (hi0 - lo0) / 2 ** RB_BISECT_STEPS}


# ---------------------------------------------------------------------------
# physics episodes (tools/physics scenarios a / b / f / c, shortened)
# ---------------------------------------------------------------------------
STAND_T = 7.0           # stand-up (STAND at ~3 s) + 3.5 s
TROT_T = 7.0            # after T_WALK; metrics over T_WALK + 2 .. T_WALK + 7 (trot, turn)
CRAWL_T = 8.0
SLIP_T = 13.0           # after T_WALK: slip mode from the friction prior, ~1.4 m onto the mu 0.15 plate
SLIP_X_REF = 2.6        # [m] base x at the end of a full-speed (0.2 m/s) slip-mode walk
PUSH_AFTER = 3.5        # [s] simulated after the push starts
PHYS_SCENARIO = {"stand": "a_stand", "trot": "b_trot", "turn": "d_turn", "push": "f_push", "crawl": "c_crawl",
                 "slip": "k_slip_both"}


def phys_episode(kind: str, params: Dict, seed: int, perception: str) -> Dict:
    from physics import scenarios as S
    from physics import sim as SM
    if kind == "slip":
        import physics_check as PC
        sc = PC.all_scenarios()[PHYS_SCENARIO[kind]]
        params = {**getattr(sc, "params", {}), **params}
        # the layout friction prior lives in the terrain mapper: needs a perception pipeline
        # (ground-truth raster while screening, depth ray casting in the confirmation)
        perception = "gt" if perception == "none" else perception
    else:
        sc = S.SCENARIOS[PHYS_SCENARIO[kind]]
    sim = SM.PhysicsSim(S.make_config(sc, seed, params, perception))
    sc.setup(sim, seed)
    if kind == "stand":
        dur = STAND_T
    elif kind in ("trot", "turn"):
        dur = S.T_WALK + TROT_T
    elif kind == "crawl":
        dur = S.T_WALK + CRAWL_T
    elif kind == "slip":
        dur = S.T_WALK + SLIP_T
    else:
        dur = sim.pushes[0].t + PUSH_AFTER
    try:
        sim.run(dur)
    finally:
        sim.close()
    A = sim.arrays()
    t_end = float(A["t"][-1]) if len(A["t"]) else 0.0
    complete = t_end >= dur - 0.02
    falls = len(S.fall_episodes(A))
    m: Dict = {"falls": falls, "complete": bool(complete), "t_end": t_end,
               "diverged": any("diverged" in msg for _, msg in sim.messages)}
    if kind == "stand":
        t_st = S.first_mode_time(A, "STAND")
        m["t_stand"] = t_st
        if math.isfinite(t_st):
            t0, t1 = t_st + 0.5, min(t_st + 3.5, t_end)
            m.update(S.attitude_stats(A, t0, t1))
            w = S.window(A, t0, t1)
            if w.sum() > 2:
                m["drift_mm"] = float(np.linalg.norm(A["pos"][w][-1, :2] - A["pos"][w][0, :2]) * 1e3)
            m["saturation_frac"] = S.torque_stats(A, t0, t1).get("saturation_frac", float("nan"))
    elif kind == "slip":
        x = A["pos"][:, 0]
        m["x_end"] = float(x[-1])
        on = (x > 1.3) & (x < 4.7)
        if on.any():
            sl = A["foot_slip"][on][A["contacts"][on]]
            m["plate_slip_p95"] = float(np.percentile(sl, 95)) if sl.size else float("nan")
            m["plate_time"] = float(on.sum() * np.median(np.diff(A["t"])))
        m.update(S.attitude_stats(A, S.T_WALK, t_end))
        on_t = [e["t"] for e in sim.events if e.get("event") == "slip_mode_on"]
        m["t_slip_mode"] = on_t[0] if on_t else float("nan")
    elif kind in ("trot", "crawl", "turn"):
        t0, t1 = S.T_WALK + 2.0, S.T_WALK + (CRAWL_T if kind == "crawl" else TROT_T)
        vx, vy, wz = S.mean_speed(A, t0, t1)
        m.update({"vx_mean": vx, "vy_mean": vy, "wz_mean": wz})
        m.update(S.attitude_stats(A, t0, t1))
        m.update(S.contact_stats(A, t0, t1))
        tq = S.torque_stats(A, S.T_WALK, t1)
        m.update({k: tq.get(k, float("nan")) for k in ("max_ratio_knee", "saturation_frac")})
        m["track_rms_vx"] = S.vel_tracking(A, t0, t1, 0)
        m["track_rms_vy"] = S.vel_tracking(A, t0, t1, 1)
        m.update({"mpc_" + k: v for k, v in S.mpc_stats(sim).items() if k in ("ms_mean", "rejected")})
        g = A["gait"][S.window(A, S.T_WALK, t1)]
        m["gait_frac"] = float(np.mean(g == ("trot" if kind == "turn" else kind))) if len(g) else 0.0
    else:
        p = sim.pushes[0]
        t0, t1 = p.t, min(p.t + 3.0, t_end)
        m["push_t"] = p.t
        m["push_dir"] = float(np.sign(p.force_body[1]))
        m.update(S.attitude_stats(A, t0, t1))
        w = S.window(A, t0, t1)
        vy = A["vel_body"][w][:, 1]
        m["peak_vy"] = float(np.abs(vy).max()) if vy.size else float("nan")
        big = np.nonzero(np.abs(vy) > 0.15)[0]
        m["recovery_s"] = float(A["t"][w][big[-1]] - p.t) if big.size else 0.0
        m["max_ratio_knee"] = S.torque_stats(A, t0, t1).get("max_ratio_knee", float("nan"))
    return m


def guard_run(key: str, params: Dict, seed: int) -> Dict:
    """One full physics_check scenario run (its own perception, duration and pass criterion)."""
    import physics_check as PC
    r = PC.job((key, seed, params, None, None, False, {"servo": "urdf"}))
    m = {"ok": bool(r.get("ok")), "falls": (r.get("metrics") or {}).get("falls")}
    if "error" in r:
        m["error"] = r["error"]
    return m


def _guard_duration(key: str) -> float:
    try:
        import physics_check as PC
        return float(PC.all_scenarios()[key].duration)
    except Exception:
        return 60.0


def required_scenarios() -> List[str]:
    import physics_check as PC
    return [k for k, sc in PC.all_scenarios().items() if sc.required]


def run_task(t: tuple) -> dict:
    """Worker entry: one (candidate, kind, seed) -> metrics (picklable, never raises)."""
    cid, kind, seed, params, tag = t
    t0 = time.perf_counter()
    try:
        if kind == "rb_track":
            m = rb_track(params)
        elif kind.startswith("pc:"):
            m = guard_run(kind[3:], params, int(seed))
        elif kind in ("rb_push_lat", "rb_push_fwd"):
            m = rb_push(params, kind[-3:], RB_PHASES_SEARCH if seed == "s" else RB_PHASES_CONFIRM)
        else:
            m = phys_episode(kind, params, int(seed), tag)
    except Exception as e:                       # report, never kill the search
        import traceback
        m = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
    return {"key": task_key(t), "metrics": m, "wall": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
W_RB = {"track": 0.35, "attitude": 0.10, "push": 0.40, "solve": 0.15}
W_PHYS = {"stand": 0.10, "trot": 0.25, "turn": 0.15, "push": 0.35, "slip": 0.15}
FAIL = 10.0                              # cost of a failed (fallen / incomplete / erroring) stand or trot run
PUSH_FALL = 6.0                          # cost of a fall after the push (> any recovered push, max 3)


def _f(m: Dict, k: str, default: float = float("nan")) -> float:
    v = m.get(k, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def rb_score(track: Dict, lat: Dict, fwd: Dict) -> Tuple[float, Dict]:
    """Rigid-body cost f_rb and its terms."""
    if "error" in track or "error" in lat or "error" in fwd:
        return math.inf, {"error": True}
    tr, cr = track.get("trot", {}), track.get("crawl", {})
    fallen = bool(tr.get("fallen", True)) or bool(cr.get("fallen", True))
    e_trot = _f(tr, "rms_vx_ca", 1.0) + _f(tr, "rms_vy_ca", 1.0) + 0.3 * _f(tr, "rms_wz", 1.0)
    e_crawl = _f(cr, "rms_vx_ca", 1.0) + _f(cr, "rms_vy_ca", 1.0) + 0.3 * _f(cr, "rms_wz", 1.0)
    tilt = max(_f(tr, "max_tilt_deg", 90.0), _f(cr, "max_tilt_deg", 90.0))
    j_lat, j_fwd = max(_f(lat, "mean_Ns", 0.0), 1.0), max(_f(fwd, "mean_Ns", 0.0), 1.0)
    terms = {
        "track": 0.5 * (min(e_trot, 1.0) / 0.01 + min(e_crawl, 1.0) / 0.01),
        "attitude": min(tilt, 90.0) / 2.0,
        "push": 0.5 * (25.0 / j_lat + 45.0 / j_fwd),
        "solve": _f(track, "solve_cpu_ms", 100.0) / 12.0,
    }
    f = sum(W_RB[k] * v for k, v in terms.items())
    tau = max(_f(tr, "max_torque_ratio", 0.0), _f(cr, "max_torque_ratio", 0.0))
    f += 2.0 * max(0.0, tau - 0.85)
    if fallen:
        f += FAIL
    det = dict(terms, e_trot=e_trot, e_crawl=e_crawl, tilt_deg=tilt, j_lat=j_lat, j_fwd=j_fwd, fallen=fallen,
               solve_cpu_ms=_f(track, "solve_cpu_ms"), solve_wall_ms=_f(track, "solve_wall_ms"),
               qp_iters=_f(track, "qp_iters"), max_torque_ratio=tau, f_rb=f)
    return f, det


def _nanmean(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _nanmax(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def _run_failed(m: Dict) -> bool:
    return "error" in m or m.get("falls", 1) > 0 or not m.get("complete", False) or m.get("diverged", False)


def stand_cost(m: Dict) -> float:
    if _run_failed(m) or not math.isfinite(_f(m, "t_stand")):
        return FAIL
    tilt = max(_f(m, "max_roll_deg", 30.0), _f(m, "max_pitch_deg", 30.0))
    return abs(_f(m, "height_mean", 0.0) - 0.34) / 0.03 + tilt / 3.0 + _f(m, "drift_mm", 500.0) / 50.0


def trot_cost(m: Dict) -> float:
    if _run_failed(m):
        return FAIL
    tilt = max(_f(m, "max_roll_deg", 30.0), _f(m, "max_pitch_deg", 30.0))
    return (abs(_f(m, "vx_mean", 0.0) - 0.4) / 0.04 + (_f(m, "track_rms_vx", 1.0) + _f(m, "track_rms_vy", 1.0)) / 0.05
            + tilt / 5.0 + _f(m, "stance_slip_p95", 1.0) / 0.2)


def turn_cost(m: Dict) -> float:
    if _run_failed(m):
        return FAIL
    tilt = max(_f(m, "max_roll_deg", 30.0), _f(m, "max_pitch_deg", 30.0))
    drift = math.hypot(_f(m, "vx_mean", 1.0), _f(m, "vy_mean", 1.0))
    return abs(_f(m, "wz_mean", 0.0) - 0.5) / 0.05 + drift / 0.05 + tilt / 5.0


def slip_cost(m: Dict) -> float:
    """Slip-mode trot onto the mu 0.15 plate (friction prior): progress, foot slip, attitude."""
    if _run_failed(m):
        return FAIL
    tilt = max(_f(m, "max_roll_deg", 30.0), _f(m, "max_pitch_deg", 30.0))
    return (max(0.0, SLIP_X_REF - _f(m, "x_end", 0.0)) / 0.3 + min(_f(m, "plate_slip_p95", 2.0), 2.0) / 0.4
            + tilt / 5.0)


def push_fell(m: Dict) -> bool:
    return _run_failed(m)


def push_cost(m: Dict) -> float:
    if push_fell(m):
        return PUSH_FALL
    return (min(_f(m, "max_roll_deg", 90.0) / 20.0, 1.0) + min(_f(m, "recovery_s", 9.0) / 3.0, 1.0)
            + min(_f(m, "peak_vy", 9.0) / 0.6, 1.0))


def phys_score(runs: Dict[str, List[Dict]]) -> Tuple[float, Dict]:
    """Physics cost f_phys (seed means per episode kind) and the safety counts."""
    costs = {"stand": stand_cost, "trot": trot_cost, "turn": turn_cost, "push": push_cost, "slip": slip_cost}
    terms = {k: float(np.mean([costs[k](m) for m in runs.get(k, [])])) if runs.get(k) else float("nan")
             for k in costs}
    used = [k for k in costs if runs.get(k)]
    f = sum(W_PHYS[k] * terms[k] for k in used) / sum(W_PHYS[k] for k in used) if used else float("nan")
    st, tr, pu = runs.get("stand", []), runs.get("trot", []), runs.get("push", [])
    tu, sl = runs.get("turn", []), runs.get("slip", [])
    det = dict(terms)
    det.update({
        "f_phys": f,
        "walk_failures": int(sum(_run_failed(m) for m in st + tr + tu + sl)),
        "turn_wz": float(np.mean([_f(m, "wz_mean") for m in tu])) if tu else float("nan"),
        "slip_x_end": float(np.mean([_f(m, "x_end") for m in sl])) if sl else float("nan"),
        "slip_plate_slip_p95": float(_nanmean([_f(m, "plate_slip_p95") for m in sl])) if sl else float("nan"),
        "slip_failures": int(sum(_run_failed(m) for m in sl)),
        "push_falls": int(sum(push_fell(m) for m in pu)), "push_runs": len(pu),
        "trot_vx": float(np.mean([_f(m, "vx_mean") for m in tr])) if tr else float("nan"),
        "trot_track_rms_vx": float(_nanmean([_f(m, "track_rms_vx") for m in tr])) if tr else float("nan"),
        "trot_slip_p95": float(_nanmean([_f(m, "stance_slip_p95") for m in tr])) if tr else float("nan"),
        "trot_tilt_deg": float(_nanmax([max(_f(m, "max_roll_deg"), _f(m, "max_pitch_deg")) for m in tr]))
        if tr else float("nan"),
        "stand_height": float(_nanmean([_f(m, "height_mean") for m in st])) if st else float("nan"),
        "stand_tilt_deg": float(_nanmax([max(_f(m, "max_roll_deg"), _f(m, "max_pitch_deg")) for m in st]))
        if st else float("nan"),
        "push_peak_roll_deg": float(_nanmean([_f(m, "max_roll_deg") for m in pu if not push_fell(m)]))
        if any(not push_fell(m) for m in pu) else float("nan"),
        "push_recovery_s": float(_nanmean([_f(m, "recovery_s") for m in pu if not push_fell(m)]))
        if any(not push_fell(m) for m in pu) else float("nan"),
    })
    if "crawl" in runs:
        cr = runs["crawl"]
        det["crawl_failures"] = int(sum(_run_failed(m) or abs(_f(m, "vx_mean", 0.0) - 0.2) > 0.04
                                        or _f(m, "gait_frac", 0.0) < 0.8 for m in cr))
        det["crawl_vx"] = float(np.mean([_f(m, "vx_mean") for m in cr])) if cr else float("nan")
    return f, det


def pareto_front(points: Sequence[Tuple[float, float]]) -> List[int]:
    """Indices of the non-dominated points (minimisation in both coordinates)."""
    pts = [(float(a), float(b)) for a, b in points]
    front = []
    for i, (a, b) in enumerate(pts):
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        dominated = any((c <= a and d <= b) and (c < a or d < b) for j, (c, d) in enumerate(pts)
                        if j != i and math.isfinite(c) and math.isfinite(d))
        if not dominated:
            front.append(i)
    return front


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation (numpy only)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def decide(base: Dict, cand: Dict, margin: float = 0.10) -> Tuple[bool, List[str]]:
    """Adoption rule on the confirmation data (see the module docstring).
    base / cand: {"F_rb", "F_phys", "rb": rb details, "phys": phys details}."""
    why = []
    if not cand["F_rb"] <= 1.0 - margin:
        why.append(f"F_rb {cand['F_rb']:.3f} > {1 - margin:.2f}")
    if not cand["F_phys"] <= 1.0 - margin:
        why.append(f"F_phys {cand['F_phys']:.3f} > {1 - margin:.2f}")
    ph, pb = cand["phys"], base["phys"]
    if ph.get("walk_failures", 1) > 0:
        why.append(f"{ph.get('walk_failures')} stand/trot/turn/slip failures")
    if ph.get("push_falls", 99) > pb.get("push_falls", 0):
        why.append(f"push falls {ph.get('push_falls')} > defaults {pb.get('push_falls')}")
    if ph.get("crawl_failures", 0) > 0:
        why.append(f"crawl guard failed on {ph.get('crawl_failures')} seeds")
    rc, rb = cand["rb"], base["rb"]
    if rc.get("fallen", True):
        why.append("rigid-body tracking fall")
    for k in ("j_lat", "j_fwd"):
        if rc.get(k, 0.0) < rb.get(k, 0.0) - 1e-9:
            why.append(f"rigid-body {k} {rc.get(k):.1f} < defaults {rb.get(k):.1f} N s")
    return not why, why


# ---------------------------------------------------------------------------
# candidates + evaluator
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    cid: str
    origin: str                  # defaults | lhs | cma<g>
    u: np.ndarray
    values: Dict[str, float]
    rung: int = -1
    order: int = 0

    @property
    def params(self) -> Dict:
        return to_params(self.values)


def make_candidate(origin: str, u: Sequence[float], order: int) -> Candidate:
    vals = decode(u)
    u_q = encode(vals)                           # the rounded values define the candidate
    h = hashlib.sha1(json.dumps(vals, sort_keys=True).encode()).hexdigest()[:10]
    return Candidate(h, origin, u_q, vals, order=order)


class Evaluator:
    """Runs tasks in a spawn process pool (or inline) and caches the metrics."""

    def __init__(self, jobs: int, runner: Callable[[tuple], dict] = run_task, log: Callable = print):
        self.jobs = max(1, int(jobs))
        self.runner = runner
        self.cache: Dict[tuple, Dict] = {}
        self.walls: Dict[str, List[float]] = {}
        self.log = log
        self.pool = None
        self.task_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        if self.jobs > 1:
            import multiprocessing as mp
            self.pool = mp.get_context("spawn").Pool(self.jobs)
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def run(self, tasks: Iterable[tuple], label: str = "") -> float:
        todo, seen = [], set()
        for t in tasks:
            k = task_key(t)
            if k not in self.cache and k not in seen:
                seen.add(k)
                todo.append(t)
        # long tasks first (better packing)
        order = {"rb_push_lat": 0, "rb_push_fwd": 0, "push": 1, "crawl": 1, "trot": 2, "rb_track": 2, "stand": 3}
        todo.sort(key=lambda t: order.get(t[1], 9))
        t0 = time.perf_counter()
        n = 0
        it = (self.pool.imap_unordered(self.runner, todo) if self.pool is not None and len(todo) > 1
              else map(self.runner, todo))
        for r in it:
            self.cache[r["key"]] = r["metrics"]
            self.walls.setdefault(r["key"][1], []).append(r["wall"])
            self.task_time += r["wall"]
            n += 1
            if "error" in r["metrics"]:
                self.log(f"    task {r['key']} ERROR {r['metrics']['error']}")
        dt = time.perf_counter() - t0
        self.elapsed += dt
        if label:
            self.log(f"  {label}: {n} tasks in {dt:.0f} s")
        return dt

    def speedup(self) -> float:
        return self.task_time / self.elapsed if self.elapsed > 0 else float(self.jobs)

    def estimate(self, tasks: Iterable[tuple], depth_factor: float = 1.0) -> float:
        """Wall-time estimate of a task list from the measured per-kind task times
        (``depth_factor``: extra cost of the depth-camera perception, not measured yet;
        unmeasured guard runs: their simulated duration)."""
        tot = 0.0
        for t in tasks:
            w = self.walls.get(t[1])
            base = float(np.mean(w)) if w else (_guard_duration(t[1][3:]) if t[1].startswith("pc:") else 10.0)
            tot += base * (depth_factor if t[4] == "depth" else 1.0)
        return tot / max(self.speedup(), 1.0)

    def get(self, cid: str, kind: str, seed, tag: str) -> Optional[Dict]:
        return self.cache.get((cid, kind, seed, tag))


def rb_of(ev: Evaluator, c: Candidate, plan: str) -> Tuple[float, Dict]:
    pl = PLANS[plan]
    tr = ev.get(c.cid, "rb_track", pl["rb_track"][0], "rb")
    la = ev.get(c.cid, "rb_push_lat", pl["rb_push_lat"][0], "rb")
    fw = ev.get(c.cid, "rb_push_fwd", pl["rb_push_fwd"][0], "rb")
    if tr is None or la is None or fw is None:
        return math.nan, {}
    return rb_score(tr, la, fw)


def phys_of(ev: Evaluator, c: Candidate, plan: str, perception: str) -> Tuple[float, Dict]:
    runs = {}
    for kind, seeds in PLANS[plan].items():
        ms = [ev.get(c.cid, kind, sd, perception) for sd in seeds]
        if any(m is None for m in ms):
            return math.nan, {}
        runs[kind] = ms
    return phys_score(runs)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    seed: int = 0
    n_explore: int = 80
    keep_rung1: int = 32
    keep_rung2: int = 12
    cma_gens: int = 4
    cma_pop: int = 12
    cma_sigma: float = 0.12
    cma_promote: int = 4
    confirm_top: int = 4
    adopt_margin: float = 0.10
    budget_min: float = 38.0
    jobs: int = 12
    search_perception: str = SEARCH_PERCEPTION
    confirm_perception: str = CONFIRM_PERCEPTION
    guard_scenarios: Optional[Tuple[str, ...]] = None      # None: every required physics_check scenario
    guard_seeds: Tuple[int, ...] = GUARD_SEEDS


def run_search(st: Settings, ev: Evaluator, log: Callable = print) -> Dict:
    t_start = time.perf_counter()
    rng = np.random.default_rng(st.seed)
    timeline = {}
    cands: List[Candidate] = []
    by_id: Dict[str, Candidate] = {}

    def add(c: Candidate) -> Candidate:
        if c.cid in by_id:
            return by_id[c.cid]
        c.order = len(cands)
        cands.append(c)
        by_id[c.cid] = c
        return c

    base = add(make_candidate("defaults", encode(default_values()), 0))
    for u in latin_hypercube(st.n_explore, len(SPACE), rng):
        add(make_candidate("lhs", u, 0))
    sp, cp = st.search_perception, st.confirm_perception

    def F_rb(c, plan="rung0"):
        f, _ = rb_of(ev, c, plan)
        fb, _ = rb_of(ev, base, plan)
        return f / fb if math.isfinite(fb) and fb > 0 else math.nan

    def F_phys(c, plan, perception):
        f, _ = phys_of(ev, c, plan, perception)
        fb, _ = phys_of(ev, base, plan, perception)
        return f / fb if math.isfinite(f) and math.isfinite(fb) and fb > 0 else math.nan

    def J(c, plan, perception=sp):
        a, b = F_rb(c), F_phys(c, plan, perception)
        return 0.5 * (a + b) if math.isfinite(a) and math.isfinite(b) else math.inf

    def safe(c, plan):
        _, d = phys_of(ev, c, plan, sp)
        return bool(d) and d.get("walk_failures", 1) == 0

    def rb_ok(c):
        f, d = rb_of(ev, c, "rung0")
        return math.isfinite(f) and not d.get("fallen", True)

    # ---------------- rung 0: rigid body, every candidate
    log(f"[rung 0] rigid-body suite for {len(cands)} candidates")
    ev.run([t for c in cands for t in plan_tasks(c.cid, c.params, PLANS["rung0"], sp)], "rung 0")
    for c in cands:
        c.rung = 0
    timeline["rung0"] = time.perf_counter() - t_start
    pool = sorted([c for c in cands if c is not base and rb_ok(c)], key=lambda c: F_rb(c))
    r1 = [base] + pool[: st.keep_rung1]
    log(f"  feasible {len(pool)}/{len(cands) - 1}; best F_rb {F_rb(pool[0]) if pool else float('nan'):.3f}")

    # ---------------- rung 1: physics, short
    log(f"[rung 1] physics (stand, trot, turn, 2 pushes, slip mode) for {len(r1)} candidates")
    ev.run([t for c in r1 for t in plan_tasks(c.cid, c.params, PLANS["rung1"], sp)], "rung 1")
    for c in r1:
        c.rung = 1
    timeline["rung1"] = time.perf_counter() - t_start
    ranked = sorted([c for c in r1 if c is not base and safe(c, "rung1")], key=lambda c: J(c, "rung1"))
    r2 = [base] + ranked[: st.keep_rung2]

    # ---------------- rung 2: physics, 3 seeds + 6 pushes
    log(f"[rung 2] physics (3 seeds, 6 pushes) for {len(r2)} candidates")
    guard_keys = list(st.guard_scenarios) if st.guard_scenarios is not None else required_scenarios()
    guard_plan = {f"pc:{k}": list(st.guard_seeds) for k in guard_keys}
    ev.run([t for c in r2 for t in plan_tasks(c.cid, c.params, PLANS["rung2"], sp)], "rung 2")
    for c in r2:
        c.rung = 2
    timeline["rung2"] = time.perf_counter() - t_start

    def J2(c):
        return J(c, "rung2")

    # ---------------- CMA-ES refinement
    start = min([c for c in r2 if safe(c, "rung2")] or [base], key=J2)
    log(f"[cma] start {start.cid} ({start.origin}) J2 {J2(start):.3f}")
    es = CMAES(start.u, st.cma_sigma, st.cma_pop, rng)
    cma_hist = []
    budget = st.budget_min * 60.0
    tail_promote = [(None, k, sd, None, sp) for k, sds in PLANS["rung2"].items() for sd in sds] * st.cma_promote
    tail_conf = ([(None, k, sd, None, cp) for k, sds in PLANS["confirm"].items() for sd in sds]
                 + [(None, k, "c", None, "rb") for k in ("rb_push_lat", "rb_push_fwd")]) * (st.confirm_top + 1)
    tail_guard = [(None, k, sd, None, "pc") for k, sds in guard_plan.items() for sd in sds] * 2
    t_gen = None
    for g in range(st.cma_gens):
        elapsed = time.perf_counter() - t_start
        est_gen = t_gen if t_gen is not None else ev.estimate(
            [(None, k, sd, None, "rb" if k.startswith("rb_") else sp)
             for k, sds in {**PLANS["rung0"], **PLANS["rung1"]}.items() for sd in sds] * st.cma_pop)
        est_tail = (ev.estimate(tail_promote) + ev.estimate(tail_conf, depth_factor=1.5)
                    + ev.estimate(tail_guard))
        if elapsed + est_gen + est_tail > budget:
            log(f"  budget: stop before generation {g} (elapsed {elapsed / 60:.1f} min, next generation "
                f"~{est_gen / 60:.1f} min, promotion + confirmation ~{est_tail / 60:.1f} min)")
            break
        tg = time.perf_counter()
        X = es.ask()
        kids = [add(make_candidate(f"cma{g}", x, 0)) for x in X]
        tasks = []
        for c in kids:
            tasks += plan_tasks(c.cid, c.params, PLANS["rung0"], sp)
            tasks += plan_tasks(c.cid, c.params, PLANS["rung1"], sp)
        ev.run(tasks, f"cma gen {g}")
        for c in kids:
            c.rung = max(c.rung, 1)
        fit = [J(c, "rung1") if rb_ok(c) and safe(c, "rung1") else 10.0 + J(c, "rung1") for c in kids]
        es.tell(np.array([c.u for c in kids]), fit)
        cma_hist.append({"gen": g, "sigma": es.sigma, "best_J1": float(np.min(fit)),
                         "median_J1": float(np.median(fit)), "mean_u": es.mean.tolist()})
        log(f"  gen {g}: best J1 {np.min(fit):.3f} median {np.median(fit):.3f} sigma {es.sigma:.3f}")
        t_gen = time.perf_counter() - tg
    timeline["cma"] = time.perf_counter() - t_start
    cma_kids = [c for c in cands if c.origin.startswith("cma") and rb_ok(c) and safe(c, "rung1")]
    promote = sorted(cma_kids, key=lambda c: J(c, "rung1"))[: st.cma_promote]
    if promote:
        log(f"[rung 2] {len(promote)} CMA-ES candidates")
        ev.run([t for c in promote for t in plan_tasks(c.cid, c.params, PLANS["rung2"], sp)], "rung 2 (cma)")
        for c in promote:
            c.rung = 2
    timeline["rung2_cma"] = time.perf_counter() - t_start

    # ---------------- Pareto front of the rung-2 candidates
    top = [c for c in cands if c.rung >= 2]
    pb = phys_of(ev, base, "rung2", sp)[1]
    feas = [c for c in top if safe(c, "rung2")
            and phys_of(ev, c, "rung2", sp)[1].get("push_falls", 99) <= pb.get("push_falls", 0)]
    pts = [(F_rb(c), F_phys(c, "rung2", sp)) for c in feas]
    front = [feas[i] for i in pareto_front(pts)]
    front_ids = {c.cid for c in front}
    order = sorted(front, key=J2) + sorted([c for c in feas if c.cid not in front_ids], key=J2)
    conf = [c for c in order if c is not base][: st.confirm_top]
    log(f"[pareto] {len(front)} non-dominated of {len(feas)} feasible rung-2 candidates; confirming "
        f"{[c.cid for c in conf]}")

    # ---------------- confirmation on held-out data
    confirm_set = [base] + conf
    tasks = []
    for c in confirm_set:
        tasks += plan_tasks(c.cid, c.params, PLANS["confirm_rb"], sp)
        tasks += plan_tasks(c.cid, c.params, PLANS["confirm"], cp)
    ev.run(tasks, "confirmation")
    timeline["confirm"] = time.perf_counter() - t_start
    fb_rb, rb_b = rb_of(ev, base, "confirm_rb")
    fb_ph, ph_b = phys_of(ev, base, "confirm", cp)
    confirmed = []
    for c in confirm_set:
        f_rb, d_rb = rb_of(ev, c, "confirm_rb")
        f_ph, d_ph = phys_of(ev, c, "confirm", cp)
        row = {"cid": c.cid, "origin": c.origin, "F_rb": f_rb / fb_rb, "F_phys": f_ph / fb_ph, "rb": d_rb,
               "phys": d_ph}
        row["J"] = 0.5 * (row["F_rb"] + row["F_phys"])
        if c is not base:
            row["adopt"], row["reasons"] = decide({"rb": rb_b, "phys": ph_b}, row, st.adopt_margin)
        confirmed.append(row)
        log(f"  {c.cid} {c.origin:8s} F_rb {row['F_rb']:.3f} F_phys {row['F_phys']:.3f} J {row['J']:.3f} "
            f"push falls {d_ph.get('push_falls')}/{d_ph.get('push_runs')} "
            f"crawl fail {d_ph.get('crawl_failures')} "
            f"{'decide() ok -> guard' if row.get('adopt') else row.get('reasons', '')}")
    # ---------------- regression guard: full physics_check required scenarios (defaults vs candidate)
    def guard_counts(c):
        return {k[3:]: int(sum(bool((ev.get(c.cid, k, sd, "pc") or {}).get("ok")) for sd in sds))
                for k, sds in guard_plan.items()}

    passing = sorted([r for r in confirmed[1:] if r.get("adopt")], key=lambda r: r["J"])
    guard = {"scenarios": guard_keys, "seeds": list(st.guard_seeds), "passes": {}}
    adopted = None
    for r in passing:
        r["adopt"] = False                       # only a passed guard adopts
        if not guard_keys:
            r["reasons"] = ["guard: no scenarios"]
            continue
        need = ([] if base.cid in guard["passes"] else plan_tasks(base.cid, {}, guard_plan, "pc")) + \
            plan_tasks(r["cid"], {}, guard_plan, "pc")
        elapsed = time.perf_counter() - t_start
        est = ev.estimate(need)
        if elapsed + est > budget:
            r["reasons"] = [f"guard not run: budget ({elapsed / 60:.1f} + ~{est / 60:.1f} min > {st.budget_min} min)"]
            log(f"  {r['cid']}: {r['reasons'][0]}")
            continue
        c = by_id[r["cid"]]
        if base.cid not in guard["passes"]:
            ev.run(plan_tasks(base.cid, base.params, guard_plan, "pc"), "guard (defaults)")
            guard["passes"][base.cid] = guard_counts(base)
        ev.run(plan_tasks(c.cid, c.params, guard_plan, "pc"), f"guard ({c.cid})")
        gc = guard["passes"][c.cid] = guard_counts(c)
        gb = guard["passes"][base.cid]
        n = len(st.guard_seeds)
        worse = [f"{k} {gc[k]}/{n} < defaults {gb[k]}/{n}" for k in guard_keys if gc[k] < gb[k]]
        log(f"  guard {c.cid}: {sum(gc.values())}/{n * len(guard_keys)} runs pass (defaults {sum(gb.values())})"
            + (f"; regressions: {worse}" if worse else ""))
        if worse:
            r["reasons"] = ["guard: " + w for w in worse]
            continue
        r["adopt"] = True
        adopted = r
        break
    timeline["guard"] = time.perf_counter() - t_start
    best = adopted or (min(confirmed[1:], key=lambda r: r["J"]) if confirmed[1:] else None)
    decision = {"adopted": adopted is not None, "margin": st.adopt_margin, "best_cid": best["cid"] if best else None,
                "guard": guard}
    if best is not None:
        bc = by_id[best["cid"]]
        decision["changes"] = {k: [default_values()[k], v] for k, v in bc.values.items()
                               if abs(v - default_values()[k]) > 1e-12}
        decision["params"] = bc.params
        decision["reasons"] = best.get("reasons", [])
    timeline["total"] = time.perf_counter() - t_start

    # ---------------- report data
    rows = []
    for c in cands:
        r = {"cid": c.cid, "origin": c.origin, "order": c.order, "rung": c.rung, "values": c.values,
             "u": [round(float(x), 4) for x in c.u]}
        f, d = rb_of(ev, c, "rung0")
        if d:
            r["F_rb"] = F_rb(c)
            r["rb"] = {k: d[k] for k in ("e_trot", "e_crawl", "tilt_deg", "j_lat", "j_fwd", "solve_cpu_ms",
                                         "qp_iters", "fallen", "f_rb")}
        if c.rung >= 1:
            r["F_phys_rung1"] = F_phys(c, "rung1", sp)
            r["J_rung1"] = J(c, "rung1")
        if c.rung >= 2:
            _, d2 = phys_of(ev, c, "rung2", sp)
            r["F_phys_rung2"] = F_phys(c, "rung2", sp)
            r["J_rung2"] = J2(c)
            r["phys_rung2"] = {k: d2.get(k) for k in ("stand", "trot", "turn", "push", "slip", "f_phys",
                                                     "walk_failures", "push_falls", "push_runs", "trot_vx",
                                                     "trot_slip_p95", "turn_wz", "slip_x_end",
                                                     "push_peak_roll_deg")}
        rows.append(r)
    sens = {}
    r0 = [c for c in cands if c.rung >= 0 and c.origin == "lhs"]
    r1s = [c for c in cands if c.rung >= 1 and c is not base]
    for j, p in enumerate(SPACE):
        sens[p.name] = {"F_rb": spearman([c.u[j] for c in r0], [F_rb(c) if rb_ok(c) else np.nan for c in r0]),
                        "F_phys_rung1": spearman([c.u[j] for c in r1s], [F_phys(c, "rung1", sp) for c in r1s])}
    return {"candidates": rows, "pareto": [c.cid for c in front], "feasible_rung2": [c.cid for c in feas],
            "confirmed": confirmed, "decision": decision, "cma": cma_hist, "sensitivity": sens,
            "timeline_s": timeline, "baseline": base.cid, "evaluator": {
                "tasks": len(ev.cache), "task_time_s": ev.task_time, "speedup": ev.speedup(),
                "mean_task_s": {k: float(np.mean(v)) for k, v in ev.walls.items()},
                "errors": sum(1 for m in ev.cache.values() if "error" in m)}}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(res: Dict, out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physics.plots import AQUA, BLUE, GRID, INK, INK2, ORANGE, SURFACE, _style
    os.makedirs(out_dir, exist_ok=True)
    files = []
    rows = {r["cid"]: r for r in res["candidates"]}
    base = res["baseline"]
    conf = {r["cid"]: r for r in res["confirmed"]}
    best = res["decision"].get("best_cid")

    # ---- Pareto
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": (1.25, 1.0)})
    ax = axs[0]
    _style(ax)
    r1 = [r for r in rows.values() if r["rung"] == 1 and r["cid"] != base and np.isfinite(r.get("F_rb", np.nan))
          and np.isfinite(r.get("F_phys_rung1", np.nan))]
    ax.scatter([r["F_rb"] for r in r1], [r["F_phys_rung1"] for r in r1], s=14, color=GRID, edgecolor=INK2,
               linewidth=0.4, label="rung 1 (physics: 1 seed per episode, 2 pushes)", zorder=2)
    for origin, col, lab in (("lhs", BLUE, "rung 2, Latin hypercube"), ("cma", ORANGE, "rung 2, CMA-ES")):
        r2 = [r for r in rows.values() if r["rung"] >= 2 and r["origin"].startswith(origin)]
        ax.scatter([r["F_rb"] for r in r2], [r["F_phys_rung2"] for r in r2], s=46, color=col,
                   edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=3)
    fr = sorted([rows[c] for c in res["pareto"]], key=lambda r: r["F_rb"])
    if fr:
        xs = [r["F_rb"] for r in fr]
        ys = [r["F_phys_rung2"] for r in fr]
        ax.step(xs, ys, where="post", color=INK, lw=1.2, zorder=4, label="Pareto front (rung 2, feasible)")
    ax.scatter([1.0], [1.0], marker="*", s=220, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=5,
               label="current defaults")
    if best and best in rows and rows[best].get("F_phys_rung2") is not None:
        ax.scatter([rows[best]["F_rb"]], [rows[best]["F_phys_rung2"]], s=180, facecolor="none", edgecolor=AQUA,
                   linewidth=2.2, zorder=6, label="best confirmed candidate")
    ax.axvline(1.0, color=INK2, lw=0.7, ls=":")
    ax.axhline(1.0, color=INK2, lw=0.7, ls=":")
    allx = [r["F_rb"] for r in r1] + [r.get("F_rb", 1) for r in rows.values() if r["rung"] >= 2] + [1.0]
    ally = [r["F_phys_rung1"] for r in r1] + [r.get("F_phys_rung2", 1) for r in rows.values() if r["rung"] >= 2]
    ax.set_xlim(max(0.0, min(allx) - 0.05), min(max(allx) + 0.05, 2.5))
    ax.set_ylim(max(0.0, min(ally + [1.0]) - 0.05), min(max(ally + [1.0]) + 0.05, 3.0))
    ax.text(0.01, 0.01, "lower-left = better than the defaults on both", transform=ax.transAxes, fontsize=7,
            color=INK2)
    ax.set_xlabel("F_rb  (rigid-body score / defaults)", fontsize=8, color=INK2)
    ax.set_ylabel("F_phys  (MuJoCo score / defaults)", fontsize=8, color=INK2)
    ax.set_title("search: rigid-body vs physics score (lower is better)", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    # confirmation (held-out data)
    ax = axs[1]
    _style(ax)
    crow = res["confirmed"]
    labels = [("defaults" if r["cid"] == base else f"{r['cid'][:6]} ({r['origin']})") for r in crow]
    y = np.arange(len(crow))[::-1]
    ax.barh(y + 0.18, [r["F_rb"] for r in crow], height=0.34, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
            label="F_rb (held-out push phases)")
    ax.barh(y - 0.18, [r["F_phys"] for r in crow], height=0.34, color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
            label="F_phys (held-out seeds, depth perception)")
    m = res["decision"]["margin"]
    ax.axvline(1.0, color=INK, lw=0.9)
    ax.axvline(1.0 - m, color=INK2, lw=0.8, ls="--")
    xmax = max([1.0] + [v for r in crow for v in (r["F_rb"], r["F_phys"]) if np.isfinite(v)])
    ax.text(1.0 - m, -0.75, f"adoption limit {1 - m:.2f} ", fontsize=6.5, color=INK2, ha="right", va="center")
    gpass = res["decision"].get("guard", {}).get("passes", {})
    gtot = len(res["decision"].get("guard", {}).get("scenarios", [])) * len(res["decision"].get("guard", {})
                                                                          .get("seeds", []))
    for yi, r in zip(y, crow):
        ph = r["phys"]
        txt = f"push falls {ph.get('push_falls')}/{ph.get('push_runs')}"
        if r["cid"] in gpass:
            txt += f", guard {sum(gpass[r['cid']].values())}/{gtot}"
        if r["cid"] != base:
            txt += ", ADOPTED" if r.get("adopt") else ", rejected"
        ax.text(max(r["F_rb"], r["F_phys"]) + 0.03, yi, txt, fontsize=6.3, color=INK, va="center")
    ax.set_xlim(0.0, xmax * 1.45)
    ax.set_ylim(-1.0, len(crow) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("score / defaults (lower is better)", fontsize=8, color=INK2)
    ax.set_title("confirmation on held-out data", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    p = os.path.join(out_dir, "tuning_pareto.png")
    fig.savefig(p, dpi=110, facecolor="white")
    plt.close(fig)
    files.append(p)

    # ---- sensitivity + parameter positions
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    names = [q.name for q in SPACE]
    y = np.arange(len(names))[::-1]
    ax = axs[0]
    _style(ax)
    s_rb = [res["sensitivity"][n]["F_rb"] for n in names]
    s_ph = [res["sensitivity"][n]["F_phys_rung1"] for n in names]
    ax.barh(y + 0.18, np.nan_to_num(s_rb), height=0.34, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
            label="vs F_rb (rung 0, Latin hypercube)")
    ax.barh(y - 0.18, np.nan_to_num(s_ph), height=0.34, color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
            label="vs F_phys (rung 1)")
    ax.axvline(0.0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rank correlation with the score (> 0: larger value = worse)", fontsize=8, color=INK2)
    ax.set_title("parameter sensitivity", fontsize=9, loc="left", color=INK, pad=16)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax = axs[1]
    _style(ax)
    u_def = encode(default_values())
    ax.scatter(u_def, y, marker="*", s=120, color=INK, zorder=5, label="defaults")
    other_labeled = False
    for k, cid in enumerate([r["cid"] for r in crow if r["cid"] != base]):
        u = np.asarray(rows[cid]["u"])
        is_best = cid == best
        lab = "best confirmed" if is_best else (None if other_labeled else "other confirmed")
        other_labeled = other_labeled or not is_best
        ax.scatter(u, y + (0.0 if is_best else 0.12 * (k + 1) - 0.25), s=40 if is_best else 18,
                   color=AQUA if is_best else INK2, edgecolor=SURFACE, linewidth=1.0, zorder=6 if is_best else 4,
                   label=lab)
    for yi, q in zip(y, SPACE):
        ax.text(1.02, yi, f"{q.lo:g}..{q.hi:g}{' (log)' if q.log else ''}", fontsize=6, color=INK2, va="center",
                transform=ax.get_yaxis_transform())
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("normalised value in the search range", fontsize=8, color=INK2)
    ax.set_title("confirmed candidates vs defaults", fontsize=9, loc="left", color=INK, pad=16)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
              borderaxespad=0.1)
    fig.tight_layout()
    p = os.path.join(out_dir, "tuning_sensitivity.png")
    fig.savefig(p, dpi=110, facecolor="white")
    plt.close(fig)
    files.append(p)

    # ---- progress + confirmed metrics
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": (1.15, 1.0)})
    ax = axs[0]
    _style(ax)
    ev = sorted([r for r in rows.values() if r["rung"] >= 1 and r["cid"] != base], key=lambda r: r["order"])
    xs = np.arange(len(ev))
    js = np.array([r.get("J_rung1", np.nan) for r in ev], dtype=float)
    cols = [ORANGE if r["origin"].startswith("cma") else BLUE for r in ev]
    ax.scatter(xs, np.clip(js, 0, 3), s=16, c=cols, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    run_best = np.minimum.accumulate(np.where(np.isfinite(js), js, np.inf))
    ax.plot(xs, np.clip(run_best, 0, 3), color=INK, lw=1.4, label="best so far")
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.scatter([], [], s=16, color=BLUE, label="Latin hypercube (after rung 0)")
    ax.scatter([], [], s=16, color=ORANGE, label="CMA-ES children")
    ax.set_xlabel("candidates with physics evaluation (in evaluation order)", fontsize=8, color=INK2)
    ax.set_ylabel("J (rung 1) = (F_rb + F_phys) / 2", fontsize=8, color=INK2)
    ax.set_title("search progress (1 = defaults)", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax = axs[1]
    _style(ax)
    b = conf.get(base)
    bb = conf.get(best) if best else None
    if b and bb:
        keys = [("rb", "e_trot", "rb trot tracking err", False), ("rb", "e_crawl", "rb crawl tracking err", False),
                ("rb", "j_lat", "rb lateral crit. impulse", True), ("rb", "j_fwd", "rb forward crit. impulse", True),
                ("rb", "solve_cpu_ms", "MPC solve CPU time", False),
                ("phys", "trot_track_rms_vx", "trot vx tracking rms", False),
                ("phys", "trot_slip_p95", "trot stance slip p95", False),
                ("phys", "trot_tilt_deg", "trot max tilt", False),
                ("phys", "turn_err", "turn rate error |wz - 0.5|", False),
                ("phys", "slip_x_end", "slip-mode progress", True),
                ("phys", "push_peak_roll_deg", "push peak roll", False),
                ("phys", "push_recovery_s", "push recovery time", False),
                ("phys", "stand_tilt_deg", "stand max tilt", False)]
        labs, vals, txt = [], [], []
        for d in (b, bb):
            wz = d["phys"].get("turn_wz")
            d["phys"]["turn_err"] = abs(wz - 0.5) if wz is not None and np.isfinite(wz) else None
        for sec, k, lab, higher_better in keys:
            vb, vc = b[sec].get(k), bb[sec].get(k)
            if vb is None or vc is None or not (np.isfinite(vb) and np.isfinite(vc)) or vb == 0 or vc == 0:
                continue
            ratio = vb / vc if higher_better else vc / vb
            labs.append(lab)
            vals.append(ratio)
            txt.append(f"{vb:.3g} -> {vc:.3g}")
        yy = np.arange(len(labs))[::-1]
        ax.barh(yy, vals, height=0.55, color=[AQUA if v <= 1 else ORANGE for v in vals], edgecolor=SURFACE,
                linewidth=1.5)
        for yi, v, tx in zip(yy, vals, txt):
            ax.text(max(v, 0) + 0.02, yi, tx, fontsize=6.3, color=INK, va="center")
        ax.axvline(1.0, color=INK, lw=0.9)
        ax.set_yticks(yy)
        ax.set_yticklabels(labs, fontsize=7)
        ax.set_xlim(0, max(vals + [1.0]) * 1.45)
        ax.set_xlabel("best candidate relative to the defaults (< 1 = better; held-out data)", fontsize=8,
                      color=INK2)
        pf = (f"push falls {b['phys'].get('push_falls')}/{b['phys'].get('push_runs')} -> "
              f"{bb['phys'].get('push_falls')}/{bb['phys'].get('push_runs')}")
        ax.set_title(f"best confirmed {best[:6]} vs defaults: "
                     f"{'ADOPTED' if res['decision']['adopted'] else 'not adopted'}", fontsize=9, loc="left",
                     color=INK)
        ax.text(0.98, 0.02, pf, transform=ax.transAxes, fontsize=7, color=INK, ha="right", va="bottom")
    fig.tight_layout()
    p = os.path.join(out_dir, "tuning_progress.png")
    fig.savefig(p, dpi=110, facecolor="white")
    plt.close(fig)
    files.append(p)
    return files


# ---------------------------------------------------------------------------
def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 6) if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:                           # other drive (scratch output)
        return path.replace("\\", "/")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    st = Settings()
    ap.add_argument("--seed", type=int, default=st.seed, help="sampling / CMA-ES seed")
    ap.add_argument("--n-explore", type=int, default=st.n_explore, help="Latin-hypercube samples")
    ap.add_argument("--keep-rung1", type=int, default=st.keep_rung1, help="rung 0 -> 1 survivors")
    ap.add_argument("--keep-rung2", type=int, default=st.keep_rung2, help="rung 1 -> 2 survivors")
    ap.add_argument("--cma-gens", type=int, default=st.cma_gens)
    ap.add_argument("--cma-pop", type=int, default=st.cma_pop)
    ap.add_argument("--cma-sigma", type=float, default=st.cma_sigma, help="initial step (normalised space)")
    ap.add_argument("--cma-promote", type=int, default=st.cma_promote, help="CMA-ES children -> rung 2")
    ap.add_argument("--confirm-top", type=int, default=st.confirm_top, help="candidates confirmed on held-out data")
    ap.add_argument("--adopt-margin", type=float, default=st.adopt_margin)
    ap.add_argument("--budget-min", type=float, default=st.budget_min,
                    help="wall-clock budget [min]; CMA-ES generations are skipped to keep it")
    ap.add_argument("--jobs", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--quick", action="store_true", help="smoke run: 6 samples, 1 CMA-ES generation, no depth")
    ap.add_argument("--out", default=None, help=f"result JSON (default {os.path.relpath(RESULTS, ROOT)}; "
                    "--quick: <tmp>/tune_locomotion_quick/)")
    ap.add_argument("--images", default=None, help="figure directory (default docs/images; --quick: as --out)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--previous", default="", help="earlier tuning.json: its decision / confirmation is kept as "
                                                   "previous_search")
    ap.add_argument("--note", default="", help="free text stored in the JSON")
    ap.add_argument("--replot", default="", help="only redraw the figures of an existing tuning.json")
    a = ap.parse_args(argv)
    if a.replot:
        with open(a.replot, encoding="utf-8") as f:
            res = json.load(f)
        for p in make_figures(res, a.images or IMAGES):
            print("wrote", p)
        return 0
    st = Settings(seed=a.seed, n_explore=a.n_explore, keep_rung1=a.keep_rung1, keep_rung2=a.keep_rung2,
                  cma_gens=a.cma_gens, cma_pop=a.cma_pop, cma_sigma=a.cma_sigma, cma_promote=a.cma_promote,
                  confirm_top=a.confirm_top, adopt_margin=a.adopt_margin, budget_min=a.budget_min, jobs=a.jobs)
    if a.quick and (a.out is None or a.images is None):      # a smoke run never overwrites the docs
        import tempfile
        qdir = os.path.join(tempfile.gettempdir(), "tune_locomotion_quick")
        a.out = a.out or os.path.join(qdir, "tuning.json")
        a.images = a.images or qdir
    a.out = a.out or RESULTS
    a.images = a.images or IMAGES
    if a.quick:
        st.n_explore, st.keep_rung1, st.keep_rung2 = 6, 3, 2
        st.cma_gens, st.cma_pop, st.cma_promote, st.confirm_top = 1, 4, 1, 1
        st.confirm_perception = "none"
        st.guard_scenarios, st.guard_seeds = ("a_stand", "b_trot"), (0,)
    try:
        from module1_locomotion import qp as Q
        Q.limit_blas_threads(1)
    except Exception:
        pass
    t0 = time.perf_counter()
    print(f"tune_locomotion: {len(SPACE)} parameters, seed {st.seed}, {st.jobs} workers, budget {st.budget_min} min")
    with Evaluator(st.jobs) as ev:
        res = run_search(st, ev)
    out = {
        "tool": "tools/tune_locomotion.py",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "command": "python tools/tune_locomotion.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "settings": st.__dict__,
        "method": ("Latin hypercube exploration + successive halving over 3 fidelity rungs (rigid body -> MuJoCo "
                   "1 seed -> MuJoCo 3 seeds / 6 pushes) + CMA-ES refinement; Pareto front of (F_rb, F_phys); "
                   "confirmation on held-out seeds / push phases with depth perception; adoption rule decide()"),
        "scores": {"W_RB": W_RB, "W_PHYS": W_PHYS, "FAIL": FAIL, "PUSH_FALL": PUSH_FALL,
                   "F": "score / score of the current defaults on the same data; J = (F_rb + F_phys) / 2"},
        "plans": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in PLANS.items()},
        "rb_push": {"phases_search": RB_PHASES_SEARCH, "phases_confirm": RB_PHASES_CONFIRM,
                    "range_Ns": RB_PUSH_RANGE, "bisect_steps": RB_BISECT_STEPS},
        "space": [{"name": p.name, "lo": p.lo, "hi": p.hi, "log": p.log, "group": p.group,
                   "phys_only": p.phys_only, "default": default_values()[p.name], "doc": p.doc} for p in SPACE],
        "defaults": default_values(),
        "note": a.note,
        **res,
    }
    if a.previous and os.path.isfile(a.previous):
        with open(a.previous, encoding="utf-8") as f:
            prev = json.load(f)
        keep = ("cid", "origin", "F_rb", "F_phys", "J", "adopt", "reasons")
        out["previous_search"] = {
            "date": prev.get("date"), "command": prev.get("command"), "note": prev.get("note", ""),
            "settings": prev.get("settings"), "decision": prev.get("decision"),
            "confirmed": [{**{k: r.get(k) for k in keep},
                           "push_falls": (r.get("phys") or {}).get("push_falls"),
                           "push_runs": (r.get("phys") or {}).get("push_runs")} for r in prev.get("confirmed", [])],
            "runtime_min": prev.get("runtime_min")}
    figs = []
    if not a.no_figures:
        figs = make_figures(out, a.images)
    out["figures"] = [_rel(f) for f in figs]
    out["runtime_min"] = (time.perf_counter() - t0) / 60.0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_clean(out), f, indent=1)
        f.write("\n")
    d = out["decision"]
    print(f"\ndecision: {'ADOPT ' + str(d.get('best_cid')) if d['adopted'] else 'keep the defaults'}"
          f" (best confirmed {d.get('best_cid')}: {d.get('reasons')})")
    for k, (old, new) in (d.get("changes") or {}).items():
        print(f"  {k:18s} {old:g} -> {new:g}")
    e = out["evaluator"]
    print("mean task wall [s]: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(e["mean_task_s"].items()))
          + f"; pool speed-up {e['speedup']:.1f}x, {e['errors']} errors")
    print(f"runtime {out['runtime_min']:.1f} min -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
