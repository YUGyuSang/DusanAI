#!/usr/bin/env python3
"""
Physics validation of the locomotion stack in MuJoCo (development tool).

Builds a MuJoCo model from simulation/urdf/spot.urdf that mirrors the Gazebo
setup (gz JointPositionController as configured in the URDF, DART-like contacts,
sensor rates, see tools/physics/), drives it with the REAL ROS-free
module1_locomotion controller (MPC, gait auto-selection, elevation map from
simulated depth, terrain classifier, fall recovery) on simulated time, runs
the scenarios (a) .. (j) of tools/physics/scenarios.py plus the slippery-plate
scenarios (k) defined here over several seeds and writes

    docs/results/physics_check.json      per scenario / seed metrics + pass/fail
    docs/images/physics_*.png            timelines, trajectories, torques, stick figures,
                                         physics_slippery.png (k: before / after)

(k) slippery steel plate (F-OBS9 mu 0.15, C-OBS4 mu 0.2; 4 m x 2.4 m, 1 cm):
trot command 0.4 m/s, auto gait, plate from x = 1 to 5 m, goal x = 6 m.
Variants of the slip handling of module1 (README §3.8): ``off`` (none =
before), ``prior`` (friction zone of the layout -> /terrain/info slippery),
``detect`` (online slip detection only), ``both`` (deployed configuration).

Usage::

    python tools/physics_check.py                     # all scenarios, seeds 0 1 2 (required) / 0 (best effort)
    python tools/physics_check.py --quick             # (a) + (b) only, seed 0, no figures
    python tools/physics_check.py --scenarios a_stand b_trot --seeds 0 1 2 3
    python tools/physics_check.py --baseline old.json # embed an earlier run as "before"
    python tools/physics_check.py --previous old.json # keep old.json's "before", store its summary as previous_run
    python tools/physics_check.py --scenarios k_slip_off k_slip_both   # slippery plate only

Requires ``pip install mujoco matplotlib`` (dev / validation only; nothing in
module1_locomotion imports mujoco).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

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

RESULTS = os.path.join(ROOT, "docs", "results", "physics_check.json")
IMAGES = os.path.join(ROOT, "docs", "images")

REQUIRED_SEEDS = (0, 1, 2)
OPTIONAL_SEEDS = (0,)


def _clean(v):
    """JSON-safe (NaN -> None, numpy -> python, rounded floats)."""
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
# (k) slippery steel plate: friction prior + online slip detection (module1 README §3.8)
# ---------------------------------------------------------------------------
SLIP_X0, SLIP_LEN = 1.0, 4.0                  # plate x range 1 .. 5 m (4 m x 2.4 m, 1 cm)
SLIP_GOAL = SLIP_X0 + SLIP_LEN + 1.0          # base passes x = 6 m (as the (i) scenarios)
SLIP_CMD = (0.4, 0.0, 0.0)                    # trot command; the slip mode lowers it on the plate
SLIP_SEEDS = (0, 1, 2)
# variant -> (controller parameter overrides, friction prior installed, label)
SLIP_VARIANTS = {
    "off": ({"slip_detection": False, "use_terrain_friction": False}, False, "no slip handling (before)"),
    "prior": ({"slip_detection": False}, True, "friction prior only"),
    "detect": ({}, False, "online slip detection only"),
    "both": ({}, True, "prior + detection (deployed)"),
}
# logged every 10 ms: controller slip state (see LocomotionController._debug)
SLIP_LOG = ("t", "slip", "slip_mode", "slip_speed", "mu_est", "mpc_mu", "torque_ff", "slippery_info",
            "body_lead_x")


def plate_friction_map(terrain):
    """FrictionMap of the scenario plate, built through the world_layout.yaml
    path of the terrain mapper (obstacles of type plate: rect + mu)."""
    from module1_locomotion.friction_map import FrictionMap
    obstacles = [{"id": f"plate{k}", "type": "plate", "rect": list(r), "mu": float(mu)}
                 for k, (r, mu) in enumerate(terrain.friction_regions)]
    return FrictionMap.from_layout({"obstacles": obstacles})


def _attach_slip_log(sim) -> None:
    """Wrap PhysicsSim._record: log the controller's slip state with every sample."""
    sim._slip_log = []
    rec0 = sim._record

    def rec(t, _rec=rec0, _sim=sim):
        _rec(t)
        d = _sim.ctrl.last_debug or {}
        info = _sim.perception.info if _sim.perception is not None else None
        wt = getattr(_sim.ctrl, "_walk_target", None)
        lead = float(wt[0][0] - _sim.data.qpos[0]) if wt is not None else float("nan")
        _sim._slip_log.append((float(t), float(bool(d.get("slip", False))), float(bool(d.get("slip_mode", False))),
                               float(d.get("slip_speed", 0.0) or 0.0), float(d.get("mu_est", float("nan"))),
                               float(d.get("mpc_mu", float("nan"))), float(d.get("torque_ff", 0.0) or 0.0),
                               float(bool(info and info.get("slippery"))), lead))
    sim._record = rec


def _setup_slip(prior: bool):
    def setup(sim, seed: int) -> None:
        from physics import scenarios as S
        sim.cmd_fn = S._cmd_after(S.T_WALK, SLIP_CMD)
        if prior and sim.perception is not None:
            sim.perception.classifier.set_friction_map(plate_friction_map(sim.cfg.terrain))
        _attach_slip_log(sim)
    return setup


def _eval_slip(sim, A):
    """(i)-style course metrics on the plate + slip-handling metrics."""
    from physics import scenarios as S
    m, ok, _ = S._eval_plate(SLIP_GOAL, SLIP_X0, SLIP_X0 + SLIP_LEN)(sim, A)
    L = np.asarray(getattr(sim, "_slip_log", []), dtype=float).reshape(-1, len(SLIP_LOG))
    x = A["pos"][:, 0]
    # the front feet (hip x offset 0.22 m ahead of the base) reach the plate
    entry = np.nonzero(x >= SLIP_X0 - rm_hip_x())[0]
    t_entry = float(A["t"][entry[0]]) if entry.size else float("nan")
    ev = sim.events
    t_det = [e["t"] for e in ev if e.get("event") == "slip_detected"]
    t_mode = [e["t"] for e in ev if e.get("event") == "slip_mode_on"]
    m["t_plate_entry"] = t_entry
    m["t_slip_detected"] = t_det[0] if t_det else float("nan")
    m["slip_detections"] = len(t_det)
    m["t_slip_mode"] = next((t for t in t_mode if t >= S.T_WALK - 1.0), float("nan"))
    m["detect_latency"] = (m["t_slip_detected"] - t_entry) if t_det and math.isfinite(t_entry) else float("nan")
    m["mode_lead_time"] = (t_entry - m["t_slip_mode"]) if math.isfinite(m["t_slip_mode"]) else float("nan")
    if L.size:
        xs = np.interp(L[:, 0], A["t"], x)
        on = (xs > SLIP_X0 + 0.3) & (xs < SLIP_X0 + SLIP_LEN - 0.3)
        if on.any():
            m["slip_mode_frac_plate"] = float(L[on, 2].mean())
            m["mpc_mu_plate_min"] = float(np.nanmin(L[on, 5]))
            m["torque_ff_plate_mean"] = float(L[on, 6].mean())
        m["mu_est_min"] = float(np.nanmin(L[:, 4]))
    m["cmd"] = list(SLIP_CMD)
    return m, ok, (f"mu {sim.cfg.terrain.friction_regions[0][1]:g} plate, trot command 0.4 m/s, auto gait: "
                   f"no fall; base passes x = {SLIP_GOAL:g} m")


def rm_hip_x() -> float:
    from dusan_common import robot_model as rm
    return float(rm.HIP_X_OFFSET)


def slip_scenarios():
    """(k) scenarios: mu 0.15 (all four variants) and mu 0.2 (off / detect / both)."""
    from dataclasses import dataclass, field
    from physics import scenarios as S
    from physics import terrain as T

    @dataclass
    class SlipScenario(S.Scenario):
        params: dict = field(default_factory=dict)       # merged into the controller parameters
        seeds: tuple = SLIP_SEEDS

    out = {}
    for mu, tag in ((0.15, ""), (0.2, "02")):
        for var, (params, prior, label) in SLIP_VARIANTS.items():
            if tag and var == "prior":
                continue
            key = f"k_slip{tag}_{var}"
            terr = (lambda mu=mu: T.plate(x0=SLIP_X0, length=SLIP_LEN, mu=mu))
            out[key] = SlipScenario(
                key, f"(k) mu {mu:g} plate, trot 0.4: {label}", 45.0, terr, _setup_slip(prior), _eval_slip,
                required=(var == "both"), stop_x=SLIP_GOAL + 0.2, params=dict(params))
    return out


def all_scenarios():
    from physics import scenarios as S
    reg = dict(S.SCENARIOS)
    reg.update(slip_scenarios())
    return reg


def series_for_plots(A, snapshots, every: int = 2):
    """Decimated time series (50 Hz) + snapshots for the figures."""
    s = slice(None, None, every)
    eff = np.array([45.0, 45.0, 60.0] * 4)
    out = {
        "t": A["t"][s], "pos": A["pos"][s], "rpy": A["rpy"][s], "body_h": A["body_h"][s],
        "ground_h": A["ground_h"][s], "vel_body": A["vel_body"][s], "omega": A["omega"][s],
        "cmd_f": A["cmd_f"][s], "mode": A["mode"][s], "gait": A["gait"][s], "rec_state": A["rec_state"][s],
        "tau_ratio": (np.abs(A["tau"]) / eff)[s], "contacts": A["contacts"][s], "foot_pos": A["foot_pos"][s],
        "foot_slip": A["foot_slip"][s],
    }
    snaps = [{"t": sn["t"], "base": sn["base"], "R": sn["R"], "mode": sn["mode"], "rec_state": sn["rec_state"],
              **{l: sn[l] for l in ("fl", "fr", "rl", "rr")}} for sn in snapshots]
    return out, snaps


def job(args):
    """Worker: run one (scenario, seed) -> result dict (picklable)."""
    key, seed, params, perception, duration, keep_series, plant = args
    from physics import scenarios as S
    t0 = time.time()
    sc = all_scenarios()[key]
    params = {**getattr(sc, "params", {}), **(params or {})}
    try:
        sim, A, m, ok, crit = S.run_scenario(sc, seed, params, perception=perception, duration=duration,
                                             plant=plant)
    except Exception as e:                   # report, never kill the whole run
        import traceback
        return {"key": key, "seed": seed, "ok": False, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(), "wall_s": time.time() - t0}
    res = {"key": key, "seed": seed, "ok": ok, "criterion": crit, "metrics": m, "wall_s": time.time() - t0,
           "events": [ev for ev in sim.events if ev.get("event") != "reset_request"][:40],
           "messages": [f"{t:.2f} {msg}" for t, msg in sim.messages[:40]]}
    if keep_series:
        ser, snaps = series_for_plots(A, sim.snapshots)
        res["series"] = ser
        res["snapshots"] = snaps
        res["terrain"] = sc.terrain()
        res["gt"] = sim._gt.gt
        if getattr(sim, "_slip_log", None):
            L = np.asarray(sim._slip_log, dtype=float)[::2]
            res["slip_series"] = {k: L[:, i] for i, k in enumerate(SLIP_LOG)}
    return res


def summarize(results, scenarios):
    """Per scenario: pass count and the key metrics (seed mean)."""
    rows = []
    by = {}
    for r in results:
        by.setdefault(r["key"], []).append(r)
    for key, sc in scenarios.items():
        rs = sorted(by.get(key, []), key=lambda r: r["seed"])
        if not rs:
            continue
        npass = sum(1 for r in rs if r.get("ok"))
        rows.append({"key": key, "title": sc.title, "required": sc.required, "passed": npass, "runs": len(rs),
                     "ok": npass == len(rs), "seeds": [r["seed"] for r in rs],
                     "criterion": rs[0].get("criterion", ""), "key_metrics": key_metrics(key, rs)})
    return rows


def _mean(rs, name):
    vals = [r["metrics"].get(name) for r in rs if r.get("metrics")]
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _max(rs, name):
    vals = [r["metrics"].get(name) for r in rs if r.get("metrics")]
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and v is not None and math.isfinite(float(v))]
    return float(np.max(vals)) if vals else float("nan")


def key_metrics(key, rs):
    falls = sum(int(r.get("metrics", {}).get("falls", r.get("metrics", {}).get("fall_episodes_metric", 0)) or 0)
                for r in rs)
    km = {"falls_total": falls}
    if key == "a_stand":
        km.update(height=_mean(rs, "height_mean"), roll_max=_max(rs, "max_roll_deg"),
                  pitch_max=_max(rs, "max_pitch_deg"), sat=_mean(rs, "saturation_frac"),
                  drift_mm=_mean(rs, "drift_mm"))
    elif key in ("b_trot", "c_crawl", "d_turn"):
        km.update(vx=_mean(rs, "vx_mean"), vy=_mean(rs, "vy_mean"), wz=_mean(rs, "wz_mean"),
                  height=_mean(rs, "height_mean"), roll_max=_max(rs, "max_roll_deg"),
                  pitch_max=_max(rs, "max_pitch_deg"), knee_max=_max(rs, "max_ratio_knee"),
                  sat=_mean(rs, "saturation_frac"), mpc_ms=_mean(rs, "mpc_ms_mean"))
    elif key == "f_push":
        km.update(peak_roll=_max(rs, "max_roll_deg"), peak_vy=_max(rs, "peak_vy"),
                  recovery_s=_mean(rs, "recovery_s"), vx_after=_mean(rs, "vx_after"))
    elif key == "g_payload":
        km.update(stand_height=_mean(rs, "stand_height_mean"), vx=_mean(rs, "vx_mean"),
                  knee_max=_max(rs, "max_ratio_knee"), sat=_mean(rs, "saturation_frac"))
    elif key.startswith("k_slip"):
        km.update(t_goal=_mean(rs, "t_goal"), x_max=_mean(rs, "x_max"), plate_speed=_mean(rs, "plate_speed"),
                  slip_dist=_mean(rs, "plate_slip_dist"), slip=_mean(rs, "plate_slip_mean"),
                  slip_p95=_mean(rs, "plate_slip_p95"), roll_max=_max(rs, "plate_max_roll_deg"),
                  detect_latency=_mean(rs, "detect_latency"), mode_lead=_mean(rs, "mode_lead_time"),
                  mode_frac=_mean(rs, "slip_mode_frac_plate"), mpc_mu=_mean(rs, "mpc_mu_plate_min"))
    elif key.startswith("h_"):
        km.update(attempts=_mean(rs, "recovery_attempts"), recoveries=_mean(rs, "recoveries"),
                  assisted=_mean(rs, "assisted_resets"), t_recover=_mean(rs, "time_to_recover"))
    else:
        km.update(t_goal=_mean(rs, "t_goal"), x_max=_mean(rs, "x_max"), roll_max=_max(rs, "max_roll_deg"),
                  pitch_max=_max(rs, "max_pitch_deg"), slip=_mean(rs, "plate_slip_mean"),
                  slip_dist=_mean(rs, "plate_slip_dist"))
    return km


def fmt_row(row):
    km = row["key_metrics"]
    parts = []
    for k, v in km.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3g}" if math.isfinite(v) else f"{k}=-")
        else:
            parts.append(f"{k}={v}")
    status = "PASS" if row["ok"] else ("FAIL" if row["required"] else "best-effort")
    return f"{row['title'][:52]:52s} {row['passed']}/{row['runs']} {status:11s} " + " ".join(parts)


def print_table(rows, title="physics check"):
    print(f"\n=== {title} ===")
    for row in rows:
        print(fmt_row(row))
    req = [r for r in rows if r["required"]]
    print(f"required scenarios passing on every seed: {sum(r['ok'] for r in req)}/{len(req)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=None, help="scenario keys (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="seeds (default 0 1 2 / best effort 0)")
    ap.add_argument("--quick", action="store_true", help="(a) stand + short trot, seed 0, no figures")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--perception", default=None, help="depth | gt | none (default: per scenario, depth)")
    ap.add_argument("--params", default="", help="JSON dict of controller parameter overrides")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--baseline", default="", help="earlier physics_check.json to embed as 'before_fixes'")
    ap.add_argument("--previous", default="",
                    help="earlier physics_check.json: keep ITS before_fixes, store its summary as previous_run")
    ap.add_argument("--label", default="", help="free text stored in the JSON")
    ap.add_argument("--servo", default="urdf", choices=("urdf", "velocity", "force"),
                    help="gz JointPositionController mode: urdf (the plugin blocks of spot.urdf, default) | "
                         "velocity | force (what-if overrides, see --servo-p/--servo-d)")
    ap.add_argument("--servo-p", type=float, default=None, help="what-if p_gain override [N m/rad]")
    ap.add_argument("--servo-d", type=float, default=None, help="what-if d_gain override [N m s/rad]")
    a = ap.parse_args(argv)
    params = json.loads(a.params) if a.params else {}
    plant = {"servo": a.servo}
    if a.servo_p is not None:
        plant["servo_p"] = a.servo_p
    if a.servo_d is not None:
        plant["servo_d"] = a.servo_d
    registry = all_scenarios()
    keys = a.scenarios or list(registry)
    unknown = [k for k in keys if k not in registry]
    if unknown:
        ap.error(f"unknown scenario(s) {unknown}; known: {sorted(registry)}")
    jobs = []
    if a.quick:
        jobs = [("a_stand", 0, params, a.perception, None, False, plant),
                ("b_trot", 0, params, a.perception, 7.5, False, plant)]
        a.no_figures = True
    else:
        for k in keys:
            sc = registry[k]
            seeds = a.seeds if a.seeds is not None else (
                getattr(sc, "seeds", None) or (REQUIRED_SEEDS if sc.required else OPTIONAL_SEEDS))
            for sd in seeds:
                jobs.append((k, sd, params, a.perception, None, sd == seeds[0] and not a.no_figures, plant))
    t0 = time.time()
    results = []
    if a.jobs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for r in pool.imap_unordered(job, jobs):
                results.append(r)
                print(f"  {r['key']:14s} seed {r['seed']}: {'PASS' if r.get('ok') else 'FAIL'} "
                      f"({r.get('wall_s', 0):.0f} s){'  ERROR ' + r['error'] if 'error' in r else ''}", flush=True)
    else:
        for jb in jobs:
            r = job(jb)
            results.append(r)
            print(f"  {r['key']:14s} seed {r['seed']}: {'PASS' if r.get('ok') else 'FAIL'} "
                  f"({r.get('wall_s', 0):.0f} s){'  ERROR ' + r['error'] if 'error' in r else ''}", flush=True)
    rows = summarize(results, {k: registry[k] for k in (keys if not a.quick else ["a_stand", "b_trot"])})
    print_table(rows)
    print(f"wall time {time.time() - t0:.0f} s")
    if a.quick:
        return 0 if all(r["ok"] for r in rows) else 1
    before = None
    previous = None
    if a.previous and os.path.isfile(a.previous):
        with open(a.previous, encoding="utf-8") as f:
            prev = json.load(f)
        previous = {"label": prev.get("label", ""), "date": prev.get("date", ""), "plant": prev.get("plant"),
                    "summary": prev.get("summary", [])}
        before = prev.get("before_fixes")
    if a.baseline and os.path.isfile(a.baseline):
        with open(a.baseline, encoding="utf-8") as f:
            base = json.load(f)
        before = {"label": base.get("label", ""), "date": base.get("date", ""), "plant": base.get("plant"),
                  "summary": base.get("summary", [])}
    figures = []
    if not a.no_figures:
        from physics import plots
        figures = plots.make_all([r for r in results if "series" in r], IMAGES, rows=rows,
                                 before=before["summary"] if before else None, all_results=results)
        fig = slippery_figure(results, rows, os.path.join(IMAGES, "physics_slippery.png"))
        if fig:
            figures.append(fig)
    out = {
        "tool": "tools/physics_check.py",
        "label": a.label,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "engine": _engine_info(),
        "params_override": params,
        "plant": plant,
        "summary": rows,
        "runs": [{k: v for k, v in r.items() if k not in ("series", "snapshots", "terrain", "gt", "trace", "slip_series")}
                 for r in sorted(results, key=lambda r: (r["key"], r["seed"]))],
        "figures": [os.path.relpath(f, ROOT).replace("\\", "/") for f in figures],
    }
    if before is not None:
        out["before_fixes"] = before
    if previous is not None:
        out["previous_run"] = previous
    slip = slippery_table(rows)
    if slip:
        out["slippery"] = slip
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_clean(out), f, indent=1)
    print(f"wrote {a.out}")
    return 0 if all(r["ok"] for r in rows if r["required"]) else 1


def slippery_table(rows):
    """Before / after comparison of the (k) variants (seed means, falls summed)."""
    out = {}
    for r in rows:
        if not r["key"].startswith("k_slip"):
            continue
        km = r["key_metrics"]
        out[r["key"]] = {"title": r["title"], "passed": r["passed"], "runs": r["runs"], "falls": km.get("falls_total"),
                         **{k: km.get(k) for k in ("plate_speed", "slip_dist", "slip", "slip_p95", "t_goal",
                                                   "detect_latency", "mode_lead", "mode_frac", "mpc_mu")}}
    if out:
        out["note"] = ("plate x 1..5 m, trot command 0.4 m/s, auto gait; slip_dist = summed horizontal speed of "
                       "the feet in contact x dt on the plate [m]; plate_speed = base speed over the plate; "
                       "detect_latency = slip detection - front feet on the plate [s]; mode_lead = slip mode "
                       "on before the front feet reach the plate [s]")
    return out


def slippery_figure(results, rows, path):
    """physics_slippery.png: (k) mu 0.15 before / after - base progress, slip state, seed summary."""
    runs = {}
    for r in sorted(results, key=lambda r: r["seed"]):
        if r["key"].startswith("k_slip_") and "series" in r and r["key"] not in runs:
            runs[r["key"]] = r
    if not runs:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physics.plots import AQUA, BLUE, CRITICAL, GRID, INK, INK2, ORANGE, SURFACE, _style
    style = {"k_slip_off": (INK2, "--", "off (before)"), "k_slip_prior": (BLUE, "-", "friction prior"),
             "k_slip_detect": (ORANGE, "-", "slip detection"), "k_slip_both": (AQUA, "-", "prior + detection")}
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0), hspace=0.42, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    _style(ax)
    ax.axhspan(SLIP_X0, SLIP_X0 + SLIP_LEN, color=GRID, alpha=0.55, lw=0)
    for key, (col, ls, lab) in style.items():
        r = runs.get(key)
        if r is None:
            continue
        s = r["series"]
        t, xb = np.asarray(s["t"]), np.asarray(s["pos"])[:, 0]
        ax.plot(t, xb, color=col, ls=ls, lw=2.0, label=f"{lab}: {'PASS' if r.get('ok') else 'FAIL'}"
                + (" (almost the prior path)" if key == "k_slip_both" else ""))
        for e in r.get("events", []):
            if e.get("event") == "fall":
                k = int(np.searchsorted(t, e["t"]))
                ax.plot(e["t"], xb[min(k, len(xb) - 1)], marker="x", ms=9, mew=2.2, color=CRITICAL)
    ax.plot([], [], ls="none", marker="x", ms=9, mew=2.2, color=CRITICAL, label="fall")
    ax.axhline(SLIP_GOAL, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(-0.5, SLIP_GOAL + 0.6)
    ax.text(0.99, (SLIP_X0 + 0.12 + 0.5) / (SLIP_GOAL + 1.1), "mu 0.15 steel plate", transform=ax.transAxes,
            fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xlabel("time [s]", fontsize=8, color=INK2)
    ax.set_ylabel("base x [m]", fontsize=8, color=INK2)
    ax.set_title("(k) trot command 0.4 m/s onto a mu 0.15 plate, seed 0: base progress", fontsize=9, loc="left",
                 color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    # slip state of the detection-only run
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    r = runs.get("k_slip_detect") or runs.get("k_slip_both")
    if r is not None and "slip_series" in r:
        ss = r["slip_series"]
        t = np.asarray(ss["t"])
        mode = np.asarray(ss["slip_mode"]) > 0.5
        ax2.fill_between(t, 0, 1, where=mode, transform=ax2.get_xaxis_transform(), color=ORANGE, alpha=0.12, lw=0,
                         label="slip mode")
        ax2.plot(t, ss["slip_speed"], color=INK2, lw=1.0, label="stance-foot speed [m/s]")
        ax2.plot(t, ss["mpc_mu"], color=BLUE, lw=2.0, label="MPC mu")
        ax2.plot(t, ss["torque_ff"], color=AQUA, lw=2.0, label="torque feed-forward gain")
        xs = np.asarray(r["series"]["pos"])[:, 0]
        tt = np.asarray(r["series"]["t"])
        k = np.nonzero(xs >= SLIP_X0 - rm_hip_x())[0]
        if k.size:
            ax2.axvline(tt[k[0]], color=INK, lw=0.8, ls=":")
            ax2.text(tt[k[0]] + 0.2, 1.12, "front feet on the plate", fontsize=6.5, color=INK2)
        ax2.set_ylim(0.0, 1.2)
        ax2.set_xlim(3.5, min(float(t[-1]), 20.0))
        ax2.legend(fontsize=6.5, frameon=False, loc="upper right")
        ax2.set_title(f"slip detection only (seed {r['seed']}): detection -> slip mode", fontsize=9, loc="left",
                      color=INK)
        ax2.set_xlabel("time [s]", fontsize=8, color=INK2)
    # seed summary
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    rowmap = {rw["key"]: rw for rw in rows}
    keys = [k for k in ("k_slip_off", "k_slip_prior", "k_slip_detect", "k_slip_both", "k_slip02_off",
                        "k_slip02_detect", "k_slip02_both") if k in rowmap]
    labels = {"k_slip_off": "0.15 off", "k_slip_prior": "0.15 prior", "k_slip_detect": "0.15 detect",
              "k_slip_both": "0.15 both", "k_slip02_off": "0.2 off", "k_slip02_detect": "0.2 detect",
              "k_slip02_both": "0.2 both"}
    colors = {"off": INK2, "prior": BLUE, "detect": ORANGE, "both": AQUA}
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        rw = rowmap[k]
        col = colors[k.rsplit("_", 1)[1]]
        ax3.barh(yi, rw["passed"], height=0.55, color=col, edgecolor=SURFACE, linewidth=2)
        km = rw["key_metrics"]
        sd = km.get("slip_dist")
        sp = km.get("plate_speed")
        txt = f"{rw['passed']}/{rw['runs']} cross, falls {km.get('falls_total', 0)}"
        if isinstance(sp, float) and math.isfinite(sp):
            txt += f", {sp:.2f} m/s"
        if isinstance(sd, float) and math.isfinite(sd):
            txt += f", slip {sd:.1f} m"
        ax3.text(max(rw["passed"], 0) + 0.08, yi, txt, fontsize=6.8, va="center", color=INK)
    ax3.set_yticks(y)
    ax3.set_yticklabels([labels[k] for k in keys], fontsize=7)
    ax3.set_xlim(0, 6.2)
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xlabel("seeds crossing without a fall (of 3)", fontsize=8, color=INK2)
    ax3.set_title("all seeds: plate mu / slip handling", fontsize=9, loc="left", color=INK)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    return path


def _engine_info():
    from physics import mjcf
    try:
        import mujoco
        ver = mujoco.__version__
    except Exception:
        ver = "?"
    desc = mjcf.parse_urdf()
    sp = desc.servos.get("fl_knee")
    servo = (f"gz JointPositionController {sp.mode} mode (spot.urdf): p {sp.p_gain:g}, d {sp.d_gain:g}, "
             f"i {sp.i_gain:g}, |cmd| <= {sp.cmd_max:g} (knee)") if sp is not None else "?"
    return {"mujoco": ver, "gz_dt": mjcf.GZ_DT, "substeps": mjcf.SUBSTEPS, "servo_kv": mjcf.SERVO_KV,
            "contact_solref": list(mjcf.CONTACT_SOLREF), "impratio": mjcf.IMPRATIO,
            "servo": servo,
            "sensors": "joint states 200 Hz, IMU 200 Hz, odom 50 Hz (rolling mean 10), controller 100 Hz",
            "mpc_latency": "10/20/30 ms (p = .25/.5/.25, seeded) of simulated time"}


if __name__ == "__main__":
    sys.exit(main())
