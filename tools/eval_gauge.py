#!/usr/bin/env python3
"""
Offline benchmark of the Module 2 analog-gauge reader (no ROS needed, runs on Windows).

Synthetic inspection-camera frames are rendered with module2_inspection.gauge_synth for all
8 gauges of simulation/config/world_layout.yaml (their real ranges, units, tags and values
plus random values over the whole scale) under a grid of conditions:

    normal | low_light | motion_blur | oblique | noise | combined      (required grid)
    occluded                                                           (extra, reported apart)

Each frame is read twice:
  * registry mode - the gauge range is known (what gauge_reader_node does when the robot
    pose associates the dial with a registry gauge); full-frame detection, no ROI prior;
  * OCR mode      - no range given; the range must be read from the dial labels.
Background-only frames (clutter + round distractors) measure false detections.
Operating-envelope sweeps (registry mode) push one degradation beyond the evaluation grid:
illumination 1.0 -> 0.025 (gamma 1.3, shot + read noise) and motion blur 0 -> 30 px, to show
where the reader starts to reject frames or to fail.

Error = |value_read - value_true| / (max - min) * 100  [% of full scale].

Outputs
  docs/images/gauge_error_hist.png, gauge_error_by_condition.png, gauge_error_vs_distance.png,
  docs/images/gauge_ocr_vs_distance.png, gauge_examples.png, gauge_sweeps.png, gauge_pipeline.png
  docs/results/gauge_eval.json

Usage (from the repository root):
  python tools/eval_gauge.py                 # 480 + 60 (+60 empty, +200 sweep) frames, all cores
  python tools/eval_gauge.py --frames 120 --sweep 0 --workers 1 --no-plots   (quick check)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ("dusan_common", "module2_inspection"):
    _p = os.path.join(ROOT, _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

REQUIRED = ("normal", "low_light", "motion_blur", "oblique", "noise", "combined")
EXTRA = ("occluded",)
TARGET_MEAN_FS = 2.0
TARGET_P95_FS = 5.0
SWEEPS = {   # operating-envelope sweeps: kind -> levels
    "sweep_light": (1.0, 0.5, 0.25, 0.15, 0.10, 0.07, 0.05, 0.035, 0.025, 0.018),
    "sweep_blur": (0.0, 4.0, 8.0, 12.0, 15.0, 18.0, 22.0, 26.0, 30.0, 35.0),
}

_WORKER: dict = {}


# ------------------------------------------------------------------------------------ jobs
def make_jobs(n_frames: int, n_extra: int, n_empty: int, seed: int, n_sweep: int = 0):
    from dusan_common.layout import Layout
    gauges = Layout.load().gauges
    rng = np.random.default_rng(seed)
    jobs = []
    per_kind = int(math.ceil(n_frames / len(REQUIRED)))
    plan = [(k, per_kind) for k in REQUIRED] + [(k, n_extra) for k in EXTRA]
    idx = 0
    for ki, (kind, n) in enumerate(plan):
        for j in range(n):
            # every gauge in every condition: gauge cycles with period 8, value type with
            # period 6 -> each (gauge, value type) pair occurs once per 24 frames
            gi = (j + ki) % len(gauges)
            g = gauges[gi]
            u = j % 6
            if u in (0, 3):
                val = float(g["value"])                                   # the registry value
            elif u == 1:
                val = float(g["min"] + rng.uniform(0.0, 0.06) * (g["max"] - g["min"]))   # near min
            elif u == 4:
                val = float(g["max"] - rng.uniform(0.0, 0.06) * (g["max"] - g["min"]))   # near max
            else:
                val = float(rng.uniform(g["min"], g["max"]))
            jobs.append({"idx": idx, "kind": kind, "gauge": gi, "value": val,
                         "seed": int(rng.integers(0, 2 ** 31 - 1)), "example": j < 2})
            idx += 1
    for j in range(n_empty):
        jobs.append({"idx": idx, "kind": "empty", "gauge": -1, "value": float("nan"),
                     "seed": int(rng.integers(0, 2 ** 31 - 1)), "example": False})
        idx += 1
    for kind, levels in SWEEPS.items():
        for level in levels:
            for j in range(n_sweep):
                gi = (j + int(level * 10)) % len(gauges)
                g = gauges[gi]
                jobs.append({"idx": idx, "kind": kind, "level": float(level), "gauge": gi,
                             "value": float(rng.uniform(g["min"], g["max"])),
                             "seed": int(rng.integers(0, 2 ** 31 - 1)), "example": False})
                idx += 1
    return jobs


def _init_worker(single_thread: bool = True):
    import cv2
    from dusan_common.layout import Layout
    from module2_inspection.gauge_reader import GaugeReader
    from module2_inspection.gauge_synth import GaugeSynth
    if single_thread:
        cv2.setNumThreads(1)          # one OpenCV thread per worker process (no oversubscription)
    synth = GaugeSynth(seed=0)
    _WORKER["synth"] = synth
    _WORKER["reader"] = GaugeReader(camera_matrix=synth.camera.K)
    _WORKER["gauges"] = Layout.load().gauges


def run_job(job: dict) -> dict:
    import cv2
    from dusan_common import gauge_design as gd
    from module2_inspection.gauge_reader import draw_reading, draw_header
    from module2_inspection.gauge_synth import GaugeSpec, random_conditions, random_view
    if not _WORKER:
        _init_worker()
    synth, reader, gauges = _WORKER["synth"], _WORKER["reader"], _WORKER["gauges"]
    rng = np.random.default_rng(job["seed"])
    kind = job["kind"]
    if kind == "empty":
        cond = random_conditions(rng, ["normal", "low_light", "noise"][job["idx"] % 3])
        img = synth.render_empty(cond, rng)
        t0 = time.perf_counter()
        res = [r for r in reader.read(img) if r.accepted]
        return {"idx": job["idx"], "kind": kind, "false_detections": len(res),
                "fp_scores": [{k: round(float(r.scores.get(k, 0.0)), 3) for k in ("cal_ncc", "cal_margin",
                                                                                   "face_contrast")}
                              | {"confidence": round(r.confidence, 3)} for r in res],
                "time_ms": (time.perf_counter() - t0) * 1000.0}
    g = gauges[job["gauge"]]
    spec = GaugeSpec.from_layout(g, job["value"])
    fs = float(g["max"] - g["min"])
    if kind in SWEEPS:
        # one degradation pushed beyond the evaluation grid, everything else 'normal'
        view = random_view(rng, "normal")
        cond = random_conditions(rng, "normal")
        if kind == "sweep_light":
            cond.illumination, cond.gamma = job["level"], 1.3
            cond.noise_sigma, cond.shot_noise = 2.0, 0.4
            cond.illum_gradient = max(cond.illum_gradient, 0.3)
        else:
            cond.motion_blur_px = job["level"]
        img, gt = synth.render(spec, view, cond, rng)
        t0 = time.perf_counter()
        rd = reader.read_one(img, (float(g["min"]), float(g["max"])))
        out = {"idx": job["idx"], "kind": kind, "level": job["level"], "gauge": g["id"], "value": job["value"],
               "distance": round(view.distance, 3), "radius_px": round(gt["radius_px"], 1),
               "time_ms": (time.perf_counter() - t0) * 1000.0,
               "detected": bool(rd is not None and rd.value is not None)}
        if out["detected"]:
            out.update({"abs_err_fs": abs(rd.value - job["value"]) / fs * 100.0, "confidence": rd.confidence,
                        "blur_est_px": rd.blur_px})
        else:
            rej = reader.last_debug.get("rejected") or []
            out["reject"] = str(rej[0][0]) if rej else "not_detected"
        return out
    view = random_view(rng, kind)
    cond = random_conditions(rng, kind)
    img, gt = synth.render(spec, view, cond, rng)
    out = {"idx": job["idx"], "kind": kind, "gauge": g["id"], "value": job["value"], "fs": fs,
           "distance": round(view.distance, 3), "incidence_deg": round(gt["incidence_deg"], 2),
           "roll_deg": round(view.roll_deg, 2), "illumination": round(cond.illumination, 3),
           "motion_blur_px": round(cond.motion_blur_px, 2), "noise_sigma": round(cond.noise_sigma, 2),
           "radius_px": round(gt["radius_px"], 1)}
    # ---- registry mode (range known)
    t0 = time.perf_counter()
    rd = reader.read_one(img, (float(g["min"]), float(g["max"])))
    out["time_ms"] = (time.perf_counter() - t0) * 1000.0
    out["detected"] = bool(rd is not None and rd.value is not None)
    if out["detected"]:
        err = (rd.value - job["value"]) / fs * 100.0
        cerr = math.hypot(rd.center[0] - gt["centre_px"][0], rd.center[1] - gt["centre_px"][1])
        out.update({"estimate": rd.value, "err_fs": err, "abs_err_fs": abs(err), "confidence": rd.confidence,
                    "centre_err_px": cerr, "blur_est_px": rd.blur_px,
                    "angle_err_deg": rd.angle_deg - gt["angle_deg"], "mirrored": rd.mirrored,
                    "ocr_registry_mode": list(rd.range_ocr) if rd.range_ocr else None,
                    "cal_ncc": rd.scores.get("cal_ncc"), "cal_margin": rd.scores.get("cal_margin"),
                    "face_contrast": rd.scores.get("face_contrast"),
                    "needle_evidence": rd.scores.get("needle_model_evidence"),
                    "needle_score": rd.scores.get("needle"),
                    "needle_method": next((k[7:] for k in rd.scores if k.startswith("needle_")
                                           and k[7:] in ("centroid", "coarse", "model")), "")})
    # ---- OCR mode (range unknown)
    rdo = reader.read_one(img, None)
    out["ocr_label_px"] = round(0.026 * gt["radius_px"] / gd.R_FACE, 2)     # label height [px]
    out["ocr_range"] = list(rdo.range_ocr) if (rdo is not None and rdo.range_ocr) else None
    out["ocr_ok"] = bool(out["ocr_range"] is not None and abs(out["ocr_range"][0] - g["min"]) < 1e-6
                         and abs(out["ocr_range"][1] - g["max"]) < 1e-6)
    out["ocr_method"] = rdo.ocr_method if rdo is not None else ""
    if rdo is not None and rdo.value is not None:
        out["ocr_err_fs"] = (rdo.value - job["value"]) / fs * 100.0
    if job.get("example"):
        vis = img.copy()
        if rd is not None:
            st = "normal"
            lines = [f"{g['id']} {rd.value:.2f} {g['unit']}" if rd.value is not None else g["id"],
                     f"true {job['value']:.2f}  err {out.get('err_fs', float('nan')):+.2f}%FS"]
            draw_reading(vis, rd, lines, st)
            cx, cy, a = rd.ellipse[0], rd.ellipse[1], rd.ellipse[2]
        else:
            cx, cy = gt["centre_px"]
            a = gt["radius_px"]
        half = int(max(a * 2.0, 110))
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half - 30))
        x1, y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
        crop = vis[y0:y1, x0:x1]
        crop = cv2.resize(crop, (300, int(300 * crop.shape[0] / max(crop.shape[1], 1))),
                          interpolation=cv2.INTER_AREA)
        # caption bar (raw camera frame below: low-light examples are shown as captured)
        hdr = np.zeros((38, 300, 3), np.uint8)
        cv2.putText(hdr, f"{kind}  {g['id']}  ({g['min']:g}-{g['max']:g} {g['unit']})", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hdr, f"d={view.distance:.2f}m inc={gt['incidence_deg']:.0f}deg light={cond.illumination:.2f} "
                         f"blur={cond.motion_blur_px:.0f}px", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200), 1, cv2.LINE_AA)
        out["example_png"] = cv2.imencode(".png", np.vstack([hdr, crop]))[1].tobytes()
    return out


# ------------------------------------------------------------------------------------ stats
def _stats(errs):
    e = np.abs(np.asarray(errs, float))
    if e.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None, "rmse": None}
    return {"n": int(e.size), "mean": float(e.mean()), "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)), "max": float(e.max()), "rmse": float(np.sqrt(np.mean(e ** 2)))}


def summarize(results, cfg_dict):
    by_kind = {}
    for k in REQUIRED + EXTRA:
        rs = [r for r in results if r["kind"] == k]
        det = [r for r in rs if r.get("detected")]
        ocr_try = [r for r in rs if r.get("ocr_range") is not None]
        by_kind[k] = {
            "frames": len(rs), "detected": len(det),
            "detection_rate": len(det) / max(1, len(rs)),
            "error_fs": _stats([r["abs_err_fs"] for r in det]),
            "within_2pct": float(np.mean([r["abs_err_fs"] < 2.0 for r in det])) if det else None,
            "within_5pct": float(np.mean([r["abs_err_fs"] < 5.0 for r in det])) if det else None,
            "mean_confidence": float(np.mean([r["confidence"] for r in det])) if det else None,
            "ocr_range_read_rate": len(ocr_try) / max(1, len(rs)),
            "ocr_range_precision": float(np.mean([r["ocr_ok"] for r in ocr_try])) if ocr_try else None,
            "mean_time_ms": float(np.mean([r["time_ms"] for r in rs])) if rs else None,
        }
    req = [r for r in results if r["kind"] in REQUIRED]
    det = [r for r in req if r.get("detected")]
    overall = {"frames": len(req), "detected": len(det), "detection_rate": len(det) / max(1, len(req)),
               "error_fs": _stats([r["abs_err_fs"] for r in det]),
               "within_2pct": float(np.mean([r["abs_err_fs"] < 2.0 for r in det])) if det else None,
               "within_5pct": float(np.mean([r["abs_err_fs"] < 5.0 for r in det])) if det else None}
    allr = [r for r in results if r["kind"] in REQUIRED + EXTRA]
    det_all = [r for r in allr if r.get("detected")]
    overall_all = {"frames": len(allr), "detected": len(det_all), "detection_rate": len(det_all) / max(1, len(allr)),
                   "error_fs": _stats([r["abs_err_fs"] for r in det_all])}
    # OCR by distance
    bins = [0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.5]
    ocr_bins = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        rs = [r for r in allr if lo <= r["distance"] < hi]
        tried = [r for r in rs if r.get("ocr_range") is not None]
        ocr_bins.append({"distance": [lo, hi], "frames": len(rs),
                         "label_px_mean": float(np.mean([r["ocr_label_px"] for r in rs])) if rs else None,
                         "read_rate": len(tried) / max(1, len(rs)),
                         "correct_rate": float(np.mean([r["ocr_ok"] for r in rs])) if rs else None,
                         "precision": float(np.mean([r["ocr_ok"] for r in tried])) if tried else None})
    tried = [r for r in allr if r.get("ocr_range") is not None]
    ocr = {"frames": len(allr), "range_read": len(tried), "range_correct": int(sum(r["ocr_ok"] for r in allr)),
           "precision": float(np.mean([r["ocr_ok"] for r in tried])) if tried else None,
           "by_distance": ocr_bins,
           "ocr_mode_value_error_fs": _stats([r["ocr_err_fs"] for r in allr if "ocr_err_fs" in r])}
    empty = [r for r in results if r["kind"] == "empty"]
    fp = {"frames": len(empty), "false_detections": int(sum(r["false_detections"] for r in empty)),
          "frames_with_false_detection": int(sum(r["false_detections"] > 0 for r in empty))}
    sweeps = {}
    for kind, levels in SWEEPS.items():
        rows = []
        for level in levels:
            rs = [r for r in results if r["kind"] == kind and abs(r["level"] - level) < 1e-9]
            if not rs:
                continue
            det_s = [r for r in rs if r.get("detected")]
            rej = {}
            for r in rs:
                if not r.get("detected"):
                    rej[r.get("reject", "?")] = rej.get(r.get("reject", "?"), 0) + 1
            rows.append({"level": level, "frames": len(rs), "detection_rate": len(det_s) / len(rs),
                         "error_fs": _stats([r["abs_err_fs"] for r in det_s]),
                         "correct_rate_5pct": float(np.mean([r.get("detected", False) and r["abs_err_fs"] < 5.0
                                                             for r in rs])),
                         "rejections": rej})
        if rows:
            sweeps[kind] = {"parameter": "illumination scale" if kind == "sweep_light" else "motion blur [px]",
                            "levels": rows}
    e = overall["error_fs"]
    target = {"mean_lt_2pct": bool(e["mean"] is not None and e["mean"] < TARGET_MEAN_FS),
              "p95_lt_5pct": bool(e["p95"] is not None and e["p95"] < TARGET_P95_FS)}
    return {"overall_required_grid": overall, "overall_including_occlusion": overall_all,
            "by_condition": by_kind, "ocr": ocr, "false_positives": fp, "targets": target,
            "sweeps": sweeps, "reader_config": cfg_dict}


# ------------------------------------------------------------------------------------ plots
def make_plots(results, summary, img_dir):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(img_dir, exist_ok=True)
    kinds = list(REQUIRED + EXTRA)
    colors = {"normal": "#2a6fdb", "low_light": "#6b4fbb", "motion_blur": "#d95f02", "oblique": "#1b9e77",
              "noise": "#7f7f7f", "combined": "#c51b7d", "occluded": "#a6761d"}
    det = [r for r in results if r.get("detected") and r["kind"] in REQUIRED]
    errs = np.array([r["abs_err_fs"] for r in det])
    # 1) histogram
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(0, max(5.5, float(np.percentile(errs, 99.5)) if errs.size else 5.5), 56)
    ax.hist(np.clip(errs, 0, bins[-1]), bins=bins, color="#2a6fdb", alpha=0.85)
    e = summary["overall_required_grid"]["error_fs"]
    ax.axvline(e["mean"], color="k", lw=1.5, label=f"mean {e['mean']:.2f} %FS")
    ax.axvline(e["p95"], color="k", lw=1.5, ls="--", label=f"95th pct {e['p95']:.2f} %FS")
    ax.axvline(TARGET_MEAN_FS, color="#d95f02", lw=1, ls=":", label="target mean 2 %FS")
    ax.axvline(TARGET_P95_FS, color="#c51b7d", lw=1, ls=":", label="target p95 5 %FS")
    ax.set_xlabel("absolute reading error [% of full scale]")
    ax.set_ylabel("frames")
    ax.set_title(f"Gauge reader error, {e['n']} detected frames (required condition grid)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "gauge_error_hist.png"), dpi=130)
    plt.close(fig)
    # 2) error by condition + detection rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    data = [[r["abs_err_fs"] for r in results if r["kind"] == k and r.get("detected")] for k in kinds]
    bp = ax1.boxplot([d if d else [np.nan] for d in data], showfliers=True, whis=(5, 95), patch_artist=True)
    for patch, k in zip(bp["boxes"], kinds):
        patch.set_facecolor(colors[k])
        patch.set_alpha(0.6)
    ax1.set_xticks(range(1, len(kinds) + 1))
    ax1.set_xticklabels(kinds, rotation=20)
    ax1.axhline(TARGET_P95_FS, color="#c51b7d", lw=1, ls=":")
    ax1.axhline(TARGET_MEAN_FS, color="#d95f02", lw=1, ls=":")
    ax1.set_ylabel("absolute error [% FS] (whiskers 5-95 %)")
    ax1.set_title("Reading error by condition")
    rates = [summary["by_condition"][k]["detection_rate"] * 100 for k in kinds]
    ax2.bar(range(len(kinds)), rates, color=[colors[k] for k in kinds], alpha=0.8)
    for i, v in enumerate(rates):
        ax2.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax2.set_xticks(range(len(kinds)))
    ax2.set_xticklabels(kinds, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("accepted readings [%]")
    ax2.set_title("Detection / acceptance rate")
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "gauge_error_by_condition.png"), dpi=130)
    plt.close(fig)
    # 3) error vs distance
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k in kinds:
        rs = [r for r in results if r["kind"] == k and r.get("detected")]
        ax.scatter([r["distance"] for r in rs], [r["abs_err_fs"] for r in rs], s=12, alpha=0.7,
                   color=colors[k], label=k)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("camera-dial distance [m]")
    ax.set_ylabel("absolute error [% FS]")
    ax.set_title("Error vs distance")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "gauge_error_vs_distance.png"), dpi=130)
    plt.close(fig)
    # 4) OCR vs distance
    ob = summary["ocr"]["by_distance"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xs = [0.5 * (b["distance"][0] + b["distance"][1]) for b in ob]
    ax.plot(xs, [100 * (b["correct_rate"] or 0) for b in ob], "o-", color="#1b9e77", label="range read correctly")
    ax.plot(xs, [100 * (b["precision"] if b["precision"] is not None else 1.0) for b in ob], "s--",
            color="#2a6fdb", label="precision of returned ranges")
    ax2 = ax.twinx()
    ax2.plot(xs, [b["label_px_mean"] or 0 for b in ob], "^:", color="#7f7f7f", label="label height [px]")
    ax2.set_ylabel("label height in the image [px]")
    ax.set_xlabel("camera-dial distance [m]")
    ax.set_ylabel("[%]")
    ax.set_ylim(-5, 105)
    ax.set_title("Scale-label OCR (range unknown) vs distance")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "gauge_ocr_vs_distance.png"), dpi=130)
    plt.close(fig)
    # 5) example overlays: one column per condition, two examples each (raw camera frames)
    ex = [r for r in results if "example_png" in r]
    cols = []
    for k in kinds:
        rs = sorted((r for r in ex if r["kind"] == k), key=lambda r: r["idx"])[:2]
        cols.append([cv2.imdecode(np.frombuffer(r["example_png"], np.uint8), cv2.IMREAD_COLOR) for r in rs])
    tiles = [t for c in cols for t in c]
    if tiles:
        h = max(t.shape[0] for t in tiles)
        blank = np.zeros((h, 300, 3), np.uint8)
        pad = [[np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)]) for t in c] for c in cols]
        n_rows = max(len(c) for c in pad)
        grid = [np.hstack([c[i] if i < len(c) else blank for c in pad]) for i in range(n_rows)]
        sep = np.full((4, grid[0].shape[1], 3), 255, np.uint8)
        cv2.imwrite(os.path.join(img_dir, "gauge_examples.png"),
                    np.vstack([x for g_ in grid for x in (g_, sep)][:-1]))
    # 6) operating-envelope sweeps
    sw = summary.get("sweeps") or {}
    if sw:
        fig, axes = plt.subplots(1, len(sw), figsize=(5.6 * len(sw), 4.0))
        axes = np.atleast_1d(axes)
        for ax, (kind, d) in zip(axes, sw.items()):
            lv = [row["level"] for row in d["levels"]]
            det_r = [100 * row["detection_rate"] for row in d["levels"]]
            ok_r = [100 * row["correct_rate_5pct"] for row in d["levels"]]
            p95 = [row["error_fs"]["p95"] if row["error_fs"]["p95"] is not None else np.nan for row in d["levels"]]
            ax.plot(lv, det_r, "o-", color="#2a6fdb", label="accepted readings [%]")
            ax.plot(lv, ok_r, "s--", color="#1b9e77", label="frames read within 5 %FS [%]")
            ax.set_ylim(-5, 105)
            ax.set_ylabel("[%]")
            ax2 = ax.twinx()
            ax2.plot(lv, p95, "^:", color="#c51b7d", label="p95 error of accepted [%FS]")
            ax2.set_ylabel("p95 error [% FS]")
            ax2.set_ylim(0, max(5.5, np.nanmax(p95) * 1.15 if np.any(np.isfinite(p95)) else 5.5))
            if kind == "sweep_light":
                ax.set_xscale("log")
                ax.invert_xaxis()
                ax.set_xlabel("illumination scale (log, darker ->)  [gamma 1.3, shot+read noise]")
                ax.set_title("Operating envelope: low light")
            else:
                ax.set_xlabel("linear motion blur length [px]")
                ax.set_title("Operating envelope: motion blur")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, "gauge_sweeps.png"), dpi=130)
        plt.close(fig)


# ------------------------------------------------------------------------------------ pipeline figure
def pipeline_figure(img_dir: str, seed: int = 5) -> dict:
    """Illustrate the reader pipeline on one dim, oblique, slightly blurred frame (G04):
    raw frame, enhanced frame + fitted ellipse / needle, metric rectification, polar unwrap,
    angular profiles (tick ring vs gauge_design template, danger band, needle annulus)."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dusan_common import gauge_design as gd
    from dusan_common.layout import Layout
    from module2_inspection.gauge_reader import (NEEDLE_COARSE_BAND, POLAR_R, RED_BAND, THETA_DEG,
                                                 TICK_OUTER_BAND, GaugeReader, dial_to_canonical, draw_reading)
    from module2_inspection.gauge_synth import GaugeSpec, GaugeSynth, SynthConditions, ViewSpec
    g = Layout.load().gauge("G04")
    synth = GaugeSynth(seed=seed)
    reader = GaugeReader(camera_matrix=synth.camera.K)
    rng = np.random.default_rng(seed)
    view = ViewSpec(distance=1.35, azimuth_deg=30.0, elevation_deg=-16.0, roll_deg=4.0, offset_px=(-60.0, 30.0))
    cond = SynthConditions(illumination=0.16, gamma=1.3, noise_sigma=2.5, shot_noise=0.4, illum_gradient=0.4,
                           motion_blur_px=5.0, plate=True, clutter=16, distractors=1)
    img, gt = synth.render(GaugeSpec.from_layout(g), view, cond, rng)
    enh = reader.enhance(img)
    rd = reader.read_one(img, (float(g["min"]), float(g["max"])))
    if rd is None:
        return {}
    cx, cy, a = rd.ellipse[0], rd.ellipse[1], rd.ellipse[2]
    half = int(a * 1.8)
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
    vis = enh["vis"].copy()
    draw_reading(vis, rd, [f"G04 {rd.value:.2f} {g['unit']} (true {g['value']:.2f})"], "normal")
    # rectified, upright printed dial: printed-dial units -> image
    Hc = rd.H_dial2img @ dial_to_canonical(rd.rotation_deg, rd.mirrored)
    S = 360
    A = np.array([[1.0 / S, 0.0, -0.5], [0.0, -1.0 / S, 0.5], [0.0, 0.0, 1.0]])
    rect = cv2.warpPerspective(enh["vis"], Hc @ A, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    # polar unwrap in the printed-dial frame (theta = gauge_design angle)
    G = reader._polar(enh["meas"], Hc, POLAR_R)
    Cb = reader._polar(enh["bgr"], Hc, POLAR_R)
    face = np.percentile(G[(POLAR_R > 0.41) & (POLAR_R < 0.435)], 75, axis=0)
    ink = float(np.percentile(G[(POLAR_R >= 0.33) & (POLAR_R <= 0.405)], 1.0))
    D = (face[None, :] - G) / np.maximum(face - ink, 1.0)[None, :]
    red = np.clip(Cb[..., 2] - 0.5 * (Cb[..., 0] + Cb[..., 1]), 0, None) / np.maximum(face, 1.0)[None, :]

    def band(arr, r0, r1):
        m = (POLAR_R >= r0 - 1e-9) & (POLAR_R <= r1 + 1e-9)
        return arr[m].mean(axis=0)

    T_out, _T_in, T_red = reader.templates.profiles(float(g["min"]), float(g["max"]))
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    ax.set_title(f"(a) raw frame: light x{cond.illumination}, gamma {cond.gamma}, noise, "
                 f"{cond.motion_blur_px:.0f} px blur, {gt['incidence_deg']:.0f} deg", fontsize=9)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(cv2.cvtColor(vis[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    ax.set_title(f"(b) enhanced (gain {enh['gain']:.1f}, gamma {enh['gamma']:.2f}, CLAHE) + fit", fontsize=9)
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
    ax.set_title(f"(c) metric rectification (circle pose from the conic), rot {rd.rotation_deg:.1f} deg", fontsize=9)
    for k in range(3):
        fig.axes[k].axis("off")
    ax = fig.add_subplot(gs[1, 0:2])
    ext = [THETA_DEG[0], THETA_DEG[-1], POLAR_R[-1], POLAR_R[0]]
    ax.imshow(np.clip(D, -0.2, 1.2), aspect="auto", cmap="gray_r", extent=ext)
    for (r0, r1), c, lab in ((TICK_OUTER_BAND, "#1b9e77", "tick ring band"), (RED_BAND, "#d62728", "danger band"),
                             (NEEDLE_COARSE_BAND, "#2a6fdb", "needle annulus")):
        ax.axhspan(r0, r1, color=c, alpha=0.18, label=lab)
    phi_n = rd.angle_deg % 360.0
    ax.axvline(phi_n, color="#ff00ff", lw=1.2, ls="--", label=f"needle {rd.angle_deg:.2f} deg")
    for p in (gd.PHI_MIN_DEG % 360, gd.PHI_MAX_DEG % 360):
        ax.axvline(p, color="k", lw=0.8, ls=":")
    ax.set_xlabel("dial angle theta [deg] (gauge_design: CCW from 3 o'clock; min 225, max 315 = -45)")
    ax.set_ylabel("radius [dial units]")
    ax.set_title("(d) polar unwrap of the darkness map D = (face - I) / (face - ink)", fontsize=9)
    ax.legend(frameon=True, fontsize=7, loc="lower right")
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(THETA_DEG, band(D, *TICK_OUTER_BAND), color="#1b9e77", lw=0.8, label="tick ring (measured)")
    ax.plot(THETA_DEG, T_out + 1.2, color="#1b9e77", lw=0.8, ls="--", label="tick ring (template) + 1.2")
    ax.plot(THETA_DEG, band(red, *RED_BAND) * 2 - 1.0, color="#d62728", lw=0.8, label="redness x2 - 1")
    ax.plot(THETA_DEG, T_red * 2 - 1.0, color="#d62728", lw=0.6, ls=":", alpha=0.6, label="redness (template)")
    ax.plot(THETA_DEG, band(D, *NEEDLE_COARSE_BAND) - 2.2, color="#2a6fdb", lw=0.9, label="needle annulus - 2.2")
    ax.axvline(phi_n, color="#ff00ff", lw=1.0, ls="--")
    ax.set_xlim(0, 360)
    ax.set_yticks([])
    ax.set_xlabel("dial angle [deg]")
    ax.set_title(f"(e) angular profiles: calibration NCC {rd.scores.get('cal_ncc', 0):.2f}, "
                 f"value {rd.value:.2f} / {g['value']:.2f}", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "gauge_pipeline.png"), dpi=120)
    plt.close(fig)
    return {"gauge": "G04", "true": float(g["value"]), "read": float(rd.value),
            "error_fs": abs(rd.value - g["value"]) / (g["max"] - g["min"]) * 100.0,
            "confidence": float(rd.confidence), "gain": float(enh["gain"]), "gamma": float(enh["gamma"])}


# ------------------------------------------------------------------------------------ latency
def latency_benchmark(n: int, seed: int) -> dict:
    """Sequential per-frame latency in this process (OpenCV default threading, as in the node).

    full_frame : read_one() with a known range, whole 800x600 frame searched (no pose)
    pose_hint  : read() with a pose-predicted ROI hint (centre off by ~15 px, radius +-8 %) -
                 what gauge_reader_node does when /odom is available
    """
    from dusan_common.layout import Layout
    from module2_inspection.gauge_reader import GaugeHint, GaugeReader
    from module2_inspection.gauge_synth import GaugeSpec, GaugeSynth, random_conditions, random_view
    gauges = Layout.load().gauges
    synth = GaugeSynth(seed=1)
    reader = GaugeReader(camera_matrix=synth.camera.K)
    rng = np.random.default_rng(seed + 7)
    frames = []
    for k in range(n):
        g = gauges[k % len(gauges)]
        kind = REQUIRED[k % len(REQUIRED)]
        img, gt = synth.render(GaugeSpec.from_layout(g, float(rng.uniform(g["min"], g["max"]))),
                               random_view(rng, kind), random_conditions(rng, kind), rng)
        frames.append((g, img, gt))
    reader.read_one(frames[0][1], (frames[0][0]["min"], frames[0][0]["max"]))        # warm-up (templates)
    t_full, t_hint = [], []
    for g, img, gt in frames:
        t0 = time.perf_counter()
        reader.read_one(img, (float(g["min"]), float(g["max"])))
        t_full.append((time.perf_counter() - t0) * 1000.0)
        du, dv = rng.normal(0, 15, 2)
        hint = GaugeHint(gauge_id=g["id"], center=(gt["centre_px"][0] + du, gt["centre_px"][1] + dv),
                         radius=gt["radius_px"] * float(rng.uniform(0.92, 1.08)),
                         vrange=(float(g["min"]), float(g["max"])), do_ocr=False)
        t0 = time.perf_counter()
        reader.read(img, [hint], detect_unhinted=False)
        t_hint.append((time.perf_counter() - t0) * 1000.0)

    def st(v):
        v = np.asarray(v)
        return {"median_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90)), "max_ms": float(v.max())}
    return {"frames": n, "full_frame": st(t_full), "pose_hint": st(t_hint)}


# ------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=480, help="frames over the required condition grid")
    ap.add_argument("--extra", type=int, default=60, help="extra occluded frames")
    ap.add_argument("--empty", type=int, default=60, help="background-only frames (false positives)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    ap.add_argument("--images", default=os.path.join(ROOT, "docs", "images"))
    ap.add_argument("--json", default=os.path.join(ROOT, "docs", "results", "gauge_eval.json"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--latency", type=int, default=24, help="frames of the sequential latency test (0 = skip)")
    ap.add_argument("--sweep", type=int, default=10, help="frames per level of the envelope sweeps (0 = skip)")
    args = ap.parse_args(argv)

    from dataclasses import asdict
    from module2_inspection.gauge_reader import GaugeReaderConfig
    jobs = make_jobs(args.frames, args.extra, args.empty, args.seed, args.sweep)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    if workers == 1:
        results = [run_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            results = list(ex.map(run_job, jobs, chunksize=4))
    wall = time.time() - t0
    summary = summarize(results, asdict(GaugeReaderConfig()))
    summary["wall_time_s"] = wall
    summary["seed"] = args.seed
    summary["workers"] = workers
    if args.latency > 0:
        summary["latency"] = latency_benchmark(args.latency, args.seed)
    if not args.no_plots:
        make_plots(results, summary, args.images)
        summary["pipeline_example"] = pipeline_figure(args.images)
    frames = []
    for r in results:
        rr = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "example_png"}
        frames.append(rr)
    summary["frames"] = frames
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o))
    # ---- console table
    print(f"\nGauge reader benchmark: {len(results)} frames in {wall:.0f} s ({workers} workers)\n")
    print(f"{'condition':12s} {'frames':>6s} {'det%':>6s} {'mean':>6s} {'median':>7s} {'p95':>6s} {'max':>7s} "
          f"{'<2%':>5s} {'OCR%':>5s} {'ms':>5s}")
    for k in REQUIRED + EXTRA:
        s = summary["by_condition"][k]
        e = s["error_fs"]
        if e["n"] == 0:
            print(f"{k:12s} {s['frames']:6d}   none")
            continue
        print(f"{k:12s} {s['frames']:6d} {100 * s['detection_rate']:6.1f} {e['mean']:6.2f} {e['median']:7.2f} "
              f"{e['p95']:6.2f} {e['max']:7.2f} {100 * s['within_2pct']:5.0f} {100 * s['ocr_range_read_rate']:5.0f} "
              f"{s['mean_time_ms']:5.0f}")
    o = summary["overall_required_grid"]
    e = o["error_fs"]
    print(f"\nOVERALL (required grid): detection {100 * o['detection_rate']:.1f} %, mean {e['mean']:.2f} %FS, "
          f"median {e['median']:.2f}, p95 {e['p95']:.2f}, max {e['max']:.2f}  -> targets "
          f"mean<2: {summary['targets']['mean_lt_2pct']}, p95<5: {summary['targets']['p95_lt_5pct']}")
    oc = summary["ocr"]
    print(f"OCR (range unknown): read {oc['range_read']}/{oc['frames']}, correct {oc['range_correct']}, "
          f"precision {oc['precision']}")
    fp = summary["false_positives"]
    print(f"False detections on {fp['frames']} background frames: {fp['false_detections']}")
    for kind, d in (summary.get("sweeps") or {}).items():
        print(f"{kind} ({d['parameter']}): " + "  ".join(
            f"{row['level']:g}:{100 * row['detection_rate']:.0f}%/"
            + (f"{row['error_fs']['p95']:.1f}" if row["error_fs"]["p95"] is not None else "-")
            for row in d["levels"]) + "   (level: accepted % / p95 %FS)")
    if "latency" in summary:
        la = summary["latency"]
        print(f"Latency (sequential, {la['frames']} frames): full frame median {la['full_frame']['median_ms']:.0f} ms "
              f"(p90 {la['full_frame']['p90_ms']:.0f}), pose-hinted ROI median {la['pose_hint']['median_ms']:.0f} ms "
              f"(p90 {la['pose_hint']['p90_ms']:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
