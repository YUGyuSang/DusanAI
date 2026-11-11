#!/usr/bin/env python3
"""
Offline benchmark of the Module 2 thermal pipeline (runs on Windows, no ROS).

For many robot poses near the layout hot spots (plus negative poses with no
hot spot in view) it renders radiometric thermal frames with
``synthetic_thermal.SyntheticThermalCamera`` and a shaded RGB proxy of the
same scene from the inspection camera, then runs ``ThermalFusion`` and
measures

* hot-spot detection precision / recall (single frame and after the 3-frame
  temporal confirmation), id accuracy of the ray -> layout association,
  max-temperature error and status accuracy;
* registration error (RGB pixels) of the thermal -> RGB homography for known
  3-D surface points, for several depth assumptions (fixed 2 m, layout median,
  matched hot spot) with and without ECC refinement, and the theoretical
  parallax error  f_rgb * b * |1/Z - 1/d|;
* run time of the renderer and the fusion.

Outputs: docs/results/thermal_eval.json, docs/images/thermal_examples.png,
docs/images/thermal_registration.png, docs/images/thermal_detection.png

    python tools/eval_thermal.py [--poses-per-hotspot 24] [--negatives 30] [--seed 0]
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("dusan_common", "module2_inspection"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dusan_common.camera import world_to_camera_link  # noqa: E402
from dusan_common.grid_map import OccupancyGridMap  # noqa: E402
from dusan_common.layout import Layout  # noqa: E402
from module2_inspection.synthetic_thermal import (LBL_HOT, SyntheticThermalCamera, full_pose,  # noqa: E402
                                                  render_visible_proxy)
from module2_inspection.thermal_fusion import (CameraRig, FusionParams, ThermalFusion, apply_homography,  # noqa: E402
                                               colorize_temperature, refine_homography_ecc)
from module2_inspection.viewpoint_planner import layout_occupancy_map  # noqa: E402

IMG_DIR = os.path.join(ROOT, "docs", "images")
RES_DIR = os.path.join(ROOT, "docs", "results")
HOT_TEMP_K = {}          # hot-spot id -> temp_k (filled by run(), used to pick the figure example)


# =============================================================================
# Scenario generation
# =============================================================================
def load_grid(layout):
    path = layout.nav_map_yaml()
    if os.path.exists(path):
        return OccupancyGridMap.load(path), os.path.relpath(path, ROOT).replace("\\", "/")
    return layout_occupancy_map(layout), "rasterised from world_layout.yaml (plant_map.yaml not found)"


def sample_hotspot_poses(layout, grid, rng, n_per_hotspot, r_range=(1.2, 5.0)):
    """Robot poses looking (with jitter) at each hot spot from free, line-of-sight positions."""
    out = []
    for h in layout.hotspots:
        hx, hy, _ = h["position"]
        got, tries = 0, 0
        while got < n_per_hotspot and tries < 4000:
            tries += 1
            r = rng.uniform(*r_range)
            a = rng.uniform(-math.pi, math.pi)
            x, y = hx + r * math.cos(a), hy + r * math.sin(a)
            if not grid.is_free(x, y, 0.35):
                continue
            # 2-D sight line, ignoring the last 0.3 m (the hot object itself may be an occupied cell)
            d = math.hypot(hx - x, hy - y)
            ex, ey = x + (hx - x) * (d - 0.3) / d, y + (hy - y) * (d - 0.3) / d
            if not grid.line_of_sight(x, y, ex, ey):
                continue
            yaw = math.atan2(hy - y, hx - x) + rng.uniform(-0.22, 0.22)
            pose = full_pose(layout, x, y, yaw, 0.34 + rng.uniform(-0.03, 0.03),
                             roll=rng.normal(0, 0.02), pitch=rng.normal(0, 0.02))
            out.append({"pose": pose, "target": h["id"], "range": r})
            got += 1
    return out


def sample_negative_poses(layout, grid, rng, n):
    out = []
    x0, y0, x1, y1 = layout["nav_map"]["bounds"]
    while len(out) < n:
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if not grid.is_free(x, y, 0.4):
            continue
        out.append({"pose": full_pose(layout, x, y, rng.uniform(-math.pi, math.pi)), "target": None, "range": None})
    return out


# =============================================================================
# Metrics helpers
# =============================================================================
def visible_hot_ids(layout, label, min_px):
    """Layout hot spots with >= min_px visible pixels (z-buffer labels)."""
    out = {}
    for i, h in enumerate(layout.hotspots):
        n = int(np.count_nonzero(label == LBL_HOT + i))
        if n >= min_px:
            out[h["id"]] = n
    return out


def true_identity(layout, label, det):
    """Hot spot the blob really belongs to: label at its peak pixel (or the
    most frequent hot-spot label inside its bbox)."""
    u, v = det["peak_px"]
    lab = int(label[v, u])
    if LBL_HOT <= lab < LBL_HOT + len(layout.hotspots):
        return layout.hotspots[lab - LBL_HOT]["id"]
    bu, bv, bw, bh = det["bbox"]
    sub = label[bv:bv + bh, bu:bu + bw]
    hot = sub[(sub >= LBL_HOT) & (sub < LBL_HOT + len(layout.hotspots))]
    if hot.size:
        vals, cnt = np.unique(hot, return_counts=True)
        return layout.hotspots[int(vals[np.argmax(cnt)]) - LBL_HOT]["id"]
    return None


def stats(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": int(a.size), "mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3)}


def registration_points(buf, layout, max_depth=15.0, n_max=400, rng=None):
    """3-D surface points seen by the thermal camera: hot-spot surfaces and
    a random subset of all pixels (world coordinates)."""
    lab = buf.label
    depth = buf.depth
    P = buf.point
    hot = (lab >= LBL_HOT) & (lab < LBL_HOT + len(layout.hotspots))
    allm = (depth > 0.2) & (depth < max_depth)
    ph, phl = P[hot], lab[hot]
    pa = P[allm]
    if rng is not None and len(pa) > n_max:
        pa = pa[rng.choice(len(pa), n_max, replace=False)]
    if rng is not None and len(ph) > n_max:
        k = rng.choice(len(ph), n_max, replace=False)
        ph, phl = ph[k], phl[k]
    return ph, phl, pa


def reg_errors(H, rig, pose, pts):
    if len(pts) == 0:
        return np.zeros(0)
    T_th, T_rgb = rig.thermal_world(pose), rig.rgb_world(pose)
    uv_t, _ = rig.thermal.project_link(world_to_camera_link(T_th, pts))
    uv_r, _ = rig.rgb.project_link(world_to_camera_link(T_rgb, pts))
    return np.linalg.norm(apply_homography(H, uv_t) - uv_r, axis=1)


# =============================================================================
# Main benchmark
# =============================================================================
CONDITIONS = {   # thermal sensor degradations (the scene and the poses are shared)
    "nominal": dict(netd_k=0.05, column_fpn_k=0.03, row_noise_k=0.02, blur_sigma_px=0.6),
    "noisy": dict(netd_k=0.40, column_fpn_k=0.30, row_noise_k=0.15, blur_sigma_px=0.6),
    "defocused": dict(netd_k=0.05, column_fpn_k=0.03, row_noise_k=0.02, blur_sigma_px=1.6),
}


def _new_counter():
    return {"tp": 0, "fp": 0, "fn": 0, "id_ok": 0}


def _prf(c):
    p = c["tp"] / max(c["tp"] + c["fp"], 1)
    r = c["tp"] / max(c["tp"] + c["fn"], 1)
    return {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-12), 4), "id_accuracy": round(c["id_ok"] / max(c["tp"], 1), 4)}


def score_detections(layout, lab, vis, dets, counter, hot_gt, acc=None):
    """Count TP / FP / FN of blob detections against the visible hot spots."""
    found = set()
    for d in dets:
        if d.get("source") != "blob":
            continue
        tid = true_identity(layout, lab, d)
        if tid is not None and tid in vis and tid not in found:
            found.add(tid)
            counter["tp"] += 1
            counter["id_ok"] += int(d.get("id") == tid)
            if acc is not None:
                e = abs(d["max_temp_k"] - hot_gt[tid]["temp_k"])
                acc["temp_err"].append(e)
                acc["per_hot"][tid]["temp_err"].append(e)
                acc["per_hot"][tid]["id_ok"] += int(d.get("id") == tid)
                acc["status_n"] += 1
                acc["status_ok"] += int(d["status"] == layout.temperature_status(hot_gt[tid]["temp_k"]))
        else:
            counter["fp"] += 1               # not a visible hot spot, or a duplicate blob
    counter["fn"] += len([h for h in vis if h not in found])
    return found


def run(args):
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    layout = Layout.load()
    grid, map_source = load_grid(layout)
    base_cam = SyntheticThermalCamera(layout, seed=args.seed)
    cams = {name: SyntheticThermalCamera(layout, seed=args.seed + 1 + i, scene=base_cam.scene, **kw)
            for i, (name, kw) in enumerate(CONDITIONS.items())}
    rig = CameraRig.from_robot_model()
    warning_k = layout.thermal["warning_k"]
    hot_gt = {h["id"]: h for h in layout.hotspots}
    HOT_TEMP_K.update({h["id"]: float(h["temp_k"]) for h in layout.hotspots})

    scenarios = sample_hotspot_poses(layout, grid, rng, args.poses_per_hotspot, (1.2, args.max_range))
    scenarios += sample_negative_poses(layout, grid, rng, args.negatives)
    print(f"{len(scenarios)} poses x {len(cams)} sensor conditions ({map_source})")

    M = {}
    for name in cams:
        M[name] = {"single": _new_counter(), "confirmed": _new_counter(), "temp_err": [], "status_ok": 0,
                   "status_n": 0, "per_hot": {h: {"visible": 0, "detected": 0, "id_ok": 0, "temp_err": []} for h in hot_gt},
                   "by_range": [], "neg_fp_frames": 0, "neg_frames": 0}
    normal_roi = {"visible": 0, "reported_normal": 0}
    strategies = ("fixed_2m", "layout_median", "layout_median+ecc", "hotspot_plane", "pipeline",
                  "per_detection_plane")
    reg_hot = {s_: [] for s_ in strategies}
    reg_all = {s_: [] for s_ in strategies}
    reg_scatter = []
    ecc_stats = {"runs": 0, "accepted": 0, "shift": []}
    t_render, t_rgb, t_fuse = [], [], []
    examples = []

    for k, sc in enumerate(scenarios):
        pose = sc["pose"]
        t0 = time.time()
        rgb, _ = render_visible_proxy(base_cam.scene, pose, rng=rng)
        t_rgb.append(time.time() - t0)
        jitter = [tuple(np.asarray(pose) + np.r_[rng.normal(0, 0.005, 2), 0.0, 0.0, 0.0, rng.normal(0, 0.002)])
                  for _ in range(3)]                     # 3 frames, tiny motion between them
        for name, cam in cams.items():
            m = M[name]
            fus = ThermalFusion(layout, FusionParams())
            confirmed, first = [], None
            for f, jp in enumerate(jitter):
                t0 = time.time()
                K = cam.render_kelvin(jp)
                if name == "nominal":
                    t_render.append(time.time() - t0)
                t0 = time.time()
                res = fus.process(K, rgb, jp, stamp=0.2 * f, render=(f == len(jitter) - 1 and name == "nominal"))
                if name == "nominal":
                    t_fuse.append(time.time() - t0)
                if f == 0:
                    first = (res, K, cam.last_buffers)
                confirmed, last_res, last_K = res.confirmed, res, K
            res0, K0, buf0 = first
            lab = buf0.label
            allvis = visible_hot_ids(layout, lab, args.min_visible_px)
            vis = {h for h in allvis if hot_gt[h]["temp_k"] >= warning_k}
            found = score_detections(layout, lab, vis, res0.detections, m["single"], hot_gt, m)
            score_detections(layout, lab, vis, confirmed, m["confirmed"], hot_gt)
            for h in vis:
                m["per_hot"][h]["visible"] += 1
                m["per_hot"][h]["detected"] += int(h in found)
                if sc["target"] == h:
                    m["by_range"].append((sc["range"], int(h in found)))
            if not vis:
                m["neg_frames"] += 1
                m["neg_fp_frames"] += int(any(d.get("source") == "blob" for d in res0.detections))
            if name != "nominal":
                continue
            # ---------------- normal (below-threshold) hot spots reported from their ROI
            for h in allvis:
                if hot_gt[h]["temp_k"] < warning_k:
                    normal_roi["visible"] += 1
                    normal_roi["reported_normal"] += int(any(d.get("id") == h and d["status"] == "normal"
                                                             for d in res0.detections))
            # ---------------- registration (nominal sensor)
            ph, phl, pa = registration_points(buf0, layout, rng=rng)
            H_fixed, _, _ = rig.plane_homography(2.0, "fronto", pose)
            d_lay, _, _ = fus.estimate_plane(pose, [], rig)
            H_lay, _, _ = rig.plane_homography(d_lay, "vertical", pose)
            d_hot, _, pt_hot = fus.estimate_plane(pose, res0.detections, rig)
            H_hot, _, _ = rig.plane_homography(d_hot, "vertical", pose, pt_hot)
            ecc_lay = refine_homography_ecc(K0, rgb, H_lay, epipolar_dir=rig.epipolar_direction())
            ecc_stats["runs"] += 1
            ecc_stats["accepted"] += int(ecc_lay.ok)
            if ecc_lay.ok:
                ecc_stats["shift"].append(abs(ecc_lay.shift_px[0]))
            Hs = {"fixed_2m": H_fixed, "layout_median": H_lay, "layout_median+ecc": ecc_lay.H,
                  "hotspot_plane": H_hot, "pipeline": res0.H}
            for s_, H in Hs.items():
                reg_hot[s_].extend(reg_errors(H, rig, pose, ph).tolist())
                reg_all[s_].extend(reg_errors(H, rig, pose, pa).tolist())
            # boxes / peaks of matched detections use the plane through their own hot spot
            Hdet = {d["id"]: d.get("_H", res0.H) for d in res0.detections if d.get("id") and d.get("source") == "blob"}
            for i, h in enumerate(layout.hotspots):
                sel = phl == LBL_HOT + i
                if np.any(sel):
                    reg_hot["per_detection_plane"].extend(reg_errors(Hdet.get(h["id"], res0.H), rig, pose, ph[sel]).tolist())
            reg_all["per_detection_plane"].extend(reg_errors(res0.H, rig, pose, pa).tolist())
            if len(ph):
                T_cw = np.linalg.inv(rig.thermal_world(pose))
                z = (ph @ T_cw[:3, :3].T + T_cw[:3, 3])[:, 0]
                e_f, e_p = reg_errors(H_fixed, rig, pose, ph), reg_errors(res0.H, rig, pose, ph)
                for i in rng.choice(len(ph), min(15, len(ph)), replace=False):
                    reg_scatter.append((float(z[i]), float(e_f[i]), float(e_p[i])))
            if (sc["target"] and len(examples) < 5 and sc["target"] not in [e[0] for e in examples]
                    and last_res.fusion is not None and confirmed and 1.6 < sc["range"] < 3.5):
                examples.append((sc["target"], last_K, rgb.copy(), last_res.fusion))
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(scenarios)} poses")

    bins = [1.0, 2.0, 3.0, 4.0, 5.0, max(7.0, args.max_range)]

    def by_range(rows):
        rr = np.array(rows, dtype=float).reshape(-1, 2)
        out = []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = (rr[:, 0] >= a) & (rr[:, 0] < b)
            out.append({"range_m": [a, b], "n": int(sel.sum()),
                        "recall": None if not sel.any() else round(float(rr[sel, 1].mean()), 4)})
        return out

    per_condition = {}
    for name, m in M.items():
        per_condition[name] = {
            "sensor": CONDITIONS[name],
            "detection_single_frame": _prf(m["single"]),
            "detection_confirmed_3_frames": _prf(m["confirmed"]),
            "max_temperature_error_k": stats(m["temp_err"]),
            "status_accuracy": round(m["status_ok"] / max(m["status_n"], 1), 4),
            "false_alarm_rate_negative_frames": round(m["neg_fp_frames"] / max(m["neg_frames"], 1), 4),
            "recall_by_range": by_range(m["by_range"]),
            "per_hotspot": {h: {"temp_k": hot_gt[h]["temp_k"], "visible": v["visible"], "detected": v["detected"],
                                "recall": None if not v["visible"] else round(v["detected"] / v["visible"], 4),
                                "id_ok": v["id_ok"], "temp_err_k": stats(v["temp_err"])}
                            for h, v in m["per_hot"].items()},
        }
    f_rgb = rig.rgb.fx
    result = {
        "description": "Thermal pipeline benchmark on synthetic radiometric frames (synthetic_thermal) + RGB proxy",
        "map_source": map_source,
        "n_poses": len(scenarios), "n_hotspot_poses": len(scenarios) - args.negatives,
        "n_negative_poses": args.negatives, "frames_per_pose": 3, "range_m": [1.2, args.max_range], "seed": args.seed,
        "thresholds_k": {"warning": warning_k, "alarm": layout.thermal["alarm_k"]},
        "conditions": per_condition,
        "normal_hotspot_roi_reports": {**normal_roi,
                                       "rate": round(normal_roi["reported_normal"] / max(normal_roi["visible"], 1), 4)},
        "registration_error_px": {
            "hotspot_surface_points": {s_: stats(v) for s_, v in reg_hot.items()},
            "all_surface_points": {s_: stats(v) for s_, v in reg_all.items()},
            "theory": {"baseline_m": 0.05, "f_rgb_px": round(f_rgb, 2), "formula": "e = f_rgb * b * |1/Z - 1/d|",
                       "fixed_2m_examples": {f"Z={z}m": round(f_rgb * 0.05 * abs(1 / z - 0.5), 2)
                                             for z in (1.0, 1.5, 3.0, 5.0, 10.0)}},
        },
        "ecc_on_layout_median_plane": {"runs": ecc_stats["runs"], "accepted": ecc_stats["accepted"],
                                       "accept_rate": round(ecc_stats["accepted"] / max(ecc_stats["runs"], 1), 4),
                                       "abs_shift_px": stats(ecc_stats["shift"])},
        "timing_ms": {"thermal_render": stats(np.array(t_render) * 1000),
                      "rgb_proxy_render": stats(np.array(t_rgb) * 1000),
                      "fusion_process": stats(np.array(t_fuse) * 1000)},
    }
    with open(os.path.join(RES_DIR, "thermal_eval.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
    make_figures(result, examples, reg_scatter, f_rgb)
    for name, c in per_condition.items():
        d1, d3 = c["detection_single_frame"], c["detection_confirmed_3_frames"]
        print(f"{name:10s} single P={d1['precision']:.3f} R={d1['recall']:.3f} id={d1['id_accuracy']:.3f} | "
              f"confirmed P={d3['precision']:.3f} R={d3['recall']:.3f} | dT median {c['max_temperature_error_k']['median']} K"
              f" | status {c['status_accuracy']:.3f}")
    print("registration (hot-spot points, median / p95 px):",
          {s_: (v["median"], v["p95"]) for s_, v in result["registration_error_px"]["hotspot_surface_points"].items()})
    return result


# =============================================================================
# Figures
# =============================================================================
def make_figures(result, examples, scatter, f_rgb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- examples panel (thermal | RGB proxy | fusion)
    if examples:
        rows = []
        for hid, K, rgb, fusion in examples:
            th = cv2.resize(colorize_temperature(K, 293.15, 393.15), (800, 600), interpolation=cv2.INTER_NEAREST)
            cv2.putText(th, f"thermal 320x240 ({hid})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(rgb, "RGB proxy 800x600", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            rows.append(np.hstack([th, rgb, fusion]))
        panel = np.vstack(rows)
        panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(IMG_DIR, "thermal_examples.png"), panel)
        # single example image: the hottest target (an alarm shows every overlay element)
        hid, _, _, fusion = max(examples, key=lambda e: HOT_TEMP_K.get(e[0], 0.0))
        cv2.imwrite(os.path.join(IMG_DIR, "thermal_fusion_example.png"), fusion)

    # ---- registration error
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    z = np.linspace(0.6, 12, 300)
    for d, ls in ((1.0, ":"), (2.0, "-"), (5.0, "--")):
        ax[0].plot(z, f_rgb * 0.05 * np.abs(1 / z - 1 / d), ls, color="#555555", label=f"theory, plane d = {d:g} m")
    if scatter:
        s = np.array(scatter)
        ax[0].scatter(s[:, 0], s[:, 1], s=8, color="#d1495b", alpha=0.6, label="measured, fixed 2 m plane")
        ax[0].scatter(s[:, 0], s[:, 2], s=8, color="#2e86ab", alpha=0.6, label="measured, pipeline (selected plane / ECC)")
    ax[0].set_xlabel("true depth Z of the surface point (m)")
    ax[0].set_ylabel("registration error (RGB px)")
    ax[0].set_title("Thermal -> RGB homography: parallax error vs depth")
    ax[0].set_ylim(0, 30)
    ax[0].set_xlim(0, 15)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    reg = result["registration_error_px"]["hotspot_surface_points"]
    names = list(reg)
    med = [reg[n]["median"] or 0 for n in names]
    p95 = [reg[n]["p95"] or 0 for n in names]
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, med, 0.4, label="median", color="#2e86ab")
    ax[1].bar(x + 0.2, p95, 0.4, label="95th percentile", color="#f6ae2d")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([n.replace("+", "\n+").replace("_plane", "\nplane") for n in names], fontsize=8)
    ax[1].set_ylabel("error on hot-spot surfaces (RGB px)")
    ax[1].set_title("Registration error by depth strategy")
    ax[1].grid(axis="y", alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "thermal_registration.png"), dpi=130)
    plt.close(fig)

    # ---- detection summary (per sensor condition)
    conds = result["conditions"]
    colors = {"nominal": "#2e86ab", "noisy": "#f6ae2d", "defocused": "#d1495b"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    names = list(conds)
    br0 = conds[names[0]]["recall_by_range"]
    xl = [f"{b['range_m'][0]:g}-{b['range_m'][1]:g}" for b in br0]
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [b["recall"] if b["recall"] is not None else 0.0 for b in conds[n]["recall_by_range"]]
        ax[0].bar(np.arange(len(xl)) + (i - (len(names) - 1) / 2) * w, vals, w, label=n, color=colors.get(n))
    for j, b in enumerate(br0):
        ax[0].text(j, 1.03, f"n={b['n']}", ha="center", fontsize=7)
    ax[0].set_xticks(np.arange(len(xl)))
    ax[0].set_xticklabels(xl)
    ax[0].set_ylim(0, 1.32)
    ax[0].set_xlabel("range to target hot spot (m)")
    ax[0].set_ylabel("single-frame recall")
    ax[0].set_title("Detection recall vs range")
    ax[0].legend(fontsize=8, loc="upper center", ncol=3)
    ids = [h for h in conds["nominal"]["per_hotspot"] if conds["nominal"]["per_hotspot"][h]["visible"]]
    for i, n in enumerate(names):
        ph = conds[n]["per_hotspot"]
        errs = [ph[h]["temp_err_k"]["median"] or 0.0 for h in ids]
        ax[1].bar(np.arange(len(ids)) + (i - (len(names) - 1) / 2) * w, errs, w, label=n, color=colors.get(n))
    ax[1].set_xticks(np.arange(len(ids)))
    ax[1].set_xticklabels([f"{h}\n{conds['nominal']['per_hotspot'][h]['temp_k'] - 273.15:.0f}C" for h in ids])
    ax[1].set_ylabel("median |max T - true T| (K)")
    ax[1].set_title("Max-temperature error per hot spot")
    ax[1].legend(fontsize=8)
    labels = ["precision", "recall", "F1", "id acc.", "status acc."]
    for i, n in enumerate(names):
        d3 = conds[n]["detection_confirmed_3_frames"]
        vals = [d3["precision"], d3["recall"], d3["f1"], d3["id_accuracy"], conds[n]["status_accuracy"]]
        ax[2].bar(np.arange(5) + (i - (len(names) - 1) / 2) * w, vals, w, label=n, color=colors.get(n))
    ax[2].set_xticks(np.arange(5))
    ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].set_ylim(0, 1.25)
    ax[2].set_title("Confirmed detections (3 frames)")
    ax[2].legend(fontsize=8, loc="upper center", ncol=3)
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "thermal_detection.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses-per-hotspot", type=int, default=24)
    ap.add_argument("--negatives", type=int, default=30)
    ap.add_argument("--max-range", type=float, default=7.0)
    ap.add_argument("--min-visible-px", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
