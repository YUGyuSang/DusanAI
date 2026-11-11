#!/usr/bin/env python3
"""
Offline evaluation of the Module 1 terrain perception pipeline (no ROS needed).

For seven synthetic scenes (flat, scattered pipes, rubble, 0.17 m stairs up,
the same stairs down, 16 deg ramp, a 0.6 m obstacle) a robot walks towards
the feature.  At every 10 Hz frame
the depth camera of ``dusan_common.robot_model.SENSORS['depth_camera']``
(212 x 120, hfov 1.5 rad, pitched 0.6 rad down) is ray-cast against a
5 mm ground-truth height field, corrupted with range-dependent Gaussian noise,
spurious near returns and drop-outs, packed into a Gazebo-like organised
PointCloud2 buffer (x y z rgb, point_step 16, inf = no return) and pushed
through EXACTLY the node pipeline:

    pointcloud2_to_xyz -> ConventionDetector -> camera_points_to_world (noisy
    odometry pose) -> ElevationMap.recenter/integrate -> layers ->
    TerrainClassifier (hysteresis) ;  FootholdPlanner on the final maps.

Reported (docs/results/terrain_eval.json, figures docs/images/terrain_*.png):
  * elevation RMSE / MAE / p95 / bias vs ground truth (all known cells and
    cells away from height discontinuities) and coverage of the feature area
  * classification + recommended gait per scene at the decision pose (feature
    ~0.8-0.9 m ahead; 0.75 m for the descending stairs, whose drop is hidden
    behind the edge until then) and along the whole approach
  * foothold safety: share of foot placements whose 5 cm neighbourhood spans a
    height jump > 3 cm (edge / pipe flank / block edge), nominal vs planned
  * point-cloud convention auto-detection accuracy (link and optical clouds)
  * ablations: higher-outlier confirmation, max-height bias and the
    occlusion-shadow hole fill
  * timing of every stage on this machine

Usage:  python tools/eval_terrain.py [--quick] [--no-figures] [--seed N]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("dusan_common", "module1_locomotion"):
    _d = os.path.join(ROOT, _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dusan_common.camera import PinholeCamera, camera_world_pose          # noqa: E402
from dusan_common.robot_model import HIP_X_OFFSET, NOMINAL_BODY_HEIGHT, SENSORS  # noqa: E402
from module1_locomotion.elevation_map import ElevationMap, ElevationMapParams, max_filter, min_filter  # noqa: E402
from module1_locomotion.foothold_planner import FootholdPlanner           # noqa: E402
from module1_locomotion.pointcloud_utils import (                          # noqa: E402
    ConventionDetector, DepthIntrinsics, camera_points_to_world, depth_image_to_points,
    detect_convention, pointcloud2_to_xyz)
from module1_locomotion.terrain_classifier import TerrainClassifier       # noqa: E402

IMG_DIR = os.path.join(ROOT, "docs", "images")
RES_DIR = os.path.join(ROOT, "docs", "results")
FOOT_CHECK_RADIUS = 0.05        # foot radius 0.025 + 0.025 placement uncertainty
UNSAFE_JUMP = 0.03              # height range inside that disk that makes a foothold unsafe

# ---------------------------------------------------------------- palette
# Reference data-viz palette (light mode), categorical slots in fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
UNKNOWN_FILL = "#f0efec"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde7dc", "#f8b99c", "#f28b61", "#eb6834", "#c4501f", "#93391a", "#5f2411"]
INK_RAMP = ["#f4f3f0", "#d5d4cd", "#a9a79f", "#7c7a73", "#52514e", "#2a2928", "#0b0b0b"]
CLASS_COLORS = {"flat": SLOTS[0], "rough": SLOTS[1], "stairs": SLOTS[2], "slope": SLOTS[3],
                "obstacle": SLOTS[4], "unknown": AXIS}
GAIT_COLORS = {"trot": GRID, "crawl": INK2}


# ============================================================ ground truth
class HeightField:
    """Ground-truth terrain sampled on a fine grid (nearest lookup)."""

    def __init__(self, x_min=-2.5, x_max=5.0, y_min=-2.5, y_max=2.5, res=0.005):
        self.x0, self.y0, self.res = x_min, y_min, res
        nx = int(round((x_max - x_min) / res)) + 1
        ny = int(round((y_max - y_min) / res)) + 1
        xs = x_min + np.arange(nx) * res
        ys = y_min + np.arange(ny) * res
        self.X, self.Y = np.meshgrid(xs, ys)
        self.Z = np.zeros_like(self.X)

    def add(self, z):
        self.Z = np.maximum(self.Z, z)

    def __call__(self, x, y):
        c = np.clip(np.rint((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.Z.shape[1] - 1)
        r = np.clip(np.rint((np.asarray(y) - self.y0) / self.res).astype(int), 0, self.Z.shape[0] - 1)
        return self.Z[r, c]

    # primitives --------------------------------------------------------
    def pipe(self, xc, yc, yaw, length, diameter):
        r = diameter / 2.0
        s = (self.X - xc) * math.cos(yaw) + (self.Y - yc) * math.sin(yaw)
        q = -(self.X - xc) * math.sin(yaw) + (self.Y - yc) * math.cos(yaw)
        inside = (np.abs(s) <= length / 2) & (np.abs(q) < r)
        self.add(np.where(inside, r + np.sqrt(np.maximum(r * r - q * q, 0.0)), 0.0))

    def box(self, xc, yc, sx, sy, yaw, h):
        s = (self.X - xc) * math.cos(yaw) + (self.Y - yc) * math.sin(yaw)
        q = -(self.X - xc) * math.sin(yaw) + (self.Y - yc) * math.cos(yaw)
        self.add(np.where((np.abs(s) <= sx / 2) & (np.abs(q) <= sy / 2), h, 0.0))

    def stairs(self, x0, rise, tread, n):
        self.add(np.clip(np.floor((self.X - x0) / tread) + 1, 0, n) * rise)

    def stairs_down(self, x0, rise, tread, n):
        """Upper landing (n * rise) for x < x0, then n steps DOWN to 0."""
        self.add(n * rise - np.clip(np.floor((self.X - x0) / tread) + 1, 0, n) * rise)

    def ramp(self, x0, deg, length):
        self.add(np.clip(self.X - x0, 0.0, length) * math.tan(math.radians(deg)))


@dataclass
class Scene:
    name: str
    title: str
    expected_class: str
    expected_gait: str
    hf: HeightField
    path_x: np.ndarray
    decision_x: float
    eval_rect: tuple                     # x_min, y_min, x_max, y_max
    edges_x: List[float] = field(default_factory=list)
    foothold_rect: Optional[tuple] = None


def build_scenes(quick: bool = False) -> Dict[str, Scene]:
    step = 0.2 if quick else 0.1
    sc = {}
    hf = HeightField()
    sc["flat"] = Scene("flat", "Flat floor", "flat", "trot", hf, np.arange(-1.2, 0.51, step), 0.1,
                       (0.4, -0.8, 2.4, 0.8))
    hf = HeightField()
    for xc, yc, yaw_deg, length, d in ((1.10, -0.30, 80, 2.4, 0.10), (1.65, 0.40, 100, 2.0, 0.12),
                                       (2.20, 0.00, 62, 2.5, 0.08), (2.80, -0.50, 95, 2.2, 0.10)):
        hf.pipe(xc, yc, math.radians(yaw_deg), length, d)
    sc["pipes"] = Scene("pipes", "Scattered pipes (d 0.08-0.12 m)", "rough", "crawl", hf,
                        np.arange(-1.2, 1.01, step), 0.25, (0.4, -0.8, 2.4, 0.8),
                        foothold_rect=(0.6, -0.8, 2.4, 0.8))
    hf = HeightField()
    rng = np.random.default_rng(7)
    for _ in range(70):
        hf.box(rng.uniform(0.7, 3.2), rng.uniform(-1.3, 1.3), rng.uniform(0.06, 0.20),
               rng.uniform(0.06, 0.20), rng.uniform(0, math.pi), rng.uniform(0.02, 0.07))
    sc["rubble"] = Scene("rubble", "Rubble (blocks 2-7 cm)", "rough", "crawl", hf,
                         np.arange(-1.2, 1.51, step), -0.15, (0.4, -0.8, 2.4, 0.8),
                         foothold_rect=(0.7, -0.8, 2.4, 0.8))
    hf = HeightField()
    hf.stairs(1.0, 0.17, 0.30, 5)
    sc["stairs"] = Scene("stairs", "Stairs (5 x 0.17 m rise, 0.30 m tread)", "stairs", "crawl", hf,
                         np.arange(-1.2, 1.91, step), 0.1, (0.4, -0.8, 2.4, 0.8),
                         edges_x=[1.0 + 0.3 * k for k in range(5)])
    # descending the same flight (upper corridor -> dock on the lower floor, every mission):
    # the risers face away from the camera and the tread behind each edge is in an occlusion
    # shadow, so the drop only becomes visible ~0.8 m before the edge
    hf = HeightField()
    hf.stairs_down(1.0, 0.17, 0.30, 5)
    sc["stairs_down"] = Scene("stairs_down", "Stairs down (5 x 0.17 m)", "stairs", "crawl", hf,
                              np.arange(-1.2, 1.91, step), 0.25, (0.4, -0.8, 2.4, 0.8),
                              edges_x=[1.0 + 0.3 * k for k in range(5)])
    hf = HeightField()
    hf.ramp(1.0, 16.0, 2.5)
    sc["ramp"] = Scene("ramp", "Ramp 16 deg", "slope", "crawl", hf, np.arange(-1.2, 2.01, step), 0.15,
                       (0.4, -0.8, 2.4, 0.8))
    hf = HeightField()
    hf.box(1.4, 0.0, 0.4, 1.0, 0.0, 0.6)
    sc["obstacle"] = Scene("obstacle", "Obstacle 0.6 m (drum / machine)", "obstacle", "crawl", hf,
                           np.arange(-1.2, 0.41, step), 0.35, (0.4, -0.8, 1.6, 0.8))
    return sc


# ============================================================== simulation
def robot_pose_on(hf: HeightField, x: float, y: float = 0.0, yaw: float = 0.0):
    """Body pose standing on the terrain: height from the hip footprint, pitch
    from the front/rear ground heights (nose up = negative pitch)."""
    hf_front = float(np.median(hf(np.full(5, x + HIP_X_OFFSET), y + np.linspace(-0.13, 0.13, 5))))
    hf_rear = float(np.median(hf(np.full(5, x - HIP_X_OFFSET), y + np.linspace(-0.13, 0.13, 5))))
    z = 0.5 * (hf_front + hf_rear) + NOMINAL_BODY_HEIGHT
    pitch = -math.atan2(hf_front - hf_rear, 2 * HIP_X_OFFSET)
    return (x, y, z, 0.0, pitch, yaw)


def render_depth(hf: HeightField, pose, intr: DepthIntrinsics, far: float = 4.0,
                 step: float = 0.015) -> np.ndarray:
    """Ray-cast the height field; depth along the optical axis (NaN = none)."""
    T = camera_world_pose(pose, "depth_camera")
    a, b = intr.ray_grid()
    dirs = np.stack([np.ones_like(a), a, b], -1).reshape(-1, 3) @ T[:3, :3].T
    o = T[:3, 3]
    n = dirs.shape[0]
    depth = np.full(n, np.nan)
    active = np.arange(n)
    prev_t = 0.05
    for t in np.arange(0.08, far + step, step):
        if active.size == 0:
            break
        p = o + t * dirs[active]
        hit = p[:, 2] <= hf(p[:, 0], p[:, 1])
        if hit.any():
            idx = active[hit]
            lo = np.full(idx.size, prev_t)
            hi = np.full(idx.size, t)
            for _ in range(9):                       # bisection to ~0.03 mm
                mid = 0.5 * (lo + hi)
                pm = o + mid[:, None] * dirs[idx]
                below = pm[:, 2] <= hf(pm[:, 0], pm[:, 1])
                hi = np.where(below, mid, hi)
                lo = np.where(below, lo, mid)
            depth[idx] = hi
            active = active[~hit]
        prev_t = t
    depth[depth > far] = np.nan
    return depth.reshape(intr.height, intr.width)


def corrupt_depth(depth: np.ndarray, intr: DepthIntrinsics, rng: np.random.Generator,
                  sigma0=0.002, k=0.002, p_outlier=0.002, p_dropout=0.01) -> np.ndarray:
    """Range-dependent Gaussian noise + spurious near returns + drop-outs."""
    a, b = intr.ray_grid()
    rng_m = depth * np.sqrt(1.0 + a * a + b * b)          # range along the ray
    d = depth + rng.normal(0.0, 1.0, depth.shape) * (sigma0 + k * rng_m ** 2)
    valid = np.isfinite(d)
    out = valid & (rng.random(d.shape) < p_outlier)
    d[out] *= rng.uniform(0.3, 0.95, int(out.sum()))
    d[valid & (rng.random(d.shape) < p_dropout)] = np.nan
    return d


class FakeField:
    def __init__(self, name, offset, datatype=7, count=1):
        self.name, self.offset, self.datatype, self.count = name, offset, datatype, count


class FakeCloud:
    """Gazebo-like organised PointCloud2 (x y z rgb float32, point_step 16)."""

    def __init__(self, pts_org: np.ndarray, optical: bool = False):
        p = PinholeCamera.link_to_optical(pts_org) if optical else pts_org
        h, w = p.shape[:2]
        rec = np.zeros((h, w), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
        p = np.where(np.isfinite(p), p, np.inf)
        rec["x"], rec["y"], rec["z"] = p[..., 0], p[..., 1], p[..., 2]
        self.height, self.width = h, w
        self.fields = [FakeField("x", 0), FakeField("y", 4), FakeField("z", 8), FakeField("rgb", 12)]
        self.point_step, self.row_step, self.is_bigendian = 16, 16 * w, False
        self.data = rec.tobytes()


@dataclass
class Frame:
    t: float
    true_pose: tuple
    odom_pose: tuple
    cloud: FakeCloud
    pts_link: np.ndarray


def render_frames(scene: Scene, intr: DepthIntrinsics, rng: np.random.Generator) -> List[Frame]:
    frames = []
    for k, x in enumerate(scene.path_x):
        pose = robot_pose_on(scene.hf, float(x))
        d = corrupt_depth(render_depth(scene.hf, pose, intr), intr, rng)
        org = depth_image_to_points(d, intr, organised=True)
        noise = np.r_[rng.normal(0, 0.003, 3), np.radians(rng.normal(0, 0.2, 3))]
        odom = tuple(float(v) for v in np.asarray(pose) + noise)
        frames.append(Frame(0.1 * k, pose, odom, FakeCloud(org), org.reshape(-1, 3)))
    return frames


def run_pipeline(scene: Scene, frames: List[Frame], params: Optional[ElevationMapParams] = None,
                 classify: bool = True):
    emap = ElevationMap(params=params or ElevationMapParams())
    tc = TerrainClassifier()
    conv = ConventionDetector("auto")
    timeline, t_int, t_layers, t_cls = [], [], [], []
    decision = None
    decision_map = None
    for fr in frames:
        t0 = time.perf_counter()
        pts = pointcloud2_to_xyz(fr.cloud)
        c = conv.update(pts)
        pw, rr, origin = camera_points_to_world(pts, fr.odom_pose, c)
        emap.recenter(fr.odom_pose[0], fr.odom_pose[1])
        emap.integrate(pw, origin, t=fr.t, ranges=rr)
        t1 = time.perf_counter()
        emap.update_layers()
        t2 = time.perf_counter()
        if classify:
            info = tc.update(emap, fr.odom_pose, fr.t)
            t3 = time.perf_counter()
            timeline.append({"x": round(fr.true_pose[0], 3), "class": info["class"],
                             "gait": info["recommended_gait"], "raw_gait": info["raw_gait"]})
            t_cls.append((t3 - t2) * 1e3)
            if decision is None and fr.true_pose[0] >= scene.decision_x - 1e-6:
                decision = dict(info)
                decision_map = emap.copy()
        t_int.append((t1 - t0) * 1e3)
        t_layers.append((t2 - t1) * 1e3)
    return {"map": emap, "timeline": timeline, "decision": decision, "decision_map": decision_map,
            "convention": conv.current, "t_int": t_int, "t_layers": t_layers, "t_cls": t_cls}


# ================================================================ metrics
def accuracy(emap: ElevationMap, hf: HeightField, rect) -> dict:
    X, Y = emap.grid_xy()
    gt = hf(X, Y)
    e = emap.elevation
    x0, y0, x1, y1 = rect
    win = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    edge = (max_filter(gt, 3) - min_filter(gt, 3)) > 0.02
    known = win & np.isfinite(e)
    err = (e - gt)[known]
    ne = known & ~edge
    err_ne = (e - gt)[ne]
    out = {"coverage": float(known.sum() / max(1, win.sum())), "cells": int(known.sum())}
    if err.size:
        out.update(rmse_m=float(np.sqrt(np.mean(err ** 2))), mae_m=float(np.mean(np.abs(err))),
                   p95_abs_m=float(np.percentile(np.abs(err), 95)), bias_m=float(np.mean(err)),
                   max_abs_m=float(np.max(np.abs(err))))
    if err_ne.size:
        out.update(rmse_non_edge_m=float(np.sqrt(np.mean(err_ne ** 2))),
                   spurious_cells=int(np.sum(np.abs(err_ne) > 0.05)))
    return out


_DISK = np.array([(r * math.cos(a), r * math.sin(a)) for r in (0.0, FOOT_CHECK_RADIUS / 2, FOOT_CHECK_RADIUS)
                  for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])


def unsafe(hf: HeightField, x: float, y: float) -> bool:
    z = hf(x + _DISK[:, 0], y + _DISK[:, 1])
    return bool(z.max() - z.min() > UNSAFE_JUMP)


def foothold_eval(emap: ElevationMap, hf: HeightField, nominals: np.ndarray, edges_x=None) -> dict:
    fp = FootholdPlanner()
    rows = []
    t0 = time.perf_counter()
    for nx, ny in nominals:
        if not math.isfinite(emap.height_at(nx, ny)):
            continue                                   # evaluate where the map has data
        p, score, valid = fp.select(emap, (nx, ny), 0.0)
        rows.append((nx, ny, p[0], p[1], p[2], valid))
    dt = (time.perf_counter() - t0) * 1e3 / max(1, len(rows))
    if not rows:
        return {"n": 0}
    a = np.array(rows, dtype=float)
    un_nom = np.array([unsafe(hf, x, y) for x, y in a[:, :2]])
    un_sel = np.array([unsafe(hf, x, y) for x, y in a[:, 2:4]])
    disp = np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    zerr = np.abs(a[:, 4] - hf(a[:, 2], a[:, 3]))
    valid = a[:, 5] > 0.5
    out = {"n": int(len(a)), "valid_fraction": float(valid.mean()),
           "unsafe_nominal_fraction": float(un_nom.mean()),
           "unsafe_planned_fraction": float(un_sel[valid].mean()) if valid.any() else None,
           "relocated_when_unsafe_fraction": float(np.mean(disp[un_nom] > 1e-6)) if un_nom.any() else None,
           "mean_displacement_m": float(disp.mean()),
           "mean_displacement_safe_nominal_m": float(disp[~un_nom].mean()) if (~un_nom).any() else None,
           "max_displacement_m": float(disp.max()),
           "mean_abs_z_error_m": float(zerr[valid].mean()) if valid.any() else None,
           "select_ms": float(dt)}
    if edges_x:
        ed = lambda x: np.min(np.abs(np.asarray(x)[:, None] - np.asarray(edges_x)[None, :]), axis=1)
        dn, ds = ed(a[:, 0]), ed(a[:, 2])
        out["min_edge_distance_planned_m"] = float(ds[valid].min()) if valid.any() else None
        out["near_edge_nominals"] = int(np.sum(dn < 0.05))
        out["samples"] = {"nominal_x": a[:, 0].round(4).tolist(), "planned_x": a[:, 2].round(4).tolist(),
                          "nominal_y": a[:, 1].round(4).tolist(), "planned_z": a[:, 4].round(4).tolist(),
                          "valid": valid.tolist()}
    return out


def nominal_grid(rect, dx=0.02, dy=0.1):
    x0, y0, x1, y1 = rect
    xs, ys = np.meshgrid(np.arange(x0, x1, dx), np.arange(y0, y1 + 1e-9, dy))
    return np.stack([xs.ravel(), ys.ravel()], -1)


# ================================================================ figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })
    return plt


def _label(name: str) -> str:
    """Compact axis label for a scene key ('stairs_down' -> 'stairs\\ndown')."""
    return name.replace("_", "\n")


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(name, colors)
    cm.set_bad(UNKNOWN_FILL)
    return cm


def fig_maps(plt, scenes, results, path):
    names = list(scenes)
    fig, axes = plt.subplots(len(names), 4, figsize=(11.5, 2.15 * len(names)), constrained_layout=True)
    cm_h, cm_e, cm_c = _cmap(BLUE_RAMP, "h"), _cmap(ORANGE_RAMP, "e"), _cmap(INK_RAMP, "c")
    titles = ["Ground truth height [m]", "Fused elevation map [m]", "|error| [m]", "Traversability cost (0 easy..1 blocked)"]
    for i, n in enumerate(names):
        sc, res = scenes[n], results[n]
        m = res["map"]
        X, Y = m.grid_xy()
        x0, y0, x1, y1 = -0.4, -1.2, 2.6, 1.2
        sel_c = (X[0] >= x0) & (X[0] <= x1)
        sel_r = (Y[:, 0] >= y0) & (Y[:, 0] <= y1)
        ext = [X[0, sel_c][0], X[0, sel_c][-1], Y[sel_r, 0][0], Y[sel_r, 0][-1]]
        gt = sc.hf(X, Y)[np.ix_(sel_r, sel_c)]
        e = m.elevation[np.ix_(sel_r, sel_c)]
        err = np.abs(e - gt)
        cost = 1.0 - m.layer("traversability")[np.ix_(sel_r, sel_c)]
        vmin, vmax = float(np.min(gt)), max(float(np.max(gt)), 0.05)
        panels = [(gt, cm_h, vmin, vmax), (np.ma.masked_invalid(e), cm_h, vmin, vmax),
                  (np.ma.masked_invalid(err), cm_e, 0.0, 0.05), (np.ma.masked_invalid(cost), cm_c, 0.0, 1.0)]
        for j, (img, cm, lo, hi) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(img, origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(length=0, labelsize=7)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{sc.title}\ny [m]", fontsize=8, color=INK)
            if i == len(names) - 1:
                ax.set_xlabel("x [m]", fontsize=8)
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=6, length=0, colors=MUTED)
            cb.outline.set_visible(False)
            if j == 1:
                acc = res["accuracy"]
                ax.text(0.02, 0.04, f"RMSE {acc.get('rmse_non_edge_m', float('nan')) * 100:.1f} cm",
                        transform=ax.transAxes, fontsize=7, color=INK,
                        bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    fig.suptitle("Elevation mapping on synthetic scenes (final map after the approach; grey = unknown)",
                 fontsize=11, color=INK, fontweight="semibold")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_profiles(plt, scenes, results, path):
    names = ["stairs", "stairs_down", "ramp", "pipes", "obstacle"]
    fig, axes = plt.subplots(1, len(names), figsize=(14.5, 2.9), constrained_layout=True)
    for ax, n in zip(axes, names):
        sc, m = scenes[n], results[n]["map"]
        xs = np.arange(-0.5, 2.6, 0.01)
        y = 0.0 if n != "pipes" else 0.2
        ax.plot(xs, sc.hf(xs, np.full_like(xs, y)), color=INK2, lw=1.6, label="ground truth")
        ax.plot(xs, m.height_at(xs, np.full_like(xs, y)), color=SLOTS[0], lw=1.6, label="elevation map")
        ax.set_title(f"{sc.title}  (y = {y:.1f} m)", fontsize=9)
        ax.set_xlabel("x [m]")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("height [m]")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Height profiles along the walking direction", fontsize=11, color=INK, fontweight="semibold")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_classification(plt, scenes, results, path):
    from matplotlib.patches import Patch
    names = list(scenes)
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(names) + 1.4), constrained_layout=True)
    for i, n in enumerate(names):
        tl = results[n]["timeline"]
        xs = [r["x"] for r in tl]
        dx = (xs[1] - xs[0]) if len(xs) > 1 else 0.1
        yc = len(names) - 1 - i
        for r in tl:
            ax.add_patch(plt.Rectangle((r["x"] - dx / 2 + 0.002, yc + 0.08), dx - 0.004, 0.42,
                                       color=CLASS_COLORS[r["class"]], lw=0))
            ax.add_patch(plt.Rectangle((r["x"] - dx / 2 + 0.002, yc - 0.22), dx - 0.004, 0.22,
                                       color=GAIT_COLORS[r["gait"]], lw=0))
        # direct labels on contiguous class runs
        k = 0
        while k < len(tl):
            j = k
            while j + 1 < len(tl) and tl[j + 1]["class"] == tl[k]["class"]:
                j += 1
            if (j - k + 1) * dx >= 0.45:
                cls = tl[k]["class"]
                fg = "#ffffff" if cls in ("flat", "obstacle") or cls == "rough" else INK
                ax.text(0.5 * (tl[k]["x"] + tl[j]["x"]), yc + 0.29, cls, ha="center", va="center",
                        fontsize=7, color=fg if cls != "unknown" else INK)
            k = j + 1
        dec = scenes[n].decision_x
        ax.plot([dec, dec], [yc - 0.27, yc + 0.55], color=INK, lw=1.0)
        d = results[n]["decision"] or {}
        ok = d.get("class") == scenes[n].expected_class
        ax.text(2.08, yc + 0.12, f"{d.get('class', '?')} / {d.get('recommended_gait', '?')}"
                                 f"  ({'correct' if ok else 'expected ' + scenes[n].expected_class})",
                fontsize=8, color=INK, va="center")
    ax.set_yticks([len(names) - 1 - i + 0.15 for i in range(len(names))])
    ax.set_yticklabels([scenes[n].title for n in names], fontsize=8, color=INK)
    ax.set_xlim(-1.35, 3.3)
    ax.set_ylim(-0.5, len(names) - 0.3)
    ax.set_xlabel("robot x [m]  (upper strip: terrain class, lower strip: recommended gait; "
                  "vertical tick = decision pose)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [Patch(color=CLASS_COLORS[c], label=c) for c in ("flat", "rough", "stairs", "slope", "obstacle", "unknown")]
    handles += [Patch(color=GAIT_COLORS["trot"], label="gait: trot"), Patch(color=GAIT_COLORS["crawl"], label="gait: crawl")]
    ax.legend(handles=handles, ncol=8, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=8)
    ax.set_title("Terrain class and gait recommendation along each approach (10 Hz, with hysteresis)",
                 fontsize=10, pad=24)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_footholds(plt, scenes, results, foot, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.6, 1.0, 1.25]})
    # (a) stairs side view: nominal -> planned footholds near the risers
    ax = axes[0]
    sc, m = scenes["stairs"], results["stairs"]["map"]
    xs = np.arange(0.6, 2.2, 0.005)
    ax.plot(xs, sc.hf(xs, np.zeros_like(xs)), color=INK2, lw=1.6, label="ground truth")
    s = foot["stairs"].get("samples", {})
    nx, px, pz = np.array(s.get("nominal_x", [])), np.array(s.get("planned_x", [])), np.array(s.get("planned_z", []))
    ny = np.array(s.get("nominal_y", []))
    sel = (ny > 0) & (np.arange(nx.size) % 3 == 0)
    nz = sc.hf(nx, np.zeros_like(nx))
    for a, b, za, zb in zip(nx[sel], px[sel], nz[sel], pz[sel]):
        if abs(a - b) > 1e-6:
            ax.annotate("", xy=(b, zb + 0.012), xytext=(a, za + 0.012),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.scatter(nx[sel], nz[sel] + 0.012, s=22, facecolors="none", edgecolors=SLOTS[1], linewidths=1.2,
               label="nominal foothold", zorder=3)
    ax.scatter(px[sel], pz[sel] + 0.012, s=22, color=SLOTS[0], edgecolors=SURFACE, linewidths=1.0,
               label="planned foothold", zorder=4)
    ax.set_xlim(0.6, 2.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("height [m]")
    ax.set_title("Stairs: footholds are pushed off the risers", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    # (b) nominal vs planned edge distance
    ax = axes[1]
    edges = np.asarray(sc.edges_x)
    dn = np.min(np.abs(nx[:, None] - edges[None, :]), axis=1) if nx.size else np.array([])
    dp = np.min(np.abs(px[:, None] - edges[None, :]), axis=1) if px.size else np.array([])
    ax.plot([0, 0.16], [0, 0.16], color=AXIS, lw=1.0)
    ax.axhline(FOOT_CHECK_RADIUS, color=INK2, lw=0.8)
    ax.text(0.158, FOOT_CHECK_RADIUS + 0.003, "5 cm safety radius", fontsize=7, color=INK2, ha="right")
    ax.scatter(dn, dp, s=14, color=SLOTS[0], edgecolors=SURFACE, linewidths=0.6)
    ax.set_xlim(0, 0.16)
    ax.set_ylim(0, 0.16)
    ax.set_xlabel("nominal distance to riser [m]")
    ax.set_ylabel("planned distance to riser [m]")
    ax.set_title("Edge clearance (stairs)", fontsize=9)
    ax.grid(True)
    ax.set_axisbelow(True)
    # (c) unsafe placements nominal vs planned
    ax = axes[2]
    names = [n for n in ("stairs", "stairs_down", "pipes", "rubble") if foot.get(n, {}).get("n")]
    xi = np.arange(len(names))
    w = 0.32
    nom = [100 * foot[n]["unsafe_nominal_fraction"] for n in names]
    pla = [100 * (foot[n]["unsafe_planned_fraction"] or 0.0) for n in names]
    b1 = ax.bar(xi - w / 2 - 0.01, nom, w, color=SLOTS[1], label="nominal (Raibert only)")
    b2 = ax.bar(xi + w / 2 + 0.01, pla, w, color=SLOTS[0], label="planned")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.6, f"{r.get_height():.1f}",
                    ha="center", fontsize=7, color=INK)
    ax.set_xticks(xi)
    ax.set_xticklabels([_label(n) for n in names])
    ax.set_ylabel("unsafe footholds [%]")
    ax.set_title("Placements spanning a > 3 cm jump", fontsize=9)
    ax.set_ylim(0, max(nom + pla + [1.0]) * 1.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    fig.suptitle("Foothold planning on the fused maps", fontsize=11, color=INK, fontweight="semibold")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_accuracy(plt, scenes, results, path):
    names = list(scenes)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0), constrained_layout=True)
    xi = np.arange(len(names))
    w = 0.34
    ax = axes[0]
    allv = [100 * results[n]["accuracy"].get("rmse_m", np.nan) for n in names]
    ne = [100 * results[n]["accuracy"].get("rmse_non_edge_m", np.nan) for n in names]
    b1 = ax.bar(xi - w / 2 - 0.01, ne, w, color=SLOTS[0], label="away from edges")
    b2 = ax.bar(xi + w / 2 + 0.01, allv, w, color=SLOTS[1], label="all known cells")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.08, f"{r.get_height():.1f}",
                    ha="center", fontsize=7, color=INK)
    if "obstacle" in names:
        k = names.index("obstacle")
        top = float(np.nanmax(allv + ne))
        ax.annotate("top of the 0.6 m face is\nnever visible (lower bound)", xy=(k - 0.2, ne[k] * 0.55),
                    xytext=(k - 0.6, top * 0.97), ha="right", va="top", fontsize=7, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_ylim(0, top * 1.08)
    ax.set_xticks(xi)
    ax.set_xticklabels([_label(n) for n in names])
    ax.set_ylabel("elevation RMSE [cm]")
    ax.set_title("Elevation error vs ground truth", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax = axes[1]
    cov = [100 * results[n]["accuracy"]["coverage"] for n in names]
    bars = ax.bar(xi, cov, 0.45, color=SLOTS[0])
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{r.get_height():.0f}",
                ha="center", fontsize=7, color=INK)
    ax.set_xticks(xi)
    ax.set_xticklabels([_label(n) for n in names])
    ax.set_ylim(0, 110)
    ax.set_ylabel("known cells in feature area [%]")
    ax.set_title("Map coverage (occlusion limits stairs / obstacle)", fontsize=9)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =================================================================== main
def _stats(v):
    """Timing summary.  The development PC runs other jobs, which inflates
    mean / p95 / max; p10 approximates the cost on an idle core, the median
    the typical cost under load."""
    v = np.asarray(v, dtype=float)
    if not v.size:
        return {}
    return {"p10": float(np.percentile(v, 10)), "median": float(np.median(v)), "mean": float(v.mean()),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true", help="coarser walk (0.2 m between frames)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    intr = DepthIntrinsics.from_sensor("depth_camera")
    scenes = build_scenes(args.quick)
    results, frames_by_scene = {}, {}
    t_start = time.time()
    for n, sc in scenes.items():
        t0 = time.time()
        frames = render_frames(sc, intr, rng)
        frames_by_scene[n] = frames
        r = run_pipeline(sc, frames)
        r["accuracy"] = accuracy(r["map"], sc.hf, sc.eval_rect)
        r["accuracy_decision"] = accuracy(r["decision_map"], sc.hf, sc.eval_rect) if r["decision_map"] else {}
        results[n] = r
        d = r["decision"] or {}
        print(f"{n:9s} frames {len(frames):3d}  RMSE {r['accuracy'].get('rmse_m', float('nan')) * 100:5.2f} cm "
              f"(non-edge {r['accuracy'].get('rmse_non_edge_m', float('nan')) * 100:5.2f})  coverage "
              f"{r['accuracy']['coverage']:.2f}  class {d.get('class')} / {d.get('recommended_gait')} "
              f"[{time.time() - t0:.1f} s]")

    # ---------------------------------------------------------- footholds
    foot = {}
    st = scenes["stairs"]
    noms = np.array([(x, y) for x in np.arange(0.5, 1.9, 0.01) for y in (-0.134, 0.134)])
    foot["stairs"] = foothold_eval(results["stairs"]["map"], st.hf, noms, st.edges_x)
    sd = scenes["stairs_down"]
    noms_down = np.array([(x, y) for x in np.arange(0.5, 2.4, 0.01) for y in (-0.134, 0.134)])
    foot["stairs_down"] = foothold_eval(results["stairs_down"]["map"], sd.hf, noms_down, sd.edges_x)
    for n in ("pipes", "rubble"):
        foot[n] = foothold_eval(results[n]["map"], scenes[n].hf, nominal_grid(scenes[n].foothold_rect))
    for n, f in foot.items():
        print(f"footholds {n:7s} n {f['n']:4d}  unsafe nominal {100 * f['unsafe_nominal_fraction']:5.1f} %  "
              f"planned {100 * (f['unsafe_planned_fraction'] or 0):5.1f} %  valid {100 * f['valid_fraction']:5.1f} %")

    # ------------------------------------------------- convention detection
    conv = {"link_correct": 0, "optical_correct": 0, "frames": 0}
    for frames in frames_by_scene.values():
        for fr in frames:
            pts = fr.pts_link[np.isfinite(fr.pts_link).all(axis=1)]
            conv["frames"] += 1
            conv["link_correct"] += detect_convention(pts)[0] == "link"
            conv["optical_correct"] += detect_convention(PinholeCamera.link_to_optical(pts))[0] == "optical"
    # end-to-end with an optical cloud through the full pipeline
    opt_frames = [Frame(f.t, f.true_pose, f.odom_pose, FakeCloud(f.pts_link.reshape(intr.height, intr.width, 3),
                                                                 optical=True), f.pts_link)
                  for f in frames_by_scene["stairs"]]
    r_opt = run_pipeline(scenes["stairs"], opt_frames)
    conv["optical_pipeline_detected"] = r_opt["convention"]
    conv["optical_pipeline_rmse_non_edge_m"] = accuracy(r_opt["map"], scenes["stairs"].hf,
                                                        scenes["stairs"].eval_rect).get("rmse_non_edge_m")
    print("convention detection:", conv)

    # ------------------------------------------------------------ ablations
    abl = {}
    for n in ("flat", "rubble"):
        for hc in (1, 2):
            r = run_pipeline(scenes[n], frames_by_scene[n], ElevationMapParams(higher_outlier_count=hc), classify=False)
            a = accuracy(r["map"], scenes[n].hf, scenes[n].eval_rect)
            abl[f"{n}_higher_outlier_count_{hc}"] = {"rmse_non_edge_m": a.get("rmse_non_edge_m"),
                                                     "spurious_cells_gt5cm": a.get("spurious_cells"),
                                                     "max_abs_m": a.get("max_abs_m")}
    for bias in (True, False):
        r = run_pipeline(scenes["stairs"], frames_by_scene["stairs"], ElevationMapParams(max_height_bias=bias),
                         classify=False)
        a = accuracy(r["map"], scenes["stairs"].hf, scenes["stairs"].eval_rect)
        f = foothold_eval(r["map"], scenes["stairs"].hf, noms, scenes["stairs"].edges_x)
        abl[f"stairs_max_height_bias_{bias}"] = {"rmse_m": a.get("rmse_m"), "rmse_non_edge_m": a.get("rmse_non_edge_m"),
                                                 "bias_m": a.get("bias_m"),
                                                 "unsafe_planned_fraction": f.get("unsafe_planned_fraction")}
    # occlusion-shadow fill (holes behind an edge filled from the far, lower side) vs plain median
    for n in ("stairs_down", "stairs", "pipes"):
        for step_m in (0.0, ElevationMapParams().shadow_fill_step):
            r = run_pipeline(scenes[n], frames_by_scene[n], ElevationMapParams(shadow_fill_step=step_m),
                             classify=False)
            a = accuracy(r["map"], scenes[n].hf, scenes[n].eval_rect)
            if n == "stairs_down":
                f = foothold_eval(r["map"], scenes[n].hf, noms_down, scenes[n].edges_x)
            elif n == "stairs":
                f = foothold_eval(r["map"], scenes[n].hf, noms, scenes[n].edges_x)
            else:
                f = foothold_eval(r["map"], scenes[n].hf, nominal_grid(scenes[n].foothold_rect))
            abl[f"{n}_shadow_fill_{'on' if step_m > 0 else 'off'}"] = {
                "rmse_non_edge_m": a.get("rmse_non_edge_m"), "spurious_cells_gt5cm": a.get("spurious_cells"),
                "unsafe_planned_fraction": f.get("unsafe_planned_fraction"),
                "min_edge_distance_planned_m": f.get("min_edge_distance_planned_m")}
    print("ablations:", json.dumps(abl, indent=None))

    # --------------------------------------------------------------- output
    t_int = sum((r["t_int"] for r in results.values()), [])
    t_lay = sum((r["t_layers"] for r in results.values()), [])
    t_cls = sum((r["t_cls"] for r in results.values()), [])
    xyz = results["stairs"]["map"].to_cloud_xyz()
    t_rebuild = []
    for _ in range(30):          # controller side: /elevation_map/points -> map + all layers
        t0 = time.perf_counter()
        ElevationMap.from_cloud_xyz(xyz)          # computes the layers eagerly
        t_rebuild.append((time.perf_counter() - t0) * 1e3)
    n_correct = sum((results[n]["decision"] or {}).get("class") == scenes[n].expected_class for n in scenes)
    n_gait = sum((results[n]["decision"] or {}).get("recommended_gait") == scenes[n].expected_gait for n in scenes)
    summary = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "tool": "tools/eval_terrain.py" + (" --quick" if args.quick else ""),
        "seed": args.seed,
        "camera": {k: SENSORS["depth_camera"][k] for k in ("width", "height", "hfov", "far", "pose")},
        "noise_model": {"depth_sigma": "0.002 + 0.002 * range^2 [m]", "spurious_near_returns": 0.002,
                        "dropouts": 0.01, "odometry_noise": "pos 3 mm, attitude 0.2 deg (1 sigma)"},
        "map": {"size_m": [6.0, 6.0], "resolution_m": 0.04},
        "foothold_unsafe_definition": f"height range > {UNSAFE_JUMP} m within {FOOT_CHECK_RADIUS} m of the foot centre",
        "classification": {"correct": int(n_correct), "total": len(scenes),
                           "gait_correct": int(n_gait)},
        "scenes": {},
        "footholds": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in foot.items()},
        "convention_detection": conv,
        "ablations": abl,
        "timing_ms": {"note": "single-threaded numpy on the Windows development PC, which was shared with "
                              "other jobs: p10 ~ idle-core cost, median ~ typical cost under load",
                      "integrate_25k_point_cloud": _stats(t_int), "layers": _stats(t_lay),
                      "classify": _stats(t_cls), "rebuild_from_cloud_plus_layers": _stats(t_rebuild),
                      "foothold_select": float(np.mean([f["select_ms"] for f in foot.values() if f.get("n")]))},
    }
    for n, sc in scenes.items():
        r = results[n]
        d = r["decision"] or {}
        counts = {}
        for row in r["timeline"]:
            counts[row["class"]] = counts.get(row["class"], 0) + 1
        summary["scenes"][n] = {
            "title": sc.title, "frames": len(r["timeline"]),
            "expected_class": sc.expected_class, "expected_gait": sc.expected_gait,
            "decision_x": sc.decision_x,
            "class_at_decision": d.get("class"), "gait_at_decision": d.get("recommended_gait"),
            "correct": d.get("class") == sc.expected_class,
            "terrain_info_at_decision": {k: d.get(k) for k in ("roughness", "max_step", "slope_deg", "ahead_clear",
                                                               "known_fraction", "edges")},
            "class_counts_along_approach": counts,
            "accuracy_final_map": r["accuracy"], "accuracy_at_decision": r["accuracy_decision"],
        }
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "terrain_eval.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"classification {n_correct}/{len(scenes)} correct, gait {n_gait}/{len(scenes)}; "
          f"wrote docs/results/terrain_eval.json [{time.time() - t_start:.0f} s]")

    if not args.no_figures:
        plt = _style()
        os.makedirs(IMG_DIR, exist_ok=True)
        fig_maps(plt, scenes, results, os.path.join(IMG_DIR, "terrain_maps.png"))
        fig_profiles(plt, scenes, results, os.path.join(IMG_DIR, "terrain_profiles.png"))
        fig_classification(plt, scenes, results, os.path.join(IMG_DIR, "terrain_classification.png"))
        fig_footholds(plt, scenes, results, foot, os.path.join(IMG_DIR, "terrain_footholds.png"))
        fig_accuracy(plt, scenes, results, os.path.join(IMG_DIR, "terrain_accuracy.png"))
        print("figures: docs/images/terrain_{maps,profiles,classification,footholds,accuracy}.png")
    return summary


if __name__ == "__main__":
    main()
