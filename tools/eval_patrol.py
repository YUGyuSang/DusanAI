#!/usr/bin/env python3
"""
Offline evaluation of the viewpoint-based patrol planner (Module 2, no ROS).

On the real navigation map (``simulation/maps/plant_map.yaml``; falls back to
a map rasterised from ``world_layout.yaml`` when the generated map is not
present) it computes

* the candidate viewpoints of every gauge, their readability scores and the
  reasons why candidates were rejected, and the selected viewpoint;
* the optimal visiting order (exact Held-Karp on A* path lengths) for the
  closed tour (dock -> gauges -> dock) and the open route (dock -> gauges),
  compared with the naive layout order, greedy nearest neighbour and the mean
  of random orders (same distance matrix);
* the effect of the terrain cost layer (dusan_common.terrain_cost, default
  on): the same plan without it, route lengths and the debris (pipes, rubble,
  plates, platform) every leg drives over with and without the layer;

and writes docs/results/patrol_eval.json, docs/images/patrol_route.png and
docs/images/patrol_candidates.png.

    python tools/eval_patrol.py [--map path/to/map.yaml] [--seed 0]
"""
import argparse
import collections
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("dusan_common", "module2_inspection"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import numpy as np  # noqa: E402

from dusan_common.grid_map import OccupancyGridMap  # noqa: E402
from dusan_common.layout import Layout  # noqa: E402
from dusan_common import terrain_cost as tc  # noqa: E402
from module2_inspection.viewpoint_planner import (ViewpointParams, ViewpointPlanner, layout_occupancy_map,  # noqa: E402
                                                  tour_cost)

IMG_DIR = os.path.join(ROOT, "docs", "images")
RES_DIR = os.path.join(ROOT, "docs", "results")


def load_grid(layout, path=None):
    path = path or layout.nav_map_yaml()
    if path and os.path.exists(path):
        return OccupancyGridMap.load(path), os.path.relpath(os.path.abspath(path), ROOT).replace("\\", "/")
    return layout_occupancy_map(layout), "rasterised from world_layout.yaml (plant_map.yaml not found)"


def greedy_nn(D, ret):
    order, left = [0], set(range(1, len(D)))
    while left:
        k = min(left, key=lambda j: D[order[-1], j])
        order.append(k)
        left.remove(k)
    return order, tour_cost(D, order, ret)


def random_orders(D, ret, rng, n=2000):
    costs = []
    for _ in range(n):
        perm = list(rng.permutation(np.arange(1, len(D))))
        costs.append(tour_cost(D, [0] + perm, ret))
    return float(np.mean(costs)), float(np.min(costs))


def summarise(plan, label, D, ret, rng):
    idx_naive = [0] + [plan.node_ids.index(g) for g in plan.naive_order]
    nn_order, nn_cost = greedy_nn(D, ret)
    rnd_mean, rnd_best = random_orders(D, ret, rng)
    return {
        "route": label,
        "order": ["dock"] + plan.order + (["dock"] if ret else []),
        "optimal_length_m": round(plan.total_length, 2),
        "naive_layout_order": ["dock"] + plan.naive_order + (["dock"] if ret else []),
        "naive_length_m": round(tour_cost(D, idx_naive, ret), 2),
        "greedy_nearest_neighbour_m": round(nn_cost, 2),
        "greedy_nearest_neighbour_order": ["dock"] + [plan.node_ids[i] for i in nn_order[1:]],
        "random_order_mean_m": round(rnd_mean, 2),
        "random_order_best_of_2000_m": round(rnd_best, 2),
        "saving_vs_naive_pct": round(100.0 * (1.0 - plan.total_length / tour_cost(D, idx_naive, ret)), 2),
        "saving_vs_random_mean_pct": round(100.0 * (1.0 - plan.total_length / rnd_mean), 2),
        "leg_lengths_m": [round(v, 2) for v in plan.leg_lengths],
        "solver": plan.method,
    }


def run(args):
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    layout = Layout.load()
    grid, map_source = load_grid(layout, args.map)
    print(f"map: {map_source}  ({grid.width}x{grid.height} @ {grid.resolution} m)")
    params = ViewpointParams()
    planner = ViewpointPlanner(layout, grid, params)
    t0 = time.time()
    closed = planner.plan(return_to_dock=True)
    t_closed = time.time() - t0
    t0 = time.time()
    open_ = planner.plan(return_to_dock=False)       # A* legs are cached -> fast
    t_open = time.time() - t0
    D = closed.distance_matrix
    # the same planner without the terrain cost layer (route length impact, debris crossed)
    t0 = time.time()
    base = ViewpointPlanner(layout, grid, ViewpointParams(terrain_cost_enabled=False)).plan(return_to_dock=True)
    t_base = time.time() - t0

    gauges = {}
    for g in layout.gauges:
        cands = closed.candidates.get(g["id"], [])
        valid = [c for c in cands if c.valid]
        reasons = collections.Counter()
        for c in cands:
            for r in c.reason.split(","):
                reasons[r] += 1
        best = next((v for v in closed.viewpoints if v.gauge_id == g["id"]), None)
        gauges[g["id"]] = {
            "tag": g.get("tag"), "position": g["position"], "normal": g.get("normal"),
            "floor_z": layout.floor_z(*g["position"][:2]),
            "n_candidates": len(cands), "n_valid": len(valid),
            "valid_score_stats": None if not valid else {
                "min": round(min(c.score for c in valid), 4), "mean": round(float(np.mean([c.score for c in valid])), 4),
                "max": round(max(c.score for c in valid), 4)},
            "rejection_reasons": dict(reasons),
            "best": None if best is None else {**best.to_json(), "ring_m": round(best.ring, 2),
                                               "angle_from_normal_deg": round(best.angle_deg, 1)},
        }
    result = {
        "description": "Viewpoint-based patrol planning (readability-scored viewpoints + exact Held-Karp on A* lengths)",
        "map_source": map_source,
        "map_size_cells": [grid.width, grid.height], "resolution_m": grid.resolution,
        "start_dock_pose": list(layout.dock_pose),
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.__dict__.items()},
        "gauges": gauges,
        "unreachable": closed.unreachable,
        "closed_tour": summarise(closed, "dock -> gauges -> dock", D, True, rng),
        "open_route": summarise(open_, "dock -> gauges", D, False, rng),
        "distance_matrix_m": {"nodes": ["dock"] + closed.node_ids[1:], "values": np.round(D, 2).tolist()},
        "terrain_cost": terrain_cost_summary(layout, params, closed, base),
        "compute_time_s": {"closed_plan_incl_astar": round(t_closed, 3), "open_plan_cached_astar": round(t_open, 3),
                           "closed_plan_without_terrain_cost": round(t_base, 3),
                           "n_astar_pairs": int(len(D) * (len(D) - 1) / 2)},
    }
    with open(os.path.join(RES_DIR, "patrol_eval.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
    make_figures(layout, grid, closed, open_, result)
    c, o = result["closed_tour"], result["open_route"]
    print("closed:", " -> ".join(c["order"]), f"{c['optimal_length_m']} m  (naive {c['naive_length_m']} m, "
          f"NN {c['greedy_nearest_neighbour_m']} m, random {c['random_order_mean_m']} m)")
    print("open  :", " -> ".join(o["order"]), f"{o['optimal_length_m']} m  (naive {o['naive_length_m']} m, "
          f"saving {o['saving_vs_naive_pct']} %)")
    tcs = result["terrain_cost"]
    print(f"terrain cost: closed {tcs['closed_tour_m']} m vs {tcs['closed_tour_without_cost_m']} m without "
          f"(+{tcs['length_increase_pct']} %); debris crossed {tcs['debris_crossed']} vs "
          f"{tcs['debris_crossed_without_cost']}")
    for gid, g in gauges.items():
        b = g["best"]
        if b:
            print(f"  {gid}: ({b['x']:.2f}, {b['y']:.2f}) yaw {math.degrees(b['yaw']):6.1f}  score {b['score']:.3f}  "
                  f"d {b['distance']:.2f} m  {b['pixel_diameter']:.0f} px  inc {b['incidence_deg']:.1f} deg  "
                  f"valid {g['n_valid']}/{g['n_candidates']}")
    return result


def terrain_cost_summary(layout, params, plan, base):
    """Route length and debris crossed (robot centre on the footprint) with / without the layer."""
    def legs(pl):
        names = ["dock"] + pl.order + (["dock"] if pl.return_to_dock else [])
        return [{"leg": f"{names[k]}->{names[k + 1]}", "length_m": round(pl.leg_lengths[k], 2),
                 "crosses": tc.debris_crossed(leg, layout)} for k, leg in enumerate(pl.legs)]

    def crossed(lg):
        return sorted({d for row in lg for d in row["crosses"]})

    with_cost, without = legs(plan), legs(base)
    prm = tc.TerrainCostParams.from_params(vars(params))
    return {
        "description": "dusan_common.terrain_cost: crossable debris as extra A* cell cost (inside w_<type>, "
                       f"margin {prm.margin_m} m at margin_fraction x w); legs cross debris only without a free "
                       "detour (C-OBS4 plate and C-OBS5 pipe span the corridor)",
        "params": {"enabled": prm.enabled, "w_pipe": prm.w_pipe, "w_rubble": prm.w_rubble, "w_plate": prm.w_plate,
                   "w_platform": prm.w_platform, "margin_m": prm.margin_m, "margin_fraction": prm.margin_fraction,
                   "inside_pad_m": tc.INSIDE_PAD},
        "closed_tour_m": round(plan.total_length, 2),
        "closed_tour_without_cost_m": round(base.total_length, 2),
        "length_increase_m": round(plan.total_length - base.total_length, 2),
        "length_increase_pct": round(100.0 * (plan.total_length / base.total_length - 1.0), 2),
        "order": ["dock"] + plan.order + ["dock"],
        "order_without_cost": ["dock"] + base.order + ["dock"],
        "debris_crossed": crossed(with_cost),
        "debris_crossed_without_cost": crossed(without),
        "legs": with_cost,
        "legs_without_cost": without,
    }


# =============================================================================
# Figures
# =============================================================================
def _map_image(grid):
    img = np.full(grid.occ.shape, 0.82)
    img[grid.occ == 0] = 1.0
    img[grid.occ >= 50] = 0.25
    extent = [grid.origin_x, grid.origin_x + grid.width * grid.resolution,
              grid.origin_y, grid.origin_y + grid.height * grid.resolution]
    return img, extent


def _draw_debris(ax, layout):
    """Crossable debris of the terrain cost layer (pipes, rubble, plates, platform)."""
    from matplotlib.patches import Polygon, Rectangle
    col = {"pipe": "#8d6e63", "rubble": "#a1887f", "plate": "#90a4ae", "platform": "#f9a825"}
    for ob in layout.obstacles:
        t = ob.get("type")
        if t == "pipe":
            cx, cy = ob["center"]
            yaw, h, r = float(ob.get("yaw", 0.0)), 0.5 * float(ob["length"]), max(0.5 * float(ob["diameter"]), 0.05)
            ux, uy, nx, ny = math.cos(yaw), math.sin(yaw), -math.sin(yaw), math.cos(yaw)
            pts = [(cx - h * ux - r * nx, cy - h * uy - r * ny), (cx + h * ux - r * nx, cy + h * uy - r * ny),
                   (cx + h * ux + r * nx, cy + h * uy + r * ny), (cx - h * ux + r * nx, cy - h * uy + r * ny)]
            ax.add_patch(Polygon(pts, closed=True, fc=col[t], ec="none", alpha=0.9, zorder=2))
        elif t in ("rubble", "plate", "platform"):
            x0, y0, x1, y1 = ob["rect"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=col[t], ec="none", alpha=0.55, zorder=2,
                                   hatch="xx" if t == "rubble" else None))


def _draw_route(ax, layout, grid, plan, show_labels=True, lw=2.0, arrow_len=0.9):
    import matplotlib.pyplot as plt
    _draw_debris(ax, layout)
    cmap = plt.get_cmap("viridis")
    n = max(1, len(plan.legs))
    for i, leg in enumerate(plan.legs):
        p = np.asarray(leg)
        ax.plot(p[:, 0], p[:, 1], "-", color=cmap(i / n), lw=lw, zorder=3)
        if len(p) >= 2:
            k = len(p) // 2
            ax.annotate("", xy=p[min(k + 1, len(p) - 1)], xytext=p[k], annotation_clip=True,
                        arrowprops=dict(arrowstyle="->", color=cmap(i / n), lw=lw), zorder=4)
    for i, vp in enumerate(plan.viewpoints):
        g = layout.gauge(vp.gauge_id)
        cx, cy = vp.x + 0.30 * math.cos(vp.yaw), vp.y + 0.30 * math.sin(vp.yaw)
        ax.plot([cx, g["position"][0]], [cy, g["position"][1]], ":", color="#d1495b", lw=1.0, zorder=4)
        ax.quiver(vp.x, vp.y, arrow_len * math.cos(vp.yaw), arrow_len * math.sin(vp.yaw),
                  color=plt.get_cmap("RdYlGn")(vp.score), angles="xy", scale_units="xy", scale=1.0,
                  width=0.005, zorder=5, edgecolor="k", linewidth=0.4)
        if show_labels:
            ax.text(vp.x, vp.y + 0.45, f"{i + 1}", fontsize=8, ha="center", va="bottom", zorder=6, clip_on=True,
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="k", lw=0.5))
    for g in layout.gauges:
        x, y = g["position"][:2]
        n = np.asarray(g["normal"], float)
        ax.plot(x, y, "s", color="#f6ae2d", mec="k", ms=6, zorder=6)
        ax.plot([x, x + 0.5 * n[0]], [y, y + 0.5 * n[1]], "-", color="#f6ae2d", lw=1.5, zorder=6)
        if show_labels:
            ax.text(x + 0.25, y - 0.55, g["id"], fontsize=7, color="#7a4f00", zorder=6, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    dx, dy, _ = layout.dock_pose
    ax.plot(dx, dy, "*", color="#2e86ab", mec="k", ms=14, zorder=7)


def make_figures(layout, grid, closed, open_, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, extent = _map_image(grid)
    fig = plt.figure(figsize=(15, 11.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.95, 0.95], width_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=extent, interpolation="nearest")
    _draw_route(ax0, layout, grid, closed, show_labels=False, lw=1.6, arrow_len=1.6)
    c = result["closed_tour"]
    o = result["open_route"]
    ax0.set_title(f"Patrol route on the nav map ({result['map_source']}; brown = pipes / rubble, grey = plates: "
                  f"terrain cost): optimal closed tour {c['optimal_length_m']:.1f} m vs layout order "
                  f"{c['naive_length_m']:.1f} m  |  open route {o['optimal_length_m']:.1f} m "
                  f"vs {result['open_route']['naive_length_m']:.1f} m", fontsize=9)
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_aspect("equal")
    panels = [(gs[1:, 0], (-10.6, 15.5, -9.6, 9.6), "Turbine hall"),
              (gs[1, 1], (14.6, 40.0, -2.3, 2.3), "Corridor, upper level"),
              (gs[2, 1], (39.0, 64.2, -2.3, 2.3), "Corridor, descent + lower level (dock = star)")]
    for spec, (x0, x1, y0, y1), title in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=extent, interpolation="nearest")
        _draw_route(ax, layout, grid, closed, show_labels=True, arrow_len=0.9)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(title + ": numbers = visiting order, arrows = viewpoint yaw (colour = score)", fontsize=9)
        ax.set_xlabel("x (m)")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.06, hspace=0.18, wspace=0.12)
    fig.savefig(os.path.join(IMG_DIR, "patrol_route.png"), dpi=130)
    plt.close(fig)

    # ---- candidate score fields per gauge
    gl = layout.gauges
    cols = 4
    rows = int(math.ceil(len(gl) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    axes = np.atleast_2d(axes)
    sc = None
    for i, g in enumerate(gl):
        ax = axes[i // cols, i % cols]
        x, y = g["position"][:2]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=extent, interpolation="nearest")
        cands = closed.candidates.get(g["id"], [])
        val = [c for c in cands if c.valid]
        inv = [c for c in cands if not c.valid]
        if inv:
            ax.plot([c.x for c in inv], [c.y for c in inv], "x", color="#999999", ms=3, mew=0.8)
        if val:
            sc = ax.scatter([c.x for c in val], [c.y for c in val], c=[c.score for c in val], cmap="RdYlGn",
                            vmin=0.4, vmax=1.0, s=12, edgecolors="none")
        best = next((v for v in closed.viewpoints if v.gauge_id == g["id"]), None)
        if best is not None:
            ax.quiver(best.x, best.y, 0.8 * math.cos(best.yaw), 0.8 * math.sin(best.yaw), color="#2e86ab",
                      angles="xy", scale_units="xy", scale=1.0, width=0.012, edgecolor="k", linewidth=0.6)
            ax.plot(best.x, best.y, "o", mfc="none", mec="#2e86ab", ms=10, mew=2)
        n = np.asarray(g["normal"], float)
        ax.plot(x, y, "s", color="#f6ae2d", mec="k", ms=8)
        ax.plot([x, x + 0.6 * n[0]], [y, y + 0.6 * n[1]], "-", color="#f6ae2d", lw=2)
        ax.set_xlim(x - 3.0, x + 3.0)
        ax.set_ylim(y - 3.0, y + 3.0)
        ax.set_aspect("equal")
        sb = "" if best is None else f"  best {best.score:.2f} @ {best.ring:.1f} m / {best.angle_deg:+.0f} deg"
        ax.set_title(f"{g['id']} ({g.get('tag', '')})\n{sb.strip()}", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(gl), rows * cols):
        axes[j // cols, j % cols].axis("off")
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="readability score (valid candidates)")
    fig.suptitle("Candidate viewpoints per gauge (grey x = rejected: occupied / unreachable / no line of sight / "
                 "dial outside image / incidence)", fontsize=10)
    fig.savefig(os.path.join(IMG_DIR, "patrol_candidates.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=None, help="nav map YAML (default: layout nav map, fallback rasterised)")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
