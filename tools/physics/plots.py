"""
Figures of the physics check (docs/images/physics_*.png).

Input: the per-run dicts of tools/physics_check.py that carry ``series``
(50 Hz decimated logs), ``snapshots`` (10 Hz body / leg positions from MuJoCo),
``terrain`` and ``gt`` (ground-truth height raster).

* physics_timelines.png     body height above the terrain + roll / pitch per scenario
* physics_trajectories.png  top view of the base path over the scenario terrain
* physics_torque.png        peak |tau| / effort per joint type and scenario
* physics_sticks.png        side-view stick figures: stand-up, trot, obstacle course
* physics_recovery.png      front-view stick figures of the self-righting sequences
* physics_summary.png       pass / fail per scenario (before / after the fixes)
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dusan_common import robot_model as rm  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-3, status, ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL, WARNING = "#0ca30c", "#d03b3b", "#fab219"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
TERRAIN = "#a8a7a2"

TITLES = {
    "a_stand": "(a) stand-up + stand", "b_trot": "(b) trot 0.4 m/s", "c_crawl": "(c) crawl 0.2 m/s",
    "d_turn": "(d) turn 0.5 rad/s", "e_obstacles": "(e) pipe + step", "f_push": "(f) 15 N s push, trot",
    "g_payload": "(g) +3 kg payload", "h_side": "(h1) start on the side", "h_push": "(h2) 60 N s push",
    "h_back": "(h3) start on the back", "h_push100": "(h4) 100 N s push", "i_plate_trot": "(i) mu 0.15, trot",
    "i_plate_crawl": "(i) mu 0.15, crawl", "j_stairs": "(j) 0.17 m stairs", "j_ramp": "(j) 16 deg ramp",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)


def _by_key(results) -> Dict[str, dict]:
    out = {}
    for r in sorted(results, key=lambda r: r["seed"]):
        if "series" in r and r["key"] not in out:
            out[r["key"]] = r
    return out


# ---------------------------------------------------------------------------
def timelines(runs: Dict[str, dict], path: str) -> Optional[str]:
    keys = [k for k in ("a_stand", "b_trot", "c_crawl", "d_turn", "f_push", "g_payload", "e_obstacles",
                        "h_side") if k in runs]
    if not keys:
        return None
    fig, axes = plt.subplots(len(keys), 2, figsize=(10, 1.55 * len(keys) + 0.6), sharex=False, squeeze=False)
    for i, k in enumerate(keys):
        s = runs[k]["series"]
        t = np.asarray(s["t"])
        ax = axes[i, 0]
        _style(ax)
        ax.plot(t, s["body_h"], color=BLUE, lw=1.4)
        ax.axhline(rm.NOMINAL_BODY_HEIGHT, color=INK2, lw=0.8, ls="--")
        ax.axhline(0.12, color=CRITICAL, lw=0.8, ls=":")
        ax.set_ylim(0.0, 0.6)
        ax.set_ylabel("height [m]", fontsize=7, color=INK2)
        ax.set_title(f"{TITLES.get(k, k)}  (seed {runs[k]['seed']}, {'PASS' if runs[k].get('ok') else 'FAIL'})",
                     fontsize=8, loc="left", color=INK)
        ax = axes[i, 1]
        _style(ax)
        rpy = np.degrees(np.asarray(s["rpy"]))
        ax.plot(t, rpy[:, 0], color=BLUE, lw=1.2, label="roll")
        ax.plot(t, rpy[:, 1], color=ORANGE, lw=1.2, label="pitch")
        ax.axhline(55, color=CRITICAL, lw=0.8, ls=":")
        ax.axhline(-55, color=CRITICAL, lw=0.8, ls=":")
        lim = max(10.0, min(185.0, float(np.nanmax(np.abs(rpy[:, :2]))) * 1.15 if len(rpy) else 10.0))
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("deg", fontsize=7, color=INK2)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", frameon=False)
    for j in range(2):
        axes[-1, j].set_xlabel("simulated time [s]", fontsize=7, color=INK2)
    fig.suptitle("Body height above the terrain (dashed 0.34 m, dotted fall 0.12 m) | roll / pitch (dotted +-55 deg)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def _terrain_top(ax, terrain):
    for p in getattr(terrain, "prims", []):
        if p.kind == "box":
            cx, cy = p.pos[0], p.pos[1]
            sx, sy = p.size[0] * abs(math.cos(p.rpy[1])), p.size[1]
            color = "#c9d6e8" if p.mu < 0.5 else "#e3e2dd"
            ax.add_patch(plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fc=color, ec=TERRAIN, lw=0.6))
        elif p.kind == "capsule":
            r, length = p.size
            ax.add_patch(plt.Rectangle((p.pos[0] - r, p.pos[1] - length / 2), 2 * r, length, fc="#d6d5cf",
                                       ec=TERRAIN, lw=0.6))


def trajectories(runs: Dict[str, dict], path: str) -> Optional[str]:
    keys = [k for k in ("b_trot", "c_crawl", "d_turn", "f_push", "g_payload", "e_obstacles", "i_plate_trot",
                        "i_plate_crawl", "j_stairs", "j_ramp") if k in runs]
    if not keys:
        return None
    n = len(keys)
    cols = min(5, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, k in zip(axes.ravel(), keys):
        _style(ax)
        r = runs[k]
        _terrain_top(ax, r.get("terrain"))
        pos = np.asarray(r["series"]["pos"])
        ax.plot(pos[:, 0], pos[:, 1], color=BLUE, lw=1.4)
        ax.plot(pos[0, 0], pos[0, 1], "o", color=INK2, ms=4)
        ax.plot(pos[-1, 0], pos[-1, 1], "s", color=ORANGE if r.get("ok") else CRITICAL, ms=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{TITLES.get(k, k)}: {'PASS' if r.get('ok') else 'FAIL'}", fontsize=8, loc="left", color=INK)
        ax.set_xlabel("x [m]", fontsize=7, color=INK2)
        ax.set_ylabel("y [m]", fontsize=7, color=INK2)
    fig.suptitle("Base path (top view; circle start, square end; blue-grey = mu 0.15 plate, grey = steps / pipe)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def torque(results: List[dict], path: str) -> Optional[str]:
    rows = []
    by: Dict[str, List[dict]] = {}
    for r in results:
        if r.get("metrics"):
            by.setdefault(r["key"], []).append(r)
    for k in TITLES:
        rs = by.get(k)
        if not rs:
            continue
        vals = []
        for name in ("max_ratio_hip_x", "max_ratio_hip_y", "max_ratio_knee"):
            v = [r["metrics"].get(name) for r in rs]
            v = [float(x) for x in v if isinstance(x, (int, float)) and x is not None and math.isfinite(x)]
            vals.append(max(v) if v else float("nan"))
        sat = [r["metrics"].get("saturation_frac") for r in rs]
        sat = [float(x) for x in sat if isinstance(x, (int, float)) and x is not None]
        rows.append((k, vals, max(sat) if sat else float("nan")))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _style(ax)
    x = np.arange(len(rows))
    w = 0.26
    for j, (lab, col) in enumerate((("hip_x (45 N m)", BLUE), ("hip_y (45 N m)", ORANGE), ("knee (60 N m)", AQUA))):
        ax.bar(x + (j - 1) * (w + 0.01), [r[1][j] for r in rows], w, color=col, label=lab, edgecolor=SURFACE, lw=1)
    for xi, r in zip(x, rows):
        if math.isfinite(r[2]):
            ax.text(xi, 1.04, f"{100 * r[2]:.1f}%", ha="center", fontsize=6.5, color=INK2)
    ax.axhline(1.0, color=CRITICAL, lw=0.8, ls=":")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([TITLES.get(r[0], r[0]) for r in rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("peak |tau| / effort", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_title("Peak joint torque vs the URDF effort (all seeds; numbers above: share of samples at the limit)",
                 fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
_BOX = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(_BOX[i] != _BOX[j]) == 1]


def _trunk(sn):
    half = np.array([0.30, 0.11, 0.07])
    pts = (_BOX * half + np.array([-0.004, 0.0, 0.002])) @ np.asarray(sn["R"]).T + np.asarray(sn["base"])
    return pts


def _stick(ax, sn, axes=(0, 2), dx=0.0, color=INK, alpha=1.0):
    a, b = axes
    tr = _trunk(sn)
    for i, j in _EDGES:
        ax.plot([tr[i, a] + dx, tr[j, a] + dx], [tr[i, b], tr[j, b]], color=color, lw=0.8, alpha=alpha)
    for leg, col in zip(("fl", "fr", "rl", "rr"), (BLUE, ORANGE, AQUA, "#4a3aa7")):
        p = np.asarray(sn[leg])
        ax.plot(p[:, a] + dx, p[:, b], "-", color=col, lw=1.6, alpha=alpha)
        ax.plot(p[-1, a] + dx, p[-1, b], "o", color=col, ms=2.5, alpha=alpha)


def _profile(ax, r, y=0.0):
    gt = r.get("gt")
    if not gt:
        ax.axhline(0.0, color=TERRAIN, lw=1.5)
        return
    H = np.asarray(gt["H"])
    row = int(round((y - gt["y0"]) / gt["res"]))
    row = min(max(row, 0), H.shape[0] - 1)
    xs = gt["x0"] + gt["res"] * np.arange(H.shape[1])
    ax.fill_between(xs, -0.05, np.nan_to_num(H[row]), color="#e3e2dd", lw=0)
    ax.plot(xs, np.nan_to_num(H[row]), color=TERRAIN, lw=1.0)


def sticks(runs: Dict[str, dict], path: str) -> Optional[str]:
    panels = []
    if "a_stand" in runs:
        panels.append(("a_stand", [0.4, 1.2, 1.9, 2.5, 3.2], 0.75, "stand-up from straight legs (frames spread in x)"))
    if "b_trot" in runs:
        panels.append(("b_trot", [8.0, 8.1, 8.2, 8.3, 8.4], 0.75, "trot 0.4 m/s, 0.1 s apart (frames spread in x)"))
    if "e_obstacles" in runs:
        sn = runs["e_obstacles"]["snapshots"]
        t_end = float(sn[-1]["t"])
        xs = np.array([s["base"][0] for s in sn])
        t_far = float(sn[int(np.argmax(xs))]["t"])
        ts = list(np.linspace(4.5, max(4.6, min(t_end, t_far)), 5))
        panels.append(("e_obstacles", ts, 0.0, "pipe (x 1.5) and step (x 3.2), true positions"))
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.5 * len(panels)), squeeze=False)
    for ax, (k, times, spread, title) in zip(axes[:, 0], panels):
        _style(ax)
        r = runs[k]
        sn = r["snapshots"]
        st = np.array([s["t"] for s in sn])
        frames = [sn[int(np.argmin(np.abs(st - tt)))] for tt in times]
        x0 = frames[0]["base"][0]
        xs = []
        for i, s in enumerate(frames):
            dx = (i * spread - (s["base"][0] - x0)) if spread else 0.0
            _stick(ax, s, (0, 2), dx=dx, alpha=0.45 + 0.55 * (i + 1) / len(frames))
            bx = s["base"][0] + dx
            xs.append(bx)
            ax.text(bx, 0.63, f"t {s['t']:.1f}\n{s['mode'][:11]}", fontsize=6, ha="center", color=INK2)
        lo, hi = min(xs) - 0.6, max(xs) + 0.6
        if k == "e_obstacles":
            lo, hi = min(lo, 1.0), max(hi, 4.6)
        if spread:
            ax.axhline(0.0, color=TERRAIN, lw=1.5)
        else:
            _profile(ax, r)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.05, 0.75)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{TITLES.get(k, k)}: {title}", fontsize=8, loc="left", color=INK)
        ax.set_xlabel("x [m]", fontsize=7, color=INK2)
    fig.suptitle("Stick figures from the MuJoCo state (trunk box + legs FL blue, FR orange, RL aqua, RR violet)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def recovery(runs: Dict[str, dict], path: str) -> Optional[str]:
    keys = [k for k in ("h_side", "h_push", "h_push100", "h_back") if k in runs]
    if not keys:
        return None
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.6 * len(keys)), squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        _style(ax)
        r = runs[k]
        sn = r["snapshots"]
        # one frame at every recovery-state change (+ the end), at most 9
        frames, last = [], None
        for s in sn:
            key = s["rec_state"] if s["rec_state"] != "NORMAL" else s["mode"]
            if key != last:
                frames.append(s)
                last = key
        frames.append(sn[-1])
        if len(frames) > 9:
            idx = np.linspace(0, len(frames) - 1, 9).round().astype(int)
            frames = [frames[i] for i in idx]
        for i, s in enumerate(frames):
            dx = i * 0.75 - s["base"][1]
            _stick(ax, s, (1, 2), dx=dx)
            roll = math.degrees(math.atan2(s["R"][2][1], s["R"][2][2]))
            lab = s["rec_state"] if s["rec_state"] != "NORMAL" else s["mode"]
            ax.text(i * 0.75, 0.66, f"t {s['t']:.1f}\n{lab[:11]}\nroll {roll:.0f}", fontsize=6, ha="center",
                    color=INK2)
        ax.axhline(0.0, color=TERRAIN, lw=1.5)
        ax.set_xlim(-0.5, 0.75 * (len(frames) - 1) + 0.5)
        ax.set_ylim(-0.05, 0.8)
        ax.set_aspect("equal", adjustable="box")
        m = r.get("metrics", {})
        ax.set_title(f"{TITLES.get(k, k)}: {'PASS' if r.get('ok') else 'FAIL'} - attempts {m.get('recovery_attempts')}, "
                     f"assisted resets {m.get('assisted_resets')} (front view, frames at state changes)",
                     fontsize=8, loc="left", color=INK)
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def summary(rows: List[dict], before: Optional[List[dict]], path: str) -> Optional[str]:
    if not rows:
        return None
    bmap = {r["key"]: r for r in (before or [])}
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(rows) + 1.0))
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    cols = ["before fixes", "after fixes"] if before else ["after fixes"]
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(0.0, y, f"{r['title']}{'' if r['required'] else '  (best effort)'}", fontsize=7.5, va="center",
                color=INK)
        cells = ([bmap.get(r["key"])] if before else []) + [r]
        for j, c in enumerate(cells):
            x = 0.62 + 0.19 * j
            if c is None:
                ax.text(x, y, "-", fontsize=7.5, ha="center", va="center", color=INK2)
                continue
            ok = c["passed"] == c["runs"]
            col = GOOD if ok else (CRITICAL if c["passed"] == 0 else WARNING)
            ax.add_patch(plt.Rectangle((x - 0.075, y - 0.38), 0.15, 0.76, fc=col, alpha=0.18, ec=col, lw=0.8))
            mark = "PASS" if ok else ("FAIL" if c["passed"] == 0 else "PART")
            ax.text(x, y, f"{mark} {c['passed']}/{c['runs']}", fontsize=7, ha="center", va="center", color=INK)
    for j, cname in enumerate(cols):
        ax.text(0.62 + 0.19 * j, len(rows) - 0.2, cname, fontsize=8, ha="center", color=INK, weight="bold")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(-0.8, len(rows) + 0.4)
    ax.set_title("MuJoCo physics check: seeds passing per scenario", fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def make_all(results: List[dict], out_dir: str, rows: Optional[List[dict]] = None,
             before: Optional[List[dict]] = None, all_results: Optional[List[dict]] = None) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    runs = _by_key(results)
    out = [
        timelines(runs, os.path.join(out_dir, "physics_timelines.png")),
        trajectories(runs, os.path.join(out_dir, "physics_trajectories.png")),
        torque(all_results if all_results is not None else results, os.path.join(out_dir, "physics_torque.png")),
        sticks(runs, os.path.join(out_dir, "physics_sticks.png")),
        recovery(runs, os.path.join(out_dir, "physics_recovery.png")),
        summary(rows or [], before, os.path.join(out_dir, "physics_summary.png")),
    ]
    return [p for p in out if p]
