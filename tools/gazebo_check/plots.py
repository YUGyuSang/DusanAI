"""Figures of the Gazebo check (docs/images/gazebo_*.png), matplotlib (dev only).

Locomotion mode comparison: ``stable`` (categorical slot 1, blue) vs ``mpc``
(slot 2, orange), the command in muted ink (dashed).

* gazebo_timelines.png - per scenario (seed 0): base height, roll, pitch and
  forward speed (heading frame) over time, fall times marked;
* gazebo_sticks.png    - side-view stick figures (base + legs from FK of the
  measured joints at the ground-truth pose) of trot / teleop / ramp runs;
* gazebo_summary.png   - passed runs per scenario and condition (flat, timing
  stress, terrain with / without the elevation map);
* gazebo_corridor.png  - the real plant world: top views of the lower corridor /
  ramp with the (j) (over the 0.10 m pipe) and (o) (around it) trajectories,
  passed runs per plant scenario and condition (repository code with the depth
  camera map / ground-truth map / no map / MPC, and the proposed fixes).
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

from dusan_common import robot_model as rm

C_STABLE = "#2a78d6"    # categorical slot 1
C_MPC = "#eb6834"       # categorical slot 2
C_INK = "#52514e"       # text secondary (axes, labels, command)
C_GRID = "#e4e3df"
C_SURFACE = "#fcfcfb"
MODES = (("mpc", C_MPC), ("stable", C_STABLE))
SHORT = {"a_stand": "(a) stand", "b_trot": "(b) trot 0.35", "c_crawl": "(c) crawl 0.2",
         "d_turn": "(d) turn 0.5", "e_teleop": "(e) teleop 60 s", "i_push": "(i) push 10 N s",
         "f_pipe": "(f) pipe 0.08 m", "g_ramp": "(g) ramp up 11.8°", "h_ramp_down": "(h) ramp down"}
TIMELINE_ROWS = (("", "b_trot"), ("", "d_turn"), ("", "e_teleop"), ("", "i_push"), ("_map", "g_ramp"))


def _load(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    except Exception:
        return None


def _run(data: Dict, label: str, scen: str, seed: str = "0"):
    r = data.get("runs", {}).get(label, {}).get("scenarios", {}).get(scen, {}).get(seed)
    if not r:
        return None, None
    return r, _load(r.get("npz"))


def _vx_heading(A) -> np.ndarray:
    yaw = A["rpy"][:, 2]
    v = A["vel"]
    return np.cos(yaw) * v[:, 0] + np.sin(yaw) * v[:, 1]


def _style(ax):
    ax.set_facecolor(C_SURFACE)
    ax.grid(True, color=C_GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b2")
    ax.tick_params(colors=C_INK, labelsize=7)
    try:
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    except (AttributeError, ValueError):
        pass


def timelines(data: Dict, out: str) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = data.get("runs", {})
    rows = [(suf, sc) for suf, sc in TIMELINE_ROWS if any(f"{m}{suf}" in runs for m, _c in MODES)]
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 2.1 * len(rows)), sharex="row", squeeze=False)
    fig.patch.set_facecolor(C_SURFACE)
    sigs = (("height [m]", lambda A: A["pos"][:, 2]), ("roll [deg]", lambda A: np.degrees(A["rpy"][:, 0])),
            ("pitch [deg]", lambda A: np.degrees(A["rpy"][:, 1])), ("v forward [m/s]", _vx_heading))
    for i, (suf, sc) in enumerate(rows):
        for j, (name, fn) in enumerate(sigs):
            ax = axes[i, j]
            _style(ax)
            last = None
            for mode, col in MODES:
                r, A = _run(data, f"{mode}{suf}", sc)
                if A is None or len(A["t"]) == 0:
                    continue
                y = fn(A)
                if j in (1, 2):
                    y = np.clip(y, -60, 60)
                fell = r.get("first_fall_t")
                ax.plot(A["t"], y, color=col, lw=1.0, label=mode + (f" (fell {fell:.1f} s)" if fell else ""))
                if fell:
                    ax.axvline(fell, color=col, lw=0.8, ls="--")
                last = A
            if j == 3 and last is not None:
                ax.plot(last["t"], last["cmd"][:, 0], color=C_INK, lw=0.9, ls="--", label="command vx")
            if i == 0:
                ax.set_title(name, fontsize=9, color=C_INK)
            if j == 0:
                ax.set_ylabel(SHORT[sc] + (" (map)" if suf == "_map" else ""), fontsize=8, color=C_INK)
            if j == 3:
                ax.legend(fontsize=6, frameon=False, loc="lower right", labelcolor=C_INK)
        if sc not in ("g_ramp", "h_ramp_down"):
            axes[i, 0].set_ylim(0.0, 0.6)
    for ax in axes[-1]:
        ax.set_xlabel("sim time [s]", fontsize=8, color=C_INK)
    fig.suptitle("Gazebo (gz-sim 8 + DART), seed 0: stable mode (blue) vs MPC mode (orange)", fontsize=10,
                 color="#0b0b0b")
    fig.tight_layout()
    path = os.path.join(out, "gazebo_timelines.png")
    fig.savefig(path, dpi=110, facecolor=C_SURFACE)
    plt.close(fig)
    return path


def _legs_side(A, i):
    """(base polyline xz, [leg polylines xz]) at log index i (side view, heading frame)."""
    from module1_locomotion import kinematics as kin
    from dusan_common.geometry import euler_to_rot
    pos, rpy, q = A["pos"][i], A["rpy"][i], A["q"][i]
    R = euler_to_rot(*rpy)
    yaw = rpy[2]
    c, s = math.cos(yaw), math.sin(yaw)

    def h(p):  # world -> heading frame x, z (origin at the base)
        d = p - pos
        return np.array([c * d[0] + s * d[1], p[2]])
    hx = rm.HIP_X_OFFSET
    base = [h(pos + R @ np.array([x, 0.0, 0.0])) for x in (hx + 0.08, -hx - 0.08)]
    legs = []
    for leg in range(4):
        q0, q1 = float(q[3 * leg]), float(q[3 * leg + 1])
        hip = np.array(rm.hip_x_position(leg))
        a = rm.LEG_SIGN_Y[leg] * rm.HIP_TO_THIGH_Y
        c0, s0 = math.cos(q0), math.sin(q0)
        zk = -rm.UPPER_LEG_LENGTH * math.cos(q1)
        hip_y = hip + np.array([0.0, a * c0, a * s0])
        knee = hip + np.array([-rm.UPPER_LEG_LENGTH * math.sin(q1), a * c0 - zk * s0, a * s0 + zk * c0])
        foot = kin.leg_fk(q[3 * leg:3 * leg + 3], leg)
        legs.append([h(pos + R @ np.asarray(p)) for p in (hip_y, knee, foot)])
    return base, legs


STICK_ROWS = (("stable", "b_trot", 8.0), ("stable", "e_teleop", 20.0), ("stable_map", "g_ramp", 9.0),
              ("mpc", "e_teleop", 20.0))


def sticks(data: Dict, out: str) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(lab, sc, t0) for lab, sc, t0 in STICK_ROWS if _run(data, lab, sc)[1] is not None]
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.7 * len(rows)))
    fig.patch.set_facecolor(C_SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (lab, sc, t0) in zip(axes, rows):
        r, A = _run(data, lab, sc)
        t = A["t"]
        if r.get("first_fall_t"):
            t0 = max(4.0, r["first_fall_t"] - 2.0)
        idx = [int(np.searchsorted(t, tt)) for tt in np.arange(t0, min(t0 + 2.4, t[-1]), 0.12)]
        col = C_MPC if lab.startswith("mpc") else C_STABLE
        zmin = 1e9
        for k, i in enumerate(idx):
            if i >= len(t):
                break
            base, legs = _legs_side(A, i)
            off = k * 0.75
            b = np.array(base)
            ax.plot(b[:, 0] + off, b[:, 1], color=col, lw=2.2)
            for leg, pts in enumerate(legs):
                p = np.array(pts)
                ax.plot(p[:, 0] + off, p[:, 1], color=col if leg in (0, 1) else C_INK, lw=1.0)
                zmin = min(zmin, float(p[:, 1].min()))
            ax.text(off, float(b[:, 1].max()) + 0.08, f"{t[i]:.2f}", fontsize=6, ha="center", color=C_INK)
        zmin = 0.0 if zmin > 1e8 else zmin
        ax.axhline(zmin - 0.025, color="#b9b8b2", lw=0.8)
        ax.set_xlim(-0.6, 0.75 * len(idx))
        ax.set_ylim(zmin - 0.06, zmin + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ttl = f"{lab.replace('_map', ' + map')}: {SHORT[sc]}" + \
            (f" - fell at {r['first_fall_t']:.2f} s" if r.get("first_fall_t") else "")
        ax.set_title(ttl, fontsize=8, color=C_INK, loc="left")
    fig.tight_layout()
    path = os.path.join(out, "gazebo_sticks.png")
    fig.savefig(path, dpi=110, facecolor=C_SURFACE)
    plt.close(fig)
    return path


def _bars(ax, groups, runs):
    """groups: [(xlabel, {mode: label})] -> grouped bars of passed / runs per mode."""
    w = 0.36
    seen = set()
    ticks, names = [], []
    for x, (name, labels) in enumerate(groups):
        sc = labels["scenario"]
        for k, (mode, col) in enumerate(MODES):
            xi = x + (k - 0.5) * (w + 0.04)
            s = runs.get(labels[mode], {}).get("summary", {}).get(sc)
            if not s:
                if labels.get(f"{mode}_note"):
                    ax.text(xi, 0.1, labels[f"{mode}_note"], ha="center", fontsize=6.5, color=C_INK, rotation=90)
                continue
            ax.bar(xi, max(s["passed"], 0.04), w, color=col, label=mode if mode not in seen else None)
            seen.add(mode)
            ax.text(xi, s["passed"] + 0.06, f"{s['passed']}/{s['runs']}", ha="center", fontsize=6.5, color=C_INK)
        ticks.append(x)
        names.append(name)
    ax.set_xticks(ticks)
    ax.set_xticklabels(names, fontsize=6.5, color=C_INK)
    ax.set_ylim(0, 3.7)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("passed runs (seeds)", fontsize=8, color=C_INK)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), labelcolor=C_INK)


def summary(data: Dict, out: str) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = data.get("runs", {})
    if not any(m in runs for m, _c in MODES):
        return None

    def g(name, sc, suffix=""):
        return (name, {"scenario": sc, "mpc": f"mpc{suffix}", "stable": f"stable{suffix}"})
    flat = [g(SHORT[sc], sc) for sc in ("a_stand", "b_trot", "c_crawl", "d_turn", "e_teleop", "i_push")]
    stress = []
    for suf, title in (("_lat0", "latency 0"), ("_lat2", "latency 20 ms"), ("_rtf05", "RTF 0.5"),
                       ("_drop30", "30 % dropped")):
        for sc in ("b_trot", "d_turn", "e_teleop"):
            stress.append(g(SHORT[sc].split(" ")[0] + " " + SHORT[sc].split(" ")[1] + "\n" + title, sc, suf))
    if "mpc_mpc100" in runs:
        grp = g("(e) teleop\nMPC +100 ms", "e_teleop", "_mpc100")
        grp[1]["stable_note"] = "no MPC"
        stress.append(grp)
    terrain = [g(SHORT[sc] + "\n" + ("map" if suf == "_map" else "blind"), sc, suf)
               for suf in ("_map", "_blind") for sc in ("f_pipe", "g_ramp", "h_ramp_down")]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.4), gridspec_kw={"hspace": 0.75})
    fig.patch.set_facecolor(C_SURFACE)
    for ax, groups, title in ((axes[0], flat, "flat floor, latency 1 control period (10 ms), MPC delay = solve time"),
                              (axes[1], stress, "timing stress (slow / loaded machine)"),
                              (axes[2], terrain, "terrain: with the ground-truth elevation map (stand-in for the "
                                                 "depth camera -> terrain_mapper) / blind")):
        _style(ax)
        _bars(ax, groups, runs)
        ax.set_title(title, fontsize=9, color="#0b0b0b", loc="left")
    fig.suptitle("Gazebo (gz-sim 8 + DART): runs without fall that meet the criteria, stable vs MPC",
                 fontsize=10, color="#0b0b0b")
    path = os.path.join(out, "gazebo_summary.png")
    fig.savefig(path, dpi=110, facecolor=C_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# plant world (simulation/worlds/plant.sdf) runs: label -> (legend, colour)
PLANT_LABELS = (("plant_depth", "stable (repo), depth camera map", C_STABLE),
                ("plant_truth", "stable (repo), ground-truth map", "#1f9d74"),
                ("plant_blind", "stable (repo), no map", "#8c8b86"),
                ("plant_mpc_depth", "mpc (repo), depth camera map", C_MPC),
                ("proposal_walk_depth", "stable + pitch fix + walk 1.8 s, depth map", "#c79a1e"),
                ("proposal_depth", "stable + pitch fix + trot only, depth map", "#7b4fd6"))
PLANT_SCEN = (("k_late_start", "(k) node 15 s late"), ("n_pipe_dock", "(n) 0.10 m pipe,\nstraight"),
              ("j_corridor", "(j) over the pipe\n-> ramp top"), ("o_corridor_detour", "(o) around the pipe\n-> ramp top"),
              ("m_corridor_fast", "(m) = (j)\nat 0.5 m/s"), ("p_rubble", "(p) rubble field,\nstraight"),
              ("r_rubble_detour", "(r) rubble, along\nthe north wall"), ("q_pipe08", "(q) 0.08 m pipe,\nstraight"),
              ("s_plate", "(s) steel plate\nmu 0.2"), ("l_corridor_full", "(l) whole corridor\n(around the pipe)"))


def _plant_topview(ax, hf, data, labels, scen, route, title):
    from . import world as W  # noqa: F401
    x0, x1, y0, y1 = 43.0, 63.0, -2.05, 2.05
    c0, c1 = int((x0 - hf.x0) / hf.res), int((x1 - hf.x0) / hf.res)
    r0, r1 = int((y0 - hf.y0) / hf.res), int((y1 - hf.y0) / hf.res)
    Z = np.clip(hf.Z[r0:r1, c0:c1], -0.9, 0.4)
    ax.imshow(Z, extent=(x0, x1, y0, y1), origin="lower", cmap="Greys", vmin=-1.3, vmax=0.6, aspect="equal",
              interpolation="nearest")
    for (x, y) in route:
        if x >= x0:
            ax.plot(x, y, "o", ms=4, mfc="none", mec=C_INK, mew=0.8)
    for txt, xy, col in (("dock", (62.3, -0.75), C_INK), ("pipe 0.10 m", (58.5, -1.62), C_INK),
                         ("pallet", (55.0, 1.0), C_INK), ("ramp 11.8° up", (49.3, 1.45), "#ffffff"),
                         ("stairs", (48.0, -1.35), "#ffffff"), ("upper level", (44.8, -1.35), "#ffffff")):
        ax.text(xy[0], xy[1], txt, fontsize=7, color=col, ha="center", va="center")
    for lab, name, col in labels:
        first = True
        for seed in ("0", "1", "2"):
            r, A = _run(data, lab, scen, seed)
            if A is None or len(A["t"]) == 0:
                continue
            p = A["pos"]
            fell = r.get("first_fall_t")
            n = int(np.searchsorted(A["t"], fell)) if fell else len(p)
            sel = p[:n, 0] >= x0
            ax.plot(p[:n, 0][sel], p[:n, 1][sel], color=col, lw=1.1, alpha=0.9, label=name if first else None)
            if fell:
                ax.plot(p[n - 1, 0], p[n - 1, 1], "x", color=col, ms=8, mew=2)
            first = False
    ax.set_xlim(x1, x0)                              # the robot walks west: right to left
    ax.set_ylim(y0, y1)
    ax.set_ylabel("y [m]", fontsize=8, color=C_INK)
    ax.tick_params(colors=C_INK, labelsize=7)
    ax.set_title(title, fontsize=9, color="#0b0b0b", loc="left")


def corridor(data: Dict, out: str) -> Optional[str]:
    """gazebo_corridor.png: the plant runs - top views of the lower corridor /
    ramp with the (j) (over the pipe) and (o) (around it) trajectories of
    seeds 0-2 (x = fall), and the passed runs per plant scenario and condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import scenarios as S
    from . import world as W
    runs = data.get("runs", {})
    labels = [(lab, name, col) for lab, name, col in PLANT_LABELS if lab in runs]
    if not labels:
        return None
    hf = W.HeightField.plant(res=0.03)
    fig = plt.figure(figsize=(13, 10.4))
    fig.patch.set_facecolor(C_SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.05], hspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    _plant_topview(ax1, hf, data, labels, "j_corridor", S.ROUTE_RAMP,
                   "(j) keyboard operator from the dock OVER the 0.10 m pipe to the ramp top: seeds 0-2, x = fall")
    ax2 = fig.add_subplot(gs[1])
    _plant_topview(ax2, hf, data, labels, "o_corridor_detour", S.ROUTE_DETOUR,
                   "(o) the same AROUND the pipe's north end (0.63 m gap at the wall)")
    ax2.set_xlabel("world x [m]  (the robot walks west: right -> left)", fontsize=8, color=C_INK)
    handles, names = [], []
    for ax in (ax1, ax2):
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h)
                names.append(n)
    fig.legend(handles, names, fontsize=7, frameon=False, loc="upper center", ncol=3, labelcolor=C_INK,
               bbox_to_anchor=(0.5, 0.955))
    bar = fig.add_subplot(gs[2])
    _style(bar)
    w = 0.86 / max(len(labels), 1)
    for k, (lab, name, col) in enumerate(labels):
        summ = runs[lab].get("summary", {})
        for i, (sc, _t) in enumerate(PLANT_SCEN):
            s = summ.get(sc)
            if not s:
                continue
            xi = i + (k - (len(labels) - 1) / 2.0) * w
            frac = s["passed"] / max(s["runs"], 1)
            bar.bar(xi, max(frac, 0.015), w * 0.92, color=col)
            bar.text(xi, frac + 0.03, f"{s['passed']}/{s['runs']}", ha="center", va="bottom", fontsize=5.5,
                     color=C_INK, rotation=90)
    bar.set_xticks(range(len(PLANT_SCEN)))
    bar.set_xticklabels([t for _s, t in PLANT_SCEN], fontsize=7, color=C_INK)
    bar.set_ylim(0, 1.3)
    bar.set_ylabel("passed / runs", fontsize=8, color=C_INK)
    bar.set_title("plant world scenarios: runs without fall that reach every waypoint (repo = code as committed; "
                  "pitch fix / walk / trot only = proposed, applied in the harness)", fontsize=8.5, color="#0b0b0b",
                  loc="left")
    fig.suptitle("Gazebo (gz-sim 8 + DART), the real simulation/worlds/plant.sdf", fontsize=10, color="#0b0b0b",
                 y=0.985)
    path = os.path.join(out, "gazebo_corridor.png")
    fig.savefig(path, dpi=110, facecolor=C_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all(data: Dict, out: str) -> List[str]:
    os.makedirs(out, exist_ok=True)
    paths = []
    for fn in (timelines, sticks, summary, corridor):
        try:
            p = fn(data, out)
        except Exception as e:  # a figure must never fail the check
            print(f"figure {fn.__name__} failed: {type(e).__name__}: {e}")
            p = None
        if p:
            paths.append(p)
    return paths
