#!/usr/bin/env python3
"""
Turn a metrics session (written by plant_simulation/metrics_logger) into
report figures.

    python tools/plot_metrics.py                       # newest ~/dusan_metrics/<session>
    python tools/plot_metrics.py ~/dusan_metrics/20261011_101500 --out docs/images/run1
    python tools/plot_metrics.py <dir> --map simulation/maps/plant_map.yaml

Figures (PNG, in <dir>/figures unless --out is given)
  trajectory.png      robot path over the 2-D plant map with gauges, hot spots,
                      leak truth vs estimate, RTH trigger points
  gauge_errors.png    reading error in % of full scale per gauge (all readings +
                      final reading) against the 5 % target
  battery_comm.png    battery and RSSI over time (two panels, shared time axis),
                      RTH activations marked
  gas.png             gas concentration at the robot with the alarm threshold
                      and the seeker states
  locomotion.png      body roll / pitch and speed over time, falls marked
  summary.md          key numbers from summary.json as a Markdown table

Needs numpy + matplotlib (development / report tool, not a runtime dependency).
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "dusan_common"), os.path.join(_REPO, "simulation")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# light report palette (validated categorical slots 1-3 + fixed status colours)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
STATUS_STYLE = {"normal": (GOOD, "o"), "warning": (WARNING, "^"), "alarm": (CRITICAL, "D")}


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def latest_session(base="~/dusan_metrics"):
    base = os.path.expanduser(base)
    dirs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isfile(os.path.join(d, "summary.json"))
            or os.path.isfile(os.path.join(d, "pose.csv"))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def _t_key(row):
    try:
        return float(row.get("t_sim", ""))
    except (TypeError, ValueError):
        return float("inf")


def read_csv(path):
    """-> dict column -> list of str (empty dict if the file is missing/empty).

    Rows are sorted by t_sim: the recorder writes rows from several topic
    callbacks, which can interleave slightly out of order."""
    if not os.path.isfile(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    if "t_sim" in rows[0]:
        rows.sort(key=_t_key)
    return {k: [r.get(k, "") for r in rows] for k in rows[0].keys()}


def col(data, name):
    """Numeric column (NaN for blanks)."""
    out = []
    for v in data.get(name, []):
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, dtype=float)


def load_session(d):
    s = {name: read_csv(os.path.join(d, name + ".csv"))
         for name in ("events", "gauge_readings", "thermal", "gas", "battery", "pose")}
    try:
        with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
            s["summary"] = json.load(f)
    except (OSError, ValueError):
        s["summary"] = {}
    return s


def load_layout_and_map(map_yaml=None):
    """(Layout or None, OccupancyGridMap or None)."""
    try:
        from dusan_common.layout import Layout
        lay = Layout.load()
    except Exception:
        return None, None
    grid = None
    try:
        from dusan_common.grid_map import OccupancyGridMap
        path = map_yaml or lay.nav_map_yaml()
        if path and os.path.isfile(path):
            grid = OccupancyGridMap.load(path)
        else:
            from plant_simulation.layout_map import rasterize_layout
            grid = rasterize_layout(lay)
    except Exception:
        grid = None
    return lay, grid


# ----------------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    })


def _intervals(t, flag):
    """[(t0, t1)] where the boolean array flag is true."""
    out, start = [], None
    for ti, f in zip(t, flag):
        if f and start is None:
            start = ti
        elif not f and start is not None:
            out.append((start, ti))
            start = None
    if start is not None and len(t):
        out.append((start, t[-1]))
    return out


def _no_data(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED)


def plot_trajectory(plt, s, lay, grid, out):
    fig, ax = plt.subplots(figsize=(14, 4.6))
    if grid is not None:
        img = np.full(grid.occ.shape, 0.97)
        img[grid.occ >= 50] = 0.55
        img[grid.occ < 0] = 0.85
        ext = [grid.origin_x, grid.origin_x + grid.width * grid.resolution,
               grid.origin_y, grid.origin_y + grid.height * grid.resolution]
        ax.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest", zorder=0)
    pose = s["pose"]
    x, y = col(pose, "x"), col(pose, "y")
    if len(x):
        ax.plot(x, y, color=S1, lw=1.6, zorder=3, label="robot path")
        ax.plot(x[:1], y[:1], marker="o", ms=7, color=S1, mec=SURFACE, mew=1.5, zorder=4)
        ax.annotate("start", (x[0], y[0]), xytext=(6, 6), textcoords="offset points", color=INK2, fontsize=8)
        ax.plot(x[-1:], y[-1:], marker="s", ms=7, color=S1, mec=SURFACE, mew=1.5, zorder=4)
        ax.annotate("end", (x[-1], y[-1]), xytext=(6, -12), textcoords="offset points", color=INK2, fontsize=8)
    summ = s.get("summary") or {}
    if lay is not None:
        per = (summ.get("gauges") or {}).get("per_gauge") or {}
        for g in lay.gauges:
            st = (per.get(g["id"]) or {}).get("status")
            color, marker = STATUS_STYLE.get(st, (MUTED, "o"))
            ax.plot(g["position"][0], g["position"][1], marker=marker, ms=7, color=color if st else SURFACE,
                    mec=color if st else MUTED, mew=1.5, zorder=5)
            ax.annotate(g["id"], g["position"][:2], xytext=(5, 5), textcoords="offset points", fontsize=8, color=INK)
        det = set((summ.get("thermal") or {}).get("detected") or [])
        for h in lay.hotspots:
            st = lay.temperature_status(float(h["temp_k"]))
            color = STATUS_STYLE.get(st, (MUTED, "o"))[0]
            ax.plot(h["position"][0], h["position"][1], marker="s", ms=8, mfc="none", mec=INK2, mew=1.0, zorder=5)
            ax.plot(h["position"][0], h["position"][1], marker="s", ms=4, color=color if h["id"] in det else MUTED,
                    zorder=6)
            ax.annotate(h["id"], h["position"][:2], xytext=(5, -11), textcoords="offset points", fontsize=7,
                        color=INK2)
        leak = lay.gas_leak.get("source")
        if leak:
            ax.plot(leak[0], leak[1], marker="x", ms=9, mew=2, color=INK, zorder=6)
            ax.annotate("leak (truth)", leak[:2], xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
        dock = lay.dock_pose
        ax.plot(dock[0], dock[1], marker="s", ms=10, mfc="none", mec=S1, mew=1.5, zorder=5)
        ax.annotate("dock", dock[:2], xytext=(-8, -16), textcoords="offset points", fontsize=8, color=INK2)
    est = (summ.get("gas") or {}).get("estimate")
    if est:
        ax.plot(est[0], est[1], marker="o", ms=7, color=S3, mec=SURFACE, mew=1.5, zorder=7)
        err = (summ.get("gas") or {}).get("source_error_m")
        ax.annotate("estimate" + (f" ({err:.2f} m)" if err is not None else ""), est[:2], xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=INK2)
    for ev in (summ.get("rth") or {}).get("events") or []:
        if ev.get("x") is not None:
            ax.plot(ev["x"], ev["y"], marker="v", ms=8, color=WARNING, mec=INK, mew=0.8, zorder=8)
            ax.annotate(f"RTH: {ev.get('reason') or '?'}", (ev["x"], ev["y"]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color=INK)
    dist = summ.get("distance_m")
    ax.set_title("Robot trajectory" + (f" - {dist:.0f} m travelled" if dist else ""))
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    ax.set_aspect("equal")
    ax.grid(False)
    if not len(x):
        _no_data(ax, "no pose.csv data")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gauge_errors(plt, s, lay, out):
    gr = s["gauge_readings"]
    summ = (s.get("summary") or {}).get("gauges") or {}
    thr = float(summ.get("threshold_pct_fs") or 5.0)
    ids = sorted(set(gr.get("gauge_id", [])) | set(g["id"] for g in (lay.gauges if lay else [])))
    fig, ax = plt.subplots(figsize=(9, 4))
    if not ids:
        _no_data(ax, "no gauge readings")
    err = col(gr, "error_pct_fs")
    gid = gr.get("gauge_id", [])
    rng = np.random.default_rng(0)
    per = summ.get("per_gauge") or {}
    for i, g in enumerate(ids):
        e = np.array([v for v, k in zip(err, gid) if k == g and np.isfinite(v)])
        if e.size:
            ax.scatter(i + rng.uniform(-0.18, 0.18, e.size), e, s=10, color=S1, alpha=0.35, lw=0, zorder=3)
        last = (per.get(g) or {}).get("error_pct_fs")
        if last is not None:
            ax.plot([i - 0.28, i + 0.28], [last, last], color=INK, lw=2.0, zorder=4, solid_capstyle="butt")
            ax.annotate(f"{last:.2f}", (i + 0.3, last), va="center", fontsize=8, color=INK2)
        elif not e.size:
            ax.annotate("not read", (i, thr * 0.1), ha="center", fontsize=8, color=MUTED)
    ax.axhline(thr, color=CRITICAL, lw=1.0, zorder=2)
    ax.annotate(f"target {thr:g} % FS", (len(ids) - 0.5 if ids else 0, thr), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids)
    ax.set_xlim(-0.6, max(0.6, len(ids) - 0.4))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("reading error (% of full scale)")
    mean, p95 = summ.get("mean_error_pct_fs"), summ.get("p95_error_pct_fs")
    sub = f"mean {mean:.2f} %, 95th percentile {p95:.2f} % (dots: every reading, bar: final)" \
        if mean is not None and p95 is not None else "dots: every reading, bar: final reading"
    ax.set_title("Gauge reading error\n" + sub, fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_battery_comm(plt, s, lay, out):
    b = s["battery"]
    t = col(b, "t_sim")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    if len(t):
        t0 = t[0]
        tt = t - t0
        pct = col(b, "battery_pct")
        a1.plot(tt, pct, color=S1)
        a1.axhline(20.0, color=CRITICAL, lw=1.0)
        a1.annotate("RTH threshold 20 %", (tt[-1], 20.0), xytext=(0, 4), textcoords="offset points", ha="right",
                    fontsize=8, color=INK2)
        a1.set_ylim(0, 105)
        rssi = col(b, "rssi_dbm")
        a2.plot(tt, rssi, color=S1)
        lim = (lay.comm.get("rssi_min_dbm", -80.0) if lay else -80.0)
        a2.axhline(lim, color=CRITICAL, lw=1.0)
        a2.annotate(f"comm shadow {lim:g} dBm", (tt[-1], lim), xytext=(0, 4), textcoords="offset points",
                    ha="right", fontsize=8, color=INK2)
        rth = np.array([v.strip() == "1" for v in b.get("rth_active", [])], dtype=bool)
        for (x0, x1) in _intervals(tt, rth):
            for ax in (a1, a2):
                ax.axvspan(x0, x1, color=WARNING, alpha=0.15, lw=0)
        if rth.any():
            a1.annotate("RTH active", (tt[np.argmax(rth)], 100), xytext=(4, -10), textcoords="offset points",
                        fontsize=8, color=INK2)
        a2.set_xlabel(f"time (s, from t = {t0:.0f} s sim)")
    else:
        _no_data(a1, "no battery.csv data")
    a1.set_ylabel("battery (%)")
    a2.set_ylabel("RSSI (dBm)")
    a1.set_title("Battery and radio link")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gas(plt, s, out):
    g = s["gas"]
    t = col(g, "t_sim")
    summ = (s.get("summary") or {}).get("gas") or {}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    if len(t):
        t0 = t[0]
        tt = t - t0
        ppm = col(g, "ppm")
        ax.fill_between(tt, 0, np.nan_to_num(ppm), color=S2, alpha=0.10, lw=0)
        ax.plot(tt, ppm, color=S2)
        ax.axhline(25.0, color=CRITICAL, lw=1.0)
        ax.annotate("alarm 25 ppm", (tt[0], 25.0), xytext=(2, 4), textcoords="offset points", ha="left",
                    fontsize=8, color=INK2)
        states = g.get("seek_state", [])
        active = np.array([st not in ("", "IDLE", "FOUND", "FAILED") for st in states], dtype=bool)
        for (x0, x1) in _intervals(tt, active):
            ax.axvspan(x0, x1, color=S3, alpha=0.12, lw=0)
            ax.annotate("seeking", (x0, ax.get_ylim()[1]), xytext=(3, -12), textcoords="offset points",
                        fontsize=8, color=INK2)
        if summ.get("found_t") is not None:
            xf = summ["found_t"] - t0
            ax.axvline(xf, color=INK2, lw=1.0)
            txt = "source found"
            if summ.get("source_error_m") is not None:
                txt += f" ({summ['source_error_m']:.2f} m error"
                txt += f", {summ['time_to_find_s']:.0f} s)" if summ.get("time_to_find_s") is not None else ")"
            ax.annotate(txt, (xf, ax.get_ylim()[1] * 0.85), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.set_xlabel(f"time (s, from t = {t0:.0f} s sim)")
        ax.set_ylim(bottom=0)
    else:
        _no_data(ax, "no gas.csv data")
    ax.set_ylabel("concentration at the robot (ppm)")
    ax.set_title("Gas sensor")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_locomotion(plt, s, out):
    p = s["pose"]
    t = col(p, "t_sim")
    summ = (s.get("summary") or {}).get("locomotion") or {}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    if len(t) > 1:
        t0 = t[0]
        tt = t - t0
        a1.plot(tt, np.degrees(col(p, "roll")), color=S1, label="roll")
        a1.plot(tt, np.degrees(col(p, "pitch")), color=S2, label="pitch")
        a1.legend(loc="lower left", ncol=2)
        speed = np.hypot(col(p, "vx"), col(p, "vy"))
        a2.plot(tt, speed, color=S1)
        ev = s["events"]
        et, kinds, srcs = col(ev, "t_sim"), ev.get("kind", []), ev.get("source", [])
        for ti, k, src in zip(et, kinds, srcs):
            if src == "fall" and np.isfinite(ti):
                for ax in (a1, a2):
                    ax.axvline(ti - t0, color=CRITICAL if k == "fall" else GOOD, lw=1.0)
                a1.annotate(k, (ti - t0, a1.get_ylim()[1]), xytext=(3, -10), textcoords="offset points",
                            fontsize=8, color=INK2)
        a2.set_xlabel(f"time (s, from t = {t0:.0f} s sim)")
    else:
        _no_data(a1, "no pose.csv data")
    a1.set_ylabel("body angle (deg)")
    a2.set_ylabel("speed (m/s)")
    falls = summ.get("falls")
    mpc = (summ.get("mpc_solve_ms") or {})
    sub = []
    if falls is not None:
        sub.append(f"{falls} falls, {summ.get('recoveries', 0)} recoveries")
    if mpc.get("mean") is not None:
        sub.append(f"MPC solve mean {mpc['mean']:.1f} ms / p95 {mpc['p95']:.1f} ms")
    a1.set_title("Locomotion" + (" - " + ", ".join(sub) if sub else ""))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_summary_md(s, out):
    summ = s.get("summary") or {}
    g = summ.get("gauges") or {}
    th = summ.get("thermal") or {}
    gas = summ.get("gas") or {}
    lo = summ.get("locomotion") or {}
    rth = summ.get("rth") or {}
    mi = summ.get("mission") or {}

    def f(v, nd=2, unit=""):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.{nd}f}{unit}"
        return f"{v}{unit}"

    rows = [
        ("Duration (sim)", f(summ.get("duration_s"), 1, " s")),
        ("Distance travelled", f(summ.get("distance_m"), 1, " m")),
        ("Mission", f"{mi.get('final_state') or 'n/a'} - completed {', '.join(mi.get('completed') or []) or 'none'}"),
        ("Gauges read", f"{len(g.get('gauges_read') or [])} / {g.get('gauges_total', 'n/a')}"),
        ("Gauge error mean / p95", f"{f(g.get('mean_error_pct_fs'))} / {f(g.get('p95_error_pct_fs'))} % FS"),
        ("Readings within target", f(None if g.get("within_threshold_rate") is None
                                     else 100.0 * g["within_threshold_rate"], 1, " %")),
        ("Hot-spot recall / precision", f"{f(th.get('recall'))} / {f(th.get('precision'))}"),
        ("Gas source error", f(gas.get("source_error_m"), 2, " m")),
        ("Time to find the source", f(gas.get("time_to_find_s"), 1, " s")),
        ("Falls / recoveries", f"{lo.get('falls', 'n/a')} / {lo.get('recoveries', 'n/a')}"),
        ("MPC solve mean / p95", f"{f((lo.get('mpc_solve_ms') or {}).get('mean'), 1)} / "
                                 f"{f((lo.get('mpc_solve_ms') or {}).get('p95'), 1)} ms"),
        ("RTH events", ", ".join(f"{e.get('reason')} @ {f(e.get('t'), 0)} s" for e in rth.get("events") or [])
         or "none"),
    ]
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Run summary - {os.path.basename(os.path.dirname(out)) or out}\n\n")
        fh.write("| metric | value |\n|---|---|\n")
        for k, v in rows:
            fh.write(f"| {k} | {v} |\n")
    return rows


def make_figures(metrics_dir, out_dir=None, map_yaml=None):
    """Create every figure; returns the list of written files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    out_dir = out_dir or os.path.join(metrics_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    s = load_session(metrics_dir)
    lay, grid = load_layout_and_map(map_yaml)
    written = []
    jobs = [
        ("trajectory.png", lambda p: plot_trajectory(plt, s, lay, grid, p)),
        ("gauge_errors.png", lambda p: plot_gauge_errors(plt, s, lay, p)),
        ("battery_comm.png", lambda p: plot_battery_comm(plt, s, lay, p)),
        ("gas.png", lambda p: plot_gas(plt, s, p)),
        ("locomotion.png", lambda p: plot_locomotion(plt, s, p)),
    ]
    for name, job in jobs:
        path = os.path.join(out_dir, name)
        try:
            job(path)
            written.append(path)
        except Exception as exc:  # keep going: one broken CSV must not cost every figure
            print(f"warning: {name} failed: {exc}", file=sys.stderr)
            plt.close("all")
    md = os.path.join(out_dir, "summary.md")
    write_summary_md(s, md)
    written.append(md)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics_dir", nargs="?", default=None, help="session directory (default: newest)")
    ap.add_argument("--out", default=None, help="output directory (default: <metrics_dir>/figures)")
    ap.add_argument("--map", default=None, help="nav map YAML (default: layout nav_map)")
    a = ap.parse_args(argv)
    d = a.metrics_dir or latest_session()
    if not d or not os.path.isdir(os.path.expanduser(d)):
        ap.error("no metrics directory given / found in ~/dusan_metrics")
    d = os.path.expanduser(d)
    files = make_figures(d, a.out, a.map)
    for p in files:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
