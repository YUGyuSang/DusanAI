"""
ROS-free metrics recorder used by metrics_logger_node.

Writes one session directory  <output_dir>/<YYYYmmdd_HHMMSS>/  containing

  events.csv          t_sim, t_wall, source, kind, detail
  gauge_readings.csv  every reading joined with the layout ground truth
                      (error and error in % of full scale)
  thermal.csv         every hot-spot detection classified TP / FP against the
                      layout hot spots (id or position match)
  gas.csv             sensor ppm + seeker state (decimated)
  battery.csv         battery / RSSI / safety state (decimated)
  pose.csv            robot pose at pose_rate (2 Hz) + travelled distance
  summary.json        aggregate KPIs, rewritten periodically and on shutdown
  gauges/<id>_<t>.jpg evidence photo of every gauge inspection (/inspection/gauge_snapshot;
                      <id>_<t>_full.jpg = whole camera frame), t = simulation time of the frame
  inspection_report.json / inspection_report.html
                      per-gauge OFFICIAL result (the reading the mission accepted, /mission/state
                      "results"), ground truth, error, photo; thermal and gas results; mission
                      summary (plant_simulation.inspection_report).  Rewritten whenever a gauge
                      result or photo arrives and at mission end (DONE / DOCKED) and on close.

All times are simulation seconds (t_sim) unless named t_wall.
"""
import base64
import binascii
import csv
import json
import math
import os
import re
import time

import numpy as np

from plant_simulation import inspection_report as report

METRICS_DEFAULTS = {
    "output_dir": "~/dusan_metrics",
    "session_name": "",                 # '' -> YYYYmmdd_HHMMSS (wall clock)
    "layout_file": "",
    "pose_rate": 2.0,                   # Hz rows in pose.csv
    "gas_rate": 2.0,                    # Hz rows in gas.csv
    "battery_rate": 1.0,                # Hz rows in battery.csv
    "summary_period": 10.0,             # s (wall) between summary.json rewrites
    "flush_period": 2.0,                # s (wall) between CSV flushes
    "hotspot_match_radius": 1.0,        # m, unnamed detection -> nearest layout hot spot
    "gauge_error_threshold_pct": 5.0,   # % FS target used for the success rate
    "gas_alarm_ppm": 25.0,
    "max_jump_m": 2.0,                  # odometry jumps above this are not counted as distance
    "snapshot_max_bytes": 4000000,      # larger evidence images (decoded) are not saved
}

REPORT_FILES = ("inspection_report.json", "inspection_report.html")
_GAUGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")

CSV_COLUMNS = {
    "events.csv": ["t_sim", "t_wall", "source", "kind", "detail"],
    "gauge_readings.csv": ["t_sim", "gauge_id", "tag", "value", "unit", "confidence", "status", "n_samples",
                           "angle_deg", "true_value", "true_status", "error", "error_pct_fs", "robot_x", "robot_y",
                           "robot_yaw", "target"],
    "thermal.csv": ["t_sim", "reported_id", "matched_id", "result", "max_temp_c", "mean_temp_c", "true_temp_c",
                    "temp_error_c", "status", "true_status", "x", "y", "z", "match_dist_m"],
    "gas.csv": ["t_sim", "ppm", "seek_state", "estimate_x", "estimate_y", "error_m", "robot_x", "robot_y"],
    "battery.csv": ["t_sim", "battery_pct", "voltage", "charging", "rssi_dbm", "safety_state", "rth_active", "reason"],
    "pose.csv": ["t_sim", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "wz", "loco_mode", "gait",
                 "mission_state", "distance_m"],
}


def _f(v, nd=4):
    """CSV cell for a number ('' for None / NaN)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    return f"{x:.{nd}f}" if math.isfinite(x) else ""


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _stats(values):
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": int(v.size), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "p95": round(float(np.percentile(v, 95)), 4), "max": round(float(v.max()), 4)}


def _tidy(v, nd=4):
    """JSON copy with floats rounded (simulation times like 457.8000000000356) and NaN -> None."""
    if isinstance(v, float):
        return round(v, nd) if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _tidy(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_tidy(x, nd) for x in v]
    return v


def make_session_dir(output_dir, session_name=""):
    base = os.path.expanduser(os.path.expandvars(output_dir or "~/dusan_metrics"))
    name = session_name or time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, name)
    k = 1
    while os.path.exists(path):           # two sessions started in the same second
        path = os.path.join(base, f"{name}_{k}")
        k += 1
    os.makedirs(path, exist_ok=False)
    return path


class MetricsRecorder:
    def __init__(self, out_dir, layout=None, params=None, wall_clock=time.time):
        self.p = dict(METRICS_DEFAULTS)
        self.p.update(params or {})
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.layout = layout
        self.wall = wall_clock
        self.t_wall0 = self.wall()
        self._files = {}
        self._writers = {}
        for name, cols in CSV_COLUMNS.items():
            f = open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(cols)
            self._files[name] = f
            self._writers[name] = w
        self._gauges = {g["id"]: g for g in (layout.gauges if layout else [])}
        self._hots = {h["id"]: h for h in (layout.hotspots if layout else [])}
        th = layout.thermal if layout else {}
        warn_k = float(th.get("warning_k", 333.15))
        self.expected_hotspots = sorted(h for h, d in self._hots.items() if float(d["temp_k"]) >= warn_k)
        leak = layout.gas_leak if layout else {}
        self.leak_truth = tuple(leak["source"][:2]) if leak.get("source") else None
        # running state
        self.t_first = None
        self.t_last = None
        self.pose = None
        self.distance = 0.0
        self._last_rows = {}
        self.loco_mode = None
        self._last_loco = None
        self.gait = None
        self.mission_state = None
        self._last_mission_msg = None
        self.mission_states = []
        self.mission_start_t = None
        self.mission_end_t = None
        self.mission_completed = []
        self.mission_missed = []
        self.target = ""
        self.falls = 0
        self.recoveries = 0
        self.recovery_failed = 0
        self.assisted_resets = 0
        self.fall_events = []
        self.mpc_ms = []
        self.max_abs_roll = 0.0
        self.max_abs_pitch = 0.0
        self.fall_count_reported = None
        self.gauge_errors = []
        self.gauge_last = {}
        self.gauge_n = {}
        self.thermal_tp = 0
        self.thermal_fp = 0
        self.thermal_detected = set()
        self.thermal_peak = {}
        self.thermal_status_ok = 0
        self.thermal_status_n = 0
        self.gas_max = None
        self.gas_first_alarm_t = None
        self.gas_ppm = None
        self.seek_state = None
        self.seek_start_t = None
        self.seek_found_t = None
        self.seek_result = None
        self.seek_estimate = None
        self.seek_error = None
        self.seek_reported_error = None
        self.battery = None
        self.battery_first = None
        self.battery_min = None
        self.voltage = None
        self.charging = None
        self.rssi = None
        self.rssi_min = None
        self.safety_state = None
        self.safety_reason = None
        self.rth_active = False
        self.rth_events = []
        self.n_events = 0
        # inspection evidence + report
        self.mission_results = {}            # /mission/state "results": the mission's own per-gauge answer
        self.mission_pending = []
        self.mission_seek = None
        self.snapshots = {}                  # gid -> metadata of the latest evidence photo (+ file names)
        self.snapshot_count = 0
        self.thermal_last_status = {}
        self.report_written = None           # wall time text of the last report
        self.report_final = False
        self.reports_archived = 0
        self.on_change = None                # callback() after a photo / report was written
        self._report_state = None
        self._closed = False

    # --------------------------------------------------------------- io
    def _row(self, name, values):
        if self._closed:
            return
        self._writers[name].writerow(values)

    def _due(self, key, t, rate):
        last = self._last_rows.get(key)
        if last is None or t - last >= 1.0 / max(1e-6, rate) or t < last:
            self._last_rows[key] = t
            return True
        return False

    def _touch(self, t):
        if t is None:
            return
        t = float(t)
        if self.t_first is None:
            self.t_first = self.t_last = t
        self.t_first = min(self.t_first, t)       # callbacks of different topics arrive out of order
        self.t_last = max(self.t_last, t)

    def event(self, t, source, kind, detail=""):
        self._touch(t)
        self.n_events += 1
        self._row("events.csv", [_f(t, 3), f"{self.wall():.3f}", source, kind, detail])

    def flush(self):
        for f in self._files.values():
            if not f.closed:
                f.flush()

    def close(self):
        if self._closed:
            return
        if self.mission_results or self.snapshots or self.mission_start_t is not None:
            self.write_report()
        self.write_summary()
        self._closed = True
        for f in self._files.values():
            if not f.closed:
                f.close()

    # ----------------------------------------------------------- robot
    def on_pose(self, t, x, y, z, roll, pitch, yaw, vx=0.0, vy=0.0, wz=0.0):
        self._touch(t)
        if self.pose is not None:
            d = math.hypot(x - self.pose[0], y - self.pose[1])
            if d <= float(self.p["max_jump_m"]):
                self.distance += d
            else:
                self.event(t, "metrics", "pose_jump", f"{d:.2f} m jump not counted")
        self.pose = (float(x), float(y), float(z), float(roll), float(pitch), float(yaw))
        if self._due("pose", t, float(self.p["pose_rate"])):
            self._row("pose.csv", [_f(t, 3), _f(x), _f(y), _f(z), _f(roll), _f(pitch), _f(yaw), _f(vx), _f(vy),
                                   _f(wz), self.loco_mode or "", self.gait or "", self.mission_state or "",
                                   _f(self.distance, 3)])

    def on_locomotion(self, t, d):
        if not isinstance(d, dict):
            return
        # the node subscribes with reliable AND best-effort QoS (any publisher
        # matches), so the same sample can arrive twice: count it once
        if d == self._last_loco:
            return
        self._last_loco = d
        self._touch(t)
        mode = d.get("mode")
        if mode and mode != self.loco_mode:
            self.event(t, "locomotion", "mode", f"{self.loco_mode or '-'} -> {mode}")
            self.loco_mode = mode
        gait = d.get("gait")
        if gait and gait != self.gait:
            if self.gait is not None:
                self.event(t, "locomotion", "gait", f"{self.gait} -> {gait}")
            self.gait = gait
        ms = _num(d.get("mpc_solve_ms"))
        if ms is not None:
            self.mpc_ms.append(ms)
            if len(self.mpc_ms) > 200000:             # bound memory on very long runs
                self.mpc_ms = self.mpc_ms[::2]
        r, p = _num(d.get("roll")), _num(d.get("pitch"))
        if r is not None:
            self.max_abs_roll = max(self.max_abs_roll, abs(r))
        if p is not None:
            self.max_abs_pitch = max(self.max_abs_pitch, abs(p))
        fc = d.get("fall_count")
        if isinstance(fc, (int, float)):
            self.fall_count_reported = int(fc)

    def on_fall_event(self, t, d):
        if not isinstance(d, dict):
            return
        ev = str(d.get("event") or "")
        if ev == "fall":
            self.falls += 1
        elif ev == "recovered":
            self.recoveries += 1
        elif ev == "recovery_failed":
            self.recovery_failed += 1
        elif ev == "assisted_reset":
            self.assisted_resets += 1
        self.fall_events.append({"t": _num(t), "event": ev, "x": _num(d.get("x")), "y": _num(d.get("y"))})
        self.event(t, "fall", ev, json.dumps({k: d.get(k) for k in ("x", "y", "roll", "pitch") if k in d}))

    # ---------------------------------------------------------- mission
    def on_mission(self, t, d):
        if not isinstance(d, dict):
            return
        self._touch(t)
        st = d.get("state")
        if st and st != self.mission_state:
            self.event(t, "mission", "state", f"{self.mission_state or '-'} -> {st}")
            self.mission_states.append({"t": _num(t), "state": st})
            if st == "PATROL" and self.mission_start_t is None:
                self.mission_start_t = _num(t)
            if st == "SEEK" and self.seek_start_t is None:
                # the seek episode starts when the mission enables the seeker; a
                # seeker that reports FOUND straight away never shows an active state
                self.seek_start_t = _num(t)
            if st in ("DONE", "DOCKED") and self.mission_start_t is not None and d.get("completed"):
                self.mission_end_t = _num(t)
            self.mission_state = st
        msg = d.get("message")
        if msg and msg != self._last_mission_msg:
            self._last_mission_msg = msg
            if not str(msg).startswith("state "):
                self.event(t, "mission", "message", str(msg))
        if isinstance(d.get("completed"), list):
            self.mission_completed = list(d["completed"])
        if isinstance(d.get("missed"), list):
            self.mission_missed = list(d["missed"])
        if isinstance(d.get("pending"), list):
            self.mission_pending = [str(x) for x in d["pending"]]
        if isinstance(d.get("seek"), dict):
            self.mission_seek = dict(d["seek"])
        if "target" in d:
            self.target = d.get("target") or ""
        changed = False
        if isinstance(d.get("results"), dict):
            changed = self._on_results(t, d["results"])
        final = self.mission_state in report.FINAL_STATES
        if final and self._report_state != self.mission_state and (self.mission_results or self.snapshots):
            self._report_state = self.mission_state
            self.write_report(final=True)
        elif changed and (self.mission_results or self.snapshots):
            self.write_report(final=final)
        if not final:
            self._report_state = None

    def _on_results(self, t, results):
        """Official per-gauge results of the mission -> 'gauge result' events; True if changed."""
        new = {str(k): dict(v) for k, v in results.items() if isinstance(v, dict)}
        if new == self.mission_results:
            return False
        if self.mission_results and not new:
            self._archive_report()              # a new mission started: keep the previous report
        for gid, r in sorted(new.items()):
            if r == self.mission_results.get(gid):
                continue
            g = self._gauges.get(gid, {})
            if r.get("status") == "missed":
                txt = f"{gid} missed: {r.get('reason') or 'no reading'}"
            else:
                val, true = _num(r.get("value")), _num(g.get("value"))
                txt = f"{gid} read: " + (f"{val:.4f} {r.get('unit') or g.get('unit', '')} ({r.get('status')})"
                                         if val is not None else "no value")
                if r.get("n_readings") is not None:
                    txt += f" from {r.get('n_readings')} readings"
                span = float(g.get("max", 1.0)) - float(g.get("min", 0.0))
                if val is not None and true is not None and span > 0:
                    txt += f", error {abs(val - true) / span * 100.0:.2f} % FS"
            self.event(t, "gauge", "result", txt)
        self.mission_results = new
        return True

    def _archive_report(self):
        """Rename the report of the previous mission to inspection_report_<k>.* (photos stay)."""
        if self.report_written is None:
            self.snapshots = {}
            return
        self.reports_archived += 1
        for name in REPORT_FILES:
            src = os.path.join(self.dir, name)
            stem, ext = os.path.splitext(name)
            try:
                if os.path.exists(src):
                    os.replace(src, os.path.join(self.dir, f"{stem}_{self.reports_archived}{ext}"))
            except OSError as exc:
                self.event(self.t_last, "metrics", "write_error", f"{name}: {exc}")
        self.snapshots = {}
        self.report_written = None
        self.report_final = False
        self._changed()                         # the dashboard drops the link of the old report

    def on_target(self, t, gid):
        self.target = gid or ""

    # ------------------------------------------------------- inspection
    def on_gauge_readings(self, t, d):
        if not isinstance(d, dict):
            return
        self._touch(t)
        rx = ry = ryaw = None
        if self.pose is not None:
            rx, ry, ryaw = self.pose[0], self.pose[1], self.pose[5]
        for r in d.get("readings") or []:
            if not isinstance(r, dict) or r.get("id") is None:
                continue
            gid = str(r["id"])
            g = self._gauges.get(gid, {})
            val = _num(r.get("value"))
            true = _num(g.get("value"))
            err = err_fs = None
            if val is not None and true is not None:
                err = val - true
                span = float(g.get("max", 1.0)) - float(g.get("min", 0.0))
                err_fs = abs(err) / span * 100.0 if span > 0 else None
            true_status = self.layout.gauge_status(gid, true) if (self.layout and true is not None) else ""
            pose = r.get("robot_pose")
            if isinstance(pose, (list, tuple)) and len(pose) >= 3:
                rx, ry, ryaw = _num(pose[0]), _num(pose[1]), _num(pose[2])
            self._row("gauge_readings.csv", [_f(t, 3), gid, r.get("tag") or g.get("tag", ""), _f(val), r.get("unit") or
                                             g.get("unit", ""), _f(r.get("confidence"), 3), r.get("status") or "",
                                             r.get("n_samples") if r.get("n_samples") is not None else "",
                                             _f(r.get("angle_deg"), 2), _f(true), true_status, _f(err),
                                             _f(err_fs, 3), _f(rx, 3), _f(ry, 3), _f(ryaw, 3), self.target])
            self.gauge_n[gid] = self.gauge_n.get(gid, 0) + 1
            if err_fs is not None:
                self.gauge_errors.append(err_fs)
                self.gauge_last[gid] = {"value": val, "true_value": true, "error_pct_fs": round(err_fs, 3),
                                        "status": r.get("status"), "true_status": true_status,
                                        "confidence": _num(r.get("confidence")), "t": _num(t)}
            elif gid not in self.gauge_last:
                self.gauge_last[gid] = {"value": val, "true_value": true, "error_pct_fs": None,
                                        "status": r.get("status"), "true_status": true_status, "t": _num(t)}

    # ---------------------------------------------------- evidence images
    def _write_bytes(self, rel, data):
        """Atomic write below the session dir; False (and an event) on any OS error."""
        path = os.path.join(self.dir, *rel.split("/"))
        tmp = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return True
        except OSError as exc:
            self.event(self.t_last, "metrics", "write_error", f"{rel}: {exc}")
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False

    def _decode_jpeg(self, b64):
        if not isinstance(b64, str) or not b64:
            return None
        if len(b64) > 4 * int(self.p["snapshot_max_bytes"]) // 3 + 8:
            return None
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return None
        return data if data[:3] == b"\xff\xd8\xff" else None

    def on_gauge_snapshot(self, t, d):
        """/inspection/gauge_snapshot: evidence photo of one inspection visit ->
        gauges/<id>_<t>.jpg (+ <id>_<t>_full.jpg), 'gauge snapshot' event, report update."""
        if not isinstance(d, dict) or not _GAUGE_ID.match(str(d.get("id") or "")):
            return None
        self._touch(t)
        gid = str(d["id"])
        stamp = _num(d.get("stamp"))
        stamp = stamp if stamp is not None else (_num(t) or 0.0)
        crop = self._decode_jpeg(d.get("jpeg_b64"))
        if crop is None:
            self.event(t, "gauge", "snapshot_error", f"{gid}: no valid JPEG in the snapshot")
            return None
        prev = self.snapshots.get(gid)
        if prev and prev.get("stamp") == stamp and prev.get("bytes") == len(crop):
            return prev                         # the same message delivered twice
        base = f"gauges/{gid}_{stamp:.1f}"
        k = 1
        while os.path.exists(os.path.join(self.dir, *(base + ".jpg").split("/"))):
            k += 1
            base = f"gauges/{gid}_{stamp:.1f}_{k}"
        meta = {key: v for key, v in d.items() if not key.endswith("_b64")}
        meta.update(stamp=stamp, t_received=_num(t), bytes=len(crop),
                    count=int((prev or {}).get("count", 0)) + 1, image=None, image_full=None)
        if self._write_bytes(base + ".jpg", crop):
            meta["image"] = base + ".jpg"
        full = self._decode_jpeg(d.get("full_jpeg_b64"))
        if full is not None and self._write_bytes(base + "_full.jpg", full):
            meta["image_full"] = base + "_full.jpg"
        self.snapshots[gid] = meta
        self.snapshot_count += 1
        val = _num(d.get("value"))
        self.event(t, "gauge", "snapshot",
                   f"{gid} " + (f"{val:.4f} {d.get('unit') or ''} ({d.get('status')})" if val is not None
                                else "no reading") + f" -> {meta['image'] or 'not saved'}")
        self.write_report(final=self.mission_state in report.FINAL_STATES)
        return meta

    def report_document(self, final=None):
        """JSON document of the inspection report (plant_simulation.inspection_report)."""
        final = (self.mission_state in report.FINAL_STATES) if final is None else bool(final)
        rows = report.gauge_rows(self._gauges, self.layout, self.mission_results, self.snapshots, self.gauge_last,
                                 pending=self.mission_pending, final=final)
        trows = report.thermal_rows(self._hots, self.layout, self.thermal_detected, self.thermal_peak,
                                    self.thermal_last_status)
        return report.build_report(self.summary(), rows, trows, self.mission_seek, final=final,
                                   session_name=os.path.basename(os.path.normpath(self.dir)),
                                   written_wall=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.wall())))

    def write_report(self, final=None):
        """inspection_report.json + inspection_report.html (atomic replace).  Returns the document."""
        if self._closed:
            return None
        try:
            doc = self.report_document(final)
            page = report.render_html(doc, self.dir)
        except Exception as exc:  # a malformed message must never stop the recording
            self.event(self.t_last, "metrics", "report_error", repr(exc))
            return None
        ok = self._write_bytes("inspection_report.json",
                               json.dumps(_tidy(doc), indent=2, ensure_ascii=False, default=str).encode("utf-8"))
        ok = self._write_bytes("inspection_report.html", page.encode("utf-8")) and ok
        if ok:
            first = self.report_written is None or (doc["final"] and not self.report_final)
            self.report_written, self.report_final = doc["written_wall"], doc["final"]
            if first:
                g = doc["gauges"]
                self.event(self.t_last, "metrics", "report",
                           f"inspection_report.html ({'final' if doc['final'] else 'partial'}: "
                           f"{g['read']}/{g['total']} gauges read)")
        self._changed()
        return doc

    def session_info(self):
        """What the dashboard needs to serve photos and the report of this session."""
        return {
            "session_dir": os.path.abspath(self.dir), "session": os.path.basename(os.path.normpath(self.dir)),
            "report": "inspection_report.html" if self.report_written else None,
            "report_json": "inspection_report.json" if self.report_written else None,
            "report_final": bool(self.report_final), "report_written": self.report_written,
            "gauges": {gid: {"image": m.get("image"), "image_full": m.get("image_full"), "stamp": m.get("stamp"),
                             "value": m.get("value"), "unit": m.get("unit"), "status": m.get("status"),
                             "wall": m.get("wall")}
                       for gid, m in sorted(self.snapshots.items()) if m.get("image")},
        }

    def _changed(self):
        cb = self.on_change
        if cb is not None:
            try:
                cb()
            except Exception as exc:  # a broken listener must not stop the recording
                self.event(self.t_last, "metrics", "listener_error", repr(exc))

    def _match_hotspot(self, h):
        """-> (matched_id or None, distance or None)."""
        pos = h.get("position")
        xy = None
        if isinstance(pos, (list, tuple)) and len(pos) >= 2 and _num(pos[0]) is not None and _num(pos[1]) is not None:
            xy = (float(pos[0]), float(pos[1]))
        hid = h.get("id")
        if hid is not None and str(hid) in self._hots:
            hd = self._hots[str(hid)]
            dist = math.hypot(xy[0] - hd["position"][0], xy[1] - hd["position"][1]) if xy else None
            return str(hid), dist
        if xy is None or not self._hots:
            return None, None
        best, best_d = None, float(self.p["hotspot_match_radius"])
        for k, hd in self._hots.items():
            dd = math.hypot(xy[0] - hd["position"][0], xy[1] - hd["position"][1])
            if dd <= best_d:
                best, best_d = k, dd
        return best, (best_d if best else None)

    def on_thermal(self, t, d):
        if not isinstance(d, dict):
            return
        self._touch(t)
        for h in d.get("hotspots") or []:
            if not isinstance(h, dict):
                continue
            mid, dist = self._match_hotspot(h)
            tmax = _num(h.get("max_temp_c"))
            true_c = true_status = err = None
            if mid is not None:
                hd = self._hots[mid]
                true_c = float(hd["temp_k"]) - 273.15
                true_status = self.layout.temperature_status(float(hd["temp_k"])) if self.layout else None
                err = (tmax - true_c) if tmax is not None else None
                self.thermal_tp += 1
                self.thermal_detected.add(mid)
                if tmax is not None:
                    self.thermal_peak[mid] = max(self.thermal_peak.get(mid, -1e9), tmax)
                if h.get("status"):
                    self.thermal_last_status[mid] = str(h.get("status"))
                if true_status is not None and h.get("status"):
                    self.thermal_status_n += 1
                    self.thermal_status_ok += int(h.get("status") == true_status)
            else:
                self.thermal_fp += 1
            pos = h.get("position") if isinstance(h.get("position"), (list, tuple)) else [None, None, None]
            pos = list(pos) + [None] * (3 - len(pos))
            self._row("thermal.csv", [_f(t, 3), h.get("id") if h.get("id") is not None else "", mid or "",
                                      "TP" if mid else "FP", _f(tmax, 2), _f(h.get("mean_temp_c"), 2), _f(true_c, 2),
                                      _f(err, 2), h.get("status") or "", true_status or "", _f(pos[0], 3),
                                      _f(pos[1], 3), _f(pos[2], 3), _f(dist, 3)])

    # --------------------------------------------------------------- gas
    def on_gas(self, t, ppm):
        v = _num(ppm)
        if v is None:
            return
        self._touch(t)
        self.gas_ppm = v
        self.gas_max = v if self.gas_max is None else max(self.gas_max, v)
        if self.gas_first_alarm_t is None and v > float(self.p["gas_alarm_ppm"]):
            self.gas_first_alarm_t = _num(t)
            self.event(t, "gas", "alarm", f"{v:.1f} ppm")
        if self._due("gas", t, float(self.p["gas_rate"])):
            self._gas_row(t)

    def _gas_row(self, t):
        est = self.seek_estimate or (None, None)
        rx, ry = (self.pose[0], self.pose[1]) if self.pose else (None, None)
        self._row("gas.csv", [_f(t, 3), _f(self.gas_ppm, 3), self.seek_state or "", _f(est[0], 3), _f(est[1], 3),
                              _f(self.seek_error, 3), _f(rx, 3), _f(ry, 3)])

    def _estimate_error(self):
        if self.seek_estimate is None or self.leak_truth is None:
            return self.seek_reported_error
        return math.hypot(self.seek_estimate[0] - self.leak_truth[0], self.seek_estimate[1] - self.leak_truth[1])

    def on_seek_status(self, t, d):
        if not isinstance(d, dict):
            return
        self._touch(t)
        st = str(d.get("state") or "")
        est = d.get("estimate")
        if isinstance(est, (list, tuple)) and len(est) >= 2 and _num(est[0]) is not None and _num(est[1]) is not None:
            self.seek_estimate = (float(est[0]), float(est[1]))
        if _num(d.get("error_m")) is not None:
            self.seek_reported_error = _num(d.get("error_m"))
        self.seek_error = self._estimate_error()
        if st and st != self.seek_state:
            self.event(t, "seeker", "state", f"{self.seek_state or '-'} -> {st}")
            if st not in ("IDLE", "FOUND", "FAILED") and self.seek_start_t is None:
                self.seek_start_t = _num(t)
            if st == "FOUND" and self.seek_found_t is None:
                self.seek_found_t = _num(t)
                self.seek_result = "FOUND"
                self.event(t, "seeker", "found", f"estimate {self.seek_estimate}, error "
                                                 f"{self.seek_error if self.seek_error is not None else 'n/a'}")
            elif st == "FAILED" and self.seek_result != "FOUND":
                self.seek_result = "FAILED"
            self.seek_state = st
            self._gas_row(t)

    def on_source_estimate(self, t, x, y):
        if _num(x) is not None and _num(y) is not None:
            self.seek_estimate = (float(x), float(y))
            self.seek_error = self._estimate_error()

    # ------------------------------------------------------ battery/comm
    def on_battery(self, t, pct, voltage=None, charging=None):
        v = _num(pct)
        if v is None:
            return
        self._touch(t)
        self.battery = v
        if self.battery_first is None:
            self.battery_first = v
        self.battery_min = v if self.battery_min is None else min(self.battery_min, v)
        self.voltage = _num(voltage)
        self.charging = charging
        if self._due("battery", t, float(self.p["battery_rate"])):
            self._battery_row(t)

    def _battery_row(self, t):
        self._row("battery.csv", [_f(t, 3), _f(self.battery, 2), _f(self.voltage, 2),
                                  "" if self.charging is None else int(bool(self.charging)), _f(self.rssi, 1),
                                  self.safety_state or "", int(bool(self.rth_active)), self.safety_reason or ""])

    def on_rssi(self, t, dbm):
        v = _num(dbm)
        if v is None:
            return
        self.rssi = v
        self.rssi_min = v if self.rssi_min is None else min(self.rssi_min, v)

    def on_safety(self, t, d):
        if not isinstance(d, dict):
            return
        self._touch(t)
        st = d.get("state")
        if st and st != self.safety_state:
            self.event(t, "safety", "state", f"{self.safety_state or '-'} -> {st}")
            self.safety_state = st
        self.safety_reason = d.get("reason")
        if "rth_active" in d and d["rth_active"] is not None:
            self.on_rth_active(t, bool(d["rth_active"]))

    def on_rth_active(self, t, flag):
        flag = bool(flag)
        if flag and not self.rth_active:
            ev = {"t": _num(t), "reason": self.safety_reason, "battery_pct": self.battery, "rssi_dbm": self.rssi,
                  "x": self.pose[0] if self.pose else None, "y": self.pose[1] if self.pose else None}
            self.rth_events.append(ev)
            self.event(t, "safety", "rth", f"RTH triggered ({self.safety_reason or 'unknown'})")
        self.rth_active = flag

    # ----------------------------------------------------------- summary
    def summary(self):
        span = None if self.t_first is None else round(self.t_last - self.t_first, 2)
        gs = _stats(self.gauge_errors)
        final = [g["error_pct_fs"] for g in self.gauge_last.values() if g.get("error_pct_fs") is not None]
        fs = _stats(final)
        thr = float(self.p["gauge_error_threshold_pct"])
        exp = self.expected_hotspots
        tp, fp = self.thermal_tp, self.thermal_fp
        mpc = _stats(self.mpc_ms)
        time_to_find = (round(self.seek_found_t - self.seek_start_t, 2)
                        if self.seek_found_t is not None and self.seek_start_t is not None else None)
        alarm_to_find = (round(self.seek_found_t - self.gas_first_alarm_t, 2)
                         if self.seek_found_t is not None and self.gas_first_alarm_t is not None else None)
        return {
            "session_dir": self.dir,
            "t_sim_start": self.t_first, "t_sim_end": self.t_last, "duration_s": span,
            "wall_duration_s": round(self.wall() - self.t_wall0, 1),
            "distance_m": round(self.distance, 2),
            "locomotion": {
                "falls": self.falls, "recoveries": self.recoveries, "recovery_failed": self.recovery_failed,
                "assisted_resets": self.assisted_resets, "fall_count_reported": self.fall_count_reported,
                "recovery_rate": round(self.recoveries / self.falls, 3) if self.falls else None,
                "mpc_solve_ms": mpc,
                "max_abs_roll_deg": round(math.degrees(self.max_abs_roll), 2),
                "max_abs_pitch_deg": round(math.degrees(self.max_abs_pitch), 2),
            },
            "gauges": {
                "n_readings": gs["n"], "gauges_read": sorted(self.gauge_last), "gauges_total": len(self._gauges),
                "error_pct_fs_all": gs, "error_pct_fs_final": fs,
                "mean_error_pct_fs": gs["mean"], "p95_error_pct_fs": gs["p95"],
                "within_threshold_rate": (round(sum(e <= thr for e in self.gauge_errors) / len(self.gauge_errors), 4)
                                          if self.gauge_errors else None),
                "threshold_pct_fs": thr,
                "status_correct_final": sum(1 for g in self.gauge_last.values()
                                            if g.get("status") and g.get("status") == g.get("true_status")),
                "per_gauge": {k: dict(v, n=self.gauge_n.get(k, 0)) for k, v in sorted(self.gauge_last.items())},
            },
            "thermal": {
                "detections": tp + fp, "tp": tp, "fp": fp,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
                "expected": exp, "detected": sorted(self.thermal_detected),
                "recall": round(len(set(exp) & self.thermal_detected) / len(exp), 4) if exp else None,
                "status_accuracy": (round(self.thermal_status_ok / self.thermal_status_n, 4)
                                    if self.thermal_status_n else None),
                "peak_temp_c": {k: round(v, 2) for k, v in sorted(self.thermal_peak.items())},
            },
            "gas": {
                "max_ppm": None if self.gas_max is None else round(self.gas_max, 2),
                "first_alarm_t": self.gas_first_alarm_t, "seek_start_t": self.seek_start_t,
                "found_t": self.seek_found_t, "time_to_find_s": time_to_find, "alarm_to_find_s": alarm_to_find,
                "result": self.seek_result or (self.seek_state or None),
                "estimate": list(self.seek_estimate) if self.seek_estimate else None,
                "truth": list(self.leak_truth) if self.leak_truth else None,
                "source_error_m": None if self.seek_error is None else round(self.seek_error, 3),
                "reported_error_m": self.seek_reported_error,
            },
            "battery": {"start_pct": self.battery_first, "min_pct": self.battery_min, "end_pct": self.battery},
            "comm": {"min_rssi_dbm": self.rssi_min, "last_rssi_dbm": self.rssi},
            "rth": {"count": len(self.rth_events), "events": list(self.rth_events)},
            "mission": {
                "start_t": self.mission_start_t, "end_t": self.mission_end_t,
                "duration_s": (round(self.mission_end_t - self.mission_start_t, 2)
                               if self.mission_end_t is not None and self.mission_start_t is not None else None),
                "completed": self.mission_completed, "missed": self.mission_missed,
                "final_state": self.mission_state, "transitions": self.mission_states[-200:],
            },
            "events_logged": self.n_events,
            "inspection": {
                "report": "inspection_report.html" if self.report_written else None,
                "report_final": bool(self.report_final), "snapshots_saved": self.snapshot_count,
                "photos": {gid: m.get("image") for gid, m in sorted(self.snapshots.items())},
                "mission_results": self.mission_results,
            },
        }

    def write_summary(self):
        if self._closed:
            return None
        s = self.summary()
        path = os.path.join(self.dir, "summary.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, default=str)
        os.replace(tmp, path)
        self.flush()
        return s
