"""
ROS-free aggregator behind the digital-twin dashboard.

The monitor_dashboard node feeds every topic into a :class:`DashboardState`;
the HTTP server thread reads JSON snapshots from it (5 Hz over Server-Sent
Events).  All public methods are thread safe (one re-entrant lock).

Besides storing the latest value of every stream the aggregator
* joins readings with the layout ground truth (gauge error in % of full
  scale, hot-spot temperature error, leak-estimate error),
* derives an event log from state transitions (mission state, safety/RTH,
  falls, gauge/thermal status changes, gas alarm, seeker result, slip mode
  on/off and slip detections ...),
* keeps short histories (gas ppm sparkline, battery),
* tracks per-topic liveness for the "links" health panel.

Slippery ground (module1 README section 3.8): the low-friction plates /
``friction_zones`` of the layout go to the static info (drawn on the twin
map); the friction prior of /terrain/info (``slippery``, ``friction_mu``,
``friction_zones``, ``friction_distance``) and the slip state of
/locomotion/state (``slip``, ``slip_mode``, ``slip_source``, ``slip_speed``,
``slip_legs``, ``slip_events``, ``mu_est``, ``mpc_mu``, ``friction_prior``,
``torque_ff``) are passed through to the snapshot.  Older nodes without these
fields simply leave them out of the snapshot (the browser shows a dash).

Inspection evidence: /metrics/session (metrics_logger, :meth:`DashboardState.set_session`)
names the metrics session directory and its photo / report files.  The snapshot then gives every
gauge with a photo ``image_url`` / ``image_full_url`` (``session/gauges/<file>.jpg``, served by
plant_simulation.web_server from that directory) and a top-level ``report`` object
(``{url, json_url, final, written, session}``) for the inspection_report.html link.  Only a
directory inside ``metrics_root`` (default ~/dusan_metrics, the metrics_logger output_dir) is
accepted.  Once the mission has accepted a gauge (``/mission/state`` ``results``) its tile shows
that official value (``value_source: "mission"``) instead of later live readings.
"""
import math
import os
import threading
import time
from collections import deque
from urllib.parse import quote

import numpy as np

from dusan_common import robot_model

GAS_ALARM_PPM = 25.0
GAS_ALARM_CLEAR_PPM = 20.0
BATTERY_LOW_PCT = 20.0

# low-friction areas of the layout, same entries and limits as module1
# friction_map.py / config/terrain.yaml: obstacles of type 'plate' (rect + mu)
# and the optional top-level 'friction_zones' list
FRICTION_ZONE_MAX_MU = 0.6     # zones with a higher mu are ordinary floor (friction_zone_max_mu)
SLIPPERY_MU = 0.35             # zones below make the terrain 'slippery' (classifier.slippery_mu)

# fields of /locomotion/state and /terrain/info passed to the browser
LOCO_KEYS = ("mode", "locomotion_mode", "gait", "gait_auto", "phase", "contacts", "body_height", "roll", "pitch",
             "fall_count", "recovery_attempts", "recovery_success", "mpc_solve_ms", "mpc_status",
             "stability_margin", "terrain", "cmd",
             # slippery ground (module1 README section 3.8)
             "slip", "slip_mode", "slip_source", "slip_speed", "slip_legs", "slip_events", "mu_est", "mpc_mu",
             "friction_prior", "torque_ff")
TERRAIN_KEYS = ("class", "roughness", "max_step", "slope_deg", "recommended_gait", "ahead_clear",
                # friction prior (module1 TERRAIN_AND_TELEOP.md section 5.1)
                "slippery", "friction_mu", "friction_zones", "friction_distance")
SLIP_SOURCE_TEXT = {"terrain": "terrain prior", "detected": "slip detected", "both": "terrain prior + slip detected"}

DASHBOARD_DEFAULTS = {
    "port": 8080,
    "host": "0.0.0.0",
    "web_dir": "",                 # '' -> <share>/plant_simulation/web (source tree fallback)
    "layout_file": "",
    "sse_rate": 5.0,               # Hz JSON snapshots on /events
    "image_max_rate": 2.0,         # Hz max re-encode of /image/*.jpg
    "image_max_width": 640,        # px, camera frames are downscaled for the browser
    "jpeg_quality": 80,
    "gas_png_max_rate": 1.0,       # Hz max re-render of /gas.png
    "markers_rate": 1.0,           # Hz /twin/markers
    "history_s": 120.0,            # s of gas / battery history kept
    "max_events": 300,
    "path_max_points": 400,
    "port_fallbacks": 5,           # try port+1 .. port+N when the port is busy
    "gas_alarm_ppm": GAS_ALARM_PPM,
    "metrics_root": "~/dusan_metrics",   # session dirs of /metrics/session must be inside (photos, report)
}

# /teleop/active is a 2 Hz (wall clock) heartbeat: shown as off once it is older
# than this (same rule as the cmd_vel mux flag_timeout - a killed teleop never sends false)
TELEOP_FLAG_TIMEOUT = 1.5

# streams shown in the "links" health panel: topic -> expected max period (s)
WATCHED_TOPICS = {
    "/odom": 1.0,
    "/locomotion/state": 2.0,
    "/terrain/info": 3.0,
    "/mission/state": 3.0,
    "/safety/status": 3.0,
    "/battery/state": 3.0,
    "/comm/rssi": 3.0,
    "/gas/sensor": 2.0,
    "/gas/concentration_grid": 5.0,
    "/gas/seek_status": 5.0,
    "/inspection/gauge_readings": 0.0,     # event driven: never "stale", only "never seen"
    "/inspection/thermal_alerts": 0.0,
    "/inspection/gauge_overlay": 3.0,
    "/inspection/thermal_fusion": 3.0,
    "/metrics/session": 0.0,
    "/patrol/plan_json": 0.0,
    "/map": 0.0,
    "/cmd_vel": 1.0,
}


def _r(v, nd=3):
    """Round finite floats, None for NaN/inf/garbage (JSON friendly)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _jsonable(v):
    """JSON-safe copy of a parsed message value: non-finite floats -> None
    (json.loads accepts NaN / Infinity, the browser's JSON.parse does not)."""
    if v is None or isinstance(v, (bool, str, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return _r(v, 6)


def _num(v):
    """Finite float or None (bools are not numbers here)."""
    if isinstance(v, bool):
        return None
    return _r(v, 6)


def _rect_area(rect):
    return abs((rect[2] - rect[0]) * (rect[3] - rect[1]))


def friction_zones_from_layout(layout, max_mu=FRICTION_ZONE_MAX_MU):
    """Low-friction areas of the layout for the twin map:
    ``[{id, kind: plate|zone, rect: [x_min, y_min, x_max, y_max], mu, slippery}]``.

    Plates are ``obstacles`` of ``type: plate`` with ``rect`` and ``mu``; other
    areas come from the optional top-level ``friction_zones`` list.  Zones with
    ``mu >= max_mu`` are ordinary floor; malformed entries are skipped."""
    if layout is None:
        return []
    items = [("plate", ob) for ob in (layout.get("obstacles", []) or [])
             if isinstance(ob, dict) and str(ob.get("type", "")).lower() == "plate"]
    items += [("zone", z) for z in (layout.get("friction_zones", []) or [])]
    out = []
    for kind, z in items:
        if not isinstance(z, dict):
            continue
        mu = _num(z.get("mu"))
        try:
            r = [float(v) for v in z.get("rect")]
        except (TypeError, ValueError):
            continue
        if len(r) != 4 or not all(math.isfinite(v) for v in r) or mu is None or not 0.0 < mu < float(max_mu):
            continue
        out.append({"id": str(z.get("id") or f"{kind}{len(out)}"), "kind": kind,
                    "rect": [round(min(r[0], r[2]), 3), round(min(r[1], r[3]), 3),
                             round(max(r[0], r[2]), 3), round(max(r[1], r[3]), 3)],
                    "mu": round(mu, 3), "slippery": mu < SLIPPERY_MU})
    return out


def downsample_path(pts, max_points=400, nd=2):
    """[(x, y, ...)] -> [[x, y]] with at most max_points (first/last kept)."""
    if not pts:
        return []
    clean = []
    for p in pts:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            clean.append((x, y))
    if len(clean) > max_points:
        idx = np.linspace(0, len(clean) - 1, max_points).round().astype(int)
        clean = [clean[i] for i in idx]
    return [[round(x, nd), round(y, nd)] for x, y in clean]


def build_static_info(layout):
    """Everything the browser needs once: layout geometry + ground truth."""
    if layout is None:
        return {"gauges": [], "hotspots": [], "zones": [], "aps": [], "obstacles": [], "machines": [],
                "friction_zones": []}
    th = layout.thermal
    gauges = []
    for g in layout.gauges:
        px, py, pz = (float(c) for c in g["position"])
        gauges.append({
            "id": g["id"], "tag": g.get("tag", ""), "x": px, "y": py, "z": pz,
            "normal": [float(c) for c in (g.get("normal") or [0, 1, 0])[:2]],
            "min": g.get("min"), "max": g.get("max"), "unit": g.get("unit", ""),
            "true_value": g.get("value"), "true_status": layout.gauge_status(g["id"], g.get("value")),
            "normal_range": [g.get("normal_min"), g.get("normal_max")],
            "alarm_range": [g.get("alarm_min"), g.get("alarm_max")],
        })
    hotspots = []
    for h in layout.hotspots:
        hotspots.append({
            "id": h["id"], "tag": h.get("tag", ""), "kind": h.get("kind", ""),
            "x": float(h["position"][0]), "y": float(h["position"][1]), "z": float(h["position"][2]),
            "true_temp_c": _r(float(h["temp_k"]) - 273.15, 1),
            "true_status": layout.temperature_status(float(h["temp_k"])),
        })
    comm = layout.comm
    aps = [{"id": a["id"], "x": a["position"][0], "y": a["position"][1]} for a in comm.get("access_points", [])]
    shadows = [{"id": s["id"], "name": s.get("name", ""), "rect": s["rect"]} for s in comm.get("shadow_zones", [])]
    zones = [{"id": z["id"], "name": z.get("name", ""), "rect": z["rect"]} for z in layout.get("zones", []) or []]
    machines = [{"id": m["id"], "footprint": m["footprint"]} for m in layout.get("machines", []) or []
                if m.get("footprint")]
    obstacles = []
    for ob in layout.obstacles:
        o = {"id": ob["id"], "type": ob.get("type")}
        for k in ("center", "rect", "yaw", "length", "diameter", "radius", "stack"):
            if k in ob:
                o[k] = ob[k]
        obstacles.append(o)
    leak = layout.gas_leak
    cs = layout.get("charging_station", {}) or {}
    dock = layout.dock_pose
    rack = layout.get("pipe_rack") or {}
    fac = (layout.get("factory") or {}).get("interior")
    cor = (layout.get("corridor") or {}).get("interior")
    feet = [[robot_model.LEG_SIGN_X[i] * robot_model.NOMINAL_FOOT_X,
             robot_model.LEG_SIGN_Y[i] * robot_model.NOMINAL_FOOT_Y] for i in range(4)]
    return {
        "world_name": layout.get("world_name", "plant"),
        "bounds": (layout.get("nav_map", {}) or {}).get("bounds"),
        "factory": fac, "corridor": cor,
        "gauges": gauges, "hotspots": hotspots, "aps": aps, "shadow_zones": shadows, "zones": zones,
        "machines": machines, "obstacles": obstacles,
        "friction_zones": friction_zones_from_layout(layout),
        "pipe_rack": {"start": rack.get("pipe", {}).get("start"), "end": rack.get("pipe", {}).get("end")}
        if rack else None,
        "gas_leak": {"id": leak.get("id"), "tag": leak.get("tag"),
                     "x": (leak.get("source") or [None, None])[0], "y": (leak.get("source") or [None, None])[1]},
        "dock": {"x": dock[0], "y": dock[1], "yaw": dock[2], "size": (cs.get("pad_size") or [1.0, 1.0])[:2]},
        "spawn": list(layout.spawn),
        "thermal": {"warning_c": _r(th.get("warning_k", 333.15) - 273.15, 1),
                    "alarm_c": _r(th.get("alarm_k", 353.15) - 273.15, 1)},
        "comm": {"rssi_min_dbm": comm.get("rssi_min_dbm", -80.0)},
        "robot": {"body": list(robot_model.BASE_BOX[:2]), "feet": feet},
    }


def _within(path, root):
    """True if the real path ``path`` is ``root`` or below it."""
    try:
        p, r = os.path.realpath(path), os.path.realpath(root)
    except (TypeError, ValueError, OSError):
        return False
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def session_rel(rel):
    """A photo / report file name of /metrics/session: plain relative path below the session dir
    (no '..', no hidden parts, no backslash / NUL / drive), else None."""
    if not isinstance(rel, str) or not rel or len(rel) > 200 or rel.startswith("/"):
        return None
    if any(c in rel for c in ("\\", "\x00", ":")):
        return None
    parts = rel.split("/")
    if any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return None
    return rel


def session_url(rel):
    """Same-origin URL of a session file (relative to the dashboard page)."""
    return "session/" + quote(rel, safe="/")


class DashboardState:
    def __init__(self, layout=None, history_s=120.0, max_events=300, path_max_points=400,
                 gas_alarm_ppm=GAS_ALARM_PPM, metrics_root=DASHBOARD_DEFAULTS["metrics_root"]):
        self._lock = threading.RLock()
        self.layout = layout
        self.static = build_static_info(layout)
        self._gauge_def = {g["id"]: g for g in (layout.gauges if layout else [])}
        self._hot_def = {h["id"]: h for h in (layout.hotspots if layout else [])}
        self._zones = sorted((z for z in (self.static.get("zones") or [])), key=lambda z: _rect_area(z["rect"]))
        self.history_s = float(history_s)
        self.path_max_points = int(path_max_points)
        self.gas_alarm_ppm = float(gas_alarm_ppm)
        self.t0_wall = time.time()
        self.sim_time = None
        self.robot = {}
        self.locomotion = {}
        self.terrain = {}
        self.mission = {}
        self.safety = {}
        self.battery = {}
        self.rssi = {}
        self.gas = {"ppm": None, "t": None, "field": {}, "seek": {}, "estimate": None, "alarm": False}
        self._gas_hist = deque()
        self._batt_hist = deque()
        self.gauges = {}
        self.thermal = {}
        self.thermal_unassigned = []
        self.paths = {}
        self.plan = None
        self.cmd = {}
        self.mux = None
        self.flags = {"teleop_active": False, "rth_active": False}
        self._flag_t = {}
        self.layers = {}
        self._layer_data = {}
        self.images = {}
        self.metrics_root = os.path.expanduser(os.path.expandvars(metrics_root)) if metrics_root else None
        self.session = None            # validated /metrics/session (session_dir, photos, report)
        self.events = deque(maxlen=int(max_events))
        self._event_seq = 0
        self.topic_seen = {}
        self._last = {}            # previous values for change detection
        self.add_event("info", "dashboard", "digital twin dashboard started")

    # ----------------------------------------------------------- utilities
    def _now(self, now):
        return time.time() if now is None else float(now)

    def touch(self, topic, now=None):
        with self._lock:
            self.topic_seen[topic] = self._now(now)

    def add_event(self, level, source, text, now=None):
        with self._lock:
            self._event_seq += 1
            self.events.append({"seq": self._event_seq, "t_wall": round(self._now(now), 2),
                                "t_sim": _r(self.sim_time, 2), "level": level, "source": source, "text": text})

    def _changed(self, key, value):
        old = self._last.get(key, "__unset__")
        self._last[key] = value
        return old != value, (None if old == "__unset__" else old)

    def set_sim_time(self, t):
        with self._lock:
            self.sim_time = float(t)

    def zone_at(self, x, y):
        for z in self._zones:
            x0, y0, x1, y1 = z["rect"]
            if min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1):
                return z["name"]
        return ""

    # ------------------------------------------------------------- robot
    def set_robot(self, x, y, z, roll, pitch, yaw, vx=0.0, vy=0.0, wz=0.0, t_sim=None, now=None):
        with self._lock:
            now = self._now(now)
            if t_sim is not None:
                self.sim_time = float(t_sim)
            self.robot = {"x": _r(x), "y": _r(y), "z": _r(z), "roll": _r(roll, 4), "pitch": _r(pitch, 4),
                          "yaw": _r(yaw, 4), "vx": _r(vx), "vy": _r(vy), "wz": _r(wz),
                          "zone": self.zone_at(float(x), float(y)), "t_wall": now}
            self.topic_seen["/odom"] = now

    def set_locomotion(self, d, now=None):
        if not isinstance(d, dict):
            return
        with self._lock:
            now = self._now(now)
            self.locomotion = dict(d)
            self.locomotion["t_wall"] = now
            self.topic_seen["/locomotion/state"] = now
            mode = str(d.get("mode") or "")
            ch, old = self._changed("loco_mode", mode)
            # WALK <-> STAND toggles at every viewpoint: keep them out of the log
            if ch and mode and not ({mode, old or "WALK"} <= {"WALK", "STAND"}):
                level = "alarm" if mode == "FALLEN" else ("warn" if mode in ("RECOVERING", "ESTOP") else "info")
                self.add_event(level, "locomotion", f"mode {old or '-'} -> {mode}", now)
            gait = str(d.get("gait") or "")
            ch, old = self._changed("loco_gait", gait)
            if ch and gait and old:
                self.add_event("info", "locomotion", f"gait {old} -> {gait}", now)
            algo = str(d.get("locomotion_mode") or "")      # stable | mpc (module1 README section 4.6)
            ch, old = self._changed("loco_algo", algo)
            if ch and algo and old:
                self.add_event("info", "locomotion", f"locomotion mode {old} -> {algo}", now)
            self._slip_events(d, now)

    @staticmethod
    def _slip_detail(d):
        legs = d.get("slip_legs")
        names = [n for n, on in zip(("FL", "FR", "RL", "RR"), legs) if on is True] \
            if isinstance(legs, (list, tuple)) else []
        speed, mu_est = _num(d.get("slip_speed")), _num(d.get("mu_est"))
        parts = []
        if names:
            parts.append("feet " + " ".join(names))
        if speed is not None:
            parts.append(f"{speed:.2f} m/s")
        if mu_est is not None:
            parts.append(f"mu est {mu_est:.2f}")
        return ", ".join(parts)

    def _slip_events(self, d, now):
        """Event log entries of the slippery-ground state (fields of newer
        locomotion controllers only; absent fields leave the log untouched)."""
        new_detection, n = False, None
        ev = _num(d.get("slip_events"))
        if ev is not None:
            n = int(ev)
            ch, old = self._changed("loco_slip_events", n)
            new_detection = ch and old is not None and n > old       # a restarted node counts from 0 again
        mode_on = None
        if "slip_mode" in d and d["slip_mode"] is not None:
            on = bool(d["slip_mode"])
            ch, old = self._changed("loco_slip_mode", on)
            if ch and (old is not None or on):                       # first sample: log only an active mode
                mode_on = on
        detail = self._slip_detail(d)
        if new_detection and mode_on is not True:
            self.add_event("warn", "locomotion", f"slip detected #{n}" + (f": {detail}" if detail else ""), now)
        if mode_on is True:
            src = SLIP_SOURCE_TEXT.get(str(d.get("slip_source") or ""), "")
            prior, mpc_mu = _num(d.get("friction_prior")), _num(d.get("mpc_mu"))
            info = [src] if src else []
            if prior is not None:
                info.append(f"prior mu {prior:.2f}")
            if mpc_mu is not None:
                info.append(f"MPC mu {mpc_mu:.2f}")
            if new_detection:
                info.append(f"detection #{n}" + (f": {detail}" if detail else ""))
            self.add_event("warn", "locomotion", "slip mode ON" + (f" ({', '.join(info)})" if info else ""), now)
        elif mode_on is False:
            self.add_event("info", "locomotion", "slip mode OFF"
                           + (f" ({n} detection{'s' if n != 1 else ''} so far)" if n else ""), now)

    def set_fall_event(self, d, now=None):
        if not isinstance(d, dict):
            return
        ev = str(d.get("event") or "fall")
        level = {"fall": "alarm", "recovered": "info", "recovery_failed": "alarm", "assisted_reset": "warn"}.get(ev, "warn")
        txt = f"{ev}"
        if _r(d.get("x")) is not None and _r(d.get("y")) is not None:
            txt += f" at ({_r(d.get('x'), 2)}, {_r(d.get('y'), 2)})"
        roll, pitch = _r(d.get("roll"), 4), _r(d.get("pitch"), 4)
        if roll is not None:
            txt += f" roll {math.degrees(roll):.0f} deg pitch {math.degrees(pitch or 0.0):.0f} deg"
        self.add_event(level, "fall", txt, now)

    def set_terrain(self, d, now=None):
        if isinstance(d, dict):
            with self._lock:
                now = self._now(now)
                self.terrain = dict(d)
                self.topic_seen["/terrain/info"] = now

    def set_cmd_vel(self, vx, vy, wz, now=None):
        with self._lock:
            now = self._now(now)
            self.cmd = {"vx": _r(vx), "vy": _r(vy), "wz": _r(wz), "t_wall": now}
            self.topic_seen["/cmd_vel"] = now

    def set_mux(self, name, now=None):
        with self._lock:
            self.mux = str(name)
            self.topic_seen["/cmd_vel_mux/active"] = self._now(now)

    def set_flag(self, name, value, now=None):
        with self._lock:
            value = bool(value)
            ch, old = self._changed(f"flag_{name}", value)
            self.flags[name] = value
            self._flag_t[name] = self._now(now)
            if ch and old is not None:
                text = {"teleop_active": ("manual override ON", "manual override OFF"),
                        "rth_active": ("return-to-home ACTIVE", "return-to-home released")}.get(
                    name, (f"{name} on", f"{name} off"))
                self.add_event("warn" if value else "info", "safety" if name == "rth_active" else "teleop",
                               text[0] if value else text[1], now)

    def _flags_view(self, now):
        flags = dict(self.flags)
        t = self._flag_t.get("teleop_active")
        if flags.get("teleop_active") and t is not None and now - t > TELEOP_FLAG_TIMEOUT:
            flags["teleop_active"] = False            # heartbeat stopped (teleop killed)
        return flags

    # ----------------------------------------------------------- mission
    def set_mission(self, d, now=None):
        if not isinstance(d, dict):
            return
        with self._lock:
            now = self._now(now)
            self.mission = dict(d)
            self.topic_seen["/mission/state"] = now
            st = str(d.get("state") or "")
            ch, old = self._changed("mission_state", st)
            if ch and st:
                level = "alarm" if st == "SEEK" else ("warn" if st in ("RTH", "MANUAL", "PAUSED") else "info")
                self.add_event(level, "mission", f"state {old or '-'} -> {st}", now)
            msg = str(d.get("message") or "")
            ch, _ = self._changed("mission_msg", msg)
            if ch and msg and not msg.startswith("state "):
                low = msg.lower()
                level = ("alarm" if ("gas" in low and ("alarm" in low or "found" in low)) else
                         "warn" if any(k in low for k in ("skipped", "stuck", "fail", "timeout", "abort", "not ready",
                                                           "manual", "waiting")) else "info")
                self.add_event(level, "mission", msg, now)

    def set_plan(self, d, now=None):
        if not isinstance(d, dict):
            return
        with self._lock:
            now = self._now(now)
            vps = []
            for v in d.get("viewpoints") or []:
                if not isinstance(v, dict) or _r(v.get("x")) is None or _r(v.get("y")) is None:
                    continue
                vps.append({"gauge_id": v.get("gauge_id"), "x": _r(v["x"]), "y": _r(v["y"]),
                            "yaw": _r(v.get("yaw", 0.0), 4), "score": _r(v.get("score"), 3)})
            new = {"order": list(d.get("order") or []), "viewpoints": vps, "total_length": _r(d.get("total_length"), 1)}
            if new != self.plan:
                self.add_event("info", "patrol", f"patrol plan: {len(vps)} viewpoints, order "
                                                 f"{', '.join(map(str, new['order']))}"
                                                 + (f", {new['total_length']} m" if new["total_length"] else ""), now)
            self.plan = new
            self.topic_seen["/patrol/plan_json"] = now

    def set_path(self, name, pts, now=None):
        with self._lock:
            self.paths[name] = downsample_path(pts, self.path_max_points)

    # ------------------------------------------------------------ safety
    def set_safety(self, d, now=None):
        if not isinstance(d, dict):
            return
        with self._lock:
            now = self._now(now)
            self.safety = dict(d)
            self.topic_seen["/safety/status"] = now
            st = str(d.get("state") or "")
            ch, old = self._changed("safety_state", st)
            if ch and st and old is not None:
                level = "warn" if st.startswith("RTH") else "info"
                reason = d.get("reason")
                self.add_event(level, "safety", f"{old} -> {st}" + (f" ({reason})" if reason and st != "MONITOR" else ""),
                               now)
            if "rth_active" in d and d["rth_active"] is not None:
                self.set_flag("rth_active", bool(d["rth_active"]), now)

    def set_battery(self, pct, voltage=None, current=None, charging=None, now=None):
        """pct in percent 0..100 (NaN allowed)."""
        with self._lock:
            now = self._now(now)
            p = _r(pct, 2)
            self.battery = {"pct": p, "voltage": _r(voltage, 2), "current": _r(current, 2),
                            "charging": charging, "t_wall": now}
            self.topic_seen["/battery/state"] = now
            if p is not None:
                self._batt_hist.append((now, p))
                self._trim(self._batt_hist, now)
                low = p < BATTERY_LOW_PCT
                ch, old = self._changed("battery_low", low)
                if ch and old is not None:
                    self.add_event("warn" if low else "info", "battery",
                                   f"battery {'LOW' if low else 'recovered'}: {p:.0f} %", now)

    def set_rssi(self, dbm, now=None):
        with self._lock:
            now = self._now(now)
            self.rssi = {"dbm": _r(dbm, 1), "t_wall": now}
            self.topic_seen["/comm/rssi"] = now

    # --------------------------------------------------------------- gas
    def _trim(self, dq, now):
        while dq and now - dq[0][0] > self.history_s:
            dq.popleft()

    def set_gas_ppm(self, ppm, now=None):
        with self._lock:
            now = self._now(now)
            v = _r(ppm, 2)
            self.gas["ppm"] = v
            self.gas["t_wall"] = now
            self.topic_seen["/gas/sensor"] = now
            if v is None:
                return
            # keep the sparkline at <= 4 Hz
            if not self._gas_hist or now - self._gas_hist[-1][0] >= 0.25:
                self._gas_hist.append((now, v))
            self._trim(self._gas_hist, now)
            alarm = self.gas["alarm"]
            if not alarm and v > self.gas_alarm_ppm:
                alarm = True
                self.add_event("alarm", "gas", f"gas {v:.1f} ppm above {self.gas_alarm_ppm:.0f} ppm", now)
            elif alarm and v < min(GAS_ALARM_CLEAR_PPM, self.gas_alarm_ppm):
                alarm = False
                self.add_event("info", "gas", f"gas back to {v:.1f} ppm", now)
            self.gas["alarm"] = alarm

    def set_gas_field(self, d, now=None):
        if isinstance(d, dict):
            with self._lock:
                self.gas["field"] = dict(d)

    def set_seek_status(self, d, now=None):
        if not isinstance(d, dict):
            return
        with self._lock:
            now = self._now(now)
            self.gas["seek"] = dict(d)
            self.topic_seen["/gas/seek_status"] = now
            est = d.get("estimate")
            # only finite numbers: a [None, None] estimate would make leak_error() raise
            if isinstance(est, (list, tuple)) and len(est) >= 2:
                ex, ey = _r(est[0], 2), _r(est[1], 2)
                if ex is not None and ey is not None:
                    self.gas["estimate"] = [ex, ey]
            st = str(d.get("state") or "")
            ch, old = self._changed("seek_state", st)
            if ch and st and old is not None:
                if st == "FOUND":
                    err = self.leak_error()
                    self.add_event("alarm", "gas", "LEAK SOURCE FOUND"
                                   + (f" at ({self.gas['estimate'][0]}, {self.gas['estimate'][1]})"
                                      if self.gas["estimate"] else "")
                                   + (f" - error {err:.2f} m" if err is not None else ""), now)
                elif st == "FAILED":
                    self.add_event("warn", "gas", "source seeking FAILED", now)
                else:
                    self.add_event("info", "gas", f"seeker {old} -> {st}", now)

    def set_source_estimate(self, x, y, now=None):
        with self._lock:
            if _r(x, 2) is not None and _r(y, 2) is not None:
                self.gas["estimate"] = [_r(x, 2), _r(y, 2)]

    def leak_error(self):
        est = self.gas.get("estimate")
        leak = self.static.get("gas_leak") or {}
        if not est or leak.get("x") is None:
            return None
        return math.hypot(est[0] - float(leak["x"]), est[1] - float(leak["y"]))

    # ------------------------------------------------------- inspection
    def set_gauge_readings(self, d, now=None):
        if not isinstance(d, dict):
            return
        with self._lock:
            now = self._now(now)
            self.topic_seen["/inspection/gauge_readings"] = now
            for r in d.get("readings") or []:
                if not isinstance(r, dict) or r.get("id") is None:
                    continue
                gid = str(r["id"])
                g = self._gauge_def.get(gid, {})
                val = _r(r.get("value"), 4)
                true = g.get("value")
                err = err_fs = None
                if val is not None and true is not None:
                    err = val - float(true)
                    span = float(g.get("max", 1.0)) - float(g.get("min", 0.0))
                    err_fs = abs(err) / span * 100.0 if span > 0 else None
                status = r.get("status") or (self.layout.gauge_status(gid, val) if self.layout and val is not None
                                             else "unknown")
                prev = self.gauges.get(gid, {})
                self.gauges[gid] = {
                    "id": gid, "tag": r.get("tag") or g.get("tag", ""), "value": val,
                    "unit": r.get("unit") or g.get("unit", ""), "confidence": _r(r.get("confidence"), 3),
                    "status": status, "angle_deg": _r(r.get("angle_deg"), 2), "n_samples": r.get("n_samples"),
                    "true_value": true, "error": _r(err, 4), "error_pct_fs": _r(err_fs, 2),
                    "n_updates": int(prev.get("n_updates", 0)) + 1, "t_sim": _r(self.sim_time, 2), "t_wall": now,
                }
                ch, old = self._changed(f"gauge_status_{gid}", status)
                if ch:
                    level = "alarm" if status == "alarm" else ("warn" if status == "warning" else "info")
                    vtxt = f"{val:.3f}" if val is not None else "n/a"
                    self.add_event(level, "gauge",
                                   f"{gid} {self.gauges[gid]['tag']}: {vtxt} {self.gauges[gid]['unit']} ({status})"
                                   + (f", error {err_fs:.1f} % FS" if err_fs is not None else ""), now)

    def set_thermal_alerts(self, d, now=None):
        if not isinstance(d, dict):
            return
        with self._lock:
            now = self._now(now)
            self.topic_seen["/inspection/thermal_alerts"] = now
            unassigned = []
            for h in d.get("hotspots") or []:
                if not isinstance(h, dict):
                    continue
                pos = h.get("position")
                pos = [_r(c, 2) for c in pos[:3]] if isinstance(pos, (list, tuple)) and len(pos) >= 2 else None
                entry = {"id": h.get("id"), "tag": h.get("tag"), "max_temp_c": _r(h.get("max_temp_c"), 1),
                         "mean_temp_c": _r(h.get("mean_temp_c"), 1), "status": h.get("status") or "unknown",
                         "position": pos, "t_sim": _r(self.sim_time, 2), "t_wall": now}
                hid = h.get("id")
                if hid is None:
                    unassigned.append(entry)
                    continue
                hid = str(hid)
                hdef = self._hot_def.get(hid)
                if hdef is not None:
                    entry["true_temp_c"] = _r(float(hdef["temp_k"]) - 273.15, 1)
                    if entry["max_temp_c"] is not None:
                        entry["temp_error_c"] = _r(entry["max_temp_c"] - entry["true_temp_c"], 1)
                    entry["tag"] = entry["tag"] or hdef.get("tag")
                prev = self.thermal.get(hid, {})
                temps = [v for v in (prev.get("peak_c"), entry["max_temp_c"]) if v is not None]
                entry["peak_c"] = max(temps) if temps else None
                self.thermal[hid] = entry
                ch, old = self._changed(f"hot_status_{hid}", entry["status"])
                if ch and entry["status"] in ("warning", "alarm"):
                    self.add_event("alarm" if entry["status"] == "alarm" else "warn", "thermal",
                                   f"{hid} {entry['tag'] or ''}: {entry['max_temp_c']} C ({entry['status']})", now)
            self.thermal_unassigned = unassigned[:10]

    # ------------------------------------------------- inspection evidence
    def set_session(self, d, now=None):
        """/metrics/session: where the metrics session lives and which photos / report exist.
        Returns False (and logs once) for a directory outside ``metrics_root``."""
        if not isinstance(d, dict):
            return False
        sdir = d.get("session_dir")
        if not isinstance(sdir, str) or not sdir or self.metrics_root is None:
            return False
        with self._lock:
            now = self._now(now)
            self.topic_seen["/metrics/session"] = now
            if not os.path.isabs(sdir) or not _within(sdir, self.metrics_root) or not os.path.isdir(sdir):
                ch, _ = self._changed("session_rejected", sdir)
                if ch:
                    self.add_event("warn", "metrics", f"session directory {sdir} ignored (not inside "
                                                      f"{self.metrics_root})", now)
                return False
            photos = {}
            gauges = d.get("gauges") if isinstance(d.get("gauges"), dict) else {}
            for gid, g in gauges.items():
                if not isinstance(g, dict) or not session_rel(g.get("image")):
                    continue
                photos[str(gid)] = {"image": g["image"], "image_full": session_rel(g.get("image_full")),
                                    "stamp": _num(g.get("stamp")), "wall": g.get("wall")}
            report = session_rel(d.get("report"))
            new = {"dir": os.path.realpath(sdir), "name": str(d.get("session") or os.path.basename(sdir)),
                   "photos": photos, "report": report, "report_json": session_rel(d.get("report_json")),
                   "report_final": bool(d.get("report_final")), "report_written": d.get("report_written")}
            old = self.session or {}
            if report and (old.get("report") != report or (new["report_final"] and not old.get("report_final"))):
                self.add_event("info", "metrics", "inspection report " + ("final" if new["report_final"] else
                                                                          "written") + f": {report}", now)
            self.session = new
            return True

    def session_dir(self):
        """Directory served under /session/ (None until a valid /metrics/session arrived)."""
        with self._lock:
            return self.session["dir"] if self.session else None

    def _report_view(self):
        s = self.session
        if not s or not s.get("report"):
            return None
        return {"url": session_url(s["report"]),
                "json_url": session_url(s["report_json"]) if s.get("report_json") else None,
                "final": s.get("report_final"), "written": s.get("report_written"), "session": s["name"]}

    # ------------------------------------------------------ layers/images
    def set_layer(self, name, data, origin_x, origin_y, resolution, now=None, extra=None):
        """Store a 2-D int8 grid (OccupancyGrid convention, row 0 = y origin)."""
        arr = np.asarray(data, dtype=np.int8)
        if arr.ndim != 2:
            raise ValueError("layer data must be 2-D")
        with self._lock:
            now = self._now(now)
            prev = self.layers.get(name, {})
            geom = {"origin": [_r(origin_x, 4), _r(origin_y, 4)], "resolution": _r(resolution, 5),
                    "width": int(arr.shape[1]), "height": int(arr.shape[0])}
            old = self._layer_data.get(name)
            # latched maps are re-published periodically: an identical grid keeps
            # its version so that browsers do not re-download / re-render it
            same = (old is not None and all(prev.get(k) == v for k, v in geom.items())
                    and old.shape == arr.shape and np.array_equal(old, arr))
            ver = int(prev.get("version", 0)) + (0 if same else 1)
            self.layers[name] = dict(geom, version=ver, t_wall=now)
            if extra:
                self.layers[name].update(extra)
            self._layer_data[name] = arr
            topic = {"map": "/map", "gas": "/gas/concentration_grid", "rssi": "/comm/rssi_map"}.get(name)
            if topic:
                self.topic_seen[topic] = now

    def get_layer(self, name):
        with self._lock:
            if name not in self._layer_data:
                return None, None
            return self._layer_data[name], dict(self.layers[name])

    def touch_image(self, name, topic=None, now=None):
        with self._lock:
            now = self._now(now)
            ver = int(self.images.get(name, {}).get("version", 0)) + 1
            self.images[name] = {"version": ver, "t_wall": now}
            if topic:
                self.topic_seen[topic] = now

    def _apply_result(self, g, res):
        """The mission's accepted reading of a gauge replaces later live readings on the tile."""
        if not isinstance(res, dict):
            return
        val = _r(res.get("value"), 4)
        if val is None:
            return
        gdef = self._gauge_def.get(g["id"], {})
        g["value"], g["value_source"] = val, "mission"
        status = res.get("status")
        if status not in ("normal", "warning", "alarm") and self.layout is not None:
            status = self.layout.gauge_status(g["id"], val)
        g["status"] = status or g.get("status")
        if _r(res.get("confidence"), 3) is not None:
            g["confidence"] = _r(res.get("confidence"), 3)
        if res.get("n_readings") is not None:
            g["n_readings"] = res.get("n_readings")
        true = gdef.get("value")
        if true is not None:
            span = float(gdef.get("max", 1.0)) - float(gdef.get("min", 0.0))
            g["error"] = _r(val - float(true), 4)
            g["error_pct_fs"] = _r(abs(val - float(true)) / span * 100.0, 2) if span > 0 else None

    # ----------------------------------------------------------- snapshot
    def topic_health(self, now=None):
        now = self._now(now)
        out = {}
        for topic, period in WATCHED_TOPICS.items():
            seen = self.topic_seen.get(topic)
            if seen is None:
                out[topic] = {"age": None, "status": "never"}
                continue
            age = now - seen
            if period <= 0.0:
                status = "ok"
            else:
                status = "ok" if age <= 2.0 * period + 0.5 else ("late" if age <= 10.0 * period + 2.0 else "stale")
            out[topic] = {"age": round(age, 1), "status": status}
        return out

    def snapshot(self, now=None, max_events=80):
        with self._lock:
            now = self._now(now)

            def age(d):
                t = d.get("t_wall") if isinstance(d, dict) else None
                return round(now - t, 1) if t is not None else None

            loco = {k: _jsonable(self.locomotion.get(k)) for k in LOCO_KEYS if k in self.locomotion}
            loco["age"] = age(self.locomotion)
            gauges = []
            results = self.mission.get("results") if isinstance(self.mission.get("results"), dict) else {}
            photos = (self.session or {}).get("photos") or {}
            for gid in sorted(set(self._gauge_def) | set(self.gauges)):
                g = dict(self.gauges.get(gid) or {"id": gid, "tag": self._gauge_def.get(gid, {}).get("tag", ""),
                                                  "value": None, "status": "unread",
                                                  "unit": self._gauge_def.get(gid, {}).get("unit", ""),
                                                  "true_value": self._gauge_def.get(gid, {}).get("value")})
                g["age"] = age(g)
                g.pop("t_wall", None)
                self._apply_result(g, results.get(gid))
                ph = photos.get(gid)
                if ph:
                    g["image_url"] = session_url(ph["image"])
                    g["image_full_url"] = session_url(ph["image_full"]) if ph.get("image_full") else None
                gauges.append(g)
            thermal = []
            for hid in sorted(set(self._hot_def) | set(self.thermal)):
                h = dict(self.thermal.get(hid) or {"id": hid, "tag": self._hot_def.get(hid, {}).get("tag"),
                                                   "status": "unseen", "max_temp_c": None})
                h["age"] = age(h)
                h.pop("t_wall", None)
                thermal.append(h)
            snap = {
                "t_wall": round(now, 2),
                "uptime": round(now - self.t0_wall, 1),
                "t_sim": _r(self.sim_time, 2),
                "robot": dict(self.robot, age=age(self.robot)) if self.robot else None,
                "locomotion": loco,
                "terrain": {k: _jsonable(self.terrain.get(k)) for k in TERRAIN_KEYS if k in self.terrain},
                "mission": dict(self.mission, age=age({"t_wall": self.topic_seen.get("/mission/state")})),
                "safety": dict(self.safety),
                "battery": dict(self.battery, age=age(self.battery),
                                history=[[round(t - now, 1), p] for t, p in self._batt_hist][-240:]),
                "rssi": dict(self.rssi, age=age(self.rssi)),
                "gas": {
                    "ppm": self.gas.get("ppm"), "age": age(self.gas), "alarm": self.gas.get("alarm"),
                    "alarm_ppm": self.gas_alarm_ppm,
                    "history": [[round(t - now, 1), v] for t, v in self._gas_hist][-480:],
                    "field": self.gas.get("field"), "seek": self.gas.get("seek"),
                    "estimate": self.gas.get("estimate"), "estimate_error_m": _r(self.leak_error(), 2),
                },
                "gauges": gauges,
                "thermal": thermal,
                "thermal_unassigned": list(self.thermal_unassigned),
                "paths": dict(self.paths),
                "plan": self.plan,
                "cmd_vel": dict(self.cmd, source=self.mux, age=age(self.cmd)) if self.cmd else {"source": self.mux},
                "flags": self._flags_view(now),
                "layers": {k: dict(v) for k, v in self.layers.items()},
                "images": {k: dict(v) for k, v in self.images.items()},
                "report": self._report_view(),
                "events": list(self.events)[-max_events:] if max_events > 0 else [],
                "topics": self.topic_health(now),
            }
            return snap
