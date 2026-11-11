"""
Offline demo of the digital-twin dashboard - no ROS, no Gazebo.

    python -m plant_simulation.dashboard_demo [--port 8080] [--speed 4] [--no-autostart] [--fall-at 120,400]

    # headless evaluation run: as fast as possible, metrics session recorded
    # with the real MetricsRecorder (then: python tools/plot_metrics.py <dir>)
    python -m plant_simulation.dashboard_demo --headless --metrics-dir /tmp/dusan_metrics

A kinematic robot is driven by the REAL mission state machine
(plant_simulation.mission_logic) through the REAL dashboard server and
aggregator; the other modules are replaced by simple synthetic models:

* gas: steady Gaussian plume from the layout leak, advected by the layout
  wind (concentration grid + sensor at the robot), and a toy seeker that
  walks up the plume when the mission enables seeking;
* comm: log-distance RSSI map from the layout access points + shadow zones;
* inspection: gauge readings = ground truth + 1 % FS noise while the mission
  dwells at a viewpoint, thermal alerts when a hot spot is in view; when the
  mission accepts a gauge, an inspection photo of the synthetic dial (same
  /inspection/gauge_snapshot message and renderer as gauge_reader_node) goes to
  the metrics session -> gauges/*.jpg, inspection_report.html, tile thumbnails
  and the '보고서 열기' link of the dashboard;
* battery drain / charging on the dock, locomotion state, camera frames;
* slippery ground (display only): the layout plates (C-OBS4 mu 0.2, F-OBS9
  mu 0.15) give the module1 friction-prior fields of /terrain/info and the
  slip-mode fields of /locomotion/state while the robot crosses them
  (:class:`SyntheticSlip`).  The kinematic robot is not slowed down, so a
  headless metrics run is the same with or without it;
* falls (--fall-at T1,T2 simulated s, off by default): the robot falls on its
  side there and rights itself through the module1 fall_recovery phases
  (recovery_state, mode, roll and the fall / recovery counters of
  /locomotion/state, fall / recovered events of /locomotion/fall_event); it
  stands still meanwhile and the mission holds as with the real controller
  (:class:`SyntheticFall`).

Useful to preview the UI on the development PC, in tests, and (headless,
with --metrics-dir) as an end-to-end check of mission -> metrics -> report
figures without ROS.  With the dashboard and without --metrics-dir the session
is recorded in a temporary directory (deleted at exit; --metrics-dir keeps it)
so that the photos and the report can be shown.
"""
import argparse
import math
import os
import random
import shutil
import tempfile
import threading
import time

import cv2
import numpy as np

from dusan_common import gauge_design
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout
from dusan_common.planning import plan as plan_path
from dusan_common.robot_model import SENSORS

from plant_simulation.dashboard_render import (encode_jpeg, placeholder_jpeg, render_gas_png, render_map_png,
                                               render_rssi_png)
from plant_simulation.dashboard_state import DashboardState, friction_zones_from_layout
from plant_simulation.layout_map import rasterize_layout
from plant_simulation.metrics import MetricsRecorder, make_session_dir
from plant_simulation.mission_logic import MissionLogic, fallback_viewpoints
from plant_simulation.web_server import DashboardServer


# synthetic thermal detections: a hot spot is reported inside the thermal camera's
# horizontal FOV (+-hfov/2) up to THERMAL_RANGE_M - the range over which the module2
# thermal pipeline was benchmarked with recall 1.0 (docs/results/thermal_eval.json,
# range_m [1.2, 7.0]); the terrain-cost route passes the C-OBS3 rubble on the north
# side, H05 on the south wall is then 5.6 .. 7 m ahead when it enters the FOV
THERMAL_RANGE_M = 7.0
THERMAL_HALF_FOV = 0.5 * float(SENSORS["thermal_camera"]["hfov"])


def default_web_dir():
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web"))


# synthetic gauge frame: 360 px dial face pasted at [60:420, 140:500] of a 640 x 480 frame
DEMO_DIAL_ROI = (320.0, 240.0, 180.0)


def demo_gauge_snapshot(layout, gid, frame, result, t, frames=1):
    """/inspection/gauge_snapshot message for the synthetic frame of ``gid`` (module2 renderer,
    crop around the known dial position).  ``result`` = the mission's accepted reading.
    None when module2_inspection is not importable."""
    try:
        from module2_inspection.gauge_snapshot import render_gauge_snapshot, snapshot_message, wall_time_text
    except ImportError:
        return None
    g = layout.gauge(gid) or {}
    res = result if isinstance(result, dict) else {}
    info = {"id": gid, "tag": g.get("tag", ""), "value": res.get("value"), "unit": g.get("unit", ""),
            "status": res.get("status") or "unknown", "confidence": res.get("confidence"),
            "n_samples": res.get("n_readings") or 0, "range": [g.get("min"), g.get("max")], "t_sim": t,
            "wall": wall_time_text()}
    imgs = render_gauge_snapshot(frame, None, info, crop_px=320, full_width=640, quality=85, roi=DEMO_DIAL_ROI)
    return snapshot_message(info, imgs, stamp=t, t_end=t, wall=info["wall"], read=res.get("value") is not None,
                            frames=int(frames), readings=int(frames), frame_value=res.get("value"),
                            frame_confidence=res.get("confidence"), robot_pose=None, distance=None,
                            incidence_deg=None, synthetic=True)


# synthetic fall on the side and self-righting: (recovery_state, mode, duration s, |roll| rad at the end)
FALL_PHASES = (("FALL_DETECTED", "FALLEN", 0.1, 1.40), ("DAMP", "FALLEN", 1.2, 1.55),
               ("TUCK", "RECOVERING", 0.8, 1.55), ("SELF_RIGHT", "RECOVERING", 1.6, 0.50),
               ("PREPARE", "RECOVERING", 1.0, 0.05), ("STAND_UP", "RECOVERING", 2.5, 0.02),
               ("VERIFY", "RECOVERING", 0.3, 0.0))


class SyntheticFall:
    """Falls at given simulated times, each followed by a successful self-righting
    (display / mission-hold check only, no physics)."""

    def __init__(self, times=()):
        self.times = sorted(float(t) for t in (times or ()))
        self.t0 = None
        self.fall_count = self.successes = self.attempts = 0

    @property
    def down(self):
        return self.t0 is not None

    def counters(self):
        return {"fall_count": self.fall_count, "recovery_success": self.successes, "assisted_resets": 0,
                "recovery_attempts": self.attempts}

    def step(self, t, x, y):
        """-> (dict of /locomotion/state fields, [fall events]); fields always carry the counters,
        mode / recovery_state / roll only while the robot is down."""
        events = []
        if self.t0 is None and self.times and t >= self.times[0]:
            self.times.pop(0)
            self.t0 = float(t)
            self.fall_count += 1
            events.append({"event": "fall", "x": x, "y": y, "roll": 1.40, "pitch": 0.05})
        fields = {"recovery_state": "NORMAL"}
        if self.t0 is not None:
            dt = t - self.t0
            for name, mode, dur, roll in FALL_PHASES:
                if dt < dur:
                    if name == "TUCK" and self.attempts < self.fall_count:
                        self.attempts += 1                       # an attempt starts after the impact
                    fields = {"recovery_state": name, "mode": mode, "roll": roll, "body_height": 0.12}
                    break
                dt -= dur
            else:
                self.t0 = None
                self.successes += 1
                events.append({"event": "recovered", "x": x, "y": y, "roll": 0.0, "pitch": 0.0})
        fields.update(self.counters())
        return fields, events


class SyntheticPlant:
    """Very small stand-in for Gazebo + modules 1-3."""

    def __init__(self, layout, grid, seed=1):
        self.layout = layout
        self.grid = grid
        self.rng = random.Random(seed)
        sx, sy, sz, syaw = layout.spawn
        self.x, self.y, self.z, self.yaw = sx, sy, sz + 0.22, syaw
        self.vx = self.vy = self.wz = 0.0
        self.battery = 92.0
        self.t = 0.0
        leak = layout.gas_leak
        self.src = np.array(leak.get("source", [0.0, 0.0])[:2], dtype=float)
        w = np.array(leak.get("wind_base", [0.4, -0.1]), dtype=float)
        self.wind = w / (np.linalg.norm(w) or 1.0)
        # gas grid (0.25 m) over the nav-map extent, ppm
        self.gres = 0.25
        x0, y0 = grid.origin_x, grid.origin_y
        gw = int(grid.width * grid.resolution / self.gres)
        gh = int(grid.height * grid.resolution / self.gres)
        self.gorigin = (x0, y0)
        xs = x0 + (np.arange(gw) + 0.5) * self.gres
        ys = y0 + (np.arange(gh) + 0.5) * self.gres
        self.gx, self.gy = np.meshgrid(xs, ys)
        self.gas = self._plume(self.gx, self.gy)
        self.rssi = self._rssi_map()
        self.seek = {"state": "IDLE", "path": [], "t0": None}

    # ------------------------------------------------------------- models
    def _plume(self, x, y):
        dx, dy = x - self.src[0], y - self.src[1]
        down = dx * self.wind[0] + dy * self.wind[1]
        cross = -dx * self.wind[1] + dy * self.wind[0]
        sigma = 0.35 + 0.18 * np.maximum(down, 0.0)
        c = 180.0 * np.exp(-0.5 * (cross / sigma) ** 2) / (1.0 + 0.6 * np.maximum(down, 0.0))
        c = np.where(down > -0.4, c, 180.0 * np.exp(-((dx ** 2 + dy ** 2) / 0.3)))
        return c

    def gas_at(self, x, y):
        return float(self._plume(np.array(x), np.array(y))) + self.rng.gauss(0.0, 0.3)

    def _rssi_map(self):
        comm = self.layout.comm
        res = 0.5
        x0, y0 = self.grid.origin_x, self.grid.origin_y
        w = int(self.grid.width * self.grid.resolution / res)
        h = int(self.grid.height * self.grid.resolution / res)
        xs = x0 + (np.arange(w) + 0.5) * res
        ys = y0 + (np.arange(h) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        best = np.full(X.shape, -120.0)
        n = float(comm.get("path_loss_exponent", 2.2))
        for ap in comm.get("access_points", []):
            d = np.maximum(0.5, np.hypot(X - ap["position"][0], Y - ap["position"][1]))
            p = float(ap.get("p0_dbm", -30.0)) - 10.0 * n * np.log10(d)
            # crude wall term: every 8 m inside the hall costs a wall
            best = np.maximum(best, p - 6.0 * (d // 8.0))
        for z in comm.get("shadow_zones", []):
            r = z["rect"]
            m = (X >= r[0]) & (X <= r[2]) & (Y >= r[1]) & (Y <= r[3])
            best[m] -= float(z.get("attenuation_db", 30.0))
        self.rssi_meta = (x0, y0, res)
        return best

    def rssi_at(self, x, y):
        x0, y0, res = self.rssi_meta
        ix = int(np.clip((x - x0) / res, 0, self.rssi.shape[1] - 1))
        iy = int(np.clip((y - y0) / res, 0, self.rssi.shape[0] - 1))
        return float(self.rssi[iy, ix]) + self.rng.gauss(0.0, 0.8)

    def step(self, cmd, dt):
        vx, vy, wz = cmd
        a = 1.0 - math.exp(-dt / 0.25)            # first-order velocity response of the gait
        self.vx += a * (vx - self.vx)
        self.vy += a * (vy - self.vy)
        self.wz += a * (wz - self.wz)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        nx = self.x + (self.vx * c - self.vy * s) * dt
        ny = self.y + (self.vx * s + self.vy * c) * dt
        if self.grid.is_free(nx, ny, 0.15):
            self.x, self.y = nx, ny
        self.yaw = (self.yaw + self.wz * dt + math.pi) % (2 * math.pi) - math.pi
        self.z = self.layout.floor_z(self.x, self.y) + 0.34
        speed = math.hypot(self.vx, self.vy)
        dock = self.layout.dock_pose
        on_dock = math.hypot(self.x - dock[0], self.y - dock[1]) < 0.4 and speed < 0.05
        self.battery = min(100.0, self.battery + 0.4 * dt) if on_dock else max(0.0, self.battery - (0.01 + 0.05 * speed) * dt)
        self.t += dt
        return on_dock


def _quads_overlap(a, b):
    """Separating-axis test of two convex quadrilaterals ((4, 2) corner arrays in order)."""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            if not np.any(n):
                continue
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() - 1e-9 or pb.max() < pa.min() - 1e-9:
                return False
    return True


class SyntheticSlip:
    """Display-only stand-in for the module1 friction prior + slip detector
    (module1 README section 3.8; same rules, synthetic values):

    * terrain: the low-friction zones (mu < ``slippery_mu``) overlapping the
      robot footprint (``rear`` behind) plus the look-ahead corridor (``far``
      ahead, ``half_width``) -> class ``slippery`` with ``friction_mu`` /
      ``friction_zones`` / ``friction_distance``; held ``hold`` s after the
      overlap ends (terrain_classifier);
    * locomotion: slip mode follows the prior; the first entry of the robot
      centre into a zone with mu <= ``detect_mu`` also raises one synthetic
      detection (feet FL FR sliding for ``slide_time``), released after
      ``release_time`` s without votes AND ``release_distance`` m of travel
      (slip_detector); ``mu_est`` drops to zone mu + 0.02 on a detection and
      recovers with ``recover_tau`` to ``mpc_mu`` once the detection is
      released; MPC mu = clip(min(mu_est, prior), ``mu_min``, ``mpc_mu``);
      torque feed-forward ramps over ``ff_ramp`` s.

    Deterministic (no random numbers), it never changes the robot motion."""

    def __init__(self, zones, far=1.0, rear=0.35, half_width=0.30, slippery_mu=0.35, hold=1.0, detect_mu=0.15,
                 slide_time=0.4, slide_speed=0.12, release_time=2.0, release_distance=3.0, mu_min=0.15,
                 mpc_mu=0.5, recover_tau=8.0, ff_ramp=0.5):
        self.zones = [z for z in zones if z["mu"] < slippery_mu]
        self.far, self.rear, self.half_width, self.hold = far, rear, half_width, hold
        self.detect_mu, self.slide_time, self.slide_speed = detect_mu, slide_time, slide_speed
        self.release_time, self.release_distance = release_time, release_distance
        self.mu_min, self.mpc_mu, self.recover_tau, self.ff_ramp = mu_min, mpc_mu, recover_tau, ff_ramp
        self.mu_est = mpc_mu
        self.torque_ff = 0.0
        self.slip_events = 0
        self._prior = None            # (mu, ids, distance) of the last overlap
        self._prior_t = None          # time of the last overlap (hold)
        self._inside = set()          # zones containing the robot centre
        self._detected = False        # detection latch (until the release rule)
        self._slide_t0 = None         # start of the synthetic foot slide
        self._last_vote_t = None
        self._vote_travel = 0.0       # travelled distance at the last vote
        self._travel = 0.0
        self._last_xy = None

    def _query(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([[-self.rear, -self.half_width], [self.far, -self.half_width],
                          [self.far, self.half_width], [-self.rear, self.half_width]])
        quad = np.stack([x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]], axis=1)
        mu, ids, dist = None, [], None
        for z in self.zones:
            x0, y0, x1, y1 = z["rect"]
            zc = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
            if not _quads_overlap(quad, zc):
                continue
            ids.append(z["id"])
            mu = z["mu"] if mu is None else min(mu, z["mu"])
            ahead = max(0.0, float(((zc[:, 0] - x) * c + (zc[:, 1] - y) * s).min()))
            dist = ahead if dist is None else min(dist, ahead)
        return mu, ids, dist

    def step(self, x, y, yaw, t, dt):
        """-> (/terrain/info dict, dict of slip fields for /locomotion/state)."""
        if self._last_xy is not None:
            self._travel += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)
        mu, ids, dist = self._query(x, y, yaw)
        if ids:
            self._prior, self._prior_t = (mu, ids, dist), t
        elif self._prior_t is not None and t - self._prior_t > self.hold:
            self._prior, self._prior_t = None, None
        slippery = self._prior is not None
        terrain = {"class": "slippery" if slippery else "flat", "recommended_gait": "trot", "roughness": 0.01,
                   "slippery": slippery, "friction_mu": self._prior[0] if slippery else None,
                   "friction_zones": list(self._prior[1]) if slippery else [],
                   "friction_distance": round(self._prior[2], 3) if slippery else None}
        # synthetic detection: robot centre enters a very slippery zone
        inside = {z["id"] for z in self.zones
                  if z["rect"][0] <= x <= z["rect"][2] and z["rect"][1] <= y <= z["rect"][3]}
        for z in self.zones:
            if z["id"] in inside and z["id"] not in self._inside and z["mu"] <= self.detect_mu + 1e-9:
                self.slip_events += 1
                self._detected = True
                self._slide_t0 = t
                self.mu_est = max(self.mu_min, min(self.mu_est, z["mu"] + 0.02))
        self._inside = inside
        sliding = self._slide_t0 is not None and t - self._slide_t0 <= self.slide_time
        if sliding:                                       # votes keep coming while the feet slide
            self._last_vote_t, self._vote_travel = t, self._travel
        if (self._detected and not sliding and t - self._last_vote_t >= self.release_time
                and self._travel - self._vote_travel >= self.release_distance):
            self._detected = False
        if not self._detected:
            self.mu_est += (self.mpc_mu - self.mu_est) * (1.0 - math.exp(-dt / self.recover_tau))
        slip_mode = slippery or self._detected
        source = "both" if slippery and self._detected else "terrain" if slippery else \
            "detected" if self._detected else "none"
        self.torque_ff = min(1.0, max(0.0, self.torque_ff + (dt if slip_mode else -dt) / self.ff_ramp))
        prior = self._prior[0] if slippery else None
        mpc_mu = min(self.mpc_mu, max(self.mu_min, min(self.mu_est, prior if prior is not None else math.inf)))
        loco = {"slip": bool(sliding), "slip_mode": bool(slip_mode), "slip_source": source,
                "slip_speed": self.slide_speed if sliding else 0.0,
                "slip_legs": [bool(sliding), bool(sliding), False, False], "slip_events": self.slip_events,
                "mu_est": round(self.mu_est, 3), "mpc_mu": round(mpc_mu, 3), "friction_prior": prior,
                "torque_ff": round(self.torque_ff, 3)}
        return terrain, loco


def run_demo(port=8080, speed=4.0, autostart=True, host="0.0.0.0", duration=None, quiet=False,
             serve=True, realtime=True, sim_duration=None, metrics_dir=None, stop_when_done=False, fall_at=()):
    """Run the synthetic plant + real mission (+ dashboard server).

    serve          start the HTTP dashboard (False: headless)
    realtime       pace the loop to wall time x speed (False: as fast as possible)
    duration       stop after this many WALL seconds
    sim_duration   stop after this many SIMULATED seconds
    metrics_dir    record a metrics session (MetricsRecorder) below this directory
    stop_when_done stop once a started mission has finished (docked / done)
    fall_at        simulated times [s] of synthetic falls + self-righting (SyntheticFall)
    Returns (DashboardState, MissionLogic); the metrics session directory (or
    None) is left in ``run_demo.session_dir``.
    """
    layout = Layout.load()
    grid = rasterize_layout(layout)
    try:   # prefer the generated map when it exists
        if os.path.exists(layout.nav_map_yaml()):
            grid = OccupancyGridMap.load(layout.nav_map_yaml())
    except Exception:
        pass
    plant = SyntheticPlant(layout, grid)
    temp_metrics = None
    if serve and not metrics_dir:           # photos + report need a session directory
        metrics_dir = temp_metrics = tempfile.mkdtemp(prefix="dusan_demo_metrics_")
    state = DashboardState(layout, metrics_root=metrics_dir)
    slip = SyntheticSlip(friction_zones_from_layout(layout))
    falls = SyntheticFall(fall_at)
    mission = MissionLogic({"autostart": bool(autostart), "autostart_delay": 3.0, "rth_ack_timeout": 3.0}, layout)
    mission.set_map(grid)
    # patrol plan + route like module2 would publish
    vps = fallback_viewpoints(layout, grid, layout.spawn[:2])
    plan = {"start": list(layout.spawn[:2]),
            "viewpoints": [{"gauge_id": v.gauge_id, "x": v.x, "y": v.y, "yaw": v.yaw, "score": 0.8} for v in vps],
            "order": [v.gauge_id for v in vps], "legs": []}
    route, prev = [], layout.spawn[:2]
    terrain = mission.terrain_cost(grid)        # same terrain cost layer as the mission planner
    for v in vps:
        leg = plan_path(grid, prev, (v.x, v.y), robot_radius=0.35, extra_cost=terrain) or [prev, (v.x, v.y)]
        plan["legs"].append([list(p) for p in leg])
        route += leg
        prev = (v.x, v.y)
    plan["total_length"] = float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(route[:-1], route[1:])))
    mission.set_plan(plan)
    state.set_plan(plan)
    state.set_path("route", route)
    state.set_layer("map", grid.occ, grid.origin_x, grid.origin_y, grid.resolution)
    rssi_occ = np.clip(np.round((plant.rssi + 100.0) / 70.0 * 100.0), 0, 100).astype(np.int8)
    state.set_layer("rssi", rssi_occ, plant.rssi_meta[0], plant.rssi_meta[1], plant.rssi_meta[2])
    vis_max = 100.0
    state.set_gas_field({"max_ppm": float(plant.gas.max()), "vis_max_ppm": vis_max, "scale": "linear",
                         "source": plant.src.tolist(), "wind": plant.wind.tolist()})

    lock = threading.Lock()
    frames = {"gauge": None, "thermal": None}
    blobs_cache = {}
    stop = threading.Event()

    def command(target, cmd):
        with lock:
            if target == "mission":
                mission.command(cmd, plant.t)
            elif target == "battery":
                plant.battery = float(cmd)
            elif target == "locomotion":
                state.add_event("info", "demo", f"locomotion '{cmd}' (not simulated in the demo)")
        state.add_event("info", "operator", f"{target} command '{cmd}'")
        return True, f"{target} '{cmd}' sent"

    def blob(name):
        if name == "map.png":
            arr, meta = state.get_layer("map")
            key = ("map", meta["version"])
            if key not in blobs_cache:
                blobs_cache[key] = render_map_png(arr)
            return blobs_cache[key], "image/png"
        if name == "rssi.png":
            arr, meta = state.get_layer("rssi")
            return render_rssi_png(arr, layout.comm.get("rssi_min_dbm", -80.0)), "image/png"
        if name == "gas.png":
            arr, meta = state.get_layer("gas")
            if arr is None:
                return None
            return render_gas_png(arr), "image/png"
        if name.startswith("image/"):
            key = name.split("/")[1].split(".")[0]
            with lock:
                f = frames.get(key)
            if f is None:
                return placeholder_jpeg("no signal"), "image/jpeg"
            return encode_jpeg(f, 80, 640), "image/jpeg"
        return None

    server = None
    if serve:
        server = DashboardServer(state.snapshot, lambda: state.static, default_web_dir(), port=port, host=host,
                                 sse_rate=5.0, command_fn=command, blob_fn=blob, port_fallbacks=5,
                                 session_dir_fn=state.session_dir)
        bound = server.start()
        if not quiet:
            print(f"dashboard demo on http://localhost:{bound}  (speed x{speed}, Ctrl-C to quit)")
    rec = None
    run_demo.session_dir = None
    if metrics_dir:
        run_demo.session_dir = make_session_dir(metrics_dir)
        rec = MetricsRecorder(run_demo.session_dir, layout)
        rec.on_change = lambda: state.set_session(rec.session_info())      # like /metrics/session
        rec.event(0.0, "demo", "start", "offline dashboard demo (synthetic plant, real mission logic)")
        state.set_session(rec.session_info())
        if not quiet:
            print(f"recording metrics to {run_demo.session_dir}"
                  + (" (temporary, deleted at exit; --metrics-dir keeps it)" if temp_metrics else ""))
    started = False

    dt = 0.1
    t_wall0 = time.time()
    last_gas_layer = -1e9
    rng = random.Random(7)
    seek_path = []
    insp = None             # current inspection visit {gid, frame, n} -> photo when the mission accepts it
    try:
        while not stop.is_set():
            t0 = time.time()
            for _ in range(max(1, int(round(speed)))):
                with lock:
                    t = plant.t
                    mission.set_pose(plant.x, plant.y, plant.yaw, t)
                    fall_fields, fall_events = falls.step(t, plant.x, plant.y)
                    if falls.down:                  # the mission holds like with the real controller
                        mission.on_locomotion_state(fall_fields, t)
                    elif fall_events or mission.loco_mode in ("FALLEN", "RECOVERING"):
                        mission.on_locomotion_state({"mode": "STAND", "recovery_state": "NORMAL"}, t)
                    for ev in fall_events:
                        state.set_fall_event(ev)
                        if rec:
                            rec.on_fall_event(t, ev)
                    ppm = plant.gas_at(plant.x, plant.y)
                    mission.on_gas(ppm, t)
                    # toy seeker
                    seeking = mission.base == "SEEK" and not mission.paused and not mission.teleop_active
                    cmd = None
                    if seeking:
                        d = plant.src - np.array([plant.x, plant.y])
                        dist = float(np.linalg.norm(d))
                        if plant.seek["state"] in ("IDLE", "FOUND", "FAILED"):
                            plant.seek = {"state": "SURGE", "t0": t}
                            seek_path = []
                        if dist < 0.6:
                            est = (plant.src + np.array([rng.gauss(0, 0.2), rng.gauss(0, 0.2)])).tolist()
                            plant.seek.update(state="FOUND", estimate=est)
                            cmd = (0.0, 0.0, 0.0)
                        else:
                            plant.seek["state"] = "GRADIENT" if dist < 4.0 else "SURGE"
                            hd = math.atan2(d[1], d[0])
                            err = (hd - plant.yaw + math.pi) % (2 * math.pi) - math.pi
                            cmd = (0.3 * max(0.0, math.cos(err)), 0.0, max(-0.6, min(0.6, 1.5 * err)))
                        seek_path.append((plant.x, plant.y))
                        st = dict(plant.seek, concentration=ppm, elapsed=t - plant.seek["t0"],
                                  path_length=float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                                                        for a, b in zip(seek_path[:-1], seek_path[1:]))))
                        est = st.get("estimate")
                        st["error_m"] = (math.hypot(est[0] - plant.src[0], est[1] - plant.src[1]) if est else None)
                        mission.on_seek_status(st, t)
                        state.set_seek_status(st)
                        if rec:
                            rec.on_seek_status(t, st)
                        state.set_path("seek", seek_path)
                    elif plant.seek["state"] != "IDLE" and mission.base != "SEEK":
                        plant.seek["state"] = "IDLE"
                        state.set_seek_status(dict(plant.seek))
                        if rec:
                            rec.on_seek_status(t, dict(plant.seek))
                    # simulated gauge reader
                    if mission.base == "INSPECTING" and mission.current is not None:
                        g = layout.gauge(mission.current.gauge_id)
                        span = g["max"] - g["min"]
                        val = g["value"] + rng.gauss(0.0, 0.01 * span)
                        rd = {"stamp": t, "readings": [{"id": g["id"], "tag": g["tag"], "value": val, "unit": g["unit"],
                                                        "confidence": 0.9, "status": layout.gauge_status(g["id"], val),
                                                        "n_samples": 5, "robot_pose": [plant.x, plant.y, plant.yaw]}]}
                        mission.on_gauge_readings(rd, t)
                        state.set_gauge_readings(rd)
                        if rec:
                            rec.on_gauge_readings(t, rd)
                        face = gauge_design.render_gauge_face(val, g["min"], g["max"], g["unit"], g["tag"], size=360)
                        frame = np.full((480, 640, 3), 30, np.uint8)
                        frame[60:420, 140:500] = face
                        cv2.circle(frame, (320, 240), 186, (0, 200, 0), 2)
                        cv2.putText(frame, f"{g['id']} {val:.2f} {g['unit']}", (150, 455), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        frames["gauge"] = frame
                        if rec is not None:
                            n = insp["n"] + 1 if (insp and insp["gid"] == g["id"]) else 1
                            insp = {"gid": g["id"], "frame": frame, "n": n}
                    # thermal alerts for hot spots in view
                    alerts = []
                    for h in layout.hotspots:
                        dx, dy = h["position"][0] - plant.x, h["position"][1] - plant.y
                        dist = math.hypot(dx, dy)
                        bearing = (math.atan2(dy, dx) - plant.yaw + math.pi) % (2 * math.pi) - math.pi
                        if dist < THERMAL_RANGE_M and abs(bearing) < THERMAL_HALF_FOV:
                            tc = h["temp_k"] - 273.15 + rng.gauss(0, 1.0)
                            if h["temp_k"] >= layout.thermal["warning_k"] - 5:
                                alerts.append({"id": h["id"], "tag": h["tag"], "max_temp_c": tc, "mean_temp_c": tc - 6,
                                               "position": h["position"], "status": layout.temperature_status(tc + 273.15),
                                               "bbox": [0, 0, 10, 10]})
                    if alerts:
                        state.set_thermal_alerts({"stamp": t, "hotspots": alerts})
                        if rec:
                            rec.on_thermal(t, {"stamp": t, "hotspots": alerts})
                        th = np.zeros((240, 320), np.float32)
                        cv2.circle(th, (160, 120), 30, 1.0, -1)
                        th = cv2.GaussianBlur(th, (0, 0), 12)
                        frames["thermal"] = cv2.applyColorMap((th / th.max() * 255).astype(np.uint8),
                                                              cv2.COLORMAP_INFERNO)
                    out = mission.step(t)
                    if insp is not None and not (mission.base == "INSPECTING" and mission.current is not None
                                                 and mission.current.gauge_id == insp["gid"]):
                        # the mission left the gauge (accepted / skipped): one inspection photo
                        snap = demo_gauge_snapshot(layout, insp["gid"], insp["frame"],
                                                   mission.results.get(insp["gid"]), t, insp["n"])
                        if snap is not None:
                            rec.on_gauge_snapshot(t, snap)
                        insp = None
                    if falls.down:
                        drive = (0.0, 0.0, 0.0)
                    elif seeking and cmd is not None:
                        drive = cmd
                    elif out.cmd_vel is not None:
                        drive = out.cmd_vel
                    elif mission.base == "RTH" and mission.phase == "yield":
                        drive = (0.0, 0.0, 0.0)
                    else:
                        drive = (0.0, 0.0, 0.0)
                    if out.path is not None:
                        state.set_path("nav", out.path)
                    on_dock = plant.step(drive, dt)
                    if mission.base == "RTH" and on_dock:
                        mission.on_safety_status({"state": "DOCKED", "docked": True, "rth_active": False}, t)
                    # the aggregator derives its event log from /mission/state, like with ROS
                    mstate = mission.state_dict(t)
                    state.set_mission(mstate)
                    started = started or mission.mode is not None
                    rssi = plant.rssi_at(plant.x, plant.y)
                    state.set_robot(plant.x, plant.y, plant.z, fall_fields.get("roll", rng.gauss(0, 0.01)),
                                    rng.gauss(0, 0.01), plant.yaw, plant.vx, plant.vy, plant.wz, t_sim=t)
                    moving = math.hypot(plant.vx, plant.vy) > 0.03 or abs(plant.wz) > 0.05
                    loco = {"mode": "WALK" if moving else "STAND", "gait": "trot", "gait_auto": True,
                            "body_height": 0.34 + rng.gauss(0, 0.004), "roll": rng.gauss(0, 0.01),
                            "pitch": rng.gauss(0, 0.01), "fall_count": 0, "recovery_success": 0,
                            "mpc_solve_ms": max(3.0, rng.gauss(12.0, 2.5)), "mpc_status": "solved",
                            "stability_margin": 0.08 + rng.gauss(0, 0.005)}
                    terrain, slip_fields = slip.step(plant.x, plant.y, plant.yaw, t, dt)   # no rng: metrics unchanged
                    loco.update(slip_fields)
                    loco.update(fall_fields)
                    state.set_locomotion(loco)
                    state.set_terrain(terrain)
                    state.set_gas_ppm(ppm)
                    state.set_battery(plant.battery, 25.2, -1.5, on_dock)
                    state.set_rssi(rssi)
                    safety = {"state": "DOCKED" if (on_dock and mission.base == "DOCKED") else "MONITOR",
                              "rth_active": False, "reason": None, "battery_pct": plant.battery / 100.0,
                              "rssi_dbm": rssi, "predicted_min_rssi": rssi - 4.0,
                              "distance_home": math.hypot(plant.x - layout.dock_pose[0],
                                                          plant.y - layout.dock_pose[1]),
                              "energy_to_home_pct": 0.05 * math.hypot(plant.x - layout.dock_pose[0],
                                                                      plant.y - layout.dock_pose[1]),
                              "docked": bool(on_dock)}
                    state.set_safety(safety)
                    if rec:
                        rec.on_pose(t, plant.x, plant.y, plant.z, loco["roll"], loco["pitch"], plant.yaw,
                                    plant.vx, plant.vy, plant.wz)
                        rec.on_locomotion(t, loco)
                        rec.on_mission(t, mstate)
                        rec.on_gas(t, ppm)
                        rec.on_battery(t, plant.battery, 25.2, on_dock)
                        rec.on_rssi(t, rssi)
                        rec.on_safety(t, safety)
                    state.set_cmd_vel(*drive)
                    state.set_mux("seek" if seeking else ("patrol" if out.cmd_vel is not None else "none"))
                    for topic in ("/inspection/gauge_overlay", "/inspection/thermal_fusion", "/patrol/plan_json"):
                        state.touch(topic)
                    if frames["gauge"] is not None:
                        state.touch_image("gauge")
                    if frames["thermal"] is not None:
                        state.touch_image("thermal")
                    if t - last_gas_layer >= 0.5:
                        last_gas_layer = t
                        g = np.clip(np.round(plant.gas / vis_max * 100.0), 0, 100).astype(np.int8)
                        state.set_layer("gas", g, plant.gorigin[0], plant.gorigin[1], plant.gres)
            if duration is not None and time.time() - t_wall0 > duration:
                break
            if sim_duration is not None and plant.t >= sim_duration:
                break
            if stop_when_done and started and mission.mode is None and mission.base in ("DOCKED", "DONE", "IDLE"):
                break
            if realtime:
                time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.stop()
        if rec is not None:
            rec.event(plant.t, "demo", "stop", mission.base)
            rec.close()
            if temp_metrics:
                shutil.rmtree(temp_metrics, ignore_errors=True)
            elif not quiet:
                print(f"metrics written to {run_demo.session_dir}")
    return state, mission


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--speed", type=float, default=4.0, help="simulation steps per wall step (time lapse)")
    ap.add_argument("--no-autostart", action="store_true", help="wait for 'Start mission' in the UI")
    ap.add_argument("--duration", type=float, default=None, help="stop after N wall seconds")
    ap.add_argument("--headless", action="store_true",
                    help="no HTTP server, run as fast as possible until the mission is docked")
    ap.add_argument("--sim-duration", type=float, default=None, help="stop after N simulated seconds")
    ap.add_argument("--metrics-dir", default=None,
                    help="record a metrics session there (plot it with tools/plot_metrics.py)")
    ap.add_argument("--fall-at", default="",
                    help="simulated times [s] of synthetic falls + self-righting, e.g. 120,400")
    a = ap.parse_args()
    try:
        fall_at = [float(v) for v in a.fall_at.replace(",", " ").split()]
    except ValueError:
        ap.error("--fall-at needs numbers, e.g. 120,400")
    sim_duration = a.sim_duration if a.sim_duration is not None else (1800.0 if a.headless else None)
    run_demo(a.port, a.speed, not a.no_autostart, a.host, a.duration, serve=not a.headless,
             realtime=not a.headless, sim_duration=sim_duration, metrics_dir=a.metrics_dir,
             stop_when_done=a.headless, fall_at=fall_at)


if __name__ == "__main__":
    main()
