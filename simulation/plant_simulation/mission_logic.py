"""
ROS-free mission state machine (DESIGN_SPEC section 5, Integration).

States reported on /mission/state:

    IDLE        nothing to do (initial / after abort)
    PATROL      driving to the next viewpoint (phases: starting, navigate,
                backup, align, hold)
    INSPECTING  at a viewpoint, /inspection/target published, dwelling until
                >= min_readings readings of that gauge or dwell_time
                (phases: dwell, scan_left, scan_right)
    SEEK        gas alarm (> gas_alarm_ppm for gas_alarm_hold s): seeker
                enabled (/gas/seek/enable), waiting for FOUND/FAILED/timeout
    RTH         yielding to return_to_home (phases: requested, yield) or, if
                that node never acknowledges, driving home itself (self;
                self_wait while there is no fresh odometry)
    DOCKED      at the charging dock
    MANUAL      overlay while /teleop/active is true (expires when the 2 Hz
                heartbeat stops for teleop_active_timeout s, e.g. a killed teleop)
    PAUSED      overlay after 'pause' until 'resume'
    DONE        a 'patrol' (patrol-only) mission finished

Design: the mission keeps a *base* activity (IDLE, PATROL, INSPECTING, SEEK,
RTH, DOCKED, DONE) and two *overlays* (MANUAL, PAUSED) that freeze it.  All
phase timers advance only while the base activity is active, so a pause or a
manual override in the middle of a dwell resumes cleanly.

The class is driven by the ROS node (mission_manager_node.py) but has no ROS
dependency: inputs are plain Python values, outputs are collected in a
:class:`MissionOutput` returned by :meth:`MissionLogic.step`.

Novel (unmapped) obstacles: elevation-map frames (:meth:`on_obstacle_frame`)
feed dusan_common.obstacle_layer.ObstacleAvoider.  While driving (patrol and
self-RTH) the path follower slows down / stops for evidence ahead, detours
left or right around a confirmed obstacle and rejoins the route, re-plans on
the static + detected map when no detour exists, and otherwise keeps the map
route so that the stuck / back-up / skip logic decides (``obstacle_*``
parameters; status in /mission/state ``obstacle``).  The node converts /
fuses every frame while the mission drives and one per
``obstacle_idle_period`` otherwise (:meth:`obstacle_frame_due`).  A hold
caused only by a stale pose (a long tick on a busy CPU lets the overdue timer
run before the queued /odom) resumes the current path without re-planning.
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np

from dusan_common.debris_crossing import CROSSING_DEFAULTS, CrossingGovernor, CrossingParams, straighten_crossings
from dusan_common.geometry import wrap_angle
from dusan_common.obstacle_layer import OBSTACLE_DEFAULTS, ObstacleAvoider
from dusan_common.planning import densify, plan as plan_path
from dusan_common.pure_pursuit import PurePursuit
from dusan_common.terrain_cost import TERRAIN_COST_DEFAULTS, TerrainCostCache

STATES = ("IDLE", "PATROL", "INSPECTING", "SEEK", "RTH", "DOCKED", "MANUAL", "PAUSED", "DONE")
MISSION_COMMANDS = ("start", "patrol", "seek", "rth", "pause", "resume", "abort", "dock")
SEEK_TERMINAL = ("FOUND", "FAILED")
LOCO_READY = ("WALK", "STAND")
SAFETY_DOCKED_STATES = ("DOCKED", "CHARGING")
COMMAND_HISTORY = 5             # /mission/state recent_commands entries

MISSION_DEFAULTS = {
    # node / io
    "control_rate": 10.0,          # Hz state machine + /cmd_vel/patrol
    "gazebo_mission_keys": True,   # t/p/g/h/o in the Gazebo window while no keyboard_teleop runs
    "state_publish_period": 0.5,   # s, /mission/state republish period (also on change)
    "layout_file": "",
    "map_yaml": "",                # only used when /map never arrives
    # navigation
    "robot_radius": 0.35,          # m, A* inflation
    "max_vx": 0.40,
    "max_wz": 0.60,
    "lookahead": 0.60,
    "goal_tolerance": 0.25,
    "slow_down_dist": 0.8,
    "rotate_in_place_angle": 0.9,
    "k_heading": 1.6,
    "use_plan_legs": True,         # follow the patrol planner's pre-computed legs when starting on them
    "leg_start_tolerance": 1.0,    # m
    "replan_deviation": 1.2,       # m cross-track error -> re-plan
    "replan_min_interval": 3.0,    # s
    "pose_timeout": 1.0,           # s without /odom -> hold
    "follow_spacing": 0.10,        # m vertex spacing of the followed path
    # terrain cost of the planned paths (dusan_common.terrain_cost: pipes / rubble high,
    # plates / platform small, margin <= 0.25 m): terrain_cost_enabled, terrain_w_pipe,
    # terrain_w_rubble, terrain_w_plate, terrain_w_platform, terrain_margin, terrain_margin_fraction
    **TERRAIN_COST_DEFAULTS,
    # novel (unmapped) obstacles from the elevation map (dusan_common.obstacle_layer):
    # slow down / stop -> left/right detour rejoining the route -> re-plan on static + dynamic
    # -> keep the map route (stuck logic below); obstacle_enabled false = legacy behaviour
    **OBSTACLE_DEFAULTS,
    # lying pipes (dusan_common/debris_crossing.py): crossings planned straight and square to the
    # pipe, walked over at a steady pace without turning / strafing; slow next to a pipe end
    **CROSSING_DEFAULTS,
    # stuck detection
    "stuck_time": 8.0,             # s without progress while driving
    "stuck_distance": 0.15,        # m that counts as progress
    "stuck_yaw": 0.35,             # rad that counts as progress (rotating in place)
    "max_replans": 4,              # per viewpoint, then the viewpoint is skipped
    "backup_time": 1.5,            # s reversing before a stuck re-plan
    "backup_speed": 0.15,          # m/s
    # viewpoint alignment / inspection
    "align_yaw_tolerance": 0.10,   # rad
    "align_timeout": 8.0,          # s
    "align_k": 1.5,
    "align_min_wz": 0.12,          # rad/s, overcomes gait dead band
    "dwell_time": 4.0,             # s
    "min_dwell_time": 1.0,         # s, minimum before leaving early on readings
    "min_readings": 5,
    "min_confidence": 0.0,
    "scan_on_miss": True,          # no reading after dwell -> look left/right once
    "scan_yaw_offset": 0.25,       # rad
    "scan_time": 3.0,              # s per side
    # start-up
    "start_timeout": 6.0,          # s waiting for the robot to stand
    "min_start_time": 1.0,
    "send_walk_after_stand": False,  # module1 auto-walks on non-zero /cmd_vel; 'walk' = trot in place
    "restand_period": 5.0,         # s between repeated 'stand' while SIT during a mission
    "recover_after": 20.0,         # s FALLEN without getting up during a mission -> 'recover' (repeated), 0 = off
    # gas
    "gas_alarm_ppm": 25.0,
    "gas_alarm_hold": 1.0,         # s above threshold before the alarm fires
    "gas_timeout": 2.0,            # s, gas reading considered stale after this
    "gas_rearm_ppm": 10.0,         # after FAILED/timeout: re-arm when below this ...
    "gas_rearm_cooldown": 60.0,    # ... and after this many seconds
    "seek_timeout": 180.0,
    "seek_start_timeout": 10.0,    # seeker never becomes active -> FAILED
    # RTH / dock
    "rth_ack_timeout": 8.0,        # s without rth_active after our own 'rth' -> self RTH
    "rth_release_grace": 2.0,      # s after rth_active drops before deciding docked/idle
    "teleop_active_timeout": 1.5,  # s without a /teleop/active message -> manual override released (0 = latched)
    "safety_timeout": 3.0,         # s, /safety/status older than this is not trusted (docked flag)
    "self_rth_fallback": True,
    "dock_tolerance": 0.35,
    "auto_resume_after_rth": False,
    "undock_distance": 1.5,        # m from the dock -> DOCKED becomes IDLE
    "dock_detect_radius": 1.0,     # m: RTH released this close to the dock counts as docked
    "release_rth_on_start": True,  # 'start'/'patrol' on the dock: send 'resume' to release return_to_home ...
    "release_min_battery_pct": 50.0,  # ... if the battery is at least this charged
    # plan fallback / autostart
    "fallback_viewpoints": True,   # build viewpoints from the layout if no patrol plan
    "fallback_standoff": 1.4,      # m in front of the dial
    "autostart": False,
    "autostart_delay": 15.0,       # s after the node starts (sim time)
    "plan_wait_timeout": 20.0,     # s autostart waits for /patrol/plan_json before falling back
    "start_plan_wait": 30.0,       # s a manual start / patrol without a patrol plan yet waits for it (0 = off)
}


# ----------------------------------------------------------------------------
# data classes
# ----------------------------------------------------------------------------
@dataclass
class Viewpoint:
    gauge_id: str
    x: float
    y: float
    yaw: float
    leg: list = None          # optional pre-planned path [(x, y), ...] ending here

    def as_list(self):
        return [round(self.x, 3), round(self.y, 3), round(self.yaw, 4)]


@dataclass
class MissionOutput:
    """Everything the node has to publish after one step."""
    cmd_vel: tuple = None           # (vx, vy, wz) for /cmd_vel/patrol, None = do not publish
    path: list = None               # new /nav/plan ([] = clear), None = unchanged
    target: str = None              # new /inspection/target ("" = none), None = unchanged
    seek_enable: bool = None        # new /gas/seek/enable, None = unchanged
    locomotion: list = field(default_factory=list)   # /locomotion/command strings
    mission: list = field(default_factory=list)      # /mission/command strings (e.g. 'rth')
    events: list = field(default_factory=list)       # (level, text)
    state_changed: bool = False


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _pt(p):
    """[x, y, ...] | (x, y) | {x:, y:} -> (x, y) floats or None."""
    try:
        if isinstance(p, dict):
            return float(p["x"]), float(p["y"])
        return float(p[0]), float(p[1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _seq(v):
    """v when it is a list / tuple, else [] (a malformed plan must not raise)."""
    return v if isinstance(v, (list, tuple)) else []


def parse_patrol_plan(d, leg_match_radius=0.6):
    """/patrol/plan_json dict -> list of Viewpoint in visiting order."""
    if not isinstance(d, dict):
        return []
    vps = {}
    raw_order = []
    for v in _seq(d.get("viewpoints")):
        try:
            gid = str(v["gauge_id"])
            vp = Viewpoint(gid, float(v["x"]), float(v["y"]), float(v.get("yaw", 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(c) for c in (vp.x, vp.y, vp.yaw)):
            continue
        vps[gid] = vp
        raw_order.append(gid)
    order = [str(g) for g in _seq(d.get("order")) if str(g) in vps]
    order += [g for g in raw_order if g not in order]          # anything the order list forgot
    # associate pre-planned legs with the viewpoint they end at
    for leg in _seq(d.get("legs")):
        pts = [q for q in (_pt(p) for p in _seq(leg)) if q is not None]
        if len(pts) < 2:
            continue
        end = pts[-1]
        best, best_d = None, leg_match_radius
        for gid in order:
            dd = math.hypot(vps[gid].x - end[0], vps[gid].y - end[1])
            if dd < best_d:
                best, best_d = gid, dd
        if best is not None and vps[best].leg is None:
            vps[best].leg = pts
    return [vps[g] for g in order]


def fallback_viewpoints(layout, grid=None, start=None, standoff=1.4, robot_radius=0.35):
    """Viewpoints straight in front of every layout gauge (used when the
    patrol planner is unavailable), visited in greedy nearest-neighbour order
    from `start`."""
    vps = []
    for g in layout.gauges:
        gx, gy = float(g["position"][0]), float(g["position"][1])
        nx, ny = (float(c) for c in (g.get("normal") or [0, 1, 0])[:2])
        n = math.hypot(nx, ny) or 1.0
        nx, ny = nx / n, ny / n
        yaw = math.atan2(-ny, -nx)              # robot looks back at the dial
        chosen = None
        for d in (standoff, standoff - 0.3, standoff + 0.3, standoff + 0.6, 1.0, 2.0):
            for lat in (0.0, 0.3, -0.3, 0.6, -0.6):
                x = gx + nx * d - ny * lat
                y = gy + ny * d + nx * lat
                if grid is None or grid.is_free(x, y, robot_radius):
                    chosen = (x, y, math.atan2(gy - y, gx - x))
                    break
            if chosen:
                break
        if chosen is None:
            x, y = gx + nx * standoff, gy + ny * standoff
            if grid is not None:
                nf = grid.nearest_free(x, y, robot_radius, 2.0)
                if nf is not None:
                    x, y = nf
            chosen = (x, y, math.atan2(gy - y, gx - x) if (x, y) != (gx, gy) else yaw)
        vps.append(Viewpoint(str(g["id"]), chosen[0], chosen[1], chosen[2]))
    if start is None or not vps:
        return vps
    ordered, cur, rest = [], (float(start[0]), float(start[1])), list(vps)
    while rest:
        k = min(range(len(rest)), key=lambda i: math.hypot(rest[i].x - cur[0], rest[i].y - cur[1]))
        vp = rest.pop(k)
        ordered.append(vp)
        cur = (vp.x, vp.y)
    return ordered


def cross_track_distance(path, x, y):
    """Distance (m) from (x, y) to a polyline."""
    if path is None or len(path) == 0:
        return 0.0
    p = np.asarray(path, dtype=float)
    if len(p) == 1:
        return float(math.hypot(p[0, 0] - x, p[0, 1] - y))
    a, b = p[:-1], p[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    tt = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / L2, 0.0, 1.0)
    proj = a + ab * tt[:, None]
    return float(np.min(np.hypot(proj[:, 0] - x, proj[:, 1] - y)))


def _default_planner(grid, start, goal, robot_radius, extra_cost=None):
    return plan_path(grid, start, goal, robot_radius=robot_radius, extra_cost=extra_cost)


# ----------------------------------------------------------------------------
# the state machine
# ----------------------------------------------------------------------------
class MissionLogic:
    def __init__(self, params=None, layout=None, planner=None):
        self.p = dict(MISSION_DEFAULTS)
        if params:
            self.p.update(params)
        self.layout = layout
        # layout debris as extra A* cost (recomputed only when the map changes)
        self.terrain_cost = TerrainCostCache.from_params(layout, self.p)
        self.planner = planner or self._terrain_planner
        self.pp = PurePursuit(lookahead=self.p["lookahead"], max_vx=self.p["max_vx"], max_wz=self.p["max_wz"],
                              goal_tolerance=self.p["goal_tolerance"], yaw_tolerance=self.p["align_yaw_tolerance"],
                              slow_down_dist=self.p["slow_down_dist"],
                              rotate_in_place_angle=self.p["rotate_in_place_angle"], k_heading=self.p["k_heading"])
        # novel obstacles seen by the robot (elevation map) on top of the static nav map
        self.obstacles = ObstacleAvoider.from_params(self.p, float(self.p["robot_radius"]))
        # lying pipes of the layout: straight crossings (planning) + crossing governor (following)
        self.crossing = CrossingGovernor(layout, CrossingParams.from_params(self.p))
        # world inputs
        self.grid = None
        self.viewpoints = []
        self.plan_source = None
        self._plan_key = None
        self.pose = None
        self.pose_t = None
        self.gas_ppm = None
        self.gas_t = None
        self._gas_above_since = None
        self.seek_status = {}
        self.seek_status_t = None
        self.safety = {}
        self.safety_t = None
        self.rth_active = False
        self._rth_fall_t = None
        self.teleop_active = False
        self._teleop_t = None          # time of the last /teleop/active true (heartbeat)
        self.loco_mode = None
        self.loco_recovery = None      # /locomotion/state recovery_state (FAILED: module1 gave up)
        self._loco_was_ready = True
        self._stale_hold = False       # 'hold' entered only for a stale pose with a valid path
        self._tick_lead = None         # pure-pursuit look-ahead the obstacle governor asked for this tick
        self._fallen_since = None
        self.latest_readings = {}
        self._last_readings_stamp = None
        # mission
        self.base = "IDLE"
        self.phase = ""
        self.mode = None
        self.paused = False
        self.queue = []
        self.current = None
        self.completed = []
        self.missed = []
        self.results = {}
        self.total = 0
        self.mission_t0 = None
        self.mission_t_end = None
        self.n_readings = 0
        self.replans = 0
        self.total_replans = 0
        self.path = None
        self._last_replan_t = -1e9
        self._prog_ref = None
        self._prog_time = 0.0
        self._phase_time = 0.0
        self._last_t = None
        self._t_first = None
        self._autostarted = False
        self._start_pending = None
        self._plan_wait = None         # (mode, t) of a start waiting for /patrol/plan_json
        self._plan_waited = False      # waited once already (autostart / an earlier start): never again
        self.cmd_seq = 0               # /mission/command counter (acknowledgement for keyboard_teleop / UI)
        self.recent_commands = []      # last COMMAND_HISTORY {seq, cmd, accepted, message}
        self.gas_armed = True
        self._gas_rearm_t = None
        self.gas_alarm = False
        self.seek_result = None
        self._seek_resume = False
        self._seek_t0 = None
        self._seek_seen_active = False
        self.rth_reason = None
        self._last_restand_t = -1e9
        self._target_sent = None
        self._seek_sent = None
        self._last_reported = None
        self.message = "ready"
        self._warned_no_map = False
        self._out = MissionOutput()

    # ================================================================ inputs
    def set_map(self, grid):
        self.grid = grid
        self.obstacles.set_static_map(grid)

    def obstacle_frame_due(self, t):
        """Should the node convert / fuse the obstacle frame arriving at t?
        Every frame while the mission drives (patrol legs, self-RTH), one per
        ``obstacle_idle_period`` s otherwise (inspecting, idle, docked,
        paused, manual, the seeker or return_to_home driving).  False when
        the frame would not be fused anyway (disabled, no fresh pose)."""
        if not self.obstacles.enabled or not self._pose_fresh(float(t)):
            return False
        driving = self._active() and ((self.base == "PATROL" and self.phase in ("navigate", "backup", "hold", ""))
                                      or (self.base == "RTH" and self.phase == "self"))
        return self.obstacles.frame_due(float(t), driving)

    def on_obstacle_frame(self, frame, t):
        """Perception frame (dusan_common.obstacle_layer.EvidenceFrame, in the
        pose frame).  Fused in every state (the memory stays current while
        paused / manual / inspecting); needs a fresh pose.  Returns True when
        the confirmed obstacle layer changed."""
        if not self.obstacles.enabled or not self._pose_fresh(float(t)):
            return False
        return self.obstacles.update(frame, self.pose, float(t))

    def set_plan(self, d):
        """Accept a /patrol/plan_json dict.  Returns True when it was new."""
        if not isinstance(d, dict):
            return False
        try:   # identity = route content; volatile fields (stamp, compute_s, reason ...) are ignored
            key = json.dumps({k: d.get(k) for k in ("viewpoints", "order", "legs")}, sort_keys=True)
        except (TypeError, ValueError):
            return False
        if key == self._plan_key:
            return False
        vps = parse_patrol_plan(d)
        if not vps:
            self._emit("warn", "patrol plan received but it contains no usable viewpoints")
            return False
        self._plan_key = key
        self.viewpoints = vps
        self._plan_waited = True
        self.plan_source = "patrol_planner"
        running = self.base in ("PATROL", "INSPECTING", "SEEK")
        self._emit("info", f"patrol plan: {len(vps)} viewpoints, order {[v.gauge_id for v in vps]}"
                   + (" (applies to the next mission)" if running else ""))
        return True

    def set_pose(self, x, y, yaw, t):
        self.pose = (float(x), float(y), float(yaw))
        self.pose_t = float(t)

    def on_gas(self, ppm, t):
        v = _finite(ppm)
        if v is not None:
            self.gas_ppm, self.gas_t = v, float(t)

    def on_seek_status(self, d, t):
        if isinstance(d, dict):
            self.seek_status = d
            self.seek_status_t = float(t)

    def on_safety_status(self, d, t):
        if isinstance(d, dict):
            self.safety = d
            self.safety_t = float(t)
            if "rth_active" in d and d["rth_active"] is not None:
                self.on_rth_active(bool(d["rth_active"]), t)

    def on_rth_active(self, flag, t):
        flag = bool(flag)
        if self.rth_active and not flag:
            self._rth_fall_t = float(t)
        if flag:
            self._rth_fall_t = None
        self.rth_active = flag

    def on_teleop_active(self, flag, t):
        flag = bool(flag)
        if flag:
            self._teleop_t = float(t)        # heartbeat (keyboard_teleop republishes every 0.5 s)
        if flag == self.teleop_active:
            return
        self.teleop_active = flag
        if flag:
            self._emit("warn", "manual override (teleop active) - mission frozen")
            if not self.paused:
                self._freeze()        # stop path following, pause the seeker
            self._out.state_changed = True
        else:
            self._emit("info", "manual override released")
            self._out.state_changed = True
            if not self.paused:
                self._unfreeze(t)

    def on_locomotion_state(self, d, t):
        if isinstance(d, dict) and d.get("mode"):
            self.loco_mode = str(d["mode"]).upper()
            rs = d.get("recovery_state")
            self.loco_recovery = str(rs).upper() if rs else None

    def on_gauge_readings(self, d, t):
        if not isinstance(d, dict):
            return
        stamp = d.get("stamp")
        if stamp is not None and stamp == self._last_readings_stamp:
            return                                    # duplicate delivery
        self._last_readings_stamp = stamp
        for r in d.get("readings") or []:
            if not isinstance(r, dict) or r.get("id") is None:
                continue
            gid = str(r["id"])
            val = _finite(r.get("value"))
            conf = _finite(r.get("confidence")) or 0.0
            self.latest_readings[gid] = {"value": val, "unit": r.get("unit"), "status": r.get("status"),
                                         "confidence": conf, "t": float(t)}
            if (self.base == "INSPECTING" and self.current is not None and gid == self.current.gauge_id
                    and val is not None and conf >= float(self.p["min_confidence"]) and self._active()):
                self.n_readings += 1

    # ============================================================== commands
    def command(self, cmd, t):
        """Handle a /mission/command string.  Returns True when accepted.

        Every command is numbered (``cmd_seq``) and its outcome (accepted, the
        message it produced) is kept in ``recent_commands`` and reported on
        /mission/state, so a sender can tell 'received and accepted',
        'received but rejected' and 'never arrived' apart."""
        self.cmd_seq += 1
        n0 = len(self._out.events)
        ok = bool(self._command(cmd, t))
        c = str(cmd or "").strip().lower()
        if ok and self._plan_wait is not None and c in ("rth", "dock", "seek"):
            # another task was asked for while a start waited for the patrol plan: the waiting
            # start must not fire later in the middle of it
            self._plan_wait = None
            self._emit("info", f"start waiting for the patrol plan cancelled by '{c}'")
        new = self._out.events[n0:]
        # the most telling event of this command: a warning / error, else the first that is not
        # a bare state transition ('full mission started: [...]', 'waiting for return_to_home ...')
        text = (next((x for lv, x in new if lv in ("warn", "alarm", "error")), None)
                or next((x for _lv, x in new if not x.startswith("state ")), None)
                or (new[0][1] if new else ""))
        self.recent_commands = (self.recent_commands + [{
            "seq": self.cmd_seq, "cmd": str(cmd or "").strip().lower(), "accepted": ok,
            "message": text}])[-COMMAND_HISTORY:]
        self._out.state_changed = True
        return ok

    def _command(self, cmd, t):
        c = str(cmd or "").strip().lower()
        if c not in MISSION_COMMANDS:
            self._emit("warn", f"unknown mission command '{cmd}'")
            return False
        if c == "abort":
            self._abort(t)
            return True
        if c in ("rth", "dock"):
            if self.base in ("RTH", "DOCKED"):
                return True                          # idempotent (also our own echo)
            self.paused = False
            self._enter_rth(t, reason="manual", requested=True)
            return True
        if c == "pause":
            if self.base in ("PATROL", "INSPECTING", "SEEK") or (self.base == "RTH" and self.phase == "self"):
                if not self.paused:
                    self.paused = True
                    self._freeze()
                    self._emit("info", "mission paused")
                return True
            self._emit("info", f"nothing to pause in {self.base}")
            return False
        if c == "resume":
            if self.paused:
                self.paused = False
                self._emit("info", "mission resumed")
                if not self.teleop_active:
                    self._unfreeze(t)
                self._out.state_changed = True
                return True
            if self._start_pending:
                return True                          # already waiting for return_to_home (or our own echo)
            if self.base in ("IDLE", "DONE", "DOCKED") and (self.queue or self.current) and self.mode:
                if self.rth_active:
                    self._start_pending = "resume"
                    self._emit("info", "resume requested - waiting for return_to_home to release")
                    return True
                self._emit("info", f"resuming interrupted mission ({len(self.queue) + bool(self.current)} "
                                   "viewpoints left)")
                self._enter_patrol(t, starting=True)
                return True
            self._emit("info", "nothing to resume")
            return False
        if c in ("start", "patrol"):
            if self.base in ("PATROL", "INSPECTING", "SEEK"):
                self._emit("warn", f"'{c}' ignored: mission already running ({self.base}); abort first")
                return False
            return self._begin_mission("full" if c == "start" else "patrol", t)
        if c == "seek":
            if self.base == "SEEK":
                return True
            if self.base in ("PATROL", "INSPECTING"):
                self._enter_seek(t, resume=True, reason="manual request")
                return True
            if self.base in ("IDLE", "DONE", "DOCKED"):
                if self.rth_active:
                    self._emit("warn", "'seek' ignored while return_to_home is active")
                    return False
                self._enter_seek(t, resume=False, reason="manual request")
                return True
            self._emit("warn", f"'seek' ignored in {self.base}")
            return False
        return False

    # ================================================================== step
    def step(self, t):
        """Advance the state machine to time t.  Returns a MissionOutput."""
        t = float(t)
        if self._t_first is None:
            self._t_first = t
        if self._last_t is None or t < self._last_t:
            dt = 0.0
        else:
            dt = min(0.5, t - self._last_t)
        self._last_t = t
        self._expire_teleop(t)
        if self._active():
            self._phase_time += dt

        self._autostart(t)
        if self._plan_wait is not None and t < self._plan_wait[1]:
            self._plan_wait = (self._plan_wait[0], t)      # sim clock reset: wait from now
        if self._plan_wait is not None and (self.viewpoints or t - self._plan_wait[1]
                                            >= float(self.p.get("start_plan_wait", 0.0) or 0.0)):
            mode, self._plan_waited = self._plan_wait[0], True
            self._plan_wait = None
            self._begin_mission(mode, t)
        self._update_gas(t)
        self._check_rth(t)
        self._watch_fallen(t)

        if self._active():
            handler = {
                "PATROL": self._step_patrol,
                "INSPECTING": self._step_inspecting,
                "SEEK": self._step_seek,
                "RTH": self._step_rth,
                "DOCKED": self._step_docked,
            }.get(self.base)
            if handler is not None:
                handler(t, dt)
        elif self.base in ("PATROL", "INSPECTING") or (self.base == "RTH" and self.phase == "self"):
            if self.paused and not self.teleop_active:
                self._out.cmd_vel = (0.0, 0.0, 0.0)      # hold position while paused

        rep = self.reported_state
        if rep != self._last_reported:
            self._last_reported = rep
            self._out.state_changed = True
        out, self._out = self._out, MissionOutput()
        return out

    @property
    def reported_state(self):
        if self.teleop_active:
            return "MANUAL"
        if self.paused:
            return "PAUSED"
        return self.base

    def state_dict(self, t=None):
        cur = self.current
        pending = ([cur.gauge_id] if cur else []) + [v.gauge_id for v in self.queue]
        remaining = self.pp.remaining_distance(*self.pose[:2]) if (self.pose and self.pp.active) else 0.0
        return {
            "state": self.reported_state,
            "base_state": self.base,
            "phase": self.phase,
            "mode": self.mode or "",
            "target": cur.gauge_id if (cur and self.base in ("PATROL", "INSPECTING")) else "",
            "completed": list(self.completed),
            "pending": pending,
            "missed": list(self.missed),
            "gas_alarm": bool(self.gas_alarm or self.base == "SEEK"),
            "gas_ppm": self.gas_ppm,
            "message": self.message,
            "viewpoint": cur.as_list() if cur else None,
            "progress": [len(self.completed) + len(self.missed), self.total],
            "readings_at_target": self.n_readings if self.base == "INSPECTING" else 0,
            "replans": self.total_replans,
            "seek": self.seek_result,
            "rth_reason": self.rth_reason,
            "paused": self.paused,
            "manual": self.teleop_active,
            "plan_source": self.plan_source,
            "results": self.results,
            "path_remaining": round(float(remaining), 2),
            "obstacle": self.obstacles.status(),
            "mission_elapsed": (round((self.mission_t_end if self.mission_t_end is not None else t)
                                      - self.mission_t0, 1)
                                if (t is not None and self.mission_t0 is not None) else None),
            "cmd_seq": self.cmd_seq,
            "recent_commands": list(self.recent_commands),
            "t": t,
        }

    # ============================================================ internals
    def _expire_teleop(self, t):
        """Release a manual override whose /teleop/active heartbeat stopped
        (teleop killed or crashed without sending false).  Same rule as the
        cmd_vel mux flag_timeout; a backward clock jump (sim reset) restarts
        the window instead of releasing at once."""
        timeout = float(self.p.get("teleop_active_timeout", 0.0) or 0.0)
        if not self.teleop_active or timeout <= 0.0 or self._teleop_t is None:
            return
        if t < self._teleop_t - 1e-6:
            self._teleop_t = t
            return
        if t - self._teleop_t > timeout:
            self.on_teleop_active(False, t)
            self._emit("warn", f"manual override expired: no /teleop/active for {timeout:.1f} s (teleop gone?)")

    def _active(self):
        return not (self.teleop_active or self.paused)

    def _emit(self, level, text):
        self.message = text
        self._out.events.append((level, text))

    def _set_base(self, state, phase=""):
        if state != self.base:
            self._emit("info", f"state {self.base} -> {state}")
            self._out.state_changed = True
        self.base = state
        self._set_phase(phase)

    def _set_phase(self, phase):
        if phase != self.phase:
            self._out.state_changed = True
        self.phase = phase
        self._phase_time = 0.0

    def _send_target(self, gid):
        gid = gid or ""
        if gid != self._target_sent:
            self._target_sent = gid
            self._out.target = gid

    def _send_seek(self, enable):
        enable = bool(enable)
        if enable != self._seek_sent:
            self._seek_sent = enable
            self._out.seek_enable = enable

    def _send_path(self, path):
        self.path = [tuple(map(float, p[:2])) for p in path] if path else []
        self._out.path = list(self.path)

    def _stop(self):
        self.pp.clear()
        self._out.cmd_vel = (0.0, 0.0, 0.0)

    def _pose_fresh(self, t):
        return self.pose is not None and (t - self.pose_t) <= float(self.p["pose_timeout"])

    def _loco_ready(self):
        return self.loco_mode is None or self.loco_mode in LOCO_READY

    # -------------------------------------------------------------- freezing
    def _freeze(self):
        self.pp.clear()
        self._out.cmd_vel = (0.0, 0.0, 0.0)
        if self.base == "SEEK":
            self._send_seek(False)
        self._out.state_changed = True

    def _unfreeze(self, t):
        """Resume the base activity after PAUSED/MANUAL (pose may have changed)."""
        if self.base == "SEEK":
            self._send_seek(True)
        elif self.base == "PATROL":
            if self.phase == "starting":
                return
            self._navigate_to_current(t)
        elif self.base == "INSPECTING":
            # re-approach the viewpoint (immediate if we are still there), then re-dwell
            self._set_base("PATROL")
            self._navigate_to_current(t)
        elif self.base == "RTH" and self.phase == "self":
            self._enter_self_rth(t)

    # ------------------------------------------------------------- autostart
    def _autostart(self, t):
        if not self.p["autostart"] or self._autostarted:
            return
        waited = t - self._t_first
        if waited < float(self.p["autostart_delay"]) or self.pose is None:
            return
        if not self.viewpoints and waited < float(self.p["autostart_delay"]) + float(self.p["plan_wait_timeout"]):
            return
        self._autostarted = True
        self._plan_waited = True          # autostart waited for the plan above
        self._emit("info", "autostart: starting the full mission")
        self.command("start", t)

    # ------------------------------------------------------------------ gas
    def _update_gas(self, t):
        fresh = self.gas_ppm is not None and self.gas_t is not None and (t - self.gas_t) <= float(self.p["gas_timeout"])
        above = fresh and self.gas_ppm > float(self.p["gas_alarm_ppm"])
        if above:
            if self._gas_above_since is None:
                self._gas_above_since = t
        else:
            self._gas_above_since = None
        alarm = above and (t - self._gas_above_since) >= float(self.p["gas_alarm_hold"])
        if alarm and not self.gas_alarm:
            self._emit("alarm", f"GAS ALARM: {self.gas_ppm:.1f} ppm > {self.p['gas_alarm_ppm']:.0f} ppm")
        self.gas_alarm = alarm
        if (not self.gas_armed and self._gas_rearm_t is not None and t >= self._gas_rearm_t
                and fresh and self.gas_ppm < float(self.p["gas_rearm_ppm"])):
            self.gas_armed = True
            self._gas_rearm_t = None
            self._emit("info", "gas alarm re-armed")
        if alarm and self.gas_armed and self.base in ("PATROL", "INSPECTING") and self._active() \
                and self.phase != "starting":
            self._enter_seek(t, resume=True, reason=f"gas {self.gas_ppm:.1f} ppm")

    # ------------------------------------------------------------------ RTH
    def _safety_docked(self):
        """Docked according to a FRESH /safety/status (a dead return_to_home must not keep us docked)."""
        if (self.safety_t is None or self._last_t is None
                or self._last_t - self.safety_t > float(self.p["safety_timeout"])):
            return False
        s = self.safety or {}
        return bool(s.get("docked")) or str(s.get("state", "")).upper() in SAFETY_DOCKED_STATES

    def _check_rth(self, t):
        # level triggered: while return_to_home is active the mission yields
        if self.rth_active and self.base not in ("RTH", "DOCKED"):
            reason = (self.safety or {}).get("reason") or "safety"
            self._enter_rth(t, reason=reason, requested=False)
        if not self.rth_active and self._start_pending and self._rth_fall_t is not None \
                and t - self._rth_fall_t >= float(self.p["rth_release_grace"]):
            kind, self._start_pending = self._start_pending, None
            self._emit("info", f"return_to_home released - {kind} the mission")
            self._enter_patrol(t, starting=True)

    def _enter_rth(self, t, reason, requested):
        if self.base == "SEEK":
            self._send_seek(False)
        if self.current is not None and self.base in ("PATROL", "INSPECTING", "SEEK"):
            self.queue.insert(0, self.current)       # keep it pending for a later resume
            self.current = None
        self.pp.clear()
        self._send_target("")
        self.paused = False
        self.rth_reason = reason
        phase = "yield" if (self.rth_active or not requested) else "requested"
        self._set_base("RTH", phase)
        self._emit("warn" if reason not in ("mission_complete",) else "info",
                   f"return to home ({reason})" + (" - waiting for return_to_home" if phase == "requested" else ""))

    def _enter_self_rth(self, t):
        if self.layout is None:
            self._emit("error", "self RTH impossible: no layout")
            self._set_base("IDLE")
            return
        if not self._pose_fresh(t):
            # no odometry: wait (phase self_wait) instead of giving up on the dock
            self.pp.clear()
            self._out.cmd_vel = (0.0, 0.0, 0.0)
            if self.phase != "self_wait":
                self._emit("warn", "self RTH: waiting for /odom")
                self._set_phase("self_wait")
            return
        if self.phase not in ("self", "self_wait"):
            self.obstacles.new_goal()
        dx, dy, dyaw = self.layout.dock_pose
        path = self._plan((dx, dy))
        if path is None:
            self._emit("error", "self RTH: no path to the dock")
            self._set_base("IDLE")
            return
        self.pp.set_path(path, goal_yaw=float(dyaw))
        self.obstacles.set_route(path)
        self._send_path(path)
        self._reset_progress()
        self._set_phase("self")

    def _step_rth(self, t, dt):
        if self._safety_docked() or (self.phase == "self" and self._at_dock()):
            self._enter_docked(t)
            return
        if self.phase == "requested":
            if self.rth_active:
                self._set_phase("yield")
            elif self._phase_time >= float(self.p["rth_ack_timeout"]):
                if self.p["self_rth_fallback"]:
                    self._emit("warn", "return_to_home did not acknowledge - driving to the dock myself")
                    self._enter_self_rth(t)
                else:
                    self._emit("error", "return_to_home did not acknowledge the RTH request")
                    self._set_base("IDLE")
            return
        if self.phase == "yield":
            if self.rth_active:
                return
            if self._rth_fall_t is None:
                self._rth_fall_t = t
            if t - self._rth_fall_t < float(self.p["rth_release_grace"]):
                return
            if self._start_pending:
                return                                   # handled by _check_rth
            if self.p["auto_resume_after_rth"] and (self.queue or self.current) and self.mode:
                self._emit("info", "return_to_home released - resuming the mission")
                self._enter_patrol(t, starting=False)
            elif self._near_dock():
                # the short DOCKED/CHARGING phase of return_to_home was not observed
                self._enter_docked(t)
            else:
                self._emit("info", "return_to_home released (not docked)")
                self._send_path([])
                self._set_base("IDLE")
            return
        if self.phase in ("self", "self_wait"):
            if self.rth_active:                          # the RTH node woke up after all
                self.pp.clear()
                self._set_phase("yield")
                return
            if self.phase == "self_wait":
                self._enter_self_rth(t)                  # plans as soon as /odom is fresh
                return
            if not self._pose_fresh(t) or not self._loco_ready():
                # never steer on a stale pose / while fallen: hold, keep the path
                self._out.cmd_vel = (0.0, 0.0, 0.0)
                return
            self._follow_path(t, dt, on_arrival=self._on_self_dock_arrival, on_fail=self._on_self_rth_fail)

    def _safety_battery_pct(self):
        """Battery from /safety/status in percent (accepts 0..1 or 0..100)."""
        v = _finite((self.safety or {}).get("battery_pct"))
        if v is None:
            return None
        return v * 100.0 if v <= 1.0 else v

    def _near_dock(self):
        if self.layout is None or self.pose is None:
            return False
        dx, dy, _ = self.layout.dock_pose
        return math.hypot(self.pose[0] - dx, self.pose[1] - dy) <= float(self.p["dock_detect_radius"])

    def _at_dock(self):
        if self.layout is None or self.pose is None:
            return False
        dx, dy, dyaw = self.layout.dock_pose
        return (math.hypot(self.pose[0] - dx, self.pose[1] - dy) <= float(self.p["dock_tolerance"])
                and abs(wrap_angle(self.pose[2] - dyaw)) <= 0.3 and not self.pp.active)

    def _on_self_dock_arrival(self, t):
        self._out.locomotion.append("sit")
        self._enter_docked(t)

    def _on_self_rth_fail(self, t, why):
        self._emit("error", f"self RTH failed: {why}")
        self._stop()
        self._set_base("IDLE")

    def _enter_docked(self, t):
        self.pp.clear()
        self._send_path([])
        done = self.mode is not None and not self.queue and self.current is None
        self._set_base("DOCKED")
        if done:
            self._emit("info", f"mission complete - docked ({len(self.completed)} read, {len(self.missed)} missed)")
            self.mode = None
            self.mission_t_end = t
        else:
            self._emit("info", "docked" + (f" ({len(self.queue)} viewpoints pending, 'resume' continues)"
                                           if self.queue else ""))

    def _step_docked(self, t, dt):
        if self.layout is None or self.pose is None or self._safety_docked():
            return
        dx, dy, _ = self.layout.dock_pose
        if math.hypot(self.pose[0] - dx, self.pose[1] - dy) > float(self.p["undock_distance"]):
            self._emit("info", "robot left the dock")
            self._set_base("IDLE")

    # --------------------------------------------------------------- abort
    def _abort(self, t):
        self._send_seek(False)
        self._send_target("")
        self._stop()
        self._send_path([])
        self.queue = []
        self.current = None
        self.mode = None
        self.paused = False
        self._start_pending = None
        self._plan_wait = None
        if self.mission_t0 is not None and self.mission_t_end is None:
            self.mission_t_end = t
        self._set_base("IDLE")
        self._emit("warn", "mission aborted")

    # ------------------------------------------------------------- mission
    def _mission_viewpoints(self):
        if self.viewpoints:
            return [Viewpoint(v.gauge_id, v.x, v.y, v.yaw, list(v.leg) if v.leg else None) for v in self.viewpoints]
        if self.p["fallback_viewpoints"] and self.layout is not None:
            start = self.pose[:2] if self.pose else None
            if start is None:
                try:
                    start = tuple(self.layout.spawn[:2])
                except (AttributeError, KeyError, TypeError):
                    start = None
            vps = fallback_viewpoints(self.layout, self.grid, start,
                                      float(self.p["fallback_standoff"]), float(self.p["robot_radius"]))
            if vps:
                self.plan_source = "layout_fallback"
                self._emit("warn", "no /patrol/plan_json received - using layout fallback viewpoints "
                                   f"{[v.gauge_id for v in vps]}")
            return vps
        return []

    def _begin_mission(self, mode, t):
        wait = float(self.p.get("start_plan_wait", 0.0) or 0.0)
        if (not self.viewpoints and not self._plan_waited and wait > 0.0 and self.p["fallback_viewpoints"]
                and self.layout is not None):
            # patrol_planner needs ~12 s (desktop) .. 30+ s (loaded laptop) after start-up: a 't'
            # pressed before would silently take the layout fallback route instead of the planned one
            if self._plan_wait is None:
                self._plan_wait = (mode, float(t))
                self._emit("info", f"{mode} accepted - waiting up to {wait:.0f} s for the patrol plan "
                                   "(patrol_planner still computing)")
            else:
                self._plan_wait = (mode, self._plan_wait[1])
            self._out.state_changed = True
            return True
        self._plan_wait = None
        vps = self._mission_viewpoints()
        if not vps:
            self._emit("error", f"cannot {mode}: no patrol plan / viewpoints available")
            return False
        self.mode = mode
        self.queue = vps
        self.current = None
        self.completed, self.missed, self.results = [], [], {}
        self.total = len(vps)
        self.total_replans = 0
        self.gas_armed, self._gas_rearm_t = True, None
        self.seek_result = None
        self.rth_reason = None
        self.paused = False
        self.mission_t0 = t
        self.mission_t_end = None
        self._emit("info", f"{'full mission' if mode == 'full' else 'patrol'} started: "
                           f"{[v.gauge_id for v in vps]}")
        if self.rth_active:
            self._start_pending = "start"
            batt = self._safety_battery_pct()
            if (self.p["release_rth_on_start"] and self._safety_docked() and batt is not None
                    and batt >= float(self.p["release_min_battery_pct"])):
                self._out.mission.append("resume")       # return_to_home leaves CHARGING on 'resume'
                self._emit("info", f"docked with {batt:.0f} % battery - releasing return_to_home")
            else:
                self._emit("info", "waiting for return_to_home to release before starting"
                           + (f" (battery {batt:.0f} %)" if batt is not None else ""))
            self._out.state_changed = True
            return True
        self._enter_patrol(t, starting=True)
        return True

    def _enter_patrol(self, t, starting):
        self._send_seek(False)
        self._set_base("PATROL", "starting" if starting else "")
        if starting:
            self._out.locomotion.append("stand")
            self._out.cmd_vel = (0.0, 0.0, 0.0)
        else:
            self._next_or_current(t)

    def _next_or_current(self, t):
        if self.current is None:
            if not self.queue:
                self._mission_complete(t)
                return
            self.current = self.queue.pop(0)
            self.replans = 0
            self.obstacles.new_goal()
            self._emit("info", f"next viewpoint {self.current.gauge_id} at ({self.current.x:.2f}, "
                               f"{self.current.y:.2f})")
        self._navigate_to_current(t)

    def _navigate_to_current(self, t):
        vp = self.current
        if vp is None:
            self._next_or_current(t)
            return
        if self.base != "PATROL":
            self._set_base("PATROL")
        if not self._pose_fresh(t):
            # Planning from a missing / stale pose would declare the viewpoint
            # unreachable ('start' pressed before /odom arrives would skip the
            # whole route).  Hold instead; _loco_gate re-plans as soon as
            # odometry is fresh again.
            self.pp.clear()
            self._out.cmd_vel = (0.0, 0.0, 0.0)
            if self.phase != "hold":
                self._emit("warn", "waiting for /odom")
                self._set_phase("hold")
            return
        if math.hypot(self.pose[0] - vp.x, self.pose[1] - vp.y) <= self.p["goal_tolerance"]:
            self.pp.clear()
            self._set_phase("align")
            return
        path = self._plan((vp.x, vp.y), leg=vp.leg if self.p["use_plan_legs"] else None)
        if path is None:
            self._skip_current(t, "no path")
            return
        self.pp.set_path(path, goal_yaw=None)
        self.obstacles.set_route(path)
        self._send_path(path)
        self._last_replan_t = t
        self._reset_progress()
        self._set_phase("navigate")

    def _static_cost(self):
        """Terrain cost layer of the static nav map (None without map / layout)."""
        return self.terrain_cost(self.grid) if self.grid is not None else None

    def _terrain_planner(self, grid, start, goal, robot_radius):
        """Default planner: A* + cost-aware smoothing with the terrain cost layer.
        A static + dynamic grid (same geometry as the nav map) reuses the nav
        map's cost layer instead of recomputing it for every new combined grid."""
        if (grid is not self.grid and self.grid is not None and grid.occ.shape == self.grid.occ.shape
                and grid.resolution == self.grid.resolution and grid.origin_x == self.grid.origin_x
                and grid.origin_y == self.grid.origin_y):
            cost = self._static_cost()
        else:
            cost = self.terrain_cost(grid)
        return _default_planner(grid, start, goal, robot_radius, extra_cost=cost)

    def _plan(self, goal, leg=None):
        """Path from the current pose to goal (pre-planned leg when possible),
        densified to `follow_spacing` for the pure-pursuit follower."""
        raw = self._plan_raw(goal, leg)
        if raw is None:
            return None
        gx, gy = float(goal[0]), float(goal[1])
        # A* snaps start/goal to free cells and legs may end a little off the
        # viewpoint: make sure the true goal is the last point
        if math.hypot(raw[-1][0] - gx, raw[-1][1] - gy) > 1e-3:
            raw.append((gx, gy))
        if len(raw) == 1:
            raw = [(self.pose[0], self.pose[1])] + raw
        raw = self._straight_crossings(raw)
        # dusan_common.pure_pursuit selects its lookahead point among path
        # VERTICES starting at the closest one; with the 1.5 m spacing of
        # planning.plan() the closest vertex can lie behind the robot at more
        # than the lookahead distance, which makes the target flip behind /
        # ahead (rotate-in-place oscillation).  Dense vertices avoid that.
        return densify(raw, float(self.p["follow_spacing"]))

    def _straight_crossings(self, path):
        """Pipe crossings of ``path`` replaced by straight segments square to the pipe
        (dusan_common.debris_crossing), each new segment free on the nav map."""
        cg = self.crossing
        if not cg.enabled or len(path) < 2:
            return path
        r = 0.7 * float(self.p["robot_radius"])
        grid = self.grid
        line_ok = (lambda a, b: grid.line_of_sight(a[0], a[1], b[0], b[1], r)) if grid is not None else None
        out, n = straighten_crossings(path, cg.pipes, line_ok, cg.p)
        return out if n else path

    def _plan_raw(self, goal, leg):
        if self.pose is None:
            return None
        x, y = self.pose[0], self.pose[1]
        r = float(self.p["robot_radius"])
        # static + detected (novel) obstacles, best first: full occlusion shadow then the near one,
        # each with robot_radius and 0.7 r from mapped walls (robot_radius + margin from detected
        # obstacles); [] while nothing is confirmed -> legacy behaviour.  A grid without a path costs
        # ~1 ms (astar's connectivity check), not a search of the whole free space.
        attempts = self.obstacles.planning_grids(self.grid, r) if self.grid is not None else []
        dyn = attempts or None
        if leg:
            arr = np.asarray(leg, dtype=float)
            d = np.hypot(arr[:, 0] - x, arr[:, 1] - y)
            k = int(np.argmin(d))
            if d[k] <= float(self.p["leg_start_tolerance"]) and len(arr) - k >= 2:
                cand = [(x, y)] + [(float(p[0]), float(p[1])) for p in arr[k + 1:]]
                if dyn is None or not self.obstacles.route_blocked(cand, (x, y)):
                    return cand
                self._emit("info", "patrol leg blocked by a detected obstacle - planning around it")
        if self.grid is None:
            if not self._warned_no_map:
                self._warned_no_map = True
                self._emit("warn", "no /map yet - driving straight lines")
            return [(x, y), (float(goal[0]), float(goal[1]))]
        g = (float(goal[0]), float(goal[1]))
        for i, (grid, radius, layer) in enumerate(attempts):
            path = self.planner(grid, (x, y), g, radius)
            if path:
                if layer != "near":
                    path = self._near_alternative(attempts[i + 1:], path, radius, (x, y), g)
                return [(float(p[0]), float(p[1])) for p in path]
        for radius in (r, 0.7 * r):
            path = self.planner(self.grid, (x, y), g, radius)
            if path:
                if dyn is not None:
                    if not self.obstacles.p.static_fallback:
                        self._emit("warn", "no path around the detected obstacle(s)")
                        return None
                    self._emit("warn", "no path around the detected obstacle(s) - keeping the map route "
                                       "(stuck logic decides)")
                return [(float(p[0]), float(p[1])) for p in path]
        return None

    def _near_alternative(self, rest, path, radius, start, goal):
        """The full-shadow route, or the near-shadow one (same radius) when that
        is much cheaper (ObstacleAvoider.near_is_cheaper)."""
        near = next((a for a in rest if a[2] == "near" and abs(a[1] - radius) < 1e-9), None)
        if near is None or self.grid is None:
            return path
        av, cost = self.obstacles, self._static_cost()
        c_full = av.route_cost(self.grid, cost, path)
        if not av.near_is_cheaper(c_full, math.hypot(goal[0] - start[0], goal[1] - start[1])):
            return path                          # no route can be switch_ratio cheaper: skip the 2nd search
        alt = self.planner(near[0], start, goal, radius)
        if not alt:
            return path
        c_near = av.route_cost(self.grid, cost, alt)
        if av.near_is_cheaper(c_full, c_near):
            self._emit("info", f"route past the near side of the detected obstacle(s) ({c_near:.1f} vs "
                               f"{c_full:.1f} around the assumed back)")
            return alt
        return path

    def _skip_current(self, t, why):
        vp = self.current
        if vp is None:
            return
        self._emit("warn", f"viewpoint {vp.gauge_id} skipped: {why}")
        self.missed.append(vp.gauge_id)
        self.results[vp.gauge_id] = {"status": "missed", "reason": why}
        self.current = None
        self._send_target("")
        self.pp.clear()
        self._next_or_current(t)

    def _mission_complete(self, t):
        self.pp.clear()
        self._send_target("")
        n_ok, n_miss = len(self.completed), len(self.missed)
        if self.mode == "full":
            self._emit("info", f"all viewpoints done ({n_ok} read, {n_miss} missed) - returning home")
            self._out.mission.append("rth")
            self._enter_rth(t, reason="mission_complete", requested=True)
        else:
            self._send_path([])
            self._out.cmd_vel = (0.0, 0.0, 0.0)
            self._set_base("DONE")
            self._emit("info", f"patrol complete ({n_ok} read, {n_miss} missed)")
            self.mode = None
            self.mission_t_end = t

    # -------------------------------------------------------------- PATROL
    def _step_patrol(self, t, dt):
        if self.phase == "starting":
            ready = self.loco_mode is None or self.loco_mode in LOCO_READY
            if (ready and self._phase_time >= float(self.p["min_start_time"])) \
                    or self._phase_time >= float(self.p["start_timeout"]):
                if self.p["send_walk_after_stand"]:
                    self._out.locomotion.append("walk")
                self._set_phase("")
                self._next_or_current(t)
            else:
                self._out.cmd_vel = (0.0, 0.0, 0.0)
            return
        if self.current is None:
            self._next_or_current(t)
            return
        if not self._loco_gate(t):
            return
        if self.phase == "align":
            self._step_align(t)
            return
        if self.phase == "backup":
            if self._phase_time >= float(self.p["backup_time"]):
                self._navigate_to_current(t)
            elif self.obstacles.blocked_behind(
                    self.pose, float(self.p["backup_speed"]) * float(self.p["backup_time"]) + 0.3):
                # the camera cannot see behind, the obstacle memory remembers: do not reverse into it
                self._emit("info", "detected obstacle behind - re-planning without backing up")
                self._navigate_to_current(t)
            else:
                self._out.cmd_vel = (-float(self.p["backup_speed"]), 0.0, 0.0)
            return
        if self.phase in ("navigate", "hold", ""):
            if self.phase != "navigate" or not self.pp.active:
                self._navigate_to_current(t)
                return
            self._follow_path(t, dt, on_arrival=self._on_viewpoint_arrival, on_fail=self._skip_current)

    def _loco_gate(self, t):
        """Hold while the robot is not able to walk (fallen, recovering, e-stop, sitting)."""
        if not self._pose_fresh(t):
            self._out.cmd_vel = (0.0, 0.0, 0.0)
            if self.phase != "hold":
                self._emit("warn", "waiting for /odom")
                # a stale pose alone keeps the path; it is resumed without re-planning when /odom is
                # back.  A long tick (a slow plan on a busy CPU) lets the next, overdue timer tick run
                # before the queued /odom: re-planning on every release would block again and loop.
                self._stale_hold = self.phase == "navigate" and self.pp.active
                self._set_phase("hold")
            return False
        if self._loco_ready():
            if not self._loco_was_ready:
                self._loco_was_ready = True
                self._stale_hold = False
                self._emit("info", f"locomotion ready again ({self.loco_mode}) - re-planning")
                if self.p["send_walk_after_stand"]:
                    self._out.locomotion.append("walk")
                self._navigate_to_current(t)
                return False
            if self.phase == "hold":
                resume, self._stale_hold = self._stale_hold, False
                if (resume and self.pp.active and self.path
                        and cross_track_distance(self.path, self.pose[0], self.pose[1])
                        <= float(self.p["replan_deviation"])):
                    self._set_phase("navigate")
                    return True                          # /odom back: continue on the current path
                self._navigate_to_current(t)
                return False
            return True
        self._stale_hold = False
        if self._loco_was_ready:
            self._loco_was_ready = False
            self._emit("warn", f"locomotion not ready ({self.loco_mode}) - holding")
        self.pp.clear()
        self._out.cmd_vel = (0.0, 0.0, 0.0)
        if self.loco_mode == "SIT" and t - self._last_restand_t >= float(self.p["restand_period"]):
            self._last_restand_t = t
            self._out.locomotion.append("stand")
        if self.phase != "hold":
            self._set_phase("hold")
        return False

    def _watch_fallen(self, t):
        """A mission must not wait forever for a robot that stays down: module1
        retries the self-righting by itself (recovery_retry_time), this is the
        mission-level backstop - FALLEN for recover_after s in any mission
        phase (also INSPECTING / SEEK / RTH) -> 'recover', repeated."""
        after = float(self.p["recover_after"])
        if after <= 0.0 or self.loco_mode != "FALLEN" or not self._active() \
                or self.base not in ("PATROL", "INSPECTING", "SEEK", "RTH"):
            self._fallen_since = None
            return
        if self._fallen_since is None:
            self._fallen_since = t
        elif t - self._fallen_since >= after:
            self._fallen_since = t
            self._emit("warn", f"robot still down after {after:.0f} s (recovery {self.loco_recovery or '?'}) "
                               "- sending 'recover'")
            self._out.locomotion.append("recover")

    def _reset_progress(self):
        self._prog_ref = self.pose
        self._prog_time = 0.0

    def _obstacle_step(self, t):
        """Novel-obstacle avoidance for this tick (dusan_common.obstacle_layer):
        returns the speed scale, or None when the tick was consumed by a
        global re-plan.  An avoidance error never stops navigation (counted,
        logged on the 1st and every 50th, legacy behaviour for the tick)."""
        av = self.obstacles
        if not av.enabled or not self.pp.active:
            return 1.0
        try:
            act = av.step(self.pose, self.pp.remaining_path(), t, self.grid, self._static_cost,
                          lead=float(self.p["lookahead"]))
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            av.errors += 1
            if av.errors == 1 or av.errors % 50 == 0:
                self._emit("error", f"obstacle avoidance failed ({av.errors}x): {type(exc).__name__}: {exc}")
            return 1.0
        self._tick_lead = act.lead               # pure-pursuit look-ahead of this tick (tight near obstacles)
        if act.kind == "detour":
            self.pp.set_path(act.path, goal_yaw=self.pp.goal_yaw)
            self._send_path(act.path)
            self._emit("info", act.message)
            self._last_replan_t = t              # cross-track checks are against the detour now
        elif act.kind == "replan" and (t - self._last_replan_t) >= float(self.p["replan_min_interval"]):
            av.note_replan()
            self._emit("warn", f"{act.message} ({av.replans}/{av.p.max_replans})")
            self._last_replan_t = t
            if self.base == "PATROL":
                self._navigate_to_current(t)
            else:
                self._enter_self_rth(t)
            return None
        return act.speed_scale

    def _follow_path(self, t, dt, on_arrival, on_fail):
        """Pure pursuit (+ novel-obstacle avoidance) + stuck / deviation monitoring."""
        scale = self._obstacle_step(t)
        if scale is None:
            return
        x, y, yaw = self.pose
        lead, self._tick_lead = self._tick_lead, None
        if lead is not None:
            self.pp.lookahead = lead
        try:
            vx, vy, wz, done = self.pp.compute(x, y, yaw, speed_scale=scale)
        finally:
            self.pp.lookahead = float(self.p["lookahead"])
        if scale <= 0.0:
            # obstacle within the stop distance on the path: a true zero command - a heading
            # correction alone keeps a legged robot trotting in place, and it creeps (Gazebo w)
            vx = vy = wz = 0.0
        else:
            vy *= min(1.0, scale)
        if done:
            self._out.cmd_vel = (0.0, 0.0, 0.0)
            on_arrival(t)
            return
        if scale > 0.0:
            # lying pipe ahead / under the robot: turn onto the crossing before the feet reach it,
            # then a steady straight pace; slow past a pipe end (dusan_common.debris_crossing)
            vx, vy, wz = self.crossing.govern(x, y, yaw, (vx, vy, wz), self.pp.remaining_path())
        self._out.cmd_vel = (vx, vy, wz)
        # progress / stuck detection
        if self._prog_ref is None:
            self._reset_progress()
        moved = math.hypot(x - self._prog_ref[0], y - self._prog_ref[1])
        turned = abs(wrap_angle(yaw - self._prog_ref[2]))
        if moved >= float(self.p["stuck_distance"]) or turned >= float(self.p["stuck_yaw"]):
            self._reset_progress()
        else:
            self._prog_time += dt
        if self._prog_time >= float(self.p["stuck_time"]):
            self.replans += 1
            self.total_replans += 1
            if self.replans > int(self.p["max_replans"]):
                on_fail(t, f"stuck ({self.replans - 1} re-plans)")
                return
            self._emit("warn", f"stuck at ({x:.2f}, {y:.2f}) - backing up and re-planning "
                               f"({self.replans}/{self.p['max_replans']})")
            self.pp.clear()
            self._reset_progress()
            if self.base == "PATROL":
                self._set_phase("backup")
            else:                                        # self RTH: re-plan straight away
                self._enter_self_rth(t)
            return
        # large deviation from the path -> re-plan (rate limited)
        if (t - self._last_replan_t) >= float(self.p["replan_min_interval"]) and self.path \
                and cross_track_distance(self.path, x, y) > float(self.p["replan_deviation"]):
            self._emit("info", "off the path - re-planning")
            self._last_replan_t = t
            if self.base == "PATROL":
                self._navigate_to_current(t)
            else:
                self._enter_self_rth(t)

    def _on_viewpoint_arrival(self, t):
        self._set_phase("align")

    def _step_align(self, t):
        vp = self.current
        err = wrap_angle(vp.yaw - self.pose[2])
        if abs(err) <= float(self.p["align_yaw_tolerance"]) or self._phase_time >= float(self.p["align_timeout"]):
            if abs(err) > float(self.p["align_yaw_tolerance"]):
                self._emit("warn", f"{vp.gauge_id}: alignment timeout (yaw error {math.degrees(err):.1f} deg)")
            self._out.cmd_vel = (0.0, 0.0, 0.0)
            self._enter_inspecting(t)
            return
        wz = float(self.p["align_k"]) * err
        wz = math.copysign(max(abs(wz), float(self.p["align_min_wz"])), wz)
        wz = max(-float(self.p["max_wz"]), min(float(self.p["max_wz"]), wz))
        self._out.cmd_vel = (0.0, 0.0, wz)

    # ----------------------------------------------------------- INSPECTING
    def _enter_inspecting(self, t):
        self.n_readings = 0
        self._set_base("INSPECTING", "dwell")
        self._send_target(self.current.gauge_id)
        self._emit("info", f"inspecting {self.current.gauge_id}")

    def _step_inspecting(self, t, dt):
        vp = self.current
        if vp is None:
            self._set_base("PATROL")
            return
        enough = self.n_readings >= int(self.p["min_readings"])
        if self.phase == "dwell":
            self._out.cmd_vel = (0.0, 0.0, 0.0)
            if enough and self._phase_time >= float(self.p["min_dwell_time"]):
                self._finish_inspection(t, True)
            elif self._phase_time >= float(self.p["dwell_time"]):
                if self.n_readings > 0:
                    self._finish_inspection(t, True)
                elif self.p["scan_on_miss"]:
                    self._emit("info", f"{vp.gauge_id}: no reading yet - scanning left/right")
                    self._set_phase("scan_left")
                else:
                    self._finish_inspection(t, False)
            return
        # scan phases: look slightly left, then right, while still counting readings
        if enough and self._phase_time >= 0.5:
            self._finish_inspection(t, True)
            return
        off = float(self.p["scan_yaw_offset"]) * (1.0 if self.phase == "scan_left" else -1.0)
        err = wrap_angle(vp.yaw + off - self.pose[2])
        wz = 0.0 if abs(err) < 0.05 else max(-float(self.p["max_wz"]), min(float(self.p["max_wz"]),
                                                                         float(self.p["align_k"]) * err))
        self._out.cmd_vel = (0.0, 0.0, wz)
        if self._phase_time >= float(self.p["scan_time"]):
            if self.phase == "scan_left":
                self._set_phase("scan_right")
            else:
                self._finish_inspection(t, self.n_readings > 0)

    def _finish_inspection(self, t, ok):
        vp = self.current
        self._send_target("")
        if ok:
            self.completed.append(vp.gauge_id)
            r = dict(self.latest_readings.get(vp.gauge_id) or {})
            r["n_readings"] = self.n_readings
            self.results[vp.gauge_id] = r
            val = r.get("value")
            self._emit("info", f"{vp.gauge_id} read: "
                               + (f"{val:.3f} {r.get('unit') or ''} ({r.get('status')})" if val is not None else "ok")
                               + f" from {self.n_readings} readings")
        else:
            self.missed.append(vp.gauge_id)
            self.results[vp.gauge_id] = {"status": "missed", "reason": "no reading"}
            self._emit("warn", f"{vp.gauge_id}: no reading obtained - skipped")
        self.current = None
        self.n_readings = 0
        self._set_base("PATROL")
        self._next_or_current(t)

    # ----------------------------------------------------------------- SEEK
    def _enter_seek(self, t, resume, reason):
        self.pp.clear()
        self._send_target("")
        self._seek_resume = bool(resume)
        self._seek_t0 = t
        self._seek_seen_active = False
        self._set_base("SEEK", "seeking")
        self._send_seek(True)
        self._emit("alarm" if "gas" in reason else "info", f"gas source seeking started ({reason})")

    def _step_seek(self, t, dt):
        st = None
        if self.seek_status_t is not None and self._seek_t0 is not None and self.seek_status_t >= self._seek_t0:
            st = str(self.seek_status.get("state") or "").upper() or None
        if st and st not in SEEK_TERMINAL and st != "IDLE":
            self._seek_seen_active = True
        result = None
        if st in SEEK_TERMINAL and (self._seek_seen_active
                                    or self._phase_time >= float(self.p["seek_start_timeout"])):
            result = st
        elif not self._seek_seen_active and self._phase_time >= float(self.p["seek_start_timeout"]):
            result = "FAILED"
            self._emit("warn", "source seeker did not become active")
        elif self._phase_time >= float(self.p["seek_timeout"]):
            result = "TIMEOUT"
        if result is None:
            return
        s = self.seek_status or {}
        self.seek_result = {"result": result, "estimate": s.get("estimate"), "error_m": s.get("error_m"),
                            "elapsed": round(self._phase_time, 1), "path_length": s.get("path_length"),
                            "t": round(t, 2)}
        self._send_seek(False)
        if result == "FOUND":
            est = s.get("estimate")
            est_txt = f" at ({est[0]:.2f}, {est[1]:.2f})" if isinstance(est, (list, tuple)) and len(est) >= 2 \
                and est[0] is not None and est[1] is not None else ""
            err = _finite(s.get("error_m"))
            self._emit("alarm", f"GAS SOURCE FOUND{est_txt}" + (f", error {err:.2f} m" if err is not None else ""))
            self.gas_armed = False
        else:
            self._emit("warn", f"gas source seeking ended: {result}")
            self.gas_armed = False
            self._gas_rearm_t = t + float(self.p["gas_rearm_cooldown"])
        if self._seek_resume and self.mode:
            self._enter_patrol(t, starting=False)
        else:
            self._send_path([])
            self._set_base("IDLE")
