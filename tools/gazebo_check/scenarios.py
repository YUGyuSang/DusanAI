"""Gazebo check scenarios (a) .. (s) with numeric pass criteria (pure python).

Time line: spawn at t = 0 (straight legs, knees at the -0.2 rad limit =
the JointPositionController ``initial_position``) at floor + 0.56 m, the
controller starts after the node's start-up gate (0.5 s), stands up in 2.5 s
(STAND at ~3.0 s); commands start at ``T_CMD`` = 4.0 s.

Fall: |roll| or |pitch| > 55 deg or base height above the ground < 0.12 m
(ground truth) for >= 0.1 s, or the controller entered FALLEN.

Plant scenarios (j .. s) run in the REAL simulation/worlds/plant.sdf, from
world_layout.yaml robot_spawn (charging dock, lower corridor level, facing
west) or in front of one obstacle (n, p, q, r, s); a :class:`KeyboardOperator`
(the person at keyboard_teleop in hold mode, watching Gazebo) drives along
waypoints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

T_CMD = 4.0
CMD_SHIFT = 0.07       # [s] per seed, see command_fn
EFFORT = np.array([45.0, 45.0, 60.0] * 4)


@dataclass
class Scenario:
    key: str
    title: str
    duration: float
    gait: Optional[str] = None                       # "trot" | "crawl" | None (auto)
    cmd: Optional[Tuple[float, float, float]] = None  # constant command from T_CMD
    teleop: bool = False                             # random teleop segments (seeded)
    window: Tuple[float, float] = (6.0, 1e9)         # evaluation window of the speed metrics
    criteria: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    terrain: Optional[str] = None                    # world terrain (world.TERRAINS), None = the CLI --terrain
    spawn: Optional[Tuple[float, float, float, float]] = None   # (x, y, z, yaw), None = origin (CLI --yaw)
    pushes: Tuple = ()                               # ((t0, duration, (fx, fy, fz) world N), ...) on base_link
    route: Tuple = ()                                # waypoints ((x, y), ...) of a KeyboardOperator (plant world)
    speed_scale: float = 0.5                         # keyboard_teleop speed scale of that operator
    controller_start: float = 0.0                    # [s] sim time at which the locomotion node starts
    t_cmd: float = T_CMD                             # [s] first operator / constant command


SCENARIOS: Dict[str, Scenario] = {
    "a_stand": Scenario("a_stand", "(a) spawn -> stand-up -> stand 10 s", 13.5,
                        window=(3.5, 13.5),
                        criteria={"max_roll_deg": (0.0, 5.0), "max_pitch_deg": (0.0, 5.0),
                                  "height_mean": (0.31, 0.37), "drift_xy": (0.0, 0.10)}),
    "b_trot": Scenario("b_trot", "(b) trot 0.35 m/s, 20 s", T_CMD + 20.0, gait="trot", cmd=(0.35, 0.0, 0.0),
                       window=(6.0, T_CMD + 20.0),
                       criteria={"vx": (0.25, 0.45), "lateral_dev": (0.0, 0.60)}),
    "c_crawl": Scenario("c_crawl", "(c) crawl 0.2 m/s, 20 s", T_CMD + 20.0, gait="crawl", cmd=(0.2, 0.0, 0.0),
                        window=(7.0, T_CMD + 20.0),
                        criteria={"vx": (0.12, 0.26), "lateral_dev": (0.0, 0.60)}),
    "d_turn": Scenario("d_turn", "(d) turn in place 0.5 rad/s, 12 s", T_CMD + 12.0, gait="trot", cmd=(0.0, 0.0, 0.5),
                       window=(6.0, T_CMD + 12.0),
                       criteria={"wz": (0.35, 0.60), "drift_xy": (0.0, 0.60)}),
    "e_teleop": Scenario("e_teleop", "(e) teleop: random walk / turn commands, 60 s", T_CMD + 60.0, teleop=True,
                         window=(T_CMD, T_CMD + 60.0)),
    "i_push": Scenario("i_push", "(i) trot 0.3 m/s, pushes 10 N s (100 N x 0.1 s) left, right, forward",
                       T_CMD + 14.0, gait="trot", cmd=(0.3, 0.0, 0.0), window=(T_CMD, T_CMD + 14.0),
                       pushes=((8.0, 0.1, (0.0, 100.0, 0.0)), (12.0, 0.1, (0.0, -100.0, 0.0)),
                               (15.0, 0.1, (100.0, 0.0, 0.0))),
                       criteria={"progress": (2.0, 1e9)}),
    # terrain (blind unless the harness feeds the ground-truth elevation map, --map)
    "f_pipe": Scenario("f_pipe", "(f) trot 0.3 m/s over a 0.08 m pipe", T_CMD + 16.0, gait="trot",
                       cmd=(0.3, 0.0, 0.0), window=(T_CMD, T_CMD + 16.0), terrain="pipe",
                       criteria={"progress": (3.0, 1e9), "lateral_dev": (0.0, 0.60)}),
    "g_ramp": Scenario("g_ramp", "(g) trot 0.3 m/s up an 11.8 deg ramp", T_CMD + 20.0, gait="trot",
                       cmd=(0.3, 0.0, 0.0), window=(T_CMD, T_CMD + 20.0), terrain="ramp",
                       criteria={"progress": (4.5, 1e9), "lateral_dev": (0.0, 0.60)}),
    "h_ramp_down": Scenario("h_ramp_down", "(h) trot 0.3 m/s down an 11.8 deg ramp", T_CMD + 20.0, gait="trot",
                            cmd=(0.3, 0.0, 0.0), window=(T_CMD, T_CMD + 20.0), terrain="ramp",
                            spawn=(5.5, 0.0, 0.615 + 0.56, math.pi),
                            criteria={"progress": (4.0, 1e9), "lateral_dev": (0.0, 0.60)}),
}
# ---------------------------------------------------------------------------
# plant world routes (world frame, simulation/config/world_layout.yaml)
# ---------------------------------------------------------------------------
# dock (61.9, 0) facing west -> C-OBS1 pipe d 0.10 m across the corridor at x ~58.5 ->
# south of the C-OBS2 pallet (x 54.3 .. 55.7, y 0.39 .. 1.61) -> ramp (y -0.17 .. 1.76,
# x 51.2 (z -0.88) .. 47.0 (z 0), 11.8 deg) -> upper corridor level
ROUTE_RAMP = ((59.8, 0.0), (58.0, -0.15), (56.8, -0.45), (53.9, -0.45), (52.5, 0.8), (51.5, 0.8),
              (47.2, 0.8), (45.6, 0.6))
# (o): around the north end of the C-OBS1 pipe instead of over it (pipe ends at y 1.08 + radius,
# corridor wall at y 1.76 -> a 0.63 m gap; robot footprint ~0.32 m wide), then as ROUTE_RAMP
ROUTE_DETOUR = ((60.6, 0.0), (59.7, 1.45), (57.7, 1.45), (56.5, -0.45)) + ROUTE_RAMP[3:]
# the C-OBS3 rubble field (x 37 .. 39, y -1.3 .. 0.5; blocks up to y 0.41) leaves a 1.26 m passage
# along the north wall (face y 1.76): routes pass it on y = RUBBLE_PASS_Y (0.6 m from both)
RUBBLE_PASS_Y = 1.1
# (l): ROUTE_DETOUR and on along the upper corridor: past C-OBS3 through the north passage, C-OBS4
# steel plate mu 0.2 (x 30 .. 32.5), C-OBS5 pipe d 0.08 m (x ~22, spans the corridor up to a 0.09 m
# gap: crossed), to the hall door frame (x 15.2 .. 15.5)
ROUTE_FULL = ROUTE_DETOUR + ((44.0, 0.4), (40.0, RUBBLE_PASS_Y), (36.6, RUBBLE_PASS_Y), (33.5, 0.0), (29.0, 0.0),
                             (23.5, 0.0), (20.5, 0.0), (16.8, 0.0))
T_LATE = 15.0           # (k): locomotion node starts this long after the spawn

SCENARIOS.update({
    "j_corridor": Scenario("j_corridor", "(j) plant: dock -> pipe 0.10 m -> pallet -> ramp up, keyboard teleop 0.25 m/s",
                           T_CMD + 110.0, terrain="plant", route=ROUTE_RAMP, window=(T_CMD, T_CMD + 110.0),
                           criteria={"wp_frac": (0.999, 1e9)}),
    "k_late_start": Scenario("k_late_start", f"(k) plant: node starts {T_LATE:.0f} s after the spawn, stand + walk",
                             T_LATE + 4.0 + 8.0, terrain="plant", cmd=(0.25, 0.0, 0.0), controller_start=T_LATE,
                             t_cmd=T_LATE + 4.0, window=(T_LATE + 4.0, T_LATE + 12.0),
                             criteria={"pre_min_height": (0.45, 1e9), "pre_max_tilt_deg": (0.0, 5.0),
                                       "progress": (1.0, 1e9)}),
    "l_corridor_full": Scenario("l_corridor_full", "(l) plant: whole corridor dock -> around the 0.10 m pipe -> ramp "
                                "-> past the rubble (north passage) -> mu 0.2 plate -> pipe 0.08 m -> hall door, "
                                "keyboard teleop 0.25 m/s",
                                T_CMD + 340.0,
                                terrain="plant", route=ROUTE_FULL, window=(T_CMD, T_CMD + 340.0),
                                criteria={"wp_frac": (0.999, 1e9)}),
    "m_corridor_fast": Scenario("m_corridor_fast", "(m) plant: as (j) at speed scale 1.0 (0.5 m/s, 0.8 rad/s)",
                                T_CMD + 80.0, terrain="plant", route=ROUTE_RAMP, speed_scale=1.0,
                                window=(T_CMD, T_CMD + 80.0), criteria={"wp_frac": (0.999, 1e9)}),
    # the 0.10 m pipe C-OBS1 alone (2 m in front of the spawn): more seeds = more gait phases at the pipe
    "n_pipe_dock": Scenario("n_pipe_dock", "(n) plant: C-OBS1 pipe d 0.10 m from 1.9 m, keyboard teleop 0.25 m/s",
                            T_CMD + 22.0, terrain="plant", route=((57.6, -0.1),), spawn=(60.4, 0.0, -0.32, math.pi),
                            window=(T_CMD, T_CMD + 22.0), criteria={"wp_frac": (0.999, 1e9)}),
    # single obstacles of the upper corridor, approached straight (more seeds = more gait phases)
    "p_rubble": Scenario("p_rubble", "(p) plant: C-OBS3 rubble field (x 37 .. 39, y -1.3 .. 0.5, blocks 2-6 cm) "
                         "straight, 0.25 m/s",
                         T_CMD + 34.0, terrain="plant", route=((35.6, 0.0),), spawn=(40.6, 0.0, 0.557, math.pi),
                         window=(T_CMD, T_CMD + 34.0), criteria={"wp_frac": (0.999, 1e9)}),
    "q_pipe08": Scenario("q_pipe08", "(q) plant: C-OBS5 pipe d 0.08 m (x ~22) straight, 0.25 m/s", T_CMD + 22.0,
                         terrain="plant", route=((20.4, 0.0),), spawn=(23.9, 0.0, 0.557, math.pi),
                         window=(T_CMD, T_CMD + 22.0), criteria={"wp_frac": (0.999, 1e9)}),
    "r_rubble_detour": Scenario("r_rubble_detour", "(r) plant: past the rubble field through the north passage "
                                "(free strip y 0.5 .. 1.76), 0.25 m/s", T_CMD + 40.0, terrain="plant",
                                route=((39.7, RUBBLE_PASS_Y), (36.6, RUBBLE_PASS_Y), (35.6, 0.0)),
                                spawn=(40.6, 0.0, 0.557, math.pi),
                                window=(T_CMD, T_CMD + 40.0), criteria={"wp_frac": (0.999, 1e9)}),
    "s_plate": Scenario("s_plate", "(s) plant: C-OBS4 steel plate mu 0.2 (x 30 .. 32.5) straight, 0.25 m/s",
                        T_CMD + 30.0, terrain="plant", route=((28.6, 0.0),), spawn=(34.0, 0.0, 0.557, math.pi),
                        window=(T_CMD, T_CMD + 30.0), criteria={"wp_frac": (0.999, 1e9)}),
    "o_corridor_detour": Scenario("o_corridor_detour", "(o) plant: as (j) but around the north end of the pipe "
                                  "(0.63 m gap at the wall)", T_CMD + 120.0, terrain="plant", route=ROUTE_DETOUR,
                                  window=(T_CMD, T_CMD + 120.0), criteria={"wp_frac": (0.999, 1e9)}),
})
ALIASES = {k.split("_")[0]: k for k in SCENARIOS}


def resolve(names: Sequence[str]) -> List[str]:
    out = []
    for n in names:
        k = ALIASES.get(n, n)
        if k not in SCENARIOS:
            raise KeyError(f"unknown scenario {n!r} (known: {', '.join(SCENARIOS)})")
        out.append(k)
    return out


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
TELEOP_LIMITS = ((-0.3, 0.5), (-0.2, 0.2), (-0.8, 0.8))


def teleop_segments(seed: int, t0: float, t1: float) -> List[Tuple[float, float, Tuple[float, float, float]]]:
    """Keyboard-teleop-like command sequence: 1.5-4 s segments, 15 % stops,
    else uniform vx in [-0.3, 0.5], vy in [-0.2, 0.2], wz in [-0.8, 0.8];
    some segments are pure turns or pure walks (the common key presses)."""
    rng = np.random.default_rng(1000 + int(seed))
    segs, t = [], float(t0)
    while t < t1:
        d = float(rng.uniform(1.5, 4.0))
        u = rng.random()
        if u < 0.15:
            c = (0.0, 0.0, 0.0)
        else:
            c = tuple(float(rng.uniform(*lim)) for lim in TELEOP_LIMITS)
            if u < 0.30:
                c = (0.0, 0.0, c[2])                 # turn key only
            elif u < 0.45:
                c = (c[0], 0.0, 0.0)                 # forward / backward only
        segs.append((t, min(t + d, t1), c))
        t += d
    return segs


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class KeyboardOperator:
    """The person at keyboard_teleop (hold mode, teleop.yaml defaults) who
    watches Gazebo and drives along waypoints.

    Every 1 / rate s (the teleop publish rate) the operator decides which keys
    are held: the turn key (a / d, +-max_wz * scale) while the bearing error to
    the next waypoint is large (pressed at > turn_on, released early enough
    that the hold_timeout run-on ends near zero error), and the forward key
    (w, max_vx * scale) while the error is below walk_max.  A released axis
    keeps its value for ``hold_timeout`` s, exactly like
    teleop_keys.TeleopState in hold mode.  Waypoints count as reached within
    ``reach`` m.  Called by the harness with the ground-truth pose (what the
    person sees), never given to the controller."""

    needs_pose = True

    def __init__(self, waypoints: Sequence[Tuple[float, float]], t_start: float = T_CMD, scale: float = 0.5,
                 max_vx: float = 0.5, max_wz: float = 0.8, hold_timeout: float = 0.6, rate: float = 20.0,
                 reach: float = 0.35, turn_on: float = 0.20, walk_max: float = 0.5):
        self.wps = [tuple(float(v) for v in w) for w in waypoints]
        self.t_start = float(t_start)
        self.vx = float(max_vx) * float(scale)
        self.wz = float(max_wz) * float(scale)
        self.hold_timeout = float(hold_timeout)
        self.period = 1.0 / float(rate)
        self.reach = float(reach)
        self.turn_on = float(turn_on)
        self.walk_max = float(walk_max)
        self.idx = 0
        self.turning = False
        self.reached_t: List[float] = []
        self.done_t: Optional[float] = None
        self._press = {"vx": (-1e9, 0.0), "wz": (-1e9, 0.0)}
        self._next = self.t_start

    def axis(self, k: str, t: float) -> float:
        tp, v = self._press[k]
        return v if t - tp <= self.hold_timeout + 1e-9 else 0.0

    def __call__(self, t: float, pose=None) -> Tuple[float, float, float]:
        if t < self.t_start - 1e-9 or pose is None:
            return (0.0, 0.0, 0.0)
        if t >= self._next - 1e-9:
            self._next = t + self.period
            self._decide(t, pose)
        return (self.axis("vx", t), 0.0, self.axis("wz", t))

    def _decide(self, t: float, pose) -> None:
        pos, rpy = np.asarray(pose[0], dtype=float), np.asarray(pose[1], dtype=float)
        while self.idx < len(self.wps):
            d = np.array(self.wps[self.idx]) - pos[:2]
            if float(np.hypot(*d)) >= self.reach:
                break
            self.reached_t.append(float(t))
            self.idx += 1
        if self.idx >= len(self.wps):
            if self.done_t is None:
                self.done_t = float(t)
            return                                            # keys released -> axes run out
        d = np.array(self.wps[self.idx]) - pos[:2]
        e = _wrap(math.atan2(d[1], d[0]) - float(rpy[2]))
        e_off = max(0.04, 0.5 * self.wz * self.hold_timeout)
        self.turning = abs(e) > (e_off if self.turning else self.turn_on)
        if self.turning:
            self._press["wz"] = (t, math.copysign(self.wz, e))
        if abs(e) < self.walk_max:
            self._press["vx"] = (t, self.vx)

    def result(self) -> Dict:
        n = len(self.wps)
        return {"wp_reached": len(self.reached_t), "wp_total": n,
                "wp_frac": (len(self.reached_t) / n) if n else 1.0,
                "route_done_t": self.done_t, "wp_times": [round(v, 2) for v in self.reached_t]}


def command_fn(sc: Scenario, seed: int) -> Callable[[float], Tuple[float, float, float]]:
    if sc.route:
        return KeyboardOperator(sc.route, t_start=sc.t_cmd + CMD_SHIFT * int(seed), scale=sc.speed_scale)
    if sc.teleop:
        segs = teleop_segments(seed, T_CMD, sc.duration)

        def f(t, _s=segs):
            for (a, b, c) in _s:
                if a <= t < b:
                    return c
            return (0.0, 0.0, 0.0)
        return f
    c = sc.cmd or (0.0, 0.0, 0.0)
    # constant commands start 0.07 s later per seed: the seeds meet obstacles /
    # pushes at different gait phases (otherwise the runs are deterministic)
    t0 = sc.t_cmd + CMD_SHIFT * int(seed)
    return lambda t, _c=c, _t0=t0: _c if t >= _t0 else (0.0, 0.0, 0.0)


def gait_commands(sc: Scenario) -> List[Tuple[float, str]]:
    return [(T_CMD - 0.5, f"gait:{sc.gait}")] if sc.gait else []


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def fall_episodes(t: np.ndarray, fall: np.ndarray, modes, min_dur: float = 0.1) -> List[float]:
    out, start = [], None
    for ti, fi, m in zip(t, fall, modes):
        f = bool(fi) or m == "FALLEN"
        if f and start is None:
            start = ti
        elif not f and start is not None:
            if ti - start >= min_dur:
                out.append(float(start))
            start = None
    if start is not None and t[-1] - start >= min_dur:
        out.append(float(start))
    return out


def evaluate(sc: Scenario, A: Dict[str, np.ndarray], extra: Optional[Dict] = None) -> Dict:
    t = A["t"]
    m: Dict = {"scenario": sc.key, "duration": sc.duration}
    if len(t) == 0:
        m.update({"pass": False, "reason": "no data"})
        return m
    falls = fall_episodes(t, A["fall_metric"], A["mode"])
    m["falls"] = len(falls)
    m["first_fall_t"] = falls[0] if falls else None
    t0, t1 = sc.window
    t_end = falls[0] if falls else t[-1]
    w = (t >= t0) & (t <= min(t1, t_end))
    if w.sum() >= 2:
        p, rpy = A["pos"][w], A["rpy"][w]
        yaw = np.unwrap(rpy[:, 2])
        dt = t[w][-1] - t[w][0]
        d = p[-1, :2] - p[0, :2]
        ym = float(np.mean(yaw))
        c, s = math.cos(ym), math.sin(ym)
        m["vx"] = float((c * d[0] + s * d[1]) / dt)
        m["vy"] = float((-s * d[0] + c * d[1]) / dt)
        m["wz"] = float((yaw[-1] - yaw[0]) / dt)
        m["max_roll_deg"] = float(np.degrees(np.abs(rpy[:, 0]).max()))
        m["max_pitch_deg"] = float(np.degrees(np.abs(rpy[:, 1]).max()))
        # base height above the ground under it (flat worlds: ground 0 = the old absolute height)
        hz = p[:, 2] - A["ground"][w] if "ground" in A and len(A["ground"]) == len(t) else p[:, 2]
        m["height_mean"] = float(hz.mean())
        m["height_min"] = float(hz.min())
        m["height_max"] = float(hz.max())
        m["drift_xy"] = float(np.linalg.norm(d))
        m["progress"] = float(math.cos(float(yaw[0])) * d[0] + math.sin(float(yaw[0])) * d[1])
        m["z_gain"] = float(p[-1, 2] - p[0, 2])
        # lateral deviation from the initial heading line through the window start
        y0 = float(yaw[0])
        rel = p[:, :2] - p[0, :2]
        m["lateral_dev"] = float(np.abs(-math.sin(y0) * rel[:, 0] + math.cos(y0) * rel[:, 1]).max())
        tau = np.abs(A["tau"][w])
        ratio = tau / EFFORT
        m["max_tau_ratio"] = float(ratio.max())
        m["sat_frac"] = float(np.mean(ratio >= 0.99))
    mpc = A["mpc_ms"][np.isfinite(A["mpc_ms"])] if len(A["mpc_ms"]) else np.array([])
    if mpc.size:
        m["mpc_ms_mean"] = float(mpc.mean())
        m["mpc_ms_p95"] = float(np.percentile(mpc, 95))
    if extra:
        m.update(extra)
    reasons = []
    if falls:
        reasons.append(f"fell at {falls[0]:.2f} s")
    stood = any(md in ("STAND", "WALK") for md in A["mode"])
    if not stood:
        reasons.append("never reached STAND")
    for k, (lo, hi) in sc.criteria.items():
        v = m.get(k)
        if v is None or not (lo <= v <= hi):
            reasons.append(f"{k}={v if v is None else round(v, 3)} not in [{lo}, {hi}]")
    m["pass"] = not reasons
    m["reason"] = "; ".join(reasons)
    return m


def with_duration(sc: Scenario, duration: float) -> Scenario:
    """Copy of a scenario capped at ``duration`` s (quick checks / tests)."""
    from dataclasses import replace
    d = min(float(duration), sc.duration)
    return replace(sc, duration=d, window=(min(sc.window[0], d), min(sc.window[1], d)))


# ---------------------------------------------------------------------------
# mission in the loop: novel (unmapped) obstacles
# ---------------------------------------------------------------------------
class MissionOperator:
    """The REAL MissionLogic (plant_simulation.mission_logic, obstacle layer
    included) driving the robot in the harness, like mission_manager +
    cmd_vel_mux do on Ubuntu:

    * every 1 / rate s of sim time: pose from the harness odometry (= /odom;
      ground truth as fallback), locomotion mode from the controller log,
      the latest terrain_mapper cloud (harness ``elevation_cloud``, i.e. the
      /elevation_map/points payload) through the same conversion as the node
      (EvidenceFrame.from_elevation_cloud), synthetic gauge readings while
      INSPECTING (no camera headless), ``step``;
    * /locomotion/command strings go into ``harness.commands``;
    * mux semantics: the last /cmd_vel/patrol is returned while it is at most
      ``mux_timeout`` s old, zero otherwise.

    Ground truth is used only for the metrics (box clearance, walked path)."""

    needs_pose = True

    def __init__(self, plan: Dict, params: Optional[Dict] = None, boxes: Sequence = (), t_start: float = 3.5,
                 mission_cmd: str = "patrol", rate: float = 10.0, mux_timeout: float = 0.5,
                 nav_map_yaml: Optional[str] = None, layout=None):
        import os
        from dusan_common.grid_map import OccupancyGridMap
        from dusan_common.layout import Layout
        from dusan_common.obstacle_sim import Box
        from plant_simulation.mission_logic import MissionLogic
        from . import world as W
        self.layout = layout if layout is not None else Layout.load(W.LAYOUT)
        p = dict(min_start_time=1.0, dwell_time=1.5, min_dwell_time=0.5, scan_on_miss=False)
        p.update(params or {})
        self.m = MissionLogic(p, self.layout)
        self.m.set_map(OccupancyGridMap.load(nav_map_yaml or os.path.join(W.ROOT, "simulation", "maps",
                                                                         "plant_map.yaml")))
        self.plan = plan
        self.m.set_plan(plan)
        self.vp_ids = [str(v["gauge_id"]) for v in plan.get("viewpoints", [])]
        self.boxes = [Box(tuple(float(v) for v in b["center"][:2]), tuple(float(v) for v in b["size"][:2]),
                          float(b.get("yaw", 0.0))) for b in boxes]
        self.t_start = float(t_start)
        self.mission_cmd = mission_cmd
        self.period = 1.0 / float(rate)
        self.mux_timeout = float(mux_timeout)
        self.h = None
        self.bound = False
        self._next = 0.0
        self._phase: Optional[float] = None           # [s] tick phase after the first call (set per seed)
        self._started = False
        self._cmd = None
        self._cmd_t = -1e9
        self._cloud_t = None
        self.frames = 0
        self.loco: List[Tuple[float, str]] = []
        self.reached_t: Dict[str, float] = {}
        self.events: List[Tuple[float, str, str]] = []
        self.min_box_clearance = math.inf
        self.traj: List[Tuple[float, float]] = []
        self.path_len = 0.0                           # [m] walked (ground truth) from the spawn
        self.trace: List[Tuple[float, ...]] = []
        self.plans: List[Tuple[float, List[Tuple[float, float]]]] = []
        self.plan_changes = 0
        self.ticks_nav = 0
        self.ticks_slow = 0
        self.n_detours = 0
        self.n_obstacle_replans = 0
        self._sides: List[str] = []

    def bind(self, harness) -> None:
        self.h = harness
        self.bound = True

    def _odom_pose(self, gt):
        h = self.h
        odom = None
        if h is not None:
            lock = getattr(h, "_lock", None)
            if lock is not None:
                with lock:
                    odom = getattr(h, "_odom", None)
            else:
                odom = getattr(h, "_odom", None)
        if odom is not None:
            return float(odom[1][0]), float(odom[1][1]), float(odom[2][2])
        if gt is not None:
            return float(gt[0][0]), float(gt[0][1]), float(gt[1][2])
        return None

    def _tick(self, t: float, gt) -> None:
        from dusan_common.obstacle_layer import EvidenceFrame
        m, h = self.m, self.h
        pose = self._odom_pose(gt)
        if pose is not None:
            m.set_pose(pose[0], pose[1], pose[2], t)
        if h is not None:
            modes = getattr(h, "log", {}).get("mode") or []
            if modes:
                m.on_locomotion_state({"mode": str(modes[-1])}, t)
            ct = getattr(h, "elevation_cloud_t", None)
            if ct is not None and ct != self._cloud_t and getattr(h, "elevation_cloud", None) is not None:
                self._cloud_t = ct
                if m.obstacle_frame_due(t):                # thinned while not driving, like mission_manager
                    frame = EvidenceFrame.from_elevation_cloud(h.elevation_cloud, "odom",
                                                               float(m.p["obstacle_step_height"]),
                                                               int(m.p["obstacle_step_window"]))
                    m.on_obstacle_frame(frame, t)
                    self.frames += 1
        if not self._started and t >= self.t_start - 1e-9:
            self._started = True
            m.command(self.mission_cmd, t)
        if m.base == "INSPECTING" and m.current is not None:
            gid = m.current.gauge_id
            self.reached_t.setdefault(gid, float(t))
            m.on_gauge_readings({"stamp": round(t, 3), "readings": [{"id": gid, "value": 1.0, "confidence": 0.9}]},
                                t)
        out = m.step(t)
        for c in out.locomotion:
            self.loco.append((t, c))
            if h is not None and hasattr(h, "commands"):
                h.commands.append((t + 1e-3, c))
        if out.cmd_vel is not None:
            self._cmd = tuple(float(v) for v in out.cmd_vel)
            self._cmd_t = t
        if out.path:
            self.plan_changes += 1
            if len(self.plans) < 20:                  # (t, every 3rd vertex) for diagnosis
                P = [(round(float(q[0]), 2), round(float(q[1]), 2)) for q in out.path]
                self.plans.append((round(float(t), 2), P[::3] + ([P[-1]] if (len(P) - 1) % 3 else [])))
        for lvl, text in out.events:
            if len(self.events) < 60:
                self.events.append((round(t, 2), lvl, text))
            self._count_event(text)
        if m.base == "PATROL" and m.phase == "navigate":
            self.ticks_nav += 1
            self.ticks_slow += int(m.obstacles.scale < 1.0)
        if gt is not None:
            x, y = float(gt[0][0]), float(gt[0][1])
            if self.traj:
                self.path_len += math.hypot(x - self.traj[-1][0], y - self.traj[-1][1])
            self.traj.append((x, y))
            c = out.cmd_vel if out.cmd_vel is not None else (math.nan,) * 3
            ft = m.obstacles.free_travel
            # per tick: t, ground-truth x / y / yaw, governor scale + confirmed free travel, /cmd_vel/patrol
            self.trace.append((float(t), x, y, float(gt[1][2]), float(m.obstacles.scale),
                               float(ft) if math.isfinite(ft) else -1.0, float(c[0]), float(c[1]), float(c[2])))
            for b in self.boxes:
                self.min_box_clearance = min(self.min_box_clearance, float(b.distance(x, y)))

    def _count_event(self, text: str) -> None:
        if "detour left" in text or "detour right" in text:
            self.n_detours += 1
            self._sides.append("left" if "detour left" in text else "right")
        if "path blocked by a detected obstacle" in text:
            self.n_obstacle_replans += 1

    def __call__(self, t: float, gt=None) -> Tuple[float, float, float]:
        if self._phase is not None:                   # tick phase relative to the first call (seeds)
            self._next, self._phase = t + self._phase, None
        if t >= self._next - 1e-9:
            self._next = t + self.period
            self._tick(t, gt)
        if self._cmd is not None and t - self._cmd_t <= self.mux_timeout + 1e-9:
            return self._cmd
        return (0.0, 0.0, 0.0)

    def result(self) -> Dict:
        from dusan_common.terrain_cost import debris_crossed
        m = self.m
        n = len(self.vp_ids)
        reached = [g for g in self.vp_ids if g in self.reached_t]
        sides = sorted(set(self._sides))
        side = sides[0] if len(sides) == 1 else ("mixed" if sides else None)
        debris = debris_crossed(self.traj, self.layout) if self.traj else []
        st = m.state_dict(None)["obstacle"]
        res = {"reached": (len(reached) / n) if n else 1.0,
               "t_reach": max(self.reached_t.values()) if reached and len(reached) == n else None,
               "min_box_clearance": self.min_box_clearance if math.isfinite(self.min_box_clearance) else None,
               "detours": float(self.n_detours), "obstacle_replans": float(self.n_obstacle_replans),
               "stuck_replans": float(m.total_replans), "missed": list(m.missed), "missed_n": float(len(m.missed)),
               "missed_stuck": float(any("stuck" in str((m.results.get(g) or {}).get("reason", ""))
                                         for g in m.missed)),
               "path_len": self.path_len, "debris": debris, "debris_n": float(len(debris)), "side": side,
               "side_left": 1.0 if side == "left" else 0.0,
               "slow_frac": (self.ticks_slow / self.ticks_nav) if self.ticks_nav else 0.0,
               "obstacle_errors": float(st["errors"]), "frames": self.frames, "plan_changes": self.plan_changes,
               "final_state": m.base, "done": 1.0 if m.base in ("DONE", "RTH", "DOCKED") else 0.0,
               "events": self.events, "plans": self.plans}
        for g in self.vp_ids:
            res[f"reached_{g}"] = 1.0 if g in self.reached_t else 0.0
        return res


@dataclass
class MissionScenario:
    key: str
    title: str
    spawn: Tuple[float, float, float, float]
    viewpoints: Tuple                                 # ((gauge_id, x, y, yaw), ...) in visiting order
    extra: Tuple = ()                                 # novel boxes {name, center, size, yaw, floor_z}
    duration: float = 60.0
    criteria: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    mission_cmd: str = "patrol"
    t_start: float = 3.5                              # [s] 'patrol' sent (the controller stands up first)
    control: Optional[str] = None                     # same route without the box (time reference)

    def plan(self) -> Dict:
        return {"viewpoints": [{"gauge_id": g, "x": x, "y": y, "yaw": yaw} for g, x, y, yaw in self.viewpoints],
                "order": [v[0] for v in self.viewpoints], "legs": []}

    def scenario(self) -> Scenario:
        return Scenario(self.key, self.title, self.duration, terrain="plant", spawn=self.spawn,
                        window=(0.0, self.duration), criteria=dict(self.criteria))

    def operator(self, params: Optional[Dict] = None, seed: int = 0) -> MissionOperator:
        """The mission starts CMD_SHIFT later and ticks with a phase offset per
        seed (the physics is deterministic: the seeds meet the obstacle at
        different gait phases; see also :meth:`spawn_for`)."""
        op = MissionOperator(self.plan(), params, self.extra, t_start=self.t_start + CMD_SHIFT * int(seed),
                             mission_cmd=self.mission_cmd)
        op._phase = (0.037 * int(seed)) % op.period
        return op

    def spawn_for(self, seed: int = 0) -> Tuple[float, float, float, float]:
        """Spawn pose of a seed: seed 0 the nominal one, the others moved by up
        to SPAWN_JITTER m / rad (deterministic physics: otherwise the seeds
        repeat the same run)."""
        if not int(seed):
            return self.spawn
        rng = np.random.default_rng(1000 + int(seed))
        dx, dy, dyaw = rng.uniform(-1.0, 1.0, 3) * np.array([SPAWN_JITTER, SPAWN_JITTER, SPAWN_JITTER])
        x, y, z, yaw = self.spawn
        return (float(x + dx), float(y + dy), float(z), float(yaw + dyaw))


Z_LOWER, Z_UPPER = -0.88, 0.0                         # corridor floors (lower: dock / pallet, upper: past the ramp)
SPAWN_DZ = 0.557                                      # spawn height above the floor
SPAWN_JITTER = 0.04                                   # [m] / [rad] spawn perturbation of the seeds > 0
_INF = 1e9


def _box(x, y, sx, sy, sz, floor_z, name="novel_box_0"):
    return {"name": name, "center": [x, y], "size": [sx, sy, sz], "yaw": 0.0, "floor_z": floor_z}


_OK = {"reached": (1.0, 1.0), "stuck_replans": (0.0, 0.0), "obstacle_errors": (0.0, 0.0)}
MISSION_SCENARIOS: Dict[str, MissionScenario] = {
    "r_control_leg1": MissionScenario(
        "r_control_leg1", "(r) mission: dock -> G08 without a box (time reference of t, no false detours)",
        (61.9, 0.0, Z_LOWER + SPAWN_DZ, math.pi), (("G08", 57.0, 0.15, -math.pi / 2),), duration=45.0,
        criteria=dict(_OK, detours=(0.0, 0.0), obstacle_replans=(0.0, 0.0), debris_n=(0.0, 0.0))),
    "t_box_start": MissionScenario(
        "t_box_start", "(t) mission: dock -> G08, novel box on the A* leg seen during the stand-up",
        (61.9, 0.0, Z_LOWER + SPAWN_DZ, math.pi), (("G08", 57.0, 0.15, -math.pi / 2),),
        extra=(_box(60.45, 0.65, 0.6, 0.6, 0.5, Z_LOWER),), duration=60.0, control="r_control_leg1",
        criteria=dict(_OK, min_box_clearance=(0.40, _INF), debris_n=(0.0, 0.0))),
    "u_box_lower": MissionScenario(
        "u_box_lower", "(u) mission: G08 -> V2 south of the pallet, novel box discovered at ~1.2 m",
        (56.9, 0.0, Z_LOWER + SPAWN_DZ, math.pi), (("V2", 52.2, -0.4, math.pi),),
        extra=(_box(54.2, -0.30, 0.5, 0.5, 0.5, Z_LOWER),), duration=50.0, control="z_control_lower",
        criteria=dict(_OK, detours=(1.0, _INF), side_left=(1.0, 1.0), min_box_clearance=(0.40, _INF))),
    "v_box_upper": MissionScenario(
        "v_box_upper", "(v) mission: upper corridor -> V1, novel box 0.6 x 0.6 x 0.5 m on the route",
        (46.5, 0.72, Z_UPPER + SPAWN_DZ, math.pi), (("V1", 39.8, 0.9, math.pi),),
        extra=(_box(43.0, 0.72, 0.6, 0.6, 0.5, Z_UPPER),), duration=50.0, control="x_control_upper",
        criteria=dict(_OK, detours=(1.0, _INF), min_box_clearance=(0.40, _INF))),
    "w_wall_upper": MissionScenario(
        "w_wall_upper", "(w) mission: as (v) but a 0.4 x 3.6 m wall closes the corridor -> skip, DONE",
        (46.5, 0.72, Z_UPPER + SPAWN_DZ, math.pi), (("V1", 39.8, 0.9, math.pi),),
        extra=(_box(43.0, 0.0, 0.4, 3.6, 0.6, Z_UPPER),), duration=100.0,
        criteria={"missed_stuck": (1.0, 1.0), "done": (1.0, 1.0), "min_box_clearance": (0.45, _INF),
                  "obstacle_replans": (0.0, 3.0), "obstacle_errors": (0.0, 0.0)}),
    "x_control_upper": MissionScenario(
        "x_control_upper", "(x) mission: (v) without the box (time reference, no false detours)",
        (46.5, 0.72, Z_UPPER + SPAWN_DZ, math.pi), (("V1", 39.8, 0.9, math.pi),), duration=45.0,
        criteria=dict(_OK, detours=(0.0, 0.0), obstacle_replans=(0.0, 0.0))),
    "y_route_no_box": MissionScenario(
        "y_route_no_box", "(y) mission: dock -> G08 -> G07 (C-OBS1 gap, pallet, ramp, upper corridor), no box: "
        "false-positive check", (61.9, 0.0, Z_LOWER + SPAWN_DZ, math.pi),
        (("G08", 57.0, 0.15, -math.pi / 2), ("G07", 26.0, -0.15, math.pi / 2)), duration=240.0,
        criteria=dict(_OK, detours=(0.0, 0.0), obstacle_replans=(0.0, 0.0), slow_frac=(0.0, 0.02))),
    "z_control_lower": MissionScenario(
        "z_control_lower", "(z) mission: (u) without the box (time reference)",
        (56.9, 0.0, Z_LOWER + SPAWN_DZ, math.pi), (("V2", 52.2, -0.4, math.pi),), duration=40.0,
        criteria=dict(_OK, detours=(0.0, 0.0), obstacle_replans=(0.0, 0.0))),
}
MISSION_ALIASES = {k.split("_")[0]: k for k in MISSION_SCENARIOS}


def resolve_mission(names: Sequence[str]) -> List[str]:
    out = []
    for n in names:
        k = MISSION_ALIASES.get(n, n)
        if k not in MISSION_SCENARIOS:
            raise KeyError(f"unknown mission scenario {n!r} (known: {', '.join(MISSION_SCENARIOS)})")
        out.append(k)
    return out
