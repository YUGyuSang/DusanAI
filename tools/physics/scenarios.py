"""
Physics-check scenarios (a) .. (j) with numeric pass criteria.

Time line of every scenario: spawn at t = 0 with straight legs (knees at the
-0.2 rad limit, gz ``initial_position``) at floor + 0.56 m, the controller
starts after the node's start-up gate (0.5 s), stands up in 2.5 s and is in
STAND at ~3.0 s.  Commands start at ``T_WALK`` = 4.0 s.

Fall definition (task): |roll| or |pitch| > 55 deg or base height above the
terrain under the base < 0.12 m; a fall episode is counted when the
condition holds for >= 0.1 s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from dusan_common import robot_model as rm

from . import terrain as T
from .sim import PhysicsSim, Push, SimConfig

T_WALK = 4.0
EFFORT = np.array([45.0, 45.0, 60.0] * 4)
DEG = math.pi / 180.0


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def fall_episodes(A, t0: float = 0.0, min_dur: float = 0.1) -> List[float]:
    """Start times of fall episodes (condition held >= min_dur) after t0."""
    t, f = A["t"], A["fall_metric"].astype(bool)
    out, start = [], None
    for ti, fi in zip(t, f):
        if ti < t0:
            continue
        if fi and start is None:
            start = ti
        elif not fi and start is not None:
            if ti - start >= min_dur:
                out.append(float(start))
            start = None
    if start is not None and t[-1] - start >= min_dur:
        out.append(float(start))
    return out


def window(A, t0: float, t1: float):
    return (A["t"] >= t0) & (A["t"] <= t1)


def mean_speed(A, t0: float, t1: float) -> Tuple[float, float, float]:
    """(vx, vy) in the mean heading frame from the displacement, and the mean yaw rate."""
    w = window(A, t0, t1)
    if w.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    p = A["pos"][w]
    yaw = np.unwrap(A["rpy"][w][:, 2])
    dt = A["t"][w][-1] - A["t"][w][0]
    d = p[-1, :2] - p[0, :2]
    ym = float(np.mean(yaw))
    c, s = math.cos(ym), math.sin(ym)
    return float((c * d[0] + s * d[1]) / dt), float((-s * d[0] + c * d[1]) / dt), float((yaw[-1] - yaw[0]) / dt)


def torque_stats(A, t0: float = 0.0, t1: float = 1e9) -> Dict:
    w = window(A, t0, t1)
    tau = np.abs(A["tau"][w])
    if len(tau) == 0:
        return {}
    ratio = tau / EFFORT
    return {
        "max_ratio_hip_x": float(ratio[:, 0::3].max()), "max_ratio_hip_y": float(ratio[:, 1::3].max()),
        "max_ratio_knee": float(ratio[:, 2::3].max()),
        "saturation_frac": float(np.mean(ratio >= 0.99)),
        "rms_ratio_knee": float(np.sqrt(np.mean(ratio[:, 2::3] ** 2))),
        "rms_ratio_hip_y": float(np.sqrt(np.mean(ratio[:, 1::3] ** 2))),
    }


def attitude_stats(A, t0: float, t1: float) -> Dict:
    w = window(A, t0, t1)
    if w.sum() == 0:
        return {}
    r, p = A["rpy"][w][:, 0], A["rpy"][w][:, 1]
    h = A["body_h"][w]
    return {"max_roll_deg": float(np.degrees(np.abs(r).max())), "max_pitch_deg": float(np.degrees(np.abs(p).max())),
            "height_mean": float(h.mean()), "height_min": float(h.min()), "height_max": float(h.max()),
            "height_std": float(h.std())}


def contact_stats(A, t0: float, t1: float) -> Dict:
    w = window(A, t0, t1)
    if w.sum() == 0:
        return {}
    c = A["contacts"][w]
    slip = A["foot_slip"][w]
    inc = slip[c]
    return {"no_contact_frac": float(np.mean(~c)),
            "stance_slip_mean": float(inc.mean()) if inc.size else float("nan"),
            "stance_slip_p95": float(np.percentile(inc, 95)) if inc.size else float("nan")}


def first_mode_time(A, mode: str, t0: float = 0.0) -> float:
    for ti, m in zip(A["t"], A["mode"]):
        if ti >= t0 and m == mode:
            return float(ti)
    return float("nan")


def mpc_stats(sim: PhysicsSim) -> Dict:
    st = np.asarray(sim.worker.solve_times, dtype=float) * 1e3
    if st.size == 0:
        return {"mpc_solves": 0}
    return {"mpc_solves": int(st.size), "mpc_ms_mean": float(st.mean()), "mpc_ms_p95": float(np.percentile(st, 95)),
            "mpc_ms_max": float(st.max()), "mpc_rejected": int(sim.ctrl.mpc_rejected)}


def vel_tracking(A, t0: float, t1: float, axis: int = 0, avg: float = 0.4) -> float:
    """RMS error of the gait-cycle averaged body velocity vs the shaped command."""
    w = window(A, t0, t1)
    if w.sum() < 10:
        return float("nan")
    v = A["vel_body"][w][:, axis] if axis < 2 else A["omega"][w][:, 2]
    c = A["cmd_f"][w][:, axis]
    n = max(1, int(round(avg / np.median(np.diff(A["t"][w])))))
    k = np.ones(n) / n
    vs = np.convolve(v, k, mode="valid")
    cs = np.convolve(c, k, mode="valid")
    return float(np.sqrt(np.mean((vs - cs) ** 2)))


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    key: str
    title: str
    duration: float
    terrain: Callable[[], T.Terrain]
    setup: Callable[[PhysicsSim, int], None]
    evaluate: Callable[[PhysicsSim, Dict], Tuple[Dict, bool, str]]
    perception: str = "depth"
    payload: Optional[dict] = None
    spawn: Tuple[float, float, float, float] = (0.0, 0.0, 0.56, 0.0)
    spawn_rpy: Tuple[float, float] = (0.0, 0.0)
    spawn_q: Optional[Tuple[float, ...]] = None
    required: bool = True
    yaw_jitter: float = 0.05
    stop_x: Optional[float] = None


def _cmd_after(t0: float, v: Tuple[float, float, float], t1: float = 1e9):
    def f(t):
        return v if t0 <= t < t1 else (0.0, 0.0, 0.0)
    return f


# (a) stand -----------------------------------------------------------------
def _setup_stand(sim: PhysicsSim, seed: int) -> None:
    sim.cmd_fn = lambda t: (0.0, 0.0, 0.0)


def _eval_stand(sim: PhysicsSim, A) -> Tuple[Dict, bool, str]:
    t_st = first_mode_time(A, "STAND")
    m: Dict = {"t_stand": t_st}
    ok = math.isfinite(t_st)
    if ok:
        t0, t1 = t_st + 0.5, t_st + 5.5
        m.update(attitude_stats(A, t0, t1))
        m.update(torque_stats(A, t0, t1))
        m.update(contact_stats(A, t0, t1))
        w = window(A, t0, t1)
        m["drift_mm"] = float(np.linalg.norm(A["pos"][w][-1, :2] - A["pos"][w][0, :2]) * 1e3)
        ok = (abs(m["height_mean"] - 0.34) <= 0.03 and m["height_min"] >= 0.31 and m["height_max"] <= 0.37
              and m["max_roll_deg"] < 3.0 and m["max_pitch_deg"] < 3.0 and A["t"][-1] >= t1 - 0.05)
    m["falls"] = len(fall_episodes(A))
    m.update({"standup_" + k: v for k, v in torque_stats(A, 0.5, 3.2).items()})
    ok = ok and m["falls"] == 0
    return m, ok, "STAND reached; 5 s window: height 0.34+-0.03, |roll|,|pitch| < 3 deg, no fall"


# (b)/(c)/(d) walking --------------------------------------------------------
def _setup_walk(cmd, gait: Optional[str], dur: float = 10.0):
    def setup(sim: PhysicsSim, seed: int) -> None:
        sim.cmd_fn = _cmd_after(T_WALK, cmd, T_WALK + dur)
        if gait:
            sim.commands.append((3.5, f"gait:{gait}"))
    return setup


def _eval_walk(cmd, axis: int, dur: float = 10.0, tol: float = 0.2, gait: Optional[str] = None):
    def ev(sim: PhysicsSim, A) -> Tuple[Dict, bool, str]:
        t0, t1 = T_WALK + 2.0, T_WALK + dur
        vx, vy, wz = mean_speed(A, t0, t1)
        m = {"vx_mean": vx, "vy_mean": vy, "wz_mean": wz, "cmd": list(cmd)}
        m.update(attitude_stats(A, t0, t1))
        m.update(torque_stats(A, T_WALK, t1))
        m.update(contact_stats(A, t0, t1))
        m.update(mpc_stats(sim))
        m["track_rms_vx"] = vel_tracking(A, t0, t1, 0)
        m["track_rms_vy"] = vel_tracking(A, t0, t1, 1)
        m["track_rms_wz"] = vel_tracking(A, t0, t1, 2)
        falls = fall_episodes(A)
        m["falls"] = len(falls)
        m["controller_falls"] = int(sim.ctrl.recovery.fall_count)
        w = window(A, T_WALK, t1)
        gaits = A["gait"][w]
        m["gait_frac"] = {g: float(np.mean(gaits == g)) for g in ("stand", "trot", "crawl")}
        m["xy_end"] = [float(v) for v in A["pos"][-1][:2]]
        target = cmd[axis] if axis < 2 else cmd[2]
        meas = (vx, vy, wz)[axis]
        ok = (len(falls) == 0 and math.isfinite(meas) and abs(meas - target) <= tol * abs(target)
              and A["t"][-1] >= t1 - 0.05)
        if gait:
            ok = ok and m["gait_frac"].get(gait, 0.0) > 0.8
        name = ("vx", "vy", "wz")[axis]
        return m, ok, f"no fall; mean {name} within {int(tol * 100)} % of {target:g}"
    return ev


# (e) pipe + step --------------------------------------------------------------
def _setup_obstacles(sim: PhysicsSim, seed: int) -> None:
    sim.cmd_fn = _cmd_after(T_WALK, (0.2, 0.0, 0.0))


def _eval_course(goal_x: float, label: str):
    def ev(sim: PhysicsSim, A) -> Tuple[Dict, bool, str]:
        x = A["pos"][:, 0]
        reached = np.nonzero(x >= goal_x)[0]
        t_goal = float(A["t"][reached[0]]) if reached.size else float("nan")
        falls = fall_episodes(A)
        m = {"t_goal": t_goal, "x_max": float(x.max()), "falls": len(falls),
             "controller_falls": int(sim.ctrl.recovery.fall_count)}
        t1 = t_goal if math.isfinite(t_goal) else float(A["t"][-1])
        m.update(attitude_stats(A, T_WALK, t1))
        m.update(torque_stats(A, T_WALK, t1))
        m.update(contact_stats(A, T_WALK, t1))
        m.update(mpc_stats(sim))
        w = window(A, T_WALK, t1)
        gaits = A["gait"][w]
        m["gait_frac"] = {g: float(np.mean(gaits == g)) for g in ("stand", "trot", "crawl")}
        cls = [c for c in A["terrain_class"][w] if c]
        m["terrain_classes"] = sorted(set(cls))
        if math.isfinite(t_goal):
            m["avg_speed"] = float((goal_x - A["pos"][window(A, T_WALK, T_WALK + 0.02)][0, 0]) / (t_goal - T_WALK))
        ok = math.isfinite(t_goal) and len(falls) == 0
        return m, ok, f"no fall; base passes x = {goal_x:g} m ({label})"
    return ev


# (f) push --------------------------------------------------------------------
PUSH_T = T_WALK + 4.0


def _setup_push(sim: PhysicsSim, seed: int) -> None:
    sim.cmd_fn = _cmd_after(T_WALK, (0.4, 0.0, 0.0), T_WALK + 10.0)
    sim.commands.append((3.5, "gait:trot"))
    rng = np.random.default_rng(1000 + seed)
    sign = 1.0 if seed % 2 == 0 else -1.0
    t = PUSH_T + float(rng.uniform(0.0, 0.4))          # random phase within a trot cycle
    sim.pushes.append(Push(t, (0.0, sign * 75.0, 0.0), 0.2))    # 15 N s, push_duration of the node (0.2 s)


def _eval_push(sim: PhysicsSim, A) -> Tuple[Dict, bool, str]:
    p = sim.pushes[0]
    falls = fall_episodes(A)
    t0, t1 = p.t, p.t + 3.0
    m = {"push_t": p.t, "push_impulse_Ns": float(np.hypot(*p.force_body[:2]) * p.duration),
         "push_dir": float(np.sign(p.force_body[1])), "falls": len(falls),
         "controller_falls": int(sim.ctrl.recovery.fall_count)}
    m.update(attitude_stats(A, t0, t1))
    w = window(A, t0, t1)
    vy = A["vel_body"][w][:, 1]
    m["peak_vy"] = float(np.abs(vy).max()) if vy.size else float("nan")
    # recovery time: lateral speed back below 0.15 m/s
    rec = float("nan")
    if vy.size:
        tt = A["t"][w]
        big = np.nonzero(np.abs(vy) > 0.15)[0]
        rec = float(tt[big[-1]] - p.t) if big.size else 0.0
    m["recovery_s"] = rec
    vx, vy_m, _ = mean_speed(A, p.t + 3.0, T_WALK + 10.0)
    m["vx_after"] = vx
    m.update(torque_stats(A, t0, t1))
    m.update(mpc_stats(sim))
    ok = len(falls) == 0 and A["t"][-1] >= T_WALK + 10.0 - 0.05
    return m, ok, "15 N s lateral push (75 N for 0.2 s = the node push_duration) while trotting 0.4 m/s: no fall"


# (g) payload ---------------------------------------------------------------
def _setup_payload(sim: PhysicsSim, seed: int) -> None:
    sim.cmd_fn = _cmd_after(T_WALK + 3.0, (0.4, 0.0, 0.0), T_WALK + 11.0)
    sim.commands.append((3.5, "gait:trot"))


def _eval_payload(sim: PhysicsSim, A) -> Tuple[Dict, bool, str]:
    t_st = first_mode_time(A, "STAND")
    m: Dict = {"t_stand": t_st}
    st = attitude_stats(A, T_WALK, T_WALK + 3.0)
    m.update({"stand_" + k: v for k, v in st.items()})
    t0, t1 = T_WALK + 5.0, T_WALK + 11.0
    vx, vy, wz = mean_speed(A, t0, t1)
    m.update({"vx_mean": vx, "vy_mean": vy})
    m.update(attitude_stats(A, t0, t1))
    m.update(torque_stats(A, 0.0, t1))
    m.update(mpc_stats(sim))
    falls = fall_episodes(A)
    m["falls"] = len(falls)
    ok = (len(falls) == 0 and math.isfinite(t_st) and abs(st.get("height_mean", 0) - 0.34) <= 0.03
          and abs(vx - 0.4) <= 0.08 and A["t"][-1] >= t1 - 0.05)
    return m, ok, "+3 kg on the trunk: stand height 0.34+-0.03, trot 0.4 m/s within 20 %, no fall"


# (h) fall + recovery ----------------------------------------------------------
def _setup_side(sim: PhysicsSim, seed: int) -> None:
    sim.cmd_fn = lambda t: (0.0, 0.0, 0.0)


def _setup_bigpush(force: float):
    """Lateral push while standing, ``force`` [N] for 0.2 s."""
    def setup(sim: PhysicsSim, seed: int) -> None:
        sim.cmd_fn = lambda t: (0.0, 0.0, 0.0)
        sign = 1.0 if seed % 2 == 0 else -1.0
        sim.pushes.append(Push(T_WALK + 0.5, (0.0, sign * force, 0.0), 0.2))
    return setup


def _eval_recovery(sim: PhysicsSim, A) -> Tuple[Dict, bool, str]:
    ev = sim.events
    rec_t = [e["t"] for e in ev if e.get("event") == "recovered"]
    fall_t = [e["t"] for e in ev if e.get("event") == "fall"]
    rc = sim.ctrl.recovery
    m = {"controller_falls": int(rc.fall_count), "recovery_attempts": int(rc.attempts_total),
         "recoveries": int(rc.successes), "assisted_resets": int(rc.assisted_resets),
         "t_fall": fall_t[0] if fall_t else float("nan"), "t_recovered": rec_t[0] if rec_t else float("nan"),
         "fall_episodes_metric": len(fall_episodes(A))}
    # recovery state sequence
    seq = []
    for s in A["rec_state"]:
        if not seq or seq[-1] != s:
            seq.append(str(s))
    m["state_sequence"] = seq
    t_end = float(A["t"][-1])
    tail = window(A, t_end - 2.0, t_end)
    m["final_mode"] = str(A["mode"][-1])
    m.update({"final_" + k: v for k, v in attitude_stats(A, t_end - 2.0, t_end).items()})
    m.update(torque_stats(A, 0.0, t_end))
    upright = (m["final_mode"] == "STAND" and m.get("final_max_roll_deg", 99) < 10
               and m.get("final_max_pitch_deg", 99) < 10 and m.get("final_height_mean", 0) > 0.28)
    m["time_to_recover"] = (m["t_recovered"] - m["t_fall"]) if rec_t and fall_t else float("nan")
    ok = bool(rec_t) and m["assisted_resets"] == 0 and upright and m["controller_falls"] >= 1
    return m, ok, "fall detected -> self-righting -> upright STAND (no assisted reset)"


# (i) slippery plate ------------------------------------------------------------
def _setup_plate(cmd, gait):
    def setup(sim: PhysicsSim, seed: int) -> None:
        sim.cmd_fn = _cmd_after(T_WALK, cmd)
        sim.commands.append((3.5, f"gait:{gait}"))
    return setup


def _eval_plate(goal_x: float, x0: float, x1: float):
    def ev(sim: PhysicsSim, A) -> Tuple[Dict, bool, str]:
        m, ok, _ = _eval_course(goal_x, "plate")(sim, A)
        on = (A["pos"][:, 0] > x0 + 0.3) & (A["pos"][:, 0] < x1 - 0.3)
        if on.any():
            c = A["contacts"][on]
            slip = A["foot_slip"][on][c]
            m["plate_slip_mean"] = float(slip.mean()) if slip.size else float("nan")
            m["plate_slip_p95"] = float(np.percentile(slip, 95)) if slip.size else float("nan")
            dt = float(np.median(np.diff(A["t"])))
            m["plate_slip_dist"] = float(np.sum(A["foot_slip"][on] * A["contacts"][on]) * dt)
            m.update({"plate_" + k: v for k, v in attitude_stats(A, A["t"][on][0], A["t"][on][-1]).items()})
            tt = A["t"][on]
            m["plate_speed"] = float((A["pos"][on][-1, 0] - A["pos"][on][0, 0]) / max(tt[-1] - tt[0], 1e-3))
        return m, ok, "best effort: cross the mu 0.15 plate (record slip / falls)"
    return ev


# ---------------------------------------------------------------------------
def _plate_terrain():
    return T.plate(x0=1.0, length=4.0, mu=0.15)


SIDE_Q = (0.0, 0.0, -0.2) * 4


def build_scenarios() -> Dict[str, Scenario]:
    S: Dict[str, Scenario] = {}
    S["a_stand"] = Scenario("a_stand", "(a) stand-up + 5 s stand", 9.0, T.flat, _setup_stand, _eval_stand)
    S["b_trot"] = Scenario("b_trot", "(b) trot 0.4 m/s, 10 s", T_WALK + 10.5, T.flat,
                           _setup_walk((0.4, 0.0, 0.0), "trot"), _eval_walk((0.4, 0.0, 0.0), 0, gait="trot"))
    S["c_crawl"] = Scenario("c_crawl", "(c) crawl 0.2 m/s, 10 s", T_WALK + 10.5, T.flat,
                            _setup_walk((0.2, 0.0, 0.0), "crawl"), _eval_walk((0.2, 0.0, 0.0), 0, gait="crawl"))
    S["d_turn"] = Scenario("d_turn", "(d) turn in place 0.5 rad/s", T_WALK + 10.5, T.flat,
                           _setup_walk((0.0, 0.0, 0.5), "trot"), _eval_walk((0.0, 0.0, 0.5), 2, gait="trot"))
    S["e_obstacles"] = Scenario("e_obstacles", "(e) 0.10 m pipe + 0.10 m step, auto gait", 50.0, T.pipe_and_step,
                                _setup_obstacles, _eval_course(5.4, "pipe at x=1.5, step 3.2-4.4"), stop_x=5.6)
    S["f_push"] = Scenario("f_push", "(f) 15 N s lateral push while trotting", T_WALK + 10.5, T.flat,
                           _setup_push, _eval_push)
    S["g_payload"] = Scenario("g_payload", "(g) +3 kg payload: stand + trot 0.4 m/s", T_WALK + 11.5, T.flat,
                              _setup_payload, _eval_payload, payload={"mass": 3.0, "pos": (0.0, 0.0, 0.10)})
    S["h_side"] = Scenario("h_side", "(h1) start lying on the side -> self-righting", 25.0, T.flat,
                           _setup_side, _eval_recovery, spawn=(0.0, 0.0, 0.20, 0.0), spawn_rpy=(math.pi / 2, 0.0),
                           spawn_q=SIDE_Q)
    # 60 N s knocks the standing robot onto its side (40 N s: it rolls on over its back,
    # 100 N s: thrown ~2 m and lands on the back) -> SELF_RIGHT keyframes
    S["h_push"] = Scenario("h_push", "(h2) 60 N s push while standing -> fall -> recovery", 30.0, T.flat,
                           _setup_bigpush(300.0), _eval_recovery)
    S["h_push100"] = Scenario("h_push100", "(h4) 100 N s push while standing (lands on the back)", 30.0, T.flat,
                              _setup_bigpush(500.0), _eval_recovery, required=False)
    S["h_back"] = Scenario("h_back", "(h3) start on the back -> rock + self-righting", 30.0, T.flat,
                           _setup_side, _eval_recovery, spawn=(0.0, 0.0, 0.20, 0.0), spawn_rpy=(math.pi, 0.0),
                           spawn_q=SIDE_Q, required=False)
    S["i_plate_trot"] = Scenario("i_plate_trot", "(i) mu 0.15 plate, trot 0.4 m/s", 30.0, _plate_terrain,
                                 _setup_plate((0.4, 0.0, 0.0), "trot"), _eval_plate(6.0, 1.0, 5.0), required=False,
                                 stop_x=6.2)
    S["i_plate_crawl"] = Scenario("i_plate_crawl", "(i) mu 0.15 plate, crawl 0.2 m/s", 45.0, _plate_terrain,
                                  _setup_plate((0.2, 0.0, 0.0), "crawl"), _eval_plate(6.0, 1.0, 5.0),
                                  required=False, stop_x=6.2)
    S["j_stairs"] = Scenario("j_stairs", "(j) 0.17 m stairs up + down, auto gait", 70.0, T.stairs,
                             _setup_obstacles, _eval_course(6.3, "4 x 0.17 m up, landing, 4 down"),
                             required=False, stop_x=6.5)
    S["j_ramp"] = Scenario("j_ramp", "(j) 16 deg ramp up + down, auto gait", 60.0, T.ramp,
                           _setup_obstacles, _eval_course(8.2, "16 deg, 2 m up, 1.5 m top, 2 m down"),
                           required=False, stop_x=8.4)
    return S


SCENARIOS = build_scenarios()


def make_config(sc: Scenario, seed: int, params: Optional[dict] = None, perception: Optional[str] = None,
                mpc_latency="jitter", sensors=None, plant: Optional[dict] = None) -> SimConfig:
    """plant: optional SimConfig overrides (servo, servo_p, servo_d, kv, substeps)."""
    kw = dict(plant or {})
    if sensors is not None:
        kw["sensors"] = sensors
    return SimConfig(terrain=sc.terrain(), spawn=sc.spawn, spawn_rpy=sc.spawn_rpy, spawn_q=sc.spawn_q,
                     params=dict(params or {}), seed=seed, perception=perception or sc.perception,
                     mpc_latency=mpc_latency, payload=sc.payload, yaw_jitter=sc.yaw_jitter, xy_jitter=0.02, **kw)


def run_scenario(sc: Scenario, seed: int, params: Optional[dict] = None, perception: Optional[str] = None,
                 duration: Optional[float] = None, mpc_latency="jitter", sensors=None, plant: Optional[dict] = None):
    """Build, run and evaluate one scenario -> (sim, arrays, metrics, ok, criterion)."""
    cfg = make_config(sc, seed, params, perception, mpc_latency, sensors, plant)
    sim = PhysicsSim(cfg)
    sc.setup(sim, seed)
    stop = None
    if sc.stop_x is not None:
        sx = sc.stop_x
        stop = lambda s: s.data.qpos[0] >= sx
    try:
        sim.run(duration if duration is not None else sc.duration, stop_fn=stop)
    finally:
        sim.close()
    A = sim.arrays()
    m, ok, crit = sc.evaluate(sim, A)
    m["sim_time"] = float(A["t"][-1]) if len(A["t"]) else 0.0
    m["wall"] = {k: round(v, 2) for k, v in sim.wall.items()}
    return sim, A, m, bool(ok), crit
