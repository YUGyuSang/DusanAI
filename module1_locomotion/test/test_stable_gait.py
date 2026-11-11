"""The "stable" locomotion mode (stable_gait.py + LocomotionController
integration): Bezier profiles, Raibert footholds, the stance force limit,
the gravity feed-forward, terrain inputs, heading hold, mode switching - and
a kinematic plant whose base pose follows the stance feet (no MPC, no
world-frame body target in this mode, so the base must be *carried* by the
commanded feet)."""
import math

import numpy as np
import pytest

from dusan_common import robot_model as rm
from dusan_common.geometry import euler_to_rot
from module1_locomotion import controller as C
from module1_locomotion import kinematics as kin
from module1_locomotion import stable_gait as SG
from module1_locomotion.gait import GaitScheduler, make_gaits

DT = 0.01


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
def test_bezier_profiles_are_smooth_and_normalised():
    s = np.linspace(0.0, 1.0, 201)
    bx, dbx = SG.bezier(SG.SWING_X_CTRL, s)
    bz, dbz = SG.bezier(SG.SWING_Z_CTRL, s)
    assert bx[0] == pytest.approx(0.0) and bx[-1] == pytest.approx(1.0)
    assert np.all(np.diff(bx) >= -1e-12)                       # monotonic progress
    assert dbx[0] == pytest.approx(0.0) and dbx[-1] == pytest.approx(0.0)   # ground-speed matched ends
    assert bz[0] == pytest.approx(0.0) and bz[-1] == pytest.approx(0.0)
    assert dbz[0] == pytest.approx(0.0) and dbz[-1] == pytest.approx(0.0)   # soft lift-off / landing
    assert bz.max() == pytest.approx(1.0, abs=1e-3) and bz[100] == pytest.approx(1.0)
    # derivative matches a finite difference
    fd = np.gradient(bx, s)
    assert np.allclose(fd[5:-5], dbx[5:-5], atol=2e-2)


def test_swing_point_endpoints_apex_and_lift():
    p0, p1 = np.array([0.1, 0.13, -0.315]), np.array([0.3, 0.15, -0.30])
    assert np.allclose(SG.swing_point(p0, p1, 0.0, 0.07), p0)
    assert np.allclose(SG.swing_point(p0, p1, 1.0, 0.07), p1)
    mid = SG.swing_point(p0, p1, 0.5, 0.07)
    assert mid[2] == pytest.approx(0.5 * (p0[2] + p1[2]) + 0.07)
    # lift: the foot rises before it moves forward
    early = SG.swing_point(p0, p1, 0.15, 0.10, lift=0.2)
    assert early[0] == pytest.approx(p0[0]) and early[2] > p0[2] + 0.02


def test_parameter_mapping_covers_the_controller_parameters():
    sp = SG.StableParams.from_controller_params(C.DEFAULT_PARAMS)
    used_by_controller = {"stable_trot_period", "stable_trot_duty", "stable_walk_period", "stable_walk_duty",
                          "stable_walk_sway_x", "stable_heading_hold_gain", "stable_heading_hold_max"}
    used_by_controller |= {f"stable_{g}_max_{a}" for g in ("trot", "walk") for a in ("vx", "vy", "wz")}
    stable_keys = {k for k in C.DEFAULT_PARAMS if k.startswith("stable_")}
    assert {k for k in stable_keys if k not in used_by_controller} <= set(SG.StableParams._MAP)
    assert set(SG.StableParams._MAP) <= set(C.DEFAULT_PARAMS)
    for key, name in SG.StableParams._MAP.items():
        assert getattr(sp, name) == pytest.approx(float(C.DEFAULT_PARAMS[key])), key
    assert C.DEFAULT_PARAMS["locomotion_mode"] == "stable"


# ---------------------------------------------------------------------------
# StableGait alone
# ---------------------------------------------------------------------------
def _scheduler(gait="trot"):
    sched = GaitScheduler(make_gaits(0.4, 0.6, 1.2, 0.8), initial="stand")
    sched.reset(0.0, gait)
    return sched


def test_touchdown_target_is_raibert_and_moves_with_the_frame():
    g = SG.StableGait(SG.StableParams(raibert_gain=0.1))
    r = g.nominal[0]
    td = g.touchdown_target(0, (0.4, 0.0, 0.0), t_stance=0.24, t_remaining=0.0)
    assert td[:2] == pytest.approx(r + np.array([0.4 * 0.12, 0.0]))
    assert td[2] == pytest.approx(g.ground_z)
    # earlier in the swing the same ground point is further ahead in the body frame
    td2 = g.touchdown_target(0, (0.4, 0.0, 0.0), t_stance=0.24, t_remaining=0.1)
    assert td2[0] == pytest.approx(td[0] + 0.04)
    # velocity error feedback (capture), limited
    td3 = g.touchdown_target(0, (0.4, 0.0, 0.0), 0.24, 0.0, v_err=(1.0, -0.2))
    assert td3[0] - td[0] == pytest.approx(0.05) and td3[1] - td[1] == pytest.approx(-0.02)
    # turning: the front-left foot steps along +y / -x (w x r)
    tdw = g.touchdown_target(0, (0.0, 0.0, 0.8), 0.24, 0.0)
    assert tdw[0] < r[0] and tdw[1] > r[1]


def test_stance_feet_sweep_backwards_at_the_command_and_force_limit():
    """Ideal joint tracking: stance targets move at -v (body frame); with the
    measured joints frozen (body not following) the horizontal target error
    of every stance foot stays within stance_max_error."""
    p = SG.StableParams(stance_max_error=0.015)
    q0 = kin.stand_joint_angles()
    g = SG.StableGait(p)
    sched = _scheduler()
    t, q = 0.0, q0.copy()
    feet_prev, prev_stance = None, np.zeros(4, dtype=bool)
    speeds = []
    for _ in range(60):
        t += DT
        gs = sched.update(t)
        q_new, info = g.step(DT, gs, (0.3, 0.0, 0.0), np.zeros(3), q, walking=True)
        stance = info["stance"]
        if feet_prev is not None:
            both = stance & prev_stance
            if both.any():
                speeds.append(float(np.mean((info["feet_L"][both, 0] - feet_prev[both, 0]) / DT)))
        feet_prev, prev_stance = info["feet_L"].copy(), stance.copy()
        q = kin.ik_all(info["feet_b"])[0]        # ideal tracking of the geometric target
    assert np.median(speeds) == pytest.approx(-0.3, abs=0.02)
    # frozen measurement: the target never runs away from the measured foot
    g2 = SG.StableGait(p)
    sched = _scheduler()
    meas_b = kin.fk_all(q0)
    t = 0.0
    for _ in range(80):
        t += DT
        gs = sched.update(t)
        _q, info = g2.step(DT, gs, (0.5, 0.0, 0.6), np.zeros(3), q0, walking=True)
        st = info["stance"]
        err = np.linalg.norm(info["feet_b"][st, :2] - meas_b[st, :2], axis=1)
        assert np.all(err <= 0.015 + 1e-3)


def test_gravity_feedforward_carries_the_weight_through_the_servo():
    p = SG.StableParams()
    g = SG.StableGait(p)
    q0 = kin.stand_joint_angles()
    sched = GaitScheduler(make_gaits(), initial="stand")
    gs = sched.update(0.0)
    for _ in range(100):                                       # load filter settles in STAND
        q, info = g.step(DT, gs, (0.0, 0.0, 0.0), np.zeros(3), q0, walking=False)
    q_geo = kin.ik_all(info["feet_b"])[0]
    tau = p.servo_kp * (q - q_geo)                             # what the PD servo produces at q_geo
    f = np.zeros((4, 3))
    for leg in range(4):
        J = kin.leg_jacobian(q_geo[3 * leg:3 * leg + 3], leg)
        f[leg] = np.linalg.solve(J.T, -tau[3 * leg:3 * leg + 3])   # tau = -J^T f
    assert f[:, 2].sum() == pytest.approx(p.mass * SG.GRAVITY, rel=1e-3)
    assert np.allclose(f[:, :2], 0.0, atol=1e-6)
    # off: pure IK
    g0 = SG.StableGait(SG.StableParams(gravity_ff=0.0))
    for _ in range(5):
        q, info = g0.step(DT, gs, (0.0, 0.0, 0.0), np.zeros(3), q0, walking=False)
    assert np.allclose(q, kin.ik_all(info["feet_b"])[0])


def test_levelling_and_terrain_inputs():
    g = SG.StableGait(SG.StableParams(kp_level=0.3, ki_level=0.0))
    q0 = kin.stand_joint_angles()
    gs = GaitScheduler(make_gaits(), initial="stand").update(0.0)
    _q, info = g.step(DT, gs, (0, 0, 0), np.array([0.0, 0.1, 0.0]), q0, walking=False)
    assert info["rp_cmd"][1] == pytest.approx(-0.03)           # nose down measured -> nose up correction
    _q, info = g.step(DT, gs, (0, 0, 0), np.array([0.0, -0.2, 0.0]), q0, walking=False,
                      rp_des=(0.0, -0.2))
    assert info["rp_cmd"][1] == pytest.approx(-0.2)            # terrain pitch reference, no error
    # ground_fn: the stance feet settle on the terrain heights (front higher on a ramp)
    slope = math.tan(0.2)
    g = SG.StableGait(SG.StableParams(stand_center_tau=0.05))
    ground = lambda xy: g.ground_z + slope * np.asarray(xy)[:, 0]     # noqa: E731
    for _ in range(100):
        _q, info = g.step(DT, gs, (0, 0, 0), np.zeros(3), q0, walking=False, ground_fn=ground)
    front, rear = info["feet_L"][:2, 2].mean(), info["feet_L"][2:, 2].mean()
    assert front - rear == pytest.approx(slope * 2 * rm.HIP_X_OFFSET, abs=0.01)
    # obstacle_fn: the swing apex clears an obstacle along the swing
    g = SG.StableGait(SG.StableParams(clearance=0.04))
    sched = _scheduler()
    top = g.ground_z - rm.FOOT_RADIUS + 0.12                    # a 0.12 m bar under every swing
    t, heights = 0.0, []
    for _ in range(40):
        t += DT
        gs = sched.update(t)
        _q, info = g.step(DT, gs, (0.3, 0, 0), np.zeros(3), q0, obstacle_fn=lambda a, b: top)
        heights.append(info["swing_height"].max())
    assert max(heights) == pytest.approx(0.12 + 0.04, abs=0.01)


def test_foothold_choice_is_applied_and_its_jumps_are_rate_limited():
    """The planner's choice is taken at lift-off; later jumps (e.g. the nominal
    point crossing a pipe) move the touch-down point by at most
    max_retarget_step per control step."""
    g = SG.StableGait(SG.StableParams(max_retarget_step=0.015))
    q0 = kin.stand_joint_angles()
    sched = _scheduler()
    state = {"n": 0}

    def foothold(leg, xy):
        state["n"] += 1
        return np.asarray(xy) + (np.array([0.06, 0.0]) if state["n"] > 6 else np.array([-0.04, 0.0]))

    t, prev = 0.0, None
    for k in range(12):
        t += DT
        gs = sched.update(t)
        _q, info = g.step(DT, gs, (0.0, 0.0, 0.0), np.zeros(3), q0, foothold_fn=foothold)
        legs = np.nonzero(g.swinging)[0]
        if k == 0:
            raw = [g.touchdown_target(int(l), (0, 0, 0), gs.stance_duration, 0.0)[:2] for l in legs]
            assert np.allclose(g.p1[legs, :2], np.array(raw) + [-0.04, 0.0], atol=1e-9)   # taken at once
        elif prev is not None:
            step = np.linalg.norm(g.p1[legs, :2] - prev[legs, :2], axis=1)
            assert np.all(step <= 0.015 + 1e-9)
        prev = g.p1.copy()


# ---------------------------------------------------------------------------
# controller integration with a kinematic plant carried by the stance feet
# ---------------------------------------------------------------------------
class FootPlant:
    """Ideal joint servo; once standing, the base pose (x, y, z, yaw) is the
    least-squares fit that keeps the stance feet where they touched down
    (roll / pitch = the commanded attitude).  Before that (stand-up) the
    base rests on its lowest foot like test_controller.Sim."""

    def __init__(self, params=None):
        p = {"use_mpc_thread": False, "stable_gravity_ff": 0.0}   # ideal servo: no sag to compensate
        p.update(params or {})
        self.ctrl = C.LocomotionController(p)
        self.st = C.RobotState()
        self.st.pos = np.array([0.0, 0.0, 0.56])
        self.st.q = np.zeros(12)
        self.t = 0.0
        self.world_feet = None
        self.stance_prev = np.zeros(4, dtype=bool)
        self.dbg = None
        self.slips = []

    def _feet_level(self, q, rp):
        return kin.fk_all(q) @ euler_to_rot(float(rp[0]), float(rp[1]), 0.0).T

    def run(self, duration, cmd=(0.0, 0.0, 0.0)):
        for _ in range(int(round(duration / DT))):
            q, dbg = self.ctrl.step(self.t, self.st, cmd, None, None)
            assert np.all(np.isfinite(q))
            assert np.all(q >= kin.JOINT_LOWER - 1e-9) and np.all(q <= kin.JOINT_UPPER + 1e-9)
            self.dbg = dbg
            self.st.q = q.copy()
            c = self.ctrl
            if c.mode in ("STAND", "WALK") and c.loco_mode == "stable" and c.stable.initialized:
                rp = c.stable.rp_cmd.copy()
                fl = self._feet_level(q, rp)
                stance = np.asarray(dbg["contacts"], dtype=bool)
                yaw = float(self.st.rpy[2])
                Rz = euler_to_rot(0.0, 0.0, yaw)
                cur_w = fl @ Rz.T + self.st.pos
                if self.world_feet is None:
                    self.world_feet = cur_w.copy()
                new = stance & ~self.stance_prev
                self.world_feet[new] = cur_w[new]
                self.stance_prev = stance.copy()
                if stance.sum() >= 2:
                    A, B = fl[stance, :2], self.world_feet[stance, :2]
                    a0, b0 = A.mean(axis=0), B.mean(axis=0)
                    H = (A - a0).T @ (B - b0)
                    yaw_new = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
                    c2, s2 = math.cos(yaw_new), math.sin(yaw_new)
                    R2 = np.array([[c2, -s2], [s2, c2]])
                    xy = b0 - R2 @ a0
                    z = float(np.mean(self.world_feet[stance, 2] - fl[stance, 2]))
                    pos = np.array([xy[0], xy[1], z])
                    # residual = how far the commanded stance feet would have to slide
                    pred = fl[stance, :2] @ R2.T + xy
                    self.slips.append(float(np.abs(pred - B).max()))
                    self.st.vel_world = (pos - self.st.pos) / DT
                    self.st.omega_body = np.array([0.0, 0.0, math.remainder(yaw_new - yaw, 2 * math.pi) / DT])
                    self.st.pos = pos
                    self.st.rpy = np.array([rp[0], rp[1], yaw_new])
            else:
                self.world_feet = None
                self.stance_prev = np.zeros(4, dtype=bool)
                fk = kin.fk_all(q)
                self.st.pos = np.array([self.st.pos[0], self.st.pos[1], rm.FOOT_RADIUS - fk[:, 2].min()])
                self.st.vel_world = np.zeros(3)
                self.st.omega_body = np.zeros(3)
            self.t += DT
        return self


def test_stable_mode_walks_at_the_commanded_velocity_without_sliding_feet():
    pl = FootPlant().run(2.6)
    assert pl.ctrl.mode == "STAND" and pl.ctrl.loco_mode == "stable"
    assert pl.dbg["locomotion_mode"] == "stable" and pl.dbg["mpc_status"] == "none"   # no MPC at all
    assert pl.st.pos[2] == pytest.approx(rm.NOMINAL_BODY_HEIGHT, abs=0.01)
    pl.run(2.0, cmd=(0.3, 0.0, 0.0))
    x0, t0 = pl.st.pos[0], pl.t
    pl.run(3.0, cmd=(0.3, 0.0, 0.0))
    assert pl.ctrl.mode == "WALK" and pl.dbg["gait"] == "trot"
    assert (pl.st.pos[0] - x0) / (pl.t - t0) == pytest.approx(0.3, abs=0.02)
    assert abs(pl.st.pos[1]) < 0.02 and abs(pl.st.rpy[2]) < 0.02
    assert pl.st.pos[2] == pytest.approx(rm.NOMINAL_BODY_HEIGHT, abs=0.01)
    assert np.percentile(pl.slips, 95) < 0.003               # commanded stance feet consistent (rigid)
    # turning in place
    yaw0, t0 = pl.st.rpy[2], pl.t
    pl.run(3.0, cmd=(0.0, 0.0, 0.5))
    assert math.remainder(pl.st.rpy[2] - yaw0, 2 * math.pi) / (pl.t - t0) == pytest.approx(0.5, abs=0.06)
    assert np.percentile(pl.slips, 95) < 0.005
    # stop: back to STAND with the feet under the hips
    pl.run(2.5)
    assert pl.ctrl.mode == "STAND"
    feet_b = kin.fk_all(pl.st.q)
    assert np.allclose(feet_b[:, :2], pl.ctrl.nominal_feet_b[:, :2], atol=0.03)


def test_stable_mode_speed_limits_walk_gait_and_estop():
    pl = FootPlant().run(2.6)
    pl.run(3.0, cmd=(2.0, 1.0, 3.0))
    lim = [C.DEFAULT_PARAMS[f"stable_trot_max_{a}"] for a in ("vx", "vy", "wz")]
    assert pl.dbg["cmd"] == pytest.approx(lim, abs=1e-6)
    pl.ctrl.handle_command("gait:walk", pl.t, pl.st)           # alias of crawl
    pl.run(3.0, cmd=(0.2, 0.0, 0.0))
    assert pl.dbg["gait"] == "crawl" and pl.dbg["cmd"][0] == pytest.approx(C.DEFAULT_PARAMS["stable_walk_max_vx"])
    p0, t0, yaw0 = pl.st.pos[:2].copy(), pl.t, float(pl.st.rpy[2])
    pl.run(2.4, cmd=(0.2, 0.0, 0.0))
    d = pl.st.pos[:2] - p0
    assert (math.cos(yaw0) * d[0] + math.sin(yaw0) * d[1]) / (pl.t - t0) == pytest.approx(0.2, abs=0.03)
    # E-STOP mid-stride: the swing foot is put down and everything is held
    while all(pl.dbg["contacts"]):
        pl.run(DT, cmd=(0.2, 0.0, 0.0))
    pl.ctrl.handle_command("estop", pl.t, pl.st)
    q_e = pl.ctrl.estop_q
    fl = pl._feet_level(q_e, pl.ctrl.stable.rp_cmd)
    assert np.allclose(fl[:, 2], pl.ctrl.stable.ground_z, atol=0.02)
    pl.run(0.2)
    assert pl.ctrl.mode == "ESTOP" and np.allclose(pl.st.q, q_e)


def test_locomotion_mode_switch_while_walking_and_toggle():
    pl = FootPlant().run(2.6)
    pl.run(1.0, cmd=(0.3, 0.0, 0.0))
    assert pl.ctrl.mode == "WALK"
    assert pl.ctrl.request_locomotion_mode("bogus") is False and pl.ctrl.loco_mode == "stable"
    pl.ctrl.handle_command("mode:mpc", pl.t, pl.st)
    assert pl.ctrl.loco_mode == "stable" and pl.ctrl._loco_mode_pending == "mpc"   # never mid-stride
    seen = set()
    for _ in range(100):                                       # stops at a safe boundary, then switches
        pl.run(DT, cmd=(0.3, 0.0, 0.0))
        seen.add((pl.ctrl.loco_mode, pl.ctrl.mode))
        if pl.ctrl.loco_mode == "mpc":
            break
    assert pl.ctrl.loco_mode == "mpc" and ("stable", "STAND") in seen
    assert pl.ctrl.gait is pl.ctrl.gait_mpc
    ev = [e for e in pl.ctrl.pop_events() if e["event"] == "locomotion_mode"]
    assert ev and ev[-1]["mode"] == "mpc" and ev[-1]["from"] == "stable"
    pl.run(0.3, cmd=(0.3, 0.0, 0.0))
    assert pl.ctrl.mode == "WALK" and pl.dbg["locomotion_mode"] == "mpc"   # resumed in the new mode
    pl.ctrl.handle_command("mode:toggle", pl.t, pl.st)
    assert pl.ctrl._loco_mode_pending == "stable"
    pl.ctrl.handle_command("mode:toggle", pl.t, pl.st)         # toggling back cancels the request
    assert pl.ctrl._loco_mode_pending is None
    # standing: applied at the next step
    c = C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "mpc"})
    assert c.loco_mode == "mpc" and c.gait is c.gait_mpc
    with pytest.raises(ValueError):
        C.LocomotionController({"use_mpc_thread": False, "locomotion_mode": "fast"})


def test_heading_hold_corrects_a_yaw_drift_only_without_a_turn_command():
    c = C.LocomotionController({"use_mpc_thread": False})
    st = C.RobotState()
    st.rpy = np.array([0.0, 0.0, 0.10])
    c.cmd_f = np.array([0.3, 0.0, 0.0])
    c.cmd_raw = np.array([0.3, 0.0, 0.0])
    cmd = c._stable_heading_hold(st, walking=True)              # latches the reference
    assert cmd[2] == 0.0
    st.rpy = np.array([0.0, 0.0, 0.15])                         # drifted left
    cmd = c._stable_heading_hold(st, walking=True)
    assert cmd[2] == pytest.approx(-0.05 * C.DEFAULT_PARAMS["stable_heading_hold_gain"])
    st.rpy = np.array([0.0, 0.0, 1.0])
    assert c._stable_heading_hold(st, True)[2] == pytest.approx(-C.DEFAULT_PARAMS["stable_heading_hold_max"])
    c.cmd_raw = np.array([0.3, 0.0, 0.4])                       # turning: no hold, reference released
    c.cmd_f = np.array([0.3, 0.0, 0.4])
    assert c._stable_heading_hold(st, True)[2] == pytest.approx(0.4) and c._yaw_ref is None


def test_stable_mode_uses_the_elevation_map_for_heights_pitch_and_apex():
    class RampMap:
        def height_at(self, x, y):
            x = np.asarray(x, dtype=float)
            z = np.where(x > 0.0, 0.2 * x, 0.0) + 0.0 * np.asarray(y, dtype=float)
            return float(z) if np.ndim(z) == 0 else z

    c = C.LocomotionController({"use_mpc_thread": False})
    c._max_height_along = None
    planner = c.planner
    c.planner = None
    st = C.RobotState()
    st.pos = np.array([1.0, 0.0, 0.2 + rm.NOMINAL_BODY_HEIGHT])
    c._terrain_now = c._query_terrain(RampMap(), st)
    c.ground_z, c.ground_valid = 0.2, True
    c.foot_w = kin.body_to_world(kin.nominal_feet_body(), st.pos, st.R)
    ground_fn, obstacle_fn, rp_des, foothold_fn = c._stable_terrain(DT, st, RampMap(), 0.2)
    assert foothold_fn is None                                 # no foothold planner in this test
    assert ground_fn is not None and rp_des[1] < -0.15        # uphill: nose up (terrain pitch)
    z = ground_fn(np.array([[0.2245, 0.134], [-0.2245, 0.134]]))
    assert z[0] - z[1] == pytest.approx(0.2 * 0.449, abs=0.01)
    top = obstacle_fn(np.array([0.0, 0.1, z[0]]), np.array([0.2, 0.1, z[0]]))
    assert top == pytest.approx(0.2 * 0.2 - c.p["body_height"], abs=0.02)
    # foothold planner (as in the MPC mode) on the touch-down point: L -> world -> planner -> L, cached

    class ShiftPlanner:
        calls = 0

        def select(self, emap, xy, z):
            ShiftPlanner.calls += 1
            return np.array([xy[0] + 0.05, xy[1], z]), 0.0, True

    c.planner = ShiftPlanner()
    _g, _o, _r, foothold_fn = c._stable_terrain(DT, st, RampMap(), 0.2)
    assert foothold_fn(0, np.array([0.3, 0.1])) == pytest.approx([0.35, 0.1])
    assert foothold_fn(0, np.array([0.305, 0.1])) == pytest.approx([0.35, 0.1])   # < 1 cm: cached choice
    assert ShiftPlanner.calls == 1
    assert planner is not None                                 # the default controller has the real planner
    # no map / implausible map: flat fallback
    assert c._stable_terrain(DT, st, None, 0.2) == (None, None, None, None)
    assert c._stable_terrain(DT, st, RampMap(), 1.5) == (None, None, None, None)
