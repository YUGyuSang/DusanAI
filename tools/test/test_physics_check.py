"""Tests of the MuJoCo physics check (tools/physics_check.py, tools/physics/).

* without MuJoCo: the actuation contract - the URDF servos (gz
  JointPositionController PID force mode) and the controller's ``servo_kp``
  agree, the URDF parser reads every plugin parameter;
* with MuJoCo (``pip install mujoco``, skipped otherwise): the MJCF mirrors the
  URDF, the gz PID emulation matches gz::math::PID, the harness reproduces the
  node's RobotState inputs, and a SHORT closed-loop run of the real controller
  (stand-up + stand, then 3 s of trot) passes the scenario criteria.
"""
import math
import os

import numpy as np
import pytest

from dusan_common import robot_model as rm
from module1_locomotion.controller import DEFAULT_PARAMS
from physics import mjcf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


# ---------------------------------------------------------------------------
# contract (no MuJoCo needed)
# ---------------------------------------------------------------------------
def test_urdf_servos_are_pid_force_mode_matching_the_controller():
    desc = mjcf.parse_urdf()
    assert set(desc.servos) == set(rm.JOINT_NAMES)
    for j, sp in desc.servos.items():
        assert sp.mode == "force"
        assert sp.p_gain == pytest.approx(rm.JOINT_SERVO_P_GAIN)
        assert sp.d_gain == pytest.approx(rm.JOINT_SERVO_D_GAIN)
        assert sp.i_gain == 0.0
        eff = desc.joints[j].effort
        assert sp.cmd_max == pytest.approx(eff) and sp.cmd_min == pytest.approx(-eff)
    # the controller feeds torques forward as tau / servo_kp: it must know the servo stiffness
    assert DEFAULT_PARAMS["servo_kp"] == pytest.approx(rm.JOINT_SERVO_P_GAIN)


def test_d_gain_respects_the_discrete_stability_bound_of_the_shank():
    """d dt / I_knee: the backward-difference D term of gz::math::PID at the
    1 ms gz step diverged in the physics check for d 6 (ratio ~1.1)."""
    desc = mjcf.parse_urdf()
    ine = desc.links["fl_lower_leg"].inertial
    com = np.asarray(ine.origin.xyz)
    i_knee = ine.inertia[1] + ine.mass * (com[0] ** 2 + com[2] ** 2)     # about the knee (y) axis
    assert 0.004 < i_knee < 0.01
    assert rm.JOINT_SERVO_D_GAIN * mjcf.GZ_DT / i_knee < 0.75


def test_parse_urdf_structure():
    desc = mjcf.parse_urdf()
    assert desc.root == "base_link"
    assert desc.total_mass == pytest.approx(rm.TOTAL_MASS, abs=1e-6)
    assert len(desc.actuated_joints()) == 12
    for leg in rm.LEG_NAMES:
        assert desc.links[f"{leg}_foot"].mu == pytest.approx(1.0)
        assert any(c.kind == "sphere" for c in desc.links[f"{leg}_foot"].collisions)
    knee = desc.joints["fl_knee"]
    assert (knee.lower, knee.upper, knee.effort) == pytest.approx((-2.8, -0.2, 60.0))
    assert desc.servos["fl_knee"].initial_position == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# MuJoCo
# ---------------------------------------------------------------------------
mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module", autouse=True)
def _single_thread_blas():
    """The MPC matrices are small: an oversubscribed BLAS pool makes the solves hit
    their wall-clock limit (tools/physics_check.py sets OPENBLAS_NUM_THREADS=1)."""
    from module1_locomotion import qp as Q
    before = Q.blas_threads()
    Q.limit_blas_threads(1)
    yield
    if before:
        Q.limit_blas_threads(before)


from physics import scenarios as S  # noqa: E402
from physics import sim as SM  # noqa: E402
from physics import terrain as T  # noqa: E402


def test_mjcf_mirrors_the_urdf():
    desc = mjcf.parse_urdf()
    m = mujoco.MjModel.from_xml_string(mjcf.build_mjcf(desc, T.flat().mjcf()))
    assert m.opt.timestep == pytest.approx(mjcf.PHYSICS_DT)
    assert float(np.sum(m.body_mass)) == pytest.approx(rm.TOTAL_MASS, abs=1e-6)
    for j in rm.JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lim = rm.JOINT_LIMITS[j.split("_", 1)[1]]
        assert m.jnt_range[jid] == pytest.approx([lim["lower"], lim["upper"]])
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j + "_servo")
        assert m.actuator_ctrlrange[aid] == pytest.approx([-lim["effort"], lim["effort"]])
        assert m.actuator_biastype[aid] == 0                      # plain motor (PID force from GzServo)
    # the velocity what-if still builds force-limited velocity actuators
    mv = mujoco.MjModel.from_xml_string(mjcf.build_mjcf(desc, T.flat().mjcf(), servo="velocity"))
    aid = mujoco.mj_name2id(mv, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_knee_servo")
    assert mv.actuator_forcerange[aid] == pytest.approx([-60.0, 60.0])


def test_gz_servo_reproduces_gz_math_pid():
    sp = mjcf.ServoSpec("j", use_velocity_commands=False, cmd_max=10.0, cmd_min=-10.0, p_gain=100.0,
                        i_gain=0.0, d_gain=2.0, initial_position=0.5)
    sv = SM.GzServo([sp], dt=0.001)
    # at the initial target: no error, no output (e_last starts at 0 like gz)
    assert sv.command(np.array([0.5]))[0] == pytest.approx(0.0)
    sv.target = np.array([0.52])
    # e = -0.02: p term 2, derivative kick -2 * (-0.02 - 0) / 1 ms = 40 -> clamped to cmd_max
    assert sv.command(np.array([0.5]))[0] == pytest.approx(10.0)
    # next step, same error: derivative 0 -> pure p term
    assert sv.command(np.array([0.5]))[0] == pytest.approx(2.0)
    # velocity mode: (target - q) / dt clamped to cmd_max
    sv2 = SM.GzServo([mjcf.ServoSpec("j", use_velocity_commands=True, cmd_max=12.0, cmd_min=-12.0)], dt=0.001)
    sv2.target = np.array([0.005])
    assert sv2.command(np.array([0.0]))[0] == pytest.approx(5.0)
    sv2.target = np.array([1.0])
    assert sv2.command(np.array([0.0]))[0] == pytest.approx(12.0)


def test_node_state_defaults_match_the_node():
    """The harness builds RobotState like locomotion_controller_node: same NODE_PARAMS defaults."""
    src = open(os.path.join(ROOT, "module1_locomotion", "module1_locomotion", "locomotion_controller_node.py"),
               encoding="utf-8").read()
    block = src.split("NODE_PARAMS = {", 1)[1].split("\n}", 1)[0]
    node = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        node[k.strip().strip('"')] = v.strip().rstrip(",").split("#")[0].strip()
    for k, v in SM.NODE_STATE_DEFAULTS.items():
        assert k in node, k
        assert str(v) == node[k].strip('"') or repr(v) == node[k], (k, v, node[k])


def test_scenario_registry():
    keys = set(S.SCENARIOS)
    for k in ("a_stand", "b_trot", "c_crawl", "d_turn", "e_obstacles", "f_push", "g_payload", "h_side", "h_push"):
        assert k in keys and S.SCENARIOS[k].required, k
    for k in ("i_plate_trot", "i_plate_crawl", "j_stairs", "j_ramp", "h_back", "h_push100"):
        assert k in keys and not S.SCENARIOS[k].required, k
    t = T.plate()
    assert t.mu_at(2.0, 0.0) == pytest.approx(0.15) and t.mu_at(-1.0, 0.0) == pytest.approx(T.FLOOR_MU)


def test_slippery_scenarios_registry_and_prior():
    import physics_check as PC
    reg = PC.all_scenarios()
    keys = {k for k in reg if k.startswith("k_slip")}
    assert keys == {"k_slip_off", "k_slip_prior", "k_slip_detect", "k_slip_both", "k_slip02_off",
                    "k_slip02_detect", "k_slip02_both"}
    assert reg["k_slip_both"].required and reg["k_slip02_both"].required and not reg["k_slip_off"].required
    assert reg["k_slip_off"].params == {"slip_detection": False, "use_terrain_friction": False}
    assert reg["k_slip_prior"].params == {"slip_detection": False} and reg["k_slip_both"].params == {}
    assert all(reg[k].seeds == (0, 1, 2) for k in keys)
    assert reg["k_slip02_both"].terrain().mu_at(2.0, 0.0) == pytest.approx(0.2)
    # the prior is built through the world_layout.yaml path of the terrain mapper
    fm = PC.plate_friction_map(reg["k_slip_both"].terrain())
    assert len(fm) == 1 and fm.mu_at(3.0, 0.0) == pytest.approx(0.15) and fm.mu_at(0.5, 0.0) == 1.0
    rows = [{"key": "k_slip_both", "title": "t", "passed": 3, "runs": 3,
             "key_metrics": {"falls_total": 0, "plate_speed": 0.19, "slip_dist": 5.1, "detect_latency": float("nan")}},
            {"key": "b_trot", "title": "b", "passed": 3, "runs": 3, "key_metrics": {}}]
    tab = PC.slippery_table(rows)
    assert set(tab) == {"k_slip_both", "note"} and tab["k_slip_both"]["falls"] == 0
    assert tab["k_slip_both"]["plate_speed"] == pytest.approx(0.19)
    assert PC.slippery_table(rows[1:]) == {}


def test_short_slippery_plate_approach_with_the_friction_prior():
    """SHORT (k) run: trot command 0.4 m/s towards the mu 0.15 plate with the layout
    friction prior: the slip mode (speed limit, torque feed-forward, MPC mu) is on
    before the front feet reach the plate, without a fall."""
    import physics_check as PC
    sc = PC.all_scenarios()["k_slip_both"]
    # MPC mode (the MuJoCo tool validates the MPC mode; the default stable mode is
    # validated in Gazebo, module1 README section 4.6) with the ground-truth height
    # map: with the ray-cast depth map the MPC trot did not reach the plate within
    # this short 5 s run any more (also with the earlier leash values) - the full
    # physics_check.py scenario keeps the depth pipeline
    prm = dict(sc.params)
    prm.setdefault("locomotion_mode", "mpc")
    sim = SM.PhysicsSim(S.make_config(sc, seed=0, params=prm, perception="gt"))
    sc.setup(sim, 0)
    try:
        sim.run(S.T_WALK + 5.0, stop_fn=lambda s: s.data.qpos[0] >= PC.SLIP_X0)
    finally:
        sim.close()
    A = sim.arrays()
    on = [e["t"] for e in sim.events if e.get("event") == "slip_mode_on" and e["t"] > S.T_WALK - 1.0]
    entry = np.nonzero(A["pos"][:, 0] >= PC.SLIP_X0 - rm.HIP_X_OFFSET)[0]
    assert on and entry.size and on[0] < A["t"][entry[0]]
    c = sim.ctrl
    assert c.slip_mode and c.slip_source in ("terrain", "both")
    assert c.cmd_f[0] <= c.p["slip_max_vx"] + 1e-9 and c.ff_gain == pytest.approx(1.0)
    assert c.mu_mpc == pytest.approx(0.15)
    assert sim.perception.info["slippery"] is True and sim.perception.info["class"] == "slippery"
    assert S.fall_episodes(A) == []
    assert len(sim._slip_log) > 100 and len(sim._slip_log[0]) == len(PC.SLIP_LOG)


def test_short_stand_and_trot_in_physics():
    """SHORT version of (a) + (b): straight legs at floor + 0.56 -> STANDING_UP
    -> STAND (height 0.34 +- 0.03, |roll|, |pitch| < 3 deg), then 3 s of trot
    at 0.4 m/s without a fall, all with the real controller on the gz-like plant."""
    sc = S.SCENARIOS["b_trot"]
    cfg = S.make_config(sc, seed=0, perception="none", params={"locomotion_mode": "mpc"})   # MuJoCo = MPC check
    sim = SM.PhysicsSim(cfg)
    sc.setup(sim, 0)
    try:
        sim.run(S.T_WALK + 3.0)
    finally:
        sim.close()
    A = sim.arrays()
    t_st = S.first_mode_time(A, "STAND")
    assert math.isfinite(t_st) and t_st < 3.5
    st = S.attitude_stats(A, t_st + 0.3, S.T_WALK)
    assert abs(st["height_mean"] - 0.34) <= 0.03
    assert st["max_roll_deg"] < 3.0 and st["max_pitch_deg"] < 3.0
    assert S.fall_episodes(A) == []
    assert sim.ctrl.mode == "WALK" and sim.ctrl.gait.name == "trot"
    vx, vy, _ = S.mean_speed(A, S.T_WALK + 1.5, S.T_WALK + 3.0)
    assert 0.3 <= vx <= 0.5, vx
    walk = S.attitude_stats(A, S.T_WALK, S.T_WALK + 3.0)
    assert walk["max_roll_deg"] < 10.0 and walk["max_pitch_deg"] < 10.0
    tq = S.torque_stats(A, 0.0, S.T_WALK + 3.0)
    assert tq["saturation_frac"] < 0.01
