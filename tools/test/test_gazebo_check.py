"""Tests of the headless Gazebo check (tools/gazebo_check.py, tools/gazebo_check/).

* always (pure python): the harness URDF / world mirror the Ubuntu runtime
  (plant.sdf physics + systems, 12 JointPositionControllers in PID force mode
  with the robot_model gains, IMU / odometry / joint-state plugins, no render
  sensors), the real plant world keeps every collision, the ground-truth
  height field / ray casts / emulated depth camera + terrain_mapper pipeline,
  the keyboard operator, the scenario commands and the pass / fall logic;
* with the gz python bindings (``GZ_PYTHON`` = path of a python with
  gz.sim8 + gz.transport13, e.g. ``C:\\gzmamba\\envs\\gz\\python.exe``; skipped
  otherwise): a SHORT closed-loop run of the real controller in gz-sim (DART)
  - stand-up + stand, then a trot - must not fall.
"""
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dusan_common import robot_model as rm
from module1_locomotion.controller import DEFAULT_PARAMS
from gazebo_check import scenarios as S
from gazebo_check import world as W

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CLI = os.path.join(ROOT, "tools", "gazebo_check.py")


# ---------------------------------------------------------------------------
# world / robot (no gz needed)
# ---------------------------------------------------------------------------
def test_world_copies_the_plant_physics_and_systems(tmp_path):
    urdf = W.write_harness_urdf(str(tmp_path / "spot.urdf"), use_xacro=False)
    sdf = W.world_sdf(urdf, "obstacles", spawn=(1.0, 2.0, 0.56, 0.5))
    root = ET.fromstring(sdf)
    world = root.find("world")
    assert world.get("name") == rm.WORLD_NAME
    phys = world.find("physics")
    assert phys.get("type") == "dart"
    assert float(phys.findtext("max_step_size")) == pytest.approx(0.001)
    files = {p.get("filename") for p in world.findall("plugin")}
    for need in ("gz-sim-physics-system", "gz-sim-user-commands-system", "gz-sim-scene-broadcaster-system",
                 "gz-sim-imu-system", "gz-sim-contact-system", "gz-sim-apply-link-wrench-system"):
        assert need in files
    assert "gz-sim-sensors-system" not in files            # rendering not needed headless
    inc = world.find("include")
    assert inc.findtext("name") == rm.ROBOT_NAME
    assert inc.findtext("uri").endswith("spot.urdf")
    assert [float(v) for v in inc.findtext("pose").split()] == pytest.approx([1.0, 2.0, 0.56, 0.0, 0.0, 0.5])
    names = {c.get("name") for c in world.iter("collision")}
    assert {"floor", "pipe", "step"} <= names


def test_spawn_roll_pitch_in_the_include_pose(tmp_path):
    """A robot can be spawned lying (fall recovery check): roll / pitch go into
    the include pose of the flat and of the plant world."""
    world, _ = W.build(str(tmp_path), "flat", (1.0, 2.0, 0.3, 0.5), tag="_rp", spawn_rp=(1.6, -0.1))
    inc = ET.parse(world).getroot().find("world/include")
    assert [float(v) for v in inc.findtext("pose").split()] == pytest.approx([1.0, 2.0, 0.3, 1.6, -0.1, 0.5])
    root = ET.fromstring(W.plant_world_sdf(str(tmp_path / "spot.urdf"), (46.5, 0.72, 0.557, math.pi),
                                           spawn_rp=(math.pi, 0.0)))
    inc = [i for i in root.find("world").findall("include") if i.findtext("name") == "spot"][0]
    assert [float(v) for v in inc.findtext("pose").split()] == pytest.approx([46.5, 0.72, 0.557, math.pi, 0.0,
                                                                              math.pi])


def test_harness_urdf_keeps_the_runtime_plugins_and_drops_render_sensors():
    robot = W.robot_tree(use_xacro=False)
    sensors = [s.get("type") for g in robot.findall("gazebo") for s in g.findall("sensor")]
    assert sensors == ["imu"]
    assert not any(link.findall("visual") for link in robot.findall("link"))
    plugins = [p.get("name") for g in robot.findall("gazebo") for p in g.findall("plugin")]
    assert plugins.count("gz::sim::systems::JointPositionController") == 12
    assert "gz::sim::systems::JointStatePublisher" in plugins
    assert "gz::sim::systems::OdometryPublisher" in plugins
    servo = W.servo_params(robot)
    assert set(servo) == set(rm.JOINT_NAMES)
    for j, sp in servo.items():
        assert not sp["use_velocity_commands"]
        assert sp["p_gain"] == pytest.approx(rm.JOINT_SERVO_P_GAIN)
        assert sp["d_gain"] == pytest.approx(rm.JOINT_SERVO_D_GAIN)
        eff = rm.JOINT_LIMITS[j.split("_", 1)[1]]["effort"]
        assert sp["cmd_max"] == pytest.approx(eff) and sp["cmd_min"] == pytest.approx(-eff)
    # the controller feeds torques forward / estimates forces with the servo stiffness
    assert DEFAULT_PARAMS["servo_kp"] == pytest.approx(rm.JOINT_SERVO_P_GAIN)


def test_servo_and_friction_overrides():
    robot = W.robot_tree(servo={"p_gain": 300.0, "d_gain": 2.0, "cmd_scale": 0.5}, foot_mu=0.6, use_xacro=False)
    servo = W.servo_params(robot)
    assert servo["fl_knee"]["p_gain"] == pytest.approx(300.0)
    assert servo["fl_knee"]["d_gain"] == pytest.approx(2.0)
    assert servo["fl_knee"]["cmd_max"] == pytest.approx(30.0)
    mus = [float(g.findtext("mu1")) for g in robot.findall("gazebo") if (g.get("reference") or "").endswith("_foot")]
    assert mus == [0.6] * 4


def test_terrain_worlds_and_ground_truth_heights(tmp_path):
    urdf = W.write_harness_urdf(str(tmp_path / "spot.urdf"), use_xacro=False)
    pipe = ET.fromstring(W.world_sdf(urdf, "pipe"))
    cyl = [c for c in pipe.iter("collision") if c.get("name") == "pipe"][0]
    assert float(cyl.find("geometry/cylinder/radius").text) == pytest.approx(0.04)
    ramp = ET.fromstring(W.world_sdf(urdf, "ramp"))
    names = {c.get("name") for c in ramp.iter("collision")}
    assert {"floor", "ramp", "plateau"} <= names
    rp = [c for c in ramp.iter("collision") if c.get("name") == "ramp"][0]
    pose = [float(v) for v in rp.findtext("pose").split()]
    assert pose[4] == pytest.approx(-W.RAMP_ANGLE)               # slope up along +x
    # top surface of the pitched box = the analytic height used for the --map stand-in
    a, L, t = W.RAMP_ANGLE, W.RAMP_LENGTH, 0.1
    for u in (0.1, 0.5, 0.9):                                   # points on the top face
        x = pose[0] + (u - 0.5) * L * math.cos(a) - 0.5 * t * math.sin(a)
        z = pose[2] + (u - 0.5) * L * math.sin(a) + 0.5 * t * math.cos(a)
        assert float(W.terrain_height("ramp", x, 0.0)) == pytest.approx(z, abs=1e-4)
    assert float(W.terrain_height("ramp", 6.0, 0.0)) == pytest.approx(W.RAMP_TOP)
    assert W.RAMP_TOP == pytest.approx(0.615, abs=1e-3)
    assert float(W.terrain_height("pipe", W.PIPE_X, 0.0)) == pytest.approx(W.PIPE_HEIGHT)
    assert float(W.terrain_height("pipe", W.PIPE_X + 0.1, 0.0)) == 0.0
    assert float(W.terrain_height("flat", 2.0, 0.0)) == 0.0
    z = W.terrain_height("obstacles", np.array([2.0, 5.0, 9.0]), np.zeros(3))
    assert z == pytest.approx([0.10, 0.10, 0.0])


# ---------------------------------------------------------------------------
# the real plant world, height field, depth camera emulation (no gz needed)
# ---------------------------------------------------------------------------
def _collision_key(c):
    return (c.get("name"), " ".join((c.findtext("pose") or "").split()),
            ET.tostring(c.find("geometry"), encoding="unicode").replace(" ", "").replace("\n", ""))


def test_plant_world_keeps_every_collision_and_spawns_at_the_layout_pose(tmp_path):
    urdf = W.write_harness_urdf(str(tmp_path / "spot.urdf"), use_xacro=False)
    text = W.plant_world_sdf(urdf)
    assert "model://" not in text                                   # includes inlined (no resource path needed)
    world = ET.fromstring(text.split("-->", 1)[1]).find("world")
    files = {pl.get("filename") for pl in world.findall("plugin")}
    assert "gz-sim-physics-system" in files and "gz-sim-imu-system" in files
    assert not files & set(W.SKIP_SYSTEMS)
    assert not list(world.iter("visual")) and not world.findall("light") and not list(world.iter("particle_emitter"))
    phys = world.find("physics")
    assert phys.get("type") == "dart" and float(phys.findtext("max_step_size")) == pytest.approx(0.001)
    # every collision of plant.sdf and of the included models, unchanged
    src = ET.parse(W.PLANT_SDF).getroot().find("world")
    want = [_collision_key(c) for c in src.iter("collision")]
    for inc in src.findall("include"):
        mdl = ET.parse(os.path.join(W._model_dir(inc.findtext("uri").strip()), "model.sdf")).getroot()
        want += [_collision_key(c) for c in mdl.iter("collision")]
    got = [_collision_key(c) for m in world.findall("model") for c in m.iter("collision")]
    assert sorted(got) == sorted(want) and len(got) > 250
    # inlined models carry the include name / pose
    pallet = [m for m in world.findall("model") if m.get("name") == "obstacle_c_obs2_0"][0]
    assert pallet.findtext("pose").split() == ["55", "1", "-0.8778", "0", "0", "1.5708"]
    inc = world.findall("include")
    assert len(inc) == 1 and inc[0].findtext("name") == rm.ROBOT_NAME
    spawn = W.layout_spawn()
    assert [float(v) for v in inc[0].findtext("pose").split()] == pytest.approx(
        [spawn[0], spawn[1], spawn[2], 0.0, 0.0, spawn[3]])
    assert spawn[:2] == pytest.approx((61.9, 0.0))


def test_ray_hit_primitives():
    I = np.eye(3)
    box = {"type": "box", "p": np.array([0.0, 0.0, 0.5]), "R": I, "size": np.array([1.0, 2.0, 1.0])}
    assert float(W.ray_hit(box, np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]))) == pytest.approx(2.0)
    assert float(W.ray_hit(box, np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, -1.0]))) == 0.0     # inside
    assert np.isinf(W.ray_hit(box, np.array([0.6, 0.0, 3.0]), np.array([0.0, 0.0, -1.0])))
    # oblique, unnormalised direction: t is in units of the direction vector
    t = float(W.ray_hit(box, np.array([-2.0, 0.0, 0.5]), np.array([2.0, 0.0, 0.0])))
    assert t == pytest.approx(0.75)
    # pitched box (a ramp): first hit on the tilted top face
    a = 0.3
    R = W._rpy_matrix(0.0, a, 0.0)
    ramp = {"type": "box", "p": np.zeros(3), "R": R, "size": np.array([4.0, 2.0, 0.1])}
    x = 0.5
    z_top = -x * math.tan(a) + 0.05 / math.cos(a)
    assert float(W.ray_hit(ramp, np.array([x, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]))) == pytest.approx(2.0 - z_top)
    # lying cylinder (pipe along y, radius 0.05, centre z 0.05)
    pipe = {"type": "cylinder", "p": np.array([0.0, 0.0, 0.05]), "R": W._rpy_matrix(1.5707963, 0.0, 0.0),
            "radius": 0.05, "length": 3.0}
    zs = 1.0 - W.ray_hit(pipe, np.stack([np.array([0.0, 0.03, 0.06]), np.zeros(3), np.full(3, 1.0)], -1),
                         np.array([0.0, 0.0, -1.0]))
    assert zs[:2] == pytest.approx([0.10, 0.05 + math.sqrt(0.05 ** 2 - 0.03 ** 2)], abs=1e-6)
    assert np.isinf(1.0 - zs[2]) or zs[2] == -np.inf
    sph = {"type": "sphere", "p": np.zeros(3), "R": I, "radius": 1.0}
    assert float(W.ray_hit(sph, np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -2.0]))) == pytest.approx(2.0)


def test_plant_height_field():
    hf = W.HeightField.plant()
    floor_lo, floor_up = -0.8778, -0.0032
    assert float(hf.height(61.9, 0.0)) == pytest.approx(floor_lo, abs=2e-3)          # dock, lower corridor
    assert float(hf.height(45.0, 0.5)) == pytest.approx(floor_up, abs=2e-3)          # upper corridor
    assert float(hf.height(58.535, 0.0)) == pytest.approx(floor_lo + 0.10, abs=0.01)  # C-OBS1 pipe crest
    assert float(hf.height(55.0, 1.0)) == pytest.approx(floor_lo + 0.18, abs=0.01)   # C-OBS2 pallet top
    assert float(hf.height(31.0, 0.0)) == pytest.approx(0.0068, abs=2e-3)           # C-OBS4 plate (1 cm)
    z_mid = float(hf.height(49.1, 0.8))                                             # ramp, 11.8 deg
    assert floor_lo < z_mid < floor_up
    slope = (float(hf.height(48.6, 0.8)) - float(hf.height(49.6, 0.8))) / 1.0
    assert math.degrees(math.atan(slope)) == pytest.approx(11.8, abs=0.3)
    assert float(hf.height(50.0, 1.9)) == pytest.approx(hf.z_top)                     # corridor wall
    assert hf.ground(58.535, 0.0) == pytest.approx(floor_lo, abs=2e-3)               # median ignores the pipe


def test_route_waypoints_are_on_free_floor():
    hf = W.HeightField.plant()
    for (x, y) in S.ROUTE_FULL + S.ROUTE_DETOUR:
        z = float(hf.height(x, y))
        assert z < 0.05 and (abs(z - 0.0) < 0.02 or abs(z + 0.878) < 0.02 or 47.0 <= x <= 51.3), (x, y, z)
        assert -1.5 < y < 1.5
    # the detour passes the pipe's north end: free floor across the whole gap at the gap waypoints
    for (x, y) in S.ROUTE_DETOUR[1:3]:
        ys = np.linspace(y - 0.2, y + 0.2, 21)
        assert np.all(np.abs(hf.height(np.full_like(ys, x), ys) + 0.878) < 0.01), (x, y)
    # the route climbs the ramp inside its band (y -0.17 .. 1.76) and ends on the upper level
    on_ramp = [(x, y) for (x, y) in S.ROUTE_RAMP if 47.0 <= x <= 51.6]
    assert on_ramp and all(0.3 <= y <= 1.3 for _, y in on_ramp)
    assert float(hf.height(*S.ROUTE_RAMP[-1])) == pytest.approx(-0.003, abs=0.01)


def test_rubble_passage_waypoints_are_clear_of_the_blocks():
    """(l) / (r) pass the C-OBS3 rubble field through the north passage: flat floor
    +-0.4 m around the passage line over the whole field (x 36.6 .. 40), rubble south of it."""
    hf = W.HeightField.plant()
    y = S.RUBBLE_PASS_Y
    for route in (S.ROUTE_FULL, S.SCENARIOS["r_rubble_detour"].route):
        assert sum(1 for (_, wy) in route if wy == y) >= 2
    xs, ys = np.meshgrid(np.arange(36.6, 40.01, 0.02), np.linspace(y - 0.4, y + 0.4, 41))
    assert np.all(np.abs(hf.height(xs, ys)) < 0.01)
    xs, ys = np.meshgrid(np.arange(37.0, 39.0, 0.02), np.arange(-1.3, 0.5, 0.02))
    assert float(np.max(hf.height(xs, ys))) > 0.03                      # the blocks are still there


def test_depth_camera_and_terrain_mapper_emulation():
    from gazebo_check.mapper import DepthCamera, TerrainMapperSim
    floor = {"name": "floor", "type": "box", "p": np.array([0.0, 0.0, -0.15]), "R": np.eye(3),
             "size": np.array([20.0, 20.0, 0.3])}
    step = {"name": "step", "type": "box", "p": np.array([1.6, 0.0, 0.05]), "R": np.eye(3),
            "size": np.array([0.4, 2.0, 0.1])}
    cam = DepthCamera([floor, step])
    pos, rpy = np.array([0.0, 0.0, 0.34]), np.zeros(3)
    d = cam.render(pos, rpy)
    s = rm.SENSORS["depth_camera"]
    assert d.shape == (s["height"], s["width"]) and d.dtype == np.float32
    # optical axis pitched 0.6 rad down from (0.34, 0, 0.34): it meets the floor at depth 0.34 / sin(0.6)
    T = cam.pose(pos, rpy)
    zc = float(T[2, 3])
    centre = float(d[s["height"] // 2, s["width"] // 2])
    assert centre == pytest.approx(zc / math.sin(0.6), rel=0.03)
    assert np.isinf(d).any() or np.isfinite(d).all()
    m = TerrainMapperSim(layout=None)
    for k in range(5):
        p = [0.05 * k, 0.0, 0.34, 0.0, 0.0, 0.0]
        m.integrate_depth(cam.render(p[:3], p[3:]), p, 0.1 * k)
    cloud, info = m.publish([0.2, 0.0, 0.34, 0.0, 0.0, 0.0], 0.5)
    assert cloud.ndim == 3 and cloud.shape[2] == 3
    from module1_locomotion.elevation_map import ElevationMap
    em = ElevationMap.from_cloud_xyz(cloud)
    assert em.height_at(1.0, 0.0) == pytest.approx(0.0, abs=0.01)                   # floor ahead
    assert em.height_at(1.6, 0.0) == pytest.approx(0.10, abs=0.02)                  # the step seen by the camera
    assert "class" in info and "recommended_gait" in info


def test_keyboard_operator_drives_a_unicycle_along_the_waypoints():
    wps = [(2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    op = S.KeyboardOperator(wps, t_start=0.5, scale=0.5)
    assert op.needs_pose and op.vx == pytest.approx(0.25) and op.wz == pytest.approx(0.4)
    x, y, yaw, dt = 0.0, 0.0, 0.0, 0.01
    assert op(0.0, (np.zeros(3), np.zeros(3))) == (0.0, 0.0, 0.0)
    for k in range(int(60.0 / dt)):
        t = k * dt
        vx, vy, wz = op(t, (np.array([x, y, 0.0]), np.array([0.0, 0.0, yaw])))
        assert vy == 0.0 and abs(vx) <= 0.25 + 1e-9 and abs(wz) <= 0.4 + 1e-9
        x += vx * math.cos(yaw) * dt
        y += vx * math.sin(yaw) * dt
        yaw += wz * dt
    r = op.result()
    assert r["wp_reached"] == 3 and r["wp_frac"] == 1.0 and r["route_done_t"] is not None
    # released keys run on for hold_timeout (teleop hold mode), then the robot stops
    t_end = r["route_done_t"]
    assert op(t_end + op.hold_timeout + 0.1, (np.array([x, y, 0.0]), np.array([0.0, 0.0, yaw]))) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# scenarios (no gz needed)
# ---------------------------------------------------------------------------
def test_scenarios_and_aliases():
    assert S.resolve(["a", "b", "c", "d", "e"]) == ["a_stand", "b_trot", "c_crawl", "d_turn", "e_teleop"]
    assert S.SCENARIOS["c_crawl"].gait == "crawl"
    assert S.gait_commands(S.SCENARIOS["b_trot"]) == [(S.T_CMD - 0.5, "gait:trot")]
    f = S.command_fn(S.SCENARIOS["b_trot"], 0)
    assert f(1.0) == (0.0, 0.0, 0.0) and f(S.T_CMD + 1.0) == (0.35, 0.0, 0.0)
    with pytest.raises(KeyError):
        S.resolve(["z"])
    q = S.with_duration(S.SCENARIOS["b_trot"], 8.0)
    assert q.duration == 8.0 and q.window[1] <= 8.0
    # constant commands start CMD_SHIFT later per seed (different gait phases at obstacles / pushes)
    f2 = S.command_fn(S.SCENARIOS["b_trot"], 2)
    assert f2(S.T_CMD + 0.1) == (0.0, 0.0, 0.0) and f2(S.T_CMD + 2 * S.CMD_SHIFT + 1e-6) == (0.35, 0.0, 0.0)
    # terrain / push scenarios carry their world, spawn and pushes
    assert S.resolve(["f", "g", "h", "i"]) == ["f_pipe", "g_ramp", "h_ramp_down", "i_push"]
    assert S.SCENARIOS["f_pipe"].terrain == "pipe" and S.SCENARIOS["g_ramp"].terrain == "ramp"
    h = S.SCENARIOS["h_ramp_down"]
    assert h.spawn[2] == pytest.approx(W.RAMP_TOP + 0.56, abs=0.005) and h.spawn[3] == pytest.approx(math.pi)
    assert W.RAMP_X1 < h.spawn[0] < W.RAMP_X1 + 4.0                 # on the plateau
    pushes = S.SCENARIOS["i_push"].pushes
    assert [p[1] * np.linalg.norm(p[2]) for p in pushes] == pytest.approx([10.0] * 3)   # 10 N s each
    # plant scenarios: real world, keyboard operator routes, late node start
    assert S.resolve(["j", "k", "l", "m"]) == ["j_corridor", "k_late_start", "l_corridor_full", "m_corridor_fast"]
    for k in ("j_corridor", "k_late_start", "l_corridor_full", "m_corridor_fast", "n_pipe_dock"):
        assert S.SCENARIOS[k].terrain == W.PLANT
    pipe = S.SCENARIOS["n_pipe_dock"]
    assert pipe.spawn[0] - 58.5 == pytest.approx(1.9, abs=0.05) and pipe.spawn[3] == pytest.approx(math.pi)
    assert pipe.route[-1][0] < 58.4                                  # beyond the pipe
    assert isinstance(S.command_fn(S.SCENARIOS["j_corridor"], 0), S.KeyboardOperator)
    assert S.command_fn(S.SCENARIOS["m_corridor_fast"], 0).vx == pytest.approx(0.5)
    late = S.SCENARIOS["k_late_start"]
    assert late.controller_start == S.T_LATE and late.t_cmd > late.controller_start
    f = S.command_fn(late, 0)
    assert f(late.controller_start + 1.0) == (0.0, 0.0, 0.0) and f(late.t_cmd + 0.01) == (0.25, 0.0, 0.0)


def test_teleop_segments_are_seeded_and_bounded():
    a = S.teleop_segments(3, 4.0, 64.0)
    assert a == S.teleop_segments(3, 4.0, 64.0)
    assert a != S.teleop_segments(4, 4.0, 64.0)
    assert a[0][0] == 4.0 and a[-1][1] == pytest.approx(64.0)
    for (t0, t1, c) in a:
        assert 0.0 < t1 - t0 <= 4.0
        for v, (lo, hi) in zip(c, S.TELEOP_LIMITS):
            assert lo <= v <= hi


def _fake_log(n=1500, fall_at=None):
    t = np.arange(n) * 0.01
    pos = np.zeros((n, 3))
    pos[:, 0] = np.where(t > S.T_CMD, 0.35 * (t - S.T_CMD), 0.0)
    pos[:, 2] = 0.34
    rpy = np.zeros((n, 3))
    if fall_at is not None:
        rpy[t >= fall_at, 0] = math.radians(80.0)
    mode = np.array(["STAND" if ti < S.T_CMD else "WALK" for ti in t], dtype=object)
    return {"t": t, "pos": pos, "rpy": rpy, "fall_metric": np.abs(rpy[:, 0]) > math.radians(55.0),
            "mode": mode, "tau": np.full((n, 12), 10.0), "mpc_ms": np.full(n, 20.0)}


def test_evaluate_pass_and_fall():
    sc = S.with_duration(S.SCENARIOS["b_trot"], 15.0)
    ok = S.evaluate(sc, _fake_log())
    assert ok["pass"], ok["reason"]
    assert ok["vx"] == pytest.approx(0.35, abs=1e-3) and ok["falls"] == 0
    assert ok["progress"] == pytest.approx(0.35 * (15.0 - 0.01 - 6.0), abs=0.01)
    bad = S.evaluate(sc, _fake_log(fall_at=9.0))
    assert not bad["pass"] and bad["falls"] == 1
    assert bad["first_fall_t"] == pytest.approx(9.0)
    assert "fell" in bad["reason"]


# ---------------------------------------------------------------------------
# novel obstacles: world extras, harness fields, mission in the loop (no gz needed)
# ---------------------------------------------------------------------------
NOVEL_BOX = {"name": "novel_box_0", "center": [43.0, 0.72], "size": [0.6, 0.6, 0.5], "yaw": 0.0, "floor_z": 0.0}


def test_world_extras_are_static_models_seen_by_the_depth_camera_and_truth(tmp_path):
    urdf = W.write_harness_urdf(str(tmp_path / "spot.urdf"), use_xacro=False)
    root = ET.fromstring(W.plant_world_sdf(urdf, extra=[NOVEL_BOX]))
    models = {m.get("name"): m for m in root.find("world").findall("model")}
    assert "novel_box_0" in models and models["novel_box_0"].findtext("static") == "true"
    shapes = [s for s in W.collision_shapes(root) if s["name"].startswith("novel_box_0/")]
    assert len(shapes) == 1 and np.allclose(shapes[0]["size"], [0.6, 0.6, 0.5])
    assert np.allclose(shapes[0]["p"], [43.0, 0.72, 0.25])
    world, _ = W.build(str(tmp_path), W.PLANT, (46.5, 0.72, 0.557, math.pi), tag="_x", extra=[NOVEL_BOX])
    assert "novel_box_0" in open(world, encoding="utf-8").read()
    flat = ET.fromstring(W.world_sdf(urdf, "flat", extra=[dict(NOVEL_BOX, center=[2.0, 0.0])]))
    assert "novel_box_0" in {m.get("name") for m in flat.find("world").findall("model")}
    fn, _ = W.terrain_height_fn("flat", [dict(NOVEL_BOX, center=[2.0, 0.0])])
    assert float(fn(2.0, 0.0)) == pytest.approx(0.5) and float(fn(3.0, 0.0)) == 0.0
    (s,) = W.extra_shapes([NOVEL_BOX])
    assert s["type"] == "box" and W.ray_hit(s, np.array([41.0, 0.72, 0.3]), np.array([1.0, 0.0, 0.0])) == \
        pytest.approx(1.7)
    assert W.world_sdf(urdf, "flat") == W.world_sdf(urdf, "flat", extra=())      # default unchanged


def test_harness_config_has_the_perception_fields():
    from gazebo_check.harness import HarnessConfig
    cfg = HarnessConfig()
    assert cfg.extra_objects == () and cfg.perception_depth is False
    assert HarnessConfig(perception_depth=True, extra_objects=(NOVEL_BOX,)).extra_objects[0]["name"] == "novel_box_0"


class _FakeHarness:
    def __init__(self, pose):
        import threading
        self._lock = threading.Lock()
        self._odom = (0.0, np.array([pose[0], pose[1], 0.34]), np.array([0.0, 0.0, pose[2]]), np.zeros(3), np.zeros(3))
        self.log = {"mode": []}
        self.commands = []
        self.elevation_cloud = None
        self.elevation_cloud_t = None


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "simulation", "maps", "plant_map.yaml")),
                    reason="plant_map.yaml missing")
def test_mission_operator_with_a_fake_harness():
    sc = S.MISSION_SCENARIOS["v_box_upper"]
    op = sc.operator()
    assert op.needs_pose and not op.bound
    h = _FakeHarness((46.5, 0.72, math.pi))
    op.bind(h)
    assert op.bound and op.h is h
    gt = (np.array([46.5, 0.72, 0.34]), np.array([0.0, 0.0, math.pi]))
    assert op(0.0, gt) == (0.0, 0.0, 0.0)                        # before the mission starts
    h.log["mode"].append("STAND")
    t = 0.0
    while t < sc.t_start + 0.05:
        t += 0.01
        op(t, gt)
    assert [c for _, c in h.commands] == ["stand"]               # forwarded to the controller
    assert op.m.base == "PATROL"
    # a terrain_mapper cloud (flat floor + the box) goes in like on /elevation_map/points
    n, res = 150, 0.04
    X, Y = np.meshgrid(46.5 - 3.0 + res * np.arange(n), 0.72 - 3.0 + res * np.arange(n))
    Z = np.where((np.abs(X - 45.4) <= 0.3) & (np.abs(Y - 0.72) <= 0.3), 0.5, 0.0)
    cmd = None
    for k in range(400):
        t += 0.01
        if k % 20 == 0:
            h.elevation_cloud, h.elevation_cloud_t = np.stack([X, Y, Z], -1).astype(np.float32), t
        cmd = op(t, gt)
    assert op.frames >= 2 and op.m.obstacles.memory.has_confirmed()
    assert cmd != (0.0, 0.0, 0.0)
    op._next = 1e18                                              # no more mission ticks: mux timeout
    assert op(t + 0.4, gt) == cmd and op(t + 0.7, gt) == (0.0, 0.0, 0.0)
    r = op.result()
    for k in ("reached", "reached_V1", "min_box_clearance", "detours", "obstacle_replans", "stuck_replans",
              "missed", "debris", "debris_n", "side", "slow_frac", "obstacle_errors", "frames", "events",
              "path_len", "plans"):
        assert k in r, k
    json.dumps(r)


def test_mission_scenarios_resolve_without_touching_the_cli_scenarios():
    keys = S.resolve_mission(["r", "t", "u", "v", "w", "x", "y", "z"])
    assert keys == ["r_control_leg1", "t_box_start", "u_box_lower", "v_box_upper", "w_wall_upper",
                    "x_control_upper", "y_route_no_box", "z_control_lower"]
    for k in keys:
        sc = S.MISSION_SCENARIOS[k]
        assert sc.scenario().terrain == W.PLANT and sc.plan()["order"]
        for e in sc.extra:
            assert e["floor_z"] in (S.Z_LOWER, S.Z_UPPER)
        assert sc.spawn[2] == pytest.approx((S.Z_LOWER if sc.spawn[0] > 51.5 else S.Z_UPPER) + S.SPAWN_DZ)
    for k in keys:                                                   # controls exist and carry no box
        c = S.MISSION_SCENARIOS[k].control
        assert c is None or (c in S.MISSION_SCENARIOS and not S.MISSION_SCENARIOS[c].extra)
    assert not set(S.MISSION_SCENARIOS) & set(S.SCENARIOS)
    with pytest.raises(KeyError):
        S.resolve(["t"])                                         # the CLI scenario list is unchanged
    with pytest.raises(KeyError):
        S.resolve_mission(["q"])


# ---------------------------------------------------------------------------
# closed loop in gz-sim (needs the gz python bindings)
# ---------------------------------------------------------------------------
GZ_PYTHON = os.environ.get("GZ_PYTHON", "")


@pytest.mark.skipif(not (GZ_PYTHON and os.path.exists(GZ_PYTHON)),
                    reason="set GZ_PYTHON to a python with the gz-sim 8 bindings (gz.sim8, gz.transport13)")
@pytest.mark.parametrize("mode", ["stable", "mpc"])
def test_closed_loop_stand_and_trot_in_gazebo(tmp_path, mode):
    env = dict(os.environ)
    if os.name == "nt":       # conda layout: DLLs next to python.exe / Library/bin
        base = os.path.dirname(GZ_PYTHON)
        extra = [base, os.path.join(base, "Library", "mingw-w64", "bin"), os.path.join(base, "Library", "usr", "bin"),
                 os.path.join(base, "Library", "bin"), os.path.join(base, "Scripts")]
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        env["CONDA_PREFIX"] = base
    out = tmp_path / "gz.json"
    cmd = [GZ_PYTHON, CLI, "--scenarios", "a", "b", "--seeds", "0", "--duration", "9", "--jobs", "2",
           "--label", "pytest", "--out", str(out), "--work-dir", str(tmp_path), "--mode", mode]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["runs"]["pytest"]["config"]["params"]["locomotion_mode"] == mode
    sc = data["runs"]["pytest"]["scenarios"]
    stand, trot = sc["a_stand"]["0"], sc["b_trot"]["0"]
    assert stand["falls"] == 0 and stand["height_mean"] == pytest.approx(rm.NOMINAL_BODY_HEIGHT, abs=0.03)
    assert trot["falls"] == 0, trot["reason"]
    assert trot["vx"] > 0.2


@pytest.mark.skipif(not (GZ_PYTHON and os.path.exists(GZ_PYTHON)),
                    reason="set GZ_PYTHON to a python with the gz-sim 8 bindings (gz.sim8, gz.transport13)")
def test_closed_loop_plant_world_dock_start_in_gazebo(tmp_path):
    """The real plant.sdf: spawn at the dock (lower corridor), stand up, walk
    west with the keyboard operator (first metres, before the pipe).  Ground-
    truth map: with ``--map depth`` the repository controller stumbles right
    after the start (``_terrain_pitch`` with unknown cells under the rear
    feet, module1_locomotion/README.md section 4.7)."""
    env = dict(os.environ)
    if os.name == "nt":
        base = os.path.dirname(GZ_PYTHON)
        extra = [base, os.path.join(base, "Library", "mingw-w64", "bin"), os.path.join(base, "Library", "usr", "bin"),
                 os.path.join(base, "Library", "bin"), os.path.join(base, "Scripts")]
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        env["CONDA_PREFIX"] = base
    out = tmp_path / "gz.json"
    cmd = [GZ_PYTHON, CLI, "--scenarios", "j", "--seeds", "0", "--duration", "11", "--jobs", "1",
           "--label", "pytest", "--out", str(out), "--work-dir", str(tmp_path), "--map", "truth"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    r = json.loads(out.read_text(encoding="utf-8"))["runs"]["pytest"]["scenarios"]["j_corridor"]["0"]
    assert r["falls"] == 0, r["reason"]
    assert r["height_mean"] == pytest.approx(rm.NOMINAL_BODY_HEIGHT, abs=0.04)      # above the lower floor
    assert r["final_pos"][0] < 61.9 - 1.0 and r["wp_reached"] >= 1


_MISSION_RUN = r"""
import json, os, sys
root = sys.argv[1]
for p in ("dusan_common", "module1_locomotion", "simulation", "tools"):
    sys.path.insert(0, os.path.join(root, p))
from gazebo_check import scenarios as S
from gazebo_check.harness import GzHarness, HarnessConfig
sc = S.MISSION_SCENARIOS["v_box_upper"]
h = GzHarness(HarnessConfig(terrain="plant", spawn=sc.spawn, perception_depth=True, extra_objects=tuple(sc.extra),
                            out_dir=sys.argv[2]))
op = sc.operator()
h.cmd_fn = op
h.run(float(sys.argv[3]))
A = h.arrays()
r = op.result()
r["falls"] = len(S.fall_episodes(A["t"], A["fall_metric"], A["mode"]))
r.pop("events", None)
json.dump(r, open(sys.argv[4], "w"))
h.close()
"""


@pytest.mark.skipif(not (GZ_PYTHON and os.path.exists(GZ_PYTHON)),
                    reason="set GZ_PYTHON to a python with the gz-sim 8 bindings (gz.sim8, gz.transport13)")
def test_mission_bypasses_a_novel_box_in_gazebo(tmp_path):
    """Mission in the loop in the real plant.sdf: a box that is not in the nav
    map stands on the route in the upper corridor (scenario v); the mission
    sees it only through the emulated depth camera -> terrain_mapper cloud."""
    env = dict(os.environ)
    if os.name == "nt":
        base = os.path.dirname(GZ_PYTHON)
        extra = [base, os.path.join(base, "Library", "mingw-w64", "bin"), os.path.join(base, "Library", "usr", "bin"),
                 os.path.join(base, "Library", "bin"), os.path.join(base, "Scripts")]
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        env["CONDA_PREFIX"] = base
    env["GZ_PARTITION"] = f"pytest_mission_{os.getpid()}"
    out = tmp_path / "mission.json"
    p = subprocess.run([GZ_PYTHON, "-c", _MISSION_RUN, ROOT, str(tmp_path), "40", str(out)], env=env,
                       capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["falls"] == 0 and r["reached_V1"] == 1.0, r
    assert r["detours"] >= 1 and r["min_box_clearance"] >= 0.35 and r["stuck_replans"] == 0
