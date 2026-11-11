"""
Validation of the Spot robot assets (no ROS / Gazebo needed):

* simulation/urdf/spot.urdf(.xacro)  - structure, kinematics vs dusan_common.robot_model,
                                        masses / inertias, collisions, Gazebo plugins + sensors
* simulation/config/ros_gz_bridge.yaml - every DESIGN_SPEC section 3 bridge entry
* simulation/models/spot/meshes        - OBJ/MTL integrity, triangle budget, link frames, size
* tools/convert_spot.py               - unit tests of the geometry algorithms on synthetic data

Run:  python -m pytest tools/test/test_spot_assets.py -q
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

import convert_spot as cs
from dusan_common import robot_model as rm

REPO = cs.REPO
URDF = os.path.join(REPO, "simulation", "urdf", "spot.urdf")
XACRO = os.path.join(REPO, "simulation", "urdf", "spot.urdf.xacro")
BRIDGE = os.path.join(REPO, "simulation", "config", "ros_gz_bridge.yaml")
MODEL_DIR = os.path.join(REPO, "simulation", "models", "spot")
MESH_DIR = os.path.join(MODEL_DIR, "meshes")
REPORT = os.path.join(MODEL_DIR, "conversion_report.json")
PKG_PREFIX = "package://plant_simulation/"


# =============================================================================
# helpers
# =============================================================================
def _floats(s):
    return np.array([float(v) for v in s.split()], dtype=float)


def _rpy_to_rot(r, p, y):
    return cs.rot_z(y) @ cs.rot_y(p) @ cs.rot_x(r)


def _origin(el):
    o = el.find("origin")
    if o is None:
        return np.zeros(3), np.zeros(3)
    xyz = _floats(o.get("xyz", "0 0 0"))
    rpy = _floats(o.get("rpy", "0 0 0"))
    return xyz, rpy


def _origin_T(el):
    xyz, rpy = _origin(el)
    return cs.homogeneous(_rpy_to_rot(*rpy), xyz)


def _axis_rot(axis, q):
    """Rodrigues rotation about a unit axis."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


@pytest.fixture(scope="module")
def urdf():
    assert os.path.isfile(URDF), "simulation/urdf/spot.urdf missing (run tools/convert_spot.py --urdf-only)"
    return ET.parse(URDF).getroot()


@pytest.fixture(scope="module")
def joints(urdf):
    return {j.get("name"): j for j in urdf.findall("joint")}


@pytest.fixture(scope="module")
def links(urdf):
    return {lk.get("name"): lk for lk in urdf.findall("link")}


def urdf_link_pose(joints, link, q):
    """Pose of `link` in base_link by chaining URDF joints (q: joint -> angle)."""
    by_child = {j.find("child").get("link"): j for j in joints.values()}
    chain = []
    cur = link
    while cur != "base_link":
        j = by_child[cur]
        chain.append(j)
        cur = j.find("parent").get("link")
    T = np.eye(4)
    for j in reversed(chain):
        T = T @ _origin_T(j)
        if j.get("type") in ("revolute", "continuous"):
            axis = _floats(j.find("axis").get("xyz"))
            T = T @ cs.homogeneous(_axis_rot(axis, q.get(j.get("name"), 0.0)))
    return T


def reference_foot(leg_index, q0, q1, q2):
    """Independent foot FK from the DESIGN_SPEC / robot_model formula."""
    sy = rm.LEG_SIGN_Y[leg_index]
    inner = np.array([0, 0, -rm.UPPER_LEG_LENGTH]) + cs.rot_y(q2) @ np.array([0, 0, -rm.LOWER_LEG_LENGTH])
    return np.asarray(rm.hip_x_position(leg_index)) + cs.rot_x(q0) @ (
        np.array([0, sy * rm.HIP_TO_THIGH_Y, 0]) + cs.rot_y(q1) @ inner)


def gazebo_plugins(urdf, filename):
    out = []
    for gz in urdf.findall("gazebo"):
        if gz.get("reference") is None:
            out += [p for p in gz.findall("plugin") if p.get("filename") == filename]
    return out


def gazebo_sensor(urdf, name):
    for gz in urdf.findall("gazebo"):
        for s in gz.findall("sensor"):
            if s.get("name") == name:
                return gz.get("reference"), s
    return None, None


# =============================================================================
# URDF structure + kinematics
# =============================================================================
def test_urdf_links(urdf, links):
    assert urdf.get("name") == rm.ROBOT_NAME
    expected = {"base_link", "imu_link", "depth_camera_link", "inspection_camera_link",
                "thermal_camera_link"}
    for leg in rm.LEG_NAMES:
        expected |= {f"{leg}_hip", f"{leg}_upper_leg", f"{leg}_lower_leg", f"{leg}_foot"}
    assert expected <= set(links)
    for cam in ("depth_camera", "inspection_camera", "thermal_camera"):
        assert f"{cam}_optical_frame" in links


def test_revolute_joints(joints):
    rev = {n: j for n, j in joints.items() if j.get("type") == "revolute"}
    assert set(rev) == set(rm.JOINT_NAMES)
    assert len(rev) == 12
    for i, leg in enumerate(rm.LEG_NAMES):
        sx, sy = rm.LEG_SIGN_X[i], rm.LEG_SIGN_Y[i]
        spec = {
            "hip_x": ("base_link", f"{leg}_hip", (1, 0, 0), rm.hip_x_position(i)),
            "hip_y": (f"{leg}_hip", f"{leg}_upper_leg", (0, 1, 0), (0, sy * rm.HIP_TO_THIGH_Y, 0)),
            "knee": (f"{leg}_upper_leg", f"{leg}_lower_leg", (0, 1, 0), (0, 0, -rm.UPPER_LEG_LENGTH)),
        }
        assert rm.hip_x_position(i) == (sx * rm.HIP_X_OFFSET, sy * rm.HIP_Y_OFFSET, 0.0)
        for suffix, (parent, child, axis, xyz) in spec.items():
            j = rev[f"{leg}_{suffix}"]
            assert j.find("parent").get("link") == parent
            assert j.find("child").get("link") == child
            np.testing.assert_allclose(_floats(j.find("axis").get("xyz")), axis)
            o_xyz, o_rpy = _origin(j)
            np.testing.assert_allclose(o_xyz, xyz, atol=1e-9)
            np.testing.assert_allclose(o_rpy, 0.0, atol=1e-12)
            lim = j.find("limit")
            ref = rm.JOINT_LIMITS[suffix]
            for key in ("lower", "upper", "effort", "velocity"):
                assert float(lim.get(key)) == pytest.approx(ref[key])
            dyn = j.find("dynamics")
            assert float(dyn.get("damping")) == pytest.approx(0.1)
            assert float(dyn.get("friction")) == pytest.approx(0.05)
        foot = joints[f"{leg}_foot_fixed"]
        assert foot.get("type") == "fixed"
        assert foot.find("parent").get("link") == f"{leg}_lower_leg"
        np.testing.assert_allclose(_origin(foot)[0], (0, 0, -rm.LOWER_LEG_LENGTH), atol=1e-12)


def test_fk_urdf_equals_robot_model(joints):
    rng = np.random.default_rng(42)
    for _trial in range(40):
        q = {}
        for name in rm.JOINT_NAMES:
            lim = rm.JOINT_LIMITS[name.split("_", 1)[1]]
            q[name] = rng.uniform(lim["lower"], lim["upper"])
        for i, leg in enumerate(rm.LEG_NAMES):
            qs = [q[f"{leg}_{s}"] for s in rm.JOINT_SUFFIXES]
            p_urdf = urdf_link_pose(joints, f"{leg}_foot", q)[:3, 3]
            np.testing.assert_allclose(p_urdf, reference_foot(i, *qs), atol=1e-9)
            # the converter's FK (used to place the mesh vertices) agrees too
            frames = cs.leg_link_frames(i, qs)
            for link in ("hip", "upper_leg", "lower_leg", "foot"):
                np.testing.assert_allclose(urdf_link_pose(joints, f"{leg}_{link}", q),
                                           frames[f"{leg}_{link}"], atol=1e-9)
    # zero pose: legs straight down, foot at -(L1+L2) below the hip_y joint
    for i, leg in enumerate(rm.LEG_NAMES):
        p = urdf_link_pose(joints, f"{leg}_foot", {})[:3, 3]
        np.testing.assert_allclose(p, np.asarray(rm.hip_y_position_zero(i))
                                   - [0, 0, rm.UPPER_LEG_LENGTH + rm.LOWER_LEG_LENGTH], atol=1e-12)


def test_joint_sign_convention(joints):
    """hip_y > 0 swings the foot backwards, knee < 0 bends the knee backwards."""
    for leg in rm.LEG_NAMES:
        p0 = urdf_link_pose(joints, f"{leg}_foot", {})[:3, 3]
        p1 = urdf_link_pose(joints, f"{leg}_foot", {f"{leg}_hip_y": 0.3})[:3, 3]
        assert p1[0] < p0[0] - 0.1
        q = {f"{leg}_hip_y": 0.8, f"{leg}_knee": -1.6}
        knee = urdf_link_pose(joints, f"{leg}_lower_leg", q)[:3, 3]
        hip = urdf_link_pose(joints, f"{leg}_upper_leg", q)[:3, 3]
        foot = urdf_link_pose(joints, f"{leg}_foot", q)[:3, 3]
        assert knee[0] < hip[0] and knee[0] < foot[0]       # knee points backwards


def _inertial(link):
    inn = link.find("inertial")
    if inn is None:
        return None
    m = float(inn.find("mass").get("value"))
    I = inn.find("inertia")
    get = lambda k: float(I.get(k))  # noqa: E731
    M = np.array([[get("ixx"), get("ixy"), get("ixz")],
                  [get("ixy"), get("iyy"), get("iyz")],
                  [get("ixz"), get("iyz"), get("izz")]])
    return m, M, _origin(inn)[0]


def test_masses_and_inertias(links):
    total = 0.0
    for name, link in links.items():
        inn = _inertial(link)
        if inn is None:
            continue
        m, M, _c = inn
        total += m
        ev = np.linalg.eigvalsh(M)
        assert np.all(ev > 0), name
        a, b, c = sorted(ev)
        assert c <= a + b + 1e-12, f"{name}: inertia violates the triangle inequality"
    assert total == pytest.approx(rm.TOTAL_MASS, abs=1e-9)
    assert total == pytest.approx(15.0)
    m, M, c = _inertial(links["base_link"])
    assert m == pytest.approx(rm.BASE_MASS)
    np.testing.assert_allclose(np.diag(M), rm.BASE_INERTIA)
    for leg in rm.LEG_NAMES:
        assert _inertial(links[f"{leg}_hip"])[0] == pytest.approx(rm.HIP_MASS)
        assert _inertial(links[f"{leg}_upper_leg"])[0] == pytest.approx(rm.UPPER_LEG_MASS)
        assert _inertial(links[f"{leg}_lower_leg"])[0] == pytest.approx(rm.LOWER_LEG_MASS)
        # COM roughly mid-link (parallel-axis placement)
        cu = _inertial(links[f"{leg}_upper_leg"])[2]
        cl = _inertial(links[f"{leg}_lower_leg"])[2]
        assert -0.7 * rm.UPPER_LEG_LENGTH < cu[2] < -0.3 * rm.UPPER_LEG_LENGTH
        assert -0.7 * rm.LOWER_LEG_LENGTH < cl[2] < -0.3 * rm.LOWER_LEG_LENGTH
        assert _inertial(links[f"{leg}_foot"]) is None           # lumped, massless


def _mesh_path(uri):
    assert uri.startswith(PKG_PREFIX), uri
    return os.path.join(REPO, "simulation", *uri[len(PKG_PREFIX):].split("/"))


def test_visual_meshes_exist(urdf, links):
    uris = [m.get("filename") for m in urdf.iter("mesh")]
    assert len(uris) == 3 + 12
    for uri in uris:
        path = _mesh_path(uri)
        assert os.path.isfile(path), uri
        assert uri.endswith(".obj")
    for leg in rm.LEG_NAMES:
        for seg in ("hip", "upper_leg", "lower_leg"):
            vis = links[f"{leg}_{seg}"].findall("visual")
            assert vis[0].find("geometry/mesh").get("filename").endswith(f"/{leg}_{seg}.obj")
            assert vis[0].find("material") is None       # mesh (MTL) colours are used
    names = {os.path.basename(u) for u in uris}
    assert {"base_link.obj", "base_cam.obj", "base_eap.obj"} <= names


def test_collisions(urdf, links):
    base = links["base_link"].findall("collision")
    assert len(base) == 1
    np.testing.assert_allclose(_floats(base[0].find("geometry/box").get("size")), rm.BASE_BOX)
    # centred on the trunk mesh
    obj = cs.read_obj(os.path.join(MESH_DIR, "base_link.obj"))
    c_mesh = 0.5 * (obj["vertices"].min(0) + obj["vertices"].max(0))
    c_box = _origin(base[0])[0]
    assert np.abs(c_box - c_mesh).max() < 0.006
    for leg in rm.LEG_NAMES:
        foot = links[f"{leg}_foot"].findall("collision")
        assert len(foot) == 1
        assert float(foot[0].find("geometry/sphere").get("radius")) == pytest.approx(rm.FOOT_RADIUS)
        np.testing.assert_allclose(_origin(foot[0])[0], 0.0)
        assert links[f"{leg}_lower_leg"].find("collision/geometry/cylinder") is not None
        assert links[f"{leg}_upper_leg"].find("collision/geometry/box") is not None
        # shin collision must stay above the foot sphere so that the sphere is the contact
        shin = links[f"{leg}_lower_leg"].find("collision")
        cz = _origin(shin)[0][2]
        half = 0.5 * float(shin.find("geometry/cylinder").get("length"))
        assert cz - half > -rm.LOWER_LEG_LENGTH
    for s in rm.SENSORS.values():
        assert links[s["link"]].find("collision") is None


def test_foot_friction(urdf):
    refs = {}
    for gz in urdf.findall("gazebo"):
        if gz.get("reference"):
            refs.setdefault(gz.get("reference"), []).append(gz)
    for leg in rm.LEG_NAMES:
        blocks = refs.get(f"{leg}_foot", [])
        mu1 = [float(b.find("mu1").text) for b in blocks if b.find("mu1") is not None]
        mu2 = [float(b.find("mu2").text) for b in blocks if b.find("mu2") is not None]
        assert mu1 == [1.0] and mu2 == [1.0]


def test_urdf_is_a_single_tree(urdf):
    """Unique names, joints between existing links, one parent per link and a
    single root base_link from which every link is reachable (KDL /
    robot_state_publisher and sdformat both need a tree)."""
    link_names = [lk.get("name") for lk in urdf.findall("link")]
    joint_list = urdf.findall("joint")
    assert len(link_names) == len(set(link_names))
    assert len({j.get("name") for j in joint_list}) == len(joint_list)
    children = [j.find("child").get("link") for j in joint_list]
    parents = [j.find("parent").get("link") for j in joint_list]
    assert len(children) == len(set(children)), "a link has two parent joints"
    assert set(children) | set(parents) <= set(link_names)
    assert set(link_names) - set(children) == {"base_link"}
    below = {}
    for p, c in zip(parents, children):
        below.setdefault(p, []).append(c)
    seen, stack = set(), ["base_link"]
    while stack:
        cur = stack.pop()
        assert cur not in seen, "kinematic loop"
        seen.add(cur)
        stack += below.get(cur, [])
    assert seen == set(link_names)


def test_gazebo_extensions_reference_existing_links(urdf, links):
    for gz in urdf.findall("gazebo"):
        ref = gz.get("reference")
        assert ref is None or ref in links, ref


def test_lumped_links_for_sdformat(links):
    """Feet, sensor links and optical frames are massless fixed-joint children,
    so sdformat's URDF->SDF converter lumps them into their parents (sensors,
    visuals, collisions and surface parameters follow, the links remain as SDF
    <frame>s).  The foot collision is named "<leg>_foot_collision": the lumped
    SDF collision name then contains "_fixed_joint_lump__<leg>_foot", which is
    how sdformat matches the <gazebo reference="<leg>_foot"> mu1/mu2 to it."""
    for leg in rm.LEG_NAMES:
        foot = links[f"{leg}_foot"]
        assert foot.find("inertial") is None
        assert [c.get("name") for c in foot.findall("collision")] == [f"{leg}_foot_collision"]
    frames = [s["link"] for s in rm.SENSORS.values()]
    frames += [f"{cam}_optical_frame" for cam in ("depth_camera", "inspection_camera", "thermal_camera")]
    for name in frames:
        assert links[name].find("inertial") is None, name
        assert links[name].find("collision") is None, name


def _stand_q(body_height):
    q = {}
    for leg, angles in cs.stand_pose(body_height).items():
        for suffix, a in zip(rm.JOINT_SUFFIXES, angles):
            q[f"{leg}_{suffix}"] = a
    return q


def test_only_the_feet_touch_the_ground(urdf):
    """At the spawn pose (q = 0) and standing at the nominal / min / max / sit
    heights, the foot spheres are the lowest collision geometry by >= 3 cm:
    shins, thighs and trunk never carry the robot."""
    poses = {"zero": {}}
    for h in (rm.NOMINAL_BODY_HEIGHT, rm.MIN_BODY_HEIGHT, rm.MAX_BODY_HEIGHT, rm.SIT_BODY_HEIGHT):
        poses[f"stand {h:.2f}"] = _stand_q(h)
    for pname, q in poses.items():
        frames = cs.urdf_link_frames(urdf, q)
        foot_bottom = min(frames[f"{leg}_foot"][2, 3] for leg in rm.LEG_NAMES) - rm.FOOT_RADIUS
        lowest = np.inf
        for lk in urdf.findall("link"):
            if lk.get("name").endswith("_foot"):
                continue
            for col in lk.findall("collision"):
                V, _F = cs.primitive_mesh(col.find("geometry"))
                lowest = min(lowest, cs.transform(frames[lk.get("name")] @ cs._xyz_rpy(col), V)[:, 2].min())
        assert lowest > foot_bottom + 0.03, (pname, lowest, foot_bottom)
    # DESIGN_SPEC spawn: straight legs with the base 0.56 m above the floor -> the feet just clear it
    assert 0.0 < 0.56 - (rm.UPPER_LEG_LENGTH + rm.LOWER_LEG_LENGTH + rm.FOOT_RADIUS) < 0.03


def test_cameras_do_not_see_the_robot(urdf):
    """No vertex of the robot's own visual meshes lies inside a camera frustum
    (beyond the near clip) at the zero, stand and sit poses, i.e. the gauge
    reader / thermal / depth images are not occluded by the payload or legs."""
    meshes = {f[:-4]: cs.read_obj(os.path.join(MESH_DIR, f))["vertices"]
              for f in os.listdir(MESH_DIR) if f.endswith(".obj")}
    for q in ({}, _stand_q(rm.NOMINAL_BODY_HEIGHT), _stand_q(rm.SIT_BODY_HEIGHT)):
        frames = cs.urdf_link_frames(urdf, q)
        for cam in ("depth_camera", "inspection_camera", "thermal_camera"):
            s = rm.SENSORS[cam]
            to_cam = cs.inv_rigid(frames[s["link"]])
            th = math.tan(0.5 * s["hfov"])
            tv = th * s["height"] / s["width"]
            for name, V in meshes.items():
                P = cs.transform(to_cam @ frames[cs.mesh_frame_of(name)], V)    # camera link: x forward
                inside = ((P[:, 0] > s["near"]) & (np.abs(P[:, 1]) < th * P[:, 0])
                          & (np.abs(P[:, 2]) < tv * P[:, 0]))
                assert not inside.any(), (cam, name, int(inside.sum()))


def test_urdf_parser_py_loads():
    """The URDF loads with urdf_parser_py (same model as urdfdom used by
    robot_state_publisher) when that dev package is installed."""
    urdf_mod = pytest.importorskip("urdf_parser_py.urdf")
    with open(URDF, encoding="utf-8") as f:
        robot = urdf_mod.URDF.from_xml_string(f.read())
    assert robot.get_root() == "base_link"
    assert sorted(j.name for j in robot.joints if j.type == "revolute") == sorted(rm.JOINT_NAMES)
    assert sum(lk.inertial.mass for lk in robot.links if lk.inertial is not None) == pytest.approx(rm.TOTAL_MASS)


def test_sensor_frames(joints):
    by_child = {j.find("child").get("link"): j for j in joints.values()}
    for name, s in rm.SENSORS.items():
        j = by_child[s["link"]]
        assert j.get("type") == "fixed"
        assert j.find("parent").get("link") == "base_link"
        xyz, rpy = _origin(j)
        np.testing.assert_allclose(xyz, s["pose"][:3], atol=1e-12)
        np.testing.assert_allclose(rpy, s["pose"][3:], atol=1e-12)
    # optical frame: z forward (= +x of the camera link), x right (= -y), y down (= -z)
    for cam in ("depth_camera", "inspection_camera", "thermal_camera"):
        R = _origin_T(joints[f"{cam}_optical_joint"])[:3, :3]
        np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, -1], atol=1e-12)


# =============================================================================
# Gazebo plugins / sensors
# =============================================================================
def test_joint_position_controllers(urdf):
    plugins = gazebo_plugins(urdf, "gz-sim-joint-position-controller-system")
    assert len(plugins) == 12
    seen = set()
    for p in plugins:
        assert p.get("name") == "gz::sim::systems::JointPositionController"
        j = p.find("joint_name").text
        seen.add(j)
        assert p.find("topic").text == rm.joint_command_topic(j)
        # PID force mode (DESIGN_SPEC §3): force clamped to the joint effort
        assert p.find("use_velocity_commands").text.strip().lower() == "false"
        lim = rm.JOINT_LIMITS[j.split("_", 1)[1]]
        assert float(p.find("cmd_max").text) == pytest.approx(lim["effort"])
        assert float(p.find("cmd_min").text) == pytest.approx(-lim["effort"])
        assert float(p.find("p_gain").text) == pytest.approx(rm.JOINT_SERVO_P_GAIN)
        assert float(p.find("d_gain").text) == pytest.approx(rm.JOINT_SERVO_D_GAIN)
        assert float(p.find("i_gain").text) == 0.0
        init = float(p.find("initial_position").text)
        assert lim["lower"] <= init <= lim["upper"]
    assert seen == set(rm.JOINT_NAMES)


def test_state_and_odometry_plugins(urdf):
    jsp = gazebo_plugins(urdf, "gz-sim-joint-state-publisher-system")
    assert len(jsp) == 1 and jsp[0].get("name") == "gz::sim::systems::JointStatePublisher"
    # default topic /world/<world>/model/<model>/joint_state == robot_model.GZ_JOINT_STATE_TOPIC
    assert jsp[0].find("topic") is None
    assert rm.GZ_JOINT_STATE_TOPIC == "/world/plant/model/spot/joint_state"
    # throttled (default would be every 1 ms physics step) but faster than the 100 Hz control loop
    assert 100.0 <= float(jsp[0].find("update_rate").text) <= 1000.0
    odo = gazebo_plugins(urdf, "gz-sim-odometry-publisher-system")
    assert len(odo) == 1
    o = odo[0]
    assert o.get("name") == "gz::sim::systems::OdometryPublisher"
    assert o.find("odom_frame").text == "odom"
    assert o.find("robot_base_frame").text == "base_link"
    assert int(o.find("dimensions").text) == 3
    assert float(o.find("odom_publish_frequency").text) == pytest.approx(50.0)
    assert o.find("odom_topic").text == rm.GZ_ODOM_TOPIC


def _camera_checks(sensor, spec, fmt):
    cam = sensor.find("camera")
    assert float(cam.find("horizontal_fov").text) == pytest.approx(spec["hfov"])
    assert int(cam.find("image/width").text) == spec["width"]
    assert int(cam.find("image/height").text) == spec["height"]
    assert cam.find("image/format").text == fmt
    assert float(cam.find("clip/near").text) == pytest.approx(spec["near"])
    assert float(cam.find("clip/far").text) == pytest.approx(spec["far"])
    assert float(sensor.find("update_rate").text) == pytest.approx(spec["rate"])
    assert sensor.find("topic").text == spec["topic"]
    assert sensor.find("always_on").text.strip().lower() in ("true", "1")


def test_sensors(urdf):
    ref, imu = gazebo_sensor(urdf, "imu")
    assert ref == "imu_link" and imu.get("type") == "imu"
    assert imu.find("topic").text == "/imu"
    assert float(imu.find("update_rate").text) == pytest.approx(rm.SENSORS["imu"]["rate"])
    assert imu.find("gz_frame_id").text == "imu_link"

    ref, depth = gazebo_sensor(urdf, "depth_camera")
    spec = rm.SENSORS["depth_camera"]
    assert ref == spec["link"] and depth.get("type") == "depth_camera"
    _camera_checks(depth, spec, "R_FLOAT32")
    assert depth.find("camera/camera_info_topic").text == spec["info_topic"]
    assert depth.find("gz_frame_id").text == spec["link"]
    # ros_gz convention: the point cloud is published on <image topic>/points
    assert depth.find("topic").text + "/points" == "/depth_camera/image/points"

    ref, insp = gazebo_sensor(urdf, "inspection_camera")
    spec = rm.SENSORS["inspection_camera"]
    assert ref == spec["link"] and insp.get("type") == "camera"
    _camera_checks(insp, spec, "R8G8B8")
    assert insp.find("camera/camera_info_topic").text == spec["info_topic"]
    assert insp.find("gz_frame_id").text == spec["link"]

    ref, th = gazebo_sensor(urdf, "thermal_camera")
    spec = rm.SENSORS["thermal_camera"]
    assert ref == spec["link"] and th.get("type") == "thermal"
    _camera_checks(th, spec, "L16")
    assert th.find("gz_frame_id").text == spec["link"]
    plug = th.find("plugin")
    assert plug.get("filename") == "gz-sim-thermal-sensor-system"
    assert plug.get("name") == "gz::sim::systems::ThermalSensor"
    assert float(plug.find("min_temp").text) == pytest.approx(spec["min_temp"])
    assert float(plug.find("max_temp").text) == pytest.approx(spec["max_temp"])
    assert float(plug.find("resolution").text) == pytest.approx(rm.THERMAL_RESOLUTION)
    # 16-bit range covers max_temp at this resolution
    assert 65535 * rm.THERMAL_RESOLUTION >= spec["max_temp"]


# =============================================================================
# xacro <-> committed URDF
# =============================================================================
def _canon(el):
    """Comment-free, whitespace-normalised, number-tolerant element signature."""
    def norm(v):
        parts = v.split()
        try:
            return tuple(round(float(p), 9) for p in parts)
        except ValueError:
            return v.strip()
    attrs = tuple(sorted((k, norm(v)) for k, v in el.attrib.items()))
    text = norm(el.text) if el.text and el.text.strip() else None
    kids = tuple(_canon(c) for c in el if isinstance(c.tag, str))
    return (el.tag, attrs, text, kids)


def _xacro():
    return pytest.importorskip("xacro")


def test_xacro_expands_to_committed_urdf():
    xacro = _xacro()
    doc = xacro.process_file(XACRO)
    expanded = ET.fromstring(doc.toxml())
    assert _canon(expanded) == _canon(ET.parse(URDF).getroot()), \
        "spot.urdf is stale: run python tools/convert_spot.py --urdf-only"


def test_xacro_args():
    xacro = _xacro()
    root = ET.fromstring(xacro.process_file(XACRO, mappings={"enable_gazebo": "false"}).toxml())
    assert not [p for p in root.iter("plugin")]
    assert not [s for s in root.iter("sensor")]
    assert len([j for j in root.findall("joint") if j.get("type") == "revolute"]) == 12
    root = ET.fromstring(xacro.process_file(XACRO, mappings={"enable_thermal": "false"}).toxml())
    assert {s.get("name") for s in root.iter("sensor")} == {"imu", "depth_camera", "inspection_camera"}
    root = ET.fromstring(xacro.process_file(
        XACRO, mappings={"mesh_uri_prefix": "file:///opt/spot/meshes"}).toxml())
    assert all(m.get("filename").startswith("file:///opt/spot/meshes/") for m in root.iter("mesh"))


# =============================================================================
# bridge
# =============================================================================
BRIDGE_TABLE = [
    ("/clock", "rosgraph_msgs/msg/Clock", "/clock", "gz.msgs.Clock", "GZ_TO_ROS"),
    ("/imu", "sensor_msgs/msg/Imu", "/imu", "gz.msgs.IMU", "GZ_TO_ROS"),
    ("/model/spot/odometry", "nav_msgs/msg/Odometry", "/model/spot/odometry", "gz.msgs.Odometry", "GZ_TO_ROS"),
    ("/joint_states", "sensor_msgs/msg/JointState", "/world/plant/model/spot/joint_state", "gz.msgs.Model",
     "GZ_TO_ROS"),
    ("/depth_camera/image", "sensor_msgs/msg/Image", "/depth_camera/image", "gz.msgs.Image", "GZ_TO_ROS"),
    ("/depth_camera/points", "sensor_msgs/msg/PointCloud2", "/depth_camera/image/points",
     "gz.msgs.PointCloudPacked", "GZ_TO_ROS"),
    ("/depth_camera/camera_info", "sensor_msgs/msg/CameraInfo", "/depth_camera/camera_info",
     "gz.msgs.CameraInfo", "GZ_TO_ROS"),
    ("/inspection_camera/image", "sensor_msgs/msg/Image", "/inspection_camera/image", "gz.msgs.Image",
     "GZ_TO_ROS"),
    ("/inspection_camera/camera_info", "sensor_msgs/msg/CameraInfo", "/inspection_camera/camera_info",
     "gz.msgs.CameraInfo", "GZ_TO_ROS"),
    ("/thermal_camera/image", "sensor_msgs/msg/Image", "/thermal_camera/image", "gz.msgs.Image", "GZ_TO_ROS"),
    ("/keyboard/keypress", "std_msgs/msg/Int32", "/keyboard/keypress", "gz.msgs.Int32", "GZ_TO_ROS"),
] + [(rm.joint_command_topic(j), "std_msgs/msg/Float64", rm.joint_command_topic(j), "gz.msgs.Double",
      "ROS_TO_GZ") for j in rm.JOINT_NAMES]

KNOWN_PAIRS = {
    ("rosgraph_msgs/msg/Clock", "gz.msgs.Clock"), ("sensor_msgs/msg/Imu", "gz.msgs.IMU"),
    ("nav_msgs/msg/Odometry", "gz.msgs.Odometry"), ("sensor_msgs/msg/JointState", "gz.msgs.Model"),
    ("sensor_msgs/msg/Image", "gz.msgs.Image"), ("sensor_msgs/msg/PointCloud2", "gz.msgs.PointCloudPacked"),
    ("sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"), ("std_msgs/msg/Float64", "gz.msgs.Double"),
    ("std_msgs/msg/Int32", "gz.msgs.Int32"),
}


def test_bridge_yaml():
    with open(BRIDGE, encoding="utf-8") as f:
        entries = yaml.safe_load(f)
    assert isinstance(entries, list)
    keys = {"ros_topic_name", "gz_topic_name", "ros_type_name", "gz_type_name", "direction"}
    rows = set()
    for e in entries:
        assert set(e) == keys, e
        assert e["direction"] in ("GZ_TO_ROS", "ROS_TO_GZ")
        assert (e["ros_type_name"], e["gz_type_name"]) in KNOWN_PAIRS, e
        rows.add((e["ros_topic_name"], e["ros_type_name"], e["gz_topic_name"], e["gz_type_name"],
                  e["direction"]))
    for row in BRIDGE_TABLE:
        assert row in rows, f"missing bridge entry {row}"
    ros_topics = [e["ros_topic_name"] for e in entries]
    assert len(ros_topics) == len(set(ros_topics))
    # the topics the URDF publishes / subscribes are the ones bridged
    root = ET.parse(URDF).getroot()
    gz_topics = {e["gz_topic_name"] for e in entries}
    for s in root.iter("sensor"):
        assert s.find("topic").text in gz_topics
        info = s.find("camera/camera_info_topic")
        if info is not None:
            assert info.text in gz_topics
    for p in root.iter("plugin"):
        t = p.find("topic")
        if t is not None:
            assert t.text in gz_topics


# =============================================================================
# meshes
# =============================================================================
@pytest.fixture(scope="module")
def objs():
    out = {}
    for f in sorted(os.listdir(MESH_DIR)):
        if f.endswith(".obj"):
            out[f[:-4]] = cs.read_obj(os.path.join(MESH_DIR, f))
    return out


def test_mesh_inventory_and_budget(objs):
    expected = {"base_link", "base_cam", "base_eap"} | {
        f"{leg}_{seg}" for leg in rm.LEG_NAMES for seg in ("hip", "upper_leg", "lower_leg")}
    assert set(objs) == expected
    faces = {k: len(v["faces"]) for k, v in objs.items()}
    total = sum(faces.values())
    assert 60000 <= total <= 90000, total
    base = faces["base_link"] + faces["base_cam"] + faces["base_eap"]
    assert 20000 <= base <= 30000, base
    for leg in rm.LEG_NAMES:
        for seg in ("hip", "upper_leg", "lower_leg"):
            assert 1500 <= faces[f"{leg}_{seg}"] <= 5000, (leg, seg, faces[f"{leg}_{seg}"])
    assert cs.dir_size(MODEL_DIR) < 15e6


def test_obj_integrity(objs):
    mtl = cs.read_mtl(os.path.join(MESH_DIR, cs.MTL_NAME))
    assert 1 <= len(mtl) <= 16
    for rgb in mtl.values():
        assert all(0.0 <= c <= 1.0 for c in rgb)
    for name, d in objs.items():
        V, F = d["vertices"], d["faces"]
        assert d["mtllib"] == cs.MTL_NAME
        assert F.min() >= 0 and F.max() < len(V), name
        assert set(d["face_material"]) <= set(mtl), name
        n, area = cs.face_normals(V, F)
        assert np.all(area > 0), f"{name}: degenerate faces"
        N = d["normals"]
        np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1.0, atol=2e-3)
        # corner normals agree with the winding (crease-aware smoothing)
        cn = N[d["face_normal_idx"]]                              # (M,3,3)
        agree = np.einsum("mkj,mj->mk", cn, n) > 0
        assert agree.mean() > 0.97, f"{name}: {agree.mean():.3f}"
    # body colours: Spot is yellow + dark grey/black + grey
    names = " ".join(mtl)
    assert "yellow" in names and ("black" in names or "dark_gray" in names)
    body_materials = [k for k in mtl if k.startswith("spot_")]
    assert len(body_materials) <= cs.N_BODY_COLOURS


def test_mesh_link_frames(objs):
    """Meshes are expressed in their link frames at zero joint angles."""
    for leg, sy in zip(rm.LEG_NAMES, rm.LEG_SIGN_Y):
        lo = objs[f"{leg}_lower_leg"]["vertices"]
        up = objs[f"{leg}_upper_leg"]["vertices"]
        hip = objs[f"{leg}_hip"]["vertices"]
        # shin hangs down from the knee (origin) to the foot ball near -LOWER_LEG_LENGTH
        assert lo[:, 2].min() == pytest.approx(-rm.LOWER_LEG_LENGTH, abs=0.015)
        assert 0.02 < lo[:, 2].max() < 0.07
        assert np.abs(lo[:, 1]).max() < 0.03
        # thigh: from above the hip_y axis down past the knee
        assert up[:, 2].min() == pytest.approx(-rm.UPPER_LEG_LENGTH, abs=0.045)
        assert up[:, 2].max() < 0.08
        # hip housing around the hip_x joint, bulging outwards (+y for left legs)
        assert np.abs(hip).max() < 0.07
        assert sy * hip[:, 1].mean() > 0
        # foot sphere visual/collision (r = FOOT_RADIUS) encloses the mesh foot
        foot = lo[lo[:, 2] < -rm.LOWER_LEG_LENGTH + 0.010]
        assert len(foot) > 20
        d = np.linalg.norm(foot - [0, 0, -rm.LOWER_LEG_LENGTH], axis=1)
        assert d.max() < rm.FOOT_RADIUS
    # left / right mirror symmetry (y -> -y)
    for a, b in (("fl", "fr"), ("rl", "rr")):
        for seg in ("hip", "upper_leg", "lower_leg"):
            A = objs[f"{a}_{seg}"]["vertices"]
            B = objs[f"{b}_{seg}"]["vertices"] * [1, -1, 1]
            np.testing.assert_allclose(A.min(0), B.min(0), atol=0.004)
            np.testing.assert_allclose(A.max(0), B.max(0), atol=0.004)
    # trunk: ~0.67 m long, nose forward, symmetric
    base = objs["base_link"]["vertices"]
    assert 0.6 < np.ptp(base[:, 0]) < 0.75
    assert abs(base[:, 1].max() + base[:, 1].min()) < 0.005
    cam = objs["base_cam"]["vertices"]
    eap = objs["base_eap"]["vertices"]
    assert cam[:, 0].mean() > 0.1 and eap[:, 0].mean() < -0.1    # camera front, payload rear
    # the inspection / thermal camera frames sit at the front of the camera payload
    for s in ("inspection_camera", "thermal_camera"):
        x, y, z = rm.SENSORS[s]["pose"][:3]
        assert cam[:, 0].max() - 0.03 < x < cam[:, 0].max() + 0.01
        assert cam[:, 2].min() < z < cam[:, 2].max()


def test_conversion_report():
    import json
    with open(REPORT, encoding="utf-8") as f:
        rep = json.load(f)
    ver = rep["verification"]
    assert ver["orig_to_dec_mean_mm"] < 0.5
    assert ver["orig_to_dec_p95_mm"] < 2.0
    assert ver["dec_to_orig_mean_mm"] < 0.5
    assert ver["link_frame_roundtrip_max_mm"] < 1e-6
    assert ver["joint_fit_max_mm"] < 5.0
    rest = rep["rest_angles_deg"]
    for leg in ("fl", "fr"):
        assert rest[leg][1] == pytest.approx(34.4, abs=0.5) and rest[leg][2] == pytest.approx(-88.9, abs=0.5)
    for leg in ("rl", "rr"):
        assert rest[leg][1] == pytest.approx(60.1, abs=0.5) and rest[leg][2] == pytest.approx(-94.3, abs=0.5)
    for leg, q in rep["rest_angles_rad"].items():
        for suffix, v in zip(rm.JOINT_SUFFIXES, q):
            lim = rm.JOINT_LIMITS[suffix]
            assert lim["lower"] <= v <= lim["upper"], (leg, suffix, v)
    assert rep["total_triangles"] == sum(m["faces"] for m in rep["meshes"].values())


def test_preview_image_exists():
    path = os.path.join(REPO, "docs", "images", "spot_model_preview.png")
    assert os.path.isfile(path) and os.path.getsize(path) > 50000


# =============================================================================
# converter algorithms (synthetic data)
# =============================================================================
def icosphere(subdiv=3):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
         (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(subdiv):
        cache, F2 = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    return np.array(V), np.array(F, dtype=np.int64)


def grid_plane(n=40):
    xs = np.linspace(-1, 1, n + 1)
    X, Y = np.meshgrid(xs, xs)
    V = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    F = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b, c, d = a + 1, a + n + 1, a + n + 2
            F += [(a, b, d), (a, d, c)]
    return V, np.array(F, dtype=np.int64)


def test_decimate_sphere_stays_closed_and_round():
    V, F = icosphere(4)                       # 5120 faces
    V2, F2, L2 = cs.decimate(V, F, np.zeros(len(F), np.int64), 600)
    assert 560 <= len(F2) <= 600
    r = np.linalg.norm(V2, axis=1)
    assert np.abs(r - 1).max() < 0.02          # vertices stay on the sphere
    topo = cs.edge_topology(F2, len(V2))
    assert np.all(topo.counts == 2)            # closed 2-manifold
    n, area = cs.face_normals(V2, F2)
    cen = V2[F2].mean(1)
    assert np.all(np.einsum("ij,ij->i", n, cen) > 0)   # orientation kept (outwards)
    assert len(F2) - len(topo.edges) + len(V2) == 2   # Euler characteristic of a sphere


def test_decimate_plane_keeps_boundary_and_colour_border():
    V, F = grid_plane(40)                      # 3200 faces, square [-1,1]^2
    cen = V[F].mean(1)
    labels = (cen[:, 0] > 0.0).astype(np.int64)
    V2, F2, L2 = cs.decimate(V, F, labels, 200)
    assert len(F2) <= 200
    assert np.abs(V2[:, 2]).max() < 1e-9      # still flat
    # outline preserved: corners kept, boundary vertices on the square border
    topo = cs.edge_topology(F2, len(V2))
    bv = np.unique(topo.edges[topo.counts == 1])
    on_border = np.isclose(np.abs(V2[bv, 0]), 1, atol=1e-6) | np.isclose(np.abs(V2[bv, 1]), 1, atol=1e-6)
    assert on_border.all()
    for corner in ([-1, -1], [1, -1], [-1, 1], [1, 1]):
        assert np.min(np.linalg.norm(V2[:, :2] - corner, axis=1)) < 1e-6
    _n, area = cs.face_normals(V2, F2)
    assert area.sum() == pytest.approx(4.0, rel=1e-6)
    # colour border stays at x = 0
    c2 = V2[F2].mean(1)
    assert np.all(c2[L2 == 0, 0] < 0.02) and np.all(c2[L2 == 1, 0] > -0.02)


def test_link_condition_closes_triangular_holes_only():
    # fan around vertex 0 with one missing triangle (0,1,2) -> boundary hole 0-1-2?
    # Build: square pyramid open at one side face
    V = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    F = np.array([[0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 4, 3], [1, 3, 2]])   # face (0,1,2) missing
    topo = cs.edge_topology(F, len(V))
    E, cnt = topo.edges, topo.counts
    opposite = F[topo.face0].sum(1) - E[:, 0] - E[:, 1]
    k = np.nonzero((E[:, 0] == 1) & (E[:, 1] == 2))[0]
    assert cnt[k[0]] == 1
    # boundary edge (1,2): extra common neighbour 0 closes the hole -> allowed
    assert cs._link_condition_ok(E, cnt, k, len(V), opposite)[0]
    assert not cs._link_condition_ok(E, cnt, k, len(V), None)[0]
    # closed tetrahedron: collapsing any edge would make it degenerate -> refused
    Vt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    Ft = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tt = cs.edge_topology(Ft, 4)
    opp = Ft[tt.face0].sum(1) - tt.edges[:, 0] - tt.edges[:, 1]
    opp2 = Ft[tt.face1].sum(1) - tt.edges[:, 0] - tt.edges[:, 1]
    assert not cs._link_condition_ok(tt.edges, tt.counts, np.arange(len(tt.edges)), 4, opp, opp2).any()
    # ... and decimation leaves it alone instead of folding it flat
    V2, F2, _l = cs.decimate(Vt, Ft, np.zeros(4, np.int64), 2, min_component_faces=1)
    assert len(F2) == 4


def test_corner_normals_crease():
    # unit cube: every corner normal equals its face normal (90 deg > crease)
    V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    N, idx = cs.corner_normals(V, F, 40.0)
    fn, _a = cs.face_normals(V, F)
    np.testing.assert_allclose(N[idx], np.repeat(fn[:, None, :], 3, 1), atol=1e-9)
    # sphere: smooth -> close to the radial direction
    Vs, Fs = icosphere(3)
    Ns, ids = cs.corner_normals(Vs, Fs, 40.0)
    radial = Vs[Fs]
    cosang = np.einsum("mkj,mkj->mk", Ns[ids], radial)
    assert cosang.min() > 0.995


def test_weighted_kmeans_and_merge():
    rng = np.random.default_rng(3)
    centres = np.array([[70, 15, 80], [25, 0, 0], [80, 0, 0], [3, 0, 0], [27, 0, 1], [55, -45, 15]], float)
    sizes = [3000, 4000, 800, 500, 400, 5]
    X = np.concatenate([c + rng.normal(0, 1.0, (n, 3)) for c, n in zip(centres, sizes)])
    w = np.ones(len(X))
    C, lab = cs.weighted_kmeans(X, w, 8, seed=0)
    merged = cs.merge_close_clusters(C, lab, w, 6, 10.0, min_area_frac=0.0025)
    groups = [set(np.unique(merged[s:s + n])) for s, n in zip(np.cumsum([0] + sizes[:-1]), sizes)]
    # near-identical greys (25 / 27) end up together, the 5-sample decal is folded in
    assert groups[1] == groups[4]
    assert len(set().union(*groups)) == 4
    assert len({min(g) for g in groups[:4]}) == 4


def test_rest_angles_and_ik_roundtrip():
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = {leg: (rng.uniform(-0.3, 0.3), rng.uniform(0.2, 1.4), rng.uniform(-2.2, -0.6)) for leg in cs.LEGS}
        frames = cs.all_link_frames(q)
        bones = {}
        for leg in cs.LEGS:
            names = cs.leg_bone_names(leg)
            bones[names["upper_leg"]] = frames[f"{leg}_upper_leg"][:3, 3]
            bones[names["lower_leg"]] = frames[f"{leg}_lower_leg"][:3, 3]
            bones[names["foot"]] = frames[f"{leg}_foot"][:3, 3]
        rec = cs.rest_angles_from_bones(bones)
        for i, leg in enumerate(cs.LEGS):
            np.testing.assert_allclose(rec[leg], q[leg], atol=1e-9)
            ik = cs.leg_ik_sagittal(i, cs.foot_position(i, q[leg]))
            np.testing.assert_allclose(cs.foot_position(i, ik), cs.foot_position(i, q[leg]), atol=1e-9)
            np.testing.assert_allclose(ik, q[leg], atol=1e-7)


def test_stand_pose_puts_feet_under_hips():
    q = cs.stand_pose(rm.NOMINAL_BODY_HEIGHT)
    feet = rm.nominal_foot_positions(rm.NOMINAL_BODY_HEIGHT)
    for i, leg in enumerate(cs.LEGS):
        np.testing.assert_allclose(cs.foot_position(i, q[leg]), feet[i], atol=1e-9)
        assert q[leg][2] < 0 and q[leg][1] > 0          # Spot style knee


def test_point_triangle_distance_against_sampling():
    rng = np.random.default_rng(11)
    A, B, C = rng.normal(size=(3, 300, 3))
    P = rng.normal(size=(300, 3)) * 2
    d = cs.point_triangle_distance(P, A, B, C)
    u, v = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 120))
    m = (u + v) <= 1
    u, v = u[m], v[m]
    for i in range(0, 300, 15):
        S = A[i] + np.outer(u, B[i] - A[i]) + np.outer(v, C[i] - A[i])
        brute = np.linalg.norm(S - P[i], axis=1).min()
        assert d[i] <= brute + 1e-9
        assert d[i] >= brute - 0.03 * np.linalg.norm(B[i] - A[i])


def test_surface_distance_and_knn():
    V, F = icosphere(3)
    q = np.array([[0, 0, 2.0], [0.5, 0, 0], [0, 0, 1.0]])
    d = cs.surface_distance(q, V, F)
    assert d[0] == pytest.approx(1.0, abs=0.01)
    assert d[1] == pytest.approx(0.5, abs=0.01)
    assert d[2] < 0.01
    idx, dist = cs.knn(V, V[:5], 1)
    np.testing.assert_array_equal(idx[:, 0], np.arange(5))
    np.testing.assert_allclose(dist[:, 0], 0, atol=1e-6)


def test_obj_roundtrip(tmp_path):
    V, F = icosphere(1)
    labels = (np.arange(len(F)) % 2).astype(np.int64)
    mats = [("red", (1, 0, 0)), ("blue", (0, 0, 1))]
    N, idx = cs.corner_normals(V, F)
    cs.write_mtl(str(tmp_path / cs.MTL_NAME), mats)
    cs.write_obj(str(tmp_path / "s.obj"), V, F, labels, mats, N, idx, header="test")
    d = cs.read_obj(str(tmp_path / "s.obj"))
    assert len(d["faces"]) == len(F)
    np.testing.assert_allclose(d["vertices"], V, atol=1e-5)
    # faces are regrouped by material, so compare as sets of sorted triangles
    got = {tuple(sorted(f)) for f in d["faces"].tolist()}
    assert got == {tuple(sorted(f)) for f in F.tolist()}
    assert cs.read_mtl(str(tmp_path / cs.MTL_NAME)) == {"red": (1.0, 0.0, 0.0), "blue": (0.0, 0.0, 1.0)}
    assert d["face_material"].count("red") == (labels == 0).sum()


def test_connected_components_and_cleanup():
    V, F = icosphere(1)
    F2 = np.concatenate([F, F + len(V)])
    V2 = np.concatenate([V, V + 5])
    comp = cs.connected_components(F2, len(V2))
    assert len(np.unique(comp)) == 2
    # duplicate + degenerate faces are removed, orientation is kept
    Fd = np.concatenate([F, F[:3], [[0, 0, 1]], F[:2][:, [1, 2, 0]]])
    Fc, Lc = cs.clean_faces(Fd, np.zeros(len(Fd), np.int64), V)
    assert len(Fc) == len(F)
