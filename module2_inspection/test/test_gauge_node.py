"""gauge_reader_node without ROS: the node module is imported against minimal in-memory stand-ins
of rclpy / std_msgs / sensor_msgs / nav_msgs (only when the real packages are not importable) and
driven with synthetic Gazebo-like messages.  Checks the parameter file against the declared
parameter types (ROS 2 rejects an int for a double parameter), the JSON contract of
/inspection/gauge_readings, the bgr8 overlay, the IMU motion gate, the "no pose" mode, the CPU
throttles (idle rate, lazy enhancement, full-search rate limit) and the shutdown path of main()."""
import importlib
import json
import math
import os
import sys
import types

import cv2
import numpy as np
import pytest
import yaml

from dusan_common.camera import camera_world_pose
from dusan_common.rclpy_logger_rules import CallSiteRules
from dusan_common.geometry import euler_to_quat
from dusan_common.layout import Layout
from dusan_common.robot_model import NOMINAL_BODY_HEIGHT
from module2_inspection.gauge_synth import GaugeSynth, SynthConditions

HERE = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(HERE, "..", "config", "gauge_reader.yaml")
LAYOUT = Layout.load()


# ------------------------------------------------------------------------------ ROS stand-ins
class _NS:
    """Attribute bag used for message sub-structures."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stamp(t=0.0):
    sec = int(math.floor(t))
    return _NS(sec=sec, nanosec=int(round((t - sec) * 1e9)))


class _Header:
    def __init__(self):
        self.stamp = _stamp()
        self.frame_id = ""


class String:
    def __init__(self, data=""):
        self.data = data


class Image:
    def __init__(self):
        self.header = _Header()
        self.height = self.width = self.step = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.data = b""


class CameraInfo:
    def __init__(self):
        self.header = _Header()
        self.width = self.height = 0
        self.k = [0.0] * 9


class Imu:
    def __init__(self):
        self.header = _Header()
        self.angular_velocity = _NS(x=0.0, y=0.0, z=0.0)


class Odometry:
    def __init__(self):
        self.header = _Header()
        self.pose = _NS(pose=_NS(position=_NS(x=0.0, y=0.0, z=0.0), orientation=_NS(x=0.0, y=0.0, z=0.0, w=1.0)))


class _Param:
    def __init__(self, value):
        self.value = value


class _Logger:
    def __init__(self):
        self.lines = []
        self.rules = CallSiteRules(skip_files=(__file__,))   # rclpy: one severity / filter set per call site

    def _log(self, level, msg, **kw):
        self.rules.check(level, **kw)
        self.lines.append((level, str(msg)))

    def info(self, msg, **kw):
        self._log("info", msg, **kw)

    def warning(self, msg, **kw):
        self._log("warning", msg, **kw)

    def error(self, msg, **kw):
        self._log("error", msg, **kw)

    def debug(self, msg, **kw):
        self._log("debug", msg, **kw)


class _Publisher:
    def __init__(self, topic):
        self.topic = topic
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _Clock:
    t = 0.0

    def now(self):
        return _NS(nanoseconds=int(self.t * 1e9))


class Node:
    """Just enough of rclpy.node.Node for the gauge_reader node (strict parameter typing)."""

    overrides: dict = {}

    def __init__(self, name):
        self.name = name
        self.params = {}
        self.subs = []
        self.pubs = {}
        self.timers = []
        self.logger = _Logger()
        self.clock = _Clock()

    def declare_parameter(self, name, default):
        value = default
        if name in Node.overrides:
            value = Node.overrides[name]
            if type(value) is not type(default):
                raise TypeError(f"parameter {name}: {type(value).__name__} given, "
                                f"{type(default).__name__} declared")
        self.params[name] = value
        return _Param(value)

    def get_parameter(self, name):
        return _Param(self.params[name])

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs.append((topic, cb, qos))

    def create_publisher(self, msg_type, topic, qos):
        self.pubs[topic] = _Publisher(topic)
        return self.pubs[topic]

    def create_timer(self, period, cb):
        self.timers.append((period, cb))

    def get_logger(self):
        return self.logger

    def get_clock(self):
        return self.clock

    def destroy_node(self):
        pass


def _ros_stubs():
    qos = types.ModuleType("rclpy.qos")

    class QoSProfile:
        def __init__(self, depth=10, **kw):
            self.depth = depth
            self.__dict__.update(kw)

    qos.QoSProfile = QoSProfile
    qos.DurabilityPolicy = _NS(TRANSIENT_LOCAL="transient_local", VOLATILE="volatile")
    qos.ReliabilityPolicy = _NS(RELIABLE="reliable", BEST_EFFORT="best_effort")
    qos.HistoryPolicy = _NS(KEEP_LAST="keep_last")
    qos.qos_profile_sensor_data = QoSProfile(depth=5, reliability="best_effort")
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = Node
    rclpy = types.ModuleType("rclpy")
    rclpy.node, rclpy.qos = node_mod, qos
    rclpy.init = lambda args=None: None
    rclpy.ok = lambda: False
    rclpy.shutdown = lambda: None
    rclpy.spin = lambda n: None
    executors = types.ModuleType("rclpy.executors")

    class ExternalShutdownException(Exception):
        pass

    executors.ExternalShutdownException = ExternalShutdownException
    rclpy.executors = executors
    mods = {"rclpy": rclpy, "rclpy.node": node_mod, "rclpy.qos": qos, "rclpy.executors": executors}
    for pkg, names in (("std_msgs", {"String": String}),
                       ("sensor_msgs", {"Image": Image, "CameraInfo": CameraInfo, "Imu": Imu}),
                       ("nav_msgs", {"Odometry": Odometry})):
        m = types.ModuleType(pkg)
        mm = types.ModuleType(pkg + ".msg")
        for k, v in names.items():
            setattr(mm, k, v)
        m.msg = mm
        mods[pkg] = m
        mods[pkg + ".msg"] = mm
    return mods


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


@pytest.fixture(scope="module")
def node_mod():
    try:
        import rclpy  # noqa: F401  (real ROS available: this test is for ROS-less machines)
        pytest.skip("real rclpy available - the stubbed node test only runs without ROS")
    except ImportError:
        pass
    saved = {k: sys.modules.get(k) for k in _ros_stubs()}
    sys.modules.update(_ros_stubs())
    sys.modules.pop("module2_inspection.gauge_reader_node", None)
    try:
        yield importlib.import_module("module2_inspection.gauge_reader_node")
    finally:
        sys.modules.pop("module2_inspection.gauge_reader_node", None)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.fixture(scope="module")
def node(node_mod):
    with open(YAML_PATH, encoding="utf-8") as f:
        params = yaml.safe_load(f)["gauge_reader"]["ros__parameters"]
    Node.overrides = _flatten(params)
    Node.overrides["stats_period_s"] = 0.0          # no stats timer in the test
    cv_threads = cv2.getNumThreads()                # the node limits OpenCV's pool (process wide)
    try:
        n = node_mod.GaugeReaderNode()
    finally:
        Node.overrides = {}
    n.test_cv_threads = cv2.getNumThreads()
    yield n
    cv2.setNumThreads(cv_threads)


# ------------------------------------------------------------------------------ message helpers
def _standoff_pose(g, dist=1.8, lateral=0.15):
    n = np.asarray(g["normal"], float)
    p = np.asarray(g["position"], float)
    base = p + dist * n + lateral * np.array([-n[1], n[0], 0.0])
    yaw = math.atan2(-n[1], -n[0])
    z = LAYOUT.floor_z(base[0], base[1]) + NOMINAL_BODY_HEIGHT
    return (float(base[0]), float(base[1]), float(z), 0.0, 0.0, float(yaw))


def _image_msg(bgr, t):
    m = Image()
    m.header.stamp = _stamp(t)
    m.header.frame_id = "inspection_camera_link"
    rgb = np.ascontiguousarray(bgr[..., ::-1])          # Gazebo publishes rgb8
    m.height, m.width = rgb.shape[:2]
    m.encoding = "rgb8"
    m.step = m.width * 3
    m.data = rgb.tobytes()
    return m


def _odom_msg(pose, t):
    m = Odometry()
    m.header.stamp = _stamp(t)
    x, y, z, r, p, yw = pose
    m.pose.pose.position = _NS(x=x, y=y, z=z)
    qx, qy, qz, qw = euler_to_quat(r, p, yw)
    m.pose.pose.orientation = _NS(x=qx, y=qy, z=qz, w=qw)
    return m


def _imu_msg(w, t):
    m = Imu()
    m.header.stamp = _stamp(t)
    m.angular_velocity = _NS(x=0.0, y=float(w), z=0.0)
    return m


def _callbacks(node):
    cbs = {}
    for topic, cb, _qos in node.subs:
        cbs.setdefault(topic, []).append(cb)
    return cbs


def _feed(node, img_msg=None, odom=None, imu=None, target=None):
    cbs = _callbacks(node)
    if target is not None:
        for cb in cbs["/inspection/target"]:
            cb(String(target))
    if odom is not None:
        cbs["/odom"][0](odom)
    if imu is not None:
        cbs["/imu"][0](imu)
    if img_msg is not None:
        cbs["/inspection_camera/image"][0](img_msg)


# ------------------------------------------------------------------------------ tests
def test_parameters_and_interfaces(node):
    topics = {t for t, _cb, _q in node.subs}
    assert {"/inspection_camera/image", "/inspection_camera/camera_info", "/odom", "/imu",
            "/inspection/target"} <= topics
    assert sum(t == "/inspection/target" for t, _cb, _q in node.subs) == 2      # latched + volatile
    assert {"/inspection/gauge_readings", "/inspection/gauge_overlay"} <= set(node.pubs)
    assert any(abs(period - 0.2) < 1e-9 for period, _cb in node.timers)        # 5 Hz processing
    assert node.reader.cfg.min_calibration_ncc == pytest.approx(0.5)            # nested 'reader.' params
    with open(YAML_PATH, encoding="utf-8") as f:
        params = _flatten(yaml.safe_load(f)["gauge_reader"]["ros__parameters"])
    assert set(params) <= set(node.params), set(params) - set(node.params)     # no dead YAML keys
    assert node.test_cv_threads == params["opencv_threads"] == 2


def test_pose_frames_publish_contract_and_fuse(node):
    synth = GaugeSynth(seed=3)
    g = LAYOUT.gauge("G05")
    pose = _standoff_pose(g)
    T = camera_world_pose(pose, "inspection_camera")
    pub = node.pubs["/inspection/gauge_readings"]
    ovl = node.pubs["/inspection/gauge_overlay"]
    n0 = len(pub.msgs)
    rng = np.random.default_rng(11)
    for k in range(3):
        t = 100.0 + 0.2 * k
        node.clock.t = t
        img, gt = synth.render_world(g, T, SynthConditions(noise_sigma=1.5, plate=True), rng=rng)
        _feed(node, _image_msg(img, t), _odom_msg(pose, t - 0.01), _imu_msg(0.05, t - 0.02), target="G05")
        node._process()
    assert len(pub.msgs) == n0 + 3
    d = json.loads(pub.msgs[-1].data)
    assert set(d) == {"stamp", "readings"} and d["stamp"] == pytest.approx(100.4)
    r = next(x for x in d["readings"] if x["id"] == "G05")
    for key in ("id", "tag", "value", "unit", "confidence", "angle_deg", "range", "status", "n_samples",
                "robot_pose"):
        assert key in r
    fs = g["max"] - g["min"]
    assert abs(r["value"] - g["value"]) / fs < 0.02
    assert r["range"] == [g["min"], g["max"]] and r["unit"] == g["unit"] and r["tag"] == g["tag"]
    assert r["status"] == LAYOUT.gauge_status("G05", r["value"])
    assert r["n_samples"] == 3 and 0.0 < r["confidence"] <= 1.0
    assert r["robot_pose"] == pytest.approx([pose[0], pose[1], pose[5]])
    o = ovl.msgs[-1]
    assert o.encoding == "bgr8" and (o.height, o.width) == (600, 800) and len(o.data) == 600 * 800 * 3


def test_imu_gate_skips_shaking_frames(node):
    synth = GaugeSynth(seed=4)
    g = LAYOUT.gauge("G05")
    pose = _standoff_pose(g)
    img, _ = synth.render_world(g, camera_world_pose(pose, "inspection_camera"), SynthConditions())
    pub = node.pubs["/inspection/gauge_readings"]
    n0, skipped0 = len(pub.msgs), node._stats["skipped_motion"]
    t = 200.0
    node.clock.t = t
    _feed(node, _image_msg(img, t), _odom_msg(pose, t), _imu_msg(2.0, t - 0.01))
    node._process()
    assert node._stats["skipped_motion"] == skipped0 + 1 and len(pub.msgs) == n0
    node._process()                                     # no new image -> nothing happens
    assert len(pub.msgs) == n0


def test_no_pose_uses_target_id(node):
    synth = GaugeSynth(seed=5)
    g = LAYOUT.gauge("G01")
    pose = _standoff_pose(g)
    img, _ = synth.render_world(g, camera_world_pose(pose, "inspection_camera"), SynthConditions(noise_sigma=1.0))
    pub = node.pubs["/inspection/gauge_readings"]
    t = 400.0                                           # odometry history is > odom_max_age_s old
    node.clock.t = t
    _feed(node, _image_msg(img, t), imu=_imu_msg(0.0, t), target="G01")
    node._process()
    d = json.loads(pub.msgs[-1].data)
    r = next(x for x in d["readings"] if x["id"] == "G01")
    assert r["robot_pose"] is None
    assert abs(r["value"] - g["value"]) / (g["max"] - g["min"]) < 0.02


def _facing_away(g):
    x, y, z, r, p, yw = _standoff_pose(g)
    return (x, y, z, r, p, yw + math.pi)


def test_idle_frames_skip_enhancement_and_throttle_overlay(node, monkeypatch):
    """No target and no gauge predicted in view: <= idle_process_rate_hz, no enhancement unless a
    search or an overlay is due, overlay at most every overlay_idle_period_s."""
    monkeypatch.setitem(node.p, "overlay_idle_period_s", 1.0)
    calls = []
    enhance = node.reader.enhance
    monkeypatch.setattr(node.reader, "enhance", lambda bgr: calls.append(1) or enhance(bgr))
    pose = _facing_away(LAYOUT.gauge("G05"))
    assert node.assoc.hints(pose, target_id=None, pose_sigma_px=30.0) == []
    img = np.full((600, 800, 3), 60, np.uint8)
    ovl = node.pubs["/inspection/gauge_overlay"]
    st = node._stats

    def frame(t):
        node.clock.t = t
        _feed(node, _image_msg(img, t), _odom_msg(pose, t), _imu_msg(0.0, t), target="")
        node._process()
        return st["processed"], st["full_searches"], len(calls), len(ovl.msgs)

    p0, f0, _e0, o0 = frame(1000.0)               # periodic full search due -> enhance + overlay
    assert node._target == ""
    p1, f1, e1, o1 = frame(1000.2)                # idle rate 2 Hz: not processed yet
    assert (p1, f1, o1) == (p0, f0, o0) and node._img_new
    p2, f2, e2, o2 = frame(1000.6)                # processed, nothing searched, overlay not due
    assert p2 == p1 + 1 and f2 == f1 and e2 == e1 and o2 == o1
    p3, f3, e3, o3 = frame(1001.2)                # overlay due -> one enhancement for it
    assert p3 == p2 + 1 and f3 == f2 and e3 == e2 + 1 and o3 == o2 + 1
    p4, f4, e4, o4 = frame(1002.4)                # full_search_period_s (2 s) elapsed
    assert p4 == p3 + 1 and f4 == f3 + 1 and o4 == o3 + 1


def test_full_search_rate_limited_without_pose(node):
    """No pose (or target not found): whole-frame search on every frame, but not more often
    than full_search_min_period_s (0.5 s)."""
    assert node.p["full_search_min_period_s"] == pytest.approx(0.5)
    img = np.full((600, 800, 3), 60, np.uint8)
    st = node._stats
    p0, f0 = st["processed"], st["full_searches"]
    for k in range(4):                            # 5 Hz, odometry history far in the past
        t = 2000.0 + 0.2 * k
        node.clock.t = t
        _feed(node, _image_msg(img, t), imu=_imu_msg(0.0, t), target="G01")
        node._process()
    assert st["processed"] == p0 + 4               # a target is set: full processing rate
    assert st["full_searches"] == f0 + 2           # t = 2000.0 and 2000.6


def test_main_catches_external_shutdown(node_mod, monkeypatch):
    """Ctrl-C from ros2 launch makes spin() raise ExternalShutdownException (Jazzy and later):
    main() returns normally and destroys the node; a failing constructor is not masked."""
    assert node_mod.ExternalShutdownException is sys.modules["rclpy.executors"].ExternalShutdownException
    monkeypatch.setitem(node_mod.NODE_DEFAULTS, "opencv_threads", 0)      # keep OpenCV's pool as is
    monkeypatch.setitem(node_mod.NODE_DEFAULTS, "stats_period_s", 0.0)
    destroyed = []
    monkeypatch.setattr(Node, "destroy_node", lambda self: destroyed.append(self.name))
    for exc in (node_mod.ExternalShutdownException, KeyboardInterrupt):
        def spin(_n, exc=exc):
            raise exc()
        monkeypatch.setattr(node_mod.rclpy, "spin", spin)
        node_mod.main()
    assert destroyed == ["gauge_reader", "gauge_reader"]

    def broken():
        raise RuntimeError("constructor failed")
    monkeypatch.setattr(node_mod, "GaugeReaderNode", broken)
    with pytest.raises(RuntimeError, match="constructor failed"):
        node_mod.main()
    assert destroyed == ["gauge_reader", "gauge_reader"]


def _snapshot_msgs(node, n0):
    return [json.loads(m.data) for m in node.pubs["/inspection/gauge_snapshot"].msgs[n0:]]


def test_snapshot_published_once_when_the_target_leaves_the_gauge(node):
    """The mission clears /inspection/target when it accepts a gauge: one evidence snapshot of that
    visit (final fused value, annotated JPEGs) - not before, and not again for the duplicate
    delivery of the latched + volatile subscriptions."""
    import base64
    assert "/inspection/gauge_snapshot" in node.pubs
    _feed(node, target="")                              # close visits left open by earlier tests
    pub = node.pubs["/inspection/gauge_snapshot"]
    n0 = len(pub.msgs)
    synth = GaugeSynth(seed=21)
    g = LAYOUT.gauge("G05")
    pose = _standoff_pose(g)
    T = camera_world_pose(pose, "inspection_camera")
    rng = np.random.default_rng(5)
    readings = node.pubs["/inspection/gauge_readings"]
    for k in range(4):
        t = 3000.0 + 0.2 * k
        node.clock.t = t
        img, _ = synth.render_world(g, T, SynthConditions(noise_sigma=1.5, plate=True), rng=rng)
        _feed(node, _image_msg(img, t), _odom_msg(pose, t - 0.01), _imu_msg(0.0, t), target="G05")
        node._process()
    assert len(pub.msgs) == n0                          # still inspecting: nothing yet
    last = next(r for r in json.loads(readings.msgs[-1].data)["readings"] if r["id"] == "G05")
    _feed(node, target="")                              # both subscriptions deliver ""
    snaps = _snapshot_msgs(node, n0)
    assert len(snaps) == 1
    d = snaps[0]
    for key in ("stamp", "t_end", "wall", "id", "tag", "value", "unit", "status", "confidence", "n_samples", "range",
                "read", "frames", "readings", "frame_value", "frame_confidence", "robot_pose", "format", "width",
                "height", "jpeg_b64", "full_jpeg_b64"):
        assert key in d, key
    assert d["id"] == "G05" and d["read"] is True and d["format"] == "jpeg"
    assert d["value"] == pytest.approx(last["value"]) and d["status"] == last["status"]   # = what the mission accepts
    assert d["frames"] == 4 and d["readings"] == 4 and 3000.0 <= d["stamp"] <= 3000.6
    assert abs(d["frame_value"] - d["value"]) <= 0.01 * (g["max"] - g["min"])    # the drawn needle agrees
    assert d["range"] == [g["min"], g["max"]] and d["unit"] == g["unit"] and d["tag"] == g["tag"]
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(d["jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR)
    full = cv2.imdecode(np.frombuffer(base64.b64decode(d["full_jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[:2] == (d["height"], d["width"]) and d["width"] == node.p["snapshot_crop_px"]
    assert full.shape[1] == node.p["snapshot_full_width"]
    assert any("inspection snapshot published" in ln for _lv, ln in node.logger.lines)


def test_snapshot_of_a_missed_gauge_and_of_a_direct_target_switch(node):
    """No detection during the visit: the latest frame, read false, value null.  A switch from one
    target to the next ends the first visit too."""
    import base64
    _feed(node, target="")
    n0 = len(node.pubs["/inspection/gauge_snapshot"].msgs)
    img = np.full((600, 800, 3), 60, np.uint8)
    pose = _facing_away(LAYOUT.gauge("G01"))
    for k in range(2):
        t = 4000.0 + 0.2 * k
        node.clock.t = t
        _feed(node, _image_msg(img, t), _odom_msg(pose, t), _imu_msg(0.0, t), target="G01")
        node._process()
    _feed(node, target="G02")
    snaps = _snapshot_msgs(node, n0)
    assert [s["id"] for s in snaps] == ["G01"]
    d = snaps[0]
    assert d["read"] is False and d["value"] is None and d["readings"] == 0 and d["frames"] == 2
    assert d["stamp"] == pytest.approx(4000.2) and base64.b64decode(d["jpeg_b64"])[:3] == b"\xff\xd8\xff"
    _feed(node, target="")                              # G02 never got a frame: no snapshot
    assert len(_snapshot_msgs(node, n0)) == 1


def test_snapshot_failure_is_logged_not_raised(node, monkeypatch):
    mod = sys.modules["module2_inspection.gauge_reader_node"]
    _feed(node, target="")
    n0 = len(node.pubs["/inspection/gauge_snapshot"].msgs)
    t = 5000.0
    node.clock.t = t
    _feed(node, _image_msg(np.full((600, 800, 3), 60, np.uint8), t), imu=_imu_msg(0.0, t), target="G03")
    node._process()

    def boom(*a, **k):
        raise RuntimeError("encoder broken")
    monkeypatch.setattr(mod, "render_gauge_snapshot", boom)
    _feed(node, target="")
    assert len(node.pubs["/inspection/gauge_snapshot"].msgs) == n0
    assert any(lv == "warning" and "snapshot failed" in ln for lv, ln in node.logger.lines)
