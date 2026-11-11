"""Smoke tests of the five Module-3 ROS nodes against strict stand-ins of
rclpy and the message packages (ROS is not installed on the development PC).

The stand-in messages accept only the fields of the real ROS definitions, so
a typo in a field name or a wrong message type fails here instead of on the
robot PC.  Every node is constructed with its default parameters (from the
package YAML), fed representative messages and its timers are ticked.
The stand-ins live in sys.modules only for the duration of each test.
"""
import importlib
import json
import math
import sys
import time
import types

import numpy as np
import pytest
import yaml
import os

from dusan_common.rclpy_logger_rules import CallSiteRules


# --------------------------------------------------------------- messages
class Msg:
    _fields = {}

    def __init__(self, **kw):
        for k, f in self._fields.items():
            object.__setattr__(self, k, f() if callable(f) else f)
        for k, v in kw.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        if k not in self._fields:
            raise AttributeError(f"{type(self).__name__} has no field '{k}'")
        object.__setattr__(self, k, v)


def _msg(name, consts=None, **fields):
    cls = type(name, (Msg,), {"_fields": fields})
    for k, v in (consts or {}).items():
        setattr(cls, k, v)
    return cls


Time = _msg("Time", sec=0, nanosec=0)
Header = _msg("Header", stamp=Time, frame_id="")
Vector3 = _msg("Vector3", x=0.0, y=0.0, z=0.0)
Point = _msg("Point", x=0.0, y=0.0, z=0.0)
Quaternion = _msg("Quaternion", x=0.0, y=0.0, z=0.0, w=1.0)
Pose = _msg("Pose", position=Point, orientation=Quaternion)
PoseWithCovariance = _msg("PoseWithCovariance", pose=Pose, covariance=list)
Twist = _msg("Twist", linear=Vector3, angular=Vector3)
TwistWithCovariance = _msg("TwistWithCovariance", twist=Twist, covariance=list)
Odometry = _msg("Odometry", header=Header, child_frame_id="", pose=PoseWithCovariance, twist=TwistWithCovariance)
MapMetaData = _msg("MapMetaData", map_load_time=Time, resolution=0.0, width=0, height=0, origin=Pose)
OccupancyGrid = _msg("OccupancyGrid", header=Header, info=MapMetaData, data=list)
PoseStamped = _msg("PoseStamped", header=Header, pose=Pose)
Path = _msg("Path", header=Header, poses=list)
PointStamped = _msg("PointStamped", header=Header, point=Point)
Vector3Stamped = _msg("Vector3Stamped", header=Header, vector=Vector3)
String = _msg("String", data="")
Bool = _msg("Bool", data=False)
Float32 = _msg("Float32", data=0.0)
MultiArrayDimension = _msg("MultiArrayDimension", label="", size=0, stride=0)
MultiArrayLayout = _msg("MultiArrayLayout", dim=list, data_offset=0)
Float32MultiArray = _msg("Float32MultiArray", layout=MultiArrayLayout, data=list)
BatteryState = _msg(
    "BatteryState",
    consts={"POWER_SUPPLY_STATUS_UNKNOWN": 0, "POWER_SUPPLY_STATUS_CHARGING": 1,
            "POWER_SUPPLY_STATUS_DISCHARGING": 2, "POWER_SUPPLY_STATUS_NOT_CHARGING": 3,
            "POWER_SUPPLY_STATUS_FULL": 4, "POWER_SUPPLY_HEALTH_UNKNOWN": 0, "POWER_SUPPLY_HEALTH_GOOD": 1,
            "POWER_SUPPLY_TECHNOLOGY_UNKNOWN": 0, "POWER_SUPPLY_TECHNOLOGY_LION": 2},
    header=Header, voltage=0.0, temperature=0.0, current=0.0, charge=0.0, capacity=0.0, design_capacity=0.0,
    percentage=0.0, power_supply_status=0, power_supply_health=0, power_supply_technology=0, present=True,
    cell_voltage=list, cell_temperature=list, location="", serial_number="")
PointField = type("PointField", (Msg,), {"_fields": {"name": "", "offset": 0, "datatype": 0, "count": 0},
                                         "INT8": 1, "UINT8": 2, "INT16": 3, "UINT16": 4, "INT32": 5, "UINT32": 6,
                                         "FLOAT32": 7, "FLOAT64": 8})
PointCloud2 = _msg("PointCloud2", header=Header, height=0, width=0, fields=list, is_bigendian=False, point_step=0,
                   row_step=0, data=bytes, is_dense=False)


# ------------------------------------------------------------------ rclpy
class World:
    def __init__(self):
        self.t = 100.0
        self.params = {}
        self.pubs, self.subs, self.timers = [], [], []

    def published(self, topic):
        return [m for p in self.pubs if p.topic == topic for m in p.msgs]

    def deliver(self, topic, m):
        subs = [s for s in self.subs if s.topic == topic]
        assert subs, f"nobody subscribes {topic}"
        for s in subs:
            assert isinstance(m, s.type), f"{topic}: {type(m).__name__} vs {s.type.__name__}"
            s.cb(m)

    def tick(self, n=1, dt=0.1):
        for _ in range(n):
            self.t += dt
            for tm in list(self.timers):
                tm.elapsed += dt
                if tm.elapsed >= tm.period - 1e-9:
                    tm.elapsed = 0.0
                    tm.cb()


class _Pub:
    def __init__(self, mtype, topic):
        self.type, self.topic, self.msgs = mtype, topic, []

    def publish(self, m):
        assert isinstance(m, self.type), f"{self.topic}: published {type(m).__name__}, expected {self.type.__name__}"
        self.msgs.append(m)


class _Sub:
    def __init__(self, mtype, topic, cb):
        self.type, self.topic, self.cb = mtype, topic, cb


class _Timer:
    def __init__(self, period, cb):
        self.period, self.cb, self.elapsed = period, cb, 0.0


class _Logger:
    """Strict: only the RcutilsLogger methods of every distro (no deprecated
    ``warn`` alias, which newer rclpy releases removed)."""

    def __init__(self):
        self.lines = []
        self.rules = CallSiteRules(skip_files=(__file__,))   # rclpy: one severity per call site

    def debug(self, s):
        self.rules.check("D")
        self.lines.append(("D", s))

    def info(self, s):
        self.rules.check("I")
        self.lines.append(("I", s))

    def warning(self, s):
        self.rules.check("W")
        self.lines.append(("W", s))

    def error(self, s):
        self.rules.check("E")
        self.lines.append(("E", s))


class _Now:
    def __init__(self, t):
        self.nanoseconds = int(t * 1e9)

    def to_msg(self):
        return Time(sec=int(self.nanoseconds // 10 ** 9), nanosec=int(self.nanoseconds % 10 ** 9))


def _make_rclpy(world):
    class Node:
        def __init__(self, name):
            self._name = name
            self._params = {}
            self._logger = _Logger()

        def get_name(self):
            return self._name

        def get_logger(self):
            return self._logger

        def declare_parameter(self, name, default):
            assert name not in self._params, f"parameter {name} declared twice"
            assert default is not None and not (isinstance(default, list) and not default), name
            v = world.params.get(self._name, {}).get(name, default)
            if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
                raise TypeError(f"{name}: int for a double parameter (rclpy would reject it)")
            self._params[name] = v

        def get_parameter(self, name):
            return types.SimpleNamespace(value=self._params[name])

        def create_publisher(self, mtype, topic, qos):
            p = _Pub(mtype, topic)
            world.pubs.append(p)
            return p

        def create_subscription(self, mtype, topic, cb, qos):
            s = _Sub(mtype, topic, cb)
            world.subs.append(s)
            return s

        def create_timer(self, period, cb):
            assert period > 0
            tm = _Timer(period, cb)
            world.timers.append(tm)
            return tm

        def get_clock(self):
            return types.SimpleNamespace(now=lambda: _Now(world.t))

        def destroy_node(self):
            pass

    mods = {}
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda args=None: None
    rclpy.ok = lambda: True
    rclpy.shutdown = lambda: None
    rclpy.spin = lambda node: None
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = Node
    qos = types.ModuleType("rclpy.qos")

    class QoSProfile:
        def __init__(self, **kw):
            self.kw = kw

    qos.QoSProfile = QoSProfile
    qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL=1, VOLATILE=2)
    qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE=1, BEST_EFFORT=2)
    qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
    qos.qos_profile_sensor_data = QoSProfile()
    rclpy.node, rclpy.qos = node_mod, qos
    mods.update({"rclpy": rclpy, "rclpy.node": node_mod, "rclpy.qos": qos})

    def pkg(name, **classes):
        root = types.ModuleType(name)
        m = types.ModuleType(name + ".msg")
        for k, v in classes.items():
            setattr(m, k, v)
        root.msg = m
        mods[name] = root
        mods[name + ".msg"] = m

    pkg("std_msgs", String=String, Bool=Bool, Float32=Float32, Float32MultiArray=Float32MultiArray,
        MultiArrayDimension=MultiArrayDimension, MultiArrayLayout=MultiArrayLayout, Header=Header)
    pkg("geometry_msgs", Twist=Twist, Vector3=Vector3, Vector3Stamped=Vector3Stamped, Point=Point,
        PointStamped=PointStamped, Pose=Pose, PoseStamped=PoseStamped, Quaternion=Quaternion)
    pkg("nav_msgs", Odometry=Odometry, OccupancyGrid=OccupancyGrid, Path=Path, MapMetaData=MapMetaData)
    pkg("sensor_msgs", BatteryState=BatteryState, PointCloud2=PointCloud2, PointField=PointField)
    return mods


NODE_MODULES = ["gas_dispersion_node", "source_seeker_node", "battery_node", "comm_node", "return_to_home_node",
                "node_common"]
_CFG = os.path.join(os.path.dirname(__file__), "..", "config")


@pytest.fixture
def ros(monkeypatch):
    world = World()
    for name, mod in _make_rclpy(world).items():
        monkeypatch.setitem(sys.modules, name, mod)
    for m in NODE_MODULES:
        sys.modules.pop("module3_gas_safety." + m, None)
    # parameters from the package YAML files (what the launch file loads)
    for f in ("gas_dispersion.yaml", "seeker.yaml", "safety.yaml"):
        with open(os.path.join(_CFG, f), encoding="utf-8") as fh:
            for node, sec in yaml.safe_load(fh).items():
                world.params[node] = dict(sec["ros__parameters"])
    yield world
    for m in NODE_MODULES:
        sys.modules.pop("module3_gas_safety." + m, None)


def _node(name):
    return importlib.import_module("module3_gas_safety." + name)


def _odom(x, y, yaw, vx=0.0, wz=0.0):
    m = Odometry()
    m.pose.pose.position.x, m.pose.pose.position.y = x, y
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    m.twist.twist.linear.x = vx
    m.twist.twist.angular.z = wz
    return m


# ------------------------------------------------------------------ tests
def test_gas_dispersion_node(ros):
    ros.params["gas_dispersion_sim"].update(spinup_seconds=120.0, spinup_gust_seconds=5.0)
    node = _node("gas_dispersion_node").GasDispersionNode()
    grids = ros.published("/gas/concentration_grid")
    assert grids, "latched grid published at start-up"
    g = grids[-1]
    assert len(g.data) == g.info.width * g.info.height
    assert min(g.data) >= -1 and max(g.data) <= 100 and max(g.data) > 50
    ros.tick(5)
    assert not ros.published("/gas/sensor")              # no pose yet
    ros.deliver("/odom", _odom(0.0, 2.0, math.pi))
    ros.tick(20)
    s = ros.published("/gas/sensor")
    a = ros.published("/gas/sensor_array")
    w = ros.published("/gas/wind")
    assert len(s) >= 15 and all(m.data >= 0.0 for m in s) and s[-1].data > 20.0
    assert len(a[-1].data) == 4 and a[-1].layout.dim[0].size == 4
    assert w[-1].header.frame_id == "map" and math.isfinite(w[-1].vector.x)
    info = json.loads(ros.published("/gas/field_info")[-1].data)
    for k in ("max_ppm", "vis_max_ppm", "scale", "source", "wind", "t"):
        assert k in info
    assert info["scale"] == "log" and info["source"] == [-3.2, 2.0]
    assert node.model.t > 1.0


def test_source_seeker_node(ros):
    node = _node("source_seeker_node").SourceSeekerNode()
    ros.tick(5)
    assert not ros.published("/cmd_vel/seek")
    ros.deliver("/odom", _odom(5.0, 1.0, math.pi))
    ros.deliver("/gas/sensor", Float32(data=30.0))
    arr = Float32MultiArray()
    arr.data = [32.0, 31.0, 28.0, 27.0]
    ros.deliver("/gas/sensor_array", arr)
    wind = Vector3Stamped()
    wind.vector.x, wind.vector.y = 0.4, -0.1
    ros.deliver("/gas/wind", wind)
    ros.deliver("/gas/seek/enable", Bool(data=True))
    for _ in range(20):
        ros.deliver("/gas/sensor", Float32(data=30.0))
        ros.deliver("/gas/sensor_array", arr)
        ros.tick(1)
    cmds = ros.published("/cmd_vel/seek")
    assert cmds and any(abs(c.linear.x) + abs(c.angular.z) > 0 for c in cmds)
    st = json.loads(ros.published("/gas/seek_status")[-1].data)
    for k in ("state", "concentration", "gradient", "estimate", "error_m", "path_length", "elapsed"):
        assert k in st
    assert st["state"] in ("SURGE", "GRADIENT")
    assert ros.published("/gas/source_estimate") and st["error_m"] is not None
    # RTH active -> no more commands
    n = len(ros.published("/cmd_vel/seek"))
    ros.deliver("/safety/rth_active", Bool(data=True))
    ros.tick(10)
    assert len(ros.published("/cmd_vel/seek")) <= n + 1
    ros.deliver("/gas/seek/enable", Bool(data=False))
    assert json.loads(ros.published("/gas/seek_status")[-1].data)["state"] == "IDLE"
    # a /map message is accepted
    grid = node.seeker.grid
    om = OccupancyGrid()
    om.info.resolution, om.info.width, om.info.height = grid.resolution, grid.width, grid.height
    om.info.origin.position.x, om.info.origin.position.y = grid.origin_x, grid.origin_y
    om.data = grid.occ.flatten().astype(int).tolist()
    ros.deliver("/map", om)


def test_battery_node(ros):
    node = _node("battery_node").BatteryNode()
    ros.deliver("/odom", _odom(30.0, 0.0, 0.0, vx=0.3))
    ros.tick(30)
    st = ros.published("/battery/state")[-1]
    assert 0.79 < st.percentage < 0.8 and st.current < 0.0
    assert st.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
    ros.deliver("/battery/set_level", Float32(data=19.0))
    assert ros.published("/battery/state")[-1].percentage == pytest.approx(0.19)
    loco = String(data=json.dumps({"mode": "SIT"}))
    ros.deliver("/locomotion/state", loco)
    ros.deliver("/odom", _odom(61.9, 0.0, math.pi))       # on the dock, still
    ros.tick(40)
    st = ros.published("/battery/state")[-1]
    assert st.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_CHARGING
    assert st.percentage > 0.19
    assert node.model.charging


def test_comm_node(ros):
    _node("comm_node").CommNode()
    maps = ros.published("/comm/rssi_map")
    assert maps and len(maps[0].data) == maps[0].info.width * maps[0].info.height
    assert 0 <= min(maps[0].data) and max(maps[0].data) <= 100
    ros.deliver("/odom", _odom(61.0, 0.0, 0.0))
    ros.tick(5)
    r_dock = ros.published("/comm/rssi")[-1].data
    ros.deliver("/odom", _odom(-8.0, -7.5, 0.0))
    ros.tick(5)
    r_pit = ros.published("/comm/rssi")[-1].data
    assert r_dock > -50.0 and r_pit < -80.0


def test_return_to_home_node(ros):
    node = _node("return_to_home_node").ReturnToHomeNode()
    ros.deliver("/odom", _odom(40.0, 0.0, math.pi))
    bs = BatteryState()
    bs.percentage = 0.8
    bs.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
    ros.deliver("/battery/state", bs)
    ros.deliver("/comm/rssi", Float32(data=-50.0))
    ros.tick(10)
    st = json.loads(ros.published("/safety/status")[-1].data)
    for k in ("state", "rth_active", "reason", "battery_pct", "rssi_dbm", "predicted_min_rssi", "distance_home",
              "energy_to_home_pct", "docked"):
        assert k in st
    assert st["state"] == "MONITOR" and ros.published("/safety/rth_active")[-1].data is False
    bs.percentage = 0.19
    ros.deliver("/battery/state", bs)
    t0 = time.time()
    while not ros.published("/nav/plan") and time.time() - t0 < 20.0:
        ros.tick(1)
        time.sleep(0.01)                # planner runs in a background thread
    assert ros.published("/nav/plan"), "home path published"
    ros.tick(20)
    assert ros.published("/safety/rth_active")[-1].data is True
    st = json.loads(ros.published("/safety/status")[-1].data)
    assert st["reason"] == "battery_low" and st["state"] == "RTH_NAVIGATING"
    assert ros.published("/cmd_vel/rth")
    path = ros.published("/nav/plan")[-1]
    assert path.header.frame_id == "map" and len(path.poses) > 2
    # own plan echoed back is ignored; manual command parsing
    ros.deliver("/nav/plan", path)
    ros.deliver("/mission/command", String(data="resume"))
    ros.tick(2)
    assert node.logic.state == "RTH_NAVIGATING"


def _feed_seeker(ros, c=30.0):
    ros.deliver("/gas/sensor", Float32(data=c))
    arr = Float32MultiArray()
    arr.data = [c + 2.0, c + 1.0, c - 2.0, c - 3.0]
    ros.deliver("/gas/sensor_array", arr)


def test_source_seeker_node_time_base_and_terminal_silence(ros):
    ros.params["source_seeker"]["timeout"] = 3.0
    node = _node("source_seeker_node").SourceSeekerNode()
    wind = Vector3Stamped()
    wind.vector.x, wind.vector.y = 0.4, -0.1
    ros.deliver("/gas/wind", wind)
    ros.deliver("/odom", _odom(5.0, 1.0, math.pi))
    # enable arrives before /clock runs (node time 0 with use_sim_time) ...
    ros.t = 0.0
    ros.deliver("/gas/seek/enable", Bool(data=True))
    # ... then the clock jumps to the simulation time: the run must not time out at once
    ros.t = 500.0
    for _ in range(10):
        _feed_seeker(ros)
        ros.tick(1)
    assert node.seeker.state in ("SURGE", "GRADIENT", "SEARCH")
    # time-out (3 s) -> FAILED: one zero Twist, then no more commands
    for _ in range(40):
        _feed_seeker(ros)
        ros.tick(1)
    assert node.seeker.state == "FAILED"
    cmds = ros.published("/cmd_vel/seek")
    last = cmds[-1]
    assert last.linear.x == 0.0 and last.linear.y == 0.0 and last.angular.z == 0.0
    n = len(cmds)
    for _ in range(20):
        _feed_seeker(ros)
        ros.tick(1)
    assert len(ros.published("/cmd_vel/seek")) == n
    assert json.loads(ros.published("/gas/seek_status")[-1].data)["state"] == "FAILED"


def test_return_to_home_node_queues_commands(ros):
    node = _node("return_to_home_node").ReturnToHomeNode()
    ros.deliver("/odom", _odom(40.0, 0.0, math.pi))
    bs = BatteryState()
    bs.percentage = 0.8
    ros.deliver("/battery/state", bs)
    ros.deliver("/mission/command", String(data="rth"))
    ros.deliver("/mission/command", String(data="pause"))      # same control period
    ros.tick(1)
    assert node.logic.reason == "manual" and node.logic.active



def _wait_for_plan(ros, n0=0, timeout=20.0):
    t0 = time.time()
    while len(ros.published("/nav/plan")) <= n0 and time.time() - t0 < timeout:
        ros.tick(1)
        time.sleep(0.01)                # planner runs in a background thread
    assert len(ros.published("/nav/plan")) > n0, "home path published"


def test_return_to_home_node_pause_abort_and_plan_clear(ros):
    node = _node("return_to_home_node").ReturnToHomeNode()
    ros.deliver("/odom", _odom(40.0, 0.0, math.pi))
    bs = BatteryState()
    bs.percentage = 0.8
    ros.deliver("/battery/state", bs)
    ros.deliver("/comm/rssi", Float32(data=-50.0))
    ros.tick(2)
    ros.deliver("/mission/command", String(data="rth"))
    _wait_for_plan(ros)
    ros.tick(2)
    assert node.logic.state == "RTH_NAVIGATING" and ros.published("/safety/rth_active")[-1].data is True
    # 'pause' holds the robot (zero /cmd_vel/rth, rth keeps the mux) until 'resume'
    ros.deliver("/mission/command", String(data="pause"))
    ros.tick(1)
    assert node.logic.paused
    n = len(ros.published("/cmd_vel/rth"))
    ros.tick(6)
    held = ros.published("/cmd_vel/rth")[n:]
    assert held and all(c.linear.x == 0.0 and c.linear.y == 0.0 and c.angular.z == 0.0 for c in held)
    st = json.loads(ros.published("/safety/status")[-1].data)
    assert st["paused"] is True and st["rth_active"] is True
    # 'abort' cancels the manual return: rth_active false and the latched home path is cleared
    ros.deliver("/mission/command", String(data="abort"))
    ros.tick(1)
    assert node.logic.state == "MONITOR" and not node.logic.paused
    assert ros.published("/safety/rth_active")[-1].data is False
    assert json.loads(ros.published("/safety/status")[-1].data)["state"] == "MONITOR"
    cleared = ros.published("/nav/plan")[-1]
    assert cleared.poses == [] and cleared.header.frame_id == "map"
    assert not any(m.data == "stand" for m in ros.published("/locomotion/command"))
    # the empty echo is not filtered as 'own plan' (a mission's empty plan must clear the look-ahead)
    ros.deliver("/nav/plan", cleared)
    assert node.plan_in == []
    ros.tick(10)
    assert node.logic.state == "MONITOR"


def test_return_to_home_node_abort_refused_on_battery_low(ros):
    node = _node("return_to_home_node").ReturnToHomeNode()
    ros.deliver("/odom", _odom(40.0, 0.0, math.pi))
    bs = BatteryState()
    bs.percentage = 0.19
    ros.deliver("/battery/state", bs)
    ros.deliver("/comm/rssi", Float32(data=-50.0))
    _wait_for_plan(ros)
    assert node.logic.reason == "battery_low"
    ros.deliver("/mission/command", String(data="abort"))
    ros.tick(2)
    assert node.logic.active and "abort refused" in node.logic.message
    assert ros.published("/safety/rth_active")[-1].data is True


def _elevation_cloud(centre, box=None, frame_id="odom", res=0.04, n=150):
    """/elevation_map/points stand-in: organised float32 x, y, z, z = 0.5 m inside box (cx, cy, hx, hy)."""
    oi = int(math.floor(centre[0] / res + 0.5)) - n // 2
    oj = int(math.floor(centre[1] / res + 0.5)) - n // 2
    X, Y = np.meshgrid((oi + np.arange(n)) * res, (oj + np.arange(n)) * res)
    Z = np.zeros_like(X)
    if box is not None:
        cx, cy, hx, hy = box
        Z[(np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)] = 0.5
    m = PointCloud2()
    m.header.frame_id = frame_id
    m.height, m.width = n, n
    m.fields = [PointField(name=k, offset=4 * i, datatype=7, count=1) for i, k in enumerate("xyz")]
    m.point_step, m.row_step = 12, 12 * n
    m.data = np.stack([X, Y, Z], -1).astype("<f4").tobytes()
    return m


def test_return_to_home_node_novel_obstacle_input(ros):
    ros.params["return_to_home"]["obstacle_idle_period"] = 0.0          # every frame (also in MONITOR)
    node = _node("return_to_home_node").ReturnToHomeNode()
    subs = [s for s in ros.subs if s.topic == "/elevation_map/points"]
    assert len(subs) == 1 and subs[0].type is PointCloud2
    ros.deliver("/odom", _odom(40.0, 0.0, math.pi))
    bs = BatteryState()
    bs.percentage = 0.8
    ros.deliver("/battery/state", bs)
    mem = node.logic.obstacles.memory
    for _ in range(2):
        ros.deliver("/elevation_map/points", _elevation_cloud((40.0, 0.0), (38.5, 0.0, 0.3, 0.3)))
        ros.tick(1)                                     # the latest frame goes into RthInputs per tick
    assert mem.frames == 2 and mem.has_confirmed() and node.obst_in is None
    ros.tick(1)
    assert mem.frames == 2                              # consumed once
    ros.deliver("/elevation_map/points", _elevation_cloud((40.0, 0.0), None, frame_id="camera_link"))
    bad = _elevation_cloud((40.0, 0.0))
    bad.fields = bad.fields[:1]
    ros.deliver("/elevation_map/points", bad)           # malformed: warned, never raised
    ros.tick(1)
    assert mem.rejected == 1 and mem.frames == 2
    warns = [s for lvl, s in node.get_logger().lines if lvl == "W"]
    assert any("camera_link" in s for s in warns) and any("ignored" in s for s in warns)
    ros.tick(10)
    st = json.loads(ros.published("/safety/status")[-1].data)
    assert st["obstacle"]["frames"] == 2 and st["obstacle"]["confirmed"] > 0


def test_source_seeker_node_novel_obstacle_input(ros):
    ros.params["source_seeker"]["obstacle_idle_period"] = 0.0           # every frame (also when idle)
    node = _node("source_seeker_node").SourceSeekerNode()
    assert [s for s in ros.subs if s.topic == "/elevation_map/points" and s.type is PointCloud2]
    ros.deliver("/elevation_map/points", _elevation_cloud((5.0, 1.0), (6.5, 1.0, 0.3, 0.3)))
    assert node.seeker.obstacles.frames == 0           # no pose yet
    ros.deliver("/odom", _odom(5.0, 1.0, 0.0))
    for _ in range(2):
        ros.deliver("/elevation_map/points", _elevation_cloud((5.0, 1.0), (6.5, 1.0, 0.3, 0.3)))
    assert node.seeker.obstacles.frames == 2 and node.seeker.obstacles.has_confirmed()
    ros.deliver("/gas/seek/enable", Bool(data=True))   # the nav grid is rebuilt when seeking starts
    assert node.seeker.clearance(6.5, 1.0) < node.seeker.cfg.robot_radius      # edges flagged (flat top seen)


def test_return_to_home_node_thins_obstacle_frames_while_monitoring(ros):
    """INT-2: MONITOR fuses one frame per obstacle_idle_period (1 s), a return every frame."""
    node = _node("return_to_home_node").ReturnToHomeNode()
    assert node.logic.obstacles.p.idle_period == 1.0
    ros.deliver("/odom", _odom(40.0, 0.0, math.pi))
    mem = node.logic.obstacles.memory
    for _ in range(10):                                 # 2 s of 5 Hz frames while monitoring
        ros.deliver("/elevation_map/points", _elevation_cloud((40.0, 0.0), (38.5, 0.0, 0.3, 0.3)))
        ros.tick(2)
    assert node.logic.state == "MONITOR" and 2 <= mem.frames <= 3, mem.frames
    ros.deliver("/mission/command", String(data="rth"))
    ros.tick(1)
    assert node.logic.state != "MONITOR"
    n0 = mem.frames
    for _ in range(5):
        ros.deliver("/elevation_map/points", _elevation_cloud((40.0, 0.0), (38.5, 0.0, 0.3, 0.3)))
        ros.tick(2)
    assert mem.frames == n0 + 5


def test_return_to_home_node_reports_its_obstacle_source(ros):
    """F5: /safety/status obstacle.source follows the node parameter obstacle_source."""
    ros.params["return_to_home"]["obstacle_source"] = "traversability"
    node = _node("return_to_home_node").ReturnToHomeNode()
    assert [s for s in ros.subs if s.topic == "/elevation_map/traversability"]
    ros.deliver("/odom", _odom(40.0, 0.0, math.pi))
    ros.tick(10)
    st = json.loads(ros.published("/safety/status")[-1].data)
    assert st["obstacle"]["source"] == "traversability" == node.logic.obstacles.p.source


def test_source_seeker_node_thins_obstacle_frames_while_idle(ros):
    node = _node("source_seeker_node").SourceSeekerNode()
    ros.deliver("/odom", _odom(5.0, 1.0, 0.0))
    for _ in range(10):                                 # 2 s of 5 Hz frames, seeker not enabled
        ros.deliver("/elevation_map/points", _elevation_cloud((5.0, 1.0), (6.5, 1.0, 0.3, 0.3)))
        ros.tick(2)
    assert not node.seeker.active and 2 <= node.seeker.obstacles.frames <= 3


def test_module3_nodes_without_obstacle_input(ros):
    ros.params["return_to_home"]["obstacle_enabled"] = False
    ros.params["source_seeker"]["obstacle_source"] = "traversability"
    _node("return_to_home_node").ReturnToHomeNode()
    _node("source_seeker_node").SourceSeekerNode()
    assert not [s for s in ros.subs if s.topic == "/elevation_map/points"]
    assert [s for s in ros.subs if s.topic == "/elevation_map/traversability" and s.type is OccupancyGrid]


@pytest.mark.parametrize("mod,cls", [("gas_dispersion_node", "GasDispersionNode"),
                                     ("source_seeker_node", "SourceSeekerNode"),
                                     ("battery_node", "BatteryNode"),
                                     ("comm_node", "CommNode"),
                                     ("return_to_home_node", "ReturnToHomeNode")])
def test_nodes_ignore_non_finite_odometry(ros, mod, cls):
    """A diverged physics step must not kill the timers (int(floor(nan)) in the grid lookups)."""
    ros.params["gas_dispersion_sim"].update(spinup_seconds=60.0, spinup_gust_seconds=5.0)
    node = getattr(_node(mod), cls)()
    seeker = mod == "source_seeker_node"
    if seeker:
        wind = Vector3Stamped()
        wind.vector.x, wind.vector.y = 0.4, -0.1
        ros.deliver("/gas/wind", wind)
    ros.deliver("/odom", _odom(5.0, 1.0, math.pi))
    if seeker:
        ros.deliver("/gas/seek/enable", Bool(data=True))
    for _ in range(3):
        if seeker:
            _feed_seeker(ros)
        ros.tick(1)
    pose = node.pose
    assert pose is not None
    nan, inf = float("nan"), float("inf")
    for bad in (_odom(nan, 1.0, 0.0), _odom(5.0, inf, 0.0), _odom(5.0, 1.0, nan), _odom(5.0, 1.0, 0.0, vx=nan)):
        ros.deliver("/odom", bad)
        for _ in range(3):
            if seeker:
                _feed_seeker(ros)
            ros.tick(1)                  # must not raise
    assert node.pose == pose
    warns = [s for lvl, s in node.get_logger().lines if lvl == "W" and "non-finite" in s]
    assert len(warns) == 1                # throttled


def test_source_seeker_node_pause_resume_continues_the_run(ros):
    node = _node("source_seeker_node").SourceSeekerNode()
    wind = Vector3Stamped()
    wind.vector.x, wind.vector.y = 0.4, -0.1
    ros.deliver("/gas/wind", wind)
    ros.deliver("/odom", _odom(5.0, 1.0, math.pi))
    ros.deliver("/gas/seek/enable", Bool(data=True))
    for _ in range(20):
        _feed_seeker(ros)
        ros.tick(1)
    st0 = node.seeker.state
    n0 = len(node.seeker._samples)
    elapsed0 = json.loads(ros.published("/gas/seek_status")[-1].data)["elapsed"]
    assert st0 in ("SURGE", "GRADIENT") and n0 > 0
    # mission pause / teleop override: enable false ... true 5 s later
    ros.deliver("/gas/seek/enable", Bool(data=False))
    assert node.seeker.state == "IDLE"
    ros.tick(50)
    ros.deliver("/gas/seek/enable", Bool(data=True))
    assert node.seeker.state == st0 and len(node.seeker._samples) == n0
    assert any(lvl == "I" and "RESUMED" in s for lvl, s in node.get_logger().lines)
    st = json.loads(ros.published("/gas/seek_status")[-1].data)
    assert st["elapsed"] < elapsed0 + 1.0           # the pause does not count
    n_cmd = len(ros.published("/cmd_vel/seek"))
    for _ in range(10):
        _feed_seeker(ros)
        ros.tick(1)
    assert len(ros.published("/cmd_vel/seek")) > n_cmd and node.seeker.active
    # a long interruption starts a new run
    ros.deliver("/gas/seek/enable", Bool(data=False))
    ros.tick(int(node.seeker.cfg.resume_window_s * 10) + 20)
    ros.deliver("/gas/seek/enable", Bool(data=True))
    assert node.seeker.state == "SEARCH" and node.seeker._samples == []
