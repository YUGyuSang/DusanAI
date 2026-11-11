"""Smoke tests of the six ROS nodes against strict stand-ins of rclpy and the
message packages (ROS is not available on the development PC).

The stand-in messages only accept the fields of the real ROS message
definitions, so a typo in a field name fails here instead of on the robot
PC.  Each node is constructed, fed representative messages through its
subscription callbacks and its timers are ticked.
"""
import importlib
import json
import math
import sys
import types
import urllib.request

import numpy as np
import pytest

from dusan_common.rclpy_logger_rules import CallSiteRules


# ----------------------------------------------------------------------------
# strict message stand-ins
# ----------------------------------------------------------------------------
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


def msg(name, base=Msg, consts=None, **fields):
    cls = type(name, (base,), {"_fields": fields})
    for k, v in (consts or {}).items():
        setattr(cls, k, v)
    return cls


Time = msg("Time", sec=0, nanosec=0)
Duration = msg("Duration", sec=0, nanosec=0)
Header = msg("Header", stamp=Time, frame_id="")
Vector3 = msg("Vector3", x=0.0, y=0.0, z=0.0)
Point = msg("Point", x=0.0, y=0.0, z=0.0)
Quaternion = msg("Quaternion", x=0.0, y=0.0, z=0.0, w=1.0)
Pose = msg("Pose", position=Point, orientation=Quaternion)
PoseWithCovariance = msg("PoseWithCovariance", pose=Pose, covariance=lambda: np.zeros(36))
Twist = msg("Twist", linear=Vector3, angular=Vector3)
TwistWithCovariance = msg("TwistWithCovariance", twist=Twist, covariance=lambda: np.zeros(36))
Odometry = msg("Odometry", header=Header, child_frame_id="", pose=PoseWithCovariance, twist=TwistWithCovariance)
Transform = msg("Transform", translation=Vector3, rotation=Quaternion)
TransformStamped = msg("TransformStamped", header=Header, child_frame_id="", transform=Transform)
MapMetaData = msg("MapMetaData", map_load_time=Time, resolution=0.0, width=0, height=0, origin=Pose)
OccupancyGrid = msg("OccupancyGrid", header=Header, info=MapMetaData, data=list)
PoseStamped = msg("PoseStamped", header=Header, pose=Pose)
Path = msg("Path", header=Header, poses=list)
PointStamped = msg("PointStamped", header=Header, point=Point)
String = msg("String", data="")
Bool = msg("Bool", data=False)
Float32 = msg("Float32", data=0.0)
Float64 = msg("Float64", data=0.0)
Image = msg("Image", header=Header, height=0, width=0, encoding="", is_bigendian=0, step=0, data=bytes)
BatteryState = msg("BatteryState", consts={"POWER_SUPPLY_STATUS_UNKNOWN": 0, "POWER_SUPPLY_STATUS_CHARGING": 1,
                                           "POWER_SUPPLY_STATUS_DISCHARGING": 2},
                   header=Header, voltage=0.0, temperature=0.0, current=0.0, charge=0.0, capacity=0.0,
                   design_capacity=0.0, percentage=0.0, power_supply_status=0, power_supply_health=0,
                   power_supply_technology=0, present=True, cell_voltage=list, cell_temperature=list, location="",
                   serial_number="")
ColorRGBA = msg("ColorRGBA", r=0.0, g=0.0, b=0.0, a=0.0)
Marker = msg("Marker", consts={"ARROW": 0, "CUBE": 1, "SPHERE": 2, "CYLINDER": 3, "LINE_STRIP": 4, "LINE_LIST": 5,
                               "CUBE_LIST": 6, "SPHERE_LIST": 7, "POINTS": 8, "TEXT_VIEW_FACING": 9,
                               "MESH_RESOURCE": 10, "TRIANGLE_LIST": 11, "ADD": 0, "MODIFY": 0, "DELETE": 2,
                               "DELETEALL": 3},
             header=Header, ns="", id=0, type=0, action=0, pose=Pose, scale=Vector3, color=ColorRGBA,
             lifetime=Duration, frame_locked=False, points=list, colors=list, texture_resource="", texture=None,
             uv_coordinates=list, text="", mesh_resource="", mesh_file=None, mesh_use_embedded_materials=False)
MarkerArray = msg("MarkerArray", markers=list)
Clock = msg("Clock", clock=Time)
PointField = type("PointField", (Msg,), {"_fields": {"name": "", "offset": 0, "datatype": 0, "count": 0},
                                         "INT8": 1, "UINT8": 2, "INT16": 3, "UINT16": 4, "INT32": 5, "UINT32": 6,
                                         "FLOAT32": 7, "FLOAT64": 8})   # 'name' field: msg() takes name itself
PointCloud2 = msg("PointCloud2", header=Header, height=0, width=0, fields=list, is_bigendian=False, point_step=0,
                  row_step=0, data=bytes, is_dense=False)


# ----------------------------------------------------------------------------
# rclpy stand-in
# ----------------------------------------------------------------------------
class World:
    """Shared fake clock + registry of every entity the nodes create."""

    def __init__(self):
        self.t = 100.0
        self.params = {}
        self.pubs = []
        self.subs = []
        self.timers = []
        self.tf = []

    def pub(self, topic):
        return [p for p in self.pubs if p.topic == topic]

    def published(self, topic):
        return [m for p in self.pub(topic) for m in p.msgs]

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
                tm.cb()


W = World()


class FakeTime:
    def __init__(self, t):
        self.nanoseconds = int(round(t * 1e9))

    def to_msg(self):
        return Time(sec=int(self.nanoseconds // 10**9), nanosec=int(self.nanoseconds % 10**9))


class FakeClock:
    def __init__(self, clock_type=None):
        self.clock_type = clock_type

    def now(self):
        return FakeTime(W.t)


class FakeLogger:
    def __init__(self):
        self.lines = []
        self.rules = CallSiteRules(skip_files=(__file__,))   # rclpy: one severity / filter set per call site

    def _log(self, level, m, **kw):
        assert set(kw) <= {"throttle_duration_sec", "once", "skip_first"}, kw
        self.rules.check(level, **kw)
        self.lines.append((level, str(m)))

    def debug(self, m, **kw):
        self._log("debug", m, **kw)

    def info(self, m, **kw):
        self._log("info", m, **kw)

    def warning(self, m, **kw):
        self._log("warn", m, **kw)

    def error(self, m, **kw):
        self._log("error", m, **kw)


class Rec:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakePublisher:
    def __init__(self, mtype, topic, qos):
        self.type, self.topic, self.qos, self.msgs = mtype, topic, qos, []

    def publish(self, m):
        assert isinstance(m, self.type), f"{self.topic}: {type(m).__name__} published on a {self.type.__name__} topic"
        self.msgs.append(m)


class FakeParameter:
    def __init__(self, value):
        self.value = value


NODE_KWARGS = {"context", "cli_args", "namespace", "use_global_arguments", "enable_rosout",
               "start_parameter_services", "parameter_overrides", "allow_undeclared_parameters",
               "automatically_declare_parameters_from_overrides"}      # rclpy.node.Node keyword arguments


class FakeNode:
    def __init__(self, name, **kwargs):
        assert set(kwargs) <= NODE_KWARGS, f"unknown rclpy Node kwargs {set(kwargs) - NODE_KWARGS}"
        self._name = name
        self.node_kwargs = kwargs
        self._params = {"use_sim_time": True}
        self._logger = FakeLogger()

    def get_name(self):
        return self._name

    def declare_parameter(self, name, default):
        assert name not in self._params, f"parameter {name} declared twice"
        v = W.params.get(self._name, {}).get(name, default)
        assert type(v) is type(default), f"{name}: override {v!r} vs default {default!r}"
        self._params[name] = v
        return FakeParameter(v)

    def get_parameter(self, name):
        return FakeParameter(self._params[name])

    def create_publisher(self, mtype, topic, qos):
        assert isinstance(topic, str) and topic.startswith("/")
        p = FakePublisher(mtype, topic, qos)
        W.pubs.append(p)
        return p

    def create_subscription(self, mtype, topic, cb, qos):
        assert callable(cb)
        s = Rec(type=mtype, topic=topic, cb=cb, qos=qos)
        W.subs.append(s)
        return s

    def create_timer(self, period, cb, clock=None):
        assert period > 0
        t = Rec(period=period, cb=cb, clock=clock)
        W.timers.append(t)
        return t

    def get_clock(self):
        return FakeClock()

    def get_logger(self):
        return self._logger

    def destroy_node(self):
        pass


class Broadcaster:
    def __init__(self, node, *a, **k):
        pass

    def sendTransform(self, t):
        assert isinstance(t, TransformStamped)
        W.tf.append(t)


def _modules():
    mods = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        mods[name] = m
        return m

    policies = {"DurabilityPolicy": type("DurabilityPolicy", (), {"TRANSIENT_LOCAL": "tl", "VOLATILE": "vol"}),
                "ReliabilityPolicy": type("ReliabilityPolicy", (), {"RELIABLE": "rel", "BEST_EFFORT": "be"}),
                "HistoryPolicy": type("HistoryPolicy", (), {"KEEP_LAST": "kl", "KEEP_ALL": "ka"})}
    state = {"ok": False}

    def init(args=None):
        state["ok"] = True

    def shutdown():
        state["ok"] = False

    def spin(node):
        raise KeyboardInterrupt

    mod("rclpy", init=init, shutdown=shutdown, ok=lambda: state["ok"], spin=spin)
    mod("rclpy.node", Node=FakeNode)
    mod("rclpy.qos", QoSProfile=lambda **kw: Rec(**kw), qos_profile_sensor_data=Rec(reliability="be"), **policies)
    mod("rclpy.clock", Clock=FakeClock, ClockType=type("ClockType", (), {"STEADY_TIME": 2, "ROS_TIME": 1}))
    mod("std_msgs")
    mod("std_msgs.msg", String=String, Bool=Bool, Float32=Float32, Float64=Float64)
    mod("geometry_msgs")
    mod("geometry_msgs.msg", Twist=Twist, TransformStamped=TransformStamped, Point=Point, PointStamped=PointStamped,
        PoseStamped=PoseStamped, Quaternion=Quaternion, Vector3=Vector3)
    mod("nav_msgs")
    mod("nav_msgs.msg", Odometry=Odometry, OccupancyGrid=OccupancyGrid, Path=Path)
    mod("sensor_msgs")
    mod("sensor_msgs.msg", BatteryState=BatteryState, Image=Image, PointCloud2=PointCloud2, PointField=PointField)
    mod("visualization_msgs")
    mod("visualization_msgs.msg", Marker=Marker, MarkerArray=MarkerArray)
    mod("rosgraph_msgs")
    mod("rosgraph_msgs.msg", Clock=Clock)
    mod("tf2_ros", TransformBroadcaster=Broadcaster)
    mod("tf2_ros.static_transform_broadcaster", StaticTransformBroadcaster=Broadcaster)
    return mods


NODE_MODULES = ("map_server_node", "odom_tf_node", "cmd_vel_mux_node", "mission_manager_node",
                "monitor_dashboard_node", "metrics_logger_node", "clock_relay_node")


@pytest.fixture()
def ros(monkeypatch):
    global W
    W = World()
    for name, m in _modules().items():
        monkeypatch.setitem(sys.modules, name, m)
    for n in NODE_MODULES:
        sys.modules.pop("plant_simulation." + n, None)
    yield W
    for n in NODE_MODULES:
        sys.modules.pop("plant_simulation." + n, None)


def load(name):
    return importlib.import_module("plant_simulation." + name)


def odom_msg(x, y, yaw, t=None):
    m = Odometry()
    m.header.stamp = FakeTime(W.t if t is None else t).to_msg()
    m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    m.twist.twist.linear.x = 0.3
    return m


def json_string(d):
    return String(data=json.dumps(d))


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test_map_server(ros):
    node = load("map_server_node").MapServerLite()
    grids = ros.published("/map")
    assert len(grids) == 1 and grids[0].info.width > 100 and grids[0].header.frame_id == "map"
    ros.tick(1, dt=5.0)
    assert len(ros.published("/map")) == 2
    assert ros.pub("/map")[0].qos.durability == "tl"
    assert set(np.unique(np.asarray(grids[0].data))) <= {-1, 0, 100}


def test_odom_tf_spawn_relative(ros):
    node = load("odom_tf_node").OdomTfNode()
    assert ros.tf and ros.tf[0].header.frame_id == "map" and ros.tf[0].child_frame_id == "odom"
    ros.deliver("/model/spot/odometry", odom_msg(0.0, 0.0, 0.0))          # spawn-relative source
    ros.deliver("/model/spot/odometry", odom_msg(1.0, 0.0, 0.0))
    out = ros.published("/odom")
    assert len(out) == 2 and out[0].header.frame_id == "odom" and out[0].child_frame_id == "base_link"
    assert out[0].pose.pose.position.x == pytest.approx(61.9)
    assert out[1].pose.pose.position.x == pytest.approx(60.9)            # spawn yaw = pi
    assert out[1].twist.twist.linear.x == pytest.approx(0.3)
    tf = [t for t in ros.tf if t.child_frame_id == "base_link"]
    assert len(tf) == 2 and tf[-1].transform.translation.x == pytest.approx(60.9)
    assert any("spawn-relative" in m for _, m in node.get_logger().lines)
    ros.tick(3, dt=1.0)                                                  # watchdog


def test_cmd_vel_mux(ros):
    ros.params["cmd_vel_mux"] = {"rate": 20.0}
    node = load("cmd_vel_mux_node").CmdVelMuxNode()
    tw = Twist()
    tw.linear.x = 0.3
    ros.deliver("/cmd_vel/patrol", tw)
    ros.tick(1, dt=0.05)
    assert ros.published("/cmd_vel")[-1].linear.x == pytest.approx(0.3)
    assert ros.published("/cmd_vel_mux/active")[-1].data == "patrol"
    ros.deliver("/teleop/active", Bool(data=True))
    ros.tick(1, dt=0.05)
    assert ros.published("/cmd_vel")[-1].linear.x == 0.0
    assert ros.published("/cmd_vel_mux/active")[-1].data == "teleop"
    ros.tick(20, dt=0.05)
    assert ros.published("/cmd_vel_mux/active")[-1].data == "teleop"


def test_mission_manager_full_cycle(ros):
    ros.params["mission_manager"] = {"min_start_time": 0.2}
    node = load("mission_manager_node").MissionManagerNode()
    assert ros.published("/inspection/target")[-1].data == ""
    assert ros.published("/gas/seek/enable")[-1].data is False
    from dusan_common.grid_map import OccupancyGridMap
    occ = np.zeros((60, 120), np.int8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 100
    grid = OccupancyGridMap(occ, 0.1, 55.0, -3.0)
    ros.deliver("/map", grid.to_msg())
    plan = {"viewpoints": [{"gauge_id": "G08", "x": 60.0, "y": 0.0, "yaw": math.pi}], "order": ["G08"], "legs": []}
    ros.deliver("/patrol/plan_json", json_string(plan))
    x, y, yaw = 61.9, 0.0, math.pi
    ros.deliver("/odom", odom_msg(x, y, yaw))
    ros.deliver("/mission/command", String(data="start"))
    ros.tick(1)                                          # outputs go out on the next control tick
    assert ros.published("/locomotion/command")[-1].data == "stand"
    for _ in range(400):
        ros.deliver("/odom", odom_msg(x, y, yaw))
        st = json.loads(ros.published("/mission/state")[-1].data)
        if st["state"] == "INSPECTING":
            ros.deliver("/inspection/gauge_readings", json_string(
                {"stamp": ros.t, "readings": [{"id": "G08", "value": 0.9, "confidence": 0.9}]}))
        ros.tick(1, dt=0.1)
        cmds = ros.published("/cmd_vel/patrol")
        if cmds:
            c = cmds[-1]
            x += (c.linear.x * math.cos(yaw) - c.linear.y * math.sin(yaw)) * 0.1
            y += (c.linear.x * math.sin(yaw) + c.linear.y * math.cos(yaw)) * 0.1
            yaw += c.angular.z * 0.1
        if st["state"] == "RTH":
            break
    st = json.loads(ros.published("/mission/state")[-1].data)
    assert st["state"] == "RTH" and st["completed"] == ["G08"]
    assert [m.data for m in ros.published("/mission/command")] == ["rth"]
    assert "G08" in [m.data for m in ros.published("/inspection/target")]
    paths = ros.published("/nav/plan")
    assert paths and paths[0].header.frame_id == "map" and len(paths[0].poses) > 5
    # the echo of our own 'rth' is harmless
    ros.deliver("/mission/command", String(data="rth"))
    ros.deliver("/safety/rth_active", Bool(data=True))
    ros.deliver("/safety/status", json_string({"state": "DOCKED", "docked": True, "rth_active": True}))
    ros.tick(2)
    assert json.loads(ros.published("/mission/state")[-1].data)["state"] == "DOCKED"
    ros.deliver("/gas/sensor", Float32(data=3.0))
    ros.deliver("/gas/seek_status", json_string({"state": "IDLE"}))
    ros.deliver("/teleop/active", Bool(data=True))
    ros.deliver("/locomotion/state", json_string({"mode": "STAND"}))
    ros.tick(2)
    assert json.loads(ros.published("/mission/state")[-1].data)["state"] == "MANUAL"


def test_mission_manager_survives_callback_and_step_errors(ros):
    """Field report: 't' printed '-> /mission/command start' but nothing
    happened and the dashboard showed /mission/state stale - one exception in
    a callback ends rclpy.spin and run_node re-raises, so the mission manager
    was gone for the rest of the run.  Now errors are logged, /mission/state
    keeps coming, commands are still accepted, and a step that keeps failing
    aborts the mission so that 't' starts afresh."""
    ros.params["mission_manager"] = {"min_start_time": 0.2}
    node = load("mission_manager_node").MissionManagerNode()
    log = node.get_logger().lines
    # a raising subscription callback is contained
    orig = node.logic.on_safety_status
    node.logic.on_safety_status = lambda d, t: 1 / 0
    ros.deliver("/safety/status", json_string({"state": "MONITOR"}))
    node.logic.on_safety_status = orig
    n_state = len(ros.published("/mission/state"))
    ros.tick(10)
    assert len(ros.published("/mission/state")) > n_state
    assert any(lvl == "error" and "_on_safety failed" in m and "ZeroDivisionError" in m for lvl, m in log)
    # start a mission, then make every step raise
    from dusan_common.grid_map import OccupancyGridMap
    occ = np.zeros((60, 120), np.int8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 100
    ros.deliver("/map", OccupancyGridMap(occ, 0.1, 55.0, -3.0).to_msg())
    plan = {"viewpoints": [{"gauge_id": "G08", "x": 60.0, "y": 0.0, "yaw": math.pi}], "order": ["G08"], "legs": []}
    ros.deliver("/patrol/plan_json", json_string(plan))
    ros.deliver("/odom", odom_msg(61.9, 0.0, math.pi))
    ros.deliver("/mission/command", String(data="start"))
    ros.tick(2)
    assert json.loads(ros.published("/mission/state")[-1].data)["state"] == "PATROL"

    def boom(t, dt):
        raise RuntimeError("logic bug")
    node.logic._step_patrol = boom
    n_state = len(ros.published("/mission/state"))
    ros.tick(1)
    assert json.loads(ros.published("/mission/state")[-1].data)["state"] == "PATROL"   # one failure: kept
    ros.tick(2)
    del node.logic._step_patrol                                   # back to the class method
    ros.tick(1)
    st = json.loads(ros.published("/mission/state")[-1].data)
    assert st["state"] == "IDLE" and "internal error" in st["message"]
    assert len(ros.published("/mission/state")) > n_state
    assert sum(1 for lvl, m in log if lvl == "error" and "mission step failed" in m) == 3
    # 't' works again
    ros.deliver("/odom", odom_msg(61.9, 0.0, math.pi))
    ros.deliver("/mission/command", String(data="start"))
    ros.tick(2)
    assert json.loads(ros.published("/mission/state")[-1].data)["state"] == "PATROL"


def elevation_cloud(centre, box=None, frame_id="odom", res=0.04, n=150):
    """/elevation_map/points stand-in: organised float32 x, y, z cloud around centre,
    z = 0.5 m inside ``box`` (cx, cy, half_x, half_y), 0 elsewhere."""
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


def test_mission_manager_novel_obstacle_input(ros):
    ros.params["mission_manager"] = {"min_start_time": 0.2, "obstacle_idle_period": 0.0}   # every frame fused
    node = load("mission_manager_node").MissionManagerNode()
    subs = [s for s in ros.subs if s.topic == "/elevation_map/points"]
    assert len(subs) == 1 and subs[0].type is PointCloud2
    assert subs[0].qos.reliability == "be" and subs[0].qos.depth == 1
    assert not [s for s in ros.subs if s.topic == "/elevation_map/traversability"]
    mem = node.logic.obstacles.memory
    ros.deliver("/elevation_map/points", elevation_cloud((60.0, 0.0), (58.5, 0.0, 0.3, 0.3)))
    assert mem.frames == 0                                    # no /odom yet: frame skipped
    ros.deliver("/odom", odom_msg(60.0, 0.0, math.pi))
    for _ in range(2):
        ros.deliver("/elevation_map/points", elevation_cloud((60.0, 0.0), (58.5, 0.0, 0.3, 0.3)))
    assert mem.frames == 2 and mem.has_confirmed()
    c = mem.confirmed_xy()
    assert np.all(np.abs(c[:, 0] - 58.5) < 0.5) and np.all(np.abs(c[:, 1]) < 0.5)
    # a cloud in another frame is rejected (counted, warned once per period)
    ros.deliver("/elevation_map/points", elevation_cloud((60.0, 0.0), None, frame_id="base_link"))
    ros.deliver("/elevation_map/points", elevation_cloud((60.0, 0.0), None, frame_id="base_link"))
    assert mem.frames == 2 and mem.rejected == 2
    warns = [m for lvl, m in node.get_logger().lines if lvl == "warn" and "rejected" in m]
    assert len(warns) == 1 and "base_link" in warns[0]
    # a malformed cloud is contained by the guarded callback
    bad = elevation_cloud((60.0, 0.0))
    bad.fields = bad.fields[:2]
    ros.deliver("/elevation_map/points", bad)
    assert any(lvl == "error" and "_on_obstacle_cloud failed" in m for lvl, m in node.get_logger().lines)
    ros.tick(2)
    st = json.loads(ros.published("/mission/state")[-1].data)
    assert st["obstacle"]["frames"] == 2 and st["obstacle"]["rejected"] == 2 and st["obstacle"]["confirmed"] > 0


def test_mission_manager_obstacle_source_and_switch(ros):
    ros.params["mission_manager"] = {"obstacle_source": "traversability", "obstacle_idle_period": 0.0}
    node = load("mission_manager_node").MissionManagerNode()
    subs = [s for s in ros.subs if s.topic == "/elevation_map/traversability"]
    assert len(subs) == 1 and subs[0].type is OccupancyGrid
    assert not [s for s in ros.subs if s.topic == "/elevation_map/points"]
    from dusan_common.grid_map import OccupancyGridMap
    occ = np.zeros((150, 150), np.int8)
    occ[60:75, 30:40] = 100
    g = OccupancyGridMap(occ, 0.04, 55.0, -3.0, "odom")
    ros.deliver("/odom", odom_msg(58.0, 0.0, math.pi))
    for _ in range(2):
        ros.deliver("/elevation_map/traversability", g.to_msg())
    assert node.logic.obstacles.memory.frames == 2 and node.logic.obstacles.memory.has_confirmed()
    # obstacle_enabled false: no subscription at all
    global W
    W.subs.clear()
    sys.modules.pop("plant_simulation.mission_manager_node", None)
    ros.params["mission_manager"] = {"obstacle_enabled": False}
    load("mission_manager_node").MissionManagerNode()
    assert not [s for s in ros.subs if s.topic.startswith("/elevation_map")]


def test_mission_manager_thins_obstacle_frames_while_not_driving(ros):
    """INT-2: idle / inspecting / docked = one frame per obstacle_idle_period (1 s), converted only then."""
    node = load("mission_manager_node").MissionManagerNode()
    assert node.logic.obstacles.p.idle_period == 1.0 and node.logic.base == "IDLE"
    mem = node.logic.obstacles.memory
    for _ in range(10):                                       # 2 s of 5 Hz frames
        ros.deliver("/odom", odom_msg(60.0, 0.0, math.pi))
        ros.deliver("/elevation_map/points", elevation_cloud((60.0, 0.0), (58.5, 0.0, 0.3, 0.3)))
        ros.tick(2)
    assert 2 <= mem.frames <= 3, mem.frames


def test_mission_manager_logs_events_of_every_level_without_dying(ros):
    """Field report (Lyrical): rclpy raises 'Logger severity cannot be changed
    between calls' when one line logs with another severity.  The mission
    manager logged all its events through one line and died on the first
    warning after info events (entering the factory).  The stub logger now
    enforces rclpy's per-call-site rule (dusan_common.rclpy_logger_rules)."""
    node = load("mission_manager_node").MissionManagerNode()
    log = node.get_logger().lines
    ros.deliver("/mission/command", String(data="pause"))       # info: nothing to pause
    ros.tick(2)
    ros.deliver("/mission/command", String(data="bogus"))       # warn: unknown mission command
    ros.tick(2)
    ros.deliver("/mission/command", String(data="pause"))       # info again from the same code path
    ros.tick(2)
    levels = {lvl for lvl, _ in log}
    assert {"info", "warn"} <= levels
    assert not [m for lvl, m in log if lvl == "error"], [m for lvl, m in log if lvl == "error"]
    assert any("unknown mission command" in m for lvl, m in log if lvl == "warn")


def test_monitor_dashboard(ros):
    ros.params["monitor_dashboard"] = {"port": 0, "host": "127.0.0.1"}
    mod = load("monitor_dashboard_node")
    node = mod.MonitorDashboardNode()
    try:
        port = node.server.port
        assert port
        ros.deliver("/odom", odom_msg(3.0, 6.0, 0.4))
        ros.deliver("/locomotion/state", json_string({"mode": "WALK", "gait": "trot", "mpc_solve_ms": 9.5}))
        ros.deliver("/mission/state", json_string({"state": "PATROL", "target": "G06", "message": "x"}))
        ros.deliver("/inspection/gauge_readings", json_string(
            {"readings": [{"id": "G06", "value": 1.2, "unit": "bar", "status": "warning", "confidence": 0.8}]}))
        ros.deliver("/inspection/thermal_alerts", json_string(
            {"hotspots": [{"id": "H02", "max_temp_c": 97.0, "status": "alarm", "position": [5.0, 6.4, 1.0]}]}))
        b = BatteryState(percentage=0.55, voltage=24.8, power_supply_status=2)
        ros.deliver("/battery/state", b)
        ros.deliver("/comm/rssi", Float32(data=-62.0))
        ros.deliver("/gas/sensor", Float32(data=4.0))
        g = OccupancyGrid()
        g.info.width, g.info.height, g.info.resolution = 8, 4, 0.25
        g.data = [10] * 32
        ros.deliver("/gas/concentration_grid", g)
        p = Path()
        p.poses = [PoseStamped() for _ in range(3)]
        ros.deliver("/nav/plan", p)
        ps = PointStamped()
        ps.point.x, ps.point.y = -3.0, 2.2
        ros.deliver("/gas/source_estimate", ps)
        img = Image(height=4, width=6, encoding="rgb8", step=18, data=bytes(72))
        ros.deliver("/inspection/gauge_overlay", img)
        tw = Twist()
        ros.deliver("/cmd_vel", tw)
        ros.deliver("/cmd_vel_mux/active", String(data="patrol"))
        ros.tick(2, dt=1.0)
        arr = ros.published("/twin/markers")
        assert arr and len(arr[-1].markers) > 10
        texts = [m.text for m in arr[-1].markers if m.text]
        assert any(t.startswith("G06 1.20") for t in texts) and any("H02 97.0" in t for t in texts)
        base = f"http://127.0.0.1:{port}"
        snap = json.loads(urllib.request.urlopen(base + "/api/state", timeout=5).read())
        assert snap["battery"]["pct"] == pytest.approx(55.0)
        assert snap["robot"]["x"] == pytest.approx(3.0)
        assert snap["gas"]["estimate"] == [-3.0, 2.2]
        jpg = urllib.request.urlopen(base + "/image/gauge.jpg", timeout=5).read()
        assert jpg[:2] == b"\xff\xd8"
        png = urllib.request.urlopen(base + "/gas.png", timeout=5).read()
        assert png[:4] == b"\x89PNG"
        req = urllib.request.Request(base + "/api/command", data=json.dumps({"target": "mission",
                                                                             "command": "start"}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"]
        assert ros.published("/mission/command")[-1].data == "start"
        req = urllib.request.Request(base + "/api/command", data=json.dumps({"target": "battery",
                                                                             "command": "19"}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        assert ros.published("/battery/set_level")[-1].data == pytest.approx(19.0)
    finally:
        node.on_shutdown()


def test_monitor_dashboard_survives_malformed_module_messages(ros):
    """Odd but plausible payloads must never raise inside a callback / timer
    (an exception there stops rclpy.spin and kills the dashboard)."""
    ros.params["monitor_dashboard"] = {"port": 0, "host": "127.0.0.1"}
    node = load("monitor_dashboard_node").MonitorDashboardNode()
    try:
        ros.deliver("/inspection/thermal_alerts", json_string({"hotspots": [
            {"id": "H01", "max_temp_c": 71.0, "status": "warning", "position": [2.0, 6.4]},        # no z
            {"id": "H03", "max_temp_c": 66.0, "status": "warning", "position": [8.0, 6.4, None]},  # z unknown
            {"id": None, "max_temp_c": 50.0, "status": "normal", "position": None}]}))
        ros.deliver("/gas/seek_status", json_string({"state": "FOUND", "estimate": [None, "x"]}))
        ros.deliver("/locomotion/fall_event", json_string({"event": "fall", "roll": "n/a", "x": "?", "y": 1}))
        ros.deliver("/patrol/plan_json", json_string({"viewpoints": ["bad", {"gauge_id": "G01", "x": 1, "y": 2}],
                                                      "order": ["G01"]}))
        ros.tick(2, dt=1.0)
        arr = ros.published("/twin/markers")
        assert arr, "marker timer must keep publishing"
        pos = {(round(m.pose.position.x, 1), round(m.pose.position.y, 1)): m.pose.position.z
               for m in arr[-1].markers if m.ns == "twin_hotspots"}
        assert (2.0, 6.4) in pos and (8.0, 6.4) in pos
        snap = node.state.snapshot()
        assert snap["gas"]["estimate"] is None and snap["gas"]["estimate_error_m"] is None
        assert [v["gauge_id"] for v in snap["plan"]["viewpoints"]] == ["G01"]
    finally:
        node.on_shutdown()


def test_metrics_logger(ros, tmp_path):
    ros.params["metrics_logger"] = {"output_dir": str(tmp_path), "session_name": "s1"}
    node = load("metrics_logger_node").MetricsLoggerNode()
    for k in range(5):
        ros.deliver("/odom", odom_msg(61.9 - 0.3 * k, 0.0, math.pi, t=ros.t))
        ros.tick(1, dt=0.5)
    ros.deliver("/inspection/gauge_readings", json_string({"readings": [{"id": "G08", "value": 0.95}]}))
    ros.deliver("/battery/state", BatteryState(percentage=0.5))
    ros.deliver("/safety/status", json_string({"state": "RTH_PLANNING", "rth_active": True, "reason": "manual"}))
    ros.deliver("/gas/seek_status", json_string({"state": "FOUND", "estimate": [-3.1, 2.0]}))
    node.on_shutdown()
    s = json.load(open(tmp_path / "s1" / "summary.json", encoding="utf-8"))
    assert s["distance_m"] == pytest.approx(1.2, abs=0.01)
    assert s["gauges"]["per_gauge"]["G08"]["error_pct_fs"] == pytest.approx(1.875, abs=1e-3)
    assert s["rth"]["count"] == 1 and s["gas"]["source_error_m"] == pytest.approx(0.1, abs=1e-6)


def test_run_node_handles_ctrl_c_and_calls_on_shutdown(ros):
    from plant_simulation.node_utils import run_node
    calls = []

    class N(FakeNode):
        def __init__(self):
            super().__init__("n")

        def on_shutdown(self):
            calls.append("shutdown")

    run_node(N)                      # stand-in spin() raises KeyboardInterrupt
    assert calls == ["shutdown"]
    assert not sys.modules["rclpy"].ok()


def test_clock_relay_decimates_the_gazebo_clock(ros, monkeypatch):
    mod = load("clock_relay_node")
    wall = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: wall[0])
    node = mod.ClockRelayNode()
    assert node.node_kwargs.get("start_parameter_services") is False
    sub = [s for s in ros.subs if s.topic == "/clock_gz"]
    assert len(sub) == 1 and sub[0].type is Clock and sub[0].qos.reliability == "be"
    assert ros.pub("/clock")[0].qos.reliability == "rel"
    assert not [s for s in ros.subs if s.topic == "/clock"]          # never listens to its own output

    def gz_clock(ms):
        ns = int(ms) * 1000000
        return Clock(clock=Time(sec=ns // 10**9, nanosec=ns % 10**9))

    for ms in range(12000, 12100):                    # 100 physics steps of 1 ms
        ros.deliver("/clock_gz", gz_clock(ms))
    out = [m.clock.sec * 1000 + m.clock.nanosec // 1000000 for m in ros.published("/clock")]
    assert out == list(range(12000, 12100, 5))        # 200 Hz on the 5 ms grid
    n = len(out)
    for _ in range(50):                               # paused: same time, 50 ms wall
        wall[0] += 0.001
        ros.deliver("/clock_gz", gz_clock(12099))
    assert len(ros.published("/clock")) == n
    wall[0] += 0.5
    ros.deliver("/clock_gz", gz_clock(12099))         # keepalive while paused
    assert len(ros.published("/clock")) == n + 1
    ros.deliver("/clock_gz", gz_clock(3))             # world reset: forwarded at once
    assert ros.published("/clock")[-1].clock.nanosec == 3000000
