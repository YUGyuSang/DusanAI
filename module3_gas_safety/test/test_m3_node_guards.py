"""Module-3 nodes survive what used to kill them: malformed JSON (lists, strings, garbage),
NaN / inf readings, a grid whose data does not fit its size, an exception inside the logic of a
timer, and the simulation clock jumping backwards (Gazebo reset).  Same strict rclpy / message
stand-ins as test_m3_nodes_stubbed."""
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_m3_nodes_stubbed import (BatteryState, Bool, Float32, Float32MultiArray, OccupancyGrid,  # noqa: E402,F401
                                   String, Vector3Stamped, _node, _odom, ros)

BAD_JSON = ["[1, 2]", '"SIT"', "null", "{not json", "", "42", json.dumps({"mode": None}),
            json.dumps({"mode": ["SIT"]})]


def _errors(node):
    return [s for lv, s in node.get_logger().lines if lv == "E"]


def test_node_common_guarded_logs_and_keeps_going(ros):
    nc = _node("node_common")

    class N:
        def __init__(self):
            from rclpy.node import Node
            self._n = Node("x")

        def get_logger(self):
            return self._n.get_logger()

    n = N()
    calls = []

    def boom(v):
        calls.append(v)
        if v < 0:
            raise ValueError("negative")
        return v * 2

    g = nc.guarded(n, "boom", boom, every=3)
    assert g(2) == 4
    for _ in range(4):
        assert g(-1) is None
    assert g(3) == 6 and g.failures() == 4
    errs = [s for lv, s in n.get_logger().lines if lv == "E"]
    assert len(errs) == 2 and "boom failed (1x): ValueError: negative" in errs[0] and "(3x)" in errs[1]
    assert nc.finite_or_none("nan") is None and nc.finite_or_none(float("inf")) is None
    assert nc.finite_or_none(None) is None and nc.finite_or_none("2.5") == 2.5
    assert nc.json_dict(String(data="[1]")) == {} and nc.json_dict(String(data='{"a": 1}')) == {"a": 1}


def test_battery_node_ignores_malformed_input(ros):
    node = _node("battery_node").BatteryNode()
    ros.deliver("/odom", _odom(30.0, 0.0, 0.0, vx=0.3))
    for bad in BAD_JSON:
        ros.deliver("/locomotion/state", String(data=bad))
    assert node.mode == "stand"
    ros.deliver("/locomotion/state", String(data=json.dumps({"mode": "SIT"})))
    assert node.mode == "sit"
    soc = node.model.soc
    for bad in (float("nan"), float("inf"), float("-inf")):
        ros.deliver("/battery/set_level", Float32(data=bad))
    assert node.model.soc == soc and math.isfinite(node.model.soc)
    ros.tick(20)
    st = ros.published("/battery/state")[-1]
    assert math.isfinite(st.percentage) and 0.0 < st.percentage <= 1.0
    assert not _errors(node)


def test_battery_node_survives_a_time_jump_and_a_logic_error(ros):
    node = _node("battery_node").BatteryNode()
    ros.deliver("/odom", _odom(30.0, 0.0, 0.0, vx=0.3))
    ros.tick(10)
    p0 = node.model.soc
    ros.t -= 50.0                                   # Gazebo world reset: sim time goes backwards
    ros.tick(10)
    assert math.isfinite(node.model.soc) and node.model.soc <= p0
    real = node.model.step
    node.model.step = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model bug"))
    ros.tick(5)                                     # raised inside the timer: logged, not fatal
    assert any("model bug" in e for e in _errors(node))
    node.model.step = real
    n = len(ros.published("/battery/state"))
    ros.tick(10)
    assert len(ros.published("/battery/state")) > n


def test_return_to_home_node_ignores_malformed_input(ros):
    node = _node("return_to_home_node").ReturnToHomeNode()
    ros.deliver("/odom", _odom(40.0, 0.0, math.pi))
    for bad in BAD_JSON:
        ros.deliver("/locomotion/state", String(data=bad))
    assert node.loco_mode == ""
    ros.deliver("/locomotion/state", String(data=json.dumps({"mode": "sit"})))
    assert node.loco_mode == "SIT"
    bs = BatteryState()
    bs.percentage = float("nan")
    ros.deliver("/battery/state", bs)
    assert node.battery_pct is None
    bs.percentage = 0.8
    ros.deliver("/battery/state", bs)
    assert node.battery_pct == pytest.approx(80.0)
    ros.deliver("/comm/rssi", Float32(data=float("nan")))
    assert node.rssi is None
    ros.deliver("/comm/rssi", Float32(data=-55.0))
    bad_grid = OccupancyGrid()
    bad_grid.info.width, bad_grid.info.height, bad_grid.info.resolution = 10, 10, 0.5
    bad_grid.data = [50] * 7                        # does not fit 10 x 10
    before = node._rssi_map
    ros.deliver("/comm/rssi_map", bad_grid)
    assert node._rssi_map is before
    ros.deliver("/mission/command", String(data=""))
    ros.tick(10)
    st = json.loads(ros.published("/safety/status")[-1].data)
    assert st["state"] == "MONITOR" and st["battery_pct"] == pytest.approx(80.0)
    assert not _errors(node)


def test_return_to_home_node_survives_a_logic_error_and_a_time_jump(ros):
    node = _node("return_to_home_node").ReturnToHomeNode()
    ros.deliver("/odom", _odom(40.0, 0.0, math.pi))
    bs = BatteryState()
    bs.percentage = 0.8
    ros.deliver("/battery/state", bs)
    ros.tick(5)
    real = node.logic.update
    node.logic.update = lambda inp: (_ for _ in ()).throw(KeyError("broken"))
    ros.tick(5)
    assert any("_on_timer failed" in e and "broken" in e for e in _errors(node))
    node.logic.update = real
    ros.t -= 100.0
    n = len(ros.published("/safety/status"))
    ros.tick(20)
    assert len(ros.published("/safety/status")) > n
    assert json.loads(ros.published("/safety/status")[-1].data)["state"] == "MONITOR"


def test_source_seeker_node_ignores_non_finite_readings(ros):
    node = _node("source_seeker_node").SourceSeekerNode()
    ros.deliver("/odom", _odom(5.0, 1.0, math.pi))
    ros.deliver("/gas/sensor", Float32(data=30.0))
    for bad in (float("nan"), float("inf")):
        ros.deliver("/gas/sensor", Float32(data=bad))
    assert node.c_nose == 30.0
    arr = Float32MultiArray()
    arr.data = [32.0, float("nan"), 28.0, 27.0]
    ros.deliver("/gas/sensor_array", arr)
    assert node.c_array is None
    arr.data = [32.0, 31.0, 28.0, 27.0]
    ros.deliver("/gas/sensor_array", arr)
    assert node.c_array == [32.0, 31.0, 28.0, 27.0]
    wind = Vector3Stamped()
    wind.vector.x, wind.vector.y = float("nan"), 0.1
    ros.deliver("/gas/wind", wind)
    assert node.wind is None
    ros.deliver("/gas/seek/enable", Bool(data=True))
    for _ in range(10):
        ros.deliver("/gas/sensor", Float32(data=30.0))
        ros.deliver("/gas/sensor_array", arr)
        ros.tick(1)
    st = json.loads(ros.published("/gas/seek_status")[-1].data)
    assert st["state"] in ("SURGE", "GRADIENT", "CAST")
    assert all(math.isfinite(c.linear.x) and math.isfinite(c.angular.z) for c in ros.published("/cmd_vel/seek"))
    assert not _errors(node)


def test_source_seeker_node_survives_a_logic_error(ros):
    node = _node("source_seeker_node").SourceSeekerNode()
    ros.deliver("/odom", _odom(5.0, 1.0, math.pi))
    ros.deliver("/gas/seek/enable", Bool(data=True))
    real = node.seeker.update
    node.seeker.update = lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError("seek"))
    for _ in range(5):
        ros.deliver("/gas/sensor", Float32(data=30.0))
        ros.tick(1)
    assert any("_on_control failed" in e for e in _errors(node))
    node.seeker.update = real
    n = len(ros.published("/cmd_vel/seek"))
    for _ in range(10):
        ros.deliver("/gas/sensor", Float32(data=30.0))
        ros.tick(1)
    assert len(ros.published("/cmd_vel/seek")) > n


@pytest.mark.parametrize("mod,cls", [("comm_node", "CommNode"), ("gas_dispersion_node", "GasDispersionNode")])
def test_comm_and_gas_timers_are_guarded(ros, mod, cls):
    if mod == "gas_dispersion_node":
        ros.params["gas_dispersion_sim"].update(spinup_seconds=20.0, spinup_gust_seconds=2.0)
    node = getattr(_node(mod), cls)()
    ros.deliver("/odom", _odom(10.0, 0.0, 0.0))
    target = node.model
    name = "rssi_at" if mod == "comm_node" else "step"
    real = getattr(target, name)
    setattr(target, name, lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    ros.tick(5)                                     # must not raise out of the timer
    assert _errors(node)
    setattr(target, name, real)
    ros.tick(5)
