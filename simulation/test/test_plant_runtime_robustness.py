"""mission_manager keeps running (and 't' keeps working) through malformed messages on every
input, NaN odometry, a backwards sim-clock jump (Gazebo reset) and repeated / early mission keys
- rclpy stand-ins of test_plant_nodes_stubbed."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import test_plant_nodes_stubbed as P  # noqa: E402
from test_plant_nodes_stubbed import load, odom_msg, ros  # noqa: E402,F401

BAD_JSON = ["[1, 2]", '"PATROL"', "null", "{not json", "", "42", "1e999",
            json.dumps({"state": None, "mode": ["x"], "battery_pct": "low", "viewpoints": "abc"}),
            json.dumps({"viewpoints": [{"gauge_id": None, "x": "a"}], "order": 5}),
            json.dumps({"readings": [None, 3, {"gauge_id": 7, "value": "nan"}]}),
            json.dumps({"state": "FOUND", "estimate": "here"})]
STRING_INPUTS = ("/patrol/plan_json", "/gas/seek_status", "/inspection/gauge_readings", "/safety/status",
                 "/locomotion/state")


def _errors(node):
    return [m for lvl, m in node.get_logger().lines if lvl == "error"]


def _state(W):
    return json.loads(W.published("/mission/state")[-1].data)


def _mission_manager(W):
    W.params["mission_manager"] = {"start_plan_wait": 0.0}
    node = load("mission_manager_node").MissionManagerNode()
    W.deliver("/odom", odom_msg(61.9, 0.0, math.pi))
    W.tick(3)
    return node


def test_mission_manager_survives_malformed_messages(ros):
    W = ros
    node = _mission_manager(W)
    for topic in STRING_INPUTS:
        for bad in BAD_JSON:
            W.deliver(topic, P.String(data=bad))
            W.tick(1)
    for bad in ("", "   ", "START!", "start; rm -rf /", "x" * 5000):
        W.deliver("/mission/command", P.String(data=bad))
    for code in (-1, 0, 2 ** 31 - 1, 84):                      # 84 = 'T'
        W.deliver("/keyboard/keypress", P.Int32(data=code))
    nan = odom_msg(float("nan"), 0.0, 0.0)
    W.deliver("/odom", nan)
    W.tick(5)
    W.deliver("/odom", odom_msg(61.9, 0.0, math.pi))
    W.tick(5)
    st = _state(W)
    assert st["state"] in ("IDLE", "PATROL", "INSPECTING")
    # the Gazebo 't' above started the mission (no teleop heartbeat); a second 't' is rejected, not fatal
    W.deliver("/mission/command", P.String(data="start"))
    st = _state(W)
    assert st["recent_commands"][-1]["cmd"] == "start"
    errs = _errors(node)
    assert not [e for e in errs if "failed" in e], errs[:3]


def test_mission_manager_survives_a_backwards_clock_jump(ros):
    W = ros
    node = _mission_manager(W)
    W.deliver("/mission/command", P.String(data="start"))
    W.tick(10)
    assert _state(W)["state"] in ("PATROL", "INSPECTING")
    W.t -= 80.0                                                  # Gazebo world reset
    n = len(W.published("/mission/state"))
    for _ in range(20):
        W.deliver("/odom", odom_msg(61.9, 0.0, math.pi))
        W.tick(1)
    assert len(W.published("/mission/state")) > n               # still publishing after the jump
    W.deliver("/mission/command", P.String(data="abort"))
    W.deliver("/mission/command", P.String(data="start"))
    st = _state(W)
    assert [r["cmd"] for r in st["recent_commands"][-2:]] == ["abort", "start"]
    assert st["recent_commands"][-1]["accepted"] is True
    assert not [e for e in _errors(node) if "failed" in e]


def test_mission_manager_step_errors_end_in_a_clean_abort(ros):
    """A logic error inside the state machine: logged, after 3 ticks the mission is aborted and
    't' starts afresh (the node itself never dies)."""
    W = ros
    node = _mission_manager(W)
    W.deliver("/mission/command", P.String(data="start"))
    W.tick(2)
    real = node.logic.step
    node.logic.step = lambda t: (_ for _ in ()).throw(RuntimeError("logic bug"))
    W.tick(4)
    node.logic.step = real
    assert any("logic bug" in e for e in _errors(node))
    W.tick(2)
    assert _state(W)["state"] == "IDLE"
    W.deliver("/mission/command", P.String(data="start"))
    W.tick(3)
    assert _state(W)["state"] in ("PATROL", "INSPECTING")
