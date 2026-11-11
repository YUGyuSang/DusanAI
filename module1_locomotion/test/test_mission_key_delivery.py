"""'t' (and the other mission keys) must not be lost: early key memory, re-send
until /mission/state acknowledges, accepted / rejected feedback, Gazebo-window
keys without a teleop, and an early start that waits for the patrol plan."""
import importlib
import json
import os
import sys
import types

import pytest

from module1_locomotion.teleop_keys import MissionCommandTracker

HERE = os.path.dirname(os.path.abspath(__file__))


def st(seq=None, recent=(), state="IDLE"):
    d = {"state": state}
    if seq is not None:
        d["cmd_seq"] = seq
        d["recent_commands"] = list(recent)
    return d


def test_key_pressed_before_mission_manager_is_kept_and_resent():
    tr = MissionCommandTracker(ack_timeout=2.0, max_sends=3, max_age=60.0)
    send, notes = tr.press("start", 100.0)                 # nobody answers yet
    assert send is True and notes and "not answering" in notes[0]
    for t in (101.0, 110.0, 130.0):                       # mission_manager still starting up
        assert tr.poll(t) == (None, [])
    assert tr.on_state(st(0), 131.0) == []                 # up now, the first 'start' never arrived
    cmd, notes = tr.poll(131.02)
    assert cmd == "start" and "re-sent" in notes[0]
    msgs = tr.on_state(st(1, [{"seq": 1, "cmd": "start", "accepted": True,
                               "message": "full mission started: ['G01']"}], "PATROL"), 131.3)
    assert msgs == ["mission: 'start' accepted - full mission started: ['G01']"]
    assert tr.pending is None and tr.poll(140.0) == (None, [])


def test_rejection_is_reported_and_not_resent():
    tr = MissionCommandTracker()
    tr.on_state(st(4), 10.0)
    assert tr.press("start", 10.1) == (True, [])
    msgs = tr.on_state(st(5, [{"seq": 5, "cmd": "start", "accepted": True, "message": "waiting for "
                               "return_to_home to release before starting (battery 21 %)"}], "RTH"), 10.3)
    assert "accepted - waiting for return_to_home" in msgs[0]
    tr.press("start", 11.0)
    msgs = tr.on_state(st(6, [{"seq": 6, "cmd": "start", "accepted": False,
                               "message": "'start' ignored: mission already running (PATROL); abort first"}]), 11.2)
    assert msgs[0].startswith("   ! mission: 'start' REJECTED - 'start' ignored")
    assert tr.poll(20.0) == (None, [])


def test_lost_command_is_resent_then_given_up():
    tr = MissionCommandTracker(ack_timeout=2.0, max_sends=3)
    tr.on_state(st(0), 0.0)
    tr.press("start", 0.1)
    sent = 1
    t = 0.1
    notes_all = []
    while t < 10.0:
        t += 0.02
        tr.on_state(st(0), t)                               # alive, but never acknowledges
        cmd, notes = tr.poll(t)
        sent += cmd is not None
        notes_all += notes
    assert sent == 3 and "not acknowledged after 3 sends" in notes_all[-1] and tr.pending is None


def test_old_mission_manager_without_ack_fields_gets_the_key_once():
    tr = MissionCommandTracker()
    tr.on_state(st(), 0.0)
    assert tr.press("start", 0.1) == (True, [])
    assert tr.poll(5.0) == (None, []) and tr.pending is None


def test_key_expires_when_mission_manager_never_comes():
    tr = MissionCommandTracker(max_age=60.0)
    tr.press("start", 0.0)
    assert tr.poll(59.0) == (None, [])
    cmd, notes = tr.poll(61.0)
    assert cmd is None and "dropped" in notes[0]
    tr.on_state(st(0), 62.0)                                # comes up later: nothing starts by itself
    assert tr.poll(62.1) == (None, [])


def test_the_latest_key_wins():
    tr = MissionCommandTracker()
    tr.press("start", 0.0)
    tr.press("patrol", 1.0)
    tr.on_state(st(0), 2.0)
    cmd, _ = tr.poll(3.5)
    assert cmd == "patrol"


# ----------------------------------------------------------- with the real mission logic
def test_duplicate_start_from_a_resend_does_not_restart_the_mission():
    from plant_simulation.mission_logic import MissionLogic
    from dusan_common.layout import Layout
    logic = MissionLogic({"min_start_time": 0.0}, Layout.load())
    logic.set_pose(61.9, 0.0, 3.14159, 0.0)
    logic.set_plan({"viewpoints": [{"gauge_id": "G08", "x": 60.0, "y": 0.0, "yaw": 3.14}], "order": ["G08"]})
    assert logic.command("start", 0.0) is True
    logic.step(0.1)
    q0 = [v.gauge_id for v in logic.queue]
    t0 = logic.mission_t0
    assert logic.command("start", 0.5) is False            # the re-sent duplicate
    d = logic.state_dict(0.5)
    assert d["cmd_seq"] == 2 and [r["accepted"] for r in d["recent_commands"]] == [True, False]
    assert d["recent_commands"][0]["message"].startswith("full mission started")
    assert logic.mission_t0 == t0 and [v.gauge_id for v in logic.queue] == q0 and logic.base == "PATROL"


def test_end_to_end_lossy_start_with_tracker_and_logic():
    """The bus drops everything until mission_manager is 'up' (t >= 20 s)."""
    from plant_simulation.mission_logic import MissionLogic
    from dusan_common.layout import Layout
    logic = MissionLogic({"min_start_time": 0.0, "start_plan_wait": 0.0}, Layout.load())
    logic.set_pose(61.9, 0.0, 3.14159, 0.0)
    tr = MissionCommandTracker()
    up = 20.0
    t = 5.0
    sent = []

    def bus(cmd, t):
        sent.append((t, cmd))
        if t >= up:
            logic.command(cmd, t)

    ok, _ = tr.press("start", t)
    bus("start", t)
    log = []
    while t < 40.0:
        t += 0.02
        if t >= up and int(t * 50) % 25 == 0:               # 2 Hz /mission/state once up
            logic.set_pose(61.9, 0.0, 3.14159, t)
            logic.step(t)
            log += tr.on_state(json.loads(json.dumps(logic.state_dict(t))), t)
        cmd, notes = tr.poll(t)
        log += notes
        if cmd:
            bus(cmd, t)
    assert logic.mode == "full" and logic.base in ("PATROL", "INSPECTING")
    assert any("accepted" in m for m in log), log
    assert len(sent) == 2 and sent[1][0] >= up            # exactly one re-send, after it came up


# ----------------------------------------------------------- the ROS wrappers (stubs)
def test_mission_manager_node_acknowledges_immediately(monkeypatch):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "simulation", "test"))
    plant = importlib.import_module("test_plant_nodes_stubbed")
    plant.W = plant.World()
    for name, m in plant._modules().items():
        monkeypatch.setitem(sys.modules, name, m)
    sys.modules.pop("plant_simulation.mission_manager_node", None)
    node = importlib.import_module("plant_simulation.mission_manager_node").MissionManagerNode()
    n0 = len(plant.W.published("/mission/state"))
    plant.W.deliver("/mission/command", plant.String(data="start"))        # no timer tick (clock paused)
    states = plant.W.published("/mission/state")
    assert len(states) == n0 + 1
    d = json.loads(states[-1].data)
    assert d["cmd_seq"] == 1 and d["recent_commands"][-1]["cmd"] == "start"
    sys.modules.pop("plant_simulation.mission_manager_node", None)


def test_keyboard_teleop_node_resends_and_reports(monkeypatch):
    spec = importlib.util.spec_from_file_location("_m1_fake_ros_helpers2", os.path.join(HERE, "test_elevation_map.py"))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    env = helpers.FakeRosEnv(monkeypatch, overrides={"use_terminal": False})
    node_mod = importlib.import_module("module1_locomotion.keyboard_teleop")
    clock = types.SimpleNamespace(t=1000.0)
    monkeypatch.setattr(node_mod, "time", types.SimpleNamespace(monotonic=lambda: clock.t, sleep=lambda s: None))
    node = node_mod.KeyboardTeleop()
    Int32 = env.msg["std_msgs.msg"].Int32
    String = env.msg["std_msgs.msg"].String
    key = lambda code: node.subs["/keyboard/keypress"][1](Int32(data=code))
    state = lambda d: node.subs["/mission/state"][1](String(data=json.dumps(d)))
    out = node.pubs["/mission/command"]
    key(ord("T"))                                        # mission_manager not up yet
    assert [m.data for m in out.msgs] == ["start"]
    assert any("not answering" in line for _, line in node.logger.lines)
    for _ in range(500):                                 # 10 s of silence: no flood
        clock.t += 0.02
        node.tick(clock.t)
    assert len(out.msgs) == 1
    state({"state": "IDLE", "cmd_seq": 0, "recent_commands": []})     # up now, first 'start' was lost
    clock.t += 0.02
    node.tick(clock.t)
    assert [m.data for m in out.msgs] == ["start", "start"]
    state({"state": "PATROL", "cmd_seq": 1, "recent_commands": [
        {"seq": 1, "cmd": "start", "accepted": True, "message": "full mission started: ['G01']"}]})
    assert any("'start' accepted" in line for _, line in node.logger.lines)
    for _ in range(300):
        clock.t += 0.02
        state({"state": "PATROL", "cmd_seq": 1, "recent_commands": []}) if _ % 25 == 0 else None
        node.tick(clock.t)
    assert len(out.msgs) == 2


def test_gazebo_t_key_starts_the_mission_without_a_teleop(monkeypatch):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "simulation", "test"))
    plant = importlib.import_module("test_plant_nodes_stubbed")
    plant.W = plant.World()
    for name, m in plant._modules().items():
        monkeypatch.setitem(sys.modules, name, m)
    sys.modules.pop("plant_simulation.mission_manager_node", None)
    mm = importlib.import_module("plant_simulation.mission_manager_node")
    clock = types.SimpleNamespace(t=500.0)
    monkeypatch.setattr(mm, "time", types.SimpleNamespace(monotonic=lambda: clock.t))
    plant.W.params["mission_manager"] = {"start_plan_wait": 0.0}
    node = mm.MissionManagerNode()
    W = plant.W
    W.deliver("/odom", plant.odom_msg(61.9, 0.0, 3.14159))
    # a running keyboard_teleop (heartbeat) handles the Gazebo keys itself: ignored here
    W.deliver("/teleop/active", plant.Bool(data=False))
    W.deliver("/keyboard/keypress", plant.Int32(data=ord("T")))
    assert node.logic.cmd_seq == 0 and node.logic.base == "IDLE"
    clock.t += 3.0                                        # teleop quit (no heartbeat for 3 s)
    W.deliver("/keyboard/keypress", plant.Int32(data=ord("T")))
    assert node.logic.cmd_seq == 1 and node.logic.base == "PATROL"
    W.deliver("/keyboard/keypress", plant.Int32(data=ord("O")))       # o = pause / resume toggle
    assert node.logic.paused is True
    for _ in range(10):                                   # o held down: OS auto-repeat every 30 ms
        clock.t += 0.03
        W.deliver("/keyboard/keypress", plant.Int32(data=ord("O")))
    assert node.logic.paused is True and node.logic.cmd_seq == 2      # one toggle per press
    clock.t += 1.0
    W.deliver("/keyboard/keypress", plant.Int32(data=ord("O")))
    assert node.logic.paused is False
    W.deliver("/keyboard/keypress", plant.Int32(data=ord("W")))       # motion keys are not the mission's
    assert node.logic.cmd_seq == 3
    sys.modules.pop("plant_simulation.mission_manager_node", None)


def _logic(**p):
    from plant_simulation.mission_logic import MissionLogic
    from dusan_common.layout import Layout
    logic = MissionLogic(dict({"min_start_time": 0.0}, **p), Layout.load())
    logic.set_pose(61.9, 0.0, 3.14159, 0.0)
    return logic


PLAN = {"viewpoints": [{"gauge_id": "G08", "x": 60.0, "y": 0.0, "yaw": 3.14},
                       {"gauge_id": "G06", "x": 40.0, "y": 0.0, "yaw": 3.14}], "order": ["G08", "G06"]}


def test_early_start_waits_for_the_patrol_plan():
    logic = _logic()
    logic.step(1.0)
    assert logic.command("start", 5.0) is True
    d = logic.state_dict(5.0)
    assert d["state"] == "IDLE" and "waiting up to 30 s for the patrol plan" in d["recent_commands"][-1]["message"]
    for k in range(60):                         # planner still computing for 6 s
        t = 5.0 + 0.1 * k
        logic.set_pose(61.9, 0.0, 3.14159, t)
        logic.step(t)
    assert logic.base == "IDLE"
    logic.set_plan(PLAN)                        # plan arrives -> the PLANNED route starts
    logic.set_pose(61.9, 0.0, 3.14159, 11.1)
    logic.step(11.1)
    assert logic.base == "PATROL" and logic.plan_source == "patrol_planner" and logic.mode == "full"
    assert logic.current.gauge_id == "G08" and [v.gauge_id for v in logic.queue] == ["G06"]


def test_start_without_planner_falls_back_after_the_wait():
    logic = _logic(start_plan_wait=30.0)
    logic.step(1.0)
    logic.command("start", 5.0)
    t = 5.0
    while logic.base == "IDLE" and t < 60.0:
        t += 0.1
        logic.set_pose(61.9, 0.0, 3.14159, t)
        logic.step(t)
    assert logic.base == "PATROL" and logic.plan_source == "layout_fallback" and 34.9 <= t <= 35.2
    logic.command("abort", t)
    logic.command("start", t + 1)                # waited once already: immediate fallback now
    assert logic.base == "PATROL"


def test_abort_cancels_a_waiting_start_and_wait_can_be_disabled():
    logic = _logic()
    logic.command("start", 1.0)
    logic.command("abort", 2.0)
    logic.set_plan(PLAN)
    logic.step(3.0)
    assert logic.base == "IDLE" and logic.mode is None
    logic2 = _logic(start_plan_wait=0.0)
    logic2.command("start", 1.0)
    assert logic2.base == "PATROL" and logic2.plan_source == "layout_fallback"


def test_rth_or_seek_cancels_a_start_that_waits_for_the_plan():
    logic = _logic()
    logic.command("start", 1.0)
    assert logic._plan_wait is not None
    assert logic.command("rth", 2.0) is True
    assert logic._plan_wait is None
    assert "cancelled by 'rth'" in " ".join(x for _, x in logic._out.events)
    logic.set_plan(PLAN)
    logic.step(3.0)
    assert logic.mode is None                     # the waiting start did not fire later


def test_plan_wait_survives_a_backwards_clock_jump():
    logic = _logic(start_plan_wait=30.0)
    logic.step(100.0)
    logic.command("start", 100.0)
    logic.step(110.0)
    for k in range(1, 400):                       # sim clock reset to 0 at t = 110
        t = 0.1 * k
        logic.set_pose(61.9, 0.0, 3.14159, t)
        logic.step(t)
        if logic.base != "IDLE":
            break
    assert logic.base == "PATROL" and 29.9 <= t <= 30.3     # waited 30 s from the jump, not 130 s
