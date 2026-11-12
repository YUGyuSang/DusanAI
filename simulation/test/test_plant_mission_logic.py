"""Mission state machine tests (ROS free): a kinematic robot follows the
mission's /cmd_vel/patrol output inside a synthetic room."""
import math
import os

import numpy as np
import pytest

from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout

from plant_simulation.layout_map import rasterize_layout
from plant_simulation.mission_logic import (MISSION_DEFAULTS, MissionLogic, cross_track_distance,
                                            fallback_viewpoints, parse_patrol_plan)

DT = 0.1
PLANT_MAP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", "plant_map.yaml"))


# ----------------------------------------------------------------------------
# fixtures / harness
# ----------------------------------------------------------------------------
def room_grid():
    """20 m x 10 m room (origin -2,-2) with a partial wall at x = 8."""
    occ = np.zeros((100, 200), np.int8)
    occ[:, 0] = occ[:, -1] = occ[0, :] = occ[-1, :] = 100
    occ[0:60, 100] = 100            # wall x = 8.0 m from y = -2 .. 4, gap above
    return OccupancyGridMap(occ, 0.1, -2.0, -2.0)


class FakeLayout:
    """Minimal Layout stand-in: dock pose + gauges."""

    def __init__(self, dock=(0.0, 0.0, 0.0)):
        self.dock_pose = dock
        self.gauges = [
            {"id": "GA", "position": [4.0, 5.0, 1.0], "normal": [0, -1, 0]},
            {"id": "GB", "position": [12.0, 1.0, 1.0], "normal": [-1, 0, 0]},
        ]


PLAN = {
    "start": [0.0, 0.0],
    "viewpoints": [
        {"gauge_id": "GA", "x": 4.0, "y": 3.6, "yaw": math.pi / 2},
        {"gauge_id": "GB", "x": 10.6, "y": 1.0, "yaw": 0.0},
    ],
    "order": ["GA", "GB"],
    "legs": [],
    "total_length": 15.0,
}


class Sim:
    """Kinematic robot + bookkeeping of everything the mission publishes."""

    def __init__(self, logic, x=0.0, y=0.0, yaw=0.0, t0=0.0, frozen=False):
        self.m = logic
        self.x, self.y, self.yaw = x, y, yaw
        self.t = t0
        self.frozen = frozen           # robot ignores commands (stuck test)
        self.targets, self.seek, self.loco, self.mcmd, self.paths, self.events = [], [], [], [], [], []
        self.states = []
        self.last_cmd = None
        self.auto_read = True          # simulated gauge reader
        self.cmd_allowed = True        # mux would pass /cmd_vel/patrol

    def step(self, hook=None):
        self.m.set_pose(self.x, self.y, self.yaw, self.t)
        if hook:
            hook(self)
        if self.auto_read and self.m.base == "INSPECTING" and self.m.current is not None:
            self.m.on_gauge_readings({"stamp": self.t, "readings": [
                {"id": self.m.current.gauge_id, "value": 1.0, "confidence": 0.9, "unit": "bar", "status": "normal"}]},
                self.t)
        out = self.m.step(self.t)
        if out.target is not None:
            self.targets.append(out.target)
        if out.seek_enable is not None:
            self.seek.append(out.seek_enable)
        self.loco += out.locomotion
        self.mcmd += out.mission
        if out.path is not None:
            self.paths.append(out.path)
        self.events += out.events
        st = self.m.reported_state
        if not self.states or self.states[-1] != st:
            self.states.append(st)
        self.last_cmd = out.cmd_vel
        if out.cmd_vel is not None and self.cmd_allowed and not self.frozen:
            vx, vy, wz = out.cmd_vel
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.x += (vx * c - vy * s) * DT
            self.y += (vx * s + vy * c) * DT
            self.yaw = (self.yaw + wz * DT + math.pi) % (2 * math.pi) - math.pi
        self.t += DT
        return out

    def run(self, seconds, hook=None, until=None):
        n = int(round(seconds / DT))
        for _ in range(n):
            self.step(hook)
            if until is not None and until(self):
                return True
        return False


def make(params=None, layout=None, plan=PLAN, grid=True):
    p = {"min_start_time": 0.5}
    p.update(params or {})
    m = MissionLogic(p, layout=layout or FakeLayout())
    if grid:
        m.set_map(room_grid())
    if plan is not None:
        m.set_plan(plan)
    return m


# ----------------------------------------------------------------------------
# helpers under test
# ----------------------------------------------------------------------------
def test_parse_plan_order_and_legs():
    plan = dict(PLAN)
    plan["order"] = ["GB", "GA"]
    plan["legs"] = [[[0, 0], [5, 0], [10.6, 1.0]], [[10.6, 1.0], [6, 3], [4.0, 3.6]]]
    vps = parse_patrol_plan(plan)
    assert [v.gauge_id for v in vps] == ["GB", "GA"]
    assert vps[0].leg[-1] == (10.6, 1.0)
    assert vps[1].leg[0] == (10.6, 1.0)
    assert parse_patrol_plan({"viewpoints": [{"gauge_id": "X"}]}) == []
    assert parse_patrol_plan(None) == []


def test_cross_track_distance():
    path = [(0, 0), (10, 0)]
    assert cross_track_distance(path, 5, 2) == pytest.approx(2.0)
    assert cross_track_distance(path, -3, 4) == pytest.approx(5.0)


def test_fallback_viewpoints_face_gauges_on_real_layout():
    lay = Layout.load()
    grid = rasterize_layout(lay)
    vps = fallback_viewpoints(lay, grid, start=lay.spawn[:2], standoff=1.4, robot_radius=0.35)
    assert len(vps) == len(lay.gauges)
    for vp in vps:
        g = lay.gauge(vp.gauge_id)
        bearing = math.atan2(g["position"][1] - vp.y, g["position"][0] - vp.x)
        assert abs((bearing - vp.yaw + math.pi) % (2 * math.pi) - math.pi) < 1e-6
        assert grid.is_free(vp.x, vp.y), vp
        assert 0.8 <= math.hypot(g["position"][0] - vp.x, g["position"][1] - vp.y) <= 2.3
    # greedy order from the spawn at the corridor end: G08 (x=57) is the nearest gauge
    assert vps[0].gauge_id == "G08"


# ----------------------------------------------------------------------------
# state machine
# ----------------------------------------------------------------------------
def test_full_mission_patrol_inspect_and_rth():
    m = make()
    sim = Sim(m)
    sim.run(1.0)
    assert m.reported_state == "IDLE"
    assert m.command("start", sim.t)
    assert m.base == "PATROL" and m.phase == "starting"
    m.on_locomotion_state({"mode": "STAND"}, sim.t)
    assert sim.run(120, until=lambda s: s.m.base == "RTH")
    assert sim.loco == ["stand"]              # module1 auto-walks on /cmd_vel; no 'walk' by default
    assert m.completed == ["GA", "GB"]
    assert sim.targets == ["GA", "", "GB", ""]
    assert sim.mcmd == ["rth"]
    assert "INSPECTING" in sim.states and "PATROL" in sim.states
    assert m.phase == "requested"
    # return_to_home acknowledges, drives (not simulated) and docks
    m.on_rth_active(True, sim.t)
    sim.run(0.5)
    assert m.phase == "yield"
    m.on_safety_status({"state": "DOCKED", "docked": True, "rth_active": True}, sim.t)
    sim.run(0.5)
    assert m.base == "DOCKED"
    st = m.state_dict(sim.t)
    assert st["state"] == "DOCKED" and st["completed"] == ["GA", "GB"] and st["pending"] == []
    assert m.mode is None


def test_viewpoint_reached_with_alignment():
    m = make()
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")
    assert math.hypot(sim.x - 4.0, sim.y - 3.6) < 0.35
    yaw_err = (sim.yaw - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
    assert abs(yaw_err) <= MISSION_DEFAULTS["align_yaw_tolerance"] + 0.05


def test_patrol_only_ends_done():
    m = make()
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(150, until=lambda s: s.m.base == "DONE")
    assert sim.mcmd == []
    assert sim.paths[-1] == []
    assert m.state_dict(sim.t)["progress"] == [2, 2]


def test_dwell_timeout_scan_then_skip():
    m = make({"scan_time": 1.0})
    sim = Sim(m)
    sim.auto_read = False
    m.command("patrol", 0.0)
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")
    t_in = sim.t
    assert sim.run(30, until=lambda s: s.m.phase == "scan_left")
    assert sim.t - t_in == pytest.approx(MISSION_DEFAULTS["dwell_time"], abs=0.25)
    assert sim.run(10, until=lambda s: s.m.base == "PATROL")
    assert m.missed == ["GA"]
    assert m.results["GA"]["status"] == "missed"


def test_leaves_early_after_enough_readings():
    m = make()
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")
    t_in = sim.t
    assert sim.run(10, until=lambda s: s.m.base == "PATROL")
    # 5 readings arrive at 10 Hz, but min_dwell_time (1 s) is respected
    assert MISSION_DEFAULTS["min_dwell_time"] - 0.15 <= sim.t - t_in <= 2.0
    assert m.completed == ["GA"]
    assert m.results["GA"]["n_readings"] >= MISSION_DEFAULTS["min_readings"]


def test_duplicate_readings_are_ignored():
    m = make()
    sim = Sim(m)
    sim.auto_read = False
    m.command("patrol", 0.0)
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")
    msg = {"stamp": 123.0, "readings": [{"id": "GA", "value": 2.0, "confidence": 1.0}]}
    for _ in range(10):
        m.on_gauge_readings(msg, sim.t)
    assert m.n_readings == 1


def test_gas_alarm_triggers_seek_then_resumes():
    m = make()
    sim = Sim(m)
    m.command("start", 0.0)
    assert sim.run(30, until=lambda s: s.m.phase == "navigate")
    # short spike (0.5 s) must not trigger
    for _ in range(5):
        m.on_gas(40.0, sim.t)
        sim.step()
    m.on_gas(5.0, sim.t)
    sim.run(0.5)
    assert m.base in ("PATROL", "INSPECTING")
    # sustained 30 ppm > 25 for >= 1 s
    gas = {"ppm": 30.0}
    hook = lambda s: s.m.on_gas(gas["ppm"], s.t)  # noqa: E731
    assert sim.run(2.0, hook=hook, until=lambda s: s.m.base == "SEEK")
    assert sim.seek[-1] is True
    assert m.state_dict(sim.t)["gas_alarm"] is True
    # seeker runs, then finds the source
    m.on_seek_status({"state": "SURGE"}, sim.t)
    sim.run(1.0, hook=hook)
    assert m.base == "SEEK"
    m.on_seek_status({"state": "FOUND", "estimate": [3.0, 2.0], "error_m": 0.4, "path_length": 7.0}, sim.t)
    sim.run(0.3, hook=hook)
    assert sim.seek[-1] is False
    assert m.base == "PATROL"
    assert m.seek_result["result"] == "FOUND" and m.seek_result["error_m"] == 0.4
    # gas still high near the source, but the alarm is disarmed after FOUND
    sim.run(3.0, hook=hook)
    assert m.base != "SEEK"
    assert any("GAS SOURCE FOUND" in e[1] for e in sim.events)


def test_seek_timeout_and_unresponsive_seeker():
    m = make({"seek_start_timeout": 2.0, "gas_rearm_cooldown": 1.0})
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(30, until=lambda s: s.m.phase == "navigate")
    assert m.command("seek", sim.t)
    assert m.base == "SEEK"
    assert sim.run(3.0, until=lambda s: s.m.base != "SEEK")
    assert m.seek_result["result"] == "FAILED"
    assert m.base == "PATROL"


def test_manual_override_freezes_and_resumes():
    m = make()
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(30, until=lambda s: s.m.phase == "navigate")
    m.on_teleop_active(True, sim.t)
    out = sim.step()
    assert m.reported_state == "MANUAL"
    x0, y0 = sim.x, sim.y
    heartbeat = lambda s: s.m.on_teleop_active(True, s.t)   # noqa: E731  keyboard_teleop republishes at 2 Hz
    for _ in range(40):
        out = sim.step(heartbeat)
        assert out.cmd_vel is None          # mission does not drive during manual control
        assert m.reported_state == "MANUAL"
    assert (sim.x, sim.y) == (x0, y0)
    # operator moves the robot elsewhere
    sim.x, sim.y = 1.0, -1.0
    m.set_pose(sim.x, sim.y, sim.yaw, sim.t)
    n_paths = len(sim.paths)
    m.on_teleop_active(False, sim.t)
    sim.step()
    assert m.reported_state == "PATROL"
    assert len(sim.paths) == n_paths + 1          # re-planned from the new pose
    assert sim.paths[-1][0] == pytest.approx((1.0, -1.0), abs=0.08)   # A* start = cell centre
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")


def test_manual_override_expires_when_the_teleop_dies():
    """A teleop killed while the override is on never sends false: the
    mission must not stay frozen in MANUAL forever."""
    m = make()
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(30, until=lambda s: s.m.phase == "navigate")
    t_alive = sim.t + 3.0
    hb = lambda s: s.m.on_teleop_active(True, s.t) if s.t <= t_alive else None   # noqa: E731
    sim.run(3.0, hook=hb)
    assert m.reported_state == "MANUAL"
    t_last = m._teleop_t
    assert sim.run(5.0, hook=hb, until=lambda s: s.m.reported_state != "MANUAL")
    assert sim.t - t_last == pytest.approx(MISSION_DEFAULTS["teleop_active_timeout"], abs=2 * DT)
    assert m.reported_state == "PATROL"
    assert any("manual override expired" in e[1] for e in sim.events)
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")       # the mission carries on


def test_manual_override_timeout_survives_sim_reset_and_can_be_disabled():
    m = make()
    m.command("patrol", 0.0)
    m.on_teleop_active(True, 500.0)
    m.step(500.0)
    assert m.reported_state == "MANUAL"
    m.step(0.1)                                    # clock jumped back: window restarts
    assert m.reported_state == "MANUAL"
    m.step(1.5)
    assert m.reported_state == "MANUAL"
    m.step(1.7)
    assert m.reported_state != "MANUAL"
    m2 = make({"teleop_active_timeout": 0.0})      # 0 = latched until 'false'
    m2.on_teleop_active(True, 0.0)
    m2.step(0.0)
    m2.step(100.0)
    assert m2.reported_state == "MANUAL"
    m2.on_teleop_active(False, 100.1)
    m2.step(100.1)
    assert m2.reported_state != "MANUAL"


def test_pause_freezes_dwell_timer():
    m = make()
    sim = Sim(m)
    sim.auto_read = False
    m.command("patrol", 0.0)
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")
    sim.run(2.0)
    assert m.command("pause", sim.t)
    sim.step()
    assert m.reported_state == "PAUSED"
    sim.run(10.0)
    assert m.base == "INSPECTING" and sim.last_cmd == (0.0, 0.0, 0.0)
    assert m.command("resume", sim.t)
    # INSPECTING re-approaches (already there -> align) then dwells again
    assert sim.run(10, until=lambda s: s.m.base == "INSPECTING")
    assert m.reported_state == "INSPECTING"


def test_safety_rth_yield_dock_and_resume():
    m = make()
    sim = Sim(m)
    m.command("start", 0.0)
    assert sim.run(30, until=lambda s: s.m.phase == "navigate")
    m.on_safety_status({"state": "RTH_PLANNING", "rth_active": True, "reason": "battery_low"}, sim.t)
    sim.step()
    assert m.base == "RTH" and m.phase == "yield" and m.rth_reason == "battery_low"
    assert [v.gauge_id for v in m.queue] == ["GA", "GB"]      # current target kept pending
    out = sim.step()
    assert out.cmd_vel is None
    m.on_safety_status({"state": "CHARGING", "rth_active": True, "docked": True}, sim.t)
    sim.step()
    assert m.base == "DOCKED"
    # still charging: resume waits for the release
    assert m.command("resume", sim.t)
    sim.run(1.0)
    assert m.base == "DOCKED"
    m.on_safety_status({"state": "MONITOR", "rth_active": False, "docked": True}, sim.t)
    assert sim.run(5.0, until=lambda s: s.m.base == "PATROL")
    assert sim.run(80, until=lambda s: s.m.base == "RTH")
    assert m.completed == ["GA", "GB"]


def test_self_rth_fallback_when_rth_node_silent():
    m = make({"rth_ack_timeout": 2.0}, layout=FakeLayout(dock=(1.0, 0.0, math.pi)))
    sim = Sim(m, x=6.0, y=6.0)
    m.command("rth", 0.0)
    assert m.base == "RTH" and m.phase == "requested"
    assert sim.run(3.0, until=lambda s: s.m.phase == "self")
    assert sim.run(60, until=lambda s: s.m.base == "DOCKED")
    assert "sit" in sim.loco
    assert math.hypot(sim.x - 1.0, sim.y) < 0.4


def test_stuck_detection_replans_then_skips():
    m = make({"stuck_time": 2.0, "max_replans": 2, "backup_time": 0.5})
    sim = Sim(m, frozen=True)
    m.command("patrol", 0.0)
    assert sim.run(40, until=lambda s: "GA" in s.m.missed)
    assert m.total_replans == 3
    assert any("stuck" in e[1] for e in sim.events)
    assert m.current is not None and m.current.gauge_id == "GB"


def test_locomotion_fallen_holds_and_replans():
    m = make({"send_walk_after_stand": True})
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(30, until=lambda s: s.m.phase == "navigate")
    m.on_locomotion_state({"mode": "FALLEN"}, sim.t)
    sim.step()
    assert m.phase == "hold" and sim.last_cmd == (0.0, 0.0, 0.0)
    sim.run(5.0)
    assert m.phase == "hold"
    n_walk = sim.loco.count("walk")
    m.on_locomotion_state({"mode": "STAND"}, sim.t)
    sim.step()
    assert m.phase == "navigate"
    assert sim.loco.count("walk") == n_walk + 1


def _fall_hook(t_fall, seq=(("FALLEN", 2.0), ("RECOVERING", 6.0), ("STANDING_UP", 1.0)),
               jump=(0.3, -0.2, 1.0)):
    """Locomotion modes of a fall + self-righting starting at t_fall; the body
    lands a little off the path, turned, and ignores commands until STAND."""
    state = {"jumped": False}

    def hook(s):
        el, acc, mode = s.t - t_fall, 0.0, "STAND"
        for name, d in seq:
            if el < acc + d:
                mode = name
                break
            acc += d
        if not state["jumped"]:
            state["jumped"] = True
            s.x, s.y, s.yaw = s.x + jump[0], s.y + jump[1], s.yaw + jump[2]
        s.frozen = mode != "STAND"
        s.auto_read = mode == "STAND"
        s.m.on_locomotion_state({"mode": mode}, s.t)
    return hook


@pytest.mark.parametrize("when", ["navigate", "align", "INSPECTING"])
def test_fall_and_self_righting_in_any_phase_resumes_the_mission(when):
    """Field report: the robot fell once mid-mission, got up - the mission
    has to carry on from wherever the robot stands then."""
    m = make({"scan_time": 1.0})
    sim = Sim(m)
    m.command("patrol", 0.0)
    pred = {"navigate": lambda s: s.m.phase == "navigate" and s.t > 3.0,
            "align": lambda s: s.m.phase == "align",
            "INSPECTING": lambda s: s.m.base == "INSPECTING"}[when]
    assert sim.run(80, until=pred)
    assert sim.run(250, hook=_fall_hook(sim.t), until=lambda s: s.m.base == "DONE")
    assert "GB" in m.completed and len(m.completed) + len(m.missed) == 2
    assert "recover" not in sim.loco


@pytest.mark.parametrize("falls", [1, 2])
def test_fall_during_the_inspection_repeats_it_instead_of_skipping_the_gauge(falls):
    """Gazebo (bug_gz d_inspect): knocked over during the INSPECTING dwell, the
    dwell / scan timers ran out while the robot lay there - 'G07: no reading
    obtained - skipped' 0.4 s after it stood up - and the scan turned a fallen
    robot.  Now the inspection is held like a patrol leg and repeated (re-approach,
    align, dwell) once the robot stands again - also after two falls in a row."""
    m = make({"scan_time": 1.0})
    sim = Sim(m)
    m.command("patrol", 0.0)
    feed = {"mode": "STAND"}
    hook = lambda s: s.m.on_locomotion_state(dict(feed), s.t)                    # noqa: E731
    assert sim.run(80, hook=hook, until=lambda s: s.m.base == "INSPECTING")
    gid = m.current.gauge_id
    sim.auto_read = False                                       # no readings until it stands there again
    sim.run(1.0, hook=hook)
    cmds_down = []
    for k in range(falls):
        sim.frozen = True
        for mode, dur in (("FALLEN", 3.0), ("RECOVERING", 8.0)):
            feed["mode"] = mode
            for _ in range(int(round(dur / DT))):
                sim.step(hook)
                cmds_down.append(sim.last_cmd)
        sim.x, sim.y, sim.yaw = sim.x + 0.4, sim.y - 0.3, sim.yaw + 1.2           # got up off the viewpoint
        sim.frozen = False
        feed["mode"] = "STAND"
        if k + 1 < falls:
            sim.run(1.5, hook=hook)
    assert gid not in m.missed
    assert all(c is None or c == (0.0, 0.0, 0.0) for c in cmds_down)           # no scan turn while down
    sim.auto_read = True
    assert sim.run(150, hook=hook, until=lambda s: s.m.base == "DONE")
    assert gid in m.completed and m.missed == []
    assert any("locomotion not ready" in e[1] for e in sim.events)


def test_hold_reason_says_why_the_robot_stands_still():
    """Field report: up again, the robot stood still while the dashboard said PATROL -
    /mission/state hold_reason now names the cause (e-stop, fallen / getting up, a
    stopped /locomotion/state stream, paused, manual), a new reason while holding is
    announced again (fallen -> ESTOP; FALLEN <-> RECOVERING is one), and a FALLEN that is only the last
    message of a stopped stream sends no 'recover' (bug_loco: a standing robot whose
    state did not arrive re-ran the stand-up every 20 s)."""
    m = make()
    sim = Sim(m)
    m.command("patrol", 0.0)
    feed = {"mode": "STAND", "recovery_state": "NORMAL"}
    hook = lambda s: s.m.on_locomotion_state(dict(feed), s.t) if feed else None   # noqa: E731
    assert sim.run(30, hook=hook, until=lambda s: s.m.phase == "navigate")
    sim.run(1.0, hook=hook)
    assert m.state_dict(sim.t)["hold_reason"] == ""
    feed.update(mode="FALLEN", recovery_state="DAMP")
    sim.run(1.0, hook=hook)
    assert m.phase == "hold" and m.state_dict(sim.t)["hold_reason"] == "robot fallen (DAMP)"
    feed.update(mode="RECOVERING", recovery_state="STAND_UP")
    sim.run(1.0, hook=hook)
    assert m.hold_reason() == "getting up (STAND_UP)"
    feed.update(mode="ESTOP")                                    # SPACE / dashboard e-stop during the recovery
    sim.run(1.0, hook=hook)
    assert m.hold_reason().startswith("e-stop engaged") and "release" in m.hold_reason()
    held = [e[1] for e in sim.events if e[1].startswith("locomotion not ready")]
    assert held == ["locomotion not ready (FALLEN) - holding", "locomotion not ready (ESTOP) - holding"]
    assert m.message == "locomotion not ready (ESTOP) - holding"
    # the /locomotion/state stream stops with a FALLEN (controller died / stopped)
    feed.update(mode="FALLEN", recovery_state="FAILED")
    sim.run(0.5, hook=hook)
    feed.clear()
    sim.run(30.0, hook=hook)
    assert m.hold_reason().startswith("no /locomotion/state for ") and "locomotion_controller" in m.hold_reason()
    assert sum("no /locomotion/state" in e[1] for e in sim.events) == 1
    assert "recover" not in sim.loco and m.phase == "hold"
    # back: ready again, driving, nothing to report
    feed.update(mode="STAND", recovery_state="NORMAL")
    assert sim.run(5.0, hook=hook, until=lambda s: s.m.phase == "navigate")
    assert any(e[1] == "/locomotion/state received again" for e in sim.events)
    sim.run(1.0, hook=hook)
    assert m.hold_reason() == ""
    m.command("pause", sim.t)
    assert m.state_dict(sim.t)["hold_reason"] == "paused"
    m.command("resume", sim.t)
    m.on_teleop_active(True, sim.t)
    assert m.hold_reason() == "manual override (teleop)"
    m.on_teleop_active(False, sim.t)
    m.set_pose(sim.x, sim.y, sim.yaw, sim.t - 5.0)               # /odom stale
    assert m.hold_reason(sim.t) == "waiting for /odom"
    # a one-off state (tests, a single message) is no stream: never reported missing
    m2 = make()
    m2.on_locomotion_state({"mode": "STAND"}, 0.0)
    m2.command("patrol", 0.0)
    sim2 = Sim(m2)
    sim2.run(20.0)
    assert m2.hold_reason() in ("", "obstacle ahead") and not any("no /locomotion" in e[1] for e in sim2.events)


def test_teleop_priority_on_the_mux_is_reported_and_not_counted_as_stuck():
    """Review of fix/resume (stub stack r10): a teleop E-STOP left engaged after the
    controller was released elsewhere keeps the teleop streaming zero velocity - the mux
    selects 'teleop', the robot stands, the mission said 'PATROL · navigate' with an
    empty hold_reason and its stuck logic skipped one gauge after the other.  Now
    /cmd_vel_mux/active 'teleop' while the mission drives is the hold_reason, warned
    once, and not counted as stuck; back on 'patrol' the mission drives on."""
    m = make()
    sim = Sim(m)
    m.command("patrol", 0.0)
    feed = {"mode": "STAND", "recovery_state": "NORMAL"}
    mux = {"src": "patrol", "n": 0}

    def hook(s):
        s.m.on_locomotion_state(dict(feed), s.t)
        mux["n"] += 1
        if mux["n"] % 5 == 0:                                    # /cmd_vel_mux/active at 2 Hz
            s.m.on_mux_active(mux["src"], s.t)
    assert sim.run(30, hook=hook, until=lambda s: s.m.phase == "navigate")
    sim.run(1.0, hook=hook)
    assert m.hold_reason() == ""
    mux["src"] = "teleop"                                        # zero stream of a stuck teleop E-STOP
    sim.cmd_allowed = False                                      # the mux passes the teleop's zeros
    sim.run(60.0, hook=hook)
    assert m.base == "PATROL" and m.phase == "navigate" and m.missed == []
    assert m.state_dict(sim.t)["hold_reason"].startswith("teleop has priority")
    assert not any("stuck" in e[1] for e in sim.events)
    assert sum("keeps 'teleop' selected" in e[1] for e in sim.events) == 1
    m.mux_t = sim.t - 10.0                                       # a stale mux report (mux gone): not trusted
    assert not m.hold_reason(sim.t).startswith("teleop")
    mux["src"] = "patrol"
    sim.cmd_allowed = True
    x0, y0 = sim.x, sim.y
    sim.run(10.0, hook=hook)
    assert math.hypot(sim.x - x0, sim.y - y0) > 1.0 and m.hold_reason() == ""
    assert sum("no longer selects the teleop" in e[1] for e in sim.events) == 1


def test_robot_staying_down_gets_recover_requests():
    m = make()
    sim = Sim(m)
    m.on_locomotion_state({"mode": "FALLEN", "recovery_state": "FAILED"}, 0.0)
    sim.run(30.0)                                                # no mission: nothing to do
    assert "recover" not in sim.loco
    m.on_locomotion_state({"mode": "STAND"}, sim.t)
    m.command("patrol", sim.t)
    assert sim.run(30, until=lambda s: s.m.phase == "navigate")

    def recovering(s):                                           # module1 is working on it: leave it alone
        s.frozen = True
        s.m.on_locomotion_state({"mode": "FALLEN" if int(s.t) % 6 < 2 else "RECOVERING"}, s.t)
    sim.run(40.0, hook=recovering)
    assert "recover" not in sim.loco and m.phase == "hold"

    def down(s):                                                 # recovery gave up (FAILED)
        s.m.on_locomotion_state({"mode": "FALLEN", "recovery_state": "FAILED"}, s.t)
    sim.run(45.0, hook=down)
    assert sim.loco.count("recover") == 2
    assert any("sending 'recover'" in text and "FAILED" in text for _lvl, text in sim.events)
    # up again: the mission continues
    sim.frozen = False
    m.on_locomotion_state({"mode": "STAND"}, sim.t)
    assert sim.run(150, until=lambda s: s.m.base == "DONE")
    m2 = make({"recover_after": 0.0})
    sim2 = Sim(m2)
    m2.command("patrol", 0.0)
    assert sim2.run(30, until=lambda s: s.m.phase == "navigate")
    sim2.run(45.0, hook=down)
    assert "recover" not in sim2.loco


def test_abort_resets_everything():
    m = make()
    sim = Sim(m)
    m.command("start", 0.0)
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")
    m.command("abort", sim.t)
    out = sim.step()
    assert m.base == "IDLE" and m.queue == [] and m.current is None
    assert sim.targets[-1] == "" and sim.paths[-1] == []


def test_start_ignored_while_running_and_unknown_command():
    m = make()
    m.command("start", 0.0)
    assert not m.command("patrol", 0.1)
    assert not m.command("fly", 0.2)


def test_start_waits_for_rth_release():
    m = make({"rth_release_grace": 0.5})
    sim = Sim(m)
    m.on_rth_active(True, 0.0)
    sim.step()
    assert m.base == "RTH"
    assert m.command("start", sim.t)
    sim.run(2.0)
    assert m.base == "RTH"
    m.on_rth_active(False, sim.t)
    assert sim.run(2.0, until=lambda s: s.m.base == "PATROL")


def test_autostart_and_layout_fallback_plan():
    m = make({"autostart": True, "autostart_delay": 2.0, "plan_wait_timeout": 1.0}, plan=None)
    sim = Sim(m)
    assert sim.run(2.5) is False
    assert m.base == "IDLE"                      # waiting for a plan
    assert sim.run(2.0, until=lambda s: s.m.base == "PATROL")
    assert m.plan_source == "layout_fallback"
    assert m.mode == "full"
    assert [v.gauge_id for v in m.queue] == ["GA", "GB"]


def test_no_map_drives_straight_lines():
    m = make(grid=False)
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")
    assert any("no /map" in e[1] for e in sim.events)


def test_start_on_dock_releases_charging_rth():
    m = make({"rth_release_grace": 0.5})
    sim = Sim(m)
    m.on_safety_status({"state": "CHARGING", "rth_active": True, "docked": True, "battery_pct": 80.0}, 0.0)
    sim.step()
    assert m.base == "DOCKED"
    assert m.command("start", sim.t)
    out_cmds = list(sim.mcmd)
    sim.step()
    assert sim.mcmd == out_cmds + ["resume"]
    assert m.command("resume", sim.t)                       # our own echo is harmless
    assert m._start_pending == "start"
    m.on_safety_status({"state": "MONITOR", "rth_active": False, "docked": True, "battery_pct": 80.0}, sim.t)
    assert sim.run(3.0, until=lambda s: s.m.base == "PATROL")


def test_start_on_dock_with_low_battery_keeps_charging():
    m = make()
    sim = Sim(m)
    m.on_safety_status({"state": "CHARGING", "rth_active": True, "docked": True, "battery_pct": 0.3}, 0.0)
    sim.step()
    assert m.command("start", sim.t)
    sim.run(1.0)
    assert "resume" not in sim.mcmd and m.base == "DOCKED"
    assert "30 %" in m.message


def test_rth_release_near_dock_counts_as_docked():
    m = make({"rth_release_grace": 0.5}, layout=FakeLayout(dock=(1.0, 0.0, 0.0)))
    sim = Sim(m, x=6.0, y=1.0)
    m.command("rth", 0.0)
    m.on_rth_active(True, 0.0)
    sim.run(0.5)
    assert m.phase == "yield"
    sim.x, sim.y = 1.3, 0.2                                  # return_to_home brought it home
    m.on_rth_active(False, sim.t)                            # ... and released before we saw DOCKED
    assert sim.run(2.0, until=lambda s: s.m.base == "DOCKED")


def test_republished_plan_with_new_stamp_is_not_new():
    m = make(plan=None)
    p1 = dict(PLAN, stamp=1.0, compute_s=0.5)
    assert m.set_plan(p1)
    assert not m.set_plan(dict(p1, stamp=2.0, compute_s=0.7, reason="timer"))
    p2 = dict(p1, order=["GB", "GA"])
    assert m.set_plan(p2)
    assert [v.gauge_id for v in m.viewpoints] == ["GB", "GA"]


def test_module2_plan_json_format_is_understood():
    """Payload shape of module2_inspection PatrolPlan.to_json (legs incl. the leg back to the dock)."""
    d = {"start": [61.9, 0.0], "start_yaw": 3.1416,
         "viewpoints": [{"gauge_id": "GA", "x": 4.0, "y": 3.6, "yaw": 1.5708, "score": 0.9, "distance": 1.4,
                         "incidence_deg": 3.0, "pixel_diameter": 180.0, "order": 1, "leg_length": 5.4, "tag": "T"},
                        {"gauge_id": "GB", "x": 10.6, "y": 1.0, "yaw": 0.0, "order": 2}],
         "order": ["GA", "GB"],
         "legs": [[[0.0, 0.0], [2.0, 2.0], [4.0, 3.6]], [[4.0, 3.6], [8.0, 5.0], [10.6, 1.0]],
                  [[10.6, 1.0], [0.0, 0.0]]],
         "leg_lengths": [5.4, 9.0, 10.6], "total_length": 25.0, "return_to_dock": True, "unreachable": [],
         "method": "held_karp", "stamp": 12.5}
    vps = parse_patrol_plan(d)
    assert [v.gauge_id for v in vps] == ["GA", "GB"]
    assert vps[0].leg[0] == (0.0, 0.0) and vps[1].leg[-1] == (10.6, 1.0)
    m = make(plan=None)
    assert m.set_plan(d)
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(120, until=lambda s: s.m.base == "DONE")
    assert m.completed == ["GA", "GB"]


def test_stale_safety_status_does_not_fake_docking():
    m = make({"rth_ack_timeout": 100.0}, layout=FakeLayout(dock=(15.0, 5.0, 0.0)))
    sim = Sim(m)
    m.on_safety_status({"state": "CHARGING", "docked": True, "rth_active": False}, 0.0)   # old message
    sim.run(5.0)
    m.command("rth", sim.t)
    sim.run(1.0)
    assert m.base == "RTH"                                   # not DOCKED: the status is 6 s old
    m.on_safety_status({"state": "DOCKED", "docked": True, "rth_active": True}, sim.t)
    sim.run(0.3)
    assert m.base == "DOCKED"


def test_mission_elapsed_freezes_when_done():
    m = make()
    sim = Sim(m)
    m.command("patrol", 0.0)
    assert sim.run(150, until=lambda s: s.m.base == "DONE")
    e1 = m.state_dict(sim.t)["mission_elapsed"]
    sim.run(5.0)
    assert m.state_dict(sim.t)["mission_elapsed"] == e1


# ----------------------------------------------------------------------------
# odometry gaps (review fixes)
# ----------------------------------------------------------------------------
def _step_blind(m, t0, seconds, events=None):
    """Advance the mission without any /odom update; returns (last output, t)."""
    out, t = None, t0
    for _ in range(int(round(seconds / DT))):
        out = m.step(t)
        if events is not None:
            events += out.events
        t += DT
    return out, t


def test_start_before_odometry_holds_instead_of_skipping_viewpoints():
    m = make()
    assert m.command("start", 0.0)
    m.on_locomotion_state({"mode": "STAND"}, 0.0)
    events = []
    out, t = _step_blind(m, 0.0, 10.0, events)    # no pose at all for 10 s
    assert m.missed == [] and m.completed == []
    assert m.base == "PATROL" and m.phase == "hold"
    assert m.current is not None and m.current.gauge_id == "GA"
    assert out.cmd_vel == (0.0, 0.0, 0.0)
    assert sum("waiting for /odom" in e[1] for e in events) == 1    # warned once, not every tick
    sim = Sim(m, t0=t)                             # odometry arrives
    assert sim.run(60, until=lambda s: s.m.base == "INSPECTING")
    assert m.current.gauge_id == "GA" and m.missed == []


def test_self_rth_waits_for_odometry_and_holds_on_stale_pose():
    m = make({"rth_ack_timeout": 2.0}, layout=FakeLayout(dock=(1.0, 0.0, math.pi)))
    m.command("rth", 0.0)
    _, t = _step_blind(m, 0.0, 4.0)               # ack timeout passes without any pose
    assert m.base == "RTH" and m.phase == "self_wait"
    sim = Sim(m, x=6.0, y=6.0, t0=t)
    assert sim.run(2.0, until=lambda s: s.m.phase == "self")
    sim.run(2.0)
    # odometry stops while driving home: the mission must stop commanding motion
    out, t = _step_blind(m, sim.t, 3.0)
    assert m.phase == "self" and out.cmd_vel == (0.0, 0.0, 0.0)
    sim.t = t
    assert sim.run(60, until=lambda s: s.m.base == "DOCKED")
    assert math.hypot(sim.x - 1.0, sim.y) < 0.4


def test_default_planner_uses_the_terrain_cost():
    """mission_manager's own A* (no patrol leg) goes around the C-OBS1 north end and through
    the C-OBS3 north passage; terrain_cost_enabled: false crosses them."""
    import os
    from dusan_common import terrain_cost as tc
    lay = Layout.load()
    if not os.path.exists(lay.nav_map_yaml()):
        pytest.skip("simulation/maps/plant_map.yaml not generated")
    grid = OccupancyGridMap.load(lay.nav_map_yaml())
    for k, v in tc.TERRAIN_COST_DEFAULTS.items():
        assert MISSION_DEFAULTS[k] == v
    paths = {}
    for enabled in (True, False):
        m = MissionLogic({"terrain_cost_enabled": enabled}, layout=lay)
        m.set_map(grid)
        m.set_pose(61.9, 0.0, math.pi, 0.0)
        a = m._plan((52.5, 0.8))
        m.set_pose(44.0, 0.0, math.pi, 1.0)
        b = m._plan((34.0, 0.0))
        paths[enabled] = (a, b)
    assert tc.debris_crossed(paths[True][0], lay) == [] and tc.debris_crossed(paths[True][1], lay) == []
    assert tc.debris_crossed(paths[False][0], lay) == ["C-OBS1"]
    assert tc.debris_crossed(paths[False][1], lay) == ["C-OBS3"]
    # a custom planner keeps the (grid, start, goal, radius) signature and bypasses the layer
    calls = []
    m = MissionLogic({}, layout=lay, planner=lambda g, s, e, r: calls.append(r) or [s, e])
    m.set_map(grid)
    m.set_pose(44.0, 0.0, math.pi, 0.0)
    assert m._plan((34.0, 0.0)) and calls == [MISSION_DEFAULTS["robot_radius"]]


@pytest.mark.skipif(not os.path.exists(PLANT_MAP), reason="simulation/maps/plant_map.yaml not generated")
def test_pipe_crossing_planned_straight_and_walked_steadily():
    """C-OBS5 (0.08 m pipe across the corridor, 20 deg from square): the A* route
    zig-zagged right at the pipe and the robot turned with its feet next to it
    (Gazebo mission check: 4 of 5 runs fell there).  The mission plans the
    crossing straight and square to the pipe, turns onto it before the feet
    reach the pipe and walks over at crossing_speed without strafing / turning."""
    from dusan_common.debris_crossing import find_crossings
    lay = Layout.load()
    grid = OccupancyGridMap.load(PLANT_MAP)
    plan = {"viewpoints": [{"gauge_id": "P1", "x": 19.8, "y": 0.3, "yaw": math.pi}], "order": ["P1"], "legs": []}
    m = MissionLogic({"min_start_time": 0.5}, layout=lay)
    m.set_map(grid)
    m.set_plan(plan)
    sim = Sim(m, x=24.2, y=0.5, yaw=math.pi)
    m.command("patrol", 0.0)
    m.on_locomotion_state({"mode": "STAND"}, 0.0)
    states, cmds = [], []

    def hook(s):
        states.append(m.crossing.state)
        if m.crossing.state == "cross" and s.last_cmd is not None:
            cmds.append(s.last_cmd)

    assert sim.run(60, hook=hook, until=lambda s: s.m.base in ("INSPECTING", "DONE"))
    path = sim.paths[0]
    crs = find_crossings(path, m.crossing.pipes)
    assert [c.pipe.id for c in crs] == ["C-OBS5"]
    c = crs[0]
    P = np.asarray(path)
    seg = P[(np.array([c.progress(*p) for p in P]) > -0.85) & (np.array([c.progress(*p) for p in P]) < 0.75)]
    d = np.diff(seg, axis=0)
    hd = np.arctan2(d[:, 1], d[:, 0])
    assert np.all(np.abs((hd - c.heading + math.pi) % (2 * math.pi) - math.pi) < 1e-6)   # straight, square
    assert "cross" in states and m.crossing.n_crossings == 1
    assert cmds and all(v[0] == pytest.approx(MISSION_DEFAULTS["crossing_speed"]) and v[1] == 0.0
                        and abs(v[2]) <= MISSION_DEFAULTS["crossing_wz"] + 1e-9 for v in cmds)


@pytest.mark.skipif(not os.path.exists(PLANT_MAP), reason="simulation/maps/plant_map.yaml not generated")
@pytest.mark.parametrize("turn", [80.0, -80.0, 28.0])
def test_fall_on_the_pipe_crossing_turns_back_onto_it_before_walking(turn):
    """Field report 'fell twice in a row': down on the C-OBS5 crossing, up again turned
    along the pipe (SLIDE).  The crossing kept from before the fall walked straight on
    at 0.25 m/s with no turn - along the pipe (Gazebo: fell on it again, 4 of 4 runs) or
    against the corridor wall (stuck, viewpoints skipped), the dashboard showing PATROL.
    Now the robot turns onto the crossing first and walks over it."""
    lay = Layout.load()
    grid = OccupancyGridMap.load(PLANT_MAP)
    plan = {"viewpoints": [{"gauge_id": "P1", "x": 19.8, "y": 0.3, "yaw": math.pi}], "order": ["P1"], "legs": []}
    m = MissionLogic({"min_start_time": 0.5}, layout=lay)
    m.set_map(grid)
    m.set_plan(plan)
    sim = Sim(m, x=24.2, y=0.5, yaw=math.pi)
    m.command("patrol", 0.0)
    feed = {"mode": "STAND"}
    hook = lambda s: s.m.on_locomotion_state(dict(feed), s.t)                    # noqa: E731
    assert sim.run(60, hook=hook, until=lambda s: s.m.crossing.state == "cross" and s.x < 22.45)
    cr = m.crossing.active
    feed["mode"] = "FALLEN"
    sim.run(3.0, hook=hook)
    feed["mode"] = "RECOVERING"
    sim.run(4.0, hook=hook)
    sim.yaw += math.radians(turn)                                                # stood up turned
    feed["mode"] = "STAND"
    x0, y0, a0 = sim.x, sim.y, cr.pipe.along(sim.x, sim.y)
    trace = []

    def watch(s):
        hook(s)
        trace.append((s.x, s.y, s.yaw, s.last_cmd, s.m.crossing.state))
    assert sim.run(80, hook=watch, until=lambda s: s.m.base in ("INSPECTING", "DONE"))
    zone = MISSION_DEFAULTS["crossing_zone"]
    along = max(abs(cr.pipe.along(x, y) - a0) for x, y, *_ in trace if abs(cr.pipe.side(x, y)) <= zone)
    assert along < 0.35, along                                 # walked along the pipe while over it?
    # no forward step while turned away from the crossing (> align_tol)
    for x, y, yaw, cmd, st in trace:
        if cmd is not None and cmd[0] > 0.0 and abs(cr.pipe.side(x, y)) <= MISSION_DEFAULTS["crossing_zone"]:
            err = abs((cr.heading - yaw + math.pi) % (2 * math.pi) - math.pi)
            assert err <= MISSION_DEFAULTS["crossing_realign_tol"] + 0.05, (x, y, math.degrees(err), st)
    assert m.missed == [] and not any("stuck" in e[1] for e in sim.events)
    assert any("locomotion ready again" in e[1] for e in sim.events)


def test_respawned_mission_manager_does_not_autostart():
    # sim time is global: a mission_manager respawned at t=900 s (e.g. after docking) must stay IDLE
    m = make({"autostart": True, "autostart_delay": 2.0, "plan_wait_timeout": 1.0, "autostart_window": 120.0})
    sim = Sim(m, t0=900.0)
    assert sim.run(40.0, until=lambda s: s.m.base != "IDLE") is False
    assert m.base == "IDLE"
    assert any("autostart skipped" in e[1] for e in sim.events)
    m.command("start", sim.t)                                   # 't' still works
    assert sim.run(2.0, until=lambda s: s.m.base == "PATROL")


def test_autostart_window_can_be_disabled():
    m = make({"autostart": True, "autostart_delay": 2.0, "plan_wait_timeout": 1.0, "autostart_window": 0.0})
    sim = Sim(m, t0=900.0)
    assert sim.run(5.0, until=lambda s: s.m.base == "PATROL")
