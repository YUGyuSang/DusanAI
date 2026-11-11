"""Cross-package contract tests of the integration layer (ROS free).

* configuration contracts: bridge topics used by odom_tf, world / robot
  names, spawn pose;
* payload contracts: the REAL module2 PatrolPlan.to_json() output is
  understood by the mission manager;
* behaviour contract: a closed-loop co-simulation of the REAL mission state
  machine, the REAL cmd_vel mux and the REAL module3 return-to-home logic
  (RthLogic) driving one kinematic robot - full mission -> 'rth' -> docking,
  battery-low RTH in the middle of a patrol -> charging -> 'resume', and a
  'start' on the charger that releases return_to_home with 'resume'.
"""
import math
import os
import re

import numpy as np
import pytest
import yaml

from dusan_common import robot_model
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout

from plant_simulation.mission_logic import MissionLogic, parse_patrol_plan
from plant_simulation.mux_logic import CmdVelMux
from plant_simulation.odom_logic import ODOM_DEFAULTS

HERE = os.path.dirname(os.path.realpath(__file__))
SIM = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(SIM, ".."))


# ----------------------------------------------------------------------------
# configuration contracts
# ----------------------------------------------------------------------------
def _bridge_entries():
    path = os.path.join(SIM, "config", "ros_gz_bridge.yaml")
    if not os.path.isfile(path):
        pytest.skip("ros_gz_bridge.yaml not written yet")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, list), "parameter_bridge config_file must be a YAML list"
    return {e["ros_topic_name"]: e for e in data}


def test_bridge_provides_the_integration_inputs():
    br = _bridge_entries()
    need = {
        "/clock": ("rosgraph_msgs/msg/Clock", "GZ_TO_ROS"),
        ODOM_DEFAULTS["input_topic"]: ("nav_msgs/msg/Odometry", "GZ_TO_ROS"),       # odom_tf input
        "/joint_states": ("sensor_msgs/msg/JointState", "GZ_TO_ROS"),               # robot_state_publisher
    }
    for topic, (rtype, direction) in need.items():
        assert topic in br, f"bridge does not provide {topic}"
        assert br[topic]["ros_type_name"] == rtype
        assert br[topic]["direction"] == direction
    assert br[ODOM_DEFAULTS["input_topic"]]["gz_topic_name"] == robot_model.GZ_ODOM_TOPIC
    assert br["/joint_states"]["gz_topic_name"] == robot_model.GZ_JOINT_STATE_TOPIC


def test_world_robot_names_and_spawn_agree():
    lay = Layout.load()
    assert lay["world_name"] == robot_model.WORLD_NAME
    assert lay["robot_name"] == robot_model.ROBOT_NAME
    world = os.path.join(SIM, "worlds", "plant.sdf")
    if os.path.isfile(world):
        head = open(world, encoding="utf-8").read(4000)
        m = re.search(r'<world\s+name="([^"]+)"', head)
        assert m and m.group(1) == lay["world_name"], "plant.sdf world name differs from world_layout.yaml"
    # sim.launch.py falls back to DEFAULT_SPAWN when the layout cannot be read
    src = open(os.path.join(SIM, "launch", "sim.launch.py"), encoding="utf-8").read()
    m = re.search(r"DEFAULT_SPAWN\s*=\s*\[([^\]]+)\]", src)
    assert m
    default = [float(v) for v in m.group(1).split(",")]
    assert default == pytest.approx(list(lay.spawn), abs=1e-6)
    # spawn on the charging dock, legs straight: z = floor + 0.56
    dx, dy, dyaw = lay.dock_pose
    sx, sy, sz, syaw = lay.spawn
    assert math.hypot(sx - dx, sy - dy) < 0.05 and abs(syaw - dyaw) < 1e-3
    assert sz == pytest.approx(lay.floor_z(sx, sy) + 0.56, abs=0.02)


# ----------------------------------------------------------------------------
# payload contract with module2 (real PatrolPlan)
# ----------------------------------------------------------------------------
def test_real_module2_patrol_plan_payload_is_understood():
    vp_mod = pytest.importorskip("module2_inspection.viewpoint_planner")
    vps = [vp_mod.Viewpoint("GA", 4.0, 3.6, math.pi / 2, score=0.9, valid=True),
           vp_mod.Viewpoint("GB", 10.6, 1.0, 0.0, score=0.8, valid=True)]
    legs = [[(0.0, 0.0), (2.0, 2.0), (4.0, 3.6)], [(4.0, 3.6), (7.0, 5.0), (10.6, 1.0)], [(10.6, 1.0), (0.0, 0.0)]]
    plan = vp_mod.PatrolPlan(start=(0.0, 0.0, 0.0), viewpoints=vps, order=["GA", "GB"], legs=legs,
                             leg_lengths=[5.0, 8.0, 10.6], total_length=23.6, return_to_dock=True,
                             naive_order=["GA", "GB"], naive_length=23.6, unreachable=[],
                             distance_matrix=np.zeros((3, 3)), node_ids=["start", "GA", "GB"])
    d = plan.to_json({"GA": "TAG-A", "GB": "TAG-B"})
    import json
    d = json.loads(json.dumps(d))                      # exactly what travels over /patrol/plan_json
    parsed = parse_patrol_plan(d)
    assert [v.gauge_id for v in parsed] == ["GA", "GB"]
    assert parsed[0].yaw == pytest.approx(math.pi / 2, abs=1e-3)
    assert parsed[0].leg[-1] == (4.0, 3.6) and parsed[1].leg[-1] == (10.6, 1.0)
    m = MissionLogic({}, layout=None)
    assert m.set_plan(d) and m.plan_source == "patrol_planner"


# ----------------------------------------------------------------------------
# closed-loop co-simulation with the real module3 RTH logic
# ----------------------------------------------------------------------------
DT = 0.1


def room_grid():
    """20 m x 10 m room (origin -2,-2) with a partial wall at x = 8 (gap above y = 4)."""
    occ = np.zeros((100, 200), np.int8)
    occ[:, 0] = occ[:, -1] = occ[0, :] = occ[-1, :] = 100
    occ[0:60, 100] = 100
    return OccupancyGridMap(occ, 0.1, -2.0, -2.0)


class RoomLayout:
    dock_pose = (0.0, 0.0, 0.0)
    gauges = []


PLAN = {"start": [0.0, 0.0],
        "viewpoints": [{"gauge_id": "GA", "x": 4.0, "y": 3.6, "yaw": math.pi / 2},
                       {"gauge_id": "GB", "x": 10.6, "y": 1.0, "yaw": 0.0}],
        "order": ["GA", "GB"], "legs": [], "total_length": 15.0}


class CoSim:
    """mission_manager + cmd_vel_mux + return_to_home around one kinematic robot.

    Mirrors the ROS wiring: /mission/command reaches both the mission (echo
    included) and return_to_home; /safety/status + /safety/rth_active go to
    the mission and the mux; the mux output drives the robot.  The battery
    drains while walking and charges on the dock (battery_simulator)."""

    def __init__(self):
        rth_mod = pytest.importorskip("module3_gas_safety.rth_logic")
        self.grid = room_grid()
        self.m = MissionLogic({"min_start_time": 0.5}, layout=RoomLayout())
        self.m.set_map(self.grid)
        self.m.set_plan(PLAN)
        self.rth = rth_mod.RthLogic(rth_mod.RthConfig(), self.grid, RoomLayout.dock_pose)
        self.mux = CmdVelMux()
        self.x = self.y = self.yaw = 0.0
        self.vxw = self.vyw = 0.0
        self.t = 0.0
        self.battery = 100.0
        self.charge_rate = 2.0            # %/s on the dock
        self.pending_cmds = []            # /mission/command samples not yet seen by return_to_home
        self.mission_cmds = []            # everything the mission published on /mission/command
        self.loco_cmds = []
        self.sources = []
        self.mission_states = []

    def command(self, c):
        self.m.command(c, self.t)
        self.pending_cmds.append(c)

    def dist_dock(self):
        return math.hypot(self.x - RoomLayout.dock_pose[0], self.y - RoomLayout.dock_pose[1])

    def step(self):
        charging = self.dist_dock() < 0.35
        moving = math.hypot(self.vxw, self.vyw) > 0.02
        self.battery = float(np.clip(self.battery + (self.charge_rate if charging else (-0.02 if moving else -0.005))
                                     * DT, 0.0, 100.0))
        # ---- return_to_home (one command per tick, like the node's latest-command slot)
        cmd = self.pending_cmds.pop(0) if self.pending_cmds else None
        rin = self.rth_inputs(cmd, charging)
        rout = self.rth.update(rin)
        if rout.locomotion:
            self.loco_cmds.append(("rth", rout.locomotion))
        # ---- mission
        self.m.set_pose(self.x, self.y, self.yaw, self.t)
        self.m.on_safety_status(rout.status, self.t)
        self.m.on_rth_active(rout.rth_active, self.t)
        if self.m.base == "INSPECTING" and self.m.current is not None:
            self.m.on_gauge_readings({"stamp": self.t, "readings": [
                {"id": self.m.current.gauge_id, "value": 1.0, "confidence": 0.9, "unit": "bar"}]}, self.t)
        mout = self.m.step(self.t)
        self.loco_cmds += [("mission", c) for c in mout.locomotion]
        for c in mout.mission:
            self.mission_cmds.append(c)
            self.command(c)                       # echo to the mission + delivery to return_to_home
        st = self.m.reported_state
        if not self.mission_states or self.mission_states[-1] != st:
            self.mission_states.append(st)
        # ---- mux
        if mout.cmd_vel is not None:
            self.mux.update("patrol", mout.cmd_vel, self.t)
        if rout.cmd is not None:
            self.mux.update("rth", rout.cmd, self.t)
        self.mux.set_flag("rth_active", rout.rth_active, self.t)
        src, (vx, vy, wz) = self.mux.select(self.t)
        if not self.sources or self.sources[-1] != src:
            self.sources.append(src)
        # ---- kinematic robot
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.vxw, self.vyw = vx * c - vy * s, vx * s + vy * c
        self.x += self.vxw * DT
        self.y += self.vyw * DT
        self.yaw = (self.yaw + wz * DT + math.pi) % (2 * math.pi) - math.pi
        assert self.grid.is_free(self.x, self.y), f"robot drove into an obstacle at ({self.x:.2f}, {self.y:.2f})"
        self.t += DT

    def rth_inputs(self, cmd, charging):
        import module3_gas_safety.rth_logic as rth_mod
        return rth_mod.RthInputs(t=self.t, x=self.x, y=self.y, yaw=self.yaw, vx=self.vxw, vy=self.vyw,
                                 battery_pct=self.battery, charging=charging, rssi_dbm=-50.0,
                                 plan=list(self.m.path) if self.m.path else None, command=cmd)

    def run(self, seconds, until=None, hook=None):
        for _ in range(int(round(seconds / DT))):
            if hook:
                hook(self)
            self.step()
            if until is not None and until(self):
                return True
        return False


def test_cosim_full_mission_ends_docked_through_return_to_home():
    sim = CoSim()
    sim.command("start")
    ok = sim.run(400, until=lambda s: s.m.base == "DOCKED" and s.rth.state in ("DOCKED", "CHARGING"))
    assert ok, (sim.mission_states, sim.rth.state, sim.m.message, sim.rth.message)
    assert sim.m.completed == ["GA", "GB"] and not sim.m.missed
    assert sim.mission_cmds == ["rth"]                       # the full mission ends with 'rth'
    assert sim.rth.reason == "manual"
    assert ("mission", "stand") in sim.loco_cmds and ("rth", "sit") in sim.loco_cmds
    assert sim.dist_dock() < 0.2
    assert "patrol" in sim.sources and "rth" in sim.sources  # the mux handed over to return_to_home
    assert sim.m.mode is None and sim.m.state_dict(sim.t)["state"] == "DOCKED"
    for st in ("PATROL", "INSPECTING", "RTH", "DOCKED"):
        assert st in sim.mission_states


def test_cosim_battery_low_yields_charges_and_resumes():
    sim = CoSim()
    sim.command("start")
    # walk until GA is inspected and the robot heads for GB, then drain the battery
    assert sim.run(200, until=lambda s: "GA" in s.m.completed and s.m.base == "PATROL")
    sim.battery = 19.0
    ok = sim.run(300, until=lambda s: s.m.base == "DOCKED" and s.rth.state in ("DOCKED", "CHARGING"))
    assert ok, (sim.mission_states, sim.rth.state, sim.rth.message)
    assert sim.m.rth_reason == "battery_low" and sim.rth.reason in ("battery_low", None)
    assert [v.gauge_id for v in sim.m.queue] == ["GB"]       # interrupted viewpoint kept pending
    assert sim.m.mode == "full"
    # charging to resume_battery_pct releases return_to_home ('stand'), the mission waits on the dock
    assert sim.run(120, until=lambda s: s.rth.state == "MONITOR")
    sim.run(3.0)
    assert sim.m.base == "DOCKED"
    sim.command("resume")
    ok = sim.run(400, until=lambda s: s.m.base == "DOCKED" and s.m.mode is None)
    assert ok, (sim.mission_states, sim.rth.state, sim.m.message)
    assert sim.m.completed == ["GA", "GB"]
    assert sim.mission_cmds.count("rth") == 1                 # only the final return home was requested


def test_cosim_start_on_the_charger_releases_return_to_home():
    sim = CoSim()
    sim.command("rth")                                        # robot already on the dock -> dock + charge
    assert sim.run(60, until=lambda s: s.rth.state == "CHARGING")
    assert sim.m.base == "DOCKED"
    sim.battery, sim.charge_rate = 70.0, 0.0                  # charged enough to work, not to 95 %
    sim.run(1.0)
    assert sim.rth.state == "CHARGING"
    sim.command("start")
    assert sim.run(10, until=lambda s: s.rth.state == "MONITOR"), "return_to_home not released"
    assert "resume" in sim.mission_cmds
    assert sim.run(20, until=lambda s: s.m.base == "PATROL" and s.m.phase == "navigate")
    assert sim.run(200, until=lambda s: "GA" in s.m.completed)
