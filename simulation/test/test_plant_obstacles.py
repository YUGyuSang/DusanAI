"""Mission-level tests of the novel-obstacle avoidance (ROS free): a kinematic
robot follows the mission's /cmd_vel/patrol output while
dusan_common.obstacle_sim.FakeElevationSensor reports what the depth camera +
terrain_mapper would see (5 Hz frames, tall faces detected within 1.2 m,
occlusion, persistence).  The ground-truth boxes are only used for the
collision metric; the mission sees them through the sensor alone."""
import json
import math
import os

import numpy as np
import pytest

from dusan_common.grid_map import OccupancyGridMap
from dusan_common.obstacle_layer import EvidenceFrame
from dusan_common.obstacle_sim import Box, FakeElevationSensor, box_clearance

from plant_simulation.mission_logic import MISSION_DEFAULTS, MissionLogic

DT = 0.1
HERE = os.path.dirname(os.path.realpath(__file__))
PLANT_MAP = os.path.normpath(os.path.join(HERE, "..", "maps", "plant_map.yaml"))


class FakeLayout:
    def __init__(self, dock=(0.0, 0.0, 0.0)):
        self.dock_pose = dock
        self.gauges = []
        self.obstacles = []


def corridor(length=24.0, width=4.0, x0=-2.0):
    """Corridor x0 .. x0 + length, y -width/2 .. width/2 (border cells walls)."""
    occ = np.zeros((int(round(width / 0.1)), int(round(length / 0.1))), np.int8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 100
    return OccupancyGridMap(occ, 0.1, x0, -width / 2.0)


def loop_map():
    """Two parallel 1.5 m corridors around a central block (x 2 .. 18, |y| <= 1)."""
    g = corridor(24.0, 5.2)
    xs = g.origin_x + (np.arange(g.width) + 0.5) * 0.1
    ys = g.origin_y + (np.arange(g.height) + 0.5) * 0.1
    X, Y = np.meshgrid(xs, ys)
    g.occ[(X >= 2.0) & (X <= 18.0) & (np.abs(Y) <= 1.0)] = 100
    return OccupancyGridMap(g.occ, 0.1, g.origin_x, g.origin_y)


def plan(*vps, legs=()):
    return {"viewpoints": [{"gauge_id": gid, "x": x, "y": y, "yaw": yaw} for gid, x, y, yaw in vps],
            "order": [v[0] for v in vps], "legs": [list(map(list, leg)) for leg in legs]}


class ObstacleSim:
    """Kinematic robot (10 Hz) + fake elevation-map frames (5 Hz) + metrics."""

    def __init__(self, logic, grid, x=0.0, y=0.0, yaw=0.0, boxes=(), sensor=None, feed=True):
        self.m = logic
        self.grid = grid
        self.x, self.y, self.yaw = x, y, yaw
        self.t = 0.0
        self.k = 0
        self.sensor = sensor or FakeElevationSensor([], static_grid=grid, render_static=True)
        self.schedule = list(boxes)            # (t_appear, Box) or (t_appear, Box, t_remove)
        self.feed = feed
        self.frozen = False
        self.cmds, self.paths, self.events, self.states, self.scales = [], [], [], [], []
        self.traj = []
        self.min_clear = math.inf
        self.reached = {}
        self.read = True

    def _update_boxes(self):
        for item in self.schedule:
            t_on, box = item[0], item[1]
            t_off = item[2] if len(item) > 2 else math.inf
            present = t_on <= self.t < t_off
            if present and box not in self.sensor.boxes:
                self.sensor.boxes.append(box)
            elif not present and box in self.sensor.boxes:
                self.sensor.boxes.remove(box)

    def step(self):
        m = self.m
        m.set_pose(self.x, self.y, self.yaw, self.t)
        self._update_boxes()
        if self.feed and self.k % 2 == 0:
            m.on_obstacle_frame(self.sensor.frame((self.x, self.y, self.yaw)), self.t)
        if m.base == "INSPECTING" and m.current is not None:
            self.reached.setdefault(m.current.gauge_id, self.t)
            if self.read:
                m.on_gauge_readings({"stamp": self.t, "readings": [
                    {"id": m.current.gauge_id, "value": 1.0, "confidence": 0.9}]}, self.t)
        out = m.step(self.t)
        self.events += out.events
        if out.path is not None:
            self.paths.append(out.path)
        if not self.states or self.states[-1] != m.reported_state:
            self.states.append(m.reported_state)
        self.cmds.append(out.cmd_vel)
        if m.base == "PATROL" and m.phase == "navigate":
            self.scales.append(m.obstacles.scale)
        if out.cmd_vel is not None and not self.frozen:
            vx, vy, wz = out.cmd_vel
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.x += (vx * c - vy * s) * DT
            self.y += (vx * s + vy * c) * DT
            self.yaw = (self.yaw + wz * DT + math.pi) % (2 * math.pi) - math.pi
        self.traj.append((self.x, self.y))
        if self.sensor.boxes:
            self.min_clear = min(self.min_clear, box_clearance(self.sensor.boxes, self.x, self.y))
        self.t += DT
        self.k += 1
        return out

    def run(self, seconds, until=None):
        for _ in range(int(round(seconds / DT))):
            self.step()
            if until is not None and until(self):
                return True
        return False

    def texts(self, key):
        return [t for _, t in self.events if key in t]


def make(grid, vps, params=None, layout=None, legs=()):
    p = {"min_start_time": 0.5, "scan_on_miss": False}
    p.update(params or {})
    m = MissionLogic(p, layout=layout or FakeLayout())
    m.set_map(grid)
    m.set_plan(plan(*vps, legs=legs))
    return m


def done(s):
    return s.m.base in ("DONE", "DOCKED")


def far_end_mission(params=None, boxes=(), **kw):
    g = corridor()
    m = make(g, [("G", 20.0, 0.0, 0.0)], params)
    sim = ObstacleSim(m, g, boxes=boxes, **kw)
    m.command("patrol", 0.0)
    return sim


# ----------------------------------------------------------------------------
# M1 / M2 / M3: a box appears on the path
# ----------------------------------------------------------------------------
def test_m1_box_appearing_on_the_path_is_bypassed():
    base = far_end_mission()
    assert base.run(120, until=done)
    t_free = base.reached["G"]
    sim = far_end_mission(boxes=[(3.0, Box((5.5, 0.0), (0.6, 0.6)))])
    assert sim.run(120, until=done)
    m = sim.m
    st = m.state_dict(sim.t)["obstacle"]
    assert m.completed == ["G"] and m.missed == [] and m.total_replans == 0
    assert sim.min_clear >= MISSION_DEFAULTS["robot_radius"] + 0.05, sim.min_clear      # no collision
    assert st["detours"] >= 1 and st["errors"] == 0 and st["replans"] == 0
    assert sim.reached["G"] <= 1.6 * t_free
    assert len(sim.paths) > len(base.paths)                    # the detour went out on /nav/plan
    assert sim.texts("detour")
    json.dumps(m.state_dict(sim.t))


def test_m2_side_choice_and_m3_no_oscillation():
    sim = far_end_mission(boxes=[(3.0, Box((5.5, -0.25), (0.6, 0.6)))])      # box south of the path
    assert sim.run(120, until=done)
    det = sim.texts("detour ")
    sides = {("left" if "detour left" in t else "right") for t in det}
    assert sides == {"left"}, det                               # north = left of an eastward route
    ys = [p[1] for p in sim.traj if 4.0 < p[0] < 7.0]
    assert max(ys) > 0.4 and sim.min_clear >= 0.4
    assert len(det) <= 3
    # commanded turn direction changes during the manoeuvre stay few (no left / right hunting)
    wz = [c[2] for c, p in zip(sim.cmds, sim.traj) if c is not None and 3.0 < p[0] < 9.0 and abs(c[2]) > 0.05]
    flips = sum(1 for a, b in zip(wz[:-1], wz[1:]) if a * b < 0)
    assert flips <= 6, flips
    assert sim.m.state_dict(sim.t)["obstacle"]["errors"] == 0


# ----------------------------------------------------------------------------
# M4: fully blocked -> fallback chain -> skip, next viewpoint visited
# ----------------------------------------------------------------------------
def test_m4_fully_blocked_corridor_falls_back_to_the_stuck_logic():
    g = corridor()
    m = make(g, [("G1", 20.0, 0.0, 0.0), ("G2", 1.5, 1.0, math.pi)])
    sim = ObstacleSim(m, g, boxes=[(0.0, Box((6.0, 0.0), (0.4, 4.0)))])
    m.command("patrol", 0.0)
    assert sim.run(200, until=done)
    st = m.state_dict(sim.t)
    assert m.missed == ["G1"] and "stuck" in m.results["G1"]["reason"] and m.completed == ["G2"]
    assert m.base == "DONE"
    assert sim.min_clear >= 0.40, sim.min_clear                 # held in front (r + margin), never touched
    replans = sim.texts("path blocked by a detected obstacle")  # per-goal counter: reset for G2 meanwhile
    assert 1 <= len(replans) <= MISSION_DEFAULTS["obstacle_max_replans"]
    assert sim.texts("no path around the detected obstacle")
    assert st["obstacle"]["errors"] == 0


def test_stop_is_a_true_zero_command():
    """Field (Gazebo w): a heading correction alone (wz != 0) keeps the legged
    controller trotting in place (it stands only after 1 s of zero command)
    and the robot creeps into the obstacle - a governor stop zeroes everything."""
    g = corridor()
    m = make(g, [("G", 20.0, 0.0, 0.0)], {"stuck_time": 30.0})
    sim = ObstacleSim(m, g, x=4.4, y=0.15, boxes=[(0.0, Box((6.0, 0.0), (0.4, 4.0)))])
    m.command("patrol", 0.0)
    stops = []
    for _ in range(150):
        out = sim.step()
        if m.phase == "navigate" and m.obstacles.scale <= 0.0 and out.cmd_vel is not None:
            stops.append(out.cmd_vel)
    assert stops and all(c == (0.0, 0.0, 0.0) for c in stops)
    assert sim.min_clear >= 0.45


def test_backup_guard_does_not_reverse_into_a_remembered_obstacle():
    g = corridor()
    m = make(g, [("G", 20.0, 0.0, 0.0)], {"stuck_time": 2.0})
    sim = ObstacleSim(m, g, x=4.0, feed=False)
    m.command("patrol", 0.0)
    sim.run(1.0)
    # a box right behind the robot, remembered from earlier frames (the camera does not see backwards)
    pose = (sim.x, sim.y, 0.0)
    n = 150
    oi = int(math.floor(pose[0] / 0.04 + 0.5)) - n // 2
    oj = int(math.floor(pose[1] / 0.04 + 0.5)) - n // 2
    X, Y = np.meshgrid((oi + np.arange(n)) * 0.04, (oj + np.arange(n)) * 0.04)
    hit = (np.abs(X - (pose[0] - 0.75)) <= 0.15) & (np.abs(Y) <= 0.4)
    for k in range(3):
        m.obstacles.update(EvidenceFrame(hit, np.ones_like(hit), oi * 0.04, oj * 0.04, 0.04), pose, sim.t + 0.1 * k)
    assert m.obstacles.blocked_behind((sim.x, sim.y, sim.yaw), 0.525)
    sim.frozen = True
    sim.run(4.0)
    assert sim.texts("stuck") and sim.texts("detected obstacle behind")
    assert not any(c is not None and c[0] < 0.0 for c in sim.cmds)            # never reversed


# ----------------------------------------------------------------------------
# M5: alternative route through the other corridor
# ----------------------------------------------------------------------------
def test_m5_global_replan_takes_the_other_corridor():
    g = loop_map()
    m = make(g, [("G", 20.0, 0.0, 0.0)])
    probe = MissionLogic({"min_start_time": 0.5}, layout=FakeLayout())
    probe.set_map(g)
    probe.set_pose(0.0, 0.0, 0.0, 0.0)
    first = probe._plan((20.0, 0.0))
    side = 1.0 if np.mean([p[1] for p in first if 5.0 < p[0] < 15.0]) > 0 else -1.0
    wall = Box((8.0, side * 1.8), (0.4, 1.8))                   # closes the corridor the route takes
    sim = ObstacleSim(m, g, boxes=[(0.0, wall)])
    m.command("patrol", 0.0)
    assert sim.run(200, until=done)
    st = m.state_dict(sim.t)["obstacle"]
    assert m.completed == ["G"] and m.missed == []
    assert st["replans"] == 1 and sim.min_clear >= 0.4
    ys = [p[1] for p in sim.traj if 9.0 < p[0] < 15.0]
    assert ys and np.mean(ys) * side < 0                        # went through the other corridor
    assert st["errors"] == 0


# ----------------------------------------------------------------------------
# M6 / M7: ghosts and a removed obstacle
# ----------------------------------------------------------------------------
def test_m6_single_frame_ghosts_do_not_cause_detours():
    base = far_end_mission()
    assert base.run(120, until=done)
    g = corridor()
    sen = FakeElevationSensor([], static_grid=g, render_static=True, ghost_rate=0.3, ghost_cells=2,
                              ghost_region=(1.0, -0.4, 21.0, 0.4), seed=4)
    sim = far_end_mission(sensor=sen)
    assert sim.run(120, until=done)
    st = sim.m.state_dict(sim.t)["obstacle"]
    assert sen.ghosts > 20
    assert st["detours"] == 0 and st["replans"] == 0 and st["confirmed"] == 0
    assert len(sim.paths) == len(base.paths)
    slow = np.mean(np.asarray(sim.scales) < 1.0)
    assert slow <= 0.10, slow
    assert sim.reached["G"] <= 1.10 * base.reached["G"]


def test_m7_removed_obstacle_clears_from_memory():
    """A removed box clears as soon as its cells are seen free again.  The
    robot stands still facing the spot (the depth camera's field of view
    starts 0.34 m ahead of base_link: once the robot turns into a detour the
    far corner of the box is no longer observed and would only be forgotten
    after obstacle_forget_time)."""
    box = Box((5.5, 0.0), (0.6, 0.6))
    sim = far_end_mission(boxes=[(0.0, box)])
    assert sim.run(30.0, until=lambda s: s.m.obstacles.memory.has_confirmed())
    sim.schedule = [(0.0, box, sim.t)]                         # removed now
    sim.frozen = True
    x_seen = sim.x
    sim.run(1.5)
    assert not sim.m.obstacles.memory.has_confirmed(), (x_seen, sim.x)
    sim.frozen = False
    assert sim.run(120, until=done)
    assert sim.m.completed == ["G"] and sim.m.total_replans == 0


# ----------------------------------------------------------------------------
# M8: disabled / no frames = legacy behaviour
# ----------------------------------------------------------------------------
def test_m8_disabled_or_without_frames_is_the_legacy_behaviour():
    off = far_end_mission({"obstacle_enabled": False}, boxes=[(3.0, Box((5.5, 0.0), (0.6, 0.6)))])
    off.run(25.0)
    st = off.m.state_dict(off.t)["obstacle"]
    assert st["enabled"] is False and st["frames"] == 0 and st["detours"] == 0 and len(off.paths) == 1
    a = far_end_mission({"obstacle_enabled": False}, feed=False)
    b = far_end_mission({}, feed=False)
    a.run(60.0)
    b.run(60.0)
    assert a.cmds == b.cmds and a.paths == b.paths and a.events == b.events


# ----------------------------------------------------------------------------
# M9: self-RTH drives around a box to the dock
# ----------------------------------------------------------------------------
def test_m9_self_rth_detours_to_the_dock():
    g = corridor()
    m = make(g, [("G", 20.0, 0.0, 0.0)], {"rth_ack_timeout": 1.0})
    sim = ObstacleSim(m, g, x=16.0, yaw=math.pi, boxes=[(0.0, Box((10.0, 0.0), (0.6, 0.6)))])
    m.command("rth", 0.0)
    assert sim.run(120, until=lambda s: s.m.base == "DOCKED")
    assert math.hypot(sim.x, sim.y) <= MISSION_DEFAULTS["dock_tolerance"] + 0.05
    st = m.state_dict(sim.t)["obstacle"]
    assert st["detours"] >= 1 and sim.min_clear >= 0.4 and st["errors"] == 0
    assert sim.texts("driving to the dock myself")


# ----------------------------------------------------------------------------
# M10 / M11: plan legs and mapped narrow passages
# ----------------------------------------------------------------------------
def test_m10_blocked_patrol_leg_is_replanned_around_the_obstacle():
    g = corridor()
    leg = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    m = make(g, [("G", 20.0, 0.0, 0.0)], legs=[leg])
    sim = ObstacleSim(m, g, boxes=[(0.0, Box((1.45, 0.0), (0.6, 0.6)))])
    m.command("patrol", 0.0)
    assert sim.run(120, until=done)
    assert sim.texts("patrol leg blocked by a detected obstacle")
    first = np.asarray(sim.paths[0])
    assert min(box_clearance([Box((1.45, 0.0), (0.6, 0.6))], *p) for p in first) >= 0.35
    assert m.completed == ["G"] and sim.min_clear >= 0.4


def test_m11_mapped_narrow_gap_is_not_a_novel_obstacle():
    g = corridor()
    xs = g.origin_x + (np.arange(g.width) + 0.5) * 0.1
    ys = g.origin_y + (np.arange(g.height) + 0.5) * 0.1
    X, Y = np.meshgrid(xs, ys)
    g.occ[(np.abs(X - 8.0) <= 0.1) & (Y <= 1.2)] = 100           # 0.75 m gap at the north wall
    g = OccupancyGridMap(g.occ, 0.1, g.origin_x, g.origin_y)
    m = make(g, [("G", 20.0, 0.0, 0.0)])
    sim = ObstacleSim(m, g)                                     # the sensor renders the mapped walls
    m.command("patrol", 0.0)
    assert sim.run(150, until=done)
    st = m.state_dict(sim.t)["obstacle"]
    assert m.completed == ["G"] and st["confirmed"] == 0 and st["detours"] == 0 and st["replans"] == 0
    assert min(sim.scales) == 1.0 and len(sim.paths) == 2       # plan + the final clear
    assert max(p[1] for p in sim.traj) > 1.3                    # went through the gap


def test_map_free_operation_still_detours():
    """No nav map at all (any site): straight-line legs, every detected obstacle
    is novel and detours are planned in the obstacle memory window."""
    m = MissionLogic({"min_start_time": 0.5, "scan_on_miss": False}, layout=FakeLayout())
    m.set_plan(plan(("G", 12.0, 0.0, 0.0)))
    sim = ObstacleSim(m, None, sensor=FakeElevationSensor([]), boxes=[(0.0, Box((5.0, 0.0), (0.6, 0.6)))])
    m.command("patrol", 0.0)
    assert sim.run(80, until=done)
    st = m.state_dict(sim.t)["obstacle"]
    assert m.completed == ["G"] and st["detours"] >= 1 and st["errors"] == 0
    assert sim.min_clear >= MISSION_DEFAULTS["robot_radius"]
    assert m.obstacles.planning_grid(None) is None              # no global re-plan without a map


def test_obstacle_frames_need_a_fresh_pose_and_errors_are_contained():
    g = corridor()
    m = make(g, [("G", 20.0, 0.0, 0.0)])
    sen = FakeElevationSensor([Box((2.0, 0.0), (0.6, 0.6))])
    f = sen.frame((1.0, 0.0, 0.0))
    assert m.on_obstacle_frame(f, 0.0) is False and m.obstacles.memory.frames == 0     # no pose yet
    m.set_pose(1.0, 0.0, 0.0, 0.0)
    assert m.on_obstacle_frame(f, 5.0) is False and m.obstacles.memory.frames == 0     # stale pose
    sim = far_end_mission(boxes=[(3.0, Box((5.5, 0.0), (0.6, 0.6)))])

    def boom(*a, **k):
        raise RuntimeError("bug")
    sim.m.obstacles.step = boom
    assert sim.run(25.0) is False
    st = sim.m.state_dict(sim.t)["obstacle"]
    assert st["errors"] > 1 and sim.texts("obstacle avoidance failed (1x)")
    assert sum(1 for t in sim.texts("obstacle avoidance failed")) <= 1 + st["errors"] // 50


# ----------------------------------------------------------------------------
# M12: the real plant map + layout (terrain cost respected)
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(PLANT_MAP), reason="simulation/maps/plant_map.yaml not generated")
@pytest.mark.parametrize("case", ["pallet_south", "dock_start"])
def test_m12_real_plant_map(case):
    from dusan_common.layout import Layout
    from dusan_common.terrain_cost import debris_crossed
    lay = Layout.load()
    g = OccupancyGridMap.load(PLANT_MAP)
    if case == "pallet_south":       # G08 viewpoint -> V2 south of pallet C-OBS2, box on the leg
        start, goal, box, side = (56.9, 0.0, math.pi), ("V2", 52.2, -0.4, math.pi), Box((54.2, -0.30), (0.5, 0.5)), \
            "left"
    else:                            # dock -> G08, box on the A* leg, seen during the stand-up
        start, goal, box, side = (61.9, 0.0, math.pi), ("G08", 57.0, 0.15, -math.pi / 2), \
            Box((60.45, 0.65), (0.6, 0.6)), None
    m = make(g, [goal], layout=lay)
    sim = ObstacleSim(m, g, *start, boxes=[(0.0, box)])
    m.command("patrol", 0.0)
    assert sim.run(90, until=done)
    st = m.state_dict(sim.t)["obstacle"]
    assert m.completed == [goal[0]] and m.missed == [] and m.total_replans == 0
    assert sim.min_clear >= 0.40, sim.min_clear
    assert debris_crossed(sim.traj, lay) == []                  # no pipe / rubble trampled on the detour
    assert st["errors"] == 0
    if side is not None:
        assert st["detours"] >= 1 and {("left" if "detour left" in t else "right")
                                       for t in sim.texts("detour ")} == {side}


# ----------------------------------------------------------------------------
# M13..M18: review regressions (unseen back of a box, rows of boxes, specks,
# tight gaps, debris-aware side choice, slow planning)
# ----------------------------------------------------------------------------
R_MARGIN = MISSION_DEFAULTS["robot_radius"] + MISSION_DEFAULTS["obstacle_margin"]


@pytest.mark.parametrize("depth,width,yc", [(0.6, 0.6, 0.0), (1.0, 0.8, -0.2), (1.5, 0.6, 0.0)])
def test_m13_detour_keeps_clear_of_the_unseen_back_of_a_deep_box(depth, width, yc):
    """The camera (0.34 m ahead of base_link, FakeElevationSensor default)
    sees the front face of a box, never its sides while passing nor its back:
    the detour must not rejoin through the unseen back corner (it did at
    0.38 m for a 0.6 m deep box, 0.15 m for 1.0 m with the real pipeline)."""
    g = corridor()
    box = Box((7.0, yc), (depth, width))
    m = make(g, [("G", 20.0, 0.0, 0.0)])
    sim = ObstacleSim(m, g, boxes=[(3.0, box)])
    m.command("patrol", 0.0)
    assert sim.run(120, until=done)
    st = m.state_dict(sim.t)["obstacle"]
    assert m.completed == ["G"] and m.total_replans == 0 and st["errors"] == 0
    assert sim.min_clear >= R_MARGIN, sim.min_clear


def test_m14_box_leaving_a_tight_gap_at_the_wall_is_still_passed():
    """The full assumed depth of the box closes the 1.0 m gap beside it: the
    detour falls back to the near shadow and passes (no skip)."""
    g = corridor(width=3.6)
    box = Box((7.0, 0.5), (0.6, 2.6))                           # touches the north wall
    m = make(g, [("G", 20.0, 0.0, 0.0)])
    sim = ObstacleSim(m, g, boxes=[(3.0, box)])
    m.command("patrol", 0.0)
    assert sim.run(150, until=done)
    assert m.completed == ["G"] and m.missed == [] and m.total_replans == 0
    assert sim.min_clear >= 0.40, sim.min_clear
    assert min(p[1] for p in sim.traj if 6.6 < p[0] < 7.4) < -1.0       # through the gap


def test_m15_boxes_in_a_row_are_bypassed_without_redundant_detours():
    """A sensor that sees 2.5 m ahead and two boxes 1.8 m apart: the second box
    is merged into the detour (the review saw 7 identical detours per goal,
    every min_interval, until the budget ran out)."""
    g = corridor()
    A, C = Box((6.0, 0.0), (0.4, 0.6)), Box((8.6, 0.0), (0.4, 0.6))
    sen = FakeElevationSensor([], static_grid=g, render_static=True, detect_range=2.5)
    m = make(g, [("G", 20.0, 0.0, 0.0)])
    sim = ObstacleSim(m, g, boxes=[(0.0, A), (0.0, C)], sensor=sen)
    m.command("patrol", 0.0)
    assert sim.run(120, until=done)
    st = m.state_dict(sim.t)["obstacle"]
    assert m.completed == ["G"] and st["replans"] == 0 and m.total_replans == 0
    assert st["detours"] <= 3, sim.texts("detour ")
    assert sim.min_clear >= R_MARGIN, sim.min_clear


def test_m16_isolated_evidence_cell_does_not_stop_the_robot():
    """One evidence cell every frame (a cost-100 speck of the traversability
    source): dropped by min_cluster for the planners, so the governor must
    not stop the robot for it either (it did until the viewpoint was skipped)."""
    g = corridor()

    class Speck(FakeElevationSensor):
        def frame(self, pose, t=None):
            f = super().frame(pose, t)
            X, Y = f.centres()
            k = np.unravel_index(np.argmin(np.hypot(X - 6.02, Y - 0.02)), X.shape)
            hit = f.hit.copy()
            if f.known[k] and math.hypot(X[k] - 6.02, Y[k] - 0.02) < 0.03:
                hit[k] = True
            return EvidenceFrame(hit, f.known, f.x0, f.y0, f.res, f.frame_id)

    m = make(g, [("G", 12.0, 0.0, 0.0)])
    sim = ObstacleSim(m, g, sensor=Speck([], static_grid=g, render_static=True))
    m.command("patrol", 0.0)
    assert sim.run(80, until=done)
    assert m.completed == ["G"] and m.total_replans == 0 and not sim.texts("stuck")
    assert min(sim.scales) > 0.0


@pytest.mark.skipif(not os.path.exists(PLANT_MAP), reason="simulation/maps/plant_map.yaml not generated")
def test_m17_detour_side_avoids_mapped_debris_when_a_clean_side_exists():
    """Real plant map + layout, hall leg to G06: the short side round the box
    crosses rubble F-OBS5, the clean side is longer - the clean side is taken
    (sides compared with length + terrain cost, the planner's own metric)."""
    from dusan_common.layout import Layout
    from dusan_common.terrain_cost import debris_crossed
    lay = Layout.load()
    g = OccupancyGridMap.load(PLANT_MAP)
    m = make(g, [("G06", 3.5, 4.3, math.pi / 2)], layout=lay)
    sim = ObstacleSim(m, g, 9.6, 0.6, 2.75, boxes=[(0.0, Box((6.19, 1.79), (0.6, 0.6)))])
    m.command("patrol", 0.0)
    assert sim.run(120, until=done)
    st = m.state_dict(sim.t)["obstacle"]
    assert m.completed == ["G06"] and st["detours"] >= 1 and st["errors"] == 0
    assert debris_crossed(sim.traj, lay) == []
    assert sim.min_clear >= R_MARGIN


def test_m18_slow_planning_cannot_lock_the_patrol_in_hold():
    """rclpy executor model: a mission tick that blocks longer than pose_timeout
    (a slow plan on a busy 8 GB laptop) lets the overdue timer run before the
    queued /odom - 'waiting for /odom' -> hold.  Releasing the hold re-planned
    every time, blocked again and looped forever (no stuck count, no skip);
    a valid path is now resumed without re-planning."""
    g = corridor()
    calls = [0]
    holder = {}

    def slow_planner(grid, start, goal, radius):     # every call 'takes' PLAN_S of wall time
        calls[0] += 1
        return holder["m"]._terrain_planner(grid, start, goal, radius)

    PLAN_S = 1.5
    m = MissionLogic({"min_start_time": 0.5, "scan_on_miss": False}, layout=FakeLayout(), planner=slow_planner)
    holder["m"] = m
    m.set_map(g)
    m.set_plan(plan(("G1", 20.0, 0.0, 0.0), ("G2", 1.5, 1.0, math.pi)))
    sen = FakeElevationSensor([Box((6.0, 0.0), (0.4, 4.0))], static_grid=g, render_static=True)
    x, y, yaw, t, last_frame = 0.0, 0.0, 0.0, 0.0, -1.0
    m.set_pose(x, y, yaw, t)
    m.command("patrol", t)
    events = []
    while t < 300.0 and m.base != "DONE":
        if m.base == "INSPECTING" and m.current is not None:
            m.on_gauge_readings({"stamp": t, "readings": [{"id": m.current.gauge_id, "value": 1.0,
                                                           "confidence": 0.9}]}, t)
        n0 = calls[0]
        out = m.step(t)
        events += out.events
        block = PLAN_S * (calls[0] - n0)
        cmd = out.cmd_vel or (0.0, 0.0, 0.0)
        dt = max(0.1, block)
        run = min(dt, 0.5)                            # the cmd_vel mux drops a command after 0.5 s
        c, s = math.cos(yaw), math.sin(yaw)
        x += (cmd[0] * c - cmd[1] * s) * run
        y += (cmd[0] * s + cmd[1] * c) * run
        yaw = (yaw + cmd[2] * run + math.pi) % (2 * math.pi) - math.pi
        t += dt
        if block > 0.1:                               # the overdue timer fires before /odom is taken
            n0 = calls[0]
            events += m.step(t).events
            t += PLAN_S * (calls[0] - n0) + 0.001
        m.set_pose(x, y, yaw, t)
        if t - last_frame >= 0.2:
            last_frame = t
            m.on_obstacle_frame(sen.frame((x, y, yaw)), t)
    texts = [e[1] for e in events]
    assert m.base == "DONE", (m.base, m.phase, calls[0], texts[-4:])
    assert m.missed == ["G1"] and "stuck" in m.results["G1"]["reason"] and m.completed == ["G2"]
    assert calls[0] < 80, calls[0]


def _band(g, x0, x1, y_abs, w):
    X, Y = np.meshgrid(g.origin_x + (np.arange(g.width) + 0.5) * g.resolution,
                       g.origin_y + (np.arange(g.height) + 0.5) * g.resolution)
    return np.where((X >= x0) & (X <= x1) & (np.abs(Y) >= y_abs), w, 0.0)


def test_m19_route_past_the_near_side_when_the_assumed_back_forces_debris():
    """Gazebo t: a box seen from afar gets the full (1.0 m) occlusion shadow;
    it closed the only clean way and the route went across a pipe (fell).
    The route on the near (0.3 m) shadow is taken when switch_ratio x cheaper
    (length + terrain cost); without that comparison the band is crossed."""
    g = corridor()
    cost = _band(g, 5.4, 6.0, 0.5, 25.0)                         # 'debris' everywhere beside the box's back

    def plan_with(compare):
        m = make(g, [("G", 11.0, 0.0, 0.0)])
        m._static_cost = lambda: cost                            # the terrain cost layer of the nav map
        if not compare:
            m.obstacles.near_is_cheaper = lambda a, b: False
        for k in range(3):                                       # front face seen from 1.6 m, back unseen
            fr = EvidenceFrame(*_box_view(g, 4.5, 0.0, 0.3, 0.3, 4.3))
            m.set_pose(2.6, 0.0, 0.0, 0.2 * k)
            m.on_obstacle_frame(fr, 0.2 * k)
        path = np.asarray(m._plan((11.0, 0.0)))
        inband = int(np.sum((path[:, 0] >= 5.4) & (path[:, 0] <= 6.0) & (np.abs(path[:, 1]) >= 0.5)))
        return m, path, inband

    m, path, inband = plan_with(True)
    assert inband == 0 and any("near side" in t for _, t in m._out.events)
    assert min(box_clearance([Box((4.5, 0.0), (0.6, 0.6))], x, y) for x, y in path) >= R_MARGIN - 0.05
    m, path, inband = plan_with(False)
    assert inband > 0


def _box_view(g, cx, cy, hx, hy, x_seen, res=0.04, n=150, centre=(2.6, 0.0)):
    """(hit, known, x0, y0, res) of a frame that sees the box only up to x_seen."""
    oi = int(math.floor(centre[0] / res + 0.5)) - n // 2
    oj = int(math.floor(centre[1] / res + 0.5)) - n // 2
    X, Y = np.meshgrid((oi + np.arange(n)) * res, (oj + np.arange(n)) * res)
    known = X < x_seen
    hit = (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy) & known
    return hit, known, oi * res, oj * res, res
