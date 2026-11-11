"""Novel-obstacle handling of Module 3 (ROS free): return_to_home driving
(detour / static fallback / dynamic safety filter) and the gas seeker's
obstacle memory, plus parameter consistency."""
import dataclasses
import math
import time

import numpy as np
import pytest

from dusan_common.grid_map import OccupancyGridMap
from dusan_common.obstacle_layer import OBSTACLE_DEFAULTS, OBSTACLE_MEMORY_KEYS, OBSTACLE_NODE_KEYS, EvidenceFrame
from dusan_common.obstacle_sim import Box, FakeElevationSensor, box_clearance, paint_boxes

from module3_gas_safety import params as P
from module3_gas_safety.rth_logic import (DOCKED, DOCKING, MONITOR, RTH_NAVIGATING, RTH_PLANNING, PlannerWorker,
                                          RthConfig, RthInputs, RthLogic, SyncPlanner)
from module3_gas_safety.sim_harness import RobotSim
from module3_gas_safety.source_seeking import SEARCH, SeekerConfig, SourceSeeker

DOCK = (20.0, 0.0, 0.0)


def corridor(length=24.0, width=4.0, x0=-2.0):
    occ = np.zeros((int(round(width / 0.1)), int(round(length / 0.1))), np.int8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 100
    return OccupancyGridMap(occ, 0.1, x0, -width / 2.0)


def box_frame(centre, pred, res=0.04, n=150):
    oi = int(math.floor(centre[0] / res + 0.5)) - n // 2
    oj = int(math.floor(centre[1] / res + 0.5)) - n // 2
    X, Y = np.meshgrid((oi + np.arange(n)) * res, (oj + np.arange(n)) * res)
    hit = pred(X, Y)
    return EvidenceFrame(hit, np.ones_like(hit), oi * res, oj * res, res)


def run_rth(boxes, t_max=150.0, start=(0.0, 0.0, 0.0), **cfg):
    """Manual RTH on a corridor; the robot (holonomic RobotSim, body radius 0.25)
    collides with the TRUE boxes; the logic sees them through the fake sensor."""
    grid = corridor()
    truth = paint_boxes(grid, boxes)
    lg = RthLogic(RthConfig(**cfg), grid, DOCK)
    sen = FakeElevationSensor(boxes, static_grid=grid, render_static=True)
    robot = RobotSim(*start, max_speed=0.45)
    t, k = 0.0, 0
    log = {"states": [lg.state], "messages": [], "min_clear": math.inf, "plans": 0}
    cmd = "rth"
    while t < t_max:
        frame = sen.frame((robot.x, robot.y, robot.yaw)) if k % 2 == 0 else None
        out = lg.update(RthInputs(t, robot.x, robot.y, robot.yaw, robot.vx_w, robot.vy_w, battery_pct=80.0,
                                  command=cmd, obstacle_frame=frame))
        cmd = None
        robot.step(out.cmd or (0.0, 0.0, 0.0), 0.1, truth)
        if out.plan:
            log["plans"] += 1
        if lg.state != log["states"][-1]:
            log["states"].append(lg.state)
        if not log["messages"] or log["messages"][-1] != lg.message:
            log["messages"].append(lg.message)
        log["min_clear"] = min(log["min_clear"], box_clearance(boxes, robot.x, robot.y))
        t += 0.1
        k += 1
        if lg.state == DOCKED:
            break
    return lg, robot, log


# ----------------------------------------------------------------------------
# return_to_home
# ----------------------------------------------------------------------------
def test_rth_detours_around_a_box_on_the_way_home():
    box = Box((6.0, 0.0), (0.6, 0.6))
    lg, robot, log = run_rth([box])
    assert lg.state == DOCKED, log["messages"][-5:]
    assert robot.contacts == 0 and log["min_clear"] >= 0.35
    st = lg.status()["obstacle"]
    assert st["detours"] >= 1 and st["errors"] == 0
    assert any("detour" in m for m in log["messages"])
    assert lg.plan_failures == 0
    assert math.hypot(robot.x - DOCK[0], robot.y - DOCK[1]) < 0.2


def test_rth_fully_blocked_falls_back_to_the_map_and_never_collides():
    wall = Box((6.0, 0.0), (0.4, 4.0))
    lg, robot, log = run_rth([wall], t_max=60.0)
    assert robot.contacts == 0
    assert log["min_clear"] >= 0.5, log["min_clear"]            # held in front of the wall
    assert any("planning on the map" in m for m in log["messages"])
    assert lg.plan_failures == 0                                # the fallback is not a planning failure
    assert lg.replans >= 1                                      # stuck -> back off -> re-plan
    assert lg.state in (RTH_PLANNING, RTH_NAVIGATING)
    assert lg.status()["obstacle"]["errors"] == 0


def test_rth_disabled_ignores_frames():
    box = Box((6.0, 0.0), (0.6, 0.6))
    lg, robot, log = run_rth([box], t_max=25.0, obstacle_enabled=False)
    st = lg.status()["obstacle"]
    assert st["enabled"] is False and st["frames"] == 0 and st["detours"] == 0


def test_rth_safety_filter_and_backoff_respect_detected_obstacles():
    grid = corridor()
    lg = RthLogic(RthConfig(), grid, DOCK)
    x, y = 5.0, 0.0                                              # corridor centre: static clearance 1.9 m
    assert lg._clearance(x, y)[0] > 1.5
    pose = (x - 1.5, 0.0, 0.0)
    for k in range(3):
        lg.obstacles.update(box_frame(pose, lambda X, Y: (np.abs(X - x) <= 0.3) & (np.abs(Y - 0.5) <= 0.1)),
                            pose, 0.2 * k)
    clr, n = lg._clearance(x, y)
    assert clr < 0.5 and n[1] < -0.9                             # away from the box (south)
    vx, vy, wz = lg._safe_cmd(x, y, math.pi / 2, (0.3, 0.0, 0.0))  # heading north into the box
    assert vx < 0.3 * 0.9
    assert lg._safe_cmd(x, y, 0.0, (0.3, 0.0, 0.0))[0] == pytest.approx(0.3, abs=0.02)   # along it: kept


def test_planners_use_the_given_grid():
    grid = corridor()
    blocked = OccupancyGridMap(grid.occ.copy(), 0.1, grid.origin_x, grid.origin_y)
    blocked.occ[:, 80:84] = 100
    sp = SyncPlanner(grid, 0.3)
    sp.submit((0.0, 0.0), (18.0, 0.0), grid=blocked)
    assert sp.poll() == (False, None)
    sp.submit((0.0, 0.0), (18.0, 0.0))
    ok, path = sp.poll()
    assert ok and path
    pw = PlannerWorker(grid, 0.3)
    pw.submit((0.0, 0.0), (18.0, 0.0), grid=blocked)
    t0 = time.time()
    res = None
    while res is None and time.time() - t0 < 20.0:
        res = pw.poll()
        time.sleep(0.01)
    assert res == (False, None)


# ----------------------------------------------------------------------------
# gas seeker
# ----------------------------------------------------------------------------
def test_seeker_merges_detected_obstacles_into_its_grid():
    grid = corridor()
    sk = SourceSeeker(SeekerConfig(), grid)
    assert sk.grid is grid
    pose = (3.0, 0.0, 0.0)
    pred = lambda X, Y: (np.abs(X - 5.0) <= 0.3) & (np.abs(Y) <= 0.3)
    assert sk.update_obstacles(box_frame(pose, pred), pose, 10.0) is False
    assert sk.grid is grid                                       # one frame: not confirmed
    assert sk.update_obstacles(box_frame(pose, pred), pose, 10.2) is True
    assert sk.obstacles.has_confirmed() and sk.grid is grid      # idle: memory only, no grid rebuilds
    sk.enable(10.3)                                              # ... until the seeker is enabled
    assert sk.grid is not grid and sk.clearance(5.0, 0.0) < sk.cfg.robot_radius
    assert sk.clearance(5.0, 0.0) < 0.1 and grid.clearance(5.0, 0.0) > 1.5
    fl = sk.free_length(np.array([3.0, 0.0]), np.array([1.0, 0.0]), 4.0)
    assert fl < 2.0 - sk.cfg.robot_radius + 0.1
    assert sk.status()["obstacles"] > 0
    # rebuilds are rate limited
    g1 = sk.grid
    pred2 = lambda X, Y: pred(X, Y) | ((np.abs(X - 5.0) <= 0.3) & (np.abs(Y - 1.0) <= 0.2))
    sk.update_obstacles(box_frame(pose, pred2), pose, 10.4)
    sk.update_obstacles(box_frame(pose, pred2), pose, 10.6)
    assert sk.grid is g1                                         # changed, but within rebuild_period
    sk.update_obstacles(box_frame(pose, pred2), pose, 11.3)
    assert sk.grid is not g1


def test_seeker_reroutes_an_active_step_across_a_new_obstacle():
    grid = corridor()
    sk = SourceSeeker(SeekerConfig(), grid)
    sk.enable(0.0)
    assert sk.state == SEARCH
    sk._go(0.0, np.array([9.0, 0.0]), None)                     # a straight leg through x = 5
    pose = (3.0, 0.0, 0.0)
    pred = lambda X, Y: (np.abs(X - 5.0) <= 0.3) & (np.abs(Y) <= 0.3)
    for k in range(2):
        sk.update_obstacles(box_frame(pose, pred), pose, 1.0 + 0.2 * k)
    assert sk._route is not None and sk.reroutes == 1
    r = np.vstack([[3.0, 0.0], sk._route])
    assert min(box_clearance([Box((5.0, 0.0), (0.6, 0.6))], *q) for q in r) > 0.25


def test_seeker_disabled_or_without_map_is_unchanged():
    grid = corridor()
    pose = (3.0, 0.0, 0.0)
    pred = lambda X, Y: (np.abs(X - 5.0) <= 0.3) & (np.abs(Y) <= 0.3)
    off = SourceSeeker(SeekerConfig(obstacle_enabled=False), grid)
    for k in range(3):
        assert off.update_obstacles(box_frame(pose, pred), pose, 0.2 * k) is False
    assert off.grid is grid and off.obstacles.frames == 0
    free = SourceSeeker(SeekerConfig(), None)
    for k in range(3):
        assert free.update_obstacles(box_frame(pose, pred), pose, 0.2 * k) is False
    assert free.grid is None


# ----------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------
def test_obstacle_parameters_are_declared_consistently():
    rth = vars(RthConfig())
    numeric = [k for k in OBSTACLE_DEFAULTS if k not in OBSTACLE_NODE_KEYS]
    for k in numeric:
        assert k in rth and rth[k] == OBSTACLE_DEFAULTS[k] and type(rth[k]) is type(OBSTACLE_DEFAULTS[k]), k
    seek = vars(SeekerConfig())
    for k in OBSTACLE_MEMORY_KEYS:
        if k in OBSTACLE_NODE_KEYS:
            assert P.SEEKER_PARAMS[k] == OBSTACLE_DEFAULTS[k], k
        else:
            assert seek[k] == OBSTACLE_DEFAULTS[k] and type(seek[k]) is type(OBSTACLE_DEFAULTS[k]), k
    assert "obstacle_rebuild_period" in seek and isinstance(seek["obstacle_rebuild_period"], float)
    for k in OBSTACLE_NODE_KEYS:
        assert P.RTH_PARAMS[k] == OBSTACLE_DEFAULTS[k] and P.SEEKER_PARAMS[k] == OBSTACLE_DEFAULTS[k]
    assert set(OBSTACLE_DEFAULTS) <= set(P.RTH_PARAMS)
    rc = P.rth_config(dict(P.RTH_PARAMS, obstacle_margin=0.2, obstacle_max_detours=2.0), None)
    assert rc.obstacle_margin == 0.2 and rc.obstacle_max_detours == 2 and isinstance(rc.obstacle_max_detours, int)
    sc = P.seeker_config(dict(P.SEEKER_PARAMS, obstacle_confirm_hits=3))
    assert sc.obstacle_confirm_hits == 3
    lg = RthLogic(rc, corridor(), DOCK)
    assert lg.obstacles.p.margin == 0.2 and lg.obstacles.p.source == "elevation"


# ----------------------------------------------------------------------------
# review regressions
# ----------------------------------------------------------------------------
def test_rth_docks_past_a_box_beside_the_dock_without_contact():
    """F1: DOCKING (straight P control to the dock) starts only when the line
    robot -> dock is free of detected obstacles; the robot used to enter it
    beside a box 0.5 m in front of the dock and push into its corner forever."""
    box = Box((19.3, 0.0), (0.4, 0.6))
    lg, robot, log = run_rth([box], t_max=150.0)
    assert lg.state == DOCKED, log["messages"][-6:]
    assert robot.contacts == 0 and log["min_clear"] >= lg.cfg.robot_radius, log["min_clear"]
    assert math.hypot(robot.x - DOCK[0], robot.y - DOCK[1]) < 0.2
    assert lg.status()["obstacle"]["errors"] == 0


def test_rth_obstacle_at_the_dock_is_never_touched_and_docking_times_out():
    """F1: a detected obstacle on the dock itself - the docking governor holds
    the robot in front (true zero command), the watchdog re-plans after
    stuck_time_s instead of DOCKING forever, and nothing is touched."""
    box = Box((20.35, 0.0), (0.3, 0.6))                         # face 0.2 m behind the dock pose
    lg, robot, log = run_rth([box], t_max=90.0)
    assert robot.contacts == 0
    assert lg.state != DOCKED
    assert any("no docking progress" in m for m in log["messages"]), log["messages"][-8:]
    assert lg.status()["obstacle"]["errors"] == 0


def test_rth_tight_gap_beside_a_box_falls_back_to_the_near_shadow():
    """The assumed (unseen) depth of a box against the wall closes the 1.0 m gap
    beside it: the home drive detours on the near shadow instead of stopping."""
    def corridor36():
        occ = np.zeros((36, 240), np.int8)
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 100
        return OccupancyGridMap(occ, 0.1, -2.0, -1.8)
    grid = corridor36()
    box = Box((7.0, 0.5), (0.6, 2.6))
    truth = paint_boxes(grid, [box])
    lg = RthLogic(RthConfig(), grid, DOCK)
    sen = FakeElevationSensor([box], static_grid=grid, render_static=True)
    robot = RobotSim(0.0, 0.0, 0.0, max_speed=0.45)
    t, k, cmd, clear = 0.0, 0, "rth", math.inf
    while t < 150.0 and lg.state != DOCKED:
        frame = sen.frame((robot.x, robot.y, robot.yaw)) if k % 2 == 0 else None
        out = lg.update(RthInputs(t, robot.x, robot.y, robot.yaw, robot.vx_w, robot.vy_w, battery_pct=80.0,
                                  command=cmd, obstacle_frame=frame))
        cmd = None
        robot.step(out.cmd or (0.0, 0.0, 0.0), 0.1, truth)
        clear = min(clear, box_clearance([box], robot.x, robot.y))
        t += 0.1
        k += 1
    assert lg.state == DOCKED and robot.contacts == 0 and clear >= lg.cfg.robot_radius, (lg.state, clear)
    assert lg.plan_failures == 0


def test_rth_reports_the_node_obstacle_source():
    """F5: RthConfig holds no str keys; the node's obstacle_source is passed in."""
    assert RthLogic(RthConfig(), corridor(), DOCK).status()["obstacle"]["source"] == "elevation"
    lg = RthLogic(RthConfig(), corridor(), DOCK, obstacle_source="traversability")
    assert lg.status()["obstacle"]["source"] == "traversability"


def test_rth_takes_the_near_side_route_when_the_assumed_back_forces_debris():
    """As mission m19 (Gazebo t): the full occlusion shadow of a box seen from
    afar leaves only a way across a costly band; the home drive plans the
    near-shadow alternative and takes it (switch_ratio cheaper)."""
    grid = corridor()

    def band(g):
        X, Y = np.meshgrid(g.origin_x + (np.arange(g.width) + 0.5) * g.resolution,
                           g.origin_y + (np.arange(g.height) + 0.5) * g.resolution)
        return np.where((X >= 5.4) & (X <= 6.0) & (np.abs(Y) >= 0.5), 25.0, 0.0)

    def first_plan(compare):
        cfg = RthConfig(obstacle_idle_period=0.0)
        lg = RthLogic(cfg, grid, DOCK, planner=SyncPlanner(grid, cfg.robot_radius, cost_fn=band))
        if not compare:
            lg.obstacles.near_is_cheaper = lambda a, b: False
        res, n = 0.04, 150
        oi, oj = int(math.floor(2.6 / res + 0.5)) - n // 2, -n // 2
        X, Y = np.meshgrid((oi + np.arange(n)) * res, (oj + np.arange(n)) * res)
        known = X < 4.3                                          # front face seen, back unseen
        frame = EvidenceFrame((np.abs(X - 4.5) <= 0.3) & (np.abs(Y) <= 0.3) & known, known, oi * res, oj * res, res)
        msgs = []
        for k in range(40):
            out = lg.update(RthInputs(0.1 * k, 2.6, 0.0, 0.0, 0.0, 0.0, battery_pct=80.0,
                                      command="rth" if k == 5 else None, obstacle_frame=frame))
            msgs.append(lg.message)
            if out.plan:
                return np.asarray(out.plan), msgs
        raise AssertionError(msgs[-5:])

    P, msgs = first_plan(True)
    dense = np.asarray([q for a, b in zip(P[:-1], P[1:]) for q in np.linspace(a, b, 20)])
    assert not np.any((dense[:, 0] >= 5.4) & (dense[:, 0] <= 6.0) & (np.abs(dense[:, 1]) >= 0.5))
    assert any("near side" in m for m in msgs)
    P, msgs = first_plan(False)
    dense = np.asarray([q for a, b in zip(P[:-1], P[1:]) for q in np.linspace(a, b, 20)])
    assert np.any((dense[:, 0] >= 5.4) & (dense[:, 0] <= 6.0) & (np.abs(dense[:, 1]) >= 0.5))
