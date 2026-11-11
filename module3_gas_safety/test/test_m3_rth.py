"""Return-to-home supervisor: triggers and full state progression with a fake robot."""
import math

import numpy as np
import pytest

from module3_gas_safety.battery_model import BatteryConfig, BatteryModel, DockDetector
from module3_gas_safety.comm_model import CommModel
from module3_gas_safety.rth_logic import (CHARGING, DOCKED, DOCKING, MONITOR, RTH_NAVIGATING, RTH_PLANNING,
                                          GeodesicField, PlannerWorker, RthConfig, RthInputs, RthLogic, SyncPlanner,
                                          path_ahead)
from module3_gas_safety.sim_harness import RobotSim


@pytest.fixture(scope="module")
def world(layout, layout_grid):
    rmap = CommModel.from_layout(layout, layout_grid).compute_map(0.5)
    geo = GeodesicField(layout_grid, layout.dock_pose[:2], 0.3)
    return layout, layout_grid, rmap, geo


def _logic(world, battery=None, **cfg):
    layout, grid, rmap, geo = world
    base = dict(battery_debounce_s=0.0, energy_debounce_s=0.0)
    base.update(cfg)
    return RthLogic(RthConfig(**base), grid, layout.dock_pose, battery or BatteryModel(),
                    rssi_lookup=rmap.at, home_distance=geo.distance)


def test_path_ahead_samples_from_projection():
    path = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    pts, off = path_ahead(path, 8.0, 1.0, 5.0, 0.5)
    assert off == pytest.approx(1.0)
    assert np.allclose(pts[0], [8.0, 0.0])
    assert np.allclose(pts[-1], [10.0, 3.0], atol=1e-6)        # 2 m + 3 m around the corner
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    assert np.all(d <= 0.5 + 1e-9)


def test_geodesic_distance_home(world):
    layout, grid, _, geo = world
    dock = layout.dock_pose
    assert geo.distance(dock[0], dock[1]) < 0.3
    d_hall = geo.distance(0.0, 0.0)
    assert d_hall >= math.hypot(dock[0], dock[1]) - 0.5
    assert d_hall < 1.15 * math.hypot(dock[0], dock[1])
    # around the turbine: longer than the straight line
    assert geo.distance(-3.0, -7.5) > math.hypot(dock[0] + 3.0, dock[1] + 7.5)


def test_battery_low_trigger_with_debounce(world):
    lg = _logic(world, battery_debounce_s=1.0)
    out = lg.update(RthInputs(0.0, 30.0, 0.0, 0.0, battery_pct=19.0))
    assert lg.state == MONITOR and not out.rth_active
    lg.update(RthInputs(0.5, 30.0, 0.0, 0.0, battery_pct=21.0))       # glitch resets the timer
    lg.update(RthInputs(0.6, 30.0, 0.0, 0.0, battery_pct=19.0))
    lg.update(RthInputs(1.5, 30.0, 0.0, 0.0, battery_pct=19.0))
    assert lg.state == MONITOR
    out = lg.update(RthInputs(1.7, 30.0, 0.0, 0.0, battery_pct=19.0))
    assert lg.reason == "battery_low" and out.rth_active
    assert lg.state == RTH_NAVIGATING and out.plan          # synchronous planner answered at once


def test_energy_reserve_trigger(world):
    bat = BatteryModel(BatteryConfig(capacity_wh=40.0))
    lg = _logic(world, battery=bat)
    lg.update(RthInputs(0.0, -1.0, -1.5, 0.0, battery_pct=40.0))      # hall, ~63 m from the dock
    assert lg.state == MONITOR
    need = lg.energy_to_home_pct * 1.3 + 3.0
    assert 22.0 < need < 40.0
    lg.update(RthInputs(1.0, -1.0, -1.5, 0.0, battery_pct=need - 1.0))
    assert lg.reason == "energy_reserve"
    # near the dock the reserve check is not needed
    lg2 = _logic(world, battery=BatteryModel(BatteryConfig(capacity_wh=40.0)))
    lg2.update(RthInputs(0.0, 61.5, 0.0, 0.0, battery_pct=21.0))
    assert lg2.state == MONITOR


def test_comm_shadow_predicted_along_plan(world):
    _, _, rmap, _ = world
    lg = _logic(world)
    plan = [(x, -7.5) for x in np.arange(0.0, -8.01, -0.25)]
    edge = -6.0                                   # east edge of zone S1
    trig_x = None
    for k, x in enumerate(np.arange(0.0, -6.0, -0.1)):
        out = lg.update(RthInputs(0.1 * k, float(x), -7.5, math.pi, battery_pct=80.0, rssi_dbm=float(rmap.at(x, -7.5)),
                                  plan=plan if k == 0 else None))
        if out.rth_active:
            trig_x = float(x)
            break
    assert lg.reason == "comm_shadow_predicted"
    assert rmap.at(trig_x, -7.5) > -80.0                  # still connected when triggering
    assert 4.0 <= trig_x - edge <= 5.3                     # ~5 m look-ahead


def test_comm_shadow_predicted_from_velocity(world):
    _, _, rmap, _ = world
    lg = _logic(world)
    x0 = -4.8
    assert rmap.at(x0, -7.5) > -80.0
    lg.update(RthInputs(0.0, x0, -7.5, math.pi, vx=0.0, vy=0.0, battery_pct=80.0))
    assert lg.state == MONITOR
    lg.update(RthInputs(0.1, x0, -7.5, math.pi, vx=-0.4, vy=0.0, battery_pct=80.0))
    assert lg.reason == "comm_shadow_predicted"


def test_manual_and_comm_lost_triggers(world):
    lg = _logic(world)
    lg.update(RthInputs(0.0, 30.0, 0.0, 0.0, battery_pct=90.0, command="rth"))
    assert lg.reason == "manual" and lg.active
    lg2 = _logic(world, comm_shadow_enabled=False)
    for k in range(21):
        lg2.update(RthInputs(0.1 * k, 30.0, 0.0, 0.0, battery_pct=90.0, rssi_dbm=-90.0))
    assert lg2.state == MONITOR                            # below -80 dBm for 2.0 s: not yet
    lg2.update(RthInputs(2.1, 30.0, 0.0, 0.0, battery_pct=90.0, rssi_dbm=-90.0))
    assert lg2.reason == "comm_lost"


def _run_fake_robot(world, start, t_max=900.0, battery_pct=19.0, resume_at=None, freeze=False):
    layout, grid, rmap, geo = world
    bat = BatteryModel()
    bat.set_level(battery_pct)
    lg = RthLogic(RthConfig(), grid, layout.dock_pose, bat, rmap.at, geo.distance)
    robot = RobotSim(start[0], start[1], start[2], max_speed=0.45)
    dock = DockDetector(layout.dock_pose[:2], 0.35)
    loco = []
    states = [lg.state]
    t = 0.0
    dt = 0.1
    while t < t_max:
        t += dt
        sp = math.hypot(robot.vx_w, robot.vy_w)
        docked = dock.update(t, robot.x, robot.y, sp)
        bat.step(dt, sp, robot.wz, "sit" if lg.state in (DOCKED, CHARGING) else "walk", docked)
        cmd = "resume" if (resume_at is not None and lg.state == CHARGING and bat.percentage >= resume_at) else None
        out = lg.update(RthInputs(t, robot.x, robot.y, robot.yaw, robot.vx_w, robot.vy_w, bat.percentage,
                                  bat.charging, float(rmap.at(robot.x, robot.y)), command=cmd))
        if out.locomotion:
            loco.append(out.locomotion)
        if not freeze:
            robot.step(out.cmd or (0.0, 0.0, 0.0), dt, grid)
        if lg.state != states[-1]:
            states.append(lg.state)
        if states[-1] == MONITOR and len(states) > 1:
            break
        if freeze and lg.replans >= 1:
            break
    return lg, robot, bat, states, loco


def test_full_rth_progression_with_fake_robot(world):
    layout = world[0]
    lg, robot, bat, states, loco = _run_fake_robot(world, (40.0, 0.5, math.pi))
    seq = [s for _, s, _ in lg.transitions]
    assert seq == [RTH_PLANNING, RTH_NAVIGATING, DOCKING, DOCKED, CHARGING, MONITOR]
    assert {r for _, _, r in lg.transitions[:-1]} == {"battery_low"}
    assert loco == ["sit", "stand"]
    assert bat.percentage >= 95.0
    dock = layout.dock_pose
    assert math.hypot(robot.x - dock[0], robot.y - dock[1]) < 0.2
    assert abs((robot.yaw - dock[2] + math.pi) % (2 * math.pi) - math.pi) < 0.2
    assert robot.contacts == 0
    assert not lg.active and lg.reason is None


def test_resume_while_charging_leaves_immediately(world):
    lg, robot, bat, states, loco = _run_fake_robot(world, (55.0, -0.5, 0.0), resume_at=30.0)
    assert states[-2:] == [CHARGING, MONITOR]
    assert loco == ["sit", "stand"]
    assert bat.percentage < 40.0


def test_replan_when_stuck(world):
    lg, robot, bat, states, loco = _run_fake_robot(world, (30.0, 0.0, 0.0), t_max=30.0, freeze=True)
    assert lg.replans >= 1
    assert RTH_PLANNING in states[2:] or lg.state in (RTH_PLANNING, RTH_NAVIGATING)


def test_velocity_safety_filter(world):
    lg = _logic(world)
    cfg = lg.cfg
    # corridor near the north wall (y = 1.8), heading north (into the wall) at 0.3 m/s
    for y in (1.45, 1.55, 1.65, 1.72):
        clr, n = lg._clearance(30.0, y)
        assert clr < cfg.safety_dist and n[1] < -0.9          # repulsion direction points south
        vx, vy, wz = lg._safe_cmd(30.0, y, math.pi / 2, (0.3, 0.0, 0.2))
        vy_world = vx                                         # yaw = +90 deg: body x = world +y
        k = min(1.0, max(0.0, (clr - 0.5 * cfg.robot_radius) / (cfg.safety_dist - 0.5 * cfg.robot_radius)))
        assert vy_world == pytest.approx(0.3 * k - cfg.safety_repulse * (1.0 - k), abs=0.02)
        assert wz == 0.2
        if clr <= 0.5 * cfg.robot_radius:
            assert vy_world < 0.0                             # pushed away, never further in
    # walking along the wall is kept
    vx2, vy2, _ = lg._safe_cmd(30.0, 1.5, 0.0, (0.3, 0.0, 0.0))
    assert vx2 == pytest.approx(0.3, abs=0.02)
    # far from obstacles the command is unchanged
    assert lg._safe_cmd(30.0, 0.0, 0.3, (0.3, 0.05, 0.1)) == (0.3, 0.05, 0.1)


def test_backoff_from_obstacle_before_replanning(world):
    layout, grid, rmap, geo = world
    lg = _logic(world)
    x, y, yaw = 30.0, 1.45, 0.0                  # pressed against the corridor wall, robot frozen
    lg.update(RthInputs(0.0, x, y, yaw, battery_pct=90.0, command="rth"))
    assert lg.state == RTH_NAVIGATING
    t = 0.0
    while lg.replans == 0 and t < 30.0:
        t += 0.1
        lg.update(RthInputs(t, x, y, yaw, battery_pct=90.0))
    assert lg.replans == 1 and lg.state == RTH_PLANNING
    out = lg.update(RthInputs(t + 0.1, x, y, yaw, battery_pct=90.0))
    assert lg.state == RTH_PLANNING and out.cmd[1] < -0.05   # body y = world y: away from the wall
    out = lg.update(RthInputs(t + 2.5, x, y - 0.4, yaw, battery_pct=90.0))   # backed off -> plans again
    assert lg.state == RTH_NAVIGATING and out.plan


def test_velocity_extrapolation_stops_at_walls():
    """Walking towards a wall must not 'predict' the attenuated RSSI behind it."""
    from dusan_common.grid_map import OccupancyGridMap
    occ = np.zeros((60, 60), dtype=np.int8)               # 6 m x 6 m, origin (-3, -3)
    occ[40:42, :] = 100                                   # wall y = 1.0 .. 1.2
    grid = OccupancyGridMap(occ, 0.1, -3.0, -3.0)

    def rssi(xs, ys):                                     # deep shadow in and behind the wall
        return np.where(np.asarray(ys) >= 0.95, -100.0, -50.0)

    lg = RthLogic(RthConfig(battery_debounce_s=0.0, energy_debounce_s=0.0, comm_lost_enabled=False),
                  grid, (0.0, -2.5, 0.0), BatteryModel(), rssi_lookup=rssi, home_distance=lambda x, y: 3.0)
    out = lg.update(RthInputs(0.0, 0.0, 0.5, math.pi / 2, vx=0.0, vy=0.4, battery_pct=90.0))
    assert lg.state == MONITOR and not out.rth_active
    assert lg.predicted_min_rssi == pytest.approx(-50.0)
    # the same motion without the wall reaches the shadow within 4 s -> trigger
    lg2 = RthLogic(RthConfig(comm_lost_enabled=False), OccupancyGridMap(np.zeros((60, 60), dtype=np.int8),
                                                                        0.1, -3.0, -3.0),
                   (0.0, -2.5, 0.0), BatteryModel(), rssi_lookup=rssi, home_distance=lambda x, y: 3.0)
    lg2.update(RthInputs(0.0, 0.0, 0.5, math.pi / 2, vx=0.0, vy=0.4, battery_pct=90.0))
    assert lg2.reason == "comm_shadow_predicted"
    # plan samples inside obstacles are ignored as well
    lg3 = RthLogic(RthConfig(comm_lost_enabled=False), grid, (0.0, -2.5, 0.0), BatteryModel(),
                   rssi_lookup=rssi, home_distance=lambda x, y: 3.0)
    plan = [(0.0, 0.5), (0.0, 1.05)]                      # last sample (y = 1.0) lies inside the wall
    from module3_gas_safety.rth_logic import path_ahead as _pa
    assert np.any(_pa(plan, 0.0, 0.5, 5.0, 0.25)[0][:, 1] >= 0.95)
    lg3.update(RthInputs(0.0, 0.0, 0.5, 0.0, battery_pct=90.0, plan=plan))
    assert lg3.state == MONITOR and lg3.predicted_min_rssi == pytest.approx(-50.0)


# ----------------------------------------------------------------------------
# operator commands while returning: abort / pause / resume
# ----------------------------------------------------------------------------
def _inp(t, x=30.0, y=0.0, yaw=0.0, **kw):
    kw.setdefault("battery_pct", 90.0)
    kw.setdefault("rssi_dbm", -50.0)
    return RthInputs(t, x, y, yaw, **kw)


def test_abort_cancels_a_manual_return(world):
    lg = _logic(world)
    out = lg.update(_inp(0.0, command="rth"))
    assert lg.state == RTH_NAVIGATING and out.rth_active and out.plan
    lg.update(_inp(0.1))
    out = lg.update(_inp(0.2, command="abort"))
    assert lg.state == MONITOR and not out.rth_active and lg.reason is None
    assert out.plan == []                          # latched home path replaced by an empty one
    assert out.locomotion is None                  # gait left alone (no 'stand' while walking)
    assert out.status["state"] == MONITOR and out.status["paused"] is False
    for k in range(3, 60):                         # stays cancelled (no trigger condition holds)
        out = lg.update(_inp(0.1 * k))
        assert not out.rth_active and out.plan is None
    # a later 'rth' starts a fresh return from the present position
    out = lg.update(_inp(6.0, x=25.0, command="rth"))
    assert lg.state == RTH_NAVIGATING and abs(out.plan[0][0] - 25.0) < 0.5


def test_abort_refused_while_a_trigger_condition_holds_and_pause_holds(world):
    lg = _logic(world)
    lg.update(_inp(0.0, battery_pct=19.0))
    assert lg.reason == "battery_low" and lg.state == RTH_NAVIGATING
    out = lg.update(_inp(0.1, battery_pct=19.0, command="abort"))
    assert lg.state == RTH_NAVIGATING and out.rth_active and "abort refused" in lg.message
    assert "battery_low" in lg.message and out.plan is None
    # pause: zero command, still active (the mux keeps rth), no stuck re-plan however long
    out = lg.update(_inp(0.2, battery_pct=19.0, command="pause"))
    assert lg.paused and out.cmd == (0.0, 0.0, 0.0) and out.rth_active and out.status["paused"]
    for k in range(3, 300):
        out = lg.update(_inp(0.1 * k, battery_pct=19.0))
        assert out.cmd == (0.0, 0.0, 0.0) and out.rth_active
    assert lg.replans == 0 and lg.state == RTH_NAVIGATING
    # resume after the robot was moved (teleop): re-plan from the present pose and drive on
    out = lg.update(_inp(30.0, x=27.0, y=0.5, battery_pct=19.0, command="resume"))
    assert not lg.paused and lg.state == RTH_NAVIGATING and out.plan
    assert math.hypot(out.plan[0][0] - 27.0, out.plan[0][1] - 0.5) < 0.5
    out = lg.update(_inp(30.1, x=27.0, y=0.5, battery_pct=19.0))
    assert out.cmd is not None and abs(out.cmd[0]) + abs(out.cmd[1]) + abs(out.cmd[2]) > 1e-3
    # condition gone (battery set back to 80 % by the test override): abort cancels
    out = lg.update(_inp(30.2, x=27.0, y=0.5, battery_pct=80.0, command="abort"))
    assert lg.state == MONITOR and not out.rth_active


def test_abort_refused_while_comm_lost(world):
    lg = _logic(world, comm_shadow_enabled=False)
    lg.update(_inp(0.0, command="rth"))
    assert lg.active
    lg.update(_inp(0.1, rssi_dbm=-90.0, command="abort"))
    assert lg.active and "comm_lost" in lg.message
    lg.update(_inp(0.2, rssi_dbm=-60.0, command="abort"))
    assert not lg.active


def test_resume_without_pause_and_abort_on_the_dock_are_ignored(world):
    dx, dy, dyaw = world[0].dock_pose
    lg = _logic(world)
    lg.update(RthInputs(0.0, dx, dy, dyaw, battery_pct=50.0, command="rth"))   # already on the dock
    t = 0.0
    while lg.state != CHARGING and t < 20.0:
        t += 0.1
        lg.update(RthInputs(t, dx, dy, dyaw, battery_pct=50.0, charging=True))
    assert lg.state == CHARGING
    out = lg.update(RthInputs(t + 0.1, dx, dy, dyaw, battery_pct=50.0, charging=True, command="abort"))
    assert lg.state == CHARGING and out.rth_active             # charging is not cancelled by 'abort'
    out = lg.update(RthInputs(t + 0.2, dx, dy, dyaw, battery_pct=50.0, charging=True, command="resume"))
    assert lg.state == MONITOR and out.locomotion == "stand" and out.plan == []
    # 'resume' while returning but not paused does nothing
    lg2 = _logic(world)
    lg2.update(_inp(0.0, command="rth"))
    lg2.update(_inp(0.1, command="resume"))
    assert lg2.state == RTH_NAVIGATING and not lg2.paused and lg2.replans == 0
    # 'pause' / 'abort' in MONITOR are the mission's business
    lg3 = _logic(world)
    lg3.update(_inp(0.0, command="pause"))
    lg3.update(_inp(0.1, command="abort"))
    assert lg3.state == MONITOR and not lg3.paused


def test_planner_worker_cancel_discards_a_late_result(world):
    layout, grid, _, _ = world
    pw = PlannerWorker(grid, 0.3, 0.4)
    pw.submit((30.0, 0.0), layout.dock_pose[:2])
    pw.cancel()
    pw._thread.join(20.0)
    assert pw.poll() is None                       # the cancelled request never answers
    pw.submit((30.0, 0.0), layout.dock_pose[:2])
    pw._thread.join(20.0)
    ok, path = pw.poll()
    assert ok and len(path) >= 2
    sp = SyncPlanner(grid)
    sp.submit((30.0, 0.0), layout.dock_pose[:2])
    sp.cancel()
    assert sp.poll() is None


def test_path_home_takes_the_terrain_detours():
    """Terrain cost (dusan_common.terrain_cost) on the generated nav map: the path home from
    the upper corridor goes through the C-OBS3 north passage and around the north end of the
    C-OBS1 pipe - both are crossed without the layer."""
    import os
    from dusan_common import terrain_cost as tc
    from dusan_common.grid_map import OccupancyGridMap
    from dusan_common.layout import Layout
    from module3_gas_safety.params import RTH_PARAMS, rth_config
    from module3_gas_safety.rth_logic import terrain_cost_for
    lay = Layout.load()
    if not os.path.exists(lay.nav_map_yaml()):
        pytest.skip("simulation/maps/plant_map.yaml not generated")
    grid = OccupancyGridMap.load(lay.nav_map_yaml())
    cfg = rth_config(dict(RTH_PARAMS), lay)
    assert cfg.terrain_cost_enabled and cfg.terrain_w_rubble > cfg.terrain_w_plate > 0.0
    fn = terrain_cost_for(cfg, lay)
    r, rs = cfg.robot_radius, cfg.robot_radius + cfg.smooth_margin
    start, dock = (34.0, 0.0), lay.dock_pose[:2]
    plain, aware = SyncPlanner(grid, r, rs), SyncPlanner(grid, r, rs, cost_fn=fn)
    plain.submit(start, dock)
    aware.submit(start, dock)
    (ok0, p0), (ok1, p1) = plain.poll(), aware.poll()
    assert ok0 and ok1
    assert {"C-OBS1", "C-OBS3"} <= set(tc.debris_crossed(p0, lay))
    assert tc.debris_crossed(p1, lay) == []
    # the background worker and the logic's default planner use the same layer
    pw = PlannerWorker(grid, r, rs, cost_fn=fn)
    pw.submit(start, dock)
    pw._thread.join(30.0)
    ok2, p2 = pw.poll()
    assert ok2 and tc.debris_crossed(p2, lay) == []
    assert RthLogic(cfg, grid, lay.dock_pose, terrain_cost=fn).planner.cost_fn is fn
    assert terrain_cost_for(rth_config(dict(RTH_PARAMS, terrain_cost_enabled=False), lay), lay) is None
    assert terrain_cost_for(cfg, None) is None
