"""Novel-obstacle layer on the REAL perception pipeline, pure python (no gz):
the headless check's DepthCamera ray-casts the collisions of plant.sdf (plus
a novel box that is not in plant_map.yaml), TerrainMapperSim runs the
terrain_mapper code (ElevationMap / PointFilter, terrain.yaml), and the
published cloud goes through the same conversion as the nodes
(EvidenceFrame.from_elevation_cloud) into dusan_common.obstacle_layer.

* P1 / P2: driving the real route dock -> G08 -> ramp -> upper corridor
  without a new object, the step source (default) reports no obstacle near
  the route, while the traversability cost-100 source does report some
  (planar slopes of the pipe flanks / ramp edges) - why it is not the default;
* P3: a 0.6 x 0.6 x 0.5 m box is first flagged ~1.2 m ahead and confirmed
  about 1 m ahead (camera pitched 0.6 rad down);
* P4: the mission in the loop (kinematic robot) bypasses the box using this
  emulated depth only.
"""
import math
import os

import numpy as np
import pytest

from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout
from dusan_common.obstacle_layer import EvidenceFrame, ObstacleMemory, ObstacleParams
from dusan_common.obstacle_sim import Box
from dusan_common.planning import densify

from gazebo_check import world as W
from gazebo_check.mapper import DepthCamera, TerrainMapperSim

PLANT_MAP = os.path.join(W.ROOT, "simulation", "maps", "plant_map.yaml")
BOX = {"name": "novel_box_0", "center": [43.0, 0.72], "size": [0.6, 0.6, 0.5], "floor_z": 0.0}
BASE_H = 0.34                                   # base_link above the floor while walking
pytestmark = pytest.mark.skipif(not os.path.exists(PLANT_MAP), reason="simulation/maps/plant_map.yaml missing")


@pytest.fixture(scope="module")
def plant():
    shapes = W.collision_shapes(W.plant_world_tree())
    static = OccupancyGridMap.load(PLANT_MAP)
    hf = W.HeightField(shapes, (35.0, 64.0), (-2.5, 2.5), res=0.05)
    return shapes, static, hf, Layout.load(W.LAYOUT)


def _route(layout, static):
    from plant_simulation.mission_logic import MissionLogic, fallback_viewpoints
    m = MissionLogic({}, layout)
    m.set_map(static)
    vps = fallback_viewpoints(layout, static, (61.9, 0.0))
    m.set_pose(61.9, 0.0, math.pi, 0.0)
    p1 = m._plan((vps[0].x, vps[0].y))
    m.set_pose(vps[0].x, vps[0].y, 0.0, 0.0)
    p2 = m._plan((40.0, 1.1))
    return densify(p1 + p2[1:], 0.1)


def test_p1_p2_no_false_positives_on_the_real_route_with_the_step_source(plant):
    shapes, static, hf, lay = plant
    path = np.asarray(_route(lay, static))
    cam = DepthCamera(shapes)
    mapper = TerrainMapperSim(layout=None)
    step_mem = ObstacleMemory(ObstacleParams(), 0.35)
    trav_mem = ObstacleMemory(ObstacleParams(source="traversability"), 0.35)
    for mem in (step_mem, trav_mem):
        mem.set_static_map(static)
    for k in range(1, len(path)):
        x, y = path[k]
        yaw = math.atan2(path[k][1] - path[k - 1][1], path[k][0] - path[k - 1][0])
        pose = [x, y, hf.ground(x, y) + BASE_H, 0.0, 0.0, yaw]
        t = 0.25 * k                                          # 0.4 m/s
        mapper.integrate_depth(cam.render(pose[:3], pose[3:]), pose, t)
        if k % 2:
            continue
        cloud, _ = mapper.publish(pose, t)
        step_mem.update(EvidenceFrame.from_elevation_cloud(cloud, "odom"), pose, t)
        ox, oy = mapper.emap.occupancy_origin()
        grid = OccupancyGridMap(mapper.emap.occupancy_data(), mapper.emap.resolution, ox, oy, "odom")
        trav_mem.update(EvidenceFrame.from_traversability(grid, 100), pose, t)

    def near_route(mem):
        c = mem.confirmed_xy()
        if not len(c):
            return 0
        d = np.hypot(c[:, None, 0] - path[None, :, 0], c[:, None, 1] - path[None, :, 1]).min(1)
        return int(np.sum(d < 0.35 + 0.10))
    assert step_mem.frames > 80
    assert near_route(step_mem) == 0, step_mem.confirmed_xy()
    assert trav_mem.confirmed_count > 0                     # slope-only cost-100 cells (pipe flanks, ramp edges)


def test_p3_box_detection_range(plant):
    shapes, static, hf, lay = plant
    cam = DepthCamera(shapes + W.extra_shapes([BOX]))
    mapper = TerrainMapperSim(layout=None)
    mem = ObstacleMemory(ObstacleParams(), 0.35)
    mem.set_static_map(static)
    first_raw = first_conf = None
    for k, x in enumerate(np.arange(47.0, 43.9, -0.04)):           # 0.4 m/s, depth at 10 Hz, map at 5 Hz
        pose = [x, 0.72, BASE_H, 0.0, 0.0, math.pi]
        t = 0.1 * k
        mapper.integrate_depth(cam.render(pose[:3], pose[3:]), pose, t)
        if k % 2:
            continue
        cloud, _ = mapper.publish(pose, t)
        mem.update(EvidenceFrame.from_elevation_cloud(cloud, "odom"), pose, t)
        raw = mem.raw_xy(t)
        if first_raw is None and len(raw):
            first_raw = float(np.min(np.hypot(raw[:, 0] - x, raw[:, 1] - 0.72)))
        if first_conf is None and mem.has_confirmed():
            c = mem.confirmed_xy()
            first_conf = float(np.min(np.hypot(c[:, 0] - x, c[:, 1] - 0.72)))
    assert first_raw is not None and 1.0 <= first_raw <= 1.45, first_raw
    assert first_conf is not None and first_conf >= 0.85, first_conf
    c = mem.confirmed_xy()
    assert np.all(np.abs(c[:, 1] - 0.72) <= 0.3 + 0.1 + 0.05) and np.all(np.abs(c[:, 0] - 43.0) <= 0.6)


def test_p4_mission_in_the_loop_with_emulated_depth(plant):
    from dusan_common.terrain_cost import debris_crossed
    from plant_simulation.mission_logic import MissionLogic
    shapes, static, hf, lay = plant
    cam = DepthCamera(shapes + W.extra_shapes([BOX]))
    mapper = TerrainMapperSim(layout=None)
    m = MissionLogic({"min_start_time": 0.5, "scan_on_miss": False}, lay)
    m.set_map(static)
    m.set_plan({"viewpoints": [{"gauge_id": "V1", "x": 39.8, "y": 0.9, "yaw": math.pi}], "order": ["V1"],
                "legs": []})
    box = Box((43.0, 0.72), (0.6, 0.6))
    x, y, yaw, t, dt = 46.5, 0.72, math.pi, 0.0, 0.1
    m.command("patrol", t)
    clear, traj, k = math.inf, [], 0
    while t < 45.0 and m.base not in ("INSPECTING", "DONE"):
        m.set_pose(x, y, yaw, t)
        pose = [x, y, BASE_H, 0.0, 0.0, yaw]
        mapper.integrate_depth(cam.render(pose[:3], pose[3:]), pose, t)
        if k % 2 == 0:
            cloud, _ = mapper.publish(pose, t)
            m.on_obstacle_frame(EvidenceFrame.from_elevation_cloud(cloud, "odom"), t)
        out = m.step(t)
        if out.cmd_vel is not None:
            vx, vy, wz = out.cmd_vel
            c, s = math.cos(yaw), math.sin(yaw)
            x += (vx * c - vy * s) * dt
            y += (vx * s + vy * c) * dt
            yaw = (yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi
        clear = min(clear, float(box.distance(x, y)))
        traj.append((x, y))
        t += dt
        k += 1
    st = m.state_dict(t)["obstacle"]
    assert m.base == "INSPECTING", (m.base, m.message)
    assert st["detours"] >= 1 and m.total_replans == 0 and st["errors"] == 0
    assert clear >= 0.40, clear
    assert debris_crossed(traj, lay) == []


def _footprint_overlap(box, x, y, yaw, hx=0.40, hy=0.19):
    """Does the leg footprint (+-0.40 x +-0.19 m around base_link) touch the box outline?"""
    sx, sy = box.size
    us, vs = np.linspace(-sx / 2, sx / 2, 13), np.linspace(-sy / 2, sy / 2, 13)
    pts = np.concatenate([np.stack([us, np.full_like(us, -sy / 2)], 1), np.stack([us, np.full_like(us, sy / 2)], 1),
                          np.stack([np.full_like(vs, -sx / 2), vs], 1), np.stack([np.full_like(vs, sx / 2), vs], 1)])
    P = pts + np.asarray(box.center)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = P[:, 0] - x, P[:, 1] - y
    return bool(np.any((np.abs(c * dx + s * dy) <= hx) & (np.abs(-s * dx + c * dy) <= hy)))


@pytest.mark.parametrize("case", ["deep_box_upper_corridor", "box_on_the_corridor_entry_bend"])
def test_p5_detour_clears_the_unseen_back_of_the_box(plant, case):
    """Review (behaviour F1): with the real camera the side faces of a box are
    never observed while passing (outside the field of view, then inside the
    own-body radius), so memory held only the front face + 0.3 m: the detour
    rejoined through the unseen back corner (1.0 m deep box: 0.15 m, legs
    overlapping it; box on the bend: 0.17 m).  The occlusion shadow now
    assumes obstacle_shadow_depth along the line of sight."""
    from plant_simulation.mission_logic import MISSION_DEFAULTS, MissionLogic
    shapes, static, hf, lay = plant
    if case == "deep_box_upper_corridor":
        start, vp, (cx, cy, sx, sy), passed, need = (45.0, 0.62, math.pi), (36.5, 0.85, math.pi), \
            (40.6, 0.78, 1.0, 0.6), (lambda x: x < 39.3), MISSION_DEFAULTS["robot_radius"] + 0.10
        hfield = hf
    else:
        start, vp, (cx, cy, sx, sy), passed, need = (24.5, 0.5, math.pi), (3.5, 4.3, math.pi / 2), \
            (22.15, 0.40, 0.6, 0.6), (lambda x: x < 20.6), MISSION_DEFAULTS["robot_radius"] + 0.05
        hfield = W.HeightField(shapes, (18.0, 27.0), (-2.5, 2.5), res=0.05)
    spec = {"name": "novel_box_1", "center": [cx, cy], "size": [sx, sy, 0.5], "floor_z": float(lay.floor_z(cx, cy))}
    cam = DepthCamera(shapes + W.extra_shapes([spec]))
    mapper = TerrainMapperSim(layout=None)
    m = MissionLogic({"min_start_time": 0.5, "scan_on_miss": False}, lay)
    m.set_map(static)
    m.set_plan({"viewpoints": [{"gauge_id": "V", "x": vp[0], "y": vp[1], "yaw": vp[2]}], "order": ["V"], "legs": []})
    box = Box((cx, cy), (sx, sy))
    x, y, yaw = start
    t, dt, k, clear, overlap = 0.0, 0.1, 0, math.inf, 0
    m.command("patrol", t)
    while t < 60.0 and m.base not in ("INSPECTING", "DONE") and not passed(x):
        m.set_pose(x, y, yaw, t)
        pose = [x, y, hfield.ground(x, y) + BASE_H, 0.0, 0.0, yaw]
        mapper.integrate_depth(cam.render(pose[:3], pose[3:]), pose, t)
        if k % 2 == 0:
            cloud, _ = mapper.publish(pose, t)
            m.on_obstacle_frame(EvidenceFrame.from_elevation_cloud(cloud, "odom"), t)
        out = m.step(t)
        if out.cmd_vel is not None:
            vx, vy, wz = out.cmd_vel
            c, s = math.cos(yaw), math.sin(yaw)
            x += (vx * c - vy * s) * dt
            y += (vx * s + vy * c) * dt
            yaw = (yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi
        clear = min(clear, float(box.distance(x, y)))
        overlap += _footprint_overlap(box, x, y, yaw)
        t += dt
        k += 1
    st = m.state_dict(t)["obstacle"]
    assert passed(x), (m.base, m.message, x, y)
    assert st["detours"] >= 1 and m.total_replans == 0 and st["errors"] == 0
    assert overlap == 0 and clear >= need, (clear, overlap)
