import math
import os
import tempfile

import numpy as np

from dusan_common import geometry, gauge_design, robot_model
from dusan_common.camera import PinholeCamera, camera_world_pose, world_to_camera_link
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout
from dusan_common.planning import plan, path_length
from dusan_common.pure_pursuit import PurePursuit


def test_euler_roundtrip():
    for r, p, y in [(0.1, -0.2, 2.5), (-1.0, 0.4, -3.0), (0.0, 0.0, 0.0)]:
        q = geometry.euler_to_quat(r, p, y)
        r2, p2, y2 = geometry.quat_to_euler(*q)
        assert np.allclose([r, p, y], [r2, p2, y2], atol=1e-9)
        R = geometry.euler_to_rot(r, p, y)
        assert np.allclose(R, geometry.quat_to_rot(*q), atol=1e-9)
        assert np.allclose(geometry.rot_to_euler(R), (r, p, y), atol=1e-9)


def test_gauge_angle_mapping():
    for v in (0.0, 2.5, 5.0, 10.0):
        phi = gauge_design.value_to_angle(v, 0, 10)
        assert abs(gauge_design.angle_to_value(phi, 0, 10) - v) < 1e-9
        assert abs(gauge_design.angle_to_value(phi + 360, 0, 10) - v) < 1e-9
    assert gauge_design.value_to_angle(0, 0, 10) == 225.0
    assert gauge_design.value_to_angle(10, 0, 10) == -45.0


def test_gauge_render():
    img = gauge_design.render_gauge_face(6.2, 0, 10, "bar", "TURBINE-PI-01", size=512)
    assert img.shape == (512, 512, 3)
    # needle pixels exist along the needle direction
    phi = math.radians(gauge_design.value_to_angle(6.2, 0, 10))
    u = int(256 + 0.25 * 512 * math.cos(phi))
    v = int(256 - 0.25 * 512 * math.sin(phi))
    assert img[v, u].mean() < 60


def test_grid_and_astar():
    occ = np.zeros((60, 100), np.int8)
    occ[:, 0] = occ[:, -1] = occ[0, :] = occ[-1, :] = 100
    occ[0:45, 50] = 100  # wall with a gap at the top
    g = OccupancyGridMap(occ, 0.1, -5.0, -3.0)
    with tempfile.TemporaryDirectory() as d:
        g.save(os.path.join(d, "m.yaml"))
        g2 = OccupancyGridMap.load(os.path.join(d, "m.yaml"))
    assert np.array_equal(g.occ, g2.occ)
    p = plan(g2, (-3.0, -1.0), (3.0, -1.0), robot_radius=0.25)
    assert p is not None
    assert path_length(p) > 6.0
    for x, y in p:
        assert g2.is_free(x, y, 0.2)


def test_pure_pursuit_reaches_goal():
    pp = PurePursuit()
    path = [(0, 0), (1, 0), (2, 0.5), (3, 1.0)]
    pp.set_path(path, goal_yaw=0.0)
    x = y = yaw = 0.0
    done = False
    for _ in range(2000):
        vx, vy, wz, done = pp.compute(x, y, yaw)
        if done:
            break
        dt = 0.05
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        yaw += wz * dt
    assert done
    assert math.hypot(x - 3, y - 1) < 0.3


def test_camera_projection():
    cam = PinholeCamera.from_sensor("inspection_camera")
    T = camera_world_pose((0, 0, 0.34, 0, 0, 0), "inspection_camera")
    # a point straight ahead along the optical axis
    ax = T[:3, :3] @ np.array([1.0, 0, 0])
    pw = T[:3, 3] + 2.0 * ax
    uv, depth = cam.project_link(world_to_camera_link(T, pw))
    assert np.allclose(uv[0], (cam.cx, cam.cy), atol=1e-6)
    assert abs(depth[0] - 2.0) < 1e-9
    # positive pitch in SENSORS = looking down; inspection camera pitch -0.3 looks up
    assert ax[2] > 0


def test_layout_loads():
    lay = Layout.load()
    assert len(lay.gauges) == 8
    assert lay.gauge("G02")["max"] == 16.0
    assert lay.gauge_status("G02", 11.4) == "warning"
    assert lay.temperature_status(372.15) == "alarm"
    assert abs(lay.floor_z(60.0, 0.0) + 0.88) < 1e-9
    assert len(robot_model.JOINT_NAMES) == 12


def test_map_yaml_is_written_with_lf(tmp_path):
    """The generated nav map is committed: LF on every OS (no CRLF from Windows text mode)."""
    g = OccupancyGridMap(np.zeros((4, 5), np.int8), 0.1, -1.0, -2.0)
    g.save(str(tmp_path / "m.yaml"))
    raw = (tmp_path / "m.yaml").read_bytes()
    assert b"\r\n" not in raw and raw.count(b"\n") >= 5
    here = os.path.dirname(os.path.abspath(__file__))
    committed = os.path.join(here, "..", "..", "simulation", "maps", "plant_map.yaml")
    if os.path.exists(committed):
        with open(committed, "rb") as f:
            assert b"\r\n" not in f.read()
