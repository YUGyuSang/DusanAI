"""Unit tests of module2_inspection.synthetic_thermal (ROS free)."""
import math
import time

import cv2
import numpy as np
import pytest

from dusan_common.camera import camera_world_pose, world_to_camera_link
from dusan_common.layout import Layout
from module2_inspection.synthetic_thermal import (
    LBL_HOT, LBL_MACHINE, LBL_OBSTACLE, LBL_PIPE, SyntheticThermalCamera, ThermalScene, full_pose,
    render_visible_proxy)

LAYOUT = Layout.load()


def _mini_layout(machine=False, debris=False):
    """Small synthetic hall with one hot flange at (3, 0, 0.8) facing a camera at x=0."""
    d = {
        "factory": {"interior": {"x_min": -2.0, "x_max": 8.0, "y_min": -4.0, "y_max": 4.0}, "height": 5.0, "floor_z": 0.0},
        "thermal": {"ambient_k": 298.15, "warning_k": 333.15, "alarm_k": 353.15, "pipe_body_k": 308.15},
        "hotspots": [{"id": "HX", "tag": "TEST", "kind": "flange", "position": [3.0, 0.0, 0.8], "axis": [0, 1, 0],
                      "radius": 0.2, "length": 0.1, "temp_k": 360.0}],
        "machines": [], "obstacles": [], "gauges": [],
    }
    if machine:
        d["machines"] = [{"id": "M", "model": "box", "footprint": {"x_min": 1.5, "x_max": 2.0, "y_min": -1.0, "y_max": 1.0}}]
    if debris:   # a cold lying pipe right in front of the flange (splat vs splat occlusion)
        d["obstacles"] = [{"id": "P", "type": "pipe", "center": [2.0, 0.0], "floor_z": 0.6, "yaw": 1.5708,
                           "length": 2.0, "diameter": 0.6}]
    return Layout(d)


POSE0 = (0.0, 0.0, 0.34, 0.0, 0.0, 0.0)


def _hot_pixel(cam: SyntheticThermalCamera, pos):
    T_wc = camera_world_pose(POSE0, "thermal_camera")
    uv, _ = cam.cam.project_link(world_to_camera_link(T_wc, np.asarray(pos)[None]))
    return int(uv[0, 0]), int(uv[0, 1])


def test_output_format_and_background():
    cam = SyntheticThermalCamera(LAYOUT, seed=1)
    pose = (30.0, 0.0, math.pi / 2)            # corridor, looking at the north wall
    l16 = cam.render_l16(pose)
    assert l16.shape == (240, 320) and l16.dtype == np.uint16
    K = l16.astype(np.float64) * 0.01
    assert abs(np.median(K) - 298.15) < 3.0    # walls / floor around ambient
    assert K.max() < 310.0                      # nothing hot in view


def test_hotspot_rendered_at_projection_with_temp():
    cam = SyntheticThermalCamera(LAYOUT, seed=2)
    pose = full_pose(LAYOUT, 5.0, 4.4, math.pi / 2)
    K = cam.render_kelvin(pose)
    buf = cam.last_buffers
    h = LAYOUT.hotspots[1]                      # H02, 372.15 K
    T_wc = camera_world_pose(pose, "thermal_camera")
    uv, depth = cam.cam.project_link(world_to_camera_link(T_wc, np.asarray(h["position"])[None]))
    u, v = int(uv[0, 0]), int(uv[0, 1])
    assert 0 <= u < 320 and 0 <= v < 240 and 1.5 < depth[0] < 2.5
    win = K[v - 5:v + 6, u - 5:u + 6]
    assert abs(win.max() - h["temp_k"]) < 1.5
    assert np.any(buf.label[v - 5:v + 6, u - 5:u + 6] == LBL_HOT + 1)
    assert np.any(buf.label == LBL_PIPE)        # pipe body visible as well
    pipe_T = buf.temp[buf.label == LBL_PIPE]
    assert 305.0 < np.median(pipe_T) < 320.0    # pipe body ~ pipe_body_k + halo


def test_zbuffer_box_occludes_hotspot():
    vis = SyntheticThermalCamera(_mini_layout(), seed=0)
    occ = SyntheticThermalCamera(_mini_layout(machine=True), seed=0)
    u, v = _hot_pixel(vis, [3.0, 0.0, 0.8])
    Kv = vis.render_kelvin(POSE0, noise=False)
    Ko = occ.render_kelvin(POSE0, noise=False)
    assert Kv[v, u] > 355.0                                 # visible: ~temp_k
    assert Ko.max() < 333.15                                # analytic box hides it
    assert occ.last_buffers.label[v, u] == LBL_MACHINE


def test_zbuffer_splat_occludes_splat():
    cam = SyntheticThermalCamera(_mini_layout(debris=True), seed=0)
    u, v = _hot_pixel(cam, [3.0, 0.0, 0.8])
    K = cam.render_kelvin(POSE0, noise=False)
    assert cam.last_buffers.label[v, u] == LBL_OBSTACLE     # nearer cold pipe wins
    assert K[v, u] < 320.0


def test_splat_surface_has_no_holes():
    cam = SyntheticThermalCamera(_mini_layout(), seed=0)
    cam.render_kelvin((1.8, 0.0, 0.34, 0.0, 0.0, 0.0), noise=False)   # 1.2 m: big flange in view
    lab = cam.last_buffers.label
    m = (lab == LBL_HOT).astype(np.uint8)
    assert m.sum() > 2000
    # the silhouette is convex (perspective view of the cylinder side): every
    # pixel inside its (slightly eroded) convex hull must be covered by splats
    pts = cv2.findNonZero(m)
    hull = np.zeros_like(m)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    inner = cv2.erode(hull, np.ones((5, 5), np.uint8)).astype(bool)
    assert m[inner].mean() > 0.999


def test_noise_model_statistics():
    cam = SyntheticThermalCamera(LAYOUT, seed=5, netd_k=0.05, column_fpn_k=0.0, row_noise_k=0.0, blur_sigma_px=0.0)
    pose = (30.0, 0.0, math.pi / 2)
    a = cam.render_kelvin(pose)
    b = cam.render_kelvin(pose)
    diff = (a - b)[60:180, 80:240]
    assert 0.05 < diff.std() < 0.09                         # sqrt(2) * NETD = 0.071
    c = cam.render_kelvin(pose, noise=False)
    d = cam.render_kelvin(pose, noise=False)
    assert np.array_equal(c, d)


def test_halo_and_emissivity():
    sc = ThermalScene(_mini_layout())
    P = np.array([[3.0, 0.10, 0.8], [3.0, 0.25, 0.8], [3.0, 0.45, 0.8], [3.0, 3.0, 0.8]])   # 5, 20, 40 cm, 3 m
    T = sc._halo(P, np.full(4, 308.15), np.ones(4, bool))
    assert T[0] > T[1] > T[2] > T[3] - 1e-9 and T[3] == pytest.approx(308.15)
    assert T[0] <= 360.0
    # conductive halo at 5 cm: 50 % of the excess * exp(-0.05 / 0.10)
    assert T[0] - 308.15 == pytest.approx(0.5 * (360.0 - 308.15) * math.exp(-0.5), rel=1e-6)


def test_render_speed_and_visible_proxy():
    cam = SyntheticThermalCamera(LAYOUT, seed=0)
    pose = full_pose(LAYOUT, 8.3, 4.6, math.pi / 2 + 0.2)
    cam.render_kelvin(pose)
    t0 = time.time()
    cam.render_kelvin(pose)
    assert time.time() - t0 < 1.0                            # 5 Hz node budget (typ. ~60 ms)
    rgb, buf = render_visible_proxy(cam.scene, pose)
    assert rgb.shape == (600, 800, 3) and rgb.dtype == np.uint8
    assert np.any(buf.label == LBL_PIPE) and rgb.std() > 5.0


def test_full_pose_floor_levels():
    p = full_pose(LAYOUT, 60.0, 0.0, 1.0)
    assert p[2] == pytest.approx(-0.88 + 0.34) and p[5] == 1.0
    assert full_pose(LAYOUT, 0.0, 0.0, 0.0)[2] == pytest.approx(0.34)
