"""Gauge association (gauge_association): camera projection of the layout gauges, visibility,
ROI hints, id assignment with / without a pose, OCR-vs-registry range reconciliation, and the
complete pose -> hint -> read -> fuse chain of gauge_reader_node without ROS."""
import math

import numpy as np
import pytest

from dusan_common import gauge_design as gd
from dusan_common.camera import PinholeCamera, camera_world_pose
from dusan_common.layout import Layout
from dusan_common.robot_model import NOMINAL_BODY_HEIGHT
from module2_inspection.gauge_association import GaugeAssociator, dial_axes
from module2_inspection.gauge_reader import GaugeFusion, GaugeReader, GaugeReaderConfig
from module2_inspection.gauge_synth import GaugeSynth, SynthConditions, texture_side
from module2_inspection.gauge_synth import dial_axes as synth_dial_axes

LAYOUT = Layout.load()
GAUGES = {g["id"]: g for g in LAYOUT.gauges}


def standoff_pose(g, dist=1.5, lateral=0.0, yaw_err=0.0):
    """Robot pose (x, y, z, roll, pitch, yaw) standing ``dist`` in front of gauge ``g``, facing it."""
    n = np.asarray(g["normal"], float)
    p = np.asarray(g["position"], float)
    side = np.array([-n[1], n[0], 0.0])
    base = p + dist * n + lateral * side
    yaw = math.atan2(-n[1], -n[0]) + yaw_err
    z = LAYOUT.floor_z(base[0], base[1]) + NOMINAL_BODY_HEIGHT
    return (float(base[0]), float(base[1]), float(z), 0.0, 0.0, float(yaw))


@pytest.fixture(scope="module")
def assoc():
    return GaugeAssociator.from_layout(LAYOUT)


@pytest.fixture(scope="module")
def synth():
    return GaugeSynth(seed=0)


# ------------------------------------------------------------------------------ geometry
def test_dial_axes_match_world_generator_convention():
    """Viewer-right = up x n (generate_world: model +Y after Rz(yaw)), up = world z, both modules."""
    for n, right in [((0, 1, 0), (-1, 0, 0)), ((1, 0, 0), (0, 1, 0)), ((0, -1, 0), (1, 0, 0))]:
        for fn in (dial_axes, synth_dial_axes):
            r, u, nn = fn(n)
            assert np.allclose(r, right) and np.allclose(u, (0, 0, 1)) and np.allclose(nn, n)
            assert np.allclose(np.cross(r, u), n)          # right x up = normal (front face)


def test_texture_side_matches_world_generator():
    """tools/generate_world.py: texture side == dial_diameter (DIAL_UV_RADIUS = 0.5), so the white
    face radius in the world is R_FACE * dial_diameter."""
    assert texture_side(0.30) == pytest.approx(0.30)
    assoc_ = GaugeAssociator(LAYOUT.gauges)
    g = GAUGES["G01"]
    pose = standoff_pose(g, dist=1.5)
    pred = next(p for p in assoc_.predict(pose) if p.gauge_id == "G01")
    # fronto-parallel-ish: projected face radius ~ f * R_FACE * D / distance
    f = assoc_.camera.fx
    assert pred.radius == pytest.approx(f * gd.R_FACE * 0.30 / pred.depth, rel=0.06)


@pytest.mark.parametrize("gid", sorted(GAUGES))
def test_projection_matches_rendered_dial(assoc, synth, gid):
    """Predicted centre / face ellipse == the synthetic render of the same pose (same geometry,
    independent code paths: association vs renderer)."""
    g = GAUGES[gid]
    pose = standoff_pose(g, dist=1.4, lateral=0.25)
    pred = next(p for p in assoc.predict(pose) if p.gauge_id == gid)
    T = camera_world_pose(pose, "inspection_camera")
    img, gt = synth.render_world(g, T, SynthConditions(clutter=4, distractors=0), rng=np.random.default_rng(0))
    assert pred.in_front and pred.facing
    assert math.hypot(pred.center[0] - gt["centre_px"][0], pred.center[1] - gt["centre_px"][1]) < 0.05
    assert pred.radius == pytest.approx(gt["radius_px"], rel=2e-3)
    assert pred.incidence_deg == pytest.approx(gt["incidence_deg"], abs=1e-6)
    # the plane-induced homographies agree as well
    pts = [[0.4 * math.cos(a), 0.4 * math.sin(a)] for a in np.linspace(0, 6, 5)]
    from module2_inspection.gauge_reader import apply_h
    assert np.allclose(apply_h(pred.H_dial2img, pts), apply_h(gt["H_dial2img"], pts), atol=1e-6)


def test_visibility_rules(assoc):
    g = GAUGES["G01"]
    pose = standoff_pose(g, dist=1.5)
    p = next(q for q in assoc.predict(pose) if q.gauge_id == "G01")
    # camera ~0.46 m below the dial at ~1.2 m: incidence = elevation angle atan(0.46 / 1.2) ~ 21 deg
    assert p.visible and p.in_image and 15.0 < p.incidence_deg < 25.0
    # turned away -> behind the camera
    x, y, z, r, pi, yaw = pose
    p = next(q for q in assoc.predict((x, y, z, r, pi, yaw + math.pi)) if q.gauge_id == "G01")
    assert not p.in_front and not p.visible
    # standing behind the dial -> seen from the back
    n = np.asarray(g["normal"], float)
    back = np.asarray(g["position"]) - 1.5 * n
    p = next(q for q in assoc.predict((back[0], back[1], z, 0.0, 0.0, math.atan2(n[1], n[0])))
             if q.gauge_id == "G01")
    assert not p.facing and not p.visible
    # far away -> too small / too far
    p = next(q for q in assoc.predict(standoff_pose(g, dist=9.0)) if q.gauge_id == "G01")
    assert not p.visible
    # a gauge in another hall is never visible from here
    assert not next(q for q in assoc.predict(pose) if q.gauge_id == "G08").visible


def test_hints_target_first_and_roi(assoc):
    # G01 and G02 are 3 m apart on the same wall: from 4 m back both are in view
    g1, g2 = GAUGES["G01"], GAUGES["G02"]
    mid = dict(g1, position=[0.5 * (g1["position"][0] + g2["position"][0]), g1["position"][1], 1.0])
    pose = standoff_pose(mid, dist=4.0)
    hints = assoc.hints(pose, target_id="G02", pose_sigma_px=25.0)
    ids = [h.gauge_id for h in hints]
    assert set(ids) == {"G01", "G02"} and ids[0] == "G02"
    for h in hints:
        g = GAUGES[h.gauge_id]
        assert h.vrange == (g["min"], g["max"]) and h.unit == g["unit"]
        assert h.search_radius == pytest.approx(2.0 * h.radius + 3 * 25.0 + 20.0)
        assert h.normal_cam[2] < 0                       # facing the camera (optical z forward)


def test_assign_with_pose_error(assoc):
    pose = standoff_pose(GAUGES["G03"], dist=1.6)
    preds = assoc.predict(pose)
    p = next(q for q in preds if q.gauge_id == "G03")
    # detection shifted by 40 px (pose / timing error) + a second, unrelated dial
    ids = assoc.assign([(p.center[0] + 40, p.center[1] - 20), (30.0, 30.0)], [p.radius * 1.1, 20.0], preds)
    assert ids == ["G03", None]
    # radius far off -> not the same gauge
    assert assoc.assign([p.center], [p.radius * 2.5], preds) == [None]
    # pose error beyond the gate (250 px): only the mission target fallback identifies it
    far = [(p.center[0] - 250, p.center[1] + 30), (790.0, 590.0)]
    assert assoc.assign(far, [p.radius, p.radius], preds) == [None, None]
    assert assoc.assign(far, [p.radius, p.radius], preds, target_id="G03") == ["G03", None]


def test_assign_without_pose_uses_target(assoc):
    ids = assoc.assign([(100.0, 100.0), (410.0, 290.0), (700.0, 500.0)], [40.0, 45.0, 60.0], None,
                       target_id="G05", image_size=(800, 600))
    assert ids == [None, "G05", None]
    assert assoc.assign([(410.0, 290.0)], [45.0], None, target_id=None) == [None]


def test_reconcile_range(assoc):
    rc = assoc.reconcile_range("G02", (0.0, 16.0))
    assert rc.agree and rc.source == "registry+ocr" and rc.range_used == (0.0, 16.0)
    rc = assoc.reconcile_range("G02", (0.0, 10.0))
    assert rc.agree is False and rc.range_used == (0.0, 16.0) and rc.ocr_range == (0.0, 10.0)
    assert rc.source == "registry(ocr_disagrees)"
    rc = assoc.reconcile_range(None, (0.0, 2.5))
    assert rc.source == "ocr" and rc.range_used == (0.0, 2.5)
    rc = assoc.reconcile_range("G08", None)
    assert rc.source == "registry" and rc.range_used == (0.0, 1.6)
    assert assoc.reconcile_range(None, None) is None


def test_camera_info_intrinsics_update():
    a = GaugeAssociator.from_layout(LAYOUT)
    K = np.array([[700.0, 0, 390.0], [0, 700.0, 310.0], [0, 0, 1]])
    a.set_intrinsics(800, 600, K=K)
    assert a.camera.fx == 700.0 and a.camera.cx == 390.0 and a.camera.cy == 310.0
    a.set_intrinsics(800, 600, hfov=0.9)
    assert a.camera.fx == pytest.approx(PinholeCamera(800, 600, 0.9).fx)


# ------------------------------------------------------------------------------ node chain (no ROS)
def test_pose_hint_read_fuse_chain(assoc, synth):
    """What gauge_reader_node does per frame: pose -> hints (ROI, registry range, normal prior)
    -> reader -> id -> temporal fusion; robot shake = pose noise between render and prediction."""
    reader = GaugeReader(GaugeReaderConfig(ocr_enabled=False), camera_matrix=synth.camera.K)
    fusion = GaugeFusion(window_n=10, window_s=10.0)
    rng = np.random.default_rng(5)
    for gid in ("G04", "G06", "G08"):
        g = GAUGES[gid]
        for k in range(4):
            # 1.6-2.2 m: closer, the camera (pitched up 17 deg) cannot see dials mounted 1.2 m high
            true_pose = standoff_pose(g, dist=float(rng.uniform(1.6, 2.2)), lateral=float(rng.uniform(-0.4, 0.4)),
                                      yaw_err=float(rng.normal(0, 0.05)))
            x, y, z, r, pi, yaw = true_pose
            # odometry lags the camera: small pose error (2 cm, 1 deg roll/pitch shake)
            odom_pose = (x + rng.normal(0, 0.02), y + rng.normal(0, 0.02), z, r + rng.normal(0, 0.015),
                         pi + rng.normal(0, 0.015), yaw + rng.normal(0, 0.01))
            img, gt = synth.render_world(g, camera_world_pose(true_pose, "inspection_camera"),
                                         SynthConditions(noise_sigma=1.5, motion_blur_px=float(rng.uniform(0, 4))),
                                         rng=rng)
            hints = [h for h in assoc.hints(odom_pose, target_id=gid) if h.gauge_id == gid]
            assert hints, (gid, k)
            res = reader.read(img, hints, detect_unhinted=False)
            assert len(res) == 1 and res[0].gauge_id == gid
            fused = fusion.add(gid, float(k), res[0].value, res[0].confidence, g["max"] - g["min"])
        assert abs(fused.value - g["value"]) / (g["max"] - g["min"]) < 0.005
        assert fused.n_samples == 4
        assert LAYOUT.gauge_status(gid, fused.value) == LAYOUT.gauge_status(gid, g["value"])
