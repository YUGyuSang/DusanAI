"""Gauge reader (gauge_reader): geometry primitives, exact clean reads, robustness, OCR, blur."""
import math

import cv2
import numpy as np
import pytest

from dusan_common import gauge_design as gd
from dusan_common.layout import Layout
from module2_inspection.gauge_reader import (GaugeHint, GaugeReader, GaugeReaderConfig, apply_h,
                                             circle_poses_from_conic, conic_to_params, ellipse_conic,
                                             fit_ellipse_direct, homography_from_circle_pose,
                                             robust_ellipse_fit, _masked_circ_ncc, _circ_ncc)
from module2_inspection.gauge_synth import (GaugeSpec, GaugeSynth, SynthConditions, ViewSpec, dial_homography,
                                            view_pose)

GAUGES = Layout.load().gauges


@pytest.fixture(scope="module")
def synth():
    return GaugeSynth(seed=0)


@pytest.fixture(scope="module")
def reader(synth):
    return GaugeReader(camera_matrix=synth.camera.K)


@pytest.fixture(scope="module")
def reader_no_ocr(synth):
    return GaugeReader(GaugeReaderConfig(ocr_enabled=False), camera_matrix=synth.camera.K)


def _hint_from_gt(g, gt, du=12.0, dv=-9.0):
    """A pose-predicted hint with a realistic prediction error (ROI search like the node)."""
    return GaugeHint(gauge_id=g["id"], center=(gt["centre_px"][0] + du, gt["centre_px"][1] + dv),
                     radius=gt["radius_px"], vrange=(g["min"], g["max"]))


def _err_fs(value, true, g):
    return abs(value - true) / (g["max"] - g["min"]) * 100.0


CLEAN = SynthConditions(illumination=1.0, gamma=1.0, illum_gradient=0.0, noise_sigma=0.0, clutter=6,
                        distractors=0)


# ------------------------------------------------------------------------------ geometry
def test_ellipse_conic_roundtrip_and_direct_fit():
    rng = np.random.default_rng(1)
    for cx, cy, a, b, ang in [(320, 240, 90, 60, 0.4), (100, 400, 40, 39, -1.2), (500, 120, 150, 70, 2.0)]:
        C = ellipse_conic(cx, cy, a, b, ang)
        coef = np.array([C[0, 0], 2 * C[0, 1], C[1, 1], 2 * C[0, 2], 2 * C[1, 2], C[2, 2]])
        p = conic_to_params(coef)
        assert np.allclose(p[:4], (cx, cy, a, b), atol=1e-6)
        assert abs(math.sin(p[4] - ang)) < 1e-6 or a == b
        t = rng.uniform(0, 2 * np.pi, 120)
        x = cx + a * np.cos(t) * math.cos(ang) - b * np.sin(t) * math.sin(ang) + rng.normal(0, 0.3, t.size)
        y = cy + a * np.cos(t) * math.sin(ang) + b * np.sin(t) * math.cos(ang) + rng.normal(0, 0.3, t.size)
        q = conic_to_params(fit_ellipse_direct(x, y))
        assert np.allclose(q[:4], (cx, cy, a, b), atol=0.4)


def test_robust_ellipse_fit_ignores_outliers():
    rng = np.random.default_rng(2)
    t = rng.uniform(0, 2 * np.pi, 160)
    x, y = 300 + 80 * np.cos(t), 200 + 55 * np.sin(t)
    x[:30] += rng.uniform(15, 40, 30)          # 19 % gross outliers (ticks, occluder edges)
    params, inl, rms = robust_ellipse_fit(x, y)
    assert np.allclose(params[:4], (300, 200, 80, 55), atol=0.3)
    assert inl[:30].sum() <= 3 and rms < 0.3


def test_circle_pose_from_conic_recovers_plane_and_centre(synth):
    K = synth.camera.K
    for az, el, roll in [(30, -10, 3), (0, -38, -5), (-40, 15, 10), (0, 0, 0)]:
        R, C = view_pose(ViewSpec(distance=1.2, azimuth_deg=az, elevation_deg=el, roll_deg=roll,
                                  offset_px=(60, -40)), synth.camera)
        H = dial_homography(K, R, C, np.zeros(3), [1, 0, 0], [0, 1, 0], 0.30)
        Hi = np.linalg.inv(H)
        conic = Hi.T @ np.diag([1, 1, -gd.R_FACE ** 2]) @ Hi
        sols = circle_poses_from_conic(conic, K)
        n_true = R @ np.array([0.0, 0.0, 1.0])
        c_true = apply_h(H, [[0.0, 0.0]])[0]
        best = min(sols, key=lambda s: np.linalg.norm(s[0] - n_true))
        assert np.linalg.norm(best[0] - n_true) < 1e-6
        Hs = homography_from_circle_pose(K, *best)
        assert np.linalg.norm(apply_h(Hs, [[0.0, 0.0]])[0] - c_true) < 1e-6
        # the rectified frame differs from the truth only by an in-plane rotation
        pts = [[0.4 * math.cos(a), 0.4 * math.sin(a)] for a in np.linspace(0, 6, 7)]
        q_true = apply_h(Hi, apply_h(Hs, pts))
        assert np.allclose(np.hypot(q_true[:, 0], q_true[:, 1]), 0.4, atol=1e-6)


def test_masked_ncc_equals_plain_ncc_without_mask():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=720), rng.normal(size=720)
    b2 = np.roll(a, 37) + 0.1 * b
    plain = _circ_ncc(b2, a)
    masked = _masked_circ_ncc(b2, a, np.ones(720))
    assert int(np.argmax(plain)) == int(np.argmax(masked)) == 37
    assert np.allclose(plain, masked, atol=1e-6)


# ------------------------------------------------------------------------------ clean accuracy
def test_needle_exact_on_clean_frontal_renders(synth, reader_no_ocr):
    """< 0.5 % FS on clean renders for every registry gauge over the whole scale."""
    errs = []
    for gi, g in enumerate(GAUGES):
        for f in (0.02, 0.5, 0.98) if gi % 2 else (0.27, 0.77, 0.5):
            val = g["min"] + f * (g["max"] - g["min"])
            img, gt = synth.render(GaugeSpec.from_layout(g, val), ViewSpec(distance=1.2, elevation_deg=-8),
                                   CLEAN, np.random.default_rng(int(f * 100)))
            res = reader_no_ocr.read(img, [_hint_from_gt(g, gt)], detect_unhinted=False)
            assert len(res) == 1 and res[0].accepted, (g["id"], val)
            rd = res[0]
            errs.append(_err_fs(rd.value, val, g))
            assert not rd.mirrored
    assert max(errs) < 0.5, max(errs)
    assert np.mean(errs) < 0.15


def test_oblique_rolled_and_far(synth, reader_no_ocr):
    cases = [ViewSpec(1.0, 38, -15, 12, (100, 60)), ViewSpec(2.4, -20, -25, -8, (-200, 120)),
             ViewSpec(0.9, 0, -40, 0, (0, -60)), ViewSpec(1.6, 30, 10, 25, (150, -100))]
    for k, v in enumerate(cases):
        g = GAUGES[k]
        val = g["min"] + 0.63 * (g["max"] - g["min"])
        img, gt = synth.render(GaugeSpec.from_layout(g, val), v, CLEAN, np.random.default_rng(k))
        rd = reader_no_ocr.read_one(img, (g["min"], g["max"]))
        assert rd is not None and _err_fs(rd.value, val, g) < 1.0, (k, rd and rd.value, val)
        assert math.hypot(rd.center[0] - gt["centre_px"][0], rd.center[1] - gt["centre_px"][1]) < 1.5


def test_mirrored_and_rotated_texture(synth, reader_no_ocr):
    """Scale calibration (tick-ring gap + red band) handles any in-plane rotation and a mirrored
    texture mapping (e.g. flipped UVs in the simulator)."""
    g = GAUGES[1]
    val = 11.4
    img, gt = synth.render(GaugeSpec.from_layout(g, val), ViewSpec(distance=1.1, elevation_deg=-10), CLEAN,
                           np.random.default_rng(5))
    variants = {"mirror": cv2.flip(img, 1), "rot90": cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                "rot180_mirror": cv2.flip(img, 0)}
    K = synth.camera.K
    # a 90 deg image rotation is a different camera: rotate the intrinsics with it
    K90 = np.array([[K[1, 1], 0, img.shape[0] - 1 - K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1.0]])
    for name, im in variants.items():
        rdr = GaugeReader(GaugeReaderConfig(ocr_enabled=False), camera_matrix=K90 if name == "rot90" else K)
        rd = rdr.read_one(im, (g["min"], g["max"]))
        assert rd is not None and _err_fs(rd.value, val, g) < 0.6, name
        assert rd.mirrored == (name != "rot90"), name


# ------------------------------------------------------------------------------ robustness
@pytest.mark.parametrize("kind", ["low_light", "motion_blur", "noise", "combined"])
def test_robust_to_degradations(synth, reader_no_ocr, kind):
    rng = np.random.default_rng({"low_light": 11, "motion_blur": 12, "noise": 13, "combined": 14}[kind])
    conds = {
        "low_light": SynthConditions(illumination=0.12, gamma=1.4, noise_sigma=2.5, shot_noise=0.4, illum_gradient=0.5),
        "motion_blur": SynthConditions(motion_blur_px=12.0, defocus_px=0.8),
        "noise": SynthConditions(noise_sigma=12.0, shot_noise=0.8, jpeg_quality=45),
        "combined": SynthConditions(illumination=0.2, gamma=1.2, noise_sigma=4.0, shot_noise=0.4,
                                    motion_blur_px=8.0, defocus_px=1.0, jpeg_quality=70, illum_gradient=0.4),
    }
    errs = []
    for k in range(4):
        g = GAUGES[(2 * k + 1) % len(GAUGES)]
        val = float(rng.uniform(g["min"], g["max"]))
        view = ViewSpec(distance=float(rng.uniform(1.0, 2.0)), azimuth_deg=float(rng.uniform(-20, 20)),
                        elevation_deg=float(rng.uniform(-25, -5)), roll_deg=float(rng.normal(0, 4)))
        img, gt = synth.render(GaugeSpec.from_layout(g, val), view, conds[kind], rng)
        rd = reader_no_ocr.read_one(img, (g["min"], g["max"])) if k % 2 else             next(iter(reader_no_ocr.read(img, [_hint_from_gt(g, gt)], detect_unhinted=False)), None)
        assert rd is not None, (kind, k)
        errs.append(_err_fs(rd.value, val, g))
    assert max(errs) < 3.0, (kind, errs)


def test_partial_occlusion(synth, reader_no_ocr):
    rng = np.random.default_rng(21)
    ok = 0
    for k in range(4):
        g = GAUGES[k]
        val = float(rng.uniform(g["min"], g["max"]))
        img, gt = synth.render(GaugeSpec.from_layout(g, val), ViewSpec(distance=1.4, elevation_deg=-12),
                               SynthConditions(occlusion=0.22), rng)
        rd = reader_no_ocr.read_one(img, (g["min"], g["max"]))
        ok += rd is not None and _err_fs(rd.value, val, g) < 2.0
    assert ok >= 3


def test_blur_metric_tracks_motion_blur(synth, reader_no_ocr):
    g = GAUGES[0]
    est = []
    for L in (0.0, 6.0, 12.0):
        img, gt = synth.render(GaugeSpec.from_layout(g), ViewSpec(distance=1.0, elevation_deg=-10),
                               SynthConditions(motion_blur_px=L, motion_blur_angle_deg=30.0, noise_sigma=0.5),
                               np.random.default_rng(3))
        rd = reader_no_ocr.read(img, [_hint_from_gt(g, gt)], detect_unhinted=False)[0]
        est.append(rd.blur_px)
    assert est[0] < 2.0 < est[1] < est[2]
    assert abs(est[2] - 12.0) < 4.0


def test_rejects_frames_without_dials(synth, reader_no_ocr):
    rng = np.random.default_rng(31)
    for k in range(3):
        img = synth.render_empty(SynthConditions(distractors=3, clutter=20), rng)
        assert [r for r in reader_no_ocr.read(img) if r.accepted] == []


def test_hint_roi_and_prior_normal(synth, reader_no_ocr):
    """With a pose-predicted hint the dial is found in its ROI and gets the hint id/range."""
    g = GAUGES[3]
    img, gt = synth.render(GaugeSpec.from_layout(g), ViewSpec(distance=1.3, azimuth_deg=35, elevation_deg=-20),
                           SynthConditions(distractors=2), np.random.default_rng(8))
    cx, cy = gt["centre_px"]
    hint = GaugeHint(gauge_id=g["id"], center=(cx + 25, cy - 20), radius=gt["radius_px"] * 1.1,
                     vrange=(g["min"], g["max"]), normal_cam=gt["normal_opt"])
    res = reader_no_ocr.read(img, [hint], detect_unhinted=False)
    assert len(res) == 1 and res[0].gauge_id == g["id"]
    assert _err_fs(res[0].value, g["value"], g) < 1.0


# ------------------------------------------------------------------------------ OCR range
def test_range_read_by_ocr_when_unknown(synth, reader):
    """At close range the scale labels give the range (no registry), value then follows."""
    for gid in ("G01", "G08", "G06", "G04"):
        g = next(x for x in GAUGES if x["id"] == gid)
        img, gt = synth.render(GaugeSpec.from_layout(g), ViewSpec(distance=0.95, elevation_deg=-12, azimuth_deg=10),
                               SynthConditions(noise_sigma=1.0), np.random.default_rng(4))
        rd = reader.read_one(img, None)
        assert rd is not None and rd.range_source == "ocr", gid
        assert rd.range_ocr == (g["min"], g["max"]), (gid, rd.range_ocr)
        assert _err_fs(rd.value, g["value"], g) < 0.5


def test_config_from_dict_casts_types():
    cfg = GaugeReaderConfig.from_dict({"max_dials": 2.0, "min_confidence": 1, "ocr_enabled": False,
                                       "needle_model": "off", "unknown_key": 3})
    assert cfg.max_dials == 2 and isinstance(cfg.max_dials, int)
    assert cfg.min_confidence == 1.0 and isinstance(cfg.min_confidence, float)
    assert cfg.ocr_enabled is False and cfg.needle_model == "off"
