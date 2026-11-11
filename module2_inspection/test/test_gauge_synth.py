"""Synthetic gauge frames (gauge_synth): projection geometry, ground truth, degradation models."""
import math

import cv2
import numpy as np
import pytest

from dusan_common import gauge_design as gd
from dusan_common.layout import Layout
from module2_inspection.gauge_synth import (CONDITION_KINDS, GaugeSpec, GaugeSynth, SynthConditions, ViewSpec,
                                            apply_h, defocus_kernel, motion_blur_kernel, random_conditions,
                                            random_view, tex_to_dial, view_pose)

GAUGES = Layout.load().gauges


@pytest.fixture(scope="module")
def synth():
    return GaugeSynth(seed=0)


def test_motion_blur_kernel_normalised_and_oriented():
    for L, ang in [(1.0, 0.0), (7.3, 30.0), (15.0, 115.0)]:
        k = motion_blur_kernel(L, ang)
        assert k.sum() == pytest.approx(1.0, abs=1e-5)
        ys, xs = np.nonzero(k > 1e-4)
        c = (k.shape[0] - 1) / 2.0
        # second moment along the blur direction = L^2 / 12 (uniform segment), across ~ 0
        d = np.array([math.cos(math.radians(ang)), -math.sin(math.radians(ang))])
        w = k[ys, xs]
        along = ((xs - c) * d[0] + (ys - c) * d[1])
        across = ((xs - c) * -d[1] + (ys - c) * d[0])
        # (+ up to ~1/4 px^2 from the bilinear splatting of the sub-pixel segment samples)
        assert np.sum(w * along ** 2) == pytest.approx(L * L / 12.0, rel=0.15, abs=0.3)
        assert np.sum(w * across ** 2) < 0.3
    assert motion_blur_kernel(0.2, 0.0).shape == (1, 1)


def test_defocus_kernel_is_a_normalised_disc():
    k = defocus_kernel(3.0)
    assert k.sum() == pytest.approx(1.0, abs=1e-5)
    c = k.shape[0] // 2
    assert k[c, c] > 0 and k[c, c + 4] == pytest.approx(0.0, abs=1e-6)
    assert defocus_kernel(0.1).shape == (1, 1)


def test_ground_truth_geometry_is_consistent(synth):
    g = GAUGES[0]
    view = ViewSpec(distance=1.3, azimuth_deg=25, elevation_deg=-15, roll_deg=7, offset_px=(80, -50))
    img, gt = synth.render(GaugeSpec.from_layout(g), view, SynthConditions(noise_sigma=0, clutter=0, distractors=0),
                           rng=np.random.default_rng(1))
    assert img.shape == (600, 800, 3) and img.dtype == np.uint8
    # the dial centre lands at the principal point + offset
    assert np.allclose(gt["centre_px"], (400 + 80, 300 - 50), atol=1e-6)
    assert gt["incidence_deg"] == pytest.approx(view.incidence_deg)
    assert gt["angle_deg"] == pytest.approx(gd.value_to_angle(g["value"], g["min"], g["max"]))
    # the needle tip in the image is dark, the face just beside it is white
    a = math.radians(gt["angle_deg"])
    tip = apply_h(gt["H_dial2img"], [[0.30 * math.cos(a), 0.30 * math.sin(a)]])[0]
    a2 = a + math.radians(12)
    face = apply_h(gt["H_dial2img"], [[0.30 * math.cos(a2), 0.30 * math.sin(a2)]])[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray[int(round(tip[1])), int(round(tip[0]))] < 80
    assert gray[int(round(face[1])), int(round(face[0]))] > 180
    # texture pixel <-> dial unit mapping
    S = synth.tex_size
    assert np.allclose(apply_h(tex_to_dial(S), [[S / 2, S / 2], [S, 0]]), [[0, 0], [0.5, 0.5]])


def test_view_pose_looks_at_the_dial(synth):
    for v in (ViewSpec(), ViewSpec(2.0, 40, -30, 10, (0, 0))):
        R, C = view_pose(v, synth.camera)
        assert np.linalg.norm(C) == pytest.approx(v.distance)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
        z = R @ (-C)
        assert z[2] > 0 and np.hypot(z[0], z[1]) / z[2] < 1e-9     # dial centre on the optical axis


def test_photometric_degradations(synth):
    spec = GaugeSpec.from_layout(GAUGES[1])
    view = ViewSpec(distance=1.2)
    base = dict(noise_sigma=0.0, clutter=5, distractors=0, illum_gradient=0.0)
    ref, _ = synth.render(spec, view, SynthConditions(**base), rng=np.random.default_rng(2))
    dark, _ = synth.render(spec, view, SynthConditions(illumination=0.15, **base), rng=np.random.default_rng(2))
    assert dark.mean() == pytest.approx(0.15 * ref.mean(), rel=0.1)
    noisy, _ = synth.render(spec, view, SynthConditions(**dict(base, noise_sigma=10.0)), rng=np.random.default_rng(2))
    assert np.std(noisy.astype(float) - ref.astype(float)) == pytest.approx(10.0, rel=0.15)
    blur, gt = synth.render(spec, view, SynthConditions(motion_blur_px=12.0, **base), rng=np.random.default_rng(2))
    lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert lap(blur) < 0.3 * lap(ref) and gt["motion_blur_angle_deg"] is not None
    jpg, _ = synth.render(spec, view, SynthConditions(jpeg_quality=30, **base), rng=np.random.default_rng(2))
    assert 0 < np.abs(jpg.astype(float) - ref.astype(float)).mean() < 8.0


def test_occlusion_and_plate_options(synth):
    spec = GaugeSpec.from_layout(GAUGES[2])
    view = ViewSpec(distance=1.0)
    base = dict(noise_sigma=0.0, clutter=0, distractors=0, illum_gradient=0.0)
    ref, gt = synth.render(spec, view, SynthConditions(**base), rng=np.random.default_rng(3))
    occ, _ = synth.render(spec, view, SynthConditions(occlusion=0.3, **base), rng=np.random.default_rng(3))
    cx, cy = gt["centre_px"]
    r = gt["radius_px"]
    roi = (slice(int(cy - r), int(cy + r)), slice(int(cx - r), int(cx + r)))
    assert np.mean(np.abs(occ[roi].astype(float) - ref[roi].astype(float)).max(axis=2) > 40) > 0.05
    pl, _ = synth.render(spec, view, SynthConditions(plate=True, **base), rng=np.random.default_rng(3))
    # the plate is bright just outside the housing (corners of the bounding square of the dial)
    corner = (int(cy + 1.08 * r), int(cx + 1.08 * r))
    assert int(cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY)[corner]) > 150


def test_random_presets_cover_the_condition_grid():
    rng = np.random.default_rng(4)
    for kind in CONDITION_KINDS:
        c = random_conditions(rng, kind)
        v = random_view(rng, kind)
        assert 0.9 <= v.distance <= 2.5
        if kind == "low_light":
            assert c.illumination <= 0.3
        if kind == "motion_blur":
            assert 4.0 <= c.motion_blur_px <= 15.0
        if kind in ("oblique", "combined"):
            assert 25.0 <= v.incidence_deg <= 42.5
        else:
            assert v.incidence_deg <= 20.5
        if kind == "occluded":
            assert c.occlusion > 0


def test_empty_frames_have_no_dial(synth):
    img = synth.render_empty(SynthConditions(distractors=3), rng=np.random.default_rng(5))
    assert img.shape == (600, 800, 3)
