"""Scale-label OCR (gauge_ocr): glyph templates, segmentation, range inference."""
import math

import cv2
import numpy as np
import pytest

from dusan_common import gauge_design as gd
from module2_inspection.gauge_ocr import (LABEL_SCALE_PER_S, DigitOCR, LabelRead, candidate_ranges,
                                          expected_labels, infer_range, major_label_angles,
                                          n_major_divisions, parse_label, render_hershey_text,
                                          verify_ranges)


@pytest.fixture(scope="module")
def ocr():
    return DigitOCR()


def _label_patch(text, tex_size, label_px=22.0, ink=20, face=240):
    """A reader-like label patch (0.12 x 0.05 dial units, label ~22 px tall) of a label printed
    exactly like gauge_design does (FONT_HERSHEY_SIMPLEX with the texture's label parameters);
    ``label_px`` < 22 emulates a lower camera resolution (down- then up-sampling)."""
    W, H = int(0.12 * tex_size), int(0.05 * tex_size)
    img = np.full((H, W), face, np.uint8)
    sc = tex_size * LABEL_SCALE_PER_S
    th = max(1, int(tex_size / 1024 * 3))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
    cv2.putText(img, text, (W // 2 - tw // 2, H // 2 + tht // 2), cv2.FONT_HERSHEY_SIMPLEX, sc, ink, th, cv2.LINE_AA)
    f = label_px / 22.0
    p = cv2.resize(img, (int(102 * f), int(42 * f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(p, (102, 42), interpolation=cv2.INTER_LINEAR)


def test_parse_label():
    assert parse_label("1.6") == 1.6
    assert parse_label("10") == 10.0
    assert parse_label("0") == 0.0
    assert parse_label("-1") == -1.0
    assert parse_label("0.25") == 0.25
    for bad in ("", ".5", "1.", "1..6", "05", "1-6", "?"):
        assert parse_label(bad) is None


def test_template_bank_covers_both_font_engines(ocr):
    names = ocr.font_variants()
    assert any(n.startswith("cv_") for n in names)          # runtime cv2.putText
    assert any(n.startswith("hershey_") for n in names)     # classic stroke font (OpenCV <= 4)
    assert any(n.startswith("cv5_") for n in names)         # embedded OpenCV 5 rendering
    for ch in "0123456789":
        assert len(ocr.templates[ch]) >= 9


LABELS = ["0", "2", "4", "6", "8", "10", "16", "25", "0.0", "0.5", "1.5", "2.5", "4.0", "0.2", "1.6",
          "12", "14", "20", "100", "-1"]


@pytest.mark.parametrize("tex_size", [512, 1024, 2048])
@pytest.mark.parametrize("label_px", [22.0, 12.0])
def test_read_rendered_labels(ocr, tex_size, label_px):
    """Labels rendered exactly like the dial texture are read character by character."""
    bad = []
    for txt in LABELS:
        r = ocr.read(_label_patch(txt, tex_size, label_px), 22.0, blur_sigma=0.6 * 22.0 / label_px)
        if r.text != txt:
            bad.append((txt, r.text))
    assert len(bad) <= 1, f"misread at texture {tex_size}, label {label_px}px: {bad}"


def test_read_classic_hershey_labels(ocr):
    """The OpenCV<=4 stroke font (vector data) is read as well (cross-version robustness)."""
    ok = 0
    for txt in ["0", "10", "16", "25", "1.6", "0.5", "4.0", "8", "6", "2"]:
        ink = render_hershey_text(txt, 22.0, 0.12)
        patch = np.full((50, ink.shape[1] + 50), 240, np.float32)
        patch[14:14 + ink.shape[0], 25:25 + ink.shape[1]] -= 220 * ink
        r = ocr.read(patch.astype(np.uint8))
        ok += r.text == txt
    assert ok >= 9


def test_read_with_noise_and_blur(ocr):
    rng = np.random.default_rng(0)
    ok = 0
    for txt in ["10", "1.6", "25", "0.5", "6"]:
        p = _label_patch(txt, 1024).astype(np.float32)
        p = cv2.GaussianBlur(p, (0, 0), 1.0) + rng.normal(0, 8, p.shape)
        r = ocr.read(np.clip(p, 0, 255).astype(np.uint8), 22.0, blur_sigma=1.0)
        ok += r.text == txt
    assert ok >= 4


def test_layout_ranges_division_counts():
    assert n_major_divisions(0, 10) == 5
    assert n_major_divisions(0, 16) == 8
    assert n_major_divisions(0, 6) == 6
    assert n_major_divisions(0, 25) == 5
    assert n_major_divisions(0, 4) == 8
    assert n_major_divisions(0, 1.6) == 8
    assert expected_labels(0, 1.6)[:3] == ["0.0", "0.2", "0.4"]
    assert expected_labels(0, 10) == ["0", "2", "4", "6", "8", "10"]
    ang = major_label_angles(5)
    assert ang[0] == gd.PHI_MIN_DEG and abs(ang[-1] - gd.PHI_MAX_DEG) < 1e-9
    for (v0, v1) in candidate_ranges(8):
        assert n_major_divisions(v0, v1) == 8
    assert (0.0, 16.0) in candidate_ranges(8) and (0.0, 10.0) in candidate_ranges(5)


def test_infer_range_ransac_rejects_misreads():
    def lr(text, conf=0.8):
        return LabelRead(text, parse_label(text), conf)
    # 0..16 with 8 divisions; one misread label ('8' read as '6') and one unreadable
    reads = [(0, lr("0")), (1, lr("2")), (2, lr("4")), (3, lr("", 0.0)), (4, lr("6")), (5, lr("10")),
             (6, lr("12")), (8, lr("16"))]
    est = infer_range(reads, 8)
    assert est is not None and (est.vmin, est.vmax) == (0.0, 16.0)
    assert est.method == "chars" and est.confidence > 0.5
    # the step must be 'nice' for the division count: 0, 3, 6 is not a gauge_design scale
    assert infer_range([(0, lr("0")), (1, lr("3")), (2, lr("6"))], 5) is None
    # two labels that are not at an end of the scale are not enough
    assert infer_range([(2, lr("4")), (3, lr("6"))], 5) is None


def test_verify_ranges_by_synthesis(ocr):
    """Analysis by synthesis picks the right standard range from small blurred labels."""
    for v0, v1 in [(0.0, 10.0), (0.0, 1.6), (0.0, 25.0)]:
        n = n_major_divisions(v0, v1)
        patches = {k: _label_patch(txt, 1024, label_px=6.0) for k, txt in enumerate(expected_labels(v0, v1))}
        est = verify_ranges(ocr, patches, n, 22.0, blur_sigma=0.6 * 22.0 / 6.0)
        assert est is not None and (est.vmin, est.vmax) == (v0, v1)


class _OCRWithoutRuntimeFont(DigitOCR):
    """The OCR as it runs on a PC whose OpenCV renders FONT_HERSHEY_SIMPLEX differently from
    the PC that generated the dial textures (e.g. textures made with OpenCV 5, reader on the
    Ubuntu robot PC with OpenCV 4): the runtime cv2.putText templates do not match the dial,
    only the embedded OpenCV-5 bitmaps and the classic Hershey strokes are available."""

    def _build_templates(self):
        from module2_inspection import gauge_ocr as go
        for ratio in go.HERSHEY_STROKE_RATIOS:
            for ch in go.DIGITS:
                self._add(ch, f"hershey_{ratio}", go.render_hershey_glyph(ch, 56.0, ratio))
        for ch, name, w, h, hexs in go._CV5_GLYPHS:
            self._add(ch, name, go._decode_glyph(w, h, hexs))


def test_cross_opencv_version_reading():
    ocr = _OCRWithoutRuntimeFont()
    assert not any(n.startswith("cv_") for n in ocr.font_variants())
    for tex_size in (512, 1024):
        for label_px in (22.0, 12.0):
            bad = [(t, r.text) for t in LABELS
                   for r in [ocr.read(_label_patch(t, tex_size, label_px), 22.0, blur_sigma=0.6 * 22.0 / label_px)]
                   if r.text != t]
            assert len(bad) <= 1, (tex_size, label_px, bad)
    for v0, v1 in [(0.0, 10.0), (0.0, 1.6), (0.0, 25.0), (0.0, 4.0)]:
        n = n_major_divisions(v0, v1)
        patches = {k: _label_patch(txt, 1024, label_px=5.0) for k, txt in enumerate(expected_labels(v0, v1))}
        est = verify_ranges(ocr, patches, n, 22.0, blur_sigma=0.6 * 22.0 / 5.0)
        assert est is not None and (est.vmin, est.vmax) == (v0, v1)
