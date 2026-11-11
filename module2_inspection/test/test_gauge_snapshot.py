"""Inspection evidence snapshot (module2_inspection.gauge_snapshot) and the overlay labels of
gauge_reader.draw_reading: label boxes never overlap each other or the header banner, the
snapshot JPEGs decode with the expected geometry, the collector keeps the frame whose own needle
agrees with the final fused value, and the message carries the images as base64."""
import base64
import math

import cv2
import numpy as np
import pytest

from dusan_common import gauge_design as gd
from module2_inspection import gauge_reader as gr
from module2_inspection.gauge_snapshot import (SnapshotCollector, format_value, render_gauge_snapshot,
                                               snapshot_message, value_decimals, wall_time_text)


class FakeReading:
    """The geometry draw_reading needs: a circular dial at (cx, cy) with radius r, needle at angle."""

    def __init__(self, cx, cy, r, angle_deg=90.0, confidence=0.9):
        self.ellipse = (float(cx), float(cy), float(r), float(r), 0.0)
        self.center = (float(cx), float(cy))
        self.angle_deg = angle_deg
        self.confidence = confidence

    def dial_to_image(self, pts):
        # dial frame: unit = texture side, y up; R_FACE (0.44) maps to the radius
        s = self.ellipse[2] / gd.R_FACE
        return np.array([[self.center[0] + s * x, self.center[1] - s * y] for x, y in pts], float)

    def needle_tip_px(self, r=gd.R_NEEDLE):
        a = math.radians(self.angle_deg)
        return tuple(self.dial_to_image([[r * math.cos(a), r * math.sin(a)]])[0])


def _boxes(monkeypatch, fn):
    """Record the filled black label boxes drawn by ``fn``."""
    boxes = []
    rect = cv2.rectangle

    def rec(img, p0, p1, color, thickness=1, *a, **k):
        if thickness == -1 and tuple(color) == (0, 0, 0):
            boxes.append((p0[0], p0[1], p1[0], p1[1]))
        return rect(img, p0, p1, color, thickness, *a, **k)
    monkeypatch.setattr(cv2, "rectangle", rec)
    fn()
    monkeypatch.setattr(cv2, "rectangle", rect)
    return boxes


@pytest.mark.parametrize("radius", [30.0, 70.0, 120.0])        # text scale 0.45 .. 0.8
@pytest.mark.parametrize("cy", [40.0, 300.0])                   # clamped below the banner / free above the dial
def test_overlay_label_boxes_never_overlap(monkeypatch, radius, cy):
    img = np.zeros((600, 800, 3), np.uint8)
    rd = FakeReading(400, cy, radius)
    lines = ["G05 3.70 bar", "NORMAL  c=0.93 n=15"]
    top = gr.HEADER_HEIGHT + 2
    boxes = _boxes(monkeypatch, lambda: gr.draw_reading(img, rd, lines, "normal", top=top))
    assert len(boxes) == 2
    for (x0, y0, x1, y1), (_a, n0, _b, _n1) in zip(boxes, boxes[1:]):
        assert y1 < n0, f"line boxes overlap: {boxes}"
    assert min(b[1] for b in boxes) >= top, f"label reaches into the header banner: {boxes}"
    if cy - radius > 120:                                       # room above the dial: label above it
        assert boxes[-1][3] < cy - radius


def test_overlay_without_lines_draws_only_the_dial():
    img = np.zeros((200, 200, 3), np.uint8)
    gr.draw_reading(img, FakeReading(100, 100, 40), [], "warning")
    assert img.any()
    gr.draw_reading(img, FakeReading(100, 100, 40), ["", None], "warning")      # empty lines are skipped


def test_value_format_follows_the_dial_range():
    assert value_decimals((0.0, 1.6)) == 3 and value_decimals((0.0, 16.0)) == 2 and value_decimals((0, 100)) == 1
    assert format_value(0.9204, "MPa", (0.0, 1.6)) == "0.920 MPa"
    assert format_value(18.5, "bar", (0.0, 25.0)) == "18.5 bar"
    assert format_value(None, "bar", (0.0, 10.0)) == format_value(float("nan"), "bar") == "NO READING"


def _scene(value_angle=60.0):
    img = np.full((600, 800, 3), 90, np.uint8)
    face = gd.render_gauge_face(4.0, 0.0, 10.0, "bar", "TEST", size=240)
    img[180:420, 280:520] = face
    return img, FakeReading(400, 300, 0.44 * 240, value_angle)


def _decode(data):
    assert data[:3] == b"\xff\xd8\xff"
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def test_render_snapshot_crop_footer_and_full_frame():
    img, rd = _scene()
    info = {"id": "G05", "tag": "TANK-PI-02", "value": 3.7012, "unit": "bar", "status": "warning",
            "confidence": 0.93, "n_samples": 12, "range": [0.0, 10.0], "t_sim": 123.4, "wall": wall_time_text(0)}
    out = render_gauge_snapshot(img, rd, info, crop_px=320, full_width=640, quality=85)
    crop, full = _decode(out["crop"]), _decode(out["full"])
    assert crop.shape[1] == 320 and crop.shape[0] == out["height"] > 320 and out["width"] == 320
    assert full.shape[:2] == (480, 640)                          # 800 x 600 downscaled
    assert len(out["crop"]) < 60 * 1024                           # small enough for a JSON message
    foot = crop[330:, :].astype(int)
    warn = np.array(gr.STATUS_COLORS["warning"])
    assert (np.abs(foot - warn).sum(axis=2) < 90).sum() > 50      # status colour bar + value text
    # the dial is centred in the crop: the face (white) fills the middle
    mid = crop[110:210, 110:210].mean()
    assert mid > 150
    assert not render_gauge_snapshot(img, rd, info, full_width=0)["full"]


def test_render_snapshot_without_detection_and_with_roi():
    img, _rd = _scene()
    info = {"id": "G01", "value": None, "status": "normal", "range": [0, 10], "t_sim": None}
    out = render_gauge_snapshot(img, None, info, crop_px=200, full_width=0)
    crop = _decode(out["crop"])
    assert crop.shape[1] == 200 and crop.shape[0] > 200
    out = render_gauge_snapshot(img, None, dict(info, value=4.0), crop_px=200, full_width=0, roi=(400, 300, 110))
    crop = _decode(out["crop"])
    assert crop[60:140, 60:140].mean() > 150                     # dial face centred in the crop
    with pytest.raises(ValueError):
        render_gauge_snapshot(None, None, info)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                 # mono cameras work too
    assert _decode(render_gauge_snapshot(gray, None, info, full_width=0)["crop"]).ndim == 3


def test_collector_prefers_the_frame_that_agrees_with_the_final_value():
    col = SnapshotCollector(candidates=3)
    frames = [np.full((10, 10, 3), k, np.uint8) for k in range(6)]
    item = {"value": 5.0, "range": [0.0, 10.0], "mad": 0.0}
    col.frame("G01", frames[0], 1.0)                              # nothing detected yet
    # (confidence, own value): the most confident frame read an outlier (6.0)
    for k, (conf, val) in enumerate([(0.95, 6.0), (0.80, 5.02), (0.85, 4.97), (0.70, 5.0), (0.60, 5.0)], start=1):
        col.frame("G01", frames[k], 1.0 + k)
        col.reading("G01", frames[k], FakeReading(5, 5, 3, confidence=conf), dict(item), val, 1.0 + k)
    v = col.finish("G01")
    assert v["frames"] == 6 and v["readings"] == 5 and v["item"]["value"] == 5.0
    assert v["frame_confidence"] == pytest.approx(0.85) and v["frame_value"] == pytest.approx(4.97)
    assert v["bgr"] is frames[3] and v["t"] == 4.0
    assert col.finish("G01") is None and col.active() == []      # the visit is over


def test_collector_falls_back_to_closest_value_and_to_the_last_frame():
    col = SnapshotCollector(candidates=2)
    f = np.zeros((4, 4, 3), np.uint8)
    col.reading("G02", f, FakeReading(1, 1, 1, confidence=0.9), {"value": 5.0, "range": [0, 10]}, 7.0, 1.0)
    col.reading("G02", f, FakeReading(1, 1, 1, confidence=0.5), {"value": 5.0, "range": [0, 10]}, 5.5, 2.0)
    assert col.finish("G02")["frame_value"] == 5.5               # nobody within 1 % FS: the closest one
    last = np.ones((4, 4, 3), np.uint8)
    col.frame("G03", f, 1.0)
    col.frame("G03", last, 2.0)
    v = col.finish("G03")
    assert v["rd"] is None and v["bgr"] is last and v["item"] is None and v["frames"] == 2
    col.frame("G04", f, 1.0)
    col.discard("G04")
    assert col.finish("G04") is None


def test_snapshot_message_carries_base64_jpegs():
    img, rd = _scene()
    info = {"id": "G05", "tag": "T", "value": 3.7, "unit": "bar", "status": "normal", "confidence": 0.9,
            "n_samples": 5, "range": [0.0, 10.0], "t_sim": 1.0, "wall": "w"}
    imgs = render_gauge_snapshot(img, rd, info)
    d = snapshot_message(info, imgs, stamp=1.0, read=True)
    assert d["format"] == "jpeg" and d["id"] == "G05" and d["stamp"] == 1.0 and d["read"] is True
    assert base64.b64decode(d["jpeg_b64"]) == imgs["crop"]
    assert base64.b64decode(d["full_jpeg_b64"]) == imgs["full"]
    assert snapshot_message(info, dict(imgs, full=None))["full_jpeg_b64"] is None
