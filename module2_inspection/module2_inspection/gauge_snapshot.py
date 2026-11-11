"""
Evidence snapshot of one gauge inspection (ROS-free, numpy + OpenCV).

When the mission accepts a gauge it clears ``/inspection/target``; gauge_reader_node then
publishes ONE annotated JPEG of that visit on ``/inspection/gauge_snapshot``:

* :class:`SnapshotCollector` keeps, per inspection visit (frames processed while the target was
  set), the latest fused reading of the target and a few candidate frames with the target
  detected (highest per-frame confidence).  When the visit ends it picks the candidate whose
  own needle reading agrees with the final fused value (so the drawn needle shows the reported
  value) - or, without any detection, the latest frame (evidence of a missed gauge).
* :func:`render_gauge_snapshot` draws the fitted dial edge and needle, crops a square around the
  dial and adds a footer with id, tag, value, unit, status, confidence, samples, simulation time
  and wall-clock time (ASCII only: the OpenCV fonts have no Hangul).  Optionally a downscaled
  full frame with the same annotation.
* :func:`snapshot_message` builds the JSON dict of the topic (images as base64 JPEG).

Message (std_msgs/String JSON, DESIGN_SPEC section 4)::

    {stamp, t_end, wall, id, tag, value, unit, status, confidence, n_samples, range: [min, max],
     read, frames, readings, frame_value, frame_confidence, robot_pose, distance, incidence_deg,
     format: "jpeg", width, height, jpeg_b64, full_jpeg_b64 | null}
"""
from __future__ import annotations

import base64
import datetime
import math
from typing import Dict, Optional

import cv2
import numpy as np

from module2_inspection.gauge_reader import HEADER_HEIGHT, STATUS_COLORS, draw_header, draw_reading

SNAPSHOT_FORMAT = "jpeg"
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _finite(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def value_decimals(vrange) -> int:
    """Decimals for a value on a dial of this range (span <= 2: 3, <= 20: 2, else 1)."""
    try:
        span = abs(float(vrange[1]) - float(vrange[0]))
    except (TypeError, ValueError, IndexError):
        return 2
    return 3 if span <= 2.0 else (2 if span <= 20.0 else 1)


def format_value(value, unit="", vrange=None) -> str:
    v = _finite(value)
    if v is None:
        return "NO READING"
    txt = f"{v:.{value_decimals(vrange) if vrange is not None else 2}f}"
    return f"{txt} {unit}".strip()


def _ascii(s) -> str:
    """OpenCV's Hershey fonts draw ASCII only."""
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in str(s or ""))


def _fit_text(img, text, org, scale, color, thick, max_w):
    """putText, shrinking the scale until the text fits ``max_w`` px."""
    text = _ascii(text)
    (tw, _th), _bl = cv2.getTextSize(text, _FONT, scale, thick)
    if tw > max_w > 0:
        scale = max(0.3, scale * max_w / float(tw))
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


# ------------------------------------------------------------------------------ collection
class SnapshotCollector:
    """Candidate frames of the current inspection visit of each target gauge."""

    def __init__(self, candidates: int = 3, agree_fs: float = 0.01):
        self.candidates = max(1, int(candidates))
        self.agree_fs = float(agree_fs)          # |frame value - fused value| tolerance [fraction of FS]
        self._visits: Dict[str, dict] = {}

    def active(self):
        return sorted(self._visits)

    def _visit(self, gid: str, t: float) -> dict:
        return self._visits.setdefault(gid, {"frames": 0, "readings": 0, "t0": t, "last": None, "cands": [],
                                             "item": None})

    def frame(self, gid: str, bgr: np.ndarray, t: float):
        """Every processed frame while ``gid`` is the inspection target."""
        v = self._visit(gid, t)
        v["frames"] += 1
        v["last"] = (bgr, t)

    def reading(self, gid: str, bgr: np.ndarray, rd, item: dict, frame_value, t: float):
        """A detection of the target in this frame: ``item`` = published (fused) reading dict,
        ``frame_value`` = the value of this frame's own needle."""
        v = self._visit(gid, t)
        v["readings"] += 1
        v["item"] = dict(item)
        conf = float(getattr(rd, "confidence", 0.0) or 0.0)
        v["cands"].append({"conf": conf, "bgr": bgr, "rd": rd, "value": _finite(frame_value), "t": t})
        # keep the best ``candidates`` frames (confidence, newer first on ties)
        v["cands"].sort(key=lambda c: (c["conf"], c["t"]), reverse=True)
        del v["cands"][self.candidates:]

    def discard(self, gid: str):
        self._visits.pop(gid, None)

    def finish(self, gid: str) -> Optional[dict]:
        """End of the visit of ``gid`` -> {bgr, rd, item, t, frames, readings, frame_value,
        frame_confidence} (rd None when the gauge was never detected) or None (no frame at all)."""
        v = self._visits.pop(gid, None)
        if v is None or (v["last"] is None and not v["cands"]):
            return None
        out = {"frames": v["frames"], "readings": v["readings"], "item": v["item"], "t0": v["t0"]}
        if v["cands"]:
            item = v["item"] or {}
            final = _finite(item.get("value"))
            rng = item.get("range") or [0.0, 1.0]
            try:
                fs = abs(float(rng[1]) - float(rng[0])) or 1.0
            except (TypeError, ValueError, IndexError):
                fs = 1.0
            best = v["cands"][0]
            if final is not None:
                tol = max(self.agree_fs * fs, 3.0 * 1.4826 * float(_finite(item.get("mad")) or 0.0))
                agree = [c for c in v["cands"] if c["value"] is not None and abs(c["value"] - final) <= tol]
                if agree:
                    best = agree[0]
                else:
                    best = min(v["cands"], key=lambda c: abs(c["value"] - final) if c["value"] is not None
                               else float("inf"))
            out.update(bgr=best["bgr"], rd=best["rd"], t=best["t"], frame_value=best["value"],
                       frame_confidence=best["conf"])
        else:
            bgr, t = v["last"]
            out.update(bgr=bgr, rd=None, t=t, frame_value=None, frame_confidence=None)
        return out


# ------------------------------------------------------------------------------ rendering
def _crop_square(img: np.ndarray, cx: float, cy: float, half: int, out_px: int) -> np.ndarray:
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    u, v = int(round(cx)) + half, int(round(cy)) + half
    crop = pad[v - half:v + half, u - half:u + half]
    interp = cv2.INTER_AREA if 2 * half > out_px else cv2.INTER_CUBIC
    return cv2.resize(crop, (out_px, out_px), interpolation=interp)


def render_gauge_snapshot(bgr: np.ndarray, rd, info: dict, crop_px: int = 320, full_width: int = 640,
                          quality: int = 85, roi=None) -> Dict[str, object]:
    """Annotated evidence images of one gauge inspection.

    ``rd``: GaugeReading of the chosen frame (None: gauge not detected); ``info``: id, tag, value,
    unit, status, confidence, n_samples, range, t_sim, wall.  ``roi`` = (cx, cy, r) px crops around
    a known dial position when there is no reading (synthetic frames).  Returns
    {"crop": JPEG bytes, "full": JPEG bytes or None, "width", "height"}: crop = square around the
    dial (fitted edge + needle drawn) with a text footer; full = the whole frame with the overlay
    label, downscaled to ``full_width`` px (0 = not rendered)."""
    if bgr is None or getattr(bgr, "ndim", 0) not in (2, 3):
        raise ValueError("snapshot needs an image")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    crop_px = int(np.clip(int(crop_px), 96, 1024))
    quality = int(np.clip(int(quality), 30, 100))
    status = str(info.get("status") or "unknown")
    if rd is None and status not in ("normal", "warning", "alarm"):
        status = "unknown"
    col = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
    unit = str(info.get("unit") or "")
    vtxt = format_value(info.get("value"), unit, info.get("range"))
    conf = _finite(info.get("confidence"))
    n = info.get("n_samples") or 0
    t_sim = _finite(info.get("t_sim"))
    gid = _ascii(info.get("id") or "?")
    H, W = bgr.shape[:2]

    # ---- crop around the dial
    if rd is not None:
        src = bgr.copy()
        cx, cy, a, b, _ang = rd.ellipse
        half = int(max(1.45 * max(a, b), 48))
        th = int(max(1, round(2.0 * (2 * half) / crop_px)))
        draw_reading(src, rd, [], status, thickness=th)
        crop = _crop_square(src, cx, cy, half, crop_px)
    elif roi is not None:
        cx, cy, r = (float(x) for x in roi)
        crop = _crop_square(bgr, cx, cy, int(max(1.3 * r, 48)), crop_px)
    else:                                         # not detected: the whole frame, letterboxed
        s = crop_px / float(max(H, W))
        small = cv2.resize(bgr, (max(1, int(round(W * s))), max(1, int(round(H * s)))), interpolation=cv2.INTER_AREA)
        crop = np.full((crop_px, crop_px, 3), 40, np.uint8)
        y0, x0 = (crop_px - small.shape[0]) // 2, (crop_px - small.shape[1]) // 2
        crop[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small

    # ---- footer: id + tag / value + status / confidence + samples / times
    fs = crop_px / 320.0
    foot_h = int(round(100 * fs))
    foot = np.full((foot_h, crop_px, 3), 18, np.uint8)
    cv2.rectangle(foot, (0, 0), (crop_px, max(3, int(round(4 * fs)))), col, -1)
    w = crop_px - int(round(16 * fs))
    x = int(round(8 * fs))
    _fit_text(foot, f"{gid}  {info.get('tag') or ''}", (x, int(round(24 * fs))), 0.55 * fs, (235, 235, 235),
              1, w)
    _fit_text(foot, f"{vtxt}   {status.upper() if rd is not None or vtxt != 'NO READING' else ''}".rstrip(),
              (x, int(round(52 * fs))), 0.75 * fs, col, 2, w)
    _fit_text(foot, (f"conf {conf:.2f}" if conf is not None else "conf -") + f"   samples {int(n)}",
              (x, int(round(73 * fs))), 0.45 * fs, (205, 205, 205), 1, w)
    _fit_text(foot, (f"t = {t_sim:.1f} s" if t_sim is not None else "t = -") + f"   {info.get('wall') or ''}",
              (x, int(round(92 * fs))), 0.45 * fs, (205, 205, 205), 1, w)
    panel = np.vstack([crop, foot])
    ok, enc = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    out = {"crop": enc.tobytes(), "full": None, "width": int(panel.shape[1]), "height": int(panel.shape[0])}

    # ---- full frame with the regular overlay label
    if full_width and int(full_width) > 0:
        full = bgr.copy()
        if rd is not None:
            draw_reading(full, rd, [f"{gid} {vtxt}", f"{status.upper()}  c={conf or 0.0:.2f} n={int(n)}"], status,
                         top=HEADER_HEIGHT + 2)
        if W > int(full_width):
            full = cv2.resize(full, (int(full_width), int(round(H * int(full_width) / float(W)))),
                              interpolation=cv2.INTER_AREA)
        draw_header(full, _ascii(f"{gid} {info.get('tag') or ''}  {vtxt}  {status.upper()}  "
                                 + (f"t={t_sim:.1f}s  " if t_sim is not None else "") + str(info.get("wall") or "")),
                    col)
        ok, enc = cv2.imencode(".jpg", full, [cv2.IMWRITE_JPEG_QUALITY, max(30, quality - 5)])
        if ok:
            out["full"] = enc.tobytes()
    return out


def wall_time_text(now: Optional[float] = None) -> str:
    d = datetime.datetime.now() if now is None else datetime.datetime.fromtimestamp(float(now))
    return d.strftime("%Y-%m-%d %H:%M:%S")


def snapshot_message(info: dict, images: dict, **extra) -> dict:
    """JSON dict of /inspection/gauge_snapshot from ``info`` (reading fields) and the images of
    :func:`render_gauge_snapshot`."""
    d = {k: info.get(k) for k in ("id", "tag", "value", "unit", "status", "confidence", "n_samples", "range")}
    d.update(extra)
    d.update(format=SNAPSHOT_FORMAT, width=images.get("width"), height=images.get("height"),
             jpeg_b64=base64.b64encode(images["crop"]).decode("ascii"),
             full_jpeg_b64=base64.b64encode(images["full"]).decode("ascii") if images.get("full") else None)
    return d
