"""Inspection evidence in the metrics session: photos of /inspection/gauge_snapshot saved under
gauges/, the mission's official per-gauge results ('gauge result' events), inspection_report.json
and the self-contained Korean inspection_report.html, /metrics/session info for the dashboard."""
import base64
import csv
import json
import os

import cv2
import numpy as np
import pytest

from dusan_common.layout import Layout

from plant_simulation import inspection_report as report
from plant_simulation.metrics import MetricsRecorder, make_session_dir


@pytest.fixture(scope="module")
def layout():
    return Layout.load()


def _jpeg(color=(30, 140, 200), size=(48, 64)):
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img[:] = color
    return cv2.imencode(".jpg", img)[1].tobytes()


def _snap(gid, stamp, value, status="normal", full=True, **kw):
    d = {"id": gid, "tag": "TAG", "value": value, "unit": "bar", "status": status, "confidence": 0.9,
         "n_samples": 8, "range": [0.0, 10.0], "stamp": stamp, "wall": "2026-11-11 10:00:00", "read": value is not None,
         "frames": 9, "readings": 8, "format": "jpeg", "jpeg_b64": base64.b64encode(_jpeg()).decode(),
         "full_jpeg_b64": base64.b64encode(_jpeg((10, 10, 10), (96, 128))).decode() if full else None}
    d.update(kw)
    return d


def _events(rec):
    rec.flush()
    with open(os.path.join(rec.dir, "events.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _report(d):
    with open(os.path.join(d, "inspection_report.json"), encoding="utf-8") as f:
        return json.load(f)


def test_snapshot_photos_are_saved_in_the_session(tmp_path, layout):
    d = make_session_dir(str(tmp_path), "s")
    rec = MetricsRecorder(d, layout)
    changes = []
    rec.on_change = lambda: changes.append(rec.session_info())
    meta = rec.on_gauge_snapshot(50.0, _snap("G05", 49.84, 3.71))
    assert meta["image"] == "gauges/G05_49.8.jpg" and meta["image_full"] == "gauges/G05_49.8_full.jpg"
    with open(os.path.join(d, "gauges", "G05_49.8.jpg"), "rb") as f:
        assert f.read() == _jpeg()
    assert os.path.isfile(os.path.join(d, "gauges", "G05_49.8_full.jpg"))
    assert "jpeg_b64" not in meta and meta["count"] == 1
    # the same message delivered twice is saved once
    assert rec.on_gauge_snapshot(50.1, _snap("G05", 49.84, 3.71)) is meta
    # a second visit of the same gauge: new file, latest photo wins
    rec.on_gauge_snapshot(90.0, _snap("G05", 89.0, 3.69, full=False))
    assert rec.snapshots["G05"]["image"] == "gauges/G05_89.0.jpg" and rec.snapshots["G05"]["image_full"] is None
    assert rec.snapshots["G05"]["count"] == 2
    # same stamp, different photo: no overwrite
    rec.on_gauge_snapshot(91.0, _snap("G06", 89.0, 1.2, jpeg_b64=base64.b64encode(_jpeg((1, 2, 3))).decode()))
    rec.on_gauge_snapshot(92.0, _snap("G06", 89.0, 1.2, jpeg_b64=base64.b64encode(_jpeg((200, 2, 3))).decode()))
    assert sorted(os.listdir(os.path.join(d, "gauges"))) == [
        "G05_49.8.jpg", "G05_49.8_full.jpg", "G05_89.0.jpg", "G06_89.0.jpg", "G06_89.0_2.jpg",
        "G06_89.0_2_full.jpg", "G06_89.0_full.jpg"]
    ev = [e for e in _events(rec) if e["source"] == "gauge" and e["kind"] == "snapshot"]
    assert len(ev) == 4 and ev[0]["detail"] == "G05 3.7100 bar (normal) -> gauges/G05_49.8.jpg"
    info = changes[-1]
    assert info["session_dir"] == os.path.abspath(d) and info["report"] == "inspection_report.html"
    assert info["gauges"]["G05"]["image"] == "gauges/G05_89.0.jpg" and info["report_final"] is False
    rec.close()


@pytest.mark.parametrize("bad", [
    {"id": "../../evil", "jpeg_b64": base64.b64encode(_jpeg()).decode()},
    {"id": "G0 1", "jpeg_b64": base64.b64encode(_jpeg()).decode()},
    {"id": "G01/x", "jpeg_b64": base64.b64encode(_jpeg()).decode()},
    {"id": None, "jpeg_b64": base64.b64encode(_jpeg()).decode()},
    {"id": "G01", "jpeg_b64": "not base64 !!"},
    {"id": "G01", "jpeg_b64": base64.b64encode(b"<html>not a jpeg</html>").decode()},
    {"id": "G01", "jpeg_b64": None},
    "not a dict",
])
def test_bad_snapshots_write_nothing(tmp_path, layout, bad):
    d = make_session_dir(str(tmp_path), "s")
    rec = MetricsRecorder(d, layout)
    assert rec.on_gauge_snapshot(1.0, bad) is None
    assert not os.path.exists(os.path.join(d, "gauges")) and rec.snapshots == {}
    assert not any(n.endswith(".jpg") for n in os.listdir(str(tmp_path)))
    rec.close()


def test_oversized_snapshot_is_refused(tmp_path, layout):
    d = make_session_dir(str(tmp_path), "s")
    rec = MetricsRecorder(d, layout, params={"snapshot_max_bytes": 100})
    assert rec.on_gauge_snapshot(1.0, _snap("G01", 1.0, 6.2)) is None
    rec.close()


def _mission(rec, t, state, results, **kw):
    d = {"state": state, "results": results, "completed": [k for k, v in results.items() if v.get("status") != "missed"],
         "missed": [k for k, v in results.items() if v.get("status") == "missed"]}
    d.update(kw)
    rec.on_mission(t, d)


def test_official_results_events_and_final_report(tmp_path, layout):
    d = make_session_dir(str(tmp_path), "run")
    wall = [1000.0]
    rec = MetricsRecorder(d, layout, wall_clock=lambda: wall[0])
    g8, g6 = layout.gauge("G08"), layout.gauge("G06")
    _mission(rec, 0.0, "IDLE", {})
    assert not os.path.exists(os.path.join(d, "inspection_report.html"))       # nothing to report yet
    _mission(rec, 1.0, "PATROL", {}, pending=["G08", "G06", "G05"])
    # the live readings drift after the mission accepted G08 (the robot walks away)
    rec.on_gauge_readings(10.0, {"readings": [{"id": "G08", "value": 0.93, "status": "normal", "confidence": 0.9}]})
    accepted = {"value": 0.9216, "unit": "MPa", "status": "normal", "confidence": 0.95, "t": 12.0, "n_readings": 6}
    _mission(rec, 12.5, "PATROL", {"G08": accepted}, pending=["G06", "G05"])
    rec.on_gauge_snapshot(12.5, _snap("G08", 12.0, 0.9216, unit="MPa"))
    rec.on_gauge_readings(20.0, {"readings": [{"id": "G08", "value": 1.05, "status": "normal", "confidence": 0.4}]})
    _mission(rec, 40.0, "PATROL", {"G08": accepted, "G06": {"status": "missed", "reason": "no reading"}},
             pending=["G05"])
    _mission(rec, 40.5, "PATROL", {"G08": accepted, "G06": {"status": "missed", "reason": "no reading"}},
             pending=["G05"])                                                   # unchanged: no new event
    ev = [e["detail"] for e in _events(rec) if e["kind"] == "result"]
    assert ev == ["G08 read: 0.9216 MPa (normal) from 6 readings, error 0.10 % FS", "G06 missed: no reading"]
    r = _report(d)
    assert r["final"] is False and r["mission"]["state"] == "PATROL"
    rows = {x["id"]: x for x in r["gauges"]["rows"]}
    assert rows["G05"]["result"] == "pending" and rows["G01"]["result"] == "not_visited"
    wall[0] = 1500.0
    _mission(rec, 300.0, "RTH", {"G08": accepted, "G06": {"status": "missed", "reason": "no reading"}}, pending=[])
    _mission(rec, 330.0, "DOCKED", {"G08": accepted, "G06": {"status": "missed", "reason": "no reading"}}, pending=[])
    r = _report(d)
    assert r["final"] is True and r["mission"]["state"] == "DOCKED" and r["session"] == "run"
    rows = {x["id"]: x for x in r["gauges"]["rows"]}
    assert len(rows) == len(layout.gauges)
    g = rows["G08"]                                         # the OFFICIAL value, not the last live reading
    assert g["result"] == "read" and g["value_source"] == "mission" and g["value"] == pytest.approx(0.9216)
    assert g["true_value"] == g8["value"] and g["status"] == "normal" and g["true_status"] == "normal"
    assert g["error_pct_fs"] == pytest.approx(abs(0.9216 - g8["value"]) / (g8["max"] - g8["min"]) * 100, abs=1e-3)
    assert g["status_correct"] is True and g["n_readings"] == 6 and g["t_sim"] == 12.0
    assert g["image"] == "gauges/G08_12.0.jpg" and g["image_full"] == "gauges/G08_12.0_full.jpg"
    assert g["wall"] == "2026-11-11 10:00:00" and g["confidence"] == pytest.approx(0.95)
    assert rows["G06"]["result"] == "missed" and rows["G06"]["value"] is None and rows["G06"]["reason"] == "no reading"
    assert rows["G06"]["true_value"] == g6["value"] and rows["G05"]["result"] == "not_visited"
    gz = r["gauges"]
    assert gz["read"] == 1 and gz["missed"] == ["G06"] and gz["status_correct"] == 1 and gz["total"] == 8
    assert gz["mean_error_pct_fs"] == pytest.approx(g["error_pct_fs"], abs=1e-3)
    reports = [e["detail"] for e in _events(rec) if e["kind"] == "report"]
    assert reports[0].startswith("inspection_report.html (partial") and reports[-1].startswith(
        "inspection_report.html (final: 1/8")
    rec.close()
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    assert s["inspection"]["report"] == "inspection_report.html" and s["inspection"]["report_final"] is True
    assert s["inspection"]["photos"] == {"G08": "gauges/G08_12.0.jpg"}
    assert s["inspection"]["mission_results"]["G06"]["status"] == "missed"


def test_report_html_is_korean_self_contained_and_escaped(tmp_path, layout):
    d = make_session_dir(str(tmp_path), "html")
    rec = MetricsRecorder(d, layout)
    rec.on_thermal(5.0, {"hotspots": [{"id": "H02", "max_temp_c": 98.4, "status": "alarm"}]})
    rec.on_seek_status(6.0, {"state": "FOUND", "estimate": [-3.0, 2.3], "error_m": 0.31})
    rec.on_gauge_snapshot(9.0, _snap("G01", 8.0, 6.25, wall="<img src=x onerror=alert(1)>"))
    _mission(rec, 9.5, "DONE", {"G01": {"value": 6.25, "status": "<script>alert(1)</script>", "confidence": 0.9,
                                        "t": 8.0, "n_readings": 5},
                                "G02": {"status": "missed", "reason": "<script>alert(2)</script>"}},
             seek={"result": "FOUND", "estimate": [-3.0, 2.3], "error_m": 0.31, "elapsed": 20.0})
    with open(os.path.join(d, "inspection_report.html"), encoding="utf-8") as f:
        page = f.read()
    assert page.startswith("<!doctype html>") and 'lang="ko"' in page
    for text in ("설비 점검 보고서", "계기 판독 결과", "열화상 점검", "가스 누출 탐지", "임무 요약", "임무 종료", "발견", "경보"):
        assert text in page, text
    assert "data:image/jpeg;base64," in page                          # photo embedded
    assert 'href="gauges/G01_8.0_full.jpg"' in page                    # full frame next to the report
    assert "<script" not in page.lower() and "<img src=x" not in page            # injected markup is inert text
    assert "&lt;script&gt;alert(2)" in page and "&lt;img src=x onerror=alert(1)&gt;" in page
    r = _report(d)
    assert {x["id"]: x for x in r["gauges"]["rows"]}["G01"]["status"] == "normal"     # unknown status: from the layout
    assert "http://" not in page and "https://" not in page          # no external resources
    r = _report(d)
    assert r["thermal"]["rows"][1]["id"] == "H02" and r["thermal"]["rows"][1]["detected"] is True
    assert r["thermal"]["rows"][1]["peak_temp_c"] == 98.4 and r["thermal"]["rows"][1]["status"] == "alarm"
    assert r["gas"]["mission_seek"]["result"] == "FOUND"
    rec.close()


def test_a_new_mission_archives_the_previous_report(tmp_path, layout):
    d = make_session_dir(str(tmp_path), "two")
    rec = MetricsRecorder(d, layout)
    res = {"G01": {"value": 6.2, "status": "normal", "t": 5.0, "n_readings": 5}}
    rec.on_gauge_snapshot(5.0, _snap("G01", 5.0, 6.2))
    _mission(rec, 5.5, "PATROL", res)
    _mission(rec, 60.0, "DOCKED", res)
    _mission(rec, 100.0, "PATROL", {})                                  # 'start' again: results cleared
    assert os.path.isfile(os.path.join(d, "inspection_report_1.html"))
    assert os.path.isfile(os.path.join(d, "inspection_report_1.json"))
    assert not os.path.exists(os.path.join(d, "inspection_report.html")) and rec.snapshots == {}
    assert os.path.isfile(os.path.join(d, "gauges", "G01_5.0.jpg"))     # photos are kept
    rec.close()
    r = _report(d)                                                      # close writes the current mission
    assert {x["id"]: x["result"] for x in r["gauges"]["rows"]}["G01"] == "not_visited"


def test_report_rows_fall_back_to_photo_and_live_values(layout):
    gauges = {g["id"]: g for g in layout.gauges}
    rows = report.gauge_rows(gauges, layout, {}, {"G02": {"value": 11.3, "status": "warning", "image": "gauges/x.jpg"}},
                             {"G03": {"value": 2.31, "status": "normal", "t": 3.0}}, pending=["G04"])
    by = {r["id"]: r for r in rows}
    assert (by["G02"]["value_source"], by["G02"]["status"], by["G02"]["image"]) == ("snapshot", "warning", "gauges/x.jpg")
    assert (by["G03"]["value_source"], by["G03"]["t_sim"]) == ("readings", 3.0)
    assert by["G04"]["result"] == "pending" and by["G01"]["result"] == "not_visited"
    rows = report.gauge_rows(gauges, layout, {}, {"G02": {"value": 1.0, "image": "../../etc/passwd"}}, {})
    assert {r["id"]: r for r in rows}["G02"]["image"] is None             # unsafe path never reaches the page
    assert report.safe_rel("gauges/G01_1.0.jpg") and not report.safe_rel("gauges/.x.jpg")
    assert not report.safe_rel("/abs.jpg") and not report.safe_rel("a/../b.jpg") and not report.safe_rel(None)


def test_malformed_results_and_snapshots_never_break_the_report(tmp_path, layout):
    d = make_session_dir(str(tmp_path), "odd")
    rec = MetricsRecorder(d, layout)
    rec.on_gauge_snapshot(1.0, _snap("G01", 1.0, "n/a", status=["x"], wall=5, readings={"a": 1}))
    _mission(rec, 2.0, "PATROL", {"G01": {"value": "6.2", "status": {"bad": 1}, "reason": ["?"]},
                                  "G02": {"status": "missed", "reason": {"x": 1}}, "Gxx": {"value": 1.0}})
    r = _report(d)
    rows = {x["id"]: x for x in r["gauges"]["rows"]}
    assert rows["G01"]["result"] == "read" and rows["G01"]["value"] == pytest.approx(6.2)
    assert rows["G01"]["status"] == layout.gauge_status("G01", 6.2) and rows["G01"]["wall"] is None
    assert rows["G02"]["result"] == "missed" and isinstance(rows["G02"]["reason"], str)
    _mission(rec, 3.0, "DONE", {"G01": {"value": float("nan"), "status": None}})
    rows = {x["id"]: x for x in _report(d)["gauges"]["rows"]}
    assert rows["G01"]["result"] == "no_value" and rows["G01"]["value"] is None          # no number, no guess
    assert rows["G03"]["result"] == "not_visited"
    assert not any(e["kind"] == "report_error" for e in _events(rec))
    with open(os.path.join(d, "inspection_report.html"), encoding="utf-8") as f:
        assert "설비 점검 보고서" in f.read()
    rec.close()
