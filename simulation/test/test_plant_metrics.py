"""Metrics recorder (CSV + summary.json) and tools/plot_metrics.py."""
import csv
import json
import math
import os
import sys

import pytest

from dusan_common.layout import Layout

from plant_simulation.metrics import CSV_COLUMNS, MetricsRecorder, make_session_dir

# tools/ is not a package: make plot_metrics importable without the repo-level conftest (colcon test)
_TOOLS = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools"))
if os.path.isdir(_TOOLS) and _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


@pytest.fixture(scope="module")
def layout():
    return Layout.load()


def _rows(d, name):
    with open(os.path.join(d, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def new_recorder(d, layout, wall):
    return MetricsRecorder(d, layout, wall_clock=lambda: wall[0])


def record_run(rec, layout, wall):
    """Synthetic 60 s run touching every recorder input."""
    t = 0.0
    x, y = 61.9, 0.0
    rec.on_battery(0.0, 80.0, 25.0, False)
    rec.on_safety(0.0, {"state": "MONITOR", "rth_active": False})
    rec.on_mission(0.0, {"state": "IDLE", "message": "ready"})
    rec.on_mission(1.0, {"state": "PATROL", "message": "full mission started", "target": "G08"})
    for k in range(600):                                 # 60 s at 10 Hz, 0.3 m/s west
        t = k * 0.1
        x -= 0.03
        rec.on_pose(t, x, y, -0.54, 0.01, -0.02, math.pi, -0.3, 0.0, 0.0)
        rec.on_locomotion(t, {"mode": "WALK", "gait": "trot", "mpc_solve_ms": 10.0 + (k % 10), "roll": 0.02,
                              "pitch": -0.03, "fall_count": 0})
        rec.on_gas(t, 2.0 + max(0.0, (t - 30.0)) * 2.0)   # rises above 25 ppm after ~41.5 s
        rec.on_rssi(t, -60.0 - t * 0.2)
        rec.on_battery(t, 80.0 - t * 0.2)
    rec.on_pose(61.0, x + 5.0, y, -0.54, 0, 0, 0)        # 5 m teleport -> not counted
    g = layout.gauge("G08")
    span = g["max"] - g["min"]
    for k in range(5):
        rec.on_gauge_readings(10.0 + k, {"readings": [{"id": "G08", "value": g["value"] + 0.02 * span,
                                                       "unit": "MPa", "confidence": 0.9, "status": "normal"}]})
    g7 = layout.gauge("G07")
    rec.on_gauge_readings(20.0, {"readings": [{"id": "G07", "value": g7["value"] + 0.08 * (g7["max"] - g7["min"]),
                                               "status": "normal", "robot_pose": [26.0, 0.0, 1.57]}]})
    rec.on_thermal(30.0, {"hotspots": [
        {"id": "H05", "max_temp_c": 84.0, "status": "alarm", "position": [40.0, -1.45, 1.2]},     # TP by id
        {"id": None, "max_temp_c": 99.5, "status": "alarm", "position": [5.2, 6.3, 1.0]},        # TP by position
        {"id": None, "max_temp_c": 65.0, "status": "warning", "position": [20.0, 0.0, 0.5]}]})   # FP
    rec.on_seek_status(42.0, {"state": "SEARCH"})
    rec.on_seek_status(50.0, {"state": "GRADIENT", "estimate": [-3.0, 2.2]})
    rec.on_seek_status(55.0, {"state": "FOUND", "estimate": [-3.0, 2.3], "error_m": 0.36})
    rec.on_safety(56.0, {"state": "RTH_PLANNING", "rth_active": True, "reason": "battery_low"})
    rec.on_fall_event(57.0, {"event": "fall", "x": 50.0, "y": 0.0, "roll": 1.3, "pitch": 0.0})
    rec.on_fall_event(58.5, {"event": "recovered", "x": 50.0, "y": 0.0})
    rec.on_mission(59.0, {"state": "DONE", "completed": ["G08", "G07"], "missed": []})
    wall[0] = 1075.0
    return rec


def test_csv_files_and_summary(tmp_path, layout):
    d = make_session_dir(str(tmp_path), "run")
    wall = [1000.0]
    rec = record_run(new_recorder(d, layout, wall), layout, wall)
    rec.close()
    for name, cols in CSV_COLUMNS.items():
        with open(os.path.join(d, name), encoding="utf-8") as f:
            assert f.readline().strip().split(",") == cols
    pose = _rows(d, "pose.csv")
    assert 110 <= len(pose) <= 125                     # 2 Hz over 60 s
    assert len(_rows(d, "gauge_readings.csv")) == 6
    th = _rows(d, "thermal.csv")
    assert [r["result"] for r in th] == ["TP", "TP", "FP"]
    assert th[1]["matched_id"] == "H02"
    assert len(_rows(d, "battery.csv")) in range(59, 63)
    assert len(_rows(d, "gas.csv")) >= 110
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    assert s["distance_m"] == pytest.approx(18.0, abs=0.1)          # teleport not counted
    assert s["duration_s"] == pytest.approx(61.0)
    assert s["wall_duration_s"] == pytest.approx(75.0)
    gg = s["gauges"]
    assert gg["gauges_read"] == ["G07", "G08"] and gg["gauges_total"] == 8
    assert gg["error_pct_fs_all"]["n"] == 6
    assert gg["mean_error_pct_fs"] == pytest.approx((5 * 2.0 + 8.0) / 6, abs=1e-3)
    assert gg["per_gauge"]["G07"]["error_pct_fs"] == pytest.approx(8.0, abs=1e-3)
    assert gg["within_threshold_rate"] == pytest.approx(5 / 6, abs=1e-3)
    tt = s["thermal"]
    assert tt["tp"] == 2 and tt["fp"] == 1 and tt["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert tt["expected"] == ["H02", "H03", "H04", "H05"]           # >= 60 C in the layout
    assert tt["recall"] == pytest.approx(0.5)
    gas = s["gas"]
    assert gas["first_alarm_t"] == pytest.approx(41.6, abs=0.15)
    assert gas["time_to_find_s"] == pytest.approx(13.0)
    assert gas["source_error_m"] == pytest.approx(math.hypot(0.2, 0.3), abs=1e-3)
    assert gas["result"] == "FOUND"
    lo = s["locomotion"]
    assert lo["falls"] == 1 and lo["recoveries"] == 1 and lo["recovery_rate"] == 1.0
    assert lo["mpc_solve_ms"]["mean"] == pytest.approx(14.5)
    assert s["rth"]["count"] == 1 and s["rth"]["events"][0]["reason"] == "battery_low"
    assert s["battery"]["start_pct"] == pytest.approx(80.0) and s["battery"]["min_pct"] < 69.0
    assert s["mission"]["completed"] == ["G08", "G07"] and s["mission"]["duration_s"] == pytest.approx(58.0)
    ev = _rows(d, "events.csv")
    kinds = {(r["source"], r["kind"]) for r in ev}
    assert ("gas", "alarm") in kinds and ("seeker", "found") in kinds and ("safety", "rth") in kinds
    assert ("metrics", "pose_jump") in kinds


def test_session_dir_unique(tmp_path):
    a = make_session_dir(str(tmp_path), "same")
    b = make_session_dir(str(tmp_path), "same")
    assert a != b and os.path.isdir(a) and os.path.isdir(b)


def test_summary_is_rewritten_atomically(tmp_path, layout):
    d = make_session_dir(str(tmp_path), "x")
    rec = MetricsRecorder(d, layout)
    s1 = rec.write_summary()
    rec.on_gas(1.0, 30.0)
    s2 = rec.write_summary()
    assert s1["gas"]["max_ppm"] is None and s2["gas"]["max_ppm"] == 30.0
    assert not os.path.exists(os.path.join(d, "summary.json.tmp"))
    rec.close()
    rec.close()                                       # idempotent
    rec.on_gas(2.0, 5.0)                              # ignored after close, no exception


def test_plot_metrics_creates_figures(tmp_path, layout):
    pytest.importorskip("matplotlib")
    plot_metrics = pytest.importorskip("plot_metrics")
    d = make_session_dir(str(tmp_path), "plots")
    wall = [1000.0]
    rec = record_run(new_recorder(d, layout, wall), layout, wall)
    rec.close()
    out = str(tmp_path / "fig")
    files = plot_metrics.make_figures(d, out)
    names = sorted(os.path.basename(p) for p in files)
    assert names == ["battery_comm.png", "gas.png", "gauge_errors.png", "locomotion.png", "summary.md",
                     "trajectory.png"]
    for p in files:
        assert os.path.getsize(p) > (5000 if p.endswith(".png") else 200)
    md = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    assert "Gas source error" in md and "0.36" in md
    assert plot_metrics.main([d, "--out", out]) == 0


def test_plot_metrics_handles_an_empty_session(tmp_path):
    pytest.importorskip("matplotlib")
    plot_metrics = pytest.importorskip("plot_metrics")
    d = make_session_dir(str(tmp_path), "empty")
    rec = MetricsRecorder(d, None)
    rec.close()
    files = plot_metrics.make_figures(d, str(tmp_path / "f"))
    assert len(files) == 6
