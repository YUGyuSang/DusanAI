"""Digital-twin dashboard: aggregator, renderer, HTTP/SSE server, offline demo."""
import json
import math
import os
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from dusan_common.layout import Layout

from plant_simulation.dashboard_render import (encode_jpeg, gas_rgba, placeholder_jpeg, render_gas_png,
                                               render_map_png, render_rssi_png, rssi_rgba)
from plant_simulation.dashboard_state import (DashboardState, build_static_info, downsample_path,
                                              friction_zones_from_layout)
from plant_simulation.layout_map import rasterize_layout
from plant_simulation.web_server import DashboardServer, validate_command

WEB_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "web"))


@pytest.fixture(scope="module")
def layout():
    return Layout.load()


@pytest.fixture()
def state(layout):
    return DashboardState(layout)


# ----------------------------------------------------------------------------
# aggregator
# ----------------------------------------------------------------------------
def test_static_info_contains_layout_truth(layout):
    st = build_static_info(layout)
    assert len(st["gauges"]) == len(layout.gauges)
    g2 = next(g for g in st["gauges"] if g["id"] == "G02")
    assert g2["true_status"] == "warning"           # 11.4 bar > normal_max 10
    h4 = next(h for h in st["hotspots"] if h["id"] == "H04")
    assert h4["true_status"] == "alarm" and h4["true_temp_c"] == pytest.approx(115.0)
    assert st["dock"]["x"] == pytest.approx(61.9)
    assert st["gas_leak"]["x"] == pytest.approx(-3.2)
    assert len(st["robot"]["feet"]) == 4
    json.dumps(st)                                   # JSON serialisable


def test_gauge_reading_joined_with_ground_truth(state):
    state.set_gauge_readings({"stamp": 1.0, "readings": [
        {"id": "G01", "tag": "TURBINE-PI-01", "value": 6.5, "unit": "bar", "confidence": 0.9, "status": "normal",
         "n_samples": 7}]}, now=100.0)
    snap = state.snapshot(now=101.0)
    g = next(x for x in snap["gauges"] if x["id"] == "G01")
    assert g["error"] == pytest.approx(0.3)
    assert g["error_pct_fs"] == pytest.approx(3.0)   # 0.3 bar of a 0..10 bar scale
    assert g["age"] == pytest.approx(1.0)
    unread = next(x for x in snap["gauges"] if x["id"] == "G05")
    assert unread["status"] == "unread" and unread["value"] is None
    assert any(e["source"] == "gauge" and "G01" in e["text"] for e in snap["events"])


def test_thermal_alerts_and_unassigned(state):
    state.set_thermal_alerts({"hotspots": [
        {"id": "H02", "max_temp_c": 98.0, "mean_temp_c": 80.0, "status": "alarm", "position": [5.0, 6.4, 1.0]},
        {"id": None, "max_temp_c": 70.0, "status": "warning", "position": [1.0, 1.0, 0.5]}]}, now=5.0)
    snap = state.snapshot(now=5.5)
    h2 = next(h for h in snap["thermal"] if h["id"] == "H02")
    assert h2["true_temp_c"] == pytest.approx(99.0)
    assert h2["temp_error_c"] == pytest.approx(-1.0)
    assert h2["peak_c"] == pytest.approx(98.0)
    assert len(snap["thermal_unassigned"]) == 1
    assert any(e["level"] == "alarm" and "H02" in e["text"] for e in snap["events"])


def test_event_log_from_state_transitions(state):
    state.set_mission({"state": "IDLE", "message": "ready"}, now=1.0)
    state.set_mission({"state": "PATROL", "message": "next viewpoint G08"}, now=2.0)
    state.set_mission({"state": "SEEK", "message": "GAS ALARM: 30.0 ppm > 25 ppm"}, now=3.0)
    state.set_safety({"state": "MONITOR", "rth_active": False}, now=3.5)
    state.set_safety({"state": "RTH_PLANNING", "rth_active": True, "reason": "battery_low"}, now=4.0)
    state.set_locomotion({"mode": "WALK"}, now=4.1)
    state.set_locomotion({"mode": "STAND"}, now=4.2)                 # not logged (routine)
    state.set_locomotion({"mode": "FALLEN"}, now=4.5)
    state.set_fall_event({"event": "fall", "x": 1.0, "y": 2.0, "roll": 1.2, "pitch": 0.1}, now=4.6)
    ev = state.snapshot(now=5.0)["events"]
    texts = [e["text"] for e in ev]
    assert any("PATROL -> SEEK" in t for t in texts)
    assert any(e["level"] == "alarm" and "GAS ALARM" in e["text"] for e in ev)
    assert any("RTH_PLANNING" in t and "battery_low" in t for t in texts)
    assert any("return-to-home ACTIVE" in t for t in texts)
    assert any("FALLEN" in t for t in texts)
    assert not any("WALK -> STAND" in t for t in texts)
    seqs = [e["seq"] for e in ev]
    assert seqs == sorted(seqs)


def test_gas_alarm_hysteresis_history_and_leak_error(state):
    for k in range(10):
        state.set_gas_ppm(5.0 + k * 3.0, now=10.0 + k * 0.3)       # crosses 25 ppm
    assert state.gas["alarm"]
    state.set_gas_ppm(22.0, now=14.0)
    assert state.gas["alarm"]                                       # above the clear level
    state.set_gas_ppm(10.0, now=14.5)
    assert not state.gas["alarm"]
    state.set_seek_status({"state": "SURGE"}, now=15.0)
    state.set_seek_status({"state": "FOUND", "estimate": [-3.0, 2.3]}, now=16.0)
    snap = state.snapshot(now=16.0)
    assert snap["gas"]["estimate_error_m"] == pytest.approx(math.hypot(0.2, 0.3), abs=0.01)
    assert len(snap["gas"]["history"]) >= 8
    assert all(h[0] <= 0 for h in snap["gas"]["history"])
    assert any("LEAK SOURCE FOUND" in e["text"] for e in snap["events"])


def test_layers_paths_and_topic_health(state):
    occ = np.zeros((20, 30), np.int8)
    state.set_layer("gas", occ, -1.0, -2.0, 0.25, now=1.0)
    state.set_layer("gas", occ.copy(), -1.0, -2.0, 0.25, now=1.5)     # identical re-publication
    assert state.get_layer("gas")[1]["version"] == 1
    occ2 = occ.copy()
    occ2[3, 4] = 50
    state.set_layer("gas", occ2, -1.0, -2.0, 0.25, now=2.0)
    arr, meta = state.get_layer("gas")
    assert meta["version"] == 2 and meta["width"] == 30 and arr.shape == (20, 30)
    state.set_layer("gas", occ2, -0.5, -2.0, 0.25, now=2.5)           # same cells, moved origin
    assert state.get_layer("gas")[1]["version"] == 3
    assert state.get_layer("rssi") == (None, None)
    with pytest.raises(ValueError):
        state.set_layer("x", np.zeros(5), 0, 0, 1)
    state.set_path("nav", [(i * 0.1, 0.0) for i in range(2000)])
    assert len(state.paths["nav"]) == 400 and state.paths["nav"][-1] == [199.9, 0.0]
    state.set_robot(1.0, 2.0, 0.3, 0.0, 0.0, 0.5, now=60.0)
    health = state.snapshot(now=60.5)["topics"]
    assert health["/odom"]["status"] == "ok"
    assert health["/gas/concentration_grid"]["status"] == "stale"     # 58 s old, expected every 5 s
    assert health["/battery/state"]["status"] == "never"
    assert state.snapshot(now=63.5)["topics"]["/odom"]["status"] == "late"


def test_teleop_flag_display_expires_without_heartbeat(state):
    state.set_flag("teleop_active", True, now=100.0)
    assert state.snapshot(now=101.0)["flags"]["teleop_active"]
    assert not state.snapshot(now=102.0)["flags"]["teleop_active"]      # teleop killed: heartbeat stopped
    state.set_flag("teleop_active", True, now=102.2)
    assert state.snapshot(now=102.5)["flags"]["teleop_active"]
    state.set_flag("rth_active", True, now=100.0)
    assert state.snapshot(now=200.0)["flags"]["rth_active"]             # sim-time heartbeat: not expired here


def test_zone_lookup_prefers_smallest_zone(state):
    assert state.zone_at(3.5, -3.0) == "High temperature zone"      # inside the turbine hall too
    assert state.zone_at(60.0, 0.0) == "Corridor lower level"
    assert state.zone_at(200.0, 0.0) == ""


def test_locomotion_mode_is_shown_and_its_changes_logged(state):
    state.set_locomotion({"mode": "STAND", "locomotion_mode": "stable"}, now=1.0)
    state.set_locomotion({"mode": "STAND", "locomotion_mode": "mpc"}, now=2.0)
    snap = state.snapshot(now=2.5)
    assert snap["locomotion"]["locomotion_mode"] == "mpc"
    assert [e["text"] for e in snap["events"] if "locomotion mode" in e["text"]] == ["locomotion mode stable -> mpc"]


def test_snapshot_is_json_and_nan_free(state):
    state.set_battery(float("nan"), now=1.0)
    state.set_robot(float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, now=1.0)
    state.set_locomotion({"mode": "WALK", "mpc_solve_ms": 12.0}, now=1.0)
    s = json.dumps(state.snapshot(now=2.0))
    assert "NaN" not in s


# ----------------------------------------------------------------------------
# slippery ground (module1 friction prior + slip detection)
# ----------------------------------------------------------------------------
SLIP_OFF = {"slip": False, "slip_mode": False, "slip_source": "none", "slip_speed": 0.01,
            "slip_legs": [False] * 4, "slip_events": 0, "mu_est": 0.5, "mpc_mu": 0.5, "friction_prior": None,
            "torque_ff": 0.0}


def _slip_events(snap):
    return [e for e in snap["events"] if "slip" in e["text"]]


def test_static_info_has_the_layout_plates(layout):
    zones = {z["id"]: z for z in build_static_info(layout)["friction_zones"]}
    assert set(zones) == {"C-OBS4", "F-OBS9"}                       # the two steel plates of world_layout.yaml
    assert zones["C-OBS4"] == {"id": "C-OBS4", "kind": "plate", "rect": [30.0, -1.2, 32.5, 1.2], "mu": 0.2,
                               "slippery": True}
    assert zones["F-OBS9"]["rect"] == [0.0, -2.2, 2.0, -0.7] and zones["F-OBS9"]["mu"] == pytest.approx(0.15)
    assert build_static_info(None)["friction_zones"] == []


def test_friction_zones_from_layout_skips_malformed_entries():
    lay = Layout({
        "obstacles": [
            {"id": "P1", "type": "plate", "rect": [2.0, 1.0, 0.0, -1.0], "mu": 0.2},     # corners swapped
            {"id": "P2", "type": "plate", "rect": [0, 0, 1, 1], "mu": 0.8},             # ordinary floor
            {"id": "P3", "type": "plate", "rect": [0, 0, 1, 1]},                        # no mu
            {"id": "P4", "type": "plate", "rect": [0, 0, 1], "mu": 0.1},                # bad rect
            {"id": "P5", "type": "plate", "rect": "abcd", "mu": 0.1},
            {"id": "R1", "type": "rubble", "rect": [0, 0, 1, 1], "mu": 0.1},            # not a plate
            "garbage"],
        "friction_zones": [
            {"id": "OIL", "rect": [5, 0, 6, 1], "mu": 0.12},
            {"id": "WET", "rect": [7, 0, 8, 1], "mu": 0.45},                            # low but not slippery
            {"id": "NAN", "rect": [float("nan"), 0, 1, 1], "mu": 0.1},
            {"id": "NEG", "rect": [0, 0, 1, 1], "mu": -0.1},
            None]})
    zones = {z["id"]: z for z in friction_zones_from_layout(lay)}
    assert set(zones) == {"P1", "OIL", "WET"}
    assert zones["P1"]["rect"] == [0.0, -1.0, 2.0, 1.0]
    assert zones["OIL"]["kind"] == "zone" and zones["OIL"]["slippery"]
    assert not zones["WET"]["slippery"]
    assert friction_zones_from_layout(Layout({})) == [] and friction_zones_from_layout(None) == []


def test_slip_fields_pass_through_to_the_snapshot(state):
    state.set_terrain({"class": "slippery", "recommended_gait": "trot", "roughness": 0.01, "slippery": True,
                       "friction_mu": 0.15, "friction_zones": ["F-OBS9"], "friction_distance": 0.4}, now=1.0)
    state.set_locomotion(dict(SLIP_OFF, mode="WALK", slip_mode=True, slip_source="terrain", mpc_mu=0.15,
                              friction_prior=0.15, torque_ff=float("nan"), cmd_raw=[0.2, 0, 0]), now=1.0)
    snap = state.snapshot(now=1.5)
    t, lo = snap["terrain"], snap["locomotion"]
    assert t["class"] == "slippery" and t["slippery"] is True and t["friction_mu"] == 0.15
    assert t["friction_zones"] == ["F-OBS9"] and t["friction_distance"] == 0.4
    assert lo["slip_mode"] is True and lo["slip_source"] == "terrain" and lo["mpc_mu"] == 0.15
    assert lo["friction_prior"] == 0.15 and lo["slip_legs"] == [False] * 4 and lo["slip_events"] == 0
    assert lo["torque_ff"] is None                                  # NaN never reaches the browser
    assert "cmd_raw" not in lo                                      # only the listed fields
    assert "NaN" not in json.dumps(snap)


def test_older_nodes_without_slip_fields(state):
    state.set_terrain({"class": "flat", "recommended_gait": "trot"}, now=1.0)
    state.set_locomotion({"mode": "WALK", "gait": "trot"}, now=1.0)
    state.set_locomotion({"mode": "WALK", "gait": "trot"}, now=2.0)
    snap = state.snapshot(now=2.5)
    assert not any(k.startswith(("slip", "mu_", "mpc_mu", "friction", "torque_ff")) for k in snap["locomotion"])
    assert set(snap["terrain"]) == {"class", "recommended_gait"}
    assert not _slip_events(snap)


def test_slip_mode_on_off_and_detections_in_the_event_log(state):
    state.set_locomotion(dict(SLIP_OFF, mode="WALK"), now=1.0)          # first sample, mode off: no entry
    state.set_locomotion(dict(SLIP_OFF, mode="WALK"), now=1.1)
    assert not _slip_events(state.snapshot(now=1.2))
    # friction prior: slip mode before the plate
    state.set_locomotion(dict(SLIP_OFF, mode="WALK", slip_mode=True, slip_source="terrain", mpc_mu=0.15,
                              friction_prior=0.15), now=2.0)
    state.set_locomotion(dict(SLIP_OFF, mode="WALK", slip_mode=True, slip_source="terrain", mpc_mu=0.15,
                              friction_prior=0.15, torque_ff=1.0), now=2.1)   # unchanged mode: no new entry
    # online detection while the mode is already on
    state.set_locomotion(dict(SLIP_OFF, mode="WALK", slip=True, slip_mode=True, slip_source="both",
                              slip_speed=0.12, slip_legs=[True, True, False, False], slip_events=1, mu_est=0.17,
                              mpc_mu=0.15, friction_prior=0.15), now=3.0)
    state.set_locomotion(dict(SLIP_OFF, mode="WALK", slip_events=1), now=9.0)
    ev = _slip_events(state.snapshot(now=9.5))
    assert [e["level"] for e in ev] == ["warn", "warn", "info"]
    assert ev[0]["text"] == "slip mode ON (terrain prior, prior mu 0.15, MPC mu 0.15)"
    assert ev[1]["text"] == "slip detected #1: feet FL FR, 0.12 m/s, mu est 0.17"
    assert ev[2]["text"] == "slip mode OFF (1 detection so far)"
    assert all(e["source"] == "locomotion" for e in ev)


def test_slip_mode_from_detection_and_node_restart(state):
    # detection without a prior: one entry with the detection details
    state.set_locomotion(dict(SLIP_OFF, mode="WALK"), now=1.0)
    state.set_locomotion(dict(SLIP_OFF, mode="WALK", slip=True, slip_mode=True, slip_source="detected",
                              slip_speed=0.2, slip_legs=[False, True, False, True], slip_events=1, mu_est=0.3,
                              mpc_mu=0.3), now=2.0)
    ev = _slip_events(state.snapshot(now=2.0))
    assert len(ev) == 1
    assert ev[0]["text"] == ("slip mode ON (slip detected, MPC mu 0.30, detection #1: feet FR RR, 0.20 m/s, "
                             "mu est 0.30)")
    # controller restart: the counter starts again at 0 -> no detection entry, mode change logged
    state.set_locomotion(dict(SLIP_OFF, mode="STAND"), now=5.0)
    ev = _slip_events(state.snapshot(now=5.0))
    assert len(ev) == 2 and ev[1]["text"] == "slip mode OFF"
    # dashboard started while the robot is already in slip mode: logged once
    fresh = DashboardState(None)
    fresh.set_locomotion(dict(SLIP_OFF, slip_mode=True, slip_source="terrain", slip_events=4), now=1.0)
    fresh.set_locomotion(dict(SLIP_OFF, slip_mode=True, slip_source="terrain", slip_events=4), now=1.1)
    ev = _slip_events(fresh.snapshot(now=1.2))
    assert [e["text"] for e in ev] == ["slip mode ON (terrain prior, MPC mu 0.50)"]
    # garbage values never raise
    fresh.set_locomotion({"slip_mode": None, "slip_events": "x", "slip_legs": 3, "mu_est": "nan"}, now=2.0)
    fresh.set_locomotion({"slip_mode": "yes", "slip_events": True}, now=2.1)
    json.dumps(fresh.snapshot(now=2.2))


def test_downsample_path_drops_garbage():
    out = downsample_path([(0, 0), (float("nan"), 1), ("a", 2), (1, 1)], 10)
    assert out == [[0.0, 0.0], [1.0, 1.0]]
    assert downsample_path(None) == []


# ----------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------
def test_map_png_is_north_up(layout):
    grid = rasterize_layout(layout)
    png = render_map_png(grid.occ)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == grid.occ.shape
    # a free cell south (low y) of the hall -> bottom rows of the image
    ix, iy = grid.world_to_grid(0.0, -8.0)
    assert grid.occ[iy, ix] == 0
    assert tuple(img[grid.height - 1 - iy, ix]) == (29, 31, 31)


def test_gas_and_rssi_rgba_ramps():
    v = np.array([[-1, 0, 1, 50, 100]], dtype=float)
    g = gas_rgba(v)
    assert g[0, 0, 3] == 0 and g[0, 1, 3] == 0              # unknown / zero transparent
    assert 0 < g[0, 2, 3] < g[0, 3, 3] < g[0, 4, 3]          # opacity grows with value
    assert g[0, 4, :3].sum() > g[0, 2, :3].sum()             # lighter when higher (dark surface)
    r = rssi_rgba(np.array([[0.0, 100.0 * 20 / 70, 100.0]]), threshold_dbm=-80.0)
    assert r[0, 0, 0] > r[0, 0, 2]                           # -100 dBm: red arm
    assert np.allclose(r[0, 1, :3], [0x38, 0x38, 0x35], atol=1)   # -80 dBm: neutral midpoint
    assert r[0, 2, 2] > r[0, 2, 0]                           # -30 dBm: blue arm
    for png in (render_gas_png(v), render_rssi_png(v)):
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        assert img.shape == (1, 5, 4)


def test_jpeg_encoding_downscales():
    big = np.zeros((600, 800, 3), np.uint8)
    jpg = encode_jpeg(big, 70, 400)
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (300, 400, 3)
    assert placeholder_jpeg("no signal")[:2] == b"\xff\xd8"


# ----------------------------------------------------------------------------
# HTTP / SSE server
# ----------------------------------------------------------------------------
def test_validate_command():
    assert validate_command("mission", "Start") == (True, "start")
    assert validate_command("locomotion", "push:30,-5.5") == (True, "push:30,-5.5")
    assert validate_command("battery", "19") == (True, "19")
    assert not validate_command("battery", "150")[0]
    assert not validate_command("mission", "rm -rf /")[0]
    assert not validate_command("shell", "start")[0]


@pytest.fixture()
def server(state, layout):
    grid = rasterize_layout(layout)
    state.set_layer("map", grid.occ, grid.origin_x, grid.origin_y, grid.resolution)
    sent = []

    def blob(name):
        if name == "map.png":
            arr, _ = state.get_layer("map")
            return render_map_png(arr), "image/png"
        if name == "image/gauge.jpg":
            return placeholder_jpeg("no signal"), "image/jpeg"
        return None

    def command(target, cmd):
        sent.append((target, cmd))
        return True, "ok"

    srv = DashboardServer(state.snapshot, lambda: state.static, WEB_DIR, port=0, host="127.0.0.1",
                          sse_rate=20.0, command_fn=command, blob_fn=blob)
    port = srv.start()
    yield f"http://127.0.0.1:{port}", sent
    srv.stop()


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_server_static_files_and_api(server):
    base, _ = server
    code, ctype, body = _get(base + "/")
    assert code == 200 and ctype.startswith("text/html") and "플랜트 디지털 트윈".encode("utf-8") in body
    assert "charset=utf-8" in ctype                                  # Korean page text
    assert _get(base + "/app.js")[1].startswith("application/javascript")
    assert _get(base + "/i18n_ko.js")[1].startswith("application/javascript")
    assert _get(base + "/style.css")[1].startswith("text/css")
    code, ctype, body = _get(base + "/api/state")
    snap = json.loads(body)
    assert "gauges" in snap and "topics" in snap
    st = json.loads(_get(base + "/api/static")[2])
    assert len(st["gauges"]) == 8
    assert json.loads(_get(base + "/api/health")[2])["ok"] is True
    code, ctype, body = _get(base + "/map.png")
    assert ctype == "image/png" and body[:4] == b"\x89PNG"
    assert _get(base + "/image/gauge.jpg")[1] == "image/jpeg"
    for bad in ("/gas.png", "/nope.html", "/..%2f..%2fsetup.py", "/../package.xml"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + bad)
        assert e.value.code == 404


def test_server_serves_symlinked_web_files(state, tmp_path):
    """colcon --symlink-install installs each web file as a symlink into the
    source tree; the server must follow them (regression: 404 'not found')."""
    src = tmp_path / "src_web"
    src.mkdir()
    (src / "index.html").write_text("<html>Plant Digital Twin</html>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    share = tmp_path / "install" / "share" / "plant_simulation" / "web"
    share.mkdir(parents=True)
    try:
        os.symlink(src / "index.html", share / "index.html")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this machine")
    srv = DashboardServer(state.snapshot, lambda: state.static, str(share), port=0, host="127.0.0.1")
    port = srv.start()
    try:
        base = f"http://127.0.0.1:{port}"
        code, ctype, body = _get(base + "/")
        assert code == 200 and b"Plant Digital Twin" in body
        for bad in ("/../../../../secret.txt", "/..%2f..%2f..%2f..%2fsecret.txt"):
            with pytest.raises(urllib.error.HTTPError) as e:
                _get(base + bad)
            assert e.value.code == 404
    finally:
        srv.stop()


def test_server_command_post(server):
    base, sent = server

    def post(obj, raw=None, ctype="application/json"):
        data = raw if raw is not None else json.dumps(obj).encode()
        req = urllib.request.Request(base + "/api/command", data=data, headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    assert post({"target": "mission", "command": "start"}) == (200, {"ok": True, "message": "ok",
                                                                    "target": "mission", "command": "start"})
    assert post({"target": "locomotion", "command": "recover"})[0] == 200
    assert post({"target": "mission", "command": "explode"})[0] == 400
    assert post(None, raw=b"not json")[0] == 400
    assert post(None, raw=b"[1, 2]")[0] == 400
    # cross-site "simple" requests (no CORS preflight) are refused
    assert post({"target": "mission", "command": "abort"}, ctype="text/plain")[0] == 415
    assert post({"target": "mission", "command": "abort"}, ctype="application/x-www-form-urlencoded")[0] == 415
    assert post({"target": "mission", "command": "pause"}, ctype="application/json; charset=utf-8")[0] == 200
    assert sent == [("mission", "start"), ("locomotion", "recover"), ("mission", "pause")]


def test_server_sse_stream(server):
    base, _ = server
    with urllib.request.urlopen(base + "/events", timeout=5) as r:
        assert r.headers.get("Content-Type").startswith("text/event-stream")
        lines = []
        while len([ln for ln in lines if ln.startswith(b"data:")]) < 3:
            lines.append(r.readline().rstrip(b"\r\n"))
    assert lines[0] == b"retry: 2000"
    assert b"event: static" in lines
    datas = [json.loads(ln[5:]) for ln in lines if ln.startswith(b"data:")]
    assert "gauges" in datas[0] and "hotspots" in datas[0]          # static layout first
    assert "t_wall" in datas[1] and "events" in datas[1]            # then snapshots


def test_server_port_fallback(state):
    a = DashboardServer(state.snapshot, lambda: state.static, WEB_DIR, port=0, host="127.0.0.1")
    pa = a.start()
    b = DashboardServer(state.snapshot, lambda: state.static, WEB_DIR, port=pa, host="127.0.0.1", port_fallbacks=3)
    try:
        pb = b.start()
        assert pb != pa
    except OSError:
        pytest.skip("no free fallback port")
    finally:
        b.stop()
        a.stop()


def test_offline_demo_runs_the_real_mission(monkeypatch):
    from plant_simulation import dashboard_demo
    state, mission = dashboard_demo.run_demo(port=0, speed=20, autostart=True, host="127.0.0.1", duration=2.0,
                                             quiet=True)
    snap = state.snapshot()
    assert snap["robot"] is not None and snap["layers"]["map"]["version"] >= 1
    assert mission.base in ("IDLE", "PATROL", "INSPECTING", "SEEK", "RTH", "DOCKED")
    assert snap["plan"]["order"]


@pytest.fixture(scope="module")
def headless_mission(tmp_path_factory):
    """One complete offline full mission (synthetic plant + REAL mission logic + REAL
    MetricsRecorder), shared by the metrics and the Korean event-log tests."""
    from plant_simulation import dashboard_demo
    state, mission = dashboard_demo.run_demo(speed=50, autostart=True, quiet=True, serve=False, realtime=False,
                                             sim_duration=1500.0, metrics_dir=str(tmp_path_factory.mktemp("metrics")),
                                             stop_when_done=True)
    return state, mission, dashboard_demo.run_demo.session_dir


def test_headless_demo_records_a_complete_mission(headless_mission):
    """End-to-end evaluation without ROS: synthetic plant + REAL mission logic
    + REAL MetricsRecorder -> summary.json of a complete full mission."""
    state, mission, session = headless_mission
    assert session and os.path.isfile(os.path.join(session, "summary.json"))
    with open(os.path.join(session, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    lay = Layout.load()
    assert mission.base == "DOCKED" and mission.mode is None
    assert sorted(s["mission"]["completed"]) == sorted(g["id"] for g in lay.gauges)
    assert s["mission"]["final_state"] == "DOCKED" and s["mission"]["duration_s"] > 0
    assert s["gauges"]["mean_error_pct_fs"] < 5.0 and s["gauges"]["within_threshold_rate"] > 0.9
    assert s["gas"]["result"] == "FOUND" and s["gas"]["source_error_m"] < 1.0
    assert s["gas"]["seek_start_t"] is not None and s["gas"]["time_to_find_s"] is not None
    assert s["thermal"]["recall"] == 1.0 and s["distance_m"] > 50.0
    for name in ("events.csv", "gauge_readings.csv", "thermal.csv", "gas.csv", "battery.csv", "pose.csv"):
        with open(os.path.join(session, name), encoding="utf-8") as f:
            assert len(f.readlines()) > 1, name
    # slip mode on / off in the event log: the patrol crosses the corridor plate C-OBS4 (no detour) and
    # passes F-OBS9 (the terrain cost routes around the mu 0.15 plate where it can, the 1 m look-ahead
    # footprint still overlaps it -> prior on)
    texts = [e["text"] for e in state.events if e["source"] == "locomotion" and "slip" in e["text"]]
    assert any(t.startswith("slip mode ON (terrain prior, prior mu 0.20") for t in texts)      # C-OBS4
    assert any(t.startswith("slip mode ON (terrain prior, prior mu 0.15") for t in texts)      # F-OBS9
    assert sum(t.startswith("slip mode ON") for t in texts) == sum(t.startswith("slip mode OFF") for t in texts)
    # a synthetic slip detection needs the robot centre on the mu 0.15 plate (pose.csv at 2 Hz: within 0.25 m);
    # detection itself is covered by test_demo_synthetic_slip_crossing_a_plate
    x0, y0, x1, y1 = next(o for o in lay.obstacles if o["id"] == "F-OBS9")["rect"]
    with open(os.path.join(session, "pose.csv"), encoding="utf-8") as f:
        hdr = f.readline().strip().split(",")
        xy = [(float(r[hdr.index("x")]), float(r[hdr.index("y")])) for r in (ln.strip().split(",") for ln in f) if r[0]]
    near = any(x0 - 0.25 <= x <= x1 + 0.25 and y0 - 0.25 <= y <= y1 + 0.25 for x, y in xy)
    assert near or not any(t.startswith("slip detected") for t in texts)


def test_demo_synthetic_slip_crossing_a_plate():
    """Display-only friction prior + slip detection of the offline demo."""
    from plant_simulation.dashboard_demo import SyntheticSlip
    zones = [{"id": "P", "kind": "plate", "rect": [1.0, -1.0, 3.0, 1.0], "mu": 0.15, "slippery": True},
             {"id": "W", "kind": "zone", "rect": [10.0, -1.0, 11.0, 1.0], "mu": 0.5, "slippery": False}]
    sim = SyntheticSlip(zones)
    assert [z["id"] for z in sim.zones] == ["P"]                    # mu 0.5 is not slippery
    dt, out = 0.1, []
    for k in range(200):                                            # 0.4 m/s along +x from x = -1 m
        t = k * dt
        out.append((t, -1.0 + 0.4 * t) + sim.step(-1.0 + 0.4 * t, 0.0, 0.0, t, dt))
    first = next(o for o in out if o[2]["slippery"])
    assert first[1] == pytest.approx(0.0, abs=0.05)                 # 1 m look-ahead reaches the plate at x = 0
    assert first[2]["class"] == "slippery" and first[2]["friction_zones"] == ["P"]
    assert first[2]["friction_distance"] == pytest.approx(1.0, abs=0.05) and first[2]["friction_mu"] == 0.15
    assert first[3]["slip_mode"] and first[3]["slip_source"] == "terrain" and first[3]["mpc_mu"] == 0.15
    det = next(o for o in out if o[3]["slip"])
    assert det[1] == pytest.approx(1.0, abs=0.05)                   # robot centre enters the plate
    assert det[3]["slip_source"] == "both" and det[3]["slip_events"] == 1 and det[3]["mu_est"] == 0.17
    assert det[3]["slip_legs"] == [True, True, False, False]
    # rear footprint (0.35 m) leaves x = 3 m at x = 3.35 m, the prior is held 1 s
    prior_off = next(o for o in out if o[1] > 3.0 and not o[2]["slippery"])
    assert prior_off[1] == pytest.approx(3.35 + 0.4, abs=0.1)
    assert prior_off[2]["class"] == "flat" and prior_off[2]["friction_mu"] is None
    assert prior_off[3]["slip_mode"] and prior_off[3]["slip_source"] == "detected"   # 3 m release distance
    off = next(o for o in out if o[1] > 3.0 and not o[3]["slip_mode"])
    assert 4.0 < off[1] < 4.3                                       # last vote at x ~ 1.15 m + 3 m of travel
    assert off[3]["slip_source"] == "none" and not off[3]["slip"]
    assert all(o[3]["slip_events"] == 1 for o in out if o[0] >= det[0])
    assert out[-1][3]["torque_ff"] == 0.0 and out[-1][3]["mu_est"] > 0.3                # recovering to 0.5
    assert max(o[3]["torque_ff"] for o in out) == 1.0
