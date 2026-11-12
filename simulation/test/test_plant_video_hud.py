"""Video-ready demo: HUD mission steps / banners / robot-down panel / result panel
(demo_hud.HudStory + compose_hud, ROS free), the demo_hud node wiring against the
strict rclpy stand-ins, and the follow camera's fall guard (levelled offset, hold /
release, a port of the gz-rendering follow law on a synthetic fall)."""
import base64
import math
import os
import sys

import cv2
import numpy as np
import pytest

from dusan_common.layout import Layout
from plant_simulation import demo_hud as dh
from plant_simulation import follow_camera as fc

import test_plant_nodes_stubbed as stub  # noqa: E402

ros = stub.ros                                             # fixture

LAY = Layout.load()
GIDS = [g["id"] for g in LAY.gauges]


@pytest.fixture(scope="module")
def en():
    return dh.TextRenderer(language="en")


@pytest.fixture(scope="module")
def ko():
    r = dh.TextRenderer(language="ko")
    if not r.korean:
        pytest.skip(f"no Pillow / Korean font here: {r.reason}")
    return r


def cam(h=480, w=640):
    img = np.full((h, w, 3), 110, np.uint8)
    img[h // 3:2 * h // 3, w // 3:2 * w // 3] = (30, 160, 220)
    return img


def jpeg_b64(color=(20, 200, 20), size=(320, 400)):
    img = np.full((size[1], size[0], 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def mission(state="PATROL", target="G04", done=(), missed=(), results=None, seek=None, rth="", total=8,
            elapsed=100.0, **kw):
    d = {"state": state, "target": target, "completed": list(done), "missed": list(missed),
         "progress": [len(done) + len(missed), total], "results": results or {}, "seek": seek, "rth_reason": rth,
         "gas_alarm": state == "SEEK", "mission_elapsed": elapsed}
    d.update(kw)
    return d


def loco(rs="NORMAL", mode="WALK", falls=0, ok=0, assisted=0, attempts=0, roll=0.0, pitch=0.0):
    return {"mode": mode, "locomotion_mode": "stable", "gait": "trot", "recovery_state": rs, "fall_count": falls,
            "recovery_success": ok, "assisted_resets": assisted, "recovery_attempts": attempts, "roll": roll,
            "pitch": pitch}


# ============================================================================
# texts
# ============================================================================
def test_mission_step_texts():
    t = dh.mission_step_text
    assert t("PATROL", "G04", [5, 8]) == "G04 계기판으로 이동 중 (6/8)"
    assert t("INSPECTING", "G04", [5, 8]) == "G04 계기판 판독 중 (6/8)"
    assert t("PATROL", "G04", [5, 8], korean=False) == "GOING TO GAUGE G04 (6/8)"
    assert t("PATROL", "", [8, 8]) == "순찰 이동 중"                         # no step once all are handled
    assert t("SEEK", "", [5, 8]) == "가스 누출원 추적 중"
    assert t("RTH", "", [8, 8], "mission_complete") == "점검 완료 · 충전소로 복귀 중"
    assert t("RTH", "", [3, 8], "battery_low") == "충전소로 복귀 중 · 사유: 배터리 부족"
    assert t("RTH", "", [3, 8], "comm_shadow_predicted", korean=False) == "RETURNING TO DOCK (comm shadow ahead)"
    assert t("RTH", "", None, "weird_reason") == "충전소로 복귀 중 · 사유: weird_reason"
    assert t("DOCKED", "", [8, 8], finished=True) == "임무 완료 · 충전소 도착"
    assert t("DOCKED", "", [3, 8]) == "충전소 도착 · 충전 중"
    assert "'t'" in t("IDLE") and t("") == "임무: --" and t("", korean=False) == "MISSION: --"
    assert t("PAUSED", "G02", [1, 8]) == "일시정지 · G02 점검 대기"
    assert t("WEIRD") == "임무: WEIRD"
    for st in ("IDLE", "PATROL", "INSPECTING", "SEEK", "RTH", "DOCKED", "MANUAL", "PAUSED", "DONE"):
        assert t(st, "G01", [1, 8]) and t(st, "G01", [1, 8], korean=False).isascii()
    assert t("PATROL", "G01", ["x", 8]) == "G01 계기판으로 이동 중"                 # garbage progress ignored


def test_value_duration_and_path_formats():
    assert dh.format_value(0.92157, "MPa") == "0.92 MPa" and dh.format_value(85.27, "°C") == "85.3 °C"
    assert dh.format_value(None, "bar") == "--" and dh.format_value(float("nan")) == "--"
    assert dh.format_duration(793.4) == "13분 13초" and dh.format_duration(42, False) == "42 s"
    assert dh.format_duration(None) == "--"
    home = os.path.expanduser("~")
    assert dh.display_dir(os.path.join(home, "dusan_metrics", "s1")) == "~/dusan_metrics/s1/"
    assert dh.display_dir("/data/x/") == "/data/x/" and dh.display_dir("") == ""


def test_recovery_texts_follow_the_fall_recovery_states():
    from module1_locomotion import fall_recovery as fr
    states = set(fr.FALLEN_STATES) | set(fr.RECOVERING_STATES)
    assert states <= set(dh.RECOVERY_LABELS) and states <= set(dh.RECOVERY_STEP_OF)
    title, detail, step, level = dh.recovery_text({"state": "DAMP", "mode": "FALLEN"})
    assert (title, step, level) == ("넘어짐 감지", 0, "alarm") and "충격 완화" in detail
    title, detail, step, level = dh.recovery_text({"state": "SELF_RIGHT", "mode": "RECOVERING", "attempt": 2})
    assert (title, step, level) == ("몸을 뒤집는 중", 1, "warning") and "2번째 시도" in detail
    assert dh.recovery_text({"state": "STAND_UP"})[:1] == ("일어서는 중",)
    assert dh.recovery_text({"state": "FAILED"})[0] == "자가 복구 실패"
    assert dh.recovery_text({"state": "", "mode": "FALLEN"})[2] == 0          # older controller: mode only
    assert dh.recovery_text({"state": "TUCK"}, korean=False)[0].isascii()


def test_banner_and_final_texts():
    main, sub, lv = dh.banner_text({"kind": "gauge", "id": "G04", "value": 0.9216, "unit": "MPa",
                                    "status": "warning", "n": 6})
    assert main == "G04 판독 완료: 0.92 MPa · 경고" and sub == "판독 6회 종합" and lv == "warning"
    assert dh.banner_text({"kind": "missed", "id": "G06"})[0] == "G06 판독 실패 · 다음 계기로 이동"
    main, sub, lv = dh.banner_text({"kind": "gas_found", "error_m": 0.35, "estimate": [21.31, -3.4]})
    assert main == "가스 누출원 발견" and sub == "위치 오차 0.35 m · 추정 위치 (21.3, -3.4)" and lv == "alarm"
    assert dh.banner_text({"kind": "gas_end", "result": "TIMEOUT"})[0] == "누출원 추적 종료: 시간 초과"
    assert dh.banner_text({"kind": "rth", "reason": "comm_lost"})[0] == "충전소 복귀 시작 · 사유: 통신 두절"
    main, sub, lv = dh.banner_text({"kind": "recovered", "falls": 2, "recovered": 2, "assisted": 0})
    assert main.startswith("복구 완료") and sub == "넘어짐 2회 · 자가 복구 2회" and lv == "normal"
    assert dh.banner_text({"kind": "recovered", "by_reset": True})[2] == "warning"
    for b in ({"kind": "gauge", "id": "G1", "value": 1.0}, {"kind": "missed", "id": "G1"}, {"kind": "gas_found"},
              {"kind": "rth", "reason": "manual"}, {"kind": "recovered"}):
        assert dh.banner_text(b, korean=False)[0].isascii()
    assert dh.banner_text({"kind": "nope"})[0] == ""
    title, sub, rows = dh.final_lines({"state": "DOCKED", "read": 7, "total": 8, "missed": 1, "warning": 2,
                                       "alarm": 0, "mean_error_pct_fs": 0.55, "max_error_pct_fs": 1.2,
                                       "seek": "FOUND", "seek_error_m": 0.35, "falls": 2, "recovered": 2,
                                       "assisted": 1, "elapsed": 793.4, "saved": "~/dusan_metrics/s1/"})
    assert title == "임무 완료" and sub == "충전소 도착 · 소요 13분 13초"
    rows = dict(rows)
    assert rows["계기 판독"] == "7 / 8 (실패 1) · 경고 2 · 경보 0"
    assert rows["판독 오차"] == "평균 0.55 % FS · 최대 1.20 % FS"
    assert rows["가스 누출원"] == "발견 · 위치 오차 0.35 m"
    assert rows["넘어짐 / 자가 복구"] == "2회 / 2회 · 보조 리셋 1회"
    assert rows["결과 저장"] == "~/dusan_metrics/s1/"
    bare = dict(dh.final_lines({"read": 8, "total": 8})[2])
    assert bare["가스 누출원"] == "가스 경보 없음" and "판독 오차" not in bare and "결과 저장" not in bare


# ============================================================================
# HudStory
# ============================================================================
def test_story_baseline_is_not_announced():
    s = dh.HudStory()
    res = {"G01": {"value": 1.0, "unit": "bar", "status": "normal", "n_readings": 4}}
    s.on_mission(mission(done=["G01"], results=res, seek={"result": "FOUND", "t": 5.0}), 0.0)
    s.on_locomotion(loco(), 0.0)
    assert s.status(0.1)["banner"] is None                       # a restarted HUD does not re-announce


def test_story_gauge_read_banner_with_photo_and_expiry():
    s = dh.HudStory(gauge_hold=7.0)
    s.on_mission(mission(done=["G01"]), 0.0)
    res = {"G04": {"value": 0.9216, "unit": "MPa", "status": "warning", "n_readings": 6}}
    s.on_mission(mission(target="G07", done=["G01", "G04"], results=res), 10.0)
    b = s.status(10.5)["banner"]
    assert b["kind"] == "gauge" and b["id"] == "G04" and b["value"] == pytest.approx(0.9216)
    assert b["status"] == "warning" and b["n"] == 6 and b["photo"] is None
    assert s.on_snapshot({"id": "G04", "jpeg_b64": jpeg_b64()}) is True        # the photo arrives a bit later
    b = s.status(11.0)["banner"]
    assert b["photo"].shape[2] == 3 and max(b["photo"].shape[:2]) <= 400
    assert s.status(17.5)["banner"] is None                                   # held 7 s
    assert s.on_snapshot({"id": "G05", "jpeg_b64": "not base64!"}) is False
    assert s.on_snapshot({"jpeg_b64": jpeg_b64()}) is False and s.on_snapshot(None) is False
    s.on_mission(mission(target="G08", done=["G01", "G04"], missed=["G07"], results=res), 20.0)
    assert s.status(20.1)["banner"] == {"kind": "missed", "id": "G07"}


def test_story_gas_rth_and_priorities():
    s = dh.HudStory()
    s.on_mission(mission(), 0.0)
    s.on_mission(mission(state="SEEK", target=""), 1.0)
    s.on_mission(mission(state="PATROL", seek={"result": "FOUND", "error_m": 0.35, "estimate": [1, 2], "t": 9.0}),
                 2.0)
    b = s.status(2.1)["banner"]
    assert b["kind"] == "gas_found" and b["error_m"] == pytest.approx(0.35)
    s.on_mission(mission(state="RTH", target="", rth="battery_low",
                         seek={"result": "FOUND", "error_m": 0.35, "t": 9.0}), 3.0)
    assert s.status(3.1)["banner"]["kind"] == "gas_found"                     # higher priority than rth
    assert s.status(8.5)["banner"] == {"kind": "rth", "reason": "battery_low"}
    assert s.status(9.5)["banner"] is None
    # the planned return after the last gauge is not an event
    s2 = dh.HudStory()
    s2.on_mission(mission(), 0.0)
    s2.on_mission(mission(state="RTH", target="", rth="mission_complete"), 1.0)
    assert s2.status(1.1)["banner"] is None
    s2.on_mission(mission(state="SEEK", target="", seek=None), 2.0)
    s2.on_mission(mission(state="PATROL", seek={"result": "TIMEOUT", "t": 4.0}), 3.0)
    assert s2.status(3.1)["banner"] == {"kind": "gas_end", "result": "TIMEOUT"}


def test_story_robot_down_panel_and_recovered_banner():
    s = dh.HudStory(recovered_hold=5.0)
    s.on_locomotion(loco(), 0.0)
    s.on_mission(mission(), 0.0)
    assert s.status(0.0)["recovery"] is None
    s.on_locomotion(loco("FALL_DETECTED", "FALLEN", falls=1, attempts=0), 1.0)
    assert s.status(1.0)["recovery"] == {"state": "FALL_DETECTED", "mode": "FALLEN", "attempt": None}
    s.on_locomotion(loco("TUCK", "RECOVERING", falls=1, attempts=1), 2.0)
    s.on_locomotion(loco("SELF_RIGHT", "RECOVERING", falls=1, attempts=1), 3.0)
    assert s.status(3.0)["recovery"] == {"state": "SELF_RIGHT", "mode": "RECOVERING", "attempt": 1}
    s.on_locomotion(loco("STAND_UP", "RECOVERING", falls=1, attempts=2), 4.0)
    assert s.status(4.0)["recovery"]["attempt"] == 2
    assert s.status(4.0, loco=False)["recovery"] is None                      # stale locomotion: nothing
    s.on_locomotion(loco("NORMAL", "STAND", falls=1, ok=1, attempts=2), 5.0)
    st = s.status(5.1)
    assert st["recovery"] is None
    assert st["banner"] == {"kind": "recovered", "by_reset": False, "falls": 1, "recovered": 1, "assisted": 0}
    assert s.status(10.5)["banner"] is None
    # only the assisted-reset counter went up: said so
    s.on_locomotion(loco("FAILED", "FALLEN", falls=2, ok=1, attempts=5), 20.0)
    assert s.status(20.0)["recovery"]["state"] == "FAILED"
    s.on_locomotion(loco("ASSISTED_RESET", "RECOVERING", falls=2, ok=1, attempts=5), 21.0)
    s.on_locomotion(loco("NORMAL", "STAND", falls=2, ok=1, assisted=1, attempts=5), 22.0)
    assert s.status(22.1)["banner"]["by_reset"] is True


def test_story_final_panel_frozen_at_the_dock():
    s = dh.HudStory(layout=LAY)
    s.on_mission(mission(), 0.0)
    s.on_locomotion(loco(falls=2, ok=2, assisted=0), 0.0)
    res = {}
    for i, gid in enumerate(GIDS):
        g = LAY.gauge(gid)
        span = g["max"] - g["min"]
        v = g["value"] + (0.01 * span if i == 0 else 0.0)                   # 1 % FS error on the first one
        res[gid] = {"value": v, "unit": g["unit"], "status": LAY.gauge_status(gid, v), "n_readings": 5}
    done = GIDS[:-1]
    seek = {"result": "FOUND", "error_m": 0.35, "t": 50.0}
    s.on_mission(mission(state="RTH", target="", done=done, missed=GIDS[-1:], results=res, seek=seek,
                         rth="mission_complete", total=len(GIDS)), 1.0)
    assert s.status(1.1)["final"] is None and not s.status(1.1)["mission_finished"]
    s.on_mission(mission(state="DOCKED", target="", done=done, missed=GIDS[-1:], results=res, seek=seek,
                         rth="mission_complete", total=len(GIDS), elapsed=793.4), 2.0)
    st = s.status(2.1)
    f = st["final"]
    assert st["mission_finished"] and f["read"] == len(GIDS) - 1 and f["missed"] == 1 and f["total"] == len(GIDS)
    assert f["mean_error_pct_fs"] == pytest.approx(1.0 / len(done), abs=1e-6)
    assert f["max_error_pct_fs"] == pytest.approx(1.0, abs=1e-6)
    want_warn = sum(LAY.gauge_status(g, res[g]["value"]) == "warning" for g in done)
    assert f["warning"] == want_warn and f["seek"] == "FOUND" and f["seek_error_m"] == pytest.approx(0.35)
    assert (f["falls"], f["recovered"], f["elapsed"]) == (2, 2, pytest.approx(793.4)) and "saved" not in f
    # the save location of /metrics/session is filled in live
    s.on_session({"session_dir": os.path.join(os.path.expanduser("~"), "dusan_metrics", "20261111_1200"),
                  "report": "inspection_report.html", "report_final": True})
    assert s.status(3.0)["final"]["saved"] == "~/dusan_metrics/20261111_1200/"
    # later falls / battery do not change the frozen numbers; a new mission clears the panel
    s.on_locomotion(loco(falls=5, ok=5), 4.0)
    assert s.status(4.1)["final"]["falls"] == 2
    assert s.status(4.1, mission=False)["final"] is None                         # stale mission
    s.on_mission(mission(state="PATROL", target="G01", done=[], total=len(GIDS)), 5.0)
    assert s.status(5.1)["final"] is None


def test_story_tolerates_garbage():
    s = dh.HudStory()
    for d in (None, [], "x", {"completed": None, "missed": None, "results": [], "seek": 3, "progress": "a"},
              {"state": 5, "completed": [None, 1], "results": {"1": "bad"}}):
        s.on_mission(d, 0.0)
        s.on_locomotion(d, 0.0)
        s.on_session(d)
        s.on_snapshot(d)
    s.on_mission({"completed": [1, 2, 3], "results": {"2": {"value": "nan?", "n_readings": "x"},
                                                      "3": {"value": 1.0, "n_readings": float("nan")}}}, 1.0)
    st = s.status(1.1)
    assert st["banner"]["id"] == "3" and st["banner"]["value"] == 1.0 and st["banner"]["n"] is None
    two = [b[2] for b in s._banners if b[2].get("id") == "2"][0]
    assert two["value"] is None and two["n"] is None


# ============================================================================
# compositor overlays
# ============================================================================
def _diff(a, b):
    return (np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 30)


def test_compose_overlays_draw_and_keep_the_size(en):
    base = dh.compose_hud(cam(), {}, renderer=en)
    rec = dh.compose_hud(cam(), {"recovery": {"state": "SELF_RIGHT", "mode": "RECOVERING", "attempt": 1}},
                         renderer=en)
    assert rec.shape == base.shape and _diff(rec, base)[200:300, 100:540].mean() > 0.3   # centre panel
    photo = np.full((400, 320, 3), (20, 200, 20), np.uint8)
    ban = dh.compose_hud(cam(), {"banner": {"kind": "gauge", "id": "G04", "value": 0.92, "unit": "MPa",
                                            "status": "normal", "photo": photo}}, renderer=en)
    d = _diff(ban, base)
    assert d[330:400, 200:440].mean() > 0.2                                      # banner above the bottom bar
    assert (np.abs(ban[150, 600].astype(int) - (20, 200, 20)).sum() < 40)          # photo in the top right corner
    fin = dh.compose_hud(cam(), {"final": {"read": 8, "total": 8, "warning": 2, "alarm": 0, "seek": "FOUND",
                                           "seek_error_m": 0.3, "falls": 1, "recovered": 1, "elapsed": 600,
                                           "saved": "~/dusan_metrics/x/"},
                                 "mission_finished": True, "mission_state": "DOCKED"}, renderer=en)
    assert _diff(fin, base)[100:380, 40:600].mean() > 0.2
    # no photo next to the result panel; the robot-down panel hides a gauge banner
    both = dh.compose_hud(cam(), {"recovery": {"state": "TUCK"}, "banner": {"kind": "gauge", "id": "G1",
                                                                           "value": 1.0, "photo": photo}},
                          renderer=en)
    assert np.array_equal(both[150, 600], base[150, 600])


@pytest.mark.parametrize("size", [(320, 240), (640, 480), (1280, 720)])
def test_compose_overlays_any_size(size, en):
    w, h = size
    st = {"recovery": {"state": "STAND_UP", "attempt": 3}, "mission_state": "RTH", "rth_reason": "battery_low",
          "mission_progress": [3, 8], "banner": {"kind": "recovered", "falls": 1, "recovered": 1}}
    assert dh.compose_hud(cam(), st, {"width": w, "height": h}, en, t=0.2).shape == (h, w, 3)
    st = {"final": {"read": 8, "total": 8, "saved": "~/" + "x" * 120}, "mission_finished": True,
          "banner": {"kind": "gas_found", "error_m": 0.2}, "gas_ppm": 600.0, "gas_alarm": True}
    assert dh.compose_hud(None, st, {"width": w, "height": h}, en, t=0.2).shape == (h, w, 3)


def test_compose_korean_overlays(ko):
    st = {"recovery": {"state": "SELF_RIGHT", "mode": "RECOVERING", "attempt": 1}, "mission_state": "PATROL",
          "mission_target": "G04", "mission_progress": [5, 8]}
    out = dh.compose_hud(cam(), st, renderer=ko)
    # the recovery title is drawn in the warning colour
    panel = out[150:330, 30:610].astype(int)
    assert (np.abs(panel - np.array(dh.COLORS["warning"])).sum(axis=2) < 60).sum() > 300


def test_long_bottom_line_is_shrunk_not_cut(en):
    r = en
    txt = dh.mission_step_text("RTH", "", [3, 8], "comm_shadow_predicted", korean=False)
    size = dh._fit_size(r, txt, 24, 300, 10)
    assert size < 24 and r.measure(txt, size)[0] <= 300
    assert dh._fit_size(r, "x" * 500, 24, 100, 12) == 12                       # floor


# ============================================================================
# demo_hud node wiring
# ============================================================================
@pytest.fixture()
def hud_node(ros):
    sys.modules.pop("plant_simulation.demo_hud_node", None)
    import importlib
    mod = importlib.import_module("plant_simulation.demo_hud_node")
    node = mod.DemoHudNode()
    yield ros, node
    sys.modules.pop("plant_simulation.demo_hud_node", None)


def _s(d):
    import json
    return stub.String(data=json.dumps(d))


def test_demo_hud_node_story_inputs(hud_node):
    W, node = hud_node
    topics = {s.topic for s in W.subs}
    assert {"/metrics/session", "/inspection/gauge_snapshot", "/locomotion/state", "/mission/state"} <= topics
    W.deliver("/mission/state", _s(mission(done=["G01"])))
    W.deliver("/locomotion/state", _s(loco()))
    res = {"G04": {"value": 0.92, "unit": "MPa", "status": "normal", "n_readings": 6}}
    W.deliver("/mission/state", _s(mission(target="G07", done=["G01", "G04"], results=res)))
    W.deliver("/inspection/gauge_snapshot", _s({"id": "G04", "jpeg_b64": jpeg_b64()}))
    st = node.status()
    assert st["banner"]["kind"] == "gauge" and st["banner"]["photo"] is not None
    W.deliver("/locomotion/state", _s(loco("TUCK", "RECOVERING", falls=1, attempts=1)))
    st = node.status()
    assert st["recovery"]["state"] == "TUCK" and st["loco_mode"] == "RECOVERING"
    W.tick(1, dt=0.2)
    assert len(W.published("/demo/hud")) == 1
    W.deliver("/inspection/gauge_snapshot", _s({"id": "G05", "jpeg_b64": "@@@"}))
    assert any("without a decodable photo" in line for _, line in node.get_logger().lines)
    W.deliver("/metrics/session", _s({"session_dir": "/tmp/dusan_metrics/s1", "report": "inspection_report.html"}))
    assert node.story.session["dir"] == "/tmp/dusan_metrics/s1"


def test_demo_hud_node_final_panel_and_stale_mission(hud_node, monkeypatch):
    W, node = hud_node
    W.deliver("/mission/state", _s(mission()))
    seek = {"result": "FOUND", "error_m": 0.4, "t": 3.0}
    W.deliver("/mission/state", _s(mission(state="DOCKED", target="", done=GIDS, total=len(GIDS), seek=seek,
                                           results={g: {"value": LAY.gauge(g)["value"]} for g in GIDS})))
    st = node.status()
    assert st["mission_finished"] and st["final"]["read"] == len(GIDS)
    assert st["final"]["mean_error_pct_fs"] == pytest.approx(0.0, abs=1e-9)        # the layout was loaded
    import time as _time
    later = _time.monotonic() + node.p["status_timeout"] + 1.0
    st = node.status(now=later)
    assert st["final"] is None and not st["mission_finished"]


# ============================================================================
# follow camera fall guard
# ============================================================================
def rot(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


@pytest.mark.parametrize("r,p,y", [(0, 0, 0), (math.pi, 0, 0.3), (1.5, 0.2, -2.0), (-2.4, -0.9, 1.0),
                                   (0.1, 1.2, 3.0)])
def test_level_offset_leaves_only_the_heading(r, p, y):
    off = (-2.5, 0.0, 1.6)
    world = rot(r, p, y) @ np.array(fc.level_offset(off, r, p))
    assert world == pytest.approx(rot(0, 0, y) @ np.array(off), abs=1e-9)


def test_fall_guard_hold_release():
    g = fc.FallGuard(tilt_hold_deg=30, tilt_release_deg=20, release_hold=1.0)
    d = math.radians
    assert g.update(0.0, d(5), d(-8)) is None and g.update(0.1, d(15), 0.0) is None    # walking / ramp sway
    assert g.update(0.2, d(40), 0.0) == "hold" and g.holding and g.holds == 1
    assert g.update(0.5, d(178), 0.0) is None                                     # on its back: still held
    assert g.update(3.0, d(10), 0.0, down=True) is None and g.holding             # upright, still recovering
    assert g.update(3.5, d(10), 0.0) is None and g.update(3.9, d(25), 0.0) is None   # 25 deg resets the timer
    assert g.update(4.2, d(5), 0.0) is None
    assert g.update(5.21, d(4), 0.0) == "release" and not g.holding
    assert g.update(6.0, 0.0, 0.0, down=True) == "hold"                            # reported fall alone
    assert g.update(7.0, "x", None) is None and g.update(7.0, float("nan"), 0.0) is None


def test_director_fall_hold_messages_levelled_and_refreshed():
    sent = []
    dr = fc.CameraDirector("spot", (-2.5, 0.0, 1.6), lambda req: sent.append(req) or "cli", log=lambda s: None,
                           t0=-10.0)
    d = math.radians
    dr.on_loco(0.0, loco(roll=d(5)))
    assert sent == []
    dr.on_loco(0.2, loco(roll=d(40)))
    hold = sent[-1]
    assert hold.mode == "USE_LAST" and hold.pos == fc.HOLD_GAIN and hold.look == fc.LOOK_GAIN   # position frozen
    assert hold.offset == pytest.approx(fc.level_offset((-2.5, 0.0, 1.6), d(40), 0.0))
    dr.on_loco(0.5, loco("TUCK", "RECOVERING", roll=d(178)))
    assert len(sent) == 1                                                         # refresh at most every 1 s
    dr.on_loco(1.3, loco("TUCK", "RECOVERING", roll=d(178)))
    assert len(sent) == 2 and sent[-1].offset[2] == pytest.approx(-1.6, abs=0.1)  # GUI flips it back up
    dr.on_loco(2.5, loco("TUCK", "RECOVERING", roll=d(179)))
    assert len(sent) == 2                                                         # no change worth a send
    for k in range(12):
        dr.on_loco(3.0 + 0.1 * k, loco(roll=d(3)))
    assert sent[-1].pos == fc.POS_GAIN and sent[-1].offset == (-2.5, 0.0, 1.6) and dr.state == "FOLLOW"


def follow_sim(trace, guard, fps=60, latency=0.4):
    """gz-rendering BaseCamera follow (local frame) per rendered frame on a pose trace
    [(t, pos, roll, pitch, yaw, down)], driven by the CameraDirector; /gui/track messages arrive after
    `latency` s and set the offset and the POSITION gain (= the message's track_pgain)."""
    D = (-2.5, 0.0, 1.6)
    off, gain = np.array(D), fc.POS_GAIN
    t_end = trace[-1][0]
    cam_pos = trace[0][1] + rot(0, 0, trace[0][4]) @ np.array(D)
    pend, k, minz = [], 0, 1e9
    now = {"t": 0.0}
    dr = fc.CameraDirector("spot", D, lambda req: pend.append((now["t"] + latency, req)) or "cli",
                           log=lambda s: None, cfg={"fall_guard": guard}, t0=-10.0)
    next_state = 0.0
    for i in range(int(t_end * fps)):
        t = now["t"] = i / fps
        while k + 1 < len(trace) and trace[k + 1][0] <= t:
            k += 1
        _, pos, r, p, y, down = trace[k]
        if t >= next_state:
            next_state += 0.05
            dr.on_loco(t, {"roll": r, "pitch": p, "mode": "RECOVERING" if down else "WALK",
                           "recovery_state": "TUCK" if down else "NORMAL"})
        while pend and pend[0][0] <= t:
            _, req = pend.pop(0)
            off, gain = np.array(req.offset), req.pos
        target = pos + rot(r, p, y) @ off
        cam_pos = cam_pos + (target - cam_pos) * gain
        minz = min(minz, cam_pos[2])
    return minz, cam_pos


def test_fall_guard_keeps_the_camera_above_the_floor_on_a_back_flip():
    # walk 3 s, roll over onto the back within 0.8 s, lie 6 s, roll back and stand up, walk 4 s
    trace = []
    for i in range(0, 1800):
        t = i * 0.01
        if t < 3.0:
            r = 0.0
        elif t < 3.8:
            r = math.pi * (t - 3.0) / 0.8
        elif t < 9.8:
            r = math.pi
        elif t < 11.0:
            r = math.pi * (1 - (t - 9.8) / 1.2)
        else:
            r = 0.0
        down = 3.2 < t < 12.5
        z = 0.32 if (t < 3.0 or t > 12.5) else 0.12
        x = 0.3 * min(t, 3.0) + (0.3 * (t - 12.5) if t > 12.5 else 0.0)
        trace.append((t, np.array([x, 0.0, z]), r, 0.0, 0.2, down))
    plain, _ = follow_sim(trace, guard=False)
    guarded, end = follow_sim(trace, guard=True)
    assert plain < -0.5                                       # what the audit saw: under the floor
    assert guarded > 0.9                                      # held where it was (position gain ~0)
    behind = trace[-1][1] + rot(0, 0, 0.2) @ np.array([-2.5, 0.0, 1.6])
    assert np.linalg.norm(end - behind) < 0.6                 # following again afterwards


class FakeInputs:
    """inputs_factory() stand-in: one list of (kind, data) per poll; the clock advances 0.1 s per poll."""

    def __init__(self, script, clock):
        self.script, self.clock, self.closed = list(script), clock, False

    def ok(self):
        return bool(self.script)

    def poll(self, timeout):
        self.clock["t"] += 0.1
        item = self.script.pop(0)
        return [(k, self.clock["t"], d) for k, d in item] if item else []

    def close(self):
        self.closed = True


def test_follow_camera_guard_loop_sends_hold_and_follow():
    import test_plant_demo as tp
    g = tp.FakeGz(ready_after=1)
    clock = {"t": 0.0}
    script = ([[("loco", loco())]] * 3 + [[("loco", loco(roll=1.2))]]
              + [[("loco", loco("TUCK", "RECOVERING", roll=3.0))]] * 12 + [[]]
              + [[("loco", loco("NORMAL", "STAND", roll=0.05))]] * 15)
    inp = FakeInputs(script, clock)
    lines = []
    f = fc.FollowCamera("gz", "spot", "plant", (-2.5, 0.0, 1.6), 0.02, runner=g.run, clock=lambda: clock["t"],
                        sleep=lambda dt: None, log=lines.append, inputs_factory=lambda: inp,
                        cam_cfg={"tilt_hold_deg": 30.0, "bogus": 1}, threaded=False)
    assert f.main() == 0 and inp.closed
    sent = [c[7] for c in g.calls if c[1:3] == ["topic", "-t"]]
    assert len(sent) >= 4                                   # engage, hold, >= 1 update, follow
    assert sent[0] == fc.follow_request("spot", (-2.5, 0.0, 1.6)).text()
    assert sent[1].startswith("track_mode: USE_LAST") and "track_pgain: 2e-05" in sent[1]
    assert "follow_pgain: 0.05" in sent[1]                  # still looking at the robot
    assert sent[-1] == fc.track_message("spot", (-2.5, 0.0, 1.6), mode="USE_LAST")
    assert any("camera held" in x for x in lines) and any("upright again" in x for x in lines)
    assert any("camera control on" in x and "fall guard on" in x for x in lines)


def test_follow_camera_guard_without_ros_or_with_errors():
    import test_plant_demo as tp
    g = tp.FakeGz(ready_after=1)
    lines = []

    def no_ros():
        raise ImportError("No module named 'rclpy'")
    f = fc.FollowCamera("gz", "spot", "plant", (-2.5, 0.0, 1.6), 0.02, runner=g.run, clock=lambda: 0.0,
                        sleep=lambda dt: None, log=lines.append, inputs_factory=no_ros)
    assert f.main() == 0 and "camera control off" in lines[-1]

    class Broken(FakeInputs):
        def poll(self, timeout):
            raise RuntimeError("context invalid")
    f2 = fc.FollowCamera("gz", "spot", "plant", (-2.5, 0.0, 1.6), 0.02, runner=g.run, clock=lambda: 0.0,
                         sleep=lambda dt: None, log=lines.append, inputs_factory=lambda: Broken([1], {"t": 0.0}),
                         threaded=False)
    assert f2.main() == 0 and any("camera control stopped" in x for x in lines)
    # no camera control when the follow mode could not be confirmed (nothing to protect)
    g3 = tp.FakeGz(ready_after=1, confirm=False, services=())
    called = []
    f3 = fc.FollowCamera("gz", "spot", "plant", (-2.5, 0.0, 1.6), 0.02, runner=g3.run, clock=lambda: 0.0,
                         sleep=lambda dt: None, log=lines.append, attempts=1,
                         inputs_factory=lambda: called.append(1))
    assert f3.main() == 0 and not called


def test_follow_camera_cli_guard_options(monkeypatch):
    seen = {}

    class Rec(fc.FollowCamera):
        def main(self):
            seen.update(cfg=dict(self.cam_cfg), factory=self.inputs_factory)
            return 0
    monkeypatch.setattr(fc, "FollowCamera", Rec)
    assert fc.main(["--gz", "gz", "--tilt-hold", "35", "--release-hold", "2", "--ros-args", "-r",
                    "__node:=follow_camera"]) == 0
    assert seen["cfg"]["tilt_hold_deg"] == 35.0 and seen["cfg"]["release_hold"] == 2.0 and seen["factory"]
    assert fc.main(["--gz", "gz", "--no-fall-guard"]) == 0 and seen["cfg"]["fall_guard"] is False
    assert seen["factory"]                                   # mouse view and keys still on
    assert fc.robot_down({"recovery_state": "TUCK"}) and fc.robot_down({"mode": "FALLEN"})
    assert not fc.robot_down({"recovery_state": "NORMAL", "mode": "WALK"}) and not fc.robot_down({})


# ============================================================================
# browser labels (i18n_ko.js, run with node.js)
# ============================================================================
@pytest.mark.skipif(__import__("shutil").which("node") is None, reason="node.js not installed")
def test_i18n_recovery_and_new_messages():
    import json
    import subprocess
    web = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "web", "i18n_ko.js")
    texts = ["full accepted - waiting up to 30 s for the patrol plan (patrol_planner still computing)",
             "start waiting for the patrol plan cancelled by 'rth'",
             "commands disabled", "command failed: RuntimeError: boom",
             "robot still down after 20.0 s (recovery FAILED) - sending 'recover'",
             "recovered at (1.2, 3.4) roll 0 deg pitch 0 deg", "mode RECOVERING -> STAND"]
    js = ("const KO = require(process.argv[1]); const t = JSON.parse(process.argv[2]);"
          "console.log(JSON.stringify({ev: t.map((x) => KO.event(x, '')),"
          " rec: ['FALL_DETECTED','DAMP','TUCK','ROCK','SLIDE','SELF_RIGHT','PREPARE','STAND_UP','VERIFY',"
          "'BACK_OFF','FAILED',"
          "'ASSISTED_RESET'].map((x) => KO.label('recovery', x))}));")
    r = subprocess.run(["node", "-e", js, os.path.realpath(web), json.dumps(texts)], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    for src, ko in zip(texts, out["ev"]):
        assert ko != src and not any(w in ko for w in ("waiting", "cancelled", "disabled", "failed", "still",
                                                        "recovery", "recovered", "mode ", "FAILED")), ko
    assert out["ev"][0] == "전체 임무 접수 - 순찰 계획 대기 (최대 30초, 경로 계산 중)"
    assert "자가 복구 실패" in out["ev"][4]
    assert all(not any("A" <= c <= "Z" for c in k) for k in out["rec"]), out["rec"]
