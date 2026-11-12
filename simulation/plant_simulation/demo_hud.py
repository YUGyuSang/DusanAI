"""
ROS-free compositor of the demo HUD (plant_simulation demo_hud node).

The HUD frame imitates the reference video (.dev/sample_video_notes.md): the
robot camera image with an overlay of

* top bar     "현재 가스 농도: 123.4 ppm" (ASCII fallback "GAS: 123.4 ppm"),
              coloured normal / warning / alarm, a log-scale ppm bar with
              tick marks at the warning and alarm thresholds, and the battery
              gauge + percent (charging marker) on the right;
* bottom bar  the current mission step in words ("G04 계기판으로 이동 중 (6/8)",
              "충전소로 복귀 중 · 사유: 배터리 부족" ...), locomotion mode /
              controller / gait, sim time;
* alarm       blinking red frame and a "가스 경보" badge while the gas alarm is
              active (mission gas_alarm, i.e. SEEK, or ppm >= gas_alarm_ppm);
* recovery    while the robot is down: a centre panel with the self-righting
              phase of /locomotion/state recovery_state and the four steps
              "넘어짐 감지 ▸ 자세 뒤집기 ▸ 일어서기 ▸ 복구 완료"; "복구 완료" for a
              few seconds after the robot stood up again;
* banners     for a few seconds after the event: the gauge value the mission
              just accepted (+ status, + the inspection photo of
              /inspection/gauge_snapshot in the corner), a skipped gauge, the
              gas source found (location error), the start of a return to the
              dock with its reason;
* final       docked after the whole mission: a result panel (gauges read,
              warnings, mean reading error against the layout ground truth, gas
              source, falls / self-recoveries, duration, where the results were
              saved - /metrics/session);
* no camera   dark placeholder "카메라 신호 없음" / "NO CAMERA SIGNAL".

:class:`HudStory` turns the latest /mission/state, /locomotion/state,
/metrics/session and /inspection/gauge_snapshot messages into these transient
items (wall-clock hold times, nothing is announced for the state found at
start-up).  Korean text needs Pillow and a Korean font (Noto Sans CJK, Nanum,
Malgun, ...).  Without either the HUD falls back to English ASCII text drawn
with OpenCV, so it works on every machine.  compose_hud() is pure numpy /
OpenCV and is unit-tested without ROS (simulation/test/test_plant_demo.py).
"""
import base64
import glob
import math
import os

import cv2
import numpy as np

from plant_simulation.dashboard_state import display_dir

HUD_DEFAULTS = {
    "rate": 5.0,                     # Hz HUD frames
    "width": 640,                    # px HUD frame (the 800 x 600 inspection camera is scaled to fit)
    "height": 480,
    "hud_topic": "/demo/hud",        # bgr8 for ROS tools (rqt_image_view, ros2 bag)
    "gz_hud_topic": "/demo/hud_rgb",  # rgb8 copy for ros_gz_bridge -> Gazebo topic /demo/hud ('' = off)
    "camera_timeout": 2.0,           # s (wall) without a camera frame -> 'no camera signal'
    "status_timeout": 5.0,           # s (wall) without gas / battery / locomotion / mission data -> '--'
    "gas_warn_ppm": 10.0,            # ppm, amber
    "gas_alarm_ppm": 25.0,           # ppm, red (= mission_manager gas_alarm_ppm)
    "gas_bar_max_ppm": 1000.0,       # ppm at the right end of the log-scale bar (leak ~1200 ppm)
    "font_path": "",                 # '' = search a Korean font (Noto Sans CJK, Nanum, Malgun ...)
    "language": "ko",                # ko | en (ko falls back to en without Pillow or a Korean font)
    "title": "DUSAN AI",
    "layout_file": "",               # world_layout.yaml for the mean reading error of the result panel ('' = installed)
    "gauge_hold": 7.0,               # s (wall) the 'gauge read' banner + photo stay up
    "event_hold": 6.0,               # s (wall) gas source / return-to-dock banners
    "recovered_hold": 5.0,           # s (wall) '복구 완료' after the robot stood up again
}

# colours (BGR)
COLORS = {
    "normal": (120, 230, 92),
    "warning": (25, 178, 250),
    "alarm": (77, 77, 255),
    "unknown": (170, 170, 170),
}
WHITE = (245, 245, 245)
GREY = (200, 200, 200)
DARK = (28, 28, 28)

# Korean-capable fonts, most specific first (Ubuntu fonts-noto-cjk / fonts-nanum, Windows, macOS)
KOREAN_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)
FONT_GLOBS = ("/usr/share/fonts/**/*CJK*", "/usr/share/fonts/**/Nanum*", "/usr/share/fonts/**/*KR*",
              "/usr/local/share/fonts/**/*CJK*", "~/.local/share/fonts/**/*CJK*", "~/.fonts/**/*CJK*")

MISSION_LABELS = {
    "IDLE": ("대기", "IDLE"),
    "PATROL": ("순찰 이동", "PATROL"),
    "INSPECTING": ("계기 판독", "INSPECTING"),
    "SEEK": ("가스 누출원 탐색", "GAS SEARCH"),
    "RTH": ("도크로 복귀", "RETURN HOME"),
    "DOCKED": ("도킹 (충전)", "DOCKED"),
    "MANUAL": ("수동 조작", "MANUAL"),
    "PAUSED": ("일시 정지", "PAUSED"),
    "DONE": ("완료", "DONE"),
}

LOCO_LABELS = {      # /locomotion/state mode, locomotion_mode, gait -> (ko, en)
    "WALK": ("걷는 중", "WALK"), "STAND": ("서 있음", "STAND"), "SIT": ("앉음", "SIT"),
    "STANDING_UP": ("일어서는 중", "STANDING UP"), "FALLEN": ("넘어짐", "FALLEN"),
    "RECOVERING": ("자가 복구 중", "RECOVERING"), "ESTOP": ("비상정지", "E-STOP"),
    "stable": ("안정 모드", "stable"), "mpc": ("MPC 모드", "MPC"), "trot": ("트롯", "trot"), "crawl": ("크롤", "crawl"),
}
# module1 fall_recovery states (/locomotion/state recovery_state) -> (ko, en)
RECOVERY_LABELS = {
    "FALL_DETECTED": ("충격 완화 준비", "FALL DETECTED"), "DAMP": ("충격 완화", "DAMPING"),
    "TUCK": ("다리 접기", "TUCKING LEGS"), "ROCK": ("몸 흔들어 뒤집기", "ROCKING OVER"),
    "SLIDE": ("몸 돌려 장애물 벗어나기", "TURNING OFF OBSTACLE"), "BACK_OFF": ("턱에서 내려오기", "STEPPING OFF LEDGE"),
    "SELF_RIGHT": ("몸 뒤집기", "SELF-RIGHTING"), "PREPARE": ("일어설 준비", "PREPARING"),
    "STAND_UP": ("다리 펴기", "EXTENDING LEGS"), "VERIFY": ("자세 확인", "CHECKING POSTURE"),
    "FAILED": ("자가 복구 실패 · 재시도 대기", "RECOVERY FAILED - WAITING"),
    "ASSISTED_RESET": ("보조 리셋 (위치 재설정)", "ASSISTED RESET"),
}
# the four steps shown while the robot is down; recovery_state -> step index
RECOVERY_STEPS = (("넘어짐 감지", "FALL"), ("자세 뒤집기", "SELF-RIGHT"), ("일어서기", "STAND UP"),
                  ("복구 완료", "RECOVERED"))
RECOVERY_STEP_OF = {"FALL_DETECTED": 0, "DAMP": 0, "FAILED": 0, "TUCK": 1, "ROCK": 1, "SLIDE": 1, "BACK_OFF": 1,
                    "SELF_RIGHT": 1,
                    "PREPARE": 2, "STAND_UP": 2, "VERIFY": 2, "ASSISTED_RESET": 2}
DOWN_MODES = ("FALLEN", "RECOVERING")
STATUS_LABELS = {"normal": ("정상", "NORMAL"), "warning": ("경고", "WARNING"), "alarm": ("경보", "ALARM")}
# return-to-home reasons (module3 rth_logic.REASONS + mission_logic) -> (ko, en)
RTH_REASONS = {
    "battery_low": ("배터리 부족", "battery low"), "energy_reserve": ("복귀 에너지 부족", "energy reserve"),
    "comm_shadow_predicted": ("통신 음영 예상", "comm shadow ahead"), "comm_lost": ("통신 두절", "comm lost"),
    "manual": ("수동 요청", "manual request"), "mission_complete": ("임무 완료", "mission complete"),
    "safety": ("안전 계층", "safety"),
}
SEEK_RESULTS = {"FOUND": ("발견", "FOUND"), "FAILED": ("실패", "FAILED"), "TIMEOUT": ("시간 초과", "TIMEOUT")}
RUNNING_STATES = ("PATROL", "INSPECTING", "SEEK", "RTH", "PAUSED", "MANUAL")

# HUD input: the node fills these keys (None = unknown / stale)
EMPTY_STATUS = {
    "gas_ppm": None,          # float ppm (/gas/sensor)
    "gas_alarm": False,       # bool (/mission/state gas_alarm)
    "battery_pct": None,      # float 0..100 (/battery/state)
    "charging": False,
    "loco_mode": "",          # controller state, e.g. WALK / STAND (/locomotion/state mode)
    "locomotion_mode": "",    # stable | mpc
    "gait": "",               # trot | crawl ...
    "mission_state": "",      # /mission/state state
    "mission_target": "",     # gauge id
    "mission_progress": None,  # [done, total]
    "rth_reason": "",         # /mission/state rth_reason
    "hold_reason": "",        # /mission/state hold_reason: why the mission does not drive ('' = it does)
    "mission_finished": False,  # docked / done after every gauge was handled
    "sim_time": None,         # s
    # HudStory.status(): None = nothing to show
    "recovery": None,         # robot down: {"state", "mode", "attempt"}
    "banner": None,           # recent event: {"kind": gauge|missed|gas_found|gas_end|rth|recovered, ...}
    "final": None,            # result panel at the dock (HudStory._summary keys)
}


# ----------------------------------------------------------------------------
# small pure helpers
# ----------------------------------------------------------------------------
def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def gas_level(ppm, warn_ppm, alarm_ppm, alarm_flag=False):
    """'unknown' | 'normal' | 'warning' | 'alarm'."""
    if alarm_flag:
        return "alarm"
    c = _finite(ppm)
    if c is None:
        return "unknown"
    if c >= float(alarm_ppm):
        return "alarm"
    if c >= float(warn_ppm):
        return "warning"
    return "normal"


def battery_level(pct):
    p = _finite(pct)
    if p is None:
        return "unknown"
    return "normal" if p >= 50.0 else ("warning" if p >= 20.0 else "alarm")


def gas_bar_fraction(ppm, max_ppm):
    """Log-scale fill fraction 0..1: log10(1 + c) / log10(1 + max)."""
    c = _finite(ppm)
    if c is None or c <= 0.0:
        return 0.0
    return float(min(1.0, math.log10(1.0 + c) / math.log10(1.0 + max(1.0, float(max_ppm)))))


def format_ppm(ppm):
    c = _finite(ppm)
    if c is None:
        return "--"
    return f"{c:.1f}" if abs(c) < 10000.0 else f"{c:.0f}"


def gas_text(ppm, korean):
    return f"현재 가스 농도: {format_ppm(ppm)} ppm" if korean else f"GAS: {format_ppm(ppm)} ppm"


def mission_text(state, target="", progress=None, korean=True):
    st = str(state or "").upper()
    if not st:
        return "임무: --" if korean else "MISSION: --"
    ko, en = MISSION_LABELS.get(st, (st, st))
    out = f"임무: {ko}" if korean else f"MISSION: {en}"
    if target and st in ("PATROL", "INSPECTING"):
        out += (" → " if korean else " -> ") + str(target)
    if isinstance(progress, (list, tuple)) and len(progress) == 2:
        try:
            done, total = int(progress[0]), int(progress[1])
            if total > 0:
                out += f" ({done}/{total})"
        except (TypeError, ValueError):
            pass
    return out


def locomotion_text(mode="", locomotion_mode="", gait="", korean=True):
    parts = [_lab(LOCO_LABELS, p, korean) for p in (mode, locomotion_mode, gait) if p]
    if not parts:
        return "보행: --" if korean else "LOCOMOTION: --"
    return ("보행: " + " · ".join(parts)) if korean else ("LOCOMOTION: " + " / ".join(parts))


def battery_text(pct):
    p = _finite(pct)
    return "--%" if p is None else f"{p:.0f}%"


def _lab(table, key, korean, default=None):
    """(ko, en) table lookup; unknown keys come back unchanged (or ``default``)."""
    v = table.get(str(key))
    if v is None:
        return str(key) if default is None else default
    return v[0 if korean else 1]


def _progress(progress):
    """(handled, total) of /mission/state progress, or None."""
    if isinstance(progress, (list, tuple)) and len(progress) == 2:
        try:
            done, total = int(progress[0]), int(progress[1])
        except (TypeError, ValueError):
            return None
        if total > 0:
            return max(0, done), total
    return None


def mission_step_text(state, target="", progress=None, rth_reason="", finished=False, korean=True):
    """The current mission step in words, e.g. "G04 계기판으로 이동 중 (6/8)" (the 6th of 8 gauges:
    progress counts the gauges already handled)."""
    st = str(state or "").upper()
    if not st:
        return "임무: --" if korean else "MISSION: --"
    pr = _progress(progress)
    step = f" ({min(pr[0] + 1, pr[1])}/{pr[1]})" if (pr and pr[0] < pr[1]) else ""
    tgt = str(target or "")
    if finished and st == "DOCKED":
        return "임무 완료 · 충전소 도착" if korean else "MISSION COMPLETE - DOCKED"
    if st == "PATROL":
        if tgt:
            return f"{tgt} 계기판으로 이동 중{step}" if korean else f"GOING TO GAUGE {tgt}{step}"
        return "순찰 이동 중" if korean else "PATROL"
    if st == "INSPECTING":
        if tgt:
            return f"{tgt} 계기판 판독 중{step}" if korean else f"READING GAUGE {tgt}{step}"
        return "계기판 판독 중" if korean else "READING GAUGE"
    if st == "SEEK":
        return "가스 누출원 추적 중" if korean else "SEARCHING THE GAS LEAK SOURCE"
    if st == "RTH":
        if rth_reason == "mission_complete":
            return "점검 완료 · 충전소로 복귀 중" if korean else "ALL GAUGES DONE - RETURNING TO DOCK"
        why = _lab(RTH_REASONS, rth_reason, korean) if rth_reason else ""
        if korean:
            return "충전소로 복귀 중" + (f" · 사유: {why}" if why else "")
        return "RETURNING TO DOCK" + (f" ({why})" if why else "")
    if st == "DOCKED":
        return "충전소 도착 · 충전 중" if korean else "DOCKED - CHARGING"
    if st == "IDLE":
        return "대기 중 · 't' 키로 임무 시작" if korean else "IDLE - press 't' to start"
    if st == "PAUSED":
        return "일시정지" + (f" · {tgt} 점검 대기" if tgt else "") if korean else "PAUSED"
    if st == "MANUAL":
        return "수동 조작 중" if korean else "MANUAL CONTROL"
    if st == "DONE":
        return "순찰 완료" if korean else "PATROL DONE"
    ko, en = MISSION_LABELS.get(st, (st, st))
    return f"임무: {ko}" if korean else f"MISSION: {en}"


# /mission/state hold_reason (mission_logic.MissionLogic.hold_reason) -> (ko, en) by prefix
HOLD_REASONS = (
    ("e-stop", ("비상정지 - 해제 필요", "E-STOP - release needed")),
    ("robot fallen", ("넘어짐", "fallen")),
    ("getting up", ("일어서는 중", "getting up")),
    ("standing up", ("일어서는 중", "standing up")),
    ("sitting", ("앉아 있음", "sitting")),
    ("no /locomotion/state", ("보행 제어기 응답 없음", "no locomotion state")),
    ("waiting for /odom", ("위치 정보 대기", "waiting for odometry")),
    ("waiting for the robot", ("일어서기 대기", "waiting to stand")),
    ("obstacle ahead", ("앞에 장애물", "obstacle ahead")),
    ("teleop has priority", ("텔레옵이 제어 중 (비상정지?)", "teleop has priority")),
    ("paused", ("일시정지", "paused")),
    ("manual override", ("수동 조작", "manual override")),
    ("locomotion", ("보행 준비 안 됨", "locomotion not ready")),
)


def hold_text(reason, korean=True):
    """Short label of a /mission/state hold_reason ('' -> '')."""
    r = str(reason or "").strip()
    if not r:
        return ""
    for prefix, (ko, en) in HOLD_REASONS:
        if r.startswith(prefix):
            return ko if korean else en
    return r


def format_value(value, unit=""):
    """Gauge value with 2 significant decimals for small numbers ("0.92 MPa", "85.3 °C")."""
    v = _finite(value)
    if v is None:
        return "--"
    txt = f"{v:.2f}" if abs(v) < 10.0 else (f"{v:.1f}" if abs(v) < 1000.0 else f"{v:.0f}")
    return f"{txt} {unit}".strip()


def format_duration(seconds, korean=True):
    s = _finite(seconds)
    if s is None:
        return "--"
    s = int(round(max(0.0, s)))
    m, sec = divmod(s, 60)
    if korean:
        return f"{m}분 {sec:02d}초" if m else f"{sec}초"
    return f"{m} min {sec:02d} s" if m else f"{sec} s"


def mission_finished(d):
    """True when a /mission/state dict says docked / done after every gauge was handled."""
    if not isinstance(d, dict):
        return False
    pr = _progress(d.get("progress"))
    handled = len(d.get("completed") or []) + len(d.get("missed") or [])
    return str(d.get("state") or "") in ("DOCKED", "DONE") and pr is not None and handled >= pr[1] > 0


def recovery_text(rec, korean=True):
    """(title, detail, step index 0..2, level) of the 'robot down' panel."""
    st = str((rec or {}).get("state") or "")
    mode = str((rec or {}).get("mode") or "")
    step = RECOVERY_STEP_OF.get(st, 0 if mode == "FALLEN" or not st else 1)
    if st == "FAILED":
        title = "자가 복구 실패" if korean else "SELF-RIGHTING FAILED"
    elif step == 0:
        title = "넘어짐 감지" if korean else "FALL DETECTED"
    elif step == 1:
        title = "몸을 뒤집는 중" if korean else "SELF-RIGHTING"
    else:
        title = "일어서는 중" if korean else "STANDING UP"
    phase = _lab(RECOVERY_LABELS, st, korean, "") if st and st != "NORMAL" else ""
    if st == "FAILED":
        phase = "재시도 대기" if korean else "waiting to retry"
    att = (rec or {}).get("attempt")
    parts = []
    if phase:
        parts.append((f"현재 단계: {phase}") if korean else f"phase: {phase}")
    if isinstance(att, int) and att > 0:
        parts.append(f"{att}번째 시도" if korean else f"attempt {att}")
    level = "alarm" if (step == 0 or st == "FAILED") else "warning"
    return title, " · ".join(parts), step, level


def banner_text(b, korean=True):
    """(main, sub, level) of a :class:`HudStory` banner."""
    b = b or {}
    kind = b.get("kind")
    gid = str(b.get("id") or "")
    if kind == "gauge":
        st = str(b.get("status") or "")
        lab = _lab(STATUS_LABELS, st, korean, "")
        val = format_value(b.get("value"), b.get("unit") or "")
        n = b.get("n")
        if korean:
            main = f"{gid} 판독 완료: {val}" + (f" · {lab}" if lab else "")
            sub = f"판독 {n}회 종합" if isinstance(n, int) and n > 0 else ""
        else:
            main = f"{gid} READ: {val}" + (f" - {lab}" if lab else "")
            sub = f"from {n} readings" if isinstance(n, int) and n > 0 else ""
        return main, sub, (st if st in ("warning", "alarm") else "normal")
    if kind == "missed":
        return ((f"{gid} 판독 실패 · 다음 계기로 이동" if korean else f"{gid}: NO READING - SKIPPED"),
                "", "warning")
    if kind == "gas_found":
        err = _finite(b.get("error_m"))
        est = b.get("estimate")
        sub = []
        if err is not None:
            sub.append(f"위치 오차 {err:.2f} m" if korean else f"location error {err:.2f} m")
        if isinstance(est, (list, tuple)) and len(est) >= 2 and _finite(est[0]) is not None \
                and _finite(est[1]) is not None:
            xy = f"({_finite(est[0]):.1f}, {_finite(est[1]):.1f})"
            sub.append(f"추정 위치 {xy}" if korean else f"estimate {xy}")
        return ("가스 누출원 발견" if korean else "GAS LEAK SOURCE FOUND"), " · ".join(sub), "alarm"
    if kind == "gas_end":
        res = _lab(SEEK_RESULTS, b.get("result"), korean)
        return ((f"누출원 추적 종료: {res}" if korean else f"LEAK SOURCE SEARCH ENDED: {res}"), "", "warning")
    if kind == "rth":
        why = _lab(RTH_REASONS, b.get("reason"), korean) if b.get("reason") else ""
        if korean:
            return "충전소 복귀 시작" + (f" · 사유: {why}" if why else ""), "", "warning"
        return "RETURNING TO DOCK" + (f" ({why})" if why else ""), "", "warning"
    if kind == "recovered":
        falls, ok, assisted = b.get("falls"), b.get("recovered"), b.get("assisted")
        sub = ""
        if isinstance(falls, int) and isinstance(ok, int):
            sub = (f"넘어짐 {falls}회 · 자가 복구 {ok}회" if korean else f"falls {falls} - self-recovered {ok}")
        if b.get("by_reset"):
            return ("보조 리셋으로 복귀" if korean else "BACK UP BY ASSISTED RESET"), sub, "warning"
        if isinstance(assisted, int) and assisted > 0:
            sub += (f" · 보조 리셋 {assisted}회" if korean else f" - assisted resets {assisted}")
        return ("복구 완료 · 스스로 일어섰습니다" if korean else "RECOVERED - BACK ON ITS FEET"), sub, "normal"
    return "", "", "normal"


def final_lines(f, korean=True):
    """(title, subtitle, [(label, value)]) of the result panel at the dock."""
    f = f or {}
    rows = []
    read, total, missed = f.get("read", 0), f.get("total", 0), f.get("missed", 0)
    warn, alarm = f.get("warning", 0), f.get("alarm", 0)
    if korean:
        g = f"{read} / {total}" + (f" (실패 {missed})" if missed else "") + f" · 경고 {warn} · 경보 {alarm}"
    else:
        g = f"{read} / {total}" + (f" ({missed} missed)" if missed else "") + f" - warning {warn}, alarm {alarm}"
    rows.append(("계기 판독" if korean else "Gauges read", g))
    err = _finite(f.get("mean_error_pct_fs"))
    if err is not None:
        mx = _finite(f.get("max_error_pct_fs"))
        rows.append(("판독 오차" if korean else "Reading error",
                     (f"평균 {err:.2f} % FS" + (f" · 최대 {mx:.2f} % FS" if mx is not None else "")) if korean else
                     (f"mean {err:.2f} % FS" + (f", max {mx:.2f} % FS" if mx is not None else ""))))
    seek, serr = f.get("seek"), _finite(f.get("seek_error_m"))
    if seek == "FOUND":
        gas = ("발견" if korean else "FOUND") + ((f" · 위치 오차 {serr:.2f} m" if korean else f", error {serr:.2f} m")
                                              if serr is not None else "")
    elif seek:
        gas = _lab(SEEK_RESULTS, seek, korean)
    else:
        gas = "가스 경보 없음" if korean else "no gas alarm"
    rows.append(("가스 누출원" if korean else "Gas leak source", gas))
    falls, ok, assisted = f.get("falls"), f.get("recovered"), f.get("assisted")
    if isinstance(falls, int):
        txt = (f"{falls}회 / {ok if isinstance(ok, int) else '--'}회" if korean else
               f"{falls} / {ok if isinstance(ok, int) else '--'}")
        if isinstance(assisted, int) and assisted > 0:
            txt += f" · 보조 리셋 {assisted}회" if korean else f" (assisted resets {assisted})"
        rows.append(("넘어짐 / 자가 복구" if korean else "Falls / self-recovered", txt))
    if f.get("saved"):
        rows.append(("결과 저장" if korean else "Saved to", str(f["saved"])))
    el = f.get("elapsed")
    if str(f.get("state") or "") == "DONE":
        sub = "순찰 완료" if korean else "patrol done"
    else:
        sub = "충전소 도착" if korean else "docked"
    if _finite(el) is not None:
        sub += (f" · 소요 {format_duration(el, True)}" if korean else f" - {format_duration(el, False)}")
    return ("임무 완료" if korean else "MISSION COMPLETE"), sub, rows


def decode_jpeg_b64(text, max_side=400):
    """BGR image of a base64 JPEG (snapshot jpeg_b64), downscaled to max_side; None if broken."""
    if not isinstance(text, str) or not text or len(text) > 4_000_000:
        return None
    try:
        buf = np.frombuffer(base64.b64decode(text, validate=True), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    s = float(max_side) / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return img


# ----------------------------------------------------------------------------
# story: transient HUD items from the latest messages
# ----------------------------------------------------------------------------
class HudStory:
    """Turns /mission/state, /locomotion/state, /metrics/session and
    /inspection/gauge_snapshot into the HUD's event items (all times are the
    caller's clock, e.g. time.monotonic()).

    * a gauge that appears in ``completed`` -> 'gauge read' banner with the
      mission's accepted value (``results``) and the inspection photo once the
      snapshot for that gauge arrived; ``missed`` -> 'skipped' banner;
    * the seek result changing to FOUND / FAILED / TIMEOUT -> gas banner;
    * the state changing to RTH for any reason but mission_complete -> banner;
    * /locomotion/state ``recovery_state`` other than NORMAL (or mode FALLEN /
      RECOVERING) -> the 'robot down' panel; back to NORMAL -> '복구 완료'
      (or '보조 리셋으로 복귀' when only ``assisted_resets`` went up);
    * docked / done after every gauge -> the result panel, frozen at that
      moment (the save location of /metrics/session is filled in live).

    The first message of every stream only sets the baseline: a restarted HUD
    does not re-announce old results."""

    PRIORITY = {"recovered": 4, "gas_found": 3, "gas_end": 2, "gauge": 1, "missed": 1, "rth": 0}

    def __init__(self, layout=None, gauge_hold=7.0, event_hold=6.0, recovered_hold=5.0, max_photos=16):
        self.layout = layout
        self.gauge_hold, self.event_hold, self.recovered_hold = float(gauge_hold), float(event_hold), \
            float(recovered_hold)
        self.max_photos = int(max_photos)
        self.mission = {}
        self.loco = {}
        self.session = {}
        self.photos = {}                 # gauge id -> BGR inspection photo (latest snapshot)
        self._banners = []               # [expires, seq, dict]
        self._seq = 0
        self._done = None                # completed / missed ids seen so far (None: no message yet)
        self._missed = None
        self._seek_key = "__unset__"
        self._state = None
        self._down = False
        self._down_base = None           # counters when the robot went down
        self._final = None

    # ------------------------------------------------------------ inputs
    def _push(self, now, hold, item):
        self._seq += 1
        self._banners.append([float(now) + float(hold), self._seq, item])
        self._banners = [b for b in self._banners if b[0] > now][-8:]

    def on_mission(self, d, now):
        if not isinstance(d, dict):
            return
        done = [str(g) for g in (d.get("completed") or []) if g is not None]
        missed = [str(g) for g in (d.get("missed") or []) if g is not None]
        results = d.get("results") if isinstance(d.get("results"), dict) else {}
        if self._done is not None:
            for gid in done:
                if gid not in self._done:
                    r = results.get(gid) if isinstance(results.get(gid), dict) else {}
                    n = _finite(r.get("n_readings"))
                    self._push(now, self.gauge_hold, {
                        "kind": "gauge", "id": gid, "value": _finite(r.get("value")), "unit": str(r.get("unit") or ""),
                        "status": str(r.get("status") or ""), "n": int(n) if n is not None else None})
            for gid in missed:
                if gid not in (self._missed or ()):
                    self._push(now, self.gauge_hold, {"kind": "missed", "id": gid})
        self._done, self._missed = set(done), set(missed)
        seek = d.get("seek") if isinstance(d.get("seek"), dict) else None
        key = (seek.get("result"), seek.get("t")) if seek else None
        if key != self._seek_key:
            if self._seek_key != "__unset__" and seek and seek.get("result"):
                if seek.get("result") == "FOUND":
                    self._push(now, self.event_hold, {"kind": "gas_found", "error_m": _finite(seek.get("error_m")),
                                                      "estimate": seek.get("estimate")})
                else:
                    self._push(now, self.event_hold, {"kind": "gas_end", "result": str(seek.get("result"))})
            self._seek_key = key
        st = str(d.get("state") or "")
        reason = str(d.get("rth_reason") or "")
        if st == "RTH" and self._state not in (None, "RTH") and reason != "mission_complete":
            self._push(now, self.event_hold, {"kind": "rth", "reason": reason})
        self._state = st
        if mission_finished(d):
            if self._final is None:
                self._final = self._summary(d)
        elif st in RUNNING_STATES or st == "IDLE":
            self._final = None
        self.mission = d

    @staticmethod
    def _counters(d):
        def i(k):
            v = d.get(k)
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
        return {k: i(k) for k in ("fall_count", "recovery_success", "assisted_resets", "recovery_attempts")}

    def on_locomotion(self, d, now):
        if not isinstance(d, dict):
            return
        rs = str(d.get("recovery_state") or "")
        mode = str(d.get("mode") or "")
        down = (rs not in ("", "NORMAL")) or mode in DOWN_MODES
        c = self._counters(d)
        if down and not self._down:
            self._down_base = c
        elif not down and self._down and self.loco:
            base = self._down_base or {}

            def up(k):
                return c.get(k) is not None and base.get(k) is not None and c[k] > base[k]
            by_reset = up("assisted_resets") and not up("recovery_success")
            self._push(now, self.recovered_hold, {"kind": "recovered", "by_reset": by_reset,
                                                  "falls": c.get("fall_count"), "recovered": c.get("recovery_success"),
                                                  "assisted": c.get("assisted_resets")})
        self._down = down
        self.loco = d

    def on_session(self, d, now=None):
        if isinstance(d, dict) and isinstance(d.get("session_dir"), str):
            self.session = {"dir": d["session_dir"], "report": d.get("report"),
                            "final": bool(d.get("report_final"))}

    def on_snapshot(self, d, now=None):
        """/inspection/gauge_snapshot: keep the gauge's photo (jpeg_b64, decoded once)."""
        if not isinstance(d, dict) or d.get("id") is None:
            return False
        img = decode_jpeg_b64(d.get("jpeg_b64"))
        if img is None:
            return False
        gid = str(d["id"])
        self.photos.pop(gid, None)
        self.photos[gid] = img
        while len(self.photos) > self.max_photos:
            self.photos.pop(next(iter(self.photos)))
        return True

    # ----------------------------------------------------------- summary
    def _summary(self, d):
        results = d.get("results") if isinstance(d.get("results"), dict) else {}
        done = [str(g) for g in (d.get("completed") or [])]
        pr = _progress(d.get("progress")) or (0, 0)
        warn = alarm = 0
        errs = []
        for gid in done:
            r = results.get(gid) if isinstance(results.get(gid), dict) else {}
            st = str(r.get("status") or "")
            v = _finite(r.get("value"))
            if self.layout is not None and v is not None:
                try:
                    g = self.layout.gauge(gid) or {}
                    span = float(g["max"]) - float(g["min"])
                    if span > 0 and g.get("value") is not None:
                        errs.append(abs(v - float(g["value"])) / span * 100.0)
                    if st not in ("normal", "warning", "alarm"):
                        st = self.layout.gauge_status(gid, v)
                except Exception:
                    pass
            warn += st == "warning"
            alarm += st == "alarm"
        seek = d.get("seek") if isinstance(d.get("seek"), dict) else {}
        c = self._counters(self.loco or {})
        return {"state": str(d.get("state") or ""), "read": len(done), "total": pr[1],
                "missed": len(d.get("missed") or []), "warning": warn, "alarm": alarm,
                "mean_error_pct_fs": (sum(errs) / len(errs)) if errs else None,
                "max_error_pct_fs": max(errs) if errs else None,
                "seek": seek.get("result"), "seek_error_m": _finite(seek.get("error_m")),
                "falls": c.get("fall_count"), "recovered": c.get("recovery_success"),
                "assisted": c.get("assisted_resets"), "elapsed": _finite(d.get("mission_elapsed"))}

    # ------------------------------------------------------------ output
    def status(self, now, mission=True, loco=True):
        """EMPTY_STATUS keys recovery / banner / final / mission_finished / rth_reason.
        mission / loco False: that stream is stale -> its items are left out."""
        out = {"recovery": None, "banner": None, "final": None, "mission_finished": False}
        if loco and self._down:
            d = self.loco
            base = self._down_base or {}
            att = None
            c = self._counters(d)
            if c.get("recovery_attempts") is not None and base.get("recovery_attempts") is not None:
                att = c["recovery_attempts"] - base["recovery_attempts"]
            out["recovery"] = {"state": str(d.get("recovery_state") or ""), "mode": str(d.get("mode") or ""),
                               "attempt": att if att and att > 0 else None}
        live = [b for b in self._banners if b[0] > now and (loco or b[2]["kind"] != "recovered")
                and (mission or b[2]["kind"] == "recovered")]
        if live:
            best = max(live, key=lambda b: (self.PRIORITY.get(b[2]["kind"], 0), b[1]))
            item = dict(best[2])
            if item["kind"] == "gauge":
                item["photo"] = self.photos.get(item["id"])
            out["banner"] = item
        if mission:
            out["rth_reason"] = str(self.mission.get("rth_reason") or "")
            fin = mission_finished(self.mission)
            out["mission_finished"] = fin
            if fin and self._final is not None:
                f = dict(self._final)
                if self.session.get("dir"):
                    f["saved"] = display_dir(self.session["dir"])
                    f["report"] = self.session.get("report")
                out["final"] = f
        return out


def fit_image(img, width, height):
    """Scale an image into width x height keeping its aspect ratio (black bars)."""
    out = np.zeros((height, width, 3), np.uint8)
    if img is None or img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return out
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(width / float(w), height / float(h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    small = cv2.resize(img[..., :3], (nw, nh), interpolation=interp)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def _shade(img, x0, y0, x1, y1, alpha=0.55, color=(0, 0, 0)):
    """Blend a translucent rectangle into img (in place)."""
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = np.empty_like(roi)
    over[:] = color
    cv2.addWeighted(over, float(alpha), roi, 1.0 - float(alpha), 0.0, dst=roi)


# ----------------------------------------------------------------------------
# text
# ----------------------------------------------------------------------------
def find_korean_font(extra=()):
    """Path of the first readable Korean-capable font, or None."""
    for p in list(extra) + list(KOREAN_FONT_CANDIDATES):
        if p and os.path.isfile(p):
            return p
    for pattern in FONT_GLOBS:
        try:
            hits = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
        except OSError:
            continue
        hits = [h for h in hits if h.lower().endswith((".ttf", ".ttc", ".otf"))]
        bold = [h for h in hits if "bold" in os.path.basename(h).lower()]
        if bold or hits:
            return (bold or hits)[0]
    return None


class TextRenderer:
    """Draws HUD text: Pillow + a Korean font when available, else OpenCV
    Hershey (ASCII only, English labels).  ``korean`` tells which one is used."""

    def __init__(self, font_path="", language="ko"):
        self.font_file = None
        self.reason = ""
        self._pil = None                                     # (Image, ImageDraw, ImageFont) modules
        self._fonts = {}
        want_korean = str(language or "ko").lower().startswith("ko")
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:                                    # Pillow missing -> OpenCV text
            self.reason = "Pillow (python3-pil) not installed"
        else:
            if font_path and not os.path.isfile(font_path):
                self.reason = f"font_path {font_path} not found"
            path = font_path if (font_path and os.path.isfile(font_path)) else find_korean_font()
            if path:
                self._pil = (Image, ImageDraw, ImageFont)
                self.font_file = path
                try:
                    self._font(20)
                except Exception as exc:                     # unreadable font file
                    self.reason = f"cannot load {path}: {exc}"
                    self._pil, self.font_file, self._fonts = None, None, {}
            elif not self.reason:
                self.reason = "no Korean font found (install fonts-noto-cjk or fonts-nanum)"
        self.pillow = self.font_file is not None
        self.korean = self.pillow and want_korean

    # -- Pillow font cache
    def _font(self, size):
        size = max(6, int(size))
        f = self._fonts.get(size)
        if f is None:
            path = self.font_file
            # Noto Sans CJK collections: face 1 = Korean (KR) glyph variants
            index = 1 if ("CJK" in os.path.basename(path) and path.lower().endswith(".ttc")) else 0
            try:
                f = self._pil[2].truetype(path, size, index=index)
            except OSError:
                f = self._pil[2].truetype(path, size, index=0)
            self._fonts[size] = f
        return f

    @staticmethod
    def _ascii(text):
        return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(text))

    @staticmethod
    def _cv_scale(size):
        return max(0.3, float(size) / 30.0), max(1, int(round(size / 15.0)))

    def measure(self, text, size):
        """(width, height) in px of text drawn at the given pixel size."""
        if self.pillow:
            f = self._font(size)
            return int(math.ceil(f.getlength(str(text)))), int(size)
        scale, th = self._cv_scale(size)
        (w, h), _ = cv2.getTextSize(self._ascii(text), cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        return int(w), int(size)

    def draw(self, img, items):
        """Draw [(text, x, y_top, size_px, bgr)] into img (BGR uint8, in place)."""
        items = [it for it in items if it and str(it[0])]
        if not items:
            return img
        if self.pillow:
            # the array is BGR; Pillow does not care about channel meaning, so
            # colours are passed in BGR order and no channel swap is needed
            pil = self._pil[0].fromarray(img)
            d = self._pil[1].Draw(pil)
            for text, x, y, size, color in items:
                f = self._font(size)
                d.text((int(x), int(y)), str(text), font=f, fill=tuple(int(c) for c in color),
                       anchor="la", stroke_width=max(1, int(size) // 14), stroke_fill=(0, 0, 0))
            img[:] = np.asarray(pil)
            return img
        for text, x, y, size, color in items:
            scale, th = self._cv_scale(size)
            t = self._ascii(text)
            (_, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
            org = (int(x), int(y + h + max(1, size * 0.12)))
            cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, tuple(int(c) for c in color), th,
                        cv2.LINE_AA)
        return img


_DEFAULT_RENDERER = {}


def default_renderer(language="ko", font_path=""):
    key = (language, font_path)
    if key not in _DEFAULT_RENDERER:
        _DEFAULT_RENDERER[key] = TextRenderer(font_path, language)
    return _DEFAULT_RENDERER[key]


# ----------------------------------------------------------------------------
# compositor
# ----------------------------------------------------------------------------
def _battery_icon(img, x, y, w, h, pct, color):
    t = max(1, int(round(h / 10.0)))
    cv2.rectangle(img, (x, y), (x + w, y + h), WHITE, t, cv2.LINE_AA)
    nub_h = max(2, h // 2)
    cv2.rectangle(img, (x + w, y + (h - nub_h) // 2), (x + w + max(2, w // 12), y + (h + nub_h) // 2), WHITE, -1)
    p = _finite(pct)
    if p is not None:
        inner = max(0, int(round((w - 4 * t) * min(100.0, max(0.0, p)) / 100.0)))
        if inner > 0:
            cv2.rectangle(img, (x + 2 * t, y + 2 * t), (x + 2 * t + inner, y + h - 2 * t), color, -1)


def compose_hud(camera_bgr, status=None, cfg=None, renderer=None, t=None):
    """One HUD frame (BGR uint8, cfg height x width).

    camera_bgr  robot camera image (BGR / mono uint8) or None (no signal)
    status      dict with the EMPTY_STATUS keys (missing keys = unknown)
    cfg         HUD_DEFAULTS overrides (width, height, thresholds, title ...)
    renderer    TextRenderer (default: shared one for cfg language/font_path)
    t           wall time [s] for the alarm blink (None = no blinking, alarm drawn)
    """
    c = dict(HUD_DEFAULTS, **(cfg or {}))
    st = dict(EMPTY_STATUS, **(status or {}))
    W, H = max(160, int(c["width"])), max(120, int(c["height"]))
    r = renderer if renderer is not None else default_renderer(c["language"], c["font_path"])
    ko = bool(r.korean)
    s = min(W / 640.0, H / 480.0)

    def S(v):
        return max(1, int(round(v * s)))

    # ------------------------------------------------------------ background
    have_cam = camera_bgr is not None and getattr(camera_bgr, "size", 0) > 0
    if have_cam:
        img = fit_image(np.asarray(camera_bgr), W, H)
    else:
        img = np.full((H, W, 3), DARK, np.uint8)
        cv2.line(img, (0, 0), (W, H), (45, 45, 45), S(2), cv2.LINE_AA)
        cv2.line(img, (0, H), (W, 0), (45, 45, 45), S(2), cv2.LINE_AA)
    texts = []
    if not have_cam:
        msg = "카메라 신호 없음" if ko else "NO CAMERA SIGNAL"
        tw, th = r.measure(msg, S(26))
        _shade(img, (W - tw) // 2 - S(14), H // 2 - S(22), (W + tw) // 2 + S(14), H // 2 + S(24), 0.7)
        texts.append((msg, (W - tw) // 2, H // 2 - S(15), S(26), GREY))

    # ------------------------------------------------------------- top bar
    top_h = S(72)
    _shade(img, 0, 0, W, top_h, 0.55)
    level = gas_level(st["gas_ppm"], c["gas_warn_ppm"], c["gas_alarm_ppm"], bool(st["gas_alarm"]))
    gcol = COLORS[level]
    texts.append((gas_text(st["gas_ppm"], ko), S(12), S(8), S(28), gcol if level != "normal" else WHITE))
    # ppm bar (log scale) + threshold ticks
    bx0, by0 = S(12), S(48)
    bx1, by1 = int(W * 0.55), S(60)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (70, 70, 70), -1)
    frac = gas_bar_fraction(st["gas_ppm"], c["gas_bar_max_ppm"])
    if frac > 0.0:
        cv2.rectangle(img, (bx0, by0), (bx0 + max(1, int((bx1 - bx0) * frac)), by1), gcol, -1)
    for thr in (c["gas_warn_ppm"], c["gas_alarm_ppm"]):
        tx = bx0 + int((bx1 - bx0) * gas_bar_fraction(thr, c["gas_bar_max_ppm"]))
        cv2.line(img, (tx, by0 - S(3)), (tx, by1 + S(3)), WHITE, max(1, S(2)))
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (150, 150, 150), 1)
    # battery (right)
    bpct = st["battery_pct"]
    bcol = COLORS[battery_level(bpct)]
    btxt = battery_text(bpct)
    tw, _ = r.measure(btxt, S(22))
    iw, ih = S(44), S(22)
    tx = W - S(12) - tw
    ix = tx - S(10) - iw - S(4)
    _battery_icon(img, ix, S(12), iw, ih, bpct, bcol)
    texts.append((btxt, tx, S(10), S(22), WHITE))
    sub = ("충전 중" if ko else "CHARGING") if st["charging"] else ("배터리" if ko else "BATTERY")
    sw, _ = r.measure(sub, S(15))
    texts.append((sub, W - S(12) - sw, S(42), S(15), COLORS["normal"] if st["charging"] else GREY))

    # ----------------------------------------------------------- bottom bar
    bot_h = S(64)
    _shade(img, 0, H - bot_h, W, H, 0.6)
    cam_label = "로봇 카메라" if ko else "ROBOT CAM"
    title = f"{c['title']}{' · ' if ko else ' | '}{cam_label}" if c["title"] else cam_label
    tw_title, _ = r.measure(title, S(15))
    texts.append((title, W - S(12) - tw_title, H - bot_h + S(9), S(15), GREY))
    simt = _finite(st["sim_time"])
    tw_time = 0
    if simt is not None:
        ttxt = f"t = {simt:.1f} s"
        tw_time, _ = r.measure(ttxt, S(15))
        texts.append((ttxt, W - S(12) - tw_time, H - bot_h + S(37), S(15), GREY))
    fin = bool(st["mission_finished"])
    step = mission_step_text(st["mission_state"], st["mission_target"], st["mission_progress"], st["rth_reason"],
                             fin, ko)
    mstate = str(st["mission_state"] or "").upper()
    hold = hold_text(st.get("hold_reason"), ko) if (mstate in ("PATROL", "INSPECTING", "RTH")
                                                    and not st.get("recovery")) else ""
    if hold:                       # the robot stands although the step says 'going to ...': say why
        step += (" · 정지: " if ko else " - HOLD: ") + hold
    mcol = (COLORS["normal"] if fin else COLORS["warning"] if (mstate == "RTH" and st["rth_reason"] not in
                                                                 ("", "mission_complete")) else WHITE)
    size = _fit_size(r, step, S(24), W - S(36) - tw_title, S(15))
    texts.append((step, S(12), H - bot_h + S(5) + (S(24) - size) // 2, size, mcol))
    loco = locomotion_text(st["loco_mode"], st["locomotion_mode"], st["gait"], ko)
    texts.append((loco, S(12), H - bot_h + S(36), _fit_size(r, loco, S(16), W - S(36) - tw_time, S(11)), GREY))

    # ---------------------------------------------------------------- alarm
    badge_bottom = top_h
    if level == "alarm":
        blink_on = t is None or int(float(t) * 2.0) % 2 == 0
        if blink_on:
            th = S(6)
            cv2.rectangle(img, (th // 2, th // 2), (W - 1 - th // 2, H - 1 - th // 2), COLORS["alarm"], th)
        badge = "가스 경보" if ko else "GAS ALARM"
        bw, _ = r.measure(badge, S(24))
        x0, y0 = (W - bw) // 2 - S(14), top_h + S(10)
        cv2.rectangle(img, (x0, y0), (x0 + bw + S(28), y0 + S(36)),
                      COLORS["alarm"] if blink_on else (40, 40, 160), -1)
        texts.append((badge, x0 + S(14), y0 + S(5), S(24), WHITE))
        badge_bottom = y0 + S(36)

    # ------------------------------------------------- robot down / result / banner
    rec = st["recovery"] if isinstance(st["recovery"], dict) else None
    ban = st["banner"] if isinstance(st["banner"], dict) else None
    fnl = st["final"] if isinstance(st["final"], dict) else None
    if rec is not None:
        _recovery_panel(img, texts, r, rec, ko, W, H, top_h, bot_h, S)
        if ban is not None and ban.get("kind") != "recovered":
            ban = None                                  # the panel says it all
    elif fnl is not None:
        _final_panel(img, texts, r, fnl, ko, W, badge_bottom + S(8), H - bot_h - S(8), S)
        if ban is not None and ban.get("kind") not in ("recovered", "gas_found"):
            ban = None
    if ban is not None:
        _banner(img, texts, r, ban, ko, W, H, top_h, bot_h, S, photo_ok=fnl is None and rec is None)

    r.draw(img, texts)
    return np.ascontiguousarray(img)


def _fit_size(r, text, size, max_w, min_size):
    """Largest font size <= size (down to min_size) that keeps text within max_w px."""
    size = int(size)
    while size > int(min_size) and r.measure(text, size)[0] > max_w:
        size -= 1
    return size


def _box(img, x0, y0, x1, y1, color, alpha=0.78, border=2):
    _shade(img, x0, y0, x1, y1, alpha)
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, max(1, int(border)), cv2.LINE_AA)


def _recovery_panel(img, texts, r, rec, ko, W, H, top_h, bot_h, S):
    """Centre panel while the robot is down: phase title, detail, the four steps."""
    title, detail, step, level = recovery_text(rec, ko)
    col = COLORS[level]
    x0, x1 = S(28), W - S(28)
    ph = S(132)
    y0 = max(top_h + S(56), (H - bot_h + top_h) // 2 - ph // 2)
    y1 = y0 + ph
    _box(img, x0, y0, x1, y1, col, 0.8, S(3))
    ts = _fit_size(r, title, S(34), x1 - x0 - S(24), S(18))
    tw, _ = r.measure(title, ts)
    texts.append((title, (W - tw) // 2, y0 + S(8), ts, col))
    if detail:
        ds = _fit_size(r, detail, S(18), x1 - x0 - S(24), S(11))
        dw, _ = r.measure(detail, ds)
        texts.append((detail, (W - dw) // 2, y0 + S(50), ds, WHITE))
    # the four steps: done = green, current = phase colour, later = grey; arrows between
    n = len(RECOVERY_STEPS)
    gap = S(16)
    cw = (x1 - x0 - S(20) - gap * (n - 1)) // n
    cy0, cy1 = y1 - S(40), y1 - S(10)
    for i, (sk, se) in enumerate(RECOVERY_STEPS):
        cx0 = x0 + S(10) + i * (cw + gap)
        if i < step:
            fill, fg = (60, 120, 48), WHITE
        elif i == step:
            fill, fg = tuple(int(v * 0.75) for v in col), WHITE
        else:
            fill, fg = (55, 55, 55), (165, 165, 165)
        cv2.rectangle(img, (cx0, cy0), (cx0 + cw, cy1), fill, -1)
        if i == step:
            cv2.rectangle(img, (cx0, cy0), (cx0 + cw, cy1), WHITE, max(1, S(2)))
        label = sk if ko else se
        ls = _fit_size(r, label, S(16), cw - S(8), S(10))
        lw, _ = r.measure(label, ls)
        texts.append((label, cx0 + (cw - lw) // 2, cy0 + (cy1 - cy0 - ls) // 2 - S(1), ls, fg))
        if i < n - 1:
            ax = cx0 + cw + S(3)
            cv2.arrowedLine(img, (ax, (cy0 + cy1) // 2), (ax + gap - S(6), (cy0 + cy1) // 2), (200, 200, 200),
                            max(1, S(2)), cv2.LINE_AA, tipLength=0.5)


def _final_panel(img, texts, r, fnl, ko, W, y_top, y_bot, S):
    """Result panel at the dock: title, subtitle, label / value rows."""
    title, sub, rows = final_lines(fnl, ko)
    x0, x1 = S(24), W - S(24)
    row_h = S(31)
    need = S(92) + row_h * len(rows)
    y0 = int(y_top)
    y1 = min(int(y_bot), y0 + need)
    col = COLORS["normal"]
    _box(img, x0, y0, x1, y1, col, 0.82, S(3))
    ts = S(34)
    tw, _ = r.measure(title, ts)
    texts.append((title, (W - tw) // 2, y0 + S(8), ts, col))
    ss = _fit_size(r, sub, S(19), x1 - x0 - S(24), S(11))
    sw, _ = r.measure(sub, ss)
    texts.append((sub, (W - sw) // 2, y0 + S(50), ss, WHITE))
    lx = x0 + S(16)
    label_w = max([r.measure(lab, S(19))[0] for lab, _ in rows] + [0]) + S(16)
    vx = lx + label_w
    y = y0 + S(84)
    for lab, val in rows:
        if y + row_h > y1 + S(4):
            break
        texts.append((lab, lx, y + S(2), S(19), GREY))
        vs = _fit_size(r, val, S(22), x1 - S(14) - vx, S(11))
        texts.append((val, vx, y + (S(22) - vs) // 2, vs, WHITE))
        y += row_h


def _banner(img, texts, r, ban, ko, W, H, top_h, bot_h, S, photo_ok=True):
    """Event banner above the bottom bar (+ the gauge photo in the top right corner)."""
    main, sub, level = banner_text(ban, ko)
    if not main:
        return
    col = COLORS.get(level, COLORS["normal"])
    ms = _fit_size(r, main, S(28), W - S(64), S(14))
    mw, _ = r.measure(main, ms)
    ss = _fit_size(r, sub, S(17), W - S(64), S(10)) if sub else 0
    sw = r.measure(sub, ss)[0] if sub else 0
    bw = max(mw, sw) + S(32)
    bh = S(46) + (S(24) if sub else 0)
    x0 = (W - bw) // 2
    y1 = H - bot_h - S(10)
    y0 = y1 - bh
    _box(img, x0, y0, x0 + bw, y1, col, 0.8, S(3))
    texts.append((main, (W - mw) // 2, y0 + S(8), ms, WHITE if level == "normal" else col))
    if sub:
        texts.append((sub, (W - sw) // 2, y0 + S(42), ss, GREY))
    photo = ban.get("photo") if photo_ok else None
    if photo is not None and getattr(photo, "size", 0) > 0 and photo.ndim == 3:
        ph_h = min(S(210), max(S(60), y0 - top_h - S(16)))
        h, w = photo.shape[:2]
        pw = max(1, int(round(w * ph_h / float(h))))
        if pw > W // 2:
            pw = W // 2
            ph_h = max(1, int(round(h * pw / float(w))))
        small = cv2.resize(photo, (pw, ph_h), interpolation=cv2.INTER_AREA)
        px0, py0 = max(0, W - S(12) - pw), top_h + S(8)
        y1c, x1c = min(img.shape[0], py0 + ph_h), min(img.shape[1], px0 + pw)
        if y1c > py0 and x1c > px0:
            img[py0:y1c, px0:x1c] = small[:y1c - py0, :x1c - px0]
            cv2.rectangle(img, (px0 - S(2), py0 - S(2)), (x1c + S(1), y1c + S(1)), col, max(1, S(3)))


class HudCompositor:
    """compose_hud() with a fixed configuration and text renderer."""

    def __init__(self, cfg=None, renderer=None):
        self.cfg = dict(HUD_DEFAULTS, **(cfg or {}))
        self.renderer = renderer if renderer is not None else TextRenderer(self.cfg["font_path"],
                                                                           self.cfg["language"])

    def compose(self, camera_bgr, status, t=None):
        return compose_hud(camera_bgr, status, self.cfg, self.renderer, t)
