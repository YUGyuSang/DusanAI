"""
Inspection report of one metrics session (ROS-free): ``inspection_report.json`` and a
self-contained Korean ``inspection_report.html`` (inline CSS, dial photos embedded as data URIs,
no script), written by plant_simulation.metrics.MetricsRecorder into the session directory.

Per gauge the OFFICIAL result is the mission's answer (``/mission/state`` ``results``: the reading
the mission accepted when it left the viewpoint, or ``missed``); without a mission result the
evidence snapshot value (``/inspection/gauge_snapshot``) and then the latest reading are used and
marked as such.  Ground truth (value, status) comes from the layout.
"""
import base64
import html
import math
import os
import re

STATUS_KO = {"normal": "정상", "warning": "경고", "alarm": "경보", "unknown": "판독 불가"}
RESULT_KO = {"read": "판독 완료", "missed": "판독 실패", "no_value": "값 없음", "pending": "점검 대기",
             "not_visited": "미점검"}
SOURCE_KO = {"mission": "임무 확정값", "snapshot": "점검 사진 값", "readings": "최근 관측값"}
STATE_KO = {"IDLE": "대기", "PATROL": "순찰", "INSPECTING": "점검 중", "SEEK": "가스 추적", "RTH": "복귀 중",
            "DOCKED": "충전소 도착", "DONE": "완료", "MANUAL": "수동 조작", "PAUSED": "일시정지"}
SEEK_KO = {"FOUND": "발견", "FAILED": "실패", "TIMEOUT": "시간 초과", "IDLE": "대기", "SEARCH": "탐색",
           "SURGE": "바람 거슬러 전진", "CAST": "좌우 탐색", "GRADIENT": "농도 경사 추종"}
FINAL_STATES = ("DONE", "DOCKED")
MAX_EMBED_BYTES = 600 * 1024              # larger photos are linked instead of embedded
_SAFE_REL = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]{0,200}$")


def num(v):
    """Finite float or None (bools are not numbers)."""
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def safe_rel(rel):
    """``rel`` if it is a plain relative path below the session dir (no '..', no hidden parts)."""
    if not isinstance(rel, str) or not _SAFE_REL.match(rel):
        return None
    parts = rel.split("/")
    if any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return None
    return rel


def value_decimals(lo, hi):
    try:
        span = abs(float(hi) - float(lo))
    except (TypeError, ValueError):
        return 2
    return 3 if span <= 2.0 else (2 if span <= 20.0 else 1)


# ------------------------------------------------------------------------------------- rows
def gauge_rows(gauges, layout, mission_results, snapshots, gauge_last, pending=(), final=False):
    """One row per layout gauge (``gauges``: id -> layout entry), in id order."""
    rows = []
    for gid in sorted(gauges):
        g = gauges[gid]
        res = mission_results.get(gid) if isinstance(mission_results.get(gid), dict) else {}
        snap = snapshots.get(gid) or {}
        last = gauge_last.get(gid) or {}
        val, src, reason = None, "", None
        if res.get("status") == "missed":
            result, reason = "missed", str(res.get("reason") or "no reading")
        elif num(res.get("value")) is not None:
            result, val, src = "read", num(res.get("value")), "mission"
        elif num(snap.get("value")) is not None:
            result, val, src = "read", num(snap.get("value")), "snapshot"
        elif num(last.get("value")) is not None:
            result, val, src = "read", num(last.get("value")), "readings"
        elif res or snap:                          # visited, but no usable number
            result = "no_value"
        else:
            result = "pending" if (gid in pending and not final) else "not_visited"
        lo, hi = num(g.get("min")), num(g.get("max"))
        span = (hi - lo) if (lo is not None and hi is not None) else None
        true = num(g.get("value"))
        err = (val - true) if (val is not None and true is not None) else None
        err_fs = abs(err) / span * 100.0 if (err is not None and span) else None
        status = ""
        if val is not None:
            status = (res.get("status") if src == "mission" else snap.get("status") if src == "snapshot"
                      else last.get("status"))
            status = status if isinstance(status, str) else ""
            if status not in STATUS_KO and layout is not None:
                status = layout.gauge_status(gid, val)
        true_status = layout.gauge_status(gid, true) if (layout is not None and true is not None) else ""
        conf = num(res.get("confidence")) if src == "mission" else num(snap.get("confidence")) \
            if src == "snapshot" else num(last.get("confidence"))
        t_res = num(res.get("t")) if src == "mission" else num(snap.get("stamp")) if src == "snapshot" \
            else num(last.get("t"))
        rows.append({
            "id": gid, "tag": g.get("tag", ""), "result": result, "value_source": src, "reason": reason,
            "value": None if val is None else round(val, 4), "unit": g.get("unit", ""),
            "range": [lo, hi], "decimals": value_decimals(lo, hi),
            "status": status, "true_value": true, "true_status": true_status,
            "error": None if err is None else round(err, 4),
            "error_pct_fs": None if err_fs is None else round(err_fs, 3),
            "status_correct": (status == true_status) if (val is not None and true_status) else None,
            "confidence": None if conf is None else round(conf, 3),
            "n_readings": res.get("n_readings") if src == "mission" else snap.get("readings"),
            "t_sim": None if t_res is None else round(t_res, 2),
            "wall": snap.get("wall") if isinstance(snap.get("wall"), str) else None,
            "image": safe_rel(snap.get("image")),
            "image_full": safe_rel(snap.get("image_full")), "snapshots": int(snap.get("count") or 0),
        })
    return rows


def thermal_rows(hotspots, layout, detected, peak, last_status):
    rows = []
    th = layout.thermal if layout is not None else {}
    warn_k = float(th.get("warning_k", 333.15))
    for hid in sorted(hotspots):
        h = hotspots[hid]
        tk = num(h.get("temp_k"))
        true_c = None if tk is None else round(tk - 273.15, 1)
        meas = num(peak.get(hid))
        rows.append({
            "id": hid, "tag": h.get("tag", ""), "kind": h.get("kind", ""),
            "true_temp_c": true_c,
            "true_status": layout.temperature_status(tk) if (layout is not None and tk is not None) else "",
            "expected": bool(tk is not None and tk >= warn_k), "detected": hid in detected,
            "peak_temp_c": None if meas is None else round(meas, 1),
            "status": str(last_status.get(hid) or ""),
            "temp_error_c": None if (meas is None or true_c is None) else round(meas - true_c, 1),
        })
    return rows


def build_report(summary, rows, trows, mission_seek=None, final=False, session_name="", written_wall=""):
    """JSON document of the report (also the input of :func:`render_html`)."""
    errs = [r["error_pct_fs"] for r in rows if r["error_pct_fs"] is not None]
    read = [r for r in rows if r["result"] == "read"]
    m = summary.get("mission", {})
    loco = summary.get("locomotion", {})
    th = summary.get("thermal", {})
    gas = dict(summary.get("gas", {}))
    if isinstance(mission_seek, dict):
        gas["mission_seek"] = mission_seek
    return {
        "title": "설비 점검 보고서", "session": session_name, "written_wall": written_wall, "final": bool(final),
        "mission": {
            "state": m.get("final_state"), "start_t": m.get("start_t"), "end_t": m.get("end_t"),
            "duration_s": m.get("duration_s"), "completed": m.get("completed") or [], "missed": m.get("missed") or [],
            "distance_m": summary.get("distance_m"), "t_sim_end": summary.get("t_sim_end"),
            "falls": loco.get("falls"), "recoveries": loco.get("recoveries"),
            "assisted_resets": loco.get("assisted_resets"), "rth_count": (summary.get("rth") or {}).get("count"),
            "battery": summary.get("battery"),
        },
        "gauges": {
            "total": len(rows), "read": len(read), "missed": [r["id"] for r in rows if r["result"] == "missed"],
            "mean_error_pct_fs": round(sum(errs) / len(errs), 3) if errs else None,
            "max_error_pct_fs": round(max(errs), 3) if errs else None,
            "status_correct": sum(1 for r in rows if r["status_correct"]),
            "status_checked": sum(1 for r in rows if r["status_correct"] is not None),
            "warnings": [r["id"] for r in rows if r["status"] == "warning"],
            "alarms": [r["id"] for r in rows if r["status"] == "alarm"],
            "rows": rows,
        },
        "thermal": {"expected": th.get("expected") or [], "detected": th.get("detected") or [],
                    "recall": th.get("recall"), "precision": th.get("precision"), "tp": th.get("tp"),
                    "fp": th.get("fp"), "rows": trows},
        "gas": gas,
    }


# ------------------------------------------------------------------------------------- html
def _e(v):
    return html.escape("" if v is None else str(v), quote=True)


def _n(v, nd=2, unit=""):
    x = num(v)
    if x is None:
        return "—"
    return f"{x:.{nd}f}" + (f" {unit}" if unit else "")


def _dur(s):
    x = num(s)
    if x is None:
        return "—"
    x = int(round(x))
    return f"{x // 60}분 {x % 60:02d}초"


def _badge(status):
    if not status:
        return '<span class="badge none">—</span>'
    return f'<span class="badge {_e(status)}">{_e(STATUS_KO.get(status, status))}</span>'


def _image_cell(session_dir, r):
    rel = r.get("image")
    if not rel:
        return '<div class="nophoto">사진 없음</div>'
    path = os.path.join(session_dir, *rel.split("/"))
    src = None
    try:
        if os.path.getsize(path) <= MAX_EMBED_BYTES:
            with open(path, "rb") as f:
                src = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return '<div class="nophoto">사진 없음</div>'
    src = src or rel
    img = f'<img src="{_e(src)}" alt="{_e(r["id"])} 점검 사진">'
    link = r.get("image_full") or rel
    return f'<a href="{_e(link)}" title="전체 화면 사진 열기">{img}</a>'


_CSS = """
:root{--ink:#16181d;--muted:#5b6070;--line:#d9dce3;--bg:#f6f7f9;--card:#fff;--good:#1a7f37;--warn:#b25e00;
--crit:#c62828;--accent:#2457c5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 "Noto Sans KR","Malgun Gothic","Apple SD Gothic Neo",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 40px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 10px}
.meta{color:var(--muted);margin:0 0 16px}.meta b{color:var(--ink)}
.state{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:700;font-size:13px;
background:#e8eefb;color:var(--accent)}.state.final{background:#e6f4ea;color:var(--good)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi .k{color:var(--muted);font-size:12px}.kpi .v{font-size:20px;font-weight:700}.kpi .s{color:var(--muted);font-size:12px}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:760px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:middle}th{background:#eef0f4;font-size:12px;color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.photo{width:150px}td.photo img{width:140px;height:auto;display:block;border-radius:6px;border:1px solid var(--line)}
.nophoto{width:140px;height:100px;display:flex;align-items:center;justify-content:center;border:1px dashed var(--line);
border-radius:6px;color:var(--muted);font-size:12px}
.id{font-weight:700}.tag{color:var(--muted);font-size:12px}.val{font-size:16px;font-weight:700}
.sub{color:var(--muted);font-size:12px}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid}
.badge.normal{color:var(--good);border-color:#9fd3ad;background:#eef8f0}
.badge.warning{color:var(--warn);border-color:#f0c58a;background:#fff6e8}
.badge.alarm{color:var(--crit);border-color:#efa3a3;background:#fdeeee}
.badge.unknown,.badge.none{color:var(--muted);border-color:var(--line);background:#f3f4f6}
.res-missed{color:var(--crit);font-weight:700}.res-no_value{color:var(--warn);font-weight:700}
.res-pending,.res-not_visited{color:var(--muted)}
.ok{color:var(--good);font-weight:700}.ng{color:var(--crit);font-weight:700}
.note{color:var(--muted);font-size:12px;margin-top:8px}
@media print{body{background:#fff}main{padding:0}.wrap,.kpi{border-color:#bbb}a{color:inherit;text-decoration:none}
tr{break-inside:avoid}}
"""


def render_html(report, session_dir):
    """Self-contained Korean HTML page of :func:`build_report` (photos embedded)."""
    m, gz, th, gas = report["mission"], report["gauges"], report["thermal"], report["gas"]
    rows = gz["rows"]
    st = m.get("state") or ""
    state_txt = STATE_KO.get(st, st or "—")
    final = report.get("final")
    out = ['<!doctype html><html lang="ko"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           f'<title>{_e(report["title"])} · {_e(report.get("session"))}</title><style>{_CSS}</style></head>',
           '<body><main>',
           f'<h1>{_e(report["title"])}</h1>',
           f'<p class="meta">플랜트 점검 로봇 · 세션 <b>{_e(report.get("session"))}</b> · 작성 {_e(report.get("written_wall"))}'
           f' · <span class="state{" final" if final else ""}">'
           f'{"임무 종료" if final else "진행 중 (중간 보고)"} · {_e(state_txt)}</span></p>']
    # ---- KPIs
    seek = gas.get("mission_seek") if isinstance(gas.get("mission_seek"), dict) else {}
    gas_res = seek.get("result") or gas.get("result")
    gas_err = num(seek.get("error_m")) if num(seek.get("error_m")) is not None else num(gas.get("source_error_m"))
    n_warn, n_alarm = len(gz["warnings"]), len(gz["alarms"])
    kpis = [
        ("계기 판독", f'{gz["read"]} / {gz["total"]}', f'판독 실패 {len(gz["missed"])}곳'),
        ("평균 판독 오차", _n(gz["mean_error_pct_fs"], 2, "% FS"), f'최대 {_n(gz["max_error_pct_fs"], 2, "% FS")}'),
        ("상태 판정 일치", f'{gz["status_correct"]} / {gz["status_checked"]}', "정상·경고·경보 판정"),
        ("이상 계기", f"{n_warn + n_alarm}개", f"경보 {n_alarm} · 경고 {n_warn}"),
        ("열화상 이상 검출", f'{len(th["detected"])} / {len(th["expected"])}곳',
         "재현율 " + (_n(100.0 * th["recall"], 0, "%") if num(th.get("recall")) is not None else "—")),
        ("가스 누출원", SEEK_KO.get(gas_res, gas_res or "추적 안 함"),
         "위치 오차 " + _n(gas_err, 2, "m") if gas_err is not None else f'최대 농도 {_n(gas.get("max_ppm"), 1, "ppm")}'),
        ("임무 소요", _dur(m.get("duration_s")), f'이동 {_n(m.get("distance_m"), 1, "m")}'),
        ("넘어짐 / 복구", f'{m.get("falls") or 0}회 / {m.get("recoveries") or 0}회',
         f'자동 복귀(RTH) {m.get("rth_count") or 0}회'),
    ]
    out.append('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div><div class="s">{_e(s)}</div></div>'
        for k, v, s in kpis) + "</div>")
    # ---- gauges
    out.append("<h2>계기 판독 결과</h2><div class=\"wrap\"><table><thead><tr><th>점검 사진</th><th>계기</th>"
               "<th>결과</th><th class=\"num\">판독값</th><th class=\"num\">참값</th><th class=\"num\">오차</th>"
               "<th>상태 판정</th><th>실제 상태</th><th class=\"num\">신뢰도</th><th>판독 시각</th></tr></thead><tbody>")
    for r in rows:
        nd = r["decimals"]
        unit = r["unit"]
        res = f'<span class="res-{_e(r["result"])}">{_e(RESULT_KO.get(r["result"], r["result"]))}</span>'
        if r["result"] == "read" and r["value_source"] != "mission":
            res += f'<div class="sub">{_e(SOURCE_KO.get(r["value_source"], r["value_source"]))}</div>'
        if r["result"] == "missed" and r.get("reason"):
            res += f'<div class="sub">{_e(r["reason"])}</div>'
        val = (f'<span class="val">{_n(r["value"], nd)}</span> {_e(unit)}' if r["value"] is not None else "—")
        if r.get("n_readings"):
            val += f'<div class="sub">판독 {_e(r["n_readings"])}회</div>'
        err = "—"
        if r["error_pct_fs"] is not None:
            err = f'{_n(r["error_pct_fs"], 2)} % FS<div class="sub">{r["error"]:+.{nd + 1}f} {_e(unit)}</div>'
        match = ""
        if r["status_correct"] is not None:
            match = '<div class="sub ok">일치</div>' if r["status_correct"] else '<div class="sub ng">불일치</div>'
        when = "—"
        if r["t_sim"] is not None or r.get("wall"):
            when = (f'{_n(r["t_sim"], 1)} s' if r["t_sim"] is not None else "") + \
                   (f'<div class="sub">{_e(r["wall"])}</div>' if r.get("wall") else "")
        out.append(
            f'<tr><td class="photo">{_image_cell(session_dir, r)}</td>'
            f'<td><div class="id">{_e(r["id"])}</div><div class="tag">{_e(r["tag"])}</div>'
            f'<div class="sub">범위 {_e(r["range"][0])}–{_e(r["range"][1])} {_e(unit)}</div></td>'
            f'<td>{res}</td><td class="num">{val}</td>'
            f'<td class="num">{_n(r["true_value"], nd)} {_e(unit) if r["true_value"] is not None else ""}</td>'
            f'<td class="num">{err}</td><td>{_badge(r["status"])}{match}</td><td>{_badge(r["true_status"])}</td>'
            f'<td class="num">{_n(r["confidence"], 2)}</td><td>{when}</td></tr>')
    out.append("</tbody></table></div>")
    out.append('<p class="note">오차 = |판독값 − 참값| ÷ 계기 전체 범위 × 100 (% FS). 판독값은 임무가 해당 계기 점검을 '
               "마칠 때 확정한 값(시간 융합 결과)이고, 사진은 그 점검에서 바늘이 확정값과 일치한 프레임입니다. "
               "사진을 누르면 전체 화면 사진이 열립니다(보고서와 같은 폴더의 gauges/ 필요).</p>")
    # ---- thermal
    out.append("<h2>열화상 점검</h2><div class=\"wrap\"><table><thead><tr><th>지점</th><th>종류</th>"
               "<th>검출</th><th class=\"num\">측정 최고 온도</th><th class=\"num\">실제 온도</th>"
               "<th class=\"num\">온도 오차</th><th>판정</th><th>실제 상태</th></tr></thead><tbody>")
    kind_ko = {"flange": "플랜지", "bearing": "베어링"}
    for h in th["rows"]:
        det = ('<span class="ok">검출</span>' if h["detected"] else
               ('<span class="ng">미검출</span>' if h["expected"] else '<span class="sub">해당 없음</span>'))
        out.append(
            f'<tr><td><div class="id">{_e(h["id"])}</div><div class="tag">{_e(h["tag"])}</div></td>'
            f'<td>{_e(kind_ko.get(h["kind"], h["kind"]))}</td><td>{det}</td>'
            f'<td class="num">{_n(h["peak_temp_c"], 1, "°C")}</td><td class="num">{_n(h["true_temp_c"], 1, "°C")}</td>'
            f'<td class="num">{_n(h["temp_error_c"], 1, "°C") if h["temp_error_c"] is not None else "—"}</td>'
            f'<td>{_badge(h["status"])}</td><td>{_badge(h["true_status"])}</td></tr>')
    out.append("</tbody></table></div>")
    out.append(f'<p class="note">검출 {th.get("tp") or 0}건 · 오검출 {th.get("fp") or 0}건 · 정밀도 '
               f'{_n(100.0 * th["precision"], 0, "%") if num(th.get("precision")) is not None else "—"}. '
               "경고 기준 이상인 지점만 검출 대상입니다.</p>")
    # ---- gas
    est = seek.get("estimate") if isinstance(seek.get("estimate"), (list, tuple)) else gas.get("estimate")
    truth = gas.get("truth")

    def xy(p):
        return (f"({_n(p[0], 2)}, {_n(p[1], 2)})" if isinstance(p, (list, tuple)) and len(p) >= 2
                and num(p[0]) is not None and num(p[1]) is not None else "—")
    gas_rows = [("최대 농도", _n(gas.get("max_ppm"), 1, "ppm")),
                ("첫 경보 시각", _n(gas.get("first_alarm_t"), 1, "s")),
                ("누출원 추적 결과", SEEK_KO.get(gas_res, gas_res or "추적 안 함")),
                ("추정 위치", xy(est)), ("실제 누출원", xy(truth)), ("위치 오차", _n(gas_err, 2, "m")),
                ("추적 소요", _n(seek.get("elapsed") if num(seek.get("elapsed")) is not None
                                 else gas.get("time_to_find_s"), 1, "s"))]
    out.append("<h2>가스 누출 탐지</h2><div class=\"wrap\"><table style=\"min-width:0\"><tbody>" + "".join(
        f"<tr><th>{_e(k)}</th><td class=\"num\">{_e(v)}</td></tr>" for k, v in gas_rows) + "</tbody></table></div>")
    # ---- mission
    batt = m.get("battery") or {}
    mrows = [("임무 상태", state_txt), ("시작 (시뮬레이션 시각)", _n(m.get("start_t"), 1, "s")),
             ("종료 (시뮬레이션 시각)", _n(m.get("end_t"), 1, "s")), ("소요 시간", _dur(m.get("duration_s"))),
             ("이동 거리", _n(m.get("distance_m"), 1, "m")),
             ("점검 완료", ", ".join(map(str, m.get("completed") or [])) or "—"),
             ("판독 실패", ", ".join(map(str, m.get("missed") or [])) or "없음"),
             ("넘어짐 / 스스로 일어남", f'{m.get("falls") or 0}회 / {m.get("recoveries") or 0}회'),
             ("자동 복귀(RTH)", f'{m.get("rth_count") or 0}회'),
             ("배터리 (시작 → 종료)", f'{_n(batt.get("start_pct"), 0, "%")} → {_n(batt.get("end_pct"), 0, "%")}')]
    out.append("<h2>임무 요약</h2><div class=\"wrap\"><table style=\"min-width:0\"><tbody>" + "".join(
        f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in mrows) + "</tbody></table></div>")
    out.append('<p class="note">같은 폴더의 inspection_report.json에 같은 내용이 기계 판독용으로, events.csv·'
               "gauge_readings.csv·summary.json에 전체 기록이 있습니다.</p></main></body></html>")
    return "\n".join(out)
