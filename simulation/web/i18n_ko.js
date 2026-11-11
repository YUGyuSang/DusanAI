/*
 * Korean labels of the control-room client (loaded before app.js).
 *
 * The ROS nodes publish English state values and English event texts; the
 * browser shows Korean:
 *   KO.label(kind, value, fallback)  state / status / reason ... value -> Korean label
 *                                    (unknown values are returned unchanged)
 *   KO.event(text, source)           event-log / mission message -> Korean, regex rules
 *                                    with captured values; unknown messages stay as they are
 *   KO.command(target, command)      operator command -> button label
 * The raw English value stays available to the page (title tooltips).
 * Works in the browser (window.KO) and in node (module.exports, used by the tests).
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------ value labels
  const LABELS = {
    // /mission/state, /safety/status and /locomotion/state modes share one table
    state: {
      IDLE: "대기", PATROL: "순찰", INSPECTING: "점검 중", SEEK: "가스 추적", RTH: "복귀 중",
      DOCKED: "충전소 도착", DONE: "완료", MANUAL: "수동 조작", PAUSED: "일시정지",
      MONITOR: "감시", RTH_PLANNING: "복귀 경로 계획", RTH_NAVIGATING: "복귀 이동 중", DOCKING: "도킹 중",
      CHARGING: "충전 중",
      WALK: "보행", STAND: "서 있음", SIT: "앉음", STANDING_UP: "일어서는 중", FALLEN: "넘어짐",
      RECOVERING: "복구 중", ESTOP: "비상정지",
    },
    // gauge / hot-spot status
    status: { normal: "정상", warning: "경고", alarm: "경보", unread: "미판독", unseen: "미관측", unknown: "알 수 없음" },
    // topic link health (dashboard_state.topic_health)
    link: { ok: "정상", late: "지연", stale: "끊김", never: "수신 없음" },
    // /gas/seek_status state + mission seek results
    seek: {
      IDLE: "대기", SEARCH: "탐색", SURGE: "바람 거슬러 전진", CAST: "좌우 탐색", GRADIENT: "농도 경사 추종",
      FOUND: "발견", FAILED: "실패", TIMEOUT: "시간 초과",
    },
    // mission phase / mode
    phase: {
      starting: "출발 준비", navigate: "이동", align: "방향 맞춤", hold: "정지 대기", backup: "후진",
      dwell: "판독 대기", scan_left: "왼쪽 살피기", scan_right: "오른쪽 살피기", seeking: "추적 중",
      yield: "복귀 노드 주행", requested: "복귀 요청", self: "자체 복귀", self_wait: "자체 복귀 대기",
    },
    mode: { full: "전체 임무", patrol: "순찰 임무" },
    // return-to-home reasons (module3 rth_logic.REASONS + mission reasons)
    reason: {
      manual: "수동 요청", battery_low: "배터리 부족", energy_reserve: "복귀 에너지 부족",
      comm_shadow_predicted: "통신 음영 예상", comm_lost: "통신 두절", mission_complete: "임무 완료",
      safety: "안전 계층", active: "작동 중",
    },
    // cmd_vel mux source
    source: { teleop: "원격조종", rth: "충전소 복귀", seek: "가스 추적", patrol: "순찰", none: "없음" },
    gait: { trot: "트롯", crawl: "크롤", walk: "걷기", auto: "자동" },
    algo: { stable: "안정 모드", mpc: "MPC" },
    terrain: {
      flat: "평지", rough: "거친 지면", stairs: "계단", slope: "경사로", obstacle: "장애물",
      unknown: "알 수 없음", slippery: "미끄러움",
    },
    slip: { terrain: "지형 사전정보", detected: "미끄럼 감지", both: "지형 사전정보 + 감지", none: "없음" },
    mpc: {
      solved: "정상", solved_inaccurate: "근사해", max_iter_reached: "반복 한도", time_limit_reached: "시간 한도",
      primal_infeasible: "해 없음", dual_infeasible: "해 없음", invalid_problem: "잘못된 문제",
    },
    scale: { linear: "선형", log: "로그" },
    kind: { flange: "플랜지", bearing: "베어링", valve: "밸브", pipe: "배관" },
    // zone names of simulation/config/world_layout.yaml
    zone: {
      "Corridor lower level": "통로 하층", "Stairs and ramp": "계단·경사로", "Corridor upper level": "통로 상층",
      "Turbine hall": "터빈 홀", "High temperature zone": "고온 구역", "RF shadow (turbine pit)": "전파 음영 (터빈 피트)",
      "turbine pit (shielded)": "터빈 피트 (차폐)",
    },
    // event-log source column
    event_source: {
      dashboard: "대시보드", mission: "임무", locomotion: "보행", fall: "넘어짐", teleop: "원격조종", safety: "안전",
      patrol: "순찰 계획", battery: "배터리", gas: "가스", gauge: "계기", thermal: "열화상", operator: "운영자",
      demo: "데모", metrics: "기록",
    },
    target: { mission: "임무", locomotion: "보행", battery: "배터리" },
    command: {
      start: "임무 시작", patrol: "순찰", seek: "가스 추적", rth: "충전소 복귀", dock: "충전소 복귀",
      pause: "일시정지", resume: "재개", abort: "임무 중단",
      stand: "서기", sit: "앉기", walk: "걷기", recover: "복구", estop: "비상정지", release: "비상정지 해제",
      reset_pose: "자세 리셋", push: "밀기", "gait:trot": "트롯 보행", "gait:crawl": "크롤 보행", "gait:auto": "보행 자동 선택",
    },
    fall: {
      fall: "넘어짐", recovered: "복구 완료", recovery_retry: "복구 재시도", recovery_failed: "복구 실패",
      assisted_reset: "보조 리셋", reset_request: "리셋 요청",
    },
    side: { left: "왼쪽", right: "오른쪽" },
    // module1 fall_recovery.py phases (/locomotion/state recovery_state)
    recovery: {
      NORMAL: "정상", FALL_DETECTED: "넘어짐 감지", DAMP: "충격 완화", TUCK: "다리 접기", ROCK: "몸 흔들어 뒤집기",
      SLIDE: "몸 돌려 장애물 벗어나기", BACK_OFF: "턱에서 내려오기",
      SELF_RIGHT: "몸 뒤집기", PREPARE: "일어설 준비", STAND_UP: "일어서기", VERIFY: "자세 확인",
      FAILED: "자가 복구 실패", ASSISTED_RESET: "보조 리셋 (위치 재설정)",
    },
  };

  const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  const blank = (v) => v === undefined || v === null || v === "";

  function label(kind, value, fallback) {
    if (blank(value)) return fallback === undefined ? "" : fallback;
    const map = LABELS[kind];
    const k = String(value);
    return map && has(map, k) ? map[k] : k;
  }

  function command(target, cmd) {
    const c = String(cmd === undefined || cmd === null ? "" : cmd);
    if (target === "battery") return "배터리 " + c + " % 설정";
    if (/^push:/i.test(c)) return "밀기 " + c.slice(5);
    return label("command", c.toLowerCase(), c);
  }

  // ------------------------------------------------------------ helpers
  const st = (v) => (v === "-" ? "-" : label("state", v));
  const arrow = (a, b) => (a === "-" || blank(a) ? b : a + " → " + b);
  // python list repr "['G01', 'G02']" (or "G01, G02") -> "G01, G02"
  const list = (s) => String(s).replace(/[[\]']/g, "");
  const na = (v) => (v === "n/a" || v === "None" ? "없음" : v);
  const readCount = (n, m) => "(판독 " + n + ", 누락 " + m + ")";
  // "(-3.53, 1.9)" -> "-3.53, 1.90"
  const xy = (a, b) => [a, b].map((v) => (isFinite(+v) ? (+v).toFixed(2) : v)).join(", ");

  function slipParts(s) {
    return String(s)
      .replace(/terrain prior \+ slip detected/g, "지형 사전정보 + 미끄럼 감지")
      .replace(/terrain prior/g, "지형 사전정보")
      .replace(/slip detected/g, "미끄럼 감지")
      .replace(/prior mu /g, "사전 μ ")
      .replace(/MPC mu /g, "MPC μ ")
      .replace(/mu est /g, "μ 추정 ")
      .replace(/detection #/g, "감지 #")
      .replace(/feet /g, "발 ");
  }

  // short reason phrases inside a message (skip / failure / seek reasons)
  function why(text) {
    const s = String(text);
    let m;
    if (s === "no path") return "경로 없음";
    if (s === "no reading") return "판독 없음";
    if (s === "manual request") return "수동 요청";
    if ((m = /^stuck \((\d+) re-plans\)$/.exec(s))) return "정체 (재계획 " + m[1] + "회)";
    if ((m = /^gas (\S+) ppm$/.exec(s))) return "가스 " + m[1] + " ppm";
    if (has(LABELS.reason, s)) return LABELS.reason[s];
    return event(s);
  }

  // ------------------------------------------------------------ event rules
  // [regex, (match) -> Korean, optional source]; first match wins, order = specific first.
  // Texts: plant_simulation/dashboard_state.py (add_event), mission_logic.py (_emit),
  // dusan_common/obstacle_layer.py (detour / re-plan messages), web_server.py /
  // monitor_dashboard_node.py (command responses).
  const RULES = [
    // ---- aggregator (dashboard_state.py)
    [/^digital twin dashboard started$/, () => "디지털 트윈 대시보드 시작"],
    [/^state (\S+) -> (\S+)$/, (m) => "임무 상태: " + arrow(st(m[1]), st(m[2]))],
    [/^mode (\S+) -> (\S+)$/, (m) => "보행 모드: " + arrow(st(m[1]), st(m[2]))],
    [/^gait (\S+) -> (\S+)$/, (m) => "보행 방식: " + arrow(label("gait", m[1]), label("gait", m[2]))],
    [/^locomotion mode (\S+) -> (\S+)$/, (m) => "제어 방식: " + arrow(label("algo", m[1]), label("algo", m[2]))],
    [/^slip detected #(\d+)(?:: (.+))?$/, (m) => "미끄럼 감지 #" + m[1] + (m[2] ? ": " + slipParts(m[2]) : "")],
    // short form (the event column is narrow): source + prior friction only, details in the tooltip
    [/^slip mode ON(?: \((.+)\))?$/, (m) => {
      const d = m[1] || "";
      const mu = /prior mu ([\d.]+)/.exec(d);
      const parts = ["미끄럼 대응 켜짐"];
      if (/terrain prior/.test(d)) parts.push("저마찰 바닥" + (mu ? " μ " + mu[1] : ""));
      if (/slip detected/.test(d)) parts.push("미끄럼 감지");
      if (parts.length === 1 && d) parts.push(slipParts(d));
      return parts.join(" · ");
    }],
    [/^slip mode OFF(?: \((\d+) detections? so far\))?$/,
      (m) => "미끄럼 대응 꺼짐" + (m[1] && m[1] !== "0" ? " (지금까지 감지 " + m[1] + "회)" : "")],
    [/^(fall|recovered|recovery_retry|recovery_failed|assisted_reset|reset_request)(?: at \((\S+), (\S+)\))?(?: roll (-?\d+) deg pitch (-?\d+) deg)?$/,
      (m) => label("fall", m[1]) + (m[2] ? " 위치 (" + m[2] + ", " + m[3] + ")" : "") +
        (m[4] ? " 롤 " + m[4] + "° 피치 " + m[5] + "°" : "")],
    [/^manual override ON$/, () => "수동 조작 시작"],
    [/^manual override OFF$/, () => "수동 조작 종료"],
    [/^return-to-home ACTIVE$/, () => "충전소 복귀 작동"],
    [/^return-to-home released$/, () => "충전소 복귀 해제"],
    [/^(\S+) -> (\S+)(?: \((.+)\))?$/,
      (m) => "안전 상태: " + arrow(st(m[1]), st(m[2])) + (m[3] ? " (" + label("reason", m[3]) + ")" : ""), "safety"],
    [/^patrol plan: (\d+) viewpoints, order (.*?)(?:, ([\d.]+) m)?( \(applies to the next mission\))?$/,
      (m) => "순찰 계획: 점검 지점 " + m[1] + "곳, 순서 " + list(m[2]) + (m[3] ? ", 총 " + m[3] + " m" : "") +
        (m[4] ? " (다음 임무부터 적용)" : "")],
    [/^battery LOW: (\S+) %$/, (m) => "배터리 부족: " + m[1] + " %"],
    [/^battery recovered: (\S+) %$/, (m) => "배터리 회복: " + m[1] + " %"],
    [/^gas (\S+) ppm above (\S+) ppm$/, (m) => "가스 " + m[1] + " ppm - 경보 기준 " + m[2] + " ppm 초과"],
    [/^gas back to (\S+) ppm$/, (m) => "가스 정상 복귀: " + m[1] + " ppm"],
    // the aggregator's error comes from the 2-decimal estimate: the page shows the
    // mission's error (mission.seek.error_m) instead, the raw text stays in the tooltip
    [/^LEAK SOURCE FOUND(?: at \((\S+), (\S+)\))?(?: - error (\S+) m)?$/,
      (m) => "누출원 발견" + (m[1] ? " (" + xy(m[1], m[2]) + ")" : "")],
    [/^source seeking FAILED$/, () => "누출원 추적 실패"],
    [/^seeker (\S+) -> (\S+)$/, (m) => "추적기: " + arrow(label("seek", m[1]), label("seek", m[2]))],
    // emitted when a gauge's status changes (normally its first reading): no number here,
    // the tile shows the latest of many readings and the raw text stays in the tooltip
    [/^(\S+) (.*?): (\S+) (\S*) \((\w+)\)(?:, error (\S+) % FS)?$/,
      (m) => m[1] + (m[2] ? " " + m[2] : "") + (na(m[3]) === m[3] ? " 판독 완료 · " : " 판독값 없음 · ") +
        label("status", m[5]), "gauge"],
    // status change of a hot spot: no temperature here either (the thermal table shows the
    // latest measurement, the first one stays in the tooltip)
    [/^(\S+) (.*?): (\S+) C \((\w+)\)$/,
      (m) => m[1] + (m[2] ? " " + m[2] : "") + (na(m[3]) === m[3] ? " 고온 검출 · " : " 온도 없음 · ") +
        label("status", m[4]), "thermal"],
    [/^THERMAL ALARM (\S+) (.*?): (\S+) C(?: at (.+))?$/,
      (m) => "열화상 경보 " + m[1] + (m[2] ? " " + m[2] : "") + (m[4] ? " · 위치 " + m[4] : "")],
    [/^(mission|locomotion|battery) command '(.*)'$/,
      (m) => "운영자 명령: " + (m[1] === "battery" ? command("battery", m[2]) : "'" + command(m[1], m[2]) + "'")],
    [/^locomotion '(.*)' \(not simulated in the demo\)$/,
      (m) => "보행 명령 '" + command("locomotion", m[1]) + "' (데모에서는 시뮬레이션하지 않음)"],

    // ---- mission manager (mission_logic.py)
    [/^ready$/, () => "준비 완료"],
    [/^patrol plan received but it contains no usable viewpoints$/, () => "순찰 계획을 받았지만 사용할 수 있는 점검 지점이 없음"],
    [/^manual override \(teleop active\) - mission frozen$/, () => "수동 조작 중 (원격조종 활성) - 임무 정지"],
    [/^manual override released$/, () => "수동 조작 해제"],
    [/^manual override expired: no \/teleop\/active for (\S+) s \(teleop gone\?\)$/,
      (m) => "수동 조작 만료: " + m[1] + " s 동안 /teleop/active 없음 (원격조종 종료?)"],
    [/^mission paused$/, () => "임무 일시정지"],
    [/^nothing to pause in (\S+)$/, (m) => st(m[1]) + " 상태에서는 일시정지할 수 없음"],
    [/^mission resumed$/, () => "임무 재개"],
    [/^resume requested - waiting for return_to_home to release$/, () => "재개 요청 - 복귀 노드(return_to_home) 해제 대기"],
    [/^resuming interrupted mission \((\d+) viewpoints left\)$/, (m) => "중단된 임무 재개 (남은 점검 지점 " + m[1] + "곳)"],
    [/^nothing to resume$/, () => "재개할 임무 없음"],
    [/^'([^']*)' ignored: mission already running \((\S+)\); abort first$/,
      (m) => "'" + command("mission", m[1]) + "' 무시: 임무가 이미 진행 중 (" + st(m[2]) + ") - 먼저 중단하세요"],
    [/^'seek' ignored while return_to_home is active$/, () => "'가스 추적' 무시: 충전소 복귀 진행 중"],
    [/^'seek' ignored in (\S+)$/, (m) => "'가스 추적' 무시: " + st(m[1]) + " 상태"],
    [/^autostart: starting the full mission$/, () => "자동 시작: 전체 임무 시작"],
    // a start before the patrol planner has published its plan waits for it (mission_logic start_plan_wait)
    [/^(full|patrol) accepted - waiting up to (\S+) s for the patrol plan \(patrol_planner still computing\)$/,
      (m) => (m[1] === "full" ? "전체 임무" : "순찰") + " 접수 - 순찰 계획 대기 (최대 " + m[2] + "초, 경로 계산 중)"],
    [/^start waiting for the patrol plan cancelled by '(.*)'$/,
      (m) => "순찰 계획 대기 중이던 시작 취소 ('" + command("mission", m[1]) + "')"],
    [/^GAS ALARM: (\S+) ppm > (\S+) ppm$/, (m) => "가스 경보: " + m[1] + " ppm > " + m[2] + " ppm"],
    [/^gas alarm re-armed$/, () => "가스 경보 감시 재개"],
    [/^return_to_home released - (start|resume) the mission$/,
      (m) => "복귀 노드 해제 - 임무 " + (m[1] === "start" ? "시작" : "재개")],
    [/^return_to_home released - resuming the mission$/, () => "복귀 노드 해제 - 임무 재개"],
    [/^return_to_home released \(not docked\)$/, () => "복귀 노드 해제 (충전소 미도착)"],
    [/^return to home \((.+?)\)( - waiting for return_to_home)?$/,
      (m) => "충전소 복귀 시작 (" + label("reason", m[1]) + ")" +
        (m[2] && m[1] !== "mission_complete" ? " - 복귀 노드 응답 대기" : "")],
    [/^self RTH impossible: no layout$/, () => "자체 복귀 불가: 레이아웃 없음"],
    [/^self RTH: waiting for \/odom$/, () => "자체 복귀: /odom 대기"],
    [/^self RTH: no path to the dock$/, () => "자체 복귀: 충전소까지 경로 없음"],
    [/^self RTH failed: (.+)$/, (m) => "자체 복귀 실패: " + why(m[1])],
    [/^return_to_home did not acknowledge - driving to the dock myself$/, () => "충전소 복귀: 자체 경로로 이동"],
    [/^return_to_home did not acknowledge the RTH request$/, () => "복귀 노드가 복귀 요청에 응답하지 않음"],
    [/^mission complete - docked \((\d+) read, (\d+) missed\)$/, (m) => "임무 완료 - 충전소 도착 " + readCount(m[1], m[2])],
    [/^docked(?: \((\d+) viewpoints pending, 'resume' continues\))?$/,
      (m) => "충전소 도착" + (m[1] ? " (남은 점검 지점 " + m[1] + "곳, '재개'로 계속)" : "")],
    [/^robot left the dock$/, () => "로봇이 충전소를 떠남"],
    [/^mission aborted$/, () => "임무 중단"],
    [/^mission aborted after an internal error - press 't' to start again$/, () => "내부 오류로 임무 중단 - 't' 키로 다시 시작"],
    [/^no \/patrol\/plan_json received - using layout fallback viewpoints (.*)$/,
      (m) => "/patrol/plan_json 미수신 - 레이아웃 기본 점검 지점 사용: " + list(m[1])],
    [/^cannot (full|patrol): no patrol plan \/ viewpoints available$/,
      (m) => (m[1] === "full" ? "전체 임무" : "순찰") + " 불가: 순찰 계획 / 점검 지점 없음"],
    [/^(full mission|patrol) started: (.*)$/, (m) => (m[1] === "patrol" ? "순찰" : "전체 임무") + " 시작: " + list(m[2])],
    [/^docked with (\S+) % battery - releasing return_to_home$/, (m) => "배터리 " + m[1] + " %로 도킹 상태 - 복귀 노드 해제"],
    [/^waiting for return_to_home to release before starting(?: \(battery (\S+) %\))?$/,
      (m) => "시작 전 복귀 노드 해제 대기" + (m[1] ? " (배터리 " + m[1] + " %)" : "")],
    [/^next viewpoint (\S+) at \((\S+), (\S+)\)$/, (m) => "다음 점검 지점 " + m[1] + " (" + m[2] + ", " + m[3] + ")"],
    [/^waiting for \/odom$/, () => "/odom 대기 중"],
    [/^patrol leg blocked by a detected obstacle - planning around it$/, () => "순찰 구간이 감지된 장애물에 막힘 - 우회 경로 계획"],
    [/^no \/map yet - driving straight lines$/, () => "/map 없음 - 직선으로 주행"],
    [/^no path around the detected obstacle\(s\)( - keeping the map route \(stuck logic decides\))?$/,
      (m) => "감지된 장애물을 피할 경로 없음" + (m[1] ? " - 지도 경로 유지 (정체 판단에 맡김)" : "")],
    [/^route past the near side of the detected obstacle\(s\) \((\S+) vs (\S+) around the assumed back\)$/,
      (m) => "감지된 장애물의 가까운 쪽으로 통과 (비용 " + m[1] + ", 뒤쪽 우회 시 " + m[2] + ")"],
    [/^viewpoint (\S+) skipped: (.+)$/, (m) => "점검 지점 " + m[1] + " 건너뜀: " + why(m[2])],
    [/^all viewpoints done \((\d+) read, (\d+) missed\) - returning home$/,
      (m) => "모든 점검 지점 완료 " + readCount(m[1], m[2]) + " - 충전소로 복귀"],
    [/^patrol complete \((\d+) read, (\d+) missed\)$/, (m) => "순찰 완료 " + readCount(m[1], m[2])],
    [/^detected obstacle behind - re-planning without backing up$/, () => "뒤쪽에 감지된 장애물 - 후진 없이 재계획"],
    [/^locomotion ready again \((\S+)\) - re-planning$/, (m) => "보행 준비 완료 (" + st(m[1]) + ") - 재계획"],
    [/^locomotion not ready \((\S+)\) - holding$/, (m) => "보행 준비 안 됨 (" + st(m[1]) + ") - 제자리 대기"],
    [/^robot still down after (\S+) s \(recovery (.+?)\) - sending 'recover'$/,
      (m) => "로봇이 " + m[1] + " s째 넘어져 있음 (복구 단계: " + label("recovery", m[2]) + ") - '복구' 명령 전송"],
    [/^obstacle avoidance failed \((\d+)x\): (.+)$/, (m) => "장애물 회피 오류 (" + m[1] + "회): " + m[2]],
    [/^obstacle at (\S+) m - detour (left|right) \(\+(\S+) m\)$/,
      (m) => m[1] + " m 앞 장애물 - " + label("side", m[2]) + "으로 우회 (+" + m[3] + " m)"],
    [/^path blocked by a detected obstacle at (\S+) m - re-planning(?: \((\d+)\/(\d+)\))?$/,
      (m) => m[1] + " m 앞 감지된 장애물로 경로 막힘 - 재계획" + (m[2] ? " (" + m[2] + "/" + m[3] + ")" : "")],
    [/^stuck at \((\S+), (\S+)\) - backing up and re-planning \((\d+)\/(\d+)\)$/,
      (m) => "(" + m[1] + ", " + m[2] + ")에서 정체 - 후진 후 재계획 (" + m[3] + "/" + m[4] + ")"],
    [/^off the path - re-planning$/, () => "경로 이탈 - 재계획"],
    [/^(\S+): alignment timeout \(yaw error (\S+) deg\)$/, (m) => m[1] + ": 방향 맞춤 시간 초과 (방향 오차 " + m[2] + "°)"],
    [/^inspecting (\S+)$/, (m) => m[1] + " 점검 중"],
    [/^(\S+): no reading yet - scanning left\/right$/, (m) => m[1] + ": 아직 판독 없음 - 좌우로 살피는 중"],
    // no value here either (the tile shows the latest reading)
    [/^(\S+) read: (?:(\S+) (\S*) \((\w*)\)|ok) from (\d+) readings$/,
      (m) => m[1] + " 판독 완료" + (m[4] ? " · " + label("status", m[4]) : "") + " (" + m[5] + "회 판독)"],
    [/^(\S+): no reading obtained - skipped$/, (m) => m[1] + ": 판독 실패 - 건너뜀"],
    [/^gas source seeking started \((.+)\)$/, (m) => "가스 누출원 추적 시작 (" + why(m[1]) + ")"],
    [/^source seeker did not become active$/, () => "누출원 추적기가 동작하지 않음"],
    [/^GAS SOURCE FOUND(?: at \((\S+), (\S+)\))?(?:, error (\S+) m)?$/,
      (m) => "가스 누출원 발견" + (m[1] ? " (" + m[1] + ", " + m[2] + ")" : "") + (m[3] ? ", 오차 " + m[3] + " m" : "")],
    [/^gas source seeking ended: (\S+)$/, (m) => "가스 누출원 추적 종료: " + label("seek", m[1])],
    // inspection report of the metrics session (dashboard_state.set_session)
    [/^inspection report final: (\S+)$/, () => "점검 보고서 작성 완료 (임무 종료)"],
    [/^inspection report written: (\S+)$/, () => "점검 보고서 작성 (중간 보고)"],
    [/^session directory (.+) ignored \(not inside (.+)\)$/, (m) => "기록 폴더 무시: " + m[1] + " (" + m[2] + " 밖)"],

    // ---- command responses (web_server.py, monitor_dashboard_node.py, dashboard_demo.py)
    [/^(mission|locomotion|battery) '(.*)' sent$/,
      (m) => (m[1] === "battery" ? command("battery", m[2]) : "'" + command(m[1], m[2]) + "' 명령") + " 전송됨"],
    [/^unknown (mission|locomotion) command '(.*)'$/, (m) => "알 수 없는 " + label("target", m[1]) + " 명령 '" + m[2] + "'"],
    [/^battery command must be a percentage$/, () => "배터리 명령은 백분율이어야 함"],
    [/^battery percentage must be within 0\.\.100$/, () => "배터리 값은 0 ~ 100 % 범위여야 함"],
    [/^unknown target '?([^' ]*)'?(?: \(mission\|locomotion\|battery\))?$/, (m) => "알 수 없는 명령 대상 '" + m[1] + "'"],
    [/^Content-Type must be application\/json$/, () => "Content-Type은 application/json이어야 함"],
    [/^missing or oversized body$/, () => "요청 본문이 없거나 너무 큼"],
    [/^body must be JSON$/, () => "요청 본문은 JSON이어야 함"],
    [/^body must be a JSON object$/, () => "요청 본문은 JSON 객체여야 함"],
    [/^not found$/, () => "찾을 수 없음"],
    [/^commands disabled$/, () => "명령 기능 꺼짐"],
    [/^command failed: (.*)$/, (m) => "명령 실패: " + m[1]],
  ];

  function event(text, source) {
    const s = blank(text) ? "" : String(text);
    for (let i = 0; i < RULES.length; i++) {
      const r = RULES[i];
      if (r[2] && r[2] !== source) continue;      // source-specific rule (generic text shape)
      const m = r[0].exec(s);
      if (!m) continue;
      try { return r[1](m); } catch (err) { return s; }
    }
    return s;
  }

  const KO = { label: label, event: event, command: command, labels: LABELS };
  root.KO = KO;
  if (typeof module !== "undefined" && module.exports) module.exports = KO;
})(typeof window !== "undefined" ? window : globalThis);
