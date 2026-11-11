# module3_gas_safety — 가스 누출, 후각 기반 누출원 탐색, 배터리 / 통신 / 자율 복귀

디지털 트윈 4족 로봇 프로젝트의 ROS 2 패키지(ament_python)입니다. 터빈 홀의 가스 누출을 환기
기류와 함께 시뮬레이션하고, 경사 하강 기반 후각 탐색기로 누출원을 찾으며, 로봇의 안전(배터리 충전
상태, Wi-Fi 커버리지, 자율 복귀와 도킹)을 감독합니다.

모든 알고리즘은 ROS에 의존하지 않는 모듈(numpy, OpenCV는 선택)에 들어 있고 Windows에서 단위
테스트합니다. 노드 5개는 얇은 rclpy 래퍼입니다.

| 노드 (실행 파일) | 모듈 | 알고리즘 모듈 | 설정 |
|---|---|---|---|
| `gas_dispersion_sim` | `gas_dispersion_node.py` | `gas_model.py` | `config/gas_dispersion.yaml` |
| `source_seeker` | `source_seeker_node.py` | `source_seeking.py` | `config/seeker.yaml` |
| `battery_simulator` | `battery_node.py` | `battery_model.py` | `config/safety.yaml` |
| `comm_simulator` | `comm_node.py` | `comm_model.py` | `config/safety.yaml` |
| `return_to_home` | `return_to_home_node.py` | `rth_logic.py` | `config/safety.yaml` |

보조 모듈: `layout_map.py`(내비 맵 로드, `plant_map.yaml`이 없을 때 `world_layout.yaml`을
래스터화), `params.py`(모든 ROS 파라미터와 기본값, dict → config 변환), `node_common.py`(rclpy
연결 코드, 지연 import), `sim_harness.py`(테스트와 `tools/eval_gas.py`가 쓰는 폐루프 오프라인
시뮬레이션).

---------------------------------------------------------------------------------------

## 1. 실행 방법

```bash
# Ubuntu PC의 colcon 워크스페이스 안에서
colcon build --symlink-install --packages-select dusan_common module3_gas_safety
source install/setup.bash

# 노드 5개 전체 (보통 plant_simulation/bringup.launch.py가 시작함)
ros2 launch module3_gas_safety gas_safety.launch.py                 # use_sim_time:=true
ros2 launch module3_gas_safety gas_safety.launch.py use_sim_time:=false map_yaml:=/path/plant_map.yaml

# YAML과 함께 단일 노드 실행
ros2 run module3_gas_safety gas_dispersion_sim --ros-args \
    --params-file install/module3_gas_safety/share/module3_gas_safety/config/gas_dispersion.yaml

# 조작
ros2 topic pub --once /gas/seek/enable std_msgs/msg/Bool "{data: true}"      # 탐색 시작
ros2 topic pub --once /battery/set_level std_msgs/msg/Float32 "{data: 19.0}" # 배터리 테스트 -> RTH
ros2 topic pub --once /mission/command std_msgs/msg/String "{data: rth}"     # 수동 RTH
ros2 topic pub --once /mission/command std_msgs/msg/String "{data: resume}"  # 충전기에서 출발 / 계속
ros2 topic pub --once /mission/command std_msgs/msg/String "{data: pause}"   # 복귀 일시정지
ros2 topic pub --once /mission/command std_msgs/msg/String "{data: abort}"   # 복귀 취소
ros2 topic echo /gas/seek_status
ros2 topic echo /safety/status
```

Launch 인자: `use_sim_time`(`true`), `map_yaml`(`''` → `world_layout.yaml`의 `nav_map.yaml`, 즉
`plant_simulation/maps/plant_map.yaml`), `layout_file`(`''` →
`plant_simulation/config/world_layout.yaml`). 생성된 맵이 없으면 모든 노드가 레이아웃에서 같은
맵을 래스터화합니다(로그에 경고). `source_seeker`와 `return_to_home`은 `map_server_lite`가
`/map`을 발행하면 자기 맵을 그것으로 바꿉니다.

오프라인 (Windows, ROS 없음):

```bash
python -m pytest module3_gas_safety -q          # 단위 + 폐루프 + 스텁 노드 테스트
python tools/eval_gas.py                         # 그림 -> docs/images, 수치 -> docs/results/gas_eval.json
```

---------------------------------------------------------------------------------------

## 2. 토픽 (DESIGN_SPEC §4)

| 토픽 | 타입 | 방향 | 비고 |
|---|---|---|---|
| `/gas/concentration_grid` | nav_msgs/OccupancyGrid | pub (latched, 2 Hz) | 내비 맵 범위 전체의 0.25 m 격자. **로그 스케일** `v = 100·log10(1+c)/log10(1+vis_max_ppm)` (c < `vis_min_ppm` → 0), 고체 셀 −1. 역변환: `ppm = (1+vis_max_ppm)^(v/100) − 1` |
| `/gas/field_info` | String JSON | pub 2 Hz | `{max_ppm, max_xy, vis_max_ppm, scale: log, scale_formula, source:[x,y], emission_rate, wind:[u,v], wind_base, gust_speed_factor, gust_angle_deg, t, step_ms, substeps, resolution, mass}` |
| `/gas/sensor` | std_msgs/Float32 | pub 10 Hz | 코 위치(base_link (0.42, 0))의 ppm, 1차 지연 τ = 1 s + 노이즈 |
| `/gas/sensor_array` | std_msgs/Float32MultiArray | pub 10 Hz | `[front_left, front_right, rear_left, rear_right]`, 위치 (±0.30, ±0.15) |
| `/gas/wind` | geometry_msgs/Vector3Stamped | pub 10 Hz | 풍속계, **월드 좌표계** (`frame_id map`), 지연 0.5 s + 노이즈 |
| `/gas/seek/enable` | std_msgs/Bool | sub | latched와 volatile 발행자 모두 동작. `false` → `true`가 `resume_window_s`(45 s) 안에 오면 끝나지 않은 실행을 이어 감(미션 일시정지 / teleop 개입: 플룸 기억, 경로, 추정값을 유지하고, 일시정지 시간은 `timeout`에 포함되지 않음). 그 밖의 경우나 FOUND / FAILED 뒤에는 새 실행을 시작 |
| `/cmd_vel/seek` | geometry_msgs/Twist | pub 10 Hz | 몸체 속도. 활성화되어 있고, 아직 탐색 중이며(FOUND / FAILED 아님), `/safety/rth_active`가 false일 때만 발행. 명령을 멈출 때 0 Twist를 한 번 보내고 그 뒤로는 침묵(mux가 순찰로 돌아감) |
| `/gas/seek_status` | String JSON | pub 2 Hz + 변경 시 | `{state, concentration, gradient:[gx,gy] (ppm/m), gradient_log, gradient_snr, estimate:[x,y]\|null, error_m\|null, path_length, elapsed, phase, step, max_ppm, n_samples, detours, message, obstacles, enabled, rth_active}`. `obstacles` = 메모리에 확정된 지도에 없는 장애물 셀 수 |
| `/gas/source_estimate` | geometry_msgs/PointStamped | pub 2 Hz | 가중 중심 추정 (z = 0.4) |
| `/gas/seek_path` | nav_msgs/Path | pub 1 Hz (latched) | 마지막 enable 이후의 궤적 |
| `/battery/state` | sensor_msgs/BatteryState | pub 2 Hz | `percentage` 0..1, 전압, 전류(< 0이면 방전), charge/capacity [Ah], `power_supply_status` CHARGING / DISCHARGING / FULL |
| `/battery/set_level` | std_msgs/Float32 | sub | 테스트용 덮어쓰기, 퍼센트 0..100 |
| `/comm/rssi` | std_msgs/Float32 | pub 5 Hz | 로봇 위치의 dBm (점 모델 + 느린 페이딩) |
| `/comm/rssi_map` | nav_msgs/OccupancyGrid | pub latched (+ 10 s마다 갱신) | 0.5 m. 0..100 ↔ −100..−30 dBm (선형) |
| `/safety/status` | String JSON | pub 2 Hz + 변경 시 | `{state, rth_active, reason, battery_pct (0..100), rssi_dbm, predicted_min_rssi, distance_home, energy_to_home_pct, docked, trigger_time, plan_length, remaining_m, replans, paused, message, rssi_map_source, geodesic_ready, obstacle}`. `obstacle` = `{enabled, source, frames, rejected, confirmed, free_travel, speed_scale, detour, detours, replans, holding, last, errors}` (§7.1) |
| `/safety/rth_active` | std_msgs/Bool | pub 5 Hz + 변경 시 | |
| `/cmd_vel/rth` | geometry_msgs/Twist | RTH 주행 중 pub 10 Hz | |
| `/nav/plan` | nav_msgs/Path | pub (latched) + sub | 복귀 경로를 발행하고, 복귀가 끝나면 빈 Path를 발행(오래된 latched 복귀 경로가 남지 않음). 미션 경로를 구독(자기 에코는 무시) |
| `/locomotion/command` | std_msgs/String | pub | 도크에서 `sit`, 충전 완료 / 재개 시(그리고 앉은 상태에서 RTH가 시작될 때) `stand` |
| `/mission/command` | std_msgs/String | sub | `rth` / `dock` → 수동 RTH, `resume` → 충전기에서 출발, `pause` / `resume` → 복귀 일시정지 / 계속, `abort` → 복귀 취소(§7 참조) (큐 처리: 0.1 s 제어 주기 하나 안에 들어온 여러 명령을 모두 처리) |
| `/odom`, `/map`, `/locomotion/state` | | sub | 오도메트리 twist는 base 좌표계(DESIGN_SPEC). 유한하지 않은 `/odom` 샘플은 버림(경고는 빈도 제한) |
| `/elevation_map/points` | sensor_msgs/PointCloud2 | sub (`return_to_home`, `source_seeker`) | terrain_mapper의 지형 지도. 지도에 없는 장애물을 찾는 데 씀(§4.2, §7.1). BEST_EFFORT depth 1, 토픽은 `obstacle_elevation_topic`. `obstacle_source: traversability`이면 대신 `/elevation_map/traversability`. `obstacle_enabled: false`면 구독하지 않음. `obstacle_frames`(`odom,map,world`)에 없는 frame_id는 버리고 셈 |

좌표계: 모두 `map`(= `odom` = Gazebo 월드)입니다.

---------------------------------------------------------------------------------------

## 3. 가스 확산 (`gas_model.py`)

### 3.1 수송 방정식
홀 + 복도의 바닥 근처 층을 깊이 평균한 2-D 영역으로 봅니다.

    ∂c/∂t + ∇·(u c) = ∇·(K ∇c) + S(x, y) − k c                        (1)

* `c` [ppm], `u(x, y, t)`는 발산이 0이고 장애물을 통과하는 법선 흐름이 0인 바람장입니다.
* `K`는 난류 확산계수입니다(0.03 m²/s. 1차 풍상 차분이 ≈ |u| h / 2 ≈ 0.05 m²/s를 더함).
* `S`는 누출 방출원입니다. 가우시안 풋프린트(σ = 0.25 m, 3.5σ에서 절단, 유체 셀에 대해 정규화)로,
  레이아웃 `gas_leak.source` 위치에서 총 유량 `Q` [ppm·m²/s]를 냅니다.
* `k`는 1차 소실(0.012 1/s)입니다. 바닥 층이 7.3 m 높이의 홀로 수직 혼입되는 것과 지붕 배기를
  나타내며, 2-D 모델에 수직 차원 대신 필요한 항입니다.

### 3.2 격자와 장애물
내비 맵 범위 전체에 엇갈린(staggered) MAC 격자의 0.25 m 유한체적 셀을 둡니다(299 × 77 셀).
내비 맵 5 × 5 하위 샘플 중 15 % 넘게 점유된 가스 셀은 고체입니다(임계값이 낮아서 내비 맵의
0.1 m 두께 벽이 가스를 새지 않음).

### 3.3 질량 보존 바람장 (진단 모델, Sherman 1978 / QUIC)
초기 추정 `u*`: 공장(`x ≤ 14.9`)에서는 `wind_base = (0.40, −0.10)` m/s, 복도에서는 채널 흐름
`(0.20, 0)`입니다. 개구부는 설정된 사각형 안에 있는 벽 면입니다. 유입 벤트(가스 탱크 남/북쪽의
서쪽 벽, 북쪽 벽)는 초기 추정의 법선 속도를 지정하고, 출구(복도 끝, 출입구 옆 동쪽 벽 루버,
남쪽 벽 루버)는 압력 출구 `p = 0`입니다. 개구부는 방향으로 걸러지며(벤트는 유체 *안으로* 불어야
하고, 출구는 하류를 향해야 함), 출구가 하나도 없는 공간의 벤트는 닫습니다(그렇지 않으면 Poisson
문제가 불일치함). 최종 장은 L2 의미에서 가장 가까운 무발산장입니다(Helmholtz–Hodge 투영).

    u = u* − ∇p,     ∇²p = ∇·u*     (Neumann at walls, p = 0 at outlets)     (2)

이 식을 유체 셀마다 `Σ_active(p_nb − p_c) = h² div(u*)`로 쓰고 벡터화한 적흑(red-black) SOR로
풉니다(ω = 2/(1+sin(π/n)), 허용 오차 max|div u| < 1e-6 1/s. 플랜트 맵에서 1500회 반복).
유체 셀과 고체 셀 사이의 면에는 흐름이 없으므로 장애물은 정확히 무유속(no-flux)입니다.

### 3.4 돌풍 (시간 변화, 시드 고정)
투영이 선형이므로 기본장 `W0`와, 90° 회전한 초기 추정의 장 `W90`을 한 번씩만 투영해 둡니다.
순간 바람장은 다음과 같습니다.

    u(t) = s(t) [cos θ(t) W0 + sin θ(t) W90] + Σ_k a_k(t) E_k                (3)

    s(t) = 1 + A_s sin(2πt/T_s + φ_s) + n_s(t)      (A_s 0.15, T_s 47 s)
    θ(t) = A_θ sin(2πt/T_θ + φ_θ) + n_θ(t)          (A_θ 8°,  T_θ 83 s)

`n_s, n_θ, a_k`는 Ornstein–Uhlenbeck 과정입니다(정확한 이산화, τ = 8 s / 15 s). 와류
`E_k = curl ψ_k`는 시드를 고정한 평활 노이즈 유선함수(규모 2.5 m)에서 만들고 벽에서 0으로 줄어들게
하므로, 이산적으로 무발산이고 벽에 접선 방향입니다. 따라서 전체 바람은 매 순간 질량 보존을
유지합니다(단위 테스트, |div| < 1e-5 1/s).

### 3.5 수치해법
유한체적: 1차 풍상 이류 유속 + 중앙 확산 유속(닫힌 면에는 둘 다 없음), 전진 오일러, 서브스텝마다
방출 추가, 정확한 지수 감쇠를 씁니다. 서브스텝은 셀별 양수성 한계로 제한합니다.

    Δt ≤ CFL / max_c [ Σ_out |u_f| / h + n_open K / h² ],   CFL = 0.9           (4)

이 조건이 스킴을 단조롭게 만듭니다(음의 농도가 생기지 않음). 출구로 나간 가스는 장부에
기록합니다. `mass + outflow + decayed = emitted`가 반올림 오차 수준으로 닫힙니다(상대 오차
3e-15). 플랜트 맵에서의 비용: **10 Hz 갱신당 0.5–2.3 ms**(PC가 한가할 때 / 다른 작업과 함께
측정. 평균 서브스텝 1.25) — 10 Hz 실시간 예산의 최대 ≈ 2 %입니다. `spin_up()`은 노드가 발행하기
전에 고정된 기본풍으로 정상상태까지 돌리고(시뮬레이션 400 s를 벽시계 ≈ 1–2.5 s에), 그다음
돌풍을 켜고 30 s를 더 돌립니다.

### 3.6 보정
`Q = 170 ppm·m²/s`. 균일풍 `u`에서 유효 측면 확산계수가 `K_eff = K + u h/2 ≈ 0.08 m²/s`인 2-D
플룸으로 해석적으로 확인합니다.

    c(x, 0) ≈ Q / sqrt(4π K_eff u x) · exp(−k x / u)

이 식은 출입구(풍하 x ≈ 18 m, u ≈ 0.4 m/s)에서 ≈ 37 ppm을 주며, 플랜트 맵의 시뮬레이션
정상상태와 맞습니다. **누출원 옆 ≈ 1470 ppm**(누출원이 가스 탱크의 후류에 있어 바람이 느림),
**동쪽 출입구 전체에서 최대 34 ppm / 평균 17 ppm**, 복도 입구 20 ppm, x = 30 m에서 7 ppm, 도크
근처 1 ppm입니다(`docs/images/gas_field.png` 참조). 목표는 누출원에서 O(500–2000) ppm,
출입구에서 20–80 ppm이었습니다.

### 3.7 센서
`LaggedSensor`: 정확한 0차 유지(zero-order-hold) 1차 지연 `y += (1 − e^(−Δt/τ))(c − y)`를 거친 뒤
가산(0.3 ppm) + 승산(3 %) 가우시안 노이즈, 0.1 ppm 양자화, 0에서 클리핑을 적용합니다. 농도는 고체
이웃을 스텐실에서 뺀 이중선형 보간으로 샘플링합니다.

---------------------------------------------------------------------------------------

## 4. 후각 기반 누출원 탐색 (`source_seeking.py`)

상태 기계 (10 Hz):

    IDLE --enable--> SEARCH --c > detect_ppm--> SURGE --SNR ok & c > gradient_min--> GRADIENT
                      ^  ^                       |  ^                                |   |
                      |  +-- cast exhausted --- CAST <---- plume lost (c < lost_ppm 5 s) -+   |
                      |                                                                    v
                   FAILED (timeout 300 s)                       FOUND (c > found_ppm + local max confirmed)

* **SEARCH** — 커버리지 탐색입니다. 후보 레그 16개(벽에서 벽까지, ≤ 20 m)를 새로움(미방문 1 m 셀),
  횡풍 성분, 풍상 성분, 길이, 역방향 벌점으로 점수화합니다. 횡풍 스윕은 홀 어디에서 출발하든
  플룸 띠를 가로지릅니다.
* **SURGE** — 풍상으로 걷되, 구배의 횡풍 성분으로 플룸 축을 향해 조향합니다.
  `d = unit(u_up + w (ĝ − (ĝ·u_up) u_up))`, `w = 0.8 min(1, SNR/SNR_min)`.
* **CAST** — 마지막 검출 지점 주위에서 진폭이 커지는(1 → 6 m) 횡풍 지그재그이며, 레그마다
  풍하로 0.3 m씩 이동합니다. 재검출 → SURGE, 소진 → SEARCH.
* **GRADIENT** — `f = −ln(c + c0)`에 대한 경사 하강입니다(로그 공간이므로 스텝이 플룸의 절대
  세기와 무관함).

      d_k   = unit(w_g ĝ + β u_up),   w_g = min(1, |g|/g_ref) · min(1, SNR / (2 SNR_min))
      m_k   = μ m_{k−1} + (1 − μ) d_k                     (momentum μ = 0.5)
      x_nose,k+1 = x_nose,k + α_k unit(m_k)                (normalised step)

  각 스텝 후 로봇은 0.8 s 머물고, 필터링된 판독값에 Armijo 유사 검사를 합니다.
  `l_{k+1} ≥ l_k + c1 α_k (g·d) − ε`이면 채택하며 `ε = max(0.1, 1.5 σ_resid)`입니다(노이즈
  인식). 채택된 개선 스텝은 α를 키우고(×1.3, ≤ 1.6 m), 거부된 스텝은 같은 방향으로 α ×0.5
  백트래킹하며(≤ 3회), 그 뒤에는 모멘텀을 초기화하고 방향을 다시 추정합니다. 약한 구배가
  3스텝 이어지면 → SURGE.
* **국소 최대 확인 / FOUND** — `c ≥ found_ppm (600)`이거나 평탄한 정점(found_ppm의 ≥ 70 %에서
  `|∇ln c| < 0.3 1/m`)이면 시작합니다. 정점 근처에서는 시간차 판독을 믿을 수 없으므로 확인에는
  *동시* 차분을 씁니다. 각 정지 지점(2 s)에서 판독별 배열 구배를 평균하고(노이즈 ∝ 1/√n),
  평균 구배가 유의한 동안(SNR ≥ 3) 그 방향으로 r만큼 이동합니다(Rprop 유사: 방향이 뒤집히면 r을
  절반으로, 일관되면 ×1.25. 점 판독이 최고값보다 15 % 넘게 낮으면 그 점은 포기). FOUND = 최고
  지점에 유의한 구배가 없고("구배 크기 작음"), 주변에 더 높은 샘플이 없으며("국소 최대"), c가
  found_ppm보다 높은 상태입니다.
* **FAILED** — 시간 초과(300 s). 실행의 시간 기준은 첫 측정부터 시작합니다(`/clock`이 돌기 전에
  받은 enable이나, 시간이 거꾸로 뛰는 시뮬레이션 리셋이 있으면 시간 초과 기준을 다시 잡고 자세 /
  샘플 이력을 지움).

### 4.1 융합 구배 추정 (로그 공간, `l = ln(c + c0)`)
1. *센서 배열* — 몸체 오프셋 `o_i`에 있는 센서 4개를 지나는 최소제곱 평면:
   `l_i = a + g_b·o_i` → `g_b = [(AᵀA)⁻¹Aᵀl]_{1:2}`,
   `Cov = σ_l² [(AᵀA)⁻¹]_{1:2,1:2}`, `σ_l² = mean((σ_abs/(c+c0))²) + σ_rel² + σ_model²`.
   (지연 보상한) yaw로 월드 좌표계로 회전합니다.
2. *슬라이딩 창* — 최근 20 s 동안의 코 센서 샘플에 대한 가중 최소제곱입니다(위치는 센서 지연
   τ만큼 지연 보상: 시각 t의 판독값은 t − τ의 자세에 속함). 돌풍에 의한 세기 변화를 흡수하는
   시간 드리프트 항을 둡니다.

       l_i = a + g·(p_i − p_0) + b (t_i − t),
       w_i = exp(−age_i/τ_w) exp(−|p_i − p_0|² / 2σ_w²),
       θ = (AᵀWA + Λ)⁻¹ AᵀW l,   Cov_g = s² [(AᵀWA + Λ)⁻¹]_{gg}

   (`Λ` = 약한 사전정보. 직선으로 걸으면 진행 방향 성분을 식별할 수 없게 되어 공분산이 커지므로,
   그 방향은 융합이 배열에 의존합니다.)
3. *융합* (정보 형태 + 평균 0 사전정보): `g = (I_a + I_w + I_p)⁻¹(I_a g_a + I_w g_w)`,
   신뢰도 `SNR = |g| / sqrt(ĝᵀ Cov ĝ)`.

### 4.2 장애물 인식
* 계획한 스텝의 직선 구간이 여유(0.35 m, 내비 맵 거리장)를 어기면 **자유 공간으로 투영**합니다.
  스텝 광선을 따라 장애물 *뒤*에서 여유가 ≥ 0.45 m인 첫 지점을 목표로 삼고, 국소 A*
  (`dusan_common.planning.astar` + 내비 맵 ±8 m 창의 `smooth_path`, 확장 횟수 제한)가 그곳까지
  경로를 냅니다 — 내부 벽의 출입구와 틈을 통과합니다. `min(20 m, 6 × straight + 2 m)`보다 긴
  경로는 거부하고, 실패는 8 s 동안 캐시합니다. 우회 스텝은 재배치로 취급합니다(Armijo 검사 없이
  채택).
* 그렇지 않으면 스텝을 가장 가까운 자유 방향으로 줄이거나 회전합니다(±25° … ±100°).
* 속도 명령에서 가장 가까운 장애물 쪽 성분을 빼고, 0.6 m 안에서는 척력을 더합니다.
* 누출원 추정: 샘플을 0.25 m 빈에서 평균하고, 상위 8개 빈의 가중 중심
  `x̂ = Σ w_i p_i / Σ w_i`, `w_i = c_i − c_(k+1)`을 확인 정지 지점들의 중심과 50/50으로 섞습니다.
* **지도에 없는 장애물**(2026-10-26, `dusan_common/obstacle_layer.py`): `/elevation_map/points`에서 찾은 장애물을
  메모리(`ObstacleMemory`)에 모읍니다. 증거는 5×5 셀 창의 높이 차 ≥ 0.22 m이고, 2프레임이면 확정합니다.
  - 탐색 중에는 확정된 장애물을 내비 격자에 합칩니다. 재구성은 최대 1 Hz(`obstacle_rebuild_period` 1.0 s)입니다.
  - 합칠 때는 가려진 뒷면을 0.3 m로 가정합니다('near' 층). 1.0 m 가정은 장애물 옆의 틈을 막을 수 있기 때문입니다.
  - 그래서 위의 자유 길이, 국소 A* 우회, 속도 투영, 척력이 탐지 장애물도 그대로 피합니다. 별도의 감속·우회 단계는 없습니다.
  - 진행 중인 우회 경로나 스텝 직선이 새 장애물을 지나게 되면 다시 계획합니다.
  - 비활성(IDLE)일 때는 프레임을 1 Hz(`obstacle_idle_period`)로 메모리에만 넣고, enable할 때 격자를 다시 만듭니다.

---------------------------------------------------------------------------------------

## 5. 배터리 (`battery_model.py`)

    P = P_compute + P_sensors + P_posture + k_v |v| + k_ω |ω_z|
      = 25 + 15 + {45 stand | 8 sit} + 110 |v| + 20 |ω_z|   [W]

14 s(직렬 14셀) Li-ion 팩(NMC OCV 곡선 표), 내부 저항 R = 0.08 Ω. 부하 P에 대한 전류는
`P = (OCV − I R) I`를 풀어 `I = (OCV − sqrt(OCV² − 4RP)) / 2R`이고, 단자 전압은 `V = OCV − IR`,
`dSoC/dt = −I V / (3600 E_cap)`입니다(에너지 기반, 200 Wh, 초기 80 %). 도크 충전: 80 %까지
정전류 0.5 %/s, 그 뒤에는 100 %에서 속도의 10 %가 되도록 선형 CV 테이퍼를 적용합니다.
FULL ≥ 99.5 %. 도킹 판정 = 도크에서 `dock_tolerance`(레이아웃, 0.35 m) 이내이고, 저역 통과 필터를
거친 속도 < 0.05 m/s가 1 s 지속됨(서 있는 다리 로봇은 흔들리므로 원시 twist에 노이즈가 많음).
`time_scale` > 1이면 데모용으로 방전을 가속합니다.

## 6. Wi-Fi 커버리지 (`comm_model.py`)

    RSSI_i(x, y) = p0_i − 10 n log10(max(d, 1 m)) − A_wall · N_i(x, y)
    RSSI(x, y)   = max_i RSSI_i − Σ_{zones ∋ (x,y)} A_zone

`d` = AP–안테나 3-D 거리(안테나 0.6 m), `n = 2.2`, AP→로봇 광선을 따라 자유→점유 전이마다
`A_wall = 12 dB`(`OccupancyGridMap.raycast_crossings`. 벡터화한 일괄 투사기가 샘플 단위로 똑같이
재현), 음영 구역 S1(터빈 피트) −40 dB.

**RF 장애물 맵.** 광선은 *`rf_min_obstacle_area_m2`(2.5 m²)보다 작은 점유 연결 성분을 뺀* 내비
맵에서 투사합니다. 2-D 내비 맵은 경로 계획을 위해 팔레트(0.18 m 높이), 드럼, 파이프 랙 기둥,
게이지 스탠드를 점유로 표시하지만, AP는 2–4.5 m 높이에 달려 있어 그런 잡동사니 너머를 봅니다.
블롭마다 12 dB를 매기면 생성된 플랜트 맵에서 홀 자유 면적의 12 %(그리고 게이지 G05의 뷰포인트,
−92 dBm)가 −80 dBm 음영 임계 아래로 떨어져, 통신 음영 트리거가 순찰 미션을 계속 중단시키게
됩니다. 필터를 쓰면 차폐된 피트 S1과 터빈 뒤 영역만 남고(홀의 1.9 %), 모든 게이지 뷰포인트가
커버됩니다(≥ −76 dBm, 단위 테스트). 벽(건물 외피와 연결됨), 기계, 베이 구조물(9–855 m²)은 여전히
감쇠합니다. `rf_min_obstacle_area_m2: 0.0`이면 문자 그대로의 블롭별 모델을 재현합니다.

맵은 0.5 m 해상도로 미리 계산하고(5.7k 셀 × AP 3개, ≈ 1 s) latched로 발행합니다. `/comm/rssi`는
정확한 점 모델에 느린 로그 정규 페이딩(OU, σ 2 dB, τ 2 s)을 더해 씁니다.

## 7. 자율 복귀 감독기 (`rth_logic.py`)

    MONITOR --trigger--> RTH_PLANNING --A* ok--> RTH_NAVIGATING --< 1 m--> DOCKING --aligned 1 s--> DOCKED ('sit')
       ^                    ^  (back-off,          |  no progress 10 s                                   |
       |                    +--  re-plan) ---------+                                                    v
       +------- battery ≥ 95 % or 'resume' ('stand') ----------------------------------------------- CHARGING

DOCKING은 도크까지 직선이 탐지 장애물에 막혀 있지 않을 때만 들어갑니다. 도킹 중 10 s 무진전이면 RTH_PLANNING으로 돌아갑니다(§7.1).

트리거 (MONITOR에서, 우선순위 순):

| 사유 | 조건 |
|---|---|
| `manual` | `/mission/command` `rth` (또는 `dock`) |
| `battery_low` | 배터리 < 20 %가 1 s 지속 |
| `energy_reserve` | `E_home% × 1.3 + 3 % > battery %`가 2 s 지속(그리고 도크에서 1 m 넘게 떨어져 있을 때). `E_home` = **측지** 귀환 거리(팽창 내비 맵 위 Dijkstra 장, 백그라운드 스레드에서 한 번 계산)를 0.3 m/s로 걷는 데 드는 에너지(배터리 전력 모델 사용) + 도킹 30 s |
| `comm_shadow_predicted` | `/nav/plan` 위 로봇 투영점부터 앞쪽 5 m 구간, 또는 4 s 속도 외삽 구간의 최소 RSSI(RSSI 맵) < `rssi_min_dbm`(−80 dBm, 레이아웃). 도달 가능한 위치만 셉니다. 장애물 안의 계획 샘플은 버리고, 외삽은 첫 점유 셀에서 끝냅니다(외벽 쪽으로 걸을 때 벽 뒤의 감쇠된 RSSI를 "예측"하면 안 됨) |
| `comm_lost` (선택) | 측정한 `/comm/rssi` < −80 dBm이 2 s 넘게 지속 |

주행: A*(`dusan_common.planning.astar`, 0.3 m 팽창, 노드에서는 백그라운드 스레드)를 0.1 m 추가
여유 마진으로 평활하고(지름길 구간이 벽 모서리에서 떨어지도록), 조밀화한 pure pursuit
(`dusan_common.pure_pursuit`)와 속도 안전 필터(`k = clip((d − r/2)/(d_safe − r/2), 0, 1)`,
`v ← v − (1−k) min(v·n, 0) n + v_rep (1−k) n`, d_safe 0.5 m)를 씁니다. 10 s 동안 진전(0.3 m)이
없으면 → 여유 기울기 방향으로 2 s 후진한 뒤 재계획합니다. 도크 1 m 이내 → 위치(≤ 0.12 m)와 yaw
(≤ 0.12 rad)의 홀로노믹 P 정렬을 1 s 유지 → DOCKED(`sit`) → CHARGING(2 s 뒤) → 95 %까지 충전 →
MONITOR(`stand`). DOCKED/CHARGING 중 `resume`이면 즉시 출발합니다. RSSI 맵을 모르면 음영을
예측하지 않습니다(측정 기반 `comm_lost` 트리거는 여전히 동작).

귀환 경로의 A*와 평활에는 **지형 비용**(`dusan_common/terrain_cost.py`)이 들어갑니다. 내비 맵에서 자유 칸인
잔해(누운 배관, 잔해 지대, 철판, 낮은 단)를 격자에 비용으로 그립니다. 로봇 중심이 배관·잔해에서 0.12 m 안이면 칸마다
25, 철판 1, 낮은 단 2이고, 그 바깥 0.25 m 여유 폭은 그 값의 0.15배입니다. 평활도 A* 경로보다 비싼 칸을 지나는 지름길은
쓰지 않습니다. 그래서 돌아갈 길이 있으면(C-OBS1 배관 북쪽 끝, C-OBS3 잔해 북쪽 통로) 돌아가고, 없으면(C-OBS4 철판,
C-OBS5 배관) 그대로 지나갑니다. 비용은 유한하므로 경로를 못 찾는 일은 생기지 않습니다. 도크까지의 측지 거리(에너지
예측)는 비용 없이 계산합니다. 우회로 늘어나는 거리는 1 m 안팎입니다.

복귀 중(RTH_PLANNING / RTH_NAVIGATING / DOCKING)의 운영자 명령:

| 명령 | 효과 |
|---|---|
| `pause` | 위치 유지(`/cmd_vel/rth` 0. mux가 로봇을 계속 잡도록 `rth_active`는 true 유지), `/safety/status`에 `paused: true`. 유지 중에는 정체 재계획을 하지 않음 |
| `resume` (또는 `rth` / `dock`) | 계속: 현재 자세에서 재계획(teleop으로 로봇이 옮겨졌을 수 있음) |
| `abort` | 복귀 취소 → MONITOR, `rth_active` false, 빈 `/nav/plan`, 보행 상태는 그대로(`stand` 없음). 자동 트리거 조건이 지금 성립하면(배터리 < 20 %, 에너지 예비, 로봇 위치의 통신 음영, 측정된 통신 끊김) **거부**합니다. 복귀가 즉시 다시 트리거될 것이므로 복귀를 계속합니다(메시지 `abort refused: ...`. 로봇을 멈추려면 `pause`나 teleop 사용). 도크에서는 무시(DOCKED / CHARGING: `resume`으로 출발) |

### 7.1 지도에 없는 장애물 (2026-10-26)

귀환 경로에 내비 맵에 없는 물체(상자 등)가 있으면 `/elevation_map/points`로 찾아 피합니다. 장애물 층은 미션 관리자와 같은
`dusan_common/obstacle_layer.py`이고, 판단은 귀환 제어 주기(10 Hz)마다 합니다. 설계와 실측은
[REPORT §7.8](../docs/REPORT.md)에 있습니다.

| 단계 | 조건 | 동작 |
|---|---|---|
| 감속 / 정지 | 따라가는 선 위 자유 주행 거리 < 1.0 m / < 0.35 m | 속도를 선형으로 줄이고, 0.35 m 미만이면 회전까지 0. 확정 전 증거는 0.5배까지만 감속(0.35 m 안이면 정지) |
| 좌·우 우회 | 확정 장애물이 2.5 m 안에서 경로를 막음 | 좌·우 중 길이 + 지형 비용이 싼 쪽으로 돌아 장애물 너머 0.8 m 이상에서 합류. 방향은 15 s 고정, 목표당 8회 |
| 재계획 | 우회가 없음 | 순서대로 시도: ① 지도 + 탐지 장애물(뒷면 1.0 m 가정) ② 같은 지도(뒷면 0.3 m 가정) ③ 지도만. ②가 ①보다 1.5배 이상 싸면 ②를 택함. 장애물로 인한 재계획은 목표당 3회 |
| 정체 | 계속 막힘 | 기존대로 10 s 무진전 → 여유 기울기 방향 후진 → 재계획 |

- 감속기는 pure pursuit가 실제로 가는 선(로봇 → 0.6 m 앞 목표점 → 경로)에서 거리를 잽니다. 그 선이 장애물 쪽 모서리나 지도의
  배관·잔해를 깎으면 그 주기의 look-ahead를 0.3 m로 줄여 경로를 바짝 따릅니다.
- 속도 안전 필터와 정체 후진은 지도 장애물과 탐지 장애물 중 가까운 쪽을 기준으로 합니다.
- **도킹**: 도크까지의 직선이 탐지 장애물에 막혀 있지 않을 때만 DOCKING에 들어갑니다. 도킹 중에도 그 직선에서 감속하고, 정지할
  때는 0 명령을 보냅니다. 이때 안전 필터는 탐지 장애물만 봅니다(도크 옆 지도 벽 제외). `stuck_time_s`(10 s) 동안 진전이 없으면
  RTH_PLANNING으로 돌아가 다시 계획합니다. 도크 앞이나 위의 상자로 밀고 들어가지 않습니다.
- 주행하지 않을 때(MONITOR, DOCKED, CHARGING)는 프레임을 1 Hz(`obstacle_idle_period`)로만 처리합니다. 메모리는 평소 순찰 중에도
  유지되므로 귀환을 시작할 때 이미 본 장애물을 압니다.
- `/safety/status`의 `obstacle`에 확정 셀 수, 속도 배율, 우회 방향, 우회·재계획 횟수, 마지막 판단(`last`)이 나옵니다. `source`는
  노드 파라미터 `obstacle_source`를 그대로 표시합니다.
- `obstacle_enabled: false`면 구독도 하지 않고 이전 동작과 같습니다.
- **한계**: 완전히 막혀도 귀환에는 건너뛸 목표가 없습니다. 후진 → 재계획을 계속 반복합니다. 카메라는 높은 면을 약 1.0–1.45 m
  앞에서만 보고, 높이 0.22 m 미만 물체는 넘습니다.

---------------------------------------------------------------------------------------

## 8. 파라미터

모든 튜닝 값은 타입이 있는 기본값으로 선언된 ROS 파라미터입니다(`params.py`). YAML 파일에는 같은
키가 같은 값으로 들어 있습니다(`test_m3_config.py`가 검사. rclpy는 double 파라미터에 정수를
거부하므로 모든 double은 소수점을 붙여 씀). 주요 파라미터:

| 노드 | 파라미터 | 기본값 | 의미 |
|---|---|---|---|
| gas | `source_override` / `source_x,y` | false / −3.2, 2.0 | false → 레이아웃 `gas_leak.source` |
| gas | `emission_rate` | 170.0 | Q [ppm·m²/s] |
| gas | `wind_override` / `wind_x,y` | false / 0.40, −0.10 | false → 레이아웃 `gas_leak.wind_base` |
| gas | `diffusivity`, `decay_rate` | 0.03, 0.012 | K [m²/s], k [1/s] |
| gas | `gusts_enabled`, `gust_*`, `eddy_*`, `seed` | true … | 돌풍 과정 |
| gas | `inflow_rects`, `outflow_rects` | YAML 참조 | 평탄한 `[x0,y0,x1,y1,…]` |
| gas | `spinup_seconds`, `spinup_gust_seconds` | 400, 30 | 발행 전 워밍업 |
| gas | `sensor_tau`, `sensor_noise_ppm`, `sensor_noise_rel` | 1.0, 0.3, 0.03 | 센서 모델 |
| gas | `vis_max_ppm`, `vis_log_scale` | 1000, true | 격자 인코딩 |
| seeker | `detect_ppm`, `lost_ppm`, `found_ppm` | 2.5, 1.2, 600 | 임계값 |
| seeker | `step_init/min/max`, `momentum`, `upwind_bias` | 0.8/0.25/1.6, 0.5, 0.5 | 경사 스텝 |
| seeker | `detour_*` | 활성, 도달 거리 5 m, ±8 m 창 | 장애물 우회 |
| seeker | `timeout` | 300 | 이 시간 [s] 뒤 FAILED |
| seeker | `resume_window_s` | 45 | enable false → true가 이 시간 안이면 실행을 이어 감 (0 = 항상 새로 시작) |
| battery | `capacity_wh`, `initial_pct` | 200, 80 | 팩 |
| battery | `p_*_w`, `charge_rate_pct_per_s` | §5 참조 | 전력 / 충전 |
| comm | `map_resolution`, `fading_sigma_db` | 0.5, 2.0 | |
| comm, rth | `rf_min_obstacle_area_m2` | 2.5 | 이 풋프린트 [m²]보다 작은 잡동사니는 RF가 투과 |
| rth | `smooth_margin`, `safety_dist`, `backoff_s` | 0.1, 0.5, 2.0 | 경로 여유 마진, 안전 필터, 후진 |
| rth | `terrain_cost_enabled`, `terrain_w_pipe`, `terrain_w_rubble`, `terrain_w_plate`, `terrain_w_platform` | true, 25, 25, 1, 2 | 귀환 경로의 지형 비용(배관·잔해 안쪽 칸 비용, 철판·낮은 단은 작게) |
| rth | `terrain_margin`, `terrain_margin_fraction` | 0.25, 0.15 | 여유 폭 [m] (0.25 m 상한), 여유 폭 비용 = 비율 × 안쪽 비용 |
| rth | `rth_battery_pct`, `resume_battery_pct` | 20, 95 | |
| rth | `energy_safety_factor`, `energy_margin_pct` | 1.3, 3 | |
| rth | `shadow_lookahead_m`, `shadow_horizon_s` | 5, 4 | |
| rth | `rssi_min_dbm` | 0 (→ 레이아웃 −80) | |
| rth, seeker | `obstacle_enabled` | true | 지도에 없는 장애물 처리. false = 이전 동작, `/elevation_map/points` 구독 안 함 |
| rth, seeker | `obstacle_source`, `obstacle_elevation_topic`, `obstacle_frames` | `elevation`, `/elevation_map/points`, `odom,map,world` | 입력 선택. `traversability`면 `obstacle_traversability_topic`의 비용 ≥ `obstacle_cost_threshold`(100) |
| rth, seeker | `obstacle_step_height`, `obstacle_step_window` | 0.22, 5 | 증거: 창 [셀] 안 높이 차 [m]. terrain.yaml `step_crit`·`step_window`와 같게 유지 |
| rth, seeker | `obstacle_confirm_hits`, `obstacle_max_hits`, `obstacle_forget_time` | 2, 5, 60.0 | 확정 프레임, 지우는 데 필요한 빈 프레임, 망각 [s] |
| rth, seeker | `obstacle_ignore_radius`, `obstacle_static_tolerance` | 0.55, 0.3 | 몸통·다리 무시 반경 [m], 지도 장애물 근처 무시 거리 [m] |
| rth, seeker | `obstacle_shadow_depth`, `obstacle_idle_period` | 1.0, 1.0 | 가려진 뒷면 가정 [m] (탐색기는 0.3 m near 층 사용), 비주행 시 프레임 주기 [s] |
| rth | `obstacle_slow_dist`, `obstacle_stop_dist`, `obstacle_unconfirmed_scale` | 1.0, 0.35, 0.5 | 감속 시작 [m], 정지 [m], 확정 전 증거의 최저 배율 |
| rth | `obstacle_margin`, `obstacle_lookahead`, `obstacle_rejoin_gap` | 0.1, 2.5, 0.8 | 탐지 장애물 추가 여유 [m], 막힘 검사 거리 [m], 합류 거리 [m] |
| rth | `obstacle_max_detours`, `obstacle_max_replans`, `obstacle_switch_ratio`, `obstacle_commit_time` | 8, 3, 1.5, 15.0 | 목표당 우회·재계획, 방향 고정 비율과 시간 [s] |
| seeker | `obstacle_rebuild_period` | 1.0 | 탐색 중 내비 격자 재구성 최소 간격 [s] |

`safety.yaml`의 `return_to_home`에는 `obstacle_*` 키가 모두 있고(값은 `mission.yaml`과 같음), `seeker.yaml`에는 인지·메모리 키와
`obstacle_rebuild_period`만 있습니다. 전체 목록은 DESIGN_SPEC §5에 있습니다.

RTH 노드는 배터리 전력 파라미터의 사본을 가지고 있으며(에너지 예측용), 테스트 스위트가 그 값이
`battery_simulator`의 값과 같은지 검사합니다.

---------------------------------------------------------------------------------------

## 9. 평가 (`tools/eval_gas.py`, 결과는 `docs/results/gas_eval.json`)

9절의 수치는 `docs/results/gas_eval.json`에 있습니다. 그림:
`docs/images/gas_field.png`(정상상태 플룸 + 바람 화살표), `gas_seek_paths.png`(몬테카를로
궤적), `gas_seek_timeline.png`(한 실행의 농도와 상태 타임라인), `rth_timeline.png`(세 RTH
시나리오의 배터리 / 에너지 / RSSI 타임라인), `rth_scenarios.png`(RTH 궤적을 그린 RSSI 맵).

아래 수치는 모두 생성된 플랜트 맵(`simulation/maps/plant_map.yaml`, 0.10 m)에서
`python tools/eval_gas.py`로 얻었습니다. 실행은 시드가 모두 고정되어 있어 재현 가능합니다.

**(a) 플룸** (기본풍으로 600 s 뒤의 정상상태, 가스 셀 299 × 77)

| 항목 | 값 | 목표 |
|---|---|---|
| 누출원 0.75 m 이내 최고 농도 | 1469 ppm | O(500–2000) ppm |
| 동쪽 출입구 (x 13.9–14.9 m, \|y\| < 2.8 m) | 최대 34 ppm, 평균 17 ppm | ~20–80 ppm |
| x = 16 / 20 / 30 / 45 / 60 m에서 복도 중심선 최대 | 20 / 14 / 6.6 / 2.5 / 1.1 ppm | 도크에서 오는 로봇에 플룸이 닿음 |
| 바람 투영 | SOR 1500회 반복, max\|div W0\| = 7e-7 1/s | 무발산 |
| 유체 내 풍속 (평균 / 최대) | 0.25 / 1.30 m/s (최대는 출입구 제트) | |
| 질량 수지 `mass + outflow + decayed − emitted` | 상대 3e-15 | 반올림 오차 수준으로 닫힘 |
| 돌풍 포함 10 Hz 갱신 1회 비용 | 0.5–2.3 ms (한가한 / 부하가 걸린 PC), 서브스텝 1.25 | 10 Hz 실시간 |
| 모델 생성 / 600 s 스핀업 (벽시계) | 0.6–0.9 s / 1.5–3.5 s | 노드 시작 |

**(b) 몬테카를로 누출원 탐색** (40회. 무작위 시작 자세: 85 %는 홀 동쪽 절반 x 2.5–14.5 m, 15 %는
출입구 / 복도 입구. 시행마다 독립 실현한 돌풍 + 와류를 켜고 20–60 s 돌풍 워밍업. 센서 지연 1 s,
노이즈 0.3 ppm + 3 %. 가속 제한과 접촉 검출이 있는 0.35 m/s 홀로노믹 로봇. 시간 초과 300 s.
성공 = FOUND이고 오차 < 1 m)

| 지표 | 값 | 목표 |
|---|---|---|
| 성공률 / FOUND 비율 | **100 % (40/40)** / 100 % | ≥ 90 % |
| 위치 오차 평균 / 중앙값 / p90 / 최대 | **0.48** / 0.48 / 0.57 / 0.67 m | 평균 < 1.0 m |
| FOUND까지 걸린 시간 평균 / 중앙값 / p90 | 116 / 111 / 175 s | |
| 경로 길이 (평균) | 22.9 m | |
| 장애물 접촉 (전체 시행) | 0 | |
| CAST가 필요했던 시행 / 플룸 밖에서 시작한 시행 | 7 / 8 | |
| 시작 영역별: 메인 홀 (20) / x ≈ 9.9 m 벽 뒤 동쪽 베이 (14) / 출입구 (6) | 100 % / 100 % / 100 %, 평균 88 / 143 / 143 s | |
| 기법 제거: 장애물 우회(국소 A*) 없음 | 성공 95 % (동쪽 베이 93 %, 출입구 83 %), 평균 125 s | 벽 뒤에서는 우회가 중요 |

**(c) RTH 시나리오** (pure pursuit 미션 로봇, 배터리 + 도크 시뮬레이션, RSSI 점 모델, 감독기
10 Hz)

| 시나리오 | 트리거 | 시점 | 트리거 시 로봇 상태 | 결과 |
|---|---|---|---|---|
| `battery_drop` (홀 순찰, 시작 21.2 %) | `battery_low` | t = 79.3 s, 19.99 % | 도크에서 56.3 m (측지 거리) | 56 m 주행 후 귀환, DOCKED 224.5 s (`sit`), 95 %까지 충전 → MONITOR 405.1 s (`stand`). 최저 17.4 %, 도킹 오차 0.11 m, 접촉 0 |
| `comm_shadow_ahead` (미션 경로가 차폐 피트 S1로 향함) | `comm_shadow_predicted` | t = 26.7 s | RSSI −56 dBm (아직 연결됨), 첫 < −80 dBm 지점까지 (경로를 따라) 5.03 m, 구역 S1까지 5.0 m | S1에 들어가지 않음. docked 197.8 s, MONITOR 270.2 s. 도킹 오차 0.12 m |
| `energy_reserve` (노후 50 Wh 팩, 서쪽 홀 루프, 시작 34 %) | `energy_reserve` | t = 236.6 s, 22.8 % (E_home 15.3 % × 1.3 + 3 % = 22.9 %) | 도크에서 63.7 m, 배터리 20 % 임계보다 높음 | 11.1 %로 도착 (1.3 마진 + 3 %가 실제 소비 11.7 %를 감당), DOCKED 402.0 s, MONITOR 595.3 s. 도킹 오차 0.11 m |

커버리지: 맵 전체 범위의 3.2 %(벽, 외부, 피트 S1)가 −80 dBm 미만입니다. 모든 게이지 뷰포인트와
계획된 순찰 경로는 ≥ −71 dBm을 유지하므로, 정상 순찰 중에는 통신 음영 트리거가 발동하지 않습니다.

## 10. 테스트

`python -m pytest module3_gas_safety -q` — 바람장 발산과 무유속 벽, 와류의 솔레노이드성, 벤트/출구
질량 수지, 정확한 질량 보존(닫힌 상자, 감쇠 없음), 방출량 계산, 개방 플랜트 질량 수지, 플룸이
바람을 따르는지(두 풍향), 출입구 보정, 양수성 + 실시간 비용, 센서 지연. 배열/창 구배 추정,
드리프트 분리, 정보 융합, 상태 기계(SEARCH → SURGE → CAST → FAILED), 자유 길이 / 벽 틈을
지나는 우회 경로, 레이아웃 맵과 생성된 플랜트 맵에서의 폐루프 탐색. 배터리 방전 / CC-CV 충전 /
전압 규약 / 도크 판정. 거리에 따라 단조로운 RSSI, 벽당 12 dB, 음영 구역, 일괄 투사기 == 공용
광선 투사기, 맵 인코딩. 모든 RTH 트리거, 디바운싱, 가짜 로봇으로 전체 상태 진행(MONITOR → … →
CHARGING → MONITOR, `sit`/`stand`), resume, 안전 필터, 후진 + 재계획, 도달 가능한 위치로 제한한
음영 예측(벽). YAML ↔ 선언된 파라미터(이름, 타입, 값), launch 파일, 엄격한 rclpy / 메시지
클래스 대체물에 대한 노드 5개 전체(`/clock` 전의 탐색기 시간 기준, FOUND / FAILED 뒤의 침묵,
큐 처리되는 `/mission/command` 포함).

지도에 없는 장애물은 `test/test_m3_obstacles.py`가 검사합니다. 귀환 우회 후 DOCKED(접촉 0), 완전 차단 시 지도 대체 계획
(계획 실패로 세지 않음), 좁은 틈에서 near 층 대체, 도크 옆·도크 위 박스(접촉 없이 재계획), 동적 안전 필터, `obstacle.source`,
탐색기의 장애물 병합·재경로·비활성, 파라미터 일관성입니다. 노드 스텁 테스트는 두 노드의 구독, 프레임 거부, 비주행 시 프레임
솎기를 확인합니다.
