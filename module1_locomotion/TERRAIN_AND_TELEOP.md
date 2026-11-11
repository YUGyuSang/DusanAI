# Module 1 - 지형 인지와 키보드 텔레옵

이 문서는 `module1_locomotion`의 지형 쪽 절반과 운용자 텔레옵을 다룹니다.

| 파일 | 역할 |
|---|---|
| `module1_locomotion/pointcloud_utils.py` | PointCloud2 <-> numpy(`sensor_msgs_py` 없음), 깊이 영상 -> 점, 클라우드 좌표 규약 자동 판별, 몸체·거리 필터를 포함한 카메라 -> 월드 변환, 시간 보간 자세 버퍼 |
| `module1_locomotion/elevation_map.py` | 로봇 중심 2.5-D elevation map, 칼만 높이 융합, 파생 레이어, organised 클라우드 전송 |
| `module1_locomotion/foothold_planner.py` | 지형을 고려한 착지점 탐색 + 스윙 정점 보조 함수(MPC 컨트롤러가 사용) |
| `module1_locomotion/terrain_classifier.py` | 로봇 앞 지형의 클래스 / 권장 보행, 히스테리시스 포함 |
| `module1_locomotion/friction_map.py` | 마찰 사전정보: `world_layout.yaml`의 저마찰 구역(철판 `type: plate` + `friction_zones`), 진행 방향 코리도 겹침 검사 |
| `module1_locomotion/terrain_mapping_node.py` | ROS 2 노드 `terrain_mapper`(얇은 래퍼) |
| `module1_locomotion/teleop_keys.py` | 키맵, 키 디코딩, hold / latch 속도 로직, 상태 줄(ROS 없음) |
| `module1_locomotion/keyboard_teleop.py` | ROS 2 노드 `keyboard_teleop`(터미널 raw 모드 + Gazebo 키) |
| `config/terrain.yaml`, `config/teleop.yaml` | 조정 가능한 모든 값, 코드 기본값과 동일 |
| `launch/locomotion.launch.py` | terrain mapper + locomotion controller (+ 선택적 텔레옵 창) |
| `tools/eval_terrain.py` | 오프라인 평가 -> `docs/images/terrain_*.png`, `docs/results/terrain_eval.json` |

모든 알고리즘은 순수 numpy입니다(SciPy, OpenCV 불필요). 두 노드 파일은 ROS 메시지 변환만 합니다.

## 1. 데이터 흐름

```
/depth_camera/points (PointCloud2, 10 Hz) --+                      +--> /elevation_map/points        (5 Hz)
/depth_camera/image  (fallback, 32FC1/16UC1)-+--> terrain_mapper --+--> /elevation_map/traversability (5 Hz)
/depth_camera/camera_info (optional)  ------+        ^             +--> /terrain/info  JSON           (5 Hz)
/odom (world-frame pose, 50 Hz) --------------------+

/elevation_map/points --> locomotion_controller: ElevationMap.from_cloud_xyz -> FootholdPlanner.select,
                          max_height_along (스윙 정점), height_at (경사에서 몸체 높이 / pitch)
/terrain/info         --> locomotion_controller: recommended_gait (gait:auto), ahead_clear (속도 스케일링),
                          slippery / friction_mu (슬립 모드, README §3.8)
```

`/elevation_map/points`는 맵의 전송 형식입니다. ORGANISED 클라우드로 `height = rows`, `width = cols`이고,
오프셋 0/4/8에 float32 `x, y, z`(`point_step` 12, 리틀 엔디언), 프레임 `odom`, `x`는 열 방향으로 증가,
`y`는 행 방향으로 증가, 미지 셀은 `z = NaN`입니다. `ElevationMap.from_cloud_xyz(xyz)`는 좌표에서 해상도와
원점을 복원하고(float32 오차에 강건) 모든 레이어를 계산하므로, 컨트롤러는 매퍼가 가진 맵과 똑같은 맵을 받습니다.

## 2. 포인트 클라우드와 깊이 영상 (`pointcloud_utils.py`)

* **파싱**은 `PointField` 표(name / offset / datatype / count, 8가지 datatype 모두), `point_step`,
  `row_step`(행 패딩), `is_bigendian`, organised·unorganised 클라우드를 모두 따릅니다. Gazebo 클라우드는
  `x y z rgb`를 담고 "반사 없음"은 `inf`입니다. 비유한 점은 버립니다.
* **좌표 규약 자동 판별.** gz-sensors는 보통 깊이 클라우드를 카메라 LINK 프레임(x 앞, y 왼쪽, z 위)으로,
  다른 드라이버는 OPTICAL 프레임(z 앞, x 오른쪽, y 아래)으로 발행합니다. 깊이 카메라는 앞쪽만 보므로
  앞 방향 축은 사실상 모든 점에서 양수이고, 측방 축은 양쪽 부호를 모두 가집니다:
  `score = P(x > 2 cm) - P(z > 2 cm)`(+1 link, -1 optical). 판별기는 프레임마다 투표하고,
  `convention_lock_frames`(5) 프레임 동안 여유가 0.3 이상이면 결과를 고정합니다.
  `cloud_convention: link|optical`로 판별을 건너뛸 수 있습니다. `frame_id`가 `odom`/`map`/`world`인
  클라우드는 이미 정합된 것으로 보고, 같은 필터를 적용하기 위해 카메라 프레임으로 되돌립니다.
* **깊이 영상 대체 경로.** `image_fallback_timeout`(1 s) 동안 쓸 수 있는 클라우드가 없으면
  `/depth_camera/image`를 역투영합니다. 32FC1 = 미터(`inf`/NaN = 반사 없음), 16UC1 = 밀리미터(0 = 반사
  없음)입니다. 내부 파라미터는 `/depth_camera/camera_info`가 유효하면 그것을, 아니면
  `PinholeCamera.from_sensor('depth_camera')`(212 x 120, hfov 1.5 rad, fx = fy = 106 / tan(0.75))를 쓰고,
  영상 크기가 다르면 비율을 맞춥니다. 깊이 d인 픽셀 (u, v)는 link 프레임 점
  `(d, -(u-cx) d/fx, -(v-cy) d/fy)`가 됩니다.
* **정합.** `T_world_cam = pose_matrix(odom pose) * sensor_extrinsic('depth_camera')`(카메라는
  `base_link` 앞 0.34 m, 아래로 0.6 rad 기울어짐). 오도메트리 자세는 클라우드 타임스탬프에서 보간합니다
  (`PoseBuffer`: 2 s 기록, 최단 호 각도 보간, 0.3 s 이하 외삽, 그 밖에는 최신 자세).
* **필터링**(융합 전): 카메라에서 0.15 ... 3.5 m 범위 밖의 점, 로봇 몸체 상자(`base_link`
  [-0.50, 0.50] x [-0.28, 0.28] x [-0.29, 0.25] m: 몸통, hip, 스윙 발) 안의 점, base보다 0.55 m 넘게 높은
  점(로봇이 아래로 지나가는 돌출부. 2.5-D 맵은 이를 표현할 수 없음)을 버립니다.

## 3. Elevation map (`elevation_map.py`)

### 격자
월드에 정렬된 로봇 중심 창이며, 기본값은 6 m x 6 m, 해상도 0.04 m(150 x 150 셀)입니다. 행 ~ y,
열 ~ x이고 `origin`은 셀 [0, 0]의 중심입니다. 셀 중심은 전역 격자(해상도의 배수) 위에 있으므로
`recenter(x, y)`는 데이터를 셀 단위로 옮기며 리샘플링 번짐이 없습니다. 창을 벗어난 셀은 버리고, 새로
들어온 셀은 미지 상태입니다.

### 한 프레임의 융합 ("ray-free", Fankhauser et al. 방식)
1. **셀별 강건 통계.** 셀의 모든 점을 측정값 하나, 즉 높이 중앙값으로 줄입니다. 셀의 점이 수직으로
   퍼져 있으면(점 3개 이상에서 `spread > max(bias_spread, 4 sigma)`: 계단 챌판, 블록 모서리, 벽면)
   대신 90 % 분위수를 써서(**최대 높이 편향**) 장애물 모서리가 평균에 묻히지 않게 합니다.
2. **측정 분산.** 깊이 노이즈는 거리의 제곱에 비례해 커집니다: `sigma(r) = 0.003 + 0.0025 r^2` m.
   셀마다 `R = sigma^2 / min(n, 4) + 0.5 * max(s_IQR^2 - sigma^2, 0) + 1e-6`
   (`n`은 점 수, `s_IQR` = 셀 내 사분위 범위 / 1.349).
3. **게이팅을 포함한 칼만 갱신.** 기존 셀 (h, P)에 대해 `d^2 = (z - h)^2 / (P + R)`입니다.
   정상치(`d <= 3`): `K = P / (P + R)`, `h <- h + K (z - h)`, `P <- (1 - K) P`.
   이상치는 융합하지 않습니다. 일관된 이상치가 이어지면
   (`|z_k - z_(k-1)| <= max(3 cm, 3 sqrt(2R))`) 셀을 교체합니다. 더 높으면 2프레임(새 장애물. 한
   프레임짜리 혼합 픽셀 스파이크는 거부), 더 낮으면 3프레임(가림 / 스치는 반사 방지)입니다. 수직으로
   퍼진 측정은 실제 높이의 하한이므로(면의 일부만 보임) 더 높은 값은 셀을 즉시 올립니다.
4. **확인.** 셀은 한 프레임에서 점 3개 이상으로 측정되거나 두 번째로 일관되게 관측된 뒤에만 파생
   레이어에 들어갑니다. 확인되지 않은 셀에 일관되지 않은 값이 들어오면 다시 초기화합니다. 이렇게 해서
   가림 그림자에 떨어진 외딴 가짜 반사를 없앱니다.
5. **망각.** 모든 기존 셀의 분산은 `1e-4 m^2/s * dt`씩 커지고, `max_variance`(0.02 m^2)를 넘으면 셀은
   다시 미지 상태가 됩니다. 맵 시계는 노드 시계(시뮬레이션 시간)입니다.

### 파생 레이어 (지연 계산, 높이가 바뀔 때까지 캐시)
| 레이어 | 정의 |
|---|---|
| `elevation` | 융합 높이. 반점 제거(외딴 셀, 모든 이웃보다 8 cm 넘게 튄 값), 기존 셀로 둘러싸인 작은 구멍 채움(형태학적 닫힘 반경 2, 기존 이웃의 중앙값. 가림 그림자는 아래 참고) |
| `variance` | 융합 분산. 채운 셀은 이웃 평균 + 4e-4 m^2 |
| `slope` [rad] | Sobel 가중치 [1,2,1] x [1,2,1]을 쓴 3 x 3 가중 최소제곱 평면에서 `atan(\|grad\|)`. 이웃이 완전하면 Sobel 연산자와 같고, 구멍 / 경계에서는 NaN을 고려 |
| `roughness` [m] | 5 x 5 평면 맞춤의 잔차 표준편차(불편, N/(N-3)). 경사에 불변이며 깨끗한 경사로의 거칠기는 약 0 |
| `step` [m] | 5 x 5(0.2 m, 발 크기) 창의 최대 - 최소 높이 |
| `traversability` [0..1] | `1 - (w_s min(slope/0.70,1) + w_st min(step/0.22,1) + w_r min(rough/0.05,1)) / (w_s+w_st+w_r)`. `step >= 0.22 m`이거나 평면(`rough < 0.02`)이 0.70 rad보다 가파르면 0으로 강제 |

**가림 그림자.** 모든 셀은 마지막 관측의 센서 위치를 기억합니다. 구멍의 시선은 기존 이웃들의 센서 위치
중앙값에서 오며, 이웃을 가까운 절반(센서 쪽)과 먼 절반으로 나눕니다. 가까운 절반이 먼 절반보다
`shadow_fill_step`(5 cm) 넘게 높으면 그 구멍은 모서리 뒤 그림자(하행 계단 모서리, 배관 / 상자 뒤,
낭떠러지)에 있습니다. 시선이 그 위를 지나갔으므로 먼(낮은) 절반의 중앙값으로 채웁니다. 단순 중앙값을
쓰면 하행 모서리 뒤 그림자가 위 디딤판 높이를 가져가고, 맵의 모서리가 아래 디딤판 쪽으로 옮겨져
착지점이 가장자리에 떨어집니다(평가: 이 규칙이 없으면 하행 계단에서 계획 착지점의 7.1 %가 위험, 있으면
0.8 %). 상행 모서리 앞의 구멍(상행 디딤판 앞면을 스치는 경우)과 매끄러운 이웃(약 22 deg까지의 경사로)은
편향 없는 중앙값을 유지합니다.

평면 맞춤의 모든 창 모멘트는 분리 가능한 상관(가중치가 곱 w(u) w(v))이라서 슬라이스 누적으로
계산합니다. 발행된 클라우드에서 150 x 150 맵과 모든 레이어를 다시 만드는 데 개발 PC에서 8.4 / 11.3 ms
(평가 실행의 p10 / 중앙값, p95 14.2 ms)가 걸리며, 직접 2-D 창 루프보다 약 3배 빠릅니다.
`/elevation_map/traversability`(`nav_msgs/OccupancyGrid`, 같은 격자, 원점 = 왼쪽 아래 모서리)는
`round(100 (1 - traversability))`이고, 미지는 -1입니다.

### 컨트롤러 API (DESIGN_SPEC §5)
```python
emap = ElevationMap.from_cloud_xyz(xyz)      # (rows, cols, 3) -> 맵, 레이어를 즉시 계산
emap.resolution, emap.origin, emap.shape, emap.elevation
emap.height_at(x, y)                         # 쌍선형 보간, NaN 고려 (가장 가까운 셀이 미지면 NaN)
emap.layer('slope' | 'roughness' | 'step' | 'traversability' | 'variance' | 'elevation')
FootholdPlanner(search_radius=0.10, **weights).select(emap, nominal_xy, fallback_z) -> (xyz, score, valid)
max_height_along(emap, p0_xy, p1_xy)         # 스윙 경로를 따라 가장 높은 기존 지형 (없으면 NaN)
```
조회는 맵을 읽기만 하므로(멱등 캐시 제외), 컨트롤러의 맵 콜백에서 다시 만든 맵을 제어 스레드와 MPC
스레드가 함께 쓸 수 있습니다.

## 4. 착지점 계획기 (`foothold_planner.py`)

후보는 명목(Raibert) 착지점에서 `search_radius`(0.10 m) 안의 모든 셀 중심과 정확한 명목 점입니다(평지에서
명목 착지점을 그대로 돌려주기 위해). 비용(낮을수록 좋음):

```
J = 1.0 slope[rad] + 20 roughness[m] + 8 step_foot[m] + 0.6 edge_proximity[0..1]
    + 4 |p - nominal|[m] (+ 미지 셀을 허용할 때 unknown_penalty)
```
`step_foot`은 발 주변 3 x 3(0.12 m) 창의 최대-최소 높이이고, `edge_proximity = 1 - d_edge / 0.10`이며
모서리 셀은 4-이웃 높이 차가 4 cm를 넘는 셀입니다. 제외 조건: traversability < 0.25, `step_foot` > 5 cm
(모서리에 너무 가까움), slope > 0.60 rad, roughness > 4 cm, 0.25 m 안의 가장 낮은 기존 지형보다 0.25 m
넘게 높음(드럼 / 상자 윗면은 평평하지만 닿을 수 없음. 0.17 m 계단 챌판 하나는 통과), 좁게 솟은 지형(배관
윗면, 작은 블록: 국소 평면을 뺀 뒤 5 x 5 창 최솟값보다 3 cm 넘게 높음). 허용되는 후보가 없으면 탐색 반경을
한 번 1.6배로 넓힙니다. 그래도 없으면 `valid = False`이고 명목 xy와 그 위치의 맵 높이(없으면 `fallback_z`)를
돌려줍니다. `z`는 지형 표면 높이입니다. `max_height_along`은 선분을 반 셀 간격으로, 양옆으로 한 셀씩
샘플링해 가장 높은 기존 높이를 돌려줍니다(스윙 정점 여유).

## 5. 지형 분류기 (`terrain_classifier.py`)

관심 영역은 heading 프레임에서 `base_link` 앞 0.3 ... 1.5 m, 폭 +-0.3 m인 코리도입니다. 기존 셀에서 다음
지표를 구합니다: `roughness`(거칠기 레이어의 q90), `max_step`(step 레이어의 q95), `slope_deg`(평면 셀에서
heading 방향의 지배적 기울기, 오르막이 +. 계단은 코리도의 최소제곱 기울기), heading 방향 높이
프로파일(6 cm 구간별 중앙값, 짧은 빈칸은 보간). 프로파일 모서리는 경사 추세를 뺀 뒤 2 cm 넘게 뛰는
4구간 이하의 연속입니다. 단차 모서리(8 ... 22 cm)는 뒤에 15 cm 이상 평평한 디딤판이 있으면 확인됩니다.
올라가는 모서리 뒤 0.3 m 안에 내려가는 모서리가 있으면 돌기(누운 배관)입니다.

| 클래스 | 규칙 (먼저 맞는 것) | 보행 | ahead_clear |
|---|---|---|---|
| obstacle | step >= 0.22 m인 셀 3개 이상, 또는 측정된 모서리 > 0.22 m | crawl | false |
| stairs | 돌기의 일부가 아닌 확인된 단차 모서리 | crawl | true |
| unknown | 관측 비율 < 0.3 (그리고 위 증거 없음) | crawl | false |
| rough | roughness > 1.2 cm, 돌기, 또는 확인되지 않은 단차 | crawl | true |
| slope | abs(slope) >= 6 deg (12 deg부터 crawl) | trot / crawl | true |
| flat | 그 밖 | trot | true |
| slippery | 마찰 사전정보 구역(μ < `slippery_mu` 0.35)과 겹치고 위 판정이 flat / slope / unknown일 때 대체 (§5.1) | `slippery_gait` (trot) | 기하 판정 그대로 |

히스테리시스: trot -> crawl은 crawl 표가 2번 연속 필요하고, crawl -> trot은 모든 문턱값에 0.75를 곱해(더
엄격하게) 평가한 trot 표가 1.5 s 동안 이어져야 합니다. 그래서 경계에 있는 지형에서 보행이 깜박이지 않습니다
(시계가 되돌아가면 유지 타이머를 다시 시작). `/terrain/info` JSON:
`{roughness, max_step, slope_deg, class, recommended_gait, ahead_clear,
known_fraction, incline_deg, height_change, blocked_cells, edges, raw_gait,
slippery, friction_mu, friction_zones, friction_distance, stamp, frame_id, source,
convention}` (NaN -> null).

### 5.1 마찰 사전정보와 `slippery` 클래스

철판(C-OBS4 μ 0.2, F-OBS9 μ 0.15)은 두께가 1 cm라서 깊이 데이터만으로는 바닥과
구분되지 않고 `flat`으로 분류됩니다. 그래서 디지털 트윈의 단일 기준인
`world_layout.yaml`에서 저마찰 구역을 읽어 분류기에 넘깁니다 (`friction_map.py`).

- **구역 출처**: `obstacles` 중 `type: plate`인 항목의 `rect`와 `mu`, 그리고 선택 항목인
  최상위 `friction_zones` 목록(`{id, rect, mu}`, 기름이나 물에 젖은 바닥 등).
  `friction_zone_max_mu`(0.6) 이상인 구역은 보통 바닥으로 보고 버립니다. 좌표는 world(=map)
  프레임이며, 이 트윈의 `/odom`은 world 기준 실제 자세라서 그대로 씁니다.
- **겹침 검사**: 로봇 발자국(`friction_rear` 0.35 m 뒤까지)과 진행 방향 코리도(`friction_far`
  1.0 m 앞, 폭 ±`friction_half_width` 0.30 m)로 이루어진 회전 사각형을 구역 사각형과
  분리축 정리(SAT)로 검사합니다. 겹치는 구역 중 가장 낮은 μ가 `slippery_mu`(0.35)보다 작으면
  미끄러운 지형입니다.
- **클래스 결정**: 미끄러운 구역은 `flat`·`slope`·`unknown`을 `slippery`로 바꾸고 권장 보행은
  `slippery_gait`(기본 trot)입니다. `unknown`도 바꾸는 이유는, 철판이 평평하다는 것을 레이아웃이
  이미 알고 있고 반짝이는 철판에서 깊이가 비는 것은 지형 위험이 아니기 때문입니다. `rough`·
  `stairs`·`obstacle`은 기하 정보를 우선해 자기 클래스와 보행(crawl)을 유지하되,
  `slippery: true`는 그대로 보고합니다.
- **권장 보행이 crawl이 아닌 이유**: MuJoCo 검증에서 μ 0.15 철판 위 crawl은 시험한 모든 변형에서
  넘어졌습니다(커밋된 best effort (i) crawl 강제, 시드 0: 넘어짐 2회). crawl의 무게중심 sway에 필요한 횡가속도(2–3.6 m/s²)가 μg = 1.5 m/s²보다 크기
  때문입니다. 반면 느린 trot에 슬립 모드를 더하면 3/3 통과했습니다
  ([README §3.8](README.md#38-미끄러운-철판-마찰-사전정보--슬립-감지)).
- **히스테리시스**: 구역이 겹치면 바로 `slippery`가 되고, 겹침이 끝난 뒤에도
  `slippery_hold_time`(1.0 s) 동안 유지합니다. 시계가 되돌아가면(시뮬레이션 리셋) 유지 상태를
  지웁니다. `slippery_gait: crawl`로 바꾸면 기존 보행 히스테리시스(crawl 2표 확인)를 그대로 거칩니다.
- **`/terrain/info` 추가 필드**: `slippery`(bool), `friction_mu`(겹친 구역의 최저 μ, 없으면
  null), `friction_zones`(구역 id 목록), `friction_distance`(가장 가까운 구역까지 앞쪽 거리 [m],
  위에 서 있으면 0, 없으면 null). 컨트롤러는 `slippery`/`friction_mu`로 슬립 모드에 들어갑니다.

## 6. 노드 `terrain_mapper`

```
ros2 run module1_locomotion terrain_mapper --ros-args --params-file \
    $(ros2 pkg prefix module1_locomotion)/share/module1_locomotion/config/terrain.yaml -p use_sim_time:=true
```

파라미터(모두 `config/terrain.yaml`): 토픽 이름, `frame_id`(odom), `publish_rate`(5 Hz),
`image_fallback_timeout`, `cloud_convention`, `filter.*`, `map.*`(모든 `ElevationMapParams` 필드),
`classifier.*`(모든 `ClassifierParams` 필드), 마찰 사전정보용 `layout_file`('' = 설치된
plant_simulation의 `config/world_layout.yaml`, launch 인자 `layout_file`로 전달),
`friction_prior`(false면 끔), `friction_zone_max_mu`. 레이아웃을 읽지 못하면 경고만 남기고
사전정보 없이 동작합니다 (컨트롤러의 온라인 슬립 감지는 그대로 동작). YAML에 잘못된 숫자 타입(예:
`6.0` 대신 `6`)이 있으면 동적 타입으로 다시 선언하고 변환하므로 치명적 오류가 되지 않습니다.
QoS: 센서 입력은 best-effort(`qos_profile_sensor_data`), 출력은 reliable/volatile(best-effort와 reliable
구독자 모두와 호환)입니다. 10 s마다 진단 줄에 클라우드 / 영상 수, 초당 점 수, 판별된 좌표 규약, 처리
시간, 관측 비율, 클래스를 남깁니다.

## 7. 키보드 텔레옵 (`keyboard_teleop`)

```
ros2 run module1_locomotion keyboard_teleop                     # 실제 터미널에서
ros2 run module1_locomotion keyboard_teleop --ros-args -p mode:=latch
ros2 launch module1_locomotion locomotion.launch.py teleop_terminal:=xterm
```

키맵(DESIGN_SPEC §7, 영문자는 대소문자를 구분하지 않음):

| 키 | 동작 |
|---|---|
| `w` / `s` | 전진 / 후진 |
| `a` / `d` | 좌회전 / 우회전 |
| `q` / `e` | 왼쪽 / 오른쪽 게걸음 |
| 화살표 | `w`/`s`/`a`/`d`와 같음 |
| `x` | 정지(속도 0) |
| `space` | E-STOP 토글(estop / release) |
| `+` / `-` (`=`와 `_`도 가능) | 속도 스케일 |
| `?` | 도움말 |
| `1` / `2` / `3` / `4` / `5` | stand / sit / trot / crawl / 자동 보행 |
| `r` / `f` / `0` | recover / 밀기 시험(넘어짐) / 자세 리셋 |
| `v` | 보행 모드 stable <-> MPC (`mode:toggle`, 서 있을 때 바뀜. 걷는 중이면 멈춘 뒤 바꿈, README §3.9). MPC는 경사로·긴 경로에서 stable보다 약함(README §4.8) |
| `m` | 수동 오버라이드 토글(다시 누를 때까지 텔레옵이 우선) |
| `p` / `g` / `h` / `t` / `o` | 순찰 / 가스 탐색 / 귀환 / 전체 미션 시작 / 일시정지·재개 |
| `b` / `n` | 배터리 시험 -> 19 % / 배터리 100 % |
| `ESC` / `Ctrl-C` | 종료 (Gazebo 창에서 ESC는 정지이며 텔레옵을 끝내지 않음) |

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/cmd_vel/teleop` | Twist | 주행 중, 수동 오버라이드 또는 E-STOP 동안 20 Hz. 멈출 때 0 twist를 정확히 한 번 보내고 이후에는 발행하지 않음(mux는 0.5 s 뒤 다른 입력으로 돌아감) |
| `/teleop/active` | Bool (transient local) | 수동 오버라이드 플래그, 바뀔 때 + 2 Hz 하트비트 |
| `/locomotion/command` | String | `stand sit gait:trot gait:crawl gait:auto mode:toggle recover push reset_pose estop release` |
| `/mission/command` | String | `patrol seek rth start pause resume` (`o`는 마지막 `/mission/state`에 따라 토글) |
| `/battery/set_level` | Float32 | 퍼센트 (19 / 100) |
| 구독 | `/keyboard/keypress` (Int32 Qt 코드, Gazebo KeyPublisher), `/locomotion/state`, `/battery/state`, `/mission/state` | 상태 줄 |

속도 모드(`mode` 파라미터):
* **hold**(기본): 이동 키는 해당 축에 `+-max * speed_scale`을 명령합니다. 키 이벤트(OS 자동 반복)마다
  갱신되고, 마지막 이벤트 후 `hold_timeout` = 0.6 s가 지나면 축이 0이 됩니다. 0.6 s는 자동 반복
  지연(첫 반복까지 250-600 ms)을 메우므로, 키를 누르고 있으면 부드럽게 주행하고 떼면 0.6 s 안에 멈춥니다.
* **latch**: 키를 누를 때마다 축 목표가 스케일된 최댓값의 `1/latch_steps`만큼 움직이고(클램프), 바꾸거나
  `x`를 누를 때까지 유지됩니다.

스케일 1.0에서 최대 속도는 0.5 m/s, 0.3 m/s, 0.8 rad/s입니다(컨트롤러가 보행별로 한 번 더 제한).
초기 스케일 0.5, 단계 0.1, 범위 0.1-1.0입니다.

터미널 처리: termios 비정규 모드, 에코 없음, 시그널 키 없음(Ctrl-C는 바이트로 들어옴 -> 0 twist -> 정상
종료). 화살표 키는 ANSI 이스케이프 시퀀스에서 디코딩하고, 단독 ESC는 50 ms 뒤에 인식합니다. 터미널 모드는
모든 종료 경로(종료 키, 예외, SIGTERM / SIGHUP, `atexit`)에서 복원합니다. rclpy는 자체 시그널 핸들러 없이
초기화하므로(`SignalHandlerOptions.NO`, 지원될 때) `ros2 launch` / Ctrl-C의 SIGINT가 컨텍스트를 먼저
종료하지 않습니다. 그래서 종료 경로에서도 마지막 0 twist와 `/teleop/active = false`를 발행합니다(그렇지
않으면 latched 수동 오버라이드 플래그 때문에 mux가 텔레옵에 머묾). 시작할 때(그리고 `?`를 누를 때) 도움말을
출력하고, 실시간 상태 줄에 모드, 속도 스케일, 현재 명령, 수동 오버라이드 / E-STOP, `/locomotion/state`의
모드와 보행, 배터리 %, 미션 상태를 보여 줍니다. `/mission/state`가 5 s 넘게 오지 않으면 미션 상태 뒤에
`(stale Ns)`를 붙이고(예: `mission PATROL(stale 12s)`), 그 상태에서 미션 키(`t` 등)를 누르면 "mission_manager가
응답하지 않음, bringup 터미널 확인" 경고를 출력합니다. 2026-10-26 이전 빌드에서는 공장 입구에서 미션 관리자가 rclpy 로거
예외(`Logger severity cannot be changed between calls`)로 죽어 이 표시가 나왔습니다(루트 README FAQ Q9, 고침). stdin이 TTY가 아니면(터미널 prefix 없이 `ros2 launch`로
시작) 노드는 "Gazebo 키 전용" 모드로 동작합니다. Gazebo 창을 클릭하고 같은 키를 쓰면 됩니다(KeyPublisher
GUI 플러그인이 `/keyboard/keypress`에 Qt 키 코드를 발행하고 `plant_simulation`이 브리지합니다).

## 8. Launch

```
ros2 launch module1_locomotion locomotion.launch.py                      # mapper + controller
ros2 launch module1_locomotion locomotion.launch.py teleop_terminal:=xterm
ros2 launch module1_locomotion locomotion.launch.py teleop_terminal:=gz   # TTY 없는 텔레옵
```
인자: `use_sim_time`(true), `start_terrain`, `start_controller`, `terrain_config`, `controller_config`,
`teleop_config`, `layout_file`('' = 설치된 `plant_simulation/config/world_layout.yaml`, 마찰 사전정보용으로
`terrain_mapper`에 전달. `bringup.launch.py`는 자기 `layout_file`을 넘김), `teleop_terminal`
(`none` | `xterm` -> prefix `xterm -e` | `gnome-terminal` | `konsole` | `gz` | `"tilix -e"` 같은 임의
prefix), `teleop_mode`(hold | latch). 컨트롤러는 실행 파일 `locomotion_controller`, 노드 이름
`locomotion_controller`, 파라미터 `config/locomotion.yaml`로 시작합니다.

## 9. 평가 (`tools/eval_terrain.py`)

합성 장면 7개를 씁니다: 평지, 흩어진 배관(지름 8-12 cm), 잔해(2-7 cm), 0.17 m x 0.30 m 계단 5단 상행,
같은 계단 하행, 16 deg 경사로, 0.6 m 장애물. 로봇이 지형 쪽으로 걸어가며, 10 Hz 프레임마다 실제 깊이
카메라 모델을 5 mm 정답 높이장에 광선 투사하고, 거리 의존 노이즈(2 mm + 2 mm/m^2), 0.2 % 가까운 가짜
반사, 1 % 누락을 더합니다. 이것을 Gazebo와 같은 `x y z rgb` PointCloud2로 묶어, 노이즈가 있는 오도메트리
자세(3 mm, 0.2 deg)로 노드 파이프라인을 그대로 돌립니다. 결과(`docs/results/terrain_eval.json`, 시드 0):

| 장면 | 모서리 밖 elevation RMSE | 관측 범위 | 판정 위치의 클래스 / 보행 |
|---|---|---|---|
| 평지 | 0.20 cm | 100 % | flat / trot |
| 배관 | 0.52 cm | 98 % | rough / crawl |
| 잔해 | 0.34 cm | 100 % | rough / crawl |
| 상행 계단 | 0.95 cm | 86 % | stairs / crawl |
| 하행 계단 | 1.73 cm | 100 % | stairs / crawl (판정 시 첫 모서리가 0.78 m 앞) |
| 경사로 16 deg | 0.43 cm | 98 % | slope / crawl |
| 장애물 0.6 m | 4.8 cm (윗면이 보이지 않아 하한) | 77 % | obstacle / crawl |

* 판정 위치(지형 0.8-0.9 m 앞)에서 분류 7/7, 보행 7/7 정답. 하행 계단의 낙차는 약 0.8 m까지 모서리 뒤에
  가려집니다.
* 발 중심 5 cm 안에 3 cm 넘는 높이 차가 걸친 착지점 비율, 명목 -> 계획: 상행 계단 25.0 % -> 0.0 %
  (챌판까지 최소 계획 거리 6 cm), 하행 계단 26.3 % -> 0.8 %(5 cm), 배관 29.0 % -> 3.1 %, 잔해
  40.7 % -> 8.8 %.
* 클라우드 좌표 규약 판별은 link와 optical 클라우드 모두 183/183 프레임 정답. optical 클라우드로
  파이프라인 전체를 돌려도 같은 맵이 나옵니다.
* 기법 제거 실험: 높은 이상치 확인(1프레임 대신 2프레임)은 잔해에서 5 cm 넘는 가짜 셀을 모두 없앱니다.
  최대 높이 편향은 계단의 모서리 밖 RMSE를 1.53에서 0.95 cm로 줄입니다. 가림 그림자 채움(§3)은 하행
  계단의 위험 계획 착지점을 7.1 %에서 0.8 %로 줄이고(5 cm 넘는 가짜 셀 100 -> 19), 배관 장면 RMSE를
  절반으로 줄이며(1.14 -> 0.52 cm), 상행 계단에는 영향이 없습니다(어느 쪽이든 위험 0 %).
* 처리 시간(단일 스레드 numpy, 다른 작업과 함께 쓴 개발 PC, p10 / 중앙값 / p95, JSON의 `timing_ms`):
  25k 점 클라우드 하나 통합 3.2 / 3.9 / 5.0 ms, 모든 레이어 4.8 / 5.7 / 7.1 ms, 분류
  0.72 / 0.79 / 1.10 ms, 컨트롤러 쪽에서 클라우드로부터 레이어 포함 재구성 8.4 / 11.3 / 14.2 ms, 착지점
  선택 1회 0.30 ms. 10 Hz 카메라 / 5 Hz 발행 예산 안에 충분히 들어옵니다.

그림: `docs/images/terrain_maps.png`(정답 / 맵 / 오차 / 통과성), `terrain_profiles.png`,
`terrain_classification.png`(접근 경로를 따른 클래스와 보행), `terrain_footholds.png`,
`terrain_accuracy.png`.

## 10. 테스트

```
python -m pytest module1_locomotion/test/test_elevation_map.py module1_locomotion/test/test_foothold.py \
                 module1_locomotion/test/test_terrain_classifier.py module1_locomotion/test/test_teleop.py \
                 module1_locomotion/test/test_friction_map.py -q
```
마찰 사전정보 테스트는 다음을 확인합니다. 실제 `world_layout.yaml`에서 철판 두 장(C-OBS4,
F-OBS9)을 읽는지, `friction_zones` 목록과 μ 문턱값, 잘못된 입력 처리, 분리축 겹침 검사,
진행 방향 코리도(뒤쪽 발자국, 회전, 대각 접근, 거리), 분류기의 `slippery` 클래스와 필드,
기하 클래스 우선, 유지 시간 히스테리시스와 시계 리셋, `slippery_gait: crawl`의 보행 투표,
그리고 fake rclpy 위에서 `layout_file`/`friction_prior` 파라미터를 받는 `terrain_mapper` 노드입니다.

그 밖에 다음을 다룹니다: 융합 수식(이론 대비 칼만 가중치, 거리 의존 분산, 게이팅과 이상치 연속, 최대 높이
편향, 확인, 망각, 재중심), 가림 그림자 규칙과 셀별 시점 기록을 포함한 구멍 채움, 레이어(Sobel 동등성,
전수 최소제곱 대비 평면 맞춤, 경사로 기울기, step / roughness / traversability), 전송 왕복, PointCloud2
파싱(엔디언, datatype, 패딩), 좌표 규약 판별, 필터, 자세 보간, 깊이 카메라를 시뮬레이션한 종단 간 장면,
착지점 선택(모서리, 장애물, 배관, 잔해, 미지 셀), 히스테리시스를 포함한 모든 장면 유형의 분류(관측이 부족한
코리도에서의 양성 증거 포함), 텔레옵 로직(Qt 코드, 이스케이프 시퀀스, hold / latch, 스트리밍, 상태 줄,
YAML / launch 일관성, 시그널 핸들러 없는 rclpy 초기화), 그리고 fake rclpy 계층 위에서 실행한 두 노드
래퍼(토픽, 메시지 필드, 파라미터 변환, 영상 대체 경로, optical 클라우드, 키 -> 토픽).

## 11. 한계

* 2.5-D: 셀마다 높이 하나만 저장합니다. base 위 0.55 m를 넘는 돌출부는 버리므로, 로봇이 그 아래로
  지나간다고 가정합니다.
* 높은 장애물: 카메라 시야보다 높은 물체의 윗면은 관측되지 않습니다. 맵은 하한을 저장하며, 이 값도
  장애물 / 통과 불가로 분류됩니다.
* 맵은 ray-free입니다: 비게 된 셀(옮겨진 물체)은 새 표면이 관측될 때(낮은 이상치는 3프레임 연속 일관되면
  셀을 교체)나 분산 증가로 잊힐 때만 고쳐집니다.
