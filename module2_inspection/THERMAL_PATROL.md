# Module 2 - 열화상/RGB 퓨전, 합성 열화상 카메라, 뷰포인트 순찰 계획

이 문서는 점검 모듈의 열화상 부분과 순찰 경로 계획기를 다룹니다(아날로그 게이지 판독기는 별도
문서에 정리되어 있습니다). 모든 알고리즘은 ROS에 의존하지 않는 모듈에 들어 있고 Windows에서 단위
테스트합니다. ROS 2 노드는 얇은 래퍼입니다.

| 파일 | 역할 |
|---|---|
| `module2_inspection/thermal_fusion.py` | 복사 측정(radiometry), 평면 유도 호모그래피, ECC 미세 정합, 분할, 위치 추정, 시간 확인, 시각화, 시간 동기화 |
| `module2_inspection/synthetic_thermal.py` | 합성 복사 측정 열화상 카메라(광선 투사 + z-버퍼 스플래팅)와 벤치마크용 RGB 대용 렌더러 |
| `module2_inspection/viewpoint_planner.py` | 후보 뷰포인트, 판독성 점수, A* 길이 기반 Held-Karp TSP, 경로 조립 |
| `thermal_fusion_node.py`, `synthetic_thermal_node.py`, `patrol_planner_node.py` | ROS 2 래퍼 (실행 파일 `thermal_fusion`, `synthetic_thermal_camera`, `patrol_planner`) |
| `config/thermal.yaml`, `config/patrol.yaml` | 모든 튜닝 값 (ROS 파라미터) |
| `launch/inspection.launch.py` | 모듈 실행 |
| `tools/eval_thermal.py`, `tools/eval_patrol.py` | 오프라인 벤치마크 -> `docs/results/*.json`, `docs/images/thermal_*.png`, `docs/images/patrol_*.png` |

```
 /thermal_camera/image (mono16, K x 100) --+
 /inspection_camera/image (rgb8) ----------+--> thermal_fusion --> /inspection/thermal_alerts (JSON)
 /odom ------------------------------------+                   --> /inspection/thermal_fusion (bgr8)
        (thermal_source:=synthetic:  /odom --> synthetic_thermal_camera --> /thermal_camera/image)

 plant_map.yaml | /map, world_layout.yaml --> patrol_planner --> /patrol/route (Path, latched)
 /patrol/replan (Empty) ------------------------^              --> /patrol/plan_json (JSON, latched)
                                                               --> /patrol/markers (MarkerArray, latched)
```

---------------------------------------------------------------------------

## 1. 열화상 / RGB 퓨전 (`thermal_fusion.py`)

### 1.1 복사 측정 (radiometry)
Gazebo 열화상 센서는 L16 영상을 발행합니다. ThermalSensor `resolution`이 0.01이면 온도는
`K = count x 0.01`입니다. `raw_to_kelvin()`은 브리지가 낼 수 있는 모든 인코딩을 받습니다.
`mono16`, `16UC1`, `L16`(카운트), `mono8`/`L8`(센서 범위 253.15 - 655 K에서 선형), `32FC1`
(Kelvin)입니다. 카운트 0은 무효 픽셀(NaN)입니다. 모든 임계값은 파라미터로 덮어쓰지 않는 한
`world_layout.yaml`(`thermal.ambient_k / warning_k / alarm_k` = 298.15 / 333.15 / 353.15 K)에서
가져옵니다.

### 1.2 카메라와 좌표계
두 카메라 모두 Gazebo 규약(링크 좌표계 x 전방, y 왼쪽, z 위)을 따르고, 내부 파라미터는 광학
좌표계(z 전방, x 오른쪽, y 아래)를 씁니다. `p_link = R_LO p_opt`,
`R_LO = [[0,0,1],[-1,0,0],[0,-1,0]]`입니다.

| | 해상도 | hfov | f (px) | base_link 기준 자세 (x y z r p y) |
|---|---|---|---|---|
| 점검 (RGB) | 800 x 600 | 0.90 | 828.06 | 0.30 0.00 0.20 0 -0.30 0 |
| 열화상 | 320 x 240 | 0.80 | 378.44 | 0.30 -0.05 0.20 0 -0.30 0 |

Pitch -0.30 rad는 두 카메라를 **위로** 17.2 deg 기울입니다. RGB 광학 좌표계에서 본 열화상 광학
좌표계의 상대 자세는 `R = I`, `t = (0.05, 0, 0)` m이며, 순수한 수평 기선 5 cm입니다.

### 1.3 평면 유도 호모그래피
열화상 광학 좌표계에서 `n^T X + d = 0`으로 쓴 장면 평면(단위 법선 `n`은 카메라 쪽을 향하고,
`d > 0`은 평면까지의 거리)에서, 평면 위의 모든 점은 `-n^T X / d = 1`을 만족합니다.
`X_rgb = R X + t`를 쓰면 다음과 같습니다.

```
X_rgb = R X + t (-n^T X / d) = (R - t n^T / d) X
x_rgb ~ K_rgb X_rgb ,  X ~ K_th^-1 x_th
=>  H = K_rgb (R - t n^T / d) K_th^-1          (x_rgb ~ H x_th)
```

`CameraRig.homography()`가 정확히 이것을 구현합니다(`H[2,2] = 1`로 정규화). 평면 모델은 두 가지가
있습니다(`plane_mode`).

* `fronto`: `n = (0, 0, -1)`, `d = Z` (광축에 수직인 평면).
* `vertical` (기본값): 카메라를 향한 수직 평면입니다. 위로 기울어진 카메라가 보는 파이프 랙,
  기계 면, 벽에 자연스러운 모델입니다. 법선은 카메라 전방 축의 수평 투영을 뒤집은 것이며, 현재
  로봇 자세(attitude)로 광학 좌표계에 변환합니다. 평면은 선택한 3-D 점을 지나고 `d = -n^T P`입니다.

**시차 오차.** 깊이 `Z`인 점을 `d`에 있는 평면으로 사상하면 오차는 순수하게 수평 방향이며
(에피폴이 무한대) `e = f_rgb * b * |1/Z - 1/d| = 41.4 px * |1/Z - 1/d|`입니다. 예를 들어
`Z = 1 m, d = 2 m`이면 20.7 px입니다. 따라서 평면 깊이가 무엇보다 중요합니다. 평면
선택(`estimate_plane`)의 우선순위는 다음과 같습니다.

1. **매칭된 열점** - 레이아웃 열점과 연관된 가장 뜨거운 블롭에 대해, 그 열점의 레이아웃 위치를
   지나는 평면(열점에서 정확함).
2. **레이아웃 중앙값** - 열화상 FOV 안에 있는 레이아웃 열 객체의 가중 중앙값 깊이(열점 가중치 3,
   파이프 랙 / 스텁 샘플 가중치 1).
3. 자세가 없으면 **고정** `plane_distance`(2 m).

*깊이 카메라를 쓰지 않는 이유 (DESIGN_SPEC에는 "depth image median"이 있음)?* 깊이 카메라는 발
앞 지형을 매핑하려고 0.60 rad **아래로** 기울어져 있고 범위가 4 m입니다. 반면 열화상 카메라는
1 - 15 m 떨어진 배관과 기계를 보려고 0.30 rad **위로** 향합니다. 두 시야가 거의 겹치지 않으므로
중앙값 깊이는 바닥이 됩니다. 레이아웃(모든 열 객체의 정답)과 로봇 자세를 쓰면 훨씬 나은 평면을,
매칭된 열점을 쓰면 정확한 평면을 얻습니다.

또한 레이아웃과 매칭된 검출은 각각 **자기 위치**를 지나는 평면의 호모그래피를 RGB 바운딩 박스와
피크 마커에 씁니다. 그래서 깊이가 다른 물체가 한 프레임에 함께 있어도 박스가 정확합니다.

### 1.4 ECC 미세 정합
`refine_homography_ecc()`는 *엣지 영상*(99 백분위수로 정규화하고 가우시안 블러를 적용한 Sobel
크기)을 정렬합니다. 열화상과 가시광의 강도는 서로 무관하지만 물체 윤곽은 일치하기 때문입니다.
열화상 엣지 영상을 `H0`로 RGB 프레임에 워핑하고, `cv2.findTransformECC`(RGB 배율 0.25와 0.5에서
거친 단계부터 세밀한 단계로, 기본은 MOTION_TRANSLATION, 마스크 = 유효 겹침 영역)가
`inp(W x) ~ tmpl(x)`인 `W`를 돌려주므로 `H = S^-1 W^-1 S H0`입니다.

타당성 한계(거부되면 `H0` 유지): 상관 >= 0.2, |이동| <= 30 px, 기선을 가로지르는 성분 <= 8 px
(깊이 오차는 점을 에피폴라 방향으로만 움직일 수 있음), 선형 부분이 항등 행렬의 6 % 이내(아핀
모드), 충분한 열 구조(p99 기울기 >= 0.4 K/px)와 겹침. 에피폴라 제약(`ecc_epipolar`)을 쓰면 병진의
수평 성분만 남깁니다. 채택된 보정은 평활하고(EMA 0.5), 평면 깊이가 25 % 넘게 바뀌면 초기화합니다.
ECC는 `ecc_every_n` 프레임마다 실행하고, 평면이 이미 매칭된 열점을 지나면 건너뜁니다
(`ecc_skip_hotspot_plane`). 벤치마크에 따르면 ECC는 근사 평면의 큰 오차를 줄이지만 정확한
평면에는 노이즈만 더합니다(4절).

### 1.5 열점 분할
`segment_hotspots()`: `T >= segmentation_k`(기본값 `warning_k`), 선택적 모폴로지 열림, 8-연결
성분(`connectedComponentsWithStats`), 면적 >= `min_area_px`. 블롭마다 최대 / 평균 온도, 온도 가중
중심(가중치 `T - threshold`. 그래서 광선이 가장 뜨거운 부분을 지남), 피크 픽셀, 바운딩 박스를
구합니다. 같은 열점과 연관된 여러 블롭(가림막에 의해 쪼개진 것)은 병합합니다. 블롭을 만들지
못한(임계값 미만) 보이는 레이아웃 열점은 투영 위치 주변 창에서 측정하고
(`report_normal_in_view`), 창 안에 주변 온도보다 `roi_min_excess_k` 이상 따뜻한 것이 있으면
상태 `normal`로 보고합니다(없으면 가려진 것으로 보고 보고하지 않음). ROI 보고는 창에 픽셀이
충분히 들어오는 `roi_max_range_m`(8 m) 이내로 제한합니다.

### 1.6 위치 추정
로봇 자세(`/odom`, 열화상 타임스탬프에서 보간)와 열화상 외부 파라미터로 `T_world_th`를 구합니다.
블롭 중심을 지나는 광선은 `o + s d`입니다. 각 레이아웃 열점 `p`에 대해 점-광선 거리는
`dist = |(p - o) - ((p - o).d) d|`입니다(`s > 0`만 셈). `dist / tol`이 가장 작은 열점을
`dist < tol`이면 채택하며, `tol = match_tolerance_m + match_tolerance_rel * s + extent`(extent =
프리미티브의 경계 반지름)입니다. 채택된 검출은 레이아웃 `id`, `tag`, `position`을 갖습니다.
광선을 따라 `max_match_range_m`(30 m) 이내의 열점만 후보입니다. 그 너머에서는 허용 오차
(> 1.7 m)가 랙 플랜지 간격 3 m에 가까워지므로, 먼 블롭(예: 복도 끝 31 m 밖에서 보이는 베어링
H04)은 틀릴 수도 있는 ID를 붙이는 대신 ID 없이 보고합니다. 매칭되지 않은 블롭은 광선이 정합
평면과 만나는 곳에 둡니다(`localization: ray_depth`). 상태 = `Layout.temperature_status(max_temp)`.

### 1.7 시간 확인
`TemporalConfirmer`: 검출은 열화상 프레임 `confirm_frames`(3)개에서 연속으로 나온 뒤에만 발행합니다.
한 프레임 누락(`max_misses`)과 `max_gap_s` 미만의 공백은 허용합니다. 레이아웃 매칭 검출은 ID로,
익명 블롭은 월드 거리(자세가 없으면 픽셀 거리)로 연관합니다.

### 1.8 시간 동기화 (노드)
`StampedBuffer`는 최근 RGB 프레임과 자세를 헤더 타임스탬프 기준으로 보관합니다. 열화상 프레임마다
타임스탬프가 가장 가까운 RGB 프레임(0.15 s 이내)과, 열화상 타임스탬프에서 보간한 자세(최단 호
각도, 0.25 s 이내)를 씁니다. yaw 0.8 rad/s에서 0.1 s가 어긋나면 오버레이가 약 66 px 밀리기
때문입니다. 발행자들의 시간 기준이 일관되지 않으면 노드는 0.5 s 안에 받은 최신 메시지로 대체하고
경고합니다.

### 1.9 시각화 (`/inspection/thermal_fusion`, bgr8, RGB 크기)
고정 스케일(주변 온도 - 5 K ... 경보 + 40 K = 20 - 120 deg C)의 INFERNO 컬러맵을
`alpha = alpha_max * clip((T - (ambient + delta)) / ramp)`로 블렌딩해 따뜻한 영역에만 색을
입힙니다. 경고(노랑)와 경보(빨강) 등온선, 상태별 색의 박스 + 라벨(`H02 99.1C ALARM`), 열화상 FOV
외곽선, 경고/경보 표시가 있는 수직 온도 스케일 바, 프레임 최고 온도·평면 깊이/출처·ECC 보정을
담은 헤더를 그립니다. RGB가 없으면 같은 크기의 열화상 전용 화면을 발행합니다.

### 1.10 `/inspection/thermal_alerts`
```json
{"stamp": 12.0,
 "hotspots": [{"id": "H02", "tag": "PIPE-J2", "max_temp_c": 99.09, "mean_temp_c": 89.7,
               "position": [5.0, 6.4, 1.0], "status": "alarm", "bbox": [131, 98, 34, 71],
               "bbox_rgb": [359, 251, 77, 157], "area_px": 2065, "source": "blob",
               "localization": "layout", "range_m": 1.762, "match_error_m": 0.006,
               "confirmed_frames": 3}],
 "n_candidates": 1, "frame_max_temp_c": 99.09, "thresholds_c": {"warning": 60.0, "alarm": 80.0},
 "registration": {"plane_depth_m": 1.762, "depth_source": "hotspot:H02", "plane_mode": "vertical",
                  "ecc_ok": null, "ecc_cc": null, "ecc_shift_px": null},
 "robot_pose": [5.0, 4.4, 1.5708], "rgb_available": true}
```
`bbox`는 **열화상** 영상 픽셀(값을 측정한 복사 측정 영상) 기준이고, `bbox_rgb`는 같은 박스를
RGB / 퓨전 영상 기준으로 나타낸 것입니다. `source` = `blob`(분할됨, >= 경고) 또는 `roi`(임계값
미만인 보이는 레이아웃 열점, 상태 `normal`)입니다.

---------------------------------------------------------------------------

## 2. 합성 열화상 카메라 (`synthetic_thermal.py`)

Gazebo 열화상 센서의 대체 수단(`thermal_source:=synthetic`)이자 벤치마크의 영상 소스입니다.
출력 계약은 Gazebo와 같습니다. 320 x 240, hfov 0.8, `mono16`, `K = count x 0.01`, 5 Hz이고
`/odom`으로 렌더링합니다. 시뮬레이션이 Gazebo 센서 없이 실행될 때만 그것을 대체합니다
(`bringup.launch.py thermal_source:=synthetic`, 또는 `sim.launch.py enable_thermal:=false`).
노드는 자기 출력 토픽도 구독해서, 자기가 발행하지 않은 프레임(페이로드 CRC32를 마지막 50프레임과
비교)이 도착하면, 즉 두 번째 열화상 카메라가 같은 토픽에 발행하면 경고합니다.

**기하** (모두 `world_layout.yaml`에서 가져옴):
* *해석적 픽셀별 광선 투사*로 z-버퍼를 초기화합니다. 구간 선형 바닥(홀, 복도 상층, 내리막,
  하층), 동쪽 개구부를 포털로 둔 홀과 복도의 벽과 천장, 기계(레이아웃 풋프린트로 만든 박스, 높이
  3.1 / 6.6 m), 파이프 랙 지지대, 팔레트, 플랫폼(방향 박스, slab 검사), 드럼(수직 원통)이
  대상입니다. 박스는 투영된 화면 사각형 안의 광선과만 교차 검사합니다.
* 곡면이면서 열적으로 중요한 기하에는 *샘플링한 3-D 표면점의 z-버퍼 스플래팅*을 씁니다. 파이프 랙
  배관, 열점 프리미티브(플랜지, 베어링), 복도 배관 스텁, 바닥에 놓인 잔해 배관, 게이지 다이얼이
  대상입니다. 원통은 0.25 m 조각으로 자르고, 조각마다 절두체 컬링을 한 뒤 간격이
  `0.9 px * depth / f`(최소 2 mm)인 `(s, theta)` 격자로 샘플링합니다. 뒷면은 컬링하고 끝 뚜껑은
  고리 위에서 샘플링합니다. 각 샘플은 `k x k` 정사각형으로 스플래팅하며
  `k = ceil(spacing * f / depth + 0.5)`입니다. 픽셀마다 가장 가까운 샘플이 해석적 깊이보다 앞에
  있으면 그 샘플이 이깁니다((pixel, depth)에 대한 lexsort). 단위 테스트가 박스-스플랫,
  스플랫-스플랫 가림과 구멍 없는 덮임을 검사합니다.

**온도**
* 열점 프리미티브: `T = T_h - 0.10 (T_h - T_base) rho^2` (rho = 정규화된 축 / 반경 방향 위치,
  `T_base` = 배관 몸체 또는 기계).
* 모든 열점 주변의 헤일로, 프리미티브 표면에서 반경 방향으로 감쇠:
  `dT = (T_h - T_surf) k exp(-dist / lambda)`. 연결된 몸체(배관, 스텁, 기계 하우징: k 0.5,
  lambda 0.10 m)에서는 전도, 그 밖에서는 복사(k 0.08, lambda 0.30 m).
* 배관 몸체 `pipe_body_k`(308 K) +- 0.4 K, 기계 303 K(수직 기울기와 텍스처),
  바닥은 주변 온도 - 1 K(약한 기울기), 벽은 성층(+0.3 K/m), 천장은 주변 온도 + 1.8 K.
* 방향 방사율: `T_app = T_amb + (1 - (1 - cos theta)^5)(T - T_amb)`.

**센서 모델**: 가우시안 PSF(sigma 0.6 px), NETD 백색 노이즈 0.05 K, 열(column) 고정 패턴 노이즈
0.03 K, 행 노이즈 0.02 K, 0.01 K 양자화, 253 - 655 K로 제한. 렌더링 시간은 프레임당 중앙값
40.1 ms(`docs/results/thermal_eval.json`의 `timing_ms`, Windows 개발 PC)로 5 Hz 예산 안에 충분히
들어갑니다.

`render_visible_proxy()`는 같은 장면을 점검 카메라 시점으로 렌더링합니다(표면 종류별 알베도,
Lambert 셰이딩, 바닥 타일과 패널 이음매, 노이즈). 벤치마크에서 ECC에 사실적인 목표 영상을 주는
데만 씁니다.

---------------------------------------------------------------------------

## 3. 뷰포인트 기반 순찰 계획 (`viewpoint_planner.py`)

### 3.1 후보
게이지마다(중심 `g`, 단위 법선 `n`, 다이얼 직경 0.30 m) 카메라-다이얼 수평 거리
`r = 0.9 ... 2.2 m`(0.1 간격)의 고리와, 수평 법선 기준 방위 오프셋 `phi = -40 ... +40 deg`
(5 간격)를 씁니다. base 위치는 `b = g_xy + (r + 0.30) (cos(alpha+phi), sin(alpha+phi))`이고
(카메라가 base보다 0.30 m 앞에 있음), yaw는 다이얼을 향하도록
`yaw = atan2(g - b) - asin(c_y / |g - b|)`입니다(c_y = 카메라 측면 오프셋, 여기서는 0).
게이지당 후보는 14 x 17 = 238개입니다.

### 3.2 판독성 점수
카메라 자세는 `camera_world_pose((x, y, floor_z(x, y) + 0.34, 0, 0, yaw))`이며, 카메라 높이는
바닥 높이를 따라갑니다(복도 하층 -0.88 m). 다이얼 테두리(24점)와 중심을
`dusan_common.camera.PinholeCamera`로 투영합니다.

필수 조건(하나라도 어기면 후보 무효, 점수 0): 로봇 반지름 0.30 m로 자유 공간일 것, 도크가 속한
자유 공간 성분(팽창 맵의 연결 성분) 안에 있을 것, 내비 맵 위에서 카메라부터 다이얼까지 가시선이
있을 것(마지막 0.35 m, 즉 게이지 스탠드는 무시), 12 px 여백을 두고 다이얼 전체가 영상 안에 있을
것, 입사각 <= 60 deg일 것.

[0, 1] 범위의 소프트 항:

| 항 | 정의 | 가중치 |
|---|---|---|
| `f_px` | smoothstep((D_px - 60) / (170 - 60)), D_px = 투영된 테두리의 장축 | 0.35 |
| `f_inc` | (cos theta - cos 60deg) / (1 - cos 60deg), theta = 다이얼 법선과 카메라 방향 사이의 각 (3-D) | 0.25 |
| `f_vert` | 1 - (v_c / c_y)^2, v_c = 주점에서 다이얼 중심까지의 수직 편차 (pitch -0.30 때문에 높거나 낮은 다이얼은 프레임 위아래로 벗어남) | 0.25 |
| `f_clr` | clip((clearance - 0.30) / (0.8 - 0.30)) | 0.15 |

`score = sum(w_i f_i) / sum(w_i) - 0.10 [base가 회피 구역 안에 있을 때, 기본 Z-HOT]`. 가장 좋은
후보(동점이면 거리가 짧은 쪽)가 그 게이지의 뷰포인트입니다. 유효 후보가 하나도 없는 게이지는
완화된 2차 패스(r <= 3.0 m, |phi| <= 60 deg)를 거치고, 그래도 없으면 `unreachable`로 보고합니다.

### 3.3 방문 순서: Held-Karp
노드 = 시작점(도크 자세 `Layout.dock_pose`) + 최적 뷰포인트들입니다. 간선 비용 = 모든 쌍에 대한
`dusan_common.planning.plan()`(여유 비용과 가시선 평활을 쓰는 8-연결 A*, 로봇 반지름 0.30 m)의
경로 길이입니다(경로는 캐시하고, 반대 방향에는 뒤집어서 씀). 게이지 노드의 부분집합 `S`에 대한
Held-Karp 동적계획법으로 정확한 최적해를 구합니다.

A*에는 **지형 비용**(`dusan_common/terrain_cost.py`, `terrain_cost_enabled` 기본 true)이 들어갑니다. 내비 맵에서
자유 칸인 잔해를 격자 비용으로 그립니다. 로봇 중심이 누운 배관·잔해 지대에서 0.12 m 안이면 칸마다 25, 철판 1, 낮은 단 2이고,
그 바깥 0.25 m 여유 폭은 그 값의 0.15배입니다. 여유 폭은 0.25 m를 넘지 않습니다. C-OBS1 배관 북쪽 끝과 벽 사이에서 로봇
중심이 지날 수 있는 띠가 약 0.3 m뿐이라, 더 넓히면 그 띠가 비용 구역에 덮여 경로가 다시 배관을 넘습니다. 가시선
평활(`smooth_path`)도 같은 비용을 봅니다. 지름길이 A* 경로가 지나간 칸보다 비싼 칸에 들어가거나 비용 합이 더 크면
쓰지 않으므로, 평활 뒤에도 배관 끝이나 잔해 모서리를 가로지르지 않습니다. 비용은 유한해서 돌아갈 길이 없는 잔해
(C-OBS4 철판, C-OBS5 배관)는 그대로 지나갑니다. 간선 비용(거리 행렬)은 이렇게 찾은 경로의 실제 길이입니다.

```
C({j}, j)      = D(0, j)
C(S + {k}, k)  = min_{j in S} C(S, j) + D(j, k)          k not in S
closed tour    = min_j C(ALL, j) + D(j, 0)      open route = min_j C(ALL, j)
```

시간 `O(2^(n-1) (n-1)^2)`, 메모리 `O(2^(n-1) (n-1))`입니다. 노드 13개(도크 + 게이지
12개)는 < 0.5 s가 걸리고, 그보다 큰 문제는 최근접 이웃 + 2-opt로 대체합니다. 단위 테스트가 무작위
대칭·비대칭 문제에서 전수 탐색과 비교합니다. 최적 순서의 구간(leg)을 이어 붙여 경로를 만들고,
비교를 위해 레이아웃 순서, 탐욕적 최근접 이웃, 무작위 순서를 같은 거리 행렬로 평가합니다.

### 3.4 출력
`/patrol/plan_json` (latched):
```json
{"start": [61.9, 0.0], "start_yaw": 3.1416,
 "viewpoints": [{"gauge_id": "G08", "tag": "CORR-PI-02", "x": 57.0, "y": 0.45, "yaw": -1.5708,
                 "score": 0.95, "distance": 1.51, "incidence_deg": 21.8, "pixel_diameter": 165.3,
                 "order": 1, "leg_length": 5.15, "...": "..."}],
 "order": ["G08", "G07", "..."], "legs": [[[61.9, 0.0], [57.0, 0.45]], "..."],
 "leg_lengths": [5.15, "..."], "total_length": 145.88, "return_to_dock": true,
 "naive_order": ["G01", "..."], "naive_length": 158.23, "improvement_pct": 7.8,
 "unreachable": [], "method": "held_karp", "compute_s": 1.08,
 "stamp": 0.0, "reason": "startup", "map_source": "file ..."}
```
`viewpoints`는 방문 순서대로이고, `x, y, yaw`는 로봇 **base** 자세, `distance`는 카메라-다이얼
3-D 거리입니다. `legs[i]`는 뷰포인트 `i`에서 끝나며, `return_to_dock`이면 마지막에 도크로 돌아가는
구간이 하나 더 붙습니다. `/patrol/route`는 이어 붙인 경로이고, 각 뷰포인트 자세에는 판독용 yaw가
들어 있습니다. `/patrol/markers`: 경로 선, 뷰포인트 화살표(색 = 점수), 카메라-게이지 연결선, 순서
라벨, 게이지 구(빨강 = 도달 불가), 시작 라벨.

---------------------------------------------------------------------------

## 4. 평가

```
python tools/eval_thermal.py      # 자세 150개 x 센서 조건 3개 x 3프레임 (개발 PC 릴리스 실행: 137 s)
python tools/eval_patrol.py       # 그림 포함 (릴리스 실행: 3 s)
```

두 스크립트 모두 생성된 내비 맵 `simulation/maps/plant_map.yaml`을 씁니다(없으면 레이아웃에서
래스터화한 맵). 아래 수치는 seed 0 기준입니다.

### 4.1 열화상 벤치마크 (`docs/results/thermal_eval.json`)
로봇 자세 150개(열점에서 1.2 - 7 m 떨어져 열점을 바라보는 120개. +-0.22 rad 흔들림과 작은
roll/pitch/높이 노이즈 포함. 여기에 무작위 음성 자세 30개), 자세당 3프레임, 같은 장면을 공유하는
열화상 센서 조건 3가지로 평가했습니다. *nominal*(NETD 0.05 K), *noisy*(NETD 0.4 K, 열 FPN 0.3 K,
행 0.15 K), *defocused*(PSF sigma 1.6 px)입니다. 정답 = 렌더러의 z-버퍼 라벨입니다(픽셀이 6개
이상 보이면 열점으로 셈).

| 조건 | 정밀도 | 재현율 | ID 정확도 | 최고온도 오차 중앙값 / p95 | 상태 정확도 | 음성 프레임 오경보 |
|---|---|---|---|---|---|---|
| nominal | 1.000 | 0.9845 (TP 127, FN 2) | 1.000 | 0.14 / 4.1 K | 1.000 | 0 % |
| noisy | 1.000 | 0.9845 (TP 127, FN 2) | 1.000 | 1.07 / 3.5 K | 1.000 | 0 % |
| defocused | 1.000 | 0.9612 (TP 124, FN 5) | 1.000 | 0.75 / 11.1 K | 1.000 | 0 % |

3프레임 확인 결과는 nominal과 noisy 센서에서 위와 같고(정지 자세), defocused에서는 약간 낮습니다
(재현율 0.9535, TP 123 / FN 6: 흐려진 H03 시야가 임계값 근처에서 깜박임). *대상* 열점의 재현율은
7 m까지 모든 거리 구간에서 1.0입니다. 놓친 것은 일부만 보이는 부차 열점입니다(H03이 10 - 19
픽셀만 보이는데, 최고 68 deg C가 경고 임계보다 8 K만 높아서 PSF 때문에 임계 아래로 내려감). ID
정확도(참양성 중 올바른 레이아웃 ID의 비율)는 세 조건 모두 1.000입니다. `max_match_range_m`보다
먼 베어링은 의도적으로 ID 없이 보고됩니다(1.6절). 최고온도 오차는 얇은 플랜지 H02(폭 0.10 m)를
비스듬히 볼 때 가장 큽니다. 측정 스폿이 PSF보다 작아지는, 열화상 측정의 전형적인 "스폿 크기"
한계이며 디포커스에서 가장 나쁩니다. 시야 안의 임계값 미만 열점(H01, 45 deg C)은 보인 경우의
82 %(27 / 33)에서 ROI로 `normal` 보고되었습니다(나머지는 대부분 가려졌거나 `roi_max_range_m`보다
멀었음).

열점 표면점에서의 정합 오차(RGB 픽셀, 점 45 146개):

| 깊이 전략 | 중앙값 | p95 |
|---|---|---|
| 고정 2 m 정면 평면 | 8.20 | 17.2 |
| 레이아웃 중앙값 (수직 평면) | 1.04 | 15.3 |
| 레이아웃 중앙값 + ECC | 2.40 | 8.5 |
| 매칭된 열점을 지나는 평면 | 0.59 | 20.2 |
| 파이프라인 오버레이 (선택된 평면, 열점 평면이 없을 때만 ECC) | 0.68 | 20.2 |
| **검출별 평면 (매칭된 열점의 박스 / 피크)** | **0.65** | **5.5** |

고정 평면의 측정 오차는 이론 곡선 `41.4 px |1/Z - 1/d|`를 그대로 따르며
(`docs/images/thermal_registration.png`), 이것이 호모그래피 유도가 맞다는 근거입니다. 단일 전역
평면의 p95는 같은 프레임 안에서 깊이가 다른 *다른* 열점들이 지배합니다. 그래서 박스에는 검출별
평면을 씁니다. ECC는 실행의 85 %(128 / 150)에서 채택되었습니다. 근사 평면의 꼬리는 줄이지만
(p95 15.3 -> 8.5 px) 중앙값에 약 1.4 px의 노이즈를 더합니다(1.04 -> 2.40 px). 그래서 평면이
이미 매칭된 열점을 지나면 ECC를 건너뜁니다.

실행 시간(Windows 개발 PC, numpy, JSON의 `timing_ms`): 합성 열화상 렌더 중앙값 40.1 ms / p95
97.5 ms / 최대 143 ms(450프레임, 시야에 파이프 랙이 얼마나 들어오는지에 따라 다름), 퓨전 중앙값
2.0 ms / p95 36.8 ms / 최대 48 ms(꼬리는 ECC + 800 x 600 오버레이)입니다. 5 Hz 카메라의 200 ms
예산 안이며, 필요하면 `process_every_n` / `fusion_scale`로 퓨전 부하를 줄일 수 있습니다. (RGB
대용 렌더, 중앙값 391 ms는 벤치마크에만 있습니다.)

그림: `docs/images/thermal_examples.png`(H01 - H05의 열화상 | RGB 대용 | 퓨전),
`docs/images/thermal_fusion_example.png`, `docs/images/thermal_registration.png`,
`docs/images/thermal_detection.png`.

### 4.2 순찰 벤치마크 (`docs/results/patrol_eval.json`)
선택된 뷰포인트(게이지당 후보 238개):

| 게이지 | base x, y (m) | yaw (deg) | 점수 | 카메라-다이얼 거리 (m) | 다이얼 직경 (px) | 입사각 (deg) | 유효 후보 |
|---|---|---|---|---|---|---|---|
| G01 | -3.06, -0.91 | -105 | 0.958 | 1.47 | 168 | 23.4 | 187 |
| G02 | -0.50, -0.85 | -90 | 0.974 | 1.47 | 169 | 18.2 | 238 |
| G03 | 4.60, -5.30 | 180 | 0.980 | 1.35 | 184 | 15.5 | 228 |
| G04 | -1.70, 5.50 | 180 | 0.909 | 1.55 | 162 | 25.2 | 178 |
| G05 | -1.70, 7.60 | 180 | 0.974 | 1.47 | 169 | 18.2 | 221 |
| G06 | 3.50, 4.30 | 90 | 0.860 | 1.68 | 150 | 26.9 | 170 |
| G07 | 26.00, -0.45 | 90 | 0.950 | 1.51 | 165 | 21.8 | 221 |
| G08 | 57.00, 0.45 | -90 | 0.950 | 1.51 | 165 | 21.8 | 197 |

(입사각에는 0.54 m 높이 카메라보다 다이얼이 약 20 deg 위에 있는 앙각이 포함됩니다. 높이 1.2 m인
G04와 1.3 m인 G06은 수직 항 점수가 가장 낮습니다.) 같은 A* 거리 행렬에서의 경로 길이:

| 경로 | 최적 (Held-Karp) | 레이아웃 순서 | 탐욕적 NN | 무작위 평균 | 레이아웃 순서 대비 단축 | 무작위 대비 단축 |
|---|---|---|---|---|---|---|
| 폐회로: 도크 -> G08 -> G06 -> G05 -> G04 -> G01 -> G02 -> G03 -> G07 -> 도크 | **150.50 m** | 164.49 m | 151.51 m | 277.85 m | 8.5 % | 45.83 % |
| 개방: 도크 -> G08 -> G07 -> G06 -> G05 -> G04 -> G01 -> G02 -> G03 | **88.92 m** | 158.75 m | 89.75 m | 225.46 m | 43.99 % | 60.56 % |

(지형 비용 기본값. JSON `terrain_cost`에 비용 없는 계획과의 비교가 있습니다.) 지형 비용이 없으면 폐회로는 145.88 m
(도크 -> G08 -> G07 -> G03 -> ... -> G06)이고 C-OBS1, C-OBS3, C-OBS4, C-OBS5, F-OBS3, F-OBS9를 지납니다. 지형 비용을 쓰면
4.62 m(+3.2 %) 길어지는 대신 돌아갈 길이 없는 C-OBS4 철판과 C-OBS5 배관만 지납니다. 우회 때문에 홀 안의 방문 방향이 바뀌어
G07을 마지막에 들릅니다. 이번에는 탐욕적 최근접 이웃(151.51 m)보다 Held-Karp가 1.01 m 짧습니다. Held-Karp는 어떤
배치에서든 최적을 *보장*합니다(전수 탐색과 비교하는 테스트). 무작위 순서 2000개 중 최선은 150.50 m(폐회로) / 94.99 m(개방)
입니다. 계획 시간(JSON `compute_time_s`): A* 36회를 포함한 폐회로 계획 4.54 s(비용 없이 1.07 s; 비용이 있으면 A*가 더 많은
칸을 펼침), 캐시된 A* 길이를 쓰는 개방 계획 0.32 s이며, 노드는 워커 스레드에서 계획합니다. 그림: `docs/images/patrol_route.png`,
`docs/images/patrol_candidates.png`(게이지별 점수장).

---------------------------------------------------------------------------

## 5. 파라미터

### thermal_fusion (`config/thermal.yaml`)
| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `thermal_topic`, `rgb_topic`, `odom_topic` | /thermal_camera/image, /inspection_camera/image, /odom | 입력 |
| `alerts_topic`, `fusion_topic` | /inspection/thermal_alerts, /inspection/thermal_fusion | 출력 |
| `publish_fusion`, `fusion_scale` | true, 1.0 | 퓨전 영상 켜기/끄기와 크기 |
| `process_every_n` | 1 | n번째 열화상 프레임마다 처리 |
| `rgb_sync_tolerance_s`, `odom_sync_tolerance_s`, `sync_fallback_age_s` | 0.15, 0.25, 0.5 | 시간 동기화 |
| `thermal_resolution`, `l8_min_k`, `l8_max_k` | 0.01, 253.15, 655.0 | 복사 측정 |
| `ambient_k`, `warning_k`, `alarm_k`, `segmentation_k` | -1 (레이아웃 값) | 임계값 |
| `min_area_px`, `open_px` | 6, 0 | 분할 |
| `plane_mode`, `depth_mode`, `plane_distance`, `depth_min`, `depth_max` | vertical, layout, 2.0, 0.4, 30.0 | 정합 평면 |
| `ecc_enabled`, `ecc_every_n`, `ecc_scales`, `ecc_motion`, `ecc_max_iter`, `ecc_eps` | true, 2, [0.25, 0.5], translation, 60, 1e-4 | ECC |
| `ecc_max_shift_px`, `ecc_max_vertical_px`, `ecc_min_cc` | 30, 8, 0.2 | ECC 타당성 한계 |
| `ecc_smoothing`, `ecc_epipolar`, `ecc_skip_hotspot_plane`, `ecc_reset_depth_change` | 0.5, true, true, 0.25 | ECC 동작 |
| `match_tolerance_m`, `match_tolerance_rel`, `max_match_range_m` | 0.3, 0.04, 30 | 광선 -> 레이아웃 연관 |
| `report_normal_in_view`, `roi_min_excess_k`, `roi_max_range_m` | true, 3.0, 8.0 | 보이는 정상 열점 보고 |
| `confirm_frames`, `max_misses`, `max_gap_s`, `track_radius_m`, `track_radius_px` | 3, 1, 2.0, 0.6, 30 | 시간 확인 |
| `blend_delta_k`, `blend_ramp_k`, `alpha_max`, `display_min_k`, `display_max_k` | 4, 6, 0.7, -1, -1 | 시각화 |
| `draw_isotherms`, `draw_legend` | true, true | 오버레이 |

### synthetic_thermal_camera (`config/thermal.yaml`)
`rate` 5.0, `topic` /thermal_camera/image, `odom_topic` /odom, `frame_id`
thermal_camera_link, `encoding` mono16, `resolution` 0.01, `width` 320,
`height` 240, `hfov` 0.8, `netd_k` 0.05, `column_fpn_k` 0.03, `row_noise_k` 0.02,
`blur_sigma_px` 0.6, `machine_k` 303.15, `seed` 0, `use_odom_z` true,
`body_height` 0.34, `publish_without_odom` false.

### patrol_planner (`config/patrol.yaml`)
| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `map_yaml` (launch 인자) | '' -> 레이아웃 내비 맵 | 맵 파일 |
| `map_topic`, `use_map_topic`, `replan_on_map_update` | /map, true, true | 맵 토픽 처리 (맵 내용이 바뀔 때만 재계획) |
| `fallback_layout_map`, `map_wait_s` | true, 10.0 | 맵이 오지 않으면 레이아웃에서 맵을 래스터화 |
| `start_mode` | dock | `robot`: 재계획을 현재 자세에서 시작 |
| `gauge_ids` | '' | 쉼표로 구분한 부분집합 |
| `ring_min`, `ring_max`, `ring_step`, `angle_max_deg`, `angle_step_deg` | 0.9, 2.2, 0.1, 40, 5 | 후보 |
| `robot_radius`, `body_height` | 0.30, 0.34 | 로봇 |
| `image_margin_px`, `px_min`, `px_good`, `incidence_max_deg`, `clearance_good`, `los_ignore_m` | 12, 60, 170, 60, 0.8, 0.35 | 점수 항 |
| `w_pixel`, `w_incidence`, `w_vertical`, `w_clearance` | 0.35, 0.25, 0.25, 0.15 | 가중치 |
| `avoid_zone_ids`, `zone_penalty` | [Z-HOT], 0.10 | 구역 벌점 |
| `fallback_ring_max`, `fallback_angle_max_deg` | 3.0, 60 | 완화 패스 |
| `return_to_dock`, `max_exact_nodes`, `smooth_paths` | true, 13, true | 경로 |
| `terrain_cost_enabled`, `terrain_w_pipe`, `terrain_w_rubble`, `terrain_w_plate`, `terrain_w_platform` | true, 25, 25, 1, 2 | 지형 비용(배관·잔해 안쪽 칸 비용, 철판·낮은 단은 작게) |
| `terrain_margin`, `terrain_margin_fraction` | 0.25, 0.15 | 여유 폭 [m] (0.25 m 상한), 여유 폭 비용 = 비율 × 안쪽 비용 |

---------------------------------------------------------------------------

## 6. 실행

```bash
# 모듈 전체 (Gazebo 열화상 센서)
ros2 launch module2_inspection inspection.launch.py
# Gazebo 센서 대신 합성 열화상 카메라 - Gazebo 열화상 카메라를 꺼야 함.
# 그렇지 않으면 둘 다 /thermal_camera/image에 발행해서 thermal_fusion이 두 렌더러의
# 프레임을 번갈아 받게 됨:
ros2 launch plant_simulation bringup.launch.py thermal_source:=synthetic    # 둘 다 처리, 또는
ros2 launch plant_simulation sim.launch.py enable_thermal:=false
ros2 launch module2_inspection inspection.launch.py thermal_source:=synthetic
# 개별 노드
ros2 run module2_inspection thermal_fusion --ros-args --params-file install/module2_inspection/share/module2_inspection/config/thermal.yaml -p use_sim_time:=true
ros2 run module2_inspection patrol_planner --ros-args -p map_yaml:=/path/to/plant_map.yaml
# 재계획 / 확인
ros2 topic pub --once /patrol/replan std_msgs/msg/Empty {}
ros2 topic echo /inspection/thermal_alerts --field data
ros2 topic echo /patrol/plan_json --once --field data --qos-durability transient_local
```

단위 테스트: `python -m pytest module2_inspection/test/test_thermal.py
module2_inspection/test/test_synthetic_thermal.py module2_inspection/test/test_viewpoint.py -q`
(launch API 대체물에 대한 `inspection.launch.py` 드라이 런 포함: 실행 파일 vs `setup.py`, YAML
섹션 이름 vs 노드 이름, `thermal_source` 조건).

알아 둘 노드 동작:
* `patrol_planner`는 **steady clock** 타이머로 계획합니다. 그래서 Gazebo가 일시정지되어 있거나
  `/clock`이 아직 발행되지 않아도 latched 계획이 나옵니다(메시지 타임스탬프는 여전히 노드 시계,
  즉 시뮬레이션 시간을 씀). `gauge_ids`의 알 수 없는 ID는 경고와 함께 무시하고 `unreachable`에
  넣습니다. `/patrol/route` 자세에는 바닥 높이(`Layout.floor_z`)가 들어 있으므로, RViz에서 경로가
  복도 하층(-0.88 m)에 그려집니다.
* `thermal_fusion`의 "열화상 영상 없음" 감시도 steady clock으로 동작합니다.
* `synthetic_thermal_camera`는 ROS 시간 타이머로 렌더링합니다(Gazebo 센서처럼 *시뮬레이션* 시간
  기준 5 Hz).
* module 2의 모든 노드(이 셋과 `gauge_reader`)는 Ctrl-C / launch 종료 시 깔끔하게 종료합니다
  (`KeyboardInterrupt`와 rclpy의 `ExternalShutdownException`).

## 7. 한계
* Gazebo 열화상 카메라는 온도가 지정된 visual(Thermal 시스템)에 대해서만 온도를 보여 주고, 나머지는
  모두 주변 온도로 나타납니다. 파이프라인에는 뜨거운 물체가 뜨겁게 보이기만 하면 됩니다. 월드가
  온도를 지정하지 않는다면 Gazebo 센서를 끄고 `thermal_source:=synthetic`을 쓰십시오
  (`bringup.launch.py thermal_source:=synthetic`이 둘 다 처리합니다).
* 프레임당 호모그래피 하나로는 깊이가 다른 물체들을 동시에 정합할 수 없습니다. 매칭된 열점의
  박스는 각자의 평면을 쓰고, 컬러 오버레이는 선택된 평면(+ ECC)을 씁니다.
* 합성 렌더러는 그럴듯한 모델일 뿐 물리적인 복사 측정 시뮬레이션이 아닙니다(다중 경로 반사, 대기
  감쇠 없음).
* 순찰 계획기는 2-D 맵에서 동작합니다. 0.54 m 높이 카메라에서 다이얼을 가리는 낮은 장애물은
  모델링하지 않습니다(생성된 맵은 이를 자유 공간으로 표시합니다).
