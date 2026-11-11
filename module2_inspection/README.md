# module2_inspection - 플랜트 점검: 게이지, 열화상, 순찰

디지털 트윈 4족 점검 로봇의 Module 2에 해당하는 ROS 2(`ament_python`) 패키지입니다.
모든 알고리즘은 ROS에 의존하지 않는 모듈(numpy + OpenCV)에 들어 있고, Windows에서 ROS 없이
단위 테스트합니다. `*_node.py` 파일은 얇은 rclpy 래퍼입니다(DESIGN_SPEC 4~6절).

| 실행 파일 (`ros2 run module2_inspection ...`) | 모듈 | 역할 |
|---|---|---|
| `gauge_reader` | `gauge_reader_node.py` | 아날로그 게이지 판독 (이 문서) |
| `thermal_fusion` | `thermal_fusion_node.py` | 열화상/RGB 정합, 열점 경보 |
| `synthetic_thermal_camera` | `synthetic_thermal_node.py` | 레이아웃으로 만든 열화상 영상 (대체 수단) |
| `patrol_planner` | `patrol_planner_node.py` | 뷰포인트 + Held-Karp 순찰 경로 |

```bash
ros2 launch module2_inspection inspection.launch.py                       # 모듈 전체
# Gazebo 센서 대신 합성 열화상 카메라 사용: 시뮬레이션을 Gazebo 열화상 카메라 없이 실행해야 함.
# 그렇지 않으면 두 소스가 모두 /thermal_camera/image에 발행함
ros2 launch plant_simulation bringup.launch.py thermal_source:=synthetic  # 둘 다 처리, 또는:
ros2 launch plant_simulation sim.launch.py enable_thermal:=false
ros2 launch module2_inspection inspection.launch.py thermal_source:=synthetic
```

이 launch 파일의 `thermal_source:=synthetic`은 `synthetic_thermal_camera`를 시작할 뿐이며,
Gazebo 센서를 끄지는 못합니다. 노드는 자기가 발행하지 않은 프레임이 출력 토픽에 들어오면
(두 번째 열화상 소스) 경고합니다.

---------------------------------------------------------------------------------------------

## 1. 아날로그 게이지 판독

`simulation/config/world_layout.yaml`의 아날로그 압력 게이지 8개를 점검 카메라(800 x 600,
hfov 0.9 rad, 로봇 몸체에서 위로 0.30 rad 기울임)로 읽습니다. 로봇이 흔들리고, 비스듬히 지나가고,
어두운 플랜트 홀에서 작업하는 상황을 전제로 합니다.

| 파일 | 내용 |
|---|---|
| `module2_inspection/gauge_reader.py` | `GaugeReader`(프레임별 파이프라인), `GaugeFusion`(시간 융합), 오버레이 그리기 |
| `module2_inspection/gauge_ocr.py` | `DigitOCR`: 눈금 숫자의 템플릿 매칭 OCR, 범위 추론 |
| `module2_inspection/gauge_association.py` | `GaugeAssociator`: 로봇 자세로 레이아웃 게이지를 투영 -> ROI 힌트, 게이지 ID, OCR 범위와 레지스트리 범위 대조 |
| `module2_inspection/gauge_synth.py` | `GaugeSynth`: 정확한 정답(ground truth)을 가진 사실적인 합성 카메라 프레임 |
| `module2_inspection/gauge_reader_node.py` | ROS 2 노드 `gauge_reader` |
| `config/gauge_reader.yaml` | 모든 튜닝 값 (노드 + 중첩된 `reader:` 블록 = `GaugeReaderConfig`) |
| `test/test_gauge_*.py` | 단위 / 통합 테스트 (ROS 불필요) |
| `../tools/eval_gauge.py` | 오프라인 벤치마크 -> `docs/images/gauge_*.png`, `docs/results/gauge_eval.json` |

### 1.1 ROS 인터페이스

| 방향 | 토픽 | 타입 | 비고 |
|---|---|---|---|
| sub | `/inspection_camera/image` | sensor_msgs/Image | rgb8 / bgr8 / mono8 (`dusan_common.image_utils`), best effort |
| sub | `/inspection_camera/camera_info` | sensor_msgs/CameraInfo | 선택 사항. 내부 파라미터 `K` (없으면 `camera_hfov`) |
| sub | `/odom` | nav_msgs/Odometry | `base_link`의 월드 좌표계 자세 -> `robot_model.SENSORS`로 카메라 자세 계산 |
| sub | `/imu` | sensor_msgs/Imu | \|omega\| > `max_angular_velocity`인 동안 프레임을 건너뜀 |
| sub | `/inspection/target` | std_msgs/String | 미션이 시야에 있으리라 기대하는 게이지 ID (`""` = 없음). transient-local **과** volatile로 모두 구독 |
| pub | `/inspection/gauge_readings` | std_msgs/String (JSON) | 프레임에 보인 모든 게이지의 최신 융합 판독값 |
| pub | `/inspection/gauge_overlay` | sensor_msgs/Image bgr8 | 적합한 타원, 바늘, 눈금 끝점, 값 / 단위 / 상태 / ID |
| pub | `/inspection/gauge_snapshot` | std_msgs/String (JSON) | 점검 1회당 사진 1장. `/inspection/target`이 그 게이지에서 바뀔 때(임무가 판독을 확정하거나 건너뛸 때) 발행 |

`/inspection/gauge_snapshot` (`module2_inspection/gauge_snapshot.py`, cv_bridge 없이 JSON + base64 JPEG):

```json
{"stamp": 132.4, "t_end": 133.0, "wall": "2026-11-11 14:32:10",
 "id": "G05", "tag": "TANK-PI-02", "value": 3.701, "unit": "bar", "status": "normal",
 "confidence": 0.93, "n_samples": 7, "range": [0.0, 10.0],
 "read": true, "frames": 21, "readings": 18, "frame_value": 3.702, "frame_confidence": 0.95,
 "robot_pose": [-1.6, 7.75, 3.1416], "distance": 1.81, "incidence_deg": 4.7,
 "format": "jpeg", "width": 320, "height": 420, "jpeg_b64": "/9j/4AAQ...", "full_jpeg_b64": "/9j/4AAQ..."}
```

* `value` / `status`는 점검이 끝날 때의 최종 융합값으로, 미션이 확정하는 값과 같습니다.
* `jpeg_b64`: 게이지 확대 사진(적합한 테두리와 바늘을 그림) 아래에 ID, 태그, 값, 단위, 상태, 신뢰도, 샘플 수,
  시뮬레이션 시각, 실제 시각을 적은 JPEG(약 25 KiB). 글자는 OpenCV 글꼴의 한계로 영문입니다.
  `full_jpeg_b64`: 같은 표시를 한 카메라 전체 화면(가로 `snapshot_full_width` px, 0이면 null).
* 사진은 점검 중 신뢰도가 높은 프레임 `snapshot_candidates`개 가운데, 그 프레임 자체의 바늘 값이 최종값과
  1 % FS(또는 3 MAD) 안에서 일치하는 프레임입니다. 한 번도 검출하지 못한 점검은 마지막 프레임을 `read: false`, `value: null`로 보냅니다.
* `metrics_logger`가 `~/dusan_metrics/<세션>/gauges/`에 저장하고 `inspection_report.html`에 넣습니다(사용 가이드 10.3).

`/inspection/gauge_readings` (DESIGN_SPEC 4절. `robot_pose` 뒤의 키는 추가 항목):

```json
{"stamp": 132.4,
 "readings": [{"id": "G05", "tag": "TANK-PI-02", "value": 3.701, "unit": "bar",
               "confidence": 0.93, "angle_deg": 125.1, "range": [0.0, 10.0],
               "status": "normal", "n_samples": 7, "robot_pose": [-1.6, 7.75, 3.1416],
               "raw_value": 3.699, "frame_confidence": 0.95, "mad": 0.002, "outliers": 0,
               "range_source": "registry+ocr", "range_ocr": [0.0, 10.0], "ocr_confidence": 0.8,
               "pixel": [402.1, 288.7], "radius_px": 61.3, "blur_px": 1.2, "mirrored": false,
               "distance": 1.81, "incidence_deg": 4.7}]}
```

* `value`는 시간 융합된 값입니다. `status` = `Layout.gauge_status(id, value)`
  (`normal | warning | alarm`), `n_samples` = 융합 창의 inlier 수입니다.
* `range`는 항상 식별된 게이지의 레지스트리 범위입니다. `range_source`는 다음 중 하나입니다.
  `registry`(이번 프레임에서 OCR을 실행하지 않음), `registry+ocr`(OCR이 같은 범위를 읽음),
  `registry(ocr_disagrees)`(OCR 결과가 다름: 레지스트리가 우선하고, `range_ocr`에 OCR 범위를
  보고하며, 경고를 로그에 남김).
* 식별할 수 없는 다이얼(자세 매칭도 없고 대상도 아님)은 `"id": null, "status": "unknown"`으로
  발행하며, 범위와 값은 OCR로만 구합니다(라벨을 읽을 수 없으면 `null`). 레지스트리 게이지만
  필요한 소비자는 `id == null`을 건너뜁니다(mission manager, 대시보드, metrics logger는 이미
  그렇게 합니다).

### 1.2 프레임별 파이프라인 (`GaugeReader.read`)

다이얼 기하는 *다이얼 단위*(다이얼 텍스처 한 변에 대한 비율, 흰 면 반지름
`gauge_design.R_FACE = 0.44`)로 표현합니다. 각도는 `dusan_common.gauge_design`을 따릅니다
(3시 방향에서 반시계 방향으로 잰 도 단위, 최솟값 225 deg, 최댓값 -45 deg, 시계 방향 270 deg 스윕,
6시 방향에 90 deg 간격, 마지막 15 %에 적색 위험 구간).

1. **영상 보정.** 화이트포인트 노출 스트레치(99.5 백분위수 -> 235, 이득 <= 12), 어두운 프레임의
   평균 밝기를 0.42로 옮기는 적응 감마(`gamma = log 0.42 / log mean`, 하한 0.4), LAB L 채널의
   CLAHE, Immerkaer 노이즈 추정(라플라시안 차분 마스크)에 따라 세기를 정하는 가우시안 디노이즈를
   적용합니다. 검출은 보정된 영상에서 하고, 측광 측정은 모두 선형 스트레치만 한 영상에서 합니다
   (감마/CLAHE는 어두움 프로파일을 왜곡하기 때문입니다).
2. **다이얼 후보.** 4단 영상 피라미드에서 Hough 원을 찾고(단마다 반지름 10-44 px, 합쳐서
   10-330 px), 심하게 기울어진 다이얼을 위해 밝은 블롭 타원(Otsu와 적응 임계값 윤곽, 볼록성 /
   타원 채움 검사)을 더합니다. 자세를 알면 예측 ROI 안에서 찾습니다. 모든 후보는 면/테두리 엣지에
   *스냅*합니다(48개 광선의 중앙값 반경 프로파일에서 가장 강한 밝음->어두움 단차. 중앙값을 쓰므로
   눈금, 바늘, 가림막이 무시됨). 중복은 병합하고, 가장 유망한 후보는 중심/형상 보정(광선별 단차
   검출 + 로버스트 타원 적합)을 한두 차례 더 거칩니다. 점수 = 면/테두리 대비 x 어두운 고리 안쪽
   흰 고리의 닫힘 정도 x (눈금 고리 에너지, 바늘 존재)입니다.
3. **타원 적합 + 블러 측정.** 광선 180개, 서브샘플 포물선 피크를 쓰는 정합 단차 필터,
   Tukey-IRLS 직접 최소제곱 타원 적합(Fitzgibbon / Halir-Flusser)을 범위를 좁혀 가며 세 차례
   수행합니다. 광선을 따라 잰 엣지의 20-80 % 폭 `w(psi)^2 = w0^2 + (0.6 L cos(psi -
   beta))^2`에서 모션 블러 길이 `L`, 방향 `beta`, 블러 범위를 얻습니다. 블러가 반지름의
   `max_blur_ratio`를 넘는 다이얼은 거부하고, 블러가 3 px를 넘으면 50 % 레벨 교차점으로 엣지를
   다시 적합합니다(대칭 블러에서 편향이 없음).
4. **계량 정류.** 원뿔 곡선에서 원의 자세를 구합니다. 시선 원뿔 `Q = K^T C K`의 고유값은
   `l1 >= l2 > 0 > l3`이고, 법선이 `(sqrt(l1-l2), 0, +-sqrt(l2-l3))`(고유기저)인 평면이 원뿔을
   원으로 자릅니다. 두 해 중 하나는 자세로 예측한 다이얼 법선으로 고르고, 그것이 없으면 허브
   검사(균일하게 어두운 허브가 정류된 중심에 있어야 함)로 고릅니다. 결과는 정규 다이얼 좌표계 ->
   영상 호모그래피입니다. 즉 입사각 40 deg에서도 정면에서 본 다이얼을 얻습니다(내부 파라미터가
   없으면 아핀으로 대체).
5. **눈금 보정.** 어두움 맵 `D = (face - I) / (face - ink)`를 극좌표로 펼칩니다(0.5 deg x
   0.005 다이얼 단위 빈). 각도별 면 기준값을 써서 조명 기울기와 글레어를 없애고, 가려진 섹터
   (흰 기준 고리가 희지 않은 곳)는 마스크합니다. 바깥 눈금 띠, 주눈금만 있는 띠, 위험 구간의
   붉은 정도에 대한 각 프로파일을, 레지스트리 범위로 `gauge_design`이 렌더링한 템플릿(범위를
   모르면 4..8 분할의 일반 템플릿)과 두 방향 모두에 대해 상관시킵니다(마스크 FFT NCC,
   Padfield 2012). NCC 피크는 인쇄된 눈금의 평면 내 회전(빈 이하 포물선 보정)을, 더 잘 맞는
   방향은 거울 플래그(적색 구간이 최대 쪽 끝을 표시)를, 템플릿은 주눈금 분할 수를 알려 줍니다.
   NCC가 낮으면 눈금이 없는 둥근 방해물(표지판, 조명)로 보고 거부합니다.
6. **바늘.** 거친 각도 = 고리 0.09-0.22 안에서 각 방향 어두움 프로파일의 피크입니다(허브,
   반대쪽 바늘 꼬리, 숫자, 90 deg 간격을 제외하며, 측정된 블러만큼 고리를 줄임). 정밀 각도 =
   반경별로 1도 미만 정밀도로 구한 어두움 중심점들을 지나는 로버스트(Tukey) 직선 적합이며, 허브에
   부드러운 피벗 제약을 둡니다. 블러가 큰 다이얼(> 2.5 px)이나 중심점 적합이 모호한 경우에는
   합성 분석(analysis-by-synthesis) 보정을 합니다. 바늘 없는 다이얼(보정된 회전/거울 적용)과
   바늘 다각형을 렌더링하고, 둘 다 추정 PSF(beta 방향 길이 L의 모션 커널 * 가우시안)로 흐린 뒤
   면 전체(숫자, 간격, 가려진 섹터 제외)에서 `O ~ a + b S - c B(theta)`를 적합합니다. 잔차
   최소점(1 deg 격자, 0.5 / 0.25 deg 포물선 보정)이 바늘 각도입니다.
7. **눈금 숫자 OCR** (`gauge_ocr.py`, 1.3절) -> 범위를 모를 때는 `(vmin, vmax)`를 구하고,
   알 때는 레지스트리 범위를 교차 확인합니다.
8. **값 + 신뢰도.** `value = gauge_design.angle_to_value(phi, vmin, vmax)`이고, 눈금 밖으로는
   +-4 % FS까지로 제한합니다. 신뢰도 = 검출 점수, 타원 적합 잔차, 블러 비율, 보정 NCC와 마진,
   바늘 근거, 다이얼 해상도의 가중 기하평균입니다. `min_confidence` 미만이거나, 눈금이 없거나
   (`no_scale`), 흰 면이 없거나(`no_face`), 블러가 너무 큰(`blurred`) 프레임은 거부하고 사유를
   `reader.last_debug`에 남깁니다.

### 1.3 OCR (`gauge_ocr.py`)

라벨은 `gauge_design`이 `cv2.FONT_HERSHEY_SIMPLEX`로 찍습니다. 그런데 Gazebo 텍스처는 개발 PC에서
만들고(OpenCV 5는 이 글꼴을 TrueType 글꼴로 렌더링함) 판독기는 Ubuntu PC에서 실행됩니다
(OpenCV 4.x: 고전 Hershey 획). 그래서 템플릿 뱅크에는 세 가지 글꼴 소스가 들어 있습니다.
256-2048 px 텍스처의 라벨 파라미터로 만든 런타임 `cv2.putText` 글리프, 퍼블릭 도메인 Hershey
simplex 획 글리프(획 굵기 3종), 내장 OpenCV-5 숫자 비트맵입니다.

* **문자 경로** (라벨 >= 7 px): 각 주눈금 주변 `R_LABEL` 위치의 패치를 똑바로 세워 정류하고,
  잉크 맵을 만들고, 투영 기반으로 분할합니다(텍스트 줄 위쪽 62 %에서 숫자 열, 간격의 기준선 영역에
  고립된 블롭으로 소수점, 그리고 마이너스 부호). 그다음 흐린 템플릿 뱅크(측정된 영상 블러에 맞춰
  흐림)와 글리프 NCC를 계산합니다. `infer_range`는 라벨 쌍에 대한 RANSAC으로
  `label_k = vmin + k * step`을 적합하며, step이 해당 범위의 `gauge_design.nice_major_step`과
  같아야 합니다.
* **가설 경로** (라벨 >= 4 px): 합성 분석입니다. 모든 표준 범위(EN 837-1 범위 0.6 .. 1000,
  그리고 -1 .. 범위)에 대해, 보정된 분할 수로 라벨 문자열을 모든 글꼴 변형으로 렌더링하고 모든
  라벨 패치와 상관시킵니다(바늘이 가릴 수 있으므로 가장 나쁜 라벨 하나는 제외). 두 경로가
  일치하면 신뢰도를 높입니다.

### 1.4 게이지 식별과 ROI (`gauge_association.py`)

`camera_world_pose(odom_pose, "inspection_camera")`로 카메라 자세를 구합니다. 모든 레이아웃
게이지를 평면 원판으로 투영해(텍스처 한 변 = `dial_diameter`, `tools/generate_world.py`의 규약)
예측 중심, 면 타원, 깊이, 입사각, 평면 유도 호모그래피, 광학 좌표계에서의 다이얼 법선을 얻습니다.
`visible` = 카메라 앞에 있고, 카메라를 향하고, 입사각 <= 70 deg, 면 전체가 영상 안,
반지름 >= 12 px, 거리 <= 6 m(+ 선택적으로 2-D 내비 맵 위 가시선)입니다. 보이는 게이지는
`GaugeHint`가 됩니다(ROI 반 크기 `2 r + 3 pose_sigma_px + 20`, 레지스트리 범위, 법선 사전정보.
`/inspection/target` 게이지가 맨 앞). ROI 밖의 검출은 게이트를 둔 탐욕적 최근접 이웃으로
할당합니다(`|d| < 70 px + 1.2 r`, 반지름 비 0.6-1.6). 여전히 할당되지 않은 대상 게이지는 남은
다이얼 중 크기가 비슷하고 가장 가까운 것을 가져가고, 자세가 전혀 없으면 영상 중심에 가장 가까운
다이얼이 대상이 됩니다. `reconcile_range`는 OCR 범위를 레지스트리와 비교합니다(허용 오차 1 % FS.
레지스트리가 우선하고 둘 다 보고).

### 1.5 시간 융합 (`GaugeFusion`)

게이지마다 `fusion_window_s`(10 s)보다 오래되지 않은 최근 `fusion_window_n`(15)개 샘플을 씁니다.
가중 중앙값 `m`(가중치 = 프레임 신뢰도), MAD 게이트
`|v - m| <= k * max(1.4826 MAD, 0.004 FS)`, `k = 3`(샘플이 3개 미만이면 하한만 사용),
융합값 = inlier의 신뢰도 가중 평균, 신뢰도 =
`mean(w_inliers) * min(1, 0.55 + 0.15 n_in) * sqrt(n_in / n)`입니다. 서로 어긋나는 샘플이 두 개면
다수가 형성될 때까지 가장 신뢰도가 높은(그다음으로 가장 최근의) 샘플을 보고합니다. 시계가 거꾸로
뛰면(시뮬레이션 리셋) 이력을 지웁니다.

### 1.6 노드 스케줄링 (`gauge_reader_node.py`)

`process_rate_hz`(5 Hz) 타이머가 가장 최신 영상만 처리합니다. `/inspection/target`이 비어 있는
동안(순찰, 기회적 판독)에는 최대 `idle_process_rate_hz`(2 Hz)로 처리합니다. 노출 직전 0.15 s 동안
IMU가 |omega| > 0.6 rad/s를 보고하면 프레임을 건너뜁니다(흔들림 / 선회. 오버레이에 표시됨).
자세는 영상 타임스탬프에 가장 가까운 오도메트리 샘플입니다(<= 0.5 s, 아니면 "자세 없음" 모드).
판독기는 먼저 예측 ROI를 탐색합니다(비용이 적음). 자세가 없거나 `/inspection/target` 게이지를
찾지 못하면 전체 프레임을 탐색하되(`full_search_min_period_s`, 0.5 s보다 자주 하지는 않음),
그 밖에는 `full_search_period_s`(2 s)마다 전체 프레임을 탐색합니다. CPU를 Gazebo, 제어기와 나눠
쓰기 때문입니다. 저조도 보정은 무언가를 탐색하는 프레임에서만 실행합니다. 탐색이 전혀 없는
프레임(시야에 예측된 게이지가 없음)과 움직임 때문에 건너뛴 프레임은 최대
`overlay_idle_period_s`(0.5 s)마다 오버레이를 받습니다. OpenCV는 코어마다 하나씩이 아니라
`opencv_threads`(2)개의 워커 스레드를 씁니다. 전체 프레임 탐색에서 찾아 레지스트리 게이지에
할당된 다이얼은 레지스트리 범위로 좁은 ROI에서 다시 측정합니다. OCR은 레지스트리 범위가 두 번
확인될 때까지 게이지마다 `ocr_period_s`(2 s)마다, 그 뒤로는 30 s마다 실행합니다. 노드는 30 s마다
처리 통계(프레임, 건너뜀, 전체 탐색, 평균 / 최대 처리 시간)를 로그에 남깁니다. Ctrl-C / launch
종료 시에는 깔끔하게 종료합니다(`KeyboardInterrupt`와 rclpy의 `ExternalShutdownException`).

### 1.7 합성 프레임 (`gauge_synth.py`)

Gazebo가 보여 주는 바로 그 다이얼 텍스처(`gauge_design.render_gauge_face`, GPU처럼 밉맵 적용)를
점검 카메라 핀홀 모델로 평면 원판으로 투영하고, 게이지 케이스(비스듬한 시야에서 보이는 원통 옆면),
스템 배관, 선택적인 밝은 장착 판(생성된 Gazebo 모델과 동일), 복잡한 산업 배경(배관, 패널, 글자,
볼트), 둥글고 밝은 방해물과 합성합니다. 그다음 열화시킵니다. 조도 배율 0.1-1.0과 공간 기울기 +
비네팅 + 조명 색조, 감마, 정반사 글레어, 디포커스(원판 PSF), 선형 모션 블러(임의 방향, 최대
15 px), 샷 + 읽기 노이즈, 8-bit 양자화, JPEG 아티팩트, 선택적인 가림 막대를 적용합니다. 정답:
값, 바늘 각도, 중심 픽셀, 면 타원, 호모그래피, 깊이, 입사각.

### 1.8 벤치마크 (`tools/eval_gauge.py`)

필수 조건 격자 480프레임(조건마다 80: normal, low light, motion blur, oblique 25-42 deg, noise,
combined) + 가림 60프레임 + 배경만 있는 60프레임 + 운용 범위 스윕 200프레임입니다. 레이아웃
게이지 8개를 모두 실제 범위, 단위, 태그로 쓰고, 값은 레지스트리 값, 최솟값 근처, 최댓값 근처,
균일 난수입니다. 거리는 0.9-2.5 m, 카메라는 대부분 다이얼보다 아래, roll +-3..5 deg입니다.
레지스트리 모드 = 범위를 아는 상태(자세로 게이지가 식별된 뒤 노드가 동작하는 방식)이고, ROI
사전정보 없이 전체 프레임에서 검출합니다.

| 조건 | 프레임 | 채택 | 평균 %FS | 중앙값 | p95 | 최대 | OCR 범위 판독 |
|---|---|---|---|---|---|---|---|
| normal | 80 | 100 % | 0.02 | 0.01 | 0.05 | 0.10 | 49 % |
| low_light (조도 0.1-0.3, 감마 1-1.6) | 80 | 100 % | 0.11 | 0.08 | 0.37 | 0.45 | 19 % |
| motion_blur (4-15 px) | 80 | 100 % | 0.28 | 0.17 | 0.76 | 2.22 | 4 % |
| oblique (25-42 deg) | 80 | 100 % | 0.03 | 0.02 | 0.08 | 0.20 | 39 % |
| noise (sigma 6-14 + 샷 노이즈, JPEG 없음/40/60) | 80 | 100 % | 0.05 | 0.02 | 0.14 | 0.23 | 39 % |
| combined (어둠 + 블러 + 노이즈 + 경사) | 80 | 100 % | 0.18 | 0.14 | 0.50 | 0.79 | 0 % |
| **필수 격자 전체** | **480** | **100 %** | **0.11** | **0.05** | **0.40** | **2.22** | 25 % |
| occluded (추가) | 60 | 98.3 % | 0.05 | 0.01 | 0.10 | 1.56 | 45 % |

* 목표: 평균 < 2 % FS **충족 (0.11)**, 95 백분위수 < 5 % FS **충족 (0.40)**. 판독의 99.8 %가
  2 % FS 이내, 100 %가 5 % FS 이내입니다.
* 둥근 방해물이 있는 배경 60프레임에서의 오검출: **0**.
* 범위를 모르는 상태의 OCR: 범위 540개 중 146개 판독, 145개 정답(정밀도 99.3 %. 오류 1건은
  라벨 높이 5 px에서 0-25를 0-2.5로 읽은 것). 판독률은 물리적으로 제한됩니다. 라벨 높이가
  0.026 x 0.30 m = 7.8 mm라서 1 m에서 6.6 px, 2.3 m에서 3 px입니다. 0.9-1.1 m에서는 프레임의
  64 %에서 범위를 읽고, 1.8 m를 넘으면 전혀 읽지 못합니다. 그래서 노드는 식별된 게이지에는
  레지스트리 범위를 쓰고, OCR은 교차 확인 / 미지 다이얼용으로만 씁니다.
* 운용 범위(`gauge_sweeps.png`): 조도 배율 0.05까지(감마 1.3, 샷 노이즈 포함) 100 % 채택,
  0.025-0.035에서 70 %, 0.018에서 30 %입니다. 모션 블러는 18 px까지 판독되고(>= 90 %), 그
  이상에서는 거부되는 프레임이 점점 늘어납니다. 스윕 전체에서 오차가 5 % FS를 넘은 채택 판독은
  없었습니다. 판독기는 추측하지 않고 거부합니다.
* 지연(24프레임, 순차 처리, 이 개발 PC, OpenCV 멀티스레드): 전체 프레임 탐색 중앙값 155 ms
  (p90 190 ms, 최대 199 ms), 자세 예측 ROI 중앙값 81 ms(p90 101 ms, 최대 112 ms).

그림: `docs/images/gauge_error_hist.png`, `gauge_error_by_condition.png`,
`gauge_error_vs_distance.png`, `gauge_ocr_vs_distance.png`, `gauge_sweeps.png`,
`gauge_examples.png`(원본 프레임 + 오버레이, 조건당 2장), `gauge_pipeline.png`
(어둡고 비스듬한 프레임 하나의 영상 보정, 타원 적합, 정류, 극좌표 펼침, 각 프로파일).
모든 수치: `docs/results/gauge_eval.json`(프레임별 기록 포함).

솔직하게 밝혀 둘 점: 벤치마크는 판독기가 보정 기준으로 삼는 것과 같은 다이얼 디자인을
렌더링합니다. 이 판독기는 알려진 플랜트 다이얼(`gauge_design`)을 위한 모델 기반 판독기이며,
디지털 트윈에서는 정확히 이 상황입니다. Gazebo의 PBR 셰이딩, 로봇 그림자, 텍스처 필터링 차이는
벤치마크에 들어 있지 않고, 위의 측광 열화(기울기, 글레어, 감마, 블러, 노이즈)로만 다룹니다.

### 1.9 파라미터 (`config/gauge_reader.yaml`)

노드: `layout_file` (`""` = plant_simulation 레이아웃), 토픽 이름, `process_rate_hz` 5.0,
`idle_process_rate_hz` 2.0, `opencv_threads` 2,
`camera_hfov` 0.9 / `use_camera_info`, `max_angular_velocity` 0.6 rad/s, `imu_window_s` 0.15,
`odom_max_age_s` 0.5, `pose_sigma_px` 30, `max_incidence_deg` 70, `min_radius_px` 12,
`max_distance` 6, `use_map_occlusion` false, `detect_unhinted` true, `full_search_period_s`
2.0, `full_search_min_period_s` 0.5, `publish_unknown` true, `publish_empty` false,
`fusion_window_n` 15, `fusion_window_s` 10, `fusion_mad_k` 3, `ocr_period_s` 2,
`ocr_confirmed_period_s` 30, `overlay_enabled`, `overlay_enhanced`, `overlay_scale`,
`overlay_idle_period_s` 0.5, `stats_period_s` 30, 점검 사진 `snapshot_enabled` true,
`snapshot_topic`, `snapshot_crop_px` 320, `snapshot_full_width` 640, `snapshot_jpeg_quality` 85,
`snapshot_enhanced` true, `snapshot_candidates` 3.
판독기(`reader.*`, `GaugeReaderConfig`의 필드와 1:1): 영상 보정(`white_percentile`,
`max_gain`, `gamma_target`, `gamma_min`, `clahe_clip`, `clahe_tiles`,
`denoise_sigma_thresh`), 검출(`min_radius_px`, `max_radius_px`, `hough_canny`,
`hough_acc`, `max_candidates`, `max_refine`, `min_detection_score`, `max_dials`, `roi_scale`,
`roi_margin_px`), 적합 / 블러(`n_rays`, `max_blur_ratio`, `mid_level_blur_px`,
`min_frame_sharpness`), 눈금 보정(`allow_mirror`, `inner_band_weight`, `red_weight`,
`min_calibration_ncc`), 바늘(`needle_model` auto|always|off, `needle_model_blur_px`), OCR
(`ocr_enabled`, `ocr_min_label_px`, `ocr_min_label_px_hyp`, `ocr_min_confidence`,
`ocr_div_ncc_margin`), 채택(`min_face_contrast`, `min_confidence`, `clamp_margin_fs`).
YAML의 모든 실수 값은 소수점을 유지합니다(ROS 2는 double 파라미터에 int를 거부합니다.
`test_gauge_node.py`가 이를 검사합니다).

### 1.10 실행

```bash
# 시뮬레이션 실행 중 (sim.launch.py / bringup.launch.py)
ros2 run module2_inspection gauge_reader --ros-args \
    --params-file $(ros2 pkg prefix module2_inspection)/share/module2_inspection/config/gauge_reader.yaml \
    -p use_sim_time:=true

ros2 topic echo /inspection/gauge_readings
ros2 run rqt_image_view rqt_image_view /inspection/gauge_overlay      # 설치된 경우. 또는 RViz / 웹 대시보드
# 판독기에 어떤 게이지를 기대하는지 알려 줌 (보통 mission manager가 latched로 보냄.
# 일반 volatile 발행자도 동작함)
ros2 topic pub --once --qos-durability transient_local /inspection/target std_msgs/msg/String "{data: G05}"
```

오프라인 (Windows 또는 Ubuntu, ROS 없음):

```bash
python -m pytest module2_inspection/test -q          # module 2 전체 테스트
python tools/eval_gauge.py                            # 벤치마크. 커밋된 실행: 평가 46 s, worker 15개 (+ 그림)
python tools/eval_gauge.py --frames 120 --sweep 0 --workers 1 --no-plots   # 빠른 확인
```

### 1.11 테스트 (`test/test_gauge_*.py`)

* `test_gauge_ocr.py` - 라벨 파싱, 템플릿 뱅크가 두 OpenCV 글꼴 엔진을 모두 포함하는지, 여러
  텍스처 크기 / 라벨 높이로 렌더링한 라벨 판독, 고전 Hershey 라벨, 노이즈와 블러가 있는 라벨,
  RANSAC 범위 추론의 오판독 거부, 합성 분석 검증, OpenCV 버전 교차 판독.
* `test_gauge_reader.py` - 원뿔 곡선 / 타원 적합, 원뿔 곡선에서 원 자세 복원, 마스크 NCC, 모든
  게이지의 깨끗한 렌더에서 눈금 전 구간에 걸쳐 바늘 오차 < 0.5 % FS, 비스듬한 / roll이 있는 /
  먼 다이얼, 거울상·회전된 텍스처, 열화된 렌더, 부분 가림, 블러 측정, 다이얼 없는 프레임 거부,
  법선 사전정보가 있는 ROI 힌트, 범위를 모를 때의 OCR 범위.
* `test_gauge_association.py` - `tools/generate_world.py`와 공유하는 축 규약, 텍스처 배율, 게이지
  8개 모두에서 투영 == 합성 렌더, 가시성 규칙, 힌트 순서 / ROI, 자세 오차가 있을 때와 자세가
  없을 때의 할당, 범위 조정, CameraInfo 내부 파라미터, 오도메트리 노이즈가 있는 자세 -> 힌트 ->
  판독 -> 융합 전체 체인.
* `test_gauge_fusion.py` - 가중 중앙값, 이상값 제거, 신뢰도 가중, 개수 / 시간 창, 값 변화,
  시뮬레이션 리셋 처리, 샘플 1개·2개일 때의 동작.
* `test_gauge_synth.py` - 블러 / 디포커스 커널, 정답 기하, 시점 자세, 측광 열화, 가림 / 판
  옵션, 조건 프리셋, 빈 프레임.
* `test_gauge_node.py` - rclpy와 메시지 패키지의 메모리 내 대체물에 대해 노드를 import
  (ROS가 설치되지 않은 경우에만): YAML 파라미터 타입, 인터페이스, `/inspection/gauge_readings`의
  JSON 계약, bgr8 오버레이, IMU 게이트, 자세 없음 모드.

### 1.12 한계

* 판독기는 `gauge_design` 다이얼(270 deg 스윕, 아래쪽 간격, 최대 쪽 끝의 적색 구간)에 맞춰
  보정되어 있습니다. 다른 다이얼 디자인에는 새 템플릿(`_TemplateBank`)이 필요합니다.
* 약 4 px 미만의 라벨은 읽을 수 없습니다. 약 1.8 m보다 멀면 범위는 레지스트리에서만 옵니다.
* 약 20 px를 넘는 블러나 정격의 약 3 % 미만 조도에서는 거부가 발생합니다(벤치마크에서 조용히
  틀린 값을 내는 일은 없음). IMU 게이트가 흔들림 블러 대부분을 피합니다.
* 기계 뒤에 가려졌는데 보인다고 예측된 게이지는 `full_search_period_s`마다 전체 프레임 탐색
  비용을 치르게 합니다. `use_map_occlusion`을 켜면 내비 맵으로 그런 예측을 버립니다.

---------------------------------------------------------------------------------------------

## 2. 열화상 퓨전과 순찰 계획 -> [THERMAL_PATROL.md](THERMAL_PATROL.md) 참조

`thermal_fusion`, `synthetic_thermal_camera`, `patrol_planner`(평면 유도 열화상->RGB 정합, 열점
경보, 합성 열화상 카메라, 뷰포인트 점수화와 Held-Karp 순찰 경로, 각각의 파라미터, 평가, 한계)는
[THERMAL_PATROL.md](THERMAL_PATROL.md)에 정리되어 있습니다.
