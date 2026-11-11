# Spot 모델 에셋

4족 로봇 `spot`의 visual 메시입니다. Codyssey 에셋 패키지의 Unity 리그
(`Doosanenerbility/.../object/robot_spot.unitypackage`
→ `SpotHPRIG_Rigging_02.fbx` + `Material.001_Base_Color.png`)에서
`tools/convert_spot.py`로 변환했습니다. 로봇 자체는 `simulation/urdf/spot.urdf.xacro`(전개한
사본: `spot.urdf`)가 기술하고, 기구학, 한계, 질량은 `dusan_common/dusan_common/robot_model.py`에서
가져옵니다.

| 파일 | 내용 |
|---|---|
| `meshes/base_link.obj` | 몸통 (몸체 스킨 클러스터 `Spot_Body_Bone_01`) |
| `meshes/base_cam.obj` | 앞쪽 상단의 카메라 페이로드 (`Spot_Cam_01`, FBX 재질 색) |
| `meshes/base_eap.obj` | 뒤쪽의 라이다 페이로드 (`Spot_EAP2_01`, FBX 재질 색) |
| `meshes/<leg>_hip.obj` | 고관절 하우징 (앞 `*_Shoulder_Bone_01` / 뒤 `*_Pelvis_Bone_01`) |
| `meshes/<leg>_upper_leg.obj` | hip_y 모터를 포함한 허벅지 (`*_UpperArm_Bone_01` / `*_Thigh_Bone_01`) |
| `meshes/<leg>_lower_leg.obj` | 발 공을 포함한 정강이 (`*_Arm_Bone_01` / `*_Calf_Bone_01`) |
| `meshes/spot.mtl` | 단색 재질 (텍스처에서 뽑은 몸체 색 5개 + 부속 재질) |
| `conversion_report.json` | 변환기가 측정한 모든 값 (rest 각도, 삼각형 수, 오차) |

`<leg>` ∈ `fl fr rl rr`이며, FBX의 "R" 본은 로봇 오른쪽(`fr`, `rr`)입니다. 모든 OBJ는 미터
단위이고 **관절 각도 0에서의 URDF 링크 좌표계**(x 전방, y 왼쪽, z 위. 다리는 똑바로 아래로
늘어뜨린 상태)로 표현되어 있으므로, URDF visual 원점은 모두 항등입니다. URDF에서 발 링크에는
25 mm 구 visual(= 충돌 구)을 붙였고, 이 구가 메시의 16.6 mm 발 공을 감쌉니다.

## 변환 파이프라인 (`python tools/convert_spot.py`)

1. **읽기**: `tools/fbx_reader.py`(numpy만 사용)로 `.unitypackage`(`tools/unitypackage.py`)에서
   FBX를 바로 읽습니다. 파일 축을 "G 좌표계"(미터, x = 로봇 오른쪽, y = 전방, z = 위)로 바꾼 뒤
   로봇 base 좌표계 `x_r = y_G - y0, y_r = -(x_G - x0), z_r = z_G - z0`로 바꿉니다.
   `(x0, y0, z0)` = hip_x 본 4개의 중심 = `(-0.00018, 0.01424, 0.36498)`입니다. 모든 곳에서
   **bind pose**(FBX `BindPose` 노드)를 씁니다. 왼쪽 앞다리 본이 스키닝 뒤에 움직여서(최대 14 mm)
   장면 자세는 메시와 맞지 않기 때문입니다.
2. **분할**: 몸체는 강체 스키닝되어 있으므로(모든 제어점이 정확히 클러스터 하나에 속하고 가중치
   1), 각 정점은 자기 본의 링크로, 각 삼각형은 꼭짓점 링크의 다수결로 보냅니다(삼각형 915 542개
   중 83개가 두 링크에 걸침). 두 페이로드는 몸체 본에 강체로 붙어 있으므로 → `base_link`.
3. **Rest 각도** (bind pose 본에서 구함): hip_x = 다리 평면의 roll, hip_y θ = 허벅지의
   atan2(−dx, −dz), knee = φ − θ (φ는 정강이의 각). 앞다리 (±0.09°, 34.44°, −88.94°), 뒷다리
   (∓0.04°, 60.12°, −94.33°)입니다. 각 링크의 정점은 자기 좌표계로 표현하며
   `v_link = T_rest⁻¹ v_robot`이고, `T_rest`는 robot_model 기하의 순기구학으로 구합니다.
4. **색**: 4096² 알베도를 면적 평균으로 1024²로 축소하고 중앙값 필터를 적용합니다(로고 / 작은
   글씨 제거). 각 면의 UV 중심에서 샘플링해 CIE-Lab에서 가중 k-means(k = 8)로 군집화한 뒤, 거의
   같은 색(ΔE < 10)을 병합하고, 면적 0.25 % 미만의 데칼은 이웃 색으로 흡수하며, 팔레트를 최대
   6개로 제한합니다 → 노랑, 진회색, 회색, 연회색, 검정. 면 인접 다수결 필터로 고립된 얼룩을
   없앱니다.
5. **감축(decimation)**: 이차 오차 척도(quadric error metric) 에지 붕괴(Garland–Heckbert)를
   numpy로 일괄·벡터화해 수행합니다. 라운드마다 모든 에지에 대해 최적 붕괴 위치(중점 기준으로
   정규화한 3×3 풀이)와 비용을 구하고, 링크 조건을 어기는 에지(비다양체 결과), 면을 뒤집거나
   찌그러뜨리는 에지는 거부합니다. 가장 싼 에지들 가운데 2-ring 독립 집합을 한 번에 붕괴합니다.
   열린 경계와 색 경계에는 제약 이차식을 둡니다. 삼각형 구멍은 닫을 수 있습니다(이 패널 기반
   메시에 필요). 976 k → 74 k 삼각형(예산: 몸통 19 k, 카메라 4.5 k, 라이다 1.5 k, 고관절 3 k,
   허벅지 5 k, 정강이 4 k).
6. **법선**: 꼭짓점별, 주름 각도 40°(매끈한 패널, 날카로운 모서리), 중복 제거. 정점은 10 µm로
   스냅합니다.
7. **검증** (`conversion_report.json → verification`): OBJ 파일을 다시 읽어 rest 각도에서 FK로
   조립하고 원본 FBX 정점과 비교합니다. 수치는 아래와 같습니다. 미리보기:
   `docs/images/spot_model_preview.png`(원본 vs 변환, rest / zero / stand 자세),
   `docs/images/spot_collision_preview.png`(URDF visual vs 충돌 기하와 카메라 축, `spot.urdf`
   자체로 렌더링).

## 검증 결과

| 지표 | 값 |
|---|---|
| 삼각형 (OBJ 파일 15개) | 73 750 (몸통 18 995, 카메라 5 270, 라이다 1 500, 다리당 약 3 000 / 5 000 / 4 000) |
| `simulation/models/spot` 크기 | 5.4 MB |
| 원본 → 변환 표면 거리 | 평균 0.20 mm, p95 0.68 mm, p99 3.7 mm |
| 변환 → 원본 표면 거리 | 평균 0.08 mm |
| 링크 좌표계 왕복 (분할 + rest 각도 FK) | 0.0 mm (정확) |
| bind pose 본 vs FK 관절 위치 | hip_x 0.4 mm, hip_y ≤ 1.7 mm, knee ≤ 3.7 mm (뒤 허벅지 본이 `UPPER_LEG_LENGTH`보다 0.7 mm 김), foot ≤ 1.5 mm |

최악 거리(최대 ≈ 32 mm)는 QEM이 정당하게 붕괴시키는 **보이지 않는 내부 표면**(예: 닫힌 고관절
하우징 안의 원판, 이중 벽 외피의 안쪽 벽)에 있습니다. 보이는 표면은 모두 약 1 mm 이내입니다
(`docs/images/spot_model_preview.png`의 열을 비교해 보십시오). 카메라 페이로드는 4.5 k가 아니라
5.3 k 삼각형에서 멈춥니다. 나머지 붕괴는 비다양체 결과를 만들기 때문에 감축기가 거부합니다.

## 재생성

```bash
python tools/convert_spot.py               # 약 4분: 메시, 보고서, 미리보기, spot.urdf
python tools/convert_spot.py --urdf-only   # spot.urdf만 다시 전개 (필요: pip install xacro)
python -m pytest tools/test/test_spot_assets.py -q
```

변환기에는 numpy + OpenCV(+ URDF 전개용 순수 Python `xacro` 패키지)만 필요합니다. Unity 패키지는
`tools/.unity_cache/`에 풀립니다.

## 모델 사용

* RViz / robot_state_publisher: `xacro spot.urdf.xacro` (메시 URI
  `package://plant_simulation/models/spot/meshes/*.obj`).
* Gazebo는 `package://plant_simulation/...`을 `GZ_SIM_RESOURCE_PATH`로 해석하므로, 이 변수에
  패키지 share 디렉터리의 **부모**(`<ws>/install/plant_simulation/share`)가 들어 있어야 합니다.
  그럴 수 없으면 `mesh_uri_prefix:=file:///abs/path/to/meshes`로 전개하십시오.
* 그 밖의 xacro 인자: `enable_gazebo`(플러그인 + 센서), `enable_cameras`, `enable_thermal`
  (모두 기본값 `true`).
* URDF의 Gazebo 플러그인: JointPositionController 12개(PID 힘 모드, p 450 / d 3, 힘은 관절
  effort로 제한, `/spot/joint/<joint>/cmd_pos`, 무릎은 −0.20 rad 한계에서 시작),
  JointStatePublisher(`update_rate` 200 Hz), OdometryPublisher(50 Hz, `/model/spot/odometry`),
  그리고 센서 `imu`, `depth_camera`, `inspection_camera`, `thermal_camera`(DESIGN_SPEC §3).
