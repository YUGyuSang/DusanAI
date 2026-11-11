# 가상 플랜트의 Gazebo 모델

모든 모델은 정적 SDF 1.9 모델(`model.config` + `model.sdf`)이며,
`GZ_SIM_RESOURCE_PATH=<share>/plant_simulation/models`(launch 파일이 설정)로 찾고
`model://<name>`으로 참조합니다. 텍스처는 그것을 쓰는 OBJ 옆에 저장하고(`.mtl`의
`map_Kd <file>`), ogre2가 파일 이름이 같은 두 이미지를 혼동하지 않도록 모델 이름을 접두사로
붙입니다.

변환된 메시는 `Doosanenerbility/`의 원본 Unity 패키지(수정하지 않음)에서
`tools/convert_environment.py`가 만들고, 생성 모델은 `simulation/config/world_layout.yaml`에서
`tools/generate_world.py`가 만듭니다. 이 파일들을 직접 고치지 말고 도구를 다시 실행하십시오.
변환 세부 사항(삼각형 수, 재질, 텍스처 출처, 바운딩 박스, 크기)은
`simulation/config/generated/model_manifest.yaml`에, 월드에 배치된 모든 것은
`simulation/config/generated/world_geometry.yaml`에 기록됩니다.

## Unity 에셋 패키지에서 변환한 모델 (visual 메시)

| 모델 | 원본 에셋 | 내용 | 충돌 |
|---|---|---|---|
| `factory_hall` | `factory_inner_1.unitypackage` / `Factory_03.fbx` (메시 FC_Fence_01, 36.7k tris) | 터빈 홀 외피: 바닥, 동쪽 개구부가 있는 벽, x = 9.8에 출입구가 있는 칸막이 벽, 블록 두 개가 있는 뒷방, 서쪽 메자닌 + 계단, 지붕, 조명 | 모델에는 없음 (이 메시에서 측정한 프리미티브 충돌체가 `plant.sdf`에 있음) |
| `corridor` | `factory_hall.unitypackage` / `Corridor_Wall_02.fbx` (10.5k tris) | 48.5 m 서비스 복도: 서쪽 문틀, 옆문, 배관, 계단(디딤판 4개, 0.18 m 챌면) + 하층(-0.88 m)으로 내려가는 11.8 deg 그레이팅 경사로, 끝 벽 | 모델에는 없음 (`plant.sdf`의 프리미티브) |
| `turbine_machine` | `fac_gastank.unitypackage` / `Main_Machine_01.fbx`: 모델 Main_Machine_01 + Gauge_01_003/004/005 | 터빈 / 주 기계(3.28 x 8.37 x 3.08 m)와 작은 압력 게이지 3개. 원점 = bbox 중심, z_min = 0 | `plant.sdf`의 충돌 박스 (낮은 높이의 조밀한 풋프린트) |
| `gas_tank` | `fac_gastank.unitypackage` / `Main_Machine_01.fbx`: 모델 Main_Machine_02 (104.6k tris, 감축하지 않음) | 가스 스키드 / 탱크 배관(배율 1에서 7.11 x 11.78 x 8.22 m. 월드에서는 배율 0.8 사용). 원점 = bbox 중심, z_min = 0 | `plant.sdf`의 충돌 박스 |
| `pallet` | `obs_palette.unitypackage` / `Plastic_Pallet_01.fbx` | 플라스틱 팔레트 1.22 x 1.40 x 0.18 m, 원점 = 바닥 위 bbox 중심 | 바운딩 박스 (모델 안) |

건물 외피는 양면입니다(뒷면 복제). Unity 장면의 여러 벽은 바깥쪽을 향하고 있으며, Unity에서
안쪽에서 보이는 것은 HDRP 재질이 양면이기 때문일 뿐입니다. 알베도 텍스처는 Unity 프리팹 재질
슬롯 / `.mat` GUID(`_BaseColorMap`, `_MainTex`, `_BaseMap`)로 찾고, 최대 1024 px로 축소해 JPG로
저장합니다.

## `tools/generate_world.py`로 생성한 모델

| 모델 | 내용 |
|---|---|
| `gauge_g01`, `gauge_g02`, `gauge_g03`, `gauge_g04`, `gauge_g05`, `gauge_g06`, `gauge_g07`, `gauge_g08` | 아날로그 압력 게이지: 다이얼 원판 OBJ(`meshes/dial.obj`, 텍스처 `gauge_gXX_dial.png`는 레이아웃의 값/범위/단위/태그로 `dusan_common.gauge_design.render_gauge_face`가 렌더링), 하우징 원통, 뒤판, 바닥 기둥(충돌). 원점 = 다이얼 중심(= 레이아웃 `position`), +X = 면 법선, 텍스처 위쪽 = +Z. 원판 직경은 레이아웃 `dial_diameter`와 같고, 텍스처 한 변이 정확히 `dial_diameter`에 걸칩니다(흰 면 반지름 = `gauge_design.R_FACE` x `dial_diameter`. module2의 게이지 합성 / 연관 모듈이 쓰는 규약). `gauge_g06`은 파이프 랙 지지대에 장착됩니다(기둥 없음). |
| `charging_station` | 도크: 바닥과 같은 높이의 1 x 1 m 착지 표시 데칼(`meshes/dock_marking.obj`, 위험 표시 테두리, H, 진행 방향 화살표), 충전 접점과 상태 LED가 있는 뒤판. 원점 = 바닥 위 패드 중심, +X = 도킹한 로봇의 진행 방향. |
| `gas_leak_valve` | 릴리프 밸브 소품(스텁, 밸브 몸체, 핸드휠, 누출원에서 끝나는 출구 배관)과 풍하 방향을 향한 초록색 반투명 가스 `particle_emitter`(`materials/textures/smoke.png`, 깊이 카메라가 무시하도록 `particle_scatter_ratio` 0). Gazebo Harmonic의 ogre2 렌더러는 `color_start`/`color_end`를 무시하고(`ParticleEmitter SetColorRange is currently disabled`) 재질 diffuse 색으로 칠하므로, 재질 diffuse도 초록(0.55 1.0 0.55)으로 지정합니다. 색은 `tools/generate_world.py`에서 바꿉니다. |
| `access_point` | Wi-Fi 액세스 포인트 소품, visual 전용. |

`spot/`(로봇)은 로봇 description이 관리하므로 여기서는 설명하지 않습니다.
