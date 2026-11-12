# 설치 가이드

Ubuntu에 저장소를 받아 설치·빌드하는 자세한 방법입니다. 짧은 버전은 [README](../README.md#install)에, 실행 방법은 [USAGE](USAGE.md)에 있습니다.

## 1. 지원 플랫폼

| | 주 대상 | 함께 지원 |
|---|---|---|
| Ubuntu | 26.04 LTS | 24.04 LTS |
| ROS 2 | Lyrical Luth (`ROS_DISTRO=lyrical`, 스크립트 기본값) | Jazzy Jalisco (`ROS_DISTRO=jazzy`) |
| Gazebo | Jetty / gz-sim 10 (`ros-lyrical-ros-gz`가 설치) | Harmonic / gz-sim 8 (`ros-jazzy-ros-gz`가 설치) |

`ros-<distro>-ros-gz` 패키지가 빌드된 Gazebo 버전을 그대로 쓰세요. packages.osrfoundation.org에서
다른 Gazebo를 따로 설치하지 마세요.

권장 사양: CPU 4코어 이상(8스레드 권장), RAM 8 GB, OpenGL 3.3 드라이버가 동작하는 GPU. Gazebo와
약 17개의 Python 노드가 여러 코어를 씁니다. 모든 노드가 시뮬레이션 시간을 따르므로 실시간 배율(RTF)이
1보다 낮아도 동작은 같습니다. RAM이 8 GB인 PC에서는 실행 스크립트에 `--low-memory`를 붙이세요([USAGE 9장](USAGE.md#low-spec)).

GPU: NVIDIA는 독점 드라이버(소프트웨어 & 업데이트 → 추가 드라이버)를 권장합니다. Intel/AMD 내장 GPU는 Mesa 드라이버로 동작합니다.
렌더러는 `glxinfo -B`(`sudo apt install mesa-utils`)로 확인하며, `check_install.sh`도 출력합니다.

## 2. 저장소 받기

```bash
sudo apt install -y git
mkdir -p ~/dusan_ws/src
git clone https://github.com/YUGyuSang/DusanAI.git ~/dusan_ws/src/DusanAI
```

1. 이 문서는 저장소가 `~/dusan_ws/src/DusanAI`에 있다고 가정합니다. 기존 colcon 워크스페이스(예: `~/ros2_ws/src/DusanAI`)에 두어도 되고,
   워크스페이스 밖(예: `~/DusanAI`)에 두면 `build.sh`가 `~/dusan_ws/src/DusanAI -> <저장소>` 링크를 만듭니다(4절).
2. 스크립트는 항상 `bash scripts/<이름>.sh`로 실행하므로 실행 권한은 필요 없습니다.
3. 줄바꿈은 LF여야 합니다. `git clone`으로 받으면 `.gitattributes`가 맞춰 줍니다. 다른 방법으로 복사한 파일이 CRLF이면
   `build.sh`와 `check_install.sh`가 경고합니다. 고치는 방법은 [USAGE FAQ Q4](USAGE.md#faq-crlf)에 있습니다.
4. 압축 파일 등으로 복사했다면 `__pycache__` 폴더를 지우고 빌드하세요(`launch/__pycache__`는 `setup.py`가 설치해 버립니다).
   `find . -name __pycache__ -type d -prune -exec rm -rf {} +`
5. 새 코드로 업데이트하는 순서(`git pull` → `build.sh` → `check_install.sh`)는 [USAGE "업데이트"](USAGE.md#update)에 있습니다.

## 3. ROS 2, Gazebo, 의존성 설치

### 3.1 스크립트로

```bash
cd ~/dusan_ws/src/DusanAI
bash scripts/install_deps.sh --add-ros-repo          # Ubuntu 26.04 + Lyrical
ROS_DISTRO=jazzy bash scripts/install_deps.sh --add-ros-repo   # Ubuntu 24.04 + Jazzy
bash scripts/install_deps.sh --help
```

하는 일:

1. `--add-ros-repo`: `curl`과 `software-properties-common`을 설치하고 `universe`를 켭니다.
   github.com/ros-infrastructure/ros-apt-source에서 Ubuntu 코드네임에 맞는 최신 `ros2-apt-source`
   패키지를 받아 설치합니다. 이것이 `packages.ros.org`와 서명 키를 설정합니다. ROS 2 저장소가 이미
   설정돼 있으면 이 옵션은 빼도 됩니다.
2. 아래 패키지를 `apt-get install`합니다. 설정된 apt 저장소에 없는 패키지는 목록을 보여 주고 건너뛰므로,
   일부만 설정된 PC에서도 동작합니다.

   | 패키지 | 이유 |
   |---|---|
   | `ros-<distro>-ros-base` | rclpy, launch / launch_ros, 메시지 패키지, tf2, ROS 2 CLI 동사(`ros2 launch`, `ros2 run`, `ros2 topic`, ...) |
   | `ros-<distro>-ros2launch`, `ros-<distro>-ros2run` | `launch` / `run` 동사를 명시적으로 (ros-base에 이미 포함) |
   | `ros-<distro>-ros-gz` | Gazebo(vendor 패키지), `ros_gz_sim`, `ros_gz_bridge`, `gz` 명령 |
   | `ros-<distro>-xacro`, `-robot-state-publisher`, `-rviz2`, `-tf2-ros` | 로봇 기술, TF, 시각화 |
   | `python3-numpy`, `libopenblas0-pthread`, `python3-opencv`, `python3-yaml` | 알고리즘 (OpenBLAS가 MPC 풀이 시간을 낮게 유지) |
   | `python3-colcon-common-extensions`, `python3-rosdep` | 빌드 도구 |
   | `xterm` | 선택: 텔레옵 창 (`teleop_terminal:=xterm`) |
   | `python3-pytest`, `python3-matplotlib` | 선택: 테스트와 `tools/plot_metrics.py` |
   | `python3-pil`, `fonts-noto-cjk` | 데모 HUD(`demo_hud`)의 한글 글자(Pillow + 한글 글꼴). 없으면 HUD 글자가 영어로 나옴 |
   | `ros-<distro>-desktop` | `--desktop`일 때만 (rqt 도구, 데모) |

3. 필요하면 `rosdep init`, 그다음 `rosdep update`와
   `rosdep install --from-paths <저장소> --ignore-src -r -y`로 `package.xml`이 선언했는데 아직 없는 것을 설치합니다.

참고: `ros2` 명령만 있는 패키지(`ros2cli`)에는 `launch`와 `run` 동사가 없습니다. 이 동사는 `ros2launch`와
`ros2run`에서 오는데, `ros-<distro>-ros-gz`는 이들을 끌어오지 않습니다. 없으면 모든 `run_*.sh` 스크립트가
`ros2: error: ... invalid choice: 'launch'`로 실패합니다. 그래서 `ros-base`를 설치하고,
`check_install.sh`가 `ros2 launch -h`와 `ros2 run -h`를 실행해 봅니다.

### 3.2 수동 설치 (결과는 같음)

```bash
export ROS_DISTRO=lyrical            # 24.04에서는 jazzy
sudo apt update && sudo apt install -y curl software-properties-common
sudo add-apt-repository -y universe
VER=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
      | grep -F '"tag_name"' | awk -F'"' '{print $4}')
CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${VER}/ros2-apt-source_${VER}.${CODENAME}_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update
sudo apt install -y ros-$ROS_DISTRO-ros-base ros-$ROS_DISTRO-ros-gz ros-$ROS_DISTRO-xacro \
  ros-$ROS_DISTRO-robot-state-publisher ros-$ROS_DISTRO-rviz2 ros-$ROS_DISTRO-tf2-ros \
  python3-numpy libopenblas0-pthread python3-opencv python3-yaml \
  python3-colcon-common-extensions python3-rosdep xterm python3-pytest python3-matplotlib \
  python3-pil fonts-noto-cjk
sudo rosdep init        # PC마다 한 번 ("already initialized"는 무시)
rosdep update
rosdep install --from-paths ~/dusan_ws/src/DusanAI --ignore-src -r -y --rosdistro $ROS_DISTRO
```

## 4. 빌드

```bash
bash scripts/build.sh                 # colcon build --symlink-install (패키지 5개)
bash scripts/build.sh --clean         # build/ install/ log/를 먼저 삭제
bash scripts/build.sh --no-symlink    # 일반 설치 (setuptools가 --symlink-install을 거부할 때)
bash scripts/build.sh --test          # 빌드 후 colcon test + colcon test-result
bash scripts/build.sh --parallel-workers 2   # 그 밖의 인자는 colcon build로 전달
```

워크스페이스 선택 규칙(`scripts/common.sh`, 모든 스크립트가 같음):

1. `DUSAN_WS=<폴더>`가 있으면 그것 (상대 경로와 `~/...` 허용)
2. 없으면 저장소가 `<ws>/src/<이름>`일 때 `<ws>`
3. 그것도 아니면 `~/dusan_ws`

그다음 `build.sh`는 colcon이 패키지를 찾을 수 있게 합니다. 저장소가 `<ws>/src` 안에 있지 않으면
`<ws>/src/DusanAI -> <저장소>` 심볼릭 링크를 만들고, 링크가 예전 복사본을 가리키면 다시 연결합니다.
`<ws>/src/DusanAI`가 다른 내용이 든 실제 폴더이면 오류로 멈춥니다(지우거나 다른 워크스페이스를 고르세요).
`DUSAN_WS`가 저장소 자신이거나 그 안의 폴더여도 멈춥니다(colcon이 저장소를 두 번 훑게 됨).

빌드하는 패키지: `dusan_common module1_locomotion module2_inspection module3_gas_safety
plant_simulation`. 수동으로 하면:

```bash
cd ~/dusan_ws
source /opt/ros/lyrical/setup.bash   # 24.04: /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select dusan_common module1_locomotion \
  module2_inspection module3_gas_safety plant_simulation
source install/setup.bash            # 새 터미널마다
```

`--symlink-install` 덕분에 저장소의 Python·YAML 수정은 다시 빌드하지 않아도 반영됩니다. 새 파일,
새 entry point, `setup.py` 변경은 `bash scripts/build.sh`를 다시 실행해야 합니다.

## 5. 설치 확인

```bash
bash scripts/check_install.sh
```

항목마다 PASS / WARN / FAIL을 출력하고, 하나라도 FAIL이면 종료 코드 1로 끝납니다.

* **System**: OS, 저장소와 워크스페이스 경로.
* **ROS 2**: `/opt/ros/<distro>`, `ros2 colcon xacro python3` 명령, CLI 동사 `ros2 launch`, `ros2 run`
  (없으면 FAIL), `ros2 topic`, `ros2 pkg` (없으면 WARN).
* **Gazebo**: `gz` 명령과 `gz sim --version`.
* **ROS packages**: ros_gz_sim, ros_gz_bridge, robot_state_publisher, xacro, rviz2, tf2_ros, rclpy,
  메시지 패키지, launch_ros.
* **Python modules**: numpy, cv2, yaml (matplotlib는 선택). 데모 HUD용 PIL과 한글 글꼴(Noto CJK / 나눔)이 없으면 WARN.
* **Workspace**: overlay가 있는지, 패키지 5개와 `setup.py`의 모든 console entry point 21개(module1 3,
  module2 4, module3 5, `clock_relay`·`demo_hud`·`follow_camera`를 포함한 plant_simulation 9)가 설치됐는지(테스트가 이 목록을 `setup.py`와
  맞춰 둡니다), 설치된 config / world / URDF / launch / web 파일이 모두 있는지, `xacro`가 로봇을 전개하는지,
  `gz sdf`가 URDF를 변환하고 월드를 검사하는지.
* **Runtime**: 8080 포트 사용 여부, 디스플레이, OpenGL renderer, LF 줄바꿈.

## 6. 실행 환경

* **source**: `source /opt/ros/<distro>/setup.bash` 다음에 `source <ws>/install/setup.bash`.
  `run_*.sh` 스크립트는 둘 다 알아서 합니다.
* **Gazebo 리소스**: `sim.launch.py`가 `<share>/plant_simulation/models`, `<share>/plant_simulation/worlds`,
  상위 share 폴더를 `GZ_SIM_RESOURCE_PATH`에 추가하므로 `model://...`와 `package://plant_simulation/...`
  URI가 해석됩니다. 직접 export할 것은 없습니다.
* **디스플레이 없음 / 원격 PC**: `gui:=false headless_rendering:=true`는 Gazebo 서버만 실행하고 카메라는
  EGL로 렌더링합니다. 이때 사용자 화면은 대시보드입니다.
* **쓸 만한 GPU가 없음 (VM, 오래된 노트북)**: `thermal_source:=synthetic`이 ogre2가 필요한 Gazebo 열화상
  카메라를 `synthetic_thermal_camera` 노드로 바꿉니다. Gazebo 자체가 렌더링하지 못하면
  `export LIBGL_ALWAYS_SOFTWARE=1`을 시도하세요(느림).
* **Wayland**: Wayland 세션이 감지되면 Gazebo GUI를 `QT_QPA_PLATFORM=xcb`로 자동 실행합니다.
* **같은 네트워크에 ROS PC가 여러 대**: 모든 터미널에서 `export ROS_DOMAIN_ID=<0..101>`을 따로 정해
  두 로봇의 토픽이 섞이지 않게 합니다.
* **다른 PC에서 대시보드 접속**: 모든 인터페이스의 8080 포트에서 수신합니다(사용 중이면 8085까지 다음 빈 포트).
  방화벽이 켜져 있으면 포트를 엽니다(`sudo ufw allow 8080/tcp`). 인증이 없으므로 신뢰할 수 없는
  네트워크에서는 `simulation/config/dashboard.yaml`에 `host: 127.0.0.1`을 설정하세요.
* **메트릭**은 `~/dusan_metrics/<YYYYmmdd_HHMMSS>/`에 기록됩니다(`metrics:=false`로 끔).

## 7. 개발 도구

알고리즘은 ROS와 분리된 Python 모듈이라, 테스트와 평가 도구는 ROS·colcon 없이 Python만으로 실행됩니다(저장소 루트에서).

```bash
python3 -m venv venv
venv/bin/pip install numpy opencv-python-headless pyyaml scipy matplotlib pytest xacro
venv/bin/pip install mujoco          # 선택: MuJoCo 물리 검증·튜닝 도구와 그 테스트
venv/bin/python -m pytest -q         # pytest.ini + conftest.py가 sys.path 설정
```

* `pytest.ini`가 모든 패키지의 테스트 폴더와 `tools/test`를 지정하고, `conftest.py`가 저장소 안의 패키지를
  `sys.path`에 넣으므로 colcon 빌드가 필요 없습니다.
* ROS 없는 대시보드 데모: `PYTHONPATH=dusan_common:simulation venv/bin/python -m plant_simulation.dashboard_demo`
  (http://localhost:8080). 미션 로직과 대시보드를 간단한 로봇 모델로 돌립니다(`--headless --metrics-dir <폴더>`는 미션 하나를
  `tools/plot_metrics.py`용 메트릭 세션으로 기록).
* 평가: `python tools/eval_<locomotion|terrain|gauge|thermal|patrol|gas>.py`가 `docs/results/*.json`과
  `docs/images/*.png`를 씁니다.
* 보행 검증 도구(`mujoco` 필요): `python tools/physics_check.py`는 Gazebo 설정(관절 서보, 1 ms 스텝, 센서 주기, MPC 지연)을
  흉내 낸 MuJoCo 플랜트에서 module1 제어기를 돌리고 `docs/results/physics_check.json`과 `docs/images/physics_*.png`를 씁니다
  (`--quick`은 아무것도 쓰지 않음. 실험할 때는 `--out <임시>.json --no-figures`). `python tools/tune_locomotion.py`는 MPC·보행 파라미터를
  탐색해 `docs/results/tuning.json`과 `docs/images/tuning_*.png`를 씁니다(smoke 실행은 `--quick`).
  방법과 결과는 [module1 README §4](../module1_locomotion/README.md#4-평가)에 있습니다.
* Gazebo 헤드리스 검증(`tools/gazebo_check.py`, `tools/obstacle_check.py`): 실제 gz-sim(DART)을 화면 없이 돌려 로봇의 gz 플러그인 위에서
  module1 제어기와 미션 로직을 시험합니다. gz-sim 8 파이썬 바인딩이 필요합니다. conda-forge 패키지 `gz-sim8`, `gz-transport13`, `gz-msgs10`,
  `gz-math7`, `gz-common5`와 numpy, pyyaml, matplotlib를 한 환경에 설치합니다. Windows의 conda 환경에서는 실행 전에 그 환경의
  `<env>`, `<env>\Library\mingw-w64\bin`, `<env>\Library\usr\bin`, `<env>\Library\bin`, `<env>\Scripts`를 PATH 앞에 넣습니다.
  명령은 [USAGE §12](USAGE.md#gazebo-check), 결과는
  [module1 README §4.5–4.7](../module1_locomotion/README.md#45-gazebo-헤드리스-검증-toolsgazebo_checkpy)에 있습니다. 런타임 패키지는 gz 파이썬
  바인딩을 import하지 않습니다.
* 에셋 파이프라인(결과는 저장소에 들어 있으므로 입력이 바뀔 때만 다시 실행):

  | 명령 | 출력 |
  |---|---|
  | `python tools/convert_spot.py` (`--urdf-only`는 URDF만 다시 전개, `xacro` 필요) | `simulation/models/spot/meshes/*.obj`, `simulation/urdf/spot.urdf`, 미리보기 그림 |
  | `python tools/convert_environment.py` | `simulation/models/{factory_hall,corridor,turbine_machine,gas_tank,pallet}` |
  | `python tools/generate_world.py` | `simulation/worlds/plant.sdf`, 게이지·도크·밸브 모델, `simulation/maps/plant_map.*`, `simulation/config/generated/` |

  두 변환 도구는 과제 제공 원본 에셋(`Doosanenerbility/` 폴더, 저장소에는 없음)이 필요합니다.
  `simulation/config/world_layout.yaml`은 위치(게이지, 열점, 장애물, 누출원, AP, 도크)의 단일 출처입니다.
  수정했다면 `generate_world.py`와 평가를 다시 실행합니다.
* LF 줄바꿈과 UTF-8을 유지하세요. YAML의 실수 파라미터는 모두 소수점을 유지합니다(`1`이 아니라 `1.0`).
  rclpy는 double로 선언된 파라미터에 정수를 거부합니다.

## 8. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `ros2: error: argument ...: invalid choice: 'launch'` | ROS 2 CLI 동사가 없음: `sudo apt install ros-$ROS_DISTRO-ros-base` (또는 `install_deps.sh` 재실행). `check_install.sh`가 FAIL로 알려 줌. |
| `E: Unable to locate package ros-lyrical-...` | ROS 2 apt 저장소가 설정되지 않음: `install_deps.sh --add-ros-repo`. 24.04에서는 `ROS_DISTRO=jazzy`. |
| `colcon build`: `Package 'dusan_common' specified with --packages-select was not found` | 워크스페이스에 저장소 링크가 없음: `bash scripts/build.sh`를 쓰고(`<ws>/src/DusanAI`를 만듦) `DUSAN_WS`를 확인. |
| `build.sh`: `workspace ... is the repository or inside it` | `DUSAN_WS`는 저장소 바깥이어야 함. 예: `DUSAN_WS=~/dusan_ws`. |
| `build.sh`: `<ws>/src/DusanAI exists and is not this repository` | 예전 복사본이 있음: 지우거나 이름을 바꾸거나 다른 `DUSAN_WS`를 사용. |
| `/usr/bin/env: 'python3\r': No such file or directory`, YAML 파싱 오류 | CRLF 줄바꿈: [USAGE FAQ Q4](USAGE.md#faq-crlf)를 실행한 뒤 다시 빌드. |
| 노드 시작 시 `ParameterException` / `InvalidParameterTypeException` | 수정한 YAML 값의 타입이 바뀜(`1.0` 대신 `1`). |
| Gazebo는 뜨는데 로봇이 생성되지 않음 | 월드 로딩이 느림: 생성은 `spawn_attempts`회(기본 5) 재시도함. `spawn_delay`를 늘리고 `gz sim` 출력에서 SDF 오류를 확인(`gz sdf -k <share>/worlds/plant.sdf`). |
| Gazebo에서 로봇이 안 보이는데 움직이고 TF도 정상 | 메쉬의 `package://` URI가 해석되지 않음. `sim.launch.py`가 `GZ_SIM_RESOURCE_PATH`를 설정함. 대안으로 `sim.launch.py`를 `mesh_uri_prefix:=file://$(ros2 pkg prefix plant_simulation)/share/plant_simulation/models/spot/meshes`로 실행. |
| 열화상 영상 없음 / Gazebo 로그에 열화상 카메라 오류 | 열화상 센서는 ogre2와 GPU가 필요함: `thermal_source:=synthetic`. |
| Gazebo 창이 검거나 GUI가 죽음 | GPU 드라이버 문제: 드라이버를 업데이트하거나 `LIBGL_ALWAYS_SOFTWARE=1`을 시도하거나 헤드리스로 실행. |
| 모든 노드가 멈춰 있고 대시보드 데이터가 갱신되지 않음 | `/clock`이 없음: 브리지나 `clock_relay` 노드(브리지의 `/clock_gz`를 200 Hz로 `/clock`에 다시 발행)가 죽었거나 Gazebo가 일시정지 상태(`ros2 topic hz /clock`, 재생 버튼). `sim.launch.py clock_relay:=false`는 브리지가 `/clock`을 직접 발행하게 함. |
| 명령을 줘도 로봇이 움직이지 않음 | mux 확인: `ros2 topic echo /cmd_vel_mux/active` (latched). 텔레옵의 수동 오버라이드(`m`)나 비상정지(space)가 켜져 있으면 텔레옵이 계속 선택됨. |
| 로봇이 출발 직후 1 m 안에서 제자리걸음하며 비틀거리다 넘어짐 | 예전 빌드일 수 있음: [USAGE '업데이트'](USAGE.md#update)대로 최신 코드를 받아 다시 빌드. 원인은 [module1 README §4.7 ①](../module1_locomotion/README.md#47-실제-plant-월드-검증-시나리오-js-수정과-남은-문제). |
| 배관·잔해 근처나 도는 중에 넘어짐 | 기본 보행 trot 유지(`5`를 눌렀다면 `3`), 속도 배율 0.5 이하, 도크 앞 0.10 m 배관은 북쪽 끝(벽 쪽)으로, 복도 위층 잔해(x 37–39)는 북쪽 벽 쪽 통로(y ≈ +1.1)로 돌아가기. [USAGE FAQ Q8](USAGE.md#faq-fall). |
| 데모 HUD 글자가 영어(`GAS: X ppm`)로 나옴 | Pillow나 한글 글꼴이 없음: `sudo apt install python3-pil fonts-noto-cjk` 후 다시 실행(빌드는 필요 없음). `check_install.sh`가 WARN으로 알려 줌. |
| 데모에서 Gazebo 화면이 로봇을 따라가지 않음 | `follow_camera` 로그 확인. Gazebo·로봇·GUI를 240 s까지 기다린 뒤 조용히 끝나므로, 그 뒤에 GUI가 떴으면 `ros2 run plant_simulation follow_camera`를 다시 실행. Gazebo 창에서 `c`는 카메라 고정 / 다시 따라가기, `z`는 기본 시점. 마우스로 바꾼 시점이 유지되지 않으면 [USAGE FAQ Q12](USAGE.md#faq-camera). |
| 넘어진 로봇이 일어나지 않음 | 순간이동 없이 일어설 때까지 다시 시도함(3회 실패하면 5 s 쉬고 반복). 배관 위나 경사로 턱에서는 30–80 s 걸릴 수 있음. 기다려도 안 되면 `/locomotion/command`에 `recover` 또는 `reset_pose`(텔레옵 `0`). 자동 순간이동을 쓰려면 `locomotion.yaml`의 `assisted_reset: true`. [USAGE FAQ Q8](USAGE.md#faq-fall). |
| 철판 근처나 평지에서 로봇이 갑자기 느려짐 | 슬립 모드(미끄러운 바닥 대응). `/locomotion/state`의 `slip_mode`, `slip_source`, `slip_events`를 확인. 평지에서 켜지면 [USAGE FAQ Q10](USAGE.md#faq-slip) 참고. |
| MPC 풀이 시간이 30 ms를 크게 넘음 (`/locomotion/state`의 `mpc_solve_ms`) | `libopenblas0-pthread` 설치, 다른 무거운 프로그램 종료. 그래도 느리면 `locomotion.yaml`의 `mpc_rate`(20.0 → 10.0), `mpc_qp_max_iter`를 낮춤. |
| 대시보드에 접속할 수 없음 | 로그에서 실제 포트(8080..8085)를 확인하고, 방화벽 포트를 열고, Ubuntu PC 자신에서 `http://127.0.0.1:8080`을 시도. |
| `check_install.sh` WARN `gz sdf -k reported problems` | `gz sdf -k`를 직접 실행해 메시지를 읽음. 알 수 없는 요소에 대한 경고는 오래된 Gazebo 버전에서는 무해함. |
