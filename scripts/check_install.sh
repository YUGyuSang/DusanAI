#!/usr/bin/env bash
# Sanity check of the Ubuntu machine before running the digital twin.
#
#   bash scripts/check_install.sh
#
# Prints PASS / WARN / FAIL per item and exits 1 if any FAIL was found.
set -e
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

N_FAIL=0
N_WARN=0
pass() { printf '  \033[1;32mPASS\033[0m  %s\n' "$*"; }
warnc() { printf '  \033[1;33mWARN\033[0m  %s\n' "$*"; N_WARN=$((N_WARN + 1)); }
fail() { printf '  \033[1;31mFAIL\033[0m  %s\n' "$*"; N_FAIL=$((N_FAIL + 1)); }
section() { printf '\n\033[1m%s\033[0m\n' "$*"; }

section "System"
if [ -f /etc/os-release ]; then
  pass "OS: $(. /etc/os-release && echo "$PRETTY_NAME")"
else
  warnc "cannot read /etc/os-release"
fi
pass "repository: $REPO_DIR"
pass "workspace:  $WS_DIR"

section "ROS 2"
if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ] || ls /opt/ros/*/setup.bash >/dev/null 2>&1; then
  source_ros
  pass "ROS_DISTRO=$ROS_DISTRO (/opt/ros/$ROS_DISTRO)"
else
  fail "no ROS 2 installation in /opt/ros (bash scripts/install_deps.sh --add-ros-repo)"
fi
for cmd in ros2 colcon xacro python3; do
  if command -v "$cmd" >/dev/null 2>&1; then pass "command '$cmd': $(command -v "$cmd")"; else fail "command '$cmd' not found"; fi
done
# The `ros2` binary alone (ros2cli) has no launch/run verbs: they come from the
# ros2launch / ros2run packages (part of ros-<distro>-ros-base).
if command -v ros2 >/dev/null 2>&1; then
  for verb in launch run topic pkg; do
    case "$verb" in
      launch|run) need=fail ;;
      *) need=warnc ;;
    esac
    if ros2 "$verb" -h >/dev/null 2>&1; then
      pass "verb 'ros2 $verb'"
    else
      "$need" "verb 'ros2 $verb' not available (sudo apt install ros-$ROS_DISTRO-ros-base, or ros-$ROS_DISTRO-ros2$verb)"
    fi
  done
fi

section "Gazebo"
if command -v gz >/dev/null 2>&1; then
  ver="$(gz sim --version 2>/dev/null | head -n 1 || true)"
  pass "gz: $(command -v gz) ${ver:+($ver)}"
else
  fail "'gz' command not found (install ros-$ROS_DISTRO-ros-gz)"
fi

section "ROS packages"
if command -v ros2 >/dev/null 2>&1; then
  pkg_list="$(ros2 pkg list 2>/dev/null || true)"
  for p in ros_gz_sim ros_gz_bridge robot_state_publisher xacro rviz2 tf2_ros rclpy nav_msgs sensor_msgs \
           visualization_msgs launch_ros; do
    if printf '%s\n' "$pkg_list" | grep -qx "$p"; then pass "$p"; else fail "$p missing"; fi
  done
fi

section "Python modules"
for m in numpy cv2 yaml; do
  if v="$(python3 -c "import $m; print(getattr($m, '__version__', 'ok'))" 2>/dev/null)"; then
    pass "python3 $m $v"
  else
    fail "python3 module '$m' missing (python3-numpy python3-opencv python3-yaml)"
  fi
done
if python3 -c "import matplotlib" >/dev/null 2>&1; then pass "python3 matplotlib (tools/plot_metrics.py)"; else warnc "python3-matplotlib missing (only needed by tools/plot_metrics.py)"; fi
# demo HUD (plant_simulation demo_hud): Korean text needs Pillow + a Korean font, else it is English
if python3 -c "import PIL" >/dev/null 2>&1; then pass "python3 PIL (demo HUD Korean text)"; else warnc "python3-pil missing (demo HUD text falls back to English: sudo apt install python3-pil)"; fi
if ls /usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc /usr/share/fonts/truetype/nanum/Nanum*.ttf >/dev/null 2>&1 \
   || { command -v fc-list >/dev/null 2>&1 && fc-list :lang=ko 2>/dev/null | grep -q .; }; then
  pass "Korean font (demo HUD)"
else
  warnc "no Korean font (demo HUD text falls back to English: sudo apt install fonts-noto-cjk)"
fi

section "Workspace"
if [ -f "$WS_DIR/install/setup.bash" ]; then
  source_ws
  pass "overlay $WS_DIR/install/setup.bash"
  pkg_list="$(ros2 pkg list 2>/dev/null || true)"
  for p in $PACKAGES; do
    if printf '%s\n' "$pkg_list" | grep -qx "$p"; then pass "package $p"; else fail "package $p not built"; fi
  done
  # Every console_scripts entry point of every setup.py (a test keeps this list in sync).
  EXECUTABLES="module1_locomotion:locomotion_controller module1_locomotion:terrain_mapper
    module1_locomotion:keyboard_teleop
    module2_inspection:gauge_reader module2_inspection:thermal_fusion
    module2_inspection:synthetic_thermal_camera module2_inspection:patrol_planner
    module3_gas_safety:gas_dispersion_sim module3_gas_safety:source_seeker
    module3_gas_safety:battery_simulator module3_gas_safety:comm_simulator module3_gas_safety:return_to_home
    plant_simulation:map_server_lite plant_simulation:odom_tf plant_simulation:cmd_vel_mux
    plant_simulation:mission_manager plant_simulation:monitor_dashboard plant_simulation:metrics_logger
    plant_simulation:clock_relay plant_simulation:demo_hud plant_simulation:follow_camera"
  exe_pkg=""
  exe=""
  for pe in $EXECUTABLES; do
    pk="${pe%%:*}"; e="${pe##*:}"
    if [ "$pk" != "$exe_pkg" ]; then
      exe="$(ros2 pkg executables "$pk" 2>/dev/null || true)"
      exe_pkg="$pk"
    fi
    if printf '%s\n' "$exe" | grep -q " $e\$"; then pass "executable $pk/$e"; else fail "executable $pk/$e missing"; fi
  done
  share="$(ros2 pkg prefix plant_simulation 2>/dev/null || true)/share/plant_simulation"
  for f in worlds/plant.sdf maps/plant_map.yaml urdf/spot.urdf.xacro config/ros_gz_bridge.yaml \
           config/world_layout.yaml config/gui.config config/gui_lean.config config/mission.yaml \
           config/dashboard.yaml config/mux.yaml \
           launch/sim.launch.py launch/bringup.launch.py rviz/plant.rviz web/index.html web/app.js; do
    if [ -e "$share/$f" ]; then pass "share/plant_simulation/$f"; else
      case "$f" in
        maps/*) warnc "share/plant_simulation/$f missing (map_server_lite will rasterise a fallback map)" ;;
        config/gui.config|config/gui_lean.config) warnc "share/plant_simulation/$f missing (default Gazebo GUI layout)" ;;
        *) fail "share/plant_simulation/$f missing" ;;
      esac
    fi
  done
  for p in module1_locomotion:locomotion.launch.py module2_inspection:inspection.launch.py \
           module3_gas_safety:gas_safety.launch.py; do
    pk="${p%%:*}"; lf="${p##*:}"
    pre="$(ros2 pkg prefix "$pk" 2>/dev/null || true)"
    if [ -n "$pre" ] && [ -f "$pre/share/$pk/launch/$lf" ]; then pass "$pk/launch/$lf"; else fail "$pk/launch/$lf missing"; fi
  done
  if [ -f "$share/urdf/spot.urdf.xacro" ] && command -v xacro >/dev/null 2>&1; then
    tmp="$(mktemp /tmp/spot_urdf.XXXXXX)"
    if xacro "$share/urdf/spot.urdf.xacro" > "$tmp" 2>/dev/null; then
      pass "xacro expands spot.urdf.xacro ($(wc -c < "$tmp") bytes)"
      if command -v gz >/dev/null 2>&1; then
        if gz sdf -p "$tmp" >/dev/null 2>&1; then pass "gz sdf converts the URDF"; else warnc "gz sdf -p failed on the expanded URDF"; fi
      fi
    else
      fail "xacro failed on spot.urdf.xacro"
    fi
    rm -f "$tmp"
  fi
  if [ -f "$share/worlds/plant.sdf" ] && command -v gz >/dev/null 2>&1; then
    if GZ_SIM_RESOURCE_PATH="$share/models:$(dirname "$share")" gz sdf -k "$share/worlds/plant.sdf" >/dev/null 2>&1; then
      pass "gz sdf -k worlds/plant.sdf"
    else
      warnc "gz sdf -k reported problems in worlds/plant.sdf"
    fi
  fi
else
  fail "workspace not built ($WS_DIR/install/setup.bash) - bash scripts/build.sh"
fi

section "Runtime"
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':8080 '; then
  warnc "port 8080 already in use (the dashboard will try 8081..8085)"
else
  pass "port 8080 free"
fi
if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
  pass "display available (${XDG_SESSION_TYPE:-unknown} session)"
else
  warnc "no display: use gui:=false headless_rendering:=true"
fi
if command -v glxinfo >/dev/null 2>&1; then
  pass "OpenGL: $(glxinfo -B 2>/dev/null | grep -m1 'OpenGL renderer' | cut -d: -f2- | sed 's/^ //')"
fi
if grep -rlI $'\r' --include='*.sh' --include='*.py' --include='*.yaml' "$REPO_DIR/scripts" "$REPO_DIR/simulation" \
     2>/dev/null | head -n 1 | grep -q .; then
  warnc "CRLF line endings found in scripts/ or simulation/ (sed -i 's/\\r\$//' <file>)"
else
  pass "LF line endings"
fi

printf '\n%s FAIL, %s WARN\n' "$N_FAIL" "$N_WARN"
[ "$N_FAIL" -eq 0 ]
