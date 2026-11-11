#!/usr/bin/env bash
# Install the Ubuntu / ROS 2 packages needed to build and run the digital twin.
#
#   bash scripts/install_deps.sh                 # ROS 2 Lyrical (default)
#   ROS_DISTRO=jazzy bash scripts/install_deps.sh
#   bash scripts/install_deps.sh --add-ros-repo  # also configure the ROS 2 apt repository
#   bash scripts/install_deps.sh --desktop       # also install ros-<distro>-desktop
#
# Always installs ros-<distro>-ros-base (ROS 2 CLI incl. `ros2 launch` / `ros2 run`),
# ros-gz (Gazebo + bridge), xacro, robot_state_publisher, rviz2, the Python deps and
# python3-pil + fonts-noto-cjk (Korean text on the demo HUD).
#
# Packages that do not exist for the selected distro are reported and skipped,
# then rosdep resolves whatever the package.xml files still need.
set -e
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ADD_REPO=0
DESKTOP=0
for a in "$@"; do
  case "$a" in
    --add-ros-repo) ADD_REPO=1 ;;
    --desktop) DESKTOP=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown option $a" ;;
  esac
done

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  command -v sudo >/dev/null 2>&1 || die "run as root or install sudo"
  SUDO="sudo"
fi

info "ROS 2 distro: $ROS_DISTRO  ($(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"))"

if [ "$ADD_REPO" -eq 1 ]; then
  info "configuring the ROS 2 apt repository (ros2-apt-source)"
  $SUDO apt-get update
  $SUDO apt-get install -y curl software-properties-common
  $SUDO add-apt-repository -y universe || true
  ver="$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}')"
  [ -n "$ver" ] || die "could not query the ros-apt-source release (offline?)"
  codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")"
  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.${codename}_all.deb"
  $SUDO dpkg -i /tmp/ros2-apt-source.deb
fi

$SUDO apt-get update

PKGS=(
  # ros_base brings the ROS 2 CLI verbs the scripts use (`ros2 launch` = ros2launch,
  # `ros2 run` = ros2run, `ros2 topic`, `ros2 param` ...), rclpy, launch_ros and the
  # message packages. ros-gz alone only pulls in the bare `ros2` command (ros2cli),
  # so `ros2 launch` would fail with "invalid choice: 'launch'".
  "ros-$ROS_DISTRO-ros-base"
  "ros-$ROS_DISTRO-ros2launch"
  "ros-$ROS_DISTRO-ros2run"
  "ros-$ROS_DISTRO-ros-gz"
  "ros-$ROS_DISTRO-xacro"
  "ros-$ROS_DISTRO-robot-state-publisher"
  "ros-$ROS_DISTRO-rviz2"
  "ros-$ROS_DISTRO-tf2-ros"
  python3-numpy
  # optimised BLAS for numpy (the module-1 MPC QP is dense linear algebra; the
  # reference libblas3 is several times slower)
  libopenblas0-pthread
  python3-opencv
  python3-yaml
  python3-colcon-common-extensions
  python3-rosdep
  xterm
  # optional: unit tests and tools/plot_metrics.py
  python3-pytest
  python3-matplotlib
  # demo HUD (plant_simulation demo_hud): Pillow + a Korean font, else the HUD text is English
  python3-pil
  fonts-noto-cjk
)
if [ "$DESKTOP" -eq 1 ]; then
  PKGS+=("ros-$ROS_DISTRO-desktop")
fi

avail=()
missing=()
for p in "${PKGS[@]}"; do
  if apt-cache show "$p" >/dev/null 2>&1; then
    avail+=("$p")
  else
    missing+=("$p")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  warn "not available from the configured apt sources: ${missing[*]}"
  case " ${missing[*]} " in
    *" ros-$ROS_DISTRO-"*) warn "is the ROS 2 apt repository configured? (re-run with --add-ros-repo)" ;;
  esac
fi
if [ "${#avail[@]}" -gt 0 ]; then
  info "installing: ${avail[*]}"
  $SUDO apt-get install -y "${avail[@]}"
fi

# --------------------------------------------------------------------- rosdep
if command -v rosdep >/dev/null 2>&1; then
  if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    info "rosdep init"
    $SUDO rosdep init || warn "rosdep init failed"
  fi
  info "rosdep update"
  rosdep update --rosdistro "$ROS_DISTRO" || warn "rosdep update failed (offline?)"
  info "rosdep install (package.xml dependencies of $REPO_DIR)"
  rosdep install --from-paths "$REPO_DIR" --ignore-src -r -y --rosdistro "$ROS_DISTRO" \
    || warn "rosdep could not resolve every key - check the output above"
else
  warn "rosdep not available - skipping dependency resolution"
fi

info "done. Next: bash scripts/build.sh && bash scripts/check_install.sh"
