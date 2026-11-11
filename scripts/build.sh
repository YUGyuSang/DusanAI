#!/usr/bin/env bash
# Build the five DusanAI packages with colcon.
#
#   bash scripts/build.sh                 # colcon build --symlink-install
#   bash scripts/build.sh --no-symlink    # plain install (if setuptools rejects develop mode)
#   bash scripts/build.sh --clean         # wipe build/ install/ log/ first
#   bash scripts/build.sh --test          # also run colcon test (pytest of every package)
#   any other argument is passed to colcon build (e.g. --parallel-workers 2)
#
# The workspace is $DUSAN_WS if set, else the parent of the 'src' directory
# containing this repository (e.g. ~/ros2_ws/src/DusanAI -> ~/ros2_ws), else
# ~/dusan_ws.  If the repository is not inside <ws>/src, <ws>/src/DusanAI -> <repo>
# (symlink) is created.  DUSAN_WS must not be the repository itself.
set -e
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SYMLINK="--symlink-install"
CLEAN=0
TEST=0
EXTRA=()
for a in "$@"; do
  case "$a" in
    --no-symlink) SYMLINK="" ;;
    --clean) CLEAN=1 ;;
    --test) TEST=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) EXTRA+=("$a") ;;
  esac
done

# <ws>/src/DusanAI -> <repo> unless the repository already lies inside <ws>/src
# (also when DUSAN_WS names another / a new workspace)
ensure_ws_src

# CRLF line endings (Windows editors / USB copy tools) break python entry points and YAML
if grep -rlI $'\r' --include='*.py' --include='*.yaml' --include='*.sh' --include='*.xacro' \
     --include='*.sdf' "$REPO_DIR" 2>/dev/null | grep -v '/Doosanenerbility/' | head -n 1 | grep -q .; then
  warn "files with CRLF line endings found, e.g.:"
  grep -rlI $'\r' --include='*.py' --include='*.yaml' --include='*.sh' --include='*.xacro' --include='*.sdf' \
    "$REPO_DIR" 2>/dev/null | grep -v '/Doosanenerbility/' | head -n 5 >&2 || true
  warn "convert them with: sed -i 's/\r\$//' <file>   (or dos2unix)"
fi

source_ros
cd "$WS_DIR"
if [ "$CLEAN" -eq 1 ]; then
  info "removing build/ install/ log/"
  rm -rf build install log
fi

info "colcon build ${SYMLINK:-(copy install)} in $WS_DIR (ROS $ROS_DISTRO)"
# shellcheck disable=SC2086
colcon build $SYMLINK --packages-select $PACKAGES "${EXTRA[@]}"

if [ "$TEST" -eq 1 ]; then
  info "colcon test"
  # shellcheck disable=SC2086
  colcon test --packages-select $PACKAGES --event-handlers console_direct+ || true
  colcon test-result --verbose || true
fi

info "build finished. In every new terminal:  source $WS_DIR/install/setup.bash"
info "then: bash scripts/run_bringup.sh   (dashboard: http://localhost:8080)"
