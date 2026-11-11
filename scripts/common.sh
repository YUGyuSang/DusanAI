# shellcheck shell=bash
# Shared helpers, sourced by the other scripts in this directory:
#   . "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Environment overrides
#   ROS_DISTRO   ROS 2 distribution (default: lyrical, or whatever is already sourced)
#   DUSAN_WS     colcon workspace (default: the parent of the 'src' directory that
#                contains this repository, else ~/dusan_ws)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-lyrical}"
PACKAGES="dusan_common module1_locomotion module2_inspection module3_gas_safety plant_simulation"

info()  { printf '\033[1;34m[dusan]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[dusan] WARNING:\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31m[dusan] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Print the workspace when the repo sits directly in <ws>/src/ (exit 1 otherwise).
# Only the immediate parent is considered so that e.g. ~/src/projects/DusanAI
# does not turn the whole home directory into a colcon workspace.
find_ws() {
  local parent
  parent="$(dirname "$REPO_DIR")"
  if [ "$(basename "$parent")" = "src" ]; then
    dirname "$parent"
    return 0
  fi
  return 1
}

if [ -n "${DUSAN_WS:-}" ]; then
  WS_DIR="$DUSAN_WS"
  # absolute path without trailing '/' (DUSAN_WS="~/ws" in quotes is not tilde-expanded)
  case "$WS_DIR" in
    "~") WS_DIR="$HOME" ;;
    "~/"*) WS_DIR="$HOME/${WS_DIR#"~/"}" ;;
    /*) ;;
    *) WS_DIR="$PWD/$WS_DIR" ;;
  esac
  while [ "${#WS_DIR}" -gt 1 ] && [ "${WS_DIR%/}" != "$WS_DIR" ]; do WS_DIR="${WS_DIR%/}"; done
elif WS_DIR="$(find_ws)"; then
  :
else
  WS_DIR="$HOME/dusan_ws"
fi

# Make sure colcon finds this repository below $WS_DIR/src (used by build.sh).
# Nothing to do when the repository already lies inside <ws>/src (e.g. the
# default ~/ros2_ws/src/DusanAI layout); otherwise <ws>/src/DusanAI -> <repo>
# is created (or re-pointed if it is a stale symlink to another copy).  This
# also covers DUSAN_WS pointing at a fresh or different workspace.
ensure_ws_src() {
  local made_ws=0 made_src=0 repo_phys src_phys link cur
  [ -d "$WS_DIR" ] || made_ws=1
  [ -d "$WS_DIR/src" ] || made_src=1
  mkdir -p "$WS_DIR/src" || die "cannot create $WS_DIR/src"
  WS_DIR="$(cd "$WS_DIR" && pwd)"            # normalise '..', '.', trailing '/'
  repo_phys="$(cd "$REPO_DIR" && pwd -P)"
  src_phys="$(cd "$WS_DIR/src" && pwd -P)"
  # The workspace must not be the repository or lie inside it: colcon would crawl
  # the repository (and a src/DusanAI link would point at its own parent).
  case "$(cd "$WS_DIR" && pwd -P)/" in
    "$repo_phys/"*)
      if [ "$made_src" -eq 1 ]; then rmdir "$WS_DIR/src" 2>/dev/null || true; fi
      if [ "$made_ws" -eq 1 ]; then rmdir "$WS_DIR" 2>/dev/null || true; fi
      die "workspace $WS_DIR is the repository or inside it - set DUSAN_WS to a directory outside $REPO_DIR (e.g. ~/dusan_ws)" ;;
  esac
  case "$repo_phys/" in "$src_phys/"*) return 0 ;; esac
  case "$REPO_DIR/" in "$WS_DIR/src/"*) return 0 ;; esac
  link="$WS_DIR/src/DusanAI"
  if [ -L "$link" ]; then
    cur="$(cd "$link" 2>/dev/null && pwd -P || true)"
    [ "$cur" = "$repo_phys" ] && return 0
    warn "$link pointed to $(readlink "$link") - re-linking it to $REPO_DIR"
    rm -f "$link"
    ln -s "$REPO_DIR" "$link"
  elif [ -e "$link" ]; then
    cur="$(cd "$link" 2>/dev/null && pwd -P || true)"
    [ "$cur" = "$repo_phys" ] && return 0
    die "$link exists and is not this repository ($REPO_DIR) - remove it or set DUSAN_WS to another workspace"
  else
    ln -s "$REPO_DIR" "$link"
  fi
  info "repository is not inside $WS_DIR/src - linked $link -> $REPO_DIR"
}

# Source /opt/ros/<distro>/setup.bash (falls back to any installed distro).
source_ros() {
  local setup="/opt/ros/$ROS_DISTRO/setup.bash"
  if [ ! -f "$setup" ]; then
    local cand
    for cand in /opt/ros/*/setup.bash; do
      if [ -f "$cand" ]; then
        warn "/opt/ros/$ROS_DISTRO not found - using $cand"
        setup="$cand"
        break
      fi
    done
  fi
  [ -f "$setup" ] || die "ROS 2 is not installed in /opt/ros (run: bash scripts/install_deps.sh)"
  # ROS setup files are not written for 'set -e' / 'set -u'
  set +e
  # shellcheck disable=SC1090
  . "$setup"
  set -e
  ROS_DISTRO="${ROS_DISTRO:-$(basename "$(dirname "$setup")")}"
}

# Script flags -> launch arguments (run_bringup.sh / run_demo.sh).  Sets the array LAUNCH_ARGS:
#   --low-memory (--low-mem, --lowmem) or DUSAN_LOW_MEMORY=1  -> low_memory:=true  (8 GB laptop profile)
#   an explicit low_memory:=... argument wins; everything else is passed through unchanged.
# Without the profile on a PC with less than 12 GB RAM a hint is printed.
parse_launch_flags() {
  LAUNCH_ARGS=()
  local low=0 explicit=0 a
  case "${DUSAN_LOW_MEMORY:-}" in 1|true|TRUE|yes|on) low=1 ;; esac
  for a in "$@"; do
    case "$a" in
      --low-memory|--low-mem|--lowmem) low=1 ;;
      low_memory:=*) explicit=1; LAUNCH_ARGS+=("$a") ;;
      *) LAUNCH_ARGS+=("$a") ;;
    esac
  done
  if [ "$low" -eq 1 ] && [ "$explicit" -eq 0 ]; then
    LAUNCH_ARGS+=("low_memory:=true")
  fi
  if [ "$low" -eq 0 ] && [ "$explicit" -eq 0 ]; then
    local kb
    kb="$(awk '/^MemTotal:/ {print $2}' "${DUSAN_MEMINFO:-/proc/meminfo}" 2>/dev/null || true)"
    if [ -n "$kb" ] && [ "$kb" -lt 12582912 ]; then
      warn "이 PC의 메모리는 약 $(( (kb + 524288) / 1048576 )) GB입니다 - 8 GB 노트북이면 --low-memory 를 붙여 실행하세요 (카메라 640x480 5 Hz, 그림자/RViz 끔, 대시보드 유지)"
    fi
  fi
}

# Source ROS + the built workspace overlay.
source_ws() {
  source_ros
  local overlay="$WS_DIR/install/setup.bash"
  [ -f "$overlay" ] || die "workspace not built ($overlay missing) - run: bash scripts/build.sh"
  set +e
  # shellcheck disable=SC1090
  . "$overlay"
  set -e
}
