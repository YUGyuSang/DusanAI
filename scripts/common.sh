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
