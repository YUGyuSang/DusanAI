#!/usr/bin/env bash
# Keyboard teleoperation in THIS terminal (needs a real TTY).
#
#   bash scripts/run_teleop.sh
#   bash scripts/run_teleop.sh -p mode:=latch      # extra --ros-args parameters
#
# Keys (DESIGN_SPEC section 7): w/s a/d q/e move, x stop, space E-STOP, m manual
# override, t start mission, p patrol, g gas seek, h return home, o pause/resume,
# 1 stand 2 sit 3 trot 4 crawl 5 auto, r recover, b battery 19 %, ESC quit.
set -e
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source_ws
[ -t 0 ] || warn "stdin is not a terminal - keyboard input will not work"

args=(--ros-args -p use_sim_time:=true)
prefix="$(ros2 pkg prefix module1_locomotion 2>/dev/null || true)"
cfg="$prefix/share/module1_locomotion/config/teleop.yaml"
if [ -n "$prefix" ] && [ -f "$cfg" ]; then
  args+=(--params-file "$cfg")
fi
info "ros2 run module1_locomotion keyboard_teleop ${args[*]} $*"
exec ros2 run module1_locomotion keyboard_teleop "${args[@]}" "$@"
