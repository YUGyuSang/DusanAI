#!/usr/bin/env bash
# Full digital twin: simulation + locomotion + inspection + gas/safety + mission,
# dashboard on http://localhost:8080 and metrics in ~/dusan_metrics/.
#
#   bash scripts/run_bringup.sh
#   bash scripts/run_bringup.sh rviz:=true thermal_source:=synthetic
#   bash scripts/run_bringup.sh mission_autostart:=true teleop_terminal:=xterm
#   bash scripts/run_bringup.sh --low-memory        # 8 GB laptop (= low_memory:=true)
#
# All arguments are passed to: ros2 launch plant_simulation bringup.launch.py
# (--low-memory is translated to low_memory:=true, see parse_launch_flags in common.sh)
set -e
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_launch_flags "$@"
source_ws
info "ros2 launch plant_simulation bringup.launch.py ${LAUNCH_ARGS[*]}"
info "digital twin dashboard: http://localhost:8080 (or the dashboard_port you passed)"
exec ros2 launch plant_simulation bringup.launch.py "${LAUNCH_ARGS[@]}"
