#!/usr/bin/env bash
# Full digital twin: simulation + locomotion + inspection + gas/safety + mission,
# dashboard on http://localhost:8080 and metrics in ~/dusan_metrics/.
#
#   bash scripts/run_bringup.sh
#   bash scripts/run_bringup.sh rviz:=true thermal_source:=synthetic
#   bash scripts/run_bringup.sh mission_autostart:=true teleop_terminal:=xterm
#
# All arguments are passed to: ros2 launch plant_simulation bringup.launch.py
set -e
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source_ws
info "ros2 launch plant_simulation bringup.launch.py $*"
info "digital twin dashboard: http://localhost:8080 (or the dashboard_port you passed)"
exec ros2 launch plant_simulation bringup.launch.py "$@"
