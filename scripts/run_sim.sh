#!/usr/bin/env bash
# Gazebo virtual plant + robot + bridge only (no autonomy).
#
#   bash scripts/run_sim.sh
#   bash scripts/run_sim.sh gui:=false headless_rendering:=true
#
# All arguments are passed to: ros2 launch plant_simulation sim.launch.py
set -e
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source_ws
info "ros2 launch plant_simulation sim.launch.py $*"
exec ros2 launch plant_simulation sim.launch.py "$@"
