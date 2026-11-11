#!/usr/bin/env bash
# One-command demo in the style of the reference video:
#   * Gazebo GUI camera follows the robot from behind / above (third person),
#   * panel "Robot HUD" (top left of the 3D view) = robot camera + gas ppm,
#     battery, locomotion and mission state; also on ROS as /demo/hud,
#   * the full mission starts by itself (autostart_delay s after launch):
#     corridor -> factory -> gauges -> gas leak search -> back to the dock,
#   * green gas plume at the leak valve, dashboard on http://localhost:8080.
#
#   bash scripts/run_demo.sh
#   bash scripts/run_demo.sh autostart_delay:=15.0 follow_offset:="-3.0 0.0 1.8"
#   bash scripts/run_demo.sh thermal_source:=synthetic rviz:=true
#
# All arguments are passed to: ros2 launch plant_simulation bringup.launch.py demo:=true
set -e
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source_ws
info "ros2 launch plant_simulation bringup.launch.py demo:=true $*"
info "Gazebo: the view follows the robot; panel 'Robot HUD' shows the robot camera with the gas ppm"
info "HUD on ROS: ros2 run rqt_image_view rqt_image_view /demo/hud   (free camera: click the 3D view, press Esc)"
info "dashboard: http://localhost:8080 (or the dashboard_port you passed)"
exec ros2 launch plant_simulation bringup.launch.py demo:=true "$@"
