#!/usr/bin/env bash
# One-command demo in the style of the reference video:
#   * Gazebo GUI camera follows the robot from behind / above (third person);
#     a view set with the mouse in the 3D view is kept, keys in the Gazebo
#     window: c = camera stays / follows again from there, z = default view,
#   * panel "Robot HUD" (top left of the 3D view) = robot camera + gas ppm,
#     battery, locomotion and mission state; also on ROS as /demo/hud,
#   * the full mission starts by itself (autostart_delay s after launch):
#     corridor -> factory -> gauges -> gas leak search -> back to the dock,
#   * green gas plume at the leak valve, dashboard on http://localhost:8080.
#
#   bash scripts/run_demo.sh
#   bash scripts/run_demo.sh autostart_delay:=15.0 follow_offset:="-3.0 0.0 1.8"
#   bash scripts/run_demo.sh thermal_source:=synthetic rviz:=true
#   bash scripts/run_demo.sh --low-memory           # 8 GB laptop (= low_memory:=true)
#   bash scripts/run_demo.sh follow_mouse:=false    # no mouse view (the GUI pulls the camera back;
#                                                   # c / z and the fall hold stay)
#
# All arguments are passed to: ros2 launch plant_simulation bringup.launch.py demo:=true
# (--low-memory is translated to low_memory:=true, see parse_launch_flags in common.sh)
set -e
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_launch_flags "$@"
source_ws
info "ros2 launch plant_simulation bringup.launch.py demo:=true ${LAUNCH_ARGS[*]}"
info "Gazebo: the view follows the robot; panel 'Robot HUD' shows the robot camera with the gas ppm"
info "HUD on ROS: ros2 run rqt_image_view rqt_image_view /demo/hud"
info "camera: wheel / middle-drag / left-drag in the 3D view sets the view (kept while following);"
info "        Gazebo window keys: c = camera stays (c again: follow from there), z = default view"
info "dashboard: http://localhost:8080 (or the dashboard_port you passed)"
exec ros2 launch plant_simulation bringup.launch.py demo:=true "${LAUNCH_ARGS[@]}"
