"""Integrated Gazebo (gz-sim) virtual plant: worlds, Spot URDF, assets, bridge, mission manager, digital-twin dashboard

ROS-free modules (unit tested without ROS)
------------------------------------------
mux_logic        cmd_vel priority multiplexer (teleop > rth > seek > patrol)
clock_relay      /clock decimation rule (1 kHz Gazebo clock -> 200 Hz /clock)
odom_logic       spawn-relative odometry detection, body twist estimation
layout_map       fallback 2-D nav map rasterised from world_layout.yaml
mission_logic    mission state machine (patrol / inspect / gas seek / RTH / dock)
dashboard_state  digital-twin aggregator (snapshots, ground-truth join, event log)
dashboard_render map / gas / RSSI PNG layers, camera JPEG encoding
web_server       stdlib HTTP + Server-Sent Events server of the control room
metrics          CSV / summary.json recorder of a run
dashboard_demo   offline dashboard demo (python -m plant_simulation.dashboard_demo);
                 --headless --metrics-dir <dir> = end-to-end evaluation run

ROS nodes (thin wrappers, console entry points in setup.py)
-----------------------------------------------------------
map_server_node, odom_tf_node, cmd_vel_mux_node, mission_manager_node,
monitor_dashboard_node, metrics_logger_node, clock_relay_node  (shared glue: node_utils)
"""
