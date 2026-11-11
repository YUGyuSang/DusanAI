"""Shared library for the Doosan Enerbility virtual-plant quadruped project.

Modules
-------
robot_model   canonical robot geometry / joint names / sensor extrinsics
geometry      rotations, quaternions, poses (numpy)
grid_map      OccupancyGridMap (map_server YAML/PGM), distance field, ray casts
planning      A* + path smoothing, nav_msgs/Path conversion
pure_pursuit  path follower -> (vx, vy, wz)
camera        pinhole model matching Gazebo cameras
image_utils   sensor_msgs/Image <-> numpy without cv_bridge
layout        access to simulation/config/world_layout.yaml
gauge_design  analog gauge dial renderer + angle/value mapping
ros_helpers   QoS presets, JSON-over-String helpers, marker factory
launch_utils  helpers of the launch files (respawn arguments per launch version)
terrain_cost  layout debris as extra A* cost (pipes, rubble, plates, platform);
              debris-blocked nav map for the gas seeker
debris_crossing  lying-pipe crossings: straight square crossing segments,
              crossing velocity governor (CrossingGovernor)
obstacle_layer  novel (unmapped) obstacles from perception: memory, detours,
              re-plan grid, speed governor (ObstacleAvoider)
obstacle_sim  synthetic elevation-map evidence for tests / evaluations
"""
