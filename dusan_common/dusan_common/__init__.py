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
terrain_cost  layout debris as extra A* cost (pipes, rubble, plates, platform)
obstacle_layer  novel (unmapped) obstacles from perception: memory, detours,
              re-plan grid, speed governor (ObstacleAvoider)
obstacle_sim  synthetic elevation-map evidence for tests / evaluations
"""
