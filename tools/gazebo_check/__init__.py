"""Headless Gazebo (gz-sim + DART) check of the locomotion stack (dev tool).

* :mod:`gazebo_check.world`     - harness URDF (cameras off, no mesh visuals), test world SDF
                                  (physics block + systems of simulation/worlds/plant.sdf), the
                                  REAL plant world with every collision unchanged, ground-truth
                                  height field / ray casts; pure python;
* :mod:`gazebo_check.mapper`    - depth camera emulation (ray cast on the collisions) + the
                                  terrain_mapper code path (ElevationMap, TerrainClassifier);
                                  pure python;
* :mod:`gazebo_check.harness`   - gz-sim TestFixture + gz-transport around the REAL ROS-free
                                  LocomotionController (needs the gz python bindings: gz.sim8 /
                                  gz.transport13 = Harmonic; Ionic / Jetty module names are tried);
* :mod:`gazebo_check.scenarios` - scenarios (a) .. (s) with pass criteria, keyboard operator;
                                  MISSION_SCENARIOS (r) .. (z): the real MissionLogic in the loop
                                  (MissionOperator) with novel boxes that are not in the nav map;
* :mod:`gazebo_check.mission_stack` - every node of the demo mission ROS free (mission manager,
                                  patrol planner, gas dispersion + seeker, battery, comm, RTH,
                                  cmd_vel mux) and the harness operator that ticks them;
* :mod:`gazebo_check.plots`     - docs/images/gazebo_*.png.

CLI: tools/gazebo_check.py (a .. s), tools/obstacle_check.py (mission scenarios r .. z),
tools/fall_check.py (fall recovery), tools/mission_check.py (the whole demo mission).
Nothing in module1_locomotion imports gz.
"""
