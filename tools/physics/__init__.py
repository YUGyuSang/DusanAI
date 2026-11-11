"""
MuJoCo physics validation of the locomotion stack (development / validation
only - nothing under module1_locomotion imports this package).

* :mod:`physics.mjcf`      URDF (simulation/urdf/spot.urdf) -> MJCF robot + terrain
* :mod:`physics.terrain`   scenario terrain primitives (floor, pipes, steps, plates, stairs, ramp)
* :mod:`physics.sim`       gz-like plant (JointPositionController as in the URDF,
                           odometry / IMU / joint-state rates) + controller harness
* :mod:`physics.scenarios` scenario definitions (a) .. (j) with pass criteria
* :mod:`physics.plots`     timelines, trajectories, torque usage, stick figures

Requires ``mujoco`` (pip install mujoco); see tools/physics_check.py.
"""
