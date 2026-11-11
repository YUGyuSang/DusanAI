# DESIGN SPEC — Digital-Twin Quadruped Plant Inspection (ROS 2 + Gazebo)

This is the **binding interface contract** for every package in this repository.
If code and this file disagree, this file wins (fix the code).

## 0. Target platform & ground rules

| item | value |
|---|---|
| OS | Ubuntu 26.04 LTS (also works on 24.04 + Jazzy + Harmonic) |
| ROS 2 | **Lyrical Luth** (`/opt/ros/lyrical`) |
| Simulator | **Gazebo Jetty (gz-sim 10)** via `ros_gz_sim` / `ros_gz_bridge` |
| Python | 3.12+ (code must not use features newer than 3.12), numpy 2.x, OpenCV 4.x |
| Build | `colcon build --symlink-install`, all packages `ament_python` |

Development happens on **Windows** (no ROS here); the user copies the repo by USB to the
Ubuntu PC and runs it there. Therefore:

* Every algorithm lives in a **ROS-free module** (pure numpy/OpenCV) that is unit-tested on
  Windows with `python -m pytest` (venv with numpy, opencv-python-headless, pyyaml, scipy,
  matplotlib).  ROS nodes are thin wrappers (`*_node.py`) importing those modules.
* Node files must import cleanly-looking code: `python -m py_compile` must pass. rclpy and
  message imports only inside `*_node.py` files (or lazily inside functions).
* **Allowed ROS deps only**: rclpy, std_msgs, geometry_msgs, nav_msgs, sensor_msgs,
  visualization_msgs, tf2_ros, rosgraph_msgs, builtin_interfaces, launch, launch_ros,
  ament_index_python, ros_gz_sim, ros_gz_bridge, robot_state_publisher, xacro, rviz2
  (the Python module `tf2_ros` is installed by package **tf2_ros_py** — a package that imports
  it in Python declares `<exec_depend>tf2_ros_py</exec_depend>`).
  **Forbidden**: cv_bridge, grid_map, nav2, ros2_control/gz_ros2_control, tf_transformations,
  custom message packages. Rich data goes as JSON in `std_msgs/String`
  (`dusan_common.ros_helpers.json_msg / parse_json_msg`).
* Python deps on Ubuntu: `python3-numpy python3-opencv python3-yaml` (scipy optional, never
  required at runtime). `python3-pil` + the system font `fonts-noto-cjk` give the demo HUD its
  Korean text (`plant_simulation/package.xml` exec_depend, `scripts/install_deps.sh`); without
  them `demo_hud` falls back to English OpenCV text, so they are never required to run.
* Every node: `use_sim_time` honoured (launch passes `use_sim_time: true`), all tunables
  declared as ROS parameters with defaults, YAML config in `<pkg>/config/`.
* **rclpy logging rule**: rclpy keys every logger call site (file, line, function) and raises
  `ValueError: Logger severity cannot be changed between calls` when the same line logs with another
  severity (or other `throttle_duration_sec` / `once` / `skip_first` filters). Never route several
  severities through one line (e.g. `getattr(log, level)(text)`); use one call per severity. The stub
  loggers of the node tests enforce this rule (`dusan_common/rclpy_logger_rules.py`, `CallSiteRules`).
* No `tests_require`, no dash-separated keys in setup.cfg. Do not add ament_flake8 /
  pep257 / copyright tests.
* Packages already exist with `package.xml`, `setup.py` (console entry points fixed — see §6),
  `setup.cfg`, `resource/`.  **Do not edit package.xml/setup.py** unless the spec owner says so;
  `setup.py` auto-installs `launch/ config/ worlds/ models/ urdf/ maps/ rviz/ web/` trees.

## 1. Repository layout (ROS workspace `src/` content)

```
DusanAI/
  README.md                  project overview, install, quick start (root)
  docs/                      USAGE.md (run/usage guide, FAQ), REPORT.md, DESIGN_SPEC.md, SETUP_GUIDE.md, images/
  dusan_common/              shared lib (DONE — read it, do not change APIs)
  module1_locomotion/        terrain mapping, MPC gait controller, fall recovery, teleop
  module2_inspection/        gauge reader, thermal fusion, patrol planner
  module3_gas_safety/        gas dispersion sim, source seeker, battery, comm, RTH
  simulation/                ROS package "plant_simulation": world, URDF, assets, launch,
                             bridge, mission manager, digital-twin dashboard, metrics
  tools/                     Windows-side asset conversion + world generator + evaluation
  Doosanenerbility/          original Unity asset package (input only, never modified)
```

## 2. Frames, units, conventions

* World/Gazebo frame = ROS `map` frame = `odom` frame (ground-truth localisation; static
  identity `map -> odom`).  Origin = turbine-hall floor centre, +X east, +Y north, +Z up.
* Robot frame `base_link`: x forward, y left, z up; origin at the centre of the four hip_x
  joints. Leg order **FL, FR, RL, RR**. Joint names `fl_hip_x, fl_hip_y, fl_knee, fr_..., rl_..., rr_...`
  (`dusan_common.robot_model.JOINT_NAMES`).
* All geometry/mass/sensor numbers come from `dusan_common/robot_model.py`.
* World content (gauges, hot spots, obstacles, gas leak, APs, dock) comes from
  `simulation/config/world_layout.yaml` (`dusan_common.layout.Layout`). Nodes take a
  `layout_file` parameter (default `''` → `dusan_common.layout.default_layout_path()`).
* 2-D nav map: `simulation/maps/plant_map.yaml/.pgm` (0.10 m, map_server format, generated by
  `tools/generate_world.py`). Nodes take `map_yaml` param (default `''` →
  `Layout.load().nav_map_yaml()`). Stairs/ramps/low debris (pipes, rubble, plates) are FREE in
  the 2-D map (the legged robot crosses them); walls, machines, drums, pallets, pipe-rack
  supports, gauge stands, drop-offs are OCCUPIED. Fallback maps rasterised from the layout alone
  (map file missing) also block the walls measured in the meshes: `factory.partition` (x_min..x_max
  over the hall depth except `door_y`), `corridor.door_frame` (outside its y range) and the ramp
  edge `corridor.dropoff` (rect).
* Terrain cost layer: `dusan_common/terrain_cost.py` rasterises the FREE debris of the layout
  onto the nav-map grid as extra A* cell cost: robot centre within 0.12 m of a lying pipe or a
  rubble field 25, of a steel plate 1, of the low platform 2; margin up to 0.25 m from the item
  at 0.15 x that cost (capped at 0.25 m: the free strip at the north end of C-OBS1 is ~0.3 m).
  mission_manager, return_to_home and patrol_planner plan with it (`terrain_cost_enabled`,
  `terrain_w_pipe|rubble|plate|platform`, `terrain_margin`, `terrain_margin_fraction`; same names
  and defaults in mission.yaml, safety.yaml, patrol.yaml), and `planning.smooth_path` /
  `planning.plan(extra_cost=...)` never shortcut across cells costlier than the A* polyline.
  Costs are finite: where no free detour exists (C-OBS4 plate, C-OBS5 pipe across the corridor)
  the path still crosses; C-OBS1 (north end) and C-OBS3 (rubble y -1.3..0.5, 1.26 m passage
  along the north wall) are passed around.
* Gauge dial: `dusan_common.gauge_design` (min at 225°, max at −45°, clockwise 270°, red band =
  last 15 %).
* Cameras: Gazebo convention (looks along +X of its link). `dusan_common.camera.PinholeCamera`.

### Robot summary (see robot_model.py)
hip_x joints at (±0.2245, ±0.044, 0); hip_y at lateral 0.134; upper leg 0.2256; lower leg
0.2997; foot sphere r 0.025; mass 15 kg (base 8, hip 0.6, upper 0.9, lower 0.25 ×4).
Zero joint angles = legs straight down. hip_y>0 swings foot backwards; knee<0 bends
backwards (all legs). Nominal stand height 0.34 m. Spawn: straight legs at floor+0.56.

## 3. Gazebo side (owner: simulation agents)

* World name **`plant`**, model name **`spot`**, file `simulation/worlds/plant.sdf`
  (generated by `tools/generate_world.py`, committed).  Physics: DART, 1 ms step, RTF 1.
  World systems: Physics, UserCommands (set_pose service), SceneBroadcaster, Sensors
  (ogre2), Imu, Contact, ApplyLinkWrench, ParticleEmitter (gas visual: translucent green; the
  Harmonic ogre2 ignores `color_start`/`color_end` and paints the particles with the material
  diffuse colour, so the diffuse is green too), Thermal (per-visual temperatures on hot spots). Ambient temperature 298.15 K.
* Robot plugins (in URDF `<gazebo>` blocks):
  * 12 × `gz-sim-joint-position-controller-system` in **PID force mode**:
    `<use_velocity_commands>false</use_velocity_commands>`, `<p_gain>450</p_gain>`,
    `<d_gain>3</d_gain>`, `<i_gain>0</i_gain>`, `<cmd_max>`/`<cmd_min>` = ±joint effort
    (45 / 45 / 60 N·m), `<topic>/spot/joint/<joint>/cmd_pos</topic>` (gains:
    `robot_model.JOINT_SERVO_P_GAIN / _D_GAIN`; the controller's `servo_kp` must equal
    `p_gain`, it feeds the MPC stance torques forward as τ/p).  Changed from the velocity
    mode (`use_velocity_commands`, cmd_max 12) after the MuJoCo physics check
    (`tools/physics_check.py`, `docs/results/physics_check.json`): reaching each 100 Hz
    target within one 1 ms step made the four stance legs rigid, position-locked and
    jerked every 10 ms (feet bouncing / slipping, falls in every walking scenario).
    `d_gain` is bounded by the 0.25 kg shank (d·dt/I_knee ≪ 1 at the 1 ms step).
  * `gz-sim-joint-state-publisher-system` → gz `/world/plant/model/spot/joint_state`
  * `gz-sim-odometry-publisher-system` (`<dimensions>3</dimensions>`, odom_frame `odom`,
    robot_base_frame `base_link`, 50 Hz) → gz `/model/spot/odometry`
  * sensors: `imu` (200 Hz, topic `/imu`), `depth_camera` (topic `/depth_camera/image`,
    212×120, hfov 1.5, 10 Hz), `inspection_camera` (camera, `/inspection_camera/image`,
    800×600, hfov 0.9, 10 Hz), `thermal_camera` (type thermal, L16, `/thermal_camera/image`,
    320×240, hfov 0.8, 5 Hz, ThermalSensor plugin `resolution 0.01` → Kelvin = value×0.01).
    Sensor poses from `robot_model.SENSORS`.
* Bridge `simulation/config/ros_gz_bridge.yaml` (ros_gz_bridge parameter_bridge):

| ROS topic | ROS type | gz topic | dir |
|---|---|---|---|
| /clock | rosgraph_msgs/msg/Clock | /clock | GZ→ROS |
| /imu | sensor_msgs/msg/Imu | /imu | GZ→ROS |
| /model/spot/odometry | nav_msgs/msg/Odometry | /model/spot/odometry | GZ→ROS |
| /joint_states | sensor_msgs/msg/JointState | /world/plant/model/spot/joint_state | GZ→ROS |
| /depth_camera/image | sensor_msgs/msg/Image | /depth_camera/image | GZ→ROS |
| /depth_camera/points | sensor_msgs/msg/PointCloud2 | /depth_camera/image/points | GZ→ROS |
| /depth_camera/camera_info | sensor_msgs/msg/CameraInfo | /depth_camera/camera_info | GZ→ROS |
| /inspection_camera/image | sensor_msgs/msg/Image | /inspection_camera/image | GZ→ROS |
| /inspection_camera/camera_info | sensor_msgs/msg/CameraInfo | /inspection_camera/camera_info | GZ→ROS |
| /thermal_camera/image | sensor_msgs/msg/Image | /thermal_camera/image | GZ→ROS |
| /spot/joint/<j>/cmd_pos (×12) | std_msgs/msg/Float64 | same | ROS→GZ |
| /demo/hud_rgb | sensor_msgs/msg/Image | /demo/hud | ROS→GZ |

  **Clock relay:** gz publishes `/clock` on every 1 ms physics step (1000 msg/s). With
  `clock_relay:=true` (default) `sim.launch.py` remaps the bridge's ROS output `/clock` to
  `/clock_gz` and `plant_simulation clock_relay` (use_sim_time false) republishes it on `/clock`
  at 200 Hz of sim time (samples on the 5 ms grid, backward jumps forwarded at once, last
  sample re-sent every 0.5 s wall while paused). Every other node keeps using `/clock`; only
  the relay handles the 1 kHz stream. `clock_relay:=false` restores the direct 1 kHz `/clock`.

* Push test / reset use the `gz` CLI via subprocess (module1 `gz_tools.py`):
  `gz topic -t /world/plant/wrench/persistent -m gz.msgs.EntityWrench -p '...'`, clear with
  `/world/plant/wrench/clear`; reset with
  `gz service -s /world/plant/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 2000 --req 'name: "spot", position: {...}, orientation: {...}'`.

## 4. ROS topic contract

All JSON payloads are `std_msgs/String` produced with `dusan_common.ros_helpers.json_msg`.

### Locomotion (module1)
| topic | type | pub | sub | notes |
|---|---|---|---|---|
| /odom | nav_msgs/Odometry | plant_simulation/odom_tf | everyone | world-frame ground truth (frame `odom`, child `base_link`), twist in base frame |
| /cmd_vel | geometry_msgs/Twist | cmd_vel_mux | locomotion_controller | body velocity (vx, vy, wz) |
| /locomotion/command | std_msgs/String | teleop, mission, safety | locomotion_controller | `stand` `sit` `walk` `gait:trot` `gait:crawl` (alias `gait:walk`) `gait:auto` `mode:stable` `mode:mpc` `mode:toggle` (locomotion mode, applied while standing; a walking robot first stops at a safe gait boundary and resumes) `recover` `estop` `release` `push` `push:<fx>,<fy>` `reset_pose` |
| /locomotion/state | String JSON | locomotion_controller | dashboard, mission, metrics | `{mode: WALK|STAND|SIT|STANDING_UP|FALLEN|RECOVERING|ESTOP, locomotion_mode: stable|mpc, locomotion_mode_pending: stable|mpc|null, gait, gait_auto, phase, contacts[4], body_height, roll, pitch, fall_count, recovery_attempts, recovery_success, mpc_solve_ms, mpc_status, stability_margin, terrain, cmd[3]}` + slippery-ground fields `slip` (detection flag), `slip_mode`, `slip_source` (`detected`/`terrain`/`both`/`none`), `slip_speed` [m/s], `slip_legs[4]`, `slip_events`, `mu_est`, `mpc_mu`, `friction_prior` (μ of /terrain/info or null), `torque_ff` (module1 README §3.8; further extra keys listed in module1 README §2) |
| /locomotion/fall_event | String JSON | locomotion_controller | metrics | `{event: fall|recovered|recovery_failed|recovery_retry|assisted_reset, t, x, y, roll, pitch}` |
| /mpc/predicted_trajectory | nav_msgs/Path | locomotion_controller | rviz, dashboard | 5 s CoM prediction (the four /mpc/* topics carry data only in `locomotion_mode` = mpc; `mpc_status` = `none` in the stable mode) |
| /mpc/footholds | visualization_msgs/MarkerArray | locomotion_controller | rviz | predicted footholds over the horizon |
| /mpc/grf | visualization_msgs/MarkerArray | locomotion_controller | rviz | ground reaction force arrows |
| /mpc/predicted_torques | sensor_msgs/JointState | locomotion_controller | metrics | effort = predicted torque at t+dt; JSON summary of horizon max in /locomotion/state |
| /spot/joint/<j>/cmd_pos | std_msgs/Float64 | locomotion_controller | gz | 12 topics |
| /joint_states | sensor_msgs/JointState | gz bridge | controller, robot_state_publisher | |
| /imu | sensor_msgs/Imu | gz bridge | controller, terrain | |
| /elevation_map/points | sensor_msgs/PointCloud2 | terrain_mapper | locomotion_controller, rviz, mission_manager, return_to_home, source_seeker | ORGANISED cloud height=rows width=cols, fields x,y,z (float32, world frame `odom`), z = NaN for unknown cells, row-major, x increasing along cols, y increasing along rows. mission_manager / return_to_home / source_seeker subscribe BEST_EFFORT depth 1 (latest frame; topic = parameter `obstacle_elevation_topic`) for novel obstacles (§5 Novel obstacles); not subscribed when `obstacle_enabled` is false |
| /elevation_map/traversability | nav_msgs/OccupancyGrid | terrain_mapper | rviz, dashboard (+ mission_manager, return_to_home, source_seeker only with `obstacle_source: traversability`, topic `obstacle_traversability_topic`) | 0 = easy … 100 = untraversable, −1 unknown |
| /terrain/info | String JSON | terrain_mapper | locomotion_controller, dashboard | `{roughness, max_step, slope_deg, class: flat|rough|stairs|slope|obstacle|unknown|slippery, recommended_gait: trot|crawl, ahead_clear}` + friction prior of the layout plates / `friction_zones`: `slippery` (bool), `friction_mu` (lowest μ of the overlapping zones or null), `friction_zones` (ids), `friction_distance` (m ahead, 0 on the zone, null if none); `slippery` replaces flat / slope / unknown only (module1 TERRAIN_AND_TELEOP.md §5.1) |
| /cmd_vel/teleop | Twist | keyboard_teleop | cmd_vel_mux | |
| /teleop/active | std_msgs/Bool | keyboard_teleop | mux, mission | true while manual override |
| /keyboard/keypress | std_msgs/Int32 | gz KeyPublisher (optional bridge) | keyboard_teleop | Qt key codes |

### Inspection (module2)
| topic | type | pub | sub | notes |
|---|---|---|---|---|
| /inspection_camera/image | Image | gz | gauge_reader, thermal_fusion | rgb8 |
| /thermal_camera/image | Image | gz (or synthetic_thermal_camera) | thermal_fusion | mono16/L16 Kelvin×100 |
| /inspection/target | String (plain gauge id or "") | mission_manager | gauge_reader | gauge expected in view |
| /inspection/gauge_readings | String JSON | gauge_reader | mission, dashboard, metrics | `{stamp, readings:[{id, tag, value, unit, confidence, angle_deg, range:[min,max], status, n_samples, robot_pose:[x,y,yaw]}]}` (latest fused reading per gauge seen in this frame) |
| /inspection/gauge_overlay | Image (bgr8) | gauge_reader | dashboard, rviz | detection overlay with value text |
| /inspection/gauge_snapshot | String JSON | gauge_reader | metrics | one evidence photo per inspection visit, published when `/inspection/target` leaves the gauge (the mission accepted or skipped it): `{stamp, t_end, wall, id, tag, value, unit, status, confidence, n_samples, range:[min,max], read, frames, readings, frame_value, frame_confidence, robot_pose, distance, incidence_deg, format: "jpeg", width, height, jpeg_b64, full_jpeg_b64|null}`; `value`/`status` = final fused reading (= the value the mission accepts), `jpeg_b64` = base64 JPEG (~25 KiB) of the dial crop with fitted edge + needle and a footer (id, tag, value, unit, status, confidence, samples, sim time, wall time), `full_jpeg_b64` = downscaled full frame with the overlay; frame = the best detection whose own needle agrees with the final value (no detection: latest frame, `read` false, value null). No cv_bridge: plain JSON (module2_inspection.gauge_snapshot) |
| /inspection/thermal_alerts | String JSON | thermal_fusion | mission, dashboard, metrics | `{stamp, hotspots:[{id|null, tag|null, max_temp_c, mean_temp_c, position:[x,y,z]|null, status: normal|warning|alarm, bbox:[u,v,w,h]}]}` |
| /inspection/thermal_fusion | Image (bgr8) | thermal_fusion | dashboard, rviz | colour-mapped thermal blended on RGB |
| /patrol/route | nav_msgs/Path | patrol_planner | mission, rviz, dashboard | full A* path through viewpoints (latched) |
| /patrol/plan_json | String JSON (latched) | patrol_planner | mission_manager, dashboard | `{start:[x,y], viewpoints:[{gauge_id, x, y, yaw, score, distance, incidence_deg, pixel_diameter}], order:[gauge ids], legs:[[x,y]...] per leg, total_length}` |
| /patrol/markers | MarkerArray | patrol_planner | rviz | |

### Gas / safety (module3)
| topic | type | pub | sub | notes |
|---|---|---|---|---|
| /gas/concentration_grid | nav_msgs/OccupancyGrid | gas_dispersion_sim | dashboard, rviz | 0..100 = 0..`vis_max_ppm` (log scale allowed, document it) 2 Hz |
| /gas/field_info | String JSON | gas_dispersion_sim | dashboard | `{max_ppm, vis_max_ppm, scale: linear|log, source:[x,y], wind:[u,v], t}` |
| /gas/sensor | std_msgs/Float32 | gas_dispersion_sim | seeker, mission, dashboard, metrics | ppm at robot nose, 10 Hz, with noise + first-order sensor lag |
| /gas/sensor_array | std_msgs/Float32MultiArray | gas_dispersion_sim | seeker | 4 sensors [front_left, front_right, rear_left, rear_right] at (±0.30, ±0.15) in base frame |
| /gas/wind | geometry_msgs/Vector3Stamped | gas_dispersion_sim | seeker | local wind at robot (world frame) — anemometer |
| /gas/seek/enable | std_msgs/Bool | mission_manager | source_seeker | |
| /cmd_vel/seek | Twist | source_seeker | mux | |
| /gas/seek_status | String JSON | source_seeker | mission, dashboard, metrics | `{state: IDLE|SEARCH|SURGE|CAST|GRADIENT|FOUND|FAILED, concentration, gradient:[gx,gy], estimate:[x,y]|null, error_m|null, path_length, elapsed}` + `obstacles` (confirmed novel-obstacle cells in the seeker's memory) |
| /gas/source_estimate | geometry_msgs/PointStamped | source_seeker | dashboard, rviz | |
| /gas/seek_path | nav_msgs/Path | source_seeker | dashboard, rviz | trajectory taken while seeking |
| /battery/state | sensor_msgs/BatteryState | battery_simulator | rth, dashboard, metrics | percentage 0..1 |
| /battery/set_level | std_msgs/Float32 | teleop/mission (test) | battery_simulator | percent 0..100 |
| /comm/rssi | std_msgs/Float32 | comm_simulator | rth, dashboard | dBm at robot |
| /comm/rssi_map | nav_msgs/OccupancyGrid (latched) | comm_simulator | rth, dashboard | 0..100 ↔ −100..−30 dBm |
| /safety/status | String JSON | return_to_home | mission, dashboard, metrics | `{state: MONITOR|RTH_PLANNING|RTH_NAVIGATING|DOCKING|DOCKED|CHARGING, rth_active, reason: battery_low|comm_shadow_predicted|energy_reserve|manual|null, battery_pct, rssi_dbm, predicted_min_rssi, distance_home, energy_to_home_pct, docked}` + `obstacle` object (see below) |
| /safety/rth_active | std_msgs/Bool | return_to_home | mux, mission | |
| /cmd_vel/rth | Twist | return_to_home | mux | |

### Integration (plant_simulation)
| topic | type | pub | sub | notes |
|---|---|---|---|---|
| /map | OccupancyGrid (latched) | map_server_lite | all planners, dashboard | from plant_map.yaml |
| /cmd_vel/patrol | Twist | mission_manager | mux | |
| /nav/plan | nav_msgs/Path | mission_manager & return_to_home | rth (comm prediction), dashboard | path currently being followed |
| /cmd_vel_mux/active | String | cmd_vel_mux | dashboard | name of the selected source |
| /mission/command | String | teleop, dashboard | mission_manager, return_to_home | `start` `patrol` `seek` `rth` `pause` `resume` `abort` `dock` |
| /mission/state | String JSON | mission_manager | dashboard, metrics | `{state: IDLE|PATROL|INSPECTING|SEEK|RTH|DOCKED|MANUAL|PAUSED|DONE, target, completed:[ids], pending:[ids], gas_alarm, message}` + `obstacle` object (see below) |
| /twin/markers | MarkerArray | monitor_dashboard | rviz | gauge status labels, hot spots, leak estimate |
| /metrics/session | String JSON (latched) | metrics_logger | dashboard | `{session_dir, session, report, report_json, report_final, report_written, gauges:{<id>:{image, image_full, stamp, value, unit, status, wall}}}`: metrics session directory and its files relative to it (`gauges/<id>_<t>.jpg`, `inspection_report.html`); republished after every photo / report write. The dashboard serves them same-origin under `/session/<file>` (only for a `session_dir` inside its `metrics_root`, default `~/dusan_metrics`) and adds `image_url` / `image_full_url` per gauge and a `report` link to its snapshot |
| /demo/hud | Image (bgr8) | demo_hud | rqt_image_view, ros2 bag | robot camera (`/inspection_camera/image`) + gas ppm bar (10 / 25 ppm colours, blinking alarm frame), battery, mission and locomotion state, sim time; 640×480 at `rate` 5 Hz |
| /demo/hud_rgb | Image (rgb8) | demo_hud | ros_gz_bridge → gz `/demo/hud` | same frame for the Gazebo GUI ImageDisplay panel "Robot HUD" (gz-gui cannot decode BGR); published only while subscribed |

**cmd_vel mux priorities** (highest first, each with 0.5 s timeout): teleop (also while
`/teleop/active` true) → rth (while `/safety/rth_active`) → seek → patrol. Output `/cmd_vel`
at 20 Hz, zero when nothing is fresh. Both flags are heartbeats (keyboard_teleop republishes
`/teleop/active` every 0.5 s, return_to_home `/safety/rth_active` at 5 Hz): a flag not
re-sent for 1.5 s counts as false (mux `flag_timeout`, mission_manager
`teleop_active_timeout`), so a killed teleop cannot leave the robot stuck in manual override.

**`obstacle` object** (novel-obstacle status, `ObstacleAvoider.status()`, in `/mission/state` and `/safety/status`):
`{enabled, source: elevation|traversability, frames, rejected, confirmed (cells), free_travel (m, null when nothing
ahead), speed_scale, detour: left|right|null, detours, replans, holding, last (latest decision text), errors}`.
`detours` / `replans` are per goal; `replans` counts obstacle-triggered global re-plans only (the mission's stuck
re-plans stay in its own counters). No new topic exists for novel obstacles: the detour is visible on `/nav/plan`.

## 5. Algorithms (must be implemented — these are graded deliverables)

**Module 1** (`module1_locomotion`)
* `kinematics.py`: analytic leg FK/IK (all 4 legs), foot Jacobian, joint-limit clamp.
* `gait.py`: gait scheduler (stand, trot [period 0.40 s, duty 0.5], crawl/walk [period 1.2 s,
  duty 0.75, 4-beat FL→RR→FR→RL]), phase/contact schedule prediction over an arbitrary horizon,
  swing trajectory (Bezier, terrain-aware apex height).
* `mpc.py`: **convex centroidal MPC** (single rigid body, linearised about yaw) with a
  **5.0 s horizon** using non-uniform steps (10 × 0.05 s + 18 × 0.25 s), decision variables =
  GRFs, friction-pyramid + force bounds, swing legs forced to zero; own QP solver (`qp.py`,
  ADMM/OSQP-style, numpy only) with warm start. Outputs: predicted state trajectory (5 s),
  predicted footholds over the horizon (Raibert heuristic + terrain adjustment), GRFs and
  predicted joint torques τ = −Jᵀ Rᵀ f for every horizon step. Must solve < 30 ms typical.
  Terrain reference (`mpc_terrain_robust`, default true): height = median of map samples on the
  body footprint (left / centre / right foot lines, 5 cm behind the rear to 5 cm ahead of the front
  nominal footholds), pitch = Theil-Sen slope (median of all pairwise slopes) of the column medians,
  so a pipe or a wall-edge cell under one foot does not tilt or lift the body; false = mean of the 4
  nominal footholds.
* `stable_gait.py` (**default locomotion mode `stable`**, parameter `locomotion_mode`): CHAMP-style
  kinematic gait chosen as the default after the Gazebo comparison (module1 README §4.6) —
  gait phase generator (trot period 0.40 s duty 0.6 / 4-beat walk 1.2 s duty 0.8 with sideways
  CoM sway), Raibert footholds from the commanded velocity (+ capture feedback), Bezier swing /
  stance push, foot trajectories in the body frame (no world-frame body target), stance force
  limit (stance foot target ≤ 1.5 cm from the measured foot), IMU roll/pitch PI levelling, static
  gravity feed-forward through the PD servos, leg IK (`kinematics.py`), heading hold; with an
  elevation map: map heights / terrain pitch / swing apex over obstacles. Shares stand-up, sit,
  fall recovery, slip speed limits and gait selection with the MPC mode; the MPC is not run.
* `locomotion_controller_node.py`: 100 Hz loop; `locomotion_mode` = `stable` (default, above) or
  `mpc`: MPC at ≥ 10 Hz; stance legs via IK from the
  MPC-predicted body pose (+ IMU roll/pitch PD), swing legs via swing trajectory to the MPC
  foothold; gait: parameter `gait_mode` = `trot` (**default**, fixed trot: chosen after the Gazebo
  plant-world routes, module1 README §4.7) | `auto` (auto-switch from `/terrain/info`, teleop key 5 /
  `gait:auto`) | `crawl`; speed limits per gait; publishes topics above. Torque realisation: Gazebo servo position mode (default) — predicted torques are
  published and used for saturation-aware speed scaling.
  MPC-mode stabilisation (Gazebo, 2026-10-26; parameters in `config/locomotion.yaml`, defaults = code):

  | parameter | default | contract |
  |---|---|---|
  | `kp_attitude` / `kd_attitude` | 0.0 / 0.0 | IMU roll/pitch PD on the IK attitude (0.5 amplified the IK roll in Gazebo; `docs/results/tuning.json` `later_changes`) |
  | `mpc_stance_max_error` | 0.02 m | stance anchor ≤ this from the measured foot (horizontal), 0 = off |
  | `mpc_stance_rel_error` | 0.02 m | horizontal stance-leg deflection beyond its static sag (anchor error + body-target lead) ≤ this, 0 = off |
  | `mpc_heading_hold_gain` / `mpc_heading_hold_max` | 1.0 1/s / 0.2 rad/s | heading hold while no turn is commanded, 0 = off |
  | `mpc_terrain_robust` | true | robust terrain reference (`mpc.py`, above) |
  | `mpc_lateral_ref_blend` | 0.5 | sideways velocity of the integrated body target: prediction (0) → reference (1) |
  | `swing_guard_clearance` / `swing_guard_radius` | 0.02 m / 0.06 m | descending swing foot kept above higher mapped terrain under the measured foot, 0 = off |
  | `retarget_check_radius` / `retarget_jump_until` | 0.05 m / 0.3 | a rate-limited re-target point next to higher terrain is not used: jump whole up to this swing progress, later keep the target |
  | `mpc_land_tau`, `mpc_contact_lift`, `mpc_contact_map_tol` | 0.05 s, 0.02 m, 0.04 m | stance foot height lowered to the map (never raised); a foot resting on mapped terrain gets the measured height |
  | `mpc_turn_pivot_x` | −0.15 m | body x of the in-place turning centre (0 = base origin): adds body vy = wz · (−x) to the MPC command |
  | `mpc_turn_pivot_fade` | 0.10 m/s | pivot faded out linearly with max(raw, shaped) commanded translation speed; off at ≥ this |

  The stable mode is not affected by these keys. Default `locomotion_mode` stays `stable`: the MPC mode
  needs a solve delay ≤ ~30 ms on the 11.8° ramp with the depth map (module1 README §4.8).
* `fall_recovery.py`: state machine NORMAL → FALL_DETECTED (|roll| or |pitch| > 55° or body
  height < 0.12 m for 0.3 s) → DAMP → TUCK → SELF_RIGHT (roll back using the legs on the
  ground side) → PREPARE (belly, feet under hips) → STAND_UP (interpolated) → NORMAL; 3
  attempts then optional assisted reset (`reset_pose` via gz CLI, counted separately); FAILED is not terminal: an
  upright robot (0.3 s) counts as recovered and stands up, otherwise the 3 attempts restart after
  `recovery_retry_time` (5 s, event `recovery_retry`; 0 or `auto_recover: false` = stay FAILED); the upright height
  check uses max(map-based, foot-kinematics) height (the map under the body can be wrong right after a fall).
* `elevation_map.py`, `foothold_planner.py`, `terrain_classifier.py`, `terrain_mapping_node.py`:
  robot-centric elevation map (6 m × 6 m, 0.04 m) fused from depth points (Kalman height
  fusion with variance, ray-free), layers slope / roughness / step / traversability; foothold
  search around the nominal foothold minimising slope, roughness, edge proximity and
  displacement. Point cloud may be in camera link (x fwd) or optical (z fwd) convention —
  auto-detect; if no points, compute them from `/depth_camera/image` + intrinsics.
* `keyboard_teleop.py`: terminal raw-mode teleop (see §7) + optional `/keyboard/keypress`. The status line shows
  `mission <state>(stale Ns)` when no `/mission/state` arrived for 5 s (it is republished every 0.5 s), and a mission
  key (`t p g h o`) pressed then prints a warning that mission_manager is not answering.
* `/elevation_map/points` is also the input of the novel-obstacle layer of mission_manager, return_to_home and
  source_seeker (below); `obstacle_step_height` / `obstacle_step_window` must equal terrain.yaml `map.step_crit` /
  `map.step_window` (tested), `publish_rate` 5 Hz sets the confirmation latency (2 frames = 0.4 s).
* `gz_tools.py`: subprocess wrappers for push / reset / wrench clear.

API used by the controller (owned by the terrain agent):
```python
class ElevationMap:
    resolution: float; origin: (x0, y0) of cell [0,0] centre; shape: (rows, cols)
    elevation: np.ndarray (rows, cols) float, NaN unknown   # rows ~ y, cols ~ x
    @classmethod
    def from_cloud_xyz(cls, xyz: np.ndarray)  # (rows, cols, 3) organised cloud -> map (+layers)
    def height_at(self, x, y) -> float          # NaN if unknown (bilinear)
    def layer(self, name) -> np.ndarray         # 'elevation','variance','slope','roughness','step','traversability'
class FootholdPlanner:
    def __init__(self, search_radius=0.10, **weights)
    def select(self, emap, nominal_xy, fallback_z) -> (np.array([x, y, z]), score, valid)
def max_height_along(emap, p0_xy, p1_xy) -> float   # swing apex helper (NaN-safe)
```

**Module 2** (`module2_inspection`)
* `gauge_reader.py`: low-light enhancement (gamma + CLAHE), denoise, blur metric & frame
  gating, dial detection (Hough circles + face/rim validation), needle angle (polar unwrap,
  angular darkness profile, sub-degree refinement), scale calibration from tick ring + red
  danger band (orientation & mirror check), digit OCR (template matching on Hershey glyphs)
  for scale labels, value mapping, temporal fusion (median/MAD over N frames), gauge ID
  association via camera projection of layout gauges (needs robot pose). Target **< 5 %
  full-scale error**; `tools`/test benchmark proves it on synthetic images with blur,
  noise, low light, perspective.
* `thermal_fusion.py`: L16→Kelvin, homography registration thermal→RGB (plane-induced
  H = K_rgb (R − t nᵀ/d) K_th⁻¹ with depth d from depth image median or 2 m), colour map
  blend, hot-spot segmentation (≥ warning_k), world localisation (ray + depth or layout
  matching), alert JSON. `synthetic_thermal.py`: renders a thermal image from layout hot
  spots + pipe bodies given the robot pose (fallback when the gz thermal camera is
  unavailable; param `thermal_source: gazebo|synthetic` decides which node runs).
* `viewpoint_planner.py`: candidate viewpoints per gauge (distance 0.9–2.2 m, ±40° around
  the normal), readability score = f(pixel diameter, incidence angle, in-FOV vertical angle
  for the camera pitch, line of sight on the 2-D map, clearance), best per gauge; visiting
  order by exact Held-Karp (≤ 12 nodes) on A* path lengths from `dusan_common.planning`
  (with the terrain cost layer); route assembly. `patrol_planner_node.py` publishes `/patrol/*` (latched) at startup / on
  `/map`.

**Module 3** (`module3_gas_safety`)
* `gas_model.py`: 2-D advection–diffusion on a 0.25 m grid over the nav-map extent;
  obstacles are no-flux; wind field = mass-consistent (divergence-free projection, Jacobi
  pressure solve) from `wind_base` + inflow vents + corridor outflow, plus gusts; explicit
  upwind advection + diffusion with CFL substeps; first-order decay; config
  `module3_gas_safety/config/gas_dispersion.yaml` (source/rate/wind/diffusivity/gust/sensor
  noise/lag; source position defaults to layout `gas_leak.source`).
* `source_seeking.py`: gradient-descent (on −c) seeker: gradient from the 4-sensor array +
  least-squares plane fit over a sliding window of samples, momentum, step size control,
  upwind SURGE when the gradient is weak, CAST (cross-wind zig-zag) when the plume is lost,
  obstacle-aware step projection on the nav map, FOUND when c > `found_ppm` and local max
  confirmed; estimate = weighted centroid of top samples.
* `battery_model.py`: SoC model (idle + speed-proportional + sensor load drain, charging on
  dock).  `comm_model.py`: log-distance path loss + wall attenuation (ray-cast nav map) +
  shadow zones → RSSI map.  `rth_logic.py`: triggers on battery < 20 %, predicted comm
  shadow along `/nav/plan` (or velocity extrapolation, 4 s lookahead), or energy reserve
  (energy to home × 1.3 > remaining); plans A* home (terrain cost layer), pure pursuit, docking alignment,
  `sit` on dock, `CHARGING` until 95 %. Novel obstacles (below): steps (a)–(c) on the way home, planning tiers
  full shadow → near shadow → nav map alone, the velocity safety filter and the stuck back-off include detected
  obstacles, DOCKING starts only when the straight line to the dock is free of detected obstacles (governor while
  docking, no progress for `stuck_time_s` → re-plan).
* `source_seeking.py` + novel obstacles: an `ObstacleMemory` (memory keys only); its `near` layer is merged into the
  seeker's nav grid at most every `obstacle_rebuild_period` (1.0 s) while seeking (idle: memory only, grid rebuilt on
  enable), so free_length / detours / the velocity projection respect detected obstacles; an active route or step that
  now crosses one is re-planned.

**Integration** (`plant_simulation`)
* `map_server_node.py`, `odom_tf_node.py` (raw `/model/spot/odometry` → `/odom` + TF
  `odom→base_link`, static `map→odom`; auto-detects a spawn-relative odometry and adds the
  layout spawn offset), `cmd_vel_mux_node.py`, `mission_manager_node.py` (state machine:
  patrol viewpoints in plan order, dwell 4 s with `/inspection/target`, gas alarm (> 25 ppm)
  → SEEK → report → resume, yields to RTH, `start` = full mission, ends with RTH),
  `monitor_dashboard_node.py` (digital-twin web UI on http://localhost:8080, stdlib
  http.server + Server-Sent Events, static files in `simulation/web/`),
  `metrics_logger_node.py` (CSV/JSON under `~/dusan_metrics/<date>/`; inspection photos `gauges/<id>_<t>.jpg`
  and `inspection_report.json` / `.html` with the mission's official per-gauge result, ground truth, error,
  photo, thermal and gas results - written on every gauge result / photo and at DONE / DOCKED),
  `clock_relay_node.py` (`/clock_gz` → `/clock` decimated to 200 Hz, see section 3),
  `demo_hud_node.py` (HUD frame drawn by the ROS-free `demo_hud.py`: `/demo/hud`, `/demo/hud_rgb`),
  `follow_camera.py` (gz CLI only: waits for Gazebo, the robot and the GUI, then sends a
  `gz.msgs.CameraTrack` FOLLOW on the GUI topic `/gui/track`, falls back to the Harmonic
  `/gui/follow` service; always exits 0). mission_manager plans with the terrain cost layer
  (section 2) and avoids novel obstacles (below: steps (a)–(d), patrol legs validated against detected obstacles).
  mission_manager robustness: every callback is wrapped by `plant_simulation.node_utils.guarded` (exception logged
  with traceback on the 1st and every 50th failure, the node keeps spinning); 3 consecutive `MissionLogic.step`
  exceptions abort the mission so that `start` begins afresh. A FALLEN robot during a mission gets `recover` every
  `recover_after` (20 s); after STAND the mission re-plans from the current pose.
  mission_manager logging: mission events `(level, text)` are logged through `_log_event` (one rclpy call
  site per severity: `warn`/`alarm` → warning, `error` → error, else info) AFTER the tick's commands
  (`/locomotion/command`, `/inspection/target`, `/gas/seek/enable`, `/nav/plan`, `/cmd_vel/patrol`,
  `/mission/command`) are published, so logging can never hold back that tick's outputs (§0 logging rule;
  field report 2026-10-26: one shared line died on the first warning after info events).
* Launch: `simulation/launch/sim.launch.py` (gz + spawn + bridge + clock_relay + rsp + odom_tf + map),
  `simulation/launch/bringup.launch.py` (everything; args `gui`, `rviz`, `thermal_source`,
  `mission_autostart`, `dashboard_port`, `demo` (false: true = mission_autostart + follow camera +
  HUD, `scripts/run_demo.sh`), `hud` (true), `follow_camera` ('' = demo, only with gui),
  `follow_offset` (-2.5,0.0,1.6)), each module has `launch/<module>.launch.py`.
* RViz config `simulation/rviz/plant.rviz`.

**Novel (unmapped) obstacles** (`dusan_common/obstacle_layer.py`, ROS-free, numpy only; shared by mission_manager
(patrol + self-RTH), return_to_home and source_seeker; enabled by default, `obstacle_enabled: false` = exact legacy
behaviour, no subscription)
* Evidence (`EvidenceFrame`): NaN-aware max − min over a 5 × 5-cell window of the `/elevation_map/points` heights
  ≥ 0.22 m. A free cell may clear earlier evidence only when its whole window was observed. Optional
  `obstacle_source: traversability`: OccupancyGrid cells ≥ `obstacle_cost_threshold` (not the default: its planar-slope
  rule flags the C-OBS1 pipe flanks, ramp and stair edges — 44 false confirmed cells on the real route vs 0 with the
  step test). Frame ids must be in `obstacle_frames` (`odom,map,world`), others are rejected and counted; no TF
  lookups (map = odom).
* Memory (`ObstacleMemory`): robot-centred rolling lattice (20 m, 0.10 m, anchored at the world origin); confirmed
  after 2 evidence frames, cleared after up to 5 free observations, forgotten after 60 s without evidence; evidence
  within 0.55 m of base_link, beyond 3.5 m or within 0.30 m of an occupied nav-map cell is ignored; 3 × 3 speckle filter
  (also for the unconfirmed layer). Layers: `output` (confirmed), `raw` (latest frame, ≤ 1 s old), `planning`
  (output + the unseen back of each obstacle assumed `obstacle_shadow_depth` 1.0 m deep along the line of sight from
  where it was first seen, until a cell seen free after the evidence; unobserved face ends extended by the same
  length), `near` (output + 0.3 m isotropic shadow: fallback where the full shadow closes a real gap).
* `ObstacleAvoider.step` once per control tick (callers catch exceptions, count them in `errors` and keep the legacy
  behaviour for that tick):
  (a) governor — free travel along the line the pure-pursuit follower drives (robot → look-ahead vertex → path)
  < `obstacle_slow_dist` 1.0 m: linear slow-down; < `obstacle_stop_dist` 0.35 m: zero command including wz;
  unconfirmed evidence slows to no less than 0.5 except within the stop distance; the look-ahead is halved where the
  chord cuts a corner towards an obstacle or across mapped debris (terrain cost) the path goes round.
  (b) a confirmed obstacle blocks the followed path within `obstacle_lookahead` 2.5 m (check radius robot_radius +
  margin/2) → local detour: A* per side in a window (≤ 16 m) with a hard cut, sides compared by length + terrain cost,
  rejoin ≥ `obstacle_rejoin_gap` 0.8 m beyond the blocked section where the route stays clear up to the look-ahead (an
  obstacle right behind is merged), side committed for 15 s per obstacle, ≤ 8 per goal; planning layer first, the near
  layer when no detour exists or when it is ≥ `obstacle_switch_ratio` 1.5 × cheaper; a followed detour is re-checked
  on its layer up to the rejoin point (a face confirmed later refines it).
  (c) no detour → global re-plan on static + detected (planning layer; the near layer when that has no path or its
  route is ≥ `obstacle_switch_ratio` × cheaper by length + terrain cost — the same rule for every mission / RTH plan
  made while detected obstacles exist; ≤ 3 re-plans per goal); detected cells are
  grown by `obstacle_margin` 0.10 m, mapped walls keep the plain robot_radius (narrow mapped gaps stay open).
  (d) no path → keep the map route (`obstacle_static_fallback`); the caller's stuck / back-up / skip logic decides
  (mission: no back-up into a remembered obstacle behind).
* A* first checks reachability with connected-component labels (cached per radius), so a request without a path
  fails fast. While the owner does not drive (mission not driving, RTH MONITOR / DOCKED / CHARGING, seeker inactive)
  frames are fused once per `obstacle_idle_period` (1.0 s).

| parameters (prefix `obstacle_`) | defaults | meaning |
|---|---|---|
| `enabled`, `detour_enabled`, `replan_enabled`, `slowdown_enabled`, `static_fallback` | true | master switch; steps (b), (c), (a); (d) keep the map route (false = skip at once, 'no path') |
| `source`, `elevation_topic`, `traversability_topic`, `frames` | `elevation`, `/elevation_map/points`, `/elevation_map/traversability`, `odom,map,world` | input selection (str node parameters) |
| `step_height`, `step_window`, `cost_threshold`, `max_range` | 0.22 m, 5, 100, 3.5 m | evidence |
| `resolution`, `memory_size` | 0.10 m, 20.0 m | memory lattice |
| `confirm_hits`, `max_hits`, `min_known_fraction` | 2, 5, 0.5 | confirmation / clearing (known share of a cell for a free observation) |
| `ignore_radius`, `forget_time`, `min_cluster`, `static_tolerance` | 0.55 m, 60.0 s, 2, 0.30 m | footprint, forgetting (0 = never), speckle filter, mapped-structure tolerance (clamped below 2 r − resolution) |
| `shadow_depth`, `idle_period` | 1.0 m, 1.0 s | unseen back (planning only, ≤ 5 m), frame period while not driving (0 = every frame) |
| `margin`, `lookahead`, `detour_window`, `rejoin_gap` | 0.10 m, 2.5 m, 2.5 m, 0.8 m | detour geometry |
| `detour_max_ratio`, `switch_ratio`, `commit_time`, `min_interval` | 3.0, 1.5, 15.0 s, 0.5 s | detour ≤ ratio × replaced route + 2 m; side commitment / near-layer choice; rate limit |
| `max_detours`, `max_replans` | 8, 3 | per goal |
| `slow_dist`, `stop_dist`, `unconfirmed_scale` | 1.0 m, 0.35 m, 0.5 | governor (free travel of the robot_radius footprint) |
| `rebuild_period` (source_seeker only) | 1.0 s | nav-grid rebuild period |

mission.yaml (mission_manager) and safety.yaml (return_to_home) carry every key; seeker.yaml (source_seeker) carries
the memory / input keys (`OBSTACLE_MEMORY_KEYS`) + `obstacle_rebuild_period`; values equal `OBSTACLE_DEFAULTS`.

## 6. Console entry points (already in setup.py — implement `main()` in these modules)

| package | executable | module |
|---|---|---|
| module1_locomotion | locomotion_controller | module1_locomotion.locomotion_controller_node |
| | terrain_mapper | module1_locomotion.terrain_mapping_node |
| | keyboard_teleop | module1_locomotion.keyboard_teleop |
| module2_inspection | gauge_reader | module2_inspection.gauge_reader_node |
| | thermal_fusion | module2_inspection.thermal_fusion_node |
| | synthetic_thermal_camera | module2_inspection.synthetic_thermal_node |
| | patrol_planner | module2_inspection.patrol_planner_node |
| module3_gas_safety | gas_dispersion_sim | module3_gas_safety.gas_dispersion_node |
| | source_seeker | module3_gas_safety.source_seeker_node |
| | battery_simulator | module3_gas_safety.battery_node |
| | comm_simulator | module3_gas_safety.comm_node |
| | return_to_home | module3_gas_safety.return_to_home_node |
| plant_simulation | map_server_lite | plant_simulation.map_server_node |
| | odom_tf | plant_simulation.odom_tf_node |
| | cmd_vel_mux | plant_simulation.cmd_vel_mux_node |
| | mission_manager | plant_simulation.mission_manager_node |
| | monitor_dashboard | plant_simulation.monitor_dashboard_node |
| | metrics_logger | plant_simulation.metrics_logger_node |
| | clock_relay | plant_simulation.clock_relay_node |
| | demo_hud | plant_simulation.demo_hud_node |
| | follow_camera | plant_simulation.follow_camera |

## 7. Keyboard teleop keys (terminal; also Gazebo KeyPublisher codes)

```
 w/s : forward / backward        a/d : turn left / right       q/e : strafe left / right
 x   : stop (zero velocity)      space: E-STOP toggle           +/- : speed scale
 1 stand  2 sit  3 trot (default)  4 crawl  5 auto-gait   r recover   f push test (fall)  0 reset pose
 v   : locomotion mode stable <-> MPC (/locomotion/command mode:toggle, switches while standing)
 m   : manual override toggle (teleop keeps priority until pressed again)
 p patrol   g gas seek   h return home   t start full mission   o pause/resume
 b   : battery test -> 19 %          n : battery 100 %
 ESC / Ctrl-C : quit
```

## 8. Evaluation outputs (for docs/REPORT.md)
Each module provides an offline evaluation script under `tools/eval_*.py` that runs on Windows
(no ROS) and writes figures to `docs/images/` and numbers to `docs/results/*.json`:
`eval_locomotion.py` (MPC disturbance rejection on a rigid-body sim, solve times, gait
stability margin), `eval_gauge.py` (≥ 300 synthetic images, error % by condition),
`eval_thermal.py` (registration error, detection precision/recall), `eval_patrol.py`
(viewpoint scores, route length vs naive order), `eval_gas.py` (Monte-Carlo source
seeking success / error / time, RTH trigger tests).
