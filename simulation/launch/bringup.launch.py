"""
Complete digital-twin bring-up: simulation + all three modules + integration.

    ros2 launch plant_simulation bringup.launch.py
    ros2 launch plant_simulation bringup.launch.py rviz:=true thermal_source:=synthetic
    ros2 launch plant_simulation bringup.launch.py mission_autostart:=true gui:=false headless_rendering:=true
    ros2 launch plant_simulation bringup.launch.py demo:=true        # = scripts/run_demo.sh
    ros2 launch plant_simulation bringup.launch.py low_memory:=true  # 8 GB laptop (scripts/run_*.sh --low-memory)

Starts
  * sim.launch.py (Gazebo, spawn, bridge, robot_state_publisher, odom_tf, map_server_lite)
  * after module_delay s (TimerAction):
      module1_locomotion/launch/locomotion.launch.py
      module2_inspection/launch/inspection.launch.py   (thermal_source gazebo|synthetic)
      module3_gas_safety/launch/gas_safety.launch.py
  * plant_simulation cmd_vel_mux, mission_manager, monitor_dashboard (http://localhost:<dashboard_port>),
    metrics_logger (metrics:=true), rviz2 (rviz:=true)
  * plant_simulation demo_hud (hud:=true, default): robot camera + gas ppm / battery / mission overlay on
    /demo/hud (bgr8) and /demo/hud_rgb (rgb8, bridged to the Gazebo GUI panel "Robot HUD")
  * plant_simulation follow_camera (follow_camera:=true, default = demo, only with gui:=true):
    the Gazebo GUI camera follows the robot from behind / above (gz topic /gui/track); a view set
    with the mouse in the 3D view is kept (follow_mouse), keys c / z in the Gazebo window switch
    following off / on and reset the view, the default view keeps in front of the building walls
    (follow_walls)

Demo mode (demo:=true, scripts/run_demo.sh) = the reference video: mission_autostart (corridor ->
factory -> gauges -> gas leak search -> return to the dock), follow camera and HUD on.

Low-memory profile (low_memory:=true, for an 8 GB laptop; every item can still be overridden by its
own argument): inspection camera 640 x 480 @ 5 Hz (gauge reading verified at this size), thermal
camera 2 Hz, world without shadows, lean Gazebo GUI (config/gui_lean.config: no camera panels /
component inspector / entity tree), no RViz, HUD at 3 Hz, MALLOC_ARENA_MAX=2 and one BLAS / OpenMP
thread per process.  The depth camera keeps its 10 Hz: the walking over pipes and debris (obstacle
pace, foothold planner, crossing) was tuned and checked on 10 Hz depth frames, and the 212 x 120
depth image costs little.  Dashboard, metrics and saved results are unchanged.

Respawn (respawn:=true, default): cmd_vel_mux, mission_manager, monitor_dashboard and demo_hud are
restarted when they die (3 s delay), as are the stateless nodes of sim.launch.py and Module 3 (see
there).  A restarted mission_manager comes back IDLE with the robot held still - the mission
progress is lost, but 't' starts again (without the restart 't' would do nothing until the whole
bring-up is restarted); results already saved by metrics_logger / the inspection nodes are kept.
metrics_logger is not respawned (a restart would split the run into two result folders), nor is
follow_camera (once the camera follows it only keeps its fall guard on /locomotion/state, and it
exits when there is no rclpy or with --no-fall-guard).

Arguments (defaults)
  gui (true)  rviz (false)  thermal_source (gazebo)  mission_autostart (false)
  autostart_delay (25.0)  dashboard_port (8080)  use_sim_time (true)  world ('')
  layout_file ('')  map_yaml ('')  module_delay (6.0)  metrics (true)
  teleop_terminal (none: none|xterm|gnome-terminal|konsole|gz)  headless_rendering (false)
  demo (false)  hud (true)  follow_camera ('' = demo)  follow_offset (-2.5,0.0,1.6)
  follow_mouse (true)  follow_walls (true)
  low_memory (false)  respawn (true)  inspection_rate inspection_width inspection_height depth_rate
  thermal_rate shadows gui_config hud_rate ('' = profile default)
"""
import os

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, SetEnvironmentVariable, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from dusan_common.launch_utils import respawn_kwargs

PKG = "plant_simulation"
MODULE_LAUNCHES = (
    ("module1_locomotion", "locomotion.launch.py"),
    ("module2_inspection", "inspection.launch.py"),
    ("module3_gas_safety", "gas_safety.launch.py"),
)
# low_memory:=true defaults of the sim.launch.py / demo_hud arguments ('' in the normal profile = the
# xacro / node defaults: inspection 800 x 600 @ 10 Hz, depth 10 Hz, thermal 5 Hz, shadows as in the world;
# the depth camera stays at the 10 Hz the walking was checked with, see the module docstring)
LOW_MEMORY = {
    "inspection_rate": "5",
    "inspection_width": "640",
    "inspection_height": "480",
    "depth_rate": "10",
    "thermal_rate": "2",
    "shadows": "false",
    "gui_config": "gui_lean.config",
    "hud_rate": "3.0",
}
SIM_PROFILE_ARGS = ("inspection_rate", "inspection_width", "inspection_height", "depth_rate", "thermal_rate",
                    "shadows", "gui_config")
# processes started by the bring-up that can be restarted without losing anything that matters
RESPAWN_NODES = ("cmd_vel_mux", "mission_manager", "monitor_dashboard", "demo_hud")


def _truthy(text):
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _respawn_kwargs(enabled, delay=3.0, max_retries=10):
    """respawn / respawn_delay (every distro) + respawn_max_retries where launch has it (Jazzy and later:
    a parameter of ExecuteProcess' base class ExecuteLocal, see dusan_common.launch_utils)."""
    return respawn_kwargs(ExecuteProcess, enabled, delay, max_retries)


def _layout_names(path):
    """(world_name, robot_name) of world_layout.yaml (defaults plant / spot)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return str(data.get("world_name", "plant")), str(data.get("robot_name", "spot"))
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        return "plant", "spot"


def _module_include(pkg, launch_file, launch_args):
    """IncludeLaunchDescription for <pkg>/launch/<file>, or a log line if it is missing."""
    try:
        path = os.path.join(get_package_share_directory(pkg), "launch", launch_file)
    except PackageNotFoundError:
        return LogInfo(msg=f"[bringup] ERROR: package {pkg} not found (not built / not sourced) - skipped")
    if not os.path.exists(path):
        return LogInfo(msg=f"[bringup] ERROR: {path} not found - {pkg} skipped")
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=launch_args.items())


def _launch_setup(context, *args, **kwargs):
    share = get_package_share_directory(PKG)
    lc = lambda name: LaunchConfiguration(name).perform(context).strip()  # noqa: E731
    use_sim_time = _truthy(lc("use_sim_time"))
    thermal_source = (lc("thermal_source") or "gazebo").lower()
    if thermal_source not in ("gazebo", "synthetic"):
        raise RuntimeError(f"thermal_source must be gazebo or synthetic, got '{thermal_source}'")
    layout_file = lc("layout_file")
    map_yaml = lc("map_yaml")
    sim_time_str = "true" if use_sim_time else "false"
    gui = _truthy(lc("gui") or "true")
    demo = _truthy(lc("demo"))
    autostart = demo or _truthy(lc("mission_autostart"))
    hud = _truthy(lc("hud") or "true")
    follow = (_truthy(lc("follow_camera")) if lc("follow_camera") else demo) and gui
    low = _truthy(lc("low_memory"))
    respawn = _truthy(lc("respawn") or "true")

    def prof(name):
        """Explicit argument, else the low-memory value, else '' (= the default of the target)."""
        return lc(name) or (LOW_MEMORY.get(name, "") if low else "")

    def cfg(name):
        p = os.path.join(share, "config", name)
        return [p] if os.path.exists(p) else []

    actions = []
    if low:
        # ~18 numpy processes next to Gazebo: one BLAS / OpenMP thread each (no per-core pools of
        # spinning threads competing with the physics), and at most 2 glibc malloc arenas per process
        # (multi-threaded executors otherwise keep up to 8 x cores arenas of freed memory resident)
        actions += [SetEnvironmentVariable(k, os.environ.get(k, "1"))
                    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")]
        actions.append(SetEnvironmentVariable("MALLOC_ARENA_MAX", os.environ.get("MALLOC_ARENA_MAX", "2")))
    rviz = _truthy(lc("rviz"))
    actions.append(LogInfo(msg=f"[bringup] thermal_source={thermal_source} rviz={rviz and not low} "
                               f"low_memory={low} respawn={respawn} autostart={autostart} "
                               f"dashboard port {lc('dashboard_port')}"))
    if low:
        actions.append(LogInfo(msg="[bringup] LOW MEMORY: inspection camera "
                                   f"{prof('inspection_width')}x{prof('inspection_height')} @ "
                                   f"{prof('inspection_rate')} Hz, depth {prof('depth_rate')} Hz, thermal "
                                   f"{prof('thermal_rate')} Hz, shadows {prof('shadows') or 'world'}, GUI "
                                   f"{prof('gui_config')}, HUD {prof('hud_rate')} Hz, MALLOC_ARENA_MAX=2"
                                   + (" - RViz skipped (low_memory)" if rviz else "")))
    if demo:
        actions.append(LogInfo(msg=f"[bringup] DEMO: mission starts {lc('autostart_delay') or '25.0'} s after "
                                   "start (corridor -> factory -> gauges -> gas leak -> dock), Gazebo camera "
                                   "follows the robot, HUD panel 'Robot HUD' / ROS topic /demo/hud"))

    # ------------------------------------------------------------ simulation
    sim_args = {
        "gui": lc("gui") or "true",
        "world": lc("world"),
        "use_sim_time": sim_time_str,
        "layout_file": layout_file,
        "map_yaml": map_yaml,
        "headless_rendering": lc("headless_rendering") or "false",
        # with the synthetic thermal camera the Gazebo one must not publish the same topic
        "enable_thermal": "false" if thermal_source == "synthetic" else "true",
        "respawn": "true" if respawn else "false",
    }
    sim_args.update({name: prof(name) for name in SIM_PROFILE_ARGS})
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "sim.launch.py")),
        launch_arguments=sim_args.items()))

    # ------------------------------------------------------- module launches
    common_args = {"use_sim_time": sim_time_str, "map_yaml": map_yaml, "layout_file": layout_file}
    extra_args = {"module1_locomotion": {"teleop_terminal": lc("teleop_terminal") or "none"},
                  "module2_inspection": {"thermal_source": thermal_source},
                  "module3_gas_safety": {"respawn": "true" if respawn else "false"}}
    modules = [_module_include(pkg, launch_file, dict(common_args, **extra_args.get(pkg, {})))
               for pkg, launch_file in MODULE_LAUNCHES]
    actions.append(TimerAction(period=float(lc("module_delay") or "6.0"), actions=modules))

    # ------------------------------------------------------- integration nodes
    common = {"use_sim_time": use_sim_time, "layout_file": layout_file}
    rs = lambda name: _respawn_kwargs(respawn and name in RESPAWN_NODES)  # noqa: E731
    actions.append(Node(package=PKG, executable="cmd_vel_mux", name="cmd_vel_mux", output="screen",
                        parameters=cfg("mux.yaml") + [{"use_sim_time": use_sim_time}], **rs("cmd_vel_mux")))
    actions.append(Node(
        package=PKG, executable="mission_manager", name="mission_manager", output="screen",
        parameters=cfg("mission.yaml") + [dict(common, map_yaml=map_yaml,
                                               autostart=autostart,
                                               autostart_delay=float(lc("autostart_delay") or "25.0"))],
        **rs("mission_manager")))
    actions.append(Node(
        package=PKG, executable="monitor_dashboard", name="monitor_dashboard", output="screen",
        parameters=cfg("dashboard.yaml") + [dict(common, port=int(lc("dashboard_port") or "8080"))],
        **rs("monitor_dashboard")))
    if _truthy(lc("metrics")):
        actions.append(Node(package=PKG, executable="metrics_logger", name="metrics_logger", output="screen",
                            parameters=cfg("dashboard.yaml") + [common]))
    # ------------------------------------------------------- demo helpers
    if hud:
        hud_params = dict(common)         # layout_file: HUD error figures use the same truth as dashboard/report
        if prof("hud_rate"):
            hud_params["rate"] = float(prof("hud_rate"))
        actions.append(Node(package=PKG, executable="demo_hud", name="demo_hud", output="screen",
                            parameters=[hud_params], **rs("demo_hud")))
    if follow:
        world_name, robot_name = _layout_names(layout_file or os.path.join(share, "config", "world_layout.yaml"))
        offset = ",".join(lc("follow_offset").replace(",", " ").split()) or "-2.5,0.0,1.6"
        fargs = ["--target", robot_name, "--world", world_name, f"--offset={offset}",
                 "--layout", layout_file or os.path.join(share, "config", "world_layout.yaml")]
        if not _truthy(lc("follow_mouse") or "true"):
            fargs.append("--no-mouse")          # the old fixed follow: the GUI pulls a moved camera back
        if not _truthy(lc("follow_walls") or "true"):
            fargs.append("--no-walls")
        actions.append(Node(package=PKG, executable="follow_camera", name="follow_camera", output="screen",
                            arguments=fargs))
    if rviz and not low:
        rviz_cfg = os.path.join(share, "rviz", "plant.rviz")
        actions.append(Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
                            arguments=["-d", rviz_cfg] if os.path.exists(rviz_cfg) else [],
                            parameters=[{"use_sim_time": use_sim_time}]))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument("gui", default_value="true", description="Gazebo GUI"),
        DeclareLaunchArgument("rviz", default_value="false",
                              description="start RViz with rviz/plant.rviz (never with low_memory)"),
        DeclareLaunchArgument("thermal_source", default_value="gazebo",
                              description="gazebo (gz thermal sensor) | synthetic (synthetic_thermal_camera)"),
        DeclareLaunchArgument("mission_autostart", default_value="false",
                              description="mission_manager starts the full mission after autostart_delay"),
        DeclareLaunchArgument("autostart_delay", default_value="25.0", description="s (sim time) before autostart"),
        DeclareLaunchArgument("dashboard_port", default_value="8080", description="digital twin web UI port"),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use the Gazebo /clock"),
        DeclareLaunchArgument("world", default_value="", description="SDF world ('' = installed plant.sdf)"),
        DeclareLaunchArgument("layout_file", default_value="", description="world_layout.yaml ('' = installed)"),
        DeclareLaunchArgument("map_yaml", default_value="", description="nav map YAML ('' = layout default)"),
        DeclareLaunchArgument("module_delay", default_value="6.0", description="s after the sim before modules start"),
        DeclareLaunchArgument("metrics", default_value="true", description="record ~/dusan_metrics/<date>/"),
        DeclareLaunchArgument("teleop_terminal", default_value="none",
                              description="keyboard teleop window: none|xterm|gnome-terminal|konsole|gz"),
        DeclareLaunchArgument("headless_rendering", default_value="false",
                              description="with gui:=false render sensors via EGL"),
        DeclareLaunchArgument("low_memory", default_value="false",
                              description="8 GB laptop: inspection camera 640x480 @ 5 Hz, thermal 2 Hz, "
                                          "no shadows, lean Gazebo GUI, no RViz, HUD 3 Hz, MALLOC_ARENA_MAX=2"),
        DeclareLaunchArgument("respawn", default_value="true",
                              description="restart nodes that died where nothing important is lost"),
        DeclareLaunchArgument("inspection_rate", default_value="",
                              description="inspection camera Hz ('' = 10, low_memory 5)"),
        DeclareLaunchArgument("inspection_width", default_value="",
                              description="inspection camera width px ('' = 800, low_memory 640)"),
        DeclareLaunchArgument("inspection_height", default_value="",
                              description="inspection camera height px ('' = 600, low_memory 480)"),
        DeclareLaunchArgument("depth_rate", default_value="", description="depth camera Hz ('' = 10, low_memory 10)"),
        DeclareLaunchArgument("thermal_rate", default_value="",
                              description="thermal camera Hz ('' = 5, low_memory 2)"),
        DeclareLaunchArgument("shadows", default_value="",
                              description="'' = as in the world (low_memory: false); false = world copy without shadows"),
        DeclareLaunchArgument("gui_config", default_value="",
                              description="Gazebo GUI layout file ('' = gui.config, low_memory gui_lean.config)"),
        DeclareLaunchArgument("hud_rate", default_value="", description="demo_hud Hz ('' = 5, low_memory 3)"),
        DeclareLaunchArgument("demo", default_value="false",
                              description="reference-video demo: mission_autostart + follow camera + HUD"),
        DeclareLaunchArgument("hud", default_value="true",
                              description="demo_hud: robot camera + gas ppm / battery / mission overlay (/demo/hud)"),
        DeclareLaunchArgument("follow_camera", default_value="",
                              description="Gazebo GUI camera follows the robot ('' = same as demo)"),
        DeclareLaunchArgument("follow_offset", default_value="-2.5,0.0,1.6",
                              description="follow camera offset x,y,z in the robot frame, m (behind / above)"),
        DeclareLaunchArgument("follow_mouse", default_value="true",
                              description="keep a view set with the mouse in the 3D view (false = the GUI "
                                          "pulls the camera back to follow_offset, also after a fall; keys "
                                          "c / z and the fall hold stay)"),
        DeclareLaunchArgument("follow_walls", default_value="true",
                              description="pull the default follow view in front of the building walls"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
