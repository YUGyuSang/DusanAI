"""
Complete digital-twin bring-up: simulation + all three modules + integration.

    ros2 launch plant_simulation bringup.launch.py
    ros2 launch plant_simulation bringup.launch.py rviz:=true thermal_source:=synthetic
    ros2 launch plant_simulation bringup.launch.py mission_autostart:=true gui:=false headless_rendering:=true
    ros2 launch plant_simulation bringup.launch.py demo:=true        # = scripts/run_demo.sh

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
    the Gazebo GUI camera follows the robot from behind / above (gz topic /gui/track)

Demo mode (demo:=true, scripts/run_demo.sh) = the reference video: mission_autostart (corridor ->
factory -> gauges -> gas leak search -> return to the dock), follow camera and HUD on.

Arguments (defaults)
  gui (true)  rviz (false)  thermal_source (gazebo)  mission_autostart (false)
  autostart_delay (25.0)  dashboard_port (8080)  use_sim_time (true)  world ('')
  layout_file ('')  map_yaml ('')  module_delay (6.0)  metrics (true)
  teleop_terminal (none: none|xterm|gnome-terminal|konsole|gz)  headless_rendering (false)
  demo (false)  hud (true)  follow_camera ('' = demo)  follow_offset (-2.5,0.0,1.6)
"""
import os

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "plant_simulation"
MODULE_LAUNCHES = (
    ("module1_locomotion", "locomotion.launch.py"),
    ("module2_inspection", "inspection.launch.py"),
    ("module3_gas_safety", "gas_safety.launch.py"),
)


def _truthy(text):
    return str(text).strip().lower() in ("1", "true", "yes", "on")


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

    def cfg(name):
        p = os.path.join(share, "config", name)
        return [p] if os.path.exists(p) else []

    actions = [LogInfo(msg=f"[bringup] thermal_source={thermal_source} rviz={lc('rviz')} "
                           f"autostart={autostart} dashboard port {lc('dashboard_port')}")]
    if demo:
        actions.append(LogInfo(msg=f"[bringup] DEMO: mission starts {lc('autostart_delay') or '25.0'} s after "
                                   "start (corridor -> factory -> gauges -> gas leak -> dock), Gazebo camera "
                                   "follows the robot, HUD panel 'Robot HUD' / ROS topic /demo/hud"))

    # ------------------------------------------------------------ simulation
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "sim.launch.py")),
        launch_arguments={
            "gui": lc("gui") or "true",
            "world": lc("world"),
            "use_sim_time": sim_time_str,
            "layout_file": layout_file,
            "map_yaml": map_yaml,
            "headless_rendering": lc("headless_rendering") or "false",
            # with the synthetic thermal camera the Gazebo one must not publish the same topic
            "enable_thermal": "false" if thermal_source == "synthetic" else "true",
        }.items()))

    # ------------------------------------------------------- module launches
    common_args = {"use_sim_time": sim_time_str, "map_yaml": map_yaml, "layout_file": layout_file}
    extra_args = {"module1_locomotion": {"teleop_terminal": lc("teleop_terminal") or "none"},
                  "module2_inspection": {"thermal_source": thermal_source}}
    modules = [_module_include(pkg, launch_file, dict(common_args, **extra_args.get(pkg, {})))
               for pkg, launch_file in MODULE_LAUNCHES]
    actions.append(TimerAction(period=float(lc("module_delay") or "6.0"), actions=modules))

    # ------------------------------------------------------- integration nodes
    common = {"use_sim_time": use_sim_time, "layout_file": layout_file}
    actions.append(Node(package=PKG, executable="cmd_vel_mux", name="cmd_vel_mux", output="screen",
                        parameters=cfg("mux.yaml") + [{"use_sim_time": use_sim_time}]))
    actions.append(Node(
        package=PKG, executable="mission_manager", name="mission_manager", output="screen",
        parameters=cfg("mission.yaml") + [dict(common, map_yaml=map_yaml,
                                               autostart=autostart,
                                               autostart_delay=float(lc("autostart_delay") or "25.0"))]))
    actions.append(Node(
        package=PKG, executable="monitor_dashboard", name="monitor_dashboard", output="screen",
        parameters=cfg("dashboard.yaml") + [dict(common, port=int(lc("dashboard_port") or "8080"))]))
    if _truthy(lc("metrics")):
        actions.append(Node(package=PKG, executable="metrics_logger", name="metrics_logger", output="screen",
                            parameters=cfg("dashboard.yaml") + [common]))
    # ------------------------------------------------------- demo helpers
    if hud:
        actions.append(Node(package=PKG, executable="demo_hud", name="demo_hud", output="screen",
                            parameters=[{"use_sim_time": use_sim_time}]))
    if follow:
        world_name, robot_name = _layout_names(layout_file or os.path.join(share, "config", "world_layout.yaml"))
        offset = ",".join(lc("follow_offset").replace(",", " ").split()) or "-2.5,0.0,1.6"
        actions.append(Node(package=PKG, executable="follow_camera", name="follow_camera", output="screen",
                            arguments=["--target", robot_name, "--world", world_name, f"--offset={offset}"]))
    if _truthy(lc("rviz")):
        rviz_cfg = os.path.join(share, "rviz", "plant.rviz")
        actions.append(Node(package="rviz2", executable="rviz2", name="rviz2", output="screen",
                            arguments=["-d", rviz_cfg] if os.path.exists(rviz_cfg) else [],
                            parameters=[{"use_sim_time": use_sim_time}]))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument("gui", default_value="true", description="Gazebo GUI"),
        DeclareLaunchArgument("rviz", default_value="false", description="start RViz with rviz/plant.rviz"),
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
        DeclareLaunchArgument("demo", default_value="false",
                              description="reference-video demo: mission_autostart + follow camera + HUD"),
        DeclareLaunchArgument("hud", default_value="true",
                              description="demo_hud: robot camera + gas ppm / battery / mission overlay (/demo/hud)"),
        DeclareLaunchArgument("follow_camera", default_value="",
                              description="Gazebo GUI camera follows the robot ('' = same as demo)"),
        DeclareLaunchArgument("follow_offset", default_value="-2.5,0.0,1.6",
                              description="follow camera offset x,y,z in the robot frame, m (behind / above)"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
