"""
Gazebo (gz-sim) virtual plant + robot spawn + ROS bridge.

    ros2 launch plant_simulation sim.launch.py
    ros2 launch plant_simulation sim.launch.py gui:=false headless_rendering:=true
    ros2 launch plant_simulation sim.launch.py x:=0.0 y:=0.0 yaw:=0.0

Starts
  * gz sim (through ros_gz_sim/gz_sim.launch.py, or `gz sim` directly if that
    launch file is missing) with  -r -v <verbosity> <world> --gui-config <share>/config/gui.config
    (server only '-s' when gui:=false)
  * robot_state_publisher  (robot_description = xacro spot.urdf.xacro)
  * ros_gz_sim create      (-name spot -topic robot_description, spawn pose from
                            world_layout.yaml robot_spawn), delayed spawn_delay s
  * ros_gz_bridge parameter_bridge (config/ros_gz_bridge.yaml)
  * plant_simulation clock_relay (clock_relay:=true): the bridge's /clock output
    is remapped to /clock_gz (1 kHz = one sample per 1 ms physics step) and the
    relay republishes it on /clock at 200 Hz of sim time, so that only one
    Python process handles the full rate instead of every use_sim_time node
  * plant_simulation odom_tf + map_server_lite

GZ_SIM_RESOURCE_PATH is extended with <share>/plant_simulation/models,
<share>/plant_simulation/worlds and the parent share directory, so model://<name>
and package://plant_simulation/... URIs both resolve.

Arguments (defaults)
  gui (true)  world ('' = <share>/worlds/plant.sdf)  use_sim_time (true)
  layout_file ('')  map_yaml ('')  x y z yaw ('' = layout robot_spawn)
  spawn_delay (3.0)  spawn_attempts (5)  verbosity (3)  headless_rendering (false)
  enable_thermal (true)  enable_cameras (true)  mesh_uri_prefix ('' = xacro default)
  bridge (true)  clock_relay (true)  start_odom_tf (true)  start_map_server (true)
"""
import os
import shlex

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG = "plant_simulation"
DEFAULT_SPAWN = [61.9, 0.0, -0.32, 3.14159265]


def _truthy(text):
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _read_layout(path):
    """(spawn [x, y, z, yaw], world_name, robot_name) from world_layout.yaml."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        spawn = [float(v) for v in data.get("robot_spawn", DEFAULT_SPAWN)][:4]
        return spawn, str(data.get("world_name", "plant")), str(data.get("robot_name", "spot"))
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return list(DEFAULT_SPAWN), "plant", "spot"


def _spawn_actions(arguments, use_sim_time, attempt, max_attempts, retry_delay):
    """`ros_gz_sim create` + an OnProcessExit handler that retries on failure.

    create makes a single request with a timeout; a slow world load (large
    meshes, first start of the ogre2 render engine) would otherwise leave the
    robot unspawned.  A non-zero exit code triggers another attempt after
    retry_delay seconds, up to max_attempts.
    """
    node = Node(package="ros_gz_sim", executable="create", name="spawn_spot", output="screen",
                arguments=list(arguments), parameters=[{"use_sim_time": use_sim_time}])

    def on_exit(event, context):
        code = getattr(event, "returncode", 0)
        if code == 0:
            return [LogInfo(msg="[sim.launch] robot spawned")]
        if attempt >= max_attempts:
            return [LogInfo(msg=f"[sim.launch] ERROR: spawning failed {attempt} times (exit code {code}) - "
                                "check that gz sim is running and the world is named as in world_layout.yaml")]
        return [LogInfo(msg=f"[sim.launch] spawn attempt {attempt} failed (exit code {code}) - retrying in "
                            f"{retry_delay:.0f} s"),
                TimerAction(period=retry_delay,
                            actions=_spawn_actions(arguments, use_sim_time, attempt + 1, max_attempts, retry_delay))]

    return [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=on_exit)), node]


def _launch_setup(context, *args, **kwargs):
    share = get_package_share_directory(PKG)
    lc = lambda name: LaunchConfiguration(name).perform(context).strip()  # noqa: E731

    gui = _truthy(lc("gui"))
    use_sim_time = _truthy(lc("use_sim_time"))
    layout_file = lc("layout_file") or os.path.join(share, "config", "world_layout.yaml")
    world = lc("world") or os.path.join(share, "worlds", "plant.sdf")
    map_yaml = lc("map_yaml")
    spawn, world_name, robot_name = _read_layout(layout_file)
    pose = []
    for i, key in enumerate(("x", "y", "z", "yaw")):
        v = lc(key)
        pose.append(float(v) if v else spawn[i])
    params_file = os.path.join(share, "config", "mux.yaml")
    bridge_file = os.path.join(share, "config", "ros_gz_bridge.yaml")
    gui_config = os.path.join(share, "config", "gui.config")
    urdf = os.path.join(share, "urdf", "spot.urdf.xacro")

    actions = [
        # model://<name> -> <share>/models/<name>; package://plant_simulation/... -> <share-parent>/plant_simulation/...
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.join(share, "models")),
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.join(share, "worlds")),
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.dirname(share)),
        LogInfo(msg=f"[sim.launch] world {world} | spawn {robot_name} at x={pose[0]:.2f} y={pose[1]:.2f} "
                    f"z={pose[2]:.2f} yaw={pose[3]:.3f} | gui={gui} use_sim_time={use_sim_time}"),
    ]
    # Gazebo's Qt GUI is unreliable on native Wayland: use XWayland unless the user chose a platform
    if gui and os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and "QT_QPA_PLATFORM" not in os.environ:
        actions += [SetEnvironmentVariable("QT_QPA_PLATFORM", "xcb"),
                    LogInfo(msg="[sim.launch] Wayland session detected - QT_QPA_PLATFORM=xcb for the Gazebo GUI")]
    if not os.path.exists(world):
        actions.append(LogInfo(msg=f"[sim.launch] ERROR: world file not found: {world} "
                                   "(run tools/generate_world.py and rebuild)"))

    # ------------------------------------------------------------- gz sim
    gz_args = f"-r -v {lc('verbosity') or '3'} {world}"
    if gui:
        if os.path.exists(gui_config):
            gz_args += f" --gui-config {gui_config}"
        else:
            actions.append(LogInfo(msg=f"[sim.launch] {gui_config} not found - default Gazebo GUI layout"))
    else:
        gz_args = "-s " + gz_args
        if _truthy(lc("headless_rendering")):
            gz_args += " --headless-rendering"
    try:
        gz_launch = os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
    except PackageNotFoundError:
        gz_launch = ""
    if gz_launch and os.path.exists(gz_launch):
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items()))
    else:
        actions += [LogInfo(msg="[sim.launch] ros_gz_sim/gz_sim.launch.py not found - running `gz sim` directly"),
                    ExecuteProcess(cmd=["gz", "sim"] + shlex.split(gz_args), output="screen")]

    # ---------------------------------------------------- robot description
    xacro_cmd = ["xacro ", urdf,
                 " enable_thermal:=", "true" if _truthy(lc("enable_thermal")) else "false",
                 " enable_cameras:=", "true" if _truthy(lc("enable_cameras")) else "false"]
    if lc("mesh_uri_prefix"):
        xacro_cmd += [" mesh_uri_prefix:=", lc("mesh_uri_prefix")]
    robot_description = ParameterValue(Command(xacro_cmd), value_type=str)
    actions.append(Node(
        package="robot_state_publisher", executable="robot_state_publisher", name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": use_sim_time}]))

    # spawn after gz is up and robot_description is latched (retried on failure)
    spawn_args = ["-world", world_name, "-name", robot_name, "-topic", "robot_description",
                  "-x", f"{pose[0]:.4f}", "-y", f"{pose[1]:.4f}", "-z", f"{pose[2]:.4f}", "-Y", f"{pose[3]:.6f}"]
    actions.append(TimerAction(
        period=float(lc("spawn_delay") or "3.0"),
        actions=_spawn_actions(spawn_args, use_sim_time, 1, int(lc("spawn_attempts") or "5"), 3.0)))

    # ---------------------------------------------------------------- bridge
    if _truthy(lc("bridge")):
        if not os.path.exists(bridge_file):
            actions.append(LogInfo(msg=f"[sim.launch] ERROR: bridge config not found: {bridge_file}"))
        relay = _truthy(lc("clock_relay"))
        actions.append(Node(
            package="ros_gz_bridge", executable="parameter_bridge", name="ros_gz_bridge", output="screen",
            parameters=[{"config_file": bridge_file, "use_sim_time": use_sim_time}],
            remappings=[("/clock", "/clock_gz")] if relay else []))
        if relay:
            # the clock source itself must not run on sim time
            actions.append(Node(package=PKG, executable="clock_relay", name="clock_relay", output="screen",
                                parameters=([params_file] if os.path.exists(params_file) else [])
                                + [{"use_sim_time": False}]))

    # ------------------------------------------------------- odom / TF / map
    common = {"use_sim_time": use_sim_time, "layout_file": layout_file}
    node_params = ([params_file] if os.path.exists(params_file) else [])
    if _truthy(lc("start_odom_tf")):
        actions.append(Node(package=PKG, executable="odom_tf", name="odom_tf", output="screen",
                            parameters=node_params + [common]))
    if _truthy(lc("start_map_server")):
        actions.append(Node(package=PKG, executable="map_server_lite", name="map_server_lite", output="screen",
                            parameters=node_params + [dict(common, map_yaml=map_yaml)]))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument("gui", default_value="true", description="start the Gazebo GUI"),
        DeclareLaunchArgument("world", default_value="", description="SDF world ('' = <share>/worlds/plant.sdf)"),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use the Gazebo /clock"),
        DeclareLaunchArgument("layout_file", default_value="", description="world_layout.yaml ('' = installed)"),
        DeclareLaunchArgument("map_yaml", default_value="", description="nav map YAML ('' = layout default)"),
        DeclareLaunchArgument("x", default_value="", description="spawn x ('' = layout robot_spawn)"),
        DeclareLaunchArgument("y", default_value="", description="spawn y ('' = layout robot_spawn)"),
        DeclareLaunchArgument("z", default_value="", description="spawn z ('' = layout robot_spawn)"),
        DeclareLaunchArgument("yaw", default_value="", description="spawn yaw ('' = layout robot_spawn)"),
        DeclareLaunchArgument("spawn_delay", default_value="3.0", description="s before spawning the robot"),
        DeclareLaunchArgument("spawn_attempts", default_value="5", description="ros_gz_sim create retries"),
        DeclareLaunchArgument("verbosity", default_value="3", description="gz sim -v level"),
        DeclareLaunchArgument("headless_rendering", default_value="false",
                              description="with gui:=false: render sensors via EGL (no display)"),
        DeclareLaunchArgument("enable_thermal", default_value="true", description="xacro: Gazebo thermal camera"),
        DeclareLaunchArgument("enable_cameras", default_value="true", description="xacro: depth/inspection cameras"),
        DeclareLaunchArgument("mesh_uri_prefix", default_value="",
                              description="xacro: mesh URI prefix ('' = package://plant_simulation/models/spot/meshes)"),
        DeclareLaunchArgument("bridge", default_value="true", description="start ros_gz_bridge"),
        DeclareLaunchArgument("clock_relay", default_value="true",
                              description="decimate the 1 kHz Gazebo clock to 200 Hz on /clock (false: bridge "
                                          "publishes /clock at the physics rate)"),
        DeclareLaunchArgument("start_odom_tf", default_value="true", description="start odom_tf"),
        DeclareLaunchArgument("start_map_server", default_value="true", description="start map_server_lite"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
