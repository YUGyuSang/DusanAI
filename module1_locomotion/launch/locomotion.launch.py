"""
Module 1 bring-up: terrain mapper + locomotion controller (+ optional teleop).

    ros2 launch module1_locomotion locomotion.launch.py
    ros2 launch module1_locomotion locomotion.launch.py teleop_terminal:=xterm
    ros2 launch module1_locomotion locomotion.launch.py teleop_terminal:=gz      # Gazebo keys only
    ros2 launch module1_locomotion locomotion.launch.py use_sim_time:=false start_controller:=false

Arguments
  use_sim_time      true            pass simulated time to every node
  start_terrain     true            run terrain_mapper
  start_controller  true            run locomotion_controller
  terrain_config    <share>/config/terrain.yaml
  controller_config <share>/config/locomotion.yaml
  teleop_config     <share>/config/teleop.yaml
  layout_file       ''              world_layout.yaml for the friction prior of terrain_mapper
                                    ('' = installed plant_simulation config)
  teleop_terminal   none            none | xterm | gnome-terminal | konsole | gz |
                                    any custom prefix containing a space (e.g. "tilix -e")
  teleop_mode       hold            hold | latch
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PKG = "module1_locomotion"

# terminal emulator -> launch prefix that runs the teleop in a new window
_TERMINAL_PREFIX = {
    "xterm": "xterm -e",
    "gnome-terminal": "gnome-terminal --wait --",
    "konsole": "konsole -e",
}


def _truthy(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _teleop(context, *args, **kwargs):
    """Teleop node, created only when teleop_terminal != none."""
    term = LaunchConfiguration("teleop_terminal").perform(context).strip()
    key = term.lower()
    if key in ("", "none", "false", "off", "no"):
        return []
    if key in _TERMINAL_PREFIX:
        prefix = _TERMINAL_PREFIX[key]
    elif key in ("gz", "gazebo", "background"):
        prefix = None                      # no TTY -> node runs in Gazebo-keys-only mode
    elif " " in term:
        prefix = term                      # custom prefix, e.g. "tilix -e"
    else:
        return [LogInfo(msg=f"[locomotion.launch] unknown teleop_terminal '{term}' - teleop not started "
                            f"(use none|xterm|gnome-terminal|konsole|gz)")]
    use_sim_time = _truthy(LaunchConfiguration("use_sim_time").perform(context))
    mode = LaunchConfiguration("teleop_mode").perform(context).strip() or "hold"
    params = [LaunchConfiguration("teleop_config").perform(context),
              {"use_sim_time": use_sim_time, "mode": mode}]
    node_kwargs = dict(package=PKG, executable="keyboard_teleop", name="keyboard_teleop",
                       output="screen", parameters=params)
    if prefix:
        node_kwargs["prefix"] = prefix
    return [LogInfo(msg=f"[locomotion.launch] keyboard_teleop via '{prefix or 'Gazebo keys only'}'"),
            Node(**node_kwargs)]


def generate_launch_description():
    share = FindPackageShare(PKG)
    use_sim_time = LaunchConfiguration("use_sim_time")

    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true",
                              description="Use the /clock topic published by Gazebo"),
        DeclareLaunchArgument("start_terrain", default_value="true",
                              description="Start the terrain_mapper node"),
        DeclareLaunchArgument("start_controller", default_value="true",
                              description="Start the locomotion_controller node"),
        DeclareLaunchArgument("terrain_config",
                              default_value=PathJoinSubstitution([share, "config", "terrain.yaml"]),
                              description="Parameter file of terrain_mapper"),
        DeclareLaunchArgument("controller_config",
                              default_value=PathJoinSubstitution([share, "config", "locomotion.yaml"]),
                              description="Parameter file of locomotion_controller"),
        DeclareLaunchArgument("teleop_config",
                              default_value=PathJoinSubstitution([share, "config", "teleop.yaml"]),
                              description="Parameter file of keyboard_teleop"),
        DeclareLaunchArgument("layout_file", default_value="",
                              description="world_layout.yaml of the friction prior ('' = installed)"),
        DeclareLaunchArgument("teleop_terminal", default_value="none",
                              description="none | xterm | gnome-terminal | konsole | gz | '<custom prefix>'"),
        DeclareLaunchArgument("teleop_mode", default_value="hold",
                              description="hold | latch"),
    ]

    terrain = Node(
        package=PKG, executable="terrain_mapper", name="terrain_mapper", output="screen",
        parameters=[LaunchConfiguration("terrain_config"),
                    {"use_sim_time": use_sim_time, "layout_file": LaunchConfiguration("layout_file")}],
        condition=IfCondition(LaunchConfiguration("start_terrain")),
    )
    controller = Node(
        package=PKG, executable="locomotion_controller", name="locomotion_controller", output="screen",
        parameters=[LaunchConfiguration("controller_config"), {"use_sim_time": use_sim_time}],
        condition=IfCondition(LaunchConfiguration("start_controller")),
    )
    return LaunchDescription(args + [terrain, controller, OpaqueFunction(function=_teleop)])
