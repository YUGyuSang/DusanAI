"""
Digital-twin dashboard only (attach to an already running simulation).

    ros2 launch plant_simulation dashboard.launch.py
    ros2 launch plant_simulation dashboard.launch.py port:=8090 metrics:=true

Arguments: port (8080), use_sim_time (true), layout_file (''), metrics (false).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG = "plant_simulation"


def generate_launch_description():
    share = get_package_share_directory(PKG)
    cfg = os.path.join(share, "config", "dashboard.yaml")
    files = [cfg] if os.path.exists(cfg) else []
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    layout_file = ParameterValue(LaunchConfiguration("layout_file"), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="8080", description="HTTP port of the web UI"),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use the Gazebo /clock"),
        DeclareLaunchArgument("layout_file", default_value="", description="world_layout.yaml ('' = installed)"),
        DeclareLaunchArgument("metrics", default_value="false", description="also run metrics_logger"),
        Node(package=PKG, executable="monitor_dashboard", name="monitor_dashboard", output="screen",
             parameters=files + [{"use_sim_time": use_sim_time, "layout_file": layout_file,
                                  "port": ParameterValue(LaunchConfiguration("port"), value_type=int)}]),
        Node(package=PKG, executable="metrics_logger", name="metrics_logger", output="screen",
             parameters=files + [{"use_sim_time": use_sim_time, "layout_file": layout_file}],
             condition=IfCondition(LaunchConfiguration("metrics"))),
    ])
