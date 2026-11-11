"""
Module 2 (inspection) bring-up:

    gauge_reader              analog gauge reading        (config/gauge_reader.yaml)
    thermal_fusion            thermal/RGB fusion, alerts  (config/thermal.yaml)
    patrol_planner            viewpoint patrol route      (config/patrol.yaml)
    synthetic_thermal_camera  only with thermal_source:=synthetic (config/thermal.yaml)

    ros2 launch module2_inspection inspection.launch.py
    ros2 launch module2_inspection inspection.launch.py thermal_source:=synthetic use_sim_time:=true

Arguments: use_sim_time (true), thermal_source (gazebo | synthetic), map_yaml (''),
layout_file ('').

thermal_source:=synthetic only adds synthetic_thermal_camera; it cannot switch the Gazebo
thermal camera off.  Run the simulation without it (``plant_simulation sim.launch.py
enable_thermal:=false``) or use ``plant_simulation bringup.launch.py thermal_source:=synthetic``,
which does both - otherwise two sources publish on /thermal_camera/image.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG = "module2_inspection"


def _params(share: str, yaml_name: str, extra: dict) -> list:
    """Parameter list: YAML file (if installed) followed by launch overrides."""
    out = []
    path = os.path.join(share, "config", yaml_name)
    if os.path.exists(path):
        out.append(path)
    out.append(extra)
    return out


def generate_launch_description():
    share = get_package_share_directory(PKG)
    use_sim_time = LaunchConfiguration("use_sim_time")
    thermal_source = LaunchConfiguration("thermal_source")
    map_yaml = LaunchConfiguration("map_yaml")
    layout_file = LaunchConfiguration("layout_file")

    common = {
        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
        "layout_file": ParameterValue(layout_file, value_type=str),
    }
    synthetic = IfCondition(PythonExpression(["'", thermal_source, "' == 'synthetic'"]))

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use the /clock of Gazebo"),
        DeclareLaunchArgument("thermal_source", default_value="gazebo",
                              description="gazebo: Gazebo thermal sensor | synthetic: synthetic_thermal_camera node "
                                          "(start the simulation with enable_thermal:=false)"),
        DeclareLaunchArgument("map_yaml", default_value="", description="nav map YAML ('' = layout default)"),
        DeclareLaunchArgument("layout_file", default_value="", description="world_layout.yaml ('' = default)"),
        LogInfo(msg=["module2_inspection: thermal_source=", thermal_source]),

        Node(package=PKG, executable="gauge_reader", name="gauge_reader", output="screen",
             parameters=_params(share, "gauge_reader.yaml", dict(common))),
        Node(package=PKG, executable="thermal_fusion", name="thermal_fusion", output="screen",
             parameters=_params(share, "thermal.yaml", dict(common))),
        Node(package=PKG, executable="patrol_planner", name="patrol_planner", output="screen",
             parameters=_params(share, "patrol.yaml",
                                {**common, "map_yaml": ParameterValue(map_yaml, value_type=str)})),
        Node(package=PKG, executable="synthetic_thermal_camera", name="synthetic_thermal_camera", output="screen",
             parameters=_params(share, "thermal.yaml", dict(common)), condition=synthetic),
    ])
