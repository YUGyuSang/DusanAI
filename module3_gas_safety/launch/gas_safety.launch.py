"""
Module 3 (gas & safety) bring-up:

    gas_dispersion_sim   gas leak + airflow simulation   (config/gas_dispersion.yaml)
    source_seeker        olfactory source seeking         (config/seeker.yaml)
    battery_simulator    battery state of charge          (config/safety.yaml)
    comm_simulator       Wi-Fi RSSI coverage              (config/safety.yaml)
    return_to_home       RTH safety supervisor            (config/safety.yaml)

    ros2 launch module3_gas_safety gas_safety.launch.py
    ros2 launch module3_gas_safety gas_safety.launch.py use_sim_time:=false

Arguments: use_sim_time (true), map_yaml (''), layout_file ('').
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG = "module3_gas_safety"


def _params(share: str, yaml_name: str, extra: dict) -> list:
    """Parameter list: YAML file (if installed) followed by the launch overrides."""
    out = []
    path = os.path.join(share, "config", yaml_name)
    if os.path.exists(path):
        out.append(path)
    out.append(extra)
    return out


def generate_launch_description():
    share = get_package_share_directory(PKG)
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map_yaml")
    layout_file = LaunchConfiguration("layout_file")

    common = {
        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
        "layout_file": ParameterValue(layout_file, value_type=str),
    }
    with_map = dict(common, map_yaml=ParameterValue(map_yaml, value_type=str))

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use the /clock of Gazebo"),
        DeclareLaunchArgument("map_yaml", default_value="", description="nav map YAML ('' = layout default)"),
        DeclareLaunchArgument("layout_file", default_value="", description="world_layout.yaml ('' = default)"),

        Node(package=PKG, executable="gas_dispersion_sim", name="gas_dispersion_sim", output="screen",
             parameters=_params(share, "gas_dispersion.yaml", dict(with_map))),
        Node(package=PKG, executable="source_seeker", name="source_seeker", output="screen",
             parameters=_params(share, "seeker.yaml", dict(with_map))),
        Node(package=PKG, executable="battery_simulator", name="battery_simulator", output="screen",
             parameters=_params(share, "safety.yaml", dict(common))),
        Node(package=PKG, executable="comm_simulator", name="comm_simulator", output="screen",
             parameters=_params(share, "safety.yaml", dict(with_map))),
        Node(package=PKG, executable="return_to_home", name="return_to_home", output="screen",
             parameters=_params(share, "safety.yaml", dict(with_map))),
    ])
