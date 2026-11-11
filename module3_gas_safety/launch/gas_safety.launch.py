"""
Module 3 (gas & safety) bring-up:

    gas_dispersion_sim   gas leak + airflow simulation   (config/gas_dispersion.yaml)
    source_seeker        olfactory source seeking         (config/seeker.yaml)
    battery_simulator    battery state of charge          (config/safety.yaml)
    comm_simulator       Wi-Fi RSSI coverage              (config/safety.yaml)
    return_to_home       RTH safety supervisor            (config/safety.yaml)

    ros2 launch module3_gas_safety gas_safety.launch.py
    ros2 launch module3_gas_safety gas_safety.launch.py use_sim_time:=false

Arguments: use_sim_time (true), map_yaml (''), layout_file (''), respawn (true).

respawn:=true restarts a node that died (3 s delay, at most 10 times where the
launch version supports a limit) when a restart loses nothing that matters:
gas_dispersion_sim (the plume is spun up again in the constructor),
comm_simulator (stateless), source_seeker (mission_manager latches
/gas/seek/enable, a restarted seeker begins a new run) and return_to_home (it
starts in MONITOR and re-evaluates battery / energy / comm shadow at once; a
manual return is taken over by mission_manager, which drives to the dock itself
when return_to_home does not acknowledge).  battery_simulator is NOT respawned:
it would come back with the start charge and hide a low battery.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from dusan_common.launch_utils import respawn_kwargs

PKG = "module3_gas_safety"
RESPAWN_SAFE = ("gas_dispersion_sim", "source_seeker", "comm_simulator", "return_to_home")


def _params(share: str, yaml_name: str, extra: dict) -> list:
    """Parameter list: YAML file (if installed) followed by the launch overrides."""
    out = []
    path = os.path.join(share, "config", yaml_name)
    if os.path.exists(path):
        out.append(path)
    out.append(extra)
    return out


def _truthy(text) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _respawn_kwargs(delay: float = 3.0, max_retries: int = 10) -> dict:
    """respawn / respawn_delay (every distro) + respawn_max_retries where launch has it (Jazzy and later:
    a parameter of ExecuteProcess' base class ExecuteLocal, see dusan_common.launch_utils)."""
    return respawn_kwargs(ExecuteProcess, True, delay, max_retries)


def _nodes(context, *args, **kwargs):
    share = get_package_share_directory(PKG)
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map_yaml")
    layout_file = LaunchConfiguration("layout_file")
    respawn = _truthy(LaunchConfiguration("respawn").perform(context) or "true")

    common = {
        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
        "layout_file": ParameterValue(layout_file, value_type=str),
    }
    with_map = dict(common, map_yaml=ParameterValue(map_yaml, value_type=str))

    def rs(name):
        return _respawn_kwargs() if respawn and name in RESPAWN_SAFE else {}

    return [
        Node(package=PKG, executable="gas_dispersion_sim", name="gas_dispersion_sim", output="screen",
             parameters=_params(share, "gas_dispersion.yaml", dict(with_map)), **rs("gas_dispersion_sim")),
        Node(package=PKG, executable="source_seeker", name="source_seeker", output="screen",
             parameters=_params(share, "seeker.yaml", dict(with_map)), **rs("source_seeker")),
        Node(package=PKG, executable="battery_simulator", name="battery_simulator", output="screen",
             parameters=_params(share, "safety.yaml", dict(common)), **rs("battery_simulator")),
        Node(package=PKG, executable="comm_simulator", name="comm_simulator", output="screen",
             parameters=_params(share, "safety.yaml", dict(with_map)), **rs("comm_simulator")),
        Node(package=PKG, executable="return_to_home", name="return_to_home", output="screen",
             parameters=_params(share, "safety.yaml", dict(with_map)), **rs("return_to_home")),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use the /clock of Gazebo"),
        DeclareLaunchArgument("map_yaml", default_value="", description="nav map YAML ('' = layout default)"),
        DeclareLaunchArgument("layout_file", default_value="", description="world_layout.yaml ('' = default)"),
        DeclareLaunchArgument("respawn", default_value="true",
                              description="restart gas_dispersion_sim / source_seeker / comm_simulator / "
                                          "return_to_home when they die (battery_simulator never)"),
        OpaqueFunction(function=_nodes),
    ])
