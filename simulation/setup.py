import os
from glob import glob
from setuptools import setup

package_name = "plant_simulation"


def tree(src):
    """Install every file below <src> into share/<pkg>/<src>/... keeping the layout."""
    out = []
    for root, _dirs, files in os.walk(src):
        if files:
            out.append((os.path.join("share", package_name, root), [os.path.join(root, f) for f in files]))
    return out


data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]
for d in ("launch", "config", "worlds", "models", "urdf", "maps", "rviz", "web"):
    if os.path.isdir(d):
        data_files += tree(d)

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="DusanAI",
    maintainer_email="YUGyuSang@users.noreply.github.com",
    description=open("package.xml", encoding="utf-8").read().split("<description>")[1].split("</description>")[0],
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "map_server_lite = plant_simulation.map_server_node:main",
            "odom_tf = plant_simulation.odom_tf_node:main",
            "cmd_vel_mux = plant_simulation.cmd_vel_mux_node:main",
            "mission_manager = plant_simulation.mission_manager_node:main",
            "monitor_dashboard = plant_simulation.monitor_dashboard_node:main",
            "metrics_logger = plant_simulation.metrics_logger_node:main",
            "clock_relay = plant_simulation.clock_relay_node:main",
            "demo_hud = plant_simulation.demo_hud_node:main",
            "follow_camera = plant_simulation.follow_camera:main",
        ],
    },
)
