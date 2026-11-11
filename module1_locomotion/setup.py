import os
from glob import glob
from setuptools import setup

package_name = "module1_locomotion"


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
            "locomotion_controller = module1_locomotion.locomotion_controller_node:main",
            "terrain_mapper = module1_locomotion.terrain_mapping_node:main",
            "keyboard_teleop = module1_locomotion.keyboard_teleop:main",
        ],
    },
)
