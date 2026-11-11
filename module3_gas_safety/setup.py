import os
from glob import glob
from setuptools import setup

package_name = "module3_gas_safety"


def tree(src):
    """Install every file below <src> into share/<pkg>/<src>/... keeping the layout
    (Python byte-code caches excluded)."""
    out = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        files = [f for f in files if not f.endswith((".pyc", ".pyo"))]
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
            "gas_dispersion_sim = module3_gas_safety.gas_dispersion_node:main",
            "source_seeker = module3_gas_safety.source_seeker_node:main",
            "battery_simulator = module3_gas_safety.battery_node:main",
            "comm_simulator = module3_gas_safety.comm_node:main",
            "return_to_home = module3_gas_safety.return_to_home_node:main",
        ],
    },
)
