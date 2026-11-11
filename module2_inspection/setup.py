import os
from glob import glob
from setuptools import setup

package_name = "module2_inspection"


def tree(src):
    """Install every file below <src> into share/<pkg>/<src>/... keeping the layout
    (Python byte-code caches, e.g. launch/__pycache__ from a test run, are skipped)."""
    out = []
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        files = sorted(f for f in files if not f.endswith((".pyc", ".pyo")))
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
            "gauge_reader = module2_inspection.gauge_reader_node:main",
            "thermal_fusion = module2_inspection.thermal_fusion_node:main",
            "synthetic_thermal_camera = module2_inspection.synthetic_thermal_node:main",
            "patrol_planner = module2_inspection.patrol_planner_node:main",
        ],
    },
)
