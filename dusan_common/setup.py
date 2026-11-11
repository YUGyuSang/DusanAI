from setuptools import setup

package_name = "dusan_common"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="DusanAI",
    maintainer_email="YUGyuSang@users.noreply.github.com",
    description="Shared library for the virtual plant quadruped project",
    license="Apache-2.0",
    entry_points={"console_scripts": []},
)
