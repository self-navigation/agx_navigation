import os
from glob import glob

from setuptools import find_packages, setup

package_name = "agx_baselines"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gmatiukhin",
    maintainer_email="contact@gmatiukhin.site",
    description="Comparison baselines; not part of the agx navigation stack",
    license="MIT",
    entry_points={"console_scripts": []},
)
