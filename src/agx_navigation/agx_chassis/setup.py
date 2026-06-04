from setuptools import setup, find_packages

package_name = "agx_chassis"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AGX Team",
    maintainer_email="claudeai@gmatiukhin.site",
    description="Per-wheel drive kinematics and wheel odometry for the AGX robot.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "twist_to_wheels = agx_chassis.twist_to_wheels:main",
            "wheel_odometry = agx_chassis.wheel_odometry:main",
        ],
    },
)
