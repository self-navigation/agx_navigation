from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def cfg_file(name: str):
    return PathJoinSubstitution(
        [
            FindPackageShare("py_robot_nav"),
            "config",
            name,
        ]
    )


def launch_file(name: str):
    return PathJoinSubstitution(
        [
            FindPackageShare("py_robot_nav"),
            "launch",
            f"{name}.launch.py",
        ]
    )
