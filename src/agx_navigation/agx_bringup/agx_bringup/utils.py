from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

THIS_PACKAGE = "agx_bringup"


def cfg_file(name: str) -> PathJoinSubstitution:
    return _find_file("config", name)


def launch_file(name: str) -> PathJoinSubstitution:
    return _find_file("launch", f"{name}.launch.py")


def rviz_file(name: str) -> PathJoinSubstitution:
    return _find_file("rviz", f"{name}.rviz")


def _find_file(dir: str, name: str) -> PathJoinSubstitution:
    return PathJoinSubstitution(
        [
            FindPackageShare(THIS_PACKAGE),
            dir,
            name,
        ]
    )
