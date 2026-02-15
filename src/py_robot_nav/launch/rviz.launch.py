from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    PathJoinSubstitution,
    LaunchConfiguration,
)
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "sim",
            default_value="false",
            description="Run in Gazebo sim (spawns the robot and uses sim time).",
        ),
    ]

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "--display-config",
            PathJoinSubstitution(
                [
                    FindPackageShare("py_robot_nav"),
                    "rviz",
                    "main.rviz",
                ]
            ),
        ],
        parameters=[{"use_sim_time": LaunchConfiguration("sim")}],
        output="screen",
    )

    return LaunchDescription(
        declared_args
        + [
            rviz_node,
        ]
    )
