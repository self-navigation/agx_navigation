from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from agx_bringup import rviz_file


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "sim",
            default_value="false",
            description="Run in Gazebo sim (spawns the robot and uses sim time).",
        ),
    ]

    sim = LaunchConfiguration("sim")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["--display-config", rviz_file("main")],
        parameters=[{"use_sim_time": sim}],
        output="screen",
    )

    return LaunchDescription(
        declared_args
        + [
            rviz_node,
        ]
    )
