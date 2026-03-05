from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from agx_bringup import launch_file


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "sim",
            default_value="false",
            description="Run in Gazebo sim (spawns the robot and uses sim time).",
        ),
    ]

    sim = LaunchConfiguration("sim")

    scout_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("robot_control")),
        launch_arguments={}.items(),
    )

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("gz_sim")),
        launch_arguments={}.items(),
        condition=IfCondition(sim),
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("slam")),
    )

    delayed_slam_launch = TimerAction(period=10.0, actions=[slam_launch])

    return LaunchDescription(
        declared_args
        + [
            gz_sim_launch,
            scout_launch,
            delayed_slam_launch,
        ]
    )
