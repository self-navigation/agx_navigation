from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "odom_topic_name",
            default_value="/odom",
            description="Odometry topic name.",
        ),
        DeclareLaunchArgument(
            "motion_cmd_topic_name",
            default_value="/cmd_vel",
            description="Motion controls topic name.",
        ),
    ]

    scout_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("scout_description"),
                    "launch",
                    "scout_mini.launch.py",
                ]
            )
        ),
        launch_arguments={
            "namespace": "",
            "use_sim_time": LaunchConfiguration("sim"),
        }.items(),
    )

    sim_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("py_robot_nav"),
                    "launch",
                    "sim_control.launch.py",
                ]
            )
        ),
        condition=IfCondition(LaunchConfiguration("sim")),
    )

    life_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("py_robot_nav"),
                    "launch",
                    "life_control.launch.py",
                ]
            )
        ),
        launch_arguments={}.items(),
        condition=UnlessCondition(LaunchConfiguration("sim")),
    )

    return LaunchDescription(
        declared_args
        + [
            scout_launch,
            sim_control_launch,
            life_control_launch,
        ]
    )
