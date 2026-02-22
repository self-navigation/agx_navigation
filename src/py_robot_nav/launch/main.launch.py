from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import (
    PathJoinSubstitution,
    LaunchConfiguration,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "sim",
            default_value="false",
            description="Run in Gazebo sim (spawns the robot and uses sim time).",
        ),
        DeclareLaunchArgument(
            "camera_color_image_topic",
            default_value="/d435_camera/color/image_raw",
            description="Unified topic name for images received from the color camera.",
        ),
        DeclareLaunchArgument(
            "camera_depth_image_topic",
            default_value="/d435_camera/depth/image_raw",
            description="Unified topic name for images received from the depth camera.",
        ),
        DeclareLaunchArgument(
            "camera_depth_points_topic",
            default_value="/d435_camera/depth/points",
            description="Unified topic name for points received from the depth camera.",
        ),
        DeclareLaunchArgument(
            "motion_cmd_topic_name",
            default_value="/cmd_vel",
            description="Motion controls topic name.",
        ),
        DeclareLaunchArgument(
            "odom_topic_name",
            default_value="/odom",
            description="Odometry topic name.",
        ),
    ]

    scout_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("py_robot_nav"),
                    "launch",
                    "robot_control.launch.py",
                ]
            )
        ),
        launch_arguments={}.items(),
    )

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("py_robot_nav"),
                    "launch",
                    "gz_sim.launch.py",
                ]
            )
        ),
        launch_arguments={}.items(),
        condition=IfCondition(LaunchConfiguration("sim")),
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("py_robot_nav"),
                    "launch",
                    "slam.launch.py",
                ]
            )
        ),
        launch_arguments={
            "robot_motion_cmd_topic": LaunchConfiguration("motion_cmd_topic_name"),
        }.items(),
    )

    return LaunchDescription(
        declared_args
        + [
            gz_sim_launch,
            scout_launch,
            TimerAction(period=10.0, actions=[slam_launch]),
        ]
    )
