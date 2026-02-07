from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.substitutions import (
    PathJoinSubstitution,
    LaunchConfiguration,
)
from launch.conditions import LaunchConfigurationNotEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_args = []

    p2s = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        remappings={
            "cloud_in": "lidar/points",
            "scan": "lidar/laserscan",
        }.items(),
        parameters=[
            {
                "target_frame": "",
                "transform_tolerance": 0.1,
                "min_height": -0.1,
                "max_height": 0.1,
                # [-pi; +pi] because lidar has 360 degrees FOV
                "angle_min": -3.14159,
                "angle_max": 3.14159,
                "angle_increment": 0.0087,  # M_PI/360.0
                "scan_time": 0.1,
                "range_min": 0.2,
                "range_max": 150.0,
                # outputs NaN for no-return rays
                "use_inf": False,
                # auto-detect
                "concurrency_level": 0,
                "use_sim_time": LaunchConfiguration("sim"),
            }
        ],
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("slam_toolbox"),
                    "launch",
                    "online_async_launch.py",
                ]
            )
        ),
        launch_arguments={
            "slam_params_file": PathJoinSubstitution(
                [FindPackageShare("py_robot_nav"), "config", "slam_params.yaml"]
            ),
            "use_sim_time": LaunchConfiguration("sim"),
        }.items(),
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("nav2_bringup"),
                    "launch",
                    "navigation_launch.py",
                ]
            )
        ),
        launch_arguments={
            "params_file": PathJoinSubstitution(
                [FindPackageShare("py_robot_nav"), "config", "nav2_params.yaml"]
            ),
            "use_sim_time": LaunchConfiguration("sim"),
        }.items(),
    )

    # TODO: replace with node level remapping once a custom nav2 launch is written
    relay_node = Node(
        package="topic_tools",
        executable="relay",
        name="nav_out_cmd_vel_relay",
        arguments=["/cmd_vel", LaunchConfiguration("motion_cmd_topic_name")],
        condition=LaunchConfigurationNotEquals("motion_cmd_topic_name", "/cmd_vel"),
    )

    return LaunchDescription(
        declared_args
        + [
            p2s,
            slam_launch,
            nav2_launch,
            relay_node,
        ]
    )
