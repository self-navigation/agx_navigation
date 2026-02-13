from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
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
            "sim_reduction": "4",
            "use_sim_time": LaunchConfiguration("sim"),
        }.items(),
    )

    sim_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("py_robot_nav"), "launch", "sim_control.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("sim")),
    )

    life_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("py_robot_nav"), "launch", "life_control.launch.py"]
            )
        ),
        launch_arguments={}.items(),
        condition=UnlessCondition(LaunchConfiguration("sim")),
    )

    imu_filter = Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        name="imu_filter",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("py_robot_nav"), "config", "imu_filter_params.yaml"]
            ),
            {"use_sim_time": LaunchConfiguration("sim")},
        ],
        remappings=[
            ("imu/mag", "/magnetic_field"),
            ("imu/data_raw", "/imu"),
            ("imu/data", "/imu/filtered"),
        ],
    )

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("py_robot_nav"), "config", "ekf_params.yaml"]
            ),
            {"use_sim_time": LaunchConfiguration("sim")},
        ],
        remappings=[
            ("odom", LaunchConfiguration("odom_topic_name")),
            ("imu", "/imu"),
            (
                "odometry/filtered",
                PathJoinSubstitution(
                    [LaunchConfiguration("odom_topic_name"), "filtered"]
                ),
            ),
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            scout_launch,
            sim_control_launch,
            life_control_launch,
            # imu_filter,
            ekf_node,
        ]
    )
