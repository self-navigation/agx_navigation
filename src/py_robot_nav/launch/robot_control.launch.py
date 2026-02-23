from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from py_robot_nav.launch import Topics, cfg_file, launch_file


def generate_launch_description():
    declared_args = []

    sim = LaunchConfiguration("sim")

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
            "sim": sim,
            "camera_depth_points_topic": Topics.CAMERA_DEPTH_POINTS,
        }.items(),
    )

    sim_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("sim_control")),
        condition=IfCondition(sim),
    )

    life_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("life_control")),
        condition=UnlessCondition(sim),
    )

    imu_filter = Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        name="imu_filter",
        output="screen",
        parameters=[
            cfg_file("imu_filter_params.yaml"),
            {"use_sim_time": sim},
        ],
        remappings=[
            ("imu/mag", Topics.MAGNETIC_FIELD),
            ("imu/data_raw", Topics.IMU),
            ("imu/data", Topics.IMU_FILTERED),
        ],
    )

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            cfg_file("ekf_params.yaml"),
            {"use_sim_time": sim},
        ],
        remappings=[
            ("odom", Topics.ODOM),
            ("imu", Topics.IMU),
            ("odometry/filtered", Topics.ODOM_FILTERED),
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
