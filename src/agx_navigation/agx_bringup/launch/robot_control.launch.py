from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
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
from agx_bringup import Topics, cfg_file, launch_file


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
            "controller_file": "wheel_velocity_controller.yaml",
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

    declared_args.append(DeclareLaunchArgument(
        "ekf_wheel_yaw", default_value="true",
        description="fuse wheel odometry's yaw rate (odom0_config index 11) in "
                    "the EKF. It is chi-biased (see ekf_params.yaml); false lets "
                    "the gyro own yaw. Default true = historical behaviour (#34)."))

    def ekf(wheel_yaw: bool):
        overrides = {"use_sim_time": sim}
        if not wheel_yaw:
            overrides["odom0_config"] = [True, True, False,
                                         False, False, False,
                                         True, True, False,
                                         False, False, False,
                                         False, False, False]
        cond = IfCondition if wheel_yaw else UnlessCondition
        return Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_filter_node",
            output="screen",
            parameters=[cfg_file("ekf_params.yaml"), overrides],
            remappings=[
                ("odom", Topics.ODOM),
                ("imu", Topics.IMU),
                ("odometry/filtered", Topics.ODOM_FILTERED),
            ],
            condition=cond(LaunchConfiguration("ekf_wheel_yaw")),
        )

    return LaunchDescription(
        declared_args
        + [
            scout_launch,
            sim_control_launch,
            life_control_launch,
            # imu_filter,
            ekf(True),
            ekf(False),
        ]
    )
