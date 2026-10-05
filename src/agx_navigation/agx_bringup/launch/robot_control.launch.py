from launch import LaunchDescription
from launch.actions import (
    OpaqueFunction,
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

    declared_args.append(DeclareLaunchArgument(
        "lidar_odom", default_value="false",
        description="fuse rf2o laser odometry (pose, differential) in the EKF "
                    "and drop the wheels' yaw rate AND their vy. Wheel odometry "
                    "reports vy=0 by construction, which tells the filter a "
                    "skidding robot is not moving sideways; under amcl the "
                    "corrector then saw its cross-track error at ~0.4x gain "
                    "and ~2.5 s lag (#34). Needs /scan (localization:=amcl)."))

    def ekf_nodes(context):
        wheel_yaw = LaunchConfiguration("ekf_wheel_yaw").perform(context) == "true"
        lidar = LaunchConfiguration("lidar_odom").perform(context) == "true"
        overrides = {"use_sim_time": sim}
        if not wheel_yaw or lidar:
            overrides["odom0_config"] = [True, True, False,
                                         False, False, False,
                                         True, not lidar, False,
                                         False, False, False,
                                         False, False, False]
        nodes = []
        if lidar:
            overrides.update({
                "odom1": "odom_lidar",
                "odom1_config": [True, True, False,
                                 False, False, True,
                                 False, False, False,
                                 False, False, False,
                                 False, False, False],
                "odom1_differential": True,
                "odom1_relative": False,
                "odom1_queue_size": 10,
            })
            nodes += [
                Node(package="rf2o_laser_odometry",
                     executable="rf2o_laser_odometry_node",
                     name="rf2o_laser_odometry", output="screen",
                     parameters=[{"laser_scan_topic": Topics.SCAN,
                                  "odom_topic": "odom_rf2o",
                                  "publish_tf": False,
                                  "base_frame_id": "base_link",
                                  "odom_frame_id": "odom",
                                  "init_pose_from_topic": "",
                                  "freq": 10.0,
                                  "use_sim_time": sim}]),
                Node(package="agx_bringup", executable="lidar_odom_relay",
                     name="lidar_odom_relay", output="screen",
                     parameters=[{"use_sim_time": sim}],
                     remappings=[("odom_in", "odom_rf2o"),
                                 ("odom_out", "odom_lidar")]),
            ]
        nodes.append(Node(
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
        ))
        return nodes

    return LaunchDescription(
        declared_args
        + [
            scout_launch,
            sim_control_launch,
            life_control_launch,
            # imu_filter,
            OpaqueFunction(function=ekf_nodes),
        ]
    )
