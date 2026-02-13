from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.substitutions import (
    PathJoinSubstitution,
    LaunchConfiguration,
    NotEqualsSubstitution,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
import math


def config_file(name: str):
    return PathJoinSubstitution(
        [
            FindPackageShare("py_robot_nav"),
            "config",
            name,
        ]
    )


def generate_launch_description():
    declared_args = []

    downsampling_params = {
        "leaf_size": 0.05,
        "filter_field_name": "z",
        # show floor
        "filter_limit_min": -0.5,
        "filter_limit_max": 5.0,
    }

    point_cloud_processor = ComposableNodeContainer(
        name="point_cloud_processing_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="pcl_ros",
                plugin="pcl_ros::VoxelGrid",
                name="camera_voxel_grid",
                parameters=[downsampling_params],
                remappings=[
                    ("input", "/camera/depth/image_raw/points"),
                    ("output", "/camera/depth/image_raw/points/downsampled"),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="pcl_ros",
                plugin="pcl_ros::VoxelGrid",
                name="lidar_voxel_grid",
                parameters=[downsampling_params],
                remappings=[
                    ("input", "/lidar/points"),
                    ("output", "/lidar/points/downsampled"),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="rtabmap_util",
                plugin="rtabmap_util::PointCloudAggregator",
                name="point_cloud_aggregator",
                parameters=[
                    {
                        "fixed_frame_id": "base_link",
                        "approx_sync": True,
                        # "approx_sync_max_interval": 0.5,
                        "count": 2,
                        "xyz_output": True,
                        "use_sim_time": LaunchConfiguration("sim"),
                    }
                ],
                remappings=[
                    ("cloud1", "/lidar/points/downsampled"),
                    ("cloud2", "/camera/depth/image_raw/points/downsampled"),
                    ("combined_cloud", "/combined_cloud"),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="pointcloud_to_laserscan",
                plugin="pointcloud_to_laserscan::PointCloudToLaserScanNode",
                name="pointcloud_to_laserscan",
                remappings=[
                    ("cloud_in", "/combined_cloud"),
                    ("scan", "/combined_cloud/laserscan"),
                ],
                parameters=[
                    {
                        "target_frame": "",
                        "transform_tolerance": 0.1,
                        "min_height": 0.1,
                        "max_height": 1.0,
                        "angle_min": -3.14159,
                        "angle_max": 3.14159,
                        "scan_time": 0.1,
                        "scan_delay": 0.1,
                        "range_min": 0.2,
                        "range_max": 150.0,
                        "use_inf": True,
                        # "inf_epsilon": 0.5,
                        "angle_increment": math.pi / 360 / 3,
                        "concurrency_level": 0,
                        "use_sim_time": LaunchConfiguration("sim"),
                    }
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
    )

    rtabmap_node = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[
            config_file("rtabmap_params.yaml"),
            {"use_sim_time": LaunchConfiguration("sim")},
        ],
        remappings=[
            ("scan", "/combined_cloud/laserscan"),
            ("scan_cloud", "/combined_cloud"),
            ("map", "/map"),
            ("imu", "/imu"),
            (
                "odom",
                PathJoinSubstitution(
                    [LaunchConfiguration("odom_topic_name"), "filtered"]
                ),
            ),
        ],
        arguments=["--delete_db_on_start"],
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
            "params_file": config_file("nav2_params.yaml"),
            "use_sim_time": LaunchConfiguration("sim"),
            "use_composition": True,
            "use_intra_process_comms": True,
        }.items(),
    )

    # TODO: replace with node level remapping once a custom nav2 launch is written
    relay_node = Node(
        package="topic_tools",
        executable="relay",
        name="nav_out_cmd_vel_relay",
        arguments=["/cmd_vel", LaunchConfiguration("motion_cmd_topic_name")],
        condition=IfCondition(
            NotEqualsSubstitution(
                LaunchConfiguration("motion_cmd_topic_name"), "/cmd_vel"
            )
        ),
    )

    return LaunchDescription(
        declared_args
        + [
            point_cloud_processor,
            rtabmap_node,
            # nav2_launch,
            # relay_node,
        ]
    )
