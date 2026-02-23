from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import math
from py_robot_nav.launch import Topics, cfg_file, launch_file


def generate_launch_description():
    declared_args = []

    sim = LaunchConfiguration("sim")

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
                parameters=[
                    downsampling_params,
                    {"use_sim_time": sim},
                ],
                remappings=[
                    ("input", Topics.CAMERA_DEPTH_POINTS),
                    ("output", f"{Topics.CAMERA_DEPTH_POINTS}/downsampled"),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="pcl_ros",
                plugin="pcl_ros::VoxelGrid",
                name="lidar_voxel_grid",
                parameters=[
                    downsampling_params,
                    {"use_sim_time": sim},
                ],
                remappings=[
                    ("input", Topics.LIDAR_POINTS),
                    ("output", f"{Topics.LIDAR_POINTS}/downsampled"),
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
                        "use_sim_time": sim,
                    }
                ],
                remappings=[
                    ("cloud1", f"{Topics.LIDAR_POINTS}/downsampled"),
                    ("cloud2", f"{Topics.CAMERA_DEPTH_POINTS}/downsampled"),
                    ("combined_cloud", Topics.POINTS),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="pointcloud_to_laserscan",
                plugin="pointcloud_to_laserscan::PointCloudToLaserScanNode",
                name="pointcloud_to_laserscan",
                remappings=[
                    ("cloud_in", Topics.POINTS),
                    ("scan", Topics.SCAN),
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
                        "use_sim_time": sim,
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
            cfg_file("rtabmap_params.yaml"),
            {"use_sim_time": sim},
        ],
        remappings=[
            ("rgb/image", Topics.CAMERA_COLOR_IMAGE),
            # specifically using rgbd image
            # that we got from projecting the depth camera image
            # onto the color camera
            ("depth/image", Topics.CAMERA_RGBD_IMAGE),
            ("rgb/camera_info", Topics.CAMERA_COLOR_INFO),
            ("scan", Topics.SCAN),
            ("map", "/map"),
            ("imu", Topics.IMU),
            ("odom", Topics.ODOM_FILTERED),
        ],
        arguments=["--delete_db_on_start"],
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("nav2")),
        launch_arguments={
            "laserscan_topic": Topics.POINTS,
            "pointcloud_topic": Topics.SCAN,
        }.items(),
    )

    delayed_nav2_launch = RegisterEventHandler(
        OnProcessStart(
            target_action=rtabmap_node,
            on_start=[TimerAction(period=5.0, actions=[nav2_launch])],
        )
    )

    return LaunchDescription(
        declared_args
        + [
            point_cloud_processor,
            rtabmap_node,
            delayed_nav2_launch,
        ]
    )
