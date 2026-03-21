from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import (
    Node,
    LoadComposableNodes,
    ComposableNodeContainer,
)
from launch.conditions import UnlessCondition
from launch_ros.descriptions import ComposableNode
from agx_bringup import Topics, cfg_file, launch_file
import math


def generate_point_cloud_processor(context):
    sim = LaunchConfiguration("sim")
    is_sim = sim.perform(context).lower() in ["true", "1", "yes"]

    downsampling_params = {
        "leaf_size": 0.05,
        "filter_field_name": "z",
        # show floor
        "filter_limit_min": -0.5,
        "filter_limit_max": 5.0,
    }

    if is_sim:
        lidar_deskewed = Topics.LIDAR_POINTS
    else:
        lidar_deskewed = f"{Topics.LIDAR_POINTS}/deskewed"

    return [
        ComposableNodeContainer(
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
                        ("output", Topics.CAMERA_DEPTH_DOWNSAMPLED),
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
                        ("input", lidar_deskewed),
                        ("output", Topics.LIDAR_DOWNSAMPLED),
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
                            "count": 2,
                            "xyz_output": True,
                            "use_sim_time": sim,
                        }
                    ],
                    remappings=[
                        ("cloud1", Topics.LIDAR_DOWNSAMPLED),
                        ("cloud2", Topics.CAMERA_DEPTH_DOWNSAMPLED),
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
                            "max_height": 0.7,
                            "angle_min": -3.14159,
                            "angle_max": 3.14159,
                            "scan_time": 0.1,
                            "scan_delay": 0.1,
                            "range_min": 0.2,
                            "range_max": 150.0,
                            "use_inf": True,
                            "angle_increment": (2 * math.pi) / 360 / 4,
                            "concurrency_level": 0,
                            "use_sim_time": sim,
                        }
                    ],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                # ComposableNode(
                #     package="pointcloud_utils",
                #     plugin="pointcloud_utils::LaserScanInterpolator",
                #     name="laserscan_interpolator",
                #     parameters=[
                #         {
                #             "input_topic": f"{Topics.SCAN}/raw",
                #             "output_topic": Topics.SCAN,
                #             "target_angle_increment": (2 * math.pi) / 360 / 4,
                #             "distance_threshold": 0.5,
                #             "use_sim_time": sim,
                #         }
                #     ],
                #     extra_arguments=[{"use_intra_process_comms": True}],
                # ),
            ],
        ),
        # Deskewing is done for the real hardware only.
        # In sim all points are geometrically perfect so deskewing is a no-op
        # and Gazebo doesn't return timestamps for them
        LoadComposableNodes(
            target_container="point_cloud_processing_container",
            condition=UnlessCondition(sim),
            composable_node_descriptions=[
                ComposableNode(
                    package="rtabmap_util",
                    plugin="rtabmap_util::LidarDeskewing",
                    name="lidar_deskew",
                    parameters=[
                        {
                            "fixed_frame_id": "odom",
                            "slerp": True,
                            "wait_for_transform": 0.2,
                            "use_sim_time": sim,
                        }
                    ],
                    remappings=[
                        ("input_cloud", Topics.LIDAR_POINTS),
                        (
                            # This node changes its output topic based on input topic like /<input_cloud>/deskewed.
                            # Renaming this to a slightly different name
                            f"{Topics.LIDAR_POINTS}/deskewed",
                            f"{Topics.LIDAR_POINTS}/deskewed/raw",
                        ),
                    ],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                ComposableNode(
                    package="pointcloud_utils",
                    plugin="pointcloud_utils::PointCloudFieldStripper",
                    name="lidar_field_stripper",
                    parameters=[{"use_sim_time": sim}],
                    remappings=[
                        ("input", f"{Topics.LIDAR_POINTS}/deskewed/raw"),
                        ("output", lidar_deskewed),
                    ],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
            ],
        ),
    ]


def generate_launch_description():
    declared_args = []

    sim = LaunchConfiguration("sim")

    point_cloud_processor = OpaqueFunction(function=generate_point_cloud_processor)

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

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("nav")),
    )

    delayed_nav_launch = RegisterEventHandler(
        OnProcessStart(
            target_action=rtabmap_node,
            on_start=[TimerAction(period=5.0, actions=[nav_launch])],
        )
    )

    return LaunchDescription(
        declared_args
        + [
            point_cloud_processor,
            rtabmap_node,
            delayed_nav_launch,
        ]
    )
