from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ComposableNode
from agx_bringup import Topics


def generate_launch_description():
    declared_args = [
        # World-frame spawn pose of the scout_mini. The default (0, 0, 0) is the
        # historical behaviour the map frame is anchored to (see
        # truth_localization.py, FRAMES): map == Gazebo world, so a plan's
        # coordinates are the robot's spawn pose ONLY under this default. A
        # comparison run that starts a plan elsewhere overrides all three; the
        # consumer that must move with it is amcl's initial_pose
        # (static_map.launch.py), which is in the map frame == this frame.
        DeclareLaunchArgument(
            "spawn_x", default_value="0.0",
            description="scout_mini world-frame spawn x [m]."),
        DeclareLaunchArgument(
            "spawn_y", default_value="0.0",
            description="scout_mini world-frame spawn y [m]."),
        DeclareLaunchArgument(
            "spawn_yaw", default_value="0.0",
            description="scout_mini world-frame spawn yaw [rad]."),
    ]

    sim = LaunchConfiguration("sim")
    spawn_x = LaunchConfiguration("spawn_x")
    spawn_y = LaunchConfiguration("spawn_y")
    spawn_yaw = LaunchConfiguration("spawn_yaw")

    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-ros-args",
            [
                "--remap joint_states:=",
                Topics.JOINT_STATES,
            ],
        ],
        output="screen",
        parameters=[{"use_sim_time": sim}],
    )

    wheel_velocity_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "wheel_velocity_controller",
            "--param-file",
            PathJoinSubstitution(
                [
                    FindPackageShare("scout_description"),
                    "config",
                    "wheel_velocity_controller.yaml",
                ]
            ),
        ],
        output="screen",
        parameters=[{"use_sim_time": sim}],
    )

    twist_to_wheels_node = Node(
        package="agx_chassis",
        executable="twist_to_wheels",
        name="twist_to_wheels",
        output="screen",
        parameters=[{"use_sim_time": sim}],
    )

    wheel_odometry_node = Node(
        package="agx_chassis",
        executable="wheel_odometry",
        name="wheel_odometry",
        output="screen",
        parameters=[
            {
                "wheel_radius": 0.08,
                "track": 0.416503,
                "use_sim_time": sim,
            }
        ],
    )

    robot_spawner = Node(
        package="ros_gz_sim",
        executable="create",
        name="scout_spawner",
        output="screen",
        arguments=[
            "-name",
            "scout_mini",
            "-topic",
            Topics.ROBOT_DESCRIPTION,
            "-allow_renaming",
            "true",
            "-x",
            spawn_x,
            "-y",
            spawn_y,
            "-z",
            "0.5",
            # Yaw needs the -Y flag (R/P/Y), not a bare argument: the create
            # CLI takes -x -y -z -R -P -Y, and a fixed 0 here would silently
            # drop the plan's start heading.
            "-Y",
            spawn_yaw,
        ],
    )

    camera_depth_pointcloud_transform = Node(
        package="topic_tools",
        executable="transform",
        name="frame_id_transformer",
        arguments=[
            Topics.CAMERA_DEPTH_POINTS_SIM_INTERMEDIATE,
            Topics.CAMERA_DEPTH_POINTS,
            "sensor_msgs/msg/PointCloud2",
            "(d:=copy.deepcopy(m), "
            "setattr(d.header, 'frame_id', 'd435_camera_depth_frame'), "
            "d)[2]",
            "--import",
            "sensor_msgs",
            "copy",
            "--wait-for-start",
        ],
        parameters=[{"use_sim_time": sim}],
        output="screen",
    )

    proc_params = {
        "use_sim_time": sim,
        "queue_size": 10,
    }

    rgbd_processing_container = ComposableNodeContainer(
        name="rgbd_processing_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package="depth_image_proc",
                plugin="depth_image_proc::RegisterNode",
                name="depth_register",
                parameters=[proc_params],
                remappings=[
                    # Inputs
                    ("depth/camera_info", Topics.CAMERA_DEPTH_INFO),
                    ("depth/image_rect", Topics.CAMERA_DEPTH_IMAGE),
                    ("rgb/camera_info", Topics.CAMERA_COLOR_INFO),
                    # Outputs
                    ("depth_registered/image_rect", Topics.CAMERA_RGBD_IMAGE),
                    ("depth_registered/camera_info", Topics.CAMERA_RGBD_INFO),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            # ComposableNode(
            #     package="depth_image_proc",
            #     plugin="depth_image_proc::PointCloudXyzrgbNode",
            #     name="depth_point_cloud",
            #     parameters=[proc_params],
            #     remappings=[
            #         # Inputs
            #         ("depth_registered/image_rect", Topics.CAMERA_RGBD_IMAGE),
            #         ("rgb/camera_info", Topics.CAMERA_COLOR_INFO),
            #         ("rgb/image_rect_color", Topics.CAMERA_COLOR_IMAGE),
            #         # Output
            #         ("points", Topics.CAMERA_RGBD_POINTS),
            #     ],
            #     extra_arguments=[{"use_intra_process_comms": True}],
            # ),
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            joint_state_spawner,
            wheel_velocity_spawner,
            twist_to_wheels_node,
            wheel_odometry_node,
            robot_spawner,
            camera_depth_pointcloud_transform,
            rgbd_processing_container,
        ]
    )
