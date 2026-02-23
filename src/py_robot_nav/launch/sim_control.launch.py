from launch import LaunchDescription
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ComposableNode
from py_robot_nav.launch import Topics


def generate_launch_description():
    declared_args = []

    sim = LaunchConfiguration("sim")

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

    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "diff_drive_controller",
            "--param-file",
            PathJoinSubstitution(
                [
                    FindPackageShare("scout_description"),
                    "config",
                    "diff_drive_controller.yaml",
                ]
            ),
            "--controller-ros-args",
            # Has to be specified with a list to be treated as one argument
            [
                "--remap diff_drive_controller/odom:=",
                Topics.ODOM,
                " ",
                "--remap diff_drive_controller/cmd_vel:=",
                Topics.CMD_VEL,
            ],
        ],
        output="screen",
        parameters=[{"use_sim_time": sim}],
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
            "-23",
            "-y",
            "-5",
            "-z",
            "0.5",
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
            ComposableNode(
                package="depth_image_proc",
                plugin="depth_image_proc::PointCloudXyzrgbNode",
                name="depth_point_cloud",
                parameters=[proc_params],
                remappings=[
                    # Inputs
                    ("depth_registered/image_rect", Topics.CAMERA_RGBD_IMAGE),
                    ("rgb/camera_info", Topics.CAMERA_COLOR_INFO),
                    ("rgb/image_rect_color", Topics.CAMERA_COLOR_IMAGE),
                    # Output
                    ("points", Topics.CAMERA_RGBD_POINTS),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            joint_state_spawner,
            diff_drive_spawner,
            robot_spawner,
            camera_depth_pointcloud_transform,
            rgbd_processing_container,
        ]
    )
