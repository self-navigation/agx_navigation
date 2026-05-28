from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import Node
from agx_bringup import RewrittenYaml, Topics, cfg_file


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "port_name", default_value="can0", description="CAN bus name, e.g. can0"
        ),
    ]

    port_name = LaunchConfiguration("port_name")

    # cmd_vel_repeater = Node(
    #     package="scout_utils",
    #     executable="cmd_vel_repeater",
    #     name="cmd_vel_repeater",
    #     parameters=[
    #         {
    #             "input_topic": Topics.CMD_VEL,
    #             "output_topic": Topics.CMD_VEL_RAW,
    #             "publish_rate_hz": 10.0,
    #             "input_timeout_sec": 3.0,
    #             "use_stamped_cmd_vel": True,
    #             "use_sim_time": False,
    #         }
    #     ],
    # )

    cmd_vel_repeater = Node(
        package="topic_tools",
        executable="relay",
        name="my_relay",
        parameters=[
            {"input_topic": Topics.CMD_VEL},
            {"output_topic": Topics.CMD_VEL_RAW},
        ],
    )

    scout_base = Node(
        package="scout_base",
        executable="scout_base_node",
        name="scout",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "port_name": port_name,
                "odom_frame": "odom",
                "base_frame": "base_link",
                "odom_topic_name": Topics.ODOM,
                "status_topic_name": Topics.STATUS,
                "motion_cmd_topic_name": Topics.CMD_VEL_RAW,
                "light_cmd_topic_name": Topics.CMD_LIGHT,
                "is_scout_mini": True,
                "is_omni_wheel": False,
                "simulated_robot": False,
                "use_sim_time": False,
                "use_stamped_cmd_vel": True,
                "publish_tf": False,
            }
        ],
        arguments=["--ros-args", "--log-level", "scout:=WARN"],
    )

    imu_driver = Node(
        package="imu_driver",
        executable="imu_reader",
        output="screen",
        parameters=[
            {
                "port": "/dev/ttyUSB0",
                "baud_rate": 115200,
                "frame_id": "imu_link",
                "use_sim_time": False,
                "publish_euler": True,
            }
        ],
        remappings=[
            ("imu/data", Topics.IMU),
            ("imu/mag", Topics.MAGNETIC_FIELD),
        ],
    )

    rslidar_params = RewrittenYaml(
        source_file=cfg_file("rslidar_config.yaml"),
        root_key="",
        param_rewrites={},
        value_rewrites={
            "POINTCLOUD": Topics.LIDAR_POINTS,
        },
        convert_types=True,
    )

    rslidar_sdk = Node(
        package="rslidar_sdk",
        executable="rslidar_sdk_node",
        output="screen",
        parameters=[{"config_path": rslidar_params}],
    )

    # Set to d435_camera to make more topics unify with sim
    realsense_node_name = "d435_camera"
    realsense = Node(
        package="realsense2_camera",
        name=realsense_node_name,
        namespace="",
        executable="realsense2_camera_node",
        parameters=[cfg_file("realsense_params.yaml")],
        output="screen",
        remappings=[
            # Color
            (f"/{realsense_node_name}/color/image_raw", Topics.CAMERA_COLOR_IMAGE),
            (f"/{realsense_node_name}/color/camera_info", Topics.CAMERA_COLOR_INFO),
            # Depth
            (f"/{realsense_node_name}/depth/color/points", Topics.CAMERA_DEPTH_POINTS),
            (f"/{realsense_node_name}/depth/camera_info", Topics.CAMERA_DEPTH_INFO),
            (
                f"/{realsense_node_name}/depth/image_rect_raw",
                Topics.CAMERA_DEPTH_IMAGE,
            ),
            # RGBD
            (
                f"/{realsense_node_name}/aligned_depth_to_color/image_raw",
                Topics.CAMERA_RGBD_IMAGE,
            ),
            (
                f"/{realsense_node_name}/aligned_depth_to_color/camera_info",
                Topics.CAMERA_RGBD_INFO,
            ),
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            cmd_vel_repeater,
            scout_base,
            imu_driver,
            rslidar_sdk,
            realsense,
        ]
    )
