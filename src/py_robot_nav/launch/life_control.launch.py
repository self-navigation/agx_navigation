from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import Node
from py_robot_nav.launch import Topics, cfg_file


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "port_name", default_value="can0", description="CAN bus name, e.g. can0"
        ),
    ]

    port_name = LaunchConfiguration("port_name")

    scout_base = Node(
        package="scout_base",
        executable="scout_base_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "port_name": port_name,
                "odom_frame": "odom",
                "base_frame": "base_link",
                "odom_topic_name": Topics.ODOM,
                "status_topic_name": Topics.STATUS,
                "motion_cmd_topic_name": Topics.CMD_VEL,
                "light_cmd_topic_name": Topics.CMD_LIGHT,
                "is_scout_mini": True,
                "is_omni_wheel": False,
                "simulated_robot": False,
                "use_sim_time": False,
            }
        ],
    )

    imu_driver = Node(
        package="imu_driver",
        executable="imu_reader",
        output="screen",
        parameters=[
            {
                "port": "/dev/ttyUSB0",
                "baud_rate": 115200,
                "frame_id": "base_link",
                "use_sim_time": False,
            }
        ],
        remappintgs=[
            ("imu/data", Topics.IMU),
            ("imu/mag", Topics.MAGNETIC_FIELD),
        ],
    )

    rslidar_sdk = Node(
        package="rslidar_sdk",
        executable="rslidar_sdk_node",
        output="screen",
        parameters=[{"config_path": cfg_file("rslidar_config.yaml")}],
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
            (f"/{realsense_node_name}/depth/color/points", Topics.CAMERA_DEPTH_POINTS),
            (f"/{realsense_node_name}/color/image_raw", Topics.CAMERA_COLOR_IMAGE),
            (
                f"/{realsense_node_name}/depth/image_rect_raw",
                Topics.CAMERA_DEPTH_IMAGE,
            ),
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            scout_base,
            imu_driver,
            rslidar_sdk,
            realsense,
        ]
    )
