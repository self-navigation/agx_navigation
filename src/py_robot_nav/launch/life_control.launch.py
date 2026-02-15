from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "port_name", default_value="can0", description="CAN bus name, e.g. can0."
        ),
        DeclareLaunchArgument(
            "odom_frame", default_value="odom", description="Odometry frame id."
        ),
        DeclareLaunchArgument(
            "base_frame", default_value="base_link", description="Base link frame id."
        ),
        DeclareLaunchArgument(
            "status_topic_name",
            default_value="/scout_status",
            description="Robot status topic name.",
        ),
        DeclareLaunchArgument(
            "light_cmd_topic_name",
            default_value="/light_control",
            description="Light controls topic name.",
        ),
    ]

    scout_base = Node(
        package="scout_base",
        executable="scout_base_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "port_name": LaunchConfiguration("port_name"),
                "odom_frame": LaunchConfiguration("odom_frame"),
                "base_frame": LaunchConfiguration("base_frame"),
                "odom_topic_name": LaunchConfiguration("odom_topic_name"),
                "status_topic_name": LaunchConfiguration("status_topic_name"),
                "motion_cmd_topic_name": LaunchConfiguration("motion_cmd_topic_name"),
                "light_cmd_topic_name": LaunchConfiguration("light_cmd_topic_name"),
                "is_scout_mini": True,
                "is_omni_wheel": False,
                "simulated_robot": False,
                "use_sim_time": LaunchConfiguration("sim"),
            }
        ],
    )

    rslidar_sdk = Node(
        package="rslidar_sdk",
        executable="rslidar_sdk_node",
        output="screen",
        parameters=[
            {
                "config_path": PathJoinSubstitution(
                    [FindPackageShare("py_robot_nav"), "config", "rslidar_config.yaml"]
                )
            }
        ],
    )

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("realsense2_camera"),
                    "launch",
                    "rs_launch.py",
                ]
            )
        )
    )

    return LaunchDescription(
        declared_args
        + [
            scout_base,
            rslidar_sdk,
            realsense,
        ]
    )
