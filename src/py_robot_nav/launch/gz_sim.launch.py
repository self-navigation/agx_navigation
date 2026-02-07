from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    ExecuteProcess,
    RegisterEventHandler,
    EmitEvent,
)
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import (
    FindExecutable,
    PathJoinSubstitution,
    LaunchConfiguration,
    EnvironmentVariable,
    TextSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition, UnlessCondition

from launch_ros.actions import Node


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "floor_number",
            default_value="3",
            description="On which floor of the RUDN building to perform the simulation.",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="False",
            description="Enable headless rendering for gz sim.",
        ),
    ]

    # Set GZ_SIM_RESOURCE_PATH to enable model:// resolution
    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            PathJoinSubstitution([FindPackageShare("scout_description"), ".."]),
            ":",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )

    world_path = PathJoinSubstitution(
        [FindPackageShare("rudn_ordjo_building"), "worlds", "ordjo_world.world"]
    )

    # Directly launch GZ Sim using ExecuteProcess, to hook into process exit
    base_cmd = [
        FindExecutable(name="gz"),
        "sim",
        "-v",
        "6",
        "-r",
        world_path,
    ]

    gz_process_gui = ExecuteProcess(
        cmd=base_cmd,
        output="screen",
        name="gz_sim",
        condition=UnlessCondition(LaunchConfiguration("headless")),
    )
    gz_process_headless = ExecuteProcess(
        cmd=base_cmd + ["-s", "--headless-rendering"],
        output="screen",
        name="gz_sim_headless",
        condition=IfCondition(LaunchConfiguration("headless")),
    )

    shutdown_handler_gui = RegisterEventHandler(
        OnProcessExit(
            target_action=gz_process_gui, on_exit=[EmitEvent(event=Shutdown())]
        )
    )
    shutdown_handler_headless = RegisterEventHandler(
        OnProcessExit(
            target_action=gz_process_headless, on_exit=[EmitEvent(event=Shutdown())]
        )
    )

    spawn_floor = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("rudn_ordjo_building"),
                    "launch",
                    "spawn_floor.launch.py",
                ]
            )
        ),
        launch_arguments={
            "floor_number": LaunchConfiguration("floor_number"),
        }.items(),
    )

    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_bridge",
        output="screen",
        arguments=[
            # GZ->ROS
            # Clock
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            # Camera
            "camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            # LiDAR Point Cloud
            "lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        parameters=[
            {
                "lazy": True,
                "use_sim_time": LaunchConfiguration("sim"),
            }
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            set_gz_resource_path,
            gz_process_gui,
            gz_process_headless,
            shutdown_handler_gui,
            shutdown_handler_headless,
            spawn_floor,
            gz_bridge,
        ]
    )
