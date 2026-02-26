from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    ExecuteProcess,
    RegisterEventHandler,
    EmitEvent,
    OpaqueFunction,
)
from launch.substitutions import (
    FindExecutable,
    PathJoinSubstitution,
    LaunchConfiguration,
    EnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from py_robot_nav.launch import Topics


def launch_gz_sim(context):
    headless = LaunchConfiguration("headless")
    is_headless = headless.perform(context).lower() in ["true", "1", "yes"]

    # Directly launch GZ Sim using ExecuteProcess, to hook into process exit
    world_path = PathJoinSubstitution(
        [FindPackageShare("rudn_ordjo_building"), "worlds", "ordjo_world.world"]
    )

    cmd = [FindExecutable(name="gz"), "sim", "-v", "6", "-r", world_path, "-s"]
    if is_headless:
        cmd.append("--headless-rendering")

    gz_process_server = ExecuteProcess(
        cmd=cmd,
        output="screen",
        name="gz_sim_server",
    )

    gz_process_gui = ExecuteProcess(
        cmd=[FindExecutable(name="gz"), "sim", "-g"],
        output="screen",
        name="gz_sim_gui",
        condition=UnlessCondition(headless),
    )

    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=gz_process_server, on_exit=[EmitEvent(event=Shutdown())]
        )
    )

    return [
        gz_process_server,
        gz_process_gui,
        shutdown_handler,
    ]


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "floor_number",
            default_value="3",
            description="On which floor of the RUDN building to perform the simulation.",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Enable headless rendering for gz sim.",
        ),
    ]

    headless = LaunchConfiguration("headless")
    floor_number = LaunchConfiguration("floor_number")
    sim = LaunchConfiguration("sim")

    # Set GZ_SIM_RESOURCE_PATH to enable model:// resolution
    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            PathJoinSubstitution([FindPackageShare("scout_description"), ".."]),
            ":",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )
    set_gz_system_plugin_path = SetEnvironmentVariable(
        name="GZ_SIM_SYSTEM_PLUGIN_PATH",
        value="/opt/ros/jazzy/lib/",
    )
    set_display = SetEnvironmentVariable(
        name="DISPLAY",
        value="",
        condition=IfCondition(headless),
    )

    gz_sim = OpaqueFunction(function=launch_gz_sim)

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
            "floor_number": floor_number,
            "x": "23",
            "y": "5",
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
            "/d435_camera/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/d435_camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/d435_camera/depth/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/d435_camera/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/d435_camera/depth/image_raw/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            # LiDAR Point Cloud
            "/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            # IMU and Magnetometer
            "/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/imu/mag@sensor_msgs/msg/MagneticField[gz.msgs.Magnetometer",
        ],
        parameters=[
            {
                "lazy": True,
                "use_sim_time": sim,
            }
        ],
        # For sim-to-life unity remapping topics to a common name.
        # NOTE: remapping depth points to an intermediate topic,
        # because its transform is fixed later in sim_control.launch.py
        remappings=[
            ("/d435_camera/color/image_raw", Topics.CAMERA_COLOR_IMAGE),
            ("/d435_camera/color/camera_info", Topics.CAMERA_COLOR_INFO),
            ("/d435_camera/depth/image_raw", Topics.CAMERA_DEPTH_IMAGE),
            (
                "/d435_camera/depth/image_raw/points",
                Topics.CAMERA_DEPTH_POINTS_SIM_INTERMEDIATE,
            ),
            ("/d435_camera/depth/camera_info", Topics.CAMERA_DEPTH_INFO),
            ("/imu/data", Topics.IMU),
            ("/imu/mag", Topics.MAGNETIC_FIELD),
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            set_gz_resource_path,
            set_gz_system_plugin_path,
            set_display,
            gz_sim,
            gz_bridge,
            spawn_floor,
        ]
    )
