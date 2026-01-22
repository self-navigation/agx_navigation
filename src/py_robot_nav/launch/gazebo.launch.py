import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess, RegisterEventHandler, EmitEvent
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import FindExecutable, PathJoinSubstitution, LaunchConfiguration, Command, EnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

import launch_ros
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory

import os

def scout_urdf():
    model_name = 'mini.xacro'
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution(
            [FindPackageShare("scout_description"), "urdf", model_name]
        ),
    ])

    robot_description_content = ParameterValue(robot_description_content, value_type=str)

    return [
        launch_ros.actions.Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'robot_description': robot_description_content
            }]),
    ]

def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation clock if true'),
        DeclareLaunchArgument('odom_frame', default_value='odom', description='Odometry frame id'),
        DeclareLaunchArgument('base_frame', default_value='base_link', description='Base link frame id'),
        DeclareLaunchArgument('odom_topic_name', default_value='odom', description='Odometry topic name'),
        DeclareLaunchArgument('floor_number', default_value='3', description='On which floor of the RUDN building to perform the simulation.'),
    ]

    # Set GZ_SIM_RESOURCE_PATH to enable model:// resolution
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            PathJoinSubstitution([FindPackageShare('scout_description'), '..']),
            ':',
            EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value='')
        ]
    )

    world_path = PathJoinSubstitution([FindPackageShare('rudn_ordjo_building'), 'worlds', "ordjo_world.world"])

    # Directly launch GZ Sim using ExecuteProcess, to hook into process exit
    gz_process = ExecuteProcess(
        cmd=[
            FindExecutable(name='gz'),
            'sim',
            '-v', '6',
            '-r',
            world_path
        ],
        output='screen',
        name='gz_sim'
    )

    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=gz_process,
            on_exit=[
                EmitEvent(event=Shutdown())
            ]
        )
    )

    spawn_floor = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('rudn_ordjo_building'),
                'launch',
                'spawn_floor.launch.py'
            ])
        ),
        launch_arguments={
            'floor_number': LaunchConfiguration('floor_number'),
        }.items()
    )

    robot_spawner = Node(
        package='ros_gz_sim',
        executable='create',
        name='scout_spawner',
        output='screen',
        arguments=[
            '-name', 'scout_mini',
            '-topic', '/robot_description',
            '-allow_renaming', 'true',
            '-x', '-23', '-y', '-5', '-z', '0.5'
        ]
    )

    rotator = Node(
        package='py_robot_nav',
        executable='sim_point_cloud_fixup',
        name='sim_point_cloud_fixup',
        output='screen',
        parameters=[
            {'input_topic': '/camera/points'},  # Bridged raw topic
            {'output_topic': '/camera/points_corrected'},
            {'rotation_angle_deg': 90.0}  # Or -90.0 if over-correcting
        ]
    )

    # ROS-GZ bridge with corrected directions ( [ for GZ->ROS, ] for ROS->GZ )
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        output='screen',
        arguments=[
            # ROS->GZ for commands (unchanged)
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            # GZ->ROS for odom (corrected: GZ type first)
            '/odom@gz.msgs.Odometry[nav_msgs/msg/Odometry',
            # Camera (GZ->ROS)
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            # LiDAR Point Cloud (GZ->ROS)
            '/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        ],
        parameters=[{'lazy': True}]  # Still useful for efficiency
    )
    return LaunchDescription(
        declared_args +
        [
            set_gz_resource_path,  
            gz_process, 
            shutdown_handler,
            spawn_floor,
            robot_spawner,
            gz_bridge,
            rotator,
        ]
        + scout_urdf()
    )
