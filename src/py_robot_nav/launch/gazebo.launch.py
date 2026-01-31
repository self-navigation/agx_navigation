import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess, RegisterEventHandler, EmitEvent
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import FindExecutable, PathJoinSubstitution, LaunchConfiguration, Command, EnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition, UnlessCondition

import launch_ros
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory

import os

def scout():
    model_name = 'mini.xacro'
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution(
            [FindPackageShare("scout_description"), "urdf", model_name]
        ),
    ])

    robot_description_content = ParameterValue(robot_description_content, value_type=str)

    robot_state = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': robot_description_content
        }]
    )

    joint_state_spawner = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster'],
        output='screen'
    )

    diff_drive_spawner = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'diff_drive_controller'],
        output='screen'
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

    return [
        robot_spawner,
        robot_state,
        joint_state_spawner,
        diff_drive_spawner,
    ]

def slam():
    p2s = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[
            ('cloud_in', '/lidar/points'),
            ('scan', '/lidar/laserscan')
        ],
        parameters=[{
            'target_frame': '',
            'transform_tolerance': 0.1,

            'min_height': -.1,
            'max_height': 0.1,

            # [-pi; +pi] because lidar has 360 degrees FOV
            'angle_min': -3.14159,
            'angle_max': 3.14159,

            'angle_increment': 0.0087,  # M_PI/360.0
            'scan_time': 0.1,

            'range_min': 0.2,
            'range_max': 150.0,

            # outputs NaN for no-return rays
            'use_inf': False,

            # auto-detect
            'concurrency_level': 0
        }],
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py',
            ])
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file': PathJoinSubstitution([
                FindPackageShare('py_robot_nav'), 'config', 'nav2_params.yaml'
            ]),
        }.items()
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py',
            ])
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'slam_params_file': PathJoinSubstitution([
                FindPackageShare('py_robot_nav'), 'config', 'slam_params.yaml'
            ]),
        }.items()
    )

    return [
        p2s,
        # nav2_launch,
        # slam_launch,
    ]

def gz_sim():
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
    base_cmd = [
        FindExecutable(name='gz'),
        'sim',
        '-v',
        '6',
        '-r',
        world_path,
    ]
    
    gz_process_gui = ExecuteProcess(
        cmd=base_cmd,
        output='screen',
        name='gz_sim',
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )
    gz_process_headless = ExecuteProcess(
        cmd=base_cmd + ['-s', '--headless-rendering'],
        output='screen',
        name='gz_sim_headless',
        condition=IfCondition(LaunchConfiguration('headless'))
    )

    shutdown_handler_gui = RegisterEventHandler(
        OnProcessExit(
            target_action=gz_process_gui,
            on_exit=[
                EmitEvent(event=Shutdown())
            ]
        )
    )
    shutdown_handler_headless = RegisterEventHandler(
        OnProcessExit(
            target_action=gz_process_headless,
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

    return [
        set_gz_resource_path,  
        gz_process_gui, 
        gz_process_headless, 
        shutdown_handler_gui,
        shutdown_handler_headless,
        spawn_floor,
    ]

def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument('odom_topic_name', default_value='/diff_drive_controller/odom', description='Odometry topic name'),
        DeclareLaunchArgument('use_sim_time', default_value='True', description='Use simulation clock if true'),
        DeclareLaunchArgument('floor_number', default_value='3', description='On which floor of the RUDN building to perform the simulation.'),
        DeclareLaunchArgument('headless', default_value='False', description='Enable headless rendering for gz sim.'),
    ]

    # ROS-GZ bridge with corrected directions ( [ for GZ->ROS, ] for ROS->GZ )
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        output='screen',
        arguments=[
            # ROS->GZ
            # Twist
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',

            # GZ->ROS
            # Clock
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Camera
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            # LiDAR Point Cloud
            '/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        ],
        parameters=[{'lazy': True}]  # Still useful for efficiency
    )

    return LaunchDescription(
        declared_args
        + gz_sim()
        + [
            gz_bridge,
        ]
        + scout()
        + slam()
    )
