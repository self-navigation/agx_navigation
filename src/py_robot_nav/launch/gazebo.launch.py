
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import FindExecutable, PathJoinSubstitution, LaunchConfiguration, Command, EnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource

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

    # controller_manager = Node(
    #     package='controller_manager',
    #     executable='ros2_control_node',
    #     parameters=[{'robot_description': robot_description_content}, PathJoinSubstitution([FindPackageShare('scout_description'), 'config', 'scout_controller.yaml'])],
    #     output='screen'
    # )

    # diff_drive_spawner = Node(
    #     package='controller_manager',
    #     executable='spawner',
    #     arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
    #     output='screen'
    # )

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

            # controller_manager,
            # diff_drive_spawner,
    ]

def generate_launch_description():
    # Declare arguments (unchanged)
    declared_args = [
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation clock if true'),
        DeclareLaunchArgument('world', default_value='empty.sdf', description='SDF world file for Gazebo'),
        DeclareLaunchArgument('odom_frame', default_value='odom', description='Odometry frame id'),
        DeclareLaunchArgument('base_frame', default_value='base_link', description='Base link frame id'),
        DeclareLaunchArgument('odom_topic_name', default_value='odom', description='Odometry topic name'),
    ]

    # Set GZ_SIM_RESOURCE_PATH to enable model:// resolution for your package
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            PathJoinSubstitution([FindPackageShare('scout_description'), '..']),
            ':',
            EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value='')
        ]
    )

    # Path to the custom world (unchanged)
    world_path = PathJoinSubstitution([FindPackageShare('py_robot_nav'), 'worlds', LaunchConfiguration('world')])

    # Launch Gazebo with custom world and explicit GUI enabled (unchanged)
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ),
        launch_arguments={
            'gz_args': ['-v 6 -r ', world_path],  # -v4: verbose; add '-r ' if you want auto-run
            # 'gui': 'true'  # Explicitly enable GUI
        }.items()
    )

    # Spawn robot
    robot_spawner = Node(
        package='ros_gz_sim',
        executable='create',
        name='scout_spawner',
        output='screen',
        arguments=[
            '-name', 'scout_mini',
            '-topic', '/robot_description',
            '-allow_renaming', 'true',
            '-x', '0.0', '-y', '0.0', '-z', '0.5'  # Adjust pose
        ]
    )

    # ROS-GZ bridge with corrected directions ( [ for GZ->ROS, ] for ROS->GZ )
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        output='screen',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',  # ROS->GZ for commands
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',    # GZ->ROS for odom
            # RGB-D Camera
            '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',  # Depth (encoding: 32FC1)
            '/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',

            # Laser (LiDAR)
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
        ],
        parameters=[{'lazy': True}]  # Lazy bridging to reduce overhead
    )

    return LaunchDescription(
        declared_args +
        [
            set_gz_resource_path,  # Add this here (before gz_launch to ensure it's set early)
            gz_launch, 
            robot_spawner,
            gz_bridge,
        ]
        + scout_urdf()
    )