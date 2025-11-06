import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import FindExecutable, PathJoinSubstitution, FileContent
from launch.substitutions import LaunchConfiguration, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource

import launch_ros
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os

def scout_urdf():
    model_name = 'mini.xacro'
    # model_path = os.path.join(get_package_share_directory('scout_description'), "urdf", model_name)
    # print(model_path)

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution(
            [FindPackageShare("scout_description"), "urdf", model_name]
        ),
    ])
    # robot_description_content = launch_ros.parameter_descriptions.ParameterValue(
    #     FileContent(
    #         PathJoinSubstitution(
    #             [FindPackageShare("pro"), "urdf", 'pro.urdf']
    #         )
    #     ),
    #     value_type=str
    # )

    return [
        launch.actions.LogInfo(msg='use_sim_time: '),
        launch.actions.LogInfo(msg=launch.substitutions.LaunchConfiguration('use_sim_time')),

        launch_ros.actions.Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time'),
                'robot_description':robot_description_content
            }]),
    ]

def generate_launch_description():
    scout_base = [
        # scout_base scout_base.launch.py is_scout_mini:=true
    DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation clock if true'),
    DeclareLaunchArgument('port_name', default_value='can0', description='CAN bus name, e.g. can0'),
    DeclareLaunchArgument('odom_frame', default_value='odom', description='Odometry frame id'),
    DeclareLaunchArgument('base_frame', default_value='base_link', description='Base link frame id'),
    DeclareLaunchArgument('odom_topic_name', default_value='odom', description='Odometry topic name'),
    #DeclareLaunchArgument('is_scout_mini', default_value='true', description='Scout mini model'),
    #DeclareLaunchArgument('is_omni_wheel', default_value='false', description='Scout mini omni-wheel model'),
    DeclareLaunchArgument('simulated_robot', default_value='false', description='Whether running with simulator'),
    DeclareLaunchArgument('control_rate', default_value='50', description='Simulation control loop update rate'),
        Node(
            package='scout_base',
            executable='scout_base_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time'),
                'port_name': launch.substitutions.LaunchConfiguration('port_name'),
                'odom_frame': launch.substitutions.LaunchConfiguration('odom_frame'),
                'base_frame': launch.substitutions.LaunchConfiguration('base_frame'),
                'odom_topic_name': launch.substitutions.LaunchConfiguration('odom_topic_name'),
                'is_scout_mini': True,
                'is_omni_wheel': False,
                'simulated_robot': launch.substitutions.LaunchConfiguration('simulated_robot'),
                'control_rate': launch.substitutions.LaunchConfiguration('control_rate'),
        }]),
    ]

    rviz_config=get_package_share_directory('rslidar_sdk')+'/rviz/rviz2.rviz'

    config_file = '' # your config file path

    rslidar_sdk = [Node(namespace='rslidar_sdk', package='rslidar_sdk', executable='rslidar_sdk_node', output='screen', parameters=[{'config_path': config_file}])]
    realsense = [launch.actions.IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py'))
    )]

    odom_publisher = [
        Node(
            package='py_robot_nav',
            executable='agx_odometry_publisher',
            output='screen',
            emulate_tty=True,
            parameters=[{
            }]),
    ]


    return LaunchDescription(scout_base + rslidar_sdk + scout_urdf() + realsense + odom_publisher)
