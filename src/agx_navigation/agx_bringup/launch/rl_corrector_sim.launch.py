"""Minimal Gazebo bringup for RL-corrector training / GazeboBridge validation.

Deliberately trimmed vs the full stack: physics + wheel controller + joint states
+ wheel odometry only. NO slam, nav, planner, corrector (the GazeboBridge IS the
command source, so a corrector node would fight it for the command topic), and NO
surface patches (the bridge spawns terrain itself per-episode; a clean flat-ground
baseline must start patch-free). The rendering sensors (GPU lidar + RGB/depth
cameras) are dropped by default (sim_sensors:=false) since nothing here consumes
them and their per-tick rendering is the dominant sim cost; the IMU/magnetometer
stay on (cheap, no rendering).

Run:
    ros2 launch agx_bringup rl_corrector_sim.launch.py headless:=true

Then, in a sourced shell:
    python3 -m agx_planning.rl_corrector.validate_gazebo
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    EmitEvent,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    headless = LaunchConfiguration("headless")

    # rl_corrector.world runs uncapped, which is right for training but floods
    # /imu/data at ~3 kHz -- see rl_corrector_rt.world's header. Tools that drive
    # the robot and measure the response (slip_ident) want world:=rl_corrector_rt.world.
    world_path = PathJoinSubstitution(
        [FindPackageShare("rudn_ordjo_building"), "worlds", LaunchConfiguration("world")]
    )

    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            PathJoinSubstitution([FindPackageShare("scout_description"), ".."]),
            ":",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )
    set_gz_plugin_path = SetEnvironmentVariable(
        name="GZ_SIM_SYSTEM_PLUGIN_PATH", value="/opt/ros/jazzy/lib/"
    )

    # Server only (-s), run (-r). Headless uses EGL offscreen rendering.
    gz_server_headed = ExecuteProcess(
        cmd=[FindExecutable(name="gz"), "sim", "-v", "4", "-r", "-s", world_path],
        output="screen",
        name="gz_sim_server",
        condition=UnlessCondition(headless),
    )
    gz_server_headless = ExecuteProcess(
        cmd=[FindExecutable(name="gz"), "sim", "-v", "4", "-r", "-s",
             "--headless-rendering", world_path],
        output="screen",
        name="gz_sim_server",
        condition=IfCondition(headless),
    )
    gz_gui = ExecuteProcess(
        cmd=[FindExecutable(name="gz"), "sim", "-g"],
        output="screen",
        name="gz_sim_gui",
        condition=UnlessCondition(headless),
    )
    shutdown_on_server_exit = RegisterEventHandler(
        OnProcessExit(target_action=gz_server_headless,
                      on_exit=[EmitEvent(event=Shutdown())])
    )

    # /clock so use_sim_time clients have a sim clock; the GazeboBridge also gates
    # each deterministic step on it. /imu/data so the corrector obs can include the
    # IMU (a slip-observing, deployable signal) -- the GazeboBridge reads it over
    # ROS, exactly as the deployed corrector does, so train and deploy match. The
    # IMU sensor stays on even with sim_sensors:=false (it has no rendering cost).
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="rl_clock_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU",
        ],
        parameters=[{"use_sim_time": True}],
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("scout_description"), "launch", "scout_mini.launch.py"]
            )
        ),
        launch_arguments={
            "namespace": "",
            "sim_reduction": "4",
            "controller_file": "wheel_velocity_controller.yaml",
            "sim": "true",
            # Drop the GPU lidar + RGB/depth cameras: the GazeboBridge consumes only
            # ground-truth pose + the /odom twist, so the rendering sensors are pure
            # per-tick overhead during training. IMU/magnetometer stay on (cheap, and
            # the IMU is a slip-observing signal the policy may use). Overridable.
            "sim_sensors": LaunchConfiguration("sim_sensors"),
        }.items(),
    )

    robot_spawner = Node(
        package="ros_gz_sim",
        executable="create",
        name="scout_spawner",
        output="screen",
        arguments=[
            "-name", "scout_mini",
            "-topic", "robot_description",
            "-z", "0.5",
        ],
    )

    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    wheel_velocity_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "wheel_velocity_controller",
            "--param-file",
            PathJoinSubstitution(
                [FindPackageShare("scout_description"), "config",
                 "wheel_velocity_controller.yaml"]
            ),
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    wheel_odometry = Node(
        package="agx_chassis",
        executable="wheel_odometry",
        name="wheel_odometry",
        output="screen",
        parameters=[{"wheel_radius": 0.08, "track": 0.416503, "use_sim_time": True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="true"),
        # Off by default here: this launch exists for training/validation, where the
        # rendering sensors are unused overhead. Set sim_sensors:=true to inspect them.
        DeclareLaunchArgument("sim_sensors", default_value="false"),
        DeclareLaunchArgument("world", default_value="rl_corrector.world"),
        set_gz_resource_path,
        set_gz_plugin_path,
        gz_server_headed,
        gz_server_headless,
        gz_gui,
        shutdown_on_server_exit,
        clock_bridge,
        robot_state_publisher,
        robot_spawner,
        joint_state_spawner,
        wheel_velocity_spawner,
        wheel_odometry,
    ])
