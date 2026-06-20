from agx_bringup.utils import launch_file
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    EqualsSubstitution,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "pmp_mode",
            default_value="offline",
            description="Which PMP planner mode to use. Allowed values: online, offline.",
        ),
        DeclareLaunchArgument(
            "use_server",
            default_value="false",
            description="Whether to use a planner on a remote server.",
        ),
        DeclareLaunchArgument(
            "do_corrections",
            default_value="true",
            description="Whether to do runtime corrections of the trajectory.",
        ),
    ]

    pmp_mode = LaunchConfiguration("pmp_mode")
    use_server = LaunchConfiguration("use_server")
    sim = LaunchConfiguration("sim")
    do_corrections = LaunchConfiguration("do_corrections")

    vector_field = Node(
        package="agx_planning",
        executable="vector_field",
        output="screen",
        name="vector_field",
        parameters=[
            {
                "map_frame": "map",
                "robot_frame": "base_link",
                "allow_unknown": True,
                "speed_profile": "exponential",
                "inflation_radius": 0.5,
                "smooth_T_before_grad": False,
                "use_sim_time": sim,
            }
        ],
        remappings=[
            ("/vector_field/optimal_path", "/plan"),
        ],
    )

    launch_pmp_planner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file("planner")),
        condition=UnlessCondition(use_server),
    )

    # Pass-through corrector between the planner's wheel output and the
    # JointGroupVelocityController input. No-op for now (see wheel_corrector.py).
    # Online mode: pmp_planner publishes /pmp_planner/wheel_cmd directly.
    # Offline mode: once runtime_corrector is rewritten to emit wheel commands,
    # point it at /pmp_planner/wheel_cmd too; until then this node just idles.
    wheel_corrector = Node(
        package="agx_planning",
        executable="wheel_corrector",
        output="screen",
        name="wheel_corrector",
        parameters=[
            {
                "expected_size": 4,
                "use_sim_time": sim,
                # Debug-plan markers (mirrors runtime_corrector viz style).
                "publish_debug": True,
                "planning_frame": "map",
                "robot_frame": "base_link",
                "corridor_epsilon": 0.2,
                "debug_marker_rate": 5.0,
            }
        ],
        remappings=[
            ("~/wheel_cmd_in", "/pmp_planner/wheel_cmd"),
            ("~/wheel_cmd_out", "/wheel_velocity_controller/commands"),
            # Planner Path for the debug centerline/corridor. Matches the
            # planner's /pmp_planner/trajectory -> /optimal_trajectory remap.
            ("~/plan", "/optimal_trajectory"),
        ],
    )

    return LaunchDescription(
        declared_args
        + [
            vector_field,
            launch_pmp_planner,
            wheel_corrector,
        ]
    )
