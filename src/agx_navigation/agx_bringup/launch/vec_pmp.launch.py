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

    pmp_interpreter = Node(
        package="agx_planning",
        executable="runtime_corrector",
        output="screen",
        name="runtime_corrector",
        parameters=[
            {
                "enable_stamped_cmd_vel": True,
                "enable_recovery": do_corrections,
                "recovery_look_ahead": 2.0,
                "recovery_v_max": 0.448,
                "path_diff_window_size": 0.5,
                "replan_cooldown": 1.0,
                "use_sim_time": sim,
            }
        ],
        remappings=[
            ("/vector_field/optimal_path", "/plan"),
        ],
        condition=IfCondition(EqualsSubstitution(pmp_mode, "offline")),
    )

    return LaunchDescription(
        declared_args
        + [
            vector_field,
            launch_pmp_planner,
            pmp_interpreter,
        ]
    )
