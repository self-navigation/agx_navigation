from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    EqualsSubstitution,
)
from launch_ros.actions import Node

from agx_bringup import Topics


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "pmp_mode",
            default_value="online",
            description="Which PMP planner mode to use. Allowed values: online, offline.",
        ),
    ]

    pmp_mode = LaunchConfiguration("pmp_mode")
    sim = LaunchConfiguration("sim")

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
                "smooth_T_before_grad": False,
                "use_sim_time": sim,
            }
        ],
        remappings=[
            ("/vector_field/optimal_path", "/plan"),
        ],
    )

    pmp_planner = Node(
        package="agx_planning",
        executable="pmp_planner",
        output="screen",
        name="pmp_planner",
        parameters=[
            {
                "mode": pmp_mode,
                "enable_stamped_cmd_vel": True,
                "enable_confidence_weighting": False,
                "use_sim_time": sim,
            }
        ],
        remappings=[
            ("/pmp_planner/trajectory", "/optimal_trajectory"),
            ("/odom", Topics.ODOM_FILTERED),
        ],
    )

    pmp_interpreter = Node(
        package="agx_planning",
        executable="pmp_interpreter",
        output="screen",
        name="pmp_interpreter",
        parameters=[
            {
                "enable_stamped_cmd_vel": True,
                "use_sim_time": sim,
            }
        ],
        condition=IfCondition(EqualsSubstitution(pmp_mode, "offline")),
    )

    pmp_plan_interpreter = Node(
        package="pmp_plan_interpreter",
        executable="pmp_plan_interpreter",
        output="screen",
        name="plan_interpreter",
        parameters=[],
        remappings=[],
        condition=IfCondition(EqualsSubstitution(pmp_mode, "offline")),
    )

    return LaunchDescription(
        declared_args
        + [
            vector_field,
            pmp_planner,
            pmp_interpreter,
            pmp_plan_interpreter,
        ]
    )
