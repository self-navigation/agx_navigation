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
            default_value="offline",
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
                "diag_log_path": "",
                # PlannerConfig
                "omega_max": 0.60,  # BVP's planning bound on desired body omega
                "alpha_max": 0.80,
                "v_max": 0.40,
                "a_max": 1.00,
                "chassis_gain_omega": 0.77,  # measured slip ratio
                "chassis_gain_v": 1.0,
                "chassis_tau_omega": 0.3,
                "chassis_tau_v": 0.05,
                "chassis_omega_max": 5.00,  # well above the inversion's peak demand
                "chassis_v_max": 5.00,
                "dt_segment": 0.5,
                "pursuit_lookahead_mult": 0.6,
                "align_gate_power": 10.0,  # was 4.0
                "L_brake": 0.30,  # was 0.5
                "w_v_barrier": 200.0,  # was 50.0
                "w_v_terminal": 15.0,  # was 5.0
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

    return LaunchDescription(
        declared_args
        + [
            vector_field,
            pmp_planner,
            pmp_interpreter,
        ]
    )
