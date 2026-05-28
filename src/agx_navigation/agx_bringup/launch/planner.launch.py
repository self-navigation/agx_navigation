from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import Node

from agx_bringup import Topics


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "sim",
            default_value="false",
            description="Run in Gazebo sim (spawns the robot and uses sim time).",
        ),
        DeclareLaunchArgument(
            "pmp_mode",
            default_value="offline",
            description="Which PMP planner mode to use. Allowed values: online, offline.",
        ),
    ]

    pmp_mode = LaunchConfiguration("pmp_mode")
    sim = LaunchConfiguration("sim")

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
                "diag_log_path": "/tmp/pmp.csv",
                # PlannerConfig
                "omega_max": 1.099,  # BVP's planning bound on desired body omega
                "alpha_max": 0.777,
                "v_max": 0.547,
                "a_max": 1.339,
                "chassis_gain_omega": 1.187,  # measured slip ratio
                "chassis_gain_v": 1.0,
                "chassis_tau_omega": 0.163,
                "chassis_tau_v": 0.135,
                "chassis_omega_max": 1.549,  # well above the inversion's peak demand
                "chassis_v_max": 1.306,
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

    return LaunchDescription(
        declared_args
        + [
            pmp_planner,
        ]
    )
