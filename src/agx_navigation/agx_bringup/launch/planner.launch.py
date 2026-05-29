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
                "T_horizon": 4.0,
                "bvp_max_nodes": 3000,
                "N": 40,
                "w_h": 10.0,
                "w_break": 200,
                "dt_segment": 0.5,
                "pursuit_lookahead_mult": 0.6,
                "align_gate_power": 15.0,
                "L_brake": 1.50,
                "w_v": 3.0,
                "w_v_barrier": 200.0,
                "w_v_terminal": 15.0,
                # calibrated
                "omega_max": 1.049,
                "alpha_max": 0.962,
                "chassis_gain_omega": 1.053,
                "chassis_tau_omega": 0.128,
                "chassis_omega_max": 1.668,
                "v_max": 0.448,
                "a_max": 1.408,
                "chassis_gain_v": 0.939,
                "chassis_tau_v": 0.122,
                "chassis_v_max": 0.990,
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
