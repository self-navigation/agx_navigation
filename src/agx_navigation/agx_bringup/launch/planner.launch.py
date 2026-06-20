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

    # Values are your previously calibrated/tuned numbers where the field
    # carried over; the PlannerConfig default is noted in-line where they
    # differ so the reconciliation is auditable. New wheel-space fields use
    # config defaults except where your calibration implies otherwise.
    pmp_planner = Node(
        package="agx_planning",
        executable="pmp_planner",
        output="screen",
        name="pmp_planner",
        parameters=[
            {
                # --- Node-level ---
                "mode": pmp_mode,
                "use_sim_time": sim,
                "diag_log_path": "/tmp/pmp.csv",
                # Route online output through wheel_corrector instead of
                # straight to /wheel_velocity_controller/commands.
                "wheel_cmd_topic": "/pmp_planner/wheel_cmd",
                # Not present in the PlannerConfig you sent; if your NodeConfig
                # still declares it this keeps prior behaviour, else ignored.
                "enable_confidence_weighting": False,
                # Online-mode BVP/publish rate [Hz].
                "control_rate": 10.0,

                # --- Horizon / collocation (your tuned values) ---
                "T_horizon": 4.0,        # default 2.5
                "N": 40,                 # default 21
                "bvp_max_nodes": 3000,   # default 2000
                "dt_segment": 0.5,       # offline only; default 1.25

                # --- Body-space behaviour bounds (soft barriers, calibrated) ---
                "v_max": 0.448,          # default 0.5
                "omega_max": 1.049,      # default 1.5

                # --- Running-cost weights (your tuned values) ---
                "w_h": 10.0,             # default 5.0
                "w_v": 3.0,              # default 0.5
                "w_brake": 200.0,        # was misspelled w_break before; default 200
                "L_brake": 1.50,         # default 0.5
                "align_gate_power": 15.0,  # default 4.0 -- near-binary; review
                "w_v_barrier": 200.0,    # default 50.0
                "w_v_terminal": 15.0,    # default 5.0
                "pursuit_lookahead_mult": 0.6,  # default 1.0
                # w_omega_run (default 0.1) is the turn-in damper now that
                # wheel-space makes turning ~5x cheaper; raise if it over-turns.

                # --- Wheel-space geometry + actuation (new model) ---
                "wheel_radius": 0.08,
                "track": 0.416503,       # == diff_drive wheel_separation
                # Sim ICR value (chi >= 1). NOT 1/chassis_gain_omega: that
                # identity was for the old feedforward model. Re-id per surface.
                "slip_chi": 1.2987,
                # = your old a_max (1.408) / wheel_radius. Config default 12.5
                # assumes a_max=1.0; use that if matching the rewrite baseline.
                "a_wheel_max": 17.6,
                "gamma_wheel": 0.0016,
                "w_wheel_max": 20.0,
                "w_wheel_barrier": 50.0,
                "wheel_cmd_max": 20.0,   # == <limit velocity="20">
                "tau_wheel": 0.0,        # gz velocity interface tracks in-step

                # --- Command deadzone (body space, before wheel mapping) ---
                "cmd_deadzone_v": 0.03,
                "cmd_deadzone_omega": 0.05,
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
