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
from launch_ros.parameter_descriptions import ParameterValue


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
        DeclareLaunchArgument(
            "corrector",
            default_value="identity",
            description=(
                "Which runtime corrector runs in _correct(). "
                "identity -- pass-through (the fail-safe); "
                "tvlqr -- neighboring-optimal feedback about the planned "
                "trajectory (no training, publishes ~/tvlqr_diagnostics); "
                "rl -- the learned residual policy (needs rl_corrector.policy_path). "
                "tvlqr and rl are alternatives, not a stack: enabling tvlqr "
                "overrides a loaded policy."
            ),
        ),
    ]

    pmp_mode = LaunchConfiguration("pmp_mode")
    use_server = LaunchConfiguration("use_server")
    sim = LaunchConfiguration("sim")
    do_corrections = LaunchConfiguration("do_corrections")
    corrector = LaunchConfiguration("corrector")

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

    # Interpreter / corrector between the planner's output and the
    # JointGroupVelocityController input. Mode follows the planner's pmp_mode:
    #   online  -- pmp_planner publishes wheel commands on /pmp_planner/wheel_cmd;
    #              this node relays them (~/wheel_cmd_in -> ~/wheel_cmd_out).
    #   offline -- pmp_planner is a PlanToGoal action server; this node is the
    #              action client, sourcing the goal from /goal_pose + TF and
    #              playing the streamed trajectory back at the planned rate.
    # _correct() is identity for both today (the runtime-correction seam).
    wheel_corrector = Node(
        package="agx_planning",
        executable="runtime_corrector",
        output="screen",
        name="wheel_corrector",
        parameters=[
            {
                "mode": pmp_mode,
                "expected_size": 4,
                "use_sim_time": sim,
                # Offline playback: action + fallback sample rate (overridden
                # per-trajectory by the planner's committed chunk dt).
                "action_name": "pmp_planner/plan_to_goal",
                "control_rate": 10.0,
                # Debug-plan markers (mirrors runtime_corrector viz style).
                "publish_debug": True,
                "planning_frame": "map",
                "robot_frame": "base_link",
                "corridor_epsilon": 0.2,
                "debug_marker_rate": 5.0,
                # Neighboring-optimal corrector. Off unless corrector:=tvlqr,
                # so the default stays the identity pass-through. ParameterValue
                # with value_type=bool is required: the substitution yields the
                # string "true"/"false", which would not match the dataclass's
                # declared bool type otherwise.
                "tvlqr.enabled": ParameterValue(
                    EqualsSubstitution(corrector, "tvlqr"), value_type=bool
                ),
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
