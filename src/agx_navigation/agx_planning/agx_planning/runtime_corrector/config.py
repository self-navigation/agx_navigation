from dataclasses import dataclass


@dataclass
class CorrectorConfig:
    enable_stamped_cmd_vel: bool = False
    robot_frame: str = "base_link"
    planning_frame: str = "map"
    # Tick-rate fallback used only before the first chunk arrives; after
    # that we use chunk.dt (whatever rate the planner committed to).
    default_tick_rate: float = 10.0
    # Perpendicular distance from the robot to the path polyline that
    # triggers a transition from PLAYING to CORRECTING.
    corridor_epsilon: float = 0.10
    # Tighter spatial threshold that must be reached to leave CORRECTING.
    # Combined with recovery_angle_tolerance to form the exit condition.
    recovery_corridor_epsilon: float = 0.05
    # Heading alignment to the path tangent required to leave CORRECTING.
    recovery_angle_tolerance: float = 0.15
    enable_recovery: bool = True
    recovery_look_ahead: float = 0.5
    recovery_v_max: float = 0.3
    recovery_omega_max: float = 1.0
    recovery_K_v: float = 1.0
    recovery_K_bearing: float = 2.0
    recovery_K_theta: float = 2.0
    path_diff_threshold: float = 0.5
    path_diff_percentile: float = 95.0
    replan_cooldown: float = 0.0
    # Arc-length [m] to skip forward along both paths before comparing.
    # Avoids transient disagreement at the robot's feet due to message latency.
    path_diff_skip_ahead: float = 0.0
    # Minimum dot product of unit tangent vectors for direction-aware matching.
    # Rejects anti-parallel plan segments (e.g. U-turn return leg).
    # Range [-1.0, 1.0]: 0.0 = same hemisphere (recommended), -1.0 = disable,
    # 1.0 = colinear only (too strict).
    path_diff_min_tangent_dot: float = 0.0
    # Sliding window size [m] for the localised-deviation check.
    # 0.0 disables this check (percentile-only mode).
    path_diff_window_size: float = 0.0
    action_name: str = "pmp_planner/plan_to_goal"
    # How long to wait for the server to ACCEPT a sent goal before retrying.
    goal_accept_timeout: float = 2.0
