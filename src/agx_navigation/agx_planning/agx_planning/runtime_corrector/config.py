from dataclasses import dataclass


@dataclass
class CorrectorConfig:
    """Pure algorithm parameters used by DeviationDetector and CorrectionController.

    No ROS2 infrastructure concerns (frames, topics, timers) belong here.
    """

    # Perpendicular distance from the robot to the path polyline that
    # triggers a transition from PLAYING to CORRECTING.
    corridor_epsilon: float = 0.10
    # Tighter spatial threshold that must be reached to leave CORRECTING.
    # Combined with recovery_angle_tolerance to form the exit condition.
    recovery_corridor_epsilon: float = 0.05
    # Heading alignment to the path tangent required to leave CORRECTING.
    recovery_angle_tolerance: float = 0.15
    recovery_look_ahead: float = 0.5
    recovery_v_max: float = 0.3
    recovery_omega_max: float = 1.0
    recovery_K_v: float = 1.0
    recovery_K_bearing: float = 2.0
    recovery_K_theta: float = 2.0
    path_diff_threshold: float = 0.5
    path_diff_percentile: float = 95.0
    # Minimum dot product of unit tangent vectors for direction-aware matching.
    # Rejects anti-parallel plan segments (e.g. U-turn return leg).
    # Range [-1.0, 1.0]: 0.0 = same hemisphere (recommended), -1.0 = disable,
    # 1.0 = colinear only (too strict).
    path_diff_min_tangent_dot: float = 0.0
    # Sliding window size [m] for the localised-deviation check.
    # 0.0 disables this check (percentile-only mode).
    path_diff_window_size: float = 0.0
    # LookAheadPursuit is suppressed when the bearing to the carrot exceeds
    # this angle, preventing the robot from spiralling around a lateral goal.
    recovery_max_pursuit_bearing_err: float = 1.5707963267948966  # pi/2 [rad]
    # Distance from the path end (measured at the nearest projection) within
    # which NearEndpointStrategy takes over to align heading before driving.
    recovery_near_endpoint_distance: float = 0.5  # [m]
