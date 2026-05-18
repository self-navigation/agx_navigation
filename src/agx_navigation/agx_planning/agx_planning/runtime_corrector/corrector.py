"""Trajectory corrector for the offline-mode PMP planner.

Uses an ActionClient to request trajectories from the planner's
`pmp_planner/plan_to_goal` action server, then plays them back and corrects
deviations from the planned path.

Behaviour is a three-state machine:

  IDLE        -- no active trajectory. Publishes zero twist on every tick.
  PLAYING     -- streaming chunks of a known trajectory_id. Each tick advances
                 one sample within the buffered chunks. When the action result
                 arrives AND the buffer is drained, transition to IDLE.
  CORRECTING  -- the robot has left the epsilon corridor around the planned
                 path. Playback is suspended; a corrective maneuver brings
                 the robot back within the tighter recovery corridor and into
                 heading alignment. A new trajectory_id from the planner
                 (replan) transitions back to PLAYING at any time.

Goal source: subscribes to /goal_pose (PoseStamped). New goals trigger an
immediate action send. The empty-frame_id sentinel (published on goal
completion) clears the cached goal.

Replan trigger: subscribes to /vector_field/optimal_path (Path), compares
the FM2 gradient path against the buffered planned trajectory, and re-fires
the action goal when the path_diff_percentile-th percentile of per-point
cross-track distances exceeds path_diff_threshold.

Corridor semantics: the entry threshold (corridor_epsilon) is compared against
the robot's perpendicular distance to the path polyline built from all buffered
chunks. There is no separate angle check on entry -- if the robot drifts far
enough off heading, it will eventually leave the spatial corridor too. The exit
threshold (recovery_corridor_epsilon) is tighter (hysteresis) and is combined
with a heading-alignment check against the path tangent at the nearest
projection point to prevent resuming while still pointed the wrong way.

Atomic switch on a new trajectory_id: any chunk with a higher trajectory_id
than the one currently playing immediately replaces the buffer and transitions
to PLAYING. The old trajectory's remaining samples are discarded -- the new
trajectory was generated from the chassis's CURRENT TF pose, so the old plan
is now stale.

Out-of-order chunk arrival within a trajectory_id: chunks are stored in
a dict keyed by chunk_index and consumed in index order. If chunk N+1
arrives before N has been fully played, that's fine; we simply pull it
when needed. If chunk N is missing when we need it, we publish zero
twist and wait. This shouldn't happen under reliable QoS but is the
graceful fallback.
"""

import math
import time
from enum import Enum, auto
from typing import Dict, Optional

import rclpy

import numpy as np

from builtin_interfaces.msg import Duration as BuiltinDuration
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64, String
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Duration, Time
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
    TransformException,
)
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

from agx_planning_msgs.action import PlanToGoal
from agx_planning.runtime_corrector import RecoveryConfig, default_strategies
from agx_planning.runtime_corrector.geometry import (
    nearest_projection_on_path,
    walk_ahead_on_path,
)

_MARKER_NS = "pmp_trajectory_corrector"
_MARKER_LIFETIME = BuiltinDuration(sec=1, nanosec=0)


# ---------------------------------------------------------------------------
# Corrector node
# ---------------------------------------------------------------------------


def _advance_by_arc(path_xy: np.ndarray, start: int, distance: float) -> int:
    """Return the index in path_xy that is at least `distance` metres of
    cumulative arc length ahead of `start`. Returns the last valid index
    if the path is shorter than `distance`.
    """
    acc = 0.0
    for i in range(start, len(path_xy) - 1):
        acc += float(np.hypot(path_xy[i + 1, 0] - path_xy[i, 0],
                              path_xy[i + 1, 1] - path_xy[i, 1]))
        if acc >= distance:
            return i + 1
    return len(path_xy) - 1


def _path_tangents(path_xy: np.ndarray) -> np.ndarray:
    """Unit tangent vectors for each point in path_xy (N, 2).

    Uses forward differences for all points, repeating the last tangent
    at the endpoint. Zero-length segments produce a zero vector (the
    direction filter will treat them as invalid matches, which is safe).
    """
    diffs = np.diff(path_xy, axis=0)          # (N-1, 2)
    norms = np.hypot(diffs[:, 0], diffs[:, 1])
    # Avoid division by zero; zero-norm tangents are left as (0, 0).
    safe = norms > 1e-9
    unit = np.where(safe[:, np.newaxis], diffs / np.where(safe, norms, 1.0)[:, np.newaxis], 0.0)
    # Repeat last tangent for the final point.
    return np.vstack([unit, unit[-1:]])        # (N, 2)


def _windowed_max_deviation(
    deviations: np.ndarray, arc_s: np.ndarray, window_size: float
) -> float:
    """Largest per-point deviation found within any arc-length window.

    Uses a two-pointer sweep so the right boundary is never moved backward;
    total pointer travel is O(N). np.max over the (typically short) window
    slice is fast enough for N <= 2000 at 5 Hz.

    inf values in deviations (points with no valid directional match)
    propagate naturally through np.max, so a window containing a genuinely
    unmatched point always reports inf and triggers a replan.
    """
    n = len(deviations)
    if n == 0:
        return 0.0
    best = 0.0
    right = 0
    for left in range(n):
        if right < left:
            right = left
        while right + 1 < n and arc_s[right + 1] - arc_s[left] <= window_size:
            right += 1
        w_max = float(np.max(deviations[left : right + 1]))
        if w_max > best:
            best = w_max
    return best


class _State(Enum):
    IDLE = auto()
    PLAYING = auto()
    CORRECTING = auto()


class TrajectoryCorrectorNode(Node):

    def __init__(self):
        super().__init__("pmp_trajectory_corrector")

        self.declare_parameter("enable_stamped_cmd_vel", False)
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("planning_frame", "map")
        # Tick-rate fallback used only before the first chunk arrives.
        # Once a chunk is in hand, we use chunk.dt instead -- whatever
        # rate the planner committed to.
        self.declare_parameter("default_tick_rate", 10.0)
        # Perpendicular distance from the robot to the path polyline that
        # triggers a transition from PLAYING to CORRECTING.
        self.declare_parameter("corridor_epsilon", 0.10)
        # Tighter spatial threshold that must be reached to leave CORRECTING.
        # Combined with recovery_angle_tolerance to form the exit condition.
        self.declare_parameter("recovery_corridor_epsilon", 0.05)
        # Heading alignment to the path tangent required to leave CORRECTING.
        self.declare_parameter("recovery_angle_tolerance", 0.15)
        # Recovery controller parameters.
        self.declare_parameter("enable_recovery", True)
        self.declare_parameter("recovery_look_ahead", 0.5)
        self.declare_parameter("recovery_v_max", 0.3)
        self.declare_parameter("recovery_omega_max", 1.0)
        self.declare_parameter("recovery_K_v", 1.0)
        self.declare_parameter("recovery_K_bearing", 2.0)
        self.declare_parameter("recovery_K_theta", 2.0)
        # Replan trigger parameters.
        self.declare_parameter("path_diff_threshold", 0.5)
        self.declare_parameter("path_diff_percentile", 95.0)
        self.declare_parameter("replan_cooldown", 0.0)
        # Arc-length [m] to skip forward along both paths from the nearest
        # point to the robot before starting the comparison. Avoids transient
        # disagreement right at the robot's feet where message latency means
        # both paths are slightly behind the actual position. 0.0 = no skip.
        self.declare_parameter("path_diff_skip_ahead", 0.0)
        # Minimum dot product of unit tangent vectors required before a plan
        # point is accepted as a valid nearest-neighbour for a gradient path
        # point. Rejects anti-parallel plan segments (opposite travel direction)
        # that arise in U-turns and hairpins, where the plan's return leg would
        # otherwise produce a spuriously small distance to the gradient path's
        # outbound leg. Range [-1.0, 1.0]:
        #   0.0  -- accept only plan points heading within 90 deg of gradient
        #            (recommended default; no effect on straight-line travel)
        #  -1.0  -- disable direction filtering (original behaviour)
        #   1.0  -- accept only perfectly co-linear segments (too strict)
        self.declare_parameter("path_diff_min_tangent_dot", 0.0)
        # Sliding window size [m] for the localised-deviation check. For each
        # arc-length window of this size along the gradient path, the maximum
        # per-point deviation is computed; if ANY window's maximum exceeds
        # path_diff_threshold, a replan is triggered. This catches short but
        # severe detours (e.g. a U-turn around a thicker-than-expected wall)
        # that the global percentile can dilute when the rest of the path is
        # fine. 0.0 disables this check (percentile-only mode).
        self.declare_parameter("path_diff_window_size", 0.0)
        # Action endpoint name.
        self.declare_parameter("action_name", "pmp_planner/plan_to_goal")
        # How long to wait for the server to ACCEPT a sent goal before
        # allowing a retry on the next field update.
        self.declare_parameter("goal_accept_timeout", 2.0)

        self._stamped: bool = self.get_parameter("enable_stamped_cmd_vel").value
        self._enable_recovery: bool = self.get_parameter("enable_recovery").value
        self._robot_frame: str = self.get_parameter("robot_frame").value
        self._planning_frame: str = self.get_parameter("planning_frame").value
        self._default_dt: float = 1.0 / float(
            self.get_parameter("default_tick_rate").value
        )
        self._corridor_epsilon: float = float(
            self.get_parameter("corridor_epsilon").value
        )
        self._recovery_corridor_epsilon: float = float(
            self.get_parameter("recovery_corridor_epsilon").value
        )
        self._recovery_angle_tolerance: float = float(
            self.get_parameter("recovery_angle_tolerance").value
        )
        self._path_diff_threshold: float = float(
            self.get_parameter("path_diff_threshold").value
        )
        self._path_diff_percentile: float = float(
            self.get_parameter("path_diff_percentile").value
        )
        self._replan_cooldown: float = float(
            self.get_parameter("replan_cooldown").value
        )
        self._path_diff_skip_ahead: float = float(
            self.get_parameter("path_diff_skip_ahead").value
        )
        self._path_diff_min_tangent_dot: float = float(
            self.get_parameter("path_diff_min_tangent_dot").value
        )
        self._path_diff_window_size: float = float(
            self.get_parameter("path_diff_window_size").value
        )
        action_name: str = self.get_parameter("action_name").value
        self._goal_accept_timeout: float = float(
            self.get_parameter("goal_accept_timeout").value
        )
        recovery_cfg = RecoveryConfig(
            recovery_corridor_epsilon=self._recovery_corridor_epsilon,
            recovery_angle_tolerance=self._recovery_angle_tolerance,
            look_ahead_distance=float(self.get_parameter("recovery_look_ahead").value),
            v_max=float(self.get_parameter("recovery_v_max").value),
            omega_max=float(self.get_parameter("recovery_omega_max").value),
            K_v=float(self.get_parameter("recovery_K_v").value),
            K_bearing=float(self.get_parameter("recovery_K_bearing").value),
            K_theta=float(self.get_parameter("recovery_K_theta").value),
        )
        self._recovery_strategies = default_strategies(recovery_cfg)

        # ---- TF -------------------------------------------------------
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- Playback state ------------------------------------------
        self._state: _State = _State.IDLE
        # Trajectory currently being played/corrected; -1 means IDLE.
        self._active_traj_id: int = -1
        # chunk_index -> feedback message, for the active trajectory only.
        self._chunks: Dict[int, "PlanToGoal.Feedback"] = {}
        self._cur_chunk_idx: int = 0
        self._cur_sample_idx: int = 0
        self._active_dt: float = self._default_dt
        self._current_traj_start_time: Time = self.get_clock().now()
        self._current_traj_samples_consumed: int = 0
        # Set when the action result for the active trajectory arrives.
        # The tick handler drains any remaining buffer before going IDLE.
        self._result_received: bool = False
        self._result_success: bool = False
        # Cumulative (x, y) of every feedback chunk seen for the active
        # trajectory. Used by _on_gradient_path to compare against the FM2
        # gradient path and detect when a replan is needed.
        self._latest_plan_xy: np.ndarray = np.zeros((0, 2), dtype=np.float32)

        # ---- Pending-send guard --------------------------------------
        # _pending_send: True from send_goal_async until server ACCEPT/REJECT.
        # _goal_in_flight: True from ACCEPT until first feedback chunk, to
        #   suppress spurious IDLE-branch replans during the BVP-solve gap.
        self._pending_send: bool = False
        self._goal_in_flight: bool = False
        self._last_send_time: float = 0.0

        # ---- Goal / field state --------------------------------------
        self._goal_xyth: Optional[np.ndarray] = None  # (gx, gy, gtheta)
        self._latest_goal_handle: Optional[ClientGoalHandle] = None
        self._latest_send_future = None
        # Wall-clock time of the last gradient-path-triggered replan; used
        # by the cooldown guard to prevent the replan loop.
        self._last_replan_wall_time: float = 0.0

        # ---- ROS plumbing --------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, qos)
        self.create_subscription(
            Path, "/vector_field/optimal_path", self._on_gradient_path, qos
        )

        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)

        if self._stamped:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self._marker_pub = self.create_publisher(
            MarkerArray, "/pmp_trajectory_corrector/debug_markers", 10
        )
        self._deviation_pub = self.create_publisher(
            Float64, "/pmp_trajectory_corrector/deviation", 10
        )
        self._state_pub = self.create_publisher(
            String, "/pmp_trajectory_corrector/state", 10
        )

        # depth=64 matches the planner's ActionServer feedback QoS to avoid
        # drops during burst chunk emission (BVP solves complete faster than
        # the playback rate consumes them).
        feedback_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._action_client = ActionClient(
            self,
            PlanToGoal,
            action_name,
            feedback_sub_qos_profile=feedback_qos,
        )

        self._tick_timer = None
        self._ensure_tick_timer(self._default_dt)

        self.get_logger().info(
            f"Trajectory corrector ready (default tick {self._default_dt:.3f} s; "
            f"corridor_epsilon={self._corridor_epsilon:.3f} m, "
            f"recovery_corridor_epsilon={self._recovery_corridor_epsilon:.3f} m; "
            f"action '{action_name}'; replan path_diff "
            f"p{self._path_diff_percentile:.0f}={self._path_diff_threshold:.3f} m)."
        )

    # ----------------------- Goal subscription -----------------------

    def _on_goal(self, msg: PoseStamped):
        if msg.header.frame_id == "":
            self._goal_xyth = None
            return
        pos = msg.pose.position
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._goal_xyth = np.array([pos.x, pos.y, yaw])
        self.get_logger().info(f"New goal: ({pos.x:.2f}, {pos.y:.2f}), yaw={yaw:.2f}")
        # Explicit user goal always sends immediately; bypass both guards.
        self._pending_send = False
        self._goal_in_flight = False
        self._send_action_goal()

    # ----------------------- Gradient path subscription -------------

    def _on_gradient_path(self, msg: Path):
        """Compare the FM2 gradient path against the buffered planned
        trajectory and re-fire the action goal if they have diverged.

        Both paths suffer from message latency: the gradient path is traced
        from the VF node's TF snapshot at publish time, and _latest_plan_xy
        grows from chunks committed before the robot's current position. By
        the time either message is processed here, the robot has moved on.
        Both are therefore trimmed symmetrically: find the nearest point on
        each path to the robot's current pose, then advance both by the same
        arc-length skip (path_diff_skip_ahead) before comparing.

        Matching is direction-aware: a plan point is only accepted as a valid
        nearest-neighbour for a gradient path point if their unit tangents
        satisfy dot >= path_diff_min_tangent_dot. This prevents anti-parallel
        segments (e.g. a U-turn's return leg) from masking genuine deviation
        on the outbound leg. Points with no valid directional match are
        assigned inf distance and always contribute to a replan.

        Two complementary trigger checks are combined with OR:
          - Global percentile: path_diff_percentile-th percentile of FINITE
            per-point deviations exceeds path_diff_threshold. Catches broad,
            distributed divergence across the whole future path.
          - Sliding window max: maximum per-point deviation in ANY arc-length
            window of path_diff_window_size exceeds path_diff_threshold.
            Catches short but severe localised detours (e.g. a U-turn around
            a thicker-than-expected wall) that the global percentile dilutes.
            Disabled when path_diff_window_size == 0.0.

        Two trigger modes:
          - No active trajectory and not waiting for an accept: the arrival
            of a gradient path message implies the field is live; (re)send
            the action goal. This replaces the old field-arrival initial-fire
            logic and also acts as a retry if the previous send was rejected.
          - Active trajectory, plan available, result not yet received: run
            the comparison and replan if either check fires.
        """
        if self._goal_xyth is None:
            return

        # Initial fire / retry: no trajectory running and we are not already
        # waiting on an accept from the server.
        if self._active_traj_id < 0 and not self._goal_in_flight:
            sent = self._send_action_goal()
            if sent:
                self.get_logger().info(
                    "Gradient path received with no active trajectory: "
                    "sent action goal."
                )
            return

        # Path comparison: only when we have something to compare against and
        # the trajectory is still in progress. Also skip when a new goal is
        # already in flight -- the stale _latest_plan_xy from the old
        # trajectory would produce a misleading comparison.
        if (
            self._latest_plan_xy.shape[0] < 2
            or len(msg.poses) < 2
            or self._goal_in_flight
        ):
            return

        # Parse gradient path into (N, 2).
        gpath = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float64,
        )

        # One TF lookup for both trims so they share the same reference pose.
        pose = self._get_current_pose_2d(self._planning_frame)
        if pose is None:
            # Can't determine robot position; skip rather than compare
            # untrimmed paths whose stale heads would bias the result.
            return
        rx, ry, _ = pose

        # Build plan_xy from the CURRENT playback position onward so that
        # argmin cannot reach back into already-consumed samples. Using
        # _latest_plan_xy (which starts from the trajectory's first pose)
        # lets argmin find a past expected position when the robot deviates,
        # causing future_plan to start near the trajectory's origin rather
        # than the robot's actual current location.
        future_path_tuples, _ = self._build_path_polyline()
        if len(future_path_tuples) < 2:
            return
        plan_xy = np.array(
            [[x, y] for x, y, _ in future_path_tuples], dtype=np.float64
        )
        plan_start = int(np.argmin((plan_xy[:, 0] - rx) ** 2 + (plan_xy[:, 1] - ry) ** 2))
        grad_start = int(np.argmin((gpath[:, 0] - rx) ** 2 + (gpath[:, 1] - ry) ** 2))

        if self._path_diff_skip_ahead > 0.0:
            plan_start = _advance_by_arc(plan_xy, plan_start, self._path_diff_skip_ahead)
            grad_start = _advance_by_arc(gpath, grad_start, self._path_diff_skip_ahead)

        future_plan = plan_xy[plan_start:]
        future_grad = gpath[grad_start:]

        if future_plan.shape[0] < 2 or future_grad.shape[0] < 2:
            return

        # The planned trajectory grows incrementally as chunks arrive, so it
        # is almost always shorter than the full-length gradient path. Comparing
        # them at different lengths means gradient path points beyond the known
        # plan have no plan counterpart and always produce inf deviation, firing
        # constant spurious replans at the start of each trajectory. Truncate
        # the gradient path to the arc-length of the known future plan so both
        # cover the same spatial extent before comparison.
        plan_arc = float(np.sum(np.hypot(np.diff(future_plan[:, 0]), np.diff(future_plan[:, 1]))))
        acc = 0.0
        grad_end = len(future_grad) - 1
        for i in range(len(future_grad) - 1):
            acc += float(np.hypot(future_grad[i + 1, 0] - future_grad[i, 0],
                                  future_grad[i + 1, 1] - future_grad[i, 1]))
            if acc >= plan_arc:
                grad_end = i + 1
                break
        future_grad = future_grad[:grad_end + 1]

        if future_grad.shape[0] < 2:
            return

        # Squared-distance matrix: (N, M) where N = gradient points, M = plan points.
        diff = future_grad[:, np.newaxis, :] - future_plan[np.newaxis, :, :]
        sq_dist = (diff ** 2).sum(axis=2)   # (N, M)

        if self._path_diff_min_tangent_dot > -1.0:
            # Direction-aware matching: reject plan points whose travel
            # direction is more than acos(min_tangent_dot) from the gradient
            # direction at the query point. This prevents a U-turn's return
            # leg from absorbing gradient path points on the outbound leg and
            # reporting a spuriously small distance.
            grad_tan = _path_tangents(future_grad)   # (N, 2)
            plan_tan = _path_tangents(future_plan)   # (M, 2)
            dot = grad_tan @ plan_tan.T              # (N, M)
            invalid = dot < self._path_diff_min_tangent_dot
            sq_dist = np.where(invalid, np.inf, sq_dist)

        min_sq = sq_dist.min(axis=1)        # (N,)
        min_dist = np.where(np.isfinite(min_sq), np.sqrt(min_sq), np.inf)

        # --- Check 1: global percentile over finite deviations ---
        # inf values (no valid directional match) are excluded from the
        # percentile; the windowed check handles them.
        finite_mask = np.isfinite(min_dist)
        pct_val = (
            float(np.percentile(min_dist[finite_mask], self._path_diff_percentile))
            if finite_mask.any()
            else 0.0
        )
        pct_triggered = pct_val > self._path_diff_threshold

        # --- Check 2: sliding window maximum ---
        # Catches short but severe detours that the global percentile dilutes.
        # inf values propagate naturally through np.max, so a window of
        # topologically-unmatched points (e.g. the apex of a U-turn the plan
        # has no segment for) always fires.
        win_val = 0.0
        win_triggered = False
        if self._path_diff_window_size > 0.0:
            segs = np.hypot(np.diff(future_grad[:, 0]), np.diff(future_grad[:, 1]))
            arc_s = np.concatenate([[0.0], np.cumsum(segs)])
            win_val = _windowed_max_deviation(min_dist, arc_s, self._path_diff_window_size)
            win_triggered = win_val > self._path_diff_threshold

        should_replan = pct_triggered or win_triggered

        if should_replan and self._replan_cooldown > 0.0:
            now = time.monotonic()
            if now - self._last_replan_wall_time < self._replan_cooldown:
                should_replan = False
            else:
                self._last_replan_wall_time = now

        if should_replan:
            sent = self._send_action_goal()
            if sent:
                reason = (
                    f"p{self._path_diff_percentile:.0f}={pct_val:.3f} m"
                    if pct_triggered
                    else f"window({self._path_diff_window_size:.1f} m) max={win_val:.3f} m"
                )
                self.get_logger().info(
                    f"Gradient path diverged ({reason} > "
                    f"{self._path_diff_threshold:.3f} m): replanning."
                )

    # ----------------------- Action client ---------------------------

    def _send_action_goal(self) -> bool:
        """Snapshot chassis pose from TF and send a PlanToGoal action goal.
        Returns True if sent, False if blocked (pending, server not ready, TF failure).
        """
        if self._goal_xyth is None:
            return False

        if self._pending_send:
            elapsed = time.monotonic() - self._last_send_time
            if elapsed < self._goal_accept_timeout:
                return False
            self.get_logger().warn(
                f"Goal accept pending for {elapsed:.1f}s "
                f"(> {self._goal_accept_timeout:.1f}s timeout); "
                "retrying with current pose."
            )

        if not self._action_client.server_is_ready():
            self.get_logger().warn(
                "Planner action server not yet discovered; will retry on next event.",
                throttle_duration_sec=5.0,
            )
            return False

        try:
            t = self._tf_buffer.lookup_transform(
                self._planning_frame,
                self._robot_frame,
                Time(),
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self._planning_frame}->{self._robot_frame} unavailable; "
                f"cannot send action goal: {e}",
                throttle_duration_sec=2.0,
            )
            return False

        tx = t.transform.translation.x
        ty = t.transform.translation.y
        q = t.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        goal_msg = PlanToGoal.Goal()
        goal_msg.frame_id = self._planning_frame
        goal_msg.start_x = float(tx)
        goal_msg.start_y = float(ty)
        goal_msg.start_theta = float(yaw)
        goal_msg.target_x = float(self._goal_xyth[0])
        goal_msg.target_y = float(self._goal_xyth[1])
        goal_msg.target_theta = float(self._goal_xyth[2])

        send_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._on_action_feedback,
        )
        self._latest_send_future = send_future
        self._pending_send = True
        self._last_send_time = time.monotonic()
        send_future.add_done_callback(
            lambda f, mine=send_future: self._on_goal_accepted(f, mine)
        )
        self.get_logger().info(
            f"Sent action goal: start=({tx:.2f}, {ty:.2f}) yaw={yaw:.2f} "
            f"-> target=({goal_msg.target_x:.2f}, {goal_msg.target_y:.2f}) "
            f"yaw={goal_msg.target_theta:.2f}"
        )
        return True

    def _on_goal_accepted(self, future, originating_future):
        if originating_future is not self._latest_send_future:
            return
        self._pending_send = False
        gh: ClientGoalHandle = future.result()
        if not gh.accepted:
            self.get_logger().warn("Action goal rejected by planner.")
            return
        self._goal_in_flight = True
        self._latest_goal_handle = gh
        gh.get_result_async().add_done_callback(
            lambda f, _gh=gh: self._on_action_result(f, _gh)
        )

    def _on_action_feedback(self, fb_msg):
        """One trajectory chunk arrived from the action server."""
        chunk = fb_msg.feedback
        traj_id = int(chunk.trajectory_id)

        if traj_id > self._active_traj_id:
            self.get_logger().info(
                f"Switching to trajectory {traj_id} (was {self._active_traj_id})."
            )
            self._goal_in_flight = False
            self._active_traj_id = traj_id
            self._state = _State.PLAYING
            self._chunks = {}
            self._cur_chunk_idx = 0
            self._cur_sample_idx = 0
            self._result_received = False
            self._result_success = False
            self._latest_plan_xy = np.zeros((0, 2), dtype=np.float32)
            self._current_traj_start_time = self.get_clock().now()
            self._current_traj_samples_consumed = 0
            if chunk.dt > 0.0 and abs(chunk.dt - self._active_dt) > 1e-6:
                self._active_dt = float(chunk.dt)
                self._ensure_tick_timer(self._active_dt)
        elif traj_id < self._active_traj_id:
            return

        self._chunks[int(chunk.chunk_index)] = chunk

        if len(chunk.pose_x) > 0:
            new_xy = np.column_stack(
                [
                    np.asarray(chunk.pose_x, dtype=np.float32),
                    np.asarray(chunk.pose_y, dtype=np.float32),
                ]
            )
            self._latest_plan_xy = np.concatenate(
                [self._latest_plan_xy, new_xy], axis=0
            )

    def _on_action_result(self, future, expected_gh: ClientGoalHandle):
        wrapped = future.result()
        result = wrapped.result
        status = wrapped.status
        traj_id = int(result.trajectory_id)

        status_name = {4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}.get(
            status, str(status)
        )
        self.get_logger().info(
            f"Action result for traj_id={traj_id}: {status_name} -- "
            f"{result.message!r}"
        )

        if traj_id == 0:
            # Planner aborted before any rollout (e.g. field not yet available).
            # Clear _goal_in_flight so the next field update can retry.
            if expected_gh is self._latest_goal_handle:
                self._goal_in_flight = False
            return

        if traj_id == self._active_traj_id:
            self._result_received = True
            self._result_success = bool(result.success)
            # Let the tick handler drain whatever chunks are already buffered
            # before transitioning to IDLE. On abort this keeps the chassis
            # moving along the last known plan until either the buffer runs out
            # or a new plan's feedback atomically supersedes this trajectory.
            return

        # No feedback was ever received for this goal (chassis already at
        # goal, or aborted before first chunk). Clear _goal_in_flight only
        # if this is still the latest accepted goal.
        if self._active_traj_id < 0:
            if expected_gh is self._latest_goal_handle:
                self._goal_in_flight = False
            if bool(result.success):
                self.get_logger().info(
                    f"Trajectory {traj_id} succeeded with no rollout "
                    "(chassis already at goal)."
                )
                self._goal_xyth = None
                self._publish_cleared_goal_sentinel()

    # ----------------------- Tick handler ----------------------------------

    def _ensure_tick_timer(self, dt: float):
        """(Re)create the periodic publish timer with the given period."""
        if self._tick_timer is not None:
            self.destroy_timer(self._tick_timer)
        self._tick_timer = self.create_timer(dt, self._on_tick)

    def _publish_metrics(self, deviation: float) -> None:
        self._deviation_pub.publish(Float64(data=deviation))
        self._state_pub.publish(String(data=self._state.name))

    def _on_tick(self):
        if self._state == _State.IDLE:
            self._publish_metrics(0.0)
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers([], self._planning_frame, None, None)
            return

        if self._state == _State.CORRECTING:
            self._on_correcting_tick()
            return

        # PLAYING ------------------------------------------------------------

        cur = self._chunks.get(self._cur_chunk_idx)
        if cur is None:
            if self._result_received and self._chunks_exhausted():
                self._finish_trajectory()
                return
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self.get_logger().warn(
                f"Chunk {self._cur_chunk_idx} of trajectory "
                f"{self._active_traj_id} not yet received; holding.",
                throttle_duration_sec=1.0,
            )
            self._current_traj_start_time = self.get_clock().now()
            self._current_traj_samples_consumed = 0
            self._publish_debug_markers([], self._planning_frame, None, None)
            return

        n = len(cur.linear_x)

        # Build path polyline and check corridor once per tick.
        path, mapping = self._build_path_polyline()
        frame = self._planning_frame
        if path:
            pose = self._get_current_pose_2d(frame)
            if pose is not None:
                rx, ry, _ = pose
                proj_x, proj_y, _, _, _ = nearest_projection_on_path(rx, ry, path)
                dist = math.hypot(rx - proj_x, ry - proj_y)
                self._publish_metrics(dist)
                if dist > self._corridor_epsilon:
                    if not self._enable_recovery:
                        self.get_logger().warn(
                            f"Corridor deviation {dist:.3f} m > "
                            f"{self._corridor_epsilon:.3f} m; "
                            f"recovery disabled, continuing playback.",
                            throttle_duration_sec=1.0,
                        )
                    else:
                        self.get_logger().warn(
                            f"Corridor deviation {dist:.3f} m > "
                            f"{self._corridor_epsilon:.3f} m; "
                            f"entering recovery."
                        )
                        self._goto_correcting(dist)
                        self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
                        self._publish_debug_markers(path, frame, None, None)
                        return

        # Consume one sample.
        if self._cur_sample_idx < n:
            v = float(cur.linear_x[self._cur_sample_idx])
            w = float(cur.angular_z[self._cur_sample_idx])
            due = self._current_traj_start_time + Duration(
                nanoseconds=int(
                    self._current_traj_samples_consumed * self._active_dt * 1e9
                )
            )
            self._publish_twist(v, w, due.to_msg())
            self._publish_debug_markers(path, frame, None, None)
            self._current_traj_samples_consumed += 1
            self._cur_sample_idx += 1
            return

        # Chunk exhausted. Drop it from the buffer and advance.
        del self._chunks[self._cur_chunk_idx]
        self._cur_chunk_idx += 1
        self._cur_sample_idx = 0

        if self._result_received and self._chunks_exhausted():
            self._finish_trajectory()
            return

        # Try to consume the first sample of the next chunk on this same tick.
        nxt = self._chunks.get(self._cur_chunk_idx)
        if nxt is not None and len(nxt.linear_x) > 0:
            v = float(nxt.linear_x[0])
            w = float(nxt.angular_z[0])
            due = self._current_traj_start_time + Duration(
                nanoseconds=int(
                    self._current_traj_samples_consumed * self._active_dt * 1e9
                )
            )
            self._publish_twist(v, w, due.to_msg())
            self._publish_debug_markers(path, frame, None, None)
            self._current_traj_samples_consumed += 1
            self._cur_sample_idx = 1
        else:
            self.get_logger().warn(
                f"Chunk {self._cur_chunk_idx} (which is meant to be received after "
                f"the current one) of trajectory {self._active_traj_id} not yet "
                f"received; holding.",
                throttle_duration_sec=1.0,
            )
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers(path, frame, None, None)
            self._current_traj_start_time = self.get_clock().now()
            self._current_traj_samples_consumed = 0

    def _chunks_exhausted(self) -> bool:
        return not self._chunks

    def _finish_trajectory(self):
        """Transition to IDLE. On success, clear the goal and signal completion."""
        success = self._result_success
        traj_id = self._active_traj_id

        self._state = _State.IDLE
        self._active_traj_id = -1
        self._goal_in_flight = False
        self._chunks.clear()
        self._cur_chunk_idx = 0
        self._cur_sample_idx = 0
        self._result_received = False
        self._result_success = False
        self._latest_plan_xy = np.zeros((0, 2), dtype=np.float32)
        self._current_traj_samples_consumed = 0
        self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())

        if success:
            self._goal_xyth = None
            self._publish_cleared_goal_sentinel()
            self.get_logger().info(
                f"Trajectory {traj_id} complete (success). Returning to IDLE."
            )
        else:
            self.get_logger().info(
                f"Trajectory {traj_id} ended without success. Returning to IDLE."
            )

    def _goto_correcting(self, dist: float):
        self._state = _State.CORRECTING
        self.get_logger().warn(
            f"Trajectory {self._active_traj_id} at chunk {self._cur_chunk_idx}, "
            f"sample {self._cur_sample_idx}: corridor deviation {dist:.3f} m "
            f"(limit {self._corridor_epsilon:.3f} m). Suspending playback."
        )

    def _on_correcting_tick(self):
        path, mapping = self._build_path_polyline()
        frame = self._planning_frame

        if not path:
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers([], frame, None, None)
            return

        pose = self._get_current_pose_2d(frame)
        if pose is None:
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers(path, frame, None, None)
            return

        rx, ry, rtheta = pose

        fx, fy, ftheta = path[-1]
        dx, dy = rx - fx, ry - fy

        if math.hypot(dx, dy) < self._recovery_corridor_epsilon:
            self.get_logger().info(
                f"Within {self._recovery_corridor_epsilon:.3f} m of trajectory end "
                f"during recovery. Returning to IDLE."
            )
            self._finish_trajectory()
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers([], frame, None, None)
            return

        overshot = dx * math.cos(ftheta) + dy * math.sin(ftheta) > 0
        if overshot:
            if self._result_received:
                # Result is in -- this was the complete trajectory.
                self.get_logger().info(
                    "Overshot trajectory end during recovery. Returning to IDLE."
                )
                self._finish_trajectory()
                self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
                self._publish_debug_markers([], frame, None, None)
            else:
                # More chunks are expected; hold until the path extends.
                self.get_logger().info(
                    "Overshot end of buffered path during recovery; "
                    "waiting for next chunk.",
                    throttle_duration_sec=1.0,
                )
                self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
                self._publish_debug_markers(path, frame, None, None)
            return

        proj_x, proj_y, proj_theta, seg_idx, t = nearest_projection_on_path(
            rx, ry, path
        )
        perp_dist = math.hypot(rx - proj_x, ry - proj_y)
        self._publish_metrics(perp_dist)
        angle_err = abs(math.remainder(rtheta - proj_theta, 2 * math.pi))

        lx, ly, _ = walk_ahead_on_path(
            path, seg_idx, t, self._recovery_strategies[0]._cfg.look_ahead_distance
        )

        exiting = (
            perp_dist < self._recovery_corridor_epsilon
            and angle_err < self._recovery_angle_tolerance
        )
        self.get_logger().info(
            f"Correcting: perp={perp_dist:.3f} m  "
            f"angle={math.degrees(angle_err):.1f} deg  "
            f"{'[acceptable -- resuming]' if exiting else '[recovering]'}",
            throttle_duration_sec=0.5,
        )

        if exiting:
            snap_idx = seg_idx if t <= 0.5 else min(seg_idx + 1, len(mapping) - 1)
            chunk_idx, sample_idx = mapping[snap_idx]
            self.get_logger().info(
                f"Pose recovered (perp={perp_dist:.3f} m, "
                f"angle={math.degrees(angle_err):.1f} deg). "
                f"Resuming at chunk {chunk_idx}, sample {sample_idx}."
            )
            self._cur_chunk_idx = chunk_idx
            self._cur_sample_idx = sample_idx
            self._current_traj_start_time = self.get_clock().now()
            self._current_traj_samples_consumed = 0
            self._state = _State.PLAYING
            self._publish_debug_markers(path, frame, (proj_x, proj_y), (lx, ly))
            return

        for strategy in self._recovery_strategies:
            if strategy.can_handle(pose, path):
                v, omega = strategy.compute_twist(pose, path)
                self.get_logger().info(
                    f"{strategy.__class__.__name__}: "
                    f"v={v:.3f} m/s  omega={omega:.3f} rad/s",
                    throttle_duration_sec=0.5,
                )
                self._publish_twist(v, omega, self.get_clock().now().to_msg())
                self._publish_debug_markers(path, frame, (proj_x, proj_y), (lx, ly))
                return

        self.get_logger().warn(
            "No recovery strategy matched -- holding.",
            throttle_duration_sec=1.0,
        )
        self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
        self._publish_debug_markers(path, frame, (proj_x, proj_y), (lx, ly))

    # ----------------------- Path helpers ----------------------------------

    def _build_path_polyline(
        self,
    ) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
        """Build an ordered path polyline from all buffered chunks.

        Returns:
          path    -- list of (x, y, theta) poses in trajectory order, starting
                     from the current playback position.
          mapping -- parallel list of (chunk_idx, sample_idx) for each point,
                     used to snap playback position after recovery.
        """
        path: list[tuple[float, float, float]] = []
        mapping: list[tuple[int, int]] = []
        for chunk_idx in sorted(self._chunks):
            chunk = self._chunks[chunk_idx]
            start = self._cur_sample_idx if chunk_idx == self._cur_chunk_idx else 0
            for sample_idx in range(start, len(chunk.pose_x)):
                path.append(
                    (
                        float(chunk.pose_x[sample_idx]),
                        float(chunk.pose_y[sample_idx]),
                        float(chunk.pose_theta[sample_idx]),
                    )
                )
                mapping.append((chunk_idx, sample_idx))
        return path, mapping

    # ----------------------- Pose lookup -----------------------------------

    def _get_current_pose_2d(self, frame: str) -> Optional[tuple[float, float, float]]:
        """Return (x, y, theta) of robot_frame in frame, or None on TF failure."""
        try:
            t = self._tf_buffer.lookup_transform(frame, self._robot_frame, Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            theta = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            return x, y, theta
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF lookup {self._robot_frame} -> {frame} failed: {e}",
                throttle_duration_sec=1.0,
            )
            return None

    # ----------------------- Visualization ---------------------------------

    def _publish_debug_markers(
        self,
        path: list[tuple[float, float, float]],
        frame: str,
        proj: Optional[tuple[float, float]],
        carrot: Optional[tuple[float, float]],
    ) -> None:
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()

        def _make(
            mid: int, mtype: int, action: int = Marker.ADD, namespace: str = _MARKER_NS
        ) -> Marker:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame
            m.ns = namespace
            m.id = mid
            m.type = mtype
            m.action = action
            m.lifetime = _MARKER_LIFETIME
            m.pose.orientation.w = 1.0
            return m

        # -- 0: path centerline (thin blue LINE_STRIP) --
        if path:
            m = _make(0, Marker.LINE_STRIP, namespace="centerline")
            m.scale.x = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 1.0
            m.points = [Point(x=x, y=y, z=0.0) for x, y, _ in path]
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(0, Marker.LINE_STRIP, Marker.DELETE, namespace="centerline")
            )

        # -- 1: corridor tube (thick semi-transparent LINE_STRIP) --
        if path:
            m = _make(1, Marker.LINE_STRIP, namespace="corridor")
            m.scale.x = 2.0 * self._corridor_epsilon
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 0.2
            m.points = [Point(x=x, y=y, z=0.0) for x, y, _ in path]
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(1, Marker.LINE_STRIP, Marker.DELETE, namespace="corridor")
            )

        # -- 2: nearest projection point (yellow sphere) --
        if proj is not None:
            m = _make(2, Marker.SPHERE, namespace="projection")
            m.pose.position.x, m.pose.position.y = proj[0], proj[1]
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 1.0
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(2, Marker.SPHERE, Marker.DELETE, namespace="projection")
            )

        # -- 3: look-ahead carrot (green sphere) --
        if carrot is not None:
            m = _make(3, Marker.SPHERE, namespace="carrot")
            m.pose.position.x, m.pose.position.y = carrot[0], carrot[1]
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(3, Marker.SPHERE, Marker.DELETE, namespace="carrot")
            )

        # -- 4: look-ahead arrow (projection → carrot) --
        if proj is not None and carrot is not None:
            m = _make(4, Marker.ARROW, namespace="lookahead")
            m.points = [
                Point(x=proj[0], y=proj[1], z=0.0),
                Point(x=carrot[0], y=carrot[1], z=0.0),
            ]
            m.scale.x = 0.04
            m.scale.y = 0.08
            m.scale.z = 0.10
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 0.4, 1.0
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(4, Marker.ARROW, Marker.DELETE, namespace="lookahead")
            )

        # -- 5: state text above robot (colour-coded) --
        robot_pose = self._get_current_pose_2d(frame)
        m = _make(5, Marker.TEXT_VIEW_FACING, namespace="state")
        if robot_pose is not None:
            m.pose.position.x = robot_pose[0]
            m.pose.position.y = robot_pose[1]
            m.pose.position.z = 0.5
        m.scale.z = 0.3
        if self._state == _State.IDLE:
            m.text = "IDLE"
            m.color.r, m.color.g, m.color.b, m.color.a = 0.7, 0.7, 0.7, 1.0
        elif self._state == _State.PLAYING:
            m.text = "PLAY"
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 1.0, 0.2, 1.0
        else:
            m.text = "FIX"
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
        markers.markers.append(m)

        self._marker_pub.publish(markers)

    # ----------------------- Publishing ------------------------------------

    def _publish_cleared_goal_sentinel(self):
        sentinel = PoseStamped()
        sentinel.header.stamp = self.get_clock().now().to_msg()
        sentinel.header.frame_id = ""
        self._goal_pub.publish(sentinel)

    def _publish_twist(self, v: float, omega: float, stamp):
        if self._stamped:
            msg = TwistStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self._robot_frame
            msg.twist.linear.x = v
            msg.twist.angular.z = omega
        else:
            msg = Twist()
            msg.linear.x = v
            msg.angular.z = omega
        self._cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryCorrectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
