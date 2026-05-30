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
to CORRECTING. The old trajectory's remaining samples are discarded -- the new
trajectory was generated from the chassis's TF pose at send time, which may
already be slightly stale by the time the first chunk arrives. CORRECTING
aligns the robot to the nearest point on the new trajectory before resuming
PLAYING, preventing the robot from executing commands intended for an old start
position.

Out-of-order chunk arrival within a trajectory_id: chunks are stored in
a dict keyed by chunk_index and consumed in index order. If chunk N+1
arrives before N has been fully played, that's fine; we simply pull it
when needed. If chunk N is missing when we need it, we publish zero
twist and wait. This shouldn't happen under reliable QoS but is the
graceful fallback.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import rclpy

import numpy as np

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
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
from visualization_msgs.msg import MarkerArray

try:
    from scout_msgs.msg import ScoutStatus

    _SCOUT_MSGS_AVAILABLE = True
except ImportError:
    ScoutStatus = None
    _SCOUT_MSGS_AVAILABLE = False

from agx_planning_msgs.action import PlanToGoal
from agx_planning.utils import declare_and_load_dataclass
from agx_planning.runtime_corrector import RecoveryConfig, default_strategies
from agx_planning.runtime_corrector.config import CorrectorConfig
from agx_planning.runtime_corrector.geometry import advance_by_arc
from agx_planning.runtime_corrector.deviation_detector import DeviationDetector
from agx_planning.runtime_corrector.correction_controller import (
    CorrectionController,
    ExitKind,
)
from agx_planning.runtime_corrector.visualization import TrajectoryVisualizer
from agx_planning.runtime_corrector.trajectory_buffer import TrajectoryBuffer
from agx_planning.runtime_corrector.strategies import State

# ---------------------------------------------------------------------------
# Corrector node
# ---------------------------------------------------------------------------


@dataclass
class CorrectorNodeConfig:
    """ROS2 infrastructure parameters for TrajectoryCorrectorNode.

    Pure algorithm parameters live in CorrectorConfig (config.py).
    """

    robot_frame: str = "base_link"
    planning_frame: str = "map"
    enable_stamped_cmd_vel: bool = False
    # Tick-rate fallback used only before the first chunk arrives; after
    # that we use chunk.dt (whatever rate the planner committed to).
    default_tick_rate: float = 10.0
    enable_recovery: bool = True
    # Topic for chassis control-mode status (scout_msgs/ScoutStatus).
    # Set to "" to disable RC-override detection (e.g. in simulation).
    scout_status_topic: str = "/scout_status"
    # If no ScoutStatus arrives for this many seconds while RC is active,
    # assume control has returned (covers driver restart / simulator).
    scout_status_timeout: float = 1.0
    # How long after a replan to suppress further gradient-path replans.
    replan_cooldown: float = 0.0
    # Arc-length [m] to skip forward along both paths before comparing.
    # Avoids transient disagreement at the robot's feet due to message latency.
    path_diff_skip_ahead: float = 0.0
    action_name: str = "pmp_planner/plan_to_goal"
    # How long to wait for the server to ACCEPT a sent goal before retrying.
    goal_accept_timeout: float = 2.0


class TrajectoryCorrectorNode(Node):

    @property
    def _default_dt(self) -> float:
        return 1.0 / self.node_cfg.default_tick_rate

    def __init__(self):
        super().__init__("pmp_trajectory_corrector")

        self.cfg: CorrectorConfig = declare_and_load_dataclass(self, CorrectorConfig())
        self.node_cfg: CorrectorNodeConfig = declare_and_load_dataclass(
            self, CorrectorNodeConfig()
        )
        recovery_cfg = RecoveryConfig(
            recovery_corridor_epsilon=self.cfg.recovery_corridor_epsilon,
            recovery_angle_tolerance=self.cfg.recovery_angle_tolerance,
            look_ahead_distance=self.cfg.recovery_look_ahead,
            v_max=self.cfg.recovery_v_max,
            omega_max=self.cfg.recovery_omega_max,
            K_v=self.cfg.recovery_K_v,
            K_bearing=self.cfg.recovery_K_bearing,
            K_theta=self.cfg.recovery_K_theta,
            max_pursuit_bearing_err=self.cfg.recovery_max_pursuit_bearing_err,
            near_endpoint_distance=self.cfg.recovery_near_endpoint_distance,
        )
        self._recovery_strategies = default_strategies(recovery_cfg)
        self._detector = DeviationDetector(self.cfg)
        self._controller = CorrectionController(self.cfg, self._recovery_strategies)

        # ---- TF -------------------------------------------------------
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- Control-override tracking -------------------------------
        # True whenever we own /cmd_vel (chassis not in RC mode).
        # Starts True so simulation (no scout_status) runs unimpeded.
        self._has_control: bool = True
        # Monotonic timestamp of the last ScoutStatus message; -inf means
        # none received yet, which keeps _has_control True via the timeout
        # path in _refresh_has_control (only activates when already False).
        self._last_status_time: float = -math.inf

        # ---- Playback state ------------------------------------------
        self._state: State = State.IDLE
        self._buf = TrajectoryBuffer(self._default_dt)
        # ROS2 timestamp of when playback timing started; reset on hold/snap.
        self._current_traj_start_time: Time = self.get_clock().now()

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
        if self.node_cfg.scout_status_topic:
            if _SCOUT_MSGS_AVAILABLE:
                self.create_subscription(
                    ScoutStatus,
                    self.node_cfg.scout_status_topic,
                    self._on_scout_status,
                    qos,
                )
            else:
                self.get_logger().warn(
                    "scout_msgs not installed; RC-override detection disabled. "
                    "Trajectory will play back regardless of control mode."
                )

        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)

        if self.node_cfg.enable_stamped_cmd_vel:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self._marker_pub = self.create_publisher(
            MarkerArray, "/pmp_trajectory_corrector/debug_markers", 10
        )
        self._visualizer = TrajectoryVisualizer(self._marker_pub, self.cfg)
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
            self.node_cfg.action_name,
            feedback_sub_qos_profile=feedback_qos,
        )

        self._tick_timer = None
        self._ensure_tick_timer(self._default_dt)

        self.get_logger().info(
            f"Trajectory corrector ready (default tick {self._default_dt:.3f} s; "
            f"corridor_epsilon={self.cfg.corridor_epsilon:.3f} m, "
            f"recovery_corridor_epsilon={self.cfg.recovery_corridor_epsilon:.3f} m; "
            f"action '{self.node_cfg.action_name}'; replan path_diff "
            f"p{self.cfg.path_diff_percentile:.0f}={self.cfg.path_diff_threshold:.3f} m)."
        )

    # ----------------------- RC-override tracking --------------------

    def _on_scout_status(self, msg) -> None:
        """Detect transitions into/out of RC mode (control_mode == 3)."""
        in_rc = msg.control_mode == 3  # AgxControlMode::CONTROL_MODE_RC
        had_control = self._has_control
        self._last_status_time = time.monotonic()
        self._has_control = not in_rc
        if had_control and not self._has_control:
            self.get_logger().warn(
                "RC override active: suspending trajectory playback."
            )
        elif not had_control and self._has_control:
            self._on_control_regained()

    def _refresh_has_control(self) -> None:
        """Time-out guard: if no ScoutStatus has arrived while RC was active,
        assume control has returned (handles driver restart and simulation)."""
        if self._has_control:
            return
        elapsed = time.monotonic() - self._last_status_time
        if elapsed > self.node_cfg.scout_status_timeout:
            self.get_logger().warn(
                f"No scout_status for {elapsed:.1f} s while RC was active; "
                "assuming control has returned."
            )
            self._has_control = True
            self._on_control_regained()

    def _on_control_regained(self) -> None:
        if self._state == State.PLAYING:
            self.get_logger().info(
                "Control regained; entering alignment correction before resuming."
            )
            self._state = State.CORRECTING

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
        from the VF node's TF snapshot at publish time, and the buffered plan
        was committed before the robot's current position. By the time either
        message is processed here, the robot has moved on.
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
          - Active trajectory, plan available: run the comparison and replan
            if either check fires.
        """
        if self._goal_xyth is None:
            return

        # Initial fire / retry: no trajectory running and we are not already
        # waiting on an accept from the server.
        if self._buf.active_traj_id < 0 and not self._goal_in_flight:
            sent = self._send_action_goal()
            if sent:
                self.get_logger().info(
                    "Gradient path received with no active trajectory: "
                    "sent action goal."
                )
            return

        # Path comparison: only when we have something to compare against and
        # the trajectory is still in progress. Also skip when a new goal is
        # already in flight -- the buffered plan is stale and would produce a
        # misleading comparison.
        if len(msg.poses) < 2 or self._goal_in_flight:
            return

        # Wrong-endpoint guard: when the planner has reported success the
        # remaining buffer shrinks to near-zero arc length, making the spatial
        # comparison below ineffective (gradient truncated to tiny window,
        # locally agrees with plan). Directly check the last buffered pose
        # against the true goal instead.
        if (
            self._buf.result_received
            and self._buf.result_success
            and self._goal_xyth is not None
        ):
            ep = self._buf.last_endpoint()
            if ep is not None:
                ep_dist = math.hypot(
                    ep[0] - self._goal_xyth[0],
                    ep[1] - self._goal_xyth[1],
                )
                if ep_dist > self.cfg.path_diff_threshold:
                    sent = self._send_action_goal()
                    if sent:
                        self.get_logger().info(
                            f"Plan endpoint is {ep_dist:.3f} m from goal "
                            f"(>{self.cfg.path_diff_threshold:.3f} m) with result "
                            "received; replanning."
                        )
                    return

        # Parse gradient path into (N, 2).
        gpath = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float64,
        )

        # One TF lookup for both trims so they share the same reference pose.
        pose = self._get_current_pose_2d(self.node_cfg.planning_frame)
        if pose is None:
            # Can't determine robot position; skip rather than compare
            # untrimmed paths whose stale heads would bias the result.
            return
        rx, ry, _ = pose

        # Build plan_xy from the CURRENT playback position onward so that
        # argmin cannot reach back into already-consumed samples.
        future_path_tuples, _ = self._buf.build_polyline()
        if len(future_path_tuples) < 2:
            return
        plan_xy = np.array([[x, y] for x, y, _ in future_path_tuples], dtype=np.float64)
        plan_start = int(
            np.argmin((plan_xy[:, 0] - rx) ** 2 + (plan_xy[:, 1] - ry) ** 2)
        )
        grad_start = int(np.argmin((gpath[:, 0] - rx) ** 2 + (gpath[:, 1] - ry) ** 2))

        if self.node_cfg.path_diff_skip_ahead > 0.0:
            plan_start = advance_by_arc(
                plan_xy, plan_start, self.node_cfg.path_diff_skip_ahead
            )
            grad_start = advance_by_arc(
                gpath, grad_start, self.node_cfg.path_diff_skip_ahead
            )

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
        plan_arc = float(
            np.sum(np.hypot(np.diff(future_plan[:, 0]), np.diff(future_plan[:, 1])))
        )
        acc = 0.0
        grad_end = len(future_grad) - 1
        for i in range(len(future_grad) - 1):
            acc += float(
                np.hypot(
                    future_grad[i + 1, 0] - future_grad[i, 0],
                    future_grad[i + 1, 1] - future_grad[i, 1],
                )
            )
            if acc >= plan_arc:
                grad_end = i + 1
                break
        future_grad = future_grad[: grad_end + 1]

        if future_grad.shape[0] < 2:
            return

        should_replan, pct_val, win_val = self._detector.check_gradient_path(
            future_grad, future_plan
        )
        pct_triggered = pct_val > self.cfg.path_diff_threshold

        if should_replan and self.node_cfg.replan_cooldown > 0.0:
            now = time.monotonic()
            if now - self._last_replan_wall_time < self.node_cfg.replan_cooldown:
                should_replan = False
            else:
                self._last_replan_wall_time = now

        if should_replan:
            sent = self._send_action_goal()
            if sent:
                reason = (
                    f"p{self.cfg.path_diff_percentile:.0f}={pct_val:.3f} m"
                    if pct_triggered
                    else f"window({self.cfg.path_diff_window_size:.1f} m) max={win_val:.3f} m"
                )
                self.get_logger().info(
                    f"Gradient path diverged ({reason} > "
                    f"{self.cfg.path_diff_threshold:.3f} m): replanning."
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
            if elapsed < self.node_cfg.goal_accept_timeout:
                return False
            self.get_logger().warn(
                f"Goal accept pending for {elapsed:.1f}s "
                f"(> {self.node_cfg.goal_accept_timeout:.1f}s timeout); "
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
                self.node_cfg.planning_frame,
                self.node_cfg.robot_frame,
                Time(),
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self.node_cfg.planning_frame}->{self.node_cfg.robot_frame} unavailable; "
                f"cannot send action goal: {e}",
                throttle_duration_sec=2.0,
            )
            return False

        tx = t.transform.translation.x
        ty = t.transform.translation.y
        q = t.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        goal_msg = PlanToGoal.Goal()
        goal_msg.frame_id = self.node_cfg.planning_frame
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

        if traj_id > self._buf.active_traj_id:
            self.get_logger().info(
                f"Switching to trajectory {traj_id} (was {self._buf.active_traj_id}): "
                "entering alignment correction before playback."
            )
            self._goal_in_flight = False
            old_dt = self._buf.active_dt
            new_dt = float(chunk.dt) if chunk.dt > 0.0 else old_dt
            self._buf.reset(traj_id, new_dt)
            self._state = (
                State.CORRECTING if self.node_cfg.enable_recovery else State.PLAYING
            )
            self._current_traj_start_time = self.get_clock().now()
            if abs(new_dt - old_dt) > 1e-6:
                self._ensure_tick_timer(new_dt)
        elif traj_id < self._buf.active_traj_id:
            return

        self._buf.add_chunk(int(chunk.chunk_index), chunk)

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

        if traj_id == self._buf.active_traj_id:
            self._buf.mark_result(bool(result.success))
            # Let the tick handler drain whatever chunks are already buffered
            # before transitioning to IDLE. On abort this keeps the chassis
            # moving along the last known plan until either the buffer runs out
            # or a new plan's feedback atomically supersedes this trajectory.
            return

        # No feedback was ever received for this goal (chassis already at
        # goal, or aborted before first chunk). Clear _goal_in_flight only
        # if this is still the latest accepted goal.
        if self._buf.active_traj_id < 0:
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
        if self._state == State.IDLE:
            self._publish_metrics(0.0)
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers([], self.node_cfg.planning_frame, None, None)
            return

        self._refresh_has_control()
        if not self._has_control:
            # RC is driving; hold the buffer cursor and send zero (ignored by HW).
            self._publish_metrics(0.0)
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self.get_logger().warn(
                "RC override active: suspending trajectory playback.",
                throttle_duration_sec=5.0,
            )
            return

        if self._state == State.CORRECTING and not self.node_cfg.enable_recovery:
            self.get_logger().warn(
                "Recovery disabled; continuing playback.",
                throttle_duration_sec=1.0,
            )
            self._state = State.PLAYING

        if self._state == State.CORRECTING:
            self._on_correcting_tick()
            return

        # PLAYING ------------------------------------------------------------

        path, mapping = self._buf.build_polyline()
        frame = self.node_cfg.planning_frame

        if path:
            pose = self._get_current_pose_2d(frame)
            if pose is not None:
                deviated, dist = self._detector.check_corridor(pose, path)
                self._publish_metrics(dist)
                if deviated:
                    if not self.node_cfg.enable_recovery:
                        self.get_logger().warn(
                            f"Corridor deviation {dist:.3f} m > "
                            f"{self.cfg.corridor_epsilon:.3f} m; "
                            f"recovery disabled, continuing playback.",
                            throttle_duration_sec=1.0,
                        )
                    else:
                        self.get_logger().warn(
                            f"Corridor deviation {dist:.3f} m > "
                            f"{self.cfg.corridor_epsilon:.3f} m; "
                            f"entering recovery."
                        )
                        self._goto_correcting(dist)
                        self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
                        self._publish_debug_markers(path, frame, None, None)
                        return

        sample = self._buf.advance()

        if sample is None:
            if self._buf.is_done():
                self._finish_trajectory()
                return
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self.get_logger().warn(
                f"Chunk {self._buf.cur_chunk_idx} of trajectory "
                f"{self._buf.active_traj_id} not yet received; holding.",
                throttle_duration_sec=1.0,
            )
            self._buf.reset_timing()
            self._current_traj_start_time = self.get_clock().now()
            self._publish_debug_markers([], frame, None, None)
            return

        due = self._current_traj_start_time + Duration(
            nanoseconds=int(sample.samples_consumed * self._buf.active_dt * 1e9)
        )
        self._publish_twist(sample.v, sample.omega, due.to_msg())
        self._publish_debug_markers(path, frame, None, None)

        if self._buf.is_done():
            self._finish_trajectory()

    def _finish_trajectory(self):
        """Transition to IDLE. On success, clear the goal and signal completion."""
        success = self._buf.result_success
        traj_id = self._buf.active_traj_id

        self._state = State.IDLE
        self._goal_in_flight = False
        self._buf.clear()
        self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())

        if success:
            # Verify the robot actually arrived near the goal before clearing it.
            # If the planner reported success but the trajectory ended at the wrong
            # place (e.g. a stale field navigated it to the wrong side of a wall),
            # retaining the goal lets the next gradient-path update fire a replan
            # via the initial-fire path instead of leaving the robot stuck.
            near_goal = True
            if self._goal_xyth is not None:
                pose = self._get_current_pose_2d(self.node_cfg.planning_frame)
                if pose is not None:
                    rx, ry, _ = pose
                    dist = math.hypot(rx - self._goal_xyth[0], ry - self._goal_xyth[1])
                    if dist > self.cfg.path_diff_threshold:
                        near_goal = False
                        self.get_logger().warn(
                            f"Trajectory {traj_id} reported success but robot is "
                            f"{dist:.2f} m from goal "
                            f"(>{self.cfg.path_diff_threshold:.2f} m); "
                            "retaining goal for replan on next gradient update."
                        )
            if near_goal:
                self._goal_xyth = None
                self._publish_cleared_goal_sentinel()
                self.get_logger().info(
                    f"Trajectory {traj_id} complete (success). Returning to IDLE."
                )
            else:
                self.get_logger().info(
                    f"Trajectory {traj_id} ended away from goal. "
                    "Returning to IDLE; will replan on next gradient update."
                )
        else:
            self.get_logger().info(
                f"Trajectory {traj_id} ended without success. Returning to IDLE."
            )

    def _goto_correcting(self, dist: float):
        self._state = State.CORRECTING
        self.get_logger().warn(
            f"Trajectory {self._buf.active_traj_id} at chunk {self._buf.cur_chunk_idx}, "
            f"sample {self._buf.cur_sample_idx}: corridor deviation {dist:.3f} m "
            f"(limit {self.cfg.corridor_epsilon:.3f} m). Suspending playback."
        )

    def _on_correcting_tick(self):
        path, mapping = self._buf.build_polyline()
        frame = self.node_cfg.planning_frame

        if not path:
            if self._buf.result_received:
                # No navigable path data and the planner result is in — the
                # trajectory is finished from the corrector's perspective.
                self.get_logger().info(
                    "Buffer has no path during recovery and result received; "
                    "finishing trajectory."
                )
                self._finish_trajectory()
            else:
                # No pose data available yet (sparse/empty chunk or
                # cur_sample_idx past chunk end). Request a fresh plan so we
                # do not stay stopped indefinitely.
                self.get_logger().warn(
                    "No path available during recovery; triggering replan.",
                    throttle_duration_sec=2.0,
                )
                self._send_action_goal()
                self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
                self._publish_debug_markers([], frame, None, None, "waiting -- no path")
            return

        pose = self._get_current_pose_2d(frame)
        if pose is None:
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers(path, frame, None, None, "tf unavailable")
            return

        result = self._controller.step(pose, path, self._buf.result_received)
        self._publish_metrics(result.perp_dist)

        if result.should_exit:
            if result.exit_kind == ExitKind.ENDPOINT:
                self.get_logger().info(
                    f"Within {self.cfg.recovery_corridor_epsilon:.3f} m of trajectory end "
                    f"during recovery. Returning to IDLE."
                )
            elif result.exit_kind == ExitKind.OVERSHOT:
                self.get_logger().info(
                    "Overshot trajectory end during recovery. Returning to IDLE."
                )
            self._finish_trajectory()
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers([], frame, None, None, result.correction_strat)
            return

        if result.waiting_for_chunks:
            self.get_logger().info(
                "Overshot end of buffered path during recovery; "
                "waiting for next chunk.",
                throttle_duration_sec=1.0,
            )
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers(path, frame, None, None, result.correction_strat)
            return

        self.get_logger().info(
            f"Correcting: perp={result.perp_dist:.3f} m strat={result.correction_strat} "
            f"{'[acceptable -- resuming]' if result.snap_index is not None else '[recovering]'}",
            throttle_duration_sec=0.5,
        )

        if result.snap_index is not None:
            chunk_idx, sample_idx = mapping[result.snap_index]
            self.get_logger().info(
                f"Pose recovered (perp={result.perp_dist:.3f} m). "
                f"Resuming at chunk {chunk_idx}, sample {sample_idx}."
            )
            self._buf.snap_to(chunk_idx, sample_idx)
            self._current_traj_start_time = self.get_clock().now()
            self._state = State.PLAYING
            self._publish_debug_markers(
                path, frame, result.proj, result.carrot, result.correction_strat
            )
            return

        v, omega = result.twist if result.twist is not None else (0.0, 0.0)
        if result.twist is None:
            self.get_logger().warn(
                "No recovery strategy matched -- holding.",
                throttle_duration_sec=1.0,
            )
        self._publish_twist(v, omega, self.get_clock().now().to_msg())
        self._publish_debug_markers(
            path, frame, result.proj, result.carrot, result.correction_strat
        )

    # ----------------------- Pose lookup -----------------------------------

    def _get_current_pose_2d(self, frame: str) -> Optional[tuple[float, float, float]]:
        """Return (x, y, theta) of robot_frame in frame, or None on TF failure."""
        try:
            t = self._tf_buffer.lookup_transform(
                frame, self.node_cfg.robot_frame, Time()
            )
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
                f"TF lookup {self.node_cfg.robot_frame} -> {frame} failed: {e}",
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
        strat_name: Optional[str] = None,
    ) -> None:
        self._visualizer.publish(
            path=path,
            frame=frame,
            proj=proj,
            carrot=carrot,
            state=self._state,
            robot_pose=self._get_current_pose_2d(frame),
            strat_name=strat_name,
            stamp=self.get_clock().now().to_msg(),
        )

    # ----------------------- Publishing ------------------------------------

    def _publish_cleared_goal_sentinel(self):
        sentinel = PoseStamped()
        sentinel.header.stamp = self.get_clock().now().to_msg()
        sentinel.header.frame_id = ""
        self._goal_pub.publish(sentinel)

    def _publish_twist(self, v: float, omega: float, stamp):
        if self.node_cfg.enable_stamped_cmd_vel:
            msg = TwistStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.node_cfg.robot_frame
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
