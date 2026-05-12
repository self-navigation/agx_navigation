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

Replan trigger: subscribes to /vector_field/planner_data (Float32MultiArray),
computes a path-masked diff (max |T_new - T_old| sampled along the buffered
planned path) against the previous field, and re-fires the action goal when
the diff exceeds field_diff_threshold.

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
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Duration, Time
from std_msgs.msg import Float32MultiArray
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
# Minimal travel-time grid for the path-masked replan diff
# ---------------------------------------------------------------------------


class _FieldGrid:
    """Minimal travel-time grid used only for the path-masked replan diff.

    Stores only T (travel time). The corrector evaluates |T_new - T_old|
    at the buffered path's (x, y) samples to decide whether to replan.
    """

    def __init__(self):
        self._T: Optional[np.ndarray] = None
        self._origin = np.zeros(2, dtype=np.float32)
        self._res: float = 1.0
        self._h: int = 0
        self._w: int = 0

    @property
    def ready(self) -> bool:
        return self._T is not None

    def update_from_msg(self, msg: Float32MultiArray) -> bool:
        """Parse a Float32MultiArray with layout [h, w, ox, oy, res, T(H*W), ...].
        Accepts 1-channel (T only), 3-channel (T, gx, gy), or 4-channel variants.
        Returns True on success, False on size mismatch.
        """
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 5:
            return False
        h = int(data[0])
        w = int(data[1])
        ox = float(data[2])
        oy = float(data[3])
        res = float(data[4])
        n = h * w
        body = data[5:]
        if body.size not in (n, 3 * n, 4 * n):
            return False
        self._T = body[0:n].reshape(h, w)
        self._origin = np.array([ox, oy], dtype=np.float32)
        self._res = res
        self._h, self._w = h, w
        return True

    def in_bounds(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cx = (xs - self._origin[0]) / self._res
        cy = (ys - self._origin[1]) / self._res
        return (cx >= 0) & (cx <= self._w - 1) & (cy >= 0) & (cy <= self._h - 1)

    def sample_T(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Bilinear sample of T at (xs, ys). OOB samples return nan."""
        cx = (xs - self._origin[0]) / self._res
        cy = (ys - self._origin[1]) / self._res
        cx_c = np.clip(cx, 0.0, self._w - 1.0001)
        cy_c = np.clip(cy, 0.0, self._h - 1.0001)
        ix = cx_c.astype(int)
        iy = cy_c.astype(int)
        fx = cx_c - ix
        fy = cy_c - iy
        T00 = self._T[iy, ix]
        T10 = self._T[iy, ix + 1]
        T01 = self._T[iy + 1, ix]
        T11 = self._T[iy + 1, ix + 1]
        T = (
            T00 * (1 - fx) * (1 - fy)
            + T10 * fx * (1 - fy)
            + T01 * (1 - fx) * fy
            + T11 * fx * fy
        )
        oob = ~self.in_bounds(xs, ys)
        return np.where(oob, np.nan, T)


# ---------------------------------------------------------------------------
# Corrector node
# ---------------------------------------------------------------------------


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
        # Replan trigger: max |T_new - T_old| along the buffered planned path
        # tolerated before re-firing an action goal.
        self.declare_parameter("field_diff_threshold", 0.5)
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
        self._field_diff_threshold: float = float(
            self.get_parameter("field_diff_threshold").value
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
        # trajectory. Used by _on_field to diff against the previous field.
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
        self._field = _FieldGrid()
        self._latest_goal_handle: Optional[ClientGoalHandle] = None
        self._latest_send_future = None

        # ---- ROS plumbing --------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, qos)
        self.create_subscription(
            Float32MultiArray, "/vector_field/planner_data", self._on_field, qos
        )

        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)

        if self._stamped:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self._marker_pub = self.create_publisher(
            MarkerArray, "/pmp_trajectory_corrector/debug_markers", 10
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
            f"action '{action_name}'; field_diff_threshold "
            f"{self._field_diff_threshold:.3f})."
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

    # ----------------------- Field subscription ----------------------

    def _on_field(self, msg: Float32MultiArray):
        new_grid = _FieldGrid()
        if not new_grid.update_from_msg(msg):
            self.get_logger().warn(
                "Vector-field message parse failed; ignoring.",
                throttle_duration_sec=5.0,
            )
            return

        should_replan = False
        if self._goal_xyth is not None:
            if self._active_traj_id < 0 and not self._goal_in_flight:
                should_replan = True
            elif (
                self._latest_plan_xy.shape[0] > 0
                and self._field.ready
                and not self._result_received
            ):
                xs = self._latest_plan_xy[:, 0]
                ys = self._latest_plan_xy[:, 1]
                T_old = self._field.sample_T(xs, ys)
                T_new = new_grid.sample_T(xs, ys)
                bad = np.isnan(T_old) | np.isnan(T_new)
                delta = np.where(bad, np.inf, np.abs(T_new - T_old))
                if float(delta.max()) > self._field_diff_threshold:
                    should_replan = True

        self._field = new_grid

        if should_replan:
            sent = self._send_action_goal()
            if sent:
                self.get_logger().info(
                    "Field change (or initial fire): sent action goal."
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

    def _on_tick(self):
        if self._state == _State.IDLE:
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
