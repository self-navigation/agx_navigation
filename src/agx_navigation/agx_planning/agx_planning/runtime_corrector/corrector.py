"""Trajectory corrector for the offline-mode PMP planner.

Subscribes to /pmp_planner/trajectory_chunks (PlannerTrajectoryChunk) and
publishes /cmd_vel (Twist or TwistStamped, configurable) at the dt rate
carried in each chunk.

Behaviour is a three-state machine:

  IDLE        -- no active trajectory. Publishes zero twist on every tick.
  PLAYING     -- streaming chunks of a known trajectory_id. Each tick advances
                 one sample within the buffered chunks. On encountering a
                 chunk with is_final=True and exhausting it, transition to
                 IDLE.
  CORRECTING  -- actual pose deviated from the predicted pose by more than
                 pose_error_epsilon. Playback is suspended; a corrective
                 maneuver should bring the robot back within tolerance. A new
                 trajectory_id from the planner (replan) transitions back to
                 PLAYING.

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
from enum import Enum, auto
from typing import Optional

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Duration, Time
from geometry_msgs.msg import Twist, TwistStamped
from tf2_ros import (
    Buffer,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
    TransformListener,
)

from agx_planning_msgs.msg import PlannerTrajectoryChunk
from agx_planning.runtime_corrector import RecoveryConfig, default_strategies


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
        # Euclidean distance (metres) / angle (radians) between predicted and
        # actual pose that triggers a transition from PLAYING to CORRECTING.
        self.declare_parameter("pose_error_epsilon", 0.10)
        self.declare_parameter("pose_angle_epsilon", 0.3)
        # Tighter thresholds that must be reached to leave CORRECTING and
        # resume PLAYING. Setting these below the trigger thresholds creates a
        # hysteresis band and prevents rapid state toggling near the boundary.
        self.declare_parameter("recovery_pos_tolerance", 0.05)
        self.declare_parameter("recovery_angle_tolerance", 0.15)
        # Recovery controller parameters.
        self.declare_parameter("recovery_v_max", 0.3)
        self.declare_parameter("recovery_omega_max", 1.0)
        self.declare_parameter("recovery_K_v", 1.0)
        self.declare_parameter("recovery_K_bearing", 2.0)
        self.declare_parameter("recovery_K_theta", 2.0)
        # How many trajectory samples (nearest-first) to pass to each strategy.
        self.declare_parameter("recovery_n_candidates", 10)

        self._stamped: bool = self.get_parameter("enable_stamped_cmd_vel").value
        self._robot_frame: str = self.get_parameter("robot_frame").value
        self._planning_frame: str = self.get_parameter("planning_frame").value
        self._default_dt: float = 1.0 / float(
            self.get_parameter("default_tick_rate").value
        )
        self._epsilon: float = float(self.get_parameter("pose_error_epsilon").value)
        self._angle_epsilon: float = float(
            self.get_parameter("pose_angle_epsilon").value
        )
        self._recovery_pos_tolerance: float = float(
            self.get_parameter("recovery_pos_tolerance").value
        )
        self._recovery_angle_tolerance: float = float(
            self.get_parameter("recovery_angle_tolerance").value
        )
        recovery_cfg = RecoveryConfig(
            pos_epsilon=self._epsilon,
            angle_epsilon=self._angle_epsilon,
            recovery_pos_tolerance=self._recovery_pos_tolerance,
            recovery_angle_tolerance=self._recovery_angle_tolerance,
            v_max=float(self.get_parameter("recovery_v_max").value),
            omega_max=float(self.get_parameter("recovery_omega_max").value),
            K_v=float(self.get_parameter("recovery_K_v").value),
            K_bearing=float(self.get_parameter("recovery_K_bearing").value),
            K_theta=float(self.get_parameter("recovery_K_theta").value),
        )
        self._recovery_strategies = default_strategies(recovery_cfg)
        self._recovery_n_candidates: int = int(
            self.get_parameter("recovery_n_candidates").value
        )

        # ---- TF -------------------------------------------------------
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- Playback state ------------------------------------------
        self._state: _State = _State.IDLE
        # Trajectory currently being played/corrected; -1 means IDLE.
        self._active_traj_id: int = -1
        # chunk_index -> chunk message, for the active trajectory only.
        # Held until consumed; popped after the last sample is published.
        self._chunks: dict[int, PlannerTrajectoryChunk] = {}
        # Index of the chunk currently being consumed (oldest unfinished).
        self._cur_chunk_idx: int = 0
        # Index of the next sample within self._chunks[self._cur_chunk_idx].
        self._cur_sample_idx: int = 0
        # Effective tick interval for the active trajectory, taken from
        # chunk.dt (all chunks of one trajectory share the same dt).
        self._active_dt: float = self._default_dt
        # ROS time when the current trajectory started playing. Used to compute
        # each sample's due timestamp: start + total_samples_consumed * dt.
        self._current_traj_start_time: Time = self.get_clock().now()
        # Total samples published from chunk data for the active trajectory.
        # Incremented only for real samples, not for held zero-twist ticks.
        self._current_traj_samples_consumed: int = 0

        # ---- ROS plumbing --------------------------------------------
        # Match the planner's chunk-publisher QoS exactly. ALL fields
        # are set explicitly -- omitting `history` was observed to cause
        # the entire profile to fall back to system defaults under some
        # rclpy/RMW combinations, manifesting as TRANSIENT_LOCAL+UNKNOWN-
        # history on this side against the planner's VOLATILE+UNKNOWN-
        # history publisher and an "incompatible QoS" error.
        chunk_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PlannerTrajectoryChunk,
            "/pmp_planner/trajectory_chunks",
            self._on_chunk,
            chunk_qos,
        )

        if self._stamped:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # The tick timer recreates itself when self._active_dt changes
        # (a new trajectory may use a different control_rate). See
        # _ensure_tick_timer.
        self._tick_timer = None
        self._ensure_tick_timer(self._default_dt)

        self.get_logger().info(
            f"Trajectory corrector ready (default tick {self._default_dt:.3f} s; "
            f"will adopt each trajectory's chunk.dt on first chunk)."
        )

    # ----------------------- Subscription handler --------------------------

    def _on_chunk(self, msg: PlannerTrajectoryChunk):
        # Newer trajectory_id supersedes whatever we were playing. The
        # planner emitted this chunk from a fresh rollout starting at
        # the chassis's current TF pose, so the old plan is stale -- we
        # discard it and switch atomically.
        if msg.trajectory_id > self._active_traj_id:
            self.get_logger().info(
                f"Switching to trajectory {msg.trajectory_id} "
                f"(was {self._active_traj_id})."
            )
            self._active_traj_id = int(msg.trajectory_id)
            self._state = _State.PLAYING
            self._chunks = {}
            self._cur_chunk_idx = 0
            self._cur_sample_idx = 0
            self._current_traj_start_time = self.get_clock().now()
            self._current_traj_samples_consumed = 0
            # Adopt the new trajectory's tick rate.
            if msg.dt > 0.0 and abs(msg.dt - self._active_dt) > 1e-6:
                self._active_dt = float(msg.dt)
                self._ensure_tick_timer(self._active_dt)
        elif msg.trajectory_id < self._active_traj_id:
            # Stale chunk from a superseded trajectory -- drop silently.
            return

        # Same-id chunk: store it.
        self._chunks[int(msg.chunk_index)] = msg

    # ----------------------- Tick handler ----------------------------------

    def _ensure_tick_timer(self, dt: float):
        """(Re)create the periodic publish timer with the given period.

        Called on construction with the default and again whenever a new
        trajectory's chunk.dt differs from the active period. The old
        timer is destroyed first to avoid double-publishing.
        """
        if self._tick_timer is not None:
            self.destroy_timer(self._tick_timer)
        self._tick_timer = self.create_timer(dt, self._on_tick)

    def _on_tick(self):
        if self._state == _State.IDLE:
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            return

        if self._state == _State.CORRECTING:
            self._on_correcting_tick()
            return

        # PLAYING ------------------------------------------------------------

        # Locate the current chunk. If missing (out-of-order delivery),
        # hold zero -- it should arrive imminently under reliable QoS.
        cur = self._chunks.get(self._cur_chunk_idx)
        if cur is None:
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self.get_logger().warn(
                f"Chunk {self._cur_chunk_idx} of trajectory "
                f"{self._active_traj_id} not yet received; holding.",
                throttle_duration_sec=1.0,
            )
            self._current_traj_start_time = self.get_clock().now()
            self._current_traj_samples_consumed = 0
            return

        n = len(cur.linear_x)

        # Empty chunk with is_final=True: end-of-trajectory sentinel
        # (planner emits this when chassis was already at goal, or when
        # a BVP failure / timeout aborts the rollout). Go to IDLE.
        if n == 0 and cur.is_final:
            self.get_logger().info(
                f"Trajectory {self._active_traj_id} terminated "
                f"(empty final chunk). Returning to IDLE."
            )
            self._goto_idle()
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            return

        # Consume one sample.
        if self._cur_sample_idx < n:
            err = self._pose_error_exceeds_epsilon(
                cur.pose_x[self._cur_sample_idx],
                cur.pose_y[self._cur_sample_idx],
                cur.pose_theta[self._cur_sample_idx],
                cur.header.frame_id,
            )
            if err is not None:
                self._goto_correcting(*err)
                self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
                return
            v = float(cur.linear_x[self._cur_sample_idx])
            w = float(cur.angular_z[self._cur_sample_idx])
            due = self._current_traj_start_time + Duration(
                nanoseconds=int(
                    self._current_traj_samples_consumed * self._active_dt * 1e9
                )
            )
            self._publish_twist(v, w, due.to_msg())
            self._current_traj_samples_consumed += 1
            self._cur_sample_idx += 1
            return

        # Chunk exhausted. Drop it from the buffer and advance.
        del self._chunks[self._cur_chunk_idx]
        last_chunk_was_final = bool(cur.is_final)
        self._cur_chunk_idx += 1
        self._cur_sample_idx = 0

        if last_chunk_was_final:
            # End of trajectory.
            self.get_logger().info(
                f"Trajectory {self._active_traj_id} complete. Returning to IDLE."
            )
            self._goto_idle()
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            return

        # Try to consume the first sample of the next chunk on this same
        # tick, so we don't drop a tick-worth of motion at chunk boundaries.
        nxt = self._chunks.get(self._cur_chunk_idx)
        if nxt is not None and len(nxt.linear_x) > 0:
            err = self._pose_error_exceeds_epsilon(
                nxt.pose_x[0],
                nxt.pose_y[0],
                nxt.pose_theta[0],
                nxt.header.frame_id,
            )
            if err is not None:
                self._goto_correcting(*err)
                self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
                return
            v = float(nxt.linear_x[0])
            w = float(nxt.angular_z[0])
            due = self._current_traj_start_time + Duration(
                nanoseconds=int(
                    self._current_traj_samples_consumed * self._active_dt * 1e9
                )
            )
            self._publish_twist(v, w, due.to_msg())
            self._current_traj_samples_consumed += 1
            self._cur_sample_idx = 1
        else:
            # Next chunk not yet here -- hold zero.
            self.get_logger().warn(
                f"Chunk {self._cur_chunk_idx} (which is meant to be received after the current one) of trajectory "
                f"{self._active_traj_id} not yet received; holding.",
                throttle_duration_sec=1.0,
            )
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._current_traj_start_time = self.get_clock().now()
            self._current_traj_samples_consumed = 0

    def _goto_idle(self):
        self._state = _State.IDLE
        self._active_traj_id = -1
        self._chunks.clear()
        self._cur_chunk_idx = 0
        self._cur_sample_idx = 0
        self._current_traj_samples_consumed = 0

    def _goto_correcting(self, pos_err: float, angle_err: float):
        self._state = _State.CORRECTING
        self.get_logger().warn(
            f"Pose error on trajectory {self._active_traj_id} at chunk "
            f"{self._cur_chunk_idx}, sample {self._cur_sample_idx}: "
            f"pos={pos_err:.3f} m (limit {self._epsilon:.3f}), "
            f"angle={math.degrees(angle_err):.1f} deg "
            f"(limit {math.degrees(self._angle_epsilon):.1f}). "
            f"Suspending playback."
        )

    def _on_correcting_tick(self):
        current_pose, raw_candidates, is_acceptable = self._find_nearest_samples()

        if current_pose is None or not raw_candidates:
            # TF failure or no buffered chunks -- hold still.
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            return

        best_chunk_idx, best_sample_idx, best_tx, best_ty, best_ttheta = raw_candidates[
            0
        ]
        rx, ry, rtheta = current_pose
        best_pos_diff = math.hypot(rx - best_tx, ry - best_ty)
        best_angle_diff = abs(math.remainder(rtheta - best_ttheta, 2 * math.pi))
        self.get_logger().info(
            f"Correcting: nearest chunk={best_chunk_idx} sample={best_sample_idx} "
            f"pos={best_pos_diff:.3f} m angle={math.degrees(best_angle_diff):.1f} deg "
            f"{'[OK]' if is_acceptable else '[recovering]'}",
            throttle_duration_sec=0.5,
        )

        if is_acceptable:
            self.get_logger().info(
                f"Pose recovered at chunk {best_chunk_idx}, "
                f"sample {best_sample_idx}. Resuming playback."
            )
            self._cur_chunk_idx = best_chunk_idx
            self._cur_sample_idx = best_sample_idx
            self._state = _State.PLAYING
            return

        # Pass the current pose and pose-only candidate list to strategies.
        pose_candidates = [(tx, ty, ttheta) for _, _, tx, ty, ttheta in raw_candidates]
        for strategy in self._recovery_strategies:
            if strategy.can_handle(current_pose, pose_candidates):
                v, omega = strategy.compute_twist(current_pose, pose_candidates)
                self.get_logger().info(
                    f"Pose recovery strategy {strategy.__class__.__name__} "
                    f"received pose={rx:.3f} m {ry:.3f} m {rtheta:.3f} rad and "
                    f"computed v={v:.3f} m/s omega={omega:.3f} rad/s."
                )
                self._publish_twist(v, omega, self.get_clock().now().to_msg())
                return

        # No strategy matched -- TODO: request replan when too far off course.
        self.get_logger().warn(
            f"No recovery strategy matched -- holding.",
            throttle_duration_sec=1.0,
        )
        self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())

    # ----------------------- Pose error detection / recovery ---------------

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

    def _pose_error_exceeds_epsilon(
        self,
        pred_x: float,
        pred_y: float,
        pred_theta: float,
        planning_frame: str,
    ) -> Optional[tuple[float, float]]:
        """Return (pos_err, angle_err) if either threshold is exceeded, else None.

        Uses chunk header.frame_id, falling back to the planning_frame
        parameter. Returns None on TF failure to avoid aborting playback.
        """
        frame = planning_frame if planning_frame else self._planning_frame
        pose = self._get_current_pose_2d(frame)
        if pose is None:
            return None
        rx, ry, rtheta = pose
        pos_err = math.hypot(rx - pred_x, ry - pred_y)
        angle_err = abs(math.remainder(rtheta - pred_theta, 2 * math.pi))
        if pos_err > self._epsilon or angle_err > self._angle_epsilon:
            return pos_err, angle_err
        return None

    def _find_nearest_samples(
        self,
    ) -> tuple[
        Optional[tuple[float, float, float]],
        list[tuple[int, int, float, float, float]],
        bool,
    ]:
        """Scan buffered chunks from the current playback position forward.

        Returns:
          current_pose -- (rx, ry, rtheta) or None on TF failure.
          candidates   -- list of (chunk_idx, sample_idx, tx, ty, ttheta)
                          sorted by weighted position+angle score, capped at
                          self._recovery_n_candidates entries.
          is_acceptable -- True if the best candidate is within both epsilons.
        """
        frame = self._planning_frame
        for chunk in self._chunks.values():
            if chunk.header.frame_id:
                frame = chunk.header.frame_id
                break

        pose = self._get_current_pose_2d(frame)
        if pose is None:
            return None, [], False
        rx, ry, rtheta = pose

        xy_weight = 10.0
        angle_weight = 1.0
        scored: list[tuple[float, int, int, float, float, float]] = []

        for chunk_idx in sorted(self._chunks):
            chunk = self._chunks[chunk_idx]
            start = self._cur_sample_idx if chunk_idx == self._cur_chunk_idx else 0
            for sample_idx in range(start, len(chunk.pose_x)):
                tx = float(chunk.pose_x[sample_idx])
                ty = float(chunk.pose_y[sample_idx])
                ttheta = float(chunk.pose_theta[sample_idx])
                pos_dist = math.hypot(rx - tx, ry - ty)
                angle_dist = abs(math.remainder(rtheta - ttheta, 2 * math.pi))
                score = xy_weight * pos_dist + angle_weight * angle_dist
                scored.append((score, chunk_idx, sample_idx, tx, ty, ttheta))

        if not scored:
            return pose, [], False

        scored.sort(key=lambda e: e[0])
        top = scored[: self._recovery_n_candidates]
        candidates = [(ci, si, tx, ty, tth) for _, ci, si, tx, ty, tth in top]

        _, _, _, best_tx, best_ty, best_tth = scored[0]
        best_pos = math.hypot(rx - best_tx, ry - best_ty)
        best_angle = abs(math.remainder(rtheta - best_tth, 2 * math.pi))
        is_acceptable = (
            best_pos < self._recovery_pos_tolerance
            and best_angle < self._recovery_angle_tolerance
        )

        return pose, candidates, is_acceptable

    # ----------------------- Publishing ------------------------------------

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
