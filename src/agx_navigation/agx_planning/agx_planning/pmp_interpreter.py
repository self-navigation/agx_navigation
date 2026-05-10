"""Dummy trajectory interpreter for the offline-mode PMP planner.

Subscribes to /pmp_planner/trajectory_chunks (PlannerTrajectoryChunk) and
publishes /cmd_vel (Twist or TwistStamped, configurable) at the dt rate
carried in each chunk.

Behaviour is a small state machine:

  IDLE     -- no active trajectory. Publishes zero twist on every tick.
  PLAYING  -- streaming chunks of a known trajectory_id. Each tick advances
              one sample within the buffered chunks. On encountering a
              chunk with is_final=True and exhausting it, transition to
              IDLE.

Atomic switch on a new trajectory_id: any chunk with a higher trajectory_id
than the one currently playing immediately replaces the buffer. The old
trajectory's remaining samples are discarded -- the new trajectory was
generated from the chassis's CURRENT TF pose, so the old plan is now
stale.

Out-of-order chunk arrival within a trajectory_id: chunks are stored in
a dict keyed by chunk_index and consumed in index order. If chunk N+1
arrives before N has been fully played, that's fine; we simply pull it
when needed. If chunk N is missing when we need it, we publish zero
twist and wait. This shouldn't happen under reliable QoS but is the
graceful fallback.

This is intentionally minimal: the interpreter performs no closed-loop
tracking, no TF lookups, no pose error compensation. It plays back the
twists exactly as the planner committed them. Future analysis hooks --
the user's stated reason for keeping pose_x / pose_y / pose_theta on the
wire -- should subscribe to the same topic in parallel and consume the
predicted poses directly without going through this node.
"""

from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.time import Duration, Time
from geometry_msgs.msg import Twist, TwistStamped

from agx_planning_msgs.msg import PlannerTrajectoryChunk


class TrajectoryInterpreterNode(Node):

    def __init__(self):
        super().__init__("pmp_trajectory_interpreter")

        self.declare_parameter("enable_stamped_cmd_vel", False)
        self.declare_parameter("robot_frame", "base_link")
        # Tick-rate fallback used only before the first chunk arrives.
        # Once a chunk is in hand, we use chunk.dt instead -- whatever
        # rate the planner committed to.
        self.declare_parameter("default_tick_rate", 10.0)

        self._stamped: bool = self.get_parameter("enable_stamped_cmd_vel").value
        self._robot_frame: str = self.get_parameter("robot_frame").value
        self._default_dt: float = 1.0 / float(
            self.get_parameter("default_tick_rate").value
        )

        # ---- Playback state ------------------------------------------
        # Trajectory currently being played; -1 means IDLE.
        self._active_traj_id: int = -1
        # chunk_index -> chunk message, for the active trajectory only.
        # Held until consumed; popped after the last sample is published.
        self._chunks: Dict[int, PlannerTrajectoryChunk] = {}
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
            f"Interpreter ready (default tick {self._default_dt:.3f} s; "
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
        # IDLE: publish zero twist and wait for a trajectory.
        if self._active_traj_id < 0:
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            return

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
        self._active_traj_id = -1
        self._chunks.clear()
        self._cur_chunk_idx = 0
        self._cur_sample_idx = 0
        self._current_traj_samples_consumed = 0

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


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryInterpreterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
