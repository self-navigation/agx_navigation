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
  CORRECTING  -- the robot has left the epsilon corridor around the planned
                 path. Playback is suspended; a corrective maneuver brings
                 the robot back within the tighter recovery corridor and into
                 heading alignment. A new trajectory_id from the planner
                 (replan) transitions back to PLAYING at any time.

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
from enum import Enum, auto
from typing import Optional

from builtin_interfaces.msg import Duration as BuiltinDuration
from geometry_msgs.msg import Point, Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Duration, Time
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from visualization_msgs.msg import Marker, MarkerArray

from agx_planning_msgs.msg import PlannerTrajectoryChunk
from agx_planning.runtime_corrector import RecoveryConfig, default_strategies
from agx_planning.runtime_corrector.geometry import (
    nearest_projection_on_path,
    walk_ahead_on_path,
)

_MARKER_NS = "pmp_trajectory_corrector"
_MARKER_LIFETIME = BuiltinDuration(sec=1, nanosec=0)


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

        self._marker_pub = self.create_publisher(
            MarkerArray, "/pmp_trajectory_corrector/debug_markers", 10
        )

        # The tick timer recreates itself when self._active_dt changes
        # (a new trajectory may use a different control_rate). See
        # _ensure_tick_timer.
        self._tick_timer = None
        self._ensure_tick_timer(self._default_dt)

        self.get_logger().info(
            f"Trajectory corrector ready (default tick {self._default_dt:.3f} s; "
            f"corridor_epsilon={self._corridor_epsilon:.3f} m, "
            f"recovery_corridor_epsilon={self._recovery_corridor_epsilon:.3f} m)."
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
            self._publish_debug_markers([], self._planning_frame, None, None)
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
            self._publish_debug_markers([], self._planning_frame, None, None)
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
            self._publish_debug_markers([], self._planning_frame, None, None)
            return

        # Build path polyline and check corridor once per tick.
        path, mapping = self._build_path_polyline()
        frame = self._get_planning_frame()
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
            self._publish_debug_markers([], frame, None, None)
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
            self._publish_debug_markers(path, frame, None, None)
            self._current_traj_samples_consumed += 1
            self._cur_sample_idx = 1
        else:
            # Next chunk not yet here -- hold zero.
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

    def _goto_idle(self):
        self._state = _State.IDLE
        self._active_traj_id = -1
        self._chunks.clear()
        self._cur_chunk_idx = 0
        self._cur_sample_idx = 0
        self._current_traj_samples_consumed = 0

    def _goto_correcting(self, dist: float):
        self._state = _State.CORRECTING
        self.get_logger().warn(
            f"Trajectory {self._active_traj_id} at chunk {self._cur_chunk_idx}, "
            f"sample {self._cur_sample_idx}: corridor deviation {dist:.3f} m "
            f"(limit {self._corridor_epsilon:.3f} m). Suspending playback."
        )

    def _on_correcting_tick(self):
        path, mapping = self._build_path_polyline()
        frame = self._get_planning_frame()

        if not path:
            # No buffered chunks -- hold still.
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers([], frame, None, None)
            return

        pose = self._get_current_pose_2d(frame)
        if pose is None:
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers(path, frame, None, None)
            return

        rx, ry, rtheta = pose

        # --- End-of-known-path checks ---
        #
        # These guard against the recovery strategy driving the robot backward
        # when the buffered path doesn't extend all the way to the trajectory
        # end (final chunk not yet received) and the robot has moved past the
        # last known sample.
        fx, fy, ftheta = path[-1]
        dx, dy = rx - fx, ry - fy

        if math.hypot(dx, dy) < self._recovery_corridor_epsilon:
            # Robot is within the recovery threshold of the last known path
            # point -- close enough to call the trajectory done.
            self.get_logger().info(
                f"Within {self._recovery_corridor_epsilon:.3f} m of trajectory end "
                f"during recovery. Returning to IDLE."
            )
            self._goto_idle()
            self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
            self._publish_debug_markers([], frame, None, None)
            return

        overshot = dx * math.cos(ftheta) + dy * math.sin(ftheta) > 0
        if overshot:
            # Robot is past the end of the currently buffered path.
            last_chunk = self._chunks.get(max(self._chunks.keys()))
            if last_chunk is not None and last_chunk.is_final:
                # This IS the complete trajectory -- robot ran past the end.
                self.get_logger().info(
                    "Overshot trajectory end during recovery. Returning to IDLE."
                )
                self._goto_idle()
                self._publish_twist(0.0, 0.0, self.get_clock().now().to_msg())
                self._publish_debug_markers([], frame, None, None)
            else:
                # More chunks are expected; hold still until they arrive and
                # extend the path rather than driving back toward the old end.
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

        # Pass full path to strategies so each can implement its own
        # projection and look-ahead logic.
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

    def _get_planning_frame(self) -> str:
        for chunk in self._chunks.values():
            if chunk.header.frame_id:
                return chunk.header.frame_id
        return self._planning_frame

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
        """Publish a MarkerArray with corridor, state, and recovery debug info.

        path   -- remaining path polyline ((x, y, theta) in order); empty = IDLE
        frame  -- ROS frame_id for all markers
        proj   -- (x, y) of nearest projection on path, or None (PLAYING/IDLE)
        carrot -- (x, y) of look-ahead carrot, or None (PLAYING/IDLE)
        """
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
            m.scale.x = 0.04  # shaft diameter
            m.scale.y = 0.08  # head diameter
            m.scale.z = 0.10  # head length
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
        m.scale.z = 0.3  # text height in metres
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
