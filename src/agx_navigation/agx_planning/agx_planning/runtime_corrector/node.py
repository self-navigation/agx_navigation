"""Wheel-command interpreter / corrector between the PMP planner and the
velocity_controllers/JointGroupVelocityController.

This node turns the planner's output into a stream of four-wheel velocity
setpoints on the controller's command topic. It works in two modes, selected
by the `mode` parameter, mirroring the planner's own two modes:

  online  -- the planner runs its own control loop and publishes ready-made
             Float64MultiArray wheel commands on a topic. This node just
             relays them (input topic -> _correct() -> output topic). The
             planner already did the (v, omega) -> wheel mapping.

  offline -- the planner is a PlanToGoal action *server* that rolls out a
             whole start-to-goal trajectory and streams it back as action
             feedback in a burst. This node is the action *client* / trajectory
             interpreter: it sources a goal (target from /goal_pose, start from
             TF), requests a plan, buffers the streamed chunks, and meters the
             per-side wheel setpoints out at the planned sample rate. See
             trajectory_buffer.TrajectoryBuffer for the buffering/timing.

Both modes funnel every command through _emit() -> _correct(). Today _correct()
is the identity (it duplicates each side's setpoint across that side's two
physical wheels): [w_l, w_r] -> [fl, rl, fr, rr] = [w_l, w_l, w_r, w_r], the
controller's joint order. _correct() is the seam where the real corrector lands:
a per-wheel residual -- ultimately a reinforcement-learned policy -- that, when
the measured pose drifts off the planned pose, tweaks each wheel's setpoint to
steer back onto the trajectory. It already receives the planned pose and the
measured pose for exactly that purpose; nothing else in this node changes when
it grows teeth.

JointGroupVelocityController is a forward controller -- it LATCHES the last
command. Every terminal path (goal reached, starvation, shutdown) therefore
publishes an explicit zero; silence would keep the wheels spinning.

Debug visualization (plan-only markers: centerline, corridor tube, state text)
mirrors the planner trajectory on ~/plan, restricted to what a pass-through can
draw. Recovery markers will return with the recovery logic.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Duration as BuiltinDuration
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
    TransformException,
)
from tf_transformations import euler_from_quaternion

from agx_planning_msgs.action import PlanToGoal

from agx_planning.rl_corrector.coeff import apply_coefficients, coefficients_from_action
from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.obs import build_observation
from agx_planning.rl_corrector.policy import load_policy
from agx_planning.utils import declare_and_load_dataclass

from .trajectory_buffer import TrajectoryBuffer

_MARKER_LIFETIME = BuiltinDuration(sec=1, nanosec=0)

# Joint order expected by wheel_velocity_controller.yaml:
#   [front_left, rear_left, front_right, rear_right] = [w_l, w_l, w_r, w_r]


class WheelCorrectorNode(Node):
    """Interprets PMP planner output into four-wheel velocity commands."""

    def __init__(self) -> None:
        super().__init__("wheel_corrector")

        # online: relay a wheel-command topic. offline: drive the planner's
        # PlanToGoal action and play the result back.
        self.declare_parameter("mode", "online")
        # Number of wheel setpoints the controller expects.
        self.declare_parameter("expected_size", 4)
        # Frames for TF lookups (offline goal start pose + debug state text).
        self.declare_parameter("planning_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        # Offline playback: action name and the fallback tick rate used before
        # the first chunk reveals the planner's committed sample dt.
        self.declare_parameter("action_name", "pmp_planner/plan_to_goal")
        self.declare_parameter("control_rate", 10.0)
        # Debug visualization.
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("corridor_epsilon", 0.2)
        self.declare_parameter("debug_marker_rate", 5.0)
        # Seconds since the last command before the state text flips to IDLE.
        self.declare_parameter("idle_after", 0.5)

        self._mode = str(self.get_parameter("mode").value).lower()
        if self._mode not in ("online", "offline"):
            raise ValueError(
                f"mode must be 'online' or 'offline', got {self._mode!r}"
            )
        self._expected_size = int(self.get_parameter("expected_size").value)
        self._planning_frame = str(self.get_parameter("planning_frame").value)
        self._robot_frame = str(self.get_parameter("robot_frame").value)
        self._publish_debug = bool(self.get_parameter("publish_debug").value)
        self._corridor_epsilon = float(self.get_parameter("corridor_epsilon").value)
        self._idle_after = float(self.get_parameter("idle_after").value)
        control_rate = float(self.get_parameter("control_rate").value)
        self._default_dt = 1.0 / control_rate if control_rate > 0.0 else 0.1

        # The forward command controller subscribes with the rclcpp default
        # (reliable, volatile, keep-last). Match it so no command is dropped.
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        #   ~/wheel_cmd_out -> /wheel_velocity_controller/commands
        self._pub = self.create_publisher(Float64MultiArray, "~/wheel_cmd_out", cmd_qos)

        self._last_cmd_time: Optional[rclpy.time.Time] = None

        # TF is needed offline (goal start pose) and for the debug state anchor.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        if self._mode == "online":
            self._init_online(cmd_qos)
        else:
            self._init_offline()

        self._init_corrector()

        if self._publish_debug:
            self._init_debug()

        self.get_logger().info(
            "wheel_corrector up (mode=%s, expected_size=%d, debug=%s)"
            % (self._mode, self._expected_size, self._publish_debug)
        )

    # ----------------------- Mode setup ------------------------------------

    def _init_online(self, cmd_qos: QoSProfile) -> None:
        #   ~/wheel_cmd_in  <- planner wheel-command topic
        self._sub = self.create_subscription(
            Float64MultiArray, "~/wheel_cmd_in", self._on_cmd, cmd_qos
        )

    def _init_offline(self) -> None:
        self._goal_xyth: Optional[Tuple[float, float, float]] = None
        self._goal_handle = None
        self._buf = TrajectoryBuffer(self._default_dt)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, qos)
        # The planner publishes the empty-frame_id sentinel here on completion;
        # we publish it ourselves when a trajectory finishes (see _finish).
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)

        # depth=64 matches the planner's feedback QoS: BVP solves complete much
        # faster than playback consumes them, so chunks arrive in a burst.
        feedback_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._action_client = ActionClient(
            self,
            PlanToGoal,
            str(self.get_parameter("action_name").value),
            feedback_sub_qos_profile=feedback_qos,
        )

        self._tick_timer = None
        self._ensure_tick_timer(self._default_dt)

    def _init_corrector(self) -> None:
        """Load the RL policy that turns the identity seam into a real corrector.

        Config is loaded as ROS params under the `rl_corrector.` prefix (same
        dataclass that trained the policy, so obs/action math matches). With no
        `rl_corrector.policy_path` set, load_policy() returns None and _correct()
        stays byte-identical to the old identity pass-through -- removing the
        policy file cleanly reverts behavior. torch is imported only inside
        load_policy, so a policy-less deployment never pulls it in.
        """
        self._rl_cfg = declare_and_load_dataclass(
            self, RLCorrectorConfig(), prefix="rl_corrector."
        )
        self._policy = load_policy(self._rl_cfg.policy_path)

        # Per-trajectory obs state: previous tracking error (for error rates) and
        # previous coefficients (smoothness feature). Reset on each new traj.
        self._prev_err = None
        self._prev_coeff = np.ones(self._rl_cfg.action_dim)
        # Latest body twist from /odom (a rate, so localization-frame agnostic).
        self._odom_twist: Tuple[float, float] = (0.0, 0.0)
        # Latest IMU reading (gyro_z, ax, ay); None until the first message. The
        # policy trained on this exact signal when cfg.use_imu, so it must be fed
        # here too -- the obs layout (hence the policy) is fixed at train time.
        self._imu: Optional[Tuple[float, float, float]] = None

        if self._policy is not None:
            # Only the policy path needs the measured twist; skip the sub otherwise.
            self.create_subscription(Odometry, "/odom", self._on_odom, 10)
            if self._rl_cfg.use_imu:
                self.create_subscription(Imu, "/imu/data", self._on_imu, 10)
            self.get_logger().info(
                "RL corrector ACTIVE (policy=%s, action_dim=%d, imu=%s, costates=%s)"
                % (self._rl_cfg.policy_path, self._rl_cfg.action_dim,
                   self._rl_cfg.use_imu, self._rl_cfg.use_costates)
            )
        else:
            self.get_logger().info(
                "RL corrector inactive (no rl_corrector.policy_path); "
                "_correct() is identity pass-through."
            )

    def _on_odom(self, msg: Odometry) -> None:
        self._odom_twist = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.angular.z),
        )

    def _on_imu(self, msg: Imu) -> None:
        self._imu = (
            float(msg.angular_velocity.z),
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
        )

    def _reset_corrector_state(self) -> None:
        """Clear per-trajectory obs history so a new plan starts with zero error
        rates and identity previous-coefficients (no carryover across plans)."""
        self._prev_err = None
        self._prev_coeff = np.ones(self._rl_cfg.action_dim)

    def _init_debug(self) -> None:
        marker_rate = float(self.get_parameter("debug_marker_rate").value)
        self._plan_xy: List[Tuple[float, float]] = []
        self._marker_pub = self.create_publisher(MarkerArray, "~/debug_markers", 10)
        #   ~/plan <- planner nav_msgs/Path (debug viz only)
        self._plan_sub = self.create_subscription(Path, "~/plan", self._on_plan, 10)
        period = 1.0 / marker_rate if marker_rate > 0.0 else 0.2
        self._marker_timer = self.create_timer(period, self._publish_debug_markers)

    # ----------------------- Command emission ------------------------------

    def _emit(
        self,
        left: float,
        right: float,
        planned_pose: Optional[Tuple[float, float, float]] = None,
        actual_pose: Optional[Tuple[float, float, float]] = None,
        costates: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """Run one (left, right) wheel-pair command through _correct() and
        publish the resulting four-wheel setpoint."""
        wheels = self._correct(left, right, planned_pose, actual_pose, costates)
        msg = Float64MultiArray()
        msg.data = [float(w) for w in wheels]
        self._pub.publish(msg)
        self._last_cmd_time = self.get_clock().now()

    def _correct(
        self,
        left: float,
        right: float,
        planned_pose: Optional[Tuple[float, float, float]],
        actual_pose: Optional[Tuple[float, float, float]],
        costates: Optional[Tuple[float, ...]] = None,
    ) -> List[float]:
        """Map a planned wheel-pair command to a four-wheel setpoint.

        With no policy loaded -- or whenever the inputs to build a valid
        observation are missing (online relay has no planned/actual pose) -- this
        is the identity: each side's planned setpoint is duplicated across that
        side's two physical wheels ->
          [front_left, rear_left, front_right, rear_right] = [l, l, r, r].

        With a policy loaded, it builds the SAME observation the env trained on
        (path-relative tracking error of `actual_pose` vs `planned_pose`, plus the
        measured /odom twist, previous coefficients, and optional costates),
        predicts per-wheel coefficients, and applies them. It FAILS SAFE to the
        identity on any error or non-finite action, so a bad policy can never
        inject motion beyond clamped multiples of the planned command.
        """
        if (self._policy is None or planned_pose is None or actual_pose is None):
            return [left, left, right, right]

        cfg = self._rl_cfg
        try:
            cs = costates if cfg.use_costates else None
            obs, err = build_observation(
                cfg, planned_pose, actual_pose, self._prev_err,
                cmd_left=left, cmd_right=right,
                v_meas=self._odom_twist[0], omega_meas=self._odom_twist[1],
                prev_coeff=self._prev_coeff,
                imu=self._imu if cfg.use_imu else None,
                wheel_speeds=None, costates=cs,
            )
            action = self._policy.predict(obs)
            if not np.all(np.isfinite(action)):
                raise ValueError("policy returned a non-finite action")
            wheels = apply_coefficients(action, left, right, cfg)
            # Commit obs history only on success, so a failed tick can't poison
            # the next step's error rate / smoothness features.
            self._prev_err = err
            self._prev_coeff = coefficients_from_action(action, cfg)
            return wheels
        except Exception as exc:  # noqa: BLE001 - fail safe to identity, never crash
            self.get_logger().warn(
                "RL corrector errored (%s); falling back to identity." % exc,
                throttle_duration_sec=2.0,
            )
            return [left, left, right, right]

    def _publish_zero(self) -> None:
        """Explicit stop. Bypasses _correct() so a correction can never add
        motion to a commanded halt; the latching controller needs the zero."""
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0]
        self._pub.publish(msg)

    # ----------------------- Online relay ----------------------------------

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)
        if len(data) >= 4:
            left, right = data[0], data[2]
        elif len(data) == 2:
            left, right = data[0], data[1]
        else:
            self.get_logger().warn(
                "expected %d (or 2) wheel setpoints, got %d -- dropping"
                % (self._expected_size, len(data)),
                throttle_duration_sec=2.0,
            )
            return
        self._emit(left, right)

    # ----------------------- Offline goal / action ------------------------

    def _on_goal(self, msg: PoseStamped) -> None:
        # The empty-frame_id sentinel means "goal cleared"; ignore it as input.
        if msg.header.frame_id == "":
            self._goal_xyth = None
            return
        pos = msg.pose.position
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._goal_xyth = (pos.x, pos.y, yaw)
        self.get_logger().info(
            "New goal: (%.2f, %.2f), yaw=%.2f" % (pos.x, pos.y, yaw)
        )
        self._send_action_goal()

    def _send_action_goal(self) -> None:
        if self._goal_xyth is None:
            return
        if not self._action_client.server_is_ready():
            self.get_logger().warn(
                "Planner action server not yet available; will retry on next goal.",
                throttle_duration_sec=5.0,
            )
            return
        start = self._robot_pose()
        if start is None:
            self.get_logger().warn(
                "TF %s->%s unavailable; cannot send action goal."
                % (self._planning_frame, self._robot_frame),
                throttle_duration_sec=2.0,
            )
            return

        goal = PlanToGoal.Goal()
        goal.frame_id = self._planning_frame
        goal.start_x, goal.start_y, goal.start_theta = (
            float(start[0]),
            float(start[1]),
            float(start[2]),
        )
        goal.target_x = float(self._goal_xyth[0])
        goal.target_y = float(self._goal_xyth[1])
        goal.target_theta = float(self._goal_xyth[2])

        future = self._action_client.send_goal_async(
            goal, feedback_callback=self._on_action_feedback
        )
        future.add_done_callback(self._on_goal_accepted)
        self.get_logger().info(
            "Sent action goal: start=(%.2f, %.2f, %.2f) -> target=(%.2f, %.2f, %.2f)"
            % (
                start[0], start[1], start[2],
                goal.target_x, goal.target_y, goal.target_theta,
            )
        )

    def _on_goal_accepted(self, future) -> None:
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn("Action goal rejected by planner.")
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_action_result)

    def _on_action_feedback(self, fb_msg) -> None:
        """One trajectory chunk arrived. Buffer it for the playback timer."""
        chunk = fb_msg.feedback
        traj_id = int(chunk.trajectory_id)

        if traj_id > self._buf.active_traj_id:
            # New trajectory (initial plan or replan): reset the buffer and
            # retime the playback tick to the planner's committed sample dt.
            dt = float(chunk.dt) if chunk.dt > 0.0 else self._default_dt
            self._buf.reset(traj_id, dt)
            self._ensure_tick_timer(dt)
            self._reset_corrector_state()
            self.get_logger().info("Playing trajectory %d (dt=%.3fs)." % (traj_id, dt))
        elif traj_id < self._buf.active_traj_id:
            # A straggler chunk from a superseded trajectory; drop it.
            return

        self._buf.add_chunk(int(chunk.chunk_index), chunk)

    def _on_action_result(self, future) -> None:
        result = future.result().result
        traj_id = int(result.trajectory_id)
        self.get_logger().info(
            "Action result traj_id=%d: success=%s -- %r"
            % (traj_id, result.success, result.message)
        )
        # Let the tick handler drain whatever is buffered before going IDLE.
        if traj_id == self._buf.active_traj_id:
            self._buf.mark_result(bool(result.success))

    # ----------------------- Offline playback tick ------------------------

    def _ensure_tick_timer(self, dt: float) -> None:
        if self._tick_timer is not None:
            self.destroy_timer(self._tick_timer)
        self._tick_timer = self.create_timer(dt, self._on_tick)

    def _on_tick(self) -> None:
        # Idle: nothing to play. The controller is latched at the last zero.
        if self._buf.active_traj_id < 0:
            return

        sample = self._buf.advance()
        if sample is None:
            if self._buf.is_done():
                self._finish()
            else:
                # Needed chunk not yet received -- hold the wheels.
                self._publish_zero()
                self.get_logger().warn(
                    "Chunk %d of trajectory %d not yet received; holding."
                    % (self._buf.cur_chunk_idx, self._buf.active_traj_id),
                    throttle_duration_sec=1.0,
                )
            return

        self._emit(sample.left, sample.right, sample.pose,
                   self._robot_pose(), sample.costates)

        # Last sample of the last chunk just went out and the result is in.
        if self._buf.is_done():
            self._finish()

    def _finish(self) -> None:
        """Trajectory drained: stop the wheels and return to IDLE. On success,
        clear the goal and signal completion ROS-wide."""
        success = self._buf.result_success
        traj_id = self._buf.active_traj_id
        self._publish_zero()
        self._buf.clear()
        if success and self._goal_xyth is not None:
            self._goal_xyth = None
            self._publish_cleared_goal_sentinel()
        self.get_logger().info(
            "Trajectory %d finished (success=%s). IDLE." % (traj_id, success)
        )

    def _publish_cleared_goal_sentinel(self) -> None:
        sentinel = PoseStamped()
        sentinel.header.stamp = self.get_clock().now().to_msg()
        sentinel.header.frame_id = ""
        self._goal_pub.publish(sentinel)

    # ----------------------- Pose lookup -----------------------------------

    def _robot_pose(self) -> Optional[Tuple[float, float, float]]:
        """(x, y, theta) of robot_frame in planning_frame, or None on TF failure."""
        try:
            t = self._tf_buffer.lookup_transform(
                self._planning_frame, self._robot_frame, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException,
                TransformException):
            return None
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        q = t.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return tx, ty, yaw

    # ----------------------- Debug visualization ---------------------------

    def _on_plan(self, msg: Path) -> None:
        self._plan_xy = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]

    def _make_marker(self, mid: int, mtype: int, ns: str, action: int = Marker.ADD,
                     stamp=None) -> Marker:
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self._planning_frame
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = action
        m.lifetime = _MARKER_LIFETIME
        m.pose.orientation.w = 1.0
        return m

    def _publish_debug_markers(self) -> None:
        """Plan-only subset of the corrector debug markers."""
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        if self._plan_xy:
            # -- 0: plan centerline (thin blue LINE_STRIP) --
            m = self._make_marker(0, Marker.LINE_STRIP, "centerline", stamp=stamp)
            m.scale.x = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 1.0
            m.points = [Point(x=x, y=y, z=0.0) for x, y in self._plan_xy]
            markers.markers.append(m)

            # -- 1: corridor tube (thick semi-transparent LINE_STRIP) --
            c = self._make_marker(1, Marker.LINE_STRIP, "corridor", stamp=stamp)
            c.scale.x = 2.0 * self._corridor_epsilon
            c.color.r, c.color.g, c.color.b, c.color.a = 0.2, 0.4, 1.0, 0.2
            c.points = [Point(x=x, y=y, z=0.0) for x, y in self._plan_xy]
            markers.markers.append(c)
        else:
            markers.markers.append(
                self._make_marker(0, Marker.LINE_STRIP, "centerline", Marker.DELETE, stamp)
            )
            markers.markers.append(
                self._make_marker(1, Marker.LINE_STRIP, "corridor", Marker.DELETE, stamp)
            )

        # -- 5: state text above the robot --
        t = self._make_marker(5, Marker.TEXT_VIEW_FACING, "state", stamp=stamp)
        robot = self._robot_pose()
        if robot is not None:
            t.pose.position.x, t.pose.position.y, t.pose.position.z = robot[0], robot[1], 0.5
        t.scale.z = 0.3
        if self._relaying():
            t.text = "PLAY (%s)" % self._mode
            t.color.r, t.color.g, t.color.b, t.color.a = 0.2, 1.0, 0.2, 1.0
        else:
            t.text = "IDLE"
            t.color.r, t.color.g, t.color.b, t.color.a = 0.7, 0.7, 0.7, 1.0
        markers.markers.append(t)

        self._marker_pub.publish(markers)

    def _relaying(self) -> bool:
        if self._last_cmd_time is None:
            return False
        dt = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        return dt <= self._idle_after


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WheelCorrectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
