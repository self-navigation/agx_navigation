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
import signal
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.signals import SignalHandlerOptions
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
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from agx_planning_msgs.action import PlanToGoal

from agx_planning.rl_corrector.coeff import apply_residual, clipped_action
from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.obs import build_observation
from agx_planning.rl_corrector.policy import load_policy
from agx_planning.runtime_corrector import tvlqr as tvlqr_mod
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
        # Offline mode: buffer the entire rollout before driving. See the long
        # comment in _on_tick -- streaming playback starves and stutters,
        # because this planner is slower than realtime on the baked map.
        self.declare_parameter("wait_for_complete", True)
        # Offline playback cursor: "time" (one sample per tick) or "progress"
        # (project the measured pose onto the plan). See the trajectory_buffer
        # module docstring. Time indexing makes every correction cost distance,
        # so the plan runs out with the robot short of the goal; progress
        # indexing trades that for a longer drive. Default "time" until the
        # fixture says otherwise -- this is the only writer of wheel commands.
        self.declare_parameter("playback_index", "time")
        # Progress mode only. How far ahead of the cursor the projection may
        # jump in one tick, in samples: a plan that loops back near itself must
        # not be able to skip the loop.
        self.declare_parameter("playback_max_skip", 20)
        # Progress mode only. Give up at this multiple of the planned duration.
        # Time indexing is self-terminating (the samples run out); progress
        # indexing is not, so a robot that stalls against a wall would otherwise
        # replay its feed-forward for ever.
        self.declare_parameter("playback_timeout_factor", 3.0)
        # Progress mode only. How far the reference may lead the robot, in
        # samples. Zero would slave the cursor to measured progress, which
        # deadlocks: the plan starts from rest, so sample 0 commands zero speed
        # and nothing ever moves. Must be > 0.
        self.declare_parameter("playback_max_lead", 10)

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
        self._playback_index = str(self.get_parameter("playback_index").value).lower()
        if self._playback_index not in ("time", "progress"):
            raise ValueError(
                "playback_index must be 'time' or 'progress', got "
                f"{self._playback_index!r}"
            )
        self._playback_max_skip = int(self.get_parameter("playback_max_skip").value)
        self._playback_max_lead = int(self.get_parameter("playback_max_lead").value)
        if self._playback_index == "progress" and self._playback_max_lead < 1:
            raise ValueError("playback_max_lead must be >= 1 (0 deadlocks at rest)")
        self._playback_timeout_factor = float(
            self.get_parameter("playback_timeout_factor").value
        )
        self._play_started: Optional[rclpy.time.Time] = None

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
        self._buf = TrajectoryBuffer(
            self._default_dt,
            max_skip=self._playback_max_skip,
            max_lead=self._playback_max_lead,
        )
        self._wait_for_complete = bool(
            self.get_parameter("wait_for_complete").value)
        self._playing = False

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

        # TVLQR (neighboring-optimal) corrector, under the `tvlqr.` prefix.
        # Disabled by default, so loading it changes nothing. When enabled it
        # TAKES PRECEDENCE over the RL policy: the two are alternative
        # correctors, not a stack, and running both would double-correct. The
        # intended end state is TVLQR as the baseline with an RL residual on
        # top, but that needs a policy trained against this baseline -- until
        # then, one or the other.
        self._tvlqr_cfg = declare_and_load_dataclass(
            self, tvlqr_mod.TVLQRConfig(), prefix="tvlqr."
        )
        self._tvlqr_gains: Optional[tvlqr_mod.GainCache] = None
        self._tvlqr_diag_pub = None
        if self._tvlqr_cfg.enabled:
            self._tvlqr_gains = tvlqr_mod.GainCache(
                self._tvlqr_cfg, self._rl_cfg.control_dt
            )
            self._tvlqr_diag_pub = self.create_publisher(
                Float64MultiArray, "~/tvlqr_diagnostics", 10
            )
            # Running tallies so a run can be judged from one log line rather
            # than by scraping the whole diagnostics stream.
            self._tvlqr_ticks = 0
            self._tvlqr_sat_ticks = 0
            self._tvlqr_sq_cross = 0.0
            self._tvlqr_max_cross = 0.0
            self.create_timer(2.0, self._log_tvlqr_summary)

        # Per-trajectory obs state: previous tracking error (for error rates) and
        # previous action (smoothness feature). Reset on each new traj.
        self._prev_err = None
        self._prev_action = np.zeros(self._rl_cfg.action_dim)
        # Latest body twist from /odom (a rate, so localization-frame agnostic).
        self._odom_twist: Tuple[float, float] = (0.0, 0.0)
        # Latest IMU reading (gyro_z, ax, ay); None until the first message. The
        # policy trained on this exact signal when cfg.use_imu, so it must be fed
        # here too -- the obs layout (hence the policy) is fixed at train time.
        self._imu: Optional[Tuple[float, float, float]] = None

        if self._tvlqr_cfg.enabled:
            c = self._tvlqr_cfg
            self.get_logger().info(
                "TVLQR corrector ACTIVE (Q=[%.2g,%.2g,%.2g] R=[%.2g,%.2g] "
                "max_dv=%.2g max_domega=%.2g)%s"
                % (c.q_along, c.q_cross, c.q_heading, c.r_v, c.r_omega,
                   c.max_dv, c.max_domega,
                   "  [RL policy present but OVERRIDDEN]"
                   if self._policy is not None else "")
            )
        elif self._policy is not None:
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
        rates and zero previous-action (no carryover across plans)."""
        self._prev_err = None
        self._prev_action = np.zeros(self._rl_cfg.action_dim)

    def _init_debug(self) -> None:
        marker_rate = float(self.get_parameter("debug_marker_rate").value)
        self._plan_xy: List[Tuple[float, float]] = []
        # The plan sample currently being commanded -- the point the robot is
        # being told to be at right now. Drawn as an arrow so a screenshot shows
        # where the reference is, not just where the path goes.
        self._ref_pose: Optional[Tuple[float, float, float]] = None
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
        measured /odom twist, previous action, and optional costates), predicts a
        per-wheel residual, and adds it to the planned command. It FAILS SAFE to
        the identity on any error or non-finite action, so a bad policy can never
        inject motion beyond the clamped residual on top of the planned command.
        """
        if self._tvlqr_cfg.enabled:
            return self._correct_tvlqr(left, right, planned_pose, actual_pose)

        if (self._policy is None or planned_pose is None or actual_pose is None):
            return [left, left, right, right]

        cfg = self._rl_cfg
        try:
            cs = costates if cfg.use_costates else None
            obs, err = build_observation(
                cfg, planned_pose, actual_pose, self._prev_err,
                cmd_left=left, cmd_right=right,
                v_meas=self._odom_twist[0], omega_meas=self._odom_twist[1],
                prev_action=self._prev_action,
                imu=self._imu if cfg.use_imu else None,
                wheel_speeds=None, costates=cs,
            )
            action = self._policy.predict(obs)
            if not np.all(np.isfinite(action)):
                raise ValueError("policy returned a non-finite action")
            wheels = apply_residual(action, left, right, cfg,
                                    prev_action=self._prev_action)
            # Commit obs history only on success, so a failed tick can't poison
            # the next step's error rate / smoothness features.
            self._prev_err = err
            self._prev_action = clipped_action(action, cfg, prev_action=self._prev_action)
            return wheels
        except Exception as exc:  # noqa: BLE001 - fail safe to identity, never crash
            self.get_logger().warn(
                "RL corrector errored (%s); falling back to identity." % exc,
                throttle_duration_sec=2.0,
            )
            return [left, left, right, right]

    def _correct_tvlqr(
        self,
        left: float,
        right: float,
        planned_pose: Optional[Tuple[float, float, float]],
        actual_pose: Optional[Tuple[float, float, float]],
    ) -> List[float]:
        """Neighboring-optimal correction of one wheel-pair command.

        The reference wheel command is converted to the reference twist, the
        feedback is applied in twist space (the space the real chassis accepts),
        and the corrected twist is mapped back to wheel speeds for the
        controller. On hardware the middle value -- the corrected (v, omega) --
        is what would be published directly, with no wheel mapping at all.

        Fails safe to the identity whenever the pose pair is missing (the online
        relay has no planned pose until the planner supplies one) or anything
        raises, so an unusable correction can never inject motion.
        """
        if planned_pose is None or actual_pose is None:
            return [left, left, right, right]

        kin = self._rl_cfg
        try:
            v_ref, omega_ref = tvlqr_mod.wheels_to_twist(left, right, kin)
            err = tvlqr_mod.tracking_error(planned_pose, actual_pose)
            K = self._tvlqr_gains.get(v_ref, omega_ref)
            v_cmd, omega_cmd, diag = tvlqr_mod.correct(
                K, err, v_ref, omega_ref, self._tvlqr_cfg
            )
            if not diag.valid:
                return [left, left, right, right]

            wl, wr = tvlqr_mod.twist_to_wheels(v_cmd, omega_cmd, kin)
            m = kin.wheel_cmd_max
            wl = float(np.clip(wl, -m, m))
            wr = float(np.clip(wr, -m, m))

            self._accumulate_tvlqr(diag)
            if self._tvlqr_diag_pub is not None:
                msg = Float64MultiArray()
                msg.data = [float(x) for x in diag.as_array()]
                self._tvlqr_diag_pub.publish(msg)
            return [wl, wl, wr, wr]
        except Exception as exc:  # noqa: BLE001 - fail safe to identity, never crash
            self.get_logger().warn(
                "TVLQR corrector errored (%s); falling back to identity." % exc,
                throttle_duration_sec=2.0,
            )
            return [left, left, right, right]

    def _accumulate_tvlqr(self, diag) -> None:
        """Fold one tick into the running summary tallies."""
        self._tvlqr_ticks += 1
        if diag.saturated_v or diag.saturated_omega:
            self._tvlqr_sat_ticks += 1
        self._tvlqr_sq_cross += diag.e_cross ** 2
        self._tvlqr_max_cross = max(self._tvlqr_max_cross, abs(diag.e_cross))

    def _log_tvlqr_summary(self) -> None:
        """Periodic one-line verdict on how well the corrector is holding.

        RMS cross-track is the headline number; `sat` is the health warning. A
        corrector that is saturated a large fraction of the time is being asked
        for more authority than it has, which means either max_dv/max_domega are
        too tight or the deviation is genuinely beyond correction and the
        trajectory needs replanning.
        """
        if not self._tvlqr_ticks:
            return
        rms = math.sqrt(self._tvlqr_sq_cross / self._tvlqr_ticks)
        sat = 100.0 * self._tvlqr_sat_ticks / self._tvlqr_ticks
        self.get_logger().info(
            "TVLQR: ticks=%d  rms_cross=%.4f m  max_cross=%.4f m  sat=%.1f%%  "
            "gain_buckets=%d"
            % (self._tvlqr_ticks, rms, self._tvlqr_max_cross, sat,
               len(self._tvlqr_gains) if self._tvlqr_gains else 0)
        )

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
            self._playing = False
            self._play_started = None
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
            return

        # A result for a trajectory we never received a chunk of.
        #
        # `active_traj_id` is set in `_on_chunk`, so a plan that fails at chunk 0
        # -- the BVP mesh-node exhaustion that fails ~36% of fresh start/goal
        # pairs -- leaves it at -1. The equality above is then false, the result
        # is dropped, and `_on_tick`'s idle guard (`active_traj_id < 0`) returns
        # before `_finish()` can run. Net effect: no zero command, no sentinel,
        # and every consumer waits forever on a goal nobody is pursuing.
        #
        # `_finish`'s docstring says the sentinel fires on ANY terminal outcome.
        # That fix covered "failed after playback began" and left this case,
        # which is the more common one: the planner rejects a goal in under a
        # second, and the whole stack goes quiet with the robot stationary.
        # Observed 2026-08-18 driving the fixture end to end (traj 2 and 5 of a
        # five-goal session hung their clients for their full timeout).
        #
        # Nothing is buffered to drain, so go IDLE directly rather than routing
        # through the tick. Publishing zero is not redundant: the controller
        # latches its last command, and this path can be reached while a
        # superseded trajectory's last non-zero command is still in force.
        if self._buf.active_traj_id < 0:
            self.get_logger().warn(
                "Plan for traj_id=%d produced no trajectory (%r). "
                "Stopping and clearing the goal." % (traj_id, result.message)
            )
            self._publish_zero()
            self._buf.clear()
            self._playing = False
            if self._goal_xyth is not None:
                self._goal_xyth = None
                self._publish_cleared_goal_sentinel()

    # ----------------------- Offline playback tick ------------------------

    def _ensure_tick_timer(self, dt: float) -> None:
        if self._tick_timer is not None:
            self.destroy_timer(self._tick_timer)
        self._tick_timer = self.create_timer(dt, self._on_tick)

    def _on_tick(self) -> None:
        # Idle: nothing to play. The controller is latched at the last zero.
        if self._buf.active_traj_id < 0:
            return

        # Don't begin playback until the whole rollout has arrived.
        #
        # The streaming design assumed BVP solves outrun playback (hence the
        # depth=64 feedback queue). Measured on the baked floor map, the
        # opposite holds: the planner emits a 0.5 s chunk about every 0.68 s,
        # so playback starves on nearly every chunk.
        #
        # Starving is not a harmless pause. The hold path below publishes ZERO,
        # and the plan's state includes the wheel speeds (w_l, w_r) -- so each
        # stall brakes the wheels and playback then resumes from a sample that
        # assumes them already spinning. During a turn the two wheels differ,
        # so the injected error lands mostly on heading, and the robot wanders
        # off a trajectory that was never actually executed.
        #
        # Waiting costs the pre-drive planning time but plays the trajectory as
        # planned, which is the point of "offline" mode. Set false to restore
        # streaming if a future planner is genuinely faster than realtime.
        if self._wait_for_complete and not self._playing:
            if not self._buf.result_received:
                return
            self._playing = True
            self._play_started = self.get_clock().now()

        pose = self._robot_pose()

        # Progress indexing has no natural end -- the cursor only moves when the
        # robot does -- so bound it by the planned duration. Time indexing needs
        # no such guard: it drains one sample per tick regardless.
        if self._playback_index == "progress" and self._play_started is not None:
            _, total = self._buf.progress
            budget = total * self._buf.active_dt * self._playback_timeout_factor
            elapsed = (self.get_clock().now() - self._play_started).nanoseconds * 1e-9
            if budget > 0.0 and elapsed > budget:
                consumed, _ = self._buf.progress
                self.get_logger().warn(
                    "Playback timed out after %.1fs (%.1fx planned); at sample "
                    "%d/%d. Stopping." % (elapsed, self._playback_timeout_factor,
                                          consumed, total)
                )
                self._finish()
                return

        # Progress mode needs the measured position to project onto the plan; if
        # TF is momentarily unavailable, fall back to a time step rather than
        # stalling the cursor -- a held cursor republishes the same feed-forward.
        actual_xy = None
        if self._playback_index == "progress" and pose is not None:
            actual_xy = (pose[0], pose[1])

        sample = self._buf.advance(actual_xy)
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

        self._ref_pose = sample.pose
        self._emit(sample.left, sample.right, sample.pose, pose, sample.costates)

        # Last sample of the last chunk just went out and the result is in.
        if self._buf.is_done():
            self._finish()

    def _finish(self) -> None:
        """Trajectory drained: stop the wheels and return to IDLE, and signal
        completion ROS-wide on ANY terminal outcome.

        The sentinel used to fire only on success, which left a failed goal
        signalling nothing at all -- every consumer sat waiting on a goal that
        nobody was pursuing any more, until its own timeout. That was not
        hypothetical: it hung the random-goal driver for its full dwell, and it
        was worst for the near-miss case the planner used to report as failure
        (see the stagnation note in pmp_planner/rollout.py).

        Firing on failure is safe for the known consumers. The sentinel means
        "no goal is being pursued", which is exactly true after a failure:
        vector_field clears current_goal and stops recomputing its field, which
        is what we want when nothing is driving. It is NOT a claim of arrival --
        anything that needs to distinguish the two must read the action result,
        which carries success and a message.
        """
        success = self._buf.result_success
        traj_id = self._buf.active_traj_id
        self._publish_zero()
        self._buf.clear()
        if self._goal_xyth is not None:
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

        robot = self._robot_pose()

        # -- 2: the sample being commanded right now (ARROW, with heading) --
        # -- 3: robot -> reference, so the LEAD/LAG is visible at a glance --
        if self._ref_pose is not None and self._relaying():
            rx, ry, rth = self._ref_pose
            a = self._make_marker(2, Marker.ARROW, "reference", stamp=stamp)
            a.pose.position.x, a.pose.position.y, a.pose.position.z = rx, ry, 0.05
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, rth)
            a.pose.orientation.x = qx
            a.pose.orientation.y = qy
            a.pose.orientation.z = qz
            a.pose.orientation.w = qw
            a.scale.x, a.scale.y, a.scale.z = 0.4, 0.08, 0.08
            a.color.r, a.color.g, a.color.b, a.color.a = 1.0, 0.6, 0.0, 1.0
            markers.markers.append(a)

            if robot is not None:
                g = self._make_marker(3, Marker.LINE_LIST, "lag", stamp=stamp)
                g.scale.x = 0.03
                g.color.r, g.color.g, g.color.b, g.color.a = 1.0, 0.6, 0.0, 0.8
                g.points = [Point(x=robot[0], y=robot[1], z=0.05),
                            Point(x=rx, y=ry, z=0.05)]
                markers.markers.append(g)
        else:
            markers.markers.append(
                self._make_marker(2, Marker.ARROW, "reference", Marker.DELETE, stamp)
            )
            markers.markers.append(
                self._make_marker(3, Marker.LINE_LIST, "lag", Marker.DELETE, stamp)
            )

        # -- 5: state text, offset from the robot in the GROUND PLANE --
        # Raising it in z does not separate it: the fixture is watched from
        # directly overhead, so a label above the robot lands on the robot. The
        # +y offset is chosen to clear run_recorder's truth label, which sits at
        # -y on the same point.
        t = self._make_marker(5, Marker.TEXT_VIEW_FACING, "state", stamp=stamp)
        if robot is not None:
            t.pose.position.x, t.pose.position.y, t.pose.position.z = (
                robot[0], robot[1] + 0.6, 0.5)
        t.scale.z = 0.2
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


def _install_stop_on_signal(node: "WheelCorrectorNode") -> None:
    """Publish an explicit stop on SIGINT/SIGTERM while the context is still up.

    JointGroupVelocityController LATCHES its last command, so every terminal path
    must publish a zero -- silence keeps the wheels spinning. The goal-reached and
    starvation paths do that themselves; signals are the gap.

    rclpy will not let us chain in front of it. `rclpy.init()` installs a
    C-LEVEL signal handler, so a Python `signal.signal()` registered afterwards
    never runs at all (verified: the handler's log line never appeared), and by
    the time ExternalShutdownException surfaces in spin() the context is already
    torn down and nothing can be published. So we ask rclpy not to install
    handlers (SignalHandlerOptions.NO) and own the signal outright: stop the
    wheels first, then shut the context down, which makes spin() exit normally.

    Publishing from a signal handler is not async-signal-safe in the strict
    sense. That is the accepted trade: the alternative is signalling a spin loop
    that is already being torn down, and a wedged shutdown is far less dangerous
    than a robot that keeps driving. Every step is wrapped so a failed stop can
    never prevent the shutdown itself.
    """

    def handler(signum, frame):
        try:
            node._publish_zero()
            node.get_logger().info("signal %d: published explicit stop" % signum)
        except Exception as exc:  # noqa: BLE001 - never block shutdown on a stop failure
            node.get_logger().warn("signal %d: failed to publish stop (%s)"
                                   % (signum, exc))
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001 - the finally block retries this
            pass

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main(args=None) -> None:
    # NO signal handlers from rclpy: we install our own so the wheels can be
    # stopped before the context dies. See _install_stop_on_signal.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = WheelCorrectorNode()
    _install_stop_on_signal(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._publish_zero()
    except ExternalShutdownException:
        # Our handler already published the stop and called shutdown; spin
        # unwinding this way is the expected path, not a crash.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
