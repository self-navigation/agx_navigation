"""Wheel-command interpreter and corrector in front of the wheel controller.

The only writer of the controller's command topic. It is the PlanToGoal
action client: it takes the target from /goal_pose and the start from TF,
buffers the streamed chunks (trajectory_buffer) and plays the per-side wheel
setpoints out at the planned sample rate. Every command goes through
_emit() -> _correct(), currently the identity [w_l, w_r] -> [w_l, w_l, w_r,
w_r]. The controller latches its last command, so every terminal path
publishes an explicit zero. With supervisor.enabled, each playback tick first
asks SupervisorCore for re-timing, a wall speed scale, a stop or a re-join.
"""

import dataclasses
import math
import signal
from concurrent.futures import ThreadPoolExecutor
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
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Imu, JointState
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
from agx_planning.runtime_corrector import tvlqg as tvlqg_mod
from agx_planning.runtime_corrector import tvlqr as tvlqr_mod
from agx_planning.supervisor.core import SupervisorConfig, SupervisorCore
from agx_planning.supervisor.recovery import backoff_and_face
from agx_planning.supervisor.walls import SignedWallDistance, rejoin_clear
from agx_planning.utils import declare_and_load_dataclass

from .trajectory_buffer import TrajectoryBuffer

_MARKER_LIFETIME = BuiltinDuration(sec=1, nanosec=0)

# Joint order expected by wheel_velocity_controller.yaml:
#   [front_left, rear_left, front_right, rear_right] = [w_l, w_l, w_r, w_r]


class WheelCorrectorNode(Node):
    """Interprets PMP planner output into four-wheel velocity commands."""

    def __init__(self) -> None:
        super().__init__("wheel_corrector")

        # Only "offline" exists: drive the planner's PlanToGoal action and
        # play the result back (the online relay was removed 2026-10-06).
        self.declare_parameter("mode", "offline")
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
        if self._mode != "offline":
            raise ValueError(
                f"mode must be 'offline' (online relay removed 2026-10-06), "
                f"got {self._mode!r}"
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
        # Supervisor wall speed scale, applied to every outgoing command.
        self._sup_scale = 1.0

        # TF is needed offline (goal start pose) and for the debug state anchor.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._init_offline()

        self._init_corrector()

        self._init_supervisor()

        if self._publish_debug:
            self._init_debug()

        self.get_logger().info(
            "wheel_corrector up (mode=%s, expected_size=%d, debug=%s)"
            % (self._mode, self._expected_size, self._publish_debug)
        )

    # ----------------------- Mode setup ------------------------------------

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

        # TVLQG (tvlqg.py, CONTROL): a surface filter estimates the slide k;
        # TVLQR's sideslip model uses the plan's k until the filter rejects it,
        # then k_hat. Needs tvlqr.enabled; replaces tvlqr.sideslip_k (the
        # typed-in constant) with that k_bar.
        self._tvlqg_cfg = declare_and_load_dataclass(
            self, tvlqg_mod.TVLQGConfig(), prefix="tvlqg."
        )
        self._tvlqg_on = bool(self._tvlqr_cfg.enabled and self._tvlqg_cfg.enabled)
        self._tvlqg_filter: Optional[tvlqg_mod.SurfaceFilter] = None
        self._tvlqg_switch: Optional[tvlqg_mod.SurfaceSwitch] = None
        self._tvlqg_t: Optional[float] = None          # sim time of the filter state
        self._tvlqg_nis, self._tvlqg_gated = float("nan"), False
        self._tvlqg_cmd: Tuple[float, float] = (0.0, 0.0)   # twist applied since then
        self._gyro: Optional[Tuple[float, float]] = None    # (sim t, gyro z)
        self._tvlqg_caches: dict = {}                       # k bucket -> GainCache
        if self._tvlqg_on:
            self._tvlqg_switch = tvlqg_mod.SurfaceSwitch(
                self._tvlqg_cfg.k_plan, self._tvlqg_cfg.switch_sigma)
            self._set_surface_level(self._tvlqg_switch.k_bar)
            self.create_subscription(Imu, "/imu/data", self._on_gyro, 50)
            self.get_logger().info(
                "TVLQG surface filter ACTIVE (k_plan=%.3g, switch at %.3g sigma, "
                "control_on_estimate=%s); tvlqr.sideslip_k is overridden"
                % (self._tvlqg_cfg.k_plan, self._tvlqg_cfg.switch_sigma,
                   self._tvlqg_cfg.control_on_estimate))

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
        # The surface filter's input is what actually went out (after any
        # correction and clipping), on every path.
        self._tvlqg_set_cmd(*tvlqr_mod.wheels_to_twist(wheels[0], wheels[2], self._rl_cfg))
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
        observation are missing (no planned/actual pose yet) -- this
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

        Fails safe to the identity whenever the pose pair is missing or anything
        raises, so an unusable correction can never inject motion.
        """
        if planned_pose is None or actual_pose is None:
            return [left, left, right, right]

        kin = self._rl_cfg
        try:
            v_ref, omega_ref = tvlqr_mod.wheels_to_twist(left, right, kin)
            if self._tvlqg_on:
                actual_pose = self._tvlqg_step(actual_pose)
            err = tvlqr_mod.slip_error(tvlqr_mod.tracking_error(planned_pose, actual_pose),
                                       v_ref, omega_ref, self._tvlqr_cfg)
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
            if self._tvlqg_on:
                f = self._tvlqg_filter
                diag.k_hat, diag.sigma_k = f.k
                diag.rho_hat = f.rho[0]
                diag.k_bar = self._tvlqg_switch.k_bar
                diag.nis_pose = self._tvlqg_nis
                diag.gated = float(self._tvlqg_gated)

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

    # ----------------------- TVLQG surface filter --------------------------

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_gyro(self, msg: Imu) -> None:
        self._gyro = (self._now_s(), float(msg.angular_velocity.z))

    def _set_surface_level(self, k_bar: float) -> None:
        """TVLQR at slide k_bar (tvlqg.py, CONTROL): the crab offset uses k_bar
        as is, the gains come from a frozen-time cache per k_bucket."""
        b = self._tvlqg_cfg.k_bucket
        kq = round(k_bar / b) * b if b > 0 else k_bar
        self._tvlqr_cfg = dataclasses.replace(self._tvlqr_cfg, sideslip_k=float(k_bar))
        g = self._tvlqg_caches.get(kq)
        if g is None:
            g = tvlqr_mod.GainCache(dataclasses.replace(self._tvlqr_cfg, sideslip_k=float(kq)),
                                    self._rl_cfg.control_dt)
            self._tvlqg_caches[kq] = g
        self._tvlqr_gains = g

    def _tvlqg_advance(self, now: float) -> None:
        """Propagate the filter to ``now`` under the twist applied since its
        last step. Every emitted command (corrected or zero) ends an interval."""
        f = self._tvlqg_filter
        if f is None or self._tvlqg_t is None:
            return
        dt = now - self._tvlqg_t
        if dt > 0.0:
            # A long gap (paused sim, no playback) carries no input knowledge:
            # cap it rather than integrate a stale command for seconds.
            f.predict(self._tvlqg_cmd[0], self._tvlqg_cmd[1], min(dt, 0.5))
        self._tvlqg_t = now

    def _tvlqg_set_cmd(self, v: float, omega: float) -> None:
        if getattr(self, "_tvlqg_on", False):
            self._tvlqg_advance(self._now_s())
            self._tvlqg_cmd = (float(v), float(omega))

    def _tvlqg_step(self, actual_pose) -> Tuple[float, float, float]:
        """Filter time + measurement update at this tick, then the surface
        switch. Returns the pose the LQR should use."""
        now = self._now_s()
        if self._tvlqg_filter is None:
            self._tvlqg_filter = tvlqg_mod.SurfaceFilter(self._tvlqg_cfg, actual_pose)
            self._tvlqg_t = now
        else:
            self._tvlqg_advance(now)
        f = self._tvlqg_filter
        f.update_pose(actual_pose)
        self._tvlqg_nis, self._tvlqg_gated = f.nis, f.gated
        if self._gyro is not None and now - self._gyro[0] < 0.2:
            f.update_gyro(self._gyro[1])
        k, sk = f.k
        if self._tvlqg_switch.update(k, sk):
            self.get_logger().info(
                "TVLQG: plan's slide k = %.3f rejected (k_hat %.3f +- %.3f s); "
                "controller now follows k_hat" % (self._tvlqg_switch.k_plan, k, sk))
        if self._tvlqg_switch.active:
            self._set_surface_level(self._tvlqg_switch.k_bar)
        if self._tvlqg_cfg.control_on_estimate:
            return tuple(float(a) for a in f.pose)
        return actual_pose

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
        self._tvlqg_set_cmd(0.0, 0.0)

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
        self._sup_full_replans_goal = 0
        if self._sup is not None:
            self._sup_counts = {k: 0 for k in self._SUP_COUNTS}
            self._sup.replans = self._sup.stops = 0
        self._wall_retry_ok = 0
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
            self._sup_start()

        pose = self._robot_pose()

        # Supervisory layer (#41): may re-time or splice the buffer, and holds
        # the wheels at zero on a wall stop (the cursor does not advance then).
        if self._sup is not None and self._playing and self._supervise(pose):
            return

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

        had_overlay = self._buf.overlay_active
        # The supervisor's wall slow-down dilates the plan clock by the same
        # factor it scales the commands by (TrajectoryBuffer.advance).
        rate = self._sup_scale if self._sup is not None else 1.0
        sample = self._buf.advance(actual_xy, rate=rate)
        if self._sup is not None and had_overlay and not self._buf.overlay_active:
            # The core runs on the dilated plan clock _sup_t, NOT sim time:
            # rebasing on sim time made lag hugely negative for the rest of
            # the run, so the lag trigger never fired again (S1 re-smoke, 00219).
            self._sup.rebase(self._sup_t)
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
        s = self._sup_scale
        self._emit(sample.left * s, sample.right * s, sample.pose, pose, sample.costates)

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
        self._sup_finish(traj_id)
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

    # ----------------------- Supervisory layer (#41) ------------------------

    def _init_supervisor(self) -> None:
        """The stack-independent supervisor (agx_planning/supervisor), wired in.

        Off by default (``supervisor.enabled``), and then nothing here runs.
        When on, every playback tick asks SupervisorCore for a Decision:

        * waypoint reached -> the time cursor is re-timed to the robot's
          progress index (TrajectoryBuffer.retime);
        * hull clearance under the core's slow threshold (relative to the
          plan's own clearance, SupervisorCore docstring) -> commands AND the
          plan clock are scaled down together; at the stop threshold the
          wheels are held at zero and the cursor waits;
        * trigger (lag doubled and over the robot radius, or a wall) -> a PMP
          re-join to the next waypoint is solved OFF the executor thread
          (a failed solve takes 1-6 s), gated by wall clearance along it
          (the re-join cost ignores walls) and spliced into the buffer
          (TrajectoryBuffer.splice), where the corrector tracks it. With
          ``rejoin_wall_w_fp`` > 0 a HELD robot's re-join that fails the gate
          is re-solved with the footprint barrier (rejoin_service "Wall
          retry"); a moving robot's is not (the retry outlives
          ``rejoin_max_age``);
        * a wall stop whose re-join fails, or is blocked by the gate, falls
          back to a full PlanToGoal from the current pose (tier 3), at most
          ``_MAX_FULL_REPLANS`` times per goal before the goal is abandoned.

        Clearance is computed at the BELIEVED pose against ``/map``, so the
        layer uses only what a real robot would have. ``~/supervisor_state``
        carries Decision.as_array() followed by the adapter counters, in the
        order of _SUP_EXTRA (append-only).
        """
        self._sup_cfg = declare_and_load_dataclass(
            self, SupervisorConfig(), prefix="supervisor.")
        self._sup: Optional[SupervisorCore] = None
        if not self._sup_cfg.enabled:
            return
        if not self._wait_for_complete:
            self.get_logger().error(
                "supervisor needs wait_for_complete=true (it indexes the whole "
                "plan); supervisor DISABLED.")
            return
        self._sup = SupervisorCore(self._sup_cfg)
        self._sup_swd: Optional[SignedWallDistance] = None
        self._sup_map_stamp = -1e9
        self._wheel_speeds: Optional[Tuple[float, float]] = None
        self._rejoin_solver = None
        self._rejoin_pool = ThreadPoolExecutor(max_workers=1)
        self._rejoin_future = None
        self._rejoin_meta: dict = {}
        self._overlay_c0 = math.inf
        self._rejoin_not_before = -math.inf
        self._rejoin_retries = 0
        self._sup_full_replans_goal = 0
        self._sup_counts = {k: 0 for k in self._SUP_COUNTS}
        self._wall_retry_ok = 0
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, "/map", self._on_map, map_qos)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        self._sup_pub = self.create_publisher(Float64MultiArray, "~/supervisor_state", 10)
        c = self._sup_cfg
        self.get_logger().info(
            "supervisor ACTIVE (waypoint %.0f%%, reach %.2f m, slow %.2f m, stop "
            "%.2f m, trigger x%.1f, rejoin=%s)" % (
                100 * c.waypoint_frac, c.reach_radius, c.slow_dist, c.stop_dist,
                c.trigger_ratio, c.rejoin))

    _SUP_EXTRA = ("overlay", "pending", "rejoin_ok", "rejoin_fail", "rejoin_blocked",
                  "rejoin_stale", "full_replans", "plan_clearance",
                  # 2026-10-09: which trigger fired (_SUP_REASONS index, 0 =
                  # none) and the off-plan threshold at k* [m].
                  "trigger_reason", "offplan_at")
    _SUP_COUNTS = ("rejoin_ok", "rejoin_fail", "rejoin_blocked", "rejoin_stale",
                   "full_replans")
    _SUP_REASONS = ("", "wall", "lag", "near_wall", "offplan")
    _MAX_FULL_REPLANS = 3

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_map(self, msg: OccupancyGrid) -> None:
        # SLAM republishes the map about once a second; the EDT is not free.
        now = self._now_s()
        if self._sup_swd is not None and now - self._sup_map_stamp < 5.0:
            return
        info = msg.info
        self._sup_swd = SignedWallDistance.from_occupancy(
            msg.data, info.width, info.height, info.resolution,
            info.origin.position.x, info.origin.position.y)
        self._sup_map_stamp = now

    def _on_joint_states(self, msg: JointState) -> None:
        v = dict(zip(msg.name, msg.velocity))
        try:
            self._wheel_speeds = (
                0.5 * (v["front_left_wheel"] + v["rear_left_wheel"]),
                0.5 * (v["front_right_wheel"] + v["rear_right_wheel"]),
            )
        except KeyError:
            pass

    def _offplan_bound(self, poses, dt: float):
        """Scenario (b)'s threshold along this plan, from TVLQG's model
        (SupervisorConfig.offplan_sigma; None = the fixed offplan_dist)."""
        n_sigma = self._sup_cfg.offplan_sigma
        if n_sigma <= 0.0:
            return None
        try:
            tw = [tvlqr_mod.wheels_to_twist(s.left, s.right, self._rl_cfg)
                  for s in self._buf.plan_samples()]
            b = tvlqg_mod.deviation_bound_along_plan(
                np.asarray(poses, dtype=float), [t[0] for t in tw], [t[1] for t in tw],
                dt, self._tvlqg_cfg, self._tvlqr_cfg, n_sigma)
        except Exception as exc:  # noqa: BLE001 -- fall back, never block playback
            self.get_logger().error("offplan bound failed (%s); using offplan_dist" % exc)
            return None
        self.get_logger().info(
            "supervisor: off-plan bound %.1f sigma: median %.3f m (min %.3f, max %.3f)"
            % (n_sigma, float(np.median(b)), float(b.min()), float(b.max())))
        return b

    def _sup_start(self) -> None:
        """Playback begins: hand the plan to the core and build the re-join solver."""
        if self._sup is None:
            return
        dt = self._buf.active_dt
        poses = self._buf.plan_poses()
        # Plan clock for the core: advances by dt * speed scale per tick, the
        # same dilation the buffer's cursor gets, so a wall slow-down is not
        # charged as lag.
        self._sup_t = 0.0
        self._sup_last_now = self._now_s()
        c_plan = None
        if self._sup_swd is not None:
            c_plan = [self._sup_swd.body_clearance(*p) for p in poses]
        self._sup.set_plan(poses, [i * dt for i in range(len(poses))], 0.0,
                           clearance=c_plan, offplan_bound=self._offplan_bound(poses, dt))
        self._sup_scale = 1.0
        if c_plan is None:
            self.get_logger().warn("supervisor: no /map yet; ABSOLUTE wall "
                                   "thresholds for this plan")
        self._rejoin_future = None
        self._rejoin_solver = None
        self._rejoin_not_before = -math.inf
        self._rejoin_retries = 0
        self._recoveries = 0
        self._await_rejoin = False
        if self._sup_cfg.rejoin:
            try:
                # Lazy: scipy + the shooting solver only when re-joins are on.
                from agx_planning.pmp_planner.rejoin_service import RejoinSolver
                self._rejoin_solver = RejoinSolver(
                    self._buf.plan_samples(), self._rejoin_wall_field(), dt=dt,
                    T_w_min=self._sup_cfg.rejoin_t_min,
                    max_nodes=self._sup_cfg.rejoin_max_nodes or None,
                    retry_w_fp=self._sup_cfg.rejoin_wall_w_fp,
                    retry_fp_margin=self._sup_cfg.rejoin_wall_margin)
            except Exception as exc:  # noqa: BLE001 - re-join is optional
                self.get_logger().error("re-join solver unavailable (%s); triggers "
                                        "will only stop or replan." % exc)
        self.get_logger().info("supervisor: %d waypoints over %d samples"
                               % (self._sup.n_waypoints, len(poses)))

    def _rejoin_wall_field(self):
        """The wall-distance field for the re-join's barrier retry, built once
        per map (it is several full-map grids). None when the retry is off."""
        if self._sup_cfg.rejoin_wall_w_fp <= 0.0 or self._sup_swd is None:
            return None
        if getattr(self, "_wall_field_src", None) is not self._sup_swd:
            from agx_planning.pmp_planner import PlannerConfig
            from agx_planning.pmp_planner.rejoin_service import wall_field
            self._wall_field = wall_field(self._sup_swd,
                                          PlannerConfig(mode="offline").fp_smooth_sigma)
            self._wall_field_src = self._sup_swd
        return self._wall_field

    def _rejoin_gate(self, samples, k0: int, tgt: int, c0: float) -> bool:
        """The wall gate a re-join must pass to be spliced: per-sample
        plan-relative stop thresholds (stop_at at the plan index each sample
        stands in for, linear over k0..tgt), supervisor.walls.rejoin_clear."""
        if self._sup_swd is None:
            return True
        ks = np.linspace(k0, tgt, len(samples)).round().astype(int)
        stops = [self._sup.wall_thresholds(int(k))[1] for k in ks]
        return rejoin_clear(samples, self._sup_swd, c0, stops)

    def _supervise(self, pose) -> bool:
        """One supervisor step. True = the wheels are held this tick."""
        now = self._now_s()
        self._poll_rejoin(now)
        if self._buf.active_traj_id < 0:  # a tier-3 replan or abandon cleared it
            return True
        if pose is None:
            self._sup_scale = 1.0
            return False
        clearance = (self._sup_swd.body_clearance(*pose)
                     if self._sup_swd is not None else math.inf)
        # Dilated plan clock (see _sup_start), using last tick's scale.
        self._sup_t += (now - self._sup_last_now) * self._sup_scale
        self._sup_last_now = now
        d = self._sup.update(pose[0], pose[1], pose[2], self._sup_t, clearance=clearance)
        overlay = self._buf.overlay_active
        if d.retime_index is not None and self._playback_index == "time":
            self._buf.retime(d.retime_index)
        stop = d.stop
        # A re-join that passed the wall gate may start inside stop_at (it
        # was solved from a robot stopped at a wall). Let it back away; stop
        # only if it takes the body closer than where it started.
        if stop and overlay and clearance >= self._overlay_c0 - 0.02:
            stop = False
        elif stop and overlay:
            # The re-join is taking the body closer to the wall: abandon it.
            # Leaving it spliced would deadlock -- a held robot never
            # advances the overlay, and no new re-join starts while one plays.
            self._buf.cancel_overlay()
            overlay = False
        if self._await_rejoin and not overlay:
            # A back-off-and-face recovery has played out. Resuming the plan
            # from its held cursor would make TVLQR drive the robot from the
            # new pose straight back to where it hit (C1, plan 00291), so hold
            # until a re-join from here is spliced (or fails into a retry).
            stop = True
            if self._rejoin_future is None and now >= self._rejoin_not_before:
                self._request_rejoin(pose, dataclasses.replace(d, stop=True, reason="wall"),
                                     clearance, now)
        self._sup_scale = 1.0 if (overlay and not stop) else d.speed_scale
        if (d.trigger and not overlay and not self._await_rejoin
                and self._rejoin_future is None and now >= self._rejoin_not_before):
            self._request_rejoin(pose, d, clearance, now)
        self._publish_sup_state(d)
        if stop:
            self._publish_zero()
            return True
        return False

    def _request_rejoin(self, pose, d, clearance: float, now: float) -> None:
        self.get_logger().info(
            "supervisor trigger (%s): k*=%d wp %d/%d lag %.2f s (prev %.2f) "
            "clearance %.2f m" % (d.reason, d.k_star, d.waypoint, d.n_waypoints,
                                  d.lag_s, d.prev_lag_s, clearance))
        if self._rejoin_solver is None:
            if d.reason == "wall":
                self._full_replan("wall stop, no re-join solver")
            return
        if self._wheel_speeds is not None:
            wl, wr = self._wheel_speeds
        else:
            # No /joint_states yet: the plan's commanded speeds are the best guess.
            s = self._buf.plan_samples()[min(d.k_star, self._buf.n_samples - 1)]
            wl, wr = s.left, s.right
        if d.stop:
            wl = wr = 0.0  # the wheels are being held
        x0 = np.array([pose[0], pose[1], pose[2], wl, wr])
        tgt = self._sup.rejoin_target(d.k_star, self._buf.active_dt, self._rejoin_t_min())
        self._rejoin_meta = dict(t0=now, tgt=tgt, k0=d.k_star, c0=clearance,
                                 reason=d.reason, held=bool(d.stop), x0=x0)
        # Only a held robot may wait for the barrier retry (SupervisorConfig).
        accept = None
        if d.stop:
            k0, c0 = d.k_star, clearance
            accept = lambda smp: self._rejoin_gate(smp, k0, tgt, c0)  # noqa: E731
        self._rejoin_future = self._rejoin_pool.submit(
            self._rejoin_solver.solve, x0, tgt, d.k_star, accept)

    def _poll_rejoin(self, now: float) -> None:
        f = self._rejoin_future
        if f is None or not f.done():
            return
        self._rejoin_future = None
        meta = self._rejoin_meta
        try:
            samples = f.result()
        except Exception:  # noqa: BLE001
            samples = None
        age = now - meta["t0"]
        wall = meta["reason"] == "wall"
        if samples is None:
            self._sup_counts["rejoin_fail"] += 1
            self.get_logger().warn("re-join failed after %.2f s (%s)" % (
                age, self._rejoin_solver.last.get("status")))
            if wall:
                self._full_replan("wall stop, re-join failed")
            return
        if age > self._sup_cfg.rejoin_max_age and not meta["held"]:
            self._sup_counts["rejoin_stale"] += 1
            self._sup.latched = False  # let the next tick ask again
            return
        if not self._rejoin_gate(samples, meta["k0"], meta["tgt"], meta["c0"]):
            self._sup_counts["rejoin_blocked"] += 1
            self.get_logger().warn("re-join blocked by wall clearance")
            if wall:
                self._full_replan("wall stop, re-join blocked")
            return
        # A moving robot is `age` further along than x0: drop the samples it
        # has already lived through. A held robot has not moved.
        skip = 0 if meta["held"] else min(int(age / self._buf.active_dt),
                                          len(samples) - 1)
        self._buf.splice(samples[skip:], meta["tgt"] + 1)
        self._overlay_c0 = meta["c0"]
        self._sup_counts["rejoin_ok"] += 1
        if self._rejoin_solver.last.get("wall_retry"):
            self._wall_retry_ok += 1
        self._rejoin_retries = 0
        self._await_rejoin = False
        self.get_logger().info("re-join spliced: %d samples, resume at %d (T_w %.1f s,"
                               " %.0f ms, %s)" % (len(samples) - skip, meta["tgt"] + 1,
                                                  self._rejoin_solver.last.get("T_w", 0.0),
                                                  self._rejoin_solver.last.get("solve_ms", 0.0),
                                                  self._rejoin_solver.last.get("status")))

    def _full_replan(self, why: str) -> None:
        """Tier 3: drop the plan and request a new one from the current pose.

        With ``full_replan`` off (ours-lib) the robot stays held and the
        re-join is retried after ``rejoin_retry_s``; past
        ``rejoin_max_retries`` the goal is abandoned.
        """
        self._publish_zero()
        if not self._sup_cfg.full_replan:
            self._rejoin_retries += 1
            if self._rejoin_retries > self._sup_cfg.rejoin_max_retries:
                self.get_logger().error("supervisor: %s; %d re-join retries used, "
                                        "abandoning the goal." % (
                                            why, self._sup_cfg.rejoin_max_retries))
                self._finish()
                return
            self._rejoin_not_before = self._now_s() + self._sup_cfg.rejoin_retry_s
            self._sup.latched = False
            self._try_recovery()
            self.get_logger().warn("supervisor: %s -> holding, re-join retry %d/%d" % (
                why, self._rejoin_retries, self._sup_cfg.rejoin_max_retries))
            return
        self._sup_counts["full_replans"] += 1
        self._sup_full_replans_goal += 1
        self._rejoin_future = None
        if self._sup_full_replans_goal > self._MAX_FULL_REPLANS:
            self.get_logger().error("supervisor: %s, and %d full replans already; "
                                    "abandoning the goal." % (why, self._MAX_FULL_REPLANS))
            self._finish()
            return
        self.get_logger().warn("supervisor: %s -> full replan (%d/%d)" % (
            why, self._sup_full_replans_goal, self._MAX_FULL_REPLANS))
        self._sup_finish(self._buf.active_traj_id)
        self._buf.clear()
        self._playing = False
        self._send_action_goal()

    def _rejoin_t_min(self) -> float:
        c = self._sup_cfg
        return min(c.rejoin_t_min + self._rejoin_retries * c.rejoin_t_step,
                   max(c.rejoin_t_max, c.rejoin_t_min))

    def _try_recovery(self) -> bool:
        """Splice a back-off-and-face motion (supervisor/recovery.py) after a
        held re-join failed or was blocked. The plan cursor does not move, so
        playback would resume where it was held, so ``_await_rejoin`` holds the
        robot after the motion until a re-join from the new pose is spliced."""
        meta = self._rejoin_meta or {}
        if (self._recoveries >= self._sup_cfg.recovery_max or self._sup_swd is None
                or not meta.get("held") or meta.get("x0") is None):
            return False
        dt = self._buf.active_dt
        tgt = self._sup.rejoin_target(self._sup.k_star, dt, self._rejoin_t_min())
        kin = self._rl_cfg
        samples = backoff_and_face(
            meta["x0"][:3], self._sup.poses[tgt, :2], self._sup_swd.body_clearance,
            lambda v, w: tvlqr_mod.twist_to_wheels(v, w, kin), dt)
        if samples is None:
            self.get_logger().warn("supervisor: no clear back-off-and-face motion")
            return False
        self._recoveries += 1
        self._await_rejoin = True
        self._buf.splice(samples, self._buf.cursor)
        self._overlay_c0 = meta["c0"]
        self.get_logger().warn("supervisor: recovery %d/%d, back off and face plan "
                               "index %d (%d samples)" % (
                                   self._recoveries, self._sup_cfg.recovery_max, tgt,
                                   len(samples)))
        return True

    def _sup_finish(self, traj_id: int) -> None:
        if self._sup is None:
            return
        self._rejoin_future = None
        self._sup_scale = 1.0
        if not len(self._sup.poses):
            return
        c = self._sup_counts
        self.get_logger().info(
            "supervisor summary traj %d: waypoints %d/%d, triggers %d, wall stops %d, "
            "re-joins ok %d fail %d blocked %d stale %d, full replans %d, "
            "wall-retried ok %d" % (
                traj_id, self._sup.next_wp, self._sup.n_waypoints, self._sup.replans,
                self._sup.stops, c["rejoin_ok"], c["rejoin_fail"], c["rejoin_blocked"],
                c["rejoin_stale"], c["full_replans"], self._wall_retry_ok))

    def _publish_sup_state(self, d) -> None:
        msg = Float64MultiArray()
        extra = [float(self._buf.overlay_active), float(self._rejoin_future is not None)]
        extra += [float(self._sup_counts[k]) for k in self._SUP_COUNTS]
        extra.append(float(self._sup.c_plan[min(d.k_star, len(self._sup.c_plan) - 1)])
                     if self._sup.c_plan is not None else math.inf)
        extra.append(float(self._SUP_REASONS.index(d.reason))
                     if d.trigger and d.reason in self._SUP_REASONS else 0.0)
        extra.append(float(d.offplan_at))
        msg.data = d.as_array() + extra
        self._sup_pub.publish(msg)

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
