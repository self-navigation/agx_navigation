"""Vector-field guided indirect-method PMP planner for a skid-steer
platform, modeled in wheel space.

Solves the optimal-control problem via Pontryagin's Maximum Principle:
the Hamiltonian, costate ODEs and the optimal-control law are derived
analytically; the resulting two-point boundary value problem (TPBVP)
is integrated with scipy.integrate.solve_bvp.

Model -- 5D wheel-space skid-steer with bounded per-wheel acceleration:
  state    x = (p_x, p_y, theta, w_l, w_r)
  control  u = (a_l, a_r),  |a_i| <= a_wheel_max
  dynamics p_x_dot = v cos(theta), p_y_dot = v sin(theta),
           theta_dot = omega,
           w_l_dot = a_l, w_r_dot = a_r
  derived  v     = c_v * (w_l + w_r),   c_v = wheel_radius / 2
           omega = c_w * (w_r - w_l),   c_w = wheel_radius / track_eff
           track_eff = track * slip_chi

w_l, w_r are the LEFT/RIGHT WHEEL-PAIR angular speeds. The four
physical wheels collapse to two controls by construction: same-side
wheels share the longitudinal contact velocity under the no-slip
rolling constraint, the symmetric effort cost makes their costates
(and hence controls) identical, and a same-side front/rear split
produces no net body wrench at first order on homogeneous terrain.
The planner therefore lives in the 2D controllable quotient; per-wheel
freedom only matters when terrain heterogeneity breaks the symmetry,
which is exactly the residual a downstream corrector exists to absorb.

Lateral skid is lumped into the kinematics: rotation behaves as if
the wheels were slip_chi * track apart (Mandow-style effective track,
slip_chi = 1 / chassis_gain_omega of the old feedforward inversion).
There is no publication-side gain anymore -- applying both would
correct the slip twice.

The published command is the BVP-planned wheel-speed STATE at the
control tick: a velocity setpoint that already respects the
acceleration bounds, so velocity_controllers/JointGroupVelocityController
has nothing to fight. Optionally a first-order lead compensates a
wheel-velocity tracking lag:

  w_cmd_i(t) = w_state_i(t) + tau_wheel * a_i*(t)

tau_wheel = 0 (default) is correct for gz_ros2_control's velocity
interface, which tracks within a physics step. The old body-level
chassis lag (tau_omega ~ 0.3 s, dominated by lateral friction) has no
per-wheel representation; that transient is part of the corrector's
residual now. Commands are clipped to wheel_cmd_max (the joint
<limit velocity>) and pass a body-space deadzone (reconstruct (v,
omega) from the commands, flush near-zeros, map back) so stationary
phases publish exact zeros.

Cost:
  L(x, u) = alpha_t + L_pos(T(p))                           # piecewise C^1 pot.
          + w_F * w_h * (1 - F_unit(p) . h(theta))          # field alignment (faded)
          + (1 - w_F) * (1/2) * w_h * (theta - theta_p)^2   # goal-yaw spring (anti-faded)
          + (1/2) * w_v * (v - v_ref_eff(p, theta))^2       # speed reference
          + (1/2) * w_brake * (1 - F_unit . h)^2 * v^2      # heading-coupled brake
          + (1/2) * w_omega_run * omega^2                   # state-omega regularizer
          + (1/2) * w_v_barrier     * max(0,|v|-v_max)^2     # soft v_max barrier
          + (1/2) * w_omega_barrier * max(0,|w|-w_max)^2     # soft omega_max barrier
          + (1/2) * w_wheel_barrier * sum_i max(0,|w_i|-w_wheel_max)^2  # joint limit
          + (1/2) * gamma_wheel * (a_l^2 + a_r^2)            # per-wheel effort

  with v, omega the DERIVED body velocities above. Linear and angular
  authority share one per-wheel budget: the old independent (a_max,
  alpha_max) corner is deliberately infeasible, as it is on the platform.

  L_pos(T) = (beta/2) * T^2 / T_horizon              if T <= T_horizon
           = beta * (T - T_horizon/2)                if T >  T_horizon
  (Gradient = beta * min(T, T_horizon) * grad(T) / T_horizon, C^0 at
   the join. Fades to zero at the goal sink so braking is governed by
   v_ref rather than residual position pull.)

  Phi(x_T) = (1/2) * w_T_terminal * T_lin(p_T)^2            # Lyapunov in T-space
           + (1/2) * w_pp * ||p_T - p_pursuit||^2           # isotropic stabilizer
           + (1/2) * w_th * (theta_T - theta_pursuit)^2     # yaw basin
           + (1/2) * w_v_terminal * v_T^2                   # stop in v
           + (1/2) * w_omega_terminal * omega_T^2           # stop in omega

with
  v_ref(p)        = v_max * tanh(||p - p_goal|| / L_brake)
  gate(x)         = ((1 + x) / 2) ** p_gate    in [0, 1]
  v_ref_eff(p,th) = v_ref(p) * gate(F_unit . h(theta))
  T_lin(p)        = T_ref - F_ref . (p - p_pursuit)
                   (linearization of T around p_pursuit; long-range pull
                    along -F_ref that complements the running L_pos)

Hamiltonian (minimum-principle convention):
  H = L + lambda_x * v cos(theta) + lambda_y * v sin(theta) + lambda_th * omega
        + lambda_wl * a_l
        + lambda_wr * a_r

Closed-form optimal control (tanh-saturated to bounds):
  a_l* = -lambda_wl / gamma_wheel   (sat |a_l| <= a_wheel_max)
  a_r* = -lambda_wr / gamma_wheel   (sat |a_r| <= a_wheel_max)

Costate ODEs (lambda_dot = -dH/dx). The pose costates are unchanged
from the unicycle model (they involve only body-space quantities, with
v and omega now derived states):
  gate'(x)   = (p_gate / 2) * ((1 + x) / 2) ** (p_gate - 1)
  cross_F_h  = F_x sin(theta) - F_y cos(theta)
  lambda_x_dot     = -beta * min(T, T_horizon) * dT/dx / T_horizon
  lambda_y_dot     = -beta * min(T, T_horizon) * dT/dy / T_horizon
  lambda_th_dot    = -w_F * w_h * cross_F_h
                     - (1 - w_F) * w_h * (theta - theta_pursuit)
                     - w_v * v_ref * (v - v_ref_eff) * gate'(F . h) * cross_F_h
                     - w_brake * (1 - F . h) * v^2 * cross_F_h
                     + pos_gate * (lambda_x * v sin(theta) - lambda_y * v cos(theta))
  (pos_gate = ((1 + F.h)/2) ** align_gate_power gates the position-
   costate coupling into lambda_th -- the anti-understeer fix; see
   PMPShootingSolver._ode.)

The wheel costates are the chain-rule images of the old (lambda_v,
lambda_omega) through (v, omega) = A (w_l, w_r), lambda_w = A^T
lambda_(v,omega). With the body-space Hamiltonian partials
  Hv  = w_v * (v - v_ref_eff) + w_brake * (1 - F . h)^2 * v
        + lambda_x cos(theta) + lambda_y sin(theta)
        + w_v_barrier * sign(v) * max(0, |v| - v_max)
  Hom = w_omega_run * omega + lambda_th
        + w_omega_barrier * sign(omega) * max(0, |omega| - omega_max)
the wheel costate ODEs are
  lambda_wl_dot = -(c_v * Hv - c_w * Hom)
                  - w_wheel_barrier * sign(w_l) * max(0, |w_l| - w_wheel_max)
  lambda_wr_dot = -(c_v * Hv + c_w * Hom)
                  - w_wheel_barrier * sign(w_r) * max(0, |w_r| - w_wheel_max)
  # No self-coupling on either wheel speed: both are integrators of
  # bounded controls (no first-order driver lag), so dH/dw_i has no
  # -lambda_wi term.

Boundary conditions:
  t = 0 :  x(0) = x_now     (pose from TF; wheel speeds from the /odom
                             twist through body_to_wheels -- the model's
                             OWN inverse kinematics, not raw /joint_states,
                             whose implied yaw rate goes through the
                             physical track and contradicts track_eff)
  t = T :  lambda_x(T)     = -w_T_terminal * T_lin * F_ref_x
                             + w_pp * (p_x_T - p_x_pursuit)
           lambda_y(T)     = -w_T_terminal * T_lin * F_ref_y
                             + w_pp * (p_y_T - p_y_pursuit)
           lambda_th(T)    = w_th * (theta_T - theta_pursuit)
           lambda_wl(T)    = c_v * w_v_terminal * v_T
                             - c_w * w_omega_terminal * omega_T
           lambda_wr(T)    = c_v * w_v_terminal * v_T
                             + c_w * w_omega_terminal * omega_T

Operating modes (selected by the `mode` parameter at launch):

  - "online" (default): a control_rate-Hz timer solves the local BVP
    each tick and publishes a Float64MultiArray wheel command on
    wheel_cmd_topic.

  - "offline": exposes a ROS2 action server `pmp_planner/plan_to_goal`
    (PlanToGoal.action). The client (typically the trajectory interpreter)
    supplies start_pose and target_pose inline; the server rolls out a
    complete start-to-goal trajectory by repeated BVP solves, streaming
    each committed dt_segment-second chunk as action feedback: planned
    poses, per-side wheel-speed setpoints, optimal wheel accelerations,
    and the PMP costates along the nominal (the gradient of the
    segment's cost-to-go -- the quantity a neighboring-extremal or
    learned corrector needs and cannot reconstruct downstream). The
    goal carries a 3D start pose (x, y, theta); wheel speeds are
    zero-initialized (planning from rest). The result signals
    end-of-trajectory (success / abort / preempt). A new goal arriving
    mid-rollout preempts the current one server-side: the in-flight
    rollout is woken via _exec_stop, returns "preempted", and the next
    goal proceeds once the previous releases _exec_lock. Replan
    triggering (path-masked field-change detection) lives in the
    interpreter -- the planner is a pure (start, goal, field) ->
    trajectory function in this mode.

Node API: ONLINE mode subscribes to /odom, /goal_pose,
/vector_field/planner_data and publishes Float64MultiArray on
wheel_cmd_topic (default /wheel_velocity_controller/commands), data
layout [w_fl, w_rl, w_fr, w_rr] = [w_l, w_l, w_r, w_r] matching the
controller's joint order. OFFLINE mode subscribes only to
/vector_field/planner_data and serves the action
`pmp_planner/plan_to_goal`. Both modes publish a nav_msgs/Path on
/pmp_planner/trajectory (online: latest BVP horizon; offline:
cumulative rolled-out trajectory).

JointGroupVelocityController is a forward controller: it LATCHES the
last received command. Every terminal path (goal reached, waiting
states, BVP failure, node shutdown) therefore publishes an explicit
zero command -- silence would keep the wheels spinning.
"""

from dataclasses import dataclass
from math import hypot, pi
from typing import Optional
import threading
import time
from time import perf_counter

import numpy as np

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException

from agx_planning_msgs.action import PlanToGoal
from agx_planning.utils import declare_and_load_dataclass, GeneratorReturnCatcher
from agx_planning.vector_field import VectorFieldGrid
from agx_planning.pmp_planner import (
    PMPShootingSolver,
    PlannerConfig,
    TurnDiagnosticLogger,
    RolloutChunk,
    RolloutResult,
    compute_diag_values,
    goal_reached,
    parse_field_array,
    rollout_generator,
)


@dataclass
class NodeConfig:
    map_frame: str = "map"
    robot_frame: str = "base_link"
    # Topic of the velocity_controllers/JointGroupVelocityController
    # command subscriber. The message is a Float64MultiArray whose data
    # order must match the controller's `joints` parameter:
    #   [front_left, rear_left, front_right, rear_right]
    # The planner publishes [w_l, w_l, w_r, w_r].
    wheel_cmd_topic: str = "/wheel_velocity_controller/commands"
    # Set to a file path (e.g. /tmp/pmp_diag.csv) to enable the diagnostic
    # logger. Empty string disables it. The logger writes planned heading
    # profiles and actual odom to CSV for post-analysis; see TurnDiagnosticLogger.
    diag_log_path: str = ""
    # How much to wait for a vector field in offline mode before giving up.
    # Set to higher than average value,
    # because the very first message takes longer to receive.
    vector_field_timeout: float = 10.0


class PlannerNode(Node):
    """Mode-aware planner.

    Online mode (cfg.mode == "online"): a control_rate-Hz timer solves
    the local BVP each tick using the 5D initial state (pose from TF,
    wheel speeds from the /odom twist via body_to_wheels) and publishes
    a Float64MultiArray wheel command on wheel_cmd_topic.

    Offline mode (cfg.mode == "offline"): exposes a ROS2 action server
    `pmp_planner/plan_to_goal`. Each goal carries an explicit
    (start_x, start_y, start_theta) and (target_x, target_y, target_theta);
    wheel speeds are zero-initialized (planning from rest). Each committed
    dt_segment-second BVP segment is streamed back as action feedback. A
    new goal arriving mid-rollout preempts the current one: _action_handle_accepted
    fires _exec_stop, the in-flight rollout exits with "preempted", and the
    new goal then runs once it acquires _exec_lock. Path-masked replan
    detection lives in the interpreter, not here -- the planner only sees
    fresh action goals.

    NOTE: offline mode requires a MultiThreadedExecutor in main() so that
    (a) two execute callbacks can coexist (outgoing one returning + incoming
    one waiting on _exec_lock), and (b) field subscription delivery is not
    blocked during long BVP solves.
    """

    def __init__(self):
        super().__init__("pmp_planner")

        self.cfg = declare_and_load_dataclass(self, PlannerConfig())
        self.node_cfg = declare_and_load_dataclass(self, NodeConfig())

        if self.cfg.mode not in ("online", "offline"):
            raise ValueError(
                f"PlannerConfig.mode must be 'online' or 'offline', "
                f"got {self.cfg.mode!r}"
            )

        # --- Shared state (both modes) ---
        self._field = VectorFieldGrid()

        # _field_lock / _field_event are used by _on_field in both modes:
        # online mode relies on the GIL for atomicity but still needs the
        # lock so _on_field has a single unconditional code path; offline
        # mode additionally waits on _field_event before starting a rollout.
        self._field_lock = threading.Lock()
        self._field_event = threading.Event()

        self._diag_logger: Optional[TurnDiagnosticLogger] = None
        if self.node_cfg.diag_log_path:
            try:
                self._diag_logger = TurnDiagnosticLogger(self.node_cfg.diag_log_path)
                self.get_logger().info(
                    f"Diagnostic logger active -> {self.node_cfg.diag_log_path}"
                )
            except OSError as e:
                self.get_logger().error(f"Cannot open diag log: {e}")

        self._solver = PMPShootingSolver(self.cfg, self._field)

        # --- Subscriptions / publishers shared across both modes ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(
            Float32MultiArray, "/vector_field/planner_data", self._on_field, qos
        )
        self._traj_pub = self.create_publisher(Path, "/pmp_planner/trajectory", 10)

        # --- Mode-specific setup ---
        if self.cfg.mode == "online":
            self._init_online(qos)
        else:
            self._init_offline()

        self.get_logger().info(f"Planner running in '{self.cfg.mode}' mode.")

    def _init_online(self, qos: QoSProfile):
        """Online-mode wiring: TF, /odom, /goal_pose, wheel command
        publisher, control timer. The planner is its own control loop
        here -- it publishes wheel-group velocity setpoints directly to
        the JointGroupVelocityController."""
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._xi: np.ndarray = np.zeros(3)  # (px, py, theta) from TF
        self._chassis_twist: np.ndarray = np.zeros(2)  # (v, omega) from /odom
        self._goal: Optional[np.ndarray] = None  # (gx, gy, gtheta)
        self._waiting_for_field = False

        # Tracks whether the previous control cycle was inside the
        # position-tolerance ball around the goal. The BVP cost landscape
        # is qualitatively different inside vs outside, so warm-starting
        # across the boundary lands Newton in the wrong basin.
        self._was_in_goal_zone: bool = False

        self.create_subscription(Odometry, "/odom", self._on_odom, qos)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, qos)

        self._cmd_pub = self.create_publisher(
            Float64MultiArray, self.node_cfg.wheel_cmd_topic, 10
        )
        # Used to publish the empty-frame_id sentinel on goal completion.
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)

        self.create_timer(1.0 / self.cfg.control_rate, self._control_loop)
        self.get_logger().info(
            f"Indirect-PMP planner running ONLINE at "
            f"{self.cfg.control_rate} Hz, horizon {self.cfg.T_horizon}s "
            f"/ {self.cfg.N + 1} mesh nodes; wheel commands on "
            f"'{self.node_cfg.wheel_cmd_topic}'."
        )

    def _init_offline(self):
        """Offline-mode wiring: action server, exec lock/stop, trajectory_id.

        No TF, no /odom, no wheel publisher, no /goal_pose -- the action
        goal carries start and target inline. The interpreter (action
        client) owns chassis-pose snapshots and goal-source subscriptions;
        the executor owns wheel-command publication.
        """
        # _exec_lock serialises rollouts so a preempting goal waits for
        # the previous to release before starting. _exec_stop is the
        # signal that wakes a still-running rollout: _action_handle_accepted
        # sets it on a new goal arriving, _do_rollout_action's per-iter
        # check sees it and exits with "preempted".
        self._exec_lock = threading.Lock()
        self._exec_stop = threading.Event()
        self._trajectory_id: int = 0

        # ReentrantCallbackGroup so (a) two execute callbacks (the
        # outgoing one returning + the incoming one waiting on
        # _exec_lock) can coexist on different threads, and (b) a long
        # BVP solve in the execute callback doesn't block /vector_field
        # subscription delivery, which lives in the default
        # mutually-exclusive group. Together with MultiThreadedExecutor
        # in main(), field updates flow through during long solves.
        self._action_cb_group = ReentrantCallbackGroup()

        # Feedback QoS: the planner solves BVPs much faster than the
        # chassis plays them back (a 30-second sim trajectory at
        # dt_segment=1.25s = ~24 segments solved in well under a second
        # of wall clock), so the feedback queue fills with many unconsumed
        # chunks during the burst. depth=64 prevents drops that would
        # manifest as missing samples and incomplete path coverage.
        feedback_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=64,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._action_server = ActionServer(
            self,
            PlanToGoal,
            "pmp_planner/plan_to_goal",
            execute_callback=self._action_execute,
            goal_callback=self._action_goal_callback,
            cancel_callback=self._action_cancel_callback,
            handle_accepted_callback=self._action_handle_accepted,
            callback_group=self._action_cb_group,
            feedback_pub_qos_profile=feedback_qos,
        )
        self.get_logger().info(
            f"Indirect-PMP planner running OFFLINE; action "
            f"'pmp_planner/plan_to_goal'; horizon {self.cfg.T_horizon}s, "
            f"dt_segment {self.cfg.dt_segment}s, chunk samples at "
            f"{self.cfg.control_rate} Hz."
        )

    def destroy_node(self):
        # Wake any in-flight rollout so it can return promptly.
        if self.cfg.mode == "offline" and hasattr(self, "_exec_stop"):
            self._exec_stop.set()
        # Online mode: the forward controller latches the last command,
        # so a node going down mid-motion would leave the wheels spinning
        # at the last setpoint. Best-effort zero on the way out.
        if self.cfg.mode == "online" and hasattr(self, "_cmd_pub"):
            try:
                self._publish_wheel_cmd(0.0, 0.0)
            except Exception:
                pass  # context already shut down -- nothing left to do
        if self._diag_logger is not None:
            self._diag_logger.close()
        super().destroy_node()

    # ---------------- Subscriptions ----------------

    def _on_odom(self, msg: Odometry):
        # Online-only callback (subscription is created only in _init_online).
        # Odom serves as the control tick AND as the source of the measured
        # chassis twist (v, omega) -- the planner maps these through
        # body_to_wheels and pins the result as the wheel-speed initial
        # conditions on the 5D BVP, so the trajectory starts from the
        # platform's actual instantaneous velocity rather than assuming it
        # can be commanded discontinuously. The twist route (rather than
        # raw /joint_states wheel velocities) is deliberate: the model's
        # kinematics use track_effective, so consistency demands the
        # model's own inverse; raw wheel speeds imply a yaw rate through
        # the PHYSICAL track and would contradict it. The pose itself is
        # read via TF (map -> base_link), since /odom may be in a
        # different frame.
        try:
            t = self._tf_buffer.lookup_transform(
                self.node_cfg.map_frame,
                self.node_cfg.robot_frame,
                rclpy.time.Time(),
            )
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            q = t.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            # Atomic single-attribute rebind: the control loop reads
            # self._xi and self._chassis_twist with single loads and
            # either gets the old or new array, never a torn write.
            self._xi = np.array([tx, ty, yaw])
            v_meas = float(msg.twist.twist.linear.x)
            w_meas = float(msg.twist.twist.angular.z)
            self._chassis_twist = np.array([v_meas, w_meas])
            if self._diag_logger is not None:
                self._diag_logger.log_odom(tx, ty, yaw, v_meas, w_meas)
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self.node_cfg.map_frame}->"
                f"{self.node_cfg.robot_frame} unavailable: {e}",
                throttle_duration_sec=2.0,
            )

    def _on_goal(self, msg: PoseStamped):
        # Online-only callback (subscription is created only in _init_online).
        # In offline mode goals arrive via the action server's PlanToGoal goals.
        # Ignore the empty-frame_id sentinel we publish on goal completion.
        if msg.header.frame_id == "":
            return
        pos = msg.pose.position
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._goal = np.array([pos.x, pos.y, yaw])
        # Drop the previous trajectory snapshot so the diff doesn't
        # try to compare against a stale path.
        self._waiting_for_field = True
        self._solver.reset_warm_start()
        self.get_logger().info(f"Goal: ({pos.x:.2f}, {pos.y:.2f}), yaw={yaw:.2f}")

    def _on_field(self, msg: Float32MultiArray):
        """Parse the field message and atomically swap in the new VectorFieldGrid.

        Layout (canonical):
          [h, w, origin_x, origin_y, resolution,
           travel_time(H*W), grad_x(H*W), grad_y(H*W), grad_mag(H*W)]
        Backward compatibility:
          - 1-channel (T only): F_unit auto-derived from -grad T.
          - 3-channel (T, gx, gy): grad_mag missing, ignored.

        Path-masked replan detection that used to live here in offline mode
        has moved to the interpreter (the action client). The planner is
        now a pure (start, goal, field) -> trajectory function; replanning
        is a fresh action goal.
        """
        data = np.asarray(msg.data, dtype=np.float32)
        new_field = parse_field_array(data, self.cfg)
        if new_field is None:
            self.get_logger().warn(
                f"Field size mismatch: got {data.size} floats, "
                f"expected header + n, 3n, or 4n body elements",
                throttle_duration_sec=5.0,
            )
            return

        with self._field_lock:
            # Atomic swap. CPython's GIL makes the bare assignment atomic, so
            # any concurrent reader (online _control_loop, or an offline rollout
            # running on the action-server thread) sees either the old or new
            # grid, never a torn update.
            self._field = new_field
            # The solver holds its own reference to the field; rebind it so
            # the next solve uses the new instance. The solver's version-counter
            # additionally drops the warm start because the new instance starts
            # at version=1, never matching the cached _last_field_version.
            self._solver.field = new_field
            self._field_event.set()

        self.get_logger().warn(
            f"Got field. Size: {self._field._tt.shape}",
            throttle_duration_sec=5.0,
        )

        # _waiting_for_field only exists in online mode.
        if self.cfg.mode == "online":
            self._waiting_for_field = False

    # ---------------- Online control loop ----------------

    def _control_loop(self):
        if self._goal is None:
            return
        if not self._field.ready:
            self.get_logger().warn(
                "Vector field not yet received -- waiting.",
                throttle_duration_sec=2.0,
            )
            return
        if self._waiting_for_field:
            self._publish_wheel_cmd(0.0, 0.0)
            return

        # Warm-start reset on goal-zone boundary crossing.
        # The BVP cost landscape is qualitatively different inside vs outside
        # (terminal pursuit collapses, theta_pursuit flips to goal_yaw, the
        # w_F fade switches), so reusing the warm start across the boundary
        # can land Newton in the wrong basin and cause oscillation or overshoot.
        d_xy = hypot(self._xi[0] - self._goal[0], self._xi[1] - self._goal[1])
        in_goal_zone = d_xy < self.cfg.goal_tolerance_xy
        if in_goal_zone != self._was_in_goal_zone:
            self._solver.reset_warm_start()
        self._was_in_goal_zone = in_goal_zone

        # goal_reached accepts N >= 3 arrays, so we can pass the 3-D pose
        # directly and avoid constructing x0 before we know we need it.
        if goal_reached(self._xi, self._goal, self.cfg):
            d_th = abs(((self._goal[2] - self._xi[2] + pi) % (2.0 * pi)) - pi)
            self._publish_wheel_cmd(0.0, 0.0)
            self._publish_empty_trajectory()
            self.get_logger().info(
                f"Goal reached (d_xy={d_xy:.3f} m, d_th={d_th:.3f} rad)."
            )
            self._clear_goal()
            return

        # Build the 5D initial state: pose from TF, wheel speeds from the
        # /odom twist via the model's inverse kinematics. The two reads
        # are GIL-atomic individually; a torn pair (e.g. pose from cycle N,
        # twist from cycle N+1) just biases the BVP initial condition by
        # one odom dt and self-corrects next solve.
        xi = self._xi
        twist = self._chassis_twist
        wl0, wr0 = self.cfg.body_to_wheels(float(twist[0]), float(twist[1]))
        x0 = np.array([xi[0], xi[1], xi[2], wl0, wr0])
        result = self._solver.solve(x0, self._goal)
        if result is None:
            self.get_logger().warn(
                f"BVP solve failed: {self._solver._last_error} -- zeroing command.",
                throttle_duration_sec=1.0,
            )
            # Zero, not silence: the forward controller would latch and
            # keep replaying the previous wheel setpoints indefinitely.
            self._publish_wheel_cmd(0.0, 0.0)
            return

        wl_cmd, wr_cmd = result
        self._publish_wheel_cmd(wl_cmd, wr_cmd)
        self._publish_trajectory()

        if self._diag_logger is not None:
            cs = self._solver._last_costate  # (m, 5): lx, ly, lth, lwl, lwr
            st = self._solver._last_state  # (m, 5): px, py, th, wl, wr
            if cs is not None and st is not None:
                lam_th_0, lam_om_0, alpha_cmd_0 = compute_diag_values(cs[0], self.cfg)
                v_prof, om_prof = self.cfg.wheels_to_body(st[:, 3], st[:, 4])
                self._diag_logger.log_plan(
                    traj_id=-1,
                    chunk=-1,
                    thetas_deg=np.degrees(st[:, 2]),
                    omegas=om_prof,
                    vs=v_prof,
                    lam_th_0=lam_th_0,
                    lam_om_0=lam_om_0,
                    alpha_cmd_0=alpha_cmd_0,
                )

    # ---------------- Action server (offline mode) ----------------

    def _action_goal_callback(self, goal_request) -> GoalResponse:
        # Accept all syntactically-valid goals; semantic validation
        # (frame_id matches map_frame, field is ready, ...) happens in
        # _action_execute_inner so we can return a meaningful result
        # message rather than just rejecting up-front.
        return GoalResponse.ACCEPT

    def _action_cancel_callback(self, goal_handle) -> CancelResponse:
        # Wake the in-flight rollout so the cancellation observed via
        # goal_handle.is_cancel_requested takes effect promptly.
        self._exec_stop.set()
        return CancelResponse.ACCEPT

    def _action_handle_accepted(self, goal_handle):
        """Called on every accepted goal. If a previous rollout is in
        flight, set _exec_stop so it exits with "preempted"; the new
        execute callback will then block briefly on _exec_lock until
        that one releases. goal_handle.execute() itself is non-blocking
        (it schedules _action_execute on a worker thread)."""
        self._exec_stop.set()
        goal_handle.execute()

    def _action_execute(self, goal_handle):
        """Execute callback wrapper: serialise rollouts via _exec_lock so
        a preempting goal cleanly waits for the previous to release
        before clearing _exec_stop and starting its own rollout."""
        with self._exec_lock:
            self._exec_stop.clear()
            return self._action_execute_inner(goal_handle)

    def _action_execute_inner(self, goal_handle):
        """Validate the goal, wait for the field, run one rollout, and
        translate the rollout's status string into the appropriate
        action terminal state.

        The trajectory_id counter is bumped here (not earlier) so a goal
        that aborts during validation gets result.trajectory_id = 0,
        which the interpreter uses to distinguish "no rollout was
        attempted" from "the rollout we were watching just ended".

        PlanToGoal carries a 3D start pose (x, y, theta). The 5D BVP also
        needs initial wheel speeds; these are zero-initialized -- planning
        from rest. The first BVP segment will converge to the correct
        velocity profile regardless.
        """
        req = goal_handle.request
        result = PlanToGoal.Result()

        # Frame validation. The planner does its math in map_frame; if
        # the client sent poses in another frame, refuse rather than
        # silently planning in the wrong frame entirely.
        if req.frame_id != self.node_cfg.map_frame:
            err = (
                f"frame_id {req.frame_id!r} does not match planner "
                f"map_frame {self.node_cfg.map_frame!r}"
            )
            self.get_logger().warn(err)
            goal_handle.abort()
            result.success = False
            result.message = err
            result.trajectory_id = 0
            return result

        # 5D initial state: 3D pose from action goal, wheel speeds zero-init.
        x0 = np.array([req.start_x, req.start_y, req.start_theta, 0.0, 0.0])
        goal = np.array([req.target_x, req.target_y, req.target_theta])
        self.get_logger().info(
            f"Action goal: start=({x0[0]:.2f}, {x0[1]:.2f}, {x0[2]:.2f}) "
            f"-> target=({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})"
        )

        # Atomically check readiness and arm the event.
        # If the field arrives between the check and the first .wait() call,
        # the event is already set and .wait() returns immediately.
        with self._field_lock:
            if not self._field.ready:
                self._field_event.clear()

        # Wait for the field if it's not yet ready. Bounded so a
        # misconfigured system (no /vector_field/planner_data publisher)
        # doesn't hang the action indefinitely.
        deadline = time.monotonic() + self.node_cfg.vector_field_timeout
        while not self._field.ready:
            if self._exec_stop.is_set():
                # Preempted by a newer goal (or shutdown) before we even
                # got a field. Honour cancel-vs-preempt distinction.
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    msg = "Cancelled while waiting for field"
                else:
                    goal_handle.abort()
                    msg = "Preempted while waiting for field"
                result.success = False
                result.message = msg
                result.trajectory_id = 0
                return result
            if time.monotonic() > deadline:
                goal_handle.abort()
                result.success = False
                result.message = "Timeout waiting for vector field"
                result.trajectory_id = 0
                return result

            # Block until the field callback signals us, but wake up periodically
            # to re-check _exec_stop and the deadline. Without this, the executor
            # thread sleeps through the subscription callback entirely.
            remaining = deadline - time.monotonic()
            self._field_event.wait(timeout=min(0.05, max(0.0, remaining)))

        self._trajectory_id += 1
        traj_id = self._trajectory_id
        # Snapshot the field for log clarity. _on_field's atomic swap may
        # rebind self._field mid-rollout; subsequent BVP segments pick up
        # whichever instance is current at that point, mirroring the original
        # behaviour.
        self._solver.field = self._field
        self._solver.reset_warm_start()

        try:
            status = self._do_rollout_action(goal_handle, traj_id, x0, goal)
        except Exception as e:
            self.get_logger().error(f"Offline rollout crashed: {e!r}")
            goal_handle.abort()
            result.success = False
            result.message = f"Rollout exception: {e!r}"
            result.trajectory_id = int(traj_id)
            return result

        if status == "success":
            goal_handle.succeed()
            result.success = True
            result.message = "Goal reached"
        elif status == "cancelled":
            goal_handle.canceled()
            result.success = False
            result.message = "Cancelled"
        elif status == "preempted":
            # New goal arrived mid-rollout. Distinct from a true failure
            # only via the message string -- both terminate as ABORTED.
            goal_handle.abort()
            result.success = False
            result.message = "Preempted"
        else:
            # "failed": BVP failure / stagnation / sim-time cap.
            goal_handle.abort()
            result.success = False
            result.message = "Plan failure (BVP / stagnation / sim-time cap)"
        result.trajectory_id = int(traj_id)
        return result

    def _do_rollout_action(
        self, goal_handle, traj_id: int, x0: np.ndarray, goal: np.ndarray
    ) -> str:
        """Thin adapter: wire ROS 2 cancel/preempt signals into rollout_generator
        and handle per-chunk publishing and diagnostics.

        Returns the RolloutResult.status string so _action_execute_inner can
        map it to the appropriate action terminal state.
        """

        def stop_fn() -> Optional[str]:
            # Cancel takes precedence: more specific terminal state than preempt.
            if goal_handle.is_cancel_requested:
                return "cancelled"
            return "preempted" if self._exec_stop.is_set() else None

        all_poses: list[np.ndarray] = []

        gen = GeneratorReturnCatcher(
            rollout_generator(self._solver, self.cfg, x0, goal, stop_fn)
        )
        for chunk in gen:
            self._handle_rollout_chunk(chunk, traj_id, all_poses, goal_handle)

        terminal: RolloutResult = gen.value
        if terminal.status == "success":
            self.get_logger().info(
                f"Offline rollout traj_id={traj_id}: {terminal.message}"
            )
            self._publish_cumulative_path(all_poses)
        else:
            self.get_logger().warn(
                f"Offline rollout traj_id={traj_id}: {terminal.message}"
            )
        return terminal.status

    def _handle_rollout_chunk(
        self,
        chunk: RolloutChunk,
        traj_id: int,
        all_poses: list[np.ndarray],
        goal_handle,
    ):
        """Process one RolloutChunk: log diagnostics, publish feedback and path."""
        if self._diag_logger is not None and chunk.costates.shape[0] > 0:
            lam_th_0, lam_om_0, alpha_cmd_0 = compute_diag_values(
                chunk.costates[0], self.cfg
            )
            # Diag CSV keeps its body-space schema: omega / v columns are
            # the body equivalents of the PUBLISHED wheel commands.
            v_cmd, om_cmd = self.cfg.wheels_to_body(
                chunk.wheel_cmds[:, 0], chunk.wheel_cmds[:, 1]
            )
            self._diag_logger.log_plan(
                traj_id=traj_id,
                chunk=chunk.chunk_idx,
                thetas_deg=np.degrees(chunk.poses[:, 2]),
                omegas=om_cmd,
                vs=v_cmd,
                lam_th_0=lam_th_0,
                lam_om_0=lam_om_0,
                alpha_cmd_0=alpha_cmd_0,
            )
        _t0 = perf_counter()
        self._publish_chunk_feedback(goal_handle, traj_id, chunk)
        _fb_ms = (perf_counter() - _t0) * 1e3

        all_poses.append(chunk.poses)
        _t1 = perf_counter()
        self._publish_cumulative_path(all_poses)
        _path_ms = (perf_counter() - _t1) * 1e3

        self.get_logger().info(
            f"chunk {chunk.chunk_idx} published:"
            f"  feedback={_fb_ms:.0f}ms"
            f"  cum_path={_path_ms:.0f}ms"
            f"  poses_total={sum(p.shape[0] for p in all_poses)}"
        )

    # ---------------- Publishing ----------------

    def _clear_goal(self):
        """Online-mode goal-completion: clear the active goal locally and
        signal it ROS-wide on /goal_pose. (Offline mode's equivalent
        signal lives in the interpreter, which reads the action result.)"""
        sentinel = PoseStamped()
        sentinel.header.stamp = self.get_clock().now().to_msg()
        sentinel.header.frame_id = ""
        self._goal_pub.publish(sentinel)

        # No lock needed: online mode uses a single-threaded executor,
        # so _clear_goal and _on_goal never run concurrently.
        self._goal = None
        self._was_in_goal_zone = False
        self._solver.reset_warm_start()

    def _publish_wheel_cmd(self, wl: float, wr: float):
        """Publish one wheel-group velocity command.

        Data order matches the controller's `joints` parameter
        [front_left, rear_left, front_right, rear_right]: each side's
        pair receives the same setpoint -- the reduction lemma made the
        front/rear split uncontrollable for the planner, so the nominal
        is symmetric by construction. A downstream corrector is free to
        split the pair when terrain breaks the symmetry.
        """
        msg = Float64MultiArray()
        msg.data = [float(wl), float(wl), float(wr), float(wr)]
        self._cmd_pub.publish(msg)

    def _publish_trajectory(self):
        """Online-mode horizon publication."""
        if self._solver._last_state is None:
            return
        now = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = now
        path.header.frame_id = self.node_cfg.map_frame
        for k in range(self._solver._last_state.shape[0]):
            x_k = self._solver._last_state[k]
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.node_cfg.map_frame
            pose.pose.position.x = float(x_k[0])
            pose.pose.position.y = float(x_k[1])
            yaw = float(x_k[2])
            pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.pose.orientation.w = float(np.cos(yaw / 2.0))
            path.poses.append(pose)
        self._traj_pub.publish(path)

    def _publish_cumulative_path(self, all_poses: list[np.ndarray]):
        """Offline-mode cumulative trajectory publication for visualization."""
        if not all_poses:
            return
        now = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = now
        path.header.frame_id = self.node_cfg.map_frame
        for block in all_poses:
            for k in range(block.shape[0]):
                pose = PoseStamped()
                pose.header.stamp = now
                pose.header.frame_id = self.node_cfg.map_frame
                pose.pose.position.x = float(block[k, 0])
                pose.pose.position.y = float(block[k, 1])
                yaw = float(block[k, 2])
                pose.pose.orientation.z = float(np.sin(yaw / 2.0))
                pose.pose.orientation.w = float(np.cos(yaw / 2.0))
                path.poses.append(pose)
        self._traj_pub.publish(path)

    def _publish_empty_trajectory(self):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.node_cfg.map_frame
        self._traj_pub.publish(msg)

    def _publish_chunk_feedback(
        self,
        goal_handle,
        traj_id: int,
        chunk: RolloutChunk,
    ):
        """Emit one trajectory chunk as PlanToGoal action feedback.

        All arrays in the chunk are parallel (row i = tick i):
          wheel_cmds (N, 2) -- published per-side setpoints [w_l, w_r]
          accels     (N, 2) -- BVP-optimal wheel accelerations [a_l*, a_r*]
          poses      (N, 3) -- planned pose [px, py, theta]
          costates   (N, 5) -- PMP costates [lx, ly, lth, lwl, lwr]

        Empty chunks are silently skipped: there's no "is_final" flag in
        the action feedback (the action result signals end-of-trajectory),
        so an empty feedback message would carry no information. The
        intra-segment-hit case in rollout_generator guards against
        ever yielding an empty truncation.
        """
        if chunk.wheel_cmds.shape[0] == 0:
            return
        fb = PlanToGoal.Feedback()
        fb.trajectory_id = int(traj_id)
        fb.chunk_index = int(chunk.chunk_idx)
        fb.dt = float(chunk.dt_sample)
        # tolist() because rosidl-generated message slots for float32[]
        # expect a Python list (or array.array), not an ndarray.
        p32 = chunk.poses.astype(np.float32)
        w32 = chunk.wheel_cmds.astype(np.float32)
        a32 = chunk.accels.astype(np.float32)
        l32 = chunk.costates.astype(np.float32)
        fb.pose_x = p32[:, 0].tolist()
        fb.pose_y = p32[:, 1].tolist()
        fb.pose_theta = p32[:, 2].tolist()
        fb.wheel_left = w32[:, 0].tolist()
        fb.wheel_right = w32[:, 1].tolist()
        fb.accel_left = a32[:, 0].tolist()
        fb.accel_right = a32[:, 1].tolist()
        fb.lam_x = l32[:, 0].tolist()
        fb.lam_y = l32[:, 1].tolist()
        fb.lam_theta = l32[:, 2].tolist()
        fb.lam_wheel_left = l32[:, 3].tolist()
        fb.lam_wheel_right = l32[:, 4].tolist()
        goal_handle.publish_feedback(fb)

    # ---------------- PMP introspection (for evaluation) ----------------

    def extract_costates(self) -> Optional[np.ndarray]:
        """Return the last costate trajectory (m, 5):
        lambda_x, lambda_y, lambda_th, lambda_wl, lambda_wr."""
        return self._solver._last_costate

    def extract_predicted_trajectory(self) -> Optional[np.ndarray]:
        """Return the last optimal state trajectory (m, 5):
        px, py, theta, w_l, w_r."""
        return self._solver._last_state
