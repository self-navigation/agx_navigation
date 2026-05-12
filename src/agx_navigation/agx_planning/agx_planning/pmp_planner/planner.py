"""Vector-field guided indirect-method PMP planner for a unicycle.

Solves the optimal-control problem via Pontryagin's Maximum Principle:
the Hamiltonian, costate ODEs and the optimal-control law are derived
analytically; the resulting two-point boundary value problem (TPBVP)
is integrated with scipy.integrate.solve_bvp.

Model -- 5D kinematic unicycle with bounded acceleration on both channels:
  state    x = (p_x, p_y, theta, v, omega)
  control  u = (a, alpha),  |a| <= a_max, |alpha| <= alpha_max
  dynamics x_dot = v cos(theta), y_dot = v sin(theta), theta_dot = omega,
           v_dot     = a       (linear acceleration control)
           omega_dot = alpha   (angular acceleration control)

The BVP plans in "desired chassis behaviour" space: the v, omega
states represent what the body actually does, not what is commanded.
Folding an explicit first-order chassis-tracking lag into the BVP
itself produces a stiff costate ODE (eigenvalue +1/tau in the
self-coupling; over T_horizon the initial costate becomes numerically
chaotic, which the optimal-control formula then turns into bang-bang
chatter). Instead we keep the BVP simple and apply a feedforward
inversion of the chassis dynamics at the PUBLICATION step:

  cmd(t) = (desired_state(t) + tau * d(desired_state)/dt) / gain

This is exact inversion of the first-order tracker
  tau * d(actual)/dt + actual = gain * cmd
so a chassis matching that model executes cmd and produces actual(t)
= desired_state(t). Identify gain and tau from a step-response test
on the target platform (typical skid-steer: gain < 1 from lateral
slip, tau ~ 0.3-1.0 s from lateral friction); the parameters live in
PlannerConfig.chassis_gain_* / chassis_tau_*.

Publication is symmetric across modes (online cmd_vel and offline
trajectory chunks): both pass the BVP state through the inversion,
then clip to chassis_v_max / chassis_omega_max (hardware command
ceiling, NOT the BVP's v_max / omega_max which bound only the state).

Cost:
  L(x, u) = alpha_t + L_pos(T(p))                           # piecewise C^1 pot.
          + w_F * w_h * (1 - F_unit(p) . h(theta))          # field alignment (faded)
          + (1 - w_F) * (1/2) * w_h * (theta - theta_p)^2   # goal-yaw spring (anti-faded)
          + (1/2) * w_v * (v - v_ref_eff(p, theta))^2       # speed reference
          + (1/2) * w_brake * (1 - F_unit . h)^2 * v^2      # heading-coupled brake
          + (1/2) * w_omega_run * omega^2                   # state-omega regularizer
          + (1/2) * w_v_barrier     * max(0,|v|-v_max)^2     # soft v_max barrier
          + (1/2) * w_omega_barrier * max(0,|w|-w_max)^2     # soft omega_max barrier
          + (1/2) * gamma_a       * a^2                      # acceleration regularizer
          + (1/2) * gamma_alpha   * alpha^2                  # angular-accel regularizer

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
        + lambda_v * a
        + lambda_omega * alpha

Closed-form optimal control (tanh-saturated to bounds):
  a*     = -lambda_v     / gamma_a       (sat |a|     <= a_max)
  alpha* = -lambda_omega / gamma_alpha   (sat |alpha| <= alpha_max)

Costate ODEs (lambda_dot = -dH/dx), frozen-field approximation in the
position costates (dF_unit/dp and dv_ref/dp dropped):
  gate'(x)   = (p_gate / 2) * ((1 + x) / 2) ** (p_gate - 1)
  cross_F_h  = F_x sin(theta) - F_y cos(theta)
  lambda_x_dot     = -beta * min(T, T_horizon) * dT/dx / T_horizon
  lambda_y_dot     = -beta * min(T, T_horizon) * dT/dy / T_horizon
  lambda_th_dot    = -w_F * w_h * cross_F_h
                     - (1 - w_F) * w_h * (theta - theta_pursuit)
                     - w_v * v_ref * (v - v_ref_eff) * gate'(F . h) * cross_F_h
                     - w_brake * (1 - F . h) * v^2 * cross_F_h
                     + lambda_x * v sin(theta) - lambda_y * v cos(theta)
  lambda_v_dot     = -w_v * (v - v_ref_eff) - w_brake * (1 - F . h)^2 * v
                     - lambda_x cos(theta) - lambda_y sin(theta)
                     - w_v_barrier * sign(v) * max(0, |v| - v_max)
  lambda_omega_dot = -w_omega_run * omega - lambda_th
                     - w_omega_barrier * sign(omega) * max(0, |omega| - omega_max)
                     # No self-coupling on either v or omega: both are integrators
                     # of bounded controls (no first-order driver lag), so dH/dv
                     # and dH/domega have no -lambda_v / -lambda_omega terms.

w_F multiplies only the alignment cost (not speed/brake); the speed
and brake contributions to lambda_th fade naturally via v_ref -> 0
and v -> 0 near the goal, so no explicit fade on them is needed.

Boundary conditions:
  t = 0 :  x(0) = x_now             (5 components: pose from TF, twist from /odom)
  t = T :  lambda_x(T)     = -w_T_terminal * T_lin * F_ref_x
                             + w_pp * (p_x_T - p_x_pursuit)
           lambda_y(T)     = -w_T_terminal * T_lin * F_ref_y
                             + w_pp * (p_y_T - p_y_pursuit)
           lambda_th(T)    = w_th * (theta_T - theta_pursuit)
           lambda_v(T)     = w_v_terminal     * v_T
           lambda_omega(T) = w_omega_terminal * omega_T

Operating modes (selected by the `mode` parameter at launch):

  - "online" (default): a control_rate-Hz timer solves the local BVP
    each tick and publishes a Twist on /cmd_vel.

  - "offline": exposes a ROS2 action server `pmp_planner/plan_to_goal`
    (PlanToGoal.action). The client (typically the trajectory interpreter)
    supplies start_pose and target_pose inline; the server rolls out a
    complete start-to-goal trajectory by repeated BVP solves, streaming
    each committed dt_segment-second chunk as action feedback. The goal
    carries a 3D start pose (x, y, theta); v and omega are zero-initialized
    (planning from rest). The result signals end-of-trajectory (success /
    abort / preempt). A new goal arriving mid-rollout preempts the current
    one server-side: the in-flight rollout is woken via _exec_stop, returns
    "preempted", and the next goal proceeds once the previous releases
    _exec_lock. Replan triggering (path-masked field-change detection)
    lives in the interpreter -- the planner is a pure (start, goal, field)
    -> trajectory function in this mode.

Node API: ONLINE mode subscribes to /odom, /goal_pose,
/vector_field/planner_data and publishes Twist (or TwistStamped) on
/cmd_vel. OFFLINE mode subscribes only to /vector_field/planner_data
and serves the action `pmp_planner/plan_to_goal`. Both modes publish a
nav_msgs/Path on /pmp_planner/trajectory (online: latest BVP horizon;
offline: cumulative rolled-out trajectory).
"""

from dataclasses import dataclass
from math import hypot, pi, tanh
from typing import Optional
import threading
import time

import numpy as np

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException

from agx_planning_msgs.action import PlanToGoal
from agx_planning.utils import declare_and_load_dataclass
from agx_planning.pmp_planner import (
    PMPShootingSolver,
    PlannerConfig,
    VectorFieldGrid,
    TurnDiagnosticLogger,
)


@dataclass
class NodeConfig:
    map_frame: str = "map"
    robot_frame: str = "base_link"
    enable_stamped_cmd_vel: bool = False
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

    Online mode (cfg.mode == "online"): preserved from the original node --
    a control_rate-Hz timer solves the local BVP each tick using the 5D
    initial state (pose from TF, twist from /odom) and publishes a Twist
    on /cmd_vel.

    Offline mode (cfg.mode == "offline"): exposes a ROS2 action server
    `pmp_planner/plan_to_goal`. Each goal carries an explicit
    (start_x, start_y, start_theta) and (target_x, target_y, target_theta);
    v and omega are zero-initialized (planning from rest). Each committed
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

        # Diagnostic logger -- None when diag_log_path is empty.
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
        """Online-mode wiring: TF, /odom, /goal_pose, /cmd_vel, control timer.
        Behaviourally identical to the original node -- the planner is its
        own control loop here."""
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

        if self.node_cfg.enable_stamped_cmd_vel:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # Used to publish the empty-frame_id sentinel on goal completion.
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)

        self.create_timer(1.0 / self.cfg.control_rate, self._control_loop)
        self.get_logger().info(
            f"Indirect-PMP planner running ONLINE at "
            f"{self.cfg.control_rate} Hz, horizon {self.cfg.T_horizon}s "
            f"/ {self.cfg.N + 1} mesh nodes."
        )

    def _init_offline(self):
        """Offline-mode wiring: action server, exec lock/stop, trajectory_id.

        No TF, no /odom, no /cmd_vel, no /goal_pose -- the action goal
        carries start and target inline. The interpreter (action client)
        owns chassis-pose snapshots and goal-source subscriptions.
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

        self._field_lock = threading.Lock()
        self._field_event = threading.Event()

        # Feedback QoS: the planner solves BVPs much faster than the
        # chassis plays them back (a 30-second sim trajectory at
        # dt_segment=1.25s = ~24 segments solved in well under a second
        # of wall clock), so the feedback queue fills with many unconsumed
        # chunks during the burst. depth=64 prevents drops that would
        # manifest as missing twists and incomplete path coverage.
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
        if self._diag_logger is not None:
            self._diag_logger.close()
        super().destroy_node()

    # ---------------- Subscriptions ----------------

    def _on_odom(self, msg: Odometry):
        # Online-only callback (subscription is created only in _init_online).
        # Odom serves as the control tick AND as the source of the measured
        # chassis twist (v, omega) -- the planner pins these as initial
        # conditions on the 5D BVP, so the trajectory starts from the
        # platform's actual instantaneous velocity rather than assuming it
        # can be commanded discontinuously. The pose itself is read via TF
        # (map -> base_link), since /odom may be in a different frame.
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
        new_field = self._parse_field_msg(msg)
        if new_field is None:
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

    def _parse_field_msg(self, msg: Float32MultiArray) -> Optional[VectorFieldGrid]:
        """Build a fresh VectorFieldGrid from a Float32MultiArray. Returns
        None on size mismatch (logged, ignored)."""
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 5:
            return None
        h = int(data[0])
        w = int(data[1])
        ox = float(data[2])
        oy = float(data[3])
        res = float(data[4])
        n = h * w

        body = data[5:]
        if body.size == n:
            channels = 1
        elif body.size == 3 * n:
            channels = 3
        elif body.size == 4 * n:
            channels = 4
        else:
            self.get_logger().warn(
                f"Field size mismatch: got {data.size}, expected "
                f"{5 + n} or {5 + 3 * n} or {5 + 4 * n}",
                throttle_duration_sec=5.0,
            )
            return None

        T = body[0:n].reshape(h, w)
        if channels >= 3:
            Fx = body[n : 2 * n].reshape(h, w)
            Fy = body[2 * n : 3 * n].reshape(h, w)
        else:
            Fx = None
            Fy = None
        # The grad_mag channel (if present) is ignored: we re-normalize
        # (Fx, Fy) to a unit field internally with our own eps regularizer.

        new_field = VectorFieldGrid()
        new_field.update(
            T,
            Fx,
            Fy,
            ox,
            oy,
            res,
            field_eps=self.cfg.field_eps,
            align_smooth_sigma=self.cfg.align_smooth_sigma,
        )
        return new_field

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
            self._publish_twist(0.0, 0.0)
            return

        d_xy = hypot(self._xi[0] - self._goal[0], self._xi[1] - self._goal[1])
        d_th_signed = ((self._goal[2] - self._xi[2] + pi) % (2.0 * pi)) - pi
        d_th = abs(d_th_signed)

        # Drop the warm start on entering or leaving the goal-tolerance ball.
        # The BVP cost landscape is qualitatively different inside vs outside
        # (terminal pursuit collapses, theta_pursuit flips to goal_yaw, the
        # w_F fade switches), so the warm start from the wrong regime can
        # land Newton in the wrong basin and cause oscillation or overshoot.
        in_goal_zone = d_xy < self.cfg.goal_tolerance_xy
        if in_goal_zone != self._was_in_goal_zone:
            self._solver.reset_warm_start()
        self._was_in_goal_zone = in_goal_zone

        if d_xy < self.cfg.goal_tolerance_xy and d_th < self.cfg.goal_tolerance_th:
            self._publish_twist(0.0, 0.0)
            self._publish_empty_trajectory()
            self.get_logger().info(
                f"Goal reached (d_xy={d_xy:.3f} m, d_th={d_th:.3f} rad)."
            )
            self._clear_goal()
            return

        # Build the 5D initial state: pose from TF, twist from /odom.
        # The two reads are GIL-atomic individually; a torn pair (e.g.
        # pose from cycle N, twist from cycle N+1) just biases the BVP
        # initial condition by one odom dt and self-corrects next solve.
        xi = self._xi
        twist = self._chassis_twist
        x0 = np.array([xi[0], xi[1], xi[2], twist[0], twist[1]])
        result = self._solver.solve(x0, self._goal)
        if result is None:
            self.get_logger().warn(
                f"BVP solve failed: {self._solver._last_error} -- holding command.",
                throttle_duration_sec=1.0,
            )
            self._publish_twist(0.0, 0.0)
            return

        v_cmd, omega_cmd = result
        self._publish_twist(v_cmd, omega_cmd)
        self._publish_trajectory()

        if self._diag_logger is not None:
            cs = self._solver._last_costate  # (m, 5): lx, ly, lth, lv, lom
            st = self._solver._last_state  # (m, 5): px, py, th, v, om
            if cs is not None and st is not None:
                lam_th_0 = float(cs[0, 2])
                lam_om_0 = float(cs[0, 4])
                alpha_cmd_0 = float(
                    self.cfg.alpha_max
                    * tanh(-lam_om_0 / (self.cfg.gamma_alpha * self.cfg.alpha_max))
                )
                self._diag_logger.log_plan(
                    traj_id=-1,
                    chunk=-1,
                    thetas_deg=np.degrees(st[:, 2]),
                    omegas=st[:, 4],
                    vs=st[:, 3],
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
        needs initial v and omega; these are zero-initialized -- planning
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

        # 5D initial state: 3D pose from action goal, v/omega zero-init.
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
        """Roll out from x0 to goal by repeated BVP solves, publishing
        each committed segment as action feedback. Returns one of:

          "success"   -- chassis arrived at goal within tolerance,
          "preempted" -- _exec_stop fired (new goal arrived, or shutdown),
          "cancelled" -- the action client requested cancellation,
          "failed"    -- BVP failure, stagnation, or sim-time cap.

        x0 is a 5D state [px, py, theta, v, omega]. goal is 3D [gx, gy, gtheta].

        Termination paths:
          (a) state-at-iteration-boundary in goal tolerance,
          (b) any sample WITHIN a segment hits goal tolerance (truncated
              chunk, last feedback emitted),
          (c) stagnation: x_next stops making meaningful progress for
              several consecutive iterations -- defends against BVP
              quasi-fixed-point behaviour near the goal that would
              otherwise burn through max_rollout_sim_time,
          (d) sim-time cap (last-resort backstop).

        The action result carries the end-of-trajectory signal, so no
        empty is_final terminator chunk is needed (unlike the legacy
        /pmp_planner/trajectory_chunks topic). The cumulative
        /pmp_planner/trajectory Path is still published for visualization.
        """
        cfg = self.cfg
        dt_sample = 1.0 / cfg.control_rate
        # n_samples per chunk = how many ticks of control_rate-spaced
        # twists each BVP solve commits. Capped by T_horizon.
        seg_len_s = min(cfg.dt_segment, cfg.T_horizon)
        n_samples = max(1, int(round(seg_len_s / dt_sample)))

        sim_t = 0.0
        chunk_idx = 0
        state = x0.copy()
        # Cumulative pose log for visualization. Each entry is an
        # (n_samples, 3) block of [px, py, theta].
        all_poses: list[np.ndarray] = []

        # Stagnation tracking: if x_next stops making progress, the
        # rollout has hit a quasi-fixed-point (e.g. BVP is happily
        # emitting "stay where you are" because the chassis is in the
        # narrow ring just outside goal_tolerance_xy). The stagnation
        # backstop ensures finite termination.
        progress_eps = max(0.5 * cfg.goal_tolerance_xy, 5e-3)  # [m]
        near_goal_thresh = 4.0 * cfg.goal_tolerance_xy  # [m]
        stagnation_limit = 5
        prev_d_xy = float("inf")
        stagnation_count = 0

        while sim_t < cfg.max_rollout_sim_time:
            # Preempt / cancel checks BEFORE the solve, so a signal
            # arriving mid-rollout cancels the next BVP rather than
            # wasting a ~30 ms solve we'll throw away. Cancel takes
            # precedence (more specific terminal state).
            if goal_handle.is_cancel_requested:
                return "cancelled"
            if self._exec_stop.is_set():
                return "preempted"

            # Termination (a): state-at-boundary in tolerance.
            d_xy_state = hypot(state[0] - goal[0], state[1] - goal[1])
            d_th_state = abs(((goal[2] - state[2] + pi) % (2.0 * pi)) - pi)
            if (
                d_xy_state < cfg.goal_tolerance_xy
                and d_th_state < cfg.goal_tolerance_th
            ):
                self._publish_cumulative_path(all_poses)
                self.get_logger().info(
                    f"Offline rollout traj_id={traj_id} reached goal "
                    f"in {sim_t:.2f}s sim, {chunk_idx} chunks."
                )
                return "success"

            # Solve and sample one segment.
            result = self._solver.sample_committed_segment(
                state,
                goal,
                dt_sample,
                n_samples,
            )
            if result is None:
                self.get_logger().warn(
                    f"Offline BVP solve failed at sim_t={sim_t:.2f}s "
                    f"(traj_id={traj_id}, chunk={chunk_idx}): "
                    f"{self._solver._last_error}",
                )
                return "failed"

            twists, poses, x_next = result

            # Diagnostic: log planned heading profile + t=0 costates.
            if self._diag_logger is not None:
                cs = self._solver._last_costate
                if cs is not None:
                    lam_th_0 = float(cs[0, 2])
                    lam_om_0 = float(cs[0, 4])
                    alpha_cmd_0 = float(
                        cfg.alpha_max
                        * tanh(-lam_om_0 / (cfg.gamma_alpha * cfg.alpha_max))
                    )
                    self._diag_logger.log_plan(
                        traj_id=traj_id,
                        chunk=chunk_idx,
                        thetas_deg=np.degrees(poses[:, 2]),
                        omegas=twists[:, 1],
                        vs=twists[:, 0],
                        lam_th_0=lam_th_0,
                        lam_om_0=lam_om_0,
                        alpha_cmd_0=alpha_cmd_0,
                    )

            # Termination (b): any sample WITHIN this segment hits the
            # tolerance ball. Truncate the chunk to twists[0:hit] and
            # poses[0:hit] -- after the interpreter applies twists[hit-1]
            # the chassis arrives at poses[hit] which is at goal. The
            # parallel-arrays invariant (twists[i] applied at poses[i])
            # is preserved; the goal-arrival pose just isn't included
            # since no twist is applied AT it.
            hit_idx = -1
            for i in range(poses.shape[0]):
                d_xy = hypot(poses[i, 0] - goal[0], poses[i, 1] - goal[1])
                d_th = abs(((goal[2] - poses[i, 2] + pi) % (2.0 * pi)) - pi)
                if d_xy < cfg.goal_tolerance_xy and d_th < cfg.goal_tolerance_th:
                    hit_idx = i
                    break

            if hit_idx >= 1:
                tw_trunc = twists[:hit_idx]
                ps_trunc = poses[:hit_idx]
                self._publish_chunk_feedback(
                    goal_handle,
                    traj_id,
                    chunk_idx,
                    tw_trunc,
                    ps_trunc,
                    dt_sample,
                )
                all_poses.append(ps_trunc)
                self._publish_cumulative_path(all_poses)
                self.get_logger().info(
                    f"Offline rollout traj_id={traj_id} reached goal "
                    f"in {sim_t:.2f}s sim (intra-segment, chunk {chunk_idx}, "
                    f"sample {hit_idx})."
                )
                return "success"
            if hit_idx == 0:
                # poses[0] == state (BVP boundary condition pins it), so
                # this means state was at goal already -- caught by the
                # (a) check above. Falling through here is defensive only.
                return "success"

            # Normal path: emit the full chunk as action feedback.
            self._publish_chunk_feedback(
                goal_handle,
                traj_id,
                chunk_idx,
                twists,
                poses,
                dt_sample,
            )
            all_poses.append(poses)
            self._publish_cumulative_path(all_poses)

            state = x_next
            sim_t += n_samples * dt_sample
            chunk_idx += 1

            # Termination (c): stagnation. Only counts when chassis is
            # ALREADY near the goal -- far-from-goal slow progress is
            # legitimate (e.g. routed around a long obstacle) and is
            # backstopped by max_rollout_sim_time, not by this check.
            new_d_xy = hypot(state[0] - goal[0], state[1] - goal[1])
            if new_d_xy < near_goal_thresh and (prev_d_xy - new_d_xy) < progress_eps:
                stagnation_count += 1
                if stagnation_count >= stagnation_limit:
                    self.get_logger().warn(
                        f"Offline rollout traj_id={traj_id} stagnated near goal "
                        f"(d_xy={new_d_xy:.3f}m, no progress for "
                        f"{stagnation_count} iterations)."
                    )
                    return "failed"
            else:
                stagnation_count = 0
            prev_d_xy = new_d_xy

        # Termination (d): sim-time cap.
        self.get_logger().warn(
            f"Offline rollout traj_id={traj_id} exceeded "
            f"max_rollout_sim_time={cfg.max_rollout_sim_time}s; aborting."
        )
        return "failed"

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

    def _publish_twist(self, v: float, omega: float):
        if self.node_cfg.enable_stamped_cmd_vel:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.node_cfg.robot_frame
            msg.twist.linear.x = v
            msg.twist.angular.z = omega
        else:
            msg = Twist()
            msg.linear.x = v
            msg.angular.z = omega
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
        chunk_idx: int,
        twists: np.ndarray,
        poses: np.ndarray,
        dt: float,
    ):
        """Emit one trajectory chunk as PlanToGoal action feedback.

        twists shape (N, 2): [v, omega] per row.
        poses  shape (N, 3): [px, py, theta] per row, parallel to twists.

        Empty chunks are silently skipped: there's no "is_final" flag in
        the action feedback (the action result signals end-of-trajectory),
        so an empty feedback message would carry no information. The
        intra-segment-hit case in _do_rollout_action guards against
        ever calling this with an empty truncation.
        """
        if twists.shape[0] == 0:
            return
        fb = PlanToGoal.Feedback()
        fb.trajectory_id = int(traj_id)
        fb.chunk_index = int(chunk_idx)
        fb.dt = float(dt)
        # tolist() because rosidl-generated message slots for float32[]
        # expect a Python list (or array.array), not an ndarray.
        t32 = twists.astype(np.float32)
        p32 = poses.astype(np.float32)
        fb.linear_x = t32[:, 0].tolist()
        fb.angular_z = t32[:, 1].tolist()
        fb.pose_x = p32[:, 0].tolist()
        fb.pose_y = p32[:, 1].tolist()
        fb.pose_theta = p32[:, 2].tolist()
        goal_handle.publish_feedback(fb)

    # ---------------- PMP introspection (for evaluation) ----------------

    def extract_costates(self) -> Optional[np.ndarray]:
        """Return the last costate trajectory (m, 3): lambda_x, lambda_y, lambda_th."""
        return self._solver._last_costate

    def extract_predicted_trajectory(self) -> Optional[np.ndarray]:
        """Return the last optimal state trajectory (m, 3): px, py, theta."""
        return self._solver._last_state
