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

  - "offline": on a new goal (or a path-masked field change), a worker
    thread rolls out a complete start-to-goal trajectory by repeated
    BVP solves -- each segment of dt_segment seconds is committed,
    the simulated state is advanced, and the next segment is solved.
    Each committed segment is published as a PlannerTrajectoryChunk
    as soon as it is solved (direct publish from the worker thread;
    rclpy publishers are thread-safe), so the interpreter accumulates
    the full trajectory in its buffer ahead of execution. A path-masked
    field diff (max |T_new - T_old| sampled along the latest plan)
    above field_diff_threshold triggers a replan: the worker aborts,
    a new trajectory_id is started from the current TF pose, and the
    interpreter atomically switches on first-chunk-arrival. Newly
    discovered cells along the path count as +inf diff, always
    triggering replan.

Node API: subscribes to /odom, /goal_pose, /vector_field/planner_data;
publishes Twist (or TwistStamped) on /cmd_vel in online mode, or
PlannerTrajectoryChunk on /pmp_planner/trajectory_chunks in offline
mode. Both modes publish a nav_msgs/Path on /pmp_planner/trajectory
(online: latest BVP horizon; offline: cumulative rolled-out trajectory).
"""

from dataclasses import dataclass
from math import hypot, pi, tanh
from typing import Optional
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException

from agx_planning_msgs.msg import PlannerTrajectoryChunk
from agx_planning.utils import declare_and_load_dataclass
from agx_planning.pmp_planner import (
    PMPShootingSolver,
    PlannerConfig,
    VectorFieldGrid,
    TurnDiagnosticLogger,
)


@dataclass
class TopicConfig:
    map_frame: str = "map"
    robot_frame: str = "base_link"
    enable_stamped_cmd_vel: bool = False
    # Set to a file path (e.g. /tmp/pmp_diag.csv) to enable the diagnostic
    # logger. Empty string disables it. The logger writes planned heading
    # profiles and actual odom to CSV for post-analysis; see TurnDiagnosticLogger.
    diag_log_path: str = ""


class PlannerNode(Node):
    """Mode-aware planner.

    Online mode (cfg.mode == "online"): preserved from the original node --
    a control_rate-Hz timer solves the local BVP and publishes a Twist on
    /cmd_vel.

    Offline mode (cfg.mode == "offline"): a worker thread does rollout-by-
    concatenation. On goal arrival or path-masked field change, the worker
    is kicked: it reads the chassis TF pose, increments trajectory_id, and
    emits PlannerTrajectoryChunk messages on /pmp_planner/trajectory_chunks
    as fast as the BVP can solve. Each chunk is published directly from the
    worker thread (rclpy publishers are thread-safe in Jazzy). Replanning
    is signalled via _kick_event; the worker checks it between BVP
    iterations and bails out, after which the main loop re-snapshots state
    and starts a fresh trajectory_id.
    """

    def __init__(self):
        super().__init__("pmp_planner")

        self.cfg = declare_and_load_dataclass(self, PlannerConfig())
        self.topic_cfg = declare_and_load_dataclass(self, TopicConfig())

        if self.cfg.mode not in ("online", "offline"):
            raise ValueError(
                f"PlannerConfig.mode must be 'online' or 'offline', "
                f"got {self.cfg.mode!r}"
            )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._xi: np.ndarray = np.zeros(3)  # (px, py, theta) from TF
        self._chassis_twist: np.ndarray = np.zeros(2)  # (v, omega) from /odom
        self._goal: Optional[np.ndarray] = None  # (gx, gy, gtheta)
        self._field = VectorFieldGrid()
        self._waiting_for_field = False

        # Diagnostic logger -- None when diag_log_path is empty.
        self._diag_logger: Optional[TurnDiagnosticLogger] = None
        if self.topic_cfg.diag_log_path:
            try:
                self._diag_logger = TurnDiagnosticLogger(self.topic_cfg.diag_log_path)
                self.get_logger().info(
                    f"Diagnostic logger active -> {self.topic_cfg.diag_log_path}"
                )
            except OSError as e:
                self.get_logger().error(f"Cannot open diag log: {e}")

        # Online-only: tracks whether the previous control cycle was inside
        # the position-tolerance ball around the goal. The BVP cost
        # landscape is qualitatively different inside vs outside, so warm-
        # starting across the boundary lands Newton in the wrong basin.
        self._was_in_goal_zone: bool = False

        self._solver = PMPShootingSolver(self.cfg, self._field)

        # --- Offline-mode thread / sync state. Created in BOTH modes so
        # destroy_node() and _on_field's swap helper don't need to mode-
        # check; they're trivially cheap. ---
        # _kick_event: set on (a) new goal, (b) field diff > threshold,
        #              (c) shutdown. Worker checks between BVP iterations
        #              and after waking from its outer wait.
        # _stop_event: set on shutdown.
        # _state_lock: guards (self._goal, self._latest_trajectory_xy,
        #              self._trajectory_id). _xi and _field are read with
        #              GIL-atomic single-attribute loads instead.
        self._kick_event = threading.Event()
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._trajectory_id: int = 0
        # World-coord (x, y) samples of the most recently planned
        # trajectory. Used by _on_field's path-masked diff for the
        # replan trigger. Empty array = no plan to compare against.
        self._latest_trajectory_xy: np.ndarray = np.zeros((0, 2))

        # --- Subscriptions / publishers ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(Odometry, "/odom", self._on_odom, qos)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, qos)
        self.create_subscription(
            Float32MultiArray, "/vector_field/planner_data", self._on_field, qos
        )

        # /cmd_vel publisher (used in online mode; dormant in offline since
        # the interpreter is the one talking to the chassis there).
        if self.topic_cfg.enable_stamped_cmd_vel:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._traj_pub = self.create_publisher(Path, "/pmp_planner/trajectory", 10)
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)
        # Offline-mode trajectory chunks. ALL fields are set explicitly --
        # omitting `history` was observed to cause the entire profile to
        # fall back to system defaults under some rclpy/RMW combinations,
        # producing a VOLATILE+UNKNOWN-history publisher despite an
        # explicit durability= argument. The interpreter declares the
        # matching profile.
        self._chunk_pub = self.create_publisher(
            PlannerTrajectoryChunk,
            "/pmp_planner/trajectory_chunks",
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=64,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

        # --- Mode-specific setup ---
        if self.cfg.mode == "online":
            self.create_timer(1.0 / self.cfg.control_rate, self._control_loop)
            self.get_logger().info(
                f"Indirect-PMP planner running ONLINE at "
                f"{self.cfg.control_rate} Hz, horizon {self.cfg.T_horizon}s "
                f"/ {self.cfg.N + 1} mesh nodes."
            )
        else:
            self._worker = threading.Thread(
                target=self._offline_worker_loop,
                name="pmp_offline_worker",
                daemon=True,
            )
            self._worker.start()
            self.get_logger().info(
                f"Indirect-PMP planner running OFFLINE; horizon "
                f"{self.cfg.T_horizon}s, dt_segment {self.cfg.dt_segment}s, "
                f"chunk samples at {self.cfg.control_rate} Hz, "
                f"replan threshold {self.cfg.field_diff_threshold}."
            )

    def destroy_node(self):
        self._stop_event.set()
        self._kick_event.set()
        if self.cfg.mode == "offline" and hasattr(self, "_worker"):
            self._worker.join(timeout=2.0)
        if self._diag_logger is not None:
            self._diag_logger.close()
        super().destroy_node()

    # ---------------- Subscriptions ----------------

    def _on_odom(self, msg: Odometry):
        # Odom serves as the control tick AND as the source of the
        # measured chassis twist (v, omega) -- the planner pins these
        # as initial conditions on the 5D BVP, so the trajectory starts
        # from the platform's actual instantaneous velocity rather than
        # assuming it can be commanded discontinuously. The pose itself
        # is read via TF (map -> base_link), since /odom may be in a
        # different frame.
        try:
            t = self._tf_buffer.lookup_transform(
                self.topic_cfg.map_frame,
                self.topic_cfg.robot_frame,
                rclpy.time.Time(),
            )
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            q = t.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            # Atomic single-attribute rebind. The offline worker reads
            # self._xi with a single load and either gets the old or new
            # array, never a torn write. Same pattern for the twist.
            self._xi = np.array([tx, ty, yaw])
            v_meas = float(msg.twist.twist.linear.x)
            w_meas = float(msg.twist.twist.angular.z)
            self._chassis_twist = np.array([v_meas, w_meas])
            if self._diag_logger is not None:
                self._diag_logger.log_odom(tx, ty, yaw, v_meas, w_meas)
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self.topic_cfg.map_frame}->"
                f"{self.topic_cfg.robot_frame} unavailable: {e}",
                throttle_duration_sec=2.0,
            )
            # explicit return so we don't check goal on stale pose
            return

        if self.cfg.mode == "offline":
            self._check_offline_goal_reached()

    def _on_goal(self, msg: PoseStamped):
        # Ignore the sentinel we publish ourselves on goal completion.
        if msg.header.frame_id == "":
            return
        pos = msg.pose.position
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        with self._state_lock:
            self._goal = np.array([pos.x, pos.y, yaw])
            # Drop the previous trajectory snapshot so the diff doesn't
            # try to compare against a stale path.
            self._latest_trajectory_xy = np.zeros((0, 2))
        self._waiting_for_field = True
        self._solver.reset_warm_start()
        if self.cfg.mode == "offline":
            self._kick_event.set()
        self.get_logger().info(f"Goal: ({pos.x:.2f}, {pos.y:.2f}), yaw={yaw:.2f}")

    def _on_field(self, msg: Float32MultiArray):
        """Parse the field message, optionally trigger an offline replan,
        and atomically swap in the new VectorFieldGrid.

        Layout (canonical):
          [h, w, origin_x, origin_y, resolution,
           travel_time(H*W), grad_x(H*W), grad_y(H*W), grad_mag(H*W)]
        Backward compatibility:
          - 1-channel (T only): F_unit auto-derived from -grad T.
          - 3-channel (T, gx, gy): grad_mag missing, ignored.
        """
        new_field = self._parse_field_msg(msg)
        if new_field is None:
            return

        # Offline mode: path-masked diff against the OLD field. Out-of-
        # bounds cells in either grid count as +inf diff so newly-
        # discovered terrain on the planned path always trips the threshold.
        should_replan = False
        if self.cfg.mode == "offline":
            old_field = self._field  # GIL-atomic load
            with self._state_lock:
                traj_xy = self._latest_trajectory_xy
                has_goal = self._goal is not None
            if has_goal and traj_xy.shape[0] > 0 and old_field.ready:
                xs = traj_xy[:, 0]
                ys = traj_xy[:, 1]
                T_old, *_ = old_field.query_vec(xs, ys)
                T_new, *_ = new_field.query_vec(xs, ys)
                oob = (~old_field.in_bounds(xs, ys)) | (~new_field.in_bounds(xs, ys))
                delta = np.where(oob, np.inf, np.abs(T_new - T_old))
                if float(delta.max()) > self.cfg.field_diff_threshold:
                    should_replan = True
            elif has_goal and not old_field.ready:
                # First field arrival while a goal is waiting: kick the
                # worker so it can start its first rollout.
                should_replan = True

        # Atomic swap. CPython's GIL makes the bare assignment atomic, so
        # threaded readers (the offline worker) never see a torn update.
        self._field = new_field
        # The solver holds its own reference to the field; rebind it so
        # the next solve uses the new instance. Solver's version-counter
        # check additionally drops the warm start because the new
        # instance starts at version=1, never matching the cached
        # _last_field_version.
        self._solver.field = new_field
        self._waiting_for_field = False

        if should_replan:
            self._kick_event.set()

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

    # ---------------- Offline worker ----------------

    def _offline_worker_loop(self):
        """Outer loop: wait for kick (new goal / replan), then roll out
        from the current TF pose under the current goal/field. The
        rollout itself checks _kick_event between BVP iterations so a
        replan signal mid-rollout aborts immediately. After abort or
        completion, loop back and wait for the next kick.
        """
        while not self._stop_event.is_set():
            # Wait for something to do. Timeout is just a safety so we
            # periodically wake to check _stop_event.
            kicked = self._kick_event.wait(timeout=0.5)
            self._kick_event.clear()
            if self._stop_event.is_set():
                break

            if not kicked:
                continue

            with self._state_lock:
                goal = None if self._goal is None else self._goal.copy()
            if goal is None:
                continue

            field_ref = self._field  # GIL-atomic
            if not field_ref.ready:
                # Goal arrived before any field. The next field arrival
                # will kick us again.
                continue

            # 5D initial state for the rollout: pose from TF, twist
            # from /odom. The two reads are individually GIL-atomic;
            # we don't need them from the exact same odom callback,
            # since the next segment's start state comes from the BVP
            # solution itself.
            xi = self._xi
            twist = self._chassis_twist
            x0 = np.array([xi[0], xi[1], xi[2], twist[0], twist[1]])
            try:
                with self._state_lock:
                    self._trajectory_id += 1
                    traj_id = self._trajectory_id
                    self._latest_trajectory_xy = np.zeros((0, 2))
                self._solver.field = field_ref
                self._solver.reset_warm_start()
                self._do_rollout(traj_id, x0, goal)
            except Exception as e:
                self.get_logger().error(
                    f"Offline rollout crashed: {e!r}",
                )

    def _check_offline_goal_reached(self):
        """Clear the goal once the real robot arrives at it in offline mode.

        Called from _on_odom so it uses the actual TF pose, not the
        simulated state inside _do_rollout. This matches the intent in
        _do_rollout's docstring: goal-clearing belongs to whoever observes
        real chassis arrival.

        Thread safety: _on_odom is an executor callback, so this runs on
        the same thread as _on_goal -- no lock needed to read _goal for
        the None check. We snapshot it under _state_lock before the
        arithmetic to avoid a torn read from the worker thread.
        """
        with self._state_lock:
            goal = self._goal  # snapshot; None means nothing to do
        if goal is None:
            return

        xi = self._xi  # GIL-atomic single-attribute load
        d_xy = hypot(xi[0] - goal[0], xi[1] - goal[1])
        d_th = abs(((goal[2] - xi[2] + pi) % (2.0 * pi)) - pi)

        if d_xy < self.cfg.goal_tolerance_xy and d_th < self.cfg.goal_tolerance_th:
            self.get_logger().info(
                f"Offline goal reached (real pose): "
                f"d_xy={d_xy:.3f} m, d_th={d_th:.3f} rad."
            )
            self._clear_goal()

    def _do_rollout(self, traj_id: int, x0: np.ndarray, goal: np.ndarray):
        """Roll out from x0 to goal by repeated BVP solves, publishing
        each committed segment as a PlannerTrajectoryChunk.

        Aborts (without is_final) if _kick_event fires mid-rollout: the
        next iteration of the outer loop will assign a new trajectory_id
        and start fresh, so the interpreter sees the new id arrive and
        atomically switches. Aborts WITH an is_final empty chunk if the
        BVP fails or sim time exceeds max_rollout_sim_time, so the
        interpreter knows the trajectory_id is dead.

        Termination paths:
          (a) state-at-iteration-boundary in goal tolerance,
          (b) any sample WITHIN a segment hits goal tolerance (truncated
              chunk, is_final),
          (c) stagnation: x_next stops making meaningful progress for
              several consecutive iterations -- defends against BVP
              quasi-fixed-point behaviour near the goal that would
              otherwise burn through max_rollout_sim_time,
          (d) sim-time cap (last-resort backstop).

        DOES NOT clear the goal on completion: in offline mode the
        upstream vector-field generator may use the goal's existence as
        a "keep regenerating the field" signal, and clearing it would
        prevent newly-discovered obstacles along the executed path from
        propagating back into the planner. The user clears the goal
        externally when chassis arrival is observed (e.g. via TF).
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
        # Cumulative pose log for the path-masked diff trigger and for
        # the visualization Path. Each entry is a (n_samples, 3) block.
        all_poses: list[np.ndarray] = []

        # Stagnation tracking: if x_next stops making progress, the
        # rollout has hit a quasi-fixed-point (e.g. BVP is happily
        # emitting "stay where you are" because the chassis is in the
        # narrow ring just outside goal_tolerance_xy). The trace_streamline
        # pre-check prevents the most common cause of this, but a
        # tightly-tuned cost can still produce sub-tolerance drift; the
        # stagnation backstop ensures finite termination either way.
        progress_eps = max(0.5 * cfg.goal_tolerance_xy, 5e-3)  # [m]
        near_goal_thresh = 4.0 * cfg.goal_tolerance_xy  # [m]
        stagnation_limit = 5
        prev_d_xy = float("inf")
        stagnation_count = 0

        while sim_t < cfg.max_rollout_sim_time:
            # Replan / shutdown check. Done BEFORE the solve, so a kick
            # signal arriving mid-rollout cancels the next BVP rather
            # than wasting a 30 ms solve we'll throw away.
            if self._kick_event.is_set() or self._stop_event.is_set():
                # Re-set the flag so the outer loop re-enters the kicked
                # branch (clearing happened in the outer loop already).
                # Don't emit is_final -- a new traj_id is coming next.
                self._kick_event.set()
                return

            # Termination (a): state-at-boundary in tolerance.
            d_xy_state = hypot(state[0] - goal[0], state[1] - goal[1])
            d_th_state = abs(((goal[2] - state[2] + pi) % (2.0 * pi)) - pi)
            if (
                d_xy_state < cfg.goal_tolerance_xy
                and d_th_state < cfg.goal_tolerance_th
            ):
                self._publish_chunk(
                    traj_id,
                    chunk_idx,
                    np.zeros((0, 2)),
                    np.zeros((0, 3)),
                    dt_sample,
                    is_final=True,
                )
                self._publish_cumulative_path(all_poses)
                self.get_logger().info(
                    f"Offline rollout traj_id={traj_id} reached goal "
                    f"in {sim_t:.2f}s sim, {chunk_idx} chunks."
                )
                # NOTE: goal is intentionally NOT cleared here. See
                # docstring -- offline mode leaves goal-clearing to the
                # external system that observes actual chassis arrival.
                return

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
                # Emit terminal marker so the interpreter doesn't wait.
                self._publish_chunk(
                    traj_id,
                    chunk_idx,
                    np.zeros((0, 2)),
                    np.zeros((0, 3)),
                    dt_sample,
                    is_final=True,
                )
                return

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
            # is preserved, the goal-arrival pose just isn't included
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
                self._publish_chunk(
                    traj_id,
                    chunk_idx,
                    tw_trunc,
                    ps_trunc,
                    dt_sample,
                    is_final=True,
                )
                all_poses.append(ps_trunc)
                with self._state_lock:
                    if self._trajectory_id != traj_id:
                        return
                    self._latest_trajectory_xy = np.concatenate(
                        [p[:, :2] for p in all_poses],
                        axis=0,
                    )
                self._publish_cumulative_path(all_poses)
                self.get_logger().info(
                    f"Offline rollout traj_id={traj_id} reached goal "
                    f"in {sim_t:.2f}s sim (intra-segment, chunk {chunk_idx}, "
                    f"sample {hit_idx})."
                )
                return
            if hit_idx == 0:
                # poses[0] == state (BVP boundary condition pins it),
                # so this means state was at goal already -- caught by
                # the (a) check above. Falling through here is defensive
                # only; emit the terminator and return.
                self._publish_chunk(
                    traj_id,
                    chunk_idx,
                    np.zeros((0, 2)),
                    np.zeros((0, 3)),
                    dt_sample,
                    is_final=True,
                )
                return

            # Normal path: publish full chunk.
            self._publish_chunk(
                traj_id,
                chunk_idx,
                twists,
                poses,
                dt_sample,
                is_final=False,
            )

            all_poses.append(poses)
            with self._state_lock:
                # Sanity: someone else may have bumped the id while we
                # were solving (unlikely; only the worker bumps it, but
                # the lock makes the read+write of latest_trajectory_xy
                # atomic w.r.t. _on_field).
                if self._trajectory_id != traj_id:
                    return
                self._latest_trajectory_xy = np.concatenate(
                    [p[:, :2] for p in all_poses],
                    axis=0,
                )

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
                        f"{stagnation_count} iterations). Marking final."
                    )
                    self._publish_chunk(
                        traj_id,
                        chunk_idx,
                        np.zeros((0, 2)),
                        np.zeros((0, 3)),
                        dt_sample,
                        is_final=True,
                    )
                    return
            else:
                stagnation_count = 0
            prev_d_xy = new_d_xy

        # Termination (d): sim-time cap. Treat as giveup with terminator.
        self.get_logger().warn(
            f"Offline rollout traj_id={traj_id} exceeded "
            f"max_rollout_sim_time={cfg.max_rollout_sim_time}s; aborting."
        )
        self._publish_chunk(
            traj_id,
            chunk_idx,
            np.zeros((0, 2)),
            np.zeros((0, 3)),
            dt_sample,
            is_final=True,
        )

    # ---------------- Publishing ----------------

    def _clear_goal(self):
        """Clear the active goal locally and signal it ROS-wide on /goal_pose."""
        sentinel = PoseStamped()
        sentinel.header.stamp = self.get_clock().now().to_msg()
        sentinel.header.frame_id = ""
        self._goal_pub.publish(sentinel)

        with self._state_lock:
            self._goal = None
            self._latest_trajectory_xy = np.zeros((0, 2))
        self._was_in_goal_zone = False
        self._solver.reset_warm_start()

    def _publish_twist(self, v: float, omega: float):
        if self.topic_cfg.enable_stamped_cmd_vel:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.topic_cfg.robot_frame
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
        path.header.frame_id = self.topic_cfg.map_frame
        for k in range(self._solver._last_state.shape[0]):
            x_k = self._solver._last_state[k]
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.topic_cfg.map_frame
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
        path.header.frame_id = self.topic_cfg.map_frame
        for block in all_poses:
            for k in range(block.shape[0]):
                pose = PoseStamped()
                pose.header.stamp = now
                pose.header.frame_id = self.topic_cfg.map_frame
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
        msg.header.frame_id = self.topic_cfg.map_frame
        self._traj_pub.publish(msg)

    def _publish_chunk(
        self,
        traj_id: int,
        chunk_idx: int,
        twists: np.ndarray,
        poses: np.ndarray,
        dt: float,
        is_final: bool,
    ):
        """Publish one PlannerTrajectoryChunk. Called from the worker thread.

        twists shape (N, 2): [v, omega] per row.
        poses  shape (N, 3): [px, py, theta] per row, parallel to twists.
        Empty arrays are valid (is_final terminator chunks).
        """
        msg = PlannerTrajectoryChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.topic_cfg.map_frame
        msg.trajectory_id = int(traj_id)
        msg.chunk_index = int(chunk_idx)
        msg.is_final = bool(is_final)
        msg.dt = float(dt)
        # tolist() because rosidl-generated message slots for float32[]
        # expect a Python list (or array.array), not an ndarray.
        if twists.shape[0] > 0:
            t32 = twists.astype(np.float32)
            p32 = poses.astype(np.float32)
            msg.linear_x = t32[:, 0].tolist()
            msg.angular_z = t32[:, 1].tolist()
            msg.pose_x = p32[:, 0].tolist()
            msg.pose_y = p32[:, 1].tolist()
            msg.pose_theta = p32[:, 2].tolist()
        # else: leave the arrays as their default empty lists.
        self._chunk_pub.publish(msg)

    # ---------------- PMP introspection (for evaluation) ----------------

    def extract_costates(self) -> Optional[np.ndarray]:
        """Return the last costate trajectory (m, 3): lambda_x, lambda_y, lambda_th."""
        return self._solver._last_costate

    def extract_predicted_trajectory(self) -> Optional[np.ndarray]:
        """Return the last optimal state trajectory (m, 3): px, py, theta."""
        return self._solver._last_state
