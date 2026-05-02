"""Vector-field guided NMPC for a skid-steer unicycle (v4).

Key design points addressed in this revision:

  1. Stage cost has a direct positional gradient via a cross-track residual
     against a per-cycle reference path. Without this, in a uniform field
     the optimizer is free to oscillate heading within an equally-cost cone
     and the projected trajectory wanders even though the robot moves OK.

  2. The reference path is regenerated every control cycle by integrating a
     field-following inner-loop from the current pose. This puts the SQP
     warm-start far from the heading-180 saddle of the alignment cost, where
     both cross- and inner-product residuals lose their gradient and HPIPM
     fails with ACADOS_MINSTEP.

  3. Speed regulation is alignment-adapted: r_speed = v - v_ref * (F . h).
     Naturally requests v = v_ref when forward-aligned, v = 0 when heading
     is perpendicular to F (turn-in-place), and v = -v_ref when heading is
     flipped (reverse). All three modes the user wanted are expressed in one
     residual without explicit logic.

  4. Terminal cost-to-go T(p_N) is linearised around the reference terminal
     node and clipped so its squared form stays comparable in magnitude to
     the summed stage cost regardless of map size.

See planner_refactor_notes.md for the full mapping from v3.
"""

from dataclasses import dataclass
from math import hypot, tanh, atan2, pi, cos, sin
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException

import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PlannerConfig:
    # Horizon: ~2 s of preview at 20 Hz replanning. dt = 0.08 s is fine
    # for ERK4 on this 5-state explicit ODE.
    N: int = 25
    T_horizon: float = 2.0
    control_rate: float = 20.0

    # Kinematic limits. Symmetric on v: backward motion is allowed but
    # discouraged by the alignment-adapted speed reference.
    a_max: float = 1.0
    alpha_max: float = 2.0
    v_max: float = 0.5
    omega_max: float = 1.5

    # Speed reference scale. v_ref(d_goal) = v_max * tanh(d_goal / L_brake)
    # decelerates smoothly inside one L_brake of the goal.
    L_brake: float = 0.5

    # Reference-path generator (inner-loop field-follower) gains.
    ref_K_omega: float = 2.0  # P-gain on heading error
    ref_brake_cos: float = 0.0  # min cos(heading_err) before v target -> 0

    # Cap T_ref before squaring at the terminal node. Without this cap, on
    # a large map T_ref can be 20+ m and (T_lin)^2 dominates the cost by
    # orders of magnitude, suppressing the stage cost shape entirely.
    # 5 m corresponds to roughly the maximum useful preview distance at
    # v_max = 0.5 m/s over a 2 s horizon plus margin.
    T_ref_cap: float = 5.0

    # ---- Stage cost weights ----
    # Heading and lateral are the two dominant shaping terms. Their ratio
    # sets the heading-priority character; their absolute magnitudes set
    # how aggressive the planner is.
    w_heading: float = 1.5  # r1 = F x h         (cross product, signed lateral)
    w_align: float = 4.0  # r2 = 1 - F . h     (forward gating, breaks -F basin)
    w_speed: float = 1.0  # r_speed = v - v_ref * (F . h)
    w_xtrack: float = 4.0  # r_lat = perpendicular distance to reference line
    w_omega: float = 0.2  # mild yaw-rate damping
    w_a: float = 0.05  # control effort
    w_alpha: float = 0.1  # control effort

    # ---- Terminal cost weights ----
    w_T_field: float = 1.0  # squared T_lin (capped)
    w_T_v: float = 0.5  # v_N -> 0
    w_T_omega: float = 0.5  # omega_N -> 0

    goal_tolerance: float = 0.05

    @property
    def dt(self) -> float:
        return self.T_horizon / self.N


# ---------------------------------------------------------------------------
# Solver build (called once at startup, cached on disk by acados)
# ---------------------------------------------------------------------------


def build_ocp_solver(cfg: PlannerConfig) -> AcadosOcpSolver:
    # State and control symbols for the dynamic unicycle model.
    p_x = ca.SX.sym("p_x")
    p_y = ca.SX.sym("p_y")
    theta = ca.SX.sym("theta")
    v = ca.SX.sym("v")
    omega = ca.SX.sym("omega")
    x = ca.vertcat(p_x, p_y, theta, v, omega)

    a = ca.SX.sym("a")
    alpha = ca.SX.sym("alpha")
    u = ca.vertcat(a, alpha)

    # Per-stage parameters. The reference position (px_ref, py_ref) and
    # field direction (Fx, Fy) come from the field-following reference
    # generator each cycle. T_ref is the (capped) FMM travel time at
    # (px_ref, py_ref). v_ref is the goal-distance-shaped speed magnitude.
    Fx = ca.SX.sym("Fx")
    Fy = ca.SX.sym("Fy")
    v_ref = ca.SX.sym("v_ref")
    T_ref = ca.SX.sym("T_ref")
    px_ref = ca.SX.sym("px_ref")
    py_ref = ca.SX.sym("py_ref")
    p = ca.vertcat(Fx, Fy, v_ref, T_ref, px_ref, py_ref)
    n_params = 6

    x_dot = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
        a,
        alpha,
    )

    # ---------------- Stage residual y_stage (target = 0) ----------------
    cos_t = ca.cos(theta)
    sin_t = ca.sin(theta)
    F_dot_h = cos_t * Fx + sin_t * Fy  # F . heading_hat in [-1, 1]
    F_x_h = sin_t * Fx - cos_t * Fy  # F x heading_hat (2D scalar)
    dpx = p_x - px_ref
    dpy = p_y - py_ref

    y_stage = ca.vertcat(
        F_x_h,  # r1: drives heading toward F
        1.0 - F_dot_h,  # r2: breaks the heading-180 basin
        v - v_ref * F_dot_h,  # r_speed: alignment-adapted
        -Fy * dpx + Fx * dpy,  # r_lat: signed perpendicular to reference line
        omega,  # yaw-rate damping
        a,  # control effort
        alpha,  # control effort
    )

    # ---------------- Terminal residual y_terminal (target = 0) ----------------
    # T_lin is the first-order Taylor expansion of T(p) at the reference
    # terminal node, using F = -normalize(grad T):
    #   T(p_N) ~= T_ref + grad(T)|_ref . (p_N - p_ref) ~= T_ref - F . (p_N - p_ref)
    # Squaring (in NL_LS) gives a Lyapunov-like terminal weight that vanishes
    # at the goal and is obstacle-aware (T encodes the FMM-routed cost-to-go).
    # T_ref is clipped at runtime to keep the squared cost well-scaled.
    T_lin = T_ref - Fx * dpx - Fy * dpy
    y_terminal = ca.vertcat(T_lin, v, omega)

    model = AcadosModel()
    model.name = "skid_steer_vfield"
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = x_dot
    model.cost_y_expr = y_stage
    model.cost_y_expr_e = y_terminal

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = cfg.N
    ocp.solver_options.tf = cfg.T_horizon
    ocp.parameter_values = np.zeros(n_params)

    ny = y_stage.shape[0]  # 7
    ny_e = y_terminal.shape[0]  # 3
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(
        [
            cfg.w_heading,
            cfg.w_align,
            cfg.w_speed,
            cfg.w_xtrack,
            cfg.w_omega,
            cfg.w_a,
            cfg.w_alpha,
        ]
    )
    ocp.cost.W_e = np.diag(
        [
            cfg.w_T_field,
            cfg.w_T_v,
            cfg.w_T_omega,
        ]
    )
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # Bounds. Symmetric v allows reverse and turn-in-place; no |omega|<=k|v|
    # coupling -- skid-steer can rotate freely.
    ocp.constraints.lbu = np.array([-cfg.a_max, -cfg.alpha_max])
    ocp.constraints.ubu = np.array([+cfg.a_max, +cfg.alpha_max])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbx = np.array([-cfg.v_max, -cfg.omega_max])
    ocp.constraints.ubx = np.array([+cfg.v_max, +cfg.omega_max])
    ocp.constraints.idxbx = np.array([3, 4])
    ocp.constraints.lbx_e = np.array([-cfg.v_max, -cfg.omega_max])
    ocp.constraints.ubx_e = np.array([+cfg.v_max, +cfg.omega_max])
    ocp.constraints.idxbx_e = np.array([3, 4])
    ocp.constraints.x0 = np.zeros(5)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = max(1, cfg.N // 5)
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.levenberg_marquardt = 1e-4
    ocp.solver_options.print_level = 0

    return AcadosOcpSolver(ocp, json_file="skid_steer_vfield.json")


# ---------------------------------------------------------------------------
# Vector field grid sampler (unchanged from v3)
# ---------------------------------------------------------------------------


class VectorFieldGrid:
    """Bilinear sampler for the unit field (Fx, Fy) and FMM travel time T."""

    def __init__(self):
        self._gx: Optional[np.ndarray] = None
        self._gy: Optional[np.ndarray] = None
        self._tt: Optional[np.ndarray] = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._res = 1.0
        self._tt_max = 1.0
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def update(
        self,
        travel_time: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ):
        self._gx = grad_x.astype(np.float64)
        self._gy = grad_y.astype(np.float64)
        self._tt = travel_time.astype(np.float64)
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._res = resolution
        if self._tt.size:
            finite = self._tt[np.isfinite(self._tt)]
            self._tt_max = float(finite.max()) if finite.size else 1.0
        else:
            self._tt_max = 1.0
        self._ready = True

    def query(self, px: float, py: float) -> tuple[float, float, float]:
        """Bilinear (Fx, Fy, T) at world (px, py).

        F is re-normalised after interpolation: bilinear blending of unit
        vectors does not preserve unit norm, and the (1 - F . h) residual
        would otherwise have a non-zero floor at perfect heading alignment.
        """
        if not self._ready:
            return 0.0, 0.0, 0.0

        # Cell centres are at (col + 0.5, row + 0.5) * res + origin.
        u = (px - self._origin_x) / self._res - 0.5
        w = (py - self._origin_y) / self._res - 0.5

        rows, cols = self._gx.shape
        if not (0.0 <= u <= cols - 1.0 and 0.0 <= w <= rows - 1.0):
            return 0.0, 0.0, self._tt_max

        x0 = min(int(u), cols - 2)
        y0 = min(int(w), rows - 2)
        fx = u - x0
        fy = w - y0

        def _bilerp(arr: np.ndarray) -> float:
            return float(
                arr[y0, x0] * (1.0 - fx) * (1.0 - fy)
                + arr[y0, x0 + 1] * fx * (1.0 - fy)
                + arr[y0 + 1, x0] * (1.0 - fx) * fy
                + arr[y0 + 1, x0 + 1] * fx * fy
            )

        Fx = _bilerp(self._gx)
        Fy = _bilerp(self._gy)
        n = (Fx * Fx + Fy * Fy) ** 0.5
        if n > 1e-6:
            Fx /= n
            Fy /= n
        else:
            Fx, Fy = 0.0, 0.0

        return Fx, Fy, _bilerp(self._tt)


# ---------------------------------------------------------------------------
# Reference trajectory generator (field-following inner loop)
# ---------------------------------------------------------------------------


def _wrap_to_pi(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


class ReferenceGenerator:
    """Forward-integrate a feasible reference trajectory along the field.

    Heading tracks atan2(Fy, Fx) via a saturated proportional controller;
    forward speed is scaled by max(cos(heading_err), 0) so the robot
    decelerates and turns in place when sharply misaligned. The output is
    used both as the SQP warm-start and as the per-stage parameters
    (px_ref, py_ref, Fx, Fy, T_ref, v_ref) of the cost.

    The reference is regenerated every cycle from the current measured
    state, so the SQP never inherits oscillation from a previous solution
    and is always seeded close to the alignment manifold (far from the
    heading-180 saddle of the cost).
    """

    def __init__(self, cfg: PlannerConfig, field: VectorFieldGrid):
        self.cfg = cfg
        self.field = field

    def generate(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        N = cfg.N
        dt = cfg.dt
        v_max = cfg.v_max
        omega_max = cfg.omega_max
        a_max = cfg.a_max
        alpha_max = cfg.alpha_max
        L_brake = cfg.L_brake
        K_omega = cfg.ref_K_omega

        seed_x = np.zeros((N + 1, 5))
        seed_u = np.zeros((N, 2))
        seed_x[0] = x0.copy()

        gx, gy = float(goal[0]), float(goal[1])

        for k in range(N):
            px, py, th, v, om = seed_x[k]
            Fx, Fy, _ = self.field.query(px, py)
            f_norm = (Fx * Fx + Fy * Fy) ** 0.5

            if f_norm < 1e-3:
                # Field undefined here (out of map / inside obstacle / at goal).
                # Brake along all axes and hold heading; the QP can do better.
                a_cmd = max(min(-v / dt, a_max), -a_max)
                alpha_cmd = max(min(-om / dt, alpha_max), -alpha_max)
            else:
                psi_d = atan2(Fy, Fx)
                e = _wrap_to_pi(psi_d - th)

                # Saturated P on heading; deadbeat alpha to reach it.
                omega_des = max(min(K_omega * e, omega_max), -omega_max)
                alpha_cmd = max(min((omega_des - om) / dt, alpha_max), -alpha_max)

                # Forward speed is goal-distance-shaped and gated by alignment.
                # When |e| > 90 deg the cosine is negative -> v_des = 0 and the
                # robot turns in place rather than driving sideways through F.
                d_goal = hypot(px - gx, py - gy)
                v_target = v_max * tanh(d_goal / L_brake)
                v_des = v_target * max(cos(e), cfg.ref_brake_cos)
                a_cmd = max(min((v_des - v) / dt, a_max), -a_max)

            seed_u[k] = (a_cmd, alpha_cmd)
            seed_x[k + 1] = _rk4_step(seed_x[k], seed_u[k], dt, v_max, omega_max)

        return seed_x, seed_u


def _rk4_step(
    state: np.ndarray,
    u: np.ndarray,
    dt: float,
    v_max: float,
    omega_max: float,
) -> np.ndarray:
    """Single ERK4 step matching the acados internal integrator."""
    a, al = float(u[0]), float(u[1])

    def f(s: np.ndarray) -> np.ndarray:
        return np.array([s[3] * np.cos(s[2]), s[3] * np.sin(s[2]), s[4], a, al])

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    ns = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    ns[3] = float(np.clip(ns[3], -v_max, v_max))
    ns[4] = float(np.clip(ns[4], -omega_max, omega_max))
    return ns


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------


class PlannerNode(Node):

    def __init__(self):
        super().__init__("pmp_planner")
        self.cfg = PlannerConfig()

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("enable_stamped_cmd_vel", False)
        self._map_frame = self.get_parameter("map_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value
        self._stamped_cmd = self.get_parameter("enable_stamped_cmd_vel").value

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._xi: np.ndarray = np.zeros(5)
        self._goal: Optional[np.ndarray] = None
        self._field = VectorFieldGrid()
        self._waiting_for_field = False

        self.get_logger().info("Compiling acados solver...")
        self._solver = build_ocp_solver(self.cfg)
        self._reference = ReferenceGenerator(self.cfg, self._field)
        self.get_logger().info("Solver ready.")

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

        if self._stamped_cmd:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._traj_pub = self.create_publisher(Path, "/pmp_planner/trajectory", 10)

        self.create_timer(1.0 / self.cfg.control_rate, self._control_loop)
        self.get_logger().info(
            f"Running at {self.cfg.control_rate} Hz, "
            f"horizon {self.cfg.T_horizon}s / {self.cfg.N} nodes."
        )

    # ---------------- Subscriptions ----------------

    def _on_odom(self, msg: Odometry):
        v_body = msg.twist.twist.linear.x
        omega_b = msg.twist.twist.angular.z
        try:
            t = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._robot_frame,
                rclpy.time.Time(),
            )
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            q = t.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self._xi = np.array([tx, ty, yaw, v_body, omega_b])
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self._map_frame}->{self._robot_frame} unavailable: {e}",
                throttle_duration_sec=2.0,
            )

    def _on_goal(self, msg: PoseStamped):
        pos = msg.pose.position
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._goal = np.array([pos.x, pos.y, yaw])
        # Hold output until the new field arrives. Solving on the old field
        # with a new goal would steer toward the previous goal location.
        self._waiting_for_field = True
        self.get_logger().info(f"Goal: ({pos.x:.2f}, {pos.y:.2f})")

    def _on_field(self, msg: Float32MultiArray):
        # Layout: [h, w, origin_x, origin_y, resolution, T(H*W), gx(H*W), gy(H*W)]
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 5:
            return
        h = int(data[0])
        w = int(data[1])
        origin_x = float(data[2])
        origin_y = float(data[3])
        resolution = float(data[4])
        n = h * w
        if data.size != 5 + 3 * n:
            self.get_logger().warn(
                f"Field size mismatch: got {data.size}, expected {5 + 3 * n}",
                throttle_duration_sec=5.0,
            )
            return
        tt = data[5 : 5 + n].reshape(h, w)
        gx = data[5 + n : 5 + 2 * n].reshape(h, w)
        gy = data[5 + 2 * n : 5 + 3 * n].reshape(h, w)
        self._field.update(tt, gx, gy, origin_x, origin_y, resolution)
        self._waiting_for_field = False

    # ---------------- Control loop ----------------

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

        d_goal = hypot(self._xi[0] - self._goal[0], self._xi[1] - self._goal[1])
        if d_goal < self.cfg.goal_tolerance:
            self._publish_twist(0.0, 0.0)
            self._publish_empty_trajectory()
            self.get_logger().info(
                f"Goal reached (dist={d_goal:.3f} m).",
                throttle_duration_sec=1.0,
            )
            return

        if self._solve_mpc() is None:
            self._publish_twist(0.0, 0.0)
            return

        # Use the predicted state at node 1: this is one ERK4 step of the
        # solver's optimal u_0, fully consistent with its internal model.
        xi_next = self._solver.get(1, "x")
        self._publish_twist(float(xi_next[3]), float(xi_next[4]))
        self._publish_trajectory()

    # ---------------- MPC solve ----------------

    def _solve_mpc(self) -> Optional[np.ndarray]:
        """Run one SQP-RTI iteration. Returns u_0 on success, None on failure."""
        solver = self._solver
        N = self.cfg.N

        # Regenerate the reference path each cycle. This is both the SQP
        # warm-start AND the per-stage cost reference -- a self-consistent
        # bundle that the QP only needs to refine slightly.
        ref_x, ref_u = self._reference.generate(self._xi, self._goal)

        # Warm-start states and controls.
        for k in range(N + 1):
            solver.set(k, "x", ref_x[k])
        for k in range(N):
            solver.set(k, "u", ref_u[k])

        # Pin x_0 to the measured robot state.
        solver.set(0, "x", self._xi)
        solver.set(0, "lbx", self._xi)
        solver.set(0, "ubx", self._xi)

        # Per-stage parameters from the reference trajectory. F and v_ref are
        # taken at the reference position so the QP sees a coherent local
        # cost landscape; T_ref is clipped so the squared terminal cost stays
        # well-scaled regardless of map size.
        v_max = self.cfg.v_max
        L_brake = self.cfg.L_brake
        T_ref_cap = self.cfg.T_ref_cap
        gx, gy = float(self._goal[0]), float(self._goal[1])

        for k in range(N + 1):
            px_ref = float(ref_x[k][0])
            py_ref = float(ref_x[k][1])
            Fx, Fy, T_ref = self._field.query(px_ref, py_ref)
            T_ref = min(T_ref, T_ref_cap)

            d = hypot(px_ref - gx, py_ref - gy)
            v_r = v_max * tanh(d / L_brake)

            solver.set(k, "p", np.array([Fx, Fy, v_r, T_ref, px_ref, py_ref]))

        status = solver.solve()
        if status != 0:
            self.get_logger().warn(
                f"acados status {status} -- holding command.",
                throttle_duration_sec=1.0,
            )
            return None
        return solver.get(0, "u")

    # ---------------- Publishing ----------------

    def _publish_twist(self, v: float, omega: float):
        if self._stamped_cmd:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self._robot_frame
            msg.twist.linear.x = v
            msg.twist.angular.z = omega
        else:
            msg = Twist()
            msg.linear.x = v
            msg.angular.z = omega
        self._cmd_pub.publish(msg)

    def _publish_trajectory(self):
        now = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = now
        path.header.frame_id = self._map_frame
        for k in range(self.cfg.N + 1):
            xi_k = self._solver.get(k, "x")
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self._map_frame
            pose.pose.position.x = float(xi_k[0])
            pose.pose.position.y = float(xi_k[1])
            yaw = float(xi_k[2])
            pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.pose.orientation.w = float(np.cos(yaw / 2.0))
            path.poses.append(pose)
        self._traj_pub.publish(path)

    def _publish_empty_trajectory(self):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        self._traj_pub.publish(msg)

    # ---------------- PMP introspection (optional) ----------------

    def extract_costates(self) -> list[np.ndarray]:
        """Costate estimates via the Covector Mapping Principle.

        psi_k = -pi_k / dt approximates the max-principle costate at t_k.
        Provided for offline PMP verification; not used in the closed loop.
        """
        dt = self.cfg.dt
        return [-self._solver.get(k, "pi") / dt for k in range(self.cfg.N)]

    def extract_predicted_trajectory(self) -> np.ndarray:
        """Full predicted state trajectory, shape (N+1, 5)."""
        return np.array([self._solver.get(k, "x") for k in range(self.cfg.N + 1)])


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
