from dataclasses import dataclass
from math import hypot
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from scipy.interpolate import RegularGridInterpolator
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException

import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


@dataclass
class PlannerConfig:
    # horizon
    N: int = 40
    T_horizon: float = 4.0   # [s]
    control_rate: float = 20.0  # [Hz]

    # kinematic limits
    a_max: float = 1.0       # [m/s^2]
    alpha_max: float = 1.0   # [rad/s^2]
    v_max: float = 0.5       # [m/s]
    omega_max: float = 0.8   # [rad/s]

    # running cost weights
    w_field_mag: float = 15.0   # T_lin gradient: primary obstacle-avoidance signal
    w_heading: float = 3.0      # heading alignment
    w_progress: float = 3.0     # forward progress along field
    w_accel: float = 0.5        # longitudinal acceleration effort
    w_ang_accel: float = 1.5    # angular acceleration effort
    w_omega: float = 4.0        # yaw rate damping (raised from 2.0 to damp wall-following wave)

    # cross-track error penalty
    # T_lin has zero first-order lateral gradient (dT_lin/de_cross = 0 analytically),
    # so the robot can drift sideways at zero marginal cost. The only lateral correction
    # from the heading term is delayed by omega/alpha limits, which produces the wave.
    # e_cross = -sin(theta_V)*(px-px_warm) + cos(theta_V)*(py-py_warm) is the signed
    # perpendicular displacement from the warm-start path. The quadratic penalty gives
    # a direct restoring force without heading-chain delay.
    # Hessian contribution is PSD; LM=10 remains sufficient.
    w_lateral: float = 10.0     # cross-track weight (running)
    w_T_lateral: float = 5.0    # cross-track weight (terminal)

    # terminal cost weights
    # w_T_pos MUST stay 0. T_lin linearises V_norm around the warm-start, so a terminal
    # node that shortcuts through an obstacle underestimates its true cost by 10-80x.
    # A Euclidean pull comparable in magnitude to that underestimation rewards placing the
    # terminal node closer to the goal through the obstacle. Empirically: shortcut total
    # 388 vs correct-path total 516 with w_T_pos=2, i.e. the optimizer takes the shortcut.
    # Restore to 0.1 only if the robot fails to converge in the last 0.2 m.
    w_T_pos: float = 0.0
    w_T_field: float = 40.0
    w_T_heading: float = 10.0
    w_T_vel: float = 5.0
    w_T_omega: float = 5.0

    # warm-start safety thresholds
    # corner_v_ratio: switch _shift_warm_start to field-following when any predicted
    # node's V_norm exceeds this multiple of the robot's current V_norm. Compared
    # against the robot (constant reference) rather than the previous node (sliding),
    # so gradual drift is caught: 15x 1.3x steps total 50x but the old consecutive
    # check never triggered. The 5th node at 3.7x robot V_norm does.
    corner_v_ratio: float = 3.0
    # drift_v_ratio: full re-seed when mean warm-start V_norm exceeds this multiple
    # of the robot's V_norm. Backup to corner_v_ratio for slower-accumulating drift.
    drift_v_ratio: float = 3.0

    # solver
    # EXACT hessian required: the w_progress * v * cos(theta - theta_V) term makes
    # the Hessian indefinite w.r.t. theta. GAUSS_NEWTON hands HPIPM an indefinite QP.
    # Worst-case negative eigenvalue: -(w_heading + w_progress * v_max) = -(3 + 1.5) = -4.5.
    # LM must exceed this; 10.0 gives comfortable margin.
    hessian_approx: str = "EXACT"
    levenberg_marquardt: float = 10.0
    integrator_type: str = "ERK"
    erk_stages: int = 4
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"

    goal_tolerance: float = 0.05  # [m]

    @property
    def dt(self) -> float:
        return self.T_horizon / self.N


def build_ocp_solver(cfg: PlannerConfig) -> AcadosOcpSolver:
    """Build and compile the acados OCP. Expensive (~seconds); called once at startup.
    Generated C code is cached on disk for subsequent runs.
    """
    p_x   = ca.SX.sym("p_x")
    p_y   = ca.SX.sym("p_y")
    theta = ca.SX.sym("theta")
    v     = ca.SX.sym("v")
    omega = ca.SX.sym("omega")
    x = ca.vertcat(p_x, p_y, theta, v, omega)

    a     = ca.SX.sym("a")      # longitudinal acceleration [m/s^2]
    alpha = ca.SX.sym("alpha")  # angular acceleration      [rad/s^2]
    u = ca.vertcat(a, alpha)

    # Runtime parameters, set per shooting node each MPC cycle:
    #   [0] V_norm   - FMM travel-time at the warm-start position
    #   [1] theta_V  - field direction atan2(gy, gx)
    #   [2] g_x      - goal x
    #   [3] g_y      - goal y
    #   [4] g_th     - goal heading
    #   [5] px_warm  - warm-start x at this node (linearisation centre)
    #   [6] py_warm  - warm-start y at this node
    # px_warm/py_warm give L a proper state gradient via T_lin:
    #   dL/dpx = -w_field_mag * cos(theta_V)
    #   dL/dpy = -w_field_mag * sin(theta_V)
    # Without this V_norm is a constant and the solver has no positional gradient,
    # causing it to drive straight through obstacles to minimise terminal distance.
    V_norm  = ca.SX.sym("V_norm")
    theta_V = ca.SX.sym("theta_V")
    g_x     = ca.SX.sym("g_x")
    g_y     = ca.SX.sym("g_y")
    g_th    = ca.SX.sym("g_th")
    px_warm = ca.SX.sym("px_warm")
    py_warm = ca.SX.sym("py_warm")
    p = ca.vertcat(V_norm, theta_V, g_x, g_y, g_th, px_warm, py_warm)

    # Unicycle with acceleration-level control (depth-2 / Newtonian structure).
    x_dot = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
        a,
        alpha,
    )

    gx_V   = ca.cos(theta_V)
    gy_V   = ca.sin(theta_V)
    dtheta = theta - theta_V

    T_lin = V_norm - gx_V * (p_x - px_warm) - gy_V * (p_y - py_warm)

    # Cross-track error: signed lateral displacement from the warm-start path
    # (positive = left of field direction). See PlannerConfig.w_lateral.
    e_cross = -gy_V * (p_x - px_warm) + gx_V * (p_y - py_warm)

    L = (
        cfg.w_field_mag  * T_lin
        + cfg.w_lateral  * e_cross**2
        + cfg.w_heading  * (1.0 - ca.cos(dtheta))
        - cfg.w_progress * v * ca.cos(dtheta)
        + 0.5 * cfg.w_accel     * a**2
        + 0.5 * cfg.w_ang_accel * alpha**2
        + 0.5 * cfg.w_omega     * omega**2
    )

    # w_T_pos=0: see PlannerConfig for why a Euclidean pull causes corner-cutting.
    Phi = (
        cfg.w_T_field   * T_lin
        + cfg.w_T_pos     * ((p_x - g_x)**2 + (p_y - g_y)**2)
        + cfg.w_T_lateral * e_cross**2
        + cfg.w_T_heading * (1.0 - ca.cos(theta - g_th))
        + cfg.w_T_vel     * v**2
        + cfg.w_T_omega   * omega**2
    )

    model = AcadosModel()
    model.name = "skid_steer_vfield"
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_ode  = x_dot
    model.f_expl_expr = x_dot
    model.cost_expr_ext_cost   = L
    model.cost_expr_ext_cost_e = Phi

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = cfg.N
    ocp.solver_options.tf        = cfg.T_horizon
    ocp.parameter_values = np.zeros(7)
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.constraints.lbu   = np.array([-cfg.a_max,    -cfg.alpha_max])
    ocp.constraints.ubu   = np.array([+cfg.a_max,    +cfg.alpha_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # Lower bound on v is 0: no reversing. The optimizer otherwise finds
    # backing-up-and-turning as locally optimal when blocked.
    ocp.constraints.lbx   = np.array([0.0, -cfg.omega_max])
    ocp.constraints.ubx   = np.array([+cfg.v_max, +cfg.omega_max])
    ocp.constraints.idxbx = np.array([3, 4])
    ocp.constraints.lbx_e   = np.array([0.0, -cfg.omega_max])
    ocp.constraints.ubx_e   = np.array([+cfg.v_max, +cfg.omega_max])
    ocp.constraints.idxbx_e = np.array([3, 4])
    ocp.constraints.x0 = np.zeros(5)

    ocp.solver_options.qp_solver              = cfg.qp_solver
    ocp.solver_options.integrator_type        = cfg.integrator_type
    ocp.solver_options.sim_method_num_stages  = cfg.erk_stages
    ocp.solver_options.sim_method_num_steps   = 1
    ocp.solver_options.nlp_solver_type        = "SQP_RTI"
    # 3 iterations: with bilinear v*cos(theta-theta_V) in the cost, 2 iterations
    # leave a residual that oscillates between cycles and shows as path wobble.
    ocp.solver_options.nlp_solver_max_iter    = 3
    ocp.solver_options.print_level            = 0
    ocp.solver_options.hessian_approx         = cfg.hessian_approx
    ocp.solver_options.levenberg_marquardt    = cfg.levenberg_marquardt

    return AcadosOcpSolver(ocp, json_file="skid_steer_vfield.json")


class VectorFieldInterpolator:
    """Queryable representation of the FMM vector field.

    Builds scipy interpolators over travel_time and the gradient direction.
    Three correctness requirements:
      1. Obstacle/unreachable cells carry max cost, not zero.
      2. Coordinate axes are cell-centred (origin is cell corner; data sits at centre).
      3. Heading angles are stored as (sin, cos) to avoid the atan2 wrap-around
         artefact at +/-pi that would yield the wrong interpolated direction.
    """

    def __init__(self):
        self._interp_cost: Optional[RegularGridInterpolator] = None
        self._interp_sin:  Optional[RegularGridInterpolator] = None
        self._interp_cos:  Optional[RegularGridInterpolator] = None
        self._ready = False
        self._obstacle_cost: float = 0.0

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
        """Rebuild interpolators from new field data.

        origin_x/y is the corner of cell (0,0); each cell centre is offset by
        +0.5*resolution, then spaced by resolution along each axis.
        """
        rows, cols = travel_time.shape
        xs = origin_x + np.arange(cols) * resolution
        ys = origin_y + np.arange(rows) * resolution

        cost = travel_time.copy().astype(np.float64)
        obstacle_cost = float(cost.max()) if cost.size > 0 else 1.0
        self._obstacle_cost = obstacle_cost

        sin_angle = np.sin(np.arctan2(grad_y, grad_x))
        cos_angle = np.cos(np.arctan2(grad_y, grad_x))

        self._interp_cost = RegularGridInterpolator(
            (ys, xs), cost,
            method="linear", bounds_error=False, fill_value=obstacle_cost,
        )
        self._interp_sin = RegularGridInterpolator(
            (ys, xs), sin_angle,
            method="linear", bounds_error=False, fill_value=0.0,
        )
        self._interp_cos = RegularGridInterpolator(
            (ys, xs), cos_angle,
            method="linear", bounds_error=False, fill_value=1.0,
        )
        self._ready = True

    def query(self, px: float, py: float) -> tuple[float, float]:
        """Return (V_norm, theta_V) at world position (px, py)."""
        if not self._ready:
            return 0.0, 0.0
        pt = np.array([[py, px]])
        v_norm  = float(self._interp_cost(pt))
        theta_v = float(np.arctan2(self._interp_sin(pt), self._interp_cos(pt)))
        return v_norm, theta_v


class PlannerNode(Node):

    def __init__(self):
        super().__init__("pmp_planner")

        self.cfg = PlannerConfig()

        self.declare_parameter("map_frame",   "map")
        self.declare_parameter("robot_frame", "base_link")
        self._map_frame   = self.get_parameter("map_frame").value
        self._robot_frame = self.get_parameter("robot_frame").value

        # Pose from TF map->base_link, not from /odom. The odom frame drifts relative
        # to the map frame as SLAM makes loop-closure corrections; using /odom directly
        # puts the planner in a different frame from the vector field and costmap.
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._xi   = np.zeros(5)
        self._goal: Optional[np.ndarray] = None
        self._field = VectorFieldInterpolator()
        self._warm_started      = False
        self._waiting_for_field = False

        # Low-pass filter for velocity output.
        # v: heavier smoothing to absorb fore-aft oscillation.
        # omega: light smoothing only -- heavy filtering delays turns, causing
        # the MPC to compensate with larger omega commands next cycle.
        self._smooth_v:     float = 0.0
        self._smooth_omega: float = 0.0
        _dt = self.cfg.dt
        self._smooth_alpha_v     = _dt / (0.15 + _dt)   # tau=0.15s
        self._smooth_alpha_omega = _dt / (0.04 + _dt)   # tau=0.04s

        self.get_logger().info("Compiling acados solver...")
        self._solver = build_ocp_solver(self.cfg)
        self.get_logger().info("Solver ready.")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(Odometry,          "/odom",                     self._on_odom,  qos)
        self.create_subscription(PoseStamped,        "/goal_pose",               self._on_goal,  qos)
        self.create_subscription(Float32MultiArray,  "/vector_field/planner_data", self._on_field, qos)

        self.declare_parameter("enable_stamped_cmd_vel", False)
        self._stamped_cmd = self.get_parameter("enable_stamped_cmd_vel").get_parameter_value().bool_value

        if self._stamped_cmd:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self._traj_pub = self.create_publisher(Path, "/pmp_planner/trajectory", 10)

        self._timer = self.create_timer(1.0 / self.cfg.control_rate, self._control_loop)

        self.get_logger().info(
            f"Running at {self.cfg.control_rate} Hz, "
            f"horizon {self.cfg.T_horizon}s / {self.cfg.N} nodes."
        )

    def _on_odom(self, msg: Odometry):
        # Velocity from odometry body-frame twist (accurate, frame-independent).
        # Position from TF map->base_link (SLAM-corrected).
        v     = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z

        try:
            t = self._tf_buffer.lookup_transform(
                self._map_frame, self._robot_frame, rclpy.time.Time(),
            )
            tx  = t.transform.translation.x
            ty  = t.transform.translation.y
            q   = t.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self._xi = np.array([tx, ty, yaw, v, omega])
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self._map_frame}->{self._robot_frame} unavailable: {e}",
                throttle_duration_sec=2.0,
            )

    def _on_goal(self, msg: PoseStamped):
        pos = msg.pose.position
        q   = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._goal = np.array([pos.x, pos.y, yaw])
        # Pause planning until the new field arrives; seeding on the old field
        # would produce a path toward the old goal.
        self._warm_started      = False
        self._waiting_for_field = True
        self.get_logger().info(f"Goal: ({pos.x:.2f}, {pos.y:.2f})")

    def _on_field(self, msg: Float32MultiArray):
        """Parse packed field from FMMVectorFieldNode.

        Layout: [height, width, origin_x, origin_y, resolution,
                 travel_time (H*W), grad_x (H*W), grad_y (H*W)]
        """
        data = np.array(msg.data, dtype=np.float32)
        if len(data) < 5:
            return

        h          = int(data[0])
        w          = int(data[1])
        origin_x   = float(data[2])
        origin_y   = float(data[3])
        resolution = float(data[4])
        n = h * w

        if len(data) != 5 + 3 * n:
            self.get_logger().warn(
                f"Field size mismatch: got {len(data)}, expected {5 + 3*n}",
                throttle_duration_sec=5.0,
            )
            return

        travel_time = data[5       : 5 +     n].reshape(h, w)
        grad_x      = data[5 +     n : 5 + 2*n].reshape(h, w)
        grad_y      = data[5 + 2 * n : 5 + 3*n].reshape(h, w)

        self._field.update(travel_time, grad_x, grad_y, origin_x, origin_y, resolution)
        self._waiting_for_field = False

    def _control_loop(self):
        if self._goal is None:
            return

        if not self._field.ready:
            self.get_logger().warn(
                "Vector field not yet received -- waiting.", throttle_duration_sec=2.0,
            )
            return

        if self._waiting_for_field:
            self._publish_twist(0.0, 0.0)
            self.get_logger().info(
                "Waiting for updated vector field...", throttle_duration_sec=1.0,
            )
            return

        dx = self._xi[0] - self._goal[0]
        dy = self._xi[1] - self._goal[1]
        if hypot(dx, dy) < self.cfg.goal_tolerance:
            self._publish_twist(0.0, 0.0)
            self._publish_empty_trajectory()
            self.get_logger().info(
                f"Goal reached (dist={hypot(dx, dy):.3f} m).", throttle_duration_sec=1.0,
            )
            return

        u_opt = self._solve_mpc()

        if u_opt is not None:
            # Use predicted state at node 1 (one dt of ERK4 integration on the
            # optimal acceleration) rather than integrating the acceleration ourselves.
            xi_next = self._solver.get(1, "x")
            self._publish_twist(float(xi_next[3]), float(xi_next[4]))
            self._publish_trajectory()
        else:
            self.get_logger().warn("Solver failed -- stopping.")
            self._publish_twist(0.0, 0.0)

    @staticmethod
    def _rk4_step(
        state: np.ndarray, u: np.ndarray, dt: float,
        v_max: float, omega_max: float,
    ) -> np.ndarray:
        """Single RK4 step matching the acados ERK4 internal integrator.

        Using Euler in the warm-start under-integrates curved heading changes,
        placing nodes slightly inside the curve each cycle. The QP corrects
        outward; Euler cuts back next cycle -- that is the oscillation seen in
        tight spaces. RK4 keeps the warm-start consistent with the solver to
        fourth order, eliminating the systematic per-cycle error.
        """
        a, al = u[0], u[1]

        def f(s: np.ndarray) -> np.ndarray:
            return np.array([s[3] * np.cos(s[2]), s[3] * np.sin(s[2]), s[4], a, al])

        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        ns = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ns[3] = np.clip(ns[3], 0.0, v_max)
        ns[4] = np.clip(ns[4], -omega_max, omega_max)
        return ns

    def _field_following_control(
        self, state: np.ndarray, v_target: float, k_heading: float = 2.0
    ) -> np.ndarray:
        """Compute [a, alpha] to follow the field direction from the given state.

        Speed is scaled by cos(heading_err) so the robot decelerates at corners
        instead of overshooting spatially (which would place warm-start nodes
        inside the obstacle on the other side). cos is clamped to >=0 to prevent
        reverse commands during seeding.
        """
        dt = self.cfg.dt
        _, theta_v   = self._field.query(state[0], state[1])
        heading_err  = theta_v - state[2]
        heading_err  = (heading_err + np.pi) % (2 * np.pi) - np.pi

        omega_desired = np.clip(
            k_heading * heading_err / dt, -self.cfg.omega_max, self.cfg.omega_max,
        )
        alpha_cmd = np.clip(
            (omega_desired - state[4]) / dt, -self.cfg.alpha_max, self.cfg.alpha_max,
        )
        cos_align = max(np.cos(heading_err), 0.0)
        a_cmd = np.clip(
            (v_target * cos_align - state[3]) / dt, -self.cfg.a_max, self.cfg.a_max,
        )
        return np.array([a_cmd, alpha_cmd])

    def _seed_trajectory_toward_goal(self):
        """Initialise every shooting node by forward-integrating along the field
        at half v_max using the speed-adaptive field-following controller.
        """
        solver  = self._solver
        N       = self.cfg.N
        dt      = self.cfg.dt
        v_seed  = self.cfg.v_max * 0.5

        state = np.array([
            self._xi[0], self._xi[1], self._xi[2],
            v_seed,
            self._xi[4],
        ])

        for k in range(N + 1):
            solver.set(k, "x", state.copy())
            if k < N:
                if hypot(state[0] - self._goal[0], state[1] - self._goal[1]) < self.cfg.goal_tolerance:
                    for j in range(k + 1, N + 1):
                        solver.set(j, "x", state.copy())
                        if j < N:
                            solver.set(j, "u", np.zeros(2))
                    break

                u_seed = self._field_following_control(state, v_seed)
                solver.set(k, "u", u_seed)
                state = self._rk4_step(state, u_seed, dt, self.cfg.v_max, self.cfg.omega_max)

    def _shift_warm_start(self):
        """Shift controls by one step (RTI) and re-integrate states from the robot.

        Corner-overshoot protection: each node's V_norm is checked against the
        robot's current V_norm (constant reference). If a node lands in a high-cost
        region, the shifted controls from that node onward are replaced by the
        field-following seed controller, which also slows at corners.
        """
        solver = self._solver
        N      = self.cfg.N
        dt     = self.cfg.dt

        v_robot_cost, _ = self._field.query(self._xi[0], self._xi[1])

        us_prev = [solver.get(k, "u") for k in range(N)]
        us_new  = [us_prev[k + 1] if k < N - 1 else us_prev[N - 1] for k in range(N)]
        for k in range(N):
            solver.set(k, "u", us_new[k])

        state      = self._xi.copy()
        using_seed = False

        for k in range(N):
            if not using_seed:
                next_state = self._rk4_step(
                    state, us_new[k], dt, self.cfg.v_max, self.cfg.omega_max,
                )
                v_next, _ = self._field.query(next_state[0], next_state[1])

                if v_next > self.cfg.corner_v_ratio * max(v_robot_cost, 0.01):
                    using_seed = True
                else:
                    state = next_state

            if using_seed:
                u_field = self._field_following_control(state, self.cfg.v_max * 0.5)
                solver.set(k, "u", u_field)
                state = self._rk4_step(
                    state, u_field, dt, self.cfg.v_max, self.cfg.omega_max,
                )

            solver.set(k + 1, "x", state)

    def _warm_start_has_drifted(self) -> bool:
        """True if the mean warm-start V_norm substantially exceeds the robot's.

        Tight spaces have naturally elevated V_norm, so the threshold must allow
        some elevation without triggering constant re-seeds.
        """
        solver = self._solver
        N      = self.cfg.N

        v_at_robot, _ = self._field.query(self._xi[0], self._xi[1])
        if v_at_robot < 1e-3:
            return False

        total, count = 0.0, 0
        for k in range(1, N + 1, 4):
            xi_k = solver.get(k, "x")
            v_k, _ = self._field.query(xi_k[0], xi_k[1])
            total += v_k
            count += 1

        if count == 0:
            return False

        return (total / count) > self.cfg.drift_v_ratio * v_at_robot

    def _solve_mpc(self) -> Optional[np.ndarray]:
        """Run one SQP-RTI iteration. Returns optimal [a, alpha] or None on failure."""
        solver = self._solver
        N      = self.cfg.N

        if not self._warm_started:
            self._seed_trajectory_toward_goal()
            self._warm_started = True
        else:
            self._shift_warm_start()

            if self._warm_start_has_drifted():
                self.get_logger().warn(
                    "Warm-start drifted into high-cost region -- re-seeding.",
                    throttle_duration_sec=1.0,
                )
                self._seed_trajectory_toward_goal()

        # Pin current robot state: set both decision variable and bounds so the
        # warm-start and the equality constraint are consistent from the QP start.
        solver.set(0, "x",   self._xi)
        solver.set(0, "lbx", self._xi)
        solver.set(0, "ubx", self._xi)

        for k in range(N + 1):
            xi_k = self._xi if k == 0 else solver.get(k, "x")
            v_norm_k, theta_v_k = self._field.query(xi_k[0], xi_k[1])
            solver.set(k, "p", np.array([
                v_norm_k, theta_v_k,
                self._goal[0], self._goal[1], self._goal[2],
                xi_k[0], xi_k[1],
            ]))

        status = solver.solve()

        if status != 0:
            self.get_logger().warn(f"acados status {status}")
            self._diagnose_solver_failure()
            self._warm_started = False
            return None

        return solver.get(0, "u")

    def _publish_twist(self, v: float, omega: float):
        # First-order low-pass to absorb high-frequency MPC oscillation.
        # Reset immediately on stop command so the robot does not coast.
        if v == 0.0 and omega == 0.0:
            self._smooth_v = self._smooth_omega = 0.0
        else:
            self._smooth_v = (
                self._smooth_alpha_v * v
                + (1.0 - self._smooth_alpha_v) * self._smooth_v
            )
            self._smooth_omega = (
                self._smooth_alpha_omega * omega
                + (1.0 - self._smooth_alpha_omega) * self._smooth_omega
            )

        if self._stamped_cmd:
            msg = TwistStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.x  = self._smooth_v
            msg.twist.angular.z = self._smooth_omega
        else:
            msg = Twist()
            msg.linear.x  = self._smooth_v
            msg.angular.z = self._smooth_omega
        self._cmd_pub.publish(msg)

    def _publish_trajectory(self):
        now  = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp    = now
        path.header.frame_id = "map"

        for k in range(self.cfg.N + 1):
            xi_k = self._solver.get(k, "x")
            pose = PoseStamped()
            pose.header.stamp    = now
            pose.header.frame_id = "map"
            pose.pose.position.x = float(xi_k[0])
            pose.pose.position.y = float(xi_k[1])
            pose.pose.position.z = 0.0
            yaw = float(xi_k[2])
            pose.pose.orientation.z = np.sin(yaw / 2.0)
            pose.pose.orientation.w = np.cos(yaw / 2.0)
            path.poses.append(pose)

        self._traj_pub.publish(path)

    def _publish_empty_trajectory(self):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        self._traj_pub.publish(msg)

    def _diagnose_solver_failure(self):
        """Log diagnostic information immediately after a failed solve().

        Covers: state, goal, field params at sampled nodes, KKT residuals,
        cost, bounds check, and the Hessian indefiniteness test.
        """
        solver = self._solver
        N      = self.cfg.N
        logger = self.get_logger()
        xi     = self._xi

        logger.warn(
            f"[DIAG] State x0: px={xi[0]:.3f}  py={xi[1]:.3f}  "
            f"th={xi[2]:.3f}  v={xi[3]:.3f}  w={xi[4]:.3f}"
        )

        g = self._goal
        logger.warn(
            f"[DIAG] Goal: gx={g[0]:.3f}  gy={g[1]:.3f}  gth={g[2]:.3f}  "
            f"dist={hypot(xi[0]-g[0], xi[1]-g[1]):.3f} m"
        )

        logger.warn(f"[DIAG] Field ready: {self._field.ready}")

        logger.warn("[DIAG] Node params (V_norm, theta_V):")
        for k in list(range(min(5, N + 1))) + [N]:
            xi_k = xi if k == 0 else solver.get(k, "x")
            v_norm_k, theta_v_k = self._field.query(xi_k[0], xi_k[1])
            p_k = solver.get(k, "p")
            logger.warn(
                f"  node {k:3d}: queried=(V={v_norm_k:.3f}, thV={theta_v_k:.3f})  "
                f"stored=(V={p_k[0]:.3f}, thV={p_k[1]:.3f})  "
                f"xi=({xi_k[0]:.2f},{xi_k[1]:.2f},{xi_k[2]:.2f})"
            )

        # Large stat -> bad gradient. Large ineq -> hard constraint violated.
        try:
            res = solver.get_residuals()
            logger.warn(
                f"[DIAG] KKT: stat={res[0]:.3e}  eq={res[1]:.3e}  "
                f"ineq={res[2]:.3e}  comp={res[3]:.3e}"
            )
        except Exception as e:
            logger.warn(f"[DIAG] Could not get residuals: {e}")

        try:
            logger.warn(f"[DIAG] Cost: {solver.get_cost():.4f}")
        except Exception as e:
            logger.warn(f"[DIAG] Could not get cost: {e}")

        logger.warn("[DIAG] State/control at first 5 nodes:")
        for k in range(min(5, N)):
            xi_k = solver.get(k, "x")
            u_k  = solver.get(k, "u")
            v_ok  = 0.0              <= xi_k[3] <= self.cfg.v_max
            w_ok  = -self.cfg.omega_max <= xi_k[4] <= self.cfg.omega_max
            a_ok  = -self.cfg.a_max     <= u_k[0]  <= self.cfg.a_max
            al_ok = -self.cfg.alpha_max <= u_k[1]  <= self.cfg.alpha_max
            logger.warn(
                f"  node {k}: v={xi_k[3]:.3f}({'OK' if v_ok  else 'OOB'})  "
                f"w={xi_k[4]:.3f}({'OK' if w_ok  else 'OOB'})  "
                f"a={u_k[0]:.3f}({'OK' if a_ok  else 'OOB'})  "
                f"al={u_k[1]:.3f}({'OK' if al_ok else 'OOB'})"
            )

        # d^2L/dtheta^2 = (w_heading + w_progress * v) * cos(theta - theta_V)
        # Negative when misaligned > 90 deg and LM < |eigenvalue|.
        p0      = solver.get(0, "p")
        dtheta  = xi[2] - p0[1]
        hess_th = (self.cfg.w_heading + self.cfg.w_progress * xi[3]) * np.cos(dtheta)
        lm      = self.cfg.levenberg_marquardt
        logger.warn(
            f"[DIAG] d2L/dtheta2 = {hess_th:.3f}  "
            f"(cos(dtheta)={np.cos(dtheta):.3f}, dtheta={np.degrees(dtheta):.1f} deg)  "
            f"LM={lm}  "
            f"{'INDEFINITE -- raise LM!' if hess_th < -lm else 'positive-definite OK'}"
        )

    def extract_costates(self) -> list[np.ndarray]:
        """Costate estimates via the Covector Mapping Principle (Gong et al. 2008).

        pi_k / dt approximates lambda(t_k); sign flip converts min to max convention.
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
