"""Vector-field guided NMPC for a skid-steer unicycle.

This is the planner refactored to:
  - load all PlannerConfig parameters dynamically from ROS2 parameters,
    with the dataclass defaults serving as the parameter defaults;
  - optionally weight the field-tracking residuals (heading, align,
    cross-track) by a per-stage confidence factor derived from the
    field's |grad T|; cells on FMM cut loci or near the goal have low
    |grad T| and become low-confidence, so the planner stops tightly
    tracking the (unreliable) direction there;
  - accept the new vector-field message format that includes the
    |grad T| channel, while remaining backward-compatible with the
    legacy 3-channel format.

When confidence weighting is disabled (the default), the planner is
bit-identical to the previous version: every residual sees confidence
= 1.0, and the OCP cost reduces to the original NL_LS form.
"""

from dataclasses import dataclass, fields, replace
from math import hypot, tanh, atan2, pi, cos
from typing import Any, Optional

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
# Generic dataclass-driven ROS2 parameter loader
# ---------------------------------------------------------------------------


def declare_and_load_dataclass(
    node: Node, instance: Any, prefix: str = ""
) -> Any:
    """Declare every dataclass field as a ROS2 parameter and return a new
    instance populated from the parameter values. The dataclass instance's
    current values become the parameter defaults.
    """
    updates: dict = {}
    for f in fields(instance):
        name = prefix + f.name
        default = getattr(instance, f.name)
        node.declare_parameter(name, default)
        updates[f.name] = node.get_parameter(name).value
    return replace(instance, **updates)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PlannerConfig:
    """All planner parameters in one dataclass.

    Loaded from ROS2 parameters at startup via declare_and_load_dataclass;
    the dataclass defaults below become the ROS2 parameter defaults.
    Adding a new tunable is a one-line change.

    The `dt` derived value remains a property because it is not a tunable
    -- it follows from N and T_horizon.
    """

    # --- Horizon ---------------------------------------------------------
    N: int = 25                        # number of shooting intervals
    T_horizon: float = 2.0             # [s]; total prediction horizon
    control_rate: float = 20.0         # [Hz]

    # --- Kinematic limits ------------------------------------------------
    a_max: float = 1.0                 # [m/s^2]
    alpha_max: float = 2.0             # [rad/s^2]
    v_max: float = 0.5                 # [m/s]; symmetric -> reverse OK
    omega_max: float = 1.5             # [rad/s]

    # --- Speed reference -------------------------------------------------
    # v_ref(d_goal) = v_max * tanh(d_goal / L_brake) decelerates smoothly
    # within one L_brake of the goal.
    L_brake: float = 0.5

    # --- Reference-path inner loop (warm-start generator) ----------------
    ref_K_omega: float = 2.0           # P-gain on heading error
    ref_brake_cos: float = 0.0         # min cos(heading_err) gating v

    # --- Terminal cost-to-go cap -----------------------------------------
    # T_lin = T_ref - F.(p_N - p_ref), squared at the terminal node.
    # Capping T_ref keeps the squared cost on the same scale as the
    # stage cost regardless of map size.
    T_ref_cap: float = 5.0

    # --- Stage cost weights ----------------------------------------------
    # The "tracking" weights (heading, align, xtrack) become upper bounds
    # when confidence weighting is enabled: the effective weight at stage
    # k is w_* * c_k, where c_k = clip(|grad T(x_k)| / mag_ref, 0, 1).
    w_heading: float = 1.5             # cross-product residual r1 = F x h
    w_align:   float = 4.0             # 1 - F . h, breaks the -F basin
    w_speed:   float = 1.0             # v - v_ref * (F . h)
    w_xtrack:  float = 4.0             # signed lateral to reference line
    w_omega:   float = 0.2             # mild yaw-rate damping
    w_a:       float = 0.05            # control effort
    w_alpha:   float = 0.1             # control effort

    # --- Terminal cost weights -------------------------------------------
    w_T_field: float = 1.0             # squared T_lin (capped)
    w_T_v:     float = 0.5             # v_N -> 0
    w_T_omega: float = 0.5             # omega_N -> 0

    # --- Goal handling ---------------------------------------------------
    goal_tolerance: float = 0.05       # [m]

    # --- Confidence weighting --------------------------------------------
    # When False, the planner publishes c=1.0 for every stage and the OCP
    # cost is identical to the unweighted version. When True, c is
    # computed from the field's |grad T| with the percentile-based
    # mag_ref normalisation.
    enable_confidence_weighting: bool = False
    # Percentile of |grad T| used as the "full confidence" reference.
    # 95.0 means cells in the top 5% of |grad T| get c=1 (capped); cells
    # below that get c < 1 in proportion. Lowering this value (e.g. 75)
    # makes more cells reach full confidence.
    confidence_mag_percentile: float = 95.0
    # Gamma shaping: c = clip(ratio, 0, 1) ** gamma. gamma=1.0 is linear;
    # gamma > 1 sharpens the transition (more cells get suppressed);
    # gamma < 1 broadens it (more cells get full confidence).
    confidence_gamma: float = 1.0
    # Minimum confidence floor. Even cut-locus cells will track the
    # field with at least this weight; prevents the planner from
    # entirely ignoring the field on extended low-confidence regions.
    confidence_floor: float = 0.0

    @property
    def dt(self) -> float:
        return self.T_horizon / self.N


@dataclass
class TopicConfig:
    map_frame: str = "map"
    robot_frame: str = "base_link"
    enable_stamped_cmd_vel: bool = False


@dataclass
class SolverConfig:
    """Numerical-robustness knobs for the acados solver.

    These are separate from PlannerConfig because they control HOW the
    NLP is solved, not WHAT is being solved. Default values match the
    historic build.

    The `levenberg_marquardt` parameter is the most useful tuning knob
    when ACADOS_MINSTEP / status 4 errors appear: it adds lambda*I to
    the Gauss-Newton Hessian, improving QP conditioning at the cost of
    slightly slower convergence. Increase from 1e-4 toward 1e-2 if the
    QP solver complains; decrease toward 1e-5 if convergence feels
    sluggish in well-posed regions.
    """
    # Gauss-Newton Hessian regulariser. Larger -> more robust QP, less
    # aggressive steps. Typical safe range: 1e-5 to 1e-1.
    levenberg_marquardt: float = 1e-4
    # SQP-RTI iterations per control cycle. 1 is fastest (real-time
    # iteration); 2-3 is more robust but ~2-3x slower per cycle.
    nlp_solver_max_iter: int = 1
    # acados log verbosity. 0 = silent, 1 = summary, 2 = per-iteration.
    print_level: int = 0


# ---------------------------------------------------------------------------
# Solver build (called once at startup, cached on disk by acados)
# ---------------------------------------------------------------------------


def build_ocp_solver(
    cfg: PlannerConfig, solver_cfg: SolverConfig,
) -> AcadosOcpSolver:
    # State and control symbols for the dynamic unicycle model.
    p_x   = ca.SX.sym("p_x")
    p_y   = ca.SX.sym("p_y")
    theta = ca.SX.sym("theta")
    v     = ca.SX.sym("v")
    omega = ca.SX.sym("omega")
    x = ca.vertcat(p_x, p_y, theta, v, omega)

    a     = ca.SX.sym("a")
    alpha = ca.SX.sym("alpha")
    u = ca.vertcat(a, alpha)

    # Per-stage parameters. The 7th entry (c) is the confidence factor.
    # When the runtime sets c = 1.0 for every stage, the residuals reduce
    # to the original unweighted form and behaviour is identical to the
    # previous planner.
    Fx     = ca.SX.sym("Fx")
    Fy     = ca.SX.sym("Fy")
    v_ref  = ca.SX.sym("v_ref")
    T_ref  = ca.SX.sym("T_ref")
    px_ref = ca.SX.sym("px_ref")
    py_ref = ca.SX.sym("py_ref")
    c      = ca.SX.sym("c")            # confidence in [0, 1]
    p = ca.vertcat(Fx, Fy, v_ref, T_ref, px_ref, py_ref, c)
    n_params = 7

    x_dot = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
        a,
        alpha,
    )

    # ---- Stage residual y_stage (target = 0) ----
    cos_t   = ca.cos(theta)
    sin_t   = ca.sin(theta)
    F_dot_h = cos_t * Fx + sin_t * Fy   # F . heading_hat in [-1, 1]
    F_x_h   = sin_t * Fx - cos_t * Fy   # F x heading_hat (2D scalar)
    dpx     = p_x - px_ref
    dpy     = p_y - py_ref

    # The three field-tracking residuals are scaled by c. The other four
    # (speed regulation, yaw damping, control effort) are NOT scaled --
    # they are platform regularisers, not field-tracking terms, and
    # should be applied uniformly across the horizon.
    y_stage = ca.vertcat(
        c * F_x_h,                       # r_heading (scaled)
        c * (1.0 - F_dot_h),             # r_align   (scaled)
        v - v_ref * F_dot_h,             # r_speed
        c * (-Fy * dpx + Fx * dpy),      # r_xtrack  (scaled)
        omega,                           # damping
        a,                               # control effort
        alpha,                           # control effort
    )

    # ---- Terminal residual y_terminal (target = 0) ----
    T_lin = T_ref - Fx * dpx - Fy * dpy
    y_terminal = ca.vertcat(T_lin, v, omega)

    model = AcadosModel()
    model.name           = "skid_steer_vfield"
    model.x              = x
    model.u              = u
    model.p              = p
    model.f_expl_expr    = x_dot
    model.cost_y_expr    = y_stage
    model.cost_y_expr_e  = y_terminal

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = cfg.N
    ocp.solver_options.tf        = cfg.T_horizon
    ocp.parameter_values         = np.zeros(n_params)

    ny   = y_stage.shape[0]
    ny_e = y_terminal.shape[0]
    ocp.cost.cost_type   = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag([
        cfg.w_heading,
        cfg.w_align,
        cfg.w_speed,
        cfg.w_xtrack,
        cfg.w_omega,
        cfg.w_a,
        cfg.w_alpha,
    ])
    ocp.cost.W_e = np.diag([
        cfg.w_T_field,
        cfg.w_T_v,
        cfg.w_T_omega,
    ])
    ocp.cost.yref   = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    ocp.constraints.lbu     = np.array([-cfg.a_max,    -cfg.alpha_max])
    ocp.constraints.ubu     = np.array([+cfg.a_max,    +cfg.alpha_max])
    ocp.constraints.idxbu   = np.array([0, 1])
    ocp.constraints.lbx     = np.array([-cfg.v_max,    -cfg.omega_max])
    ocp.constraints.ubx     = np.array([+cfg.v_max,    +cfg.omega_max])
    ocp.constraints.idxbx   = np.array([3, 4])
    ocp.constraints.lbx_e   = np.array([-cfg.v_max,    -cfg.omega_max])
    ocp.constraints.ubx_e   = np.array([+cfg.v_max,    +cfg.omega_max])
    ocp.constraints.idxbx_e = np.array([3, 4])
    ocp.constraints.x0      = np.zeros(5)

    ocp.solver_options.qp_solver             = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N      = max(1, cfg.N // 5)
    ocp.solver_options.integrator_type       = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type       = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter   = solver_cfg.nlp_solver_max_iter
    ocp.solver_options.hessian_approx        = "GAUSS_NEWTON"
    ocp.solver_options.levenberg_marquardt   = solver_cfg.levenberg_marquardt
    ocp.solver_options.print_level           = solver_cfg.print_level

    return AcadosOcpSolver(ocp, json_file="skid_steer_vfield.json")


# ---------------------------------------------------------------------------
# Vector field grid sampler
# ---------------------------------------------------------------------------


class VectorFieldGrid:
    """Bilinear sampler for (Fx, Fy, T, |grad T|).

    Backward-compatible: if the upstream message lacks the |grad T|
    channel (legacy 3-channel format), the magnitude grid is filled with
    1.0 -- which when normalised by mag_ref also gives 1.0 -- so
    downstream confidence computation evaluates to "full confidence
    everywhere" and the planner reduces to the unweighted form.
    """

    def __init__(self):
        self._gx: Optional[np.ndarray] = None
        self._gy: Optional[np.ndarray] = None
        self._tt: Optional[np.ndarray] = None
        self._mag: Optional[np.ndarray] = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._res = 1.0
        self._tt_max = 1.0
        self._mag_ref = 1.0      # see set_mag_ref
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def update(
        self, travel_time: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray,
        grad_mag: Optional[np.ndarray],
        origin_x: float, origin_y: float, resolution: float,
        mag_percentile: float,
    ):
        self._gx = grad_x.astype(np.float64)
        self._gy = grad_y.astype(np.float64)
        self._tt = travel_time.astype(np.float64)
        if grad_mag is None:
            # Legacy upstream: fill magnitudes with 1.0 so confidence
            # computation reduces to "full confidence everywhere".
            self._mag = np.ones_like(self._gx)
            self._mag_ref = 1.0
        else:
            self._mag = grad_mag.astype(np.float64)
            valid = self._mag[(self._mag > 0) & np.isfinite(self._mag)]
            if valid.size:
                self._mag_ref = float(np.percentile(valid, mag_percentile))
                if self._mag_ref < 1e-8:
                    self._mag_ref = 1.0
            else:
                self._mag_ref = 1.0
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._res = resolution
        if self._tt.size:
            finite = self._tt[np.isfinite(self._tt)]
            self._tt_max = float(finite.max()) if finite.size else 1.0
        else:
            self._tt_max = 1.0
        self._ready = True

    @property
    def mag_ref(self) -> float:
        return self._mag_ref

    def query(
        self, px: float, py: float
    ) -> tuple[float, float, float, float]:
        """Bilinear (Fx, Fy, T, |grad T|) at world (px, py).

        F is re-normalised after interpolation: bilinear blending of unit
        vectors does not preserve unit norm, and the (1 - F . h) residual
        would otherwise have a non-zero floor at perfect heading alignment.
        |grad T| is interpolated linearly without renormalisation -- it
        is a confidence signal and should reflect the local field quality.
        """
        if not self._ready:
            return 0.0, 0.0, 0.0, 0.0

        u = (px - self._origin_x) / self._res - 0.5
        w = (py - self._origin_y) / self._res - 0.5

        rows, cols = self._gx.shape
        if not (0.0 <= u <= cols - 1.0 and 0.0 <= w <= rows - 1.0):
            # Out of map: zero direction, no confidence, max-T sentinel.
            return 0.0, 0.0, self._tt_max, 0.0

        x0 = min(int(u), cols - 2)
        y0 = min(int(w), rows - 2)
        fx = u - x0
        fy = w - y0

        def _bilerp(arr: np.ndarray) -> float:
            return float(
                arr[y0,     x0]     * (1.0 - fx) * (1.0 - fy)
                + arr[y0,     x0 + 1] * fx       * (1.0 - fy)
                + arr[y0 + 1, x0]     * (1.0 - fx) * fy
                + arr[y0 + 1, x0 + 1] * fx       * fy
            )

        Fx = _bilerp(self._gx)
        Fy = _bilerp(self._gy)
        n  = (Fx * Fx + Fy * Fy) ** 0.5
        if n > 1e-6:
            Fx /= n
            Fy /= n
        else:
            Fx, Fy = 0.0, 0.0

        return Fx, Fy, _bilerp(self._tt), _bilerp(self._mag)


# ---------------------------------------------------------------------------
# Reference trajectory generator (field-following inner loop)
# ---------------------------------------------------------------------------


def _wrap_to_pi(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


class ReferenceGenerator:
    """Forward-integrate a feasible reference trajectory along the field.

    Heading tracks atan2(Fy, Fx) via a saturated proportional controller;
    forward speed is scaled by max(cos(heading_err), 0) so the robot
    decelerates and turns in place when sharply misaligned. The output
    is used both as the SQP warm-start and as the per-stage parameters
    of the cost.
    """

    def __init__(self, cfg: PlannerConfig, field: VectorFieldGrid):
        self.cfg = cfg
        self.field = field

    def generate(
        self, x0: np.ndarray, goal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg       = self.cfg
        N         = cfg.N
        dt        = cfg.dt
        v_max     = cfg.v_max
        omega_max = cfg.omega_max
        a_max     = cfg.a_max
        alpha_max = cfg.alpha_max
        L_brake   = cfg.L_brake
        K_omega   = cfg.ref_K_omega

        seed_x = np.zeros((N + 1, 5))
        seed_u = np.zeros((N, 2))
        seed_x[0] = x0.copy()

        gx, gy = float(goal[0]), float(goal[1])

        for k in range(N):
            px, py, th, v, om = seed_x[k]
            Fx, Fy, _, _ = self.field.query(px, py)
            f_norm = (Fx * Fx + Fy * Fy) ** 0.5

            if f_norm < 1e-3:
                a_cmd     = max(min(-v  / dt, a_max),     -a_max)
                alpha_cmd = max(min(-om / dt, alpha_max), -alpha_max)
            else:
                psi_d = atan2(Fy, Fx)
                e     = _wrap_to_pi(psi_d - th)
                omega_des = max(min(K_omega * e, omega_max), -omega_max)
                alpha_cmd = max(min((omega_des - om) / dt, alpha_max), -alpha_max)
                d_goal    = hypot(px - gx, py - gy)
                v_target  = v_max * tanh(d_goal / L_brake)
                v_des     = v_target * max(cos(e), cfg.ref_brake_cos)
                a_cmd     = max(min((v_des - v) / dt, a_max), -a_max)

            seed_u[k] = (a_cmd, alpha_cmd)
            seed_x[k + 1] = _rk4_step(seed_x[k], seed_u[k], dt, v_max, omega_max)

        return seed_x, seed_u


def _rk4_step(
    state: np.ndarray, u: np.ndarray, dt: float, v_max: float, omega_max: float,
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
    ns[3] = float(np.clip(ns[3], -v_max,     v_max))
    ns[4] = float(np.clip(ns[4], -omega_max, omega_max))
    return ns


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------


class PlannerNode(Node):

    def __init__(self):
        super().__init__("pmp_planner")

        # Dataclass-driven parameter loading. Adding a new tunable means
        # adding one line to PlannerConfig / TopicConfig / SolverConfig.
        self.cfg        = declare_and_load_dataclass(self, PlannerConfig())
        self.topic_cfg  = declare_and_load_dataclass(self, TopicConfig())
        self.solver_cfg = declare_and_load_dataclass(self, SolverConfig())

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._xi: np.ndarray = np.zeros(5)
        self._goal: Optional[np.ndarray] = None
        self._field = VectorFieldGrid()
        self._waiting_for_field = False

        self.get_logger().info("Compiling acados solver...")
        self._solver = build_ocp_solver(self.cfg, self.solver_cfg)
        self._reference = ReferenceGenerator(self.cfg, self._field)
        self.get_logger().info(
            f"Solver ready. confidence_weighting="
            f"{self.cfg.enable_confidence_weighting}, "
            f"LM={self.solver_cfg.levenberg_marquardt:.1e}, "
            f"max_iter={self.solver_cfg.nlp_solver_max_iter}"
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(Odometry,           "/odom",                       self._on_odom,  qos)
        self.create_subscription(PoseStamped,        "/goal_pose",                  self._on_goal,  qos)
        self.create_subscription(Float32MultiArray,  "/vector_field/planner_data",  self._on_field, qos)

        if self.topic_cfg.enable_stamped_cmd_vel:
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
        v_body  = msg.twist.twist.linear.x
        omega_b = msg.twist.twist.angular.z
        try:
            t = self._tf_buffer.lookup_transform(
                self.topic_cfg.map_frame, self.topic_cfg.robot_frame,
                rclpy.time.Time(),
            )
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            q  = t.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self._xi = np.array([tx, ty, yaw, v_body, omega_b])
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self.topic_cfg.map_frame}->"
                f"{self.topic_cfg.robot_frame} unavailable: {e}",
                throttle_duration_sec=2.0,
            )

    def _on_goal(self, msg: PoseStamped):
        pos = msg.pose.position
        q   = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._goal = np.array([pos.x, pos.y, yaw])
        self._waiting_for_field = True
        self.get_logger().info(f"Goal: ({pos.x:.2f}, {pos.y:.2f})")

    def _on_field(self, msg: Float32MultiArray):
        """Parse the vector-field message.

        Supports both formats:
          legacy : [h, w, ox, oy, res, T, gx, gy]               size = 5 + 3*n
          new    : [h, w, ox, oy, res, T, gx, gy, |grad T|]     size = 5 + 4*n

        The legacy format is treated as "no magnitude information"; the
        VectorFieldGrid fills magnitudes with 1.0, and confidence
        weighting (if enabled) ends up evaluating to 1.0 everywhere.
        """
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 5:
            return
        h        = int(data[0])
        w        = int(data[1])
        origin_x = float(data[2])
        origin_y = float(data[3])
        resolution = float(data[4])
        n = h * w

        if data.size == 5 + 4 * n:
            tt  = data[5         : 5 + n        ].reshape(h, w)
            gx  = data[5 +     n : 5 + 2 * n    ].reshape(h, w)
            gy  = data[5 + 2 * n : 5 + 3 * n    ].reshape(h, w)
            mag = data[5 + 3 * n : 5 + 4 * n    ].reshape(h, w)
        elif data.size == 5 + 3 * n:
            tt  = data[5         : 5 + n        ].reshape(h, w)
            gx  = data[5 +     n : 5 + 2 * n    ].reshape(h, w)
            gy  = data[5 + 2 * n : 5 + 3 * n    ].reshape(h, w)
            mag = None
            self.get_logger().warn(
                "Vector field message has no |grad T| channel "
                "(legacy format); confidence weighting will be inactive.",
                throttle_duration_sec=30.0,
            )
        else:
            self.get_logger().warn(
                f"Field size mismatch: got {data.size}, "
                f"expected {5 + 3*n} (legacy) or {5 + 4*n} (new)",
                throttle_duration_sec=5.0,
            )
            return

        self._field.update(
            tt, gx, gy, mag, origin_x, origin_y, resolution,
            mag_percentile=self.cfg.confidence_mag_percentile,
        )
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

        xi_next = self._solver.get(1, "x")
        self._publish_twist(float(xi_next[3]), float(xi_next[4]))
        self._publish_trajectory()

    # ---------------- MPC solve ----------------

    def _confidence_at(self, mag: float) -> float:
        """Map raw |grad T| at a stage to a confidence factor in [floor, 1].

        Returns 1.0 unconditionally when confidence weighting is disabled,
        which makes the cost expression bit-identical to the unweighted
        version. The mag_ref percentile and gamma shaping live in cfg.
        """
        if not self.cfg.enable_confidence_weighting:
            return 1.0
        ref = self._field.mag_ref
        if ref < 1e-8:
            return 1.0
        ratio = max(0.0, min(mag / ref, 1.0))
        if self.cfg.confidence_gamma != 1.0:
            ratio = ratio ** self.cfg.confidence_gamma
        return max(self.cfg.confidence_floor, ratio)

    def _solve_mpc(self) -> Optional[np.ndarray]:
        """Run one SQP-RTI iteration. Returns u_0 on success, None on failure."""
        solver = self._solver
        N      = self.cfg.N

        ref_x, ref_u = self._reference.generate(self._xi, self._goal)

        for k in range(N + 1):
            solver.set(k, "x", ref_x[k])
        for k in range(N):
            solver.set(k, "u", ref_u[k])

        solver.set(0, "x",   self._xi)
        solver.set(0, "lbx", self._xi)
        solver.set(0, "ubx", self._xi)

        v_max     = self.cfg.v_max
        L_brake   = self.cfg.L_brake
        T_ref_cap = self.cfg.T_ref_cap
        gx, gy    = float(self._goal[0]), float(self._goal[1])

        for k in range(N + 1):
            px_ref = float(ref_x[k][0])
            py_ref = float(ref_x[k][1])
            Fx, Fy, T_ref, mag = self._field.query(px_ref, py_ref)
            T_ref = min(T_ref, T_ref_cap)

            d   = hypot(px_ref - gx, py_ref - gy)
            v_r = v_max * tanh(d / L_brake)

            c = self._confidence_at(mag)

            solver.set(k, "p", np.array([Fx, Fy, v_r, T_ref, px_ref, py_ref, c]))

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
        if self.topic_cfg.enable_stamped_cmd_vel:
            msg = TwistStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = self.topic_cfg.robot_frame
            msg.twist.linear.x  = v
            msg.twist.angular.z = omega
        else:
            msg = Twist()
            msg.linear.x  = v
            msg.angular.z = omega
        self._cmd_pub.publish(msg)

    def _publish_trajectory(self):
        now  = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp    = now
        path.header.frame_id = self.topic_cfg.map_frame
        for k in range(self.cfg.N + 1):
            xi_k = self._solver.get(k, "x")
            pose = PoseStamped()
            pose.header.stamp    = now
            pose.header.frame_id = self.topic_cfg.map_frame
            pose.pose.position.x = float(xi_k[0])
            pose.pose.position.y = float(xi_k[1])
            yaw = float(xi_k[2])
            pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.pose.orientation.w = float(np.cos(yaw / 2.0))
            path.poses.append(pose)
        self._traj_pub.publish(path)

    def _publish_empty_trajectory(self):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.topic_cfg.map_frame
        self._traj_pub.publish(msg)

    # ---------------- PMP introspection (optional) ----------------

    def extract_costates(self) -> list[np.ndarray]:
        dt = self.cfg.dt
        return [-self._solver.get(k, "pi") / dt for k in range(self.cfg.N)]

    def extract_predicted_trajectory(self) -> np.ndarray:
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
