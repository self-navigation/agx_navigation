"""Vector-field guided indirect-method PMP planner for a unicycle.

Solves the optimal-control problem via Pontryagin's Maximum Principle:
the Hamiltonian, costate ODEs and the optimal-control law are derived
analytically; the resulting two-point boundary value problem (TPBVP)
is integrated with scipy.integrate.solve_bvp.

Model -- 3D kinematic unicycle:
  state    x = (p_x, p_y, theta)
  control  u = (v, omega),  |v| <= v_max, |omega| <= omega_max
  dynamics x_dot = v cos(theta), y_dot = v sin(theta), theta_dot = omega

Cost:
  L(x, u) = alpha_t + L_pos(T(p))                           # piecewise C^1 pot.
          + w_F * w_h * (1 - F_unit(p) . h(theta))          # field alignment
          + (1 - w_F) * (1/2) * w_h * (theta - theta_p)^2   # terminal-yaw spring
          + (1/2) * w_v * (v - v_ref_eff(p, theta))^2       # speed reference
          + (1/2) * w_brake * (1 - F_unit . h)^2 * v^2      # heading-coupled brake
          + gamma_v * v^2 + gamma_w * omega^2               # control regularizers

  L_pos(T) = (beta/2) * T^2 / T_horizon              if T <= T_horizon
           = beta * (T - T_horizon/2)                if T >  T_horizon
  (quadratic near the goal so the gradient fades to zero at the sink,
   linear during navigation so it doesn't swamp w_h on routed paths;
   gradient = beta * min(T, T_horizon) * grad(T) / T_horizon, C^0 at
   the join, C^-1 in the second derivative.)

  Phi(x_T) = (1/2) * w_pp * ||p_T - p_pursuit||^2           # pursuit-point pull
           + (1/2) * w_th * (theta_T - theta_pursuit)^2     # terminal yaw target

with
  v_ref(p)        = v_max * tanh(||p - p_goal|| / L_brake)
  gate(x)         = ((1 + x) / 2) ** p_gate    in [0, 1]
  v_ref_eff(p,th) = v_ref(p) * gate(F_unit . h(theta))

The non-negative gate replaces the older `v_ref * (F . h)` heading-aware
target: when |F . h| < 1 the forward target fades, so the cost no longer
asks for reverse motion under any heading. The brake term separately
penalises any v != 0 in proportion to the SQUARE of the misalignment;
quadratic-in-(1 - F . h) buys two properties at once:
  - gentle near alignment (small misalignment -> tiny brake), so the
    chassis follows F-curvature smoothly through corners and stays
    clean during straight-line drive on a noisy interpolated F field;
  - strong at large misalignment (4 w_brake at anti-aligned), enough
    mass to overpower the position-pursuit costates that would
    otherwise pull the BVP toward reverse-while-turning.
The lambda_th_dot brake contribution picks up a (1 - F . h) prefactor
that vanishes at alignment, so heading dynamics aren't pumped by
cross_F_h jitter during forward drive.

Together: misaligned -> v ~ 0 + omega != 0 (pure rotation), aligned ->
drive forward at v_ref. The mechanism is general -- it handles the
goal-yaw fix, an initial heading mismatch, and sharp F-curvature
mid-trajectory through one cost shape, removing the historical
TURN_IN_PLACE supervisor.

The terminal cost is **field-following**: p_pursuit is the streamline
endpoint traced from x_now for arc length v_max * T_horizon *
pursuit_lookahead_mult, and theta_pursuit is the F-tangent at
p_pursuit. Both targets sit on the field's flow line, so the BVP is
not asked to land at the *goal* (which would bias toward shortcuts
when far from goal) but at the appropriate point along the streamline.
When the streamline reaches the goal sink within the lookahead
distance, p_pursuit collapses to p_goal and theta_pursuit falls back
to theta_goal -- so goal arrival behaves as expected.

Hamiltonian (minimum-principle convention):
  H = L + lambda_x * v cos(theta) + lambda_y * v sin(theta) + lambda_th * omega

Closed-form optimal control (tanh-saturated to bounds):
  denom_v     = 2 gamma_v + w_v + w_brake * (1 - F . h)^2
  v_unsat     = (w_v * v_ref_eff - lambda_x cos(theta) - lambda_y sin(theta)) / denom_v
  omega_unsat = -lambda_th / (2 gamma_w)

Costate ODEs (lambda_dot = -dH/dx), frozen-field approximation in the
position costates (dF_unit/dp and dv_ref/dp dropped). The speed-reference
term contributes via gate'(F . h), and the brake's quadratic form yields
a (1 - F . h) prefactor in lambda_th_dot:
  gate'(x)   = (p_gate / 2) * ((1 + x) / 2) ** (p_gate - 1)
  cross_F_h  = F_x sin(theta) - F_y cos(theta)
  lambda_x_dot  = -beta * min(T, T_horizon) * dT/dx / T_horizon
  lambda_y_dot  = -beta * min(T, T_horizon) * dT/dy / T_horizon
  lambda_th_dot = -w_F * w_h * cross_F_h
                  - w_F * w_v * v_ref(p) * (v - v_ref_eff) * gate'(F . h) * cross_F_h
                  - w_F * w_brake * (1 - F . h) * v^2 * cross_F_h
                  - (1 - w_F) * w_h * (theta - theta_pursuit)
                  + lambda_x * v sin(theta) - lambda_y * v cos(theta)

Boundary conditions:
  t = 0 :  x(0) = x_now              (initial pose pinned)
  t = T :  lambda_x(T)  = 2 w_pp (p_x_T - p_x_pursuit)
           lambda_y(T)  = 2 w_pp (p_y_T - p_y_pursuit)
           lambda_th(T) = w_th * (theta_T - theta_pursuit)

Goal handling: a single REACHED check sits on top of the BVP -- when
both position and yaw tolerances are met, the planner publishes zero
twist. The historical TURN_IN_PLACE supervisor (pure-rotate at the goal
to fix yaw) is no longer needed: at the goal the chassis has
d_to_goal ~ 0, so v_ref_eff ~ 0 regardless of misalignment, and the
BVP emits v ~ 0 + omega != 0 -- pure rotation -- on its own. The same
mechanism handles initial-heading mismatches and sharp F-curvature
mid-trajectory, with no special-case code anywhere.

Field-update reactivity: VectorFieldGrid carries a version counter; a
new field invalidates the warm start so the BVP doesn't initialize
near a now-stale optimum.

Node API: subscribes to /odom, /goal_pose, /vector_field/planner_data;
publishes Twist (or TwistStamped) on /cmd_vel and a Path on
/pmp_planner/trajectory.
"""

from dataclasses import dataclass, fields, replace
from math import hypot, atan2, pi, sin, cos, tanh
from typing import Any, Optional, Tuple

import numpy as np
from scipy.integrate import solve_bvp
from scipy.ndimage import gaussian_filter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException


# ---------------------------------------------------------------------------
# Generic dataclass-driven ROS2 parameter loader
# ---------------------------------------------------------------------------


def declare_and_load_dataclass(
    node: Node, instance: Any, prefix: str = "",
) -> Any:
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
    """Indirect-method planner parameters."""

    # --- Horizon ---
    N: int = 21                         # mesh nodes for the BVP initial guess
    T_horizon: float = 2.5              # [s] prediction horizon
    control_rate: float = 10.0          # [Hz]; warm-start solves are typically <30 ms

    # --- Control bounds (also act as tanh saturation in _ode) ---
    v_max: float = 0.5                  # [m/s], symmetric
    omega_max: float = 1.5              # [rad/s]

    # --- Running-cost weights ---
    alpha_t: float = 1.0                # constant time penalty
    beta:    float = 5.0                # T(p)^2 potential pull. The
                                        # cost is (beta/2)*T^2/T_horizon
                                        # so the gradient -beta*T*grad(T)
                                        # /T_horizon fades at the goal
                                        # sink (T->0), letting v_ref
                                        # dominate v_unsat near goal
                                        # for clean braking.
    w_h:     float = 5.0                # alignment (1 - F_unit . h).
                                        # Tuning: clearance is non-monotonic
                                        # in w_h. Both w_h <= 3 (position
                                        # cost dominates) and w_h >= 12
                                        # (rigid F-tracking) clear corners
                                        # well; the in-between regime can
                                        # find shortcuts.
    w_v:     float = 0.5                # speed-reference (v - v_ref_eff(p, theta))^2
    # Heading-coupled brake on v^2: penalises any v != 0 in proportion to
    # the SQUARE of the heading misalignment. Cost form
    #   (1/2) * w_brake * (1 - F . h)^2 * v^2
    # is zero when aligned and contributes
    #   (1/2) * w_brake * v^2  at perpendicular,
    #   2 * w_brake * v^2      at anti-aligned.
    # Magnitude needs to overpower the position-pursuit costates' pull
    # on v -- empirically those scale as 2 * w_pp times the chassis-to-
    # pursuit distance plus beta * T_horizon, putting |lambda . h| in
    # the 10-30 range under the iter-7 weights.
    #
    # Quadratic-in-(1 - F . h) (vs an earlier linear form) buys two key
    # properties:
    #   * gentle near alignment: at 30 deg misalignment the brake denom
    #     contribution is w_brake * 0.018 ~ 4 (vs 27 with linear), so the
    #     chassis follows F-curvature smoothly through corners and stays
    #     clean during straight-line drive on a noisy interpolated F field.
    #   * vanishing lambda_th_dot at alignment: the brake's theta-derivative
    #     picks up a (1 - F . h) prefactor which is *exactly* zero at
    #     F . h = 1, so cross_F_h jitter from bilinear interp doesn't pump
    #     omega during forward drive.
    #
    # 200 is the chosen default. Set to 0 to disable (re-introduces the
    # reverse-while-turning shuffle at misaligned configurations; useful
    # only for debugging the gate's behaviour in isolation).
    w_brake: float = 200.0
    L_brake: float = 0.5                # [m]; v_ref scales as
                                        # v_max * tanh(d_to_goal / L_brake).
                                        # Set near the chassis stopping distance.
    # Sharpness of the heading-alignment gate that multiplies v_ref:
    #   gate(F . h) = ((1 + F . h) / 2) ** align_gate_power.
    # Higher values force a sharper drive/turn split: at p=4, gate(perp)
    # = 0.06 (chassis effectively brakes when perpendicular to the field
    # and pure-rotates), while gate(aligned) stays at 1. p=2 gives
    # racing-line cornering (gate(perp) = 0.25). p=8 is near-binary.
    # Setting p=0 disables the gate entirely (cost reduces to a plain
    # (v - v_ref)^2 with v_ref >= 0 -- chassis always tries to drive
    # forward at v_ref regardless of heading; this re-introduces the
    # at-goal yaw-fix shuffle, so do not use unless you re-enable a
    # supervisor downstream).
    align_gate_power: float = 4.0
    gamma_v: float = 0.5                # quadratic regularizer on v
    gamma_w: float = 0.2                # quadratic regularizer on omega.
                                        # Higher values stiffen ALL angular
                                        # dynamics globally, which can make
                                        # tight maneuvers infeasible.

    # --- Field smoothing ---
    # F_unit = F_raw / sqrt(|F_raw|^2 + field_eps^2). Smooth re-norm makes
    # |F_unit| -> 0 where |F_raw| -> 0 (goal sink, saddles), so the
    # alignment cost fades there instead of fighting the terminal yaw
    # target. Set comparable to the upstream gradient noise floor.
    field_eps: float = 1e-2

    # In-planner Gaussian smoothing on T (in grid cells) before deriving
    # F. Smooths corner curvature in the alignment direction; preserves
    # the goal singularity (smooths T, not F directly). Default off.
    # Reach for it (sigma 2..4) only if upstream T is unusually sharp.
    align_smooth_sigma: float = 0.0

    # --- Terminal-cost weights ---
    # Phi_pos(p_T) = (1/2) * w_pp * ||p_T - p_pursuit||^2,  with
    #     p_pursuit     = streamline endpoint at arc-length lookahead
    #     theta_pursuit = atan2(F_y, F_x) at p_pursuit, fallback theta_goal
    # The terminal targets sit on the field's flow line, removing the
    # corner-cut bias the old Lyapunov-T terminal had.
    w_pp: float = 5.0                   # pursuit-point pull. Raise toward
                                        # ~10 if endpoint feels under-pulled,
                                        # lower if Newton struggles.
    w_th: float = 2.0                   # terminal heading basin (1 - cos).

    # Pursuit-point lookahead, as a multiple of v_max * T_horizon (the
    # BVP's open-space reach distance). 1.0 lands the target where the
    # chassis would be at horizon end if it tracked F at full speed.
    # < 1 leaves slack at the endpoint (eases Newton on hard fields);
    # > 1 makes the BVP "reach" past comfortably reachable (tighter
    # tracking, harder convergence).
    pursuit_lookahead_mult: float = 1.0

    # --- Cross-track residual (opt-in, off by default) ---
    # Adds (1/2)*w_xt*r_xt(p)^2 to the running cost, where r_xt is the
    # signed perpendicular drift from the F-streamline at a Gaussian-
    # weighted projection. Penalises lateral drift directly (alignment
    # only constrains heading). Off by default because empirically the
    # BVP either doesn't see this term (low w_xt) or stalls at it (high
    # w_xt); kept as a knob for future cost structures that condition it
    # better. See PMP_INDIRECT_NOTES sec 4.8.
    w_xt:         float = 0.0
    xt_horizon_m: float = 2.5            # [m] streamline arc length to trace.
                                         # Should exceed v_max * T_horizon.
    xt_sigma_mult: float = 3.0           # Gaussian bandwidth = mult * trace_ds

    # --- Goal tolerances ---
    # Both must be met for REACHED (zero twist). The at-goal yaw fix
    # itself is handled implicitly by the BVP via the gated v_ref --
    # there is no longer a TURN_IN_PLACE supervisor stage.
    goal_tolerance_xy: float = 0.05     # [m]
    goal_tolerance_th: float = 0.20     # [rad]

    # --- BVP solver knobs ---
    bvp_tol: float = 1e-3               # solve_bvp residual tolerance
    bvp_max_nodes: int = 2000           # adaptive-mesh cap; raise if the
                                        # field has unusually sharp features.
    bvp_verbose: int = 0                # 0 silent, 1 summary, 2 per-iter

    # Reuse previous solution as the initial guess for the next solve.
    # Auto-dropped on (a) a new goal, (b) a new vector-field message
    # (tracked via VectorFieldGrid.version).
    reuse_previous_solution: bool = True

    @property
    def dt(self) -> float:
        return self.T_horizon / self.N


@dataclass
class TopicConfig:
    map_frame: str = "map"
    robot_frame: str = "base_link"
    enable_stamped_cmd_vel: bool = False


# ---------------------------------------------------------------------------
# Vector field grid
# ---------------------------------------------------------------------------


class VectorFieldGrid:
    """T(x, y), its gradient, and a unit-vector direction field F.

    Two field sources are kept because they serve different purposes:

    - dT/dx, dT/dy: recomputed internally with np.gradient so that the
      derivative used in the position-costate ODEs (beta * grad T) is
      consistent with the T grid actually being penalised, regardless
      of upstream smoothing or sign convention.
    - F_unit: derived from the upstream's published (Fx, Fy) channels
      and re-normalised here with eps regularisation so |F_unit| -> 0
      where the underlying field magnitude collapses (goal sink, flat
      regions, saddles).

    Sign convention: (Fx, Fy) is treated as the "follow this direction"
    field. If the upstream publishes the raw +grad T (away from goal),
    flip the sign upstream or in _on_field.
    """

    def __init__(self):
        self._tt: Optional[np.ndarray] = None
        self._dT_dx: Optional[np.ndarray] = None
        self._dT_dy: Optional[np.ndarray] = None
        self._Fu_x: Optional[np.ndarray] = None
        self._Fu_y: Optional[np.ndarray] = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._res = 1.0
        self._tt_max = 1.0
        self._ready = False
        # Monotonic counter; bumped on every successful update so the solver
        # can detect a replaced field and drop a now-stale warm start.
        self._version = 0

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def version(self) -> int:
        """Monotonic counter; changes => warm start is no longer valid."""
        return self._version

    def update(
        self,
        T_field: np.ndarray,
        Fx_field: Optional[np.ndarray],
        Fy_field: Optional[np.ndarray],
        origin_x: float, origin_y: float, resolution: float,
        field_eps: float = 1e-2,
        align_smooth_sigma: float = 0.0,
    ):
        T = T_field.astype(np.float64)
        # FMM may produce inf for unreachable cells; replace with the largest
        # finite value before differentiation so the gradient stays finite.
        finite_mask = np.isfinite(T)
        if finite_mask.any():
            T_max_finite = float(T[finite_mask].max())
        else:
            T_max_finite = 1.0
        T_filled = np.where(finite_mask, T, T_max_finite)

        # np.gradient returns (dT/drow, dT/dcol). ROS map convention:
        # rows index y, cols index x. Position costates use the raw
        # (un-smoothed) gradient so the position penalty stays sharp.
        d_drow, d_dcol = np.gradient(T_filled, resolution, resolution)

        # Alignment direction field. Three sources, in priority order:
        #   1. align_smooth_sigma > 0: derive from grad(gaussian_filter(T)).
        #      Smooths corner curvature in the alignment direction while
        #      preserving the goal singularity (we smooth T, not F).
        #   2. Upstream-provided (Fx, Fy): use as-is.
        #   3. Fallback: derive from -grad T (legacy 1-channel message).
        if align_smooth_sigma > 0.0:
            T_align = gaussian_filter(T_filled, sigma=align_smooth_sigma)
            d_drow_align, d_dcol_align = np.gradient(
                T_align, resolution, resolution,
            )
            Fx = -d_dcol_align
            Fy = -d_drow_align
        elif Fx_field is not None and Fy_field is not None:
            Fx = Fx_field.astype(np.float64)
            Fy = Fy_field.astype(np.float64)
        else:
            Fx = -d_dcol
            Fy = -d_drow
        # Smooth re-normalize: |F_unit| -> 1 for |F| >> eps and -> 0 for
        # |F| << eps. The latter fades the alignment cost in flat regions.
        norm = np.sqrt(Fx * Fx + Fy * Fy + field_eps * field_eps)
        Fu_x = Fx / norm
        Fu_y = Fy / norm

        self._tt = T_filled
        self._dT_dx = d_dcol
        self._dT_dy = d_drow
        self._Fu_x = Fu_x
        self._Fu_y = Fu_y
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._res = resolution
        self._tt_max = T_max_finite
        self._ready = True
        self._version += 1

    def query_vec(
        self, px: np.ndarray, py: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized bilinear sample at world (px, py).

        Returns (T, dT/dx, dT/dy, F_unit_x, F_unit_y). World (origin_x,
        origin_y) is the corner of cell (0, 0); grid index = (world -
        origin) / resolution. Out-of-bounds: returns zero gradient and
        zero direction (no force) and the max-T sentinel (still penalised).
        """
        if not self._ready:
            zero = np.zeros_like(px)
            return zero.copy(), zero.copy(), zero.copy(), zero.copy(), zero.copy()

        u = (px - self._origin_x) / self._res
        w = (py - self._origin_y) / self._res

        rows, cols = self._tt.shape
        in_bounds = (u >= 0) & (u <= cols - 1) & (w >= 0) & (w <= rows - 1)
        u_c = np.clip(u, 0.0, cols - 1.0001)
        w_c = np.clip(w, 0.0, rows - 1.0001)

        x0 = u_c.astype(int)
        y0 = w_c.astype(int)
        fx = u_c - x0
        fy = w_c - y0

        def bilerp(arr: np.ndarray) -> np.ndarray:
            return (arr[y0,     x0    ] * (1.0 - fx) * (1.0 - fy)
                  + arr[y0,     x0 + 1] * fx         * (1.0 - fy)
                  + arr[y0 + 1, x0    ] * (1.0 - fx) * fy
                  + arr[y0 + 1, x0 + 1] * fx         * fy)

        T  = bilerp(self._tt)
        dx = bilerp(self._dT_dx)
        dy = bilerp(self._dT_dy)
        fux = bilerp(self._Fu_x)
        fuy = bilerp(self._Fu_y)

        # Out-of-bounds: see docstring.
        T   = np.where(in_bounds, T,   self._tt_max)
        dx  = np.where(in_bounds, dx,  0.0)
        dy  = np.where(in_bounds, dy,  0.0)
        fux = np.where(in_bounds, fux, 0.0)
        fuy = np.where(in_bounds, fuy, 0.0)
        return T, dx, dy, fux, fuy

    def query_scalar(
        self, px: float, py: float,
    ) -> Tuple[float, float, float, float, float]:
        T, dx, dy, fux, fuy = self.query_vec(np.array([px]), np.array([py]))
        return float(T[0]), float(dx[0]), float(dy[0]), float(fux[0]), float(fuy[0])

    def trace_streamline(
        self,
        x0: float, y0: float,
        length_m: float,
        ds: Optional[float] = None,
        goal_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trace dp/ds = F_unit(p) starting at (x0, y0) for up to length_m.

        Stops on (a) |F| < 1e-3 (goal sink or saddle), (b) length_m of arc
        length, (c) leaving the grid, (d) within sqrt(0.01) m of goal_xy.

        Returns (ref_pts (N, 2), n_perp (N, 2)) where n_perp is the unit
        normal rotated 90 deg CCW from F at each sample. Empty arrays if
        the trace cannot start.
        """
        if not self._ready:
            return np.zeros((0, 2)), np.zeros((0, 2))
        if ds is None:
            ds = self._res
        n_max = max(8, int(np.ceil(length_m / ds)) + 1)

        ref_x = np.empty(n_max, dtype=np.float64)
        ref_y = np.empty(n_max, dtype=np.float64)
        nx = np.empty(n_max, dtype=np.float64)
        ny = np.empty(n_max, dtype=np.float64)

        px, py = float(x0), float(y0)
        n = 0
        for _ in range(n_max):
            _, _, _, fux, fuy = self.query_scalar(px, py)
            mag = float(np.hypot(fux, fuy))
            if mag < 1e-3:
                break
            # F unit tangent and 90-deg-CCW perpendicular
            tx, ty = fux / mag, fuy / mag
            ref_x[n] = px
            ref_y[n] = py
            nx[n] = -ty   # 90-CCW rotation of (tx, ty) is (-ty, tx)
            ny[n] = tx
            n += 1
            if goal_xy is not None:
                if (px - goal_xy[0]) ** 2 + (py - goal_xy[1]) ** 2 < 1e-2:
                    break
            px = px + ds * tx
            py = py + ds * ty
        if n == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        ref_pts = np.column_stack([ref_x[:n], ref_y[:n]])
        n_perp = np.column_stack([nx[:n], ny[:n]])
        return ref_pts, n_perp


# ---------------------------------------------------------------------------
# Indirect-method PMP solver
# ---------------------------------------------------------------------------


class PMPShootingSolver:
    """TPBVP solver for the unicycle PMP problem.

    The optimal controls are eliminated analytically via dH/du = 0 and
    tanh-saturated, so scipy.integrate.solve_bvp only ever sees 6 ODEs
    (3 state + 3 costate) -- no decision variables.

    Warm-starting (reuse the previous solution as the next initial guess)
    is a major convergence aid when the goal and field are stable.
    """

    def __init__(self, cfg: PlannerConfig, field: VectorFieldGrid):
        self.cfg = cfg
        self.field = field
        self._prev_sol = None
        # Cached per-solve for the BC/ODE closures.
        self._x0: Optional[np.ndarray] = None
        self._goal: Optional[np.ndarray] = None
        # Last successful trajectory for introspection / publishing.
        self._last_state: Optional[np.ndarray] = None      # (m, 3)
        self._last_costate: Optional[np.ndarray] = None    # (m, 3)
        self._last_error: Optional[str] = None
        # Field-version tracking: a replaced field invalidates the warm start.
        self._last_field_version: int = -1
        # Terminal targets (set per-solve in solve()):
        #   p_pursuit     -- streamline endpoint at the lookahead distance.
        #   theta_pursuit -- F-tangent at p_pursuit, falling back to
        #                    theta_goal when |F| collapses (near goal sink).
        self._p_pursuit: np.ndarray = np.zeros(2, dtype=np.float64)
        self._theta_pursuit: float = 0.0
        # Field-alignment-cost fade: scalar in [0, 1], set in solve() from
        # chassis distance to goal. Default 1.0 = full strength.
        self._align_fade: float = 1.0
        # Cross-track reference (only populated when cfg.w_xt > 0). Empty
        # arrays cause _ode's cross-track block to short-circuit.
        self._xt_ref: np.ndarray = np.zeros((0, 2))
        self._xt_n_perp: np.ndarray = np.zeros((0, 2))
        self._xt_sigma: float = 0.15

    # --- Augmented dynamics ------------------------------------------------

    def _ode(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """RHS of the (state, costate) ODE, vectorized over the BVP mesh.

        y has shape (6, m) where rows are
            [p_x, p_y, theta, lambda_x, lambda_y, lambda_th].
        """
        cfg = self.cfg
        px, py, th = y[0], y[1], y[2]
        lx, ly, lt = y[3], y[4], y[5]
        cos_t = np.cos(th)
        sin_t = np.sin(th)

        T_now, dT_dx, dT_dy, Fux, Fuy = self.field.query_vec(px, py)
        F_dot_h = Fux * cos_t + Fuy * sin_t

        # Heading-aware speed reference, gated by alignment so the cost
        # itself prefers v ~ 0 when |F . h| < 1 -- the BVP then emits pure
        # rotation in misaligned regimes without any external supervisor.
        # Frozen-field: dv_ref/dp dropped from the position costates;
        # the theta-dependence flows into lambda_th via gate'(F . h).
        d_to_goal = np.sqrt((px - self._goal[0]) ** 2
                            + (py - self._goal[1]) ** 2)
        v_ref = np.zeros_like(px)
        v_ref_eff = np.zeros_like(px)
        gate_prime = np.zeros_like(px)
        if cfg.w_v > 0.0:
            v_ref = cfg.v_max * np.tanh(d_to_goal / cfg.L_brake)
            half_one_plus = 0.5 * (1.0 + F_dot_h)
            half_one_plus = np.clip(half_one_plus, 0.0, 1.0)  # numerical safety
            p_gate = cfg.align_gate_power
            gate = half_one_plus ** p_gate
            # gate'(F . h) = (p / 2) * ((1 + F . h) / 2) ** (p - 1).
            # The (p - 1) exponent can dip to 0 for p == 1 (constant) and
            # is mathematically fine for any p >= 1; np.power handles it.
            if p_gate > 0.0:
                gate_prime = (0.5 * p_gate) * (half_one_plus ** (p_gate - 1.0))
            else:
                gate_prime = np.zeros_like(half_one_plus)
            v_ref_eff = v_ref * gate

        # Closed-form unconstrained optimal controls from dH/du = 0.
        # Brake term (1/2) w_brake (1 - F.h)^2 v^2 contributes
        # w_brake (1 - F.h)^2 to the v denominator, ramping from 0 at
        # alignment, w_brake at perpendicular, 4 w_brake at anti-aligned.
        # Quadratic-in-misalignment makes the brake gentle near alignment
        # (near-zero contribution at modest cornering angles, so the chassis
        # follows F-curvature smoothly) while still strong enough at
        # |F.h| <~ 0 to overpower the position-pursuit costates that would
        # otherwise pull v negative. Crucially, the theta-derivative below
        # picks up a (1 - F.h) factor that vanishes at alignment, so a
        # straight-line drive on a slightly noisy F field stays clean.
        one_minus_dot = 1.0 - F_dot_h
        denom_v = 2.0 * cfg.gamma_v + cfg.w_v + cfg.w_brake * one_minus_dot * one_minus_dot
        v_unsat = (cfg.w_v * v_ref_eff
                   - lx * cos_t - ly * sin_t) / denom_v
        w_unsat = -lt / (2.0 * cfg.gamma_w)

        # Smooth tanh saturation. A hard np.clip is the exact PMP saturation
        # for box-constrained u, but its slope kink at the bound breaks
        # solve_bvp's Newton (mesh refines forever). tanh asymptotes to
        # +-u_max with slope 1 at the origin and is C-inf everywhere.
        v = cfg.v_max     * np.tanh(v_unsat / cfg.v_max)
        w = cfg.omega_max * np.tanh(w_unsat / cfg.omega_max)

        dpx = v * cos_t
        dpy = v * sin_t
        dth = w
        # Position costates from a piecewise C^1 potential L_pos:
        #   T <= T_h :  L_pos = (beta/2) * T(p)^2 / T_horizon       (quadratic)
        #   T >  T_h :  L_pos = beta * (T(p) - T_horizon/2)         (linear)
        # The gradient is
        #   dL_pos/dp = beta * min(T, T_horizon) * grad(T) / T_horizon
        # which is continuous at the join and gives:
        #   * goal sink (T -> 0)        -> grad -> 0,  the running pull
        #     stops once the chassis arrives and v_ref dominates v_unsat
        #     for clean braking;
        #   * navigation (T >> T_h)     -> grad -> beta * grad(T),  same
        #     magnitude as the original LINEAR cost. With the unbounded
        #     quadratic form the gradient grew without bound (=beta*T/T_h),
        #     swamping w_h * cross_F_h on routed paths and biasing the BVP
        #     toward the lookahead pursuit point regardless of the local
        #     streamline tangent. On a path that has to round an obstacle,
        #     L_align lost authority and the chassis cut through the wall's
        #     repulsion zone instead of following F.
        # Capping T at T_horizon preserves the brake fix near the goal and
        # restores the original L_align/L_pos balance during navigation.
        T_clip = np.minimum(T_now, cfg.T_horizon)
        dlx = -cfg.beta * T_clip * dT_dx / cfg.T_horizon
        dly = -cfg.beta * T_clip * dT_dy / cfg.T_horizon

        # Cross-track residual (opt-in): adds -w_xt * r_xt * n_perp to
        # (dlx, dly) where r_xt is the signed perpendicular drift from the
        # streamline at a Gaussian-weighted projection. Frozen-reference
        # approximation drops dr_xt/dp_ref. Soft projection (vs nearest-
        # sample / nearest-segment) is what makes Newton converge: the
        # discrete lookups are piecewise in n_perp and the mesh-refinement
        # iteration oscillates at the Voronoi boundaries.
        n_ref = self._xt_ref.shape[0]
        if cfg.w_xt > 0.0 and n_ref >= 2:
            mesh = np.column_stack([px, py])
            sigma2 = max(self._xt_sigma * self._xt_sigma, 1e-9)
            diffs = mesh[:, None, :] - self._xt_ref[None, :, :]
            d2 = (diffs * diffs).sum(axis=2)
            d2_min = d2.min(axis=1, keepdims=True)
            w_raw = np.exp(-(d2 - d2_min) / (2.0 * sigma2))
            w_sum = w_raw.sum(axis=1, keepdims=True)
            w_norm = w_raw / np.maximum(w_sum, 1e-30)
            p_ref = w_norm @ self._xt_ref
            n_lin = w_norm @ self._xt_n_perp
            # Re-normalise the weighted-mean normal back to unit length.
            n_mag = np.sqrt((n_lin * n_lin).sum(axis=1))
            n_perp = n_lin / np.maximum(n_mag[:, None], 1e-9)
            delta = mesh - p_ref
            r_xt = (delta * n_perp).sum(axis=1)
            dlx = dlx - cfg.w_xt * r_xt * n_perp[:, 0]
            dly = dly - cfg.w_xt * r_xt * n_perp[:, 1]

        # Heading costate. Two complementary running heading drives, smoothly
        # cross-faded by self._align_fade (= w_F, the chassis-to-goal
        # smoothstep set by solve()):
        #
        # (1) Field-aligned terms (active when w_F ~ 1, navigation regime),
        #     all proportional to cross_F_h = (F_x sin t - F_y cos t):
        #       L_align = w_h (1 - F.h),           -dL/dtheta = -w_h * cross
        #       L_speed = (1/2) w_v (v - v_ref_eff)^2,
        #         v_ref_eff = v_ref(p) * gate(F.h),
        #         d v_ref_eff / d theta = -v_ref * gate'(F.h) * cross,
        #         -dL_speed/dtheta = -w_v v_ref (v - v_ref_eff) gate'(F.h) cross.
        #       L_brake = (1/2) w_brake (1 - F.h)^2 v^2,
        #         dL_brake/dtheta = w_brake v^2 (1 - F.h) cross,    (the
        #         (1 - F.h) factor makes the brake derivative vanish at
        #         alignment, so a straight drive on a slightly noisy F field
        #         doesn't pump lambda_th and the chassis tracks F cleanly.)
        #
        # (2) Terminal-yaw running spring (active when w_F ~ 0, at-goal
        #     regime):
        #       L_yaw = (1/2) w_h (theta - theta_pursuit)^2,
        #         dL_yaw/dtheta = w_h (theta - theta_pursuit).
        #     Inside the goal ball, w_F = 0 so the field-aligned terms vanish
        #     and (1 - w_F) = 1, leaving only the BC plus this running spring
        #     to drive omega. Without (2), at-goal omega is governed by the
        #     terminal BC alone, with constant-omega LQR solution
        #         omega = w_th * delta / (2 gamma_w + w_th * T_horizon),
        #     bottoming out at delta / T_horizon as w_th -> infinity, which
        #     is too slow (~0.6 rad/s for a quarter turn at T_horizon = 2.5).
        #     With (2) the unconstrained omega ~ -sqrt(w_h / (2 gamma_w)) *
        #     delta, saturating omega_max for moderate delta -- so the
        #     at-goal turn runs at the same speed as the navigation turn.
        #     Quadratic form (linear gradient) avoids the multi-basin
        #     pathology a (1 - cos(.)) form would reintroduce, and
        #     theta_pursuit is already unwrapped relative to chassis_theta
        #     by solve() so (theta - theta_pursuit) is well-defined within
        #     the horizon.
        #
        # Plus the kinematic coupling lx v sin t - ly v cos t from
        # lambda . df/dx (not a running-cost gradient, never faded).
        #
        # The (1) terms are faded by w_F as a per-solve scalar, NOT a
        # per-mesh-point gate -- per-mesh-point fading would suppress the
        # brake-on-misalignment at the trajectory's late mesh points and
        # cause approach overshoot.
        fade = self._align_fade
        cross_F_h = Fux * sin_t - Fuy * cos_t
        delta_yaw = th - self._theta_pursuit
        dlt = (-cfg.w_h * cross_F_h * fade
               - cfg.w_v * v_ref * (v - v_ref_eff) * gate_prime * cross_F_h * fade
               - cfg.w_brake * one_minus_dot * v * v * cross_F_h * fade
               - cfg.w_h * delta_yaw * (1.0 - fade)
               + lx * v * sin_t - ly * v * cos_t)

        return np.vstack([dpx, dpy, dth, dlx, dly, dlt])

    # --- Boundary conditions ----------------------------------------------

    def _bc(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """Six BC residuals; solve_bvp drives these to zero.

        Initial state is pinned to x_now (3 BCs). The terminal three are
        the transversalities from Phi(x_T):
            lambda_x(T)  = 2 w_pp (p_x_T - p_x_pursuit)
            lambda_y(T)  = 2 w_pp (p_y_T - p_y_pursuit)
            lambda_th(T) = w_th sin(theta_T - theta_pursuit)
        """
        cfg = self.cfg
        x0 = self._x0
        ppx = float(self._p_pursuit[0])
        ppy = float(self._p_pursuit[1])
        thp = self._theta_pursuit
        return np.array([
            ya[0] - x0[0],
            ya[1] - x0[1],
            ya[2] - x0[2],
            yb[3] - 2.0 * cfg.w_pp * (yb[0] - ppx),
            yb[4] - 2.0 * cfg.w_pp * (yb[1] - ppy),
            yb[5] - cfg.w_th * (yb[2] - thp),
        ])

    # --- Initial guess ----------------------------------------------------

    def _rollout_guess(
        self, x0: np.ndarray, goal: np.ndarray, t_mesh: np.ndarray,
    ) -> np.ndarray:
        """Forward-rollout a feasible state trajectory descending the field.

        Heuristic: pick a desired heading from F_unit (or -grad T, or the
        goal direction as fallbacks); ramp omega to track it; modulate
        speed by the same alignment gate the cost uses, so the rollout
        naturally pure-rotates when misaligned and drives forward only
        when aligned. When the chassis is already inside the position-
        tolerance ball, the desired heading is theta_goal so the rollout
        spends the remaining horizon rotating toward the goal yaw -- the
        BVP's at-goal degenerate case starts from a sensible warm guess.
        """
        cfg = self.cfg
        m = t_mesh.size
        dt = t_mesh[1] - t_mesh[0] if m > 1 else cfg.T_horizon

        state = np.zeros((3, m))
        state[:, 0] = x0
        px, py, th = float(x0[0]), float(x0[1]), float(x0[2])

        for k in range(1, m):
            d_goal = hypot(px - goal[0], py - goal[1])

            # Desired heading: F_unit when well-defined, then -grad T,
            # then goal direction. Once inside the position tolerance,
            # switch the target heading to theta_goal so the rollout
            # finishes with a pure yaw rotation. Matches the solve()-side
            # theta_pursuit gate, which uses chassis-at-goal as the
            # criterion for switching to goal[2].
            _, dx, dy, fux, fuy = self.field.query_scalar(px, py)
            f_norm = (fux * fux + fuy * fuy) ** 0.5
            if d_goal < cfg.goal_tolerance_xy:
                psi_d = float(goal[2])
            elif f_norm > 0.5:
                psi_d = atan2(fuy, fux)
            else:
                grad_norm = (dx * dx + dy * dy) ** 0.5
                if grad_norm > 1e-6:
                    psi_d = atan2(-dy, -dx)
                else:
                    psi_d = atan2(goal[1] - py, goal[0] - px)

            e = ((psi_d - th + pi) % (2.0 * pi)) - pi
            omega = max(min(2.0 * e, cfg.omega_max), -cfg.omega_max)

            # Same gate as the running cost: ((1 + cos(e)) / 2) ** p.
            # cos(e) = F_unit . h(theta) when psi_d is the F-direction;
            # in the goal-direction or theta_goal fallback branches the
            # geometric meaning shifts but the qualitative effect (brake
            # when misaligned, drive when aligned) is the same.
            half_one_plus = max(0.0, 0.5 * (1.0 + cos(e)))
            gate = half_one_plus ** cfg.align_gate_power
            v_target = cfg.v_max * tanh(d_goal / cfg.L_brake)
            v = v_target * gate

            px += v * cos(th) * dt
            py += v * sin(th) * dt
            th += omega * dt
            state[:, k] = (px, py, th)

        # Linear costate ramp from 0 at t=0 to the terminal transversality
        # evaluated at the rolled-out endpoint. The rollout follows F, so
        # state[:, -1] is near p_pursuit and the ramped lambda is small --
        # which is what solve_bvp's Newton likes.
        ppx = float(self._p_pursuit[0])
        ppy = float(self._p_pursuit[1])
        s = (t_mesh - t_mesh[0]) / max(t_mesh[-1] - t_mesh[0], 1e-9)
        lx_T = 2.0 * cfg.w_pp * (state[0, -1] - ppx)
        ly_T = 2.0 * cfg.w_pp * (state[1, -1] - ppy)
        lt_T = cfg.w_th * (state[2, -1] - self._theta_pursuit)

        return np.vstack([
            state[0], state[1], state[2],
            s * lx_T, s * ly_T, s * lt_T,
        ])

    # --- Solve API --------------------------------------------------------

    def _fallback_command(self) -> Optional[Tuple[float, float]]:
        """Evaluate the previous solution one step ahead when the current
        BVP fails. Closer to the true optimum than zeroing the chassis;
        returns None if no warm start is available so the caller decides.
        """
        if self._prev_sol is None:
            return None
        cfg = self.cfg
        t_eval = min(cfg.dt, cfg.T_horizon)
        try:
            y_eval = self._prev_sol(t_eval)
        except Exception:
            self._prev_sol = None
            return None
        th = float(y_eval[2])
        lx, ly, lt = float(y_eval[3]), float(y_eval[4]), float(y_eval[5])
        v_unsat = -(lx * cos(th) + ly * sin(th)) / (2.0 * cfg.gamma_v)
        w_unsat = -lt / (2.0 * cfg.gamma_w)
        # Hard clip on the fallback path: prev_sol may extrapolate freely.
        v_cmd = float(np.clip(cfg.v_max     * np.tanh(v_unsat / cfg.v_max),
                              -cfg.v_max,     +cfg.v_max))
        w_cmd = float(np.clip(cfg.omega_max * np.tanh(w_unsat / cfg.omega_max),
                              -cfg.omega_max, +cfg.omega_max))
        return v_cmd, w_cmd

    def solve(
        self, x0: np.ndarray, goal: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Solve the TPBVP. Returns (v_cmd, omega_cmd) or None on failure."""
        cfg = self.cfg
        self._x0 = x0
        self._goal = goal

        # A new field is a different optimisation landscape; the cached
        # warm start sits at the OLD field's optimum, so drop it.
        if self.field.version != self._last_field_version:
            self.reset_warm_start()
            self._last_field_version = self.field.version

        # Streamline trace from x_now -- one call serves two consumers:
        # (1) pursuit point + heading, sampled at pursuit_dist arc length;
        # (2) cross-track reference, if w_xt > 0, uses the full trace.
        # We trace at the larger of the two required lengths and slice.
        pursuit_dist = cfg.v_max * cfg.T_horizon * cfg.pursuit_lookahead_mult
        xt_active    = cfg.w_xt > 0.0
        trace_dist   = max(pursuit_dist, cfg.xt_horizon_m if xt_active else 0.0)
        ds_used      = float(self.field._res) if hasattr(self.field, "_res") else 0.05

        if self.field.ready and trace_dist > 0.0:
            ref_pts, n_perp = self.field.trace_streamline(
                float(x0[0]), float(x0[1]),
                length_m=trace_dist, ds=None,
                goal_xy=(float(goal[0]), float(goal[1])),
            )
        else:
            ref_pts = np.zeros((0, 2))
            n_perp  = np.zeros((0, 2))

        # Pursuit position: trace sample at arc length pursuit_dist,
        # clamped to actual trace length (the streamline may stop early
        # at the goal sink, a saddle, or off-grid).
        if ref_pts.shape[0] >= 1:
            n_target = int(np.ceil(pursuit_dist / ds_used))
            n_use    = max(1, min(n_target, ref_pts.shape[0]))
            self._p_pursuit = ref_pts[n_use - 1].astype(np.float64)
        else:
            # Trace cannot start. Falling back to the goal makes the
            # terminal a goal-anchored quadratic, which loses the corner-
            # cut protection but is the only sensible default.
            self._p_pursuit = np.array([goal[0], goal[1]], dtype=np.float64)

        # Pursuit heading: vector-blend the streamline tangent at
        # p_pursuit with goal_yaw, weighted by chassis distance to
        # goal. Inside the goal ball (d <= gt), w_F = 0 -> pure
        # goal_yaw; outside the approach zone (d >= 4*gt), w_F = 1
        # -> pure F-tangent; smooth cubic interpolation in between.
        # The user's own goal_tolerance_xy sets the natural scale,
        # so the regime indicator is a property of the user's config,
        # not an arbitrary internal threshold. (The previous |F|-based
        # weight failed because the FMM gradient at the source isn't
        # reliably small in this setup -- |F| stays near 1 right up to
        # the sink in most cells, leaving theta_pursuit stuck on the
        # streamline tangent when the chassis was already at the goal
        # and the chassis didn't turn to goal_yaw.)
        _, _, _, fux_pp, fuy_pp = self.field.query_scalar(
            float(self._p_pursuit[0]), float(self._p_pursuit[1]),
        )
        chassis_to_goal_sq = ((float(x0[0]) - goal[0]) ** 2
                              + (float(x0[1]) - goal[1]) ** 2)
        # chassis_at_goal stays only as a signal to _control_loop for
        # warm-start drops on regime entry/exit. It no longer affects
        # theta_pursuit or align_fade.
        chassis_at_goal = chassis_to_goal_sq < cfg.goal_tolerance_xy ** 2

        chassis_to_goal = float(np.sqrt(chassis_to_goal_sq))
        gt = max(cfg.goal_tolerance_xy, 1e-3)
        s = float(np.clip((chassis_to_goal - gt) / (3.0 * gt), 0.0, 1.0))
        w_F = s * s * (3.0 - 2.0 * s)   # cubic smoothstep, C^1 continuous

        F_mag_pp = float(np.hypot(fux_pp, fuy_pp))
        inv_F = 1.0 / max(F_mag_pp, 1e-9)
        F_hat_x = fux_pp * inv_F
        F_hat_y = fuy_pp * inv_F
        goal_hat_x = float(np.cos(goal[2]))
        goal_hat_y = float(np.sin(goal[2]))
        target_x = w_F * F_hat_x + (1.0 - w_F) * goal_hat_x
        target_y = w_F * F_hat_y + (1.0 - w_F) * goal_hat_y
        # Antipodal degenerate case: F_hat and g_hat point opposite ways
        # and w_F~0.5 collapses target to the origin, where atan2 is ill-
        # conditioned. Fall back to the dominant unit vector. In practice
        # this only triggers if the streamline tangent at p_pursuit is
        # ~pi away from goal_yaw, which is unusual but possible if the
        # field has been redirected (e.g. obstacle avoidance arcing
        # around).
        target_norm = np.hypot(target_x, target_y)
        if target_norm < 1e-3:
            if w_F >= 0.5:
                self._theta_pursuit = atan2(F_hat_y, F_hat_x)
            else:
                self._theta_pursuit = float(goal[2])
        else:
            self._theta_pursuit = float(atan2(target_y, target_x))

        # The same w_F modulates L_align/L_brake/L_speed-coupling in the
        # heading costate dynamics (consumed by _ode via self._align_fade).
        # When the chassis is at goal, L_align becomes inert and the BC
        # alone shapes theta_T -> goal_yaw. This is a per-solve scalar,
        # NOT a per-mesh-point gate -- per-mesh-point fading would
        # suppress brake-on-misalignment at the trajectory's late mesh
        # points and cause approach overshoot.
        self._align_fade = w_F

        # Unwrap theta_pursuit relative to chassis heading so the terminal
        # heading basin has a single minimum at the shortest-turn target.
        # Without this, Phi_th = w_th*(1 - cos(theta_T - thp)) has stable
        # fixed points at thp + 2*k*pi AND saddle-adjacent stable points
        # near thp + pi (saturated-omega CCW solutions); Newton can latch
        # onto the wrong one, especially when the warm start drifts.
        theta_now = float(x0[2])
        delta = ((self._theta_pursuit - theta_now + pi) % (2.0 * pi)) - pi
        self._theta_pursuit = theta_now + delta

        # Cross-track reference: full trace if the term is enabled.
        if xt_active and ref_pts.shape[0] >= 2:
            self._xt_ref    = ref_pts
            self._xt_n_perp = n_perp
            # Bandwidth set from streamline spacing so the projection
            # blends ~5-7 adjacent samples regardless of map resolution.
            self._xt_sigma  = max(cfg.xt_sigma_mult * ds_used, 1e-3)
        else:
            self._xt_ref    = np.zeros((0, 2))
            self._xt_n_perp = np.zeros((0, 2))

        t_mesh = np.linspace(0.0, cfg.T_horizon, cfg.N + 1)
        if cfg.reuse_previous_solution and self._prev_sol is not None:
            try:
                y_init = self._prev_sol(t_mesh)
                # If the chassis's wrapped theta differs from the warm-start's
                # theta(0) by ~2*pi*k (i.e. the chassis crossed +-pi between
                # cycles), shift the entire warm-start theta trajectory by
                # 2*pi*k. Otherwise re-anchoring y_init[2,0] = x0[2] leaves a
                # 2*pi discontinuity at the start of the mesh that solve_bvp
                # cannot resolve and the adaptive mesh refines forever.
                prev_theta_0 = float(y_init[2, 0])
                n_shift = round((prev_theta_0 - float(x0[2])) / (2.0 * pi))
                if n_shift != 0:
                    y_init[2, :] -= n_shift * 2.0 * pi
                y_init[0:3, 0] = x0
            except Exception:
                y_init = self._rollout_guess(x0, goal, t_mesh)
        else:
            y_init = self._rollout_guess(x0, goal, t_mesh)

        try:
            sol = solve_bvp(
                self._ode, self._bc, t_mesh, y_init,
                tol=cfg.bvp_tol, max_nodes=cfg.bvp_max_nodes,
                verbose=cfg.bvp_verbose,
            )
        except Exception as e:
            self._last_error = f"solve_bvp raised: {e}"
            return self._fallback_command()

        if not sol.success:
            self._last_error = sol.message
            return self._fallback_command()

        self._prev_sol = sol.sol

        # Extract the optimal control at t = 0 via the same closed-form
        # law _ode uses, so the published command matches what the BVP
        # actually solved.
        y0 = sol.y[:, 0]
        px0, py0, th = y0[0], y0[1], y0[2]
        lx, ly, lt = y0[3], y0[4], y0[5]
        cos_t = float(np.cos(th))
        sin_t = float(np.sin(th))

        if cfg.w_v > 0.0:
            _, _, _, Fux0, Fuy0 = self.field.query_scalar(float(px0), float(py0))
            d_to_goal = float(hypot(px0 - goal[0], py0 - goal[1]))
            v_ref0 = cfg.v_max * np.tanh(d_to_goal / cfg.L_brake)
            F_dot_h0 = Fux0 * cos_t + Fuy0 * sin_t
            half_one_plus0 = max(0.0, min(1.0, 0.5 * (1.0 + F_dot_h0)))
            gate0 = half_one_plus0 ** cfg.align_gate_power
            v_ref_eff0 = v_ref0 * gate0
            one_minus0 = 1.0 - F_dot_h0
            denom_v0 = (2.0 * cfg.gamma_v + cfg.w_v
                        + cfg.w_brake * one_minus0 * one_minus0)
            v_unsat = (cfg.w_v * v_ref_eff0
                       - lx * cos_t - ly * sin_t) / denom_v0
        else:
            # No w_v: brake-only coupling. Same denominator structure
            # without the w_v contribution.
            _, _, _, Fux0, Fuy0 = self.field.query_scalar(float(px0), float(py0))
            F_dot_h0 = Fux0 * cos_t + Fuy0 * sin_t
            one_minus0 = 1.0 - F_dot_h0
            denom_v0 = 2.0 * cfg.gamma_v + cfg.w_brake * one_minus0 * one_minus0
            v_unsat = -(lx * cos_t + ly * sin_t) / denom_v0
        w_unsat = -lt / (2.0 * cfg.gamma_w)

        # Smooth tanh saturation, then defensive hard clip.
        v_cmd = float(cfg.v_max     * np.tanh(v_unsat / cfg.v_max))
        w_cmd = float(cfg.omega_max * np.tanh(w_unsat / cfg.omega_max))
        v_cmd = float(np.clip(v_cmd, -cfg.v_max,     +cfg.v_max))
        w_cmd = float(np.clip(w_cmd, -cfg.omega_max, +cfg.omega_max))

        # Densely resample for visualization and downstream introspection.
        t_dense = np.linspace(0.0, cfg.T_horizon, cfg.N + 1)
        y_dense = self._prev_sol(t_dense)
        self._last_state = y_dense[0:3].T
        self._last_costate = y_dense[3:6].T
        self._last_error = None

        return v_cmd, w_cmd

    def reset_warm_start(self):
        self._prev_sol = None


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------


class PlannerNode(Node):

    def __init__(self):
        super().__init__("pmp_planner")

        self.cfg = declare_and_load_dataclass(self, PlannerConfig())
        self.topic_cfg = declare_and_load_dataclass(self, TopicConfig())

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._xi: np.ndarray = np.zeros(3)              # (px, py, theta)
        self._goal: Optional[np.ndarray] = None         # (gx, gy, gtheta)
        self._field = VectorFieldGrid()
        self._waiting_for_field = False

        # Tracks whether the previous control cycle was inside the
        # position-tolerance ball around the goal. The BVP cost landscape
        # is qualitatively different inside (terminal pursuit collapses,
        # the optimal solution is dominated by w_th) vs outside (terminal
        # pulls toward p_pursuit, optimal solution drives + turns), so
        # the warm-started solution from one regime is a poor initial
        # guess for the other and can land Newton in the wrong basin.
        # Drop the warm start on every boundary crossing.
        self._was_in_goal_zone: bool = False

        self._solver = PMPShootingSolver(self.cfg, self._field)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(Odometry,          "/odom",                       self._on_odom,  qos)
        self.create_subscription(PoseStamped,       "/goal_pose",                  self._on_goal,  qos)
        self.create_subscription(Float32MultiArray, "/vector_field/planner_data",  self._on_field, qos)

        if self.topic_cfg.enable_stamped_cmd_vel:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._traj_pub  = self.create_publisher(Path,        "/pmp_planner/trajectory", 10)
        # Published when the goal is reached to signal "no active goal" to all
        # other subscribers of /goal_pose. The sentinel is frame_id == "" --
        # every legitimate goal carries a non-empty frame (e.g. "map"), so the
        # empty string is an unambiguous cleared state. Other nodes should check
        # msg.header.frame_id != "" before treating an incoming PoseStamped as
        # a new goal.
        self._goal_pub  = self.create_publisher(PoseStamped, "/goal_pose",             qos)

        self.create_timer(1.0 / self.cfg.control_rate, self._control_loop)
        self.get_logger().info(
            f"Indirect-PMP planner running at {self.cfg.control_rate} Hz, "
            f"horizon {self.cfg.T_horizon}s / {self.cfg.N + 1} mesh nodes."
        )

    # ---------------- Subscriptions ----------------

    def _on_odom(self, msg: Odometry):
        # Odom serves as a tick; the actual pose is read via TF
        # (map -> base_link). msg.twist is unused by the kinematic model.
        try:
            t = self._tf_buffer.lookup_transform(
                self.topic_cfg.map_frame, self.topic_cfg.robot_frame,
                rclpy.time.Time(),
            )
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            q  = t.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self._xi = np.array([tx, ty, yaw])
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self.topic_cfg.map_frame}->"
                f"{self.topic_cfg.robot_frame} unavailable: {e}",
                throttle_duration_sec=2.0,
            )

    def _on_goal(self, msg: PoseStamped):
        # Ignore the sentinel we publish ourselves on goal completion.
        if msg.header.frame_id == "":
            return
        pos = msg.pose.position
        q   = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._goal = np.array([pos.x, pos.y, yaw])
        self._waiting_for_field = True
        # New goal -> previous BVP solution is no longer a useful warm-start.
        self._solver.reset_warm_start()
        self.get_logger().info(
            f"Goal: ({pos.x:.2f}, {pos.y:.2f}), yaw={yaw:.2f}"
        )

    def _on_field(self, msg: Float32MultiArray):
        """Parse the field message and update the grid.

        Layout (canonical, what the upstream actually publishes):
          [h, w, origin_x, origin_y, resolution,
           travel_time(H*W), grad_x(H*W), grad_y(H*W), grad_mag(H*W)]

        We keep backward compatibility with two older variants:
          - 1-channel (T only): F_unit is auto-derived as -grad(T)/|grad T|.
          - 3-channel (T, gx, gy): grad_mag missing, ignored.
        """
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 5:
            return
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
            return

        T = body[0:n].reshape(h, w)
        if channels >= 3:
            Fx = body[n:2 * n].reshape(h, w)
            Fy = body[2 * n:3 * n].reshape(h, w)
        else:
            Fx = None
            Fy = None
        # The grad_mag channel (if present) is ignored: we re-normalize
        # (Fx, Fy) to a unit field internally with our own eps regularizer.

        self._field.update(T, Fx, Fy, ox, oy, res,
                           field_eps=self.cfg.field_eps,
                           align_smooth_sigma=self.cfg.align_smooth_sigma)
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

        d_xy = hypot(self._xi[0] - self._goal[0], self._xi[1] - self._goal[1])
        # Wrapped heading error magnitude.
        d_th_signed = ((self._goal[2] - self._xi[2] + pi) % (2.0 * pi)) - pi
        d_th = abs(d_th_signed)

        # Drop the warm start on entering or leaving the goal-tolerance
        # ball. The BVP terminal target (theta_pursuit) flips between
        # F-tangent and goal[2] at this boundary (see comment in solve()
        # around the theta_pursuit gate); a stale warm guess from the
        # wrong regime can land Newton on the wrong-direction-then-
        # overshoot mode.
        in_goal_zone = d_xy < self.cfg.goal_tolerance_xy
        if in_goal_zone != self._was_in_goal_zone:
            self._solver.reset_warm_start()
        self._was_in_goal_zone = in_goal_zone

        # Single REACHED check on top of the BVP. When both tolerances are
        # met (strict goal_tolerance_xy, NOT the wider regime band above)
        # the planner publishes zero twist and stops.
        # Everything else -- including the at-goal yaw fix that used to
        # be a dedicated TURN_IN_PLACE supervisor stage -- is handed to
        # the BVP. With the gated v_ref the chassis emits v ~ 0 + omega
        # whenever |F . h| < 1 (or d_to_goal ~ 0), so pure rotation
        # emerges as a property of the cost rather than a special case.
        if d_xy < self.cfg.goal_tolerance_xy and d_th < self.cfg.goal_tolerance_th:
            self._publish_twist(0.0, 0.0)
            self._publish_empty_trajectory()
            self.get_logger().info(
                f"Goal reached (d_xy={d_xy:.3f} m, d_th={d_th:.3f} rad)."
            )
            self._clear_goal()
            return

        result = self._solver.solve(self._xi, self._goal)
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

    # ---------------- Publishing ----------------

    def _clear_goal(self):
        """Clear the active goal locally and signal it ROS-wide on /goal_pose.

        The sentinel is a PoseStamped with frame_id == "".  All other nodes
        that key off /goal_pose should treat an empty frame_id as "no goal".
        """
        sentinel = PoseStamped()
        sentinel.header.stamp    = self.get_clock().now().to_msg()
        sentinel.header.frame_id = ""   # convention: empty frame => cleared
        self._goal_pub.publish(sentinel)

        self._goal             = None
        self._was_in_goal_zone = False
        self._solver.reset_warm_start()

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
        if self._solver._last_state is None:
            return
        now  = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp    = now
        path.header.frame_id = self.topic_cfg.map_frame
        for k in range(self._solver._last_state.shape[0]):
            x_k = self._solver._last_state[k]
            pose = PoseStamped()
            pose.header.stamp    = now
            pose.header.frame_id = self.topic_cfg.map_frame
            pose.pose.position.x = float(x_k[0])
            pose.pose.position.y = float(x_k[1])
            yaw = float(x_k[2])
            pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.pose.orientation.w = float(np.cos(yaw / 2.0))
            path.poses.append(pose)
        self._traj_pub.publish(path)

    def _publish_empty_trajectory(self):
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.topic_cfg.map_frame
        self._traj_pub.publish(msg)

    # ---------------- PMP introspection (for evaluation) ----------------

    def extract_costates(self) -> Optional[np.ndarray]:
        """Return the last costate trajectory (m, 3): lambda_x, lambda_y, lambda_th."""
        return self._solver._last_costate

    def extract_predicted_trajectory(self) -> Optional[np.ndarray]:
        """Return the last optimal state trajectory (m, 3): px, py, theta."""
        return self._solver._last_state


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
