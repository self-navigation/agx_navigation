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
  (Quadratic near the goal so the gradient fades to zero at the sink,
   linear during navigation so it doesn't swamp w_h on routed paths.
   Gradient = beta * min(T, T_horizon) * grad(T) / T_horizon, C^0 at
   the join.)

  Phi(x_T) = (1/2) * w_pp * ||p_T - p_pursuit||^2           # pursuit-point pull
           + (1/2) * w_th * (theta_T - theta_pursuit)^2     # terminal yaw target

with
  v_ref(p)        = v_max * tanh(||p - p_goal|| / L_brake)
  gate(x)         = ((1 + x) / 2) ** p_gate    in [0, 1]
  v_ref_eff(p,th) = v_ref(p) * gate(F_unit . h(theta))

The non-negative gate replaces an older `v_ref * (F . h)` heading-aware
target: when |F . h| < 1 the forward target fades, so the cost never
asks for reverse motion under any heading. The brake term separately
penalises v != 0 in proportion to the SQUARE of the misalignment, giving:
  - gentle near alignment (small misalignment -> tiny brake), so the
    chassis follows F-curvature smoothly through corners;
  - strong at large misalignment (4 w_brake at anti-aligned), enough to
    overpower the position-pursuit costates that would otherwise pull the
    BVP toward reverse-while-turning.
Together: misaligned -> v ~ 0 + omega != 0 (pure rotation), aligned ->
drive forward at v_ref. This mechanism handles the goal-yaw fix, initial
heading mismatches, and sharp F-curvature mid-trajectory through one
unified cost shape -- no dedicated TURN_IN_PLACE supervisor needed.

The terminal cost is field-following: p_pursuit is the streamline
endpoint traced from x_now for arc length v_max * T_horizon *
pursuit_lookahead_mult, and theta_pursuit is the F-tangent at p_pursuit,
blended toward goal_yaw inside the goal approach zone. When the streamline
reaches the goal sink, p_pursuit collapses to p_goal and theta_pursuit
falls back to theta_goal.

Hamiltonian (minimum-principle convention):
  H = L + lambda_x * v cos(theta) + lambda_y * v sin(theta) + lambda_th * omega

Closed-form optimal control (tanh-saturated to bounds):
  denom_v     = 2 gamma_v + w_v + w_brake * (1 - F . h)^2
  v_unsat     = (w_v * v_ref_eff - lambda_x cos(theta) - lambda_y sin(theta)) / denom_v
  omega_unsat = -lambda_th / (2 gamma_w)

Costate ODEs (lambda_dot = -dH/dx), frozen-field approximation in the
position costates (dF_unit/dp and dv_ref/dp dropped):
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

from dataclasses import dataclass, fields, replace
from math import hypot, atan2, pi, sin, cos, tanh
from typing import Any, List, Optional, Tuple
import threading

import numpy as np
from scipy.integrate import solve_bvp
from scipy.ndimage import gaussian_filter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener, TransformException

from agx_planning_msgs.msg import PlannerTrajectoryChunk


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

    # --- Operating mode ---
    # "online":  per-tick BVP solve + Twist publication at control_rate Hz.
    # "offline": worker-thread rollout-by-concatenation; chunks streamed
    #            on /pmp_planner/trajectory_chunks for an interpreter to
    #            execute. See module docstring for full semantics.
    mode: str = "online"

    # --- Horizon ---
    # N: number of mesh nodes for the BVP initial guess (the adaptive
    #    mesh refines beyond this, up to bvp_max_nodes). More nodes give a
    #    better initial guess for curved trajectories at the cost of a
    #    slower first solve per cycle. 21 nodes (20 intervals) is a good
    #    default; raise to 31-41 if the warm-start repeatedly fails on
    #    sharp fields.
    # T_horizon: prediction window [s]. The BVP optimises over this entire
    #    window; only the t=0 command is applied (receding horizon). Longer
    #    horizons see farther ahead but widen the TPBVP and slow solve time.
    #    Set to roughly the distance-to-goal / v_max at the desired
    #    lookahead; 2.5 s at v_max = 0.5 m/s gives a 1.25 m lookahead.
    # control_rate: timer frequency [Hz] for online mode. Warm-start solves
    #    are typically < 30 ms, so 10 Hz leaves margin. Raise if your
    #    environment changes rapidly; lower if CPU is constrained.
    N: int = 21
    T_horizon: float = 2.5
    control_rate: float = 10.0

    # --- Control bounds (also act as tanh saturation scale in _ode) ---
    # v_max: peak forward/reverse speed [m/s]. The tanh saturation in the
    #    ODE is C-inf (unlike a hard clip), so solve_bvp's Newton doesn't
    #    see slope kinks at the bounds. Set to the chassis's safe speed.
    # omega_max: peak angular rate [rad/s]. 1.5 rad/s ~ 86 deg/s. Raise if
    #    the chassis needs tighter turns; lower if it tends to oscillate.
    v_max: float = 0.5
    omega_max: float = 1.5

    # --- Running-cost weights ---
    # alpha_t: constant time-penalty per second. Acts as a mild urgency
    #    signal; usually dominated by beta on long paths. Safe to leave at 1.
    # beta: weight on the piecewise-C^1 position potential L_pos(T).
    #    The gradient is beta * min(T, T_horizon) * grad(T) / T_horizon,
    #    so it fades to zero at the goal sink (T->0) for clean braking,
    #    and is capped to beta * grad(T) during navigation so it doesn't
    #    swamp the heading terms. Raise to pull harder toward the goal on
    #    long paths; lower if the chassis shortcuts around obstacles.
    # w_h: heading-alignment weight. Controls (1 - F_unit . h) in the
    #    running cost and the terminal-yaw spring. Tuning is non-monotonic:
    #    both w_h <= 3 (position cost dominates) and w_h >= 12 (rigid
    #    F-tracking) clear corners well; intermediate values (3-12) can
    #    find shortcuts. Default 5 is a good starting point.
    # w_v: speed-reference tracking weight (v - v_ref_eff)^2.
    #    Set to 0 to disable; useful to isolate brake/alignment behavior.
    #    Raise if the chassis doesn't reach v_max on straight segments.
    # w_brake: heading-coupled brake on v^2:
    #    (1/2) * w_brake * (1 - F . h)^2 * v^2.
    #    Quadratic-in-misalignment: gentle near alignment (cornering is
    #    smooth), strong when anti-aligned (overwrites position-pursuit
    #    costates that would otherwise drive reverse). Magnitude needs to
    #    exceed 2 * w_pp * (pursuit distance) + beta * T_horizon to
    #    reliably prevent reverse; empirically |lambda . h| is in the
    #    10-30 range, so 200 is the safe default. Set to 0 to debug the
    #    gate in isolation (re-introduces reverse-while-turning).
    # L_brake: speed-reference length scale [m]. v_ref = v_max *
    #    tanh(d_to_goal / L_brake). Set near the chassis stopping distance.
    #    Smaller values brake earlier; larger values maintain speed closer
    #    to the goal.
    # align_gate_power: sharpness of the heading-alignment gate multiplying
    #    v_ref: gate = ((1 + F.h) / 2)^p. p=4 (default) gives gate(perp)
    #    = 0.06 -- effective braking when perpendicular. p=2 gives
    #    gate(perp) = 0.25 (racing-line cornering). p=8 is near binary.
    #    p=0 disables the gate entirely (reintroduces the at-goal yaw
    #    shuffle -- don't use without a downstream supervisor).
    # gamma_v: quadratic regularizer on v. Prevents v_unsat from blowing
    #    up when the denominator is small. Raise if v oscillates.
    # gamma_w: quadratic regularizer on omega. Stiffens ALL angular
    #    dynamics globally; too large makes tight maneuvers infeasible.
    #    Lower if the chassis won't turn fast enough at the goal.
    alpha_t: float = 1.0
    beta:    float = 5.0
    w_h:     float = 5.0
    w_v:     float = 0.5
    w_brake: float = 200.0
    L_brake: float = 0.5
    align_gate_power: float = 4.0
    gamma_v: float = 0.5
    gamma_w: float = 0.2

    # --- Field smoothing ---
    # field_eps: soft re-normalisation denominator for F_unit:
    #    |F_unit| = |F| / sqrt(|F|^2 + eps^2). Makes |F_unit| -> 0 where
    #    the underlying field collapses (goal sink, saddles), fading the
    #    alignment cost instead of letting it fight the terminal target.
    #    Set comparable to the upstream gradient noise floor (~1e-2).
    # align_smooth_sigma: in-planner Gaussian smoothing on T (in grid
    #    cells) before deriving F for the alignment direction. Smooths
    #    corner curvature; preserves the goal singularity (T, not F, is
    #    smoothed). Leave at 0 unless the upstream T field has unusually
    #    sharp features; sigma 2-4 cells is the useful range.
    field_eps: float = 1e-2
    align_smooth_sigma: float = 0.0

    # --- Terminal-cost weights ---
    # w_pp: pursuit-point position pull: (1/2)*w_pp*||p_T - p_pursuit||^2.
    #    p_pursuit is the streamline endpoint at arc-length lookahead.
    #    Sitting on the flow line removes the corner-cut bias the old
    #    Lyapunov-T terminal had. Raise toward ~10 if endpoint feels
    #    under-pulled; lower if Newton struggles to converge.
    # w_th: terminal heading basin: (1/2)*w_th*(theta_T - theta_pursuit)^2.
    #    Also drives the running heading spring when w_F ~ 0 (at goal).
    #    See _ode for why a quadratic form is preferred over (1-cos).
    # pursuit_lookahead_mult: target arc length as a multiple of
    #    v_max * T_horizon. 1.0 places the terminal target where the
    #    chassis would arrive if it tracked F at full speed. < 1 leaves
    #    slack (eases Newton on hard fields); > 1 reaches past the natural
    #    horizon (tighter tracking, harder convergence).
    w_pp: float = 5.0
    w_th: float = 2.0
    pursuit_lookahead_mult: float = 1.0

    # --- Cross-track residual (opt-in, off by default) ---
    # Adds (1/2)*w_xt*r_xt(p)^2 to the running cost, where r_xt is the
    # signed perpendicular drift from the F-streamline at a soft Gaussian-
    # weighted projection. Penalises lateral drift directly (alignment
    # only constrains heading). Off by default: empirically the BVP either
    # ignores a small w_xt or stalls at a large one, because the
    # frozen-reference gradient doesn't accurately reflect the curvature of
    # the streamline in the BVP mesh. Useful only when the upstream F field
    # is unusually straight and the chassis still drifts laterally.
    # xt_horizon_m: streamline arc length to trace [m]; should exceed
    #    v_max * T_horizon.
    # xt_sigma_mult: Gaussian projection bandwidth = mult * grid_resolution.
    #    ~3 blends 5-7 adjacent samples; too small causes Voronoi-boundary
    #    oscillation during mesh refinement.
    w_xt:         float = 0.0
    xt_horizon_m: float = 2.5
    xt_sigma_mult: float = 3.0

    # --- Goal tolerances ---
    # Both must be satisfied simultaneously to trigger REACHED (zero twist).
    # The at-goal yaw correction is handled by the BVP itself (gated v_ref
    # yields v~0 + omega!=0 at the goal) -- no separate supervisor needed.
    goal_tolerance_xy: float = 0.05     # [m]
    goal_tolerance_th: float = 0.20     # [rad]

    # --- BVP solver knobs ---
    # bvp_tol: solve_bvp residual tolerance. 1e-3 is the practical sweet
    #    spot: tighter (1e-4) roughly doubles solve time with little
    #    command improvement; looser (1e-2) saves ~20% but may leave visible
    #    residual on sharp fields.
    # bvp_max_nodes: adaptive mesh node cap. 2000 is sufficient for most
    #    smooth fields; raise to 3000-5000 if the field has obstacle
    #    boundaries that create sharp T gradients and solve_bvp consistently
    #    hits the cap (logged as a warning via bvp_verbose >= 1).
    # bvp_verbose: 0 = silent, 1 = per-solve summary, 2 = per-iteration.
    #    Use 1 to diagnose convergence issues without flooding the log.
    # reuse_previous_solution: warm-start each solve from the previous BVP
    #    solution. This is the single biggest convergence aid; disable only
    #    to measure cold-start performance or diagnose warm-start pathology.
    #    Automatically dropped on new goal, new field, or goal-zone boundary
    #    crossing.
    bvp_tol: float = 1e-3
    bvp_max_nodes: int = 2000
    bvp_verbose: int = 0
    reuse_previous_solution: bool = True

    # --- Offline-mode parameters (ignored when mode == "online") ---
    # dt_segment: committed arc length per BVP solve [s]. Each solve
    #    optimises over the full T_horizon, but only the first dt_segment
    #    seconds are published and the sim state is advanced by that amount.
    #    Smaller values re-solve more frequently (better quality, more CPU);
    #    larger values exploit more of each BVP's tail (lower CPU, slightly
    #    degraded quality at segment ends where the tail is dominated by the
    #    terminal-cost regulariser). Capped to T_horizon at runtime.
    #    Default = T_horizon / 2.
    dt_segment: float = 1.25            # [s]

    # field_diff_threshold: max |T_new - T_old| along the planned path
    #    that triggers a replan [s]. Out-of-bounds cells in either grid are
    #    treated as +inf diff (newly discovered terrain always replans).
    #    Tune conservatively: too low causes spurious replans on minor field
    #    updates; too high lets the chassis execute a now-invalid plan
    #    through a changed obstacle.
    field_diff_threshold: float = 0.5   # [s]

    # max_rollout_sim_time: safety cap on total rollout time [s]. If the
    #    simulated chassis hasn't reached goal tolerance within this many
    #    seconds of trajectory time, the rollout is aborted and a terminal
    #    chunk is published. Guards against degenerate fields where the
    #    chassis orbits the goal indefinitely.
    max_rollout_sim_time: float = 60.0  # [s]

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

    - dT/dx, dT/dy: recomputed from np.gradient so the position-costate
      ODEs (beta * grad T) stay consistent with the T grid being penalised,
      regardless of upstream smoothing or sign convention.
    - F_unit: derived from the upstream (Fx, Fy) channels and re-normalised
      with eps regularisation so |F_unit| -> 0 where the underlying field
      magnitude collapses (goal sink, saddles, flat regions), fading the
      alignment cost instead of fighting the terminal target.

    Sign convention: (Fx, Fy) is the "follow this direction" field. If the
    upstream publishes raw +grad T (away from goal), flip the sign upstream
    or set align_smooth_sigma > 0 to derive F from -grad(smooth(T)) here.

    Concurrency: instances are immutable after update(). The node uses
    atomic-reference-swap of self._field on field arrival (atomic under
    CPython's GIL on bare attribute assignment), so threaded readers
    never observe a torn update. Replan-trigger code keeps a reference
    to the previous instance for path-masked diffing against the new one.
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
        # Monotonic counter; bumped on every update so the solver can
        # detect a replaced field and drop a now-stale warm start. With
        # atomic-swap of grid instances (offline mode), this counter
        # resets per-instance, so the node also calls reset_warm_start()
        # on every swap regardless of version.
        self._version = 0

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def version(self) -> int:
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

        # Alignment direction field source priority:
        #   1. align_smooth_sigma > 0: derive from grad(gaussian_filter(T)).
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

    def in_bounds(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Boolean mask: which (px, py) lie within the grid extent.

        Used by the offline-mode path-masked field-diff replan trigger:
        cells out-of-bounds in either old-or-new grid are tagged as
        infinite diff so newly-discovered terrain on the planned path
        always triggers a replan.
        """
        if not self._ready:
            return np.zeros_like(px, dtype=bool)
        u = (px - self._origin_x) / self._res
        w = (py - self._origin_y) / self._res
        rows, cols = self._tt.shape
        return (u >= 0) & (u <= cols - 1) & (w >= 0) & (w <= rows - 1)

    def trace_streamline(
        self,
        x0: float, y0: float,
        length_m: float,
        ds: Optional[float] = None,
        goal_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trace dp/ds = F_unit(p) starting at (x0, y0) for up to length_m.

        Stops on (a) |F| < 1e-3 (goal sink or saddle), (b) length_m of arc
        length consumed, (c) leaving the grid bounds, or (d) within
        sqrt(0.01) m of goal_xy. Early goal-stop means p_pursuit naturally
        collapses to p_goal when the streamline reaches it within the
        lookahead distance.

        Returns (ref_pts (N, 2), n_perp (N, 2)) where n_perp is the unit
        normal rotated 90 deg CCW from F at each sample (used for the
        cross-track residual). Both arrays are empty if the trace cannot
        start (e.g. already at goal, F collapses immediately).
        """
        if not self._ready:
            return np.zeros((0, 2)), np.zeros((0, 2))
        if ds is None:
            ds = self._res

        # If we're already inside the goal stop ball, the loop below would
        # record exactly one ref point (chassis) before tripping the stop
        # check and breaking. The caller (solve) then sets p_pursuit =
        # chassis position, which makes the BVP terminal cost pull toward
        # "stay where you are" -- chassis never moves. Return empty
        # instead so the caller's fallback sets p_pursuit = goal directly.
        # (This fixes a long-standing online-mode bug where the chassis
        # would idle in the [goal_tolerance_xy, 0.1m] ring around the goal
        # until TF noise pushed it inside; offline mode exposed it because
        # there's no observation noise to bail us out.)
        if goal_xy is not None:
            if (x0 - goal_xy[0]) ** 2 + (y0 - goal_xy[1]) ** 2 < 1e-2:
                return np.zeros((0, 2)), np.zeros((0, 2))

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
            tx, ty = fux / mag, fuy / mag
            ref_x[n] = px
            ref_y[n] = py
            nx[n] = -ty
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

    Same solver in both modes. Online mode calls solve() per tick;
    offline mode calls solve() once per rollout segment AND uses the
    BVP solution object (self._prev_sol) to densely sample the
    committed segment. The closed-form control law is applied per
    sample via _control_law_pointwise().
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
        #   theta_pursuit -- F-tangent at p_pursuit, blended toward
        #                    theta_goal inside the approach zone.
        self._p_pursuit: np.ndarray = np.zeros(2, dtype=np.float64)
        self._theta_pursuit: float = 0.0
        # Field-alignment-cost fade: w_F in [0, 1], set per-solve from the
        # cubic smoothstep of chassis distance to goal. 1.0 = full field
        # alignment; 0.0 = pure goal-yaw spring. Per-solve scalar, NOT a
        # per-mesh-point fade -- per-mesh-point fading suppresses the brake
        # at the trajectory's late mesh points and causes approach overshoot.
        self._align_fade: float = 1.0
        # Cross-track reference (only populated when cfg.w_xt > 0). Empty
        # arrays cause _ode's cross-track block to short-circuit cheaply.
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
            # Mathematically fine for any p >= 1; p=0 special-cased to
            # avoid 0^(-1) and to make the cost reduction explicit.
            if p_gate > 0.0:
                gate_prime = (0.5 * p_gate) * (half_one_plus ** (p_gate - 1.0))
            else:
                gate_prime = np.zeros_like(half_one_plus)
            v_ref_eff = v_ref * gate

        # Closed-form unconstrained optimal controls from dH/du = 0.
        # Brake denom: w_brake * (1 - F.h)^2 ramps from 0 at alignment to
        # 4*w_brake at anti-aligned. Quadratic-in-misalignment is key:
        #   * gentle near alignment (cornering: small angle -> tiny denom
        #     contribution, chassis follows F-curvature smoothly);
        #   * strong when anti-aligned (overwrites position-pursuit
        #     costates that would otherwise pull v into reverse).
        # The theta-derivative of the brake term picks up a (1 - F.h)
        # factor that vanishes exactly at alignment, so cross_F_h jitter
        # from bilinear interpolation doesn't pump omega during straight
        # forward drive.

        one_minus_dot = 1.0 - F_dot_h
        denom_v = 2.0 * cfg.gamma_v + cfg.w_v + cfg.w_brake * one_minus_dot * one_minus_dot
        v_unsat = (cfg.w_v * v_ref_eff
                   - lx * cos_t - ly * sin_t) / denom_v
        w_unsat = -lt / (2.0 * cfg.gamma_w)

        v = cfg.v_max     * np.tanh(v_unsat / cfg.v_max)
        w = cfg.omega_max * np.tanh(w_unsat / cfg.omega_max)

        dpx = v * cos_t
        dpy = v * sin_t
        dth = w
        # Position costates from the piecewise-C^1 potential:
        #   gradient = beta * min(T, T_horizon) * grad(T) / T_horizon.
        # The T_horizon cap is critical: without it the gradient grows
        # without bound (beta * T / T_horizon) on long paths, swamping
        # w_h * cross_F_h and biasing the BVP toward shortcuts regardless
        # of the field streamline tangent. The cap also makes the gradient
        # fade to zero as T->0 (goal sink), so braking is governed by
        # v_ref rather than a residual position pull.
        # (Hard np.clip is the exact PMP saturation for box-constrained u,
        # but its slope kink breaks solve_bvp's Newton; tanh is C-inf and
        # asymptotes to +-u_max.)
        T_clip = np.minimum(T_now, cfg.T_horizon)
        dlx = -cfg.beta * T_clip * dT_dx / cfg.T_horizon
        dly = -cfg.beta * T_clip * dT_dy / cfg.T_horizon

        # Cross-track residual (opt-in): adds -w_xt * r_xt * n_perp to
        # (dlx, dly) where r_xt is the signed perpendicular drift from the
        # streamline at a Gaussian-weighted projection. Frozen-reference
        # approximation drops dr_xt/dp_ref. Soft (Gaussian) projection is
        # what makes Newton converge: discrete nearest-sample lookups are
        # piecewise-constant in n_perp and mesh-refinement oscillates at
        # Voronoi boundaries.
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
            n_mag = np.sqrt((n_lin * n_lin).sum(axis=1))
            n_perp = n_lin / np.maximum(n_mag[:, None], 1e-9)
            delta = mesh - p_ref
            r_xt = (delta * n_perp).sum(axis=1)
            dlx = dlx - cfg.w_xt * r_xt * n_perp[:, 0]
            dly = dly - cfg.w_xt * r_xt * n_perp[:, 1]

        # Heading costate: two complementary running heading drives, cross-
        # faded by self._align_fade (= w_F, a per-solve scalar set in
        # solve() from the chassis-to-goal smoothstep):
        #
        # (1) Field-aligned terms (active when w_F ~ 1, navigation regime),
        #     all proportional to cross_F_h = F_x sin(theta) - F_y cos(theta):
        #       -dL_align/dtheta  = -w_h * cross_F_h
        #       -dL_speed/dtheta  = -w_v * v_ref * (v - v_ref_eff) * gate'(F.h) * cross_F_h
        #       -dL_brake/dtheta  = -w_brake * (1 - F.h) * v^2 * cross_F_h
        #     The (1 - F.h) factor in the brake term vanishes at alignment,
        #     so a straight drive on a slightly noisy F field stays clean.
        #
        # (2) Terminal-yaw running spring (active when w_F ~ 0, goal regime):
        #       -dL_yaw/dtheta = -w_h * (theta - theta_pursuit)
        #     Without this, at-goal omega is governed by the terminal BC
        #     alone (constant-omega LQR), which is too slow for large yaw
        #     errors. With it, unconstrained omega ~ sqrt(w_h / (2*gamma_w))
        #     * |delta|, saturating omega_max at moderate yaw error.
        #
        # Plus the kinematic coupling lx*v*sin(theta) - ly*v*cos(theta)
        # (from lambda . df/dtheta), which is never faded.
        #
        # w_F is a per-solve scalar, NOT per-mesh-point: per-mesh-point
        # fading would suppress the brake at the trajectory's late nodes
        # and cause approach overshoot when the chassis is still far away.
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
            lambda_th(T) = w_th * (theta_T - theta_pursuit)
        (Quadratic Phi gives linear transversalities, which Newton handles
        cleanly. A (1-cos) Phi would give sin() here and introduce
        multi-basin pathology.)
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
        speed with the same alignment gate as the running cost. This
        naturally pure-rotates when misaligned and drives forward only
        when aligned. When already inside the position-tolerance ball,
        the desired heading is theta_goal so the rollout spends the
        horizon rotating toward goal yaw -- matching the BVP's at-goal
        regime. The costate is linearly ramped from 0 to the terminal
        transversality evaluated at the rollout endpoint; since the rollout
        tracks F, the endpoint is near p_pursuit and the ramped lambda is
        small, which is the warm start solve_bvp's Newton prefers.
        """
        cfg = self.cfg
        m = t_mesh.size
        dt = t_mesh[1] - t_mesh[0] if m > 1 else cfg.T_horizon

        state = np.zeros((3, m))
        state[:, 0] = x0
        px, py, th = float(x0[0]), float(x0[1]), float(x0[2])

        for k in range(1, m):
            d_goal = hypot(px - goal[0], py - goal[1])
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

            half_one_plus = max(0.0, 0.5 * (1.0 + cos(e)))
            gate = half_one_plus ** cfg.align_gate_power
            v_target = cfg.v_max * tanh(d_goal / cfg.L_brake)
            v = v_target * gate

            px += v * cos(th) * dt
            py += v * sin(th) * dt
            th += omega * dt
            state[:, k] = (px, py, th)

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
        returns None if no warm start is available so the caller can decide
        (online mode publishes zero twist; offline mode aborts the rollout).
        The prev_sol callable is freed on failure to prevent a second
        fallback attempt with a further-extrapolated bad solution.
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
        v_cmd = float(np.clip(cfg.v_max     * np.tanh(v_unsat / cfg.v_max),
                              -cfg.v_max,     +cfg.v_max))
        w_cmd = float(np.clip(cfg.omega_max * np.tanh(w_unsat / cfg.omega_max),
                              -cfg.omega_max, +cfg.omega_max))
        return v_cmd, w_cmd

    def solve(
        self, x0: np.ndarray, goal: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Solve the TPBVP. Returns (v_cmd, omega_cmd) or None on failure.

        Side effects (used by both online and offline paths):
          - self._prev_sol     : OdeSolution callable, t in [0, T_horizon]
          - self._last_state   : (N+1, 3) dense state samples
          - self._last_costate : (N+1, 3) dense costate samples
          - self._last_error   : str or None
        """
        cfg = self.cfg
        self._x0 = x0
        self._goal = goal

        if self.field.version != self._last_field_version:
            self.reset_warm_start()
            self._last_field_version = self.field.version

        # Streamline trace from x_now -- one call serves two consumers:
        # (1) pursuit point + heading, sampled at pursuit_dist arc length;
        # (2) cross-track reference, if w_xt > 0, using the full trace.
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

        if ref_pts.shape[0] >= 1:
            n_target = int(np.ceil(pursuit_dist / ds_used))
            n_use    = max(1, min(n_target, ref_pts.shape[0]))
            self._p_pursuit = ref_pts[n_use - 1].astype(np.float64)
        else:
            # Trace cannot start (goal sink, off-grid, or already at goal).
            # Fall back to the goal itself: loses corner-cut protection but
            # is the only sensible default in a degenerate field.
            self._p_pursuit = np.array([goal[0], goal[1]], dtype=np.float64)

        _, _, _, fux_pp, fuy_pp = self.field.query_scalar(
            float(self._p_pursuit[0]), float(self._p_pursuit[1]),
        )
        chassis_to_goal_sq = ((float(x0[0]) - goal[0]) ** 2
                              + (float(x0[1]) - goal[1]) ** 2)
        chassis_to_goal = float(np.sqrt(chassis_to_goal_sq))
        gt = max(cfg.goal_tolerance_xy, 1e-3)
        s = float(np.clip((chassis_to_goal - gt) / (3.0 * gt), 0.0, 1.0))
        # w_F: field-vs-goal-yaw blend for theta_pursuit and the running-
        # cost fade. Cubic smoothstep (C^1 continuous) transitions from
        # w_F = 0 (pure goal_yaw) when chassis is inside goal_tolerance_xy,
        # to w_F = 1 (pure F-tangent) when chassis is > 4 * goal_tolerance_xy
        # away. Using goal_tolerance_xy as the natural scale means the
        # regime boundary is a property of the user's config, not an
        # arbitrary internal threshold.
        w_F = s * s * (3.0 - 2.0 * s)   # cubic smoothstep, C^1 continuous

        F_mag_pp = float(np.hypot(fux_pp, fuy_pp))
        inv_F = 1.0 / max(F_mag_pp, 1e-9)
        F_hat_x = fux_pp * inv_F
        F_hat_y = fuy_pp * inv_F
        goal_hat_x = float(np.cos(goal[2]))
        goal_hat_y = float(np.sin(goal[2]))
        target_x = w_F * F_hat_x + (1.0 - w_F) * goal_hat_x
        target_y = w_F * F_hat_y + (1.0 - w_F) * goal_hat_y
        target_norm = np.hypot(target_x, target_y)
        if target_norm < 1e-3:
            # Antipodal degenerate case: F_hat and goal_hat point nearly
            # opposite ways and w_F ~ 0.5 collapses target to ~zero.
            # Fall back to the dominant unit vector.
            if w_F >= 0.5:
                self._theta_pursuit = atan2(F_hat_y, F_hat_x)
            else:
                self._theta_pursuit = float(goal[2])
        else:
            self._theta_pursuit = float(atan2(target_y, target_x))

        # The same w_F modulates L_align/L_brake/L_speed-coupling in the
        # heading costate dynamics (consumed by _ode via self._align_fade).
        self._align_fade = w_F

        # Unwrap theta_pursuit relative to the current chassis heading so
        # the heading basin has a single minimum at the shortest-turn target.
        # Without this, a (1/2)*w_th*(theta_T - thp)^2 form has stable
        # fixed points at thp + 2*k*pi; Newton can latch onto the wrong one,
        # especially when the warm start drifts between cycles.
        theta_now = float(x0[2])
        delta = ((self._theta_pursuit - theta_now + pi) % (2.0 * pi)) - pi
        self._theta_pursuit = theta_now + delta

        if xt_active and ref_pts.shape[0] >= 2:
            self._xt_ref    = ref_pts
            self._xt_n_perp = n_perp
            self._xt_sigma  = max(cfg.xt_sigma_mult * ds_used, 1e-3)
        else:
            self._xt_ref    = np.zeros((0, 2))
            self._xt_n_perp = np.zeros((0, 2))

        t_mesh = np.linspace(0.0, cfg.T_horizon, cfg.N + 1)
        if cfg.reuse_previous_solution and self._prev_sol is not None:
            try:
                y_init = self._prev_sol(t_mesh)
                # If the chassis crossed +-pi between cycles, the warm-start's
                # theta(0) differs from x0[2] by ~2*pi*k. Re-anchoring
                # y_init[2,0] = x0[2] without fixing the rest leaves a 2*pi
                # discontinuity that solve_bvp cannot resolve (mesh refines
                # forever). Shift the entire theta trajectory by 2*pi*k first.
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

        # Optimal control at t = 0 via the same closed-form law _ode uses.
        v_cmd, w_cmd = self._control_law_pointwise(sol.y[:, 0], goal)

        # Densely resample for visualization and downstream introspection.
        t_dense = np.linspace(0.0, cfg.T_horizon, cfg.N + 1)
        y_dense = self._prev_sol(t_dense)
        self._last_state = y_dense[0:3].T
        self._last_costate = y_dense[3:6].T
        self._last_error = None

        return v_cmd, w_cmd

    def reset_warm_start(self):
        self._prev_sol = None

    # --- Closed-form control law (shared by online + offline paths) -------

    def _control_law_pointwise(
        self, y6: np.ndarray, goal: np.ndarray,
    ) -> Tuple[float, float]:
        """Apply the closed-form optimal control law at one (state, costate).

        Same algebra as _ode, but scalar -- used to extract v(t), omega(t)
        from the BVP solution at arbitrary time samples in offline mode,
        and for the t=0 command in online mode.

        Mirrors _ode exactly; any change to the control law there must be
        reflected here. The two branches (w_v > 0 vs w_v == 0) match the
        original solve()'s extraction logic.
        """
        cfg = self.cfg
        px, py, th = float(y6[0]), float(y6[1]), float(y6[2])
        lx, ly, lt = float(y6[3]), float(y6[4]), float(y6[5])
        cos_t = cos(th)
        sin_t = sin(th)
        _, _, _, Fux, Fuy = self.field.query_scalar(px, py)
        F_dot_h = Fux * cos_t + Fuy * sin_t
        one_minus = 1.0 - F_dot_h

        if cfg.w_v > 0.0:
            d_to_goal = hypot(px - goal[0], py - goal[1])
            v_ref = cfg.v_max * tanh(d_to_goal / cfg.L_brake)
            half_one_plus = max(0.0, min(1.0, 0.5 * (1.0 + F_dot_h)))
            gate = half_one_plus ** cfg.align_gate_power
            v_ref_eff = v_ref * gate
            denom_v = (2.0 * cfg.gamma_v + cfg.w_v
                       + cfg.w_brake * one_minus * one_minus)
            v_unsat = (cfg.w_v * v_ref_eff - lx * cos_t - ly * sin_t) / denom_v
        else:
            denom_v = 2.0 * cfg.gamma_v + cfg.w_brake * one_minus * one_minus
            v_unsat = -(lx * cos_t + ly * sin_t) / denom_v
        w_unsat = -lt / (2.0 * cfg.gamma_w)

        v_cmd = float(np.clip(cfg.v_max     * tanh(v_unsat / cfg.v_max),
                              -cfg.v_max,     +cfg.v_max))
        w_cmd = float(np.clip(cfg.omega_max * tanh(w_unsat / cfg.omega_max),
                              -cfg.omega_max, +cfg.omega_max))
        return v_cmd, w_cmd

    # --- Offline-mode segment extraction ----------------------------------

    def sample_committed_segment(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        dt_sample: float,
        n_samples: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Solve the BVP from x0 toward goal, then densely sample the
        first n_samples * dt_sample seconds of the optimal trajectory.

        This is the rollout-by-concatenation primitive: each call solves
        a fresh local BVP (reusing the warm start across calls if the
        config allows) and returns the controls + predicted state at
        dt_sample-spaced ticks. The caller commits this segment, advances
        its sim state to the segment endpoint, and calls again.

        Returns (twists, poses, x_next) where:
          twists  : (n_samples, 2) [v, omega]
          poses   : (n_samples, 3) [px, py, theta] -- the predicted state
                    AT each tick, i.e. poses[i] is where the chassis is
                    just before applying twists[i] (poses[0] == x0).
          x_next  : (3,) state AT t = n_samples * dt_sample, the start
                    of the next segment.

        Returns None if the BVP fails. Caller decides how to recover
        (typically: zero twist + abort the rollout).

        Total committed time is n_samples * dt_sample, capped to
        T_horizon by the caller. Warm start is preserved between calls.
        """
        if self.solve(x0, goal) is None:
            return None
        if self._prev_sol is None:
            return None

        cfg = self.cfg
        seg_T = n_samples * dt_sample
        if seg_T > cfg.T_horizon + 1e-9:
            # Caller overcommitted -- shorten to fit the BVP horizon.
            n_samples = max(1, int(cfg.T_horizon / dt_sample))
            seg_T = n_samples * dt_sample

        # Sample at tick i = 0, 1, ..., n_samples - 1 (the time at which
        # twists[i] starts being applied). x_next is at i = n_samples.
        t_ticks = np.arange(n_samples + 1, dtype=np.float64) * dt_sample
        # Numerical guard: dense_output may complain if t > T_horizon
        # by float epsilon; clamp.
        t_ticks = np.minimum(t_ticks, cfg.T_horizon)
        try:
            y_ticks = self._prev_sol(t_ticks)   # shape (6, n_samples + 1)
        except Exception as e:
            self._last_error = f"prev_sol resample failed: {e}"
            return None

        twists = np.zeros((n_samples, 2), dtype=np.float64)
        poses  = y_ticks[0:3, :n_samples].T.copy()    # (n_samples, 3)
        for i in range(n_samples):
            v_i, w_i = self._control_law_pointwise(y_ticks[:, i], goal)
            twists[i, 0] = v_i
            twists[i, 1] = w_i

        x_next = y_ticks[0:3, n_samples].copy()
        return twists, poses, x_next


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------


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

        self._xi: np.ndarray = np.zeros(3)              # (px, py, theta)
        self._goal: Optional[np.ndarray] = None         # (gx, gy, gtheta)
        self._field = VectorFieldGrid()
        self._waiting_for_field = False

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
        self.create_subscription(Odometry,          "/odom",                       self._on_odom,  qos)
        self.create_subscription(PoseStamped,       "/goal_pose",                  self._on_goal,  qos)
        self.create_subscription(Float32MultiArray, "/vector_field/planner_data",  self._on_field, qos)

        # /cmd_vel publisher (used in online mode; dormant in offline since
        # the interpreter is the one talking to the chassis there).
        if self.topic_cfg.enable_stamped_cmd_vel:
            self._cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._traj_pub  = self.create_publisher(Path, "/pmp_planner/trajectory", 10)
        self._goal_pub  = self.create_publisher(PoseStamped, "/goal_pose", qos)
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
                target=self._offline_worker_loop, name="pmp_offline_worker",
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
        super().destroy_node()

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
            # Atomic single-attribute rebind. The offline worker reads
            # self._xi with a single load and either gets the old or new
            # array, never a torn write.
            self._xi = np.array([tx, ty, yaw])
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
        q   = msg.pose.orientation
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
        self.get_logger().info(
            f"Goal: ({pos.x:.2f}, {pos.y:.2f}), yaw={yaw:.2f}"
        )

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
            old_field = self._field   # GIL-atomic load
            with self._state_lock:
                traj_xy  = self._latest_trajectory_xy
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
            Fx = body[n:2 * n].reshape(h, w)
            Fy = body[2 * n:3 * n].reshape(h, w)
        else:
            Fx = None
            Fy = None
        # The grad_mag channel (if present) is ignored: we re-normalize
        # (Fx, Fy) to a unit field internally with our own eps regularizer.

        new_field = VectorFieldGrid()
        new_field.update(T, Fx, Fy, ox, oy, res,
                         field_eps=self.cfg.field_eps,
                         align_smooth_sigma=self.cfg.align_smooth_sigma)
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
            self._kick_event.wait(timeout=0.5)
            self._kick_event.clear()
            if self._stop_event.is_set():
                break

            with self._state_lock:
                goal = None if self._goal is None else self._goal.copy()
            if goal is None:
                continue

            field_ref = self._field        # GIL-atomic
            if not field_ref.ready:
                # Goal arrived before any field. The next field arrival
                # will kick us again.
                continue

            x0 = self._xi.copy()           # GIL-atomic load + copy
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
            goal = self._goal   # snapshot; None means nothing to do
        if goal is None:
            return

        xi = self._xi           # GIL-atomic single-attribute load
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
        all_poses: List[np.ndarray] = []

        # Stagnation tracking: if x_next stops making progress, the
        # rollout has hit a quasi-fixed-point (e.g. BVP is happily
        # emitting "stay where you are" because the chassis is in the
        # narrow ring just outside goal_tolerance_xy). The trace_streamline
        # pre-check prevents the most common cause of this, but a
        # tightly-tuned cost can still produce sub-tolerance drift; the
        # stagnation backstop ensures finite termination either way.
        progress_eps = max(0.5 * cfg.goal_tolerance_xy, 5e-3)   # [m]
        near_goal_thresh = 4.0 * cfg.goal_tolerance_xy          # [m]
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
            if d_xy_state < cfg.goal_tolerance_xy and d_th_state < cfg.goal_tolerance_th:
                self._publish_chunk(
                    traj_id, chunk_idx,
                    np.zeros((0, 2)), np.zeros((0, 3)),
                    dt_sample, is_final=True,
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
                state, goal, dt_sample, n_samples,
            )
            if result is None:
                self.get_logger().warn(
                    f"Offline BVP solve failed at sim_t={sim_t:.2f}s "
                    f"(traj_id={traj_id}, chunk={chunk_idx}): "
                    f"{self._solver._last_error}",
                )
                # Emit terminal marker so the interpreter doesn't wait.
                self._publish_chunk(
                    traj_id, chunk_idx,
                    np.zeros((0, 2)), np.zeros((0, 3)),
                    dt_sample, is_final=True,
                )
                return

            twists, poses, x_next = result

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
                    traj_id, chunk_idx, tw_trunc, ps_trunc,
                    dt_sample, is_final=True,
                )
                all_poses.append(ps_trunc)
                with self._state_lock:
                    if self._trajectory_id != traj_id:
                        return
                    self._latest_trajectory_xy = np.concatenate(
                        [p[:, :2] for p in all_poses], axis=0,
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
                    traj_id, chunk_idx,
                    np.zeros((0, 2)), np.zeros((0, 3)),
                    dt_sample, is_final=True,
                )
                return

            # Normal path: publish full chunk.
            self._publish_chunk(
                traj_id, chunk_idx, twists, poses,
                dt_sample, is_final=False,
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
                    [p[:, :2] for p in all_poses], axis=0,
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
                        traj_id, chunk_idx,
                        np.zeros((0, 2)), np.zeros((0, 3)),
                        dt_sample, is_final=True,
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
            traj_id, chunk_idx,
            np.zeros((0, 2)), np.zeros((0, 3)),
            dt_sample, is_final=True,
        )

    # ---------------- Publishing ----------------

    def _clear_goal(self):
        """Clear the active goal locally and signal it ROS-wide on /goal_pose."""
        sentinel = PoseStamped()
        sentinel.header.stamp    = self.get_clock().now().to_msg()
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
        """Online-mode horizon publication."""
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

    def _publish_cumulative_path(self, all_poses: List[np.ndarray]):
        """Offline-mode cumulative trajectory publication for visualization."""
        if not all_poses:
            return
        now  = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp    = now
        path.header.frame_id = self.topic_cfg.map_frame
        for block in all_poses:
            for k in range(block.shape[0]):
                pose = PoseStamped()
                pose.header.stamp    = now
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
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.topic_cfg.map_frame
        self._traj_pub.publish(msg)

    def _publish_chunk(
        self,
        traj_id: int, chunk_idx: int,
        twists: np.ndarray, poses: np.ndarray,
        dt: float, is_final: bool,
    ):
        """Publish one PlannerTrajectoryChunk. Called from the worker thread.

        twists shape (N, 2): [v, omega] per row.
        poses  shape (N, 3): [px, py, theta] per row, parallel to twists.
        Empty arrays are valid (is_final terminator chunks).
        """
        msg = PlannerTrajectoryChunk()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.topic_cfg.map_frame
        msg.trajectory_id   = int(traj_id)
        msg.chunk_index     = int(chunk_idx)
        msg.is_final        = bool(is_final)
        msg.dt              = float(dt)
        # tolist() because rosidl-generated message slots for float32[]
        # expect a Python list (or array.array), not an ndarray.
        if twists.shape[0] > 0:
            t32 = twists.astype(np.float32)
            p32 = poses.astype(np.float32)
            msg.linear_x    = t32[:, 0].tolist()
            msg.angular_z   = t32[:, 1].tolist()
            msg.pose_x      = p32[:, 0].tolist()
            msg.pose_y      = p32[:, 1].tolist()
            msg.pose_theta  = p32[:, 2].tolist()
        # else: leave the arrays as their default empty lists.
        self._chunk_pub.publish(msg)

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
