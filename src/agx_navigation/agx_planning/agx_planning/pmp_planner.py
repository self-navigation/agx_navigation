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

from dataclasses import dataclass, fields, replace
from math import hypot, atan2, pi, sin, cos, tanh
from typing import Any, List, Optional, Tuple
import csv as _csv
import time as _time
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
# Diagnostic logger
# ---------------------------------------------------------------------------


class TurnDiagnosticLogger:
    """Thread-safe CSV logger for comparing planned vs actual heading.

    Enabled by setting the ROS parameter ``diag_log_path`` to a non-empty
    file path (e.g. ``/tmp/pmp_diag.csv``).  Two row types are interleaved:

      source = "odom"    -- actual measured state, written on every /odom callback.
                           theta_deg, omega, v are EKF-fused chassis readings.
      source = "plan"    -- BVP-planned profile, written after each solve.
                           tick=0 is t=0 (anchored to actual state by BVP BC),
                           tick=k is the k-th committed/lookahead sample.
                           theta_deg is the BVP-planned heading (state).
                           omega, v are the PUBLISHED commands at this tick
                           (post chassis-model inversion), NOT the BVP state.
                           To recover the BVP state from omega_cmd, invert:
                             omega_state ~= gain_omega * omega_cmd - tau_omega * domega_cmd/dt
                           but in practice it is easier to log states
                           separately if needed.
                           lam_th, lam_om, alpha_cmd are only written on tick=0.

    Quick analysis (requires pandas + matplotlib)::

        import pandas as pd, matplotlib.pyplot as plt, numpy as np
        df = pd.read_csv('/tmp/pmp_diag.csv')
        odom = df[df.source == 'odom']
        p0   = df[(df.source == 'plan') & (df.tick == 0)]

        fig, axes = plt.subplots(3, 1, sharex=True)
        axes[0].plot(odom.wall_s, odom.theta_deg, label='actual')
        axes[0].plot(p0.wall_s,   p0.theta_deg,   '.', label='plan t=0')
        axes[0].set_ylabel('heading (deg)')
        axes[0].legend()
        # lambda_omega at t=0: must be negative for a CCW turn.
        # If it hovers near 0 the cold-start costate fix isn't firing.
        axes[1].plot(p0.wall_s, p0.lam_om)
        axes[1].axhline(0, color='k', ls='--')
        axes[1].set_ylabel('lambda_omega(0)')
        # alpha command at t=0: should be near +/-alpha_max while turning.
        axes[2].plot(p0.wall_s, p0.alpha_cmd)
        axes[2].set_ylabel('alpha*(0)  [rad/s^2]')
        axes[2].set_xlabel('wall time (s)')
        plt.tight_layout(); plt.show()

    To plot the full planned heading *profile* for each solve::

        plan_all = df[df.source == 'plan']
        # group by chunk; each group is one BVP solve's committed arc
        for (chunk,), grp in plan_all.groupby(['chunk']):
            plt.plot(grp.tick, grp.theta_deg, label=f'chunk {chunk}')
        plt.xlabel('tick within chunk'); plt.ylabel('planned heading (deg)')
        plt.legend(); plt.show()
    """

    _HEADER = [
        'wall_s', 'source', 'traj_id', 'chunk', 'tick',
        'x', 'y', 'theta_deg', 'omega', 'v',
        'lam_th', 'lam_om', 'alpha_cmd',
    ]

    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._f = open(path, 'w', newline='', buffering=1)  # line-buffered
        self._w = _csv.writer(self._f)
        self._w.writerow(self._HEADER)
        self._t0 = _time.monotonic()

    def _now(self) -> float:
        return _time.monotonic() - self._t0

    def log_odom(self, x: float, y: float, theta: float, v: float, omega: float) -> None:
        row = [f'{self._now():.4f}', 'odom', '', '', '',
               f'{x:.3f}', f'{y:.3f}', f'{float(np.degrees(theta)):.3f}',
               f'{float(omega):.5f}', f'{float(v):.5f}',
               '', '', '']
        with self._lock:
            self._w.writerow(row)

    def log_plan(
        self,
        traj_id: int, chunk: int,
        thetas_deg: np.ndarray,
        omegas:     np.ndarray,
        vs:         np.ndarray,
        lam_th_0:   float,
        lam_om_0:   float,
        alpha_cmd_0: float,
    ) -> None:
        """Log the full planned heading profile for one BVP solve.

        The lam_th, lam_om, alpha_cmd columns are only populated on tick=0
        so the CSV stays readable without pivoting.
        """
        t0 = self._now()
        rows = []
        for i in range(len(thetas_deg)):
            rows.append([
                f'{t0:.4f}', 'plan', traj_id, chunk, i,
                '', '',
                f'{float(thetas_deg[i]):.3f}',
                f'{float(omegas[i]):.5f}',
                f'{float(vs[i]):.5f}',
                f'{lam_th_0:.5f}' if i == 0 else '',
                f'{lam_om_0:.5f}' if i == 0 else '',
                f'{alpha_cmd_0:.5f}' if i == 0 else '',
            ])
        with self._lock:
            self._w.writerows(rows)

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None


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

    # --- Speed reference scale and state bounds ---
    # v_max, omega_max: bounds on the chassis velocity STATES inside the
    # BVP -- not on the published cmd. Enforced softly via the
    # w_v_barrier / w_omega_barrier terms in the cost. The published cmd
    # is the inversion of these states through the chassis model
    # (chassis_gain_*, chassis_tau_*); during transients the cmd
    # legitimately exceeds v_max / omega_max by the lead term
    # |tau * accel_state|.
    v_max:     float = 0.5
    omega_max: float = 1.5

    # a_max, alpha_max: HARD bounds on the linear/angular acceleration
    # controls [m/s^2, rad/s^2], tanh-saturated in _ode. Measure from a
    # chassis step-response: peak observed |dv/dt| when commanding a
    # v_max step (for a_max), peak |domega/dt| when commanding an
    # omega_max step (for alpha_max). The states v, omega evolve as
    # dv/dt = a and domega/dt = alpha. These bounds also set how large
    # the inversion's lead term (tau * accel) can grow, so an over-
    # estimated alpha_max paired with a too-large chassis_tau_omega is
    # what drives published omega_cmd into saturation during ramps.
    a_max:     float = 1.0
    alpha_max: float = 3.0

    # cmd_deadzone_v, cmd_deadzone_omega: lower threshold for the
    # published twist (applied AFTER the chassis-model inversion).
    # The BVP-predicted v, omega can be genuinely near-zero values
    # during turn-in-place phases (brake cost wants v=0, speed
    # reference is small). Below the deadzone threshold, the planner
    # publishes 0 instead -- chassis stays stationary, the BVP's plan
    # and the chassis's reality agree. Set above the noise floor of
    # your BVP solve but well below the smallest "intended" movement
    # command. Defaults give ~3 cm/s linear, ~3 deg/s angular dead band.
    cmd_deadzone_v:     float = 0.03
    cmd_deadzone_omega: float = 0.05

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
    #    (1/2) * w_brake * (1 - F . h)^2 * v^2. With v as a state, the
    #    brake's effect appears in lambda_v_dot rather than directly in
    #    the optimal control, but the qualitative behavior is the same:
    #    gentle near alignment (cornering smooth), strong when anti-
    #    aligned (drives a* < 0 via large lambda_v).
    # L_brake: speed-reference length scale [m]. v_ref = v_max *
    #    tanh(d_to_goal / L_brake). Set near the chassis stopping distance.
    #    Smaller values brake earlier; larger values maintain speed closer
    #    to the goal.
    # align_gate_power: sharpness of the heading-alignment gate multiplying
    #    v_ref: gate = ((1 + F.h) / 2)^p. p=4 (default) gives gate(perp)
    #    = 0.06 -- effective braking when perpendicular. p=2 gives
    #    gate(perp) = 0.25 (racing-line cornering). p=8 is near binary.
    # w_omega_run: quadratic regularizer on state omega (NOT the control).
    #    Penalises being in a rotating state without active commanding.
    #    Small (~0.1) by default; larger values damp residual rotation
    #    but also blunt sharp turns.
    # gamma_a, gamma_alpha: quadratic regularizers on the controls.
    #    Set the closed-form laws
    #        a*     = -lambda_v     / gamma_a,     sat by a_max
    #        alpha* = -lambda_omega / gamma_alpha, sat by alpha_max
    #    Smaller gamma = more aggressive; larger = smoother. Tune so
    #    unsaturated controls at typical operating points sit in the
    #    middle of the saturation range.
    # w_v_barrier, w_omega_barrier: soft state-bound barriers, kept as
    #    safety nets. The acceleration models on v and omega allow the
    #    states to drift above v_max / omega_max if the controls are
    #    saturated and held; the barriers drive the corresponding costate
    #    large in the right direction to brake.
    alpha_t: float = 1.0
    beta:    float = 5.0
    w_h:     float = 5.0
    w_v:     float = 0.5
    w_brake: float = 200.0
    L_brake: float = 0.5
    align_gate_power: float = 4.0
    w_omega_run:     float = 0.1
    gamma_a:         float = 0.5
    gamma_alpha:     float = 0.2
    w_v_barrier:     float = 50.0
    w_omega_barrier: float = 50.0

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
    # w_T_terminal: Lyapunov-in-T-space terminal weight on T_lin^2 where
    #    T_lin = T_ref - F_ref . (p_T - p_pursuit). Provides LONG-RANGE
    #    pull along -F_ref (the geometry of the field), complementary to
    #    the local isotropic w_pp well. The transversality is
    #    lambda_xy(T) = -w_T_terminal * T_lin * F_ref + w_pp * (p_T - p_pursuit),
    #    so the two terminal-position terms cooperate when the BVP
    #    endpoint sits along the streamline and disagree softly when it
    #    drifts off-axis. Raise to bias the endpoint toward the
    #    streamline; lower if Newton struggles on sharp T fields.
    # w_pp: small isotropic pursuit-point pull (1/2)*w_pp*||p_T - p_pursuit||^2.
    #    Breaks the half-pipe degeneracy of T_lin^2 alone (which is
    #    degenerate perpendicular to F_ref). Keep small relative to
    #    w_T_terminal so it doesn't fight the streamline pull.
    # w_th: terminal heading basin: (1/2)*w_th*(theta_T - theta_pursuit)^2.
    #    Also drives the running heading spring when w_F ~ 0 (at goal).
    # w_v_terminal, w_omega_terminal: stop-condition weights on v_T^2,
    #    omega_T^2. Set both > 0 to tell the planner the chassis should
    #    arrive stationary; lambda_v(T) = w_v_terminal * v_T and
    #    lambda_omega(T) = w_omega_terminal * omega_T are the
    #    corresponding transversalities. Setting to 0 leaves terminal
    #    velocity free (chassis "drives through" the goal).
    # pursuit_lookahead_mult: target arc length as a multiple of
    #    v_max * T_horizon. 1.0 places the terminal target where the
    #    chassis would arrive if it tracked F at full speed. < 1 leaves
    #    slack (eases Newton on hard fields); > 1 reaches past the natural
    #    horizon (tighter tracking, harder convergence).
    w_T_terminal:     float = 2.0
    w_pp:             float = 0.5
    w_th:             float = 2.0
    w_v_terminal:     float = 5.0
    w_omega_terminal: float = 5.0
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

    # --- Chassis model for feedforward inversion at publication ---
    # The BVP plans in "desired body behaviour" space; published cmds
    # are
    #   v_cmd     = (v_state     + tau_v     * a_star)     / gain_v
    #   omega_cmd = (omega_state + tau_omega * alpha_star) / gain_omega
    # where a_star, alpha_star are the BVP-optimal controls.
    #
    # These parameters do NOT enter the BVP itself -- they only
    # transform the BVP-planned (state, control) into a published cmd
    # which, after the chassis's static gain and first-order tracking
    # lag, produces the BVP-planned state. Identify gain and tau from
    # an angular / linear step-response test on the target platform.
    #
    # gain < 1 (typical for skid-steer): the chassis rotates / drives
    #   at a fraction of the cmd's nominal rate, because lateral wheel
    #   slip during rotation (or longitudinal slip for v) eats some of
    #   the wheel-speed differential. Setting gain = 0.77 means commanding
    #   omega = 1.3 to achieve 1.0 rad/s of body rotation.
    # tau: first-order time constant of the velocity-tracking loop.
    #   For skid-steer, dominated by lateral-friction dynamics rather
    #   than the inner wheel-velocity loop; can vary with omega
    #   magnitude. ERR ON THE LOW SIDE -- under-estimating tau makes
    #   the chassis lag the BVP plan slightly; over-estimating it
    #   amplifies cmd transients and causes overshoot. tau = 0 gives
    #   static-gain-only compensation, which preserves the integral of
    #   the planned motion exactly but lags in timing.
    chassis_gain_v:     float = 1.0
    chassis_gain_omega: float = 0.77   # measured for Scout Mini in sim
    chassis_tau_v:      float = 0.10
    chassis_tau_omega:  float = 0.30   # err low; over-estimate causes overshoot

    # Hard caps on the PUBLISHED cmd (post-inversion). Should be >= the
    # chassis controller's max_velocity so the inversion's lead term
    # isn't clipped. The BVP's v_max / omega_max are SEPARATE bounds in
    # the planning model and apply to the BVP state, not the cmd.
    chassis_v_max:      float = 5.0
    chassis_omega_max:  float = 5.0

    @property
    def dt(self) -> float:
        return self.T_horizon / self.N


@dataclass
class TopicConfig:
    map_frame: str = "map"
    robot_frame: str = "base_link"
    enable_stamped_cmd_vel: bool = False
    # Set to a file path (e.g. /tmp/pmp_diag.csv) to enable the diagnostic
    # logger. Empty string disables it. The logger writes planned heading
    # profiles and actual odom to CSV for post-analysis; see TurnDiagnosticLogger.
    diag_log_path: str = ""


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
    """TPBVP solver for the 5D unicycle PMP problem.

    State (5): (p_x, p_y, theta, v, omega).
    Control (2): (a, alpha) -- linear and angular acceleration.

    The published twist is the planner's PREDICTED chassis velocity at
    each tick, i.e. the state (v, omega) trajectory -- not the control.
    The chassis driver receives a velocity setpoint that already respects
    acceleration bounds, so its low-level tracker has nothing to fight.
    Online mode reads (v_now, omega_now) from /odom and pins them as
    initial conditions; offline mode propagates simulated (v, omega)
    across segments via the BVP's own state evolution.

    Same solver in both modes. Online mode calls solve() per tick and
    publishes the state at the next control tick; offline mode calls
    solve() once per rollout segment and densely samples the BVP solution.
    """

    def __init__(self, cfg: PlannerConfig, field: VectorFieldGrid):
        self.cfg = cfg
        self.field = field
        self._prev_sol = None
        # Cached per-solve for the BC/ODE closures.
        self._x0: Optional[np.ndarray] = None        # 5-vector now
        self._goal: Optional[np.ndarray] = None
        # Last successful trajectory for introspection / publishing.
        self._last_state: Optional[np.ndarray] = None      # (m, 5)
        self._last_costate: Optional[np.ndarray] = None    # (m, 5)
        self._last_error: Optional[str] = None
        # Field-version tracking: a replaced field invalidates the warm start.
        self._last_field_version: int = -1
        # Terminal targets (set per-solve in solve()):
        #   p_pursuit     -- streamline endpoint at the lookahead distance.
        #   theta_pursuit -- F-tangent at p_pursuit, blended toward
        #                    theta_goal inside the approach zone.
        #   T_ref, F_ref  -- T(p_pursuit) and F_unit(p_pursuit), reused
        #                    by _bc for the T_lin Lyapunov transversality.
        self._p_pursuit: np.ndarray = np.zeros(2, dtype=np.float64)
        self._theta_pursuit: float = 0.0
        self._T_ref: float = 0.0
        self._F_ref: np.ndarray = np.zeros(2, dtype=np.float64)
        # Field-alignment-cost fade: w_F in [0, 1], cubic smoothstep of
        # chassis-to-goal distance. Applied ONLY to the alignment cost
        # (w_F * w_h * (1 - F.h)); speed and brake fade naturally via
        # v_ref -> 0 and v -> 0 near the goal, no explicit fade needed.
        self._align_fade: float = 1.0
        # Cross-track reference (only populated when cfg.w_xt > 0). Empty
        # arrays cause _ode's cross-track block to short-circuit cheaply.
        self._xt_ref: np.ndarray = np.zeros((0, 2))
        self._xt_n_perp: np.ndarray = np.zeros((0, 2))
        self._xt_sigma: float = 0.15
        # Duration of the last committed segment [s].  Used to time-shift the
        # offline warm start: for online mode this is one tick (~ 0 shift
        # relative to T_horizon); for offline mode it is dt_segment, and
        # evaluating prev_sol at [0, T_h] instead of [seg_T, T_h] feeds the
        # *start* of the previous plan rather than its tail -- a 50 % phase
        # error in the costate waveform that systematically kills the turn.
        self._last_seg_T: float = 0.0

    # --- Augmented dynamics ------------------------------------------------

    def _ode(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """RHS of the (state, costate) ODE, vectorized over the BVP mesh.

        y has shape (10, m) where rows are
            [p_x, p_y, theta, v, omega, lambda_x, lambda_y, lambda_th,
             lambda_v, lambda_omega].
        """
        cfg = self.cfg
        px, py, th, v, w = y[0], y[1], y[2], y[3], y[4]
        lx, ly, lt, lv, lw = y[5], y[6], y[7], y[8], y[9]
        cos_t = np.cos(th)
        sin_t = np.sin(th)

        T_now, dT_dx, dT_dy, Fux, Fuy = self.field.query_vec(px, py)
        F_dot_h = Fux * cos_t + Fuy * sin_t
        cross_F_h = Fux * sin_t - Fuy * cos_t

        # Speed-reference scaffolding (same as 3D version).
        v_ref = np.zeros_like(px)
        v_ref_eff = np.zeros_like(px)
        gate_prime = np.zeros_like(px)
        if cfg.w_v > 0.0:
            d_to_goal = np.sqrt((px - self._goal[0]) ** 2
                                + (py - self._goal[1]) ** 2)
            v_ref = cfg.v_max * np.tanh(d_to_goal / cfg.L_brake)
            half_one_plus = 0.5 * (1.0 + F_dot_h)
            half_one_plus = np.clip(half_one_plus, 0.0, 1.0)
            p_gate = cfg.align_gate_power
            gate = half_one_plus ** p_gate
            if p_gate > 0.0:
                gate_prime = (0.5 * p_gate) * (half_one_plus ** (p_gate - 1.0))
            else:
                gate_prime = np.zeros_like(half_one_plus)
            v_ref_eff = v_ref * gate

        # Closed-form optimal controls. Both channels use acceleration
        # control (a, alpha are controls; v, omega are integrators).
        # a*     = -lambda_v     / gamma_a       (sat |a|     <= a_max)
        # alpha* = -lambda_omega / gamma_alpha   (sat |alpha| <= alpha_max)
        #
        # The true PMP optimal is clip(unsat, -bound, +bound). We use tanh
        # with K=1 rather than a hard clip to keep the ODE smooth for
        # solve_bvp's collocation.  K=1 underestimates the control by 24 %
        # exactly at the saturation boundary, but with the physics-based
        # cold-start costate (|lam_om_0| ~= 1.2 = 2x the saturation threshold
        # gamma_alpha * alpha_max = 0.6), the actual unsaturated value is
        # |lam_om_0|/gamma_alpha = 6, giving tanh(2) = 0.964 -- only 3.6 %
        # error in practice.  Using a sharper K (e.g. 8) collapses tanh to a
        # near-step function at the switching point; solve_bvp then meshes
        # the near-discontinuity to death and hits bvp_max_nodes every solve.
        a_unsat     = -lv / cfg.gamma_a
        alpha_unsat = -lw / cfg.gamma_alpha
        a     = cfg.a_max     * np.tanh(a_unsat     / cfg.a_max)
        alpha = cfg.alpha_max * np.tanh(alpha_unsat / cfg.alpha_max)

        # State dynamics. Both v and omega are integrators of bounded
        # controls; no first-order chassis-lag modeling on either channel.
        dpx = v * cos_t
        dpy = v * sin_t
        dth = w
        dv  = a
        dw  = alpha

        # Position costates -- same as 3D, only L_pos contributes under
        # the frozen-field approximation.
        T_clip = np.minimum(T_now, cfg.T_horizon)
        dlx = -cfg.beta * T_clip * dT_dx / cfg.T_horizon
        dly = -cfg.beta * T_clip * dT_dy / cfg.T_horizon

        # Cross-track residual (opt-in). Same as 3D: adds
        # -w_xt * r_xt * n_perp to (dlx, dly) with a soft Gaussian
        # projection onto the streamline reference.
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

        # Heading costate. Field-aligned terms are listed below; the
        # alignment cost itself is faded by w_F, but speed and brake
        # contributions are NOT explicitly faded -- they vanish naturally
        # near the goal via v_ref -> 0 and v -> 0, respectively. The
        # (1 - F.h) factor on the brake term vanishes exactly at
        # alignment, so noise in F doesn't pump omega during straight
        # drive.
        fade = self._align_fade
        delta_yaw = th - self._theta_pursuit
        one_minus_dot = 1.0 - F_dot_h
        dlt = (-cfg.w_h * cross_F_h * fade
               - cfg.w_h * delta_yaw * (1.0 - fade)
               - cfg.w_v * v_ref * (v - v_ref_eff) * gate_prime * cross_F_h
               - cfg.w_brake * one_minus_dot * v * v * cross_F_h
               + lx * v * sin_t - ly * v * cos_t)

        # v-costate. No self-coupling -- a is an unrestricted control
        # affecting only dv/dt, so dH/dv has no -lambda_v term. This
        # is the same structure as the original 5D acceleration model.
        dlv = (-cfg.w_v * (v - v_ref_eff)
               - cfg.w_brake * one_minus_dot * one_minus_dot * v
               - lx * cos_t - ly * sin_t)

        # omega-costate. No self-coupling -- alpha is an unrestricted
        # control affecting only domega/dt, so dH/domega has no
        # -lambda_omega term. Symmetric with lambda_v.
        dlw = -cfg.w_omega_run * w - lt

        # Soft state-bound barriers. The closed-form a*, alpha* are
        # bounded but the resulting v, omega trajectories aren't; without
        # these the BVP can plan v > v_max (e.g. ~0.66 vs v_max=0.5 with
        # default tuning, driven by the position pull via lambda_x,y).
        # The publish-side clip then masks the overshoot online but
        # leaves the offline next-segment IC inconsistent with what the
        # chassis can execute.
        # Cost: (1/2) * w_bar * max(0, |state| - bound)^2.
        # Gradient w.r.t. state: w_bar * sign(state) * max(0, |state| - bound).
        # Contributes -gradient to the costate dynamics.
        v_excess = np.maximum(0.0, np.abs(v) - cfg.v_max)
        w_excess = np.maximum(0.0, np.abs(w) - cfg.omega_max)
        dlv -= cfg.w_v_barrier     * np.sign(v) * v_excess
        dlw -= cfg.w_omega_barrier * np.sign(w) * w_excess

        return np.vstack([dpx, dpy, dth, dv, dw, dlx, dly, dlt, dlv, dlw])

    # --- Boundary conditions ----------------------------------------------

    def _bc(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """Ten BC residuals; solve_bvp drives these to zero.

        Initial state (5): pose from TF and twist from /odom are pinned.
        Terminal (5): transversalities from Phi(x_T):
            lambda_x(T)  = -w_T_terminal * T_lin * F_ref_x
                           + w_pp * (p_x_T - p_x_pursuit)
            lambda_y(T)  = -w_T_terminal * T_lin * F_ref_y
                           + w_pp * (p_y_T - p_y_pursuit)
            lambda_th(T) = w_th * (theta_T - theta_pursuit)
            lambda_v(T)  = w_v_terminal * v_T
            lambda_om(T) = w_omega_terminal * omega_T
        T_lin = T_ref - F_ref . (p_T - p_pursuit) is the linearization
        of the T-field around the pursuit point, giving long-range
        gradient along -F_ref that complements the running L_pos.
        """
        cfg = self.cfg
        x0 = self._x0
        ppx = float(self._p_pursuit[0])
        ppy = float(self._p_pursuit[1])
        thp = self._theta_pursuit
        F_ref_x = float(self._F_ref[0])
        F_ref_y = float(self._F_ref[1])
        T_lin = (self._T_ref
                 - F_ref_x * (yb[0] - ppx)
                 - F_ref_y * (yb[1] - ppy))

        # Terminal transversalities derived from Phi.
        lam_x_T = -cfg.w_T_terminal * T_lin * F_ref_x + cfg.w_pp * (yb[0] - ppx)
        lam_y_T = -cfg.w_T_terminal * T_lin * F_ref_y + cfg.w_pp * (yb[1] - ppy)
        lam_th_T = cfg.w_th * (yb[2] - thp)
        lam_v_T  = cfg.w_v_terminal     * yb[3]
        lam_om_T = cfg.w_omega_terminal * yb[4]

        return np.array([
            ya[0] - x0[0],     # p_x(0)
            ya[1] - x0[1],     # p_y(0)
            ya[2] - x0[2],     # theta(0)
            ya[3] - x0[3],     # v(0)
            ya[4] - x0[4],     # omega(0)
            yb[5] - lam_x_T,   # lambda_x(T_h)
            yb[6] - lam_y_T,   # lambda_y(T_h)
            yb[7] - lam_th_T,  # lambda_th(T_h)
            yb[8] - lam_v_T,   # lambda_v(T_h)
            yb[9] - lam_om_T,  # lambda_omega(T_h)
        ])

    # --- Initial guess ----------------------------------------------------

    def _rollout_guess(
        self, x0: np.ndarray, goal: np.ndarray, t_mesh: np.ndarray,
    ) -> np.ndarray:
        """Forward-rollout a feasible 5D state trajectory descending the field.

        Picks a desired heading from F_unit (or -grad T, or the goal
        direction as fallbacks), tracks heuristic v, omega targets with
        bounded-acceleration P controllers, and integrates state under
        the same integrator dynamics as the BVP _ode.

        Costates ramp linearly from 0 to the terminal transversality
        evaluated at the rollout endpoint -- small warm-start magnitudes
        are what Newton prefers.
        """
        cfg = self.cfg
        m = t_mesh.size
        dt = t_mesh[1] - t_mesh[0] if m > 1 else cfg.T_horizon

        state = np.zeros((5, m))
        state[:, 0] = x0
        px, py, th, v, w = (float(x0[0]), float(x0[1]), float(x0[2]),
                            float(x0[3]), float(x0[4]))

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
            half_one_plus = max(0.0, 0.5 * (1.0 + cos(e)))
            gate = half_one_plus ** cfg.align_gate_power
            v_target = cfg.v_max     * tanh(d_goal / cfg.L_brake) * gate
            w_target = max(min(2.0 * e, cfg.omega_max), -cfg.omega_max)

            # Both channels: bounded-acceleration tracking of the heuristic
            # velocity target. Matches the BVP _ode's integrator dynamics.
            a_cmd     = max(min(2.0 * (v_target - v), cfg.a_max),     -cfg.a_max)
            alpha_cmd = max(min(2.0 * (w_target - w), cfg.alpha_max), -cfg.alpha_max)
            v  += a_cmd     * dt
            w  += alpha_cmd * dt
            px += v * cos(th) * dt
            py += v * sin(th) * dt
            th += w * dt

            state[:, k] = (px, py, th, v, w)

        # Terminal transversalities evaluated at the rolled-out endpoint.
        ppx = float(self._p_pursuit[0])
        ppy = float(self._p_pursuit[1])
        F_ref_x = float(self._F_ref[0])
        F_ref_y = float(self._F_ref[1])
        px_T, py_T, th_T, v_T, w_T = (state[0, -1], state[1, -1], state[2, -1],
                                      state[3, -1], state[4, -1])
        T_lin = self._T_ref - F_ref_x * (px_T - ppx) - F_ref_y * (py_T - ppy)

        lam_x_T  = -cfg.w_T_terminal * T_lin * F_ref_x + cfg.w_pp * (px_T - ppx)
        lam_y_T  = -cfg.w_T_terminal * T_lin * F_ref_y + cfg.w_pp * (py_T - ppy)
        lam_th_T = cfg.w_th             * (th_T - self._theta_pursuit)
        lam_v_T  = cfg.w_v_terminal     * v_T
        lam_om_T = cfg.w_omega_terminal * w_T

        # Physical cold-start estimate for the initial costates.
        #
        # The naive ramp (s * terminal_value) leaves lambda_omega ~= 0
        # throughout when omega_T ~= 0 (turn-to-stop), giving alpha* ~= 0
        # from the start. That is inconsistent with the state trajectory
        # and can trap Newton in a low-rotation local minimum on cold
        # starts.
        #
        # Instead we derive estimates from the simplified costate dynamics:
        #
        #   d(lam_th)/dt ~= -w_h * cross_F_h            (dominant term)
        #   lam_th(T)    ~= 0                           (successful-turn BC)
        #
        # Integrating forward from t=0 to T:
        #   lam_th(T) - lam_th(0) = -w_h * integral(cross_F_h dt)
        #                         ~= -w_h * cross_F_h_0 / 2 * T_turn
        #   =>  lam_th(0) ~= w_h * cross_F_h_0 * T_turn / 2
        #
        # Sign check for CCW turn (cross_F_h_0 = -1):
        #   lam_th(0) = 5 * (-1) * T/2 < 0   [ok]
        #
        #   d(lam_om)/dt = -w_omega_run * omega - lam_th;
        #   with lam_th < 0 this drives lam_om upward in forward time, so
        #   lam_om(0) must be negative (below its terminal ~= 0). We pin
        #   it just past the saturation boundary gamma_alpha * alpha_max:
        #     lam_om(0) ~= gamma_alpha * alpha_max * cross_F_h_0 * K
        #
        # Sign check for CCW (cross_F_h_0 = -1):
        #   lam_om(0) = 0.2 * 3 * (-1) * 2 = -1.2 < 0
        #   =>  alpha* = -(-1.2) / 0.2 = +6  ->  saturates at +alpha_max  [ok]
        th_0 = float(x0[2])
        _, _, _, fux_0, fuy_0 = self.field.query_scalar(float(x0[0]), float(x0[1]))
        cross_F_h_0 = fux_0 * sin(th_0) - fuy_0 * cos(th_0)

        # Fraction of T_horizon we expect to be in the "active turn" phase;
        # clamp to [0, 1] so the estimate is sensible near the goal.
        heading_err_0 = ((self._theta_pursuit - th_0 + pi) % (2.0 * pi)) - pi
        turn_frac = min(1.0, abs(heading_err_0) / (pi / 4.0))

        # Note the sign: NO leading minus on w_h or gamma_alpha.
        # The cross_F_h_0 factor carries the correct sign for both channels.
        lam_th_0 = cfg.w_h * cross_F_h_0 * cfg.T_horizon * 0.5 * turn_frac

        lam_om_0 = (cfg.gamma_alpha * cfg.alpha_max
                    * cross_F_h_0 * 2.0 * turn_frac + lam_om_T * (1.0 - turn_frac))

        s = (t_mesh - t_mesh[0]) / max(t_mesh[-1] - t_mesh[0], 1e-9)
        return np.vstack([
            state[0], state[1], state[2], state[3], state[4],
            s * lam_x_T,
            s * lam_y_T,
            (1.0 - s) * lam_th_0 + s * lam_th_T,
            s * lam_v_T,
            (1.0 - s) * lam_om_0 + s * lam_om_T,
        ])

    # --- Solve API --------------------------------------------------------

    def _predistort(
        self,
        v_state: np.ndarray,
        omega_state: np.ndarray,
        lam_v: np.ndarray,
        lam_w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Feedforward inversion of the chassis's static gain + first-order
        tracking lag. Returns (v_cmd, omega_cmd) -- the commands that,
        after the chassis dynamics, produce the BVP-planned state.

        Chassis model:
            tau * d(actual)/dt + actual = gain * cmd
        Inversion (matched IC implied):
            cmd(t) = (desired(t) + tau * d(desired)/dt) / gain
        Here desired = BVP state (v or omega); its derivative is the
        BVP-optimal control a* / alpha*, recovered from the costates
        by the same tanh-saturated formula used in _ode.

        Vectorizes over array inputs (used by sample_committed_segment)
        and also accepts scalars via numpy broadcasting (used by solve,
        _fallback_command, _control_law_pointwise). Clipping is applied
        here; deadzone is the caller's responsibility (it's the boundary
        where "command 0" stationarity vs "command something tiny"
        matters, and that distinction lives at the publication site).
        """
        cfg = self.cfg
        a_star     = cfg.a_max     * np.tanh(-lam_v / (cfg.gamma_a     * cfg.a_max))
        alpha_star = cfg.alpha_max * np.tanh(-lam_w / (cfg.gamma_alpha * cfg.alpha_max))
        v_pre = (v_state     + cfg.chassis_tau_v     * a_star)     / cfg.chassis_gain_v
        w_pre = (omega_state + cfg.chassis_tau_omega * alpha_star) / cfg.chassis_gain_omega
        v_cmd = np.clip(v_pre, -cfg.chassis_v_max,     +cfg.chassis_v_max)
        w_cmd = np.clip(w_pre, -cfg.chassis_omega_max, +cfg.chassis_omega_max)
        return v_cmd, w_cmd

    def _fallback_command(self) -> Optional[Tuple[float, float]]:
        """Evaluate the fallback published twist from the previous BVP
        solution at the next control tick. Applies the same chassis-model
        inversion as the main publication path. The prev_sol callable is
        freed on failure to prevent a second fallback attempt with a
        further-extrapolated bad solution.
        """
        if self._prev_sol is None:
            return None
        cfg = self.cfg
        t_eval = min(1.0 / cfg.control_rate, cfg.T_horizon)
        try:
            y_eval = self._prev_sol(t_eval)
        except Exception:
            self._prev_sol = None
            return None
        v_cmd_a, w_cmd_a = self._predistort(
            v_state=y_eval[3], omega_state=y_eval[4],
            lam_v=y_eval[8],   lam_w=y_eval[9],
        )
        v_cmd, w_cmd = float(v_cmd_a), float(w_cmd_a)
        if abs(v_cmd) < cfg.cmd_deadzone_v:     v_cmd = 0.0
        if abs(w_cmd) < cfg.cmd_deadzone_omega: w_cmd = 0.0
        return v_cmd, w_cmd

    def solve(
        self, x0: np.ndarray, goal: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Solve the TPBVP. x0 is the 5-vector (px, py, theta, v, omega).

        Returns (v_cmd, omega_cmd) -- the publication-ready twist for
        /cmd_vel, post chassis-model inversion (see _predistort) -- or
        None on failure.

        Side effects (used by both online and offline paths):
          - self._prev_sol     : OdeSolution callable, t in [0, T_horizon]
          - self._last_state   : (N+1, 5) dense state samples
          - self._last_costate : (N+1, 5) dense costate samples
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
            self._p_pursuit = np.array([goal[0], goal[1]], dtype=np.float64)

        # T_ref and F_ref at the pursuit point -- used by both _bc (for
        # the T_lin transversality) and the rollout guess.
        T_pp, _, _, fux_pp, fuy_pp = self.field.query_scalar(
            float(self._p_pursuit[0]), float(self._p_pursuit[1]),
        )
        self._T_ref = float(T_pp)

        chassis_to_goal_sq = ((float(x0[0]) - goal[0]) ** 2
                              + (float(x0[1]) - goal[1]) ** 2)
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
        target_norm = np.hypot(target_x, target_y)
        if target_norm < 1e-3:
            if w_F >= 0.5:
                self._theta_pursuit = atan2(F_hat_y, F_hat_x)
            else:
                self._theta_pursuit = float(goal[2])
        else:
            self._theta_pursuit = float(atan2(target_y, target_x))

        # F_ref blends F_hat with goal_hat too -- the T_lin pull direction
        # should track the same target as theta_pursuit. Outside the goal
        # zone (w_F = 1) this is just F_hat as before.
        if target_norm < 1e-3:
            self._F_ref = (np.array([F_hat_x, F_hat_y]) if w_F >= 0.5
                           else np.array([goal_hat_x, goal_hat_y]))
        else:
            self._F_ref = np.array([target_x / target_norm,
                                    target_y / target_norm])

        self._align_fade = w_F

        # Unwrap theta_pursuit relative to the current chassis heading.
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
                # Time-shift the warm start for offline mode.
                #
                # Online: prev_sol was solved one tick (0.1 s) ago; evaluating
                # it at [0, T_h] is a near-perfect warm start.
                #
                # Offline: prev_sol was solved seg_T seconds ago (~ 1.25 s).
                # The tail of that solution -- prev_sol(t + seg_T) -- is the
                # planner's best prior belief about the new time window. Evaluating
                # at [0, T_h] instead gives the *start* of the old plan, which
                # has the wrong costate phase for the current heading error.
                t_shifted = np.minimum(t_mesh + self._last_seg_T, cfg.T_horizon)
                y_init = self._prev_sol(t_shifted)
                # Theta unwrap: if chassis crossed +-pi between cycles
                # the warm-start theta(0) differs from x0[2] by ~2*pi*k.
                # Shift the entire theta trajectory; then anchor all 5
                # initial-state components.
                prev_theta_0 = float(y_init[2, 0])
                n_shift = round((prev_theta_0 - float(x0[2])) / (2.0 * pi))
                if n_shift != 0:
                    y_init[2, :] -= n_shift * 2.0 * pi
                y_init[0:5, 0] = x0
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
        # For the next warm start: online commits one control tick.
        self._last_seg_T = min(1.0 / cfg.control_rate, cfg.T_horizon)

        # Online publication: invert the chassis model so the cmd, after
        # the chassis's static gain and first-order lag, produces the
        # BVP-planned state. See _predistort for the math. The deadzone
        # below catches genuinely-stationary phases (turn-in-place
        # produces v ~ 0 throughout) and the small numerical residuals.
        t_lookahead = min(1.0 / cfg.control_rate, cfg.T_horizon)
        y_at_dt = self._prev_sol(t_lookahead)
        v_cmd_a, w_cmd_a = self._predistort(
            v_state=y_at_dt[3], omega_state=y_at_dt[4],
            lam_v=y_at_dt[8],   lam_w=y_at_dt[9],
        )
        v_cmd, w_cmd = float(v_cmd_a), float(w_cmd_a)
        if abs(v_cmd) < cfg.cmd_deadzone_v:     v_cmd = 0.0
        if abs(w_cmd) < cfg.cmd_deadzone_omega: w_cmd = 0.0

        # Densely resample for visualization and downstream introspection.
        t_dense = np.linspace(0.0, cfg.T_horizon, cfg.N + 1)
        y_dense = self._prev_sol(t_dense)
        self._last_state = y_dense[0:5].T
        self._last_costate = y_dense[5:10].T
        self._last_error = None

        return v_cmd, w_cmd

    def reset_warm_start(self):
        self._prev_sol = None

    # --- Pointwise readout (shared by online + offline paths) -------------

    def _control_law_pointwise(
        self, y10: np.ndarray, goal: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute the published twist (v, omega) at a single mesh point
        from a (state, costate) slice y10. Same chassis-model inversion
        as the main publication paths -- see _predistort.
        """
        cfg = self.cfg
        v_cmd_a, w_cmd_a = self._predistort(
            v_state=y10[3], omega_state=y10[4],
            lam_v=y10[8],   lam_w=y10[9],
        )
        v_cmd, w_cmd = float(v_cmd_a), float(w_cmd_a)
        if abs(v_cmd) < cfg.cmd_deadzone_v:     v_cmd = 0.0
        if abs(w_cmd) < cfg.cmd_deadzone_omega: w_cmd = 0.0
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

        Returns (twists, poses, x_next) where:
          twists  : (n_samples, 2) [v_cmd, omega_cmd] -- the
                    chassis-model-inverted commands at each tick. These
                    are the values to publish; the chassis, after its
                    static gain and first-order lag, executes the
                    BVP-planned state. See _predistort for the math.
                    Clipped to chassis_v_max / chassis_omega_max.
          poses   : (n_samples, 3) [px, py, theta] -- BVP-planned chassis
                    pose at each tick, in parallel with twists.
          x_next  : (5,) BVP state at t = n_samples * dt_sample, the
                    start of the next segment (used to seed the next
                    solve). Assumes the chassis tracks the inverted
                    commands so its actual state matches the BVP's.

        Returns None if the BVP fails.
        """
        if self.solve(x0, goal) is None:
            return None
        if self._prev_sol is None:
            return None

        cfg = self.cfg
        seg_T = n_samples * dt_sample
        if seg_T > cfg.T_horizon + 1e-9:
            n_samples = max(1, int(cfg.T_horizon / dt_sample))
            seg_T = n_samples * dt_sample

        t_ticks = np.arange(n_samples + 1, dtype=np.float64) * dt_sample
        t_ticks = np.minimum(t_ticks, cfg.T_horizon)
        try:
            y_ticks = self._prev_sol(t_ticks)   # shape (10, n_samples + 1)
        except Exception as e:
            self._last_error = f"prev_sol resample failed: {e}"
            return None

        # twists are post-inversion commands. _predistort transforms
        # (v_state, omega_state, lambda_v, lambda_omega) into the cmd
        # that the chassis (per its first-order tracking model) needs
        # to receive to actually realise the BVP-planned state at this
        # tick. Clipping to chassis_v_max / chassis_omega_max is
        # already inside _predistort; the deadzone here matches the
        # online publication path.
        twists = np.zeros((n_samples, 2), dtype=np.float64)
        v_cmd_v, w_cmd_v = self._predistort(
            v_state=y_ticks[3, :n_samples],
            omega_state=y_ticks[4, :n_samples],
            lam_v=y_ticks[8, :n_samples],
            lam_w=y_ticks[9, :n_samples],
        )
        twists[:, 0] = v_cmd_v
        twists[:, 1] = w_cmd_v
        twists[np.abs(twists[:, 0]) < cfg.cmd_deadzone_v,     0] = 0.0
        twists[np.abs(twists[:, 1]) < cfg.cmd_deadzone_omega, 1] = 0.0
        poses = y_ticks[0:3, :n_samples].T.copy()

        # x_next is the BVP-predicted chassis state at the end of the
        # committed segment, used as the IC for the next BVP solve. The
        # publication inversion is designed so a chassis matching the
        # (gain, tau) model executes the inverted cmds and ends the
        # segment at exactly this state. To the extent the actual
        # chassis deviates from the model, x_next drifts from reality
        # -- that's the open-loop error budget for offline mode.
        x_next = y_ticks[0:5, n_samples].copy()

        # Record committed duration so solve() can time-shift the warm start.
        self._last_seg_T = seg_T

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

        self._xi: np.ndarray = np.zeros(3)              # (px, py, theta) from TF
        self._chassis_twist: np.ndarray = np.zeros(2)   # (v, omega) from /odom
        self._goal: Optional[np.ndarray] = None         # (gx, gy, gtheta)
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
                self.topic_cfg.map_frame, self.topic_cfg.robot_frame,
                rclpy.time.Time(),
            )
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            q  = t.transform.rotation
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
            cs = self._solver._last_costate   # (m, 5): lx, ly, lth, lv, lom
            st = self._solver._last_state     # (m, 5): px, py, th, v, om
            if cs is not None and st is not None:
                lam_th_0   = float(cs[0, 2])
                lam_om_0   = float(cs[0, 4])
                alpha_cmd_0 = float(
                    self.cfg.alpha_max
                    * tanh(-lam_om_0 / (self.cfg.gamma_alpha * self.cfg.alpha_max))
                )
                self._diag_logger.log_plan(
                    traj_id=-1, chunk=-1,
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

            field_ref = self._field        # GIL-atomic
            if not field_ref.ready:
                # Goal arrived before any field. The next field arrival
                # will kick us again.
                continue

            # 5D initial state for the rollout: pose from TF, twist
            # from /odom. The two reads are individually GIL-atomic;
            # we don't need them from the exact same odom callback,
            # since the next segment's start state comes from the BVP
            # solution itself.
            xi    = self._xi
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

            # Diagnostic: log planned heading profile + t=0 costates.
            if self._diag_logger is not None:
                cs = self._solver._last_costate
                if cs is not None:
                    lam_th_0    = float(cs[0, 2])
                    lam_om_0    = float(cs[0, 4])
                    alpha_cmd_0 = float(
                        cfg.alpha_max
                        * tanh(-lam_om_0 / (cfg.gamma_alpha * cfg.alpha_max))
                    )
                    self._diag_logger.log_plan(
                        traj_id=traj_id, chunk=chunk_idx,
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
