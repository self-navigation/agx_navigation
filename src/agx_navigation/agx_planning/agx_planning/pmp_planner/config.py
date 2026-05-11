from dataclasses import dataclass


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
    v_max: float = 0.5
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
    a_max: float = 1.0
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
    cmd_deadzone_v: float = 0.03
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
    beta: float = 5.0
    w_h: float = 5.0
    w_v: float = 0.5
    w_brake: float = 200.0
    L_brake: float = 0.5
    align_gate_power: float = 4.0
    w_omega_run: float = 0.1
    gamma_a: float = 0.5
    gamma_alpha: float = 0.2
    w_v_barrier: float = 50.0
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
    w_T_terminal: float = 2.0
    w_pp: float = 0.5
    w_th: float = 2.0
    w_v_terminal: float = 5.0
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
    w_xt: float = 0.0
    xt_horizon_m: float = 2.5
    xt_sigma_mult: float = 3.0

    # --- Goal tolerances ---
    # Both must be satisfied simultaneously to trigger REACHED (zero twist).
    # The at-goal yaw correction is handled by the BVP itself (gated v_ref
    # yields v~0 + omega!=0 at the goal) -- no separate supervisor needed.
    goal_tolerance_xy: float = 0.05  # [m]
    goal_tolerance_th: float = 0.20  # [rad]

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
    dt_segment: float = 1.25  # [s]

    # field_diff_threshold: max |T_new - T_old| along the planned path
    #    that triggers a replan [s]. Out-of-bounds cells in either grid are
    #    treated as +inf diff (newly discovered terrain always replans).
    #    Tune conservatively: too low causes spurious replans on minor field
    #    updates; too high lets the chassis execute a now-invalid plan
    #    through a changed obstacle.
    field_diff_threshold: float = 0.5  # [s]

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
    chassis_gain_v: float = 1.0
    chassis_gain_omega: float = 0.77  # measured for Scout Mini in sim
    chassis_tau_v: float = 0.10
    chassis_tau_omega: float = 0.30  # err low; over-estimate causes overshoot

    # Hard caps on the PUBLISHED cmd (post-inversion). Should be >= the
    # chassis controller's max_velocity so the inversion's lead term
    # isn't clipped. The BVP's v_max / omega_max are SEPARATE bounds in
    # the planning model and apply to the BVP state, not the cmd.
    chassis_v_max: float = 5.0
    chassis_omega_max: float = 5.0

    @property
    def dt(self) -> float:
        return self.T_horizon / self.N
