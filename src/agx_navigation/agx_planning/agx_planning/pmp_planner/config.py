from dataclasses import dataclass


@dataclass
class PlannerConfig:
    """Indirect-method planner parameters (wheel-space skid-steer model).

    The planner state is (p_x, p_y, theta, w_l, w_r) where w_l, w_r are
    the left/right wheel-pair angular speeds [rad/s]; controls are the
    wheel angular accelerations (a_l, a_r). Body velocities are derived
    through the lumped-slip kinematics

        v     = (wheel_radius / 2)            * (w_l + w_r)
        omega = (wheel_radius / track_eff)    * (w_r - w_l)

    with track_eff = track * slip_chi. Behaviour-shaping costs (speed
    reference, brake, alignment, omega regularizer, v/omega barriers)
    stay expressed in body space; actuation limits and control effort
    live in wheel space, where the speed/turn trade-off is physical.
    """

    # --- Operating mode ---
    # "online":  per-tick BVP solve + wheel-command publication at
    #            control_rate Hz on the JointGroupVelocityController topic.
    # "offline": worker-thread rollout-by-concatenation; chunks streamed
    #            as PlanToGoal action feedback for an interpreter/executor.
    #            See the node module docstring for full semantics.
    mode: str = "online"

    # --- Horizon ---
    # N: number of mesh nodes for the BVP initial guess (the adaptive
    #    mesh refines beyond this, up to bvp_max_nodes). More nodes give a
    #    better initial guess for curved trajectories at the cost of a
    #    slower first solve per cycle. 21 nodes (20 intervals) is a good
    #    default; raise to 31-41 if the warm-start repeatedly fails on
    #    sharp fields.
    # T_horizon: prediction window [s]. The BVP optimises over this entire
    #    window; only the t=0 command is applied (receding horizon).
    # control_rate: timer frequency [Hz] for online mode and sample rate
    #    for offline chunks. Warm-start solves are typically < 30 ms.
    N: int = 21
    T_horizon: float = 2.5
    control_rate: float = 10.0

    # --- Chassis geometry / skid-steer kinematics ---
    # wheel_radius, track: physical dimensions from scout_mini.xacro.
    # slip_chi: effective-track expansion factor (Mandow et al., IROS'07
    #    ICR model). Lateral skid during rotation makes the chassis turn
    #    as if the wheels were chi * track apart, chi >= 1. This replaces
    #    the old chassis_gain_omega feedforward inversion:
    #        slip_chi = 1 / chassis_gain_omega
    #    so the slip lives in the planning kinematics instead of a
    #    publication-side gain (applying both would correct twice).
    #    1.2987 = 1 / 0.77, the value identified for Scout Mini in sim
    #    (mu2 = 0.7 lateral friction). Re-identify per surface.
    # Re-identified 2026-07-29 (slip_ident.py, wheels mode, mean over 6 arcs
    # r=0.23-0.75m, spread 0.0255): 1.373, up from 1.2987 (~6%, not the ~30%
    # a since-retracted mis-measurement suggested -- see
    # rl-corrector-turn-induced-corridor-breach memory). Real, but does not
    # explain the multi-meter identity-baseline drift on Tier-B trajectories;
    # that cause is still open.
    wheel_radius: float = 0.08
    track: float = 0.416503
    slip_chi: float = 1.373

    # --- Body-space behaviour bounds ---
    # v_max, omega_max: soft bounds on the DERIVED body velocities,
    # enforced via w_v_barrier / w_omega_barrier. These encode desired
    # behaviour (cruise speed, comfortable yaw rate), not hardware
    # limits -- the hardware net is w_wheel_max below. v_max also sets
    # the v_ref scale and the pursuit lookahead distance.
    v_max: float = 0.5
    omega_max: float = 1.5

    # --- Wheel-space actuation ---
    # a_wheel_max: HARD bound on each wheel's angular acceleration
    #    control [rad/s^2], tanh-saturated in _ode. 12.5 = old
    #    a_max / wheel_radius, so a pure-linear ramp matches the old
    #    planner; a pure turn at the old alpha_max = 3.0 needs
    #    alpha_max * track_eff / (2 r) ~= 10.1, also within bound. The
    #    old (a_max, alpha_max) CORNER needed ~22.6 and is now
    #    deliberately infeasible: linear and angular acceleration share
    #    one per-wheel budget, as they do on the platform.
    # gamma_wheel: quadratic effort weight on (a_l, a_r). Sets the
    #    closed-form law a_i* = -lambda_wi / gamma_wheel (tanh-sat by
    #    a_wheel_max). 0.0016 = old gamma_a * r^2 / 2, which reproduces
    #    the old LINEAR-channel aggressiveness exactly; the implied
    #    angular effort weight becomes gamma_wheel * track_eff^2 /
    #    (2 r^2) ~= 0.037 vs the old 0.2, i.e. turning is ~5x cheaper.
    #    Raise w_omega_run if turn-in becomes too eager.
    # w_wheel_max, w_wheel_barrier: soft per-wheel SPEED barrier -- the
    #    true hardware net. 20.0 matches <limit velocity="20"> on the
    #    wheel joints; gz_ros2_control clips commands there anyway, so
    #    the planner should not plan past it. Rarely active at the
    #    default v_max / omega_max (their corner needs ~11.3 rad/s).
    a_wheel_max: float = 12.5
    gamma_wheel: float = 0.0016
    w_wheel_max: float = 20.0
    w_wheel_barrier: float = 50.0

    # --- Command deadzone (body space) ---
    # Applied to the body velocities reconstructed from the wheel
    # commands, BEFORE mapping back to wheels: a near-zero v with
    # active omega (turn-in-place) zeroes only the symmetric part.
    # Below threshold the planner publishes exact zeros so the BVP's
    # plan and the chassis's reality agree during stationary phases.
    cmd_deadzone_v: float = 0.03
    cmd_deadzone_omega: float = 0.05

    # --- Running-cost weights ---
    # alpha_t: constant time-penalty per second. Acts as a mild urgency
    #    signal; usually dominated by beta on long paths. Safe to leave at 1.
    # beta: weight on the piecewise-C^1 position potential L_pos(T).
    #    The gradient is beta * min(T, T_horizon) * grad(T) / T_horizon,
    #    so it fades to zero at the goal sink (T->0) for clean braking,
    #    and is capped to beta * grad(T) during navigation so it doesn't
    #    swamp the heading terms.
    # w_h: heading-alignment weight. Controls (1 - F_unit . h) in the
    #    running cost and the terminal-yaw spring. Tuning is non-monotonic:
    #    both w_h <= 3 and w_h >= 12 clear corners well; intermediate
    #    values (3-12) can find shortcuts.
    # w_v: speed-reference tracking weight (v - v_ref_eff)^2, with v
    #    derived from the wheel states. Set to 0 to disable.
    # w_brake: heading-coupled brake on v^2:
    #    (1/2) * w_brake * (1 - F . h)^2 * v^2. Acts through the wheel
    #    costates via dH/dv; gentle near alignment, strong when
    #    anti-aligned.
    # L_brake: speed-reference length scale [m]. v_ref = v_max *
    #    tanh(d_to_goal / L_brake). Set near the chassis stopping distance.
    #    LARGER values brake EARLIER (the tanh leaves v_max sooner).
    # align_gate_power: sharpness of the heading-alignment gate multiplying
    #    v_ref: gate = ((1 + F.h) / 2)^p. p=4 (default) gives gate(perp)
    #    = 0.06; p=2 gives 0.25 (racing-line cornering); p=8 near binary.
    # w_omega_run: quadratic regularizer on the DERIVED yaw-rate state.
    #    Penalises being in a rotating state without active commanding,
    #    and is the natural damper if gamma_wheel makes turning too cheap.
    # w_v_barrier, w_omega_barrier: soft body-state barriers (see above).
    alpha_t: float = 1.0
    beta: float = 5.0
    w_h: float = 5.0
    w_v: float = 0.5
    w_brake: float = 200.0
    L_brake: float = 0.5
    align_gate_power: float = 4.0
    w_omega_run: float = 0.1
    w_v_barrier: float = 50.0
    w_omega_barrier: float = 50.0

    # --- Field smoothing ---
    # field_eps: soft re-normalisation denominator for F_unit:
    #    |F_unit| = |F| / sqrt(|F|^2 + eps^2). Makes |F_unit| -> 0 where
    #    the underlying field collapses (goal sink, saddles), fading the
    #    alignment cost instead of letting it fight the terminal target.
    field_eps: float = 1e-2

    # --- Terminal-cost weights ---
    # w_T_terminal: Lyapunov-in-T-space terminal weight on T_lin^2 where
    #    T_lin = T_ref - F_ref . (p_T - p_pursuit). Long-range pull along
    #    -F_ref, complementary to the local isotropic w_pp well.
    # w_pp: small isotropic pursuit-point pull. Breaks the half-pipe
    #    degeneracy of T_lin^2 alone (degenerate perpendicular to F_ref).
    # w_th: terminal heading basin: (1/2)*w_th*(theta_T - theta_pursuit)^2.
    # w_v_terminal, w_omega_terminal: stop-condition weights on the
    #    DERIVED terminal body velocities v_T^2, omega_T^2. The wheel
    #    transversalities are the chain-rule images:
    #      lambda_wl(T) = c_v * w_v_terminal * v_T - c_w * w_omega_terminal * omega_T
    #      lambda_wr(T) = c_v * w_v_terminal * v_T + c_w * w_omega_terminal * omega_T
    #    Keeping them in body space preserves independent tuning of
    #    "stop translating" vs "stop rotating".
    # pursuit_lookahead_mult: target arc length as a multiple of
    #    v_max * T_horizon.
    w_T_terminal: float = 2.0
    w_pp: float = 0.5
    w_th: float = 2.0
    w_v_terminal: float = 5.0
    w_omega_terminal: float = 5.0
    pursuit_lookahead_mult: float = 1.0

    # --- Cross-track residual (opt-in, off by default) ---
    # Adds (1/2)*w_xt*r_xt(p)^2 to the running cost, where r_xt is the
    # signed perpendicular drift from the F-streamline at a soft Gaussian-
    # weighted projection. Off by default: empirically the BVP either
    # ignores a small w_xt or stalls at a large one.
    # xt_horizon_m: streamline arc length to trace [m]; should exceed
    #    v_max * T_horizon.
    # xt_sigma_mult: Gaussian projection bandwidth = mult * grid_resolution.
    w_xt: float = 0.0
    xt_horizon_m: float = 2.5
    xt_sigma_mult: float = 3.0

    # --- Goal tolerances ---
    # Both must be satisfied simultaneously to trigger REACHED (zero
    # wheel command). The at-goal yaw correction is handled by the BVP
    # itself (gated v_ref yields v~0 + omega!=0 at the goal).
    goal_tolerance_xy: float = 0.05  # [m]
    goal_tolerance_th: float = 0.20  # [rad]

    # --- BVP solver knobs ---
    # bvp_tol: solve_bvp residual tolerance. 1e-3 is the practical sweet
    #    spot: tighter (1e-4) roughly doubles solve time with little
    #    command improvement.
    # bvp_max_nodes: adaptive mesh node cap. 2000 is sufficient for most
    #    smooth fields; raise if sharp obstacle boundaries make solve_bvp
    #    consistently hit the cap.
    # bvp_verbose: 0 = silent, 1 = per-solve summary, 2 = per-iteration.
    # reuse_previous_solution: warm-start each solve from the previous BVP
    #    solution. The single biggest convergence aid. Automatically
    #    dropped on new goal, new field, or goal-zone boundary crossing.
    bvp_tol: float = 1e-3
    bvp_max_nodes: int = 2000
    bvp_verbose: int = 0
    reuse_previous_solution: bool = True

    # --- Offline-mode parameters (ignored when mode == "online") ---
    # dt_segment: committed arc length per BVP solve [s]. Each solve
    #    optimises over the full T_horizon, but only the first dt_segment
    #    seconds are published and the sim state is advanced by that amount.
    dt_segment: float = 1.25  # [s]

    # field_diff_threshold: max |T_new - T_old| along the planned path
    #    that triggers a replan [s] (consumed by the interpreter).
    field_diff_threshold: float = 0.5  # [s]

    # max_rollout_sim_time: safety cap on total rollout time [s].
    max_rollout_sim_time: float = 60.0  # [s]

    # --- Wheel-level publication model ---
    # The published command is the BVP-planned wheel-speed STATE (a
    # velocity setpoint that already respects the acceleration bounds),
    # optionally predistorted for a first-order wheel-speed tracking lag:
    #     w_cmd_i = w_state_i + tau_wheel * a_i*
    # tau_wheel: time constant of the wheel-velocity loop. 0.0 (default)
    #    is correct for gz_ros2_control's velocity interface, which
    #    tracks within a physics step; identify on real hardware from a
    #    wheel-speed step response. The old chassis_tau_omega (~0.3 s)
    #    was dominated by body-level lateral-friction dynamics, which
    #    have no per-wheel representation -- that transient is now part
    #    of the residual the downstream corrector handles.
    # wheel_cmd_max: hard cap on the PUBLISHED wheel command [rad/s].
    #    Matches the joint <limit velocity="20">; gz_ros2_control would
    #    clip there regardless, so clipping here keeps the plan honest.
    tau_wheel: float = 0.0
    wheel_cmd_max: float = 20.0

    # --- Derived quantities (not ROS parameters) ---

    @property
    def dt(self) -> float:
        return self.T_horizon / self.N

    @property
    def track_effective(self) -> float:
        """Slip-expanded track width [m] used by the kinematic map."""
        return self.track * self.slip_chi

    @property
    def c_v(self) -> float:
        """v = c_v * (w_l + w_r)."""
        return 0.5 * self.wheel_radius

    @property
    def c_w(self) -> float:
        """omega = c_w * (w_r - w_l)."""
        return self.wheel_radius / self.track_effective

    def wheels_to_body(self, wl, wr):
        """(w_l, w_r) -> (v, omega). Broadcasts over numpy arrays."""
        return self.c_v * (wl + wr), self.c_w * (wr - wl)

    def body_to_wheels(self, v, omega):
        """(v, omega) -> (w_l, w_r). Exact inverse of wheels_to_body."""
        half_sum = v / (2.0 * self.c_v)
        half_diff = omega / (2.0 * self.c_w)
        return half_sum - half_diff, half_sum + half_diff
