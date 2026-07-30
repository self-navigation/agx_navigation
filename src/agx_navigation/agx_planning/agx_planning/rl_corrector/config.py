"""Configuration for the RL runtime corrector (training + deployment).

A plain @dataclass so it loads through agx_planning.utils.declare_and_load_dataclass
exactly like PlannerConfig/CorrectorConfig. It carries no ROS or torch import, so
the pure modules (coeff/obs/reward/nominal) and their unit tests stay light.

KINEMATICS NOTE: wheel_radius/track/slip_chi and the body<->wheel formulas mirror
pmp_planner/config.py PlannerConfig. They are duplicated here (not imported) only
because importing PlannerConfig pulls in the scipy/scikit-fmm solver via the
pmp_planner package __init__. Keep the values and formulas in sync with the planner.
"""

from dataclasses import dataclass


@dataclass
class RLCorrectorConfig:
    # --- Action ---------------------------------------------------------
    # 4 = independent per-wheel residuals [fl, rl, fr, rr]; 2 = per-side
    # [left, right]. 2 is the default: the real Scout Mini chassis only accepts a
    # twist and does its own wheel-effort mapping, so a 4-D front/rear-asymmetric
    # residual is unrealizable on hardware (see scout-twist-only-chassis memory)
    # and would invite a sim-only exploit that can't transfer.
    action_dim: int = 2
    # residual_i = wheel_residual_max * a_i, with a_i in [-1, 1]. a_i = 0 -> zero
    # residual -> exact identity (the fail-safe contract; also where SAC's
    # zero-centred tanh policy naturally sits). ADDITIVE, in rad/s -- replaces the
    # old multiplicative coeff_k scheme, whose authority was proportional to
    # |nominal| and exactly zero whenever a nominal wheel command was zero (a
    # structural flaw independent of hyperparameters; see rl-corrector-diagnosis
    # memory). wheel_residual_max is the ACTION AUTHORITY dial, analogous to the
    # old coeff_k=0.2 ([0.8,1.2] on a ~20 rad/s command -- keep the same rough
    # authority budget as a starting point, i.e. ~0.2 * wheel_cmd_max, and
    # re-tune from there; the old bound existed because a wider one let the
    # policy over-rotate and breach the pi/2 heading corridor on turns (SAC_6/8:
    # heading breaches at |e_heading|~1.6-2.0 rad)). SINGLE SOURCE OF TRUTH for
    # train+deploy -- the deployed node reads this same value, so a policy must
    # be deployed with the wheel_residual_max it trained on.
    wheel_residual_max: float = 4.0
    # Per-wheel command clamp [rad/s]; matches the joint <limit velocity>.
    wheel_cmd_max: float = 20.0
    # Max |a_t - a_{t-1}| per control_dt in the clipped [-1,1] action, applied
    # in coeff.clipped_action. 0 disables. A policy trained on KinematicBridge
    # (no actuator dynamics -- see wheel_residual_max's comment) can learn to
    # flip the action between -1 and +1 every single step at zero cost there;
    # real Gazebo inertia turns that chatter into actual angular-velocity
    # spikes (~4 rad/s measured, 2026-07-29, vs <0.6 rad/s anywhere in the
    # recorded trajectories) that spin the chassis into a heading breach. This
    # is a hard structural bound on top of (not instead of) w_smooth, which
    # only discourages chatter through the reward and wasn't enough alone.
    # 0.3 caps a full -1->+1 swing at ~7 steps (0.7s).
    action_rate_limit: float = 0.3

    # --- Control / episode ---------------------------------------------
    control_dt: float = 0.1          # [s] one env step (10 Hz)
    max_steps: int = 600             # truncation horizon

    # --- Observation feature toggles -----------------------------------
    # IMPORTANT: these define the obs LAYOUT, which is baked into the policy's
    # input width. They are NOT runtime switches -- a policy must be deployed with
    # the exact toggles it trained with (and held fixed across curriculum phases),
    # or the obs won't match and the corrector fails safe to identity. This
    # dataclass is the single source of truth: train.py and the deployed node both
    # load it, so the defaults here are what keeps train and deploy in lockstep.
    use_prev_action: bool = True     # previous step's clipped action (smoothness)
    use_imu: bool = True             # IMU gyro_z + body accel (slip-observing, on-robot)
    use_wheel_speeds: bool = False   # measured per-wheel speeds from /joint_states
    # PMP costates only exist for recorded planner nominals (Tier-B). The working
    # training path is parametric and has none, so enabling this would feed zeros
    # at train but real costates at deploy -- a silent mismatch. Keep OFF until the
    # recorded-nominal loader exists and training actually supplies them; flip here
    # (one place) when it does.
    use_costates: bool = False

    # --- Observation normalization scales ------------------------------
    pos_err_norm: float = 1.0        # [m]
    rate_norm: float = 2.0           # [m/s] / [rad/s]
    twist_v_norm: float = 0.5        # ~v_max
    twist_w_norm: float = 1.5        # ~omega_max
    imu_gyro_norm: float = 1.5       # [rad/s] yaw rate (~omega_max)
    imu_accel_norm: float = 5.0      # [m/s^2] body linear accel
    costate_norm: float = 1.0

    # --- Reward weights -------------------------------------------------
    # NOTE (SAC_4 post-mortem): the original weights made every trajectory net
    # negative -- progress (w=5 * ~0.03 m/step ~= 0.15/step) was swamped by the
    # quadratic tracking penalties and the -50 terminal, so the policy had no
    # signal pulling it toward the goal and the high-variance terminal spike
    # diverged the critic. w_progress is now the dominant DENSE term (staying on
    # the path keeps earning it; a corridor breach ends the episode and forfeits
    # the rest), and term_penalty is small enough not to dominate the return.
    # Dense POSITIVE on-track reward: each step the robot is near the current
    # target point earns up to w_ontrack, decaying to 0 at the corridor edge (see
    # reward.py). This is the continuous "stay close to the moving target" signal
    # that was missing -- previously every trajectory was net-negative and the only
    # positive term was the rarely-collected success bonus, so SAC had nothing to
    # climb toward (SAC_5/8 reward slid negative). Per-step, so over a ~30-step
    # episode perfect tracking is worth ~+30, comparable to success_bonus.
    w_ontrack: float = 1.0           # dense positive closeness-to-target reward
    w_cross: float = 10.0            # cross-track error (penalty, bites near the edge)
    w_heading: float = 2.0           # heading error
    # Progress is now CAPPED at the nominal's per-step advance in env.step (see
    # the comment there), so it rewards keeping pace with the nominal but cannot
    # be farmed by racing/drifting. Weight kept co-equal with w_cross so tracking
    # accuracy and forward pace matter equally -- w=25 (SAC_5) made progress
    # dominate and the policy drifted off-centre to chase arc-length.
    w_progress: float = 10.0         # along-track progress (anti-stall, capped)
    # Raised 0.1 -> 0.3 (2026-07-30). w_effort is the strength of the IDENTITY
    # PRIOR: the residual should cost something, so the policy only spends
    # authority where the feed-forward is actually wrong. At 0.1 against
    # w_cross=10 / w_progress=10 it was nearly free to wander anywhere inside
    # the corridor, and that is what the deployment comparison showed -- on a
    # straight PMP plan the learned residual drifted monotonically to one side
    # (a steady bias, not chatter) and turned a 0.62 m open-loop run into 4.6 m.
    # 0.3 still loses to the cross-track term whenever the error is real
    # (at |e_cross| = 0.5 m, 10*0.25 = 2.5 dwarfs a full-authority 0.6), so
    # genuine corrections remain worth making.
    w_effort: float = 0.3            # deviation from identity coefficient
    # Raised 0.1 -> 0.5 (2026-07-29): the reward-side discouragement of action
    # chatter, on top of the hard action_rate_limit clamp above -- neither
    # alone was trusted to prevent the KinematicBridge->Gazebo instability
    # (see action_rate_limit's comment), so both apply.
    w_smooth: float = 0.5            # coefficient change between steps
    term_penalty: float = 15.0       # failure terminal penalty
    success_bonus: float = 50.0      # success terminal bonus

    # --- Termination (failure) bounds ----------------------------------
    corridor_epsilon: float = 0.5    # [m] |cross-track| breach
    max_heading_err: float = 1.5708  # [rad] heading breach (pi/2)
    # Whether a corridor/heading breach ENDS the episode, or merely keeps
    # costing w_cross * e_cross^2 while the robot tries to get back.
    #
    # Terminating is what the training distribution has always done, and it is
    # the reason the learned corrector has no recovery behaviour: with
    # start_offset ~0.08 m and an episode that dies at 0.5 m, the policy never
    # once observes a state outside a half-metre tube around a path it started
    # on. Measured 2026-07-30, deployed against real PMP plans with termination
    # off: the policy diverged to 4-6 m on a STRAIGHT trajectory that open-loop
    # identity held to 0.62 m, because everything past 0.5 m is extrapolation
    # for it. TVLQR recovers from a 19 m excursion on the same rig for exactly
    # the opposite reason -- a Riccati law extrapolates by construction.
    #
    # False keeps the episode alive so recovery is inside the training
    # distribution. The quadratic cross-track penalty is unbounded and keeps
    # pointing home, so the gradient outside the corridor is still informative;
    # `on_track` simply clamps to 0 out there. Success/goal logic is unchanged.
    corridor_terminates: bool = True

    # --- Success tolerances (mirror PlannerConfig.goal_tolerance_*) ----
    goal_tolerance_xy: float = 0.10  # [m]
    goal_tolerance_th: float = 0.30  # [rad]
    # Extra steps the episode keeps running once the nominal is EXHAUSTED, holding
    # the final feedforward command, so the corrector can drive the last few cm
    # into goal tolerance before the episode truncates. The nominal is open-loop
    # and progress is capped at its own per-step advance (env.step), so the robot
    # typically ends up a hair short of the goal and the +success_bonus was almost
    # never collected (SAC_6/8: success starved). These grace steps give the
    # corrector -- which still has authority over the held command -- a chance to
    # close that gap and let success actually fire. Success is checked every grace
    # step and fires the instant tolerance is met (so an already-on-goal straight
    # succeeds immediately, no overshoot). 0 -> truncate the moment the path ends
    # (the old behaviour).
    goal_grace_steps: int = 5

    # --- Kinematics (mirror PlannerConfig; keep in sync) ---------------
    wheel_radius: float = 0.08
    track: float = 0.416503
    slip_chi: float = 1.373  # re-identified 2026-07-29, see PlannerConfig's note

    # --- Deployment -----------------------------------------------------
    # Path to the saved SB3 policy (.zip). Empty -> _correct() stays identity.
    policy_path: str = ""

    # ------------------------------------------------------------------
    # Kinematics (same model as PlannerConfig). Plain arithmetic so these
    # broadcast over numpy arrays as well as scalars.
    # ------------------------------------------------------------------
    @property
    def track_effective(self) -> float:
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
        """(w_l, w_r) -> (v, omega)."""
        return self.c_v * (wl + wr), self.c_w * (wr - wl)

    def body_to_wheels(self, v, omega):
        """(v, omega) -> (w_l, w_r). Exact inverse of wheels_to_body."""
        half_sum = v / (2.0 * self.c_v)
        half_diff = omega / (2.0 * self.c_w)
        return half_sum - half_diff, half_sum + half_diff
