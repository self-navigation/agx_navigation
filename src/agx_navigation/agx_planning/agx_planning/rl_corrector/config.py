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
    # 4 = per-wheel coefficients [fl, rl, fr, rr]; 2 = per-side [left, right].
    action_dim: int = 4
    # coeff_i = 1 + coeff_k * a_i, with a_i in [-1, 1]. a_i = 0 -> coeff 1 -> exact
    # identity (the fail-safe contract; also where SAC's zero-centred tanh policy
    # naturally sits). coeff_k is the ACTION AUTHORITY dial: k=0.2 -> coeff in
    # [0.8, 1.2]. Dropped from 0.5 ([0.5,1.5]) because that let the policy command
    # such asymmetric wheels that it over-rotated and breached the pi/2 heading
    # corridor on turns (SAC_6/8: heading breaches at |e_heading|~1.6-2.0 rad). A
    # residual corrector should nudge the feedforward, not overpower it. SINGLE
    # SOURCE OF TRUTH for train+deploy -- the deployed node reads this same value,
    # so a policy must be deployed with the coeff_k it trained on.
    coeff_k: float = 0.2
    # Per-wheel command clamp [rad/s]; matches the joint <limit velocity>.
    wheel_cmd_max: float = 20.0

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
    use_prev_coeff: bool = True      # previous step's coefficients (smoothness)
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
    w_effort: float = 0.1            # deviation from identity coefficient
    w_smooth: float = 0.1            # coefficient change between steps
    term_penalty: float = 15.0       # failure terminal penalty
    success_bonus: float = 50.0      # success terminal bonus

    # --- Termination (failure) bounds ----------------------------------
    corridor_epsilon: float = 0.5    # [m] |cross-track| breach
    max_heading_err: float = 1.5708  # [rad] heading breach (pi/2)

    # --- Success tolerances (mirror PlannerConfig.goal_tolerance_*) ----
    goal_tolerance_xy: float = 0.10  # [m]
    goal_tolerance_th: float = 0.30  # [rad]

    # --- Kinematics (mirror PlannerConfig; keep in sync) ---------------
    wheel_radius: float = 0.08
    track: float = 0.416503
    slip_chi: float = 1.2987

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
