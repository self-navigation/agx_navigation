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
    # coeff_i = 1 + coeff_k * a_i, with a_i in [-1, 1]. k=0.5 -> coeff in [0.5, 1.5].
    # a_i = 0 -> coeff 1 -> exact identity (the fail-safe contract).
    coeff_k: float = 0.5
    # Per-wheel command clamp [rad/s]; matches the joint <limit velocity>.
    wheel_cmd_max: float = 20.0

    # --- Control / episode ---------------------------------------------
    control_dt: float = 0.1          # [s] one env step (10 Hz)
    max_steps: int = 600             # truncation horizon

    # --- Observation feature toggles -----------------------------------
    use_prev_coeff: bool = True      # previous step's coefficients (smoothness)
    use_wheel_speeds: bool = False   # measured per-wheel speeds from /joint_states
    use_costates: bool = True        # PMP costates (recorded-nominal training only)

    # --- Observation normalization scales ------------------------------
    pos_err_norm: float = 1.0        # [m]
    rate_norm: float = 2.0           # [m/s] / [rad/s]
    twist_v_norm: float = 0.5        # ~v_max
    twist_w_norm: float = 1.5        # ~omega_max
    costate_norm: float = 1.0

    # --- Reward weights -------------------------------------------------
    w_cross: float = 10.0            # cross-track error (dominant)
    w_heading: float = 2.0           # heading error
    w_progress: float = 5.0          # along-track progress (anti-stall)
    w_effort: float = 0.1            # deviation from identity coefficient
    w_smooth: float = 0.1            # coefficient change between steps
    term_penalty: float = 50.0       # failure terminal penalty
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
