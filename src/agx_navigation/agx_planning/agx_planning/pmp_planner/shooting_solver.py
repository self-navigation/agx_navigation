from math import hypot, atan2, pi, sin, cos, tanh
from typing import Optional

import numpy as np
from scipy.integrate import solve_bvp

from agx_planning.pmp_planner.config import PlannerConfig
from agx_planning.pmp_planner.vf_grid import VectorFieldGrid


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
        self._x0: Optional[np.ndarray] = None  # 5-vector now
        self._goal: Optional[np.ndarray] = None
        # Last successful trajectory for introspection / publishing.
        self._last_state: Optional[np.ndarray] = None  # (m, 5)
        self._last_costate: Optional[np.ndarray] = None  # (m, 5)
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
            d_to_goal = np.sqrt((px - self._goal[0]) ** 2 + (py - self._goal[1]) ** 2)
            v_ref = cfg.v_max * np.tanh(d_to_goal / cfg.L_brake)
            half_one_plus = 0.5 * (1.0 + F_dot_h)
            half_one_plus = np.clip(half_one_plus, 0.0, 1.0)
            p_gate = cfg.align_gate_power
            gate = half_one_plus**p_gate
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
        a_unsat = -lv / cfg.gamma_a
        alpha_unsat = -lw / cfg.gamma_alpha
        a = cfg.a_max * np.tanh(a_unsat / cfg.a_max)
        alpha = cfg.alpha_max * np.tanh(alpha_unsat / cfg.alpha_max)

        # State dynamics. Both v and omega are integrators of bounded
        # controls; no first-order chassis-lag modeling on either channel.
        dpx = v * cos_t
        dpy = v * sin_t
        dth = w
        dv = a
        dw = alpha

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
        dlt = (
            -cfg.w_h * cross_F_h * fade
            - cfg.w_h * delta_yaw * (1.0 - fade)
            - cfg.w_v * v_ref * (v - v_ref_eff) * gate_prime * cross_F_h
            - cfg.w_brake * one_minus_dot * v * v * cross_F_h
            + lx * v * sin_t
            - ly * v * cos_t
        )

        # v-costate. No self-coupling -- a is an unrestricted control
        # affecting only dv/dt, so dH/dv has no -lambda_v term. This
        # is the same structure as the original 5D acceleration model.
        dlv = (
            -cfg.w_v * (v - v_ref_eff)
            - cfg.w_brake * one_minus_dot * one_minus_dot * v
            - lx * cos_t
            - ly * sin_t
        )

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
        dlv -= cfg.w_v_barrier * np.sign(v) * v_excess
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
        T_lin = self._T_ref - F_ref_x * (yb[0] - ppx) - F_ref_y * (yb[1] - ppy)

        # Terminal transversalities derived from Phi.
        lam_x_T = -cfg.w_T_terminal * T_lin * F_ref_x + cfg.w_pp * (yb[0] - ppx)
        lam_y_T = -cfg.w_T_terminal * T_lin * F_ref_y + cfg.w_pp * (yb[1] - ppy)
        lam_th_T = cfg.w_th * (yb[2] - thp)
        lam_v_T = cfg.w_v_terminal * yb[3]
        lam_om_T = cfg.w_omega_terminal * yb[4]

        return np.array(
            [
                ya[0] - x0[0],  # p_x(0)
                ya[1] - x0[1],  # p_y(0)
                ya[2] - x0[2],  # theta(0)
                ya[3] - x0[3],  # v(0)
                ya[4] - x0[4],  # omega(0)
                yb[5] - lam_x_T,  # lambda_x(T_h)
                yb[6] - lam_y_T,  # lambda_y(T_h)
                yb[7] - lam_th_T,  # lambda_th(T_h)
                yb[8] - lam_v_T,  # lambda_v(T_h)
                yb[9] - lam_om_T,  # lambda_omega(T_h)
            ]
        )

    # --- Initial guess ----------------------------------------------------

    def _rollout_guess(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        t_mesh: np.ndarray,
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
        px, py, th, v, w = (
            float(x0[0]),
            float(x0[1]),
            float(x0[2]),
            float(x0[3]),
            float(x0[4]),
        )

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
            gate = half_one_plus**cfg.align_gate_power
            v_target = cfg.v_max * tanh(d_goal / cfg.L_brake) * gate
            w_target = max(min(2.0 * e, cfg.omega_max), -cfg.omega_max)

            # Both channels: bounded-acceleration tracking of the heuristic
            # velocity target. Matches the BVP _ode's integrator dynamics.
            a_cmd = max(min(2.0 * (v_target - v), cfg.a_max), -cfg.a_max)
            alpha_cmd = max(min(2.0 * (w_target - w), cfg.alpha_max), -cfg.alpha_max)
            v += a_cmd * dt
            w += alpha_cmd * dt
            px += v * cos(th) * dt
            py += v * sin(th) * dt
            th += w * dt

            state[:, k] = (px, py, th, v, w)

        # Terminal transversalities evaluated at the rolled-out endpoint.
        ppx = float(self._p_pursuit[0])
        ppy = float(self._p_pursuit[1])
        F_ref_x = float(self._F_ref[0])
        F_ref_y = float(self._F_ref[1])
        px_T, py_T, th_T, v_T, w_T = (
            state[0, -1],
            state[1, -1],
            state[2, -1],
            state[3, -1],
            state[4, -1],
        )
        T_lin = self._T_ref - F_ref_x * (px_T - ppx) - F_ref_y * (py_T - ppy)

        lam_x_T = -cfg.w_T_terminal * T_lin * F_ref_x + cfg.w_pp * (px_T - ppx)
        lam_y_T = -cfg.w_T_terminal * T_lin * F_ref_y + cfg.w_pp * (py_T - ppy)
        lam_th_T = cfg.w_th * (th_T - self._theta_pursuit)
        lam_v_T = cfg.w_v_terminal * v_T
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

        lam_om_0 = (
            cfg.gamma_alpha * cfg.alpha_max * cross_F_h_0 * 2.0 * turn_frac
            + lam_om_T * (1.0 - turn_frac)
        )

        s = (t_mesh - t_mesh[0]) / max(t_mesh[-1] - t_mesh[0], 1e-9)
        return np.vstack(
            [
                state[0],
                state[1],
                state[2],
                state[3],
                state[4],
                s * lam_x_T,
                s * lam_y_T,
                (1.0 - s) * lam_th_0 + s * lam_th_T,
                s * lam_v_T,
                (1.0 - s) * lam_om_0 + s * lam_om_T,
            ]
        )

    # --- Solve API --------------------------------------------------------

    def _predistort(
        self,
        v_state: np.ndarray,
        omega_state: np.ndarray,
        lam_v: np.ndarray,
        lam_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        a_star = cfg.a_max * np.tanh(-lam_v / (cfg.gamma_a * cfg.a_max))
        alpha_star = cfg.alpha_max * np.tanh(-lam_w / (cfg.gamma_alpha * cfg.alpha_max))
        v_pre = (v_state + cfg.chassis_tau_v * a_star) / cfg.chassis_gain_v
        w_pre = (
            omega_state + cfg.chassis_tau_omega * alpha_star
        ) / cfg.chassis_gain_omega
        v_cmd = np.clip(v_pre, -cfg.chassis_v_max, +cfg.chassis_v_max)
        w_cmd = np.clip(w_pre, -cfg.chassis_omega_max, +cfg.chassis_omega_max)
        return v_cmd, w_cmd

    def _fallback_command(self) -> Optional[tuple[float, float]]:
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
            v_state=y_eval[3],
            omega_state=y_eval[4],
            lam_v=y_eval[8],
            lam_w=y_eval[9],
        )
        v_cmd, w_cmd = float(v_cmd_a), float(w_cmd_a)
        if abs(v_cmd) < cfg.cmd_deadzone_v:
            v_cmd = 0.0
        if abs(w_cmd) < cfg.cmd_deadzone_omega:
            w_cmd = 0.0
        return v_cmd, w_cmd

    def solve(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
    ) -> Optional[tuple[float, float]]:
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
        xt_active = cfg.w_xt > 0.0
        trace_dist = max(pursuit_dist, cfg.xt_horizon_m if xt_active else 0.0)
        ds_used = float(self.field._res) if hasattr(self.field, "_res") else 0.05

        if self.field.ready and trace_dist > 0.0:
            ref_pts, n_perp = self.field.trace_streamline(
                float(x0[0]),
                float(x0[1]),
                length_m=trace_dist,
                ds=None,
                goal_xy=(float(goal[0]), float(goal[1])),
            )
        else:
            ref_pts = np.zeros((0, 2))
            n_perp = np.zeros((0, 2))

        if ref_pts.shape[0] >= 1:
            n_target = int(np.ceil(pursuit_dist / ds_used))
            n_use = max(1, min(n_target, ref_pts.shape[0]))
            self._p_pursuit = ref_pts[n_use - 1].astype(np.float64)
        else:
            self._p_pursuit = np.array([goal[0], goal[1]], dtype=np.float64)

        # T_ref and F_ref at the pursuit point -- used by both _bc (for
        # the T_lin transversality) and the rollout guess.
        T_pp, _, _, fux_pp, fuy_pp = self.field.query_scalar(
            float(self._p_pursuit[0]),
            float(self._p_pursuit[1]),
        )
        self._T_ref = float(T_pp)

        chassis_to_goal_sq = (float(x0[0]) - goal[0]) ** 2 + (
            float(x0[1]) - goal[1]
        ) ** 2
        chassis_to_goal = float(np.sqrt(chassis_to_goal_sq))
        gt = max(cfg.goal_tolerance_xy, 1e-3)
        s = float(np.clip((chassis_to_goal - gt) / (3.0 * gt), 0.0, 1.0))
        w_F = s * s * (3.0 - 2.0 * s)  # cubic smoothstep, C^1 continuous

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
            self._F_ref = (
                np.array([F_hat_x, F_hat_y])
                if w_F >= 0.5
                else np.array([goal_hat_x, goal_hat_y])
            )
        else:
            self._F_ref = np.array([target_x / target_norm, target_y / target_norm])

        self._align_fade = w_F

        # Unwrap theta_pursuit relative to the current chassis heading.
        theta_now = float(x0[2])
        delta = ((self._theta_pursuit - theta_now + pi) % (2.0 * pi)) - pi
        self._theta_pursuit = theta_now + delta

        if xt_active and ref_pts.shape[0] >= 2:
            self._xt_ref = ref_pts
            self._xt_n_perp = n_perp
            self._xt_sigma = max(cfg.xt_sigma_mult * ds_used, 1e-3)
        else:
            self._xt_ref = np.zeros((0, 2))
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
                self._ode,
                self._bc,
                t_mesh,
                y_init,
                tol=cfg.bvp_tol,
                max_nodes=cfg.bvp_max_nodes,
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
            v_state=y_at_dt[3],
            omega_state=y_at_dt[4],
            lam_v=y_at_dt[8],
            lam_w=y_at_dt[9],
        )
        v_cmd, w_cmd = float(v_cmd_a), float(w_cmd_a)
        if abs(v_cmd) < cfg.cmd_deadzone_v:
            v_cmd = 0.0
        if abs(w_cmd) < cfg.cmd_deadzone_omega:
            w_cmd = 0.0

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
        self,
        y10: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[float, float]:
        """Compute the published twist (v, omega) at a single mesh point
        from a (state, costate) slice y10. Same chassis-model inversion
        as the main publication paths -- see _predistort.
        """
        cfg = self.cfg
        v_cmd_a, w_cmd_a = self._predistort(
            v_state=y10[3],
            omega_state=y10[4],
            lam_v=y10[8],
            lam_w=y10[9],
        )
        v_cmd, w_cmd = float(v_cmd_a), float(w_cmd_a)
        if abs(v_cmd) < cfg.cmd_deadzone_v:
            v_cmd = 0.0
        if abs(w_cmd) < cfg.cmd_deadzone_omega:
            w_cmd = 0.0
        return v_cmd, w_cmd

    # --- Offline-mode segment extraction ----------------------------------

    def sample_committed_segment(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        dt_sample: float,
        n_samples: int,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
            y_ticks = self._prev_sol(t_ticks)  # shape (10, n_samples + 1)
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
        twists[np.abs(twists[:, 0]) < cfg.cmd_deadzone_v, 0] = 0.0
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
