import logging
import time
from math import hypot, atan2, pi, sin, cos, tanh
from typing import Optional

import numpy as np
from scipy.integrate import solve_bvp

_logger = logging.getLogger(__name__)

from agx_planning.pmp_planner.config import PlannerConfig
from agx_planning.vector_field import VectorFieldGrid


class PMPShootingSolver:
    """TPBVP solver for the 5D wheel-space skid-steer PMP problem.

    State (5):   (p_x, p_y, theta, w_l, w_r) -- pose plus left/right
                 wheel-pair angular speeds [rad/s].
    Control (2): (a_l, a_r) -- wheel angular accelerations [rad/s^2].

    Body velocities are derived states (cfg.wheels_to_body):
        v     = c_v * (w_l + w_r),    c_v = wheel_radius / 2
        omega = c_w * (w_r - w_l),    c_w = wheel_radius / track_effective
    with track_effective = track * slip_chi absorbing the lateral-skid
    yaw deficit that used to live in chassis_gain_omega.

    This is a linear reparametrization of the previous unicycle model:
    with (v, omega) = A (w_l, w_r) the costates transform covariantly,
    lambda_w = A^T lambda_(v,omega), and the costate ODEs below are the
    exact images of the old ones. What genuinely changes is where the
    actuation constraints live: effort and acceleration bounds are
    per-wheel, so linear and angular authority share one budget, and a
    per-wheel speed barrier matches the joint velocity limit.

    The published command is the planner's PREDICTED wheel-speed state
    at each tick -- a velocity setpoint that already respects the
    acceleration bounds, so JointGroupVelocityController has nothing to
    fight. Online mode derives (w_l, w_r) initial conditions from the
    measured /odom twist via the model's own inverse kinematics (NOT
    from raw /joint_states velocities, whose implied yaw rate goes
    through the physical track and would contradict track_effective);
    offline mode propagates simulated wheel speeds across segments via
    the BVP's own state evolution.

    Same solver in both modes. Online mode calls solve() per tick and
    publishes the state at the next control tick; offline mode calls
    solve() once per rollout segment and densely samples the BVP solution.
    """

    def __init__(self, cfg: PlannerConfig, field: VectorFieldGrid):
        self.cfg = cfg
        self.field = field
        self._prev_sol = None
        # Cached per-solve for the BC/ODE closures.
        self._x0: Optional[np.ndarray] = None  # 5-vector
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
        # Duration of the last committed segment [s]. Used to time-shift the
        # offline warm start: evaluating prev_sol at [0, T_h] instead of
        # [seg_T, T_h] feeds the *start* of the previous plan rather than
        # its tail -- a phase error in the costate waveform that
        # systematically kills the turn.
        self._last_seg_T: float = 0.0

    # --- Augmented dynamics ------------------------------------------------

    def _ode(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """RHS of the (state, costate) ODE, vectorized over the BVP mesh.

        y has shape (10, m) where rows are
            [p_x, p_y, theta, w_l, w_r, lambda_x, lambda_y, lambda_th,
             lambda_wl, lambda_wr].

        The wheel costate ODEs are assembled from the body-channel
        Hamiltonian gradients via the chain rule:
            dv/dw_l = dv/dw_r = c_v;   domega/dw_l = -c_w, domega/dw_r = +c_w
            d(lam_wl)/dt = -c_v * H_v + c_w * H_om - (wheel barrier grad)
            d(lam_wr)/dt = -c_v * H_v - c_w * H_om - (wheel barrier grad)
        which is exactly the A^T image of the old (lam_v, lam_om) ODEs,
        so every body-space cost term carries over verbatim inside
        H_v / H_om.
        """
        cfg = self.cfg
        c_v = cfg.c_v
        c_w = cfg.c_w
        px, py, th, wl, wr = y[0], y[1], y[2], y[3], y[4]
        lx, ly, lt, lwl, lwr = y[5], y[6], y[7], y[8], y[9]
        cos_t = np.cos(th)
        sin_t = np.sin(th)

        # Derived body velocities -- every body-space cost term below
        # sees these instead of independent states.
        v = c_v * (wl + wr)
        w = c_w * (wr - wl)

        T_now, dT_dx, dT_dy, Fux, Fuy = self.field.query_vec(px, py)
        F_dot_h = Fux * cos_t + Fuy * sin_t
        cross_F_h = Fux * sin_t - Fuy * cos_t

        # Alignment gate, needed unconditionally: pos_gate uses it even
        # when the speed-reference block (w_v == 0) is disabled. (The
        # old code left half_one_plus undefined in that branch.)
        half_one_plus = np.clip(0.5 * (1.0 + F_dot_h), 0.0, 1.0)
        pos_gate = half_one_plus**cfg.align_gate_power

        # Speed-reference scaffolding (body space, unchanged).
        v_ref = np.zeros_like(px)
        v_ref_eff = np.zeros_like(px)
        gate_prime = np.zeros_like(px)
        if cfg.w_v > 0.0:
            d_to_goal = np.sqrt((px - self._goal[0]) ** 2 + (py - self._goal[1]) ** 2)
            v_ref = cfg.v_max * np.tanh(d_to_goal / cfg.L_brake)
            p_gate = cfg.align_gate_power
            if p_gate > 0.0:
                gate_prime = (0.5 * p_gate) * (half_one_plus ** (p_gate - 1.0))
            v_ref_eff = v_ref * pos_gate

        # Closed-form optimal controls, per wheel:
        #   a_i* = -lambda_wi / gamma_wheel   (sat |a_i| <= a_wheel_max)
        # tanh with K=1 rather than a hard clip keeps the ODE smooth for
        # solve_bvp's collocation; a sharper K meshes the near-
        # discontinuity to death (see the unicycle version's history).
        # The cold-start costate pin in _rollout_guess places the
        # antisymmetric component at 2x the saturation threshold
        # gamma_wheel * a_wheel_max, so tanh(2) = 0.96 -- 4 % error at
        # the operating point, same design margin as before.
        sat = cfg.gamma_wheel * cfg.a_wheel_max
        al = cfg.a_wheel_max * np.tanh(-lwl / sat)
        ar = cfg.a_wheel_max * np.tanh(-lwr / sat)

        # State dynamics. Both wheel speeds are integrators of bounded
        # controls; no first-order driver lag on either channel.
        dpx = v * cos_t
        dpy = v * sin_t
        dth = w
        dwl = al
        dwr = ar

        # Position costates -- only L_pos contributes under the
        # frozen-field approximation.
        T_clip = np.minimum(T_now, cfg.T_horizon)
        dlx = -cfg.beta * T_clip * dT_dx / cfg.T_horizon
        dly = -cfg.beta * T_clip * dT_dy / cfg.T_horizon

        # Cross-track residual (opt-in). Adds -w_xt * r_xt * n_perp to
        # (dlx, dly) with a soft Gaussian projection onto the streamline
        # reference.
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

        # Heading costate -- identical in form to the unicycle version;
        # v is now the derived quantity. Alignment is faded by w_F;
        # speed and brake contributions vanish naturally near the goal
        # via v_ref -> 0 and v -> 0.
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

        # Body-channel Hamiltonian gradients. H_v gates the position-
        # costate contribution by heading alignment: without this the
        # position costates (O(beta*T)) overwhelm the quadratic
        # brake/speed costs instantly, driving forward acceleration at
        # any heading error -- the root cause of consistent
        # understeering. The gate matches v_ref_eff: no position pull
        # when the heading gate is closed, full pull when aligned.
        H_v = (
            cfg.w_v * (v - v_ref_eff)
            + cfg.w_brake * one_minus_dot * one_minus_dot * v
            + (lx * cos_t + ly * sin_t) * pos_gate
            + cfg.w_v_barrier * np.sign(v) * np.maximum(0.0, np.abs(v) - cfg.v_max)
        )
        H_om = (
            cfg.w_omega_run * w
            + lt
            + cfg.w_omega_barrier * np.sign(w) * np.maximum(0.0, np.abs(w) - cfg.omega_max)
        )

        # Wheel costates: chain-rule recombination of the body gradients
        # plus the per-wheel speed barrier (the hardware net at the
        # joint velocity limit).
        wl_excess = np.maximum(0.0, np.abs(wl) - cfg.w_wheel_max)
        wr_excess = np.maximum(0.0, np.abs(wr) - cfg.w_wheel_max)
        dlwl = -c_v * H_v + c_w * H_om - cfg.w_wheel_barrier * np.sign(wl) * wl_excess
        dlwr = -c_v * H_v - c_w * H_om - cfg.w_wheel_barrier * np.sign(wr) * wr_excess

        return np.vstack([dpx, dpy, dth, dwl, dwr, dlx, dly, dlt, dlwl, dlwr])

    # --- Boundary conditions ----------------------------------------------

    def _bc(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """Ten BC residuals; solve_bvp drives these to zero.

        Initial state (5): pose from TF and wheel speeds (derived from
        the measured /odom twist, or propagated offline) are pinned.
        Terminal (5): transversalities from Phi(x_T):
            lambda_x(T)  = -w_T_terminal * T_lin * F_ref_x
                           + w_pp * (p_x_T - p_x_pursuit)
            lambda_y(T)  = -w_T_terminal * T_lin * F_ref_y
                           + w_pp * (p_y_T - p_y_pursuit)
            lambda_th(T) = w_th * (theta_T - theta_pursuit)
            lambda_wl(T) = c_v * w_v_terminal * v_T - c_w * w_omega_terminal * omega_T
            lambda_wr(T) = c_v * w_v_terminal * v_T + c_w * w_omega_terminal * omega_T
        The terminal stop costs stay quadratic in the DERIVED body
        velocities, so "stop translating" and "stop rotating" remain
        independently tunable; the wheel rows are their chain-rule
        images. T_lin = T_ref - F_ref . (p_T - p_pursuit) is the
        linearization of the T-field around the pursuit point.
        """
        cfg = self.cfg
        x0 = self._x0
        ppx = float(self._p_pursuit[0])
        ppy = float(self._p_pursuit[1])
        thp = self._theta_pursuit
        F_ref_x = float(self._F_ref[0])
        F_ref_y = float(self._F_ref[1])
        T_lin = self._T_ref - F_ref_x * (yb[0] - ppx) - F_ref_y * (yb[1] - ppy)

        v_T, om_T = cfg.wheels_to_body(yb[3], yb[4])

        # Terminal transversalities derived from Phi.
        lam_x_T = -cfg.w_T_terminal * T_lin * F_ref_x + cfg.w_pp * (yb[0] - ppx)
        lam_y_T = -cfg.w_T_terminal * T_lin * F_ref_y + cfg.w_pp * (yb[1] - ppy)
        lam_th_T = cfg.w_th * (yb[2] - thp)
        lam_wl_T = cfg.c_v * cfg.w_v_terminal * v_T - cfg.c_w * cfg.w_omega_terminal * om_T
        lam_wr_T = cfg.c_v * cfg.w_v_terminal * v_T + cfg.c_w * cfg.w_omega_terminal * om_T

        return np.array(
            [
                ya[0] - x0[0],  # p_x(0)
                ya[1] - x0[1],  # p_y(0)
                ya[2] - x0[2],  # theta(0)
                ya[3] - x0[3],  # w_l(0)
                ya[4] - x0[4],  # w_r(0)
                yb[5] - lam_x_T,  # lambda_x(T_h)
                yb[6] - lam_y_T,  # lambda_y(T_h)
                yb[7] - lam_th_T,  # lambda_th(T_h)
                yb[8] - lam_wl_T,  # lambda_wl(T_h)
                yb[9] - lam_wr_T,  # lambda_wr(T_h)
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
        direction as fallbacks), maps heuristic (v, omega) targets to
        per-wheel speed targets, tracks them with bounded-acceleration
        P controllers, and integrates state under the same integrator
        dynamics as the BVP _ode.

        Costates ramp linearly toward the terminal transversality
        evaluated at the rollout endpoint -- small warm-start magnitudes
        are what Newton prefers -- except for the heading/wheel rows,
        which get a physics-based cold-start pin (below).
        """
        cfg = self.cfg
        m = t_mesh.size
        dt = t_mesh[1] - t_mesh[0] if m > 1 else cfg.T_horizon

        state = np.zeros((5, m))
        state[:, 0] = x0
        px, py, th, wl, wr = (
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

            # Map the body targets to wheel targets and track each wheel
            # with a bounded-acceleration P controller -- matches the
            # BVP _ode's per-wheel integrator dynamics.
            wl_t, wr_t = cfg.body_to_wheels(v_target, w_target)
            al_cmd = max(min(2.0 * (wl_t - wl), cfg.a_wheel_max), -cfg.a_wheel_max)
            ar_cmd = max(min(2.0 * (wr_t - wr), cfg.a_wheel_max), -cfg.a_wheel_max)
            wl += al_cmd * dt
            wr += ar_cmd * dt
            v_now = cfg.c_v * (wl + wr)
            w_now = cfg.c_w * (wr - wl)
            px += v_now * cos(th) * dt
            py += v_now * sin(th) * dt
            th += w_now * dt

            state[:, k] = (px, py, th, wl, wr)

        # Terminal transversalities evaluated at the rolled-out endpoint.
        ppx = float(self._p_pursuit[0])
        ppy = float(self._p_pursuit[1])
        F_ref_x = float(self._F_ref[0])
        F_ref_y = float(self._F_ref[1])
        px_T, py_T, th_T, wl_T, wr_T = (
            state[0, -1],
            state[1, -1],
            state[2, -1],
            state[3, -1],
            state[4, -1],
        )
        T_lin = self._T_ref - F_ref_x * (px_T - ppx) - F_ref_y * (py_T - ppy)
        v_T, om_T = cfg.wheels_to_body(wl_T, wr_T)

        lam_x_T = -cfg.w_T_terminal * T_lin * F_ref_x + cfg.w_pp * (px_T - ppx)
        lam_y_T = -cfg.w_T_terminal * T_lin * F_ref_y + cfg.w_pp * (py_T - ppy)
        lam_th_T = cfg.w_th * (th_T - self._theta_pursuit)
        lam_wl_T = cfg.c_v * cfg.w_v_terminal * v_T - cfg.c_w * cfg.w_omega_terminal * om_T
        lam_wr_T = cfg.c_v * cfg.w_v_terminal * v_T + cfg.c_w * cfg.w_omega_terminal * om_T

        # Physical cold-start estimate for the initial costates.
        #
        # A naive ramp (s * terminal_value) leaves the ANTISYMMETRIC
        # wheel-costate component ~= 0 throughout when the endpoint is a
        # turn-to-stop, giving zero differential acceleration from the
        # start -- inconsistent with the state trajectory, and a trap
        # for Newton in a low-rotation local minimum on cold starts.
        #
        # The heading-costate estimate is unchanged from the unicycle
        # version (theta is the same coordinate):
        #   d(lam_th)/dt ~= -w_h * cross_F_h    =>
        #   lam_th(0) ~= w_h * cross_F_h_0 * T_turn / 2
        #
        # The angular channel now lives in the wheel-costate DIFFERENCE.
        # Through the covector transform lam_w = A^T lam_(v,om), the old
        # "pin lam_om just past the saturation threshold" becomes: pin
        # the antisymmetric component at 2x the per-wheel threshold
        # gamma_wheel * a_wheel_max, signs such that the outer wheel
        # accelerates and the inner decelerates.
        #
        # Sign check for a CCW turn (cross_F_h_0 = -1):
        #   d0 = gamma_wheel * a_wheel_max * (-1) * 2 < 0
        #   lam_wr(0) = +d0 < 0  =>  a_r* = -lam_wr/gamma > 0  [outer up, ok]
        #   lam_wl(0) = -d0 > 0  =>  a_l* < 0                  [inner down, ok]
        th_0 = float(x0[2])
        _, _, _, fux_0, fuy_0 = self.field.query_scalar(float(x0[0]), float(x0[1]))
        cross_F_h_0 = fux_0 * sin(th_0) - fuy_0 * cos(th_0)

        # Fraction of T_horizon we expect to be in the "active turn" phase;
        # clamp to [0, 1] so the estimate is sensible near the goal.
        heading_err_0 = ((self._theta_pursuit - th_0 + pi) % (2.0 * pi)) - pi
        turn_frac = min(1.0, abs(heading_err_0) / (pi / 4.0))

        lam_th_0 = cfg.w_h * cross_F_h_0 * cfg.T_horizon * 0.5 * turn_frac

        d0 = cfg.gamma_wheel * cfg.a_wheel_max * cross_F_h_0 * 2.0 * turn_frac
        lam_wl_0 = -d0 + lam_wl_T * (1.0 - turn_frac)
        lam_wr_0 = +d0 + lam_wr_T * (1.0 - turn_frac)

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
                (1.0 - s) * lam_wl_0 + s * lam_wl_T,
                (1.0 - s) * lam_wr_0 + s * lam_wr_T,
            ]
        )

    # --- Solve API --------------------------------------------------------

    def _predistort(
        self,
        wl_state: np.ndarray,
        wr_state: np.ndarray,
        lam_wl: np.ndarray,
        lam_wr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Turn BVP-planned wheel states into publishable wheel commands.

        Three steps, vectorized over array inputs (scalars broadcast):
          1. Optional first-order lead for a wheel-velocity tracking lag:
                 w_cmd_i = w_state_i + tau_wheel * a_i*
             with a_i* recovered from the costates by the same tanh-
             saturated law as _ode. tau_wheel = 0 (default, exact for
             the gz velocity interface) makes this a passthrough. The
             static slip gain that the old chassis inversion carried is
             gone: it lives in track_effective inside the kinematics.
          2. Body-space deadzone, applied to the (v, omega) RECONSTRUCTED
             from the commands and mapped back: turn-in-place keeps its
             wheel difference while a near-zero symmetric part is
             flushed to exactly zero, matching the old semantics. The
             deadzone moved here from the call sites because the
             body-space reconstruction is needed anyway.
          3. Clip to wheel_cmd_max (the joint velocity limit;
             gz_ros2_control would clip there regardless).
        """
        cfg = self.cfg
        sat = cfg.gamma_wheel * cfg.a_wheel_max
        al_star = cfg.a_wheel_max * np.tanh(-lam_wl / sat)
        ar_star = cfg.a_wheel_max * np.tanh(-lam_wr / sat)
        wl_pre = wl_state + cfg.tau_wheel * al_star
        wr_pre = wr_state + cfg.tau_wheel * ar_star

        v_pre, om_pre = cfg.wheels_to_body(wl_pre, wr_pre)
        v_pre = np.where(np.abs(v_pre) < cfg.cmd_deadzone_v, 0.0, v_pre)
        om_pre = np.where(np.abs(om_pre) < cfg.cmd_deadzone_omega, 0.0, om_pre)
        wl_cmd, wr_cmd = cfg.body_to_wheels(v_pre, om_pre)

        wl_cmd = np.clip(wl_cmd, -cfg.wheel_cmd_max, +cfg.wheel_cmd_max)
        wr_cmd = np.clip(wr_cmd, -cfg.wheel_cmd_max, +cfg.wheel_cmd_max)
        return wl_cmd, wr_cmd

    def _fallback_command(self) -> Optional[tuple[float, float]]:
        """Evaluate the fallback wheel command from the previous BVP
        solution at the next control tick. Applies the same publication
        transform as the main path. The prev_sol callable is freed on
        failure to prevent a second fallback attempt with a
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
        wl_cmd_a, wr_cmd_a = self._predistort(
            wl_state=y_eval[3],
            wr_state=y_eval[4],
            lam_wl=y_eval[8],
            lam_wr=y_eval[9],
        )
        return float(wl_cmd_a), float(wr_cmd_a)

    def solve(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
    ) -> Optional[tuple[float, float]]:
        """Solve the TPBVP. x0 is the 5-vector (px, py, theta, w_l, w_r).

        Returns (wl_cmd, wr_cmd) -- the publication-ready wheel-group
        velocity setpoints [rad/s] (post lead/deadzone/clip, see
        _predistort) -- or None on failure. The caller duplicates each
        per side for the four-joint controller message.

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
                # planner's best prior belief about the new time window.
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

        warm_start = self._prev_sol is not None
        _t_bvp = time.perf_counter()
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
            _logger.info("BVP EXCEPTION warm=%s dt=%.0fms: %s", warm_start,
                         (time.perf_counter() - _t_bvp) * 1e3, e)
            return self._fallback_command()

        _bvp_ms = (time.perf_counter() - _t_bvp) * 1e3
        _logger.info(
            "BVP %s  warm=%s  nodes=%d  niter=%d  dt=%.0fms  msg=%s",
            "ok  " if sol.success else "FAIL",
            warm_start,
            sol.x.size,
            getattr(sol, "niter", -1),
            _bvp_ms,
            sol.message if not sol.success else "",
        )

        if not sol.success:
            self._last_error = sol.message
            return self._fallback_command()

        self._prev_sol = sol.sol
        # For the next warm start: online commits one control tick.
        self._last_seg_T = min(1.0 / cfg.control_rate, cfg.T_horizon)

        # Online publication: the wheel-speed state at the next control
        # tick, passed through the publication transform (lead +
        # deadzone + clip). The chassis ramps toward where the plan says
        # it should be one tick later.
        t_lookahead = min(1.0 / cfg.control_rate, cfg.T_horizon)
        y_at_dt = self._prev_sol(t_lookahead)
        wl_cmd_a, wr_cmd_a = self._predistort(
            wl_state=y_at_dt[3],
            wr_state=y_at_dt[4],
            lam_wl=y_at_dt[8],
            lam_wr=y_at_dt[9],
        )

        # Densely resample for visualization and downstream introspection.
        t_dense = np.linspace(0.0, cfg.T_horizon, cfg.N + 1)
        y_dense = self._prev_sol(t_dense)
        self._last_state = y_dense[0:5].T
        self._last_costate = y_dense[5:10].T
        self._last_error = None

        return float(wl_cmd_a), float(wr_cmd_a)

    def reset_warm_start(self):
        self._prev_sol = None

    # --- Pointwise readout (shared by online + offline paths) -------------

    def _control_law_pointwise(
        self,
        y10: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[float, float]:
        """Compute the published wheel command (wl, wr) at a single mesh
        point from a (state, costate) slice y10. Same publication
        transform as the main paths -- see _predistort.
        """
        wl_cmd_a, wr_cmd_a = self._predistort(
            wl_state=y10[3],
            wr_state=y10[4],
            lam_wl=y10[8],
            lam_wr=y10[9],
        )
        return float(wl_cmd_a), float(wr_cmd_a)

    # --- Offline-mode segment extraction ----------------------------------

    def sample_committed_segment(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        dt_sample: float,
        n_samples: int,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Solve the BVP from x0 toward goal, then densely sample the
        first n_samples * dt_sample seconds of the optimal trajectory.

        Returns (wheel_cmds, accels, poses, costates, x_next) where:
          wheel_cmds : (n_samples, 2) [wl_cmd, wr_cmd] -- publication-
                       ready wheel-group velocity setpoints at each tick
                       (post lead/deadzone/clip, see _predistort).
          accels     : (n_samples, 2) [a_l*, a_r*] -- the BVP-optimal
                       wheel accelerations (pre-lead planned controls),
                       for feedforward / neighboring-optimal use.
          poses      : (n_samples, 3) [px, py, theta] -- BVP-planned
                       chassis pose at each tick, parallel to wheel_cmds.
          costates   : (n_samples, 5) [lx, ly, lth, lwl, lwr] -- the PMP
                       costates along the nominal at each tick. lambda
                       is the gradient of the horizon cost-to-go; it
                       cannot be reconstructed downstream (it depends on
                       per-solve pursuit references and the field
                       snapshot), so it is exported here or never.
          x_next     : (5,) BVP state at t = n_samples * dt_sample, the
                       start of the next segment. Assumes the chassis
                       tracks the setpoints so its actual state matches
                       the BVP's -- the open-loop error budget of
                       offline mode, and the residual the downstream
                       corrector exists to absorb.

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

        wl_cmd_v, wr_cmd_v = self._predistort(
            wl_state=y_ticks[3, :n_samples],
            wr_state=y_ticks[4, :n_samples],
            lam_wl=y_ticks[8, :n_samples],
            lam_wr=y_ticks[9, :n_samples],
        )
        wheel_cmds = np.column_stack([wl_cmd_v, wr_cmd_v])

        sat = cfg.gamma_wheel * cfg.a_wheel_max
        accels = np.column_stack(
            [
                cfg.a_wheel_max * np.tanh(-y_ticks[8, :n_samples] / sat),
                cfg.a_wheel_max * np.tanh(-y_ticks[9, :n_samples] / sat),
            ]
        )

        poses = y_ticks[0:3, :n_samples].T.copy()
        costates = y_ticks[5:10, :n_samples].T.copy()
        x_next = y_ticks[0:5, n_samples].copy()

        # Record committed duration so solve() can time-shift the warm start.
        self._last_seg_T = seg_T

        return wheel_cmds, accels, poses, costates, x_next
