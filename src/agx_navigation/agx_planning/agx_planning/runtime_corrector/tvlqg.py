"""Surface-estimating Kalman filter for the TVLQG corrector. Pure (numpy only).

An EKF on x = [p_x, p_y, theta, omega, k, rho, b_g]: omega follows
(1 + rho) omega_cmd with first-order lag yaw_tau, the body slides sideways by
v_lat = k v omega, and k, rho, b_g are random walks. v is an input (commanded
speed or odometer). Pose and gyro (omega + b_g) updates are applied as they
arrive. covariance_along_plan evaluates the covariance along a plan before the
drive. Updates whose NIS exceeds the gate_p bound still correct the pose but
not k, rho or b_g, and are flagged as gated. Never feed the chassis /odom
omega as the gyro (it echoes the command), and set pose noise to the real
localizer's values even when the sim provides ground truth.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# chi^2 upper quantiles by tail probability and dof (no scipy in pure modules).
CHI2_GATE = {
    0.01: {1: 6.635, 3: 11.345},
    0.001: {1: 10.828, 3: 16.266},
}

# State layout. The surface/sensor parameters come last, contiguous from IK:
# the innovation gate freezes rows IK: of the gain.
IX, IY, ITH, IW, IK, IRHO, IBG = range(7)
N_S = 7


def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class TVLQGConfig:
    """Noise model and priors of the surface filter.

    Densities are continuous-time (per sqrt(s)); discrete variances are formed
    as density^2 * dt, so the filter is invariant to the update rate.
    """

    # --- Priors -------------------------------------------------------------
    # The active surface's slide constant. Gazebo's plant is -0.33 s (job 237),
    # the real wood corridor -0.03 s (2026-10-09).
    k0: float = 0.0
    sigma_k0: float = 0.3          # [s] wide enough to cover both surfaces
    rho0: float = 0.0
    sigma_rho0: float = 0.2
    sigma_w0: float = 0.05         # [rad/s] initial yaw rate (robot at rest)
    bg0: float = 0.0
    sigma_bg0: float = 0.01        # [rad/s]
    sigma_pose0: float = 0.05      # [m]
    sigma_theta0: float = 0.05     # [rad]

    # --- Process noise densities -------------------------------------------
    # FITTED 2026-10-09 on the job-237 LTN Gazebo replay (34 runs, synthetic
    # measurement noise = the r_* below, so only these are free): pose NEES
    # against truth mean 2.4 (dof 3, slightly conservative), rho_hat +0.015
    # (regression ~+-0.02), k_hat -0.36 +- 0.05 (regression -0.33). A smaller
    # q_omega (0.003) matches the gyro NIS median but gates 12% of ordinary
    # driving, which would make "sustained gating" a useless deviation signal;
    # 0.03 or more lets wall contacts through the gate and drags rho_hat to
    # -0.47. See handover 2026-10-09 evening.
    # Input noise: how well the chassis realizes (v, omega_cmd).
    q_v: float = 0.03              # [m/s / sqrt(Hz)]
    q_omega: float = 0.01          # [rad/s / sqrt(Hz)]
    # Unmodelled change of the actual yaw rate itself (YAW RESPONSE).
    q_w: float = 0.03              # [rad/s / sqrt(s)]
    # Unmodelled motion acting on the pose directly (contact, discretization).
    # Without it every model mismatch is absorbed into k_hat / rho_hat: at
    # 0.01, k_hat came out -0.40 against the regression's -0.33.
    q_pos: float = 0.03            # [m / sqrt(s)]
    q_theta: float = 0.02          # [rad / sqrt(s)]
    # Parameter random walks. q_k SETS the switch threshold's width on
    # straights -- see the module docstring before changing it.
    q_k: float = 0.01              # [s / sqrt(s)]
    q_rho: float = 0.002           # [1 / sqrt(s)]
    q_bg: float = 1e-5             # [rad/s / sqrt(s)]; real gyro: no drift to 126 s

    # --- Chassis yaw response (module docstring, YAW RESPONSE) -------------
    # First-order time constant from omega_cmd to the actual yaw rate. 0 =
    # instant (Gazebo's velocity controller). Real Scout Mini on wood,
    # 2026-10-09 slalom2: 0.27 s, no dead time (best 0.02 s), gain 1.07.
    yaw_tau: float = 0.0           # [s]

    # --- Measurement noise (standard deviations per sample) ----------------
    # Pose: the real localizer's, NOT ground truth's (module docstring).
    r_pos: float = 0.05            # [m] amcl-level
    r_theta: float = 0.035         # [rad] ~2 deg
    # Gyro: 7.0e-5 rad/s/sqrt(Hz) at rest on the real robot -> 7e-4 rad/s at
    # 100 Hz. Floored well above that: vibration while driving is unmeasured.
    r_gyro: float = 0.01           # [rad/s]

    # --- Innovation gate ------------------------------------------------------
    # Tail probability of the chi^2 bound beyond which an update may not move
    # k / rho / b_g (module docstring, INNOVATION GATE). A key of CHI2_GATE.
    gate_p: float = 0.001

    def p0(self) -> np.ndarray:
        return np.diag([
            self.sigma_pose0, self.sigma_pose0, self.sigma_theta0, self.sigma_w0,
            self.sigma_k0, self.sigma_rho0, self.sigma_bg0,
        ]) ** 2

    def x0(self, pose: Sequence[float]) -> np.ndarray:
        return np.array([pose[0], pose[1], pose[2], 0.0, self.k0, self.rho0, self.bg0],
                        dtype=float)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


def _yaw_step(w0: float, u: float, dt: float, tau: float) -> Tuple[float, float]:
    """First-order yaw response over one interval with the target rate u held:
    (rate at the end, angle turned). tau = 0 is the instant response."""
    if tau <= 0.0:
        return u, u * dt
    a = np.exp(-dt / tau)
    return a * w0 + (1.0 - a) * u, u * dt + (w0 - u) * tau * (1.0 - a)


def propagate(x: np.ndarray, v: float, omega_cmd: float, dt: float,
              tau: float = 0.0) -> np.ndarray:
    """One step of the nonlinear model, heading at the interval midpoint."""
    th, w0, k, rho = x[ITH], x[IW], x[IK], x[IRHO]
    w1, dth = _yaw_step(w0, (1.0 + rho) * omega_cmd, dt, tau)
    thm = th + 0.5 * dth
    vl = k * v * dth / dt          # slide on the interval's mean actual rate
    out = np.array(x, dtype=float)
    out[IX] += (v * np.cos(thm) - vl * np.sin(thm)) * dt
    out[IY] += (v * np.sin(thm) + vl * np.cos(thm)) * dt
    out[ITH] = wrap_to_pi(th + dth)
    out[IW] = w1
    return out


def _diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    d[ITH] = wrap_to_pi(d[ITH])
    return d


def jacobians(x: np.ndarray, v: float, omega_cmd: float, dt: float,
              tau: float = 0.0, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """(F, G): d propagate / d state, and d propagate / d (v, omega_cmd).

    Central differences: 7 states, so 18 model evaluations per step, which is
    nothing at the corrector's rate, and the Jacobian cannot drift out of sync
    with ``propagate`` when the model changes."""
    F = np.empty((N_S, N_S))
    for j in range(N_S):
        d = np.zeros(N_S)
        d[j] = eps
        F[:, j] = _diff(propagate(x + d, v, omega_cmd, dt, tau),
                        propagate(x - d, v, omega_cmd, dt, tau)) / (2 * eps)
    G = np.empty((N_S, 2))
    for j, (dv, dw) in enumerate(((eps, 0.0), (0.0, eps))):
        G[:, j] = _diff(propagate(x, v + dv, omega_cmd + dw, dt, tau),
                        propagate(x, v - dv, omega_cmd - dw, dt, tau)) / (2 * eps)
    return F, G


def process_noise(x: np.ndarray, v: float, omega_cmd: float, dt: float,
                  cfg: TVLQGConfig, G: Optional[np.ndarray] = None) -> np.ndarray:
    if G is None:
        _, G = jacobians(x, v, omega_cmd, dt, cfg.yaw_tau)
    # Input noise enters over the interval: variance density^2 / dt per
    # sample, mapped through G (which carries dt), i.e. ~density^2 * dt on p.
    Qu = np.diag([cfg.q_v, cfg.q_omega]) ** 2 / dt
    Q = G @ Qu @ G.T
    Q += np.diag([cfg.q_pos ** 2, cfg.q_pos ** 2, cfg.q_theta ** 2, cfg.q_w ** 2,
                  cfg.q_k ** 2, cfg.q_rho ** 2, cfg.q_bg ** 2]) * dt
    return Q


# Measurement matrices (constant: both measurements are linear in the state).
H_POSE = np.zeros((3, N_S))
H_POSE[0, IX] = H_POSE[1, IY] = H_POSE[2, ITH] = 1.0
H_GYRO = np.zeros((1, N_S))
H_GYRO[0, IW] = H_GYRO[0, IBG] = 1.0


# ----------------------------------------------------------------------------
# Filter
# ----------------------------------------------------------------------------


class SurfaceFilter:
    """EKF over [pose, k, rho, b_g]. Call ``predict`` with each control
    interval's input, ``update_pose`` / ``update_gyro`` whenever a measurement
    arrives. ``nis`` holds the last update's normalized innovation squared."""

    def __init__(self, cfg: TVLQGConfig, pose: Sequence[float]) -> None:
        self.cfg = cfg
        self.x = cfg.x0(pose)
        self.P = cfg.p0()
        self.nis: float = float("nan")
        self.nis_dof: int = 0
        self.gated: bool = False
        self.n_gated: int = 0

    # ---- time update -------------------------------------------------
    def predict(self, v: float, omega_cmd: float, dt: float) -> None:
        if dt <= 0.0:
            return
        tau = self.cfg.yaw_tau
        F, G = jacobians(self.x, v, omega_cmd, dt, tau)
        Q = process_noise(self.x, v, omega_cmd, dt, self.cfg, G)
        self.x = propagate(self.x, v, omega_cmd, dt, tau)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # ---- measurement updates ------------------------------------------
    def _update(self, innov: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        K = self.P @ H.T @ Sinv
        self.nis = float(innov @ Sinv @ innov)
        self.nis_dof = int(innov.shape[0])
        # Innovation gate (module docstring, INNOVATION GATE): outside the
        # chi^2 bound the model does not hold, so the update may correct the
        # pose but must not teach the surface parameters.
        self.gated = self.nis > CHI2_GATE[self.cfg.gate_p][self.nis_dof]
        if self.gated:
            K = K.copy()
            K[IK:, :] = 0.0
            self.n_gated += 1
        self.x = self.x + K @ innov
        self.x[ITH] = wrap_to_pi(self.x[ITH])
        # Joseph form: valid for ANY gain (including the gated one) and stays
        # symmetric positive definite over long drives.
        IKH = np.eye(N_S) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def update_pose(self, pose: Sequence[float],
                    r_pos: Optional[float] = None,
                    r_theta: Optional[float] = None) -> None:
        rp = self.cfg.r_pos if r_pos is None else r_pos
        rt = self.cfg.r_theta if r_theta is None else r_theta
        innov = np.array([pose[0] - self.x[IX], pose[1] - self.x[IY],
                          wrap_to_pi(pose[2] - self.x[ITH])])
        self._update(innov, H_POSE, np.diag([rp, rp, rt]) ** 2)

    def update_linear(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Any measurement z = H x + noise. For a partially observed pose,
        e.g. ICP's across-corridor position only (ICP is degenerate along a
        corridor). Not for rows involving theta,
        whose innovation must be wrapped."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        H = np.atleast_2d(np.asarray(H, dtype=float))
        self._update(z - H @ self.x, H, np.atleast_2d(np.asarray(R, dtype=float)))

    def update_gyro(self, omega_meas: float, r_gyro: Optional[float] = None) -> None:
        """z = actual yaw rate + bias. rho is learned through the yaw-rate
        state's dynamics, not from this row directly."""
        r = self.cfg.r_gyro if r_gyro is None else r_gyro
        pred = self.x[IW] + self.x[IBG]
        self._update(np.array([omega_meas - pred]), H_GYRO, np.array([[r * r]]))

    # ---- readout ---------------------------------------------------------
    @property
    def pose(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def k(self) -> Tuple[float, float]:
        """(k_hat, sigma_k)."""
        return float(self.x[IK]), float(np.sqrt(self.P[IK, IK]))

    @property
    def rho(self) -> Tuple[float, float]:
        return float(self.x[IRHO]), float(np.sqrt(self.P[IRHO, IRHO]))

    def surface_departure(self, k_surface: float) -> float:
        """|k_hat - k_surface| in units of sigma_k: the surface-switch statistic."""
        k, s = self.k
        return abs(k - k_surface) / s if s > 0 else float("inf")


def covariance_along_plan(
    poses: np.ndarray,
    v: Sequence[float],
    omega_cmd: Sequence[float],
    dt: float,
    cfg: TVLQGConfig,
    k: Optional[float] = None,
    pose_every: int = 1,
    gyro_every: int = 1,
) -> np.ndarray:
    """P(t) along a plan, before any drive: the LTV filter covariance.

    The linearization point is the PLAN (poses, (v, omega_cmd) per tick, k =
    cfg.k0 unless given, rho = b_g = 0, the yaw rate at its commanded value),
    so the result depends on the plan and the measurement schedule only.
    ``pose_every`` / ``gyro_every``: a measurement every n ticks (0 = never).
    Returns (N, N_S, N_S), the posterior
    covariance after tick n's updates.
    """
    poses = np.asarray(poses, dtype=float)
    n = len(poses)
    kk = cfg.k0 if k is None else k
    P = cfg.p0()
    out = np.empty((n, N_S, N_S))
    Rp = np.diag([cfg.r_pos, cfg.r_pos, cfg.r_theta]) ** 2
    Rg = np.array([[cfg.r_gyro ** 2]])
    for i in range(n):
        if i > 0:
            xp = np.array([*poses[i - 1], omega_cmd[max(i - 2, 0)], kk, 0.0, 0.0])
            F, G = jacobians(xp, v[i - 1], omega_cmd[i - 1], dt, cfg.yaw_tau)
            P = F @ P @ F.T + process_noise(xp, v[i - 1], omega_cmd[i - 1], dt, cfg, G)
        meas = ((H_POSE, Rp, pose_every), (H_GYRO, Rg, gyro_every))
        for H, R, every in meas:
            if every and i % every == 0:
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                IKH = np.eye(N_S) - K @ H
                P = IKH @ P @ IKH.T + K @ R @ K.T
        out[i] = 0.5 * (P + P.T)
    return out
