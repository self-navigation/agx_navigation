"""Neighboring-optimal (time-varying LQR) trajectory corrector. Pure (numpy only).

WHAT THIS IS
------------
The planner solves the optimal-control problem ONCE and hands us a frozen
trajectory. Reality then deviates (wheel slip, odometry error, battery sag). The
question this module answers is:

    given an optimal trajectory, what is the OPTIMAL correction for a small
    deviation from it?

That question has a classical answer -- *neighboring optimal control* (Bryson &
Ho, "Applied Optimal Control", ch. 6), also called neighboring extremals. Take
the second variation of the same cost the planner minimized, about the extremal
it found; the result is an accessory linear-quadratic problem whose solution is a
time-varying linear feedback

    u(t) = u_ref(t) - K(t) . e(t)

with K(t) from a Riccati recursion along the trajectory. To first order in the
deviation this IS the optimal correction -- it is not a heuristic tracker. That
distinction is the whole point: a pursuit controller steers back to the path by a
rule invented from scratch, discarding the optimality the planner paid for, while
this feedback is derived FROM the planner's own problem.

Runtime cost is a 2x3 matrix-vector product. K(t) is precomputed once, with the
trajectory.

WHAT IS AND IS NOT MODELLED (read before trusting the word "optimal")
--------------------------------------------------------------------
Two deliberate simplifications; both are engineering choices, not oversights, and
both bound how literally "optimal" should be read.

1. STATE/CONTROL SPACE. The planner's model is 5D wheel-space
   (p_x, p_y, theta, w_l, w_r) with per-wheel accelerations as control. We
   correct in the 3D pose error (e_along, e_cross, e_heading) with the body twist
   (v, omega) as control, because:
     - the planner PUBLISHES the wheel-speed state, not accelerations, and its
       own docstring notes tau_wheel=0 is right since the velocity interface
       tracks within a physics step -- so wheel speed is effectively a direct
       control and the acceleration states are not independent tracking dof;
     - the physical chassis accepts ONLY a twist and does its own wheel mapping,
       so a per-wheel correction is not realizable on hardware today;
     - the planner's docstring states it "lives in the 2D controllable quotient"
       anyway, per-wheel freedom mattering only under terrain heterogeneity.
   Consequence: this is the neighboring-optimal feedback for the TRACKING problem
   in that quotient, not the exact second variation of the planner's full 5D
   functional.

2. THE COST HESSIANS. The exact accessory problem uses Q = H_xx, R = H_uu,
   N = H_xu of the planner's Hamiltonian along the trajectory. H_uu is clean
   (the effort term is quadratic, so H_uu = gamma_wheel * I and H_xu = 0), but
   H_xx involves second derivatives of the Fast-Marching potential T(p), which is
   only piecewise smooth and numerically noisy. We therefore use an explicit
   diagonal Q on the tracking error instead. This is the standard practical form
   of neighboring optimal control (TVLQR about the extremal); recovering the true
   H_xx is a refinement, not a prerequisite.

Bottom line: this is principled feedback derived from the trajectory's own
linearization, with tunable tracking weights -- strictly better founded than a
pursuit heuristic, and honestly short of the full second variation.

ERROR DYNAMICS
--------------
With the error expressed in the REFERENCE frame (same convention as
rl_corrector.obs.tracking_error, so the two are directly comparable):

    e = R(theta_ref)^T (p - p_ref),   e_heading = wrap(theta - theta_ref)

the exact kinematics are

    e_along_dot   = omega_ref * e_cross + v cos(e_heading) - v_ref
    e_cross_dot   = -omega_ref * e_along + v sin(e_heading)
    e_heading_dot = omega - omega_ref

Linearizing about e = 0 with u = u_ref + du gives the classic time-varying pair

    A(t) = [[0,           omega_ref, 0    ],
            [-omega_ref,  0,         v_ref],
            [0,           0,         0    ]]
    B    = [[1, 0],
            [0, 0],
            [0, 1]]

Note A depends only on the REFERENCE twist, so the whole gain schedule is a
function of the planned trajectory alone and precomputes offline.

The (e_cross, e_heading) pair is controllable only when v_ref != 0 -- a
stationary robot cannot steer itself sideways. That is a property of the vehicle,
not a modelling artifact, and `min_speed_for_steering` handles it explicitly
rather than letting the Riccati solution blow up.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Error / control dimensions.
N_X = 3
N_U = 2


def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class TVLQRConfig:
    """Weights and limits for the neighboring-optimal corrector.

    Q/R are the accessory problem's weights (see the module docstring on why they
    are explicit rather than read off the planner's Hamiltonian). Their RATIO is
    what matters: larger Q = track harder, larger R = correct more gently.
    """

    # --- Tracking weights (diagonal Q on [e_along, e_cross, e_heading]) ----
    # Cross-track is weighted hardest: leaving the corridor is the failure mode,
    # while an along-track lag is benign (the robot is on the path, just behind).
    q_along: float = 1.0
    q_cross: float = 10.0
    q_heading: float = 5.0
    # Terminal weights (Riccati seed). Heavier than the running Q so the
    # trajectory is driven onto the goal rather than merely near it.
    qf_scale: float = 10.0

    # --- Control weights (diagonal R on [dv, domega]) ----------------------
    # Correction effort. Angular correction is cheaper than linear because
    # steering is how a differential-drive platform actually fixes cross-track
    # error; penalizing it hard would leave only the (useless) along-track dof.
    r_v: float = 1.0
    r_omega: float = 0.25

    # --- Authority limits --------------------------------------------------
    # Absolute caps on the CORRECTION, not the total command. Additive (unlike
    # the RL corrector's multiplicative coefficients), so authority does not
    # vanish when the reference command is zero.
    max_dv: float = 0.25          # [m/s]
    max_domega: float = 1.0       # [rad/s]
    # Below this reference speed the (e_cross, e_heading) block is uncontrollable
    # (see module docstring): steering authority is faded out rather than letting
    # the gains grow without bound. Along-track correction stays active.
    min_speed_for_steering: float = 0.05   # [m/s]

    # --- Solver ------------------------------------------------------------
    # Iterations for the steady-state (online / frozen-time) Riccati solve.
    dare_iters: int = 500
    dare_tol: float = 1e-9

    def q_matrix(self) -> np.ndarray:
        return np.diag([self.q_along, self.q_cross, self.q_heading]).astype(float)

    def qf_matrix(self) -> np.ndarray:
        return self.qf_scale * self.q_matrix()

    def r_matrix(self) -> np.ndarray:
        return np.diag([self.r_v, self.r_omega]).astype(float)


@dataclass
class CorrectionDiagnostics:
    """Everything needed to judge whether the corrector is working, per tick.

    Deliberately verbose: the corrector is the component we cannot easily eyeball
    from a Gazebo window, so every tick reports what it saw, what it did, and
    whether it ran out of authority. `saturated_*` is the flag to watch -- a
    corrector that is persistently saturated is being asked to fix a deviation
    beyond its authority, which means either the limits are too tight or the
    trajectory needs replanning.
    """

    e_along: float = 0.0
    e_cross: float = 0.0
    e_heading: float = 0.0
    e_norm: float = 0.0            # sqrt(e_along^2 + e_cross^2), position only
    v_ref: float = 0.0
    omega_ref: float = 0.0
    dv: float = 0.0                # applied linear correction
    domega: float = 0.0            # applied angular correction
    dv_raw: float = 0.0            # before saturation
    domega_raw: float = 0.0
    saturated_v: bool = False
    saturated_omega: bool = False
    steering_faded: bool = False   # v_ref below min_speed_for_steering
    gain_norm: float = 0.0         # ||K||_F, for spotting ill-conditioned solves
    index: int = -1                # trajectory tick this correction used
    valid: bool = True             # False -> caller should fall back to identity

    def as_array(self) -> np.ndarray:
        """Flat float vector for publishing on a Float64MultiArray.

        ORDER IS API -- append only, never reorder, or logged runs stop being
        comparable across versions.
        """
        return np.array([
            self.e_along, self.e_cross, self.e_heading, self.e_norm,
            self.v_ref, self.omega_ref,
            self.dv, self.domega, self.dv_raw, self.domega_raw,
            float(self.saturated_v), float(self.saturated_omega),
            float(self.steering_faded), self.gain_norm,
            float(self.index), float(self.valid),
        ], dtype=float)

    # Field order of as_array(), for log headers and offline analysis.
    FIELDS = (
        "e_along", "e_cross", "e_heading", "e_norm",
        "v_ref", "omega_ref",
        "dv", "domega", "dv_raw", "domega_raw",
        "saturated_v", "saturated_omega",
        "steering_faded", "gain_norm", "index", "valid",
    )


# ----------------------------------------------------------------------------
# Linearization
# ----------------------------------------------------------------------------


def error_dynamics(v_ref: float, omega_ref: float) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous-time (A, B) of the tracking-error dynamics at one tick.

    Depends only on the REFERENCE twist -- see the module docstring for the
    derivation. This is what makes the gain schedule precomputable.
    """
    A = np.array([
        [0.0, omega_ref, 0.0],
        [-omega_ref, 0.0, v_ref],
        [0.0, 0.0, 0.0],
    ], dtype=float)
    B = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ], dtype=float)
    return A, B


def discretize(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """First-order (Euler) discretization. Exact enough at control_dt=0.1 s for
    the speeds this platform runs; the eigenvalues of A are O(v_ref, omega_ref),
    so |A dt| << 1."""
    Ad = np.eye(N_X) + A * dt
    Bd = B * dt
    return Ad, Bd


# ----------------------------------------------------------------------------
# Riccati solves
# ----------------------------------------------------------------------------


def _riccati_step(
    S: np.ndarray, Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """One backward step of the discrete Riccati recursion.

    Returns (K, S_prev) with the control law du = -K e.
    """
    BtS = Bd.T @ S
    K = np.linalg.solve(R + BtS @ Bd, BtS @ Ad)
    AmBK = Ad - Bd @ K
    # Joseph-style symmetric form: numerically better behaved than
    # Q + Ad^T S (Ad - Bd K), which loses symmetry over long horizons.
    S_prev = Q + K.T @ R @ K + AmBK.T @ S @ AmBK
    return K, 0.5 * (S_prev + S_prev.T)


def gain_schedule(
    v_ref: Sequence[float],
    omega_ref: Sequence[float],
    dt: float,
    cfg: TVLQRConfig,
) -> List[np.ndarray]:
    """Backward Riccati pass over a whole trajectory -> per-tick gains K[k].

    This is the finite-horizon, genuinely TIME-VARYING solution: it knows the
    trajectory ends, so gains stiffen near the goal instead of being tuned for an
    infinite horizon that does not exist. Call once when a trajectory arrives.

    v_ref/omega_ref are the reference body twists per tick (length N). Returns N
    gain matrices, K[k] of shape (2, 3).
    """
    n = len(v_ref)
    if n != len(omega_ref):
        raise ValueError("v_ref and omega_ref must have equal length")
    if n == 0:
        return []

    Q = cfg.q_matrix()
    R = cfg.r_matrix()
    S = cfg.qf_matrix()

    gains: List[Optional[np.ndarray]] = [None] * n
    for k in range(n - 1, -1, -1):
        vk = _faded_speed(float(v_ref[k]), cfg)
        A, B = error_dynamics(vk, float(omega_ref[k]))
        Ad, Bd = discretize(A, B, dt)
        K, S = _riccati_step(S, Ad, Bd, Q, R)
        gains[k] = K
    return [g for g in gains if g is not None]


def steady_state_gain(
    v_ref: float, omega_ref: float, dt: float, cfg: TVLQRConfig
) -> np.ndarray:
    """Frozen-time (infinite-horizon) gain for one reference twist.

    Used in ONLINE mode, where commands stream in one tick at a time and there is
    no future trajectory to run a backward pass over. Iterates the Riccati
    recursion to convergence rather than pulling in scipy's DARE solver, keeping
    this module dependency-free.
    """
    Q = cfg.q_matrix()
    R = cfg.r_matrix()
    A, B = error_dynamics(_faded_speed(v_ref, cfg), omega_ref)
    Ad, Bd = discretize(A, B, dt)

    S = Q.copy()
    K = np.zeros((N_U, N_X))
    for _ in range(cfg.dare_iters):
        K, S_next = _riccati_step(S, Ad, Bd, Q, R)
        if np.max(np.abs(S_next - S)) < cfg.dare_tol:
            S = S_next
            break
        S = S_next
    return K


def _faded_speed(v_ref: float, cfg: TVLQRConfig) -> float:
    """Reference speed used for the linearization, floored in magnitude.

    At v_ref = 0 the (e_cross, e_heading) block is uncontrollable and the Riccati
    solution for those states is meaningless. Rather than let that happen we
    linearize about a minimum speed; the resulting steering gain is then faded to
    zero by `correct()` so we never command a turn the robot cannot execute.
    Sign is preserved -- reversing flips which way you steer to fix cross-track.
    """
    if abs(v_ref) >= cfg.min_speed_for_steering:
        return v_ref
    return float(np.sign(v_ref) or 1.0) * cfg.min_speed_for_steering


# ----------------------------------------------------------------------------
# Applying a correction
# ----------------------------------------------------------------------------


def tracking_error(planned_pose, actual_pose) -> np.ndarray:
    """(along, cross, heading) error in the reference frame.

    Identical convention to rl_corrector.obs.tracking_error so the two correctors
    can be compared on the same numbers.
    """
    px, py, pth = planned_pose
    ax, ay, ath = actual_pose
    dx = ax - px
    dy = ay - py
    c, s = np.cos(pth), np.sin(pth)
    return np.array([
        c * dx + s * dy,
        -s * dx + c * dy,
        wrap_to_pi(ath - pth),
    ], dtype=float)


def correct(
    K: np.ndarray,
    err: np.ndarray,
    v_ref: float,
    omega_ref: float,
    cfg: TVLQRConfig,
    index: int = -1,
) -> Tuple[float, float, CorrectionDiagnostics]:
    """Apply du = -K e, saturate, and report diagnostics.

    Returns (v_cmd, omega_cmd, diagnostics) -- the CORRECTED twist, ready to be
    published to the chassis directly or mapped to wheel speeds for the sim.

    Fails safe: any non-finite input yields the uncorrected reference twist with
    `valid=False`, so a bad solve can never inject motion.
    """
    diag = CorrectionDiagnostics(v_ref=float(v_ref), omega_ref=float(omega_ref),
                                 index=int(index))

    err = np.asarray(err, dtype=float).ravel()
    if err.shape[0] != N_X or not np.all(np.isfinite(err)) or not np.all(np.isfinite(K)):
        diag.valid = False
        return float(v_ref), float(omega_ref), diag

    diag.e_along, diag.e_cross, diag.e_heading = (float(e) for e in err)
    diag.e_norm = float(np.hypot(err[0], err[1]))
    diag.gain_norm = float(np.linalg.norm(K))

    du = -K @ err
    dv_raw, domega_raw = float(du[0]), float(du[1])
    diag.dv_raw, diag.domega_raw = dv_raw, domega_raw

    # Fade steering when the reference is too slow to steer (see _faded_speed).
    if abs(v_ref) < cfg.min_speed_for_steering:
        diag.steering_faded = True
        domega_raw = 0.0

    dv = float(np.clip(dv_raw, -cfg.max_dv, cfg.max_dv))
    domega = float(np.clip(domega_raw, -cfg.max_domega, cfg.max_domega))
    diag.saturated_v = abs(dv_raw) > cfg.max_dv
    diag.saturated_omega = abs(domega_raw) > cfg.max_domega
    diag.dv, diag.domega = dv, domega

    return float(v_ref) + dv, float(omega_ref) + domega, diag


# ----------------------------------------------------------------------------
# Twist <-> wheel-pair conversion
# ----------------------------------------------------------------------------


def twist_to_wheels(v: float, omega: float, kin) -> Tuple[float, float]:
    """(v, omega) -> (w_l, w_r) using the shared chassis kinematics.

    `kin` is anything exposing body_to_wheels (RLCorrectorConfig or PlannerConfig
    -- they intentionally mirror each other).
    """
    wl, wr = kin.body_to_wheels(v, omega)
    return float(wl), float(wr)


def wheels_to_twist(w_l: float, w_r: float, kin) -> Tuple[float, float]:
    """(w_l, w_r) -> (v, omega). Exact inverse of twist_to_wheels.

    This is the reduction the REAL ROBOT needs: the chassis accepts only a twist
    and does its own wheel mapping, so any per-wheel command must collapse
    through here before it can leave the machine.
    """
    v, omega = kin.wheels_to_body(w_l, w_r)
    return float(v), float(omega)
