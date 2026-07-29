"""Action -> additive per-wheel residual -> clamped wheel command. Pure (numpy only).

Shared verbatim by the training env and the deployed _correct() so the mapping
is byte-identical. The fail-safe invariant is load-bearing: action == 0 maps to
a zero residual, which reproduces the current identity corrector exactly -- and,
unlike a multiplicative coefficient, that holds even when the nominal command
itself is zero (a wheel at rest still gets a residual, not a residual scaled by
zero authority).
"""

from typing import List

import numpy as np


def clipped_action(action, cfg, prev_action=None) -> np.ndarray:
    """Clip a raw policy action to [-1, 1]^action_dim, then (if `prev_action` is
    given and cfg.action_rate_limit > 0) slew-limit it toward prev_action. This
    clipped/limited value (not the residual in rad/s) is what reward/obs track
    as the "previous action" -- it's already zero-centered and scale-stable
    across changes to wheel_residual_max.

    The rate limit exists because a policy trained on KinematicBridge (no
    actuator dynamics -- each action maps instantly and exactly to a velocity)
    can learn to chatter the action every step at no cost there, then spin the
    real chassis into a heading breach on GazeboBridge, whose real inertia
    amplifies rapid sign flips into actual angular-velocity spikes far beyond
    anything in the recorded trajectories (see rl-corrector session notes,
    2026-07-29: a frozen phase-1 policy hit |omega|~4 rad/s on step 5-9 of a
    Gazebo rollout it had never trained on, purely from alternating a in
    [-1,1] every tick). Rate-limiting is a hard structural bound independent of
    what SAC learns, on top of (not instead of) the reward's w_smooth term."""
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
    limit = getattr(cfg, "action_rate_limit", 0.0)
    if prev_action is not None and limit > 0:
        prev = np.asarray(prev_action, dtype=float).ravel()
        a = prev + np.clip(a - prev, -limit, limit)
    return a


def residual_from_action(action, cfg, prev_action=None) -> np.ndarray:
    """Map a raw policy action in [-1, 1]^action_dim to an additive per-wheel
    residual in rad/s: residual_i = wheel_residual_max * a_i. action == 0 ->
    zero residual (identity)."""
    return cfg.wheel_residual_max * clipped_action(action, cfg, prev_action)


def apply_residual(action, left: float, right: float, cfg, prev_action=None) -> List[float]:
    """Add the action's residual to the nominal per-side wheel commands and
    return the four-wheel setpoint [front_left, rear_left, front_right,
    rear_right], clamped to +/- wheel_cmd_max.

    4-D action: independent per-wheel residuals (front/rear may differ).
    2-D action: one residual per side (front/rear share).
    """
    r = residual_from_action(action, cfg, prev_action)
    if cfg.action_dim == 2:
        r_l, r_r = r[0], r[1]
        wheels = [left + r_l, left + r_l, right + r_r, right + r_r]
    elif cfg.action_dim == 4:
        wheels = [left + r[0], left + r[1], right + r[2], right + r[3]]
    else:
        raise ValueError(f"action_dim must be 2 or 4, got {cfg.action_dim}")

    m = cfg.wheel_cmd_max
    return [float(np.clip(w, -m, m)) for w in wheels]
