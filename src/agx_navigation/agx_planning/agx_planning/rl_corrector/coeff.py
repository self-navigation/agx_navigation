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


def clipped_action(action, cfg) -> np.ndarray:
    """Clip a raw policy action to [-1, 1]^action_dim. This clipped value (not
    the residual in rad/s) is what reward/obs track as the "previous action" --
    it's already zero-centered and scale-stable across changes to
    wheel_residual_max."""
    return np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)


def residual_from_action(action, cfg) -> np.ndarray:
    """Map a raw policy action in [-1, 1]^action_dim to an additive per-wheel
    residual in rad/s: residual_i = wheel_residual_max * a_i. action == 0 ->
    zero residual (identity)."""
    return cfg.wheel_residual_max * clipped_action(action, cfg)


def apply_residual(action, left: float, right: float, cfg) -> List[float]:
    """Add the action's residual to the nominal per-side wheel commands and
    return the four-wheel setpoint [front_left, rear_left, front_right,
    rear_right], clamped to +/- wheel_cmd_max.

    4-D action: independent per-wheel residuals (front/rear may differ).
    2-D action: one residual per side (front/rear share).
    """
    r = residual_from_action(action, cfg)
    if cfg.action_dim == 2:
        r_l, r_r = r[0], r[1]
        wheels = [left + r_l, left + r_l, right + r_r, right + r_r]
    elif cfg.action_dim == 4:
        wheels = [left + r[0], left + r[1], right + r[2], right + r[3]]
    else:
        raise ValueError(f"action_dim must be 2 or 4, got {cfg.action_dim}")

    m = cfg.wheel_cmd_max
    return [float(np.clip(w, -m, m)) for w in wheels]
