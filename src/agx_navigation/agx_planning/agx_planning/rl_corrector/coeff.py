"""Action -> per-wheel coefficients -> clamped wheel command. Pure (numpy only).

Shared verbatim by the training env and the deployed _correct() so the mapping
is byte-identical. The fail-safe invariant is load-bearing: action == 0 maps to
coefficient 1, which reproduces the current identity corrector exactly.
"""

from typing import List

import numpy as np


def coefficients_from_action(action, cfg) -> np.ndarray:
    """Map a raw policy action in [-1, 1]^action_dim to multiplicative
    coefficients c_i = 1 + coeff_k * a_i. action == 0 -> all-ones (identity)."""
    a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
    return 1.0 + cfg.coeff_k * a


def apply_coefficients(action, left: float, right: float, cfg) -> List[float]:
    """Apply the action's coefficients to the nominal per-side wheel commands
    and return the four-wheel setpoint [front_left, rear_left, front_right,
    rear_right], clamped to +/- wheel_cmd_max.

    4-D action: independent per-wheel coefficients (front/rear may differ).
    2-D action: one coefficient per side (front/rear share).
    """
    c = coefficients_from_action(action, cfg)
    if cfg.action_dim == 2:
        c_l, c_r = c[0], c[1]
        wheels = [c_l * left, c_l * left, c_r * right, c_r * right]
    elif cfg.action_dim == 4:
        wheels = [c[0] * left, c[1] * left, c[2] * right, c[3] * right]
    else:
        raise ValueError(f"action_dim must be 2 or 4, got {cfg.action_dim}")

    m = cfg.wheel_cmd_max
    return [float(np.clip(w, -m, m)) for w in wheels]
