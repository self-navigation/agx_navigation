"""Unit tests for the RL corrector coefficient mapping (pure logic)."""

import numpy as np
import pytest

from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.coeff import (
    apply_coefficients,
    coefficients_from_action,
)


def test_identity_invariant_4d():
    """action == 0 must reproduce the current identity corrector exactly."""
    cfg = RLCorrectorConfig(action_dim=4)
    out = apply_coefficients(np.zeros(4), left=3.0, right=-2.0, cfg=cfg)
    assert out == pytest.approx([3.0, 3.0, -2.0, -2.0])


def test_identity_invariant_2d():
    cfg = RLCorrectorConfig(action_dim=2)
    out = apply_coefficients(np.zeros(2), left=3.0, right=-2.0, cfg=cfg)
    assert out == pytest.approx([3.0, 3.0, -2.0, -2.0])


def test_coefficient_bounds():
    cfg = RLCorrectorConfig(coeff_k=0.5)
    assert coefficients_from_action([1, 1, 1, 1], cfg) == pytest.approx([1.5] * 4)
    assert coefficients_from_action([-1, -1, -1, -1], cfg) == pytest.approx([0.5] * 4)
    # Out-of-range actions are clipped to [-1, 1] before mapping.
    assert coefficients_from_action([5, -5, 0, 0], cfg) == pytest.approx([1.5, 0.5, 1.0, 1.0])


def test_per_wheel_split_4d():
    """4-D action gives front/rear differential authority within a side."""
    cfg = RLCorrectorConfig(action_dim=4, coeff_k=0.5)
    out = apply_coefficients([1.0, -1.0, 0.0, 0.0], left=10.0, right=10.0, cfg=cfg)
    # fl = 1.5*10, rl = 0.5*10, fr = rr = 10
    assert out == pytest.approx([15.0, 5.0, 10.0, 10.0])


def test_per_side_shares_front_rear_2d():
    cfg = RLCorrectorConfig(action_dim=2, coeff_k=0.5)
    out = apply_coefficients([1.0, -1.0], left=10.0, right=10.0, cfg=cfg)
    assert out == pytest.approx([15.0, 15.0, 5.0, 5.0])


def test_clip_to_wheel_cmd_max():
    cfg = RLCorrectorConfig(action_dim=4, coeff_k=0.5, wheel_cmd_max=20.0)
    out = apply_coefficients([1, 1, 1, 1], left=18.0, right=-18.0, cfg=cfg)
    # 1.5 * 18 = 27 -> clipped to 20; -27 -> -20
    assert out == pytest.approx([20.0, 20.0, -20.0, -20.0])


def test_zero_command_stays_zero():
    """A coefficient on a zero nominal cannot inject motion (multiplicative)."""
    cfg = RLCorrectorConfig(action_dim=4)
    out = apply_coefficients([1, 1, -1, -1], left=0.0, right=0.0, cfg=cfg)
    assert out == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_invalid_action_dim_raises():
    cfg = RLCorrectorConfig(action_dim=3)
    with pytest.raises(ValueError):
        apply_coefficients([0, 0, 0], 1.0, 1.0, cfg)
