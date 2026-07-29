"""Unit tests for the RL corrector additive residual mapping (pure logic)."""

import numpy as np
import pytest

from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.coeff import (
    apply_residual,
    clipped_action,
    residual_from_action,
)


def test_identity_invariant_4d():
    """action == 0 must reproduce the current identity corrector exactly."""
    cfg = RLCorrectorConfig(action_dim=4)
    out = apply_residual(np.zeros(4), left=3.0, right=-2.0, cfg=cfg)
    assert out == pytest.approx([3.0, 3.0, -2.0, -2.0])


def test_identity_invariant_2d():
    cfg = RLCorrectorConfig(action_dim=2)
    out = apply_residual(np.zeros(2), left=3.0, right=-2.0, cfg=cfg)
    assert out == pytest.approx([3.0, 3.0, -2.0, -2.0])


def test_residual_bounds():
    cfg = RLCorrectorConfig(wheel_residual_max=2.0)
    assert residual_from_action([1, 1, 1, 1], cfg) == pytest.approx([2.0] * 4)
    assert residual_from_action([-1, -1, -1, -1], cfg) == pytest.approx([-2.0] * 4)
    # Out-of-range actions are clipped to [-1, 1] before mapping.
    assert residual_from_action([5, -5, 0, 0], cfg) == pytest.approx([2.0, -2.0, 0.0, 0.0])


def test_clipped_action():
    assert clipped_action([2, -2, 0.5, 0], None) == pytest.approx([1.0, -1.0, 0.5, 0.0])


def test_per_wheel_split_4d():
    """4-D action gives front/rear differential authority within a side."""
    cfg = RLCorrectorConfig(action_dim=4, wheel_residual_max=2.0)
    out = apply_residual([1.0, -1.0, 0.0, 0.0], left=10.0, right=10.0, cfg=cfg)
    # fl = 10+2, rl = 10-2, fr = rr = 10
    assert out == pytest.approx([12.0, 8.0, 10.0, 10.0])


def test_per_side_shares_front_rear_2d():
    cfg = RLCorrectorConfig(action_dim=2, wheel_residual_max=2.0)
    out = apply_residual([1.0, -1.0], left=10.0, right=10.0, cfg=cfg)
    assert out == pytest.approx([12.0, 12.0, 8.0, 8.0])


def test_clip_to_wheel_cmd_max():
    cfg = RLCorrectorConfig(action_dim=4, wheel_residual_max=5.0, wheel_cmd_max=20.0)
    out = apply_residual([1, 1, 1, 1], left=18.0, right=-18.0, cfg=cfg)
    # 18 + 5 = 23 -> clipped to 20; -18 + 5 = -13 (within bounds, no clip)
    assert out == pytest.approx([20.0, 20.0, -13.0, -13.0])


def test_zero_command_still_gets_authority():
    """A residual on a zero nominal DOES inject motion (additive) -- the fix
    for the multiplicative scheme's zero-authority-at-zero-nominal flaw."""
    cfg = RLCorrectorConfig(action_dim=4, wheel_residual_max=2.0)
    out = apply_residual([1, 1, -1, -1], left=0.0, right=0.0, cfg=cfg)
    assert out == pytest.approx([2.0, 2.0, -2.0, -2.0])


def test_invalid_action_dim_raises():
    cfg = RLCorrectorConfig(action_dim=3)
    with pytest.raises(ValueError):
        apply_residual([0, 0, 0], 1.0, 1.0, cfg)
