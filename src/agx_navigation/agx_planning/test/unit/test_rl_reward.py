"""Unit tests for the RL corrector reward + predicates (pure logic)."""

import numpy as np
import pytest

from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.reward import (
    compute_reward,
    is_failure,
    is_success,
)


def test_is_failure_corridor_breach():
    cfg = RLCorrectorConfig(corridor_epsilon=0.5, max_heading_err=1.0)
    assert not is_failure(cfg, [0.0, 0.4, 0.0])
    assert is_failure(cfg, [0.0, 0.6, 0.0])
    assert is_failure(cfg, [0.0, -0.6, 0.0])


def test_is_failure_heading_breach():
    cfg = RLCorrectorConfig(corridor_epsilon=10.0, max_heading_err=1.0)
    assert not is_failure(cfg, [0.0, 0.0, 0.9])
    assert is_failure(cfg, [0.0, 0.0, 1.1])


def test_is_failure_nonfinite():
    cfg = RLCorrectorConfig()
    assert is_failure(cfg, [0.0, np.nan, 0.0])
    assert is_failure(cfg, [np.inf, 0.0, 0.0])


def test_is_success():
    cfg = RLCorrectorConfig(goal_tolerance_xy=0.1, goal_tolerance_th=0.3)
    assert is_success(cfg, 0.05, 0.2)
    assert not is_success(cfg, 0.2, 0.2)   # too far
    assert not is_success(cfg, 0.05, 0.5)  # heading off


def test_reward_penalizes_cross_track():
    cfg = RLCorrectorConfig(w_cross=10.0, w_heading=0, w_progress=0,
                            w_effort=0, w_smooth=0, w_ontrack=0)
    ones = np.zeros(4)
    r_small = compute_reward(cfg, [0, 0.1, 0], ones, ones, 0.0)
    r_big = compute_reward(cfg, [0, 0.3, 0], ones, ones, 0.0)
    assert r_big < r_small < 0


def test_reward_rewards_progress():
    cfg = RLCorrectorConfig(w_cross=0, w_heading=0, w_progress=5.0,
                            w_effort=0, w_smooth=0, w_ontrack=0)
    ones = np.zeros(4)
    assert compute_reward(cfg, [0, 0, 0], ones, ones, 0.2) == pytest.approx(1.0)


def test_reward_effort_anchored_at_identity():
    cfg = RLCorrectorConfig(w_cross=0, w_heading=0, w_progress=0,
                            w_effort=1.0, w_smooth=0, w_ontrack=0)
    zero = np.zeros(4)
    # zero (identity) action -> zero effort penalty
    assert compute_reward(cfg, [0, 0, 0], zero, zero, 0.0) == pytest.approx(0.0)
    # nonzero action -> negative
    a = np.array([0.5, -0.5, 0.0, 0.0])
    assert compute_reward(cfg, [0, 0, 0], a, a, 0.0) == pytest.approx(-0.5)


def test_reward_smoothness_penalizes_change():
    cfg = RLCorrectorConfig(w_cross=0, w_heading=0, w_progress=0,
                            w_effort=0, w_smooth=1.0, w_ontrack=0)
    a = np.array([0.2, 0.2, 0.2, 0.2])
    prev = np.array([0.0, 0.0, 0.0, 0.0])
    # sum((0.2-0.0)^2)*4 = 0.04*4 = 0.16
    assert compute_reward(cfg, [0, 0, 0], a, prev, 0.0) == pytest.approx(-0.16)


def test_reward_terminal_penalty_and_bonus():
    cfg = RLCorrectorConfig(w_cross=0, w_heading=0, w_progress=0,
                            w_effort=0, w_smooth=0, w_ontrack=0,
                            term_penalty=50.0, success_bonus=40.0)
    ones = np.zeros(4)
    assert compute_reward(cfg, [0, 0, 0], ones, ones, 0.0, failed=True) == pytest.approx(-50.0)
    assert compute_reward(cfg, [0, 0, 0], ones, ones, 0.0, succeeded=True) == pytest.approx(40.0)


def test_reward_prev_action_none_defaults_zero():
    cfg = RLCorrectorConfig(w_cross=0, w_heading=0, w_progress=0,
                            w_effort=0, w_smooth=1.0, w_ontrack=0)
    zero = np.zeros(4)
    assert compute_reward(cfg, [0, 0, 0], zero, None, 0.0) == pytest.approx(0.0)


def test_reward_ontrack_bonus_decays_with_distance():
    cfg = RLCorrectorConfig(w_cross=0, w_heading=0, w_progress=0,
                            w_effort=0, w_smooth=0, w_ontrack=2.0,
                            corridor_epsilon=0.5)
    ones = np.zeros(4)
    # On the target point -> full bonus.
    assert compute_reward(cfg, [0, 0, 0], ones, ones, 0.0) == pytest.approx(2.0)
    # At the corridor edge -> decays to zero.
    assert compute_reward(cfg, [0, 0.5, 0], ones, ones, 0.0) == pytest.approx(0.0)
    # Past the edge -> clamped at zero, never negative.
    assert compute_reward(cfg, [0, 0.7, 0], ones, ones, 0.0) == pytest.approx(0.0)
    # Uses full distance to target (along + cross), not just cross.
    expect = 2.0 * (1 - (0.3 / 0.5) ** 2)
    assert compute_reward(cfg, [0.3, 0.0, 0], ones, ones, 0.0) == pytest.approx(expect)
