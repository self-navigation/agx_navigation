"""Unit tests for the RL corrector observation builder (pure logic)."""

import math

import numpy as np
import pytest

from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.obs import (
    build_observation,
    observation_dim,
    tracking_error,
    wrap_to_pi,
)


def _raw_cfg(**kw):
    # Norms = 1 so assertions read raw quantities.
    base = dict(
        pos_err_norm=1.0, rate_norm=1.0, twist_v_norm=1.0, twist_w_norm=1.0,
        costate_norm=1.0, wheel_cmd_max=1.0, control_dt=0.1,
    )
    base.update(kw)
    return RLCorrectorConfig(**base)


def test_wrap_to_pi():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    assert wrap_to_pi(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_to_pi(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)


def test_tracking_error_zero():
    err = tracking_error((1.0, 2.0, 0.5), (1.0, 2.0, 0.5))
    assert err == pytest.approx([0.0, 0.0, 0.0])


def test_tracking_error_pure_cross_track():
    """Robot displaced to the planner's left (heading +x) -> +cross, 0 along."""
    err = tracking_error((0.0, 0.0, 0.0), (0.0, 0.5, 0.0))
    assert err[0] == pytest.approx(0.0)   # along
    assert err[1] == pytest.approx(0.5)   # cross (left positive)
    assert err[2] == pytest.approx(0.0)


def test_tracking_error_frame_rotation():
    """With planned heading +90deg, a +x map displacement is cross-track (right)."""
    err = tracking_error((0.0, 0.0, math.pi / 2), (1.0, 0.0, math.pi / 2))
    assert err[0] == pytest.approx(0.0)    # along (no +y disp)
    assert err[1] == pytest.approx(-1.0)   # +x is to the right of +y heading
    assert err[2] == pytest.approx(0.0)


def test_tracking_error_heading_wraps():
    err = tracking_error((0, 0, math.pi - 0.05), (0, 0, -math.pi + 0.05))
    assert err[2] == pytest.approx(0.1, abs=1e-6)


def test_observation_dim_toggles():
    cfg = RLCorrectorConfig(action_dim=4, use_prev_coeff=True,
                            use_wheel_speeds=True, use_costates=True)
    assert observation_dim(cfg) == 10 + 4 + 4 + 5
    cfg2 = RLCorrectorConfig(action_dim=2, use_prev_coeff=True,
                             use_wheel_speeds=False, use_costates=False)
    assert observation_dim(cfg2) == 10 + 2


def test_build_observation_dim_matches():
    cfg = _raw_cfg(use_costates=True, use_wheel_speeds=True)
    obs, err = build_observation(
        cfg, (0, 0, 0), (0, 0, 0), prev_err=None,
        cmd_left=1.0, cmd_right=1.0, v_meas=0.0, omega_meas=0.0,
        prev_coeff=np.ones(4), wheel_speeds=np.zeros(4), costates=np.zeros(5),
    )
    assert obs.shape == (observation_dim(cfg),)
    assert obs.dtype == np.float32
    assert err == pytest.approx([0.0, 0.0, 0.0])


def test_build_observation_rates_zero_on_first_step():
    cfg = _raw_cfg(use_costates=False)
    obs, err = build_observation(
        cfg, (0, 0, 0), (0.3, 0.2, 0.0), prev_err=None,
        cmd_left=2.0, cmd_right=4.0, v_meas=0.5, omega_meas=-0.5,
    )
    # error dims
    assert obs[0] == pytest.approx(0.3)   # along
    assert obs[1] == pytest.approx(0.2)   # cross
    # rates zero (prev_err None)
    assert obs[3:6] == pytest.approx([0.0, 0.0, 0.0])
    # commands + twist passthrough (norms = 1)
    assert obs[6] == pytest.approx(2.0)
    assert obs[7] == pytest.approx(4.0)
    assert obs[8] == pytest.approx(0.5)
    assert obs[9] == pytest.approx(-0.5)


def test_build_observation_rates_from_prev_err():
    cfg = _raw_cfg(use_costates=False, use_prev_coeff=False, control_dt=0.1)
    obs, err = build_observation(
        cfg, (0, 0, 0), (0.3, 0.0, 0.0), prev_err=np.array([0.1, 0.0, 0.0]),
        cmd_left=0.0, cmd_right=0.0, v_meas=0.0, omega_meas=0.0,
    )
    # along rate = (0.3 - 0.1)/0.1 = 2.0
    assert obs[3] == pytest.approx(2.0)


def test_prev_coeff_centered_at_identity():
    cfg = _raw_cfg(action_dim=4, use_prev_coeff=True, use_costates=False)
    obs, _ = build_observation(
        cfg, (0, 0, 0), (0, 0, 0), prev_err=None,
        cmd_left=0.0, cmd_right=0.0, v_meas=0.0, omega_meas=0.0,
        prev_coeff=np.array([1.5, 0.5, 1.0, 1.0]),
    )
    # last 4 entries are prev_coeff - 1
    assert obs[-4:] == pytest.approx([0.5, -0.5, 0.0, 0.0])
