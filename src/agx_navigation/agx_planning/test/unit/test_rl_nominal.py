"""Unit tests for the parametric nominal generator + kinematics (pure logic)."""

import math

import numpy as np
import pytest

from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.nominal import generate_primitive


def test_kinematics_round_trip():
    """body_to_wheels then wheels_to_body recovers (v, omega)."""
    cfg = RLCorrectorConfig()
    for v, omega in [(0.3, 0.0), (0.2, 0.5), (-0.1, -0.4), (0.0, 1.0)]:
        wl, wr = cfg.body_to_wheels(v, omega)
        v2, w2 = cfg.wheels_to_body(wl, wr)
        assert v2 == pytest.approx(v)
        assert w2 == pytest.approx(omega)


def test_straight_primitive_moves_along_heading():
    cfg = RLCorrectorConfig(control_dt=0.1)
    nom = generate_primitive(cfg, "straight", v=0.5, omega=0.0, duration=1.0, theta0=0.0)
    assert len(nom) == 10
    # First pose is the start.
    assert nom.poses[0] == pytest.approx([0.0, 0.0, 0.0])
    # Heading unchanged; pure +x motion; y stays 0.
    assert np.allclose(nom.poses[:, 2], 0.0)
    assert np.allclose(nom.poses[:, 1], 0.0)
    assert nom.poses[-1, 0] > nom.poses[0, 0]
    # Straight => equal left/right wheel speeds.
    assert nom.wheels[:, 0] == pytest.approx(nom.wheels[:, 1])
    # Tier-A nominals carry no costates.
    assert nom.costates is None


def test_arc_primitive_curves():
    cfg = RLCorrectorConfig(control_dt=0.1)
    nom = generate_primitive(cfg, "arc", v=0.3, omega=0.5, duration=2.0)
    # Heading should accumulate (turning left).
    assert nom.poses[-1, 2] > 0.5
    # Right wheel faster than left for a left turn (omega>0 => wr>wl).
    assert np.all(nom.wheels[:, 1] > nom.wheels[:, 0])


def test_scurve_reverses_curvature():
    cfg = RLCorrectorConfig(control_dt=0.1)
    nom = generate_primitive(cfg, "scurve", v=0.3, omega=0.5, duration=2.0)
    n = len(nom)
    # First half turns one way, second half the other.
    first_turn = nom.poses[n // 2 - 1, 2] - nom.poses[0, 2]
    second_turn = nom.poses[-1, 2] - nom.poses[n // 2, 2]
    assert first_turn > 0.0
    assert second_turn < 0.0


def test_wheel_command_matches_pose_tick():
    """wheels[i] should be the body_to_wheels of the commanded (v, omega) at i."""
    cfg = RLCorrectorConfig(control_dt=0.1)
    v = 0.4
    nom = generate_primitive(cfg, "straight", v=v, omega=0.0, duration=0.5)
    wl_expected, wr_expected = cfg.body_to_wheels(v, 0.0)
    assert nom.wheels[0, 0] == pytest.approx(wl_expected)
    assert nom.wheels[0, 1] == pytest.approx(wr_expected)


def test_unknown_kind_raises():
    cfg = RLCorrectorConfig()
    with pytest.raises(ValueError):
        generate_primitive(cfg, "spiral", v=0.3, omega=0.1, duration=1.0)
