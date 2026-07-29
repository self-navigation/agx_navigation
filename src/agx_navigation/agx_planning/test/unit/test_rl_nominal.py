"""Unit tests for the parametric nominal generator + kinematics (pure logic)."""

import math

import numpy as np
import pytest

from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.nominal import (
    generate_primitive,
    load_recorded,
    load_recorded_dir,
)


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


def test_load_recorded_round_trip(tmp_path):
    """load_recorded() must reconstruct a Nominal matching what a RolloutChunk-
    shaped .npz saves -- the exact contract generate_trajectories.py writes and
    the training sampler reads back."""
    poses = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    wheel_cmds = np.array([[1.0, 1.0], [1.0, 1.0]])
    costates = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
    path = tmp_path / "traj_00000.npz"
    np.savez(path, poses=poses, wheel_cmds=wheel_cmds, costates=costates,
             dt_sample=0.1)

    nom = load_recorded(str(path))
    # RolloutChunk's poses/wheels are parallel (both length N); Nominal.poses
    # must gain the extra (N+1)-th row env.py's grace-window indexing assumes
    # (poses[min(k, n)]) -- regression test for the IndexError this caused when
    # an episode ran a recorded nominal all the way to its end.
    assert nom.poses.shape == (poses.shape[0] + 1, 3)
    assert np.allclose(nom.poses[:-1], poses)
    assert np.allclose(nom.poses[-1], poses[-1])
    assert np.allclose(nom.wheels, wheel_cmds)
    assert np.allclose(nom.costates, costates)
    assert nom.dt == pytest.approx(0.1)
    assert len(nom) == wheel_cmds.shape[0]


def test_load_recorded_dir_lists_npz_files(tmp_path):
    for i in range(3):
        np.savez(tmp_path / f"traj_{i:05d}.npz",
                 poses=np.zeros((1, 3)), wheel_cmds=np.zeros((1, 2)),
                 costates=np.zeros((1, 5)), dt_sample=0.1)
    (tmp_path / "not_a_trajectory.txt").write_text("ignore me")

    paths = load_recorded_dir(str(tmp_path))
    assert len(paths) == 3
    assert all(p.endswith(".npz") for p in paths)


def test_load_recorded_dir_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recorded_dir(str(tmp_path))
