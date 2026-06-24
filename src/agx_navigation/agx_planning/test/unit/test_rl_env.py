"""Env contract tests using the pure KinematicBridge (no Gazebo, no torch)."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from agx_planning.rl_corrector.config import RLCorrectorConfig
from agx_planning.rl_corrector.env import WheelCorrectorEnv
from agx_planning.rl_corrector.kinematic_bridge import KinematicBridge
from agx_planning.rl_corrector.nominal import generate_primitive
from agx_planning.rl_corrector.obs import observation_dim


def _straight_sampler(cfg, v=0.3, duration=3.0):
    return lambda rng: generate_primitive(cfg, "straight", v=v, omega=0.0, duration=duration)


def _make_env(cfg=None, slip=None, start_offset=(0.0, 0.0, 0.0), **kw):
    cfg = cfg or RLCorrectorConfig(use_costates=False)
    bridge = KinematicBridge(cfg, slip=slip)
    return WheelCorrectorEnv(
        cfg, bridge, nominal_sampler=_straight_sampler(cfg, **kw),
        start_offset=start_offset, seed=0,
    )


def test_spaces():
    cfg = RLCorrectorConfig(action_dim=4, use_costates=False)
    env = _make_env(cfg)
    assert env.action_space.shape == (4,)
    assert env.observation_space.shape == (observation_dim(cfg),)


def test_check_env():
    from gymnasium.utils.env_checker import check_env

    env = _make_env()
    # warn=False: unbounded obs box only triggers warnings, not failures.
    check_env(env, skip_render_check=True)


def test_reset_returns_valid_obs():
    cfg = RLCorrectorConfig(use_costates=False)
    env = _make_env(cfg)
    obs, info = env.reset(seed=1)
    assert obs.shape == (observation_dim(cfg),)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_identity_tracks_noslip_nominal():
    """action=0 on a no-slip bridge from an on-path start tracks the nominal
    essentially perfectly -- the env/MDP wiring correctness anchor."""
    env = _make_env(slip=None, start_offset=(0.0, 0.0, 0.0))
    env.reset(seed=0)
    max_cross = 0.0
    done = False
    while not done:
        _o, _r, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        max_cross = max(max_cross, abs(info["e_cross"]))
        done = terminated or truncated
    assert max_cross < 1e-6


def test_slip_makes_identity_drift():
    """With asymmetric slip, the identity corrector drifts off the path --
    confirming the env actually exposes the error the policy must fix."""
    # Right side slips (slower) -> robot veers right -> cross-track grows.
    env = _make_env(slip=[1.0, 1.0, 0.6, 0.6], start_offset=(0.0, 0.0, 0.0))
    env.reset(seed=0)
    max_cross = 0.0
    done = False
    while not done:
        _o, _r, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        max_cross = max(max_cross, abs(info["e_cross"]))
        done = terminated or truncated
    assert max_cross > 0.05


def test_nonzero_action_changes_trajectory():
    """A nonzero coefficient must actually steer differently from identity."""
    env_id = _make_env(start_offset=(0.0, 0.0, 0.0))
    env_id.reset(seed=0)
    _o, _r, _t, _tr, info_id = env_id.step(np.zeros(4, dtype=np.float32))

    env_a = _make_env(start_offset=(0.0, 0.0, 0.0))
    env_a.reset(seed=0)
    # Speed up left wheels, slow right -> yaw -> heading diverges this step
    # (cross-track only diverges after the heading change propagates).
    _o, _r, _t, _tr, info_a = env_a.step(np.array([1, 1, -1, -1], dtype=np.float32))
    assert info_id["e_heading"] == pytest.approx(0.0, abs=1e-9)
    assert abs(info_a["e_heading"]) > 1e-3


def test_goal_grace_extends_episode():
    """Once the nominal is exhausted the episode keeps running for
    goal_grace_steps extra steps so the corrector can reach goal tolerance.
    Success is disabled here (negative tolerance) to isolate the grace window:
    identity on a no-slip straight stays in-corridor and never succeeds, so the
    episode must truncate at exactly n + goal_grace_steps."""
    for grace in (0, 4):
        cfg = RLCorrectorConfig(
            use_costates=False, goal_grace_steps=grace, goal_tolerance_xy=-1.0,
        )
        env = _make_env(cfg)
        env.reset(seed=0)
        n = len(env.nominal)
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            _o, _r, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            steps += 1
        assert not terminated          # success disabled, no-slip => no failure
        assert truncated
        assert steps == n + grace
        assert info["outcome_kind"] == "ran_out"


def test_seed_determinism():
    env1 = _make_env()
    env2 = _make_env()
    o1, _ = env1.reset(seed=42)
    o2, _ = env2.reset(seed=42)
    assert np.allclose(o1, o2)
    a = np.array([0.3, -0.2, 0.1, 0.0], dtype=np.float32)
    s1 = env1.step(a)
    s2 = env2.step(a)
    assert np.allclose(s1[0], s2[0])
    assert s1[1] == pytest.approx(s2[1])
