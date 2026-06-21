"""Gymnasium environment for training the residual wheel corrector.

The robot must follow a FROZEN nominal trajectory (it never replans). Each step:
the agent observes the current tracking error (path-relative), outputs per-wheel
coefficients, those scale the nominal feedforward, the bridge advances the sim by
one control_dt, and the resulting error drives the reward.

The env is bridge-agnostic (see bridge.py): KinematicBridge for fast, Gazebo-free
validation; GazeboBridge for real training. All MDP math is the shared pure logic
in obs/coeff/reward, so the observation matches deployment exactly.

Episode:
  reset()  -- sample a nominal, randomize start offset, reset the bridge.
  step(a)  -- apply coeff(a)*nominal[k], advance dt, score error vs nominal[k+1].
  ends     -- terminated on failure (corridor/heading/contact) or success
              (reached goal in tolerance); truncated on reaching the nominal end
              without success or hitting max_steps.
"""

import math
from typing import Callable, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "gymnasium is required for the RL corrector env. "
        "Install with: pip install gymnasium"
    ) from e

from .coeff import apply_coefficients, coefficients_from_action
from .config import RLCorrectorConfig
from .geometry import cumulative_arclength, project_arclength
from .obs import build_observation, observation_dim, wrap_to_pi
from .reward import compute_reward, is_failure, is_success
from . import nominal as nominal_mod


def default_nominal_sampler(cfg: RLCorrectorConfig) -> Callable:
    """Return a sampler(rng) -> Nominal drawing varied parametric primitives."""

    def sample(rng: np.random.Generator):
        kind = rng.choice(["straight", "arc", "arc", "scurve"])
        v = float(rng.uniform(0.15, 0.45))
        omega = float(rng.uniform(0.2, 1.0)) * (1.0 if rng.random() < 0.5 else -1.0)
        duration = float(rng.uniform(2.0, 5.0))
        return nominal_mod.generate_primitive(cfg, kind, v, omega, duration)

    return sample


class WheelCorrectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: RLCorrectorConfig,
        bridge,
        nominal_sampler: Optional[Callable] = None,
        start_offset=(0.10, 0.15, 0.10),
        terrain_sampler: Optional[Callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.bridge = bridge
        self.sampler = nominal_sampler or default_nominal_sampler(cfg)
        self.terrain_sampler = terrain_sampler  # None until Phase 3
        self.start_offset = start_offset

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim(cfg),), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)

        # Episode state (populated in reset()).
        self.nominal = None
        self._cum = None
        self.k = 0
        self._steps = 0
        self.prev_coeff = np.ones(cfg.action_dim)
        self._prev_err = None
        self._prev_proj = 0.0

    # ------------------------------------------------------------------

    def _costate_at(self, idx: int):
        if not self.cfg.use_costates or self.nominal.costates is None:
            return None
        i = min(idx, self.nominal.costates.shape[0] - 1)
        return self.nominal.costates[i]

    def _make_obs(self, planned_pose, st, cmd, prev_coeff, costates):
        return build_observation(
            self.cfg,
            planned_pose,
            st.pose,
            self._prev_err,
            cmd_left=float(cmd[0]),
            cmd_right=float(cmd[1]),
            v_meas=st.v,
            omega_meas=st.omega,
            prev_coeff=prev_coeff,
            wheel_speeds=st.wheel_speeds if self.cfg.use_wheel_speeds else None,
            costates=costates,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.nominal = self.sampler(self._rng)
        self._cum = cumulative_arclength(self.nominal.poses[:, :2])
        self.k = 0
        self._steps = 0
        self.prev_coeff = np.ones(self.cfg.action_dim)
        self._prev_err = None

        ox = self._rng.uniform(-self.start_offset[0], self.start_offset[0])
        oy = self._rng.uniform(-self.start_offset[1], self.start_offset[1])
        oth = self._rng.uniform(-self.start_offset[2], self.start_offset[2])
        sp = self.nominal.poses[0]
        start = (sp[0] + ox, sp[1] + oy, sp[2] + oth)

        terrain = self.terrain_sampler(self._rng) if self.terrain_sampler else None
        st = self.bridge.reset(start, terrain=terrain)

        self._prev_proj, _ = project_arclength(start[:2], self.nominal.poses[:, :2], self._cum)
        obs, err = self._make_obs(
            self.nominal.poses[0], st, self.nominal.wheels[0],
            prev_coeff=self.prev_coeff, costates=self._costate_at(0),
        )
        self._prev_err = err
        return obs, {}

    def step(self, action):
        cfg = self.cfg
        k = self.k
        cmd_l, cmd_r = self.nominal.wheels[k]
        coeff = coefficients_from_action(action, cfg)
        wheels = apply_coefficients(action, float(cmd_l), float(cmd_r), cfg)

        st = self.bridge.step(wheels, cfg.control_dt)
        self.k = k + 1
        self._steps += 1

        n = len(self.nominal)
        planned_next = self.nominal.poses[self.k]            # poses has n+1 entries
        cmd_next = self.nominal.wheels[min(self.k, n - 1)]

        obs, err = self._make_obs(
            planned_next, st, cmd_next,
            prev_coeff=coeff, costates=self._costate_at(self.k),
        )

        proj, _ = project_arclength(st.pose[:2], self.nominal.poses[:, :2], self._cum)
        progress = proj - self._prev_proj
        self._prev_proj = proj

        reached_end = self.k >= n
        succeeded = False
        if reached_end:
            goal = self.nominal.poses[-1]
            d = math.hypot(st.pose[0] - goal[0], st.pose[1] - goal[1])
            hd = wrap_to_pi(st.pose[2] - goal[2])
            succeeded = is_success(cfg, d, hd)

        failed = bool(is_failure(cfg, err) or st.contact)
        reward = compute_reward(cfg, err, coeff, self.prev_coeff, progress, failed, succeeded)

        self.prev_coeff = coeff
        self._prev_err = err

        terminated = bool(failed or succeeded)
        truncated = bool((reached_end and not succeeded) or self._steps >= cfg.max_steps)
        info = {
            "e_cross": float(err[1]),
            "e_heading": float(err[2]),
            "progress": float(progress),
            "succeeded": succeeded,
            "failed": failed,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.bridge.close()
