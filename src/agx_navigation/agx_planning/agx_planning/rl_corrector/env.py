"""Gymnasium environment for training the residual wheel corrector.

The robot must follow a FROZEN nominal trajectory (it never replans). Each step:
the agent observes the current tracking error (path-relative), outputs a
per-wheel residual (rad/s), that residual is ADDED to the nominal feedforward,
the bridge advances the sim by one control_dt, and the resulting error drives
the reward.

The env is bridge-agnostic (see bridge.py): KinematicBridge for fast, Gazebo-free
validation; GazeboBridge for real training. All MDP math is the shared pure logic
in obs/coeff/reward, so the observation matches deployment exactly.

Episode:
  reset()  -- sample a nominal, randomize start offset, reset the bridge.
  step(a)  -- apply nominal[k] + residual(a), advance dt, score error vs nominal[k+1].
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

from .coeff import apply_residual, clipped_action
from .config import RLCorrectorConfig
from .geometry import cumulative_arclength, project_arclength
from .obs import build_observation, observation_dim, wrap_to_pi
from .reward import compute_reward, is_failure, is_success
from . import nominal as nominal_mod


def make_nominal_sampler(
    cfg: RLCorrectorConfig,
    kinds=None,
    v_range=(0.15, 0.45),
    omega_max: float = 1.0,
    duration_range=(2.0, 5.0),
) -> Callable:
    """Return a sampler(rng) -> Nominal over parametric primitives, with the ranges
    exposed so training can run a CURRICULUM (easy primitives first, then widen).

    kinds:     list sampled uniformly each episode; repeat an entry to weight it.
               None -> the default mix ["straight","arc","arc","scurve"] (arcs 2x).
               Restrict to e.g. ["straight","arc"] for a gentle first phase.
    v_range:   (min, max) commanded body speed [m/s].
    omega_max: max |omega| for arcs/scurves; sampled from [0.2, omega_max] with a
               random sign. Lower it for gentler turns (the hard, high-|omega|
               turns are where the corrector breaches the heading corridor).
    duration_range: (min, max) primitive duration [s].
    """
    pool = list(kinds) if kinds else ["straight", "arc", "arc", "scurve"]
    vlo, vhi = v_range
    dlo, dhi = duration_range
    omega_lo = min(0.2, omega_max)  # stay valid if a phase sets omega_max < 0.2

    def sample(rng: np.random.Generator):
        kind = str(rng.choice(pool))
        v = float(rng.uniform(vlo, vhi))
        omega = float(rng.uniform(omega_lo, omega_max)) * (1.0 if rng.random() < 0.5 else -1.0)
        duration = float(rng.uniform(dlo, dhi))
        return nominal_mod.generate_primitive(cfg, kind, v, omega, duration)

    return sample


def default_nominal_sampler(cfg: RLCorrectorConfig) -> Callable:
    """Return a sampler(rng) -> Nominal drawing the default full-difficulty mix."""
    return make_nominal_sampler(cfg)


def make_recorded_sampler(paths) -> Callable:
    """Return a sampler(rng) -> Nominal that loads a uniformly-random recorded
    (Tier-B) trajectory from `paths` (see nominal.load_recorded_dir) each
    episode. Loads from disk every call rather than caching in memory -- a
    trajectory library is expected to be far larger than fits comfortably in
    RAM, and training throughput is bridge-bound, not I/O-bound."""
    paths = list(paths)

    def sample(rng: np.random.Generator):
        path = paths[int(rng.integers(len(paths)))]
        return nominal_mod.load_recorded(path)

    return sample


def make_blended_sampler(samplers, weights) -> Callable:
    """Return a sampler(rng) -> Nominal that each episode picks one of
    `samplers` with the given `weights` (need not sum to 1) and delegates to it.
    Used to mix Tier-A synthetic primitives with Tier-B recorded trajectories in
    one curriculum phase."""
    samplers = list(samplers)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    def sample(rng: np.random.Generator):
        idx = int(rng.choice(len(samplers), p=w))
        return samplers[idx](rng)

    return sample


class WheelCorrectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: RLCorrectorConfig,
        bridge,
        nominal_sampler: Optional[Callable] = None,
        # Initial pose perturbation (x, y, theta) the corrector must null. Kept
        # at/below the goal tolerance (goal_tolerance_xy=0.10 m) so success is
        # actually reachable -- the SAC_4 default lateral offset of 0.15 m
        # exceeded the 0.10 m goal tolerance, making success impossible from the
        # start and starving SAC of any positive signal.
        start_offset=(0.08, 0.08, 0.08),
        terrain_sampler: Optional[Callable] = None,
        seed: Optional[int] = None,
        debug_steps: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.bridge = bridge
        self.sampler = nominal_sampler or default_nominal_sampler(cfg)
        self.terrain_sampler = terrain_sampler  # None until Phase 3
        self.start_offset = start_offset
        # >0 -> print the first `debug_steps` steps of every episode (measured
        # twist, tracking error, error-RATE, reward). Aimed at the GazeboBridge
        # reset-transition suspect: a set_pose teleport that leaves stale twist or
        # a pose jump makes the first-step error rate (divided by control_dt) spike,
        # which is the kind of outlier a bootstrapping critic amplifies into the
        # divergence. The KinematicBridge (which trains cleanly) is the control --
        # run both with --debug-steps and compare the first post-reset rows.
        self.debug_steps = int(debug_steps)

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
        self.prev_action = np.zeros(cfg.action_dim)
        self._prev_err = None
        self._prev_proj = 0.0

    # ------------------------------------------------------------------

    def _costate_at(self, idx: int):
        if not self.cfg.use_costates or self.nominal.costates is None:
            return None
        i = min(idx, self.nominal.costates.shape[0] - 1)
        return self.nominal.costates[i]

    def _make_obs(self, planned_pose, st, cmd, prev_action, costates):
        return build_observation(
            self.cfg,
            planned_pose,
            st.pose,
            self._prev_err,
            cmd_left=float(cmd[0]),
            cmd_right=float(cmd[1]),
            v_meas=st.v,
            omega_meas=st.omega,
            prev_action=prev_action,
            imu=st.imu if self.cfg.use_imu else None,
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
        self.prev_action = np.zeros(self.cfg.action_dim)
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
            prev_action=self.prev_action, costates=self._costate_at(0),
        )
        self._prev_err = err
        return obs, {}

    def step(self, action):
        cfg = self.cfg
        k = self.k
        n = len(self.nominal)
        # Once the nominal is exhausted (k >= n) we are in the grace window: hold
        # the LAST nominal command (wheels[n-1]) so the corrector still has a
        # feedforward to scale and can nudge the robot the last few cm into goal
        # tolerance. min() also keeps the normal in-path steps unchanged.
        cmd_idx = min(k, n - 1)
        cmd_l, cmd_r = self.nominal.wheels[cmd_idx]
        action_c = clipped_action(action, cfg)
        wheels = apply_residual(action, float(cmd_l), float(cmd_r), cfg)

        st = self.bridge.step(wheels, cfg.control_dt)
        self.k = k + 1
        self._steps += 1

        # poses has n+1 entries (index n is the goal); clamp into the grace window.
        planned_next = self.nominal.poses[min(self.k, n)]
        cmd_next = self.nominal.wheels[min(self.k, n - 1)]

        obs, err = self._make_obs(
            planned_next, st, cmd_next,
            prev_action=action_c, costates=self._costate_at(min(self.k, n)),
        )

        proj, _ = project_arclength(st.pose[:2], self.nominal.poses[:, :2], self._cum)
        raw_progress = proj - self._prev_proj
        self._prev_proj = proj
        # Cap rewarded progress at the nominal's OWN per-step advance. The task is
        # to TRACK the nominal, not to race ahead of it: without the cap the agent
        # farms the (dense) progress reward by over-throttling (positive residual) or by
        # drifting so its path-projection jumps forward on curves -- both trade
        # tracking accuracy for arc-length and walk it into the corridor wall (the
        # SAC_5 drift). Falling behind / going backward is still penalized via the
        # lower side (min keeps negative deltas intact). In the grace window the
        # advance is 0 (both indices clamp to n), so grace can't farm progress.
        nominal_advance = float(self._cum[min(self.k, n)] - self._cum[min(k, n)])
        progress = min(raw_progress, nominal_advance)

        reached_end = self.k >= n
        succeeded = False
        if reached_end:
            goal = self.nominal.poses[-1]
            d = math.hypot(st.pose[0] - goal[0], st.pose[1] - goal[1])
            hd = wrap_to_pi(st.pose[2] - goal[2])
            succeeded = is_success(cfg, d, hd)

        failed = bool(is_failure(cfg, err) or st.contact)
        reward = compute_reward(cfg, err, action_c, self.prev_action, progress, failed, succeeded)

        # Error rate this step (same quantity the obs feeds the policy, divided by
        # control_dt): a teleport/stale-twist spike at reset shows up here first.
        prev = self._prev_err if self._prev_err is not None else err
        err_rate = float(np.linalg.norm((err - prev) / cfg.control_dt))
        if self.debug_steps and self._steps <= self.debug_steps:
            print(f"  [dbg step {self._steps:2d}] v={st.v:+.3f} w={st.omega:+.3f} "
                  f"e_along={err[0]:+.3f} e_cross={err[1]:+.3f} e_head={err[2]:+.3f} "
                  f"rate={err_rate:7.2f} r={reward:+.2f}", flush=True)

        self.prev_action = action_c
        self._prev_err = err

        # Steps spent past the nominal's end. We only declare "ran out of path"
        # (truncate without success) once the grace window is used up, giving the
        # corrector goal_grace_steps tries to reach tolerance first.
        grace_used = self.k - n
        grace_expired = reached_end and grace_used >= cfg.goal_grace_steps

        terminated = bool(failed or succeeded)
        truncated = bool((grace_expired and not succeeded) or self._steps >= cfg.max_steps)
        info = {
            "e_cross": float(err[1]),
            "e_heading": float(err[2]),
            "err_rate": err_rate,
            "v": float(st.v),
            "omega": float(st.omega),
            "progress": float(progress),
            "succeeded": succeeded,
            "failed": failed,
            "nominal_label": getattr(self.nominal, "label", ""),
            # Granular reason this episode ended (None mid-episode). `outcome_kind`
            # is a STABLE category key for logging/aggregation (goal/corridor/
            # heading/collision/ran_out/timeout/nonfinite); `outcome` is the
            # human-readable string for the console. `failed` stays the coarse
            # train-metric flag.
            "outcome_kind": None,
            "outcome": None,
        }
        if terminated or truncated:
            info["outcome_kind"], info["outcome"] = self._classify_outcome(
                err, st, succeeded, reached_end)
        return obs, float(reward), terminated, truncated, info

    def _classify_outcome(self, err, st, succeeded: bool, reached_end: bool):
        """Why this episode ended, mirroring the terminated/truncated logic in
        step(). Returns (kind, human_string): `kind` is a stable category key for
        aggregation, the string is for the console."""
        cfg = self.cfg
        if succeeded:
            return "goal", "reached goal"
        if not np.all(np.isfinite(err)):
            return "nonfinite", "diverged (non-finite state)"
        if st.contact:
            return "collision", "collision"
        if abs(err[1]) > cfg.corridor_epsilon:
            return "corridor", f"left the corridor (|e_cross|={abs(err[1]):.2f}m)"
        if abs(err[2]) > cfg.max_heading_err:
            return "heading", f"heading breach (|e_heading|={abs(err[2]):.2f}rad)"
        if reached_end:
            return "ran_out", f"ran out of path, off-tolerance (|e_cross|={abs(err[1]):.2f}m)"
        return "timeout", "timed out (max_steps)"

    def close(self):
        self.bridge.close()
