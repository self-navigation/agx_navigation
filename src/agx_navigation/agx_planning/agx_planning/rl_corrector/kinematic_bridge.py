"""A pure-Python no-slip (optionally slipping) kinematic bridge.

Integrates the same chassis kinematics the nominals are generated from, so an
identity policy on a slip-free bridge tracks the nominal exactly -- the
correctness anchor for the env's MDP wiring and unit tests. An optional per-wheel
slip vector lets it crudely fake terrain without Gazebo (a fast sanity backend),
but it is NOT a physics sim; real training uses GazeboBridge.
"""

from typing import List, Optional

import numpy as np

from .bridge import StateReading


class KinematicBridge:
    def __init__(self, cfg, slip: Optional[List[float]] = None) -> None:
        self.cfg = cfg
        # Per-wheel multiplicative slip [fl, rl, fr, rr]; 1.0 == no slip.
        self.slip = None if slip is None else np.asarray(slip, dtype=float)
        self._x = self._y = self._th = 0.0

    def reset(self, start_pose, terrain=None) -> StateReading:
        self._x, self._y, self._th = (float(v) for v in start_pose)
        # `terrain` may carry per-wheel slip for a quick fake; ignored otherwise.
        if isinstance(terrain, dict) and "slip" in terrain:
            self.slip = np.asarray(terrain["slip"], dtype=float)
        return StateReading((self._x, self._y, self._th), 0.0, 0.0, [0.0] * 4, False)

    def step(self, wheels, dt: float) -> StateReading:
        w = np.asarray(wheels, dtype=float)
        if self.slip is not None:
            w = w * self.slip
        # Per-side effective speed = mean of that side's two wheels.
        wl = 0.5 * (w[0] + w[1])
        wr = 0.5 * (w[2] + w[3])
        v, omega = self.cfg.wheels_to_body(wl, wr)
        self._x += v * np.cos(self._th) * dt
        self._y += v * np.sin(self._th) * dt
        self._th += omega * dt
        return StateReading(
            (self._x, self._y, self._th), float(v), float(omega), list(map(float, w)), False
        )

    def close(self) -> None:
        pass
