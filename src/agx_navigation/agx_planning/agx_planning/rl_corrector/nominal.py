"""Frozen reference trajectories for training. Pure (numpy only).

Tier-A parametric primitives: generated analytically by inverting the chassis
kinematics (RLCorrectorConfig.body_to_wheels), so no planner is involved and the
nominal never reacts to deviation. Each Nominal carries, per tick:
  poses[i]  = (x, y, theta) reference pose at the START of step i
  wheels[i] = (w_left, w_right) nominal command APPLIED during step i
which mirrors the planner's "wheel command at tick i is applied at pose i".

Tier-B (recorded planner runs, which also carry costates) will load into the
same Nominal container later; load_recorded() is a stub for now.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Nominal:
    poses: np.ndarray   # (N+1, 3) x, y, theta -- start-of-step refs plus final goal
    wheels: np.ndarray  # (N, 2) w_left, w_right -- command applied during each step
    dt: float
    costates: Optional[np.ndarray] = None  # (N, 5) or None (Tier-A has none)

    def __len__(self) -> int:
        # Number of commands / steps. poses has one extra entry (the goal).
        return self.wheels.shape[0]


def generate_primitive(
    cfg,
    kind: str,
    v: float,
    omega: float,
    duration: float,
    x0: float = 0.0,
    y0: float = 0.0,
    theta0: float = 0.0,
) -> Nominal:
    """Generate one motion primitive.

    kind:
      "straight" -- omega forced to 0
      "arc"      -- constant omega
      "scurve"   -- omega for the first half, -omega for the second
    """
    n = max(1, int(round(duration / cfg.control_dt)))
    dt = cfg.control_dt
    poses = np.zeros((n + 1, 3))  # start-of-step refs + final goal pose
    wheels = np.zeros((n, 2))

    x, y, th = x0, y0, theta0
    for i in range(n):
        if kind == "straight":
            w_i = 0.0
        elif kind == "arc":
            w_i = omega
        elif kind == "scurve":
            w_i = omega if i < n // 2 else -omega
        else:
            raise ValueError(f"unknown primitive kind {kind!r}")

        wl, wr = cfg.body_to_wheels(v, w_i)
        poses[i] = (x, y, th)
        wheels[i] = (wl, wr)

        # Integrate forward using the DERIVED body velocity (round-trips to v, w_i).
        v_i, w_body = cfg.wheels_to_body(wl, wr)
        x += v_i * np.cos(th) * dt
        y += v_i * np.sin(th) * dt
        th += w_body * dt

    poses[n] = (x, y, th)  # final/goal pose after the last command
    return Nominal(poses=poses, wheels=wheels, dt=dt)


def load_recorded(path: str) -> Nominal:  # pragma: no cover - Tier-B stub
    """Load a once-recorded planner rollout (poses, wheels, costates) from disk.

    Implemented in the Tier-B phase; recorded chunks come from the offline
    planner's PlanToGoal feedback (pose_x/y/theta, wheel_left/right, lam_*).
    """
    raise NotImplementedError("Tier-B recorded-nominal loading not implemented yet")
