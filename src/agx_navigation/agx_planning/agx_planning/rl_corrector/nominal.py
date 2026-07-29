"""Frozen reference trajectories for training. Pure (numpy only).

Tier-A parametric primitives: generated analytically by inverting the chassis
kinematics (RLCorrectorConfig.body_to_wheels), so no planner is involved and the
nominal never reacts to deviation. Each Nominal carries, per tick:
  poses[i]  = (x, y, theta) reference pose at the START of step i
  wheels[i] = (w_left, w_right) nominal command APPLIED during step i
which mirrors the planner's "wheel command at tick i is applied at pose i".

Tier-B (recorded PMP-planner rollouts, which also carry costates) load into the
same Nominal container via load_recorded() / load_recorded_dir() below. They are
produced offline by generate_trajectories.py from a real PMPShootingSolver
rollout (poses, wheel_cmds, costates, dt_sample -- see pmp_planner/rollout.py's
RolloutChunk, whose fields map 1:1 onto Nominal's) and saved as .npz files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Nominal:
    poses: np.ndarray   # (N+1, 3) x, y, theta -- start-of-step refs plus final goal
    wheels: np.ndarray  # (N, 2) w_left, w_right -- command applied during each step
    dt: float
    costates: Optional[np.ndarray] = None  # (N, 5) or None (Tier-A has none)
    label: str = ""     # human-readable description, e.g. "S-bend (v=0.30, w=+-0.70)"

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

    if kind == "straight":
        label = f"straight (v={v:.2f}, {duration:.1f}s)"
    elif kind == "arc":
        side = "left" if omega >= 0 else "right"
        label = f"{side} arc (v={v:.2f}, w={omega:+.2f}, {duration:.1f}s)"
    else:  # scurve (the only remaining valid kind; others raised above)
        label = f"S-bend (v={v:.2f}, w=+-{abs(omega):.2f}, {duration:.1f}s)"

    return Nominal(poses=poses, wheels=wheels, dt=dt, label=label)


def load_recorded(path: str) -> Nominal:
    """Load a once-recorded PMP rollout (poses, wheels, costates, dt) from an
    .npz file written by generate_trajectories.py (or any producer matching
    RolloutChunk's field names: poses, wheel_cmds, costates, dt_sample)."""
    with np.load(path) as f:
        poses = np.asarray(f["poses"], dtype=float)
        wheels = np.asarray(f["wheel_cmds"], dtype=float)
        costates = np.asarray(f["costates"], dtype=float) if "costates" in f else None
        dt = float(f["dt_sample"])
    # RolloutChunk's poses/wheel_cmds are PARALLEL arrays (both length N, one pose
    # per commanded tick) -- unlike Tier-A's generate_primitive(), which appends
    # one extra integrated final pose. Nominal.poses is contractually (N+1, 3):
    # env.py's grace-window indexing (poses[min(k, n)]) relies on that extra row
    # existing. rollout_generator only yields a chunk on success, terminating
    # within goal tolerance, so duplicating the last recorded pose as the "goal"
    # row is accurate to within that tolerance, not a placeholder.
    poses = np.concatenate([poses, poses[-1:]], axis=0)
    label = f"recorded ({Path(path).stem})"
    return Nominal(poses=poses, wheels=wheels, dt=dt, costates=costates, label=label)


def load_recorded_dir(directory: str) -> list:
    """Return the sorted list of .npz trajectory paths in `directory`, for a
    sampler to draw from at random each episode. Raises if the directory has no
    trajectories -- an empty recorded library silently degrading to Tier-A-only
    training would be easy to miss."""
    paths = sorted(str(p) for p in Path(directory).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz trajectories found under {directory}")
    return paths
