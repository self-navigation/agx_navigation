"""Aggregate per-trajectory tracking errors into the number the search minimizes.

Every evaluation drives the same trajectories in the same order with the same
terrain seed. If any trajectory is missing, the evaluation scores `inf` rather
than a mean over the survivors, since dropping a hard trajectory would make a
broken candidate look better. The objective is the mean of max|e_cross| in
metres; final_err is reported but not optimized.

Pure module: no ROS, no Gazebo, no numpy needed.
"""

import math
from typing import Dict, Sequence


def aggregate(per_traj: Dict[str, float], expected: Sequence[str]) -> float:
    """Mean max|e_cross| over `expected`, or inf if any is missing/non-finite.

    `expected` is the fixed trajectory list, so the denominator cannot drift.
    """
    if not expected:
        raise ValueError("expected trajectory list is empty")
    total = 0.0
    for name in expected:
        if name not in per_traj:
            return math.inf
        v = per_traj[name]
        if v is None or not math.isfinite(v):
            return math.inf
        total += float(v)
    return total / len(expected)
