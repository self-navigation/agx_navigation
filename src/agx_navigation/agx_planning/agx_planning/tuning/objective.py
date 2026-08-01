"""Turn per-trajectory tracking errors into the single number the search minimizes.

FIXED TRAJECTORY SET, ALL-OR-NOTHING
------------------------------------
Every evaluation drives the *same* trajectories, in the same order, with the same
terrain seed. Nothing is sampled per evaluation, so two candidate gain sets are
always compared on identical work.

That invariant is load-bearing, and the way it breaks in practice is not
sampling but FAILURE: if one rollout dies (bridge timeout, crashed episode) and
the aggregate is a mean over "whatever finished", the denominator changes
between candidates. Because the trajectories differ enormously in difficulty --
a straight costs 0.1 m of error where a corner costs 1.5 m -- losing the hard one
makes a candidate look *better* the more it broke. The search would then be
actively rewarded for gains that crash rollouts.

So a missing trajectory makes the whole evaluation invalid (`inf`), never a mean
over the survivors. `inf` is a value Nelder-Mead handles natively: it is simply
the worst possible vertex and gets reflected away.

WHY MEAN OF max|e_cross| AND NOT sum, rms, OR final_err
-------------------------------------------------------
`mean` keeps the number in metres and comparable across different set sizes, so
a tuning log stays readable when the trajectory list changes between runs (the
cache key records the list, so results are never mixed across such a change).
`max|e_cross|` is the quantity TVLQR is being tuned to fix -- it is what the
oscillation on curved plans shows up as. `final_err` is reported alongside but
not optimized: a corrector can arrive at the goal having wandered metres off the
path in between, which is precisely the behaviour under investigation.

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
