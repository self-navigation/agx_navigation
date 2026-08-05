"""Aggregate per-trajectory tracking errors into the number the search minimizes.

Every evaluation drives the same trajectories in the same order with the same
terrain seed. If any trajectory is missing, the evaluation scores `inf` rather
than a mean over the survivors, since dropping a hard trajectory would make a
broken candidate look better. The objective is the mean of max|e_cross| in
metres; final_err is reported but not optimized.

Repeated evaluations are combined with `how="mean"` by default; the median is
available for per-trajectory diagnostics.

Pure module: no ROS, no Gazebo, no numpy needed.
"""

import math
from typing import Dict, List, Sequence


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


def median(values: Sequence[float]) -> float:
    """Median of a non-empty sequence. inf if anything is missing/non-finite.

    Deliberately NOT statistics.median: a single inf must poison the result
    (a failed rollout invalidates the evaluation, same rule as `aggregate`),
    whereas statistics.median would happily return a finite middle value with a
    dead rollout sitting in the tail.
    """
    vals = list(values)
    if not vals:
        raise ValueError("no values to take a median of")
    if any(v is None or not math.isfinite(v) for v in vals):
        return math.inf
    vals = sorted(float(v) for v in vals)
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def reduce_repeats(repeats: Sequence[Dict[str, float]],
                   expected: Sequence[str],
                   how: str = "mean") -> Dict[str, float]:
    """Collapse per-repeat {traj: err} dicts into one {traj: err}.

    Reduces ACROSS REPEATS, PER TRAJECTORY -- see the module docstring for why
    that axis and not the other. A trajectory missing from ANY repeat is missing
    from the result, so `aggregate` will correctly return inf: a candidate must
    not look good because one of its rollouts died.
    """
    if not repeats:
        raise ValueError("no repeats to reduce")
    if how not in ("median", "mean", "max"):
        raise ValueError(f"unknown reduction {how!r}; "
                         "use 'median' (search) or 'mean'/'max' (validation)")
    out: Dict[str, float] = {}
    for name in expected:
        vals: List[float] = []
        for rep in repeats:
            if name not in rep:
                vals = []
                break
            vals.append(rep[name])
        if not vals:
            continue
        if how == "median":
            out[name] = median(vals)
        elif how == "mean":
            out[name] = (math.inf
                         if any(v is None or not math.isfinite(v) for v in vals)
                         else sum(float(v) for v in vals) / len(vals))
        else:
            out[name] = (math.inf
                         if any(v is None or not math.isfinite(v) for v in vals)
                         else max(float(v) for v in vals))
    return out
