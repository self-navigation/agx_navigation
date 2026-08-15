"""Aggregate per-trajectory tracking errors into the number the search minimizes.

Every evaluation drives the same trajectories in the same order with the same
terrain seed. If any trajectory is missing, the evaluation scores `inf` rather
than a mean over the survivors, since dropping a hard trajectory would make a
broken candidate look better. The objective is the mean of max|e_cross| in
metres; final_err is reported but not optimized.

Repeated evaluations are combined with `how="mean"` by default; the median is
available for per-trajectory diagnostics.

`metric_values` selects max|e_cross| or `J` (see `epsilon.py`). J spans orders
of magnitude across plans, so it is aggregated with `how="geometric"`;
max|e_cross| keeps the arithmetic mean.

Pure module: no ROS, no Gazebo, no numpy needed.
"""

import math
from typing import Dict, List, Mapping, Sequence

#: Per-trajectory metrics a search may minimize. All are "lower is better".
#: `j_total` is the SVCM currency and wants `how="geometric"`; see the module
#: docstring for the measurement that establishes that.
METRICS = ("max_cross", "j_total", "final_err")

#: The aggregator each metric should be reduced ACROSS TRAJECTORIES with.
DEFAULT_HOW = {"max_cross": "arithmetic", "j_total": "geometric",
               "final_err": "arithmetic"}


def aggregate(per_traj: Dict[str, float], expected: Sequence[str],
              how: str = "arithmetic") -> float:
    """Reduce per-trajectory scores to one number, or inf if any is missing.

    `expected` is the fixed trajectory list, so the denominator cannot drift.
    `how` is "arithmetic" (the default, correct for `max|e_cross|`) or
    "geometric" (correct for `J`) -- see the module docstring for the
    measurement behind that split.

    The metric is not named here on purpose: this reduces whatever scalar the
    caller selected with `metric_values`, so adding a metric does not mean
    touching this function.
    """
    if not expected:
        raise ValueError("expected trajectory list is empty")
    if how not in ("arithmetic", "geometric"):
        raise ValueError(f"unknown aggregator {how!r}; use 'arithmetic' or 'geometric'")
    vals: List[float] = []
    for name in expected:
        if name not in per_traj:
            return math.inf
        v = per_traj[name]
        if v is None or not math.isfinite(v):
            return math.inf
        vals.append(float(v))
    if how == "arithmetic":
        return sum(vals) / len(vals)
    # Geometric mean, computed in logs so a long list cannot overflow the
    # product. A metric of exactly 0 is a PERFECT trajectory, not an error, so
    # it is floored rather than allowed to send the log to -inf and drag the
    # whole aggregate to zero -- one flawless plan must not make every candidate
    # look equally perfect.
    if any(v < 0 for v in vals):
        raise ValueError("geometric aggregation needs non-negative values; "
                         "got a negative score")
    floor = 1e-12
    return math.exp(sum(math.log(max(v, floor)) for v in vals) / len(vals))


def metric_values(per_traj: Mapping[str, Mapping[str, float]],
                  metric: str) -> Dict[str, float]:
    """Pull one metric out of per-trajectory records, ready for `aggregate`.

    `per_traj` maps a trajectory name to its record (`{"max_cross": ...,
    "j_total": ...}`), as a rollout now reports directly -- `J` is accumulated
    online by `epsilon.EpsilonAccumulator`, so no trace file is involved.

    A trajectory whose record LACKS the metric is dropped, which makes
    `aggregate` return inf. That is the same all-or-nothing rule the module
    docstring argues for, and it matters most here: rollouts recorded before a
    metric existed carry no `j_total`, and silently averaging over the ones that
    do would compare candidates on different trajectory sets.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; known: {METRICS}")
    out: Dict[str, float] = {}
    for name, rec in per_traj.items():
        if metric in rec and rec[metric] is not None:
            out[name] = float(rec[metric])
    return out


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
