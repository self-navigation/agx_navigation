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

REPEATS AND THE MEDIAN (added 2026-08-04)
-----------------------------------------
The objective is STOCHASTIC, and not in a friendly way. Re-measuring one gain
pair 26 times (from the 2026-08-03 tuning log) shows the noise is confined to a
minority of trajectories and is BIMODAL, not Gaussian:

    floor_6_00018  sd 0.000     floor_6_00023  sd 0.174
    floor_6_00031  sd 0.001     floor_6_00056  sd 0.362
    floor_6_00025  sd 0.012     floor_1_00049  sd 0.530
    floor_6_00047  sd 0.157

    floor_1_00049: 0.295 0.297 ... 0.353 0.368 0.488 | 1.582 1.604 1.682 1.745
                   <------------ 21 draws ---------> | <--- 5 draws, ~3x --->

Five of seven trajectories are effectively deterministic; the other two switch
into a high mode about 19% of the time. The outliers are NOT correlated across
trajectories within an evaluation, so this is per-rollout mode switching, not a
contaminated evaluation.

Bimodal contamination argues for a robust estimator, so the median was the
obvious choice. IT LOSES -- measured, not assumed, by resampling the 26 real
repeats above (4000 draws per cell):

    estimator        sd of the estimate    cost/eval
    single sample                0.0933         35 s
    median-of-3                  0.0692        105 s
    mean-of-3                    0.0547        105 s
    median-of-5                  0.0533        175 s
    mean-of-5                    0.0413        175 s

The mean wins at every budget, and it is exactly sqrt(n) averaging
(0.0933/sqrt(3) = 0.054, 0.0933/sqrt(5) = 0.042). The reason the robustness
argument fails: the aggregate ALREADY averages over 7 trajectories, of which at
most two are contaminated on any given repeat, so the outlier is diluted 7-fold
before the estimator ever sees it -- while the median pays its usual variance
penalty in full. Robust estimators earn their keep against contamination that
survives aggregation; this does not.

So `how="mean"` is the default and the right choice for SEARCH. The median is
kept because it is cheap to have and is the better diagnostic when looking at a
SINGLE trajectory (where there is no aggregation to dilute the outlier) --
notably in `--per-traj` analyses of the log.

Budget guidance: mean-of-3 (sd 0.055) resolves the ~0.2 m effect a real gain
improvement would produce; use mean-of-5 to VALIDATE a winner, where the extra
precision is cheap because only a few candidates need it.

SCORING IN `J`, AND WHY IT NEEDS A DIFFERENT AGGREGATOR (added 2026-08-15)
--------------------------------------------------------------------------
`max|e_cross|` is a tracking error in metres; `J` (see `epsilon.py`) is the cost
functional the advisor's SVCM framework is stated in, so it is the quantity a
claim should be made in. `metric_values` selects between them, and the tuner can
now search either.

**They cannot share an aggregator.** `max|e_cross|` is bounded and comparable
across plans -- a hard plan costs 2 m where an easy one costs 0.05 m, a factor of
40. `J` is an INTEGRAL of a quadratic, so across the 51-plan library sweep it
spans 0.2 to 1043, a factor of 5000. Measured on that sweep
(`epsilon_data/libsweep_J.jsonl`, 51 plans, both arms):

    aggregator    default    tuned    ratio    agrees with per-plan win rate?
    mean            42.26    10.40     4.06    NO  -- 45/51 plans won, ratio 1.6
    median           8.14     6.64     1.23    understates
    geometric        9.72     5.14     1.89    YES

The plain mean is a ONE-PLAN objective: `floor_6_00031` alone contributes 1043 of
the default arm's 2155 total, i.e. **48% of the aggregate comes from 2% of the
plans**. Optimizing that mean would tune the corrector to whichever plan happens
to be worst, which is precisely the overfitting the broad plan library was built
to escape.

So `how="geometric"` is the default for `J`: the geometric mean is a mean of
logs, so it scores a 2x improvement on an easy plan the same as a 2x improvement
on a hard one, which is the intent -- "make every plan better" rather than
"rescue the worst plan". It also matches the per-plan win rate, which is the
model-free check on any aggregator.

Keep `how="arithmetic"` for `max|e_cross|`, where it is already validated and
where the metric's bounded range makes the outlier problem much milder.

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
