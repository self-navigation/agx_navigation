"""Nelder-Mead simplex minimizer, pure and ROS-free so it can be unit-tested.

WHY A SIMPLEX AND NOT GRADIENT DESCENT
--------------------------------------
The objective here is "drive three recorded trajectories in Gazebo and report
the tracking error", which costs ~20 s per trajectory, has no derivative, and is
not even perfectly repeatable (see the offline-mode variance note in CLAUDE.md).
Finite-difference gradients would spend two extra evaluations per parameter to
estimate a slope from differences the measurement noise can flip the sign of.
Nelder-Mead only ever *compares* objective values, so a noise floor costs it
convergence speed rather than correctness of direction.

CONVENTION: THIS MINIMIZES.
Lower f is better, everywhere, with no sign flips anywhere in the file. The
caller is responsible for handing in a cost, not a score -- a maximization
dressed as this API is the failure mode the unit tests exist to catch.

The search runs in whatever coordinates the caller provides; `tune_tvlqr.py`
passes log-parameters so that positive gains stay positive and a step means a
ratio rather than an absolute amount.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Result:
    x: np.ndarray
    fx: float
    n_evals: int
    history: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    """Every (x, f(x)) evaluated, in order -- an expensive objective should never
    have to be re-run just to plot what the search did."""

    best_curve: List[float] = field(default_factory=list)
    """Best-so-far after each evaluation. Non-increasing BY CONSTRUCTION; a
    rising value here means the implementation is broken, and test_simplex.py
    asserts on exactly that."""


class _BudgetExhausted(Exception):
    """Raised inside the evaluator the moment the budget is spent.

    The alternative -- checking the budget at the top of the loop -- silently
    overshoots, because one iteration can evaluate a reflection AND an expansion
    or contraction after the check passed. At ~75 s per Gazebo evaluation an
    overshoot is minutes, so the limit is enforced where the spending happens.
    """


def _clip(x, bounds):
    if bounds is None:
        return x
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.minimum(np.maximum(x, lo), hi)


def minimize(
    f: Callable[[np.ndarray], float],
    x0: Sequence[float],
    step: Sequence[float] = None,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    max_evals: Optional[int] = 60,
    xtol: float = 1e-3,
    ftol: float = 1e-4,
    callback: Optional[Callable[[np.ndarray, float, float], None]] = None,
) -> Result:
    """Minimize `f` from `x0`. Returns the BEST point ever evaluated.

    `max_evals=None` means "run until the simplex converges" -- appropriate when
    the search has an unattended night to work in and a converged answer is worth
    more than a predictable finish time. Convergence (`xtol`/`ftol`) is then the
    only stopping rule, so the objective MUST be able to fail loudly rather than
    return quickly-and-wrongly forever; tune_tvlqr enforces that with a
    per-evaluation timeout and a consecutive-failure abort.

    `step` sizes the initial simplex (one vertex per dimension, offset along each
    axis). `bounds` is clipped, not penalised, so the objective is never called
    outside the box -- it is a physical validity range, not a soft preference.
    `callback(x, fx, best)` fires after every evaluation, for progress logging on
    a search that may run for hours.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    if step is None:
        step = np.maximum(np.abs(x0) * 0.1, 0.1)
    step = np.asarray(step, dtype=float)

    history: List[Tuple[np.ndarray, float]] = []
    best_curve: List[float] = []
    state = {"best": np.inf}

    def ev(x):
        if max_evals is not None and len(history) >= max_evals:
            raise _BudgetExhausted
        x = _clip(np.asarray(x, dtype=float), bounds)
        fx = float(f(x))
        # NaN would silently win every comparison it takes part in (all
        # comparisons with NaN are False), so a crashed rollout must not be
        # allowed to look like the best point in the simplex.
        if not np.isfinite(fx):
            fx = np.inf
        history.append((x.copy(), fx))
        state["best"] = min(state["best"], fx)
        best_curve.append(state["best"])
        if callback is not None:
            callback(x, fx, state["best"])
        return x, fx

    if max_evals is not None and max_evals < 1:
        raise ValueError("max_evals must allow at least one evaluation")

    try:
        _search(f, x0, n, step, bounds, xtol, ftol, ev, history)
    except _BudgetExhausted:
        pass

    # Return the best point ever SEEN, not the best current vertex: a shrink cut
    # short by the budget can leave the simplex holding worse points than one
    # already evaluated, and the caller wants the best gains found, full stop.
    bi = int(np.argmin([h[1] for h in history]))
    return Result(x=history[bi][0].copy(), fx=history[bi][1],
                  n_evals=len(history), history=history, best_curve=best_curve)


def _search(f, x0, n, step, bounds, xtol, ftol, ev, history):
    """The simplex itself. Exits by returning (converged) or by `ev` raising
    `_BudgetExhausted`; either way `minimize` reports the best point seen."""
    # --- initial simplex: x0 plus one axis-offset vertex per dimension
    pts, vals = [], []
    x, fx = ev(x0)
    pts.append(x)
    vals.append(fx)
    for i in range(n):
        y = np.array(x0, dtype=float)
        y[i] += step[i]
        y, fy = ev(y)
        pts.append(y)
        vals.append(fy)

    pts = [np.asarray(p, dtype=float) for p in pts]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    while True:
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]

        # Converged when the simplex is small in BOTH x and f. Either alone is a
        # false positive: a flat valley shrinks f while x still wanders, and a
        # noisy objective shrinks x while f jitters.
        spread_x = max(float(np.max(np.abs(p - pts[0]))) for p in pts[1:])
        spread_f = abs(vals[-1] - vals[0])
        if spread_x < xtol and spread_f < ftol:
            break

        centroid = np.mean(pts[:-1], axis=0)
        worst = pts[-1]

        xr, fr = ev(centroid + alpha * (centroid - worst))
        if fr < vals[0]:
            xe, fe = ev(centroid + gamma * (centroid - worst))
            if fe < fr:
                pts[-1], vals[-1] = xe, fe
            else:
                pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            if fr < vals[-1]:
                xc, fc = ev(centroid + rho * (xr - centroid))       # outside
                accept = fc <= fr
            else:
                xc, fc = ev(centroid + rho * (worst - centroid))    # inside
                accept = fc < vals[-1]
            if accept:
                pts[-1], vals[-1] = xc, fc
            else:
                # Shrink toward the best vertex.
                new_pts, new_vals = [pts[0]], [vals[0]]
                for p in pts[1:]:
                    xs, fs = ev(pts[0] + sigma * (p - pts[0]))
                    new_pts.append(xs)
                    new_vals.append(fs)
                pts, vals = new_pts, new_vals
