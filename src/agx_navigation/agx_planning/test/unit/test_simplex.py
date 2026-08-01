"""Tests for the Nelder-Mead minimizer.

The point of these is narrow and deliberate: prove the search MINIMIZES. A tuner
that quietly maximizes still produces a plausible-looking log, a converging
simplex and a final "best" number -- and hands back the worst gains it found.
The objective it will be driving costs ~75 s per evaluation in Gazebo, so a sign
error would not be noticed for hours of wall clock.

Pure module: no ROS, no Gazebo, no torch (see CLAUDE.md on keeping these clean).

    PYTHONPATH=src/agx_navigation/agx_planning python3 -m pytest \
      src/agx_navigation/agx_planning/test/unit/test_simplex.py -v
"""

import numpy as np
import pytest

from agx_planning.tuning.simplex import minimize


def quadratic(target):
    """Bowl with its minimum at `target` -- f > 0 everywhere except the optimum."""
    t = np.asarray(target, dtype=float)

    def f(x):
        return float(np.sum((np.asarray(x, dtype=float) - t) ** 2))

    return f


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float((1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def test_finds_minimum_of_a_bowl():
    r = minimize(quadratic([1.5, -2.0]), x0=[0.0, 0.0], step=[0.5, 0.5],
                 max_evals=200, xtol=1e-6, ftol=1e-9)
    assert r.x == pytest.approx([1.5, -2.0], abs=1e-2)
    assert r.fx < 1e-3


def test_result_is_never_worse_than_the_starting_point():
    """The anti-pessimizer test. If the search maximized, or returned the last
    vertex instead of the best, this is what would catch it."""
    f = quadratic([3.0, 3.0])
    for x0 in ([0.0, 0.0], [10.0, -10.0], [3.0, 3.0], [-5.0, 8.0]):
        r = minimize(f, x0=x0, step=[0.7, 0.7], max_evals=120)
        assert r.fx <= f(x0) + 1e-12, f"search moved AWAY from the minimum from {x0}"


def test_best_curve_is_monotonically_non_increasing():
    r = minimize(rosenbrock, x0=[-1.2, 1.0], step=[0.4, 0.4], max_evals=250)
    curve = r.best_curve
    assert len(curve) == r.n_evals
    assert all(b <= a + 1e-15 for a, b in zip(curve, curve[1:])), \
        "best-so-far increased -- the minimizer is not tracking the best point"


def test_returned_point_matches_the_best_evaluation():
    """Guards the 'return best seen, not best vertex' contract."""
    r = minimize(rosenbrock, x0=[-1.2, 1.0], step=[0.4, 0.4], max_evals=80)
    best_hist = min(fx for _, fx in r.history)
    assert r.fx == pytest.approx(best_hist)
    assert r.fx == pytest.approx(min(r.best_curve))


def test_makes_real_progress_on_rosenbrock():
    start = rosenbrock([-1.2, 1.0])
    r = minimize(rosenbrock, x0=[-1.2, 1.0], step=[0.4, 0.4], max_evals=400,
                 xtol=1e-8, ftol=1e-12)
    assert r.fx < start / 100.0
    assert r.x == pytest.approx([1.0, 1.0], abs=0.15)


def test_respects_bounds():
    """Bounds are a physical validity range: the objective must never be called
    outside them, not merely be penalised there."""
    seen = []

    def f(x):
        seen.append(np.array(x, dtype=float))
        return quadratic([5.0, 5.0])(x)

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    r = minimize(f, x0=[0.0, 0.0], step=[0.6, 0.6], bounds=bounds, max_evals=80)
    for x in seen:
        assert np.all(x >= -1.0 - 1e-12) and np.all(x <= 1.0 + 1e-12)
    # The true optimum is outside the box, so it should sit on the corner.
    assert r.x == pytest.approx([1.0, 1.0], abs=1e-2)


def test_never_exceeds_the_evaluation_budget():
    """`max_evals` counts CANDIDATE PARAMETER SETS, not steps within a rollout.

    Every evaluation drives each selected trajectory start-to-goal regardless;
    the budget only bounds how many gain pairs get tried. Nothing here ever
    truncates a trajectory, so a candidate is never judged on a partial run.
    It matters because one evaluation is ~75 s of Gazebo per trajectory.
    """
    for budget in (5, 17, 40):
        r = minimize(rosenbrock, x0=[0.0, 0.0], step=[0.3, 0.3], max_evals=budget)
        assert r.n_evals <= budget


def test_nan_objective_cannot_win():
    """A crashed rollout returns NaN, and every comparison with NaN is False --
    so an unguarded implementation adopts it as the best point and then steers
    the whole search from it."""
    def f(x):
        x = np.asarray(x, dtype=float)
        if x[0] > 0.5:
            return float("nan")
        return float(np.sum(x ** 2))

    r = minimize(f, x0=[0.0, 0.0], step=[0.4, 0.4], max_evals=60)
    assert np.isfinite(r.fx)
    assert r.fx <= f([0.0, 0.0]) + 1e-12


def test_one_dimensional_search_works():
    r = minimize(lambda x: float((x[0] - 2.0) ** 2), x0=[0.0], step=[0.5],
                 max_evals=100, xtol=1e-7, ftol=1e-10)
    assert r.x[0] == pytest.approx(2.0, abs=1e-2)


def test_callback_sees_every_evaluation():
    calls = []
    r = minimize(quadratic([1.0, 1.0]), x0=[0.0, 0.0], step=[0.5, 0.5],
                 max_evals=30, callback=lambda x, fx, best: calls.append((fx, best)))
    assert len(calls) == r.n_evals
    assert calls[-1][1] == pytest.approx(r.fx)
