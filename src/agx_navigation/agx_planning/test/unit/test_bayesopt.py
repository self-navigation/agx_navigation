"""Tests for the GP / Bayesian-optimization tuner.

Same intent as test_simplex.py: a search that MAXIMIZES, or that reports the
luckiest sample, produces a log that looks identical to a correct one from the
outside. These tests are the only thing standing between that and a night of VM
time spent confidently finding the worst gains available.

The noise-specific tests are the point of the module. Nelder-Mead already
minimizes correctly on a clean objective; what it cannot do is rank candidates
whose measurements overlap, which is the situation this project is actually in.
"""

import math

import numpy as np
import pytest

from agx_planning.tuning.bayesopt import (
    GP, BOResult, expected_improvement, fit_gp, minimize,
)


# --- GP mechanics --------------------------------------------------------

def test_gp_interpolates_noiseless_observations():
    x = np.array([[0.0], [0.3], [0.7], [1.0]])
    y = np.array([1.0, 0.2, 0.5, 2.0])
    gp = GP(lengthscale=0.3, amplitude=1.0, noise=1e-4).fit(x, y)
    mean, sd = gp.predict(x)
    assert mean == pytest.approx(y, abs=1e-2)
    assert np.all(sd < 0.1)


def test_gp_uncertainty_grows_away_from_data():
    x = np.array([[0.0], [0.1]])
    y = np.array([0.0, 1.0])
    gp = GP(lengthscale=0.2, amplitude=1.0, noise=1e-3).fit(x, y)
    _, sd_near = gp.predict(np.array([[0.05]]))
    _, sd_far = gp.predict(np.array([[5.0]]))
    assert sd_far[0] > sd_near[0]


def test_gp_smooths_rather_than_interpolating_when_noise_is_large():
    """The whole reason for the noise term: repeated draws at ONE point must
    pull the posterior toward their average, not chase each sample."""
    x = np.array([[0.5]] * 6)
    y = np.array([1.0, 2.0, 1.0, 2.0, 1.5, 1.5])
    gp = GP(lengthscale=0.3, amplitude=1.0, noise=0.5).fit(x, y)
    mean, _ = gp.predict(np.array([[0.5]]))
    assert mean[0] == pytest.approx(1.5, abs=0.2)


def test_gp_handles_constant_observations():
    """Zero spread must not divide by zero when standardizing."""
    x = np.array([[0.0], [1.0]])
    y = np.array([2.5, 2.5])
    mean, _ = GP(0.3, 1.0, 0.1).fit(x, y).predict(np.array([[0.5]]))
    assert mean[0] == pytest.approx(2.5, abs=1e-6)


def test_fit_gp_rejects_mismatched_or_empty_input():
    with pytest.raises(ValueError, match="rows but"):
        fit_gp(np.zeros((3, 1)), np.zeros(2))
    with pytest.raises(ValueError, match="no observations"):
        fit_gp(np.zeros((0, 1)), np.zeros(0))


def test_fit_gp_honours_a_pinned_noise():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(6 * x).ravel()
    assert fit_gp(x, y, noise=0.25).noise == pytest.approx(0.25)


# --- acquisition ---------------------------------------------------------

def test_ei_is_nonnegative_and_prefers_lower_mean():
    x = np.array([[0.0], [1.0]])
    y = np.array([1.0, 0.0])
    gp = fit_gp(x, y, noise=0.01)
    xs = np.linspace(0, 1, 21).reshape(-1, 1)
    ei = expected_improvement(gp, xs, incumbent=float(y.min()))
    assert np.all(ei >= 0.0)
    # Near the good end EI should beat the bad end.
    assert ei[-3] > ei[2]


def test_ei_is_zero_at_a_known_point_no_better_than_the_incumbent():
    x = np.array([[0.0], [0.5], [1.0]])
    y = np.array([0.0, 1.0, 2.0])
    gp = fit_gp(x, y, noise=1e-4)
    ei = expected_improvement(gp, np.array([[1.0]]), incumbent=0.0)
    assert ei[0] < 1e-3


# --- the search actually minimizes --------------------------------------

def test_minimizes_a_clean_quadratic():
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return (x[0] - 0.3) ** 2 + (x[1] + 0.7) ** 2

    res = minimize(f, [(-2.0, 2.0), (-2.0, 2.0)], max_evals=40, noise=1e-3,
                   seed=0)
    assert res.x[0] == pytest.approx(0.3, abs=0.25)
    assert res.x[1] == pytest.approx(-0.7, abs=0.25)
    assert calls["n"] == res.n_evals == 40


def test_finds_the_better_of_two_basins():
    def f(x):
        return min((x[0] - 3.0) ** 2 + 1.0, (x[0] + 3.0) ** 2)

    res = minimize(f, [(-6.0, 6.0)], max_evals=35, noise=1e-3, seed=1)
    assert res.x[0] == pytest.approx(-3.0, abs=0.7)


def test_does_not_maximize():
    """The failure that looks identical from the outside."""
    res = minimize(lambda x: x[0], [(-1.0, 1.0)], max_evals=25, noise=1e-3,
                   seed=0)
    assert res.x[0] < -0.5
    assert res.fx < 0.0


def test_respects_bounds_and_the_evaluation_budget():
    seen = []

    def f(x):
        seen.append(x[0])
        return abs(x[0] - 10.0)          # optimum lies OUTSIDE the box

    res = minimize(f, [(-1.0, 1.0)], max_evals=15, seed=0)
    assert all(-1.0 <= v <= 1.0 for v in seen)
    assert len(seen) == 15
    assert res.x[0] == pytest.approx(1.0, abs=0.15)


def test_x0_is_evaluated_first():
    seen = []
    minimize(lambda x: (seen.append(tuple(x)), x[0] ** 2)[1],
             [(-2.0, 2.0)], max_evals=6, n_init=3, x0=[1.234], seed=0)
    assert seen[0][0] == pytest.approx(1.234)


def test_bad_bounds_raise():
    with pytest.raises(ValueError, match="lo < hi"):
        minimize(lambda x: x[0], [(1.0, 1.0)], max_evals=5)
    with pytest.raises(ValueError, match="sequence of"):
        minimize(lambda x: x[0], [1.0, 2.0], max_evals=5)


def test_non_finite_objective_does_not_break_the_fit():
    """A failed rollout returns inf; the surrogate must survive it."""
    def f(x):
        return math.inf if x[0] > 0.5 else (x[0] - 0.1) ** 2

    res = minimize(f, [(-1.0, 1.0)], max_evals=25, noise=1e-2, seed=3)
    assert math.isfinite(res.fx)
    assert res.x[0] < 0.5


# --- the noise-robustness properties that motivated the module ----------

def test_recommendation_reduces_the_TAIL_of_location_error():
    """What the posterior mean actually buys for the LOCATION, measured.

    The tempting claim is "the posterior mean lands closer to the optimum than
    the luckiest sample does". MEASURED OVER 30 SEEDS, THAT IS BARELY TRUE: it
    wins about half the time (0.53 on a quadratic, 0.47 at low noise), because
    EI concentrates its samples near the optimum anyway, so the best observed
    point is already in the right neighbourhood.

    What IS reproducible is the tail. On a FLAT objective -- where noise, not
    curvature, decides which sample looks best, i.e. our situation -- the
    posterior avoids the large misses:

        seeds   0-19 : p90 error 0.878 (posterior) vs 1.021 (best observed)
        seeds 100-119: p90 error 0.789 (posterior) vs 0.876 (best observed)

    So the honest claim is: no worse typically, materially safer at the tail.
    The unambiguous win is on the reported VALUE -- see the next test.
    """
    post_err, obs_err = [], []
    for seed in range(20):
        rng = np.random.default_rng(5000 + seed)
        res = minimize(lambda x: 0.2 * (x[0] - 0.3) ** 2 + rng.normal(0.0, 0.30),
                       [(-2.0, 2.0)], max_evals=40, noise=0.35, seed=seed)
        post_err.append(abs(res.x[0] - 0.3))
        obs_err.append(abs(res.x_observed[0] - 0.3))
    post_err, obs_err = np.array(post_err), np.array(obs_err)
    assert np.percentile(post_err, 90) < np.percentile(obs_err, 90)
    assert post_err.mean() <= obs_err.mean()


def test_reported_value_is_not_the_optimistic_observed_minimum():
    """Winner's curse, stated as a test: with heavy noise around a flat
    function, the best observed draw is biased low and the posterior mean should
    not inherit that bias."""
    rng = np.random.default_rng(7)
    res = minimize(lambda x: 1.0 + rng.normal(0.0, 0.5), [(-1.0, 1.0)],
                   max_evals=30, noise=0.5, seed=0)
    assert res.fx > res.fx_observed          # not the lucky draw
    assert res.fx == pytest.approx(1.0, abs=0.4)   # near the truth


def test_repeated_evaluations_do_not_derail_the_search():
    """Nelder-Mead's failure mode: re-sampling one point forever. The GP should
    still recover the optimum when many draws land at the same place."""
    rng = np.random.default_rng(3)

    def f(x):
        return (x[0] - 0.5) ** 2 + rng.normal(0.0, 0.2)

    res = minimize(f, [(-2.0, 2.0)], max_evals=45, noise=0.25, seed=2)
    assert res.x[0] == pytest.approx(0.5, abs=0.6)


def test_result_records_every_evaluation_for_later_reanalysis():
    res = minimize(lambda x: x[0] ** 2, [(-1.0, 1.0)], max_evals=12, seed=0)
    assert isinstance(res, BOResult)
    assert len(res.xs) == len(res.ys) == 12
    assert all(len(x) == 1 for x in res.xs)


def test_two_dimensional_noisy_problem():
    rng = np.random.default_rng(11)

    def f(x):
        return (x[0] - 1.0) ** 2 + (x[1] - 0.5) ** 2 + rng.normal(0.0, 0.15)

    res = minimize(f, [(-3.0, 3.0), (-3.0, 3.0)], max_evals=60, noise=0.2,
                   seed=5)
    assert math.hypot(res.x[0] - 1.0, res.x[1] - 0.5) < 1.0
