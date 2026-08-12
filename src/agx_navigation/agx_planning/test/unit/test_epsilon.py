"""Tests for the epsilon (cost-functional) scorer.

Aimed at the properties a MEASUREMENT must have, not at the happy path. A scorer
that silently returns a plausible number for un-scoreable input is worse than one
that crashes, because its output looks like evidence -- the same reasoning as
`test_soak_gains.py` and `objective.py`'s refusal to average over survivors.

Pure: no ROS, no Gazebo, no skfmm.
"""

import numpy as np
import pytest

from agx_planning.tuning.epsilon import (CostWeights, EpsilonScore, compare,
                                         cost_functional)


def _score(e, u, dt=0.1, final=0.0, w=None):
    return cost_functional(e, u, dt, final, w)


# ------------------------------------------------------------------ zero point


def test_perfect_tracking_costs_nothing():
    """The load-bearing property: J = 0 for a rollout that tracks exactly with
    no correction. That is what makes J an upper bound on epsilon rather than an
    arbitrary score -- the ideal has to sit at zero."""
    s = _score(np.zeros((50, 3)), np.zeros((50, 2)))
    assert s.j_total == 0.0
    assert (s.j_tracking, s.j_control, s.j_terminal) == (0.0, 0.0, 0.0)


def test_stopping_short_is_penalised_even_with_perfect_tracking():
    """A rollout can track the path perfectly and stop short; an integral-only
    functional would score that as ideal. It must not."""
    s = _score(np.zeros((50, 3)), np.zeros((50, 2)), final=0.5)
    assert s.j_total > 0.0
    assert s.j_terminal == pytest.approx(10.0 * 0.25)


# ------------------------------------------------------------------ monotonicity


def test_more_deviation_costs_more():
    small = _score(np.full((30, 3), 0.01), np.zeros((30, 2)))
    big = _score(np.full((30, 3), 0.10), np.zeros((30, 2)))
    assert big.j_total > small.j_total


def test_more_correction_effort_costs_more():
    lazy = _score(np.zeros((30, 3)), np.zeros((30, 2)))
    busy = _score(np.zeros((30, 3)), np.full((30, 2), 0.2))
    assert busy.j_control > lazy.j_control


def test_cost_is_quadratic_in_the_error():
    """Doubling every error must quadruple the integral term -- the functional
    is quadratic, and a linear scorer would rank differently under saturation."""
    one = _score(np.full((20, 3), 0.05), np.zeros((20, 2)))
    two = _score(np.full((20, 3), 0.10), np.zeros((20, 2)))
    assert two.j_tracking == pytest.approx(4.0 * one.j_tracking)


def test_parts_sum_to_total():
    s = _score(np.full((17, 3), 0.03), np.full((17, 2), 0.04), final=0.2)
    assert s.j_total == pytest.approx(s.j_tracking + s.j_control + s.j_terminal)


# ------------------------------------------------------------------ weighting


def test_weights_are_applied_per_channel():
    """Cross-track is weighted ~3.6x LESS than along-track under the tuned gains,
    which is the whole point of the 2026-08-13 retune. If this test ever reads
    backwards, CostWeights has drifted from TVLQRConfig."""
    w = CostWeights()
    assert w.q_cross < w.q_along
    along = _score(np.tile([0.1, 0.0, 0.0], (10, 1)), np.zeros((10, 2)), w=w)
    cross = _score(np.tile([0.0, 0.1, 0.0], (10, 1)), np.zeros((10, 2)), w=w)
    assert cross.j_tracking < along.j_tracking


def test_dt_scales_the_integral_not_the_terminal():
    a = _score(np.full((10, 3), 0.1), np.zeros((10, 2)), dt=0.1, final=1.0)
    b = _score(np.full((10, 3), 0.1), np.zeros((10, 2)), dt=0.2, final=1.0)
    assert b.j_tracking == pytest.approx(2.0 * a.j_tracking)
    assert b.j_terminal == pytest.approx(a.j_terminal)


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_error_raises_rather_than_propagating(bad):
    """`nan` is EXPECTED in recorded tracking columns before the planner
    publishes, and `float('nan')` parses without raising. One NaN would turn the
    sum into NaN and be reported as a score."""
    e = np.zeros((10, 3))
    e[4, 1] = bad
    with pytest.raises(ValueError):
        _score(e, np.zeros((10, 2)))


def test_non_finite_correction_raises():
    u = np.zeros((10, 2))
    u[0, 0] = np.nan
    with pytest.raises(ValueError):
        _score(np.zeros((10, 3)), u)


def test_non_finite_final_err_raises():
    with pytest.raises(ValueError):
        _score(np.zeros((5, 3)), np.zeros((5, 2)), final=np.nan)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        _score(np.zeros((10, 3)), np.zeros((9, 2)))


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_non_positive_dt_raises(dt):
    with pytest.raises(ValueError):
        _score(np.zeros((5, 3)), np.zeros((5, 2)), dt=dt)


@pytest.mark.parametrize("shape", [(10, 2), (10, 4), (10,)])
def test_wrong_error_width_raises(shape):
    with pytest.raises(ValueError):
        _score(np.zeros(shape), np.zeros((10, 2)))


def test_wrong_correction_width_raises():
    with pytest.raises(ValueError):
        _score(np.zeros((10, 3)), np.zeros((10, 3)))


# ------------------------------------------------------------------ compare


def test_compare_delta_sign_favours_the_cheaper_arm():
    """delta > 0 must mean arm A is cheaper. A comparison helper that reports the
    sign backwards produces an identical-looking log and the wrong conclusion --
    the same failure `simplex.py`'s tests exist to rule out."""
    cheap = [EpsilonScore(1.0, 1.0, 0.0, 0.0, 10, 0.1) for _ in range(3)]
    dear = [EpsilonScore(5.0, 5.0, 0.0, 0.0, 10, 0.1) for _ in range(3)]
    out = compare(cheap, dear)
    assert out["delta"] == pytest.approx(4.0)
    assert out["mean_a"] < out["mean_b"]


def test_compare_reports_spread_not_just_means():
    """The samples are frequently bimodal, so a mean alone is not reportable."""
    arm = [EpsilonScore(v, v, 0.0, 0.0, 10, 0.1) for v in (1.0, 1.0, 5.0)]
    out = compare(arm, arm)
    assert out["sd_a"] > 0.0
    assert out["min_a"] == 1.0 and out["max_a"] == 5.0


def test_compare_requires_both_arms():
    with pytest.raises(ValueError):
        compare([], [EpsilonScore(1.0, 1.0, 0.0, 0.0, 10, 0.1)])
