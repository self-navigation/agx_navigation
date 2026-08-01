"""Tests for the tuning objective aggregation.

Central claim: an evaluation that did not drive every trajectory is INVALID, not
partial. Averaging the survivors would reward a candidate for crashing the hard
rollouts, because the trajectories differ hugely in difficulty.
"""

import math

import pytest

from agx_planning.tuning.objective import aggregate

EXPECTED = ["straight", "s_curve", "corner"]


def test_mean_over_the_fixed_set():
    assert aggregate({"straight": 0.1, "s_curve": 0.2, "corner": 1.2}, EXPECTED) \
        == pytest.approx(0.5)


def test_missing_trajectory_is_invalid_not_partial():
    """The failure mode this guards: dropping the hard trajectory would give
    mean(0.1, 0.2) = 0.15, beating every honest candidate."""
    partial = {"straight": 0.1, "s_curve": 0.2}
    assert aggregate(partial, EXPECTED) == math.inf
    naive_mean = sum(partial.values()) / len(partial)
    assert naive_mean < 0.5, "sanity: the naive mean really would look better"


def test_non_finite_result_is_invalid():
    for bad in (float("nan"), float("inf"), None):
        assert aggregate({"straight": 0.1, "s_curve": bad, "corner": 1.2},
                         EXPECTED) == math.inf


def test_extra_trajectories_are_ignored():
    """Only the fixed set counts, so a stale entry cannot change the score."""
    assert aggregate({"straight": 0.1, "s_curve": 0.2, "corner": 1.2,
                      "leftover": 99.0}, EXPECTED) == pytest.approx(0.5)


def test_denominator_is_the_expected_set_not_the_dict():
    d = {n: 1.0 for n in EXPECTED}
    assert aggregate(d, EXPECTED) == pytest.approx(1.0)
    assert aggregate(d, EXPECTED[:2]) == pytest.approx(1.0)


def test_ordering_is_a_total_order_on_valid_results():
    """inf must lose to every real measurement, however bad."""
    good = aggregate({n: 5.0 for n in EXPECTED}, EXPECTED)
    bad = aggregate({"straight": 0.0}, EXPECTED)
    assert good < bad


def test_empty_expected_set_is_a_programming_error():
    with pytest.raises(ValueError):
        aggregate({"a": 1.0}, [])
