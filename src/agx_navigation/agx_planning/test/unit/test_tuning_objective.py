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


# --- repeats -------------------------------------------------------------
# These guard the property that made repeats necessary in the first place: a
# candidate must never look BETTER because one of its rollouts died.

from agx_planning.tuning.objective import median, reduce_repeats  # noqa: E402


def test_median_of_odd_and_even():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 2.0, 3.0]) == 2.5


def test_median_poisons_on_non_finite():
    assert median([1.0, 2.0, float("inf")]) == float("inf")
    assert median([1.0, float("nan")]) == float("inf")


def test_reduce_mean_is_the_default():
    reps = [{"a": 1.0}, {"a": 2.0}, {"a": 6.0}]
    assert reduce_repeats(reps, ["a"]) == {"a": 3.0}
    assert reduce_repeats(reps, ["a"], how="median") == {"a": 2.0}
    assert reduce_repeats(reps, ["a"], how="max") == {"a": 6.0}


def test_trajectory_missing_from_any_repeat_is_dropped_entirely():
    """The all-or-nothing rule, extended across repeats: one dead rollout must
    invalidate the candidate, not silently average the two that survived."""
    reps = [{"a": 1.0, "b": 2.0}, {"a": 1.0}, {"a": 1.0, "b": 2.0}]
    out = reduce_repeats(reps, ["a", "b"])
    assert "b" not in out
    assert aggregate(out, ["a", "b"]) == float("inf")


def test_non_finite_in_one_repeat_poisons_that_trajectory():
    reps = [{"a": 1.0}, {"a": float("inf")}]
    assert aggregate(reduce_repeats(reps, ["a"]), ["a"]) == float("inf")


def test_reduce_rejects_unknown_reduction_and_empty_repeats():
    import pytest
    with pytest.raises(ValueError, match="unknown reduction"):
        reduce_repeats([{"a": 1.0}], ["a"], how="geometric")
    with pytest.raises(ValueError, match="no repeats"):
        reduce_repeats([], ["a"])


def test_single_repeat_reproduces_old_behaviour():
    per = {"a": 1.0, "b": 3.0}
    assert reduce_repeats([per], ["a", "b"]) == per
    assert aggregate(reduce_repeats([per], ["a", "b"]), ["a", "b"]) == 2.0
