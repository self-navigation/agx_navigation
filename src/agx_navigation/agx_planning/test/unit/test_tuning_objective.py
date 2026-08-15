"""Tests for the tuning objective aggregation.

Central claim: an evaluation that did not drive every trajectory is INVALID, not
partial. Averaging the survivors would reward a candidate for crashing the hard
rollouts, because the trajectories differ hugely in difficulty.
"""

import math

import pytest

from agx_planning.tuning import objective
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


# ------------------------------------------------ metric selection and J's mean


def test_geometric_aggregate_is_not_dominated_by_one_trajectory():
    """The reason `J` needs a different aggregator, as a test rather than a
    docstring. One plan 100x worse than the rest moves the arithmetic mean by
    ~14x and the geometric mean by ~1.6x -- so an arithmetic search on J would
    be tuning to whichever plan is worst."""
    base = {f"t{i}": 1.0 for i in range(7)}
    spiked = dict(base, t0=100.0)
    names = list(base)
    a_base = objective.aggregate(base, names, how="arithmetic")
    a_spike = objective.aggregate(spiked, names, how="arithmetic")
    g_base = objective.aggregate(base, names, how="geometric")
    g_spike = objective.aggregate(spiked, names, how="geometric")
    assert a_spike / a_base > 10.0
    assert g_spike / g_base < 2.0


def test_geometric_aggregate_matches_hand_computation():
    vals = {"a": 1.0, "b": 4.0, "c": 16.0}
    got = objective.aggregate(vals, ["a", "b", "c"], how="geometric")
    assert got == pytest.approx(4.0)  # cube root of 64


def test_geometric_survives_a_perfect_trajectory():
    """A metric of exactly 0 is a flawless rollout, not bad input. It must not
    send the whole aggregate to zero, which would make every candidate that
    nails one plan look equally perfect."""
    got = objective.aggregate({"a": 0.0, "b": 1.0}, ["a", "b"], how="geometric")
    assert math.isfinite(got)
    assert got > 0.0


def test_geometric_still_invalidates_on_a_missing_trajectory():
    """The all-or-nothing rule is not weakened by the new aggregator."""
    assert objective.aggregate({"a": 1.0}, ["a", "b"], how="geometric") == math.inf


def test_unknown_aggregator_raises():
    with pytest.raises(ValueError, match="unknown aggregator"):
        objective.aggregate({"a": 1.0}, ["a"], how="rms")


def test_metric_values_selects_and_drops_records_lacking_it():
    """A rollout recorded before J existed carries no j_total. Dropping it makes
    `aggregate` return inf, rather than silently comparing candidates on
    different trajectory sets."""
    recs = {"a": {"max_cross": 0.5, "j_total": 12.0},
            "b": {"max_cross": 0.2}}
    assert objective.metric_values(recs, "max_cross") == {"a": 0.5, "b": 0.2}
    assert objective.metric_values(recs, "j_total") == {"a": 12.0}
    assert objective.aggregate(objective.metric_values(recs, "j_total"),
                               ["a", "b"], how="geometric") == math.inf


def test_metric_values_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        objective.metric_values({"a": {"max_cross": 1.0}}, "rms_cross")


def test_every_metric_has_a_default_aggregator():
    """A metric added without an aggregator would silently get the arithmetic
    one, which is wrong for anything with J's dynamic range."""
    assert set(objective.METRICS) == set(objective.DEFAULT_HOW)
