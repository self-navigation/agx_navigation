"""Tests for the candidate-screening geometry.

These are aimed at the two things that have actually gone wrong before:

  * the pivot dominating every descriptor, which mislabelled 58 of 100 plans;
  * a screen that admits straight corridors, which is how the current library
    ended up mostly straight lines.

Everything here is pure geometry -- no ROS, no Gazebo, no map files.
"""

import math

import pytest

from agx_planning.tuning.shape import (
    Candidate,
    Descriptors,
    descriptors,
    interest_score,
    label,
    line_blocked,
    pivot_demand,
    resample,
    stratify,
    trim_pivot,
)


def straight(n=60, step=0.2):
    return [(i * step, 0.0) for i in range(n)]


def arc(total_turn, n=80, radius=3.0):
    """A constant-curvature arc turning `total_turn` radians."""
    return [(radius * math.sin(t), radius * (1 - math.cos(t)))
            for t in (total_turn * i / (n - 1) for i in range(n))]


def s_curve(n=160, radius=3.0):
    left = arc(math.pi / 2, n // 2, radius)
    x0, y0 = left[-1]
    # Mirror the second half so the net turn cancels but the gross turn does not.
    right = [(x0 + radius * (1 - math.cos(t)), y0 + radius * math.sin(t))
             for t in (math.pi / 2 * i / (n // 2 - 1) for i in range(n // 2))]
    return left + right


def pivot_then(path, n_pivot=24, turn=2.8, travel=0.7):
    """Prepend a realistic PMP opening pivot.

    Measured from the real library (floor_6_00031 / 00047): the opening turns
    ~2.8 rad within the first ~0.7 m of travel -- a very tight arc, NOT a pure
    spin. That distinction is what the test needs: a zero-displacement spin is
    collapsed by arc-length resampling on its own and would prove nothing,
    whereas this survives resampling and corrupts the descriptors, which is the
    behaviour trimming exists to fix.
    """
    radius = travel / turn
    x0, y0 = path[0]
    # Arc arriving at (x0, y0) heading +x, so the straight part continues cleanly.
    arc_pts = []
    for i in range(n_pivot):
        t = turn * (i / (n_pivot - 1) - 1.0)      # sweeps [-turn, 0]
        arc_pts.append((x0 + radius * math.sin(t), y0 + radius * (math.cos(t) - 1.0)))
    return arc_pts + path


# --------------------------------------------------------------- resample


def test_resample_preserves_endpoints():
    pts = straight()
    out = resample(pts, step=0.5)
    assert out[0] == pts[0]
    assert out[-1] == pts[-1]


def test_resample_is_insensitive_to_input_density():
    """The whole point: an FM2 streamline and a PMP rollout sample differently."""
    # Same 12 m line, sampled 8x apart -- only the density differs. Trimming is
    # off because it lands on the nearest SAMPLE to the threshold, so it is
    # density-dependent by up to one sample spacing; that is inherent and
    # irrelevant here, where the question is whether resampling itself is.
    coarse = descriptors([(i * 0.4, 0.0) for i in range(31)], trim=False)
    fine = descriptors([(i * 0.05, 0.0) for i in range(241)], trim=False)
    assert coarse is not None and fine is not None
    assert coarse.length == pytest.approx(fine.length, abs=0.05)
    assert coarse.straightness == pytest.approx(fine.straightness, abs=1e-6)


# --------------------------------------------------------------- trim_pivot


def test_trim_pivot_drops_the_spin():
    path = straight()
    trimmed = trim_pivot(pivot_then(path))
    # The spin contributes ~no travel, so trimming lands at the real start.
    assert trimmed[0][0] == pytest.approx(0.0, abs=0.35)
    assert len(trimmed) < len(pivot_then(path))


def test_trim_pivot_keeps_a_path_that_never_travels():
    """Degenerate input must come back unchanged, not empty."""
    stuck = [(0.0, 0.0)] * 10
    assert trim_pivot(stuck) == [(0.0, 0.0)] * 10


def test_pivot_does_not_change_the_label():
    """The regression that mislabelled 58 of 100 plans as CORNER.

    Uses the measured pivot profile, so it also pins PIVOT_TRAVEL_M: at the
    inherited 0.30 m this fails, because a real pivot extends to ~0.7 m and
    half of it survived the trim and read as a corner.
    """
    path = straight()
    assert label(descriptors(path)) == "STRAIGHT"
    assert label(descriptors(pivot_then(path))) == "STRAIGHT"


def test_trim_threshold_covers_the_measured_pivot():
    """Guards the constant against being 'tidied' back down to 0.30."""
    from agx_planning.tuning.shape import PIVOT_TRAVEL_M
    assert PIVOT_TRAVEL_M >= 0.7


def test_untrimmed_descriptors_are_corrupted_by_the_pivot():
    """Documents WHY trimming is the default -- trim=False reproduces the bug."""
    raw = descriptors(pivot_then(straight()), trim=False)
    trimmed = descriptors(pivot_then(straight()), trim=True)
    assert raw.total_abs_turn > trimmed.total_abs_turn


# --------------------------------------------------------------- descriptors


def test_straight_line_descriptors():
    d = descriptors(straight())
    assert d.straightness == pytest.approx(1.0, abs=1e-6)
    assert d.total_abs_turn < 1e-6
    assert d.sign_changes == 0


def test_corner_has_net_turn_equal_to_gross():
    d = descriptors(arc(math.pi / 2))
    assert d.net_turn == pytest.approx(d.total_abs_turn, rel=1e-3)
    assert label(d) == "CORNER"


def test_s_curve_cancels_net_turn_but_not_gross():
    d = descriptors(s_curve())
    assert d.total_abs_turn > 2.0
    assert abs(d.net_turn) < 0.5 * d.total_abs_turn
    assert d.sign_changes >= 1
    assert label(d) == "S"


def test_uturn_is_not_labelled_a_corner():
    """A U-turn passes every corner test too, so ordering in `label` matters."""
    d = descriptors(arc(math.pi, n=120))
    assert label(d) == "UTURN"


# --------------------------------------------------------------- line_blocked


def clear_grid(w=40, h=40):
    return [[False] * w for _ in range(h)]


def test_line_blocked_false_on_empty_map():
    assert not line_blocked((1.0, 1.0), (5.0, 5.0), clear_grid(), (0.0, 0.0), 0.5)


def test_line_blocked_true_through_a_wall():
    g = clear_grid()
    for y in range(40):
        g[y][10] = True                      # vertical wall at x = 5.0 m
    assert line_blocked((1.0, 1.0), (9.0, 1.0), g, (0.0, 0.0), 0.5)


def test_line_blocked_respects_a_doorway():
    g = clear_grid()
    for y in range(40):
        g[y][10] = True
    g[2][10] = False                         # a gap on the line's own row
    assert not line_blocked((1.0, 1.25), (9.0, 1.25), g, (0.0, 0.0), 0.5)


def test_diagonal_cannot_squeeze_through_a_corner():
    """Supercover, not Bresenham: clipping a wall corner counts as blocked."""
    g = clear_grid()
    g[5][5] = True
    g[6][6] = True
    assert line_blocked((2.25, 2.25), (3.75, 3.75), g, (0.0, 0.0), 0.5)


def test_off_map_counts_as_blocked():
    assert line_blocked((1.0, 1.0), (100.0, 100.0), clear_grid(), (0.0, 0.0), 0.5)


# --------------------------------------------------------------- pivot_demand


def test_pivot_demand_zero_when_already_aligned():
    path = straight()
    start_p, goal_p = pivot_demand(0.0, 0.0, path)
    assert start_p == pytest.approx(0.0, abs=1e-6)
    assert goal_p == pytest.approx(0.0, abs=1e-6)


def test_pivot_demand_detects_a_reversed_start():
    start_p, _ = pivot_demand(math.pi, 0.0, straight())
    assert start_p == pytest.approx(math.pi, abs=1e-6)


def test_pivot_demand_is_bounded_by_pi():
    for theta in (-3.0, -1.0, 0.5, 2.9, 3.1):
        s, g = pivot_demand(theta, -theta, straight())
        assert 0.0 <= s <= math.pi + 1e-9
        assert 0.0 <= g <= math.pi + 1e-9


# --------------------------------------------------------------- scoring


def make_candidate(**kw):
    base = dict(
        start=(0.0, 0.0), goal=(10.0, 0.0), start_theta=0.0, goal_theta=0.0,
        blocked=False, detour=1.0, desc=descriptors(straight()),
        start_pivot=0.0, goal_pivot=0.0, shape="STRAIGHT",
    )
    base.update(kw)
    return Candidate(**base)


def test_blocked_line_of_sight_raises_the_score():
    assert interest_score(make_candidate(blocked=True)) > \
           interest_score(make_candidate(blocked=False))


def test_a_straight_clear_run_scores_near_zero():
    """The shape the current library is full of must rank last."""
    assert interest_score(make_candidate()) == pytest.approx(0.0, abs=1e-6)


def test_score_terms_are_clipped():
    """No single runaway term may dominate the ranking."""
    wild = make_candidate(detour=100.0, start_pivot=math.pi, goal_pivot=math.pi)
    assert interest_score(wild) < 10.0


def test_a_tortuous_candidate_outranks_a_blocked_straight_one():
    tortuous = make_candidate(desc=descriptors(s_curve()), shape="S", detour=1.4)
    assert interest_score(tortuous) > interest_score(make_candidate(blocked=True))


# --------------------------------------------------------------- stratify


def test_stratify_takes_the_best_of_each_shape():
    cands = (
        [make_candidate(shape="S", detour=1.0 + 0.1 * i) for i in range(5)]
        + [make_candidate(shape="CORNER", detour=1.0 + 0.1 * i) for i in range(5)]
    )
    picked = stratify(cands, per_shape=2)
    assert len(picked) == 4
    assert {c.shape for c in picked} == {"S", "CORNER"}
    # Best-scoring first within each bucket.
    s_picks = [c for c in picked if c.shape == "S"]
    assert s_picks[0].score >= s_picks[1].score


def test_stratify_returns_a_short_bucket_rather_than_padding():
    """A set that quietly substitutes 3 corners for a missing U-turn is the
    failure this whole module exists to prevent."""
    cands = [make_candidate(shape="S")] + [make_candidate(shape="CORNER")] * 5
    picked = stratify(cands, per_shape=3)
    assert len([c for c in picked if c.shape == "S"]) == 1
    assert len([c for c in picked if c.shape == "CORNER"]) == 3


# ----------------------------------------------------------- screen_score


def test_screen_score_ignores_turning():
    """A cheap route predicts turning at corr ~+0.3, so screening must not use
    it. Two candidates differing ONLY in predicted shape must rank equally."""
    from agx_planning.tuning.shape import screen_score
    straight_c = make_candidate(desc=descriptors(straight()))
    wiggly = make_candidate(desc=descriptors(s_curve()))
    assert screen_score(straight_c) == pytest.approx(screen_score(wiggly))


def test_screen_score_uses_the_reliable_signals():
    from agx_planning.tuning.shape import screen_score
    base = make_candidate()
    assert screen_score(make_candidate(blocked=True)) > screen_score(base)
    assert screen_score(make_candidate(detour=1.5)) > screen_score(base)
    assert screen_score(make_candidate(start_pivot=math.pi)) > screen_score(base)
