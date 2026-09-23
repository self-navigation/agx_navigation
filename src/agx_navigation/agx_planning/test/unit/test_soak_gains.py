"""Tests for `soak.parse_gains`, the input validation for unattended soak runs.

The parser must reject anything it does not fully understand rather than fall
back to a default, so these tests target the failure cases. Pure: no ROS, no
Gazebo, no torch.
"""

import pytest

from agx_planning.tuning.soak import parse_gains


def test_default_is_the_two_validated_points():
    """No --gains must mean the 2026-08-12 pair, not an empty cycle."""
    gains = parse_gains(None)
    assert len(gains) == 2
    tuned, default = gains
    assert tuned[0] == pytest.approx(0.2763, abs=1e-3)
    assert tuned[1] == pytest.approx(2.6183, abs=1e-3)
    assert default == (10.0, 0.25)


def test_empty_list_also_defaults():
    assert parse_gains([]) == parse_gains(None)


def test_parses_and_preserves_order():
    """Order matters: gains are the OUTER loop, so it decides what an
    interrupted soak has balanced coverage of."""
    assert parse_gains(["1,2", "3.5,0.25"]) == [(1.0, 2.0), (3.5, 0.25)]


def test_identity_is_the_open_loop_arm():
    assert parse_gains(["identity", "2.5,2.618"]) == [(None, None), (2.5, 2.618)]


@pytest.mark.parametrize("bad", [
    "Identity",     # near-miss spelling must not silently become a gain arm
    "1",            # no comma at all
    "1,2,3",        # too many fields
    "a,2",          # non-numeric q
    "1,b",          # non-numeric r
    "",             # empty
    ",",            # both empty
])
def test_malformed_raises_rather_than_defaulting(bad):
    with pytest.raises(SystemExit):
        parse_gains([bad])


@pytest.mark.parametrize("bad", ["0,1", "1,0", "-1,2", "1,-2"])
def test_non_positive_gains_rejected(bad):
    """Both gains are positive scale factors -- the whole search runs in log10
    for that reason. A zero or negative one is a typo, not a request."""
    with pytest.raises(SystemExit):
        parse_gains([bad])


def test_one_bad_entry_rejects_the_whole_run():
    """Not 'skip the bad one and carry on': a partially-honoured --gains would
    silently collect a different experiment than the one asked for."""
    with pytest.raises(SystemExit):
        parse_gains(["1,2", "oops"])
