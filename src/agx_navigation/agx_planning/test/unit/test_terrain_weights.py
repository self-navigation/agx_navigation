"""Weighted profile sampling for the terrain randomizers.

These exist because the failure mode is SILENT: an unweighted or mis-weighted
draw produces patches that look fine in the log and on screen, and only shows up
as a controller tuned for a floor that does not exist. The uniform draw this
replaced put half of every patch set at or below tyre-on-ice friction.
"""

import numpy as np
import pytest

from agx_planning.rl_corrector.terrain import (
    _weighted_profiles, along_path_terrain_sampler, ground_friction_sampler,
)
from rudn_ordjo_building.surface_patches import DEFAULT_PATCH_WEIGHTS, PROFILES


POSES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])


def test_every_default_weight_names_a_real_profile():
    unknown = [n for n in DEFAULT_PATCH_WEIGHTS if n not in PROFILES]
    assert unknown == []


def test_defaults_are_majority_realistic_floor():
    """The whole point of the reweighting: ice is the exception, not the norm."""
    realistic = sum(w for n, w in DEFAULT_PATCH_WEIGHTS.items()
                    if PROFILES[n].mu >= 0.30)
    total = sum(DEFAULT_PATCH_WEIGHTS.values())
    assert realistic / total > 0.5
    assert DEFAULT_PATCH_WEIGHTS["icy"] / total <= 0.15


def test_probabilities_normalize():
    names, probs = _weighted_profiles(["icy", "linoleum"],
                                      {"icy": 1.0, "linoleum": 3.0})
    assert names == ["icy", "linoleum"]
    assert probs == pytest.approx([0.25, 0.75])


def test_dict_form_is_accepted():
    names, probs = _weighted_profiles({"icy": 2.0, "wet_tile": 2.0}, {})
    assert sorted(names) == ["icy", "wet_tile"]
    assert probs == pytest.approx([0.5, 0.5])


def test_missing_weight_raises_rather_than_silently_never_drawing():
    with pytest.raises(ValueError, match="no sampling weight"):
        _weighted_profiles(["linoleum"], {"icy": 1.0})


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown terrain profile"):
        _weighted_profiles({"no_such_surface": 1.0}, {})


def test_degenerate_weights_raise():
    with pytest.raises(ValueError, match="non-negative"):
        _weighted_profiles({"icy": 0.0, "linoleum": 0.0}, {})
    with pytest.raises(ValueError, match="non-negative"):
        _weighted_profiles({"icy": -1.0, "linoleum": 2.0}, {})


def test_along_path_sampler_respects_weights():
    """A weight of 0 must never be drawn -- the empirical check that the
    probabilities reach rng.choice at all, not just that they were computed."""
    sampler = along_path_terrain_sampler(
        POSES, profiles={"linoleum": 1.0, "icy": 0.0}, n_range=(3, 3))
    rng = np.random.default_rng(0)
    drawn = {p["profile"] for _ in range(50) for p in sampler(rng)}
    assert drawn == {"linoleum"}


def test_ground_sampler_respects_weights_and_names_the_ground():
    sampler = ground_friction_sampler(POSES, profiles={"wet_tile": 1.0})
    rng = np.random.default_rng(0)
    for _ in range(10):
        base = sampler(rng)[0]
        assert base["name"] == "rl_ground"
        assert base["profile"] == "wet_tile"


def test_ground_sampler_excludes_directional_by_default():
    """A whole floor that grips one axis and slides the other is not a surface
    that exists; the directional profiles are a LOCAL-patch idea only."""
    sampler = ground_friction_sampler(POSES)
    rng = np.random.default_rng(1)
    drawn = {sampler(rng)[0]["profile"] for _ in range(300)}
    assert not any(n.startswith("directional_") for n in drawn)


def test_default_draw_is_dominated_by_realistic_surfaces():
    sampler = along_path_terrain_sampler(POSES, n_range=(1, 1))
    rng = np.random.default_rng(7)
    drawn = [sampler(rng)[0]["profile"] for _ in range(2000)]
    icy_frac = drawn.count("icy") / len(drawn)
    good_frac = sum(1 for d in drawn if PROFILES[d].mu >= 0.30) / len(drawn)
    assert icy_frac < 0.15
    assert good_frac > 0.5
