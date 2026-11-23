"""Check resumed FMM with the production speed and gradient pipeline."""

from pathlib import Path
import sys

import numpy as np
import pytest
import skfmm

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import field as vf
    from replanning_baseline.inputs import cell, load_suite
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


@pytest.mark.parametrize("index", range(8))
def test_resumed_march_matches_production_field(index):
    profile, snapshots, _ = load_suite(REPO / "config/replanning/field_updates.yaml")
    metadata, occupancy = snapshots[index]
    geometry = metadata["geometry"]
    resolution = geometry["resolution"]
    goal = cell(metadata["goal"], geometry)
    cfg = vf.VectorFieldConfig(**profile["field"])
    assert not cfg.early_exit_enable
    obstacles = vf._obstacle_mask(occupancy, profile["occupancy_threshold"], profile["allow_unknown"])
    prepared = vf._prepare_map(obstacles, resolution, cfg)
    phi = np.ones(occupancy.shape)
    phi[goal[1], goal[0]] = -1
    state = skfmm.TravelTimeState(np.ma.array(phi, mask=obstacles), prepared.speed, dx=resolution)
    state.advance(7)
    saved = state.checkpoint()
    state.advance()
    state.restore(saved)
    state.advance()
    travel_time = np.ma.filled(state.result(), np.nan)
    reference, _ = vf.compute_field(
        occupancy, *goal, resolution, *geometry["origin"][:2], cfg,
        occupancy_threshold=profile["occupancy_threshold"], allow_unknown=profile["allow_unknown"],
    )
    np.testing.assert_array_equal(travel_time, reference.travel_time)
    sigma = cfg.smooth_T_sigma if cfg.smooth_T_before_grad else 0.0
    channels = vf.field_from_T(travel_time, obstacles, prepared.speed, resolution, sigma)
    for actual, expected in zip(channels, (reference.grad_x, reference.grad_y, reference.grad_mag)):
        np.testing.assert_array_equal(actual, expected)
