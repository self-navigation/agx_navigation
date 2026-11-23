"""Field reuse must match a fresh solve for each request."""

from dataclasses import fields, replace
import sys
from unittest.mock import patch

import numpy as np
import pytest

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from agx_planning.vector_field import field as vf
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


@pytest.fixture
def field_request():
    occupancy = np.zeros((41, 61), dtype=np.int8)
    occupancy[[0, -1], :] = 100
    occupancy[:, [0, -1]] = 100
    occupancy[1:15, 30] = 100
    occupancy[25:-1, 30] = 100
    return dict(map_array=occupancy, goal_col=48, goal_row=20,
                resolution=0.1, origin_x=-3.0, origin_y=-2.0,
                cfg=vf.VectorFieldConfig(early_exit_enable=False),
                robot_col=12, robot_row=20)


def assert_same(actual, expected):
    assert (actual is None) == (expected is None)
    if actual is None:
        return
    assert actual[1] == expected[1]
    for item in fields(vf.VectorFieldResult):
        a, b = getattr(actual[0], item.name), getattr(expected[0], item.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=item.name)
        else:
            assert a == b, item.name


@pytest.mark.parametrize("config", [
    {},
    {"speed_profile": "exponential"},
    {"centring": True, "speed_profile": "exponential", "speed_decay_rate": 1.5},
    {"centring": True, "centring_smooth": 0.15,
     "smooth_T_before_grad": True, "smooth_T_sigma": 0.12},
    {"bottleneck_align": True},
    {"early_exit_enable": True},
])
def test_update_sequence_matches_full_solve(field_request, config):
    field_request["cfg"] = replace(field_request["cfg"], **config)
    updater = FieldUpdater()
    rng = np.random.default_rng(17)
    for step in range(10):
        if step in (2, 3):
            field_request["goal_row"] = 22
        elif step == 4:
            field_request["map_array"][15:26, 30] = 100
        elif step == 5:
            field_request["map_array"][15:26, 30] = 0
        elif step > 5:
            rows = rng.integers(2, 39, 12)
            cols = rng.integers(20, 40, 12)
            field_request["map_array"][rows, cols] = rng.choice([-1, 0, 100], 12)
        actual = updater.update(**field_request)
        assert_same(actual, vf.compute_field(**field_request))
        expected = ("field_reuse" if step in (1, 3) else
                    "map_reuse" if step == 2 else "full_recompute")
        assert updater.last_update == expected


def test_reuse_skips_the_expected_stages(field_request):
    updater = FieldUpdater()
    with patch.object(vf, "_prepare_map", wraps=vf._prepare_map) as prepare, \
            patch.object(vf, "solve_eikonal", wraps=vf.solve_eikonal) as solve:
        updater.update(**field_request)
        updater.update(**field_request)
        assert (prepare.call_count, solve.call_count) == (1, 1)
        field_request["goal_col"] -= 1
        updater.update(**field_request)
        assert (prepare.call_count, solve.call_count) == (1, 2)
        field_request["map_array"][10, 10] = 100
        updater.update(**field_request)
        assert (prepare.call_count, solve.call_count) == (2, 3)


def test_occupancy_changes_with_the_same_mask_reuse_field(field_request):
    updater = FieldUpdater()
    field_request["allow_unknown"] = True
    expected = updater.update(**field_request)
    field_request["map_array"][10, 10] = -1
    field_request["map_array"][10, 11] = 64
    field_request["map_array"][0, 0] = 65
    assert_same(updater.update(**field_request), expected)
    assert updater.last_update == "field_reuse"


@pytest.mark.parametrize("change", [
    {"resolution": 0.2}, {"origin_x": -4.0}, {"origin_y": 1.0},
    {"occupancy_threshold": 40}, {"allow_unknown": True},
])
def test_geometry_and_policy_changes_invalidate(field_request, change):
    updater = FieldUpdater()
    updater.update(**field_request)
    field_request.update(change)
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_update == "full_recompute"


def test_config_mutation_and_shape_change_invalidate(field_request):
    updater = FieldUpdater()
    updater.update(**field_request)
    field_request["cfg"].centring = True
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_update == "full_recompute"
    field_request["map_array"] = np.pad(field_request["map_array"], ((0, 2), (0, 2)), constant_values=100)
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_update == "full_recompute"


@pytest.mark.parametrize("early_exit", [False, True])
def test_robot_motion_uses_the_requested_domain(field_request, early_exit):
    field_request["cfg"].early_exit_enable = early_exit
    updater = FieldUpdater()
    updater.update(**field_request)
    field_request["robot_col"] += 20
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_update == ("map_reuse" if early_exit else "field_reuse")


def test_returned_data_cannot_change_the_cache(field_request):
    field_request["cfg"].bottleneck_align = True
    updater = FieldUpdater()
    reference = vf.compute_field(**field_request)
    for _ in range(2):
        outcome = updater.update(**field_request)
        assert_same(outcome, reference)
        result = outcome[0]
        result.origin_x = 999.0
        for item in fields(vf.VectorFieldResult):
            value = getattr(result, item.name)
            if isinstance(value, np.ndarray):
                value[:] = 999.0
        assert result.bottlenecks
        result.bottlenecks[0][0].width = 999.0
        result.bottlenecks.clear()
    assert_same(updater.update(**field_request), reference)


@pytest.mark.parametrize("change", [
    {"goal_col": -1}, {"goal_row": 41}, {"goal_col": 0},
    {"robot_col": None}, {"robot_row": -1}, {"robot_row": 0},
])
def test_invalid_request_clears_cached_result(field_request, change):
    field_request["cfg"].early_exit_enable = True
    updater = FieldUpdater()
    updater.update(**field_request)
    invalid = {**field_request, **change}
    assert updater.update(**invalid) is None
    assert updater.last_update == "invalid"
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_update == "full_recompute"


def test_failed_solve_and_explicit_clear_discard_cache(field_request):
    updater = FieldUpdater()
    updater.update(**field_request)
    field_request["goal_col"] -= 1
    with patch.object(vf, "solve_eikonal", side_effect=RuntimeError("injected failure")):
        with pytest.raises(RuntimeError, match="injected failure"):
            updater.update(**field_request)
    assert updater.last_update == "invalid"
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_update == "full_recompute"
    updater.clear()
    assert updater.last_update == "empty"
    updater.update(**field_request)
    assert updater.last_update == "full_recompute"
