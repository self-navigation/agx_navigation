"""Check full interior reconstruction from one occupied bounding box."""

from dataclasses import fields
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt as full_edt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from agx_planning.vector_field import field as vf
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def assert_construction(obstacles, resolution):
    saved = obstacles.copy()
    with patch.object(vf, "distance_transform_edt", wraps=full_edt) as transform:
        actual, info = vf._update_interior_distance(obstacles, resolution, None)
    np.testing.assert_array_equal(actual, full_edt(obstacles) * resolution)
    assert info["status"] == "full_recompute"
    assert info["reason"] == "new_map"
    assert info["updated_cells"] == info["total_cells"] == obstacles.size
    assert info["region_count"] == info["transform_calls"] == transform.call_count == 1
    assert info["computed_cells"] == transform.call_args.args[0].size
    actual[:] = -100
    np.testing.assert_array_equal(obstacles, saved)
    return info, transform.call_args.args[0]


@pytest.mark.parametrize("resolution", [0.001, 0.05, 0.2, 7.0, np.float32(0.05)])
@pytest.mark.parametrize("edge", [False, True])
def test_block_uses_exact_free_halo_at_each_resolution(resolution, edge):
    obstacles = np.zeros((101, 151), dtype=bool)
    row, col = (0, 0) if edge else (30, 40)
    obstacles[row:row + 31, col:col + 51] = True
    info, source = assert_construction(obstacles, resolution)
    assert source.shape == ((32, 52) if edge else (33, 53))
    assert info["computed_cells"] < obstacles.size / 5


def test_sparse_random_crops_and_array_layouts_match():
    rng = np.random.default_rng(372)
    for density in (0.01, 0.1, 0.9):
        obstacles = np.zeros((129, 173), dtype=bool)
        obstacles[23:110, 31:145] = rng.random((87, 114)) < density
        for selected in (obstacles, obstacles.T, obstacles[::-1, ::2]):
            info, _ = assert_construction(selected, 0.1)
            assert info["computed_cells"] < selected.size


@pytest.mark.parametrize("shape", [(0, 3), (1, 51), (51, 1), (31, 43)])
@pytest.mark.parametrize("layout", ["empty", "blocked", "corners"])
def test_full_extent_and_empty_cases_use_direct_scipy(shape, layout):
    obstacles = np.zeros(shape, dtype=bool)
    if layout == "blocked":
        obstacles[:] = True
    elif layout == "corners" and obstacles.size:
        obstacles[0, 0] = obstacles[-1, -1] = True
    info, source = assert_construction(obstacles, 0.1)
    assert source is obstacles
    assert info["computed_cells"] == obstacles.size


@pytest.mark.parametrize("resolution", [0.0, -0.1, np.inf, np.nan])
def test_unsupported_resolution_keeps_full_fallback_semantics(resolution):
    obstacles = np.zeros((31, 43), dtype=bool)
    obstacles[10:20, 15:25] = True
    with np.errstate(invalid="ignore"):
        _, source = assert_construction(obstacles, resolution)
    assert source is obstacles


def test_non_grid_geometry_keeps_full_fallback():
    obstacles = np.zeros((7, 9, 11), dtype=bool)
    obstacles[2:5, 3:6, 4:7] = True
    _, source = assert_construction(obstacles, 0.1)
    assert source is obstacles


def test_crop_work_is_added_to_failed_local_attempts():
    old = np.zeros((301, 401), dtype=bool)
    old[140:150, 190:200] = True
    new = old.copy()
    new[160, 210] = True
    previous = SimpleNamespace(obstacle_mask=old,
                               wall_dist=(full_edt(~old) - full_edt(old)) * 0.1)
    attempts = []

    def uncertified(source):
        attempts.append(source.shape)
        if len(attempts) <= 3:
            return np.full(source.shape, 1e6)
        return full_edt(source)

    with patch.object(vf, "distance_transform_edt", side_effect=uncertified):
        actual, info = vf._update_interior_distance(new, 0.1, previous)
    np.testing.assert_array_equal(actual, full_edt(new) * 0.1)
    assert info["status"] == "full_recompute"
    assert info["reason"] == "uncertified_region"
    assert info["transform_calls"] == len(attempts) == 4
    assert attempts[-1] == (23, 23)
    assert info["computed_cells"] == sum(h * w for h, w in attempts)
    assert info["computed_cells"] < new.size


def test_global_free_distance_stays_full_size():
    obstacles = np.zeros((101, 151), dtype=bool)
    obstacles[30:61, 40:91] = True
    cfg = vf.VectorFieldConfig(centring=False, early_exit_enable=False)
    with patch.object(vf, "distance_transform_edt", wraps=full_edt) as transform:
        actual = vf._prepare_map(obstacles, 0.1, cfg, incremental=True)
    assert [call.args[0].shape for call in transform.call_args_list] == [
        obstacles.shape, (33, 53)
    ]
    np.testing.assert_array_equal(transform.call_args_list[0].args[0], ~obstacles)
    expected = vf._prepare_map(obstacles, 0.1, cfg)
    np.testing.assert_array_equal(actual.wall_dist, expected.wall_dist)
    np.testing.assert_array_equal(actual.speed, expected.speed)


def test_cold_field_channels_match_canonical_solver():
    occupancy = np.zeros((71, 101), dtype=np.int8)
    occupancy[25:41, 35:66] = 100
    cfg = vf.VectorFieldConfig(centring=True, centring_cap=0.4,
                                early_exit_enable=False, smooth_T_before_grad=True)
    args = (occupancy, 85, 35, 0.1, 0.0, 0.0, cfg)
    updater = FieldUpdater(incremental=True, max_checkpoints=0)
    actual, expected = updater.update(*args), vf.compute_field(*args)
    assert actual is not None and expected is not None
    assert actual[1] == expected[1]
    for item in fields(vf.VectorFieldResult):
        value, reference = getattr(actual[0], item.name), getattr(expected[0], item.name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, reference, err_msg=item.name)
        else:
            assert value == reference, item.name
    assert updater.last_interior["status"] == "full_recompute"
    assert updater.last_interior["computed_cells"] < occupancy.size
