"""Check interior distance updates against independent full SciPy transforms."""

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


def previous_map(obstacles, resolution):
    return SimpleNamespace(obstacle_mask=obstacles.copy(),
                           wall_dist=full_edt(~obstacles) * resolution
                           - full_edt(obstacles) * resolution)


def room(shape=(101, 151)):
    obstacles = np.zeros(shape, dtype=bool)
    obstacles[[0, -1], :] = True
    obstacles[:, [0, -1]] = True
    return obstacles


def assert_update(old, new, resolution):
    previous = previous_map(old, resolution)
    saved_wall, saved_mask, saved_new = previous.wall_dist.copy(), old.copy(), new.copy()
    with patch.object(vf, "distance_transform_edt", wraps=full_edt) as transform:
        actual, info = vf._update_interior_distance(new, resolution, previous)
    np.testing.assert_array_equal(actual, full_edt(new) * resolution)
    assert info["transform_calls"] == transform.call_count
    assert info["computed_cells"] == sum(call.args[0].size for call in transform.call_args_list)
    assert info["total_cells"] == new.size
    actual[:] = -100
    np.testing.assert_array_equal(previous.wall_dist, saved_wall)
    np.testing.assert_array_equal(previous.obstacle_mask, saved_mask)
    np.testing.assert_array_equal(new, saved_new)
    return info


@pytest.mark.parametrize("resolution", [0.001, 0.05, 0.2, 7.0])
@pytest.mark.parametrize("shape", [(1, 83), (61, 1), (61, 83)])
def test_repeated_mixed_edits_match_at_each_resolution(resolution, shape):
    rng = np.random.default_rng(724)
    old = rng.random(shape) < 0.15
    for step in range(6):
        new = old.copy()
        for _ in range(step + 1):
            row, col = rng.integers(shape[0]), rng.integers(shape[1])
            new[row, col] = ~new[row, col]
        assert_update(old, new, resolution)
        old = new


def test_separated_edits_use_small_certified_regions():
    old = room((201, 301))
    new = old.copy()
    new[20:25, 20] = True
    new[170:175, 280] = True
    info = assert_update(old, new, 0.05)
    assert info["status"] == "local_update"
    assert info["region_count"] == 2
    assert info["computed_cells"] < old.size / 20


def test_filling_a_thick_cavity_can_increase_distance_beyond_the_old_maximum():
    old = np.zeros((201, 301), dtype=bool)
    old[40:161, 60:241] = True
    old[50:151, 70:231] = False
    new = old.copy()
    new[50:151, 70:231] = True
    assert full_edt(new).max() > full_edt(old).max()
    assert_update(old, new, 0.05)
    reopened = new.copy()
    reopened[95:106, 145:156] = False
    assert_update(new, reopened, 0.05)


def test_removing_one_of_two_equal_nearest_free_sources():
    old = np.zeros((61, 83), dtype=bool)
    old[10:51, 10:73] = True
    old[30, 30] = old[30, 52] = False
    new = old.copy()
    new[30, 30] = True
    assert full_edt(old)[30, 41] == full_edt(new)[30, 41]
    assert_update(old, new, 0.1)


@pytest.mark.parametrize("shape", [(1, 1), (1, 43), (43, 1), (43, 59)])
@pytest.mark.parametrize("blocked", [False, True])
def test_empty_and_full_maps_with_border_edits(shape, blocked):
    old = np.full(shape, blocked, dtype=bool)
    new = old.copy()
    new[0, 0] = ~new[0, 0]
    info = assert_update(old, new, 0.1)
    if old.all() or new.all():
        assert info["status"] == "full_recompute"
        assert info["reason"] == "no_free_source"
    changed = new.copy()
    changed[-1, -1] = ~changed[-1, -1]
    assert_update(new, changed, 0.1)


def test_unchanged_mask_returns_a_copy_without_transforms():
    old = room()
    info = assert_update(old, old.copy(), 0.1)
    assert info["status"] == "unchanged"
    assert info["updated_cells"] == info["computed_cells"] == info["region_count"] == 0


def test_uncertified_attempts_are_discarded_and_counted_before_full_fallback():
    old = room((301, 401))
    new = old.copy()
    new[150, 200] = True
    previous = previous_map(old, 0.1)
    saved = previous.wall_dist.copy()

    def uncertified(source):
        if source.shape != new.shape:
            return np.full(source.shape, 1e6)
        return full_edt(source)

    with patch.object(vf, "distance_transform_edt", side_effect=uncertified) as transform:
        actual, info = vf._update_interior_distance(new, 0.1, previous)
    np.testing.assert_array_equal(actual, full_edt(new) * 0.1)
    np.testing.assert_array_equal(previous.wall_dist, saved)
    assert info["status"] == "full_recompute"
    assert info["reason"] == "uncertified_region"
    assert info["region_count"] == 1
    assert info["transform_calls"] == transform.call_count == 4
    assert info["computed_cells"] == sum(call.args[0].size for call in transform.call_args_list)
    assert info["computed_cells"] > new.size


def test_source_free_crop_uses_full_fallback():
    old = np.zeros((101, 151), dtype=bool)
    new = old.copy()
    new[20:81, 30:121] = True
    previous = previous_map(old, 0.1)
    # Force an undersized output region to exercise the source-free guard.
    region = [((50, 51, 75, 76), (49, 52, 74, 77))]
    with patch.object(vf, "_speed_update_regions", return_value=region):
        actual, info = vf._update_interior_distance(new, 0.1, previous)
    np.testing.assert_array_equal(actual, full_edt(new) * 0.1)
    assert info["reason"] == "uncertified_region"
    assert info["transform_calls"] == 1


def test_nonfinite_previous_distance_uses_full_fallback():
    old = room()
    new = old.copy()
    new[30, 40] = True
    previous = previous_map(old, 0.1)
    previous.wall_dist[20, 20] = np.nan
    actual, info = vf._update_interior_distance(new, 0.1, previous)
    np.testing.assert_array_equal(actual, full_edt(new) * 0.1)
    assert info["reason"] == "unsupported_geometry"


def assert_field(actual, expected):
    assert actual is not None and expected is not None
    assert actual[1] == expected[1]
    for item in fields(vf.VectorFieldResult):
        value, reference = getattr(actual[0], item.name), getattr(expected[0], item.name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, reference, err_msg=item.name)
        else:
            assert value == reference, item.name


def test_field_channels_cache_resets_and_changed_resolution():
    occupancy = room().astype(np.int8) * 100
    cfg = vf.VectorFieldConfig(centring=True, early_exit_enable=False,
                                centring_cap=0.4, smooth_T_before_grad=True)
    request = dict(map_array=occupancy, goal_col=75, goal_row=50,
                   resolution=0.1, origin_x=0, origin_y=0, cfg=cfg)
    updater = FieldUpdater(incremental=True, max_checkpoints=0)
    assert_field(updater.update(**request), vf.compute_field(**request))
    assert updater.last_interior["reason"] == "new_map"
    occupancy[20:25, 30] = 100
    actual = updater.update(**request)
    assert_field(actual, vf.compute_field(**request))
    assert updater.last_interior["status"] == "local_update"
    actual[0].wall_dist[:] = -100
    assert_field(updater.update(**request), vf.compute_field(**request))
    assert updater.last_interior == {}
    request["resolution"] = 0.2
    assert_field(updater.update(**request), vf.compute_field(**request))
    assert updater.last_interior["reason"] == "new_map"
    request["goal_col"] = -1
    assert updater.update(**request) is None
    assert updater.last_interior == {}
    request["goal_col"] = 75
    updater.update(**request)
    updater.clear()
    assert updater.last_interior == {}
    with patch.object(vf, "_prepare_map", side_effect=RuntimeError("failed preprocessing")):
        with pytest.raises(RuntimeError, match="failed preprocessing"):
            updater.update(**request)
    assert updater.last_interior == {}


def test_full_and_cached_paths_keep_the_original_interior_transform():
    obstacles = room((51, 71))
    cfg = vf.VectorFieldConfig(early_exit_enable=False)
    with patch.object(vf, "_update_interior_distance", side_effect=AssertionError("unexpected update")):
        vf._prepare_map(obstacles, 0.1, cfg)
        updater = FieldUpdater()
        assert updater.update(obstacles.astype(np.int8) * 100, 35, 25, 0.1, 0, 0, cfg)
        assert updater.last_interior == {}
