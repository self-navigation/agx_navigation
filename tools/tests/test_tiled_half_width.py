"""Compare tiled centring with the unchanged full-map calculation."""

from dataclasses import fields, replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from agx_planning.vector_field import field as vf
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def room(shape=(151, 201)):
    obstacles = np.zeros(shape, dtype=bool)
    obstacles[[0, -1], :] = True
    obstacles[:, [0, -1]] = True
    return obstacles


def assert_width(edt, resolution, cap):
    saved = edt.copy()
    expected = vf.local_half_width(edt, resolution, cap)
    actual = vf._local_half_width_tiled(edt, resolution, cap)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    actual[:] = -1
    np.testing.assert_array_equal(edt, saved)


@pytest.mark.parametrize("resolution,cap", [
    (0.1, 0.05), (0.1, 0.3), (0.1, 0.3 - 5e-11), (0.1, 0.3 + 5e-11),
    (0.05, 0.8), (0.01, 0.08), (0.2, 0.35), (7.0, 21.0 - 3e-9),
])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_radius_limits_and_random_grids_match(resolution, cap, dtype):
    rng = np.random.default_rng(8934)
    for shape in [(1, 31), (31, 1), (43, 57)]:
        for density in (0.0, 0.08, 0.5, 1.0):
            obstacles = rng.random(shape) < density
            edt = (vf.distance_transform_edt(~obstacles) * resolution).astype(dtype)
            assert_width(edt, resolution, cap)


@pytest.mark.parametrize("layout", ["room", "corridor", "random", "empty", "blocked"])
@pytest.mark.parametrize("cap", [0.3, 0.8])
def test_tiles_and_empty_centre_crops_match(layout, cap):
    obstacles = room((259, 321))
    if layout == "corridor":
        obstacles[:] = True
        obstacles[126:133, 1:255] = False
        obstacles[20:240, 240:300] = False
    elif layout == "random":
        obstacles = np.random.default_rng(932).random(obstacles.shape) < 0.07
    elif layout == "empty":
        obstacles[:] = False
    elif layout == "blocked":
        obstacles[:] = True
    edt = vf.distance_transform_edt(~obstacles) * 0.05
    assert_width(edt, 0.05, cap)


def test_dense_pending_cells_can_use_the_whole_map():
    edt = np.full((129, 129), 0.1)
    edt[::10, ::10] = 0.8
    with patch.object(vf, "distance_transform_edt", wraps=vf.distance_transform_edt) as transform:
        actual = vf._local_half_width_tiled(edt, 0.05, 0.8)
    assert any(call.args[0].shape == edt.shape for call in transform.call_args_list)
    np.testing.assert_array_equal(actual, vf.local_half_width(edt, 0.05, 0.8))


def test_strided_and_transposed_arrays_match():
    edt = vf.distance_transform_edt(~room((259, 321))) * 0.05
    for selected in (edt.T, edt[::-1, ::2]):
        assert_width(selected, 0.05, 0.8)


@pytest.mark.parametrize("resolution,cap", [(0.001, 0.8), (0.05, 80.0)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_high_cap_skips_empty_levels_without_full_grid_scans(resolution, cap, dtype):
    obstacles = np.ones((259, 321), dtype=bool)
    obstacles[128, :] = False
    edt = (vf.distance_transform_edt(~obstacles) * resolution).astype(dtype)
    edt[0, 0] = np.nan
    comparisons = []

    class CountedArray(np.ndarray):
        def __ge__(self, other):
            if self.shape == edt.shape:
                comparisons.append(other)
            return super().__ge__(other)

    expected = vf.local_half_width(edt, resolution, cap)
    actual = vf._local_half_width_tiled(edt.view(CountedArray), resolution, cap)
    np.testing.assert_array_equal(actual, expected)
    assert len(comparisons) <= 2


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (9, 13)])
def test_no_positive_distances_need_no_maximum_search(shape):
    for value in (0.0, np.nan):
        edt = np.full(shape, value)
        assert_width(edt, 0.05, 80.0)


@pytest.mark.parametrize("resolution,cap", [(0.1, 0.0), (0.1, -0.2), (-0.1, 0.3)])
def test_unsupported_settings_keep_the_reference_path(resolution, cap):
    edt = vf.distance_transform_edt(~room((11, 17))) * 0.1
    with patch.object(vf, "local_half_width", wraps=vf.local_half_width) as canonical:
        actual = vf._local_half_width_tiled(edt, resolution, cap)
    canonical.assert_called_once()
    np.testing.assert_array_equal(actual, vf.local_half_width(edt, resolution, cap))


def test_non_grid_input_keeps_the_reference_path():
    edt = np.array([0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0])
    assert_width(edt, 0.1, 0.3)


@pytest.mark.parametrize("changes", [
    {}, {"centring_smooth": 0.1},
    {"smooth_T_before_grad": True, "smooth_T_sigma": 0.12, "bottleneck_align": True},
])
def test_incremental_field_channels_match_full_reference(changes):
    occupancy = room().astype(np.int8) * 100
    cfg = replace(vf.VectorFieldConfig(centring=True, centring_cap=0.4,
                                        speed_profile="exponential",
                                        early_exit_enable=False), **changes)
    inputs = dict(map_array=occupancy, goal_col=100, goal_row=75, resolution=0.1,
                  origin_x=-3.0, origin_y=-2.0, cfg=cfg)
    updater = FieldUpdater(incremental=True, max_checkpoints=0)
    for step in range(2):
        if step:
            occupancy[20:25, 20] = 100
            occupancy[130:135, 180] = 100
        with patch.object(vf, "_local_half_width_tiled", wraps=vf._local_half_width_tiled) as tiled:
            actual = updater.update(**inputs)
        assert tiled.call_count > 0
        expected = vf.compute_field(**inputs)
        assert actual is not None and expected is not None
        assert actual[1] == expected[1]
        for item in fields(vf.VectorFieldResult):
            observed, reference = getattr(actual[0], item.name), getattr(expected[0], item.name)
            if isinstance(observed, np.ndarray):
                np.testing.assert_array_equal(observed, reference, err_msg=item.name)
            else:
                assert observed == reference, item.name


def test_full_and_cached_backends_keep_canonical_half_width():
    obstacles = room((51, 71))
    cfg = vf.VectorFieldConfig(centring=True, early_exit_enable=False)
    with patch.object(vf, "_local_half_width_tiled", side_effect=AssertionError("unexpected tiled path")):
        vf._prepare_map(obstacles, 0.1, cfg)
        updater = FieldUpdater()
        assert updater.update(obstacles.astype(np.int8) * 100, 35, 25, 0.1, 0, 0, cfg)
