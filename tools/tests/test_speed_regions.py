"""Check exact speed updates across separated edit regions."""

from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import field as vf
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def room(shape=(201, 301)):
    obstacles = np.zeros(shape, dtype=bool)
    obstacles[[0, -1], :] = True
    obstacles[:, [0, -1]] = True
    return obstacles


def config(**changes):
    return replace(vf.VectorFieldConfig(centring=True, centring_cap=0.4,
                                        speed_profile="exponential",
                                        early_exit_enable=False), **changes)


def update_and_compare(obstacles, changed, cfg):
    previous = vf._prepare_map(obstacles, 0.1, cfg)
    saved = {name: getattr(previous, name).copy()
             for name in ("obstacle_mask", "speed", "wall_dist")}
    edt = vf.distance_transform_edt(~changed) * 0.1
    saved_changed, saved_edt = changed.copy(), edt.copy()
    with patch.object(vf, "build_speed_field", wraps=vf.build_speed_field) as build:
        actual, stats = vf._update_speed_field(edt, changed, cfg, 0.1, previous)
    np.testing.assert_array_equal(actual, vf.build_speed_field(edt, changed, cfg, 0.1))
    assert build.call_count == stats["region_count"]
    assert sum(call.args[0].size for call in build.call_args_list) == stats["computed_cells"]
    actual[:] = -1.0
    for name, expected in saved.items():
        np.testing.assert_array_equal(getattr(previous, name), expected, err_msg=name)
    np.testing.assert_array_equal(changed, saved_changed)
    np.testing.assert_array_equal(edt, saved_edt)
    return stats


@pytest.mark.parametrize("profile", ["linear", "exponential"])
@pytest.mark.parametrize("edit", ["opposite_corners", "boundary", "nearby"])
def test_separated_edits_match_full_speed_and_preserve_inputs(profile, edit):
    obstacles = room()
    changed = obstacles.copy()
    if edit == "opposite_corners":
        changed[3:6, 3:6] = True
        changed[-6:-3, -6:-3] = True
    elif edit == "boundary":
        changed[0, 0] = False
        changed[-1, -1] = False
    else:
        changed[30, 30] = True
        changed[49, 49] = True
    stats = update_and_compare(obstacles, changed, config(speed_profile=profile))
    assert stats["status"] == "local_update"
    assert stats["region_count"] == 2
    assert 0 < stats["changed_cells"] <= stats["updated_cells"]
    assert stats["updated_cells"] <= stats["computed_cells"] < changed.size / 4


def test_overlapping_source_crops_have_disjoint_outputs():
    dirty = np.zeros((61, 81), dtype=bool)
    dirty[20, 20] = dirty[31, 31] = True
    regions = vf._speed_update_regions(dirty, 5)
    assert len(regions) == 2
    output_count = np.zeros(dirty.shape, dtype=np.uint8)
    source_count = np.zeros(dirty.shape, dtype=np.uint8)
    for (r0, r1, c0, c1), (sr0, sr1, sc0, sc1) in regions:
        output_count[r0:r1, c0:c1] += 1
        source_count[sr0:sr1, sc0:sc1] += 1
    assert np.max(output_count) == 1
    assert np.max(source_count) == 2
    assert np.all(output_count[dirty] == 1)
    assert regions == vf._speed_update_regions(dirty, 5)


def test_many_edits_keep_at_most_four_regions():
    obstacles = room((241, 381))
    changed = obstacles.copy()
    changed[np.ix_([20, 100, 180], [20, 130, 240, 350])] = True
    stats = update_and_compare(obstacles, changed, config())
    assert stats["status"] == "local_update"
    assert stats["region_count"] == 4
    assert stats["computed_cells"] < changed.size


def test_dense_edits_use_full_speed_once():
    obstacles = room((61, 81))
    changed = obstacles.copy()
    changed[1:-1:2, 1:-1:2] = True
    stats = update_and_compare(obstacles, changed, config())
    assert stats["status"] == "full_recompute"
    assert stats["reason"] == "full_region"
    assert stats["region_count"] == 1
    assert stats["updated_cells"] == stats["computed_cells"] == changed.size


def test_unchanged_inputs_have_no_regions():
    obstacles = room()
    stats = update_and_compare(obstacles, obstacles.copy(), config())
    assert stats["status"] == "unchanged"
    assert stats["region_count"] == stats["updated_cells"] == stats["computed_cells"] == 0
