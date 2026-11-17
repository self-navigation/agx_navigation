import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

# The unit suite installs a ROS mock without an import spec.
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from replanning_baseline.inputs import apply_rectangles, cell, load_map, load_suite
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def test_yaml_flip_negate_thresholds(tmp_path):
    Image.fromarray(np.array([[0, 255], [205, 255]], dtype=np.uint8)).save(tmp_path / "map.png")
    path = tmp_path / "map.yaml"
    text = ("image: map.png\nresolution: 0.1\norigin: [-1, -2, 0]\n"
            "occupied_thresh: 0.65\nfree_thresh: 0.196\nnegate: 0\n")
    path.write_text(text)
    grid, geometry, _ = load_map({"yaml": "map.yaml"}, tmp_path)
    np.testing.assert_array_equal(grid, [[-1, 0], [100, 0]])
    assert cell([-0.95, -1.95], geometry) == (0, 0)
    with pytest.raises(ValueError, match="outside"):
        cell([-1.001, -1.95], geometry)
    path.write_text(text.replace("negate: 0", "negate: 1"))
    grid, _, _ = load_map({"yaml": "map.yaml"}, tmp_path)
    np.testing.assert_array_equal(grid, [[100, 100], [0, 100]])


def test_rectangles_copy_and_validate():
    original = np.zeros((4, 5), dtype=np.int8)
    updated = apply_rectangles(original, [[1, 1, 3, 2, 100]])
    assert np.count_nonzero(updated) == 2
    assert not original.any()
    for rect in ([0, 0, 6, 2, 100], [0, 0, 1, 1, 128], [0, 0, 1.0, 1, 100]):
        with pytest.raises(ValueError):
            apply_rectangles(original, [rect])


def test_suite_updates_are_cumulative_and_cases_are_independent():
    profile, snapshots, _ = load_suite(REPO / "config/replanning/synthetic.yaml")
    assert profile["field"]["early_exit_enable"] is False
    assert profile["planner"]["N"] == 40
    assert len(snapshots) == 5
    assert snapshots[1][1][2, 50] == 100
    assert snapshots[2][1][30, 50] == 100
    assert snapshots[3][1][30, 50] == 0
    assert snapshots[4][1][2, 50] == 0
    assert snapshots[1][0]["map_sha256"] == snapshots[3][0]["map_sha256"]
    with pytest.raises(ValueError, match="Unknown case"):
        load_suite(REPO / "config/replanning/synthetic.yaml", ["missing"])
