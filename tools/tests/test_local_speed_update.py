"""Check local centred speed updates against full preprocessing."""

from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from agx_planning.vector_field import field as vf
    from replanning_baseline.inputs import load_suite
    from summarize_replanning import summarize
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def room(shape=(101, 151)):
    obstacles = np.zeros(shape, dtype=bool)
    obstacles[[0, -1], :] = True
    obstacles[:, [0, -1]] = True
    return obstacles


def config(**changes):
    return replace(vf.VectorFieldConfig(centring=True, centring_cap=0.4,
                                        speed_profile="exponential", speed_decay_rate=1.5,
                                        early_exit_enable=False), **changes)


def assert_prepared(actual, expected):
    for name in ("obstacle_mask", "speed", "wall_dist"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name),
                                      err_msg=name)


def assert_field(actual, expected):
    assert actual is not None and expected is not None
    assert actual[1] == expected[1]
    for item in fields(vf.VectorFieldResult):
        observed, reference = getattr(actual[0], item.name), getattr(expected[0], item.name)
        if isinstance(observed, np.ndarray):
            np.testing.assert_array_equal(observed, reference, err_msg=item.name)
        else:
            assert observed == reference, item.name


def test_small_edit_uses_a_local_speed_crop():
    obstacles = room()
    cfg = config()
    previous = vf._prepare_map(obstacles, 0.1, cfg)
    changed = obstacles.copy()
    changed[48:53, 100] = True
    actual = vf._prepare_map(changed, 0.1, cfg, previous=previous, incremental=True)
    assert_prepared(actual, vf._prepare_map(changed, 0.1, cfg))
    stats = actual.speed_update
    assert stats["status"] == "local_update"
    assert 0 < stats["changed_cells"] <= stats["updated_cells"]
    assert stats["updated_cells"] <= stats["computed_cells"] < changed.size
    assert stats["total_cells"] == changed.size
    assert np.any(actual.speed != previous.speed)


@pytest.mark.parametrize("cap,resolution", [
    (0.05, 0.1), (0.3, 0.1), (0.3 - 5e-11, 0.1),
    (0.8, 0.1), (0.2, 0.05), (0.35, 0.2), (21.0 - 3e-9, 7.0),
])
@pytest.mark.parametrize("profile", ["linear", "exponential"])
def test_repeated_random_edits_match_full_speed(cap, resolution, profile):
    rng = np.random.default_rng(5107)
    obstacles = room((65, 85))
    obstacles[20:32, 38:43] = True
    cfg = config(centring_cap=cap, speed_profile=profile)
    previous = vf._prepare_map(obstacles, resolution, cfg)
    local_updates = 0
    for _ in range(8):
        changed = obstacles.copy()
        row, col = rng.integers(0, 62), rng.integers(0, 81)
        changed[row:row + 3, col:col + 4] ^= rng.random((3, 4)) < 0.5
        actual = vf._prepare_map(changed, resolution, cfg,
                                 previous=previous, incremental=True)
        assert_prepared(actual, vf._prepare_map(changed, resolution, cfg))
        stats = actual.speed_update
        assert 0 <= stats["changed_cells"] <= stats["updated_cells"] <= changed.size
        assert stats["updated_cells"] <= stats["computed_cells"] <= changed.size
        local_updates += stats["status"] == "local_update"
        obstacles, previous = changed, actual
    assert local_updates > 0


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (9, 13), (65, 85)])
@pytest.mark.parametrize("initially_blocked", [False, True])
def test_empty_and_full_maps_match_at_borders(shape, initially_blocked):
    obstacles = np.full(shape, initially_blocked, dtype=bool)
    cfg = config()
    previous = vf._prepare_map(obstacles, 0.1, cfg)
    points = [(0, 0), (-1, -1), (0, -1), (-1, 0), (shape[0] // 2, shape[1] // 2)]
    for point in points:
        changed = obstacles.copy()
        changed[point] = ~changed[point]
        actual = vf._prepare_map(changed, 0.1, cfg, previous=previous, incremental=True)
        assert_prepared(actual, vf._prepare_map(changed, 0.1, cfg))
        obstacles, previous = changed, actual


def test_local_crop_without_large_centres_matches_global_centres():
    obstacles = np.ones((101, 151), dtype=bool)
    obstacles[48:53, 5:110] = False
    obstacles[10:90, 90:140] = False
    cfg = config(centring_cap=0.8)
    previous = vf._prepare_map(obstacles, 0.1, cfg)
    changed = obstacles.copy()
    changed[50, 20] = True
    crops = []
    full_speed = vf.build_speed_field

    def capture(edt, *args, **kwargs):
        crops.append(edt.copy())
        return full_speed(edt, *args, **kwargs)

    with patch.object(vf, "build_speed_field", side_effect=capture):
        actual = vf._prepare_map(changed, 0.1, cfg, previous=previous, incremental=True)
    assert_prepared(actual, vf._prepare_map(changed, 0.1, cfg))
    assert actual.speed_update["status"] == "local_update"
    assert len(crops) == 1
    assert crops[0].max() < cfg.centring_cap
    assert np.maximum(actual.wall_dist, 0).max() > cfg.centring_cap


def test_local_update_preserves_all_input_arrays():
    obstacles = room()
    cfg = config()
    previous = vf._prepare_map(obstacles, 0.1, cfg)
    saved = {name: getattr(previous, name).copy()
             for name in ("obstacle_mask", "speed", "wall_dist")}
    changed = obstacles.copy()
    changed[48:53, 100] = True
    edt = vf.distance_transform_edt(~changed) * 0.1
    saved_changed, saved_edt = changed.copy(), edt.copy()
    speed, stats = vf._update_speed_field(edt, changed, cfg, 0.1, previous)
    np.testing.assert_array_equal(speed, vf.build_speed_field(edt, changed, cfg, 0.1))
    assert stats["status"] == "local_update"
    speed[:] = -1.0
    for name, expected in saved.items():
        np.testing.assert_array_equal(getattr(previous, name), expected, err_msg=name)
    np.testing.assert_array_equal(changed, saved_changed)
    np.testing.assert_array_equal(edt, saved_edt)


def test_unchanged_inputs_skip_speed_recomputation():
    obstacles = room()
    cfg = config()
    previous = vf._prepare_map(obstacles, 0.1, cfg)
    with patch.object(vf, "build_speed_field", wraps=vf.build_speed_field) as build:
        actual = vf._prepare_map(obstacles.copy(), 0.1, cfg,
                                 previous=previous, incremental=True)
    assert_prepared(actual, previous)
    assert actual.speed_update["status"] == "unchanged"
    assert actual.speed_update["changed_cells"] == 0
    assert actual.speed_update["updated_cells"] == 0
    assert actual.speed_update["computed_cells"] == 0
    build.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"centring": False}, {"centring_smooth": 0.15},
    {"centring_cap": 0.0}, {"centring_cap": -0.2},
])
def test_unsupported_local_settings_use_full_speed(changes):
    obstacles = room()
    cfg = config(**changes)
    previous = vf._prepare_map(obstacles, 0.1, cfg)
    changed = obstacles.copy()
    changed[48:53, 100] = True
    actual = vf._prepare_map(changed, 0.1, cfg, previous=previous, incremental=True)
    assert_prepared(actual, vf._prepare_map(changed, 0.1, cfg))
    assert actual.speed_update["status"] == "full_recompute"
    assert actual.speed_update["computed_cells"] == changed.size


@pytest.mark.parametrize("previous_kind", ["none", "different_shape"])
def test_missing_compatible_map_uses_full_speed(previous_kind):
    obstacles = room()
    cfg = config()
    previous = (None if previous_kind == "none"
                else vf._prepare_map(room((11, 17)), 0.1, cfg))
    actual = vf._prepare_map(obstacles, 0.1, cfg, previous=previous, incremental=True)
    assert_prepared(actual, vf._prepare_map(obstacles, 0.1, cfg))
    assert actual.speed_update["status"] == "full_recompute"
    assert actual.speed_update["computed_cells"] == obstacles.size


@pytest.mark.parametrize("smoothing", [0.0, 0.15])
def test_updated_fields_match_fresh_fields_with_centring(smoothing):
    occupancy = room((61, 101)).astype(np.int8) * 100
    cfg = config(centring_cap=0.8, centring_smooth=smoothing,
                 smooth_T_before_grad=True, smooth_T_sigma=0.12)
    inputs = dict(map_array=occupancy, goal_col=85, goal_row=30, resolution=0.05,
                  origin_x=0.0, origin_y=0.0, cfg=cfg)
    updater = FieldUpdater(incremental=True, checkpoint_interval=100,
                           max_checkpoints=4, cache_bytes=8 * 1024**2)
    for step in range(5):
        if step == 1:
            occupancy[1:31, 50] = 100
        elif step == 2:
            occupancy[12:20, 50] = 0
        elif step == 3:
            occupancy[30, 84] = 100
        elif step == 4:
            occupancy[1:31, 50] = 0
            occupancy[30, 84] = 0
        assert_field(updater.update(**inputs), vf.compute_field(**inputs))
        if smoothing:
            assert updater.last_speed["status"] == "full_recompute"


def test_goal_change_reuses_speed_and_config_change_resets_it():
    occupancy = room().astype(np.int8) * 100
    inputs = dict(map_array=occupancy, goal_col=20, goal_row=50, resolution=0.1,
                  origin_x=0.0, origin_y=0.0, cfg=config())
    updater = FieldUpdater(incremental=True, checkpoint_interval=100,
                           max_checkpoints=4, cache_bytes=8 * 1024**2)
    updater.update(**inputs)
    inputs["goal_col"] += 1
    with patch.object(vf, "_prepare_map", wraps=vf._prepare_map) as prepare:
        actual = updater.update(**inputs)
        prepare.assert_not_called()
    assert_field(actual, vf.compute_field(**inputs))
    assert updater.last_speed == {}
    occupancy[48:53, 100] = 100
    assert_field(updater.update(**inputs), vf.compute_field(**inputs))
    assert updater.last_speed["status"] == "local_update"
    inputs["cfg"].speed_decay_rate = 2.0
    assert_field(updater.update(**inputs), vf.compute_field(**inputs))
    assert updater.last_speed["status"] == "full_recompute"
    assert_field(updater.update(**inputs), vf.compute_field(**inputs))
    assert updater.last_update == "field_reuse"
    assert updater.last_speed == {}
    updater.clear()
    assert updater.last_speed == {}


def test_cached_backend_keeps_full_speed_preprocessing():
    occupancy = room().astype(np.int8) * 100
    inputs = dict(map_array=occupancy, goal_col=20, goal_row=50, resolution=0.1,
                  origin_x=0.0, origin_y=0.0, cfg=config())
    updater = FieldUpdater()
    for value in (0, 100):
        occupancy[48:53, 100] = value
        assert_field(updater.update(**inputs), vf.compute_field(**inputs))
        assert updater.last_speed == {}


def test_floor6_fixture_adds_repeats_and_removes_eight_obstacles():
    profile, snapshots, _ = load_suite(REPO / "config/replanning/floor6_field_updates.yaml")
    assert [metadata["snapshot_id"] for metadata, _ in snapshots] == [
        "base", "added", "repeated", "removed"
    ]
    base, added, repeated, removed = [occupancy for _, occupancy in snapshots]
    assert base.shape == (1200, 1200)
    old_mask, new_mask = [vf._obstacle_mask(occupancy, profile["occupancy_threshold"],
                                           profile["allow_unknown"])
                          for occupancy in (base, added)]
    assert np.count_nonzero(new_mask & ~old_mask) == 8
    assert not np.any(old_mask & ~new_mask)
    assert snapshots[1][0]["changed_cells"] == 8
    np.testing.assert_array_equal(repeated, added)
    np.testing.assert_array_equal(removed, base)


def test_cli_records_local_speed_without_double_counting_stages(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"""schema_version: 1
profile: {REPO / 'config/replanning/baseline_profile.yaml'}
map:
  grid:
    shape: [101, 151]
    resolution: 0.1
    origin: [0, 0, 0]
    rectangles:
      - [0, 0, 151, 1, 100]
      - [0, 100, 151, 101, 100]
      - [0, 0, 1, 101, 100]
      - [150, 0, 151, 101, 100]
cases:
  - id: local_speed
    start: [3.0, 5.0, 0, 0, 0]
    goal: [2.0, 5.0, 0]
    updates:
      - id: obstacle
        rectangles: [[100, 48, 101, 53, 100]]
      - id: repeated
""")
    output = tmp_path / "run"
    subprocess.run([sys.executable, str(REPO / "tools/bench_replanning.py"),
                    "--config", str(suite), "--out", str(output), "--field-only",
                    "--field-backend", "incremental", "--checkpoint-interval", "100",
                    "--max-checkpoints", "4", "--field-cache-mib", "8", "--repeat", "2"],
                   env={**os.environ, "PYTHONPATH": str(REPO / "tools")},
                   check=True, capture_output=True, text=True)
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    for record in records:
        assert record["status"] == "field_only"
        assert record["compute_field_s"] >= sum(
            record[key] for key in ("edt_s", "speed_s", "fmm_s", "gradient_s")
        )
        if record["snapshot_id"] == "repeated":
            assert "speed_update" not in record
            assert record["speed_s"] == 0
        else:
            assert record["speed_s"] > 0
            status = "local_update" if record["snapshot_id"] == "obstacle" else "full_recompute"
            assert record["speed_update"]["status"] == status
            if status == "local_update":
                assert record["speed_update"]["computed_cells"] < record["speed_update"]["total_cells"]
    report = summarize(output)
    assert report["complete"]
    for group in report["groups"]:
        if group["snapshot_id"] == "repeated":
            assert group["speed_updates"] == group["speed_reasons"] == {}
        else:
            status = "local_update" if group["snapshot_id"] == "obstacle" else "full_recompute"
            assert group["speed_updates"] == {status: 2}
            reasons = [record["speed_update"]["reason"] for record in records
                       if record["snapshot_id"] == group["snapshot_id"]]
            assert len(set(reasons)) == 1
            assert group["speed_reasons"] == {reasons[0]: 2}
