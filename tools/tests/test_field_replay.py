"""Check incremental fields against fresh solves without ROS."""

from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import skfmm

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from agx_planning.vector_field import field as vf
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


@pytest.fixture
def field_request():
    occupancy = np.zeros((41, 81), dtype=np.int8)
    occupancy[[0, -1], :] = 100
    occupancy[:, [0, -1]] = 100
    occupancy[1:14, 40] = 100
    occupancy[27:-1, 40] = 100
    return dict(map_array=occupancy, goal_col=10, goal_row=20,
                resolution=0.1, origin_x=-3.0, origin_y=-2.0,
                cfg=vf.VectorFieldConfig(early_exit_enable=False, inflation_radius=0.2),
                robot_col=20, robot_row=20)


def make_updater(**options):
    settings = dict(incremental=True, checkpoint_interval=100,
                    max_checkpoints=4, cache_bytes=4 * 1024**2)
    return FieldUpdater(**{**settings, **options})


def assert_same(actual, expected):
    assert actual is not None and expected is not None
    assert actual[1] == expected[1]
    for item in fields(vf.VectorFieldResult):
        observed, reference = getattr(actual[0], item.name), getattr(expected[0], item.name)
        if isinstance(observed, np.ndarray):
            np.testing.assert_array_equal(observed, reference, err_msg=item.name)
        else:
            assert observed == reference, item.name


@pytest.mark.parametrize("config", [
    {},
    {"speed_profile": "exponential", "speed_decay_rate": 1.5},
    {"smooth_T_before_grad": True, "smooth_T_sigma": 0.12, "bottleneck_align": True},
])
def test_repeated_add_remove_and_mixed_edits_match_fresh_fields(field_request, config):
    field_request["cfg"] = replace(field_request["cfg"], **config)
    occupancy = field_request["map_array"]
    updater = make_updater()
    for step in range(8):
        if step == 2:
            occupancy[4:10, 65] = 100
        elif step == 4:
            occupancy[4:10, 65] = 0
        elif step == 5:
            occupancy[20:25, 65] = 100
            occupancy[6:10, 40] = 0
        elif step == 6:
            occupancy[1:-1, 40] = 100
        elif step == 7:
            occupancy[18:24, 40] = 0
        assert_same(updater.update(**field_request), vf.compute_field(**field_request))
        if step in (1, 3):
            assert updater.last_update == "field_reuse"
            assert updater.last_fmm == {}
        else:
            assert updater.last_update in ("full_recompute", "fmm_replay", "fmm_reuse")
            assert updater.last_fmm["checkpoint_count"] <= 4
            assert updater.last_fmm["cache_array_bytes"] <= 4 * 1024**2


def test_far_edit_reuses_completed_batches(field_request):
    updater = make_updater()
    updater.update(**field_request)
    field_request["map_array"][4:10, 65] = 100
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_update == "fmm_replay"
    stats = updater.last_fmm
    assert stats["status"] == "replay"
    assert stats["initialization_check"] == "unchanged_support"
    assert stats["changed_cells"] > 0
    assert 0 < stats["reused_batches"] < stats["first_affected_batch"]
    assert stats["computed_batches"] == stats["total_batches"] - stats["reused_batches"]
    assert 0 < stats["checkpoint_count"] <= 4
    assert 0 < stats["cache_array_bytes"] <= 4 * 1024**2


def test_edit_in_disconnected_region_reuses_the_completed_march(field_request):
    field_request["map_array"][:, 40] = 100
    updater = make_updater(checkpoint_interval=10000)
    updater.update(**field_request)
    completed_batches = updater.last_fmm["total_batches"]
    field_request["map_array"][4:10, 65] = 100
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_update == "fmm_replay"
    assert updater.last_fmm["reused_batches"] == completed_batches
    assert updater.last_fmm["computed_batches"] == 0
    assert updater.last_fmm["initialization_check"] == "unchanged_support"


def test_map_edit_rebuilds_speed_and_gradients(field_request):
    updater = make_updater()
    with patch.object(vf, "_prepare_map", wraps=vf._prepare_map) as prepare, \
            patch.object(vf, "build_speed_field", wraps=vf.build_speed_field) as speed, \
            patch.object(vf, "field_from_T", wraps=vf.field_from_T) as gradient:
        updater.update(**field_request)
        field_request["map_array"][4:10, 65] = 100
        actual = updater.update(**field_request)
        assert (prepare.call_count, speed.call_count, gradient.call_count) == (2, 2, 2)
    assert_same(actual, vf.compute_field(**field_request))


def test_centring_edit_can_increase_speed_and_decrease_time():
    occupancy = np.zeros((61, 101), dtype=np.int8)
    occupancy[[0, -1], :] = 100
    occupancy[:, [0, -1]] = 100
    cfg = vf.VectorFieldConfig(early_exit_enable=False, centring=True,
                               speed_profile="exponential", speed_decay_rate=1.5)
    inputs = dict(map_array=occupancy, goal_col=85, goal_row=30, resolution=0.05,
                  origin_x=0.0, origin_y=0.0, cfg=cfg)
    updater = make_updater()
    before = updater.update(**inputs)
    old_speed = vf._prepare_map(occupancy >= 65, 0.05, cfg).speed
    occupancy[1:31, 50] = 100
    new_speed = vf._prepare_map(occupancy >= 65, 0.05, cfg).speed
    after = updater.update(**inputs)
    assert_same(after, vf.compute_field(**inputs))
    free = occupancy == 0
    assert np.any(new_speed[free] > old_speed[free])
    comparable = free & np.isfinite(before[0].travel_time) & np.isfinite(after[0].travel_time)
    assert np.any(after[0].travel_time[comparable] < before[0].travel_time[comparable])


@pytest.mark.parametrize("options", [{"cache_bytes": 0}, {"max_checkpoints": 0}])
def test_disabled_native_cache_falls_back_to_full_solve(field_request, options):
    updater = make_updater(**options)
    for value in (100, 0):
        field_request["map_array"][4:10, 65] = value
        assert_same(updater.update(**field_request), vf.compute_field(**field_request))
        assert updater.last_update == "full_recompute"
        assert updater.last_fmm["status"] == "full_recompute"
        assert updater.last_fmm["reason"] == "cache_budget"
        assert updater.last_fmm["reused_batches"] == 0
        assert updater.last_fmm["checkpoint_count"] == 0
        assert updater.last_fmm["cache_array_bytes"] == 0


def test_cache_must_fit_the_working_state_and_first_checkpoint(field_request):
    obstacles = field_request["map_array"] >= 65
    prepared = vf._prepare_map(obstacles, field_request["resolution"], field_request["cfg"])
    phi = np.ones(obstacles.shape)
    phi[field_request["goal_row"], field_request["goal_col"]] = -1.0
    state = skfmm.TravelTimeState(np.ma.array(phi, mask=obstacles), prepared.speed,
                                 dx=field_request["resolution"], track_dependencies=True)
    stats = state.stats
    budget = stats["working_array_bytes"] + stats["checkpoint_array_bytes"] - 1
    assert 0 < stats["working_array_bytes"] <= budget
    updater = make_updater(cache_bytes=budget)
    for value in (0, 100):
        field_request["map_array"][4:10, 65] = value
        assert_same(updater.update(**field_request), vf.compute_field(**field_request))
        assert updater.last_update == "full_recompute"
        assert updater.last_fmm["status"] == "full_recompute"
        assert updater.last_fmm["reason"] == "cache_budget"
        assert updater.last_fmm["checkpoint_count"] == 0
        assert updater.last_fmm["cache_array_bytes"] == 0


def test_small_cache_stays_within_the_array_budget(field_request):
    budget = 500_000
    updater = make_updater(cache_bytes=budget, checkpoint_interval=20)
    for value in (0, 100, 0, 100):
        field_request["map_array"][4:10, 65] = value
        assert_same(updater.update(**field_request), vf.compute_field(**field_request))
        assert updater.last_fmm["cache_array_bytes"] <= budget
        assert updater.last_fmm["checkpoint_count"] <= 4


@pytest.mark.parametrize("change", [
    {"goal_col": 11}, {"resolution": 0.2}, {"origin_x": -4.0},
    {"origin_y": 1.0}, {"occupancy_threshold": 40}, {"allow_unknown": True},
])
def test_goal_geometry_and_policy_changes_restart_native_state(field_request, change):
    updater = make_updater()
    updater.update(**field_request)
    field_request.update(change)
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_fmm["status"] == "full_recompute"
    assert updater.last_fmm["reused_batches"] == 0


def test_config_mutation_and_shape_change_restart_native_state(field_request):
    updater = make_updater()
    updater.update(**field_request)
    field_request["cfg"].centring = True
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_fmm["status"] == "full_recompute"
    assert updater.last_fmm["reused_batches"] == 0
    field_request["map_array"] = np.pad(field_request["map_array"], ((0, 2), (0, 2)),
                                        constant_values=100)
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_fmm["status"] == "full_recompute"
    assert updater.last_fmm["reused_batches"] == 0


def test_source_neighbour_edit_matches_fresh_initialization(field_request):
    updater = make_updater()
    updater.update(**field_request)
    field_request["map_array"][20, 11] = 100
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_fmm["status"] == "full_recompute"
    assert updater.last_fmm["initialization_check"] == "full_comparison"
    assert updater.last_fmm["reused_batches"] == 0


def test_early_exit_uses_the_same_partial_domain(field_request):
    field_request["cfg"] = replace(field_request["cfg"], early_exit_enable=True,
                                    early_exit_margin=0.0)
    updater = make_updater()
    for robot_col in (20, 25):
        field_request["robot_col"] = robot_col
        actual = updater.update(**field_request)
        assert_same(actual, vf.compute_field(**field_request))
        assert updater.last_fmm["status"] == "full_recompute"
        assert updater.last_fmm["reason"] == "early_exit"
        assert updater.last_fmm["cache_array_bytes"] == 0
        assert np.any(np.isnan(actual[0].travel_time[field_request["map_array"] == 0]))


def test_results_and_diagnostics_do_not_change_cached_data(field_request):
    field_request["cfg"].bottleneck_align = True
    field_request["map_array"][14:16, 40] = 100
    field_request["map_array"][25:27, 40] = 100
    updater = make_updater()
    updater.update(**field_request)
    field_request["map_array"][4:10, 65] = 100
    reference = vf.compute_field(**field_request)
    outcome = updater.update(**field_request)
    assert_same(outcome, reference)
    outcome[0].origin_x = 999.0
    for item in fields(vf.VectorFieldResult):
        value = getattr(outcome[0], item.name)
        if isinstance(value, np.ndarray):
            value[:] = 999.0
    assert outcome[0].bottlenecks
    outcome[0].bottlenecks[0][0].width = 999.0
    outcome[0].bottlenecks.clear()
    updater.last_fmm["status"] = "changed by caller"
    assert_same(updater.update(**field_request), reference)
    assert updater.last_update == "field_reuse"
    assert updater.last_fmm == {}
    field_request["map_array"][4:10, 65] = 0
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_fmm["status"] in ("replay", "full_recompute", "unchanged")


@pytest.mark.parametrize("change", [{"goal_col": -1}, {"goal_row": 41}, {"goal_col": 0}])
def test_rejected_input_clears_native_state(field_request, change):
    updater = make_updater()
    updater.update(**field_request)
    assert updater.update(**{**field_request, **change}) is None
    assert updater.last_update == "invalid"
    assert updater.last_fmm == {}
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_fmm["status"] == "full_recompute"
    assert updater.last_fmm["reused_batches"] == 0


def test_explicit_clear_discards_native_state(field_request):
    updater = make_updater()
    updater.update(**field_request)
    updater.clear()
    assert updater.last_update == "empty"
    assert updater.last_fmm == {}
    assert_same(updater.update(**field_request), vf.compute_field(**field_request))
    assert updater.last_fmm["status"] == "full_recompute"
    assert updater.last_fmm["reused_batches"] == 0


def test_cli_records_incremental_timing_and_resets_between_cases(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"""schema_version: 1
profile: {REPO / 'config/replanning/baseline_profile.yaml'}
map:
  grid:
    shape: [21, 31]
    resolution: 0.1
    origin: [0, 0, 0]
    rectangles: [[0, 0, 31, 1, 100]]
cases:
  - id: first
    start: [0.5, 1, 0, 0, 0]
    goal: [1.5, 1, 0]
    updates:
      - id: repeated
      - id: obstacle
        rectangles: [[25, 3, 26, 8, 100]]
  - id: second
    start: [0.5, 1, 0, 0, 0]
    goal: [1.5, 1, 0]
    updates:
      - id: repeated
      - id: obstacle
        rectangles: [[25, 3, 26, 8, 100]]
""")
    script = """
import sys
from bench_replanning import main
sys.argv = ['bench_replanning', '--config', sys.argv[1], '--out', sys.argv[2],
            '--field-only', '--field-backend', 'incremental', '--checkpoint-interval', '20',
            '--max-checkpoints', '4', '--field-cache-mib', '4', '--warmups', '1', '--repeat', '2']
assert main() == 0
assert 'rclpy' not in sys.modules
"""
    output = tmp_path / "run"
    subprocess.run([sys.executable, "-c", script, str(suite), str(output)],
                   env={**os.environ, "PYTHONPATH": str(REPO / "tools")},
                   check=True, capture_output=True, text=True)
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(records) == 18
    for record in records:
        assert record["status"] == "field_only"
        if record["snapshot_id"] == "repeated":
            assert record["field_update"] == "field_reuse"
            assert record["fmm_s"] == 0
            assert "fmm_update" not in record
        else:
            assert record["fmm_s"] > 0
            assert record["fmm_update"]["status"] in ("full_recompute", "replay", "unchanged")
        if record["snapshot_id"] == "base":
            assert record["field_update"] == "full_recompute"
            assert record["fmm_update"]["reused_batches"] == 0
        assert record["compute_field_s"] >= sum(
            record[key] for key in ("edt_s", "speed_s", "fmm_s", "gradient_s")
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["options"]["field_backend"] == "incremental"
