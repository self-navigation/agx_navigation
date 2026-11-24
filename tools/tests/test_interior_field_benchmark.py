"""Check cavity edits through the field benchmark and reference verifier."""

from copy import deepcopy
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "config/replanning/interior_field_updates.yaml"
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from agx_planning.vector_field import field as vf
    from replanning_baseline.inputs import cell, load_suite
    from replanning_baseline.measure import measure_snapshot
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


@pytest.fixture(scope="module")
def suite():
    profile, snapshots, _ = load_suite(SUITE)
    return profile, snapshots


def field_args(occupancy, metadata, profile):
    geometry = metadata["geometry"]
    start, goal = cell(metadata["start"], geometry), cell(metadata["goal"], geometry)
    return (occupancy, *goal, geometry["resolution"], *geometry["origin"][:2],
            vf.VectorFieldConfig(**profile["field"]), *start,
            profile["occupancy_threshold"], profile["allow_unknown"])


def assert_same_field(actual, expected):
    assert actual is not None and expected is not None
    assert actual[1] == expected[1]
    for item in fields(vf.VectorFieldResult):
        observed, reference = getattr(actual[0], item.name), getattr(expected[0], item.name)
        if isinstance(observed, np.ndarray):
            np.testing.assert_array_equal(observed, reference, err_msg=item.name)
        else:
            assert observed == reference, item.name


def test_incremental_fields_match_full_fields_when_cavities_change(suite):
    profile, snapshots = suite
    updater = FieldUpdater(incremental=True, checkpoint_interval=1_000_000,
                           max_checkpoints=2, cache_bytes=32 * 1024**2)
    baseline = None
    local_updates = 0
    for metadata, occupancy in snapshots:
        if metadata["snapshot_id"] == "base":
            updater.clear()
        args = field_args(occupancy, metadata, profile)
        expected = vf.compute_field(*args)
        actual = updater.update(*args)
        assert_same_field(actual, expected)
        result = actual[0]
        if metadata["snapshot_id"] == "base":
            baseline = result
            assert not np.isfinite(result.travel_time[90:110, 140:160]).any()
            assert not np.isfinite(result.travel_time[45:55, 35:45]).any()
            assert not np.isfinite(result.travel_time[160:170, 245:255]).any()
        else:
            np.testing.assert_array_equal(result.travel_time, baseline.travel_time)
        if metadata["snapshot_id"] == "closed":
            changed = np.count_nonzero(result.wall_dist != baseline.wall_dist)
            assert changed > metadata["changed_cells"]
        elif metadata["snapshot_id"] == "repeated":
            assert updater.last_update == "field_reuse"
            assert updater.last_fmm == updater.last_speed == updater.last_interior == {}
        elif metadata["snapshot_id"] == "reopened":
            np.testing.assert_array_equal(result.wall_dist, baseline.wall_dist)
        if metadata["snapshot_id"] in ("closed", "reopened"):
            interior = updater.last_interior
            if (interior["status"] == "local_update"
                    and interior["computed_cells"] < interior["total_cells"]):
                local_updates += 1
    assert local_updates > 0


def test_verifier_detects_stale_wall_distances_with_correct_travel_times(suite):
    profile, snapshots = suite
    _, base = snapshots[0]
    metadata, closed = snapshots[1]
    outcome = vf.compute_field(*field_args(closed, metadata, profile))
    stale = deepcopy(outcome)
    old_obstacles = vf._obstacle_mask(base, profile["occupancy_threshold"], profile["allow_unknown"])
    resolution = metadata["geometry"]["resolution"]
    stale[0].wall_dist = (vf.distance_transform_edt(~old_obstacles)
                          - vf.distance_transform_edt(old_obstacles)) * resolution
    assert not np.array_equal(stale[0].wall_dist, outcome[0].wall_dist)
    np.testing.assert_array_equal(stale[0].travel_time, outcome[0].travel_time)

    class StaleWallUpdater:
        last_update = "injected"
        last_fmm = {}
        last_speed = {}
        last_interior = {}

        def update(self, *args, **kwargs):
            return deepcopy(stale)

    record, arrays = measure_snapshot(closed, metadata, profile, field_only=True,
                                      updater=StaleWallUpdater(), verify_fields=True)
    assert record["status"] == "field_only"
    assert record["verification"]["status"] == "mismatch"
    assert record["verification"]["differing_fields"] == ["wall_dist"]
    assert arrays == {}


def test_cli_verifies_cavity_edits_and_clears_cached_diagnostics(tmp_path):
    output = tmp_path / "run"
    completed = subprocess.run(
        [sys.executable, str(REPO / "tools/bench_replanning.py"),
         "--config", str(SUITE), "--out", str(output), "--field-only",
         "--field-backend", "incremental", "--checkpoint-interval", "1000000",
         "--max-checkpoints", "2", "--field-cache-mib", "32", "--verify-fields"],
        env={**os.environ, "PYTHONPATH": str(REPO / "tools")},
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(records) == 8
    assert (output / "COMPLETE").exists()
    for record in records:
        assert record["status"] == "field_only"
        assert record["verification"]["status"] == "match"
        assert record["verification"]["differing_fields"] == []
        if record["snapshot_id"] == "repeated":
            assert record["field_update"] == "field_reuse"
            assert sum(record[key] for key in ("edt_s", "speed_s", "fmm_s", "gradient_s")) == 0
            assert not {"interior_update", "speed_update", "fmm_update"} & record.keys()
        else:
            assert record["interior_update"]["status"] in ("local_update", "full_recompute", "unchanged")
