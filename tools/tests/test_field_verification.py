"""Check optional field verification without adding it to measured times."""

from copy import deepcopy
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
    from agx_planning.vector_field.bottleneck import Bottleneck
    from replanning_baseline.inputs import cell, load_suite
    from replanning_baseline import measure
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


@pytest.fixture
def sample(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"""schema_version: 1
profile: {REPO / 'config/replanning/baseline_profile.yaml'}
map:
  grid:
    shape: [21, 31]
    resolution: 0.1
    origin: [0, 0, 0]
    rectangles:
      - [0, 0, 31, 1, 100]
      - [0, 20, 31, 21, 100]
      - [0, 0, 1, 21, 100]
      - [30, 0, 31, 21, 100]
cases:
  - id: verification
    start: [0.5, 1.0, 0, 0, 0]
    goal: [2.5, 1.0, 0]
    updates: [{{id: repeated}}]
""")
    profile, snapshots, _ = load_suite(suite)
    metadata, occupancy = snapshots[0]
    return occupancy, metadata, profile, suite


def fresh_field(sample):
    occupancy, metadata, profile, _ = sample
    geometry = metadata["geometry"]
    start, goal = cell(metadata["start"], geometry), cell(metadata["goal"], geometry)
    return vf.compute_field(occupancy, *goal, geometry["resolution"], *geometry["origin"][:2],
                            vf.VectorFieldConfig(**profile["field"]), *start,
                            profile["occupancy_threshold"], profile["allow_unknown"])


class FixedUpdater:
    last_update = "injected"
    last_fmm = {}
    last_speed = {}

    def __init__(self, result):
        self.result = result

    def update(self, *args, **kwargs):
        return deepcopy(self.result)


def verify_snapshot(sample, updater=None, **options):
    occupancy, metadata, profile, _ = sample
    return measure.measure_snapshot(occupancy, metadata, profile, field_only=True,
                                    updater=updater, verify_fields=True, **options)


def test_verification_is_opt_in_and_does_not_require_saved_arrays(sample):
    occupancy, metadata, profile, _ = sample
    with patch.object(vf, "compute_field", wraps=vf.compute_field) as compute:
        record, arrays = measure.measure_snapshot(occupancy, metadata, profile, field_only=True)
    assert record["status"] == "field_only"
    assert "verification" not in record
    assert arrays == {}
    assert compute.call_count == 1
    verified, arrays = verify_snapshot(sample)
    assert verified["verification"]["status"] == "match"
    assert verified["verification"]["differing_fields"] == []
    assert arrays == {}


@pytest.mark.parametrize("name", [
    "travel_time", "grad_x", "grad_y", "grad_mag", "wall_dist",
    "free_max_T", "origin_x", "origin_y", "resolution", "dir_x", "dir_y", "bottlenecks",
])
def test_verification_checks_every_raw_field(sample, name):
    outcome = fresh_field(sample)
    result = outcome[0]
    value = getattr(result, name)
    if isinstance(value, np.ndarray):
        value[5, 5] += 0.25
    elif name in ("dir_x", "dir_y"):
        setattr(result, name, np.zeros_like(result.travel_time))
    elif name == "bottlenecks":
        result.bottlenecks = [(Bottleneck(1.0, 1.0, 1.0, 0.0, 0.5), 1.0)]
    else:
        setattr(result, name, value + 0.25)
    record, _ = verify_snapshot(sample, FixedUpdater(outcome))
    assert record["verification"]["status"] == "mismatch"
    assert record["verification"]["differing_fields"] == [name]


@pytest.mark.parametrize("change,mask_changed", [
    ("remove_reached", True), ("add_reached", True), ("move_reached", True),
    ("nan_to_infinity", False),
])
def test_verification_checks_nan_values_and_reached_locations(sample, change, mask_changed):
    outcome = fresh_field(sample)
    times = outcome[0].travel_time
    if change in ("remove_reached", "move_reached"):
        times[5, 5] = np.nan
    if change in ("add_reached", "move_reached"):
        times[0, 0] = 1.0
    elif change == "nan_to_infinity":
        times[0, 0] = np.inf
    record, _ = verify_snapshot(sample, FixedUpdater(outcome))
    verification = record["verification"]
    assert verification["status"] == "mismatch"
    assert "travel_time" in verification["differing_fields"]
    assert ("reached_mask" in verification["differing_fields"]) == mask_changed
    if change == "move_reached":
        assert verification["reached_cells"] == verification["reference_reached_cells"]


def test_verification_detects_a_wrong_speed_field(sample):
    build = vf.build_speed_field

    class WrongSpeedUpdater(FixedUpdater):
        def update(self, *args, **kwargs):
            def wrong_speed(*build_args, **build_kwargs):
                speed = build(*build_args, **build_kwargs)
                speed[5:15, 15:17] *= 0.5
                return speed

            with patch.object(vf, "build_speed_field", side_effect=wrong_speed):
                return vf.compute_field(*args, **kwargs)

    record, _ = verify_snapshot(sample, WrongSpeedUpdater(None))
    assert record["verification"]["status"] == "mismatch"
    assert "travel_time" in record["verification"]["differing_fields"]
    assert "grad_x" in record["verification"]["differing_fields"]


@pytest.mark.parametrize("change", [False, True])
def test_nested_optional_fields_compare_nan_values_safely(sample, change):
    reference = fresh_field(sample)
    reference[0].dir_x = np.full_like(reference[0].travel_time, np.nan)
    reference[0].dir_y = np.zeros_like(reference[0].travel_time)
    reference[0].bottlenecks = [(Bottleneck(1.0, 1.0, 1.0, 0.0, np.nan), 1.0)]
    observed = deepcopy(reference)
    if change:
        observed[0].bottlenecks[0][0].x += 1.0
    with patch.object(vf, "compute_field", return_value=reference):
        record, _ = verify_snapshot(sample, FixedUpdater(observed))
    assert record["verification"]["status"] == ("mismatch" if change else "match")
    assert record["verification"]["differing_fields"] == (["bottlenecks"] if change else [])


def test_cached_field_is_verified_against_a_fresh_solve(sample):
    updater = FieldUpdater()
    with patch.object(vf, "compute_field", wraps=vf.compute_field) as compute:
        first, _ = verify_snapshot(sample, updater)
        repeated, _ = verify_snapshot(sample, updater)
    assert compute.call_count == 2
    assert first["verification"]["status"] == repeated["verification"]["status"] == "match"
    assert repeated["field_update"] == "field_reuse"
    assert repeated["fmm_s"] == 0


def test_unreachable_start_still_verifies_the_available_field(sample):
    sample[0][:, 15] = 100
    record, _ = verify_snapshot(sample, FieldUpdater())
    assert record["status"] == "start_unreached"
    assert record["verification"]["status"] == "match"
    assert record["verification"]["reference_reached_cells"] > 0


@pytest.mark.parametrize("blocked", ["start", "goal"])
def test_blocked_endpoint_skips_reference_computation(sample, blocked):
    occupancy, metadata, _, _ = sample
    col, row = cell(metadata[blocked], metadata["geometry"])
    occupancy[row, col] = 100
    with patch.object(vf, "compute_field", wraps=vf.compute_field) as compute:
        record, _ = verify_snapshot(sample)
    assert record["status"] == f"blocked_{blocked}"
    assert record["verification"]["status"] == "skipped"
    assert record["verification"]["reason"] == record["status"]
    compute.assert_not_called()


def test_measurement_exception_is_skipped_not_reported_as_a_match(sample):
    with patch.object(measure, "measure_field", side_effect=RuntimeError("measurement failed")), \
            patch.object(vf, "compute_field", wraps=vf.compute_field) as compute:
        record, _ = verify_snapshot(sample)
    assert record["status"] == "exception"
    assert record["verification"]["status"] == "skipped"
    assert record["verification"]["reason"] == "exception"
    compute.assert_not_called()


def test_reference_exception_is_recorded_separately(sample):
    outcome = fresh_field(sample)
    with patch.object(vf, "compute_field", side_effect=RuntimeError("reference failed")):
        record, _ = verify_snapshot(sample, FixedUpdater(outcome))
    assert record["status"] == "field_only"
    assert record["verification"]["status"] == "error"
    assert record["verification"]["exception"] == "RuntimeError: reference failed"


@pytest.mark.parametrize("missing", ["observed", "reference"])
def test_a_missing_field_is_a_mismatch(sample, missing):
    outcome = fresh_field(sample)
    if missing == "observed":
        record, _ = verify_snapshot(sample, FixedUpdater(None))
        assert record["status"] == "field_failed"
    else:
        with patch.object(vf, "compute_field", return_value=None):
            record, _ = verify_snapshot(sample, FixedUpdater(outcome))
    assert record["verification"]["status"] == "mismatch"
    assert record["verification"]["differing_fields"] == ["field_presence"]


def test_reference_time_is_outside_all_measured_times(sample, monkeypatch):
    outcome = fresh_field(sample)
    clock = {"wall": 100.0, "cpu": 10.0}
    build_speed = vf.build_speed_field
    solve = vf.solve_eikonal

    class ClockedUpdater(FixedUpdater):
        def update(self, *args, **kwargs):
            clock["wall"] += 2.0
            clock["cpu"] += 1.0
            return super().update(*args, **kwargs)

    def reference(*args, **kwargs):
        assert vf.build_speed_field is build_speed
        assert vf.solve_eikonal is solve
        clock["wall"] += 50.0
        clock["cpu"] += 17.0
        return deepcopy(outcome)

    monkeypatch.setattr(measure.time, "perf_counter", lambda: clock["wall"])
    monkeypatch.setattr(measure.time, "process_time", lambda: clock["cpu"])
    with patch.object(vf, "compute_field", side_effect=reference):
        record, _ = verify_snapshot(sample, ClockedUpdater(outcome))
    assert record["verification"]["status"] == "match"
    assert record["verification"]["wall_s"] == 50.0
    assert record["total_wall_s"] == record["compute_field_s"] == 2.0
    assert record["total_cpu_s"] == 1.0
    assert sum(record[key] for key in ("edt_s", "speed_s", "fmm_s", "gradient_s")) == 0.0


@pytest.mark.parametrize("mode", ["match", "mismatch", "error"])
def test_cli_records_all_samples_and_returns_verification_failure(sample, tmp_path, mode):
    script = """
import sys
from bench_replanning import main
from agx_planning.vector_field import FieldUpdater
from agx_planning.vector_field import field as vf

mode, suite, output = sys.argv[1:]
if mode == 'mismatch':
    update = FieldUpdater.update
    count = [0]
    def changed(self, *args, **kwargs):
        result = update(self, *args, **kwargs)
        count[0] += 1
        if count[0] == 1:
            result[0].grad_x[5, 5] += 0.25
        return result
    FieldUpdater.update = changed
elif mode == 'error':
    def failed(*args, **kwargs):
        raise RuntimeError('reference failed')
    vf.compute_field = failed
sys.argv = ['bench_replanning', '--config', suite, '--out', output,
            '--field-only', '--field-backend', 'cached', '--verify-fields',
            '--warmups', '1', '--repeat', '1']
sys.exit(main())
"""
    output = tmp_path / "run"
    completed = subprocess.run([sys.executable, "-c", script, mode, str(sample[3]), str(output)],
                               env={**os.environ, "PYTHONPATH": str(REPO / "tools")},
                               capture_output=True, text=True)
    assert completed.returncode == (0 if mode == "match" else 1), completed.stderr
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert (output / "COMPLETE").exists()
    assert all(record["status"] == "field_only" for record in records)
    assert all("artifact" not in record for record in records)
    if mode == "mismatch":
        assert records[0]["warmup"]
        assert records[0]["verification"]["status"] == "mismatch"
        assert records[0]["verification"]["differing_fields"] == ["grad_x"]
        assert all(record["verification"]["status"] == "match" for record in records[1:])
    else:
        assert all(record["verification"]["status"] == mode for record in records)
    assert mode in completed.stdout
