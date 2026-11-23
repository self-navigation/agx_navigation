"""Check cached field measurements and case boundaries without ROS."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from replanning_baseline.inputs import load_suite
    from replanning_baseline.measure import measure_snapshot
    from summarize_replanning import summarize
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def test_cached_measurements_match_full_fields():
    profile, snapshots, _ = load_suite(REPO / "config/replanning/field_updates.yaml")
    updater = FieldUpdater()
    modes = []
    for metadata, occupancy in snapshots:
        expected, reference = measure_snapshot(occupancy, metadata, profile, True, True)
        actual, observed = measure_snapshot(occupancy, metadata, profile, True, True, updater)
        assert actual["status"] == expected["status"]
        assert actual["reached_cells"] == expected["reached_cells"]
        assert actual["travel_time_at_start"] == expected["travel_time_at_start"]
        for name in reference:
            np.testing.assert_array_equal(observed[name], reference[name])
        mode = actual["field_update"]
        modes.append(mode)
        if mode == "field_reuse":
            assert sum(actual[k] for k in ("edt_s", "speed_s", "fmm_s", "gradient_s")) == 0
        elif mode == "map_reuse":
            assert actual["edt_s"] == actual["speed_s"] == 0
            assert actual["fmm_s"] > 0
        assert actual["compute_field_s"] >= sum(
            actual[k] for k in ("edt_s", "speed_s", "fmm_s", "gradient_s")
        )
    assert modes == ["full_recompute", "field_reuse", "map_reuse", "field_reuse",
                     "full_recompute", "full_recompute", "full_recompute", "field_reuse"]


def test_cli_resets_between_cases_and_repetitions(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"""schema_version: 1
profile: {REPO / 'config/replanning/baseline_profile.yaml'}
map:
  grid:
    shape: [21, 21]
    resolution: 0.1
    origin: [0, 0, 0]
    rectangles: [[0, 0, 21, 1, 100]]
cases:
  - id: first
    start: [0.5, 1, 0, 0, 0]
    goal: [1.5, 1, 0]
    updates: [{{id: repeated}}]
  - id: second
    start: [0.5, 1, 0, 0, 0]
    goal: [1.5, 1, 0]
    updates: [{{id: repeated}}]
""")
    script = """
import sys
from bench_replanning import main
sys.argv = ['bench_replanning', '--config', sys.argv[1], '--out', sys.argv[2],
            '--field-only', '--field-backend', 'cached', '--warmups', '1', '--repeat', '2']
assert main() == 0
assert 'rclpy' not in sys.modules
"""
    output = tmp_path / "run"
    subprocess.run([sys.executable, "-c", script, str(suite), str(output)],
                   env={**os.environ, "PYTHONPATH": str(REPO / "tools")},
                   check=True, capture_output=True, text=True)
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(records) == 12
    for record in records:
        expected = "full_recompute" if record["snapshot_id"] == "base" else "field_reuse"
        assert record["field_update"] == expected
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["arm"] == "cached_field_fresh_plan"
    summary = summarize(output)
    assert summary["field_backend"] == "cached" and summary["complete"]
    for group in summary["groups"]:
        expected = "full_recompute" if group["snapshot_id"] == "base" else "field_reuse"
        assert group["field_updates"] == {expected: 2}
        assert group["samples"] == 2
