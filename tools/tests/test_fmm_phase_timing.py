"""Check benchmark FMM phases without depending on machine speed."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import skfmm

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from agx_planning.vector_field import field as vf
    from replanning_baseline import measure
    from replanning_baseline.inputs import cell, load_suite
    from summarize_replanning import summarize
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def phase_names():
    return (*measure.FMM_PHASES, "fmm_other_s")


def simple_inputs():
    occupancy = np.zeros((3, 3), dtype=np.int8)
    metadata = {"geometry": {"shape": [3, 3], "resolution": 1.0, "origin": [0, 0, 0]},
                "start": [0.5, 0.5, 0, 0, 0], "goal": [1.5, 1.5, 0]}
    profile = {"field": {}, "occupancy_threshold": 65, "allow_unknown": False}
    return occupancy, metadata, profile


@pytest.fixture
def clocked_native(monkeypatch):
    clock = [0.0]
    output = np.arange(9, dtype=float).reshape(3, 3)

    class State:
        def __init__(self, *args, **kwargs):
            clock[0] += 1

        def replay(self, *args, **kwargs):
            clock[0] += 2

        def checkpoint(self, *args, **kwargs):
            clock[0] += 3

        def advance(self, *args, **kwargs):
            clock[0] += 4

        def result(self, *args, **kwargs):
            clock[0] += 5
            return output

    def stateless(*args, **kwargs):
        clock[0] += 6
        return output

    monkeypatch.setattr(measure.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(skfmm, "TravelTimeState", State)
    monkeypatch.setattr(skfmm, "travel_time", stateless)
    return clock, State, output


def test_phases_are_distinct_and_only_count_inside_fmm(clocked_native, monkeypatch):
    clock, State, output = clocked_native
    before = output.copy()

    def solve(*args, **kwargs):
        state = skfmm.TravelTimeState()
        state.replay()
        state.checkpoint()
        state.advance()
        assert state.result() is output
        assert skfmm.travel_time() is output
        clock[0] += 7
        return output

    def compute(*args, **kwargs):
        skfmm.travel_time()  # This call is outside the FMM wrapper.
        return vf.solve_eikonal(), "field"

    monkeypatch.setattr(vf, "solve_eikonal", solve)
    updater = SimpleNamespace(update=compute, last_update="injected", last_fmm={}, last_speed={})
    actual, stages = measure.measure_field(*simple_inputs(), updater)
    assert actual is output
    np.testing.assert_array_equal(output, before)
    expected = dict(fmm_init_s=1, fmm_replay_s=2, fmm_checkpoint_s=3,
                    fmm_advance_s=4, fmm_export_s=5, fmm_stateless_s=6, fmm_other_s=7)
    assert {name: stages[name] for name in phase_names()} == expected
    assert stages["fmm_s"] == sum(expected.values()) == 28
    assert stages["compute_field_s"] == 34
    assert "interior_update" not in stages
    assert skfmm.TravelTimeState is State
    assert vf.solve_eikonal is solve


def test_recursive_fallback_and_nested_native_calls_count_once(clocked_native, monkeypatch):
    clock, _, output = clocked_native

    def stateless(*args, **kwargs):
        clock[0] += 6
        skfmm.TravelTimeState().advance()
        return output

    def solve(*args, inner=False, **kwargs):
        clock[0] += 1
        return skfmm.travel_time() if inner else vf.solve_eikonal(inner=True)

    monkeypatch.setattr(skfmm, "travel_time", stateless)
    monkeypatch.setattr(vf, "solve_eikonal", solve)
    updater = SimpleNamespace(update=lambda *args: (vf.solve_eikonal(), "field"),
                              last_update="injected", last_fmm={}, last_speed={})
    actual, stages = measure.measure_field(*simple_inputs(), updater)
    assert actual is output
    assert stages["fmm_stateless_s"] == 11
    assert stages["fmm_other_s"] == 2
    assert stages["fmm_s"] == stages["compute_field_s"] == 13
    assert all(stages[name] == 0 for name in measure.FMM_PHASES if name != "fmm_stateless_s")


def test_stateless_measurement_works_without_state_api(clocked_native, monkeypatch):
    _, _, output = clocked_native
    monkeypatch.delattr(skfmm, "TravelTimeState")
    monkeypatch.setattr(vf, "solve_eikonal", lambda *args: skfmm.travel_time())
    monkeypatch.setattr(vf, "compute_field", lambda *args: (vf.solve_eikonal(), "field"))
    actual, stages = measure.measure_field(*simple_inputs())
    assert actual is output
    assert stages["fmm_s"] == stages["fmm_stateless_s"] == 6
    assert all(stages[name] == 0 for name in phase_names() if name != "fmm_stateless_s")


@pytest.fixture
def sample(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"""schema_version: 1
profile: {REPO / 'config/replanning/baseline_profile.yaml'}
map:
  grid:
    shape: [21, 41]
    resolution: 0.1
    origin: [0, 0, 0]
    rectangles:
      - [0, 0, 41, 1, 100]
      - [0, 20, 41, 21, 100]
      - [0, 0, 1, 21, 100]
      - [40, 0, 41, 21, 100]
cases:
  - id: phases
    start: [0.5, 1.0, 0, 0, 0]
    goal: [1.5, 1.0, 0]
    updates:
      - id: edited
        rectangles: [[33, 5, 34, 8, 100]]
      - id: repeated
""")
    profile, snapshots, _ = load_suite(suite)
    return profile, snapshots, suite


def compute_args(occupancy, metadata, profile):
    geometry = metadata["geometry"]
    start, goal = cell(metadata["start"], geometry), cell(metadata["goal"], geometry)
    return (occupancy, *goal, geometry["resolution"], *geometry["origin"][:2],
            vf.VectorFieldConfig(**profile["field"]), *start,
            profile["occupancy_threshold"], profile["allow_unknown"])


@pytest.mark.parametrize("max_checkpoints", [0, 4])
def test_measurement_keeps_fields_inputs_and_controller_stats_exact(sample, max_checkpoints):
    profile, snapshots, _ = sample
    options = dict(incremental=True, checkpoint_interval=10, max_checkpoints=max_checkpoints)
    measured, reference = FieldUpdater(**options), FieldUpdater(**options)
    for metadata, occupancy in snapshots:
        before = occupancy.copy()
        expected, _ = reference.update(*compute_args(occupancy, metadata, profile))
        actual, stages = measure.measure_field(occupancy, metadata, profile, measured)
        assert measure._equal_field_values(actual, expected)
        np.testing.assert_array_equal(occupancy, before)
        assert measured._eikonal.last_stats == reference._eikonal.last_stats
        assert measured.last_fmm == reference.last_fmm
        assert all(not name.endswith("_s") for name in measured.last_fmm)
        assert all(stages[name] >= 0 for name in phase_names())
        assert sum(stages[name] for name in phase_names()) == pytest.approx(stages["fmm_s"])
        if metadata["snapshot_id"] == "repeated":
            assert stages["field_update"] == "field_reuse"
            assert stages["fmm_s"] == 0
            assert all(stages[name] == 0 for name in phase_names())


def test_verification_runs_after_native_patches_are_restored(sample, monkeypatch):
    profile, snapshots, _ = sample
    metadata, occupancy = snapshots[0]
    expected = vf.compute_field(*compute_args(occupancy, metadata, profile))
    originals = {name: getattr(skfmm.TravelTimeState, name)
                 for name in ("__init__", "replay", "checkpoint", "advance", "result")}
    stateless = skfmm.travel_time
    clock = [0.0]
    monkeypatch.setattr(measure.time, "perf_counter", lambda: clock[0])

    def reference(*args, **kwargs):
        assert skfmm.travel_time is stateless
        assert all(getattr(skfmm.TravelTimeState, name) is value for name, value in originals.items())
        clock[0] += 50
        return deepcopy(expected)

    def cached(*args, **kwargs):
        clock[0] += 2
        return deepcopy(expected)

    monkeypatch.setattr(vf, "compute_field", reference)
    updater = SimpleNamespace(update=cached, last_update="field_reuse", last_fmm={}, last_speed={})
    record, _ = measure.measure_snapshot(occupancy, metadata, profile, field_only=True,
                                         updater=updater, verify_fields=True)
    assert record["verification"]["status"] == "match"
    assert record["verification"]["wall_s"] == 50
    assert record["total_wall_s"] == record["compute_field_s"] == 2
    assert record["fmm_s"] == 0
    assert all(record[name] == 0 for name in phase_names())


def write_summary_run(directory, samples, snapshots, repeat):
    manifest = {"schema_version": 1, "profile_sha256": "profile", "environment": {},
                "options": {"repeat": repeat, "warmups": 0, "field_only": True},
                "snapshots": [{"case_id": "phases", "snapshot_id": name,
                               "input_sha256": f"input-{index}"}
                              for index, name in enumerate(snapshots)]}
    records = [{"schema_version": 1, "profile_sha256": "profile",
                "input_sha256": f"input-{row['snapshot_index']}", "warmup": False,
                "status": "field_only", "solves": [], **row} for row in samples]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (directory / "COMPLETE").touch()


def test_summary_uses_per_repetition_phase_totals(tmp_path):
    samples = []
    for repetition, values in enumerate([(1, 100), (100, 1), (101, 101)]):
        for index, value in enumerate(values):
            samples.append({"snapshot_index": index, "repetition": repetition,
                            **{name: value * (phase + 1) for phase, name in enumerate(phase_names())}})
    write_summary_run(tmp_path, samples, ["base", "edited"], repeat=3)
    report = summarize(tmp_path)
    assert report["complete"]
    for phase, name in enumerate(phase_names()):
        values = report["case_totals"][0]["metrics"][name]
        assert values["n"] == 3
        assert values["median"] == 101 * (phase + 1)
        assert sum(group["metrics"][name]["median"] for group in report["groups"]) == 200 * (phase + 1)


def test_summary_accepts_old_records_without_phase_timings(tmp_path):
    write_summary_run(tmp_path, [{"snapshot_index": 0, "repetition": 0,
                                 "fmm_s": 2, "total_wall_s": 3}], ["base"], repeat=1)
    report = summarize(tmp_path)
    empty = {"n": 0, "median": None, "p95": None, "min": None, "max": None}
    for metrics in (report["groups"][0]["metrics"], report["case_totals"][0]["metrics"]):
        assert metrics["fmm_s"]["median"] == 2
        assert all(metrics[name] == empty for name in phase_names())


def test_cli_writes_phase_fields_and_verifies_all_snapshots(sample, tmp_path):
    _, _, suite = sample
    output = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": str(REPO / "src/agx_navigation/agx_planning"),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    process = subprocess.run(
        [sys.executable, str(REPO / "tools/bench_replanning.py"), "--config", str(suite),
         "--field-only", "--field-backend", "incremental", "--checkpoint-interval", "10",
         "--warmups", "0", "--repeat", "1", "--verify-fields", "--out", str(output)],
        cwd=REPO, env=environment, text=True, capture_output=True, timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert all(record["verification"]["status"] == "match" for record in records)
    for record in records:
        assert all(isinstance(record[name], (float, int)) and record[name] >= 0 for name in phase_names())
        assert sum(record[name] for name in phase_names()) == pytest.approx(record["fmm_s"])
    assert records[-1]["field_update"] == "field_reuse"
    assert all(records[-1][name] == 0 for name in phase_names())
    report = summarize(output)
    assert report["complete"]
    assert all(group["metrics"][name]["n"] == 1 for group in report["groups"] for name in phase_names())
