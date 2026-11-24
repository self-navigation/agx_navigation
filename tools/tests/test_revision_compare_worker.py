"""The common worker must not require features absent from the old revision."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]
_mock = sys.modules.pop("rclpy", None)
try:
    from replanning_baseline.inputs import load_suite
    from revision_compare import inputs, measure, worker
    from agx_planning.pmp_planner import PlannerConfig, parse_field_array
finally:
    if _mock is not None:
        sys.modules["rclpy"] = _mock


@pytest.fixture
def sequence():
    profile, snapshots, _ = load_suite(REPO / "config/replanning/synthetic.yaml", ["straight"])
    return {"profile": profile, "snapshots": [metadata for metadata, _ in snapshots]}, {
        f"map_{i}": value for i, (_, value) in enumerate(snapshots)}


def test_common_instrumentation_accepts_historical_optional_timer_absence(sequence, monkeypatch):
    manifest, arrays = sequence
    calls = []
    fake = SimpleNamespace(VectorFieldConfig=lambda **kwargs: kwargs)
    fake.solve_eikonal = lambda: calls.append("march")
    fake.skfmm = SimpleNamespace(travel_time=lambda *a, **k: None)

    def compute(*args):
        fake.solve_eikonal()
        return object(), "computed"

    fake.compute_field = compute
    monkeypatch.setattr(measure, "vf", fake)
    result, stages = measure.measure_field(arrays["map_0"], manifest["snapshots"][0], manifest["profile"])
    assert result is not None and calls == ["march"]
    assert stages["field_update"] == "full_recompute"
    assert stages["fmm_s"] >= 0
    assert fake.solve_eikonal() is None


def test_common_input_hashes_and_sequence_order_are_checked(sequence):
    manifest, arrays = sequence
    worker.validate_sequence(manifest, arrays)
    wrong = {key: value.copy() for key, value in arrays.items()}
    wrong["map_0"][10, 10] = 100
    with pytest.raises(ValueError, match="hash"):
        worker.validate_sequence(manifest, wrong)
    wrong_manifest = deepcopy(manifest)
    wrong_manifest["snapshots"][0]["start"][0] += 0.01
    with pytest.raises(ValueError, match="metadata hash"):
        worker.validate_sequence(wrong_manifest, arrays)
    wrong_manifest = deepcopy(manifest)
    wrong_manifest["snapshots"][0]["snapshot_id"] = "edit"
    with pytest.raises(ValueError, match="starting with base"):
        worker.validate_sequence(wrong_manifest, arrays)


def test_common_complete_field_profile_is_explicit(sequence):
    manifest, arrays = sequence
    for key, value in (("early_exit_enable", True), ("bottleneck_align", True)):
        wrong = deepcopy(manifest)
        wrong["profile"]["field"][key] = value
        with pytest.raises(ValueError):
            worker.validate_sequence(wrong, arrays)
    wrong = deepcopy(manifest)
    wrong["profile"]["field_path"] = "raw_float64"
    with pytest.raises(ValueError, match="packed_float32"):
        worker.validate_sequence(wrong, arrays)


def test_common_floor_lookup_rejects_negative_outside_and_upper_extent():
    geometry = {"shape": [10, 20], "resolution": 0.1, "origin": [-1., -2., 0.]}
    assert inputs.cell([-0.95, -1.95], geometry) == (0, 0)
    for point in ([-1.00001, -1.95], [1., -1.95], [-0.95, -1.], [float("nan"), -1.5]):
        with pytest.raises(ValueError):
            inputs.cell(point, geometry)


def test_selected_module_and_native_prefix_guards(tmp_path):
    core, prefix = tmp_path / "core", tmp_path / "venv"
    worker.check_module_paths(core, {"agx_planning": core / "agx_planning/__init__.py"},
                              prefix / "lib/native.so", prefix)
    with pytest.raises(ValueError, match="outside the selected"):
        worker.check_module_paths(core, {"agx_planning": tmp_path / "old/core.py"},
                                  prefix / "lib/native.so", prefix)
    with pytest.raises(ValueError, match="outside the revision-local"):
        worker.check_module_paths(core, {}, tmp_path / "other/native.so", prefix)


def test_new_bvp_failure_does_not_commit_an_old_spline(sequence, monkeypatch):
    manifest, arrays = sequence
    profile, metadata = manifest["profile"], manifest["snapshots"][0]
    field, _ = measure.measure_field(arrays["map_0"], metadata, profile)
    grid = parse_field_array(measure.vf.pack_field_array(field), PlannerConfig(**profile["planner"]))
    original = measure.shooting_solver.solve_bvp
    calls = 0

    def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("injected current solve failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(measure.shooting_solver, "solve_bvp", failing)
    result, trajectory = measure.measure_rollout(grid, metadata, profile)
    assert result["status"] == "bvp_failed"
    assert len(trajectory["states"]) == len(trajectory["wheel_cmds"]) + 1
    assert result["chunks"] == 1
    assert len(result["solves"]) == 2


def test_final_speed_and_controls_are_independent_of_success_label(sequence):
    manifest, _ = sequence
    arrays = {"states": np.array([[0., 0., 0., 2., 2.]]), "accels": np.empty((0, 2))}
    record = {"status": "completed", "pose_tolerance_met": True}
    worker.evaluate_planner(record, arrays, manifest["profile"], {"goal_wheel_speed_rad_s": 0.5})
    assert not record["evaluation"]["goal_speed_tolerance_met"]
    assert not record["evaluation"]["common_numerical_acceptance"]
    assert not record["evaluation"]["collision_checked"]


def test_field_verification_artifact_has_masks_and_no_pickle(sequence):
    manifest, arrays = sequence
    result, raw = measure.measure_snapshot(arrays["map_0"], manifest["snapshots"][0],
                                           manifest["profile"], field_only=True,
                                           save_fields=True, verify_fields=True)
    assert result["verification"]["status"] == "match"
    assert {"travel_time", "grad_x", "grad_y", "grad_mag", "reached_mask", "obstacle_mask"} <= raw.keys()
    np.testing.assert_array_equal(raw["reached_mask"], np.isfinite(raw["travel_time"]))
    assert all(value.dtype != object for value in raw.values())


def test_worker_preserves_warmups_failures_and_atomic_rows(sequence, tmp_path, monkeypatch):
    manifest, arrays = sequence
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    np.savez(directory / "inputs.npz", **arrays)
    provenance = {"git_head": "test-revision", "capabilities": {"incremental_field": False}}
    monkeypatch.setattr(worker, "configure_imports", lambda repo: REPO)
    monkeypatch.setattr(worker, "environment", lambda repo, core: provenance)
    calls = []

    def collect(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "field_failed", "total_wall_s": 0.01}, {}

    monkeypatch.setattr(measure, "measure_snapshot", collect)
    args = argparse.Namespace(repo=str(REPO), inputs=str(directory), experiment="planner", backend="full",
                              out=str(tmp_path / "out"), repetition=2, phase="measure", warmups=1)
    assert worker.run(args) == 0
    output = Path(args.out)
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * len(manifest["snapshots"])
    count = len(manifest["snapshots"])
    assert [row["phase"] for row in rows] == ["warmup"] * count + ["measure"] * count
    assert [row["warmup_index"] for row in rows] == [0] * count + [None] * count
    assert all(row["status"] == "field_failed" for row in rows)
    assert all(row["trajectory_scope"] == "initial_state_only" for row in rows)
    assert all(not call["verify_fields"] and not call["save_fields"] for call in calls)
    assert len(list((output / "records").glob("*.json"))) == len(rows)
    assert json.loads((output / "COMPLETE").read_text())["recorded_rows"] == len(rows)
    for row in rows:
        with np.load(output / row["artifact"], allow_pickle=False) as artifact:
            assert artifact["states"].shape == (1, 5)


def test_unsupported_backend_still_records_every_planned_input(sequence, tmp_path, monkeypatch):
    manifest, arrays = sequence
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    np.savez(directory / "inputs.npz", **arrays)
    monkeypatch.setattr(worker, "configure_imports", lambda repo: REPO)
    monkeypatch.setattr(worker, "environment", lambda repo, core: {
        "git_head": "old", "capabilities": {"incremental_field": False}})
    args = argparse.Namespace(repo=str(REPO), inputs=str(directory), experiment="field", backend="incremental",
                              out=str(tmp_path / "out"), repetition=0, phase="verify", warmups=0)
    worker.run(args)
    rows = [json.loads(line) for line in (Path(args.out) / "results.jsonl").read_text().splitlines()]
    assert len(rows) == len(manifest["snapshots"])
    assert all(row["status"] == "unsupported" for row in rows)
    assert all(row["phase"] == "verify" for row in rows)
