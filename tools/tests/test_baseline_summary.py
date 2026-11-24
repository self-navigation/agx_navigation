import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_replanning import summarize


def test_summary_keeps_failures_excludes_warmups_and_handles_partial_runs(tmp_path):
    manifest = {"schema_version": 1, "profile_sha256": "profile", "environment": {},
                "options": {"repeat": 2, "warmups": 1, "field_only": False},
                "snapshots": [{"case_id": "route", "snapshot_id": "base", "input_sha256": "input"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows = [{"schema_version": 1, "profile_sha256": "profile", "input_sha256": "input",
             "snapshot_index": 0, "repetition": i, "warmup": i == -1,
             "status": "completed" if i < 1 else "bvp_failed", "total_wall_s": duration,
             "solves": [{"success": i < 1, "wall_s": duration / 2}]}
            for i, duration in [(-1, 100), (0, 2), (1, 4)]]
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = summarize(tmp_path)
    assert not report["complete"]
    group = report["groups"][0]
    assert group["outcomes"] == {"completed": 1, "bvp_failed": 1}
    assert group["metrics"]["total_wall_s"]["median"] == 3
    assert group["metrics"]["completed_total_wall_s"]["median"] == 2
    assert group["metrics"]["compute_field_s"]["n"] == 0
    assert group["bvp_failures"] == 1
    (tmp_path / "COMPLETE").touch()
    assert summarize(tmp_path)["complete"]
    results.write_text(results.read_text() + json.dumps(rows[-1]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(tmp_path)


def write_run(tmp_path, snapshot_ids, samples, repeat=3, warmups=0, extra_options=None):
    snapshots = [{"case_id": case, "snapshot_id": snapshot, "input_sha256": f"input-{index}"}
                 for index, (case, snapshot) in enumerate(snapshot_ids)]
    options = {"repeat": repeat, "warmups": warmups, "field_only": True, **(extra_options or {})}
    manifest = {"schema_version": 1, "profile_sha256": "profile", "environment": {},
                "options": options, "snapshots": snapshots}
    rows = []
    for sample in samples:
        index = sample["snapshot_index"]
        rows.append({"schema_version": 1, "profile_sha256": "profile",
                     "input_sha256": snapshots[index]["input_sha256"],
                     "status": "field_only", "solves": [],
                     "warmup": sample["repetition"] < 0, **sample})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "COMPLETE").touch()
    return manifest


def test_case_totals_sum_each_repetition_before_taking_medians(tmp_path):
    samples = []
    for repetition, durations in enumerate([(1, 10, 100), (100, 20, 1), (101, 30, 101)]):
        for index, duration in enumerate(durations):
            samples.append({"snapshot_index": index, "repetition": repetition,
                            "total_wall_s": duration,
                            "status": "start_unreached" if (repetition, index) == (1, 2)
                            else "field_only"})
    samples.extend({"snapshot_index": index, "repetition": -1, "total_wall_s": 10000}
                   for index in range(3))
    write_run(tmp_path, [("first", "base"), ("second", "base"), ("first", "added")],
              samples, warmups=1)
    report = summarize(tmp_path)
    assert report["complete"]
    first, second = report["case_totals"]
    assert first["case_id"] == "first"
    assert first["snapshot_ids"] == ["base", "added"]
    assert first["expected_repetitions"] == first["complete_repetitions"] == 3
    assert first["incomplete_repetitions"] == []
    assert first["outcomes"] == {"field_only": 5, "start_unreached": 1}
    totals = first["metrics"]["total_wall_s"]
    assert totals["n"] == 3
    assert totals["median"] == totals["min"] == 101
    assert totals["max"] == 202
    assert sum(report["groups"][index]["metrics"]["total_wall_s"]["median"]
               for index in (0, 2)) == 200
    assert second["case_id"] == "second"
    assert second["snapshot_ids"] == ["base"]
    assert second["metrics"]["total_wall_s"]["median"] == 20
    assert second["outcomes"] == {"field_only": 3}


def test_case_totals_report_partial_and_entirely_missing_repetitions(tmp_path):
    samples = [{"snapshot_index": index, "repetition": repetition, "total_wall_s": duration,
                "status": status}
               for repetition, entries in [
                   (0, [(0, 1, "field_only"), (1, 2, "field_only"), (2, 3, "field_only")]),
                   (1, [(0, 4, "field_only"), (2, 6, "exception")]),
                   (3, [(0, 10, "field_only"), (1, 20, "bvp_failed"), (2, 30, "field_only")]),
               ] for index, duration, status in entries]
    write_run(tmp_path, [("route", name) for name in ("base", "added", "removed")],
              samples, repeat=4)
    report = summarize(tmp_path)
    assert not report["complete"]
    case = report["case_totals"][0]
    assert case["expected_repetitions"] == 4
    assert case["complete_repetitions"] == 2
    assert case["incomplete_repetitions"] == [
        {"repetition": 1, "missing_snapshot_ids": ["added"], "samples": 2,
         "outcomes": {"field_only": 1, "exception": 1}},
        {"repetition": 2, "missing_snapshot_ids": ["base", "added", "removed"],
         "samples": 0, "outcomes": {}},
    ]
    assert case["outcomes"] == {"field_only": 6, "exception": 1, "bvp_failed": 1}
    assert case["metrics"]["total_wall_s"]["n"] == 2
    assert case["metrics"]["total_wall_s"]["median"] == 33


def test_case_metrics_require_every_snapshot_and_keep_verification_separate(tmp_path):
    samples = [
        {"snapshot_index": 0, "repetition": 0, "total_wall_s": 2, "compute_field_s": 1,
         "speed_s": 0, "verification": {"status": "match", "wall_s": 0.5}},
        {"snapshot_index": 1, "repetition": 0, "total_wall_s": 3, "compute_field_s": None,
         "speed_s": 2, "verification": {"status": "match", "wall_s": 0.7}},
        {"snapshot_index": 0, "repetition": 1, "total_wall_s": 4, "compute_field_s": 2,
         "speed_s": None, "verification": {"status": "error", "wall_s": 0.9}},
        {"snapshot_index": 1, "repetition": 1, "total_wall_s": 5, "compute_field_s": 3,
         "speed_s": 0, "verification": {"status": "match", "wall_s": 0.1}},
        {"snapshot_index": 0, "repetition": 2, "total_wall_s": 6, "speed_s": 0},
        {"snapshot_index": 1, "repetition": 2, "total_wall_s": 7, "compute_field_s": 4,
         "speed_s": 0, "verification": {"status": "match", "wall_s": 2}},
    ]
    write_run(tmp_path, [("route", "base"), ("route", "added")], samples)
    report = summarize(tmp_path)
    case = report["case_totals"][0]
    assert case["complete_repetitions"] == 3
    metrics = case["metrics"]
    assert metrics["total_wall_s"]["median"] == 9
    assert metrics["compute_field_s"]["n"] == 1
    assert metrics["compute_field_s"]["median"] == 5
    assert metrics["speed_s"]["n"] == 2
    assert metrics["speed_s"]["median"] == 1
    assert metrics["verification_wall_s"]["n"] == 2
    assert metrics["verification_wall_s"]["median"] == pytest.approx(1.1)
    assert metrics["total_cpu_s"] == {"n": 0, "median": None, "p95": None,
                                      "min": None, "max": None}
    assert report["groups"][1]["metrics"]["compute_field_s"]["n"] == 2


def test_native_work_filters_unknown_values_and_warmups_and_keeps_options(tmp_path):
    fmm_work = {"reused_batches": 0, "total_batches": 20, "computed_batches": 20,
                "checkpoint_count": 2, "checkpoint_copies": 4,
                "checkpoint_copy_bytes": 400, "cache_array_bytes": 500}
    speed_work = {"changed_cells": 5, "updated_cells": 9, "computed_cells": 100,
                  "total_cells": 200, "region_count": 2}
    samples = [
        {"snapshot_index": 0, "repetition": -1, "total_wall_s": 100,
         "fmm_update": {"status": "replay", "reason": "checkpoint_replayed",
                        "initialization_check": "full_comparison",
                        **{key: 10000 for key in fmm_work}},
         "speed_update": {"status": "local_update", "reason": "changed_region",
                          **{key: 10000 for key in speed_work}}},
        {"snapshot_index": 0, "repetition": 0, "total_wall_s": 1,
         "fmm_update": {"status": "replay", "reason": "checkpoint_replayed",
                        "initialization_check": "unchanged_support", **fmm_work},
         "speed_update": {"status": "local_update", "reason": "changed_region", **speed_work}},
        {"snapshot_index": 0, "repetition": 1, "total_wall_s": 2,
         "fmm_update": {"status": "full_recompute", "reason": "cache_budget",
                        "reused_batches": None, "total_batches": None, "computed_batches": None,
                        "checkpoint_count": 0, "checkpoint_copies": 0,
                        "checkpoint_copy_bytes": 0, "cache_array_bytes": 0},
         "speed_update": {"status": "unchanged", "reason": "inputs_unchanged",
                          "changed_cells": 0, "updated_cells": 0, "computed_cells": 0,
                          "total_cells": 200, "region_count": None}},
        {"snapshot_index": 0, "repetition": 2, "total_wall_s": 3},
    ]
    manifest = write_run(tmp_path, [("route", "base")], samples, warmups=1,
                         extra_options={"field_backend": "incremental", "checkpoint_interval": 64,
                                        "future_settings": {"enabled": True, "labels": ["small", 3]}})
    report = summarize(tmp_path)
    assert report["options"] == manifest["options"]
    group = report["groups"][0]
    assert set(group["fmm_work"]) == set(fmm_work)
    assert set(group["speed_work"]) == set(speed_work)
    for name in ("reused_batches", "total_batches", "computed_batches"):
        assert group["fmm_work"][name]["n"] == 1
        assert group["fmm_work"][name]["median"] == fmm_work[name]
    for name in ("checkpoint_count", "checkpoint_copies", "checkpoint_copy_bytes", "cache_array_bytes"):
        assert group["fmm_work"][name]["n"] == 2
        assert group["fmm_work"][name]["median"] == fmm_work[name] / 2
    for name in ("changed_cells", "updated_cells", "computed_cells"):
        assert group["speed_work"][name]["n"] == 2
        assert group["speed_work"][name]["median"] == speed_work[name] / 2
    assert group["speed_work"]["total_cells"]["median"] == 200
    assert group["speed_work"]["region_count"]["n"] == 1
    assert group["speed_work"]["region_count"]["median"] == 2
    assert group["fmm_updates"] == {"replay": 1, "full_recompute": 1}
    assert group["fmm_initialization_checks"] == {"unchanged_support": 1}
    assert group["speed_updates"] == {"local_update": 1, "unchanged": 1}


def test_old_results_have_empty_work_counters_and_valid_case_totals(tmp_path):
    write_run(tmp_path, [("old", "base")],
              [{"snapshot_index": 0, "repetition": 0, "total_wall_s": 4}], repeat=1)
    report = summarize(tmp_path)
    assert report["complete"]
    assert report["field_backend"] == "full"
    group = report["groups"][0]
    empty = {"n": 0, "median": None, "p95": None, "min": None, "max": None}
    assert group["fmm_work"] and group["speed_work"]
    assert all(values == empty for values in group["fmm_work"].values())
    assert all(values == empty for values in group["speed_work"].values())
    assert group["interior_updates"] == group["interior_reasons"] == {}
    assert group["fmm_initialization_checks"] == {}
    assert group["interior_work"]
    assert all(values == empty for values in group["interior_work"].values())
    case = report["case_totals"][0]
    assert case["complete_repetitions"] == 1
    assert case["metrics"]["total_wall_s"]["median"] == 4
    assert case["metrics"]["verification_wall_s"] == empty


def test_interior_work_counts_retries_and_omits_unknown_values_and_warmups(tmp_path):
    counters = ("changed_cells", "updated_cells", "computed_cells", "total_cells",
                "region_count", "transform_calls")
    samples = [
        {"snapshot_index": 0, "repetition": -1,
         "interior_update": {"status": "full_recompute", "reason": "new_map",
                             **dict.fromkeys(counters, 10000)}},
        {"snapshot_index": 0, "repetition": 0,
         "interior_update": {"status": "local_update", "reason": "changed_region",
                             "changed_cells": 3, "updated_cells": 16, "computed_cells": 30,
                             "total_cells": 100, "region_count": 1, "transform_calls": 1}},
        {"snapshot_index": 0, "repetition": 1,
         "interior_update": {"status": "full_recompute", "reason": "work_limit",
                             "changed_cells": 4, "updated_cells": 100, "computed_cells": 150,
                             "total_cells": 100, "region_count": 1, "transform_calls": 3}},
        {"snapshot_index": 0, "repetition": 2,
         "interior_update": {"status": "unchanged", "reason": "inputs_unchanged",
                             **dict.fromkeys(counters, 0), "total_cells": 100}},
        {"snapshot_index": 0, "repetition": 3,
         "interior_update": {"status": "unchanged", "reason": "inputs_unchanged",
                             "changed_cells": None, "transform_calls": None}},
        {"snapshot_index": 0, "repetition": 4},
    ]
    write_run(tmp_path, [("route", "updated")], samples, repeat=5, warmups=1)
    group = summarize(tmp_path)["groups"][0]
    assert group["interior_updates"] == {"local_update": 1, "full_recompute": 1, "unchanged": 2}
    assert group["interior_reasons"] == {"changed_region": 1, "work_limit": 1, "inputs_unchanged": 2}
    assert set(group["interior_work"]) == set(counters)
    assert all(values["n"] == 3 for values in group["interior_work"].values())
    assert group["interior_work"]["changed_cells"]["median"] == 3
    assert group["interior_work"]["computed_cells"]["median"] == 30
    assert group["interior_work"]["computed_cells"]["max"] == 150
    assert group["interior_work"]["transform_calls"]["median"] == 1
    assert group["interior_work"]["transform_calls"]["min"] == 0


def test_field_timing_includes_interior_checks_and_copies_without_nested_edt(monkeypatch):
    from types import SimpleNamespace
    import numpy as np

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo / "src/agx_navigation/agx_planning"))
    _rclpy_mock = sys.modules.pop("rclpy", None)
    try:
        from replanning_baseline import measure
    finally:
        if _rclpy_mock is not None:
            sys.modules["rclpy"] = _rclpy_mock

    clock = [0.0]
    monkeypatch.setattr(measure.time, "perf_counter", lambda: clock[0])

    def edt(obstacles):
        clock[0] += 3
        return np.zeros_like(obstacles, dtype=float)

    diagnostics = {"status": "local_update", "reason": "changed_region",
                   "computed_cells": 9, "transform_calls": 1}

    def update_interior(obstacles, resolution, previous):
        clock[0] += 2  # Bounds and source checks.
        result = measure.vf.distance_transform_edt(obstacles)
        clock[0] += 2  # Copy the accepted region into the result.
        return result, diagnostics

    monkeypatch.setattr(measure.vf, "distance_transform_edt", edt)
    monkeypatch.setattr(measure.vf, "_update_interior_distance", update_interior)
    updater = SimpleNamespace(last_update="full_recompute", last_speed={}, last_fmm={})

    def compute(occupancy, *args):
        measure.vf.distance_transform_edt(occupancy)
        _, updater.last_interior = measure.vf._update_interior_distance(occupancy, 1.0, None)
        return "field", "message"

    updater.update = compute
    metadata = {"geometry": {"shape": [3, 3], "resolution": 1.0, "origin": [0, 0, 0]},
                "start": [0.5, 0.5, 0, 0, 0], "goal": [1.5, 1.5, 0]}
    profile = {"field": {}, "occupancy_threshold": 65, "allow_unknown": False}
    result, stages = measure.measure_field(np.zeros((3, 3)), metadata, profile, updater)
    assert result == "field"
    assert stages["edt_s"] == stages["compute_field_s"] == 10
    assert stages["speed_s"] == stages["fmm_s"] == stages["gradient_s"] == 0
    assert stages["interior_update"] == diagnostics
    stages["interior_update"]["computed_cells"] = 123
    assert updater.last_interior["computed_cells"] == 9

    # Older callers and test doubles do not need to provide the optional record.
    del updater.last_interior
    updater.last_update = "field_reuse"
    updater.update = lambda *args: ("cached", "message")
    result, stages = measure.measure_field(np.zeros((3, 3)), metadata, profile, updater)
    assert result == "cached"
    assert stages["field_update"] == "field_reuse"
    assert stages["edt_s"] == stages["compute_field_s"] == 0
    assert "interior_update" not in stages
