"""Raw collection auditing retains failed work and rejects ambiguous pairing."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_compare.audit import audit_collection


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def planned(arm="before_full", *, phase="measure", index=0, case="route", repetition=0):
    return {"block_id": f"{arm}-{phase}-{repetition}", "arm": arm,
            "revision": "a" * 40 if arm == "before_full" else "b" * 40,
            "experiment": "field", "suite": "test", "phase": phase,
            "repetition": repetition, "warmup_index": 0 if phase == "warmup" else None,
            "snapshot_index": index, "case_id": case, "snapshot_id": f"step-{index}",
            "pair_id": f"{phase}-{repetition}-{case}-{index}"}


def record(plan, **overrides):
    return {**{key: plan[key] for key in ("experiment", "phase", "repetition", "warmup_index",
                                         "snapshot_index", "case_id", "snapshot_id")},
            "git_head": plan["revision"], "status": "field_only", "total_wall_s": 1.25,
            "total_cpu_s": 1., "compute_field_s": 1.1,
            "geometry": {"shape": [2, 3], "origin": [0., 0., 0.], "resolution": 0.1},
            "start": [0.05, 0.05, 0., 0., 0.], "goal": [0.25, 0.15, 0.],
            "input_sha256": "input-" + str(plan["snapshot_index"]),
            "map_sha256": "map-" + str(plan["snapshot_index"]),
            "profile_sha256": "profile",
            **({"verification": {"status": "match"}} if plan["phase"] == "verify" else {}),
            **overrides}


def collection(root, plans, observations=None, *, block_changes=None):
    root.mkdir(exist_ok=True)
    observations = observations or {}
    block_changes = block_changes or {}
    blocks = {}
    for plan in plans:
        block_id = plan["block_id"]
        if block_id in blocks:
            continue
        blocks[block_id] = {**{key: plan[key] for key in ("block_id", "arm", "revision", "experiment",
                                                        "suite", "phase", "repetition")},
                            "returncode": 0, "timed_out": False, "out": f"runs/{block_id}",
                            **block_changes.get(block_id, {})}
        directory = root / blocks[block_id]["out"]
        directory.mkdir(parents=True)
        block_plans = [item for item in plans if item["block_id"] == block_id]
        rows = observations.get(block_id, [record(item) for item in block_plans])
        (directory / "results.jsonl").write_text("".join((item if isinstance(item, str) else json.dumps(item)) + "\n"
                                                        for item in rows))
        write_json(directory / "COMPLETE", {"expected_rows": len(block_plans), "recorded_rows": len(rows)})
    write_json(root / "planned_runs.json", plans)
    write_json(root / "blocks.json", list(blocks.values()))
    return blocks


def attach_fields(root, plan, *, change=None):
    directory = root / "runs" / plan["block_id"]
    path = directory / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(item for item in rows if item["snapshot_index"] == plan["snapshot_index"])
    tt = np.array([[0., 1., 2.], [3., 4., np.nan]])
    arrays = {"travel_time": tt, "reached_mask": np.isfinite(tt),
              "obstacle_mask": np.array([[False] * 3, [False, False, True]]),
              "grad_x": np.ones((2, 3)), "grad_y": np.zeros((2, 3)),
              "grad_mag": np.ones((2, 3)), "wall_dist": np.ones((2, 3))}
    if change:
        change(arrays)
    artifact = directory / f"field-{plan['snapshot_index']}.npz"
    np.savez(artifact, **arrays)
    row.update(artifact=artifact.name, artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    return artifact


def read_rows(root):
    return [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]


def test_raw_records_keep_failures_and_each_metric_without_averaging(tmp_path):
    plans = [planned(index=0), planned(index=1)]
    collection(tmp_path, plans, {plans[0]["block_id"]: [record(plans[0]),
        record(plans[1], status="exception", total_wall_s=0.3,
               exception="declared numerical failure", planner_wall_s=0.2,
               evaluation={"position_error_m": 1.5, "success": False})]})
    summary = audit_collection(tmp_path)
    assert summary["data_complete"]
    assert (summary["expected"], summary["recorded"], summary["missing"], summary["errors"]) == (2, 2, 0, 0)
    assert summary["worker_outcomes"] == {"field_only": 1, "exception": 1}
    assert all(row["measured"] and row["recorded"] for row in read_rows(tmp_path))
    with (tmp_path / "raw_metrics.csv").open() as stream:
        values = list(csv.DictReader(stream))
    assert [float(row["total_wall_s"]) for row in values] == [1.25, 0.3]
    assert values[1]["evaluation.position_error_m"] == "1.5"
    assert not any("median" in name or "mean" in name for name in values[0])


def test_missing_timeout_rows_are_explicit_and_never_measured(tmp_path):
    plans = [planned(index=0), planned(index=1)]
    collection(tmp_path, plans, {plans[0]["block_id"]: [record(plans[0])]},
               block_changes={plans[0]["block_id"]: {"timed_out": True, "returncode": -9}})
    (tmp_path / "runs" / plans[0]["block_id"] / "COMPLETE").unlink()
    summary = audit_collection(tmp_path)
    assert not summary["data_complete"]
    assert summary["recorded"] == summary["missing"] == 1
    missing = next(row for row in read_rows(tmp_path) if row["audit_status"] == "missing")
    assert missing["status"] == "timeout"
    assert missing["synthetic"] and not missing["measured"] and not missing["recorded"]


def test_duplicate_rows_are_preserved_and_all_ambiguous_samples_are_ineligible(tmp_path):
    plan = planned()
    collection(tmp_path, [plan], {plan["block_id"]: [record(plan), record(plan, total_wall_s=2.)]})
    summary = audit_collection(tmp_path)
    assert summary["expected"] == summary["recorded"] == summary["duplicate"] == 1
    rows = read_rows(tmp_path)
    assert len(rows) == 2
    assert all(row["audit_status"] == "duplicate" and not row["measured"] for row in rows)
    assert {row["total_wall_s"] for row in rows} == {1.25, 2.}


def test_malformed_unexpected_and_wrong_revision_rows_are_kept(tmp_path):
    plan = planned()
    unexpected = record(plan, snapshot_index=9)
    wrong = record(plan, git_head="wrong")
    collection(tmp_path, [plan], {plan["block_id"]: ["not JSON", "[]", '{"number":NaN}', unexpected, wrong]})
    summary = audit_collection(tmp_path)
    assert summary["malformed"] == 3
    assert summary["unexpected"] == summary["identity_errors"] == summary["missing"] == 1
    rows = read_rows(tmp_path)
    assert len(rows) == 6
    assert all(not row["measured"] for row in rows)
    assert any(row.get("worker_record", {}).get("git_head") == "wrong" for row in rows)


def test_complete_verify_pairs_compare_raw_channels_with_exact_masks(tmp_path):
    plans = [planned(arm, phase="verify") for arm in ("before_full", "after_full", "after_incremental")]
    collection(tmp_path, plans)
    for plan in plans:
        attach_fields(tmp_path, plan)
    summary = audit_collection(tmp_path)
    assert summary["data_complete"]
    cross = json.loads((tmp_path / "cross_fields.json").read_text())
    assert cross["outcomes"] == {"match": 2}
    for pair in cross["comparisons"]:
        assert pair["channels"]["reached_mask"]["differing_cells"] == 0
        assert pair["channels"]["travel_time"]["common_finite_cells"] == 5
        assert pair["channels"]["travel_time"]["max_abs_error"] == 0
    assert not any(row["measured"] for row in read_rows(tmp_path))


def test_cross_revision_difference_is_descriptive_but_incremental_difference_is_an_error(tmp_path):
    plans = [planned(arm, phase="verify") for arm in ("before_full", "after_full", "after_incremental")]
    collection(tmp_path, plans)
    attach_fields(tmp_path, plans[0])
    attach_fields(tmp_path, plans[1], change=lambda arrays: arrays["travel_time"].__setitem__((0, 1), 1.5))
    attach_fields(tmp_path, plans[2], change=lambda arrays: arrays["travel_time"].__setitem__((0, 1), 1.6))
    summary = audit_collection(tmp_path)
    cross = json.loads((tmp_path / "cross_fields.json").read_text())
    assert cross["outcomes"] == {"mismatch": 2}
    assert summary["errors"] == 1
    assert summary["issues"][0]["code"] == "incremental_field_mismatch"
    assert cross["comparisons"][0]["channels"]["travel_time"]["max_abs_error"] == 0.5


@pytest.mark.parametrize("field", ["input_sha256", "map_sha256", "profile_sha256", "goal", "geometry"])
def test_different_inputs_or_profiles_are_not_paired(tmp_path, field):
    plans = [planned(arm, phase="verify") for arm in ("before_full", "after_full")]
    altered = record(plans[1], **{field: "different"})
    collection(tmp_path, plans, {plans[1]["block_id"]: [altered]})
    summary = audit_collection(tmp_path)
    cross = json.loads((tmp_path / "cross_fields.json").read_text())
    pair = next(item for item in cross["comparisons"] if item["left_arm"] == "before_full")
    assert pair["status"] == "input_mismatch"
    assert pair["differing_inputs"] == [field]
    assert "channels" not in pair
    assert not summary["data_complete"]


def test_no_field_comparisons_use_measurement_or_warmup_files(tmp_path):
    plans = [planned(arm, phase=phase) for arm in ("before_full", "after_full")
             for phase in ("measure", "warmup")]
    collection(tmp_path, plans)
    summary = audit_collection(tmp_path)
    assert summary["cross_field_outcomes"] == {}
    assert sum(row["measured"] for row in read_rows(tmp_path)) == 2
    assert summary["data_complete"]


def test_missing_and_corrupt_artifacts_remain_visible(tmp_path):
    plans = [planned(arm, phase="verify") for arm in ("before_full", "after_full")]
    collection(tmp_path, plans)
    attach_fields(tmp_path, plans[0])
    artifact = attach_fields(tmp_path, plans[1])
    artifact.write_bytes(b"damaged")
    summary = audit_collection(tmp_path)
    assert summary["cross_field_outcomes"]["artifact_error"] == 1
    assert any(item["code"] == "cross_field_artifact_error" for item in summary["issues"])


def test_block_and_artifact_paths_cannot_escape_the_collection(tmp_path):
    plans = [planned(arm, phase="verify") for arm in ("before_full", "after_full")]
    collection(tmp_path, plans, {plans[0]["block_id"]: [record(plans[0], artifact="../../../outside.npz")]})
    summary = audit_collection(tmp_path)
    assert any(item["code"] == "artifact_path_error" for item in summary["issues"])
    blocks = json.loads((tmp_path / "blocks.json").read_text())
    blocks[0]["out"] = "../outside"
    write_json(tmp_path / "blocks.json", blocks)
    summary = audit_collection(tmp_path)
    assert any(item["code"] == "block_read_error" for item in summary["issues"])
    assert summary["missing"] == 1


def test_invalid_reached_mask_does_not_pass_an_artifact_comparison(tmp_path):
    plans = [planned(arm, phase="verify") for arm in ("before_full", "after_full")]
    collection(tmp_path, plans)
    attach_fields(tmp_path, plans[0])
    attach_fields(tmp_path, plans[1], change=lambda arrays: arrays["reached_mask"].__setitem__((0, 0), False))
    assert audit_collection(tmp_path)["cross_field_outcomes"]["artifact_error"] == 1


def test_completeness_keeps_cases_profiles_and_phases_separate(tmp_path):
    plans = [planned(index=0, case="first"), planned(index=1, case="second"),
             planned(phase="warmup", index=0, case="first")]
    collection(tmp_path, plans)
    audit_collection(tmp_path)
    with (tmp_path / "completeness.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {(row["case_id"], row["phase"]) for row in rows} == {
        ("first", "measure"), ("second", "measure"), ("first", "warmup")}


def test_audit_rejects_duplicate_planned_keys_and_cli_returns_failure_for_missing(tmp_path):
    plan = planned()
    collection(tmp_path, [plan])
    write_json(tmp_path / "planned_runs.json", [plan, plan])
    with pytest.raises(ValueError, match="duplicated"):
        audit_collection(tmp_path)
    write_json(tmp_path / "planned_runs.json", [plan])
    (tmp_path / "runs" / plan["block_id"] / "results.jsonl").unlink()
    module = Path(__file__).resolve().parents[1] / "revision_compare/audit.py"
    result = subprocess.run([sys.executable, str(module), "--root", str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert "missing=1" in result.stdout


@pytest.mark.parametrize("status", ["mismatch", "error"])
def test_per_row_verification_failures_are_errors_without_dropping_data(tmp_path, status):
    plan = planned("after_full", phase="verify")
    collection(tmp_path, [plan], {plan["block_id"]: [record(plan, verification={"status": status})]})
    summary = audit_collection(tmp_path)
    assert summary["recorded"] == 1 and summary["missing"] == 0
    assert summary["errors"] == 1 and not summary["data_complete"]
    assert summary["issues"][0]["code"] == "worker_verification_" + status
    assert read_rows(tmp_path)[0]["verification"]["status"] == status


@pytest.mark.parametrize("field", ["input_sha256", "map_sha256", "profile_sha256"])
def test_predeclared_input_hashes_are_required_to_match_the_worker(tmp_path, field):
    plan = {**planned(), field: "expected"}
    collection(tmp_path, [plan], {plan["block_id"]: [record(plan, **{field: "changed"})]})
    summary = audit_collection(tmp_path)
    assert summary["identity_errors"] == summary["missing"] == 1
    observed = next(row for row in read_rows(tmp_path) if row["audit_status"] == "identity_mismatch")
    assert observed["worker_record"][field] == "changed"
    assert observed["differing_identity"] == [field]


def test_unplanned_incremental_arm_is_not_reported_as_a_missing_pair(tmp_path):
    plans = [planned(arm, phase="verify") for arm in ("before_full", "after_full")]
    collection(tmp_path, plans)
    for plan in plans:
        attach_fields(tmp_path, plan)
    assert audit_collection(tmp_path)["cross_field_outcomes"] == {"match": 1}


def test_overflowed_json_numbers_are_preserved_as_malformed_lines(tmp_path):
    plan = planned()
    line = json.dumps(record(plan)).replace('"total_wall_s": 1.25', '"total_wall_s": 1e309')
    collection(tmp_path, [plan], {plan["block_id"]: [line]})
    summary = audit_collection(tmp_path)
    assert summary["malformed"] == summary["missing"] == 1
    assert "1e309" in next(row["raw_line"] for row in read_rows(tmp_path) if row["audit_status"] == "malformed")
