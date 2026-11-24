"""Audit raw before/after records without computing performance summaries.

Every planned attempt remains visible, including missing and failed work.
Duplicate or malformed rows are retained but never selected as measurements.
Cross-field comparisons use verification artifacts only and require identical
input/profile identities. They are numerical checks, not a speedup analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


PLAN_FIELDS = ("block_id", "arm", "revision", "experiment", "suite", "phase",
               "repetition", "warmup_index", "snapshot_index", "case_id",
               "snapshot_id", "pair_id")
ROW_KEY = ("phase", "warmup_index", "snapshot_index")
GROUP_FIELDS = ("block_id", "arm", "revision", "experiment", "suite", "phase",
                "repetition", "warmup_index", "case_id", "profile_sha256")
PAIR_FIELDS = ("pair_id", "experiment", "suite", "case_id", "snapshot_id",
               "repetition", "warmup_index", "snapshot_index")
FIELD_CHANNELS = ("travel_time", "reached_mask", "obstacle_mask", "grad_x",
                  "grad_y", "grad_mag", "wall_dist")


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _loads(value):
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("JSON number is outside the finite float range")
        return value
    return json.loads(value, parse_constant=_reject_constant, parse_float=finite_float)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _within(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact and block paths must be nonempty relative paths")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError("artifact or block path escapes its collection directory")
    return result


def _key(value):
    phase, warmup, index = (value.get(name) for name in ROW_KEY)
    if phase not in ("verify", "warmup", "measure"):
        raise ValueError("invalid record phase")
    if type(index) is not int or index < 0:
        raise ValueError("snapshot_index must be a nonnegative integer")
    if phase == "warmup":
        if type(warmup) is not int or warmup < 0:
            raise ValueError("warmup rows require a nonnegative warmup_index")
    elif warmup is not None:
        raise ValueError("non-warmup rows require a null warmup_index")
    return phase, warmup, index


def _load_plan(root):
    planned = _loads((root / "planned_runs.json").read_text())
    blocks = _loads((root / "blocks.json").read_text())
    if not isinstance(planned, list) or not isinstance(blocks, list):
        raise ValueError("planned_runs.json and blocks.json must contain lists")
    block_map = {}
    for block in blocks:
        if (not isinstance(block, dict) or not isinstance(block.get("block_id"), str)
                or not block["block_id"] or block["block_id"] in block_map):
            raise ValueError("block identities must be unique nonempty strings")
        block_map[block["block_id"]] = block
    seen = set()
    for row in planned:
        if not isinstance(row, dict) or any(name not in row for name in PLAN_FIELDS):
            raise ValueError("planned row lacks required identity fields")
        for name in ("block_id", "arm", "revision", "experiment", "suite", "case_id",
                     "snapshot_id", "pair_id"):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError(f"planned {name} must be a nonempty string")
        if type(row["repetition"]) is not int:
            raise ValueError("planned repetition must be an integer")
        identity = row["block_id"], *_key(row)
        if identity in seen:
            raise ValueError("planned run keys are duplicated")
        seen.add(identity)
        block = block_map.get(row["block_id"])
        if block:
            for name in ("arm", "revision", "experiment", "suite", "repetition"):
                if name in block and block[name] != row[name]:
                    raise ValueError(f"planned and block {name} differ")
    return planned, block_map


def _flatten(value, prefix="", output=None):
    output = {} if output is None else output
    for name, child in value.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(child, dict):
            _flatten(child, key, output)
        elif isinstance(child, list):
            output[key] = json.dumps(child, separators=(",", ":"), allow_nan=False)
        else:
            output[key] = child
    return output


def _write_csv(path, rows, leading):
    columns = list(leading)
    columns.extend(sorted(set().union(*(row.keys() for row in rows)) - set(columns)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _numeric_fields(left, right):
    import numpy as np
    result = {"left_shape": list(left.shape), "right_shape": list(right.shape),
              "left_dtype": str(left.dtype), "right_dtype": str(right.dtype)}
    if left.shape != right.shape:
        return {**result, "status": "shape_mismatch", "equal": False}
    if left.dtype.kind not in "biuf" or right.dtype.kind not in "biuf":
        raise ValueError("field arrays must contain real numeric or boolean values")
    equal_values = bool(np.array_equal(left, right, equal_nan=True))
    equal_dtype = left.dtype == right.dtype
    result.update(equal=equal_values and equal_dtype,
                  values_equal=equal_values, dtype_equal=equal_dtype,
                  bitwise_equal=equal_dtype and left.tobytes() == right.tobytes(),
                  status=("match" if equal_values and equal_dtype else "mismatch"))
    if left.dtype.kind in "biu" and right.dtype.kind in "biu":
        result["differing_cells"] = int(np.count_nonzero(left != right))
        return result
    common = np.isfinite(left) & np.isfinite(right)
    result["left_finite_cells"] = int(np.isfinite(left).sum())
    result["right_finite_cells"] = int(np.isfinite(right).sum())
    result["finite_mask_equal"] = bool(np.array_equal(np.isfinite(left), np.isfinite(right)))
    result["common_finite_cells"] = int(common.sum())
    result["differing_cells"] = int(np.count_nonzero(~((left == right) | (np.isnan(left) & np.isnan(right)))))
    result["max_abs_error"] = result["max_relative_error"] = None
    if common.any():
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            difference = np.abs(left[common].astype(float) - right[common].astype(float))
            denominator = np.maximum(np.maximum(np.abs(left[common]), np.abs(right[common])),
                                     np.finfo(float).tiny)
            absolute, relative = float(difference.max()), float((difference / denominator).max())
        result["max_abs_error"] = absolute if np.isfinite(absolute) else None
        result["max_relative_error"] = relative if np.isfinite(relative) else None
        result["error_overflow"] = not np.isfinite(absolute) or not np.isfinite(relative)
    return result


def _load_fields(root, row):
    import numpy as np
    path = _within(root, row["artifact_path"])
    if not isinstance(row.get("artifact_sha256"), str):
        raise ValueError("artifact SHA-256 is missing")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["artifact_sha256"]:
        raise ValueError("artifact SHA-256 differs from its raw record")
    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in FIELD_CHANNELS if name in archive.files}
    required = set(FIELD_CHANNELS) - {"wall_dist"}
    if required - fields.keys():
        raise ValueError("verification artifact lacks required raw field channels")
    shape = tuple(row["geometry"]["shape"])
    if any(array.shape != shape for array in fields.values()):
        raise ValueError("raw field shape differs from input geometry")
    if not np.array_equal(fields["reached_mask"], np.isfinite(fields["travel_time"])):
        raise ValueError("saved reached mask differs from finite travel-time cells")
    return fields


def compare_fields(root, rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("phase") == "verify" and all(name in row for name in PAIR_FIELDS):
            grouped[tuple(row[name] for name in PAIR_FIELDS)][row["arm"]].append(row)
    reports = []
    for identity, arms in grouped.items():
        for left_arm, right_arm in (("before_full", "after_full"),
                                    ("after_full", "after_incremental")):
            # Synthetic missing rows retain every planned arm. An absent arm
            # here was not planned, so do not invent an unsupported comparison.
            if left_arm not in arms or right_arm not in arms:
                continue
            report = {**dict(zip(PAIR_FIELDS, identity)), "left_arm": left_arm,
                      "right_arm": right_arm, "phase": "verify"}
            left, right = arms.get(left_arm, []), arms.get(right_arm, [])
            if len(left) > 1 or len(right) > 1:
                reports.append({**report, "status": "duplicate_pair"})
                continue
            if not left or not right or any(row.get("audit_status") != "recorded" for row in left + right):
                reports.append({**report, "status": "missing_pair"})
                continue
            left, right = left[0], right[0]
            report.update(left_revision=left["revision"], right_revision=right["revision"])
            required = ("input_sha256", "map_sha256", "profile_sha256", "geometry", "start", "goal")
            mismatch = [name for name in required if name not in left or name not in right
                        or left[name] != right[name]]
            if mismatch:
                reports.append({**report, "status": "input_mismatch", "differing_inputs": mismatch})
                continue
            report.update(input_sha256=left["input_sha256"], profile_sha256=left["profile_sha256"])
            if not left.get("artifact_path") or not right.get("artifact_path"):
                reports.append({**report, "status": ("artifact_error" if left.get("artifact_path_error")
                                                     or right.get("artifact_path_error") else "unavailable"),
                                "left_status": left.get("status"),
                                "right_status": right.get("status")})
                continue
            try:
                a, b = _load_fields(root, left), _load_fields(root, right)
                channels = {}
                for name in FIELD_CHANNELS:
                    if name not in a and name not in b:
                        continue
                    channels[name] = (_numeric_fields(a[name], b[name]) if name in a and name in b
                                      else {"status": "missing_channel", "equal": False,
                                            "left_present": name in a, "right_present": name in b})
                reports.append({**report, "status": "match" if all(item["equal"] for item in channels.values()) else "mismatch",
                                "channels": channels, "left_artifact": left["artifact_path"],
                                "right_artifact": right["artifact_path"]})
            except (OSError, ValueError, TypeError, KeyError) as error:
                reports.append({**report, "status": "artifact_error", "message": str(error)})
    return {"schema_version": 1, "scope": "verification_only_raw_field_values",
            "comparisons": reports, "outcomes": dict(Counter(row["status"] for row in reports))}


def audit_collection(root):
    root = Path(root).resolve()
    planned, blocks = _load_plan(root)
    expected = defaultdict(dict)
    for row in planned:
        expected[row["block_id"]][_key(row)] = row
    rows, issues, counts = [], [], Counter()
    groups = {}

    def group_for(row):
        key = tuple(row.get(name) for name in GROUP_FIELDS)
        if key not in groups:
            groups[key] = {**dict(zip(GROUP_FIELDS, key)), "expected": 0, "recorded": 0,
                           "missing": 0, "duplicate": 0, "unexpected": 0,
                           "malformed": 0, "identity_errors": 0, "outcomes": Counter()}
        return groups[key]

    def issue(code, block_id, **details):
        issues.append({"code": code, "block_id": block_id, **details})

    for block_id in dict.fromkeys([row["block_id"] for row in planned] + list(blocks)):
        block = blocks.get(block_id, {"block_id": block_id})
        block_expected = expected[block_id]
        context = {name: block.get(name) for name in GROUP_FIELDS}
        if block_expected:
            context.update({name: next(iter(block_expected.values())).get(name) for name in GROUP_FIELDS})
        for plan in block_expected.values():
            group_for(plan)["expected"] += 1
        found = defaultdict(list)
        source = None
        try:
            directory = _within(root, block.get("out", f"runs/{block_id}"))
            source = directory / "results.jsonl"
            if source.exists():
                with source.open() as stream:
                    for number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        counts["worker_rows"] += 1
                        provenance = {"source_file": str(source.relative_to(root)), "source_line": number,
                                      "block_id": block_id, "recorded": True}
                        try:
                            raw = _loads(line)
                            if not isinstance(raw, dict):
                                raise ValueError("worker row must be an object")
                            key = _key(raw)
                        except (ValueError, TypeError) as error:
                            rows.append({**context, **provenance, "audit_status": "malformed",
                                         "status": "malformed", "measured": False, "raw_line": line.rstrip("\n"),
                                         "audit_message": str(error)})
                            counts["malformed"] += 1
                            group_for(context)["malformed"] += 1
                            issue("malformed_row", block_id, line=number, message=str(error))
                            continue
                        plan = block_expected.get(key)
                        if plan is None:
                            rows.append({**raw, **context, **provenance, "worker_record": raw,
                                         "audit_status": "unexpected", "measured": False})
                            counts["unexpected"] += 1
                            group_for(context)["unexpected"] += 1
                            issue("unexpected_row", block_id, line=number, key=list(key))
                            continue
                        mismatch = [name for name in ("repetition", "case_id", "snapshot_id", "experiment")
                                    if raw.get(name) != plan[name]]
                        if type(raw.get("repetition")) is not int and "repetition" not in mismatch:
                            mismatch.append("repetition")
                        if raw.get("git_head") != plan["revision"]:
                            mismatch.append("git_head")
                        for name in ("arm", "revision", "block_id", "suite", "pair_id"):
                            if name in raw and raw[name] != plan[name]:
                                mismatch.append(name)
                        for name in ("input_sha256", "map_sha256", "profile_sha256", "geometry", "start", "goal"):
                            if name in plan and raw.get(name) != plan[name]:
                                mismatch.append(name)
                        if not isinstance(raw.get("status"), str) or not raw["status"]:
                            mismatch.append("status")
                        if mismatch:
                            rows.append({**raw, **plan, **provenance, "worker_record": raw,
                                         "audit_status": "identity_mismatch", "measured": False,
                                         "differing_identity": mismatch})
                            counts["identity_errors"] += 1
                            group_for(plan)["identity_errors"] += 1
                            issue("row_identity_mismatch", block_id, line=number, fields=mismatch)
                            continue
                        row = {**raw, **plan, **provenance}
                        if raw.get("artifact"):
                            try:
                                row["artifact_path"] = str(_within(directory, raw["artifact"]).relative_to(root))
                            except ValueError as error:
                                row["artifact_path_error"] = str(error)
                                issue("artifact_path_error", block_id, line=number, message=str(error))
                        found[key].append(row)
            else:
                issue("missing_results_file", block_id)
            completion = directory / "COMPLETE"
            if completion.exists():
                complete = _loads(completion.read_text())
                if (not isinstance(complete, dict)
                        or complete.get("expected_rows") != len(block_expected)
                        or complete.get("recorded_rows") != sum(len(items) for items in found.values())):
                    issue("completion_count_mismatch", block_id)
            else:
                issue("missing_completion_marker", block_id)
        except (OSError, ValueError, TypeError) as error:
            issue("block_read_error", block_id, message=str(error))
        if block_id not in blocks:
            issue("missing_block_record", block_id)
        if block.get("timed_out"):
            issue("block_timed_out", block_id)
        elif block.get("returncode") not in (None, 0):
            issue("block_failed", block_id, returncode=block["returncode"])
        for key, plan in block_expected.items():
            matches = found[key]
            group = group_for(plan)
            if not matches:
                status = ("timeout" if block.get("timed_out") else "process_failed"
                          if block.get("returncode") not in (None, 0) else "missing")
                rows.append({**plan, "status": status, "audit_status": "missing",
                             "recorded": False, "measured": False, "synthetic": True,
                             "block_returncode": block.get("returncode"),
                             "block_timed_out": bool(block.get("timed_out"))})
                counts["missing"] += 1
                group["missing"] += 1
                group["outcomes"][status] += 1
                continue
            counts["recorded"] += 1
            group["recorded"] += 1
            duplicate = len(matches) > 1
            if duplicate:
                counts["duplicate"] += len(matches) - 1
                group["duplicate"] += len(matches) - 1
                issue("duplicate_row", block_id, key=list(key), count=len(matches))
            for row in matches:
                row.update(audit_status="duplicate" if duplicate else "recorded",
                           measured=not duplicate and row["phase"] == "measure",
                           block_returncode=block.get("returncode"),
                           block_timed_out=bool(block.get("timed_out")))
                rows.append(row)
                group["outcomes"][row["status"]] += 1
                verification = row.get("verification")
                if isinstance(verification, dict) and verification.get("status") in ("mismatch", "error"):
                    issue("worker_verification_" + verification["status"], block_id,
                          line=row["source_line"], key=list(key), verification=verification)
                elif (row["phase"] == "verify" and row["status"] != "unsupported"
                      and (not isinstance(verification, dict)
                           or verification.get("status") not in ("match", "mismatch", "error", "skipped"))):
                    issue("missing_verification_record", block_id, line=row["source_line"], key=list(key))
    cross = compare_fields(root, rows)
    for comparison in cross["comparisons"]:
        if comparison["status"] in ("input_mismatch", "artifact_error"):
            issue("cross_field_" + comparison["status"], None, comparison=comparison)
        elif (comparison["left_arm"] == "after_full" and comparison["right_arm"] == "after_incremental"
              and comparison["status"] == "mismatch"):
            issue("incremental_field_mismatch", None, comparison=comparison)
    summary = {"schema_version": 1, "expected": len(planned),
               **{name: counts[name] for name in ("recorded", "missing", "duplicate", "worker_rows",
                                                  "unexpected", "malformed", "identity_errors")},
               "errors": len(issues), "issues": issues,
               "data_complete": not issues and counts["missing"] == 0,
               "cross_field_outcomes": cross["outcomes"],
               "worker_outcomes": dict(Counter(row.get("status", "unknown") for row in rows
                                               if row.get("audit_status") == "recorded")),
               "analysis_scope": "raw_records_and_completeness_only"}
    with (root / "runs.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    _write_csv(root / "raw_metrics.csv", [_flatten(row) for row in rows],
               (*PLAN_FIELDS, "audit_status", "status", "recorded", "measured"))
    completeness = [{**group, "outcomes": json.dumps(dict(group["outcomes"]), sort_keys=True)}
                    for group in groups.values()]
    _write_csv(root / "completeness.csv", completeness, (*GROUP_FIELDS, "expected", "recorded", "missing", "duplicate"))
    _write_json(root / "cross_fields.json", cross)
    _write_json(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    try:
        result = audit_collection(args.root)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Audit error: {error}\n")
    print(f"expected={result['expected']} recorded={result['recorded']} missing={result['missing']} "
          f"duplicate={result['duplicate']} errors={result['errors']}")
    return 0 if result["data_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
