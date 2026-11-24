#!/usr/bin/env python3
"""Summarize baseline samples without pooling configurations or dropping failures."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys


METRICS = ("total_wall_s", "total_cpu_s", "compute_field_s", "edt_s", "speed_s", "fmm_s",
           "fmm_init_s", "fmm_replay_s", "fmm_checkpoint_s", "fmm_advance_s",
           "fmm_export_s", "fmm_stateless_s", "fmm_other_s",
           "gradient_s", "pack_s", "parse_s", "planner_wall_s", "planner_cpu_s")
FMM_WORK = ("reused_batches", "total_batches", "computed_batches", "checkpoint_count",
            "checkpoint_copies", "checkpoint_copy_bytes", "cache_array_bytes")
SPEED_WORK = ("changed_cells", "updated_cells", "computed_cells", "total_cells", "region_count")
INTERIOR_WORK = SPEED_WORK + ("transform_calls",)


def distribution(values):
    if not values:
        return {"n": 0, "median": None, "p95": None, "min": None, "max": None}
    values = sorted(values)
    # Linear interpolation matches NumPy's default percentile convention.
    index = (len(values) - 1) * 0.95
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return {"n": len(values), "median": statistics.median(values),
            "p95": values[lo] + (values[hi] - values[lo]) * (index - lo),
            "min": values[0], "max": values[-1]}


def work_distributions(updates, names):
    return {name: distribution([update[name] for update in updates
                                if update.get(name) is not None]) for name in names}


def summarize_cases(snapshots, groups, repetitions):
    cases = defaultdict(list)
    for index, metadata in enumerate(snapshots):
        cases[metadata["case_id"]].append(index)
    totals = []
    for case_id, indices in cases.items():
        samples = [sample for index in indices for sample in groups[index]]
        by_repetition = defaultdict(dict)
        for sample in samples:
            by_repetition[sample["repetition"]][sample["snapshot_index"]] = sample
        complete, incomplete = [], []
        for repetition in range(repetitions):
            observed = by_repetition[repetition]
            missing = [snapshots[index]["snapshot_id"] for index in indices if index not in observed]
            if missing:
                incomplete.append({"repetition": repetition, "missing_snapshot_ids": missing,
                                   "samples": len(observed),
                                   "outcomes": dict(Counter(sample["status"]
                                                            for sample in observed.values()))})
            else:
                complete.append([observed[index] for index in indices])
        metrics = {
            name: distribution([sum(sample[name] for sample in repetition)
                                for repetition in complete
                                if all(sample.get(name) is not None for sample in repetition)])
            for name in METRICS
        }
        metrics["verification_wall_s"] = distribution([
            sum(sample["verification"]["wall_s"] for sample in repetition)
            for repetition in complete
            if all((sample.get("verification") or {}).get("wall_s") is not None
                   for sample in repetition)
        ])
        totals.append({"case_id": case_id,
                       "snapshot_ids": [snapshots[index]["snapshot_id"] for index in indices],
                       "expected_repetitions": repetitions, "complete_repetitions": len(complete),
                       "incomplete_repetitions": incomplete,
                       "outcomes": dict(Counter(sample["status"] for sample in samples)),
                       "metrics": metrics})
    return totals


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema")
    records = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    groups, seen = defaultdict(list), set()
    for record in records:
        if record["schema_version"] != 1 or record["profile_sha256"] != manifest["profile_sha256"]:
            raise ValueError("Result schema/profile differs from the manifest")
        metadata = manifest["snapshots"][record["snapshot_index"]]
        if record["input_sha256"] != metadata["input_sha256"]:
            raise ValueError("Result input differs from the manifest")
        key = (record["snapshot_index"], record["repetition"])
        if key in seen:
            raise ValueError("Duplicate sample")
        seen.add(key)
        if not record["warmup"]:
            groups[record["snapshot_index"]].append(record)
    summaries = []
    for index, metadata in enumerate(manifest["snapshots"]):
        samples = groups[index]
        outcomes = dict(Counter(r["status"] for r in samples))
        metrics = {key: distribution([r[key] for r in samples if r.get(key) is not None])
                   for key in METRICS}
        solves = [s for r in samples for s in r["solves"]]
        metrics["bvp_wall_s"] = distribution([s["wall_s"] for s in solves])
        metrics["completed_total_wall_s"] = distribution([r["total_wall_s"] for r in samples
                                                          if r["status"] == "completed"])
        fmm_updates = [r["fmm_update"] for r in samples if r.get("fmm_update")]
        speed_updates = [r["speed_update"] for r in samples if r.get("speed_update")]
        interior_updates = [r["interior_update"] for r in samples if r.get("interior_update")]
        verification = [r["verification"] for r in samples if r.get("verification")]
        metrics["verification_wall_s"] = distribution([r["wall_s"] for r in verification
                                                       if r.get("wall_s") is not None])
        summaries.append({**metadata, "samples": len(samples), "outcomes": outcomes,
                          "field_updates": dict(Counter(r["field_update"] for r in samples
                                                        if "field_update" in r)),
                          "fmm_updates": dict(Counter(r["status"] for r in fmm_updates)),
                          "fmm_reasons": dict(Counter(r["reason"] for r in fmm_updates)),
                          "fmm_work": work_distributions(fmm_updates, FMM_WORK),
                          "speed_updates": dict(Counter(r["status"] for r in speed_updates)),
                          "speed_reasons": dict(Counter(r["reason"] for r in speed_updates)),
                          "speed_work": work_distributions(speed_updates, SPEED_WORK),
                          "interior_updates": dict(Counter(r["status"] for r in interior_updates)),
                          "interior_reasons": dict(Counter(r["reason"] for r in interior_updates)),
                          "interior_work": work_distributions(interior_updates, INTERIOR_WORK),
                          "field_verification": dict(Counter(r["status"] for r in verification)),
                          "bvp_calls": len(solves), "bvp_failures": sum(not s["success"] for s in solves),
                          "metrics": metrics})
    options = manifest["options"]
    expected = (options["repeat"] + options["warmups"]) * len(manifest["snapshots"])
    complete = (directory / "COMPLETE").exists() and len(records) == expected
    return {"run": str(directory.resolve()), "complete": complete,
            "options": options,
            "field_backend": options.get("field_backend", "full"),
            "profile_sha256": manifest["profile_sha256"], "environment": manifest["environment"],
            "field_only": options["field_only"], "groups": summaries,
            "case_totals": summarize_cases(manifest["snapshots"], groups, options["repeat"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True, help="new summary directory")
    args = parser.parse_args()
    try:
        reports = [summarize(path) for path in args.runs]
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
        with (output / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["run", "complete", "case_id", "snapshot_id",
                                                        "samples", "outcomes", "metric", "n",
                                                        "median", "p95", "min", "max"])
            writer.writeheader()
            for report in reports:
                for group in report["groups"]:
                    base = {key: report[key] for key in ("run", "complete")}
                    base.update({key: group[key] for key in ("case_id", "snapshot_id", "samples")})
                    base["outcomes"] = json.dumps(group["outcomes"], sort_keys=True)
                    for metric, values in group["metrics"].items():
                        writer.writerow({**base, "metric": metric, **values})
                    print(f"{base['run']} {base['case_id']}/{base['snapshot_id']}: "
                          f"n={base['samples']} {base['outcomes']}")
                if not report["complete"]:
                    print(f"WARNING: incomplete run: {report['run']}", file=sys.stderr)
    except (ValueError, OSError, KeyError, IndexError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
