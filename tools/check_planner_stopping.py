#!/usr/bin/env python3
"""Check planner stopping on the six saved planner inputs from a full collection.

Run the current checkout with its current worker and an existing native build.
The previous collection is read-only. This is a correctness check, not a new
matched timing experiment.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
SUITES = {
    "planner-00-synthetic": (("straight", "base"), ("straight", "leaf"),
                             ("straight", "closed"), ("straight", "reopened"),
                             ("already_at_goal", "base")),
    "planner-01-floor6_smoke": (("floor6_straight", "base"),),
}
REQUIRED = {("straight", "base"), ("floor6_straight", "base"),
            ("already_at_goal", "base")}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker_environment():
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        env.pop(name, None)
    env.update({name: "1" for name in THREADS})
    env.update(PYTHONNOUSERSITE="1", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    return env


def read_results(directory, expected):
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    keys = [(row["case_id"], row["snapshot_id"]) for row in rows]
    complete = json.loads((directory / "COMPLETE").read_text())
    if (len(keys) != len(expected) or set(keys) != set(expected)
            or complete.get("recorded_rows") != len(expected)
            or complete.get("expected_rows") != len(expected)
            or complete.get("verification_failures") != 0):
        raise ValueError(f"Incomplete or duplicate planner records: {directory.name}")
    result = []
    for row in rows:
        if (row.get("phase") != "measure" or row.get("repetition") != 0
                or row.get("warmup_index") is not None
                or row.get("experiment") != "planner" or row.get("backend") != "full"):
            raise ValueError(f"Unexpected planner record identity: {directory.name}")
        key = row["case_id"], row["snapshot_id"]
        evaluation = row["evaluation"]
        accepted = (row["status"] == "completed"
                    and evaluation.get("common_numerical_acceptance") is True)
        result.append({"case_id": key[0], "snapshot_id": key[1],
                       "status": row["status"], "required": key in REQUIRED,
                       "accepted": accepted, "evaluation": evaluation})
    return result


def run(args):
    collection = Path(args.collection).resolve()
    output = Path(args.out).resolve()
    if output == collection or output.is_relative_to(collection):
        raise ValueError("Output must be outside the previous collection")
    output.mkdir(parents=True, exist_ok=False)
    # Do not resolve the venv executable symlink to the base interpreter.
    python = Path(args.python).absolute() if args.python else collection / "work/envs/after/bin/python"
    summary = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
               "collection": str(collection), "repo": str(REPO), "python": str(python),
               "scope": "targeted_planner_stopping_check", "status": "running",
               "expected_rows": sum(map(len, SUITES.values())), "commands": [], "results": []}
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    try:
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError("After venv is unavailable; use --python with an existing compatible venv")
        for suite, expected in SUITES.items():
            inputs = collection / "inputs" / suite
            manifest = json.loads((inputs / "manifest.json").read_text())
            keys = [(row["case_id"], row["snapshot_id"]) for row in manifest["snapshots"]]
            if len(keys) != len(expected) or set(keys) != set(expected):
                raise ValueError(f"Saved planner inputs differ from the six-case check: {suite}")
            if not (inputs / "inputs.npz").is_file():
                raise ValueError(f"Saved planner arrays are missing: {suite}")
        for suite, expected in SUITES.items():
            command = [str(python), str(REPO / "tools/revision_compare/worker.py"),
                       "--repo", str(REPO), "--inputs", str(collection / "inputs" / suite),
                       "--experiment", "planner", "--backend", "full",
                       "--out", str(output / suite), "--repetition", "0",
                       "--phase", "measure", "--warmups", "0"]
            log = output / (suite + ".log")
            entry = {"suite": suite, "command": command, "log": log.name}
            summary["commands"].append(entry)
            write_json(summary_path, summary)
            print(f"Checking {suite}", flush=True)
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=REPO, env=worker_environment(),
                                           stdout=stream, stderr=subprocess.STDOUT, timeout=900)
            entry["returncode"] = completed.returncode
            if completed.returncode:
                raise ValueError(f"Planner worker failed; see {log}")
            results = read_results(output / suite, expected)
            summary["results"].extend(results)
            for row in results:
                detail = "accepted" if row["accepted"] else row["status"]
                if row["required"] and not row["accepted"]:
                    detail = "acceptance FAILED"
                print(f"{row['case_id']}/{row['snapshot_id']}: {detail}", flush=True)
            write_json(summary_path, summary)
        failed = [row for row in summary["results"] if row["required"] and not row["accepted"]]
        summary.update(status="acceptance_failed" if failed else "passed",
                       failed_required_cases=[f"{row['case_id']}/{row['snapshot_id']}" for row in failed])
        code = int(bool(failed))
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        summary.update(status="collection_error", error=f"{type(error).__name__}: {error}")
        code = 2
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(summary_path, summary)
    print(f"Planner stopping check: {summary['status']}; details: {summary_path}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="run_data/kt4-kt5-collection")
    parser.add_argument("--out", required=True, help="new directory outside the previous collection")
    parser.add_argument("--python", help="existing venv Python with the matching native dependency")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
