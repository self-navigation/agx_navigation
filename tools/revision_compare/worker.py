#!/usr/bin/env python3
"""Run one frozen input sequence against one isolated production checkout.

The harness lives outside the selected checkout. Both revisions use these same
measurement adapters. The caller builds each native extension in its own venv.
Timing excludes verification, artifact writes, and process setup. The common
planner adapter rejects stale failed solves; this is guarded kernel collection,
not an untouched legacy execution experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Commit each record atomically; a killed worker cannot corrupt old rows."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def git(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def configure_imports(repo):
    """Choose exactly one production package root before importing NumPy/core."""
    repo = Path(repo).resolve()
    core = repo / "src/agx_navigation/agx_planning"
    if not (core / "agx_planning/pmp_planner/shooting_solver.py").is_file():
        raise ValueError("selected checkout does not contain the common planner")
    if any(name == "agx_planning" or name.startswith("agx_planning.") for name in sys.modules):
        raise ValueError("production modules were imported before selecting a revision")
    harness = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(core), str(harness)]
    os.environ["PYTHONNOUSERSITE"] = "1"
    return core.resolve()


def check_module_paths(core, modules, native_path, prefix):
    for name, path in modules.items():
        if not Path(path).resolve().is_relative_to(core):
            raise ValueError(f"{name} was loaded outside the selected checkout: {path}")
    if not Path(native_path).resolve().is_relative_to(Path(prefix).resolve()):
        raise ValueError("native extension is outside the revision-local venv; install a fresh non-editable wheel")


def environment(repo, core):
    import numpy as np
    import scipy
    import skfmm
    from skfmm import cfmm
    from agx_planning.pmp_planner import PMPShootingSolver, RolloutChunk
    from agx_planning import vector_field

    modules = {name: str(Path(module.__file__).resolve())
               for name, module in sys.modules.items()
               if (name == "agx_planning" or name.startswith("agx_planning."))
               and getattr(module, "__file__", None)}
    native_path = Path(cfmm.__file__).resolve()
    check_module_paths(core, modules, native_path, sys.prefix)
    if sys.prefix == sys.base_prefix:
        raise ValueError("comparison worker requires a revision-local virtual environment")
    capabilities = {
        "full_field": hasattr(vector_field, "compute_field"),
        "incremental_field": hasattr(vector_field, "FieldUpdater") and hasattr(skfmm, "TravelTimeState"),
        "explicit_solve_outcomes": hasattr(PMPShootingSolver, "solve_outcome"),
        "raw_rollout_boundaries": "raw_states" in getattr(RolloutChunk, "__dataclass_fields__", {}),
        "identified_field": (core / "agx_planning/vector_field/contracts.py").is_file(),
        "full_stop_execution_runner": (core / "agx_planning/replanning/execution.py").is_file(),
    }
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "git_head": git(repo, "rev-parse", "HEAD"),
        "git_status": git(repo, "status", "--porcelain", "--untracked-files=normal"),
        "tracked_diff_sha256": hashlib.sha256(git(repo, "diff", "HEAD", "--binary").encode()).hexdigest(),
        "submodules": git(repo, "submodule", "status", "--recursive").splitlines(),
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "packages": dict(sorted((distribution.metadata["Name"], distribution.version)
                                for distribution in importlib.metadata.distributions())),
        "production_modules": modules,
        "skfmm_path": str(Path(skfmm.__file__).resolve()),
        "skfmm_binary_path": str(native_path),
        "skfmm_binary_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "capabilities": capabilities,
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }


def validate_sequence(manifest, inputs):
    """Validate the entire sequence before charging any sample time."""
    from revision_compare.inputs import validate_input

    profile = manifest["profile"]
    if profile["field_path"] != "packed_float32":
        raise ValueError("the common historical planner path requires packed_float32")
    if profile["field"].get("early_exit_enable"):
        raise ValueError("matched field collection requires complete fields")
    if profile["field"].get("bottleneck_align"):
        raise ValueError("the frozen common profile requires bottleneck shaping disabled")
    snapshots = manifest["snapshots"]
    if not snapshots:
        raise ValueError("prepared sequence is empty")
    seen, closed_cases = set(), set()
    current_case = None
    for index, metadata in enumerate(snapshots):
        pair = metadata["case_id"], metadata["snapshot_id"]
        if pair in seen:
            raise ValueError("prepared sequence has duplicate case/snapshot keys")
        seen.add(pair)
        if pair[0] != current_case:
            if pair[0] in closed_cases or pair[1] != "base":
                raise ValueError("each case must be one contiguous sequence starting with base")
            if current_case is not None:
                closed_cases.add(current_case)
            current_case = pair[0]
        validate_input(metadata, inputs[f"map_{index}"])


def evaluate_planner(record, arrays, profile, evaluation):
    """Report common numerical final-state criteria independently of solver status.

    Geometric and physical execution success are deliberately not inferred from
    a converged BVP, finite samples, or these final-state tolerances.
    """
    import numpy as np

    tolerance = float(evaluation.get("goal_wheel_speed_rad_s", 0.5))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("goal wheel-speed tolerance must be finite and nonnegative")
    state = arrays["states"][-1]
    speed_met = bool(np.max(np.abs(state[3:])) <= tolerance)
    controls = arrays["accels"]
    control_met = bool(np.isfinite(controls).all()
                       and (not len(controls) or np.max(np.abs(controls)) <= profile["planner"]["a_wheel_max"] + 1e-6))
    finite = all(np.isfinite(value).all() for value in arrays.values())
    final_met = bool(record.get("pose_tolerance_met", False) and speed_met)
    record.update(
        evaluation={"goal_wheel_speed_rad_s": tolerance,
                    "goal_speed_tolerance_met": speed_met,
                    "sampled_acceleration_bound_met": control_met,
                    "finite_trajectory": finite,
                    "common_final_state_met": final_met,
                    "common_numerical_acceptance": bool(final_met and control_met and finite
                                                        and record["status"] == "completed"),
                    "collision_checked": False,
                    "execution_simulated": False},
    )


def save_arrays(path, arrays):
    import numpy as np

    path = Path(path)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "records").mkdir()
    started = utc_now()
    try:
        core = configure_imports(args.repo)
        import numpy as np
        from revision_compare.inputs import array_digest, digest
        from revision_compare.measure import measure_snapshot
        from agx_planning import vector_field

        manifest = json.loads((Path(args.inputs) / "manifest.json").read_text())
        with np.load(Path(args.inputs) / "inputs.npz", allow_pickle=False) as archive:
            inputs = {name: archive[name] for name in archive.files}
        validate_sequence(manifest, inputs)
        for array in inputs.values():
            array.flags.writeable = False
        profile, snapshots = manifest["profile"], manifest["snapshots"]
        provenance = environment(Path(args.repo).resolve(), core)
        write_json(output / "environment.json", provenance)
        supported = args.backend != "incremental" or provenance["capabilities"]["incremental_field"]
        updater = (vector_field.FieldUpdater(incremental=True, checkpoint_interval=10000,
                                             checkpoint_growth=1, max_checkpoints=4,
                                             cache_bytes=256 * 1024**2)
                   if supported and args.backend == "incremental" else None)
        passes = ([("warmup", i) for i in range(args.warmups)] + [("measure", None)]
                  if args.phase == "measure" else [(args.phase, 0 if args.phase == "warmup" else None)])
        expected = len(snapshots) * len(passes)
        write_json(output / "worker_protocol.json", {
            "schema_version": 1, "options": vars(args), "expected_rows": expected,
            "profile_sha256": digest(profile), "input_manifest_sha256": digest(manifest),
            "measurement_scope": "guarded_kernel_computation",
            "planner_transport": "packed_float32",
            "field_domain": "complete",
            "checkpoint_policy": {"interval": 10000, "growth": 1, "count": 4, "cache_bytes": 256 * 1024**2},
            "unsupported_capabilities": [name for name, value in provenance["capabilities"].items() if not value],
        })
        recorded = 0
        bad_verification = 0
        with (output / "results.jsonl").open("w") as stream:
            for phase, warmup_index in passes:
                for index, metadata in enumerate(snapshots):
                    occupancy = inputs[f"map_{index}"]
                    if updater is not None and metadata["snapshot_id"] == "base":
                        updater.clear()
                    begin = utc_now()
                    if supported:
                        record, arrays = measure_snapshot(
                            occupancy, metadata, profile, field_only=args.experiment == "field",
                            save_fields=phase == "verify", updater=updater,
                            verify_fields=phase == "verify")
                    else:
                        record, arrays = {"status": "unsupported", "reason": "incremental field API is absent"}, {}
                    end = utc_now()
                    if array_digest(occupancy) != metadata["map_sha256"]:
                        record.update(status="input_mutated", reason="production call changed the frozen input")
                    if args.experiment == "planner":
                        # Even a failed field request gets an explicit empty
                        # trajectory with its requested start, never a fake plan.
                        if "states" not in arrays:
                            arrays.update(states=np.array([metadata["start"]], float), times=np.array([0.]),
                                          wheel_cmds=np.empty((0, 2)), accels=np.empty((0, 2)),
                                          poses=np.empty((0, 3)), costates=np.empty((0, 5)))
                            record["trajectory_scope"] = "initial_state_only"
                        else:
                            record["trajectory_scope"] = "accepted_committed_prefix"
                        evaluate_planner(record, {name: arrays[name] for name in (
                            "states", "times", "wheel_cmds", "accels", "poses", "costates")},
                            profile, manifest.get("evaluation", {}))
                    record.update(schema_version=1, experiment=args.experiment, backend=args.backend,
                                  repetition=args.repetition, phase=phase, warmup_index=warmup_index,
                                  snapshot_index=index, **metadata,
                                  started_utc=begin, ended_utc=end,
                                  git_head=provenance["git_head"],
                                  profile_sha256=digest(profile),
                                  measurement_scope="guarded_kernel_computation")
                    stem = f"{recorded:05d}_{phase}_{index:04d}"
                    if arrays:
                        artifact = stem + ".npz"
                        saved = time.perf_counter()
                        record["artifact_sha256"] = save_arrays(output / artifact, arrays)
                        record["artifact"] = artifact
                        record["artifact_write_s"] = time.perf_counter() - saved
                        record["array_sha256"] = {name: array_digest(value) for name, value in arrays.items()}
                    write_json(output / "records" / (stem + ".json"), record)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    recorded += 1
                    bad_verification += record.get("verification", {}).get("status") in ("mismatch", "error")
                    print(f"{phase} {metadata['case_id']}/{metadata['snapshot_id']}: {record['status']}", flush=True)
        write_json(output / "COMPLETE", {"expected_rows": expected, "recorded_rows": recorded,
                                        "started_utc": started, "ended_utc": utc_now(),
                                        "verification_failures": bad_verification})
        return int(bad_verification > 0)
    except Exception as error:
        write_json(output / "failure.json", {"started_utc": started, "ended_utc": utc_now(),
                                             "type": type(error).__name__, "message": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--experiment", choices=("field", "planner"), required=True)
    parser.add_argument("--backend", choices=("full", "incremental"), default="full")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--phase", choices=("verify", "warmup", "measure"), required=True)
    parser.add_argument("--warmups", type=int, default=0)
    args = parser.parse_args()
    if args.warmups < 0 or (args.phase != "measure" and args.warmups):
        parser.error("warmups must be nonnegative and apply only to measure phase")
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
