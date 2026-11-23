#!/usr/bin/env python3
"""Measure field construction and cold-start PMP plans without ROS."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))


def run(args):
    # Set thread limits before importing NumPy or SciPy.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import numpy as np
    from replanning_baseline.inputs import digest, load_suite, provenance, write_json
    from replanning_baseline.measure import measure_snapshot
    from agx_planning.vector_field import FieldUpdater

    setup = time.perf_counter()
    profile, snapshots, source = load_suite(args.config, args.case)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    updater = None
    if args.field_backend == "cached":
        updater = FieldUpdater()
    elif args.field_backend == "incremental":
        updater = FieldUpdater(incremental=True, checkpoint_interval=args.checkpoint_interval,
                               max_checkpoints=args.max_checkpoints,
                               cache_bytes=args.field_cache_mib * 1024**2)
    arm = ("full_recompute_fresh_plan" if updater is None
           else f"{args.field_backend}_field_fresh_plan")
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "arm": arm, "profile": profile, "profile_sha256": digest(profile),
                "source": source, "snapshots": [m for m, _ in snapshots], "environment": provenance(),
                "options": vars(args), "setup_wall_s": time.perf_counter() - setup}
    write_json(output / "manifest.json", manifest)
    np.savez(output / "inputs.npz", **{f"map_{i}": a for i, (_, a) in enumerate(snapshots)})
    with (output / "results.jsonl").open("w") as stream:
        for repetition in range(args.warmups + args.repeat):
            warmup = repetition < args.warmups
            for index, (metadata, occupancy) in enumerate(snapshots):
                if updater is not None and metadata["snapshot_id"] == "base":
                    updater.clear()
                record, arrays = measure_snapshot(occupancy, metadata, profile,
                                                   args.field_only, args.save_fields, updater)
                record.update(schema_version=1, snapshot_index=index, **metadata,
                              repetition=repetition - args.warmups, warmup=warmup,
                              profile_sha256=manifest["profile_sha256"])
                if arrays and not warmup:
                    filename = f"sample_{repetition - args.warmups:04d}_{index:04d}.npz"
                    io_start = time.perf_counter()
                    np.savez(output / filename, **arrays)
                    record.update(artifact=filename, artifact_write_s=time.perf_counter() - io_start)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                print(f"{metadata['case_id']}/{metadata['snapshot_id']} "
                      f"{'warmup' if warmup else record['repetition']}: "
                      f"{record['status']} ({record['total_wall_s']:.3f}s)", flush=True)
    (output / "COMPLETE").write_text("All requested samples were recorded.\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--case", action="append", default=[], help="select case ID; repeatable")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--field-only", action="store_true")
    parser.add_argument("--field-backend", choices=("full", "cached", "incremental"), default="full",
                        help="select full computation, cached fields, or experimental FMM replay")
    parser.add_argument("--checkpoint-interval", type=int, default=10000,
                        help="equal-time batches between incremental checkpoints")
    parser.add_argument("--max-checkpoints", type=int, default=4,
                        help="maximum retained incremental checkpoints")
    parser.add_argument("--field-cache-mib", type=int, default=256,
                        help="incremental native working and checkpoint array budget in MiB")
    parser.add_argument("--save-fields", action="store_true", help="save raw fields outside the timed region")
    args = parser.parse_args()
    if min(args.repeat, args.threads) < 1 or args.warmups < 0:
        parser.error("repeat and threads must be positive; warmups must be nonnegative")
    if args.checkpoint_interval < 1 or args.max_checkpoints < 0 or args.field_cache_mib < 0:
        parser.error("checkpoint interval must be positive; checkpoint count and cache MiB must be nonnegative")
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
