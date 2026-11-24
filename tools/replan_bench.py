#!/usr/bin/env python3
"""Run the ROS-free full-stop execution reference with production FMM and BVP."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))


def run(args):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import numpy as np
    import yaml
    from agx_planning.maps import GridGeometry, MapSnapshot, OccupancyPolicy
    from agx_planning.maps.io import load_map
    from agx_planning.profiles import load_profile, expand_profile, profile_digest, planner_config
    from agx_planning.pmp_planner.plan_store import save_plan
    from agx_planning.replanning.braking import BrakeConfig
    from agx_planning.replanning.execution import ExecutionConfig, FullStopRunner, ProductionPlanner
    from agx_planning.replanning.reveal import HiddenObject, ObjectRevealer
    from agx_planning.replanning.validity import Footprint
    from replanning_baseline.inputs import provenance, write_json, array_digest

    path = Path(args.config).resolve()
    spec = yaml.safe_load(path.read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("expected execution schema_version 1")
    profile = load_profile(path.parent / spec["profile"])
    # Explicit precision change; the expanded profile and its new digest are
    # stored. The existing packed_float32 field benchmark remains unchanged.
    profile["field_path"] = "raw_float64"
    profile["name"] = profile.get("name", "profile") + "-identified-float64"
    profile = expand_profile(profile)
    occupancy, geometry, map_source = load_map(spec["map"], path.parent)
    snapshot = MapSnapshot(spec.get("session", "r1-experiment"), 0,
                           GridGeometry.from_dict(geometry), occupancy,
                           OccupancyPolicy(profile["occupancy_threshold"], profile["allow_unknown"]))
    objects = [HiddenObject(item["id"], item["rectangles"]) for item in spec.get("objects", [])]
    sensor = spec.get("sensor", {"max_range": 1.0})
    revealer = ObjectRevealer(snapshot, objects, **sensor)
    config = ExecutionConfig(**spec.get("execution", {}))
    brake = BrakeConfig(**spec["brake"])
    cfg = planner_config(profile)
    footprint = Footprint(**spec.get("footprint", {}))
    environment = provenance()
    # Include the parent and native revision and binary hash in the numerical
    # identity, rather than identifying a locally rebuilt module by its name.
    numerical = (environment["git_head"] + ":" + ";".join(environment["submodules"])
                 + ":" + environment["skfmm_binary_sha256"])
    planner = ProductionPlanner(profile, numerical, session_id=snapshot.session)
    runner = FullStopRunner(revealer, planner, c_v=cfg.c_v, c_w=cfg.c_w,
                           config=config, brake=brake, footprint=footprint)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "profile": profile, "profile_sha256": profile_digest(profile),
                "scenario": spec, "map_source": map_source,
                "base_map_sha256": array_digest(snapshot.occupancy),
                "true_map_sha256": array_digest(revealer.true_map.occupancy),
                "environment": environment, "options": vars(args)}
    write_json(output / "manifest.json", manifest)
    np.savez(output / "inputs.npz", revealed_map=snapshot.occupancy, true_map=revealer.true_map.occupancy)
    schedule = [(item["time"], item["ids"]) for item in spec.get("scheduled_reveals", [])]
    try:
        result = runner.run(spec["start"], spec["goal"], scheduled_reveals=schedule)
    except Exception as error:
        write_json(output / "failure.json", {"status": "exception", "type": type(error).__name__,
                                             "message": str(error)})
        raise
    np.savez(output / "execution.npz", times=result.times, states=result.states)
    with (output / "events.jsonl").open("w") as stream:
        for event in result.events:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
    for index, plan in enumerate(result.plans):
        save_plan(plan, output / f"plan_{index:03d}")
    write_json(output / "result.json", {"status": result.status, "message": result.message,
                                       **result.metadata, "plan_count": len(result.plans)})
    (output / "COMPLETE").write_text("The requested execution and its outcome were recorded.\n")
    print(f"{result.status}: {result.message} ({result.metadata['execution_s']:.3f}s execution)")
    return 0 if result.status == "success" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
