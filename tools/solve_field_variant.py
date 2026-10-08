#!/usr/bin/env python3
"""Re-solve a plan library's start/goal pairs under named vector-field variants.

    .venv/bin/python tools/solve_field_variant.py --lib <traj_data dir> --out <dir> --variant base='{}' --variant align='{"bottleneck_align": true}' [--jobs 3]

Writes <out>/<variant>/<plan>.npz in the traj_data_v2 schema. Each variant is
the centred field recipe with its VectorFieldConfig overrides on top, solved
with bench_pmp_rollout.STACK_PLANNER_PARAMS. Re-solving does not reproduce a
stored library exactly, so solve every compared arm, base included, in one
invocation. Failed solves get status != success and no poses file.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning"),
                os.path.join(R, "src/rudn-ordjo-building")]
FIELD = dict(speed_profile="exponential", inflation_radius=0.5, centring=True, centring_cap=0.8,
             speed_decay_rate=1.5, speed_v_min=0.1)


def solve(job):
    src, out, overrides = job
    if os.path.exists(out):
        return out, "cached"
    from bench_pmp_rollout import STACK_PLANNER_PARAMS
    from replan_footprint import load_floor
    from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
    from agx_planning.pmp_planner import rollout as ro
    from agx_planning.vector_field import VectorFieldConfig, compute_field
    from agx_planning.vector_field.field import field_result_to_grid
    z = np.load(src, allow_pickle=True)
    start, goal_xy = z["start_xy"], z["goal_xy"]
    arr, meta = load_floor(6)
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    g = lambda p: (int((p[0] - ox) / r), int((p[1] - oy) / r))  # noqa: E731
    gc, sc = g(goal_xy), g(start)
    res, _ = compute_field(arr, gc[0], gc[1], r, ox, oy, VectorFieldConfig(**dict(FIELD, **overrides)),
                           sc[0], sc[1], 65, allow_unknown=True)
    cfg = PlannerConfig(mode="offline")
    for k, v in STACK_PLANNER_PARAMS.items():
        setattr(cfg, k, v)
    solver = PMPShootingSolver(cfg, field_result_to_grid(res, field_eps=cfg.field_eps,
                                                         wall_sigma=cfg.fp_smooth_sigma))
    solver.reset_warm_start()
    x0 = np.array([*start, float(z["poses"][0, 2]), 0.0, 0.0])
    goal = np.array([*goal_xy, float(z["poses"][-1, 2])])
    poses, cmds, cos, status = [], [], [], "?"
    try:
        gen = ro.rollout_generator(solver, cfg, x0.copy(), goal)
        while True:
            ch = next(gen)
            poses.append(ch.poses); cmds.append(ch.wheel_cmds); cos.append(ch.costates)
    except StopIteration as stop:
        status = str(getattr(stop.value, "status", "?"))
    except Exception as e:  # noqa: BLE001 -- a crashed solve is a failed solve
        status = "exception:" + repr(e)[:120]
    if status != "success" or not poses:
        return out, status
    keep = {k: z[k] for k in z.files if k not in ("poses", "wheel_cmds", "costates", "dt_sample")}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, poses=np.concatenate(poses), wheel_cmds=np.concatenate(cmds),
             costates=np.concatenate(cos), dt_sample=1.0 / cfg.control_rate, **keep)
    return out, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", action="append", required=True, help="name=JSON overrides")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    variants = dict(v.split("=", 1) for v in a.variant)
    files = sorted(f for f in os.listdir(a.lib) if f.endswith(".npz"))
    jobs = [(os.path.join(a.lib, f), os.path.join(a.out, name, f), json.loads(js))
            for f in files for name, js in variants.items()]
    with mp.Pool(a.jobs) as pool:
        for out, status in pool.imap_unordered(solve, jobs):
            print(os.path.relpath(out, a.out), status, flush=True)


if __name__ == "__main__":
    main()
