#!/usr/bin/env python3
"""Score recorded per-step traces in the cost functional `J`, not in metres.

WHY
---
Every corrector comparison in this project reports `max|e_cross|`. The SVCM
framework it implements is stated as `J[u] <= J*[z] + epsilon`, so `epsilon` is a
COST-FUNCTIONAL GAP and we have never computed one. `tuning/epsilon.py` holds the
pure functional (and its tests); this script is the offline driver that feeds it
real rollouts.

OFFLINE ONLY, like `plot_run.py` and the map baker: matplotlib + numpy from a
venv, never imported by anything that runs on the robot.

INPUT: the per-step CSVs written by `GazeboBridge.enable_trace`, plus the plan
`.npz` they were driven against. Note this is exactly why the ~4000 soak rollouts
cannot be scored -- `variance_probe.drive` reduces each to scalars, and `J` needs
the track.

THE CORRECTION IS RECOVERED, NOT LOGGED
---------------------------------------
The trace records the APPLIED wheel commands (`cmd0..3`) and the plan holds the
NOMINAL ones (`wheel_cmds`), so the corrector's own output is their difference,
mapped into `(dv, domega)` by the same kinematics the planner uses. That matters
for the reading: charging the nominal command again would score every corrector
for the plan's cost instead of its own (see `epsilon.py`).
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src", "agx_navigation", "agx_planning"))
from agx_planning.tuning.epsilon import CostWeights, cost_functional  # noqa: E402

# Kinematics, duplicated from PlannerConfig for the usual reason (keeping the
# offline tools free of the control stack). Keep in sync manually.
WHEEL_RADIUS = 0.08
TRACK = 0.416503
SLIP_CHI = 1.373
TRACK_EFF = TRACK * SLIP_CHI


def wheels_to_twist(w_l, w_r):
    """(w_l, w_r) -> (v, omega), the planner's map."""
    return (WHEEL_RADIUS / 2.0) * (w_l + w_r), (WHEEL_RADIUS / TRACK_EFF) * (w_r - w_l)


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def tracking_error(planned, actual):
    """(along, cross, heading), same convention as `tvlqr.tracking_error`."""
    px, py, pth = planned
    dx, dy = actual[0] - px, actual[1] - py
    c, s = np.cos(pth), np.sin(pth)
    return np.array([c * dx + s * dy, -s * dx + c * dy,
                     wrap_to_pi(actual[2] - pth)])


def score_trace(path, plan, dt_sample, weights=None):
    """One trace CSV + its plan -> (EpsilonScore, max|e_cross|, final_err)."""
    rows = [r for r in csv.DictReader(open(path)) if r["phase"] == "step"]
    if not rows:
        raise ValueError(f"{path}: no step rows")
    poses = plan["poses"]
    nominal_cmds = plan["wheel_cmds"]

    t0 = float(rows[0]["sim_time"])
    errors, corrections = [], []
    for r in rows:
        k = int(round((float(r["sim_time"]) - t0) / dt_sample))
        k = min(k, len(poses) - 1)          # playback is time-indexed; clamp at the end
        e = tracking_error(poses[k], (float(r["x"]), float(r["y"]), float(r["yaw"])))
        # joint order is [front_left, rear_left, front_right, rear_right]
        applied = wheels_to_twist(float(r["cmd0"]), float(r["cmd2"]))
        nom = wheels_to_twist(*nominal_cmds[k])
        errors.append(e)
        corrections.append((applied[0] - nom[0], applied[1] - nom[1]))

    errors = np.asarray(errors)
    final_xy = np.array([float(rows[-1]["x"]), float(rows[-1]["y"])])
    final_err = float(np.hypot(*(final_xy - poses[-1][:2])))
    score = cost_functional(errors, corrections, dt_sample, final_err, weights)
    return score, float(np.abs(errors[:, 1]).max()), final_err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="trace CSVs or directories of them")
    ap.add_argument("--plan", required=True, help="the .npz the traces were driven against")
    ap.add_argument("--label", default="", help="arm name recorded in each row")
    ap.add_argument("--out", help="JSONL to append per-rollout scores to")
    args = ap.parse_args()

    plan = np.load(args.plan)
    dt_sample = float(plan["dt_sample"])
    files = []
    for t in args.traces:
        files.extend(sorted(glob.glob(os.path.join(t, "*.csv"))) if os.path.isdir(t) else [t])

    out = open(args.out, "a") if args.out else None
    js, crosses = [], []
    for f in files:
        try:
            score, max_cross, final_err = score_trace(f, plan, dt_sample)
        except ValueError as exc:
            print(f"[skip] {os.path.basename(f)}: {exc}", file=sys.stderr)
            continue
        js.append(score.j_total)
        crosses.append(max_cross)
        rec = dict(score.as_dict(), trace=os.path.basename(f), label=args.label,
                   max_cross=max_cross, final_err=final_err,
                   plan=os.path.basename(args.plan))
        if out:
            out.write(json.dumps(rec) + "\n")
        print(f"{os.path.basename(f):45s} J={score.j_total:9.2f} "
              f"(track {score.j_tracking:8.2f} ctrl {score.j_control:7.2f} "
              f"term {score.j_terminal:8.2f})  max|e_cross|={max_cross:.3f}")
    if out:
        out.close()
    if js:
        js, crosses = np.array(js), np.array(crosses)
        print(f"\n{args.label or 'all'}: n={len(js)}  "
              f"J {js.mean():.2f} +- {js.std():.2f}   "
              f"max|e_cross| {crosses.mean():.3f} +- {crosses.std():.3f}")


if __name__ == "__main__":
    main()
