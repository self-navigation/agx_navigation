#!/usr/bin/env python3
"""Plot corrector quality against TRAINING PROGRESS, per trajectory shape.

Consumes a directory of compare_correctors outputs, one subdirectory per
checkpoint, named so the training step can be recovered:

    <sweep>/step_000000/summary.csv
    <sweep>/step_005000/summary.csv
    ...

and draws max|e_cross| vs training step, one line per trajectory, with the
identity and TVLQR baselines as horizontal reference lines.

The point is to see whether the learned corrector is IMPROVING, and on which
shapes -- a single end-of-run checkpoint cannot distinguish "still learning"
from "converged to something bad", and the 2026-07-30 comparison found the RL
corrector helping on corners while actively hurting on straights, which one
aggregate number hides completely.

OFFLINE TOOL (matplotlib in a venv), same rule as plot_run.py.

    .venv/bin/python tools/plot_checkpoints.py sweep_data --out figures
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_COLORS = {"identity": "#eb6834", "tvlqr": "#2a78d6"}
SERIES = ["#1baf7a", "#eda100", "#9b59b6", "#16a085", "#c0392b"]


def read_summary(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--metric", default="max_cross",
                    choices=("max_cross", "rms_cross", "final_err"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # step -> trajectory -> corrector -> value
    data = defaultdict(lambda: defaultdict(dict))
    baselines = defaultdict(dict)
    for summary in sorted(glob.glob(os.path.join(args.sweep_dir, "*", "summary.csv"))):
        tag = os.path.basename(os.path.dirname(summary))
        m = re.search(r"(\d+)", tag)
        if not m:
            continue
        step = int(m.group(1))
        for row in read_summary(summary):
            val = float(row[args.metric])
            traj, corr = row["trajectory"], row["corrector"]
            if corr == "rl":
                data[step][traj]["rl"] = val
            else:
                # Baselines do not depend on the checkpoint; keep the first seen.
                baselines[traj].setdefault(corr, val)

    if not data:
        raise SystemExit(f"no step_*/summary.csv under {args.sweep_dir}")

    steps = sorted(data)
    trajectories = sorted({t for s in steps for t in data[s]})

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, traj in enumerate(trajectories):
        col = SERIES[i % len(SERIES)]
        xs = [s for s in steps if "rl" in data[s].get(traj, {})]
        ys = [data[s][traj]["rl"] for s in xs]
        ax.plot(xs, ys, "-o", color=col, lw=2.0, ms=4, label=f"{traj} — RL")
        for corr, style in (("identity", ":"), ("tvlqr", "--")):
            if corr in baselines.get(traj, {}):
                ax.axhline(baselines[traj][corr], color=col, ls=style, lw=1.2,
                           alpha=0.75)

    ax.set_xlabel("training step")
    ax.set_ylabel(f"{args.metric} [m]")
    ax.set_title(f"RL corrector {args.metric} vs training progress\n"
                 f"(dotted = identity baseline, dashed = TVLQR, same colour)")
    # Errors span orders of magnitude between a held straight and a diverging
    # corner; a linear axis makes everything but the worst trajectory unreadable.
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(args.out, f"checkpoints_{args.metric}.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
