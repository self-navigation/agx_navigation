#!/usr/bin/env python3
"""RL tracking error vs training step, re-measured on the FIXED simulator.

The 2026-08-01 checkpoint sweep is unusable: it ran against a world that
free-ran between control steps and against friction patches that silently failed
to spawn roughly half the time, so its per-checkpoint numbers pool two different
plants and depend on CPU load. That sweep is what "no checkpoint beats identity"
was based on.

This redraws the same question against the repaired bridge. Only the `rl` leg is
re-measured: identity and TVLQR are checkpoint-INDEPENDENT, so re-running them at
every checkpoint would spend two-thirds of the sweep re-deriving two constants.
They are drawn as horizontal reference lines from the single clean comparison
instead, which is both cheaper and less noisy.

    left    mean over the eval set vs training step, against both baselines
    right   per-trajectory, so a checkpoint that wins on average by ruining one
            shape is visible rather than hidden in the mean

OFFLINE TOOL -- matplotlib in the venv, never a ROS dependency.

    .venv/bin/python tools/plot_checkpoints_clean.py sweep_clean \\
        --baselines compare_data_new/summary.csv --out figures_new
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

C_RL = "#1baf7a"
C_IDENTITY = "#eb6834"
C_TVLQR = "#2a78d6"
# Per-trajectory hues for the right panel, in the fixed categorical order. Seven
# shapes fits inside the eight-slot palette, so no hue is ever generated.
TRAJ_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7"]

SHAPES = {
    "floor_1_00049": "STRAIGHT",
    "floor_6_00023": "CORNER",
    "floor_6_00018": "S-CURVE",
    "floor_6_00047": "ZIGZAG",
    "floor_6_00056": "TIGHT V",
    "floor_6_00031": "U-TURN",
    "floor_6_00025": "LOOP",
}


def read_sweep(root):
    """{step: {trajectory: max_cross}} from sweep_*/step_*/summary.csv."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "step_*"))):
        m = re.search(r"step_(\d+)", os.path.basename(d))
        if not m:
            continue
        path = os.path.join(d, "summary.csv")
        if not os.path.exists(path):
            continue
        per = {}
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row["corrector"] == "rl":
                    per[row["trajectory"]] = float(row["max_cross"])
        if per:
            out[int(m.group(1))] = per
    return out


def read_baselines(path):
    """{corrector: mean max_cross over the eval set}."""
    vals = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            vals[row["corrector"]].append(float(row["max_cross"]))
    return {c: sum(v) / len(v) for c, v in vals.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="sweep directory holding step_*/summary.csv")
    ap.add_argument("--baselines", help="summary.csv with the identity/tvlqr legs")
    ap.add_argument("--out", default="figures", help="output directory")
    args = ap.parse_args()

    sweep = read_sweep(args.src)
    if not sweep:
        raise SystemExit(f"no step_*/summary.csv under {args.src}")
    steps = sorted(sweep)
    trajs = sorted({t for per in sweep.values() for t in per})

    # Mean only over checkpoints that measured the WHOLE set -- a checkpoint that
    # lost a rollout would otherwise get a mean over the survivors, which flatters
    # it, since the shapes differ hugely in difficulty.
    complete = [s for s in steps if len(sweep[s]) == len(trajs)]
    means = [sum(sweep[s].values()) / len(sweep[s]) for s in complete]

    fig, (ax, axp) = plt.subplots(1, 2, figsize=(15, 5.4))

    ax.plot([s / 1e6 for s in complete], means, color=C_RL, lw=2.2,
            marker="o", ms=5, zorder=4, label="RL residual")
    base = read_baselines(args.baselines) if args.baselines else {}
    for corr, color, label in (("identity", C_IDENTITY, "identity (open loop)"),
                               ("tvlqr", C_TVLQR, "TVLQR")):
        if corr in base:
            ax.axhline(base[corr], color=color, lw=1.8, ls="--", zorder=3,
                       label=f"{label} ({base[corr]:.2f} m)")
    if means:
        best_i = min(range(len(means)), key=lambda i: means[i])
        ax.annotate(f"best: {means[best_i]:.2f} m\n@ {complete[best_i]/1e6:.2f}M",
                    xy=(complete[best_i] / 1e6, means[best_i]),
                    xytext=(12, 18), textcoords="offset points", fontsize=9,
                    color="#0b0b0b",
                    arrowprops=dict(arrowstyle="->", color="#52514e", lw=1))
    ax.set_xlabel("training steps  [millions]")
    ax.set_ylabel("mean max |cross-track error|  [m]")
    ax.set_title("RL performance vs training step (re-measured, fixed simulator)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    for i, t in enumerate(trajs):
        xs = [s / 1e6 for s in steps if t in sweep[s]]
        ys = [sweep[s][t] for s in steps if t in sweep[s]]
        axp.plot(xs, ys, lw=1.7, color=TRAJ_COLORS[i % len(TRAJ_COLORS)],
                 marker="o", ms=3.5, label=SHAPES.get(t, t), zorder=3)
    axp.set_xlabel("training steps  [millions]")
    axp.set_ylabel("max |cross-track error|  [m]")
    axp.set_title("Per trajectory shape")
    axp.legend(frameon=False, fontsize=8, ncol=2)
    axp.grid(alpha=0.25, zorder=0)
    axp.set_axisbelow(True)
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "rl_checkpoints_clean.png")
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    print(f"wrote {dest}   ({len(complete)} complete checkpoints)")
    for s, m in zip(complete, means):
        print(f"  {s:>9d}  {m:.3f} m")


if __name__ == "__main__":
    main()
