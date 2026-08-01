#!/usr/bin/env python3
"""Overlay the paths successive RL CHECKPOINTS drove on the same frozen plan.

Companion to plot_checkpoints.py, which reduces each checkpoint to one scalar
against training step. That answers "is it improving"; this answers "improving
*how*" -- whether a shrinking max|e_cross| is the policy learning to hold the
line, or the same wrong turn taken slightly less far.

Consumes a compare_correctors sweep laid out one directory per checkpoint:

    <sweep>/step_000005000/<traj>__<corrector>.csv
    <sweep>/step_000100000/<traj>__rl.csv
    ...

Checkpoints are drawn in a dark->bright colour ramp (early->late), with the
identity and TVLQR baselines kept in their usual fixed colours so a figure from
this tool and one from plot_compare.py can be read side by side. Baselines are
checkpoint-independent, so the first directory that carries them wins -- the
sweep only needs to measure them once.

Only `--max-lines` checkpoints are drawn (evenly spaced, first and last always
kept): past ~8 overlaid paths the individual lines stop being traceable, which
defeats the purpose of a path plot.

OFFLINE TOOL (matplotlib in a venv), same rule as plot_run.py.

    .venv/bin/python tools/plot_checkpoint_paths.py sweep_data --out figures
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_COLORS = {"identity": "#eb6834", "tvlqr": "#2a78d6"}
BASELINE_LABELS = {"identity": "identity (open loop)", "tvlqr": "TVLQR"}
PLAN_COLOR = "#8a8985"


def read_run(path):
    cols = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                try:
                    cols[k].append(float(v))
                except (TypeError, ValueError):
                    cols[k].append(float("nan"))
    return cols


def finite(xs, ys):
    """Drop non-finite pairs -- NaN parses silently and poisons any max()/mean()."""
    out = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    return [p[0] for p in out], [p[1] for p in out]


def read_summary(d):
    path = os.path.join(d, "summary.csv")
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return {(r["trajectory"], r["corrector"]): r for r in csv.DictReader(fh)}


def pick(steps, n):
    """Evenly spaced subset of `steps`, always keeping the first and last."""
    if len(steps) <= n:
        return steps
    idx = sorted({round(i * (len(steps) - 1) / (n - 1)) for i in range(n)})
    return [steps[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--max-lines", type=int, default=7,
                    help="how many checkpoints to draw (evenly spaced)")
    ap.add_argument("--corridor", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # traj -> step -> cols ; traj -> corrector -> (cols, summary_row)
    rl = defaultdict(dict)
    baselines = defaultdict(dict)
    summaries = {}
    for d in sorted(glob.glob(os.path.join(args.sweep_dir, "step_*"))):
        m = re.search(r"(\d+)", os.path.basename(d))
        if not m:
            continue
        step = int(m.group(1))
        summ = read_summary(d)
        for p in sorted(glob.glob(os.path.join(d, "*__*.csv"))):
            traj, corr = os.path.basename(p)[:-4].split("__", 1)
            cols = read_run(p)
            if corr == "rl":
                rl[traj][step] = cols
                summaries[(traj, "rl", step)] = summ.get((traj, "rl"))
            else:
                if corr not in baselines[traj]:
                    baselines[traj][corr] = (cols, summ.get((traj, corr)))

    if not rl:
        raise SystemExit(f"no step_*/<traj>__rl.csv under {args.sweep_dir}")

    cmap = plt.get_cmap("viridis")
    for traj in sorted(rl):
        steps = pick(sorted(rl[traj]), args.max_lines)
        fig, (ax_p, ax_e) = plt.subplots(
            1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})

        any_cols = rl[traj][steps[0]]
        px, py = finite(any_cols["plan_x"], any_cols["plan_y"])
        ax_p.plot(px, py, "--", color=PLAN_COLOR, lw=2.2, label="planned (PMP)", zorder=2)
        if px:
            ax_p.plot(px[0], py[0], "o", color=PLAN_COLOR, ms=9, zorder=6)
            ax_p.plot(px[-1], py[-1], "*", color=PLAN_COLOR, ms=17, zorder=6)

        for corr in ("identity", "tvlqr"):
            if corr not in baselines[traj]:
                continue
            cols, s = baselines[traj][corr]
            lab = BASELINE_LABELS[corr]
            if s:
                lab += f"  (max {float(s['max_cross']):.2f} m)"
            tx, ty = finite(cols["true_x"], cols["true_y"])
            ax_p.plot(tx, ty, color=BASELINE_COLORS[corr], lw=2.4, ls=(0, (6, 2)),
                      label=lab, zorder=3, alpha=0.9)
            kk, ee = finite(cols["k"], cols["e_cross"])
            ax_e.plot(kk, ee, color=BASELINE_COLORS[corr], lw=2.0, ls=(0, (6, 2)),
                      label=BASELINE_LABELS[corr])

        for i, step in enumerate(steps):
            frac = i / max(len(steps) - 1, 1)
            col = cmap(0.08 + 0.85 * frac)
            cols = rl[traj][step]
            s = summaries.get((traj, "rl", step))
            lab = f"RL {step/1000:.0f}k"
            if s:
                lab += f"  (max {float(s['max_cross']):.2f} m)"
            tx, ty = finite(cols["true_x"], cols["true_y"])
            ax_p.plot(tx, ty, "-", color=col, lw=1.8, label=lab, zorder=4)
            if tx:
                ax_p.plot(tx[-1], ty[-1], "o", color=col, ms=6, zorder=5)
            kk, ee = finite(cols["k"], cols["e_cross"])
            ax_e.plot(kk, ee, "-", color=col, lw=1.5, label=lab.split("  ")[0])

        ax_p.set_title(f"{traj} — RL checkpoints over training (dark → bright = early → late)")
        ax_p.set_xlabel("x [m]")
        ax_p.set_ylabel("y [m]")
        # Equal aspect is not cosmetic: it is what makes a corner readable.
        ax_p.set_aspect("equal", adjustable="datalim")
        ax_p.grid(alpha=0.3)
        ax_p.legend(fontsize=7, loc="best")

        ax_e.axhline(0, color=PLAN_COLOR, lw=1.2, ls="--")
        for sign in (1, -1):
            ax_e.axhline(sign * args.corridor, color="#c0392b", lw=1.0, ls=":",
                         label="corridor" if sign == 1 else None)
        ax_e.set_title("signed cross-track error")
        ax_e.set_xlabel("step")
        ax_e.set_ylabel("e_cross [m]")
        ax_e.grid(alpha=0.3)
        ax_e.legend(fontsize=7, loc="best", ncol=2)

        fig.tight_layout()
        out = os.path.join(args.out, f"{traj}_checkpoints.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
