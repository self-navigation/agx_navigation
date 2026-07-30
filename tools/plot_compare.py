#!/usr/bin/env python3
"""Overlay the paths three correctors drove on the SAME planned trajectory.

Consumes what agx_planning.rl_corrector.compare_correctors writes:

    <dir>/<trajectory>__<corrector>.csv     per-step plan + true pose + errors
    <dir>/summary.csv                       one row per (trajectory, corrector)

and renders, per trajectory, one figure with two panels:

    left    the planned path and each corrector's TRUE path, drawn on top of one
            another in the map frame, equal-aspect so a corner looks like a
            corner (a stretched aspect ratio once made a 0.16 rad/step curve read
            as a 90-degree hairpin -- see the rl-corrector session notes).
    right   signed cross-track error over the trajectory, same colours, with the
            corridor width marked.

OFFLINE TOOL, same rule as plot_run.py and the map baker: matplotlib is
deliberately not a dependency of any ROS package here.

    python3 -m venv .venv && .venv/bin/pip install matplotlib
    .venv/bin/python tools/plot_compare.py /tmp/compare --out figures
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed per-corrector colours: a corrector keeps its colour across every figure,
# so two trajectories can be compared at a glance without re-reading the legend.
COLORS = {"identity": "#eb6834", "tvlqr": "#2a78d6", "rl": "#1baf7a"}
LABELS = {"identity": "identity (open loop)", "tvlqr": "TVLQR", "rl": "RL residual"}
PLAN_COLOR = "#8a8985"


def read_run(path):
    cols = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    f = float("nan")
                cols[k].append(f)
    return cols


def finite(xs, ys):
    """Drop non-finite pairs. NaN parses silently and poisons any max()/mean();
    the same trap plot_run.py documents."""
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if math.isfinite(x) and math.isfinite(y):
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def read_summary(d):
    path = os.path.join(d, "summary.csv")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[(row["trajectory"], row["corrector"])] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("compare_dir")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--corridor", type=float, default=0.5,
                    help="corridor half-width to mark on the error panel [m]")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = read_summary(args.compare_dir)

    runs = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(args.compare_dir, "*__*.csv"))):
        base = os.path.basename(p)[:-4]
        traj, corrector = base.split("__", 1)
        runs[traj][corrector] = read_run(p)

    if not runs:
        raise SystemExit(f"no <traj>__<corrector>.csv files in {args.compare_dir}")

    order = ["identity", "tvlqr", "rl"]
    for traj, legs in sorted(runs.items()):
        fig, (ax_p, ax_e) = plt.subplots(
            1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})

        any_leg = next(iter(legs.values()))
        px, py = finite(any_leg["plan_x"], any_leg["plan_y"])
        ax_p.plot(px, py, "--", color=PLAN_COLOR, lw=2.2, label="planned (PMP)", zorder=1)
        if px:
            ax_p.plot(px[0], py[0], "o", color=PLAN_COLOR, ms=9, zorder=4)
            ax_p.plot(px[-1], py[-1], "*", color=PLAN_COLOR, ms=17, zorder=4)

        for c in order:
            if c not in legs:
                continue
            col = legs[c]
            tx, ty = finite(col["true_x"], col["true_y"])
            s = summary.get((traj, c))
            lab = LABELS[c]
            if s:
                lab += (f"  (max {float(s['max_cross']):.2f} m, "
                        f"end {float(s['final_err']):.2f} m)")
            ax_p.plot(tx, ty, "-", color=COLORS[c], lw=2.0, label=lab, zorder=3)
            if tx:
                ax_p.plot(tx[-1], ty[-1], "o", color=COLORS[c], ms=7, zorder=5)

            ks = col["k"]
            ec = col["e_cross"]
            kk, ee = finite(ks, ec)
            ax_e.plot(kk, ee, "-", color=COLORS[c], lw=1.8, label=LABELS[c])

        ax_p.set_title(f"{traj} — path driven by each corrector")
        ax_p.set_xlabel("x [m]")
        ax_p.set_ylabel("y [m]")
        # Equal aspect is not cosmetic: it is what makes a corner readable.
        ax_p.set_aspect("equal", adjustable="datalim")
        ax_p.grid(alpha=0.3)
        ax_p.legend(fontsize=8, loc="best")

        ax_e.axhline(0, color=PLAN_COLOR, lw=1.2, ls="--")
        for sign in (1, -1):
            ax_e.axhline(sign * args.corridor, color="#c0392b", lw=1.0, ls=":",
                         label="corridor" if sign == 1 else None)
        ax_e.set_title("signed cross-track error")
        ax_e.set_xlabel("step")
        ax_e.set_ylabel("e_cross [m]")
        ax_e.grid(alpha=0.3)
        ax_e.legend(fontsize=8, loc="best")

        fig.tight_layout()
        out = os.path.join(args.out, f"{traj}_compare.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
