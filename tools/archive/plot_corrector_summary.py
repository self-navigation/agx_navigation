#!/usr/bin/env python3
"""One figure summarising a whole corrector comparison across every shape.

`plot_compare.py` draws one figure PER trajectory, which is the right form when
the question is "what did this corrector do wrong on this plan". It is the wrong
form for "which corrector should we ship": seven figures cannot be read as a
ranking, and the eye cannot average them.

This renders the same `summary.csv` as a grouped bar chart -- shape on the x
axis, one bar per corrector -- plus a mean panel on the right. Bars because the
job is MAGNITUDE COMPARISON at identical x positions; the shapes are unordered
categories, so a line joining them would imply a progression that does not exist.

The y axis is max|e_cross| only. `final_err` is deliberately NOT plotted here:
it is not reproducible run-to-run (sd 0.26 m against max|e_cross|'s 0.0002 m on
the old single-trajectory probe -- see CLAUDE.md), so putting it beside a
reproducible metric in the same figure invites reading noise as signal.

OFFLINE TOOL, same rule as plot_compare.py -- matplotlib in a venv, never a
dependency of a ROS package.

    .venv/bin/python tools/plot_corrector_summary.py compare_data_new --out figures_new
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same fixed per-corrector colours as plot_compare.py: a corrector keeps its
# colour across every figure in the report, so the grouped bars and the path
# overlays can be read side by side without re-learning the legend.
COLORS = {"identity": "#eb6834", "tvlqr": "#2a78d6", "rl": "#1baf7a"}
LABELS = {
    "identity": "identity (open loop)",
    "tvlqr": "TVLQR",
    "rl": "RL residual (800k)",
}
ORDER = ["identity", "tvlqr", "rl"]

# Shape labels come from config/eval_trajectories.yaml. Hard-coded rather than
# parsed because the mapping is the POINT of the eval set -- if a name here stops
# matching the config, the figure should be regenerated deliberately, not
# silently relabelled.
SHAPES = {
    "floor_1_00049": "STRAIGHT",
    "floor_6_00023": "CORNER",
    "floor_6_00018": "S-CURVE",
    "floor_6_00047": "ZIGZAG",
    "floor_6_00056": "TIGHT V",
    "floor_6_00031": "U-TURN",
    "floor_6_00025": "LOOP",
}
# Plot order: easiest shape first, so the left-to-right read is "difficulty".
SHAPE_ORDER = [
    "floor_1_00049",
    "floor_6_00023",
    "floor_6_00018",
    "floor_6_00047",
    "floor_6_00056",
    "floor_6_00031",
    "floor_6_00025",
]


def read_summary(path):
    """summary.csv -> {trajectory: {corrector: max_cross}}."""
    out = defaultdict(dict)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[row["trajectory"]][row["corrector"]] = float(row["max_cross"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="directory holding summary.csv")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--title", default=None, help="override the figure title")
    args = ap.parse_args()

    data = read_summary(os.path.join(args.src, "summary.csv"))
    trajs = [t for t in SHAPE_ORDER if t in data]
    # Anything the eval set gained since this tool was written still gets drawn,
    # appended after the known shapes rather than dropped on the floor.
    trajs += sorted(t for t in data if t not in SHAPE_ORDER)
    correctors = [c for c in ORDER if any(c in data[t] for t in trajs)]

    fig, (ax, ax_mean) = plt.subplots(
        1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [3.4, 1.0]}
    )

    n = len(correctors)
    width = 0.78 / n
    for i, corr in enumerate(correctors):
        xs, ys = [], []
        for j, t in enumerate(trajs):
            if corr in data[t]:
                # 2px-equivalent surface gap between adjacent bars: shrink the
                # bar slightly rather than butting the fills together.
                xs.append(j + (i - (n - 1) / 2) * width)
                ys.append(data[t][corr])
        bars = ax.bar(xs, ys, width * 0.92, label=LABELS.get(corr, corr),
                      color=COLORS.get(corr), zorder=3)
        # Direct value labels: 21 bars is few enough that every one can carry its
        # number without collision, and it removes a lookup against the y axis.
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#52514e")

    ax.set_xticks(range(len(trajs)))
    ax.set_xticklabels([f"{SHAPES.get(t, '?')}\n{t}" for t in trajs], fontsize=8.5)
    ax.set_ylabel("max |cross-track error|  [m]")
    ax.set_title(args.title or
                 "Worst-case path deviation by trajectory shape  (lower is better)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Mean panel. Same colours, same units, same y meaning -- it is the same
    # measure aggregated, NOT a second scale, so it is legitimately its own axes
    # rather than a twin axis on the left plot.
    means = [sum(data[t][c] for t in trajs if c in data[t])
             / sum(1 for t in trajs if c in data[t]) for c in correctors]
    bars = ax_mean.bar(range(len(correctors)), means, 0.6,
                       color=[COLORS.get(c) for c in correctors], zorder=3)
    for b, v in zip(bars, means):
        ax_mean.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=10, color="#0b0b0b")
    ax_mean.set_xticks(range(len(correctors)))
    ax_mean.set_xticklabels([LABELS.get(c, c).split(" (")[0] for c in correctors],
                            fontsize=9)
    ax_mean.set_ylabel("mean over all shapes  [m]")
    ax_mean.set_title("Mean across the eval set")
    ax_mean.grid(axis="y", alpha=0.25, zorder=0)
    ax_mean.set_axisbelow(True)
    for s in ("top", "right"):
        ax_mean.spines[s].set_visible(False)

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "corrector_summary.png")
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    print(f"wrote {dest}")
    for c, m in zip(correctors, means):
        print(f"  {c:10s} mean max|e_cross| = {m:.3f} m")


if __name__ == "__main__":
    main()
