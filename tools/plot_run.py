#!/usr/bin/env python3
"""Plot fixture runs recorded by agx_planning/run_recorder.py.

Turns the CSVs `just fetch-runs` pulls into run_data/ into two figures:

  <name>_path.png   the planned path and each run's true path, in map frame
  <name>_cross.png  deviation from the planned path over time

Pass one run to look at it, or several to compare them -- comparing is the point,
since the fixture exists so that two correctors can be run against an identical
map and trajectory.

OFFLINE TOOL. matplotlib is deliberately not a dependency of any ROS package
here (same rule as trimesh for the map baker): nothing on the robot plots
anything. Run it from a venv:

    python3 -m venv .venv && .venv/bin/pip install matplotlib
    .venv/bin/python tools/plot_run.py run_data/identity_clean run_data/tvlqr_clean

USAGE
  tools/plot_run.py run_data/identity_clean
  tools/plot_run.py run_data/identity_clean run_data/tvlqr_clean \\
      --labels "Identity (open loop)" "TVLQR" --out figures

NaN IN THE CSVs IS EXPECTED, AND IS A TRAP
------------------------------------------
cross_track/plan_x/plan_y are 'nan' for every sample recorded before the planner
published its path -- with wait_for_complete the robot stands still through the
whole planning phase, so that is typically the first 20+ seconds of a run.

The trap: `float("nan")` parses happily, so a reader that only guards against
ValueError silently admits NaN into its statistics, and one NaN turns any max()
or mean() into NaN. (An HTML version of these charts rendered completely empty
for exactly this reason.) This script masks non-finite values out explicitly --
do the same in anything else that consumes these files.
"""

import argparse
import csv
import math
import os
import sys

# The categorical order is fixed, not cycled: run N always gets colour N, so a
# series keeps its colour when another run is added to the comparison.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
REF_COLOR = "#8a8985"


def read_track(prefix):
    """(t, x, y, cross) lists from <prefix>_track.csv, NaNs preserved."""
    path = prefix + "_track.csv"
    t, x, y, cross = [], [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["t"]))
            x.append(float(row["true_x"]))
            y.append(float(row["true_y"]))
            cross.append(float(row["cross_track"]))
    return t, x, y, cross


def read_plan(prefix):
    path = prefix + "_plan.csv"
    if not os.path.isfile(path):
        return [], []
    xs, ys = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["plan_x"]))
            ys.append(float(row["plan_y"]))
    return xs, ys


def read_summary(prefix):
    path = prefix + "_summary.txt"
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for line in fh:
            if ":" in line:
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
    return out


def finite_pairs(a, b):
    """Drop index pairs where either value is non-finite. See the NaN note."""
    fa, fb = [], []
    for u, v in zip(a, b):
        if math.isfinite(u) and math.isfinite(v):
            fa.append(u)
            fb.append(v)
    return fa, fb


def plot_paths(runs, labels, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    # One planned path is drawn: with the same goal and seed every run shares it,
    # and overlaying identical curves just thickens the line.
    px, py = read_plan(runs[0])
    if px:
        ax.plot(px, py, color=REF_COLOR, lw=3.0, solid_capstyle="round",
                label="Planned", zorder=1)

    for i, (prefix, label) in enumerate(zip(runs, labels)):
        _, x, y, _ = read_track(prefix)
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(x, y, color=color, lw=2.0, solid_capstyle="round",
                label=label, zorder=2 + i)
        # End point, ringed in the surface colour so overlapping ends stay legible.
        ax.plot(x[-1], y[-1], "o", ms=9, color=color,
                mec="white", mew=2.0, zorder=10)

    summary = read_summary(runs[0])
    goal = summary.get("goal")
    if goal and goal != "None":
        gx, gy = (float(v) for v in goal.strip("()").split(","))
        ax.plot(gx, gy, "o", ms=11, color=REF_COLOR, mec="white", mew=2.0,
                zorder=9)
        ax.annotate("goal", (gx, gy), textcoords="offset points",
                    xytext=(12, -3), color=REF_COLOR, fontsize=10)

    ax.set_aspect("equal")          # a distorted map misrepresents the path
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Path through the building", loc="left", fontsize=12,
                 fontweight="bold")
    ax.grid(True, color="#e3e3de", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"wrote {out_path}")
    plt.close(fig)


def plot_cross(runs, labels, out_path, trim=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 3.8))

    for i, (prefix, label) in enumerate(zip(runs, labels)):
        t, x, y, cross = read_track(prefix)
        if trim:
            # Re-zero each run at the moment it STARTS MOVING, and drop what
            # came before. Two reasons this is not the first finite sample:
            # with wait_for_complete the robot stands still through the whole
            # planning phase, and the plan is published while it is still
            # parked -- so cross-track goes finite, but pinned at ~0 because a
            # stationary robot sits exactly on the path start. Trimming on
            # "plan exists" still left ~15 s of flat zero.
            # Runs also plan for different durations, so a shared raw clock
            # offsets the curves and makes them look different where they
            # are not.
            t = list(t)
            start = 0
            for k in range(len(t)):
                if math.hypot(x[k] - x[0], y[k] - y[0]) > 0.02:
                    start = k
                    break
            t0 = t[start]
            t, cross = [v - t0 for v in t[start:]], cross[start:]
        ft, fc = finite_pairs(t, cross)
        if not ft:
            print(f"warning: {prefix} has no finite cross-track samples; "
                  f"was a plan ever published?", file=sys.stderr)
            continue
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        summary = read_summary(prefix)
        rms = summary.get("cross-track rms", "")
        text = f"{label}  (rms {rms})" if rms else label
        ax.plot(ft, fc, color=color, lw=2.0, solid_capstyle="round", label=text)

    ax.set_xlabel("time since playback start (s)" if trim else "time (s)")
    ax.set_ylabel("deviation (m)")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Deviation from the planned path", loc="left", fontsize=12,
                 fontweight="bold")
    ax.grid(True, color="#e3e3de", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"wrote {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+",
                    help="run prefixes, e.g. run_data/identity_clean "
                         "(without the _track.csv suffix)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="legend labels, one per run (default: the basenames)")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep the standing-still planning phase on the time "
                         "axis instead of zeroing each run at playback start")
    ap.add_argument("--name", default=None,
                    help="output basename (default: joined run basenames)")
    args = ap.parse_args()

    for prefix in args.runs:
        if not os.path.isfile(prefix + "_track.csv"):
            ap.error(f"{prefix}_track.csv not found -- pass the prefix without "
                     f"_track.csv, and run `just fetch-runs` first")

    labels = args.labels or [os.path.basename(r) for r in args.runs]
    if len(labels) != len(args.runs):
        ap.error(f"got {len(labels)} labels for {len(args.runs)} runs")

    os.makedirs(args.out, exist_ok=True)
    name = args.name or "_vs_".join(os.path.basename(r) for r in args.runs)

    plot_paths(args.runs, labels, os.path.join(args.out, f"{name}_path.png"))
    plot_cross(args.runs, labels, os.path.join(args.out, f"{name}_cross.png"),
               trim=not args.no_trim)

    print("\nsummaries:")
    for prefix, label in zip(args.runs, labels):
        s = read_summary(prefix)
        if s:
            print(f"  {label}: final {s.get('final error', '?')}, "
                  f"rms {s.get('cross-track rms', '?')}, "
                  f"max {s.get('cross-track max', '?')}")


if __name__ == "__main__":
    main()
