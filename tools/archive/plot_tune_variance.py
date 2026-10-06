#!/usr/bin/env python3
"""Why the 2026-08-02 TVLQR tuning run cannot be read as a tuning result.

`plot_tune_landscape.py` draws where the simplex searched and reports the best
value it saw. That figure is honest about the SEARCH but silently misleading
about the RESULT, because it plots one sample per gain pair and Nelder-Mead
reports the minimum -- and on this eval set the objective turned out to be noisy
enough that the minimum is mostly a lucky draw.

This figure makes the noise the subject instead:

  left    every evaluation in search order, with the evaluations that landed on
          the collapsed simplex point drawn separately. The simplex stopped
          moving at eval 49 and then re-sampled ONE gain pair 61 times, so
          everything after that vertical line is a repeat measurement, not a
          search.
  right   the distribution of those repeats at IDENTICAL gains, against the
          reported "best" and the starting baseline. If the spread brackets the
          claimed improvement, the run resolved noise rather than signal.

The point is not that the tuner is broken -- it is deterministic and internally
consistent. The point is that "noise floor 0.0002 m" was established on ONE
trajectory (floor_6_00042) that has since been DROPPED from the eval set, and it
does not transfer to the seven-shape set that replaced it.

OFFLINE TOOL -- matplotlib in a venv, never a ROS dependency.

    .venv/bin/python tools/plot_tune_variance.py tune_data/tvlqr_tune.jsonl --out figures_new
"""

import argparse
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Roles, not series identities: this figure has no competing categories, so it
# uses one hue for measurements plus reserved ink for the reference lines.
C_SEARCH = "#2a78d6"    # evaluations while the simplex was still moving
C_REPEAT = "#eb6834"    # evaluations after it collapsed -- repeats, not search
C_BEST = "#008300"      # the value the tuner reported
C_BASE = "#52514e"      # the starting gains it was trying to beat


def load(path):
    recs = []
    for line in open(path):
        d = json.loads(line)
        if "_meta" in d or d.get("failed"):
            continue
        recs.append(d)
    recs.sort(key=lambda r: r["eval_index"])
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="tvlqr_tune.jsonl")
    ap.add_argument("--out", default="figures", help="output directory")
    args = ap.parse_args()

    recs = load(args.src)

    # Find the gain pair the simplex collapsed onto: the one evaluated most.
    groups = defaultdict(list)
    for r in recs:
        groups[(round(r["q_cross"], 2), round(r["r_omega"], 3))].append(r)
    collapsed, rep = max(groups.items(), key=lambda kv: len(kv[1]))
    rep_vals = [r["fx"] for r in rep]
    first_rep = min(r["eval_index"] for r in rep)

    baseline = recs[0]["fx"]
    best = min(r["fx"] for r in recs)

    fig, (ax, axd) = plt.subplots(
        1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [2.5, 1.0]}
    )

    # --- left: evaluations in search order -------------------------------
    search = [r for r in recs if r["eval_index"] < first_rep]
    ax.scatter([r["eval_index"] for r in search], [r["fx"] for r in search],
               s=34, color=C_SEARCH, zorder=3, label="simplex still moving")
    ax.scatter([r["eval_index"] for r in rep], rep_vals,
               s=34, color=C_REPEAT, zorder=3,
               label=f"repeats at ONE gain pair (n={len(rep)})")
    ax.axvline(first_rep, color="#8a8985", lw=1.2, ls="--", zorder=2)
    ax.text(first_rep + 2, ax.get_ylim()[1], " simplex collapsed", fontsize=8.5,
            color="#52514e", va="top")
    ax.axhline(baseline, color=C_BASE, lw=1.6, ls=":", zorder=2,
               label=f"starting gains ({baseline:.3f} m, 1 sample)")
    ax.axhline(best, color=C_BEST, lw=1.6, zorder=2,
               label=f'reported "best" ({best:.3f} m)')
    ax.set_xlabel("evaluation index")
    ax.set_ylabel("mean max|cross-track error|  [m]")
    ax.set_title("Every evaluation of the overnight tuning run, in order")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- right: distribution of the repeats -------------------------------
    # A histogram, not a box plot: the question is whether the reported minimum
    # sits inside the bulk of the SAME measurement repeated, and a box plot hides
    # exactly that by summarising it into quartiles.
    axd.hist(rep_vals, bins=14, color=C_REPEAT, alpha=0.85, zorder=3,
             orientation="horizontal")
    axd.axhline(best, color=C_BEST, lw=1.6, zorder=4)
    axd.axhline(baseline, color=C_BASE, lw=1.6, ls=":", zorder=4)
    axd.axhline(st.mean(rep_vals), color="#0b0b0b", lw=1.4, ls="--", zorder=4)
    axd.text(axd.get_xlim()[1], st.mean(rep_vals),
             f"  mean {st.mean(rep_vals):.3f}", fontsize=8.5, va="center",
             color="#0b0b0b")
    axd.set_xlabel("count")
    axd.set_title(f"Same gains, {len(rep)} times\n"
                  f"q={collapsed[0]:.2f}  r={collapsed[1]:.3f}", fontsize=10)
    axd.grid(axis="x", alpha=0.25, zorder=0)
    axd.set_axisbelow(True)
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)

    sd = st.pstdev(rep_vals)
    fig.suptitle(
        f"The tuning objective is noisy on the 7-shape set: sd = {sd:.3f} m at FIXED gains, "
        f"spread {max(rep_vals) - min(rep_vals):.3f} m "
        f"-- larger than the {baseline - best:.3f} m 'improvement' claimed",
        fontsize=11, y=1.0,
    )

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "tvlqr_tune_variance.png")
    fig.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    print(f"wrote {dest}")
    print(f"  collapsed at eval {first_rep}, q={collapsed[0]} r={collapsed[1]}")
    print(f"  repeats n={len(rep)}  min={min(rep_vals):.4f} max={max(rep_vals):.4f} "
          f"mean={st.mean(rep_vals):.4f} sd={sd:.4f}")
    print(f"  baseline (1 sample) = {baseline:.4f}   reported best = {best:.4f}")


if __name__ == "__main__":
    main()
