#!/usr/bin/env python3
"""Render the 2026-08-14 figure set. See figures/2026-08-14/README.md.

Offline only: matplotlib + numpy from the repo venv, like every other
`tools/plot_*.py`. Reads the gitignored `soak_data/` and writes PNGs into
`figures/2026-08-14/`; nothing here imports the control stack.

    .venv/bin/python figures/2026-08-14/render.py

Plant for every panel: 2026-08-07 wheel `mu2=0.45`, world ground `mu=1.0`.
"""

import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
LADDER = "soak_data/soak_r_ladder.jsonl"

SHAPES = [("floor_1_00049", "STRAIGHT"), ("floor_6_00023", "CORNER"),
          ("floor_6_00018", "S"), ("floor_6_00047", "ZIGZAG"),
          ("floor_6_00056", "TIGHT V"), ("floor_6_00031", "U-TURN"),
          ("floor_6_00025", "LOOP")]
# Read off the scatter, not tuned -- the clusters are separated by a wide gap.
BAD = {"floor_6_00031": 2.0, "floor_6_00018": 2.0}


def rows():
    out = []
    with open(LADDER) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("max_cross") is not None:
                out.append(r)
    return out


def cells(rs):
    by = collections.defaultdict(list)
    for r in rs:
        by[(r["trajectory"], r["r_omega"])].append(r["max_cross"])
    return by


def fig_ladder():
    """Figure 1 -- max|e_cross| against r_omega, one panel per shape.

    The point of the panel grid rather than one aggregate line: the aggregate
    says r=2.618 wins, and the panels say five of seven shapes cannot tell the
    rungs apart at all. Only a per-shape view distinguishes "r is a good gain"
    from "one plan has a notch at r=2.618".
    """
    rs = rows()
    by = cells(rs)
    r_vals = sorted({r["r_omega"] for r in rs})
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8))
    for ax, (name, lab) in zip(axes.ravel(), SHAPES):
        for i, rv in enumerate(r_vals):
            v = np.array(by[(name, rv)])
            jitter = (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.25
            ax.scatter(np.full(len(v), i) + jitter, v, s=7, alpha=0.45,
                       color="#1f77b4", edgecolors="none")
            ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color="#d62728", lw=2)
        thr = BAD.get(name)
        if thr:
            ax.axhline(thr, color="#999", ls=":", lw=1)
        ax.set_xticks(range(len(r_vals)))
        ax.set_xticklabels([f"{v:g}" for v in r_vals], fontsize=8)
        ax.set_title(f"{lab}\n{name}", fontsize=9)
        ax.set_xlabel("r_omega", fontsize=8)
        ax.tick_params(labelsize=8)
    ax = axes.ravel()[-1]
    means = [np.mean([np.mean(by[(n, rv)]) for n, _ in SHAPES]) for rv in r_vals]
    no_u = [np.mean([np.mean(by[(n, rv)]) for n, _ in SHAPES
                     if n != "floor_6_00031"]) for rv in r_vals]
    ax.plot(range(len(r_vals)), means, "o-", color="#d62728", label="all 7")
    ax.plot(range(len(r_vals)), no_u, "s--", color="#555", label="without U-turn")
    ax.set_xticks(range(len(r_vals)))
    ax.set_xticklabels([f"{v:g}" for v in r_vals], fontsize=8)
    ax.set_title("aggregate\n(the win is one plan)", fontsize=9)
    ax.set_xlabel("r_omega", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    fig.suptitle("r_omega ladder, q_cross=0.276 fixed -- 1035 rollouts, n=30 per cell "
                 "(plant: wheel mu2=0.45, ground mu=1.0)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "01_r_ladder.png"), dpi=130)
    plt.close(fig)


def fig_modes():
    """Figure 2 -- the bad-mode rate, which is what the ladder actually moves.

    Same story the q ladder told, mirrored: the rungs differ in the FREQUENCY of
    a reproducible bad outcome, not in a level. Here it is also nearly all one
    shape.
    """
    rs = rows()
    by = cells(rs)
    r_vals = sorted({r["r_omega"] for r in rs})
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, thr, color in [("floor_6_00031", 2.0, "#d62728"),
                             ("floor_6_00018", 2.0, "#1f77b4")]:
        pct = [100 * np.mean(np.array(by[(name, rv)]) > thr) for rv in r_vals]
        lab = dict(SHAPES)[name]
        ax.plot(range(len(r_vals)), pct, "o-", color=color,
                label=f"{lab} ({name}), bad = >{thr:g} m")
    ax.set_xticks(range(len(r_vals)))
    ax.set_xticklabels([f"{v:g}" for v in r_vals])
    ax.set_xlabel("r_omega  (q_cross = 0.276 throughout)")
    ax.set_ylabel("% of rollouts in the bad mode")
    ax.set_ylim(-4, 104)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("r_omega=2.618 is an isolated notch, not a plateau\n"
                 "the other five shapes are flat and are not plotted", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_r_modes.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_ladder()
    fig_modes()
    print("wrote", OUT)
