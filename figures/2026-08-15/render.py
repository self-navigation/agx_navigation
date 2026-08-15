#!/usr/bin/env python3
"""Figures for 2026-08-15 — closing the r_omega and U-turn-generality questions.

Plant: 2026-08-07-wheel-mu2-045 (wheel.xacro mu2=0.45), world ground mu=1.0,
terrain patches on.

Inputs (all gitignored):
    soak_data/soak_r_ladder_low.jsonl        job 30, 840 rollouts
    soak_data/soak_uturn_generality.jsonl    job 40, 1200 rollouts
    traj_data_v2/*.npz                       constructed plan library (320)
    traj_data/floor_6_00031.npz              the original U-turn

Run from the repo root:  .venv/bin/python figures/2026-08-15/render.py
"""
import json, os, statistics as st, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SHAPES = {"floor_1_00049": "straight", "floor_6_00023": "corner",
          "floor_6_00018": "S", "floor_6_00047": "zigzag",
          "floor_6_00056": "tight V", "floor_6_00031": "U-turn",
          "floor_6_00025": "loop"}
ORDER = ["straight", "corner", "S", "zigzag", "tight V", "U-turn", "loop"]


def load(rel):
    """Rows with a metric. Failed rollouts carry no max_cross and are dropped."""
    out = []
    with open(os.path.join(ROOT, rel)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "max_cross" in d:
                out.append(d)
    return out


def fig_r_ladder_low():
    """r_omega below the previous ladder's floor: does the move off 0.25 pay?"""
    rows = load("soak_data/soak_r_ladder_low.jsonl")
    rs = sorted({r["r_omega"] for r in rows})
    cross = collections.defaultdict(list)
    final = collections.defaultdict(list)
    for r in rows:
        k = (SHAPES.get(r["trajectory"], r["trajectory"]), r["r_omega"])
        cross[k].append(r["max_cross"])
        final[k].append(r["final_err"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    x = np.arange(len(rs))

    ax = axes[0]
    for s in ORDER:
        m = [st.mean(cross[(s, r)]) for r in rs]
        e = [st.pstdev(cross[(s, r)]) for r in rs]
        ax.errorbar(x, m, yerr=e, marker="o", capsize=3,
                    lw=2.4 if s in ("zigzag", "U-turn") else 1.2,
                    alpha=1.0 if s in ("zigzag", "U-turn") else 0.55, label=s)
    ax.set_title("max|e_cross| per shape  (q_cross = 0.276)")
    ax.set_ylabel("max|e_cross|  [m]")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    for sub, lbl, style in ((ORDER, "all 7 shapes", "-o"),
                            ([s for s in ORDER if s != "U-turn"], "6 shapes, no U-turn", "-s")):
        m = [st.mean([st.mean(cross[(s, r)]) for s in sub]) for r in rs]
        ax.plot(x, m, style, lw=2.5, label=lbl)
    ax.set_title("aggregate — the U-turn is excluded to isolate\nwhat r buys everywhere else")
    ax.set_ylabel("mean max|e_cross|  [m]")
    ax.legend(fontsize=9)

    ax = axes[2]
    for s in ORDER:
        ax.plot(x, [st.mean(final[(s, r)]) for r in rs], marker="o", alpha=0.5, lw=1.2, label=s)
    ax.plot(x, [st.mean([st.mean(final[(s, r)]) for s in ORDER]) for r in rs],
            "k-o", lw=3, label="MEAN")
    ax.set_title("final_err — monotone in r, unlike max|e_cross|")
    ax.set_ylabel("final_err  [m]")
    ax.legend(fontsize=8, ncol=2)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in rs])
        ax.set_xlabel("r_omega")
        ax.grid(alpha=0.3)
    fig.suptitle("r_omega below the ladder floor — n≈29/cell, plant 2026-08-07-wheel-mu2-045",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "01_r_ladder_low.png"), dpi=120)


def fig_uturn_generality():
    """Does the q_cross basin reproduce on U-turns other than floor_6_00031?"""
    rows = load("soak_data/soak_uturn_generality.jsonl")
    qs = sorted({r["q_cross"] for r in rows})
    trajs = sorted({r["trajectory"] for r in rows})
    cross = collections.defaultdict(list)
    for r in rows:
        cross[(r["trajectory"], r["q_cross"])].append(r["max_cross"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(qs))
    for t in trajs:
        orig = t == "floor_6_00031"
        m = [st.mean(cross[(t, q)]) for q in qs]
        e = [st.pstdev(cross[(t, q)]) for q in qs]
        axes[0].errorbar(x, m, yerr=e, marker="o", capsize=3,
                         lw=3 if orig else 1.6, color="k" if orig else None,
                         label=t + (" (control)" if orig else ""))
        axes[1].plot(x, [100 * sum(v > 2.0 for v in cross[(t, q)]) / len(cross[(t, q)])
                         for q in qs], marker="o", lw=3 if orig else 1.6,
                     color="k" if orig else None, label=t)
    axes[0].set_ylabel("max|e_cross|  [m]")
    axes[0].set_title("mean ± sd, n≈59 per cell")
    axes[1].set_ylabel("% rollouts > 2.0 m")
    axes[1].set_title("only the control plan has a basin;\nthe others are flat, uniformly good, or uniformly bad")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(q) for q in qs])
        ax.set_xlabel("q_cross      (r_omega = 2.618)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("The q_cross basin is a property of ONE PLAN, not of the U-turn shape",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "02_uturn_generality.png"), dpi=120)


def fig_uturn_plans():
    """The five plans job 40 actually drove — the labels do not survive looking."""
    plans = [("traj_data/floor_6_00031.npz", "control: the original"),
             ("traj_data_v2/floor_6_v2_00008.npz", "NEAR-DUPLICATE of the control"),
             ("traj_data_v2/floor_6_v2_00010.npz", "the only true hairpin"),
             ("traj_data_v2/floor_6_v2_00004.npz", "rectangular U (two 90° corners)"),
             ("traj_data_v2/floor_6_v2_00003.npz", "NOT a U-turn — a bent line")]
    fig, axes = plt.subplots(1, len(plans), figsize=(3.5 * len(plans), 4.4))
    for ax, (rel, note) in zip(axes, plans):
        d = np.load(os.path.join(ROOT, rel))
        p = d["poses"]
        xs, ys, th = p[:, 0], p[:, 1], p[:, 2]
        ax.plot(xs, ys, lw=2.2)
        ax.plot(xs[0], ys[0], "go", ms=9)
        ax.plot(xs[-1], ys[-1], "r*", ms=15)
        n = max(1, len(xs) // 14)
        ax.quiver(xs[::n], ys[::n], np.cos(th[::n]), np.sin(th[::n]),
                  scale=22, width=0.005, color="gray")
        turn = float(np.sum(np.abs(np.diff(np.unwrap(th)))))
        label = str(d["shape"]) if "shape" in d.files else "n/a"
        ax.set_title(f"{os.path.basename(rel)[:-4]}\nlabel={label}  turn={turn:.2f} rad\n{note}",
                     fontsize=8.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle("All five are labelled UTURN and all have total_abs_turn ≈ 7–9 rad. "
                 "Only two are hairpins.", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "03_uturn_plans.png"), dpi=120)


if __name__ == "__main__":
    fig_r_ladder_low()
    fig_uturn_generality()
    fig_uturn_plans()
    print("wrote 01_r_ladder_low.png, 02_uturn_generality.png, 03_uturn_plans.png")
