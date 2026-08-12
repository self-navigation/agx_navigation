#!/usr/bin/env python3
"""Render the 2026-08-13 figure set. See figures/2026-08-13/README.md.

Offline only: matplotlib + numpy from the repo venv, like every other
`tools/plot_*.py`. Reads the gitignored data directories and writes PNGs into
`figures/2026-08-13/`; nothing here imports the control stack.

    .venv/bin/python figures/2026-08-13/render.py

Plant for every panel: 2026-08-07 wheel `mu2=0.45`, world ground `mu=1.0`.
"""

import collections
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SET = [("floor_1_00049", "STRAIGHT"), ("floor_6_00023", "CORNER"),
       ("floor_6_00018", "S"), ("floor_6_00047", "ZIGZAG"),
       ("floor_6_00056", "TIGHT V"), ("floor_6_00031", "U-TURN"),
       ("floor_6_00025", "LOOP")]
# The threshold that splits each bimodal shape's two outcomes. Chosen from the
# ladder scatter (figure 3), where the gap between clusters is wide and obvious;
# they are read off the data, not tuned.
BAD = {"floor_6_00047": ("ZIGZAG", 1.5), "floor_6_00031": ("U-TURN", 2.0),
       "floor_6_00056": ("TIGHT V", 0.6)}


def plan(name):
    return np.load(f"traj_data/{name}.npz")["poses"]


def fig_shapes():
    """Figure 1 -- the seven evaluation plans, to scale, with length and turning."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, (name, lab) in zip(axes.ravel(), SET):
        p = plan(name)
        c = "#d62728" if lab == "U-TURN" else "#555"
        ax.plot(p[:, 0], p[:, 1], lw=2, color=c)
        ax.plot(*p[0, :2], "o", ms=5, color="g")
        ax.plot(*p[-1, :2], "s", ms=5, color="k")
        d = np.hypot(*np.diff(p[:, :2], axis=0).T).sum()
        dth = np.abs(np.diff(np.unwrap(p[:, 2]))).sum()
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.set_title(f"{lab}\n{d:.1f} m, turn {np.degrees(dth):.0f}°",
                     fontsize=9, color=c)
    axes.ravel()[-1].axis("off")
    fig.suptitle("The seven evaluation plans (green = start, black = goal)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_shapes_overview.png", dpi=140)
    plt.close(fig)


def _trace_xy(path):
    """Driven-phase xy of one trace, or None if the rollout never drove.

    Empty traces are real and expected at ~1% -- a rollout whose terrain patches
    failed to spawn is invalidated after the reset phase is already traced, so
    the file exists with reset rows and no step rows. Skipping it is correct;
    the alternative (plotting a rollout that never happened) is not."""
    rows = [r for r in csv.DictReader(open(path)) if r["phase"] == "step"]
    if not rows:
        return None
    return np.array([[float(r["x"]), float(r["y"])] for r in rows])


def fig_uturn_modes():
    """Figure 2 -- where on the U-turn the deviation actually happens."""
    p = plan("floor_6_00031")[:, :2]

    def dev(xy):
        return np.min(np.hypot(xy[:, None, 0] - p[None, :, 0],
                               xy[:, None, 1] - p[None, :, 1]), axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    for ax, arm in zip(axes[:2], ["tuned", "default"]):
        ax.plot(p[:, 0], p[:, 1], "k--", lw=2.5, label="nominal PMP plan", zorder=5)
        for f in sorted(glob.glob(f"uturn_traces/{arm}/*.csv")):
            xy = _trace_xy(f)
            if xy is None:
                continue
            bad = dev(xy).max() > 2.0
            ax.plot(xy[:, 0], xy[:, 1], lw=1.2, alpha=0.85,
                    color="#d62728" if bad else "#1f77b4", zorder=4 if bad else 3)
        ax.plot(*p[0], "o", ms=8, color="g", zorder=6)
        ax.plot(*p[-1], "s", ms=8, color="k", zorder=6)
        ax.set_aspect("equal")
        ax.set_title(f"{arm} gains  (blue = good mode, red = bad)", fontsize=11)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8, loc="lower left")
    ax = axes[2]
    for arm, ls in (("tuned", "-"), ("default", "--")):
        for f in sorted(glob.glob(f"uturn_traces/{arm}/*.csv")):
            xy = _trace_xy(f)
            if xy is None:
                continue
            d = dev(xy)
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
            ax.plot(s, d, ls, lw=1.0, alpha=.8,
                    color="#d62728" if d.max() > 2.0 else "#1f77b4")
    ax.set_xlabel("distance travelled (m)")
    ax.set_ylabel("deviation from plan (m)")
    ax.set_title("when the deviation happens", fontsize=11)
    ax.grid(alpha=.25)
    fig.suptitle("U-turn (floor_6_00031): 30 rollouts, two gain settings", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_uturn_modes.png", dpi=140)
    plt.close(fig)


def fig_ladder():
    """Figure 3 -- bad-mode frequency across the q_cross ladder."""
    rows = [json.loads(l) for l in open("soak_data/soak_20260813_ladder.jsonl") if l.strip()]
    ok = [r for r in rows if r.get("max_cross") is not None]
    g = collections.defaultdict(list)
    for r in ok:
        g[(r["trajectory"], round(r["q_cross"], 4), round(r["r_omega"], 4))].append(r["max_cross"])

    qs = [0.1, 0.276, 0.6, 1.5, 10.0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for t, (lab, thr) in BAD.items():
        y = [100 * np.mean(np.array(g[(t, q, 2.618)]) > thr) for q in qs]
        ax.plot(qs, y, "o-", lw=2, ms=7, label=f"{lab} (bad > {thr} m)")
        yd = 100 * np.mean(np.array(g[(t, 10.0, 0.25)]) > thr)
        ax.plot([10.0], [yd], "x", ms=11, mew=2.5, color=ax.lines[-1].get_color())
    ax.set_xscale("log")
    ax.set_xlabel("q_cross   (r_omega = 2.618;  × = old default r=0.25)")
    ax.set_ylabel("% of rollouts in the bad mode")
    ax.set_ylim(-4, 104)
    ax.axvline(0.276, color="k", ls=":", lw=1.5)
    ax.text(0.276, 103, " adopted", fontsize=8, va="top")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    ax.set_title("Bad-mode frequency vs q_cross  (n≈58 per point)")

    ax = axes[1]
    rng = np.random.default_rng(0)          # jitter only; fixed so the figure redraws identically
    for i, q in enumerate(qs):
        v = np.array(g[("floor_6_00031", q, 2.618)])
        ax.scatter(np.full(len(v), i) + rng.uniform(-.16, .16, len(v)), v, s=9, alpha=.6,
                   color="#1f77b4" if np.mean(v > 2.0) < 0.5 else "#d62728")
    v = np.array(g[("floor_6_00031", 10.0, 0.25)])
    ax.scatter(np.full(len(v), len(qs)) + rng.uniform(-.16, .16, len(v)), v, s=9,
               alpha=.6, color="#1f77b4")
    ax.set_xticks(range(len(qs) + 1))
    ax.set_xticklabels([f"{q:g}" for q in qs] + ["10\n(r=.25)"], fontsize=9)
    ax.axhline(2.0, color="k", ls=":", lw=1)
    ax.set_xlabel("q_cross")
    ax.set_ylabel("max |e_cross| (m)")
    ax.set_title("U-turn: every rollout, per rung")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_ladder_modes.png", dpi=140)
    plt.close(fig)


def fig_epsilon(src="epsilon_data/jsweep.jsonl"):
    """Figure 4 -- does scoring in J change which corrector wins?

    The whole point of the figure is that the two metrics can DISAGREE, so it
    plots them against each other rather than side by side: a point below the
    diagonal-equivalent is a rollout the two metrics rank differently.
    """
    if not os.path.exists(src):
        print(f"[skip] {src} not present")
        return
    rows = [json.loads(l) for l in open(src) if l.strip()]
    names = {n: lab for n, lab in SET}
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["plan"].replace(".npz", ""), r["label"])].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    # Panel 1: per-shape mean, both metrics, tuned vs default.
    ax = axes[0]
    shapes = [n for n, _ in SET if (n, "tuned") in by]
    x = np.arange(len(shapes))
    for i, (metric, key, scale) in enumerate((("max|e_cross| (m)", "max_cross", 1.0),
                                              ("J", "j_total", 1.0))):
        for arm, off, hatch in (("tuned", -0.2, None), ("default", 0.2, "//")):
            v = [np.mean([r[key] for r in by[(s, arm)]]) * scale for s in shapes]
            axes[i].bar(x + off, v, 0.38, label=arm, hatch=hatch,
                        color="#1f77b4" if arm == "tuned" else "#ff7f0e", alpha=.85)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([names[s] for s in shapes], rotation=35, fontsize=8, ha="right")
        axes[i].set_ylabel(metric)
        axes[i].set_title(f"per shape, mean of 5 — {metric}")
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=.3, axis="y")

    # Panel 3: the disagreement, per shape. Ratio default/tuned under each
    # metric; >1 means the tuned gains win. A shape where the two ratios sit on
    # opposite sides of 1 is one the metrics rank differently.
    ax = axes[2]
    rc, rj = [], []
    for s in shapes:
        rc.append(np.mean([r["max_cross"] for r in by[(s, "default")]]) /
                  np.mean([r["max_cross"] for r in by[(s, "tuned")]]))
        rj.append(np.mean([r["j_total"] for r in by[(s, "default")]]) /
                  np.mean([r["j_total"] for r in by[(s, "tuned")]]))
    ax.axhline(1, color="k", lw=1)
    ax.axvline(1, color="k", lw=1)
    ax.scatter(rc, rj, s=70, zorder=4, color="#333")
    # Log axes: these are RATIOS, so 0.8x and 1.25x are the same size of
    # disagreement and a linear axis would hide the first and shout the second.
    lo, hi = 0.6, max(max(rc), max(rj)) * 1.6
    for s, a, b in zip(shapes, rc, rj):
        ax.annotate(names[s], (a, b), fontsize=8.5,
                    xytext=(7, -3) if a > 1 else (-7, 6), ha="left" if a > 1 else "right",
                    textcoords="offset points")
    ax.fill_between([lo, 1], 1, hi, color="#d62728", alpha=.09)
    ax.fill_between([1, hi], lo, 1, color="#d62728", alpha=.09)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("default / tuned  in max|e_cross|   (>1 = tuned wins)")
    ax.set_ylabel("default / tuned  in J   (>1 = tuned wins)")
    ax.set_title("shaded = the two metrics DISAGREE")
    ax.grid(alpha=.3)
    fig.suptitle("Does changing the objective change the conclusion?", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_epsilon_vs_cross.png", dpi=140)
    plt.close(fig)


def fig_what_is_j(src="epsilon_data/uturn.jsonl"):
    """Figure 5 -- what the two metrics ARE, on one picture.

    `max|e_cross|` is the HEIGHT of the tallest spike in the deviation curve;
    `J` is (the weighted) AREA under the whole curve, plus what the corrector
    spent correcting, plus a penalty for stopping short of the goal. Written for
    someone who finds metres intuitive and `J` abstract -- which is everyone,
    including us.
    """
    p = plan("floor_6_00031")[:, :2]

    def curve(path):
        xy = _trace_xy(path)
        d = np.min(np.hypot(xy[:, None, 0] - p[None, :, 0],
                            xy[:, None, 1] - p[None, :, 1]), axis=1)
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        return s, d

    scored = {r["trace"]: r for r in
              (json.loads(l) for l in open(src))} if os.path.exists(src) else {}
    files = sorted(glob.glob("uturn_traces/tuned/*.csv") + glob.glob("uturn_traces/default/*.csv"))
    files = [f for f in files if os.path.basename(f) in scored]
    if not files:
        print("[skip] no scored U-turn traces")
        return
    files.sort(key=lambda f: scored[os.path.basename(f)]["j_total"])
    picks = [files[0], files[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, f in zip(axes, picks):
        rec = scored[os.path.basename(f)]
        s, d = curve(f)
        ax.fill_between(s, d, alpha=.30, color="#1f77b4",
                        label=f"area ≈ J's tracking term ({rec['j_tracking']:.0f})")
        ax.plot(s, d, lw=1.6, color="#1f77b4")
        k = int(np.argmax(d))
        ax.annotate(f"max|e_cross| = {d[k]:.2f} m", (s[k], d[k]),
                    xytext=(-14, -26), textcoords="offset points", fontsize=10,
                    ha="right", arrowprops=dict(arrowstyle="->", color="#d62728"),
                    color="#d62728")
        ax.axhline(d[k], color="#d62728", ls=":", lw=1.2)
        ax.set_xlabel("distance travelled (m)")
        ax.set_title(f"{rec['label']} gains\nJ = {rec['j_total']:.0f}   "
                     f"(track {rec['j_tracking']:.0f} + effort {rec['j_control']:.0f}"
                     f" + stopped-short {rec['j_terminal']:.0f})", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("deviation from plan (m)")
    fig.suptitle("max|e_cross| is the HEIGHT of the worst spike;  J is the AREA "
                 "under the whole curve, plus effort, plus stopping short", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/05_what_is_J.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_shapes()
    fig_uturn_modes()
    fig_ladder()
    fig_epsilon()
    fig_what_is_j()
    print(f"wrote {OUT}/")
