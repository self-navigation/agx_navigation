#!/usr/bin/env python3
"""Plot what the TVLQR gain search actually explored, from its evaluation cache.

Consumes the JSONL written by `agx_planning.tuning.tune_tvlqr` (one record per
evaluation: gains, score, per-trajectory breakdown, wall time) and draws:

    left    the (q_cross, r_omega) plane, log-log, one marker per evaluation,
            coloured by mean max|e_cross| and numbered in evaluation order, with
            the simplex's path drawn between consecutive points. This is the
            "is it converging, or wandering" picture -- a healthy Nelder-Mead
            shows a spread of early probes collapsing onto a small cluster.
    right   best-so-far against evaluation number, plus each trajectory's own
            error at the best point. Non-increasing by construction; the shape
            says whether the budget was spent or wasted.

Per-trajectory errors are drawn because the aggregate can improve while one
shape gets worse -- exactly the straight-vs-corner split that a single number
hid in the RL work.

OFFLINE TOOL (matplotlib in a venv), same rule as plot_run.py.

    .venv/bin/python tools/plot_tune_landscape.py tvlqr_tune.jsonl --out figures
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#1baf7a", "#eda100", "#2a78d6", "#9b59b6", "#c0392b", "#16a085"]


def read(path):
    meta, rows = {}, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn final line from an interrupted run
            if rec.get("_meta"):
                meta = rec["_meta"]
            elif "x" in rec and "fx" in rec:
                rows.append(rec)
    return meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", help="tune_tvlqr JSONL evaluation cache")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--name", default="tvlqr_tune_landscape")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta, rows = read(args.cache)
    if not rows:
        raise SystemExit(f"no evaluations in {args.cache}")

    q = np.array([r.get("q_cross", 10.0 ** r["x"][0]) for r in rows])
    r_om = np.array([r.get("r_omega", 10.0 ** r["x"][1]) for r in rows])
    fx = np.array([r["fx"] for r in rows], dtype=float)

    # inf marks an invalid evaluation (a rollout failed). Drawing it would blow
    # the colour scale; it is reported in the title instead.
    valid = np.isfinite(fx)
    n_invalid = int((~valid).sum())

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6.2),
                                     gridspec_kw={"width_ratios": [1.1, 1]})

    ax_l.plot(q[valid], r_om[valid], "-", color="#c9c9c6", lw=0.9, zorder=1)
    sc = ax_l.scatter(q[valid], r_om[valid], c=fx[valid], cmap="viridis_r",
                      s=90, zorder=3, edgecolor="white", linewidth=0.6)
    for i, (qi, ri, ok) in enumerate(zip(q, r_om, valid), start=1):
        if ok:
            ax_l.annotate(str(i), (qi, ri), fontsize=6, ha="center", va="center",
                          color="white", zorder=4)
    if n_invalid:
        ax_l.scatter(q[~valid], r_om[~valid], marker="x", c="#c0392b", s=70,
                     zorder=3, label=f"invalid ({n_invalid})")
        ax_l.legend(fontsize=8, loc="best")

    best_i = int(np.argmin(np.where(valid, fx, np.inf)))
    ax_l.plot(q[best_i], r_om[best_i], "*", ms=22, mfc="none", mec="#c0392b",
              mew=2.0, zorder=5)
    ax_l.annotate(f"best\nq={q[best_i]:.3g}\nr={r_om[best_i]:.3g}",
                  (q[best_i], r_om[best_i]), textcoords="offset points",
                  xytext=(12, 12), fontsize=8, color="#c0392b")

    # Log axes: the search runs in log10, so equal visual distance is equal
    # search distance. A linear plot would bunch every small gain into a corner.
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel("q_cross")
    ax_l.set_ylabel("r_omega")
    ax_l.set_title("gains explored (numbered in evaluation order)")
    ax_l.grid(alpha=0.3, which="both")
    fig.colorbar(sc, ax=ax_l, label="mean max|e_cross| [m]")

    best_curve, best = [], math.inf
    for v in fx:
        best = min(best, v) if math.isfinite(v) else best
        best_curve.append(best)
    steps = np.arange(1, len(rows) + 1)
    ax_r.plot(steps, best_curve, "-o", color="#c0392b", lw=2.0, ms=4,
              label="best so far")
    ax_r.plot(steps[valid], fx[valid], "o", color="#8a8985", ms=4, alpha=0.6,
              label="each evaluation")

    names = sorted({k for r in rows for k in (r.get("per_traj") or {})})
    for j, nm in enumerate(names):
        ys = [(r.get("per_traj") or {}).get(nm, float("nan")) for r in rows]
        ax_r.plot(steps, ys, "-", color=SERIES[j % len(SERIES)], lw=1.2,
                  alpha=0.85, label=nm)

    start = rows[0]["fx"]
    if math.isfinite(start):
        ax_r.axhline(start, color="#2a78d6", ls=":", lw=1.2,
                     label=f"start ({start:.3f} m)")
    ax_r.set_xlabel("evaluation")
    ax_r.set_ylabel("max|e_cross| [m]")
    ax_r.set_title("convergence, and each trajectory underneath it")
    ax_r.grid(alpha=0.3)
    ax_r.legend(fontsize=7, loc="best")

    sub = f"{len(rows)} evaluations"
    if meta.get("trajectories"):
        sub += f" · {len(meta['trajectories'])} trajectories · seed {meta.get('seed')}"
    fig.suptitle(f"TVLQR gain search — {sub}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(args.out, f"{args.name}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  best: q_cross={q[best_i]:.4g}  r_omega={r_om[best_i]:.4g}  "
          f"-> {fx[best_i]:.4f} m   (start {start:.4f} m)")


if __name__ == "__main__":
    main()
