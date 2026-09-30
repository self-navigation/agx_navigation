"""Re-join Phase 0 (#11): where does the re-join TPBVP succeed, and does a
better initial guess move it?

Inputs (tools/rejoin_phase0.py run, same 300 problems, seed 0, local, 12 jobs):
  rejoin_phase0_local300.jsonl  --guess blend (linear x0->target, zero costates), a67083d
  rejoin_phase0_plan300.jsonl   --guess plan  (nominal segment, deviation faded out,
                                               plan costates for cost A)
Copied here because /tmp does not survive.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RUNS = {"blend guess": "rejoin_phase0_local300.jsonl", "plan guess": "rejoin_phase0_plan300.jsonl"}
COSTS = {"field": ("A: field cost", "tab:blue"), "effort": ("B: min effort", "tab:orange")}
STYLE = {"blend guess": dict(ls="--", marker="o", mfc="none"), "plan guess": dict(ls="-", marker="o")}


def load(f):
    return [json.loads(l) for l in open(HERE / f)]


def wilson(k, n, z=1.96):
    n = np.maximum(n, 1)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binned(x, s, edges):
    idx = [(x >= a) & (x < b) for a, b in zip(edges[:-1], edges[1:])]
    k = np.array([s[i].sum() for i in idx])
    n = np.array([i.sum() for i in idx])
    return k, n


fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
edges = np.geomspace(0.1, 10, 9)
mid = np.sqrt(edges[:-1] * edges[1:])
ax = axes[0]
for run, f in RUNS.items():
    rows = load(f)
    Tw = np.array([r["T_w"] for r in rows])
    for c, (lab, col) in COSTS.items():
        s = np.array([r[c]["status"] == "ok" for r in rows])
        k, n = binned(Tw, s, edges)
        lo, hi = wilson(k, n)
        p = k / np.maximum(n, 1)
        ax.errorbar(mid * (1.03 if run == "plan guess" else 0.97), p,
                    yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)],
                    color=col, capsize=3, label=f"{lab}, {run} ({s.mean():.0%})", **STYLE[run])
ax.set_xscale("log")
ax.set_xticks([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"])
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("re-join window $T_w$ [s]")
ax.set_ylabel("solve success rate")
ax.set_title("Success vs window (95% Wilson CI; n~37 per bin)")
ax.grid(alpha=.3)
ax.legend(fontsize=8, loc="upper left")

# Feasibility margin: T_w over the time the wheels need just to change speed.
redges = np.array([0, 1, 1.5, 3, 10, 30, 1e9])
tedges = np.array([0.1, 0.3, 1, 3, 10.01])
rows = load(RUNS["plan guess"])
Tw = np.array([r["T_w"] for r in rows])
ratio = Tw / np.maximum([r["t_min_wheel"] for r in rows], 1e-9)
for ax, (c, (lab, _)) in zip(axes[1:], COSTS.items()):
    s = np.array([r[c]["status"] == "ok" for r in rows])
    P = np.full((len(redges) - 1, len(tedges) - 1), np.nan)
    N = np.zeros_like(P, dtype=int)
    for i in range(len(redges) - 1):
        for j in range(len(tedges) - 1):
            m = (ratio >= redges[i]) & (ratio < redges[i + 1]) & (Tw >= tedges[j]) & (Tw < tedges[j + 1])
            N[i, j] = m.sum()
            if m.any():
                P[i, j] = s[m].mean()
    im = ax.imshow(P, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if N[i, j]:
                ax.text(j, i, f"{P[i, j]:.0%}\nn={N[i, j]}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(tedges) - 1), ["0.1–0.3", "0.3–1", "1–3", "3–10"])
    ax.set_yticks(range(len(redges) - 1), ["<1", "1–1.5", "1.5–3", "3–10", "10–30", ">30"])
    ax.set_xlabel("re-join window $T_w$ [s]")
    ax.set_ylabel("feasibility margin $T_w / t_{min,wheel}$")
    ax.set_title(f"{lab}, plan guess")
fig.colorbar(im, ax=axes[1:], label="success rate", shrink=.8)
fig.suptitle("Re-join TPBVP Phase 0 (#11): 300 problems, same draws in both runs")
fig.savefig(HERE / "rejoin_phase0_regimes.png", dpi=130, bbox_inches="tight")
