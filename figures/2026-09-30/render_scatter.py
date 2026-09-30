"""Re-join Phase 0 (#11) as scatter plots, one point per problem, no bucketing.

rejoin_phase0_scatter.png: rows = cost A (field) / cost B (min effort),
columns = T_w vs feasibility margin (T_w / t_min_wheel), T_w vs lateral
offset |e_cross|, T_w vs heading error |e_theta|. Colour = outcome: solved,
singular Jacobian, mesh-node cap. Data: rejoin_phase0_vm2000.jsonl (2000
problems, plan guess, seed 1, VM).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
rows = [json.loads(l) for l in open(HERE / "rejoin_phase0_vm2000.jsonl")]
OUT = {"ok": ("solved", "tab:green", 6, .45),
       "sing": ("singular Jacobian", "tab:red", 9, .7),
       "nodes": ("mesh-node cap", "tab:purple", 14, .9)}


def kind(r):
    if r["status"] == "ok":
        return "ok"
    return "nodes" if "mesh nodes" in r.get("message", "") else "sing"


tw = np.array([r["T_w"] for r in rows])
X = {"feasibility margin  $T_w / t_{min,wheel}$":
         np.array([r["T_w"] / max(r["t_min_wheel"], 1e-3) for r in rows]),
     # Floored at 1 mm / 0.1 deg (drawn on the floor): below that the
     # deviation is ~zero along that axis and a log scale wastes the panel.
     "lateral offset $|e_{cross}|$ (m), floored at 1 mm":
         np.maximum(np.abs([r["e_cross"] for r in rows]), 1e-3),
     "heading error $|e_\\theta|$ (deg), floored at 0.1°":
         np.maximum(np.degrees(np.abs([r["e_theta"] for r in rows])), 0.1)}
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
for i, (cost, lab) in enumerate((("field", "A: field cost"), ("effort", "B: min effort"))):
    ks = np.array([kind(r[cost]) for r in rows])
    for j, (yl, y) in enumerate(X.items()):
        ax = axes[i, j]
        for kk, (nm, c, s, a) in OUT.items():
            sel = ks == kk
            ax.scatter(tw[sel], y[sel], s=s, c=c, alpha=a, lw=0,
                       label=f"{nm} ({sel.sum()})", zorder=3 if kk != "ok" else 2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(yl); ax.grid(alpha=.3, which="both", lw=.4)
        if j == 0:
            ax.axhline(1, color="k", lw=.8, ls="--")
            ax.set_title(f"{lab}", loc="left", fontweight="bold")
            ax.legend(fontsize=8, loc="lower right")
for ax in axes[1]:
    ax.set_xlabel("re-join window $T_w$ (s)")
fig.suptitle("Re-join Phase 0: every problem (2000, plan guess). Below the dashed line the wheels "
             "cannot even reach the target speed in time")
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(HERE / "rejoin_phase0_scatter.png", dpi=110)
