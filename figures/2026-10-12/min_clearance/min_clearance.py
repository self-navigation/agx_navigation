#!/usr/bin/env python3
"""Continuous companion to the contact count: each run's minimum body-to-wall clearance (row field wall_min_clear,
from the footprint over the 5 cm baked map; <= 0 means the footprint overlapped a wall pixel), per arm, as an ECDF.
One panel per localizer, curves per controller. Also prints medians, the share below 5 / 10 cm, and a plan-level
Wilcoxon on per-plan mean clearance for the headline pairs. usage: min_clearance.py
"""
import glob, json, os, collections
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RD = os.path.join(HERE, "../../../run_data")
J5, J6, J7 = "2026-10-12_job255_steerable", "2026-10-12_job256_steerable-rtab", "2026-10-12_job257_kprior"
LOCS = {"true pose": [(J5, "TR"), (J5, "T"), (J7, "FT"), (J7, "PT")],
        "amcl": [(J5, "QR"), (J5, "Q0"), (J7, "FQ"), (J7, "PQ")],
        "rtab": [(J6, "RR"), (J6, "RQ"), (J7, "FR"), (J7, "PR")]}
NAMES = ["TVLQR", "TVLQG", "TVLQR + fixed k", "TVLQG + prior"]
COL = ["0.5", "tab:blue", "tab:orange", "tab:green"]


def load(job, cell):
    d = {}
    for f in glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("wall_min_clear") is not None:
                    d[(os.path.basename(r["plan"]), r["seed"])] = float(r["wall_min_clear"])
    return d


plt.rcParams.update({"font.size": 17})
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
lines = []
for ax, (loc, cells) in zip(axs, LOCS.items()):
    D = [load(*c) for c in cells]
    for d, n, c in zip(D, NAMES, COL):
        v = np.sort(list(d.values()))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=2.5, label=n)
        lines.append(f"{loc:9s} {n:16s} n {len(v)}  median {np.median(v):.3f} m  <=0 {np.mean(v <= 0):.2f}  "
                     f"<0.05 {np.mean(v < 0.05):.2f}  <0.10 {np.mean(v < 0.10):.2f}")
    for i, j in ((0, 1), (1, 3), (0, 2)):
        a, b = D[i], D[j]; k = set(a) & set(b)
        pa, pb = collections.defaultdict(list), collections.defaultdict(list)
        for p, s in k: pa[p].append(a[(p, s)]); pb[p].append(b[(p, s)])
        x = np.array([np.mean(pa[p]) for p in sorted(pa)]); y = np.array([np.mean(pb[p]) for p in sorted(pa)])
        lines.append(f"    plan-level {NAMES[i]} vs {NAMES[j]}: median per-plan diff {np.median(y - x):+.3f} m, "
                     f"Wilcoxon p {wilcoxon(x, y).pvalue:.2g} ({len(x)} plans)")
    ax.axvline(0, color="k", lw=1.5); ax.set_xlim(-0.06, 0.3); ax.set_ylim(0, 1)
    ax.set_title(f"{loc}", fontweight="bold"); ax.set_ylabel("share of runs"); ax.grid(alpha=.3)
axs[0].legend(loc="lower right")
axs[-1].set_xlabel("closest the robot body came to a wall in the run (m); left of 0 = touched")
fig.suptitle("How close each run came to a wall\n(126 runs per curve, mu 0.5 patches; further right = safer)", fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "min_clearance.png"), dpi=110)
open(os.path.join(HERE, "min_clearance.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
