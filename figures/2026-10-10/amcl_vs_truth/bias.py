#!/usr/bin/env python3
"""Is amcl's error a map-dependent BIAS (same direction at the same place in
every run) or run-specific drift?  Bins the world-frame error vector
(believed - truth) of every LANq/LAN/LANqe tick into 1 m map cells and
compares the magnitude of the mean vector with the mean magnitude.
    .venv/bin/python figures/2026-10-10/amcl_vs_truth/bias.py
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, OUT)
from render import rows, analyse, _map_axes
R_ = rows()
cells = {}
pts = []
for (plan, c, seed), r in sorted(R_.items()):
    if c not in ("LAN", "LANq", "LANqe"): continue
    st, s = analyse(r)
    if s is None: continue
    m = s["drive"] & (s["t"] > 2)
    e = s["bel"][m, :2] - s["tru"][m, :2]; p = s["tru"][m, :2]
    for (x, y), (ex, ey) in zip(p, e):
        cells.setdefault((int(np.floor(x)), int(np.floor(y))), []).append((ex, ey, f"{plan}/{seed}"))
    pts.append((p, e))
rows_out = []
for k, v in cells.items():
    a = np.array([(x, y) for x, y, _ in v]); nruns = len({r for _, _, r in v})
    if len(a) < 100 or nruns < 6: continue
    mean = a.mean(axis=0); mag = np.linalg.norm(a, axis=1).mean()
    rows_out.append((k, len(a), nruns, mean, mag, np.linalg.norm(mean) / mag))
rows_out.sort(key=lambda r: -r[1])
print("cell(x,y)  ticks runs  mean err vector (m)   mean|err|  consistency=|mean|/mean|err|")
for k, n, nr, mean, mag, cons in rows_out[:40]:
    print(f"{k}  {n:5d} {nr:3d}   ({mean[0]:+.3f},{mean[1]:+.3f})  {mag:.3f}  {cons:.2f}")
cons = [r[5] for r in rows_out]; w = [r[1] for r in rows_out]
print(f"\n{len(rows_out)} map cells with >=100 ticks from >=6 runs: tick-weighted consistency {np.average(cons, weights=w):.2f}; "
      f"cells with consistency > 0.5: {sum(c > 0.5 for c in cons)}, > 0.7: {sum(c > 0.7 for c in cons)}")
# figure: mean error vector per map cell
fig, ax = plt.subplots(figsize=(11, 9))
P = np.vstack([p for p, _ in pts]); _map_axes(ax, P[:, 0], P[:, 1], pad=1.0)
ax.plot(P[::20, 0], P[::20, 1], ",", color="0.7")
for k, n, nr, mean, mag, c in rows_out:
    ax.arrow(k[0] + 0.5, k[1] + 0.5, mean[0] * 2, mean[1] * 2, head_width=0.12, color=plt.cm.viridis(c), lw=1)
    ax.text(k[0] + 0.5, k[1] + 0.5, f"{c:.1f}", fontsize=5, color="k")
ax.set_title("amcl: mean (believed - truth) per 1 m map cell, arrows x2, over LAN+LANq+LANqe; label/colour = consistency |mean|/mean|err| (1 = same direction every run)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "amcl_bias_map.png"), dpi=130)
