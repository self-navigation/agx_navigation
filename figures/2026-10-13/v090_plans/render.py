#!/usr/bin/env python3
"""v_max 0.9 plans (63 of 83 solved) against their v_max 0.448 twins: path shape, wall clearance, speed.

Writes v090_vs_slow.png (12 plans over the floor_6 map) and stats.txt (all 63). Clearance is the
centreline's minimum distance to a wall cell (EDT of the baked map); the robot half-width is ~0.29 m.
    .venv/bin/python figures/2026-10-13/v090_plans/render.py
"""
import glob, os
import numpy as np, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../.."))
FAST = f"{R}/run_data/2026-10-12_nojob_plan-set-v3-fast/v090/plans"
SLOW = f"{R}/run_data/2026-10-10_nojob_plan-set-v3/plans"
meta = yaml.safe_load(open(f"{R}/src/rudn-ordjo-building/maps/floor_6.yaml"))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
img = np.array(Image.open(f"{R}/src/rudn-ordjo-building/maps/floor_6.png").convert("L"))[::-1]
edt = distance_transform_edt(img != 0) * res


def clear(p):
    i = np.clip(((p[:, 1] - oy) / res).astype(int), 0, img.shape[0] - 1)
    j = np.clip(((p[:, 0] - ox) / res).astype(int), 0, img.shape[1] - 1)
    return edt[i, j]


def speed(d):
    p = d["poses"]; return np.hypot(*np.diff(p[:, :2], axis=0).T) / float(d["dt_sample"])


rows, names = [], sorted(os.path.basename(f) for f in glob.glob(f"{FAST}/*.npz"))
for n in names:
    f, s = np.load(f"{FAST}/{n}", allow_pickle=True), np.load(f"{SLOW}/{n}", allow_pickle=True)
    pf, ps = f["poses"][:, :2], s["poses"][:, :2]
    dev = np.min(np.linalg.norm(pf[:, None] - ps[None], axis=2), axis=1).max()  # fast path's worst distance from slow
    rows.append((n[:-4], len(pf) * float(f["dt_sample"]), len(ps) * float(s["dt_sample"]), speed(f).max(),
                 np.abs(f["wheel_cmds"]).max(), clear(pf).min(), clear(ps).min(), dev))
a = np.array([r[1:] for r in rows])
out = [f"{len(rows)} v090 plans vs their v0.448 twins (medians; min/max in brackets)"]
for k, lab in enumerate(["duration fast s", "duration slow s", "peak speed m/s", "peak wheel rad/s",
                         "min clearance fast m", "min clearance slow m", "max path offset fast-vs-slow m"]):
    out.append(f"  {lab:32s} {np.median(a[:, k]):6.2f} [{a[:, k].min():.2f}, {a[:, k].max():.2f}]")
out.append(f"  duration ratio median {np.median(a[:, 0] / a[:, 1]):.2f}")
out.append(f"  plans whose fast clearance < slow clearance - 5 cm: {int((a[:, 4] < a[:, 5] - 0.05).sum())}")
out.append(f"  plans with fast centreline clearance < 0.35 m: {int((a[:, 4] < 0.35).sum())} (slow: {int((a[:, 5] < 0.35).sum())})")
out += [f"  {r[0]:22s} clr fast {r[5]:.2f} slow {r[6]:.2f} offset {r[7]:.2f} peak {r[3]:.2f}" for r in
        sorted(rows, key=lambda r: r[7], reverse=True)[:8]]
open(f"{H}/stats.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))

pick = [r[0] for r in sorted(rows, key=lambda r: r[7], reverse=True)[:6]] + [rows[i][0] for i in range(0, len(rows), max(1, len(rows) // 6))][:6]
fig, ax = plt.subplots(3, 4, figsize=(16, 12))
for x, n in zip(ax.flat, pick):
    pf = np.load(f"{FAST}/{n}.npz")["poses"]; ps = np.load(f"{SLOW}/{n}.npz")["poses"]
    lo, hi = np.minimum(pf.min(0), ps.min(0))[:2] - 1.5, np.maximum(pf.max(0), ps.max(0))[:2] + 1.5
    x.imshow(img, cmap="gray", origin="lower", extent=[ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res])
    x.plot(ps[:, 0], ps[:, 1], "b-", lw=1.5, label="v_max 0.448")
    x.plot(pf[:, 0], pf[:, 1], "r-", lw=1.5, label="v_max 0.9")
    x.set_xlim(lo[0], hi[0]); x.set_ylim(lo[1], hi[1]); x.set_title(n, fontsize=9); x.set_aspect("equal")
ax.flat[0].legend(fontsize=8)
fig.suptitle("Plan set v3: v_max 0.9 (red) vs 0.448 (blue). Top half: largest path offset; rest: evenly spaced")
fig.tight_layout(); fig.savefig(f"{H}/v090_vs_slow.png", dpi=110)
