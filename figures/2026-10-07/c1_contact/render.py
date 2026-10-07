#!/usr/bin/env python3
"""Job 231 (C1, centred library, layer OFF, truth): plan vs driven track at
wall contact. Plan coloured by hull clearance (green >= 0.5 m, orange
0.15-0.5, red < 0.15); driven track in blue; red footprints where the body
overlaps a wall. Rows are plans, columns arms.

    .venv/bin/python figures/2026-10-07/c1_contact/render.py
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, yaml
from matplotlib.collections import LineCollection
from PIL import Image
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "figures/2026-10-07/s1_supervisor"))
from render import robot
RUN = os.path.join(ROOT, "run_data/2026-10-07_job231_c1-centred")
MAP = os.path.join(ROOT, "src/rudn-ordjo-building/maps/floor_6")
PLANS = ["00388", "00115", "00150", "00105"]
ARMS = [("LTN", "ours-lib"), ("RTN", "RPP"), ("VTN", "VPP")]
exec(open(os.path.join(os.path.dirname(__file__), "clear.py")).read())

meta = yaml.safe_load(open(MAP + ".yaml")); img = np.asarray(Image.open(MAP + ".png").convert("L"))
res, (ox, oy, _) = meta["resolution"], meta["origin"]; ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
fig, axs = plt.subplots(len(PLANS), len(ARMS), figsize=(6.5 * len(ARMS), 5 * len(PLANS)), squeeze=False)
for i, plan in enumerate(PLANS):
    for j, (cfg, lab) in enumerate(ARMS):
        ax = axs[i, j]
        rows = {(r := json.loads(l))["plan"]: r for l in open(f"{RUN}/{cfg}/rows.s0.jsonl")}
        r = rows["floor_6_v2_" + plan]
        if not r.get("track_path"):
            ax.set_title(f"{lab}, plan {plan}: {r['outcome']}, no track"); ax.axis("off"); continue
        z = np.load(os.path.join(RUN, cfg, "s0", os.path.basename(r["track_path"])))
        P, T = z["plan_poses"], z["track"][:, 1:4]
        cp, ct = clear(P), clear(T)
        lo = np.minimum(P[:, :2].min(0), T[:, :2].min(0)) - 1; hi = np.maximum(P[:, :2].max(0), T[:, :2].max(0)) + 1
        ax.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        seg = np.stack([P[:-1, :2], P[1:, :2]], 1)
        col = np.where(cp[:-1] >= 0.5, "tab:green", np.where(cp[:-1] >= 0.15, "tab:orange", "tab:red"))
        ax.add_collection(LineCollection(seg, colors=col, lw=4, alpha=0.6, zorder=2))
        ax.plot(T[:, 0], T[:, 1], color="tab:blue", lw=1.5, zorder=3)
        for k in np.flatnonzero(ct < 0)[::4]:
            robot(ax, T[k], "tab:red", alpha=0.3, z=4)
        robot(ax, P[0], "k", fill=False, lw=1.2, z=5); robot(ax, P[-1], "tab:blue", fill=False, lw=1.5, z=5)
        ax.set_title(f"{lab}, plan {plan}: {r['outcome']}\nplan min clearance {cp.min():.2f} m, "
                     f"track min {ct.min():.2f} m, contact {float((ct < 0).mean()):.0%} of track", fontsize=10)
fig.suptitle("Job 231 (centred plans, layer off, truth loc.): thick = plan (green >=0.5 m, orange 0.15-0.5, red <0.15), "
             "blue = driven, red boxes = body in wall", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.98)); out = os.path.join(os.path.dirname(__file__), "contact_examples.png")
fig.savefig(out, dpi=80); print(out)
