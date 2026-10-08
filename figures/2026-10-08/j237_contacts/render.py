#!/usr/bin/env python3
"""Job 237 orig, ours-lib: every run that touched a wall. Plan (blue), ground
truth track (black), hull-on-wall samples (red), first contact (X).
    .venv/bin/python figures/2026-10-08/j237_contacts/render.py
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, yaml
from PIL import Image
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(R, "src/agx_navigation/agx_planning"))
from agx_planning.supervisor.walls import SignedWallDistance
D = os.path.join(R, "run_data/2026-10-08_job237_door-slow-ab/")
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
swd = SignedWallDistance.from_map_yaml(MAP + ".yaml")
meta = yaml.safe_load(open(MAP + ".yaml")); img = np.asarray(Image.open(MAP + ".png").convert("L"))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
rows = [json.loads(l) for l in open(D + "orig/all_rows.jsonl")]
rows = [r for r in rows if r["arm"] == "ours-lib" and (r["wall_overlap_s"] or 0) > 0]
n = len(rows); nc = 6; nr = -(-n // nc)
fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 4 * nr))
for ax in axs.flat: ax.axis("off")
for ax, r in zip(axs.flat, rows):
    z = np.load(D + r["track_path"].split("door-slow-ab/")[1]); T, P = z["track"], z["plan_poses"]
    c = np.array([swd.body_clearance(*p) for p in T[:, 1:4]]); hit = c < 0; i = int(np.argmax(hit))
    cx, cy = T[i, 1], T[i, 2]
    ax.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255)
    ax.plot(P[:, 0], P[:, 1], "tab:blue", lw=1.5); ax.plot(T[:, 1], T[:, 2], "k", lw=1)
    ax.plot(T[hit, 1], T[hit, 2], ".", color="tab:red", ms=4); ax.plot(cx, cy, "X", color="red", ms=12)
    ax.set_xlim(cx - 2.5, cx + 2.5); ax.set_ylim(cy - 2.5, cy + 2.5); ax.set_aspect("equal"); ax.axis("on")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{r['plan'][-5:]}: {'доехал' if r['outcome']=='arrived' else 'не доехал'}, касание {r['wall_overlap_s']} с", fontsize=9)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color="tab:blue", lw=2, label="план (PMP)"),
                    Line2D([], [], color="k", lw=1.5, label="фактическая траектория (Gazebo)"),
                    Line2D([], [], color="tab:red", marker=".", ls="", ms=10, label="корпус касается стены"),
                    Line2D([], [], color="red", marker="X", ls="", ms=12, label="первое касание")],
           loc="upper center", ncol=4, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.suptitle("Касания стен (ours-lib, задание 237): окно 5x5 м вокруг первого касания; подпись: план, исход, время касания",
             fontsize=13, y=0.965)
fig.tight_layout(rect=(0, 0, 1, 0.95)); out = os.path.join(os.path.dirname(__file__), "contacts.png"); fig.savefig(out, dpi=70); print(out)
