#!/usr/bin/env python3
"""Job 240 (#50, track D): ours-lib in the full stack (truth, phantom walls, patches)
with plain TVLQR (LTN) vs sideslip TVLQR k = -0.33 (LTNk). Tracks over the map for
the four plans where LTNk's max cross-track regressed most and the two where it
improved most; per-plan contact table in stats.txt.
    .venv/bin/python figures/2026-10-08/j240_sideslip_fullstack/render.py
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, yaml
from PIL import Image

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
J = os.path.join(R, "run_data/2026-10-08_job240_sideslip-fullstack")
OUT = os.path.dirname(os.path.abspath(__file__))


def rows(c):
    out = {}
    for line in open(os.path.join(J, c, "rows.s0.jsonl")):
        r = json.loads(line)
        r["track_local"] = os.path.join(J, c, "s0", os.path.basename(r["track_path"])) if r.get("track_path") else None
        out[r["plan"]] = r
    return out


A, B = rows("LTN"), rows("LTNk")
P = sorted(set(A) & set(B))
d = {p: B[p]["max_cross_track"] - A[p]["max_cross_track"] for p in P if B[p]["outcome"] != "planner-timeout"}
order = sorted(d, key=d.get)
pick = order[-4:][::-1] + order[:2]
lines = ["plan | LTN outcome, max_cross, overlap_s | LTNk outcome, max_cross, overlap_s"]
for p in P:
    a, b = A[p], B[p]
    lines.append(f"{p} | {a['outcome']}, {a['max_cross_track']:.3f}, {a.get('wall_overlap_s') or 0:.2f} | "
                 f"{b['outcome']}, {b['max_cross_track']:.3f}, {b.get('wall_overlap_s') or 0:.2f}")
open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(lines) + "\n")

MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
meta = yaml.safe_load(open(MAP + ".yaml")); img = np.asarray(Image.open(MAP + ".png").convert("L"))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
EXT = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
fig, axs = plt.subplots(2, 3, figsize=(16, 11))
for ax, p in zip(axs.flat, pick):
    za, zb = np.load(A[p]["track_local"]), np.load(B[p]["track_local"])
    plan = za["plan_poses"]
    ax.imshow(img, cmap="gray", extent=EXT, vmin=0, vmax=255)
    ax.plot(plan[:, 0], plan[:, 1], c="tab:green", lw=1.2, label="plan")
    for z, c, lab, r in ((za, "k", "LTN (tvlqr)", A[p]), (zb, "tab:orange", "LTNk (sideslip)", B[p])):
        t = z["track"]
        ax.plot(t[:, 1], t[:, 2], c=c, lw=1.6, label=f"{lab}: {r['outcome']}, max {r['max_cross_track']:.2f} m, "
                                                    f"contact {r.get('wall_overlap_s') or 0:.1f} s")
    allx = np.r_[plan[:, 0], za["track"][:, 1], zb["track"][:, 1]]; ally = np.r_[plan[:, 1], za["track"][:, 2], zb["track"][:, 2]]
    ax.set_xlim(allx.min() - 0.5, allx.max() + 0.5); ax.set_ylim(ally.min() - 0.5, ally.max() + 0.5)
    ax.set_aspect("equal"); ax.set_title(f"{p}: max cross-track {d[p]:+.2f} m with sideslip", fontsize=9)
    ax.legend(fontsize=7, loc="best")
fig.suptitle("Job 240, ours-lib full stack (truth, patches): the four plans where the sideslip model's max cross-track "
             "regressed most (top row, first of bottom) and the two largest gains", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tracks.png"), dpi=100); plt.close(fig)
print("wrote", OUT, pick)
