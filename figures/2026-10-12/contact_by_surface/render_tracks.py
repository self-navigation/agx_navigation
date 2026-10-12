"""Contact runs with the terrain drawn: per panel the floor map, the plan (dashed), the truth track,
the patches (filled rectangles, labelled by profile; orange = steerable), red x at first contact.
Usage: python render_tracks.py JOB_DIR CELL [OUT_PNG] (default: job 253, T -> tracks_253_T.png)
"""
import glob, json, os, sys
import numpy as np, yaml
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path[:0] = [os.path.join(R, "src/agx_navigation/agx_planning"), os.path.join(R, "src/rudn-ordjo-building")]
from agx_planning.rl_corrector.terrain import along_path_terrain_sampler
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-11_job253_rtab-test")
CELL = sys.argv[2] if len(sys.argv) > 2 else "T"
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, f"tracks_{os.path.basename(JOB)[11:17]}_{CELL}.png")
COL = {"icy": "#9ecae1", "slippery": "#3182bd", "wet_tile": "#6baed6", "linoleum": "#d9c27a",
       "directional_x": "#08519c", "directional_y": "#08519c"}
meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
rows = []
for f in sorted(glob.glob(f"{JOB}/{CELL}/rows.s*.jsonl")):
    rows += [r for r in map(json.loads, open(f)) if r.get("track_path") and (r.get("wall_overlap_s") or 0) > 0]
rows.sort(key=lambda r: (r["plan"], r["seed"]))
n = len(rows); nc = 5; nr = (n + nc - 1) // nc
fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.1 * nr), squeeze=False)
for ax, r in zip(axs.flat, rows):
    P = np.load(os.path.join(R, r["plan_path"].replace("/home/programmer/", "")))["poses"]
    pats = along_path_terrain_sampler(P[:, :3])(np.random.default_rng(r["seed"]))
    T = np.load(os.path.join(JOB, CELL, f"s{r['seed']}", os.path.basename(r["track_path"])))["track"]
    i = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1)
    ax.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255)
    for pa in pats:
        c, s = np.cos(pa["yaw"]), np.sin(pa["yaw"]); w, l = pa["width"] / 2, pa["length"] / 2
        corners = np.array([[w, l], [-w, l], [-w, -l], [w, -l]]) @ np.array([[c, s], [-s, c]]) + [pa["x"], pa["y"]]
        prof = pa["profile"] if r.get("patch_mu") is None else "steerable"
        ax.add_patch(Polygon(corners, color=COL.get(prof, "orange"), alpha=0.6, lw=0))
        ax.text(pa["x"], pa["y"], prof[:4], fontsize=5, ha="center", clip_on=True)
    ax.plot(P[:, 0], P[:, 1], "k--", lw=0.8)
    ax.plot(T[:, 1], T[:, 2], "C3", lw=1)
    ax.plot(T[i, 1], T[i, 2], "rx", ms=8, mew=2)
    ax.set_xlim(T[i, 1] - 3, T[i, 1] + 3); ax.set_ylim(T[i, 2] - 2.5, T[i, 2] + 2.5)
    ax.set_title(f"{r['plan'][3:]} s{r['seed']} {r['outcome'][:6]}", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axs.flat[n:]:
    ax.axis("off")
fig.suptitle(f"{os.path.basename(JOB)} cell {CELL}: contact runs, patches drawn (blue/tan = at or below the steering knee)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT, dpi=90); print(n, "->", OUT)
