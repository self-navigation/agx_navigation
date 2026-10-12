"""Job 254: truth-localizer (T, TVLQG) contact runs over their leaf map, one per plan.
Per panel: leaf map, re-solved plan (dashed), truth track, red x at first contact,
body outline at first contact. Usage: python render_tracks.py [JOB_DIR] -> tracks.png
"""
import glob, json, os, sys
import numpy as np, yaml
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(R, "tools"))
import compare_run as cr
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-12_job254_doorleaf")
PL = os.path.join(R, "run_data/2026-10-12_nojob_doorleaf-plans")
CELL = "T"
seen, cases = set(), []
for f in sorted(glob.glob(f"{JOB}/{CELL}/rows.s*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if (r.get("wall_overlap_s") or 0) > 0 and r["plan"] not in seen and r.get("track_path"):
            seen.add(r["plan"]); cases.append(r)
cases.sort(key=lambda r: r["plan"])
n = len(cases); nc = 4; nr = (n + nc - 1) // nc
fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.8 * nr), squeeze=False)
for ax, r in zip(axs.flat, cases):
    yml = os.path.join(PL, "maps", os.path.basename(r["map_yaml"]))
    meta = yaml.safe_load(open(yml)); img = np.asarray(Image.open(os.path.join(PL, "maps", meta["image"])))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ax.imshow(img, cmap="gray", extent=[ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res], origin="upper", vmin=0, vmax=255)
    z = np.load(f"{PL}/plans/{r['plan']}.npz", allow_pickle=True)
    for lv in json.loads(str(z["leaves"])):
        ax.plot([lv["hinge"][0], lv["tip"][0]], [lv["hinge"][1], lv["tip"][1]], "r-", lw=2)
    P = z["poses"]; T = np.load(os.path.join(JOB, CELL, f"s{r['seed']}", os.path.basename(r["track_path"])))["track"]
    ax.plot(P[:, 0], P[:, 1], "k--", lw=1, label="plan")
    ax.plot(T[:, 1], T[:, 2], "C0", lw=1.2, label="truth track")
    i = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1)
    x, y, th = T[i, 1:4]; L, W = cr.FP_HALF_LENGTH, cr.FP_HALF_WIDTH
    c = np.array([[L, W], [L, -W], [-L, -W], [-L, W], [L, W]])
    ax.plot(x + c[:, 0] * np.cos(th) - c[:, 1] * np.sin(th), y + c[:, 0] * np.sin(th) + c[:, 1] * np.cos(th), "m-", lw=1)
    ax.plot(x, y, "rx", ms=9, mew=2)
    ax.set_xlim(x - 3, x + 3); ax.set_ylim(y - 2.5, y + 2.5)
    ax.set_title(f"{r['plan']} s{r['seed']}  {r['outcome']}, overlap {r['wall_overlap_s']:.1f}s", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axs.flat[n:]:
    ax.axis("off")
axs.flat[0].legend(fontsize=6, loc="best")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "tracks.png"), dpi=100)
print(n, "plans")
