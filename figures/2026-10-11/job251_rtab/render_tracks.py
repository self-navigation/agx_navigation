"""Job 251: tracks of the runs where rtab+TVLQG (RQ) touched a wall and amcl+TVLQG (Q0)
did not, same plan and seed. Per panel: the plan, RQ truth, RQ localizer pose
(map->odom composed with /odom/filtered), Q0 truth; a red x at RQ's first contact.
Usage: python render_tracks.py [JOB_DIR] -> tracks.png
"""
import glob, json, os, sys
import numpy as np, yaml
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-11_job251_rtab-dev")
PLANS = os.path.join(R, "run_data/2026-10-10_nojob_plan-set-v3/plans")
meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]


def rows(c):
    out = {}
    for f in glob.glob(f"{JOB}/{c}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            out[(r["plan"], r["seed"])] = r
    return out


def loc_pose(z):
    T, MO, O = z["track"], z["map_odom"], z["odom_filtered"]
    st = O[:, 6]
    m = (st >= max(T[0, 0], MO[0, 4])) & (st <= min(T[-1, 0], MO[-1, 4]))
    st, x0, y0 = st[m], O[m, 1], O[m, 2]
    mx, my = np.interp(st, MO[:, 4], MO[:, 1]), np.interp(st, MO[:, 4], MO[:, 2])
    myaw = np.interp(st, MO[:, 4], np.unwrap(MO[:, 3]))
    return mx + np.cos(myaw) * x0 - np.sin(myaw) * y0, my + np.sin(myaw) * x0 + np.cos(myaw) * y0


def track(c, r):
    return np.load(os.path.join(JOB, c, f"s{r['seed']}", os.path.basename(r["track_path"])))


A, B = rows("RQ"), rows("Q0")
cases = sorted(k for k in A if k in B and A[k].get("wall_overlap_s", 0) > 0
               and not B[k].get("wall_overlap_s", 0) > 0 and A[k].get("track_path"))
print(len(cases), "cases:", cases)
cases = cases[:12]
n = len(cases); nc = 4; nr = (n + nc - 1) // nc
fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 3.6 * nr), squeeze=False)
for ax, k in zip(axs.flat, cases):
    p = np.load(os.path.join(PLANS, k[0] + ".npz"))["poses"]
    za, zb = track("RQ", A[k]), track("Q0", B[k])
    Ta, Tb = za["track"], zb["track"]
    lx, ly = loc_pose(za)
    ax.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255)
    ax.plot(p[:, 0], p[:, 1], "k--", lw=1, label="plan")
    ax.plot(Tb[:, 1], Tb[:, 2], "C2", lw=1, label="Q0 truth")
    ax.plot(Ta[:, 1], Ta[:, 2], "C0", lw=1.2, label="RQ truth")
    ax.plot(lx, ly, "C1", lw=0.8, label="RQ localizer")
    i = np.searchsorted(Ta[:, 0], A[k]["wall_first_t"])
    i = min(i, len(Ta) - 1)
    ax.plot(Ta[i, 1], Ta[i, 2], "rx", ms=9, mew=2)
    allx = np.r_[p[:, 0], Ta[:, 1]]; ally = np.r_[p[:, 1], Ta[:, 2]]
    ax.set_xlim(allx.min() - 1, allx.max() + 1); ax.set_ylim(ally.min() - 1, ally.max() + 1)
    ax.set_title(f"{k[0]} s{k[1]}  RQ {A[k]['outcome']}, overlap {A[k]['wall_overlap_s']:.1f}s", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axs.flat[n:]:
    ax.axis("off")
axs.flat[0].legend(fontsize=6, loc="best")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "tracks.png"), dpi=110)
