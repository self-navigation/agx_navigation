#!/usr/bin/env python3
"""Is the localizer's error tied to PLACES? And which contact sites are hard even with a perfect pose?

1. Bias map. Per run, the error vector e = (map->odom o odom->base_link) - truth at every odom_filtered stamp
   (same composition as job251_rtab/render.py loc_error). Per run, e is averaged inside each GRID x GRID cell of
   the truth position. Per cell, over runs: n_runs, the mean of the run means (the BIAS, |b|), and the agreement
   = |mean of run means| / mean |run mean|. Agreement is 1 when every run is off in the same direction and about
   1/sqrt(n) for random directions. A spot with a large |b| and high agreement is a place-tied localization bias.
   Runs are weighted equally, so a robot standing still for a long time does not dominate a cell.
2. Contact sites, per controller (mu 0.5 jobs only; stock patches confound contact with no-steer patches). The truth position at
   wall_first_t, clustered on a SITE grid. Per site and localizer: contact runs / runs whose truth track passes
   within PASS_R of the site. A site that truth also hits often is hard for the tracking itself, not for
   localization.

usage: bias_map.py   (paths are fixed below; output goes next to this file)
"""
import glob, json, os, collections
import numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
RD = os.path.join(R, "run_data")
GRID, SITE, PASS_R = 0.5, 1.0, 0.3
LOC = {  # localizer -> (job dir, cell)
    "amcl": [("2026-10-11_job253_rtab-test", "Q0"), ("2026-10-11_job253_rtab-test", "QR"),
             ("2026-10-12_job255_steerable", "Q0"), ("2026-10-12_job255_steerable", "QR")],
    "rtab": [("2026-10-11_job253_rtab-test", "RQ"), ("2026-10-11_job253_rtab-test", "RR"),
             ("2026-10-12_job256_steerable-rtab", "RQ"), ("2026-10-12_job256_steerable-rtab", "RR")],
    "truth": [("2026-10-12_job255_steerable", "T"), ("2026-10-12_job255_steerable", "TR")],
}
TVLQG = {"Q0", "RQ", "T"}  # the rest (QR, RR, TR) are TVLQR
MU05 = {"2026-10-12_job255_steerable", "2026-10-12_job256_steerable-rtab"}


def rows(job, cell):
    for f in sorted(glob.glob(os.path.join(RD, job, cell, "rows.s*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                tp = os.path.join(RD, job, cell, f"s{r['seed']}", os.path.basename(r.get("track_path") or ""))
                if r.get("track_path") and os.path.isfile(tp):
                    yield r, tp


def err_vec(z):
    T, MO, O = z["track"], z["map_odom"], z["odom_filtered"]
    if len(T) < 3 or len(MO) < 3 or len(O) < 3:
        return None
    st = O[:, 6]
    m = (st >= max(T[0, 0], MO[0, 4])) & (st <= min(T[-1, 0], MO[-1, 4]))
    if m.sum() < 3:
        return None
    st, ox, oy = st[m], O[m, 1], O[m, 2]
    mx, my = np.interp(st, MO[:, 4], MO[:, 1]), np.interp(st, MO[:, 4], MO[:, 2])
    myaw = np.interp(st, MO[:, 4], np.unwrap(MO[:, 3]))
    x = mx + np.cos(myaw) * ox - np.sin(myaw) * oy
    y = my + np.sin(myaw) * ox + np.cos(myaw) * oy
    tx, ty = np.interp(st, T[:, 0], T[:, 1]), np.interp(st, T[:, 0], T[:, 2])
    return tx, ty, x - tx, y - ty


meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))
res, (ox0, oy0, _) = meta["resolution"], meta["origin"]
ext = [ox0, ox0 + img.shape[1] * res, oy0, oy0 + img.shape[0] * res]

cell_runs = {k: collections.defaultdict(list) for k in ("amcl", "rtab")}   # loc -> cell -> [run-mean vec]
contacts, tracks = [], []   # (loc, job, site, x, y) ; (loc, job, xy array)
for loc, srcs in LOC.items():
    for job, cell in srcs:
        for r, tp in rows(job, cell):
            z = np.load(tp)
            if job in MU05:
                tracks.append((loc, cell in TVLQG, z["track"][:, 1:3]))
                if (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None:
                    T = z["track"]; i = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1)
                    contacts.append((loc, cell in TVLQG, T[i, 1], T[i, 2]))
            if loc == "truth":
                continue
            ev = err_vec(z)
            if ev is None:
                continue
            tx, ty, ex, ey = ev
            key = np.floor(np.c_[tx, ty] / GRID).astype(int)
            for k in {tuple(a) for a in key}:
                m = (key[:, 0] == k[0]) & (key[:, 1] == k[1])
                cell_runs[loc][k].append((ex[m].mean(), ey[m].mean(), hash(r["plan"]) % 10**9))

lines = []
def out(s=""):
    print(s); lines.append(s)

stats = {}
for loc in ("amcl", "rtab"):
    S = []
    for k, v in cell_runs[loc].items():
        v = np.asarray(v); npl = len(set(v[:, 2]))
        v = v[:, :2]
        if len(v) < 5:
            continue
        b = v.mean(0); agree = np.linalg.norm(b) / np.linalg.norm(v, axis=1).mean()
        S.append(((k[0] + .5) * GRID, (k[1] + .5) * GRID, b[0], b[1], np.linalg.norm(b), agree, len(v), npl))
    stats[loc] = S = np.asarray(S)
    big = S[(S[:, 4] > 0.10) & (S[:, 5] > 0.7) & (S[:, 7] >= 3)]
    out(f"{loc}: {len(S)} cells with >=5 runs | |bias| median {np.median(S[:, 4]):.3f} p90 {np.percentile(S[:, 4], 90):.3f} max {S[:, 4].max():.3f} m"
        f" | agreement median {np.median(S[:, 5]):.2f} | cells with |bias|>0.10 m AND agreement>0.7 AND >=3 distinct plans: {len(big)}")
    for c in big[np.argsort(-big[:, 4])][:12]:
        out(f"    ({c[0]:6.2f},{c[1]:6.2f})  bias ({c[2]:+.3f},{c[3]:+.3f}) |b| {c[4]:.3f}  agree {c[5]:.2f}  runs {int(c[6])} plans {int(c[7])}")

# contact sites: cluster by SITE grid; pass counts per localizer
out(f"\ncontact sites (mu 0.5 jobs 255/256; site = {SITE} m cell of the first-contact truth position):")
def bias_at(loc, x, y):
    S = stats[loc]; d = np.hypot(S[:, 0] - x, S[:, 1] - y); i = d.argmin()
    return S[i, 4] if d[i] < GRID else float("nan")
sites = collections.Counter((int(np.floor(x / SITE)), int(np.floor(y / SITE))) for _, _, x, y in contacts)
for q, qname in ((True, "TVLQG (T, Q0, RQ)"), (False, "TVLQR (TR, QR, RR)")):
    out(f"\n{qname}: hits/passes per site; |b| = localizer bias at the site centre")
    out(f"{'site centre':>16s}  {'truth':>9s} {'amcl':>9s} {'rtab':>9s}  amcl|b| rtab|b|")
    for s_, tot in sites.most_common():
        if tot < 4:
            continue
        cx, cy = (s_[0] + .5) * SITE, (s_[1] + .5) * SITE
        hits = collections.Counter(l for l, qq, x, y in contacts if qq == q and (int(np.floor(x / SITE)), int(np.floor(y / SITE))) == s_)
        passes = collections.Counter(l for l, qq, xy in tracks if qq == q and np.hypot(xy[:, 0] - cx, xy[:, 1] - cy).min() < SITE / 2 + PASS_R)
        f = lambda l: f"{hits[l]:3d}/{passes[l]:<4d}"
        out(f"({cx:6.1f},{cy:6.1f})  {f('truth'):>9s} {f('amcl'):>9s} {f('rtab'):>9s}  {bias_at('amcl', cx, cy):.3f}   {bias_at('rtab', cx, cy):.3f}")
open(os.path.join(HERE, "bias_map.txt"), "w").write("\n".join(lines) + "\n")

fig, axs = plt.subplots(1, 2, figsize=(18, 8))
for ax, loc in zip(axs, ("amcl", "rtab")):
    S = stats[loc]
    ax.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255)
    sc = ax.scatter(S[:, 0], S[:, 1], c=S[:, 4], s=14, cmap="magma_r", vmin=0, vmax=0.25, marker="s", alpha=.8)
    good = S[:, 5] > 0.7
    ax.quiver(S[good, 0], S[good, 1], S[good, 2], S[good, 3], angles="xy", scale_units="xy", scale=0.1, width=0.002, color="tab:blue")
    cx = np.asarray([(x, y) for l, q, x, y in contacts if l == loc and q] or np.zeros((0, 2)))
    tcx = np.asarray([(x, y) for l, q, x, y in contacts if l == "truth" and q] or np.zeros((0, 2)))
    if len(cx): ax.plot(cx[:, 0], cx[:, 1], "x", color="tab:green", ms=6, label=f"{loc} TVLQG first contact (mu 0.5)")
    if len(tcx): ax.plot(tcx[:, 0], tcx[:, 1], "o", mfc="none", color="tab:red", ms=8, label="truth TVLQG first contact")
    ax.set_xlim(S[:, 0].min() - 2, S[:, 0].max() + 2); ax.set_ylim(S[:, 1].min() - 2, S[:, 1].max() + 2)
    ax.set_title(f"{loc}: localization bias per {GRID} m cell (colour |bias|, arrows x10 where runs agree > 0.7)")
    ax.legend(loc="lower right", fontsize=8); ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.03, label="|bias| m")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "bias_map.png"), dpi=130)
