#!/usr/bin/env python3
"""Job 249: does the lidar scan fit amcl's (wrong) pose or the true pose better?

Every recorded scan (compare_run --record-scan: base_link frame, 1440 beams) is
scored against the baked floor-6 map with amcl's own sensor model (likelihood
field, nav2_params.yaml: z_hit 0.9, z_rand 0.05, sigma_hit 0.1, max_dist 1.0,
180 beams, amcl's per-beam p^3 sum) at three poses:
  truth     Gazebo pose at the scan stamp
  amcl      /amcl_pose published for that scan (same stamp; else latest before)
  best      the best pose in a +-0.6 m / +-12 deg grid around truth
During an EXCURSION (|amcl - truth| > 0.3 m):
  score(truth) > score(amcl): the filter missed a better pose -> filter settings
  score(amcl) >= score(truth): the scan genuinely prefers the wrong pose -> sensor
  model / ambiguous geometry.
    python3 render.py [job_dir] -> table.txt, scan_vs_map.png, examples.png
"""
import glob, os, sys
import numpy as np, yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-11_job249_scan-rec")
Z_HIT, Z_RAND, SIG, MAXD, BEAMS, RMAX = 0.9, 0.05, 0.1, 1.0, 180, 100.0
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))[::-1]
res = meta["resolution"]; ox, oy = meta["origin"][:2]
occ = img == 0
DIST = np.minimum(distance_transform_edt(~occ) * res, MAXD)


def score(pts, x, y, th):
    """amcl likelihood_field: sum over beams of (z_hit*exp(-d^2/2s^2) + z_rand/zmax)^3.
    pts (n, 2) in base_link; pose arrays broadcast -> scores."""
    x, y, th = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(th)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    wx = x[:, None] + c * pts[:, 0] - s * pts[:, 1]; wy = y[:, None] + s * pts[:, 0] + c * pts[:, 1]
    d = map_coordinates(DIST, [(wy - oy) / res - 0.5, (wx - ox) / res - 0.5], order=1, mode="nearest", cval=MAXD)
    p = Z_HIT * np.exp(-d * d / (2 * SIG * SIG)) + Z_RAND / RMAX
    return (p ** 3).sum(1), np.median(d, 1)


DX = np.arange(-0.6, 0.601, 0.05); DTH = np.radians(np.arange(-12, 12.01, 1.5))
GX, GY, GT = [g.ravel() for g in np.meshgrid(DX, DX, DTH, indexing="ij")]

rows = []; examples = []
for f in sorted(glob.glob(f"{JOB}/*/s*/track_*.npz")):
    z = np.load(f)
    if "scan_ranges" not in z.files or len(z["amcl"]) < 3:
        continue
    T, AM, ST, RG = z["track"], z["amcl"], z["scan_t"], z["scan_ranges"].astype(float)
    a0, da, rmin, rmax = z["scan_meta"]
    ang = a0 + da * np.arange(RG.shape[1])
    tyaw = np.unwrap(T[:, 3])
    sel = np.linspace(0, RG.shape[1] - 1, BEAMS).astype(int)
    for k in range(len(ST)):
        stamp = ST[k, 1]
        if stamp < T[0, 0] or stamp > T[-1, 0]:
            continue
        r = RG[k, sel]; a = ang[sel]; ok = np.isfinite(r) & (r > rmin) & (r < min(rmax, RMAX))
        if ok.sum() < 30:
            continue
        pts = np.c_[r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])]
        tx, ty, tt = np.interp(stamp, T[:, 0], T[:, 1]), np.interp(stamp, T[:, 0], T[:, 2]), wrap(np.interp(stamp, T[:, 0], tyaw))
        j = np.where(np.abs(AM[:, 7] - stamp) < 1e-3)[0]
        exact = len(j) > 0
        if not exact:
            j = np.where(AM[:, 7] <= stamp)[0]
            if not len(j):
                continue
        j = j[-1]; ax_, ay_, at_ = AM[j, 1], AM[j, 2], AM[j, 3]
        err = np.hypot(ax_ - tx, ay_ - ty)
        st_, dt_ = score(pts, tx, ty, tt); sa, da_ = score(pts, ax_, ay_, at_)
        row = dict(f=f, k=k, err=err, yaw_err=abs(wrap(at_ - tt)), s_truth=st_[0], s_amcl=sa[0], d_truth=dt_[0], d_amcl=da_[0], exact=exact,
                   n=len(pts), tx=tx, ty=ty)
        if err > 0.3 or k % 5 == 0:   # grid search: all excursion scans + a 1-in-5 sample of the rest
            sg, _ = score(pts, tx + GX, ty + GY, tt + GT); i = int(np.argmax(sg))
            row.update(s_best=sg[i], best_dx=GX[i], best_dy=GY[i], best_dth=GT[i],
                       best_to_amcl=np.hypot(tx + GX[i] - ax_, ty + GY[i] - ay_))
        rows.append(row)
        if err > 0.3 and len(examples) < 400:
            examples.append((err, f, k, pts, (tx, ty, tt), (ax_, ay_, at_)))

L = []
def out(s=""):
    print(s); L.append(s)
E = np.array([r["err"] for r in rows]); exc = E > 0.3
sT = np.array([r["s_truth"] for r in rows]); sA = np.array([r["s_amcl"] for r in rows])
out(f"scans {len(rows)} in {len(set(r['f'] for r in rows))} runs; amcl pose matched by stamp for {100*np.mean([r['exact'] for r in rows]):.0f}%")
out(f"excursion scans (|amcl-truth| > 0.3 m): {exc.sum()} ({100*exc.mean():.1f}%)")
for name, m in [("normal (<0.15 m)", E < 0.15), ("0.15-0.3 m", (E >= 0.15) & (E <= 0.3)), ("excursion >0.3 m", exc)]:
    if not m.any(): continue
    rel = sA[m] / sT[m]
    out(f"{name:18s} n {m.sum():6d}: score(amcl)/score(truth) median {np.median(rel):.3f}; amcl pose scores >= truth in {100*np.mean(rel >= 1):.0f}%;"
        f" median scan-wall dist at truth {np.median(np.array([r['d_truth'] for r in rows])[m]):.3f} m, at amcl {np.median(np.array([r['d_amcl'] for r in rows])[m]):.3f} m")
B = [r for r in rows if "s_best" in r]
for name, sel_ in [("normal (<0.15 m)", lambda r: r["err"] < 0.15), ("excursion >0.3 m", lambda r: r["err"] > 0.3)]:
    b = [r for r in B if sel_(r)]
    if not b: continue
    off = np.array([np.hypot(r["best_dx"], r["best_dy"]) for r in b]); toa = np.array([r["best_to_amcl"] for r in b])
    gain = np.array([r["s_best"] / r["s_truth"] for r in b])
    out(f"best pose in a +-0.6 m grid around truth, {name}: n {len(b)}; offset from truth median {np.median(off):.3f} m (p90 {np.percentile(off,90):.3f});"
        f" nearer amcl's pose than truth in {100*np.mean(toa < off):.0f}%; score gain over truth median {np.median(gain):.3f}")
open(os.path.join(HERE, "table.txt"), "w").write("\n".join(L) + "\n")

# figure 1: score ratio vs error
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axs[0]
ax.scatter(np.clip(E, 0, 2), np.clip(sA / sT, 0, 2), s=2, alpha=0.3)
ax.axhline(1, color="k", lw=0.8); ax.axvline(0.3, color="r", lw=0.8, ls="--")
ax.set_xlabel("|amcl - truth| at the scan [m] (clipped 2)"); ax.set_ylabel("score(amcl pose) / score(truth pose)")
ax.set_title("above 1: the scan prefers amcl's pose")
ax = axs[1]
off = np.array([np.hypot(r["best_dx"], r["best_dy"]) for r in B]); eb = np.array([r["err"] for r in B])
ax.scatter(np.clip(eb, 0, 2), off, s=3, alpha=0.4)
ax.set_xlabel("|amcl - truth| [m]"); ax.set_ylabel("best-fit pose offset from truth [m] (grid +-0.6)")
ax.set_title("where the scan fits best, near truth")
fig.suptitle("Job 249: each recorded scan scored with amcl's likelihood field on the baked map")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "scan_vs_map.png"), dpi=100)

# figure 2: 6 example excursion scans (largest error per distinct run)
seen, ex = set(), []
for e in sorted(examples, key=lambda e: -e[0]):
    if e[1] in seen: continue
    seen.add(e[1]); ex.append(e)
    if len(ex) == 6: break
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for ax, (err, f, k, pts, tp, ap) in zip(axs.ravel(), ex):
    ax.imshow(np.where(occ, 0.2, np.where(img == 205, 0.85, 1.0)), cmap="gray", vmin=0, vmax=1, origin="lower",
              extent=[ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res], interpolation="nearest")
    for (x, y, th), col, lab in [(tp, "tab:blue", "scan at truth"), (ap, "tab:red", "scan at amcl")]:
        c, s = np.cos(th), np.sin(th)
        ax.plot(x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1], ".", ms=2.5, color=col, label=lab)
        ax.plot(x, y, "o", color=col, ms=6); ax.arrow(x, y, 0.5 * c, 0.5 * s, color=col, width=0.03)
    ax.set_xlim(tp[0] - 5, tp[0] + 5); ax.set_ylim(tp[1] - 5, tp[1] + 5); ax.set_aspect("equal")
    ax.set_title(f"{f.split('/')[-3]} {os.path.basename(f)[9:-4]} scan {k}: amcl off {err:.2f} m", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "examples.png"), dpi=90)
print("wrote figures")
