"""At each job-255 contact site (first-contact point of every Q0/QR/T/TR contact run), the plan's
LOCAL hull clearance (min over plan poses within 0.6 m of the site), original v3 plan vs re-solved
sets. The plan-wide minimum is set by doorways (~0.11-0.15 m, unchangeable), so it is the wrong
statistic; this asks whether the re-solve moved the route away from where the robot actually hit.
Usage: python contact_site_clearance.py SET_DIR [SET_DIR ...] -> stdout
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(R, "tools"))
import compare_run as cr
JOB = os.path.join(R, "run_data/2026-10-12_job255_steerable")
BASE = os.path.join(R, "run_data/2026-10-10_nojob_plan-set-v3/plans")


def body(P):
    nl, nw = int(2 * cr.FP_HALF_LENGTH / cr.FP_SPACING) + 1, int(2 * cr.FP_HALF_WIDTH / cr.FP_SPACING) + 1
    lx, wy = np.linspace(-cr.FP_HALF_LENGTH, cr.FP_HALF_LENGTH, nl), np.linspace(-cr.FP_HALF_WIDTH, cr.FP_HALF_WIDTH, nw)
    b = np.vstack([np.c_[lx, np.full(nl, cr.FP_HALF_WIDTH)], np.c_[lx, np.full(nl, -cr.FP_HALF_WIDTH)],
                   np.c_[np.full(nw, cr.FP_HALF_LENGTH), wy], np.c_[np.full(nw, -cr.FP_HALF_LENGTH), wy]])
    c, s = np.cos(P[:, 2])[:, None], np.sin(P[:, 2])[:, None]
    return P[:, 0:1] + b[None, :, 0] * c - b[None, :, 1] * s, P[:, 1:2] + b[None, :, 0] * s + b[None, :, 1] * c


def clear(P):
    d, r, ox, oy = cr._signed_wall_dist(6)
    qx, qy = body(P)
    return d[np.clip(((qy - oy) / r).astype(int), 0, d.shape[0] - 1),
             np.clip(((qx - ox) / r).astype(int), 0, d.shape[1] - 1)].min(axis=1)


sites = []
for f in glob.glob(f"{JOB}/*/rows.s*.jsonl"):
    c = f.split("/")[-2]
    for l in open(f):
        r = json.loads(l)
        if r.get("track_path") and (r.get("wall_overlap_s") or 0) > 0:
            T = np.load(os.path.join(JOB, c, f"s{r['seed']}", os.path.basename(r["track_path"])))["track"]
            i = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1)
            sites.append((c, r["plan"], T[i, 1], T[i, 2]))
sets = [("base", BASE)] + [(os.path.basename(s)[24:], os.path.join(s, "plans")) for s in sys.argv[1:]]
cache = {}


def local_clear(sd, plan, x, y):
    if (sd, plan) not in cache:
        f = os.path.join(sd, plan + ".npz")
        cache[(sd, plan)] = None if not os.path.exists(f) else (lambda P: (P, clear(P)))(np.load(f)["poses"])
    if cache[(sd, plan)] is None:
        return None
    P, cl = cache[(sd, plan)]
    near = np.hypot(P[:, 0] - x, P[:, 1] - y) <= 0.6
    return cl[near].min() if near.any() else None


print(f"{len(sites)} contact sites in job 255")
print("set        sites(solved)  median local clearance  sites < 0.20 m   per-site mean change vs base")
base = {s: local_clear(BASE, s[1], s[2], s[3]) for s in sites}
for name, sd in sets:
    v = [(s, local_clear(sd, s[1], s[2], s[3])) for s in sites]
    v = [(s, x) for s, x in v if x is not None and base[s] is not None]
    a = np.array([x for _, x in v]); b = np.array([base[s] for s, _ in v])
    print(f"{name:10s} {len(v):5d}          {np.median(a):.3f} (base {np.median(b):.3f})     {(a < 0.2).sum():3d} (base {(b < 0.2).sum()})"
          f"      {np.mean(a - b):+.3f}")
