"""Hull clearance of each PLAN (its own poses scored with compare_run.wall_contact, the same body
rectangle and map the runs are scored on), vs. contact runs per plan in jobs 252/253/254.
Question: are contacts concentrated on plans that already graze a wall (corrector cannot help),
or on plans with room (corrector failed to hold the plan)?
Usage: python plan_hull.py -> stdout
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(R, "tools"))
import compare_run as cr
JOBS = {"252 dev": "run_data/2026-10-11_job252_rtab-lagfix-dev", "253 test": "run_data/2026-10-11_job253_rtab-test",
        "254 leaf": "run_data/2026-10-12_job254_doorleaf"}
BINS = [(-9, 0.0), (0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9)]


def local(p):
    return os.path.join(R, p.replace("/home/programmer/", ""))


def hull(path, map_yaml):
    P = np.load(local(path))["poses"]
    T = np.c_[np.arange(len(P)) * 0.1, P[:, 0], P[:, 1], P[:, 2]]
    return cr.wall_contact(T, local(map_yaml) if map_yaml else 6)["wall_min_clear"]


cache = {}
for job, d in JOBS.items():
    hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in glob.glob(os.path.join(R, d, "*/rows.s*.jsonl")):
        c = f.split("/")[-2]
        for l in open(f):
            r = json.loads(l)
            if not r.get("track_path"):
                continue
            if r["plan"] not in cache:
                cache[r["plan"]] = hull(r["plan_path"], r.get("map_yaml"))
            h = hits[r["plan"]][c]; h[0] += 1; h[1] += (r.get("wall_overlap_s") or 0) > 0
    cells = sorted({c for v in hits.values() for c in v})
    print(f"\n== job {job}: {len(hits)} plans; contact runs/runs by the plan's own hull clearance")
    print("hull clear [m]   plans  " + "  ".join(f"{c:>9s}" for c in cells))
    for lo, hi in BINS:
        ps = [p for p in hits if lo <= cache[p] < hi]
        print(f"[{lo:5.2f},{hi:5.2f})  {len(ps):5d}  " + "  ".join(
            f"{sum(hits[p][c][1] for p in ps):4d}/{sum(hits[p][c][0] for p in ps):<4d}" for c in cells))
    bad = sorted((cache[p], p) for p in hits if cache[p] < 0.05)
    print("plans with hull clearance < 0.05:", ", ".join(f"{p} {h:.2f}" for h, p in bad))


# Per contact: the plan's LOCAL hull clearance at the plan pose nearest the first-contact point
# (min over plan poses within 0.5 m of it). "Room" = the plan was >= 0.15 m clear there, so the
# robot had to leave the plan by > 0.15 m to touch: a tracking failure, not a plan that grazes.
def body(P):
    nl, nw = int(2 * cr.FP_HALF_LENGTH / cr.FP_SPACING) + 1, int(2 * cr.FP_HALF_WIDTH / cr.FP_SPACING) + 1
    lx, wy = np.linspace(-cr.FP_HALF_LENGTH, cr.FP_HALF_LENGTH, nl), np.linspace(-cr.FP_HALF_WIDTH, cr.FP_HALF_WIDTH, nw)
    b = np.vstack([np.c_[lx, np.full(nl, cr.FP_HALF_WIDTH)], np.c_[lx, np.full(nl, -cr.FP_HALF_WIDTH)],
                   np.c_[np.full(nw, cr.FP_HALF_LENGTH), wy], np.c_[np.full(nw, -cr.FP_HALF_LENGTH), wy]])
    c, s = np.cos(P[:, 2])[:, None], np.sin(P[:, 2])[:, None]
    return P[:, 0:1] + b[None, :, 0] * c - b[None, :, 1] * s, P[:, 1:2] + b[None, :, 0] * s + b[None, :, 1] * c


def per_pose_clear(path, map_yaml):
    P = np.load(local(path))["poses"]
    d, r, ox, oy = cr._signed_wall_dist(local(map_yaml) if map_yaml else 6)
    qx, qy = body(P)
    col = np.clip(((qx - ox) / r).astype(int), 0, d.shape[1] - 1)
    row = np.clip(((qy - oy) / r).astype(int), 0, d.shape[0] - 1)
    return P, d[row, col].min(axis=1)


ROOM = 0.15
pp = {}
print(f"\n== contact runs by the plan's LOCAL hull clearance near first contact (room = >= {ROOM} m)")
for job, d in JOBS.items():
    cnt = defaultdict(lambda: [0, 0, 0])
    for f in glob.glob(os.path.join(R, d, "*/rows.s*.jsonl")):
        c = f.split("/")[-2]
        for l in open(f):
            r = json.loads(l)
            if not r.get("track_path") or not (r.get("wall_overlap_s") or 0) > 0:
                continue
            if r["plan"] not in pp:
                pp[r["plan"]] = per_pose_clear(r["plan_path"], r.get("map_yaml"))
            P, cl = pp[r["plan"]]
            T = np.load(os.path.join(R, d, c, f"s{r['seed']}", os.path.basename(r["track_path"])))["track"]
            i = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1)
            dist = np.hypot(P[:, 0] - T[i, 1], P[:, 1] - T[i, 2])
            near = dist <= max(0.5, dist.min() + 0.05)
            loc = cl[near].min()
            cnt[c][0] += 1; cnt[c][1] += loc >= ROOM; cnt[c][2] += dist.min() > 0.3
    print(f"job {job}: cell contacts / with room (plan >= {ROOM} m clear there) / robot > 0.3 m from the plan at contact")
    for c in sorted(cnt):
        print(f"   {c:4s} {cnt[c][0]:4d} {cnt[c][1]:4d} {cnt[c][2]:4d}")
