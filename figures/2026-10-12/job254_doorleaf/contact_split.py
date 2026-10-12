"""Job 254: split each contact run's wall overlap into leaf contact vs original-wall contact.
A body point counts as on a leaf if it is inside the leaf map's wall AND not inside floor_6's wall.
Also: per-plan contact counts across cells vs the plan's min centre-to-leaf distance.
Usage: python contact_split.py [JOB_DIR] -> contact_split.txt (stdout)
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(R, "tools"))
import compare_run as cr
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-12_job254_doorleaf")
PL = os.path.join(R, "run_data/2026-10-12_nojob_doorleaf-plans")
CELLS = ["Q0", "QR", "RQ", "RR", "T", "TR"]


def body_pts(T):
    nl, nw = int(2 * cr.FP_HALF_LENGTH / cr.FP_SPACING) + 1, int(2 * cr.FP_HALF_WIDTH / cr.FP_SPACING) + 1
    lx, wy = np.linspace(-cr.FP_HALF_LENGTH, cr.FP_HALF_LENGTH, nl), np.linspace(-cr.FP_HALF_WIDTH, cr.FP_HALF_WIDTH, nw)
    b = np.vstack([np.c_[lx, np.full(nl, cr.FP_HALF_WIDTH)], np.c_[lx, np.full(nl, -cr.FP_HALF_WIDTH)],
                   np.c_[np.full(nw, cr.FP_HALF_LENGTH), wy], np.c_[np.full(nw, -cr.FP_HALF_LENGTH), wy]])
    c, s = np.cos(T[:, 3])[:, None], np.sin(T[:, 3])[:, None]
    return T[:, 1:2] + b[None, :, 0] * c - b[None, :, 1] * s, T[:, 2:3] + b[None, :, 0] * s + b[None, :, 1] * c


def inside(floor, qx, qy):
    d, r, ox, oy = cr._signed_wall_dist(floor)
    col = np.clip(((qx - ox) / r).astype(int), 0, d.shape[1] - 1)
    row = np.clip(((qy - oy) / r).astype(int), 0, d.shape[0] - 1)
    return d[row, col] < 0


def seg_dist(p, a, b):
    a, b = np.asarray(a), np.asarray(b); ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


plan_min = {}
for f in glob.glob(f"{PL}/plans/*.npz"):
    z = np.load(f, allow_pickle=True)
    P = z["poses"][:, :2]; lv = json.loads(str(z["leaves"]))
    plan_min[os.path.basename(f)[:-4]] = min((seg_dist(P, l["hinge"], l["tip"]).min() for l in lv), default=9)

kind = {}; per_plan = {}
for c in CELLS:
    for f in glob.glob(f"{JOB}/{c}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if not r.get("track_path"):
                continue
            hit = (r.get("wall_overlap_s") or 0) > 0
            per_plan.setdefault(r["plan"], {}).setdefault(c, []).append(hit)
            if not hit:
                continue
            T = np.load(os.path.join(JOB, c, f"s{r['seed']}", os.path.basename(r["track_path"])))["track"]
            qx, qy = body_pts(T)
            lm = os.path.join(PL, "maps", os.path.basename(r["map_yaml"]))
            on_leafmap = inside(lm, qx, qy); on_wall = inside(6, qx, qy)
            leaf = (on_leafmap & ~on_wall).any(axis=1); wall = on_wall.any(axis=1)
            dt = np.diff(T[:, 0], append=T[-1, 0])
            kind.setdefault(c, []).append((r["plan"], r["seed"], float(dt[leaf].sum()), float(dt[wall].sum())))

print("cell  contact_runs  leaf_only  wall_only  both")
for c in CELLS:
    k = kind.get(c, [])
    lo = sum(1 for _, _, a, b in k if a > 0 and b == 0); wo = sum(1 for _, _, a, b in k if b > 0 and a == 0)
    print(f"{c:4s}  {len(k):12d}  {lo:9d}  {wo:9d}  {len(k)-lo-wo:4d}")
print("\nplans with any contact: plan  min_centre_to_leaf  hits per cell (of runs)")
for p in sorted(per_plan, key=lambda p: plan_min.get(p, 9)):
    h = per_plan[p]; tot = sum(sum(v) for v in h.values())
    if tot:
        print(f"{p:24s} {plan_min.get(p, 9):5.2f}  " + "  ".join(f"{c}:{sum(h.get(c, []))}/{len(h.get(c, []))}" for c in CELLS))

# The plan's own hull clearance: its poses scored like a track, on the map it was solved on,
# against the base plan's on floor_6. A plan with ~0 hull clearance cannot be held off the wall.
BASE = os.path.join(R, "run_data/2026-10-10_nojob_plan-set-v3/plans")


def hull_clear(P, floor):
    T = np.c_[np.arange(len(P)) * 0.05, P[:, 0], P[:, 1], P[:, 2]]
    return cr.wall_contact(T, floor)["wall_min_clear"]


print("\nplan hull clearance [m]: plan  leaf-plan on leaf map | base plan on floor_6 | contact runs (all cells)")
rows_hc = []
for p in sorted(plan_min):
    z = np.load(f"{PL}/plans/{p}.npz", allow_pickle=True)
    b = os.path.join(BASE, str(z["base_plan"]) + ".npz")
    hb = hull_clear(np.load(b)["poses"], 6) if os.path.exists(b) else float("nan")
    hl = hull_clear(z["poses"], os.path.join(PL, "maps", os.path.basename(str(z["map_yaml"]))))
    n = sum(sum(v) for v in per_plan.get(p, {}).values()); m = sum(len(v) for v in per_plan.get(p, {}).values())
    rows_hc.append((hl, hb, n, m)); print(f"{p:24s} {hl:6.3f} | {hb:6.3f} | {n}/{m}")
a = np.array(rows_hc)
for lo, hi in [(-1, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 9)]:
    s = (a[:, 0] >= lo) & (a[:, 0] < hi)
    print(f"leaf-plan hull clearance [{lo:.2f},{hi:.2f}): {s.sum():2d} plans, contact {a[s,2].sum():.0f}/{a[s,3].sum():.0f}")
