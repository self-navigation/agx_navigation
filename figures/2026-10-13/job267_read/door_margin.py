"""Job 267 re-read (Fable brainstorm, 2026-10-13; post hoc, descriptive): are the speed contacts the PLAN's margin or a slide?
Per (plan, door), using door_overlay.py's door detection and frame:
  plan margin = min over plan points within 0.6 m of the door of (EDT clearance - hull half-width 0.2925)
  per run: lateral deviation of the truth track from the plan at the door line (x = 0 in the door frame; + = inward,
  - = outward side of the turn), and whether the run's first contact is at this door.
Prints: contact rate per plan-door binned by plan margin (all 12 cells pooled); per arm pair the per-plan-door mean
deviation compared by Wilcoxon over plan-doors; the same deviation for the old tyre (job 255, slow plans) for scale.
usage: door_margin.py > door_margin.txt"""
import glob, json, os, sys, collections, importlib.util
import numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.argv = [sys.argv[0], "--out=/dev/null/x"]          # import door_overlay's helpers without its plotting side effects
src = open(os.path.join(R, "figures/2026-10-12/door_overlay/door_overlay.py")).read().split("\nNR = ")[0]
g = {"__file__": os.path.join(R, "figures/2026-10-12/door_overlay/door_overlay.py")}; exec(src, g)
doors, to_frame, clearance, HW, RD = g["doors"], g["to_frame"], g["clearance"], g["HW"], g["RD"]


def plan_margin(pp, o):
    P = np.load(os.path.join(R, pp.replace("/home/programmer/", "")))["poses"][:, :2]
    near = P[np.hypot(*(P - o).T) < 0.6]
    return float(clearance(near).min() - HW) if len(near) else np.nan, P


def y_at_door(q):
    i = np.where(np.diff(np.sign(q[:, 0])) > 0)[0]
    if not len(i): return np.nan
    i = i[0]; a, b = q[i], q[i + 1]
    return float(a[1] + (b[1] - a[1]) * (-a[0]) / (b[0] - a[0]))


def runs(spec):
    for one in spec.split("+"):
        job, cell = one.rsplit(":", 1)
        for f in glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl"):
            for l in open(f):
                if not l.strip(): continue
                r = json.loads(l)
                tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
                if os.path.isfile(tp): yield r, np.load(tp)["track"]


def per_door(spec):
    """{(plan, door idx): [(seed, deviation, contact_here, plan_margin)]}"""
    out = collections.defaultdict(list)
    for r, T in runs(spec):
        if len(T) < 5: continue
        D = doors(r["plan_path"])
        hit = (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None
        ic = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1) if hit else None
        own = int(np.argmin([np.hypot(*to_frame(T[ic:ic + 1, 1:3], o, h, m)[0]) for o, h, m, *_ in D])) if hit and D else None
        for di, (o, h, m, *_) in enumerate(D):
            pm, P = plan_margin(r["plan_path"], o)
            dev = y_at_door(to_frame(T[:, 1:3], o, h, m)) - y_at_door(to_frame(P, o, h, m))
            here = hit and di == own and np.hypot(*to_frame(T[ic:ic + 1, 1:3], o, h, m)[0]) < 1.0
            out[(os.path.basename(r["plan"]), di)].append((r["seed"], dev, here, pm))
    return out


J = "2026-10-13_job267_v090-grid"
CELLS = ["QR", "Q0", "PQ", "FQ", "RR", "RQ", "PR", "FR", "TR", "T", "PT", "FT"]
data = {c: per_door(f"{J}/test:{c}+{J}/dev:{c}") for c in CELLS}

print("1. Contact at a door vs the PLAN's hull margin there (all 12 cells pooled; unit = run x door)")
bins = [(-1, 0.0), (0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9)]
cnt = collections.Counter(); hits = collections.Counter(); pdoors = collections.defaultdict(set)
for c, d in data.items():
    for k, v in d.items():
        for seed, dev, here, pm in v:
            b = next(i for i, (lo, hi) in enumerate(bins) if lo <= pm < hi) if np.isfinite(pm) else None
            if b is None: continue
            cnt[b] += 1; hits[b] += here; pdoors[b].add(k)
for i, (lo, hi) in enumerate(bins):
    print(f"  plan margin [{lo:5.2f}, {hi:4.2f}) m: plan-doors {len(pdoors[i]):3d}  passages {cnt[i]:5d}  door contacts {hits[i]:4d}  rate {hits[i] / max(cnt[i], 1):.3f}")
tot = sum(hits.values()); low = hits[0] + hits[1] + hits[2]
print(f"  door contacts at plan-doors with margin < 0.10 m: {low} of {tot}")

print("\n2. Deviation from the plan at the door line (m; + = inward, - = outward), per localizer and arm")
print("   per plan-door mean over seeds; median over plan-doors [IQR]; Wilcoxon vs TVLQR over plan-doors")
for loc, (base, *others) in {"amcl": ["QR", "Q0", "PQ", "FQ"], "rtab": ["RR", "RQ", "PR", "FR"], "truth": ["TR", "T", "PT", "FT"]}.items():
    def m(c): return {k: np.nanmean([x[1] for x in v]) for k, v in data[c].items()}
    B = m(base)
    print(f"  {loc}: {base} median {np.nanmedian(list(B.values())):+.3f} [{np.nanpercentile(list(B.values()), 25):+.3f}, {np.nanpercentile(list(B.values()), 75):+.3f}]")
    for c in others:
        A = m(c); ks = [k for k in B if k in A and np.isfinite(A[k]) and np.isfinite(B[k])]
        d = np.array([A[k] - B[k] for k in ks])
        print(f"     {c}: median {np.nanmedian([A[k] for k in ks]):+.3f}; paired diff median {np.median(d):+.4f} m over {len(ks)} plan-doors, Wilcoxon p {wilcoxon(d).pvalue:.3g}")

print("\n3. Scale: the same deviation on the OLD tyre (job 255, slow plans, mu 0.5), truth")
for c, lab in [("TR", "TVLQR"), ("T", "TVLQG")]:
    d = per_door(f"2026-10-12_job255_steerable:{c}")
    v = [np.nanmean([x[1] for x in vv]) for vv in d.values()]
    pm = [vv[0][3] for vv in d.values()]
    print(f"  {lab}: deviation median {np.nanmedian(v):+.3f} [{np.nanpercentile(v, 25):+.3f}, {np.nanpercentile(v, 75):+.3f}] over {len(v)} plan-doors; plan margin median {np.nanmedian(pm):.3f}")
pm7 = [vv[0][3] for vv in data["TR"].values()]
print(f"  job 267 plan margin median {np.nanmedian(pm7):.3f} over {len(pm7)} plan-doors")
