"""Job 268 registered read (rules in run_data/2026-10-13_job268_door-deviation/README.md, written before the run).
Primary: per PLAN, mean signed door-line deviation (+ = inward) over its doors and seeds, using door_margin.py's
per_door unchanged; FIXED k minus TVLQR, Wilcoxon over plans, two-sided, per localizer. Secondary (descriptive):
contact runs, door contacts by plan-margin bin, arrival, miss rate. usage: read268.py > read268.txt"""
import os, sys, json, glob, collections
import numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.argv = [sys.argv[0]]
P = os.path.join(R, "figures/2026-10-13/job267_read/door_margin.py")
g = {"__file__": P}; exec(open(P).read().split('\nJ = "2026')[0], g)
J = "2026-10-13_job268_door-deviation"
data = {c: g["per_door"](f"{J}/test:{c}+{J}/dev:{c}") for c in ["QR", "FQ", "RR", "FR", "TR", "FT"]}


def per_plan(d):
    acc = collections.defaultdict(list)
    for (plan, di), v in d.items():
        acc[plan] += [x[1] for x in v if np.isfinite(x[1])]
    return {p: np.mean(v) for p, v in acc.items() if v}


print("PRIMARY: fixed k minus TVLQR, per-plan mean door deviation (m, + = inward), Wilcoxon over plans, two-sided")
for loc, (b, f) in {"amcl": ("QR", "FQ"), "rtab": ("RR", "FR"), "truth": ("TR", "FT")}.items():
    B, F = per_plan(data[b]), per_plan(data[f]); ks = sorted(set(B) & set(F))
    d = np.array([F[k] - B[k] for k in ks])
    print(f"  {loc:5s}: plans {len(ks)} | TVLQR median {np.median([B[k] for k in ks]):+.4f} | fixed median {np.median([F[k] for k in ks]):+.4f} | "
          f"paired shift median {np.median(d):+.4f} mean {d.mean():+.4f} | inward on {int((d > 0).sum())}/{len(d)} plans | p {wilcoxon(d).pvalue:.3g}")

print("\nSECONDARY (descriptive): contact runs, arrival, miss > 0.5 m")
for c in data:
    rs = [json.loads(l) for h in ("test", "dev") for f in glob.glob(f"{g['RD']}/{J}/{h}/{c}/rows.s*.jsonl") for l in open(f) if l.strip()]
    fe = np.array([r["final_err"] for r in rs if r.get("final_err") is not None])
    print(f"  {c}: n {len(rs)} contact {sum((r.get('wall_overlap_s') or 0) > 0 for r in rs)} arrived {sum(r['outcome'] == 'arrived' for r in rs)} miss {int((fe > 0.5).sum())}")
print("\nSECONDARY: door contacts by the plan's hull margin at the door (6 cells pooled)")
bins = [(-1, 0.0), (0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9)]
cnt = collections.Counter(); hits = collections.Counter()
for d in data.values():
    for v in d.values():
        for seed, dev, here, pm in v:
            if not np.isfinite(pm): continue
            b = next(i for i, (lo, hi) in enumerate(bins) if lo <= pm < hi); cnt[b] += 1; hits[b] += here
for i, (lo, hi) in enumerate(bins):
    print(f"  [{lo:5.2f}, {hi:4.2f}) m: passages {cnt[i]:5d} door contacts {hits[i]:4d} rate {hits[i] / max(cnt[i], 1):.3f}")
