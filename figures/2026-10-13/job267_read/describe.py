"""Job 267 descriptive read (README items): per cell contacts / arrival / miss / median final_err; the sensitivity
check without f6_corridor_0016; wheel saturation (|cmd| >= 19 rad/s) vs contact; the job 265 slow twins on the same
63 plans (cross-job, descriptive). usage: describe.py > describe.txt"""
import glob, json, os, collections
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
J7 = f"{R}/run_data/2026-10-13_job267_v090-grid"
J5 = glob.glob(f"{R}/run_data/2026-10-12_job265*")[0]
CELLS = ["QR", "Q0", "PQ", "FQ", "RR", "RQ", "PR", "FR", "TR", "T", "PT", "FT"]


def rows(job, cell):
    out = []
    for h in ("test", "dev"):
        for f in glob.glob(f"{job}/{h}/{cell}/rows.s*.jsonl"):
            for l in open(f):
                if l.strip():
                    r = json.loads(l); r["_dir"] = f"{job}/{h}/{cell}/s{r['seed']}"; out.append(r)
    return out


def sat(r):
    tp = f"{r['_dir']}/{os.path.basename(r.get('track_path') or 'x')}"
    if not os.path.isfile(tp): return np.nan
    z = np.load(tp); w = np.abs(z["wheel_cmd"])
    return float((w.max(1) >= 19.0).mean()) if len(w) else np.nan


fast = {os.path.basename(r["plan"]) for r in rows(J7, "QR")}
print(f"plans {len(fast)}")
print("cell  n  contact  arrived  miss>0.5  med_ef | no f6_corridor_0016: contact | job265 same plans: contact miss")
for c in CELLS:
    rs = rows(J7, c)
    con = sum((r.get("wall_overlap_s") or 0) > 0 for r in rs)
    arr = sum(r["outcome"] == "arrived" for r in rs)
    fe = np.array([r["final_err"] for r in rs if r.get("final_err") is not None])
    miss = int((fe > 0.5).sum())
    con2 = sum((r.get("wall_overlap_s") or 0) > 0 for r in rs if os.path.basename(r["plan"]) != "f6_corridor_0016")
    old = [r for r in rows(J5, c) if os.path.basename(r["plan"]) in fast]
    ocon = sum((r.get("wall_overlap_s") or 0) > 0 for r in old)
    ofe = np.array([r["final_err"] for r in old if r.get("final_err") is not None])
    print(f"{c:3s} {len(rs):4d} {con:5d} {arr:7d} {miss:7d} {np.median(fe):8.3f} | {con2:5d} | {ocon}/{len(old)} {int((ofe>0.5).sum())}/{len(ofe)}")
print("\nwheel saturation: mean fraction of commands with max|w| >= 19 rad/s, contact runs vs clean runs")
for c in CELLS:
    rs = rows(J7, c)
    s = np.array([sat(r) for r in rs]); k = np.array([(r.get("wall_overlap_s") or 0) > 0 for r in rs])
    print(f"{c:3s} contact {np.nanmean(s[k]) if k.any() else float('nan'):.3f} (n {k.sum()})  clean {np.nanmean(s[~k]):.3f}")
print("\ncontact runs per plan (all 12 cells)")
pc = collections.Counter()
for c in CELLS:
    for r in rows(J7, c):
        if (r.get("wall_overlap_s") or 0) > 0: pc[os.path.basename(r["plan"])] += 1
for p, n in pc.most_common(): print(f"  {p} {n}")
