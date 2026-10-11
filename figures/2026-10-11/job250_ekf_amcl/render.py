"""Job 250 (2026-10-11): EKF skid covariance / IMU accel and amcl alphas, vs Q0.

Per run: max and median |amcl - truth| (amcl pose matched to truth by its stamp),
excursion = max > 0.3 m; contact (wall_overlap_s > 0) and arrival from the rows.
Paired over (plan, seed) vs Q0: sign tests on excursion and contact flips,
Wilcoxon on max error. Pre-registered rules: run_data/.../README.md.
Usage: python render.py [JOB_DIR]  -> table.txt, job250.png
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon, binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-11_job250_ekf-amcl")
cells = [l.split()[0] for l in open(os.path.join(R, "tools/jobs/c1/ekf_amcl.txt")) if l.strip() and l[0] != "#"]

runs = {}
for c in cells:
    for f in glob.glob(f"{JOB}/{c}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            pid = os.path.basename(r["plan_path"]).replace(".npz", "")
            tp = os.path.join(JOB, c, f"s{r['seed']}", os.path.basename(r.get("track_path") or ""))
            d = dict(ok=False, contact=r.get("wall_overlap_s", 0) > 0, arr=r["outcome"] == "arrived")
            if r["outcome"] not in ("planner-timeout", "stack-failed") and os.path.isfile(tp):
                z = np.load(tp); T, A = z["track"], z["amcl"]
                if len(T) > 2 and len(A) > 2:
                    st = A[:, 7]; m = (st >= T[0, 0]) & (st <= T[-1, 0])
                    e = np.hypot(A[m, 1] - np.interp(st[m], T[:, 0], T[:, 1]), A[m, 2] - np.interp(st[m], T[:, 0], T[:, 2]))
                    if len(e):
                        d.update(ok=True, emax=float(e.max()), emed=float(np.median(e)))
            runs[(c, pid, r["seed"])] = d

lines = []
def out(s=""):
    print(s); lines.append(s)

out(f"{'cell':6s} {'n':>4s} {'exc>0.3':>8s} {'exc>0.5':>8s} {'med|e|':>7s} {'contact':>8s} {'arrived':>8s} | vs Q0: exc fix/new p | contact fix/new p | max|e| wilcoxon p")
stats = {}
for c in cells:
    ks = [(p, s) for (cc, p, s) in runs if cc == c and runs[(cc, p, s)]["ok"]]
    v = [runs[(c, p, s)] for p, s in ks]
    if not v: continue
    exc = np.array([x["emax"] > 0.3 for x in v]); stats[c] = (exc.mean(), len(v), np.mean([x["contact"] for x in v]))
    row = f"{c:6s} {len(v):4d} {exc.sum():4d} {exc.mean()*100:3.0f}% {sum(x['emax'] > 0.5 for x in v):7d} {np.median([x['emed'] for x in v]):7.3f} {sum(x['contact'] for x in v):8d} {sum(x['arr'] for x in v):8d}"
    if c != "Q0":
        pr = [(runs[(c, p, s)], runs[("Q0", p, s)]) for p, s in ks if runs.get(("Q0", p, s), {}).get("ok")]
        def flips(key):
            fix = sum(b[key] and not a[key] for a, b in pr); new = sum(a[key] and not b[key] for a, b in pr)
            return fix, new, binomtest(fix, fix + new).pvalue if fix + new else 1.0
        for a, b in pr: a["exc"], b["exc"] = a["emax"] > 0.3, b["emax"] > 0.3
        f1, n1, p1 = flips("exc"); f2, n2, p2 = flips("contact")
        dd = np.array([a["emax"] - b["emax"] for a, b in pr])
        row += f" | {f1:3d}/{n1:<3d} p {p1:.3f} | {f2:3d}/{n2:<3d} p {p2:.3f} | {wilcoxon(dd).pvalue if np.any(dd) else 1:.3g} (median d {np.median(dd):+.3f})"
    out(row)
open(os.path.join(HERE, "table.txt"), "w").write("\n".join(lines) + "\n")

fig, ax = plt.subplots(figsize=(8, 3.5))
cs = list(stats)
ax.bar(np.arange(len(cs)) - 0.2, [stats[c][0] * 100 for c in cs], 0.4, label="excursion runs (max |amcl-truth| > 0.3 m)")
ax.bar(np.arange(len(cs)) + 0.2, [stats[c][2] * 100 for c in cs], 0.4, label="contact runs")
ax.set_xticks(range(len(cs)), cs); ax.set_ylabel("% of runs"); ax.legend(fontsize=8)
ax.set_title("Job 250: EKF / amcl motion input, dev half, TVLQG")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "job250.png"), dpi=130)
