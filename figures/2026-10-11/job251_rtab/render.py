"""Job 251 (2026-10-11): lidar-only rtabmap localization vs amcl, TVLQG and TVLQR.

Localizer pose for EVERY cell = map->odom (as recorded, 10 Hz) composed with the
EKF's odom->base_link (/odom/filtered), at the odom stamps. That is the pose the
corrector acts on, and it is defined the same way for amcl, rtab and truth.
Per run: max and median |localizer - truth|, excursion = max > 0.3 m; contact
(wall_overlap_s > 0) and arrival from the rows. Paired over (plan, seed): sign
tests on excursion / contact / arrival flips, Wilcoxon on max error.
Pre-registered rules: run_data/2026-10-11_job251_rtab-dev/README.md.
Usage: python render.py [JOB_DIR]  -> table.txt, job251.png
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon, binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-11_job251_rtab-dev")
CONFIG = os.path.join(R, "tools/jobs/c1/rtab.txt")
cells = [l.split()[0] for l in open(CONFIG) if l.strip() and l[0] != "#"]
# (cell, reference) pairs to test; the first is the pre-registered primary.
PAIRS = [("RQ", "Q0"), ("RQ", "RR"), ("Q0", "QR"), ("RQ", "T")]


def loc_error(z):
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
    return np.hypot(x - np.interp(st, T[:, 0], T[:, 1]), y - np.interp(st, T[:, 0], T[:, 2]))


runs = {}
for c in cells:
    for f in glob.glob(f"{JOB}/{c}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            pid = os.path.basename(r["plan_path"]).replace(".npz", "")
            tp = os.path.join(JOB, c, f"s{r['seed']}", os.path.basename(r.get("track_path") or ""))
            d = dict(ok=False, contact=r.get("wall_overlap_s", 0) > 0, arr=r["outcome"] == "arrived")
            if r["outcome"] not in ("planner-timeout", "stack-failed") and os.path.isfile(tp):
                e = loc_error(np.load(tp))
                if e is not None:
                    d.update(ok=True, emax=float(e.max()), emed=float(np.median(e)), exc=bool(e.max() > 0.3))
            runs[(c, pid, r["seed"])] = d

lines = []
def out(s=""):
    print(s); lines.append(s)

out(f"{'cell':5s} {'n':>4s} {'exc>0.3':>9s} {'exc>0.5':>7s} {'med|e|':>7s} {'p90 max|e|':>10s} {'contact':>8s} {'arrived':>8s}")
stats = {}
for c in cells:
    v = [d for (cc, _, _), d in runs.items() if cc == c and d["ok"]]
    if not v:
        out(f"{c:5s}    0"); continue
    stats[c] = v
    out(f"{c:5s} {len(v):4d} {sum(x['exc'] for x in v):4d} {np.mean([x['exc'] for x in v])*100:3.0f}% "
        f"{sum(x['emax'] > 0.5 for x in v):7d} {np.median([x['emed'] for x in v]):7.3f} "
        f"{np.percentile([x['emax'] for x in v], 90):10.3f} {sum(x['contact'] for x in v):8d} {sum(x['arr'] for x in v):8d}")

out()
out("paired (cell vs ref): fixed/new, sign-test p  [fixed = bad in ref, good in cell]")
for a, b in PAIRS:
    pr = [(runs[(a, p, s)], runs[(b, p, s)]) for (c, p, s) in runs
          if c == a and runs[(c, p, s)]["ok"] and runs.get((b, p, s), {}).get("ok")]
    if not pr:
        continue
    row = f"{a:3s} vs {b:3s} n {len(pr):3d}"
    for key, bad in (("exc", True), ("contact", True), ("arr", False)):
        fix = sum(y[key] == bad and x[key] != bad for x, y in pr)
        new = sum(x[key] == bad and y[key] != bad for x, y in pr)
        p = binomtest(fix, fix + new).pvalue if fix + new else 1.0
        row += f" | {key} {fix:3d}/{new:<3d} p {p:.3g}"
    dd = np.array([x["emax"] - y["emax"] for x, y in pr])
    row += f" | max|e| wilcoxon p {wilcoxon(dd).pvalue if np.any(dd) else 1:.3g} (median d {np.median(dd):+.3f})"
    out(row)
open(os.path.join(HERE, "table.txt"), "w").write("\n".join(lines) + "\n")

fig, ax = plt.subplots(figsize=(7, 3.5))
cs = list(stats)
ax.bar(np.arange(len(cs)) - 0.2, [np.mean([x["exc"] for x in stats[c]]) * 100 for c in cs], 0.4,
       label="localizer excursion runs (max |pose - truth| > 0.3 m)")
ax.bar(np.arange(len(cs)) + 0.2, [np.mean([x["contact"] for x in stats[c]]) * 100 for c in cs], 0.4,
       label="contact runs")
ax.set_xticks(range(len(cs)), cs); ax.set_ylabel("% of runs"); ax.legend(fontsize=8)
ax.set_title("Job 251: lidar-only rtabmap vs amcl, dev half")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "job251.png"), dpi=130)
