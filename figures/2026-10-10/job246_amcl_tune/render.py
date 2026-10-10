"""Job 246: amcl tuning cells. Left: |amcl - truth| position error per cell
(truth interpolated at the amcl message stamp). Right: arrival / contact runs
per cell (102 runs = 34 plans x 3 seeds). Data: run_data/2026-10-10_job246_amcl-tune."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
D = os.path.join(R, "run_data/2026-10-10_job246_amcl-tune")
CELLS = [("A0", "base"), ("Aa", "alpha 0.5"), ("Ao", "omni"), ("Ag", "gate 5cm"),
         ("As", "sharp"), ("Aogs", "omni+gate+sharp")]

def amcl_err(c):
    e, per_run = [], []
    for f in sorted(glob.glob(f"{D}/{c}/s*/*.npz")):
        z = np.load(f)
        tr, am = z["track"], z["amcl"]
        if len(am) < 2 or len(tr) < 2:
            continue
        t = am[:, 7] if np.all(np.isfinite(am[:, 7])) and am[:, 7].max() > 0 else am[:, 0]
        m = (t >= tr[0, 0]) & (t <= tr[-1, 0])
        x = np.interp(t[m], tr[:, 0], tr[:, 1]); y = np.interp(t[m], tr[:, 0], tr[:, 2])
        d = np.hypot(am[m, 1] - x, am[m, 2] - y)
        e.append(d); per_run.append(np.median(d))
    return np.concatenate(e), np.array(per_run)

def rows(c):
    out = []
    for f in glob.glob(f"{D}/{c}/rows.s*.jsonl"):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out

if __name__ == "__main__":
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    stats = []
    for i, (c, lab) in enumerate(CELLS):
        e, pr = amcl_err(c)
        rs = rows(c)
        arr = sum(r.get("outcome") == "arrived" for r in rs)
        con = sum((r.get("wall_overlap_s") or 0) > 0 for r in rs)
        stats.append((c, lab, np.median(e), np.percentile(e, 90), np.median(pr), arr, con, len(rs)))
        a1.boxplot(e, positions=[i], widths=0.6, showfliers=False)
    a1.set_xticks(range(len(CELLS)), [f"{c}\n{l}" for c, l in CELLS], fontsize=8)
    a1.set_ylabel("|amcl − truth| position (m)"); a1.set_title("amcl position error, all samples")
    x = np.arange(len(CELLS))
    a2.bar(x - 0.2, [s[5] for s in stats], 0.4, label="arrived")
    a2.bar(x + 0.2, [s[6] for s in stats], 0.4, label="runs with wall contact")
    a2.set_xticks(x, [c for c, _ in CELLS]); a2.set_ylabel("runs of 102"); a2.legend(fontsize=8)
    a2.set_title("TVLQG under amcl, broad34c x 3 seeds")
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "amcl_tune.png"), dpi=120)
    for s in stats:
        print("%-5s %-16s err med %.3f p90 %.3f run-med %.3f | arrived %d contact %d / %d" % s)
