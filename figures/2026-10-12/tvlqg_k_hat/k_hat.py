#!/usr/bin/env python3
"""Does a real localizer shrink TVLQG's sideslip estimate? (Fable lead, 2026-10-12)

The TVLQG cells of jobs 255 (Q0 amcl, T truth) and 256 (RQ rtab) on mu 0.5. The diag columns are
CorrectionDiagnostics.FIELDS (runtime_corrector/tvlqr.py): k_hat 16, k_bar 18 (the k the gains are linearized at),
nis_pose 20, gated 21. Per run: the mean k_hat and k_bar over valid ticks while moving (v_ref > 0.1), and the mean
k_bar in the 3 s before first contact. Paired over (plan, seed): per-run mean k_bar, localizer vs truth. Gazebo's
plant k is -0.33.
"""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__)); RD = os.path.join(HERE, "../../../run_data")
CELLS = {"truth": ("2026-10-12_job255_steerable", "T"), "amcl": ("2026-10-12_job255_steerable", "Q0"),
         "rtab": ("2026-10-12_job256_steerable-rtab", "RQ")}
runs = {}
for loc, (job, cell) in CELLS.items():
    for f in glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l); tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
            if not os.path.isfile(tp): continue
            z = np.load(tp); D, t = z["diag"], z["diag_t"]
            if len(D) < 10: continue
            m = (D[:, 15] > 0) & (D[:, 4] > 0.1)
            if m.sum() < 10: continue
            d = dict(k_hat=D[m, 16].mean(), k_bar=D[m, 18].mean(), nis=np.nanmedian(D[m, 20]), gated=D[m, 21].mean(),
                     contact=(r.get("wall_overlap_s") or 0) > 0, kpre=np.nan)
            if d["contact"] and r.get("wall_first_t") is not None:
                w = m & (t >= r["wall_first_t"] - 3) & (t <= r["wall_first_t"])
                if w.any(): d["kpre"] = D[w, 18].mean()
            runs[(loc, os.path.basename(r["plan"]), r["seed"])] = d
for loc in CELLS:
    v = [d for k, d in runs.items() if k[0] == loc]
    kp = [d["kpre"] for d in v if not np.isnan(d["kpre"])]
    print(f"{loc:5s} n {len(v):3d} | mean k_hat {np.mean([d['k_hat'] for d in v]):+.3f} | mean k_bar {np.mean([d['k_bar'] for d in v]):+.3f}"
          f" (p10 {np.percentile([d['k_bar'] for d in v], 10):+.3f}, p90 {np.percentile([d['k_bar'] for d in v], 90):+.3f})"
          f" | median nis {np.median([d['nis'] for d in v]):.2f} | gated frac {np.mean([d['gated'] for d in v]):.3f}"
          f" | k_bar 3 s before contact {np.mean(kp) if kp else float('nan'):+.3f} (n {len(kp)})")
for loc in ("amcl", "rtab"):
    ks = [k[1:] for k in runs if k[0] == loc and ("truth",) + k[1:] in runs]
    a = np.array([runs[(loc,) + k]["k_bar"] for k in ks]); b = np.array([runs[("truth",) + k]["k_bar"] for k in ks])
    print(f"paired {loc} vs truth k_bar n {len(ks)}: median diff {np.median(a - b):+.3f}, {loc} weaker (closer to 0) in {np.sum(a > b)}/{len(ks)}, wilcoxon p {wilcoxon(a, b).pvalue:.2g}")
    hit = [runs[(loc,) + k]["k_bar"] for k in ks if runs[(loc,) + k]["contact"]]
    ok = [runs[(loc,) + k]["k_bar"] for k in ks if not runs[(loc,) + k]["contact"]]
    print(f"   {loc} k_bar contact runs {np.mean(hit):+.3f} (n {len(hit)}) vs clean {np.mean(ok):+.3f} (n {len(ok)}), MWU p {mannwhitneyu(hit, ok).pvalue:.2g}")

# Did the surface switch ever fire? Level 0 keeps k_bar = k_plan = 0, which is plain TVLQR.
print("\nsurface switch (k_bar leaves k_plan = 0): fraction of runs that ever activate, and time of activation")
for loc, (job, cell) in CELLS.items():
    act, tact, hit_na, n_na = [], [], 0, 0
    for f in glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l); tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
            if not os.path.isfile(tp): continue
            z = np.load(tp); D, t = z["diag"], z["diag_t"]
            if len(D) < 10: continue
            on = np.abs(D[:, 18]) > 1e-9
            act.append(on.any())
            mv = np.flatnonzero(D[:, 4] > 0.1)
            if on.any() and len(mv): tact.append(t[np.argmax(on)] - t[mv[0]])
            if not on.any():
                n_na += 1; hit_na += (r.get("wall_overlap_s") or 0) > 0
    print(f"{loc:5s} activated {np.mean(act):.0%} of {len(act)} | median time to activate {np.median(tact):.1f} s after moving"
          f" | never-activated runs: {n_na}, of which contact {hit_na}")

print("\ncontacts: was the switch already active at first contact?")
for loc, (job, cell) in CELLS.items():
    before = after = 0
    for f in glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l); tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
            if not os.path.isfile(tp) or not (r.get("wall_overlap_s") or 0) > 0 or r.get("wall_first_t") is None: continue
            z = np.load(tp); D, t = z["diag"], z["diag_t"]
            on = np.abs(D[:, 18]) > 1e-9
            ta = t[np.argmax(on)] if on.any() else np.inf
            before += r["wall_first_t"] < ta; after += r["wall_first_t"] >= ta
    print(f"{loc:5s} contacts BEFORE activation (plain TVLQR at that moment) {before}, after {after}")
