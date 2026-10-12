#!/usr/bin/env python3
"""Two exploratory checks on the mu 0.5 TVLQG runs (Fable ideas B1, B2; 2026-10-12). Post hoc, not pre-registered.

B1, step kick: is first contact preceded by a map->odom correction STEP sooner than chance? A step is a jump > STEP m
   between consecutive map_odom samples (10 Hz). For each contact run: the time from the last step to wall_first_t.
   Null: the same statistic at a random moving time in the same localizer's CLEAN runs (10 draws per run).
B2, yaw: the localizer's heading error |yaw_est - yaw_truth| (map_odom yaw + odom_filtered yaw vs track yaw), mean
   over the 1 s before contact, vs the mean over clean runs. Also e_heading as the corrector saw it (diag col 2).
usage: step_and_yaw.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__)); RD = os.path.join(HERE, "../../../run_data")
CELLS = {"amcl": ("2026-10-12_job255_steerable", "Q0"), "rtab": ("2026-10-12_job256_steerable-rtab", "RQ")}
STEP = 0.03
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 20, "axes.titlesize": 21, "axes.titleweight": "bold"})




SITE_R = 1.0
def sites_of(job, cell):
    xy = []
    for f in glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l); tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
            if os.path.isfile(tp) and (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None:
                T = np.load(tp)["track"]; i = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1); xy.append(T[i, 1:3])
    return np.asarray(xy)

res = {}
for loc, (job, cell) in CELLS.items():
    S = sites_of(job, cell); dt_site = []
    dt_hit, dt_null, yaw_hit, yaw_clean, eh_hit, eh_clean = [], [], [], [], [], []
    for f in glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l); tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
            if not os.path.isfile(tp): continue
            z = np.load(tp); MO, T, O, D, td = z["map_odom"], z["track"], z["odom_filtered"], z["diag"], z["diag_t"]
            if len(MO) < 5 or len(T) < 5 or len(O) < 5 or len(D) < 5: continue
            ts = MO[:, 4]
            jump = np.hypot(np.diff(MO[:, 1]), np.diff(MO[:, 2]))
            steps = ts[1:][jump > STEP]
            # heading error of the composed estimate: map_odom yaw + odom yaw (col 3 of odom_filtered) vs truth yaw
            st = O[:, 6]; m = (st >= max(T[0, 0], ts[0])) & (st <= min(T[-1, 0], ts[-1]))
            if m.sum() < 5: continue
            st = st[m]
            yaw_est = np.interp(st, ts, np.unwrap(MO[:, 3])) + np.unwrap(O[m, 3])
            yaw_tru = np.interp(st, T[:, 0], np.unwrap(T[:, 3]))
            dy = np.abs((yaw_est - yaw_tru + np.pi) % (2 * np.pi) - np.pi)
            mv = D[:, 4] > 0.1
            if (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None:
                tc = r["wall_first_t"]
                prev = steps[steps <= tc]
                dt_hit.append(tc - prev[-1] if len(prev) else np.inf)
                w = (st >= tc - 1) & (st <= tc)
                if w.any(): yaw_hit.append(dy[w].mean())
                w = mv & (td >= tc - 1) & (td <= tc)
                if w.any(): eh_hit.append(np.abs(D[w, 2]).mean())
            else:
                tm = td[mv]
                for tc in rng.choice(tm, size=min(10, len(tm)), replace=False) if len(tm) else []:
                    prev = steps[steps <= tc]
                    dt_null.append(tc - prev[-1] if len(prev) else np.inf)
                # same doorways: clean-run moments within SITE_R of any contact site (one draw per 0.5 s)
                near = np.hypot(T[:, 1][:, None] - S[:, 0], T[:, 2][:, None] - S[:, 1]).min(1) < SITE_R
                for tc in T[near, 0][::5]:
                    prev = steps[steps <= tc]
                    dt_site.append(tc - prev[-1] if len(prev) else np.inf)
                yaw_clean.append(np.median(dy))
                if mv.any(): eh_clean.append(np.abs(D[mv, 2]).mean())
    res[loc] = dict(dt_hit=np.array(dt_hit), dt_null=np.array(dt_null), dt_site=np.array(dt_site), yaw_hit=np.array(yaw_hit),
                    yaw_clean=np.array(yaw_clean), eh_hit=np.array(eh_hit), eh_clean=np.array(eh_clean))

lines = []
def out(s=""):
    print(s); lines.append(s)
for loc, d in res.items():
    h, n = d["dt_hit"], d["dt_null"]
    hf, nf = h[np.isfinite(h)], n[np.isfinite(n)]
    for w in (0.5, 1.0):
        out(f"{loc} B1: step (> {STEP * 100:.0f} cm) within {w} s before contact: {np.mean(h < w):.0%} of {len(h)} contacts"
            f" vs {np.mean(n < w):.0%} of {len(n)} random clean moments")
    sn = d["dt_site"]
    for w in (0.5, 1.0):
        out(f"{loc} B1 SAME DOORWAYS: step within {w} s: contacts {np.mean(h < w):.0%} vs clean runs passing a contact site {np.mean(sn < w):.0%} (n {len(sn)} moments)")
    out(f"{loc} B1: median time since last step: contact {np.median(hf):.2f} s, random {np.median(nf):.2f} s, MWU p {mannwhitneyu(hf, nf).pvalue:.2g}")
    out(f"{loc} B2: localizer heading error, 1 s before contact {np.degrees(np.median(d['yaw_hit'])):.1f} deg (n {len(d['yaw_hit'])})"
        f" vs clean-run median {np.degrees(np.median(d['yaw_clean'])):.1f} deg, MWU p {mannwhitneyu(d['yaw_hit'], d['yaw_clean']).pvalue:.2g}")
    out(f"{loc} B2: corrector |e_heading|, 1 s before contact {np.degrees(np.median(d['eh_hit'])):.1f} deg vs clean-run mean {np.degrees(np.median(d['eh_clean'])):.1f} deg")
open(os.path.join(HERE, "step_and_yaw.txt"), "w").write("\n".join(lines) + "\n")

fig, axs = plt.subplots(2, 1, figsize=(10.8, 16))
bins = np.linspace(0, 5, 21)
for ax, (loc, d) in zip(axs, res.items()):
    h, n = d["dt_hit"], d["dt_null"]
    ax.hist(np.clip(n[np.isfinite(n)], 0, 5), bins, density=True, color="0.7", label="random moment, clean runs")
    sn = d["dt_site"]
    ax.hist(np.clip(sn[np.isfinite(sn)], 0, 5), bins, density=True, histtype="step", lw=4, color="tab:blue", label="clean runs at the same doors")
    ax.hist(np.clip(h[np.isfinite(h)], 0, 5), bins, density=True, histtype="step", lw=4, color="tab:red", label="first wall contact")
    ax.set_title(f"{loc}: time since last pose jump > {STEP * 100:.0f} cm")
    ax.set_xlabel("seconds"); ax.set_ylabel("density"); ax.legend(fontsize=16)
fig.suptitle("Do wall hits follow a localizer jump?\n(TVLQG, mu 0.5; exploratory)", fontsize=23, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "step_timing.png"), dpi=100)
