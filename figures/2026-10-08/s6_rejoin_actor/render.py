#!/usr/bin/env python3
"""S6 (#46): learning curves for TD3 / TD3+BC / eps-TD3+BC / BC, cost vs the PMP
teacher, and the beta(s) map. Reads run_data/2026-10-08_nojob_s6-rejoin-actor.
    .venv/bin/python figures/2026-10-08/s6_rejoin_actor/render.py
"""
import glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from scipy.stats import binomtest

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RUN = os.path.join(R, "run_data/2026-10-08_nojob_s6-rejoin-actor")
OUT = os.path.dirname(os.path.abspath(__file__))
KINDS = ["td3", "td3bc", "eps_td3bc", "bc"]
COL = dict(td3="tab:red", td3bc="tab:orange", eps_td3bc="tab:blue", bc="tab:green")
LAB = dict(td3="TD3", td3bc="TD3+BC (const beta)", eps_td3bc="eps-TD3+BC", bc="BC only")


def done(kind):
    """run dirs that finished (eval_final written); a seed still training is skipped"""
    return [d for d in sorted(glob.glob(os.path.join(RUN, f"{kind}_s*")))
            if os.path.exists(os.path.join(d, "eval_final.npz"))]


def parts(z, st, who=""):
    """(j_track, 10 pin^2, contact) per sample; contact is what the cost has left over"""
    p = "teacher_" if who else ""
    c, j, pin = z[f"{p}{st}_cost"], z[f"{p}{st}_j_track"], 10 * z[f"{p}{st}_pin"] ** 2
    return j, pin, c - j - pin


def curves(kind):
    out = {}
    for d in done(kind):
        rows = [json.loads(l) for l in open(os.path.join(d, "curve.jsonl"))]
        out[os.path.basename(d)] = [r for r in rows if not r.get("final")]
    return out


C = {k: curves(k) for k in KINDS}
lines = []

# --- 1. learning curves --------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
for ax, key, title in zip(axs, ("test", "test_nominal", "val"),
                          ("held-out plans, random chi (G3)", "held-out plans, nominal chi",
                           "val (training plans), random chi")):
    for k in KINDS:
        for i, (name, rows) in enumerate(C[k].items()):
            x = [r["env_samples"] / 1e3 for r in rows]
            y = [r[key]["ratio"] for r in rows]
            ax.plot(x, y, c=COL[k], lw=1.3, alpha=.85, label=LAB[k] if i == 0 else None)
    ax.axhline(1.0, c="k", lw=1, label="PMP teacher")
    ax.axhline(1.1, c="k", ls="--", lw=.8, label="G3 bound (1.10)")
    ax.set_yscale("log"); ax.set_ylim(0.6, 5); ax.grid(alpha=.3, which="both")
    ax.set_xlabel("env samples [k]"); ax.set_ylabel("mean cost / teacher mean cost")
    ax.set_title(title, fontsize=10)
axs[0].legend(fontsize=8)
fig.suptitle("S6 re-join actor: cost relative to the PMP teacher (one line per seed)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "learning_curves.png"), dpi=110); plt.close(fig)

# --- 2. final table + per-sample comparison --------------------------------------
lines.append("final checkpoint (policy_final), per seed: ratio = mean cost / teacher mean cost")
lines.append(f"{'run':16s} {'test':>7s} {'test_nom':>8s} {'val':>7s} {'val_nom':>8s} {'win':>5s} "
             f"{'contact':>7s} {'pin_med':>7s}")
fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
for ax, k in zip(axs, KINDS):
    for d in done(k):
        rows = [json.loads(l) for l in open(os.path.join(d, "curve.jsonl"))]
        f = rows[-1]
        z = np.load(os.path.join(d, "eval_final.npz"))
        sc, tcost = z["test_cost"], z["teacher_test_cost"]
        nb, nw = int(np.sum(sc < tcost)), int(np.sum(sc > tcost))
        lines.append(f"{os.path.basename(d):16s} {f['test']['ratio']:7.3f} {f['test_nominal']['ratio']:8.3f} "
                     f"{f['val']['ratio']:7.3f} {f['val_nominal']['ratio']:8.3f} {f['test']['win']:5.2f} "
                     f"{f['test']['contact']:7.3f} {f['test']['pin_med']:7.3f}  "
                     f"sign test student<teacher {nb}/{nb + nw} p={binomtest(nb, nb + nw).pvalue:.2g}")
        ax.loglog(tcost + 1e-3, sc + 1e-3, ".", ms=2, alpha=.4)
    ax.plot([1e-3, 1e3], [1e-3, 1e3], "k:", lw=1)
    ax.set_title(LAB[k], fontsize=10); ax.set_xlabel("teacher cost (test sample)"); ax.grid(alpha=.3)
axs[0].set_ylabel("student cost (same sample, same chi)")
fig.suptitle("Held-out plans, random chi: per-sample cost, student vs PMP teacher (below the diagonal = student better)",
             fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "per_sample_vs_teacher.png"), dpi=110); plt.close(fig)

# --- 3. where the cost goes: tracking, landing, wall contact ----------------------
# The re-join teacher (cost B, effort) never queries the wall field, so its cost
# carries wall contact the students can learn to avoid. This panel separates that
# from the slip question S6 exists for.
lines.append("")
lines.append("cost components on test (random chi), seed means: track / 10 pin^2 / contact; "
             "ratio without contact; ratio and win rate on samples where NEITHER touches a wall")
fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
names, comp, clean_ratio = [], [], []
for k in KINDS:
    P, T, cr, cw, nc = [], [], [], [], []
    for d in done(k):
        z = np.load(os.path.join(d, "eval_final.npz"))
        s3, t3 = parts(z, "test"), parts(z, "test", "teacher")
        P.append([x.mean() for x in s3]); T.append([x.mean() for x in t3])
        sn, tn = s3[0] + s3[1], t3[0] + t3[1]
        cl = (s3[2] < 1e-6) & (t3[2] < 1e-6)
        cr.append(sn[cl].mean() / tn[cl].mean()); cw.append(np.mean(sn[cl] < tn[cl]))
        nc.append(sn.mean() / tn.mean())
    P, T = np.mean(P, 0), np.mean(T, 0)
    names.append(LAB[k]); comp.append(P); clean_ratio.append(cr)
    lines.append(f"  {k:10s} n_seeds={len(cr)}  student {P[0]:.2f}/{P[1]:.2f}/{P[2]:.2f}  teacher {T[0]:.2f}/{T[1]:.2f}/{T[2]:.2f}"
                 f"  no-contact ratio {np.mean(nc):.3f}  both-clean ratio {np.mean(cr):.3f} win {np.mean(cw):.3f}")
names.append("PMP teacher"); comp.append(T)
comp = np.array(comp)
x = np.arange(len(names))
for j, (lab, c) in enumerate(zip(("tracking J", "10 pin^2 (landing miss)", "10 x wall-contact s"),
                                  ("tab:gray", "tab:purple", "tab:red"))):
    axs[0].bar(x, comp[:, j], bottom=comp[:, :j].sum(1), color=c, label=lab)
axs[0].set_xticks(x, names, fontsize=8); axs[0].set_ylabel("mean cost per re-join (test, random chi)")
axs[0].set_title("where the cost goes: the teacher's deficit is wall contact, not slip", fontsize=10)
axs[0].legend(fontsize=8)
for i, (k, cr) in enumerate(zip(KINDS, clean_ratio)):
    axs[1].plot([i] * len(cr), cr, "o", c=COL[k])
axs[1].axhline(1.0, c="k", lw=1); axs[1].axhline(1.1, c="k", ls="--", lw=.8)
axs[1].set_xticks(range(len(KINDS)), [LAB[k] for k in KINDS], fontsize=8)
axs[1].set_ylabel("mean cost / teacher (one dot per seed)")
axs[1].set_title("samples where neither touches a wall: the slip question", fontsize=10)
axs[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cost_components.png"), dpi=110); plt.close(fig)

# --- 4. beta map ----------------------------------------------------------------
d = done("eps_td3bc")[0]
z = np.load(os.path.join(d, "beta_map.npz"))
a = json.load(open(os.path.join(d, "args.json")))
m = z["m"] > 0
s, beta, chi, eps = z["s"][m], z["beta"][m], z["chi"][m], z["eps_hat"][m]
plan_om = np.abs(s[:, 11::2]).max(1)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
sc = axs[0].scatter(chi, plan_om, c=beta, s=2, cmap="viridis", vmin=0, vmax=1)
axs[0].set_xlabel("plant chi"); axs[0].set_ylabel("max |plan omega| over the window [rad/s]")
axs[0].set_title("beta(s) over the last 50k training states", fontsize=10)
fig.colorbar(sc, ax=axs[0], label="beta")
axs[1].hist(beta, bins=50, color="tab:blue")
axs[1].set_xlabel("beta"); axs[1].set_ylabel("states"); axs[1].set_yscale("log")
axs[1].set_title(f"eps0 = {a['eps0']:.3f}: beta < 0.1 on {np.mean(beta < 0.1):.0%} of labelled states", fontsize=10)
bins = np.linspace(1.3, 2.0, 8)
idx = np.digitize(chi, bins)
axs[2].plot([chi[idx == i].mean() for i in range(1, len(bins))],
            [np.mean(beta[idx == i] < 0.1) for i in range(1, len(bins))], "o-")
axs[2].set_xlabel("plant chi"); axs[2].set_ylabel("fraction with beta < 0.1")
axs[2].set_title("where the actor is released from the teacher", fontsize=10); axs[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "beta_map.png"), dpi=110); plt.close(fig)
lines.append("")
lines.append(f"beta map ({os.path.basename(d)}): eps0 {a['eps0']:.4f}; beta<0.1 on {np.mean(beta < 0.1):.3f} "
             f"of labelled states; eps_hat median {np.median(eps):.4f} p90 {np.percentile(eps, 90):.3f}")
for lo, hi in ((0, 0.3), (0.3, 0.8), (0.8, 10)):
    w = (plan_om >= lo) & (plan_om < hi)
    lines.append(f"  max|plan omega| in [{lo}, {hi}): n={w.sum():6d}  beta<0.1 frac {np.mean(beta[w] < 0.1):.3f}")
open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
