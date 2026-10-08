#!/usr/bin/env python3
"""Track D (#50): the speed-gated sideslip TVLQR (crab offset faded in up to |v_ref| = 0.3 m/s,
4395ad8) vs the un-gated one and plain TVLQR, closed-loop Gazebo bench, flat rl_corrector.world,
34 centred plans. tvlqr and un-gated tvlqr_slip come from the first (deterministic) bench run;
the gated arm from 2026-10-08_nojob_sideslip-gate-bench.
    .venv/bin/python figures/2026-10-08/d_sideslip_gate_bench/render.py
"""
import glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, yaml
from PIL import Image
from scipy.stats import binomtest

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(R, "tools"))
from compare_run import wall_contact  # noqa: E402

RD = os.path.join(R, "run_data")
UNGATED = os.path.join(RD, "2026-10-08_nojob_sideslip-bench/gazebo")
GATED = os.path.join(RD, "2026-10-08_nojob_sideslip-gate-bench/gazebo")
OUT = os.path.dirname(os.path.abspath(__file__))
START_M = 3.0   # the start transient: first metres along the plan
END_M = 1.0     # the end turn: last metres along the plan


def load(d, rename=None):
    out = {}
    for f in glob.glob(os.path.join(d, "**", "*__*.csv"), recursive=True):
        name, corr = os.path.basename(f)[:-4].split("__")
        corr = (rename or {}).get(corr, corr)
        a = np.genfromtxt(f, delimiter=",", names=True)
        wc = wall_contact(np.c_[a["k"] * 0.1, a["true_x"], a["true_y"], a["true_theta"]], 6)
        s = np.r_[0, np.cumsum(np.hypot(np.diff(a["plan_x"]), np.diff(a["plan_y"])))]
        e = np.abs(a["e_cross"])
        out[(name, corr)] = dict(
            a=a, s=s, max=float(np.nanmax(e)),
            start=float(np.nanmax(e[s <= START_M])),
            end=float(np.nanmax(e[s >= s[-1] - END_M])),
            mid=float(np.nanmax(e[(s > START_M) & (s < s[-1] - END_M)])) if np.any((s > START_M) & (s < s[-1] - END_M)) else np.nan,
            clear=wc["wall_min_clear"], contact=wc["wall_overlap_s"] > 0,
            final=float(np.hypot(a["true_x"][-1] - a["plan_x"][-1], a["true_y"][-1] - a["plan_y"][-1])))
    return out


B = load(UNGATED)
B.update(load(GATED, {"tvlqr_slip": "gated"}))
ARMS = ("tvlqr", "tvlqr_slip", "gated")
plans = sorted({k[0] for k in B if all((k[0], c) in B for c in ARMS)})
L = [f"n = {len(plans)} plans; arms: tvlqr (plain), tvlqr_slip (un-gated, k=-0.33), gated (v_gate 0.3 m/s)",
     f"start = max|e| over the first {START_M} m, end = over the last {END_M} m, mid = in between"]


def pair(key, a, b):
    x = np.array([B[(p, a)][key] for p in plans], float)
    y = np.array([B[(p, b)][key] for p in plans], float)
    d = y - x
    sgn = -1 if key == "clear" else 1
    nb, nw = int(np.sum(sgn * d < -1e-9)), int(np.sum(sgn * d > 1e-9))
    pv = binomtest(nb, nb + nw).pvalue if nb + nw else 1.0
    return x, y, d, nb, nw, pv


for a, b in (("tvlqr", "gated"), ("tvlqr_slip", "gated"), ("tvlqr", "tvlqr_slip")):
    L.append(f"\n{b} vs {a}:")
    for key in ("max", "start", "mid", "end", "final", "clear"):
        x, y, d, nb, nw, pv = pair(key, a, b)
        L.append(f"  {key:6s} {a} {np.nanmean(x):.3f}  {b} {np.nanmean(y):.3f}  delta mean {np.nanmean(d):+.3f}"
                 f" median {np.nanmedian(d):+.3f}  {b} better {nb}/{nb + nw}  p={pv:.2g}")
    co = np.array([B[(p, a)]["contact"] for p in plans]); cs = np.array([B[(p, b)]["contact"] for p in plans])
    L.append(f"  contact plans {a} {co.sum()}  {b} {cs.sum()}  (fixed {int(np.sum(co & ~cs))}, new {int(np.sum(~co & cs))})")
open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# --- 1. e_cross along the plan, all three arms, for the plans where gating matters most ---
COL = {"tvlqr": "k", "tvlqr_slip": "tab:orange", "gated": "tab:blue"}
dstart = {p: B[(p, "gated")]["start"] - B[(p, "tvlqr_slip")]["start"] for p in plans}
dend = {p: B[(p, "gated")]["end"] - B[(p, "tvlqr_slip")]["end"] for p in plans}
pick = [min(plans, key=dstart.get), min(plans, key=dend.get), max(plans, key=lambda p: B[(p, "gated")]["max"] - B[(p, "tvlqr_slip")]["max"])]
fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), gridspec_kw=dict(height_ratios=[3, 1.5]))
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
meta = yaml.safe_load(open(MAP + ".yaml")); img = np.asarray(Image.open(MAP + ".png").convert("L"))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
EXT = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
titles = ("largest start-transient gain", "largest end-turn gain", "largest max|e| loss vs un-gated")
for j, (p, t) in enumerate(zip(pick, titles)):
    ax, ae = axs[0, j], axs[1, j]
    g = B[(p, "tvlqr")]
    ax.imshow(img, cmap="gray", extent=EXT, vmin=0, vmax=255)
    ax.plot(g["a"]["plan_x"], g["a"]["plan_y"], c="tab:green", lw=1.2, label="plan")
    for corr in ARMS:
        r = B[(p, corr)]
        ax.plot(r["a"]["true_x"], r["a"]["true_y"], c=COL[corr], lw=1.6, label=corr)
        ae.plot(r["s"], r["a"]["e_cross"], c=COL[corr], lw=1.3)
    px, py = g["a"]["plan_x"], g["a"]["plan_y"]
    ax.set_xlim(px.min() - 1, px.max() + 1); ax.set_ylim(py.min() - 1, py.max() + 1); ax.set_aspect("equal")
    ax.set_title(f"{t}\n{p}: max|e| " + " / ".join(f"{B[(p, c)]['max']:.3f}" for c in ARMS), fontsize=9)
    ae.axhline(0, c="0.7", lw=.7); ae.axvspan(0, START_M, color="0.9"); ae.axvspan(g["s"][-1] - END_M, g["s"][-1], color="0.9")
    ae.set_xlabel("distance along plan [m]"); ae.set_ylabel("e_cross [m]"); ae.grid(alpha=.3)
axs[0, 0].legend(fontsize=8, loc="best")
fig.suptitle("Gazebo bench, flat: plain TVLQR (black), un-gated sideslip (orange), speed-gated sideslip (blue). "
             "Shaded: start and end windows", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tracks.png"), dpi=110); plt.close(fig)

# --- 2. per-plan scatter, gated vs un-gated ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, key in zip(axs, ("start", "mid", "end")):
    for other, col in (("tvlqr_slip", "tab:orange"), ("tvlqr", "k")):
        x, y, *_ = pair(key, other, "gated")
        ax.scatter(x, y, c=col, s=22, label=f"x = {other}")
    lim = np.nanmax([B[(p, c)][key] for p in plans for c in ARMS]) * 1.05
    ax.plot([0, lim], [0, lim], "k:", lw=1); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel(f"other arm, max|e| in '{key}' [m]"); ax.set_ylabel("gated [m]"); ax.grid(alpha=.3); ax.set_title(key)
axs[0].legend(fontsize=8)
fig.suptitle("below the diagonal = gated better", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "paired_scatter.png"), dpi=110); plt.close(fig)
print("wrote", OUT)
