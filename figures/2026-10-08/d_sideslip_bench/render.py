#!/usr/bin/env python3
"""Track D (#50): does a sideslip-aware TVLQR (v_lat = k v omega, k = -0.33 s) hold the
door turns that plain TVLQR runs wide? Closed-loop Gazebo bench (compare_correctors,
flat rl_corrector.world) on the 34 centred plans, tvlqr vs tvlqr_slip in one process
per worker. Plain tvlqr is also checked against the earlier p50-closedloop run.
    .venv/bin/python figures/2026-10-08/d_sideslip_bench/render.py
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
BENCH = os.path.join(RD, "2026-10-08_nojob_sideslip-bench/gazebo")
REF = os.path.join(RD, "2026-10-08_nojob_p50-closedloop/gazebo")
OUT = os.path.dirname(os.path.abspath(__file__))


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "**", "*__*.csv"), recursive=True):
        name, corr = os.path.basename(f)[:-4].split("__")
        a = np.genfromtxt(f, delimiter=",", names=True)
        wc = wall_contact(np.c_[a["k"] * 0.1, a["true_x"], a["true_y"], a["true_theta"]], 6)
        s = np.r_[0, np.cumsum(np.hypot(np.diff(a["plan_x"]), np.diff(a["plan_y"])))]
        out[(name, corr)] = dict(a=a, s=s, max=float(np.nanmax(np.abs(a["e_cross"]))),
                                 rms=float(np.sqrt(np.nanmean(a["e_cross"] ** 2))),
                                 clear=wc["wall_min_clear"], contact=wc["wall_overlap_s"] > 0,
                                 final=float(np.hypot(a["true_x"][-1] - a["plan_x"][-1], a["true_y"][-1] - a["plan_y"][-1])))
    return out


B, P = load(BENCH), load(REF)
plans = sorted({k[0] for k in B if (k[0], "tvlqr_slip") in B and (k[0], "tvlqr") in B})
L = [f"n = {len(plans)} plans (delta = tvlqr_slip - tvlqr, paired)"]


def pair(key, a="tvlqr", b="tvlqr_slip", src=B):
    x = np.array([src[(p, a)][key] for p in plans], float)
    y = np.array([B[(p, b)][key] for p in plans], float)
    d = y - x
    nb, nw = int(np.sum(d < -1e-9)), int(np.sum(d > 1e-9))
    pv = binomtest(nb, nb + nw).pvalue if nb + nw else 1.0
    return x, y, d, nb, nw, pv


for key in ("max", "rms", "final", "clear"):
    x, y, d, nb, nw, pv = pair(key)
    better = "lower" if key != "clear" else "higher"
    if key == "clear":
        nb, nw = nw, nb
    L.append(f"{key:6s} tvlqr {x.mean():.3f}  slip {y.mean():.3f}  delta mean {d.mean():+.3f} median {np.median(d):+.3f}"
             f"  slip better ({better}) {nb}/{nb + nw}  p={pv:.2g}")
co = np.array([B[(p, "tvlqr")]["contact"] for p in plans]); cs = np.array([B[(p, "tvlqr_slip")]["contact"] for p in plans])
L.append(f"contact plans tvlqr {co.sum()}  slip {cs.sum()}  (fixed {int(np.sum(co & ~cs))}, new {int(np.sum(~co & cs))})")
ref = [p for p in plans if (p, "tvlqr") in P]
if ref:
    r = np.array([B[(p, "tvlqr")]["max"] - P[(p, "tvlqr")]["max"] for p in ref])
    L.append(f"sanity: bench tvlqr vs p50-closedloop tvlqr max|e| over {len(ref)} plans: mean |diff| {np.abs(r).mean():.4f}, max {np.abs(r).max():.4f}")
open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# --- 1. per-plan scatter ---------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, key, lab in zip(axs, ("max", "final"), ("max|e_cross| [m]", "final error [m]")):
    x, y, *_ = pair(key)
    c = np.where(co & ~cs, "tab:green", np.where(~co & cs, "tab:red", np.where(co, "0.3", "tab:blue")))
    ax.scatter(x, y, c=c, s=28)
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k:", lw=1); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel(f"tvlqr {lab}"); ax.set_ylabel(f"tvlqr_slip {lab}"); ax.grid(alpha=.3)
axs[0].set_title("below the diagonal = sideslip model better\n"
                 "green: contact fixed, red: new contact, grey: both, blue: neither", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "paired_scatter.png"), dpi=110); plt.close(fig)

# --- 2. tracks at plain TVLQR's worst point ------------------------------------
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
meta = yaml.safe_load(open(MAP + ".yaml")); img = np.asarray(Image.open(MAP + ".png").convert("L"))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
EXT = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
dmax = {p: B[(p, "tvlqr_slip")]["max"] - B[(p, "tvlqr")]["max"] for p in plans}
order = sorted(plans, key=lambda p: dmax[p])
pick = [order[0], order[len(order) // 2], order[-1]]
fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), gridspec_kw=dict(height_ratios=[3, 1.5]))
for j, p in enumerate(pick):
    ax, ae = axs[0, j], axs[1, j]
    g = B[(p, "tvlqr")]
    k = int(np.nanargmax(np.abs(g["a"]["e_cross"])))
    ax.imshow(img, cmap="gray", extent=EXT, vmin=0, vmax=255)
    ax.plot(g["a"]["plan_x"], g["a"]["plan_y"], c="tab:green", lw=1.2, label="plan")
    for corr, col in (("tvlqr", "k"), ("tvlqr_slip", "tab:orange")):
        r = B[(p, corr)]
        ax.plot(r["a"]["true_x"], r["a"]["true_y"], c=col, lw=1.8, label=corr)
        ae.plot(r["s"], r["a"]["e_cross"], c=col, lw=1.4)
    ax.set_xlim(g["a"]["plan_x"][k] - 1.6, g["a"]["plan_x"][k] + 1.6)
    ax.set_ylim(g["a"]["plan_y"][k] - 1.6, g["a"]["plan_y"][k] + 1.6)
    ax.set_title(f"{p}: max|e| {g['max']:.3f} -> {B[(p, 'tvlqr_slip')]['max']:.3f} m", fontsize=9)
    ae.axvline(g["s"][k], c="0.6", lw=.8); ae.axhline(0, c="0.7", lw=.7)
    ae.set_xlabel("distance along plan [m]"); ae.set_ylabel("e_cross [m]"); ae.grid(alpha=.3)
axs[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("Gazebo, flat ground, at plain TVLQR's worst point: the largest gain from the sideslip model, the median plan, "
             "and the largest loss", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tracks.png"), dpi=110); plt.close(fig)
print("wrote", OUT)
