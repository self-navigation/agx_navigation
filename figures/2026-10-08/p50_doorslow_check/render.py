#!/usr/bin/env python3
"""#50: does the fitted surrogate predict the CHANGE a fix makes? Test case: job 237's door slowing.

Closed-loop bench (compare_correctors, flat ground) on the 34 centred plans, orig vs
door-slowed (same paths, clock slowed near walls). Gazebo vs fitted DynamicBridge vs
KinematicBridge; per-plan delta = slow - orig.
    .venv/bin/python figures/2026-10-08/p50_doorslow_check/render.py
"""
import glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, yaml
from PIL import Image
from scipy.stats import binomtest, spearmanr

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(R, "tools"))
from compare_run import wall_contact  # noqa: E402

RD = os.path.join(R, "run_data")
CK = os.path.join(RD, "2026-10-08_nojob_p50-doorslow-check")
SRC = {("gazebo", "orig"): os.path.join(RD, "2026-10-08_nojob_p50-closedloop/gazebo"),
       ("gazebo", "slow"): os.path.join(CK, "gazebo"),
       ("dyn", "orig"): os.path.join(CK, "dynamic_fit/orig"), ("dyn", "slow"): os.path.join(CK, "dynamic_fit/slow"),
       ("kin", "orig"): os.path.join(CK, "kinematic/orig"), ("kin", "slow"): os.path.join(CK, "kinematic/slow")}
OUT = os.path.dirname(os.path.abspath(__file__))
COL = {"gazebo": "k", "dyn": "tab:red", "kin": "tab:blue"}
LAB = {"gazebo": "Gazebo", "dyn": "surrogate (fitted)", "kin": "kinematic"}


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "**", "*__*.csv"), recursive=True):
        name, corr = os.path.basename(f)[:-4].split("__")
        a = np.genfromtxt(f, delimiter=",", names=True)
        wc = wall_contact(np.c_[a["k"] * 0.1, a["true_x"], a["true_y"], a["true_theta"]], 6)
        s = np.r_[0, np.cumsum(np.hypot(np.diff(a["plan_x"]), np.diff(a["plan_y"])))]
        out[(name, corr)] = dict(a=a, s=s, max=float(np.nanmax(np.abs(a["e_cross"]))),
                                 clear=wc["wall_min_clear"], contact=wc["wall_overlap_s"] > 0,
                                 final=float(np.hypot(a["true_x"][-1] - a["plan_x"][-1], a["true_y"][-1] - a["plan_y"][-1])))
    return out


D = {k: load(v) for k, v in SRC.items()}
plans = sorted({k[0] for k in D[("gazebo", "orig")]})
lines = []
for corr in ("identity", "tvlqr"):
    lines.append(f"=== {corr}  (delta = slow - orig, per plan; n={len(plans)})")
    g = np.array([D[("gazebo", "slow")][(p, corr)]["max"] - D[("gazebo", "orig")][(p, corr)]["max"] for p in plans])
    for m in ("gazebo", "dyn", "kin"):
        o = np.array([D[(m, "orig")][(p, corr)]["max"] for p in plans])
        s = np.array([D[(m, "slow")][(p, corr)]["max"] for p in plans])
        co = np.array([D[(m, "orig")][(p, corr)]["contact"] for p in plans])
        cs = np.array([D[(m, "slow")][(p, corr)]["contact"] for p in plans])
        fo = np.array([D[(m, "orig")][(p, corr)]["final"] for p in plans])
        fs = np.array([D[(m, "slow")][(p, corr)]["final"] for p in plans])
        d = s - o
        nb = int(np.sum(d < 0)); nw = int(np.sum(d > 0))
        lines.append(f"{LAB[m]:20s} max|e| orig {o.mean():.3f} slow {s.mean():.3f}  delta mean {d.mean():+.3f} "
                     f"median {np.median(d):+.3f}  slow better {nb}/{nb + nw} (p={binomtest(nb, nb + nw).pvalue:.2g})  "
                     f"contacts {co.sum()}->{cs.sum()}  final {fo.mean():.3f}->{fs.mean():.3f}"
                     + ("" if m == "gazebo" else f"  | delta vs Gazebo delta: spearman {spearmanr(d, g)[0]:+.2f}, "
                        f"same sign {int(np.sum(np.sign(d) == np.sign(g)))}/34, mean|err| {np.abs(d - g).mean():.3f}"))
    lines.append("")
open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(lines))
print("\n".join(lines))

# --- 1. per-plan delta scatter --------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, corr, lim in zip(axs, ("identity", "tvlqr"), (1.0, 0.12)):
    g = np.array([D[("gazebo", "slow")][(p, corr)]["max"] - D[("gazebo", "orig")][(p, corr)]["max"] for p in plans])
    for m in ("dyn", "kin"):
        d = np.array([D[(m, "slow")][(p, corr)]["max"] - D[(m, "orig")][(p, corr)]["max"] for p in plans])
        ax.scatter(g, d, c=COL[m], s=26, label=f"{LAB[m]} (Spearman {spearmanr(d, g)[0]:+.2f})")
    ax.plot([-lim, lim], [-lim, lim], "k:", lw=1); ax.axhline(0, c="0.7", lw=.7); ax.axvline(0, c="0.7", lw=.7)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("Gazebo: slow - orig max|e_cross| [m]"); ax.set_ylabel("model: slow - orig [m]")
    ax.set_title(f"{corr}: per-plan effect of door slowing", fontsize=10); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Does the model predict the change? (points on the diagonal = yes; below 0 = slowing helps)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "delta_scatter.png"), dpi=110); plt.close(fig)

# --- 2. tracks at the door, orig vs slow, Gazebo vs surrogate --------------------
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
meta = yaml.safe_load(open(MAP + ".yaml")); img = np.asarray(Image.open(MAP + ".png").convert("L"))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
EXT = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
gd = {p: D[("gazebo", "slow")][(p, "tvlqr")]["max"] - D[("gazebo", "orig")][(p, "tvlqr")]["max"] for p in plans}
pick = sorted(plans, key=lambda p: gd[p])[:2] + sorted(plans, key=lambda p: gd[p])[-1:]
fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), gridspec_kw=dict(height_ratios=[3, 1.5]))
for j, p in enumerate(pick):
    ax, ae = axs[0, j], axs[1, j]
    go = D[("gazebo", "orig")][(p, "tvlqr")]
    k = int(np.nanargmax(np.abs(go["a"]["e_cross"])))
    ax.imshow(img, cmap="gray", extent=EXT, vmin=0, vmax=255)
    ax.plot(go["a"]["plan_x"], go["a"]["plan_y"], c="tab:green", lw=1.2, label="plan (same path)")
    for m in ("gazebo", "dyn", "kin"):
        for v, ls in (("orig", "-"), ("slow", "--")):
            r = D[(m, v)][(p, "tvlqr")]
            lw = 2 if m == "gazebo" else 1.2
            ax.plot(r["a"]["true_x"], r["a"]["true_y"], c=COL[m], ls=ls, lw=lw, label=f"{LAB[m]}, {v}")
            ae.plot(r["s"], r["a"]["e_cross"], c=COL[m], ls=ls, lw=lw)
    ax.set_xlim(go["a"]["plan_x"][k] - 1.6, go["a"]["plan_x"][k] + 1.6)
    ax.set_ylim(go["a"]["plan_y"][k] - 1.6, go["a"]["plan_y"][k] + 1.6)
    dd = D[("dyn", "slow")][(p, "tvlqr")]["max"] - D[("dyn", "orig")][(p, "tvlqr")]["max"]
    ax.set_title(f"{p}, TVLQR: slow - orig max|e|\nGazebo {gd[p]:+.3f} m, surrogate {dd:+.3f} m", fontsize=9)
    ae.axvline(go["s"][k], c="0.6", lw=.8)
    ae.set_xlabel("distance along plan [m]"); ae.set_ylabel("e_cross [m]"); ae.grid(alpha=.3)
axs[0, 0].legend(fontsize=7, loc="lower left")
fig.suptitle("Orig (solid) vs door-slowed (dashed) plans at Gazebo's worst turn: the two plans with the largest "
             "Gazebo gain from slowing, and the largest loss", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "door_tracks_orig_vs_slow.png"), dpi=110); plt.close(fig)
print("wrote", OUT)
