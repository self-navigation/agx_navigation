#!/usr/bin/env python3
"""#50: is the fitted DynamicBridge a faithful stand-in for Gazebo?

Open loop: held-out Gazebo excitation sequences (random seeds 30-39 + 17 of the
34 plans' nominal commands) vs the fitted surrogate and the kinematic model.
Closed loop: compare_correctors (identity, tvlqr) on the 34 centred plans,
Gazebo vs surrogate vs kinematic. Flat ground, no patches, everywhere.
    .venv/bin/python figures/2026-10-08/p50_surrogate/render.py
"""
import glob, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, yaml
from PIL import Image

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(R, "tools"))
sys.path.insert(0, os.path.join(R, "src/agx_navigation/agx_planning"))
from fit_surrogate import kinematic, load, windows  # noqa: E402
from compare_run import wall_contact  # noqa: E402
from agx_planning.rl_corrector.dynamic_bridge import DynamicParams, simulate  # noqa: E402

EX = os.path.join(R, "run_data/2026-10-08_nojob_p50-excitation")
CL = os.path.join(R, "run_data/2026-10-08_nojob_p50-closedloop")
OUT = os.path.dirname(os.path.abspath(__file__))
FIT = json.load(open(os.path.join(EX, "fit.json")))
P = DynamicParams.load(os.path.join(EX, "fit.json"))
CHI_FIT, CHI_CFG = FIT["fit_report"]["chi_kinematic"], 1.373
HO_PLANS = open(os.path.join(EX, "holdout_plans.txt")).read().split()
HOLDOUT = [f"rand_s{s}" for s in range(30, 40)] + HO_PLANS
STY = {"gazebo": dict(c="k", lw=2.2, label="Gazebo"),
       "dyn": dict(c="tab:red", lw=1.3, label="surrogate (fitted)"),
       "kin": dict(c="tab:blue", lw=1.1, ls="--", label=f"kinematic, χ fitted {CHI_FIT:.2f}"),
       "kcfg": dict(c="tab:cyan", lw=1.1, ls=":", label=f"kinematic, χ config {CHI_CFG}")}

seqs = {s["name"]: s for s in load(sorted(glob.glob(os.path.join(EX, "gazebo/*/*.npz"))))}
held = [s for n, s in seqs.items() if n in HOLDOUT]


def rollout(s):
    """Whole sequence open loop from rest (Gazebo's own t=0 state)."""
    st0 = np.r_[s["pose"][0], 0, 0, 0][None]
    C = s["cmd"][None]
    dyn = simulate(P, st0, np.zeros((1, 4)), C, s["dt"])[0][0, :, :3]
    return dict(gazebo=s["pose"], dyn=dyn, kin=kinematic(st0, C, s["dt"], CHI_FIT)[0],
                kcfg=kinematic(st0, C, s["dt"], CHI_CFG)[0])


# --- 1. open loop, whole held-out sequences -------------------------------------
show = ["rand_s30", "rand_s31", "rand_s32", "rand_s33", HO_PLANS[0], HO_PLANS[5]]
fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
for ax, name in zip(axs.flat, show):
    tr = rollout(seqs[name])
    for k, xy in tr.items():
        ax.plot(xy[:, 0], xy[:, 1], **STY[k])
    ax.plot(*tr["gazebo"][0, :2], "go")
    end = {k: np.hypot(*(v[-1, :2] - tr["gazebo"][-1, :2])) for k, v in tr.items() if k != "gazebo"}
    ax.set_title(f"{name}  ({len(seqs[name]['cmd']) * 0.02:.0f} s)\nend error: surrogate {end['dyn']:.2f} m, "
                 f"kin {end['kin']:.2f} m, kin cfg {end['kcfg']:.2f} m", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
axs[0, 0].legend(fontsize=8)
fig.suptitle("Open loop from rest, HELD-OUT sequences (not used in the fit): same wheel commands into each model", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "openloop_heldout_tracks.png"), dpi=110); plt.close(fig)

# --- 2. 3 s window errors, held out ---------------------------------------------
S, O, C, T, dt = windows(held, 3.0, 1.0)
pred = dict(dyn=simulate(P, S, O, C, dt)[0][..., :3], kin=kinematic(S, C, dt, CHI_FIT), kcfg=kinematic(S, C, dt, CHI_CFG))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
for k, p in pred.items():
    e = np.sort(np.hypot(*(p[:, -1, :2] - T[:, -1, :2]).T))
    h = np.sort(np.degrees(np.abs(p[:, -1, 2] - T[:, -1, 2])))
    q = np.arange(1, len(e) + 1) / len(e)
    a1.plot(e, q, **{**STY[k], "label": f"{STY[k]['label']}: p50 {np.median(e):.3f}, p90 {np.percentile(e, 90):.3f} m"})
    a2.plot(h, q, **{**STY[k], "label": f"{STY[k]['label']}: p50 {np.median(h):.2f}°"})
a1.set_xlabel("position error after 3 s [m]"); a2.set_xlabel("heading error after 3 s [deg]")
for a in (a1, a2):
    a.set_ylabel("fraction of windows"); a.grid(alpha=.3); a.legend(fontsize=8, loc="upper left"); a.set_xscale("log")
a1.set_xlim(1e-3, 1); a2.set_xlim(1e-2, 100)
fig.suptitle(f"{len(S)} held-out 3 s windows, each started from Gazebo's own state "
             "(the ~25% at the left edge are windows at rest: every model is exact there)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "window_error_cdf.png"), dpi=110); plt.close(fig)


# --- 3. closed loop, per plan ---------------------------------------------------
def closed(sub):
    out = {}
    for f in glob.glob(os.path.join(CL, sub, "**", "*__*.csv"), recursive=True):
        name, corr = os.path.basename(f)[:-4].split("__")
        a = np.genfromtxt(f, delimiter=",", names=True)
        wc = wall_contact(np.c_[a["k"] * 0.1, a["true_x"], a["true_y"], a["true_theta"]], 6)
        out[(name, corr)] = dict(a=a, max=float(np.nanmax(np.abs(a["e_cross"]))), contact=wc["wall_overlap_s"] > 0)
    return out


CLR = {"gazebo": closed("gazebo"), "dyn": closed("dynamic_fit"), "kin": closed("kinematic_sdf")}
plans = sorted({k[0] for k in CLR["gazebo"]})
ho = np.array(["plan_" + p in HO_PLANS for p in plans])
fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, corr, lim in zip(axs, ("identity", "tvlqr"), (2.2, 0.3)):
    g = np.array([CLR["gazebo"][(p, corr)]["max"] for p in plans])
    for k in ("dyn", "kin"):
        m = np.array([CLR[k][(p, corr)]["max"] for p in plans])
        ax.scatter(g[ho], m[ho], c=STY[k]["c"], marker="o", s=28, label=f"{STY[k]['label']}, held-out plan")
        ax.scatter(g[~ho], m[~ho], facecolors="none", edgecolors=STY[k]["c"], marker="o", s=28, label=f"{STY[k]['label']}, plan in fit")
    ax.plot([0, lim], [0, lim], "k:", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("Gazebo max|e_cross| [m]"); ax.set_ylabel("model max|e_cross| [m]")
    gc = np.array([CLR["gazebo"][(p, corr)]["contact"] for p in plans])
    dc = np.array([CLR["dyn"][(p, corr)]["contact"] for p in plans])
    kc = np.array([CLR["kin"][(p, corr)]["contact"] for p in plans])
    ax.set_title(f"{corr}: wall contact (baked map) Gazebo {gc.sum()}/34, surrogate {dc.sum()}/34, kinematic {kc.sum()}/34\n"
                 f"contact agrees with Gazebo: surrogate {np.sum(dc == gc)}/34, kinematic {np.sum(kc == gc)}/34", fontsize=9)
axs[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Closed loop on the 34 centred plans, flat ground (compare_correctors, 10 Hz)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "closedloop_max_cross.png"), dpi=110); plt.close(fig)

# --- 4. closed-loop TVLQR tracks at the worst turn, over the map ----------------
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
meta = yaml.safe_load(open(MAP + ".yaml")); img = np.asarray(Image.open(MAP + ".png").convert("L"))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
EXT = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
pick = [p for p, h in zip(plans, ho) if h][:3]
fig, axs = plt.subplots(2, 3, figsize=(15, 9), gridspec_kw=dict(height_ratios=[3, 1.4]))
for j, p in enumerate(pick):
    ax, ae = axs[0, j], axs[1, j]
    g = CLR["gazebo"][(p, "tvlqr")]["a"]
    k = int(np.nanargmax(np.abs(g["e_cross"])))
    ax.imshow(img, cmap="gray", extent=EXT, vmin=0, vmax=255)
    ax.plot(g["plan_x"], g["plan_y"], color="tab:green", lw=1.2, label="plan")
    for key in ("gazebo", "dyn", "kin"):
        a = CLR[key][(p, "tvlqr")]["a"]
        ax.plot(a["true_x"], a["true_y"], **STY[key])
        ae.plot(a["k"] * 0.1, a["e_cross"], **STY[key])
    ax.set_xlim(g["plan_x"][k] - 2, g["plan_x"][k] + 2); ax.set_ylim(g["plan_y"][k] - 2, g["plan_y"][k] + 2)
    ax.set_title(f"{p} (held out): worst Gazebo turn, TVLQR", fontsize=9)
    ae.axvline(k * 0.1, color="0.6", lw=.8); ae.set_xlabel("t [s]"); ae.set_ylabel("e_cross [m]"); ae.grid(alpha=.3)
axs[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "closedloop_tvlqr_tracks.png"), dpi=110); plt.close(fig)
print("wrote", OUT)
