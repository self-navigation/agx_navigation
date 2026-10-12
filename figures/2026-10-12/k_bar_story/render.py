#!/usr/bin/env python3
"""Why TVLQG hits door jambs under real localizers: the sideslip k it controls with (k_bar), one plan, one seed.

Plan f6_medium_0376, seed 0, mu 0.5 patches. TVLQG cells: job 255 T (truth), Q0 (amcl), job 256 RQ (rtab); job 257
P{T,Q,R} (TVLQG with prior and plan-level k = -0.33). diag columns from CorrectionDiagnostics.FIELDS: v_ref 4,
valid 15, k_bar 18. diag_t, track[:, 0] and the row's wall_first_t share the sim clock (as in ../tvlqg_k_hat/k_hat.py).
Time axis = seconds since the robot starts moving (first diag tick with v_ref > 0.1). The switch activation is the
first tick where k_bar leaves k_plan (0 without a prior; with the prior k_bar = k_plan = -0.33 from the first tick).
Contact = row wall_overlap_s > 0, at wall_first_t. Bottom: ground-truth tracks over the baked floor_6 map near the
contact door, robot footprint (half length 0.315, half width 0.2925) at each first contact.

usage: render.py   (paths fixed below; writes k_bar_story.png next to this file)
"""
import json, os, glob
import numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
RD = os.path.join(R, "run_data")
PLAN, SEED, K_PLANT = "f6_medium_0376", 0, -0.33
HL, HW = 0.315, 0.2925
RUNS = [  # label, job dir, cell, colour, linestyle
    ("truth", "2026-10-12_job255_steerable", "T", "black", "-"),
    ("amcl", "2026-10-12_job255_steerable", "Q0", "tab:blue", "-"),
    ("rtab", "2026-10-12_job256_steerable-rtab", "RQ", "tab:red", "-"),
    ("truth + prior", "2026-10-12_job257_kprior", "PT", "black", "--"),
    ("amcl + prior", "2026-10-12_job257_kprior", "PQ", "tab:blue", "--"),
    ("rtab + prior", "2026-10-12_job257_kprior", "PR", "tab:red", "--"),
]


def load(job, cell):
    for l in open(os.path.join(RD, job, cell, f"rows.s{SEED}.jsonl")):
        r = json.loads(l)
        if PLAN in r["plan"]:
            return r, np.load(os.path.join(RD, job, cell, f"s{SEED}", os.path.basename(r["track_path"])))
    raise KeyError((job, cell))


plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 18, "legend.fontsize": 14})
fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 17), gridspec_kw={"height_ratios": [1, 1.15]})
contacts, data = [], []
for lab, job, cell, col, ls in RUNS:
    r, z = load(job, cell)
    D, t, T = z["diag"], z["diag_t"], z["track"]
    t0 = t[np.argmax(D[:, 4] > 0.1)]
    kb, kplan = D[:, 18], D[0, 18]
    hit = (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None
    on = np.abs(kb - kplan) > 1e-9
    ta = t[np.argmax(on)] if on.any() else None
    data.append((lab, col, ls, T, r, hit))
    off = {"truth + prior": 0.006, "rtab + prior": -0.006}.get(lab, 0.0)  # prior arms coincide at -0.33; nudge to see all
    ax.plot(t - t0, kb + off, color=col, ls=ls, lw=3 if ls == "-" else 2.2, label=lab + (": contact" if hit else ": clean"))
    if ta is not None:
        ax.plot(ta - t0, kplan, "o", ms=13, mfc="white", mec=col, mew=3, zorder=5)
        print(f"{lab:14s} switch active {ta - t0:5.1f} s after moving")
    if hit:
        tc = r["wall_first_t"]
        ax.plot(tc - t0, np.interp(tc, t, kb), "X", ms=20, color=col, mec="k", zorder=6)
        i = min(np.searchsorted(T[:, 0], tc), len(T) - 1)
        contacts.append((col, T[i, 1], T[i, 2], T[i, 3]))
        print(f"{lab:14s} contact {tc - t0:5.1f} s after moving, k_bar there {np.interp(tc, t, kb):+.3f}")
    else:
        print(f"{lab:14s} clean, mean k_bar while moving {kb[(D[:, 4] > 0.1) & (D[:, 15] > 0)].mean():+.3f}")
ax.axhline(K_PLANT, color="tab:green", lw=6, alpha=0.35, zorder=0, label="Gazebo plant k = -0.33")
ax.plot([], [], "o", ms=13, mfc="white", mec="k", mew=3, ls="none", label="switch activates (k_bar leaves 0)")
ax.plot([], [], "X", ms=18, color="grey", mec="k", ls="none", label="first wall contact")
ax.set_xlim(-1, 25); ax.set_ylim(-0.37, 0.03)
ax.axvspan(9.6, 10.8, color="0.5", alpha=0.12, lw=0)
ax.text(11.1, -0.06, "door turn\n(all runs reach it ~10 s in)", fontsize=15, color="0.3", va="center")
ax.set_xlabel("time since the robot starts moving [s]"); ax.set_ylabel("k_bar: sideslip k the gains use [s]")
ax.set_title(f"TVLQG's sideslip k, plan {PLAN}, seed {SEED}")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, framealpha=0.9); ax.grid(alpha=0.3)

meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))
res, (ox0, oy0, _) = meta["resolution"], meta["origin"]
ext = [ox0, ox0 + img.shape[1] * res, oy0, oy0 + img.shape[0] * res]
bx.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255)
P = data[0][4] and load(RUNS[0][1], RUNS[0][2])[1]["plan_poses"]
bx.plot(P[:, 0], P[:, 1], color="tab:green", lw=2, ls=":", label="frozen PMP plan")
for lab, col, ls, T, r, hit in data:
    bx.plot(T[:, 1], T[:, 2], color=col, ls=ls, lw=3 if ls == "-" else 2.2, label=lab)
c, s = np.cos, np.sin
for col, x, y, yaw in contacts:
    loc = np.array([[HL, HW], [-HL, HW], [-HL, -HW], [HL, -HW]])
    xy = np.c_[x + c(yaw) * loc[:, 0] - s(yaw) * loc[:, 1], y + s(yaw) * loc[:, 0] + c(yaw) * loc[:, 1]]
    bx.add_patch(Polygon(xy, closed=True, fc=col, alpha=0.3, ec=col, lw=3))
    bx.plot(x + c(yaw) * HL * 0.8, y + s(yaw) * HL * 0.8, "^", color=col, ms=8)
cx, cy = np.mean([p[1] for p in contacts]), np.mean([p[2] for p in contacts])
bx.set_xlim(cx - 2.5, cx + 2.5); bx.set_ylim(cy - 2.0, cy + 2.0); bx.set_aspect("equal")
bx.set_xlabel("x [m]"); bx.set_ylabel("y [m]")
bx.set_title("Ground-truth tracks at the contact door (shaded: footprint at first contact)")
bx.legend(loc="lower left", fontsize=13, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "k_bar_story.png"), dpi=110)
