#!/usr/bin/env python3
"""Stop-and-aim (b), #50: base vs bottleneck-aligned plans (both re-solved fresh, one run of
tools/solve_field_variant.py) on the closed-loop Gazebo bench, flat rl_corrector.world,
plain TVLQR and gated sideslip TVLQR. Data: run_data/2026-10-08_nojob_bottleneck-bench.
    .venv/bin/python figures/2026-10-08/bottleneck_bench/render.py
"""
import glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from scipy.stats import binomtest

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning")]
from compare_run import wall_contact  # noqa: E402
from replan_footprint import load_floor  # noqa: E402
from agx_planning.vector_field.bottleneck import find_bottlenecks  # noqa: E402

D = os.path.join(R, "run_data/2026-10-08_nojob_bottleneck-bench")
OUT = os.path.dirname(os.path.abspath(__file__))
DOOR_R = 0.6   # [m] "at a door": plan point within this of a bottleneck centre
arr, meta = load_floor(6)
res, (ox, oy) = meta["resolution"], meta["origin"][:2]
DOORS = np.array([(b.x, b.y) for b in find_bottlenecks(arr >= 65, res, ox, oy)])


def load(lib):
    out = {}
    for f in glob.glob(os.path.join(D, "gazebo", lib, "**", "*__*.csv"), recursive=True):
        name, corr = os.path.basename(f)[:-4].split("__")
        a = np.genfromtxt(f, delimiter=",", names=True)
        wc = wall_contact(np.c_[a["k"] * 0.1, a["true_x"], a["true_y"], a["true_theta"]], 6)
        z = np.load(os.path.join(D, "libs", lib, name + ".npz"))
        pw = wall_contact(np.c_[np.arange(len(z["poses"])) * 0.1, z["poses"]], 6)
        e = np.abs(a["e_cross"])
        dd = np.min(np.hypot(a["plan_x"][:, None] - DOORS[:, 0], a["plan_y"][:, None] - DOORS[:, 1]), axis=1)
        door = e[dd < DOOR_R]
        out[(name, corr)] = dict(
            a=a, max=float(np.nanmax(e)), door=float(np.nanmax(door)) if door.size else np.nan,
            clear=wc["wall_min_clear"], contact=wc["wall_overlap_s"] > 0, plan_clear=pw["wall_min_clear"],
            final=float(np.hypot(a["true_x"][-1] - a["plan_x"][-1], a["true_y"][-1] - a["plan_y"][-1])),
            dur=len(a) * 0.1)
    return out


B, A = load("base"), load("align")
L = [f"bottleneck-aligned (align) vs base plans, both re-solved fresh; doors = {len(DOORS)} detected bottlenecks,"
     f" 'door' = max|e| where the plan is within {DOOR_R} m of one; clear = min hull clearance (Gazebo truth);"
     " plan_clear = the plan's own"]
for corr in ("tvlqr", "tvlqr_slip"):
    plans = sorted(p for (p, c) in B if c == corr and (p, c) in A)
    L.append(f"\n{corr}: n = {len(plans)}")
    for key in ("max", "door", "final", "clear", "plan_clear", "dur"):
        x = np.array([B[(p, corr)][key] for p in plans]); y = np.array([A[(p, corr)][key] for p in plans])
        d = y - x
        sgn = -1 if key in ("clear", "plan_clear") else 1
        nb, nw = int(np.sum(sgn * d < -1e-9)), int(np.sum(sgn * d > 1e-9))
        pv = binomtest(nb, nb + nw).pvalue if nb + nw else 1.0
        L.append(f"  {key:10s} base {np.nanmean(x):.3f}  align {np.nanmean(y):.3f}  delta median {np.nanmedian(d):+.3f}"
                 f"  align better {nb}/{nb + nw}  p={pv:.2g}")
    cb = np.array([B[(p, corr)]["contact"] for p in plans]); ca = np.array([A[(p, corr)]["contact"] for p in plans])
    L.append(f"  contact plans base {cb.sum()}  align {ca.sum()}  (fixed {int(np.sum(cb & ~ca))}, new {int(np.sum(~cb & ca))})")
open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# --- tracks at the doors of the plans where align changes the door error most (plain tvlqr) ---
corr = "tvlqr"
plans = sorted(p for (p, c) in B if c == corr and (p, c) in A)
dd = {p: A[(p, corr)]["door"] - B[(p, corr)]["door"] for p in plans if np.isfinite(A[(p, corr)]["door"] - B[(p, corr)]["door"])}
pick = sorted(dd, key=dd.get)[:2] + sorted(dd, key=dd.get)[-2:]
wall = arr >= 65
EXT = [ox, ox + arr.shape[1] * res, oy, oy + arr.shape[0] * res]
fig, axs = plt.subplots(1, len(pick), figsize=(5 * len(pick), 5.5))
for ax, p in zip(axs, pick):
    ax.imshow(np.where(wall, 0.0, np.where(arr < 0, 0.75, 1.0)), cmap="gray", origin="lower", extent=EXT, vmin=0, vmax=1)
    for lib, S, c in (("base", B, "tab:blue"), ("align", A, "tab:red")):
        a = S[(p, corr)]["a"]
        ax.plot(a["plan_x"], a["plan_y"], c=c, lw=0.8, ls=":")
        ax.plot(a["true_x"], a["true_y"], c=c, lw=1.6, label=f"{lib}: door max|e| {S[(p, corr)]['door']:.2f}"
                f"{' CONTACT' if S[(p, corr)]['contact'] else ''}")
    a = B[(p, corr)]["a"]
    e = np.abs(a["e_cross"]); k = int(np.nanargmax(np.where(np.isfinite(e), e, -1)))
    # centre on the door nearest the base run's worst point
    j = int(np.argmin(np.hypot(DOORS[:, 0] - a["plan_x"][k], DOORS[:, 1] - a["plan_y"][k])))
    cx, cy = DOORS[j]
    ax.set_xlim(cx - 2, cx + 2); ax.set_ylim(cy - 2, cy + 2); ax.set_aspect("equal")
    ax.set_title(f"{p} ({corr})", fontsize=9); ax.legend(fontsize=7, loc="lower left")
fig.suptitle("Gazebo truth (solid) vs plan (dotted), base blue / aligned red; the two plans where align helps "
             "most at the doors (left two), and the two where it hurts most (right two; worse by <= 1 cm, panel centred on the door nearest base's worst point)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "door_tracks.png"), dpi=85); plt.close(fig)
