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

# --- footprints at the door each plan passes, for the plans whose wall contact changes (plain tvlqr) ---
from matplotlib.patches import Polygon  # noqa: E402
HL, HW = 0.315, 0.2925   # compare_run.FP_HALF_LENGTH / FP_HALF_WIDTH


def hull(x, y, th):
    c, s = np.cos(th), np.sin(th)
    pts = np.array([(HL, HW), (HL, -HW), (-HL, -HW), (-HL, HW)])
    return np.c_[x + pts[:, 0] * c - pts[:, 1] * s, y + pts[:, 0] * s + pts[:, 1] * c]


import compare_run as _cr  # noqa: E402
_d, _r, _ox, _oy = _cr._signed_wall_dist(6)
_nl, _nw = int(2 * HL / _cr.FP_SPACING) + 1, int(2 * HW / _cr.FP_SPACING) + 1
_lx, _wy = np.linspace(-HL, HL, _nl), np.linspace(-HW, HW, _nw)
_BODY = np.vstack([np.c_[_lx, np.full(_nl, HW)], np.c_[_lx, np.full(_nl, -HW)],
                   np.c_[np.full(_nw, HL), _wy], np.c_[np.full(_nw, -HL), _wy]])


def hull_clear(a):
    """Per-sample min clearance of the hull perimeter (compare_run.wall_contact's raster)."""
    c, s = np.cos(a["true_theta"]), np.sin(a["true_theta"])
    px = a["true_x"][:, None] + _BODY[:, 0] * c[:, None] - _BODY[:, 1] * s[:, None]
    py = a["true_y"][:, None] + _BODY[:, 0] * s[:, None] + _BODY[:, 1] * c[:, None]
    cc = np.clip(((px - _ox) / _r).astype(int), 0, _d.shape[1] - 1)
    rr = np.clip(((py - _oy) / _r).astype(int), 0, _d.shape[0] - 1)
    return _d[rr, cc].min(axis=1)


def worst_door(S, p, corr):
    """The door this run passes with the largest |e|: (door xy, sample index of the max)."""
    a = S[(p, corr)]["a"]
    e = np.abs(a["e_cross"])
    dd = np.hypot(a["plan_x"][:, None] - DOORS[:, 0], a["plan_y"][:, None] - DOORS[:, 1])
    near = dd.min(axis=1) < DOOR_R
    k = int(np.nanargmax(np.where(near & np.isfinite(e), e, -1)))
    return DOORS[int(np.argmin(dd[k]))], k


corr = "tvlqr"
plans = sorted(p for (p, c) in B if c == corr and (p, c) in A)
fixed = [p for p in plans if B[(p, corr)]["contact"] and not A[(p, corr)]["contact"]]
new = [p for p in plans if A[(p, corr)]["contact"] and not B[(p, corr)]["contact"]]
fixed.sort(key=lambda p: A[(p, corr)]["door"] - B[(p, corr)]["door"])
pick = fixed[:3] + new[:1]
wall = arr >= 65
EXT = [ox, ox + arr.shape[1] * res, oy, oy + arr.shape[0] * res]
fig, axs = plt.subplots(2, len(pick), figsize=(4.6 * len(pick), 9.4), squeeze=False)
for j, p in enumerate(pick):
    # centre on base's tightest pose if it touched, else on align's
    ref = B if B[(p, corr)]["contact"] else A
    ar = ref[(p, corr)]["a"]
    kk = int(np.argmin(hull_clear(ar)))
    cx, cy = ar["true_x"][kk], ar["true_y"][kk]
    for i, (lib, S, col) in enumerate((("base", B, "tab:blue"), ("align", A, "tab:red"))):
        ax = axs[i, j]
        ax.imshow(np.where(wall, 0.0, np.where(arr < 0, 0.75, 1.0)), cmap="gray", origin="lower", extent=EXT,
                  vmin=0, vmax=1)
        a = S[(p, corr)]["a"]
        win = np.hypot(a["true_x"] - cx, a["true_y"] - cy) < 1.6
        ax.plot(a["plan_x"], a["plan_y"], c="0.4", lw=1, ls="--", label="plan")
        for k in np.nonzero(win)[0][::4]:   # every 0.4 s
            ax.add_patch(Polygon(hull(a["true_x"][k], a["true_y"][k], a["true_theta"][k]), closed=True,
                                 fc="none", ec=col, lw=0.6, alpha=0.7))
        ax.plot(a["true_x"], a["true_y"], c=col, lw=1.5, label="robot centre (Gazebo truth)")
        hc = hull_clear(a)
        kw = int(np.argmin(np.where(win, hc, np.inf)))
        ax.add_patch(Polygon(hull(a["true_x"][kw], a["true_y"][kw], a["true_theta"][kw]), closed=True,
                             fc=col, ec="k", lw=1.0, alpha=0.45, label="tightest pose", zorder=5))
        r = S[(p, corr)]
        ax.set_title(f"{p[11:]} {lib}: door max|e| {r['door']:.2f} m, min hull clearance "
                     f"{r['clear']:+.2f} m{'  CONTACT' if r['contact'] else ''}", fontsize=8.5,
                     color="crimson" if r["contact"] else "k")
        ax.set_xlim(cx - 1.0, cx + 1.0); ax.set_ylim(cy - 1.0, cy + 1.0); ax.set_aspect("equal")
        if i == 0 and j == 0:
            ax.legend(fontsize=7, loc="lower left")
fig.suptitle(f"Plain TVLQR, Gazebo bench: robot hull every 0.4 s around the base run's tightest pose (filled = tightest pose in view). "
             f"Top: base plan; bottom: bottleneck-aligned plan.\nAlign removed wall contact on {len(fixed)} plans "
             f"(first {min(3, len(fixed))} shown, largest door gain first) and added it on {len(new)} "
             f"(right column: in the final turn to the goal, 5.7 m from any door)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "door_tracks.png"), dpi=85); plt.close(fig)
