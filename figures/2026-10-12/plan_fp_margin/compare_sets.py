"""Plan sets re-solved with the PMP footprint barrier (PLAN_PARAMS w_fp / fp_margin) vs the
original v3 held-out plans: per plan, hull clearance (min over the plan), duration, and how far
the path moved. Usage: python compare_sets.py SET_DIR [SET_DIR ...] -> stdout
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(R, "tools"))
import compare_run as cr
BASE = os.path.join(R, "run_data/2026-10-10_nojob_plan-set-v3/plans")
JAMB = ["f6_medium_0376", "f6_medium_0575", "f6_medium_0587", "f6_short_0068", "f6_short_0315"]
ids = [os.path.basename(l.strip())[:-4] for l in open(os.path.join(R, "tools/jobs/v3f6_test.txt")) if l.strip()]


def hull(P):
    T = np.c_[np.arange(len(P)) * 0.1, P[:, 0], P[:, 1], P[:, 2]]
    return cr.wall_contact(T, 6)["wall_min_clear"]


def shift(A, B):
    d = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    return d.min(axis=1).max()


sets = sys.argv[1:]
print("plan               base hull/dur | " + " | ".join(f"{os.path.basename(s)[24:]:>22s}" for s in sets))
agg = {s: [] for s in sets}
for i in ids:
    B = np.load(os.path.join(BASE, i + ".npz"))["poses"]
    hb = hull(B); line = f"{i:18s} {hb:5.2f} {len(B) / 10:5.1f}s |"
    for s in sets:
        f = os.path.join(s, "plans", i + ".npz")
        if not os.path.exists(f):
            line += f" {'FAILED':>22s} |"; agg[s].append(None); continue
        P = np.load(f)["poses"]; h = hull(P)
        agg[s].append((hb, h, len(P) / len(B), shift(P, B)))
        line += f" {h:5.2f} {len(P) / 10:5.1f}s sh{shift(P, B):4.2f} |"
    print(line + ("  <- jamb plan" if i in JAMB else ""))
for s in sets:
    a = [x for x in agg[s] if x]
    hb, h, dr, sh = map(np.array, zip(*a))
    jm = [x for x, i in zip(agg[s], ids) if x and i in JAMB]
    print(f"{os.path.basename(s)}: solved {len(a)}/{len(ids)}; hull median {np.median(hb):.2f} -> {np.median(h):.2f}, "
          f"min {hb.min():.2f} -> {h.min():.2f}; plans with hull < 0.20: {(hb < 0.2).sum()} -> {(h < 0.2).sum()}; "
          f"duration x{np.median(dr):.3f} (max x{dr.max():.2f}); path shift median {np.median(sh):.2f} m; "
          f"jamb plans hull " + ", ".join(f"{b:.2f}->{c:.2f}" for b, c, _, _ in jm))
