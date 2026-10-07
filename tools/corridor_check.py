#!/usr/bin/env python3
"""Check plan libraries against the corrector's measured tracking deviation.

    .venv/bin/python tools/corridor_check.py

A plan point is inside the ε-corridor only if its hull clearance exceeds the
deviation the corrector shows there, measured as the ground-truth distance from
tracked runs to the plan polyline. For each library it reports the share of
plan length whose clearance is below that deviation and whether that share
predicts wall contact.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src/agx_navigation/agx_planning"))
from agx_planning.supervisor.walls import SignedWallDistance  # noqa: E402

C1 = os.path.join(ROOT, "run_data/2026-10-07_job230_c1-supervisor")
LIBS = {"old (traj_data_v2)": os.path.join(ROOT, "run_data/2026-08-14_job020_v2-library/traj_data_v2"),
        "centred (v2c, #49)": os.path.join(ROOT, "run_data/2026-10-07_nojob_centred-library/traj_data_v2c")}
SW = SignedWallDistance.from_map_yaml(os.path.join(ROOT, "src/rudn-ordjo-building/maps/floor_6.yaml"))
PLANS = [l.strip().split("/")[-1][:-4] for l in open(os.path.join(ROOT, "tools/jobs/broad40.txt")) if l.strip()]


def seg_dist(P, Q):
    """Distance from each point of Q to the polyline P (both (n, >=2))."""
    a, b = P[:-1, :2], P[1:, :2]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    out = np.empty(len(Q))
    for i, q in enumerate(Q[:, :2]):
        t = np.clip(((q - a) * ab).sum(1) / L2, 0, 1)
        out[i] = np.sqrt(((a + t[:, None] * ab - q) ** 2).sum(1)).min()
    return out


def deviations(cfg):
    dev, per = [], {}
    for line in open(os.path.join(C1, cfg, "rows.s0.jsonl")):
        r = json.loads(line)
        f = glob.glob(os.path.join(C1, cfg, "s0", f"track_*{r['plan']}.npz"))
        if not f or (r.get("path_length") or 0) < 2:
            continue
        z = np.load(f[0], allow_pickle=True)
        d = seg_dist(z["plan_poses"], z["track"][:, 1:3])
        dev.append(d)
        per[r["plan"]] = r
    return np.concatenate(dev), per


def clearance(P):
    return np.array([SW.body_clearance(*p[:3]) for p in P])


def main():
    devs = {}
    for cfg, name in (("LTN", "truth"), ("LAN", "amcl")):
        d, _ = deviations(cfg)
        devs[name] = d
        print(f"ours-lib deviation from plan, {name}: p50 {np.percentile(d, 50):.3f}  "
              f"p90 {np.percentile(d, 90):.3f}  p95 {np.percentile(d, 95):.3f} m  (n={len(d)})")
    req = {f"p{q} {n}": float(np.percentile(d, q)) for n, d in devs.items() for q in (50, 90)}
    print()
    clr = {}
    for lib, path in LIBS.items():
        clr[lib] = {}
        for p in PLANS:
            f = os.path.join(path, p + ".npz")
            if os.path.exists(f):
                clr[lib][p] = clearance(np.load(f)["poses"])
        allc = np.concatenate(list(clr[lib].values()))
        print(f"{lib}: {len(clr[lib])} plans; median plan clearance {np.median(allc):.3f} m")
        for k, v in req.items():
            sh = [np.mean(c < v) for c in clr[lib].values()]
            print(f"   share of plan points with clearance < {k} ({v:.3f} m): "
                  f"median {np.median(sh):.2f}, mean {np.mean(sh):.2f}, plans with any {sum(s > 0 for s in sh)}")
    print()
    # Does the corridor deficit of the OLD plans predict C1 contact without the layer?
    d90 = req["p90 truth"]
    for cfg in ("LTN", "LAN", "VTN", "RTN"):
        rows = {json.loads(l)["plan"]: json.loads(l) for l in open(os.path.join(C1, cfg, "rows.s0.jsonl"))}
        xs, ys = [], []
        for p, c in clr["old (traj_data_v2)"].items():
            if p in rows and rows[p].get("wall_overlap_s") is not None:
                xs.append(np.mean(c < d90))
                ys.append(rows[p]["wall_overlap_s"])
        rho, pv = spearmanr(xs, ys)
        print(f"C1 {cfg}: share of plan < p90 deviation vs wall-contact seconds: rho {rho:+.2f} (p={pv:.3f}, n={len(xs)})")


if __name__ == "__main__":
    main()
