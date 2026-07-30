#!/usr/bin/env python3
"""Classify planned trajectories by SHAPE, so a comparison can pick genuinely
different ones instead of the same archetype three times.

This exists because of a real trap. The static-map fixture's goals are drawn by
random_goals from the robot's spawn point, and the six goals used for the
2026-07-25 TVLQR validation all came out near-straight, 6-9 m, heading the same
way -- two of them were literally the same goal. A corrector "validated" on that
set has only ever been shown to work on one kind of path. Sorting candidates by
shape first is what stops that happening again.

Reads either producer of a planned path:
    *.npz       recorded PMP rollouts (generate_trajectories.py) -- has `poses`
    *_plan.csv  run_recorder's planned-path dump -- has plan_x / plan_y

SHAPE DESCRIPTORS
  straightness  net displacement / path length. 1.0 is a straight line.
  |turn|        total absolute heading change [rad] -- how much steering happens.
  net turn      signed total -- distinguishes a corner (net ~ |turn|) from an
                S-curve (net ~ 0 while |turn| is large).
  max turn      largest single-step heading change; flags degenerate plans.
  sgn           sign changes in the smoothed turn signal -- the S-curve tell.

Paths are ARC-LENGTH RESAMPLED before any of this, so the descriptors describe
geometry rather than however densely the producer happened to sample it.

OFFLINE TOOL -- numpy only for .npz, nothing else. No ROS.

    python3 tools/classify_plans.py '~/pmp_trajectories_v2/*.npz'
    python3 tools/classify_plans.py 'run_data/identity_*_plan.csv'
"""

import argparse
import csv
import glob
import math
import os
from collections import Counter


def load_csv(path):
    xs, ys = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                x, y = float(row["plan_x"]), float(row["plan_y"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
    return xs, ys, {}


def load_npz(path):
    import numpy as np
    with np.load(path) as z:
        poses = z["poses"]
        meta = {}
        if "map_yaml" in z:
            meta["floor"] = str(z["map_yaml"]).split("floor_")[-1].split(".")[0]
        if "start_xy" in z:
            meta["start"] = tuple(float(v) for v in z["start_xy"])
        if "goal_xy" in z:
            meta["goal"] = tuple(float(v) for v in z["goal_xy"])
    return poses[:, 0].tolist(), poses[:, 1].tolist(), meta


def resample(xs, ys, step=0.15):
    """Arc-length resample, so curvature is not dominated by sample density."""
    out = [(xs[0], ys[0])]
    acc = 0.0
    for i in range(1, len(xs)):
        acc += math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        if acc >= step:
            out.append((xs[i], ys[i]))
            acc = 0.0
    if out[-1] != (xs[-1], ys[-1]):
        out.append((xs[-1], ys[-1]))
    return out


def describe(xs, ys):
    pts = resample(xs, ys)
    if len(pts) < 4:
        return None
    L = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts)))
    net = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    hd = [math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
          for i in range(len(pts) - 1)]
    turns = []
    for i in range(1, len(hd)):
        d = hd[i] - hd[i - 1]
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        turns.append(d)
    if not turns:
        return None
    w = 3
    sm = [sum(turns[max(0, i - w):i + w + 1]) / len(turns[max(0, i - w):i + w + 1])
          for i in range(len(turns))]
    sign_changes, prev = 0, 0
    for t in sm:
        s = 1 if t > 0.03 else (-1 if t < -0.03 else 0)
        if s:
            if prev and s != prev:
                sign_changes += 1
            prev = s
    return dict(L=L, net=net, straightness=net / L if L > 0 else float("nan"),
                total_abs=sum(abs(t) for t in turns), net_turn=sum(turns),
                max_turn=max(abs(t) for t in turns), sign_changes=sign_changes)


def label(d):
    if d["straightness"] > 0.985 and d["total_abs"] < 0.6:
        return "STRAIGHT"
    if d["sign_changes"] >= 2 or (d["sign_changes"] >= 1
                                  and abs(d["net_turn"]) < 0.4
                                  and d["total_abs"] > 1.2):
        return "S-CURVE"
    if abs(d["net_turn"]) > 1.0 or d["max_turn"] > 0.35:
        return "CORNER"
    if d["total_abs"] > 0.8:
        return "curved"
    return "gentle"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="glob for *.npz or *_plan.csv (quote it)")
    ap.add_argument("--top", type=int, default=6, help="entries to show per shape")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.expanduser(args.pattern))):
        xs, ys, meta = (load_npz(p) if p.endswith(".npz") else load_csv(p))
        if len(xs) < 4:
            continue
        d = describe(xs, ys)
        if not d:
            continue
        d.update(meta)
        name = os.path.basename(p).replace(".npz", "").replace("_plan.csv", "")
        rows.append((name, d))

    if not rows:
        raise SystemExit(f"no usable paths matched {args.pattern!r}")

    print(Counter(label(d) for _, d in rows))
    print()
    hdr = (f"{'trajectory':22s} {'len':>6s} {'strt':>5s} {'|turn|':>7s} "
           f"{'netturn':>8s} {'maxturn':>8s} {'sgn':>4s}")
    for want in ("STRAIGHT", "S-CURVE", "CORNER", "curved", "gentle"):
        sel = [(n, d) for n, d in rows if label(d) == want]
        if not sel:
            continue
        # Most extreme example of each shape first -- that is the useful one to
        # pick for a comparison; a marginal S-curve tests nothing a straight does not.
        sel.sort(key=lambda r: (r[1]["total_abs"] if want == "STRAIGHT"
                                else -r[1]["total_abs"]))
        print(f"--- {want} ({min(args.top, len(sel))} of {len(sel)}) ---")
        print(hdr)
        for n, d in sel[:args.top]:
            line = (f"{n:22s} {d['L']:6.2f} {d['straightness']:5.2f} "
                    f"{d['total_abs']:7.2f} {d['net_turn']:8.2f} "
                    f"{d['max_turn']:8.2f} {d['sign_changes']:4d}")
            if "goal" in d:
                line += (f"  start=({d['start'][0]:7.2f},{d['start'][1]:7.2f})"
                         f" goal=({d['goal'][0]:7.2f},{d['goal'][1]:7.2f})")
            print(line)
        print()


if __name__ == "__main__":
    main()
