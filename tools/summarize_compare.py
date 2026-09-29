#!/usr/bin/env python3
"""Summarise compare_run.py output: per-arm table, paired tests, diagrams.

    .venv/bin/python tools/summarize_compare.py run_data/compare/ -o figures/tmp/

Reads every rows*.jsonl and the track_*.npz files they point at (paths are
rewritten to the given directory, so fetched copies work). Adds one metric the
row cannot hold, computed offline from the ground-truth track and the baked
map: `start_delay` (goal sent -> ground truth first 5 cm from the start;
our arm plans the whole trajectory before driving, Nav2 drives at once),
`drive_time` (travel_time minus it), and `min_clearance`, the smallest distance from the robot centre to a wall
cell minus the footprint's inscribed radius (0.29 m) -- negative means the
footprint overlapped a wall, i.e. a likely contact. Nothing is measured from
/odom.

Outputs: summary.txt (table + paired sign tests of each nav2 arm vs `ours`
over the plans both finished), metrics.png (per-arm box plots), and
tracks_<plan>.png (every arm's track over the map, one panel per arm).
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage, stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(os.path.dirname(HERE), "src", "rudn-ordjo-building", "maps")
INSCRIBED = 0.29          # half the 0.585 m footprint width
MOVED = 0.05              # [m] ground-truth displacement that counts as "started moving"
RES, X0, Y0 = 0.05, -30.0, -30.0   # floor_N.yaml
METRICS = [  # (key, label, lower is better)
    ("final_err", "final error [m]", True),
    ("travel_time", "goal -> end [sim s]", True),
    ("start_delay", "goal -> first motion [sim s]", True),
    ("drive_time", "first motion -> end [sim s]", True),
    ("path_length", "path length [m]", True),
    ("control_energy", "control energy", True),
    ("min_clearance", "min wall clearance [m]", False),
]


def load_map(floor):
    img = plt.imread(os.path.join(MAP_DIR, f"floor_{floor}.png"))
    if img.ndim == 3:
        img = img[..., 0]
    wall = img < 0.1
    edt = ndimage.distance_transform_edt(~wall) * RES
    return img, edt


def clearance(track_xy, edt):
    h = edt.shape[0]
    col = np.clip(((track_xy[:, 0] - X0) / RES).astype(int), 0, edt.shape[1] - 1)
    row = np.clip((h - 1 - (track_xy[:, 1] - Y0) / RES).astype(int), 0, h - 1)
    return float(edt[row, col].min()) - INSCRIBED


def load_rows(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "rows*.jsonl"))):
        for ln in open(p):
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def local(path, d):
    return os.path.join(d, os.path.basename(path)) if path else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--floor", type=int, default=6)
    ap.add_argument("--tolerance", type=float, default=0.5)
    a = ap.parse_args()
    out = a.out or a.dir
    os.makedirs(out, exist_ok=True)
    img, edt = load_map(a.floor)

    rows = load_rows(a.dir)
    tracks = {}
    for r in rows:
        tp = local(r.get("track_path"), a.dir)
        if tp and os.path.isfile(tp):
            d = np.load(tp)
            tr = d["track"]
            tr = tr[tr[:, 0] >= float(d["goal_t"])]
            tracks[(r["plan"], r["arm"])] = (tr, d["plan_poses"])
            if len(tr):
                r["min_clearance"] = round(clearance(tr[:, 1:3], edt), 3)
                moved = np.hypot(*(tr[:, 1:3] - tr[0, 1:3]).T) > MOVED
                if moved.any():
                    r["start_delay"] = round(float(tr[moved.argmax(), 0] - float(d["goal_t"])), 2)
                    if isinstance(r.get("travel_time"), (int, float)):
                        r["drive_time"] = round(r["travel_time"] - r["start_delay"], 2)

    arms = sorted({r["arm"] for r in rows}, key=lambda s: (s != "ours", s))
    by = {(r["plan"], r["arm"]): r for r in rows}
    plans = sorted({r["plan"] for r in rows})

    lines = [f"{len(rows)} rows, {len(plans)} plans, arms: {', '.join(arms)}", ""]
    # A planner-failed run never moved: its final_err is just start-to-goal
    # distance and its other metrics describe standing still. Count it as a
    # non-arrival, but keep it out of every metric mean and sign test --
    # averaging it in inflated `ours` final_err by metres on 2026-09-29.
    def drove(r):
        return r is not None and r.get("outcome") not in ("stack-failed", "planner-failed")

    hdr = f"{'arm':<11}{'n':>4}{'arrived':>9}{'stack-fail':>11}{'plan-fail':>10}" + "".join(
        f"{k[:14]:>16}" for k, _, _ in METRICS)
    lines += [hdr + "   (arrived / reached goal phase; means over runs that DROVE)",
              "-" * len(hdr)]
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        ok = [r for r in rs if r.get("outcome") != "stack-failed"]
        dr = [r for r in ok if drove(r)]
        arr = sum(r.get("outcome") == "arrived" for r in ok)
        cells = []
        for k, _, _ in METRICS:
            v = [r[k] for r in dr if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])]
            cells.append(f"{np.mean(v):>16.3f}" if v else f"{'-':>16}")
        lines.append(f"{arm:<11}{len(rs):>4}{arr:>5}/{len(ok):<3}"
                     f"{len(rs) - len(ok):>11}{len(ok) - len(dr):>10}" + "".join(cells))

    lines += ["", "Paired sign tests vs ours (+ = ours better). arrival: plans where both",
              "reached the goal phase; metrics: plans where both DROVE."]
    for arm in arms:
        if arm == "ours":
            continue
        ad = []
        for p in plans:
            o, b = by.get((p, "ours")), by.get((p, arm))
            if o and b and "stack-failed" not in (o.get("outcome"), b.get("outcome")):
                d = int(o.get("outcome") == "arrived") - int(b.get("outcome") == "arrived")
                if d:
                    ad.append(d)
        if ad:
            ours = sum(d > 0 for d in ad)
            p = stats.binomtest(ours, len(ad)).pvalue
            lines.append(f"  {arm:<11}{'arrival':<16} ours better {ours}/{len(ad)}  p={p:.3f}")
        for k, _, lower in METRICS:
            diffs = []
            for p in plans:
                o, b = by.get((p, "ours")), by.get((p, arm))
                if not drove(o) or not drove(b) or not all(isinstance(x.get(k), (int, float)) for x in (o, b)):
                    continue
                dd = (b[k] - o[k]) if lower else (o[k] - b[k])
                if dd != 0:
                    diffs.append(dd)
            if diffs:
                wins = sum(d < 0 for d in diffs)  # d<0: nav2 arm better
                ours = len(diffs) - wins
                p = stats.binomtest(ours, len(diffs)).pvalue
                lines.append(f"  {arm:<11}{k:<16} ours better {ours}/{len(diffs)}  p={p:.3f}")
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    print(txt)

    fig, axes = plt.subplots(1, len(METRICS) + 1, figsize=(4 * (len(METRICS) + 1), 4))
    rate = [np.mean([by[(p, arm)].get("outcome") == "arrived"
                     for p in plans if (p, arm) in by]) for arm in arms]
    axes[0].bar(arms, rate, color="C0")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("arrival rate (<= %.1f m)" % a.tolerance)
    for ax, (k, label, _) in zip(axes[1:], METRICS):
        data = [[r[k] for r in rows if r["arm"] == arm and isinstance(r.get(k), (int, float))]
                for arm in arms]
        ax.boxplot(data, tick_labels=arms, showmeans=True)
        ax.set_title(label)
        if k == "min_clearance":
            ax.axhline(0, color="r", lw=0.8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "metrics.png"), dpi=110)

    colors = {arm: f"C{i}" for i, arm in enumerate(arms)}
    for p in plans:
        have = [arm for arm in arms if (p, arm) in tracks]
        if not have:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap="gray", extent=(-30, 30, -30, 30), vmin=0, vmax=1)
        plan_xy = tracks[(p, have[0])][1][:, :2]
        ax.plot(*plan_xy.T, "--", color="0.4", lw=1.5, label="reference plan")
        pts = [plan_xy]
        for arm in have:
            tr = tracks[(p, arm)][0]
            r = by[(p, arm)]
            if len(tr):
                ax.plot(tr[:, 1], tr[:, 2], color=colors[arm], lw=2,
                        label=f"{arm}: {r.get('outcome')} {r.get('final_err', float('nan')):.2f} m")
                ax.plot(tr[-1, 1], tr[-1, 2], "x", color=colors[arm], ms=10)
                pts.append(tr[:, 1:3])
        ax.plot(*plan_xy[0], "go", ms=9)
        ax.plot(*plan_xy[-1], "r*", ms=15)
        allp = np.vstack(pts)
        lo, hi = allp.min(0) - 1.5, allp.max(0) + 1.5
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(p)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"tracks_{p}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
