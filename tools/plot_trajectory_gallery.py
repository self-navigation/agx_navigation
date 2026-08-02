#!/usr/bin/env python3
"""Contact sheet of every recorded PMP trajectory, for picking evaluation cases by eye.

`classify_plans.py` ranks plans by numeric shape descriptors; this renders them,
because a descriptor tuple does not tell you that the "S-CURVE" you selected is
really an L with a rounded corner (which is exactly what floor_6_00042 turned
out to be, after a whole corrector comparison had been built on it).

ROTATION
--------
Each path is rotated so its FIRST PRINCIPAL AXIS is horizontal, then flipped so
it runs left-to-right and its net turn goes upward. Without this, a 12 m path
lying diagonally wastes most of its cell and reads as smaller and straighter
than the same path drawn horizontally, so the eye compares the map frame rather
than the shape.

The rotation is DISPLAY ONLY. Nothing downstream consumes it: the .npz files are
untouched, and a trajectory selected here is replayed in its original frame.
Aspect ratio is equal within each cell, so a corner still looks like a corner.

OFFLINE TOOL (matplotlib in a venv), same rule as plot_run.py.

    .venv/bin/python tools/plot_trajectory_gallery.py traj_data --out figures
"""

import argparse
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Same palette idea as the corrector figures: a class keeps its colour so two
# sheets can be compared without re-reading the legend.
SHAPE_COLORS = {
    "STRAIGHT": "#eb6834",
    "S-CURVE": "#1baf7a",
    "CORNER": "#2a78d6",
    "curved": "#9b59b6",
    "gentle": "#8a8985",
}


def resample(pts, step=0.15):
    """Arc-length resample, so curvature is not dominated by sample density.

    Not optional. Skipping it reclassified this library as 57 S-CURVE / 1
    STRAIGHT instead of 15 / 12: densely-sampled slow sections accumulate enough
    small alternating heading deltas to trip the sign-change test.
    """
    out = [(pts[0][0], pts[0][1])]
    acc = 0.0
    for i in range(1, len(pts)):
        acc += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if acc >= step:
            out.append((pts[i][0], pts[i][1]))
            acc = 0.0
    last = (pts[-1][0], pts[-1][1])
    if out[-1] != last:
        out.append(last)
    return out


def trim_pivot(raw, min_travel=0.30):
    """Drop the in-place reorientation the PMP planner puts at the start.

    A plan typically begins by spinning on the spot to face the path: a large
    heading change over ~no distance. Every shape descriptor is computed from
    heading deltas, so that pivot dominates them -- which is why the automatic
    labels called 58 of 100 plans CORNER when the gallery shows most of them are
    visually straight lines. Discarding the leading samples until the robot has
    actually travelled `min_travel` measures the shape of the PATH rather than
    the shape of the pirouette.

    Selection/display only, like the rotation: the .npz is untouched and a
    trajectory is replayed in full. This is deliberately NOT mirrored into
    classify_plans.py, which reports the raw descriptors.
    """
    acc = 0.0
    for i in range(1, len(raw)):
        acc += math.hypot(raw[i][0] - raw[i - 1][0], raw[i][1] - raw[i - 1][1])
        if acc >= min_travel:
            # Keep the true start point so the drawn path still begins where the
            # robot did; only the descriptor window moves.
            return raw[i - 1:]
    return raw


def descriptors(raw):
    """Shape descriptors -- kept numerically identical to tools/classify_plans.py.

    Duplicated rather than imported: classify_plans is a standalone script with
    its own argparse main, and this file must stay runnable from a bare venv.
    If the thresholds there change, change them here in the same edit.
    """
    pts = resample(raw)
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


def canonical(xy):
    """Rotate to principal axis, oriented left-to-right and turning upward."""
    p = np.asarray(xy, dtype=float)
    p = p - p.mean(axis=0)
    # Principal axis via the covariance eigenvector -- robust for a path whose
    # start and end nearly coincide, where a start->end vector says nothing.
    _, vecs = np.linalg.eigh(np.cov(p.T))
    axis = vecs[:, -1]
    rot = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
    p = p @ rot.T
    if p[-1, 0] < p[0, 0]:            # run left-to-right
        p = p * np.array([-1.0, 1.0])
    # Flip so the dominant excursion is upward, so two mirror-image corners do
    # not read as different shapes.
    if abs(p[:, 1].min()) > abs(p[:, 1].max()):
        p = p * np.array([1.0, -1.0])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir", help="directory of recorded .npz plans")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--name", default="trajectory_gallery")
    ap.add_argument("--per-page", type=int, default=24,
                    help="plans per PNG. One sheet of 100 shrinks each cell to "
                         "the point where a gentle S and a straight line are "
                         "indistinguishable, which is how the current eval set "
                         "came to include an L labelled S-CURVE. 0 = one sheet.")
    ap.add_argument("--sort", default="curvature",
                    choices=("curvature", "shape", "name"),
                    help="'curvature' puts the most interesting plans on page 1 "
                         "(most total turning per metre); 'shape' groups by class")
    ap.add_argument("--raw-descriptors", action="store_true",
                    help="do not trim the leading in-place pivot before "
                         "measuring shape (see trim_pivot)")
    ap.add_argument("--no-rotate", action="store_true",
                    help="draw in the map frame instead of the principal axis")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    entries = []
    for path in sorted(glob.glob(os.path.join(args.traj_dir, "*.npz"))):
        poses = np.load(path)["poses"]
        raw = [(p[0], p[1]) for p in poses]
        window = raw if args.raw_descriptors else trim_pivot(raw)
        d = descriptors(window)
        if d is None:
            continue
        entries.append((os.path.basename(path)[:-4], poses[:, :2], d, label(d)))

    if not entries:
        raise SystemExit(f"no .npz plans in {args.traj_dir}")

    if args.sort == "shape":
        order = {k: i for i, k in enumerate(
            ["STRAIGHT", "S-CURVE", "CORNER", "curved", "gentle"])}
        entries.sort(key=lambda e: (order.get(e[3], 9), -e[2]["total_abs"]))
    elif args.sort == "curvature":
        # Turning PER METRE, not total: a long plan accumulates turning just by
        # being long, and the interesting cases are the ones that turn a lot in
        # a short distance -- which is also what stresses a corrector.
        entries.sort(key=lambda e: -(e[2]["total_abs"] / max(e[2]["L"], 1e-6)))

    per_page = args.per_page if args.per_page > 0 else len(entries)
    n_pages = math.ceil(len(entries) / per_page)
    cols = args.cols

    for page in range(n_pages):
        chunk = entries[page * per_page:(page + 1) * per_page]
        rows = math.ceil(len(chunk) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.set_axis_off()

        for i, (name, xy, d, shape) in enumerate(chunk):
            ax = axes[i // cols][i % cols]
            p = np.asarray(xy, float) if args.no_rotate else canonical(xy)
            col = SHAPE_COLORS.get(shape, "#8a8985")
            ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.8)
            ax.plot(p[0, 0], p[0, 1], "o", color=col, ms=5)
            ax.plot(p[-1, 0], p[-1, 1], "*", color=col, ms=11)
            # Equal aspect per cell: a corner must not be flattened into a bend.
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(
                f"{name}\n{shape}  {d['L']:.1f} m  "
                f"turn/m {d['total_abs'] / max(d['L'], 1e-6):.2f}",
                fontsize=7.5, color=col, pad=3)
            ax.set_axis_off()

        suffix = "" if n_pages == 1 else f"_p{page + 1:02d}"
        fig.suptitle(
            f"PMP plans {page * per_page + 1}-{page * per_page + len(chunk)} "
            f"of {len(entries)}, sorted by {args.sort} — rotated to principal "
            "axis, equal aspect per cell (● start, ★ goal)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = os.path.join(args.out, f"{args.name}{suffix}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}  ({len(chunk)} plans)")

    counts = {}
    for _, _, _, s in entries:
        counts[s] = counts.get(s, 0) + 1
    print("  " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())))

    print("\nmost turning per metre (candidate hard cases):")
    for name, _, d, shape in entries[:12]:
        print("  %-18s %-9s %5.1f m  turn/m %.2f  net_turn %+.2f  signs %d"
              % (name, shape, d["L"], d["total_abs"] / max(d["L"], 1e-6),
                 d["net_turn"], d["sign_changes"]))


if __name__ == "__main__":
    main()
