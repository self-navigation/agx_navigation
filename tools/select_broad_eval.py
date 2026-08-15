#!/usr/bin/env python3
"""Pick a broad, geometry-diverse evaluation subset from a plan library.

WHY STRATIFY ON A LABEL WE DO NOT TRUST. The `shape` label baked into each plan
is a `total_abs_turn` / sign-change descriptor, and 2026-08-15 showed it cannot
separate a 180 degree hairpin from two same-sign 90 degree corners (see
figures/2026-08-15/03_uturn_plans.png). That makes it unusable for a per-shape
CLAIM. It remains fine for what it is used for here -- SPREADING a sample so the
subset is not 40 near-copies of one corridor -- because a stratifier only has to
correlate with geometry, not to name it. Length is used as the second axis for
the same reason.

Emits plan paths, one per line, rewritten under --prefix for the remote box.
"""
import argparse, collections, os, glob
import numpy as np


def descriptors(path):
    d = np.load(path)
    p = d["poses"]
    xy = p[:, :2]
    length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    label = str(d["shape"]) if "shape" in d.files else "n/a"
    return label, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="traj_data_v2")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--prefix", default="", help="rewrite dirname to this, e.g. $HOME/traj_data_v2")
    args = ap.parse_args()

    plans = sorted(glob.glob(os.path.join(args.library, "*.npz")))
    by_label = collections.defaultdict(list)
    for p in plans:
        label, length = descriptors(p)
        by_label[label].append((length, p))

    # Proportional to sqrt(population), so the rare labels are not crowded out by
    # the common ones -- the point of the subset is variety, not representativeness.
    weights = {k: len(v) ** 0.5 for k, v in by_label.items()}
    total = sum(weights.values())
    picked = []
    for label, items in sorted(by_label.items()):
        n = max(2, round(args.count * weights[label] / total))
        items.sort()
        # even spread over the length order, deterministic
        idx = np.linspace(0, len(items) - 1, min(n, len(items))).round().astype(int)
        for i in sorted(set(idx.tolist())):
            picked.append((label, items[i][0], items[i][1]))

    for label, length, p in picked:
        out = os.path.join(args.prefix, os.path.basename(p)) if args.prefix else p
        print(out)
    counts = collections.Counter(l for l, _, _ in picked)
    print(f"# {len(picked)} plans: {dict(counts)}", file=os.sys.stderr)
    print(f"# length {min(l for _, l, _ in picked):.1f}-{max(l for _, l, _ in picked):.1f} m",
          file=os.sys.stderr)


if __name__ == "__main__":
    main()
