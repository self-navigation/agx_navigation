#!/usr/bin/env python3
"""Score a whole traced sweep in `J`, one arm/shape tree at a time.

WHY THIS EXISTS SEPARATELY FROM score_epsilon.py
------------------------------------------------
`score_epsilon.py` scores traces against ONE plan (`--plan`), which is the right
shape for looking at a single trajectory closely. A sweep is the other case: a
`<root>/<arm>/<shape>/*.csv` tree over dozens of plans, where the plan is
implied by the directory name and pairing them by hand is both tedious and the
easiest place to get a silent mismatch -- scoring shape A's track against shape
B's plan produces large, plausible, meaningless numbers rather than an error.

So this walks the tree, pairs each shape directory with `<plans>/<shape>.npz` by
name, and refuses rather than guesses when a plan is missing.

It reuses `score_epsilon.score_trace` unchanged; the functional itself lives in
`tuning/epsilon.py` (pure, unit-tested). Nothing here re-implements the cost.

OFFLINE ONLY, and needs NO SIM -- it reads recorded CSVs, so it can run while
the machine is busy driving something else.

    .venv/bin/python tools/score_sweep.py libsweep --plans traj_data \\
        --out epsilon_data/libsweep.jsonl

READING THE OUTPUT
------------------
`J` is an UPPER BOUND on `epsilon`, never `epsilon` itself: `epsilon` is
`J[u] - J*[z]`, and `J*` is unknown and strictly positive under slip. Report it
as such. The per-shape table also carries `max|e_cross|` so the two metrics can
be compared directly -- they disagreed on 3 of 7 shapes on 2026-08-13, which is
the whole reason this is worth computing.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_epsilon import score_trace  # noqa: E402


def discover_from_jsonl(path, trace_root=None):
    """soak.py rows -> {arm: {shape: [csv, ...]}}, keyed on each row's `trace`.

    PREFER THIS over `discover()` wherever the rows carry a trace path. The
    directory layout is a convention and conventions drift (soak.py writes one
    flat dir with the gains in the FILENAME; variance_probe writes
    <arm>/<shape>/), but the row records where its own trace went, so the
    pairing cannot silently go wrong.

    Rows with no `trace` key are untraced (`--trace-every N`) and are skipped
    rather than guessed at. `trace_root` re-points the recorded absolute path
    at a local copy, since traces are usually scored after being fetched.
    """
    tree = {}
    n_untraced = 0
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("failed") or "trace" not in row:
                n_untraced += 1
                continue
            trace = row["trace"]
            if trace_root:
                trace = os.path.join(trace_root, os.path.basename(trace))
            if not os.path.exists(trace):
                n_untraced += 1
                continue
            arm = f"q{row['q_cross']:g}_r{row['r_omega']:g}"
            tree.setdefault(arm, {}).setdefault(row["trajectory"], []).append(trace)
    if n_untraced:
        print(f"[score_sweep] skipped {n_untraced} rows with no readable trace",
              file=sys.stderr)
    return {a: {s: sorted(v) for s, v in shapes.items()}
            for a, shapes in tree.items()}


def discover(root):
    """`<root>/<arm>/<shape>/*.csv` -> {arm: {shape: [csv, ...]}}.

    Sorted throughout so a re-run lists rollouts in the same order, which makes
    two score files diffable.
    """
    tree = {}
    for arm in sorted(os.listdir(root)):
        arm_dir = os.path.join(root, arm)
        if not os.path.isdir(arm_dir):
            continue
        shapes = {}
        for shape in sorted(os.listdir(arm_dir)):
            csvs = sorted(glob.glob(os.path.join(arm_dir, shape, "*.csv")))
            if csvs:
                shapes[shape] = csvs
        if shapes:
            tree[arm] = shapes
    return tree


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?",
                    help="sweep trace root, laid out <arm>/<shape>/*.csv "
                         "(variance_probe layout). Omit when using --from-jsonl.")
    ap.add_argument("--from-jsonl",
                    help="a soak.py JSONL; pair traces via each row's `trace` "
                         "field instead of the directory layout. Preferred "
                         "whenever the rows have it.")
    ap.add_argument("--trace-root",
                    help="with --from-jsonl, look for each trace's basename "
                         "here instead of at its recorded absolute path "
                         "(traces are normally scored after being fetched).")
    ap.add_argument("--plans", default="traj_data",
                    help="directory of <shape>.npz plans (default: traj_data)")
    ap.add_argument("--out", help="JSONL to write per-rollout scores to")
    ap.add_argument("--quiet", action="store_true",
                    help="per-shape table only, no per-rollout lines")
    args = ap.parse_args()

    if bool(args.root) == bool(args.from_jsonl):
        raise SystemExit("[score_sweep] give exactly one of <root> or --from-jsonl")
    if args.from_jsonl:
        tree = discover_from_jsonl(args.from_jsonl, args.trace_root)
        source = args.from_jsonl
    else:
        tree = discover(args.root)
        source = args.root
    if not tree:
        raise SystemExit(f"[score_sweep] no scoreable traces found via {source}")

    # Resolve every plan BEFORE scoring anything: a sweep is dozens of minutes of
    # machine time upstream, and discovering a missing plan halfway through means
    # a partial table that looks complete.
    shapes = sorted({s for arm in tree.values() for s in arm})
    plans = {}
    missing = []
    for shape in shapes:
        path = os.path.join(args.plans, f"{shape}.npz")
        if os.path.exists(path):
            plans[shape] = path
        else:
            missing.append(shape)
    if missing:
        raise SystemExit(f"[score_sweep] no plan in {args.plans} for: "
                         + ", ".join(missing))

    out = open(args.out, "a") if args.out else None
    # arm -> shape -> (list of J, list of max_cross)
    table = {arm: {} for arm in tree}
    n_scored = n_skipped = 0
    for arm, arm_shapes in tree.items():
        for shape, csvs in arm_shapes.items():
            plan = np.load(plans[shape])
            dt_sample = float(plan["dt_sample"])
            js, crosses, finals = [], [], []
            for f in csvs:
                try:
                    score, max_cross, final_err = score_trace(f, plan, dt_sample)
                except ValueError as exc:
                    # A truncated trace (the soak was killed mid-rollout) is
                    # normal and must not abort the table.
                    print(f"[skip] {arm}/{shape}/{os.path.basename(f)}: {exc}",
                          file=sys.stderr)
                    n_skipped += 1
                    continue
                js.append(score.j_total)
                crosses.append(max_cross)
                finals.append(final_err)
                n_scored += 1
                rec = dict(score.as_dict(), trace=os.path.basename(f), arm=arm,
                           shape=shape, max_cross=max_cross, final_err=final_err,
                           plan=os.path.basename(plans[shape]))
                if out:
                    out.write(json.dumps(rec) + "\n")
                if not args.quiet:
                    print(f"{arm:10s} {shape:15s} {os.path.basename(f):40s} "
                          f"J={score.j_total:9.2f} max|e_cross|={max_cross:.3f}")
            if js:
                table[arm][shape] = (np.array(js), np.array(crosses),
                                     np.array(finals))
    if out:
        out.close()

    arms = list(tree)
    print(f"\n{'shape':<16}" + "".join(
        f"{a + ' J':>14}{a + ' m':>12}" for a in arms))
    for shape in shapes:
        row = f"{shape:<16}"
        for arm in arms:
            if shape in table[arm]:
                js, crosses, _ = table[arm][shape]
                row += f"{js.mean():14.2f}{crosses.mean():12.3f}"
            else:
                row += f"{'-':>14}{'-':>12}"
        print(row)

    # Aggregate over shapes, not over rollouts: every shape must weigh the same
    # regardless of how many repeats it happened to get, or a shape that failed
    # half its rollouts quietly counts less than the rest.
    print(f"{'MEAN':<16}", end="")
    for arm in arms:
        per_shape_j = [table[arm][s][0].mean() for s in shapes if s in table[arm]]
        per_shape_c = [table[arm][s][1].mean() for s in shapes if s in table[arm]]
        print(f"{np.mean(per_shape_j):14.2f}{np.mean(per_shape_c):12.3f}", end="")
    print(f"\n\nscored {n_scored} rollouts over {len(shapes)} shapes "
          f"x {len(arms)} arms" + (f", skipped {n_skipped}" if n_skipped else ""))
    print("J is an UPPER BOUND on epsilon (J* > 0 and unknown), not epsilon.")


if __name__ == "__main__":
    main()
