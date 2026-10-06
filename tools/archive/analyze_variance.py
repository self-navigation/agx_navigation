#!/usr/bin/env python3
"""Compare the within-process and across-process spreads from variance_probe.

Reports each arm as a DISTRIBUTION, never as a single number -- collapsing an
arm to its mean is exactly the mistake that made the tuning result ambiguous in
the first place. Also checks two specific shapes the raw spread cannot show:

  * a TREND against rollout index (within-process only) -- the signature of
    something accumulating, e.g. sim-clock drift. Reported as Spearman rank
    correlation, which does not assume the drift is linear.
  * BIMODALITY -- the signature of a discrete event happening or not, e.g. the
    robot catching a terrain patch edge. A gap statistic: the largest jump
    between consecutive sorted values, relative to the total range. Near 1 means
    the samples fall in two tight clusters rather than scattering.

Usage:  python3 tools/analyze_variance.py tune_data/variance_probe.jsonl
"""

import argparse
import json
import math


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # torn final line from a killed run
            rows.append(rec)
    return rows


def spearman(xs, ys):
    """Rank correlation. Returns None when it cannot be defined."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def gap_statistic(vals):
    """Largest consecutive gap / total range. ~1 = two clusters, ~0 = scattered."""
    if len(vals) < 3:
        return None
    s = sorted(vals)
    rng = s[-1] - s[0]
    if rng <= 0:
        return 0.0
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    return max(gaps) / rng


def describe(vals):
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    s = sorted(vals)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return dict(n=n, mean=mean, sd=sd, min=s[0], max=s[-1], median=med,
                spread=s[-1] - s[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--metric", default="max_cross")
    args = ap.parse_args()

    rows = load(args.path)
    failed = [r for r in rows if "failed" in r or args.metric not in r]
    good = [r for r in rows if "failed" not in r and args.metric in r]
    if failed:
        print(f"!! {len(failed)} failed rollout(s) excluded")
    if not good:
        raise SystemExit("no usable records")

    print(f"metric: {args.metric}   trajectory: {good[0].get('trajectory')}   "
          f"gains: q={good[0].get('q_cross')} r={good[0].get('r_omega')}\n")

    stats = {}
    for mode in ("within", "across"):
        vals = [float(r[args.metric]) for r in good if r.get("mode") == mode]
        if not vals:
            continue
        d = describe(vals)
        stats[mode] = d
        procs = len({r["pid"] for r in good if r.get("mode") == mode})
        print(f"[{mode}]  n={d['n']} in {procs} process(es)")
        print(f"    mean {d['mean']:.3f}   sd {d['sd']:.3f}   "
              f"median {d['median']:.3f}")
        print(f"    min  {d['min']:.3f}   max {d['max']:.3f}   "
              f"SPREAD {d['spread']:.3f} m")
        g = gap_statistic(vals)
        if g is not None:
            print(f"    bimodality (gap/range) {g:.2f}"
                  f"{'   <- two clusters, not scatter' if g > 0.5 else ''}")
        if mode == "within":
            seq = [r for r in good if r.get("mode") == "within"]
            seq.sort(key=lambda r: r["index"])
            rho = spearman([r["index"] for r in seq],
                           [float(r[args.metric]) for r in seq])
            if rho is not None:
                verdict = ("consistent with accumulating drift"
                           if abs(rho) > 0.6 else "no monotone trend with index")
                print(f"    trend vs. rollout index: rho={rho:+.2f}  ({verdict})")
        print(f"    values: {[round(v, 3) for v in vals]}\n")

    if "within" in stats and "across" in stats:
        w, a = stats["within"]["spread"], stats["across"]["spread"]
        print(f"within spread {w:.3f} m  vs  across spread {a:.3f} m")
        ratio = w / a if a > 0 else float("inf")
        if 0.5 <= ratio <= 2.0:
            print("  -> COMPARABLE. The noise is not caused by running many\n"
                  "     rollouts in one process; a fresh process is just as\n"
                  "     variable. Repeats are needed per measurement, but a\n"
                  "     long-lived search is not systematically biased by order.")
        elif ratio > 2.0:
            print("  -> WITHIN is worse. Something accumulates across rollouts\n"
                  "     in one process; multi-rollout tools measure a moving\n"
                  "     target and their run order contaminates the ranking.")
        else:
            print("  -> ACROSS is worse. Per-process world setup dominates;\n"
                  "     a single process is internally consistent.")


if __name__ == "__main__":
    main()
