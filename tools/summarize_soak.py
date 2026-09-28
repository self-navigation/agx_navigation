#!/usr/bin/env python3
"""Per-plan, per-arm summary of a soak JSONL, plus paired sign tests between arms.

One file = one campaign. Never feed this rows from two campaigns and put them in
one table: cross-run comparability is an assumption we do not hold (CLAUDE.md,
"Reading a gain result").

An arm is `identity` or `tvlqr q/r`. Per plan and arm it prints the mean and sd
of max|e_cross|, the bad-mode rate (share of rollouts with max|e_cross| above
--bad), mean final_err, the miss rate (final_err > 0.5 m) and the J parts. The
identity arm's j_control is 0 by definition, so J alone flatters open loop: read
j_tracking + j_terminal and arrival beside it.

Across plans it prints, per arm: arithmetic mean of max_cross / final_err,
geometric mean of J, and for every pair of arms a two-sided sign test over the
per-plan means (ties dropped).

    python3 tools/summarize_soak.py soak_data/soak_seven_three_arms.jsonl --bad 1.5
"""
import argparse
import collections
import itertools
import json
import math
import statistics


def arm_of(row):
    if row["corrector"] == "identity":
        return "identity"
    return f"{row['corrector']} {row['q_cross']:g}/{row['r_omega']:g}"


def sign_test(a, b):
    """Wins of a (lower is better), losses, two-sided binomial p."""
    wins = sum(x < y for x, y in zip(a, b))
    losses = sum(x > y for x, y in zip(a, b))
    n = wins + losses
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return wins, losses, p


def geo(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jsonl")
    ap.add_argument("--bad", type=float, default=1.0,
                    help="max|e_cross| threshold of the bad mode, m (default 1.0)")
    ap.add_argument("--miss", type=float, default=0.5, help="final_err miss threshold, m")
    ap.add_argument("--quiet-plans", action="store_true", help="aggregate only")
    args = ap.parse_args()

    cells = collections.defaultdict(list)
    for line in open(args.jsonl):
        r = json.loads(line)
        vals = (r["max_cross"], r["final_err"], r["j_total"])
        if not all(math.isfinite(v) for v in vals):
            continue  # a failed rollout invalidates its sample; never average survivors
        cells[(r["trajectory"], arm_of(r))].append(r)

    plans = sorted({p for p, _ in cells})
    arms = sorted({a for _, a in cells}, key=lambda a: (a != "identity", a))
    per = {}  # (plan, arm) -> dict of per-plan statistics
    for key, rows in cells.items():
        mc = [r["max_cross"] for r in rows]
        fe = [r["final_err"] for r in rows]
        per[key] = dict(
            n=len(rows),
            mc=statistics.fmean(mc), sd=statistics.pstdev(mc),
            bad=sum(x > args.bad for x in mc) / len(mc),
            fe=statistics.fmean(fe),
            miss=sum(x > args.miss for x in fe) / len(fe),
            j=statistics.fmean(r["j_total"] for r in rows),
            jt=statistics.fmean(r["j_tracking"] for r in rows),
            jc=statistics.fmean(r["j_control"] for r in rows),
            jf=statistics.fmean(r["j_terminal"] for r in rows),
        )

    if not args.quiet_plans:
        print(f"{'plan':15} {'arm':18} {'n':>4} {'max_cross':>15} {'bad>'+str(args.bad):>8} "
              f"{'final_err':>9} {'miss':>6} {'J':>8} {'J_trk':>8} {'J_ctl':>7} {'J_term':>7}")
        for p in plans:
            for a in arms:
                s = per.get((p, a))
                if s is None:
                    continue
                print(f"{p:15} {a:18} {s['n']:4d} {s['mc']:7.3f} ± {s['sd']:5.3f} "
                      f"{s['bad']:8.1%} {s['fe']:9.3f} {s['miss']:6.1%} {s['j']:8.2f} "
                      f"{s['jt']:8.2f} {s['jc']:7.2f} {s['jf']:7.2f}")
            print()

    complete = [p for p in plans if all((p, a) in per for a in arms)]
    if len(complete) < len(plans):
        print(f"# {len(plans) - len(complete)} plan(s) lack some arm; aggregates use {len(complete)}")
    print(f"{'arm':18} {'mean max_cross':>14} {'mean final_err':>14} {'miss':>6} "
          f"{'geo J':>8} {'geo J_trk+term':>14}")
    for a in arms:
        ss = [per[(p, a)] for p in complete]
        print(f"{a:18} {statistics.fmean(s['mc'] for s in ss):14.3f} "
              f"{statistics.fmean(s['fe'] for s in ss):14.3f} "
              f"{statistics.fmean(s['miss'] for s in ss):6.1%} {geo(s['j'] for s in ss):8.2f} "
              f"{geo(s['jt'] + s['jf'] for s in ss):14.2f}")

    print("\nsign tests over plans (wins = first arm lower):")
    for a, b in itertools.combinations(arms, 2):
        for metric in ("mc", "fe", "j"):
            w, l, p = sign_test([per[(q, a)][metric] for q in complete],
                                [per[(q, b)][metric] for q in complete])
            print(f"  {a:18} vs {b:18} {metric:3} {w:3d}/{w + l:<3d} p={p:.4f}")


if __name__ == "__main__":
    main()
