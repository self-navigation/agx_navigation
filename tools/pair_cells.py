#!/usr/bin/env python3
"""Paired comparison of compare_run cells over the plans they share.

Compares arrival, contact runs (wall_overlap_s > 0), final_err and max
cross-track, each with a sign test over the discordant plans.

    python3 tools/pair_cells.py <job dir> A:B [A:B ...] [--seed 0]

Per pair A:B (A = reference, B = treatment) prints n, arrivals A->B
(gained/lost, p), contact runs A->B (fixed/new, p), median final_err and how
often B is better or worse by > 2 cm (p), and the same for max cross-track. A
plan that failed to run (no row, or stack_up_ok false) is dropped from that
pair.
"""
import argparse, json, math, os, statistics as st
from scipy.stats import binomtest


def load(d, c, seed):
    f = os.path.join(d, c, f"rows.s{seed}.jsonl")
    out = {}
    for line in open(f):
        r = json.loads(line)
        if r.get("stack_up_ok") is False:
            continue
        out[r["plan"]] = r
    return out


def num(r, k, bad=99.0):
    v = r.get(k)
    return v if isinstance(v, (int, float)) and math.isfinite(v) else bad


def p2(a, b):
    return binomtest(a, a + b).pvalue if a + b else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("pairs", nargs="+"); ap.add_argument("--seed", default="0")
    a = ap.parse_args()
    print("A -> B | n | arrived A->B (+gained/-lost, p) | contact runs A->B (fixed/new, p) | "
          "final_err med A/B, B better/worse >2cm (p) | max_cross med A/B, better/worse >2cm (p)")
    for pr in a.pairs:
        ca, cb = pr.split(":")
        A, B = load(a.dir, ca, a.seed), load(a.dir, cb, a.seed)
        P = sorted(set(A) & set(B))
        arr = lambda r: r.get("outcome") == "arrived"
        con = lambda r: num(r, "wall_overlap_s", 0.0) > 0
        g = sum(arr(B[p]) and not arr(A[p]) for p in P); l = sum(arr(A[p]) and not arr(B[p]) for p in P)
        fx = sum(con(A[p]) and not con(B[p]) for p in P); nw = sum(con(B[p]) and not con(A[p]) for p in P)
        def sgn(k):
            b = sum(num(B[p], k) < num(A[p], k) - 0.02 for p in P); w = sum(num(B[p], k) > num(A[p], k) + 0.02 for p in P)
            return (f"{st.median(num(A[p], k) for p in P):.3f}/{st.median(num(B[p], k) for p in P):.3f}, "
                    f"{b}/{w} (p={p2(b, w):.3g})")
        print(f"{ca} -> {cb} | {len(P)} | {sum(arr(A[p]) for p in P)}->{sum(arr(B[p]) for p in P)} "
              f"(+{g}/-{l}, p={p2(g, l):.3g}) | {sum(con(A[p]) for p in P)}->{sum(con(B[p]) for p in P)} "
              f"({fx}/{nw}, p={p2(fx, nw):.3g}) | {sgn('final_err')} | {sgn('max_cross_track')}")


if __name__ == "__main__":
    main()
