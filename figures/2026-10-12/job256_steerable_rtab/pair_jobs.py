#!/usr/bin/env python3
"""Pair cells across jobs by (plan, seed): contact (wall_overlap_s > 0) and arrival flips, sign tests.

usage: pair_jobs.py JOB_A:CELL_A JOB_B:CELL_B [...more pairs]
Each pair prints A vs B; fixed = contact in B and not in A, new = contact in A and not in B.
"""
import glob, json, os, sys
from scipy.stats import binomtest


def load(job, cell):
    rows = {}
    for f in glob.glob(os.path.join(job, cell, "rows.s*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[(os.path.basename(r["plan"]), r["seed"])] = r
    return rows


def p(a, b):
    return binomtest(a, a + b).pvalue if a + b else 1.0


args = sys.argv[1:]
for i in range(0, len(args), 2):
    (ja, ca), (jb, cb) = (s.rsplit(":", 1) for s in args[i:i + 2])
    A, B = load(ja, ca), load(jb, cb)
    keys = sorted(set(A) & set(B))
    con = lambda r: r.get("wall_overlap_s", 0) > 0
    arr = lambda r: r["outcome"] == "arrived"
    fx = sum(con(B[k]) and not con(A[k]) for k in keys); nw = sum(con(A[k]) and not con(B[k]) for k in keys)
    af = sum(arr(A[k]) and not arr(B[k]) for k in keys); an = sum(arr(B[k]) and not arr(A[k]) for k in keys)
    print(f"{os.path.basename(ja)}:{ca} vs {os.path.basename(jb)}:{cb}  n {len(keys)} | contact "
          f"{sum(con(A[k]) for k in keys)} vs {sum(con(B[k]) for k in keys)}  A-better/B-better {fx}/{nw} p {p(fx, nw):.2g}"
          f" | arrived {sum(arr(A[k]) for k in keys)} vs {sum(arr(B[k]) for k in keys)}  {af}/{an} p {p(af, an):.2g}")
