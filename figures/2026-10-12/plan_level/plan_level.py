#!/usr/bin/env python3
"""Headline pairs re-tested at PLAN level (seeds of one plan share the plan, so (plan, seed) units are not independent).

Per plan: the contact fraction over its seeds in each arm. Tests over plans: a sign test on plans whose fraction differs,
and a Wilcoxon signed-rank on the per-plan differences. usage: plan_level.py [JOB:CELL JOB:CELL ...]
"""
import glob, json, os, sys, collections
import numpy as np
from scipy.stats import binomtest, wilcoxon

RD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../run_data"))
J5, J6, J7, J8 = ("2026-10-12_job255_steerable", "2026-10-12_job256_steerable-rtab",
                  "2026-10-12_job257_kprior", "2026-10-12_job258_estpose")
PAIRS = [(f"{J5}:QR", f"{J5}:Q0"), (f"{J6}:RR", f"{J6}:RQ"), (f"{J5}:TR", f"{J5}:T"),
         (f"{J5}:Q0", f"{J7}:PQ"), (f"{J6}:RQ", f"{J7}:PR"), (f"{J5}:T", f"{J7}:PT"),
         (f"{J5}:QR", f"{J7}:FQ"), (f"{J6}:RR", f"{J7}:FR"), (f"{J5}:TR", f"{J7}:FT"),
         (f"{J7}:FT", f"{J8}:OF"), (f"{J8}:OF", f"{J8}:OT"), (f"{J7}:PR", f"{J8}:ER")]
a = sys.argv[1:]
if a: PAIRS = list(zip(a[::2], a[1::2]))


def frac(spec):
    """spec = JOB:CELL, or several joined by '+' to POOL plan sets (plan names are disjoint between test and dev)."""
    d = collections.defaultdict(dict)
    for one in spec.split("+"):
        frac1(one, d)
    return d


def frac1(spec, d):
    job, cell = spec.rsplit(":", 1)
    for f in glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l); d[os.path.basename(r["plan"])][r["seed"]] = (r.get("wall_overlap_s") or 0) > 0


out = []
for A, B in PAIRS:
    fa, fb = frac(A), frac(B)
    plans = sorted(p for p in set(fa) & set(fb) if set(fa[p]) & set(fb[p]))
    da = np.array([np.mean([fa[p][s] for s in set(fa[p]) & set(fb[p])]) for p in plans])
    db = np.array([np.mean([fb[p][s] for s in set(fa[p]) & set(fb[p])]) for p in plans])
    better_b, better_a = int((db < da).sum()), int((da < db).sum())
    ps = binomtest(better_b, better_b + better_a).pvalue if better_a + better_b else 1.0
    pw = wilcoxon(da, db).pvalue if (da != db).any() else 1.0
    lab = lambda X: "+".join(f"{x.rsplit(':', 1)[1]}({x.split('_job')[1][:3]})" for x in X.split("+"))
    out.append(f"{lab(A)} vs {lab(B)}  plans {len(plans)} | contact runs {int(sum(len(fa[p]) * a_ for p, a_ in zip(plans, da)))} vs {int(sum(len(fb[p]) * b_ for p, b_ in zip(plans, db)))}"
               f" | plans with any contact {int((da > 0).sum())} vs {int((db > 0).sum())}"
               f" | B-better/A-better plans {better_b}/{better_a} sign p {ps:.2g} | Wilcoxon p {pw:.2g}")
print("\n".join(out))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("OUT", "plan_level") + ".txt"), "w").write("\n".join(out) + "\n")
