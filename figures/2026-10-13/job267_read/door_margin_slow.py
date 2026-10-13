"""Section 3 of door_margin.py on job 265, the slow (0.448 m/s) realistic-tyre twin: does the outward deviation scale ~v^2?
usage: door_margin_slow.py > door_margin_slow.txt"""
import os, sys
import numpy as np
from scipy.stats import wilcoxon
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "door_margin.py")).read().split('\nJ = "2026')[0]
g = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "door_margin.py")}; exec(src, g)
J = "2026-10-12_job265_tyre-grid"
for base, other in [(b, o) for b, os_ in {"QR": ["Q0", "PQ", "FQ"], "RR": ["RQ", "PR", "FR"], "TR": ["T", "PT", "FT"]}.items() for o in os_]:
    B = {k: np.nanmean([x[1] for x in v]) for k, v in g["per_door"](f"{J}/test:{base}+{J}/dev:{base}").items()}
    A = {k: np.nanmean([x[1] for x in v]) for k, v in g["per_door"](f"{J}/test:{other}+{J}/dev:{other}").items()}
    ks = [k for k in B if k in A and np.isfinite(A[k]) and np.isfinite(B[k])]; d = np.array([A[k] - B[k] for k in ks])
    print(f"job 265 {base} median {np.nanmedian([B[k] for k in ks]):+.3f} | {other} median {np.nanmedian([A[k] for k in ks]):+.3f}; paired diff {np.median(d):+.4f} over {len(ks)} plan-doors, p {wilcoxon(d).pvalue:.3g}")
