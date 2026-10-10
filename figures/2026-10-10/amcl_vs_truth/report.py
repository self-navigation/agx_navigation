#!/usr/bin/env python3
"""Numbers for the amcl vs truth comparison from stats.json (render.py).
    .venv/bin/python figures/2026-10-10/amcl_vs_truth/report.py
"""
import json
import os
import sys

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
from render import classify, sign_p  # noqa: E402

S = json.load(open(os.path.join(OUT, "stats.json")))
CELLS = ("LTNq", "LAN", "LANq", "LANqe")


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


def q(xs, p):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else float("nan")


print("== 1. localization error of the pose the corrector used (per-run medians, then the median over runs)")
for c in CELLS:
    V = list(S[c].values())
    gl = [v["gain_lag"] for v in V if v["gain_lag"]]
    print(f"{c:6s} n={len(V):3d}  |err| med {med([v['loc_med'] for v in V]):.3f}  p95 {med([v['loc_p95'] for v in V]):.3f}  "
          f"max {med([v['loc_max'] for v in V]):.3f} (p90 of max {q([v['loc_max'] for v in V], 90):.2f})  "
          f"along {med([v['loc_along_med'] for v in V]):.3f} cross {med([v['loc_cross_med'] for v in V]):.3f}  "
          f"yaw {np.degrees(med([v['loc_yaw_med'] for v in V])):.1f} deg (p95 {np.degrees(med([v['loc_yaw_p95'] for v in V])):.1f})")
    print(f"       first 3 s {med([v['loc_first3_med'] for v in V]):.3f}  last quarter {med([v['loc_last_q_med'] for v in V]):.3f}  "
          f"turning {med([v['loc_turn_med'] for v in V]):.3f} straight {med([v['loc_straight_med'] for v in V]):.3f}  "
          f"on patch {med([v['loc_patch_med'] for v in V]):.3f} off patch {med([v['loc_nopatch_med'] for v in V]):.3f}")
    print(f"       ticks >0.2 m {100*np.mean([v['frac_gt_020'] for v in V]):.0f}%  >0.5 m {100*np.mean([v['frac_gt_050'] for v in V]):.0f}%  "
          f"jumps/run >5 cm med {med([v['jumps_5cm'] for v in V]):.0f} (p90 {q([v['jumps_5cm'] for v in V], 90):.0f})  >10 cm med {med([v['jumps_10cm'] for v in V]):.0f} "
          f"(runs with any {sum(v['jumps_10cm'] > 0 for v in V)})  max jump med {med([v['jump_max'] for v in V]):.3f}")
    print(f"       believed/true e_cross fit: gain med {med([g[0] for g in gl]):.2f} (IQR {q([g[0] for g in gl],25):.2f}-{q([g[0] for g in gl],75):.2f})  "
          f"lag med {med([g[1] for g in gl]):.1f} s (IQR {q([g[1] for g in gl],25):.1f}-{q([g[1] for g in gl],75):.1f})  R2 med {med([g[2] for g in gl]):.2f}  n={len(gl)}")
    print(f"       max|e_cross| true med {med([v['max_ec_true'] for v in V]):.3f}  believed med {med([v['max_ec_bel'] for v in V]):.3f}  "
          f"sat frac med {med([v['sat_frac'] for v in V]):.2f}")

# per-seed terrain
print("\n   per seed (LANq): |err| med / frac>0.2 / contacts")
for sd in ("s0", "s1", "s2"):
    V = [v for k, v in S["LANq"].items() if k.endswith("/" + sd)]
    print(f"   {sd}: {med([v['loc_med'] for v in V]):.3f} / {100*np.mean([v['frac_gt_020'] for v in V]):.0f}% / {sum(v['contact'] for v in V)}/{len(V)}")

print("\n== 2. paired contact table LTNq vs LANq (same plan/seed)")
keys = sorted(set(S["LTNq"]) & set(S["LANq"]))
both = [k for k in keys if S["LTNq"][k]["contact"] and S["LANq"][k]["contact"]]
amcl_only = [k for k in keys if not S["LTNq"][k]["contact"] and S["LANq"][k]["contact"]]
truth_only = [k for k in keys if S["LTNq"][k]["contact"] and not S["LANq"][k]["contact"]]
neither = [k for k in keys if not S["LTNq"][k]["contact"] and not S["LANq"][k]["contact"]]
print(f"n={len(keys)} both {len(both)}  amcl-only {len(amcl_only)}  truth-only {len(truth_only)}  neither {len(neither)}")
w = sum(S["LANq"][k]["max_ec_true"] > S["LTNq"][k]["max_ec_true"] for k in keys)
print(f"true max|e_cross| larger under amcl: {w}/{len(keys)} (p {sign_p(w, len(keys)-w):.3g}); "
      f"med truth {med([S['LTNq'][k]['max_ec_true'] for k in keys]):.3f} amcl {med([S['LANq'][k]['max_ec_true'] for k in keys]):.3f}")
print(f"believed max|e_cross| under amcl med {med([S['LANq'][k]['max_ec_bel'] for k in keys]):.3f} (true {med([S['LANq'][k]['max_ec_true'] for k in keys]):.3f})")

print("\n== 3. attribution of amcl-only contacts (LANq contact, LTNq clean)")
cls = {}
for k in amcl_only:
    a = S["LANq"][k]
    why = classify(a, S["LTNq"][k])
    cls[k] = why
    print(f"  {k:24s} t_c {a['t_contact']:5.1f}s  loc {a['loc_at_contact']:.2f} (cross {a['loc_cross_at_contact']:+.2f})  "
          f"e_cross true {a['ec_true_at_contact']:+.2f} bel {a['ec_bel_at_contact']:+.2f}  e_along true {a['ea_true_at_contact']:+.2f}  "
          f"jump3s {a['jump_pre3_max']:.2f}  plan clr {a['plan_clear_near_contact']:.2f}  loc>0.2 for {a['t_loc_gt_020_before_contact']}s  "
          f"patch {int(a['on_patch_at_contact'])}  yawrate {a['yaw_rate_at_contact']:.2f}  sat3s {a['sat_pre3']:.2f}  bel_clear {a['clear_at_contact_believed']:+.2f}  -> {''.join(why)}")
from collections import Counter
print("  primary (first) class:", Counter(w[0] for w in cls.values()))
print("  any-class counts:", Counter(c for w in cls.values() for c in w))
A = [S["LANq"][k] for k in amcl_only]
print(f"  at contact: loc err med {med([a['loc_at_contact'] for a in A]):.2f} (IQR {q([a['loc_at_contact'] for a in A],25):.2f}-{q([a['loc_at_contact'] for a in A],75):.2f}); "
      f"|e_cross true| med {med([abs(a['ec_true_at_contact']) for a in A]):.2f}, believed med {med([abs(a['ec_bel_at_contact']) for a in A]):.2f}; "
      f"loc err >= 0.10 on {sum(a['loc_at_contact']>=0.1 for a in A)}, >= 0.20 on {sum(a['loc_at_contact']>=0.2 for a in A)}; "
      f"believed clearance >= 0 (thought it was clear) on {sum(a['clear_at_contact_believed']>=0 for a in A)}/{len(A)}; "
      f"on patch {sum(a['on_patch_at_contact'] for a in A)}; jump>=0.1 in 3 s {sum(a['jump_pre3_max']>=0.1 for a in A)}; "
      f"plan clr <0.12 {sum((a['plan_clear_near_contact'] or 1)<0.12 for a in A)}; "
      f"loc>0.2 for >=3 s before {sum((a['t_loc_gt_020_before_contact'] or 0)>=3 for a in A)}")
print(f"  truth-cell contacts, for comparison: |e_cross true| at contact med "
      f"{med([abs(S['LTNq'][k]['ec_true_at_contact']) for k in keys if S['LTNq'][k]['contact']]):.2f}; "
      f"plan clr near contact med {med([S['LTNq'][k]['plan_clear_near_contact'] for k in keys if S['LTNq'][k]['contact']]):.2f}; "
      f"plan min clearance over all plans med {med([v['plan_min_clear'] for v in S['LTNq'].values()]):.3f}")
print(f"  both-contact pairs: truth contact at t {med([S['LTNq'][k]['t_contact'] for k in both]):.1f} s vs amcl {med([S['LANq'][k]['t_contact'] for k in both]):.1f} s; "
      f"overlap s truth {med([S['LTNq'][k]['wall_overlap_s'] for k in both]):.2f} amcl {med([S['LANq'][k]['wall_overlap_s'] for k in both]):.2f}")
print(f"  truth-only: {truth_only}")

print("\n== 4. TVLQG surface estimate: LTNq vs LANq vs LANqe")
for c in ("LTNq", "LANq", "LANqe"):
    V = list(S[c].values())
    sw = [v["k_switch_t"] for v in V if v["k_switch_t"] is not None]
    print(f"{c:6s} switched {len(sw)}/{len(V)} at med {med(sw):.1f} s; k_hat end med {med([v['k_hat_end'] for v in V]):.3f} "
          f"(IQR {q([v['k_hat_end'] for v in V],25):.3f}..{q([v['k_hat_end'] for v in V],75):.3f}); k_hat min med {med([v['k_hat_min'] for v in V]):.2f} "
          f"max med {med([v['k_hat_max'] for v in V]):.2f}; k_bar sd after switch med {med([v['k_bar_sd_after'] for v in V]):.3f}; "
          f"gated frac med {med([v['gated_frac'] for v in V]):.2f} (p90 {q([v['gated_frac'] for v in V],90):.2f}); NIS med {med([v['nis_med'] for v in V]):.2f}")
kk = sorted(set(S["LTNq"]) & set(S["LANq"]))
d = [S["LANq"][k]["k_hat_end"] - S["LTNq"][k]["k_hat_end"] for k in kk]
print(f"  paired k_hat_end amcl - truth: med {med(d):+.3f}, |d| med {med([abs(x) for x in d]):.3f}, |d|>0.1 on {sum(abs(x)>0.1 for x in d)}/{len(d)}")

print("\n== 5. LAN vs LANq vs LANqe paired (same plan/seed)")
for a, b in (("LAN", "LANq"), ("LANq", "LANqe"), ("LAN", "LANqe")):
    kk = sorted(set(S[a]) & set(S[b]))
    for f in ("final_err", "max_ec_true", "loc_med", "loc_max", "sat_frac"):
        w = sum(S[b][k][f] < S[a][k][f] for k in kk); l = sum(S[b][k][f] > S[a][k][f] for k in kk)
        print(f"  {a}->{b} {f:12s} better {w}/{w+l} (p {sign_p(w,l):.3f})  med {med([S[a][k][f] for k in kk]):.3f} -> {med([S[b][k][f] for k in kk]):.3f}")
    print(f"  {a}->{b} contacts {sum(S[a][k]['contact'] for k in kk)} -> {sum(S[b][k]['contact'] for k in kk)}  arrived "
          f"{sum(S[a][k]['outcome']=='arrived' for k in kk)} -> {sum(S[b][k]['outcome']=='arrived' for k in kk)}")
# where does LANq lose to LAN: relate to k_hat/sat
kk = sorted(set(S["LAN"]) & set(S["LANq"]))
lost = [k for k in kk if S["LANq"][k]["final_err"] > S["LAN"][k]["final_err"] + 0.1]
won = [k for k in kk if S["LANq"][k]["final_err"] < S["LAN"][k]["final_err"] - 0.1]
print(f"  LANq loses by >0.1 m on {len(lost)}, wins on {len(won)}")
for name, L in (("lost", lost), ("won", won)):
    V = [S["LANq"][k] for k in L]
    print(f"   {name}: k_hat_end med {med([v['k_hat_end'] for v in V]):.3f}  switch t med {med([v['k_switch_t'] for v in V]):.1f}  "
          f"sat med {med([v['sat_frac'] for v in V]):.2f}  loc_med {med([v['loc_med'] for v in V]):.3f}  loc_max {med([v['loc_max'] for v in V]):.2f}  "
          f"LAN final_err med {med([S['LAN'][k]['final_err'] for k in L]):.2f} -> LANq {med([v['final_err'] for v in V]):.2f}")
