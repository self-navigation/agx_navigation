#!/usr/bin/env python3
"""Check the pre-drive deviation bound against recorded full-stack runs.

For each run, computes Sigma_e(t) from the plan alone with
tvlqg.tracking_covariance_along_plan and the Mahalanobis distance of the
logged tracking error against the chi^2 quantile. Reports calibration on
contact-free runs, first exceedance vs first contact on runs that touch a
wall, and the cross-track 3-sigma half-width along each plan.

    .venv/bin/python tools/deviation_bound.py <run dir> [--cells LTN,LTNg] [--p 0.001] [--out rows.jsonl]
"""
import argparse
import json
import os
import sys

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning"),
                os.path.join(R, "figures/2026-10-09/tvlqg_target_tracks")]
import render as target  # noqa: E402  (contact_mask)
from agx_planning.runtime_corrector import tvlqg, tvlqr  # noqa: E402

CHI2_3 = {0.01: 11.345, 0.001: 16.266, 0.0001: 21.108}
DT = 0.1


def bound(z, cfg, tcfg):
    """(t_diag, d2, Sigma_e, sigma_k) for one track npz."""
    d, dt_ = z["diag"], z["diag_t"]
    ok = np.isfinite(d[:, :6]).all(axis=1) & (d[:, 15] > 0)
    d, dt_ = d[ok], dt_[ok]
    plan = z["plan_poses"][ok]
    v, w = d[:, 4], d[:, 5]
    P = tvlqg.covariance_along_plan(plan, v, w, DT, cfg)
    sk = np.sqrt(P[:, tvlqg.IK, tvlqg.IK])
    S = tvlqg.tracking_covariance_along_plan(v, w, DT, sk, cfg, tcfg)
    e = d[:, :3]
    d2 = np.einsum("ni,nij,nj->n", e, np.linalg.inv(S), e)
    return dt_ - z["track"][0, 0], d2, S, sk, v, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--cells", default="LTN,LTNg,LTNq")
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--skip-start", type=float, default=0.0,
                    help="ignore exceedances in the first N s (turn in place)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg, tcfg = tvlqg.TVLQGConfig(), tvlqr.TVLQRConfig()
    thr = CHI2_3[a.p]
    rows = []
    for line in open(os.path.join(a.dir, "all_rows.jsonl")):
        r = json.loads(line)
        parts = (r.get("track_path") or "").split("/")
        if len(parts) < 3 or parts[-3] not in a.cells.split(","):
            continue
        z = np.load(os.path.join(a.dir, *parts[-3:]))
        if len(z["diag"]) < 10:
            print("no diag:", r["plan"], parts[-3], parts[-2], r.get("outcome"))
            continue
        t, d2, S, sk, v, w = bound(z, cfg, tcfg)
        tr = z["track"]
        hit = target.contact_mask(tr)
        t_hit = tr[np.argmax(hit), 0] - tr[0, 0] if hit.any() else np.nan
        out = (d2 > thr) & (t >= a.skip_start)
        # Only ticks before any contact count for calibration.
        pre = t < t_hit if np.isfinite(t_hit) else np.ones_like(t, bool)
        t_ex = t[np.argmax(out)] if out.any() else np.nan
        moving = np.abs(v) > 0.2
        turn = moving & (np.abs(w) > 0.3)
        hw = 3 * np.sqrt(S[:, 1, 1])
        rows.append(dict(
            plan=r["plan"], cell=parts[-3], seed=parts[-2], outcome=r.get("outcome"),
            overlap=r.get("wall_overlap_s") or 0.0, t_hit=t_hit, t_exceed=t_ex,
            frac_out_pre=float(out[pre].mean()) if pre.any() else np.nan,
            n_pre=int(pre.sum()),
            hw_straight=float(np.median(hw[moving & ~turn])) if (moving & ~turn).any() else np.nan,
            hw_turn=float(np.median(hw[turn])) if turn.any() else np.nan,
            max_cross=float(np.abs(z["diag"][:, 1][np.isfinite(z["diag"][:, 1])]).max()),
        ))
    if a.out:
        with open(a.out, "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
    print(f"threshold chi2_3({a.p}) = {thr}; skip-start {a.skip_start} s")
    for cell in a.cells.split(","):
        rc = [x for x in rows if x["cell"] == cell]
        clean = [x for x in rc if not x["overlap"]]
        hit = [x for x in rc if x["overlap"]]
        fr = np.array([x["frac_out_pre"] for x in clean])
        any_ex = sum(np.isfinite(x["t_exceed"]) for x in clean)
        lead = np.array([x["t_hit"] - x["t_exceed"] for x in hit])
        before = np.sum(np.isfinite(lead) & (lead > 0))
        print(f"{cell}: clean runs {len(clean)}: ticks outside {100*np.mean(fr):.2f}% "
              f"(model {100*a.p:.2f}%), runs with any exceedance {any_ex}/{len(clean)}; "
              f"contact runs {len(hit)}: exceeded before contact {before}/{len(hit)}, "
              f"lead median {np.nanmedian(np.where(lead > 0, lead, np.nan)):.1f} s")
    hs = np.array([x["hw_straight"] for x in rows])
    ht = np.array([x["hw_turn"] for x in rows])
    print(f"cross-track 3-sigma half-width: straights median {np.nanmedian(hs):.3f} m, "
          f"turns median {np.nanmedian(ht):.3f} m (supervisor offplan_dist 0.5)")


if __name__ == "__main__":
    main()
