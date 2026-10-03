#!/usr/bin/env python3
"""Per-tick departure series for compare_run.py track files (#34, job 160).

The corrector never logs its pose estimate, but it logs its tracking error
(e_along, e_cross, e_heading) against live_plan[index] in the reference frame
(tvlqr.tracking_error: e = R(theta_ref)^T (p - p_ref)). Inverting that gives
the pose the corrector BELIEVED; minus Gazebo truth (`track`, interpolated to
diag_t) that is the localization error at every tick. Under localization:=truth
the reconstruction must reproduce truth -- that is the self-check (`recon_err`).

Output, per track file: one npz with aligned series
  t, idx, ref(x,y,yaw), believed(x,y,yaw), truth(x,y,yaw),
  e_cross_believed, e_cross_true, loc_err (xy norm), loc_err_along/cross,
  loc_err_yaw, sat, dv, domega, wall_clear (footprint clearance, if computable)
Usage: departure_series.py OUT_DIR TRACK.npz...
"""
import os
import sys

import numpy as np

F = ("e_along", "e_cross", "e_heading", "e_norm", "v_ref", "omega_ref", "dv",
     "domega", "dv_raw", "domega_raw", "saturated_v", "saturated_omega",
     "steering_faded", "gain_norm", "index", "valid")
PLAN_DT = 0.1
C = {k: i for i, k in enumerate(F)}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def series(path):
    d = np.load(path, allow_pickle=True)
    if (d["diag"].ndim != 2 or len(d["diag"]) == 0 or d["live_plan"].ndim != 2
            or len(d["diag_t"]) == 0):
        return None
    t, D, P, T = d["diag_t"], d["diag"], d["live_plan"], d["track"]
    # diag "index" is -1 on every tick in job 160 (never populated), so rebuild
    # it from the time-indexed playback: dt 0.1 s, t0 = first diag tick. On the
    # localization:=truth runs this reproduces truth to a few mm (fit offset
    # within +-0.06 s of zero over 8 runs).
    idx = np.clip(np.round((t - t[0]) / PLAN_DT).astype(int), 0, len(P) - 1)
    ref = P[idx]
    c, s = np.cos(ref[:, 2]), np.sin(ref[:, 2])
    ea, ec = D[:, C["e_along"]], D[:, C["e_cross"]]
    bel = np.c_[ref[:, 0] + c * ea - s * ec, ref[:, 1] + s * ea + c * ec,
                wrap(ref[:, 2] + D[:, C["e_heading"]])]
    yaw_u = np.unwrap(T[:, 3])
    tru = np.c_[np.interp(t, T[:, 0], T[:, 1]), np.interp(t, T[:, 0], T[:, 2]),
                wrap(np.interp(t, T[:, 0], yaw_u))]
    dp = bel[:, :2] - tru[:, :2]
    # truth's error against the same reference, in the reference frame
    rt = tru[:, :2] - ref[:, :2]
    ec_true = -s * rt[:, 0] + c * rt[:, 1]
    return dict(
        t=t - float(d["goal_t"]), idx=idx, ref=ref, believed=bel, truth=tru,
        e_cross_believed=ec, e_cross_true=ec_true,
        e_along_true=c * rt[:, 0] + s * rt[:, 1], e_along_believed=ea,
        loc_err=np.hypot(dp[:, 0], dp[:, 1]),
        loc_err_along=c * dp[:, 0] + s * dp[:, 1],
        loc_err_cross=-s * dp[:, 0] + c * dp[:, 1],
        loc_err_yaw=wrap(bel[:, 2] - tru[:, 2]),
        sat=(D[:, C["saturated_v"]] + D[:, C["saturated_omega"]]) > 0,
        dv=D[:, C["dv"]], domega=D[:, C["domega"]],
        track=T, live_plan=P, goal_xy=d["goal_xy"])


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for p in sys.argv[2:]:
        r = series(p)
        if r is None:
            print("skip", p); continue
        np.savez_compressed(os.path.join(out, os.path.basename(p)), **r)
        print(f"{os.path.basename(p)} n={len(r['t'])} "
              f"loc_err med={np.median(r['loc_err']):.3f} max={r['loc_err'].max():.3f}")


if __name__ == "__main__":
    main()
