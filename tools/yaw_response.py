#!/usr/bin/env python3
"""Measure Gazebo's yaw and lateral response to wheel commands from track files.

Holds the commands onto the ground-truth track's intervals and computes, per
interval, omega_act from truth, omega_nom and v_cmd under the planner's
nominal chi, and the body-frame lateral speed. Prints
chi_eff = chi_nom * omega_nom / omega_act binned by |omega_nom| and by v_cmd
(only where |omega_nom| > 0.1 rad/s) with the lateral speed per bin, and
writes the table to --out as JSON.

    python3 tools/yaw_response.py <run dirs or track files> [--out f.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "src/agx_navigation/agx_planning")]


def intervals(f, c_v, c_w):
    z = np.load(f)
    tr, wt, wc = z["track"], z["wheel_t"], z["wheel_cmd"]
    if len(tr) < 10 or len(wt) < 10:
        return None
    t, x, y, th = tr.T
    dt = np.diff(t)
    ok = dt > 1e-4
    tm = 0.5 * (t[1:] + t[:-1])
    th_u = np.unwrap(th)
    om_act = np.diff(th_u) / np.where(ok, dt, 1.0)
    vx = np.diff(x) / np.where(ok, dt, 1.0)
    vy = np.diff(y) / np.where(ok, dt, 1.0)
    thm = 0.5 * (th_u[1:] + th_u[:-1])
    v_lon = vx * np.cos(thm) + vy * np.sin(thm)
    v_lat = -vx * np.sin(thm) + vy * np.cos(thm)
    j = np.searchsorted(wt, tm, side="right") - 1
    ok &= (j >= 0) & (tm <= wt[-1] + 0.2)
    j = np.clip(j, 0, len(wt) - 1)
    wl = 0.5 * (wc[j, 0] + wc[j, 1])
    wr = 0.5 * (wc[j, 2] + wc[j, 3])
    om_nom = c_w * (wr - wl)
    v_cmd = c_v * (wl + wr)
    return dict(om_act=om_act[ok], om_nom=om_nom[ok], v_cmd=v_cmd[ok], v_lon=v_lon[ok],
                v_lat=v_lat[ok])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    _rc = sys.modules.pop("rclpy", None)
    from agx_planning.pmp_planner.config import PlannerConfig
    cfg = PlannerConfig(mode="offline")
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "**", "track_*.npz"), recursive=True)) \
            if os.path.isdir(p) else [p]
    rows = [r for r in (intervals(f, cfg.c_v, cfg.c_w) for f in files) if r]
    Z = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    print(f"{len(files)} track files, {len(rows)} usable, {len(Z['om_act'])} intervals")
    turn = np.abs(Z["om_nom"]) > 0.1
    same = np.sign(Z["om_act"]) == np.sign(Z["om_nom"])
    chi_eff = np.where(turn & same, cfg.slip_chi * Z["om_nom"] / np.where(
        np.abs(Z["om_act"]) > 1e-3, Z["om_act"], np.nan), np.nan)
    out = {"chi_nominal": cfg.slip_chi, "by_omega": [], "by_v": []}

    def row(m, lo, hi, key):
        c = chi_eff[m]
        c = c[np.isfinite(c)]
        lat = np.abs(Z["v_lat"][m])
        if len(c) < 20:
            return
        q = np.percentile(c, [25, 50, 75])
        r = dict(lo=lo, hi=hi, n=int(len(c)), chi_p25=float(q[0]), chi_med=float(q[1]),
                 chi_p75=float(q[2]), vlat_med=float(np.median(lat)),
                 vlat_p90=float(np.percentile(lat, 90)),
                 reversed=float(np.mean(~same[m])))
        out[key].append(r)
        print(f"  {key[3:]:5s} [{lo:4.2f},{hi:4.2f}) n={r['n']:6d} chi_eff med {q[1]:.3f} "
              f"IQR [{q[0]:.3f}, {q[2]:.3f}]  |v_lat| med {r['vlat_med']:.3f} p90 {r['vlat_p90']:.3f}"
              f"  sign-reversed {r['reversed']:.2f}")

    print(f"chi_eff = {cfg.slip_chi} * omega_nom / omega_act, |omega_nom| > 0.1 rad/s")
    for lo, hi in ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.2), (1.2, 3.0)):
        row(turn & (np.abs(Z["om_nom"]) >= lo) & (np.abs(Z["om_nom"]) < hi), lo, hi, "by_omega")
    for lo, hi in ((-1, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.5), (0.5, 2)):
        row(turn & (Z["v_cmd"] >= lo) & (Z["v_cmd"] < hi), lo, hi, "by_v")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
