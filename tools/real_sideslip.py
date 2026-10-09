#!/usr/bin/env python3
"""Fit the sideways slide k (v_lat = k * v * omega) from a real-robot drive.

Estimates the model in TVLQRConfig.sideslip_k without ground truth: heading
from the integrated gyro (imu.csv), position from lidar odometry (icp.csv from
tools/real_icp.py) shifted by --lag and smoothed by a local quadratic over
--win seconds, and v, omega from the odometer (CAN 0x311 x --odo-scale) and the
gyro rate. Over moving samples (|v| > 0.1) it fits v_lat = k * v * omega + c1 *
omega + c0, where c1 absorbs the lidar's lever arm and c0 any misalignment, and
prints k, its standard error and binned medians of v_lat against v*omega.

    python3 tools/real_sideslip.py <run dir> [--lag 0.17] [--odo-scale 1.023]
"""
import argparse
import os
import struct

import numpy as np


def smooth_deriv(t, y, win):
    """Local quadratic fit of y(t) within +-win/2 -> (value, slope) at each t."""
    val, der = np.full(len(t), np.nan), np.full(len(t), np.nan)
    j0 = j1 = 0
    for i, ti in enumerate(t):
        while t[j0] < ti - win / 2:
            j0 += 1
        while j1 < len(t) and t[j1] <= ti + win / 2:
            j1 += 1
        if j1 - j0 < 5:
            continue
        dt = t[j0:j1] - ti
        c = np.polyfit(dt, y[j0:j1], 2)
        val[i], der[i] = c[2], c[1]
    return val, der


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--lag", type=float, default=0.17)
    ap.add_argument("--odo-scale", type=float, default=1.023)
    ap.add_argument("--win", type=float, default=0.6)
    ap.add_argument("--vmin", type=float, default=0.1)
    a = ap.parse_args()
    d = a.run_dir
    icp = np.loadtxt(os.path.join(d, "icp.csv"), delimiter=",")
    imu = np.loadtxt(os.path.join(d, "imu.csv"), delimiter=",")
    od = []
    for line in open(os.path.join(d, "can.log")):
        p = line.split()
        if len(p) > 11 and p[2] == "311":
            od.append((float(p[0].strip("()")),
                       *struct.unpack(">ii", bytes(int(x, 16) for x in p[4:12]))))
    od = np.array(od, float)
    od[:, 1:] *= a.odo_scale / 1000.0

    t = icp[:, 0] + a.lag
    x, _ = smooth_deriv(t, icp[:, 1], a.win)
    _, vx = smooth_deriv(t, icp[:, 1], a.win)
    _, vy = smooth_deriv(t, icp[:, 2], a.win)
    # gyro heading, anchored to the ICP heading at the first scan
    g = np.concatenate([[0], np.cumsum(0.5 * (imu[1:, 1] + imu[:-1, 1]) * np.diff(imu[:, 0]))])
    th = np.interp(t, imu[:, 0], g)
    th += np.unwrap(icp[:, 3])[0] - th[0]
    wz = np.interp(t, imu[:, 0], np.convolve(imu[:, 1], np.ones(10) / 10, mode="same"))
    v_lon = vx * np.cos(th) + vy * np.sin(th)
    v_lat = -vx * np.sin(th) + vy * np.cos(th)
    _, vl = smooth_deriv(od[:, 0], od[:, 1], a.win)
    _, vr = smooth_deriv(od[:, 0], od[:, 2], a.win)
    v_odo = np.interp(t, od[:, 0], 0.5 * (vl + vr))

    ok = np.isfinite(v_lat) & np.isfinite(v_odo) & (np.abs(v_odo) > a.vmin)
    print(f"{ok.sum()} moving samples of {len(t)}; v_lon/v_odo median {np.median(v_lon[ok] / v_odo[ok]):.3f}")
    if ok.sum() < 20:
        return
    vw = v_odo[ok] * wz[ok]
    A = np.c_[vw, wz[ok], np.ones(ok.sum())]
    coef, res, *_ = np.linalg.lstsq(A, v_lat[ok], rcond=None)
    r = v_lat[ok] - A @ coef
    cov = np.linalg.inv(A.T @ A) * r.var(ddof=3)
    # samples are correlated over ~win; inflate the SE by sqrt(samples per window)
    rate = 1.0 / np.median(np.diff(t))
    infl = np.sqrt(max(1.0, a.win * rate))
    se = np.sqrt(np.diag(cov)) * infl
    print(f"k = {coef[0]:+.3f} s (SE {se[0]:.3f}),  c1(omega) = {coef[1]:+.4f} m (SE {se[1]:.4f}),  "
          f"c0 = {coef[2]:+.4f} m/s;  resid sd {r.std():.4f} m/s")
    print(f"slip angle at the median |v w|: {np.degrees(np.arctan(coef[0] * np.median(np.abs(vw)) / np.median(np.abs(v_odo[ok])))):+.1f} deg")
    edges = np.quantile(vw, np.linspace(0, 1, 7))
    print(f"{'v*w bin':>17} {'n':>4} {'med v_lat':>9} {'pred':>7}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vw >= lo) & (vw <= hi)
        pred = coef[0] * np.median(vw[m]) + coef[1] * np.median(wz[ok][m]) + coef[2]
        print(f"[{lo:+.3f},{hi:+.3f}] {m.sum():4d} {np.median(v_lat[ok][m]):+9.4f} {pred:+7.4f}")


if __name__ == "__main__":
    main()
