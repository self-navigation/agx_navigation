#!/usr/bin/env python3
"""Fit the sideways slide k of the real Scout Mini from a corridor slalom.

Takes the across-corridor velocity vy from lidar ICP, the forward speed v
from the odometer (scaled by --odo-scale) and the heading th from the gyro,
then v_lat = (vy - v sin(th)) / cos(th) and fits v_lat = k v omega + c0,
also per speed. --axis is the corridor direction in the ICP frame in degrees.

    python3 tools/real_corridor_k.py <run dir> [--axis 0] [--until s]
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_sideslip import smooth_deriv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--axis", type=float, default=0.0)
    ap.add_argument("--until", type=float, default=None,
                    help="use only samples up to this many seconds after the first IMU sample")
    ap.add_argument("--lag", type=float, default=0.17)
    ap.add_argument("--odo-scale", type=float, default=1.023)
    ap.add_argument("--win", type=float, default=0.6)
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
    ax = np.radians(a.axis)
    X = icp[:, 1] * np.cos(ax) + icp[:, 2] * np.sin(ax)
    Y = -icp[:, 1] * np.sin(ax) + icp[:, 2] * np.cos(ax)
    _, vx_icp = smooth_deriv(t, X, a.win)
    _, vy = smooth_deriv(t, Y, a.win)
    g = np.concatenate([[0], np.cumsum(0.5 * (imu[1:, 1] + imu[:-1, 1]) * np.diff(imu[:, 0]))])
    th = np.interp(t, imu[:, 0], g) - np.interp(t[0], imu[:, 0], g) + np.unwrap(icp[:, 3])[0] - ax
    wz = np.interp(t, imu[:, 0], np.convolve(imu[:, 1], np.ones(10) / 10, mode="same"))
    _, vl = smooth_deriv(od[:, 0], od[:, 1], a.win)
    _, vr = smooth_deriv(od[:, 0], od[:, 2], a.win)
    v = np.interp(t, od[:, 0], 0.5 * (vl + vr))
    vlat = (vy - v * np.sin(th)) / np.cos(th)

    ok = np.isfinite(vlat) & (np.abs(v) > 0.1)
    if a.until is not None:  # e.g. cut a run off before a collision
        ok &= t - imu[0, 0] <= a.until
    print(f"ICP along-axis speed / odometer: {np.median(vx_icp[ok] / (v[ok] * np.cos(th[ok]))):.3f}")
    # samples are correlated over the smoothing window (~6 scans at 10 Hz)
    infl = np.sqrt(max(1.0, a.win / np.median(np.diff(t))))
    for name, m in (("all", ok), ("v<0.3", ok & (np.abs(v) < 0.3)), ("v>0.3", ok & (np.abs(v) > 0.3))):
        if m.sum() < 15:
            continue
        vw = v[m] * wz[m]
        A = np.c_[vw, np.ones(m.sum())]
        c = np.linalg.lstsq(A, vlat[m], rcond=None)[0]
        r = vlat[m] - A @ c
        se = np.sqrt(np.diag(np.linalg.inv(A.T @ A) * r.var(ddof=2))) * infl
        print(f"{name:6s} n={m.sum():3d} k = {c[0]:+.3f} s (SE {se[0]:.3f})  c0 {c[1]:+.4f}  "
              f"resid sd {r.std():.4f} m/s  |vw| max {np.abs(vw).max():.2f}")


if __name__ == "__main__":
    main()
