#!/usr/bin/env python3
"""Replay the TVLQG surface filter offline on a real Scout Mini corridor run.

Reads imu.csv, cmd.csv, can.log and icp.csv. Inputs are the odometer speed
(CAN 0x311 scaled by --odo-scale) and the commanded omega; measurements are
the gyro and the ICP across-corridor position and yaw, delayed by --lag.
Prints k_hat, its sigma and rho_hat for the default noise model and for
q_k = 0 (recursive least squares on k).

    python3 tools/tvlqg_replay_real.py <run dir> [--axis 0] [--out f.npz]
"""
import argparse
import dataclasses
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "src/agx_navigation/agx_planning")]

from agx_planning.runtime_corrector import tvlqg  # noqa: E402


def load(d, odo_scale):
    imu = np.loadtxt(os.path.join(d, "imu.csv"), delimiter=",")
    cmd = np.loadtxt(os.path.join(d, "cmd.csv"), delimiter=",")
    icp = np.loadtxt(os.path.join(d, "icp.csv"), delimiter=",")
    od = []
    for line in open(os.path.join(d, "can.log")):
        p = line.split()
        if len(p) > 11 and p[2] == "311":
            od.append((float(p[0].strip("()")),
                       *struct.unpack(">ii", bytes(int(x, 16) for x in p[4:12]))))
    od = np.array(od, float)
    od[:, 1:] *= odo_scale / 1000.0
    return imu, cmd, icp, od


def replay(imu, cmd, icp, od, cfg, axis, lag, r_cross, icp_yaw=False, cmd_lag=0.0):
    # Odometer forward speed: mean of the two sides' distance derivative.
    s = 0.5 * (od[:, 1] + od[:, 2])
    tm = 0.5 * (od[1:, 0] + od[:-1, 0])
    dt_o = np.diff(od[:, 0])
    v_od = np.where(dt_o > 1e-4, np.diff(s) / np.where(dt_o > 1e-4, dt_o, 1.0), 0.0)
    # The odometer's sign convention against the command: forward must be +.
    if np.median(v_od[np.abs(v_od) > 0.05]) < 0:
        v_od = -v_od

    ax = np.radians(axis)
    nrm = np.array([-np.sin(ax), np.cos(ax)])        # across-corridor unit vector
    t_icp = icp[:, 0] + lag
    ev = sorted([(t, "g", i) for i, t in enumerate(imu[:, 0])]
                + [(t, "p", i) for i, t in enumerate(t_icp)])
    t_start = max(imu[0, 0], t_icp[0], cmd[0, 0])
    f = tvlqg.SurfaceFilter(cfg, (icp[0, 1], icp[0, 2], icp[0, 3]))
    t_prev = t_start
    hist = []
    for t, kind, i in ev:
        if t < t_start:
            continue
        j = np.searchsorted(cmd[:, 0], t_prev - cmd_lag, side="right") - 1
        w_c = cmd[j, 2] if j >= 0 else 0.0
        v = float(np.interp(0.5 * (t + t_prev), tm, v_od))
        f.predict(v, w_c, t - t_prev)
        t_prev = t
        if kind == "g":
            f.update_gyro(imu[i, 1])
        else:
            H = np.zeros((1, tvlqg.N_S))
            H[0, tvlqg.IX], H[0, tvlqg.IY] = nrm
            f.update_linear([nrm @ icp[i, 1:3]], H, [[r_cross ** 2]])
            if icp_yaw:
                f.update_linear([icp[i, 3]], np.eye(tvlqg.N_S)[[tvlqg.ITH]],
                                [[cfg.r_theta ** 2]])
            hist.append((t - t_start, *f.k, *f.rho, v, w_c))
    return f, np.array(hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--axis", type=float, default=0.0)
    ap.add_argument("--lag", type=float, default=0.17)
    ap.add_argument("--odo-scale", type=float, default=1.023)
    ap.add_argument("--r-cross", type=float, default=0.02,
                    help="ICP across-corridor position noise [m]")
    ap.add_argument("--cmd-lag", type=float, default=0.0,
                    help="command dead time [s]; the yaw LAG is modelled (--yaw-tau), the fit found no dead time")
    ap.add_argument("--yaw-tau", type=float, default=0.27,
                    help="chassis yaw lag [s]; 0.27 fitted on slalom2 (gyro vs command)")
    ap.add_argument("--r-gyro", type=float, default=0.03,
                    help="gyro noise while driving [rad/s]; 0.03 on slalom2 (0.01, the sim value, gates 5%% and invents a -0.006 bias)")
    ap.add_argument("--icp-yaw", action="store_true",
                    help="also use ICP's yaw (off: heading from the gyro alone, as real_corridor_k.py)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = load(a.run_dir, a.odo_scale)
    out = {}
    for name, kw in (("default", {}), ("q_k=0", dict(q_k=0.0))):
        if a.r_gyro is not None:
            kw = dict(kw, r_gyro=a.r_gyro)
        cfg = dataclasses.replace(tvlqg.TVLQGConfig(), yaw_tau=a.yaw_tau, **kw)
        f, h = replay(*data, cfg, a.axis, a.lag, a.r_cross, a.icp_yaw, a.cmd_lag)
        k, sk = f.k
        r, sr = f.rho
        print(f"{name:8s} k {k:+.3f} +- {sk:.3f} s   rho {r:+.3f} +- {sr:.3f}   "
              f"b_g {f.x[tvlqg.IBG]:+.4f}   gated {f.n_gated} of {len(data[0]) + len(h)}")
        out[name] = h
    if a.out:
        np.savez_compressed(a.out, **{k.replace("=", "_"): v for k, v in out.items()})


if __name__ == "__main__":
    main()
