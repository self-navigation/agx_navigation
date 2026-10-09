#!/usr/bin/env python3
"""Pre-drive clearance check for a drive.py profile on the real robot.

Takes one lidar scan (scan_save.py, used as the body frame) and rolls the
profile out as a unicycle with drive.py's 1 s ramps and the measured command
gains (v 1.01, omega 1.10), optionally with an outward slide v_lat = k v omega.
Plots the scan, the path and the hull every 0.25 s, and prints the minimum
hull-to-point clearance.

    python3 tools/plan_check.py scan.npy '[[0.3,0,1],[0.3,0.5,1.5]]' --out f.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HL, HW = 0.315, 0.2925  # pmp_planner fp_half_length / fp_half_width
RAMP, DT = 1.0, 0.05


def rollout(segs, gv=1.01, gw=1.10, k=0.0):
    """drive.py's command sequence -> body poses (t, x, y, th)."""
    cmds, cv, cw = [], 0.0, 0.0
    for v, w, hold in [[0, 0, 1]] + segs + [[0, 0, 1]]:
        for i in range(1, int(RAMP / DT) + 1):
            a = i * DT / RAMP
            cmds.append((cv + (v - cv) * a, cw + (w - cw) * a))
        cv, cw = v, w
        cmds += [(v, w)] * int(round(hold / DT))
    x = y = th = 0.0
    out = [(0.0, 0.0, 0.0, 0.0)]
    for j, (v, w) in enumerate(cmds):
        va, wa = gv * v, gw * w
        vl = k * va * wa
        x += (va * np.cos(th) - vl * np.sin(th)) * DT
        y += (va * np.sin(th) + vl * np.cos(th)) * DT
        th += wa * DT
        out.append(((j + 1) * DT, x, y, th))
    return np.array(out)


def hull(x, y, th, margin=0.0):
    c = np.array([[HL, HW], [HL, -HW], [-HL, -HW], [-HL, HW]]) + margin * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return c @ R.T + [x, y]


def clearance(P, pose):
    """Distance from points P (N,2) to the hull rectangle at pose (0 if inside)."""
    _, x, y, th = pose
    d = P - [x, y]
    lx = d[:, 0] * np.cos(th) + d[:, 1] * np.sin(th)
    ly = -d[:, 0] * np.sin(th) + d[:, 1] * np.cos(th)
    ex = np.maximum(np.abs(lx) - HL, 0)
    ey = np.maximum(np.abs(ly) - HW, 0)
    return np.hypot(ex, ey)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan")
    ap.add_argument("profile")
    ap.add_argument("--k", type=float, default=-0.33, help="slide used for the pessimistic rollout")
    ap.add_argument("--out", default="plan_check.png")
    a = ap.parse_args()
    p = np.load(a.scan)
    P = p[:, :2][np.hypot(p[:, 0], p[:, 1]) > 0.45]  # drop self-returns
    segs = json.loads(a.profile)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.scatter(P[:, 0], P[:, 1], s=1, c="k", label="lidar")
    for k, col in ((0.0, "tab:blue"), (a.k, "tab:red")):
        R = rollout(segs, k=k)
        ax.plot(R[:, 1], R[:, 2], color=col, label=f"path k={k:+.2f}")
        cl = np.array([clearance(P, r).min() for r in R[::5]])
        i = int(np.argmin(cl)) * 5
        for r in R[::5]:
            h = hull(*r[1:])
            ax.fill(h[:, 0], h[:, 1], fc="none", ec=col, lw=0.4, alpha=0.5)
        h = hull(*R[i, 1:])
        ax.fill(h[:, 0], h[:, 1], fc=col, alpha=0.3)
        print(f"k={k:+.2f}: path end ({R[-1, 1]:+.2f}, {R[-1, 2]:+.2f}) m, "
              f"min clearance {cl.min():.2f} m at t={R[i, 0]:.1f} s (x={R[i, 1]:+.2f}, y={R[i, 2]:+.2f})")
    h = hull(0, 0, 0)
    ax.fill(h[:, 0], h[:, 1], fc="g", alpha=0.4, label="robot now")
    xe = max(10.0, R[:, 1].max() + 2)
    ax.set_xlim(-3, xe)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.set_xlabel("x forward [m]")
    ax.set_ylabel("y left [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"plan check: {a.profile}", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
