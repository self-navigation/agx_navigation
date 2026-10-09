#!/usr/bin/env python3
"""Real Scout Mini, corridor slaloms 2026-10-09 (#50): the sideways slide k on
wood and linoleum, and a check of the lidar odometry's along-corridor distance.
    .venv/bin/python figures/2026-10-09/real_corridor_k/render.py
Inputs: run_data/2026-10-09_slalom2_wood, run_data/2026-10-09_slalom1_linoleum_doorhit
(icp.csv from tools/real_icp.py, imu.csv, can.log; scan_first/last.npy = the
first and last /lidar/points scans of slalom2's bag, z -0.25..0.5 m).
"""
import os
import struct
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(R, "tools"))
from real_sideslip import smooth_deriv  # noqa: E402

RD = os.path.join(R, "run_data")
WOOD = os.path.join(RD, "2026-10-09_slalom2_wood")
LINO = os.path.join(RD, "2026-10-09_slalom1_linoleum_doorhit")
OUT = os.path.dirname(os.path.abspath(__file__))
K_SIM = -0.331
stats = []


def odometer(d, scale=1.023):
    od = []
    for line in open(os.path.join(d, "can.log")):
        p = line.split()
        if len(p) > 11 and p[2] == "311":
            od.append((float(p[0].strip("()")),
                       *struct.unpack(">ii", bytes(int(x, 16) for x in p[4:12]))))
    od = np.array(od, float)
    od[:, 1:] *= scale / 1000.0
    return od


def lateral(d, until=None):
    """As tools/real_corridor_k.py, axis 0: (v*omega, v_lat, v) over moving samples."""
    icp = np.loadtxt(os.path.join(d, "icp.csv"), delimiter=",")
    imu = np.loadtxt(os.path.join(d, "imu.csv"), delimiter=",")
    od = odometer(d)
    t = icp[:, 0] + 0.17
    _, vy = smooth_deriv(t, icp[:, 2], 0.6)
    g = np.concatenate([[0], np.cumsum(0.5 * (imu[1:, 1] + imu[:-1, 1]) * np.diff(imu[:, 0]))])
    th = np.interp(t, imu[:, 0], g) - np.interp(t[0], imu[:, 0], g) + np.unwrap(icp[:, 3])[0]
    wz = np.interp(t, imu[:, 0], np.convolve(imu[:, 1], np.ones(10) / 10, mode="same"))
    _, vl = smooth_deriv(od[:, 0], od[:, 1], 0.6)
    _, vr = smooth_deriv(od[:, 0], od[:, 2], 0.6)
    v = np.interp(t, od[:, 0], 0.5 * (vl + vr))
    vlat = (vy - v * np.sin(th)) / np.cos(th)
    ok = np.isfinite(vlat) & (np.abs(v) > 0.1)
    if until is not None:
        ok &= t - imu[0, 0] <= until
    return v[ok] * wz[ok], vlat[ok], v[ok]


def fit(vw, vlat):
    A = np.c_[vw, np.ones(len(vw))]
    c = np.linalg.lstsq(A, vlat, rcond=None)[0]
    r = vlat - A @ c
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A) * r.var(ddof=2))) * np.sqrt(6)
    return c, se[0], r.std()


# --- k scatter ---------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, (name, d, until) in zip(axs, (("wood (slalom2)", WOOD, None),
                                      ("linoleum (slalom1, before the door hit)", LINO, 16.9))):
    vw, vlat, v = lateral(d, until)
    c, se, sd = fit(vw, vlat)
    for lo, hi, col in ((0.1, 0.3, "tab:blue"), (0.3, 1.0, "tab:orange")):
        m = (np.abs(v) >= lo) & (np.abs(v) < hi)
        if m.any():
            ax.scatter(vw[m], vlat[m], s=6, c=col, alpha=0.6, label=f"v {lo}-{hi} m/s (n={m.sum()})")
    x = np.linspace(-0.25, 0.25, 2)
    ax.plot(x, c[0] * x + c[1], "k", label=f"fit k = {c[0]:+.2f} ± {se:.2f} s")
    ax.plot(x, K_SIM * x, "r--", label=f"Gazebo k = {K_SIM:+.2f} s")
    ax.axhline(0, c="0.6", lw=0.5)
    ax.set_title(name)
    ax.set_xlabel("v·ω [m/s · rad/s]")
    ax.grid(lw=0.3)
    ax.legend(fontsize=8, loc="upper right")
    stats.append(f"{name}: n={len(vw)} k={c[0]:+.3f} s SE {se:.3f} c0 {c[1]:+.4f} resid sd {sd:.4f} m/s")
axs[0].set_ylabel("v_lat [m/s] (ICP across-corridor, odometer forward)")
fig.suptitle("Real Scout Mini: sideways slide v_lat = k·v·ω, corridor slaloms 2026-10-09", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "k_scatter.png"), dpi=110)

# --- along-corridor distance: door features vs ICP vs odometer --------------
icp = np.loadtxt(os.path.join(WOOD, "icp.csv"), delimiter=",")
imu = np.loadtxt(os.path.join(WOOD, "imu.csv"), delimiter=",")
od = odometer(WOOD)
g = np.concatenate([[0], np.cumsum(0.5 * (imu[1:, 1] + imu[:-1, 1]) * np.diff(imu[:, 0]))])
th = np.interp(od[1:, 0], imu[:, 0], g) - np.interp(icp[0, 0], imu[:, 0], g)
m = (od[1:, 0] >= icp[0, 0]) & (od[1:, 0] <= icp[-1, 0])
x_odo = np.sum(np.diff(0.5 * (od[:, 1] + od[:, 2]))[m] * np.cos(th[m]))
A = np.load(os.path.join(WOOD, "scan_first.npy"))[:, :2]
B = np.load(os.path.join(WOOD, "scan_last.npy"))[:, :2]


def place(P, x, y, t):
    c, s = np.cos(t), np.sin(t)
    return np.c_[x + c * P[:, 0] - s * P[:, 1], y + s * P[:, 0] + c * P[:, 1]]


def walls(P):
    out = []
    for sel in (P[:, 1] > 1.3, P[:, 1] < -1.0):
        Q = P[sel & (P[:, 0] > 0) & (P[:, 0] < 8)]
        for _ in range(3):
            c = np.polyfit(Q[:, 0], Q[:, 1], 1)
            Q = Q[np.abs(Q[:, 1] - np.polyval(c, Q[:, 0])) < 0.05]
        out.append(c)
    return out


WA = walls(A)


def features(P):
    """Points off both wall lines: door leaves, frames, objects -- what pins x."""
    off = np.min([np.abs(P[:, 1] - np.polyval(c, P[:, 0])) for c in WA], axis=0)
    return P[(off > 0.12) & (np.abs(P[:, 1]) < 2.3)]


FA = features(A)
T = cKDTree(FA)
y_end, th_end = icp[-1, 2], icp[-1, 3]  # across-corridor and yaw: ICP is well constrained there
xs = np.arange(5.0, 8.0001, 0.01)
score = np.array([np.mean(T.query(features(place(B, x, y_end, th_end)))[0] < 0.05) for x in xs])
x_feat = xs[np.argmax(score)]
half = xs[score >= score.max() / 2]
stats.append(f"slalom2 end position along the corridor: features {x_feat:.2f} m (half-max {half.min():.2f}-{half.max():.2f}), "
             f"ICP {icp[-1, 1]:.3f} m ({icp[-1, 1] / x_feat:.3f}x), odometer+gyro {x_odo:.3f} m ({x_odo / x_feat:.3f}x)")

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.7])
for i, (x, lab) in enumerate(((icp[-1, 1], "ICP"), (x_feat, "feature alignment"))):
    ax = fig.add_subplot(gs[i])
    Fb = features(place(B, x, y_end, th_end))
    ax.scatter(FA[:, 0], FA[:, 1], s=1, c="k", label="first scan, off-wall points")
    ax.scatter(Fb[:, 0], Fb[:, 1], s=1, c="r", label="last scan, off-wall points")
    ax.set_title(f"last scan placed at x = {x:.2f} m ({lab}); y and yaw from ICP", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 16)
    ax.set_ylim(-2.3, 2.3)
    ax.grid(lw=0.3)
    ax.legend(fontsize=7, loc="lower left")
ax = fig.add_subplot(gs[2])
ax.plot(xs, score, "k")
for x, lab, col in ((icp[-1, 1], "ICP", "tab:blue"), (x_odo, "odometer", "tab:green"), (x_feat, "features", "r")):
    ax.axvline(x, c=col, ls="--", label=f"{lab} {x:.2f} m")
ax.set_xlabel("end x along the corridor [m] (first-scan frame)")
ax.set_ylabel("feature match\n(frac < 5 cm)")
ax.legend(fontsize=8)
ax.grid(lw=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "along_track_check.png"), dpi=100)

rev = subprocess.run(["git", "-C", R, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
with open(os.path.join(OUT, "stats.txt"), "w") as f:
    f.write(f"rendered at {rev}\n" + "\n".join(stats) + "\n")
print("\n".join(stats))

# --- corridor maps: lidar scans placed along the driven track ---------------
from plan_check import HL, HW, hull, rollout  # noqa: E402


def track(d, x_scale=1.0):
    """Pose (t, x, y, th) at the odometer rate: x by odometer + gyro dead
    reckoning (ICP is degenerate along the corridor), y from ICP (well
    constrained across it), th from ICP yaw (well constrained)."""
    icp = np.loadtxt(os.path.join(d, "icp.csv"), delimiter=",")
    imu = np.loadtxt(os.path.join(d, "imu.csv"), delimiter=",")
    od = odometer(d)
    t = od[:, 0]
    g = np.concatenate([[0], np.cumsum(0.5 * (imu[1:, 1] + imu[:-1, 1]) * np.diff(imu[:, 0]))])
    th = np.interp(t, icp[:, 0] + 0.17, np.unwrap(icp[:, 3]))  # ICP yaw: 0.6 deg rms from the gyro, sharper walls
    ds = np.r_[0, np.diff(0.5 * (od[:, 1] + od[:, 2]))] * x_scale
    x = np.cumsum(ds * np.cos(th))
    x -= np.interp(icp[0, 0] + 0.17, t, x)
    y = np.interp(t, icp[:, 0] + 0.17, icp[:, 2])
    return t, x, y, th, imu[0, 0]


def segments(d):
    segs = []
    for line in open(os.path.join(d, "drive.log")):
        if " seg " in line:
            s = line.split()
            segs.append((float(s[0]), float(s[2][2:]), float(s[3][2:]), float(s[4][5:])))
    return segs


fig, axs = plt.subplots(2, 1, figsize=(15, 8.5))
for ax, (title, d, xs_, t_hit) in zip(axs, (
        ("wood, slalom2: v 0.2 then 0.4 m/s, ω ±0.5 rad/s", WOOD, x_feat / x_odo, None),
        ("linoleum, slalom1: v 0.3 m/s, ω ±0.5 rad/s, first swing full length -> drifted into an open door", LINO, 1.0, 17.2))):
    t, x, y, th, t0 = track(d, xs_)
    S = np.load(os.path.join(d, "scans_sub.npz"))
    t_last = t[-1] if t_hit is None else t0 + t_hit  # after the hit: RC driving, pose unreliable
    for i, ts in enumerate(S["t"]):
        P = S[f"pts_{i}"]
        P = P[np.hypot(P[:, 0], P[:, 1]) > 0.45]
        # deskew: each point at the pose of its own time. Stamp = scan start; the
        # 0.17 s lag was fitted against KISS-ICP's mid-scan (0.05 s) pose.
        tp = ts + 0.17 + P[:, 2] - 0.05
        if tp.max() > t_last:
            continue
        c, s_ = np.cos(np.interp(tp, t, th)), np.sin(np.interp(tp, t, th))
        Q = np.c_[np.interp(tp, t, x) + c * P[:, 0] - s_ * P[:, 1],
                  np.interp(tp, t, y) + s_ * P[:, 0] + c * P[:, 1]]
        ax.scatter(Q[:, 0], Q[:, 1], s=0.3, c="0.55", lw=0, rasterized=True)
    segs = segments(d)
    # the commanded profile, rolled out ideally from the pose at the first non-zero segment
    prof = [[v, w, h] for _, v, w, h in segs if (v, w) != (0, 0)]
    j0 = np.searchsorted(t, segs[1][0] - 1.0)
    R = rollout(prof, gv=1.01, gw=1.10)
    c, s_ = np.cos(th[j0]), np.sin(th[j0])
    ax.plot(x[j0] + c * R[:, 1] - s_ * R[:, 2], y[j0] + s_ * R[:, 1] + c * R[:, 2], "b--", lw=1,
            label="commanded profile, ideal (gains 1.01 / 1.10, no slip)")
    end = t[-1] if t_hit is None else t0 + t_hit
    m = (t >= segs[0][0]) & (t <= end + 1.0)
    ax.plot(x[m], y[m], "r", lw=1.2, label="driven track (odometer along, lidar across and heading)")
    cm = plt.get_cmap("viridis")
    tt = np.arange(segs[1][0] - 1.0, min(end, t[-1]), 0.5)
    for k_, ti in enumerate(tt):
        j = np.searchsorted(t, ti)
        h = hull(x[j], y[j], th[j])
        ax.fill(h[:, 0], h[:, 1], fc="none", ec=cm(k_ / max(1, len(tt) - 1)), lw=0.6)
    if t_hit is not None:
        j = np.searchsorted(t, t0 + t_hit)
        h = hull(x[j], y[j], th[j])
        ax.fill(h[:, 0], h[:, 1], fc="r", alpha=0.35, label=f"contact with the door, t = {t_hit} s")
    ax.set_aspect("equal")
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlim(-3, 13)
    ax.grid(lw=0.3)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylabel("y [m]")
axs[-1].set_xlabel("x along the corridor [m] (first-scan frame); grey = lidar, outlines = robot hull every 0.5 s")
fig.suptitle("Real Scout Mini in the university corridor, 2026-10-09: lidar map and driven track", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "corridor_maps.png"), dpi=110)
