#!/usr/bin/env python3
"""Replay the TVLQG surface filter over Gazebo compare_run track files.

Each track_*.npz holds truth (t, x, y, yaw) at ~20 Hz and the wheel commands
the corrector sent. The filter (runtime_corrector/tvlqg.py) is fed the commands
(zero-order hold) as (v_cmd, omega_cmd) under the planner's nominal chi, and
measurements synthesized from truth at the track rate with its own measurement
noise (r_pos, r_theta, r_gyro) added, so its sigma stays meaningful. The --out
history holds t, k, sigma_k, rho, sigma_rho, v, omega_cmd, pose and gyro NIS
and gating, and pose NEES against truth. Per run it prints the final k_hat +-
sigma, rho_hat and the time until sigma_k first drops below 0.1 s; pooled, the
inverse-variance mean of the final k_hats and the fraction of runs whose
2-sigma band holds the reference k = -0.33 s.

    python3 tools/tvlqg_replay.py <run dirs or track files> [--out f.npz] [--seed 0]
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "src/agx_navigation/agx_planning")]

from agx_planning.runtime_corrector import tvlqg  # noqa: E402


def replay(f, c_v, c_w, cfg, rng, lag=0.0):
    z = np.load(f)
    tr, wt, wc = z["track"], z["wheel_t"], z["wheel_cmd"]
    if len(tr) < 20 or len(wt) < 20:
        return None
    t, x, y, th = tr.T
    filt = tvlqg.SurfaceFilter(cfg, (x[0], y[0], th[0]))
    th_u = np.unwrap(th)
    hist = []
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 1e-4:
            continue
        j = np.searchsorted(wt, t[i - 1] - lag, side="right") - 1
        if j < 0:
            wl = wr = 0.0
        else:
            wl = 0.5 * (wc[j, 0] + wc[j, 1])
            wr = 0.5 * (wc[j, 2] + wc[j, 3])
        v, w = c_v * (wl + wr), c_w * (wr - wl)
        filt.predict(v, w, dt)
        filt.update_pose((x[i] + rng.normal(0, cfg.r_pos), y[i] + rng.normal(0, cfg.r_pos),
                          th[i] + rng.normal(0, cfg.r_theta)))
        nis_p, gate_p = filt.nis, filt.gated
        filt.update_gyro((th_u[i] - th_u[i - 1]) / dt + rng.normal(0, cfg.r_gyro))
        k, sk = filt.k
        r, sr = filt.rho
        e = np.array([x[i] - filt.x[0], y[i] - filt.x[1],
                      tvlqg.wrap_to_pi(th[i] - filt.x[2])])
        nees = float(e @ np.linalg.solve(filt.P[:3, :3], e))
        hist.append((t[i] - t[0], k, sk, r, sr, v, w, nis_p, gate_p, filt.nis, filt.gated, nees))
    return np.array(hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-true", type=float, default=-0.33)
    ap.add_argument("--lag", type=float, default=0.0,
                    help="command-to-motion delay [s]: the command applied at t is the one sent at t - lag")
    a = ap.parse_args()
    sys.modules.pop("rclpy", None)
    from agx_planning.pmp_planner.config import PlannerConfig
    pc = PlannerConfig(mode="offline")
    cfg = tvlqg.TVLQGConfig()
    rng = np.random.default_rng(a.seed)
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "**", "track_*.npz"), recursive=True)) \
            if os.path.isdir(p) else [p]
    res, hists = [], {}
    for f in files:
        h = replay(f, pc.c_v, pc.c_w, cfg, rng, a.lag)
        if h is None or not len(h):
            continue
        name = os.path.relpath(f)
        hists[name] = h
        below = np.nonzero(h[:, 2] < 0.1)[0]
        t01 = h[below[0], 0] if len(below) else np.nan
        res.append((h[-1, 1], h[-1, 2], h[-1, 3], h[-1, 4], t01, h[-1, 0]))
        print(f"{name}: k {h[-1,1]:+.3f} +- {h[-1,2]:.3f}  rho {h[-1,3]:+.3f} +- {h[-1,4]:.3f}"
              f"  sigma_k<0.1 at {t01:5.1f} s of {h[-1,0]:5.1f}")
    R = np.array(res)
    wgt = 1.0 / R[:, 1] ** 2
    print(f"\n{len(R)} runs. k_hat: median {np.median(R[:,0]):+.3f}, IQR "
          f"[{np.percentile(R[:,0],25):+.3f}, {np.percentile(R[:,0],75):+.3f}], "
          f"inverse-variance mean {np.sum(wgt*R[:,0])/np.sum(wgt):+.3f}; median sigma "
          f"{np.median(R[:,1]):.3f}")
    inside = np.mean(np.abs(R[:, 0] - a.k_true) < 2 * R[:, 1])
    print(f"2-sigma band holds k = {a.k_true}: {inside:.0%} of runs")
    print(f"rho_hat: median {np.median(R[:,2]):+.3f}, IQR [{np.percentile(R[:,2],25):+.3f}, "
          f"{np.percentile(R[:,2],75):+.3f}]")
    print(f"time to sigma_k < 0.1 s: median {np.nanmedian(R[:,4]):.1f} s")
    if a.out:
        np.savez_compressed(a.out, summary=R, names=np.array(list(hists)),
                            **{f"h{i}": h for i, h in enumerate(hists.values())})


if __name__ == "__main__":
    main()
