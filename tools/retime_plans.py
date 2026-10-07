#!/usr/bin/env python3
"""Slow a solved plan down where its hull is close to a wall.

    .venv/bin/python tools/retime_plans.py --src DIR --dst DIR [--plans FILE]
        [--c-slow 0.10] [--c-full 0.40] [--s-min 0.4]

Same path, slower clock. Per sample the speed scale is s = s_min + (1 - s_min)
* clip((c - c_slow) / (c_full - c_slow), 0, 1), with c the hull clearance of
the plan pose; s is then taken as the minimum over +-0.75 s and moving-averaged
over 1 s, so the robot is already slow at the door and ramps instead of
stepping. Each segment's duration becomes dt / s; poses (unwrapped yaw), wheel
commands (times s) and costates are resampled onto a uniform dt grid. Scaling
the wheel commands by s keeps the path, since playback is time-indexed. The
retimed plan is no longer PMP-optimal.
"""
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src/agx_navigation/agx_planning"))
from agx_planning.supervisor.walls import SignedWallDistance  # noqa: E402

MAP = os.path.join(ROOT, "src/rudn-ordjo-building/maps/floor_6.yaml")


def scale(c, dt, c_slow, c_full, s_min):
    s = s_min + (1.0 - s_min) * np.clip((c - c_slow) / (c_full - c_slow), 0.0, 1.0)
    s = minimum_filter1d(s, size=max(1, int(round(1.5 / dt))), mode="nearest")
    return np.clip(uniform_filter1d(s, size=max(1, int(round(1.0 / dt))), mode="nearest"), s_min, 1.0)


def retime(z, swd, a):
    P, W = z["poses"], z["wheel_cmds"]
    dt = float(z["dt_sample"])
    c = np.array([swd.body_clearance(*p) for p in P])
    s = scale(c, dt, a.c_slow, a.c_full, a.s_min)
    # sample i holds for dt / s_i of new time
    tau = np.concatenate([[0.0], np.cumsum(dt / s[:-1])])
    tn = np.arange(0.0, tau[-1] + 1e-9, dt)
    u = np.unwrap(P[:, 2])
    out = {k: z[k] for k in z.files}
    out["poses"] = np.stack([np.interp(tn, tau, P[:, 0]), np.interp(tn, tau, P[:, 1]),
                             np.interp(tn, tau, u)], axis=1)
    sn = np.interp(tn, tau, s)
    out["wheel_cmds"] = np.stack([np.interp(tn, tau, W[:, j]) for j in range(W.shape[1])], axis=1) * sn[:, None]
    if "costates" in z.files:
        C = z["costates"]
        out["costates"] = np.stack([np.interp(tn, tau, C[:, j]) for j in range(C.shape[1])], axis=1)
    out["retime_scale"] = sn
    return out, len(P) * dt, len(tn) * dt, float(np.mean(s < 0.99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--plans", help="plan list (one npz path per line); default all in --src")
    ap.add_argument("--c-slow", type=float, default=0.10)
    ap.add_argument("--c-full", type=float, default=0.40)
    ap.add_argument("--s-min", type=float, default=0.4)
    a = ap.parse_args()
    swd = SignedWallDistance.from_map_yaml(MAP)
    if a.plans:
        names = [os.path.basename(l.strip()) for l in open(a.plans) if l.strip() and not l.startswith("#")]
    else:
        names = sorted(n for n in os.listdir(a.src) if n.endswith(".npz"))
    os.makedirs(a.dst, exist_ok=True)
    t0s, t1s = 0.0, 0.0
    for n in names:
        out, t0, t1, frac = retime(np.load(os.path.join(a.src, n)), swd, a)
        np.savez(os.path.join(a.dst, n), **out)
        t0s, t1s = t0s + t0, t1s + t1
        print(f"{n}: {t0:.1f} s -> {t1:.1f} s (+{t1 / t0 - 1:.0%}), slowed {frac:.0%} of samples")
    print(f"total {t0s:.0f} s -> {t1s:.0f} s (+{t1s / t0s - 1:.0%})")


if __name__ == "__main__":
    main()
