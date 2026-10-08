#!/usr/bin/env python3
"""Fit DynamicBridge parameters to Gazebo excitation runs and score them.

Input is ``rl_corrector.excite_plant --bridge gazebo`` output: one npz per
open-loop sequence with ``cmd`` (N,4), ground-truth ``pose`` (N+1,3) and
``wheel_speeds``. Each sequence is cut into ``--window`` second windows every
``--stride`` seconds; a window starts from Gazebo's state and rolls the model
out under the recorded commands. The loss is the mean squared position error
plus (``--yaw-weight`` m/rad x heading error)^2. The no-slip kinematic model,
with chi fitted and at the deployed chi, is scored on the same windows as a
baseline, and sequences matching ``--holdout`` are held out.

    python3 tools/fit_surrogate.py <run>/gazebo --holdout rand_s3 plan_floor_6_v2_004 \\
        --out <run>/fit.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agx_navigation", "agx_planning"))
from agx_planning.rl_corrector.dynamic_bridge import DynamicParams, simulate  # noqa: E402

R, TRACK = 0.08, 0.416503


def load(paths):
    out = []
    for p in paths:
        d = np.load(p)
        out.append(dict(name=str(d["name"]), dt=float(d["dt"]), cmd=d["cmd"], pose=d["pose"],
                        wheels=d["wheel_speeds"]))
    return out


def windows(seqs, win_s, stride_s):
    """Stack windows: state0 (B,6), om0 (B,4), cmds (B,T,4), truth (B,T+1,3)."""
    S0, O0, C, P = [], [], [], []
    for s in seqs:
        dt = s["dt"]; T = int(round(win_s / dt)); st = int(round(stride_s / dt))
        pose = s["pose"].copy(); pose[:, 2] = np.unwrap(pose[:, 2])
        n = len(s["cmd"])
        for k in range(1, n - T, st):
            dx, dy = (pose[k + 1, :2] - pose[k - 1, :2]) / (2 * dt)
            th = pose[k, 2]
            vx = np.cos(th) * dx + np.sin(th) * dy
            vy = -np.sin(th) * dx + np.cos(th) * dy
            r = (pose[k + 1, 2] - pose[k - 1, 2]) / (2 * dt)
            S0.append([*pose[k], vx, vy, r])
            w = s["wheels"][k]
            O0.append(np.where(np.isfinite(w), w, s["cmd"][k - 1]))
            C.append(s["cmd"][k:k + T]); P.append(pose[k:k + T + 1])
    return np.array(S0), np.array(O0), np.array(C), np.array(P), dt


def err(pred, truth, yaw_w):
    e_pos = np.sum((pred[..., :2] - truth[..., :2]) ** 2, axis=-1)
    e_yaw = (yaw_w * (pred[..., 2] - truth[..., 2])) ** 2
    return e_pos + e_yaw


def kinematic(S0, C, dt, chi):
    B, T, _ = C.shape
    x = S0[:, :3].copy(); out = np.empty((B, T + 1, 3)); out[:, 0] = x
    for k in range(T):
        wl = 0.5 * (C[:, k, 0] + C[:, k, 1]); wr = 0.5 * (C[:, k, 2] + C[:, k, 3])
        v = 0.5 * R * (wl + wr); om = R * (wr - wl) / (TRACK * chi)
        x[:, 0] += dt * v * np.cos(x[:, 2]); x[:, 1] += dt * v * np.sin(x[:, 2]); x[:, 2] += dt * om
        out[:, k + 1] = x
    return out


def summary(pred, truth):
    e = np.hypot(*(pred[..., :2] - truth[..., :2]).transpose(2, 0, 1))
    h = np.degrees(np.abs(pred[..., 2] - truth[..., 2]))
    return dict(pos_end_median=float(np.median(e[:, -1])), pos_end_p90=float(np.percentile(e[:, -1], 90)),
                pos_mean=float(e.mean()), yaw_end_median_deg=float(np.median(h[:, -1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--yaw-weight", type=float, default=0.3, help="metres per radian")
    ap.add_argument("--max-train-windows", type=int, default=800)
    ap.add_argument("--friction", default="pyramid")
    ap.add_argument("--keys", nargs="*", default=list(DynamicParams.FIT_KEYS))
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = sorted(p for d in args.dirs for p in glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))
    seqs = load(paths)
    test = [s for s in seqs if any(h in s["name"] for h in args.holdout)]
    train = [s for s in seqs if s not in test]
    print(f"{len(train)} train / {len(test)} held-out sequences")
    Str, Otr, Ctr, Ptr, dt = windows(train, args.window, args.stride)
    rng = np.random.default_rng(0)
    if len(Str) > args.max_train_windows:
        idx = rng.choice(len(Str), args.max_train_windows, replace=False)
        Str, Otr, Ctr, Ptr = Str[idx], Otr[idx], Ctr[idx], Ptr[idx]
    print(f"train windows {len(Str)} x {Ctr.shape[1]} steps")

    # Baseline: kinematic, chi fitted on train.
    kres = minimize_scalar(lambda c: err(kinematic(Str, Ctr, dt, c), Ptr, args.yaw_weight).mean(),
                           bounds=(1.0, 2.0), method="bounded")
    chi = float(kres.x)

    base = DynamicParams(friction=args.friction)
    x0 = np.log([getattr(base, k) for k in args.keys])

    def params(z):
        p = DynamicParams(friction=args.friction)
        for k, v in zip(args.keys, np.exp(z)):
            setattr(p, k, float(v))
        return p

    def loss(z):
        st, _, _ = simulate(params(z), Str, Otr, Ctr, dt)
        val = err(st[..., :3], Ptr, args.yaw_weight).mean()
        return val if np.isfinite(val) else 1e9

    print(f"kinematic chi={chi:.4f} loss={kres.fun:.5f}; dynamic start loss={loss(x0):.5f}", flush=True)
    best = None
    for start in (x0, x0 + np.log(0.3) * (np.array(args.keys) == "c2"), x0 + np.log(0.01) * np.isin(args.keys, ["c1", "c2"])):
        res = minimize(loss, start, method="Nelder-Mead",
                       options=dict(maxiter=args.maxiter, xatol=1e-3, fatol=1e-7))
        print(f"  start {np.round(np.exp(start), 4)} -> loss {res.fun:.5f}", flush=True)
        if best is None or res.fun < best.fun:
            best = res
    p = params(best.x)
    print("fitted:", {k: round(getattr(p, k), 5) for k in args.keys}, f"loss {best.fun:.5f}")

    report = dict(chi_kinematic=chi, fitted={k: getattr(p, k) for k in args.keys},
                  window_s=args.window, n_train_seqs=len(train), n_test_seqs=len(test))
    for label, sset in (("train", train), ("heldout", test)):
        if not sset:
            continue
        S, O, C, P, _ = windows(sset, args.window, args.stride)
        st, _, _ = simulate(p, S, O, C, dt)
        report[label] = dict(windows=len(S), dynamic=summary(st[..., :3], P),
                             kinematic=summary(kinematic(S, C, dt, chi), P),
                             kinematic_cfg=summary(kinematic(S, C, dt, 1.373), P),
                             dynamic_sdf=summary(simulate(DynamicParams(friction=args.friction), S, O, C, dt)[0][..., :3], P))
    print(json.dumps(report, indent=1))
    if args.out:
        p.save(args.out, fit_report=report)


if __name__ == "__main__":
    main()
