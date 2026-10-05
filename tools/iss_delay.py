#!/usr/bin/env python3
"""ISS of the corrector's tracking-error loop under a delayed, attenuated pose
estimate (#34, advisor 2026-10-05: "рассчитать устойчивость и влияние задержки").

Model (frozen time, one reference twist (v, w) at a time):

    e(t+dt) = Ad e(t) + Bd u(t) + Ed w(t)            plant, exact ZOH of
                                                    tvlqr.error_dynamics
    u(t)    = -K e_hat(t)                            the corrector's law
    e_hat(t)= k e(t - N dt) + d(t)                   localization: gain k,
                                                    lag tau = N dt, noise d

e = (e_along, e_cross, e_heading); w = (lateral skid speed, yaw-rate loss),
the two things wheel odometry cannot see. K is the corrector's own frozen-time
gain (tvlqr.steady_state_gain, q_cross=2.5, r_omega=2.618, dt=0.1).

The delayed loop is a finite-dimensional linear system on the stacked state
(e(t), e(t-dt), ..., e(t-N dt)). For it:
  * stable  <=>  spectral radius rho(A_cl(k, N)) < 1;
  * ISS with |e_cross(t)| <= c rho'^t |x0| + g_w ||w||_inf + g_d ||d||_inf, where
    g = sum_j |C A_cl^j E| (the l1 norm of the impulse response) is the exact
    peak-to-peak gain -- the tightest constant of that form.
Delay margin tau*(k) = first N*dt at which rho >= 1.

Usage (laptop, repo .venv):  .venv/bin/python tools/iss_delay.py [--plans N]
Prints per-twist margins over the 40 broad plans and the gain tables used in
the paper (sec. ISS).
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src/agx_navigation/agx_planning"))
from agx_planning.runtime_corrector.tvlqr import (  # noqa: E402
    TVLQRConfig, error_dynamics, steady_state_gain)

DT = 0.1                     # corrector tick (rl control_dt / default_dt)
R_WHEEL, TRACK, CHI = 0.08, 0.416503, 1.373   # pmp_planner/config.py


def zoh(A, B, dt):
    n, m = B.shape
    M = np.zeros((n + m, n + m)); M[:n, :n] = A; M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def closed_loop(v, w, k, N, cfg):
    """Stacked closed-loop matrices (A_cl, E_w, E_d, C_cross)."""
    A, B = error_dynamics(v, w)
    K = steady_state_gain(v, w, DT, cfg)
    Ad, Bd = zoh(A, B, DT)
    # disturbance: lateral skid speed enters e_cross, yaw-rate loss e_heading
    Ew = zoh(A, np.array([[0., 0.], [1., 0.], [0., 1.]]), DT)[1]
    n = 3; S = n * (N + 1)
    Acl = np.zeros((S, S))
    Acl[:n, :n] = Ad
    Acl[:n, N * n:(N + 1) * n] += -Bd @ K * k          # u uses e(t - N dt)
    for i in range(N):                                  # shift register
        Acl[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = np.eye(n)
    E_w = np.zeros((S, 2)); E_w[:n] = Ew
    E_d = np.zeros((S, 3)); E_d[:n] = -Bd @ K
    C = np.zeros((1, S)); C[0, 1] = 1.0
    return Acl, E_w, E_d, C


def rho(Acl):
    return float(np.max(np.abs(np.linalg.eigvals(Acl))))


def l1_gain(Acl, E, C, steps=4000):
    g = 0.0; X = E.copy()
    for _ in range(steps):
        g += np.abs(C @ X).sum(axis=1).max()
        X = Acl @ X
        if np.abs(X).max() < 1e-10:
            break
    return g


def margin(v, w, k, cfg, nmax=200):
    for N in range(nmax + 1):
        if rho(closed_loop(v, w, k, N, cfg)[0]) >= 1.0:
            return N * DT
    return np.inf


def plan_twists(n_plans):
    files = [l.strip().replace("__HOME__/", "") for l in
             open("tools/jobs/broad40.txt") if l.strip()][:n_plans]
    tw = []
    for f in files:
        if not os.path.exists(f):
            continue
        wc = np.load(f)["wheel_cmds"]
        v = R_WHEEL / 2 * (wc[:, 0] + wc[:, 1])
        om = R_WHEEL / (TRACK * CHI) * (wc[:, 1] - wc[:, 0])
        tw.append(np.c_[v, om])
    return np.vstack(tw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=int, default=40)
    a = ap.parse_args()
    cfg = TVLQRConfig()
    tw = plan_twists(a.plans)
    mv = tw[np.abs(tw[:, 0]) > 0.05]
    print(f"samples {len(tw)}, moving {len(mv)}; |v| median {np.median(np.abs(mv[:,0])):.2f} "
          f"p90 {np.percentile(np.abs(mv[:,0]),90):.2f} m/s; |w| median "
          f"{np.median(np.abs(mv[:,1])):.2f} p90 {np.percentile(np.abs(mv[:,1]),90):.2f} rad/s")

    # representative twists: median straight, median turn, fast turn
    v0 = float(np.median(np.abs(mv[:, 0])))
    turning = np.abs(mv[:, 1]) > 0.1
    w0 = float(np.median(np.abs(mv[turning, 1])))
    w9 = float(np.percentile(np.abs(mv[:, 1]), 90))
    reps = {"straight": (v0, 0.0), "turn_med": (v0, w0), "turn_p90": (v0, w9)}

    print("\ndelay margin tau* [s]")
    for name, (v, w) in reps.items():
        print(f"  {name:9s} v={v:.2f} w={w:.2f}:  " + "  ".join(
            f"k={k}: {margin(v, w, k, cfg):.1f}" for k in (1.0, 0.4, 0.05)))

    # distribution of tau* over the plans' own twists (subsample)
    rng = np.random.default_rng(0)
    sub = mv[rng.choice(len(mv), size=min(150, len(mv)), replace=False)]
    for k in (1.0, 0.4):
        m = np.array([margin(v, w, k, cfg) for v, w in sub])
        print(f"  over plan twists, k={k}: tau* min {m.min():.1f} p10 {np.percentile(m,10):.1f} "
              f"median {np.median(m):.1f} s;  share with tau*<=2.5 s: {np.mean(m<=2.5)*100:.0f}%")

    print("\nISS gains on e_cross (peak-to-peak), turn_med twist")
    v, w = reps["turn_med"]
    print("  k    tau   rho     g_w(lat skid) [m per m/s]  g_w(yaw loss) [m per rad/s]  g_d [m/m]")
    for k in (1.0, 0.4):
        for tau in (0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            N = int(round(tau / DT))
            Acl, Ew, Ed, C = closed_loop(v, w, k, N, cfg)
            r = rho(Acl)
            if r >= 1:
                print(f"  {k:<4} {tau:<5} {r:.4f}  unstable"); continue
            gl = l1_gain(Acl, Ew[:, :1], C); gy = l1_gain(Acl, Ew[:, 1:], C)
            gd = l1_gain(Acl, Ed, C)
            print(f"  {k:<4} {tau:<5} {r:.4f}  {gl:8.2f}  {gy:8.2f}  {gd:6.2f}")
    hidden_table(reps, cfg)


def hidden_skid_gain(v, w, cfg, T_L, N=1, k=1.0):
    """Peak-to-peak gain v_skid -> e_cross when the lateral skid is HIDDEN from
    the estimate and removed by the localizer with time constant T_L:

        delta' = v_skid - delta / T_L,   e_hat = e(t - N dt) - (0, delta, 0)

    The corrector's own motion is seen (odometry observes commanded motion),
    so the in-loop delay is only the pose latency N dt; the localizer lag acts
    as a bounded input, not as a loop delay.  Returns (rho, gain)."""
    Acl, Ew, Ed, C = closed_loop(v, w, k, N, cfg)
    S = Acl.shape[0]
    a = np.exp(-DT / T_L)
    A2 = np.zeros((S + 1, S + 1)); A2[:S, :S] = Acl
    A2[S, S] = a
    A2[:S, S] = -Ed[:, 1]          # e_hat = ... - delta  ->  u gets +K delta
    E2 = np.zeros((S + 1, 1)); E2[:S, 0] = Ew[:, 0]; E2[S, 0] = T_L * (1 - a)
    C2 = np.zeros((1, S + 1)); C2[0, 1] = 1.0
    return rho(A2), l1_gain(A2, E2, C2, steps=20000)


def hidden_table(reps, cfg):
    print("\nhidden lateral skid (in-loop latency 0.1 s): gain v_skid -> e_cross [m per m/s]")
    for name, (v, w) in reps.items():
        row = []
        for T_L in (0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            r, g = hidden_skid_gain(v, w, cfg, T_L)
            row.append(f"T_L={T_L}: {g:.2f}")
        print(f"  {name:9s} rho={r:.4f}  " + "  ".join(row))


if __name__ == "__main__":
    main()
