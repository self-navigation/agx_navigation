#!/usr/bin/env python3
"""TVLQG surface filter (runtime_corrector/tvlqg.py) along one Gazebo run and
the real wood slalom: k_hat with its 3-sigma band, the excitation v*omega that
makes k observable, and sigma_k live vs computed BEFORE the drive from the
trajectory alone (covariance_along_plan).

    .venv/bin/python figures/2026-10-09/tvlqg_surface/render.py

Inputs: run_data/2026-10-08_job237_door-slow-ab/orig/LTN/s0/track_w15_ours-lib_floor_6_v2_00208.npz
(contact-free, final_err 0.03 m) and run_data/2026-10-09_slalom2_wood.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning")]
import tvlqg_replay  # noqa: E402
import tvlqg_replay_real  # noqa: E402
from agx_planning.runtime_corrector import tvlqg  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(R, "run_data/2026-10-08_job237_door-slow-ab/orig/LTN/s0/"
                      "track_w15_ours-lib_floor_6_v2_00208.npz")
WOOD = os.path.join(R, "run_data/2026-10-09_slalom2_wood")
K_SIM, K_WOOD = -0.331, -0.03
C_V, C_W = 0.04, 0.13989471757878671       # PlannerConfig(mode="offline").c_v / c_w


def first_exclusion(h, k_other):
    """First time |k_hat - k_other| > 3 sigma (the surface-switch test)."""
    m = np.abs(h[:, 1] - k_other) > 3.0 * h[:, 2]
    return h[np.argmax(m), 0] if m.any() else np.nan


def sim_series():
    cfg = tvlqg.TVLQGConfig()
    h = tvlqg_replay.replay(SIM, C_V, C_W, cfg, np.random.default_rng(0))
    # Pre-drive covariance on a uniform grid at the track's median step,
    # linearized about the recorded truth and commands (the corrector's
    # corrections are small, so this is the plan up to centimetres).
    z = np.load(SIM)
    tr, wt, wc = z["track"], z["wheel_t"], z["wheel_cmd"]
    dt = float(np.median(np.diff(tr[:, 0])))
    tg = np.arange(tr[0, 0], tr[-1, 0], dt)
    poses = np.c_[np.interp(tg, tr[:, 0], tr[:, 1]), np.interp(tg, tr[:, 0], tr[:, 2]),
                  np.interp(tg, tr[:, 0], np.unwrap(tr[:, 3]))]
    j = np.searchsorted(wt, tg, side="right") - 1
    ok = j >= 0
    j = np.clip(j, 0, None)
    wl = np.where(ok, 0.5 * (wc[j, 0] + wc[j, 1]), 0.0)
    wr = np.where(ok, 0.5 * (wc[j, 2] + wc[j, 3]), 0.0)
    v, w = C_V * (wl + wr), C_W * (wr - wl)
    P = tvlqg.covariance_along_plan(poses, v, w, dt, cfg)
    pre = np.c_[tg - tr[0, 0], np.sqrt(P[:, tvlqg.IK, tvlqg.IK])]
    return h, pre


def real_series():
    data = tvlqg_replay_real.load(WOOD, 1.023)
    cfg = tvlqg.TVLQGConfig(yaw_tau=0.27, r_gyro=0.03)
    _, h = tvlqg_replay_real.replay(*data, cfg, 0.0, 0.17, 0.02)
    return h


def panel(axs, h, pre, title, k_true, k_other, other_name, gated=None):
    t = h[:, 0]
    a0, a1, a2 = axs
    a0.plot(t, h[:, 5] * h[:, 6], lw=0.8, color="tab:gray")
    a0.set_ylabel("v·ω_cmd [m/s²]")
    a0.set_title(title, fontsize=10)
    k, s = h[:, 1], h[:, 2]
    a1.fill_between(t, k - 3 * s, k + 3 * s, color="tab:blue", alpha=0.25, lw=0,
                    label="k̂ ± 3σ")
    a1.plot(t, k, color="tab:blue", lw=1.0)
    a1.axhline(K_SIM, color="tab:red", lw=1, ls="--", label=f"Gazebo k = {K_SIM}")
    a1.axhline(K_WOOD, color="tab:green", lw=1, ls="--", label=f"real wood k = {K_WOOD}")
    te = first_exclusion(h, k_other)
    if np.isfinite(te):
        a1.axvline(te, color="k", lw=0.8, ls=":")
        a1.annotate(f"{other_name} excluded\n(>3σ) at {te:.1f} s", (te, 0.25),
                    fontsize=8, ha="left", va="top", xytext=(4, 0), textcoords="offset points")
    if gated is not None and gated.any():
        a1.plot(t[gated], np.full(gated.sum(), -0.95), "|", color="tab:orange", ms=5,
                label="gyro update gated")
    a1.set_ylim(-1.0, 0.4)
    a1.set_ylabel("k [s]")
    a1.legend(fontsize=7, loc="lower left")
    a2.semilogy(t, s, color="tab:blue", lw=1.0, label="σ_k live")
    if pre is not None:
        a2.semilogy(pre[:, 0], pre[:, 1], color="k", lw=0.8, ls="--",
                    label="σ_k before the drive (from the trajectory)")
    a2.set_ylabel("σ_k [s]")
    a2.set_xlabel("t [s]")
    a2.legend(fontsize=7)
    return te


def main():
    hs, pre = sim_series()
    hr = real_series()
    fig, ax = plt.subplots(3, 2, figsize=(12, 7.5), sharex="col",
                           gridspec_kw=dict(height_ratios=[1, 2, 1.3]))
    te_s = panel(ax[:, 0], hs, pre, "Gazebo, job 237 LTN floor_6_v2_00208 (contact-free)",
                 K_SIM, K_WOOD, "wood", gated=hs[:, 10] > 0)
    te_r = panel(ax[:, 1], hr, None, "Real Scout Mini, wood corridor slalom (2026-10-09)",
                 K_WOOD, K_SIM, "Gazebo's k")
    fig.suptitle("TVLQG surface filter: slide k is learned only in MOVING turns; "
                 "the 3σ band is the surface-switch threshold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tvlqg_surface.png"), dpi=130)
    # Numbers for the README.
    i_s = np.searchsorted(pre[:, 0], hs[:, 0]).clip(0, len(pre) - 1)
    ratio = hs[:, 2] / pre[i_s, 1]
    mv = np.abs(hs[:, 5] * hs[:, 6]) > 0.05
    t_first = hs[np.argmax(mv), 0]
    print(f"sim: final k {hs[-1,1]:+.3f} +- {hs[-1,2]:.3f}; first moving turn {t_first:.1f} s; "
          f"wood excluded at {te_s:.1f} s; live/pre-drive sigma ratio median "
          f"{np.median(ratio):.2f} (p5 {np.percentile(ratio,5):.2f}, p95 {np.percentile(ratio,95):.2f}); "
          f"gyro gated {100*np.mean(hs[:,10]>0):.1f}%")
    print(f"real: final k {hr[-1,1]:+.3f} +- {hr[-1,2]:.3f}; Gazebo k excluded at {te_r:.1f} s "
          f"of {hr[-1,0]:.1f}")


if __name__ == "__main__":
    main()
