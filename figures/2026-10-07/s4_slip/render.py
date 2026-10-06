#!/usr/bin/env python3
"""Render the S4 (#44) slip-model figures from g2_results.json (+ _tracks.npz).

Regenerate the data first (needs the main checkout's traj_data_v2/ and
soak_data/soak_broad_open_loop.jsonl):
    PYTHONPATH=src/rudn-ordjo-building .venv/bin/python tools/validate_slip_model.py \
        --out figures/2026-10-07/s4_slip/g2_results.json
then:  .venv/bin/python figures/2026-10-07/s4_slip/render.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src", "agx_navigation", "agx_planning"))
from agx_planning.rl_corrector import slip_model as sm  # noqa: E402

PATCH_COLOR = {"linoleum": "#d8cca6", "wet_tile": "#8cc0cc", "slippery": "#4099ff",
               "icy": "#c8e8ff", "directional_x": "#1a5ae6", "directional_y": "#1a5ae6",
               "rough": "#6b4220"}


def fig_chi():
    mu = np.linspace(0.25, 1.05, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    tm = np.array([r[0] for r in sm.CHI_TABLE])
    tc = np.array([r[1] for r in sm.CHI_TABLE])
    tg = np.array([r[2] for r in sm.CHI_TABLE])
    ts = np.array([r[4] for r in sm.CHI_TABLE])
    a1.plot(mu, [sm.chi_of_mu(m) for m in mu], "k-", lw=1.5, label="модель χ(μ) = 1/k(μ)")
    a1.errorbar(tm, tc, yerr=ts, fmt="o", color="C3", capsize=3, label="измерено (slip_ident, разброс)")
    a1.set_yscale("log")
    a1.set_xlabel("коэффициент трения μ")
    a1.set_ylabel("коэффициент скольжения χ")
    a2.plot(mu, [sm.yaw_gain_of_mu(m) for m in mu], "k-", lw=1.5, label="модель k(μ)")
    a2.plot(tm, tg, "o", color="C3", label="измерено")
    a2.set_xlabel("коэффициент трения μ")
    a2.set_ylabel("передача по рысканию k = ω/ω_ид")
    for a in (a1, a2):
        a.axvspan(0.25, sm.KNEE_MU, color="C0", alpha=0.12,
                  label="излом: μ ≤ 0,45 — робот не управляется по курсу")
        a.axvline(sm.KNEE_MU, color="C0", ls="--", lw=1)
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="best")
    fig.suptitle("Модель проскальзывания χ(μ); колесо μ2 = 0,45, грунт по оси абсцисс", y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "chi_of_mu.png"), dpi=150)


def fig_tracks(res, tracks):
    plans = res["plans"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    for ax, r in zip(axes.flat, plans):
        n = r["plan"]
        for p in r["patches"]:
            rect = Rectangle((-p["width"] / 2, -p["length"] / 2), p["width"], p["length"],
                             facecolor=PATCH_COLOR.get(p["profile"], "grey"), alpha=0.6,
                             edgecolor="none")
            rect.set_transform(Affine2D().rotate(p.get("yaw", 0.0)).translate(p["x"], p["y"])
                               + ax.transData)
            ax.add_patch(rect)
        plan = tracks[f"{n}__plan"]
        ax.plot(plan[:, 0], plan[:, 1], "k-", lw=1.2, label="план")
        s, z = tracks[f"{n}__slip"], tracks[f"{n}__noslip"]
        ax.plot(s[:, 0], s[:, 1], "C3-", lw=1.2, label="кинематика + χ(μ)")
        ax.plot(z[:, 0], z[:, 1], "C2--", lw=1, label="кинематика без скольжения")
        ge = np.array(r["gz_end"])
        ax.plot(ge[:, 0], ge[:, 1], "bx", ms=7, mew=1.5, label="Gazebo: конечные точки (5 повторов)")
        ax.plot(*plan[-1, :2], "k*", ms=10)
        mark = "✓" if r["slip_dir_agree"] else "✗"
        ax.set_title(f"{n.replace('floor_6_v2_', 'план ')}\n"
                     f"e_кон Gazebo {r['gz_final_err_mean']:.2f} м, модель {r['slip_final_err']:.2f} м; "
                     f"направление {mark}", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=6, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("x, м")
    for ax in axes[:, 0]:
        ax.set_ylabel("y, м")
    sm_ = res["summary"]["slip"]
    fig.suptitle("G2: разомкнутый контур, кинематическая модель со скольжением vs Gazebo "
                 f"(направление {sm_['dir_agree']}/10, e_кон в пределах ×2: {sm_['fe_within2']}/10). "
                 "Пятна: сид 0 вдоль пути", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "g2_tracks.png"), dpi=130)


def main():
    res = json.load(open(os.path.join(HERE, "g2_results.json")))
    tracks = np.load(os.path.join(HERE, "g2_results_tracks.npz"))
    fig_chi()
    fig_tracks(res, tracks)
    for r in res["plans"]:
        print(f"| {r['plan']} | {r['gz_final_err_mean']:.3f} ± {r['gz_final_err_sd']:.3f} "
              f"| {r['gz_lat_mean']:+.3f} ({r['gz_lat_sign_agree']}/{r['n_gz']}) "
              f"| {r['slip_final_err']:.3f} | {r['slip_lat']:+.3f} | {r['slip_fe_ratio']:.2f} "
              f"| {'yes' if r['slip_dir_agree'] else 'NO'} | {'yes' if r['slip_fe_within2'] else 'NO'} "
              f"| {r['noslip_final_err']:.3f} | {r['frac_below_knee']:.0%} |")


if __name__ == "__main__":
    main()
