#!/usr/bin/env python3
"""S5 re-join examples in the advisor's preferred style (figures/2026-09-30/render_examples.py):
footprint robots instead of dots. Same 6 problems as examples.png (same pick rule), baked-map walls.

    .venv/bin/python figures/2026-10-07/s5_rejoin/render_examples_robot.py

`traj` holds 41 samples uniform over [0, T_w], so T_w/4 moments are indices 0,10,20,30,40.
The robot() helper is copied from 2026-09-30/render_examples.py (importing that module
runs its solver figures).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import MAIN, MAP, OUT, SMOKE, load  # noqa: E402

BODY_L, BODY_W = 0.61, 0.50
WHEEL_X, WHEEL_Y, WHEEL_L, WHEEL_W = 0.23, 0.21, 0.18, 0.08


def robot(ax, q, color, fill=True, alpha=1.0, ls="-", lw=1.2, z=2):
    c, s_ = np.cos(q[2]), np.sin(q[2])
    R = np.array([[c, -s_], [s_, c]])

    def rect(cx, cy, L, W):
        p = np.array([[-L, -W], [L, -W], [L, W], [-L, W]]) / 2 + [cx, cy]
        return p @ R.T + q[:2]
    kw = dict(ec=color, lw=lw, ls=ls, zorder=z)
    ax.add_patch(Polygon(rect(0, 0, BODY_L, BODY_W), fc=color if fill else "none",
                         alpha=alpha * (0.35 if fill else 1), **kw))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.add_patch(Polygon(rect(sx * WHEEL_X, sy * WHEEL_Y, WHEEL_L, WHEEL_W),
                                 fc="0.2" if fill else "none", alpha=alpha, **kw))
    nose = np.array([[0.02, -0.14], [BODY_L / 2 - 0.02, 0], [0.02, 0.14]])
    ax.plot(*(nose @ R.T + q[:2]).T, color=color, lw=lw + 0.6, alpha=alpha, zorder=z + .1)


LEGEND = [
    Patch(fc=(0.84, 0.15, 0.16, 0.35), ec="tab:red", label="фактическое положение робота при отклонении"),
    Patch(fc="none", ec="tab:blue", lw=2, label="цель: состояние плана в момент $T_w$"),
    Line2D([], [], color="0.4", ls="--", marker="o", ms=3.5,
           label="где робот должен был быть по плану (точка каждые $T_w$/4)"),
    Line2D([], [], color="tab:green", lw=2.2, marker="o", ms=4,
           label="траектория возврата (те же моменты; штрихи — отставание от графика)"),
    Patch(fc="none", ec="0.5", ls=":", label="контуры робота в момент $T_w$/2"),
    Patch(fc="0.8", ec="k", lw=2, label="стены и препятствия (запечённая карта)"),
]


def main():
    z = load(SMOKE)
    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
    ok = np.where((z["m"] == 1) & (z["s"][:, 9] >= 3.0))[0]
    dv = np.hypot(z["s"][ok, 0], z["s"][ok, 1])
    pick = ok[np.argsort(dv)[np.linspace(len(ok) * 0.5, len(ok) - 1, 6).astype(int)]]
    fig, axs = plt.subplots(2, 3, figsize=(16, 12.5))
    for a, i in zip(axs.ravel(), pick):
        plan = np.load(f"{MAIN}/traj_data_v2/{z['plan'][i]}.npz")["poses"]
        k, tgt, tr = z["k"][i], z["tgt"][i], z["traj"][i]
        m = tgt - k
        a.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255, zorder=0)
        ts = np.linspace(0, m, 5)
        sched = np.stack([np.interp(ts, np.arange(m + 1), np.unwrap(plan[k:tgt + 1, j]))
                          for j in range(3)], 1)
        act = tr[np.linspace(0, len(tr) - 1, 5).round().astype(int), :3]
        a.plot(plan[:, 0], plan[:, 1], color="0.75", lw=1, zorder=1)
        a.plot(plan[k:tgt + 1, 0], plan[k:tgt + 1, 1], color="0.4", lw=1.6, ls="--", zorder=2)
        a.plot(sched[:, 0], sched[:, 1], "o", color="0.4", ms=3.5, zorder=2)
        a.plot(tr[:, 0], tr[:, 1], color="tab:green", lw=2.2, zorder=3)
        a.plot(act[:, 0], act[:, 1], "o", color="tab:green", ms=4, zorder=3)
        for q, sq in zip(act[1:-1], sched[1:-1]):
            a.plot([q[0], sq[0]], [q[1], sq[1]], color="tab:green", lw=.8, alpha=.7, zorder=2)
        robot(a, act[2], "tab:green", fill=False, lw=0.9, alpha=.6, z=3)
        robot(a, sched[2], "0.4", fill=False, ls=":", lw=0.9, alpha=.6, z=2)
        robot(a, plan[tgt], "tab:blue", fill=False, lw=2, z=4)
        robot(a, tr[0, :3], "tab:red", z=5)
        pts = np.vstack([plan[k:tgt + 1, :2], tr[:, :2]])
        c = (pts.max(0) + pts.min(0)) / 2
        h = max((pts.max(0) - pts.min(0)).max() / 2 + 0.7, 1.3)
        a.set_xlim(c[0] - h, c[0] + h)
        a.set_ylim(c[1] - h, c[1] + h)
        a.set_aspect("equal")
        a.tick_params(labelsize=8)
        a.set_xlabel("x, м", fontsize=9)
        a.set_ylabel("y, м", fontsize=9)
        s = z["s"][i]
        a.set_title(f"{z['plan'][i]}, $T_w$={s[9]:.1f} с\n"
                    f"отклонение {np.hypot(*s[:2]):.2f} м, по курсу {round(np.degrees(s[2])) + 0:+d}°",
                    fontsize=10)
    fig.legend(handles=LEGEND, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(.5, .955))
    fig.suptitle("Примеры возврата на план (PMP, мин. усилие), этаж 6, стены — запечённая карта",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .915), h_pad=2.5)
    fig.savefig(f"{OUT}/examples_robot.png", dpi=130)
    for i in pick:
        print(z["plan"][i], z["s"][i, 9])


if __name__ == "__main__":
    main()
