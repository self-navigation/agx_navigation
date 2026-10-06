#!/usr/bin/env python3
"""S2 (#42) figures: the supervisory layer on the Nav2 / GMPC arms.

Data: run_data/2026-10-07_scratch_s2-smoke/ (VM ~/run_data/, same name). Cells
used here (the final code, commit ac4ef97):
  G0  pmp-gmpc, no layer, plan 00369        G1  pmp-gmpc + layer, plan 00369
  M4  pmp-mppi, no layer, plan 00419        M3  pmp-mppi + layer, plan 00419
Each holds compare_run's track_*.npz (ground truth) and supervisor.json (the
supervisor's topics, recorded by the smoke wrapper). The baked map
floor_6.{png,yaml} is copied into that directory. Clearance for the no-layer
run is recomputed from its truth track with the layer's own walls.py.

    .venv/bin/python figures/2026-10-07/s2_nav2_layer/render.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(WS, "src", "agx_navigation", "agx_planning"))
from agx_planning.supervisor.walls import SignedWallDistance  # noqa: E402

DATA = os.path.join(WS, "run_data", "2026-10-07_scratch_s2-smoke")
STOP, SLOW = 0.15, 0.5

# Footprint style of figures/2026-10-07/s5_rejoin/render_examples_robot.py.
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


def load(cell):
    d = os.path.join(DATA, cell)
    row = json.loads(open(os.path.join(DATA, f"{cell}.jsonl")).readline())
    z = np.load(glob.glob(os.path.join(d, "track_*.npz"))[0])
    row["goal_t"] = float(z["goal_t"])
    tr = z["track"]
    tr = tr[tr[:, 0] >= row["goal_t"]]
    sp = os.path.join(d, "supervisor.json")
    sup = json.load(open(sp)) if os.path.exists(sp) else None
    return row, tr, sup


def plan_path(row):
    z = np.load(os.path.join(DATA, os.path.basename(row["plan_path"]))) \
        if os.path.exists(os.path.join(DATA, os.path.basename(row["plan_path"]))) else None
    return z


def draw_map(ax, pts, pad=1.0):
    meta = yaml.safe_load(open(os.path.join(DATA, "floor_6.yaml")))
    img = np.asarray(Image.open(os.path.join(DATA, meta["image"])).convert("L"))
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    h, w = img.shape
    ax.imshow(img, cmap="gray", origin="upper", extent=[ox, ox + w * r, oy, oy + h * r],
              vmin=0, vmax=255, zorder=0)
    pts = np.vstack(pts)
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    c, half = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")


def footprints(ax, tr, color, n=6, **kw):
    for i in np.linspace(0, len(tr) - 1, n).round().astype(int):
        robot(ax, tr[i, 1:4], color, fill=False, lw=0.9, alpha=0.7, **kw)


def progress_time(track, path_xy, path_t):
    """Plan time [s] of the robot's progress (nearest plan sample, monotone)."""
    out, k = [], 0
    for x, y in track:
        hi = min(len(path_xy), k + 400)
        k = k + int(np.argmin(np.hypot(path_xy[k:hi, 0] - x, path_xy[k:hi, 1] - y)))
        out.append(path_t[k])
    return np.array(out)


def fig_gmpc():
    (r0, t0, _), (r1, t1, s1) = load("G0"), load("G1")
    sp = np.array(s1["sent_path"])  # (stamp, x, y): the path both runs were sent
    path_t, path_xy = sp[:, 0] - sp[0, 0], sp[:, 1:]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    a = ax[0]
    draw_map(a, [t0[:, 1:3], t1[:, 1:3], path_xy])
    a.plot(path_xy[:, 0], path_xy[:, 1], color="0.4", ls="--", lw=1.4, zorder=2,
           label="план PMP (одинаковый для обоих)")
    a.plot(t0[:, 1], t0[:, 2], color="tab:orange", lw=2, zorder=3,
           label=f"GMPC без слоя: недоезд {r0['final_err']:.2f} м")
    a.plot(t1[:, 1], t1[:, 2], color="tab:green", lw=2, zorder=3,
           label=f"GMPC со слоем: недоезд {r1['final_err']:.2f} м, "
                 f"перепланирований {s1['terminal']['replans']}")
    footprints(a, t0, "tab:orange", z=3)
    footprints(a, t1, "tab:green", z=4)
    robot(a, t1[0, 1:4], "tab:red", z=5)
    a.legend(fontsize=8, loc="lower left")
    a.set_title("План 00369, истинная поза: траектории (контуры робота)")

    b = ax[1]
    g1 = r1["goal_t"]
    b.plot(t0[:, 0] - r0["goal_t"], progress_time(t0[:, 1:3], path_xy, path_t),
           color="tab:orange", label="продвижение по плану, без слоя")
    # Progress on the ORIGINAL plan only until the layer replaced it.
    b.plot(t1[:, 0] - g1, progress_time(t1[:, 1:3], path_xy, path_t),
           color="tab:green", label="продвижение по исходному плану, со слоем")
    T = max(t0[-1, 0] - r0["goal_t"], t1[-1, 0] - g1)
    b.plot([0, path_t[-1]], [0, path_t[-1]], color="tab:orange", ls=":",
           label="эталон GMPC без слоя: t")
    for ts, _i in s1["retime"]:
        b.axvline(ts - g1, color="tab:green", lw=0.6, alpha=0.4)
    st = s1["state"]
    f = s1["state_fields"]
    trig = [t - g1 for t, d in st if d[f.index("trigger")] == 1.0]
    if trig:
        b.plot(trig, np.full(len(trig), -0.5), "v", color="tab:red", ms=5,
               label="запрос перепланирования")
    b.plot([], [], color="tab:green", lw=0.6, alpha=0.4,
           label="перевременение эталона (точка маршрута достигнута)")
    b.set_xlim(0, T)
    b.set_xlabel("время от выдачи цели, с")
    b.set_ylabel("время по исходному плану, с")
    b.set_title("Индекс эталона и продвижение робота во времени")
    b.grid(alpha=0.3)
    b.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "gmpc_with_without_layer.png"), dpi=140)


def fig_mppi():
    walls = SignedWallDistance.from_map_yaml(os.path.join(DATA, "floor_6.yaml"))
    (rL, tL, sL), (r0, t0, _) = load("M3"), load("M4")
    f = sL["state_fields"]
    st = [(t, d) for t, d in sL["state"] if t is not None and t >= rL["goal_t"]]
    tt = np.array([t for t, _ in st]) - rL["goal_t"]
    D = np.array([d for _, d in st])
    clear, scale = D[:, f.index("clearance")], D[:, f.index("speed_scale")]
    applied = D[:, f.index("applied_scale")] if "applied_scale" in f and D.shape[1] > len(f) - 1 \
        else scale
    c0 = np.array([walls.body_clearance(*q) for q in t0[:, 1:4]])
    sp = np.array(sL["sent_path"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    a = ax[0]
    draw_map(a, [tL[:, 1:3], t0[:, 1:3]], pad=1.5)  # the plan runs far beyond: both stall early
    a.plot(sp[:, 1], sp[:, 2], color="0.4", ls="--", lw=1.4, zorder=2, label="план PMP")
    a.plot(t0[:, 1], t0[:, 2], color="tab:orange", lw=2, zorder=3,
           label=f"MPPI без слоя: мин. зазор {c0.min():.2f} м")
    a.plot(tL[:, 1], tL[:, 2], color="tab:blue", lw=2, zorder=3,
           label=f"MPPI со слоем: мин. зазор {clear.min():.2f} м")
    i0 = int(np.argmin(c0))
    robot(a, t0[i0, 1:4], "tab:orange", z=5)
    iL = int(np.argmin(np.abs(tL[:, 0] - (tt[int(np.argmin(clear))] + rL["goal_t"]))))
    robot(a, tL[iL, 1:4], "tab:blue", z=5)
    a.legend(fontsize=8, loc="lower left")
    a.set_title("План 00419 (начало), фантомные стены: контуры в момент мин. зазора")

    b = ax[1]
    b.plot(t0[:, 0] - r0["goal_t"], c0, color="tab:orange", label="зазор без слоя")
    b.plot(tt, clear, color="tab:blue", label="зазор со слоем")
    b.axhline(STOP, color="red", ls="--", lw=1, label=f"порог остановки {STOP} м")
    b.axhline(SLOW, color="gray", ls=":", lw=1, label=f"начало замедления {SLOW} м")
    b.set_ylim(-0.02, 1.6)
    b.set_xlabel("время от выдачи цели, с")
    b.set_ylabel("зазор корпуса до стены, м")
    c = b.twinx()
    c.plot(tt, applied, color="tab:green", lw=1, alpha=0.8,
           label="применённый множитель скорости (со слоем)")
    c.set_ylim(-0.05, 1.05)
    c.set_ylabel("множитель скорости")
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = c.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    b.set_title("Слой безопасности: зазор и множитель скорости")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "mppi_wall_layer.png"), dpi=140)
    return c0.min(), clear.min()


if __name__ == "__main__":
    fig_gmpc()
    print("mppi min clearance without / with:", fig_mppi())
