#!/usr/bin/env python3
"""#48 plan wall clearance, before vs after the doorway-preserving centring.

    .venv/bin/python figures/2026-10-07/p48_clearance/render.py

Reads gitignored data in the main checkout:
  traj_data_v2/                              the library plans ("before", as used by ours-lib)
  run_data/2026-10-07_nojob_p48-clearance/         offline replans of the same 40 start/goal pairs
                                             (see its README.md; made by its regen.py)
Clearance is hull clearance against the baked floor-6 map, exactly as the
supervisor measures it (supervisor/walls.py SignedWallDistance.body_clearance).
"""
import json
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

MAIN = "/home/danya/university/clown-car/agx_navigation"
OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RUN = f"{MAIN}/run_data/2026-10-07_nojob_p48-clearance"
MAP = f"{MAIN}/src/rudn-ordjo-building/maps/floor_6"
sys.path.insert(0, f"{REPO}/src/agx_navigation/agx_planning")
from agx_planning.supervisor.walls import SignedWallDistance  # noqa: E402

# arm -> (label, colour, loader)
AFTER = "c_r15"  # centring=True, centring_cap=0.8, speed_decay_rate=1.5 (full4.jsonl)
ARMS = [
    ("lib", "до: библиотека traj_data_v2", "tab:red"),
    ("stack", "параметры стека, без центрирования", "tab:orange"),
    (AFTER, "после: стек + центрирование", "tab:green"),
]
BODY_L, BODY_W = 0.61, 0.50
WHEEL_X, WHEEL_Y, WHEEL_L, WHEEL_W = 0.23, 0.21, 0.18, 0.08


def robot(ax, q, color, fill=True, alpha=1.0, ls="-", lw=1.2, z=2):
    """Copied from figures/2026-09-30/render_examples.py (via s5_rejoin)."""
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


def plan(arm, name):
    if arm == "lib":
        return np.load(f"{MAIN}/traj_data_v2/{name}.npz")["poses"]
    for run in RUNS:
        f = f"{RUN}/{run}.poses/{name}__{arm}.npz"
        if os.path.exists(f):
            return np.load(f)["poses"]
    return None


RUNS = ["broad.jsonl", "full4.jsonl"]


def solved():
    recs = [json.loads(ln) for run in RUNS for ln in open(f"{RUN}/{run}")]
    ok = {}
    for r in recs:
        ok.setdefault(r["var"], set())
        if r["status"] == "success":
            ok[r["var"]].add(r["case"])
    return ok


def table(swd, names):
    """Paired stats over the pairs both stack arms solved (printed for the README)."""
    print(f"{len(names)} pairs solved by stack and {AFTER}")
    for arm, lab, _ in ARMS:
        cs = [np.array([swd.body_clearance(*p) for p in plan(arm, n)]) for n in names]
        f05 = np.median([(c < .5).mean() for c in cs])
        f015 = np.median([(c < .15).mean() for c in cs])
        a015 = sum((c < .15).any() for c in cs)
        ov = sum((c < 0).any() for c in cs)
        allc = np.concatenate(cs)
        print(f"  {arm:8s} median plan-time <0.5 {100 * f05:.0f}%  <0.15 {100 * f015:.1f}%  "
              f"pooled <0.15 {100 * (allc < .15).mean():.1f}%  plans <0.15 {a015}  overlap {ov}  "
              f"median min {np.median([c.min() for c in cs]):.2f}")


def fig_hist(swd, names):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.2))
    # The map is 5 cm cells, so clearance takes quantised values; 10 cm bins
    # centred on multiples of 0.1 avoid aliasing.
    bins = np.arange(-0.15, 2.0, 0.1)
    for arm, lab, col in ARMS:
        c = np.concatenate([[swd.body_clearance(*p) for p in plan(arm, n)] for n in names])
        ax.hist(c, bins=bins, histtype="step", lw=2, color=col, weights=np.full(len(c), 100 / len(c)),
                label=f"{lab}: <0,15 м {100 * (c < .15).mean():.1f}%, <0,5 м {100 * (c < .5).mean():.0f}%")
        cs = np.sort(c)
        bx.step(cs, 100 * np.arange(1, len(cs) + 1) / len(cs), color=col, lw=2, where="post", label=lab)
    for a in (ax, bx):
        for x, t in ((0.0, "касание"), (0.15, "стоп 0,15"), (0.5, "замедление 0,5")):
            a.axvline(x, color="k", ls=":", lw=1)
            a.text(x + 0.01, a.get_ylim()[1] * 0.97, t, fontsize=8, va="top")
        a.set_xlabel("зазор корпуса до стены, м")
        a.set_xlim(-0.15, 2.0)
    ax.set_ylabel("доля времени плана, %")
    bx.set_ylabel("доля времени с зазором не больше x, %")
    bx.set_xlim(-0.15, 1.0)
    ax.legend(fontsize=8, loc="center right")
    fig.suptitle(f"Зазор корпуса вдоль плана, этаж 6: {len(names)} пар старт/цель из 40, решённых обоими вариантами стека")
    fig.tight_layout()
    fig.savefig(f"{OUT}/clearance_hist.png", dpi=130)


# (title, plan, centre of the window, half-size)
EXAMPLES = [
    ("дверной проём 0,95 м", "floor_6_v2_00350", (-11.5, -3.2), 2.0),
    ("коридор 1,2 м и поворот", "floor_6_v2_00102", (-19.3, 6.3), 2.0),
]


def fig_examples(swd):
    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
    fig, axs = plt.subplots(len(EXAMPLES), len(ARMS), figsize=(15, 11.5))
    for row, (title, name, c, h) in zip(axs, EXAMPLES):
        for a, (arm, lab, col) in zip(row, ARMS):
            P = plan(arm, name)
            a.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255, zorder=0)
            a.plot(P[:, 0], P[:, 1], color=col, lw=1.6, zorder=3)
            inw = (np.abs(P[:, 0] - c[0]) < h) & (np.abs(P[:, 1] - c[1]) < h)
            idx = np.where(inw)[0]
            cl = np.array([swd.body_clearance(*p) for p in P[idx]]) if len(idx) else np.array([np.nan])
            for k in idx[::4]:
                robot(a, P[k], col, fill=False, lw=0.8, alpha=0.7, z=2)
            if len(idx):
                kmin = idx[np.argmin(cl)]
                robot(a, P[kmin], col, fill=True, lw=1.2, z=4)
            a.set_xlim(c[0] - h, c[0] + h)
            a.set_ylim(c[1] - h, c[1] + h)
            a.set_aspect("equal")
            a.tick_params(labelsize=8)
            a.set_xlabel("x, м", fontsize=9)
            a.set_ylabel("y, м", fontsize=9)
            a.set_title(f"{title}\n{lab}\n{name}, мин. зазор в окне {np.nanmin(cl):.2f} м", fontsize=10)
    legend = [
        Line2D([], [], color="0.3", lw=1.6, label="план (цвет — вариант)"),
        Patch(fc="none", ec="0.3", label="контур корпуса каждые 0,4 с"),
        Patch(fc=(0.5, 0.5, 0.5, 0.35), ec="0.3", label="поза с наименьшим зазором в окне"),
        Patch(fc="0.0", ec="k", label="стены (запечённая карта)"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(.5, .965))
    fig.suptitle("Проход дверного проёма и коридора: до и после центрирования, этаж 6", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .935), h_pad=2.0)
    fig.savefig(f"{OUT}/footprint_examples.png", dpi=130)


def main():
    swd = SignedWallDistance.from_map_yaml(MAP + ".yaml")
    ok = solved()
    names = sorted(ok["stack"] & ok[AFTER])
    table(swd, names)
    fig_hist(swd, names)
    fig_examples(swd)


if __name__ == "__main__":
    main()
