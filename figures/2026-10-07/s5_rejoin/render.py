#!/usr/bin/env python3
"""S5 (#45) re-join teacher figures. Reads gitignored run_data in the main checkout.

    .venv/bin/python figures/2026-10-07/s5_rejoin/render.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

MAIN = "/home/danya/university/clown-car/agx_navigation"
OUT = os.path.dirname(os.path.abspath(__file__))
SWEEP = f"{MAIN}/run_data/2026-10-07_rejoin-sweep"
SMOKE = f"{MAIN}/run_data/2026-10-07_rejoin-labels-smoke"
MAP = f"{MAIN}/src/rudn-ordjo-building/maps/floor_6"


def load(d):
    zs = [np.load(f) for f in sorted(glob.glob(f"{d}/shard_*.npz"))]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}


def rate(ok, x, bins):
    c, r, n = [], [], []
    for lo, hi in zip(bins, bins[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum():
            c.append(np.sqrt(lo * hi) if lo > 0 else hi / 2)
            r.append(100 * ok[sel].mean())
            n.append(sel.sum())
    return np.array(c), np.array(r), np.array(n)


def fig_rates(z):
    ok = z["m"] == 1
    tw = z["s"][:, 9]
    dv = np.hypot(z["s"][:, 0], z["s"][:, 1])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, x, bins, lab in ((ax[0], tw, [0.3, 0.5, 1, 1.5, 2, 3, 5, 10.01], "Окно возврата $T_w$, с"),
                            (ax[1], dv, [0, 0.05, 0.1, 0.2, 0.4, 0.8, 5], "Отклонение $|e_{xy}|$, м")):
        c, r, n = rate(ok, x, bins)
        a.plot(c, r, "o-", label="все $T_w$")
        if x is dv:
            c3, r3, _ = rate(ok[tw >= 3], dv[tw >= 3], bins)
            a.plot(c3, r3, "s--", label="$T_w \\geq$ 3 с")
            a.legend(fontsize=8)
        for ci, ri, ni in zip(c, r, n):
            a.annotate(f"n={ni}", (ci, ri), textcoords="offset points", xytext=(0, -14),
                       ha="center", fontsize=7)
        a.set_xscale("log")
        a.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xlabel(lab)
        a.set_ylabel("Доля решённых задач, %")
        a.set_ylim(0, 105)
        a.grid(alpha=0.3)
    ax[0].axvline(3.0, color="k", ls="--", lw=0.8)
    ax[0].text(3.1, 8, "$T_{w,min}$ = 3 с", fontsize=8)
    fig.suptitle(f"Решаемость задачи возврата на план (PMP, мин. усилие), N={len(ok)}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/solve_rate.png", dpi=150)


def fig_times(sweep, smoke):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(0, 4.5, 50)
    s_ok = smoke["solve_ms"]
    w_fail = sweep["solve_ms"][sweep["m"] == 0]
    ax.hist(s_ok, bins, alpha=0.7, label=f"$T_w\\geq$3 с (смоук, n={len(s_ok)}): "
            f"медиана {np.median(s_ok):.1f} мс, p90 {np.percentile(s_ok, 90):.1f} мс")
    if len(w_fail):
        ax.hist(w_fail, bins, alpha=0.6, label=f"неудачи (развёртка, n={len(w_fail)}): "
                f"медиана {np.median(w_fail):.0f} мс")
    ax.set_xscale("log")
    ax.set_xlabel("Время решения, мс (1 ядро)")
    ax.set_ylabel("Число задач")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/solve_time.png", dpi=150)


def fig_examples(z):
    import yaml
    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
    ok = np.where((z["m"] == 1) & (z["s"][:, 9] >= 3.0))[0]
    dv = np.hypot(z["s"][ok, 0], z["s"][ok, 1])
    pick = ok[np.argsort(dv)[np.linspace(len(ok) * 0.5, len(ok) - 1, 6).astype(int)]]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.5))
    for a, i in zip(axs.ravel(), pick):
        plan = np.load(f"{MAIN}/traj_data_v2/{z['plan'][i]}.npz")["poses"]
        k, tgt, tr = z["k"][i], z["tgt"][i], z["traj"][i]
        a.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255)
        a.plot(plan[:, 0], plan[:, 1], "b-", lw=1, label="номинальный план")
        a.plot(plan[k:tgt + 1, 0], plan[k:tgt + 1, 1], "b-", lw=2.5, alpha=0.5,
               label="участок плана $k..k+T_w/dt$")
        a.plot(tr[:, 0], tr[:, 1], "r-", lw=1.8, label="траектория возврата (PMP)")
        a.plot(*tr[0, :2], "ro", ms=6, label="отклонённое начальное состояние")
        a.quiver(tr[0, 0], tr[0, 1], np.cos(tr[0, 2]), np.sin(tr[0, 2]), color="r",
                 scale=12, width=0.008)
        a.plot(*plan[k, :2], "bo", ms=4)
        a.plot(*plan[tgt, :2], "gs", ms=6, label="точка возврата на план")
        pts = np.vstack([plan[k:tgt + 1, :2], tr[:, :2]])
        c, h = pts.mean(0), max(np.ptp(pts, 0).max() / 2 + 0.6, 1.2)
        a.set_xlim(c[0] - h, c[0] + h)
        a.set_ylim(c[1] - h, c[1] + h)
        a.set_title(f"{z['plan'][i]}  $|e_{{xy}}|$={np.hypot(*z['s'][i, :2]):.2f} м, "
                    f"$e_\\theta$={np.degrees(z['s'][i, 2]):.0f}°, $T_w$={z['s'][i, 9]:.1f} с",
                    fontsize=8)
        a.set_xlabel("x, м")
        a.set_ylabel("y, м")
        a.set_aspect("equal")
    axs[0, 0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Примеры возврата на план, этаж 6 (запечённая карта)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/examples.png", dpi=150)


if __name__ == "__main__":
    sweep, smoke = load(SWEEP), load(SMOKE)
    fig_rates(sweep)
    fig_times(sweep, smoke)
    fig_examples(smoke)
    for name, z in (("sweep", sweep), ("smoke", smoke)):
        ok = z["m"] == 1
        tw = z["s"][:, 9]
        print(f"{name}: n={len(ok)} solved {100 * ok.mean():.1f}%  "
              f"T_w>=3: {100 * ok[tw >= 3].mean():.1f}% (n={(tw >= 3).sum()})  "
              f"T_w<1: {100 * ok[tw < 1].mean() if (tw < 1).any() else float('nan'):.1f}%  "
              f"ms med {np.median(z['solve_ms']):.1f} p90 {np.percentile(z['solve_ms'], 90):.1f}  "
              f"ok-only med {np.median(z['solve_ms'][ok]):.1f} p90 {np.percentile(z['solve_ms'][ok], 90):.1f}")
