#!/usr/bin/env python3
"""C1 (job 230, #41/#47): every arm with vs without the supervisory layer.

    .venv/bin/python figures/2026-10-07/c1_layer_regression/render.py

Reads run_data/2026-10-07_job230_c1-supervisor/ (mirror of the VM's ~/run_data,
gitignored). Top: arrivals of 40 per arm, layer off vs on, truth and amcl.
Below: four example plans (truth). Left: map, plan, both tracks, footprints of
the layer-on run every 2 s, trigger and stop marks. Right: the layer-on run's
hull clearance and stop/trigger state over time. Russian labels.
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
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
DATA = os.path.join(ROOT, "run_data/2026-10-07_job230_c1-supervisor")
MAP = os.path.join(ROOT, "src/rudn-ordjo-building/maps/floor_6")
sys.path.insert(0, os.path.join(HERE, "../s1_supervisor"))
from render import robot, at  # noqa: E402

ARMS = [("L", "наш (TVLQR)"), ("M", "MPPI"), ("R", "RPP"), ("V", "VPP"), ("G", "Graceful"), ("K", "GMPC")]
EXAMPLES = [("V", "00105"), ("R", "00249"), ("L", "00291"), ("K", "00249")]
# ours-lib: ~/supervisor_state (Decision.FIELDS + node extras); nav2: nsup.
CLR, STOP, TRIG = 7, 9, 10
OUT = {"arrived": "доехал", "failed": "не доехал", "timeout": "таймаут", "planner-timeout": "таймаут планировщика"}


def rows(cfg):
    return {(r := json.loads(l))["plan"]: r for l in open(os.path.join(DATA, cfg, "rows.s0.jsonl"))}


def track(cfg, plan):
    return np.load(glob.glob(os.path.join(DATA, cfg, "s0", f"track_*{plan}.npz"))[0], allow_pickle=True)


def sup(z):
    return (z["sup_t"], z["sup"]) if len(z["sup_t"]) else (z["nsup_t"], z["nsup"])


def main():
    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]

    fig = plt.figure(figsize=(16, 4.6 + 5.6 * len(EXAMPLES)))
    gs = fig.add_gridspec(1 + len(EXAMPLES), 2, height_ratios=[0.8] + [1] * len(EXAMPLES),
                          width_ratios=[1.15, 1])
    for j, (loc, name) in enumerate([("T", "истинная локализация"), ("A", "AMCL")]):
        ax = fig.add_subplot(gs[0, j])
        x = np.arange(len(ARMS))
        n = [sum(r["outcome"] == "arrived" for r in rows(a + loc + "N").values()) for a, _ in ARMS]
        s = [sum(r["outcome"] == "arrived" for r in rows(a + loc + "S").values()) for a, _ in ARMS]
        ax.bar(x - 0.2, n, 0.4, color="tab:orange", label="без слоя")
        ax.bar(x + 0.2, s, 0.4, color="tab:red", label="со слоем")
        for i in x:
            ax.text(i - 0.2, n[i] + 0.5, n[i], ha="center", fontsize=9)
            ax.text(i + 0.2, s[i] + 0.5, s[i], ha="center", fontsize=9)
        ax.set_xticks(x, [a[1] for a in ARMS])
        ax.set_ylim(0, 44)
        ax.set_ylabel("доехали, из 40 планов")
        ax.set_title(f"Прибытие (<0.5 м), {name}", fontsize=11)
        ax.legend(fontsize=9)

    for i, (a, plan) in enumerate(EXAMPLES, start=1):
        p = "floor_6_v2_" + plan
        rn, rs = rows(a + "TN")[p], rows(a + "TS")[p]
        zn, zs = track(a + "TN", plan), track(a + "TS", plan)
        P, trn, trs = zs["plan_poses"], zn["track"], zs["track"]
        st, S = sup(zs)
        am, at_ = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        am.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255, zorder=0)
        am.plot(P[:, 0], P[:, 1], color="0.35", lw=1.4, ls="--", zorder=1)
        am.plot(trn[:, 1], trn[:, 2], color="tab:orange", lw=1.8, zorder=2)
        am.plot(trs[:, 1], trs[:, 2], color="tab:red", lw=1.8, zorder=2)
        for t in np.arange(trs[0, 0], trs[-1, 0], 2.0):
            robot(am, at(trs, t), "tab:red", alpha=0.45)
        robot(am, trs[-1, 1:4], "tab:red", lw=1.6)
        robot(am, trn[-1, 1:4], "tab:orange", fill=False, lw=1.6)
        robot(am, P[-1], "tab:blue", fill=False, lw=2)
        if len(S):
            for k in np.where(np.diff(np.r_[0, S[:, TRIG]]) > 0.5)[0]:
                q = at(trs, st[k])
                am.plot(q[0], q[1], marker="*", ms=14, color="gold", mec="k", zorder=5)
            for k in np.where(np.diff(np.r_[0, S[:, STOP]]) > 0.5)[0]:
                q = at(trs, st[k])
                am.plot(q[0], q[1], marker="X", ms=11, color="k", zorder=5)
        xy = np.vstack([P[:, :2], trs[:, 1:3], trn[:, 1:3]])
        lo, hi = xy.min(0) - 0.8, xy.max(0) + 0.8
        am.set_xlim(lo[0], hi[0])
        am.set_ylim(lo[1], hi[1])
        am.set_aspect("equal")
        nm = dict(ARMS)[a]
        am.set_title(f"{nm}, план {plan}: без слоя {OUT.get(rn['outcome'], rn['outcome'])} {rn['final_err']:.2f} м, "
                     f"со слоем {OUT.get(rs['outcome'], rs['outcome'])} {rs['final_err']:.2f} м\n"
                     f"путь со слоем {rs['path_length']:.1f} м (без слоя {rn['path_length']:.1f} м), "
                     f"точки маршрута {rs.get("sup_waypoints", "?")}", fontsize=10)
        if len(S):
            tt = st - st[0]
            at_.plot(tt, S[:, CLR], color="tab:red", lw=1.3, label="зазор корпуса до стены, м")
            at_.fill_between(tt, 0, 1, where=S[:, STOP] > 0.5, transform=at_.get_xaxis_transform(),
                             color="k", alpha=0.25, lw=0, label="робот остановлен слоем")
            for k in np.where(np.diff(np.r_[0, S[:, TRIG]]) > 0.5)[0]:
                at_.axvline(tt[k], color="gold", lw=0.8)
            at_.set_ylim(-0.05, 1.0)
            at_.set_xlabel("время, с")
            at_.set_ylabel("зазор, м")
            at_.legend(fontsize=8, loc="upper right")
        at_.set_title("слой: жёлтые линии — перепланирование, серое — остановка", fontsize=9)

    fig.legend(handles=[
        Line2D([], [], color="0.35", ls="--", label="план"),
        Line2D([], [], color="tab:orange", lw=2, label="без слоя"),
        Line2D([], [], color="tab:red", lw=2, label="со слоем; контуры каждые 2 с"),
        Line2D([], [], color="gold", marker="*", mec="k", ls="", ms=12, label="срабатывание слоя"),
        Line2D([], [], color="k", marker="X", ls="", ms=10, label="остановка у стены"),
    ], loc="lower center", ncol=5, fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = os.path.join(HERE, "layer_off_vs_on.png")
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
