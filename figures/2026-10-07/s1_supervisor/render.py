#!/usr/bin/env python3
"""S1 (#41) deliverable: our stack (ours-lib, TVLQR) with and without the supervisor.

    .venv/bin/python figures/2026-10-07/s1_supervisor/render.py

Reads run_data/2026-10-07_s1-supervisor-smoke/{sup4,nosup}/ (mirror of the VM's
~/run_data, gitignored). Footprint style of figures/2026-09-30 (advisor's
preference), Russian labels. One row per plan:
  left  -- baked map, plan, waypoints (reach radius), both ground-truth tracks,
           footprints of the supervised run every 2 s, trigger and stop marks;
  right -- the supervised run over time: segment lag [s], speed scale, and
           hull clearance against the plan's own clearance at k*.
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
from matplotlib.patches import Circle, Patch, Polygon  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
DATA = os.path.join(ROOT, "run_data/2026-10-07_s1-supervisor-smoke")
MAP = os.path.join(ROOT, "src/rudn-ordjo-building/maps/floor_6")
sys.path.insert(0, os.path.join(ROOT, "src/agx_navigation/agx_planning"))
from agx_planning.supervisor.core import SupervisorCore  # noqa: E402

BODY_L, BODY_W = 0.61, 0.50
WHEEL_X, WHEEL_Y, WHEEL_L, WHEEL_W = 0.23, 0.21, 0.18, 0.08
# ~/supervisor_state columns: Decision.FIELDS then the node's _SUP_EXTRA.
LAG, CLR, SCALE, STOP, TRIG, OVERLAY, CPLAN = 4, 7, 8, 9, 10, 11, 18


def robot(ax, q, color, fill=True, alpha=1.0, ls="-", lw=1.0, z=3):
    c, s_ = np.cos(q[2]), np.sin(q[2])
    R = np.array([[c, -s_], [s_, c]])

    def rect(cx, cy, L, W):
        p = np.array([[-L, -W], [L, -W], [L, W], [-L, W]]) / 2 + [cx, cy]
        return p @ R.T + q[:2]
    kw = dict(ec=color, lw=lw, ls=ls, zorder=z)
    ax.add_patch(Polygon(rect(0, 0, BODY_L, BODY_W), fc=color if fill else "none",
                         alpha=alpha * (0.3 if fill else 1), **kw))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.add_patch(Polygon(rect(sx * WHEEL_X, sy * WHEEL_Y, WHEEL_L, WHEEL_W),
                                 fc="0.2" if fill else "none", alpha=alpha, **kw))
    nose = np.array([[0.02, -0.14], [BODY_L / 2 - 0.02, 0], [0.02, 0.14]])
    ax.plot(*(nose @ R.T + q[:2]).T, color=color, lw=lw + 0.6, alpha=alpha, zorder=z + .1)


def rows(arm):
    out = {}
    p = os.path.join(DATA, arm, "rows.jsonl")
    for line in open(p):
        r = json.loads(line)
        out[r["plan"]] = r
    return out


def track(r):
    rel = r["track_path"].split("2026-10-07_s1-supervisor-smoke/", 1)[1]
    return np.load(os.path.join(DATA, rel), allow_pickle=True)


def at(tr, t):
    i = np.clip(np.searchsorted(tr[:, 0], t), 0, len(tr) - 1)
    return tr[i, 1:4]


def main():
    sup, nos = rows("sup4"), rows("nosup")
    plans = [p for p in sup if p in nos]
    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]

    fig, axs = plt.subplots(len(plans), 2, figsize=(16, 6.2 * len(plans)),
                            gridspec_kw=dict(width_ratios=[1.15, 1]))
    axs = np.atleast_2d(axs)
    for (am, at_), plan in zip(axs, plans):
        rs, rn = sup[plan], nos[plan]
        zs, zn = track(rs), track(rn)
        P = zs["plan_poses"]
        trs, trn = zs["track"], zn["track"]
        st, S = zs["sup_t"], zs["sup"]
        core = SupervisorCore()
        core.set_plan(P)
        t0 = st[0] if len(st) else trs[0, 0]

        am.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255, zorder=0)
        am.plot(P[:, 0], P[:, 1], color="0.35", lw=1.4, ls="--", zorder=1)
        for w in core.wp_index:
            am.add_patch(Circle(P[w, :2], 0.35, fc="none", ec="tab:blue", lw=0.8, alpha=0.6, zorder=1))
        am.plot(trn[:, 1], trn[:, 2], color="tab:orange", lw=1.8, zorder=2)
        am.plot(trs[:, 1], trs[:, 2], color="tab:red", lw=1.8, zorder=2)
        for t in np.arange(t0, trs[-1, 0], 2.0):
            robot(am, at(trs, t), "tab:red", alpha=0.55)
        robot(am, trs[-1, 1:4], "tab:red", lw=1.6)
        robot(am, trn[-1, 1:4], "tab:orange", fill=False, lw=1.6)
        robot(am, P[-1], "tab:blue", fill=False, lw=2)
        if len(S):
            for k in np.where(np.diff(np.r_[0, S[:, TRIG]]) > 0.5)[0]:
                q = at(trs, st[k])
                am.plot(q[0], q[1], marker="*", ms=15, color="gold", mec="k", zorder=5)
            for k in np.where(np.diff(np.r_[0, S[:, STOP]]) > 0.5)[0]:
                q = at(trs, st[k])
                am.plot(q[0], q[1], marker="X", ms=12, color="k", zorder=5)
        allxy = np.vstack([P[:, :2], trs[:, 1:3], trn[:, 1:3]])
        lo, hi = allxy.min(0) - 0.8, allxy.max(0) + 0.8
        am.set_xlim(lo[0], hi[0])
        am.set_ylim(lo[1], hi[1])
        am.set_aspect("equal")
        cs = rs.get("sup_counts", {})
        am.set_title(
            f"{plan}: без слоя {rn['final_err']:.2f} м, со слоем {rs['final_err']:.2f} м до цели\n"
            f"точки маршрута {rs.get('sup_waypoints', ['?', '?'])[0]}/{rs.get('sup_waypoints', ['?', '?'])[1]}, "
            f"возвратов {cs.get('rejoin_ok', 0)} (неудач {cs.get('rejoin_fail', 0)}, "
            f"запрещено стенами {cs.get('rejoin_blocked', 0)}), перепланирований {cs.get('full_replans', 0)}",
            fontsize=10)

        if len(S):
            tt = st - t0
            at_.plot(tt, S[:, LAG], color="tab:purple", lw=1.4, label="отставание на отрезке, с")
            ov = S[:, OVERLAY] > 0.5
            at_.fill_between(tt, 0, 1, where=ov, transform=at_.get_xaxis_transform(),
                             color="tab:green", alpha=0.15, lw=0)
            for k in np.where(np.diff(np.r_[0, S[:, 3]]) > 0.5)[0]:
                at_.axvline(tt[k], color="tab:blue", lw=0.5, alpha=0.5)
            at_.set_xlabel("время от начала движения, с")
            at_.set_ylabel("отставание, с")
            ab = at_.twinx()
            ab.plot(tt, S[:, CLR], color="tab:red", lw=1.2, label="зазор корпуса до стены, м")
            ab.plot(tt, S[:, CPLAN], color="0.4", lw=1.2, ls="--", label="зазор плана в той же точке, м")
            ab.plot(tt, S[:, SCALE], color="tab:olive", lw=1.0, label="множитель скорости")
            ab.set_ylim(-0.1, 1.2)
            ab.set_ylabel("зазор, м / множитель")
            h1, l1 = at_.get_legend_handles_labels()
            h2, l2 = ab.get_legend_handles_labels()
            at_.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        at_.set_title("слой со стороны робота: синие линии — достигнута точка маршрута, "
                      "зелёное — идёт возврат на план", fontsize=9)

    fig.legend(handles=[
        Line2D([], [], color="0.35", ls="--", label="план PMP"),
        Patch(fc="none", ec="tab:blue", label="точки маршрута каждые 5% длины (радиус робота 0.35 м)"),
        Line2D([], [], color="tab:orange", lw=2, label="без слоя (TVLQR)"),
        Line2D([], [], color="tab:red", lw=2, label="со слоем; контуры каждые 2 с"),
        Line2D([], [], color="gold", marker="*", mec="k", ls="", ms=12, label="срабатывание слоя"),
        Line2D([], [], color="k", marker="X", ls="", ms=10, label="остановка у стены"),
        Patch(fc="none", ec="tab:blue", lw=2, label="цель"),
    ], loc="lower center", ncol=4, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(HERE, "with_without_layer.png")
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
