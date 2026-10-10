#!/usr/bin/env python3
"""Job 244: what the covariance-triggered PMP re-join (scenario b) did on
individual runs. Per column, one plan/seed:

  top     the map, the library plan (green), TVLQG alone (LTNq, black) and
          TVLQG + the deviation re-join (LTDq, orange); each off-plan trigger
          as a marker coloured by what the re-join did (ok / blocked / fail);
          red dots = LTDq's body on a wall.
  middle  zoom on the triggers, LTDq body outlines (red fill = on a wall).
  bottom  distance from the true pose to the library plan for both runs, the
          per-sample 3.29-sigma bound the supervisor compared against
          (offplan_at), trigger times and LTDq's wall contact.

    .venv/bin/python figures/2026-10-10/rejoin_runs/render.py [--runs 00152:0,00355:1] [--out f.png]
Input: run_data/2026-10-09_job244_tvlqg-deviation (fetched from the VM).
"""
import argparse
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
R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "figures/2026-10-09/tvlqg_target_tracks")]
import render as target  # noqa: E402  (contact_mask, hull)

J = os.path.join(R, "run_data/2026-10-09_job244_tvlqg-deviation")
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
# ~/supervisor_state columns: Decision.FIELDS (0-10), then node._SUP_EXTRA.
S_TRIG, S_OK, S_FAIL, S_BLK, S_STALE, S_REASON, S_AT = 10, 13, 14, 15, 16, 19, 20
OUTC = {S_OK: ("ok", "tab:green"), S_FAIL: ("fail", "tab:red"),
        S_BLK: ("blocked", "tab:purple"), S_STALE: ("stale", "tab:gray")}


def rows():
    out = {}
    for line in open(os.path.join(J, "all_rows.jsonl")):
        r = json.loads(line)
        parts = (r.get("track_path") or "").split("/")
        if len(parts) >= 3:
            out[(r["plan"][-5:], parts[-3], r["seed"])] = r
    return out


def load(r):
    return np.load(os.path.join(J, *r["track_path"].split("/")[-3:]))


def dist_to_plan(tr, plan):
    """Distance from each track pose to the plan polyline (nearest segment)."""
    a, b = plan[:-1, :2], plan[1:, :2]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    out = np.empty(len(tr))
    for i, p in enumerate(tr[:, 1:3]):
        t = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
        out[i] = np.hypot(*(a + t[:, None] * ab - p).T).min()
    return out


def triggers(z):
    """(sim_t, outcome column) per off-plan trigger: the first counter that
    moves after it says what the re-join did."""
    st, s = z["sup_t"], z["sup"]
    idx = np.nonzero((s[:, S_TRIG] > 0.5) & (s[:, S_REASON] == 4))[0]
    out = []
    for n, i in enumerate(idx):
        end = idx[n + 1] if n + 1 < len(idx) else len(s)
        oc = None
        for j in range(i, end):
            moved = [c for c in OUTC if s[j, c] > s[max(i - 1, 0), c]]
            if moved:
                oc = moved[0]
                break
        out.append((st[i], oc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="00152:0,00355:1,00208:2,00320:2")
    ap.add_argument("--out", default="rejoin_runs.png")
    a = ap.parse_args()
    runs = [(p, int(s)) for p, s in (x.split(":") for x in a.runs.split(","))]
    R_ = rows()
    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]

    fig, axs = plt.subplots(3, len(runs), figsize=(5.4 * len(runs), 14), squeeze=False,
                            gridspec_kw=dict(height_ratios=[2.2, 2.2, 1.3]))
    for j, (p, s) in enumerate(runs):
        rb, rd = R_[(p, "LTNq", s)], R_[(p, "LTDq", s)]
        zb, zd = load(rb), load(rd)
        plan = zd["plan_poses"]
        tb, td = zb["track"], zd["track"]
        hit = target.contact_mask(td)
        trg = triggers(zd)
        ti = [min(np.searchsorted(td[:, 0], t), len(td) - 1) for t, _ in trg]
        cx, cy = (td[ti, 1].mean(), td[ti, 2].mean()) if ti else (td[-1, 1], td[-1, 2])
        for ax, zoom in ((axs[0, j], False), (axs[1, j], True)):
            ax.imshow(img, cmap="gray", extent=ext, vmin=0, vmax=255)
            ax.plot(plan[:, 0], plan[:, 1], c="tab:green", lw=1.2, label="library plan")
            ax.plot(tb[:, 1], tb[:, 2], c="k", lw=1.2,
                    label=f"TVLQG alone: {rb['outcome']}, {rb['final_err']:.2f} m")
            ax.plot(td[:, 1], td[:, 2], c="tab:orange", lw=1.4,
                    label=f"+ re-join: {rd['outcome']}, {rd['final_err']:.2f} m")
            ax.plot(td[hit, 1], td[hit, 2], ".", ms=3, c="tab:red", label="re-join run: body on a wall")
            ax.plot(*rd["goal_xy"], "x", ms=10, mew=2, c="tab:blue", label="goal")
            for n, ((t, oc), i) in enumerate(zip(trg, ti)):
                name, col = OUTC.get(oc, ("?", "k"))
                ax.plot(td[i, 1], td[i, 2], "o", ms=11, mfc=col, mec="k", zorder=5,
                        label=f"trigger: re-join {name}")
                ax.annotate(str(n + 1), (td[i, 1], td[i, 2]), ha="center", va="center",
                            fontsize=7, color="w", zorder=6)
            if zoom:
                near = np.hypot(td[:, 1] - cx, td[:, 2] - cy) < 1.6
                for i in np.nonzero(near)[0][::5]:
                    ax.add_patch(Polygon(target.hull(*td[i, 1:4]), closed=True, fill=bool(hit[i]),
                                         fc="tab:red", alpha=0.35 if hit[i] else 0.6,
                                         ec="tab:orange", lw=0.5))
                ax.set_xlim(cx - 1.6, cx + 1.6)
                ax.set_ylim(cy - 1.6, cy + 1.6)
                ax.set_title("zoom on the triggers (re-join run body outlines)", fontsize=9)
            else:
                xs = np.r_[plan[:, 0], td[:, 1]]
                ys = np.r_[plan[:, 1], td[:, 2]]
                ax.set_xlim(xs.min() - 1, xs.max() + 1)
                ax.set_ylim(ys.min() - 1, ys.max() + 1)
                prof = ", ".join(sorted({q["profile"] for q in rd["patches"]}))
                ax.set_title(f"floor_6_v2_{p}, seed {s} ({prof})\nwall overlap: alone "
                             f"{rb['wall_overlap_s'] or 0:.1f} s, re-join {rd['wall_overlap_s'] or 0:.1f} s",
                             fontsize=9)
                h, l_ = ax.get_legend_handles_labels()
                keep = dict(zip(l_, h))
                ax.legend(keep.values(), keep.keys(), fontsize=6.5, loc="best")
            ax.set_aspect("equal")
        ax = axs[2, j]
        t0 = td[0, 0]
        ax.plot(tb[:, 0] - tb[0, 0], dist_to_plan(tb, plan), c="k", lw=1, label="alone: dist to plan")
        ax.plot(td[:, 0] - t0, dist_to_plan(td, plan), c="tab:orange", lw=1.2, label="re-join: dist to plan")
        st, ss = zd["sup_t"], zd["sup"]
        ax.plot(st - t0, ss[:, S_AT], c="tab:blue", ls="--", lw=1, label="3.29σ bound at k*")
        for n, (t, oc) in enumerate(trg):
            ax.axvline(t - t0, c=OUTC.get(oc, ("?", "k"))[1], lw=1.5)
            ax.text(t - t0, 0.02, str(n + 1), fontsize=7, transform=ax.get_xaxis_transform())
        ax.fill_between(td[:, 0] - t0, 0, 1, where=hit, color="tab:red", alpha=0.2,
                        transform=ax.get_xaxis_transform(), label="re-join run on a wall")
        ax.set_ylim(0, max(1.5, float(np.nanmax(ss[:, S_AT])) * 1.2))
        ax.set_xlabel("t [s] from the corrector's first state message")
        ax.set_ylabel("m")
        ax.legend(fontsize=6.5, loc="upper left")
        print(p, s, "alone", rb["outcome"], rb["final_err"], "| re-join", rd["outcome"], rd["final_err"],
              "| triggers", [(round(t - t0, 1), OUTC.get(oc, ("?",))[0]) for t, oc in trg])
    fig.suptitle("Job 244 (Gazebo, truth pose): TVLQG alone vs TVLQG + PMP re-join on the "
                 "covariance-derived off-plan bound", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, a.out), dpi=110)


if __name__ == "__main__":
    main()
