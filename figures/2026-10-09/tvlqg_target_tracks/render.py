#!/usr/bin/env python3
"""What the TVLQG control step is for, on job 241's full-stack tracks: plain
TVLQR (LTN, black) vs TVLQR with the slide k = -0.33 typed in (LTNg, blue), on
the three seed-0 plans where plain TVLQR overlapped a wall longest and the
sideslip arm did not. On the plain run, the TVLQG surface FILTER is replayed
(it drives nothing): the star marks where it first excludes k = 0 at 3 sigma,
i.e. from where an estimated k could stand in for the typed-in one.

    .venv/bin/python figures/2026-10-09/tvlqg_target_tracks/render.py
Input: run_data/2026-10-08_job241_sideslip-gate-fullstack (fetched from the VM 2026-10-09).
"""
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

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning")]
import compare_run as cr  # noqa: E402
import tvlqg_replay  # noqa: E402
from agx_planning.runtime_corrector import tvlqg  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(R, "run_data/2026-10-08_job241_sideslip-gate-fullstack")
C_V, C_W = 0.04, 0.13989471757878671       # PlannerConfig(mode="offline").c_v / c_w
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
COL = {"LTN": "k", "LTNg": "tab:blue"}
LAB = {"LTN": "plain TVLQR (k unknown)", "LTNg": "TVLQR with k = -0.33 typed in"}


def rows():
    out = {}
    for line in open(os.path.join(J, "all_rows.jsonl")):
        r = json.loads(line)
        tp = r.get("track_path") or ""
        parts = tp.split("/")
        if len(parts) < 3:
            continue
        out[(r["plan"], parts[-3], parts[-2])] = r
    return out


def contact_mask(tr):
    """Per sample: does the body rectangle touch a wall pixel (compare_run.wall_contact)."""
    d, r, ox, oy = cr._signed_wall_dist(6)
    nl = int(2 * cr.FP_HALF_LENGTH / cr.FP_SPACING) + 1
    nw = int(2 * cr.FP_HALF_WIDTH / cr.FP_SPACING) + 1
    lx = np.linspace(-cr.FP_HALF_LENGTH, cr.FP_HALF_LENGTH, nl)
    wy = np.linspace(-cr.FP_HALF_WIDTH, cr.FP_HALF_WIDTH, nw)
    body = np.vstack([np.c_[lx, np.full(nl, cr.FP_HALF_WIDTH)], np.c_[lx, np.full(nl, -cr.FP_HALF_WIDTH)],
                      np.c_[np.full(nw, cr.FP_HALF_LENGTH), wy], np.c_[np.full(nw, -cr.FP_HALF_LENGTH), wy]])
    c, s = np.cos(tr[:, 3])[:, None], np.sin(tr[:, 3])[:, None]
    qx = tr[:, 1:2] + body[None, :, 0] * c - body[None, :, 1] * s
    qy = tr[:, 2:3] + body[None, :, 0] * s + body[None, :, 1] * c
    col = np.clip(((qx - ox) / r).astype(int), 0, d.shape[1] - 1)
    row = np.clip(((qy - oy) / r).astype(int), 0, d.shape[0] - 1)
    return d[row, col].min(axis=1) < 0


def hull(x, y, th):
    L, W = cr.FP_HALF_LENGTH, cr.FP_HALF_WIDTH
    c, s = np.cos(th), np.sin(th)
    pts = [(L, W), (L, -W), (-L, -W), (-L, W)]
    return [(x + a * c - b * s, y + a * s + b * c) for a, b in pts]


def main():
    R_ = rows()
    cand = [p for (p, cell, seed), r in R_.items()
            if cell == "LTN" and seed == "s0" and (r.get("wall_overlap_s") or 0) > 0
            and (p, "LTNg", "s0") in R_ and not R_[(p, "LTNg", "s0")].get("wall_overlap_s")]
    pick = sorted(cand, key=lambda p: -R_[(p, "LTN", "s0")]["wall_overlap_s"])[:3]

    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]

    fig, axs = plt.subplots(3, 3, figsize=(16, 13), gridspec_kw=dict(height_ratios=[2.2, 2.2, 1.2]))
    for j, p in enumerate(pick):
        a_full, a_zoom, a_k = axs[0, j], axs[1, j], axs[2, j]
        tr = {}
        for cell in ("LTN", "LTNg"):
            tr[cell] = np.load(os.path.join(J, cell, "s0", f"track_w{_worker(R_, p, cell)}_ours-lib_{p}.npz"))
        plan = tr["LTN"]["plan_poses"]
        T = tr["LTN"]["track"]
        hit = contact_mask(T)
        ih = np.nonzero(hit)[0]
        # Zoom on the longest contact episode of the plain run.
        cx, cy = T[ih[len(ih) // 2], 1], T[ih[len(ih) // 2], 2]
        h = tvlqg_replay.replay(os.path.join(J, "LTN", "s0", f"track_w{_worker(R_, p, 'LTN')}_ours-lib_{p}.npz"),
                                C_V, C_W, tvlqg.TVLQGConfig(), np.random.default_rng(0))
        ex = np.abs(h[:, 1]) > 3.0 * h[:, 2]
        t_ex = h[np.argmax(ex), 0] if ex.any() else np.nan
        i_ex = np.searchsorted(T[:, 0] - T[0, 0], t_ex) if ex.any() else None
        t_hit = T[ih[0], 0] - T[0, 0]
        for ax, zoom in ((a_full, False), (a_zoom, True)):
            ax.imshow(img, cmap="gray", extent=ext, vmin=0, vmax=255)
            ax.plot(plan[:, 0], plan[:, 1], c="tab:green", lw=1.2, label="PMP plan")
            for cell in ("LTN", "LTNg"):
                t = tr[cell]["track"]
                ax.plot(t[:, 1], t[:, 2], c=COL[cell], lw=1.4, label=LAB[cell])
            if zoom:
                for cell in ("LTN", "LTNg"):
                    t = tr[cell]["track"]
                    near = np.hypot(t[:, 1] - cx, t[:, 2] - cy) < 1.3
                    for i in np.nonzero(near)[0][::4]:
                        ax.add_patch(Polygon(hull(*t[i, 1:4]), closed=True, fill=False,
                                             ec=COL[cell], lw=0.6, alpha=0.6))
                for i in ih[::2]:
                    if np.hypot(T[i, 1] - cx, T[i, 2] - cy) < 1.3:
                        ax.add_patch(Polygon(hull(*T[i, 1:4]), closed=True, fill=True,
                                             fc="tab:red", ec="none", alpha=0.35))
                ax.set_xlim(cx - 1.3, cx + 1.3)
                ax.set_ylim(cy - 1.3, cy + 1.3)
                ax.set_title("zoom: plain TVLQR's wall contact (red hulls)", fontsize=9)
            else:
                if i_ex is not None:
                    ax.plot(T[i_ex, 1], T[i_ex, 2], "*", ms=16, c="gold", mec="k",
                            label="filter knows k ≠ 0 (3σ)")
                ax.plot(T[ih, 1], T[ih, 2], ".", ms=4, c="tab:red", label="plain: body on a wall")
                ax.plot([cx], [cy], "s", ms=22, mfc="none", mec="tab:red")
                ax.set_xlim(plan[:, 0].min() - 1, plan[:, 0].max() + 1)
                ax.set_ylim(plan[:, 1].min() - 1, plan[:, 1].max() + 1)
                r0, r1 = R_[(p, "LTN", "s0")], R_[(p, "LTNg", "s0")]
                ax.set_title(f"{p}\nwall overlap: plain {r0['wall_overlap_s']} s, typed-in k "
                             f"{r1['wall_overlap_s']} s; final_err {r0['final_err']:.2f} / "
                             f"{r1['final_err']:.2f} m", fontsize=9)
            ax.set_aspect("equal")
        a_k.fill_between(h[:, 0], h[:, 1] - 3 * h[:, 2], h[:, 1] + 3 * h[:, 2],
                         color="tab:blue", alpha=0.25, lw=0)
        a_k.plot(h[:, 0], h[:, 1], c="tab:blue", lw=1)
        a_k.axhline(-0.331, c="tab:red", ls="--", lw=1, label="Gazebo k = -0.33")
        a_k.axhline(0.0, c="0.4", ls=":", lw=1, label="k = 0 (what plain TVLQR assumes)")
        if np.isfinite(t_ex):
            a_k.axvline(t_ex, c="gold", lw=2, label=f"k ≠ 0 known at {t_ex:.1f} s")
        a_k.axvline(t_hit, c="tab:red", lw=1, label=f"first wall contact {t_hit:.1f} s")
        a_k.set_ylim(-0.9, 0.5)
        a_k.set_xlabel("t [s] (plain TVLQR run)")
        a_k.set_ylabel("k̂ ± 3σ [s]")
        a_k.legend(fontsize=7, loc="lower right")
        print(f"{p}: k!=0 known at {t_ex:.1f} s, first contact {t_hit:.1f} s, final k "
              f"{h[-1,1]:+.3f} +- {h[-1,2]:.3f}")
    axs[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Job 241 (Gazebo, full stack, seed 0): plain TVLQR hits door jambs; with the slide k "
                 "typed in it does not. TVLQG's job: get blue with k ESTIMATED (the star), not typed in.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tvlqg_target_tracks.png"), dpi=110)


def _worker(R_, p, cell):
    tp = R_[(p, cell, "s0")]["track_path"]
    return os.path.basename(tp).split("_")[1][1:]


if __name__ == "__main__":
    main()
