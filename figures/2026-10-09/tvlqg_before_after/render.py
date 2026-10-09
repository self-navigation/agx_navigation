#!/usr/bin/env python3
"""TVLQG before/after on job 243's full-stack tracks: plain TVLQR (black),
TVLQR with k = -0.33 typed in (blue), and TVLQG, with k estimated live by the
surface filter (orange). Plans are picked where the layer's effect is most
visible: plain TVLQR's wall overlap, summed over both seeds, largest among
plans where TVLQG is contact-free on both seeds. Bottom row: TVLQG's live
k_hat +- 3 sigma and the moment it rejected k = 0 (the controller's switch).

    .venv/bin/python figures/2026-10-09/tvlqg_before_after/render.py [--plans p1,p2,p3] [--seed 0] [--out f.png] [--zoom-on LTNq]
Input: run_data/2026-10-09_job243_tvlqg-fullstack (fetched from the VM).
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

J = os.path.join(R, "run_data/2026-10-09_job243_tvlqg-fullstack")
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
CELLS = ("LTN", "LTNg", "LTNq")
COL = {"LTN": "k", "LTNg": "tab:blue", "LTNq": "tab:orange"}
LAB = {"LTN": "plain TVLQR (k = 0)", "LTNg": "TVLQR, k = -0.33 typed in",
       "LTNq": "TVLQG: k estimated live"}
I_K, I_S, I_KB = 16, 17, 18      # CorrectionDiagnostics.FIELDS: k_hat, sigma_k, k_bar


def rows():
    out = {}
    for line in open(os.path.join(J, "all_rows.jsonl")):
        r = json.loads(line)
        parts = (r.get("track_path") or "").split("/")
        if len(parts) >= 3:
            out[(r["plan"], parts[-3], parts[-2])] = r
    return out


def ov(r):
    return (r or {}).get("wall_overlap_s") or 0.0


def pick(R_, n):
    plans = sorted({p for p, _, _ in R_})
    ok = [p for p in plans if all((p, c, s) in R_ for c in CELLS for s in ("s0", "s1"))]
    clean = [p for p in ok if ov(R_[(p, "LTNq", "s0")]) == 0 and ov(R_[(p, "LTNq", "s1")]) == 0]
    score = {p: ov(R_[(p, "LTN", "s0")]) + ov(R_[(p, "LTN", "s1")]) for p in clean}
    return sorted(clean, key=lambda p: -score[p])[:n]


def track_file(r):
    p = r["track_path"]
    return os.path.join(J, *p.split("/")[-3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--out", default="tvlqg_before_after.png")
    ap.add_argument("--zoom-on", default="LTN", help="cell whose contact the zoom centres on")
    a = ap.parse_args()
    R_ = rows()
    seed = "s" + a.seed
    plans = a.plans.split(",") if a.plans else pick(R_, 3)
    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]

    fig, axs = plt.subplots(3, len(plans), figsize=(5.4 * len(plans), 13), squeeze=False,
                            gridspec_kw=dict(height_ratios=[2.2, 2.2, 1.2]))
    for j, p in enumerate(plans):
        a_full, a_zoom, a_k = axs[0, j], axs[1, j], axs[2, j]
        z = {c: np.load(track_file(R_[(p, c, seed)])) for c in CELLS}
        plan = z["LTN"]["plan_poses"]
        T = z[a.zoom_on]["track"]
        hit = target.contact_mask(T)
        ih = np.nonzero(hit)[0]
        if len(ih):
            cx, cy = T[ih[len(ih) // 2], 1], T[ih[len(ih) // 2], 2]
        else:
            cx, cy = T[len(T) // 2, 1], T[len(T) // 2, 2]
        for ax, zoom in ((a_full, False), (a_zoom, True)):
            ax.imshow(img, cmap="gray", extent=ext, vmin=0, vmax=255)
            ax.plot(plan[:, 0], plan[:, 1], c="tab:green", lw=1.2, label="PMP plan")
            for c in CELLS:
                t = z[c]["track"]
                ax.plot(t[:, 1], t[:, 2], c=COL[c], lw=1.4, label=LAB[c])
            if zoom:
                for c in ("LTN", "LTNq"):
                    t = z[c]["track"]
                    hc = target.contact_mask(t)
                    near = np.hypot(t[:, 1] - cx, t[:, 2] - cy) < 1.3
                    for i in np.nonzero(near)[0][::4]:
                        ax.add_patch(Polygon(target.hull(*t[i, 1:4]), closed=True,
                                             fill=bool(hc[i]), fc="tab:red", alpha=0.35 if hc[i] else 0.7,
                                             ec=COL[c], lw=0.6))
                ax.set_xlim(cx - 1.3, cx + 1.3)
                ax.set_ylim(cy - 1.3, cy + 1.3)
                ax.set_title(f"zoom at {LAB[a.zoom_on].split(':')[0].split(',')[0]}'s contact: body outlines,\nblack = plain, orange = TVLQG, "
                             "red fill = on a wall", fontsize=9)
            else:
                ax.plot(T[ih, 1], T[ih, 2], ".", ms=4, c="tab:red", label=f"{LAB[a.zoom_on].split(':')[0].split(',')[0]}: body on a wall")
                ax.plot([cx], [cy], "s", ms=22, mfc="none", mec="tab:red")
                d, dt = z["LTNq"]["diag"], z["LTNq"]["diag_t"]
                act = np.nonzero(np.isfinite(d[:, I_KB]) & (d[:, I_KB] != 0.0))[0]
                if len(act):
                    tq = z["LTNq"]["track"]
                    i = np.searchsorted(tq[:, 0], dt[act[0]])
                    i = min(i, len(tq) - 1)
                    ax.plot(tq[i, 1], tq[i, 2], "*", ms=16, c="gold", mec="k",
                            label="TVLQG switches on (k = 0 rejected)")
                ax.set_xlim(plan[:, 0].min() - 1, plan[:, 0].max() + 1)
                ax.set_ylim(plan[:, 1].min() - 1, plan[:, 1].max() + 1)
                o = [f"{ov(R_[(p, c, s)]):.1f}" for c in CELLS for s in ("s0", "s1")]
                ax.set_title(f"{p}  (seed {a.seed} drawn)\nwall overlap [s], seed 0/1: plain {o[0]}/{o[1]}, "
                             f"typed-in {o[2]}/{o[3]}, TVLQG {o[4]}/{o[5]}", fontsize=9)
            ax.set_aspect("equal")
        d, dt = z["LTNq"]["diag"], z["LTNq"]["diag_t"]
        t = dt - z["LTNq"]["track"][0, 0]
        k, s, kb = d[:, I_K], d[:, I_S], d[:, I_KB]
        a_k.fill_between(t, k - 3 * s, k + 3 * s, color="tab:orange", alpha=0.25, lw=0)
        a_k.plot(t, k, c="tab:orange", lw=1, label="k̂ ± 3σ (TVLQG run)")
        a_k.plot(t, kb, c="k", lw=1.2, ls="-", alpha=0.7, label="k the controller uses")
        a_k.axhline(-0.331, c="tab:red", ls="--", lw=1, label="Gazebo k = -0.33")
        act = np.nonzero(np.isfinite(kb) & (kb != 0.0))[0]
        if len(act):
            a_k.axvline(t[act[0]], c="gold", lw=2, label=f"switch at {t[act[0]]:.1f} s")
        a_k.set_ylim(-0.9, 0.5)
        a_k.set_xlabel("t [s] (TVLQG run)")
        a_k.set_ylabel("k [s]")
        a_k.legend(fontsize=7, loc="lower right")
        print(p, {c: [ov(R_[(p, c, s)]) for s in ("s0", "s1")] for c in CELLS},
              "switch", f"{t[act[0]]:.1f}" if len(act) else "never", f"k_end {k[-1]:+.3f}")
    axs[1, 0].legend(*axs[0, 0].get_legend_handles_labels(), fontsize=7, loc="lower left")
    if a.zoom_on == "LTNq":
        fig.suptitle(f"Job 243 (Gazebo, full stack, seed {a.seed}): where TVLQG still touches a wall",
                     fontsize=11)
    else:
        fig.suptitle("Job 243 (Gazebo, full stack): TVLQG estimates the slide k during the drive and "
                     "uses it; no constant typed in", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, a.out), dpi=110)
    limitations(R_)


def limitations(R_):
    """Every TVLQG run that still touched a wall: first contact vs the switch."""
    print("\nTVLQG contact runs: plan seed | overlap plain/typed/TVLQG | first contact, switch [s]")
    n_before = n_total = 0
    for (p, c, s), r in sorted(R_.items()):
        if c != "LTNq" or not ov(r):
            continue
        z = np.load(track_file(r))
        tr, d, dt = z["track"], z["diag"], z["diag_t"]
        ih = np.nonzero(target.contact_mask(tr))[0]
        t_hit = tr[ih[0], 0] - tr[0, 0] if len(ih) else np.nan
        act = np.nonzero(np.isfinite(d[:, I_KB]) & (d[:, I_KB] != 0.0))[0]
        t_sw = dt[act[0]] - tr[0, 0] if len(act) else np.inf
        n_total += 1
        n_before += bool(t_hit < t_sw)
        print(f"  {p} {s} | {ov(R_.get((p, 'LTN', s))):.1f}/{ov(R_.get((p, 'LTNg', s))):.1f}/{ov(r):.1f} "
              f"| {t_hit:5.1f}, {t_sw:5.1f} {'BEFORE switch' if t_hit < t_sw else ''}")
    print(f"  {n_before} of {n_total} contact runs hit before the switch")


if __name__ == "__main__":
    main()
