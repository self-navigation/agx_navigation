#!/usr/bin/env python3
"""Why wall contacts are ~75% under amcl and ~45% under ground truth (job 244).

The corrector never logs the pose it used, but it logs its tracking error
against the plan sample it was following, and playback is time-indexed; so the
pose the controller BELIEVED is rebuilt per tick (tools/departure_series.py)
and compared with Gazebo truth. Under localization:=truth the rebuild
reproduces truth to ~5 mm median / 2-5 cm p95 (the sampling floor), which is
the self-check.

    .venv/bin/python figures/2026-10-10/amcl_vs_truth/render.py          # stats.json + PNGs
Input: run_data/2026-10-09_job244_tvlqg-deviation (fetched 2026-10-10).
Cells: LTNq (TVLQG, truth), LAN (plain TVLQR, amcl), LANq (TVLQG, raw amcl
pose), LANqe (TVLQG, filter pose). LADq (re-join) is counted from its rows only:
a re-join re-splices the plan, which breaks the time-indexed rebuild.
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
sys.path[:0] = [os.path.join(R, "tools")]
import compare_run as cr  # noqa: E402
from departure_series import series  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(R, "run_data/2026-10-09_job244_tvlqg-deviation")
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
CELLS = ("LTNq", "LAN", "LANq", "LANqe")
COL = {"LTNq": "k", "LAN": "tab:green", "LANq": "tab:red", "LANqe": "tab:blue"}
LAB = {"LTNq": "TVLQG, truth pose", "LAN": "plain TVLQR, amcl",
       "LANq": "TVLQG, raw amcl pose", "LANqe": "TVLQG, filter pose (amcl)"}
# diag columns 16-21 (tvlqr.CorrectionDiagnostics.FIELDS)
K_HAT, SIGMA_K, K_BAR, RHO, NIS, GATED = 16, 17, 18, 19, 20, 21
PATCH_R = 1.5     # patches are 1-3 m squares; the row keeps only the centre


def rows():
    out = {}
    for line in open(os.path.join(J, "all_rows.jsonl")):
        r = json.loads(line)
        parts = (r.get("track_path") or "").split("/")
        if len(parts) < 3:
            continue
        out[(r["plan"], parts[-3], parts[-2])] = r
    return out


def local_track(r):
    parts = r["track_path"].split("/")
    return os.path.join(J, *parts[-3:])


def footprint_clear(tr):
    """Per sample: min signed wall distance over the body rectangle's perimeter
    (compare_run.wall_contact's body; < 0 = a body point on a wall pixel)."""
    d, res, ox, oy = cr._signed_wall_dist(6)
    nl = int(2 * cr.FP_HALF_LENGTH / cr.FP_SPACING) + 1
    nw = int(2 * cr.FP_HALF_WIDTH / cr.FP_SPACING) + 1
    lx = np.linspace(-cr.FP_HALF_LENGTH, cr.FP_HALF_LENGTH, nl)
    wy = np.linspace(-cr.FP_HALF_WIDTH, cr.FP_HALF_WIDTH, nw)
    body = np.vstack([np.c_[lx, np.full(nl, cr.FP_HALF_WIDTH)], np.c_[lx, np.full(nl, -cr.FP_HALF_WIDTH)],
                      np.c_[np.full(nw, cr.FP_HALF_LENGTH), wy], np.c_[np.full(nw, -cr.FP_HALF_LENGTH), wy]])
    c, s = np.cos(tr[:, 3])[:, None], np.sin(tr[:, 3])[:, None]
    qx = tr[:, 1:2] + body[None, :, 0] * c - body[None, :, 1] * s
    qy = tr[:, 2:3] + body[None, :, 0] * s + body[None, :, 1] * c
    col = np.clip(((qx - ox) / res).astype(int), 0, d.shape[1] - 1)
    row = np.clip(((qy - oy) / res).astype(int), 0, d.shape[0] - 1)
    return d[row, col].min(axis=1)


def contact_mask(tr):
    return footprint_clear(tr) < 0


def hull(x, y, th):
    L, W = cr.FP_HALF_LENGTH, cr.FP_HALF_WIDTH
    c, s = np.cos(th), np.sin(th)
    pts = [(L, W), (L, -W), (-L, -W), (-L, W)]
    return [(x + a * c - b * s, y + a * s + b * c) for a, b in pts]


def gain_lag(t, eb, et):
    """e_believed(t) ~ k e_true(t - tau), tau in [0, 3] s (tools/believed_gain.py)."""
    if len(t) < 50 or np.ptp(et) < 0.15:
        return None
    best = None
    for tau in np.arange(0, 3.01, 0.1):
        x = np.interp(t - tau, t, et)
        if x @ x == 0:
            continue
        k = (x @ eb) / (x @ x)
        r2 = 1 - np.sum((eb - k * x) ** 2) / max(np.sum((eb - eb.mean()) ** 2), 1e-12)
        if best is None or r2 > best[2]:
            best = (float(k), float(tau), float(r2))
    return best


def analyse(r, d_npz=None):
    """Per-run series + scalar stats for one track file."""
    p = local_track(r)
    s = series(p)
    if s is None:
        return None, None
    d = np.load(p, allow_pickle=True)
    D = d["diag"]
    t = s["t"]
    drive = t > 0
    T = s["track"]
    # truth speed at diag ticks, for the 'moving' mask
    vx = np.gradient(T[:, 1], T[:, 0]); vy = np.gradient(T[:, 2], T[:, 0])
    spd = np.interp(t + float(d["goal_t"]), T[:, 0], np.hypot(vx, vy))
    moving = drive & (spd > 0.05)
    yaw_rate = np.abs(np.gradient(np.unwrap(s["truth"][:, 2]), t))
    le = s["loc_err"]
    # believed-pose jumps: per-tick believed displacement minus truth displacement
    db = np.linalg.norm(np.diff(s["believed"][:, :2], axis=0), axis=1)
    dt_ = np.linalg.norm(np.diff(s["truth"][:, :2], axis=0), axis=1)
    jump = np.r_[0.0, np.abs(db - dt_)]
    # time at the first truth-footprint contact (rows' wall_first_t is on the
    # same track, absolute sim time)
    clear = footprint_clear(np.c_[t, s["truth"]])
    contact = clear < 0
    ic = int(np.argmax(contact & drive)) if (contact & drive).any() else -1
    # patch proximity (truth within PATCH_R of a patch centre)
    pc = np.array([[q["x"], q["y"]] for q in r.get("patches", []) if q.get("ok")] or np.zeros((0, 2)))
    on_patch = (np.min(np.linalg.norm(s["truth"][:, None, :2] - pc[None], axis=2), axis=1) < PATCH_R
                if len(pc) else np.zeros(len(t), bool))
    # plan clearance (the live plan itself, with its own heading)
    P = s["live_plan"]
    pclear = footprint_clear(np.c_[np.arange(len(P)), P])
    # k_hat
    kh, kb, gated = D[:, K_HAT], D[:, K_BAR], D[:, GATED]
    sw = np.flatnonzero(kb != 0)
    st = dict(
        plan=r["plan"], seed=r["seed"], outcome=r["outcome"], final_err=r["final_err"],
        wall_overlap_s=r["wall_overlap_s"], wall_first_t=r["wall_first_t"],
        n=int(drive.sum()),
        loc_med=float(np.median(le[drive])), loc_p95=float(np.percentile(le[drive], 95)),
        loc_max=float(le[drive].max()),
        loc_along_med=float(np.median(np.abs(s["loc_err_along"][drive]))),
        loc_cross_med=float(np.median(np.abs(s["loc_err_cross"][drive]))),
        loc_yaw_med=float(np.median(np.abs(s["loc_err_yaw"][drive]))),
        loc_yaw_p95=float(np.percentile(np.abs(s["loc_err_yaw"][drive]), 95)),
        loc_first3_med=float(np.median(le[drive & (t < 3)])) if (drive & (t < 3)).any() else None,
        loc_last_q_med=float(np.median(le[drive & (t > 0.75 * t[drive].max())])),
        loc_turn_med=float(np.median(le[moving & (yaw_rate > 0.3)])) if (moving & (yaw_rate > 0.3)).sum() > 5 else None,
        loc_straight_med=float(np.median(le[moving & (yaw_rate < 0.1)])) if (moving & (yaw_rate < 0.1)).sum() > 5 else None,
        loc_patch_med=float(np.median(le[moving & on_patch])) if (moving & on_patch).sum() > 5 else None,
        loc_nopatch_med=float(np.median(le[moving & ~on_patch])) if (moving & ~on_patch).sum() > 5 else None,
        frac_gt_020=float(np.mean(le[drive] > 0.20)), frac_gt_050=float(np.mean(le[drive] > 0.50)),
        jumps_5cm=int(np.sum(jump[drive] > 0.05)), jumps_10cm=int(np.sum(jump[drive] > 0.10)),
        jump_max=float(jump[drive].max()),
        max_ec_true=float(np.abs(s["e_cross_true"][drive]).max()),
        max_ec_bel=float(np.abs(s["e_cross_believed"][drive]).max()),
        gain_lag=gain_lag(t[drive], s["e_cross_believed"][drive], s["e_cross_true"][drive]),
        contact=bool(ic >= 0), t_contact=float(t[ic]) if ic >= 0 else None,
        plan_min_clear=float(pclear.min()),
        k_switch_t=float(t[sw[0]]) if len(sw) else None,
        k_hat_end=float(kh[-1]), k_hat_min=float(kh.min()), k_hat_max=float(kh.max()),
        k_bar_end=float(kb[-1]),
        k_bar_sd_after=float(np.std(kb[sw[0]:])) if len(sw) else None,
        gated_frac=float(np.mean(gated[drive])), nis_med=float(np.median(D[drive, NIS])),
        sat_frac=float(np.mean(s["sat"][drive])),
    )
    if ic >= 0:
        w = drive & (t >= t[ic] - 3) & (t <= t[ic])
        i0 = max(ic - 1, 0)
        # plan clearance near the contact point (within 1 m of the plan sample
        # the controller was following)
        near = np.linalg.norm(P[:, :2] - s["ref"][ic, :2], axis=1) < 1.0
        st.update(
            loc_at_contact=float(le[ic]), loc_cross_at_contact=float(s["loc_err_cross"][ic]),
            loc_yaw_at_contact=float(s["loc_err_yaw"][ic]),
            ec_true_at_contact=float(s["e_cross_true"][ic]), ec_bel_at_contact=float(s["e_cross_believed"][ic]),
            ea_true_at_contact=float(s["e_along_true"][ic]),
            jump_pre3_max=float(jump[w].max()) if w.any() else 0.0,
            plan_clear_near_contact=float(pclear[near].min()) if near.any() else None,
            clear_at_contact_believed=float(footprint_clear(np.c_[[0.0], s["believed"][i0:i0 + 1]])[0]),
            sat_pre3=float(np.mean(s["sat"][w])) if w.any() else 0.0,
            on_patch_at_contact=bool(on_patch[ic]),
            yaw_rate_at_contact=float(yaw_rate[ic]),
            t_loc_gt_020_before_contact=(float(t[ic] - t[np.flatnonzero(drive & (le > 0.2) & (t <= t[ic]))[0]])
                                         if (drive & (le > 0.2) & (t <= t[ic])).any() else None),
        )
    ser = dict(t=t, le=le, lec=s["loc_err_cross"], lea=s["loc_err_along"], ley=s["loc_err_yaw"],
               ecb=s["e_cross_believed"], ect=s["e_cross_true"], bel=s["believed"], tru=s["truth"],
               ref=s["ref"], P=P, track=T, goal=s["goal_xy"], kh=kh, kb=kb, sk=D[:, SIGMA_K],
               gated=gated, clear=clear, contact=contact, jump=jump, drive=drive, patches=pc,
               sat=s["sat"], dv=s["dv"], domega=s["domega"])
    return st, ser


def classify(a_amcl, a_truth):
    """Why did this plan/seed contact under amcl and not under truth."""
    why = []
    if a_amcl["loc_at_contact"] >= 0.10 and abs(a_amcl["ec_bel_at_contact"]) < abs(a_amcl["ec_true_at_contact"]) - 0.05:
        why.append("a")   # under-believed error: thought it was on the plan
    if a_amcl["jump_pre3_max"] >= 0.10:
        why.append("b")   # a believed-pose jump just before contact
    pc = a_amcl.get("plan_clear_near_contact")
    if pc is not None and pc < 0.12 and abs(a_amcl["ec_true_at_contact"]) < 0.20:
        why.append("c")   # the plan itself nearly touches; small error tips it
    if not why:
        why.append("d")
    return why


def sign_p(w, l):
    from math import comb
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    R_ = rows()
    stats = {c: {} for c in CELLS}
    for (plan, cell, seed), r in sorted(R_.items()):
        if cell not in CELLS:
            continue
        try:
            st, _ = analyse(r)
        except Exception as e:  # noqa: BLE001
            print("skip", plan, cell, seed, e); continue
        if st is None:
            print("skip (no diag/plan)", plan, cell, seed); continue
        stats[cell][f"{plan}/{seed}"] = st
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    # headline numbers
    for c in CELLS:
        S = list(stats[c].values())
        print(f"{c}: n={len(S)} contact={sum(s['contact'] for s in S)} "
              f"loc_med={np.median([s['loc_med'] for s in S]):.3f} "
              f"p95(med over runs)={np.median([s['loc_p95'] for s in S]):.3f} "
              f"max(med)={np.median([s['loc_max'] for s in S]):.3f} "
              f"along={np.median([s['loc_along_med'] for s in S]):.3f} cross={np.median([s['loc_cross_med'] for s in S]):.3f} "
              f"yaw={np.degrees(np.median([s['loc_yaw_med'] for s in S])):.1f}deg "
              f"jumps>5cm/run={np.median([s['jumps_5cm'] for s in S]):.0f} >10cm={np.median([s['jumps_10cm'] for s in S]):.0f} "
              f"frac>0.2={np.mean([s['frac_gt_020'] for s in S]):.2f} frac>0.5={np.mean([s['frac_gt_050'] for s in S]):.2f}")
    # ---- figures ----
    fig_loc_err(stats)
    fig_examples(R_, stats)
    fig_khat(R_, stats)
    print("done")


def fig_loc_err(stats):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for c in CELLS:
        S = list(stats[c].values())
        ax[0].hist([s["loc_med"] for s in S], bins=np.linspace(0, 0.6, 31), histtype="step",
                   color=COL[c], label=LAB[c], lw=1.5)
        ax[1].hist([s["loc_max"] for s in S], bins=np.linspace(0, 2.5, 26), histtype="step",
                   color=COL[c], lw=1.5)
        gl = [s["gain_lag"] for s in S if s["gain_lag"]]
        ax[2].scatter([g[1] for g in gl], [g[0] for g in gl], s=10, color=COL[c], alpha=0.6)
    ax[0].set_xlabel("per-run MEDIAN |believed - true| [m]"); ax[0].set_ylabel("runs"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("per-run MAX |believed - true| [m]")
    ax[2].set_xlabel("lag tau [s] of believed e_cross vs true"); ax[2].set_ylabel("gain k")
    ax[2].set_ylim(-0.5, 1.5); ax[2].axhline(1, color="grey", lw=0.5)
    fig.suptitle("job 244: localization error of the pose the corrector used, 102 runs per cell (self-check: truth cell)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "loc_err_dist.png"), dpi=130); plt.close(fig)


def _map_axes(ax, xs, ys, pad=1.2):
    img = np.array(Image.open(MAP + ".png"))
    meta = yaml.safe_load(open(MAP + ".yaml"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    h = img.shape[0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=[ox, ox + img.shape[1] * res, oy, oy + h * res],
              origin="lower" if meta.get("negate", 0) else "upper", interpolation="nearest")
    ax.set_xlim(min(xs) - pad, max(xs) + pad); ax.set_ylim(min(ys) - pad, max(ys) + pad); ax.set_aspect("equal")


def fig_examples(R_, stats, picks=None):
    """Truth track (solid), believed track (dashed), plan (grey) for the plan/seed
    pairs that contact under LANq and not under LTNq; the four with the largest
    localization error at contact, plus the LTNq twin."""
    pairs = [k for k, s in stats["LANq"].items() if s["contact"] and k in stats["LTNq"]
             and not stats["LTNq"][k]["contact"]]
    pairs.sort(key=lambda k: -stats["LANq"][k]["loc_at_contact"])
    picks = picks or pairs[:4]
    fig, axes = plt.subplots(2, len(picks), figsize=(4.2 * len(picks), 8.5))
    for j, key in enumerate(picks):
        plan, seed = key.split("/")
        for i, cell in enumerate(("LTNq", "LANq")):
            r = R_[(plan, cell, seed)]
            st, S = analyse(r)
            ax = axes[i, j]
            m = S["drive"]
            _map_axes(ax, S["tru"][m, 0], S["tru"][m, 1])
            ax.plot(S["P"][:, 0], S["P"][:, 1], color="0.6", lw=1, label="plan")
            ax.plot(S["tru"][m, 0], S["tru"][m, 1], color=COL[cell], lw=1.5, label="truth")
            ax.plot(S["bel"][m, 0], S["bel"][m, 1], color="tab:orange", lw=1, ls="--", label="believed")
            cm = S["contact"] & m
            ax.scatter(S["tru"][cm, 0], S["tru"][cm, 1], s=6, color="red", zorder=5)
            for q in S["patches"]:
                ax.add_patch(plt.Circle(q, 0.5, fill=False, color="tab:purple", lw=0.8))
            if st["contact"]:
                ic = int(np.argmax(cm))
                ax.add_patch(Polygon(hull(*S["tru"][ic]), fill=False, color="red", lw=1))
                ax.add_patch(Polygon(hull(*S["bel"][ic]), fill=False, color="tab:orange", lw=1, ls="--"))
                ttl = (f"{cell} {plan[-5:]} {seed}: contact t={st['t_contact']:.1f}s, loc err {st['loc_at_contact']:.2f} m\n"
                       f"e_cross true {st['ec_true_at_contact']:+.2f} believed {st['ec_bel_at_contact']:+.2f}; "
                       f"jump(3s) {st['jump_pre3_max']:.2f}")
            else:
                ttl = f"{cell} {plan[-5:]} {seed}: no contact, max loc err {st['loc_max']:.2f} m"
            ax.set_title(ttl, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("contact under amcl (LANq, bottom) and not under truth (LTNq, top); red = footprint on a wall pixel; "
                 "orange dashed = pose the corrector believed; purple = slip-patch centres", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "contact_examples.png"), dpi=130); plt.close(fig)
    # time series for the same picks
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.6 * len(picks)), sharex=False)
    for j, key in enumerate(picks):
        plan, seed = key.split("/")
        r = R_[(plan, "LANq", seed)]
        st, S = analyse(r)
        ax = axes[j]; m = S["drive"]
        ax.plot(S["t"][m], S["ect"][m], color="k", label="true e_cross")
        ax.plot(S["t"][m], S["ecb"][m], color="tab:orange", label="believed e_cross")
        ax.plot(S["t"][m], S["le"][m], color="tab:red", lw=0.8, label="|believed - true|")
        ax.fill_between(S["t"][m], -1.5, 1.5, where=S["contact"][m], color="red", alpha=0.12)
        ax.fill_between(S["t"][m], -1.5, 1.5, where=S["jump"][m] > 0.05, color="tab:blue", alpha=0.25)
        ax.set_ylim(-1.2, 1.2); ax.set_ylabel("[m]"); ax.set_title(f"LANq {plan[-5:]} {seed}", fontsize=8)
        if j == 0:
            ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("t since goal [s]   (red band = footprint on wall; blue = believed-pose jump > 5 cm)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "contact_examples_series.png"), dpi=130); plt.close(fig)


def fig_khat(R_, stats):
    """k_hat and k_bar over time, LTNq vs LANq vs LANqe, on 4 plans (seed 0)."""
    plans = sorted({k.split("/")[0] for k in stats["LANq"]})[:4]
    fig, axes = plt.subplots(2, len(plans), figsize=(4 * len(plans), 6), sharey="row")
    for j, plan in enumerate(plans):
        for cell in ("LTNq", "LANq", "LANqe"):
            r = R_.get((plan, cell, "s0"))
            if r is None:
                continue
            st, S = analyse(r)
            if S is None:
                continue
            m = S["drive"]
            axes[0, j].plot(S["t"][m], S["kh"][m], color=COL[cell], lw=1, label=LAB[cell])
            axes[0, j].fill_between(S["t"][m], S["kh"][m] - S["sk"][m], S["kh"][m] + S["sk"][m], color=COL[cell], alpha=0.12)
            axes[1, j].plot(S["t"][m], S["kb"][m], color=COL[cell], lw=1)
        axes[0, j].axhline(-0.33, color="grey", lw=0.5, ls=":"); axes[1, j].axhline(-0.33, color="grey", lw=0.5, ls=":")
        axes[0, j].set_title(plan[-5:], fontsize=9); axes[1, j].set_xlabel("t [s]")
    axes[0, 0].set_ylabel("k_hat +- sigma_k [s]"); axes[1, 0].set_ylabel("k_bar used by the LQR [s]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("TVLQG surface estimate under truth vs amcl (dotted = Gazebo's k = -0.33)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "khat_truth_vs_amcl.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
