#!/usr/bin/env python3
"""Job 160 departure figures (#34). PRELIMINARY.

Data: run_data/job160/series/<cfg>/*.npz (tools/departure_series.py run on the
VM over ~/compare_factors/<cfg>/s*/track_*.npz) and run_data/job160/rows/.
Config E (open loop) has no TVLQR diagnostics, so it is absent here.

"Departure" = first tick after goal with TRUE |e_cross| > 0.2 m against the
time-indexed live-plan reference (the same reference the corrector used).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(WS, "tools"))
from compare_run import (_signed_wall_dist, FP_HALF_LENGTH, FP_HALF_WIDTH,  # noqa: E402
                         FP_SPACING)

DATA = os.path.join(WS, "run_data", "job160")
CFG = {"A": "amcl, solid", "B": "truth, solid", "C": "amcl, phantom", "D": "truth, phantom"}
COL = {"A": "tab:red", "B": "tab:orange", "C": "tab:blue", "D": "tab:green"}
DEPART = 0.2
STAMP = "PRELIMINARY — job 160, not for publishing"


def body_clearance(x, y, yaw, floor):
    """Per-tick min signed clearance of the footprint perimeter [m]."""
    d, r, ox, oy = _signed_wall_dist(floor)
    nl = int(2 * FP_HALF_LENGTH / FP_SPACING) + 1
    nw = int(2 * FP_HALF_WIDTH / FP_SPACING) + 1
    lx = np.linspace(-FP_HALF_LENGTH, FP_HALF_LENGTH, nl)
    wy = np.linspace(-FP_HALF_WIDTH, FP_HALF_WIDTH, nw)
    b = np.vstack([np.c_[lx, np.full(nl, FP_HALF_WIDTH)], np.c_[lx, np.full(nl, -FP_HALF_WIDTH)],
                   np.c_[np.full(nw, FP_HALF_LENGTH), wy], np.c_[np.full(nw, -FP_HALF_LENGTH), wy]])
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    qx = x[:, None] + b[None, :, 0] * c - b[None, :, 1] * s
    qy = y[:, None] + b[None, :, 0] * s + b[None, :, 1] * c
    col = np.clip(((qx - ox) / r).astype(int), 0, d.shape[1] - 1)
    row = np.clip(((qy - oy) / r).astype(int), 0, d.shape[0] - 1)
    return d[row, col].min(axis=1)


def load():
    runs = {c: [] for c in CFG}
    for c in CFG:
        rows = {}
        for f in glob.glob(os.path.join(DATA, "rows", c, "rows.s*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r.get("track_path"):
                    rows[os.path.basename(r["track_path"])] = r
        for f in sorted(glob.glob(os.path.join(DATA, "series", c, "*.npz"))):
            r = rows.get(os.path.basename(f))
            if r is None:
                continue
            d = dict(np.load(f))
            tr = d["truth"]
            d["clear"] = body_clearance(tr[:, 0], tr[:, 1], tr[:, 2], r["floor"])
            d["ect"], d["ecb"] = np.abs(d["e_cross_true"]), np.abs(d["e_cross_believed"])
            out = np.nonzero((d["t"] >= 0) & (d["ect"] > DEPART))[0]
            d["dep"] = int(out[0]) if len(out) else None
            hit = np.nonzero(d["clear"] < 0)[0]
            d["hit"] = int(hit[0]) if len(hit) else None
            d["row"], d["name"] = r, os.path.basename(f)
            d["miss"] = r["final_err"] > 0.5
            runs[c].append(d)
    return runs


def fig_event_aligned(runs):
    """Median / IQR of each quantity, aligned at the departure tick."""
    lag = np.arange(-100, 101)  # ticks of 0.1 s
    qty = [("loc_err", "amcl pose error [m]"), ("ect", "TRUE |e_cross| [m]"),
           ("ecb", "BELIEVED |e_cross| [m]"), ("clear", "footprint wall clearance [m]")]
    fig, ax = plt.subplots(len(qty), 1, figsize=(9, 11), sharex=True)
    for c in CFG:
        deps = [d for d in runs[c] if d["dep"] is not None]
        for a, (k, lab) in zip(ax, qty):
            M = np.full((len(deps), len(lag)), np.nan)
            for i, d in enumerate(deps):
                j = d["dep"] + lag
                ok = (j >= 0) & (j < len(d[k]))
                M[i, ok] = d[k][j[ok]]
            # at least 5 runs contribute
            enough = np.sum(np.isfinite(M), 0) >= 5
            med = np.where(enough, np.nanmedian(M, 0), np.nan)
            lo, hi = (np.where(enough, np.nanpercentile(M, q, 0), np.nan) for q in (25, 75))
            a.plot(lag / 10, med, color=COL[c], label=f"{c}: {CFG[c]} (n={len(deps)})")
            a.fill_between(lag / 10, lo, hi, color=COL[c], alpha=0.15)
            a.set_ylabel(lab)
    for a in ax:
        a.axvline(0, color="k", lw=0.8, ls="--")
        a.grid(alpha=0.3)
    ax[3].axhline(0, color="k", lw=0.8)
    ax[1].axhline(DEPART, color="grey", lw=0.8, ls=":")
    ax[0].legend(fontsize=8)
    ax[-1].set_xlabel(f"time from departure (first true |e_cross| > {DEPART} m) [s]")
    fig.suptitle(f"Departures aligned, median + IQR\n{STAMP}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "event_aligned.png"), dpi=130)


def excursion_onset(d):
    """Upward DEPART crossing that starts the excursion that matters.

    Miss: the LAST upward crossing (the excursion it never came back from).
    Arrival: the crossing before its largest true |e_cross| (the worst scare
    it did recover from). None if the run never crosses.
    """
    e, t = d["ect"], d["t"]
    up = np.nonzero((e[1:] > DEPART) & (e[:-1] <= DEPART) & (t[1:] >= 0))[0] + 1
    if not len(up):
        return None
    if d["miss"]:
        return int(up[-1])
    peak = int(np.argmax(np.where(t >= 0, e, -1)))
    before = up[up <= peak]
    return int(before[-1]) if len(before) else int(up[0])


def fig_failure_vs_recovery(runs):
    """Miss vs arrival, aligned at the excursion onset (median + IQR)."""
    lag = np.arange(-60, 101)
    qty = [("loc_err", "pose error [m]"), ("ect", "TRUE |e_cross| [m]"),
           ("ecb", "BELIEVED |e_cross| [m]"), ("clear", "wall clearance [m]")]
    fig, ax = plt.subplots(len(qty), 4, figsize=(17, 11), sharex=True, sharey="row")
    for j, c in enumerate(CFG):
        for grp, ls in ((True, "-"), (False, "--")):
            sel = [(d, excursion_onset(d)) for d in runs[c] if d["miss"] == grp]
            sel = [(d, k) for d, k in sel if k is not None]
            for i, (key, lab) in enumerate(qty):
                M = np.full((len(sel), len(lag)), np.nan)
                for r, (d, k) in enumerate(sel):
                    jj = k + lag
                    ok = (jj >= 0) & (jj < len(d[key]))
                    M[r, ok] = d[key][jj[ok]]
                enough = np.sum(np.isfinite(M), 0) >= 4
                if not enough.any():
                    continue
                med = np.where(enough, np.nanmedian(M, 0), np.nan)
                lo, hi = (np.where(enough, np.nanpercentile(M, q, 0), np.nan) for q in (25, 75))
                col = "tab:red" if grp else "tab:grey"
                a = ax[i, j]
                a.plot(lag / 10, med, color=col, ls=ls,
                       label=f"{'miss' if grp else 'arrived'} (n={len(sel)})")
                a.fill_between(lag / 10, lo, hi, color=col, alpha=0.15)
                a.set_ylabel(lab) if j == 0 else None
        ax[0, j].set_title(f"{c}: {CFG[c]}")
        ax[0, j].legend(fontsize=8)
        for i in range(len(qty)):
            ax[i, j].axvline(0, color="k", lw=0.8, ls=":")
            ax[i, j].grid(alpha=0.3)
        ax[3, j].axhline(0, color="k", lw=0.8)
        ax[-1, j].set_xlabel("s from excursion onset")
    fig.suptitle("Miss: final excursion. Arrived: worst excursion it recovered from. "
                 f"Median + IQR. {STAMP}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "failure_vs_recovery.png"), dpi=120)


def fig_believed_vs_true(runs):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), sharex=True, sharey=True)
    for a, c in zip(ax, CFG):
        for d in runs[c]:
            a.scatter(d["ect"].max(), d["ecb"].max(), s=14,
                      color="tab:red" if d["miss"] else "tab:grey",
                      marker="x" if d["miss"] else "o")
        a.plot([0, 4], [0, 4], "k", lw=0.7)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(0.03, 6); a.set_ylim(0.03, 6)
        a.set_title(f"{c}: {CFG[c]}"); a.set_xlabel("max TRUE |e_cross| [m]")
        a.grid(alpha=0.3, which="both")
    ax[0].set_ylabel("max |e_cross| the corrector BELIEVED [m]")
    ax[0].scatter([], [], color="tab:red", marker="x", label="miss (>0.5 m)")
    ax[0].scatter([], [], color="tab:grey", label="arrived")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Below the diagonal = the corrector did not see the deviation. {STAMP}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "believed_vs_true.png"), dpi=130)


def fig_loc_err(runs):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for c in ("A", "C"):
        v = np.sort(np.concatenate([d["loc_err"] for d in runs[c]]))
        ax[0].plot(v, np.linspace(0, 1, len(v)), color=COL[c], label=f"{c}: {CFG[c]}")
        x = [d["loc_err"].max() for d in runs[c]]
        y = [d["row"]["final_err"] for d in runs[c]]
        ax[1].scatter(x, y, s=16, color=COL[c], label=f"{c}: {CFG[c]}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("per-tick amcl pose error [m]")
    ax[0].set_ylabel("CDF"); ax[0].grid(alpha=0.3, which="both"); ax[0].legend()
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].axhline(0.5, color="grey", ls=":", lw=0.8)
    ax[1].set_xlabel("max amcl pose error over the run [m]"); ax[1].set_ylabel("final_err (truth) [m]")
    ax[1].grid(alpha=0.3, which="both"); ax[1].legend()
    fig.suptitle(f"amcl error: with and without wall contact. {STAMP}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "loc_err.png"), dpi=130)


def casebook(d, c, path):
    """Map overlay + time series for one run."""
    floor = d["row"]["floor"]
    dist, r, ox, oy = _signed_wall_dist(floor)
    tr, be, P = d["truth"], d["believed"], d["live_plan"]
    pts = np.vstack([tr[:, :2], be[:, :2], P[:, :2]])
    lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
    fig = plt.figure(figsize=(14, 6.5))
    a0 = fig.add_subplot(1, 2, 1)
    a0.imshow(dist < 0, origin="lower", cmap="Greys", alpha=0.8,
              extent=(ox, ox + dist.shape[1] * r, oy, oy + dist.shape[0] * r))
    a0.plot(P[:, 0], P[:, 1], "k--", lw=1, label="live plan")
    a0.plot(tr[:, 0], tr[:, 1], color="tab:green", lw=1.5, label="truth")
    a0.plot(be[:, 0], be[:, 1], color="tab:red", lw=1.2, label="believed (amcl)" if c in "AC" else "believed")
    for i, m, lab in ((d["dep"], "*", "departure"), (d["hit"], "X", "first wall contact")):
        if i is not None:
            a0.plot(tr[i, 0], tr[i, 1], m, ms=13, mec="k", label=f"{lab} t={d['t'][i]:.1f}s")
    a0.plot(*d["goal_xy"], "o", mfc="none", mec="k", ms=10, label="goal")
    a0.set_xlim(lo[0], hi[0]); a0.set_ylim(lo[1], hi[1]); a0.set_aspect("equal")
    a0.legend(fontsize=7, loc="best")
    a0.set_title(f"{c} ({CFG[c]}) {d['name'][9:-4]}  final_err={d['row']['final_err']:.2f} m", fontsize=9)
    t = d["t"]
    ax = [fig.add_subplot(3, 2, k) for k in (2, 4, 6)]
    ax[0].plot(t, d["ect"], color="tab:green", label="true |e_cross|")
    ax[0].plot(t, d["ecb"], color="tab:red", label="believed |e_cross|")
    ax[0].axhline(DEPART, color="grey", ls=":", lw=0.8)
    ax[1].plot(t, d["loc_err"], color="k", label="pose error |xy|")
    ax[1].plot(t, np.abs(d["loc_err_yaw"]), color="tab:purple", lw=0.8, label="|yaw err| [rad]")
    ax[2].plot(t, d["clear"], color="tab:brown", label="wall clearance")
    ax[2].axhline(0, color="k", lw=0.8)
    for a in ax:
        if d["sat"].any():
            a.fill_between(t, 0, 1, where=d["sat"], transform=a.get_xaxis_transform(),
                           color="orange", alpha=0.25, lw=0)
        for i, ls in ((d["dep"], "--"), (d["hit"], "-.")):
            if i is not None:
                a.axvline(t[i], color="k", ls=ls, lw=0.8)
        a.legend(fontsize=7, loc="upper left"); a.grid(alpha=0.3)
    ax[2].set_xlabel("t since goal [s]  (orange = TVLQR saturated; -- departure, -. contact)")
    fig.suptitle(STAMP, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    runs = load()
    fig_event_aligned(runs)
    fig_failure_vs_recovery(runs)
    fig_believed_vs_true(runs)
    fig_loc_err(runs)
    # Casebook: the 3 worst misses in A, and the same plans under B, C, D.
    worst = sorted((d for d in runs["A"] if d["miss"]), key=lambda d: -d["row"]["final_err"])[:3]
    for w in worst:
        for c in CFG:
            m = [d for d in runs[c] if d["name"].split("_", 3)[3] == w["name"].split("_", 3)[3]
                 and d["row"]["seed"] == w["row"]["seed"]]
            if m:
                casebook(m[0], c, os.path.join(HERE, f"case_{w['name'][-9:-4]}_s{w['row']['seed']}_{c}.png"))
    # Stats printed for README
    for c in CFG:
        deps = [d for d in runs[c] if d["dep"] is not None]
        before = [d for d in deps if d["hit"] is not None and d["hit"] <= d["dep"]]
        loc = [d["loc_err"][d["dep"]] for d in deps]
        print(f"{c}: runs {len(runs[c])}, departed {len(deps)}, contact at/before departure {len(before)}, "
              f"loc_err at departure median {np.median(loc):.3f}, "
              f"believed/true max e_cross median ratio "
              f"{np.median([d['ecb'].max() / d['ect'].max() for d in runs[c]]):.2f}")


if __name__ == "__main__":
    main()
