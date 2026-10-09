#!/usr/bin/env python3
"""Scenario (b)'s threshold, derived before the drive: TVLQG's predicted
cross-track error +- 3.29 sigma (tvlqg.deviation_bound_along_plan) along
job 243 TVLQG runs, against the error the controller actually saw, and the
supervisor's bench-chosen 0.5 m.

    .venv/bin/python figures/2026-10-09/tvlqg_deviation_bound/render.py
Input: run_data/2026-10-09_job243_tvlqg-fullstack.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning"),
                os.path.join(R, "figures/2026-10-09/tvlqg_target_tracks")]
import render as target  # noqa: E402  (contact_mask)
from agx_planning.runtime_corrector import tvlqg, tvlqr  # noqa: E402

J = os.path.join(R, "run_data/2026-10-09_job243_tvlqg-fullstack")
DT, Z = 0.1, 3.29
RUNS = [("floor_6_v2_00366", "s0", "clean run"),
        ("floor_6_v2_00047", "s1", "brief door-jamb graze, stays inside the bound"),
        ("floor_6_v2_00355", "s1", "door-jamb hit, knocked > 1 m off: fires 0.3 s after contact")]


def row(plan, seed):
    for line in open(os.path.join(J, "all_rows.jsonl")):
        r = json.loads(line)
        p = (r.get("track_path") or "").split("/")
        if r["plan"] == plan and len(p) >= 3 and p[-3] == "LTNq" and p[-2] == seed:
            return r, os.path.join(J, *p[-3:])
    raise KeyError((plan, seed))


def main():
    cfg, tc = tvlqg.TVLQGConfig(), tvlqr.TVLQRConfig()
    fig, axs = plt.subplots(len(RUNS), 1, figsize=(10, 3.0 * len(RUNS)), sharex=False)
    for ax, (plan, seed, what) in zip(axs, RUNS):
        r, f = row(plan, seed)
        z = np.load(f)
        d, dt_ = z["diag"], z["diag_t"]
        ok = np.isfinite(d[:, :6]).all(axis=1) & (d[:, 15] > 0)
        d, dt_ = d[ok], dt_[ok]
        v, w = d[:, 4], d[:, 5]
        b = tvlqg.deviation_bound_along_plan(z["plan_poses"][ok], v, w, DT, cfg, tc, Z)
        tr = z["track"]
        t = dt_ - tr[0, 0]
        hit = target.contact_mask(tr)
        moving = np.abs(v) > 0.2
        ax.fill_between(t, -b, b, color="tab:orange", alpha=0.2, lw=0,
                        label=f"TVLQG model: ±{Z}σ cross-track (from the plan, before the drive)")
        ax.axhline(0.5, c="k", ls=":", lw=1, label="supervisor's bench-chosen 0.5 m")
        ax.axhline(-0.5, c="k", ls=":", lw=1)
        ax.plot(t, d[:, 1], c="tab:orange", lw=1.2, label="e_cross the controller saw")
        out = (np.abs(d[:, 1]) > b) & moving
        if out.any():
            ax.plot(t[out], d[out, 1], ".", c="tab:red", ms=5, label="outside → re-join (scenario b)")
        if hit.any():
            th = tr[hit, 0] - tr[0, 0]
            ax.plot(th, np.full(len(th), -0.9), "|", c="tab:red", ms=8, label="body on a wall")
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("e_cross [m]")
        ax.set_title(f"{plan} {seed} (TVLQG, job 243): {what}; wall overlap "
                     f"{r.get('wall_overlap_s') or 0:.1f} s", fontsize=9)
        print(f"{plan} {seed}: bound median {np.median(b[moving]):.3f} m "
              f"(min {b[moving].min():.3f}, max {b[moving].max():.3f}); "
              f"ticks outside {out.sum()}; first outside "
              f"{t[np.argmax(out)] if out.any() else float('nan'):.1f} s; first contact "
              f"{(tr[np.argmax(hit), 0] - tr[0, 0]) if hit.any() else float('nan'):.1f} s")
    axs[-1].set_xlabel("t [s]")
    axs[0].legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("Scenario (b) threshold derived from TVLQG's covariance: ~0.5 m, the value "
                 "the bench had chosen; it fires after a wall knock, not before contact", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "tvlqg_deviation_bound.png"), dpi=110)


if __name__ == "__main__":
    main()
