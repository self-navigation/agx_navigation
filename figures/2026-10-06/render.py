#!/usr/bin/env python3
"""Job 220 part 1 figures (#10, #33): full stacks under amcl, solid walls.

Data: run_data/2026-10-06_job220_fullstack/ (all_rows.jsonl + <cfg>/s<seed>/
track_*.npz), fetched from the VM. Configs: O = ours (FM2 + PMP + TVLQR),
M2 = Smac2D + MPPI, MH = Hybrid-A* + MPPI, R2 = Smac2D + RPP; MPPI/RPP use
the compare_skid profile. Scoring as tools/summarize_arms.py: arrived/failed/
timeout rows with a finite final_err; planner-/stack-failed excluded.

    .venv/bin/python figures/2026-10-06/render.py
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(WS, "run_data", "2026-10-06_job220_fullstack")
MAP = os.path.join(WS, "src", "rudn-ordjo-building", "maps", "floor_6.png")
MAP_RES, MAP_ORIGIN = 0.05, (-30.0, -30.0)
CFG = {"O": "ours (FM$^2$+PMP+TVLQR)", "M2": "Smac2D + MPPI",
       "MH": "Hybrid-A* + MPPI", "R2": "Smac2D + RPP"}
COL = {"O": "tab:blue", "M2": "tab:orange", "MH": "tab:green", "R2": "tab:red"}
SCORED = ("arrived", "failed", "timeout")


def load():
    by = {c: {} for c in CFG}
    for line in open(os.path.join(DATA, "all_rows.jsonl")):
        r = json.loads(line)
        if not r.get("track_path"):
            continue
        c = r["track_path"].split("/")[-3]
        v = r.get("final_err")
        r["scored"] = (r["outcome"] in SCORED and v is not None and math.isfinite(v))
        r["local_track"] = os.path.join(DATA, *r["track_path"].split("/")[-3:])
        by[c][(r["plan"], r["seed"])] = r
    return by


def paired(by, c, key):
    o = by["O"]
    out = []
    for k, r in by[c].items():
        b = o.get(k)
        if b and r["scored"] and b["scored"] and r.get(key) is not None and b.get(key) is not None:
            out.append((b[key], r[key]))
    return np.array(out)


def fig_paired(by):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    for a, c in zip(ax, ("M2", "MH", "R2")):
        p = paired(by, c, "final_err")
        lo, hi = 0.02, 15
        a.loglog(np.clip(p[:, 0], lo, hi), np.clip(p[:, 1], lo, hi), "o", ms=4,
                 color=COL[c], alpha=0.7)
        a.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        a.axvline(0.5, color="grey", ls=":", lw=0.8)
        a.axhline(0.5, color="grey", ls=":", lw=0.8)
        w = int((p[:, 0] < p[:, 1]).sum())
        n = int((p[:, 0] != p[:, 1]).sum())
        a.set_title(f"{CFG[c]}: ours closer in {w}/{n}")
        a.set_xlabel("ours final error [m]")
        a.set_ylabel(f"{CFG[c]} final error [m]")
        a.set_xlim(lo, hi), a.set_ylim(lo, hi)
    fig.suptitle("Job 220: paired final error per run (amcl, solid walls; "
                 "above the diagonal = ours ended closer)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "job220_final_err_paired.png"), dpi=130)
    plt.close(fig)


def fig_ratios(by):
    keys = [("travel_time", "travel time"), ("control_energy", "control energy"),
            ("path_length", "path length")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (k, lab) in zip(ax, keys):
        data = [paired(by, c, k) for c in ("M2", "MH", "R2")]
        rat = [d[:, 1] / d[:, 0] for d in data]
        a.boxplot(rat, showfliers=False)
        for i, r in enumerate(rat, 1):
            a.plot(np.random.default_rng(i).normal(i, 0.06, len(r)), r, ".",
                   color=COL[("M2", "MH", "R2")[i - 1]], alpha=0.5)
        a.axhline(1, color="k", lw=0.8)
        a.set_yscale("log")
        a.text(0.02, 0.02, "points far below 1 on path length are Nav2 runs\n"
               "that aborted early; their low time/energy is not a win",
               transform=a.transAxes, fontsize=7, va="bottom")
        a.set_xticks([1, 2, 3], ["Smac2D\n+MPPI", "Hybrid-A*\n+MPPI", "Smac2D\n+RPP"])
        a.set_title(f"{lab}: Nav2 / ours (paired)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "job220_cost_ratios.png"), dpi=130)
    plt.close(fig)


def draw_map(a, xs, ys, pad=1.5):
    img = np.asarray(Image.open(MAP).convert("L"))
    h = img.shape[0]
    ext = [MAP_ORIGIN[0], MAP_ORIGIN[0] + img.shape[1] * MAP_RES,
           MAP_ORIGIN[1], MAP_ORIGIN[1] + h * MAP_RES]
    a.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255)
    a.set_xlim(min(xs) - pad, max(xs) + pad)
    a.set_ylim(min(ys) - pad, max(ys) + pad)
    a.set_aspect("equal")


def fig_tracks(by, cases):
    fig, ax = plt.subplots(1, len(cases), figsize=(5 * len(cases), 5))
    for a, (plan, seed) in zip(np.atleast_1d(ax), cases):
        xs, ys, lines, plan_poses = [], [], [], np.zeros((0, 3))
        for c in CFG:
            r = by[c].get((plan, seed))
            if not r or not os.path.exists(r["local_track"]):
                continue
            d = np.load(r["local_track"])
            t = d["track"]
            xs.append(t[:, 1]); ys.append(t[:, 2])
            fe = r.get("final_err")
            lines.append((c, t, f"{CFG[c]}: {r['outcome']}"
                          + (f", {fe:.2f} m" if fe is not None else "")))
            goal = d["goal_xy"]
            if c == "O":
                plan_poses = d["plan_poses"]
            start = t[0, 1:3]
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        draw_map(a, xs, ys)
        if len(plan_poses):
            a.plot(plan_poses[:, 0], plan_poses[:, 1], "k--", lw=1, label="ours: live PMP plan")
        for c, t, lab in lines:
            a.plot(t[:, 1], t[:, 2], color=COL[c], lw=1.4, label=lab)
        a.plot(*goal, "k*", ms=12)
        a.plot(*start, "ko", ms=6, mfc="none")
        a.set_title(f"{plan} seed {seed}")
        a.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "job220_tracks.png"), dpi=130)
    plt.close(fig)


def main():
    by = load()
    fig_paired(by)
    fig_ratios(by)
    # Cases: the largest ours-vs-MPPI disagreement each way, and one where
    # both arrive. Picked mechanically, printed so the README can name them.
    diffs = []
    for k, r in by["M2"].items():
        o = by["O"].get(k)
        if o and o["scored"] and r["scored"]:
            diffs.append((o["final_err"] - r["final_err"], k))
    diffs.sort()
    both = [k for d, k in diffs if by["O"][k]["final_err"] < 0.5 and by["M2"][k]["final_err"] < 0.5]
    cases = [diffs[0][1], diffs[-1][1]] + both[:1]
    print("track cases (ours better, MPPI better, both arrive):", cases)
    fig_tracks(by, cases)


if __name__ == "__main__":
    main()
