#!/usr/bin/env python3
"""Job 221 (= job 220 part 2) figures (#10, #33 Type A): six controllers on the
SAME stored library plan, phantom walls, under truth (`*T`) and amcl (`*A`).

Data: run_data/2026-10-06_job221_controllers/ (all_rows.jsonl, or the per-config
rows.s*.jsonl while the job runs). Scoring as tools/summarize_arms.py; the
reference arm is ours-lib (L) within each localization.

    .venv/bin/python figures/2026-10-06/render_221.py
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(WS, "run_data", "2026-10-06_job221_controllers")
ARMS = {"L": "поправка ū", "M": "MPPI", "R": "RPP", "G": "Graceful",
        "V": "Vector Pursuit", "K": "GMPC"}
COL = {"L": "tab:blue", "M": "tab:orange", "R": "tab:red", "G": "tab:purple",
       "V": "tab:brown", "K": "tab:green"}
LOC = {"T": "true pose", "A": "amcl"}
LOC_RU = {"T": "истинная поза", "A": "amcl"}
# Graceful and GMPC are left out of the figure: their integration is broken
# (95-98% miss), see the paper. They stay in the summary.
PAPER_ARMS = ("L", "M", "R", "V")
SCORED = ("arrived", "failed", "timeout")


def load():
    by = {}
    for f in glob.glob(os.path.join(DATA, "*", "rows.s*.jsonl")):
        cfg = f.split(os.sep)[-2]
        for line in open(f):
            r = json.loads(line)
            v = r.get("final_err")
            r["scored"] = (r["outcome"] in SCORED and v is not None and math.isfinite(v))
            by.setdefault(cfg, {})[(r["plan"], r["seed"])] = r
    return by


def summary(by):
    lines = []
    for loc in LOC:
        base = by.get("L" + loc, {})
        for a in ARMS:
            d = by.get(a + loc, {})
            v = [r["final_err"] for r in d.values() if r["scored"]]
            if not v:
                continue
            miss = sum(x > 0.5 for x in v)
            w = l = 0
            for k, r in d.items():
                b = base.get(k)
                if a != "L" and b and r["scored"] and b["scored"]:
                    w += b["final_err"] < r["final_err"]
                    l += b["final_err"] > r["final_err"]
            p = binomtest(w, w + l).pvalue if w + l else float("nan")
            tt = [r["travel_time"] for r in d.values() if r["scored"] and r.get("travel_time")]
            en = [r["control_energy"] for r in d.values() if r["scored"] and r.get("control_energy")]
            wall = sum(1 for r in d.values() if r["scored"] and (r.get("wall_episodes") or 0) > 0)
            lines.append(f"{a+loc:3s} {ARMS[a]:15s} n={len(v):3d} miss={miss}/{len(v)} "
                         f"({miss/len(v):.0%}) med_fe={np.median(v):.3f} "
                         f"ours_closer={w}/{w+l} p={p:.2g} med_t={np.median(tt) if tt else float('nan'):.1f} "
                         f"med_E={np.median(en) if en else float('nan'):.0f} wall={wall}")
    return "\n".join(lines)


def fig_final_err(by):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a_, loc in zip(ax, LOC):
        data, labels, cols = [], [], []
        for a in ARMS:
            d = by.get(a + loc, {})
            v = [r["final_err"] for r in d.values() if r["scored"]]
            if v and a in PAPER_ARMS:
                data.append(np.clip(v, 0.01, 30)), labels.append(ARMS[a]), cols.append(COL[a])
        if not data:
            continue
        a_.boxplot(data, showfliers=False)
        for i, (v, c) in enumerate(zip(data, cols), 1):
            a_.plot(np.random.default_rng(i).normal(i, 0.07, len(v)), v, ".", color=c, alpha=0.5)
            m = sum(x > 0.5 for x in v)
            a_.text(i, 0.012, f"{m}/{len(v)}", ha="center", fontsize=7)
        a_.axhline(0.5, color="grey", ls=":", lw=0.8)
        a_.set_yscale("log")
        a_.set_xticks(range(1, len(labels) + 1), labels, rotation=20)
        a_.set_title(f"{LOC_RU[loc]}")
    ax[0].set_ylabel("терминальная ошибка, м")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "job221_final_err.png"), dpi=130)
    plt.close(fig)


def main():
    by = load()
    s = summary(by)
    print(s)
    open(os.path.join(HERE, "job221_summary.txt"), "w").write(s + "\n")
    fig_final_err(by)


if __name__ == "__main__":
    main()
