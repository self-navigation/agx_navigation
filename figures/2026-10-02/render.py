"""Nav2 comparison v7 (#10, #34), 2026-10-02 -- PRELIMINARY, stamped as such.

    .venv/bin/python figures/2026-10-02/render.py

Runs tools/summarize_compare.py on run_data/compare_v7_seed{0,1}/, stamps a
banner on every figure it keeps, and adds failures.png: why each arm's runs
end where they do (wall contact, stalls) and how close the reference plans
themselves pass to the walls. See README.
"""
import collections
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import summarize_compare as sc  # noqa: E402

PLANS = ["floor_6_v2_00105", "floor_6_v2_00249", "floor_6_v2_00419", "floor_6_v2_00428"]
BANNER = "PRELIMINARY - NOT FOR PUBLICATION (2026-10-02): failure causes under investigation (#34)"
ARMS = ["ours", "nav2-mppi", "nav2-rpp"]
DRIVEN = ("arrived", "failed", "timeout")


def stamp(src, dst):
    img = plt.imread(src)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100 + 0.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, h / (h + 40)])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.5, 1 - 20 / (h + 40), BANNER, ha="center", va="center",
             color="white", fontsize=min(11, w / 75), weight="bold",
             bbox=dict(facecolor="red", edgecolor="none", pad=4))
    fig.savefig(dst)
    plt.close(fig)


def classify():
    """Per arm: outcome x {wall contact, stalled, other}; plus plan clearances."""
    _, edt = sc.load_map(6)
    cause = {a: collections.Counter() for a in ARMS}
    plan_clear = {}
    for seed in (0, 1):
        d = os.path.join(ROOT, "run_data", f"compare_v7_seed{seed}")
        for r in sc.load_rows(d):
            if r["outcome"] not in DRIVEN:
                cause[r["arm"]][r["outcome"]] += 1  # planner-/stack-failed
                continue
            z = np.load(sc.local(r["track_path"], d))
            tr = z["track"]
            tr = tr[tr[:, 0] >= float(z["goal_t"])]
            pp = np.asarray(z["plan_poses"])
            if len(pp):
                plan_clear[r["plan"]] = sc.clearance(pp[:, :2], edt)
            if r["outcome"] == "arrived":
                cause[r["arm"]]["arrived"] += 1
            elif len(tr) and sc.clearance(tr[:, 1:3], edt) <= 0:
                cause[r["arm"]]["missed: touched wall"] += 1
            else:
                last = tr[tr[:, 0] >= tr[-1, 0] - 10] if len(tr) else tr
                still = len(last) and np.hypot(*np.ptp(last[:, 1:3], axis=0)) < 0.05
                cause[r["arm"]]["missed: stalled, no contact" if still
                                else "missed: other"] += 1
    return cause, plan_clear


def failures_png(dst):
    cause, plan_clear = classify()
    cats = ["arrived", "missed: touched wall", "missed: stalled, no contact",
            "missed: other", "planner-failed", "stack-failed"]
    colors = ["tab:green", "tab:red", "tab:orange", "tab:gray", "tab:purple", "black"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    left = np.zeros(len(ARMS))
    for c, col in zip(cats, colors):
        v = np.array([cause[a][c] for a in ARMS])
        a1.barh(ARMS, v, left=left, color=col, label=c)
        for i, x in enumerate(v):
            if x:
                a1.text(left[i] + x / 2, i, str(x), ha="center", va="center",
                        color="white", fontsize=9)
        left += v
    a1.invert_yaxis()
    a1.set_xlabel("runs (40 plans x 2 seeds)")
    a1.set_title("How each run ended (ground truth vs baked map)")
    a1.legend(fontsize=8, loc="lower right")
    pc = np.array(list(plan_clear.values()))
    a2.hist(pc * 100, bins=20, color="tab:blue")
    a2.axvline(0, color="red")
    a2.set_xlabel("reference plan's min clearance beyond footprint [cm]")
    a2.set_ylabel("plans")
    a2.set_title(f"PMP plans hug walls: median {np.median(pc) * 100:.1f} cm, "
                 f"{(pc <= 0).mean():.0%} overlap a wall")
    fig.tight_layout()
    tmp = dst + ".tmp.png"
    fig.savefig(tmp, dpi=100)
    plt.close(fig)
    stamp(tmp, dst)
    os.remove(tmp)


for seed in (0, 1):
    tmp = os.path.join(ROOT, "figures", "tmp", f"v7s{seed}")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "summarize_compare.py"),
                    os.path.join(ROOT, "run_data", f"compare_v7_seed{seed}"), "-o", tmp],
                   check=True, stdout=subprocess.DEVNULL)
    shutil.copy(os.path.join(tmp, "summary.txt"), os.path.join(HERE, f"summary_seed{seed}.txt"))
    stamp(os.path.join(tmp, "metrics.png"), os.path.join(HERE, f"metrics_seed{seed}.png"))
    if seed == 0:
        for p in PLANS:
            stamp(os.path.join(tmp, f"tracks_{p}.png"), os.path.join(HERE, f"tracks_{p}.png"))
failures_png(os.path.join(HERE, "failures.png"))
