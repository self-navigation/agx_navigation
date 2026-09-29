"""Overnight Nav2 comparison (#10), 2026-09-29 -- INVALID data, stamped as such.

    .venv/bin/python figures/2026-09-29/render.py

Runs tools/summarize_compare.py on run_data/compare_seed{0,1}/ and stamps a
banner on every figure it keeps: amcl re-seeded its filter on every /map
heartbeat (every 2 s) in all three arms, so these are not results. See README.
"""
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PLANS = ["floor_6_v2_00249", "floor_6_v2_00096", "floor_6_v2_00147", "floor_6_v2_00369"]
BANNER = "INVALID: amcl reset every 2 s in all arms (2026-09-29). Pipeline preview, not a result."


def stamp(src, dst):
    img = plt.imread(src)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100 + 0.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, h / (h + 40)])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.5, 1 - 20 / (h + 40), BANNER, ha="center", va="center",
             color="white", fontsize=11, weight="bold",
             bbox=dict(facecolor="red", edgecolor="none", pad=4))
    fig.savefig(dst)
    plt.close(fig)


for seed in (0, 1):
    tmp = os.path.join(ROOT, "figures", "tmp", f"cmp_s{seed}")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "summarize_compare.py"),
                    os.path.join(ROOT, "run_data", f"compare_seed{seed}"), "-o", tmp],
                   check=True, stdout=subprocess.DEVNULL)
    shutil.copy(os.path.join(tmp, "summary.txt"), os.path.join(HERE, f"summary_seed{seed}.txt"))
    stamp(os.path.join(tmp, "metrics.png"), os.path.join(HERE, f"metrics_seed{seed}.png"))
    if seed == 0:
        for p in PLANS:
            stamp(os.path.join(tmp, f"tracks_{p}.png"), os.path.join(HERE, f"tracks_{p}.png"))
