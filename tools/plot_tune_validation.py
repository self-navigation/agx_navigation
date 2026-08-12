#!/usr/bin/env python3
"""The 2026-08-07 tuning result: it validated, and then the validation moved the
goalposts.

`plot_tune_variance.py` exists to show that the two EARLIER tuning runs resolved
noise. This one is about the third run, which did not -- the improvement
(1.004 -> 0.621 m mean max|e_cross|) is ~10x the measurement noise and it
reproduced on an independent mean-of-5. That much is a clean win.

The figure is about what the validation then revealed, which is the actually
interesting part and is invisible in any aggregate plot:

  left    q_cross profile at the tuned r_omega, spanning BELOW the tuner's own
          search-box floor of 0.1, because 23 of the run's 100 evaluations piled
          against that floor. Nothing below it is better, so the optimum is
          interior and the bounds were not the limitation. Read this panel WITH
          the middle one: on its own it suggests a knife-edge optimum, and the
          grid shows the neighbourhood is actually a shallow bowl.

  middle  the 5x3 local (q, r) grid. It softens the left panel: 13 of 15 points
          beat the default, so the neighbourhood is a shallow bowl with a notch
          at the optimum rather than a knife edge. It also shows r_omega matters
          LOCALLY (0.799 / 0.643 / 0.865 at q=0.276 as r goes 1.0 / 2.618 / 6.0),
          which the run's binned r_omega table denied -- that table averaged over
          q, the same smoothing that hid the spike in the left panel.

  right   where the improvement actually comes from, per shape. The loop and the
          zigzag are 88% of it and the S adds 20%, while the U-turn is slightly
          NEGATIVE. This panel exists because the first reading of this data
          attributed the win to the U-turn and the S "landing in their good
          mode" -- computed against the SWEEP NEIGHBOURS rather than against the
          default, which is the baseline the improvement is measured from. The
          U-turn really is bistable (33% good over 45 grid rollouts, no smooth
          dependence on the gains), but both arms of the headline comparison sit
          in its good mode 5 times out of 5, so it explains none of the gap.

Offline-only, like every tool in here: matplotlib in a venv, reads the gitignored
JSONL that `just fetch-tune` pulls back. Reads data, draws, writes a PNG --
never touches the sim.

    python3 tools/plot_tune_validation.py --data-dir tune_data --out figures/tvlqr_validation.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Order and labels are the eval set's, not the file's: the JSONL key order
# differs between the _meta block and the per_traj dicts, and the shape names are
# what a reader needs. Kept explicit so a renamed trajectory fails loudly here
# rather than silently drawing the wrong series.
SHAPES = [
    ("floor_1_00049", "straight"),
    ("floor_6_00023", "corner"),
    ("floor_6_00018", "S"),
    ("floor_6_00047", "zigzag"),
    ("floor_6_00056", "tight V"),
    ("floor_6_00031", "U-turn"),
    ("floor_6_00025", "loop"),
]


def load(path):
    """Return (meta, [evaluation records]) from a tuner JSONL.

    Failed evaluations are dropped: `inf` means "the sim broke", almost never
    "these gains are bad", and plotting them as data is how a broken run gets
    read as a landscape.
    """
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    meta = rows[0].get("_meta", {}) if rows else {}
    return meta, [r for r in rows[1:] if not r.get("_failed")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tune_data")
    ap.add_argument("--out", default="figures/tvlqr_validation.png")
    args = ap.parse_args()

    d = args.data_dir
    _, sweep = load(os.path.join(d, "qwall_20260812.jsonl"))
    _, vt = load(os.path.join(d, "validate_20260812_tuned.jsonl"))
    _, vd = load(os.path.join(d, "validate_20260812_default.jsonl"))
    tuned, default = vt[0], vd[0]

    names = [n for n, _ in SHAPES]
    # The sweep is the q_cross profile; the two validated points belong on it
    # too -- same r_omega for the tuned one, and the default is drawn separately
    # since its r_omega differs and it is a reference line, not a sweep point.
    pts = sorted(sweep + [tuned], key=lambda r: r["q_cross"])
    qs = [r["q_cross"] for r in pts]
    fx = [r["fx"] for r in pts]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # ---- left: the aggregate profile ------------------------------------
    ax1.axvspan(1e-3, 0.1, color="0.92", zorder=0)
    ax1.text(1.4e-3, 0.66, "below the tuner's\nown search floor",
             fontsize=8, color="0.35", va="bottom")
    ax1.plot(qs, fx, "o-", color="#2c6fbb", lw=1.8, ms=6, zorder=3)
    ax1.axhline(default["fx"], color="#c0392b", ls="--", lw=1.4,
                label=f"default q=10 / r=0.25  ({default['fx']:.3f} m, n=5)")
    ax1.plot([tuned["q_cross"]], [tuned["fx"]], "*", ms=20, color="#e8a33d",
             mec="#6d4c00", mew=0.8, zorder=4,
             label=f"tuned q={tuned['q_cross']:.3f}  ({tuned['fx']:.3f} m, n=5)")
    ax1.set_xscale("log")
    ax1.set_xlabel("$q_{cross}$   (log scale, $r_\\omega$ = 2.618)")
    ax1.set_ylabel("mean max|$e_{cross}$|  [m]")
    ax1.set_title("1-D scan: the search floor was not the limit", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    # ---- middle: the local (q, r) grid -----------------------------------
    _, grid = load(os.path.join(d, "local2d_20260812.jsonl"))
    grid = [r for r in grid if not r.get("_failed")]
    r_values = sorted({round(r["r_omega"], 3) for r in grid})
    for i, r_val in enumerate(r_values):
        row = sorted((r for r in grid if round(r["r_omega"], 3) == r_val),
                     key=lambda r: r["q_cross"])
        ax2.plot([r["q_cross"] for r in row], [r["fx"] for r in row], "o-",
                 color=plt.get_cmap("viridis")(i / max(len(r_values) - 1, 1)),
                 lw=1.9, ms=6, label=f"$r_\\omega$ = {r_val:g}")
    ax2.axhline(default["fx"], color="#c0392b", ls="--", lw=1.4,
                label="default (1.004 m)")
    ax2.set_xscale("log")
    ax2.set_xlabel("$q_{cross}$")
    ax2.set_ylabel("mean max|$e_{cross}$|  [m]")
    ax2.set_title("A shallow bowl, and $r_\\omega$ does matter locally",
                  fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ---- right: where the improvement comes from -------------------------
    deltas = sorted(((label, default["per_traj"][k] - tuned["per_traj"][k])
                     for k, label in SHAPES), key=lambda x: x[1])
    labels = [x[0] for x in deltas]
    vals = [x[1] for x in deltas]
    colours = ["#2c8f4a" if v > 0 else "#c0392b" for v in vals]
    ax3.barh(range(len(vals)), vals, color=colours, height=0.68)
    ax3.set_yticks(range(len(vals)))
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.axvline(0, color="0.3", lw=0.9)
    total = sum(vals)
    for i, v in enumerate(vals):
        ax3.text(v + (0.03 if v > 0 else -0.03), i, f"{v:+.2f} ({100*v/total:.0f}%)",
                 va="center", ha="left" if v > 0 else "right", fontsize=8)
    ax3.set_xlim(-0.35, 1.55)
    ax3.set_xlabel("default $-$ tuned   [m]      (positive = tuned is better)")
    ax3.set_title("Loop + zigzag are 88% of the win; U-turn is not",
                  fontsize=11)
    ax3.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "TVLQR gain tuning, 2026-08-07 run validated 2026-08-12  "
        "(plant 2026-08-07-wheel-mu2-045, 7-shape eval set)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
