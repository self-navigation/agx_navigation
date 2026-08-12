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
          search-box floor of 0.1. The optimum is interior (so the floor was not
          the problem) but it is a NARROW SPIKE -- the neighbours at 0.1 and 0.6
          score level with the default. The monotone trend seen when the 100
          evaluations are binned by q_cross is a smoothing artifact of averaging
          over what the middle panel shows.

  middle  the same sweep resolved PER SHAPE. Five of the seven shapes are flat
          across three orders of magnitude in gain. The U-turn and the S are
          bistable -- each sits at ~1.2 or ~2.7 and nothing between -- and those
          two alone account for 0.34 m of the 0.38 m improvement. So the
          objective is substantially selecting MODES, not tracking quality.

  right   the individual repeats behind the two validated points. This is the
          honest argument for adopting the tuned gains, and it is not the mean:
          the default gains are visibly BIMODAL across repeats (sd 0.137) while
          the tuned point is tight (sd 0.020). More repeatable, not just lower.

Offline-only, like every tool in here: matplotlib in a venv, reads the gitignored
JSONL that `just fetch-tune` pulls back. Reads data, draws, writes a PNG --
never touches the sim.

    python3 tools/plot_tune_validation.py --data-dir tune_data --out figures/tvlqr_validation.png
"""

import argparse
import json
import os
import statistics as st

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
# The two bistable ones, drawn heavy because they are the finding.
BISTABLE = {"floor_6_00031", "floor_6_00018"}


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


def repeat_aggregates(rec, names):
    """The per-repeat aggregate scores behind one evaluation.

    The record stores per-trajectory values per repeat; the aggregate the tuner
    optimises is the mean over trajectories, so reduce the same way here or the
    spread shown will not be the spread the search saw.
    """
    reps = rec.get("per_traj_repeats") or []
    return [st.mean(d[n] for n in names) for d in reps if all(n in d for n in names)]


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
    ax1.set_title("The optimum is a narrow spike, not a basin", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    # ---- middle: per shape ----------------------------------------------
    cmap = plt.get_cmap("tab10")
    for i, (key, label) in enumerate(SHAPES):
        heavy = key in BISTABLE
        ax2.plot([r["q_cross"] for r in pts], [r["per_traj"][key] for r in pts],
                 "o-", color=cmap(i), label=label,
                 lw=2.6 if heavy else 1.1, ms=6 if heavy else 3.5,
                 alpha=1.0 if heavy else 0.55, zorder=3 if heavy else 2)
    ax2.axvline(tuned["q_cross"], color="#e8a33d", lw=1.2, ls=":", zorder=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("$q_{cross}$")
    ax2.set_ylabel("max|$e_{cross}$|  [m]")
    ax2.set_title("Two shapes are bistable; five are flat", fontsize=11)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)

    # ---- right: the repeats ---------------------------------------------
    arms = [("tuned\nq=0.276 / r=2.618", tuned, "#2c6fbb"),
            ("default\nq=10 / r=0.25", default, "#c0392b")]
    for i, (label, rec, colour) in enumerate(arms):
        vals = repeat_aggregates(rec, names)
        ax3.scatter([i] * len(vals), vals, s=90, color=colour, zorder=3,
                    edgecolor="white", linewidth=0.8)
        ax3.hlines(st.mean(vals), i - 0.22, i + 0.22, color=colour, lw=2.5)
        ax3.text(i + 0.30, st.mean(vals),
                 f"mean {st.mean(vals):.3f}\nsd {st.pstdev(vals):.3f}",
                 fontsize=9, va="center", color=colour)
    ax3.set_xticks(range(len(arms)))
    ax3.set_xticklabels([a[0] for a in arms], fontsize=9)
    ax3.set_xlim(-0.5, 1.9)
    ax3.set_ylabel("mean max|$e_{cross}$| per repeat  [m]")
    ax3.set_title("The default is bimodal across repeats", fontsize=11)
    ax3.grid(alpha=0.3, axis="y")

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
