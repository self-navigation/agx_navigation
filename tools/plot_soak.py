#!/usr/bin/env python3
"""What a soak buys that a tuning run cannot: the SHAPE of each distribution.

Every comparison in this project up to 2026-08-13 reduced a shape to a mean of
3-5 rollouts. That is the right estimator for a unimodal metric and the wrong one
for a bimodal one, and this figure is the evidence that three of our seven shapes
are bimodal -- so their reported "run-to-run variance" was never measurement
error, it was a mixture width being sampled too few times to see.

  left    per-shape distributions, tuned vs default, as horizontal strip plots
          with the individual rollouts drawn. The bimodal shapes are immediately
          visible as two clumps; the unimodal ones as a single tick. Drawn as raw
          points rather than a violin/box on purpose: a box plot of a bimodal
          sample draws a plausible-looking box in the empty gap between the
          modes, which is exactly the artifact this figure exists to expose.

  right   the aggregate, one point per COMPLETE cycle of the 7-shape set. This is
          the quantity the tuner optimizes, and at n=14 per arm the two arms
          separate cleanly -- which is the adoption argument.

Reads the raw JSONL the soak appends to, which stores per-rollout results and
never aggregates, so this script (and any future re-weighting of the objective)
can be changed without re-driving anything.

Offline-only: matplotlib in a venv, reads gitignored data, writes a PNG.

    python3 tools/plot_soak.py --data soak_data/soak_20260813_twopoint.jsonl \
        --out figures/soak_distributions.png
"""

import argparse
import collections
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHAPES = [
    ("floor_1_00049", "straight"),
    ("floor_6_00023", "corner"),
    ("floor_6_00018", "S"),
    ("floor_6_00047", "zigzag"),
    ("floor_6_00056", "tight V"),
    ("floor_6_00031", "U-turn"),
    ("floor_6_00025", "loop"),
]

TUNED = "#2c6fbb"
DEFAULT = "#c0392b"


def load(path):
    """Usable rollouts only.

    A failed rollout means the sim broke (in practice: terrain patches did not
    spawn), never that these gains are bad. Plotting it as a sample is the
    survivorship mistake `objective.py` exists to prevent, in reverse.
    """
    rows = [json.loads(line) for line in open(path) if line.strip()]
    ok = [r for r in rows if not r.get("failed")]
    return ok, len(rows) - len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="soak_data/soak_20260813_twopoint.jsonl")
    ap.add_argument("--out", default="figures/soak_distributions.png")
    ap.add_argument("--q-split", type=float, default=1.0,
                    help="q_cross below this is the 'tuned' arm")
    args = ap.parse_args()

    ok, n_failed = load(args.data)
    arm = lambda r: "tuned" if r["q_cross"] < args.q_split else "default"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [2.1, 1]})

    # ---- left: per-shape distributions -----------------------------------
    for i, (key, label) in enumerate(SHAPES):
        for sign, name, colour in ((+1, "tuned", TUNED), (-1, "default", DEFAULT)):
            v = [r["max_cross"] for r in ok
                 if r["trajectory"] == key and arm(r) == name]
            if not v:
                continue
            # Jitter only in the offset direction, so the metric axis stays exact.
            ys = [i + sign * (0.10 + 0.16 * (j % 7) / 6.0) for j in range(len(v))]
            ax1.plot(v, ys, ".", ms=2.2, color=colour, alpha=0.45,
                     label=name if i == 0 else None)
            ax1.plot([st.mean(v)], [i + sign * 0.18], "|", ms=16, mew=2.4,
                     color=colour)
    ax1.set_yticks(range(len(SHAPES)))
    ax1.set_yticklabels([s for _, s in SHAPES])
    ax1.set_xscale("log")
    ax1.set_xlabel("max|$e_{cross}$| per rollout  [m]  (log scale)")
    ax1.set_title(f"Every rollout, by shape  (n={len(ok)}, "
                  f"{n_failed} failed)\nthree shapes are BIMODAL; four are not",
                  fontsize=11)
    ax1.grid(alpha=0.3, axis="x")
    ax1.legend(fontsize=9, loc="lower right", markerscale=4)
    ax1.invert_yaxis()

    # ---- right: the aggregate, per complete cycle -------------------------
    per_cycle = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in ok:
        per_cycle[arm(r)][r["cycle"]][r["trajectory"]] = r["max_cross"]
    for x, (name, colour) in enumerate(((("tuned"), TUNED), ("default", DEFAULT))):
        # Only complete cycles: a partial one is a mean over a different set of
        # shapes, which is not the same quantity.
        means = [st.mean(d.values()) for d in per_cycle[name].values()
                 if len(d) == len(SHAPES)]
        ax2.plot([x + 0.06 * ((j % 5) - 2) for j in range(len(means))], means,
                 "o", ms=7, color=colour, alpha=0.75)
        ax2.plot([x - 0.22, x + 0.22], [st.mean(means)] * 2, "-", lw=2.5,
                 color="0.15")
        ax2.text(x, max(means) + 0.05,
                 f"{st.mean(means):.3f} ± {st.pstdev(means):.3f}\nn={len(means)}",
                 ha="center", fontsize=9)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["tuned\nq=0.276 / r=2.618", "default\nq=10 / r=0.25"])
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_ylabel("mean max|$e_{cross}$| over the 7 shapes  [m]")
    ax2.set_title("The tuning objective, per complete cycle", fontsize=11)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("TVLQR gain soak, 2026-08-13 — plant 2026-08-07-wheel-mu2-045",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
