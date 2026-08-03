#!/usr/bin/env python3
"""Plot SAC training diagnostics from a TensorBoard event file.

Panels: ent_coef, critic_loss, actor_loss, ep_rew_mean, terminal e_cross and
episode outcome rates. Offline tool; needs matplotlib and tensorboard.

    .venv/bin/pip install tensorboard
    .venv/bin/python tools/plot_training_diagnostics.py tb_data/SAC_1 --out figures_new
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

C_MAIN = "#2a78d6"
C_WARN = "#eb6834"
C_OK = "#1baf7a"
C_MUTED = "#8a8985"
C_REF = "#52514e"


def smooth(ys, k):
    """Centred rolling mean. The raw series is drawn faintly behind it -- a
    smoothed line alone hides how noisy the underlying signal was, which for
    `ent_coef` is part of the finding."""
    if k <= 1 or len(ys) < k:
        return ys
    out, acc = [], 0.0
    from collections import deque
    win = deque()
    for y in ys:
        win.append(y)
        acc += y
        if len(win) > k:
            acc -= win.popleft()
        out.append(acc / len(win))
    return out


def series(ea, tag):
    s = ea.Scalars(tag)
    return [p.step / 1e6 for p in s], [p.value for p in s]


def panel(ax, xs, ys, color, title, ylabel, log=False, smooth_k=25):
    ax.plot(xs, ys, color=color, lw=0.8, alpha=0.22, zorder=2)
    ax.plot(xs, smooth(ys, smooth_k), color=color, lw=2.0, zorder=3)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("training steps  [millions]", fontsize=9)
    ax.grid(alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="TB run directory (e.g. tb_data/SAC_1)")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--best-step", type=float, default=0.8,
                    help="checkpoint (millions) to mark as the sweep's best")
    args = ap.parse_args()

    ea = event_accumulator.EventAccumulator(args.src, size_guidance={"scalars": 0})
    ea.Reload()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))

    # --- entropy coefficient: the headline defect ------------------------
    xs, ys = series(ea, "train/ent_coef")
    ax = axes[0][0]
    # Shade the usable band rather than drawing a single threshold line: the
    # claim is "well below 1", not "exactly 1".
    ax.axhspan(0, 1.0, color=C_OK, alpha=0.10, zorder=1)
    ax.text(xs[-1], 0.5, "usable range  ", ha="right", va="center",
            fontsize=8.5, color=C_REF)
    panel(ax, xs, ys, C_WARN, "Entropy coefficient — ran away", "ent_coef")
    ax.annotate(f"{ys[-1]:.2f}", xy=(xs[-1], ys[-1]), xytext=(-38, -14),
                textcoords="offset points", fontsize=10, color=C_WARN,
                fontweight="bold")

    # --- critic loss ------------------------------------------------------
    xs, ys = series(ea, "train/critic_loss")
    panel(axes[0][1], xs, ys, C_WARN, "Critic loss — unbounded return",
          "critic_loss", log=True)
    axes[0][1].annotate(f"{ys[-1]:.1e}", xy=(xs[-1], ys[-1]), xytext=(-46, -16),
                        textcoords="offset points", fontsize=10, color=C_WARN,
                        fontweight="bold")

    # --- actor loss -------------------------------------------------------
    xs, ys = series(ea, "train/actor_loss")
    panel(axes[0][2], xs, ys, C_MUTED,
          "Actor loss — dominated by the entropy term", "actor_loss", log=True)

    # --- episode return ---------------------------------------------------
    xs, ys = series(ea, "rollout/ep_rew_mean")
    panel(axes[1][0], xs, ys, C_MAIN, "Mean episode return — barely moved",
          "ep_rew_mean")

    # --- terminal tracking error: the task actually improved --------------
    xs, ys = series(ea, "rollout/terminal_abs_e_cross")
    ax = axes[1][1]
    panel(ax, xs, ys, C_MUTED,
          "Terminal error as the BROKEN env logged it\n(not confirmed by re-measurement)",
          "|e_cross| [m]")
    ax.axvline(args.best_step, color=C_REF, ls="--", lw=1.2, zorder=4)
    ax.text(args.best_step, ax.get_ylim()[1], " best checkpoint", fontsize=8,
            color=C_REF, va="top")

    # --- why episodes ended ----------------------------------------------
    ax = axes[1][2]
    for tag, color, label in (
        ("outcomes/corridor_rate", C_WARN, "left the corridor"),
        ("outcomes/heading_rate", C_MAIN, "heading violation"),
        ("outcomes/ran_out_rate", C_MUTED, "ran out of plan"),
    ):
        xs, ys = series(ea, tag)
        ax.plot(xs, smooth(ys, 25), color=color, lw=2.0, label=label, zorder=3)
    ax.set_title("Why episodes ended", fontsize=10.5)
    ax.set_ylabel("fraction of episodes", fontsize=9)
    ax.set_xlabel("training steps  [millions]", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle(
        "SAC training run 20260730 — 1.5M steps, 8864 episodes, 2 successes.  "
        "Two named defects: the entropy coefficient diverged (top left) and the "
        "return was unbounded (top centre).\n"
        "Optimiser diagnostics (top row) are valid — they describe SAC itself. The rollout "
        "metrics (bottom row) were logged by a mis-stepped environment\n"
        "and are NOT evidence of learning: a clean 20-checkpoint re-measurement shows no "
        "trend against training step.",
        fontsize=10.5, y=0.99,
    )

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "rl_training_diagnostics.png")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(dest, dpi=150)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
