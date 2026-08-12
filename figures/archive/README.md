# archive/ — figures from before the dating convention (pre 2026-08-13)

These were rendered ad hoc, mostly to settle one question in one session, and
their provenance is **not** recoverable from the files themselves: no date, no
gains, no plant. Do not put any of them in a write-up without re-rendering the
underlying measurement under today's convention.

What is known, from the CLAUDE.md sections that reference them:

| file | what it is | trust |
| --- | --- | --- |
| `tvlqr_validation.png` | three-panel validation of the tuned gains, 2026-08-12, current plant | **usable** — the write-up figure named in the handover |
| `soak_distributions.png` | per-shape distributions from the 4065-rollout two-point soak, 2026-08-13 | **usable**, current plant |
| `tvlqr_tune_landscape.png` | Nelder-Mead landscape, 2026-08-02 — the run whose 0.183 m result was winner's curse | historical only |
| `trajectory_gallery*.png` | all 100 plans, rotated onto principal axes; `tools/plot_trajectory_gallery.py` | usable, plans do not change |
| `checkpoints_max_cross.png`, `floor_*_checkpoints.png` | RL checkpoint sweep | historical — no checkpoint showed a learning trend |
| `corrector_compare_*.png`, `floor_*_compare.png` | corrector comparisons | **old plant (`mu2=0.7`)** unless proven otherwise — do not quote |

Anything not listed above is unidentified.
