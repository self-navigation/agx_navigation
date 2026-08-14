# 2026-08-14 — the `r_omega` ladder

**Plant for every figure:** wheel `mu2=0.45` (2026-08-07), world ground `mu=1.0`,
terrain patches on, seed fixed.

**Data:** `soak_data/soak_r_ladder.jsonl` — 1050 rollouts, 1035 usable (15 lost
to the patch-spawn guard), 7 shapes × `r_omega` ∈ {1.0, 1.8, 2.618, 3.5, 5.0}
with `q_cross` held at the adopted 0.276, n≈30 per cell. Produced by
`tools/jobs/10_r_ladder.sh` on the VM.

Regenerate both with:

    .venv/bin/python figures/2026-08-14/render.py

**What this set establishes:** `r_omega` is flat on five of seven shapes across a
5× range. The aggregate's preference for the adopted 2.618 comes almost entirely
from **one plan** — the U-turn, which is ~100% bad at every other rung. So
`r_omega=2.618` sits on an isolated notch, the opposite of `q_cross=0.276`, which
the 2026-08-13 ladder put on a *wide plateau* for the zigzag.

Not scoreable in `J`: the traces this run wrote are unusable (see CLAUDE.md,
"the traced-soak path had two bugs"), so both figures are in metres only.

| figure | shows | establishes |
| --- | --- | --- |
| `01_r_ladder.png` | `max\|e_cross\|` per rung, one panel per shape, plus the aggregate with and without the U-turn | Watch the y-axes: straight spans 0.04–0.08 m, corner 0.30–0.34, tight V 0.28–0.30. Five shapes cannot tell the rungs apart. The aggregate line moves; the "without U-turn" line is flat at ~0.52 from r=1.0 to r=2.618, so `r` earns nothing outside that one plan. |
| `02_r_modes.png` | bad-mode rate vs `r`, for the only two shapes that have one | The U-turn is 100% bad at 1.0, 1.8, 3.5 and 5.0, and 20% bad at 2.618 — a notch one rung wide. The S is clean everywhere except a **deterministic** bad spike at r=3.5 (2.548–2.552, sd 0.001), which is a second isolated feature rather than noise. |

**Caveat carried forward:** the ladder's floor is `r=1.0`, while the gain it
replaced is `r=0.25`. Nothing here says whether `r`'s half of the adopted joint
move bought anything below 1.0. `tools/jobs/30_r_ladder_low.sh` closes that.
