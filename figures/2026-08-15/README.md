# 2026-08-15 — the `r_omega` question closed, the U-turn basin retracted

Plant: **2026-08-07-wheel-mu2-045** (`wheel.xacro` `mu2=0.45`), world ground
`mu=1.0`, terrain patches on. Both soaks ran from `tools/jobs/` through the VM
job queue; rows are gitignored under `soak_data/`.

Regenerate all three:

```bash
.venv/bin/python figures/2026-08-15/render.py
```

| figure | what it shows | what it establishes |
| --- | --- | --- |
| `01_r_ladder_low.png` | `r_omega` ∈ {0.25, 0.5, 1.0, 2.618} at `q_cross=0.276`, 7 shapes, n≈29/cell, from `soak_data/soak_r_ladder_low.jsonl` (840 rollouts, job 30) | **The move off `r=0.25` was NOT a one-plan result.** Excluding the U-turn, the six-shape aggregate falls 0.635 → 0.495 m between r=0.25 and r=1.0, driven by the **zigzag halving** (0.803 → 0.387) and the S (1.982 → 1.577). `final_err` is monotone in `r` across the whole range (0.661 → 0.343 m). So the previous ladder's "flat" verdict was an artifact of its floor being r=1.0 — `r` matters *below* 1.0 and is flat above it. |
| `02_uturn_generality.png` | `q_cross` ∈ {0.2, 0.276, 0.4, 0.5} at `r=2.618` on four constructed U-turn plans plus `floor_6_00031` as an in-run control, n≈59/cell, from `soak_data/soak_uturn_generality.jsonl` (1200 rollouts, job 40) | **The `q_cross` basin belongs to one plan, not to the U-turn shape.** Only the control shows it (97 / 18 / 0 / 100 % bad). Of the four others: two are flat and easy (~0.29 m at every rung), one is 100% bad at every rung, one is weakly bimodal with no `q` dependence. Retracts CLAUDE.md's "the U-turn is a narrow notch in `q_cross`" as a shape claim. |
| `03_uturn_plans.png` | the five plans job 40 actually drove, with the leading-pivot-inclusive `total_abs_turn` printed | **The `UTURN` label does not survive looking**, and the refutation above is stronger than the numbers alone said. All five score 7–9 rad of total turning; only two are hairpins. `floor_6_v2_00003` is a bent line. Critically, **`floor_6_v2_00008` is a near-duplicate of the control** — same corridor, same three-sided route, ~1 m different start — and it shows **no basin at all**, which is what rules out "the basin is a property of this route" rather than merely "of this shape". |

## The methodological point worth keeping

`03` is the figure that mattered. The numbers in `02` say the basin does not
generalise; they cannot say *why*, and a reader could answer "you picked four
plans that were not really U-turns". Rendering them turned that objection into
the evidence: the plan most similar to the control — geometrically almost the
same path — is the one that most clearly lacks the basin.

`total_abs_turn ≈ 7 rad` cannot separate a 180° hairpin from two same-sign 90°
corners, and it still includes the leading in-place pivot. This is the second
time an automatic shape label has misled here (the first was
`classify_plans.py` calling 58 of 100 plans CORNER). **Label from the picture
before making a per-shape claim.**
