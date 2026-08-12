# 2026-08-13 — the U-turn's one corner, the `q_cross` notch, and `J` vs metres

**Plant:** `wheel.xacro` `mu2=0.45` (the 2026-08-07 fix), world ground `mu=1.0`,
terrain patches on, seed 0. Gains: **tuned** = `q_cross 0.276 / r_omega 2.618`
(adopted this day), **default** = `10 / 0.25`.

Regenerate all five:

```bash
.venv/bin/python figures/2026-08-13/render.py
```

Inputs, all gitignored — the pictures are committed because the renderer alone
cannot reproduce them (see [../README.md](../README.md)):

| directory | what | how it was made |
| --- | --- | --- |
| `traj_data/` | the 100 PMP plans | `rsync` from the VM's `~/pmp_trajectories_v2` |
| `uturn_traces/{tuned,default}/` | 30 per-step traces of `floor_6_00031` | `variance_probe --trace-dir`, 2026-08-13 |
| `soak_data/soak_20260813_ladder.jsonl` | 1047 rollouts, 3 shapes × 6 gain points | `just soak`, ~3 h |
| `jtraces/{tuned,default}/<shape>/` | 70 traces, 7 shapes × 2 arms × 5 | `~/jsweep.sh` on the VM, ~14 min |
| `epsilon_data/*.jsonl` | those traces scored in `J` | `tools/score_epsilon.py` |

---

## 01_shapes_overview.png — the seven evaluation plans

All seven to scale, with path length and total turning. **Establishes that the
U-turn is not a uniquely hard shape**: 17.2 m and 460° of turning puts it second
in both, behind the zigzag (20.4 m, 563°), and by deviation it scores better than
the S. Asked and answered, because "the U-turn is the hard one" had been assumed.

## 02_uturn_modes.png — where the deviation actually happens

30 rollouts, blue = good mode, red = bad (> 2.0 m). **The first 12.5 of 17 metres
are tracked to within ~0.25 m in every rollout, in both arms.** All the deviation
is produced at the *second* 90° corner, in the last quarter of the run. So
`max|e_cross|` on this trajectory is a **single-event metric** — one overshoot,
at one corner — which is why it looked bistable and resisted explanation.

## 03_ladder_modes.png — bad-mode frequency vs `q_cross`

`r_omega` held at 2.618 across the ladder so `q`'s effect is separable; × marks
the old default's `r=0.25`. Two different phenomena:

- **Zigzag: a threshold.** Flat at 0-2% bad from `q=0.1` to `1.5`, then a cliff to
  ~89% at `q=10`. `r` is irrelevant to it (86.0% vs 89.7% at `q=10`). The adopted
  point sits well inside a wide plateau.
- **U-turn: a narrow notch, and NOT a mode frequency.** The scatter shows
  `q=0.1` giving 59 rollouts inside 2.634–2.665 and `q=0.6` giving 56 inside
  2.701–2.709 — tight, deterministic, unimodal and bad. Only `q=0.276` and the
  old default drop to ~1.4. **This retracts "the gains do not control the U-turn
  at all"** (2026-08-13 morning, from two points); they control it sharply and
  non-monotonically.

## 04_epsilon_vs_cross.png — does changing the objective change the conclusion?

Per shape, mean of 5, scored both ways. Right panel plots the ratios against each
other; shaded = the metrics rank the arms differently.

**The overall conclusion survives and strengthens.** In `max|e_cross|` the tuned
gains win 4 shapes and lose 3 (corner, tight V, U-turn). **In `J` they win all
seven**, by 1.3× to 8.3×. Every one of the three "losses" was a shape where the
tuned arm gave up a little peak deviation and bought back much more in
accumulated error, correction effort and stopping short.

So `J` is not a different answer, it is a **less noisy version of the same
answer** — which is the useful outcome, since it is also the quantity the
advisor's framework (SVCM, `J[u] <= J* + eps`) is stated in.

## 05_what_is_J.png — what the two metrics *are*

Written for the question "I can picture cross-track error; I cannot picture `J`."
One good and one bad U-turn rollout: `max|e_cross|` is the **height of the tallest
spike**; `J` is the **weighted area under the whole curve**, plus what the
corrector spent correcting, plus a penalty for stopping short of the goal.

The right-hand panel is the clearest case: it peaks at 2.30 m against 1.46 m —
1.6× worse by the old metric — but it is off-plan for *most of the second half*
and ends 2.7 m from the goal, so it costs `J = 188` against `18`, a 10× gap.
**Note `J` is an upper bound on `epsilon`, not `epsilon`** (`J* > 0` under slip
and is unknown); see `tuning/epsilon.py`'s docstring before quoting it.
