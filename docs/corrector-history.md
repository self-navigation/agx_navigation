# Corrector work — history

Superseded tables, resolved-bug investigations, and the reasoning behind retractions. **Split out of CLAUDE.md on 2026-08-13** because it was ~1750 lines and loads into context every session.

The split rule is **not** chronological. A finding stays in CLAUDE.md if it still constrains a decision — a 'do not re-propose X' is the highest-value text in the file and costs one line. What moved here is the *narrative* behind those conclusions and *measurement tables taken on plants or rigs that no longer exist*. Every moved section left a one-line stub in CLAUDE.md pointing here; if you are about to re-propose something, the stub should stop you and this file should explain why.

Read this when you want to know *why* a claim was retracted, or when a stub in CLAUDE.md is too terse to act on. Do not re-measure anything here without checking whether the plant changed (`PLANT_VERSION`).


### The 20260730 training run did not learn a usable policy

1.5M steps, 9h41m, 8864 episodes, **2 successes**. `failure_rate 0.88`. Two
diagnostics matter more than the step count:

- **`ent_coef` ran away to 3.31.** SAC's auto-tuned entropy coefficient belongs
  well below 1; at 3.31 the entropy bonus dominates the return and the optimal
  policy under the *effective* objective is near-random. `actor_loss 4.46e3` is
  mostly that term.
- **`critic_loss` is back to 1.23e4.** The Huber reward fix had brought it to
  58.5. Huber bounds the *slope*, not the return: over a 200-step episode with
  no corridor termination the accumulated linear tail still diverges.

The 16-checkpoint sweep (2026-08-01, `figures/checkpoints_max_cross.png` and
`figures/*_checkpoints.png`) settles it: **no RL checkpoint beats the identity
baseline on any of the three shapes, at any point in training.** There is no
trend — max|e_cross| wanders between roughly 0.3 and 6.6 m — and the best
checkpoint is **800k** (0.59 / 0.91 / 1.06 m on straight / S-curve / corner),
after which it degrades: by 1.5M the S-curve is back to 4.65 m. Training past
~800k made the policy worse, which is what an `ent_coef` of 3.31 predicts.
More steps is not the fix; the entropy target and the unbounded return are.

Three-way comparison at the best (800k) checkpoint, clean world
(max|e_cross| / final_err, m) — `figures/*_compare.png`:

| trajectory | shape | identity | tvlqr | rl (800k) |
| --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | 0.11 / 0.49 | **0.01 / 0.02** | 0.66 / 0.25 |
| floor_6_00042 | S-CURVE | **0.20 / 0.18** | 1.55 / 0.52 | 0.96 / 0.43 |
| floor_6_00023 | CORNER | 1.51 / 3.90 | **0.23 / 0.05** | 1.04 / 1.46 |

So on a clean world: TVLQR is excellent on the straight and the corner and
**bad on the S-curve** (it oscillates — visible as loops in the path panel);
identity is the best S-curve tracker and only fails at the corner, where it ends
3.9 m out; RL is never best at anything.

Note identity re-measured as 0.11 / 0.20 / 1.51 here against 0.28 / 0.22 / 1.21
in the sweep an hour earlier — same seed, same terrain, deterministic stepping.
That residual run-to-run spread is the still-unexplained offline-mode variance
(see [[rl-corrector-diagnosis]]); it is small enough not to affect any
conclusion above, but do not read two-decimal differences as signal.


### Deterministic mode was never actually paused (found 2026-08-02, evening)

**This supersedes every measurement in this file, including the terrain-spawn
result below.** `WorldControl.pause` is a plain proto3 bool, so a request that
sets only `multi_step` sends `pause: false` — and gz applies it. Every step we
issued therefore stepped the world `n` ticks *and un-paused it*, leaving it
FREE-RUNNING until the next call, for however long the CPU gave it.

So "deterministic mode" was running an unbounded, wall-clock-dependent amount of
extra physics per control step. Symptom, once the trace made it visible: control
steps advancing **0.42 s of sim time instead of `control_dt` = 0.1**. Nothing
reported a problem — `lost_steps` and `stale_pose_steps` were both 0, correctly:
the world was not dropping steps, it was doing *extra* ones.

Fixed in `_world_control`: deterministic mode re-asserts `pause=True` on every
multi_step. Plus `_ensure_paused()`, which pauses and then **verifies the sim
clock stopped**, retrying and finally raising — the old code fired a best-effort
pause at construction and never checked, and the ack is unreliable.

**Result on floor_6_00042, 5 rollouts: max|e_cross| spread 0.0013 m**
(1.9539-1.9552), from 0.375 m before this and 6.70 m before the terrain fix.

Two more seeds were then closed, both wall-clock-paced work that fed real
physics ticks:

- **The teleport confirm loop ran for 0.5 s of WALL time**, so the robot got
  12-31 physics ticks to fall and settle depending on machine timing
  (`reset_ticks`). Now a fixed 20 (`_set_pose_stepped`). It had been written off
  as self-correcting because each retry yanks the body back — true of x/y, false
  of the vertical and contact state.
- **The reset settle ran a fixed 5 steps** and left the robot micro-bouncing by
  ~2e-5 m in z. Now converges (`reset_settle_z_tol`), leaving ~1e-6.


### Where the reproducibility floor actually is (2026-08-02, 10 rollouts)

| metric | mean | sd | spread |
| --- | --- | --- | --- |
| **max\|e_cross\|** | 1.9551 | **0.0002** | 0.0007 |
| rms_cross | 0.6214 | 0.0154 | 0.0503 |
| final_err | 0.5877 | **0.2633** | 0.8360 |

**`max|e_cross|` — the tuner's objective — is reproducible to four decimals ON
THIS TRAJECTORY.** That was read as "single-sample ranking is finally
legitimate", and **that generalisation is wrong** — see "The 0.0002 m noise floor
does not transfer" below. `floor_6_00042` has since been dropped from the eval
set, so this figure now describes a trajectory nothing is measured on.

**`final_err` is NOT reproducible and must not be used as an objective**, or must
be averaged over repeats. Why, from the per-column onsets: with everything else
fixed, the remaining seed is ~1e-13 in the wheels' residual speed (they settle to
~1e-9 rad/s, not to zero), and it is amplified at the **turn reversal around step
165**, where `omega` crosses zero (+1.37 → 0.02 → -1.29) and the skid-steer's
lateral friction switches direction. A contact-mode switch at float-level
asymmetry: genuine chaos, not a bug, and not worth chasing further.

Ruled out along the way, so don't re-propose: the ROS-publish-vs-gz-step race
(wheel speeds diverge at step 2, *after* the pose at step 1 — the command path is
a consequence, not a cause), and any terrain difference (`terrain`, `sim_time`
and `world_steps` now never differ between rollouts).


### The 0.0002 m noise floor does not transfer (found 2026-08-03)

The overnight tuning run on the **7-trajectory** set (138 evals, 1.3 h, results in
`tune_data/`) reported `q_cross=9.996 / r_omega=1.252` → **0.9412 m** from a
1.1405 m baseline. **Do not adopt it.** Reading the full JSONL rather than the
reported optimum:

- the simplex **stopped moving at eval 49** and then re-evaluated ONE gain pair
  **71 times** (68 distinct pairs over 131 valid evals; the rest are that point);
- those 71 repeats — identical gains, trajectories and seed — span
  **0.9412-1.3052 m**, sd **0.0886**, mean **1.0468**.

So the reported best is the **minimum of 71 noisy draws**, biased low by
winner's curse, and the claimed 0.199 m improvement is smaller than the 0.364 m
spread it was selected from. Nelder-Mead cannot converge on a noisy objective —
it shrinks and re-samples forever, which is exactly what the log shows.

**The root error is the generalisation, not the tuner.** The 0.0002 m floor was
measured on `floor_6_00042` alone — since dropped for being the wrong shape — and
assumed to carry to the seven-shape set that replaced it. It does not: noise
there is ~400x higher, because the harder shapes contain turn reversals and a
turn reversal is the chaotic amplifier already documented above.

Per-trajectory sd across the run, which is where the noise actually lives:

| trajectory | shape | sd | spread |
| --- | --- | --- | --- |
| floor_6_00018 | S | 0.026 | 0.33 |
| floor_6_00031 | U-TURN | 0.148 | 1.51 |
| floor_6_00023 | CORNER | 0.187 | 0.89 |
| floor_6_00047 | ZIGZAG | 0.215 | 1.68 |
| floor_6_00025 | LOOP | 0.280 | 1.43 |
| floor_1_00049 | STRAIGHT | 0.501 | 1.50 |
| floor_6_00056 | TIGHT V | 0.582 | 2.80 |

**Consequence: single-sample ranking is not valid on this eval set.** Any tuning
result needs repeats and a comparison of distributions. This raises the priority
of parallel sims (queue item 7) — repeats are now mandatory and embarrassingly
parallel. `tools/plot_tune_variance.py` draws this; `figures_new/`.


### The first tuning run that is probably real (2026-08-07, read 2026-08-12)

100 Bayesian-optimization evaluations, mean-of-3, 3.2 h, **zero failures**, on the
repaired `mu2=0.45` plant. History in `tune_data/tvlqr_tune_v4_newplant.jsonl`,
result in `tune_data/tvlqr_tuned.json`. `converged: false` — the 100-evaluation
budget ran out, it did not stop moving.

| | `q_cross` | `r_omega` | mean max\|e_cross\| |
| --- | --- | --- | --- |
| default (eval 1) | 10.0 | 0.25 | **1.042** |
| best (eval 48) | **0.276** | **2.618** | **0.614** |

**Unlike the 2026-08-02 and 2026-08-03 runs, the improvement is far larger than
the noise it was selected from.** The within-evaluation SEM of a mean-of-3 is
**0.026 m** (median over the 100 evals; mean 0.046), against a 0.43 m
improvement — roughly 10x. The mean-of-3 change of 2026-08-04 is what bought
this; the two earlier runs were selecting draws from a spread bigger than their
claimed gain. BO also reported the **posterior mean** (0.6147) essentially equal
to the observed draw (0.6144), which is what a well-conditioned surrogate does.

**The robust finding is qualitative, and it does not depend on trusting the
single best point.** Binning all 100 evaluations shows a clean monotone trend in
`q_cross` and none at all in `r_omega`:

| `q_cross` bin | n | mean fx | | `r_omega` bin | n | mean fx |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10-0.32 | 23 | **0.942** | | 0.01-0.1 | 32 | 1.038 |
| 0.32-1.0 | 22 | 0.995 | | 0.1-1 | 16 | 1.227 |
| 1.0-3.2 | 20 | 1.021 | | 1-10 | 30 | 1.026 |
| 3.2-10 | 10 | 1.144 | | 10-100 | 21 | 1.200 |
| 10-32 | 10 | 1.198 | | 100-1000 | 1 | 1.256 |
| 32-1000 | 15 | 1.515 | | | | |

So: **cross-track feedback should be ~30x weaker than the default, and `r_omega`
is close to irrelevant.** That is consistent with the standing "TVLQR oscillates
on the S-curve" claim — an over-aggressive cross-track gain is exactly what
oscillates. Per-trajectory, the tuned point wins big on the zigzag (1.80 → 0.42)
and the loop (1.50 → 0.43), and is a wash on the corner (0.23 → 0.31) and the
U-turn (1.14 → 1.25).

**Two things the tuner structurally cannot answer about its own result:**

- **It never re-measured its best point.** 100 evaluations, **100 distinct gain
  pairs, zero repeats** — the opposite failure mode from the Nelder-Mead run that
  re-sampled one point 71 times. The posterior mean protects against winner's
  curse in the *reporting*, but nothing here is a direct measurement of the
  optimum.
- **The search piled against its own lower bound.** `BOUNDS_LOG` floors
  `q_cross` at 0.1; **23 of 100 evaluations landed in [0.1, 0.32)** and the best
  sits at 0.276. The optimum may be outside the box. It is *not* simply "turn
  TVLQR off" — identity scores 2.127 on this plant against TVLQR's 1.127 — so
  there is a genuine interior minimum, but we do not know where.

Both were tested on 2026-08-12; see the next two sections. Short version: the
point **validates**, the **bounds were not the issue**, and the *reason* it wins
is not the one the binned table implies.


### The optimum is a narrow spike, not a basin (2026-08-12)

`q_cross` walked down through the search box's 0.1 floor at the tuned
`r_omega`, mean-of-3 each (`tune_data/qwall_20260812.jsonl`; needs the new
`--q-bounds` flag, because `x0` is **clipped** into the box and a probe at
q=0.003 without it silently measures q=0.1).

| `q_cross` | 0.003 | 0.010 | 0.030 | 0.100 | **0.276** | 0.600 | 1.500 | 10.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean max\|e_cross\| | 0.900 | 1.261 | 0.960 | 1.080 | **0.621** | 0.993 | 0.916 | 1.004 |

**The bounds were not the problem** — nothing below 0.1 is any good, so the
optimum is interior after all, and the "23 evaluations against the floor" reading
was wrong. But the minimum is **narrow**: the immediate neighbours at 0.1 and 0.6
score ~1.0, level with the default. A factor of two in either direction throws
away the entire improvement.

**And the monotone `q_cross` trend in the binned table above is a SMOOTHING
ARTIFACT.** Averaging 100 evaluations per bin hid a landscape that is not smooth
at all.

**Why it wins, per shape — this is the part that matters:**

| `q_cross` | straight | corner | **S** | zigzag | tight V | **U-turn** | loop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.048 | 0.316 | 2.537 | 1.266 | 0.287 | 2.649 | 0.458 |
| **0.276** | 0.054 | 0.309 | **1.604** | 0.408 | 0.286 | **1.237** | 0.451 |
| 0.600 | 0.096 | 0.298 | 2.526 | 0.384 | 0.282 | 2.707 | 0.661 |
| 10.0 | 0.068 | 0.226 | 2.130 | 1.537 | 0.234 | **1.146** | 1.684 |

**The U-turn is BISTABLE** — either ~1.2 m or ~2.7 m across this row, nothing
between — and at most gains in the low-`q` region it sits in the bad mode, while
q=0.276 and the default both reach the good one.

**RETRACTED, same day, by the 2026-08-12 grid below: "the U-turn and the S are
bistable and the tuned gains win by landing both in their good mode", and the
"0.34 m of the 0.38 m" attribution that went with it.** That was computed against
the q-SWEEP NEIGHBOURS (q=0.1, q=0.6), where the U-turn happens to be in its bad
mode — not against the **default**, which is the baseline the improvement is
actually measured from. The default reaches the good U-turn mode reliably
(1.070-1.345 over 5 repeats), so the U-turn contributes **nothing** to the
headline improvement. Decomposition below; the error was picking the wrong
baseline, and the lesson is that "shape X drives the result" must always name
which comparison it is driving.

**Repeat-level evidence that the tuned point is nonetheless the stable one:**

| gains | n | the individual repeats | sd |
| --- | --- | --- | --- |
| tuned 0.276 | 5 | 0.640 0.586 0.619 0.620 0.642 | **0.020** |
| default 10.0 | 5 | 1.133 0.835 1.061 0.846 1.144 | **0.137** |
| 0.600 | 3 | 0.985 0.995 1.000 | 0.006 |
| 0.100 | 3 | 0.980 1.314 0.946 | 0.166 |

**The default gains are visibly BIMODAL across repeats** (two clusters, ~0.84 and
~1.11) while the tuned point is tight. So the tuned gains are not merely better
on average, they are *more repeatable* — which is a better argument for adopting
them than the mean is.


### The local grid: where the improvement actually comes from (2026-08-12)

5 `q_cross` x 3 `r_omega` around the validated point, mean-of-3 each,
`tune_data/local2d_20260812.jsonl`. One evaluation (q=0.15, r=2.618) failed on
its first attempt — `terrain patches failed to spawn: ['rl_patch_2']` — and was
correctly invalidated rather than averaged over survivors, then re-run.

| `r_omega` \ `q_cross` | 0.150 | 0.200 | **0.276** | 0.350 | 0.450 |
| --- | --- | --- | --- | --- | --- |
| 1.000 | 1.085 | 0.896 | 0.799 | 0.847 | 1.123 |
| **2.618** | 0.696 | 0.847 | **0.643** | 0.784 | 0.807 |
| 6.000 | 0.869 | 0.720 | 0.865 | 0.853 | 0.874 |

Three things follow, and the third supersedes this morning's reading.

**1. The tuned point is confirmed a third time and is the grid's best.** It has
now measured 0.6144 (mean-of-3, the tuning run), 0.6212 (mean-of-5, validation)
and 0.6429 (mean-of-3, here) in three independent processes. Nothing else in the
grid goes below 0.72.

**2. "Narrow spike" was too strong — soften it.** 13 of the 15 grid points beat
the default's 1.004 and the other two roughly tie it, across ±60% in `q` and a
factor of 6 in `r`. The neighbourhood is a shallow bowl with a deeper notch at
the optimum, not a knife edge; the 1-D sweep's alarming neighbours (q=0.1 →
1.080, q=0.6 → 0.993) are at its *edges*. The gains are less fragile than this
morning's section says. (The re-run of the failed cell landed at 0.696 — the
grid's second-best — so the row is non-monotone but uniformly good.)

**3. `r_omega` DOES matter locally**, contradicting the binned table from the
100-eval run. At q=0.276 it runs 0.799 / 0.643 / 0.865 as `r` goes 1.0 / 2.618 /
6.0. That table averaged over `q`, exactly the smoothing that hid the `q`
structure — **second instance of the same artifact in one day.** Treat any
one-variable summary of this landscape as suspect by default.

**Where the improvement really comes from** (tuned vs default, both mean-of-5):

| shape | default | tuned | delta | share |
| --- | --- | --- | --- | --- |
| **loop** | 1.684 | 0.451 | **+1.233** | **46%** |
| **zigzag** | 1.537 | 0.408 | **+1.129** | **42%** |
| S | 2.130 | 1.604 | +0.526 | 20% |
| straight | 0.068 | 0.054 | +0.014 | 1% |
| tight V | 0.234 | 0.286 | -0.051 | -3% |
| corner | 0.226 | 0.309 | -0.083 | -3% |
| U-turn | 1.146 | 1.237 | -0.090 | -3% |
| **mean** | **1.004** | **0.621** | **+0.383** | |

**The loop and the zigzag alone are 88% of it, the S adds 20%, and the U-turn is
slightly NEGATIVE.** So the headline improvement is *not* mode luck — it is three
shapes improving substantially, on a metric that is reproducible at those shapes.
That is a stronger result than this morning's, not a weaker one.

**The bistability finding survives, with its scope corrected.** It is real: over
all 45 grid rollouts the U-turn is bimodal, 33% in the good mode (mean 1.555) and
67% in the bad (mean 2.575), with no smooth dependence on `(q, r)` — at r=1.0 it
is bad at every `q`, at r=2.618 good at 0.276 and 0.45, at r=6.0 good only at
0.20. But it is a hazard for **comparing neighbouring gain points to each other**,
not a contaminant of the tuned-vs-default headline, where both arms sit in the
good mode 5 times out of 5.

**Practical rule this establishes: a per-shape claim must name its baseline.**
The retraction above happened because "which shape drives the difference" was
computed against the sweep neighbours while the difference being explained was
against the default.

**ADOPTED 2026-08-13** — `TVLQRConfig` now defaults to `q_cross=0.276 /
r_omega=2.618`. Measured four times in four processes (0.614 / 0.621 / 0.643 /
0.669), best of 15 grid points, and the 4065-rollout soak below shows the win is
mode frequency on the zigzag plus clean level shifts, not mode luck. The
U-turn's bistability is *still* unexplained, but the soak shows the gains do not
affect it (10.3% vs 12.4% bad), so it was never a reason to withhold adoption.


### Clean three-way comparison (2026-08-03) — measured on the BROKEN plant

**Historical only** — superseded by the table above; `mu2=0.7` meant the robot
could not steer on any of its own patch profiles.

Seven shapes, fixed bridge, terrain on, RL at the 800k checkpoint.
`tools/plot_corrector_summary.py`, data in `compare_data_new/`
(max|e_cross| / final_err, m):

| trajectory | shape | identity | tvlqr | rl (800k) |
| --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | **0.11**/0.47 | 0.38/0.21 | 0.65/0.68 |
| floor_6_00023 | CORNER | 1.19/3.49 | 1.36/3.14 | **1.16**/2.63 |
| floor_6_00018 | S-CURVE | 4.22/5.42 | **1.06**/0.99 | 1.47/1.60 |
| floor_6_00047 | ZIGZAG | 5.16/5.38 | **1.41**/1.12 | 2.87/2.87 |
| floor_6_00056 | TIGHT V | 3.01/3.01 | **1.15**/1.22 | 3.69/4.61 |
| floor_6_00031 | U-TURN | 5.53/6.30 | 1.40/0.25 | **1.39**/1.43 |
| floor_6_00025 | LOOP | 5.47/5.56 | **1.63**/0.78 | 3.33/3.34 |
| **mean** | | **3.53** | **1.20** | **2.08** |

**TVLQR at the DEFAULT gains cuts worst-case deviation by 66% over open loop**,
and wins 5 of 7 shapes. Two retractions follow:

- ~~**"TVLQR oscillates on S-curves" is dead.**~~ **This retraction was itself
  wrong** — see the 2026-08-07 baseline above. TVLQR did win the S here (1.06 vs
  identity's 4.22), but only because open loop was failing everywhere on a plant
  that could not steer. On the repaired plant it loses the S 2.13 to 0.84, at
  sd 0.003. The oscillation claim is reinstated.
- **"No RL checkpoint beats identity on anything" is dead** — but only just, and
  only at the best checkpoints. RL(800k) beats identity on 6 of 7 shapes here.
  **That checkpoint is not representative**; see the clean sweep below.


### The clean checkpoint sweep kills the "RL was learning" reading (2026-08-03)

20 checkpoints of `runs_20260730` (stride 15, every 75k steps) re-measured on the
repaired bridge, `--correctors rl` only since identity/TVLQR are
checkpoint-independent. `tools/plot_checkpoints_clean.py`, data in `sweep_clean/`.

| | value |
| --- | --- |
| mean over 20 checkpoints | **3.475 m** (identity is 3.527) |
| sd across checkpoints | 0.716 |
| range | 2.030 (@980k) - 4.579 (@305k) |
| Pearson r vs training step | **0.111 — no trend** |
| beat identity (3.527) | **8 of 20** |
| beat TVLQR (1.198) | **0 of 20** |

**There is no learning trend on the real task**, and the typical checkpoint is
level with open loop. The 800k checkpoint used in the comparison table above
(2.08 m) sits in one of only three good pockets (755k/830k/980k) — it was picked
because the *old contaminated* sweep called it best, so quoting it as "the RL
result" is the same winner's-curse error as the tuning run. Quote it as **best
checkpoint**, never as typical.

**The checkpoint-to-checkpoint swing is real, not measurement noise.** Adjacent
checkpoints 75k steps apart differ by ~2 m, against a measurement sd of ~0.09 m
on this eval set. So the policy genuinely lurches between saves — exactly what
`ent_coef=3.31` predicts, since a near-random policy makes every save a different
random draw.

**Retract the TB reading.** `rollout/terminal_abs_e_cross` falls 3.9 → 1.6 m
across training, which looks like the policy learning the task while the
optimiser diverged. It is not: that metric was logged **by the mis-stepped
environment**, so it reports progress at a task that was not the task. The clean
sweep is the out-of-band check and it shows no trend. The optimiser panels
(`ent_coef`, `critic_loss`) remain valid — they describe SAC, not the plant.
`tools/plot_training_diagnostics.py` now says so on the figure itself.


### The wheel-velocity residual was NOT the seed (2026-08-04)

The handover's "do this first" experiment is done and the answer is **no**. Two
arms on `floor_6_00056` (TIGHT V, the worst offender), 5 rollouts each,
everything else fixed: A = today's reset, B = `--reset-world` (a full gz
`WorldControl.reset.all`, the only mechanism that zeroes JOINT velocities).
`tools/run_reset_world_probe.sh`, `GazeboBridge(reset_world=True)`,
`variance_probe --reset-world`.

**The premise was wrong on its own terms.** The wheels do not settle to ~1e-9
rad/s — `trace_diff --eps 0` shows them already agreeing to **1e-16..1e-19**
between rollouts in the BASELINE arm, i.e. to the last bits of a double. There
was no 1e-9 wheel-speed seed to remove. (The 1e-9 figure came from a single
absolute reading, not from a difference between two rollouts.)

**What actually differs at t=0, in both arms, is the IMU** — `imu_ax`, `imu_ay`,
`imu_gz` differ by **0.01-0.04**, which is twelve to fifteen orders of magnitude
above every other column (pose 1e-12, quaternion 1e-13, wheels 1e-17). That is
not physics: the IMU arrives as an async ROS message and `_read_state` takes
whatever the latest one is, so which physics tick it was sampled on depends on
ROS timing. It is the **stale-pose readout bug of 2026-08-02 again, in the IMU
channel** — and the pose channel got a `_wait_pose_advance` gate that the IMU
never got.

This matters unevenly, so do not over-read it: TVLQR does not consume the IMU,
so it cannot be *this* that moves TVLQR's commands. But `use_imu` is in the RL
observation layout, so **every RL measurement ever taken has had a
timing-dependent 0.04 jitter injected straight into the policy input.** That is
a live candidate for RL's unexplained measurement noise, which the handover
notes has never been measured.

World reset did buy about three decades of initial-state agreement (pose
mismatch 1e-9 → 1e-12) and the two comparable rollouts diverged visibly later
(xy separation reaching 1e-3 m at step 111 vs step 83; final separation 0.39 m
vs 3.41 m). **Three decades is not enough** — chaotic amplification at the turn
reversal spends them in ~30 steps. Bit-identity is the only thing that would
have worked, and a world reset does not deliver it.

**`reset_world=True` DESTROYS THE ROBOT — do not use it.** A gz
`WorldControl.reset.all` drops every entity spawned at runtime, and the
`scout_mini` is spawned at runtime by the launch. So the reset deleted the robot
out from under the running sim: from the third episode on `set_pose` stopped
landing (6.92 m from target, `reset_ticks` at the full 400, `v0=+0.25`) and
rollouts 2-4 scored an identical 5.5773 — a reproducible *broken* mode, not a
measurement. Some time later the ROS side collapsed entirely, leaving an
orphaned `gz sim` with only `/rosout` and `/parameter_events` alive, and
`_wait_ready` failing on all three streams at once. Only the first two rollouts
of arm B are valid data, and the fix after using it is `just kill-sim` +
`just remote-sim`, not debugging the bridge. The flag is left off by default and
should probably be deleted; it is kept only so this note has something to point
at.

**Consequences.** Item 4 stays open but the wheel-velocity hypothesis is closed.
Single-sample ranking on the hard shapes remains illegitimate, so **repeats stay
mandatory and parallel sims (queue item 7) are now unblocked and top of the
queue** — that was the handover's stated "if it does not work" branch. Do not
re-run the tuner before that lands.

Next lead, cheap and worth doing before anything expensive: gate the IMU read
the way the pose read is gated, then re-run this probe. It will not make
rollouts bit-identical (the pose/quaternion mismatch at 1e-12 survives), but it
removes the one initial-state difference that is 12 orders of magnitude larger
than the rest, and it is the only known contaminant of the RL observation.


### The run-to-run variance was an 8 mm reset error amplified by patch edges (solved 2026-08-02)

The blocking mystery — TVLQR scoring 0.22 m and 1.55 m on identical inputs — is
resolved, and it was **not** a bug in the stepping. Evidence, in the order it
landed (`tuning/variance_probe.py`, `tuning/reset_probe.py`):

- **It is not process reuse.** 10 rollouts in one process spread 6.70 m; 10 in
  ten processes spread 3.86 m. Comparable, with **no trend against rollout
  index** (Spearman rho = -0.14). The accumulated-sim-clock-float hypothesis in
  `_wait_clock_advance` is **wrong** — delete it as a candidate.
- **The physics is perfectly deterministic.** Four of those ten rollouts
  returned 0.223 m agreeing to three decimals, the rest landing on distinct
  reproducible values (1.5, 1.7, 4.7, 6.9). Discrete modes, not smooth noise.
- **`reset()` is clean, including from motion.** `reset_probe` resets after
  idle / forward / spin / reverse: `v` and `omega` come back **exactly 0.00000**
  every time, heading to 0.00000, `lost_steps` 0. Residual velocity was never
  the problem. (`reset_ticks` does vary 1..67 because the `_set_pose` confirm
  loop is wall-clock-paced, but it is self-correcting — each retry yanks the
  body back — so it is harmless.)
- **What remained was ~8 mm of positional spread**, from the robot sliding as it
  fell from `reset_z=0.20` onto its settled height. Tightening this to **exactly
  0.000 m** (see below) changed the spread not at all — so the reset was *not*
  the cause, and the "8 mm amplified by patch edges" theory is **retracted**.
- **THE ACTUAL CAUSE: patches were silently not spawning.** With the world
  paused, `/world/<w>/create` blocks for the whole ack timeout and returns
  **False** — while creating the entity anyway, some ticks later
  (`tuning/spawn_diag.py` proves this: ack False, entity present afterwards).
  `_apply_terrain` requested a create and started driving immediately, so
  whether an episode had its patches came down to service timing. Roughly
  **4-5 of every 10 rollouts ran on BARE GROUND**, scoring ~0.2247 m —
  identical to `--no-terrain`, which is exactly the "clean mode" that made the
  data look bimodal. The rest scored 1.5-6.9 m. Two different plants, pooled.

**Fixes**, all in `GazeboBridge`:
- `_wait_entities` / `_wait_entities_gone` step the world until pose/info
  actually shows the patches present (or gone) before the rollout starts.
  pose/info is the only honest answer — the ack is useless in both directions.
- Do **not** re-issue a create for a "missing" patch: it is almost always in
  flight, and re-creating then genuinely fails because the name now exists. An
  obvious-looking retry loop made things worse before this was understood.
- `terrain_missing` records any patch that never appeared, and a rollout with
  one raises rather than being quietly recorded as a sample.
- Reset tightening (kept, though it was not the bug): `reset_z` = measured
  settled height 0.1806 m plus a re-place/re-settle loop
  (`reset_place_tol=0.001`), giving x/y/theta spread of exactly 0.00000 across
  resets from idle/forward/spin/reverse. `lost_steps` counts steps where the
  world did not advance before the wall-clock deadline (observed: 0).

**Result on floor_6_00042, 8 rollouts:** spread **0.375 m** (1.71-2.08), down
from 6.70 m, with zero failures and no bare-ground mode. Not yet perfect —
0.375 m is still above what several queued experiments want to resolve — but
the plant is now the same one every run.

**Consequences for past results.** Any single-rollout comparison made before
this is suspect, including the 2026-08-02 three-way table and — most of all —
the converged tuning run: 132 evaluations ranked on one noisy sample each, at a
noise level (metres) far above the claimed 0.487 → 0.183 m improvement. **The
tuned gains `q_cross=7.22 / r_omega=0.369` are not supported by that run** and
must be re-derived. Re-measure with the fixed reset before trusting anything.


---

## Moved from CLAUDE.md on 2026-08-24 (the gain-tuning arc, closed by job 100)

Everything below was live text in CLAUDE.md until the tuning decision closed.
It is kept verbatim, in the order it was written, because several of these are
multi-thousand-rollout measurements on a plant that will not exist forever.
The stubs that replaced them are in CLAUDE.md under "Settled".

### Terrain patches are inherited across processes (found 2026-08-01)

`GazeboBridge._remove_terrain` only removed patches that *its own process*
spawned, and the sim deliberately outlives any one process. So a trainer that
exited mid-episode left its patches in the world, and the next
`compare_correctors` inherited them. Two distinct failures, not one:

- a leftover `rl_ground` from a `--ground-friction` run is a low-friction slab
  under the entire trajectory — a different plant than the run intends;
- `create` on an existing name *fails*, so an inherited `rl_patch_0` also
  displaces the patch this run meant to spawn.

Fixed: the bridge now sweeps `rl_ground` and `rl_patch_0..7` by name at
construction. At most 4 patches ever exist (`n_range=(1,3)` at every call site,
plus `rl_ground`); the rest is headroom.

**The 2026-07-30 three-way comparison table is not trustworthy** — it was
measured with a trainer's patches almost certainly still in the world. Same
correctors, same trajectories, clean world, 2026-08-01 (max|e_cross| / final_err):

| trajectory | shape | identity | tvlqr | 2026-07-30 identity |
| --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | 0.28 / 0.48 | 0.01 / 0.02 | 0.62 / 4.88 |
| floor_6_00042 | S-CURVE | 0.22 / 0.19 | 1.55 / 1.08 | 6.83 / 6.86 |
| floor_6_00023 | CORNER | 1.21 / 3.45 | 0.23 / 0.05 | 19.45 / 17.18 |

On a clean world **TVLQR loses to identity on the S-curve**, which the
contaminated table hid. "TVLQR beats identity on every shape" is retracted.

### TVLQR gain tuning (built 2026-08-01)

`agx_planning/tuning/` searches `(q_cross, r_omega)` by Nelder-Mead against real
Gazebo rollouts. `just tune-tvlqr` (detached, ~75 s per evaluation, ~70 min for
60), then `just fetch-tune && just plot-tune`.

- `simplex.py`, `objective.py`, `cache.py` are **pure and unit-tested** — no ROS,
  no Gazebo, no torch, same rule as the RL pure modules. 25 tests, and they are
  aimed at one thing: proving the search *minimizes*. A tuner that maximizes
  produces an identical-looking log and hands back the worst gains it found.
- **`--max-evals` bounds candidate GAIN PAIRS, never rollout length.** Every
  evaluation drives every selected trajectory start-to-goal.
- **The trajectory set is fixed, never sampled**, so candidates are always
  compared on identical work. A rollout that *fails* makes the whole evaluation
  invalid (`inf`), never a mean over the survivors: the trajectories differ
  hugely in difficulty, so averaging whatever finished rewards gains that crash
  the hard rollouts.
- **Resumable at zero cost.** Nelder-Mead is deterministic given its objective,
  so a resumed run replays the JSONL cache to reconstruct the search and
  re-measures nothing. The cache is keyed on the trajectory list + seed and
  refuses to resume onto a different problem.
- **`--max-evals 0` (the default) runs to convergence**, since the search has
  nights available and a converged answer beats a predictable finish time.
  Three guards make unbounded mode safe: a 15-minute per-evaluation timeout (a
  healthy one is ~75 s), an abort after 5 consecutive failures, and —
- **failures are never cached.** This one was learned the hard way: killing the
  tuner mid-evaluation invalidated the bridge's rclpy context, after which every
  rollout failed in 2 ms. 56 bogus `inf` evaluations were written in three
  seconds, and had they been memoized, every future resume would have replayed
  them as real measurements. `inf` means "the sim broke", almost never "these
  gains are bad". Failed records are still written (with `_failed: true`) for
  diagnosis, but never returned from the cache.
- Search runs in **log10** of both gains: a step is a ratio, and no move can
  propose a negative gain.
- Every evaluation records its **per-trajectory** errors, not just the aggregate,
  so the landscape can be re-analysed per shape without re-driving anything.

Baseline at the current `q_cross=10 / r_omega=0.25`: **0.487 m** mean
max|e_cross| over the three-trajectory set (0.243 straight / 0.224 S-curve /
0.993 corner), measured 2026-08-01 by the tuner itself.

**First converged run (2026-08-02, 132 evaluations, ~2.3 h):**
`q_cross=7.22`, `r_omega=0.369` → **0.183 m**, from 0.487 m.
`figures/tvlqr_tune_landscape.png`, history in `tune_data/tvlqr_tune.jsonl`.

**Do not adopt those gains on this evidence alone.** The per-trajectory panel
shows `floor_6_00042` swinging between ~0.2 m and 7 m *throughout* the run at
near-identical gains — the run-to-run variance is much larger than the claimed
improvement. Taking the minimum over 132 noisy draws selects partly for a lucky
draw (winner's curse), so 0.183 m is biased low and the true value at those
gains is likely worse. What the run does support is that the *region* around
q≈7, r≈0.37 is better than the default, since the simplex clustered there.

Before adopting: re-measure the tuned gains and the defaults ~5x each and
compare distributions, not single numbers (~12 min). More generally, until the
variance in the "ideas queued" list is understood, **any tuning result needs
repeat measurements**, and a search that ranks candidates on one sample each is
resolving noise as often as signal.

**Unexplained, flagged not fixed:** that same run scores TVLQR at 0.224 m on
floor_6_00042 where `just compare` scored 1.549 m an hour earlier — same gains,
same seed, same code path. It is the offline-mode variance again but far larger
than the identity-leg spread. The tuner is *internally* consistent (one process,
one code path, fixed seed), which is what the search needs, but **do not compare
a tuned number against a `compare` number** until this is understood.

### Baseline on the repaired plant (2026-08-07) — the current reference

**This supersedes the 2026-08-03 table below for every purpose except history.**
That one was measured with the wheel's `mu2` at 0.7, i.e. on a robot that lost
steering on its own test terrain (see "The wheel fix"). Five repeats of the full
seven-shape set, identity and TVLQR, terrain on, `just compare` (~82 s per
repeat). max|e_cross| in metres, mean ± sd over 5:

| trajectory | shape | identity | tvlqr | verdict |
| --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | 0.132 ± 0.000 | **0.086 ± 0.028** | tvlqr, marginal |
| floor_6_00023 | CORNER | 0.721 ± 0.196 | **0.226 ± 0.000** | tvlqr |
| floor_6_00018 | S | **0.839 ± 0.031** | 2.128 ± 0.003 | **identity** |
| floor_6_00047 | ZIGZAG | 7.063 ± 0.712 | **2.186 ± 0.276** | tvlqr |
| floor_6_00056 | TIGHT V | 0.418 ± 0.014 | **0.253 ± 0.026** | tvlqr |
| floor_6_00031 | U-TURN | 1.666 ± 0.771 | 1.411 ± 0.534 | **tie**, overlapping |
| floor_6_00025 | LOOP | 4.053 ± 0.010 | **1.600 ± 0.035** | tvlqr |
| **mean** | | **2.127** | **1.127** | |

**TVLQR halves worst-case deviation** (2.13 → 1.13 m), winning 5 shapes, tying 1,
losing 1. Open loop is what moved most in the re-baseline (3.53 → 2.13): a plant
that can steer makes the nominal plan far more followable unaided, so the easy
shapes stopped needing rescue (tight V 3.01 → 0.42, U-turn 5.53 → 1.67). TVLQR
itself barely moved (1.20 → 1.13), which is the expected signature.

**The noise REDISTRIBUTED, it did not shrink.** The old plant's worst trajectories
were the straight (sd 0.501) and the tight V (0.582); both are now effectively
deterministic (0.000 / 0.014), while the zigzag (0.712) and U-turn (0.771) became
the noisy ones. Coherent: when everything slides, everything is noisy; now the
chaos is confined to the reversal-heavy shapes, where contact modes switch.
**Consequence: 5 of 7 trajectories now support single-sample ranking, but the
zigzag and U-turn do not.** Keep mean-of-3 as the tuning default.

**"TVLQR oscillates on S-curves" is REINSTATED**, and the 2026-08-03 retraction of
it is itself retracted. TVLQR loses the genuine S by 2.5x with sd **0.003** — as
reproducible as anything measured here. The retraction came from the old plant,
where TVLQR scored 1.06 against identity's 4.22; that was open loop failing
everywhere, masking a real TVLQR weakness. The original claim had the phenomenon
right and the evidence wrong.

**The U-turn is a TIE, not a TVLQR win.** The distributions overlap completely.
Any earlier U-turn claim from single samples was reading noise.

RL was deliberately not re-measured: the existing policy is a 4-wheel *residual*
trained on the old plant with an IMU-bearing observation layout, so its number
here would describe neither this plant nor the re-planner architecture that
replaced it (see handover).

### The tuned point validates (2026-08-12)

Independent mean-of-5 at each gain pair, tuner code path, fresh caches, clean
sim. `tune_data/validate_20260812_{tuned,default}.jsonl`. The validation is run
as `tune_tvlqr --max-evals 1 --repeats 5 --q-cross Q --r-omega R`: BO seeds its
design with `x0`, so a 1-evaluation run is exactly a clean measurement at a
chosen point, in the same code path that produced the number being checked.

| | tuning run (mean-of-3) | validation (mean-of-5) |
| --- | --- | --- |
| default `q=10 / r=0.25` | 1.0417 | **1.0037** |
| tuned `q=0.276 / r=2.618` | 0.6144 | **0.6212** |

**This is the first tuning result on this project that survives an independent
re-measurement.** Both earlier ones (0.183 m, 0.9412 m) were minima of noisy
draws and evaporated on inspection.

### The U-turn's bad mode is seeded by ~1e-6 m of settle height (2026-08-13)

30 traced rollouts of `floor_6_00031` at fixed gains, 14 tuned / 16 default
(`uturn_traces/`, `uturn_{tuned,default}.jsonl`). The reset is **not** the
variable, which is what the probe was built to check:

| | tuned | default |
| --- | --- | --- |
| n | 14 | 16 |
| mean / sd | 1.423 / 0.346 | 1.270 / 0.389 |
| bad-mode rollouts | 1 (2.572) | 1 (2.591) |
| **start-pose spread (x, y, θ)** | **3e-11, 5e-11, 9e-15** | **1e-11, 1e-11, 6e-14** |
| `reset_ticks` / `lost_steps` | 20 / 0 | 20 / 0 |

**Start poses agree to ~1e-11 m and outcomes still span 1.5 m.** So the
bimodality is not initial-position spread, and the reset work of 2026-08-02 is
not implicated. `reset_ticks` is a constant 20 and no steps were lost.

`trace_diff` on a good (1.255) vs the bad (2.572) rollout puts the seed exactly:

- **at reset, before any command: `z` differs by 4e-6 m**, plus IMU by ~0.017;
- `y` moves at **step 1**, the commands only at **step 2**.

State first, commands second — so by `trace_diff`'s own reading this is physics,
not our controller. And the 4e-6 m is the **residual of the reset settle loop**,
which converges to ~1e-6 (`reset_settle_z_tol`). The IMU difference is a red
herring here: TVLQR does not consume the IMU, so it cannot be what moved the
commands.

**Reading:** the U-turn's bad mode is chaotic amplification of a ~1e-6 m
*vertical* settle residual, through the reversal mechanism already documented.
Supporting evidence that it is a property of the plant and not the corrector: the
bad-mode value is **2.572 vs 2.591** in the two arms — the same attractor at gains
differing by 36x — and the frequency is ~7% here against the soak's 10.3%/12.4%.

**Do not chase this with a tighter `z` tolerance.** The 2026-08-04 world-reset
experiment bought three decades of initial-state agreement and chaotic
amplification spent them in ~30 steps. A tolerance is the wrong instrument
against an exponential. The actionable consequence is unchanged and already in
force: on reversal-heavy shapes, report mode frequencies over many samples rather
than means over few.

### What 4065 rollouts say: the mechanism is MODE FREQUENCY (2026-08-13)

The overnight soak (`tuning/soak.py`, `just soak`) accumulated **4108 rollouts,
43 failed (1.0%), 4065 usable** across 21 processes and 14 complete cycles of the
7-shape set at both gain points — roughly **290 samples per shape per arm**,
against the n=3-5 every earlier claim rested on. Raw rows in `soak_data/`.

**The headline is now settled beyond argument.** Mean over the 7 shapes, per
complete cycle, n=14 cycles per arm:

| gains | mean | sd | min | max |
| --- | --- | --- | --- | --- |
| tuned `q=0.276 / r=2.618` | **0.6686** | 0.0882 | 0.618 | 0.982 |
| default `q=10 / r=0.25` | **1.1273** | 0.1108 | 0.897 | 1.334 |

The distributions do not overlap at all (worst tuned cycle 0.982 < best default
cycle 0.897 is *nearly* true; the single exception is one tuned cycle). This is
the fourth independent measurement of the tuned point (0.6144 / 0.6212 / 0.6429 /
**0.6686**) and the third of the default (1.0417 / 1.0037 / **1.1273**).
**The gains are now ADOPTED in `TVLQRConfig`** — `q_cross` 10.0 → 0.276,
`r_omega` 0.25 → 2.618. Note `r_omega` moving *up* reverses the reasoning in its
own docstring comment ("angular correction is cheaper"); the comment is left in
place with the correction beside it, because the argument was sound and the
measurement disagreed anyway.

**The important finding is not the mean, it is HOW the win happens.** Per shape,
pooled (mean ± sd over ~290):

| shape | tuned | default | character |
| --- | --- | --- | --- |
| straight | 0.054 ± 0.001 | 0.078 ± 0.054 | level |
| corner | 0.311 ± 0.009 | **0.226 ± 0.000** | level, default wins |
| S | 1.617 ± 0.088 | 2.131 ± 0.016 | level |
| **zigzag** | **0.503 ± 0.352** | **2.009 ± 0.533** | **mode frequency** |
| tight V | 0.286 ± 0.001 | 0.357 ± 0.219 | mode frequency |
| **U-turn** | 1.502 ± 0.520 | **1.316 ± 0.470** | **modes, unchanged** |
| loop | 0.459 ± 0.055 | 1.678 ± 0.090 | level |

Four shapes are **unimodal and near-deterministic in both arms** (sd ≤ 0.09,
straight/corner/S/loop): those are plain level shifts, three won by the tuned
gains and one — the corner — genuinely won by the default, by 0.085 m at sd
0.000. Nothing there is noise.

The other three are **bimodal, and the gains move the FREQUENCY of the bad mode,
not its depth**:

| shape | bad mode at | tuned %bad | default %bad |
| --- | --- | --- | --- |
| zigzag | ~2.2-2.5 m | **2.6%** | **88.8%** |
| tight V | ~0.78 m | 0.0% | 20.4% |
| U-turn | ~2.5-2.9 m | 10.3% | 12.4% |

**The zigzag flip is 88.8% → 2.6%, and that single number is most of the
result.** It also explains why the zigzag looked "noisy" (sd 0.215-0.533) in every
earlier table: it was never noise, it was a ~90/10 mixture of two reproducible
outcomes being sampled 3 times.

**RETRACT: "the U-turn is bistable and the tuned gains land it in the good
mode."** It is bistable — but the bad-mode rate is **10.3% vs 12.4%**, i.e.
statistically indistinguishable, so *the gains do not control it at all*
(**itself corrected the same evening — see "The `q_cross` ladder" below; the
gains control it sharply, but non-monotonically, so two points could not see
it**). This
also corrects the 2026-08-12 grid figure of "33% good / 67% bad": that pooled 45
rollouts across many gain pairs, and at neither validated point is the bad mode
anywhere near a majority. The U-turn's mode is driven by something else, and it
is the one open mechanism left. (Consistent with the U-turn contributing -3% to
the improvement: the tuned arm is slightly *worse* there, 1.502 vs 1.316.)

**Methodological point worth keeping.** Every "run-to-run variance" number in
this file above is a mixture width, not a measurement error, on exactly the
shapes that turn out to be bimodal. The right estimator for a bimodal metric is
not the mean of 3 — it is the mode frequency, which needs ~100 samples and was
unaffordable until a rollout cost 5 s. Mean-of-3 remains fine for *searching*
(it is what found these gains) but a per-shape *claim* now wants soak-scale n.

**The 1.0% failure rate is all `terrain patches failed to spawn`** (43/4108,
spread evenly over all 7 shapes), correctly invalidating the rollout rather than
being recorded as a sample. That is the 2026-08-01 guard working; at soak scale
it is now measurable, and it is small enough not to bias anything.

- **`reset_world=True` DESTROYS THE ROBOT — never use it.** A gz `WorldControl.reset.all` deletes runtime-spawned entities, including the `scout_mini`. Recovery is `just kill-sim` + `just remote-sim`, not debugging.

### The `q_cross` ladder: a zigzag threshold and a U-turn notch (2026-08-13)

1047 rollouts, the three mode-bearing shapes × six gain points, `r_omega` held at
2.618 across the ladder so `q`'s effect is separable (`soak_data/soak_20260813_ladder.jsonl`,
figure `figures/2026-08-13/03_ladder_modes.png`). n≈58 per cell. **They are two
different phenomena, and only one of them is a mode frequency.**

| shape | q=0.1 | q=0.276 | q=0.6 | q=1.5 | q=10 | q=10, r=0.25 |
| --- | --- | --- | --- | --- | --- | --- |
| zigzag %bad (>1.5 m) | 1.7 | 1.7 | 0.0 | 0.0 | **89.7** | 86.0 |
| U-turn %bad (>2.0 m) | **100** | **15** | **100** | 82.8 | 96.6 | 14.0 |
| tight V %bad (>0.6 m) | 0 | 0 | 0 | 0 | 0 | 10.5 |

**The zigzag is a THRESHOLD in `q_cross`, and it is where most of the tuned
gains' win comes from.** Flat at 0-2% across a 15× range of `q`, then a cliff
between 1.5 and 10. `r_omega` is irrelevant to it (86.0 vs 89.7 at `q=10`). So
the adopted point sits inside a *wide plateau*, not on an edge — the robust half
of the result.

**The U-turn is a NARROW NOTCH, and calling it bimodal was wrong.** At `q=0.1`
all 59 rollouts fall in 2.634–2.665; at `q=0.6`, all 56 in 2.701–2.709. Those are
tight, deterministic, **unimodal** distributions that happen to be bad — not a
mixture whose frequency shifted. Only `q=0.276` and the old default drop to ~1.4,
and they are isolated: their immediate neighbours in `q` are uniformly bad.
**This retracts the same morning's "the gains do not control the U-turn at
all"** — they control it sharply; two sample points could not see a notch.
Whether 0.276 has a usable basin or is a spike was the open question; it is
answered below.

**The tight V's mode was an `r_omega` effect, not a `q` one** — 0% bad at every
rung with `r=2.618`, 10.5% at the old `r=0.25`. Small either way.

**`max|e_cross|` and `final_err` rank the ladder DIFFERENTLY, and that is not
noise.** On the U-turn, `q=1.5` is the *worst* rung by max|e_cross| (2.150) and
the *best* at arriving (mean `final_err` 0.255, **0%** of 58 rollouts ending more
than 0.5 m out), while the adopted `q=0.276` scores 1.605 and leaves 55% of
rollouts short. A metric that disagrees with "did it get there" needs justifying,
which is what the `J` work below is for.

### Scoring in `J`: the objective change does not reverse the conclusion (2026-08-13)

`tuning/epsilon.py` (pure, 23 tests) holds the tracking functional; `tools/score_epsilon.py`
is the offline driver that scores recorded per-step traces with it. First real
measurement: 7 shapes × {tuned, default} × 5 repeats **with traces**, so every
rollout is scored both ways (`jtraces/`, `epsilon_data/jsweep.jsonl`, figure
`figures/2026-08-13/04_epsilon_vs_cross.png`). Mean of 5:

| shape | max\|e_cross\| tuned / default | J tuned / default | metrics agree? |
| --- | --- | --- | --- |
| straight | 0.054 / 0.074 | 0.22 / 1.84 | yes |
| corner | 0.309 / **0.226** | **1.50** / 1.92 | **no** |
| S | 1.582 / 2.129 | 21.8 / 84.8 | yes |
| zigzag | 0.410 / 1.587 | 13.2 / 62.3 | yes |
| tight V | 0.286 / **0.256** | **5.23** / 12.2 | **no** |
| U-turn | 1.316 / **1.180** | **29.9** / 35.7 | **no** |
| loop | 0.423 / 1.714 | 21.5 / 66.6 | yes |

**In `max|e_cross|` the tuned gains win 4 and lose 3; in `J` they win all seven**,
by 1.3× to 8.3×. Every one of the three losses is a shape where the tuned arm
concedes a little peak deviation and buys back much more in accumulated error,
correction effort and terminal miss. So `J` is not a different answer — it is a
**cleaner version of the same answer**, and it is the quantity SVCM is stated in.

Two things to hold onto:

- **`J` is an upper bound on `epsilon`, never `epsilon` itself** (`J* > 0` under
  slip and is unknown). The module docstring says so; say it in the write-up too.
- **The correction, not the total command, is what `R` charges.** The nominal
  command is what the planner already paid for; charging it again would score
  every corrector for the plan's cost. `score_epsilon.py` recovers it as
  applied-minus-nominal in wheel space.

**Scoring needs the TRACK, so it must be captured at rollout time.** The ~4000
soak rollouts cannot be rescored — `variance_probe.drive` reduces each to
scalars. Any future soak whose numbers should be readable in `J` must be run with
`--trace-dir`. **`soak.py` now HAS `--trace-dir` (plus `--trace-every N` to keep
an unbounded soak from filling the disk at ~130 kB a rollout), so this is wired
rather than merely known.** Untraced rows carry no `trace` field and are skipped
by the scorer rather than silently scored against someone else's file.

### The U-turn basin has walls, and `q=0.276` stays adopted (2026-08-13 evening)

Two runs closed the ladder question. **The adopted gains survive**, and the
reason is more interesting than "they were right".

**1. The sub-ladder: the basin is real, and 0.276 sits on its LEFT EDGE.**
5906 usable rollouts, `floor_6_00031` only, n≈985 per rung, `r_omega=2.618`
throughout (`soak_data/soak_20260813_uturn_subladder.jsonl`):

| `q_cross` | mean | sd | %bad (>2.0 m) | mean `final_err` | %miss (>0.5 m) |
| --- | --- | --- | --- | --- | --- |
| 0.200 | 2.647 | 0.120 | 99.2 | 1.153 | 99.4 |
| **0.276 (adopted)** | 1.637 | **0.594** | **17.9** | 0.643 | 46.6 |
| 0.320 | **1.494** | 0.154 | 1.8 | **0.347** | **9.8** |
| 0.400 | 1.546 | 0.090 | **0.7** | 0.363 | 25.0 |
| 0.500 | 2.685 | 0.029 | 99.9 | 0.619 | 100.0 |
| 0.600 | 2.707 | 0.002 | 100.0 | 0.530 | 96.0 |

So the notch is a **basin roughly `[0.276, 0.4]` with near-vertical walls** —
0.2 and 0.5 are ~100% bad and nearly deterministic (sd 0.12 / 0.03), not
mixtures. **The adopted point is the noisiest rung in the entire ladder** (sd
0.594, 17.9% bad) precisely because it straddles a wall: the interior is
near-deterministic (0.7-1.8% bad). That is a better explanation of the U-turn's
"bimodality" than the shape itself — it was never intrinsic, it was an artifact
of measuring at a gain sitting on the edge.

**Scope correction (2026-08-15): this basin is `floor_6_00031`'s, not the
U-turn's.** See "The U-turn basin does not generalise" below. Everything in this
section is still true *of that plan* — 5906 rollouts do not stop being 5906
rollouts — but it must never again be written as a statement about U-turns, and
it is not a reason to prefer any `q` on a plan we have not driven.

**2. The seven-shape check: `q=0.32` is NOT free, so do not re-adopt it.**
7 shapes × `q` ∈ {0.276, 0.32, 0.40} at `r=2.618`, 5 repeats, **traced**, so it
is scored in both currencies (`gaincheck/`, `epsilon_data/gaincheck_J.jsonl`):

| shape | `J` 0.276 / 0.32 / 0.40 | max\|e_cross\| 0.276 / 0.32 / 0.40 |
| --- | --- | --- |
| straight | 0.22 / **0.21** / 0.84 | 0.055 / **0.046** / 0.127 |
| S | **22.3** / 30.0 / 41.8 | **1.605** / 1.885 / 2.522 |
| corner | 1.49 / 1.45 / **1.44** | 0.310 / 0.306 / **0.303** |
| loop | 21.5 / **20.8** / 23.1 | 0.422 / **0.383** / 0.558 |
| U-turn | 41.8 / 35.6 / **33.6** | 1.623 / **1.417** / 1.540 |
| zigzag | 13.5 / 16.6 / **12.1** | **0.445** / 0.711 / 0.464 |
| tight V | **5.23** / 5.30 / 5.94 | 0.286 / **0.285** / 0.297 |
| **MEAN** | **15.15** / 15.71 / 16.98 | **0.678** / 0.719 / 0.830 |

**`q=0.32` buys the U-turn and pays for it on the S and the zigzag**, and the
aggregate says the trade is not worth making. The U-turn gain is real (`J` 41.8
→ 35.6, `final_err` 0.641 → 0.252) — it is simply smaller than what it costs
elsewhere. **`TVLQRConfig` is unchanged: `q_cross=0.276`, `r_omega=2.618`.**

**`J` and `max|e_cross|` AGREE here, for the first time.** Both rank
0.276 < 0.32 < 0.40 on the aggregate. That is worth noting because the same-day
`J` section above found them disagreeing on 3 of 7 shapes: the metrics diverge
on *per-shape* verdicts, not necessarily on aggregate ranking. Do not
generalise either way from one case.

**The methodological point, which is the durable part:** a per-shape optimum
(`q=0.32` on the U-turn, at n≈985 and beyond argument) **did not survive the
seven-shape aggregate**. A gain chosen on one trajectory is a gain overfitted to
one trajectory, however many samples back it. The sub-ladder was still worth the
night — it explains *why* 0.276 looks noisy on the U-turn — but the decision was
always the aggregate's to make.

### The `r_omega` ladder: `r` is flat ABOVE 1.0, and matters below it (2026-08-14, corrected 2026-08-15)

1035 usable rollouts (1050 run, 15 lost to the patch-spawn guard), 7 shapes ×
`r_omega` ∈ {1.0, 1.8, 2.618, 3.5, 5.0}, `q_cross` held at the adopted 0.276 so
`r` is separable — the mirror of the `q` ladder. `soak_data/soak_r_ladder.jsonl`,
figures `figures/2026-08-14/`. n≈30 per cell. mean ± sd of max|e_cross|:

| shape | r=1.0 | r=1.8 | r=2.618 | r=3.5 | r=5.0 |
| --- | --- | --- | --- | --- | --- |
| straight | 0.053±.003 | 0.048±.000 | 0.054±.001 | 0.054±.002 | 0.066±.008 |
| corner | 0.304±.002 | 0.327±.000 | 0.310±.001 | 0.313±.005 | 0.312±.001 |
| S | 1.704±.583 | 1.576±.095 | 1.589±.084 | **2.550±.001** | 1.696±.041 |
| zigzag | 0.409±.088 | 0.471±.207 | 0.460±.129 | 0.446±.107 | 0.409±.020 |
| tight V | 0.283±.020 | 0.303±.070 | 0.286±.000 | 0.287±.000 | 0.289±.000 |
| U-turn | 2.646±.006 | 2.684±.005 | **1.663±.628** | 2.591±.006 | 2.644±.024 |
| loop | 0.363±.021 | 0.448±.092 | 0.436±.051 | 0.465±.020 | 0.439±.104 |
| **mean** | 0.823 | 0.837 | **0.685** | 0.958 | 0.837 |
| **mean, no U-turn** | **0.519** | 0.529 | 0.522 | 0.686 | 0.535 |

**`r_omega` does nothing on five of seven shapes across this 5× range.** Straight
spans 0.048–0.066, corner 0.304–0.327, tight V 0.283–0.303, zigzag 0.409–0.471,
loop 0.363–0.465 — every one of those is inside its own cell-to-cell spread.
Drop the U-turn and the aggregate is **flat from r=1.0 to r=2.618** (0.519 vs
0.522), i.e. the adopted value earns nothing *within this range*.

Within this range `r=2.618` is an isolated notch and it is one plan's: the U-turn
is 100% bad (>2.0 m, near-deterministic: sd 0.005–0.024) at 1.0, 1.8, 3.5 and 5.0,
and 20% bad only at 2.618. The S's spike at 3.5 is the only other place a
non-U-turn shape reacts to `r` at all.

**But this ladder's floor was r=1.0, and the value it replaced is r=0.25 — below
it. The low half (2026-08-15, next section) shows `r` DOES matter down there, so
the conclusion drawn here that "the tuning result is essentially a `q_cross`
result" is RETRACTED.** What survives is narrower and still useful: `r` is a free
parameter **above ~1.0**, so the specific value 2.618 is not load-bearing outside
`floor_6_00031` — but the move *off* 0.25 is.

**Second isolated feature: the S has a deterministic bad spike at r=3.5** —
2.548–2.552 over 30 rollouts, sd 0.001, 100% of them, against ~1.6 at every other
rung. Not a mixture, not noise; unexplained, and the only place in this ladder
where a non-U-turn shape cares about `r` at all.

**Not scoreable in `J`** — this run's traces are unusable, see below.

### The low half of the `r_omega` ladder: the move off 0.25 DID pay (2026-08-15)

840 rollouts, 7 shapes × `r_omega` ∈ {0.25, 0.5, 1.0, 2.618}, `q_cross` held at
0.276, `r=2.618` carried in the same process as the overlap arm so the two
ladders join without assuming cross-run comparability.
`soak_data/soak_r_ladder_low.jsonl`, figure `figures/2026-08-15/01_r_ladder_low.png`.
n≈29 per cell. mean ± sd of max|e_cross|:

| shape | r=0.25 | r=0.5 | r=1.0 | r=2.618 |
| --- | --- | --- | --- | --- |
| straight | 0.067±.014 | 0.048±.026 | 0.053±.003 | 0.054±.001 |
| corner | 0.304±.003 | 0.306±.001 | 0.306±.004 | 0.311±.007 |
| S | 1.982±1.110 | **2.566±.014** | 1.577±.249 | 1.584±.091 |
| **zigzag** | **0.803±.010** | **0.812±.039** | **0.387±.029** | 0.486±.145 |
| tight V | 0.271±.000 | 0.275±.000 | 0.279±.000 | 0.289±.021 |
| U-turn | 2.381±1.363 | 2.619±.006 | 2.604±.199 | **1.714±.740** |
| loop | 0.385±.024 | 0.436±.005 | 0.369±.032 | 0.438±.042 |
| **mean** | 0.885 | 1.009 | 0.796 | **0.697** |
| **mean, no U-turn** | 0.635 | 0.740 | **0.495** | 0.527 |

**The zigzag HALVES between r=0.5 and r=1.0** (0.812 → 0.387), at sd 0.010–0.039
on both sides — as reproducible as anything measured here, and nothing to do with
the U-turn. That alone moves the six-shape aggregate from 0.635 to 0.495. So the
previous ladder's "flat" verdict was **an artifact of where its floor was**: `r`
has a threshold somewhere in (0.5, 1.0) and is genuinely flat above it.

**`final_err` is monotone in `r` across the whole range** — 0.661 / 0.440 / 0.379
/ **0.343** m averaged over the seven shapes, improving at every rung including
the ones where `max|e_cross|` is flat. The U-turn (2.102 → 0.655) and the S
(1.291 → 0.556) carry most of it, but no shape gets worse. A metric that
improves monotonically where the other is flat is the second time these two have
ranked a ladder differently (see the `q` ladder), and it is again `final_err`
that agrees with "did it arrive".

**Consequence: the joint move stands as a joint move**, and `r_omega=2.618`
remains adopted. The honest description is two-part, and the halves have
different strengths: **the move off `r=0.25` is real and general** (zigzag,
`final_err`, six-shape aggregate), while **the specific value 2.618 versus
anything in [1.0, 5.0] is defensible only through `floor_6_00031`** — and after
2026-08-15's generality run, not even as a shape claim. Anything ≥ 1.0 would
serve; 2.618 is kept because it is measured, not because it is special.

### The U-turn basin does not generalise, and the labels do not survive looking (2026-08-15)

1200 rollouts, `q_cross` ∈ {0.2, 0.276, 0.4, 0.5} at `r=2.618`, on four U-turn
plans from the constructed v2 library **plus `floor_6_00031` as an in-run
control**, n≈59 per cell. `soak_data/soak_uturn_generality.jsonl`, figures
`figures/2026-08-15/02_*.png` and `03_*.png`. mean ± sd, %bad = >2.0 m:

| plan | q=0.2 | q=0.276 | q=0.4 | q=0.5 |
| --- | --- | --- | --- | --- |
| **floor_6_00031** (control) | 2.612 / 97% | 1.613 / 18% | **1.540 / 0%** | 2.686 / 100% |
| floor_6_v2_00003 | 0.325 / 0% | 0.290 / 0% | 0.441 / 0% | 0.355 / 0% |
| floor_6_v2_00004 | 0.289 / 0% | 0.290 / 0% | 0.287 / 0% | 0.286 / 0% |
| floor_6_v2_00008 | 2.497 / 100% | 2.616 / 100% | 2.305 / 100% | 2.333 / 100% |
| floor_6_v2_00010 | 1.709 / 30% | 1.673 / 20% | 1.844 / 34% | 1.466 / 18% |

**Only the control has a basin.** The others are flat-and-easy (two of them, ~0.29 m
at every rung, sd 0.000–0.198), flat-and-hopeless (00008, 100% bad everywhere,
sd 0.001–0.018), or weakly bimodal with **no `q` dependence at all** (00010,
18–34% bad in no order). So `q_cross`'s near-vertical walls are a property of one
plan, and the U-turn sections above are hereby scoped to it.

**The stronger form of the refutation came from LOOKING at the plans, not from
the numbers.** Rendered (`03_uturn_plans.png`), the five are: one true hairpin
(00010), one rectangular U of two same-sign 90° corners (00004), one **bent line
that is not a U-turn at all** (00003) — and 00008, which is a **near-duplicate of
the control**: same corridor, same three-sided route, ~1 m of difference at the
start. That is what closes the obvious objection ("you picked plans that were not
really U-turns"): the plan most geometrically similar to `floor_6_00031` is the
one *most* clearly lacking the basin. It is the route's exact plan, not its shape
and not even its corridor.

**The `shape` label is a ranking aid and nothing more.** All five score
`total_abs_turn` 7.0–9.3 rad — the descriptor cannot separate a 180° hairpin from
two same-sign 90° corners, and it still counts the leading in-place pivot. This
is the **second** automatic shape labeller to mislead here (the first:
`classify_plans.py` calling 58 of 100 plans CORNER). **Render the plans before
making any per-shape claim**; using a label to *stratify* a sample is fine, since
that only needs correlation with geometry, not a correct name.

### The broad gain check: the move off the default holds, the adopted VALUE does not (2026-08-15)

**The generality question is now answered on plans we did not choose.** 40 plans
from the constructed v2 library (`tools/select_broad_eval.py`, stratified on
label and length, 8.7–36.0 m, **none of them among the seven**), six gain pairs,
3 repeats, every rollout traced so both currencies are available. 720 rollouts,
9 invalidated by the patch-spawn guard (1.2%), **711 usable**.
`soak_data/soak_broad_gains.jsonl` → `epsilon_data/broad_gains_J.jsonl`.

Per-plan mean-of-3, then aggregated over the 40 (`J` geometric, per
`objective.DEFAULT_HOW`):

| gains | geo `J` | mean max\|e_cross\| | mean `final_err` | miss rate (>0.5 m) |
| --- | --- | --- | --- | --- |
| **q=0.276 r=2.618 (adopted)** | 11.82 | 0.656 | 0.378 | 20.0% |
| q=0.276 r=1.0 | 12.52 | 0.627 | 0.366 | 22.0% |
| q=0.6 r=2.618 | **11.39** | 0.646 | 0.292 | 18.1% |
| q=1.5 r=2.618 | 12.27 | 0.599 | **0.265** | **11.8%** |
| q=4 r=2.618 | 14.08 | 0.688 | 0.328 | 14.2% |
| q=10 r=0.25 (old default) | 17.59 | **0.575** | 0.314 | 15.3% |

**1. The move off the old default is CONFIRMED, and this is the strongest
version of that claim we have.** In `J` the old default is **1.49x worse** than
the adopted point and wins only **5 of 40** plans (paired sign test p<0.001), on
a plan set chosen mechanically and disjoint from the one the gains were tuned
on. Everything the seven-plan work concluded about *moving* survives.

**2. The adopted VALUE does not survive.** Paired against every other arm, `q=0.276`
is the **worst of the six in metres** — every other arm beats it at p<=0.038,
*including the old default* (0.79x, 29/40) — and it has the **worst miss rate**
of all six. In `J` it is statistically indistinguishable from everything between
q=0.276 and q=4 (p = 0.27 to 0.88). So the plateau is real and broad, and 0.276
sits on its bad edge: it buys nothing in `J` and pays in both peak deviation and
arrival.

**3. `q≈1.5, r=2.618` dominates the adopted point** — `J` tied (+3.8%, p=0.88),
max|e_cross| 0.917x (p=0.038), `final_err` 0.680x (p=0.006), miss rate 11.8% vs
20.0%. Not adopted yet: it wants one confirmatory mean-of-5 at the three
candidate points before `TVLQRConfig` moves again (see handover).

**4. Where the default actually loses is CONTROL EFFORT, and that is the SVCM
prescription showing up as a measurement.** Mean `J` split over the 40 plans:

| gains | tracking | control | terminal |
| --- | --- | --- | --- |
| q=0.276 r=2.618 | 13.18 | 2.22 | 3.11 |
| q=0.6 r=2.618 | 12.16 | **2.07** | 1.57 |
| q=1.5 r=2.618 | 13.52 | 2.43 | **1.47** |
| q=10 r=0.25 | 15.81 | **6.37** | 3.38 |

The default achieves slightly *tighter peak tracking* while spending **~3x the
control** to do it. That is exactly the trade p. 78 prescribes (a larger `R` in
low traction), and it is why the two metrics rank the arms oppositely rather
than noisily.

**5. The metres-vs-`J` disagreement REVERSED direction from the library sweep,
and the plan population is why.** The 51-plan sweep had the tuned gains gaining
9.99 m and losing 2.03 m; here they gain 3.75 m and lose **7.02 m**, including
**15 of 15 easy plans**. The v2 library is 100% turning shapes by construction,
so this set has far fewer easy plans to lose cheaply on. **Neither sweep is
"the" answer in metres** — which is itself the argument for scoring in `J`.

**Consequence for tuning: a search on the seven plans cannot resolve `q` at
all.** `J` is flat across a 15x range of `q` on independent plans, so any
seven-plan optimum inside that range is an artifact of those seven. Validate a
tuned point on the broad set before adopting it.

### A failed plan hung every client, and the ROS pipeline had never been run (2026-08-18)

Driving the fixture end to end for the first time — see handover, "The ROS 2
stack is now drivable unattended" — turned up a **real bug in
`runtime_corrector`, in the path that handles a planner failure.**

**Symptom:** publish a goal, the planner fails to solve it in under a second,
and the whole stack goes quiet with the robot stationary. No zero command, no
completion sentinel, no log line beyond the action result. Every consumer waits
for its own timeout; `random_goals` burns its full dwell, and anything without a
timeout waits forever.

**Cause:** `TrajectoryBuffer.active_traj_id` is set in `_on_chunk`, so a plan
that fails at **chunk 0** never sets it. `_on_action_result` then compares
`traj_id == self._buf.active_traj_id` against `-1`, drops the result, and
`_on_tick`'s idle guard (`active_traj_id < 0`) returns before `_finish()` — the
function that publishes the zero and the sentinel — can run.

`_finish`'s docstring already claims the sentinel fires on **any** terminal
outcome, and that was true of the case it was written for: a failure *after*
playback began. This is the other one, and it is the more common: **BVP mesh-node
exhaustion fails ~36% of fresh start/goal pairs**, the same failure rate measured
building the v2 library. Fixed in `_on_action_result`, which now stops, clears
the goal and publishes the sentinel directly when no trajectory ever arrived.

**Verified in the wild**, in a six-goal `random_goals` session on the repaired
build: goal 2 failed BVP, logged the new `Plan for traj_id=2 produced no
trajectory ... Stopping and clearing the goal`, and the driver advanced to the
next goal **0.4 s later** off the sentinel.

**Evidence, from one five-goal fixture session** (`/tmp/fixture5.log`): goals 1,
3 and 4 logged `Trajectory N finished (success=True). IDLE.`; goals 2 and 5 both
logged `BVP solve failed ... maximum number of mesh nodes is exceeded`, and
**neither produced a `finished` line at all.** Both hung their client for its
full timeout. The corrector's own status line sat frozen at the previous run's
counters (`ticks=605 rms_cross=1.2069 max_cross=7.8170 sat=38.2%`), which is
what "idle but not IDLE" looks like from outside.

**Two things this changes beyond the fix:**

- **The 36% BVP failure rate is not only a data-generation problem.** It was
  logged as a concern for the re-join re-planner's teacher; it is *also* a
  runtime failure mode on ordinary goals, today, on the real pipeline. A goal
  the planner cannot solve must degrade visibly, not silently.
- **The same goal failed once and succeeded later** from a different start pose
  (`(6.00, 3.00)`: traj 2 failed, traj 3 and 4 arrived). So the failure is a
  property of the start/goal *pair*, not of the goal — consistent with the
  library build, and worth remembering before calling a goal "unreachable".

**Method note.** This is exactly what the handover predicted would happen on the
first real run ("expect bit-rot; that is the main cost") — except the pipeline
itself was healthy and the bug was in the failure path, which no amount of
successful driving would have exposed. **A stack that works is not evidence
about what it does when a component says no.**

### The broad ladders: `q` and `r` INTERACT, and job 60's tuned point is an artifact (2026-08-18)

Jobs 70/80/90 all finished cleanly on 2026-08-15 and sat unread for ~62 h (the
queue was idle, nothing was stuck). All three ran on the 40 broad v2 plans, none
of which is among the seven. **`j_total` is now inline in every soak row** — the
online `EpsilonAccumulator` — so none of this needed trace scoring; aggregation
is geometric on `J` per `objective.DEFAULT_HOW`, arithmetic on the rest, and
arms are compared by paired sign test over the 40 plans.

**Job 70 — the `q` ladder decides on ARRIVAL, because `J` cannot decide at all.**
1200 rollouts, mean-of-5, `r=2.618` throughout. `soak_data/soak_broad_q.jsonl`:

| q (r=2.618) | geo `J` | mean max\|e_cross\| | mean `final_err` | miss rate |
| --- | --- | --- | --- | --- |
| 0.276 (adopted) | 13.19 | 0.671 | 0.379 | 20.5% |
| 0.600 | 13.08 | 0.672 | 0.327 | 21.0% |
| 1.000 | 13.90 | 0.677 | 0.346 | 13.5% |
| 1.500 | 14.57 | 0.678 | 0.287 | 14.0% |
| **2.500** | 13.52 | **0.619** | **0.236** | **10.5%** |
| 4.000 | 15.90 | 0.664 | 0.308 | 14.0% |

`J` is flat across the whole 15x range (13.1–15.9, no rung beats 0.276 at
p<0.08) — exactly what job 50 predicted. **The separation is entirely on
arrival, and `q=2.5` wins it**: vs the adopted 0.276 it is better on `final_err`
34/40 (p<0.0001) and max|e_cross| 32/40 (p=0.0002), and it beats `q=1.5` too
(`final_err` 27/40, p=0.038; max|e_cross| 31/40, p=0.0007). So job 50's
provisional pick of 1.5 was one rung short of the real optimum.

**Job 80 — `q` and `r` INTERACT, so every `r` claim in this file is scoped to
`q=0.276`.** 600 rollouts, `r ∈ {0.25, 0.5, 1.0, 2.618, 5.0}` at `q=1.5`:

| r (q=1.5) | geo `J` | max\|e_cross\| | `final_err` | miss |
| --- | --- | --- | --- | --- |
| 0.250 | 15.03 | **0.538** | **0.264** | **12.5%** |
| 0.500 | 15.52 | 0.593 | 0.285 | 13.3% |
| 1.000 | 16.03 | 0.642 | 0.339 | 15.0% |
| 2.618 | 14.77 | 0.653 | 0.312 | 15.8% |
| 5.000 | **13.74** | 0.695 | 0.304 | 17.5% |

**At `q=1.5` the r story inverts.** `r=0.25` — the value we moved off — is now
best on max|e_cross| and `final_err`, losing only on `J` (11/40, p=0.006), and
**the r=0.5→1.0 threshold that justified the move is absent**. The 2026-08-15
zigzag halving was measured at `q=0.276` and does not survive a change of `q`.
`J` and metres rank this ladder in opposite directions monotonically, which is
the control-effort trade of job 50 showing up again along `r`.

**Job 90 — job 60's `J`-tuned point is the artifact job 50 predicted; do not
adopt it.** The seven-plan `J` search returned `q=0.880, r=25.6` (posterior mean
6.94). On the broad 40, with two known points measured **in the same process**:

| gains | geo `J` | max\|e_cross\| | `final_err` | miss |
| --- | --- | --- | --- | --- |
| 0.276 / 2.618 | 12.91 | 0.702 | 0.394 | 24.0% |
| 0.880 / 25.61 (job 60) | 13.40 | 0.804 | 0.447 | 26.5% |
| 1.500 / 2.618 | 12.93 | **0.624** | **0.235** | **10.0%** |

It is the worst of the three on every axis, losing to 1.5/2.618 on `final_err`
33/40 (p<0.0001). It also **failed to beat the adopted point on its own search
set** (6.94 vs 5.98). This is the second time a seven-plan optimum evaporated on
independent plans, and it closes the question: **a seven-plan search cannot
resolve the gains, in either currency.** Do not run another one.

**`J` is the right objective and a poor discriminator.** It ranked the move off
the old default correctly and decisively (job 50, 1.49x, 5/40), and it cannot
separate anything inside the plateau — every within-plateau p is 0.08 to 0.88.
Arrival (`final_err`, miss rate) is what separates them, and it is also what the
robot is for. Read a gain decision on `final_err` and `J` together; max|e_cross|
ranks the old default best while it spends ~3x the control.

### The traced-soak path had two bugs, and neither could fail loudly (2026-08-14)

The `r_omega` ladder was the first soak run with `--trace-every`, and none of its
210 traces can be scored. Both bugs are fixed; both are worth knowing because
their failure mode is a *plausible number*, never an error.

1. **The stride aliased against the cycle length.** `soak.py` cycles gains ×
   trajectories — 5 × 7 = 35 rollouts per cycle — and `--trace-every 5` sampled
   every 5th *rollout index*. `gcd(5, 35) = 5`, so the same 7 cells were traced
   every cycle and the other 28 **never once**. Subsampling is now by **cycle**,
   which keeps coverage balanced by construction.
2. **Tracing is armed by FILE, not by rollout.** `enable_trace(path)` opened a
   new file and kept writing; nothing turned it off, so the four untraced
   rollouts after each traced one were appended to the traced one's CSV. Every
   file held ~5 rollouts from ~5 different cells. Scored, that is one 1084-row
   "track" for a 186-step plan: `max|e_cross|` came out at 29 m against the
   soak's own 0.05 m for the same rollout. `GazeboBridge.disable_trace()` now
   exists and `soak.py` calls it on every untraced rollout.

**Every `J` number already in this file is safe.** They all came from
`variance_probe`, which traces *every* rollout when `--trace-dir` is set, so it
never armed a file it did not fill. Only `soak.py --trace-every` was affected,
and only the `r_omega` ladder ever used it.

**The general lesson, which is the reusable part:** a subsample stride and a
cycle length are not independent, and a trace file is a *resource with a
lifetime*, not a flag. Both failures produce data that parses, scores, and looks
like a measurement.

### The library sweep: it generalises, but as a ROBUSTNESS TRADE (2026-08-14)

Every corrector claim to date rested on 7 hand-picked plans, so the obvious
reviewer question was whether they generalise. **They do.** 51 plans (every plan
>= 10 m, chosen by a mechanical rule rather than by us), tuned `0.276/2.618` vs
default `10/0.25`, 3 repeats, traced. 306 rollouts, 4 invalidated by the
patch-spawn guard, 302 usable. Rows in `soak_data/libsweep.jsonl`, scored into
`epsilon_data/libsweep_J.jsonl`.

| | tuned | default | tuned wins |
| --- | --- | --- | --- |
| mean `max\|e_cross\|` | **0.449** | 0.605 | 21/51 |
| median `max\|e_cross\|` | 0.255 | **0.214** | |
| mean `J` | **10.40** | 42.26 | **45/51** |
| median `J` | **6.64** | 8.14 | |
| mean `final_err` | **0.294** | 0.387 | |

**The aggregate holds in both currencies, but in metres the default wins the
MAJORITY of plans (30/51) while losing the aggregate badly.** Split by
difficulty, that resolves completely:

| bucket | n | tuned | default | tuned wins |
| --- | --- | --- | --- | --- |
| easy (default < 0.3 m) | 37 | 0.213 | **0.182** | 11/37 |
| medium (0.3-1.0) | 3 | 0.791 | **0.589** | 0/3 |
| hard (> 1.0) | 11 | **1.150** | 2.032 | **10/11** |

Totals: 9.99 m gained, 2.03 m lost, worst single regression 0.282 m. So the
tuned gains **trade ~3 cm of precision on easy plans for preventing blow-ups on
hard ones** — and `J` scores that trade as a near-sweep (45/51), because the
easy-plan "losses" in peak deviation are paid back in accumulated error, control
effort and terminal miss. `floor_6_00031` alone goes 1042.88 -> 40.73 in `J`.

**Two things to carry into the write-up.** First, the honest claim is not "TVLQR
tuning halves deviation everywhere" — it is a robustness trade that pays on hard
trajectories and is ~free on easy ones. Second, **the 7-shape set is enriched for
hard plans** (11 of 51 library plans are hard; most of our seven are), which is
why it reads 0.67 vs 1.13 where the library reads 0.449 vs 0.605. That is the
right design for a corrector test set, but it must be described as one and not
as a representative sample of the robot's work.

**`J` and `max|e_cross|` disagree on 24 of 51 plans while agreeing on the
aggregate direction.** The 2026-08-13 gaincheck found them agreeing on the
aggregate too; at n=51 the per-plan divergence is much larger than that suggested.

### Realistic friction, IMU gating, and repeats (2026-08-04)

Three changes landed together. **They re-baseline everything: no number measured
before 2026-08-04 is comparable with one measured after**, because the plant
(friction distribution) changed deliberately.

**1. The patch friction distribution is now a floor, not an ice rink.** Added
`linoleum` (mu 0.45, the actual deployment surface) and `wet_tile` (0.30) to
`PROFILES`, plus `DEFAULT_PATCH_WEIGHTS` — the samplers no longer draw
uniformly. Was: uniform over `[slippery, icy, directional_x, directional_y]`, so
**half of every patch set was at or below tyre-on-ice friction** and a quarter of
episodes ran on black ice end to end. Now ~60% at mu >= 0.30 and 10% ice, so the
hard surfaces are the exception they are in reality. `ground_friction_sampler`
is weighted too and **excludes the directional profiles** — a whole floor that
grips one axis and slides the other has no physical analogue. Rationale: gains
tuned against black ice are over-aggressive on linoleum, and the deployment
target is rubber on university linoleum and tile. Tests in
`test_terrain_weights.py`; a profile with no weight raises rather than silently
never being drawn. Still to do: a `sand` profile (the advisor wants ice *and*
sand, and Coulomb mu alone does not model granular flow).

**The world's own ground is still `mu=1.0`** (concrete-like) in
`rl_corrector.world`. If deployment is linoleum everywhere, the nominal
no-patch plant is grippier than reality and every gain inherits that bias. Not
changed yet — it is a bigger re-baseline and wants a decision.

**2. The IMU read is gated** (`_wait_imu_advance`), matching the pose gate that
has existed since 2026-08-02. Motivation is in "The wheel-velocity residual was
NOT the seed" above: the un-gated IMU was the largest difference between two
otherwise-identical rollouts, by twelve orders of magnitude, and it feeds the RL
observation. `stale_imu_steps` counts giving up, and the gate is only paid when
`use_imu` is on, so TVLQR tuning costs nothing for it. **The goal is not
determinism** — the real robot's IMU is noisy and laggy and the policy must
tolerate that. The goal is that the sim's sensor error be a knob we choose
(inject a deliberate latency/noise model in the env, matched to the measured
real IMU) rather than an artifact of VM CPU load with no real-world counterpart.
The deliberate sensor model is **not written yet**; when it is, it must live in
the bridge/env and leave the obs layout untouched, or the policy stops being
deployable.

**3. The tuner averages repeats.** `tune_tvlqr --repeats 3` (default) drives the
whole trajectory set n times per candidate and reduces per trajectory via
`objective.reduce_repeats`. Measured on the 26 real repeats in the old log:

| estimator | sd of the estimate | cost/eval |
| --- | --- | --- |
| single sample | 0.0933 | 35 s |
| median-of-3 | 0.0692 | 105 s |
| **mean-of-3** | **0.0547** | 105 s |
| mean-of-5 | 0.0413 | 175 s |

**The median LOSES, which was not the prediction.** The aggregate already
averages over 7 trajectories, of which at most two are contaminated per repeat,
so the outlier is diluted 7-fold before the estimator sees it while the median
pays its variance penalty in full. It is plain sqrt(n) averaging. Use mean-of-3
to search, mean-of-5 to validate a winner. The cache key now includes
`repeats`, `reduce` and `patch_weights`, so an old cache cannot be replayed into
a run whose numbers mean something different.

Still outstanding for the tuner: Nelder-Mead reports the **minimum observed
draw**, which is the winner's-curse machine that produced both bad results. With
sd 0.055 that bias is much smaller, but the structural fix is a noise-aware
optimizer (Bayesian optimization with a nugget term, reporting the posterior-mean
optimum). Not yet done.

**Parallel sims are NOT a prerequisite** any more, contrary to the 2026-08-03
handover: mean-of-3 is ~105 s/eval serially, so a converged run is ~2 h. They
remain the right answer for RL *training* throughput.

**RL action space, decided 2026-08-04:** keep the 4-wheel residual. The physical
Scout takes only `(v, omega)` and computes wheel efforts in firmware, so a 4-D
residual cannot deploy as-is — that is accepted as an implementation detail that
could be changed (firmware), and **TVLQR is what any real-world demo runs**.


### Moved 2026-08-24: the pre-wheel-fix friction sweep and the fix itself

Condensed in CLAUDE.md under "`slip_chi` is a function of the SURFACE"; the
full curves, both measured on plants that no longer exist, are here.

### `slip_chi` is a function of the SURFACE, and that is the live bug (2026-08-05)

Dropping the world ground from `mu=1.0` to `0.45` made the default-gain tuning
objective jump from 1.1706 m to **20.6148 m**. A no-patch diagnostic
(`/tmp/mu_diag`, bare linoleum, zero terrain) shows it is **not** a corrector
failure and **not** the patch distribution:

| trajectory | identity | tvlqr |
| --- | --- | --- |
| floor_1_00049 (straight) | 0.53 / 0.54 | 0.59 / 0.86 |
| floor_6_00023 (corner) | **27.25 / 27.95** | **30.31 / 30.92** |

Open loop diverges as badly as TVLQR, so the **nominal PMP plan is unfollowable
on a realistic floor**. The straight is fine and the corner is not, which points
straight at the yaw model.

**Measured cause (same day, `slip_ident` on the real-time world):** it is not a
mis-tuned chi, it is that **an isotropic low-friction ground deletes the tyre
model's anisotropy, which IS the steering mechanism.**

`tools/sweep_ground_mu.sh` measures chi against ground `mu`
(`sweep_data/ground_mu_chi.csv`, logs per point):

| ground `mu` | chi | yaw gain | usable arcs | spread across radii |
| --- | --- | --- | --- | --- |
| 1.0 | **1.3718** | 0.729 | 6 | 0.030 |
| 0.9 | 1.3879 | 0.721 | 6 | 0.051 |
| 0.8 | 1.4438 | 0.694 | 6 | 0.129 |
| **0.7** | **10.147** | **0.100** | 2 | 2.672 |
| 0.6 | 11.760 | 0.086 | 2 | 2.547 |
| 0.5 | 14.441 | 0.071 | 2 | 4.101 |
| 0.45 | **16.478** | 0.061 | 2 | 3.619 |

(This whole section describes the plant as it was, with the wheel's `mu2` at 0.7.
That was changed to 0.45 on 2026-08-07 and the cliff moved with it — see "The
wheel fix" below. The mechanism and the method are unchanged; only the number
where it breaks moved.)

**The cliff is at 0.7 — the wheel's own `mu2`, to the digit.** Above it only the
longitudinal channel erodes, so chi drifts up gently while the spread across
radii quadruples (a single `slip_chi` is already losing validity at 0.8, before
anything looks broken). At 0.7 both coefficients meet and yaw gain falls
seven-fold in one step of 0.1.

The 1.0 row reproduces `PlannerConfig.slip_chi = 1.373` to four figures with an
0.028 spread across radii, so the measurement chain is sound and the two rows are
comparable. At 0.45 the robot achieves **5% of commanded yaw rate** — it barely
rotates at all, which is why seven of eight arcs were rejected as unmeasurable.

Why: [wheel.xacro](src/scout_ros2/scout_description/urdf/wheel.xacro#L63) gives
each wheel `mu1=200.0` (rolling) and `mu2=0.7` (lateral), and Gazebo combines two
contacting surfaces by taking the **smaller** coefficient. So `mu1=200` is never
realized — it encodes "the wheel is never the longitudinal limit, the ground
decides", which is what its comment says. Against a ground of 1.0 the effective
pair is (1.0 rolling, 0.7 lateral): a **1.43:1** ratio, a perfectly physical
number. (An earlier version of this note called it "a 285:1 anisotropy". That was
wrong — 285:1 is the nominal wheel pair, which the ground caps away.)

That 1.43:1 is the steering mechanism, since a skid-steer yaws by gripping
longitudinally while scrubbing sideways. It survives only while
`ground > wheel mu2`: there the ground binds longitudinally and the wheel binds
laterally, two independent constraints. Below 0.7 the ground binds **both**, so
lowering it reduces grip *and* collapses the ratio to 1:1 — inseparably. The
sweep shows exactly that shape.

Provenance: Grigorii Matiukhin, 2026-02-13, in the team's own `scout_ros2` fork —
not AgileX upstream, so it is ours to change.

So "make the floor realistic" cannot be done by lowering an isotropic ground
plane. The anisotropy that matters is in the **wheel** frame (rolling vs lateral)
and only the wheel can express it; a ground `fdir1` is world-fixed, which is
exactly why the `directional_x/y` patch profiles are unphysical. And per-zone
friction has to live in the ground, because patches are ground entities. That
tension is the thing to solve before any re-baselining.

**Every profile in `surface_patches.py` is below 0.7** — linoleum 0.45, wet_tile
0.30, slippery 0.20, icy 0.05 — so every slip patch ever driven over has been
simulating "loses steering", not "slides". That includes the RL training terrain
and every corrector comparison. The friction values were not merely uncalibrated
(as the section above says); they were outside the model's valid domain.

### The wheel fix: `mu2` 0.7 → 0.45, applied and confirmed (2026-08-07)

`mu2=0.7` was the problem — it sat at the top of the realistic range for rubber,
so every physically plausible floor landed at or below it. It is now **0.45** in
`wheel.xacro`. **`mu1` deliberately stays at 200**: it is never realized (the
ground caps it), and it encodes "the wheel is never the longitudinal limit, the
GROUND decides", which is exactly what makes a friction patch's `mu` mean
anything in the rolling direction. Only `mu2` sets the knee, so only `mu2` moved.

Re-run of `sweep_ground_mu.sh` (`sweep_data/ground_mu_chi_mu2_045.csv`), against
the old curve:

| ground `mu` | chi @ `mu2=0.7` | chi @ `mu2=0.45` | yaw gain | arcs | spread |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 1.3718 | **1.3575** | 0.737 | 6 | 0.007 |
| 0.8 | 1.4438 | **1.3651** | 0.733 | 6 | 0.023 |
| 0.6 | 11.760 | **1.4037** | 0.713 | 6 | 0.058 |
| 0.5 | 14.441 | **1.5713** | 0.641 | 6 | 0.341 |
| **0.45** | 16.478 | **15.583** | 0.065 | 2 | 3.449 |
| 0.4 | — | 18.729 | 0.055 | 2 | 5.733 |
| 0.3 | — | 25.362 | 0.040 | 2 | 6.084 |

**The knee moved to 0.45, to the digit** — predicted before the run, exactly as
the 0.7 knee was. Min-combination is settled, not a hypothesis.

**The curve TRANSLATED rather than deformed, and the free variable is the RATIO
`ground/mu2`, not absolute friction.** 0.5/0.45 (ratio 1.11) gives chi 1.57;
the old 0.8/0.7 (ratio 1.14) gave 1.44. So `sweep_ground_mu.sh` measures one
curve in one variable and `mu2` slides it — that is what makes it a reusable
instrument for "does this tyre model steer" rather than a one-off.

**Chi at nominal barely moved: 1.3718 → 1.3575, ~1%.** The prediction was a
visible drop from the improved ratio (1.43:1 → 2.22:1) and that was **wrong** —
above the knee chi cares *that* you are above it, not by how much. Consequence:
`PlannerConfig.slip_chi = 1.373` is still within ~1% at ground 1.0, so the
existing baked plans did **not** need re-planning for this change. The spread
across radii also improved (0.0299 → 0.0072), so a single `slip_chi` describes
this plant *better* than it described the old one.

**Usable band is ground >= 0.5.** Linoleum at 0.45 sits exactly on the knee and
is marginal; wet_tile (0.30), slippery (0.20) and icy (0.05) are all still below
it. Ice being uncontrollable is **correct** rather than a limitation — on real
ice `mu_long ~= mu_lat` and a real skid-steer genuinely cannot steer. The model
was never broken for ice; it was broken for linoleum, because the wheel was
parameterised for concrete.

**The limit that survives the fix, unchanged:** under min-combination with an
isotropic ground, *any* ground below the wheel's `mu2` gives ratio 1. "Slides but
still steers" is inexpressible at low friction — a surface is either above the
knee or it has no steering authority. **`sand` is still blocked on this**, at any
wheel setting; it needs a different mechanism, and that is a question for the
advisor alongside the planner-vs-corrector one.

**What still stands:** chi is genuinely a property of the SURFACE (1.36 vs 25.4
across this curve), so it cannot be a constant on a floor with ice/sand zones —
it is not constant *within one trajectory*. That remains a modelling gap rather
than a tuning problem, and the structural reason a frozen PMP plan cannot handle
zones.

**Do not lower the ground plane to model a slippery floor.** Both worlds are at
`mu=1.0` and should stay there; slipperiness belongs in the wheel pair or in a
patch, and any patch below 0.45 now means "no steering", deliberately.

**Everything measured before this change was measured on a different plant.**
The identity / TVLQR / RL comparison and every tuning result predate it.

