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

