# Handover — 2026-08-12

**Read this first, and keep it current.** It is the primary record of what we are
doing; CLAUDE.md's "Current work" section is the cumulative record of what we
have *established*. This file describes **now**: what is running, what is
half-finished, what to do next, and the reasoning behind decisions that have not
yet become findings. Rewrite it rather than appending. (The rule is written down
at the top of CLAUDE.md's "Current work" section.)

**2026-08-12 in one line:** the 2026-08-07 tuning run is in, and
`q_cross=0.276 / r_omega=2.618` is the **first tuning result on this project that
survives independent re-measurement** — 0.621 m against the default's 1.004 m,
measured three times in three processes, best of a 15-point local grid.

## What happened this session

1. **Read the completed run** — `tvlqr_tune_v4_newplant.jsonl`, 100 BO
   evaluations, mean-of-3, 3.2 h, zero failures, plant `2026-08-07-wheel-mu2-045`.
   Default 1.042 m, best 0.614 m. The within-evaluation SEM is 0.026 m, so unlike
   the two previous runs the improvement is ~10x the noise it was selected from.
2. **Validated it** — independent mean-of-5 at each point: **1.0037 vs 0.6212**.
3. **Probed below the search box's `q_cross` floor** (23 of 100 evals had piled
   against it). The bounds were **not** the problem — nothing below 0.1 is better.
   Needed a new `--q-bounds` flag, because `x0` is clipped into the box and a
   probe outside it silently measures the boundary.
4. **Mapped the neighbourhood** — 5 `q` x 3 `r`, mean-of-3 each. The tuned point
   is the grid's best (0.643 here) and 13 of 15 points beat the default.

Full tables in CLAUDE.md; raw JSONL in gitignored `tune_data/`; the figure is
`figures/tvlqr_validation.png` from `tools/plot_tune_validation.py`.

## One reading was wrong, and the correction is the interesting part

Mid-session I attributed the improvement to "the U-turn and the S landing in
their good mode", i.e. mostly mode luck. **That was computed against the q-sweep
NEIGHBOURS rather than against the default**, which is the baseline the
improvement is actually measured from. Decomposed properly:

| shape | delta (default − tuned) | share |
| --- | --- | --- |
| loop | +1.233 | 46% |
| zigzag | +1.129 | 42% |
| S | +0.526 | 20% |
| U-turn | **−0.090** | **−3%** |

**The loop and zigzag are 88% of it; the U-turn contributes nothing.** So the
result is *stronger* than the mid-session reading: three shapes improve
substantially and smoothly, and it is not mode selection.

**The bistability is still real, with its scope corrected.** Over 45 grid
rollouts the U-turn is bimodal — 33% good (mean 1.555), 67% bad (mean 2.575),
with no smooth dependence on `(q, r)`. It is a hazard when comparing
**neighbouring gain points to each other**, not a contaminant of tuned-vs-default,
where both arms sit in the good mode 5 times out of 5.

**Rule this establishes: a per-shape claim must name its baseline.**

## Do this next, in order

1. **Explain the U-turn's two modes.** Drive `floor_6_00031` ~10x at fixed gains
   with `--trace-dir`, then `tuning/trace_diff.py` to find the step where the two
   modes part and which column moves first — it distinguishes "our controller"
   from "physics" from "a dropped step" directly. ~15 min, and a 1.5 m bimodal
   split at fixed gains and fixed seed is a plant phenomenon worth more than
   further tuning. The soak (below) is already accumulating the frequency data
   this needs.
2. **Then adopt the gains** in `tvlqr.TVLQRConfig` (still `10 / 0.25` in code).
   The evidence supports them; the reason to wait is only that an unexplained
   bimodality in the eval set is a bad thing to bake a default on top of.
3. Reconsider the objective. An unweighted mean over seven shapes lets one
   bistable shape move the aggregate by ~0.2 m. Per-shape normalisation against
   the identity baseline would tune for "tracks well everywhere". Design
   question, not a tweak — write down the reasoning before changing it.
4. **Do not re-run a wide search yet.** Two summaries of this landscape have now
   been wrong in the same way (binned over `q` hid the `r` structure, binned over
   `r` hid the `q` structure). Treat any one-variable summary as suspect.

## For the write-up

A clean, self-contained story: three tuning runs, the first two winner's curse
and documented as such, the third surviving validation — and then the validation
correcting our own reading of *why* it won. The methodological progression is
itself the content. Figures: `figures/tvlqr_validation.png` (three panels) and
the repeat-level bimodality.

## State

- **VM:** one headless `gz sim` on `rl_corrector.world` (ground mu=1.0), started
  fresh 2026-08-12; the 5-day-old instance from 2026-08-07 was killed first.
- **A soak is running** (`just soak`, tmux window `rl:soak`, log `/tmp/soak.log`,
  data `~/soak.jsonl`). It cycles the tuned and default gain points forever,
  writing raw per-rollout results, and never optimizes anything. **Stop it with
  `just kill-sim` before any focused test** — `just check-sim` will refuse
  otherwise, which is correct. It loses nothing when killed.
- **Caches on the VM, all on the current plant and safe to resume onto:**
  `~/tvlqr_tune_v4_newplant.jsonl` (+ `~/tvlqr_tuned.json`),
  `~/validate_20260812_{tuned,default}.jsonl`, `~/qwall_20260812.jsonl`,
  `~/local2d_20260812.jsonl`. All fetched into local `tune_data/`.
- **Poisoned caches, do not resume onto them:** `~/tvlqr_tune_v2.jsonl`,
  `~/tvlqr_tune_v3.jsonl`, `~/tvlqr_tune.jsonl` — abandoned plants.
  `PLANT_VERSION` will refuse them.
- One grid evaluation failed with `terrain patches failed to spawn:
  ['rl_patch_2']` and was correctly invalidated rather than averaged over
  survivors, then re-run. The 2026-08-01 guard working as designed; worth knowing
  it still fires occasionally.
- Both worlds stay at **mu=1.0**. Slipperiness belongs in the wheel pair or a
  patch; a patch below 0.45 deliberately means "no steering".
- **Watch for `--` in XML comments** — a dash written that way in `wheel.xacro`
  made xacro fail to parse and the sim never came up.

## Still open, unchanged from 2026-08-07

- `sand` is unbuilt and blocked on the friction-anisotropy question: under
  min-combination with an isotropic ground, any ground below the wheel's `mu2`
  gives ratio 1, so "slides but still steers" is inexpressible at low friction.
  A question for the advisor.
- chi is a property of the **surface** (1.36 to 25.4 across the sweep), so it
  cannot be a constant on a floor with ice/sand zones. See "Handling chi when it
  is not constant" below.
- The RL re-planner architecture below is decided but **not started**. The PMP
  solver needs a "re-join from an arbitrary state onto the nominal path" boundary
  condition, and its per-solve cost needs measuring early since it sets the data
  budget.

## Architecture: RL as a rough re-planner, not a residual

The advisor's reply was *"RL нужен … надо прикрутить его, как **грубый
планировщик**. Пока 2 зоны — лед и песок"* — RL at the **planning** layer, for
dynamically appearing problem zones, with exactly two zone types. The user's
sharpening of it is the key idea: **the policy's job is to reproduce the PMP plan
that brings the robot back on track.**

That makes it **amortized optimization with PMP as the teacher**, so it is
supervised, not SAC:

```
sample (nominal plan, deviation state, local chi)
    -> run the real PMP solver -> optimal re-join trajectory
    -> regress
```

Which kills, one for one, every open RL problem: no exploration ⇒ no `ent_coef`
runaway (queue item 1); no bootstrapped value ⇒ no `critic_loss` divergence (item
2); no reward shaping; each label is an exact optimum rather than a noisy return;
and **data generation needs no Gazebo at all**, so it is CPU-parallel. Gazebo
returns only for validation, on the existing eval set and comparison harness.

Runtime layering, which preserves the project's philosophy (one expensive optimal
solve offline, cheap corrections online — nothing online is an optimizer):

| layer | rate | cost | job |
| --- | --- | --- | --- |
| PMP | once/goal, offline | seconds | optimal plan on the **nominal** surface |
| RL re-planner | ~2–5 Hz | one MLP forward pass | reference has become infeasible ⇒ emit a short re-join |
| TVLQR | 50 Hz | precomputed gains | track whichever reference is active |

The split is real rather than nominal: TVLQR handles small deviations around a
*feasible* reference, which is what LQR is optimal at; when a zone makes the
reference infeasible no gain matrix helps and the *reference* must change, which
is structurally outside TVLQR's reach. PMP is the right thing to call there and
is a ~20 s BVP solve, so the policy is precisely its fast approximation.

Side benefits: a planner emits waypoints, so the **4-wheel-vs-twist deployability
problem disappears** (this supersedes the 2026-08-04 "keep the 4-wheel residual"
decision); RL no longer has to beat TVLQR to justify itself; and the existing
residual work becomes a documented negative result, which is usable material for
the intro the advisor asked for.

**SETTLED 2026-08-07 (user):** corrections run **on top of the frozen PMP path**.
A full PMP re-solve is reserved for situations that cannot be corrected around at
all — the example given was an unexpected wall in the way. So the frozen plan is
the default reference and re-planning is an exceptional event, not a control
layer running at some rate. Two consequences worth holding onto:

- the RL re-planner's job is bounded — *re-join the existing path*, never *find a
  new one* — which is a much smaller function to approximate and keeps the
  project's "one expensive optimal solve offline" philosophy intact;
- a full re-solve needs a **trigger**, and "the reference is infeasible" is a
  different test from "we are far off it". Nothing implements that yet.

Open items on this: the PMP solver needs a "re-join from an arbitrary state onto
the nominal path" boundary condition (a BC change, not a new solver), and its
per-solve cost over a 2–3 s horizon needs measuring early since it sets the data
budget.

## Handling chi when it is not constant

Decided today, and the user confirmed reactive is acceptable ("that matches our
original idea"):

1. **The planner keeps one nominal chi** — correct by definition, since "nominal"
   means the surface it was measured on. Today's bug is that the constant was
   measured on the wrong surface, not that it is a constant.
2. **chi becomes a measured signal online** in the correction layers:
   `chi_hat = omega_ideal(from wheel commands) / omega_measured(gyro)`, i.e.
   `slip_ident`'s computation run recursively. Feed it to TVLQR (its
   linearization uses `track_eff = track * chi`) and to the policy as an input.
   **Observability caveat:** undefined when `omega ≈ 0` — you cannot measure yaw
   loss driving straight — so it needs a validity gate that holds the last value.
   Free side effect: this is exactly what the known-wrong `wheel_odometry` yaw
   integration and the EKF bias are missing.
3. **Reactive by construction.** You learn the ice is slippery by sliding on it.
   Pre-emptive routing needs perception (a camera recognising ice) and is out of
   scope.

## Measuring chi on the real robot

Yes, and it is the intended use of `slip_ident` — it references the **gyro**, so
nothing about the method is sim-specific (`calibrator.py` cannot substitute: it
compares commands against `/odom` and both sides share the missing slip term).
Two things to get right, both now in CLAUDE.md:

- **`cmd_mode:=wheels` is sim-only.** On the real Scout use `cmd_mode:=twist`,
  which yields `chassis_gain_omega` — chi folded together with the firmware's
  conversion. Do not paste that into `PlannerConfig.slip_chi`.
- Drive **arcs at several radii**, both directions. Spins are load-dependent and
  reported separately; do not fit chi to them.

**This is how the sim gets an empirical anchor.** Measuring `mu` directly is
awkward; measuring chi is not. Drive the real robot on linoleum, on the ice
mock-up and on sand, then tune each sim profile until the *sim's* chi matches.
Chi, not mu, is the quantity to match — it is what the model consumes.

