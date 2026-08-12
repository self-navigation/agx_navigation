# Handover — 2026-08-12

**Read this first, and keep it current.** It is the primary record of what we are
doing; CLAUDE.md's "Current work" section is the cumulative record of what we
have *established*. This file describes **now**: what is running, what is
half-finished, what to do next, and the reasoning behind decisions that have not
yet become findings. Rewrite it rather than appending. (The rule is written down
at the top of CLAUDE.md's "Current work" section.)

**2026-08-12 in one line:** the overnight tuning run from 2026-08-07 is in, and
**for the first time a tuning result survives independent re-measurement** —
`q_cross=0.276 / r_omega=2.618` gives 0.621 m against the default's 1.004 m — but
the follow-up sweep shows the win is mostly **two bistable shapes landing in
their good mode**, not uniformly better tracking.

## What happened this session

1. **Read the completed run** (`tvlqr_tune_v4_newplant.jsonl`: 100 BO
   evaluations, mean-of-3, 3.2 h, zero failures, plant `2026-08-07-wheel-mu2-045`).
   Default 1.042 m, best 0.614 m at `q=0.276 / r=2.618`. The within-evaluation
   SEM is 0.026 m, so unlike the two previous runs the improvement is ~10x the
   noise it was selected from.
2. **Validated it** — mean-of-5 at both points, fresh sim, fresh caches:
   **1.0037 (default) vs 0.6212 (tuned)**. It holds.
3. **Probed below the search box's `q_cross` floor**, since 23 of 100 evaluations
   had piled against it. **The bounds were not the problem** — everything below
   0.1 is worse. But the minimum turned out to be a **narrow spike**: the
   neighbours at q=0.1 and q=0.6 both score ~1.0, level with the default.
4. **Found why it wins**, and it is not what the aggregate suggests. See CLAUDE.md
   "The optimum is a narrow spike, not a basin" for the per-shape table.

All three results, the per-shape breakdown and the caveats are written up in
CLAUDE.md; the raw JSONL is in gitignored `tune_data/`.

## The finding that should drive the next session

**The U-turn (`floor_6_00031`) and the S (`floor_6_00018`) are bistable.** Each
lands at either ~1.2 m or ~2.7 m and nothing in between, and those two shapes
alone account for 0.34 m of the 0.38 m improvement. The other five barely move.

So the tuner is substantially **selecting modes, not tracking quality** — an
unweighted mean over seven shapes, two of which flip across a ~1.5 m gap, is
dominated by which side of the flip those two land on. This is the
"discrete modes, not smooth noise" phenomenon from 2026-08-02 appearing in the
*objective* rather than in a repeat.

The saving grace, and the better argument for adopting the gains: across 5
repeats the tuned point is tight (sd **0.020**) while the default is visibly
**bimodal** (0.835 / 0.846 / 1.061 / 1.133 / 1.144, sd 0.137). The tuned gains
are *more repeatable*, not just lower on average.

**What the two U-turn modes physically are is unknown, and finding out is
probably worth more than any further tuning.** A 1.5 m bimodal split on a fixed
trajectory with fixed gains and a fixed seed is a plant/controller phenomenon,
not measurement scatter.

## Do this next, in order

1. **Characterise the bistability.** Drive `floor_6_00031` ~10 times at the tuned
   gains with `--trace-dir`, and use `tuning/trace_diff.py` to find the step
   where the good and bad modes part company and which column moves first. That
   tool exists precisely for this and answers "our controller vs physics vs a
   dropped step" directly. Cheap (~15 min) and it is the highest-information
   experiment available.
2. **Map the width of the good window** — a fine scan of `q_cross` over
   [0.15, 0.5] at `r_omega=2.618`, mean-of-3, ~8 points, ~15 min. A gain that
   only works within a factor of 1.5 is fragile and we should know that before
   it goes anywhere near the real robot.
3. **Only then** consider adopting the gains as the default in
   `tvlqr.TVLQRConfig`. They are currently NOT adopted — the defaults in the code
   are still `q_cross=10 / r_omega=0.25`.
4. Reconsider the objective. If two bistable shapes dominate an unweighted mean,
   a per-shape normalisation (each shape relative to its identity baseline) would
   tune for something closer to "tracks well everywhere". This is a real design
   question, not a tweak — write down the reasoning before changing it.

## For the write-up

This session is a clean, self-contained story worth a section: *a tuning result
that validated, and then the validation revealed the metric was measuring
something other than what it claimed.* Three tuning runs, of which the first two
were winner's curse (and are documented as such), the third survived — that
progression is itself the methodological content. The per-shape table and the
repeat-level bimodality are the two figures.

## State

- **VM:** one headless `gz sim` on `rl_corrector.world` (ground mu=1.0), started
  fresh 2026-08-12. The 5-day-old instance from 2026-08-07 was killed first.
  `just check-sim` before launching anything.
- **Nothing is running now.** Both of today's jobs completed; logs
  `/tmp/agx-run-20260812-143333.log` (validation) and
  `/tmp/agx-run-20260812-144027.log` (q sweep).
- **New caches on the VM**, all on the current plant and safe to resume onto:
  `~/validate_20260812_{tuned,default}.jsonl`, `~/qwall_20260812.jsonl`,
  `~/tvlqr_tune_v4_newplant.jsonl` (+ `~/tvlqr_tuned.json`). Fetched into local
  `tune_data/`.
- **Poisoned caches, still do not resume onto them:** `~/tvlqr_tune_v2.jsonl`,
  `~/tvlqr_tune_v3.jsonl`, `~/tvlqr_tune.jsonl` — all measured plants we have
  abandoned. `PLANT_VERSION` in `tune_tvlqr.py` will refuse them.
- **Code changed this session** (uncommitted): `tune_tvlqr.py` gains
  `--q-bounds` / `--r-bounds`, because `x0` is **clipped** into the search box —
  a single-point probe outside the default bounds silently measures the boundary
  and reports it under the label you asked for. 166 unit tests pass. Also two
  `.claude/hooks/` timer scripts hardened (stale-stamp sweep, non-numeric guard,
  human-readable durations over 90 s).
- Both worlds stay at **mu=1.0**. Slipperiness belongs in the wheel pair or a
  patch; a patch below 0.45 deliberately means "no steering".
- **Watch for `--` in XML comments** — writing a dash that way in `wheel.xacro`
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

