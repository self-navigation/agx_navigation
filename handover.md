# Handover — 2026-08-13

**Read this first, and keep it current.** It is the primary record of what we are
doing; CLAUDE.md's "Current work" section is the cumulative record of what we
have *established*. This file describes **now**: what is running, what is
half-finished, what to do next, and the reasoning behind decisions that have not
yet become findings. Rewrite it rather than appending. (The rule is written down
at the top of CLAUDE.md's "Current work" section.)

**2026-08-13 in one line:** the tuned gains are **adopted**, and by evening two
things about *why* had been corrected — the U-turn is a **narrow notch in
`q_cross`**, not a mode frequency the gains can't touch, and scoring in the
advisor's own **`J`** turns a 4-3 win into **7-0**.

## What happened this session

1. **Read the soak.** 4108 rollouts, 43 failed (1.0%, all `terrain patches failed
   to spawn`, correctly invalidated), 21 processes, 14 complete cycles.
   Aggregate per cycle: tuned **0.6686 ± 0.0882** vs default **1.1273 ± 0.1108**,
   n=14 each, distributions essentially disjoint. Fourth independent measurement
   of the tuned point (0.614 / 0.621 / 0.643 / 0.669).
2. **Adopted the gains** — `TVLQRConfig.q_cross` 10.0 → 0.276, `r_omega` 0.25 →
   2.618. Note `r_omega` going *up* contradicts its own docstring comment
   ("angular correction is cheaper"); the comment is left with the correction
   beside it, since the argument was sound and the measurement disagreed.
3. **Started the mode-frequency ladder** (below) — the natural follow-up now that
   the mechanism is known to be mode frequency.

4. **Ran the `q_cross` ladder** (1047 rollouts). The zigzag is a wide *threshold*
   with the adopted point safely inside it; the **U-turn is a narrow notch**, and
   the morning's "the gains do not control the U-turn" is retracted — at `q=0.1`
   and `q=0.6` it is deterministically bad (sd 0.007 / 0.002), so it was never a
   mixture. Only two sampled points hid a non-monotonic curve.
5. **Traced the U-turn and looked at it.** All the deviation is produced at ONE
   corner, in the last quarter of the run; the first 12.5 of 17 m track to within
   0.25 m in all 30 rollouts. `max|e_cross|` there is a single-event metric.
6. **Computed `J` for the first time**, on 70 freshly traced rollouts (7 shapes ×
   2 arms × 5). Tuned wins **all seven** shapes in `J` versus 4 of 7 in
   `max|e_cross|`; the three flips are shapes that conceded peak deviation and
   bought back accumulated error, effort and terminal miss.
7. **Restructured `figures/`** — dated directories, committed images, a README
   index per date, `render.py` beside the pictures it makes. See
   `figures/README.md` and `figures/2026-08-13/README.md`.

Full tables in CLAUDE.md, "What 4065 rollouts say", "The `q_cross` ladder" and
"Scoring in `J`". Raw rows in gitignored
`soak_data/soak_20260813_twopoint.jsonl`.

## What the soak changed about our understanding

**Four of seven shapes are near-deterministic** (sd ≤ 0.09): straight, corner, S,
loop. Those are plain level shifts — three won by the tuned gains, and the
**corner genuinely won by the default** (0.226 vs 0.311 at sd 0.000). No noise
anywhere in them.

**Three are bimodal, and the gains move the bad mode's FREQUENCY, not its depth:**

| shape | bad mode | tuned %bad | default %bad |
| --- | --- | --- | --- |
| zigzag | ~2.2–2.5 m | **2.6%** | **88.8%** |
| tight V | ~0.78 m | 0.0% | 20.4% |
| U-turn | ~2.5–2.9 m | 10.3% | 12.4% |

**RETRACT "the tuned gains land the U-turn in its good mode."** 10.3% vs 12.4% is
no difference at n≈285 — the gains do not control it at all. This also corrects
yesterday's "33% good / 67% bad": that pooled 45 rollouts across *many* gain
pairs, and at neither validated point is the bad mode remotely a majority.

**The zigzag was never noisy.** Its sd of 0.2–0.5 in every earlier table was a
~90/10 mixture of two reproducible outcomes sampled 3 times. That reframes every
"run-to-run variance" number above it: on the bimodal shapes those are mixture
widths, not measurement error, and the right estimator is a mode *frequency*
(~100 samples) rather than a mean of 3. Mean-of-3 stays fine for **searching**;
a per-shape **claim** now wants soak-scale n.

## THE SOURCE DOCUMENTS CHANGE THE PRIORITIES (read 2026-08-13)

The advisor's dissertation draft and the paper seed were read directly for the
first time. Full findings in CLAUDE.md, "What the source documents actually say".
The method is **SVCM**, `epsilon`-optimality is `J[u] <= J* + epsilon`, and
**Theorem 1** gives a dichotomy: either some admissible control achieves
acceptable `J` and the template-catalogue scheme is provably ε-optimal, or none
does and the failure is physics rather than algorithm.

Three of those findings reorder the work:

- **We measure `max|e_cross|`; the framework is stated in `J[u] - J*`.** Never
  computed. It is computable offline from `PlannerConfig`'s cost weights plus
  recorded tracks — but **not** retroactively. `variance_probe.drive` reduces
  each rollout to scalars, so the ~4000 soak rollouts cannot be rescored; only
  fixture runs (`run_recorder` writes `_track.csv`) can. Capturing tracks for
  future rollouts is a small change and should land before the next long soak.
- **The `mu2` steering cliff is an instance of Theorem 1's second branch**, which
  the dissertation asserts but never demonstrates. That upgrades the friction
  sweep from side-quest to contribution.
- **RL belongs at the costates / transversality conditions**, not on the wheel
  commands. An independent reason the residual was the wrong object.

## Do this next, in order

1. **Read the U-turn sub-ladder** (running, see State). Does `q=0.276` have a
   basin, or is it a spike between two 100%-bad neighbours? This decides whether
   the adopted gains are defensible as a *choice* or merely as a *measurement*.
   ~30 rollouts per rung per 18-minute batch.
2. **Score it in `J` too — and capture tracks from now on.** The sub-ladder is
   running without `--trace-dir`, so it produces metres only. Given that `J`
   reversed three of seven per-shape verdicts, any future soak whose numbers
   should be readable in the thesis's currency needs the track. Wiring
   `--trace-dir` into `soak.py` is small and should land before the next long run.
3. **Explain the U-turn's last corner.** The pictures narrow it from "17 m of
   trajectory" to ~20 steps at one corner. Trace a few rollouts either side of the
   notch edge (`q=0.4` vs `q=0.5`) and run `tuning/trace_diff.py`: `cmd*` moving
   first means our controller, state moving first means the plant. ~10 min of sim.
4. **Then reconsider the objective.** `J` is now computable and better behaved
   than the 7-shape mean of `max|e_cross|`; the open question is whether to *tune*
   against it, which needs `J` inside `objective.py` rather than as a post-hoc
   script.
5. **Measure the re-join PMP solve time** before committing to any architecture.
   `200388e` added acados and `0454d3d` removed it, after which online mode was
   documented as infeasible — so the "cannot solve online" premise is currently a
   statement about scipy, not about the problem. The source's own justification
   for offloading cites **DShot and PWM** frame budgets, i.e. a flight
   controller, not a Jetson. The architecture is sound where its premise holds;
   whether the premise holds *here* is unverified and cheap to check.
6. **Do not re-run a wide search yet.**

## For the write-up

A clean, self-contained story: three tuning runs, the first two winner's curse
and documented as such, the third surviving validation — and then the validation
correcting our own reading of *why* it won, twice in one day (mode frequency,
then a notch). The methodological progression is itself the content.

Figures now live in dated, committed directories with a README index each —
`figures/2026-08-13/` is the model, `figures/README.md` the convention. The
paper-ready ones so far: `2026-08-13/02_uturn_modes.png` (the deviation is one
corner), `03_ladder_modes.png` (threshold vs notch), `04_epsilon_vs_cross.png`
(the objective change does not reverse the conclusion), and
`archive/tvlqr_validation.png` (still the three-panel validation figure).

## State

- **VM:** one headless `gz sim` on `rl_corrector.world` (ground mu=1.0), started
  fresh 2026-08-12; the 5-day-old instance from 2026-08-07 was killed first.
- **The U-TURN SUB-LADDER soak is running** (`just soak`, tmux `rl:soak`, log
  `/tmp/soak.log`, data `~/soak.jsonl`). `floor_6_00031` only, `q_cross` =
  0.2 / 0.276 / 0.32 / 0.4 / 0.5 / 0.6 at `r_omega=2.618`, ~6 s a rollout,
  180 per batch, self-restarting. **Question it answers:** the notch at 0.276 has
  neighbours at 0.1 and 0.6 that are 100% bad, so does the adopted point have a
  usable basin or is it a spike? First cycle already put 0.4 at 1.53 and 0.5 at
  2.70, so the edge is between them. **Stop it with `just soak-stop`** before any
  focused test; it loses nothing when killed.
- Finished earlier today and archived on the VM: `~/soak_20260813_ladder.jsonl`
  (1047 rollouts, the 6-point q ladder) and
  `~/soak_20260813_uturn_subladder_partial.jsonl` (83 rows, first sub-ladder
  attempt — same conditions as the running one, safe to pool). Both fetched into
  gitignored `soak_data/`.
- `~/jsweep.jsonl` + `~/jtraces/` — the 70-rollout traced sweep that produced the
  first `J` numbers. Fetched locally into `jtraces/`, scored into `epsilon_data/`.
- The finished two-point soak is archived on the VM as
  `~/soak_20260813_twopoint.jsonl` (log `/tmp/soak_twopoint.log`) and locally in
  gitignored `soak_data/`. Do not append the ladder to it — different conditions.
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

