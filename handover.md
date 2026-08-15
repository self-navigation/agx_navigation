# Handover — 2026-08-15

**Read this first, and keep it current.** It is the primary record of what we
are doing; CLAUDE.md's "Current work" section is the cumulative record of what we
have *established*. This file describes **now**: what is running, what is
half-finished, what to do next, and the reasoning behind decisions that have not
yet become findings. Rewrite it rather than appending.

**This session in one line:** the queue emptied cleanly overnight and both
questions it was holding got answered — **`r_omega`'s move off 0.25 is real and
general** (the previous ladder's "flat" was an artifact of its floor), and **the
U-turn `q_cross` basin belongs to one plan, not to the shape** — so the open
question from last session's "For the user" is settled by data rather than by
judgement.

**Then, same session: parallel sims landed.** `WORKER=n` gives a sim its own
Gazebo partition and DDS domain, verified live beside a running job — see
"Parallel sims are done" below. **The demo is no longer blocked by the queue.**

**FIRST THING NEXT SESSION:** score job 50 — its data is already fetched
(`soak_data/soak_broad_gains.jsonl`, 720 rows, + `broad_gains_traces/`), it just
has not been read yet.

## What the two overnight jobs said

**Job 30, `r_omega` below the ladder floor (840 rollouts).** The previous ladder
swept r ∈ [1.0, 5.0] and found `r` flat everywhere but the U-turn, which made the
adopted 2.618 look like one plan's notch. **The low half reverses that reading.**
The zigzag *halves* between r=0.5 and r=1.0 (0.812 → 0.387, sd ≤ 0.04 on both
sides), the six-shape aggregate excluding the U-turn goes 0.635 → 0.495, and
`final_err` is **monotone across the whole range** (0.661 → 0.343 m) with no
shape getting worse. So `r` has a threshold in (0.5, 1.0) and is flat above it.

**The consequence is a two-part claim, and the halves differ in strength:** the
move *off* r=0.25 is real and general; the specific value **2.618 vs anything in
[1.0, 5.0] is not load-bearing** — it is kept because it is measured, not because
it is special. Write it up that way. `TVLQRConfig` is unchanged.

**Job 40, does the U-turn basin generalise (1200 rollouts)? No.** Four v2 U-turn
plans plus `floor_6_00031` as an in-run control, q ∈ {0.2, 0.276, 0.4, 0.5}. Only
the control has the basin (97 / 18 / 0 / 100 % bad). Two of the others are flat
and easy at every rung, one is 100% bad at every rung, one is weakly bimodal with
**no `q` dependence at all**.

**The decisive evidence was the picture, not the table** — and this is the part
worth carrying forward. Rendered, the five plans are one true hairpin, one
rectangular U, one **bent line that is not a U-turn**, and one **near-duplicate
of the control** (same corridor, same three-sided route, ~1 m different start).
That near-duplicate shows *no basin at all*, which is what turns "you picked the
wrong plans" from an objection into the finding: the basin belongs to that exact
plan, not to its shape and not even to its corridor. All five score
`total_abs_turn` 7–9 rad, so the label cannot tell a hairpin from two same-sign
90° corners. **Render plans before any per-shape claim.**

## RUNNING NOW / QUEUED

**`50_broad_gain_generality.sh` — DONE (rc=0, 14:05 UTC), 720 rows, all traced,
FETCHED but NOT YET SCORED.** Local copies: `soak_data/soak_broad_gains.jsonl`
and `broad_gains_traces/` (720 files). Scoring it is the first job next session.

This is the direct answer to the user's question — *are the gains overfitted to
seven plans, and can we optimise for all of them?* 40 plans chosen mechanically
by `tools/select_broad_eval.py` from the 320-plan constructed library (none of
the seven appear; four labels, 8.7–36.0 m), against six gain pairs that map the
`q` plateau and its cliff and test r=1.0 against 2.618 on ground that is not
`floor_6_00031`:

    (0.276, 2.618) adopted   (0.276, 1.0)   (0.6, 2.618)
    (1.5, 2.618)             (4.0, 2.618)   (10, 0.25) old default

**RUNNING NOW: `60_tune_on_J.sh`** (started 14:05 UTC, at eval 5 of 60 as of
14:15, ~103 s per evaluation ⇒ ~1.7 h) — re-tunes `(q_cross, r_omega)` against
`J` instead of metres (~2-3 h, bounded at 60 evals), then measures the adopted
0.276/2.618 in `J` in the same code path for comparison. Results
`~/tvlqr_tuned_J.json` + `~/tvlqr_validate_J_adopted.json`, caches
`~/tvlqr_tune_J.jsonl`. It searches the SEVEN-plan set on purpose — that stays
the fast search set (mean-of-3 must stay ~105 s/eval); the broad library is for
validating a winner, not for searching.

**The scheduling conflict this section used to warn about is GONE.** Job 60 no
longer monopolises Gazebo: run the demo in its own partition beside it —

    just remote-fixture tvlqr true truth 1

Score job 50 (data already local):

    .venv/bin/python tools/score_sweep.py --from-jsonl soak_data/soak_broad_gains.jsonl \
        --trace-root broad_gains_traces --plans traj_data_v2

**How to read it.** The aggregate over 40 independent plans is the number that
either does or does not support the adopted point. Three outcomes worth naming in
advance: (a) 0.276 wins broadly ⇒ the seven-plan result generalises, say so and
stop tuning; (b) some other rung wins broadly ⇒ **re-tune against the broad set,
not the seven**, and treat every per-shape number in CLAUDE.md as the enriched
subset it is; (c) the ranking is flat and dominated by which plans are hard ⇒ the
gains matter less than the objective does, which points at item 2 below.
Score in **both** currencies — they disagreed on 24 of 51 plans in the library
sweep.

## Parallel sims are done (2026-08-15)

`WORKER=n` (1-9) → `GZ_PARTITION=agxn` + `ROS_DOMAIN_ID=40+n`. **No code
changed** — both transport libraries read the environment at init, so
`GazeboBridge`, the launch files and the tuner are untouched. Full detail and
the verification table are in CLAUDE.md, "Parallel sims: `WORKER`".

    make rl-sim WORKER=1 / make fixture WORKER=1 / make rl-train WORKER=1
    just remote-sim rl_corrector.world 1     just gui 1
    just remote-fixture tvlqr true truth 1
    tools/with-worker 1 python3 -m agx_planning.tuning.soak …

**Verified beside a live job**, which is the only test worth trusting: worker 1
scored `floor_6_v2_00004` at **0.2901 twice** against the default partition's
**0.2901 ± 0.0001 over 60 samples**, and job 60's evaluation time was 103 s
before and 103 s during. A worker is the same plant, so parallel numbers are
comparable with everything already measured.

`just check-sim` is scoped by partition (defaults to `default`, so it is as
strict as before for anyone not passing a worker) and now runs
`tools/kill_stack.sh list` rather than its own `pgrep`, so the guard and the
sweep cannot disagree. `just kill-sim` takes a partition, defaulting to `all`.

**What is NOT done: the job queue is still single-lane.** `tools/jobq.sh` runs
one job at a time in the default partition, so using workers today means driving
them by hand. Per-worker queues are the follow-up, and they are what turns the
re-planner's ~55 core-hours of PMP labelling into an overnight run. Two things
to get right when building it: each lane needs its own lock and log directory,
and `just queue-add` currently rsyncs the tree under whatever is running, which
with N lanes means N in-flight jobs can pick up edited code at their next
`python3 -m`.

## Do this next, in order

1. **Read job 50** (above). Nothing else should start a wide gain search before it.
2. ~~Put `J` inside `objective.py`.~~ **DONE 2026-08-15.** `J` is accumulated
   online by `epsilon.EpsilonAccumulator`, so `variance_probe.drive` returns
   `j_total` directly and no trace file is involved. `objective.metric_values`
   selects the metric; `aggregate(..., how="geometric")` reduces it. **The tuner
   is wired too**: `tune_tvlqr --metric j_total`, with `--aggregate` defaulting
   to the right reducer per metric, every metric recorded on every rollout
   (`per_traj_metrics`) so a finished run is re-readable in the other currency,
   and metric+aggregator in the cache key so a `max_cross` cache cannot be
   replayed into a `J` search. Job 60 is the first run of it. Read
   `objective.py`'s docstring first: `J` needs the geometric mean because one
   plan is 48% of the arithmetic one.
3. **Read `25_uturn_notch_edge.sh`'s traces** — 10 rollouts either side of the
   U-turn's `q` wall (0.4 vs 0.5), all traced, on the VM as `~/uturn_edge.jsonl`
   + `~/uturn_edge_traces/` and fetched to `soak_data/uturn_edge.jsonl`. Run
   `tuning/trace_diff.py` on a good/bad pair: `cmd*` moving first means our
   controller, state moving first under an equal command means the plant. ~5 min
   of reading, data already paid for. **Lower value than it was** — now that the
   basin is known to be one plan's, this explains a curiosity rather than a
   mechanism. Do it for completeness, not before item 1 or 2.
4. **Stratify a second eval set** into `config/eval_trajectories.yaml`, *added*
   alongside the seven rather than replacing them. The seven stay the fast search
   set (mean-of-3 tuning must stay ~105 s/eval); the broad set is for **claims**.
   Swapping would silently re-baseline every number in CLAUDE.md. Job 50's 40
   plans are a ready-made candidate — see `~/broad_eval_plans.txt` on the VM.
4.5. **Read [docs/corrector-design.md](docs/corrector-design.md)** before
   starting any RL work. Written 2026-08-15 from a proper transcription of the
   source ([docs/svcm-source.md](docs/svcm-source.md) — the formulas are WMF
   images and every earlier extraction dropped them silently). It separates the
   two things RL could compress, which the previous RL effort conflated: the
   **template library** (~9 continuous dims, needs a network) and the **cost
   matrices per surface class** (one categorical index, needs a table — and is
   our existing tuner, run once per surface). It also recommends **retiring the
   SAC residual** rather than repairing it, and identifies the escalation
   trigger as `CorrectionDiagnostics.saturated_*`, which is already implemented
   and unused.

5. **Build the re-join re-planner — THIS IS THE NEXT REAL GOAL**, agreed with the
   user 2026-08-15. Full plan in [docs/corrector-design.md](docs/corrector-design.md)
   ("The build plan for Level A"). Four phases, and **phase 0 is the one to do
   first because it can invalidate the other three**:

   - **Phase 0:** add the re-join boundary condition (`x(t0)` = actual off-plan
     state, `x(t0+T_w)` = the plan's state at index `k + T_w/dt`, all five
     components including wheel speeds) and solve ~200 sampled re-join problems.
     **The number that decides everything is the solve failure rate** — the
     library build failed 36% (timeouts + BVP mesh exhaustion). Inherit that and
     there is no reliable teacher and the supervised plan is wrong. Offline, no
     Gazebo, cheap. Do it before any training code.
   - **Phase 1:** ~100k **randomly sampled** re-join solves. Never a grid — a
     grid at 5 points/axis over ~10 dims is 1e7 solves (~5000 core-hours);
     random sampling is ~55 core-hours at the measured 1.84 s/solve. The curse
     of dimensionality applies to TABULATING the catalogue, not to regressing it.
   - **Phase 2:** DAgger, not RL — roll out the student, query PMP at the states
     it actually reaches, retrain. (The user proposed an RL version of this; the
     instinct is right, the mechanism is not — with the expert's answer in hand
     the difference IS the exact gradient, so a reward estimator is strictly
     worse. Written up in the design doc.)
   - **Phase 3, optional:** RL fine-tuning from the supervised policy, which is
     the one place RL exceeds the teacher — PMP is optimal for a model that
     assumes nominal chi, and the plant does not. Reward is
     `-epsilon.step_cost(...)`, free and principled.

   Why the wheel speeds must be in the terminal BC: playback resumes by feeding
   the plan's commands from index `k+m`, which assume the wheels already turn at
   the plan's rate — matching position but not phase reproduces the documented
   stall-path bug deliberately. And because playback is time-indexed, landing at
   `k + T_w/dt` keeps the robot on SCHEDULE, not merely on the path; `m` is
   therefore determined by `T_w`, and `T_w` is the one free scalar (a
   free-terminal-time problem whose missing equation is a transversality
   condition — which is exactly where the advisor said learning acts).
6. **Do not lower the ground plane to model a slippery floor.** Standing rule.

## The demo is closer than the re-planner — and should come first

Roadmap in [docs/corrector-design.md](docs/corrector-design.md) ("Roadmap to a
visible demo"). **"Robot drives, slips, gets back on track" is TWO demos and the
first is already built**: TVLQR *is* a closed-loop corrector that returns to the
trajectory. Checked in the tree 2026-08-15 — the rig
(`just remote-fixture tvlqr true truth 1` — the trailing worker id puts it in
its own partition, so it no longer has to wait for the queue), the
plan-and-reference
markers (`runtime_corrector` `~/debug_markers`), the RViz display (`corrector
status` in `main.rviz`), the plan remap, deliberate patch placement, and
`run_recorder` all exist and are wired.

**What is missing is that nobody has ever watched it.** Every corrector number in
this repo came from `compare_correctors`/`soak`, which drive `GazeboBridge`
directly and **bypass the whole ROS stack** — so `vector_field` → `pmp_planner`
→ `runtime_corrector` → playback → controllers has not been exercised during any
of the corrector work. Expect bit-rot on the first run; that is the main cost.

A second reason to run it side by side now that workers exist: two fixtures on
one desktop (`worker 1` tvlqr, `worker 2` identity) stream to Moonlight as one
picture, which is the cheapest possible before/after for a demo video.

Do this **before** the re-planner build: it is the only end-to-end check that
what we tuned for weeks works in the runtime pipeline and not just in the
measurement harness. Two small gaps to fix while there: the patch list puts
`icy` under the spawn point (fine for testing, useless for a demo — you want
clean → one patch → excursion → recovery), and it is a hardcoded JSON literal in
`gz_sim.launch.py`, so a `PATCH_LAYOUT` config file is the one code change worth
making. Keep demo patches just ABOVE the wheel's mu2=0.45 knee — below it there
is no steering at all, which looks like a broken robot rather than a slipping one.

## Where the corrector work actually stands

Worth stating plainly, because six sessions of gain tuning can obscure it: the
goal is a **closed-loop, immediate-mode corrector** that holds a frozen
open-loop PMP plan under slip. TVLQR *is* that corrector and it works — the
library sweep (51 plans) put it at 0.449 m mean vs 0.605 m, and in `J` it won
45 of 51. The gain tuning is now at diminishing returns: job 50 is the check
that closes it, not the start of another round.

**What is NOT built is the part the advisor asked for and the part that handles
the actual physics:** the RL re-planner (item 5), the online `chi` estimate
(below), and the trigger that decides a reference has become infeasible. Those
are the remaining work, and none of them is a gain.

## State

- **VM:** one headless `gz sim` on `rl_corrector.world` (ground mu=1.0), started
  fresh 2026-08-12, still up. The jobq runner is up, with the subshell fix, and
  has now run five jobs to completion without dying.
- **Queue:** `50_broad_gain_generality.sh` running, nothing pending.
- **Fetched and analysed this session:** `soak_data/soak_r_ladder_low.jsonl`
  (840), `soak_data/soak_uturn_generality.jsonl` (1200),
  `soak_data/uturn_edge.jsonl` (20), and the whole `traj_data_v2/` library (320
  plans, 7.7 MB).
- **`traj_data_v2/` is now local** and committed-adjacent (gitignored). Note the
  two similarly-named remote directories: `~/pmp_trajectories_v2` is the OLD
  100-plan random-goal library that `config/eval_trajectories.yaml` still points
  at; `~/traj_data_v2` is the new constructed one.
- **Unusable, keep only as evidence of the bug:** `r_ladder_traces/` (210 files)
  — written before the `--trace-every` fixes, each file holds ~5 rollouts from
  ~5 different cells.
- **Caches on the VM, current plant, safe to resume onto:**
  `~/tvlqr_tune_v4_newplant.jsonl` (+ `~/tvlqr_tuned.json`),
  `~/validate_20260812_{tuned,default}.jsonl`, `~/qwall_20260812.jsonl`,
  `~/local2d_20260812.jsonl`. All fetched into local `tune_data/`.
- **Poisoned caches, do not resume onto them:** `~/tvlqr_tune_v2.jsonl`,
  `~/tvlqr_tune_v3.jsonl`, `~/tvlqr_tune.jsonl` — abandoned plants.
  `PLANT_VERSION` will refuse them.
- Both worlds stay at **mu=1.0**.
- **Watch for `--` in XML comments** — a dash written that way in `wheel.xacro`
  made xacro fail to parse and the sim never came up.
- **More than one Claude session can be live on this checkout at once.** On
  2026-08-13 another session's `git add -A` committed this session's work under
  an unrelated message. Habit: `git status` before committing, explicit paths on
  `git add`. Do not wake other sessions to coordinate — ask the user.
- **`notify_and_wait` takes a REPLY** (`tools/attention_mcp/server.py`), via
  `zenity --entry`. The desktop notification truncates at ~60 characters, so
  **the question must come first** in the message. Its best use is the one no
  metric covers: asking the user to look at the sim on Moonlight and say whether
  the robot is actually driving the plan.

## The queue is the way to run long things

`just queue-start` once, then `just queue-add <job>.sh` at any time — including
while another job runs. Jobs live in `tools/jobs/`, logs in `~/jobq/logs/`.

```
just queue-status              # runner, running/pending/done/failed, recent logs
just queue-add 50_broad_gain_generality.sh
just queue-log 50_broad_gain_generality   # a specific job's log
just queue-stop                # stops the RUNNER; leaves the in-flight job alone
```

Rules that are easy to get wrong:

- **A queued job must terminate on its own.** `just soak` runs until stopped; a
  job that does that blocks everything behind it forever. `tools/jobs/` scripts
  use a bounded batch count instead.
- **The runner sources ROS, jobs do not.** That is what makes the `set -u` trap
  unreachable rather than merely known. Do not add sourcing to a job script.
- **`queue-add` runs `sync` first**, so queueing rsyncs the working tree onto the
  VM *under whatever is running.* Usually wanted; but an in-flight batch loop can
  pick up edited code at its next iteration.
- **Check the runner is alive, not just the queue.** A job in `running` whose log
  already says `EXIT` means the runner died between the two.

## The trajectory generator, and why its design changed

`agx_planning/tuning/shape.py` (pure, 27 tests) + `tools/sample_eval_trajectories.py`.

The point is not "more plans" — it is that **every per-shape claim rests on one
plan of that shape**, which 2026-08-15 turned from a worry into a demonstrated
error. It works: 320 plans, **zero straight lines**, against a random-goal
library that was ~64% straight.

It was first written to screen candidates by predicted tortuosity (a cheap grid
route, ranked by turning). **Validating the proxy against all 100 recorded plans
killed that**, and the number is worth remembering:

    length +0.99   straightness +0.96   total_abs_turn +0.30   sign_changes +0.34

The cheap route knows *where* a plan goes and not *how* it turns. Smoothing the
lattice staircase raises label agreement 15% → 52% — but only because both
distributions shift toward STRAIGHT, i.e. agreement with no predictive power,
which would have passed for success. So the screen ranks on **blocked line of
sight, detour and pivot demand** (all downstream of the +0.96/+0.99 signals) and
**shape is labelled from the solved plan.** Do not re-propose ranking candidates
by predicted turning.

**And the label from the solved plan is itself only a ranking aid** — it cannot
separate a hairpin from two same-sign 90° corners (2026-08-15). Fine for
stratifying a sample, never for a claim. Render the plans.

## Still open, unchanged

- `sand` is unbuilt and blocked on the friction-anisotropy question: under
  min-combination with an isotropic ground, any ground below the wheel's `mu2`
  gives ratio 1, so "slides but still steers" is inexpressible at low friction.
  A question for the advisor.
- chi is a property of the **surface** (1.36 to 25.4 across the sweep), so it
  cannot be a constant on a floor with ice/sand zones.
- The S-curve's **deterministic** bad spike at `r_omega=3.5` (2.548–2.552, sd
  0.001, 100% of 30) is unexplained. The low ladder adds a second one at
  `r=0.5` (2.566, sd 0.014), so the S has *two* isolated bad rungs in `r` with
  good rungs on either side — the same non-monotone signature the U-turn showed
  in `q`, on a different shape and a different gain.
- 36% of PMP solves fail on constructed start/goal pairs (110 timeouts, 70 mesh
  exhaustion of 500). Fine for building a library; **not** fine for an online
  re-join solver, and worth understanding before item 5 leans on it.

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
runaway; no bootstrapped value ⇒ no `critic_loss` divergence; no reward shaping;
each label is an exact optimum rather than a noisy return; and **data generation
needs no Gazebo at all**, so it is CPU-parallel. Gazebo returns only for
validation, on the existing eval set and comparison harness.

| layer | rate | cost | job |
| --- | --- | --- | --- |
| PMP | once/goal, offline | ~1.84 s (measured 2026-08-15) | optimal plan on the **nominal** surface |
| RL re-planner | exceptional | one MLP forward pass | reference infeasible ⇒ emit a short re-join |
| TVLQR | 50 Hz | precomputed gains | track whichever reference is active |

**SETTLED 2026-08-07 (user):** corrections run **on top of the frozen PMP path**.
A full PMP re-solve is reserved for situations that cannot be corrected around at
all — the example given was an unexpected wall. So the frozen plan is the default
reference and re-planning is an exceptional event, not a control layer running at
some rate. Two consequences:

- the RL re-planner's job is bounded — *re-join the existing path*, never *find a
  new one* — a much smaller function to approximate;
- a full re-solve needs a **trigger**, and "the reference is infeasible" is a
  different test from "we are far off it". Nothing implements that yet.

Open: the PMP solver needs a "re-join from an arbitrary state onto the nominal
path" boundary condition (a BC change, not a new solver). Its per-solve cost is
now measured and is not the obstacle; the **36% failure rate** on fresh
start/goal pairs might be.

## Handling chi when it is not constant

1. **The planner keeps one nominal chi** — correct by definition, since
   "nominal" means the surface it was measured on.
2. **chi becomes a measured signal online** in the correction layers:
   `chi_hat = omega_ideal(from wheel commands) / omega_measured(gyro)`, i.e.
   `slip_ident`'s computation run recursively. Feed it to TVLQR (its
   linearization uses `track_eff = track * chi`) and to the policy as an input.
   **Observability caveat:** undefined when `omega ≈ 0`, so it needs a validity
   gate that holds the last value.
3. **Reactive by construction.** You learn the ice is slippery by sliding on it.
   Pre-emptive routing needs perception and is out of scope. (User confirmed:
   "that matches our original idea".)

## Measuring chi on the real robot

The intended use of `slip_ident` — it references the **gyro**, so nothing about
the method is sim-specific (`calibrator.py` cannot substitute: it compares
commands against `/odom` and both sides share the missing slip term).

- **`cmd_mode:=wheels` is sim-only.** On the real Scout use `cmd_mode:=twist`,
  which yields `chassis_gain_omega` — chi folded with the firmware's conversion.
  Do not paste that into `PlannerConfig.slip_chi`.
- Drive **arcs at several radii**, both directions. Spins are load-dependent and
  reported separately; do not fit chi to them.
- **It cannot run against an unthrottled sim** — `make rl-sim` puts `/imu/data`
  at ~3295 Hz and the node drops nearly every sample.
