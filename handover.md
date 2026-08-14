# Handover — 2026-08-14

**Read this first, and keep it current.** It is the primary record of what we
are doing; CLAUDE.md's "Current work" section is the cumulative record of what we
have *established*. This file describes **now**: what is running, what is
half-finished, what to do next, and the reasoning behind decisions that have not
yet become findings. Rewrite it rather than appending.

**2026-08-14 in one line:** the tuned gains **generalise across 51 plans** — but
as a *robustness trade*, not a uniform win — and the VM now has a **job queue**,
so idle time between sessions gets absorbed instead of lost.

**FIRST THING NEXT SESSION:** `just queue-status`. Two jobs are queued (below).
Read the `r_omega` ladder first, then the v2 trajectory library.

## What happened this session

1. **Read the 51-plan library sweep** (finished unattended overnight). Full
   tables in CLAUDE.md, "The library sweep". The headline: tuned wins 45/51 in
   `J` and the aggregate holds in both currencies — but in metres the **default
   wins 30 of 51 plans**, because the tuned gains concede ~3 cm on easy plans to
   prevent blow-ups on hard ones. 9.99 m gained against 2.03 m lost.
2. **Found why the `r_omega` ladder never ran.** `tools/queue_r_ladder.sh` had
   `set -u` on across `source /opt/ros/jazzy/setup.bash`, which reads
   `AMENT_TRACE_SETUP_FILES` while unset. It waited for its predecessor
   correctly, logged `starting the r_omega ladder`, and died on the next line.
   **The VM sat idle ~17 h.** Written into CLAUDE.md's VM section.
3. **Built a job queue** (`tools/jobq.sh`, `just queue-*`, jobs in
   `tools/jobs/`) to replace bespoke chain scripts. See below.
4. **Built the trajectory generator** (queue item 8) — and its design changed
   under validation, which is the interesting part. See below.
5. **Measured `trim_pivot`'s threshold** instead of inheriting it: 0.30 m was
   too small, 0.70 m is where the label counts plateau.

## The queue is the new way to run long things

`just queue-start` once, then `just queue-add <job>.sh` at any time — including
while another job runs. Jobs live in `tools/jobs/`, logs in `~/jobq/logs/`.

```
just queue-status              # runner, running/pending/done/failed, recent logs
just queue-add 20_generate_v2_library.sh
just queue-log 10_r_ladder     # a specific job's log
just queue-stop                # stops the RUNNER; leaves the in-flight job alone
```

Two rules that are easy to get wrong:

- **A queued job must terminate on its own.** `just soak` runs until stopped;
  a job that does that blocks everything behind it forever. `tools/jobs/` scripts
  use a bounded batch count instead.
- **The runner sources ROS, jobs do not.** That is what makes the `set -u` trap
  unreachable rather than merely known. Do not add sourcing to a job script.

## Do this next, in order

1. **Read the `r_omega` ladder** when it finishes (~1.8 h from 06:15 UTC on
   2026-08-14). Question: `r` moved 0.25 → 2.618 as half of a **joint** move, so
   we know the pair is good and nothing about whether 2.618 is mid-basin or on a
   wall — `q` turned out to be on one. `q=0.276` is held fixed so `r` is
   separable, mirroring the `q` ladder. Read it with:

       scp -F ssh_config agx:~/soak_r_ladder.jsonl soak_data/
       rsync -a -e "ssh -F ssh_config" agx:~/r_ladder_traces/ r_ladder_traces/
       .venv/bin/python tools/score_sweep.py --from-jsonl soak_data/soak_r_ladder.jsonl \
           --trace-root r_ladder_traces --plans traj_data

2. **Read the v2 trajectory library** (queued behind the ladder). It screens
   1200 pairs per map on floors 1 and 6, keeps the 500 best, and PMP-solves
   them into `~/traj_data_v2/`, labelling each from the SOLVED plan. Then:
   - fetch it, render a gallery, and **look at it** — the labels are a ranking
     aid and the picture is the authority, which is how the last automatic
     labelling was caught being wrong;
   - stratify ~5 plans per shape into a **second** eval set in
     `config/eval_trajectories.yaml`, *added* alongside the seven, not replacing
     them. The seven stay the fast search set (mean-of-3 tuning must stay
     ~105 s/eval); the stratified set is for **claims**. Swapping would silently
     re-baseline every number in CLAUDE.md.
   - then the real prize: **re-run the U-turn sub-ladder on 3-5 different
     U-turns.** If the `q ∈ [0.276, 0.4]` basin is a property of U-turns it
     reproduces; if it is a property of `floor_6_00031`, the "notch" story needs
     rewriting and `q=0.276` wants re-examining.

3. **Explain the U-turn's last corner** (unchanged from yesterday). All the
   deviation is produced at ONE corner in the last quarter of the run. Trace
   either side of the notch edge (`q=0.4` vs `q=0.5`) and run
   `tuning/trace_diff.py`: `cmd*` moving first means our controller, state
   moving first means the plant. ~10 min of sim.

4. **Then reconsider the objective.** `J` is computable and better behaved than
   the 7-shape mean of `max|e_cross|`; the open question is whether to *tune*
   against it, which needs `J` inside `objective.py` rather than as a post-hoc
   script. The library sweep strengthens the case: `J` and metres disagree on 24
   of 51 plans.

5. **Measure the re-join PMP solve time** before committing to any architecture.
   `200388e` added acados and `0454d3d` removed it, after which online mode was
   documented as infeasible — so "cannot solve online" is currently a statement
   about scipy, not about the problem. The source's justification for offloading
   cites **DShot and PWM** frame budgets, i.e. a flight controller, not a Jetson.

6. **Do not re-run a wide gain search yet.**

## The trajectory generator, and why its design changed

`agx_planning/tuning/shape.py` (pure, 27 tests) + `tools/sample_eval_trajectories.py`.

The point is not "more plans" — it is that **every per-shape claim rests on one
plan of that shape**. The U-turn notch is 5906 rollouts of `floor_6_00031`.

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

## State

- **VM:** one headless `gz sim` on `rl_corrector.world` (ground mu=1.0), started
  fresh 2026-08-12. The jobq runner is up.
- **RUNNING: `10_r_ladder.sh`** — 7 shapes, `q=0.276` fixed, `r_omega` ∈
  {1.0, 1.8, 2.618, 3.5, 5.0}, 6 batches × 175 = 1050 rollouts (n=30 per cell),
  traced every 5th. Data `~/soak_r_ladder.jsonl`, traces `~/r_ladder_traces`,
  log `~/jobq/logs/10_r_ladder.log`. Started 06:15 UTC, ~1.8 h.
- **PENDING: `20_generate_v2_library.sh`** — screens 1200 pairs each on floors 1
  and 6, keeps 500, PMP-solves into `~/traj_data_v2/`; candidates in
  `~/candidates_v2.json`. Needs no Gazebo but is queued anyway (CPU-heavy, and
  one thing at a time is the rule). Solve time per plan is **unmeasured** — if
  the log shows it is slow, that number is itself worth recording, since it sets
  the data budget for the RL re-planner.
- **Fetched and scored this session:** `soak_data/libsweep.jsonl` (306 rows),
  traces in `libsweep/`, scored into `epsilon_data/libsweep_J.jsonl`.
- **The U-turn sub-ladder** (5906 rollouts) is archived on the VM as
  `~/soak_20260813_uturn_subladder.jsonl` and fetched. Metres only — it predates
  `--trace-dir` on `soak.py`, so it can never be scored in `J`.
- Archived on the VM: `~/soak_20260813_ladder.jsonl` (q ladder, 1047),
  `~/soak_20260813_twopoint.jsonl`, `~/gaincheck.jsonl` + `~/gaincheck/`,
  `~/jsweep.jsonl` + `~/jtraces/`, `~/libsweep.jsonl` + `~/libsweep/`.
- **Caches on the VM, current plant, safe to resume onto:**
  `~/tvlqr_tune_v4_newplant.jsonl` (+ `~/tvlqr_tuned.json`),
  `~/validate_20260812_{tuned,default}.jsonl`, `~/qwall_20260812.jsonl`,
  `~/local2d_20260812.jsonl`. All fetched into local `tune_data/`.
- **Poisoned caches, do not resume onto them:** `~/tvlqr_tune_v2.jsonl`,
  `~/tvlqr_tune_v3.jsonl`, `~/tvlqr_tune.jsonl` — abandoned plants.
  `PLANT_VERSION` will refuse them.
- Both worlds stay at **mu=1.0**. Slipperiness belongs in the wheel pair or a
  patch; a patch below 0.45 deliberately means "no steering".
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

## Still open, unchanged

- `sand` is unbuilt and blocked on the friction-anisotropy question: under
  min-combination with an isotropic ground, any ground below the wheel's `mu2`
  gives ratio 1, so "slides but still steers" is inexpressible at low friction.
  A question for the advisor.
- chi is a property of the **surface** (1.36 to 25.4 across the sweep), so it
  cannot be a constant on a floor with ice/sand zones.
- The RL re-planner architecture (below) is decided but **not started**.

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
| PMP | once/goal, offline | seconds | optimal plan on the **nominal** surface |
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
path" boundary condition (a BC change, not a new solver), and its per-solve cost
needs measuring early since it sets the data budget. **Job 20's log will give a
first read on PMP solve cost** — not the re-join problem, but the same solver.

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
