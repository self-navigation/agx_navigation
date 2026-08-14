# Handover — 2026-08-14 (evening)

**Read this first, and keep it current.** It is the primary record of what we
are doing; CLAUDE.md's "Current work" section is the cumulative record of what we
have *established*. This file describes **now**: what is running, what is
half-finished, what to do next, and the reasoning behind decisions that have not
yet become findings. Rewrite it rather than appending.

**This session in one line:** the `r_omega` ladder came back and says **`r` is
nearly a free parameter** — flat on five of seven shapes, with the adopted 2.618
winning through **one plan** — and the queue that was supposed to keep the box
busy had killed itself, so half a day was lost again.

**FIRST THING NEXT SESSION:** `just queue-status`, then read the jobs below in
order. **There is one question for you (the user) at the bottom, under "For the
user".**

## What happened this session

1. **The runner was dead and job 20 never started.** `jobq.sh` ran each job in a
   **brace group** ending in `exit $rc` — not a subshell, so that `exit` ended
   the runner itself. Job 10 finished at 05:00 UTC, wrote `EXIT rc=0`, and the
   runner died on the next statement; job 20 sat pending for 13 h. Fixed
   (subshell), and the signature is written into CLAUDE.md so it is recognisable:
   **a job in `running` whose log already says `EXIT` + runner NOT RUNNING**.
2. **Read the `r_omega` ladder** (1035 usable rollouts). Full table in CLAUDE.md.
   Headline: `r` does nothing on straight, corner, zigzag, tight V and loop
   across a 5× range; the aggregate's preference for 2.618 is the U-turn alone
   (100% bad at every other rung, 20% at 2.618). **2.618 is an isolated notch —
   the opposite of `q=0.276`, which sat on a wide plateau.**
3. **Found two bugs in the traced-soak path**, which is why the ladder has no `J`
   numbers. `--trace-every 5` aliased against the 35-rollout cycle (only 7 of 35
   cells ever traced), and tracing was never turned off between traced rollouts,
   so each file held ~5 rollouts from ~5 different cells. Both fixed
   (`disable_trace()`, subsample by cycle). **Every existing `J` number is safe**
   — they came from `variance_probe`, which traces every rollout.
4. **Queued three jobs** (below) and rendered `figures/2026-08-14/`.

## Do this next, in order

1. **Read `30_r_ladder_low.sh`** — it decides whether the adopted `r_omega`
   survives. The ladder swept r ∈ [1.0, 5.0]; the gain it *replaced* is r=0.25,
   below the floor. So we still do not know whether r's half of the joint move
   bought anything outside `floor_6_00031`. This sweeps {0.25, 0.5, 1.0, 2.618}
   at q=0.276, carrying 2.618 in the same process so the ladders join without
   assuming cross-run comparability.

       scp -F ssh_config agx:~/soak_r_ladder_low.jsonl soak_data/
       rsync -a -e "ssh -F ssh_config" agx:~/r_ladder_low_traces/ r_ladder_low_traces/
       .venv/bin/python tools/score_sweep.py --from-jsonl soak_data/soak_r_ladder_low.jsonl \
           --trace-root r_ladder_low_traces --plans traj_data

   **How to read it.** Exclude the U-turn from the aggregate first — that is the
   whole point. If r=0.25 is flat with r=2.618 on the other six shapes, then
   `r_omega` is a free parameter that we tuned to one plan's notch, and the
   honest write-up claim is that **the tuning result is a `q_cross` result**.
   That is not a reason to un-adopt 2.618 (it costs nothing elsewhere and buys
   the U-turn), but it is a reason not to *claim* it. If instead r=0.25 is
   visibly worse across the six, the joint move stands as a joint move.

   This is also the **first traced soak since the trace fix**, so scoring it in
   `J` doubles as the check that the fix works. If `score_sweep` reports
   `max|e_cross|` wildly different from the JSONL's own `max_cross` for the same
   rollout, the fix did not take — that mismatch is exactly how the bug was
   caught.

2. **Read `40_uturn_generality.sh`** — the real prize, and the reason the v2
   library was built. The `q ∈ [0.276, 0.4]` basin with near-vertical walls is
   5906 rollouts of `floor_6_00031` and nothing else. This runs the same rungs
   {0.2, 0.276, 0.4, 0.5} on up to 4 other library U-turns **plus 00031 as a
   control in the same run**, n≈60 per cell.

   - basin reproduces on the new U-turns ⇒ property of the SHAPE, notch story
     stands;
   - basin only on 00031 ⇒ `q=0.276` was adopted partly on a coincidence, and
     CLAUDE.md's U-turn sections want rewriting.

   **Before trusting it, look at the plans it picked** (the job prints their
   paths). The labels come from the solved plan, not from the screen, but
   `label()`'s own docstring calls itself a ranking aid; render them and confirm
   they are U-turns by eye. This is the step that caught the last automatic
   labelling being wrong.

3. **Read the v2 library itself** (`20_generate_v2_library.sh`, running now) —
   fetch `~/traj_data_v2`, render a gallery, **look at it**, and stratify ~5
   plans per shape into a **second** eval set in `config/eval_trajectories.yaml`,
   *added* alongside the seven rather than replacing them. The seven stay the
   fast search set (mean-of-3 tuning must stay ~105 s/eval); the stratified set
   is for **claims**. Swapping would silently re-baseline every number in
   CLAUDE.md.

   Its log also gives the **first read on PMP solve cost per plan**, which is
   unmeasured and sets the data budget for the RL re-planner. Write the number
   down whatever it is.

4. **Read `25_uturn_notch_edge.sh`'s traces** — 10 rollouts either side of the
   U-turn's `q` wall (0.4 vs 0.5), every one traced. Run `tuning/trace_diff.py`
   on a good/bad pair: `cmd*` moving first means our controller, state moving
   first under an equal command means the plant. All the U-turn's deviation is
   produced at ONE corner in the last quarter of the run and nobody has explained
   it. ~5 min of reading, the data is already paid for.

5. **Then reconsider the objective.** `J` is computable and better behaved than
   the 7-shape mean of `max|e_cross|`; the open question is whether to *tune*
   against it, which needs `J` inside `objective.py` rather than as a post-hoc
   script. The library sweep strengthens the case: `J` and metres disagree on 24
   of 51 plans.

6. **Measure the re-join PMP solve time** before committing to any architecture.
   `200388e` added acados and `0454d3d` removed it, after which online mode was
   documented as infeasible — so "cannot solve online" is currently a statement
   about scipy, not about the problem. The source's justification for offloading
   cites **DShot and PWM** frame budgets, i.e. a flight controller, not a Jetson.

7. **Do not re-run a wide gain search yet**, and in particular do not start one
   before item 1 — a search that includes `r_omega` as a free axis is mostly
   resolving one plan's notch.

## For the user — one question I could not settle alone

**Should `r_omega` stay in the tuned story at all?**

The measurement says it does essentially nothing on six of seven shapes, and its
one win is a notch on the single U-turn plan we happen to own. Three defensible
positions, and it is a judgement about what we are willing to claim rather than
something another run resolves:

- **keep 2.618, describe the result as `q_cross` tuning** — honest, and costs
  nothing, since 2.618 is free elsewhere and buys the U-turn. My preference.
- **keep 2.618 and claim the pair** — only defensible if job 40 shows the basin
  is a property of U-turns rather than of `floor_6_00031`.
- **revert to r=0.25** — cleanest story (one tuned gain), but throws away the
  U-turn win and re-introduces the tight V's 10.5% bad mode.

Job 30 (r=0.25 in the same conditions) and job 40 (other U-turns) between them
supply the evidence for all three; the choice of what to *claim* is yours.

## State

- **VM:** one headless `gz sim` on `rl_corrector.world` (ground mu=1.0), started
  fresh 2026-08-12, still up. The jobq runner is up **with the subshell fix**.
- **RUNNING: `20_generate_v2_library.sh`** — screens 1200 pairs each on floors 1
  and 6, keeps 500, PMP-solves into `~/traj_data_v2/`; candidates in
  `~/candidates_v2.json`. Started 18:08 UTC 2026-08-14, screening at ~200 per
  8 min ⇒ ~1.6 h for both maps, then an unmeasured solve phase. Log
  `~/jobq/logs/20_generate_v2_library.log`. Needs no Gazebo.
- **PENDING, in order:**
  - `25_uturn_notch_edge.sh` — ~2 min, 20 rollouts, all traced,
    `~/uturn_edge.jsonl` + `~/uturn_edge_traces/`.
  - `30_r_ladder_low.sh` — ~1.4 h, 840 rollouts, `~/soak_r_ladder_low.jsonl` +
    `~/r_ladder_low_traces/`.
  - `40_uturn_generality.sh` — ~2 h, 1200 rollouts,
    `~/soak_uturn_generality.jsonl` + `~/uturn_generality_traces/`. **Exits 1
    and is skipped** if job 20 produced fewer than 2 UTURN plans; that is
    deliberate — an empty library is a reason to skip, never to idle the box.
- **Fetched and analysed this session:** `soak_data/soak_r_ladder.jsonl` (1050
  rows) and `r_ladder_traces/` (210 files, **unscoreable — see the trace bugs**;
  keep them only as evidence of the bug, they measure nothing).
- **The U-turn sub-ladder** (5906 rollouts) is archived on the VM as
  `~/soak_20260813_uturn_subladder.jsonl` and fetched. Metres only — it predates
  `--trace-dir` on `soak.py`.
- Archived on the VM: `~/soak_20260813_ladder.jsonl` (q ladder, 1047),
  `~/soak_20260813_twopoint.jsonl`, `~/gaincheck.jsonl` + `~/gaincheck/`,
  `~/jsweep.jsonl` + `~/jtraces/`, `~/libsweep.jsonl` + `~/libsweep/`,
  `~/soak_r_ladder.jsonl` + `~/r_ladder_traces/`.
- **Caches on the VM, current plant, safe to resume onto:**
  `~/tvlqr_tune_v4_newplant.jsonl` (+ `~/tvlqr_tuned.json`),
  `~/validate_20260812_{tuned,default}.jsonl`, `~/qwall_20260812.jsonl`,
  `~/local2d_20260812.jsonl`. All fetched into local `tune_data/`.
- **Poisoned caches, do not resume onto them:** `~/tvlqr_tune_v2.jsonl`,
  `~/tvlqr_tune_v3.jsonl`, `~/tvlqr_tune.jsonl` — abandoned plants.
  `PLANT_VERSION` will refuse them.
- **Two similarly-named plan directories on the VM.** The eval config points at
  `~/pmp_trajectories_v2` (the existing 100-plan library); the new one being
  built is `~/traj_data_v2`. Job scripts read the path out of
  `config/eval_trajectories.yaml` rather than guessing.
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

## The queue is the way to run long things

`just queue-start` once, then `just queue-add <job>.sh` at any time — including
while another job runs. Jobs live in `tools/jobs/`, logs in `~/jobq/logs/`.

```
just queue-status              # runner, running/pending/done/failed, recent logs
just queue-add 30_r_ladder_low.sh
just queue-log 30_r_ladder_low # a specific job's log
just queue-stop                # stops the RUNNER; leaves the in-flight job alone
```

Rules that are easy to get wrong:

- **A queued job must terminate on its own.** `just soak` runs until stopped; a
  job that does that blocks everything behind it forever. `tools/jobs/` scripts
  use a bounded batch count instead.
- **The runner sources ROS, jobs do not.** That is what makes the `set -u` trap
  unreachable rather than merely known. Do not add sourcing to a job script.
- **`queue-add` runs `sync` first**, so queueing rsyncs the working tree onto the
  VM *under whatever is running*. Usually wanted; but an in-flight batch loop can
  pick up edited code at its next iteration.
- **Check the runner is alive, not just the queue.** See the failure signature in
  CLAUDE.md — a job in `running` whose log says `EXIT` means the runner died.

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

## Still open, unchanged

- `sand` is unbuilt and blocked on the friction-anisotropy question: under
  min-combination with an isotropic ground, any ground below the wheel's `mu2`
  gives ratio 1, so "slides but still steers" is inexpressible at low friction.
  A question for the advisor.
- chi is a property of the **surface** (1.36 to 25.4 across the sweep), so it
  cannot be a constant on a floor with ice/sand zones.
- The S-curve's **deterministic** bad spike at `r_omega=3.5` (2.548–2.552, sd
  0.001, 100% of 30) is unexplained. New this session, and the only place a
  non-U-turn shape cares about `r` at all.
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
