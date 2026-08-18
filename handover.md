# Handover — 2026-08-18

**Read this first, and keep it current.** It is the primary record of what we
are doing; CLAUDE.md's "Current work" section is the cumulative record of what we
have *established*. This file describes **now**: what is running, what is
half-finished, what to do next, and the reasoning behind decisions that have not
yet become findings. Rewrite it rather than appending.

**This session in one line:** the three overnight jobs had been **finished and
unread for ~62 h** (nothing was stuck — the queue was idle), and reading them
says **`q_cross ≈ 2.5` beats the adopted 0.276, `q` and `r` interact, and job
60's `J`-tuned point is an artifact**; one confirming job (100) is running now,
and separately **the ROS 2 runtime pipeline was driven end to end for the first
time** and the tooling to do that unattended now exists.

**FIRST THING NEXT SESSION:** `just queue-status`. If job 100 is done, fetch and
score it (line under "RUNNING NOW"), pick the gain pair, and **edit
`TVLQRConfig` — that is the one open decision and it should be closed.**

## What the three unread jobs said

Full tables in CLAUDE.md, "The broad ladders". The short version:

- **Job 70 (`q` ladder, 1200 rollouts, mean-of-5).** `J` is FLAT across a 15x
  range of `q` — it cannot decide this. Arrival can, and **`q=2.5` wins it**:
  vs the adopted 0.276, `final_err` 34/40 (p<0.0001) and max|e_cross| 32/40
  (p=0.0002); it beats `q=1.5` too (27/40, p=0.038). Miss rate 10.5% vs 20.5%.
- **Job 80 (`r` ladder at `q=1.5`).** **`q` and `r` INTERACT.** At `q=1.5` the r
  story inverts: `r=0.25` is best on metres and arrival, and the r=0.5→1.0
  threshold that justified moving off 0.25 is **absent**. Every `r` claim in
  CLAUDE.md is now scoped to `q=0.276`.
- **Job 90 (validate job 60).** Job 60's seven-plan `J`-tuned point
  (`q=0.880, r=25.6`) is the **worst of three arms on every axis** on the broad
  40, and did not even beat the adopted point on its own search set. **Do not
  adopt it.** Second time a seven-plan optimum evaporated on independent plans:
  **a seven-plan search cannot resolve the gains, in either currency. Do not run
  another one.**

## RUNNING NOW

**`100_broad_r_at_q25.sh`** — started 08:29 UTC 2026-08-18. `r ∈ {0.25, 1.0,
2.618, 5.0}` at **q=2.5**, plus `1.5/2.618` and `0.276/2.618` carried **in the
same process** as controls, 40 broad plans, mean-of-5, 1200 rollouts, 4 workers.
Nothing is known about `r` at the `q` we are about to adopt, and job 80 proved
the two do not separate — so this is the run that settles the pair rather than
one axis. It exists to be the LAST gain job.

**Expect ~1.5–2 h, not the 40 min the job comment claims** — 6 arms over 4
workers deals 2/2/1/1, so the long pole is 400 rollouts. Fix the estimate in
the script if it matters; the fan-out itself is fine.

Fetch and score (no traces: `j_total` is inline in every row now, from the
online `EpsilonAccumulator`, so trace scoring is obsolete for gain work):

    scp -F ssh_config agx:~/soak_broad_r_at_q25.jsonl soak_data/

Then aggregate **geometrically on `J`**, arithmetically on the rest, and compare
arms with a **paired sign test over the 40 plans** — means alone did not
separate signal from noise in jobs 50/70/80. Read it on **`final_err` and `J`**;
max|e_cross| ranks the old default best while it spends ~3x the control.

**How to read it.** If `q=2.5` holds up against the two controls, adopt the best
`(q, r)` in `TVLQRConfig` and stop tuning. If `r=0.25` wins at `q=2.5` as it did
at `q=1.5`, that is a coherent result and not a surprise — say plainly in the
write-up that the original "move off r=0.25" was a `q=0.276` phenomenon.

## The ROS 2 stack is now drivable unattended (built 2026-08-18)

**The runtime pipeline ran end to end for the first time.** Every corrector
number in this repo came from `GazeboBridge`, which bypasses the ROS graph
entirely, so `vector_field` → `pmp_planner` → `runtime_corrector` → playback →
controllers had never been exercised during the corrector work. It was expected
to have bit-rotted. **It had not** — it came up ready on the first attempt and
drove a sampled goal to completion:

    [random_goals] 3 subscribers matched; settling 3.0s before publishing
    [random_goals] goal 1/1: (-9.77, -2.22)
    [random_goals] goal 1 reached

That closes the standing worry that weeks of tuning had been validated only
inside the measurement harness.

**Four things exist now that did not.** All are worker-scoped, so they run
beside the queue rather than waiting for it.

| tool | what it is for |
| --- | --- |
| `tools/stack_ready.py` | readiness/health probe: is the stack up, and if not, WHICH check failed |
| `tools/fixture_up.sh` / `just fixture-up [worker]` | bring the fixture up, wait on the probe, **tear down and retry** if it did not come up |
| `just screenshot [out]` | capture the VM desktop and pull the PNG back — **the Moonlight substitute** |
| `tools/drive_goal.py` | publish one goal safely, wait for the terminal outcome, score arrival against ground truth |

**`just screenshot` is the important one for working without you.** Moonlight
needs the VPN, which has been down ~5 days; the jump route carries ssh but not
a video stream. `DISPLAY=:0 import -window root` needs no GUI interaction and
no X client of ours, so it works identically over either route. Verified: the
captured desktop shows RViz (corrector status, ground truth + goal, vector
field) and the Gazebo GUI with `scout_mini` and the spawned surface patches.

**On the launch races.** `just fixture-up` does not fix them — it *detects*
them and restarts, which is the honest thing until the ordering is fixed
properly. The probe is what makes that possible, and it is deliberately split
into GRAPH checks (does a publisher/subscriber exist) and LIVENESS checks (is
data flowing): a node that crashed after advertising passes the first forever,
and only the second catches it. `/clock` is checked by VALUE — a paused or dead
Gazebo keeps a `/clock` publisher, so requiring the stamp to ADVANCE is the only
test that separates "world stepping" from "world stopped".

**No race actually fired in this session** (ready in 24 s, first attempt), so
the failure ordering is still uncharacterised. `fixture_up.sh` prints which
check failed on every retry — the next few restarts will accumulate that for
free, and *that* is the evidence needed to fix the launch file rather than
paper over it. Do not attempt the fix before there is a failing case to read.

**Two traps found while building this, both worth keeping:**

- **A wrong QoS looks like a broken stack, not like a QoS problem.**
  `drive_goal.py` was first written with `TRANSIENT_LOCAL` on `/goal_pose`,
  which is INCOMPATIBLE with the stack's `VOLATILE` publishers: the goal went
  out, the robot drove, and the tool sat waiting for a sentinel it could not
  physically receive. `stack_ready.py` now checks durability consistency across
  all `/goal_pose` endpoints. The stack's profile is RELIABLE + VOLATILE,
  depth 1 — match it.
- **Interrupting a local ssh does NOT kill the remote process.** A locally
  timed-out `drive_goal` kept running on the VM, kept a mismatched subscriber in
  the graph, and had already driven the robot to the goal the *next* run then
  measured — which is why that run "arrived" in 4.6 s. This is the documented
  trap and it bit anyway. `pgrep -af` before trusting a fast result.

**A REAL BUG was found in `runtime_corrector` and fixed** — a plan that fails
at chunk 0 (the ~36% BVP mesh-node failure) never set `active_traj_id`, so the
action result was dropped, `_finish()` never ran, and no zero command and no
completion sentinel were ever published. Every client hung for its own timeout
with the robot stationary. Full account in CLAUDE.md, "A failed plan hung every
client". **Verified in the wild**, in a six-goal `random_goals` session after
the fix: goal 2 hit the BVP failure, logged the new
`Plan for traj_id=2 produced no trajectory ... Stopping and clearing the goal`,
and the driver advanced to the next goal **0.4 s later** off the sentinel —
where before it would have burned its full dwell. `drive_goal` likewise
terminates in 28.7 s (`finished`) rather than sitting out its 180 s timeout.

**Side effect worth knowing: `random_goals` now says "reached" for plans that
never ran.** Its log is driven by the sentinel, and the sentinel means "nobody
is pursuing a goal" -- so in the verification session `goal 1 reached` appeared
**1.0 s** after the goal went out, for the goal whose BVP solve had just failed.
That ambiguity always existed; the fix makes it fire promptly instead of after a
90 s dwell, which makes it far easier to misread. **Do not read `random_goals`'
"reached" as arrival** -- it is "no longer being pursued". Anything that needs
the distinction must read the action result, or use `tools/drive_goal.py`, which
scores the final distance against ground truth. Worth fixing in `random_goals`
itself: it could subscribe to the action result and log "failed" honestly.

**A second observation from the same session, not yet explained:** of six goals,
five completed and **goal 4 `(7.77, -5.02)` did not finish within its 90 s
dwell** -- neither arriving nor failing. That is a third outcome (plan solved,
playback did not terminate) and nobody has looked at it. It is the natural next
thing to point `drive_goal` at.

**The pipeline itself was healthy; the bug was in the failure path.** No amount
of successful driving would have exposed it — a stack that works is not evidence
about what it does when a component says no.

**One bug fixed in `drive_goal.py` that would have produced plausible numbers:**
it composed `map->odom` with `odom->base` by ADDING translations. Under
`localization:=truth` the first carries the rotation that corrects odometry
drift, so the second must be rotated into the map frame first. The symptom was a
reported 15 m path over 4.6 sim-seconds (3.3 m/s, above what the chassis can
do). Now composed properly with yaw.

## Do this next, in order

1. **Close the gain decision and STOP TUNING.** Read job 100, pick the pair,
   edit `TVLQRConfig`. The evidence is already overwhelming that (a) the move
   off the old default is right, (b) `q=0.276` is on the bad edge of a wide
   plateau, and (c) a seven-plan search cannot resolve the rest. Job 100 exists
   to choose `r` at the new `q`, not to reopen anything. **Do not queue another
   gain job after it** — three independent broad-set runs now agree that `J` is
   flat inside the plateau, so further search resolves noise.

   Note for the write-up: the honest claim is a **robustness trade**, not
   "tuning halves deviation". It pays on hard trajectories, is ~free on easy
   ones, and where the old default loses is **control effort** (~3x), which is
   the SVCM prescription of p. 78 turning up as a measurement.

2. **Watch a fixture run properly, now that watching is possible.** `just
   fixture-up 5` + `just screenshot` works, but the Gazebo GUI camera is not
   pointed at the robot, so the screenshot shows the building rather than the
   demo. Two small things to fix, both named in the roadmap already: point the
   camera, and move the demo patch layout out of the hardcoded JSON literal in
   `gz_sim.launch.py` into a `PATCH_LAYOUT` config file — the current layout
   puts `icy` under the spawn point, which is right for testing and useless for
   a demo (you want clean → one patch → excursion → recovery). Keep demo patches
   just ABOVE the wheel's `mu2=0.45` knee; below it there is no steering at all,
   which reads as a broken robot rather than a slipping one.

   With workers, the before/after is cheap: `just fixture-up 1 tvlqr true` and
   `just fixture-up 2 identity true` side by side on one desktop.

3. **Characterise the launch races, then fix them.** `fixture_up.sh` retries and
   prints the failing check each time, so the evidence accumulates for free.
   Nothing fired this session (ready in 24 s, first attempt), so there is **no
   failing case to fix yet** — collect a few before touching `main.launch.py`.
   The probe already knows the shape of the answer: map, `map->odom`,
   `/goal_pose` subscriber count, `/clock` advancing.

4. **Read `25_uturn_notch_edge.sh`'s traces** — 10 rollouts either side of the
   U-turn's `q` wall, already paid for, in `soak_data/uturn_edge.jsonl` +
   `~/uturn_edge_traces/`. `tuning/trace_diff.py` on a good/bad pair: `cmd*`
   moving first means our controller, state moving first under an equal command
   means the plant. ~5 min. Low value now that the basin is known to be one
   plan's — a curiosity, not a mechanism.

5. **Stratify a second eval set** into `config/eval_trajectories.yaml`, *added*
   alongside the seven rather than replacing them. The seven stay the fast
   search set; the broad 40 are for **claims**. Swapping would silently
   re-baseline every number in CLAUDE.md. Job 50's 40 plans are ready-made —
   `~/broad_eval_plans.txt` on the VM, and `tools/jobs/broad40.txt` in the tree.

6. **Read [docs/corrector-design.md](docs/corrector-design.md)** before starting
   any RL work. It separates the two things RL could compress, which the
   previous RL effort conflated: the **template library** (~9 continuous dims,
   needs a network) and the **cost matrices per surface class** (one categorical
   index, needs a table — and is our existing tuner, run once per surface). It
   recommends **retiring the SAC residual** rather than repairing it, and
   identifies the escalation trigger as `CorrectionDiagnostics.saturated_*`,
   already implemented and unused.

7. **Build the re-join re-planner — THE NEXT REAL GOAL.** Full plan in the
   design doc ("The build plan for Level A"). **Phase 0 first, because it can
   invalidate the other three:** add the re-join boundary condition (`x(t0)` =
   actual off-plan state, `x(t0+T_w)` = the plan's state at index `k + T_w/dt`,
   **all five components including wheel speeds**) and solve ~200 sampled
   re-join problems. **The number that decides everything is the solve failure
   rate** — the library build failed 36%. Inherit that and there is no reliable
   teacher and the supervised plan is wrong. Offline, no Gazebo, cheap. Do it
   before any training code.

   Then phase 1 (~100k **randomly sampled** solves, never a grid — ~55 core-hours
   at the measured 1.84 s/solve, against ~5000 for a 5-points/axis grid), phase 2
   (**DAgger, not RL** — with the expert's answer in hand the difference IS the
   exact gradient, so a reward estimator is strictly worse), and optionally
   phase 3 (RL fine-tuning above the teacher, reward `-epsilon.step_cost(...)`).

   Why wheel speeds must be in the terminal BC: playback resumes by feeding the
   plan's commands from index `k+m`, which assume the wheels already turn at the
   plan's rate — matching position but not phase reproduces the documented stall
   bug deliberately. And because playback is time-indexed, landing at
   `k + T_w/dt` keeps the robot on SCHEDULE; `m` is therefore determined by
   `T_w`, and `T_w` is the one free scalar.

8. **Do not lower the ground plane to model a slippery floor.** Standing rule.

## State

- **VM:** the long-lived headless `gz sim` (default partition, `rl_corrector.world`,
  ground mu=1.0) is still up. Job 100 owns **workers 1-4**; a **fixture runs in
  worker 5**. The jobq runner is up and has now run six jobs without dying.
- **Queue:** `100_broad_r_at_q25.sh` running, nothing pending.
- **`TVLQRConfig` is UNCHANGED** (`q_cross=0.276`, `r_omega=2.618`) pending job
  100. This is the one decision waiting on a person.
- **Fetched and analysed this session:** `soak_data/soak_broad_q.jsonl` (1200),
  `soak_data/soak_broad_r.jsonl` (600), `soak_data/soak_validate_J_broad.jsonl`
  (600). No traces were needed — `j_total` is inline in every row.
- **Traces NOT fetched and not needed:** `~/broad_q_traces/` (1200),
  `~/broad_r_traces/` (600), `~/validate_J_broad_traces/` (600) are still on the
  VM. They are only useful for re-scoring in a currency other than the ones
  already recorded; job 100 was deliberately run **without** traces.
- **A build ran while job 100 was in flight** (`just remote-build`, to pick up
  the corrector fix). It did not disturb it — the soak workers are long-lived
  Python processes that imported their modules at start, and the change was to
  `runtime_corrector`, which the soak path does not use. Still: **prefer not to
  build under a running job**, and check `wc -l` on the per-worker files after
  one, as was done here.
- **Caches on the VM, current plant, safe to resume onto:**
  `~/tvlqr_tune_v4_newplant.jsonl` (+ `~/tvlqr_tuned.json`),
  `~/validate_20260812_{tuned,default}.jsonl`, `~/qwall_20260812.jsonl`,
  `~/local2d_20260812.jsonl`, `~/tvlqr_tune_J.jsonl` (job 60's `J` search — safe,
  but see job 90: **its answer is an artifact, do not resume onto it expecting
  a better one**).
- **Poisoned caches, do not resume onto them:** `~/tvlqr_tune_v2.jsonl`,
  `~/tvlqr_tune_v3.jsonl`, `~/tvlqr_tune.jsonl` — abandoned plants.
  `PLANT_VERSION` will refuse them.
- **Unusable, keep only as evidence of the bug:** `r_ladder_traces/` (210 files)
  — written before the `--trace-every` fixes.
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
  metric covers — though `just screenshot` now covers part of it.

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

**The queue stays single-lane on purpose (user's call, 2026-08-15).** Parallelism
belongs INSIDE a job — one job at a time, each free to fan out over several
workers — not in a multi-lane runner. That keeps the queue's one real guarantee
(a job never shares a world with another job) while still using the box, and it
avoids per-lane locks, per-lane logs, and the fact that `just queue-add` rsyncs
the tree under everything currently running. So: when a job is big enough to
want parallelism, give *that script* a worker loop; do not touch `jobq.sh`.

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
