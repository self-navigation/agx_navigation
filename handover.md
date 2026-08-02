# Handover — 2026-08-02, end of evening session

## TL;DR

Deterministic mode had **never actually been deterministic**. Two bugs, both
found tonight, both fixed. Run-to-run spread on the tuning metric went from
**6.70 m → 0.0013 m**. Every measurement taken before tonight is invalid.

A fresh TVLQR tuning run is **running now** against a new 7-trajectory eval set.
It should be finished by morning.

## First thing to check when you wake up

```bash
just tune-log                      # last 40 lines
just fetch-tune && just plot-tune  # when it has converged
```

It started 2026-08-02 ~21:20 local, ~35 s per evaluation, in tmux `rl:tune` on
the VM, logging to `/tmp/tune_tvlqr.log`. The previous search converged in 132
evaluations, so expect ~75 min — i.e. it was almost certainly done by ~22:40 and
has been sitting finished for hours. Results land in
`/home/programmer/tvlqr_tuned.json`, history in `/home/programmer/tvlqr_tune.jsonl`.

**Baseline to beat: 1.1405 m** mean max|e_cross| at the current
`q_cross=10 / r_omega=0.25`. By eval 6 it had already found 0.9617
(`q_cross=10, r_omega=1.253`) — note that r_omega wants to go **up**, not down
as last time's (invalid) run suggested.

If it crashed: the cache is resumable at zero cost, just re-run `just tune-tvlqr`.
Failed evaluations are never cached, so a crash costs nothing but time.

## What was wrong (the important part)

**1. Every `multi_step` un-paused the world.** `WorldControl.pause` is a plain
proto3 bool, so a request setting only `multi_step` sends `pause: false`, and gz
applies it. The world stepped the n ticks we asked for and then **free-ran** until
the next call, for however long the CPU gave it. Control steps were advancing
0.42 s of sim time instead of `control_dt`=0.1.

Nothing detected this. `lost_steps` and `stale_pose_steps` were both 0 and
correctly so — the world was not dropping steps, it was doing *extra* ones. Every
counter we had was built to catch the opposite failure.

**2. Two wall-clock-paced loops fed variable physics ticks into the reset.** The
teleport confirm loop ran for 0.5 s of *wall* time (12–31 ticks depending on
machine load); the settle ran a fixed 5 steps and left ~2e-5 m of z micro-bounce.
Both now fixed/converged.

Fixes are in `GazeboBridge`: `_world_control` re-asserts `pause=True` on every
multi_step; `_ensure_paused()` verifies the sim clock actually stopped rather than
trusting the unreliable ack; `_set_pose_stepped()` uses a fixed 20-tick budget.

## Where the reproducibility floor is now (10 rollouts, measured)

| metric | sd | verdict |
| --- | --- | --- |
| **max\|e_cross\|** | **0.0002** | the tuning objective — single-sample ranking is now legitimate |
| rms_cross | 0.0154 | fine |
| final_err | **0.2633** | **not reproducible — never use as an objective** |

`final_err` is chaotic: the residual seed is ~1e-13 in the wheels' resting speed,
amplified at a turn reversal where `omega` crosses zero and the skid-steer's
lateral friction switches direction. Genuine physics, not worth chasing. The
tuning objective already uses `max|e_cross|` only — I verified this before
launching.

## What else changed tonight

- **Eval set 3 → 7 trajectories**, one per shape archetype, in
  `config/eval_trajectories.yaml`: straight, corner, S, zigzag, tight V, U-turn,
  loop. Affordable because a rollout now costs ~5 s not ~25 s.
  `floor_6_00042` — the "S-curve" that was really an L — is **dropped**.
- **Gallery paginated** to 24 plans/sheet (`figures/trajectory_gallery_p01..05.png`),
  sorted by turning per metre, with the leading in-place pivot trimmed before
  measuring shape. Looking at it: **only ~20 of the 100 plans have any shape at
  all**, all on page 1. That is the real constraint on the eval set now.
- **New instrumentation**: `GazeboBridge.enable_trace()` writes per-step state
  CSVs; `tuning/trace_diff.py` finds the first step two rollouts differ at and
  which column moved first; `tuning/trace_dump.py` prints one run. 14 unit tests.
  This is what found both bugs and it will find the next one.
- `tune_tvlqr` now reads `--trajectory-config` instead of the Justfile carrying
  its own hard-coded copy of the trajectory list (which had already gone stale).

## What is invalid and must be re-derived

Everything measured before tonight, because it ran against a partly free-running
world whose speed depended on CPU load:

- the three-way identity/TVLQR/RL comparison table
- the 16-checkpoint RL sweep
- the converged gains `q_cross=7.22 / r_omega=0.369`
- the "TVLQR oscillates on S-curves" claim — measured on a plan that was an L

This also explains why `compare` and the tuner disagreed on identical gains: they
ran at different CPU loads. That mystery is closed.

## Suggested order tomorrow

1. Read off the tuned gains; sanity-check them with a few repeat measurements
   (cheap now — repeats are ~35 s, and the noise floor is 0.0002 m).
2. Re-run `just compare` on the new 7-trajectory set to rebuild the
   identity/TVLQR baseline table from scratch.
3. Only then revisit RL. The queue in CLAUDE.md is unchanged and still correct:
   fix SAC's `ent_coef` runaway (it hit 3.31) and bound the per-episode return
   before any retrain — and the advisor's requirement that RL be in the system
   still stands, so "ship TVLQR alone" is not an option.

## Open, not urgent

- The patch friction values are still unvalidated (`icy` mu=0.05 is black ice,
  and half of every patch set is at or below tyre-on-ice). Deployment target is
  linoleum/tile; advisor wants ice **and sand**. Change is written up in
  CLAUDE.md but deliberately not made, since it re-baselines everything again —
  do it right after the corrector work, not during.
- Parallel sims via `GZ_PARTITION` + `ROS_DOMAIN_ID` (queue item 7) is now more
  attractive: repeats are mandatory practice and embarrassingly parallel.
- Your idea from tonight — generating interesting trajectories by construction
  (sample map pairs, A*-screen for tortuosity, only PMP-solve the good ones) — is
  written up as queue item 8.

## State of the VM

One `gz sim` (the rl-sim, headless, no GUI attached), one tmux session `rl` with
the `tune` window running. Nothing else. `just check-sim` before launching
anything, and `just kill-sim` clears it properly — `tmux kill-server` does not.
