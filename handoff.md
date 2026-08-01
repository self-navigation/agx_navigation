# Handoff: 2026-08-01 — the RL run's verdict, a contaminated baseline, and TVLQR tuning

## State as of 2026-08-01 ~23:00

Branch `tvlqr-corrector`. **A TVLQR gain search is running unattended on the VM**
— see "Running overnight" before touching Gazebo.

Access: `ssh programmer@172.26.13.37`. Jump host route:
`ssh -J llm_test2@kron.botik.ru programmer@192.168.71.113 -p2202`.

The whole ROS 2 / Gazebo workspace lives on that VM. Code gets there with
`just sync` (rsync of the working tree), *not* by pulling commits — the VM's own
`git log` is stale and misleading. Don't fix it with git there; re-sync.

**Standing context now lives in `CLAUDE.md`, section "Current work: making the
corrector work".** It is a living log: append to it, or rewrite what a new result
contradicts, and date each claim. This handoff is the session-to-session
snapshot; CLAUDE.md is the accumulated state.

## Running overnight

```
tmux window   rl:tune  on the VM        (just tune-log  to read it)
log           /tmp/tune_tvlqr.log
cache         ~/tvlqr_tune.jsonl        every evaluation, resumable
result        ~/tvlqr_tuned.json        written on convergence
```

Nelder-Mead over `(q_cross, r_omega)`, **no evaluation budget — runs to
convergence**, ~75 s per evaluation. The `rl:sim` window must stay up; the
search talks to it. When it finishes:

```bash
just fetch-tune && just plot-tune       # -> figures/tvlqr_tune_landscape.png
```

Baseline at the current `q_cross=10 / r_omega=0.25` is **0.487 m** mean
max|e_cross| (0.243 straight / 0.224 S-curve / 0.993 corner).

If it died, just re-run `just tune-tvlqr` — the cache replays for free and it
continues from where it stopped. Check `pgrep -af 'gz[ -]sim'` first.

## What this session established

### 1. The 20260730 RL run is a negative result, with an identifiable cause

1.5M steps, 9h41m, 8864 episodes, **2 successes**. A 16-checkpoint sweep shows
**no checkpoint beats identity on any shape at any point in training** — no
trend, wandering between 0.3 and 6.6 m. Best is 800k; past that it degrades
(S-curve 0.91 m → 4.65 m by 1.5M).

Two numbers say why, and neither is "not enough steps":

- **`ent_coef` ran away to 3.31.** SAC's auto-tuned entropy coefficient belongs
  well below 1. At 3.31 the entropy bonus dominates the return, so the optimal
  policy under the *effective* objective is near-random.
- **`critic_loss` back to 1.23e4**, from the 58.5 the Huber fix achieved. Huber
  bounds the reward's *slope*, not the accumulated return over 200
  non-terminating steps.

Fix those before spending another nine hours. Figures:
`figures/checkpoints_max_cross.png`, `figures/*_checkpoints.png`.

**These stats were found by screenshotting the VM's desktop**, not from the log —
the final stats block was on screen and never written to the tail I was reading.
`ssh <host> 'DISPLAY=:0 import -window root /tmp/x.png'` works with no GUI
interaction; use it.

### 2. The 2026-07-30 comparison table was measured on a contaminated world

`GazeboBridge._remove_terrain` only removed patches *its own process* spawned,
and the sim outlives any one process by design. A trainer that exited
mid-episode left its patches behind, and the next `compare_correctors` inherited
them: a leftover `rl_ground` slab from a `--ground-friction` run is a different
plant, and `create` on an existing name *fails*, so an inherited `rl_patch_0`
also displaced the patch the new run meant to spawn.

Fixed — the bridge now sweeps `rl_ground` and `rl_patch_0..7` at construction.

Re-measured clean (max|e_cross| / final_err, m), best RL checkpoint = 800k:

| trajectory | shape | identity | tvlqr | rl (800k) |
| --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | 0.11 / 0.49 | **0.01 / 0.02** | 0.66 / 0.25 |
| floor_6_00042 | S-CURVE | **0.20 / 0.18** | 1.55 / 0.52 | 0.96 / 0.43 |
| floor_6_00023 | CORNER | 1.51 / 3.90 | **0.23 / 0.05** | 1.04 / 1.46 |

**"TVLQR beats identity on every shape" is retracted** — it loses on the
S-curve. So is "identity's failure scales with curvature".

### 3. The trajectory library is not what the labels say

`just gallery` renders all 100 plans, each rotated onto its principal axis
(`figures/trajectory_gallery.png`). `classify_plans.py` calls 58 of them CORNER,
but most are visually straight — the descriptor is tripped by the in-place
reorientation the PMP planner puts at the *start* of a plan, a large heading
change over no distance. **Trust the picture over the label.**

`floor_6_00042`, "the S-curve" in every comparison to date, is really an L with
one rounded bend. Genuine S-curves: `floor_6_00028` (cleanest), `00024`,
`00047` (zigzag), `00056` (tight V). True U-turn: `floor_6_00031`.
Working set + candidates now live in `config/eval_trajectories.yaml`.

### 4. Tuning machinery, and the traps found building it

`agx_planning/tuning/` — `simplex.py`, `objective.py`, `cache.py` are pure and
ROS-free (same rule as the RL pure modules), 25 unit tests aimed squarely at
proving the search *minimizes*. A tuner that maximizes produces an
identical-looking log and returns the worst gains it found.

Design decisions worth not re-litigating:

- `--max-evals` bounds candidate **gain pairs**, never rollout length. Every
  evaluation drives every trajectory start-to-goal.
- The trajectory set is **fixed, never sampled**. A rollout that *fails* makes
  the whole evaluation invalid (`inf`), never a mean over survivors — the
  trajectories differ hugely in difficulty, so averaging whatever finished
  rewards gains that crash the hard rollouts.
- **Failures are never cached.** Killing the tuner mid-evaluation invalidated
  the bridge's rclpy context; every rollout then failed in 2 ms and 56 bogus
  `inf`s were written in three seconds. Memoized, they would have replayed as
  real measurements on every future resume. `inf` means "the sim broke", not
  "these gains are bad".
- Unbounded mode is guarded by a 15-minute per-evaluation timeout and an abort
  after 5 consecutive failures.
- Search runs in log10 of both gains; every evaluation records its
  per-trajectory errors, so the landscape can be re-analysed per shape without
  re-driving anything.

## Open, in rough priority order

1. **Read the tuning result.** `just fetch-tune && just plot-tune`. If the
   landscape shows the simplex collapsing onto a corner of the bounds, widen
   `BOUNDS_LOG`; if it wanders without converging, the noise floor (item 4) is
   above the effect size.
2. **Fix SAC's entropy runaway before any retrain** — pin `ent_coef` or set an
   explicit `target_entropy`. The default `-dim(A)` is far too permissive for a
   4-D residual with a tiny useful range.
3. **Bound the per-episode return.** Normalize it, cap per-step cost, or
   reinstate termination with a large-but-finite terminal penalty (*not* the
   0.5 m corridor that caused the original no-recovery problem).
4. **Explain the run-to-run variance — this is now blocking.** The tuner scores
   TVLQR at 0.224 m on floor_6_00042 where `just compare` scored 1.549 m an hour
   earlier, same gains, same seed, same code path, deterministic stepping. That
   spread is larger than most effects being measured. Cheap first experiment:
   drive one trajectory 10x in one process and 10x in ten processes, and see
   which spread is larger. Untried candidate remains accumulated sim-clock
   floating point in `GazeboBridge._wait_clock_advance`.
5. **Re-measure on a genuine S-curve** (`floor_6_00028`) now that the gallery
   shows the current one is not.
6. **Widen tuning to the full Q/R diagonal** (`q_along`, `q_heading`, `r_v`)
   once the 2-D search proves the machinery. Item 4 sets what is resolvable.
7. **Train RL on top of *tuned TVLQR*** rather than on top of identity, so the
   policy learns the residual a good linear controller cannot supply instead of
   re-deriving feedback from scratch. Most likely to work, and the most
   defensible framing given that **the advisor requires RL in the system** —
   "drop RL, ship TVLQR" is not an available recommendation.
8. `floor_1_00050` is still a degenerate PMP plan (`max_turn = 3.14 rad/step`
   over 6 m). Planner bug, excluded from comparisons, unexplained.

## Facts that cost time to establish

- `compare_correctors` builds the bridge with `deterministic=True` — the world
  is paused and multi-stepped, so results do **not** depend on CPU load and a
  `gz sim -g` viewer cannot perturb them.
- Identity/TVLQR baselines are checkpoint-independent. A sweep that re-measures
  them per checkpoint burns two-thirds of its runtime; measure once, then
  `--correctors rl`. 15 checkpoints ≈ 35 min.
- `make rl-sim` is server-only (no rendering), so there is no 3D view unless a
  GUI client is attached (`just gui`).
- Nothing has been pushed to `origin`.
