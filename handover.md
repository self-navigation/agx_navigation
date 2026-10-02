# Handover — 2026-10-02 (rewritten from scratch 21:55)

**Read this first.** It says what we are doing, why, what we know, what is
running, and what comes next. CLAUDE.md holds the cumulative *established*
claims and is the reference. Open work is tracked in Forgejo issues (list them
with the `fj` tools). Older versions of this file are in git history
(`git log -p handover.md`); nothing in them is needed to continue.

## 1. The goal

**A paper for MDPI *Mathematics*.** The method is a frozen PMP plan plus a
runtime corrector, framed by the advisor's SVCM / ε-optimality theory. The
advisor reviewed the draft on 2026-09-16. Gating items, in order:

1. **A comparison against other methods (#10)**, i.e. Nav2 MPPI/RPP, plus
   recent-paper planners (#33; the user has the names). The advisor said a Q1
   venue rejects without it, and he wants a *dynamic* environment eventually,
   which needs the re-join re-planner (tier 2 below).
2. The theorem, kept in its old single-statement form (#8), which is blocked on
   the advisor.
3. Paper chores: #4-#7 and #16-#20.

**What we are doing right now:** the comparison showed that **our arm misses
far more in the full stack than on the soak bench** (#34). That gap has to be
explained before the comparison can be quoted.

## 2. The corrector architecture (three tiers, "escalation ladder")

Design: docs/corrector-design.md.

| tier | role | status |
| --- | --- | --- |
| 1. **TVLQR** every tick | small deviations | **done, tuning closed**: `q_cross=2.5, r_omega=2.618` |
| 2. **re-join network** | medium deviations: a learned PMP re-join back onto the plan at `k + T_w/dt` | Phase 0 done (#11: min-effort re-join solves 99% at `T_w >= 3 s`). Phase 1 (dataset, #35) not started. Its trigger was meant to be persistent TVLQR saturation |
| 3. **full replan** | large deviations, or the plan becoming infeasible | not built; tied to #10's dynamic environment |

## 3. What we know (results that hold)

**The soak bench** runs Gazebo with slip patches but no walls, takes its pose
from Gazebo ground truth, and runs TVLQR in-process with no ROS. On the 40
broad plans:

- The adopted gains miss **11.5%** of the time (`final_err > 0.5 m`).
- Open loop misses **78%**.
- The corrector beats open loop on arrival on 38 of the 40 plans.

**The full-stack comparison** (`tools/compare_run.py`, job 140) is ours vs Nav2
MPPI vs Nav2 RPP: 40 plans × 2 seeds, baked map, amcl, a fresh stack per cell.

- **v7** (2026-10-02) is the first clean run: **zero freezes**.
  - The freezes were #32. Their mechanism is unknown, but they vanished in v7.
    The suspects are the `agx-run --detach` launch path and the bpftrace probe
    (now stopped).
  - Figures: `figures/2026-10-02/`, stamped PRELIMINARY / not for publishing.
- **In v7, ours arrives only 25 of 80 times.**
- What holds across both seeds: ours drives about 3.5× faster than MPPI and
  uses 2.5–3× less energy. `final_err` is equal to MPPI's and better than RPP's.

**The #34 investigation (why ours misses in the stack):**

- ~~The plans clip walls~~ is **retracted**. The v7 track files stored the
  *library* plan, not the plan the stack actually solved. Against the real live
  plans (reproduced offline by `tools/replan_footprint.py`), only 1 of 40
  overlaps a wall.
- The yaw-aware **footprint barrier `w_fp`** in the PMP exists but is **off**
  (default 0). Sweeps at 0/20/100/500 change nothing, and all 19 narrow doors
  on floors 2–6 are crossed aligned (≤5.5°) even at 0.
- **Misses are tracking failures.** The robot strays from the live plan by a
  median of 0.51 m at its peak, and by more than 0.2 m in 65 of 66 runs.
- **It is NOT amcl.** The corrector's own error matches ground-truth error.
- **It is NOT schedule lag.** When the robot first leaves the path by more than
  0.2 m it is only 0.2 s behind (median). The lag builds up later, after
  contact.
- **TVLQR is NOT saturated when the robot departs** (0% saturation in the
  preceding 4 s). So the tier-2 saturation trigger would never fire on these
  failures.
- Playback is time-indexed (`playback_index=time`). The "nearest point" metric
  was only ever used in our analysis, never in the controller.
- **Bare-ground soak of the live plans (job 150, final, 34 pairs × 5, `soak_data/soak_live_vs_lib.jsonl`):**

  | | live plans | their library plans | live worse (sign test) |
  | --- | --- | --- | --- |
  | mean max cross-track | 0.472 m | 0.615 m | 7/34, p=0.0008 |
  | mean `final_err` | 0.182 m | 0.257 m | 14/34, p=0.39 |
  | miss rate | 6.5% | 10.0% | — |
  | geo `J` per step | 0.0555 | 0.0571 | 23/34, p=0.06 |

  **The stack's plans are not hard to track** — better on peak deviation, equal
  on arrival. Their higher total `J` (20 vs 13) is length (373 vs 237 steps);
  per step it is the same. So the v7 miss gap comes from something only the full
  stack has: walls (contact), amcl's pose, or ROS timing in the corrector node.

## 4. Running now

**Job 160, overnight stack-factor sweep** (launched 22:08 MSK 2026-10-02,
~5 h on 3 workers). `tools/jobs/160_stack_factors.sh`; header explains it.
Log `/tmp/fac.log`; per-worker logs and rows in `~/compare_factors/<cfg>/rows.s<seed>.jsonl`,
tracks (with per-tick diag + live plan) in `~/compare_factors/<cfg>/s<seed>/`.
Ours only, 40 broad plans x seeds 0,1:

| cfg | localization | walls | corrector |
| --- | --- | --- | --- |
| A | amcl | solid | tvlqr (= v7 replicate; freeze re-test #32) |
| B | truth | solid | tvlqr |
| C | amcl | phantom | tvlqr |
| D | truth | phantom | tvlqr |
| E | truth | phantom | identity (open loop) |

Phantom walls = `phantom_walls:=true` (new launch flag): the building is
spawned with visuals but no collision, so lidar/amcl see it and the robot
drives through; every row now carries `wall_overlap_s`, `wall_first_t`,
`wall_episodes`, `wall_min_clear` (footprint vs baked map, 5 cm raster).
Read: A vs B and C vs D = amcl; A vs C and B vs D = contact; D vs job 150's
6.5% = ROS timing/node; D vs E = does the corrector help in the stack.
Smoke (1 plan, cfg C) ran clean: floor spawned, no wall overlap, failed at 0.76 m.

## 5. Next, in order

1. **Read job 160** (running; it is the v8 below, with phantom walls added).
   - Arms: `localization:=truth` vs `amcl` on the same plans, each recording
     the new per-tick `tvlqr_diagnostics` and `live_plan` (`compare_run.py`,
     added today, not yet exercised).
     - truth fixes the misses: amcl's pose noise is the cause.
     - truth still misses: walls or ROS timing are the cause. Next, check
       the per-tick data for command latency or jitter, and the departure
       points vs wall proximity.
   - v8 is also the freeze re-test (#32): launch it via
     `tools/agx-run --detach`, as v6 was.
2. Move the analysis scripts `/tmp/lag.py` and `/tmp/lag2.py` (time-indexed vs
   corrector error, departure, lag, saturation windows) into `tools/`. They are
   on the laptop's /tmp and will vanish on reboot.
3. Then go back to the comparison (#10, #33) and the re-join Phase 1 (#35).
   The tier-2 trigger may need to be deviation-based rather than
   saturation-based, given the finding above.

## 6. Standing rules that a new session breaks first

- **`localization:=none` cannot evaluate a pose-feedback corrector.** Measure
  the corrector under `truth` and the system under `amcl`.
- **Do not run another gain search.** Tuning is closed.
- **Evaluate on the 40 broad plans**, mean-of-5, paired sign tests, with any
  control arm carried in the same process.
- **One sim per partition.** `tmux kill-server` orphans sims rather than
  stopping them. Check `pgrep -af 'gz[ -]sim'` before launching.
- **Never use `MultiThreadedExecutor` under sim time.**
- **Wrap ROS sourcing in `set +u`.**
- **Ask before occupying the VM for hours.** It is the user's resource.
