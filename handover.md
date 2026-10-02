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
- **Bare-ground soak of the live plans (job 150, PRELIMINARY: 257 of 340 rows):**

  | | live plans | their library plans |
  | --- | --- | --- |
  | mean max cross-track | 0.474 m | 0.609 m |
  | mean `final_err` | 0.183 m | 0.259 m |
  | miss rate | 7.4% | 10.5% |

  **The stack's plans are not hard to track; on bare ground they track at
  least as well as the library plans.** So the miss gap comes from something
  only the full stack has: walls (contact), amcl's pose (even if unbiased on
  average), or ROS timing and latency in the corrector node. The live plans'
  higher `J` (geometric mean 20 vs 13) is probably just their length (about 280
  vs 186 samples). That is unchecked.

## 4. Running now

- **Job 150** on the VM, started 18:18 UTC, ~40 min total, so it should be done
  by about 22:00 MSK.
  - Command: `tools/jobs/150_live_vs_library.sh`
  - Log: `/tmp/lvl.log`
  - Output: `~/soak_live_vs_lib.jsonl`. It is only written at the end; the
    in-progress rows are in `.jsonl.w1`.
  - Traces: `~/soak_live_vs_lib_traces/`
  - Each live plan alternates with its library pair, in `~/live40/` (live)
    and `~/traj_data_v2/` (library).
  - Read the result the same way as the preliminary table above: pair by case,
    with `trajectory` ending in `__wfp0` = live.
  - The four failed rows are the usual ~1% patch-spawn failures.
- Note: `tools/agx-run` does not cd into the repo. Launch jobs with
  `cd ~/agx_navigation && set +u && source /opt/ros/jazzy/setup.bash && source install/setup.bash && bash tools/jobs/…`.

## 5. Next, in order

1. **Finish job 150**: final table, then a comment on #34.
2. **Separate the three stack-only suspects with one comparison run (v8, ours
   only).** It needs the user's OK, since it occupies the VM for hours.
   - Arms: `localization:=truth` vs `amcl` on the same plans, each recording
     the new per-tick `tvlqr_diagnostics` and `live_plan` (`compare_run.py`,
     added today, not yet exercised).
     - truth fixes the misses: amcl's pose noise is the cause.
     - truth still misses: walls or ROS timing are the cause. Next, check
       the per-tick data for command latency or jitter, and the departure
       points vs wall proximity.
   - v8 is also the freeze re-test (#32): launch it via
     `tools/agx-run --detach`, as v6 was.
3. Move the analysis scripts `/tmp/lag.py` and `/tmp/lag2.py` (time-indexed vs
   corrector error, departure, lag, saturation windows) into `tools/`. They are
   on the laptop's /tmp and will vanish on reboot.
4. Then go back to the comparison (#10, #33) and the re-join Phase 1 (#35).
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
