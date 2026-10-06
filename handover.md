# Handover — 2026-10-02 (rewritten from scratch 21:55)

**Read this first.** It says what we are doing, why, what we know, what is
running, and what comes next. CLAUDE.md holds the cumulative *established*
claims and is the reference. Open work is tracked in Forgejo issues (list them
with the `fj` tools). Older versions of this file are in git history
(`git log -p handover.md`); nothing in them is needed to continue.

## 0b. Session 2026-10-06 (day, from 10:56 MSK) — resume here

**Running on the VM:**
- **Job 221 = job 220 part 2** (controllers only, every arm on the SAME library plan, phantom walls; #10/#33 Type A). Launched 11:22 MSK, workers 1-12, frozen checkout `~/agx_navigation_job221` (commit 7e0c96f), log `/tmp/job221.log`, out `~/run_data/2026-10-06_job221_controllers/` (README there). 12 configs = {ours-lib, pmp-mppi, pmp-rpp, pmp-graceful, pmp-vpp, pmp-gmpc} x {truth (`*T`), amcl (`*A`)}, seeds 0 1, 960 runs, ETA ~5.5 h (about 17:00 MSK). FINISHED 16:20 MSK, 960 rows, read. Truth: ours miss 16% vs MPPI 48%, RPP 27%, VPP 24%, ours closer on all three (p<1e-4). Amcl: beats MPPI (p=0.003), ties RPP/VPP (p=0.18/0.11). Graceful and GMPC miss 95-98% (broken integration, excluded). In the paper as tab:ctrl; figures/2026-10-06/job221_*.
  - Read: `python3 tools/summarize_arms.py all_rows.jsonl LT` (truth block) and `... LA` (amcl block), and pair only within the same localization. Truth = the corrector claim, amcl = the system claim. Then render figures like `figures/2026-10-06/` and write the controllers paragraph in sec:stack. The user wants the advisor to see it all.
  - The first launch (08:03 UTC) was ABORTED, data in `~/run_data/2026-10-06_scratch_job221-aborted/`. There were three bugs, all fixed in 7e0c96f. (1) Under truth the fixture ran without the lidar, so Nav2's collision_monitor stopped the robot ("invalid source"). Now `make fixture` turns sensors on for nav2, and static_map launches pointcloud_to_laserscan for amcl OR nav2. (2) The GMPC submodule was never checked out in the laptop clone, so the frozen checkout had no sources. Once checked out, its `scout_description` clashed in colcon, so the Makefile now builds only `third_party/vector_pursuit_controller`. (3) compare_run labelled "terminal + no motion" as planner-failed even for library-plan arms. Those are now scored as `failed`.
  - Smoke after the fixes (`~/run_data/2026-10-06_scratch_job221-smoke2/`, 00369, truth, n=1): pmp-gmpc arrived 0.45 m; pmp-mppi drove and failed 3.6 m (MPPI's known skid-steer stall, see compare_skid).

**Done today:**
- Job 220 part 1 read. Under amcl with solid walls, paired final_err vs ours: Smac2D+MPPI tie (ours closer 36/70, p=0.9); Hybrid-A*+MPPI borderline worse (43/71, p=0.1); RPP worse (51/72, p=5e-4). Miss P_0.5: ours 62% (45/72), M2 56%, MH 77%, R2 82%. Ours is 3x faster and spends 4.8x less control than M2 (69/70, 68/70). Wall contact: ours 52/72 runs, M2 22/77, MH 10/79. Ours had 8/80 planner failures (#39).
- **Scoring rule changed (2026-10-06):** `summarize_arms.py` now SCORES timeouts (the robot ends where it stopped). Only planner-/stack-failed runs are excluded. Earlier numbers in this file (job 210 etc.) were computed without timeouts.
- Paper (`../paper` 185e249): sec:stack's comparison paragraphs now use job 220 (v7 is gone). The #39 planner failures are counted as excluded and explained in Limitations (`sec:limits`). Style follows the advisor rules: short sentences, no `;:—`, no lists.
- Figures: `figures/2026-10-06/` (paired final_err, cost ratios, tracks; README there). The tracks show Nav2 stalling at narrow-corridor turns (00219) and only MPPI making a door turn (00061).
- Killed the stuck ours-lib smoke loop (it had waited ~5 h on worker 9, held by an orphan) and 5 orphan sims (workers 9, 11-14). The one ours-lib smoke that ran arrived 0.32 m on 00047 (the live-planner-fails plan).
- **Cleanup equivalence: PASSED** (`~/run_data/2026-10-06_scratch_cleanup-equiv/`, 10 plans, truth, phantom walls). Same outcomes on 9/10 (both planner-fail 00443; 00369 straddles 0.5 m). Post-cleanup closer on 4/9, p=1. Item 3 of the list below is done.
- Lesson: `pkill -f <pattern>` inside `ssh '...'` kills the ssh's own bash. Use the `[x]yz` bracket trick or explicit PIDs.

**Next, in order:** job 221 is done and written up. Then items 3-5 of the list below (cleanup equivalence, deferred refactor, #38 and the cover note).

## 0a. Session 2026-10-06 (night) — superseded by 0b; kept for the history


**Running on the VM (nothing needs babysitting):**
- **Job 220 part 1** (full stacks, amcl, SOLID walls; #10/#33 Type B), workers 1-8, frozen checkout `~/agx_navigation_job220` (commit 1a80fa7), log `/tmp/job220.log`, out `~/run_data/2026-10-06_job220_fullstack/` (README there). Arms: O ours, M2 Smac2D+MPPI, MH Hybrid-A*+MPPI, R2 Smac2D+RPP; Nav2 arms use `compare_skid`. Slower than estimated: ETA about 07:00 MSK; the MPPI arms are slowest. Read: `python3 tools/summarize_arms.py all_rows.jsonl O`. Also report on the common set (ours planner-fails ~5/40, #39). Compare per-run wall times against job 210 to see whether 9-14 concurrent sims slowed it (#32).
- **ours-lib smokes**, worker 9: `~/run_data/2026-10-06_smoke_ours-lib/run.sh`, log `run.log`, rows `rows_{amcl,truth}.jsonl`. Plans 00047 (live-planner-fails) and 00369 under amcl, then 00369 under truth. CHECK: outcome, `libsrv` start offset, and that the GT track follows the library poses (compare the diag `e_norm` against the distance from GT to `plan_poses`; `live_plan` proves nothing).

**Next, in order:**
1. Read the ours-lib smokes. If sane, launch **job 220 part 2** (controllers only, every arm on the SAME library plan, phantom walls, under BOTH truth and amcl): ours-lib, pmp-mppi, pmp-rpp, pmp-graceful, pmp-vpp, pmp-gmpc. Workers up to 14 now. Write a configs file like `tools/jobs/fullstack/configs.txt`, freeze a checkout, and put the README in at launch. GMPC needs the `~/.venvs/gmpc` venv (exists on the VM) and `--nav-mode gmpc` (compare_run handles it).
2. Read job 220 part 1, then write jobs 210/220 into the paper (sec:stack). Job 210 compared DIFFERENT plans (Nav2 on the library plan, ours on a live re-solve), so it is not a controllers-only result. Use part 2 for that claim, and part 1 for the system claim.
3. Cleanup equivalence check: pre-cleanup (`~/agx_navigation`, synced at 18be0b4) vs cleanup (`~/agx_navigation_cleanup`), 10 plans, truth, phantom walls. Single smokes ran fine but n=1 on 00369 cannot show equivalence (data `~/run_data/2026-10-06_cleanup_smoke/`). One stray "no /clock" stack-failure on worker 13 did not reproduce.
4. Deferred refactor (audit plan; summary of it in this file's git history for 05:00): consolidate launch args into main.launch.py with explicit passing; merge copy-pasted node defs; full CLAUDE.md rewrite (~250 lines, outline from the audit). Decision pending: PlannerConfig defaults := stack values, plus a `library_v2()` preset for the library tools (#39). Do it only after the paper's library is frozen. Also: Nav2 arms' max turn rate differs from ours (fairness, #10); `slip_chi` 1.373 vs 1.3736.
5. #38 review list, Russian cover note for the advisor.

**Done tonight (all pushed: main d2f1788, scout_ros2 d2cf3f0, paper aa2256d):**
- Job 190 read (negative, in the paper); job 210 read (#10). Job 200 superseded (MPPI stall), data in run_data.
- #39 filed: the live BVP fails on 5/40 because the stack's L_brake/w_v_barrier/w_v_terminal make it stiff; the library used defaults. Fix proposal: weight continuation on the live map (a library warm-start is invalid on live maps).
- New package `agx_baselines`, comparison-only (README says so): Nav2 overlays/profiles, Graceful + Vector Pursuit (submodule, Apache-2.0), GMPC wrapper (submodule, NO licence upstream: run-only, not redistributable). `nav2_profile` takes a comma list. compare_run arms: ours-lib, nav2-/pmp-graceful, nav2-/pmp-vpp, pmp-gmpc.
- Smokes on 00369, amcl, phantom (n=1 each, not a ranking): pmp-gmpc arrived 0.40 m; pmp-graceful failed 3.56 m; pmp-vpp failed 1.39 m (`~/run_data/2026-10-06_smoke_{gmpc,graceful-vpp}/`).
- Cleanup merged: online PMP mode removed (#29); `make run` defaults to vec-pmp; frontier only under nav2; `make test` path fixed (run it with `PYTHON=.venv/bin/python` locally); dead modules/tools/recipes archived. 267 tests pass.
- Worker cap 9 → 14; the real limit is RAM, ~1.5 GB per sim. Assigned tonight: 9 ours-lib, 10 GMPC, 11 ctrl, 12-14 smokes.
- VM disk: ollama removed and pip/uv caches cleared, 39 GB free. typeA checkout deleted. Agent VM checkouts `~/agx_navigation_{gmpc,ctrl,lib,cleanup}` can be deleted once part 2 runs from a fresh frozen checkout.
- Lesson: never poll with `pgrep -f "<pattern>"` from a shell whose own command line contains the pattern. It matches itself forever.

## 0. Session 2026-10-05: jobs 170/180 read, #34 mechanism found

Full numbers are on #34 (comment 2026-10-05). Scripts: `tools/believed_gain.py` (gain/lag; args are cfg dirs) and `tools/summarize_arms.py` (miss + paired sign test; args `<all_rows.jsonl> <base_cfg>`), which superseded `tools/loc_analysis/` (deleted 2026-10-06). Run them on the VM from the repo root; the data now lives under `~/run_data/`.

- **amcl tuning cannot close the gap.** All 32 cells of job 180 miss 13–26%, against 9% under truth. EKF wheel yaw has no effect at all (job 170 Cy ≈ C, and p=1.0 in job 180).
- **Mechanism:** the corrector sees its cross-track error at about **0.4x gain and about 2.5 s lag** under amcl, and only ~5% of it while turning. Under truth it sees 0.95x at 0.1 s. Turn skid is invisible to wheel odometry, and amcl corrects it only slowly.
- Ruled out: pose latency, the scan-band/bake-band mismatch (real, but it changes ~100 cells), lidar extrinsics, slip patches.
- **DONE: job 190** (lidar odometry, #34), read 01:50 MSK 2026-10-06. NEGATIVE: rf2o in the EKF leaves amcl's believed-error gain/lag unchanged (k 0.38→0.43, τ 2.7→2.4 s); P_0.5 C 14% (10/70), L 24% (16/68), LA 62% (42/68); L vs C 33/66, p=1. The lag is amcl's scan correction, not the odom source. Paper sec:stack filled (paper aa2256d); data moved to `~/run_data/2026-10-05_job190_lidar-odom/` with README.
  - How to read it: run `tools/believed_gain.py ~/run_data/2026-10-05_job190_lidar-odom/{C,L}` on the VM, and sign-test L vs C with `tools/summarize_arms.py <rows> C`.
  - If L has k near 1 and tau near 0.1 s with miss near 9%, the gap was skid observability: adopt lidar odometry, put it in the paper, and test it on the real robot (#36).
  - If L does not move: the wheels' vy=0 is not what blinds amcl, so look at amcl's resampling or motion model next.
  - Note: the VM's /tmp logs vanished between 10-03 and 10-05. The data dirs are what persists.
- **RUNNING: job 200** (`tools/jobs/200_hybrid.sh`, launched 23:08 MSK, workers 4-7, log `/tmp/hybrid200.log`, out `~/compare_hybrid/`, ~3-4 h). It compares Smac Hybrid-A* + MPPI (profile `compare_hybrid`) with Smac2D + MPPI (`compare_static`, the control), under amcl with solid walls, so the conditions match v7. Note that in v7, 84/160 MPPI runs ended at `sim_timeout`, so read timeouts separately from misses.
- **DONE: job 210** (read 03:50 MSK 2026-10-06; README in its run dir has the table). P_0.5 ours 21% / RPP 38% / MPPI(compare_skid) 55%; final_err pair wins vs ours 28/70 (p=0.12), 27/68 (p=0.11); same ordering on the 68-run common set. Ours planner-failed on 5 plans × both seeds (live BVP mesh-node exhaustion where the library solved the pair) — investigation running.
  - (was) job 210 (#10 Type A; RELAUNCHED 00:12 MSK 2026-10-06 from the SEPARATE checkout `~/agx_navigation_typeA`, which has commit 49a33e1 plus the new profile copied in and built): Nav2 MPPI/RPP track OUR PMP plan via controller_server FollowPath (arms `pmp-mppi`/`pmp-rpp` in `compare_run.py`) vs ours (P), amcl + phantom walls, 40 broad plans. **PM/PR now use nav2 profile `compare_skid`** (commit 3a6d071): compare_static + MPPI `wz_std` 0.3→0.8 and PathAngleCritic 1.5→5.
  - Why the change: under compare_static, MPPI stalled mid-route on 00369. It commanded wz≈0.1 rad/s at v<0.04 m/s, the gyro read 0 (the skid-steer scrubs through that), and the chi-biased EKF wheel yaw told MPPI it was turning. Turning the wheel yaw off alone did not help (MPPI then parks at the 90° turn). With compare_skid the smoke made the turn and ended 0.39 m from the goal. It still trips the progress checker while pivoting to the goal yaw, so expect residual FAILED_TO_MAKE_PROGRESS near the goal. Evidence: `~/run_data/2026-10-05_job210_pmp-trackers/diag_skid/` and the run README.
  - Old compare_static PM rows are in `_old_compare_static/`. P rows were kept and resume.
  - Seed 0 is on workers 8,9 (log `/tmp/job210_s0.log`, ETA about 03:45 MSK). Seed 1: the waiter started it at 01:34 MSK but all three workers were `Terminated` within 15 s (cause not found; the old log is `/tmp/job210_s1.killed.log`). Relaunched by hand at 01:43 MSK on workers 1-3 with setsid (log `/tmp/job210_s1.log`), ETA about 04:30 MSK. A local watcher plays an alarm when both logs print `[fac] finished`; expect 240 rows.
  - Done when both logs contain `[fac] finished`.
  - Read: paired per-plan sign tests P vs PM and P vs PR on final_err and miss, with timeouts kept separate.
  - Write-up caveat: Nav2 MPPI's DiffDrive model has no minimum executable yaw rate.
- Data inventory: `docs/run-data-index.md`. The move plan `tools/data_tidy_plan.sh` is **not run**; it waits for the user's approval.
- Advisor tracker: #37. Questions were sent to him at 23:07 (deadline, comparison scope, framing, the τ term, naming TVLQR, the theorem's last sentence).
- The comparison candidates for #10/#33 are now on #33. The real-robot skid-lag test is #36, deferred because of the deadline.
- **Paper: the advisor wants a final version on 10-05/06.** A subagent is editing `../paper/draft.tex`: it is adding the full-stack comparison and mechanism section and doing the #19 style pass (wording from `advisor-revision-2026-09-16.tex`). Placeholders `% PENDING lidar-odom result` are waiting for job 190.
- Paper (`../paper/draft.tex`, Russian): it currently says nothing about Nav2, amcl or wall contact. The survey of phrasing problems is in this session's notes. Write-up waits for the fix result.

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

## 4. Job 160 result (finished 01:30 MSK 2026-10-03, read 05:35)

**Job 160, stack-factor sweep**: 10 units × 40 broad plans, all rc=0, no freezes
(#32 re-test passed, cfg A = v7 replicate). Data on the VM in
`~/compare_factors/<cfg>/rows.s<seed>.jsonl`, tracks (per-tick diag + live plan)
in `~/compare_factors/<cfg>/s<seed>/`; header in `tools/jobs/160_stack_factors.sh`.
Miss = `final_err > 0.5 m` over driven runs (8-11 planner-failed per cfg excluded):

| cfg | localization | walls | corrector | miss | arrived |
| --- | --- | --- | --- | --- | --- |
| A | amcl | solid | tvlqr | **61%** (43/71) | 28 |
| B | truth | solid | tvlqr | 39% (27/70) | 43 |
| C | amcl | phantom | tvlqr | 25% (18/72) | 54 |
| D | truth | phantom | tvlqr | **9%** (6/69) | 63 |
| E | truth | phantom | identity | 80% (55/69) | 14 |

Paired sign tests on `final_err`: B>A 54/72, D>C 57/72 (amcl hurts, p<1e-4);
C>A 53/72 (p=1e-4), D>B 48/70 (p=0.003) (contact hurts); D>E 64/71.

- **ROS timing / the corrector node is ruled out**: D (9%) matches the soak bench (6.5-11.5%).
- **The v7 gap is amcl + wall contact, compounding** (9% -> 25%/39% -> 61%).
- **This partly contradicts "it is NOT amcl" (section 3)**: the corrector's error
  matched truth error, yet swapping amcl for truth changes outcomes. Hypothesis:
  amcl's pose is wrong in a way that both measures share. Under investigation.
- Caveat: `wall_overlap_s > 0` in ~55/80 rows even in D; about half start after the
  plan should have ended (so include time stopped near the goal). Also check the
  5 cm raster before trusting the wall fields.

**Departure analysis (2026-10-03, `figures/2026-10-03/`, `tools/departure_series.py`):**
the corrector's believed pose is rebuilt from its diagnostics (self-check: mm under
truth). **The section-3 claim "it is NOT amcl" is RETRACTED.** Under amcl the
corrector believes about 0.65x its true peak cross-track error. amcl's pose error
climbs 0.1 -> 0.3 m in the 3 s before a departure, while the believed error stays
flat, so TVLQR never reacts. Under truth, believed = true and departures are
corrected within ~3 s. With solid walls a wall pin makes amcl follow odometry
(wheels spin), and it drifts metres (00350 s0: 8 m). Bug: the diag `index` field
is -1 on every tick (k must be rebuilt from time, dt 0.1 s).
Next: why amcl lags (update rate / `update_min_d`, odom bias from wheel_odometry's
missing chi feeding amcl's motion model), and wall-pin detection.
- **Wall contact is a SAFETY problem, not a re-plan trigger (user, 2026-10-03).** On
  the real robot, driving into a wall gives rubber-wheel clatter and unpredictable
  slip. Pushing on at speed is likely to **flip the robot onto its back
  ("turtle")**, which it cannot recover from and which likely breaks the hardware
  mounted on it. So a pin detector must cut the command FAST (an e-stop-like
  guard below the corrector), and it cannot assume clean "wheels turn, IMU still"
  signals.
- amcl config (`nav2_params.yaml`): `update_min_d 0.25`, `update_min_a 0.2`,
  alpha1-5 0.2, likelihood_field, 500-2000 particles; odom = EKF (chi-biased wheel
  yaw, see CLAUDE.md). Crude probe: under amcl the believed pose jumps ~3 cm every
  ~0.12 m of travel (truth configs: almost never). So amcl corrects often, and the
  update gate alone does not explain the 0.2-0.3 m error. Suspect the odom bias /
  motion-model noise next.
- **Real-robot question open:** the user recalls the nav2 quickstart stack worked OK
  on the real robot. Whether the real localization error is bounded the same way is
  unknown until there is real data (see the questions asked in chat 2026-10-03).

## 4b. Running now (launched 2026-10-03 ~06:05 MSK)

- **Job 170** (`tools/jobs/170_loc_factors.sh`, log `/tmp/loc170.log`, out
  `~/compare_loc/`), ~1h40 on workers 1-3: `C` (amcl, phantom, control) vs `Cy`
  (same, `--no-ekf-wheel-yaw`) vs `Ay` (amcl, solid, no wheel yaw), 40 plans x 2 seeds.
  **Cy << C miss** = the chi-biased wheel yaw is what blinds amcl, so fix the EKF
  (this also affects the real robot: same EKF under rtabmap). **Cy ~ C** = look at amcl
  itself (job 180).
- **Job 180** (`tools/jobs/180_amcl_sweep.sh`, log `/tmp/amcl180.log`, out
  `~/compare_amcl/`) waits for 170, then ~18 h: 2^4 amcl factors (gate, alpha,
  beams, sigma_hit) x EKF wheel yaw, phantom walls, `base` = control. Codes in
  `tools/jobs/amcl_sweep/configs.txt`. Read main effects by paired sign test vs
  `base` on miss / final_err, and the believed/true |e_cross| ratio via
  `tools/departure_series.py`.
- **Real robot ran rtabmap, not amcl** (`slam.launch.py`; commit 92fe846, 2026-02-26;
  odom from vendor `scout_base`, same EKF). Our amcl block was first launched in #34.
  So job 160-180 bound sim amcl only. The user can get robot access ~Thu 2026-10-08:
  plan a real localization-error measurement for then.
- **Real-robot protocol (agreed 2026-10-03, to write before Thu):** (1) rosbag of
  lidar/IMU/odom/TF on the real stack (rtabmap + EKF); reference = offline batch
  re-map of the same bag; replay with EKF wheel yaw on/off as the A/B test. Test the
  pipeline on a sim bag first. (2) Tape marks, without stopping ON them: the robot
  stops anywhere; plumb bobs (or a line laser) from front- and rear-centre chassis
  points (~0.5 m apart); tape 4 distances to 2 fixed floor marks, giving x, y, yaw
  (~+-1 deg). 3-4 floor marks per area, measured to each other once; hold ~5 s still,
  timestamp via a joystick button in the bag. (3) A total station from the surveying
  department only to fix the marks' coordinates, if it is available. No mocap or
  total station is available in the lab.

## 5. Next, in order

1. **Job 160 is read (section 4).** Next: departure analysis + figures (figures/2026-10-03/). The old plan for it follows.
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
