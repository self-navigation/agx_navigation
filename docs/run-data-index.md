# Run-data index

Inventory taken 2026-10-05 (read-only pass). Nothing has been moved yet. The
proposed move script is [tools/data_tidy_plan.sh](../tools/data_tidy_plan.sh),
which has **not** been run.

VM paths are relative to `/home/programmer` (`~`). Local paths are relative to
the repo root. "Cited by" lists the places that name the path: handover,
CLAUDE.md, docs/, tools/, the Justfile, figures/*/render.py|README.md, and issues.

Classes:
- **a**: cited by the paper or a committed figure.
- **b**: cited by the handover, an issue or the docs.
- **c**: superseded or invalid. Invalid eras: before 2026-08-02 evening, and plant changes on 08-04 and 08-07.
- **d**: scratch or smoke.
- **e**: unknown.

**VM-only** means the cited data has a single copy.

Totals: VM home 49 G. About 31 G of that is not experiment data (`ollama-models` 28 G, `ollama` 2.1 G, `torch-cu126` 805 M, `.keras` 341 M). Experiment data on the VM is about 2.6 G, plus the live `compare_lidar`/`compare_hybrid`. Local untracked data is about 0.9 G.

## Live (not inspected; jobs writing now)

| path(s) | job | date | what it is | cited by | class |
|---|---|---|---|---|---|
| VM `compare_lidar/` | 190 | 2026-10-05 | lidar-odom (rf2o) EKF arms (#34) | jobs/190 | live |
| VM `compare_hybrid/` | 200 | 2026-10-05 | hybrid nav2 profile; job script untracked | jobs/200 (untracked) | live |
| VM `/tmp/smoke200/`, `/tmp/smoke*.log`, `/tmp/smoke1.txt` | 170-200 smoke | 10-0x | smoke runs in /tmp | none | d (live) |

## Comparison campaigns (#10, #32, #34), Sep 29 to Oct 5

| path(s) | job | date | what it is | cited by | class |
|---|---|---|---|---|---|
| VM `compare_smoke.jsonl`, `compare_smoke2/`+`.log`, `compare_mp_probe8.jsonl`, `cg28_smoke.jsonl` | 140 smoke | 09-28..10-01 | stage-1 smoke tests and probes (cg28 = #28 cgroup) | jobs/140 (smoke) | d |
| VM `compare_broad40/` (87M) + `.log`; local `run_data/compare_seed{0,1}/` (69M) | 140 | 09-29 | stage-1 v1. **INVALID**: amcl re-initialized on the map heartbeat | #10, figures/2026-09-29 | a (figure) / c (numbers invalid) |
| local `run_data/compare_tune{3,4}/` | pre-140 tuning | 09-29 | Nav2 profile tuning, 3 plans | #10 comment (by name) | b |
| VM `compare_broad40_v2/` + `.log` | 140 rerun | 09-29 | v2, 75 files, apparently aborted | none | e |
| VM `compare_broad40_v3/`, `cmp_v3_s*.log`; local `run_data/compare_v3_seed{0,1}/` | 140 v3 | 09-30 | v3; freezes make it unquotable | #32 | b |
| VM `compare_broad40_v4/`, `cmp_v4_s*.log`, `ram_v3.log`; local `run_data/compare_v4_seed{0,1}/` | 140 v4 | 10-01 | v4; the freeze analysis source | #32 (`~/compare_broad40_v4/seed*/`) | b |
| VM `compare_broad40_v5/` (incl. `freeze_manual/`), `cmp_v5_*.log`, `freeze_watch_v5.log`; local `run_data/compare_v5/` (1 file) | 140 v5 | 10-01 | v5 freeze live captures | #32 (`~/compare_broad40_v5/.../freeze`) | b, **VM-only** |
| VM `compare_broad40_v6/` (80M), `cmp_v6_*.log`, `freeze_watch_v6.log` | 140 v6 | 10-01..02 | SHM A/B test (negative) | #32 | b, **VM-only** |
| VM `sigtrace.bt`, `sigtrace_v7.log` | #32 | 10-02 | bpftrace signal trace during v7 | none found | b? (likely #32 follow-up); e |
| VM `compare_broad40_v7/` (86M), `cmp_v7_*.log`; local `run_data/compare_v7_seed{0,1}/`, `figures/tmp/v7s{0,1}/` | 140 v7 | 10-02 | first complete comparison | figures/2026-10-02, #34 | a |
| local `figures/tmp/cmp_s{0,1}/` | 140 v1 | 09-29 | `summarize_compare.py` output for v1 | summarize_compare.py (default out) | d |
| VM `live40/`, `live_vs_lib.txt`, `soak_live_vs_lib.jsonl`(+`.w1`), `soak_live_vs_lib_traces/` (60M); local `soak_data/soak_live_vs_lib.jsonl` | 150 | 10-02 | live-stack plans vs library on the bare-ground soak | handover, #34, jobs/150 | b (traces **VM-only**) |
| VM `compare_factors/` (45M); local `run_data/job160/` | 160 | 10-02..03 | stack-factor factorial (A-E, truth vs amcl) | figures/2026-10-03, handover | a |
| VM `dep_series/` (25M) | 160 analysis? | 10-03 | departure series (`tools/departure_series.py` output?) | none by name | e |
| VM `compare_loc/`, `compare_loc_smoke/`; local `run_data/loc170/` (all_rows only) | 170 | 10-03 | localization factors (Cy, Ay) | handover, #34 | b (tracks **VM-only**); smoke d |
| VM `compare_amcl/` (304M), `compare_amcl_smoke/`; local `run_data/amcl180/` (all_rows only) | 180 | 10-03 | amcl 2^5 factorial | handover, #34 | b (tracks **VM-only**); smoke d |

## Re-join Phase 0 (#11) and probes, Sep 30 to Oct 1

| path(s) | job | date | what it is | cited by | class |
|---|---|---|---|---|---|
| VM `rejoin_p0_2000.jsonl`+`.log` | #11 | 09-30 | 2000-problem Phase 0 run | copy committed as figures/2026-09-30/rejoin_phase0_vm2000.jsonl (paper fig `rejoin_phase0_controlled`) | a |
| VM `rejoin_rescue.jsonl`+`.log`; local `run_data/rescue/` | #11 | 09-30 | rescue run | tools/rejoin_phase0.py, docs (`rescue`) | b |
| VM `bias27/`; local `run_data/bias27/` | #27? | 09-30 | identity vs tvlqr rows, w7 tracks | none | e |
| VM `rtf_probe_camsoff/`, `rtf_probe_{a,b}.log` | #32 era | 09-30 | RTF probes with cameras off | none | d |
| VM `fieldprobe*`, `fp2..fp7*`, `fieldprobe_plans.txt`, `one_plan.txt` | none | 10-01 | vector-field/planner probes | none | d |
| VM `.pyspy/` (24M), `spy_planner.txt`, `spy_watch.{sh,log}`, `build_ex.log` | none | 10-01 | py-spy dumps (MultiThreadedExecutor finding) | tools/freeze_watch.py (`spy_`) | d |

## Gain tuning and soak era, Aug 12 to Sep 23 (valid plant, after 08-07)

| path(s) | job | date | what it is | cited by | class |
|---|---|---|---|---|---|
| VM `jobq/`, `jobq.sh` | runner | 08-14..09-28 | job queue state + logs | CLAUDE.md, docs/vm-operations.md | b (queue infra, not data) |
| VM `jsweep.jsonl`, `jsweep.sh`, `jtraces/`; local `jtraces/`, `epsilon_data/` | none (pre-queue) | 08-12..13 | J sweep | figures/2026-08-13, history | a |
| VM `uturn_default.jsonl`, `uturn_tuned.jsonl`, `uturn_traces/`; local same + `soak_data/uturn_*` | none | 08-12 | U-turn default vs tuned | figures/2026-08-13 | a |
| VM `soak_20260813_*.jsonl`; local `soak_data/soak_20260813_*` | none | 08-12..13 | ladder / two-point / U-turn subladder | figures/2026-08-13, figures/archive ladder_modes (paper `ladder_modes.png`) | a |
| VM `validate_20260812_*`, `local2d_20260812.jsonl`, `qwall_20260812.jsonl`; local `tune_data/` copies | none | 08-12 | 7-plan tune validation | history, plot_tune_validation.py | b (Settled: seven-plan search refuted) |
| VM `gaincheck/`+`.jsonl`, `libsweep/`+`.jsonl`, `sweep2.sh`; local `gaincheck/`, `libsweep/`, `soak_data/libsweep.jsonl` | none | 08-13 | 51-plan library sweep | history, Justfile, CLAUDE.md (51-plan claim) | b |
| VM `traj_data_v2/`, `candidates_v2.json`; local `traj_data_v2/` | 20 | 08-14 | v2 plan library (the broad 40) | many tools, figures/2026-08-15, 10-02 | a (`candidates_v2.json` **VM-only**) |
| VM `pmp_trajectories_v2/`; local `traj_data/` | pre-queue | 07-29 | original 7-shape plans (v1 library) | figures/2026-08-13/15, Justfile | a (plans; pre-era date is fine, they are inputs) |
| VM `pmp_trajectories/` (416 files) | pre-queue | 07-29 | older/larger plan set | Justfile/tools (by prefix) | e (may be superseded by _v2) |
| VM `soak_r_ladder.jsonl`, `r_ladder_traces/`; local both | 10 | 08-14 | r ladder | figures/2026-08-14 | a |
| VM `uturn_edge.jsonl`, `uturn_edge_traces/`; local `soak_data/uturn_edge.jsonl` | 25 | 08-14 | U-turn notch edge | jobs/25 | b (traces **VM-only**) |
| VM `soak_r_ladder_low.jsonl`, `r_ladder_low_traces/`; local jsonl | 30 | 08-14 | low-r ladder | figures/2026-08-15 | a (traces **VM-only**) |
| VM `soak_uturn_generality.jsonl`, `uturn_generality_traces/`; local jsonl | 40 | 08-14..15 | U-turn generality | figures/2026-08-15, CLAUDE.md Settled | a (traces **VM-only**) |
| VM `broad_eval_plans.txt`, `soak_broad_gains.jsonl`, `broad_gains_traces/`; local jsonl + `broad_gains_traces/` | 50 | 08-15 | broad gain generality | history, rejoin_phase0.py | b |
| VM `tvlqr_tune_J.jsonl`, `tvlqr_tuned_J.json`, `tvlqr_validate_J_adopted.json{,l}` | 60 | 08-15 | tune on J | jobs/60, jobs/90 | b, **VM-only** |
| VM `soak_broad_q.jsonl`(+`.w1-4`), `broad_q_traces/` (176M); local jsonl | 70 | 08-15 | broad q ladder (gains table in CLAUDE.md) | CLAUDE.md, history | b (traces, shards **VM-only**) |
| VM `soak_broad_r.jsonl`(+`.w*`), `broad_r_traces/`; local jsonl | 80 | 08-15 | broad r ladder | CLAUDE.md table, gramian tool | b (traces **VM-only**) |
| VM `soak_validate_J_broad.jsonl`(+`.w*`), `validate_J_broad_traces/`; local jsonl | 90 | 08-15 | validation of the J-tuned gains on broad | CLAUDE.md table | b (traces **VM-only**) |
| VM `soak_broad_r_at_q25.jsonl`(+`.w*`); local jsonl | 100 | 08-18 | adopted-gain decision run | CLAUDE.md (job 100) | b |
| VM `gramian_*.csv` (4); local `soak_data/gramian_*` | 105, 130 | 09-23 | controllability Gramian | CLAUDE.md Settled | b |
| VM `soak_seven_three_arms.jsonl`(+`.w*`); local jsonl | 110 | 09-23 | seven plans, three arms | CLAUDE.md, summarize_soak.py | b |
| VM `soak_broad_open_loop.jsonl`(+`.w*`); local jsonl | 120 | 09-23 | broad 40, open loop | CLAUDE.md | b |
| VM `soak_20260813_uturn_subladder_partial.jsonl` | none | 08-12 | partial run of the subladder | none | d |
| VM `.w1..w4` shards (all of the above) | 70-120 | | per-worker shards; merged file exists | none | d (merged copy is canonical) |

## Invalid-era data (before 2026-08-07 plant change)

| path(s) | job | date | what it is | cited by | class |
|---|---|---|---|---|---|
| VM `checkpoints/` (179M), `rl_corrector_p*.zip`, `rl_corrector_*_best/`, `rl_smoke_test2.zip`, `rl_tb/`, `train_*.log`; local `tb_data/` | RL p0-p2 | 07-24..29 | SAC checkpoints + TensorBoard | CLAUDE.md (RL Settled), Justfile, plot_* tools | c (kept as evidence for a Settled stub; checkpoints **VM-only**) |
| VM `runs_20260730/` (916M) | RL 1.5M run | 07-30..31 | 20260730 SAC run | history, CLAUDE.md Settled | c, **VM-only**, largest data dir |
| VM `tvlqr_tune.jsonl`(+`.pre20260802`), `tvlqr_tune_v2/v3.jsonl`, `tvlqr_tune_v4_newplant.jsonl`, `tvlqr_tuned.json`; local `tune_data/` | none | 08-01..07 | early TVLQR tune | history, figures/archive | c |
| VM `reset_probe*.jsonl`, `reset_world_probe.jsonl`, `reset_world_traces/`, `variance_*.jsonl` | none | 08-02..04 | determinism probes | history, CLAUDE.md Settled | c |
| local `compare_data/`, `compare_data_new/`, `figures_new/`, `sweep_clean/`, `sweep_data/`, `reports/` | none | 08-01..07 | three-way comparison, ground-mu sweep, report draft | history, plot_* tools | c |

## Not experiment data

| path(s) | what | class |
|---|---|---|
| VM `ollama-models/` (28G), `ollama/` (2.1G), `.ollama/` | LLM runtime/models (08-18) | e (not this project) |
| VM `torch-cu126/` (805M), `.keras/` (341M) | wheel cache / keras datasets | d |
| VM `ai-practicum/` (49M) | unrelated project (uv, task3) | e (not this project) |
| VM `agx_navigation/` (1.5G) | repo checkout; its `run_data/` (888K) and `soak_data/` (2.9M) are small leftovers | infra |
| local `acados/` (121M), `.kilo/` | untracked scratch (acados is noted in CLAUDE.md) | d |
