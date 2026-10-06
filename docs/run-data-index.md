# Run-data index

All run data lives in one tree: `~/run_data/` on the VM, mirrored to the laptop's
gitignored `run_data/` (rule: CLAUDE.md, Conventions). One directory per run,
named `YYYY-MM-DD_job<NNN>_<slug>`. Runs from before the job queue use `nojob`.
The tags `INVALID_`, `UNKNOWN_` and `scratch_` replace the job tag. Every
directory has a `README.md`.

**Migration status (2026-10-05).** The tidy was done by
[tools/data_tidy_plan.sh](../tools/data_tidy_plan.sh).
- **VM:** done, `vm` mode. 46 directories and 184 entries were moved, and each old `~/<name>` is a symlink into its run directory. Nothing was deleted.
- **Laptop:** `rsync -a` from the VM is done.
- **Not yet done:** the `local` merge, which folds the old loose copies into these directories and leaves symlinks. The paths listed under "old local path" are still real directories until it runs.
- **Still to move once jobs 190/200 finish (~03:00 2026-10-06):** `~/compare_lidar` and `~/compare_hybrid`.

Classes:
- **a**: cited by the paper or a committed figure.
- **b**: cited by the handover, an issue or the docs.
- **c**: superseded or invalid (before 2026-08-02 evening; plant changes on 08-04 and 08-07).
- **d**: scratch or smoke.
- **e**: unknown.

**L** = laptop-only before the tidy.

| run dir (`run_data/...`) | old VM path(s) `~/` | old local path(s) | job | what | cited by | class |
|---|---|---|---|---|---|---|
| 2026-07-24_INVALID_rl-p0-p2 | checkpoints, rl_corrector_p*, rl_smoke_test2.zip, rl_tb, train_*.log | | none | SAC RL corrector curriculum p0-p2 | CLAUDE.md Settled, Justfile, plot_checkpoint* | c |
| 2026-07-25_INVALID_fixture-run-recorder | | run_data/{identity,tvlqr}_* (L) | none | early fixture run_recorder CSVs | plot_run.py docstring | c |
| 2026-07-29_nojob_plan-library-v1 | pmp_trajectories_v2 | traj_data | none | v1 plan library incl. the seven shapes (input) | figures 08-13/08-15, Justfile, jobs 25/40 | a |
| 2026-07-29_UNKNOWN_pmp-trajectories | pmp_trajectories | | ? | 416-file plan dir | nothing | e |
| 2026-07-30_INVALID_sac-1.5M | runs_20260730 | tb_data (L) | none | 1.5M-step SAC run | CLAUDE.md Settled, history | c |
| 2026-08-01_INVALID_tvlqr-tune | tvlqr_tune*, tvlqr_tuned.json | tune_data/tvlqr_*, tune_v4_progress.log | none | early 7-plan TVLQR tuning | history, figures/archive | c |
| 2026-08-01_INVALID_three-way-compare | | compare_data, compare_data_new, figures_new (L) | none | identity/TVLQR/RL comparison | history, plot_corrector_summary.py | c |
| 2026-08-01_INVALID_sweeps | | sweep_data, sweep_clean (L) | none | checkpoint and ground-mu sweeps | history, measurement-rig.md | c |
| 2026-08-02_INVALID_determinism-probes | reset_*probe*, reset_world_traces, variance_* | tune_data/{reset,variance}_* | none | determinism probes | CLAUDE.md Settled, history | c |
| 2026-08-12_nojob_jsweep | jsweep.*, jtraces | jtraces | none | J sweep | figures/2026-08-13 | a |
| 2026-08-12_nojob_uturn-default-vs-tuned | uturn_default/tuned.jsonl, uturn_traces | uturn_traces, soak_data/uturn_{default,tuned} | none | U-turn default vs tuned | figures/2026-08-13 | a |
| 2026-08-12_nojob_seven-plan-validate | validate_20260812_*, local2d_*, qwall_* | tune_data/ same | none | 7-plan tune validation | history, plot_tune_validation.py | b |
| 2026-08-13_nojob_soak-ladders | soak_20260813_* | soak_data/soak_20260813_* | none | q ladder, two-point, U-turn subladder | figures/2026-08-13, archive ladder_modes (paper) | a |
| 2026-08-13_nojob_libsweep | gaincheck*, libsweep*, sweep2.sh | gaincheck, libsweep, soak_data/libsweep.jsonl | none | 51-plan library sweep | CLAUDE.md, history | b |
| 2026-08-13_nojob_epsilon-scores | | epsilon_data (L) | none | J scores of the trace sets | figures/2026-08-13 | a |
| 2026-08-14_job010_r-ladder | soak_r_ladder.jsonl, r_ladder_traces | r_ladder_traces, soak_data/ | 10 | r ladder at q=0.276 | figures/2026-08-14 | a |
| 2026-08-14_job020_v2-library | traj_data_v2, candidates_v2.json | traj_data_v2 | 20 | v2 plan library = broad 40 (input) | many tools, figures 08-15/10-02 | a |
| 2026-08-14_job025_uturn-edge | uturn_edge.*, uturn_edge_traces | soak_data/uturn_edge.jsonl | 25 | U-turn notch edge traces | jobs/25 | b |
| 2026-08-14_job030_r-ladder-low | soak_r_ladder_low.jsonl, r_ladder_low_traces | soak_data/ | 30 | low r ladder | figures/2026-08-15 | a |
| 2026-08-15_job040_uturn-generality | soak_uturn_generality.jsonl, uturn_generality_traces | soak_data/ | 40 | U-turn basin generality | figures/2026-08-15, CLAUDE.md | a |
| 2026-08-15_job050_broad-gains | broad_eval_plans.txt, soak_broad_gains.jsonl, broad_gains_traces | broad_gains_traces, soak_data/ | 50 | broad gain generality | history, rejoin_phase0.py | b |
| 2026-08-15_job060_tune-on-J | tvlqr_tune_J*, tvlqr_tuned_J.json, tvlqr_validate_J_adopted.* | | 60 | tune on J | jobs/60, 90 | b |
| 2026-08-15_job070_broad-q | soak_broad_q.jsonl(+.w*), broad_q_traces | soak_data/ | 70 | broad q ladder | CLAUDE.md gains table | b |
| 2026-08-15_job080_broad-r | soak_broad_r.jsonl(+.w*), broad_r_traces | soak_data/ | 80 | broad r ladder | CLAUDE.md | b |
| 2026-08-15_job090_validate-J-broad | soak_validate_J_broad.jsonl(+.w*), validate_J_broad_traces | soak_data/ | 90 | J-tune validation on broad | CLAUDE.md | b |
| 2026-08-18_job100_broad-r-at-q25 | soak_broad_r_at_q25.jsonl(+.w*) | soak_data/ | 100 | gain adoption run | CLAUDE.md | b |
| 2026-09-23_job105_gramian-broad40 | gramian_broad40_q25.csv | soak_data/ | 105 | controllability vs J | CLAUDE.md Settled | b |
| 2026-09-23_job110_seven-three-arms | soak_seven_three_arms.jsonl(+.w*) | soak_data/ | 110 | seven shapes, three arms | CLAUDE.md | b |
| 2026-09-23_job120_broad40-open-loop | soak_broad_open_loop.jsonl(+.w*) | soak_data/ | 120 | broad 40 open loop | CLAUDE.md | b |
| 2026-09-23_job130_gramian-joins | gramian_{seven_q25,broad40_identity,broad40_q25_job120}.csv | soak_data/ | 130 | Gramian on jobs 110/120 | jobs/130 | b |
| 2026-09-28_job140_smoke | compare_smoke.jsonl, compare_smoke2(.log) | | 140 | harness smoke | jobs/140 | d |
| 2026-09-29_job140_nav2-tuning | | run_data/compare_tune{3,4} (L) | 140 | Nav2 profile tuning | #10 | b |
| 2026-09-29_INVALID_job140-compare-v1 | compare_broad40(.log) | run_data/compare_seed{0,1}, figures/tmp/cmp_s* | 140 | comparison v1 (amcl heartbeat bug) | figures/2026-09-29, #10 | a/c |
| 2026-09-29_UNKNOWN_compare-v2 | compare_broad40_v2(.log) | | 140? | 75-file aborted(?) v2 | nothing | e |
| 2026-09-30_job140_compare-v3 | compare_broad40_v3, cmp_v3_*.log, ram_v3.log | run_data/compare_v3_seed* | 140 | v3, freezes | #32 | b |
| 2026-09-30_nojob_rejoin-phase0 | rejoin_p0_2000.*, rejoin_rescue.* | run_data/rescue | none | re-join Phase 0 | figures/2026-09-30 (paper), #11 | a |
| 2026-09-30_UNKNOWN_bias27 | bias27 | run_data/bias27 | ? | identity vs tvlqr, worker 7 | nothing | e |
| 2026-09-30_scratch_rtf-probe | rtf_probe_* | | | RTF probes | nothing | d |
| 2026-10-01_job140_compare-v4 | compare_broad40_v4, cmp_v4_*.log | run_data/compare_v4_seed* | 140 | v4, freeze analysis source | #32 | b |
| 2026-10-01_job140_compare-v5 | compare_broad40_v5, cmp_v5_*.log, freeze_watch_v5.log | run_data/compare_v5 | 140 | v5, live freeze captures | #32 | b |
| 2026-10-01_scratch_field-probes | fieldprobe*, fp2-7*, one_plan.txt | | | field probes | nothing | d |
| 2026-10-01_scratch_pyspy | .pyspy, spy_*, build_ex.log | | | py-spy dumps | freeze_watch.py | d |
| 2026-10-01_scratch_compare-probes | cg28_smoke.jsonl, compare_mp_probe8.jsonl | | | harness probes | nothing | d |
| 2026-10-02_job140_compare-v6 | compare_broad40_v6, cmp_v6_*.log, freeze_watch_v6.log | | 140 | SHM A/B | #32 | b |
| 2026-10-02_job140_compare-v7 | compare_broad40_v7, cmp_v7_*.log | run_data/compare_v7_seed*, figures/tmp/v7s* | 140 | v7, first clean comparison | figures/2026-10-02, #34 | a |
| 2026-10-02_UNKNOWN_sigtrace | sigtrace.bt, sigtrace_v7.log | | ? | bpftrace during v7 | nothing | e |
| 2026-10-02_job150_live-vs-lib | live40, live_vs_lib.txt, soak_live_vs_lib*(+traces) | soak_data/soak_live_vs_lib.jsonl | 150 | live vs library plans, bare ground | handover, #34 | b |
| 2026-10-02_job160_stack-factors | compare_factors | run_data/job160 | 160 | stack-factor factorial | figures/2026-10-03, handover | a |
| 2026-10-03_UNKNOWN_dep-series | dep_series | | ? | departure series(?) | nothing | e |
| 2026-10-03_job170_loc-factors | compare_loc, compare_loc_smoke | run_data/loc170 | 170 | localization factors | handover, #34 | b |
| 2026-10-03_job180_amcl-sweep | compare_amcl, compare_amcl_smoke | run_data/amcl180 | 180 | amcl x EKF factorial | handover, #34 | b |
| 2026-10-05_job190_lidar-odom | compare_lidar | | 190 | rf2o lidar odom in EKF (negative) | paper sec:stack, #34 | b |
| 2026-10-05_job200_hybrid | compare_hybrid | | 200 | Hybrid-A*+MPPI, SUPERSEDED by 220 (MPPI stall) | #33 | c |
| 2026-10-05_job210_pmp-trackers | | | 210 | Nav2 MPPI/RPP on PMP plan vs ours (ours on live plan) | #10 | b |
| 2026-10-06_job220_fullstack | | | 220 | full stacks: ours vs Smac2D/Hybrid+MPPI, RPP | #10, #33 | live |

Not moved, not run data: `~/agx_navigation`, `~/jobq` + `~/jobq.sh` (queue
infrastructure), `ollama*`, `torch-cu126`, `.keras`, `ai-practicum` (unrelated
project), `/tmp` (agx-run logs, smoke*). Local `acados/`, `.kilo/` and
`reports/` are scratch or docs and were left in place.
