#!/usr/bin/env bash
# PROPOSED data tidy (2026-10-05). NOT EXECUTED. Review before running.
# Index: docs/run-data-index.md.
#
# Layout: ~/data/<YYYY-MM-DD>_job<NNN>_<slug>/ on the VM, mirrored locally under
# data/ (gitignored). The old path stays as a symlink, so every script, doc and
# renderer that names it keeps working. Path references to update later (they
# still work via the symlinks):
#   tools/jobs/*.sh OUT=/OUT_DIR= defaults (jobs 10-200), Justfile fetch recipes,
#   figures/2026-08-13..10-03 render.py/README.md (epsilon_data, jtraces,
#   uturn_traces, soak_data/*, traj_data*, run_data/compare_*, run_data/job160,
#   ~/compare_broad40_v7, ~/compare_factors), tools/rejoin_phase0.py,
#   tools/summarize_compare.py, tools/plot_*.py, handover.md (loc170, amcl180,
#   compare_loc, compare_amcl, soak_live_vs_lib), docs/corrector-history.md,
#   CLAUDE.md (soak_data), issues #10 #32 #34.
# Do NOT touch: compare_lidar, compare_hybrid, /tmp/smoke* (live jobs 190/200).
set -euo pipefail

mv_link() {  # mv_link <src> <dest-dir>  : move src into dest, leave a symlink
  local src=$1 dst=$2
  [ -e "$src" ] && [ ! -L "$src" ] || { echo "skip $src"; return; }
  mkdir -p "$dst"
  mv -n "$src" "$dst/"
  ln -s "$dst/$(basename "$src")" "$src"
}

vm() {
  cd "$HOME"
  D=$HOME/data
  mv_link compare_smoke.jsonl            $D/2026-09-28_job140_smoke
  mv_link compare_smoke2                 $D/2026-09-28_job140_smoke
  mv_link compare_smoke2.log             $D/2026-09-28_job140_smoke
  mv_link compare_broad40                $D/2026-09-29_job140_compare-v1-INVALID
  mv_link compare_broad40.log            $D/2026-09-29_job140_compare-v1-INVALID
  mv_link compare_broad40_v2             $D/2026-09-29_job140_compare-v2
  mv_link compare_broad40_v2.log         $D/2026-09-29_job140_compare-v2
  for v in 3 4 5 6 7; do
    d=$(ls -d compare_broad40_v$v >/dev/null && date -r compare_broad40_v$v +%F)
    mv_link compare_broad40_v$v          $D/${d}_job140_compare-v$v
    for f in cmp_v${v}_s*.log freeze_watch_v$v.log; do [ -e "$f" ] && mv_link "$f" $D/${d}_job140_compare-v$v; done
  done
  mv_link ram_v3.log                     $D/2026-09-30_job140_compare-v3
  mv_link sigtrace.bt                    $D/2026-10-02_issue32_sigtrace
  mv_link sigtrace_v7.log                $D/2026-10-02_issue32_sigtrace
  for f in live40 live_vs_lib.txt soak_live_vs_lib.jsonl soak_live_vs_lib.jsonl.w1 soak_live_vs_lib_traces; do
    mv_link $f $D/2026-10-02_job150_live-vs-lib; done
  mv_link compare_factors                $D/2026-10-02_job160_stack-factors
  mv_link dep_series                     $D/2026-10-03_UNKNOWN_dep-series
  for f in compare_loc compare_loc_smoke; do mv_link $f $D/2026-10-03_job170_loc-factors; done
  for f in compare_amcl compare_amcl_smoke; do mv_link $f $D/2026-10-03_job180_amcl-sweep; done
  for f in rejoin_p0_2000.jsonl rejoin_p0_2000.log rejoin_rescue.jsonl rejoin_rescue.log; do
    mv_link $f $D/2026-09-30_issue11_rejoin-phase0; done
  mv_link bias27                         $D/2026-09-30_UNKNOWN_bias27
  for f in rtf_probe_camsoff rtf_probe_a.log rtf_probe_b.log fieldprobe.jsonl fieldprobe.log fieldprobe_logs \
           fieldprobe_plans.txt one_plan.txt fp{2,3,4,5,6,7}.jsonl fp{2,3,4,5,6,7}.log fp{2,3,4,5,6,7}_logs \
           spy_planner.txt spy_watch.sh spy_watch.log build_ex.log cg28_smoke.jsonl compare_mp_probe8.jsonl; do
    mv_link $f $D/2026-10-01_scratch_probes; done
  # Aug soak era (valid plant)
  for f in jsweep.jsonl jsweep.sh jtraces; do mv_link $f $D/2026-08-12_jsweep; done
  for f in uturn_default.jsonl uturn_tuned.jsonl uturn_traces; do mv_link $f $D/2026-08-12_uturn-default-vs-tuned; done
  for f in soak_20260813_*.jsonl; do mv_link $f $D/2026-08-13_soak-ladders; done
  for f in validate_20260812_* local2d_20260812.jsonl qwall_20260812.jsonl; do mv_link $f $D/2026-08-12_seven-plan-validate; done
  for f in gaincheck gaincheck.jsonl libsweep libsweep.jsonl sweep2.sh; do mv_link $f $D/2026-08-13_libsweep; done
  for f in traj_data_v2 candidates_v2.json; do mv_link $f $D/2026-08-14_job020_v2-library; done
  for f in soak_r_ladder.jsonl r_ladder_traces; do mv_link $f $D/2026-08-14_job010_r-ladder; done
  for f in uturn_edge.jsonl uturn_edge_traces; do mv_link $f $D/2026-08-14_job025_uturn-edge; done
  for f in soak_r_ladder_low.jsonl r_ladder_low_traces; do mv_link $f $D/2026-08-14_job030_r-ladder-low; done
  for f in soak_uturn_generality.jsonl uturn_generality_traces; do mv_link $f $D/2026-08-15_job040_uturn-generality; done
  for f in broad_eval_plans.txt soak_broad_gains.jsonl broad_gains_traces; do mv_link $f $D/2026-08-15_job050_broad-gains; done
  for f in tvlqr_tune_J.jsonl tvlqr_tuned_J.json tvlqr_validate_J_adopted.json tvlqr_validate_J_adopted.jsonl; do
    mv_link $f $D/2026-08-15_job060_tune-on-J; done
  for f in soak_broad_q.jsonl* broad_q_traces; do mv_link $f $D/2026-08-15_job070_broad-q; done
  for f in soak_broad_r.jsonl soak_broad_r.jsonl.w* broad_r_traces; do mv_link $f $D/2026-08-15_job080_broad-r; done
  for f in soak_validate_J_broad.jsonl* validate_J_broad_traces; do mv_link $f $D/2026-08-15_job090_validate-J-broad; done
  for f in soak_broad_r_at_q25.jsonl*; do mv_link $f $D/2026-08-18_job100_broad-r-at-q25; done
  for f in gramian_*.csv; do mv_link $f $D/2026-09-23_job105-130_gramian; done
  for f in soak_seven_three_arms.jsonl*; do mv_link $f $D/2026-09-23_job110_seven-three-arms; done
  for f in soak_broad_open_loop.jsonl*; do mv_link $f $D/2026-09-23_job120_broad-open-loop; done
  # Invalid era (kept for the Settled stubs)
  for f in checkpoints rl_corrector_p0.zip rl_corrector_p1_v2.zip rl_corrector_p1_v3.zip rl_corrector_*_best \
           rl_smoke_test2.zip rl_tb train_*.log; do mv_link $f $D/2026-07-24_INVALID_rl-p0-p2; done
  mv_link runs_20260730                  $D/2026-07-30_INVALID_sac-1.5M
  for f in tvlqr_tune.jsonl tvlqr_tune.jsonl.pre20260802 tvlqr_tune_v2.jsonl tvlqr_tune_v3.jsonl \
           tvlqr_tune_v4_newplant.jsonl tvlqr_tuned.json; do mv_link $f $D/2026-08-01_INVALID_tvlqr-tune; done
  for f in reset_probe.jsonl reset_probe2.jsonl reset_world_probe.jsonl reset_world_traces variance_*.jsonl; do
    mv_link $f $D/2026-08-02_INVALID_determinism-probes; done
  for f in pmp_trajectories pmp_trajectories_v2; do mv_link $f $D/2026-07-29_plan-library-v1; done
}

local_side() {  # run from the repo root on the laptop; data/ must be gitignored first
  D=$PWD/data
  for f in compare_data compare_data_new figures_new sweep_clean sweep_data reports tb_data tune_data; do
    mv_link $f $D/2026-08-0x_INVALID_$f; done
  mv_link traj_data          $D/2026-07-29_plan-library-v1
  mv_link traj_data_v2       $D/2026-08-14_job020_v2-library
  mv_link epsilon_data       $D/2026-08-13_epsilon
  mv_link jtraces            $D/2026-08-12_jsweep
  mv_link uturn_traces       $D/2026-08-12_uturn-default-vs-tuned
  mv_link gaincheck          $D/2026-08-13_libsweep
  mv_link libsweep           $D/2026-08-13_libsweep
  mv_link r_ladder_traces    $D/2026-08-14_job010_r-ladder
  mv_link broad_gains_traces $D/2026-08-15_job050_broad-gains
  # soak_data/ and run_data/ stay as they are (cited everywhere); subdirs map 1:1 to VM dirs above.
}

case "${1:-}" in vm) vm ;; local) local_side ;; *) echo "usage: $0 vm|local  (REVIEW FIRST)"; exit 1 ;; esac

# ---------------------------------------------------------------------------
# DELETIONS: separate approval needed; all commented out.
# ---------------------------------------------------------------------------
# VM, merged copy exists:   rm ~/data/*/soak_*.jsonl.w[0-9]
# VM, partial / smoke:      rm ~/soak_20260813_uturn_subladder_partial.jsonl
#                           rm -r ~/data/2026-09-28_job140_smoke ~/compare_loc_smoke ~/compare_amcl_smoke
# VM, scratch probes:       rm -r ~/data/2026-10-01_scratch_probes ~/.pyspy
# VM, caches (not data):    rm -r ~/torch-cu126 ~/.keras
# VM, not this project (ask the owner): ~/ollama-models (28G) ~/ollama ~/ai-practicum
# local scratch:            rm -r figures/tmp/cmp_s0 figures/tmp/cmp_s1
