#!/usr/bin/env bash
# Data tidy, 2026-10-05: every run moves into ONE directory under run_data/.
# Rule: CLAUDE.md "Conventions" (run_data). Index: docs/run-data-index.md.
#
#   tools/data_tidy_plan.sh vm      # on the VM, from ~ : moves ~/<x> into ~/run_data/<dir>/<x>, symlink at ~/<x>
#   tools/data_tidy_plan.sh local   # on the laptop, from the repo root: merges the loose local copies
#                                   # into run_data/<dir>/ (VM copy wins), symlink at the old path
#
# Names: YYYY-MM-DD_job<NNN>_<slug>. Runs that predate the job queue use
# `nojob`. Classes INVALID_ / UNKNOWN_ / scratch_ replace the job tag.
# The original file/dir name is kept INSIDE the run directory, so the
# symlink and the moved data share a basename.
#
# NOTHING IS DELETED. A source that any process holds open (lsof) is skipped.
# Idempotent: an existing symlink at the source means "already moved".
set -uo pipefail

# ---------------------------------------------------------------------------
# Manifest: <run dir> <source> [<source> ...]   (VM home paths)
# ---------------------------------------------------------------------------
MANIFEST='
2026-07-24_INVALID_rl-p0-p2  checkpoints rl_corrector_p0.zip rl_corrector_p1_v2.zip rl_corrector_p1_v3.zip rl_corrector_p0_best rl_corrector_p1_v2_best rl_corrector_p1_v3_best rl_smoke_test2.zip rl_tb train_20260724_224034.log train_20260724_224728.log train_20260724_224903.log train_full1.log train_p1_kin.log train_p1_v2.log train_p1_v3.log train_p2_noslip.log
2026-07-29_nojob_plan-library-v1  pmp_trajectories_v2
2026-07-29_UNKNOWN_pmp-trajectories  pmp_trajectories
2026-07-30_INVALID_sac-1.5M  runs_20260730
2026-08-01_INVALID_tvlqr-tune  tvlqr_tune.jsonl tvlqr_tune.jsonl.pre20260802 tvlqr_tune_v2.jsonl tvlqr_tune_v3.jsonl tvlqr_tune_v4_newplant.jsonl tvlqr_tuned.json
2026-08-02_INVALID_determinism-probes  reset_probe.jsonl reset_probe2.jsonl reset_world_probe.jsonl reset_world_traces variance_fixed.jsonl variance_fixed2.jsonl variance_fixed3.jsonl variance_fixed4.jsonl variance_noterrain.jsonl variance_probe.jsonl
2026-08-12_nojob_jsweep  jsweep.jsonl jsweep.sh jtraces
2026-08-12_nojob_uturn-default-vs-tuned  uturn_default.jsonl uturn_tuned.jsonl uturn_traces
2026-08-12_nojob_seven-plan-validate  validate_20260812_default.json validate_20260812_default.jsonl validate_20260812_tuned.json validate_20260812_tuned.jsonl local2d_20260812.jsonl qwall_20260812.jsonl
2026-08-13_nojob_soak-ladders  soak_20260813_ladder.jsonl soak_20260813_twopoint.jsonl soak_20260813_uturn_subladder.jsonl soak_20260813_uturn_subladder_partial.jsonl
2026-08-13_nojob_libsweep  gaincheck gaincheck.jsonl libsweep libsweep.jsonl sweep2.sh
2026-08-14_job010_r-ladder  soak_r_ladder.jsonl r_ladder_traces
2026-08-14_job020_v2-library  traj_data_v2 candidates_v2.json
2026-08-14_job025_uturn-edge  uturn_edge.jsonl uturn_edge_traces
2026-08-14_job030_r-ladder-low  soak_r_ladder_low.jsonl r_ladder_low_traces
2026-08-15_job040_uturn-generality  soak_uturn_generality.jsonl uturn_generality_traces
2026-08-15_job050_broad-gains  broad_eval_plans.txt soak_broad_gains.jsonl broad_gains_traces
2026-08-15_job060_tune-on-J  tvlqr_tune_J.jsonl tvlqr_tuned_J.json tvlqr_validate_J_adopted.json tvlqr_validate_J_adopted.jsonl
2026-08-15_job070_broad-q  soak_broad_q.jsonl soak_broad_q.jsonl.w1 soak_broad_q.jsonl.w2 soak_broad_q.jsonl.w3 soak_broad_q.jsonl.w4 broad_q_traces
2026-08-15_job080_broad-r  soak_broad_r.jsonl soak_broad_r.jsonl.w1 soak_broad_r.jsonl.w2 soak_broad_r.jsonl.w3 soak_broad_r.jsonl.w4 broad_r_traces
2026-08-15_job090_validate-J-broad  soak_validate_J_broad.jsonl soak_validate_J_broad.jsonl.w1 soak_validate_J_broad.jsonl.w2 soak_validate_J_broad.jsonl.w3 validate_J_broad_traces
2026-08-18_job100_broad-r-at-q25  soak_broad_r_at_q25.jsonl soak_broad_r_at_q25.jsonl.w1 soak_broad_r_at_q25.jsonl.w2 soak_broad_r_at_q25.jsonl.w3 soak_broad_r_at_q25.jsonl.w4
2026-09-23_job105_gramian-broad40  gramian_broad40_q25.csv
2026-09-23_job110_seven-three-arms  soak_seven_three_arms.jsonl soak_seven_three_arms.jsonl.w1 soak_seven_three_arms.jsonl.w2 soak_seven_three_arms.jsonl.w3
2026-09-23_job120_broad40-open-loop  soak_broad_open_loop.jsonl soak_broad_open_loop.jsonl.w1 soak_broad_open_loop.jsonl.w2 soak_broad_open_loop.jsonl.w3
2026-09-23_job130_gramian-joins  gramian_broad40_identity.csv gramian_broad40_q25_job120.csv gramian_seven_q25.csv
2026-09-28_job140_smoke  compare_smoke.jsonl compare_smoke2 compare_smoke2.log
2026-09-29_INVALID_job140-compare-v1  compare_broad40 compare_broad40.log
2026-09-29_UNKNOWN_compare-v2  compare_broad40_v2 compare_broad40_v2.log
2026-09-30_job140_compare-v3  compare_broad40_v3 cmp_v3_s0.log cmp_v3_s1.log ram_v3.log
2026-09-30_nojob_rejoin-phase0  rejoin_p0_2000.jsonl rejoin_p0_2000.log rejoin_rescue.jsonl rejoin_rescue.log
2026-09-30_UNKNOWN_bias27  bias27
2026-09-30_scratch_rtf-probe  rtf_probe_camsoff rtf_probe_a.log rtf_probe_b.log
2026-10-01_job140_compare-v4  compare_broad40_v4 cmp_v4_s0.log cmp_v4_s1.log
2026-10-01_job140_compare-v5  compare_broad40_v5 cmp_v5_s0.log cmp_v5_s1.log freeze_watch_v5.log
2026-10-01_scratch_field-probes  fieldprobe.jsonl fieldprobe.log fieldprobe_logs fieldprobe_plans.txt one_plan.txt fp2.jsonl fp2.log fp2_logs fp3.jsonl fp3.log fp3_logs fp4.jsonl fp4.log fp4_logs fp5.jsonl fp5.log fp5_logs fp6.jsonl fp6.log fp6_logs fp7.jsonl fp7.log fp7_logs
2026-10-01_scratch_pyspy  .pyspy spy_planner.txt spy_watch.sh spy_watch.log build_ex.log
2026-10-01_scratch_compare-probes  cg28_smoke.jsonl compare_mp_probe8.jsonl
2026-10-02_job140_compare-v6  compare_broad40_v6 cmp_v6_s0.log cmp_v6_s1.log freeze_watch_v6.log
2026-10-02_job140_compare-v7  compare_broad40_v7 cmp_v7_s0.log cmp_v7_s1.log
2026-10-02_UNKNOWN_sigtrace  sigtrace.bt sigtrace_v7.log
2026-10-02_job150_live-vs-lib  live40 live_vs_lib.txt soak_live_vs_lib.jsonl soak_live_vs_lib.jsonl.w1 soak_live_vs_lib_traces
2026-10-02_job160_stack-factors  compare_factors
2026-10-03_UNKNOWN_dep-series  dep_series
2026-10-03_job170_loc-factors  compare_loc compare_loc_smoke
2026-10-03_job180_amcl-sweep  compare_amcl compare_amcl_smoke
'
# MOVE AFTER JOBS 190/200 FINISH (~03:00 2026-10-06). Not in the manifest on purpose:
#   2026-10-05_job190_lidar-odom   compare_lidar
#   2026-10-05_job200_hybrid       compare_hybrid
# Left in place, not project data or not ours: ollama ollama-models .ollama
# torch-cu126 .keras ai-practicum agx_navigation; queue infra jobq jobq.sh
# (the runner and `just queue-*` use ~/jobq); /tmp (agx-run logs, smoke*).

mv_link() {  # mv_link <root> <rundir> <src>
  local root=$1 dir=$2 src=$3
  if [ -L "$src" ]; then echo "done    $src"; return 0; fi
  if [ ! -e "$src" ]; then echo "MISSING $src"; return 0; fi
  if [ -n "$(lsof +D "$src" 2>/dev/null | tail -n +2; lsof "$src" 2>/dev/null | tail -n +2)" ]; then
    echo "OPEN    $src (skipped)"; return 0; fi
  mkdir -p "$root/$dir"
  if [ -e "$root/$dir/$(basename "$src")" ]; then echo "EXISTS  $root/$dir/$src (skipped)"; return 0; fi
  mv "$src" "$root/$dir/" && ln -s "$root/$dir/$(basename "$src")" "$src" && echo "moved   $src -> $dir/"
}

vm() {
  cd "$HOME" || exit 1
  echo "$MANIFEST" | while read -r dir srcs; do
    [ -n "$dir" ] || continue
    for s in $srcs; do mv_link "$HOME/run_data" "$dir" "$s"; done
  done
}

# merge <local old path> <target path inside run_data>: VM copy wins; a local
# file that differs from the VM's is kept beside it as <name>.laptop; then the
# old path becomes a symlink. Nothing is lost.
merge() {
  local old=$1 tgt=$2
  if [ -L "$old" ]; then echo "done    $old"; return 0; fi
  [ -e "$old" ] || { echo "MISSING $old"; return 0; }
  mkdir -p "$(dirname "$tgt")"
  if [ -d "$old" ]; then
    mkdir -p "$tgt"
    rsync -a --ignore-existing "$old/" "$tgt/"
    rsync -rcn --out-format='%n' "$old/" "$tgt/" | grep -v '/$' | while read -r f; do
      cp -p "$old/$f" "$tgt/$f.laptop"; echo "        differs, kept $tgt/$f.laptop"; done
  else
    if [ ! -e "$tgt" ]; then cp -p "$old" "$tgt"
    elif ! cmp -s "$old" "$tgt"; then cp -p "$old" "$tgt.laptop"; echo "        differs, kept $tgt.laptop"; fi
  fi
  # verify every old file now exists in the target before replacing it
  if [ -d "$old" ]; then
    local miss; miss=$(cd "$old" && find . -type f | while read -r f; do [ -e "$tgt/$f" ] || echo "$f"; done)
    [ -z "$miss" ] || { echo "VERIFY FAILED $old: $miss"; return 1; }
    rm -rf "$old"
  else
    rm -f "$old"
  fi
  ln -s "$(realpath --relative-to="$(dirname "$old")" "$tgt")" "$old" && echo "merged  $old -> $tgt"
}

local_side() {
  local R=run_data
  [ -d .git ] || { echo "run from the repo root"; exit 1; }
  # old local run_data/ subdirs (fetched copies of VM runs, or local-only)
  merge $R/compare_seed0     $R/2026-09-29_INVALID_job140-compare-v1/compare_broad40/seed0
  merge $R/compare_seed1     $R/2026-09-29_INVALID_job140-compare-v1/compare_broad40/seed1
  merge $R/compare_v3_seed0  $R/2026-09-30_job140_compare-v3/compare_broad40_v3/seed0
  merge $R/compare_v3_seed1  $R/2026-09-30_job140_compare-v3/compare_broad40_v3/seed1
  merge $R/compare_v4_seed0  $R/2026-10-01_job140_compare-v4/compare_broad40_v4/seed0
  merge $R/compare_v4_seed1  $R/2026-10-01_job140_compare-v4/compare_broad40_v4/seed1
  merge $R/compare_v5        $R/2026-10-01_job140_compare-v5/compare_broad40_v5
  merge $R/compare_v7_seed0  $R/2026-10-02_job140_compare-v7/compare_broad40_v7/seed0
  merge $R/compare_v7_seed1  $R/2026-10-02_job140_compare-v7/compare_broad40_v7/seed1
  merge $R/compare_tune3     $R/2026-09-29_job140_nav2-tuning/compare_tune3
  merge $R/compare_tune4     $R/2026-09-29_job140_nav2-tuning/compare_tune4
  merge $R/job160            $R/2026-10-02_job160_stack-factors/job160
  merge $R/loc170            $R/2026-10-03_job170_loc-factors/compare_loc
  merge $R/amcl180           $R/2026-10-03_job180_amcl-sweep/compare_amcl
  merge $R/bias27            $R/2026-09-30_UNKNOWN_bias27/bias27
  merge $R/rescue            $R/2026-09-30_nojob_rejoin-phase0/rescue
  for f in $R/identity_*.* $R/tvlqr_*.*; do
    [ -e "$f" ] && merge "$f" "$R/2026-07-25_INVALID_fixture-run-recorder/$(basename "$f")"; done
  # loose top-level dirs
  merge traj_data            $R/2026-07-29_nojob_plan-library-v1/pmp_trajectories_v2
  merge traj_data_v2         $R/2026-08-14_job020_v2-library/traj_data_v2
  merge jtraces              $R/2026-08-12_nojob_jsweep/jtraces
  merge uturn_traces         $R/2026-08-12_nojob_uturn-default-vs-tuned/uturn_traces
  merge gaincheck            $R/2026-08-13_nojob_libsweep/gaincheck
  merge libsweep             $R/2026-08-13_nojob_libsweep/libsweep
  merge r_ladder_traces      $R/2026-08-14_job010_r-ladder/r_ladder_traces
  merge broad_gains_traces   $R/2026-08-15_job050_broad-gains/broad_gains_traces
  merge epsilon_data         $R/2026-08-13_nojob_epsilon-scores/epsilon_data
  merge tb_data              $R/2026-07-30_INVALID_sac-1.5M/tb_data
  merge compare_data         $R/2026-08-01_INVALID_three-way-compare/compare_data
  merge compare_data_new     $R/2026-08-01_INVALID_three-way-compare/compare_data_new
  merge figures_new          $R/2026-08-01_INVALID_three-way-compare/figures_new
  merge sweep_data           $R/2026-08-01_INVALID_sweeps/sweep_data
  merge sweep_clean          $R/2026-08-01_INVALID_sweeps/sweep_clean
  merge figures/tmp/cmp_s0   $R/2026-09-29_INVALID_job140-compare-v1/summary_laptop/seed0
  merge figures/tmp/cmp_s1   $R/2026-09-29_INVALID_job140-compare-v1/summary_laptop/seed1
  merge figures/tmp/v7s0     $R/2026-10-02_job140_compare-v7/summary_laptop/seed0
  merge figures/tmp/v7s1     $R/2026-10-02_job140_compare-v7/summary_laptop/seed1
  # soak_data/ and tune_data/ stay directories (code globs them); each FILE
  # becomes a symlink into its run dir.
  local -A SOAK=(
    [gramian_broad40_identity.csv]=2026-09-23_job130_gramian-joins [gramian_broad40_q25_job120.csv]=2026-09-23_job130_gramian-joins
    [gramian_seven_q25.csv]=2026-09-23_job130_gramian-joins [gramian_broad40_q25.csv]=2026-09-23_job105_gramian-broad40
    [libsweep.jsonl]=2026-08-13_nojob_libsweep
    [soak_20260813_ladder.jsonl]=2026-08-13_nojob_soak-ladders [soak_20260813_twopoint.jsonl]=2026-08-13_nojob_soak-ladders
    [soak_20260813_uturn_subladder.jsonl]=2026-08-13_nojob_soak-ladders
    [soak_broad_gains.jsonl]=2026-08-15_job050_broad-gains [soak_broad_open_loop.jsonl]=2026-09-23_job120_broad40-open-loop
    [soak_broad_q.jsonl]=2026-08-15_job070_broad-q [soak_broad_r_at_q25.jsonl]=2026-08-18_job100_broad-r-at-q25
    [soak_broad_r.jsonl]=2026-08-15_job080_broad-r [soak_live_vs_lib.jsonl]=2026-10-02_job150_live-vs-lib
    [soak_r_ladder.jsonl]=2026-08-14_job010_r-ladder [soak_r_ladder_low.jsonl]=2026-08-14_job030_r-ladder-low
    [soak_seven_three_arms.jsonl]=2026-09-23_job110_seven-three-arms [soak_uturn_generality.jsonl]=2026-08-15_job040_uturn-generality
    [soak_validate_J_broad.jsonl]=2026-08-15_job090_validate-J-broad
    [uturn_default.jsonl]=2026-08-12_nojob_uturn-default-vs-tuned [uturn_tuned.jsonl]=2026-08-12_nojob_uturn-default-vs-tuned
    [uturn_edge.jsonl]=2026-08-14_job025_uturn-edge
  )
  for f in "${!SOAK[@]}"; do merge "soak_data/$f" "$R/${SOAK[$f]}/$f"; done
  local -A TUNE=(
    [local2d_20260812.jsonl]=2026-08-12_nojob_seven-plan-validate [qwall_20260812.jsonl]=2026-08-12_nojob_seven-plan-validate
    [validate_20260812_default.jsonl]=2026-08-12_nojob_seven-plan-validate [validate_20260812_tuned.jsonl]=2026-08-12_nojob_seven-plan-validate
    [reset_probe.jsonl]=2026-08-02_INVALID_determinism-probes [variance_noterrain.jsonl]=2026-08-02_INVALID_determinism-probes
    [variance_probe.jsonl]=2026-08-02_INVALID_determinism-probes
    [tvlqr_tune.jsonl]=2026-08-01_INVALID_tvlqr-tune [tvlqr_tune_v4_newplant.jsonl]=2026-08-01_INVALID_tvlqr-tune
    [tvlqr_tuned.json]=2026-08-01_INVALID_tvlqr-tune [tune_v4_progress.log]=2026-08-01_INVALID_tvlqr-tune
  )
  for f in "${!TUNE[@]}"; do merge "tune_data/$f" "$R/${TUNE[$f]}/$f"; done
}

case "${1:-}" in
  vm) vm ;;
  local) local_side ;;
  *) echo "usage: $0 vm|local"; exit 1 ;;
esac
