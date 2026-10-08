#!/usr/bin/env bash
# Job 242 = stop-and-aim (b): FM2 vs door-aligned plans (ours-lib serves the stored
# plan, so no live-planner change is needed), plain and gated-sideslip TVLQR, layer
# off, 2 seeds. Configs: tools/jobs/c1/bottleneck_fullstack.txt. Same harness as
# jobs 170/236/239-241. Needs ~/run_data/2026-10-08_nojob_bottleneck-bench/libs.
CONFIG_FILE=tools/jobs/c1/bottleneck_fullstack.txt \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-08_job242_bottleneck-fullstack} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/bottleneck32_base.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8"} \
SEEDS=${SEEDS:-"0 1"} \
bash tools/jobs/170_loc_factors.sh
