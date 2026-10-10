#!/usr/bin/env bash
# Job 247: TVLQR vs frozen TVLQG, truth and amcl, on the DEV half of plan set v3
# floor 6 (41 plans x seeds 0-2). Config tools/jobs/c1/final_dev.txt. Same harness
# as jobs 170/236/239-246. The held-out half (tools/jobs/v3f6_test.txt) is the
# final run, after any dev-set tuning.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/final_dev.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-10_job247_final-dev} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_dev.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
