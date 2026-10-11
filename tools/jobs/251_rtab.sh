#!/usr/bin/env bash
# Job 251: lidar-only rtabmap localization (localization:=rtab, d6927d5) vs amcl and
# truth, each under TVLQG and TVLQR, on the DEV half of plan set v3 floor 6
# (41 plans x seeds 0-2 x 6 cells = 738 runs, ~2 h on 12 workers). Config
# tools/jobs/c1/rtab.txt. The held-out half (tools/jobs/v3f6_test.txt) stays untouched.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/rtab.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-11_job251_rtab-dev} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_dev.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
