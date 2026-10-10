#!/usr/bin/env bash
# Job 245 = weekend plan J1: re-join fixes (wall-crossing gate, heading/tail
# hold) under the truth pose vs plain TVLQR and TVLQG. Configs:
# tools/jobs/c1/rejoin_fixes.txt. Same harness as jobs 170/236/239-244.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/rejoin_fixes.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-10_job245_rejoin-fixes} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/broad34c.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
