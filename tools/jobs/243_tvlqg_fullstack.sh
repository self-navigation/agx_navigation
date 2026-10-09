#!/usr/bin/env bash
# Job 243 = TVLQG step 2 (estimated slide) in the full stack against plain and
# typed-in-k TVLQR, layer off, 2 seeds. Configs: tools/jobs/c1/tvlqg_fullstack.txt.
# Same harness as jobs 170/236/239-241. Run from a checkout built at or after ce276e5.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/tvlqg_fullstack.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-09_job243_tvlqg-fullstack} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/broad34c.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8"} \
SEEDS=${SEEDS:-"0 1"} \
bash tools/jobs/170_loc_factors.sh
