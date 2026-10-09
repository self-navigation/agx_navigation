#!/usr/bin/env bash
# Job 244 = TVLQG scenario (b): deviation-triggered PMP re-join with the
# threshold derived from TVLQG's covariance, plus TVLQG under amcl.
# Configs: tools/jobs/c1/tvlqg_deviation.txt. Same harness as jobs 170/236/239-243.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/tvlqg_deviation.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-09_job244_tvlqg-deviation} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/broad34c.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
