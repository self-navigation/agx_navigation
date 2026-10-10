#!/usr/bin/env bash
# Job 246: amcl tuning (motion model / gate / likelihood) under TVLQG.
# Configs: tools/jobs/c1/amcl_tune.txt, params tools/jobs/amcl_tune/. Same harness
# as jobs 170/236/239-245.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/amcl_tune.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-10_job246_amcl-tune} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/broad34c.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
