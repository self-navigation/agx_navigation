#!/usr/bin/env bash
# Job 256: job 255 (mu 0.5 patches, held-out plans) under rtab: does better localization remove the
# remaining doorway contacts? 42 plans x 3 seeds x 2 cells = 252 runs, ~45 min. Pairs with 255 by (plan, seed).
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/steerable_rtab.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job256_steerable-rtab} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_test.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
