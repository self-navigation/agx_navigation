#!/usr/bin/env bash
# Job 255: is the slip plant SOLVABLE? Job 253's held-out plans, seeds and patch layouts, with every
# patch at mu=mu2=0.5 instead of the stock profiles (all of which are at or below the wheel's
# mu2=0.45 steering knee). TVLQG vs TVLQR under amcl and truth. 42 plans x 3 seeds x 4 cells = 504
# runs, ~1.5 h. Config tools/jobs/c1/steerable.txt. Pairs with job 253 by (plan, seed).
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/steerable.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job255_steerable} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_test.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
