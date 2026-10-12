#!/usr/bin/env bash
# Job 259: the 3x3 main grid (+ fixed-k TVLQR) replicated on v3f6_dev (41 plans), seeds 0-1, mu 0.5. 12 cells, 984 runs.
# Runs beside job 260 (workers 9-12), so this one takes workers 1-8.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/dev_replicate.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job259_dev-replicate} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_dev.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8"} \
SEEDS=${SEEDS:-"0 1"} \
bash tools/jobs/170_loc_factors.sh
