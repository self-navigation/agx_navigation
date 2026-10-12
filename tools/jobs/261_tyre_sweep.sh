#!/usr/bin/env bash
# Job 261: tyre lateral friction (wheel mu2) calibration sweep on plain ground, TVLQR + truth, v3f6_dev seed 0.
# 5 cells x 41 plans = 205 runs, 12 workers.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/tyre_sweep.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job261_tyre-sweep} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_dev.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0"} \
bash tools/jobs/170_loc_factors.sh
