#!/usr/bin/env bash
# Job 262: tyre lateral creep (wheel slip2) x mu2 calibration sweep on plain ground, TVLQR + truth, v3f6_dev seed 0.
# 6 cells x 41 plans = 246 runs, 12 workers.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/slip_sweep.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job262_slip-sweep} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_dev.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0"} \
bash tools/jobs/170_loc_factors.sh
