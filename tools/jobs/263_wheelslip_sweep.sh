#!/usr/bin/env bash
# Job 263: tyre lateral slip compliance (WheelSlip) calibration sweep on plain ground, TVLQR + truth, v3f6_dev seed 0.
# 7 cells x 41 plans = 287 runs, 12 workers.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/wheelslip_sweep.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job263_wheelslip-sweep} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_dev.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0"} \
bash tools/jobs/170_loc_factors.sh
