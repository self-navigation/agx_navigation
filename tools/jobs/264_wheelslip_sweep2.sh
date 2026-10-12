#!/usr/bin/env bash
# Job 264: tyre lateral slip compliance (WheelSlip) calibration sweep, extension (compliance 1.0, 2.0) on plain ground, TVLQR + truth, v3f6_dev seed 0.
# 2 cells x 41 plans = 82 runs, 12 workers.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/wheelslip_sweep2.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job264_wheelslip-sweep2} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_dev.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0"} \
bash tools/jobs/170_loc_factors.sh
