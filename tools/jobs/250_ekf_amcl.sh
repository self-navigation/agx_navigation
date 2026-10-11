#!/usr/bin/env bash
# Job 250: EKF skid covariance + IMU accel, and amcl alpha3, under TVLQG + Aogs on the DEV half
# of plan set v3 floor 6 (41 plans x seeds 0-2 x 8 cells = 984 runs, ~2.5 h on 12
# workers). Config tools/jobs/c1/ekf_amcl.txt. Same harness as job 247; the
# held-out half (tools/jobs/v3f6_test.txt) stays untouched for the final run.
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/ekf_amcl.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-11_job250_ekf-amcl} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_dev.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
