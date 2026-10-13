#!/usr/bin/env bash
# Job 267: the 4-controller x 3-localizer grid of job 265 (realistic tyre, mu 0.5 patches) on the v_max 0.9 plans.
# Same seeds/layouts as job 265 on the same plans: v090_test (32) seeds 0-2 (test/), v090_dev (31) seeds 0-1 (dev/).
# 12 cells x (32x3 + 31x2) = 1896 runs, 12 workers, ~3 h (plans run 0.63x as long).
D=${OUT_ROOT:-$HOME/run_data/2026-10-13_job267_v090-grid}
JOB_NAME=job267-test CONFIG_FILE=tools/jobs/c1/tyre_grid_v090.txt OUT_DIR=$D/test PLAN_FILE=tools/jobs/v090_test.txt \
  WORKERS="1 2 3 4 5 6 7 8 9 10 11 12" SEEDS="0 1 2" bash tools/jobs/170_loc_factors.sh
JOB_NAME=job267-dev CONFIG_FILE=tools/jobs/c1/tyre_grid_v090.txt OUT_DIR=$D/dev PLAN_FILE=tools/jobs/v090_dev.txt \
  WORKERS="1 2 3 4 5 6 7 8 9 10 11 12" SEEDS="0 1" bash tools/jobs/170_loc_factors.sh
