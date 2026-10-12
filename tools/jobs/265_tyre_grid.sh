#!/usr/bin/env bash
# Job 265: the 4-controller x 3-localizer main grid on the realistic tyre (wheel_slip_lat 1.0), mu 0.5 patches.
# Same design as the pooled jobs 255-257 + 259: v3f6_test seeds 0-2 (test/), then v3f6_dev seeds 0-1 (dev/).
# 12 cells x (42x3 + 41x2) = 2496 runs, 12 workers, ~5.7 h.
D=${OUT_ROOT:-$HOME/run_data/2026-10-12_job265_tyre-grid}
JOB_NAME=job265-test CONFIG_FILE=tools/jobs/c1/tyre_grid.txt OUT_DIR=$D/test PLAN_FILE=tools/jobs/v3f6_test.txt \
  WORKERS="1 2 3 4 5 6 7 8 9 10 11 12" SEEDS="0 1 2" bash tools/jobs/170_loc_factors.sh
JOB_NAME=job265-dev CONFIG_FILE=tools/jobs/c1/tyre_grid.txt OUT_DIR=$D/dev PLAN_FILE=tools/jobs/v3f6_dev.txt \
  WORKERS="1 2 3 4 5 6 7 8 9 10 11 12" SEEDS="0 1" bash tools/jobs/170_loc_factors.sh
