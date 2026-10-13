#!/usr/bin/env bash
# Job 268: registered replication of job 267's door-deviation lead. Fixed k -0.10 vs TVLQR under amcl/rtab/truth,
# realistic tyre, mu 0.5, the 63 v_max 0.9 plans, FRESH seeds 3-5. 6 cells x 63 x 3 = 1134 runs, 12 workers, ~1.7 h.
D=${OUT_ROOT:-$HOME/run_data/2026-10-13_job268_door-deviation}
JOB_NAME=job268-test CONFIG_FILE=tools/jobs/c1/door_dev_v090.txt OUT_DIR=$D/test PLAN_FILE=tools/jobs/v090_test.txt \
  WORKERS="1 2 3 4 5 6 7 8 9 10 11 12" SEEDS="3 4 5" bash tools/jobs/170_loc_factors.sh
JOB_NAME=job268-dev CONFIG_FILE=tools/jobs/c1/door_dev_v090.txt OUT_DIR=$D/dev PLAN_FILE=tools/jobs/v090_dev.txt \
  WORKERS="1 2 3 4 5 6 7 8 9 10 11 12" SEEDS="3 4 5" bash tools/jobs/170_loc_factors.sh
