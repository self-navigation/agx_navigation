#!/usr/bin/env bash
# Job 253: HELD-OUT half (v3f6_test), TVLQG vs TVLQR under amcl, rtab and truth, after the
# corrector pose-lag fix (job 252). 42 plans x seeds 0-2 x 6 cells = 756 runs, ~2 h.
# Config tools/jobs/c1/rtab_test.txt.
# UDP only, as job 251 (VM FastDDS SHM wedged 2026-10-11).
export FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/rtab_test.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-11_job253_rtab-test} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_test.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
