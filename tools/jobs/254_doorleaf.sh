#!/usr/bin/env bash
# Job 254: door-leaf corridor test. 60 plans = ajar-door-leaf variants of the floor-6
# corridor-only plans, re-solved on per-plan map copies (run_data/2026-10-12_nojob_doorleaf-plans,
# tools/door_leaves.py). TVLQG vs TVLQR under amcl, rtab and truth; 6 cells x 60 x seeds 0-2 =
# 1080 runs, ~3 h on 12 workers. Config tools/jobs/c1/doorleaf.txt. Default DDS transport
# (shared memory works again after the 2026-10-12 reboot).
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/doorleaf.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job254_doorleaf} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/doorleaf60.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
