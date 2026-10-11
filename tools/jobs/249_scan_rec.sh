#!/usr/bin/env bash
# Job 249: record /scan (compare_run --record-scan) on the 10 plans where amcl
# most often goes >0.3 m off in job 248 (tools/jobs/amcl_worst10.txt), TVLQR and
# TVLQG under amcl Aogs, seeds 0-5 (12 units -> 12 workers). For scan-vs-map
# scoring at the truth pose vs amcl's pose (figures/2026-10-11/loc_error_along_drive).
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/scan_rec.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-11_job249_scan-rec} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/amcl_worst10.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2 3 4 5"} \
bash tools/jobs/170_loc_factors.sh
