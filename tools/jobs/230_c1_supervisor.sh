#!/usr/bin/env bash
# Job 230 = C1 (#47): every arm with vs without the supervisory layer (#40),
# truth + amcl, 40 broad plans, configs in tools/jobs/c1/configs.txt.
# Same harness as jobs 170/221. Run from a frozen checkout.
CONFIG_FILE=tools/jobs/c1/configs.txt \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-07_job230_c1-supervisor} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11"} \
SEEDS=${SEEDS:-0} \
exec bash tools/jobs/170_loc_factors.sh
