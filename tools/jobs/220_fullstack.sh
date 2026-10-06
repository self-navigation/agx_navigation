#!/usr/bin/env bash
# Job 220 part 1 (#10, #33): full-stack comparison, ours vs Nav2 Smac2D+MPPI,
# Hybrid-A*+MPPI and Smac2D+RPP; configs in tools/jobs/fullstack/configs.txt.
# Same harness as job 170. Run from the frozen checkout ~/agx_navigation_job220.
CONFIG_FILE=tools/jobs/fullstack/configs.txt \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-06_job220_fullstack} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8"} \
exec bash tools/jobs/170_loc_factors.sh
