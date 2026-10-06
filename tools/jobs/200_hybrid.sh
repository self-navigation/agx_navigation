#!/usr/bin/env bash
# NOTE (2026-10-06): the default OUT_DIR below (~/compare_*) is historical; this
# run's data now lives in ~/run_data/2026-10-05_job200_hybrid/ (see docs/run-data-index.md).
# Commands left unchanged so the script stays as it was run.
# Job 200 (#33): Hybrid-A*+MPPI vs Smac2D+MPPI, 40 broad plans x 2 seeds,
# workers 4-7 (runs alongside job 190 on 1-3). Read: arrival, final_err,
# path length / travel time vs ours from v7 (and job 190's L if adopted).
set -uo pipefail
cd "$(dirname "$0")/../.."
CONFIG_FILE=tools/jobs/hybrid/configs.txt OUT_DIR=${OUT_DIR:-$HOME/compare_hybrid} \
    WORKERS=${WORKERS:-4 5 6 7} exec bash tools/jobs/170_loc_factors.sh
