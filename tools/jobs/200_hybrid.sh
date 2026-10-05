#!/usr/bin/env bash
# Job 200 (#33): Hybrid-A*+MPPI vs Smac2D+MPPI, 40 broad plans x 2 seeds,
# workers 4-7 (runs alongside job 190 on 1-3). Read: arrival, final_err,
# path length / travel time vs ours from v7 (and job 190's L if adopted).
set -uo pipefail
cd "$(dirname "$0")/../.."
CONFIG_FILE=tools/jobs/hybrid/configs.txt OUT_DIR=${OUT_DIR:-$HOME/compare_hybrid} \
    WORKERS=${WORKERS:-4 5 6 7} exec bash tools/jobs/170_loc_factors.sh
