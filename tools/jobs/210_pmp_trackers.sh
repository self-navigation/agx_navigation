#!/usr/bin/env bash
# Job 210 (#10 Type A): Nav2 MPPI / RPP tracking OUR PMP plan (FollowPath)
# vs our corrector, same plan, amcl + phantom walls. Read: arrival / miss
# (>0.5 m) with timeouts separate, final_err, max_cross_track, paired per plan.
set -uo pipefail
cd "$(dirname "$0")/../.."
CONFIG_FILE=tools/jobs/pmp_trackers/configs.txt \
    OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-05_job210_pmp-trackers} \
    CONFIGS=${CONFIGS:-P PM PR} WORKERS=${WORKERS:-8 9} \
    exec bash tools/jobs/170_loc_factors.sh
