#!/usr/bin/env bash
# NOTE (2026-10-06): the default OUT_DIR below (~/compare_*) is historical; this
# run's data now lives in ~/run_data/2026-10-05_job190_lidar-odom/ (see docs/run-data-index.md).
# Commands left unchanged so the script stays as it was run.
# Job 190 (#34): C vs L vs LA (tools/jobs/lidar_odom/configs.txt), 40 broad
# plans x 2 seeds, workers 1-3, ~1h40. Read: miss + final_err sign test L vs C,
# and the believed-vs-true cross-track gain/lag (tools/believed_gain.py):
# L near k~0.95, tau~0.1 s (job 160 D) = skid observability was the gap.
set -uo pipefail
CONFIG_FILE=tools/jobs/lidar_odom/configs.txt OUT_DIR=${OUT_DIR:-$HOME/compare_lidar} \
    exec bash tools/jobs/170_loc_factors.sh
