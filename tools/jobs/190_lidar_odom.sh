#!/usr/bin/env bash
# Job 190 (#34): C vs L vs LA (tools/jobs/lidar_odom/configs.txt), 40 broad
# plans x 2 seeds, workers 1-3, ~1h40. Read: miss + final_err sign test L vs C,
# and the believed-vs-true cross-track gain/lag (tools/loc_analysis/gain.py):
# L near k~0.95, tau~0.1 s (job 160 D) = skid observability was the gap.
set -uo pipefail
CONFIG_FILE=tools/jobs/lidar_odom/configs.txt OUT_DIR=${OUT_DIR:-$HOME/compare_lidar} \
    exec bash tools/jobs/170_loc_factors.sh
