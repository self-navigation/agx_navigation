#!/usr/bin/env bash
# Job 180 (#34): weekend amcl x EKF factorial (32 cells, ~18 h on 3 workers).
# Waits for job 170 to finish (one sim per partition), then runs the
# 170 harness over tools/jobs/amcl_sweep/configs.txt (codes explained there).
# Read: per-cell miss rate + believed/true |e_cross| ratio vs 'base' (control,
# same process); factor main effects by paired sign test over plan x seed.
set -uo pipefail
while pgrep -f "tools/jobs/170_loc_factors.sh" | grep -v $$ >/dev/null; do sleep 60; done
echo "[180] 170 done, starting $(date -Is)"
CONFIG_FILE=tools/jobs/amcl_sweep/configs.txt OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-03_job180_amcl-sweep/compare_amcl} \
    exec bash tools/jobs/170_loc_factors.sh
