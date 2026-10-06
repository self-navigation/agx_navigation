#!/usr/bin/env bash
# Job 220 part 2 (#10, #33 Type A): controllers only on the library plan,
# truth + amcl; configs in tools/jobs/fullstack/configs_part2.txt.
# Same harness as job 170. Run from the frozen checkout ~/agx_navigation_job221.
CONFIG_FILE=tools/jobs/fullstack/configs_part2.txt \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-06_job221_controllers} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
exec bash tools/jobs/170_loc_factors.sh
