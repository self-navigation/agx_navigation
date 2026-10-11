#!/usr/bin/env bash
# Job 252: job 251 again after the corrector pose-lag fix. Under localization:=rtab the
# corrector's single map->base_link lookup returned a pose 0.16 s old (icp_odometry
# stamps icp_odom->odom with the scan time); it now composes the latest map->odom with
# the latest odom->base_link. DEV half of plan set v3 floor 6, 41 plans x seeds 0-2 x
# 5 cells = 615 runs, ~1.7 h on 12 workers. Config tools/jobs/c1/rtab_lag.txt.
# UDP only, as job 251 (VM FastDDS SHM wedged 2026-10-11).
export FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/rtab_lag.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-11_job252_rtab-lagfix-dev} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_dev.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
