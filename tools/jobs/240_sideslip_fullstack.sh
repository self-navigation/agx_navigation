#!/usr/bin/env bash
# Job 240 = track D in the full stack: ours-lib (truth, phantom walls) with and
# without TVLQRConfig.sideslip_k, layer off (N) and on (S). Configs:
# tools/jobs/c1/sideslip_fullstack.txt; K fills sideslip_k. Same harness as
# jobs 170/236/239. Run from a checkout built at or after the tvlqr_params commit.
K=${K:--0.33}
CFGS=$(mktemp /tmp/job240_cfg.XXXXXX)
sed "s|__K__|$K|g" tools/jobs/c1/sideslip_fullstack.txt >"$CFGS"
CONFIG_FILE=$CFGS \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-08_job240_sideslip-fullstack} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/broad34c.txt} \
WORKERS=${WORKERS:-"17 18"} \
SEEDS=${SEEDS:-0} \
bash tools/jobs/170_loc_factors.sh
rc=$?
rm -f "$CFGS"
exit $rc
