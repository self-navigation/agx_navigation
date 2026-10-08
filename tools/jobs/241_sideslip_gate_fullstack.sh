#!/usr/bin/env bash
# Job 241 = the speed-gated sideslip TVLQR in the full stack, against plain and
# un-gated TVLQR, layer off (N) and on (S), 2 seeds. Configs:
# tools/jobs/c1/sideslip_gate_fullstack.txt; K fills sideslip_k, VG sideslip_v_gate.
# Same harness as jobs 170/236/239/240. Run from a checkout built at or after 4395ad8.
K=${K:--0.33}
VG=${VG:-0.3}
CFGS=$(mktemp /tmp/job241_cfg.XXXXXX)
sed -e "s|__K__|$K|g" -e "s|__VG__|$VG|g" tools/jobs/c1/sideslip_gate_fullstack.txt >"$CFGS"
CONFIG_FILE=$CFGS \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-08_job241_sideslip-gate-fullstack} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/broad34c.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10"} \
SEEDS=${SEEDS:-"0 1"} \
bash tools/jobs/170_loc_factors.sh
rc=$?
rm -f "$CFGS"
exit $rc
