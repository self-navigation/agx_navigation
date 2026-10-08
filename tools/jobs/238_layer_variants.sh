#!/usr/bin/env bash
# Job 238 = layer variants after C1/job 236: nav2 trigger ablation (N/S/Sn/Sx for
# MPPI, RPP, VPP under truth) and the ours-lib wall-aware re-join A/B (N/S/W,
# truth + amcl). Configs: tools/jobs/c1/layer_variants.txt; WFP/WM fill the
# W cells' barrier weight / margin. Same harness as jobs 170/221/230. Run from a
# frozen checkout.
WFP=${WFP:-1000} WM=${WM:-0.15}
CFGS=$(mktemp /tmp/job238_cfg.XXXXXX)
sed "s|__WFP__|$WFP|g; s|__WM__|$WM|g" tools/jobs/c1/layer_variants.txt >"$CFGS"
CONFIG_FILE=$CFGS \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-08_job238_layer-variants} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/broad34c.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"} \
SEEDS=${SEEDS:-0} \
bash tools/jobs/170_loc_factors.sh
rc=$?
rm -f "$CFGS"
exit $rc
