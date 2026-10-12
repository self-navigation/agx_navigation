#!/usr/bin/env bash
# Job 260: too-big prior k -0.6 (TVLQG) vs fixed k -0.6 (TVLQR) under amcl and rtab, v3f6_test, seeds 0-2, mu 0.5.
# 504 runs. Pairs with 257 (PQ, PR, FQ, FR) and 258 (OT, OF). Workers 9-12 (job 259 has 1-8).
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/wrongprior_loc.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job260_wrongprior-loc} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_test.txt} \
WORKERS=${WORKERS:-"9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
