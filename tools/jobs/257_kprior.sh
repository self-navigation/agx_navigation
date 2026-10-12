#!/usr/bin/env bash
# Job 257: sideslip prior / fixed-k arms under amcl, rtab, truth on job 255's plans, seeds and mu 0.5 patches.
# 42 plans x 3 seeds x 6 cells = 756 runs, shared queue on 12 workers. Pairs with 255 (Q0, T) and 256 (RQ) by (plan, seed).
CONFIG_FILE=${CONFIG_FILE:-tools/jobs/c1/kprior.txt} \
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-12_job257_kprior} \
PLAN_FILE=${PLAN_FILE:-tools/jobs/v3f6_test.txt} \
WORKERS=${WORKERS:-"1 2 3 4 5 6 7 8 9 10 11 12"} \
SEEDS=${SEEDS:-"0 1 2"} \
bash tools/jobs/170_loc_factors.sh
