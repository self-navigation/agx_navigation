#!/bin/bash
# Build a second trajectory library by CONSTRUCTION, so per-shape claims stop
# resting on one plan each.
#
# Every shape claim this project has rests on exactly ONE trajectory: the U-turn
# notch is 5906 rollouts of floor_6_00031 and nothing else, so we cannot tell a
# property of U-turns from a property of that U-turn. The existing library
# cannot fix that -- it came from uniform random pairs with a distance filter,
# which a long straight corridor passes perfectly, so ~64 of its 100 plans are
# straight lines.
#
# Two stages, deliberately split (see tools/sample_eval_trajectories.py):
#   1. SCREEN -- cheap. Rank pairs on blocked line of sight, detour and the
#      in-place rotation their headings force. NOT on predicted turning: that
#      correlates with the real planner at only +0.30, measured 2026-08-14.
#   2. SOLVE  -- expensive. Run the real PMP solver, and label the shape from
#      the SOLVED plan, where the descriptors are real.
#
# Needs no Gazebo, but it is queued anyway rather than run alongside: it is
# CPU-heavy, and the queue is what keeps one thing at a time on this box.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

MAPS=src/rudn-ordjo-building/maps
CAND=$HOME/candidates_v2.json
OUT=$HOME/traj_data_v2

echo "[gen] screening at $(date -Is)"
python3 tools/sample_eval_trajectories.py \
    --map "$MAPS/floor_1.yaml" \
    --map "$MAPS/floor_6.yaml" \
    --count 1200 \
    --keep 500 \
    --min-range 6.0 --max-range 25.0 \
    --seed 20260814 \
    --out "$CAND" || exit 1

echo "[gen] solving at $(date -Is)"
# Solving is where the time goes and where failures live (a BVP that does not
# converge is normal, not an error) -- so this is bounded by the candidate list
# rather than by a wall clock, and partial output is useful.
python3 tools/sample_eval_trajectories.py \
    --solve "$CAND" --out-dir "$OUT" || exit 1

echo "[gen] done at $(date -Is); $(ls "$OUT" | wc -l) plans in $OUT"
