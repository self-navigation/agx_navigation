#!/bin/bash
# The LOW half of the r_omega ladder -- the half that decides whether r matters.
#
# WHY, given the ladder already ran (2026-08-14, 1035 rollouts):
# it swept r in [1.0, 5.0] and found `r` FLAT on five of seven shapes (straight,
# corner, loop, zigzag, tight V all move by less than their own spread across a
# 5x range of r). The entire aggregate ranking came from two cells: the U-turn,
# which is 100% bad at every rung except r=2.618, and the S, which has a
# deterministic bad spike at r=3.5.
#
# But the ADOPTED move was r: 0.25 -> 2.618, and 0.25 is BELOW the ladder's
# floor. So we still do not know whether r's half of the joint move bought
# anything outside the U-turn -- exactly the question the ladder was meant to
# settle. This closes it from underneath. q_cross is held at the adopted 0.276
# throughout so r stays separable.
#
# Also the first traced soak since --trace-every was fixed (it aliased against
# the cycle length AND left tracing armed between traced rollouts, so the
# r-ladder's traces hold five cells' tracks per file and cannot be scored).
# Subsampling is by CYCLE now, so every cell gets equal J coverage.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

OUT=$HOME/soak_r_ladder_low.jsonl
TRACES=$HOME/r_ladder_low_traces

echo "[r_low] starting at $(date -Is); out=$OUT traces=$TRACES"

# 4 gain points x 7 shapes = 28 cells; 6 batches of 140 = 840 rollouts, n=30 per
# cell -- the same n the ladder resolved its notch at. FINITE, because a queued
# job that does not end blocks everything behind it. ~1.4 h at ~6 s a rollout.
# r=2.618 is carried as the overlap arm: it is measured in the same process as
# the low rungs, so the two ladders join without assuming cross-run comparability.
for batch in $(seq 1 6); do
    echo "[r_low] batch $batch/6 at $(date -Is)"
    python3 -m agx_planning.tuning.soak \
        --trajectory-config "$PWD/config/eval_trajectories.yaml" \
        --gains 0.276,0.25 \
        --gains 0.276,0.5 \
        --gains 0.276,1.0 \
        --gains 0.276,2.618 \
        --trace-dir "$TRACES" --trace-every 3 \
        --max-rollouts 140 --out "$OUT" || break
    echo "[r_low] batch finished at $(date -Is); rows so far: $(wc -l <"$OUT")"
done

echo "[r_low] done at $(date -Is); total rows $(wc -l <"$OUT")"
