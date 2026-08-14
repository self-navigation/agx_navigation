#!/bin/bash
# r_omega ladder: is r=2.618 mid-basin, or on a wall like q=0.276 turned out to be?
#
# `r` moved 0.25 -> 2.618 as half of a JOINT move, so we know the PAIR is better
# and nothing about r on its own. The q ladder (2026-08-13) found a near-vertical
# notch, which is precisely why this is worth asking about the other axis.
# q_cross is held at the adopted 0.276 so r's effect is separable, mirroring how
# the q ladder held r at 2.618.
#
# Traced every 5th rollout so the result is readable in J as well as in metres.
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

OUT=$HOME/soak_r_ladder.jsonl
TRACES=$HOME/r_ladder_traces

echo "[r_ladder] starting at $(date -Is); out=$OUT traces=$TRACES"

# FINITE, unlike `just soak`. A soak runs until someone stops it; a QUEUED job
# must end on its own or it blocks every job behind it forever -- which is the
# whole point of the queue. 5 gain points x 7 shapes = 35 cells; 6 batches of
# 175 is 1050 rollouts, n=30 per cell, ~1.75 h at ~6 s a rollout. That is the
# same n the q ladder resolved its notch at (n~58 over 18 cells), and enough for
# a mode FREQUENCY, which is what this measures.
#
# Restarting per batch (rather than one 1050-rollout process) bounds any
# within-process accumulation and labels rows with their own pid.
for batch in $(seq 1 6); do
    echo "[r_ladder] batch $batch/6 at $(date -Is)"
    python3 -m agx_planning.tuning.soak \
        --trajectory-config "$PWD/config/eval_trajectories.yaml" \
        --gains 0.276,1.0 \
        --gains 0.276,1.8 \
        --gains 0.276,2.618 \
        --gains 0.276,3.5 \
        --gains 0.276,5.0 \
        --trace-dir "$TRACES" --trace-every 5 \
        --max-rollouts 175 --out "$OUT" || break
    echo "[r_ladder] batch finished at $(date -Is); rows so far: $(wc -l <"$OUT")"
done

echo "[r_ladder] done at $(date -Is); total rows $(wc -l <"$OUT")"
