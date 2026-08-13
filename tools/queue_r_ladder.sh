#!/bin/bash
# Queue an r_omega ladder behind whatever is currently driving the sim.
#
# WHY THIS RUNS ON THE VM AND NOT FROM THE LAPTOP
# -----------------------------------------------
# The laptop closes, the VPN drops, and an ssh-driven loop dies with it. So the
# chaining lives HERE: this script waits for the in-flight sweep to exit, then
# starts the soak itself. Launch it detached (tools/agx-run --detach) and it
# survives the session that launched it -- which is the whole point.
#
# WHAT IT MEASURES
# ----------------
# The 2026-08-13 q_cross ladder found a NOTCH: near-vertical walls at 0.2 and
# 0.5 around a basin at 0.276-0.4. `r_omega` has never had the same treatment.
# It moved 0.25 -> 2.618 as half of a JOINT move, so we know the pair is better
# but not whether 2.618 is in the middle of its own basin or on an edge like
# q=0.276 turned out to be. Same question, same instrument, one variable.
#
# q_cross is held at the ADOPTED 0.276 so r's effect is separable, exactly as
# the q ladder held r at 2.618.
#
# Traced (every 5th rollout, ~130 kB each) so the result is readable in J, which
# is the currency the SVCM framework is stated in. Score it with:
#   .venv/bin/python tools/score_sweep.py --from-jsonl soak_data/r_ladder.jsonl \
#       --trace-root r_traces --plans traj_data
set -u

REMOTE=/home/programmer/agx_navigation
OUT=/home/programmer/soak_r_ladder.jsonl
TRACES=/home/programmer/r_ladder_traces
LOG=/tmp/r_ladder.log

{
    echo "[queue] waiting for the in-flight sweep to finish..."
    # Wait on the SCRIPT, not on a single rollout: sweep2.sh spawns a fresh
    # variance_probe per trajectory, so watching for "no python running" would
    # fire in the gap between two of them and put two drivers on one sim.
    while pgrep -u "$(id -u)" -f 'bash /home/programmer/sweep2.sh' >/dev/null; do
        sleep 30
    done
    echo "[queue] sweep2 done at $(date -Is); starting the r_omega ladder"

    cd "$REMOTE" || exit 1
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    export PYTHONPATH=src/agx_navigation/agx_planning:$PYTHONPATH

    # Self-restarting, like `just soak`: each batch is a fresh process, which
    # bounds any within-process accumulation and labels rows with their pid.
    while true; do
        python3 -m agx_planning.tuning.soak \
            --trajectory-config "$REMOTE/config/eval_trajectories.yaml" \
            --gains 0.276,1.0 \
            --gains 0.276,1.8 \
            --gains 0.276,2.618 \
            --gains 0.276,3.5 \
            --gains 0.276,5.0 \
            --trace-dir "$TRACES" --trace-every 5 \
            --max-rollouts 175 --out "$OUT" || break
    done
    echo "[queue] ladder exited at $(date -Is)"
} >>"$LOG" 2>&1
