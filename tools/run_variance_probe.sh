#!/usr/bin/env bash
# Both arms of the variance probe, in order. Lives in a script rather than
# inline in the Justfile because it runs a shell loop inside a tmux window
# inside ssh, and three layers of quoting is how a loop silently becomes a
# single iteration.
#
# Arm 1 (within): one process, N rollouts   -> exposes accumulated drift.
# Arm 2 (across): N processes, 1 rollout    -> exposes per-process setup noise.
# Everything else is held fixed, so the two spreads are directly comparable.
# No `set -u`: ROS's setup.bash reads unset variables (AMENT_TRACE_SETUP_FILES)
# and aborts under it.

N="${1:-10}"
TRAJ="${2:-/home/programmer/pmp_trajectories_v2/floor_6_00042.npz}"
OUT="${3:-/home/programmer/variance_probe.jsonl}"

source /opt/ros/jazzy/setup.bash
source install/setup.bash
export PYTHONPATH="src/agx_navigation/agx_planning:${PYTHONPATH:-}"

# Fresh file: the arms are compared by their `mode` field, so records from an
# earlier probe with different gains would be pooled in silently.
rm -f "$OUT"

echo "=== arm 1: within-process, $N rollouts in one process ==="
python3 -m agx_planning.tuning.variance_probe --mode within \
    --repeats "$N" --trajectory "$TRAJ" --out "$OUT"

echo "=== arm 2: across-process, $N processes x 1 rollout ==="
for i in $(seq 1 "$N"); do
    echo "--- process $i/$N ---"
    python3 -m agx_planning.tuning.variance_probe --mode across \
        --repeats 1 --trajectory "$TRAJ" --out "$OUT"
done

echo VARIANCE_PROBE_DONE
