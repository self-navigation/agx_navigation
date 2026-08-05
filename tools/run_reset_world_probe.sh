#!/usr/bin/env bash
# The bit-identical-reset experiment (handover 2026-08-03, "Do this first").
#
# Question: is the ~1e-9 rad/s residual wheel speed left by the settle loops the
# LAST remaining seed of the run-to-run spread on the hard shapes? A full gz
# world reset is the only mechanism that zeroes joint velocities, so it is the
# cheap way to falsify that before building the surgical `set_state` version.
#
# Two arms on the same trajectory, everything else fixed:
#   A (baseline)     -- today's reset
#   B (--reset-world) -- world reset first
#
# Traces are written for BOTH arms because the scalar score agreeing is NOT the
# claim being tested: `max|e_cross|` agreed to four decimals on floor_6_00042
# while the rollouts were not identical at all. trace_diff over arm B's traces
# is the actual test -- it must report NO first-differing step.
# No `set -u`: ROS's setup.bash reads unset variables and aborts under it.

N="${1:-5}"
TRAJ="${2:-/home/programmer/pmp_trajectories_v2/floor_6_00056.npz}"
OUT="${3:-/home/programmer/reset_world_probe.jsonl}"
TRACE="${4:-/home/programmer/reset_world_traces}"

source /opt/ros/jazzy/setup.bash
source install/setup.bash
export PYTHONPATH="src/agx_navigation/agx_planning:${PYTHONPATH:-}"

rm -f "$OUT"
rm -rf "$TRACE"; mkdir -p "$TRACE/base" "$TRACE/rw"

echo "=== arm A: baseline reset, $N rollouts ==="
python3 -m agx_planning.tuning.variance_probe --mode within \
    --repeats "$N" --trajectory "$TRAJ" --out "$OUT" --trace-dir "$TRACE/base"

echo "=== arm B: --reset-world, $N rollouts ==="
python3 -m agx_planning.tuning.variance_probe --mode within \
    --repeats "$N" --trajectory "$TRAJ" --out "$OUT" --reset-world \
    --trace-dir "$TRACE/rw"

echo "=== trace_diff within arm B (rollout 0 vs each later one) ==="
mapfile -t RW < <(ls -1 "$TRACE"/rw/*.csv | sort)
for f in "${RW[@]:1}"; do
    echo "--- $(basename "${RW[0]}") vs $(basename "$f") ---"
    python3 -m agx_planning.tuning.trace_diff --eps 0 "${RW[0]}" "$f"
done

echo "=== trace_diff within arm A, for comparison ==="
mapfile -t BA < <(ls -1 "$TRACE"/base/*.csv | sort)
for f in "${BA[@]:1}"; do
    echo "--- $(basename "${BA[0]}") vs $(basename "$f") ---"
    python3 -m agx_planning.tuning.trace_diff --eps 0 "${BA[0]}" "$f"
done

echo RESET_WORLD_PROBE_DONE
