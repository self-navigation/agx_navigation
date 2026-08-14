#!/bin/bash
# Traces on BOTH SIDES of the U-turn's q_cross wall, for trace_diff.py.
#
# The sub-ladder (2026-08-13) put a near-vertical wall between q=0.4 (0.7% bad)
# and q=0.5 (99.9% bad), and all of the U-turn's deviation is produced at ONE
# corner in the last quarter of the run. The question trace_diff answers is the
# only one that separates the two candidate causes:
#
#   cmd* moves first, state identical  -> OUR CONTROLLER
#   state moves first under equal cmd  -> THE PLANT
#
# Deliberately tiny (~20 rollouts, ~2 min): this produces traces to READ next
# session, not a distribution. Every rollout is traced, so no subsampling.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

OUT=$HOME/uturn_edge.jsonl
TRACES=$HOME/uturn_edge_traces
# The plan dir is the eval config's, not a guess: it is ~/pmp_trajectories_v2
# on the VM, which is easy to confuse with the new ~/traj_data_v2 library.
PLAN=$(python3 -c "import yaml,os;c=yaml.safe_load(open('$PWD/config/eval_trajectories.yaml'));print(os.path.join(os.path.expanduser(c['trajectory_dir']),'floor_6_00031.npz'))")

if [[ ! -f $PLAN ]]; then
    echo "[edge] no $PLAN -- skipping"; exit 1
fi

echo "[edge] starting at $(date -Is); out=$OUT traces=$TRACES"
python3 -m agx_planning.tuning.soak \
    --trajectories "$PLAN" \
    --gains 0.4,2.618 \
    --gains 0.5,2.618 \
    --trace-dir "$TRACES" --trace-every 1 \
    --max-rollouts 20 --out "$OUT" || exit 1

echo "[edge] done at $(date -Is); rows $(wc -l <"$OUT"), traces $(ls "$TRACES" | wc -l)"
