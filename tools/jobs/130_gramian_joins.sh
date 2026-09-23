#!/bin/bash
# Re-run the controllability check against the fresh campaigns from jobs 110 and 120.
#
# WHY. Job 105 tests the theorem's prediction (Gramian energy predicts per-plan
# J) on the adopted corrector over the 40 broad plans. This adds two views:
#   - the seven shapes, adopted arm of job 110 (100 repeats, so per-plan means
#     are tight; only 7 points, so read it as a sanity check, not a test);
#   - the 40 broad plans, OPEN-LOOP arm of job 120: does controllability also
#     predict how badly the plan fails with no corrector at all? If it predicts
#     the corrector's J but not open loop's, that is the theorem's point: the
#     bound is about what feedback can do.
# No Gazebo; about a minute. See job 105 for the prediction and how to read it.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

SEVEN=$(python3 -c "
import yaml, os
c = yaml.safe_load(open('config/eval_trajectories.yaml'))
root = os.path.expanduser(c['trajectory_dir'])
print(' '.join(os.path.join(root, n + '.npz') for n in c['selected']))
")
BROAD=$(sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt)

echo "[gram2] starting at $(date -Is)"
python3 tools/controllability_gramian.py --plans $SEVEN \
    --soak "$HOME/soak_seven_three_arms.jsonl" --arm 2.5,2.618 \
    --out "$HOME/gramian_seven_q25.csv"
python3 tools/controllability_gramian.py --plans $BROAD \
    --soak "$HOME/soak_broad_open_loop.jsonl" --arm identity \
    --out "$HOME/gramian_broad40_identity.csv"
python3 tools/controllability_gramian.py --plans $BROAD \
    --soak "$HOME/soak_broad_open_loop.jsonl" --arm 2.5,2.618 \
    --out "$HOME/gramian_broad40_q25_job120.csv"
echo "[gram2] finished at $(date -Is)"
exit 0
