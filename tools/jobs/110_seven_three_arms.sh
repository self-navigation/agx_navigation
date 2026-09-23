#!/bin/bash
# The seven-shape table, re-measured in ONE campaign: open loop, old gains, adopted gains.
#
# WHY. The paper's headline (tab:seven, "2.127 -> 1.127 m") is the 2026-08-07
# baseline: open loop vs TVLQR at the ORIGINAL gains 10/0.25, five repeats. The
# adopted gains 2.5/2.618 have never been measured on these seven plans, and the
# abstract reads as if they had. Found 2026-09-23 while porting the paper
# (../paper/porting-notes.md, re-injection notes). This job produces the table
# the paper should actually print, with all three arms in one process group, so
# no row is borrowed from another campaign.
#
# 100 repeats, not 5: three of the seven shapes are bimodal (zigzag, tight V,
# U-turn), and for those the right number is a MODE FREQUENCY over ~100 samples,
# not a mean over 5 (CLAUDE.md, "Metrics and estimators"). Report per shape:
# mean max|e_cross|, mean final_err, geometric-mean J, and the bad-mode rate
# (zigzag > 1.5 m, U-turn > 2.0 m, tight V > 0.6 m, as in the 2026-08-13 ladder).
#
# HOW TO READ IT
#   - identity arm should land near the 2026-08-07 open-loop column (2.127 mean).
#     If it does not, the plant has drifted since August: stop and find out why
#     before quoting anything from this job.
#   - 10/0.25 should land near 1.127 for the same reason.
#   - 2.5/2.618 is the new number for the abstract. Expect the zigzag bad-mode
#     rate to be low, since q=2.5 sits between 1.5 (0-2%) and 10 (~90%) in the
#     ladder, BUT 2.5 itself was never measured there. If it is high, the
#     cliff lies below 2.5 and the adopted point is on the wrong side of it on this
#     shape: report that, do not re-tune (tuning is closed).
#   - U-turn: the notch result says only q=0.276 and 10/0.25 were good. Expect
#     2.5 to be bad on it; that is a property of that one plan (Settled).
#
# 3 arms x 7 plans x 100 = 2100 rollouts, 3 workers (one arm each),
# ~8 s/rollout (job 100) -> ~95 min.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

OUT=$HOME/soak_seven_three_arms.jsonl
PLANS=$(mktemp /tmp/seven_plans.XXXXXX)
python3 -c "
import yaml, os
c = yaml.safe_load(open('config/eval_trajectories.yaml'))
root = os.path.expanduser(c['trajectory_dir'])
print('\n'.join(os.path.join(root, n + '.npz') for n in c['selected']))
" >"$PLANS" || exit 1

echo "[seven] starting at $(date -Is); $(grep -c . "$PLANS") plans; out=$OUT"
tools/parallel_soak.sh \
    --out "$OUT" --plans "$PLANS" \
    --repeats 100 --workers 3 \
    -- identity 10,0.25 2.5,2.618
rc=$?
echo "[seven] finished at $(date -Is) rc=$rc; rows $(grep -c . "$OUT" 2>/dev/null || echo 0)"
rm -f "$PLANS"
exit 0
