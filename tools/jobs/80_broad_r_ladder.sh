#!/bin/bash
# r_omega on the broad plan set, at a HIGHER q than every previous r ladder.
#
# WHY. Both r ladders (2026-08-14 high half, 2026-08-15 low half) held
# `q_cross` at 0.276 -- the value job 50 has just shown to be on the bad edge of
# the plateau. So everything we believe about r was measured at a q we are about
# to leave, on the seven plans rather than the broad set. Two specific claims
# are at risk:
#
#   * "r has a threshold in (0.5, 1.0) and is flat above it" -- the zigzag
#     halved between those two rungs, but that was one shape at q=0.276;
#   * "final_err is monotone in r" -- likewise.
#
# This re-measures r at q=1.5, the leading candidate from job 50, on the 40
# broad plans. It is worth running WHATEVER job 70 concludes: if 1.5 wins, this
# is the matching r evidence; if it does not, this is still the first r ladder
# at a second q, which is what would tell us whether q and r interact at all --
# every ladder so far has assumed they separate, and none has checked.
#
# 5 arms x 40 plans x 3 = 600 rollouts, ~20 min over 4 workers. mean-of-3 rather
# than 5 because this is a mapping run, not an adoption decision (job 50's rule:
# mean-of-3 to search, mean-of-5 to validate a winner).
#
# r=0.25 is included deliberately even though it is the old default's value: it
# is the only rung that tests whether the threshold is still there at q=1.5.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

OUT=$HOME/soak_broad_r.jsonl
TRACES=$HOME/broad_r_traces
PLANS=$(mktemp /tmp/broad_r_plans.XXXXXX)
sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt >"$PLANS"

echo "[br] starting at $(date -Is); $(grep -c . "$PLANS") plans; out=$OUT"

tools/parallel_soak.sh \
    --out "$OUT" --plans "$PLANS" \
    --trace-dir "$TRACES" --trace-every 1 \
    --repeats 3 --workers 4 \
    -- 1.5,0.25 1.5,0.5 1.5,1.0 1.5,2.618 1.5,5.0
rc=$?

echo "[br] finished at $(date -Is) rc=$rc; rows $(grep -c . "$OUT" 2>/dev/null || echo 0)"
rm -f "$PLANS"
exit 0
