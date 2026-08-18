#!/bin/bash
# The LAST gain job: settle r_omega at the winning q, and settle the pair.
#
# WHY. Job 70 (1200 rollouts, mean-of-5, broad 40) put the best q at 2.5, not
# the 1.5 job 50 predicted: J is flat across the whole plateau (13.1-15.9, no
# rung beats the adopted 0.276 at p<0.08) and the separation is entirely on
# arrival -- q=2.5 beats 0.276 on final_err 34/40 (p<0.0001) and max_cross
# 32/40 (p=0.0002), and beats q=1.5 on final_err 27/40 (p=0.038).
#
# But job 80 showed q and r INTERACT, which every earlier ladder assumed away.
# At q=1.5 the old r story inverts: r=0.25 -- the value we moved off -- is best
# on max_cross (0.538) and final_err, losing only on J (11/40, p=0.006), and the
# r=0.5->1.0 threshold that justified the move is absent. Every r claim in
# CLAUDE.md is therefore scoped to q=0.276, and NOTHING is known about r at the
# q we are about to adopt.
#
# So: r at q=2.5, with both previously-considered points carried IN PROCESS as
# controls, so the adoption decision is made from a single run and needs no
# cross-run comparison (the standing caveat job 90 removed for the tuner).
#
# 6 arms x 40 plans x 5 = 1200 rollouts, ~40 min over 4 workers. mean-of-5
# because this IS the adoption decision (mean-of-3 to search, mean-of-5 to
# validate a winner).
#
# Read it on final_err and J. NOT max|e_cross|: job 50 showed that metric ranks
# the old default best while it spends ~3x the control, which is the SVCM
# prescription rather than a better controller.
#
# No traces: j_total is accumulated online by EpsilonAccumulator and lands in
# every row, so there is nothing left for offline scoring to recover.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

OUT=$HOME/soak_broad_r_at_q25.jsonl
PLANS=$(mktemp /tmp/broad_r_q25_plans.XXXXXX)
sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt >"$PLANS"

echo "[rq] starting at $(date -Is); $(grep -c . "$PLANS") plans; out=$OUT"

tools/parallel_soak.sh \
    --out "$OUT" --plans "$PLANS" \
    --repeats 5 --workers 4 \
    -- 2.5,0.25 2.5,1.0 2.5,2.618 2.5,5.0 1.5,2.618 0.276,2.618
rc=$?

echo "[rq] finished at $(date -Is) rc=$rc; rows $(grep -c . "$OUT" 2>/dev/null || echo 0)"
rm -f "$PLANS"
exit 0
