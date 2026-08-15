#!/bin/bash
# Resolve q_cross on the BROAD plan set, at mean-of-5, fanned out over workers.
#
# WHY. Job 50 (40 plans, 6 arms, mean-of-3) settled two things and opened one:
#
#   * the move off the old default q=10/r=0.25 is confirmed on plans we did not
#     choose -- 1.49x worse in J, 5/40 wins, sign test p<0.001;
#   * the ADOPTED VALUE q=0.276 does not survive. It is the worst of the six
#     arms in metres (beaten by every other at p<=0.038, INCLUDING the old
#     default) and has the worst miss rate, 20.0%; in J it is indistinguishable
#     from everything out to q=4. It sits on the bad edge of a wide plateau.
#   * open: where in that plateau to sit. q=1.5 dominated the adopted point on
#     every axis that separated them, but at n=3 per cell and only three rungs
#     inside the plateau, the minimum is not located.
#
# So: six rungs spanning the plateau, r held at 2.618 so q is separable (the
# mirror of every earlier ladder), mean-of-5 rather than 3, on the same 40 plans
# so this is directly comparable with job 50 rather than a new baseline.
#
# 6 arms x 40 plans x 5 = 1200 rollouts. Serially that is ~2.7 h; over 4 workers
# it is ~40 min. This is the first job to use tools/parallel_soak.sh -- read its
# header for why parallel measurement is sound (the world is paused and
# multi-stepped, so CPU contention costs wall time and nothing else).
#
# READ IT ON final_err AND J, NOT on max|e_cross|. Job 50 established that
# max|e_cross| ranks the OLD DEFAULT best while it spends ~3x the control effort
# to get there -- the two metrics are measuring the two sides of one trade, and
# the peak-deviation side is the one that disagrees with "did it arrive".
#
# Everything traced, so it is scoreable in J:
#   .venv/bin/python tools/score_sweep.py --from-jsonl soak_data/soak_broad_q.jsonl \
#       --trace-root broad_q_traces --plans traj_data_v2
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

OUT=$HOME/soak_broad_q.jsonl
TRACES=$HOME/broad_q_traces
PLANS=$(mktemp /tmp/broad_q_plans.XXXXXX)
sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt >"$PLANS"

echo "[bq] starting at $(date -Is); $(grep -c . "$PLANS") plans; out=$OUT"

tools/parallel_soak.sh \
    --out "$OUT" --plans "$PLANS" \
    --trace-dir "$TRACES" --trace-every 1 \
    --repeats 5 --workers 4 \
    -- 0.276,2.618 0.6,2.618 1.0,2.618 1.5,2.618 2.5,2.618 4.0,2.618
rc=$?

echo "[bq] finished at $(date -Is) rc=$rc; rows $(grep -c . "$OUT" 2>/dev/null || echo 0)"
rm -f "$PLANS"
exit 0   # never fail the queue: a partial result is still worth reading
