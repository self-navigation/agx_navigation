#!/bin/bash
# Open loop vs corrector on the 40 broad plans, in one campaign.
#
# WHY. Every "the corrector beats open loop" claim in the paper rests on the
# seven hand-picked, hard-enriched plans. The 40 broad plans (mechanically
# selected, the set CLAUDE.md says claims must be made on) have only ever been
# run with TVLQR arms, never open loop. So "does closing the loop help on plans
# we did not choose?" is unanswered. Found 2026-09-23 while porting the paper.
#
# Arms: identity (open loop), 10/0.25 (original gains), 2.5/2.618 (adopted),
# all in this one campaign. Mean-of-5, the validation standard (CLAUDE.md,
# "Reading a gain result"). Compare arms by paired sign test over the 40 plans
# on J (geometric mean), final_err and miss rate (>0.5 m). Note that J's
# control term is zero for identity BY DEFINITION, so on J alone open loop gets
# its control for free; read J's tracking + terminal parts, and arrival.
#
# HOW TO READ IT
#   - corrector beats identity on most plans: the paper can state the
#     corrector's value on an independent set, not only on the seven.
#   - it does not: the seven-shape result is a property of hard plans, and
#     the paper must say so plainly next to tab:seven.
#   - 2.5/2.618 here should reproduce job 100's row (J 13.84, final 0.244,
#     miss 11.5%) within between-campaign spread (~1 in J, see porting-notes
#     "Gain-number provenance"). Quote only in-campaign comparisons regardless.
#
# 3 arms x 40 plans x 5 = 600 rollouts, 3 workers, ~8 s each -> ~30 min.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

OUT=$HOME/run_data/2026-09-23_job120_broad40-open-loop/soak_broad_open_loop.jsonl
PLANS=$(mktemp /tmp/broad_ol_plans.XXXXXX)
sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt >"$PLANS"

echo "[bol] starting at $(date -Is); $(grep -c . "$PLANS") plans; out=$OUT"
tools/parallel_soak.sh \
    --out "$OUT" --plans "$PLANS" \
    --repeats 5 --workers 3 \
    -- identity 10,0.25 2.5,2.618
rc=$?
echo "[bol] finished at $(date -Is) rc=$rc; rows $(grep -c . "$OUT" 2>/dev/null || echo 0)"
rm -f "$PLANS"
exit 0
