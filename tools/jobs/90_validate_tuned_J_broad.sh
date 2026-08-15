#!/bin/bash
# Validate whatever job 60 found, on the BROAD set, against known controls.
#
# WHY THIS JOB READS ITS OWN INPUT AT RUN TIME. Job 60 searches the SEVEN plans
# against J, and job 50 has just shown that J is FLAT across a 15x range of
# q_cross on 40 independent plans (p = 0.27 to 0.88 between every rung from
# 0.276 to 4). So a seven-plan optimum inside that range is an artifact of those
# seven, and adopting it directly is exactly the mistake the last three tuning
# results made. It has to be re-measured on ground it was not fitted to.
#
# We cannot know its answer in advance, so this job reads ~/tvlqr_tuned_J.json
# and validates whatever is in it. That is the only way to queue the check
# alongside the search rather than a session later.
#
# CONTROLS IN THE SAME PROCESS, not from an earlier run's numbers. The two
# comparison points are the currently adopted 0.276/2.618 and job 50's leading
# candidate 1.5/2.618. Measuring all three in one run removes the "do not
# compare a tuned number against a compare number" caveat that has been standing
# unresolved since 2026-08-02.
#
# mean-of-5 on 40 plans: 3 arms x 40 x 5 = 600 rollouts, ~25 min over 3 workers.
# Traced, so it is readable in J -- which is the currency job 60 optimised, and
# the only one in which its answer can be checked at all.
#
# If job 60's winner duplicates a control (well within the plateau), the job
# still runs: two arms is a valid, cheaper validation and the duplicate is
# visible in the output rather than silently collapsing the comparison.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

TUNED=$HOME/tvlqr_tuned_J.json
OUT=$HOME/soak_validate_J_broad.jsonl
TRACES=$HOME/validate_J_broad_traces
PLANS=$(mktemp /tmp/validate_J_plans.XXXXXX)
sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt >"$PLANS"

if [ ! -f "$TUNED" ]; then
    echo "[vJ] $TUNED MISSING -- job 60 did not finish or did not write it."
    echo "[vJ] Validating the two known points only; re-queue this job after"
    echo "[vJ] job 60 completes to get its winner measured."
    ARMS=(0.276,2.618 1.5,2.618)
else
    Q=$(python3 -c "import json;print(f\"{json.load(open('$TUNED'))['q_cross']:.4f}\")")
    R=$(python3 -c "import json;print(f\"{json.load(open('$TUNED'))['r_omega']:.4f}\")")
    echo "[vJ] job 60's winner: q_cross=$Q r_omega=$R"
    cat "$TUNED"
    ARMS=("$Q,$R" 0.276,2.618 1.5,2.618)
fi

echo "[vJ] starting at $(date -Is); arms: ${ARMS[*]}; $(grep -c . "$PLANS") plans"

tools/parallel_soak.sh \
    --out "$OUT" --plans "$PLANS" \
    --trace-dir "$TRACES" --trace-every 1 \
    --repeats 5 --workers 3 \
    -- "${ARMS[@]}"
rc=$?

echo "[vJ] finished at $(date -Is) rc=$rc; rows $(grep -c . "$OUT" 2>/dev/null || echo 0)"
rm -f "$PLANS"
exit 0
