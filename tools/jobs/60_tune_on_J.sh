#!/bin/bash
# Re-tune (q_cross, r_omega) against J instead of metres.
#
# WHY. Every gain we hold was chosen against `max|e_cross|` on seven hand-picked
# plans. Two things now argue for redoing it in J:
#
#   * J is the quantity SVCM is stated in (docs/svcm-source.md), and the
#     2026-08-13 sweep found the tuned gains winning ALL SEVEN shapes in J where
#     they won only four in metres -- the metres losses were peak-deviation
#     trades paid back in accumulated error and terminal miss.
#   * the two metrics disagreed on 24 of 51 plans in the library sweep, so
#     "they mostly agree" is not available as a reason to skip this.
#
# The aggregator is left to default (GEOMETRIC for j_total). Do not override it
# without reading objective.py: J spans ~4 orders of magnitude across plans and
# floor_6_00031 alone is 48% of the arithmetic mean, so an arithmetic search on
# J tunes to whichever plan happens to be worst.
#
# The cache key records metric AND aggregator, so this cannot resume onto -- or
# be resumed by -- a max_cross run. That is deliberate; use a fresh cache path.
#
# ~105 s per evaluation at --repeats 3 (7 plans x 3), so a converged run is
# ~2-3 h. --max-evals 60 bounds it so the queue behind it is not blocked.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

CACHE=$HOME/tvlqr_tune_J.jsonl
OUT=$HOME/tvlqr_tuned_J.json

echo "[tuneJ] starting at $(date -Is); cache=$CACHE out=$OUT"

python3 -m agx_planning.tuning.tune_tvlqr \
    --trajectory-config "$PWD/config/eval_trajectories.yaml" \
    --metric j_total \
    --optimizer bayes \
    --repeats 3 \
    --max-evals 60 \
    --cache "$CACHE" \
    --out "$OUT"
rc=$?

echo "[tuneJ] finished at $(date -Is) rc=$rc"
if [ -f "$OUT" ]; then
    echo "[tuneJ] result:"
    cat "$OUT"
fi
# The comparison that decides whether this changes anything: the J-optimum's
# gains vs the adopted 0.276/2.618, both measured in J. `--max-evals 1` with an
# x0 IS a clean single-point measurement in the same code path (see CLAUDE.md).
echo "[tuneJ] measuring the ADOPTED point in J for comparison"
python3 -m agx_planning.tuning.tune_tvlqr \
    --trajectory-config "$PWD/config/eval_trajectories.yaml" \
    --metric j_total --max-evals 1 --repeats 5 \
    --q-cross 0.276 --r-omega 2.618 \
    --cache "$HOME/tvlqr_validate_J_adopted.jsonl" \
    --out "$HOME/tvlqr_validate_J_adopted.json"

echo "[tuneJ] done at $(date -Is)"
