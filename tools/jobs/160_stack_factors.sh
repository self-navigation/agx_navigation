#!/bin/bash
# Which stack-only factor makes our arm miss? (#34, overnight 2026-10-02)
#
# WHY. Job 150 showed the stack's live plans track on bare ground as well as
# the library plans (miss 6.5%), yet in compare v7 ours arrived 25/80. What
# the full stack adds: walls (contact), amcl's pose, ROS timing. This crosses
# them, ours only, 40 broad plans x seeds 0,1, per-tick TVLQR diagnostics and
# the live plan in every track file (compare_run.py, added 2026-10-02):
#
#   cfg    localization  walls     corrector
#   A      amcl          solid     tvlqr      = v7 replicate (+ freeze re-test, #32)
#   B      truth         solid     tvlqr
#   C      amcl          phantom   tvlqr
#   D      truth         phantom   tvlqr      ~ soak conditions through ROS
#   E      truth         phantom   identity   open loop in the stack
#
# phantom = walls visible to the lidar (so amcl still localizes on them) but
# without collision; the drive-through is recorded per row as wall_overlap_s /
# wall_first_t / wall_episodes / wall_min_clear (footprint vs baked map).
#
# HOW TO READ IT (paired sign tests over the 40, per seed and pooled)
#   A vs B   amcl's effect with walls;   C vs D   amcl's effect without contact
#   A vs C   contact's effect under amcl; B vs D   contact's effect under truth
#   D vs job 150's live-plan soak (6.5% miss)  -> ROS timing / node effect
#   D vs E   does the corrector help inside the stack at all
#
# Units (config, seed) are dealt round-robin to the workers in priority order,
# so every config's seed 0 lands before any seed 1. Resumable: compare_run
# skips rows already in a unit's JSONL, so re-running continues a killed night.
#
#   tools/agx-run --detach 'cd ~/agx_navigation && bash tools/jobs/160_stack_factors.sh'
set -uo pipefail

if [ -z "${ROS_VERSION:-}" ]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    source "$(pwd)/install/setup.bash"
    set -u
fi

OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-02_job160_stack-factors/compare_factors}
WORKERS=(${WORKERS:-1 2 3})
SEEDS=(${SEEDS:-0 1})
declare -A CFG=(
    [A]="--localization amcl"
    [B]="--localization truth"
    [C]="--localization amcl --phantom-walls"
    [D]="--localization truth --phantom-walls"
    [E]="--localization truth --phantom-walls --corrector identity"
)
ORDER=(${CONFIGS:-A B C D E})

mkdir -p "$OUT_DIR"
PLANS=$(mktemp /tmp/factor_plans.XXXXXX)
sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt | head -n "${N_PLANS:-1000}" >"$PLANS"

cleanup() {
    local w
    for w in "${WORKERS[@]}"; do
        bash tools/kill_stack.sh "$PWD" kill "agx$w" >/dev/null 2>&1
    done
}
trap cleanup EXIT

UNITS=()
for s in "${SEEDS[@]}"; do for c in "${ORDER[@]}"; do UNITS+=("$c:$s"); done; done
echo "[fac] starting at $(date -Is); ${#UNITS[@]} units x $(grep -c . "$PLANS") plans on ${#WORKERS[@]} workers; out=$OUT_DIR"

run_worker() {  # $1 = worker, rest = units
    local w=$1; shift
    local u c s
    for u in "$@"; do
        c=${u%%:*}; s=${u##*:}
        mkdir -p "$OUT_DIR/$c"
        echo "[fac] $(date -Is) worker $w start $c seed $s"
        # shellcheck disable=SC2086
        tools/with-worker "$w" python3 tools/compare_run.py \
            --plans "$PLANS" --arm ours --worker "$w" --seed "$s" ${CFG[$c]} \
            --out "$OUT_DIR/$c/rows.s$s.jsonl" --log-dir "$OUT_DIR/$c/s$s" \
            >>"$OUT_DIR/w$w.log" 2>&1
        echo "[fac] $(date -Is) worker $w done $c seed $s rc=$? rows $(grep -c . "$OUT_DIR/$c/rows.s$s.jsonl" 2>/dev/null)"
    done
}

PIDS=()
for i in "${!WORKERS[@]}"; do
    mine=()
    for j in "${!UNITS[@]}"; do
        [ $((j % ${#WORKERS[@]})) -eq "$i" ] && mine+=("${UNITS[$j]}")
    done
    echo "[fac] worker ${WORKERS[$i]} -> ${mine[*]}"
    run_worker "${WORKERS[$i]}" "${mine[@]}" &
    PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done

cat "$OUT_DIR"/*/rows.s*.jsonl >"$OUT_DIR/all_rows.jsonl" 2>/dev/null
echo "[fac] finished at $(date -Is); rows $(grep -c . "$OUT_DIR/all_rows.jsonl" 2>/dev/null || echo 0)"
rm -f "$PLANS"
