#!/usr/bin/env bash
# Job 170 (#34): localization factor runs, same harness as job 160.
#
# Configs come from $CONFIG_FILE: lines "NAME  compare_run.py args...". Every
# run carries its own control arm (rule: a reference point is carried in the
# same process). Defaults to the 2026-10-03 short test:
#   C   amcl, phantom walls                     (= job 160 C, the control)
#   Cy  amcl, phantom walls, EKF without wheel yaw  (does chi-biased odom hurt amcl?)
#   Ay  amcl, solid walls,  EKF without wheel yaw   (same, with contact)
# Out: $OUT_DIR/<cfg>/rows.s<seed>.jsonl, tracks in $OUT_DIR/<cfg>/s<seed>/.
set -uo pipefail

if [ -z "${ROS_VERSION:-}" ]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    source "$(pwd)/install/setup.bash"
    set -u
fi

OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-03_job170_loc-factors/compare_loc}
WORKERS=(${WORKERS:-1 2 3})
SEEDS=(${SEEDS:-0 1})
declare -A CFG=()
ORDER_DEFAULT=()
CONFIG_FILE=${CONFIG_FILE:-}
if [ -n "$CONFIG_FILE" ]; then
    while read -r name args; do
        [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
        CFG[$name]="$args"; ORDER_DEFAULT+=("$name")
    done < <(sed "s|__REPO__|$PWD|g" "$CONFIG_FILE")
else
    CFG[C]="--localization amcl --phantom-walls"
    CFG[Cy]="--localization amcl --phantom-walls --no-ekf-wheel-yaw"
    CFG[Ay]="--localization amcl --no-ekf-wheel-yaw"
    ORDER_DEFAULT=(C Cy Ay)
fi
ORDER=(${CONFIGS:-${ORDER_DEFAULT[*]}})

mkdir -p "$OUT_DIR"
PLANS=$(mktemp /tmp/factor_plans.XXXXXX)
sed "s|__HOME__|$HOME|" "${PLAN_FILE:-tools/jobs/broad40.txt}" | head -n "${N_PLANS:-1000}" >"$PLANS"

source tools/jobs/lib_ntfy.sh
JOB_NAME=${JOB_NAME:-$(basename "$OUT_DIR")}
rm -f "$OUT_DIR/failed_units.txt"

cleanup() {
    local rc=$? w
    ntfy_job_exit "$rc"
    for w in "${WORKERS[@]}"; do
        bash tools/kill_stack.sh "$PWD" kill "agx$w" >/dev/null 2>&1
    done
}
trap cleanup EXIT

UNITS=()
for s in "${SEEDS[@]}"; do for c in "${ORDER[@]}"; do UNITS+=("$c:$s"); done; done
echo "[fac] starting at $(date -Is); ${#UNITS[@]} units x $(grep -c . "$PLANS") plans on ${#WORKERS[@]} workers; out=$OUT_DIR"
N_TOTAL=$(( ${#UNITS[@]} * $(grep -c . "$PLANS") ))
count_rows() { cat "$OUT_DIR"/*/rows.s*.jsonl 2>/dev/null | grep -c .; }
progress() {
    local n; n=$(count_rows)
    local el=$(( $(date +%s) - T_START )) eta=""
    [ "$n" -gt 0 ] && [ "$n" -lt "$N_TOTAL" ] && \
        eta="; ETA $(date -d "@$(( $(date +%s) + el * (N_TOTAL - n) / n ))" '+%H:%M %Z')"
    echo "rows $n of $N_TOTAL ($(( 100 * n / (N_TOTAL > 0 ? N_TOTAL : 1) ))%)$eta; ${#WORKERS[@]} workers; $OUT_DIR"
}
T_START=$(date +%s)
ntfy_job_begin "$JOB_NAME" progress count_rows "$N_TOTAL"

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
        local rc=$?
        echo "[fac] $(date -Is) worker $w done $c seed $s rc=$rc rows $(grep -c . "$OUT_DIR/$c/rows.s$s.jsonl" 2>/dev/null)"
        [ "$rc" -ne 0 ] && echo "$c seed $s worker $w rc=$rc" >>"$OUT_DIR/failed_units.txt"
    done
    # Tear down our own last stack HERE, before "finished at": a job launched
    # on these workers after that line must find them idle. (The EXIT-trap
    # sweep used to do this after "finished", one worker at a time, ~2-10 min
    # each under load -- and kill_stack agxN also matches worker N's harness,
    # so it killed the NEXT job's compare_run worker by worker: job 236 lost
    # 10/24 units to job 235's sweep, 2026-10-08.)
    bash tools/kill_stack.sh "$PWD" kill "agx$w" >/dev/null 2>&1
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

trap - EXIT  # workers already cleaned up; the trap is for aborts only
cat "$OUT_DIR"/*/rows.s*.jsonl >"$OUT_DIR/all_rows.jsonl" 2>/dev/null
echo "[fac] finished at $(date -Is); rows $(grep -c . "$OUT_DIR/all_rows.jsonl" 2>/dev/null || echo 0)"
ntfy_job_finished "$(progress; head -n 5 "$OUT_DIR/failed_units.txt" 2>/dev/null)" \
    "$(grep -c . "$OUT_DIR/failed_units.txt" 2>/dev/null || echo 0)"
rm -f "$PLANS"
