#!/usr/bin/env bash
# Job 170 (#34): localization factor runs, same harness as job 160.
#
# Configs come from $CONFIG_FILE: lines "NAME  compare_run.py args...". Every
# run carries its own control arm (rule: a reference point is carried in the
# same process). Defaults to the 2026-10-03 short test:
#   C   amcl, phantom walls                     (= job 160 C, the control)
#   Cy  amcl, phantom walls, EKF without wheel yaw  (does chi-biased odom hurt amcl?)
#   Ay  amcl, solid walls,  EKF without wheel yaw   (same, with contact)
# Out: $OUT_DIR/<cfg>/rows.s<seed>.jsonl (merged at the end from <cfg>/parts/), tracks in
# $OUT_DIR/<cfg>/s<seed>/. Scheduling: one shared queue of (cfg, seed, plan) tasks, see below.
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

# One task per (cell, seed, plan) in ONE shared queue: a free worker takes the next
# line. Units used to be whole (cell, seed) plan lists dealt round-robin, which
# left workers idle whenever units < workers and made the job wait for the
# slowest unit (job 256: 6 units on 12 workers, 6 never used). compare_run
# already brings up a fresh stack per plan, so one plan per call costs only the
# interpreter start. Each task writes its own part file (concurrent appends to one
# rows file are not safe for multi-KB lines); rows.s<seed>.jsonl is merged at the
# end, so readers see the old layout. A killed job resumes: done parts are skipped.
QDIR=$(mktemp -d /tmp/factor_queue.XXXXXX)
TASKS=$QDIR/tasks
while read -r plan; do
    [ -z "$plan" ] && continue
    for s in "${SEEDS[@]}"; do for c in "${ORDER[@]}"; do echo "$c $s $plan"; done; done
done <"$PLANS" >"$TASKS"
echo 0 >"$QDIR/next"
N_TOTAL=$(grep -c . "$TASKS")
echo "[fac] starting at $(date -Is); $N_TOTAL tasks (${#ORDER[@]} cells x ${#SEEDS[@]} seeds x $(grep -c . "$PLANS") plans) on ${#WORKERS[@]} workers, shared queue; out=$OUT_DIR"
count_rows() { cat "$OUT_DIR"/*/parts/*.jsonl 2>/dev/null | grep -c .; }
progress() {
    local n; n=$(count_rows)
    local el=$(( $(date +%s) - T_START )) eta=""
    [ "$n" -gt 0 ] && [ "$n" -lt "$N_TOTAL" ] && \
        eta="; ETA $(TZ=Europe/Moscow date -d "@$(( $(date +%s) + el * (N_TOTAL - n) / n ))" '+%H:%M MSK')"
    echo "rows $n of $N_TOTAL ($(( 100 * n / (N_TOTAL > 0 ? N_TOTAL : 1) ))%)$eta; ${#WORKERS[@]} workers; $OUT_DIR"
}
T_START=$(date +%s)
ntfy_job_begin "$JOB_NAME" progress count_rows "$N_TOTAL"

next_task() {  # prints the next task line, or nothing when the queue is empty
    local n
    {
        flock 9
        n=$(( $(cat "$QDIR/next") + 1 ))
        echo "$n" >"$QDIR/next"
    } 9>"$QDIR/lock"
    sed -n "${n}p" "$TASKS"
}

run_worker() {  # $1 = worker; pulls tasks until the queue is empty
    local w=$1 c s plan name part rc
    while read -r c s plan < <(next_task) && [ -n "$c" ]; do
        name=$(basename "$plan" .npz)
        part="$OUT_DIR/$c/parts/s$s.$name.jsonl"
        mkdir -p "$OUT_DIR/$c/parts" "$OUT_DIR/$c/s$s"
        echo "$plan" >"$QDIR/plan.w$w"
        echo "[fac] $(date -Is) worker $w start $c seed $s $name"
        # shellcheck disable=SC2086
        tools/with-worker "$w" python3 tools/compare_run.py \
            --plans "$QDIR/plan.w$w" --arm ours --worker "$w" --seed "$s" ${CFG[$c]} \
            --out "$part" --log-dir "$OUT_DIR/$c/s$s" \
            >>"$OUT_DIR/w$w.log" 2>&1
        rc=$?
        echo "[fac] $(date -Is) worker $w done $c seed $s $name rc=$rc"
        [ "$rc" -ne 0 ] && echo "$c seed $s $name worker $w rc=$rc" >>"$OUT_DIR/failed_units.txt"
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
for w in "${WORKERS[@]}"; do
    run_worker "$w" &
    PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done

trap - EXIT  # workers already cleaned up; the trap is for aborts only
for c in "${ORDER[@]}"; do for s in "${SEEDS[@]}"; do
    cat "$OUT_DIR/$c/parts/s$s".*.jsonl >"$OUT_DIR/$c/rows.s$s.jsonl" 2>/dev/null
done; done
cat "$OUT_DIR"/*/rows.s*.jsonl >"$OUT_DIR/all_rows.jsonl" 2>/dev/null
echo "[fac] finished at $(date -Is); rows $(grep -c . "$OUT_DIR/all_rows.jsonl" 2>/dev/null || echo 0)"
ntfy_job_finished "$(progress; head -n 5 "$OUT_DIR/failed_units.txt" 2>/dev/null)" \
    "$(grep -c . "$OUT_DIR/failed_units.txt" 2>/dev/null || echo 0)"
rm -rf "$PLANS" "$QDIR"
