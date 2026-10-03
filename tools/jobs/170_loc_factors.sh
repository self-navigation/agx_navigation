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

OUT_DIR=${OUT_DIR:-$HOME/compare_loc}
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
