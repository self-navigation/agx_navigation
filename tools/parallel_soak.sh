#!/usr/bin/env bash
# Fan a soak out over several worker sims, one Gazebo partition each.
#
#     tools/parallel_soak.sh --out ~/x.jsonl --plans ~/plans.txt --repeats 5 \
#         --workers 3 --trace-dir ~/x_traces -- 0.276,2.618 0.6,2.618 1.5,2.618
#
# WHY THIS AND NOT A MULTI-LANE QUEUE. The job queue stays single-lane on
# purpose (one job never shares a world with another); parallelism belongs
# INSIDE a job. So a job that wants the box brings up its own workers, uses
# them, and takes them down again -- which is what this does.
#
# WHY PARALLEL MEASUREMENT IS SOUND HERE. The bridge pauses the world and
# multi-steps it, so a rollout's result does not depend on how fast the process
# gets scheduled (CLAUDE.md, "Measurement facts worth not rediscovering"). CPU
# contention costs wall time and nothing else. Verified 2026-08-15 the harder
# way too: a worker reproduced the default partition's `floor_6_v2_00004` to
# four decimals while another sim was running.
#
# SPLIT BY GAIN ARM, not by plan. Each worker gets whole arms and every plan, so
# a worker that dies costs one arm rather than a random slice of every arm --
# and the surviving arms are still complete, comparable measurements. It also
# keeps each worker's rows independent, so the concatenated output needs no
# repair.
#
# Every worker's sim is torn down on EXIT, including on error or Ctrl-C. A
# queued job that leaves sims running blocks everything behind it.
set -uo pipefail

OUT=""; PLANS=""; TRACE_DIR=""; REPEATS=3; WORKERS=3; TRACE_EVERY=1; SEED=0
while [ $# -gt 0 ]; do
    case "$1" in
        --out)         OUT=$2; shift 2 ;;
        --plans)       PLANS=$2; shift 2 ;;
        --trace-dir)   TRACE_DIR=$2; shift 2 ;;
        --trace-every) TRACE_EVERY=$2; shift 2 ;;
        --repeats)     REPEATS=$2; shift 2 ;;
        --workers)     WORKERS=$2; shift 2 ;;
        --seed)        SEED=$2; shift 2 ;;
        --) shift; break ;;
        *) echo "parallel_soak: unknown option '$1'" >&2; exit 2 ;;
    esac
done
ARMS=("$@")

[ -n "$OUT" ] && [ -n "$PLANS" ] && [ ${#ARMS[@]} -gt 0 ] || {
    echo "usage: parallel_soak.sh --out F --plans F [--repeats N] [--workers N]" >&2
    echo "                        [--trace-dir D] [--trace-every N] -- q,r [q,r ...]" >&2
    exit 2
}
[ "$WORKERS" -ge 1 ] && [ "$WORKERS" -le 9 ] || { echo "workers must be 1-9" >&2; exit 2; }
# More workers than arms would start a sim nothing drives.
[ "$WORKERS" -le "${#ARMS[@]}" ] || WORKERS=${#ARMS[@]}

N_PLANS=$(grep -c . "$PLANS")
LOGDIR=${LOGDIR:-/tmp/parallel_soak.$$}
mkdir -p "$LOGDIR"
[ -n "$TRACE_DIR" ] && mkdir -p "$TRACE_DIR"

echo "[fanout] $WORKERS workers, ${#ARMS[@]} arms, $N_PLANS plans, repeats=$REPEATS"
echo "[fanout] logs in $LOGDIR"

# --- teardown -------------------------------------------------------------
# Registered BEFORE the first sim starts, so a failure between "started" and
# "recorded which ones started" still cleans up.
STARTED=()
cleanup() {
    local w
    echo "[fanout] tearing down workers: ${STARTED[*]:-none}"
    for w in "${STARTED[@]:-}"; do
        [ -n "$w" ] || continue
        bash "$(dirname "${BASH_SOURCE[0]}")/kill_stack.sh" \
            "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" kill "agx${w}" >/dev/null 2>&1
    done
}
trap cleanup EXIT

# --- bring the worker sims up --------------------------------------------
# `ros2 launch` directly, NOT `make rl-sim`, for two reasons. `make rl-sim`
# depends on `build`, so N workers would run N concurrent colcon builds over one
# install tree -- and worse, they would rewrite install/ underneath any job
# already running from it. Building is `just remote-build`'s job (and the
# queue's runner sources the existing install tree); a job never builds.
if [ ! -d install ]; then
    echo "[fanout] no install/ -- run 'just remote-build' first" >&2; exit 1
fi

for w in $(seq 1 "$WORKERS"); do
    echo "[fanout] starting sim for worker $w"
    setsid nohup "$(dirname "${BASH_SOURCE[0]}")/with-worker" "$w" \
        ros2 launch agx_bringup rl_corrector_sim.launch.py \
        headless:=true sim_sensors:=false world:=rl_corrector.world \
        >"$LOGDIR/sim$w.log" 2>&1 </dev/null &
    STARTED+=("$w")
done

# Readiness: the pose topic is what GazeboBridge subscribes to for ground
# truth, so a partition advertising it has a world with a robot in it. The
# bridge does its own waiting after that; this loop only needs to fail loudly
# if a sim never came up at all, rather than leaving every rollout to time out.
for w in "${STARTED[@]}"; do
    for _ in $(seq 1 60); do
        if GZ_PARTITION="agx${w}" timeout 5 gz topic -l 2>/dev/null | grep -q "pose/info"; then
            echo "[fanout] worker $w ready"; break
        fi
        sleep 5
    done
    GZ_PARTITION="agx${w}" timeout 5 gz topic -l 2>/dev/null | grep -q "pose/info" || {
        echo "[fanout] worker $w NEVER CAME UP -- see $LOGDIR/sim$w.log"; exit 1; }
done

# --- deal the arms out, round-robin --------------------------------------
PIDS=(); WOF=()
for w in $(seq 1 "$WORKERS"); do
    args=(); n_arms=0
    for i in "${!ARMS[@]}"; do
        [ $(( i % WORKERS + 1 )) -eq "$w" ] || continue
        args+=(--gains "${ARMS[$i]}"); n_arms=$((n_arms + 1))
    done
    [ "$n_arms" -gt 0 ] || continue
    # soak cycles gains x plans internally, so max_rollouts = repeats * one cycle
    # gives exactly `repeats` complete cycles and no partial one.
    total=$(( REPEATS * n_arms * N_PLANS ))
    trace_args=()
    [ -n "$TRACE_DIR" ] && trace_args=(--trace-dir "$TRACE_DIR" --trace-every "$TRACE_EVERY")
    echo "[fanout] worker $w: $n_arms arms x $N_PLANS plans x $REPEATS = $total rollouts"
    # shellcheck disable=SC2086
    setsid nohup "$(dirname "${BASH_SOURCE[0]}")/with-worker" "$w" \
        python3 -m agx_planning.tuning.soak \
        --trajectories $(cat "$PLANS") \
        "${args[@]}" "${trace_args[@]}" \
        --seed "$SEED" --max-rollouts "$total" --out "${OUT}.w${w}" \
        >"$LOGDIR/soak$w.log" 2>&1 </dev/null &
    PIDS+=($!); WOF+=("$w")
done

# --- wait, then merge -----------------------------------------------------
rc=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[fanout] worker ${WOF[$i]} finished ok ($(grep -c . "${OUT}.w${WOF[$i]}" 2>/dev/null || echo 0) rows)"
    else
        echo "[fanout] worker ${WOF[$i]} FAILED -- see $LOGDIR/soak${WOF[$i]}.log"; rc=1
    fi
done

# Merge whatever exists. A failed worker still contributes the arms it did
# finish; `arm` is recorded on every row, so a short arm is visible in the
# analysis rather than silently averaged in.
: >"$OUT"
for w in "${WOF[@]}"; do
    [ -f "${OUT}.w${w}" ] && cat "${OUT}.w${w}" >>"$OUT"
done
echo "[fanout] merged $(grep -c . "$OUT") rows into $OUT (rc=$rc)"
exit $rc
