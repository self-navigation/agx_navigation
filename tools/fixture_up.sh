#!/usr/bin/env bash
# Bring the ROS 2 fixture stack up and PROVE it came up -- retrying from
# scratch if it did not.
#
# WHY. main.launch.py has start-up races we have not fixed: the map, the
# map->odom transform and the /goal_pose subscribers appear in a
# non-deterministic order, and a launch that loses one of them does not fail --
# it sits there with a robot that never moves. The historical workaround was
# "restart it and see", performed by a human. This performs it, gated on
# tools/stack_ready.py rather than on a guess about how long start-up takes.
#
#     tools/fixture_up.sh [--worker N] [--corrector tvlqr] [--patches true]
#                         [--localization truth] [--tries 3] [--timeout 120]
#
# Exit 0 means the stack is up AND every required readiness check passed, so a
# caller may go straight to publishing a goal. Exit 1 means every attempt
# failed; the last probe's report is on stdout and the launch log is named.
#
# Run this ON THE VM (it sources ROS itself). `just fixture-up` is the wrapper.
#
# Deliberately NOT idempotent-by-skipping: it always tears the partition down
# first. A half-up stack is the exact state this exists to escape, and
# "reuse whatever is already running" would inherit it -- along with any
# orphaned launch children, which stack a second planner trio on top of the
# first and plan from a stale odom belief (see tools/kill_stack.sh).
set -uo pipefail

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORKER=""
CORRECTOR=tvlqr
PATCHES=true
LOCALIZATION=truth
TRIES=3
TIMEOUT=120
SESSION=${TMUX_SESSION:-rl}

while [ $# -gt 0 ]; do
    case "$1" in
        --worker)       WORKER=$2; shift 2 ;;
        --corrector)    CORRECTOR=$2; shift 2 ;;
        --patches)      PATCHES=$2; shift 2 ;;
        --localization) LOCALIZATION=$2; shift 2 ;;
        --tries)        TRIES=$2; shift 2 ;;
        --timeout)      TIMEOUT=$2; shift 2 ;;
        *) echo "fixture_up.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

PARTITION=${WORKER:+agx$WORKER}
PARTITION=${PARTITION:-default}
WINDOW="fixture${WORKER}"
LOG="/tmp/fixture${WORKER}.log"

cd "$WORKSPACE" || exit 2

# ROS's setup scripts read AMENT_TRACE_SETUP_FILES while it is unset, so `set -u`
# across the source exits on that line. This has cost this project time twice;
# the guard is deliberate and belongs around every sourcing in the repo.
set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u

for attempt in $(seq 1 "$TRIES"); do
    echo "[fixture-up] attempt $attempt/$TRIES  (partition=$PARTITION corrector=$CORRECTOR patches=$PATCHES loc=$LOCALIZATION)"

    bash tools/kill_stack.sh "$WORKSPACE" kill "$PARTITION" >/dev/null 2>&1

    tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n scratch
    tmux kill-window -t "$SESSION:$WINDOW" 2>/dev/null
    tmux new-window -d -t "$SESSION" -n "$WINDOW" \
        "cd $WORKSPACE && DISPLAY=:0 vglrun -d egl0 make fixture WORKER=$WORKER \
         CORRECTOR=$CORRECTOR SURFACE_PATCHES=$PATCHES LOCALIZATION=$LOCALIZATION \
         HEADLESS=false USE_GPU_RENDER_ACCELERATION=false 2>&1 | tee $LOG"

    # The probe must run in the stack's own partition and domain, or it will
    # correctly report an empty graph and we would restart a healthy stack.
    if tools/with-worker "$WORKER" python3 tools/stack_ready.py --wait "$TIMEOUT" --settle 3; then
        echo "[fixture-up] READY on attempt $attempt (log: $LOG)"
        exit 0
    fi

    echo "[fixture-up] attempt $attempt did not come up; last 20 lines of $LOG:"
    tail -20 "$LOG" 2>/dev/null | sed 's/^/    /'
done

echo "[fixture-up] FAILED after $TRIES attempts -- see $LOG"
exit 1
