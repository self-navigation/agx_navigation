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
#                         [--nav-mode vec-pmp|nav2] [--nav2-controller mppi|dwb|rpp]
#                         [--nav2-profile compare_static]
#                         [--floor N] [--frontier true|false] [--spawn X Y YAW]
#
# Exit 0 means the stack is up AND every required readiness check passed, so a
# caller may go straight to publishing a goal. Exit 1 means every attempt
# failed; the last probe's report is on stdout and the launch log is named.
#
# Run this ON THE VM (it sources ROS itself). `just fixture-up` is the wrapper.
#
# The nav2 mode (--nav-mode nav2) runs the SAME fixture (static baked map, amcl
# localization, same spawn pose) with NAV_MODE=nav2 instead of vec-pmp, so the
# comparison arms differ in the nav layer and nothing else; readiness is then
# gated on stack_ready.py --mode nav2 (lifecycle nodes ACTIVE, the
# navigate_to_pose action server, the /cmd_vel chain), which is a different
# definition of "up" than the planner trio's.
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
# Headless unless someone is watching over Moonlight: a GUI client burns ~100%
# CPU in software rendering per sim and was the prime suspect in the v3 rtf
# collapses (2026-09-30). FIXTURE_GUI=true to watch.
HEADLESS_=$([ "${FIXTURE_GUI:-false}" = true ] && echo false || echo true)
SESSION=${TMUX_SESSION:-rl}
NAV_MODE=vec-pmp
NAV2_CONTROLLER=mppi
NAV2_PROFILE=""
FLOOR_NUMBER=""
FRONTIER=false
SPAWN_X=0.0
SPAWN_Y=0.0
SPAWN_YAW=0.0

while [ $# -gt 0 ]; do
    case "$1" in
        --worker)         WORKER=$2; shift 2 ;;
        --corrector)      CORRECTOR=$2; shift 2 ;;
        --patches)        PATCHES=$2; shift 2 ;;
        --localization)   LOCALIZATION=$2; shift 2 ;;
        --tries)          TRIES=$2; shift 2 ;;
        --timeout)        TIMEOUT=$2; shift 2 ;;
        --nav-mode)       NAV_MODE=$2; shift 2 ;;
        --nav2-controller) NAV2_CONTROLLER=$2; shift 2 ;;
        --nav2-profile)   NAV2_PROFILE=$2; shift 2 ;;
        --floor)          FLOOR_NUMBER=$2; shift 2 ;;
        --frontier)       FRONTIER=$2; shift 2 ;;
        --spawn)          SPAWN_X=$2; SPAWN_Y=$3; SPAWN_YAW=$4; shift 4 ;;
        --wheel-bias)     WHEEL_BIAS=$2; shift 2 ;;
        *) echo "fixture_up.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

PARTITION=${WORKER:+agx$WORKER}
PARTITION=${PARTITION:-default}
WINDOW="fixture${WORKER}"
LOG="/tmp/fixture${WORKER}.log"

# The launch args that PARAM_VARS does not carry. All consumed by the launch
# files from the command line: spawn_* by sim_control (spawner) AND
# static_map (amcl initial_pose, which must match -- see its docstring);
# nav2_controller by nav2.launch.py; frontier by nav.launch.py.
EXTRA="spawn_x:=$SPAWN_X spawn_y:=$SPAWN_Y spawn_yaw:=$SPAWN_YAW frontier:=$FRONTIER"
# Sim-only actuator fault, vec-pmp only (#27): 'fl,rl,fr,rr' command scale.
[ -n "${WHEEL_BIAS:-}" ] && EXTRA="$EXTRA wheel_bias:=$WHEEL_BIAS"
# Patches: false is passed EXPLICITLY (the comparison spawns its own along-path
# patches via tools/spawn_patches.py, so the fixture's near-origin patches must
# be OFF to keep one plant per run); true is the launch default and is simply
# not passed, exactly as the classic fixture behaves.
if [ "$PATCHES" = "false" ]; then
    EXTRA="$EXTRA surface_patches:=false"
fi
if [ "$NAV_MODE" = "nav2" ]; then
    EXTRA="$EXTRA nav2_controller:=$NAV2_CONTROLLER"
    [ -n "$NAV2_PROFILE" ] && EXTRA="$EXTRA nav2_profile:=$NAV2_PROFILE"
fi
FLOOR_VAR=""
[ -n "$FLOOR_NUMBER" ] && FLOOR_VAR="FLOOR_NUMBER=$FLOOR_NUMBER"

# CGROUP SCOPE (#28). The stack runs inside a transient `systemd --user` scope
# named agx-w<N> (agx-w0 = default partition), so that (a) kill_stack.sh can
# stop the WHOLE stack by cgroup, not just what its name/env sweep finds, and
# (b) compare_run.py can read per-stack cpu.stat / memory.peak / pids.current.
# Needs `loginctl enable-linger` for the user (set on the VM 2026-10-02) so the
# user manager outlives the ssh session. If no user bus is reachable we launch
# bare, exactly as before, and say so; compare_run then records null stats.
# The harness (compare_run.py) deliberately stays OUTSIDE the scope, as a
# sibling: its own rclpy spinning must not be billed to the stack.
SCOPE_UNIT="agx-w${WORKER:-0}.scope"
SCOPE_PREFIX=""
USER_MGR=$(systemctl --user is-system-running 2>/dev/null)
if command -v systemd-run >/dev/null 2>&1 \
   && { [ "$USER_MGR" = running ] || [ "$USER_MGR" = degraded ]; }; then
    SCOPE_PREFIX="systemd-run --user --scope --quiet --collect --unit=$SCOPE_UNIT -p TimeoutStopSec=15 --"
else
    echo "[fixture-up] WARNING: no systemd user manager (linger off / no user bus); stack runs WITHOUT a cgroup scope, stats will be null"
fi

READY_MODE=vec-pmp
[ "$NAV_MODE" = "nav2" ] && READY_MODE=nav2
# amcl publishes map->odom before it has localized; wait for its pose too.
AMCL_FLAG=""
[ "$LOCALIZATION" = "amcl" ] && AMCL_FLAG="--require-amcl"

cd "$WORKSPACE" || exit 2

# ROS's setup scripts read AMENT_TRACE_SETUP_FILES while it is unset, so `set -u`
# across the source exits on that line. This has cost this project time twice;
# the guard is deliberate and belongs around every sourcing in the repo.
set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u

for attempt in $(seq 1 "$TRIES"); do
    echo "[fixture-up] attempt $attempt/$TRIES  (partition=$PARTITION nav=$NAV_MODE loc=$LOCALIZATION corrector=$CORRECTOR patches=$PATCHES spawn=$SPAWN_X,$SPAWN_Y,$SPAWN_YAW floor=${FLOOR_NUMBER:-default})"

    bash tools/kill_stack.sh "$WORKSPACE" kill "$PARTITION" >/dev/null 2>&1

    tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n scratch
    tmux kill-window -t "$SESSION:$WINDOW" 2>/dev/null
    # A stale scope of the same name would make systemd-run refuse the unit.
    [ -n "$SCOPE_PREFIX" ] && systemctl --user reset-failed "$SCOPE_UNIT" >/dev/null 2>&1
    # tmux windows inherit the tmux SERVER's env, not ours: forward the DDS
    # transport override explicitly (#32 runs an arm with SHM disabled).
    tmux new-window -d -t "$SESSION" -n "$WINDOW" \
        "cd $WORKSPACE && DISPLAY=:0 ${FASTDDS_BUILTIN_TRANSPORTS:+FASTDDS_BUILTIN_TRANSPORTS=$FASTDDS_BUILTIN_TRANSPORTS} $SCOPE_PREFIX vglrun -d egl0 make fixture $FLOOR_VAR WORKER=$WORKER \
         CORRECTOR=$CORRECTOR LOCALIZATION=$LOCALIZATION FIXTURE_NAV_MODE=$NAV_MODE \
         HEADLESS=$HEADLESS_ USE_GPU_RENDER_ACCELERATION=false \
         FIXTURE_EXTRA_PARAMS=\"$EXTRA\" 2>&1 | tee $LOG"

    # The probe must run in the stack's own partition and domain, or it will
    # correctly report an empty graph and we would restart a healthy stack.
    if tools/with-worker "$WORKER" python3 tools/stack_ready.py --mode "$READY_MODE" $AMCL_FLAG --wait "$TIMEOUT" --settle 3; then
        if [ -n "$SCOPE_PREFIX" ]; then
            if systemctl --user is-active --quiet "$SCOPE_UNIT"; then
                echo "[fixture-up] scope $SCOPE_UNIT active"
            else
                echo "[fixture-up] WARNING: scope $SCOPE_UNIT not active; stats will be null"
            fi
        fi
        echo "[fixture-up] READY on attempt $attempt (log: $LOG)"
        exit 0
    fi

    echo "[fixture-up] attempt $attempt did not come up; last 20 lines of $LOG:"
    tail -20 "$LOG" 2>/dev/null | sed 's/^/    /'
done

echo "[fixture-up] FAILED after $TRIES attempts -- see $LOG"
exit 1
