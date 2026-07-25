#!/usr/bin/env bash
# Kill every Gazebo process and every ROS 2 node belonging to this workspace,
# then verify the table is actually clear.
#
# Processes are identified by PROVENANCE, not by name: a node launched from this
# workspace has the workspace path in its own environment (AMENT_PREFIX_PATH,
# GZ_SIM_RESOURCE_PATH, ...), inherited from `install/setup.bash`. Matching on
# that means a newly added package is covered the day it is added, with nothing
# to update here -- a hand-maintained list of node names silently stops covering
# the stack the moment someone adds a package.
#
# Run it by piping it in -- `ssh host bash -s < tools/kill_stack.sh` -- never as
# `ssh host 'pkill -f ...'` with the patterns inline. pkill -f matches full
# command lines, and a shell invoked with the patterns as arguments carries them
# in its OWN command line: the pkill matches the shell running it, the connection
# dies, and nothing else is killed. It looks like a clean sweep and is a no-op.
#
# Killing `ros2 launch` alone is not enough either. Launch children are separate
# processes that outlive their parent, so a partial kill leaves a second
# vector_field/pmp_planner/runtime_corrector trio alive; the next fixture stacks
# on top of it, and two planners plan from two different odom beliefs. The
# visible symptom is a plan drawn from where the robot ended up LAST run.
set -u

WORKSPACE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# RViz is deliberately spared: it only subscribes, so it cannot corrupt a run,
# and it is the only view of a headless sim. Keeping it alive means a sweep
# between runs does not cost the operator their window.
#
# Sparing the rviz2 process alone is not enough. It is normally started as
# `make rviz`, and that wrapper has the workspace in its environment while its
# command line says nothing about rviz -- so the wrapper got killed, and then
# lingered long enough to fail the verify pass and abort a whole sweep. Spare
# the ancestor chain too, whatever the launcher happens to be.
SPARE_RE='rviz'

# Never kill ourselves or anything we are running under -- this script's own
# shell has the workspace in its environment too, as does the ssh session and
# any tmux pane that sourced setup.bash.
ancestors() {
    local pid=$1
    while [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null; do
        echo "$pid"
        pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    done
}

spared_tree() {
    local pid
    # Every RViz-ish process, plus everything it was started from.
    for pid in $(pgrep -u "$(id -u)" -f "$SPARE_RE" 2>/dev/null); do
        ancestors "$pid"
    done
}

# Never kill ourselves, anything we run under, or the operator's RViz window.
MYPID=$$
readarray -t EXCLUDE < <({ ancestors "$MYPID"; spared_tree; } | sort -u)

is_excluded() {
    local pid=$1 e p hops=0
    for e in "${EXCLUDE[@]}"; do [ "$pid" = "$e" ] && return 0; done
    # ...and anything descended from us. `left=$(collect)` runs collect in a
    # command-substitution subshell, which inherits the workspace environment and
    # is born AFTER the exclusion list was built -- so the verify pass sees that
    # subshell, calls it a survivor, and the script reports failure on a table it
    # just cleaned. Whether that happens at all is a race, which is why this
    # passed when run by hand and failed inside the sweep.
    p=$pid
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null && [ "$hops" -lt 20 ]; do
        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
        [ "$p" = "$MYPID" ] && return 0
        hops=$((hops + 1))
    done
    return 1
}

collect() {
    local pid environ args
    # Only our own processes: /proc/<pid>/environ is unreadable for anyone
    # else's, and scanning them just produces a screenful of permission errors.
    for pid in $(pgrep -u "$(id -u)" '' 2>/dev/null || ps -u "$(id -u)" -o pid=); do
        is_excluded "$pid" && continue
        # NUL-separated; tr makes it greppable. Unreadable => not ours.
        environ=$({ tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null) || continue
        grep -qF "$WORKSPACE" <<<"$environ" || continue
        args=$({ tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null)
        [ -z "$args" ] && continue          # kernel thread
        echo "$pid"
    done
}

for sig in TERM TERM KILL; do
    pids=$(collect)
    [ -z "$pids" ] && break
    # shellcheck disable=SC2086
    kill "-$sig" $pids 2>/dev/null
    sleep 4
done

left=$(collect)
if [ -n "$left" ]; then
    echo "STILL RUNNING:"
    # shellcheck disable=SC2046
    ps -o pid=,args= -p $(echo "$left" | tr '\n' ',') 2>/dev/null | cut -c1-140
    exit 1
fi
echo "stack clear -- no workspace ROS nodes or Gazebo running (RViz spared)"
