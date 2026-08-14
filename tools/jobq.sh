#!/bin/bash
# A one-at-a-time job queue for the VM's single Gazebo instance.
#
# WHY THIS EXISTS
# ---------------
# Only ONE sim may run at a time (two drivers on one sim silently corrupt every
# rollout -- see CLAUDE.md), so long measurement runs must be serialized. Until
# now that was done by bespoke "wait for the previous script, then run mine"
# chain scripts. Two problems with those, both paid for:
#
#   * they encode ONE successor, so a new idea has to either interrupt the
#     in-flight run or wait for a human to be present at the exact moment it
#     ends -- and the user checks in occasionally, not continuously;
#   * queue_r_ladder.sh died on `set -u` + ROS's setup.bash AFTER waiting
#     correctly for its predecessor, and the box sat idle ~17 h (2026-08-13).
#
# So: a directory queue with a single long-lived runner. Drop a script in
# `pending/` at any time -- while a job runs, or while the queue is empty -- and
# it executes in lexical order without anyone being present. The idle VM between
# a finished run and the next session is the resource this recovers.
#
# THE RUNNER SOURCES ROS, NOT THE JOBS. That is deliberate: it makes the
# `set -u` trap unreachable by construction rather than by everyone remembering.
# Jobs are plain command scripts and inherit a working environment.
#
# LAYOUT   ~/jobq/{pending,running,done,failed}/  and  ~/jobq/logs/<job>.log
#
# USAGE (on the VM)          just queue-* wraps these from the laptop
#   tools/jobq.sh runner     # start the runner (detach it)
#   tools/jobq.sh status
set -uo pipefail

Q=${JOBQ_ROOT:-$HOME/jobq}
REMOTE=${AGX_REPO:-$HOME/agx_navigation}
mkdir -p "$Q"/{pending,running,done,failed,logs}

runner() {
    # One runner only. Without this a second `queue-start` puts two jobs on one
    # sim, which is exactly the failure the queue exists to prevent.
    exec 9>"$Q/.runner.lock"
    if ! flock -n 9; then
        echo "[jobq] a runner already holds the lock; not starting another" >&2
        exit 1
    fi

    # See the header: sourcing lives HERE so no job can trip over it.
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    # shellcheck disable=SC1091
    source "$REMOTE/install/setup.bash"
    set -u
    export PYTHONPATH="$REMOTE/src/agx_navigation/agx_planning:${PYTHONPATH:-}"
    cd "$REMOTE" || exit 1

    echo "[jobq] runner up at $(date -Is), pid $$, queue $Q"
    while true; do
        # Lexically first pending job. `sort` not glob order, so a job added
        # mid-scan cannot jump the queue by luck of readdir ordering.
        job=$(find "$Q/pending" -maxdepth 1 -type f -name '*.sh' -printf '%f\n' \
              | sort | head -n1)
        if [[ -z $job ]]; then
            sleep 30
            continue
        fi

        mv "$Q/pending/$job" "$Q/running/$job" 2>/dev/null || continue
        log="$Q/logs/${job%.sh}.log"
        echo "[jobq] === START $job at $(date -Is) -> $log"
        {
            echo "[jobq] START $job $(date -Is)"
            bash "$Q/running/$job"
            rc=$?
            echo "[jobq] EXIT $job rc=$rc $(date -Is)"
            exit $rc
        } >>"$log" 2>&1
        rc=$?

        if [[ $rc -eq 0 ]]; then
            mv "$Q/running/$job" "$Q/done/$job"
            echo "[jobq] === DONE  $job at $(date -Is)"
        else
            mv "$Q/running/$job" "$Q/failed/$job"
            echo "[jobq] === FAIL  $job rc=$rc at $(date -Is)"
        fi
        # A failed job must not take the queue down with it: the next one may
        # be unrelated, and an idle box is the thing being fixed here.
    done
}

# Is a runner alive? Tested by trying to take its lock, NOT by pgrep: a pgrep
# pattern is matched against every cmdline including the ssh/bash wrapper that
# is asking the question, so `pgrep -f 'jobq.sh runner'` reports itself and the
# check is always true. The lock cannot lie -- only a live runner holds it.
# Exits 0 if a runner is up, 1 if not.
runner_alive() {
    ( flock -n 8 ) 8>"$Q/.runner.lock" && return 1 || return 0
}

status() {
    echo "== runner"
    if runner_alive; then
        echo "  UP (holding $Q/.runner.lock)"
    else
        echo "  NOT RUNNING"
    fi
    for d in running pending done failed; do
        echo "== $d"
        find "$Q/$d" -maxdepth 1 -type f -name '*.sh' -printf '  %f\n' | sort || true
    done
    echo "== recent log lines"
    for f in $(ls -t "$Q"/logs/*.log 2>/dev/null | head -3); do
        echo "  --- $f"
        tail -n 3 "$f" | sed 's/^/    /'
    done
}

case "${1:-status}" in
    runner) runner ;;
    status) status ;;
    # For scripts: exit 0 if a runner is up. Used by `just queue-start` so it
    # is idempotent without a self-matching pgrep.
    check)  runner_alive ;;
    *) echo "usage: $0 {runner|status|check}" >&2; exit 2 ;;
esac
