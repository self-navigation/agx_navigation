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
#
# PARTITION SCOPING (added 2026-08-15, with parallel sims). Workers run in
# separate Gazebo partitions and DDS domains (see tools/with-worker), so an
# unscoped sweep would take down every worker to clean one. The third argument
# narrows the sweep by the process's OWN GZ_PARTITION:
#
#     kill_stack.sh <workspace> [kill|list] [all|default|agxN]
#
#   all      -- (default) every workspace process, whatever partition. This is
#               the pre-existing behaviour and stays the default deliberately:
#               an operator typing `just kill-sim` after a bad night wants
#               everything gone, and having to name partitions to clean up is
#               exactly the kind of step that gets skipped.
#   default  -- only processes with NO GZ_PARTITION, i.e. the unnamespaced sim
#               all earlier measurements were taken against.
#   agxN     -- only worker N.
#
# `list` prints what would be killed and exits non-zero if anything matched --
# that is the guard `just check-sim` runs, so the guard and the sweep can never
# disagree about what counts as a conflicting process.
set -u

WORKSPACE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${2:-kill}"
PARTITION="${3:-all}"

case "$MODE" in
    kill|list) ;;
    *) echo "kill_stack.sh: mode must be kill|list (got '$MODE')" >&2; exit 2 ;;
esac

# Does this process belong to the partition we were asked about? (in_partition
# inside collect() below.) Reads the
# process's own environment rather than its command line: GZ_PARTITION is
# inherited, so a gz server started under a worker carries it even though
# nothing about `gz sim -s -r <world>` says so.
# It checks ROS_DOMAIN_ID too, and that is not belt-and-braces. A process can
# be isolated on the ROS side alone -- `ROS_DOMAIN_ID=41 ros2 topic list` leaves
# a daemon behind with no GZ_PARTITION -- and on GZ_PARTITION alone that daemon
# reads as "default", so `check-sim default` refuses to launch over a process
# that shares nothing with the default stack. Worker processes always carry both
# (make and tools/with-worker set them together), so requiring both costs
# nothing and removes the false positive.

# RViz is deliberately spared: it only subscribes, so it cannot corrupt a run,
# and it is the only view of a headless sim. Keeping it alive means a sweep
# between runs does not cost the operator their window.
#
# Sparing the rviz2 process alone is not enough. It is normally started as
# `make rviz`, and that wrapper has the workspace in its environment while its
# command line says nothing about rviz -- so the wrapper got killed, and then
# lingered long enough to fail the verify pass and abort a whole sweep. Spare
# the ancestor chain too, whatever the launcher happens to be.
#
# Never kill ourselves or anything we are running under either -- this
# script's own shell has the workspace in its environment too, as does the ssh
# session and any tmux pane that sourced setup.bash -- nor anything descended
# from us: `left=$(collect)` runs in a command-substitution subshell that
# inherits the workspace environment, and the verify pass would otherwise call
# it a survivor and report failure on a table it just cleaned.
#
# ONE /proc PASS (2026-10-08). The old bash collect() forked ps/tr/grep per
# process and per ancestor hop: 4 s with 55 processes on an idle VM, and
# teardown calls it at least three times. With 14 stacks up it was a median
# 113 s per teardown, 31 of job 236's 56 worker-hours. Same selection rules,
# now read straight from /proc in one python process.
MYPID=$$
collect() {
    python3 - "$WORKSPACE" "$PARTITION" "$MYPID" <<'PY'
import os, re, sys
ws, partition, mypid = sys.argv[1], sys.argv[2], int(sys.argv[3])
uid = os.getuid()
ppid, cmd, env = {}, {}, {}
for d in os.listdir("/proc"):
    if not d.isdigit():
        continue
    pid = int(d)
    try:
        if os.stat(f"/proc/{d}").st_uid != uid:
            continue
        with open(f"/proc/{d}/stat") as fh:
            ppid[pid] = int(fh.read().rsplit(")", 1)[1].split()[1])
        with open(f"/proc/{d}/cmdline", "rb") as fh:
            cmd[pid] = fh.read().replace(b"\0", b" ").decode(errors="replace")
        with open(f"/proc/{d}/environ", "rb") as fh:
            env[pid] = fh.read().split(b"\0")
    except (OSError, ValueError, IndexError):
        ppid.pop(pid, None); cmd.pop(pid, None); env.pop(pid, None)

def chain(pid):
    while pid > 1 and pid in ppid:
        yield pid
        pid = ppid[pid]

exclude = set(chain(mypid))
for pid, c in cmd.items():
    if re.search("rviz", c):
        exclude.update(chain(pid))

def var(e, name):
    pre = name.encode() + b"="
    for kv in e:
        if kv.startswith(pre):
            return kv[len(pre):].decode(errors="replace")
    return None

def in_partition(e):                              # see the comment above collect()
    if partition == "all":
        return True
    part, dom = var(e, "GZ_PARTITION"), var(e, "ROS_DOMAIN_ID")
    if partition == "default":
        return not part and dom in (None, "", "0")
    return part == partition

wsb = ws.encode()
for pid in sorted(env):
    if pid in exclude or mypid in chain(pid) or not cmd.get(pid, "").strip():
        continue                                  # us, our tree, rviz, kernel threads
    if any(wsb in kv for kv in env[pid]) and in_partition(env[pid]):
        print(pid)
PY
}

if [ "$MODE" = "list" ]; then
    found=$(collect)
    if [ -n "$found" ]; then
        # paste, not `tr '\n' ','`: tr leaves a TRAILING comma and `ps -p 1,2,`
        # prints nothing at all, silently -- so the guard said "see above" with
        # nothing above it.
        ps -o pid=,args= -p "$(echo "$found" | paste -sd,)" 2>/dev/null | cut -c1-140
        exit 1
    fi
    echo "clear -- no workspace ROS nodes or Gazebo in partition '$PARTITION'"
    exit 0
fi

# CGROUP STOP FIRST (#28). fixture_up.sh launches each stack in a
# `systemd --user` scope agx-w<N>; stopping the scope signals EVERY process in
# the cgroup (SIGTERM, then SIGKILL after TimeoutStopSec), including children
# whose environment the sweep below would not recognise. The env sweep stays as
# the fallback: stacks launched without a scope, or a user manager that is
# unreachable. Failure here is never fatal.
scope_units() {
    case "$PARTITION" in
        all)     systemctl --user list-units --plain --no-legend 'agx-w*.scope' 2>/dev/null | awk '{print $1}' ;;
        default) echo agx-w0.scope ;;
        agx*)    echo "agx-w${PARTITION#agx}.scope" ;;
    esac
}
if command -v systemctl >/dev/null 2>&1; then
    for unit in $(scope_units); do
        if systemctl --user is-active --quiet "$unit" 2>/dev/null; then
            echo "stopping scope $unit"
            systemctl --user stop "$unit" 2>/dev/null \
                || systemctl --user kill -s KILL "$unit" 2>/dev/null
        fi
        systemctl --user reset-failed "$unit" >/dev/null 2>&1
    done
fi

for sig in TERM TERM KILL; do
    pids=$(collect)
    [ -z "$pids" ] && break
    # shellcheck disable=SC2086
    kill "-$sig" $pids 2>/dev/null
    # Poll instead of a fixed 4 s: most stacks are gone in well under a
    # second, and the fixed sleeps were 10-26 s of every bring-up (job 232).
    for _ in $(seq 1 20); do
        [ -z "$(collect)" ] && break
        sleep 0.2
    done
done

left=$(collect)
if [ -n "$left" ]; then
    echo "STILL RUNNING:"
    # See the note on the same construct above: tr's trailing comma makes ps
    # print nothing, which turned this diagnostic into a bare "STILL RUNNING:".
    ps -o pid=,args= -p "$(echo "$left" | paste -sd,)" 2>/dev/null | cut -c1-140
    exit 1
fi
# FastDDS leaves its SHM segments/port mutexes in /dev/shm when a process is
# killed (#32: ~7900 entries after one day of campaigns). `shm clean` only
# removes segments whose owner is dead, so live stacks in other partitions
# are safe.
( set +u; source /opt/ros/jazzy/setup.bash && fastdds shm clean ) 2>&1 | sed 's/^/shm: /'
echo "stack clear -- no workspace ROS nodes or Gazebo in partition '$PARTITION' (RViz spared)"
