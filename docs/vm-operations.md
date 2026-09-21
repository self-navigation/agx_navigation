# VM operations: routing, parallel sims, and the job queue

*Moved out of CLAUDE.md on 2026-09-21. This is reference detail, not a decision log — it is all still current.*

Training can run on a Proxmox VM (`danya02-gmatiukhin-ros2-gazebo`, VM 200) with a
Tesla V100 passed through. The [Justfile](Justfile) holds the remote workflow —
new commands go there rather than in the Makefile, which stays the source of truth
for building and running; the recipes only drive `make` over ssh.

## Reaching the VM: never choose a route by hand

There are two routes to the **same** machine — direct over the lab VPN
(`172.30.187.53`) and via a jump host (`llm_test2@kron.botik.ru` → `192.168.71.113`,
port 2202 on the *target*, not on kron). The VPN drops whenever the laptop lid
closes and takes ~15 minutes to come back, so which route works changes several
times a day. **Nothing should ever hardcode one.**

`ssh_config` defines a single host alias `agx` whose `ProxyCommand`
([tools/agx-route](tools/agx-route)) probes both and prefers direct. Because it
is a ProxyCommand it sits inside connection setup, so `ssh`, `scp`, `rsync -e
ssh`, `git` and every Justfile recipe are routed identically.

```bash
just <recipe>                  # already routed — nothing to override
ssh -F ssh_config agx          # works straight from a fresh clone
just ssh-setup                 # one-time: then plain `ssh agx`, `scp x agx:` work
just route-check               # which route is live; also busts the cache
AGX_ROUTE=jump just sync       # force a route (testing only)
tools/agx-run --detach 'make rl-train …'   # long run, returns immediately
tools/agx-run --tail /tmp/x.log            # poll it
```

The probe is a **TCP connect to port 22**, not a ping: the VPN's half-up states
answer ICMP while sshd is unreachable. The answer is cached in
`/tmp/agx-route.$UID` for 60 s (`AGX_ROUTE_TTL`), so a burst of recipes pays for
one probe — direct adds ~10 ms, the jump route ~1 s. If **neither** route works
the ProxyCommand exits non-zero with both failures spelled out rather than
hanging or silently falling back.

Two things that bit during implementation and will bite again:

- **A ProxyCommand's stdout is the encrypted channel and its stdin is the
  peer's.** Every diagnostic must go to stderr, and every probe must be run with
  `</dev/null` — a probe `ssh` that inherits stdin eats the parent's handshake
  bytes, and the connection dies with `Bad packet length`.
- `tools/agx-run --detach` exists because `ssh host 'setsid nohup … &'` holds the
  channel open for minutes after the remote process detaches: ssh waits for the
  streams the child inherited. `-f` plus redirecting all three fixes it (verified:
  returns in 0.19 s against a 12 s remote command). The mirror-image trap is that
  `ssh host 'cmd | tail'` prints nothing until exit, so a healthy long run looks
  frozen — detach, then `--tail` the log.

A PreToolUse hook ([.claude/hooks/agx-route-guard.sh](.claude/hooks/agx-route-guard.sh))
catches any Bash command that hardcodes an IP or the jump host and points it back
at `agx`. It only greps the command string — no probe, no latency.

```bash
just sync / remote-build      # rsync the working tree up, build there
just check-sim / kill-sim     # guard: refuse to start a 2nd Gazebo / clear it
just remote-sim               # headless sim in tmux (only ever one)
just remote-train p1          # a phase in its own tmux window, TB=runs
just remote-fixture tvlqr false   # corrector test rig, GUI on; false = no slip patches
just remote-log sim|train|fixture   # attach read-only
just tb                       # TensorBoard tunnelled to localhost:6006
just fetch-policies           # pull ~/rl_corrector_p*.zip back
just fetch-runs               # pull fixture run CSVs into gitignored run_data/
```

Non-obvious facts about that box, all of which cost time to work out:

- **Detach long runs and poll a log file** — `tools/agx-run --detach` does this
  correctly, see "Reaching the VM" above. Interrupting the local ssh does *not*
  kill the remote processes, which then fight the next launch (`pgrep -af`
  before relaunching). A fixture run is ~90 s: ~10 s discovery, ~20 s planning,
  ~15-60 s driving.
- **Never leave `set -u` on across `source .../setup.bash`.** ROS's setup scripts
  read `AMENT_TRACE_SETUP_FILES` while it is unset, so the script exits on that
  line. Wrap the sourcing in `set +u` / `set -u` rather than dropping `set -u`.
  This has cost time twice; on 2026-08-13 it killed `tools/queue_r_ladder.sh`
  *after* it had correctly waited for the in-flight sweep and logged `starting
  the r_omega ladder`, and the VM sat idle ~17 h before anyone read the log.
  Corollary for any detached overnight job: **make the log print progress after
  the setup, not just an intent line before it.** An "I am starting X" line that
  is the last line in a log means the failure is in the next few statements.
- **`packages.osrfoundation.org` is throttled to ~6 KB/s** from the VM (other
  mirrors run at ~800 KB/s), so `apt install gz-harmonic` stalls indefinitely.
  Workaround: `apt-get install --print-uris`, fetch the osrfoundation `.deb`s from
  a machine with a working route, drop them in `/var/cache/apt/archives/`, re-run
  the install. PyPI is *not* affected.
- **GPU passthrough (`hostpci0`) pins all guest RAM**, so virtio ballooning cannot
  return memory and the guest gets stuck at the `balloon:` floor while the host
  reserves the full `memory:`. Set `balloon: 0` for any passthrough VM — it costs
  the host nothing, since the pages are pinned regardless.
- **A Tesla V100 has no display engine.** The usual headless `ConnectedMonitor
  "DP-0"` xorg.conf cannot work — modes validate to `NULL` and you get a 640x480
  stub screen, which is why Sunshine reported "no display connected" and failed
  *every* encoder including software. The fix is a virtual framebuffer (`vga:
  virtio` on the VM) plus VirtualGL to route OpenGL to the V100.
- Cores beyond ~4 don't help: training is a single env stepping one Gazebo
  instance, and `TORCH_THREADS` is deliberately 1. Parallel envs would need
  per-instance `GZ_PARTITION`/`ROS_DOMAIN_ID` plumbing that does not exist yet.

## Parallel sims: `WORKER` (built and verified 2026-08-15)

**Several Gazebos now run on one box, and the "only ever ONE sim" rule is
retired** — replaced by "only ever one sim *per partition*". Everything above
that says otherwise is describing the default partition, where it remains true.

`WORKER=n` (1-9) is the single knob. It sets `GZ_PARTITION=agxn` and
`ROS_DOMAIN_ID=40+n`, and that is the entire mechanism: **no code changed
anywhere** — not in `GazeboBridge`, not in the launch files, not in the tuner.
Both libraries read their isolation setting from the environment at init, so a
`gz_transport.Node()` and an rclpy node constructed by unmodified code land in
whichever world their process was started in.

- `make rl-sim WORKER=1` / `make rl-train WORKER=1` / `make fixture WORKER=1`
- `just remote-sim rl_corrector.world 1`, `just remote-fixture tvlqr true truth 1`,
  `just gui 1`
- `tools/with-worker 1 python3 -m agx_planning.tuning.soak …` for anything not
  going through make. **`WORKER` unset execs with the environment untouched** —
  not `GZ_PARTITION=""` — because every number in this file was measured in the
  default partition and a change that silently moved it would invalidate the lot.

**Both variables are required and they cover different failures.** Without
`GZ_PARTITION` two `gz sim` servers both advertise `/world/rl_corrector/set_pose`
and resets go to whichever answers first. Without `ROS_DOMAIN_ID` both stacks
publish `/clock`, `/joint_states` and the wheel command topic, and each robot
receives the other's commands — *that* is the documented "wheels detach, links
fall through the floor" failure, and it is ROS-side, so **`GZ_PARTITION` alone
does not prevent it**.

**Verified live against a running job**, not in isolation — worker 1 was brought
up beside the sim job 60 was driving:

| check | result |
| --- | --- |
| gz topics visible per partition | 14 in default, 14 in `agx1`, no overlap |
| ROS topics | 36 in domain 0 (full stack), 21 in domain 41 (minimal sim) |
| entities in each world | 52 default (job 60's patches), 20 in `agx1` |
| **worker rollout `floor_6_v2_00004`** | **0.2901, 0.2901** vs the default partition's **0.2901 ± 0.0001** over 60 |
| job 60's per-eval time | 103 s before, 103 s during |

**The four-decimal agreement is the real result**: a worker is the same plant,
so numbers measured in parallel are comparable with everything already in this
file. Load went 3.5 → 7.8 on 12 cores with two sims plus a soak.

`just check-sim` is **scoped by partition** and defaults to `default`, so it is
exactly as strict as before for anyone not passing a worker, while worker 2 can
start beside worker 1. It now shells out to `tools/kill_stack.sh` in a new
`list` mode instead of running its own `pgrep`, so **the guard and the sweep can
no longer disagree** about what counts as a conflicting process. `just kill-sim`
takes a partition too but defaults to `all` — someone typing it after a bad
night wants everything gone, and having to enumerate partitions is the kind of
step that gets skipped.

Two traps found while building it, both of which produce a *convincing* wrong
answer rather than an error:

- **`ps -p "1,2,"` prints nothing, silently.** `tr '\n' ','` leaves a trailing
  comma, so the guard printed "REFUSING TO LAUNCH … (see above)" with nothing
  above it, and the pre-existing `STILL RUNNING:` diagnostic had the same bug —
  meaning a failed sweep had been reporting an empty list all along. Use
  `paste -sd,`.
- **A process can be isolated on the ROS side alone.** `ROS_DOMAIN_ID=41 ros2
  topic list` leaves a daemon with no `GZ_PARTITION`, which reads as "default"
  and blocks `check-sim default` over a process sharing nothing with it. The
  filter checks both variables; worker processes always carry both.

Not done, and the next thing to want: **the job queue is still single-lane**.
`tools/jobq.sh` runs one job at a time in the default partition, so parallelism
today means driving workers by hand. Per-worker queues are the obvious follow-up
and are what would turn ~55 core-hours of PMP labelling into an overnight run.

## Serializing VM work: the job queue (2026-08-14)

`tools/jobq.sh` + `just queue-{start,add,status,log,stop}` and a job library in
`tools/jobs/`. A directory queue with one long-lived runner; jobs can be added
at any time, including mid-run, so the idle stretch between a finished run and
the next session gets absorbed instead of lost. **A queued job must terminate on
its own** — a `while true` soak blocks everything behind it, so `tools/jobs/`
scripts use a bounded batch count.

It replaces per-run chain scripts, which failed twice over: they encode one
successor, and `queue_r_ladder.sh` died on `set -u` + ROS's `setup.bash` after
waiting correctly for its predecessor, idling the VM ~17 h. **The runner sources
ROS itself**, so that trap is unreachable by construction rather than by anyone
remembering. Its liveness check takes the lock rather than grepping the process
table — a `pgrep -f` pattern matches the ssh wrapper asking the question, so it
always reported a runner alive.

**The runner killed itself on its first job (fixed 2026-08-14).** The job body
ran inside a **brace group** ending in `exit $rc` — and a brace group is not a
subshell, so that `exit` ended the *runner*. Job 10 wrote its `EXIT rc=0` line,
the runner vanished before the `mv` to `done/`, and the next job sat pending
13 h. It is a subshell now. The signature to recognise: `status` shows a job in
**running** whose log already says `EXIT … rc=0`, and the runner NOT RUNNING —
that combination means the runner died between the two, not that the job hung.

`just queue-add` runs `sync` first, so **queueing a job rsyncs the working tree
onto the VM under whatever is currently running.** That is usually what you want
(a fix lands for the jobs behind it) but it does mean an in-flight job can pick
up edited code at its next `python3 -m`; the batch loops make that a real window.
