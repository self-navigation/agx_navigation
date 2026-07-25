# Remote GPU-server recipes.
#
# The Makefile stays the source of truth for building and running the stack;
# these recipes only *drive* it over ssh on the training VM. New commands go
# here rather than in the Makefile.
#
# Two routes to the same box. Direct (default):
#     just sync
# Via the jump host, from outside the lab network:
#     just host='programmer@192.168.71.113' \
#          ssh_opts='-J llm_test2@kron.botik.ru -p2202' sync

host     := "programmer@172.26.13.37"
ssh_opts := ""
remote   := "/home/programmer/agx_navigation"

# tmux window names on the server; `sim` and `train` are long-lived.
session := "rl"

_ssh := "ssh " + ssh_opts + " " + host

default:
    @just --list

# ---------------------------------------------------------------- sync / build

# Push the local working tree to the server (source only -- no build artifacts).
sync:
    rsync -az --delete --info=stats1 \
        -e "ssh {{ssh_opts}}" \
        --exclude='.git/' --exclude='build/' --exclude='install/' \
        --exclude='log/' --exclude='.*.stamp' --exclude='__pycache__/' \
        --exclude='*.pyc' --exclude='*.egg-info/' --exclude='acados/' \
        ./ {{host}}:{{remote}}/

# Build the workspace on the server.
remote-build: sync
    {{_ssh}} 'cd {{remote}} && make build'

# Interactive shell on the server, already in the workspace.
remote-shell:
    {{_ssh}} -t 'cd {{remote}} && exec bash -l'

# ---------------------------------------------------------------- sim guard

# Every recipe that starts a sim depends on this one, so the check is not
# something anyone has to remember.
#
# Two Gazebo instances share the default transport partition, and the damage is
# worse than the "resets silently break" note below suggests: BOTH instances
# spawn a scout_mini and BOTH run their own controllers and robot_state_publisher
# on the same topics. The robot then receives contradictory joint commands --
# observed result is wheels detaching and links sinking through the floor, plus a
# TF pose that wanders tens of metres while the wheels sit uncommanded.
#
# `tmux kill-server` does NOT protect you: it kills the tmux sessions but ORPHANS
# the gz processes they started, which keep running and keep publishing. That is
# exactly how the duplicate above got created. Check the process table, not tmux.
#
# Checking for `gz sim` alone is NOT sufficient, and that gap cost a whole
# measurement sweep: killing a fixture's `ros2 launch` leaves its children
# running, so Gazebo is gone while a full vector_field/pmp_planner/
# runtime_corrector trio is still alive and subscribed. The next fixture starts
# clean by this check and then stacks a second planner on the first, each
# planning from a different odom belief. So look for workspace nodes too.
check-sim:
    @{{_ssh}} 'if pgrep -u "$(id -u)" -a -f . 2>/dev/null | grep -v "pgrep\|grep\|rviz" \
            | grep -E "gz[ -]sim|{{remote}}"; then \
        echo; \
        echo "REFUSING TO LAUNCH: Gazebo and/or workspace ROS nodes are already"; \
        echo "running (see above). Stop them first:  just kill-sim"; \
        exit 1; \
      else echo "process table clear -- no Gazebo, no workspace ROS nodes"; fi'

# Kill Gazebo AND every ROS 2 node of this workspace on the server, then confirm
# the table is actually clear. Use this instead of `tmux kill-server`, which
# kills the panes and orphans everything they started.
#
# The sweep lives in tools/kill_stack.sh and is piped over stdin rather than
# inlined here -- see the long comment at the top of that script for why an
# inline `ssh host 'pkill -f ...'` silently kills only itself. It matches on
# process provenance (the workspace in the process's own environment), so new
# packages are covered without editing anything, and it spares RViz, which only
# subscribes and is the only view of a headless sim.
kill-sim:
    -@{{_ssh}} 'bash -s {{remote}}' < tools/kill_stack.sh

# ---------------------------------------------------------------- training

# Two sims share Gazebo's transport partition and resets silently break, so:
# Start the headless RL sim in a detached tmux window. Only ever run ONE.
remote-sim: sync check-sim
    {{_ssh}} 'tmux has-session -t {{session}} 2>/dev/null || tmux new-session -d -s {{session}} -n scratch; \
        tmux kill-window -t {{session}}:sim 2>/dev/null; \
        tmux new-window -d -t {{session}} -n sim \
        "cd {{remote}} && make rl-sim HEADLESS=true USE_GPU_RENDER_ACCELERATION=false 2>&1 | tee /tmp/rl-sim.log"'
    @echo "sim starting -- follow it with:  just remote-log sim"

# Needs `remote-sim` up for every phase except p0 (kinematic, no Gazebo).
# Run a training phase in tmux, e.g. `just remote-train p1`.
remote-train target='p1': sync
    {{_ssh}} 'tmux has-session -t {{session}} 2>/dev/null || tmux new-session -d -s {{session}} -n scratch; \
        tmux kill-window -t {{session}}:train 2>/dev/null; \
        tmux new-window -d -t {{session}} -n train \
        "cd {{remote}} && make {{target}} TB=runs 2>&1 | tee /tmp/rl-train.log"'
    @echo "training started -- follow it with:  just remote-log train"

# The controller test rig (`make fixture`): vec-pmp on the pre-baked map, no
# SLAM and no rendering sensors. GUI on, so it can be watched over Moonlight.
#   just remote-fixture                    # tvlqr, the corrector under test
#   just remote-fixture identity           # the do-nothing baseline
#   just remote-fixture tvlqr false        # no slip patches -- isolates planner
#                                          # geometry from slip excursions
remote-fixture corrector='tvlqr' patches='true': sync check-sim
    {{_ssh}} 'tmux has-session -t {{session}} 2>/dev/null || tmux new-session -d -s {{session}} -n scratch; \
        tmux kill-window -t {{session}}:fixture 2>/dev/null; \
        tmux new-window -d -t {{session}} -n fixture \
        "cd {{remote}} && DISPLAY=:0 vglrun -d egl0 make fixture CORRECTOR={{corrector}} \
         SURFACE_PATCHES={{patches}} \
         HEADLESS=false USE_GPU_RENDER_ACCELERATION=false 2>&1 | tee /tmp/fixture.log"'
    @echo "fixture starting -- follow it with:  just remote-log fixture"

# Attach a Gazebo GUI to the already-running headless server, on the server's
# desktop (reach it with Moonlight). Open and close it as often as you like --
# `gz sim -s` (the server) and `gz sim -g` (the GUI) are separate processes, so
# closing the window leaves physics stepping untouched. vglrun renders on the
# V100; without it the GUI falls back to llvmpipe and steals training cores.
gui:
    {{_ssh}} -n 'tmux has-session -t {{session}} 2>/dev/null || tmux new-session -d -s {{session}} -n scratch; \
        tmux kill-window -t {{session}}:gui 2>/dev/null; \
        tmux new-window -d -t {{session}} -n gui \
        "cd {{remote}} && source /opt/ros/jazzy/setup.bash && source install/setup.bash && \
         DISPLAY=:0 vglrun -d egl0 gz sim -g 2>&1 | tee /tmp/gz-gui.log"'
    @echo "GUI opening on the server desktop -- connect with Moonlight."

# Close the Gazebo GUI, leaving the server (and training) running.
gui-close:
    -{{_ssh}} 'pkill -f "gz sim -g"; pkill -f "gz-sim-gui"'

# Tail a tmux window's output (sim | train).
remote-log window='train':
    {{_ssh}} -t 'tmux attach -t {{session}}:{{window}} -r'

# Stop everything sim/training related on the server. Kills the tmux session
# AND the processes it spawned -- killing the session alone orphans Gazebo.
remote-kill:
    -{{_ssh}} 'cd {{remote}} && make rl-kill; tmux kill-session -t {{session}} 2>/dev/null'
    @just kill-sim

# ---------------------------------------------------------------- observability

# TensorBoard on the server, tunnelled to http://localhost:6006 locally.
tb:
    @echo "http://localhost:6006  (Ctrl-C to close the tunnel)"
    ssh {{ssh_opts}} -L 6006:localhost:6006 {{host}} \
        'cd {{remote}} && python3 -m tensorboard.main --logdir runs --port 6006'

# GPU / memory / process snapshot.
remote-status:
    {{_ssh}} 'nvidia-smi; free -h; \
        tmux list-windows -t {{session}} 2>/dev/null || echo "no {{session}} tmux session"'

# Pull fixture run data (run_recorder's CSVs) back for plotting. The
# destination is gitignored -- these are regenerable measurements, not source.
#   <run>_track.csv    per-sample true pose, nearest planned point, cross-track
#   <run>_plan.csv     the planned path, for drawing it alongside the real one
#   <run>_summary.txt  rms/max/final error
fetch-runs dest='run_data':
    mkdir -p {{dest}}
    rsync -az --info=stats1 -e "ssh {{ssh_opts}}" \
        {{host}}:/tmp/runs/ {{dest}}/
    @ls -1 {{dest}} | tail -20

# POLICY_OUT defaults to ~/rl_corrector_policy, phases to ~/rl_corrector_pN.
# Pull trained policies back from the server's $HOME.
fetch-policies dest='policies':
    mkdir -p {{dest}}
    rsync -az --info=stats1 -e "ssh {{ssh_opts}}" \
        {{host}}:'/home/programmer/rl_corrector_*' {{dest}}/
