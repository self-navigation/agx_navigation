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

# ---------------------------------------------------------------- training

# Two sims share Gazebo's transport partition and resets silently break, so:
# Start the headless RL sim in a detached tmux window. Only ever run ONE.
remote-sim: sync
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

# Stop everything sim/training related on the server.
remote-kill:
    -{{_ssh}} 'cd {{remote}} && make rl-kill; tmux kill-session -t {{session}} 2>/dev/null'

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

# POLICY_OUT defaults to ~/rl_corrector_policy, phases to ~/rl_corrector_pN.
# Pull trained policies back from the server's $HOME.
fetch-policies dest='policies':
    mkdir -p {{dest}}
    rsync -az --info=stats1 -e "ssh {{ssh_opts}}" \
        {{host}}:'/home/programmer/rl_corrector_*' {{dest}}/
