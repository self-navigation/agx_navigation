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
#   just remote-fixture tvlqr true amcl    # localize off the lidar instead of
#                                          # ground truth (slower: needs sensors)
remote-fixture corrector='tvlqr' patches='true' localization='truth': sync check-sim
    {{_ssh}} 'tmux has-session -t {{session}} 2>/dev/null || tmux new-session -d -s {{session}} -n scratch; \
        tmux kill-window -t {{session}}:fixture 2>/dev/null; \
        tmux new-window -d -t {{session}} -n fixture \
        "cd {{remote}} && DISPLAY=:0 vglrun -d egl0 make fixture CORRECTOR={{corrector}} \
         SURFACE_PATCHES={{patches}} LOCALIZATION={{localization}} \
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

# ------------------------------------------------- corrector comparison

# Rank recorded PMP trajectories by SHAPE, so a comparison can be run on
# genuinely different paths rather than the same archetype three times.
#
# This is not busywork. Every goal used in the 2026-07-25 TVLQR validation came
# out near-straight, 6-9 m, heading the same way (two were the same goal), so
# "TVLQR beats identity" had only ever been shown for one kind of path. Pick a
# STRAIGHT, an S-CURVE and a CORNER from this listing before running `compare`.
classify-plans pattern='/home/programmer/pmp_trajectories_v2/*.npz':
    {{_ssh}} 'cd {{remote}} && python3 tools/classify_plans.py "{{pattern}}"'

# Replay the SAME frozen plan under identity / TVLQR / RL and record the true
# path each drove. Needs `just remote-sim` up (GazeboBridge talks to it).
#
# Each corrector uses its OWN authority limits, deliberately: routing TVLQR
# through the RL residual channel would cap it with a limit it does not have
# when deployed, making the result a statement about the channel and not about
# the two control laws.
#
# `trajs` is a space-separated list of .npz paths -- quote it.
compare trajs policy='/home/programmer/rl_corrector_p0.zip' correctors='identity tvlqr rl' terrain='true': sync
    {{_ssh}} 'cd {{remote}} && source /opt/ros/jazzy/setup.bash \
        && source install/setup.bash \
        && PYTHONPATH=src/agx_navigation/agx_planning:$PYTHONPATH \
        python3 -m agx_planning.rl_corrector.compare_correctors \
        --trajectories {{trajs}} --correctors {{correctors}} \
        --policy {{policy}} --bridge gazebo \
        {{ if terrain == "true" { "--terrain" } else { "" } }} \
        --out-dir /tmp/compare 2>&1 | tee /tmp/compare.log'
    @echo "done -- pull it back with:  just fetch-compare"

# Pull the comparison CSVs back into gitignored compare_data/.
fetch-compare dest='compare_data':
    mkdir -p {{dest}}
    rsync -az --delete --info=stats1 -e "ssh {{ssh_opts}}" \
        {{host}}:/tmp/compare/ {{dest}}/
    @ls -1 {{dest}}

# Draw each corrector's true path on top of the others, one figure per
# trajectory. Offline/matplotlib, same rule as plot_run.py -- venv, not ROS.
plot-compare src='compare_data' out='figures':
    .venv/bin/python tools/plot_compare.py {{src}} --out {{out}} \
        || python3 tools/plot_compare.py {{src}} --out {{out}}
    @ls -1 {{out}}

# ------------------------------------------------- dated training runs

# Start a long training run in a DATE-LABELLED tree, so runs stay tellable
# apart after the fact: policies, checkpoints and TensorBoard logs all carry
# the same tag. `label` defaults to today's date; pass one to disambiguate a
# second run on the same day (e.g. `just train-long 20260730b`).
#
# Defaults encode the 2026-07-30 comparison's conclusions:
#   --no-corridor-terminates  episodes survive a breach, so the policy actually
#                             experiences (and can learn to recover from) error
#                             beyond the corridor. Training inside a 0.5 m tube
#                             is why the learned corrector had no recovery at all.
#   --start-offset 0.25       episodes BEGIN off-path, so recovery is on-policy
#                             rather than a state only reachable by failing.
#   --ground-friction         randomize the PLANT per episode (see terrain.py on
#                             why this and not a randomized slip_chi).
#   --recorded-dir            train on the real PMP library, not analytic
#                             primitives: the analytic curriculum exists to make
#                             the task survivable, which the corridor fix now
#                             does directly, and its 2-5 s episodes are nothing
#                             like the 200-step plans this is deployed on.
train-long timesteps='1500000' label=`date +%Y%m%d` recorded='/home/programmer/pmp_trajectories_v2': sync
    {{_ssh}} 'tmux has-session -t {{session}} 2>/dev/null || tmux new-session -d -s {{session}} -n scratch; \
        tmux kill-window -t {{session}}:train 2>/dev/null; \
        mkdir -p ~/runs_{{label}}; \
        tmux new-window -d -t {{session}} -n train \
        "cd {{remote}} && make rl-train \
            TIMESTEPS={{timesteps}} \
            POLICY_OUT=$HOME/runs_{{label}}/rl_corrector \
            TB=$HOME/runs_{{label}}/tb \
            TRAIN_ARGS=\"--recorded-dir {{recorded}} --no-corridor-terminates \
                        --start-offset 0.25 --ground-friction\" \
            2>&1 | tee /tmp/train_{{label}}.log"'
    @echo "training '{{label}}' started -- log: /tmp/train_{{label}}.log"
    @echo "watch it with:  just watch-train {{label}}   (opens on the server desktop)"

# Open a terminal ON THE SERVER'S DESKTOP (reach it with Moonlight) showing the
# live training output. Detached from this ssh, so closing the connection leaves
# it up.
#
# Attaches to the tmux window READ-ONLY (-r) rather than tailing the log file:
# the tqdm progress bar redraws with carriage returns, which `tail -f` renders
# as a wall of repeated lines instead of a moving bar. The tmux window is the
# real terminal, so it shows the bar as intended. -r means a stray keystroke on
# the desktop cannot kill the run.
watch-train label=`date +%Y%m%d`:
    -{{_ssh}} 'DISPLAY=:0 setsid nohup xfce4-terminal \
        --title="training {{label}}" \
        --command="tmux attach -t {{session}}:train -r" \
        </dev/null >/dev/null 2>&1 &'
    @echo "terminal opened on the server desktop for run '{{label}}'"

# Replay the shape-comparison over the checkpoints of a run, so the RL leg can
# be seen improving (or not) rather than judged on one arbitrary snapshot.
# Baselines are re-measured per checkpoint dir but are checkpoint-independent;
# plot_checkpoints.py keeps the first of each.
#
# `stride` subsamples: training checkpoints stay FREQUENT (they are the crash
# recovery for a multi-hour run, and the VM has been stopped mid-run before), so
# a 1.5M-step run leaves ~300 of them -- far more than a sweep wants to replay at
# ~9 Gazebo episodes each. stride=10 replays every 10th.
compare-checkpoints trajs label=`date +%Y%m%d` correctors='identity tvlqr rl' stride='10':
    {{_ssh}} 'set -e; cd {{remote}}; \
        source /opt/ros/jazzy/setup.bash; source install/setup.bash; \
        i=0; \
        for ck in $(ls -1v ~/runs_{{label}}/checkpoints/*.zip 2>/dev/null); do \
            i=$((i+1)); \
            [ $(( (i-1) % {{stride}} )) -ne 0 ] && continue; \
            step=$(echo "$ck" | grep -oE "[0-9]+_steps" | grep -oE "^[0-9]+"); \
            [ -z "$step" ] && continue; \
            outdir=/tmp/sweep_{{label}}/step_$(printf "%09d" "$step"); \
            echo "=== $ck -> $outdir"; \
            PYTHONPATH=src/agx_navigation/agx_planning:$PYTHONPATH \
            python3 -m agx_planning.rl_corrector.compare_correctors \
                --trajectories {{trajs}} --correctors {{correctors}} \
                --policy "$ck" --bridge gazebo --terrain \
                --out-dir "$outdir" || echo "  (failed, skipping)"; \
        done'
    @echo "sweep done -- pull it with:  just fetch-sweep {{label}}"

fetch-sweep label=`date +%Y%m%d` dest='sweep_data':
    mkdir -p {{dest}}
    rsync -az --delete --info=stats1 -e "ssh {{ssh_opts}}" \
        {{host}}:/tmp/sweep_{{label}}/ {{dest}}/
    @ls -1 {{dest}} | head

# Draw RL error vs training step per trajectory, with identity/TVLQR baselines.
plot-checkpoints src='sweep_data' out='figures' metric='max_cross':
    .venv/bin/python tools/plot_checkpoints.py {{src}} --out {{out}} --metric {{metric}} \
        || python3 tools/plot_checkpoints.py {{src}} --out {{out}} --metric {{metric}}

# ------------------------------------------------- TVLQR gain tuning

# Nelder-Mead search over (q_cross, r_omega) against real Gazebo rollouts.
# Needs `just remote-sim` up, and NOTHING else touching the sim -- one Gazebo.
#
# ~75 s per evaluation (three trajectories driven to completion), so 60
# evaluations is ~75 min. Runs in its OWN TMUX WINDOW and tees to a log, like
# `train-long`: it outlives the ssh session that started it (so a laptop can be
# shut), and `tail -f` on the log is not enough to drive it interactively.
# `ssh host 'cmd | tail'` shows nothing until the command exits, so a working
# run looks frozen -- read the log file, don't wait on the pipe.
#
# RESUMABLE: every evaluation is appended to the JSONL cache before the next
# starts, and re-running the same command replays it for free. A killed run
# loses at most the evaluation in flight. Delete the cache to start over --
# editing the trajectory list does the same thing, on purpose (the cache is
# keyed on it and refuses to resume onto a different problem).
# Trajectories come from config/eval_trajectories.yaml, NOT from a copy of the
# list here: this recipe used to carry its own hard-coded three, which went stale
# the moment the eval set changed, so the run would have been tuned against a
# different set than every document described.
tune-tvlqr evals='0' cache='/home/programmer/tvlqr_tune.jsonl': sync
    {{_ssh}} 'tmux has-session -t {{session}} 2>/dev/null || tmux new-session -d -s {{session}} -n scratch; \
        tmux kill-window -t {{session}}:tune 2>/dev/null; \
        tmux new-window -d -t {{session}} -n tune \
        "cd {{remote}} && source /opt/ros/jazzy/setup.bash \
         && source install/setup.bash \
         && PYTHONPATH=src/agx_navigation/agx_planning:\$PYTHONPATH \
         python3 -m agx_planning.tuning.tune_tvlqr \
             --trajectory-config {{remote}}/config/eval_trajectories.yaml \
             --max-evals {{evals}} \
             --cache {{cache}} --out /home/programmer/tvlqr_tuned.json \
             2>&1 | tee /tmp/tune_tvlqr.log"'
    @echo "tuning started in tmux window '{{session}}:tune' -- follow it with:  just tune-log"
    @echo "when it finishes:  just fetch-tune && just plot-tune"

tune-log:
    -{{_ssh}} 'tail -40 /tmp/tune_tvlqr.log'

# Separate the run-to-run variance into within-process drift vs. per-process
# noise: drive ONE trajectory n times inside a single process, then n times in n
# fresh processes, with everything else held fixed (same gains, same terrain
# seed, deterministic stepping). Whichever arm spreads wider names the cause --
# and they want opposite fixes, so this has to be settled before any tuning or
# corrector comparison means anything. ~25 s per rollout, so n=10 is ~10 min.
variance-probe n='10' traj='/home/programmer/pmp_trajectories_v2/floor_6_00042.npz': sync
    {{_ssh}} 'tmux has-session -t {{session}} 2>/dev/null || tmux new-session -d -s {{session}} -n scratch; \
        tmux kill-window -t {{session}}:var 2>/dev/null; \
        tmux new-window -d -t {{session}} -n var \
        "cd {{remote}} && bash {{remote}}/tools/run_variance_probe.sh {{n}} {{traj}} \
         2>&1 | tee /tmp/variance_probe.log; sleep 86400"'
    @echo "started in tmux window '{{session}}:var' -- follow with:  just variance-log"

variance-log:
    -{{_ssh}} 'tail -40 /tmp/variance_probe.log'

analyze-variance src="tune_data/variance_probe.jsonl":
    python3 tools/analyze_variance.py {{src}}

fetch-variance dest="tune_data":
    mkdir -p {{dest}}
    rsync -az --info=stats1 -e "ssh {{ssh_opts}}" \
        {{host}}:/home/programmer/variance_probe.jsonl {{dest}}/

# Pull the evaluation cache back so the landscape can be drawn locally.
fetch-tune dest='tune_data':
    mkdir -p {{dest}}
    rsync -az --info=stats1 -e "ssh {{ssh_opts}}" \
        {{host}}:/home/programmer/tvlqr_tune.jsonl {{dest}}/
    -rsync -az -e "ssh {{ssh_opts}}" {{host}}:/home/programmer/tvlqr_tuned.json {{dest}}/
    @ls -1 {{dest}}

# Draw what the search explored: the gain plane and the convergence curve.
plot-tune src='tune_data/tvlqr_tune.jsonl' out='figures':
    .venv/bin/python tools/plot_tune_landscape.py {{src}} --out {{out}} \
        || python3 tools/plot_tune_landscape.py {{src}} --out {{out}}

# Contact sheet of every recorded plan, for picking evaluation trajectories by
# eye (the IDs then go in config/eval_trajectories.yaml).
fetch-trajectories dest='traj_data':
    mkdir -p {{dest}}
    rsync -az --info=stats1 -e "ssh {{ssh_opts}}" \
        {{host}}:/home/programmer/pmp_trajectories_v2/ {{dest}}/

gallery src='traj_data' out='figures':
    .venv/bin/python tools/plot_trajectory_gallery.py {{src}} --out {{out}}

# Interactive single-step console against the running sim: step one physics tick
# at a time and watch WHEN entity changes actually commit. Built to chase the
# residual patch nondeterminism that batch rollouts cannot show. Attach a GUI
# with `just gui` and watch on Moonlight while driving this.
sim-console traj='/home/programmer/pmp_trajectories_v2/floor_6_00042.npz': sync
    {{_ssh}} -t 'cd {{remote}} && source /opt/ros/jazzy/setup.bash \
        && source install/setup.bash \
        && PYTHONPATH=src/agx_navigation/agx_planning:$PYTHONPATH \
        python3 -m agx_planning.tuning.sim_console --trajectory {{traj}}'
