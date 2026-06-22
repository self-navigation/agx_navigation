# Scout Mini self driving

## Installation

Recursively clone this repository
to ensure all submodule dependencies are cloned as well:

```bash
git clone https://github.com/self-navigation/agx_navigation.git --recursive
```

## Dependencies

This ROS2 workspace is built for ROS2 Jazzy and Gazebo Harmonic.
You can prepare the workspace with:

```bash
make setup
```

This will install the latest versions of ROS2 Jazzy and Gazebo Harmonic
before installing the dependenceis for this project.

If you only want to install the dependencies use:

```bash
make deps
```

## Usage

Run a Gazebo simulation:

```bash
make run SIM=true
```

Run on an Agilex Scout Mini R&D Kit:

```bash
make run SIM=false
```

More options are available with variables exposed in the Makefile,
which map directly to launch parameters.

If you want to build without running:

```bash
make build # or simply make
```

## RL runtime corrector

The wheel corrector (`agx_planning/rl_corrector/`) can run a reinforcement-learned
residual policy that adjusts per-wheel velocities to keep the robot on the
planner's frozen trajectory when terrain (ice/mud/oil) induces slip. Reward and
the tracking error are computed on Gazebo **ground-truth** pose, never `/odom`
(wheel odometry can't observe slip — that's the whole phenomenon being corrected).

### What the policy observes

The observation (built identically at train time and on-robot, so the policy sees
the same features both places) is the path-relative tracking error and its rates,
the nominal feed-forward wheel commands, the measured body twist, the previous
step's coefficients, and the **IMU** (yaw rate + body-frame acceleration). The IMU
is the key slip-observing input: unlike the `/odom` twist, which is wheel-derived
and so blind to slip, the gyro/accelerometer measure the body's *actual* motion, so
the policy can see the gap between commanded and real motion — the thing it corrects.

The obs layout is **fixed at training time and baked into the policy** (it's the
network's input width). It is a build-time choice, not a runtime switch: a policy
must be deployed with the exact `use_*` toggles it trained with, and they must be
held constant across curriculum phases, or the observation won't line up and the
corrector falls back to identity. `RLCorrectorConfig` (`rl_corrector/config.py`) is
the single source of truth for those toggles. PMP **costates** are off by default —
they only exist for recorded planner nominals (not the parametric training path),
so enabling them now would feed zeros at train but real costates at deploy.

### Training dependencies

The SAC training stack (stable-baselines3 + torch) is **not** part of `make deps`,
so a normal build never drags in torch. Install it separately:

```bash
make rl-deps
```

`torch` is also what the corrector needs on-robot for inference;
`stable-baselines3` is only required to train.

### Running a training session

Training uses two terminals: a **minimal** sim (physics + wheel controller +
odom only — no nav/planner/corrector, since the trainer is itself the command
source), and the trainer that drives it.

```bash
# terminal 1: bring up the minimal sim
make rl-sim HEADLESS=true

# terminal 2: train against it
make rl-train TIMESTEPS=200000
```

This trains with deterministic stepping (rtf>1 headless), terrain randomization
on, and saves the policy to `~/rl_corrector_policy.zip`. Override via Make
variables:

- `TIMESTEPS` — total environment steps (default `200000`).
- `TERRAIN` — `false` for a flat-ground bootstrap run (default `true`).
- `POLICY_OUT` — save path; `.zip` is appended (default `~/rl_corrector_policy`).
- `LOAD` — continue training from an existing policy `.zip` (see curriculum below).
- `TB` — TensorBoard log dir; sets `--tensorboard` (see monitoring below).
- `TRAIN_ARGS` — passthrough to the trainer, e.g. `TRAIN_ARGS="--device cuda"`.
- `SIM_SENSORS` — `true` to spawn the rendering sensors in the training sim
  (default `false`; see throughput note below).

**Throughput.** Training is tuned to run many× real-time, two ways. The training
sim drops the GPU lidar + RGB/depth cameras (`SIM_SENSORS=false`, the default) —
nothing in the loop consumes them and their per-tick rendering is the dominant
cost; the IMU/magnetometer stay on (no rendering). And the trainer lifts the
world's real-time cap (raises rtf, unlimits the update rate) so deterministic
stepping isn't pinned near 1×; it restores the cap on exit. To watch a run at
real-time instead, pass `TRAIN_ARGS="--realtime"`. (The IMU is on by default;
`TRAIN_ARGS="--no-imu"` drops it from the obs, but then deployment must match.)

> Only run **one** sim at a time. Two `rl-sim`/`rl_corrector_sim` instances share
> Gazebo's default transport partition and both advertise
> `/world/ordjo_world/set_pose` + `pose/info`, so a teleport can hit one server
> while the bridge confirms against the other — resets then fail. Run `make
> rl-kill` between runs if a previous sim/trainer was Ctrl-C'd or orphaned.

### Monitoring a run

The trainer prints a live progress bar (step count + ETA), and with `verbose=1`
SB3 logs a rollout table (`ep_rew_mean`, `ep_len_mean`, losses) to stdout every
few episodes — the quickest health check is whether `ep_rew_mean` trends up.

For TensorBoard, point the run at a log dir and open it alongside:

```bash
make rl-train TIMESTEPS=200000 TB=~/rl_tb
tensorboard --logdir ~/rl_tb        # then browse http://localhost:6006
```

TensorBoard ships with the `stable-baselines3[extra]` install from `make rl-deps`.

### Curriculum training

Learning slip correction is more stable if the policy first masters plain
tracking on flat ground, then continues on slippery terrain. `LOAD` chains the
phases: the second run restores the first run's weights and keeps training (so it
is a genuine continuation, not a fresh policy).

```bash
# phase 1: warm up on flat ground (fresh policy)
make rl-train TERRAIN=false TIMESTEPS=50000 POLICY_OUT=~/policy_flat

# phase 2: continue on terrain from the flat-ground policy
make rl-train TIMESTEPS=300000 POLICY_OUT=~/policy_terrain LOAD=~/policy_flat.zip
```

A loaded run **reuses the saved hyperparameters** (per-phase `--learning-rate`
etc. do not apply) and keeps counting timesteps from the loaded total, so the two
phases read as one curriculum in checkpoints/tensorboard. The replay buffer is
**not** restored — the terrain phase collects fresh experience, since the flat
phase's transitions came from different dynamics. Keep the observation/action
layout (`--action-dim`, `--imu`, `--costates`) identical across phases or the load
fails (the network's input width is fixed at the first phase).

### Deploying the policy

The corrector loads its policy from the `rl_corrector.policy_path` parameter.
Leaving it unset (the default) keeps the corrector a byte-identical identity
pass-through, so removing the policy file cleanly reverts behaviour.

```bash
ros2 run agx_planning runtime_corrector --ros-args \
    -p mode:=offline \
    -p rl_corrector.policy_path:=$HOME/rl_corrector_policy.zip
```
