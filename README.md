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
- `TRAIN_ARGS` — passthrough to the trainer, e.g.
  `TRAIN_ARGS="--device cuda --tensorboard /tmp/tb"`.

A flat-ground warmup followed by a terrain run, for example:

```bash
make rl-train TERRAIN=false TIMESTEPS=50000 POLICY_OUT=~/policy_flat
make rl-train TIMESTEPS=300000 POLICY_OUT=~/policy_terrain
```

### Deploying the policy

The corrector loads its policy from the `rl_corrector.policy_path` parameter.
Leaving it unset (the default) keeps the corrector a byte-identical identity
pass-through, so removing the policy file cleanly reverts behaviour.

```bash
ros2 run agx_planning runtime_corrector --ros-args \
    -p mode:=offline \
    -p rl_corrector.policy_path:=$HOME/rl_corrector_policy.zip
```
