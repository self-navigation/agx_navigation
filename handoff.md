# Handoff: corrector comparison + the 2026-07-30 training run

## State as of 2026-07-30 ~20:30

Branch `tvlqr-corrector`. **A ~7.5 h training run is in flight on the VM** --
see "The run in flight" below before starting anything that touches Gazebo.

Access (VPN is up): `ssh programmer@172.26.13.37`. Backup route via the jump
host: `ssh -J llm_test2@kron.botik.ru programmer@192.168.71.113 -p2202`.

**The whole ROS 2 / Gazebo workspace lives on that VM, not on the laptop.**
Code gets there with `just sync` (an rsync of the working tree) -- *not* by
pulling commits, so the VM's own `git branch`/`git log` is stale and misleading.
Don't try to fix it with git on the VM; just re-sync.

## What changed this session

### 1. Online-mode PMP is dead, and is not a bug to fix

The previous handoff proposed fixing a "slow `solve_bvp` on large
reorientations" bug in online mode. **That framing was wrong.** Online mode
worked when the planner used ACADOS; after the move to
`scipy.integrate.solve_bvp` it is not viable at all, at any heading delta --
the solver simply cannot run inside a real-time control loop. Do not spend more
time seeding initial guesses or adding solve deadlines to rescue it.

The only viable pipeline is **plan once offline, then have the runtime
corrector hold the frozen trajectory** -- which is what `PMP_MODE=offline` and
`make fixture` already do.

### 2. The old corrector validation was measuring one trajectory shape

Every goal used in the 2026-07-25 TVLQR validation came out near-straight,
6-9 m, heading the same way; `s23` and `clean` were literally the same goal.
So "TVLQR beats identity" had only ever been shown for one archetype.

`tools/classify_plans.py` now ranks planned paths by shape (straightness, total
/net/max turn, sign changes, arc-length resampled so the descriptors are
geometry and not sample density). Use it to pick a STRAIGHT + S-CURVE + CORNER
before making any corrector claim. Of the 100 files in
`~/pmp_trajectories_v2`: 58 CORNER, 15 S-CURVE, 12 STRAIGHT.

### 3. Three-way comparison, on shapes that differ

`just compare` replays one frozen plan under identity / TVLQR / RL and records
the true path each drove. Results (max|e_cross| / final_err, metres; Gazebo
physics, slip terrain):

| trajectory | shape | identity | tvlqr | rl (p0) | rl (p1_v3) |
| --- | --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | 0.62 / 4.88 | **0.26 / 1.71** | 4.59 / 5.21 | 6.34 / 8.12 |
| floor_6_00042 | S-CURVE | 6.83 / 6.86 | **2.24 / 2.58** | 7.89 / 8.09 | 3.66 / 3.93 |
| floor_6_00023 | CORNER | 19.45 / 17.18 | 3.73 / 0.54 | **1.23 / 1.05** | 1.62 / **0.41** |

- **TVLQR beats identity on every shape** -- first time shown off a straight.
- **RL is worse than open-loop on the straight and S-curve**, helping only on
  the corner. On the straight its error grows *monotonically* to one side: a
  steady bias, not chatter.
- **Identity's failure scales with curvature**: 0.62 -> 6.83 -> 19.45 m.
- TVLQR visibly oscillates about the path on curved plans (`q_cross=10`,
  `r_omega=0.25` look too hot). **Gain tuning is the obvious untried lever.**

Figures: `just fetch-compare && just plot-compare` -> `figures/*_compare.png`.

### 4. Why the RL corrector had no recovery behaviour

`is_failure` ended the episode at `|e_cross| > 0.5 m` and `start_offset` was
±0.08 m, so the *entire* training distribution was a 0.5 m tube around a path
the robot started on. It never observed a large error, so it never learned to
recover -- the 4-6 m divergences above are pure extrapolation. TVLQR recovers
from a 19 m excursion on the same rig because a Riccati law extrapolates by
construction.

Fixed for the new run (all in `config.py` / `train.py`, defaults unchanged so
nothing else moves):

- `corridor_terminates: bool = True` -- new config field; `--no-corridor-terminates`
  keeps the episode alive through a breach.
- `--start-offset` -- episodes begin off-path (0.25 used, was 0.08).
- `--ground-friction` -- randomizes the **plant** per episode.
- `w_effort` 0.1 -> 0.3 -- the identity prior now bites where identity works.

**`slip_chi` is the wrong knob for domain randomization here**, despite being
the obvious candidate. It only enters the *assumed* model (`track_effective` ->
`c_w`), never the physics; and with recorded Tier-B nominals the feed-forward
wheel commands were baked at a fixed chi, so perturbing it at training time
moves no command at all. `terrain.ground_friction_sampler` randomizes the ground
instead, which is the thing that actually changes the plant.

### 5. Reward fix that removing termination made mandatory

The cross-track penalty was purely quadratic, which was safe only while a breach
ended the episode at 0.5 m. Non-terminating, the robot reaches 14 m where
`w_cross * e^2` is ~1960 per step: measured `ep_rew_mean = -8.9e4` and
`critic_loss = 1.6e5`, returns spanning five orders of magnitude -- exactly the
critic divergence this reward's own SAC_5/8 history already records.

Now a **Huber**: quadratic inside `corridor_epsilon`, linear beyond, value *and*
slope matched at the knee, so every previously-tuned in-corridor behaviour is
untouched. `critic_loss` 1.6e5 -> **58.5**, `ep_rew_mean` -> -8.6e3.

**Do not revert to a pure quadratic while `corridor_terminates` is False.**

## The run in flight

Label `20260730`, 1.5M steps, ~47 fps, so ~7.5 h from ~20:30 on 2026-07-30.

```
~/runs_20260730/rl_corrector.zip     final policy
~/runs_20260730/checkpoints/         every 5k steps (crash recovery -- the VM
                                     has been stopped mid-run before)
~/runs_20260730/tb/                  TensorBoard  (just tb)
/tmp/train_20260730.log              log
just watch-train 20260730            live tqdm bar on the server desktop
                                     (Moonlight; read-only tmux attach)
```

Trained on the **recorded PMP library only**, not the analytic p1/p2/p3
curriculum. The analytic phases exist to make the task survivable, which the
corridor fix now does directly, and their 2-5 s (~30 step) primitives are
nothing like the 200-step plans this is deployed and evaluated on.

**Only one Gazebo at a time.** `just check-sim` refuses to launch if anything is
alive; `just kill-sim` clears it properly (`tmux kill-server` orphans `gz sim`).
The trainer needs the `rl:sim` window up -- it is, don't restart it.

## When the run finishes

```bash
just compare-checkpoints "/home/programmer/pmp_trajectories_v2/floor_1_00049.npz \
                          /home/programmer/pmp_trajectories_v2/floor_6_00042.npz \
                          /home/programmer/pmp_trajectories_v2/floor_6_00023.npz" 20260730
just fetch-sweep 20260730 && just plot-checkpoints
```

That draws RL error vs training step, one line per trajectory, with identity and
TVLQR as horizontal baselines -- so "is it improving, and *on which shapes*" is
answerable. A single aggregate number hides the straight-vs-corner split
entirely, which is the mistake this session found. `stride` defaults to 10
(checkpoints stay frequent for crash recovery; the sweep subsamples).

## Open, in rough priority order

1. **Tune the TVLQR gains.** It wins everywhere already but oscillates on curved
   plans; `q_cross`/`r_omega` are untouched since they were written. Cheapest
   remaining win, and it needs no training at all.
2. **Read the checkpoint sweep.** If RL still loses to identity on straights
   after the distribution fix, the residual channel or the identity prior is
   still wrong -- not the step count. A previous 500k-step run was already flat.
3. **`floor_1_00050` is a degenerate PMP plan**: `max_turn = 3.14 rad/step`, a
   literal pi-radian flip in one step over a 6 m path. Planner bug, not a
   control failure. Excluded from the comparison; worth its own look.
4. Nothing has been pushed to `origin`.

## Still unresolved from before

The offline-mode variance: the same recorded trajectory, run 4x in one
100-episode batch, gave max cross-track of 5.33/2.24/1.89/2.93 m. Chaos, reset
hygiene, dt mismatch and controller windup were all ruled out (see
`rl-corrector-diagnosis`). Isolated reruns are byte-identical, so it is specific
to the long-batch context; the untried candidate is accumulated sim-clock
floating-point precision in `GazeboBridge._wait_clock_advance`.

## Relevant memory files

- `corrector-three-way-comparison` -- the table above, the reward fix, run setup
- `online-mode-pmp-infeasible` -- why online mode is not worth fixing
- `ros2-gazebo-remote-only` -- VM-only workspace, `just sync` semantics
- `rl-corrector-diagnosis` -- full investigation history, retracted theories
- `corrector-validation-state`, `gpu-training-server`,
  `rl-corrector-turn-induced-corridor-breach`, `rl-corrector-checkpoint-path-bug`
