## What this is

A ROS 2 **Jazzy** / Gazebo **Harmonic** workspace for autonomous navigation of an
Agilex Scout Mini (skid-steer, 4 wheels). The repo root *is* the colcon workspace;
everything lives under [src/](src/). Vendored dependencies are git submodules
(`scout_ros2`, `ugv_sdk`, `rslidar_sdk`, `ch10x-imu-driver`, `rudn-ordjo-building`
world/models, and a `scikit-fmm` fork under [depend/](depend/)) — clone with
`--recursive`, and treat submodule code as upstream unless a change is clearly
intended for it.

First-party code is in [src/agx_navigation/](src/agx_navigation/):

| package | role |
| --- | --- |
| `agx_bringup` | all launch files + config; the only place topology/params are wired |
| `agx_planning` | vector field, PMP planner, runtime corrector, RL corrector (Python) |
| `agx_chassis` | `twist_to_wheels`, `wheel_odometry` |
| `agx_planning_msgs` | `PlannerTrajectoryChunk` msg, `PlanToGoal` action |
| `pointcloud_utils`, `rviz_toggles` | C++ helpers / RViz plugin |

## Commands

Everything goes through the [Makefile](Makefile); it sources
`/opt/ros/jazzy/setup.bash` + `install/setup.bash` for you. Prefer `make <target>`
over raw `colcon`/`ros2` so the env and launch params stay consistent.

```bash
make                      # == make build (deps + colcon build --base-paths src)
make clean                # rm -rf install build log .*.stamp
make setup                # one-time: install ROS 2 Jazzy, Gazebo, system deps
make deps                 # rosdep + pip install of the workspace python packages
make test                 # unit tests (pytest, no ROS needed)
make run SIM=true         # full stack in Gazebo;  SIM=false runs on the robot
make online / offline     # run with NAV_MODE=vec-pmp and the matching PMP_MODE
make nav2                 # run with the nav2 stack instead
make fixture              # controller test rig: pre-baked map, no SLAM/sensors
make fixture CORRECTOR=identity SURFACE_PATCHES=false   # baseline, no slip
make rviz / make teleop
```

Analysis nodes that only make sense against the fixture (see "Measuring a
fixture run"):

```bash
ros2 run agx_planning run_recorder --ros-args -p use_sim_time:=true -p run_name:=x
ros2 run agx_bringup random_goals --ros-args -p use_sim_time:=true -p count:=5
ros2 run agx_planning slip_ident  --ros-args -p use_sim_time:=true -p cmd_mode:=wheels
```

Anything that drives the robot and measures the response must take its timing
from the **ROS clock**, not the wall clock: `make rl-sim` is unthrottled and runs
at ~30x realtime, so wall-clock phase timing moves the robot ~30x further than
intended while sensors (stamped in sim time) report the true duration. A result
wrong by a suspiciously round factor of 20-30 is this, not physics.

`make build` is stamp-gated (`.build.stamp` vs. every file under `src/`), so a
touched file forces a rebuild; delete the stamp if a build seems stale.

Run a single test:

```bash
PYTHONPATH=src/agx_navigation/agx_planning python3 -m pytest \
  src/agx_navigation/agx_planning/test/unit/test_rl_reward.py::test_name -v
```

Unit tests live in [src/agx_navigation/agx_planning/test/unit/](src/agx_navigation/agx_planning/test/unit/)
and cover the *pure* RL-corrector modules (`coeff`, `obs`, `reward`, `nominal`,
`env` via the kinematic bridge) — no ROS, no Gazebo, no torch import path.
Keeping those modules ROS-free is deliberate; don't add `rclpy` imports to them.

Make variables in `PARAM_VARS` (`SIM`, `HEADLESS`, `FLOOR_NUMBER`, `NAV_MODE`,
`PMP_MODE`, `USE_SERVER`, `DO_CORRECTIONS`, `PORT_NAME`) are lowercased and passed
straight through as launch arguments — adding a new launch arg usually means
adding its name there too.

## Architecture

### Runtime pipeline (vec-pmp mode)

```
SLAM/rtabmap map ─► vector_field ─► pmp_planner ─► runtime_corrector ─► JointGroupVelocityController
                    (FM2 field)     (PMP TPBVP)    (playback + residual)   (4 wheel velocities)
```

- **`vector_field`** ([vector_field/node.py](src/agx_navigation/agx_planning/agx_planning/vector_field/node.py)) —
  Fast Marching Square: EDT → speed profile → `skfmm.travel_time` → `-grad T`.
  Publishes the packed field on `/vector_field/planner_data`; the gradient
  magnitude doubles as a confidence signal (low ⇒ cut locus / near goal).
- **`pmp_planner`** ([pmp_planner/node.py](src/agx_navigation/agx_planning/agx_planning/pmp_planner/node.py)) —
  indirect optimal control (Pontryagin) on a **5D wheel-space skid-steer model**
  `x = (p_x, p_y, θ, w_l, w_r)`, control = per-wheel accelerations, solved as a
  TPBVP with `scipy.integrate.solve_bvp`. The long module docstring is the
  authoritative spec of the model and cost terms — read it before touching the
  dynamics or the cost weights. Two modes:
  - `online` — runs its own control loop, publishes wheel commands on
    `/pmp_planner/wheel_cmd`.
  - `offline` — a `PlanToGoal` action **server** that rolls out a whole trajectory
    and streams it back as chunked feedback.
- **`runtime_corrector`** ([runtime_corrector/node.py](src/agx_navigation/agx_planning/agx_planning/runtime_corrector/node.py)) —
  the only writer of `/wheel_velocity_controller/commands`. Mirrors the planner's
  mode (relay in `online`, action *client* + trajectory playback in `offline`, see
  `trajectory_buffer.py`). Everything funnels through `_emit() -> _correct()`,
  which is the seam where the RL residual applies. `JointGroupVelocityController`
  latches its last command, so **every terminal path must publish an explicit
  zero** — silence keeps the wheels spinning.

In `offline` mode the corrector buffers the **whole** rollout before driving
(`wait_for_complete`, default true): the planner emits a 0.5 s chunk roughly
every 0.68 s on the baked map, so streaming playback starved, and the stall path
publishes zero — which brakes the wheels mid-trajectory and then resumes from a
sample assuming them already spinning. The robot stands still for the entire
planning phase and then drives; that is expected, not a hang.

Playback is indexed by **time**, so anything that costs forward speed (including
the corrector's own work) makes the robot fall short of the goal rather than
arrive late. Note also that "arrival" is looser than `goal_tolerance_xy` (0.05 m,
tighter than the chassis holds): stopping within `4x` that counts as success, and
the completion sentinel on `/goal_pose` fires on **any** terminal outcome —
it means "nobody is pursuing a goal", not "arrived". Read the action result to
tell those apart.

Left/right pair speeds expand to the controller's joint order
`[front_left, rear_left, front_right, rear_right] = [w_l, w_l, w_r, w_r]`.

### Launch topology

[main.launch.py](src/agx_navigation/agx_bringup/launch/main.launch.py) is the entry
point and composes: `gz_sim` (only when `sim:=true`) → `robot_control`
(scout_description + `sim_control` or `life_control` + EKF) → `slam` (delayed 10 s,
rtabmap) → `nav` which branches on `nav_mode` into `nav2.launch.py` or
`vec_pmp.launch.py`. Launch files resolve each other via
`agx_bringup.utils.launch_file` / `cfg_file`; topic names come from
`agx_bringup.constants.Topics` rather than string literals.

`localization` picks what provides `/map` and `map`→`odom` — one knob, because
an estimator needs a map to localize against. `slam` (the default, rtabmap as
above); everything else swaps in
[static_map.launch.py](src/agx_navigation/agx_bringup/launch/static_map.launch.py)
with the baked map and differs only in the estimator:

| value | `map`→`odom` from | what it represents |
| --- | --- | --- |
| `slam` | rtabmap, mapping online | real deployment |
| `amcl` | nav2_amcl vs. the baked grid | deployment after a good site survey; realistic *and* repeatable |
| `truth` | Gazebo pose ([truth_localization.py](src/agx_navigation/agx_bringup/agx_bringup/truth_localization.py)) | sim-only; the corrector's performance **ceiling** |
| `none` | pinned to identity | raw wheel odometry; fastest, least honest |

Only `amcl` consumes the lidar, so only it needs `sim_sensors:=true`.

**`none` cannot evaluate a pose-feedback corrector**, and that is not a subtlety
— it invalidated a whole day of measurements. Wheel odometry over-reports
distance travelled by 0.6–0.7 m over one fixture run here. Open-loop control
ignores pose and is unharmed; a pose-feedback corrector *drives to the bias*, so
TVLQR stopped 0.74 m short while believing it had arrived within 5 cm. Measure
the corrector under `truth`, the system under `amcl`, and use `none` only for
things that genuinely do not care (planner geometry, throughput, smoke tests).
Ground truth reaches the control path **only** as that transform, exactly where a
real estimator's output would go.

### Static map fixture (controller testing)

`make fixture` == `make run NAV_MODE=vec-pmp PMP_MODE=offline LOCALIZATION=truth
sim_sensors:=false`. `LOCALIZATION` defaults to `truth` *here* (unlike `make run`,
which defaults to `slam`) because this is the corrector rig — see the table above
for why `none` is the wrong default for it. It exists because rtabmap is a bad fixture for testing the
*corrector*: it needs the robot to drive before any map exists (so no plan from a
standing start), it intermittently never initialises on this world's untextured
walls, and it yields a slightly different map each run — so two corrector runs
are never comparable.

The map is baked ahead of time from the same meshes Gazebo collides against, by
[tools/bake_floor_map.py](src/rudn-ordjo-building/tools/bake_floor_map.py) in the
`rudn-ordjo-building` submodule (which owns the meshes, hence the map). It slices
the floor GLBs over the lidar height band and rasterizes to `maps/floor_N.png` +
`.yaml`; `rudn_ordjo_building/map_publisher.py` serves that as a latched
`OccupancyGrid`. The PNG is deliberately a plain greyscale image (254 free / 0
wall / 205 unknown) so it can be hand-edited to block a doorway or carve a
shortcut.

Three couplings the baker cannot verify at runtime — if any changes, the baked
map goes silently stale:

- the GLBs are Y-up and `model_template.sdf` rolls the link +90°, so mesh
  `(x,y,z)` → world `(x,-z,y)`;
- `gz_sim.launch.py` spawns the floor at `(23, 5)` (`--floor-origin`);
- `spawn_floor.launch.py` drops `center` on floors ≥4 and `right` on floors ≥6.

`trimesh` is an *offline* dependency only — run the baker from a venv; it is
deliberately not in any package's `install_requires`.

Under `truth` and `none` nothing consumes the lidar or cameras, so the whole
pointcloud pipeline is gone and the sim runs much faster; `amcl` pays for the
lidar because it localizes off it. Either way the map itself never updates, so
this is the wrong rig for testing navigation — use `localization:=slam` for that.

`SURFACE_PATCHES=false` removes the low-friction ground patches. They default to
on (slip is what the corrector exists to handle), but they sit *under the spawn
point* — the robot starts inside the `icy` patch — so with them on, a wall strike
has two candidate causes and the log cannot tell them apart. Turn them off when
debugging planner geometry, on when testing the corrector.

### Measuring a fixture run

Scoring uses Gazebo **ground truth**, never `/odom` — wheel odometry is a
prediction from wheel speeds and shares the errors being measured. It once
reported 7 mm of cross-track error on a run that ended metres off course.

- `run_recorder` ([run_recorder.py](src/agx_navigation/agx_planning/agx_planning/run_recorder.py))
  writes `<run>_track.csv` / `_plan.csv` / `_summary.txt`. Sim-only and publishes
  nothing — an instrument, never part of the control path.
- `random_goals` ([random_goals.py](src/agx_navigation/agx_bringup/agx_bringup/random_goals.py))
  samples reachable goals from the baked map, inflated by a clearance radius and
  restricted to the robot's connected component.
- `just fetch-runs` pulls the CSVs into gitignored `run_data/`;
  [tools/plot_run.py](tools/plot_run.py) renders path + deviation figures
  (matplotlib in a venv, offline-only like the map baker).

Each measured run needs a **freshly started fixture**: odometry is never reset,
and after one run it is already ~0.6 m from truth, so a second run plans from a
lie. Teleporting the robot does not help — odom would not know it moved.

Two traps when consuming these CSVs or driving the fixture by hand:

- **`nan` is expected** in `cross_track`/`plan_*` for every sample before the
  planner publishes its path. `float("nan")` parses without raising, so guarding
  only `ValueError` silently admits NaN and one NaN turns any `max()`/`mean()`
  into NaN. Mask non-finite values explicitly.
- **Publishing a goal races discovery.** `/goal_pose` has several subscribers and
  a single message reaches only those already matched; losing `vector_field`
  gives `'Timeout waiting for vector field'`, losing the corrector means nothing
  drives. Use `ros2 topic pub -w <n>`, and note `random_goals`'
  `expected_subscribers` counts its *own* subscription. A matched count is still
  not a ready channel, hence its `settle` delay.

### Corrector-model constants

`slip_chi` is the skid-steer yaw loss: `omega_actual = omega_ideal / chi`. Measure
it with `ros2 run agx_planning slip_ident`
([slip_ident.py](src/agx_navigation/agx_planning/agx_planning/slip_ident.py)),
which references the **gyro** — so it runs unchanged on the real robot, and a
sim/real difference is a statement about friction, not method. `calibrator.py`
cannot identify it: it compares commands against `/odom`, and both sides share
the missing slip term.

Two known-wrong things left deliberately unchanged, because they affect the real
robot and want a decision rather than a patch:

- `wheel_odometry` integrates heading with the ideal relation and no `chi`, so
  its yaw overstates rotation by ~`chi`.
- `ekf_params.yaml` fuses that biased wheel yaw rate (`odom0_config` index 11)
  alongside the gyro's unbiased one. A Kalman filter cannot reject bias, only
  noise, so the estimate settles between right and wrong. The file documents the
  mask layout and the fix.

### RL runtime corrector

[rl_corrector/](src/agx_navigation/agx_planning/agx_planning/rl_corrector/) trains a
SAC residual policy that scales per-wheel feed-forward commands to hold the frozen
planner trajectory under slip. Key invariants:

- **`config.py` (`RLCorrectorConfig`) is the single source of truth** shared by
  training and deployment. The `use_*` toggles define the observation *layout*,
  which is baked into the network's input width — they are build-time choices, not
  runtime switches. A policy must be deployed with the exact toggles (and
  `coeff_k`) it trained with, and they must stay fixed across curriculum phases.
- Reward and tracking error use Gazebo **ground-truth** pose, never `/odom` —
  wheel odometry cannot observe slip, which is the whole phenomenon being corrected.
  The IMU is the slip-observing input in the observation.
- The env talks to a **`Bridge`** ([bridge.py](src/agx_navigation/agx_planning/agx_planning/rl_corrector/bridge.py)),
  never to Gazebo directly: `KinematicBridge` (fast, Gazebo-free, used by tests and
  `make p0`) or `GazeboBridge` (real physics, needs `make rl-sim` up).
- Kinematics constants in `RLCorrectorConfig` intentionally duplicate
  `pmp_planner/config.py` (to avoid pulling scipy/skfmm into the pure modules) —
  **keep them in sync manually**.
- With `rl_corrector.policy_path` unset (the default) the corrector is a
  byte-identical identity pass-through; that's the fail-safe.

Training (see [README.md](README.md) for full detail):

```bash
make rl-deps                          # stable-baselines3[extra] + torch + gymnasium
make rl-sim HEADLESS=true             # terminal 1: minimal sim (no nav/planner/corrector)
make rl-train TIMESTEPS=200000        # terminal 2
make p0 | p1 | p2 | p3 | curriculum   # phased curriculum, chained with --load
make rl-kill                          # ALWAYS run this after a Ctrl-C'd/orphaned run
```

Only **one** sim at a time — two instances share Gazebo's default transport
partition and both advertise `set_pose`/`pose/info`, so resets silently break.

Worse, if the second instance is a full `make run`/`make fixture` rather than
`make rl-sim`, both spawn a `scout_mini` and both run their own controllers and
`robot_state_publisher` on the same topics. The robot gets contradictory joint
commands and **physically disintegrates** — wheels detach, links fall through the
floor — while TF reports a pose wandering tens of metres with the wheels sitting
uncommanded. If you see that, count the `gz sim` processes before debugging
anything else; it is not a physics bug.

**`tmux kill-server` does not stop the sim.** It kills the tmux sessions but
*orphans* the `gz sim` processes they started, which keep running and keep
publishing. Always check the process table (`pgrep -af 'gz[ -]sim'`) before
launching, not the tmux session list. `just check-sim` does this and refuses to
launch if anything is alive; every sim-starting `just` recipe depends on it, and
`just kill-sim` clears the table properly.

`make rl-deps` is *not* the only thing that installs torch, despite the name:
`agx_planning/setup.py` lists `torch` and `stable-baselines3` in `install_requires`,
so plain `make deps` already drags in the whole CUDA wheel stack (~3-4 GB of
`nvidia_*` wheels). Budget for that on a fresh machine or a slow link; `make
rl-deps` afterwards is nearly a no-op. Move torch to an extra if a robot/laptop
install ever needs to avoid it.

`make rl-sim` already runs **without the rendering sensors** — `SIM_SENSORS`
defaults to `false`, dropping the GPU lidar and RGB/depth cameras, since the
trainer consumes only ground-truth pose, the `/odom` twist and the IMU. A slow
realtime factor during training is physics, not rendering; don't go looking for
sensor overhead to trim. (`make run` is the one that pays for sensors + SLAM.)

## Remote GPU training server

Training can run on a Proxmox VM (`danya02-gmatiukhin-ros2-gazebo`, VM 200) with a
Tesla V100 passed through. The [Justfile](Justfile) holds the remote workflow —
new commands go there rather than in the Makefile, which stays the source of truth
for building and running; the recipes only drive `make` over ssh.

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

- **Detach long runs and poll a log file.** `ssh host 'cmd | tail'` shows nothing
  until the command exits, so a working run looks frozen; and interrupting the
  local ssh does not kill the remote processes, which then fight the next launch
  (`pgrep -af` before relaunching). Use
  `setsid nohup script </dev/null >/tmp/x.log 2>&1 &`, then read `/tmp/x.log`.
  A fixture run is ~90 s: ~10 s discovery, ~20 s planning, ~15-60 s driving.
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

## Current work: making the corrector work

**This section is a living log — append to it, or rewrite the parts it
contradicts, whenever a run or an experiment establishes something new.** It is
the standing context for what is being worked on right now; the rest of this
file describes the system, which changes far more slowly. Date each claim, and
delete a claim outright when it is superseded rather than leaving both versions.

The single active goal is getting the runtime corrector to hold a frozen PMP
trajectory under slip. Nothing else is in progress.

### Terrain patches are inherited across processes (found 2026-08-01)

`GazeboBridge._remove_terrain` only removed patches that *its own process*
spawned, and the sim deliberately outlives any one process. So a trainer that
exited mid-episode left its patches in the world, and the next
`compare_correctors` inherited them. Two distinct failures, not one:

- a leftover `rl_ground` from a `--ground-friction` run is a low-friction slab
  under the entire trajectory — a different plant than the run intends;
- `create` on an existing name *fails*, so an inherited `rl_patch_0` also
  displaces the patch this run meant to spawn.

Fixed: the bridge now sweeps `rl_ground` and `rl_patch_0..7` by name at
construction. At most 4 patches ever exist (`n_range=(1,3)` at every call site,
plus `rl_ground`); the rest is headroom.

**The 2026-07-30 three-way comparison table is not trustworthy** — it was
measured with a trainer's patches almost certainly still in the world. Same
correctors, same trajectories, clean world, 2026-08-01 (max|e_cross| / final_err):

| trajectory | shape | identity | tvlqr | 2026-07-30 identity |
| --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | 0.28 / 0.48 | 0.01 / 0.02 | 0.62 / 4.88 |
| floor_6_00042 | S-CURVE | 0.22 / 0.19 | 1.55 / 1.08 | 6.83 / 6.86 |
| floor_6_00023 | CORNER | 1.21 / 3.45 | 0.23 / 0.05 | 19.45 / 17.18 |

On a clean world **TVLQR loses to identity on the S-curve**, which the
contaminated table hid. "TVLQR beats identity on every shape" is retracted.

### The 20260730 training run did not learn a usable policy

1.5M steps, 9h41m, 8864 episodes, **2 successes**. `failure_rate 0.88`. Two
diagnostics matter more than the step count:

- **`ent_coef` ran away to 3.31.** SAC's auto-tuned entropy coefficient belongs
  well below 1; at 3.31 the entropy bonus dominates the return and the optimal
  policy under the *effective* objective is near-random. `actor_loss 4.46e3` is
  mostly that term.
- **`critic_loss` is back to 1.23e4.** The Huber reward fix had brought it to
  58.5. Huber bounds the *slope*, not the return: over a 200-step episode with
  no corridor termination the accumulated linear tail still diverges.

The 16-checkpoint sweep (2026-08-01, `figures/checkpoints_max_cross.png` and
`figures/*_checkpoints.png`) settles it: **no RL checkpoint beats the identity
baseline on any of the three shapes, at any point in training.** There is no
trend — max|e_cross| wanders between roughly 0.3 and 6.6 m — and the best
checkpoint is **800k** (0.59 / 0.91 / 1.06 m on straight / S-curve / corner),
after which it degrades: by 1.5M the S-curve is back to 4.65 m. Training past
~800k made the policy worse, which is what an `ent_coef` of 3.31 predicts.
More steps is not the fix; the entropy target and the unbounded return are.

Three-way comparison at the best (800k) checkpoint, clean world
(max|e_cross| / final_err, m) — `figures/*_compare.png`:

| trajectory | shape | identity | tvlqr | rl (800k) |
| --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | 0.11 / 0.49 | **0.01 / 0.02** | 0.66 / 0.25 |
| floor_6_00042 | S-CURVE | **0.20 / 0.18** | 1.55 / 0.52 | 0.96 / 0.43 |
| floor_6_00023 | CORNER | 1.51 / 3.90 | **0.23 / 0.05** | 1.04 / 1.46 |

So on a clean world: TVLQR is excellent on the straight and the corner and
**bad on the S-curve** (it oscillates — visible as loops in the path panel);
identity is the best S-curve tracker and only fails at the corner, where it ends
3.9 m out; RL is never best at anything.

Note identity re-measured as 0.11 / 0.20 / 1.51 here against 0.28 / 0.22 / 1.21
in the sweep an hour earlier — same seed, same terrain, deterministic stepping.
That residual run-to-run spread is the still-unexplained offline-mode variance
(see [[rl-corrector-diagnosis]]); it is small enough not to affect any
conclusion above, but do not read two-decimal differences as signal.

### TVLQR gain tuning (built 2026-08-01)

`agx_planning/tuning/` searches `(q_cross, r_omega)` by Nelder-Mead against real
Gazebo rollouts. `just tune-tvlqr` (detached, ~75 s per evaluation, ~70 min for
60), then `just fetch-tune && just plot-tune`.

- `simplex.py`, `objective.py`, `cache.py` are **pure and unit-tested** — no ROS,
  no Gazebo, no torch, same rule as the RL pure modules. 25 tests, and they are
  aimed at one thing: proving the search *minimizes*. A tuner that maximizes
  produces an identical-looking log and hands back the worst gains it found.
- **`--max-evals` bounds candidate GAIN PAIRS, never rollout length.** Every
  evaluation drives every selected trajectory start-to-goal.
- **The trajectory set is fixed, never sampled**, so candidates are always
  compared on identical work. A rollout that *fails* makes the whole evaluation
  invalid (`inf`), never a mean over the survivors: the trajectories differ
  hugely in difficulty, so averaging whatever finished rewards gains that crash
  the hard rollouts.
- **Resumable at zero cost.** Nelder-Mead is deterministic given its objective,
  so a resumed run replays the JSONL cache to reconstruct the search and
  re-measures nothing. The cache is keyed on the trajectory list + seed and
  refuses to resume onto a different problem.
- **`--max-evals 0` (the default) runs to convergence**, since the search has
  nights available and a converged answer beats a predictable finish time.
  Three guards make unbounded mode safe: a 15-minute per-evaluation timeout (a
  healthy one is ~75 s), an abort after 5 consecutive failures, and —
- **failures are never cached.** This one was learned the hard way: killing the
  tuner mid-evaluation invalidated the bridge's rclpy context, after which every
  rollout failed in 2 ms. 56 bogus `inf` evaluations were written in three
  seconds, and had they been memoized, every future resume would have replayed
  them as real measurements. `inf` means "the sim broke", almost never "these
  gains are bad". Failed records are still written (with `_failed: true`) for
  diagnosis, but never returned from the cache.
- Search runs in **log10** of both gains: a step is a ratio, and no move can
  propose a negative gain.
- Every evaluation records its **per-trajectory** errors, not just the aggregate,
  so the landscape can be re-analysed per shape without re-driving anything.

Baseline at the current `q_cross=10 / r_omega=0.25`: **0.487 m** mean
max|e_cross| over the three-trajectory set (0.243 straight / 0.224 S-curve /
0.993 corner), measured 2026-08-01 by the tuner itself.

**First converged run (2026-08-02, 132 evaluations, ~2.3 h):**
`q_cross=7.22`, `r_omega=0.369` → **0.183 m**, from 0.487 m.
`figures/tvlqr_tune_landscape.png`, history in `tune_data/tvlqr_tune.jsonl`.

**Do not adopt those gains on this evidence alone.** The per-trajectory panel
shows `floor_6_00042` swinging between ~0.2 m and 7 m *throughout* the run at
near-identical gains — the run-to-run variance is much larger than the claimed
improvement. Taking the minimum over 132 noisy draws selects partly for a lucky
draw (winner's curse), so 0.183 m is biased low and the true value at those
gains is likely worse. What the run does support is that the *region* around
q≈7, r≈0.37 is better than the default, since the simplex clustered there.

Before adopting: re-measure the tuned gains and the defaults ~5x each and
compare distributions, not single numbers (~12 min). More generally, until the
variance in the "ideas queued" list is understood, **any tuning result needs
repeat measurements**, and a search that ranks candidates on one sample each is
resolving noise as often as signal.

**Unexplained, flagged not fixed:** that same run scores TVLQR at 0.224 m on
floor_6_00042 where `just compare` scored 1.549 m an hour earlier — same gains,
same seed, same code path. It is the offline-mode variance again but far larger
than the identity-leg spread. The tuner is *internally* consistent (one process,
one code path, fixed seed), which is what the search needs, but **do not compare
a tuned number against a `compare` number** until this is understood.

### Choosing evaluation trajectories

`config/eval_trajectories.yaml` holds the working set and a candidate list.
`just gallery` renders all 100 plans (`figures/trajectory_gallery.png`), each
rotated onto its principal axis so shape is comparable at a glance.

**The automatic labels mislead.** `classify_plans.py` calls 58 of 100 CORNER, but
in the gallery most of those are visually straight lines: the descriptor is
tripped by the in-place reorientation the PMP planner puts at the *start* of a
plan — a large heading change over no distance. Trust the picture. Likewise
`floor_6_00042`, used as "the S-curve" in every comparison so far, is really an
L with one rounded bend. Genuine S-curves: `floor_6_00028` (cleanest),
`00024`, `00047` (zigzag), `00056` (tight V). A true U-turn: `floor_6_00031`.

### Deterministic mode was never actually paused (found 2026-08-02, evening)

**This supersedes every measurement in this file, including the terrain-spawn
result below.** `WorldControl.pause` is a plain proto3 bool, so a request that
sets only `multi_step` sends `pause: false` — and gz applies it. Every step we
issued therefore stepped the world `n` ticks *and un-paused it*, leaving it
FREE-RUNNING until the next call, for however long the CPU gave it.

So "deterministic mode" was running an unbounded, wall-clock-dependent amount of
extra physics per control step. Symptom, once the trace made it visible: control
steps advancing **0.42 s of sim time instead of `control_dt` = 0.1**. Nothing
reported a problem — `lost_steps` and `stale_pose_steps` were both 0, correctly:
the world was not dropping steps, it was doing *extra* ones.

Fixed in `_world_control`: deterministic mode re-asserts `pause=True` on every
multi_step. Plus `_ensure_paused()`, which pauses and then **verifies the sim
clock stopped**, retrying and finally raising — the old code fired a best-effort
pause at construction and never checked, and the ack is unreliable.

**Result on floor_6_00042, 5 rollouts: max|e_cross| spread 0.0013 m**
(1.9539-1.9552), from 0.375 m before this and 6.70 m before the terrain fix.

Two more seeds were then closed, both wall-clock-paced work that fed real
physics ticks:

- **The teleport confirm loop ran for 0.5 s of WALL time**, so the robot got
  12-31 physics ticks to fall and settle depending on machine timing
  (`reset_ticks`). Now a fixed 20 (`_set_pose_stepped`). It had been written off
  as self-correcting because each retry yanks the body back — true of x/y, false
  of the vertical and contact state.
- **The reset settle ran a fixed 5 steps** and left the robot micro-bouncing by
  ~2e-5 m in z. Now converges (`reset_settle_z_tol`), leaving ~1e-6.

### Where the reproducibility floor actually is (2026-08-02, 10 rollouts)

| metric | mean | sd | spread |
| --- | --- | --- | --- |
| **max\|e_cross\|** | 1.9551 | **0.0002** | 0.0007 |
| rms_cross | 0.6214 | 0.0154 | 0.0503 |
| final_err | 0.5877 | **0.2633** | 0.8360 |

**`max|e_cross|` — the tuner's objective — is now reproducible to four decimals,
so single-sample ranking is finally legitimate.** The 0.487 → 0.183 m effect the
first tuning run claimed is three orders of magnitude above this noise floor.

**`final_err` is NOT reproducible and must not be used as an objective**, or must
be averaged over repeats. Why, from the per-column onsets: with everything else
fixed, the remaining seed is ~1e-13 in the wheels' residual speed (they settle to
~1e-9 rad/s, not to zero), and it is amplified at the **turn reversal around step
165**, where `omega` crosses zero (+1.37 → 0.02 → -1.29) and the skid-steer's
lateral friction switches direction. A contact-mode switch at float-level
asymmetry: genuine chaos, not a bug, and not worth chasing further.

Ruled out along the way, so don't re-propose: the ROS-publish-vs-gz-step race
(wheel speeds diverge at step 2, *after* the pose at step 1 — the command path is
a consequence, not a cause), and any terrain difference (`terrain`, `sim_time`
and `world_steps` now never differ between rollouts).

### Instrumentation: per-step state traces

`GazeboBridge.enable_trace(path)` writes one CSV row per control step and per
reset phase: full pose (incl. z and quaternion), twist, wheel speeds, IMU, the
command that produced the step, sim clock, step counters, and a digest of every
`rl_*` entity pose. `variance_probe --trace-dir` writes one per rollout.

`tuning/trace_diff.py` (pure, unit-tested) reports the FIRST step two rollouts
differ at and which column moved first — the distinction that matters:
`cmd*` moving while state is identical means **our controller** is the
non-determinism; state moving under an identical command means physics; a
`world_steps`/`lost_steps` difference means a step was dropped, not physics at
all. It also timestamps when the xy separation crosses each order of magnitude,
which is how "flat for 150 steps then grows" was distinguished from chaos.
Cumulative counters are rebased per rollout — the world is deliberately not
restarted between runs, so comparing `sim_time` raw reports a fake 250-step
divergence on every pair.

`tuning/trace_dump.py` prints selected rows of one trace, for when a run is bad
in isolation rather than merely different from another.

### The run-to-run variance was an 8 mm reset error amplified by patch edges (solved 2026-08-02)

The blocking mystery — TVLQR scoring 0.22 m and 1.55 m on identical inputs — is
resolved, and it was **not** a bug in the stepping. Evidence, in the order it
landed (`tuning/variance_probe.py`, `tuning/reset_probe.py`):

- **It is not process reuse.** 10 rollouts in one process spread 6.70 m; 10 in
  ten processes spread 3.86 m. Comparable, with **no trend against rollout
  index** (Spearman rho = -0.14). The accumulated-sim-clock-float hypothesis in
  `_wait_clock_advance` is **wrong** — delete it as a candidate.
- **The physics is perfectly deterministic.** Four of those ten rollouts
  returned 0.223 m agreeing to three decimals, the rest landing on distinct
  reproducible values (1.5, 1.7, 4.7, 6.9). Discrete modes, not smooth noise.
- **`reset()` is clean, including from motion.** `reset_probe` resets after
  idle / forward / spin / reverse: `v` and `omega` come back **exactly 0.00000**
  every time, heading to 0.00000, `lost_steps` 0. Residual velocity was never
  the problem. (`reset_ticks` does vary 1..67 because the `_set_pose` confirm
  loop is wall-clock-paced, but it is self-correcting — each retry yanks the
  body back — so it is harmless.)
- **What remained was ~8 mm of positional spread**, from the robot sliding as it
  fell from `reset_z=0.20` onto its settled height. Tightening this to **exactly
  0.000 m** (see below) changed the spread not at all — so the reset was *not*
  the cause, and the "8 mm amplified by patch edges" theory is **retracted**.
- **THE ACTUAL CAUSE: patches were silently not spawning.** With the world
  paused, `/world/<w>/create` blocks for the whole ack timeout and returns
  **False** — while creating the entity anyway, some ticks later
  (`tuning/spawn_diag.py` proves this: ack False, entity present afterwards).
  `_apply_terrain` requested a create and started driving immediately, so
  whether an episode had its patches came down to service timing. Roughly
  **4-5 of every 10 rollouts ran on BARE GROUND**, scoring ~0.2247 m —
  identical to `--no-terrain`, which is exactly the "clean mode" that made the
  data look bimodal. The rest scored 1.5-6.9 m. Two different plants, pooled.

**Fixes**, all in `GazeboBridge`:
- `_wait_entities` / `_wait_entities_gone` step the world until pose/info
  actually shows the patches present (or gone) before the rollout starts.
  pose/info is the only honest answer — the ack is useless in both directions.
- Do **not** re-issue a create for a "missing" patch: it is almost always in
  flight, and re-creating then genuinely fails because the name now exists. An
  obvious-looking retry loop made things worse before this was understood.
- `terrain_missing` records any patch that never appeared, and a rollout with
  one raises rather than being quietly recorded as a sample.
- Reset tightening (kept, though it was not the bug): `reset_z` = measured
  settled height 0.1806 m plus a re-place/re-settle loop
  (`reset_place_tol=0.001`), giving x/y/theta spread of exactly 0.00000 across
  resets from idle/forward/spin/reverse. `lost_steps` counts steps where the
  world did not advance before the wall-clock deadline (observed: 0).

**Result on floor_6_00042, 8 rollouts:** spread **0.375 m** (1.71-2.08), down
from 6.70 m, with zero failures and no bare-ground mode. Not yet perfect —
0.375 m is still above what several queued experiments want to resolve — but
the plant is now the same one every run.

**Consequences for past results.** Any single-rollout comparison made before
this is suspect, including the 2026-08-02 three-way table and — most of all —
the converged tuning run: 132 evaluations ranked on one noisy sample each, at a
noise level (metres) far above the claimed 0.487 → 0.183 m improvement. **The
tuned gains `q_cross=7.22 / r_omega=0.369` are not supported by that run** and
must be re-derived. Re-measure with the fixed reset before trusting anything.

### The patch friction values are unvalidated, and half of them are black ice

**Nobody has ever checked these against reality** — they were picked on a laptop,
at a much slower real-time factor, by eye, to make the robot visibly slip. Asked
about twice before and lost both times; hence this section. Current `PROFILES`
(`src/rudn-ordjo-building/rudn_ordjo_building/surface_patches.py`) vs. real
rubber-on-surface coefficients:

| profile | mu | real-world equivalent |
| --- | --- | --- |
| `rough` | 2.5 | above dry rubber on concrete — effectively "cannot slip" |
| `directional_x/y` | 1.0 / 0.15 | grips one axis, slides the other |
| `slippery` | 0.2 | wet smooth tile / oily floor — **realistic** |
| `icy` | 0.05 | polished or wet black ice — real, but not an indoor floor |

For reference: dry concrete 0.7-1.0, wet concrete 0.5-0.7, tyre on ice 0.1-0.15.

`along_path_terrain_sampler` draws uniformly from
`["slippery", "icy", "directional_x", "directional_y"]`, so **half of every patch
set is at or below tyre-on-ice friction**. That is an adversarial worst case, not
a representative indoor floor — describe it that way in any write-up.

**The deployment target is rubber tyres on university linoleum and tile**, and
the advisor specifically wants a comparison on **ice and sand**. So the wanted
change (not yet made, because it would invalidate comparison with everything
measured so far):

- add a realistic `linoleum` / `wet_tile` (mu ~= 0.35) and make it the common case;
- add a `sand` profile (high mu but low shear strength — needs thought, since a
  Coulomb mu alone does not model granular flow);
- re-weight the sampler so `icy` is the rare adversarial case, not 25% of draws.

Also still unanswered and cheap: `slip1`/`slip2` sit in a `<friction><ode>` block
while gz-sim runs **DARTSIM**, which likely ignores ODE's force-dependent-slip
parameters — meaning `mu` may be the only knob ever connected. The `icy_noslip`
profile exists solely to test this: drive across `icy` and `icy_noslip` in one
run, and identical behaviour proves `slip1/slip2` are decorative.

### Ideas queued, roughly in order of expected value

1. **Fix SAC's entropy runaway before any retrain.** `ent_coef` reached 3.31.
   Either pin it (`ent_coef=0.05` instead of `"auto"`) or set an explicit
   `target_entropy` — the default `-dim(A)` is far too permissive for a 4-D
   residual whose useful range is tiny. This is the single highest-value change:
   every hour of the 20260730 run after ~800k steps made the policy worse.
2. **Bound the per-episode return.** Huber bounded the reward's slope, not the
   accumulated return over 200 non-terminating steps, and `critic_loss` still
   reached 1.2e4. Options: normalize the return, cap per-step cost outright, or
   reinstate termination with a large-but-finite terminal penalty (which is not
   the same as the 0.5 m corridor that caused the original no-recovery problem).
3. **Re-measure everything on a genuine S-curve** (`floor_6_00028`) now the
   gallery shows the current one is not. Cheap; changes what "TVLQR oscillates
   on curved plans" is even claiming.
4. ~~Explain the run-to-run variance properly.~~ **SOLVED 2026-08-02, see
   "The run-to-run variance was an 8 mm reset error amplified by patch edges".**
5. **Widen the tuning search to the full Q/R diagonal** (`q_along`, `q_heading`,
   `r_v`) once the 2-D search proves the machinery. Nelder-Mead handles 5-D, but
   the evaluation budget grows and the variance in (4) sets the noise floor on
   what can be resolved.
6. **Give the RL residual a fair fight**: train it *on top of* tuned TVLQR rather
   than on top of identity, so the policy learns the residual that a good linear
   controller cannot supply, instead of re-deriving feedback from scratch. This
   is also the version most defensible in a write-up — the advisor's requirement
   is that RL be part of the system, not that it beat everything alone.
7. **Parallel sims via namespacing** (agreed 2026-08-02, after the corrector
   work). Two env vars are all it takes: `GZ_PARTITION` isolates Gazebo
   transport — a partition collision is exactly the "both spawn a scout_mini and
   the robot disintegrates" failure above — and `ROS_DOMAIN_ID` isolates DDS. No
   launch-file surgery, no topic remapping. **Unset must keep today's behaviour
   exactly**, so make the parallel path opt-in via an explicit `WORKER` variable.
   Payoff is twofold: RL training is ~linear in workers (SAC is off-policy and
   sample-hungry), and repeated measurements — now known to be *mandatory*, see
   the variance section — are embarrassingly parallel, so repeats become nearly
   free instead of a 5x slowdown. The VM is CPU-bound here, not GPU-bound (a
   ~30-D obs, 4-D action MLP barely touches the V100, and `TORCH_THREADS` is 1),
   and cores/RAM on the VM are adjustable while the single GPU is not — so size
   it as `cores ~= workers + 2`, watching RAM since each Gazebo loads the world
   meshes independently. The one thing that must be got right first:
   `just check-sim` currently refuses to launch if *any* Gazebo lives, and it has
   to become per-partition without losing its teeth.
8. **`floor_1_00050` is a degenerate PMP plan** (`max_turn = 3.14 rad/step` over
   6 m). Still unexplained, still excluded, still a planner bug rather than a
   control one.

### Measurement facts worth not rediscovering

- `compare_correctors` builds the bridge with `deterministic=True` — the world
  is paused and multi-stepped, so results do **not** depend on CPU load, and a
  `gz sim -g` viewer cannot perturb them. **This was false until 2026-08-02
  evening** (the pause was being cleared by every step request, see above), so
  results measured before that fix DID depend on CPU load. (`make rl-sim` itself is headless
  server-only, so there is no 3D view unless a GUI client is attached.)
- The identity and TVLQR baselines are checkpoint-independent. A checkpoint
  sweep that re-measures them per checkpoint spends two-thirds of its runtime
  re-deriving the same two numbers; measure them once and run `--correctors rl`
  for the rest. 15 checkpoints ≈ 35 min, ~20-25 s per episode.
- The VM's desktop can be screenshotted without any GUI interaction:
  `ssh <host> 'DISPLAY=:0 import -window root /tmp/x.png'` (ImageMagick, already
  installed; 1920x1080 virtual framebuffer). This is how the final training
  stats block above was found — it was on screen but not in the log tail.
- `tools/plot_checkpoint_paths.py` draws the checkpoint *paths* in a colour
  ramp, next to `tools/plot_checkpoints.py` which reduces each to one scalar.
  Past ~8 overlaid paths the lines stop being individually traceable.

## Conventions

- Non-obvious design decisions are documented in long module docstrings (the PMP
  planner, the runtime corrector, `RLCorrectorConfig`). When changing behaviour
  there, update the docstring in the same edit — they are treated as the spec.
- Node parameters are loaded from dataclasses via
  `agx_planning.utils.declare_and_load_dataclass`; add a field to the dataclass
  rather than a bare `declare_parameter`.
- `acados/` at the repo root is untracked scratch; the Makefile's `ACADOS_*` /
  `t_renderer` bits are vestigial and unset by default.
