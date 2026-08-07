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
PYTHONPATH=src/agx_navigation/agx_planning:src/rudn-ordjo-building python3 -m pytest \
  src/agx_navigation/agx_planning/test/unit/test_rl_reward.py::test_name -v
```

The submodule on the path is needed by `test_terrain_weights.py`, which imports
`rudn_ordjo_building.surface_patches` — the friction profiles are defined there,
not in `agx_planning`. Without it pytest fails at *collection*, so every test in
the run reports as an error rather than just that file.

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

### Reaching the VM: never choose a route by hand

There are two routes to the **same** machine — direct over the lab VPN
(`172.26.13.37`) and via a jump host (`llm_test2@kron.botik.ru` → `192.168.71.113`,
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

**`max|e_cross|` — the tuner's objective — is reproducible to four decimals ON
THIS TRAJECTORY.** That was read as "single-sample ranking is finally
legitimate", and **that generalisation is wrong** — see "The 0.0002 m noise floor
does not transfer" below. `floor_6_00042` has since been dropped from the eval
set, so this figure now describes a trajectory nothing is measured on.

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

### The 0.0002 m noise floor does not transfer (found 2026-08-03)

The overnight tuning run on the **7-trajectory** set (138 evals, 1.3 h, results in
`tune_data/`) reported `q_cross=9.996 / r_omega=1.252` → **0.9412 m** from a
1.1405 m baseline. **Do not adopt it.** Reading the full JSONL rather than the
reported optimum:

- the simplex **stopped moving at eval 49** and then re-evaluated ONE gain pair
  **71 times** (68 distinct pairs over 131 valid evals; the rest are that point);
- those 71 repeats — identical gains, trajectories and seed — span
  **0.9412-1.3052 m**, sd **0.0886**, mean **1.0468**.

So the reported best is the **minimum of 71 noisy draws**, biased low by
winner's curse, and the claimed 0.199 m improvement is smaller than the 0.364 m
spread it was selected from. Nelder-Mead cannot converge on a noisy objective —
it shrinks and re-samples forever, which is exactly what the log shows.

**The root error is the generalisation, not the tuner.** The 0.0002 m floor was
measured on `floor_6_00042` alone — since dropped for being the wrong shape — and
assumed to carry to the seven-shape set that replaced it. It does not: noise
there is ~400x higher, because the harder shapes contain turn reversals and a
turn reversal is the chaotic amplifier already documented above.

Per-trajectory sd across the run, which is where the noise actually lives:

| trajectory | shape | sd | spread |
| --- | --- | --- | --- |
| floor_6_00018 | S | 0.026 | 0.33 |
| floor_6_00031 | U-TURN | 0.148 | 1.51 |
| floor_6_00023 | CORNER | 0.187 | 0.89 |
| floor_6_00047 | ZIGZAG | 0.215 | 1.68 |
| floor_6_00025 | LOOP | 0.280 | 1.43 |
| floor_1_00049 | STRAIGHT | 0.501 | 1.50 |
| floor_6_00056 | TIGHT V | 0.582 | 2.80 |

**Consequence: single-sample ranking is not valid on this eval set.** Any tuning
result needs repeats and a comparison of distributions. This raises the priority
of parallel sims (queue item 7) — repeats are now mandatory and embarrassingly
parallel. `tools/plot_tune_variance.py` draws this; `figures_new/`.

### Baseline on the repaired plant (2026-08-07) — the current reference

**This supersedes the 2026-08-03 table below for every purpose except history.**
That one was measured with the wheel's `mu2` at 0.7, i.e. on a robot that lost
steering on its own test terrain (see "The wheel fix"). Five repeats of the full
seven-shape set, identity and TVLQR, terrain on, `just compare` (~82 s per
repeat). max|e_cross| in metres, mean ± sd over 5:

| trajectory | shape | identity | tvlqr | verdict |
| --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | 0.132 ± 0.000 | **0.086 ± 0.028** | tvlqr, marginal |
| floor_6_00023 | CORNER | 0.721 ± 0.196 | **0.226 ± 0.000** | tvlqr |
| floor_6_00018 | S | **0.839 ± 0.031** | 2.128 ± 0.003 | **identity** |
| floor_6_00047 | ZIGZAG | 7.063 ± 0.712 | **2.186 ± 0.276** | tvlqr |
| floor_6_00056 | TIGHT V | 0.418 ± 0.014 | **0.253 ± 0.026** | tvlqr |
| floor_6_00031 | U-TURN | 1.666 ± 0.771 | 1.411 ± 0.534 | **tie**, overlapping |
| floor_6_00025 | LOOP | 4.053 ± 0.010 | **1.600 ± 0.035** | tvlqr |
| **mean** | | **2.127** | **1.127** | |

**TVLQR halves worst-case deviation** (2.13 → 1.13 m), winning 5 shapes, tying 1,
losing 1. Open loop is what moved most in the re-baseline (3.53 → 2.13): a plant
that can steer makes the nominal plan far more followable unaided, so the easy
shapes stopped needing rescue (tight V 3.01 → 0.42, U-turn 5.53 → 1.67). TVLQR
itself barely moved (1.20 → 1.13), which is the expected signature.

**The noise REDISTRIBUTED, it did not shrink.** The old plant's worst trajectories
were the straight (sd 0.501) and the tight V (0.582); both are now effectively
deterministic (0.000 / 0.014), while the zigzag (0.712) and U-turn (0.771) became
the noisy ones. Coherent: when everything slides, everything is noisy; now the
chaos is confined to the reversal-heavy shapes, where contact modes switch.
**Consequence: 5 of 7 trajectories now support single-sample ranking, but the
zigzag and U-turn do not.** Keep mean-of-3 as the tuning default.

**"TVLQR oscillates on S-curves" is REINSTATED**, and the 2026-08-03 retraction of
it is itself retracted. TVLQR loses the genuine S by 2.5x with sd **0.003** — as
reproducible as anything measured here. The retraction came from the old plant,
where TVLQR scored 1.06 against identity's 4.22; that was open loop failing
everywhere, masking a real TVLQR weakness. The original claim had the phenomenon
right and the evidence wrong.

**The U-turn is a TIE, not a TVLQR win.** The distributions overlap completely.
Any earlier U-turn claim from single samples was reading noise.

RL was deliberately not re-measured: the existing policy is a 4-wheel *residual*
trained on the old plant with an IMU-bearing observation layout, so its number
here would describe neither this plant nor the re-planner architecture that
replaced it (see handover).

### Clean three-way comparison (2026-08-03) — measured on the BROKEN plant

**Historical only** — superseded by the table above; `mu2=0.7` meant the robot
could not steer on any of its own patch profiles.

Seven shapes, fixed bridge, terrain on, RL at the 800k checkpoint.
`tools/plot_corrector_summary.py`, data in `compare_data_new/`
(max|e_cross| / final_err, m):

| trajectory | shape | identity | tvlqr | rl (800k) |
| --- | --- | --- | --- | --- |
| floor_1_00049 | STRAIGHT | **0.11**/0.47 | 0.38/0.21 | 0.65/0.68 |
| floor_6_00023 | CORNER | 1.19/3.49 | 1.36/3.14 | **1.16**/2.63 |
| floor_6_00018 | S-CURVE | 4.22/5.42 | **1.06**/0.99 | 1.47/1.60 |
| floor_6_00047 | ZIGZAG | 5.16/5.38 | **1.41**/1.12 | 2.87/2.87 |
| floor_6_00056 | TIGHT V | 3.01/3.01 | **1.15**/1.22 | 3.69/4.61 |
| floor_6_00031 | U-TURN | 5.53/6.30 | 1.40/0.25 | **1.39**/1.43 |
| floor_6_00025 | LOOP | 5.47/5.56 | **1.63**/0.78 | 3.33/3.34 |
| **mean** | | **3.53** | **1.20** | **2.08** |

**TVLQR at the DEFAULT gains cuts worst-case deviation by 66% over open loop**,
and wins 5 of 7 shapes. Two retractions follow:

- ~~**"TVLQR oscillates on S-curves" is dead.**~~ **This retraction was itself
  wrong** — see the 2026-08-07 baseline above. TVLQR did win the S here (1.06 vs
  identity's 4.22), but only because open loop was failing everywhere on a plant
  that could not steer. On the repaired plant it loses the S 2.13 to 0.84, at
  sd 0.003. The oscillation claim is reinstated.
- **"No RL checkpoint beats identity on anything" is dead** — but only just, and
  only at the best checkpoints. RL(800k) beats identity on 6 of 7 shapes here.
  **That checkpoint is not representative**; see the clean sweep below.

### The clean checkpoint sweep kills the "RL was learning" reading (2026-08-03)

20 checkpoints of `runs_20260730` (stride 15, every 75k steps) re-measured on the
repaired bridge, `--correctors rl` only since identity/TVLQR are
checkpoint-independent. `tools/plot_checkpoints_clean.py`, data in `sweep_clean/`.

| | value |
| --- | --- |
| mean over 20 checkpoints | **3.475 m** (identity is 3.527) |
| sd across checkpoints | 0.716 |
| range | 2.030 (@980k) - 4.579 (@305k) |
| Pearson r vs training step | **0.111 — no trend** |
| beat identity (3.527) | **8 of 20** |
| beat TVLQR (1.198) | **0 of 20** |

**There is no learning trend on the real task**, and the typical checkpoint is
level with open loop. The 800k checkpoint used in the comparison table above
(2.08 m) sits in one of only three good pockets (755k/830k/980k) — it was picked
because the *old contaminated* sweep called it best, so quoting it as "the RL
result" is the same winner's-curse error as the tuning run. Quote it as **best
checkpoint**, never as typical.

**The checkpoint-to-checkpoint swing is real, not measurement noise.** Adjacent
checkpoints 75k steps apart differ by ~2 m, against a measurement sd of ~0.09 m
on this eval set. So the policy genuinely lurches between saves — exactly what
`ent_coef=3.31` predicts, since a near-random policy makes every save a different
random draw.

**Retract the TB reading.** `rollout/terminal_abs_e_cross` falls 3.9 → 1.6 m
across training, which looks like the policy learning the task while the
optimiser diverged. It is not: that metric was logged **by the mis-stepped
environment**, so it reports progress at a task that was not the task. The clean
sweep is the out-of-band check and it shows no trend. The optimiser panels
(`ent_coef`, `critic_loss`) remain valid — they describe SAC, not the plant.
`tools/plot_training_diagnostics.py` now says so on the figure itself.

### The wheel-velocity residual was NOT the seed (2026-08-04)

The handover's "do this first" experiment is done and the answer is **no**. Two
arms on `floor_6_00056` (TIGHT V, the worst offender), 5 rollouts each,
everything else fixed: A = today's reset, B = `--reset-world` (a full gz
`WorldControl.reset.all`, the only mechanism that zeroes JOINT velocities).
`tools/run_reset_world_probe.sh`, `GazeboBridge(reset_world=True)`,
`variance_probe --reset-world`.

**The premise was wrong on its own terms.** The wheels do not settle to ~1e-9
rad/s — `trace_diff --eps 0` shows them already agreeing to **1e-16..1e-19**
between rollouts in the BASELINE arm, i.e. to the last bits of a double. There
was no 1e-9 wheel-speed seed to remove. (The 1e-9 figure came from a single
absolute reading, not from a difference between two rollouts.)

**What actually differs at t=0, in both arms, is the IMU** — `imu_ax`, `imu_ay`,
`imu_gz` differ by **0.01-0.04**, which is twelve to fifteen orders of magnitude
above every other column (pose 1e-12, quaternion 1e-13, wheels 1e-17). That is
not physics: the IMU arrives as an async ROS message and `_read_state` takes
whatever the latest one is, so which physics tick it was sampled on depends on
ROS timing. It is the **stale-pose readout bug of 2026-08-02 again, in the IMU
channel** — and the pose channel got a `_wait_pose_advance` gate that the IMU
never got.

This matters unevenly, so do not over-read it: TVLQR does not consume the IMU,
so it cannot be *this* that moves TVLQR's commands. But `use_imu` is in the RL
observation layout, so **every RL measurement ever taken has had a
timing-dependent 0.04 jitter injected straight into the policy input.** That is
a live candidate for RL's unexplained measurement noise, which the handover
notes has never been measured.

World reset did buy about three decades of initial-state agreement (pose
mismatch 1e-9 → 1e-12) and the two comparable rollouts diverged visibly later
(xy separation reaching 1e-3 m at step 111 vs step 83; final separation 0.39 m
vs 3.41 m). **Three decades is not enough** — chaotic amplification at the turn
reversal spends them in ~30 steps. Bit-identity is the only thing that would
have worked, and a world reset does not deliver it.

**`reset_world=True` DESTROYS THE ROBOT — do not use it.** A gz
`WorldControl.reset.all` drops every entity spawned at runtime, and the
`scout_mini` is spawned at runtime by the launch. So the reset deleted the robot
out from under the running sim: from the third episode on `set_pose` stopped
landing (6.92 m from target, `reset_ticks` at the full 400, `v0=+0.25`) and
rollouts 2-4 scored an identical 5.5773 — a reproducible *broken* mode, not a
measurement. Some time later the ROS side collapsed entirely, leaving an
orphaned `gz sim` with only `/rosout` and `/parameter_events` alive, and
`_wait_ready` failing on all three streams at once. Only the first two rollouts
of arm B are valid data, and the fix after using it is `just kill-sim` +
`just remote-sim`, not debugging the bridge. The flag is left off by default and
should probably be deleted; it is kept only so this note has something to point
at.

**Consequences.** Item 4 stays open but the wheel-velocity hypothesis is closed.
Single-sample ranking on the hard shapes remains illegitimate, so **repeats stay
mandatory and parallel sims (queue item 7) are now unblocked and top of the
queue** — that was the handover's stated "if it does not work" branch. Do not
re-run the tuner before that lands.

Next lead, cheap and worth doing before anything expensive: gate the IMU read
the way the pose read is gated, then re-run this probe. It will not make
rollouts bit-identical (the pose/quaternion mismatch at 1e-12 survives), but it
removes the one initial-state difference that is 12 orders of magnitude larger
than the rest, and it is the only known contaminant of the RL observation.

### Realistic friction, IMU gating, and repeats (2026-08-04)

Three changes landed together. **They re-baseline everything: no number measured
before 2026-08-04 is comparable with one measured after**, because the plant
(friction distribution) changed deliberately.

**1. The patch friction distribution is now a floor, not an ice rink.** Added
`linoleum` (mu 0.45, the actual deployment surface) and `wet_tile` (0.30) to
`PROFILES`, plus `DEFAULT_PATCH_WEIGHTS` — the samplers no longer draw
uniformly. Was: uniform over `[slippery, icy, directional_x, directional_y]`, so
**half of every patch set was at or below tyre-on-ice friction** and a quarter of
episodes ran on black ice end to end. Now ~60% at mu >= 0.30 and 10% ice, so the
hard surfaces are the exception they are in reality. `ground_friction_sampler`
is weighted too and **excludes the directional profiles** — a whole floor that
grips one axis and slides the other has no physical analogue. Rationale: gains
tuned against black ice are over-aggressive on linoleum, and the deployment
target is rubber on university linoleum and tile. Tests in
`test_terrain_weights.py`; a profile with no weight raises rather than silently
never being drawn. Still to do: a `sand` profile (the advisor wants ice *and*
sand, and Coulomb mu alone does not model granular flow).

**The world's own ground is still `mu=1.0`** (concrete-like) in
`rl_corrector.world`. If deployment is linoleum everywhere, the nominal
no-patch plant is grippier than reality and every gain inherits that bias. Not
changed yet — it is a bigger re-baseline and wants a decision.

**2. The IMU read is gated** (`_wait_imu_advance`), matching the pose gate that
has existed since 2026-08-02. Motivation is in "The wheel-velocity residual was
NOT the seed" above: the un-gated IMU was the largest difference between two
otherwise-identical rollouts, by twelve orders of magnitude, and it feeds the RL
observation. `stale_imu_steps` counts giving up, and the gate is only paid when
`use_imu` is on, so TVLQR tuning costs nothing for it. **The goal is not
determinism** — the real robot's IMU is noisy and laggy and the policy must
tolerate that. The goal is that the sim's sensor error be a knob we choose
(inject a deliberate latency/noise model in the env, matched to the measured
real IMU) rather than an artifact of VM CPU load with no real-world counterpart.
The deliberate sensor model is **not written yet**; when it is, it must live in
the bridge/env and leave the obs layout untouched, or the policy stops being
deployable.

**3. The tuner averages repeats.** `tune_tvlqr --repeats 3` (default) drives the
whole trajectory set n times per candidate and reduces per trajectory via
`objective.reduce_repeats`. Measured on the 26 real repeats in the old log:

| estimator | sd of the estimate | cost/eval |
| --- | --- | --- |
| single sample | 0.0933 | 35 s |
| median-of-3 | 0.0692 | 105 s |
| **mean-of-3** | **0.0547** | 105 s |
| mean-of-5 | 0.0413 | 175 s |

**The median LOSES, which was not the prediction.** The aggregate already
averages over 7 trajectories, of which at most two are contaminated per repeat,
so the outlier is diluted 7-fold before the estimator sees it while the median
pays its variance penalty in full. It is plain sqrt(n) averaging. Use mean-of-3
to search, mean-of-5 to validate a winner. The cache key now includes
`repeats`, `reduce` and `patch_weights`, so an old cache cannot be replayed into
a run whose numbers mean something different.

Still outstanding for the tuner: Nelder-Mead reports the **minimum observed
draw**, which is the winner's-curse machine that produced both bad results. With
sd 0.055 that bias is much smaller, but the structural fix is a noise-aware
optimizer (Bayesian optimization with a nugget term, reporting the posterior-mean
optimum). Not yet done.

**Parallel sims are NOT a prerequisite** any more, contrary to the 2026-08-03
handover: mean-of-3 is ~105 s/eval serially, so a converged run is ~2 h. They
remain the right answer for RL *training* throughput.

**RL action space, decided 2026-08-04:** keep the 4-wheel residual. The physical
Scout takes only `(v, omega)` and computes wheel efforts in firmware, so a 4-D
residual cannot deploy as-is — that is accepted as an implementation detail that
could be changed (firmware), and **TVLQR is what any real-world demo runs**.

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

### `slip_chi` is a function of the SURFACE, and that is the live bug (2026-08-05)

Dropping the world ground from `mu=1.0` to `0.45` made the default-gain tuning
objective jump from 1.1706 m to **20.6148 m**. A no-patch diagnostic
(`/tmp/mu_diag`, bare linoleum, zero terrain) shows it is **not** a corrector
failure and **not** the patch distribution:

| trajectory | identity | tvlqr |
| --- | --- | --- |
| floor_1_00049 (straight) | 0.53 / 0.54 | 0.59 / 0.86 |
| floor_6_00023 (corner) | **27.25 / 27.95** | **30.31 / 30.92** |

Open loop diverges as badly as TVLQR, so the **nominal PMP plan is unfollowable
on a realistic floor**. The straight is fine and the corner is not, which points
straight at the yaw model.

**Measured cause (same day, `slip_ident` on the real-time world):** it is not a
mis-tuned chi, it is that **an isotropic low-friction ground deletes the tyre
model's anisotropy, which IS the steering mechanism.**

`tools/sweep_ground_mu.sh` measures chi against ground `mu`
(`sweep_data/ground_mu_chi.csv`, logs per point):

| ground `mu` | chi | yaw gain | usable arcs | spread across radii |
| --- | --- | --- | --- | --- |
| 1.0 | **1.3718** | 0.729 | 6 | 0.030 |
| 0.9 | 1.3879 | 0.721 | 6 | 0.051 |
| 0.8 | 1.4438 | 0.694 | 6 | 0.129 |
| **0.7** | **10.147** | **0.100** | 2 | 2.672 |
| 0.6 | 11.760 | 0.086 | 2 | 2.547 |
| 0.5 | 14.441 | 0.071 | 2 | 4.101 |
| 0.45 | **16.478** | 0.061 | 2 | 3.619 |

(This whole section describes the plant as it was, with the wheel's `mu2` at 0.7.
That was changed to 0.45 on 2026-08-07 and the cliff moved with it — see "The
wheel fix" below. The mechanism and the method are unchanged; only the number
where it breaks moved.)

**The cliff is at 0.7 — the wheel's own `mu2`, to the digit.** Above it only the
longitudinal channel erodes, so chi drifts up gently while the spread across
radii quadruples (a single `slip_chi` is already losing validity at 0.8, before
anything looks broken). At 0.7 both coefficients meet and yaw gain falls
seven-fold in one step of 0.1.

The 1.0 row reproduces `PlannerConfig.slip_chi = 1.373` to four figures with an
0.028 spread across radii, so the measurement chain is sound and the two rows are
comparable. At 0.45 the robot achieves **5% of commanded yaw rate** — it barely
rotates at all, which is why seven of eight arcs were rejected as unmeasurable.

Why: [wheel.xacro](src/scout_ros2/scout_description/urdf/wheel.xacro#L63) gives
each wheel `mu1=200.0` (rolling) and `mu2=0.7` (lateral), and Gazebo combines two
contacting surfaces by taking the **smaller** coefficient. So `mu1=200` is never
realized — it encodes "the wheel is never the longitudinal limit, the ground
decides", which is what its comment says. Against a ground of 1.0 the effective
pair is (1.0 rolling, 0.7 lateral): a **1.43:1** ratio, a perfectly physical
number. (An earlier version of this note called it "a 285:1 anisotropy". That was
wrong — 285:1 is the nominal wheel pair, which the ground caps away.)

That 1.43:1 is the steering mechanism, since a skid-steer yaws by gripping
longitudinally while scrubbing sideways. It survives only while
`ground > wheel mu2`: there the ground binds longitudinally and the wheel binds
laterally, two independent constraints. Below 0.7 the ground binds **both**, so
lowering it reduces grip *and* collapses the ratio to 1:1 — inseparably. The
sweep shows exactly that shape.

Provenance: Grigorii Matiukhin, 2026-02-13, in the team's own `scout_ros2` fork —
not AgileX upstream, so it is ours to change.

So "make the floor realistic" cannot be done by lowering an isotropic ground
plane. The anisotropy that matters is in the **wheel** frame (rolling vs lateral)
and only the wheel can express it; a ground `fdir1` is world-fixed, which is
exactly why the `directional_x/y` patch profiles are unphysical. And per-zone
friction has to live in the ground, because patches are ground entities. That
tension is the thing to solve before any re-baselining.

**Every profile in `surface_patches.py` is below 0.7** — linoleum 0.45, wet_tile
0.30, slippery 0.20, icy 0.05 — so every slip patch ever driven over has been
simulating "loses steering", not "slides". That includes the RL training terrain
and every corrector comparison. The friction values were not merely uncalibrated
(as the section above says); they were outside the model's valid domain.

### The wheel fix: `mu2` 0.7 → 0.45, applied and confirmed (2026-08-07)

`mu2=0.7` was the problem — it sat at the top of the realistic range for rubber,
so every physically plausible floor landed at or below it. It is now **0.45** in
`wheel.xacro`. **`mu1` deliberately stays at 200**: it is never realized (the
ground caps it), and it encodes "the wheel is never the longitudinal limit, the
GROUND decides", which is exactly what makes a friction patch's `mu` mean
anything in the rolling direction. Only `mu2` sets the knee, so only `mu2` moved.

Re-run of `sweep_ground_mu.sh` (`sweep_data/ground_mu_chi_mu2_045.csv`), against
the old curve:

| ground `mu` | chi @ `mu2=0.7` | chi @ `mu2=0.45` | yaw gain | arcs | spread |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 1.3718 | **1.3575** | 0.737 | 6 | 0.007 |
| 0.8 | 1.4438 | **1.3651** | 0.733 | 6 | 0.023 |
| 0.6 | 11.760 | **1.4037** | 0.713 | 6 | 0.058 |
| 0.5 | 14.441 | **1.5713** | 0.641 | 6 | 0.341 |
| **0.45** | 16.478 | **15.583** | 0.065 | 2 | 3.449 |
| 0.4 | — | 18.729 | 0.055 | 2 | 5.733 |
| 0.3 | — | 25.362 | 0.040 | 2 | 6.084 |

**The knee moved to 0.45, to the digit** — predicted before the run, exactly as
the 0.7 knee was. Min-combination is settled, not a hypothesis.

**The curve TRANSLATED rather than deformed, and the free variable is the RATIO
`ground/mu2`, not absolute friction.** 0.5/0.45 (ratio 1.11) gives chi 1.57;
the old 0.8/0.7 (ratio 1.14) gave 1.44. So `sweep_ground_mu.sh` measures one
curve in one variable and `mu2` slides it — that is what makes it a reusable
instrument for "does this tyre model steer" rather than a one-off.

**Chi at nominal barely moved: 1.3718 → 1.3575, ~1%.** The prediction was a
visible drop from the improved ratio (1.43:1 → 2.22:1) and that was **wrong** —
above the knee chi cares *that* you are above it, not by how much. Consequence:
`PlannerConfig.slip_chi = 1.373` is still within ~1% at ground 1.0, so the
existing baked plans did **not** need re-planning for this change. The spread
across radii also improved (0.0299 → 0.0072), so a single `slip_chi` describes
this plant *better* than it described the old one.

**Usable band is ground >= 0.5.** Linoleum at 0.45 sits exactly on the knee and
is marginal; wet_tile (0.30), slippery (0.20) and icy (0.05) are all still below
it. Ice being uncontrollable is **correct** rather than a limitation — on real
ice `mu_long ~= mu_lat` and a real skid-steer genuinely cannot steer. The model
was never broken for ice; it was broken for linoleum, because the wheel was
parameterised for concrete.

**The limit that survives the fix, unchanged:** under min-combination with an
isotropic ground, *any* ground below the wheel's `mu2` gives ratio 1. "Slides but
still steers" is inexpressible at low friction — a surface is either above the
knee or it has no steering authority. **`sand` is still blocked on this**, at any
wheel setting; it needs a different mechanism, and that is a question for the
advisor alongside the planner-vs-corrector one.

**What still stands:** chi is genuinely a property of the SURFACE (1.36 vs 25.4
across this curve), so it cannot be a constant on a floor with ice/sand zones —
it is not constant *within one trajectory*. That remains a modelling gap rather
than a tuning problem, and the structural reason a frozen PMP plan cannot handle
zones.

**Do not lower the ground plane to model a slippery floor.** Both worlds are at
`mu=1.0` and should stay there; slipperiness belongs in the wheel pair or in a
patch, and any patch below 0.45 now means "no steering", deliberately.

**Everything measured before this change was measured on a different plant.**
The identity / TVLQR / RL comparison and every tuning result predate it.

### Measuring `slip_chi` on the real robot (method, 2026-08-05)

**Yes, chi is measurable by driving the real robot, and this is the intended use
of `slip_ident`.** It references the **gyro**, which owes nothing to the wheels,
so nothing about the method is sim-specific. `calibrator.py` cannot substitute:
it compares commands against `/odom`, and both sides share the missing slip term.

Two things to get right on hardware:

- **`cmd_mode:=wheels` is sim-only.** The physical Scout takes only `(v, omega)`
  and computes wheel efforts in firmware, so on the robot you run
  `cmd_mode:=twist`, which yields `chassis_gain_omega` — chi folded together with
  the firmware's own twist→wheel conversion. That composite is the right quantity
  on hardware, because the wheel-level command is not reachable anyway. The tool
  prints which of the two it measured; do not paste a twist-mode number into
  `PlannerConfig.slip_chi`.
- Drive **arcs at several radii**, both directions, on the surface in question.
  A spin scrubs all four contact patches and is strongly load-dependent —
  `slip_ident` reports spins separately and says not to fit chi to them.

**This closes the sim-to-real loop on the one parameter the planner consumes.**
The patch friction values have been unvalidated for weeks (see above) and
measuring `mu` directly is awkward; measuring **chi** is not. So: drive the real
robot on linoleum, on the ice mock-up and on sand, get chi per surface, then tune
each sim profile's `mu` until the *sim's* chi matches the measured one. Chi, not
mu, is the quantity to match — it is what the model actually uses.

**`slip_ident` cannot run against an unthrottled sim (found 2026-08-05).**
`make rl-sim` runs uncapped (~33x), which puts `/imu/data` at **3295 Hz**; with
`depth=10` on a single-threaded executor the node drops almost every sample and
the integrated gyro yaw comes out ~800x too small. It fails loudly (`gyro
measured only +0.0034 rad`, then `No usable arcs`) rather than reporting a wrong
chi, which is the correct behaviour, but it means **chi must be measured against a
real-time-throttled world**. Also note the node's `imu_topic` default is `/imu`
while the actual topic is `/imu/data` — its own usage line has it right, the
default does not.

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
3. ~~Re-measure everything on a genuine S-curve.~~ **DONE 2026-08-02:** the eval
   set is now **seven shape-distinct plans** (`config/eval_trajectories.yaml`) —
   straight, corner, S, zigzag, tight V, U-turn, loop. It grew from three
   because a rollout now costs ~5 s instead of ~20-25 s (the world was
   free-running between steps). `floor_6_00042`, the "S-curve" that was really
   an L, is dropped. Still to do: actually re-measure on it.
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
8. **Generate interesting trajectories by construction, instead of hoping for
   them** (user's idea, 2026-08-02). The 100 recorded plans came from random
   goals, and it shows: ~20 have any shape at all, all on page 1 of the gallery,
   and pages 3-5 are straight lines. So the evaluation set is capped by what the
   library happens to contain. Proposed instead: sample start/goal pairs from the
   baked map, run a cheap A* (or the FM2 field itself) to *predict* the route
   without paying for a PMP solve, score each route for tortuosity — turning per
   metre, number of curvature sign changes, reversals — and keep only the
   high-scoring pairs to run through the real planner. Screening is cheap and the
   PMP solve is the expensive part, so this inverts the current ratio. Two things
   to get right: the A* route must be a fair proxy for what PMP actually produces
   (worth validating on the existing 100 before trusting it), and the score must
   ignore the leading in-place pivot for the same reason `trim_pivot` does.
9. **`floor_1_00050` is a degenerate PMP plan** (`max_turn = 3.14 rad/step` over
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
