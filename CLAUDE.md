# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**This file is loaded into every session, so it is kept short on purpose.** It
holds orientation and the claims that still *constrain a decision*. Detail lives
in [docs/](docs/) — see "Where everything else is" below. When you learn
something new, decide which file it belongs in rather than defaulting to this one.

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

## Where everything else is

| file | what it holds | lifetime |
| --- | --- | --- |
| **[handover.md](handover.md)** | **what is running right now**, what is half-finished, what to do first | rewritten each session |
| this file | orientation + claims that still constrain a decision | cumulative, dated |
| [docs/measurement-rig.md](docs/measurement-rig.md) | plant (friction, `slip_chi`), eval trajectories, tuning machinery, instrumentation | current reference |
| [docs/vm-operations.md](docs/vm-operations.md) | VM routing, `WORKER` parallel sims, the job queue | current reference |
| [docs/corrector-design.md](docs/corrector-design.md) | the re-join re-planner design | current reference |
| [docs/svcm-source.md](docs/svcm-source.md) | full transcription of the advisor's source framework | reference |
| [docs/corrector-history.md](docs/corrector-history.md) | superseded tables, resolved-bug narratives | archive |

**Keep `handover.md` current — it outranks anyone's memory of what we are doing.**
Sessions here are days apart and the user explicitly relies on that file rather
than recall, so a session that learns something and does not write it down has
lost it. Update it **in the same session that produced the change**; if an
experiment is still in flight at the end of a session, `handover.md` must say so,
say where its log is, and say what to conclude from either outcome.

**Split rule.** The axis is **"does this still constrain a decision?"**, not
chronology — a retraction that stops someone re-proposing a dead idea is *live*
however old it is, and gets a one-line stub under "Settled" below. What moves to
`docs/corrector-history.md` is the investigation behind it and any measurement
taken on a plant or rig that no longer exists. **Always leave the stub**: an
unstubbed removal means the idea comes back. Date each claim, and delete a claim
outright when it is superseded rather than leaving both versions.

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

Analysis nodes that only make sense against the fixture:

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
and cover the *pure* modules (RL corrector `coeff`/`obs`/`reward`/`nominal`/`env`,
and `tuning/`) — no ROS, no Gazebo, no torch import path. Keeping those modules
ROS-free is deliberate; don't add `rclpy` imports to them.

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
  which is the seam where a residual applies. `JointGroupVelocityController`
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
for why `none` is the wrong default for it. It exists because rtabmap is a bad
fixture for testing the *corrector*: it needs the robot to drive before any map
exists (so no plan from a standing start), it intermittently never initialises on
this world's untextured walls, and it yields a slightly different map each run —
so two corrector runs are never comparable.

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
the missing slip term. Method and the measured surface dependence:
[docs/measurement-rig.md](docs/measurement-rig.md).

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

```bash
make rl-deps                          # stable-baselines3[extra] + torch + gymnasium
make rl-sim HEADLESS=true             # terminal 1: minimal sim (no nav/planner/corrector)
make rl-train TIMESTEPS=200000        # terminal 2
make p0 | p1 | p2 | p3 | curriculum   # phased curriculum, chained with --load
make rl-kill                          # ALWAYS run this after a Ctrl-C'd/orphaned run
```

`make rl-deps` is *not* the only thing that installs torch, despite the name:
`agx_planning/setup.py` lists `torch` and `stable-baselines3` in `install_requires`,
so plain `make deps` already drags in the whole CUDA wheel stack (~3-4 GB of
`nvidia_*` wheels). Budget for that on a fresh machine or a slow link.

`make rl-sim` already runs **without the rendering sensors** — `SIM_SENSORS`
defaults to `false`, dropping the GPU lidar and RGB/depth cameras, since the
trainer consumes only ground-truth pose, the `/odom` twist and the IMU. A slow
realtime factor during training is physics, not rendering; don't go looking for
sensor overhead to trim. (`make run` is the one that pays for sensors + SLAM.)

## Remote GPU training server

Training and every measurement run on a Proxmox VM (`danya02-gmatiukhin-ros2-gazebo`,
VM 200) with a Tesla V100 passed through. The [Justfile](Justfile) holds the remote
workflow — new commands go there rather than in the Makefile, which stays the
source of truth for building and running.

**Never choose a route by hand.** There are two routes to the same machine (direct
over the lab VPN, and a jump host) and which one works changes several times a day.
`ssh_config` defines one host alias `agx` whose `ProxyCommand`
([tools/agx-route](tools/agx-route)) probes both and prefers direct, so `ssh`,
`scp`, `rsync -e ssh`, `git` and every Justfile recipe are routed identically. A
PreToolUse hook rejects any command that hardcodes an IP or the jump host.

```bash
just <recipe>                  # already routed — nothing to override
ssh -F ssh_config agx          # works straight from a fresh clone
just ssh-setup                 # one-time: then plain `ssh agx`, `scp x agx:` work
just route-check               # which route is live; also busts the cache
tools/agx-run --detach 'make rl-train …'   # long run, returns immediately
tools/agx-run --tail /tmp/x.log            # poll it
```

**Detach long runs and poll a log file.** Interrupting the local ssh does *not*
kill the remote processes, which then fight the next launch — `pgrep -af 'gz[ -]sim'`
before relaunching, or let `just check-sim` do it. `tmux kill-server` does **not**
stop the sim; it orphans the `gz sim` processes, which keep running and publishing.

**Only one sim per partition.** `WORKER=n` (1-9) sets `GZ_PARTITION`/`ROS_DOMAIN_ID`
and is the entire isolation mechanism; two sims in one partition silently break
resets and, if both spawn a robot, make it **physically disintegrate**. If you see
wheels detaching, count the `gz sim` processes before debugging anything else.

Full routing internals, the `WORKER` verification, the job queue (`tools/jobq.sh`,
`just queue-*`) and the VM's non-obvious operational facts:
**[docs/vm-operations.md](docs/vm-operations.md)**.

## Current work: making the corrector work

The single active goal is getting the runtime corrector to hold a frozen PMP
trajectory under slip, and writing it up. **Read [handover.md](handover.md) for
what is running and what to do first.** This section is the cumulative record of
what has been *established* — append to it, or rewrite the parts it contradicts,
whenever a run establishes something new.

### Settled — do not re-propose, do not re-measure

One line each, because a 'we already ruled that out' is the cheapest useful
text in this file and the narrative behind it is not. Full reasoning in
[docs/corrector-history.md](docs/corrector-history.md); these stubs exist so a
bad idea gets stopped before anyone reads it.

**Plant and measurement validity**

- **Nothing measured before 2026-08-02 evening is valid.** `WorldControl.pause` is a proto3 bool, so every `multi_step` silently un-paused the world and it free-ran between control steps. Fixed; `_ensure_paused` verifies the clock stopped.
- **Nothing measured before 2026-08-04 is comparable with anything after**, and again before **2026-08-07**: the patch friction distribution was re-weighted, then the wheel's `mu2` moved 0.7 → 0.45. Both were deliberate plant changes.
- **The old run-to-run variance was patches silently not spawning**, not an 8 mm reset slide (that theory is retracted). With the world paused, `create` returns False while creating the entity anyway. `_wait_entities` now confirms via pose/info; **never re-issue a create for a 'missing' patch** — it is in flight, and re-creating genuinely fails.
- **Terrain patches are inherited across processes**: the bridge sweeps `rl_ground` and `rl_patch_0..7` by name at construction, because the sim outlives any one trainer. Tables measured before 2026-08-01 ran on someone else's leftover terrain.
- **`reset_world=True` DESTROYS THE ROBOT — never use it.** A gz `WorldControl.reset.all` deletes runtime-spawned entities, including the `scout_mini`. Recovery is `just kill-sim` + `just remote-sim`, not debugging.
- **The wheel-velocity residual was NOT the determinism seed** (wheels agree to 1e-16); the un-gated IMU read was, and is now gated.
- **Do not chase the last decades of initial-state agreement.** A ~1e-6 m settle residual is amplified to metres in ~30 steps on reversal-heavy shapes; a tighter tolerance is the wrong instrument against an exponential.

**Metrics and estimators**

- **`final_err` is NOT reproducible and must never be a tuning objective** (sd 0.26 on the trajectory it was measured on). Genuine chaos at turn reversals, not a bug. (It *is* fine as an aggregate over 40 plans — see "Reading a gain result".)
- **The 0.0002 m noise floor was ONE trajectory's and does not generalise.** Nelder-Mead cannot converge on a noisy objective — it re-sampled one point 71 times. Superseded by mean-of-3 repeats.
- **The median-of-3 estimator LOSES to the mean-of-3** — the 7-trajectory aggregate already dilutes an outlier 7-fold, so the median pays its variance penalty for nothing. Mean-of-3 to search, mean-of-5 to validate.
- **Every "run-to-run variance" number written before 2026-08-13 is a mixture width, not a measurement error.** Several shapes are bimodal; the right estimator for them is a mode frequency over ~100 samples, not a mean over 3.
- **`J` needs the GEOMETRIC mean across trajectories.** `J` spans 0.2 to 1043 across a library sweep and one plan can carry 48% of the arithmetic mean. `objective.DEFAULT_HOW` encodes it: arithmetic for `max_cross`, geometric for `j_total`.
- **A per-shape claim must name its baseline** — computing one against sweep neighbours rather than the default produced a retracted attribution.
- **An automatic shape label is a ranking aid, never a claim.** Two labellers have now misled here. **Render the plans before making any per-shape claim.**
- **Controllability along a plan does not predict how hard it is** (2026-09-23, job 105): Gramian energy of the tracking linearisation vs per-plan J of the adopted corrector over the 40 broad plans, rho = +0.09 (p = 0.59), prediction recorded beforehand. Plan difficulty is the disturbance a plan meets, not its controllability; `tools/controllability_gramian.py`.
- **Screening candidate start/goal pairs on PREDICTED turning does not work** — the cheap A*/lattice route predicts *where* a plan goes (r=+0.99 on length) and not *how it turns* (+0.30). Shape is labelled from the solved plan instead.

**Gains and tuning (closed — see below)**

- **A seven-plan gain search cannot resolve the gains, in either currency.** Two of them produced optima that evaporated on 40 independent plans. Do not run another one. Validate on the broad set.
- **`J` is the right objective and a poor discriminator**: it ranks the move off the old default decisively (1.49x, 5/40) and cannot separate anything *inside* the plateau (every within-plateau p is 0.08–0.88).
- **`q` and `r` INTERACT**, so every `r_omega` claim is scoped to the `q_cross` it was measured at. The `r=0.5→1.0` threshold that once justified moving off `r=0.25` is a `q=0.276` phenomenon and is absent at `q=1.5` and `q=2.5`.
- **The U-turn "basin" with near-vertical walls is a property of `floor_6_00031`'s exact plan, not of U-turns** — four other U-turn plans show no `q` dependence at all. Never restate it as a shape claim.
- **RL checkpoints show no learning trend on the real task** (r=0.111 over 20 checkpoints; 0 of 20 beat TVLQR). The TB metric that looked like progress was logged by the mis-stepped env. Quote 800k as *best* checkpoint, never as typical.
- **The 20260730 SAC run (1.5M steps) learned nothing usable** — 2 successes in 8864 episodes, `ent_coef` ran away to 3.31, `critic_loss` 1.2e4. Fix the entropy target and bound the return before any retrain (queue 2-3).

**Bugs fixed, kept as stubs**

- **A plan that failed at chunk 0 hung every client** (fixed 2026-08-18): `active_traj_id` was never set, so the result was dropped as a mismatch and the idle guard returned before `_finish()` could publish the zero and the sentinel. What outlives the fix: **BVP mesh-node exhaustion fails ~36% of fresh start/goal pairs**, it is a property of the start/goal **PAIR** and not of the goal, and a goal the planner cannot solve must degrade **visibly**.
- **Two traced-soak bugs** (fixed 2026-08-14) produced data that parses, scores and looks like a measurement: `--trace-every` subsampled by rollout index against a cycle of commensurate length, and tracing was armed by *file* so untraced rollouts appended to the previous CSV. See [docs/measurement-rig.md](docs/measurement-rig.md).
- **The job runner killed itself on its first job** (fixed 2026-08-14): the job body ran in a brace group ending in `exit`. Signature: a job shown **running** whose log already says `EXIT rc=0`, and the runner NOT RUNNING.

### The gains are settled: `q_cross=2.5, r_omega=2.618` (job 100, 2026-08-18)

**TUNING IS CLOSED.** Adopted in `TVLQRConfig`. The decision rests on five
independent runs over the **40 broad v2 plans** (`tools/select_broad_eval.py`,
mechanically chosen, none of them among the seven), mean-of-5, paired sign tests
over the 40 (`soak_data/soak_broad_*.jsonl`):

| gains | geo `J` | mean max\|e_cross\| | mean `final_err` | miss rate (>0.5 m) |
| --- | --- | --- | --- | --- |
| 0.276 / 2.618 (previously adopted) | 12.98 | 0.686 | 0.388 | 22.0% |
| 1.5 / 2.618 | 12.81 | 0.620 | 0.267 | 13.5% |
| 2.5 / 0.25 | 16.94 | **0.520** | 0.272 | **10.0%** |
| **2.5 / 2.618 (ADOPTED)** | 13.84 | 0.609 | **0.244** | 11.5% |
| 2.5 / 5.0 | **13.79** | 0.655 | 0.277 | 13.0% |
| 10 / 0.25 (the original default) | 17.59 | 0.575 | 0.314 | 15.3% |

Four facts to keep, because they are what a reader will ask about:

- **The move off the original default `10 / 0.25` is the durable result**, and it
  is 1.49x in `J` winning 45 of 51 plans on a library sweep and 35 of 40 on the
  broad set. **Where the default loses is CONTROL EFFORT, not tracking**: it
  achieves slightly *tighter* peak deviation while spending ~3x the control
  (mean `J` control term 6.37 vs 2.22). That is exactly the trade the source
  prescribes for low traction (p. 78, larger `R`), reproduced empirically before
  anyone read the prescription — it belongs in the write-up.
- **The value inside the plateau is not load-bearing.** `J` is flat across a 15x
  range of `q`; `q=2.5` is the rung that won **arrival** (`final_err` 34/40 vs
  the old point, p<0.0001; miss rate 22% → 11.5%), and at `q=2.5` all four `r`
  rungs are indistinguishable on arrival (p>=0.27).
- **Read a gain decision on `final_err` and `J` together.** `max|e_cross|` ranks
  the old default best while it burns 3x the control, and it has disagreed with
  arrival on every ladder run here.
- **The honest write-up claim is a robustness trade, not "tuning halves
  deviation".** Over the 51-plan library sweep the tuned gains gained 9.99 m on
  hard plans (10/11 wins where the default exceeds 1 m) and lost 2.03 m on easy
  ones (~3 cm each), worst single regression 0.282 m. The seven-shape set is
  **enriched for hard plans** and must be described as a corrector test set, not
  as a representative sample of the robot's work.

### Reading a gain result: the method that survived

Distilled from ~12000 rollouts. Any future controller comparison should follow it.

- **Evaluate on the 40 broad v2 plans**, not on the seven. A result on the seven
  is a result about the seven.
- **`J` (`tuning/epsilon.py`) is the objective**, geometric mean across plans,
  and it is an **upper bound on `epsilon`, never `epsilon` itself** (`J*` under
  slip is unknown and positive — say so in the write-up). `J` is computed
  **online** by `EpsilonAccumulator` and is inline in every soak row as
  `j_total`. `tools/score_epsilon.py` remains for already-recorded runs; the two
  paths differ where a wheel saturates (online takes the correction before
  clipping). **`R` charges the CORRECTION, not the total command** — the nominal
  command is what the planner already paid for.
- **Report arrival beside it**: mean `final_err` and the miss rate (>0.5 m).
  Arrival is what separates arms inside the `J` plateau, and it is what the
  robot is for.
- **Compare arms by paired sign test over the plans**, mean-of-5, with any
  reference point **carried in the same process** — cross-run comparability is an
  assumption, and carrying a control costs one arm.
- Rollouts that fail (the patch-spawn guard fires on ~1% of them) invalidate
  their sample and are never averaged over survivors.

### What the source framework requires (SVCM)

Full transcription with LaTeX, page numbers and a glossary in
[docs/svcm-source.md](docs/svcm-source.md); the design it implies is
[docs/corrector-design.md](docs/corrector-design.md). Read those before
arguing about architecture. The parts that constrain what we build:

- **The method is SVCM** and its object is the cost-functional gap: `u` is
  **ε-optimal** on trajectory `z` when `J[u] <= J*[z] + epsilon`. So `J`, not
  `max|e_cross|`, is the currency the write-up must report.
- **`u_adm = u_J + u_bar`** (p. 52): the control the agent already has plus a
  correction. Our frozen PMP plan *is* `u_J` and the corrector *is* `u_bar`, so
  the corrector structure is the theory's own form rather than a shortcut.
- **The catalogue is stored as small networks in the source's own words**
  (p. 77), indexed by environment type and deviation magnitude, holding the
  trajectory, **its costates and the Hamiltonian parameters**. So "RL as the
  library compressor" is the source's proposal, not our extrapolation.
- **RL acts at the PLANNING layer** — server-side, actor-critic with MPC in the
  actor, tuning costates / transversality conditions / cost weights. A 4-wheel
  multiplicative residual on the commands was never this, which is an
  independent reason it was the wrong object.
- **LQR is on-plan**: the paper seed names a lightweight scenario-recognising
  LQR arm as part of the contribution, alongside the PMP catalogue.
- **The remote server is the central claim, not a design smell.** The comms
  delay `τ` is an explicit hypothesis of the dichotomy theorem (the source's
  **Theorem 2**), so the latency objection is retracted as stated. The narrower
  concern that survives: the named protocols (DShot, PWM) describe a flight
  controller, not a Jetson, so the *premise* wants re-checking on our platform.
- **Our `mu2` steering cliff is a demonstration of the theorem's SECOND
  branch** — below the knee no admissible control tracks the plan, so the
  failure is physics, not corrector deficiency. That makes the friction sweep a
  contribution to the advisor's own theory, and it argues for keeping ice in the
  evaluation deliberately as the uncontrollable case.
- Two unresolved mismatches: the theory is written for a 3-wheel `(x,y,θ)` robot
  with `(v,ω)` controls and has **no counterpart to the skid-steer `chi`**; and
  the catalogue is indexed by surface type, which is our patch profiles turned
  into an architecture — but the mapping from a measured `chi` to a catalogue
  index is the "scenario recognition" step both documents assume and neither
  specifies.
- `epsilon.py` cites the functional **(1.7), §1.3.1**, and deliberately differs
  from it twice: (1.7) penalizes deviation from the **terminal target** and
  charges the **total** control; we penalize deviation from the **moving
  reference** and charge only the **correction**. Argued in its docstring.

### The next build: the re-join re-planner (agreed 2026-08-15)

Design and phasing in [docs/corrector-design.md](docs/corrector-design.md);
current status in `handover.md`. Three things recorded here because they are the
ones most likely to be re-litigated:

- **The re-join BC is a change to the PMP problem statement, not a new solver.**
  `x(t0)` = the actual off-plan state; `x(t0 + T_w)` = the nominal plan's state
  at index `k + T_w/dt`, **all five components including wheel speeds** —
  playback resumes by feeding the plan's commands from that index, which assume
  the wheels already turn at the plan's rate. Because playback is time-indexed,
  landing at `k + T_w/dt` keeps the robot on *schedule*, so `m` is determined by
  `T_w` and `T_w` is the only free scalar.
- **Sample randomly, NEVER on a grid.** The curse of dimensionality applies to
  *tabulating* a ~10-D catalogue (5 points/axis is 1e7 solves, ~5000 core-hours
  at the measured 1.84 s) and **not** to *regressing* it (~100k random samples,
  ~55 core-hours). That distinction is the whole reason a network is the right
  storage format, so do not let "curse of dimensionality" argue against the
  sampling too.
- **Label generation on demand is DAgger, not RL.** Querying the expert where the
  student is currently wrong is right and necessary (distribution shift), but
  when PMP's answer is in hand the student-minus-expert difference IS the exact
  gradient — turning it into a scalar reward substitutes a high-variance
  estimator for a quantity already known exactly. RL earns its keep in exactly
  one place: **fine-tuning above the teacher**, since PMP is optimal for a model
  that assumes nominal chi and the plant does not.

**Phase 0 gates everything: measure the re-join SOLVE FAILURE RATE first.** The
library build failed 36%. A teacher that answers two-thirds of the time cannot
label a dataset, and that would make the supervised plan wrong rather than slow.
It is offline, needs no Gazebo, and is cheap.

### Ideas queued, roughly in order of expected value

1. **Compare against Nav2 baselines (DWA, MPPI, TEB)** — the advisor's explicit
   requirement as of 2026-09-16: a paper that only compares the method against
   itself will not pass Q1 review. Carried over from the paper draft's
   Experiment 4, which named them and was never run. Metrics proposed there:
   path length, travel time, max curvature, and control energy
   `int ||(a_l, a_r)||^2 dt`. The comparison is **not like-for-like**: those are
   closed-loop planners, so the comparable object is the whole FM2+PMP+TVLQR
   stack, not the corrector alone.
2. **Fix SAC's entropy runaway before any retrain.** `ent_coef` reached 3.31.
   Either pin it (`ent_coef=0.05` instead of `"auto"`) or set an explicit
   `target_entropy` — the default `-dim(A)` is far too permissive for a 4-D
   residual whose useful range is tiny. Every hour of the 20260730 run after
   ~800k steps made the policy worse.
3. **Bound the per-episode return.** Huber bounded the reward's slope, not the
   accumulated return over 200 non-terminating steps, and `critic_loss` still
   reached 1.2e4. Options: normalize the return, cap per-step cost outright, or
   reinstate termination with a large-but-finite terminal penalty (not the same
   as the 0.5 m corridor that caused the original no-recovery problem).
   `-epsilon.step_cost(...)` is the per-step integrand and is the reward any
   future RL should use.
4. **Give the RL residual a fair fight**: train it *on top of* tuned TVLQR
   rather than on top of identity, so the policy learns the residual a good
   linear controller cannot supply instead of re-deriving feedback from scratch.
   Also the most defensible version in a write-up — the advisor's requirement is
   that RL be part of the system, not that it beat everything alone.
5. **Per-worker job queues.** `tools/jobq.sh` is single-lane in the default
   partition, so parallelism today means driving `WORKER=n` by hand. This is
   what would turn ~55 core-hours of PMP labelling into an overnight run.
6. **Widen the search to the full Q/R diagonal** (`q_along`, `q_heading`,
   `r_v`) — the 2-D machinery is proven, but the 2-D result needed 40 plans and
   mean-of-5 to resolve, so budget accordingly. Low priority: `J` could not
   separate points inside the 2-D plateau at all.
7. **A `sand` profile.** The advisor wants ice *and* sand. Blocked on a
   modelling limit, not on parameter choice: a Coulomb `mu` alone does not model
   granular flow, and under min-combination any ground below the wheel's `mu2`
   has no steering authority at all. See [docs/measurement-rig.md](docs/measurement-rig.md).
8. **`floor_1_00050` is a degenerate PMP plan** (`max_turn = 3.14 rad/step` over
   6 m). Still unexplained, still excluded, still a planner bug rather than a
   control one.

## Conventions

- Non-obvious design decisions are documented in long module docstrings (the PMP
  planner, the runtime corrector, `RLCorrectorConfig`). When changing behaviour
  there, update the docstring in the same edit — they are treated as the spec.
- Node parameters are loaded from dataclasses via
  `agx_planning.utils.declare_and_load_dataclass`; add a field to the dataclass
  rather than a bare `declare_parameter`.
- **Figures are dated, indexed and committed** — `figures/YYYY-MM-DD/{render.py,
  README.md,*.png}`, convention in [figures/README.md](figures/README.md). This
  reverses the usual "version the tool, not its output" rule on purpose: these
  render from gitignored data directories, several of which cost hours of machine
  time on a plant that will not exist forever, so the renderer alone does not
  reproduce the picture. `tools/plot_*.py` keeps the old rule. Everything from
  before 2026-08-13 is in `figures/archive/` with what provenance survives.
- `acados/` at the repo root is untracked scratch; the Makefile's `ACADOS_*` /
  `t_renderer` bits are vestigial and unset by default.
