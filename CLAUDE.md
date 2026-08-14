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

## Current work: making the corrector work

**This section is a living log — append to it, or rewrite the parts it
contradicts, whenever a run or an experiment establishes something new.**

**Split rule (2026-08-13):** superseded tables and resolved-bug narratives live in
[docs/corrector-history.md](docs/corrector-history.md), not here. The axis is
**"does this still constrain a decision?"**, not chronology — a retraction that
stops someone re-proposing a dead idea is *live* however old it is, and gets a
one-line stub under "Settled" below. What moves out is the investigation behind
it and any measurement taken on a plant or rig that no longer exists. When you
retire something, leave the stub: an unstubbed removal means the idea comes back. It is
the standing context for what is being worked on right now; the rest of this
file describes the system, which changes far more slowly. Date each claim, and
delete a claim outright when it is superseded rather than leaving both versions.

The single active goal is getting the runtime corrector to hold a frozen PMP
trajectory under slip. Nothing else is in progress.

**Keep [handover.md](handover.md) current — it is the primary record of what we
are doing, and it outranks anyone's memory of it.** Sessions here are days apart
and the user explicitly relies on that file rather than recall, so a session that
learns something and does not write it down has lost it. The two files split by
lifetime, not by importance:

- **this section** accumulates *established findings* — a measured number, a
  retracted claim, a mechanism understood. It is cumulative and dated.
- **`handover.md`** is *state*: what is running on the VM right now, what is
  half-finished, which caches are poisoned, what the next session should do
  first, and the reasoning behind an in-flight decision that has not yet become
  a finding. Rewrite it rather than appending — it describes now, not history.

Update both **in the same session that produced the change**, not at the end of a
run of them; and when a launched experiment is still in flight at the end of a
session, `handover.md` must say so, say where its log is, and say what to
conclude from either outcome.

### Settled — do not re-propose, do not re-measure

One line each, because a 'we already ruled that out' is the cheapest useful
text in this file and the narrative behind it is not. Full reasoning in
[docs/corrector-history.md](docs/corrector-history.md); these stubs exist so a
bad idea gets stopped before anyone reads it.

- **The 20260730 SAC run (1.5M steps) learned nothing usable** — 2 successes in 8864 episodes, `ent_coef` ran away to 3.31, `critic_loss` 1.2e4. Fix the entropy target and bound the return before any retrain (queue 1-2). Detail: [docs/corrector-history.md](docs/corrector-history.md).
- **Nothing measured before 2026-08-02 evening is valid.** `WorldControl.pause` is a proto3 bool, so every `multi_step` silently un-paused the world and it free-ran between control steps. Fixed; `_ensure_paused` verifies the clock stopped. Detail: [docs/corrector-history.md](docs/corrector-history.md).
- **`final_err` is NOT reproducible and must never be a tuning objective** (sd 0.26 vs max|e_cross|'s 0.0002 on the trajectory it was measured on). The seed is ~1e-13 in residual wheel speed, amplified at turn reversals where lateral friction switches direction — genuine chaos, not a bug. Do not re-propose the ROS-publish-vs-gz-step race or terrain differences; both were ruled out. Detail: [docs/corrector-history.md](docs/corrector-history.md).
- **The 0.0002 m noise floor was ONE trajectory's and does not generalise.** The 7-shape set is ~400x noisier because turn reversals are chaotic amplifiers. Nelder-Mead cannot converge on a noisy objective — it re-sampled one point 71 times. Superseded operationally by mean-of-3 repeats. Detail: [docs/corrector-history.md](docs/corrector-history.md).
- The 2026-08-07 BO run (100 evals, mean-of-3) found `q_cross=0.276 / r_omega=2.618` at 0.614 m vs the default's 1.042 m. Its binned per-variable tables are **smoothing artifacts** — treat any one-variable summary of this landscape as suspect. Detail: [docs/corrector-history.md](docs/corrector-history.md).
- The `q_cross` search-box floor was **not** the limit (nothing below 0.1 is better, so the optimum is interior). `--q-bounds` exists because `x0` is CLIPPED into the box, so a probe outside it silently measures the boundary. Detail: [docs/corrector-history.md](docs/corrector-history.md).
- The neighbourhood is a shallow bowl with a notch: 13 of 15 grid points beat the default across ±60% in `q` and 6x in `r`. `r_omega` matters locally even though the binned table denied it. **A per-shape claim must name its baseline** — computing one against sweep neighbours rather than the default produced a retracted attribution. Detail: [docs/corrector-history.md](docs/corrector-history.md).
- **The 2026-08-03 three-way table was measured on the broken plant** (`mu2=0.7`, robot could not steer on its own patches) and is superseded by the 2026-08-07 baseline above. Detail: [docs/corrector-history.md](docs/corrector-history.md).
- **No RL checkpoint shows a learning trend on the real task** (r=0.111 over 20 checkpoints; 0 of 20 beat TVLQR). The TB metric that looked like progress was logged by the mis-stepped env. Quote 800k as *best* checkpoint, never as typical. Detail: [docs/corrector-history.md](docs/corrector-history.md).
- The wheel-velocity residual was **not** the determinism seed (wheels already agree to 1e-16); the un-gated IMU read was, and is now gated. Detail: [docs/corrector-history.md](docs/corrector-history.md).
- **The old run-to-run variance was patches silently not spawning**, not an 8 mm reset slide (that theory is retracted). With the world paused, `create` returns False while creating the entity anyway, so ~half of rollouts ran on bare ground. `_wait_entities` now confirms via pose/info; **never re-issue a create for a 'missing' patch** — it is in flight, and re-creating genuinely fails. Detail: [docs/corrector-history.md](docs/corrector-history.md).

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

### The tuned point validates (2026-08-12)

Independent mean-of-5 at each gain pair, tuner code path, fresh caches, clean
sim. `tune_data/validate_20260812_{tuned,default}.jsonl`. The validation is run
as `tune_tvlqr --max-evals 1 --repeats 5 --q-cross Q --r-omega R`: BO seeds its
design with `x0`, so a 1-evaluation run is exactly a clean measurement at a
chosen point, in the same code path that produced the number being checked.

| | tuning run (mean-of-3) | validation (mean-of-5) |
| --- | --- | --- |
| default `q=10 / r=0.25` | 1.0417 | **1.0037** |
| tuned `q=0.276 / r=2.618` | 0.6144 | **0.6212** |

**This is the first tuning result on this project that survives an independent
re-measurement.** Both earlier ones (0.183 m, 0.9412 m) were minima of noisy
draws and evaporated on inspection.

### What the source documents actually say (read 2026-08-13)

Read directly from the advisor's own dissertation draft (`Киселёв_докторская_v1.docx`,
§1.2.2-1.3.3 and ch. 2) and the paper seed (`Затравка статьи.docx`), both in
`~/Downloads/Telegram Desktop/`. **This section supersedes several inferences
made from second-hand descriptions of the concept, including some made earlier in
this file and in conversation.** The paper seed's §"Синтез управления в зонах
изменения физических значений среды" is largely lifted from dissertation §1.3.2,
so the two agree.

**The method has a name and a formal definition.** It is **SVCM** —
*STRL-Variative Control Method*. A control `u` is **ε-optimal** on a trajectory
`z` when

    J[u] <= J*[z] + epsilon

where `J*[z]` is the best achievable value of the cost functional on `z`. The set
of such controls is `U_eps`. Applied reading, in the advisor's words: a bounded
set of admissible agent states within which the robot can keep moving **without
critical effect on the final distance to the goal**.

**The architecture, exactly as specified:**

1. **Offline, on a remote server:** build a finite family of environment
   scenarios per external factor `w` — *dry asphalt, wet surface, ice, mud*. For
   each, solve with PMP for a control that is ε-optimal. Store the trajectory,
   **its conjugate (costate) trajectory, and the Hamiltonian parameters** as a
   *catalogue of suboptimal crisis templates*, indexed by environment type and by
   the magnitude of deviation from expected system characteristics.
2. **Onboard, in real time:** the agent tracks state `x_k` plus diagnostic
   features `d_k` (explicitly: *wheel-slip indicators*). **It first tries to
   compensate with the control it already has**, and only escalates if that
   fails.
3. **On a traction-loss event:** send `(x_k, d_k)` to the server; the server
   replies with an index `i` and "apply template `u_i` over horizon `T_w`"
   (`T_w` = a fixed local control window, e.g. *the time to drive out of the
   puddle*). Onboard stores either the parameterised trajectory or a compact rule
   for reproducing it.

**Theorem 1 ("On realizational ε-admissibility and controllability accounting for
real time")** assumes scenario coverage (the catalogue δ-approximates the real
dynamics on `T_w`), a communication delay `τ` small relative to `T_w` with a
Lipschitz bound on the resulting drift, and templates that respect the actuator
and state constraints. It then gives a **dichotomy**:

- **Controllable** — if *some* admissible control achieves acceptable `J`, the
  real-time "event → server → template" scheme achieves `J <= J* + eps'`.
- **Uncontrollable** — if *no* admissible control does, then no finite-catalogue,
  real-time-bounded strategy can either, **and the cause is the physics, not the
  algorithm or the network**.

**Consequences for this project, in order of how much they change what we do:**

**1. We have been measuring the wrong quantity.** Every table above scores
`max|e_cross|` — a tracking error. The theory is stated entirely in terms of the
**cost-functional gap `J[u] - J*`**, which is what `epsilon` *is*. These are
different quantities and we have never computed the one the framework uses. We
can: `PlannerConfig` fully specifies the running cost (`w_h`, `w_v`, `w_brake`,
`w_omega_run`, the barriers) and the terminal cost, and `run_recorder` already
stores the executed track. **CORRECTED same day:** `J` needs the per-step track, and
`variance_probe.drive` reduces each rollout to scalars, so the ~4000 soak
rollouts **cannot** be rescored — only fixture runs, where `run_recorder`
writes `<run>_track.csv`, can. Capturing the track for future rollouts is a
small change; the existing soak data is not recoverable for this. Doing so
would let every result be reported in the thesis's own currency, and would give
`epsilon` an empirical value instead of a symbol.

**2. Our `mu2` / steering-cliff result is a demonstration of Theorem 1's SECOND
branch.** Below the knee the robot achieves ~5% of commanded yaw rate — no
admissible control tracks the plan, so the failure is fundamental rather than a
corrector deficiency. That is exactly the "uncontrollable" case, which the
dissertation asserts but does not demonstrate experimentally. **This makes the
friction sweep a contribution to the advisor's own theory rather than a
side-quest**, and it argues for keeping ice in the evaluation deliberately, as
the branch-2 case, rather than treating "ice is uncontrollable" as a limitation.

**3. The remote server is not a design smell, it is the central claim.** It was
argued against earlier today on latency/connectivity grounds; **that objection is
retracted as stated** — the delay `τ` is an explicit hypothesis of Theorem 1, and
§1.2.2 justifies offloading from the platform's side: the named comms protocols
are **DShot and PWM** (drone ESC protocols, 16-bit frames), and the doc proposes
*extending DShot* to carry the model updates. The concern that survives is
narrower and worth stating as such: those constraints describe a flight
controller, not a Jetson, so on our platform the *premise* wants re-checking even
though the *architecture* is sound where it holds.

**4. LQR is on-plan, not a detour.** The paper seed says outright that a
"lightweight algorithm" which recognises the scenario and **applies LQR** is part
of the contribution, alongside the PMP catalogue. §1.2.2-1.2.3 develop an
LQG-based sibling method (SFCC) to SVCM's PMP-based one. So the TVLQR work is one
of the two intended arms.

**5. RL's role is at the PMP problem's parameters, not on the wheels.** Ch. 2
puts RL on the **server side**, in actor-critic form with MPC embedded in the
actor, and the defended claim is that *"the weights of the conjugate system can
be tuned by the author's experience-transfer algorithm"* and that
**transversality conditions** are set from experimental checking of situations.
So RL adjusts the **costates / transversality conditions / cost weights of the
optimal control problem** — i.e. it acts at the planning layer. Our 4-wheel
multiplicative residual on the commands was never this, which is an additional,
independent reason it was the wrong object — separate from why it failed to
train.

**6. The dissertation pre-empts the obvious alternatives** (§1.3.3), which is
useful for our related-work section: abstract ε-optimality existence theory
(Uryson integral equations, admissible set a closed ball in `L_p`) proves
existence non-constructively with no real-time or agent-server story;
ε-optimal-policy results for MDPs (and hence RL) are for discrete states and
actions, average-cost criteria and stationary policies, with no comms delay or
event-triggered switching; and invariant/admissible-set methods target constraint
satisfaction at mode switches rather than ε-optimality of a cost functional.

**Two mismatches between the theory and our implementation, both unresolved:**

- The theory is written for a **three-wheeled robot with a driving wheel**, state
  `(x, y, theta)` with `(v, omega)` controls. Our PMP is a **5D skid-steer
  wheel-space** model with per-wheel accelerations. The skid-steer `chi` has no
  counterpart in the source formulation.
- The catalogue is indexed by **surface type** (asphalt/wet/ice/mud), which is
  precisely the `slip_chi`-is-a-property-of-the-surface finding of 2026-08-05
  turned into an architecture. Our patch profiles are the same idea; they have
  never been indexed or catalogued, and the mapping from a measured `chi` to a
  catalogue index is exactly the "scenario recognition" step both documents
  assume and neither specifies.

### The U-turn's bad mode is seeded by ~1e-6 m of settle height (2026-08-13)

30 traced rollouts of `floor_6_00031` at fixed gains, 14 tuned / 16 default
(`uturn_traces/`, `uturn_{tuned,default}.jsonl`). The reset is **not** the
variable, which is what the probe was built to check:

| | tuned | default |
| --- | --- | --- |
| n | 14 | 16 |
| mean / sd | 1.423 / 0.346 | 1.270 / 0.389 |
| bad-mode rollouts | 1 (2.572) | 1 (2.591) |
| **start-pose spread (x, y, θ)** | **3e-11, 5e-11, 9e-15** | **1e-11, 1e-11, 6e-14** |
| `reset_ticks` / `lost_steps` | 20 / 0 | 20 / 0 |

**Start poses agree to ~1e-11 m and outcomes still span 1.5 m.** So the
bimodality is not initial-position spread, and the reset work of 2026-08-02 is
not implicated. `reset_ticks` is a constant 20 and no steps were lost.

`trace_diff` on a good (1.255) vs the bad (2.572) rollout puts the seed exactly:

- **at reset, before any command: `z` differs by 4e-6 m**, plus IMU by ~0.017;
- `y` moves at **step 1**, the commands only at **step 2**.

State first, commands second — so by `trace_diff`'s own reading this is physics,
not our controller. And the 4e-6 m is the **residual of the reset settle loop**,
which converges to ~1e-6 (`reset_settle_z_tol`). The IMU difference is a red
herring here: TVLQR does not consume the IMU, so it cannot be what moved the
commands.

**Reading:** the U-turn's bad mode is chaotic amplification of a ~1e-6 m
*vertical* settle residual, through the reversal mechanism already documented.
Supporting evidence that it is a property of the plant and not the corrector: the
bad-mode value is **2.572 vs 2.591** in the two arms — the same attractor at gains
differing by 36x — and the frequency is ~7% here against the soak's 10.3%/12.4%.

**Do not chase this with a tighter `z` tolerance.** The 2026-08-04 world-reset
experiment bought three decades of initial-state agreement and chaotic
amplification spent them in ~30 steps. A tolerance is the wrong instrument
against an exponential. The actionable consequence is unchanged and already in
force: on reversal-heavy shapes, report mode frequencies over many samples rather
than means over few.

### What 4065 rollouts say: the mechanism is MODE FREQUENCY (2026-08-13)

The overnight soak (`tuning/soak.py`, `just soak`) accumulated **4108 rollouts,
43 failed (1.0%), 4065 usable** across 21 processes and 14 complete cycles of the
7-shape set at both gain points — roughly **290 samples per shape per arm**,
against the n=3-5 every earlier claim rested on. Raw rows in `soak_data/`.

**The headline is now settled beyond argument.** Mean over the 7 shapes, per
complete cycle, n=14 cycles per arm:

| gains | mean | sd | min | max |
| --- | --- | --- | --- | --- |
| tuned `q=0.276 / r=2.618` | **0.6686** | 0.0882 | 0.618 | 0.982 |
| default `q=10 / r=0.25` | **1.1273** | 0.1108 | 0.897 | 1.334 |

The distributions do not overlap at all (worst tuned cycle 0.982 < best default
cycle 0.897 is *nearly* true; the single exception is one tuned cycle). This is
the fourth independent measurement of the tuned point (0.6144 / 0.6212 / 0.6429 /
**0.6686**) and the third of the default (1.0417 / 1.0037 / **1.1273**).
**The gains are now ADOPTED in `TVLQRConfig`** — `q_cross` 10.0 → 0.276,
`r_omega` 0.25 → 2.618. Note `r_omega` moving *up* reverses the reasoning in its
own docstring comment ("angular correction is cheaper"); the comment is left in
place with the correction beside it, because the argument was sound and the
measurement disagreed anyway.

**The important finding is not the mean, it is HOW the win happens.** Per shape,
pooled (mean ± sd over ~290):

| shape | tuned | default | character |
| --- | --- | --- | --- |
| straight | 0.054 ± 0.001 | 0.078 ± 0.054 | level |
| corner | 0.311 ± 0.009 | **0.226 ± 0.000** | level, default wins |
| S | 1.617 ± 0.088 | 2.131 ± 0.016 | level |
| **zigzag** | **0.503 ± 0.352** | **2.009 ± 0.533** | **mode frequency** |
| tight V | 0.286 ± 0.001 | 0.357 ± 0.219 | mode frequency |
| **U-turn** | 1.502 ± 0.520 | **1.316 ± 0.470** | **modes, unchanged** |
| loop | 0.459 ± 0.055 | 1.678 ± 0.090 | level |

Four shapes are **unimodal and near-deterministic in both arms** (sd ≤ 0.09,
straight/corner/S/loop): those are plain level shifts, three won by the tuned
gains and one — the corner — genuinely won by the default, by 0.085 m at sd
0.000. Nothing there is noise.

The other three are **bimodal, and the gains move the FREQUENCY of the bad mode,
not its depth**:

| shape | bad mode at | tuned %bad | default %bad |
| --- | --- | --- | --- |
| zigzag | ~2.2-2.5 m | **2.6%** | **88.8%** |
| tight V | ~0.78 m | 0.0% | 20.4% |
| U-turn | ~2.5-2.9 m | 10.3% | 12.4% |

**The zigzag flip is 88.8% → 2.6%, and that single number is most of the
result.** It also explains why the zigzag looked "noisy" (sd 0.215-0.533) in every
earlier table: it was never noise, it was a ~90/10 mixture of two reproducible
outcomes being sampled 3 times.

**RETRACT: "the U-turn is bistable and the tuned gains land it in the good
mode."** It is bistable — but the bad-mode rate is **10.3% vs 12.4%**, i.e.
statistically indistinguishable, so *the gains do not control it at all*
(**itself corrected the same evening — see "The `q_cross` ladder" below; the
gains control it sharply, but non-monotonically, so two points could not see
it**). This
also corrects the 2026-08-12 grid figure of "33% good / 67% bad": that pooled 45
rollouts across many gain pairs, and at neither validated point is the bad mode
anywhere near a majority. The U-turn's mode is driven by something else, and it
is the one open mechanism left. (Consistent with the U-turn contributing -3% to
the improvement: the tuned arm is slightly *worse* there, 1.502 vs 1.316.)

**Methodological point worth keeping.** Every "run-to-run variance" number in
this file above is a mixture width, not a measurement error, on exactly the
shapes that turn out to be bimodal. The right estimator for a bimodal metric is
not the mean of 3 — it is the mode frequency, which needs ~100 samples and was
unaffordable until a rollout cost 5 s. Mean-of-3 remains fine for *searching*
(it is what found these gains) but a per-shape *claim* now wants soak-scale n.

**The 1.0% failure rate is all `terrain patches failed to spawn`** (43/4108,
spread evenly over all 7 shapes), correctly invalidating the rollout rather than
being recorded as a sample. That is the 2026-08-01 guard working; at soak scale
it is now measurable, and it is small enough not to bias anything.

- **`reset_world=True` DESTROYS THE ROBOT — never use it.** A gz `WorldControl.reset.all` deletes runtime-spawned entities, including the `scout_mini`. Recovery is `just kill-sim` + `just remote-sim`, not debugging.

### The `q_cross` ladder: a zigzag threshold and a U-turn notch (2026-08-13)

1047 rollouts, the three mode-bearing shapes × six gain points, `r_omega` held at
2.618 across the ladder so `q`'s effect is separable (`soak_data/soak_20260813_ladder.jsonl`,
figure `figures/2026-08-13/03_ladder_modes.png`). n≈58 per cell. **They are two
different phenomena, and only one of them is a mode frequency.**

| shape | q=0.1 | q=0.276 | q=0.6 | q=1.5 | q=10 | q=10, r=0.25 |
| --- | --- | --- | --- | --- | --- | --- |
| zigzag %bad (>1.5 m) | 1.7 | 1.7 | 0.0 | 0.0 | **89.7** | 86.0 |
| U-turn %bad (>2.0 m) | **100** | **15** | **100** | 82.8 | 96.6 | 14.0 |
| tight V %bad (>0.6 m) | 0 | 0 | 0 | 0 | 0 | 10.5 |

**The zigzag is a THRESHOLD in `q_cross`, and it is where most of the tuned
gains' win comes from.** Flat at 0-2% across a 15× range of `q`, then a cliff
between 1.5 and 10. `r_omega` is irrelevant to it (86.0 vs 89.7 at `q=10`). So
the adopted point sits inside a *wide plateau*, not on an edge — the robust half
of the result.

**The U-turn is a NARROW NOTCH, and calling it bimodal was wrong.** At `q=0.1`
all 59 rollouts fall in 2.634–2.665; at `q=0.6`, all 56 in 2.701–2.709. Those are
tight, deterministic, **unimodal** distributions that happen to be bad — not a
mixture whose frequency shifted. Only `q=0.276` and the old default drop to ~1.4,
and they are isolated: their immediate neighbours in `q` are uniformly bad.
**This retracts the same morning's "the gains do not control the U-turn at
all"** — they control it sharply; two sample points could not see a notch.
Whether 0.276 has a usable basin or is a spike was the open question; it is
answered below.

**The tight V's mode was an `r_omega` effect, not a `q` one** — 0% bad at every
rung with `r=2.618`, 10.5% at the old `r=0.25`. Small either way.

**`max|e_cross|` and `final_err` rank the ladder DIFFERENTLY, and that is not
noise.** On the U-turn, `q=1.5` is the *worst* rung by max|e_cross| (2.150) and
the *best* at arriving (mean `final_err` 0.255, **0%** of 58 rollouts ending more
than 0.5 m out), while the adopted `q=0.276` scores 1.605 and leaves 55% of
rollouts short. A metric that disagrees with "did it get there" needs justifying,
which is what the `J` work below is for.

### Scoring in `J`: the objective change does not reverse the conclusion (2026-08-13)

`tuning/epsilon.py` (pure, 23 tests) holds the tracking functional; `tools/score_epsilon.py`
is the offline driver that scores recorded per-step traces with it. First real
measurement: 7 shapes × {tuned, default} × 5 repeats **with traces**, so every
rollout is scored both ways (`jtraces/`, `epsilon_data/jsweep.jsonl`, figure
`figures/2026-08-13/04_epsilon_vs_cross.png`). Mean of 5:

| shape | max\|e_cross\| tuned / default | J tuned / default | metrics agree? |
| --- | --- | --- | --- |
| straight | 0.054 / 0.074 | 0.22 / 1.84 | yes |
| corner | 0.309 / **0.226** | **1.50** / 1.92 | **no** |
| S | 1.582 / 2.129 | 21.8 / 84.8 | yes |
| zigzag | 0.410 / 1.587 | 13.2 / 62.3 | yes |
| tight V | 0.286 / **0.256** | **5.23** / 12.2 | **no** |
| U-turn | 1.316 / **1.180** | **29.9** / 35.7 | **no** |
| loop | 0.423 / 1.714 | 21.5 / 66.6 | yes |

**In `max|e_cross|` the tuned gains win 4 and lose 3; in `J` they win all seven**,
by 1.3× to 8.3×. Every one of the three losses is a shape where the tuned arm
concedes a little peak deviation and buys back much more in accumulated error,
correction effort and terminal miss. So `J` is not a different answer — it is a
**cleaner version of the same answer**, and it is the quantity SVCM is stated in.

Two things to hold onto:

- **`J` is an upper bound on `epsilon`, never `epsilon` itself** (`J* > 0` under
  slip and is unknown). The module docstring says so; say it in the write-up too.
- **The correction, not the total command, is what `R` charges.** The nominal
  command is what the planner already paid for; charging it again would score
  every corrector for the plan's cost. `score_epsilon.py` recovers it as
  applied-minus-nominal in wheel space.

**Scoring needs the TRACK, so it must be captured at rollout time.** The ~4000
soak rollouts cannot be rescored — `variance_probe.drive` reduces each to
scalars. Any future soak whose numbers should be readable in `J` must be run with
`--trace-dir`. **`soak.py` now HAS `--trace-dir` (plus `--trace-every N` to keep
an unbounded soak from filling the disk at ~130 kB a rollout), so this is wired
rather than merely known.** Untraced rows carry no `trace` field and are skipped
by the scorer rather than silently scored against someone else's file.

### The U-turn basin has walls, and `q=0.276` stays adopted (2026-08-13 evening)

Two runs closed the ladder question. **The adopted gains survive**, and the
reason is more interesting than "they were right".

**1. The sub-ladder: the basin is real, and 0.276 sits on its LEFT EDGE.**
5906 usable rollouts, `floor_6_00031` only, n≈985 per rung, `r_omega=2.618`
throughout (`soak_data/soak_20260813_uturn_subladder.jsonl`):

| `q_cross` | mean | sd | %bad (>2.0 m) | mean `final_err` | %miss (>0.5 m) |
| --- | --- | --- | --- | --- | --- |
| 0.200 | 2.647 | 0.120 | 99.2 | 1.153 | 99.4 |
| **0.276 (adopted)** | 1.637 | **0.594** | **17.9** | 0.643 | 46.6 |
| 0.320 | **1.494** | 0.154 | 1.8 | **0.347** | **9.8** |
| 0.400 | 1.546 | 0.090 | **0.7** | 0.363 | 25.0 |
| 0.500 | 2.685 | 0.029 | 99.9 | 0.619 | 100.0 |
| 0.600 | 2.707 | 0.002 | 100.0 | 0.530 | 96.0 |

So the notch is a **basin roughly `[0.276, 0.4]` with near-vertical walls** —
0.2 and 0.5 are ~100% bad and nearly deterministic (sd 0.12 / 0.03), not
mixtures. **The adopted point is the noisiest rung in the entire ladder** (sd
0.594, 17.9% bad) precisely because it straddles a wall: the interior is
near-deterministic (0.7-1.8% bad). That is a better explanation of the U-turn's
"bimodality" than the shape itself — it was never intrinsic, it was an artifact
of measuring at a gain sitting on the edge.

**2. The seven-shape check: `q=0.32` is NOT free, so do not re-adopt it.**
7 shapes × `q` ∈ {0.276, 0.32, 0.40} at `r=2.618`, 5 repeats, **traced**, so it
is scored in both currencies (`gaincheck/`, `epsilon_data/gaincheck_J.jsonl`):

| shape | `J` 0.276 / 0.32 / 0.40 | max\|e_cross\| 0.276 / 0.32 / 0.40 |
| --- | --- | --- |
| straight | 0.22 / **0.21** / 0.84 | 0.055 / **0.046** / 0.127 |
| S | **22.3** / 30.0 / 41.8 | **1.605** / 1.885 / 2.522 |
| corner | 1.49 / 1.45 / **1.44** | 0.310 / 0.306 / **0.303** |
| loop | 21.5 / **20.8** / 23.1 | 0.422 / **0.383** / 0.558 |
| U-turn | 41.8 / 35.6 / **33.6** | 1.623 / **1.417** / 1.540 |
| zigzag | 13.5 / 16.6 / **12.1** | **0.445** / 0.711 / 0.464 |
| tight V | **5.23** / 5.30 / 5.94 | 0.286 / **0.285** / 0.297 |
| **MEAN** | **15.15** / 15.71 / 16.98 | **0.678** / 0.719 / 0.830 |

**`q=0.32` buys the U-turn and pays for it on the S and the zigzag**, and the
aggregate says the trade is not worth making. The U-turn gain is real (`J` 41.8
→ 35.6, `final_err` 0.641 → 0.252) — it is simply smaller than what it costs
elsewhere. **`TVLQRConfig` is unchanged: `q_cross=0.276`, `r_omega=2.618`.**

**`J` and `max|e_cross|` AGREE here, for the first time.** Both rank
0.276 < 0.32 < 0.40 on the aggregate. That is worth noting because the same-day
`J` section above found them disagreeing on 3 of 7 shapes: the metrics diverge
on *per-shape* verdicts, not necessarily on aggregate ranking. Do not
generalise either way from one case.

**The methodological point, which is the durable part:** a per-shape optimum
(`q=0.32` on the U-turn, at n≈985 and beyond argument) **did not survive the
seven-shape aggregate**. A gain chosen on one trajectory is a gain overfitted to
one trajectory, however many samples back it. The sub-ladder was still worth the
night — it explains *why* 0.276 looks noisy on the U-turn — but the decision was
always the aggregate's to make.

### The library sweep: it generalises, but as a ROBUSTNESS TRADE (2026-08-14)

Every corrector claim to date rested on 7 hand-picked plans, so the obvious
reviewer question was whether they generalise. **They do.** 51 plans (every plan
>= 10 m, chosen by a mechanical rule rather than by us), tuned `0.276/2.618` vs
default `10/0.25`, 3 repeats, traced. 306 rollouts, 4 invalidated by the
patch-spawn guard, 302 usable. Rows in `soak_data/libsweep.jsonl`, scored into
`epsilon_data/libsweep_J.jsonl`.

| | tuned | default | tuned wins |
| --- | --- | --- | --- |
| mean `max\|e_cross\|` | **0.449** | 0.605 | 21/51 |
| median `max\|e_cross\|` | 0.255 | **0.214** | |
| mean `J` | **10.40** | 42.26 | **45/51** |
| median `J` | **6.64** | 8.14 | |
| mean `final_err` | **0.294** | 0.387 | |

**The aggregate holds in both currencies, but in metres the default wins the
MAJORITY of plans (30/51) while losing the aggregate badly.** Split by
difficulty, that resolves completely:

| bucket | n | tuned | default | tuned wins |
| --- | --- | --- | --- | --- |
| easy (default < 0.3 m) | 37 | 0.213 | **0.182** | 11/37 |
| medium (0.3-1.0) | 3 | 0.791 | **0.589** | 0/3 |
| hard (> 1.0) | 11 | **1.150** | 2.032 | **10/11** |

Totals: 9.99 m gained, 2.03 m lost, worst single regression 0.282 m. So the
tuned gains **trade ~3 cm of precision on easy plans for preventing blow-ups on
hard ones** — and `J` scores that trade as a near-sweep (45/51), because the
easy-plan "losses" in peak deviation are paid back in accumulated error, control
effort and terminal miss. `floor_6_00031` alone goes 1042.88 -> 40.73 in `J`.

**Two things to carry into the write-up.** First, the honest claim is not "TVLQR
tuning halves deviation everywhere" — it is a robustness trade that pays on hard
trajectories and is ~free on easy ones. Second, **the 7-shape set is enriched for
hard plans** (11 of 51 library plans are hard; most of our seven are), which is
why it reads 0.67 vs 1.13 where the library reads 0.449 vs 0.605. That is the
right design for a corrector test set, but it must be described as one and not
as a representative sample of the robot's work.

**`J` and `max|e_cross|` disagree on 24 of 51 plans while agreeing on the
aggregate direction.** The 2026-08-13 gaincheck found them agreeing on the
aggregate too; at n=51 the per-plan divergence is much larger than that suggested.

### Generating evaluation trajectories by construction (2026-08-14)

Queue item 8, built. Motivation is sharper than "more plans": **every per-shape
claim rests on exactly ONE plan of that shape.** The U-turn notch is 5906
rollouts of `floor_6_00031`; nothing distinguishes a property of U-turns from a
property of that U-turn, and the sub-ladder's near-vertical walls make that a
real risk rather than a pedantic one.

`agx_planning/tuning/shape.py` (pure, 27 tests) holds the descriptors and the
screen; `tools/sample_eval_trajectories.py` is the offline driver.

**Measured: `trim_pivot`'s inherited 0.30 m is too small.** Real plans turn ~2.8
rad within their first ~0.7 m of travel — a very tight arc, not a pure spin, so
arc-length resampling does not remove it on its own. Sweeping the threshold over
the 100-plan library moves the label counts until 0.7 m and then not at all
(STRAIGHT/CORNER 41/25 at 0.30, 64/16 at 0.70, 65/16 at 2.00). The plateau is
the evidence, and 64 STRAIGHT is what the gallery actually shows.
`PIVOT_TRAVEL_M = 0.70`. (`tools/plot_trajectory_gallery.py` keeps 0.30 on
purpose: display-only, and its figures are committed.)

**The cheap route predicts WHERE a plan goes, not HOW it turns** — validated
against all 100 recorded plans before being trusted, which is the whole reason
the screen is not what it was first written to be:

| descriptor | corr(PMP, predicted) |
| --- | --- |
| `length` | **+0.99** |
| `straightness` | **+0.96** |
| `total_abs_turn` | +0.30 |
| `sign_changes` | +0.34 |

Smoothing the 8-connected lattice staircase does not rescue it (+0.32 at best).
Label agreement rises 15% -> 52% with smoothing, but **only because both
distributions shift toward STRAIGHT** — agreement without predictive power, and
exactly the kind of number that would have looked like success. **So screening
on predicted tortuosity is out**; `screen_score` ranks on blocked line of sight,
detour and pivot demand only, and shape is labelled afterwards from the SOLVED
plan. Do not re-propose ranking candidates by predicted turning.

The two screening constraints, both the user's: the straight line start->goal
must be **blocked** (a distance filter cannot work — a long straight corridor
passes it perfectly, which is how the current library became ~64% straight
lines), and the start/goal **headings force in-place rotation**, which is the
realistic case for a real deployment.

### Serializing VM work: the job queue (2026-08-14)

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
`rl_*` entity pose. `variance_probe --trace-dir` writes one per rollout, and so
does `soak.py --trace-dir` (with `--trace-every N` to subsample a long soak).

**The two write DIFFERENT layouts, and that is the reason to score from the
JSONL rather than the directory tree.** `variance_probe` writes
`<dir>/<shape>_<pid>_<i>.csv` and is normally pointed at
`<root>/<arm>/<shape>/`; `soak.py` writes one flat directory with the gains in
the *filename*, because it cycles gain pairs within a single run. So
`tools/score_sweep.py` has two pairing modes — a directory walker for the
former, and `--from-jsonl`, which keys on each row's `trace` field, for the
latter. **Prefer `--from-jsonl`**: a layout is a convention and conventions
drift, but a row records where its own trace actually went. Scoring shape A's
track against shape B's plan produces large, plausible, meaningless numbers
rather than an error, so the pairing is worth making unguessable.

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
8. ~~Generate interesting trajectories by construction.~~ **BUILT 2026-08-14**,
   see "Generating evaluation trajectories by construction". The A*-as-proxy
   caveat below turned out to be the important part: the proxy predicts route
   *position* well and *turning* not at all, so shape is labelled from the
   solved plan instead. Original note: The 100 recorded plans came from random
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
- **Figures are dated, indexed and committed** — `figures/YYYY-MM-DD/{render.py,
  README.md,*.png}`, convention in [figures/README.md](figures/README.md). This
  reverses the usual "version the tool, not its output" rule on purpose: these
  render from gitignored data directories, several of which cost hours of machine
  time on a plant that will not exist forever, so the renderer alone does not
  reproduce the picture. `tools/plot_*.py` keeps the old rule. Everything from
  before 2026-08-13 is in `figures/archive/` with what provenance survives.
- `acados/` at the repo root is untracked scratch; the Makefile's `ACADOS_*` /
  `t_renderer` bits are vestigial and unset by default.
