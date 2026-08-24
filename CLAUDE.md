# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

Only **one sim per partition** (since 2026-08-15 — see "Parallel sims:
`WORKER`"; before that it was one sim, full stop, and everything below describes
what still happens when two land in the *same* partition). Two instances sharing
one transport partition both advertise `set_pose`/`pose/info`, so resets silently
break.

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

**Split rule (2026-08-13, enforced again 2026-08-24):** superseded tables and
resolved-bug narratives live in
[docs/corrector-history.md](docs/corrector-history.md), not here. The axis is
**"does this still constrain a decision?"**, not chronology — a retraction that
stops someone re-proposing a dead idea is *live* however old it is, and gets a
one-line stub under "Settled" below. What moves out is the investigation behind
it and any measurement taken on a plant or rig that no longer exists. When you
retire something, leave the stub: an unstubbed removal means the idea comes
back. Date each claim, and delete a claim outright when it is superseded rather
than leaving both versions.

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

**Plant and measurement validity**

- **Nothing measured before 2026-08-02 evening is valid.** `WorldControl.pause` is a proto3 bool, so every `multi_step` silently un-paused the world and it free-ran between control steps. Fixed; `_ensure_paused` verifies the clock stopped.
- **Nothing measured before 2026-08-04 is comparable with anything after**, and again before **2026-08-07**: the patch friction distribution was re-weighted, then the wheel's `mu2` moved 0.7 → 0.45. Both were deliberate plant changes.
- **The old run-to-run variance was patches silently not spawning**, not an 8 mm reset slide (that theory is retracted). With the world paused, `create` returns False while creating the entity anyway. `_wait_entities` now confirms via pose/info; **never re-issue a create for a 'missing' patch** — it is in flight, and re-creating genuinely fails.
- **Terrain patches are inherited across processes**: the bridge now sweeps `rl_ground` and `rl_patch_0..7` by name at construction, because the sim outlives any one trainer. Tables measured before 2026-08-01 ran on someone else's leftover terrain.
- **`reset_world=True` DESTROYS THE ROBOT — never use it.** A gz `WorldControl.reset.all` deletes runtime-spawned entities, including the `scout_mini`. Recovery is `just kill-sim` + `just remote-sim`, not debugging.
- **The wheel-velocity residual was NOT the determinism seed** (wheels agree to 1e-16); the un-gated IMU read was, and is now gated.
- **Do not chase the last decades of initial-state agreement.** A ~1e-6 m settle residual is amplified to metres in ~30 steps on reversal-heavy shapes; a tighter tolerance is the wrong instrument against an exponential.

**Metrics and estimators**

- **`final_err` is NOT reproducible and must never be a tuning objective** (sd 0.26 on the trajectory it was measured on). Genuine chaos at turn reversals, not a bug — the ROS-publish-vs-gz-step race and terrain differences were both ruled out. (It *is* fine as an aggregate over 40 plans — see "Reading a gain result".)
- **The 0.0002 m noise floor was ONE trajectory's and does not generalise.** Nelder-Mead cannot converge on a noisy objective — it re-sampled one point 71 times. Superseded by mean-of-3 repeats.
- **The median-of-3 estimator LOSES to the mean-of-3** — the 7-trajectory aggregate already dilutes an outlier 7-fold, so the median pays its variance penalty for nothing. Mean-of-3 to search, mean-of-5 to validate.
- **Every "run-to-run variance" number written before 2026-08-13 is a mixture width, not a measurement error.** Several shapes are bimodal; the right estimator for them is a mode frequency over ~100 samples, not a mean over 3.
- **`J` needs the GEOMETRIC mean across trajectories.** `J` spans 0.2 to 1043 across a library sweep and one plan can carry 48% of the arithmetic mean, so an arithmetic search on `J` tunes to whichever plan is worst. `objective.DEFAULT_HOW` encodes it: arithmetic for `max_cross`, geometric for `j_total`.
- **A per-shape claim must name its baseline** — computing one against sweep neighbours rather than the default produced a retracted attribution.
- **An automatic shape label is a ranking aid, never a claim.** Two labellers have now misled here (`classify_plans.py` called 58 of 100 plans CORNER; `total_abs_turn` cannot separate a hairpin from two same-sign corners). **Render the plans before making any per-shape claim.**
- **Screening candidate start/goal pairs on PREDICTED turning does not work** — the cheap A*/lattice route predicts *where* a plan goes (r=+0.99 on length, +0.96 on straightness) and not *how it turns* (+0.30). Shape is labelled from the solved plan instead.

**Gains and tuning (closed — see "The gains are settled" below)**

- **A seven-plan gain search cannot resolve the gains, in either currency.** Two of them (`max|e_cross|` in 2026-08-12, `J` in job 60) produced optima that evaporated on 40 independent plans. Do not run another one. Validate on the broad set.
- **`J` is the right objective and a poor discriminator**: it ranks the move off the old default decisively (1.49x, 5/40) and cannot separate anything *inside* the plateau (every within-plateau p is 0.08–0.88).
- **`q` and `r` INTERACT**, so every `r_omega` claim is scoped to the `q_cross` it was measured at. The `r=0.5→1.0` threshold that once justified moving off `r=0.25` is a `q=0.276` phenomenon and is absent at `q=1.5` and `q=2.5`.
- **The U-turn "basin" with near-vertical walls is a property of `floor_6_00031`'s exact plan, not of U-turns** — four other U-turn plans (one a near-duplicate of its route) show no `q` dependence at all. Never restate it as a shape claim.
- **RL checkpoints show no learning trend on the real task** (r=0.111 over 20 checkpoints; 0 of 20 beat TVLQR). The TB metric that looked like progress was logged by the mis-stepped env. Quote 800k as *best* checkpoint, never as typical.
- **The 20260730 SAC run (1.5M steps) learned nothing usable** — 2 successes in 8864 episodes, `ent_coef` ran away to 3.31, `critic_loss` 1.2e4. Fix the entropy target and bound the return before any retrain (queue 1-2).

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
  `j_total`, so scoring from trace files is no longer the route to a number.
  `tools/score_epsilon.py` remains for already-recorded runs; the two paths
  differ where a wheel saturates (online takes the correction before clipping).
  **`R` charges the CORRECTION, not the total command** — the nominal command is
  what the planner already paid for.
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
  `max|e_cross|`, is the currency the write-up must report — that is why
  `tuning/epsilon.py` exists.
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

### A failed plan hung every client (fixed 2026-08-18)

`TrajectoryBuffer.active_traj_id` is set in `_on_chunk`, so a plan that failed at
**chunk 0** never set it, `_on_action_result` dropped the result as a mismatch,
and `_on_tick`'s idle guard returned before `_finish()` could publish the zero
and the completion sentinel. The stack went silent with the robot stationary and
every client waited out its own timeout. Fixed in `_on_action_result`; verified
live (a failed goal now advances the driver 0.4 s later).

Two things that outlive the fix:

- **BVP mesh-node exhaustion fails ~36% of fresh start/goal pairs** — the same
  rate measured building the v2 library. That is not only a data-generation
  problem; it is a runtime failure mode on ordinary goals, and a goal the
  planner cannot solve must degrade **visibly**.
- **The failure is a property of the start/goal PAIR, not of the goal.** The
  same goal failed from one start pose and was reached from another.
- Method note: the pipeline itself was healthy and the bug was in the failure
  path. **A stack that works is not evidence about what it does when a component
  says no.**

### Choosing evaluation trajectories

`config/eval_trajectories.yaml` holds the seven-shape working set and a
candidate list — kept for per-shape diagnosis, but **the 40 broad v2 plans are
what a gain or controller decision is read from** (see "Reading a gain result").
`just gallery` renders all 100 plans (`figures/trajectory_gallery.png`), each
rotated onto its principal axis so shape is comparable at a glance.

**The automatic labels mislead.** `classify_plans.py` calls 58 of 100 CORNER, but
in the gallery most of those are visually straight lines: the descriptor is
tripped by the in-place reorientation the PMP planner puts at the *start* of a
plan — a large heading change over no distance. Trust the picture. Likewise
`floor_6_00042`, used as "the S-curve" in every comparison so far, is really an
L with one rounded bend. Genuine S-curves: `floor_6_00028` (cleanest),
`00024`, `00047` (zigzag), `00056` (tight V). A true U-turn: `floor_6_00031`.

### Parallel sims: `WORKER` (built and verified 2026-08-15)

**Several Gazebos now run on one box, and the "only ever ONE sim" rule is
retired** — replaced by "only ever one sim *per partition*". Everything above
that says otherwise is describing the default partition, where it remains true.

`WORKER=n` (1-9) is the single knob. It sets `GZ_PARTITION=agxn` and
`ROS_DOMAIN_ID=40+n`, and that is the entire mechanism: **no code changed
anywhere** — not in `GazeboBridge`, not in the launch files, not in the tuner.
Both libraries read their isolation setting from the environment at init, so a
`gz_transport.Node()` and an rclpy node constructed by unmodified code land in
whichever world their process was started in.

- `make rl-sim WORKER=1` / `make rl-train WORKER=1` / `make fixture WORKER=1`
- `just remote-sim rl_corrector.world 1`, `just remote-fixture tvlqr true truth 1`,
  `just gui 1`
- `tools/with-worker 1 python3 -m agx_planning.tuning.soak …` for anything not
  going through make. **`WORKER` unset execs with the environment untouched** —
  not `GZ_PARTITION=""` — because every number in this file was measured in the
  default partition and a change that silently moved it would invalidate the lot.

**Both variables are required and they cover different failures.** Without
`GZ_PARTITION` two `gz sim` servers both advertise `/world/rl_corrector/set_pose`
and resets go to whichever answers first. Without `ROS_DOMAIN_ID` both stacks
publish `/clock`, `/joint_states` and the wheel command topic, and each robot
receives the other's commands — *that* is the documented "wheels detach, links
fall through the floor" failure, and it is ROS-side, so **`GZ_PARTITION` alone
does not prevent it**.

**Verified live against a running job**, not in isolation — worker 1 was brought
up beside the sim job 60 was driving:

| check | result |
| --- | --- |
| gz topics visible per partition | 14 in default, 14 in `agx1`, no overlap |
| ROS topics | 36 in domain 0 (full stack), 21 in domain 41 (minimal sim) |
| entities in each world | 52 default (job 60's patches), 20 in `agx1` |
| **worker rollout `floor_6_v2_00004`** | **0.2901, 0.2901** vs the default partition's **0.2901 ± 0.0001** over 60 |
| job 60's per-eval time | 103 s before, 103 s during |

**The four-decimal agreement is the real result**: a worker is the same plant,
so numbers measured in parallel are comparable with everything already in this
file. Load went 3.5 → 7.8 on 12 cores with two sims plus a soak.

`just check-sim` is **scoped by partition** and defaults to `default`, so it is
exactly as strict as before for anyone not passing a worker, while worker 2 can
start beside worker 1. It now shells out to `tools/kill_stack.sh` in a new
`list` mode instead of running its own `pgrep`, so **the guard and the sweep can
no longer disagree** about what counts as a conflicting process. `just kill-sim`
takes a partition too but defaults to `all` — someone typing it after a bad
night wants everything gone, and having to enumerate partitions is the kind of
step that gets skipped.

Two traps found while building it, both of which produce a *convincing* wrong
answer rather than an error:

- **`ps -p "1,2,"` prints nothing, silently.** `tr '\n' ','` leaves a trailing
  comma, so the guard printed "REFUSING TO LAUNCH … (see above)" with nothing
  above it, and the pre-existing `STILL RUNNING:` diagnostic had the same bug —
  meaning a failed sweep had been reporting an empty list all along. Use
  `paste -sd,`.
- **A process can be isolated on the ROS side alone.** `ROS_DOMAIN_ID=41 ros2
  topic list` leaves a daemon with no `GZ_PARTITION`, which reads as "default"
  and blocks `check-sim default` over a process sharing nothing with it. The
  filter checks both variables; worker processes always carry both.

Not done, and the next thing to want: **the job queue is still single-lane**.
`tools/jobq.sh` runs one job at a time in the default partition, so parallelism
today means driving workers by hand. Per-worker queues are the obvious follow-up
and are what would turn ~55 core-hours of PMP labelling into an overnight run.

### The constructed plan library, and the first PMP solve cost (2026-08-15)

`tools/jobs/20_generate_v2_library.sh` screened 1200 start/goal pairs per floor,
kept 500, and PMP-solved them into `~/traj_data_v2` (fetched to `traj_data_v2/`).

**320 solved, 180 failed (36%)** — 110 `Exceeded max_rollout_sim_time=60 s`,
70 `BVP solve failed … maximum number of mesh nodes is exceeded`. The screen
works as designed: the library is **100% turning shapes, zero straight lines**
(labels, unreliable as above, are 131 UTURN / 100 S / 76 CORNER / 13 ZIGZAG),
against a random-goal library that was ~64% straight.

**PMP solve cost, measured for the first time: mean 1.84 s, median 1.7 s, p90
2.5 s, max 4.3 s per plan**, single core. This is the number the RL re-planner's
data budget depends on: ~2 s a label means 100k supervised labels is ~55 core-hours,
i.e. an overnight run on a handful of cores and **no Gazebo at all**. It also
bounds the re-join solve from above — the re-join problem is strictly smaller
than a full plan — so "PMP cannot be solved online" remains a statement about
scipy's `solve_bvp`, not about the problem.

`tools/select_broad_eval.py` picks a geometry-diverse subset from a library,
stratifying on the (untrusted) label and on path length.

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

**The runner killed itself on its first job (fixed 2026-08-14).** The job body
ran inside a **brace group** ending in `exit $rc` — and a brace group is not a
subshell, so that `exit` ended the *runner*. Job 10 wrote its `EXIT rc=0` line,
the runner vanished before the `mv` to `done/`, and the next job sat pending
13 h. It is a subshell now. The signature to recognise: `status` shows a job in
**running** whose log already says `EXIT … rc=0`, and the runner NOT RUNNING —
that combination means the runner died between the two, not that the job hung.

`just queue-add` runs `sync` first, so **queueing a job rsyncs the working tree
onto the VM under whatever is currently running.** That is usually what you want
(a fix lands for the jobs behind it) but it does mean an in-flight job can pick
up edited code at its next `python3 -m`; the batch loops make that a real window.

### The current plant: patch weights, IMU gating, repeats (2026-08-04)

Three changes landed together and **re-baselined everything measured before
them**, because the friction distribution changed deliberately.

**1. The patch friction distribution is a floor, not an ice rink.** `PROFILES`
gained `linoleum` (mu 0.45, the actual deployment surface) and `wet_tile`
(0.30), and `DEFAULT_PATCH_WEIGHTS` replaced uniform sampling — which had put
**half of every patch set at or below tyre-on-ice friction**. Now ~60% at
mu >= 0.30 and 10% ice. `ground_friction_sampler` is weighted too and
**excludes the directional profiles**: a whole floor that grips one axis and
slides the other has no physical analogue. A profile with no weight raises
rather than silently never being drawn (`test_terrain_weights.py`).

**The world's own ground is still `mu=1.0`** (concrete-like) in
`rl_corrector.world`. If deployment is linoleum everywhere, the nominal
no-patch plant is grippier than reality and every gain inherits that bias. Not
changed — it is a re-baseline and wants a decision.

**2. The IMU read is gated** (`_wait_imu_advance`), matching the pose gate. The
un-gated read was the largest difference between two otherwise-identical
rollouts, by twelve orders of magnitude, and it feeds the RL observation.
`stale_imu_steps` counts giving up; the gate is only paid when `use_imu` is on,
so TVLQR work costs nothing for it. **The goal is not determinism** — the real
IMU is noisy and laggy and the policy must tolerate that. The goal is that the
sim's sensor error be a knob we choose (a deliberate latency/noise model matched
to the measured real IMU) rather than an artifact of VM CPU load. **That model
is not written yet**; when it is, it must live in the bridge/env and leave the
obs layout untouched, or the policy stops being deployable.

**3. The tuner averages repeats.** `--repeats 3` (default), reduced per
trajectory by `objective.reduce_repeats`; the cache key includes `repeats`,
`reduce` and `patch_weights`, so an old cache cannot be replayed into a run
whose numbers mean something different. Measured sd of the estimate: single
sample 0.0933, median-of-3 0.0692, **mean-of-3 0.0547**, mean-of-5 0.0413 — see
the Settled stub for why the median loses.

### The tuning machinery (`agx_planning/tuning/`)

Still the rig for any controller comparison, even though the gain search itself
is closed. `just tune-tvlqr` (detached), then `just fetch-tune && just
plot-tune`; `soak.py` / `just soak` is the batch driver and `variance_probe` the
single-rollout one.

- `simplex.py`, `objective.py`, `cache.py`, `epsilon.py`, `shape.py`,
  `trace_diff.py` are **pure and unit-tested** — no ROS, no Gazebo, no torch,
  same rule as the RL pure modules. The simplex tests are aimed at one thing:
  proving the search *minimizes*. A tuner that maximizes produces an
  identical-looking log and hands back the worst gains it found.
- **`--max-evals` bounds candidate GAIN PAIRS, never rollout length**, and
  `--max-evals 1 --repeats 5 --q-cross Q --r-omega R` is exactly a clean
  measurement at a chosen point in the same code path — that is how a tuned
  point gets validated.
- **The trajectory set is fixed, never sampled**, so candidates are compared on
  identical work. A failed rollout makes the whole evaluation `inf`, never a
  mean over survivors.
- **Failures are never cached.** `inf` means "the sim broke", almost never
  "these gains are bad": killing the tuner mid-evaluation invalidated the
  bridge's rclpy context and wrote 56 bogus `inf` evaluations in three seconds.
  They are written with `_failed: true` for diagnosis and never returned.
- **Resumable at zero cost** — the JSONL cache replays the search and re-measures
  nothing; it refuses to resume onto a different problem.
- Search runs in **log10** of both gains, so a step is a ratio and no move can
  propose a negative gain. `--q-bounds` exists because `x0` is CLIPPED into the
  box, so a probe outside it silently measures the boundary.
- Every evaluation records **per-trajectory** errors, so the landscape can be
  re-analysed per shape without re-driving anything.
- Outstanding: Nelder-Mead reports the **minimum observed draw**, which is a
  winner's-curse machine. The structural fix is a noise-aware optimizer (BO with
  a nugget, reporting the posterior-mean optimum). Partly done — the 2026-08-07
  and job-60 runs used BO — but the reporting is still by best draw.

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

**Two bugs in the traced-soak path (fixed 2026-08-14) are worth knowing because
neither could fail loudly.** `--trace-every N` subsampled by *rollout index*
against a 35-rollout gain x trajectory cycle, so `gcd(5,35)=5` traced the same 7
cells forever — subsampling is by **cycle** now. And tracing is armed by
**file**, not by rollout: untraced rollouts were appended to the previous
traced one's CSV, producing a 1084-row "track" for a 186-step plan that scored
29 m. `disable_trace()` now exists and `soak.py` calls it. The reusable lesson:
a subsample stride and a cycle length are not independent, and a trace file is a
**resource with a lifetime**, not a flag. Both failures produce data that
parses, scores, and looks like a measurement. (`variance_probe` was never
affected — it traces every rollout.)

`tuning/trace_dump.py` prints selected rows of one trace, for when a run is bad
in isolation rather than merely different from another.

### The patch friction values are still unvalidated against reality

The *distribution* was fixed on 2026-08-04; the *values* never were. **Nobody
has ever checked these against a real surface** — they were picked on a laptop,
by eye, to make the robot visibly slip. Asked about twice before and lost both
times; hence this section. Current `PROFILES`
(`src/rudn-ordjo-building/rudn_ordjo_building/surface_patches.py`) vs. real
rubber-on-surface coefficients (dry concrete 0.7-1.0, wet concrete 0.5-0.7,
tyre on ice 0.1-0.15):

| profile | mu | real-world equivalent |
| --- | --- | --- |
| `rough` | 2.5 | above dry rubber on concrete — effectively "cannot slip" |
| `directional_x/y` | 1.0 / 0.15 | grips one axis, slides the other — **unphysical**, a ground `fdir1` is world-fixed |
| `linoleum` | 0.45 | the deployment surface; sits exactly on the steering knee |
| `wet_tile` | 0.30 | below the knee — "no steering", deliberately |
| `slippery` | 0.2 | wet smooth tile / oily floor |
| `icy` | 0.05 | polished or wet black ice — real, but not an indoor floor |

**Chi, not mu, is the quantity to match** — it is what the model consumes, and
unlike mu it is measurable on the real robot (see "Measuring `slip_chi` on the
real robot"). The route is: drive the real robot on linoleum, on the ice
mock-up and on sand, get chi per surface, then tune each profile's `mu` until
the sim's chi matches. That closes the sim-to-real loop on the one parameter
the planner takes, and nothing else here does.

Still unanswered and cheap: `slip1`/`slip2` sit in a `<friction><ode>` block
while gz-sim runs **DARTSIM**, which likely ignores ODE's force-dependent-slip
parameters — meaning `mu` may be the only knob ever connected. The `icy_noslip`
profile exists solely to test this: drive across `icy` and `icy_noslip` in one
run, and identical behaviour proves `slip1/slip2` are decorative.

### `slip_chi` is a function of the SURFACE, and the steering knee is `mu2`

Measured 2026-08-05 with `slip_ident` on a real-time world, then re-measured
after the wheel fix (`sweep_data/ground_mu_chi*.csv`, driver
`tools/sweep_ground_mu.sh`). The pre-fix curve and its narrative are in
[docs/corrector-history.md](docs/corrector-history.md); the mechanism below is
unchanged and is the reason a frozen PMP plan cannot handle friction zones.

**The mechanism.** [wheel.xacro](src/scout_ros2/scout_description/urdf/wheel.xacro#L63)
gives each wheel `mu1=200.0` (rolling) and `mu2` (lateral), and Gazebo combines
two contacting surfaces by taking the **smaller** coefficient. So `mu1=200` is
never realized — it encodes "the wheel is never the longitudinal limit, the
GROUND decides", which is what makes a patch's `mu` mean anything at all in the
rolling direction. A skid-steer yaws by gripping longitudinally while scrubbing
sideways, so **the longitudinal:lateral ratio IS the steering mechanism**, and
it survives only while `ground > wheel mu2`: there the ground binds
longitudinally and the wheel binds laterally, two independent constraints. At or
below `mu2` the ground binds **both**, the ratio collapses to 1:1, and grip and
steering are lost inseparably. Provenance: Grigorii Matiukhin, 2026-02-13, in
the team's own `scout_ros2` fork — not AgileX upstream, so it is ours to change.

**The knee is at the wheel's `mu2`, to the digit**, and it moved with it when
`mu2` went 0.7 → 0.45 — predicted before the run both times. Min-combination is
settled, not a hypothesis. Above the knee chi is ~1.36-1.57 and the spread
across turn radii is small; at or below it the robot achieves **4-7% of
commanded yaw rate** and most arcs are unmeasurable.

**The free variable is the RATIO `ground/mu2`, not absolute friction** — the
curve translated rather than deformed. That is what makes
`sweep_ground_mu.sh` a reusable instrument for "does this tyre model steer".

**Chi barely moved at nominal: 1.3718 → 1.3575 (~1%).** Above the knee chi cares
*that* you are above it, not by how much — so `PlannerConfig.slip_chi = 1.373`
was still within ~1% and **the baked plans did not need re-planning** for the
wheel fix. The spread across radii improved (0.0299 → 0.0072), so a single
`slip_chi` describes this plant better than it did the old one.

**Consequences that constrain what we can model:**

- **Do not lower the ground plane to model a slippery floor.** Both worlds are
  at `mu=1.0` and should stay there; slipperiness belongs in the wheel pair or
  in a patch, and any patch below 0.45 now means "no steering", deliberately.
- **Usable band is ground >= 0.5.** Linoleum at 0.45 sits exactly on the knee
  and is marginal; every other profile is below it.
- **Ice being uncontrollable is CORRECT, not a limitation** — on real ice
  `mu_long ≈ mu_lat` and a real skid-steer genuinely cannot steer. The model was
  never broken for ice; it was broken for linoleum, because the wheel was
  parameterised for concrete. This is the SVCM dichotomy's second branch.
- **"Slides but still steers" is inexpressible at low friction** under
  min-combination with an isotropic ground, at any wheel setting. `sand` is
  blocked on this, and it is a question for the advisor.
- **Chi is a property of the SURFACE** (1.36 vs 25.4 across the sweep), so it
  cannot be a constant on a floor with ice/sand zones — it is not constant
  *within one trajectory*. That is a modelling gap, not a tuning problem.

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
library build failed 36% (110 timeouts, 70 mesh exhaustions of 500). A teacher
that answers two-thirds of the time cannot label a dataset, and that would make
the supervised plan wrong rather than slow. It is offline, needs no Gazebo, and
is cheap.

### Ideas queued, roughly in order of expected value

1. **Fix SAC's entropy runaway before any retrain.** `ent_coef` reached 3.31.
   Either pin it (`ent_coef=0.05` instead of `"auto"`) or set an explicit
   `target_entropy` — the default `-dim(A)` is far too permissive for a 4-D
   residual whose useful range is tiny. Every hour of the 20260730 run after
   ~800k steps made the policy worse.
2. **Bound the per-episode return.** Huber bounded the reward's slope, not the
   accumulated return over 200 non-terminating steps, and `critic_loss` still
   reached 1.2e4. Options: normalize the return, cap per-step cost outright, or
   reinstate termination with a large-but-finite terminal penalty (not the same
   as the 0.5 m corridor that caused the original no-recovery problem).
   `-epsilon.step_cost(...)` is the per-step integrand and is the reward any
   future RL should use.
3. **Give the RL residual a fair fight**: train it *on top of* tuned TVLQR
   rather than on top of identity, so the policy learns the residual a good
   linear controller cannot supply instead of re-deriving feedback from scratch.
   Also the most defensible version in a write-up — the advisor's requirement is
   that RL be part of the system, not that it beat everything alone.
4. **Per-worker job queues.** `tools/jobq.sh` is single-lane in the default
   partition, so parallelism today means driving `WORKER=n` by hand. This is
   what would turn ~55 core-hours of PMP labelling into an overnight run.
5. **Compare against Nav2 baselines (DWA, MPPI, TEB)** — carried over from the
   paper draft's Experiment 4, which named them and was never run. Metrics
   proposed there: path length, travel time, max curvature, and control energy
   `int ||(a_l, a_r)||^2 dt`. The comparison is not like-for-like: those are
   closed-loop planners, so the comparable object is the whole FM2+PMP+TVLQR
   stack, not the corrector alone.
6. **Widen the search to the full Q/R diagonal** (`q_along`, `q_heading`,
   `r_v`) — the 2-D machinery is proven, but note the 2-D result needed 40 plans
   and mean-of-5 to resolve, so budget accordingly and validate on the broad set.
   Low priority: `J` could not separate points inside the 2-D plateau at all.
7. **A `sand` profile.** The advisor wants ice *and* sand. Blocked on the
   modelling limit below, not on parameter choice: a Coulomb `mu` alone does not
   model granular flow, and under min-combination any ground below the wheel's
   `mu2` has no steering authority at all.
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
- **Figures are dated, indexed and committed** — `figures/YYYY-MM-DD/{render.py,
  README.md,*.png}`, convention in [figures/README.md](figures/README.md). This
  reverses the usual "version the tool, not its output" rule on purpose: these
  render from gitignored data directories, several of which cost hours of machine
  time on a plant that will not exist forever, so the renderer alone does not
  reproduce the picture. `tools/plot_*.py` keeps the old rule. Everything from
  before 2026-08-13 is in `figures/archive/` with what provenance survives.
- `acados/` at the repo root is untracked scratch; the Makefile's `ACADOS_*` /
  `t_renderer` bits are vestigial and unset by default.
