# Running the planner and controller

This guide runs the navigation pipeline from the paper in the Gazebo emulator:

```
map ─► FM² vector field ─► PMP trajectory ─► playback + TVLQR feedback ─► 4 wheel speeds
```

1. **Vector field** (`vector_field` node). From the occupancy map it computes the
   Fast Marching Square arrival-time field to the goal and its direction field.
2. **Trajectory** (`pmp_planner` node). It solves the Pontryagin boundary-value problem
   for the skid-steer model along that field and returns the whole trajectory: poses,
   wheel speeds and costates every 0.1 s. This happens once per goal, before the
   robot moves. The robot stands still while the plan is solved.
3. **Execution** (`runtime_corrector` node). It plays the planned wheel speeds back
   in time order and adds a time-varying LQR correction computed from the pose
   error. It is the only node that commands the wheels.

## 1. Install and build

Ubuntu 24.04 with ROS 2 Jazzy and Gazebo Harmonic. A GPU is optional; without one, run
headless.

```bash
sudo apt update && sudo apt install -y git make
git clone --recursive https://github.com/self-navigation/agx_navigation.git
cd agx_navigation
make setup        # once: installs ROS 2 Jazzy, Gazebo Harmonic and system packages
make              # rosdep + pip dependencies, then colcon build
make test         # unit tests of the pure Python modules, no ROS needed
```

On a fresh machine `make setup` takes about 25 minutes; most of it is pip downloading
PyTorch and its CUDA libraries (3-4 GB). If a download fails, run `make setup` again: the
steps that already finished are repeated quickly. apt and pip use the usual proxy
settings, if you need one.

`make` sources ROS for you. In a new shell, run `source install/setup.bash` before any
`ros2` command.

## 2. Start the emulator with the paper's configuration

First, in every shell you use from here on, keep ROS traffic on this machine:

```bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 daemon stop
```

Otherwise the nodes also talk to any other ROS 2 system on your network in the same
domain. A foreign `/map` then replaces this one, and every goal is reported as outside
the map. `ros2 daemon stop` matters: a daemon started earlier keeps the old setting.

```bash
make fixture FLOOR_NUMBER=6 CORRECTOR=tvlqr SURFACE_PATCHES=false
```

This starts Gazebo with floor 6 of the building, the pre-built map of that floor,
ground-truth localization, and the three nodes above. With a display, RViz opens with
the map, the field and the plan. The stack is ready for a goal when the log shows
`VectorFieldNode (FM2) ready` followed by `Map received: 1200x1200 res=0.050 m/cell`.

- `CORRECTOR` is `identity` by default, which plays the plan back without feedback.
  The paper's controller is `tvlqr`.
- `SURFACE_PATCHES=false` removes the default low-friction patches. They sit under the
  spawn point, so leave them off for a first run.
- Add `HEADLESS=true` to run without the Gazebo window.

## 3. Send a goal

In a second terminal (with the `export` and `ros2 daemon stop` above), publish a goal in
the `map` frame. This one is known to solve from the spawn point:

```bash
ros2 topic pub -w 2 --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: -8.65, y: 1.87}, orientation: {w: 1.0}}}"
```

`-w 2` waits until both subscribers have matched (the vector field and the runtime
corrector, which asks the planner for the trajectory). A goal published before that is lost.

To let the emulator pick a reachable goal 2-15 m away instead:

```bash
ros2 run agx_bringup random_goals --ros-args -p use_sim_time:=true \
    -p floor_number:=6 -p count:=1 -p rng_seed:=<n>
```

The same `rng_seed` always picks the same goal. Its "goal reached" message only means
that the goal has ended, successfully or not; read the outcome from the stack's log.

Not every goal can be solved. For some start/goal pairs the boundary-value solver gives
up (`BVP solve failed ... The maximum number of mesh nodes is exceeded`); the planner
then reports a failure and the robot does not move. Try another goal.

Restart the fixture before every goal. Wheel odometry is never reset, so a second goal
plans from a start pose that has already drifted, and after a goal has ended the
`-w 2` publish can wait forever.

What happens next:

1. The field appears in RViz within a few seconds.
2. The planner solves the trajectory. This takes seconds to a minute, depending on the
   route. The planned path is drawn when it is complete.
3. The robot drives. The controller's per-tick state (tracking errors, correction,
   gain, side-slip parameter) is published on `/wheel_corrector/tvlqr_diagnostics`.

The planner's `Arrived near goal (d_xy=...)` refers to the planned trajectory, before
the robot moves. The drive ends with `Trajectory 1 finished (success=...). IDLE.` from
the runtime corrector, which then publishes zero wheel speeds. To see where the robot
stopped, run `ros2 run tf2_ros tf2_echo map base_link` and compare with the goal.
"Arrived" means within 0.2 m of the goal (four times the 0.05 m planning tolerance).

Watch the main signals:

```bash
ros2 topic echo /wheel_velocity_controller/commands     # the four wheel speeds
ros2 topic echo /wheel_corrector/tvlqr_diagnostics      # errors and correction
```

## 4. Options used in the paper

| what | how |
| --- | --- |
| side-slip term with a fixed k | `FIXTURE_EXTRA_PARAMS="tvlqr_params:=sideslip_k=-0.10,sideslip_v_gate=0.3"` |
| adaptive TVLQR (online k) | `FIXTURE_EXTRA_PARAMS="tvlqr_params:=tvlqg.enabled=true,sideslip_v_gate=0.3"` |
| adaptive TVLQR with a prior k | add `,tvlqg.k0=-0.10,tvlqg.k_plan=-0.10` to the line above |
| AMCL localization (lidar) | `LOCALIZATION=amcl` |
| RTAB-Map localization | `LOCALIZATION=rtab` |
| SLAM from scratch, no pre-built map | `make run SIM=true FLOOR_NUMBER=6 CORRECTOR=tvlqr` |

As in section 3, restart the fixture for each run.

## 5. Reproduce one experiment run

The experiments ran each route through `tools/compare_run.py`, which does the
following:

1. starts a fresh stack with the robot at the route's start;
2. spawns the low-friction patches for the run's seed;
3. sends the goal and waits for the end of the run;
4. scores the run from Gazebo ground truth.

It writes one JSON row per run and a track file. For example, one run of the
high-speed replication (fixed k, ground truth, calibrated tire, mu = 0.5 patches):

```bash
tools/with-worker 1 python3 tools/compare_run.py \
    --plan <dataset>/plans/v090/f6_long_0161.npz --arm ours-lib --worker 1 \
    --localization truth --phantom-walls --patch-mu 0.5 --wheel-slip-lat 1.0 \
    --tvlqr-params sideslip_k=-0.10,sideslip_v_gate=0.3 --seed 3 \
    --out rows.jsonl
```

- `--arm ours-lib` replays a stored plan instead of solving it again, so the
  controllers are compared on identical trajectories.
- `--phantom-walls` keeps the walls visible to the lidar but removes their collision
  geometry, so a run continues after a wall contact. The contact is scored afterwards
  from the robot's footprint against the map.
- `tools/with-worker N` isolates the run in its own Gazebo partition and ROS domain,
  so several workers can run side by side.

The exact command line of every experiment cell is in the `config.txt` of its directory
in the dataset record.

## 6. Solve plans without the emulator

Plan sets are built offline, with no ROS and no Gazebo:

```bash
python3 tools/build_plan_set.py sample --out plans/ --floors 6
python3 tools/build_plan_set.py solve  --out plans/ --jobs 8
```

Each solved plan is an `.npz` file with the poses, wheel speeds and costates on a 0.1 s
grid. The format is described in the dataset record's `plans/README.md`.

## 7. On the robot

```bash
make run SIM=false CORRECTOR=tvlqr
```

This runs the same pipeline on an AgileX Scout Mini over CAN (`PORT_NAME=can0`), with
RTAB-Map SLAM for localization. The physical robot takes body velocity commands
(v, omega), not individual wheel speeds; the chassis node converts between them.
