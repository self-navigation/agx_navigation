# Experimental field updates in ROS

The vector field node can use the experimental incremental backend. The default
remains `cached`, with early exit enabled. The node creates one updater at
startup and keeps it across map and goal callbacks.

These settings are read once. Pass them at startup and restart the node to
change them; runtime parameter changes do not rebuild the updater.

| Node parameter and launch argument | Default | Allowed values |
| --- | --- | --- |
| `field_backend` | `cached` | `cached` or `incremental` |
| `field_checkpoint_interval` | `10000` | Integer greater than zero |
| `field_checkpoint_growth` | `1` | Integer at least one |
| `field_max_checkpoints` | `4` | Integer at least zero |
| `field_cache_mib` | `256` | Integer at least zero |

The node validates these settings during startup, including in cached mode.
The cache limit covers retained native FMM arrays, not total process memory.
Zero checkpoints or zero cache MiB disables FMM replay. Incremental map speed
updates can still run. Growth 1 keeps fixed checkpoint intervals; larger growth
spreads the retained points. Keep 1 unless measurements support another value
for the expected edits. See [fmm_state.md](fmm_state.md).

The launch argument `field_early_exit` defaults to `true` and sets the existing
node parameter `early_exit_enable`. FMM replay needs complete fields, so set it
to `false` explicitly. With early exit enabled, the incremental backend uses
the same partial FMM solve and reports `full_recompute/early_exit`.

## Run the field node directly

Build and source the ROS workspace as usual. Its Python environment must use
the pinned scikit-fmm fork; see [fmm_state.md](fmm_state.md) for the rebuild step.
With `/map`, `/goal_pose`, and the `map` to `base_link` transform available:

```bash
ros2 run agx_planning vector_field --ros-args \
  -p field_backend:=incremental \
  -p field_checkpoint_interval:=10000 \
  -p field_checkpoint_growth:=1 \
  -p field_max_checkpoints:=4 \
  -p field_cache_mib:=256 \
  -p early_exit_enable:=false \
  -p centring:=true \
  -p speed_profile:=exponential \
  -p speed_decay_rate:=1.5 \
  -p allow_unknown:=true \
  -p map_frame:=map \
  -p robot_frame:=base_link \
  -p use_sim_time:=false
```

The example selects the centred speed profile and unknown-cell policy used by
the planning launch configuration. These are explicit choices; a plain direct
node invocation keeps its existing field defaults.

## Use the planner launch file

```bash
ros2 launch agx_bringup vec_pmp.launch.py \
  sim:=false \
  plan_centring:=true \
  field_backend:=incremental \
  field_early_exit:=false \
  field_checkpoint_interval:=10000 \
  field_checkpoint_growth:=1 \
  field_max_checkpoints:=4 \
  field_cache_mib:=256
```

Supply `sim:=false` when launching this file directly with wall-clock time.
Use `sim:=true` only when the system supplies simulation time. This launch also
starts the planner and runtime corrector. The existing corrector default is
`identity`; the field settings do not change it. The launch keeps `map` and
`base_link` as the planning and robot frames.

## Read the logs

Startup logs show the backend, early-exit setting, checkpoint settings, and
cache budget. A computed incremental field also logs speed and FMM status and
reason, plus `reused_batches`. For example:

```text
speed=local_update/changed_region; fmm=replay/checkpoint_replayed, reused_batches=10000
```

A replay can reuse zero batches. Source initialization changes and missing
safe checkpoints use full FMM computation. A `field_reuse` request does not
repeat stale stage diagnostics. Successful requests still publish planner data
first, then the cost map and arrows; the path keeps its existing timer.

Tests cover configuration, startup wiring, numerical callback results, cache
clearing, and publication order with ROS stubs. Launch syntax and typed argument
wiring are checked without a live ROS system. Live ROS transport and timing
have not been tested here.
