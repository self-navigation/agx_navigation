# Persistent field updates

`FieldUpdater` keeps one prepared map and one field. Its default mode uses
cached map data and results. Experimental local speed updates and FMM replay use
`FieldUpdater(incremental=True)` and the benchmark's `incremental` backend.
The ROS field node keeps the default cached mode.

| Request | Default work | `last_update` |
| --- | --- | --- |
| First request, changed obstacle mask, geometry, policy, or field settings | Full field computation | `full_recompute` |
| Same prepared map, new goal or early-exit target | FMM and field derivatives; reuse speed and signed wall distance | `map_reuse` |
| Same prepared map, goal, and early-exit target | Copy the cached result | `field_reuse` |
| Invalid goal or early-exit target | Clear the cache and return `None` | `invalid` |

Map comparison uses the effective obstacle mask. A changed occupancy value
that stays on the same side of the threshold does not require a new field.
The unknown-cell policy is included. All field settings are part of the
cache key, even settings that only affect derivatives. The origin, resolution,
and shape are checked as well.

With early exit disabled, robot motion does not change the field. With early
exit enabled, a changed robot cell triggers a new FMM solve. The updater does
not assume that a cached partial field covers a new robot position.

Results are copied on return, including arrays and bottleneck data. A caller
can edit its result without changing later requests. Input maps and settings
can also be edited between calls. One instance handles one serial request
stream; it has no thread synchronization. `clear()` releases its cached data.
Exceptions also clear the cache and are raised to the caller.

## Experimental field updates

With incremental mode enabled, map edits still run both global distance
transforms and rebuild signed wall distance. Compatible requests with centring
enabled and `centring_smooth=0` update speed in affected rectangles. Cap and
resolution must be finite and positive. Other settings, an incompatible
previous map, or source crops whose total area reaches the map area use full
speed construction.

Affected regions start from changed mask or clipped free-distance values.
They include the influence of the centring level radii. A second halo supplies
all centres needed to recompute each region with the existing speed function.
Both bounds include its `1e-9` distance tolerance. Empty centre sets use the
existing stop rule. This keeps the global EDT algorithm and the centring
formula unchanged.

Separated edits can use up to four output rectangles. The updater splits at
empty row or column gaps only when the outputs stay disjoint and the total
source crop area decreases. Source halos may overlap. If splitting does not
reduce work, it keeps one rectangle. The split order is deterministic.

The controller then tries to reuse a prefix of the previous FMM march. It
recomputes field derivatives from the resulting complete travel-time field.
Local speed construction and FMM replay are separate: a local speed update may
still need a full FMM solve. The second-order stencil and gradient rules are
unchanged.

Replay supports complete fields only. Early exit uses a full FMM solve for
the requested partial field. Local speed updates can still apply.
A new goal, geometry, policy, or field setting starts a new tracked solve.
If a map edit changes initialization or no retained checkpoint is safe, the
controller also runs a full solve. Obstacle additions, removals, and mixed
changes use the same checks; there is no assumption that travel times can only
increase.

The native state records the first batch that reads each cell's inputs.
It checks new initialization exactly and selects a checkpoint before the first
affected read. Every changed cell must still be Far or Mask with no heap ID at
that checkpoint. Accepted replay restores the checkpoint, applies the changed
inputs, and finishes the march. See [fmm_state.md](fmm_state.md) for the native
checks and checkpoint rules.

`last_update` is `fmm_replay` after accepted replay, or `fmm_reuse` when the
complete FMM inputs are unchanged. Other requests use the modes in the table.
`last_fmm` records the FMM status and reason when the FMM controller runs. It is
empty for a copied field result. `last_speed` records local speed construction,
full speed construction, or unchanged speed inputs when incremental map
preprocessing runs. It is empty for the default backend and when map
preprocessing is skipped. A `full_recompute` field mode can still use local
speed construction; inspect the two stage reports separately.

## Python use

Use the environment from [replanning_baseline.md](replanning_baseline.md).
Set `PYTHONPATH=src/agx_navigation/agx_planning` for direct imports.

```python
from agx_planning.vector_field import FieldUpdater, VectorFieldConfig

updater = FieldUpdater(incremental=True, checkpoint_interval=10000,
                       max_checkpoints=4, cache_bytes=256 * 1024**2)
cfg = VectorFieldConfig(early_exit_enable=False)
outcome = updater.update(occupancy, goal_col, goal_row, resolution,
                         origin_x, origin_y, cfg)
if outcome is not None:
    result, message = outcome
    print(updater.last_update, updater.last_speed, updater.last_fmm, message)
```

The arguments and return value of `update` match `compute_field`. Keep the same
updater for later requests. Use `FieldUpdater()` for the default cached mode.

The checkpoint interval counts equal-time batches, including each batch's
neighbor updates. It must be positive. The checkpoint count and byte budget
must be nonnegative. Setting either the count or budget to zero disables FMM
replay and uses full solves. The defaults are 10000 batches, four checkpoints,
and 256 MiB. Checkpoints use this fixed interval up to the count and byte limits.
Accepted replay drops all previous revision checkpoints. New checkpoints start
at the reused point, so a later edit that affects earlier work may need a full
solve.

The budget covers retained native working arrays and checkpoint arrays only.
It includes dependency history and the initialization signature. It excludes
temporary initialization arrays, preprocessing data, returned field arrays,
Python objects, and other process memory. It is not a process memory limit.

The ROS field node owns one default updater and reports the selected mode in
its log. It still publishes each successful request, including cache hits.
A removed or invalid goal clears the node's current result. The message format
is unchanged; this does not revoke fields already held by downstream
subscribers. The callback tests use ROS stubs. Live ROS transport is a separate
check.

## Compare against full recomputation

The benchmark defaults to `--field-backend full`. Select `cached` for map and
result caching, or `incremental` for local speed updates and FMM replay. All modes create
a fresh planner per snapshot. An update may supply a new `goal`; otherwise it
keeps the previous goal.

```bash
.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/field_updates.yaml --field-only --save-fields \
  --field-backend full --out run_data/fields-full

.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/field_updates.yaml --field-only --save-fields \
  --field-backend incremental --checkpoint-interval 100 \
  --max-checkpoints 4 --field-cache-mib 256 --out run_data/fields-incremental

.venv-baseline/bin/python tools/summarize_replanning.py \
  run_data/fields-full run_data/fields-incremental --out run_data/fields-summary
```

The fixture covers repeated inputs, a moved goal, obstacle insertion,
closure, and removal. Use `--warmups 1 --repeat 10` for a local timing run.
Each case starts with an empty cache in every repetition, including warmups.
Separate cases and repetitions cannot supply cached work to each other.

For a local edit on the 1200 by 1200 floor-6 map, replace the config with
`config/replanning/floor6_field_updates.yaml`. It adds an eight-cell obstacle,
repeats that map, then removes the obstacle. This small fixture checks local
speed work when centring changes source speeds and FMM must restart. It does
not represent route coverage or a moving robot experiment. Compare `added`
and `removed` separately from the initial solve and repeated-input cache hit.

`compute_field_s` includes mask construction and comparison, cache lookup,
numerical work, and copies of the returned result. `speed_s` includes change
detection, cropped or full speed construction, and speed array copies. Nested
speed and EDT calls are counted once. `fmm_s` includes replay
validation, checkpoint work, the remaining march, and result export. Stage
times are zero for skipped stages. `field_update` records the mode per sample.
When the FMM controller runs, `fmm_update` contains:

| Key | Meaning |
| --- | --- |
| `status`, `reason` | Replay, unchanged FMM inputs, or full recomputation and its cause |
| `changed_cells` | Cells with changed effective speed or mask inputs; speeds masked in both maps are ignored; null for stateless fallback |
| `reused_batches` | Completed batches retained from a safe checkpoint |
| `first_affected_batch` | First affected input-read batch, -1 when no such read was recorded, or null for stateless fallback |
| `total_batches`, `computed_batches` | Final batch count and batches computed for this request; null for stateless fallback |
| `checkpoint_count`, `cache_array_bytes` | Retained checkpoint count and native array payload |

Incremental map preprocessing also records `speed_update`:

| Key | Meaning |
| --- | --- |
| `status`, `reason` | `local_update`, `full_recompute`, or `unchanged`, and its cause |
| `changed_cells` | Changed clipped distance or mask inputs; null when not checked |
| `updated_cells` | Total area of disjoint output rectangles replaced in the speed array |
| `computed_cells` | Sum of source crop areas passed to speed construction, counting overlapping halos |
| `total_cells` | Full map area |
| `region_count` | Number of speed crops: up to four for local updates, one for full construction, zero for unchanged inputs |

JSON summaries include backend, field mode counts, and FMM and speed status and
reason counts. A full solve with reason `new_field` starts a tracked field;
other full-solve reasons explain why replay was unavailable. Map loading and
rectangle application remain outside sample timing. Cache release at case
boundaries is setup work.

Compare matching snapshots and inspect failures. A repeated-input cache hit
does not measure the cost of an obstacle update. The closed passage still
reports `start_unreached`. Saved raw arrays permit direct equality checks.
A replay may reuse zero batches, including when centring spreads speed changes
near the source. Check `reused_batches` and total measured cost; replay status
alone does not show a time saving. Measurements on these small fixtures do not
establish performance across routes or map changes.

## Validation

```bash
PYTHONPATH=src/agx_navigation/agx_planning .venv-baseline/bin/python -m pytest -q \
  src/agx_navigation/agx_planning/test/unit/test_field_update.py \
  src/agx_navigation/agx_planning/test/unit/test_field_node_update.py \
  tools/tests
```

Tests compare field channels and reached masks exactly against fresh solves.
They cover map edits, centring, smoothing, bottleneck shaping, early-exit target
motion, cache invalidation, failed solves, caller edits, replay, and fallback.
Local speed tests compare edge edits, empty centre sets, and level boundaries
with full speed construction.
CLI tests check case and repetition boundaries in a process without ROS.
