# Persistent field updates

`FieldUpdater` is the first field reuse step. It keeps one prepared map and
one field. It uses the same numerical functions as `compute_field`.

| Request | Work | `last_update` |
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

This step does not reuse the FMM march after a map edit. Additions, removals,
and mixed changes all trigger full recomputation. The second-order FMM,
centring formula, early-exit behavior, and gradient rules are unchanged.
The known early-exit tie issue still needs a separate fix before partial-field
performance claims. The supplied measurement profile uses complete fields.

## Python use

Use the environment from [replanning_baseline.md](replanning_baseline.md).
Set `PYTHONPATH=src/agx_navigation/agx_planning` for direct imports.

```python
from agx_planning.vector_field import FieldUpdater, VectorFieldConfig

updater = FieldUpdater()
cfg = VectorFieldConfig(early_exit_enable=False)
outcome = updater.update(occupancy, goal_col, goal_row, resolution,
                         origin_x, origin_y, cfg)
if outcome is not None:
    result, message = outcome
    print(updater.last_update, message)
```

The arguments and return value match `compute_field`. Keep the same updater
for later requests. Creating a new instance for each request prevents reuse.

The ROS field node owns one updater and reports the selected mode in its log.
It still publishes each successful request, including cache hits. A removed
or invalid goal clears the node's current result. The message format is
unchanged; this does not revoke fields already held by downstream subscribers.
The callback tests use ROS stubs. Live ROS transport is a separate check.

## Compare against full recomputation

The benchmark defaults to `--field-backend full`. Select `cached` to retain
field state within a case. Both modes create a fresh planner per snapshot.
An update may supply a new `goal`; otherwise it keeps the previous goal.

```bash
.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/field_updates.yaml --field-only --save-fields \
  --field-backend full --out run_data/fields-full

.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/field_updates.yaml --field-only --save-fields \
  --field-backend cached --out run_data/fields-cached

.venv-baseline/bin/python tools/summarize_replanning.py \
  run_data/fields-full run_data/fields-cached --out run_data/fields-summary
```

The fixture covers repeated inputs, a moved goal, obstacle insertion,
closure, and removal. Use `--warmups 1 --repeat 10` for a local timing run.
Each case starts with an empty cache in every repetition, including warmups.
Separate cases and repetitions cannot supply cached work to each other.

`compute_field_s` includes mask construction and comparison, cache lookup,
numerical work, and copies of the returned result. Stage times are zero for
skipped stages. `field_update` records the mode per sample. JSON summaries
include the backend and mode counts. Map loading and rectangle application
remain outside sample timing. Cache release at case boundaries is setup work.

Compare matching snapshots and inspect failures. A repeated-input cache hit
does not measure the cost of an obstacle update. The closed passage still
reports `start_unreached`. Saved raw arrays permit direct equality checks.

## Validation

```bash
PYTHONPATH=src/agx_navigation/agx_planning .venv-baseline/bin/python -m pytest -q \
  src/agx_navigation/agx_planning/test/unit/test_field_update.py \
  src/agx_navigation/agx_planning/test/unit/test_field_node_update.py \
  tools/tests
```

Tests compare all field channels and reached masks exactly against fresh
solves. They cover repeated map edits, centring, smoothing, bottleneck shaping,
early-exit target motion, cache invalidation, failed solves, and caller edits.
CLI tests check case and repetition boundaries in a process without ROS.
