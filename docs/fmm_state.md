# FMM state and checkpoint tests

The pinned scikit-fmm fork provides `skfmm.TravelTimeState`. It can pause,
save, restore, and resume the existing travel-time solver with fixed inputs.
It uses the same second-order stencil and heap operations as a full solve.

This is a prerequisite for map updates. `FieldUpdater` still runs a full
solve when the effective obstacle mask changes. It does not use checkpoints
yet. Before enabling changed-input reuse, we need to track input dependencies
and select a checkpoint before the first affected operation.

## Rebuild the fork

Use the numerical environment from `docs/replanning_baseline.md`. After
applying source patches, rebuild the installed binary:

```bash
PATH="$PWD/.venv-baseline/bin:$PATH" .venv-baseline/bin/python -m pip install \
  --no-build-isolation --no-deps --force-reinstall ./depend/scikit-fmm
```

The installed binary and checked-out sources are separate. A new submodule
revision alone does not update the Python extension.

## State API

```python
import numpy as np
import skfmm

phi = np.ones(obstacles.shape)
phi[goal_row, goal_col] = -1.0
state = skfmm.TravelTimeState(np.ma.array(phi, mask=obstacles), speed,
                             dx=resolution, order=2)
state.advance(1000)
saved = state.checkpoint(max_bytes=256 * 1024 * 1024)
state.advance()
travel_time = np.ma.filled(state.result(), np.nan)

state.restore(saved)
state.advance()
np.testing.assert_array_equal(np.ma.filled(state.result(), np.nan), travel_time)
```

The count passed to `advance` is a count of equal-time groups, not cells.
A group includes its neighbor updates. Checkpoints copy working times,
cell flags, heap order, heap IDs, and back pointers. Exporting a result leaves
the working distances and frontier intact.

Inputs are copied and fixed for the lifetime of a state. Returned arrays are
also copies. A checkpoint can only be restored to its original state instance.
The API does not support early exit, narrow bands, periodic grids, or map edits.
The existing stateless `travel_time` API remains available for those modes.

`stats` reports the batch count, completion, and array byte counts.
`max_bytes` limits one checkpoint's array payload; it is not a total process
memory limit. The caller owns the checkpoint list and must budget all retained
copies. No automatic checkpoint cache is created.

The full contract and native memory-check command are in
[the fork's state documentation](../depend/scikit-fmm/doc/state.md).

## Validation

```bash
.venv-baseline/bin/python -m pytest -q depend/scikit-fmm/tests
.venv-baseline/bin/python -c 'import skfmm; raise SystemExit(skfmm.test())'
PYTHONPATH=src/agx_navigation/agx_planning .venv-baseline/bin/python -m pytest -q \
  tools/tests/test_fmm_state.py
```

The fork tests compare reached masks and times exactly after repeated pauses
and restores. Cases include first- and second-order updates, ties, random
speeds, masked and disconnected cells, and 1D, 2D, and 3D grids.

The integration test uses the production centring speed and gradient pipeline
on all eight field-update snapshots. It compares raw travel time, both field
components, and gradient confidence with full recomputation.

The separate early-exit regression tests check targets in tied groups,
initialized source cells, unreachable targets, and travel-time margins.
Early exit now counts every accepted target. The margin is added in travel-time
units after multiplication by the smallest grid spacing; it is not a guaranteed
physical radius around the robot.
