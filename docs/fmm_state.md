# FMM state, checkpoints, and replay

The pinned scikit-fmm fork provides `skfmm.TravelTimeState`. It can pause,
save, restore, and resume the existing travel-time solver. Optional dependency
tracking allows changed speed and mask inputs to replay from a safe checkpoint.
It uses the same stencil and heap operations as a full solve.

`FieldUpdater(incremental=True)` uses this API for complete fields. The default
updater and ROS node keep cached map and result reuse. See
[field_updates.md](field_updates.md) for integration and benchmark options.

## Rebuild the fork

Use the numerical environment from [replanning_baseline.md](replanning_baseline.md).
After changing C++ sources, rebuild the installed binary from the repository root:

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
                             dx=resolution, order=2, track_dependencies=True)
saved = state.checkpoint(max_bytes=256 * 1024 * 1024)
state.advance()
travel_time = np.ma.filled(state.result(), np.nan)

state.restore(saved)
state.advance()
np.testing.assert_array_equal(np.ma.filled(state.result(), np.nan), travel_time)

info = state.replay(new_speed, new_obstacles, [saved])
if info["status"] == "full_required":
    state = skfmm.TravelTimeState(np.ma.array(phi, mask=new_obstacles), new_speed,
                                 dx=resolution, order=2, track_dependencies=True)
state.advance()
updated_time = np.ma.filled(state.result(), np.nan)
```

The count passed to `advance` is a count of equal-time groups, not cells.
A group includes its neighbor updates. Checkpoints copy working times,
cell flags, heap order, heap IDs, and back pointers. Exporting a result leaves
the working distances and frontier intact. Inputs and returned arrays are
copies. Tracking is disabled by default.

Replay requires a complete state with tracking enabled. Supply full speed and
explicit mask arrays with the original shape. Phi, spacing, order, and shape
stay fixed. Speed values below machine epsilon also mask their cells.
Speed changes in cells masked before and after the edit are ignored.

The native replay checks are exact:

- Record the first runtime read of each cell's inputs. Mask checks are recorded
  before rejecting a masked neighbor. Speed reads are recorded in the solver.
- Reinitialize the new inputs and compare all initial Frozen and Narrow cells,
  raw values, heap IDs, and heap contents and order. Inert Far/Mask differences
  are allowed.
- Select the latest supplied checkpoint before the first affected read. Every
  changed cell must be Far or Mask with no heap ID in that checkpoint.

An accepted replay returns `status="replay"` without advancing the march.
It applies the new inputs and invalidates every checkpoint from the previous
input revision. Ordinary restore with unchanged inputs keeps the accumulated
read history. Changed-input replay drops reads after the selected checkpoint.

`status="unchanged"` preserves the state and its checkpoints. A result of
`full_required` with reason `initialization_changed` or `no_safe_checkpoint`
leaves the old state unchanged. The caller must create a fresh state for the
new inputs. Invalid arrays or checkpoints raise an error. Checkpoints belong
to one state and one input revision.

The native report also contains `changed_cells`, `reused_batches`, and
`first_affected_batch`. Batches are numbered from one for input reads; -1 means
no affected runtime read. Initialization is checked separately.

The state API supports complete fields only. It does not support early exit,
narrow bands, or periodic grids. Use the stateless `travel_time` API for those
modes. The field controller uses full recomputation for early exit.

## Memory and checkpoint policy

`stats` reports batches, completion, tracking, and array byte counts. Working
array bytes include dependency history and the sparse initialization signature.
Snapshots do not copy dependency history. `max_bytes` limits one checkpoint's
array payload. Direct API callers own and budget their checkpoint lists.

The field controller stores checkpoints at a fixed batch interval up to a
configured count and byte budget. After accepted replay it drops the previous
revision's checkpoints and stores new ones from the reused point onward. An
edit affecting earlier work can therefore require a full solve. The budget
covers retained native working and checkpoint arrays only. Temporary
initialization, preprocessing, returned fields, Python objects, and other
process memory are outside this budget.

## Validation

```bash
.venv-baseline/bin/python -m pytest -q depend/scikit-fmm/tests
.venv-baseline/bin/python -c 'import skfmm; raise SystemExit(skfmm.test())'
PYTHONPATH=src/agx_navigation/agx_planning .venv-baseline/bin/python -m pytest -q \
  tools/tests/test_fmm_state.py tools/tests/test_field_replay.py
```

The fork tests compare reached masks and times exactly after pauses, restores,
and input changes. Integration tests compare travel time and field derivatives
with fresh production solves, including replay rejection and full fallback.
The native memory-check command is in
[the fork's state documentation](../depend/scikit-fmm/doc/state.md).

Separate early-exit tests cover targets in tied groups, initialized source
cells, unreachable targets, and travel-time margins. The margin uses travel-time
units after multiplication by the smallest grid spacing; it is not a guaranteed
physical radius around the robot.
