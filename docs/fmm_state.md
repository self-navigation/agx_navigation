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
- Check whether changed inputs can affect initialization. Possible initial
  Frozen cells are fixed-phi zeros and endpoints of axial sign crossings,
  including masked cells. Their axial neighbors can become initial Narrow
  cells. Changes outside this area leave initialization unchanged.
- Check that area only around changed cells, without a full-grid cache or
  another scan during construction. If a changed cell can affect it, or the
  changed count exceeds grid size divided by `2 * dimensions + 1`, reconstruct
  initialization and compare all active values, heap IDs, and heap order
  exactly. Inert Far/Mask differences are allowed.
- If no changed input was read and every changed cell is currently Far or
  Mask with no heap entry, retain the current complete march. This needs no
  checkpoint and does not restore or advance the heap.
- Otherwise select the latest supplied checkpoint before the first affected
  read. Every changed cell must be Far or Mask with no heap ID in that checkpoint.

An accepted replay returns `status="replay"` without advancing the march.
It applies the new inputs and invalidates every checkpoint from the previous
input revision. Reason `checkpoint_replayed` means a saved checkpoint was
restored. Reason `complete_state_reused` means the state is already complete;
`reused_batches` is the current total and `first_affected_batch` is -1. An empty
checkpoint sequence is allowed for this complete-state case. Every supplied
checkpoint is still validated.

Ordinary restore with unchanged inputs keeps the accumulated read history.
Checkpoint replay drops reads after the selected checkpoint. Complete-state
reuse keeps the full read history so that a later edit to a previously read
input cannot take the unread path.

`status="unchanged"` preserves the state and its checkpoints. A result of
`full_required` with reason `initialization_changed` or `no_safe_checkpoint`
leaves the old state unchanged. The caller must create a fresh state for the
new inputs. Invalid arrays or checkpoints raise an error. Checkpoints belong
to one state and one input revision.

The native report also contains `changed_cells`, `reused_batches`, and
`first_affected_batch`. Batches are numbered from one for input reads; -1 means
no affected runtime read. Initialization is checked separately.

`initialization_check` records `unchanged_support` when the local check proves
that initialization inputs are unchanged, `full_comparison` when initialization
was reconstructed, or `not_needed` when replay returned before this check.
The field controller carries this value in `last_fmm`; summaries count it under
`fmm_initialization_checks`. It describes the selected check, not elapsed time.

The state API supports complete fields only. It does not support early exit,
narrow bands, or periodic grids. Use the stateless `travel_time` API for those
modes. The field controller uses full recomputation for early exit.

## Memory and checkpoint policy

`stats` reports batches, completion, tracking, and array byte counts. Working
array bytes include dependency history and the sparse initialization signature.
Snapshots do not copy dependency history. `max_bytes` limits one checkpoint's
array payload. Direct API callers own and budget their checkpoint lists.

The field controller stores checkpoints up to a configured count and byte
budget. Its default `checkpoint_growth=1` keeps the fixed interval: offsets
`0, I, 2*I, 3*I` for interval `I` and four slots. An integer growth greater
than one spaces the offsets geometrically. For example,
`checkpoint_growth=4` uses `0, I, 4*I, 16*I`. The count and byte limits still
apply, and a solve that finishes before the next offset can store its completed
state when a slot and enough bytes remain. Growth must be an integer of at
least one.

Set the option on `FieldUpdater(incremental=True, checkpoint_growth=4)` or
pass `--checkpoint-growth 4` to `tools/bench_replanning.py`. Growth is opt-in:
later checkpoints can save more work for distant edits, but give less coverage
between early offsets. Compare complete base and update costs with
`--verify-fields`; a larger reused prefix alone does not establish a speedup.

After accepted checkpoint replay the controller drops the previous revision's
checkpoints and stores new ones from the reused point onward. An offset is
relative to that reused point, not the original batch zero. An edit affecting
earlier work can therefore require a full solve.

For `complete_state_reused`, the controller drops the old checkpoints and
skips both marching and checkpoint creation. It reports zero computed batches,
zero checkpoint copies, and zero retained checkpoints. Later unread edits can
reuse the complete state again with an empty checkpoint list. A later edit to
a previously read input can require a full solve because checkpoints from the
old input revision were discarded. The existing count and byte-budget policy
still applies; disabling checkpoints continues to select stateless full solves.
The budget
covers retained native working and checkpoint arrays only. Temporary
initialization, preprocessing, returned fields, Python objects, and other
process memory are outside this budget.

`last_fmm` and the benchmark's `fmm_update` include `checkpoint_batches`, the
batch numbers of retained checkpoints. `checkpoint_copies` and
`checkpoint_copy_bytes` count checkpoint copies and their array payload for
the current request. Unchanged FMM inputs retain the batch list but report
zero new copies and bytes. Complete-state reuse and stateless fallback report
an empty list and zero copy work. Complete-state reuse retains the working
state, which is still included in `cache_array_bytes`. These values describe
snapshot work, not elapsed copy time or total process memory.

## Validation

```bash
.venv-baseline/bin/python -m pytest -q depend/scikit-fmm/tests
.venv-baseline/bin/python -c 'import skfmm; raise SystemExit(skfmm.test())'
PYTHONPATH=src/agx_navigation/agx_planning .venv-baseline/bin/python -m pytest -q \
  tools/tests/test_fmm_state.py tools/tests/test_field_replay.py \
  tools/tests/test_checkpoint_spacing.py tools/tests/test_complete_state_reuse.py
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
