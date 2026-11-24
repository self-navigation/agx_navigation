# Implementation status

Audit date: 2026-11-24. User base: `898061a`.
Validated implementation: parent `5e3ee28`, native `ef17210`.
This report describes code and regression coverage. It does not claim a new
live ROS, Gazebo, or physical execution result.

The new work is a field-update implementation and a computation benchmark.
It is not yet the complete map-aware trajectory replanning system from the
historical implementation plan.

## Implemented by the field-update work

| Area | Implemented behavior | Main limit |
| --- | --- | --- |
| Map and field caching | Compare effective masks, geometry, settings, policy, goal, and target; reuse preprocessing or copy a cached result | One serial request stream |
| Centred speed updates | Exact local regions with source halos; descending radius levels and tiled half-width construction | Unsupported settings use full speed construction |
| Interior wall distance | Certified local occupied-distance updates, crop expansion, and fallback; occupied bounding-box optimization for full reconstruction | Only the obstacle-side distance is localized |
| Native FMM replay | Track input reads, check source initialization, restore a safe checkpoint, and resume the same march | Complete fields only |
| Initialization support | Prove unchanged initialization from fixed phi near changed cells, without reconstruction | Source-area and dense edits retain exact reconstruction |
| Complete-state reuse | Keep a completed march for unread, inert input edits; skip marching and checkpoint copies | Invalidates old checkpoints; a later read edit can need a full solve |
| Checkpoint policy | Count and byte limits, fixed or geometric spacing, checkpoint and copy diagnostics | Budget covers retained native arrays, not total process memory |
| ROS field configuration | Persistent updater, backend/checkpoint parameters, stage logs, and unchanged publication order | Computation remains synchronous in callbacks |
| Benchmark | Full/cached/incremental modes, profiles, provenance, native behavior checks, FMM phase times, exact verification, and complete-case summaries | Fresh planner per snapshot; no moving execution model |

Tests cover exact raw fields and reached masks, additions/removals, mixed
changes, centring, source changes, ties, disconnected regions, cache ownership,
fallbacks, memory limits, and checkpoint spacing. Interior tests include cavity
closure where travel times stay equal but signed wall distances change.
ROS callback and launch tests use stubs; they do not establish live transport.

Main code and tests are in:

- `src/agx_navigation/agx_planning/agx_planning/vector_field/`
- `tools/bench_replanning.py`
- `tools/replanning_baseline/`
- `tools/summarize_replanning.py`
- `tools/tests/`
- `src/agx_navigation/agx_planning/test/unit/test_field_node_update.py`
- `depend/scikit-fmm/`

## Field limits that remain

- The global free-space EDT still runs after a map edit.
- Field gradients are rebuilt over the full field, and returned fields are copied.
- Early exit uses the stateless partial FMM solve. Partial-domain checkpoint
  replay and domain extension are not implemented.
- Source initialization changes or missing safe checkpoints can require a
  full march. Replay status alone does not establish a speed improvement.
- ROS defaults remain cached, early exit enabled, and checkpoint growth 1.
  Complete-field replay requires an explicit configuration change.
- There is no dedicated field worker, versioned field transport, or reliable
  cross-process map/goal revision identity. Existing fields at downstream
  subscribers are not revoked when a request becomes invalid.
- Live ROS latency, stale-message handling, and transport behavior have not
  been validated by these tests.

Details: `docs/field_updates.md`, `docs/fmm_state.md`,
`docs/field_updates_ros.md`, and `docs/replanning_baseline.md`.
All code paths are relative to the repository root.

## Existing planner capabilities are inherited

The planner, supervisor, and runtime-corrector directories are unchanged from
the historical baseline `e395ad967f414b21bf24962397ce4b486f8aec0e` in this audit.
They already provide useful capabilities:

| Existing capability | Current scope |
| --- | --- |
| Previous-BVP warm starts | Shift the previous segment, unwrap heading, and pin the initial state within a rollout |
| Solver process and actions | Stream chunks, cancel/preempt work, and restart a worker after a grace period |
| Map-dependent supervision | Wall slowdown/stop, waypoint, lag, near-wall, and off-plan responses |
| Rejoin and recovery | Candidate landing windows, wall checks/retries, asynchronous rejoin, buffer splicing, and full-replan recovery |
| Playback buffer | Reorder chunks, support timed/progress playback, and resume an old suffix after a rejoin overlay |

These are foundations for new reuse logic. They are not a revision-tagged
map-change coordinator with a validated prefix, reserved braking, and a
certified replacement handover. Full replanning currently clears playback;
there is no separate active/pending plan store with atomic activation.
See `pmp_planner/`, `supervisor/`, and `runtime_corrector/` under the planning
package path listed above.

## Historical plan gates still open

The benchmark detects failed BVPs and checks raw state continuity, but its
guard does not repair the production solver's fallback behavior. Production
failure propagation therefore remains an unfinished Phase A gate.

Field identity is also unfinished. A newly parsed grid starts at version zero
and becomes version one; warm-start invalidation compares that local number.
Two replacement grids can therefore appear to have the same version.
See `vector_field/grid.py`, `pmp_planner/rollout.py`, and
`pmp_planner/shooting_solver.py` under the planning package path above.

The branch does not yet add a versioned full-spline plan store, a map-aware
KEEP/REPAIR/SPLICE/STOP coordinator, full remaining-suffix validation, reserved
wheel-space braking, validated moving handover, or a reveal/execution runner.
The benchmark is not the moving-robot full-stop reference arm. Gazebo/Nav2
campaigns and no-stop navigation claims need separate implementation and tests.

## Validation and next steps

The current batch passed 596 Python tests, 374 fork doctests, and the native
marcher/heap test under ASan and UBSan (leak checking disabled). All 59 separate
field checks matched fresh complete solves: floor matrix 18 at each of growth
1 and 4, disconnected-room sequence 7, cavity edits 8, separated edits 4, and
repeated floor edits 4.

The corrected target upload has 33 exact matches and confirms complete-state
reuse. Ordinary replay cost about 33.4-33.7 ms before this batch. Growth 4 reduced
the far-edit case median by 8.16%; other case differences were below 1%. Growth
therefore remains an explicit workload choice, with the default still 1.

The new local support check avoids initialization reconstruction on 11 of 13
floor edits under both policies. The two source-adjacent edits retain the full
comparison. Ordinary replay phase medians fell from about 36-39 ms to 4.3-4.8 ms.
The complete floor-sequence medians changed from 5124.4 to 4516.4 ms at growth 1,
and 4992.1 to 4416.7 ms at growth 4. The unread sequence changed from 360.1 to
339.1 ms. All existing FMM work, checkpoint, and memory counters stayed equal.
These are local serial measurements; variation in other stages also contributes
to total times. They do not establish target-machine speed improvements.

Incremental benchmark runs now check the loaded native behavior before creating
output. This catches older replay or initialization code and prints a rebuild
command for the Python executable in use. The check does not prove that every
installed source file matches the checkout, so binary hashes remain recorded.

Measure the new replay phase on the target before choosing the next field
optimization. The remaining global free-space EDT and march work still need
measurement. Complete production failure and field identity gates before
extending the inherited planner toward moving, map-aware trajectory reuse and
certified handover.
