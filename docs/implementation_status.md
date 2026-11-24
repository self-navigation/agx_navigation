# Implementation status

Audit date: 2026-11-24. User base: `27262d4`.
Validated implementation: parent `09d5f6b`, native `4d04364`.
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
| Complete-state reuse | Keep a completed march for unread, inert input edits; skip marching and checkpoint copies | Invalidates old checkpoints; a later read edit can need a full solve |
| Checkpoint policy | Count and byte limits, fixed or geometric spacing, checkpoint and copy diagnostics | Budget covers retained native arrays, not total process memory |
| ROS field configuration | Persistent updater, backend/checkpoint parameters, stage logs, and unchanged publication order | Computation remains synchronous in callbacks |
| Benchmark | Full/cached/incremental modes, profiles, provenance, stage and FMM phase times, exact reference verification, and complete-case summaries | Fresh planner per snapshot; no moving execution model |

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

The current batch passed 564 Python tests, 374 fork doctests, and the native
marcher/heap test under ASan and UBSan (leak checking disabled). All 41 separate
field checks matched fresh complete solves: floor matrix 18, disconnected-room
sequence 7, cavity edits 8, separated edits 4, and repeated floor edits 4.

The uploaded target results already had 26 exact matches: floor matrix 18 and
cavities 8. Target edit request medians were 162.3-216.5 ms. FMM took 90.2-140.1
ms and is the main remaining stage cost in those runs. These uploaded results
precede the new native fast path and FMM phase timers.

The new local disconnected-room run removes marching and checkpoint copies
for the two hidden edits. Its complete sequence median changed from 758.8 to
736.0 ms, with overlapping ranges. The connected floor matrix does not use the
new fast path and shows no general timing improvement. Do not treat this batch
as a speedup for every map edit.

Local floor phase measurements identify both marching and replay validation
as substantial costs. Measure these phases on the target machine, including
checkpoint growth 1 and 4, before choosing the next optimization. Then complete
production failure and field identity gates before extending the inherited
planner toward moving, map-aware trajectory reuse and certified handover.
