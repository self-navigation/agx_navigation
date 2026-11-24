# Offline replanning measurement baseline

This harness measures the existing FM^2 and PMP kernels on explicit occupancy
snapshots. It provides a full-recomputation reference for later field updates
and planner reuse. It requires neither ROS nor Gazebo.

By default, every sample builds a new field and a new planner. The first BVP is cold; later
segments use the planner's ordinary within-rollout warm starts. Snapshots do
not share planner state. Use `--field-backend cached` for map and result reuse,
or `--field-backend incremental` for experimental local speed updates and FMM
checkpoint replay. See
[field_updates.md](field_updates.md) for their scope and cache reset rules.
The ROS node keeps its default cached backend.

This is a computation benchmark, not the moving-robot full-stop reference arm.
It does not simulate discovery, braking, command execution, collision checking,
IPC or activation. Nominal trajectory duration is not measured driving time.

## Setup

Start from a recursive checkout. Use Python 3.12, a C++ compiler and Python
development headers. The requirements pin the tested numerical/build packages;
pip's supporting dependencies are resolved normally. Avoid installing the ROS
package with its unrelated RL dependencies.

```bash
git submodule update --init --recursive
python3.12 -m venv .venv-baseline
.venv-baseline/bin/python -m pip install -r requirements-baseline.txt
.venv-baseline/bin/python -m pip install --no-build-isolation --no-deps ./depend/scikit-fmm
.venv-baseline/bin/python -m pytest -q tools/tests --basetemp=run_data/baseline-test-tmp
```

Use the checked-out scikit-fmm fork, not the PyPI release. Rebuild it after
changing its C++ sources; the manifest records the installed binary's hash as
well as the submodule revision. The two are not automatically synchronized.

## Run

Run commands from the repository root. Each output directory must be new.

```bash
# Small deterministic cases: straight, partial leaf, closed and reopened passage.
.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/synthetic.yaml \
  --out run_data/baseline-synthetic

# Actual floor-6 map, one short route; validates the pipeline, not route coverage.
.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/floor6_smoke.yaml \
  --out run_data/baseline-floor6 --save-fields

# Timing repetitions on your hardware. Warmups are logged but excluded by summaries.
.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/synthetic.yaml --case straight \
  --warmups 1 --repeat 10 --threads 1 \
  --out run_data/baseline-repeated

# Isolate field construction and the production pack/parse path.
.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/synthetic.yaml --field-only \
  --warmups 1 --repeat 10 --out run_data/baseline-field

# Complete-field replay smoke test; inspect reused batches and total cost.
.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/field_updates.yaml --field-only --save-fields \
  --field-backend incremental --checkpoint-interval 100 \
  --max-checkpoints 4 --field-cache-mib 256 --out run_data/field-replay

# Local centred speed updates on floor 6, including full FMM fallback.
.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/floor6_field_updates.yaml --field-only \
  --field-backend incremental --warmups 1 --repeat 5 \
  --out run_data/floor6-field-updates

.venv-baseline/bin/python tools/summarize_replanning.py \
  run_data/baseline-repeated run_data/baseline-field \
  --out run_data/baseline-summary
```

The CLI sets BLAS/OpenMP thread environment variables before loading numerical
libraries. It runs serially. Repetitions use the manifest order; warmup means
process/library/hardware warmup, not reuse of a previous plan or FMM state.
Use separate invocations with zero warmups to examine first-invocation effects.
Imports are excluded from sample timing. No external processes are timed.

The incremental backend requires a positive checkpoint interval and nonnegative
checkpoint count and cache MiB. Defaults are 10000 batches, four checkpoints,
and 256 MiB. `--checkpoint-growth 4` spreads four checkpoint offsets over
0, 10000, 40000, and 160000 batches; the default 1 keeps fixed intervals.
Growth must be an integer at least one. Count and byte limits still apply.
Each case starts cold in every repetition and warmup. Replay is
limited to complete fields; early exit uses full FMM computation. Map edits still
run both global distance transforms and rebuild signed wall distance. Compatible
centring requests with smoothing disabled use local speed construction; other
settings use full speed construction. The budget covers retained native working and
checkpoint arrays, excluding temporary initialization, preprocessing, output,
and other process memory. See [fmm_state.md](fmm_state.md) for exact native checks.

## Inputs

`config/replanning/baseline_profile.yaml` freezes all planner/field dataclass
values. It uses the calibrated planner launch overrides, centring with
exponential decay 1.5 and cap 0.8 m, unknown cells allowed, threshold 65, and
the full map domain. FMM early exit is disabled. The existing float32
pack/parse path feeds the planner; raw field output remains float64. This
avoids mixing the known differing direct and transported field paths.

This profile is a versioned experiment choice, not a new launch default.
Copy it under a new name to change parameters. Both expanded settings and
their hash are saved; the summary never pools different run directories.
The pinned FMM fork fixes early exit for tied targets and initial source cells.
Interpolation conventions remain unchanged. Keep the recorded source revision
when comparing partial-field results with older runs.

The supplied synthetic cases are fixtures, not a replacement for the original
83-route/60-variant dataset. Those route files are not present in this checkout.
Add real routes as cases when available. No old trajectory NPZ is required.

Example manifest:

```yaml
schema_version: 1
profile: baseline_profile.yaml
map:
  yaml: ../../src/rudn-ordjo-building/maps/floor_6.yaml
cases:
  - id: route_001
    parent_route_id: route_001
    start: [-18.625, -1.425, 0.0, 0.0, 0.0]
    goal: [-16.625, -1.425, 0.0]
    updates:
      - id: leaf_observed
        rectangles: [[255, 565, 257, 569, 100]]
        start: [-18.0, -1.425, 0.0, 0.0, 0.0]
```

Paths are relative to the manifest. Start is `[x, y, yaw, left_wheel_speed,
right_wheel_speed]`; goal is `[x, y, yaw]`, in metres/radians/radians per second.
Map arrays have row zero at minimum world y. Rectangles are half-open integer
`[col0, row0, col1, row1, occupancy]` ranges. Values are -1 for unknown or 0-100.
Updates apply cumulatively within each case. Each case starts from the base
map; each snapshot produces an independent full plan. An omitted update start
retains the previous start. An update may set `goal`; otherwise it retains the
previous goal. Explicit moving states are supported, but the
harness does not derive them by executing the preceding plan.

For synthetic maps, replace `map.yaml` with `map.grid`, containing `shape`
(`[height, width]`), `resolution`, `origin` and `rectangles`. Initial cells are
free; add boundary walls explicitly. YAML image maps support the building
publisher's trinary thresholds, image flip and negate convention. Nonzero
origin yaw and out-of-map endpoints are rejected. This loader is local to the
benchmark; existing map publishers are unchanged.

## Field validation matrix

`floor6_field_matrix.yaml` covers source-adjacent, near, middle, far, enlarged,
and separated map edits. Each case removes its added obstacles before ending.
`separated_field_updates.yaml` stresses distant edits in a large synthetic room.
These are deterministic field fixtures, not recorded sensor updates or route
coverage. Each case and repetition starts with an empty cache.

Use `--verify-fields` to compare every raw field channel, metadata value, and
reached mask with a fresh full computation. Verification runs after the sample
timers stop and records its own wall time. It does not save large field arrays.
Mismatch or verification error makes the benchmark exit with code 1 after all
samples are recorded. Skipped comparisons are explicit. `COMPLETE` means all
samples were recorded; inspect `verification` for correctness.

```bash
.venv-baseline/bin/python tools/bench_replanning.py \
  --config config/replanning/floor6_field_matrix.yaml --field-only \
  --field-backend incremental --verify-fields --out run_data/field-matrix-check
```

Run timing repetitions separately without verification, since a reference solve
changes the process and CPU cache state between samples. Compare matching
snapshots and include initial solves when assessing checkpoint overhead.
Compare growth 1 and 4 in separate timing runs. Inspect retained batch positions,
copy counts, copy bytes, and reused batches; wider spacing does not guarantee
a faster request.

## Outputs and timing boundaries

| File                                   | Contents                                                                                                                                                              |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `manifest.json`                        | Expanded profile, input hashes, geometry, source manifest, Git/submodule revisions and status, Python/package versions, thread settings and installed FMM binary hash |
| `inputs.npz`                           | Exact occupancy snapshots as `map_0`, `map_1`, etc.; indices match the manifest                                                                                       |
| `results.jsonl`                        | One record per snapshot/repetition, including warmups and failures                                                                                                    |
| `sample_XXXX_XXXX.npz`                 | Accepted nominal trajectory prefix; optionally raw fields with `--save-fields`                                                                                        |
| `COMPLETE`                             | All requested samples were recorded; does not mean every plan completed                                                                                               |
| Summary `summary.json`                | Run options, per-snapshot distributions and work counters, and complete-case timing totals                                                                            |
| Summary `timings.csv`                 | Per-run/per-snapshot timing distributions, outcome counts and metric sample counts                                                                                     |

Use a clean committed tree for recorded campaigns. Dirty status and a tracked
diff hash are diagnostic, not a substitute for preserving uncommitted sources.
Maps are stored once per snapshot; raw field saving is optional because a full
floor field is large. Artifacts use ordinary NPZ arrays, never pickled objects.

| Measurement                       | Included work                                                                                 |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| `total_wall_s`, `total_cpu_s`     | Endpoint/occupancy checks, field construction, pack/parse, planning and in-memory diagnostics |
| `compute_field_s`                 | Field computation or update, including cache checks, result copies, and stage timing overhead |
| `edt_s`                           | Top-level free and occupied distance transforms                                               |
| `speed_s`                         | Speed change detection, local or full construction, and copies; includes internal EDT and smoothing |
| `fmm_s`                           | Eikonal wrapper, including replay checks, checkpoints, marching, and output masking          |
| `gradient_s`                      | Production gradient/direction construction                                                    |
| `pack_s`, `parse_s`               | Float32 packing and planner-grid construction, including its derivatives/wall smoothing       |
| `planner_wall_s`, `planner_cpu_s` | Rollout, BVPs, sampling and numerical checks                                                  |
| `solves[].wall_s`                 | Each `solve_bvp` call, including failed calls                                                 |
| `chunk_ready_s`                   | Chunk availability relative to planner start; not validated execution readiness               |
| `artifact_write_s`                | NPZ writing, outside computation timing                                                       |

The four field stages do not overlap. They do not account for every allocation
and bookkeeping operation, so their sum can be below `compute_field_s`.
Stage timers are nested within total timings: do not add both levels together.
Instrumentation and extra state sampling remain part of measured computation
and should stay identical across comparisons.

Samples that call the incremental FMM controller include `fmm_update`: status,
reason, changed cells, reused batches, first affected batch, total and computed
batches, checkpoint count, and retained native array bytes. Stateless fallback
has null values for unavailable counts. Field result cache hits omit this
object. JSON summaries count FMM statuses and reasons per snapshot. Replay can
reuse zero batches, so inspect `reused_batches` and total measured cost before
interpreting any timing difference.

Incremental map preprocessing also includes `speed_update`: status, reason,
changed input cells, updated output cells, computed crop cells, and total map
cells. Changed cells count differences in clipped free distance or mask values;
the count is null when not checked. Status is `local_update`, `full_recompute`,
or `unchanged`. Map and field cache hits omit this object. JSON summaries count
speed statuses and reasons. Local speed construction preserves the full formula
with a halo based on the centring radii and tolerance. It does not change either
global distance transform. Compare `speed_s` and total measured cost alongside
the diagnostics; a smaller crop alone does not establish a speed improvement.

Map loading, rectangle application, input hashing, environment capture and
result-file writes are outside sample timing. The manifest's setup time covers
input preparation/provenance, not Python startup or all artifact I/O. These
numbers therefore measure computation from an available occupancy snapshot,
not sensor-to-command latency. With cached or incremental fields, mask comparison
and result copies are included in `compute_field_s`; map loading and rectangle
application remain outside it.

Trajectory artifacts have N commands, accelerations, poses and costates, with
N+1 raw five-state boundaries and times. Command k acts over `[times[k],
times[k+1])`. Raw wheel states are sampled from the BVP spline, not inferred
from clipped/deadzoned commands. The terminal state of a truncated final chunk
is included. These are sampled records, not saved full-horizon BVP warm seeds.

## Outcome interpretation

`completed` requires the production rollout to report success, all used BVPs
to converge, finite sampled arrays, consistent initial/terminal state pins,
and the final raw pose to satisfy the configured position/heading tolerances.
It does not certify collision clearance, terminal rest, continuous-time limits
or physical tracking. Terminal speed, sampled maxima and pose errors are
reported separately. The default footprint penalty remains off.

`pose_not_reached` exposes near-goal stagnation labelled success by the current
rollout. `bvp_failed` includes solver exceptions and nonconvergence. The
benchmark stops accepting chunks at the first failed BVP, so an older cached
solution cannot masquerade as a new valid segment. This guard affects benchmark
acceptance only; it does not repair the production fallback.

Other outcomes include blocked endpoints, `start_unreached`, nonfinite or
discontinuous trajectories, planner time-cap failure, and unexpected exceptions.
With the default complete field, `start_unreached` means the field did not
reach the start; it is not inferred from the packed invalid-cell sentinel.
With an experimental partial field it does not prove global disconnection.
`field_only` means construction and pack/parse completed; no plan was attempted.

All outcomes remain in the summary denominator. General latency distributions
include failures that reached that stage; each metric has its own `n`.
`completed_total_wall_s` is explicitly conditional on completion. BVP timings
pool calls within the indicated snapshot, not independent route trials. Small
sample p95 values are descriptive only. Incomplete runs are labelled; malformed
or duplicate records are rejected. The CLI exits zero when it recorded the
requested experiments, even if every plan failed, unless requested field
verification found a mismatch or error; inspect outcome counts.

## Summary accounting

`summary.json` includes the manifest's complete `options` object. Snapshot
groups keep timing distributions, outcome counts, and field verification
counts separate. Their `fmm_work` and `speed_work` objects contain distributions
of the recorded work counters. Null or missing counters are excluded, while
zero counts remain valid samples. A field cache hit therefore does not become
a zero-work native sample when no native diagnostics were recorded.

`case_totals` keeps each case separate. Within each requested repetition, it
sums the times for every expected snapshot, including the initial field and
all updates. It then forms distributions of those sums. This gives the cost
of the complete request sequence; adding snapshot medians can give a different
answer.

Only repetitions with the complete expected snapshot set enter these totals.
Failed snapshots remain included when that set is complete. A timing metric
also needs a non-null value in every snapshot of that repetition; missing
stages are not filled with zero. Each metric's `n` shows how many complete
repetitions supplied it. `verification_wall_s` is summed separately and is
never added to `total_wall_s` or the other computation timings.

Each case reports `expected_repetitions`, `complete_repetitions`, and an
`incomplete_repetitions` list with missing snapshot IDs and observed outcomes.
The list also includes repetitions with no records. Case outcome counts cover
all observed non-warmup snapshots, including incomplete repetitions. Warmups
are excluded from all summary distributions. `timings.csv` retains its
per-snapshot format; complete-case totals are in `summary.json`.

## Validation and next use

The focused tests check map conversion and cumulative updates, unchanged
instrumented field output, nonoverlapping stage timing, unreachable-start
handling, injected post-success BVP failure, output/sample conventions,
warmup/failure accounting, and a separate-process CLI run without ROS or torch.

Before comparing a future backend, run the same manifests/profile on both
revisions and retain both directories. Keep complete/partial domains and
precision matched. Use saved raw T/reached masks for field differential tests,
and the recorded outcome/pose/state diagnostics for planner comparisons.
The baseline's defects must not become numerical equivalence targets: record
correctness fixes as a separately identified baseline before claiming speedups.
