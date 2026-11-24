# Clean before and after collection

This script collects the evidence for the working-software/testing checkpoint
and the experimental-data checkpoint. Statistical analysis, plots, and claims
about improvement belong to the later results checkpoint.

## Run

From the repository root after applying this patch batch:

```bash
.venv/bin/python tools/compare_revisions.py \
  --out run_data/kt4-kt5-collection
```

The default before revision is `e395ad967f414b21bf24962397ce4b486f8aec0e`, the
historical pre-change baseline. After defaults to the current committed HEAD.
Both refs are resolved to full commit IDs once at startup. To choose explicitly:

```bash
.venv/bin/python tools/compare_revisions.py \
  --before e395ad967f414b21bf24962397ce4b486f8aec0e \
  --after HEAD --repeat 5 --warmups 1 --threads 1 \
  --out run_data/kt4-kt5-collection
```

Use a new output directory for every collection. The script creates detached
worktrees and never switches the caller's checkout. Uncommitted caller changes,
including a local Makefile change, are not part of either tested revision.
Commit any intended code/config change before selecting it for measurement.

Git, a C++ compiler, Python development headers, venv and pip are required.
The script is tested on Linux with Python 3.12. Use `--python python3.12` to
select the interpreter for both new environments. The caller's environment
only starts the collector; numerical packages and scikit-fmm are installed in
separate fresh environments. No ROS installation or RL package install is
required. Initial setup downloads wheels and initializes the two relevant
submodules. Run on the same otherwise idle machine for the full collection.

## What is held fixed

One external measurement harness is copied into the collection. It calls each
checkout's production code in a separate process. It does not add the head's
planning implementation to the old checkout. Map arrays, source/goal states,
expanded profile settings, precision, input order and evaluation rules are
frozen once and used by both revisions.

The common protocol uses complete fields, raw float64 field output, and the
packed float32 planner path. Planner geometry interpretation remains the native
behavior of each revision. The geometry corrections are therefore part of the
before/after change; resulting plan differences are not pure timing effects.

Each revision checks out its own pinned native dependency. The script builds a
new non-editable native wheel, installs it only in that revision's environment,
and checks the loaded binary path. Both environments use the same downloaded
dependency wheels. Package versions, source hashes, compiler environment,
thread limits and native binary hashes are retained. An already installed FMM
binary from the caller's environment cannot silently serve both revisions.

## Tests and data arms

| Evidence | Before | After |
| --- | --- | --- |
| Frozen common behavior probes | Same nine targeted checks | Same nine targeted checks |
| Native doctests | Revision's own suite | Revision's own suite |
| Planner unit tests | Revision's own suite | Revision's own suite |
| Tool tests | Unsupported in the historical baseline | Revision's own suite |
| Full field and fresh planner | Common inputs and measurement code | Common inputs and measurement code |
| Incremental field | Unsupported in the historical baseline | Separate field-only arm |
| R1 execution smoke | Unsupported in the historical baseline | Recorded separately from paired timings |

The common probes target known geometry, sentinel, stale-solution, cancellation,
stagnation and early-exit defects. They are not an overall correctness score.
Mock BVP probes are labelled separately from compiled native checks. Each
revision's own suite has a different scope, so its test count is not a matched
performance metric. Absent features are recorded as unsupported.

The planner measurement adapter rejects failed or stale numerical output in
both revisions. This is a guarded kernel comparison, not a replay of the
historical runtime's unsafe fallback commands. Solver status, finite data,
continuity and final position/heading/wheel-speed criteria are kept separately.
No collision or physical-execution success is inferred from those criteria.

Defaults collect the floor-6 field edit matrix, cavity edits, unread-region
edits, synthetic planning cases and the short floor-6 planning route. There are
five measured repetitions. Every measurement process first executes one logged
warmup sequence. Cases reset field state before their base snapshot, and their
edits retain their declared order. Initial construction cost is retained.
Incremental settings are fixed at interval 10000, growth 1, four checkpoints and
256 MiB, and are written in the protocol.

All runs are serial. The first arm rotates between repetitions, with reversed
orders on alternating complete cycles. Verification is a separate pass; its
fresh-reference work and raw-field writes are excluded from timing data.
Field verification compares each arm with its own fresh full solve and also
compares saved before/after fields and current full/incremental fields.

The full ROS2 stack is excluded to control inputs, numerical threads and process
scheduling around the measured kernels. These results support algorithm-level
testing. They do not measure ROS transport, Gazebo behavior, physical tracking,
or moving-worker activation. R1 is a model-level smoke test of the implemented
reference behavior, not a matched historical moving-replanning comparison.

## Output and failure handling

The main shareable output is `OUT/raw-data.zip`. It excludes worktrees, virtual
environments and build caches, while retaining the harness and all raw evidence.

| File or directory | Contents |
| --- | --- |
| `protocol.json`, `protocol/` | Fixed revisions, options, build settings, dependency requirements and wheel hashes |
| `planned_runs.csv`, `planned_runs.json` | Full attempt matrix saved before tests and measurement |
| `inputs/` | Frozen arrays, expanded profiles, source metadata and scenarios |
| `environments/` | Package/source/native identity and final clean-tree checks |
| `tests/`, `features/` | Common expected/actual checks, JUnit XML and revision-specific feature evidence |
| `logs/`, `commands.json`, `observations.jsonl` | Exact commands, dates, timeouts, exit codes and complete process logs |
| `runs/`, `runs.jsonl` | Per-attempt records, verification fields and planner trajectories |
| `raw_metrics.csv` | Raw timings, statuses and counters; no pooled averages |
| `completeness.csv`, `summary.json` | Planned/recorded/missing/duplicate counts and audit findings |
| `cross_fields.json` | Numerical field equality, masks, channel differences and artifact integrity |
| `SHA256SUMS` | Integrity hashes for all archived evidence |

Failed plans and unreachable cases remain in the denominator. Timed-out workers
retain completed rows; missing attempts are explicit records with no invented
zero timing. Duplicate, malformed or mismatched records are excluded from usable
measurements and reported by the audit. There are no automatic retries and no
overwriting or resuming of old runs. A retry should be a separate collection.

Exit codes:

- 0: collection complete, test commands passed.
- 1: collection complete, but one or more test/feature commands failed. Inspect
  the logs; this is expected when targeted probes expose historical defects.
- 2: setup, collection or audit failed. Partial data and logs remain available.
- 130: interrupted; partial data and the planned denominator are retained.

`COMPLETE` means all planned measurement attempts are recorded and the collection
audit passed. It does not mean every plan succeeded or every historical test
passed. Check outcomes and test logs before using a subset in later analysis.

## Small check and repeat options

For a quick check of the collection process, explicitly label the run as smoke:

```bash
.venv/bin/python tools/compare_revisions.py \
  --smoke --repeat 1 --warmups 0 --out run_data/comparison-smoke
```

This uses small synthetic inputs. Do not pool it with the full collection.
`--field-config` and `--planner-config` select committed paths relative to the
after worktree; each flag can be repeated. `--no-incremental` retains just the
two full-computation arms. `--skip-tests` explicitly omits the common/native/
Python regression suites; it is recorded and is unsuitable as the sole KT4
testing evidence. The separate R1 smoke still runs when supported.

To reuse downloaded dependency wheels, pass
`--offline --wheelhouse PREVIOUS_OUT/work/wheels`. Source/native builds remain
fresh. Git may still fetch a pinned submodule if its object is absent locally.
There is no implicit network update of the selected parent refs.

Large reproducible setup files remain in `OUT/work`. Git registers the two
worktrees. After retaining the data archive, remove those worktrees with
`git worktree remove OUT/work/checkouts/before` and the corresponding `after`
path before deleting the remaining setup directory. Normal worktree commands
refuse removal if there are uncommitted changes.

## Collector validation

The isolated smoke collection tested historical parent `e395ad9` with native
`3f185af` and current parent `403b4aa` with native `cc7e63f`. Two measured
repetitions with one warmup per block produced all 125 planned rows, no missing
or duplicate records, and 15 matching raw-field comparisons.

The old revision failed the nine selected known-defect probes and reported
118 native doctest failures, including its distance-wrapper keyword interface
and dependent examples. These failures were preserved without changing the
historical source. Its planner unit suite passed 371 tests with one skip.
The current revision passed the nine probes, native doctests, 423 planner unit
tests with one skip, and 541 existing tool tests. Its R1 smoke also passed.
These are collector checks, not a target-machine performance result.

The collector has 54 focused regression tests. The complete current tool test
suite, including those checks, passed 595 tests. Tests cover pairing, frozen
inputs, environment isolation, timeout cleanup, missing/duplicate records,
corrupt artifacts, failed verification and native interface compatibility.
