# Field implementation freeze

Date: 2026-11-24. Evidence: the current `summary(5).json` and `timings(3).csv`
attachments. Earlier uploads are not part of this audit. No tests or benchmarks
were run during this audit.

Freeze the supported complete-field numerical implementation and its current
defaults. The reported correctness and native behavior checks pass. Further
field tuning is not required before completing the original replanning agenda.
This freeze permits correctness fixes and the interfaces needed for that agenda;
it ends the current performance work.

## Evidence checks

- All 1,974 CSV metric rows match the JSON exactly, including sample counts,
  outcomes, completeness, medians, percentiles, minima, and maxima. No rows are
  missing, duplicated, or extra.
- All seven runs report complete. Every case has its expected repetitions.
- All 51 separate exact field comparisons report `match`: floor matrix growth
  1 has 18, floor matrix growth 4 has 18, cavity edits have 8, and the unread
  sequence has 7. No mismatch or error is reported.
- All 215 measured timing requests report `field_only`. Timing runs use one
  warmup and five measured repetitions, with verification in separate runs.
- Every run reports both native checks as passed: completed-state reuse and
  initialization support.
- Floor inputs, hashes, geometry, starts, and goals match across checks and
  timings. The unread check and timing inputs also match. Floor timing options
  differ only in growth and output directory.

## Frozen numerical scope

The field implementation supports persistent caching, exact local speed and
interior-distance work, complete second-order FMM checkpoint replay, safe full
fallbacks, completed-state reuse for unread edits, and local proof of unchanged
initialization. The latest runs select `unchanged_support` for 11 of 13 floor
edits and retain `full_comparison` for the two source-adjacent edits under both
checkpoint policies.

Both hidden edits use `complete_state_reused` with zero computed march batches,
zero checkpoint copies, and zero copied checkpoint bytes. Opening the room uses
the documented full fallback; later reached edits use ordinary replay.
Recorded retained native array sizes remain within the configured 256 MiB limit.
This is not a measurement of total process memory.

Early-exit replay and domain extension remain unsupported. Their existing
stateless fallback remains valid. The global free-space EDT and full gradient
construction are allowed by the original plan and do not block moving on.
The planned field worker, reliable field identity, and execution integration
remain functional agenda work; this audit does not claim those are complete.

## Descriptive target timings

These values describe the current upload only. No fresh before/after comparison
with an earlier upload was possible or attempted. All values are milliseconds.

| Measured quantity | Growth 1 | Growth 4 |
| --- | ---: | ---: |
| Ordinary floor replay, range of request medians | 3.975-4.431 | 4.036-4.264 |
| Source-adjacent replay comparison, request medians | 31.027-31.055 | 31.072-31.081 |
| Floor EDT after edits, range of request medians | 28.826-29.356 | 28.605-29.072 |
| Far-edit complete-case median | 845.956 | 763.332 |

The two hidden edits have replay medians of 0.467 and 0.469 ms. The complete
unread sequence has a median of 342.246 ms. Case medians above come from the
supplied complete-case summaries, not sums of snapshot medians. These timings
do not establish fewer stops, faster arrival, or safe moving handover.

## Source and runtime identity

All seven runs have identical reported environment metadata:

| Item | Recorded value |
| --- | --- |
| Parent revision | `3a785b9524c1506cd4659098ab92f8611cd9604b` |
| Native revision | `cc7e63f98582c0ad2cf913963362cf1605d4632d` |
| Installed native SHA-256 | `22edd316dbf20dcea80a568348ac5ff231b84eac50393ccab3ba66704b5f62eb` |
| Python | 3.14.10 |
| NumPy / SciPy | 2.5.5 / 1.18.1 |
| Numerical threads | 1 |
| Checkpoint interval/count/budget | 10000 / 4 / 256 MiB |
| Profile SHA-256 | `1ab84270b2828309fc25397cd50ca02dbd5d8e2371e4032f35743afdafcdacaa` |

The recorded checkout is not clean: every run reports `M Makefile`, with tracked
diff SHA-256 `9e74e8e49ccc2dfed3dd5b2574d4bb42c2416911ed6c631ea0c260a51980a80f`.
The Makefile diff itself was not supplied, so its content and effect cannot be
established from these attachments. Record this provenance limit; do not label
the measurements as an exact clean-checkout reproduction.

## Decision

Accept this as the supported complete-field development baseline, subject to
the provenance limit above. Keep correctness regression coverage and full
fallbacks. Direct implementation effort to production failure handling, field
identity, full-state plan storage, execution scenarios, validity and braking,
trajectory decisions, and activation. No further field optimization campaign
is needed to begin that work.
