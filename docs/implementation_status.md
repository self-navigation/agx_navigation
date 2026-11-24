# Implementation status

Date: 2026-11-24. This batch is based on the pushed parent `3a785b9` and
scikit-fmm `cc7e63f`. The native dependency and its pin are unchanged.

Field performance work is frozen. The corrected target upload has 51 exact
field matches, seven complete runs, and passing native preflight checks.
The next work is functional replanning integration, not further field tuning.
See `docs/field_freeze.md` for the evidence and its Makefile provenance limit.

## Progress against the original agenda

| Phase | Implemented in the current branch | Remaining gate |
| --- | --- | --- |
| A: baseline | Explicit solve outcomes; no stale solution commit; full raw boundaries; field identity and frozen rollouts; shared sentinel and cell-centre geometry; pure map and profile loaders | Live launch profile wiring, canonical publisher adapter, and typed deployed field transport |
| B: execution reference | Whole-object reveal; nominal and held-command replay; production full-stop R1 runner; stopped compute time, stale-result rejection, and independent final checks | Floor-6 route/scenario campaign and other comparison arms |
| C: validity and braking | Filled-footprint and swept checks; delayed wheel braking with numerical bounds; latest safe declared discrete brake anchor | Physical brake calibration and live deadline scheduling |
| D: plan storage | Immutable N+1 raw states, commands and full spline horizons; NPZ/JSON persistence; bounded active/pending store; explicit warm-seed modes and repair provenance | Connect cross-plan warm seeds to candidate production and measure reuse only after integration |
| E: trajectory decisions | Pure KEEP/REPAIR/SPLICE/STOP coordinator; full suffix checks; exact pending-record binding; state/map/deadline rejection; checked discrete-anchor activation | Automatic candidate construction/ranking, state-deviation propagation, moving-worker scheduling, and execution-runner integration |
| F: field updates | Complete second-order replay, exact preprocessing updates, safe full fallbacks, and verified initialization support | Frozen; partial-domain replay and further tuning are deferred |
| G: runtime | Existing ROS worker now holds one field per rollout and queues later updates | Independent field worker, typed revision transport, pending-plan interpreter activation, reset/pause handling, and live integration tests |
| H/I: experiments | Computation benchmark and a small production-kernel execution smoke scenario | Gazebo/Nav2 comparison campaign, route-level metrics, figures, and paper results |

This is meaningful implementation progress, but the original agenda is not
complete. The new coordinator is a checked core API. It does not yet construct
and schedule moving candidates through the live interpreter. The full-stop
runner intentionally uses fresh full fields and complete replacement plans.
It is the reference needed before connecting and comparing moving reuse.

## What changed in the baseline

A failed BVP can no longer return an old command or commit an earlier spline.
Stagnation has its own status. Each rollout uses a single owned field snapshot;
object-local version numbers no longer identify numerical inputs. Typed fields
check the goal/source association. Legacy packed ROS arrays remain explicitly
unidentified and cannot establish that association.

Field preparation now uses the same invalid-cell sentinel rule on direct and
packed paths. Float64 and float32 precision choices remain distinct. Queries
use cell centres, floor cell lookup, and explicit outer-half-cell handling.
Affected historical plans need regeneration for corrected comparisons. These
are baseline correctness changes; the frozen raw FMM algorithm is unchanged.

Plan records retain full raw states and numerical horizons, with bounded
storage and no pickled SciPy objects. A repair preserves the source identities
of the reused suffix. A rejected candidate leaves the active plan intact.
A changed measured state cannot reuse the old nominal braking certificate.

## Current validation

- 541 tool/core tests passed.
- 423 existing planner unit tests passed, with one skip. Two expected SciPy
  overflow warnings occur in an existing infeasible-pin regression.
- The local production-kernel R1 smoke scenario passed final pose, heading,
  wheel-speed and swept-collision checks after a reveal, brake and replacement.
- Existing benchmark inputs, expanded profiles, metadata and hashes remained
  identical for all 55 snapshot definitions after loader extraction.
- Tests cover field ownership, old-result rejection, cancellation, persistence,
  repaired suffix provenance, thin walls, turns near jambs, braking, stale maps,
  changed state, late activation, and observation timing independent of the
  collision step.

No new target timing campaign is required for this batch. These checks do not
establish physical tracking, live ROS behavior, or a no-stop navigation result.
Full runnable details and limitations are in `docs/replanning_execution.md`.

## Next implementation order

1. Connect candidate construction and controlled warm starts to the coordinator.
2. Execute complete replacements while the old validated prefix continues,
   with worker deadlines and a reserved stopping fallback.
3. Wire shared profiles, typed revisions and active/pending activation into ROS.
4. Run the floor-6/Gazebo comparison campaign and close the remaining protocol
   and reporting gates.
5. Revisit optimization only after the functional system and comparisons work.
