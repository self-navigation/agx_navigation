# ROS-free execution and plan reuse

This batch returns to the original functional agenda. Field performance work is
frozen. The reference runner uses the production full-field and BVP kernels.

## Run the full-stop reference

From the repository root, using the environment already used for field tests:

```bash
.venv/bin/python tools/replan_bench.py \
  --config config/replanning/execution_smoke.yaml \
  --out run_data/execution-smoke
```

Use a new output directory. The supplied small scenario reveals an object away
from the route after one second. R1 deliberately stops after any map reveal.
It validates and executes a wheel-space brake, builds a complete replacement
from the stopped state, validates the replacement, and resumes. This checks the
event sequence; it is not a route-quality or floor-6 evaluation campaign.

The runner records expanded settings, map hashes, source and native binary
identity, input maps, events, execution states, and every activated full plan.
Known planning failures are retained as outcomes. An unexpected exception
creates failure.json. Exit code 0 means the final execution checks passed;
1 means a recorded unsuccessful outcome; 2 means input or runtime error.

Initial plan construction is setup time. Later field, BVP, and candidate
validation costs advance the execution clock while the robot is stopped.
Brake duration is separate. The simulator's replay and true-map scoring cost
is excluded. An injected delay is explicitly labelled injected_stop_replay.
Measured stop replay does not establish moving-worker scheduling performance.

## Execution and discovery models

- `nominal` follows stored full BVP polynomials.
- `command` integrates exact arcs under ideal held wheel setpoints. It does not
  model actuator ramps or prove physical tracking. Published commands remain
  separate from raw BVP wheel states.
- The brake model ramps each current wheel speed independently to zero after
  a declared delay. Its acceleration is an explicit model assumption, not a
  measured stopping capability. Position integration has a conservative bound.
- Filled rectangles are checked against occupied cell areas, including motion
  between samples. Curved paths require conservative interval motion bounds.
- Observation ticks are independent of the collision step. Initial and plan-end
  observations are explicit additional observation points. Whole-object reveal
  uses range, FOV and conservative grid visibility, or a declared schedule.
- Only the reveal model and evaluator use the true map. The planner callback
  receives the revealed immutable snapshot. Updates observed during computation
  reject the stale candidate and cause a fresh request before activation.

Evaluation success requires final position, heading, wheel-speed and collision
checks. A solver success is insufficient. Near-goal stagnation remains a
separate solver status, even if the final execution passes all common checks.
An unsafe brake result means this configured brake has no certificate; it does
not prove that every possible maneuver is unsafe.

## Plan storage and coordinator

PlanRecord owns N commands and N+1 raw state boundaries. SegmentSolution stores
the full spline, including its uncommitted horizon. Save/load uses NPZ plus
versioned JSON, with no pickle. Legacy files expose their missing capabilities.
Warm seeds can reuse stored states or states plus costates. They cannot
extrapolate past a saved horizon without an explicit fresh-tail builder.

ReplanningCoordinator is a separate pure API. It does not publish commands and
is not yet the execution runner's moving-reuse arm. It provides:

- KEEP after checking the complete remaining path on the current map.
- REPAIR and SPLICE after checking the complete candidate, joints, identity,
  departure time and stopping support.
- STOP_REPLAN at a reserved discrete brake anchor, or FAILED if assumptions or
  deadlines no longer support that reserve.
- Atomic activation of the exact checked immutable pending record. A replaced
  pending record, stale map, changed measured state, or late result cannot
  activate. State deviation also invalidates the old nominal brake certificate.

Repair composition keeps the old suffix's source field keys. It never labels
old splines as newly solved. Retiming applies only to static-map experiments.
The composed plan's field key identifies its current validation request.

The coordinator operates at declared raw state boundaries and initially chooses
segment boundaries for moving departure. The caller supplies certified motion
bounds and a solver/dynamic-limit gate. Automatic candidate construction,
state-deviation propagation, ranking, independent workers, typed ROS transport,
and interpreter activation still need integration. Streaming remains deferred.

## Corrected baseline

Geometry now uses floor for cell lookup and cell centres for interpolation.
The outer half-cell band is clamped; the physical map extent is half-open.
Invalid-cell preparation is shared across direct and packed paths. Float64 and
packed float32 remain different, explicit precision choices. Existing plans
made with corner-based sampling must be regenerated for corrected comparisons.

Typed fields carry map, goal, geometry, policy, recipe and numerical identity.
The legacy packed ROS array remains unidentified. Each rollout freezes one
field; queued worker updates apply to a later request. A failed solve cannot
commit a cached successful solution from an earlier invocation.

Shared profile loading is available to core code and tools. Live launch profile
wiring and canonical map-publisher adaptation are still pending. These Python
checks do not establish live ROS or Gazebo behavior.
