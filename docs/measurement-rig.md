# The measurement rig: plant, trajectories, tuning machinery, instrumentation

*Moved out of CLAUDE.md on 2026-09-21. This is reference detail, not a decision log — it is all still current, and the plant facts here still constrain what can be modelled.*

- [The constructed plan library, and the first PMP solve cost (2026-08-15)](#the-constructed-plan-library-and-the-first-pmp-solve-cost-2026-08-15)
- [Generating evaluation trajectories by construction (2026-08-14)](#generating-evaluation-trajectories-by-construction-2026-08-14)
- [Choosing evaluation trajectories](#choosing-evaluation-trajectories)
- [The current plant: patch weights, IMU gating, repeats (2026-08-04)](#the-current-plant-patch-weights-imu-gating-repeats-2026-08-04)
- [The tuning machinery (`agx_planning/tuning/`)](#the-tuning-machinery-agx_planningtuning)
- [Instrumentation: per-step state traces](#instrumentation-per-step-state-traces)
- [The patch friction values are still unvalidated against reality](#the-patch-friction-values-are-still-unvalidated-against-reality)
- [`slip_chi` is a function of the SURFACE, and the steering knee is `mu2`](#slip_chi-is-a-function-of-the-surface-and-the-steering-knee-is-mu2)
- [Measuring `slip_chi` on the real robot (method, 2026-08-05)](#measuring-slip_chi-on-the-real-robot-method-2026-08-05)
- [Measurement facts worth not rediscovering](#measurement-facts-worth-not-rediscovering)

## The constructed plan library, and the first PMP solve cost (2026-08-15)

`tools/jobs/20_generate_v2_library.sh` screened 1200 start/goal pairs per floor,
kept 500, and PMP-solved them into `~/traj_data_v2` (fetched to `traj_data_v2/`).

**320 solved, 180 failed (36%)** — 110 `Exceeded max_rollout_sim_time=60 s`,
70 `BVP solve failed … maximum number of mesh nodes is exceeded`. The screen
works as designed: the library is **100% turning shapes, zero straight lines**
(labels, unreliable as above, are 131 UTURN / 100 S / 76 CORNER / 13 ZIGZAG),
against a random-goal library that was ~64% straight.

**PMP solve cost, measured for the first time: mean 1.84 s, median 1.7 s, p90
2.5 s, max 4.3 s per plan**, single core. This is the number the RL re-planner's
data budget depends on: ~2 s a label means 100k supervised labels is ~55 core-hours,
i.e. an overnight run on a handful of cores and **no Gazebo at all**. It also
bounds the re-join solve from above — the re-join problem is strictly smaller
than a full plan — so "PMP cannot be solved online" remains a statement about
scipy's `solve_bvp`, not about the problem.

`tools/select_broad_eval.py` picks a geometry-diverse subset from a library,
stratifying on the (untrusted) label and on path length.

## Generating evaluation trajectories by construction (2026-08-14)

Queue item 8, built. Motivation is sharper than "more plans": **every per-shape
claim rests on exactly ONE plan of that shape.** The U-turn notch is 5906
rollouts of `floor_6_00031`; nothing distinguishes a property of U-turns from a
property of that U-turn, and the sub-ladder's near-vertical walls make that a
real risk rather than a pedantic one.

`agx_planning/tuning/shape.py` (pure, 27 tests) holds the descriptors and the
screen; `tools/sample_eval_trajectories.py` is the offline driver.

**Measured: `trim_pivot`'s inherited 0.30 m is too small.** Real plans turn ~2.8
rad within their first ~0.7 m of travel — a very tight arc, not a pure spin, so
arc-length resampling does not remove it on its own. Sweeping the threshold over
the 100-plan library moves the label counts until 0.7 m and then not at all
(STRAIGHT/CORNER 41/25 at 0.30, 64/16 at 0.70, 65/16 at 2.00). The plateau is
the evidence, and 64 STRAIGHT is what the gallery actually shows.
`PIVOT_TRAVEL_M = 0.70`. (`tools/plot_trajectory_gallery.py` keeps 0.30 on
purpose: display-only, and its figures are committed.)

**The cheap route predicts WHERE a plan goes, not HOW it turns** — validated
against all 100 recorded plans before being trusted, which is the whole reason
the screen is not what it was first written to be:

| descriptor | corr(PMP, predicted) |
| --- | --- |
| `length` | **+0.99** |
| `straightness` | **+0.96** |
| `total_abs_turn` | +0.30 |
| `sign_changes` | +0.34 |

Smoothing the 8-connected lattice staircase does not rescue it (+0.32 at best).
Label agreement rises 15% -> 52% with smoothing, but **only because both
distributions shift toward STRAIGHT** — agreement without predictive power, and
exactly the kind of number that would have looked like success. **So screening
on predicted tortuosity is out**; `screen_score` ranks on blocked line of sight,
detour and pivot demand only, and shape is labelled afterwards from the SOLVED
plan. Do not re-propose ranking candidates by predicted turning.

The two screening constraints, both the user's: the straight line start->goal
must be **blocked** (a distance filter cannot work — a long straight corridor
passes it perfectly, which is how the current library became ~64% straight
lines), and the start/goal **headings force in-place rotation**, which is the
realistic case for a real deployment.

## Choosing evaluation trajectories

`config/eval_trajectories.yaml` holds the seven-shape working set and a
candidate list — kept for per-shape diagnosis, but **the 40 broad v2 plans are
what a gain or controller decision is read from** (see "Reading a gain result").
`just gallery` renders all 100 plans (`figures/trajectory_gallery.png`), each
rotated onto its principal axis so shape is comparable at a glance.

**The automatic labels mislead.** `classify_plans.py` calls 58 of 100 CORNER, but
in the gallery most of those are visually straight lines: the descriptor is
tripped by the in-place reorientation the PMP planner puts at the *start* of a
plan — a large heading change over no distance. Trust the picture. Likewise
`floor_6_00042`, used as "the S-curve" in every comparison so far, is really an
L with one rounded bend. Genuine S-curves: `floor_6_00028` (cleanest),
`00024`, `00047` (zigzag), `00056` (tight V). A true U-turn: `floor_6_00031`.

## The current plant: patch weights, IMU gating, repeats (2026-08-04)

Three changes landed together and **re-baselined everything measured before
them**, because the friction distribution changed deliberately.

**1. The patch friction distribution is a floor, not an ice rink.** `PROFILES`
gained `linoleum` (mu 0.45, the actual deployment surface) and `wet_tile`
(0.30), and `DEFAULT_PATCH_WEIGHTS` replaced uniform sampling — which had put
**half of every patch set at or below tyre-on-ice friction**. Now ~60% at
mu >= 0.30 and 10% ice. `ground_friction_sampler` is weighted too and
**excludes the directional profiles**: a whole floor that grips one axis and
slides the other has no physical analogue. A profile with no weight raises
rather than silently never being drawn (`test_terrain_weights.py`).

**The world's own ground is still `mu=1.0`** (concrete-like) in
`rl_corrector.world`. If deployment is linoleum everywhere, the nominal
no-patch plant is grippier than reality and every gain inherits that bias. Not
changed — it is a re-baseline and wants a decision.

**2. The IMU read is gated** (`_wait_imu_advance`), matching the pose gate. The
un-gated read was the largest difference between two otherwise-identical
rollouts, by twelve orders of magnitude, and it feeds the RL observation.
`stale_imu_steps` counts giving up; the gate is only paid when `use_imu` is on,
so TVLQR work costs nothing for it. **The goal is not determinism** — the real
IMU is noisy and laggy and the policy must tolerate that. The goal is that the
sim's sensor error be a knob we choose (a deliberate latency/noise model matched
to the measured real IMU) rather than an artifact of VM CPU load. **That model
is not written yet**; when it is, it must live in the bridge/env and leave the
obs layout untouched, or the policy stops being deployable.

**3. The tuner averages repeats.** `--repeats 3` (default), reduced per
trajectory by `objective.reduce_repeats`; the cache key includes `repeats`,
`reduce` and `patch_weights`, so an old cache cannot be replayed into a run
whose numbers mean something different. Measured sd of the estimate: single
sample 0.0933, median-of-3 0.0692, **mean-of-3 0.0547**, mean-of-5 0.0413 — see
the Settled stub for why the median loses.

## The tuning machinery (`agx_planning/tuning/`)

Still the rig for any controller comparison, even though the gain search itself
is closed. `just tune-tvlqr` (detached), then `just fetch-tune && just
plot-tune`; `soak.py` / `just soak` is the batch driver and `variance_probe` the
single-rollout one.

- `simplex.py`, `objective.py`, `cache.py`, `epsilon.py`, `shape.py`,
  `trace_diff.py` are **pure and unit-tested** — no ROS, no Gazebo, no torch,
  same rule as the RL pure modules. The simplex tests are aimed at one thing:
  proving the search *minimizes*. A tuner that maximizes produces an
  identical-looking log and hands back the worst gains it found.
- **`--max-evals` bounds candidate GAIN PAIRS, never rollout length**, and
  `--max-evals 1 --repeats 5 --q-cross Q --r-omega R` is exactly a clean
  measurement at a chosen point in the same code path — that is how a tuned
  point gets validated.
- **The trajectory set is fixed, never sampled**, so candidates are compared on
  identical work. A failed rollout makes the whole evaluation `inf`, never a
  mean over survivors.
- **Failures are never cached.** `inf` means "the sim broke", almost never
  "these gains are bad": killing the tuner mid-evaluation invalidated the
  bridge's rclpy context and wrote 56 bogus `inf` evaluations in three seconds.
  They are written with `_failed: true` for diagnosis and never returned.
- **Resumable at zero cost** — the JSONL cache replays the search and re-measures
  nothing; it refuses to resume onto a different problem.
- Search runs in **log10** of both gains, so a step is a ratio and no move can
  propose a negative gain. `--q-bounds` exists because `x0` is CLIPPED into the
  box, so a probe outside it silently measures the boundary.
- Every evaluation records **per-trajectory** errors, so the landscape can be
  re-analysed per shape without re-driving anything.
- Outstanding: Nelder-Mead reports the **minimum observed draw**, which is a
  winner's-curse machine. The structural fix is a noise-aware optimizer (BO with
  a nugget, reporting the posterior-mean optimum). Partly done — the 2026-08-07
  and job-60 runs used BO — but the reporting is still by best draw.

## Instrumentation: per-step state traces

`GazeboBridge.enable_trace(path)` writes one CSV row per control step and per
reset phase: full pose (incl. z and quaternion), twist, wheel speeds, IMU, the
command that produced the step, sim clock, step counters, and a digest of every
`rl_*` entity pose. `variance_probe --trace-dir` writes one per rollout, and so
does `soak.py --trace-dir` (with `--trace-every N` to subsample a long soak).

**The two write DIFFERENT layouts, and that is the reason to score from the
JSONL rather than the directory tree.** `variance_probe` writes
`<dir>/<shape>_<pid>_<i>.csv` and is normally pointed at
`<root>/<arm>/<shape>/`; `soak.py` writes one flat directory with the gains in
the *filename*, because it cycles gain pairs within a single run. So
`tools/score_sweep.py` has two pairing modes — a directory walker for the
former, and `--from-jsonl`, which keys on each row's `trace` field, for the
latter. **Prefer `--from-jsonl`**: a layout is a convention and conventions
drift, but a row records where its own trace actually went. Scoring shape A's
track against shape B's plan produces large, plausible, meaningless numbers
rather than an error, so the pairing is worth making unguessable.

`tuning/trace_diff.py` (pure, unit-tested) reports the FIRST step two rollouts
differ at and which column moved first — the distinction that matters:
`cmd*` moving while state is identical means **our controller** is the
non-determinism; state moving under an identical command means physics; a
`world_steps`/`lost_steps` difference means a step was dropped, not physics at
all. It also timestamps when the xy separation crosses each order of magnitude,
which is how "flat for 150 steps then grows" was distinguished from chaos.
Cumulative counters are rebased per rollout — the world is deliberately not
restarted between runs, so comparing `sim_time` raw reports a fake 250-step
divergence on every pair.

**Two bugs in the traced-soak path (fixed 2026-08-14) are worth knowing because
neither could fail loudly.** `--trace-every N` subsampled by *rollout index*
against a 35-rollout gain x trajectory cycle, so `gcd(5,35)=5` traced the same 7
cells forever — subsampling is by **cycle** now. And tracing is armed by
**file**, not by rollout: untraced rollouts were appended to the previous
traced one's CSV, producing a 1084-row "track" for a 186-step plan that scored
29 m. `disable_trace()` now exists and `soak.py` calls it. The reusable lesson:
a subsample stride and a cycle length are not independent, and a trace file is a
**resource with a lifetime**, not a flag. Both failures produce data that
parses, scores, and looks like a measurement. (`variance_probe` was never
affected — it traces every rollout.)

`tuning/archive/trace_dump.py` (archived 2026-10-06; run it by path) prints selected rows of one trace, for when a run is bad
in isolation rather than merely different from another.

## The patch friction values are still unvalidated against reality

The *distribution* was fixed on 2026-08-04; the *values* never were. **Nobody
has ever checked these against a real surface** — they were picked on a laptop,
by eye, to make the robot visibly slip. Asked about twice before and lost both
times; hence this section. Current `PROFILES`
(`src/rudn-ordjo-building/rudn_ordjo_building/surface_patches.py`) vs. real
rubber-on-surface coefficients (dry concrete 0.7-1.0, wet concrete 0.5-0.7,
tyre on ice 0.1-0.15):

| profile | mu | real-world equivalent |
| --- | --- | --- |
| `rough` | 2.5 | above dry rubber on concrete — effectively "cannot slip" |
| `directional_x/y` | 1.0 / 0.15 | grips one axis, slides the other — **unphysical**, a ground `fdir1` is world-fixed |
| `linoleum` | 0.45 | the deployment surface; sits exactly on the steering knee |
| `wet_tile` | 0.30 | below the knee — "no steering", deliberately |
| `slippery` | 0.2 | wet smooth tile / oily floor |
| `icy` | 0.05 | polished or wet black ice — real, but not an indoor floor |

**Chi, not mu, is the quantity to match** — it is what the model consumes, and
unlike mu it is measurable on the real robot (see "Measuring `slip_chi` on the
real robot"). The route is: drive the real robot on linoleum, on the ice
mock-up and on sand, get chi per surface, then tune each profile's `mu` until
the sim's chi matches. That closes the sim-to-real loop on the one parameter
the planner takes, and nothing else here does.

Still unanswered and cheap: `slip1`/`slip2` sit in a `<friction><ode>` block
while gz-sim runs **DARTSIM**, which likely ignores ODE's force-dependent-slip
parameters — meaning `mu` may be the only knob ever connected. The `icy_noslip`
profile exists solely to test this: drive across `icy` and `icy_noslip` in one
run, and identical behaviour proves `slip1/slip2` are decorative.

## `slip_chi` is a function of the SURFACE, and the steering knee is `mu2`

Measured 2026-08-05 with `slip_ident` on a real-time world, then re-measured
after the wheel fix (`sweep_data/ground_mu_chi*.csv`, driver
`tools/archive/sweep_ground_mu.sh`). The pre-fix curve and its narrative are in
[docs/corrector-history.md](docs/corrector-history.md); the mechanism below is
unchanged and is the reason a frozen PMP plan cannot handle friction zones.

**The mechanism.** [wheel.xacro](src/scout_ros2/scout_description/urdf/wheel.xacro#L63)
gives each wheel `mu1=200.0` (rolling) and `mu2` (lateral), and Gazebo combines
two contacting surfaces by taking the **smaller** coefficient. So `mu1=200` is
never realized — it encodes "the wheel is never the longitudinal limit, the
GROUND decides", which is what makes a patch's `mu` mean anything at all in the
rolling direction. A skid-steer yaws by gripping longitudinally while scrubbing
sideways, so **the longitudinal:lateral ratio IS the steering mechanism**, and
it survives only while `ground > wheel mu2`: there the ground binds
longitudinally and the wheel binds laterally, two independent constraints. At or
below `mu2` the ground binds **both**, the ratio collapses to 1:1, and grip and
steering are lost inseparably. Provenance: Grigorii Matiukhin, 2026-02-13, in
the team's own `scout_ros2` fork — not AgileX upstream, so it is ours to change.

**The knee is at the wheel's `mu2`, to the digit**, and it moved with it when
`mu2` went 0.7 → 0.45 — predicted before the run both times. Min-combination is
settled, not a hypothesis. Above the knee chi is ~1.36-1.57 and the spread
across turn radii is small; at or below it the robot achieves **4-7% of
commanded yaw rate** and most arcs are unmeasurable.

**The free variable is the RATIO `ground/mu2`, not absolute friction** — the
curve translated rather than deformed. That is what makes
`sweep_ground_mu.sh` a reusable instrument for "does this tyre model steer".

**Chi barely moved at nominal: 1.3718 → 1.3575 (~1%).** Above the knee chi cares
*that* you are above it, not by how much — so `PlannerConfig.slip_chi = 1.373`
was still within ~1% and **the baked plans did not need re-planning** for the
wheel fix. The spread across radii improved (0.0299 → 0.0072), so a single
`slip_chi` describes this plant better than it did the old one.

**Consequences that constrain what we can model:**

- **Do not lower the ground plane to model a slippery floor.** Both worlds are
  at `mu=1.0` and should stay there; slipperiness belongs in the wheel pair or
  in a patch, and any patch below 0.45 now means "no steering", deliberately.
- **Usable band is ground >= 0.5.** Linoleum at 0.45 sits exactly on the knee
  and is marginal; every other profile is below it.
- **Ice being uncontrollable is CORRECT, not a limitation** — on real ice
  `mu_long ≈ mu_lat` and a real skid-steer genuinely cannot steer. The model was
  never broken for ice; it was broken for linoleum, because the wheel was
  parameterised for concrete. This is the SVCM dichotomy's second branch.
- **"Slides but still steers" is inexpressible at low friction** under
  min-combination with an isotropic ground, at any wheel setting. `sand` is
  blocked on this, and it is a question for the advisor.
- **Chi is a property of the SURFACE** (1.36 vs 25.4 across the sweep), so it
  cannot be a constant on a floor with ice/sand zones — it is not constant
  *within one trajectory*. That is a modelling gap, not a tuning problem.

## Measuring `slip_chi` on the real robot (method, 2026-08-05)

**Yes, chi is measurable by driving the real robot, and this is the intended use
of `slip_ident`.** It references the **gyro**, which owes nothing to the wheels,
so nothing about the method is sim-specific. `calibrator.py` (archived) cannot substitute:
it compares commands against `/odom`, and both sides share the missing slip term.

Two things to get right on hardware:

- **`cmd_mode:=wheels` is sim-only.** The physical Scout takes only `(v, omega)`
  and computes wheel efforts in firmware, so on the robot you run
  `cmd_mode:=twist`, which yields `chassis_gain_omega` — chi folded together with
  the firmware's own twist→wheel conversion. That composite is the right quantity
  on hardware, because the wheel-level command is not reachable anyway. The tool
  prints which of the two it measured; do not paste a twist-mode number into
  `PlannerConfig.slip_chi`.
- Drive **arcs at several radii**, both directions, on the surface in question.
  A spin scrubs all four contact patches and is strongly load-dependent —
  `slip_ident` reports spins separately and says not to fit chi to them.

**This closes the sim-to-real loop on the one parameter the planner consumes.**
The patch friction values have been unvalidated for weeks (see above) and
measuring `mu` directly is awkward; measuring **chi** is not. So: drive the real
robot on linoleum, on the ice mock-up and on sand, get chi per surface, then tune
each sim profile's `mu` until the *sim's* chi matches the measured one. Chi, not
mu, is the quantity to match — it is what the model actually uses.

**`slip_ident` cannot run against an unthrottled sim (found 2026-08-05).**
`make rl-sim` runs uncapped (~33x), which puts `/imu/data` at **3295 Hz**; with
`depth=10` on a single-threaded executor the node drops almost every sample and
the integrated gyro yaw comes out ~800x too small. It fails loudly (`gyro
measured only +0.0034 rad`, then `No usable arcs`) rather than reporting a wrong
chi, which is the correct behaviour, but it means **chi must be measured against a
real-time-throttled world**. Also note the node's `imu_topic` default is `/imu`
while the actual topic is `/imu/data` — its own usage line has it right, the
default does not.

## Measurement facts worth not rediscovering

- `compare_correctors` builds the bridge with `deterministic=True` — the world
  is paused and multi-stepped, so results do **not** depend on CPU load, and a
  `gz sim -g` viewer cannot perturb them. **This was false until 2026-08-02
  evening** (the pause was being cleared by every step request, see above), so
  results measured before that fix DID depend on CPU load. (`make rl-sim` itself is headless
  server-only, so there is no 3D view unless a GUI client is attached.)
- The identity and TVLQR baselines are checkpoint-independent. A checkpoint
  sweep that re-measures them per checkpoint spends two-thirds of its runtime
  re-deriving the same two numbers; measure them once and run `--correctors rl`
  for the rest. 15 checkpoints ≈ 35 min, ~20-25 s per episode.
- The VM's desktop can be screenshotted without any GUI interaction:
  `ssh <host> 'DISPLAY=:0 import -window root /tmp/x.png'` (ImageMagick, already
  installed; 1920x1080 virtual framebuffer). This is how the final training
  stats block above was found — it was on screen but not in the log tail.
- `tools/plot_checkpoint_paths.py` draws the checkpoint *paths* in a colour
  ramp, next to `tools/archive/plot_checkpoints.py` which reduces each to one scalar.
  Past ~8 overlaid paths the lines stop being individually traceable.
