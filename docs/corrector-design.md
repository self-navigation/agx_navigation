# The corrector: design, and what tuning it still needs

**Written 2026-08-15.** Source of record for *what we are building and why*.
CLAUDE.md's "Current work" holds measured findings; `handover.md` holds current
state; this holds the architecture and the argument for it. Formulas cited here
are transcribed in [svcm-source.md](svcm-source.md) with page numbers.

## The objective, stated once

Hold a **frozen, open-loop PMP plan** under slip that the planner did not know
about, using **closed-loop, immediate-mode corrections computed onboard**, and
stay ε-optimal in the cost functional — not merely close in metres.

The source states the decomposition we are implementing directly (p. 52):

$$u_{adm} = u_J + \bar{u}$$

the control the agent already has, plus a correction drawn from the ε-optimal
set. **Our frozen PMP plan is $u_J$ and the corrector is $\bar u$.** That is
worth stating because it settles a recurring question: the residual/corrector
structure is not a shortcut around the theory, it is the theory's own form.

## What the source actually prescribes, and where we already agree

Three findings from reading the dissertation directly (2026-08-15), each of
which changes what we build:

**1. The catalogue is stored as small networks — explicitly.** p. 77: PMP
solutions "можно хранить как в форме параметрических траекторий, так и в виде
компактных аппроксиматоров, например, небольших сетей". So "RL as the library
compressor" is the source's own proposal, not an extrapolation of it. The
motivation we arrived at independently — a tabulated catalogue over a continuous
deviation state is defeated by dimensionality — is the reason the source offers
the alternative.

**2. Nominal and crisis scenarios are the same PMP problem with different
$Q_c, R_c, P_f$** (p. 76–78). The scenario index is the surface/environment
class. Crisis matrices are prescribed as: **larger yaw penalty, larger
wheel-speed-difference penalty (to suppress skid), and an INCREASED
control-effort penalty $R$ to limit aggressive action in low-traction zones.**

**3. We independently reproduced (2) empirically, and did not notice.** Tuning
against slip terrain moved `r_omega` **up** 0.25 → 2.618 and `q_cross` **down**
10 → 0.276 — an increased control-effort penalty on $\omega$ and a relaxed
cross-track penalty. That is the crisis-matrix prescription, found by Bayesian
optimization against Gazebo without reference to the text. It is the strongest
available evidence that the two lines of work describe the same system, and it
belongs in the write-up.

It also reframes what our tuner *is*: the source says $\theta^p$ (which fixes
$Q_c, R_c$) is "фиксируется в результате обучения" — determined by learning. Our
BO tuner is that step, done with a different optimizer. We have simply only ever
run it for **one** scenario.

## The architecture

Four components. Two exist, two do not.

| # | component | rate | status |
| --- | --- | --- | --- |
| 1 | PMP planner — nominal plan on the nominal surface | once per goal, offline | **built** |
| 2 | TVLQR — track the active reference | 50 Hz onboard | **built, tuned** |
| 3 | Scenario recognition — measure the surface, pick the catalogue index | continuous onboard | **not built** |
| 4 | Re-join template network — the compressed catalogue | one forward pass per tick, when triggered | **not built** |

### The distinction that was blurred, and matters

There are **two different things RL could compress, at different levels**, and
conflating them is why the previous RL work had no clear target:

- **Level A — the template library.** A map from (deviation state, surface
  class) to a short re-join trajectory. Input is ~8–10 continuous dimensions:
  $(e_{along}, e_{cross}, e_\theta, v, \omega, \hat\chi)$ plus a little local
  geometry. **This is where the curse of dimensionality bites** — a tabulated
  catalogue at even 5 samples per axis is $5^9 \approx 2\times10^6$ entries, and
  the sampling is not the problem, the *storage and interpolation* is. A small
  network is the compression. **This is the RL/regression job.**
- **Level B — the cost matrices $(Q_c, R_c)$ per scenario.** Input is one
  categorical index (nominal / ice / sand). Output is ~5 numbers. **This is a
  table with a handful of rows and needs no compression at all.**

The source runs actor-critic at Level B and stores networks at Level A; it does
not separate them sharply, and read quickly it suggests "RL" is one thing. It is
not. **Level B is our existing tuner, run once per surface class. Level A is a
supervised regression problem.** Neither is a SAC residual on wheel commands.

### 3. Scenario recognition — the step both documents assume and neither specifies

`chi_hat = omega_ideal(from wheel commands) / omega_measured(gyro)`, i.e.
`slip_ident`'s computation run recursively online. Undefined when $\omega
\approx 0$, so it needs a validity gate that holds the last value. Maps to a
catalogue index by thresholding.

This is well-founded here because **chi is a property of the surface** — measured
1.36 to 25.4 across the ground-friction sweep — which is exactly what makes it
both an index into a surface-keyed catalogue and, unavoidably, *non-constant
within one trajectory*. The latter is the structural reason a frozen plan needs
correction at all.

### 4. The re-join template network — Level A

```
sample (nominal plan, deviation state, local chi)
    -> solve the real PMP re-join problem      (teacher)
    -> regress                                  (student = the compressed catalogue)
```

Supervised, not SAC, and every open RL failure dies with the switch: no
exploration so no `ent_coef` runaway; no bootstrapped value so no `critic_loss`
divergence; no reward shaping; each label is an exact optimum rather than a noisy
return. **Data generation needs no Gazebo**, so it is CPU-parallel. Gazebo
returns only for validation.

The data budget is now measured rather than guessed: **PMP solves in ~1.84 s
mean / 2.5 s p90 on one core**, so ~100k labels is ~55 core-hours. That is an
overnight run on a few cores.

Two things stand between here and there:

- **The re-join boundary condition.** The solver currently goes start → goal; it
  needs "from an arbitrary state, back onto the nominal path". A BC change, not
  a new solver.
- **A 36% solve-failure rate** on fresh start/goal pairs (110 timeouts, 70 mesh
  exhaustion of 500). Tolerable when building a library offline — you discard
  the failures — and **not** tolerable if it carries over to the re-join
  problem, because those failures become gaps in the catalogue. Measure it on
  the re-join problem before committing.

### The build plan for Level A, in order (agreed with the user 2026-08-15)

**This is the goal to build next.** Four phases, each of which produces something
usable and each of which can stop the project honestly if it fails.

**Phase 0 — measure whether there is a teacher at all.** Add the re-join BC and
solve ~200 sampled re-join problems. The number that decides everything is the
**solve failure rate**: the library build failed 36% of the time (timeouts + BVP
mesh exhaustion). If the re-join problem inherits that, there is no reliable
teacher and the whole supervised plan is wrong — see "when RL becomes necessary"
below. Cheap, offline, no Gazebo. **Do this before writing any training code.**

**Phase 1 — a dataset, sampled RANDOMLY and never on a grid.** This is the one
place the design is easy to get wrong, because "curse of dimensionality" names
two different things:

- **Tabulating** the catalogue over a ~10-D input is hopeless — a grid at 5
  points per axis is $5^{10} \approx 10^7$ solves, ~5000 core-hours at the
  measured 1.84 s. That is the curse, and it is the reason a table is not an
  option.
- **Regressing** it is not. The sample count a network needs scales with the
  complexity of the function, not the dimension of its input. ~100k *random*
  samples is ~55 core-hours and covers the region that actually occurs.

So: **sample the deviation distribution, do not enumerate it.** Inputs are the
deviation state $(e_{along}, e_{cross}, e_\theta, w_l, w_r)$, local $\hat\chi$,
the nominal geometry ahead, and possibly $T_w$ — ~10-13 dims. **The terminal
condition is NOT sampled**: it is determined by the plan and the re-join index.

**Phase 2 — close the distribution loop with DAgger, not RL.** A network trained
on randomly sampled deviations visits a *different* distribution once it is
driving, which is the standard imitation-learning failure. The fix is dataset
aggregation: roll out the current student, observe the states it actually
reaches, query PMP at exactly those states, add them, retrain.

The user proposed this as an RL loop — sample a point, compare the network's
output against PMP's, reward the difference. **The instinct (generate labels
where they are needed rather than precomputing everything) is right and is what
makes the space tractable. The mechanism should not be RL:** when the expert's
answer is in hand, the network-minus-expert difference IS the exact gradient, so
converting it to a scalar reward and using a policy-gradient estimator replaces
an exact gradient with a high-variance estimate of the same quantity. Strictly
worse, no compensating benefit.

Where `J` enters: "how far apart are two re-join trajectories" needs a metric,
and `J` is the principled one — weight the regression error by its consequence
in the cost functional rather than treating all output coordinates equally.

**Phase 3 (optional) — RL fine-tuning, which is the ONE place RL earns its keep.**
PMP is optimal *for its model*, and the model is wrong under slip: it assumes the
nominal $\chi$. So supervised learning is capped at teacher level by
construction. Starting from the supervised policy and fine-tuning against the
real plant can *exceed* the teacher, and the reward is available for free and
principled: $-\texttt{epsilon.step\_cost}(\cdot)$, the same functional we report.
This is also the version most defensible in the write-up — RL is contributing
something PMP cannot, rather than re-deriving what it already knows.

**When RL becomes necessary rather than optional:** if phase 0 shows the re-join
solve fails often. A teacher that answers 64% of the time cannot label a dataset,
and the fallback is to learn without one. That is the risk this ordering is
designed to expose first and cheaply.

### It is not a "planner", and what its output should be (revised 2026-08-15)

**The word "re-planner" was wrong and is retired.** Planning implies search or
optimization at runtime; this does **one forward pass, no search**. The
dissertation's own word is **шаблон — template** — and applying a stored
template is not planning. Call it the **re-join template network**.

That leaves a real design question, raised by the user: does it emit a
**trajectory** (which something else then tracks) or a **command applied
directly to the wheels**? Both work, and the tradeoff is worth stating because
the first draft of this document assumed the first without arguing for it.

| | **A. reference generator** | **B. amortized controller** |
| --- | --- | --- |
| output | a short re-join trajectory | the wheel command, this tick |
| target to regress | PMP's whole re-join solution | PMP's *first* command at that state |
| TVLQR | still runs, tracks the new reference | replaced while active |
| NN error | attenuated by TVLQR feedback | goes straight to the plant |
| runtime plumbing | must splice a segment into playback | none |

**The resolution is a hybrid, and it is better than either.** Have the network
answer *"what should the REFERENCE be right now?"* rather than *"what should the
wheels do right now?"*:

- one forward pass per control tick, input = current deviation state + $\hat\chi$
  + local plan geometry, output = a reference state (the 5-vector) plus its
  feedforward command;
- **TVLQR closes the loop on that reference**, exactly as it already does on the
  frozen plan — so the network's errors are attenuated by feedback instead of
  reaching the plant;
- **no splicing plumbing at all.** The reference is simply whatever the network
  says this tick, so `trajectory_buffer` needs no notion of substituting a
  segment and resuming. This retracts the claim in the roadmap below that
  splicing is required.

So it keeps the user's instinct (a single pass, applied immediately, nothing that
looks like planning) *and* keeps TVLQR as the stabilizer. Regression target is
still PMP's re-join solution — sampled at the current tick rather than emitted
whole.

### The escalation ladder, and what actually distinguishes the tiers

The user's three tiers are right. The refinement is that **magnitude is a proxy;
what actually separates them is which assumption has become false.**

| tier | what has become false | test |
| --- | --- | --- |
| **TVLQR** | nothing — deviation is within the corrector's authority | default |
| **template network** | the *plan is still valid* but the corrector cannot close the error from here | TVLQR persistently saturating (`CorrectionDiagnostics.saturated_*`) |
| **full PMP re-solve** | the *reference itself* is invalid — the path is no longer feasible or no longer leads to the goal | path validity, below |

Cross-track distance is a reasonable first cut for tier 2 and is what to start
with, but saturation is the honest test: a large deviation the corrector is
comfortably closing does not need a template, and a small one it cannot close
does.

**Quantifying "the map changed enough" (the user's open question).** It does not
need a planner to *detect*, only to fix. The remaining frozen path is a list of
poses; the live occupancy grid is available. So the test is a **clearance check
of the un-driven remainder of the path against the current map** — if any
remaining pose loses its clearance radius, the reference is invalid and no local
correction can rescue it. Cheap (a few hundred grid lookups), no optimization,
and it is a statement about *feasibility* rather than about error magnitude,
which is what tier 3 is supposed to key on. The surface-change analogue is the
dissertation's own (p. 80): accumulated model-prediction error above a threshold
over a fixed window triggers a "new coverage" event.

### Isolating the corrector arms — the seam already exists

Yes, and it is already built, because it is how the old RL residual was measured:

- **measurement**: `compare_correctors --correctors identity tvlqr rl` — arms are
  a list, each driving the same trajectories on the same seeded terrain.
- **live demo**: `make fixture CORRECTOR=identity|tvlqr`, and
  `just remote-fixture <corrector>`.

So adding the template network is a **new arm, not new architecture**. The
ablation set that isolates each contribution:

| arm | reference | feedback | isolates |
| --- | --- | --- | --- |
| `identity` | frozen plan | none | how bad slip is unaided |
| `tvlqr` | frozen plan | TVLQR | feedback alone — **today's system** |
| `nn` | network | none (open loop) | whether the re-join REFERENCE is right, independent of tracking |
| `tvlqr+nn` | network | TVLQR | the full system |

The `nn` arm is the interesting one for a demo: run open-loop it answers "is the
learned template a good maneuver?" separately from "can we track it?", which are
the two ways the system can fail and are otherwise confounded. Note it is only
meaningful under option A/hybrid above — if the network emitted raw wheel
commands there would be no reference to run open-loop.

### The trigger — and it already exists

Escalating from TVLQR to a re-join template needs a test for *"the reference has
become infeasible"*, which is **not** the same as *"we are far off it"*. The
signal is already implemented and unused: `CorrectionDiagnostics.saturated_v` /
`saturated_omega`, whose own docstring says a persistently saturated corrector
"is being asked to fix a deviation beyond its authority, which means either the
limits are too tight or the trajectory needs replanning". That is the trigger,
stated in the codebase before we knew we needed it. The source's own version is
the accumulated model-prediction error (p. 80).

## Does the existing RL code match this? No — and here is exactly where

`rl_corrector/` trains a SAC policy emitting a **4-wheel multiplicative residual
on the commands**, rewarded by a hand-shaped 8-term function.

| piece | verdict |
| --- | --- |
| action = per-wheel multiplicative coefficient | **wrong object.** Not a re-join trajectory (Level A) and not a cost matrix (Level B). It is a third thing the framework has no place for. Also undeployable: the physical Scout takes only $(v,\omega)$. |
| reward = 8 independently tuned weights (`w_ontrack`, `w_cross`, `w_heading`, `w_progress`, `w_effort`, `w_smooth`, `term_penalty`, `success_bonus`) | **wrong currency.** Not comparable to anything we report, and not $J$. |
| SAC / off-policy exploration | **wrong method** for a problem with an exact teacher. |
| observation layout, `Bridge`, terrain sampler, env harness | **keep.** Sound, tested, and reusable. |
| `compare_correctors`, eval sets, soak, tuner | **keep.** Plant-independent measurement machinery. |
| identity fail-safe (`policy_path` unset ⇒ byte-identical pass-through) | **keep.** It is why none of this is deployed and nothing is at risk. |

**Recommendation: retire the SAC residual rather than repair it.** Its reward is
not worth converting to $J$, because the object it rewards is wrong at the action
level; fixing the currency of a mis-specified action buys nothing. The harness
around it is most of the value and survives intact.

This is consistent with what was already measured: no RL checkpoint ever showed
a learning trend on the real task (r = 0.111 over 20 checkpoints, 0 of 20 beat
TVLQR). We now have an architectural reason for that, not just an empirical one.

## What $J$ changes, now that it is computed online

`epsilon.EpsilonAccumulator` sums the functional as a rollout runs, so `j_total`
comes back with `max_cross` and no trace file is involved. Three consequences:

1. **Tuning can target $J$.** `objective.metric_values` selects the metric and
   `aggregate(..., how="geometric")` reduces it.
2. **$J$'s aggregator had to differ from metres'.** $J$ spans 0.2 to 1043 across
   the 51-plan library, and `floor_6_00031` alone is **48% of the arithmetic
   mean** — so a search on that mean tunes to whichever plan is worst. The
   geometric mean tracks the per-plan win rate (45/51) where the arithmetic mean
   does not.
3. **Any future RL gets its reward for free**: `-epsilon.step_cost(...)` is the
   per-step integrand, so the training signal and the reported score become the
   same functional instead of two hand-weighted opinions.

**$J$ is an upper bound on $\varepsilon$, never $\varepsilon$ itself** — $J^* > 0$
under slip and is unknown. That is the honest reading and the useful one: an
ε-admissibility claim is established by bounding.

## The tuning that remains

In priority order. Note that (1) and (2) are *the same tuner*, run more times —
not new machinery.

1. **Re-tune against $J$ on the broad plan library, not the seven.** Every gain
   we hold was chosen on 7 hand-picked plans against `max|e_cross|`. The
   library sweep already showed that set is enriched for hard plans, and the
   U-turn episode showed a per-shape optimum need not survive an aggregate.
2. **One $(Q_c, R_c)$ per surface class — the Level B catalogue.** We have one
   row, measured on mixed terrain. The source prescribes distinct crisis
   matrices; we should *measure* them, per profile, with terrain pinned. This is
   both a deliverable and a test of finding (3) above: if tuning on pinned ice
   moves the gains further in the same direction, the correspondence is real.
3. **Widen beyond $(q_{cross}, r_\omega)$** to the full diagonal
   ($q_{along}, q_{heading}, r_v$). Deferred until (1) fixes the objective —
   widening a search whose objective is the wrong quantity resolves noise.
4. **Do not** re-run a wide 2-D search on the seven-plan set. That is finished.

## Roadmap to a visible demo

Written 2026-08-15 in answer to "how much work until we can watch the robot drive,
slip, and get back on track?"

**The key realisation: that is TWO demos, and the first one is already built.**
TVLQR *is* a closed-loop corrector that returns the robot to the trajectory —
that is what 45-of-51 plans better in $J$ means. The re-planner is only needed
for the case TVLQR **cannot** fix, i.e. where the reference has become
infeasible. So:

| | what you see | needs |
| --- | --- | --- |
| **Demo A** | drives the plan → slips → TVLQR pulls it back | **believed complete; never watched** |
| **Demo B** | deviation exceeds the corrector's authority → a re-join is planned and executed | the phase 0–2 build above |

### Demo A — the parts, and they are all present

Checked in the tree 2026-08-15, not remembered:

- **The rig**: `just remote-fixture tvlqr true` — full ROS stack, GUI on the VM
  desktop, `vglrun` rendering on the V100, reachable over Moonlight.
- **The plan is drawn**: `runtime_corrector` publishes `~/debug_markers`
  (`MarkerArray`) carrying the plan path *and* an arrow at the sample currently
  being commanded — "where it should be right now" versus where it is, which is
  exactly what makes an excursion legible.
- **RViz already displays it**: `main.rviz` has a `corrector status` MarkerArray
  on `/wheel_corrector/debug_markers`, plus the vector field group.
- **The plan reaches the corrector**: `vec_pmp.launch.py` remaps
  `~/plan → /optimal_trajectory`.
- **Slip exists and is deliberately placed**: `gz_sim.launch.py` holds an
  explicit patch list (icy/slippery/rough/directional at named coordinates).
- **The record**: `run_recorder` writes track/plan CSVs, `tools/plot_run.py`
  renders path + deviation figures, and the desktop can be screenshotted headless
  with `DISPLAY=:0 import -window root`.

### What is actually missing for Demo A

Small, and none of it is research:

1. **Nobody has ever watched it.** Every corrector number in this repo came from
   `compare_correctors` / `soak`, which drive `GazeboBridge` directly with
   recorded plans and **bypass the entire ROS stack**. So the fixture path —
   `vector_field` → `pmp_planner` action server → `runtime_corrector` action
   client → playback → controllers — has not been exercised during any of the
   corrector work. **Expect bit-rot; budget the first run for finding it.** This
   is the single largest unknown in Demo A and the reason it is "believed
   complete" rather than "complete".
2. **The patch layout is wrong for a demo.** The current list puts `icy` *under
   the spawn point*, so the robot starts already slipping — deliberate for
   corrector testing, useless for a demo, where you want: drive clean → hit one
   patch → visible excursion → visible recovery. It is a hardcoded JSON literal
   in `gz_sim.launch.py`, so authoring a demo scenario currently means editing a
   launch file. A `PATCH_LAYOUT` variable pointing at a config file would fix
   that and is the one code change Demo A wants.
3. **Patch strength for visibility.** The excursions we measure are 0.3–2 m.
   Whether that reads as "off track" on screen is an eyeball question nobody has
   asked. Adjacent knob: patches below the wheel's $\mu_2 = 0.45$ mean *no
   steering at all* (deliberately), which looks like a broken robot rather than a
   slipping one — so a demo patch should sit just above the knee, not at `icy`.
4. **Recording**, if the demo needs to be shown rather than watched live —
   `ffmpeg -f x11grab` on the VM, or capture Moonlight-side.

**Expect this to be an afternoon, dominated by (1).** It is worth doing *before*
the re-planner build, for a reason beyond the demo: it is the only end-to-end
check that the thing we have been tuning for weeks behaves in the actual runtime
pipeline, and not merely in the measurement harness.

One thing that will look like a hang and is not: in `offline` mode the corrector
buffers the **whole** rollout before driving, so the robot stands still for the
entire planning phase (~20 s on the baked map) and then drives. Documented
behaviour, not a fault.

### Demo B — what it adds

Everything in "The build plan for Level A", plus:

- a **scenario where TVLQR genuinely cannot cope** — the natural one is a patch
  below the friction knee, which is also Theorem 2's branch-2 case, so it doubles
  as the experimental demonstration of the dichotomy the dissertation asserts but
  never shows;
- the **trigger** wired into `runtime_corrector._correct()`, reading
  `CorrectionDiagnostics.saturated_*`;
- **no splicing plumbing** — under the hybrid output above the reference is
  simply whatever the network says this tick, so `trajectory_buffer` is
  untouched. (An earlier draft of this document called splicing the one
  genuinely new piece of runtime plumbing; that is retracted.)
- a fourth `CORRECTOR` arm, which the existing seam already accommodates.

Demo B is gated on phase 0 (does the re-join solve reliably?), so the honest
estimate is "unknown until phase 0 runs", and phase 0 is cheap.

## Open questions for the advisor

- **Sand** is inexpressible in the current friction model: under min-combination
  with an isotropic ground, any ground below the wheel's $\mu_2$ gives ratio 1,
  so "slides but still steers" cannot be represented at low friction. Needs a
  different mechanism, not a parameter.
- **The platform mismatch.** The source's model is a 3-state unicycle
  $(x, y, \theta)$ with controls $(v, \omega)$; ours is a 5D skid-steer
  wheel-space model with per-wheel accelerations, and the skid-steer $\chi$ has
  no counterpart in the source formulation.
- **The server.** Theorem 2 makes the communication delay $\tau$ an explicit
  hypothesis, and the offloading argument cites DShot/PWM frame budgets — a
  flight controller, not a Jetson. On our platform the *premise* wants
  re-checking even though the architecture is sound where it holds. Note the
  user's position, which is narrower and compatible: keep the catalogue onboard
  in compressed form, so no solve and no round trip happens at correction time.
