# Supervisory layer + ε-TD3+BC: implementation plan (#40)

Agreed with the advisor on 2026-10-06. Deadline: at most 2 weeks; quality comes before speed.
This plan was drafted by a planning agent from the code and amended by the main session.
Each segment has its own Forgejo issue and a **reporting deliverable**: a figure or table
that can go to the advisor as soon as the segment is done.

## Design decisions (fixed before coding)

- **Q1. Nav2/GMPC interception.** A separate node `supervisor_node` sits between the
  controller output and `/cmd_vel`. collision_monitor's `cmd_vel_out_topic` is renamed to
  `/cmd_vel_nav_out`, and GMPC's `cmd_vel_topic` is pointed at the same topic. A BT plugin
  was rejected because it would be C++ and the arms differ in their BTs. collision_monitor
  was rejected because it handles geometry only and is disabled in `compare_skid`.
- **Q2. Replanning.**
  - Full Nav2 stacks: cancel and re-send NavigateToPose, so Smac2D replans from the current pose.
  - Same-plan arms and FollowPath servers: call `ComputePathToPose` (Smac2D) and re-send FollowPath.
  - GMPC runs under `nav_mode:=nav2`, so `planner_server` exists, with its own action name.
  - Ours: PMP re-join (tier 2). If that fails, stop and do a full PlanToGoal (tier 3).
    `ours-lib` cannot replan, so it is stopped and scored as failed.
- **Q3. Our stack.** A pure `SupervisorCore` is called inside `WheelCorrectorNode._on_tick()`
  before `_emit`. It re-times and splices the TrajectoryBuffer, which only the corrector can do.
  The same core is wrapped by `supervisor_node` for the other stacks.
- **Q4. Re-join.** `TrajectoryBuffer.splice(samples, resume_index)` splices the re-join in, and
  TVLQR tracks the spliced reference. The target is plan point N+1. `T_w = max(3 s, time to N+1)`;
  if N+1 is closer than 3 s, the target moves to N+2.
- **Q5. Waypoints.**
  - A waypoint every 5% of plan **arc length**. It counts as reached within 0.35 m (the robot radius), and the index is monotone.
  - The timer runs per segment and resets at each waypoint. On reaching a waypoint, the cursor is re-timed to it.
  - The trigger fires when `e_i >= 2 e_{i-1}` AND the lag, converted to distance through the segment's plan speed, is more than 0.35 m. The paper states this convention.
  - Nav2 controllers are already path-indexed, so for them only the trigger and the wall layer apply.
- **Wall safety.**
  - Clearance comes from the baked-map EDT at the hull perimeter, evaluated at the believed pose (`map->base_link`).
  - The speed scale is `clip((d-0.15)/(0.5-0.15), 0, 1)`. At 0.15 m the robot stops and replans.
  - It is the same for every arm.
- **Q6. Reward.** `epsilon.step_cost` with a new `CostWeights.adopted()` (q_cross 2.5, r_omega 2.618).
  The defaults stay as they are, so soak `j_total` stays comparable. Terms are added for the
  terminal pin error at N+1 and for contact.
- **Q7. Slip model.**
  - χ(μ) is interpolated from `sweep_data/ground_mu_chi_mu2_045.csv` (μ ≥ 0.5).
  - At μ ≤ 0.45 the yaw gain is about 0.05: the branch where the robot cannot steer.
  - The longitudinal factor 0.9 below the knee is an ASSUMPTION.
  - The surface comes from the same patch list `spawn_patches` generates.
- **Q8. TD3 action.** 6 (v, ω) knots over T_w, so the action is 12-dimensional. The PMP label is
  sampled at the same knots. The state has 10 dimensions: e_along, e_cross, e_θ, Δw_l, Δw_r to
  N+1, v, ω, wall clearance, χ̂ and T_w.

**ε-TD3+BC objective** (the novelty, accepted "for now"):
`max E[λ Q(s,π(s)) − β(s) m(s) ‖π(s) − a_PMP(s)‖²]`. Here
`β(s) = β0 exp(−ε̂(s)/ε0)` and `ε̂(s) = J_χ[a_PMP] − J_χ0[a_PMP]`, the teacher's SVCM ε-gap under
the measured slip. `m(s) = 1` iff the PMP solve succeeds. With no admissible control,
`m = 0` and the safety layer stops the robot.

## Segments

| id | segment | owns | depends on | effort | deliverable for the advisor |
|---|---|---|---|---|---|
| S1 (#41) | SupervisorCore + our stack | `agx_planning/supervisor/{core,walls}.py`, `runtime_corrector/{node,trajectory_buffer}.py`, `vec_pmp.launch.py`, Makefile `SUPERVISOR` | — | 1.5 d | One plan, robot track with waypoints, the per-segment time error, the trigger moments and the wall-clearance speed scale (truth, with vs without the layer) |
| S2 (#42) | supervisor_node for Nav2 + GMPC | `supervisor/node.py`, `nav2.launch.py`, `gmpc.launch.py`, `gmpc/node.py` (retime), `stack_ready.py` | S1 core API | 2 d | GMPC on one plan with and without the layer (reference index vs time); MPPI stopping at a phantom wall |
| S3 (#43) | Harness + metrics + job scripts | `compare_run.py`, `fixture_up.sh`, `summarize_arms.py`, `tools/jobs/23x,24x` | S1, S2 | 1.5 d | Table schema + a smoke cell per arm type |
| S4 (#44) | Slip model in KinematicBridge + G2 check vs Gazebo | `rl_corrector/{slip_model,kinematic_bridge}.py`, `tools/validate_slip_model.py` | — | 1 d | Kinematic vs Gazebo tracks on 10 plans + χ(μ) curve, as a gate table |
| S5 (#45) | PMP teacher: label dataset + runtime RejoinSolver | `tools/rejoin_dataset.py`, `pmp_planner/rejoin_service.py`, `test_rejoin.py` | S4 (for J_χ) | 1.5 d | Solve-rate and solve-time histograms on triggered states; example re-join trajectories |
| S6 (#46) | TD3 / ε-TD3+BC | `rl_corrector/{rejoin_env,td3_bc,train_rejoin,rejoin_policy}.py` | S4, S5 | 2.5 d | Learning curves for TD3 vs ε-TD3+BC vs BC-only; −J against the PMP teacher; a map of β(s) |
| C1 (#47) | Campaign: same plan ± layer | job 230 | S1–S3 | ~10 h VM | Miss rate / contact-free arrival / final error for each arm, with vs without the layer, truth + amcl |
| C2 | Campaign: full stacks ± layer | job 240 | C1 | ~16 h VM (2 seeds) | The same for full stacks, plus self-reported vs actual arrival |
| C3 | Campaign: re-join source | job 250 | S6, C1 | ~2 h VM | ours + layer with PMP vs (ε-)TD3 re-join |
| S7 | Paper | `../paper/draft.tex` | all | 3 d | PDF |

Amendments to the drafted plan:
- C2 keeps **2 seeds**, run overnight, rather than cutting to seed 0.
- GMPC replanning is settled as `nav_mode:=nav2` (see Q2).

## Schedule (d1 = 2026-10-07)

- d1: S1 core API frozen first (main agent). S4, S5 and the S2 skeleton run in parallel worktrees.
- d2: S1 wiring and smoke, S2 wiring and smoke, S3 flags. Gate G2 (S4).
- d3: launch C1. S5 labels with `--jobs 4` while sims run. S6 env and trainer.
- d4: C1 running. S6 trains TD3, then ε-TD3+BC, on the V100.
- d5: C2. Gate G3 (S6).
- d6: C3. Read C1 and C2.
- d7–10: write (S7). d11–14 is buffer for one re-run.

Critical path: S1 core → S1 wiring → C1 → C2 → writing. S6 is off the critical path until d6.

## Gates

- **G1, re-join solve rate:** PASSED (#11, 99% at T_w ≥ 3 s). The residual check is the solve rate on
  states that actually triggered in C1. Below 90%, the m=0 branch dominates and the paper says so.
- **G2, slip model vs Gazebo** (d2): the deviation direction must agree on ≥ 8/10 plans and the
  final error must be within ×2. On failure, train on `GazeboBridge` (workers 13/14) or with χ randomised
  over [1.3, 2.0].
- **G3, TD3 vs PMP** (d5): −J within 10% of PMP on 2k held-out samples. If plain TD3 fails,
  use ε-TD3+BC. If both fail, the runtime PMP re-join ships, and ε-TD3+BC is reported as the objective
  with a partial or negative result.

## Risks

- A launch regression costs a day of VM time. Freeze a checkout per job and smoke every arm type before launch.
- Same-plan arms with replanning are no longer "same plan". Label them "same plan + layer" and
  compare misses and contact, not J.
- Under amcl the pose error (0.2–0.3 m) exceeds the 0.15 m stop threshold, so expect false stops.
  Measure `supervisor_stops` and report it as the cost of localization.
- The baked-map EDT does not see new obstacles. Dynamic environments are out of scope.
- The corrector does not read measured wheel speeds yet. The re-join x0 needs them (`/joint_states`).

## Minimal publishable fallback

S1–S3 with C1 and C2, the PMP runtime re-join (S5), the slip model with its G2 table (S4), and
ε-TD3+BC as the stated objective with whatever G3 produced.
