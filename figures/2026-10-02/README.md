# 2026-10-02 — Nav2 comparison v7 (#10, #34): PRELIMINARY, not for publication

This is the first complete comparison run with no freezes (#32): 40 broad v2 plans ×
{ours, nav2-mppi, nav2-rpp} × 2 seeds, 240 cells, with a fresh full stack for each cell.
Conditions: amcl localization, floor 6, `mu2=0.45`, surface patches on. The amcl
`/map` reset bug from 2026-09-29 is fixed. Scoring uses Gazebo ground truth.

**Why it is stamped preliminary:** arrival is low for every arm. Ours arrives in
25/80 runs, MPPI 28/78, RPP 13/78. The failure analysis below changes how the
corrector result reads, and the fix belongs to the planner, so these numbers
are not the system's final ones.

| file | shows |
| --- | --- |
| `failures.png` | left: how each run ended (arrived / touched a wall / stalled with no contact / other / planner failed). Right: the reference plans' own minimum clearance |
| `metrics_seed{0,1}.png`, `summary_seed{0,1}.txt` | per-arm table, box plots and paired sign tests vs ours |
| `tracks_floor_6_v2_00105.png` | ours and RPP clip a doorway corner the plan cuts; MPPI goes round and arrives |
| `tracks_floor_6_v2_00249.png` | all three arrive |
| `tracks_floor_6_v2_00419.png` | a long route only ours completes (MPPI times out, RPP fails) |
| `tracks_floor_6_v2_00428.png` | ours and RPP arrive; MPPI times out |

**What it establishes:**
- **Our failures are wall strikes.** 32 of our 42 missed runs touched a wall, and
  21 of those plans already pass within 5 cm of a wall. The reference plans
  themselves have a median clearance of only 3 cm beyond the footprint, and 20%
  of them overlap a wall. The soak bench that reported an 11% miss rate
  (`rl_corrector.world`) is a bare ground plane, so it never had walls to hit.
  Tracking error that costs nothing there becomes a collision in the building.
  The bottleneck is **the plan's clearance (FM2 speed profile / footprint
  margin), not the corrector.**
- **MPPI fails differently.** It almost never touches a wall (1/80). It stalls or
  wanders near the goal and times out: 18 stalls with no contact, 32 "other".
- **Robust in both seeds:** compared with MPPI, ours drives about 3.5× faster
  (drive time ~28 s vs ~105–115 s) and uses about 2.5–3× less control energy,
  with no difference in `final_err` (p≥0.39). Compared with RPP, ours has better
  `final_err` (p=0.005 and p=0.014). Nav2 starts moving within 2–4 s; ours
  stands still for about 9.5 s while it plans.
- Ours fails to plan 13 of 80 times: the known BVP mesh exhaustion.

**Next (#34):** add a footprint/clearance margin to the FM2 speed profile (or
inflate the map the planner uses), then re-run. The question is whether wall
strikes disappear without losing the speed/energy advantage.

Data: `run_data/compare_v7_seed{0,1}/` (gitignored; VM `~/compare_broad40_v7`).
Regenerate with `.venv/bin/python figures/2026-10-02/render.py`.

## Follow-up: the FM2 field and the true footprint (`field_failures.png`, `render_field.py`)

This recomputes the field exactly as `vec_pmp.launch.py` configures it
(exponential profile, R = 0.5 m) and checks the real **0.63 × 0.585 m rectangle at
the plan's yaw**, instead of the 0.29 m circle used above. The circle statistic
above ("20% of plans overlap") is a rough proxy. These numbers replace it:

- **The doorway mechanism works.** Plans go through the middle of doorways (00105).
- **Where a plan does overlap a wall, it is a yawed corner.** In 92% of overlapping
  plan poses the robot's centre is 0.30–0.41 m from the wall: outside the
  inscribed radius, inside the circumscribed one (0.43 m). FM2 treats the robot
  as a point and slows it isotropically, so nothing in the field knows
  that a turning rectangle sweeps its corners outward.
- **About half our failures have a clear plan.** 41 of our 42 misses touched a wall
  (true footprint). In 21 of those the plan itself overlaps. In the other 20 the
  plan is clear and the robot left it: median 0.40 m off-plan at first
  contact (IQR 0.23–0.58 m), typically by turning early or overshooting a turn
  (00105, 00369). That is the corrector's known weakness at turns.
- 13 of our 25 arrivals also grazed a wall on the way.

## RETRACTION (2026-10-02 evening): the "plan overlaps" were the wrong plan

`plan_poses` in the v7 track files (`tools/compare_run.py`) is the **library**
plan from `traj_data_v2` (186 poses for 00105), not the plan the stack solved live.
The live planner runs with `STACK_PLANNER_PARAMS`, and for 00105 it produced 280 poses. `tools/replan_footprint.py`
reproduces the live plan exactly (same pose count, same start). Against the
**live** plans, only 1/40 (00419) overlaps a wall. The "21 plans overlap" split
above is withdrawn. **Nearly all of our wall strikes are tracking**: the
robot's max distance from the live plan has a median of 0.51 m, and 65/70 runs exceed 0.2 m.
The yaw-aware footprint barrier (`w_fp`, commit 2046190) does no harm: across 0/20/100/500,
nothing changes on the 40 broad plans or the 19 narrow doors on floors 2–6. All
doors are crossed with ≤5.5° yaw error at w_fp=0, because there is nothing for it to fix, so it stays off.
