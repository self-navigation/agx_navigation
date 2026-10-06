# 2026-10-07 / s4_slip — slip model χ(μ) and gate G2 (#44)

Plant: `2026-08-07-wheel-mu2-045` (wheel `mu2 = 0.45`, the patch weights in
`DEFAULT_PATCH_WEIGHTS`). Code: `rl_corrector/slip_model.py`,
`rl_corrector/kinematic_bridge.py` (`surface_fn`), `tools/validate_slip_model.py`.

Regenerate:

```bash
PYTHONPATH=src/rudn-ordjo-building .venv/bin/python tools/validate_slip_model.py \
    --out figures/2026-10-07/s4_slip/g2_results.json   # writes g2_results{.json,_tracks.npz}
.venv/bin/python figures/2026-10-07/s4_slip/render.py  # PNGs + the table rows below
```

Inputs (gitignored, in the main checkout): `sweep_data/ground_mu_chi_mu2_045.csv`
(copied into `slip_model.CHI_TABLE` as a literal), `traj_data_v2/`, the first 10
plans of `tools/jobs/broad40.txt`, and the identity rows of job 120
(`soak_data/soak_broad_open_loop.jsonl`, seed 0, terrain on, 5 repeats per plan).

## chi_of_mu.png — the model

χ(μ) (log scale) and the yaw gain k = 1/χ against ground μ, with the measured
`slip_ident` points and their spread. Two branches: μ ≥ 0.5 steers (χ 1.36–1.57);
μ ≤ 0.45 (the knee, = wheel `mu2`) has a yaw gain of 4–7 %, so the robot cannot
turn. That is the second branch of the dichotomy theorem. The knee rows have only 2
usable arcs each, so they are good to an order of magnitude, not a calibration.
The 0.45–0.5 gap is unmeasured and interpolated linearly in k. The longitudinal
factor 0.9 below the knee is an **assumption**. Note that `linoleum` (μ = 0.45)
sits exactly on the knee, so every patch profile in use is in the "cannot steer"
branch.

## g2_tracks.png — kinematic model vs Gazebo, open loop, 10 plans

Black: the plan. Red: KinematicBridge with χ(μ) over the seed-0 along-path
patches (drawn as rectangles). Green dashed: KinematicBridge with no slip.
Blue ×: Gazebo end points, 5 repeats. **Job 120 recorded no per-step traces for
the identity arm**, so only Gazebo end points can be drawn, not Gazebo tracks.
To get the tracks, rerun job 120's identity arm on these 10 plans with
`--trace-dir`: `tools/parallel_soak.sh --plans <10 plans> --repeats 1 --workers 1
--trace-dir ~/run_data/<dir>/traces -- identity`, about 2 min of VM time.

## G2 gate table

Direction = sign of the end point's lateral offset from the goal, in the goal
pose's frame. Gazebo's direction is the sign of the mean over its 5 repeats, and
the count in brackets is how many repeats share that sign.

| plan | Gazebo e_final (m) | Gazebo lat (m) | model e_final | model lat | ratio | dir ok | ×2 ok | no-slip e_final | time below knee |
|---|---|---|---|---|---|---|---|---|---|
| floor_6_v2_00369 | 2.584 ± 0.759 | -2.227 (5/5) | 3.888 | -2.602 | 1.50 | yes | yes | 0.202 | 28% |
| floor_6_v2_00482 | 0.397 ± 0.129 | -0.272 (5/5) | 0.341 | -0.232 | 0.86 | yes | yes | 0.038 | 45% |
| floor_6_v2_00147 | 0.463 ± 0.246 | -0.342 (5/5) | 0.867 | +0.752 | 1.87 | NO | yes | 0.181 | 36% |
| floor_6_v2_00105 | 0.181 ± 0.007 | +0.143 (5/5) | 0.490 | +0.356 | 2.71 | yes | NO | 0.054 | 44% |
| floor_6_v2_00208 | 1.621 ± 0.803 | -0.012 (2/5) | 2.974 | -0.562 | 1.83 | yes | yes | 0.410 | 18% |
| floor_6_v2_00486 | 0.191 ± 0.084 | -0.162 (5/5) | 0.424 | -0.389 | 2.22 | yes | NO | 0.131 | 34% |
| floor_6_v2_00096 | 0.803 ± 0.322 | -0.677 (5/5) | 2.954 | -2.774 | 3.68 | yes | NO | 0.062 | 17% |
| floor_6_v2_00249 | 0.489 ± 0.032 | +0.307 (5/5) | 0.413 | +0.233 | 0.84 | yes | yes | 0.147 | 27% |
| floor_6_v2_00302 | 0.441 ± 0.028 | -0.405 (5/5) | 0.451 | -0.447 | 1.02 | yes | yes | 0.026 | 20% |
| floor_6_v2_00443 | 2.716 ± 0.192 | -2.658 (5/5) | 1.093 | -1.090 | 0.40 | yes | NO | 0.566 | 11% |

| bridge | direction agrees | e_final within ×2 | G2 |
|---|---|---|---|
| **slip model χ(μ)** | **9/10** | **6/10** | **FAIL** (needs ≥ 8 on both) |
| no slip (reference) | 7/10 | 1/10 | FAIL |

**What it establishes.** The slip model gets the *direction* of open-loop
deviation right (9/10), and it moves the end-point error from about 10x too small
(no slip) to the right order of magnitude. It fails the ×2 criterion on 4 of
10 plans: it over-predicts 3 (×2.2–3.7) and under-predicts 1 (×0.4). Plan 00208
has no stable Gazebo direction (2/5 repeats agree), so its "agree" is weak.
Plausible causes, none of them tested: (1) the robot is a point, so a patch under
one side's wheels counts as nothing or as everything; (2) the knee rows are 2-arc
estimates with a spread of 3–6 in χ; (3) the 0.9 longitudinal factor is assumed.
The model was **not** tuned to pass. Per the plan's G2 fallback: train on
`GazeboBridge` (workers 13/14), or use this model with χ randomised.
