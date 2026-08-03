# Handover — 2026-08-03, evening session

## TL;DR

The overnight tuning run finished and **its result must not be adopted** — it
resolved noise, not signal. Separately, the first *trustworthy* three-way
comparison now exists and TVLQR looks genuinely good. RL is back to inconclusive
after a clean re-measurement contradicted the training curves.

One concrete experiment is queued and it is the highest-value thing to do next:
**zero the wheel velocities exactly at reset** (see "Do this first").

## Do this first: the bit-identical reset experiment

Everything downstream — tuning, checkpoint ranking, parallel sims — hinges on
whether a rollout can be made exactly reproducible. Current belief chain:

- deterministic stepping is now honest (fixed last night, verified);
- but the **initial condition is not bit-identical**: wheels settle to ~1e-9
  rad/s rather than exactly 0, and `reset_settle_z_tol` converges position, not
  wheel speed;
- that ~1e-13 asymmetry is amplified at turn reversals (omega crosses zero,
  skid-steer lateral friction switches direction), which is why the hard shapes
  are noisy and the straight-ish one was not.

CLAUDE.md previously called this "genuine chaos, not worth chasing". **That was
written about `final_err` and over-generalised.** Chaos amplifies a seed; it does
not create one. If the seed is removed the rollout should be bit-identical.

So: after the settle loop in `GazeboBridge._reset`, explicitly set every wheel
joint velocity to exactly 0.0 (gz `set state` / joint velocity reset, not a
command — the controller latches) and re-verify. Then re-run
`tuning/variance_probe.py` on `floor_6_00056` (TIGHT V, sd 0.582 — the worst
offender) and `floor_6_00047` (ZIGZAG, sd 0.215).

**If it works**, single-sample ranking becomes legitimate again, the tuner can be
re-run as-is, and parallel sims stop being mandatory (still nice for training).
**If it does not**, the variance is genuinely irreducible, repeats become
compulsory, and parallel sims (queue item 7) move to the top. Either way this is
~1 h and it decides the shape of the next week — do not start another tuning run
before knowing the answer.

## The overnight tuning run: do not adopt its gains

Reported `q_cross=9.996 / r_omega=1.252` → 0.9412 m from a 1.1405 m baseline.
138 evals, 1.3 h. Data pulled to `tune_data/tvlqr_tune.jsonl`.

Reading the full JSONL rather than the reported optimum:

- the simplex **stopped moving at eval 49** and then re-evaluated ONE gain pair
  **71 times**;
- those 71 repeats — identical gains, trajectories, seed — span
  **0.9412–1.3052 m**, sd 0.0886, mean 1.0468.

The reported best is the **minimum of 71 noisy draws**. The claimed 0.199 m gain
is smaller than the 0.364 m spread it was selected from. Nelder-Mead cannot
converge on a noisy objective — it shrinks and re-samples forever, which is
exactly what the log shows after eval 49.

**Root cause is a generalisation, not a tuner bug.** The 0.0002 m noise floor was
measured on `floor_6_00042` alone — since dropped from the eval set for being an
L — and assumed to carry to the 7-shape set. It does not; noise there is ~400x
higher. Per-trajectory sd is in CLAUDE.md.

Figure: `figures_new/tvlqr_tune_variance.png` via `tools/plot_tune_variance.py`.

## First clean three-way comparison (this is the good news)

Seven shapes, repaired bridge, terrain on, RL at the 800k checkpoint.
`compare_data_new/`, figure `figures_new/corrector_summary.png` via
`tools/plot_corrector_summary.py`. Mean max|e_cross|:

| corrector | mean | note |
| --- | --- | --- |
| identity | 3.53 m | |
| **TVLQR** | **1.20 m** | −66%, best on 5 of 7 shapes, DEFAULT gains |
| RL (800k) | 2.08 m | best checkpoint, NOT typical — see below |

Two old claims die: **"TVLQR oscillates on S-curves"** (it is the best corrector
on the genuine S, 1.06 m vs identity's 4.22 m — the claim came from the L) and
**"no RL checkpoint beats identity"**.

## RL: re-measured clean, and it is inconclusive again

20 checkpoints of `runs_20260730` (stride 15) re-measured on the repaired bridge,
`--correctors rl` only. `sweep_clean/`, figure
`figures_new/rl_checkpoints_clean.png`.

| | |
| --- | --- |
| mean over 20 checkpoints | 3.475 m (identity 3.527) |
| Pearson r vs training step | **0.111 — no trend** |
| beat identity | 8 of 20 |
| beat TVLQR | **0 of 20** |
| range | 2.030 (@980k) – 4.579 (@305k) |

**The 800k checkpoint is a lucky pocket, not typical** — it was selected by the
*old contaminated* sweep, so quoting it as "the RL result" repeats the same
winner's-curse error as the tuning run.

**Retracted mid-session:** `rollout/terminal_abs_e_cross` falls 3.9 → 1.6 m in
TensorBoard, which reads as the policy learning while the optimiser diverged. It
does not survive — that metric was logged **by the mis-stepped environment**. The
clean sweep is the out-of-band check and shows no trend. The optimiser panels
(`ent_coef` → 3.31, `critic_loss` → 1.2e4) remain valid; they describe SAC, not
the plant. `tools/plot_training_diagnostics.py` says so on the figure now.

Checkpoint-to-checkpoint swings are ~2 m against ~0.09 m measurement noise, so
the policy genuinely lurches between saves — consistent with `ent_coef=3.31`.
Caveat: that 0.09 m is TVLQR's repeat sd used as a proxy; **RL's own measurement
noise has never been measured.** Worth one probe if anyone leans on that number.

## New tooling (all offline, venv matplotlib, no ROS dep)

| tool | figure |
| --- | --- |
| `tools/plot_corrector_summary.py` | grouped bars + mean, whole eval set in one figure |
| `tools/plot_tune_variance.py` | search order + repeat distribution at the collapsed point |
| `tools/plot_training_diagnostics.py` | 6-panel SAC diagnostics from the TB event file |
| `tools/plot_checkpoints_clean.py` | RL error vs training step, baselines as reference lines |

`tools/plot_training_diagnostics.py` needs `tensorboard` in the venv (installed).
TB events pulled to `tb_data/`.

## State of the VM

One `gz sim` (the long-lived rl-sim, headless), tmux session `rl` with 2 windows.
No compare/sweep/tuner running. `just check-sim` before launching anything.

The direct VPN route is back up, so plain `just <recipe>` works again — the jump
host (`just host='programmer@192.168.71.113' ssh_opts='-J llm_test2@kron.botik.ru
-p2202' …`) is no longer needed.

## Reporting artefacts (not code — safe to delete once sent)

- `otchyot.md` — exhaustive 41-item reference version, kept for fact lookup
- `otchyot_chat.md` — the conversational post sequence actually sent to the
  advisor on 2026-08-03, 6 messages with figure attachment points

Both are in Russian. The advisor has been sent: the corrector summary, two path
overlays, the RL training diagnostics, the clean checkpoint sweep, and the tuning
variance figure.

## Queue changes

- Item 4 ("explain the run-to-run variance") is **reopened** — the 2026-08-02
  answer was correct about the patch-spawn bug but the remaining variance on the
  hard shapes is unexplained-in-practice until the bit-identical reset experiment
  above resolves it.
- Item 7 (parallel sims) is **conditional** on that experiment failing.
- Items 1 and 2 (entropy runaway, bounded return) are unchanged and still the
  prerequisites for any retrain.
