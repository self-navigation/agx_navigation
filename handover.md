# Handover — 2026-09-21

**Read this first, and keep it current.** It is the primary record of what we
are doing; CLAUDE.md is the cumulative record of what we have *established*.
This file describes **now**: what is running, what is half-finished, what to do
next, and the reasoning behind decisions that have not yet become findings.
Rewrite it rather than appending.

**This session in one line:** paper work, no experiments. **The advisor's
rewrite is now the base for `draft.tex`** (branch `advisor-revision-port` in
`../paper`), our corrected draft-5 is preserved at commit `97687d4`, and the
one real data bug in the paper is fixed. Nothing is running anywhere.

**Drive the paper work from this repo**, not from `../paper` — nearly every
open question needs the source, the soak data or CLAUDE.md's Settled stubs.

**The work has an external driver: the advisor reviewed the paper on
2026-09-16.** That review, not the re-planner, sets the next few sessions.

## The advisor's review (2026-09-16) — what he actually asked for

His revised TeX (a model-assisted rewrite of our draft, cut to 23 pages) came
over Telegram, so it is always re-downloadable.

**Superseded 2026-09-21: his rewrite is now the base, not a reference.** The
earlier plan here said draft-5 stays the source and his file must not be
committed. That predated reading his chat, which asks for **his** text
corrected (*«прошу ознакомиться с новым текстом, если нейронка наврала —
исправить»*), not ours shortened. Two things settled it: our corrected
draft-5 is safe in git history, and **the excess length is not where he thinks
it is** — cutting his named sections 8 and 9 recovers at most 2 of the 9 pages,
because our experiments were already 8pp and §8 was 1pp. The oversized block was
the OCP+PMP derivation at 7pp, which his rewrite already compresses to 4pp.

**The working checklist is `../paper/porting-notes.md`** — re-injection items,
the gain-number provenance, the substantive review comments with source-line
anchors, and how to re-derive them.

His four asks, in his order:

1. **Compare against other methods, not only against ourselves.** His words:
   without it "q1 не примут 100%" — a Q1 venue will reject it outright. This is
   the one that gates publication.
2. **Cut sections 8 and 9** (he would drop 9 entirely) and spend the space on
   the comparison experiments instead.
3. **Restore the figures from the original article.**
4. **Check the revised text for things the model got wrong** ("если нейронка
   наврала — исправить") and add the new experiments.

He also said he is impressed by the volume of work and thinks it has become a
genuinely good piece of research — so the framing is "finish it", not "rescue
it".

**Ask 4 is the one with a trap.** CLAUDE.md's live claims are the ground truth
for checking his revision, and several of our own numbers are easy to overstate
— in particular the tuning result is a **robustness trade**, not "tuning halves
deviation", and `J` is an **upper bound on epsilon, never epsilon itself**. If
the revision states either more strongly than that, it is wrong and we fix it.

Audited 2026-09-21. Both traps were sprung in **his** version, not ours: he
dropped the epsilon caveat entirely and blunted the robustness trade. Our
draft-5 was already correct on both, so ask 4 is mostly a **preservation**
problem while compressing. Both are now re-injected into the new base.

**One real data bug was ours, and he inherited it.** The 40-plan gain table
spliced two measurement campaigns — three rows from job 50 (mean-of-3) and two
from job 100 (mean-of-5). Cross-run comparability is an assumption we do not
hold. Fixed in both files; provenance in `porting-notes.md`. The visible
consequence: the miss-rate claim is **22.0% → 11.5%** read inside job 100, not
20.0% → 11.5% read across both.

**Ask 1 is bigger than "add Nav2 arms".** He wants a **dynamic** environment
(*«среда должна быть динамическая»*), and the advantage it is meant to expose —
committing to one plan and perturbing it locally, instead of Nav2 re-running A\*
and oscillating between equal routes — **depends on the re-join re-planner,
which does not exist yet**. So ask 1 and re-planner Phase 0 are the same piece
of work. See the `dynamic-obstacle-comparison-framing` memory.

## State

- **VM: clean and idle.** `pgrep gz sim` empty, job queue empty (`pending/` and
  `running/` both empty), load decaying from 9.9 to ~5 and still falling.
- **Cleaned this session:** the worker-1 fixture + GUI client from the
  2026-08-19 demo, and an **rviz2 that had been running 54 days at 74% CPU** —
  that, not the Gazebos, was the standing load. The old `rl` tmux session
  (created 2026-07-30) is empty but still listed.
- **New VPN IP `172.30.187.53`** (was `172.26.13.37`), fixed in `ssh_config`,
  `tools/agx-route`, CLAUDE.md and the memory file. The host key is **unchanged**
  — verified byte-identical to the one stored for the old IP before trusting it,
  so it is the same machine on a new address. `just route-check` picks `direct`.
- **CLAUDE.md was split** (see below). No code changed; `ssh_config` and
  `tools/agx-route` are the only non-doc edits.
- **Gains are closed** at `q_cross=2.5, r_omega=2.618` and the ROS stack is
  drivable unattended (`tools/stack_ready.py`, `just fixture-up`). Neither has
  been touched since 2026-08-19.

## What moved, so you can find it again

CLAUDE.md is loaded into every session and had grown to 1169 lines, most of it
reference detail. It is now 622 lines, and two new docs hold the rest **verbatim**:

- **[docs/vm-operations.md](docs/vm-operations.md)** — routing internals, the
  `WORKER` parallel-sim verification, the job queue.
- **[docs/measurement-rig.md](docs/measurement-rig.md)** — the plant (friction
  profiles, the `mu2` steering knee, `slip_chi`), eval trajectory selection, the
  tuning machinery, instrumentation traces.

The "failed plan hung every client" narrative moved to
`docs/corrector-history.md`; its durable facts survive as a stub under
"Settled". Nothing was deleted — every old section was checked to a new home.

## Do this next, in order

1. **Finish the re-injection checklist in `../paper/porting-notes.md`.**
   Items 1 (gain table) and 2 (epsilon bound) are done. Remaining: the
   robustness-trade numbers, the seven-shape enrichment guard, the evidence he
   cut that favours us (zigzag 88.8% → 2.6%, the mu/mu2 ratio invariance), the
   abstract's unqualified "2.127 → 1.127", and the mu2 contact-patch wording.

2. **Trim to make room for the comparison section.** The base built at 23pp and
   is now **24pp**; body 21pp, references 3pp. A comparison section needs 2-3pp,
   so **find 3-4pp**. The lever is his own instruction — *«сократить эксперимент
   — сделать таблицей, оставить 2 строки подписи»* — applied to
   Акспериментальная методика, which is **7pp**, the only block big enough.
   Nothing else exceeds 3pp. Free duplication: the geometric-mean rationale is
   stated three times (draft.tex lines ~407, 409, 411).

3. **Plan the baseline comparison (ask 1) — the gating item.** It is the same
   work as re-planner Phase 0 below, because the advantage only shows in a
   dynamic environment and the local re-plan does not exist yet. Design notes
   before anyone writes code:
   - **The comparable object is the whole FM2+PMP+TVLQR stack, not the
     corrector.** DWB/MPPI/TEB are closed-loop planners; our corrector is not a
     planner. Comparing "TVLQR vs MPPI" is a category error and a reviewer will
     say so.
   - **It must run through the ROS stack, not `GazeboBridge`.** Every corrector
     number in this repo came from the bridge, which bypasses the ROS graph —
     but Nav2 only exists inside the ROS stack. So this uses `make fixture` /
     `make nav2` and `run_recorder`, at ~90 s a run, not the soak harness. That
     is the main cost driver and the reason to scope it before starting.
   - **Same plans, same world, same patches, same seeds.** The 40 broad v2 plans
     are the set (`tools/jobs/broad40.txt`, `~/broad_eval_plans.txt` on the VM);
     Nav2 gets start/goal pairs, not our solved plans.
   - **`localization:=amcl` is the honest choice here**, not `truth`: this is a
     system comparison, and `truth` is our ceiling rather than anyone's
     deployment. Note it costs the lidar, so runs get slower.
   - Metrics from the draft's Experiment 4: path length, travel time, max
     curvature, control energy `int ||(a_l, a_r)||^2 dt` — plus arrival rate and
     `final_err`, which is what actually separated our own arms.
   - **Budget it with `WORKER`.** 40 plans x 4 arms x repeats at ~90 s is an
     overnight job only if it runs several partitions wide, and **the job queue
     is still single-lane** (CLAUDE.md queue item 5). Either drive workers by
     hand or build per-worker queues first.

4. **Re-planner Phase 0** — measure the re-join solve failure rate over ~200
   sampled problems. It is **offline and needs no Gazebo**, so it can run
   alongside the baseline work rather than competing for the sim. The library
   build failed 36%; inherit that and the supervised plan is wrong rather than
   slow. Full plan in [docs/corrector-design.md](docs/corrector-design.md).

5. **GUI camera follow — still BLOCKED on Moonlight.** The demo itself drove to
   arrival twice on 2026-08-19 (final_err 0.047 m). What fails is making the
   Gazebo camera track the robot through the API: the `CameraTracking` plugin
   loads via `--gui-config` with no error and does not follow (the keys exist —
   confirmed with `strings` on the .so), and an `xdotool` right-click on the
   Entity Tree row produced no visible context menu. **With Moonlight:**
   right-click `scout_mini` → Follow, then
   `tools/with-worker 1 python3 tools/drive_goal.py --x 6.0 --y 3.0`.

6. **Stratify a second eval set** into `config/eval_trajectories.yaml`, *added*
   alongside the seven rather than replacing them — swapping would silently
   re-baseline every number in CLAUDE.md. The seven stay the fast search set;
   the broad 40 are for claims.

7. **Launch races: still no failing case to collect.** `fixture_up.sh` retries
   and prints the failing check, so evidence accumulates for free. Four
   bring-ups have now been ready on attempt 1. Do not touch `main.launch.py`
   until something actually fires.

## Standing rules that a new session breaks first

- **Do not lower the ground plane to model a slippery floor.** Slipperiness
  belongs in the wheel pair or a patch.
- **Do not run another seven-plan gain search.** Twice now an optimum found on
  the seven evaporated on 40 independent plans. Tuning is closed.
- **`localization:=none` cannot evaluate a pose-feedback corrector.**
- **One sim per partition**, and `tmux kill-server` does not stop a sim — it
  orphans it. This session cleaned up five weeks of exactly that.
