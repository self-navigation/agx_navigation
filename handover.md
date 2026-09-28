# Handover — 2026-09-23

**Read this first, and keep it current.** It is the primary record of what we
are doing; CLAUDE.md is the cumulative record of what we have *established*.
This file describes **now**: what is running, what is half-finished, what to do
next, and the reasoning behind decisions that have not yet become findings.
Rewrite it rather than appending.

**This session in one line:** paper work, no experiments. Re-injection items
3-7 are done, and so is the derivation audit. The advisor has **answered the
questions**. Nothing is running anywhere.

**2026-09-23: the advisor's answers change the plan** (full text and reading in
`../paper/porting-notes.md`, "The advisor's answers"):

- **The venue is now MDPI *Mathematics*, and 23 pp is only a recommendation.**
  The trim that used to be step 2 is **cancelled**. The draft is now 26 pp.
- **External baselines (Nav2) are required.** Our self-comparisons do not
  count. Step 3 below stands, and it is still the gating item.
- **The theorem has no source to cite, and it is ours to rebuild** so that
  every experiment follows from it. For a mathematics journal this is now the
  second gating item. The obstacles are listed in porting-notes, "Theorem
  rebuild".
- **Gazebo is enough.** Anything that also runs on the real Scout gets
  reported there, and the paper says why the rest is Gazebo-only.

**Three more provenance errors were found and fixed** while re-injecting, all
of the form "number quoted under the wrong gains". The seven-shape headline
2.127 → 1.127 is TVLQR at the **original** `10/0.25`. The 51-plan robustness
trade (9.99 m / 2.03 m) is `0.276/2.618`. And 88.8% → 2.6% is not from the
1047-run ladder. **The adopted `2.5/2.618` has only ever been measured on the
40 broad plans (job 100).** Keep that in mind before quoting any other table as
"the tuned corrector".

**The derivation audit found one substantive gap.** The solved BVP is the PMP
system only up to `pos_gate`, a heuristic gate in `H_v` with no counterpart in
`L`. The paper now says so, and the PMP docstring in `node.py` was fixed to
match the code. It had put the gate on `λ̇_θ`.

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

## Jobs 110-130 (FINISHED 2026-09-23, all rc=0 — see issues #1-#3)

Three jobs in the VM queue, one after another. **Each job file's header says
why it exists and how to read either outcome**. Read the header before reading
the numbers.

| job | what | output (VM `~`) | est. |
| --- | --- | --- | --- |
| `110_seven_three_arms` | the seven shapes: open loop, `10/0.25`, `2.5/2.618`, **one campaign**, 100 repeats | `soak_seven_three_arms.jsonl` | ~2 h |
| `120_broad40_open_loop` | the 40 broad plans: same three arms, mean-of-5 | `soak_broad_open_loop.jsonl` | ~45 min |
| `130_gramian_joins` | the controllability check re-run against 110/120 | `gramian_*.csv` | ~1 min |

Logs: `~/jobq/logs/<job>.log`. Status: `just queue-status`. **Why these
exist**: job 110 gives the paper the seven-shape table at the gains it actually
adopted, because the current table uses the old ones. Job 120 answers "does the
corrector beat open loop on plans we did not choose?", which has never been
measured. Early rows match August (open-loop zigzag 7.67 m vs 7.06 +/- 0.71;
`10/0.25` S-curve 2.130 vs 2.128), so the plant has not drifted.

The soak harness gained an **`identity` arm** for these jobs (`--gains
identity`, `soak.py` + `variance_probe.drive`). Its `j_control` is 0 by
definition, so J alone flatters open loop. Read J's tracking and terminal
parts, plus arrival.

**Already answered (job 105, run by hand 2026-09-23): the theorem's
controllability hypothesis does NOT explain which plans are hard.** Gramian
energy vs per-plan J of the adopted corrector over the 40 broad plans:
Spearman rho = +0.09 (p = 0.59). The prediction was recorded before the run
and failed. The only correlate is `frac_slow` (share of time with |v| < 0.05):
+0.43 (p = 0.006) vs J. That may be a duration artefact, since J is an
integral, so do not quote it as a finding. Output:
`~/gramian_broad40_q25.csv`. Consequence written into the theorem proposal:
the differences between plans come from delta (how much slip a plan meets), not
from controllability.

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

## Do this next

**Moved to the Forgejo tracker on 2026-09-28** — list open issues with the `fj`
tools. Jobs 110-130 finished cleanly on 2026-09-23, so start with #1 (plant
drift check + `tab:seven`), then #2 and #3. Gating: Nav2 baselines #10 and the
theorem #8 (blocked on the advisor).

## Standing rules that a new session breaks first

- **Do not lower the ground plane to model a slippery floor.** Slipperiness
  belongs in the wheel pair or a patch.
- **Do not run another seven-plan gain search.** Twice now an optimum found on
  the seven evaporated on 40 independent plans. Tuning is closed.
- **`localization:=none` cannot evaluate a pose-feedback corrector.**
- **One sim per partition**, and `tmux kill-server` does not stop a sim — it
  orphans it. This session cleaned up five weeks of exactly that.
