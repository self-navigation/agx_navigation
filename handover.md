# Handover — 2026-09-30

**2026-09-30 15:45: TWO VM JOBS RUNNING, beside v3.**
- `rejoin_phase0.py rescue` on the 2000-problem failures: `~/rejoin_rescue.{log,jsonl}`, ~30-40 min.
  Per failed pair it tries a bigger mesh, then multistart (blend, sideways bumps), then homotopy in the
  deviation (`alpha_max`), then a stretched window (`tw_solved`). On a 24-pair laptop trial, nothing
  rescued at the same `T_w`. Stretching solved them at 1.0-2.1 s, so the failures are "window too short",
  not solver luck. Phase 0 is deterministic: repeats are pointless, only the guess matters.
- #27 bias, 40 plans, all wheels x0.9: TVLQR on worker 7 (`~/bias27/rows.tvlqr.jsonl`) vs identity on
  worker 8 (`rows.identity.jsonl`). **Still WITH GUI** (launched before the headless default was
  deployed). The smoke row timed out at rtf 0.022, the same collapse as v3. Discount wall-backstop
  timeouts. `fixture_up.sh` is now headless by default (`FIXTURE_GUI=true` to watch), but only
  after the next `just sync`.

**2026-09-30 15:30: STATUS.**
- **The v3 rerun (#10) is still running on workers 1-6.** It is slower than planned, so the ETA is late
  tonight or tomorrow morning, not 17:45. At ~15:10 seed 0 had ours 13 / MPPI 3 / RPP 4 rows, and
  seed 1 had 7 / 3 / 7.
  - Cause: some cells' sim RTF **collapses** to 0.001-0.05 (normally 0.45-0.63) and then hits the 900 s wall
    backstop, recorded as a timeout. This happens **in all arms, ours included** (4 cases).
  - The track's sim-time sampling stays even, so the sim slows rather than freezes.
  - VM at the time: 48 cores ~50% idle, load ~40, CPU PSI some ~12% / full 0, RAM 14/31 GiB, GPU 38%.
    So the CPU upgrade helped, and this is not raw CPU starvation.
  - Suspects: `fixture_up.sh` runs `HEADLESS=false` with software rendering (libEGL falls back to
    `kms_swrast`), so six `gz sim -g` GUI clients burn ~100% CPU each.
  - **Before quoting v3, treat wall-backstop timeouts as invalid cells (not arm failures), and
    re-run them.** For the next campaign, consider headless.
- **Phase 0 (#11) is done**: figures and README are in `figures/2026-09-30/`, and there is a 2000-problem VM run.
  Min-effort re-join solves 99% at `T_w >= 3 s` (4 ms median). Short windows fail because of physics. Next
  for #11: decide `T_w` (>= 3 s) and move to Phase 1 (dataset), per `docs/corrector-design.md`.
- **#27 bias test is wired**: `wheel_bias` node (agx_chassis) plus the launch arg `wheel_bias:=fl,rl,fr,rr`,
  with `--wheel-bias` in `fixture_up.sh` / `compare_run.py`. The row carries `wheel_bias`. Deployed.
  - A smoke test (1 plan, all ×0.9, worker 7) is running: `~/bias27/smoke.{log,jsonl}`.
  - If it arrives, run all 40 plans on worker 7 (and 8), comparing ours+TVLQR vs `--corrector identity`, both biased.
  - If the node is missing from the stack log or the robot never moves, suspect the launch remap
    (`PythonExpression` in `vec_pmp.launch.py`).

**2026-09-30 14:00: host recovered and hardened (#26 comment has details).** VM 200
is now 32 GiB, `onboot 0` (start it by hand after a host reboot: `qm start 200`);
VM 100 is 160 GiB, `onboot 1`. Guest autostart waits 300 s; `touch
/etc/pve-hold-guests` on the host to skip it. VMs cannot swap, earlyoom kills kvm
first (ours before VM 100), and `psi-reboot` reboots on a sustained memory stall.
Untested by a deliberate wedge (no one on site). #26 closed after both VMs ran
together (host: 54 GiB available, no swap growth, PSI 0).

**2026-09-30 14:12: the v3 comparison rerun (#10) IS RUNNING**, planned job 2
below, deployed at the current HEAD. Seed 0 is on workers 1-3 and seed 1 on 4-6,
writing to `~/compare_broad40_v3/seed{0,1}/` with logs `~/cmp_v3_s{0,1}.log`.
ETA ~17:45 local. The first cells are clean: 1 map receipt per stack, and
`solve=` lines are present (cold 579 ms, warm 135-146 ms under 6 sims).
`~/ram_v3.log` logs guest RAM each minute (PVE can't see it, since the balloon
is off): ~16 GiB used of 31 with 6 sims. If it died, relaunch from
`~/agx_navigation`, not `~` (the job path is relative); it resumes. Next: read
it with step 5.

**2026-09-29 14:20: THE PROXMOX HOST IS WEDGED (OOM); do #26 BEFORE ANYTHING
ELSE.** Resizing VM 200 to 48 cores / 64 GiB overcommitted the 251 GiB host.
The other VM has 200000 MiB, and the V100 passthrough pins all of our RAM at
start, so the balloon can't help. The host answers ping but not SSH, and the
on-site admins were asked to reboot it. Recovery steps and the hardening list
are in #26. Target for ours: **48 cores, 32 GiB**.

**Planned jobs once the host is healthy, in order:**
1. Verify the guest: `nproc`, `free -h`, `nvidia-smi`, `just route-check`, and
   no leftover sims (`pgrep -af 'gz[ -]sim'`). Then run `just remote-build`
   (b1a43f8 and da8508f must be deployed).
2. Fresh comparison rerun (#10), both seeds at once, into a NEW directory
   (never resume `~/compare_broad40_v2/`):
   `tools/agx-run --detach 'SEED=0 OUT_DIR=$HOME/compare_broad40_v3/seed0 WORKERS="1 2 3" bash tools/jobs/140_compare_broad40.sh > ~/cmp_v3_s0.log 2>&1'`,
   and the same with `SEED=1 …/seed1 WORKERS="4 5 6"` into `~/cmp_v3_s1.log`.
   Takes about 3.5 h. In the first cells, check the stack logs for one map
   receipt and for `solve=… warm=` lines. Also watch the guest's real RAM peak
   (it goes into #26 sizing).
3. Phase 0 (#11), niced, ~8 processes beside the rerun. This needs code first
   (written locally, no VM needed): an opt-in re-join mode in
   `shooting_solver.py`, with 5 terminal pins, a cost A/B switch and mesh
   T = T_w. Plus a sampler of 200–400 problems from the broad 40 (random k,
   deviations sized from the soak's TVLQR deviations, T_w log-uniform
   0.1–10 s) and unit tests. Report the failure rate vs T_w and deviation
   size for A and B, the failure causes, and the solve times.
4. #5: from the rerun's stack logs, `grep solve=` gives the warm/cold
   distribution. It replaces the 36/59 ms in `../paper/draft.tex:350,355`.
5. Read the rerun with `tools/summarize_compare.py` (instructions below).
   Compare ours against the soak (miss ~11%), note the amcl-vs-truth gap, and
   update the paper.
6. #25: 2-D track animations (no VM cost), later a Gazebo replay.

**2026-09-29 (afternoon): THE OVERNIGHT COMPARISON WAS INVALID; NOTHING IS
RUNNING. Rerun pending a VM CPU upgrade.**
- The static map publisher sent `/map` on a 2 s heartbeat
  (`publish_map.launch.py` default), and amcl rebuilds its particle filter on
  every map it receives, so the pose jumped every 2 s **in all three arms**
  (31–83 re-inits per run). Our corrector chased 11–28 m of phantom
  cross-track while ground truth said 0.39 m. Its rows are in
  `run_data/compare_seed{0,1}/`: **do not quote them**. Details in the #10
  comment; the live-map design follow-up is #24.
- Fixed in b1a43f8 (`publish_period 0` + amcl `first_map_only`). The smoke test
  (`~/compare_smoke2/`, 2 plans × 3 arms) confirmed **1 map receipt per run**.
  Plan 00369 defeats every arm (it crosses the ice patch).
- The same commit logs **PMP solve time with a warm/cold tag**
  (`[pmp_rollout] … solve=…ms warm=`), for #5. First reading under 3-sim load:
  cold 194 ms, warm 105–126 ms, vs the paper's unsourced 36/59 ms.
- `summarize_compare.py` now keeps planner-failed runs (never moved) out of the
  metric means, and adds a paired arrival test (da8508f).
- The rerun started 13:26 and was **stopped on purpose at ~13:45**: the user is
  adding vCPUs to the VM (no hotplug, so it needs a reboot). **Start it fresh**
  into a new directory, and do not resume `~/compare_broad40_v2/` (its few
  cells ran on the old CPU count). With more cores, run both seeds at once on
  workers 1-3 and 4-6.
- **Phase 0 (#11) is designed; code not written yet.** Re-join BC with all 5
  terminal components hard-pinned; `T_w` log-uniform 0.1–10 s; two cost
  formulations (A = the planner's field cost, B = minimum effort). The
  re-join network runs **onboard**; the server does replans and library
  growth (see the #11 comments). Run it after or beside the rerun, depending
  on the core count.
- Video previews: #25 (2-D from tracks first, then Gazebo replay with a color
  per arm).

**Superseded, kept for its read-out instructions: the overnight run (#10, stage 1).**
Launched 03:28 local (00:28 VM time). Two passes in sequence, seed 0 then
seed 1, each 40 broad plans x {ours, nav2-mppi, nav2-rpp}, one arm per worker
(1-3), every cell a fresh full stack (baked map, amcl, no SLAM). ~3.5 h per
pass; it finished 08:58 local, rc=0, and was found invalid (see above).
- Log: `~/compare_broad40.log` on the VM. Rows, tracks and stack logs are in
  `~/compare_broad40/seed{0,1}/` (`rows.w*.jsonl`, `track_*.npz`,
  `stack_*.log`, `up_*.log`). Each pass ends with a `[cmp] finished ... rc=`
  line.
- Read it: `rsync -az -e "ssh -F ssh_config" --include='rows*' --include='track_*'
  --exclude='*' agx:compare_broad40/seed0/ run_data/compare_seed0/`, then
  `.venv/bin/python tools/summarize_compare.py run_data/compare_seed0 -o <out>`,
  which writes summary.txt, metrics.png and tracks_<plan>.png.
- If it died: the job resumes (it skips rows already written). Rerun the same
  command, but check first with `pgrep -af 'gz[ -]sim'` that no sims are left.
- How to read it:
  - RPP's fix (`use_collision_detection: false`) is untested. If RPP still
    fails near its spawn, it is not a fair baseline yet; fix it or drop it.
  - MPPI arrives but stalls (tune4: 104 s vs our 28 s). Report the stalls as
    its behaviour on this map.
  - Ours: compare against its soak numbers (miss ~11% on the broad 40). A much
    worse miss rate means the full stack (amcl, the ROS pipeline) costs
    something the stack-less soak hid. That is itself a finding.
  - Seed 1 vs seed 0 gives the run-to-run spread per arm.
- Tuning decisions and the tune3/tune4 numbers are in the comment on #10;
  their outputs are in `run_data/compare_tune{3,4}/`. DWB is excluded (untuned).
- New issues: #21 (long corner-to-corner trajectories), #22 (labyrinth
  showcase figure).

**Read this first, and keep it current.** It is the primary record of what we
are doing; CLAUDE.md is the cumulative record of what we have *established*.
This file describes **now**: what is running, what is half-finished, what to do
next, and the reasoning behind decisions that have not yet become findings.
Rewrite it rather than appending.

**This session in one line:** paper work, no experiments. Re-injection items
3-7 are done, and so is the derivation audit. The advisor has **answered the
questions**. Nothing is running anywhere.

**2026-09-28: the theorem proposal was rejected, and the comparison is now the ONLY priority.**
The advisor said: no assumption list, keep the old form, papers don't have
"branches", leave the theorem as it is for now, and he is waiting for the
comparison with analogues. So the theorem rebuild is **parked**. The likely
substantive reason is that part (b) was Riccati/TVLQR, which he had already
asked to keep out of the paper three times; details in #8. Work #10 first.
Every remaining paper ask is now an issue (#16–#20). His original files
(seed docx, annotated draft-5) are in `../paper/advisor/` with a README.

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

- **VM: idle and clean as of 13:47 on 2026-09-29** (0 sims, 0 stacks),
  waiting for the vCPU upgrade and reboot. See the top of this file.
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
tools. **2026-09-28: jobs 110-130 are read and applied** (#1-#3 closed):
`tab:seven` rebuilt from job 110, job 120's open-loop comparison is in the
paper, `tools/summarize_soak.py` exists, outputs are in `soak_data/`. Job 130
did not support theorem part (b): controllability predicts open-loop
difficulty no better than the corrector's (all rho<0, none significant), so
nothing was added to the proposal. Gating now: Nav2 baselines #10 and the
theorem #8 (blocked on the advisor). Unblocked paper items: #4-#7.

## Standing rules that a new session breaks first

- **Do not lower the ground plane to model a slippery floor.** Slipperiness
  belongs in the wheel pair or a patch.
- **Do not run another seven-plan gain search.** Twice now an optimum found on
  the seven evaporated on 40 independent plans. Tuning is closed.
- **`localization:=none` cannot evaluate a pose-feedback corrector.**
- **One sim per partition**, and `tmux kill-server` does not stop a sim — it
  orphans it. This session cleaned up five weeks of exactly that.
