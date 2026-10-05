#!/bin/bash
# Does controllability along a plan predict how hard that plan is? (theorem check)
#
# WHY. The advisor wants the paper's theorem rebuilt so the experiments follow
# from it (../paper/porting-notes.md, "Theorem rebuild", 2026-09-23). The
# proposed main hypothesis is that the tracking-error dynamics stay UNIFORMLY
# CONTROLLABLE along the plan, and the proposed bound degrades as the
# controllability Gramian shrinks. Before anyone writes a proof around that
# hypothesis, check whether it separates easy plans from hard ones at all.
#
# PREDICTION (recorded before the run, in tools/controllability_gramian.py):
# Spearman rho between the per-plan Gramian energy (energy_p90) and the
# per-plan mean J of the adopted corrector is POSITIVE.
#   - rho clearly > 0 : the theorem's main branch explains the experiments;
#                       say so in the paper, with this correlation as evidence.
#   - rho ~ 0         : controllability is not what makes a plan hard here.
#                       The theorem can still be true, but must not be sold as
#                       explaining the gain tables. Record it in CLAUDE.md.
#
# DATA. Joins job 100 (soak_broad_r_at_q25.jsonl), arm 2.5/2.618 only: one arm,
# one campaign, mean-of-5 over the 40 broad plans. No Gazebo; takes ~a minute.
# Queued ahead of the Gazebo jobs so its answer exists first.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

PLANS=$(sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt)
echo "[gram] starting at $(date -Is)"
python3 tools/controllability_gramian.py --plans $PLANS \
    --soak "$HOME/run_data/2026-08-18_job100_broad-r-at-q25/soak_broad_r_at_q25.jsonl" --arm 2.5,2.618 \
    --out "$HOME/run_data/2026-09-23_job105_gramian-broad40/gramian_broad40_q25.csv"
echo "[gram] finished at $(date -Is) rc=$?"
exit 0
