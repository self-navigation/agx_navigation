# 2026-10-06 — job 220 part 1: full stacks vs Nav2 (#10, #33)

Data: `run_data/2026-10-06_job220_fullstack/` (VM `~/run_data/` same name), 40
broad plans x 2 seeds, amcl, solid walls. O = ours (FM² + live PMP + TVLQR
2.5/2.618), M2 = Smac2D + MPPI, MH = Hybrid-A* + MPPI, R2 = Smac2D + RPP;
Nav2 arms use `compare_skid`. Scoring as `tools/summarize_arms.py` (timeouts
scored; planner-/stack-failed excluded — O had 8 planner failures, #39).
Regenerate: `.venv/bin/python figures/2026-10-06/render.py`.

| figure | shows | establishes |
| --- | --- | --- |
| `job220_final_err_paired.png` | per-run final error, ours (x) vs each Nav2 stack (y), log-log | ours ties Smac2D+MPPI (36/70, p=0.9), borderline beats Hybrid-A*+MPPI (43/71, p=0.1), beats RPP (51/72, p=5e-4). The scatter is wide on both sides: per-plan winners differ |
| `job220_cost_ratios.png` | paired Nav2/ours ratios of travel time, control energy, path length | ours ~3x faster and ~4.8x cheaper than Smac2D+MPPI in nearly every pair. Ratios far BELOW 1 on path length are Nav2 runs that aborted early, so their low time/energy is not a win |
| `job220_tracks.png` | tracks of all four stacks on three mechanically picked plans (largest ours-better, largest MPPI-better, first both-arrive) over the baked map | 00219: all Nav2 stacks stall at the corner into the narrow north corridor, ours enters it. 00061: everyone reaches the door turn at x≈-4, only MPPI gets through. 00486: all comparable |

Status: exploratory, not yet for the paper. The paper text (sec:stack, paper
commit 185e249) quotes the same numbers.
