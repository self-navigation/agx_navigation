# 2026-09-29 — overnight Nav2 comparison (#10): INVALID, pipeline preview only

**These are not results.** During the overnight run the static map publisher
re-sent `/map` every 2 s, and amcl rebuilt its particle filter on each receipt
(31–83 times per run), **in all three arms**. Every controller drove on a pose
that jumped every 2 s. The tracks themselves are Gazebo ground truth, so they
really show where the robot went; what's invalid is the conditions it drove
under. Fixed in b1a43f8. A clean rerun is pending the host repair (#26).

What they do show: the comparison pipeline works end to end. That means 40
broad plans × {ours, nav2-mppi, nav2-rpp} × 2 seeds, a fresh full stack per
cell, ground-truth scoring, and paired tests.

Data: `run_data/compare_seed{0,1}/` (gitignored). Regenerate with
`.venv/bin/python figures/2026-09-29/render.py`. Plant: `mu2=0.45` with
surface patches on, floor 6, amcl localization.

| file | shows |
| --- | --- |
| `metrics_seed{0,1}.png` | per-arm arrival rate and box plots of final error, timing, path length, control energy and wall clearance |
| `summary_seed{0,1}.txt` | the table and the paired sign tests (planner-failed runs are excluded from the means) |
| `tracks_floor_6_v2_00249.png` | a plan all three arms complete |
| `tracks_floor_6_v2_00096.png` | only ours arrives |
| `tracks_floor_6_v2_00147.png` | the Nav2 arms arrive and ours fails |
| `tracks_floor_6_v2_00369.png` | crosses the ice patch; still fails every arm after the fix (smoke test) |

Two observations that should survive into the valid rerun:
- Ours stands still about 10 s while it plans the whole trajectory; Nav2
  starts in about 2 s.
- MPPI takes about 2.5× our drive time.
