# 2026-10-07 / s5_rejoin — the PMP re-join teacher (#45, S5; #35 phase 1)

Regenerate: `.venv/bin/python figures/2026-10-07/s5_rejoin/render.py` (reads
`run_data/2026-10-07_rejoin-sweep/` and `run_data/2026-10-07_rejoin-labels-smoke/`
in the main checkout; both are gitignored, see their README.md).

Plant: none — offline. Every problem is solved by `pmp_planner/rejoin_service.RejoinSolver`
(cost B, min wheel-acceleration effort, nominal `slip_chi=1.373`), the same code the
runtime corrector will call. Problems are drawn as in Phase 0 (`tools/rejoin_phase0.make_problems`):
one of the 40 broad floor-6 plans, a random index, a REAL TVLQR deviation
(`broad_gains_traces`, q1.5/r2.618) times a log-uniform scale.

| figure | what it shows | what it establishes |
| --- | --- | --- |
| `solve_rate.png` | solve rate vs `T_w` (left) and vs `\|e_xy\|` (right; all `T_w` and `T_w>=3 s`); sweep, n=2000, `T_w` log-uniform [0.3, 10] s, scale [0.5, 8] | Reproduces Phase 0 through the runtime wrapper: **99.3% solved at `T_w>=3 s`** (693/698), 5.7% below 1 s. At `T_w>=3 s` even large deviations solve (98% at `\|e_xy\|>=0.4 m`, 94% at `>=0.8 m`, n=68). Justifies `T_W_MIN=3 s`. |
| `solve_time.png` | solve-time histogram, one core | Runtime policy (`T_w>=3 s`, smoke n=200): **median 5.1 ms, p90 8.5 ms**, 100% solved. Failures are the expensive case (1–6 s each, mesh exhaustion), so a runtime caller should bound the solve with `max_nodes` or a timeout. |
| `examples.png` | 6 solved re-joins on the floor-6 baked map, picked by increasing deviation: nominal plan (blue), the window segment, deviated start (red dot + heading), PMP re-join (red), landing point (green) | The re-join converges smoothly onto the plan with all 5 states pinned. **Caveat visible in panel 4**: cost B knows nothing about walls — the re-join swings wide toward the left wall around a corner. Wall clearance is in the label features (`s[:,7]`) but not in the cost; the supervisor's clearance check must still gate the re-join. |

Also note: in the sweep (scale up to 8x the TVLQR deviation) 147/2000 deviated
starts already have negative hull clearance (inside a wall); they are labelled
anyway since the effort cost ignores walls. The production dataset uses scale
[0.5, 3].
