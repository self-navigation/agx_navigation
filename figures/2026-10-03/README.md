# 2026-10-03 — job 160 departures (#34). PRELIMINARY, not for publishing

Render: `.venv/bin/python figures/2026-10-03/render.py`. Data: `run_data/job160/`
(gitignored; `series/` = `tools/departure_series.py` over the VM's
`~/compare_factors/<cfg>/s*/track_*.npz`, `rows/` = the job's row files).
Configs A-D of job 160 (E is open loop, no TVLQR diagnostics).

The corrector's believed pose is rebuilt from its diagnostics:
believed = live_plan[k] + R(θ_ref)·(e_along, e_cross), k = (t − t_first_diag)/0.1 s.
The diag `index` field is −1 throughout, so k is rebuilt from time. Under
`truth` this reproduces truth to mm–cm, which is the self-check.

- `event_aligned.png` — median/IQR aligned at departure (first true |e_cross| > 0.2 m).
  Under amcl the pose error climbs 0.1 → 0.3 m in the ~3 s BEFORE departure,
  and the BELIEVED error stays flat (~0.05 m) through it. The corrector does not
  see the departure. Under truth the believed error tracks the true one, and
  both are corrected within ~3 s. The 0.2 m threshold is crossed by nearly every
  run (even D, 91% arrival), so it marks "first wobble", not failure.
- `believed_vs_true.png` — per-run max believed vs true |e_cross|. Truth configs
  sit on the diagonal. amcl configs sit below it (median ratio 0.65 / 0.68).
- `loc_err.png` — amcl per-tick error CDF and max error vs final_err (A vs C).
- `case_<plan>_s<seed>_<cfg>.png` — map + time series for the 3 worst A misses
  and the same plan in B/C/D. The 00350 s0 A case: the body pins on a door
  corner (clearance < 0 from t = 13.6 s). amcl's pose keeps "driving the plan"
  (odometry says the wheels turn) and drifts 8 m from truth. The corrector
  believes it is on track the whole time.
