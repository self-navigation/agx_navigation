#!/usr/bin/env python3
"""Replay one plan's offline PMP rollout without ROS and time every BVP solve.

Same field construction as tools/rejoin_phase0.py (baked map -> FM2 field for
the plan's start/goal) and the same `rollout_generator` the planner node runs,
so the solve sequence is the one the stack would produce from that start.
Per solve it logs wall ms, mesh nodes and Newton iterations -- solve_bvp's own
progress signal (verbose=2 prints per-iteration residuals; --verbose passes it
through).

    .venv/bin/python tools/bench_pmp_rollout.py traj_data_v2/floor_6_v2_00369.npz
    .venv/bin/python tools/bench_pmp_rollout.py PLAN --jac off     # finite differences
    .venv/bin/python tools/bench_pmp_rollout.py PLAN --check-jac   # analytic vs FD

Writes one JSON line per solve to --out (default: stdout summary only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from rejoin_phase0 import _imports  # noqa: E402  (sets sys.path too)


STACK_PLANNER_PARAMS = dict(
    T_horizon=2.5, N=40, bvp_max_nodes=3000, dt_segment=0.5, control_rate=10.0,
    v_max=0.448, omega_max=1.049, w_h=10.0, w_v=3.0, w_brake=200.0, L_brake=1.50,
    align_gate_power=15.0, w_v_barrier=200.0, w_v_terminal=15.0,
    pursuit_lookahead_mult=0.6, wheel_radius=0.08, track=0.416503,
    slip_chi=1.3736, a_wheel_max=17.6, gamma_wheel=0.0016, w_wheel_max=20.0,
    w_wheel_barrier=50.0, wheel_cmd_max=20.0, tau_wheel=0.0,
    cmd_deadzone_v=0.03, cmd_deadzone_omega=0.05,
)


def build(plan_path: str, jac: str, stack_field: bool = False):
    PlannerConfig, PMPShootingSolver, _build_field, load_occupancy_grid = _imports()
    z = np.load(plan_path)
    my = str(z["map_yaml"])
    if not os.path.isabs(my):
        my = os.path.join(REPO, my)
    data, width, height, meta = load_occupancy_grid(my)
    cfg = PlannerConfig(mode="offline")
    if stack_field:
        # agx_bringup/launch/planner.launch.py's overrides -- the stack does
        # NOT run the dataclass defaults (align_gate_power 15 vs 4, ...).
        for k, v in STACK_PLANNER_PARAMS.items():
            setattr(cfg, k, v)
    if hasattr(cfg, "bvp_analytic_jac"):
        cfg.bvp_analytic_jac = (jac == "on")
    if stack_field:
        # The stack's vector_field params (vec_pmp.launch.py), not the
        # VectorFieldConfig defaults rejoin_phase0 uses.
        from agx_planning.vector_field import (VectorFieldConfig, VectorFieldGrid,
                                               compute_field, world_to_grid)
        arr = np.asarray(data, dtype=np.int8).reshape(height, width)
        o, r = meta["origin"], meta["resolution"]
        gc = world_to_grid(*z["goal_xy"], o[0], o[1], r, width, height)
        sc = world_to_grid(*z["start_xy"], o[0], o[1], r, width, height)
        out = compute_field(arr, gc[0], gc[1], r, o[0], o[1],
                            VectorFieldConfig(speed_profile="exponential",
                                              inflation_radius=0.5),
                            sc[0], sc[1], 65, allow_unknown=True)
        field = VectorFieldGrid()
        field.update(out[0].travel_time, o[0], o[1], r, field_eps=cfg.field_eps)
    else:
        field = _build_field(data, width, height, meta["resolution"], meta["origin"],
                             z["start_xy"], z["goal_xy"], 65, cfg)
    solver = PMPShootingSolver(cfg, field)
    x0 = np.array([z["poses"][0, 0], z["poses"][0, 1], z["poses"][0, 2], 0.0, 0.0])
    goal = np.array([z["goal_xy"][0], z["goal_xy"][1], z["poses"][-1, 2]])
    return cfg, solver, x0, goal


def check_jac(solver, x0, goal) -> None:
    """Compare the analytic Jacobians against central differences on the first
    solve's mesh and initial guess."""
    solver.solve(x0, goal)  # sets pursuit targets etc.
    sol = solver._prev_sol
    t = np.linspace(0.0, solver.cfg.T_horizon, 41)
    y = sol(t)
    rng = np.random.default_rng(0)
    y = y + 1e-2 * rng.standard_normal(y.shape)
    J = solver._ode_jac(t, y)
    h = 1e-6
    Jfd = np.zeros_like(J)
    for i in range(10):
        e = np.zeros((10, 1)); e[i] = h
        Jfd[:, i, :] = (solver._ode(t, y + e) - solver._ode(t, y - e)) / (2 * h)
    err = np.abs(J - Jfd)
    scale = np.maximum(np.abs(Jfd), 1.0)
    print(f"ode jac: max abs err {err.max():.3e}, max rel {np.max(err / scale):.3e}")
    worst = np.unravel_index(np.argmax(err / scale), err.shape)
    print(f"  worst entry d f[{worst[0]}] / d y[{worst[1]}] at node {worst[2]}: "
          f"analytic {J[worst]:.6g} fd {Jfd[worst]:.6g}")
    ya, yb = y[:, 0], y[:, -1]
    Ja, Jb = solver._bc_jac(ya, yb)
    Jafd = np.zeros((10, 10)); Jbfd = np.zeros((10, 10))
    for i in range(10):
        e = np.zeros(10); e[i] = h
        Jafd[:, i] = (solver._bc(ya + e, yb) - solver._bc(ya - e, yb)) / (2 * h)
        Jbfd[:, i] = (solver._bc(ya, yb + e) - solver._bc(ya, yb - e)) / (2 * h)
    print(f"bc jac:  max abs err {max(np.abs(Ja - Jafd).max(), np.abs(Jb - Jbfd).max()):.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--jac", choices=("on", "off"), default="on")
    ap.add_argument("--check-jac", action="store_true")
    ap.add_argument("--stack-field", action="store_true",
                    help="use the stack's planner + vector_field params (launch overrides)")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg, solver, x0, goal = build(a.plan, a.jac, a.stack_field)
    if a.check_jac:
        check_jac(solver, x0, goal)
        return 0

    from agx_planning.pmp_planner import rollout as ro
    import scipy.integrate as si

    # Wrap solve_bvp to record nodes / iterations of every call.
    stats = []
    orig = si.solve_bvp

    def wrapped(*args, **kw):
        t0 = time.perf_counter()
        sol = orig(*args, **kw)
        stats.append(dict(ms=round((time.perf_counter() - t0) * 1e3, 1),
                          nodes=int(sol.x.size), niter=int(getattr(sol, "niter", -1)),
                          ok=bool(sol.success), status=int(sol.status)))
        return sol

    import agx_planning.pmp_planner.shooting_solver as ss
    ss.solve_bvp = wrapped

    gen = ro.rollout_generator(solver, cfg, x0.copy(), goal)
    n = 0
    t0 = time.perf_counter()
    status = "?"
    try:
        while True:
            chunk = next(gen)
            n += 1
            if a.max_chunks and n >= a.max_chunks:
                status = "max-chunks"
                break
    except StopIteration as stop:
        status = getattr(stop.value, "status", "?")
    wall = time.perf_counter() - t0
    ms = np.array([s["ms"] for s in stats])
    print(f"{os.path.basename(a.plan)} jac={a.jac}: {n} chunks, {len(stats)} solves, "
          f"status={status}, total {wall:.1f} s")
    print(f"  solve ms: median {np.median(ms):.0f}  p90 {np.percentile(ms, 90):.0f}  "
          f"max {ms.max():.0f};  nodes median {np.median([s['nodes'] for s in stats]):.0f} "
          f"max {max(s['nodes'] for s in stats)};  niter median "
          f"{np.median([s['niter'] for s in stats]):.0f} max {max(s['niter'] for s in stats)};  "
          f"failed {sum(not s['ok'] for s in stats)}")
    last = getattr(solver, "_prev_sol", None)
    if a.out:
        with open(a.out, "w") as fh:
            for s in stats:
                fh.write(json.dumps(s) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
