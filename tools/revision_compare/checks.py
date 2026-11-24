#!/usr/bin/env python3
"""Frozen behavioral checks for historical and current planning kernels.

These checks call interfaces present before the replanning work. Synthetic BVP
results isolate failure handling; native FMM checks execute the compiled code.
Historical failures are observations, not reasons to rewrite the old checkout.
Run this module in a fresh process with the selected revision's Python build.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


PROTOCOL = "common-behavior-v1"


class UnsupportedCheck(Exception):
    """The selected historical API cannot perform this declared check."""


def _load_modules(repo):
    repo = Path(repo).resolve()
    package = repo / "src/agx_navigation/agx_planning"
    if not (package / "agx_planning").is_dir():
        raise ValueError("selected repository has no planning package")
    for name, module in tuple(sys.modules.items()):
        if name == "agx_planning" or name.startswith("agx_planning."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(package):
                raise ValueError("another revision is already imported; use a fresh Python process")
    sys.path.insert(0, str(package))
    modules = SimpleNamespace(
        np=importlib.import_module("numpy"),
        scipy=importlib.import_module("scipy"),
        skfmm=importlib.import_module("skfmm"),
        vf=importlib.import_module("agx_planning.vector_field.field"),
        grid=importlib.import_module("agx_planning.vector_field.grid"),
        planner=importlib.import_module("agx_planning.pmp_planner"),
        rollout=importlib.import_module("agx_planning.pmp_planner.rollout"),
        solver=importlib.import_module("agx_planning.pmp_planner.shooting_solver"),
    )
    for module in (modules.vf, modules.grid, modules.planner, modules.rollout, modules.solver):
        if not Path(module.__file__).resolve().is_relative_to(package):
            raise ValueError("planning import escaped the selected repository")
    return modules


def _cell_lookup(m):
    observed = [m.vf.world_to_grid(x, y, -1., -2., .25, 6, 5)
                for x, y in [(-1.01, -1.5), (-.5, -2.01), (-1., -2.), (.5, -1.5)]]
    expected = [None, None, (0, 0), None]
    return observed == expected, {"cells": observed}, {"cells": expected}


def _cell_samples(m):
    np = m.np
    values = np.arange(1., 31.).reshape(5, 6)
    grid = m.grid.VectorFieldGrid()
    grid.update(values, -1., -2., .25)
    # Interior centre, first outer half-cell band, last outer half-cell band.
    x = np.array([-1. + 3.5 * .25, -1. + .05, -1. + 6 * .25 - .05])
    y = np.array([-2. + 2.5 * .25, -2. + .05, -2. + 5 * .25 - .05])
    actual = grid.query_vec(x, y)[0]
    expected = np.array([values[2, 3], values[0, 0], values[-1, -1]])
    return bool(np.allclose(actual, expected, rtol=0, atol=1e-12)), \
        {"travel_time": actual.tolist()}, {"travel_time": expected.tolist(), "atol": 1e-12}


def _packed_parity(m):
    np = m.np
    # Quantize deliberately before both paths; comparison is not between
    # different precisions. Integer values/sentinel remain exactly float32.
    values = np.arange(1, 31, dtype=np.float32).reshape(5, 6).astype(np.float64)
    values[2, 3] = np.nan
    zero = np.zeros_like(values)
    result = m.vf.VectorFieldResult(values, zero.copy(), zero.copy(), zero.copy(),
                                   30., 0., 0., .25, np.ones_like(values))
    cfg = m.planner.PlannerConfig(fp_smooth_sigma=0.)
    direct = m.vf.field_result_to_grid(result, field_eps=cfg.field_eps, wall_sigma=0.)
    packed = m.rollout.parse_field_array(m.vf.pack_field_array(result), cfg)
    if packed is None:
        return False, {"packed_grid": None}, {"prepared_query_parity": True}
    rows, cols = np.indices(values.shape)
    x, y = (cols.ravel() + .5) * .25, (rows.ravel() + .5) * .25
    left, right = direct.query_vec(x, y), packed.query_vec(x, y)
    errors = [float(np.max(np.abs(a - b))) for a, b in zip(left, right)]
    return all(error <= 1e-12 for error in errors), \
        {"max_abs_error_per_query_channel": errors, "input_precision": "deliberately_quantized_float32"}, \
        {"prepared_query_parity": True, "atol": 1e-12}


def _problem(m, *, near_goal=False):
    np = m.np
    cfg = m.planner.PlannerConfig(N=8, T_horizon=1., control_rate=10., dt_segment=.2,
                                 max_rollout_sim_time=2., goal_tolerance_xy=.05)
    grid = m.grid.VectorFieldGrid()
    grid.update(np.tile(100. - np.arange(100), (20, 1)), 0., 0., .1)
    solver = m.planner.PMPShootingSolver(cfg, grid)
    start = np.array([1., 1., 0., 0., 0.])
    goal = np.array([1.1 if near_goal else 4., 1., 0.])
    return solver, start, goal


def _candidate(m, knots, guess):
    from scipy.interpolate import PPoly
    np = m.np
    x = np.array([0., knots[-1]])
    coefficients = np.zeros((2, 1, 10))
    coefficients[1, 0, :5] = guess[:5, 0]
    spline = PPoly(coefficients.transpose(2, 0, 1), x, axis=1)
    return SimpleNamespace(success=True, status=0, message="controlled BVP test double",
                           sol=spline, x=x, y=spline(x), niter=1, rms_residuals=np.zeros(1))


def _failed_solve(m, failure):
    solver, start, goal = _problem(m)

    def success(_ode, _bc, knots, guess, **_options):
        return _candidate(m, knots, guess)

    def fail(_ode, _bc, knots, guess, **_options):
        if failure == "exception":
            raise RuntimeError("controlled failure after prior success")
        result = _candidate(m, knots, guess)
        result.success, result.status = False, 1
        return result

    with patch.object(m.solver, "solve_bvp", success):
        first = solver.sample_committed_segment(start, goal, .1, 2)
    if first is None:
        raise RuntimeError("controlled successful BVP was rejected during check setup")
    moved = start.copy()
    moved[0] += .1
    with patch.object(m.solver, "solve_bvp", fail):
        second = solver.sample_committed_segment(moved, goal, .1, 2)
    observed = {"first_segment_present": True, "failed_segment_present": second is not None}
    if second is not None:
        observed["returned_start_x"] = float(second[2][0, 0])
        observed["requested_start_x"] = float(moved[0])
    outcome = getattr(solver, "last_outcome", None)
    observed["structured_status"] = getattr(outcome, "status", None)
    return second is None, observed, {"failed_segment_present": False}


def _collect(generator):
    count = 0
    while True:
        try:
            next(generator)
            count += 1
        except StopIteration as terminal:
            return count, terminal.value


def _stagnation(m):
    solver, start, goal = _problem(m, near_goal=True)

    def stationary(_ode, _bc, knots, guess, **_options):
        return _candidate(m, knots, guess)

    with patch.object(m.solver, "solve_bvp", stationary):
        chunks, terminal = _collect(m.rollout.rollout_generator(solver, solver.cfg, start, goal))
    observed = {"terminal_status": terminal.status, "chunks": chunks,
                "pose_tolerance_met": bool(m.rollout.goal_reached(start, goal, solver.cfg))}
    return terminal.status == "stagnated" and not observed["pose_tolerance_met"], \
        observed, {"terminal_status": "stagnated", "pose_tolerance_met": False}


def _cancel_during_solve(m):
    solver, start, goal = _problem(m)
    cancelled = False

    def successful_but_late(_ode, _bc, knots, guess, **_options):
        nonlocal cancelled
        cancelled = True
        return _candidate(m, knots, guess)

    with patch.object(m.solver, "solve_bvp", successful_but_late):
        chunks, terminal = _collect(m.rollout.rollout_generator(
            solver, solver.cfg, start, goal, lambda: "cancelled" if cancelled else None))
    observed = {"chunks_after_cancel": chunks, "terminal_status": terminal.status}
    expected = {"chunks_after_cancel": 0, "terminal_status": "cancelled"}
    return observed == expected, observed, expected


def _early_exit(m, *, initialized):
    np = m.np
    phi = np.ones((41, 41))
    phi[20, 20] = -1.
    speed = np.ones_like(phi)
    full = m.skfmm.travel_time(phi, speed, dx=.2, order=2)
    targets = [(20, 20), (20, 21), (19, 20)] if initialized else [(20, 30), (20, 10), (30, 20), (10, 20)]
    counts, reached_targets, parity = [], [], []
    for target in targets:
        try:
            partial = m.skfmm.travel_time(phi, speed, dx=.2, order=2,
                                           early_exit_target=target, early_exit_margin=0)
        except TypeError as error:
            raise UnsupportedCheck(str(error)) from error
        reached = ~np.ma.getmaskarray(partial)
        counts.append(int(reached.sum()))
        reached_targets.append(bool(reached[target]))
        parity.append(bool(np.array_equal(np.asarray(partial)[reached], np.asarray(full)[reached])))
    correct = (all(reached_targets) and all(parity) and all(count < phi.size for count in counts)
               and (all(5 <= count < 100 for count in counts) if initialized else len(set(counts)) == 1))
    observed = {"targets": targets, "reached_counts": counts, "target_reached": reached_targets,
                "exact_full_parity_on_reached": parity, "total_cells": int(phi.size)}
    expected = {"target_reached": True, "exact_full_parity_on_reached": True,
                "stops_before_full_domain": True,
                "count_rule": "5 <= count < 100" if initialized else "equal for symmetric targets"}
    return bool(correct), observed, expected


def _record(name, kind, callback):
    start = time.perf_counter()
    try:
        passed, actual, expected = callback()
        record = {"name": name, "kind": kind, "status": "pass" if passed else "fail",
                  "expected": expected, "actual": actual}
    except UnsupportedCheck as error:
        record = {"name": name, "kind": kind, "status": "unsupported", "reason": str(error)}
    except Exception as error:
        record = {"name": name, "kind": kind, "status": "error",
                  "error": f"{type(error).__name__}: {error}"}
    record["check_wall_s"] = time.perf_counter() - start
    return record


def run_checks(repo):
    repo = Path(repo).resolve()
    m = _load_modules(repo)
    checks = [
        ("floor_cell_lookup", "unit_geometry", lambda: _cell_lookup(m)),
        ("cell_centres_and_outer_half_cells", "unit_geometry", lambda: _cell_samples(m)),
        ("direct_packed_preparation_parity", "unit_precision_controlled", lambda: _packed_parity(m)),
        ("failed_bvp_cannot_commit_previous_solution", "unit_mock_bvp", lambda: _failed_solve(m, "status")),
        ("bvp_exception_cannot_commit_previous_solution", "unit_mock_bvp", lambda: _failed_solve(m, "exception")),
        ("stagnation_is_not_strict_success", "unit_mock_bvp", lambda: _stagnation(m)),
        ("cancellation_during_solve_commits_no_chunk", "unit_mock_bvp", lambda: _cancel_during_solve(m)),
        ("early_exit_symmetric_tied_targets", "compiled_native", lambda: _early_exit(m, initialized=False)),
        ("early_exit_initialized_targets", "compiled_native", lambda: _early_exit(m, initialized=True)),
    ]
    records = [_record(name, kind, callback) for name, kind, callback in checks]
    counts = {status: sum(record["status"] == status for record in records)
              for status in ("pass", "fail", "error", "unsupported")}
    native = importlib.import_module("skfmm.cfmm")
    module_paths = {module.__name__: str(Path(module.__file__).resolve())
                    for module in (m.vf, m.grid, m.planner, m.rollout, m.solver, m.skfmm, native)}
    package = repo / "src/agx_navigation/agx_planning/agx_planning"
    capabilities = {
        "structured_solve_outcome": hasattr(m.planner.PMPShootingSolver, "solve_outcome"),
        "incremental_field": (package / "vector_field/incremental.py").is_file(),
        "full_plan_store": (package / "pmp_planner/plan_store.py").is_file(),
        "full_stop_execution": (package / "replanning/execution.py").is_file(),
        "replanning_coordinator": (package / "replanning/coordinator.py").is_file(),
        "native_checkpoint_replay": hasattr(m.skfmm, "TravelTimeState"),
    }
    return {"schema_version": 1, "protocol": PROTOCOL, "repo": str(repo),
            "revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python_executable": sys.executable, "module_paths": module_paths,
            "native_binary_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "capabilities": capabilities, "counts": counts, "checks": records,
            "scope": "common behavioral checks; mock BVP checks are not physical execution results"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True, help="new output directory")
    args = parser.parse_args(argv)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = run_checks(args.repo)
    except Exception as error:
        report = {"schema_version": 1, "protocol": PROTOCOL, "status": "setup_error",
                  "error": f"{type(error).__name__}: {error}"}
        code = 2
    else:
        code = int(report["counts"]["fail"] + report["counts"]["error"] > 0)
    (output / "checks.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"protocol": PROTOCOL, "counts": report.get("counts"),
                      "status": report.get("status", "recorded"), "output": str(output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
