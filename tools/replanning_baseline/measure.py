"""Timing and validity diagnostics around unchanged production kernels."""

from contextlib import ExitStack
from dataclasses import fields, is_dataclass
import time
from unittest.mock import patch

import numpy as np

from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
from agx_planning.pmp_planner import rollout, shooting_solver
from agx_planning.vector_field import field as vf

from .inputs import cell


def finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _equal_field_values(observed, reference):
    if observed is None or reference is None:
        return observed is reference
    if isinstance(observed, np.ndarray) or isinstance(reference, np.ndarray):
        return (isinstance(observed, np.ndarray) and isinstance(reference, np.ndarray)
                and np.array_equal(observed, reference, equal_nan=True))
    if is_dataclass(observed) or is_dataclass(reference):
        return (type(observed) is type(reference)
                and all(_equal_field_values(getattr(observed, item.name),
                                             getattr(reference, item.name))
                        for item in fields(observed)))
    if isinstance(observed, (list, tuple)) or isinstance(reference, (list, tuple)):
        return (type(observed) is type(reference) and len(observed) == len(reference)
                and all(_equal_field_values(a, b) for a, b in zip(observed, reference)))
    try:
        return bool(np.array_equal(observed, reference, equal_nan=True))
    except TypeError:
        return observed == reference


def verify_field(result, occupancy, metadata, profile):
    """Compare the measured raw field with a fresh solve outside the timers."""
    verification = {"status": "error", "differing_fields": []}
    start = time.perf_counter()
    try:
        geometry = metadata["geometry"]
        sc, gc = cell(metadata["start"], geometry), cell(metadata["goal"], geometry)
        outcome = vf.compute_field(
            occupancy, *gc, geometry["resolution"], *geometry["origin"][:2],
            vf.VectorFieldConfig(**profile["field"]), *sc,
            profile["occupancy_threshold"], profile["allow_unknown"],
        )
        reference = outcome[0] if outcome is not None else None
        differing = verification["differing_fields"]
        if result is None or reference is None:
            if result is not reference:
                differing.append("field_presence")
        else:
            for item in fields(vf.VectorFieldResult):
                if not _equal_field_values(getattr(result, item.name), getattr(reference, item.name)):
                    differing.append(item.name)
            if not np.array_equal(np.isfinite(result.travel_time), np.isfinite(reference.travel_time)):
                differing.append("reached_mask")
        verification.update(
            status="mismatch" if differing else "match",
            reached_cells=(int(np.isfinite(result.travel_time).sum()) if result is not None else None),
            reference_reached_cells=(int(np.isfinite(reference.travel_time).sum())
                                     if reference is not None else None),
        )
    except Exception as error:
        verification["exception"] = f"{type(error).__name__}: {error}"
    finally:
        verification["wall_s"] = time.perf_counter() - start
    return verification


def measure_field(occupancy, metadata, profile, updater=None):
    geometry = metadata["geometry"]
    sc, gc = cell(metadata["start"], geometry), cell(metadata["goal"], geometry)
    stages = dict.fromkeys(("edt_s", "speed_s", "fmm_s", "gradient_s"), 0.0)
    active = []

    def timer(name, function):
        def call(*args, **kwargs):
            if active:
                return function(*args, **kwargs)
            active.append(name)
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                stages[name] = stages.get(name, 0.0) + time.perf_counter() - start
                active.pop()
        return call

    start = time.perf_counter()
    with ExitStack() as stack:
        for name, label in (("distance_transform_edt", "edt_s"),
                            ("_update_interior_distance", "edt_s"),
                            ("build_speed_field", "speed_s"),
                            ("_update_speed_field", "speed_s"),
                            ("solve_eikonal", "fmm_s"), ("field_from_T", "gradient_s")):
            stack.enter_context(patch.object(vf, name, timer(label, getattr(vf, name))))
        compute = vf.compute_field if updater is None else updater.update
        result = compute(occupancy, *gc, geometry["resolution"], *geometry["origin"][:2],
                         vf.VectorFieldConfig(**profile["field"]), *sc,
                         profile["occupancy_threshold"], profile["allow_unknown"])
    stages["compute_field_s"] = time.perf_counter() - start
    stages["field_update"] = "full_recompute" if updater is None else updater.last_update
    if updater is not None and updater.last_fmm:
        stages["fmm_update"] = dict(updater.last_fmm)
    if updater is not None and updater.last_speed:
        stages["speed_update"] = dict(updater.last_speed)
    interior = getattr(updater, "last_interior", None)
    if interior:
        stages["interior_update"] = dict(interior)
    return (result[0] if result else None), stages


def measure_rollout(field, metadata, profile):
    cfg = PlannerConfig(**profile["planner"])
    solver = PMPShootingSolver(cfg, field)
    initial = np.asarray(metadata["start"], dtype=float)
    goal = np.asarray(metadata["goal"], dtype=float)
    solves, chunks, chunk_ready = [], [], []
    states = [initial[None, :]]
    segment = {}
    original_bvp = shooting_solver.solve_bvp
    original_sample = solver.sample_committed_segment
    pin_error = 0.0
    fault = None

    def observed_bvp(*args, **kwargs):
        start = time.perf_counter()
        record = {"index": len(solves), "success": False,
                  "warm_seed_available": bool(cfg.reuse_previous_solution and solver._prev_sol is not None),
                  "fun_jac_supplied": kwargs.get("fun_jac") is not None,
                  "bc_jac_supplied": kwargs.get("bc_jac") is not None}
        try:
            solution = original_bvp(*args, **kwargs)
            record.update(success=bool(solution.success), status=int(solution.status),
                          message=str(solution.message), nodes=int(solution.x.size),
                          iterations=int(solution.niter),
                          max_rms_residual=finite_number(np.max(solution.rms_residuals)))
            return solution
        except Exception as error:
            record["exception"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["wall_s"] = time.perf_counter() - start
            solves.append(record)

    def observed_sample(x0, target, dt, count):
        nonlocal fault, pin_error
        first_call = len(solves)
        result = original_sample(x0, target, dt, count)
        current = solves[first_call:]
        if not current or not all(s["success"] for s in current):
            fault = "bvp_failed"
            return None
        if result is None or solver._prev_sol is None:
            fault = "invalid_trajectory"
            return None
        commands, accels, poses, costates, endpoint = result
        raw = solver._prev_sol(np.arange(len(poses) + 1) * dt)[:5].T
        if not all(np.isfinite(a).all() for a in (raw, commands, accels, poses, costates, endpoint)):
            fault = "nonfinite_trajectory"
            return None
        error = max(float(np.max(np.abs(raw[0] - x0))),
                    float(np.max(np.abs(raw[-1] - endpoint))),
                    float(np.max(np.abs(raw[:-1, :3] - poses))))
        pin_error = max(pin_error, error)
        if error > profile["pin_tolerance"]:
            fault = "discontinuous_trajectory"
            return None
        segment["states"] = raw
        return result

    start = time.perf_counter()
    cpu = time.process_time()
    with patch.object(shooting_solver, "solve_bvp", observed_bvp), \
            patch.object(solver, "sample_committed_segment", observed_sample):
        generator = rollout.rollout_generator(solver, cfg, initial.copy(), goal)
        try:
            while True:
                try:
                    chunk = next(generator)
                except StopIteration as stop:
                    terminal = stop.value
                    break
                chunks.append(chunk)
                chunk_ready.append(time.perf_counter() - start)
                states.append(segment["states"][1:len(chunk.poses) + 1].copy())
        finally:
            generator.close()
    elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
    raw_states = np.concatenate(states)
    final = raw_states[-1]
    pose_met = rollout.goal_reached(final, goal, cfg)
    status = fault or ("completed" if terminal.status == "success" and pose_met else
                       "pose_not_reached" if terminal.status == "success" else "planner_failed")
    arrays = {"states": raw_states, "times": np.arange(len(raw_states)) / cfg.control_rate}
    for key, columns in (("wheel_cmds", 2), ("accels", 2), ("poses", 3), ("costates", 5)):
        arrays[key] = np.concatenate([getattr(c, key) for c in chunks]) if chunks else np.empty((0, columns))
    v, omega = cfg.wheels_to_body(final[3], final[4])
    return {"status": status, "planner_status": terminal.status, "planner_message": terminal.message,
            "planner_wall_s": elapsed, "planner_cpu_s": cpu_elapsed,
            "chunk_ready_s": chunk_ready,
            "solves": solves, "chunks": len(chunks), "commands": len(arrays["wheel_cmds"]),
            "nominal_duration_s": float(arrays["times"][-1]),
            "path_length_m": float(np.linalg.norm(np.diff(raw_states[:, :2], axis=0), axis=1).sum()),
            "pose_tolerance_met": bool(pose_met), "max_pin_error": pin_error,
            "goal_position_error_m": float(np.linalg.norm(final[:2] - goal[:2])),
            "goal_heading_error_rad": float(abs((final[2] - goal[2] + np.pi) % (2 * np.pi) - np.pi)),
            "terminal_body_speed_m_s": float(v), "terminal_yaw_rate_rad_s": float(omega),
            "max_sampled_accel_rad_s2": float(np.abs(arrays["accels"]).max()) if chunks else 0.0,
            "max_sampled_wheel_rad_s": float(np.abs(raw_states[:, 3:]).max()),
            "collision_checked": False, "execution_simulated": False}, arrays


def measure_snapshot(occupancy, metadata, profile, field_only=False, save_fields=False,
                     updater=None, verify_fields=False):
    """Measure a snapshot, then optionally verify its field outside all timers."""
    record = {"status": "exception", "field_only": field_only, "solves": []}
    arrays = {}
    result, field_computed = None, False
    start, cpu = time.perf_counter(), time.process_time()
    try:
        obstacles = occupancy >= profile["occupancy_threshold"]
        if not profile["allow_unknown"]:
            obstacles |= occupancy < 0
        sc, gc = cell(metadata["start"], metadata["geometry"]), cell(metadata["goal"], metadata["geometry"])
        if obstacles[sc[1], sc[0]] or obstacles[gc[1], gc[0]]:
            if updater is not None:
                updater.clear()
            record["status"] = "blocked_start" if obstacles[sc[1], sc[0]] else "blocked_goal"
            return record, arrays
        result, stages = measure_field(occupancy, metadata, profile, updater)
        field_computed = True
        record.update(stages)
        if result is None:
            record["status"] = "field_failed"
            return record, arrays
        reached = np.isfinite(result.travel_time)
        record.update(reached_cells=int(reached.sum()), free_cells=int((~obstacles).sum()),
                      travel_time_at_start=finite_number(result.travel_time[sc[1], sc[0]]))
        if save_fields:
            arrays.update(travel_time=result.travel_time, grad_x=result.grad_x, grad_y=result.grad_y,
                          wall_dist=result.wall_dist)
        if not reached[sc[1], sc[0]]:
            record["status"] = "start_unreached"
            return record, arrays
        conversion = time.perf_counter()
        packed = vf.pack_field_array(result)
        record["pack_s"] = time.perf_counter() - conversion
        conversion = time.perf_counter()
        grid = rollout.parse_field_array(packed, PlannerConfig(**profile["planner"]))
        record.update(parse_s=time.perf_counter() - conversion, packed_bytes=int(packed.nbytes))
        if grid is None or not grid.ready:
            record["status"] = "field_failed"
        elif field_only:
            record["status"] = "field_only"
        else:
            measured, trajectory = measure_rollout(grid, metadata, profile)
            record.update(measured)
            arrays.update(trajectory)
    except Exception as error:
        record.update(status="exception", exception=f"{type(error).__name__}: {error}")
    finally:
        record.update(total_wall_s=time.perf_counter() - start, total_cpu_s=time.process_time() - cpu)
        if verify_fields:
            record["verification"] = (
                verify_field(result, occupancy, metadata, profile)
                if field_computed and record["status"] != "exception" else
                {"status": "skipped", "reason": record["status"],
                 "differing_fields": [], "wall_s": 0.0}
            )
    return record, arrays
