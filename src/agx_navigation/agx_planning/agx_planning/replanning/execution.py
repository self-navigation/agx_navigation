"""R1 full-stop reference experiment, using production numerical kernels.

The robot executes between observations. On a reveal it brakes, computes a
complete replacement while stopped, and resumes from its actual model state.
Measured computation advances the execution clock only after stopping. This
reference does not claim moving-worker overlap or physical actuator fidelity.
"""

from dataclasses import asdict, dataclass, replace
import math
import time

import numpy as np

from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
from agx_planning.pmp_planner.plan_store import PlanRecord
from agx_planning.pmp_planner.rollout import rollout_generator
from agx_planning.vector_field.contracts import FieldKey, FieldSnapshot
from agx_planning.vector_field.field import VectorFieldConfig, compute_field
from .braking import BrakeConfig, reserve_brake
from .replay import replay_plan
from .validity import Footprint, GridCollisionChecker


@dataclass(frozen=True)
class PlanAttempt:
    status: str
    message: str
    plan: PlanRecord | None = None
    field_s: float = 0.0
    solve_s: float = 0.0


class ProductionPlanner:
    """Build a new complete field and full BVP rollout from revealed data only."""

    def __init__(self, profile, numerical_version, session_id="r1"):
        self.cfg = PlannerConfig(**profile["planner"])
        self.field_cfg = VectorFieldConfig(**profile["field"])
        if self.field_cfg.early_exit_enable:
            raise ValueError("R1 reference requires complete fields")
        if profile["field_path"] != "raw_float64":
            raise ValueError("identified R1 plans require the raw_float64 profile")
        self.recipe = dict(profile["field"])
        self.policy = (profile["occupancy_threshold"], profile["allow_unknown"])
        self.numerical_version = numerical_version
        self.session_id = session_id
        self.count = 0

    def __call__(self, snapshot, state, goal, start_epoch, parent_id=None):
        begin = time.perf_counter()
        field_s = 0.0
        self.count += 1
        try:
            if self.policy != (snapshot.policy.threshold, snapshot.policy.allow_unknown):
                raise ValueError("map occupancy policy differs from the experiment profile")
            start_cell = snapshot.geometry.world_to_cell(*state[:2])
            goal_cell = snapshot.geometry.world_to_cell(*goal[:2])
            if start_cell is None or goal_cell is None:
                return PlanAttempt("invalid_state", "start or goal is outside the map")
            sc, sr = start_cell
            gc, gr = goal_cell
            if snapshot.obstacles[sr, sc] or snapshot.obstacles[gr, gc]:
                return PlanAttempt("invalid_state", "start or goal cell is occupied")
            result = compute_field(snapshot.occupancy, gc, gr, snapshot.geometry.resolution,
                                   *snapshot.geometry.origin[:2], self.field_cfg,
                                   sc, sr, *self.policy)
            field_s = time.perf_counter() - begin
            if result is None or not np.isfinite(result[0].travel_time[sr, sc]):
                return PlanAttempt("unreachable", "start is disconnected from the goal", field_s=field_s)
            key = FieldKey.for_map(snapshot, goal_cell, self.recipe, self.numerical_version)
            field = FieldSnapshot.from_result(key, result[0], snapshot.obstacles)
            solver = PMPShootingSolver(self.cfg, field.build_grid(self.cfg))
            generator = rollout_generator(solver, self.cfg, np.array(state, float), np.array(goal, float))
            chunks = []
            while True:
                try:
                    chunks.append(next(generator))
                except StopIteration as terminal:
                    outcome = terminal.value
                    break
            solve_s = time.perf_counter() - begin - field_s
            if outcome.status not in ("success", "stagnated"):
                return PlanAttempt(outcome.solve_status or outcome.status, outcome.message,
                                   field_s=field_s, solve_s=solve_s)
            plan = PlanRecord.from_chunks(
                chunks, initial_state=state, session_id=self.session_id,
                plan_id=f"plan-{self.count:04d}", parent_id=parent_id, field_key=key,
                start_epoch=start_epoch, status=outcome.status,
                metadata={"solver_message": outcome.message, "field_precision": "float64"},
            )
            return PlanAttempt(outcome.status, outcome.message, plan, field_s, solve_s)
        except (ValueError, FloatingPointError) as error:
            return PlanAttempt("invalid_input", str(error), field_s=field_s,
                               solve_s=max(0.0, time.perf_counter() - begin - field_s))


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "command"
    sample_period: float = 0.1
    collision_step: float = 0.02
    goal_xy: float = 0.05
    goal_heading: float = 0.2
    goal_wheel_speed: float = 0.5
    max_execution_s: float = 120.0
    max_replans: int = 8
    latency_override_s: float | None = None

    def __post_init__(self):
        if self.mode not in ("command", "nominal"):
            raise ValueError("mode must be command or nominal")
        for name in ("sample_period", "collision_step", "goal_xy", "goal_heading",
                     "goal_wheel_speed", "max_execution_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_replans) is not int or self.max_replans < 0:
            raise ValueError("max_replans must be a nonnegative integer")
        if (self.latency_override_s is not None
                and (not math.isfinite(self.latency_override_s) or self.latency_override_s < 0)):
            raise ValueError("latency override must be finite and nonnegative")


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    message: str
    times: np.ndarray
    states: np.ndarray
    events: tuple
    plans: tuple
    metadata: dict


def checker_for(snapshot, footprint):
    return GridCollisionChecker(snapshot.obstacles, snapshot.geometry.resolution,
                                snapshot.geometry.origin[:2], map_revision=snapshot.identity,
                                footprint=footprint)


class FullStopRunner:
    """One deterministic observation schedule, with measured or injected delays.

    The planner callback receives a revealed snapshot, a copied measured state,
    goal, start epoch, and parent ID. The true map is used only by the reveal
    model and collision evaluator. Controlled reveals are observed at the next
    observation boundary; event records retain their requested times.
    """

    def __init__(self, revealer, planner, *, c_v, c_w, brake: BrakeConfig,
                 config=ExecutionConfig(), footprint=Footprint()):
        if not all(math.isfinite(v) and v > 0 for v in (c_v, c_w)):
            raise ValueError("kinematic constants must be finite and positive")
        self.revealer, self.planner = revealer, planner
        self.c_v, self.c_w = c_v, c_w
        self.brake, self.cfg, self.footprint = brake, config, footprint

    def run(self, initial_state, goal, *, scheduled_reveals=()):
        state, goal = np.array(initial_state, float), np.array(goal, float)
        if (state.shape != (5,) or goal.shape != (3,) or not np.isfinite(state).all()
                or not np.isfinite(goal).all()):
            raise ValueError("experiment needs a finite five-state start and three-value goal")
        schedule = sorted((float(t), tuple(ids)) for t, ids in scheduled_reveals)
        if any(not math.isfinite(t) or t < 0 for t, _ in schedule):
            raise ValueError("scheduled observation times must be finite and nonnegative")
        now, times, states, events, plans = 0.0, [0.0], [state.copy()], [], []
        next_scan, schedule_index, replans = 0.0, 0, 0
        true_checker = checker_for(self.revealer.true_map, self.footprint)
        status, message = "failed", "execution did not complete"
        setup_s, stopped_compute_s, brake_s = 0.0, 0.0, 0.0

        def event(kind, **details):
            events.append({"time": float(now), "kind": kind,
                           "map_revision": self.revealer.current.revision, **details})

        def observation_times(duration):
            first = math.floor((now + 1e-10) / self.cfg.sample_period) + 1
            last = math.floor((now + duration) / self.cfg.sample_period)
            ticks = np.arange(first, last + 1) * self.cfg.sample_period - now
            return ticks[(ticks > 0) & (ticks < duration)]

        def observe(force=False):
            nonlocal next_scan, schedule_index
            before = self.revealer.current.identity
            if not force and now + 1e-10 < next_scan:
                return False
            while schedule_index < len(schedule) and schedule[schedule_index][0] <= now + 1e-10:
                requested, ids = schedule[schedule_index]
                _, added = self.revealer.reveal_ids(ids)
                if added:
                    event("map_reveal", object_ids=list(added), source="schedule", requested_time=requested)
                schedule_index += 1
            _, added = self.revealer.scan(state[:3])
            if added:
                event("map_reveal", object_ids=list(added), source="visibility")
            next_scan = (math.floor((now + 1e-10) / self.cfg.sample_period) + 1) * self.cfg.sample_period
            return before != self.revealer.current.identity

        def append(t, value):
            nonlocal now, state
            if t <= now:
                raise ValueError("execution time must advance")
            now, state = float(t), np.array(value, float)
            times.append(now)
            states.append(state.copy())

        def hold(duration):
            # Only stopped compute is modeled. Observe scheduled changes while
            # waiting so a completed result cannot hide intervening revisions.
            if np.max(np.abs(state[3:])) > 1e-12:
                raise ValueError("compute wait requires a stopped state")
            end = now + duration
            while now < end - 1e-12:
                observe()
                append(min(end, next_scan), state)
            observe()

        def move(trajectory, *, brake_motion=False):
            nonlocal status, message
            start = now
            for i in range(len(trajectory.times) - 1):
                if now >= self.cfg.max_execution_s:
                    status, message = "time_limit", "execution time limit reached"
                    return "failed"
                if not brake_motion and observe():
                    return "reveal"
                poses = trajectory.states[i:i + 2, :3]
                validity = true_checker.check_path(
                    poses, trajectory.times[i:i + 2],
                    interval_motion_bounds=trajectory.motion_bounds[i:i + 1],
                    pose_error_bounds=(trajectory.position_error_bounds[i:i + 2]
                                       if brake_motion else None),
                )
                if not validity.valid:
                    status, message = "collision", "true-map swept-footprint check failed"
                    event("collision", reason=validity.reason, interval_start=float(now))
                    return "failed"
                append(start + trajectory.times[i + 1], trajectory.states[i + 1])
                if brake_motion:
                    observe()
            return "complete"

        def stop(reason):
            nonlocal status, message, brake_s
            reserve = reserve_brake(checker_for(self.revealer.current, self.footprint),
                                    state, self.c_v, self.c_w, self.brake,
                                    sample_times=observation_times(
                                        self.brake.delay_s + np.max(np.abs(state[3:])) / self.brake.deceleration))
            event("brake_reservation", reason=reason, status=reserve.status,
                  duration=(float(reserve.trajectory.times[-1]) if reserve.trajectory is not None else None))
            if not reserve.feasible:
                status, message = reserve.status, reserve.message
                return False
            start = now
            if move(reserve.trajectory, brake_motion=True) == "failed":
                return False
            brake_s += now - start
            event("stopped", reason=reason)
            return True

        observe(force=True)
        known = checker_for(self.revealer.current, self.footprint)
        if not known.check_pose(state[:3]).valid or not true_checker.check_pose(state[:3]).valid:
            status, message = "invalid_start", "initial filled footprint is invalid"
        elif not known.check_pose(goal).valid:
            status, message = "invalid_goal", "goal filled footprint is invalid"
        elif np.max(np.abs(state[3:])) > 1e-12:
            status, message = "invalid_start", "R1 starts from a stopped state"
        else:
            initial, parent_id = True, None
            while True:
                if replans > self.cfg.max_replans:
                    status, message = "replan_limit", "declared replacement budget exhausted"
                    break
                snapshot = self.revealer.current
                started = time.perf_counter()
                try:
                    attempt = self.planner(snapshot, state.copy(), goal.copy(), now, parent_id)
                except Exception as error:
                    attempt = PlanAttempt("planner_exception", f"{type(error).__name__}: {error}")
                elapsed = time.perf_counter() - started
                event("plan_result", status=attempt.status, message=attempt.message,
                      field_s=attempt.field_s, solve_s=attempt.solve_s,
                      observed_compute_s=elapsed, initial=initial)
                plan, replay = attempt.plan, None
                if plan is not None and (attempt.status not in ("success", "stagnated")
                                         or plan.status != attempt.status):
                    plan = None
                    attempt = PlanAttempt("invalid_plan", "failed or inconsistent outcome cannot supply a plan")
                if plan is not None:
                    key = plan.field_key
                    if (key.map_session != snapshot.session or key.map_revision != snapshot.revision
                            or key.geometry != snapshot.geometry
                            or key.occupancy_policy_id != snapshot.policy.identity
                            or key.goal_cell != snapshot.geometry.world_to_cell(*goal[:2])
                            or not np.allclose(plan.raw_states[0], state, rtol=0, atol=1e-6)):
                        plan = None
                        attempt = PlanAttempt("invalid_plan", "candidate identity or initial pin differs")
                    else:
                        checked = time.perf_counter()
                        try:
                            replay = replay_plan(plan, self.c_v, self.c_w, mode=self.cfg.mode,
                                                 max_step=self.cfg.collision_step, initial_state=state,
                                                 footprint=self.footprint)
                            validity = checker_for(snapshot, self.footprint).check_path(
                                replay.states[:, :3], replay.times, interval_motion_bounds=replay.motion_bounds)
                            event("plan_validation", valid=validity.valid, reason=validity.reason)
                            if not validity.valid:
                                plan = None
                                attempt = PlanAttempt("invalid_plan", "candidate fails swept-footprint validation")
                        except (ValueError, FloatingPointError) as error:
                            plan = None
                            attempt = PlanAttempt("invalid_plan", str(error))
                        elapsed += time.perf_counter() - checked
                if initial:
                    setup_s += elapsed
                else:
                    delay = elapsed if self.cfg.latency_override_s is None else self.cfg.latency_override_s
                    stopped_compute_s += delay
                    hold(delay)
                    event("compute_wait", duration=delay)
                if snapshot.identity != self.revealer.current.identity:
                    event("candidate_rejected", reason="stale_map")
                    initial = False
                    replans += 1
                    continue
                if now > self.cfg.max_execution_s:
                    status, message = "time_limit", "compute exceeded the execution time limit"
                    break
                if plan is None:
                    status, message = attempt.status, attempt.message
                    break
                plan = replace(plan, start_epoch=now)
                plans.append(plan)
                parent_id = plan.plan_id
                event("plan_activated", plan_id=plan.plan_id, solver_status=plan.status)
                # Add exact observation ticks to the simulation grid. This is
                # simulator bookkeeping, like true-map scoring, not planner
                # computation. The previously checked path is unchanged.
                replay = replay_plan(plan, self.c_v, self.c_w, mode=self.cfg.mode,
                                     max_step=self.cfg.collision_step, initial_state=state,
                                     footprint=self.footprint,
                                     sample_times=observation_times(plan.times[-1]))
                result = move(replay)
                if result == "failed":
                    break
                changed = result == "reveal" or observe(force=True)
                if changed:
                    if not stop("map_change"):
                        break
                    replans += 1
                    initial = False
                    continue
                if not stop("plan_end"):
                    break
                xy = float(np.linalg.norm(state[:2] - goal[:2]))
                heading = abs((state[2] - goal[2] + math.pi) % (2 * math.pi) - math.pi)
                wheels = float(np.max(np.abs(state[3:])))
                success = (xy <= self.cfg.goal_xy and heading <= self.cfg.goal_heading
                           and wheels <= self.cfg.goal_wheel_speed
                           and true_checker.check_pose(state[:3]).valid)
                status = "success" if success else "goal_miss"
                message = "common final pose, heading, speed and collision checks passed" if success else "common final-state tolerances were not met"
                event("final_check", success=success, position_error=xy,
                      heading_error=heading, max_wheel_speed=wheels, solver_status=plan.status)
                break
        event("finished", status=status, message=message)
        metadata = {"arm": "R1_full_stop", "execution_model": self.cfg.mode,
                    "latency_mode": ("measured_stop_replay" if self.cfg.latency_override_s is None else "injected_stop_replay"),
                    "setup_s": setup_s, "stopped_compute_s": stopped_compute_s,
                    "brake_s": brake_s, "execution_s": now, "replans": replans,
                    "config": asdict(self.cfg), "brake": asdict(self.brake),
                    "footprint": asdict(self.footprint)}
        return ExecutionResult(status, message, np.array(times), np.array(states),
                               tuple(events), tuple(plans), metadata)
