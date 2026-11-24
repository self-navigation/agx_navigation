"""Wheel-space stopping trajectories and checked braking reserves without ROS."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .validity import Footprint, GridCollisionChecker, ValidityResult


@dataclass(frozen=True)
class BrakeConfig:
    """Declared stopping model, not measured hardware performance.

    deceleration is the achieved wheel ramp magnitude assumed by this model.
    Physical use needs a calibrated conservative value and delay/uncertainty
    allowances; a planner's maximum acceleration alone does not establish it.
    """
    deceleration: float
    delay_s: float = 0.0
    max_step_s: float = 0.02
    wheel_speed_limit: float = 20.0
    position_uncertainty: float = 0.0
    heading_uncertainty: float = 0.0
    max_duration_s: float = 60.0
    max_steps: int = 100_000

    def __post_init__(self):
        values = (self.deceleration, self.delay_s, self.max_step_s, self.wheel_speed_limit,
                  self.position_uncertainty, self.heading_uncertainty, self.max_duration_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("brake parameters must be finite")
        if min(self.deceleration, self.max_step_s, self.wheel_speed_limit, self.max_duration_s) <= 0:
            raise ValueError("brake rate, step, speed limit, and duration limit must be positive")
        if min(self.delay_s, self.position_uncertainty, self.heading_uncertainty) < 0:
            raise ValueError("brake delay and uncertainty must be nonnegative")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")


@dataclass(frozen=True)
class BrakeTrajectory:
    times: np.ndarray
    states: np.ndarray
    motion_bounds: np.ndarray
    position_error_bounds: np.ndarray
    brake_start_time: float

    def __post_init__(self):
        for name in ("times", "states", "motion_bounds", "position_error_bounds"):
            array = np.asarray(getattr(self, name), dtype=float)
            array = np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)
            object.__setattr__(self, name, array)
        n = self.times.size
        if self.times.shape != (n,) or n < 1 or self.states.shape != (n, 5):
            raise ValueError("brake times and states have incompatible shapes")
        if self.motion_bounds.shape != (n - 1,) or self.position_error_bounds.shape != (n,):
            raise ValueError("brake bounds have incompatible shapes")
        if not all(np.isfinite(array).all() for array in (self.times, self.states, self.motion_bounds,
                                                         self.position_error_bounds)):
            raise ValueError("brake arrays must be finite")
        if self.times[0] != 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("brake times must start at zero and increase")
        if np.any(self.motion_bounds < 0) or np.any(self.position_error_bounds < 0):
            raise ValueError("brake bounds must be nonnegative")
        if np.any(self.states[-1, 3:] != 0):
            raise ValueError("brake endpoint must have zero wheel speeds")
        if not math.isfinite(self.brake_start_time) or not 0 <= self.brake_start_time <= self.times[-1]:
            raise ValueError("brake start time must lie within the trajectory")


@dataclass(frozen=True)
class BrakeReservation:
    feasible: bool
    status: str
    message: str
    trajectory: BrakeTrajectory | None
    validity: ValidityResult | None


@dataclass(frozen=True)
class PrefixBrakeReservation:
    feasible: bool
    status: str
    anchor_index: int | None
    decision_time: float | None
    brake: BrakeReservation | None
    prefix_validity: ValidityResult | None


def build_brake(state, c_v, c_w, config: BrakeConfig, *, footprint=Footprint(), sample_times=()):
    """Hold current wheels during delay, then ramp each independently to zero.

    The delay represents the declared observation/scheduling/actuation model.
    Unmodeled physical deviations require the explicit uncertainty allowance.
    Stop times split the integration so wheel speed and yaw are exact on each
    piece. Position uses midpoint quadrature with an analytic error bound on
    each piece, accumulated into position_error_bounds. The bound is a numerical
    model bound; this function does not establish physical braking calibration.
    Optional sample_times are exact extra knots relative to brake time zero.
    Each must lie within the modeled delay and stopping duration.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (5,) or not np.isfinite(state).all():
        raise ValueError("brake state must contain five finite values")
    if not all(math.isfinite(v) and v > 0 for v in (c_v, c_w)):
        raise ValueError("c_v and c_w must be finite and positive")
    if np.max(np.abs(state[3:])) > config.wheel_speed_limit + 1e-12:
        raise ValueError("initial wheel speed exceeds the declared brake model limit")
    wheel_start = state[3:].copy()
    stop_times = config.delay_s + np.abs(wheel_start) / config.deceleration
    duration = float(stop_times.max())
    if not math.isfinite(duration) or duration > config.max_duration_s:
        raise ValueError("braking duration exceeds the configured model limit")
    sample_times = np.asarray(sample_times, dtype=float)
    if (sample_times.ndim != 1 or not np.isfinite(sample_times).all()
            or np.any(sample_times < 0) or np.any(sample_times > duration)):
        raise ValueError("sample_times must be finite one-dimensional values within the brake duration")
    knots = sorted(set([0.0, config.delay_s, *stop_times.tolist(), *sample_times.tolist()]))
    times = [0.0]
    for left, right in zip(knots[:-1], knots[1:]):
        count = max(1, math.ceil((right - left) / config.max_step_s))
        if len(times) + count > config.max_steps + 1:
            raise ValueError("braking trajectory exceeds the configured step limit")
        times.extend(np.linspace(left, right, count + 1)[1:].tolist())
    times = np.asarray(times)

    def wheels(t):
        elapsed = max(0.0, t - config.delay_s)
        result = np.sign(wheel_start) * np.maximum(0.0, np.abs(wheel_start) - config.deceleration * elapsed)
        return np.where(t >= stop_times, 0.0, result)

    states = np.empty((len(times), 5))
    states[0] = state
    errors = np.zeros(len(times))
    motion = np.zeros(max(0, len(times) - 1))
    radius = footprint.radius
    for i, (left, right) in enumerate(zip(times[:-1], times[1:])):
        dt = right - left
        wa, wb = wheels(left), wheels(right)
        va, vb = c_v * wa.sum(), c_v * wb.sum()
        oa, ob = c_w * (wa[1] - wa[0]), c_w * (wb[1] - wb[0])
        dv, do = (vb - va) / dt, (ob - oa) / dt
        theta = states[i, 2]
        theta_mid = theta + oa * dt / 2 + do * dt * dt / 8
        theta_next = theta + (oa + ob) * dt / 2
        displacement = (va + vb) * dt / 2
        if dv == 0 and do == 0:
            # Exact constant-wheel arc, evaluated without division by yaw rate.
            displacement *= float(np.sinc(oa * dt / (2 * math.pi)))
            local_error = 0.0
        else:
            max_v, max_o = max(abs(va), abs(vb)), max(abs(oa), abs(ob))
            second_derivative = 2 * abs(dv) * max_o + max_v * abs(do) + max_v * max_o**2
            local_error = dt**3 * second_derivative / 24
        states[i + 1, :2] = states[i, :2] + displacement * np.array([math.cos(theta_mid), math.sin(theta_mid)])
        states[i + 1, 2] = theta_next
        states[i + 1, 3:] = wb
        # Small floating arithmetic allowance in addition to the quadrature bound.
        rounding = 32 * np.finfo(float).eps * (1 + np.linalg.norm(states[i, :2]) + abs(displacement) + radius * abs(theta_next))
        errors[i + 1] = errors[i] + local_error + rounding
        motion[i] = dt * (max(abs(va), abs(vb)) + radius * max(abs(oa), abs(ob)))
    states[-1, 3:] = 0.0
    if not np.isfinite(states).all() or not np.isfinite(errors).all() or not np.isfinite(motion).all():
        raise ValueError("braking integration produced non-finite data")
    return BrakeTrajectory(times, states, motion, errors, config.delay_s)


def reserve_brake(checker: GridCollisionChecker, state, c_v, c_w, config: BrakeConfig, *, sample_times=()):
    """Return a reserve only when delay and the entire modeled brake are clear."""
    try:
        trajectory = build_brake(state, c_v, c_w, config, footprint=checker.footprint,
                                 sample_times=sample_times)
        # A heading error also rotates subsequent center motion. A fixed
        # footprint-only inflation would miss that growing position error.
        travelled_bound = np.r_[0.0, np.cumsum(trajectory.motion_bounds)]
        extra = (config.position_uncertainty
                 + (checker.footprint.radius + travelled_bound) * config.heading_uncertainty)
        validity = checker.check_path(
            trajectory.states[:, :3], trajectory.times,
            interval_motion_bounds=trajectory.motion_bounds,
            pose_error_bounds=trajectory.position_error_bounds + extra,
        )
    except ValueError as error:
        return BrakeReservation(False, "invalid_input", str(error), None, None)
    if not validity.valid:
        initial = checker.check_pose(trajectory.states[0, :3], extra_margin=float(extra[0]))
        status = "initial_state_invalid" if not initial.valid else "unsafe_brake"
        return BrakeReservation(False, status, validity.reason, trajectory, validity)
    status = "already_stopped" if np.all(np.asarray(state)[3:] == 0) else "reserved"
    return BrakeReservation(True, status, "modeled delay and stopping motion are clear", trajectory, validity)


def reserve_prefix_brake(checker: GridCollisionChecker, states, times, c_v, c_w,
                         config: BrakeConfig, *, interval_motion_bounds,
                         pose_error_bounds=None, candidate_indices=None):
    """Choose the latest certified brake anchor from declared sample boundaries.

    This is a discrete deadline, not the latest possible continuous time.
    The supplied prefix motion bounds must describe the actual chosen execution
    model. Position error bounds must cover each prefix sample. The configured
    initial heading uncertainty also inflates position uncertainty as the
    prefix moves. The model assumes no added wheel or heading error in time.
    A rejected pending plan does not invalidate this returned reserve. It may
    be used only for this map identity and unchanged prefix/state assumptions.
    """
    states = np.asarray(states, dtype=float)
    times = np.asarray(times, dtype=float)
    if states.ndim != 2 or states.shape[1] != 5 or len(states) < 1 or not np.isfinite(states).all():
        raise ValueError("prefix states must be a finite nonempty N by 5 array")
    if times.shape != (len(states),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("prefix times must be finite and strictly increasing")
    errors = np.zeros(len(states)) if pose_error_bounds is None else np.asarray(pose_error_bounds, dtype=float)
    if errors.shape != (len(states),) or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("prefix position error bounds must be finite and nonnegative")
    bounds = np.asarray(interval_motion_bounds, dtype=float)
    if bounds.shape != (len(states) - 1,) or not np.isfinite(bounds).all() or np.any(bounds < 0):
        raise ValueError("prefix motion bounds must cover each interval and be finite and nonnegative")
    candidates = list(range(len(states))) if candidate_indices is None else list(candidate_indices)
    if not candidates or any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
                             or not 0 <= i < len(states) for i in candidates):
        raise ValueError("candidate indices must select prefix sample boundaries")
    candidates = [int(i) for i in candidates]
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate indices must be unique")
    # Requiring the entire supplied prefix certificate first identifies the
    # first interval that cannot be used. Only earlier anchor boundaries qualify.
    travelled_bound = np.r_[0.0, np.cumsum(bounds)]
    anchor_errors = errors + config.position_uncertainty + travelled_bound * config.heading_uncertainty
    prefix_errors = anchor_errors + checker.footprint.radius * config.heading_uncertainty
    whole = checker.check_path(states[:, :3], times, interval_motion_bounds=bounds,
                               pose_error_bounds=prefix_errors)
    limit = len(states) - 1 if whole.valid else whole.first_unsafe_interval
    for index in sorted((i for i in candidates if i <= limit), reverse=True):
        reservation = reserve_brake(
            checker, states[index], c_v, c_w,
            replace(config, position_uncertainty=float(anchor_errors[index])),
        )
        if not reservation.feasible:
            continue
        prefix = checker.check_path(states[:index + 1, :3], times[:index + 1],
                                     interval_motion_bounds=bounds[:index],
                                     pose_error_bounds=prefix_errors[:index + 1])
        if prefix.valid:
            return PrefixBrakeReservation(True, "reserved", index, float(times[index]), reservation, prefix)
    return PrefixBrakeReservation(False, "no_safe_anchor", None, None, None, whole)
